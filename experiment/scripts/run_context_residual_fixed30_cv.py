#!/usr/bin/env python3
"""Five-fold fixed-30-epoch OOF training for the residual-hybrid rebuild.

This is deliberately separate from the frozen validation-selected v1 workflow.
Every local and context expert trains on its complete outer-training partition
for exactly 30 epochs. There is no inner validation, checkpoint selection, or
early stopping. The final-epoch outer-fold probabilities fit a new residual
stacker without accessing any independent or external test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.context import (  # noqa: E402
    LongContextSiteDataset,
    build_validated_contexts,
    load_sequence_cache,
    sequence_cache_sha256,
)
from ubipred.data import (  # noqa: E402
    SiteDataset,
    load_normalized_aaindex,
    make_loader,
    seed_everything,
)
from ubipred.engine import (  # noqa: E402
    load_checkpoint_model_state,
    predict,
    refit_model,
)
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.fixed30 import (  # noqa: E402
    FIXED_EPOCHS,
    PROTOCOL_VERSION,
    fixed30_training_arguments,
    validate_fixed30_protocol,
)
from ubipred.metrics import compute_metrics, write_json  # noqa: E402
from ubipred.model import build_model  # noqa: E402

from run_context_residual_cv import finish_oof_summary  # noqa: E402


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def indices_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def train_fixed_fold_expert(
    *,
    expert_name: str,
    expert_config: dict[str, object],
    dataset: torch.utils.data.Dataset,
    records: list,
    outer_train_indices: np.ndarray,
    outer_validation_indices: np.ndarray,
    aaindex_lookup: np.ndarray,
    window_size: int,
    fold: int,
    output_dir: Path,
    device: torch.device,
    num_workers: int,
    resume: bool,
    context_cache_hash: str,
    config_hash: str,
) -> tuple[np.ndarray, np.ndarray]:
    prediction_path = output_dir / "outer_predictions.npz"
    expected_indices = np.sort(outer_validation_indices)
    expected_cache_hash = context_cache_hash if expert_name == "context" else ""
    if prediction_path.exists():
        if not resume:
            raise FileExistsError(
                f"Existing predictions require --resume: {prediction_path}"
            )
        with np.load(prediction_path) as saved:
            saved_indices = saved["dataset_indices"].astype(np.int64)
            if not np.array_equal(saved_indices, expected_indices):
                raise RuntimeError(
                    f"Stale outer predictions for fold {fold} expert {expert_name}"
                )
            if int(saved["fixed_epochs"].item()) != FIXED_EPOCHS:
                raise RuntimeError("Completed predictions are not from epoch 30")
            if str(saved["context_cache_sha256"].item()) != expected_cache_hash:
                raise RuntimeError("Completed predictions use a different cache")
            if str(saved["config_sha256"].item()) != config_hash:
                raise RuntimeError("Completed predictions use a different config")
            return saved_indices, saved["probabilities"].copy()

    expert_seed = int(expert_config["seed"])
    fold_seed = expert_seed + fold
    train_loader = make_loader(
        dataset,
        indices=outer_train_indices,
        batch_size=int(expert_config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        seed=fold_seed,
    )
    outer_loader = make_loader(
        dataset,
        indices=outer_validation_indices,
        batch_size=int(expert_config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        seed=fold_seed,
    )
    seed_everything(fold_seed)
    model_config = dict(expert_config["model"])
    model = build_model(
        aaindex_lookup=aaindex_lookup,
        window_size=window_size,
        model_config=model_config,
    )
    checkpoint_metadata = {
        "experiment": "mmubipred_context_residual_fixed30_exploratory_v1",
        "protocol_version": PROTOCOL_VERSION,
        "analysis_role": "post-test exploratory fixed-duration rebuild",
        "confirmatory_claim_allowed": False,
        "expert": expert_name,
        "fold": fold,
        "seed": expert_seed,
        "fold_seed": fold_seed,
        "fixed_epochs": FIXED_EPOCHS,
        "early_stopping": False,
        "inner_validation": False,
        "window_size": window_size,
        "model": model_config,
        "outer_train_indices_sha256": indices_sha256(outer_train_indices),
        "outer_validation_indices_sha256": indices_sha256(
            outer_validation_indices
        ),
        "context_sequence_cache_sha256": (
            context_cache_hash if expert_name == "context" else None
        ),
        "config_sha256": config_hash,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "fold_protocol.json",
        {
            **checkpoint_metadata,
            "training_records": len(outer_train_indices),
            "outer_validation_records": len(outer_validation_indices),
            "independent_or_external_test_accessed": False,
        },
    )

    completed_summary = output_dir / "refit_summary.json"
    completed_checkpoint = output_dir / "best.pt"
    if resume and completed_summary.exists() and completed_checkpoint.exists():
        checkpoint = torch.load(
            completed_checkpoint, map_location="cpu", weights_only=False
        )
        if checkpoint.get("metadata") != checkpoint_metadata:
            raise ValueError("Completed fold checkpoint metadata mismatch")
        load_checkpoint_model_state(model, checkpoint)
        print(
            f"fold={fold} expert={expert_name} epoch30 checkpoint reused",
            flush=True,
        )
    else:
        summary = refit_model(
            model=model,
            train_loader=train_loader,
            device=device,
            output_dir=output_dir,
            epochs=FIXED_EPOCHS,
            validation_selected_threshold=0.5,
            checkpoint_metadata=checkpoint_metadata,
            resume=resume,
            compact_checkpoint=True,
            **fixed30_training_arguments(expert_config),
        )
        summary.update(
            {
                "expert": expert_name,
                "fold": fold,
                "fixed_epochs": FIXED_EPOCHS,
                "early_stopping": False,
                "inner_validation": False,
                "outer_fold_accessed_during_training": False,
                "independent_or_external_test_accessed": False,
            }
        )
        write_json(completed_summary, summary)

    model.to(device)
    labels, probabilities, dataset_indices, diagnostics = predict(
        model, outer_loader, device
    )
    order = np.argsort(dataset_indices)
    dataset_indices = dataset_indices[order]
    probabilities = probabilities[order]
    labels = labels[order]
    diagnostics = diagnostics[order]
    if not np.array_equal(dataset_indices, expected_indices):
        raise RuntimeError("Outer prediction indices changed unexpectedly")
    np.savez_compressed(
        prediction_path,
        dataset_indices=dataset_indices,
        labels=labels,
        probabilities=probabilities,
        diagnostics=diagnostics,
        fixed_epochs=np.asarray(FIXED_EPOCHS),
        context_cache_sha256=np.asarray(expected_cache_hash),
        config_sha256=np.asarray(config_hash),
    )
    write_json(
        output_dir / "outer_metrics.json",
        {
            "fixed_epoch": FIXED_EPOCHS,
            "early_stopping": False,
            "checkpoint_selection": "final epoch only",
            "outer_fixed_threshold": compute_metrics(
                labels, probabilities, threshold=0.5
            ),
            "outer_threshold_was_not_selected": True,
            "outer_samples": len(labels),
            "outer_indices_sha256": indices_sha256(dataset_indices),
        },
    )
    model.to("cpu")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dataset_indices, probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fold", type=int, action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-post-test-exploration", action="store_true")
    args = parser.parse_args()
    if not args.allow_post_test_exploration:
        parser.error(
            "Acknowledge this post-test fixed-duration rebuild with "
            "--allow-post-test-exploration"
        )

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_fixed30_protocol(config)
    config_hash = sha256_file(config_path)
    output_dir = resolve_project_path(args.output_dir or str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_project_path(config["data_dir"])
    sequence_cache_path = resolve_project_path(config["sequence_cache"])
    fold_count = int(config["outer_folds"])
    requested_folds = (
        sorted(set(args.fold)) if args.fold is not None else list(range(fold_count))
    )
    if any(fold < 0 or fold >= fold_count for fold in requested_folds):
        raise ValueError(f"fold must be between 0 and {fold_count - 1}")

    all_records, preprocessing = load_released_split(
        data_dir, split="train", window_size=49
    )
    sequences, cache_metadata = load_sequence_cache(sequence_cache_path)
    original_indices, contexts, context_report = build_validated_contexts(
        all_records,
        sequences,
        context_window_size=int(config["context_window_size"]),
    )
    if context_report["validated_fraction"] < float(
        config["minimum_validated_fraction"]
    ):
        raise RuntimeError(
            "Validated context coverage is below the configured minimum: "
            f"{context_report['validated_fraction']:.4f}"
        )
    records = [all_records[index] for index in original_indices]
    local_dataset = SiteDataset(records)
    context_dataset = LongContextSiteDataset(records, contexts)
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    groups = np.asarray([record.protein_id for record in records])
    splitter = StratifiedGroupKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=int(config["seed"]),
    )
    folds = list(splitter.split(np.arange(len(records)), labels, groups))
    fold_assignments = np.full(len(records), -1, dtype=np.int64)
    for fold, (_, validation_indices) in enumerate(folds):
        fold_assignments[validation_indices] = fold
    if np.any(fold_assignments < 0):
        raise AssertionError("Not every record was assigned to an outer fold")

    cache_hash = sequence_cache_sha256(sequence_cache_path)
    manifest = {
        "experiment": str(config["experiment_name"]),
        "protocol_version": PROTOCOL_VERSION,
        "analysis_role": str(config["analysis_role"]),
        "project_git_commit": git_revision(),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "fixed_epochs_per_expert_per_fold": FIXED_EPOCHS,
        "early_stopping": False,
        "inner_validation": False,
        "fold_training_scope": "complete outer-training partition",
        "records_before_context_validation": len(all_records),
        "records_after_context_validation": len(records),
        "context_validation": context_report,
        "sequence_cache": {
            "path": str(sequence_cache_path),
            "sha256": cache_hash,
            "metadata": cache_metadata,
        },
        "preprocessing": {
            name: vars(report) for name, report in preprocessing.items()
        },
        "original_valid_indices_sha256": indices_sha256(
            np.asarray(original_indices)
        ),
        "fold_assignments_sha256": indices_sha256(fold_assignments),
        "historical_test_informed_exploration": True,
        "confirmatory_claim_allowed": False,
        "independent_or_external_test_accessed": False,
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "config_sha256",
            "original_valid_indices_sha256",
            "fold_assignments_sha256",
        ):
            if existing.get(key) != manifest.get(key):
                raise ValueError(f"Existing fixed30 run manifest changed: {key}")
        existing["last_resume_git_commit"] = git_revision()
        write_json(manifest_path, existing)
    else:
        write_json(manifest_path, manifest)
    np.savez_compressed(
        output_dir / "fold_manifest.npz",
        original_dataset_indices=np.asarray(original_indices, dtype=np.int64),
        fold_assignments=fold_assignments,
        labels=labels,
    )

    aaindex_lookup = load_normalized_aaindex(data_dir / "aaindex31.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} validated_records={len(records)}/{len(all_records)} "
        f"folds={fold_count} requested={requested_folds} "
        f"fixed_epochs={FIXED_EPOCHS} early_stopping=False",
        flush=True,
    )
    for fold in requested_folds:
        outer_train_indices, outer_validation_indices = folds[fold]
        print(
            f"fold={fold}/{fold_count - 1} train={len(outer_train_indices)} "
            f"outer_validation={len(outer_validation_indices)}",
            flush=True,
        )
        for expert_name, dataset, window_size in (
            ("local", local_dataset, 49),
            ("context", context_dataset, int(config["context_window_size"])),
        ):
            print(
                f"fold={fold} expert={expert_name} fixed30 started",
                flush=True,
            )
            indices, probabilities = train_fixed_fold_expert(
                expert_name=expert_name,
                expert_config=dict(config[f"{expert_name}_expert"]),
                dataset=dataset,
                records=records,
                outer_train_indices=outer_train_indices,
                outer_validation_indices=outer_validation_indices,
                aaindex_lookup=aaindex_lookup,
                window_size=window_size,
                fold=fold,
                output_dir=output_dir / "folds" / f"fold_{fold}" / expert_name,
                device=device,
                num_workers=int(config["num_workers"]),
                resume=args.resume,
                context_cache_hash=cache_hash,
                config_hash=config_hash,
            )
            fixed = compute_metrics(labels[indices], probabilities, threshold=0.5)
            print(
                f"fold={fold} expert={expert_name} "
                f"epoch30_outer_mcc@0.5={fixed['mcc']:.5f}",
                flush=True,
            )

    completed = all(
        (
            output_dir
            / "folds"
            / f"fold_{fold}"
            / expert
            / "outer_predictions.npz"
        ).exists()
        for fold in range(fold_count)
        for expert in ("local", "context")
    )
    if completed:
        summary = finish_oof_summary(
            output_dir,
            records,
            fold_assignments,
            l2_strength=float(config["stacker_l2_strength"]),
        )
        summary.update(
            {
                "experiment": str(config["experiment_name"]),
                "analysis_role": str(config["analysis_role"]),
                "fixed_epochs_per_expert_per_fold": FIXED_EPOCHS,
                "early_stopping": False,
                "inner_validation": False,
                "historical_test_informed_exploration": True,
                "confirmatory_claim_allowed": False,
                "independent_or_external_test_accessed": False,
            }
        )
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        completed_count = sum(
            (
                output_dir
                / "folds"
                / f"fold_{fold}"
                / expert
                / "outer_predictions.npz"
            ).exists()
            for fold in range(fold_count)
            for expert in ("local", "context")
        )
        print(
            f"Completed expert-fold predictions: {completed_count}/"
            f"{fold_count * 2}. Run remaining folds with --resume.",
            flush=True,
        )
    print("Released independent and external test sets were not accessed.")


if __name__ == "__main__":
    main()
