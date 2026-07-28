#!/usr/bin/env python3
"""Frozen full-data refit for the advanced long-context ESM-2 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.context import (  # noqa: E402
    LongContextSiteDataset,
    build_validated_contexts,
    load_sequence_cache,
    sequence_cache_sha256,
)
from ubipred.data import (  # noqa: E402
    load_normalized_aaindex,
    make_loader,
    seed_everything,
)
from ubipred.engine import refit_model  # noqa: E402
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.metrics import write_json  # noqa: E402
from ubipred.model import build_model  # noqa: E402


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def int_array_sha256(values: list[int] | np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype=np.int64).tobytes()
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


def frozen_context_epochs(
    development_run_dir: Path,
    fold_count: int,
) -> tuple[list[int], int]:
    best_epochs: list[int] = []
    for fold in range(fold_count):
        metrics_path = (
            development_run_dir
            / "folds"
            / f"fold_{fold}"
            / "context"
            / "outer_metrics.json"
        )
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing completed context-fold metrics: {metrics_path}"
            )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        best_epochs.append(int(metrics["inner_selection"]["best_epoch"]))

    median_epoch = statistics.median(best_epochs)
    if int(median_epoch) != median_epoch:
        raise RuntimeError(
            "The median context best epoch is not an integer; freeze an "
            "explicit rule before refitting"
        )
    return best_epochs, int(median_epoch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume exactly after the most recently completed refit epoch.",
    )
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    final_config = dict(config["final_context_refit"])
    development_run_dir = resolve_project_path(
        args.development_run_dir or str(config["output_dir"])
    )
    output_dir = resolve_project_path(
        args.output_dir or str(final_config["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stabilized_summary_path = development_run_dir / "stabilized_summary.json"
    if not stabilized_summary_path.exists():
        raise FileNotFoundError(
            "The corrected five-fold comparison must be completed first: "
            f"{stabilized_summary_path}"
        )
    stabilized_summary = json.loads(
        stabilized_summary_path.read_text(encoding="utf-8")
    )
    if not bool(stabilized_summary.get("comparison_valid")):
        raise RuntimeError("The corrected development comparison is invalid")
    decision = str(stabilized_summary.get("amended_candidate_decision"))
    if decision != "ADVANCE_CONTEXT_ONLY":
        raise RuntimeError(
            "Frozen refit requires amended_candidate_decision="
            f"ADVANCE_CONTEXT_ONLY, received {decision}"
        )

    fold_count = int(config["outer_folds"])
    best_epochs, selected_epochs = frozen_context_epochs(
        development_run_dir, fold_count
    )
    configured_epochs = int(final_config["epochs"])
    configured_best_epochs = [
        int(value) for value in final_config["source_best_epochs"]
    ]
    if best_epochs != configured_best_epochs:
        raise RuntimeError(
            "Recorded fold best epochs do not match the frozen configuration: "
            f"observed={best_epochs}, configured={configured_best_epochs}"
        )
    if selected_epochs != configured_epochs:
        raise RuntimeError(
            "Median fold epoch does not match the frozen configuration: "
            f"median={selected_epochs}, configured={configured_epochs}"
        )

    data_dir = resolve_project_path(config["data_dir"])
    sequence_cache_path = resolve_project_path(config["sequence_cache"])
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
    dataset = LongContextSiteDataset(
        records,
        contexts,
        original_indices=original_indices,
    )

    context_config = dict(config["context_expert"])
    model_config = dict(context_config["model"])
    seed = int(final_config["seed"])
    cache_hash = sequence_cache_sha256(sequence_cache_path)
    source_summary_hash = sha256_file(stabilized_summary_path)
    valid_indices_hash = int_array_sha256(original_indices)
    checkpoint_metadata = {
        "experiment": str(final_config["experiment_name"]),
        "candidate": "context_only",
        "seed": seed,
        "refit_epochs": configured_epochs,
        "epoch_selection_rule": str(final_config["epoch_selection_rule"]),
        "source_best_epochs": best_epochs,
        "model": model_config,
        "context_window_size": int(config["context_window_size"]),
        "validated_training_records": len(records),
        "validated_training_indices_sha256": valid_indices_hash,
        "context_sequence_cache_sha256": cache_hash,
        "stabilized_summary_sha256": source_summary_hash,
        "training": {
            key: context_config.get(key)
            for key in (
                "batch_size",
                "learning_rate",
                "weight_decay",
                "optimizer",
                "optimizer_epsilon",
                "gradient_clip_norm",
                "gradient_accumulation_steps",
                "use_amp",
            )
        },
    }
    manifest = {
        "experiment": str(final_config["experiment_name"]),
        "status": "frozen full-data refit",
        "candidate": "context_only",
        "project_git_commit": git_revision(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "development_run_dir": str(development_run_dir),
        "stabilized_summary": {
            "path": str(stabilized_summary_path),
            "sha256": source_summary_hash,
            "comparison_valid": True,
            "amended_candidate_decision": decision,
        },
        "epoch_selection": {
            "rule": str(final_config["epoch_selection_rule"]),
            "fold_best_epochs": best_epochs,
            "selected_epochs": configured_epochs,
        },
        "released_training_records_before_context_validation": len(all_records),
        "released_training_records_after_context_validation": len(records),
        "context_validation": context_report,
        "validated_training_indices_sha256": valid_indices_hash,
        "sequence_cache": {
            "path": str(sequence_cache_path),
            "sha256": cache_hash,
            "metadata": cache_metadata,
        },
        "preprocessing": {
            name: vars(report) for name, report in preprocessing.items()
        },
        "model": model_config,
        "training": checkpoint_metadata["training"],
        "seed": seed,
        "reporting_threshold": 0.5,
        "validation_or_test_used_during_refit": False,
        "independent_test_accessed": False,
    }
    write_json(output_dir / "refit_manifest.json", manifest)

    completed_summary_path = output_dir / "refit_summary.json"
    completed_checkpoint_path = output_dir / "best.pt"
    if (
        args.resume
        and completed_summary_path.exists()
        and completed_checkpoint_path.exists()
    ):
        completed_checkpoint = torch.load(
            completed_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if completed_checkpoint.get("metadata") != checkpoint_metadata:
            raise ValueError(
                "Existing completed checkpoint does not match this frozen "
                "code/configuration"
            )
        if (
            completed_checkpoint.get("state_dict_scope")
            != "trainable_parameters"
        ):
            raise ValueError(
                "Existing completed checkpoint is not the expected compact "
                "trainable-parameter artifact"
            )
        completed = json.loads(
            completed_summary_path.read_text(encoding="utf-8")
        )
        print(json.dumps(completed, indent=2, sort_keys=True))
        print("Completed frozen refit already exists; training was skipped.")
        print("Released independent test set was not accessed.")
        return

    seed_everything(seed)
    train_loader = make_loader(
        dataset,
        indices=None,
        batch_size=int(context_config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        seed=seed,
    )
    aaindex_lookup = load_normalized_aaindex(data_dir / "aaindex31.txt")
    model = build_model(
        aaindex_lookup=aaindex_lookup,
        window_size=int(config["context_window_size"]),
        model_config=model_config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"device={device} records={len(records)} epochs={configured_epochs} "
        f"trainable_parameters={trainable_parameters}",
        flush=True,
    )

    clip = context_config.get("gradient_clip_norm", 1.0)
    summary = refit_model(
        model=model,
        train_loader=train_loader,
        device=device,
        output_dir=output_dir,
        epochs=configured_epochs,
        learning_rate=float(context_config["learning_rate"]),
        weight_decay=float(context_config["weight_decay"]),
        optimizer_name=str(context_config.get("optimizer", "adamw")),
        optimizer_epsilon=float(
            context_config.get("optimizer_epsilon", 1e-8)
        ),
        gradient_clip_norm=None if clip is None else float(clip),
        gradient_accumulation_steps=int(
            context_config.get("gradient_accumulation_steps", 1)
        ),
        use_amp=bool(context_config.get("use_amp", False)),
        validation_selected_threshold=0.5,
        checkpoint_metadata=checkpoint_metadata,
        resume=args.resume,
        compact_checkpoint=True,
    )
    summary["epoch_selection_rule"] = str(
        final_config["epoch_selection_rule"]
    )
    summary["source_best_epochs"] = best_epochs
    summary["independent_test_accessed"] = False
    write_json(output_dir / "refit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Frozen context-only refit completed without validation or test access.")
    print("Released independent test set was not accessed.")


if __name__ == "__main__":
    main()
