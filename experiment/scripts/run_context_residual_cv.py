#!/usr/bin/env python3
"""Leakage-safe OOF benchmark for MMUbiPred–Context Residual v1."""

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
    make_train_validation_indices,
    seed_everything,
)
from ubipred.engine import predict, train_model  # noqa: E402
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.metrics import compute_metrics, select_mcc_threshold, write_json  # noqa: E402
from ubipred.model import build_model  # noqa: E402
from ubipred.stacking import fit_residual_stacker  # noqa: E402


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def indices_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()


def training_arguments(config: dict[str, object]) -> dict[str, object]:
    clip = config.get("gradient_clip_norm", 1.0)
    return {
        "epochs": int(config["epochs"]),
        "patience": int(config["early_stopping_patience"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "optimizer_name": str(config.get("optimizer", "adamw")),
        "optimizer_epsilon": float(config.get("optimizer_epsilon", 1e-8)),
        "gradient_clip_norm": None if clip is None else float(clip),
        "gradient_accumulation_steps": int(
            config.get("gradient_accumulation_steps", 1)
        ),
        "use_amp": bool(config.get("use_amp", False)),
    }


def train_and_predict_fold(
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
    seed: int,
    output_dir: Path,
    device: torch.device,
    num_workers: int,
    resume: bool,
    context_cache_hash: str,
) -> tuple[np.ndarray, np.ndarray]:
    prediction_path = output_dir / "outer_predictions.npz"
    if prediction_path.exists() and resume:
        saved = np.load(prediction_path)
        expected_indices = np.sort(outer_validation_indices)
        saved_indices = saved["dataset_indices"].astype(np.int64)
        if not np.array_equal(saved_indices, expected_indices):
            raise RuntimeError(
                f"Stale outer predictions for fold {fold} expert {expert_name}"
            )
        saved_cache_hash = str(saved["context_cache_sha256"].item())
        expected_cache_hash = (
            context_cache_hash if expert_name == "context" else ""
        )
        if saved_cache_hash != expected_cache_hash:
            raise RuntimeError(
                f"Sequence cache changed for fold {fold} expert {expert_name}"
            )
        return saved_indices, saved["probabilities"]

    outer_training_records = [records[index] for index in outer_train_indices]
    inner_train_relative, inner_validation_relative = make_train_validation_indices(
        outer_training_records,
        validation_fraction=float(expert_config["inner_validation_fraction"]),
        seed=seed + fold,
        strategy="protein_grouped",
    )
    inner_train_indices = outer_train_indices[inner_train_relative]
    inner_validation_indices = outer_train_indices[inner_validation_relative]
    outer_groups = {records[index].protein_id for index in outer_validation_indices}
    if outer_groups & {
        records[index].protein_id for index in inner_train_indices
    }:
        raise AssertionError("Outer-fold protein leakage into expert training")

    train_loader = make_loader(
        dataset,
        indices=inner_train_indices,
        batch_size=int(expert_config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        seed=seed + fold,
    )
    inner_validation_loader = make_loader(
        dataset,
        indices=inner_validation_indices,
        batch_size=int(expert_config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        seed=seed + fold,
    )
    outer_loader = make_loader(
        dataset,
        indices=outer_validation_indices,
        batch_size=int(expert_config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        seed=seed + fold,
    )
    seed_everything(seed + fold)
    model = build_model(
        aaindex_lookup=aaindex_lookup,
        window_size=window_size,
        model_config=dict(expert_config["model"]),
    )
    checkpoint_metadata = {
        "experiment": "mmubipred_context_residual_v1",
        "expert": expert_name,
        "fold": fold,
        "seed": seed,
        "window_size": window_size,
        "model": expert_config["model"],
        "outer_train_indices_sha256": indices_sha256(outer_train_indices),
        "outer_validation_indices_sha256": indices_sha256(
            outer_validation_indices
        ),
        "inner_train_indices_sha256": indices_sha256(inner_train_indices),
        "inner_validation_indices_sha256": indices_sha256(
            inner_validation_indices
        ),
        "context_sequence_cache_sha256": (
            context_cache_hash if expert_name == "context" else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=inner_validation_loader,
        device=device,
        output_dir=output_dir,
        checkpoint_metadata=checkpoint_metadata,
        resume=resume,
        **training_arguments(expert_config),
    )
    labels, probabilities, dataset_indices, diagnostics = predict(
        model, outer_loader, device
    )
    order = np.argsort(dataset_indices)
    dataset_indices = dataset_indices[order]
    probabilities = probabilities[order]
    labels = labels[order]
    diagnostics = diagnostics[order]
    np.savez_compressed(
        prediction_path,
        dataset_indices=dataset_indices,
        labels=labels,
        probabilities=probabilities,
        diagnostics=diagnostics,
        context_cache_sha256=np.asarray(
            context_cache_hash if expert_name == "context" else ""
        ),
    )
    write_json(
        output_dir / "outer_metrics.json",
        {
            "inner_selection": summary,
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


def finish_oof_summary(
    output_dir: Path,
    records: list,
    fold_assignments: np.ndarray,
    l2_strength: float,
    local_directory_name: str = "local",
    summary_filename: str = "summary.json",
    predictions_filename: str = "oof_predictions.npz",
) -> dict[str, object]:
    sample_count = len(records)
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    expert_probabilities: dict[str, np.ndarray] = {}
    per_fold_metrics: list[dict[str, object]] = []
    collapsed_expert_folds: list[dict[str, object]] = []
    expert_directories = {
        "local": local_directory_name,
        "context": "context",
    }
    for expert_name in ("local", "context"):
        probabilities = np.full(sample_count, np.nan, dtype=np.float64)
        for fold in sorted(set(fold_assignments.tolist())):
            saved = np.load(
                output_dir
                / "folds"
                / f"fold_{fold}"
                / expert_directories[expert_name]
                / "outer_predictions.npz"
            )
            indices = saved["dataset_indices"].astype(np.int64)
            expected_indices = np.flatnonzero(fold_assignments == fold)
            if not np.array_equal(indices, expected_indices):
                raise RuntimeError(
                    f"OOF indices do not match fold {fold} for {expert_name}"
                )
            if not np.array_equal(
                saved["labels"].astype(np.int64), labels[indices]
            ):
                raise RuntimeError(
                    f"OOF labels do not match fold {fold} for {expert_name}"
                )
            if np.any(~np.isnan(probabilities[indices])):
                raise RuntimeError(f"Duplicate OOF predictions for {expert_name}")
            fold_probabilities = saved["probabilities"]
            probabilities[indices] = fold_probabilities
            fixed = compute_metrics(labels[indices], fold_probabilities, 0.5)
            collapsed = (
                (
                    float(fixed["sensitivity"]) == 1.0
                    and float(fixed["specificity"]) == 0.0
                )
                or (
                    float(fixed["sensitivity"]) == 0.0
                    and float(fixed["specificity"]) == 1.0
                )
            )
            per_fold_metrics.append(
                {
                    "fold": int(fold),
                    "expert": expert_name,
                    "collapsed": collapsed,
                    "fixed_threshold": fixed,
                }
            )
            if collapsed:
                collapsed_expert_folds.append(
                    {"fold": int(fold), "expert": expert_name}
                )
        if np.isnan(probabilities).any():
            raise RuntimeError(f"Incomplete OOF predictions for {expert_name}")
        expert_probabilities[expert_name] = probabilities

    crossfit_probabilities = np.full(sample_count, np.nan, dtype=np.float64)
    crossfit_stackers: list[dict[str, object]] = []
    for fold in sorted(set(fold_assignments.tolist())):
        held_out = fold_assignments == fold
        stacker = fit_residual_stacker(
            labels[~held_out],
            expert_probabilities["local"][~held_out],
            expert_probabilities["context"][~held_out],
            l2_strength=l2_strength,
        )
        crossfit_probabilities[held_out] = stacker.predict_proba(
            expert_probabilities["local"][held_out],
            expert_probabilities["context"][held_out],
        )
        crossfit_stackers.append({"held_out_fold": int(fold), **stacker.as_dict()})

    final_stacker = fit_residual_stacker(
        labels,
        expert_probabilities["local"],
        expert_probabilities["context"],
        l2_strength=l2_strength,
    )
    average_probabilities = (
        expert_probabilities["local"] + expert_probabilities["context"]
    ) / 2.0
    probability_sets = {
        "local_expert": expert_probabilities["local"],
        "context_expert": expert_probabilities["context"],
        "unweighted_average": average_probabilities,
        "residual_stack_crossfit": crossfit_probabilities,
    }
    metrics: dict[str, object] = {}
    for name, probabilities in probability_sets.items():
        threshold, _ = select_mcc_threshold(labels, probabilities)
        metrics[name] = {
            "fixed_threshold": compute_metrics(labels, probabilities, 0.5),
            "oof_selected_threshold_exploratory": compute_metrics(
                labels, probabilities, threshold
            ),
        }
    local_fixed = metrics["local_expert"]["fixed_threshold"]
    context_fixed = metrics["context_expert"]["fixed_threshold"]
    stack_fixed = metrics["residual_stack_crossfit"]["fixed_threshold"]
    deltas = {
        metric: float(stack_fixed[metric]) - float(local_fixed[metric])
        for metric in ("mcc", "accuracy", "auroc", "auprc")
    }
    decision_checks = {
        "mcc_gain_at_least_0.01": deltas["mcc"] >= 0.01,
        "accuracy_drop_at_most_0.005": deltas["accuracy"] >= -0.005,
        "auroc_drop_at_most_0.002": deltas["auroc"] >= -0.002,
        "auprc_drop_at_most_0.002": deltas["auprc"] >= -0.002,
    }
    context_minus_local = {
        metric: float(context_fixed[metric]) - float(local_fixed[metric])
        for metric in ("mcc", "accuracy", "auroc", "auprc")
    }
    stack_minus_context = {
        metric: float(stack_fixed[metric]) - float(context_fixed[metric])
        for metric in ("mcc", "accuracy", "auroc", "auprc")
    }
    context_advancement_checks = {
        "context_mcc_gain_over_local_at_least_0.01": (
            context_minus_local["mcc"] >= 0.01
        ),
        "context_accuracy_drop_vs_local_at_most_0.005": (
            context_minus_local["accuracy"] >= -0.005
        ),
        "context_auroc_drop_vs_local_at_most_0.002": (
            context_minus_local["auroc"] >= -0.002
        ),
        "context_auprc_drop_vs_local_at_most_0.002": (
            context_minus_local["auprc"] >= -0.002
        ),
    }
    fusion_advancement_checks = {
        "stack_mcc_gain_over_context_at_least_0.005": (
            stack_minus_context["mcc"] >= 0.005
        ),
        "stack_accuracy_drop_vs_context_at_most_0.005": (
            stack_minus_context["accuracy"] >= -0.005
        ),
        "stack_auroc_drop_vs_context_at_most_0.002": (
            stack_minus_context["auroc"] >= -0.002
        ),
        "stack_auprc_drop_vs_context_at_most_0.002": (
            stack_minus_context["auprc"] >= -0.002
        ),
    }
    if collapsed_expert_folds:
        amended_candidate_decision = "INVALID_COLLAPSED_EXPERT"
    elif all(decision_checks.values()) and all(
        fusion_advancement_checks.values()
    ):
        amended_candidate_decision = "ADVANCE_RESIDUAL_STACK"
    elif all(context_advancement_checks.values()):
        amended_candidate_decision = "ADVANCE_CONTEXT_ONLY"
    else:
        amended_candidate_decision = "STOP"
    np.savez_compressed(
        output_dir / predictions_filename,
        labels=labels,
        fold_assignments=fold_assignments,
        local_probabilities=expert_probabilities["local"],
        context_probabilities=expert_probabilities["context"],
        unweighted_average_probabilities=average_probabilities,
        residual_stack_crossfit_probabilities=crossfit_probabilities,
    )
    summary = {
        "evaluation_scope": (
            "protein-grouped out-of-fold predictions on validated released "
            "training records only"
        ),
        "primary_reporting_threshold": 0.5,
        "metrics": metrics,
        "per_fold_metrics": per_fold_metrics,
        "collapsed_expert_folds": collapsed_expert_folds,
        "comparison_valid": not collapsed_expert_folds,
        "stack_minus_local_fixed_threshold": deltas,
        "context_minus_local_fixed_threshold": context_minus_local,
        "stack_minus_context_fixed_threshold": stack_minus_context,
        "predeclared_decision_checks": decision_checks,
        "predeclared_decision": (
            "INVALID_COLLAPSED_EXPERT"
            if collapsed_expert_folds
            else ("GO" if all(decision_checks.values()) else "STOP")
        ),
        "amended_context_advancement_checks": context_advancement_checks,
        "amended_fusion_advancement_checks": fusion_advancement_checks,
        "amended_candidate_decision": amended_candidate_decision,
        "crossfit_stackers": crossfit_stackers,
        "final_stacker_for_future_refit": final_stacker.as_dict(),
        "independent_test_accessed": False,
    }
    write_json(output_dir / summary_filename, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--fold",
        type=int,
        action="append",
        help="Run only this outer fold; repeat the option for multiple folds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume checkpoints and reuse completed outer-fold predictions.",
    )
    args = parser.parse_args()
    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = resolve_project_path(
        args.output_dir or str(config["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_project_path(config["data_dir"])
    sequence_cache_path = resolve_project_path(config["sequence_cache"])
    seed = int(config["seed"])
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
        n_splits=fold_count, shuffle=True, random_state=seed
    )
    folds = list(splitter.split(np.arange(len(records)), labels, groups))
    fold_assignments = np.full(len(records), -1, dtype=np.int64)
    for fold, (_, validation_indices) in enumerate(folds):
        fold_assignments[validation_indices] = fold
    if np.any(fold_assignments < 0):
        raise AssertionError("Not every validated record was assigned to a fold")

    cache_hash = sequence_cache_sha256(sequence_cache_path)
    manifest = {
        "config": config,
        "config_path": str(config_path),
        "project_git_commit": git_revision(),
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
        "independent_test_accessed": False,
    }
    write_json(output_dir / "run_manifest.json", manifest)
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
        f"folds={fold_count} requested={requested_folds}",
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
            print(f"fold={fold} expert={expert_name} started", flush=True)
            indices, probabilities = train_and_predict_fold(
                expert_name=expert_name,
                expert_config=dict(config[f"{expert_name}_expert"]),
                dataset=dataset,
                records=records,
                outer_train_indices=outer_train_indices,
                outer_validation_indices=outer_validation_indices,
                aaindex_lookup=aaindex_lookup,
                window_size=window_size,
                fold=fold,
                seed=seed,
                output_dir=output_dir
                / "folds"
                / f"fold_{fold}"
                / expert_name,
                device=device,
                num_workers=int(config["num_workers"]),
                resume=args.resume,
                context_cache_hash=cache_hash,
            )
            fixed = compute_metrics(
                labels[indices], probabilities, threshold=0.5
            )
            print(
                f"fold={fold} expert={expert_name} "
                f"outer_mcc@0.5={fixed['mcc']:.5f}",
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
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "Requested folds completed. The OOF summary will be generated after "
            "both experts have completed every fold.",
            flush=True,
        )
    print("Released independent test set was not accessed.")


if __name__ == "__main__":
    main()
