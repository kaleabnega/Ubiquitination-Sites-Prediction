#!/usr/bin/env python3
"""Evaluate the frozen long-context candidate once on the historical test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
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
from ubipred.data import load_normalized_aaindex, make_loader  # noqa: E402
from ubipred.engine import load_checkpoint_model_state, predict  # noqa: E402
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.metrics import compute_metrics, write_json  # noqa: E402
from ubipred.model import build_model  # noqa: E402


PAPER_BASELINE = {
    "mcc": 0.5458386140298045,
    "accuracy": 0.7725035719955549,
    "sensitivity": 0.7498020585906572,
    "specificity": 0.8067729083665338,
    "confusion_matrix": [[4050, 970], [1896, 5682]],
    "support": {"negative": 5020, "positive": 7578, "total": 12598},
}


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def int_array_sha256(values: list[int] | np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype=np.int64).tobytes()
    ).hexdigest()


def ensure_outputs_absent(run_dir: Path) -> None:
    metrics_path = run_dir / "context_test_metrics.json"
    if metrics_path.exists():
        raise FileExistsError(
            "Refusing to repeat the frozen historical-test evaluation because "
            f"the completed metrics file already exists: {metrics_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sequence-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--expected-experiment")
    parser.add_argument("--expected-refit-epochs", type=int)
    parser.add_argument(
        "--allow-test-informed-exploration",
        action="store_true",
        help=(
            "Acknowledge that this checkpoint was designed after inspecting "
            "the historical test and cannot support a confirmatory claim."
        ),
    )
    parser.add_argument(
        "--allow-locked-test",
        action="store_true",
        help="Required acknowledgement that this evaluates the historical test.",
    )
    args = parser.parse_args()
    if not args.allow_locked_test:
        parser.error(
            "Refusing to evaluate the independent test without "
            "--allow-locked-test"
        )

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = resolve_project_path(args.run_dir)
    sequence_cache_path = resolve_project_path(args.sequence_cache)
    ensure_outputs_absent(run_dir)

    checkpoint_path = run_dir / "best.pt"
    manifest_path = run_dir / "refit_manifest.json"
    summary_path = run_dir / "refit_summary.json"
    for required_path in (checkpoint_path, manifest_path, summary_path):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Missing frozen-refit artifact: {required_path}"
            )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("Frozen checkpoint metadata is missing")
    final_config = dict(config["final_context_refit"])
    expected_experiment = (
        args.expected_experiment or str(final_config["experiment_name"])
    )
    expected_refit_epochs = (
        args.expected_refit_epochs
        if args.expected_refit_epochs is not None
        else int(final_config["epochs"])
    )
    is_test_informed_exploration = (
        expected_experiment != final_config["experiment_name"]
        or expected_refit_epochs != int(final_config["epochs"])
    )
    if (
        is_test_informed_exploration
        and not args.allow_test_informed_exploration
    ):
        raise ValueError(
            "Non-frozen checkpoint evaluation requires explicit "
            "--allow-test-informed-exploration acknowledgement"
        )
    if metadata.get("experiment") != expected_experiment:
        raise ValueError("Checkpoint experiment identity does not match")
    if int(checkpoint.get("refit_epochs", -1)) != expected_refit_epochs:
        raise ValueError("Checkpoint refit duration does not match the protocol")
    if checkpoint.get("state_dict_scope") != "trainable_parameters":
        raise ValueError("Expected a compact trainable-parameter checkpoint")

    refit_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if refit_manifest.get("candidate") != "context_only":
        raise ValueError("Refit manifest is not for the context-only candidate")

    data_dir = resolve_project_path(config["data_dir"])
    all_records, preprocessing = load_released_split(
        data_dir, split="test", window_size=49
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
            "Independent-test context coverage is below the frozen minimum: "
            f"{context_report['validated_fraction']:.4f}"
        )
    records = [all_records[index] for index in original_indices]
    dataset = LongContextSiteDataset(
        records,
        contexts,
        original_indices=original_indices,
    )
    loader = make_loader(
        dataset,
        indices=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=0,
    )

    model_config = dict(config["context_expert"]["model"])
    if metadata.get("model") != model_config:
        raise ValueError("Checkpoint model configuration has changed")
    model = build_model(
        aaindex_lookup=load_normalized_aaindex(data_dir / "aaindex31.txt"),
        window_size=int(config["context_window_size"]),
        model_config=model_config,
    )
    load_checkpoint_model_state(model, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(
        f"device={device} validated_test_records={len(records)}/"
        f"{len(all_records)}",
        flush=True,
    )
    labels, probabilities, indices, diagnostics = predict(
        model, loader, device
    )
    order = np.argsort(indices)
    labels = labels[order]
    probabilities = probabilities[order]
    indices = indices[order]
    diagnostics = diagnostics[order]
    if not np.array_equal(
        indices, np.asarray(original_indices, dtype=np.int64)
    ):
        raise RuntimeError("Prediction indices do not match validated test sites")

    fixed_metrics = compute_metrics(labels, probabilities, threshold=0.5)
    full_cohort_comparison_valid = (
        len(records) == len(all_records)
        and np.array_equal(indices, np.arange(len(all_records)))
    )
    difference_from_paper = (
        {
            metric: float(fixed_metrics[metric]) - float(PAPER_BASELINE[metric])
            for metric in ("mcc", "accuracy", "sensitivity", "specificity")
        }
        if full_cohort_comparison_valid
        else None
    )
    diagnostic_name = str(
        getattr(model, "diagnostic_name", "branch_diagnostics")
    )
    results = {
        "status": (
            "post-test exploratory context model evaluated"
            if is_test_informed_exploration
            else "frozen context-only historical test evaluated"
        ),
        "historical_test_informed_exploration": (
            is_test_informed_exploration
        ),
        "confirmatory_claim_allowed": not is_test_informed_exploration,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_role": (
            f"post-test exploratory epoch-{expected_refit_epochs} extension"
            if is_test_informed_exploration
            else "fixed seven-epoch full-data refit"
        ),
        "training_scope": (
            "all released training records passing frozen context validation"
        ),
        "device": str(device),
        "test_context_cache": {
            "path": str(sequence_cache_path),
            "sha256": sequence_cache_sha256(sequence_cache_path),
            "metadata": cache_metadata,
        },
        "preprocessing": {
            name: asdict(report) for name, report in preprocessing.items()
        },
        "context_validation": context_report,
        "validated_test_indices_sha256": int_array_sha256(original_indices),
        "fixed_threshold": fixed_metrics,
        "primary_threshold": 0.5,
        "branch_names": list(getattr(model, "branch_names", ())),
        "branch_diagnostic_name": diagnostic_name,
        f"mean_{diagnostic_name}": diagnostics.mean(axis=0).tolist(),
        "historical_full_paper_baseline": PAPER_BASELINE,
        "direct_full_cohort_paper_comparison_valid": (
            full_cohort_comparison_valid
        ),
        "comparison_scope": (
            "exploratory only; training duration was chosen after the "
            "historical test was inspected"
            if is_test_informed_exploration
            else "frozen primary historical-test evaluation"
        ),
        "fixed_threshold_difference_from_full_paper_baseline": (
            difference_from_paper
        ),
        "matched_comparison_note": (
            "Both results use every processed released test record."
            if full_cohort_comparison_valid
            else (
                "The context model could not evaluate every released test "
                "record. MMUbiPred must be evaluated on these exact validated "
                "indices before making a matched claim."
            )
        ),
    }
    metrics_path = run_dir / "context_test_metrics.json"
    predictions_path = run_dir / "context_test_predictions.npz"
    table_path = run_dir / "context_test_predictions.tsv"
    np.savez_compressed(
        predictions_path,
        labels=labels,
        probabilities=probabilities,
        dataset_indices=indices,
        branch_diagnostics=diagnostics,
    )
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "processed_test_index",
                "protein_id",
                "position",
                "label",
                "probability",
                "prediction_at_0.5",
                *[
                    f"{name}_{diagnostic_name}"
                    for name in getattr(model, "branch_names", ())
                ],
            ]
        )
        for label, probability, index, diagnostic_values in zip(
            labels, probabilities, indices, diagnostics
        ):
            record = all_records[int(index)]
            writer.writerow(
                [
                    int(index),
                    record.protein_id,
                    record.position,
                    int(label),
                    float(probability),
                    int(probability >= 0.5),
                    *[float(value) for value in diagnostic_values],
                ]
            )

    refit_manifest["independent_test_accessed"] = True
    refit_manifest["historical_test_evaluation"] = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": results["checkpoint_sha256"],
        "test_sequence_cache_sha256": results["test_context_cache"]["sha256"],
        "validated_test_indices_sha256": results[
            "validated_test_indices_sha256"
        ],
        "primary_threshold": 0.5,
        "metrics_file": metrics_path.name,
        "predictions_file": predictions_path.name,
        "predictions_table": table_path.name,
        "direct_full_cohort_paper_comparison_valid": (
            full_cohort_comparison_valid
        ),
        "historical_test_informed_exploration": (
            is_test_informed_exploration
        ),
    }
    write_json(manifest_path, refit_manifest)
    # Write the one-time guard only after every supporting artifact and the
    # manifest are complete, so a storage interruption can safely retry the
    # identical frozen model.
    write_json(metrics_path, results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
