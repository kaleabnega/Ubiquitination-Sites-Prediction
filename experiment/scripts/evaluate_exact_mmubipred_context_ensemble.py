#!/usr/bin/env python3
"""Evaluate a frozen equal-weight exact-MMUbiPred/context ensemble."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.ensemble import equal_probability_average  # noqa: E402
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.metrics import compute_metrics, write_json  # noqa: E402


EXPECTED_RELEASED_MODEL_SHA256 = (
    "a9393c05635f8019d08fe916885bdf3f31225859b2d0cb4130ec4c2bd89a5001"
)
EXPECTED_FULL_PAPER_CONFUSION_MATRIX = [[4050, 970], [1896, 5682]]
COMPARISON_METRICS = (
    "mcc",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "auroc",
    "auprc",
)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def int_array_sha256(values: np.ndarray) -> str:
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


def metric_differences(
    candidate: dict[str, object],
    reference: dict[str, object],
) -> dict[str, float]:
    return {
        metric: float(candidate[metric]) - float(reference[metric])
        for metric in COMPARISON_METRICS
    }


def assert_metrics_match(
    observed: dict[str, object],
    recorded: dict[str, object],
    name: str,
) -> None:
    if observed["confusion_matrix"] != recorded["confusion_matrix"]:
        raise RuntimeError(f"{name} confusion matrix changed")
    for metric in COMPARISON_METRICS:
        if not np.isclose(
            float(observed[metric]),
            float(recorded[metric]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"{name} metric changed: {metric}")


def paired_correctness(
    labels: np.ndarray,
    first_probabilities: np.ndarray,
    second_probabilities: np.ndarray,
    threshold: float,
    first_name: str,
    second_name: str,
) -> dict[str, int]:
    first_correct = (first_probabilities >= threshold).astype(int) == labels
    second_correct = (second_probabilities >= threshold).astype(int) == labels
    return {
        "both_correct": int(np.sum(first_correct & second_correct)),
        f"{first_name}_only_correct": int(
            np.sum(first_correct & ~second_correct)
        ),
        f"{second_name}_only_correct": int(
            np.sum(~first_correct & second_correct)
        ),
        "both_wrong": int(np.sum(~first_correct & ~second_correct)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--context-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-post-test-exploration",
        action="store_true",
        help="Acknowledge that this is not a confirmatory test.",
    )
    parser.add_argument(
        "--allow-locked-test",
        action="store_true",
        help="Acknowledge reuse of saved historical-test predictions.",
    )
    args = parser.parse_args()
    if not args.allow_post_test_exploration:
        parser.error(
            "Exact-model ensemble requires --allow-post-test-exploration"
        )
    if not args.allow_locked_test:
        parser.error("Exact-model ensemble requires --allow-locked-test")

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ensemble_config = dict(
        config["exploratory_exact_paper_context_ensemble"]
    )
    if (
        ensemble_config.get("fusion") != "equal_probability_average"
        or float(ensemble_config.get("paper_weight", -1)) != 0.5
        or float(ensemble_config.get("context_weight", -1)) != 0.5
        or bool(ensemble_config.get("trained_fusion_parameters", True))
    ):
        raise ValueError(
            "The primary ensemble must remain an untrained equal-probability "
            "average"
        )
    threshold = float(ensemble_config["reporting_threshold"])
    if threshold != 0.5:
        raise ValueError("The primary reporting threshold must remain 0.5")

    context_run_dir = resolve_project_path(
        args.context_run_dir or str(ensemble_config["context_run_dir"])
    )
    output_dir = resolve_project_path(
        args.output_dir or str(ensemble_config["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "ensemble_test_metrics.json"
    predictions_path = output_dir / "ensemble_test_predictions.npz"
    predictions_table_path = output_dir / "ensemble_test_predictions.tsv"
    manifest_path = output_dir / "ensemble_manifest.json"
    if metrics_path.exists():
        raise FileExistsError(
            "Refusing to repeat the completed exact-model ensemble "
            f"evaluation: {metrics_path}"
        )

    context_metrics_path = context_run_dir / "context_test_metrics.json"
    context_predictions_path = (
        context_run_dir / "context_test_predictions.npz"
    )
    matched_comparison_path = (
        context_run_dir / "matched_mmubipred_comparison.json"
    )
    matched_predictions_path = (
        context_run_dir / "matched_mmubipred_predictions.npz"
    )
    context_checkpoint_path = context_run_dir / "best.pt"
    for required in (
        context_metrics_path,
        context_predictions_path,
        matched_comparison_path,
        matched_predictions_path,
        context_checkpoint_path,
    ):
        if not required.exists():
            raise FileNotFoundError(
                "Run the frozen seven-epoch context and exact-MMUbiPred "
                f"matched comparison first; missing {required}"
            )

    context_result = json.loads(
        context_metrics_path.read_text(encoding="utf-8")
    )
    if bool(
        context_result.get("historical_test_informed_exploration", False)
    ):
        raise ValueError(
            "The ensemble must use the frozen primary context checkpoint"
        )
    context_checkpoint_hash = sha256_file(context_checkpoint_path)
    if context_checkpoint_hash != context_result["checkpoint_sha256"]:
        raise ValueError("Frozen context checkpoint checksum changed")
    context_checkpoint = torch.load(
        context_checkpoint_path, map_location="cpu", weights_only=False
    )
    context_metadata = context_checkpoint.get("metadata")
    if not isinstance(context_metadata, dict):
        raise TypeError("Frozen context checkpoint metadata is missing")
    final_context_config = dict(config["final_context_refit"])
    if (
        context_metadata.get("experiment")
        != final_context_config["experiment_name"]
        or int(context_checkpoint.get("refit_epochs", -1))
        != int(final_context_config["epochs"])
        or context_metadata.get("model")
        != config["context_expert"]["model"]
    ):
        raise ValueError(
            "The ensemble requires the frozen seven-epoch context model"
        )

    matched_result = json.loads(
        matched_comparison_path.read_text(encoding="utf-8")
    )
    if not bool(matched_result.get("comparison_valid")):
        raise RuntimeError("Saved exact-MMUbiPred matched comparison is invalid")
    released_model = matched_result.get("released_mmubipred_model")
    if (
        not isinstance(released_model, dict)
        or released_model.get("sha256") != EXPECTED_RELEASED_MODEL_SHA256
    ):
        raise ValueError("Saved predictions are not from the exact released model")
    if (
        matched_result["paper_model_full_test_verification"][
            "confusion_matrix"
        ]
        != EXPECTED_FULL_PAPER_CONFUSION_MATRIX
    ):
        raise RuntimeError("Saved released-model replication changed")

    saved = np.load(matched_predictions_path)
    matched_indices = saved["dataset_indices"].astype(np.int64)
    labels = saved["labels"].astype(np.int64)
    paper_probabilities = saved["paper_probabilities"].astype(np.float64)
    context_probabilities = saved["context_probabilities"].astype(
        np.float64
    )
    expected_indices_hash = str(
        matched_result["matched_test_indices_sha256"]
    )
    if int_array_sha256(matched_indices) != expected_indices_hash:
        raise RuntimeError("Matched test indices changed")

    context_saved = np.load(context_predictions_path)
    if not np.array_equal(
        context_saved["dataset_indices"].astype(np.int64), matched_indices
    ):
        raise RuntimeError("Context and matched test indices differ")
    if not np.array_equal(
        context_saved["labels"].astype(np.int64), labels
    ):
        raise RuntimeError("Context and matched test labels differ")
    if not np.allclose(
        context_saved["probabilities"].astype(np.float64),
        context_probabilities,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("Frozen context probabilities changed")

    data_dir = resolve_project_path(config["data_dir"])
    all_records, preprocessing = load_released_split(
        data_dir, split="test", window_size=49
    )
    all_labels = np.asarray(
        [record.label for record in all_records], dtype=np.int64
    )
    if not np.array_equal(all_labels[matched_indices], labels):
        raise RuntimeError("Released test labels do not match saved predictions")

    ensemble_probabilities = equal_probability_average(
        [paper_probabilities, context_probabilities]
    )
    paper_metrics = compute_metrics(labels, paper_probabilities, threshold)
    context_metrics = compute_metrics(
        labels, context_probabilities, threshold
    )
    ensemble_metrics = compute_metrics(
        labels, ensemble_probabilities, threshold
    )
    assert_metrics_match(
        paper_metrics,
        matched_result["paper_model_matched_test"],
        "exact released MMUbiPred",
    )
    assert_metrics_match(
        context_metrics,
        matched_result["context_model_matched_test"],
        "frozen context",
    )
    ensemble_minus_paper = metric_differences(
        ensemble_metrics, paper_metrics
    )
    ensemble_minus_context = metric_differences(
        ensemble_metrics, context_metrics
    )
    improved_over_paper = [
        metric
        for metric, difference in ensemble_minus_paper.items()
        if difference > 0.0
    ]
    tied_with_paper = [
        metric
        for metric, difference in ensemble_minus_paper.items()
        if difference == 0.0
    ]
    decreased_vs_paper = [
        metric
        for metric, difference in ensemble_minus_paper.items()
        if difference < 0.0
    ]

    results = {
        "status": (
            "valid matched-cohort post-test exploratory exact-MMUbiPred/"
            "context equal-probability ensemble"
        ),
        "comparison_valid": True,
        "historical_test_informed_exploration": True,
        "confirmatory_claim_allowed": False,
        "experiment": ensemble_config["experiment_name"],
        "architecture": {
            "paper_expert": (
                "exact released DeepUBI_AAindex_One_Hot_Emb_drop_out2423.h5 "
                "saved probabilities"
            ),
            "context_expert": (
                "frozen seven-epoch 257-residue LoRA-ESM2 saved probabilities"
            ),
            "fusion": "equal arithmetic mean of positive-class probabilities",
            "paper_weight": 0.5,
            "context_weight": 0.5,
            "trained_fusion_parameters": False,
            "threshold": threshold,
        },
        "matched_test_indices_sha256": expected_indices_hash,
        "matched_support": ensemble_metrics["support"],
        "matched_coverage": matched_result["matched_coverage"],
        "source_artifacts": {
            "released_mmubipred_model": released_model,
            "context_checkpoint": {
                "path": str(context_checkpoint_path),
                "sha256": context_checkpoint_hash,
                "epochs": final_context_config["epochs"],
            },
            "matched_comparison": {
                "path": str(matched_comparison_path),
                "sha256": sha256_file(matched_comparison_path),
            },
            "matched_predictions": {
                "path": str(matched_predictions_path),
                "sha256": sha256_file(matched_predictions_path),
            },
        },
        "released_mmubipred_matched_test": paper_metrics,
        "context_model_matched_test": context_metrics,
        "equal_ensemble_matched_test": ensemble_metrics,
        "ensemble_minus_released_mmubipred": ensemble_minus_paper,
        "ensemble_minus_context": ensemble_minus_context,
        "metric_direction_summary_vs_released_mmubipred": {
            "improved": improved_over_paper,
            "tied": tied_with_paper,
            "decreased": decreased_vs_paper,
            "all_reported_metrics_improved": (
                len(improved_over_paper) == len(COMPARISON_METRICS)
            ),
        },
        "paired_correctness": {
            "paper_vs_context": paired_correctness(
                labels,
                paper_probabilities,
                context_probabilities,
                threshold,
                "paper",
                "context",
            ),
            "paper_vs_equal_ensemble": paired_correctness(
                labels,
                paper_probabilities,
                ensemble_probabilities,
                threshold,
                "paper",
                "equal_ensemble",
            ),
        },
        "released_test_preprocessing": {
            name: vars(report) for name, report in preprocessing.items()
        },
    }
    np.savez_compressed(
        predictions_path,
        dataset_indices=matched_indices,
        labels=labels,
        paper_probabilities=paper_probabilities,
        context_probabilities=context_probabilities,
        equal_ensemble_probabilities=ensemble_probabilities,
    )
    with predictions_table_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "processed_test_index",
                "protein_id",
                "position",
                "label",
                "paper_probability",
                "context_probability",
                "equal_ensemble_probability",
                "equal_ensemble_prediction_at_0.5",
            ]
        )
        for (
            index,
            label,
            paper_probability,
            context_probability,
            ensemble_probability,
        ) in zip(
            matched_indices,
            labels,
            paper_probabilities,
            context_probabilities,
            ensemble_probabilities,
        ):
            record = all_records[int(index)]
            writer.writerow(
                [
                    int(index),
                    record.protein_id,
                    record.position,
                    int(label),
                    float(paper_probability),
                    float(context_probability),
                    float(ensemble_probability),
                    int(ensemble_probability >= threshold),
                ]
            )

    manifest = {
        "experiment": ensemble_config["experiment_name"],
        "status": "post-test exploratory fixed ensemble evaluated",
        "project_git_commit": git_revision(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_test_informed_exploration": True,
        "confirmatory_claim_allowed": False,
        "fusion": results["architecture"],
        "matched_test_indices_sha256": expected_indices_hash,
        "test_labels_used_for_weight_or_threshold_selection": False,
        "new_training_or_model_inference_performed": False,
        "metrics_file": metrics_path.name,
        "predictions_file": predictions_path.name,
        "predictions_table": predictions_table_path.name,
        "source_artifacts": results["source_artifacts"],
    }
    write_json(manifest_path, manifest)
    # Metrics are the one-time completion guard and are written last.
    write_json(metrics_path, results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
