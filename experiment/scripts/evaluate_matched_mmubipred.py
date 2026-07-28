#!/usr/bin/env python3
"""Compare released MMUbiPred and context predictions on identical test sites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.data import TOKEN_TO_INDEX  # noqa: E402
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.metrics import compute_metrics, write_json  # noqa: E402


EXPECTED_MODEL_SHA256 = (
    "a9393c05635f8019d08fe916885bdf3f31225859b2d0cb4130ec4c2bd89a5001"
)
EXPECTED_FULL_METRICS = {
    "mcc": 0.5458386140298045,
    "accuracy": 0.7725035719955549,
    "sensitivity": 0.7498020585906572,
    "specificity": 0.8067729083665338,
    "confusion_matrix": [[4050, 970], [1896, 5682]],
}


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


def verify_full_replication(metrics: dict[str, object]) -> None:
    if metrics["confusion_matrix"] != EXPECTED_FULL_METRICS["confusion_matrix"]:
        raise RuntimeError(
            "The released model did not reproduce the expected full-test "
            f"confusion matrix: {metrics['confusion_matrix']}"
        )
    for metric in ("mcc", "accuracy", "sensitivity", "specificity"):
        if not np.isclose(
            float(metrics[metric]),
            float(EXPECTED_FULL_METRICS[metric]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "The released model did not reproduce the expected full-test "
                f"{metric}: {metrics[metric]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--context-run-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "replication" / "MMUbiPred",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model_path = args.model_file.resolve()
    run_dir = args.context_run_dir.resolve()
    data_dir = args.data_dir.resolve()
    output_path = run_dir / "matched_mmubipred_comparison.json"
    output_predictions_path = (
        run_dir / "matched_mmubipred_predictions.npz"
    )
    if output_path.exists():
        raise FileExistsError(
            "Refusing to repeat the completed matched comparison: "
            f"{output_path}"
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Released MMUbiPred model not found: {model_path}")
    actual_model_hash = sha256_file(model_path)
    if actual_model_hash != EXPECTED_MODEL_SHA256:
        raise ValueError(
            "Released MMUbiPred model checksum mismatch: "
            f"{actual_model_hash}"
        )

    context_metrics_path = run_dir / "context_test_metrics.json"
    context_predictions_path = run_dir / "context_test_predictions.npz"
    if not context_metrics_path.exists() or not context_predictions_path.exists():
        raise FileNotFoundError(
            "Run the frozen context-model test evaluation before the matched "
            "MMUbiPred comparison"
        )
    context_result = json.loads(
        context_metrics_path.read_text(encoding="utf-8")
    )
    context_saved = np.load(context_predictions_path)
    matched_indices = context_saved["dataset_indices"].astype(np.int64)
    context_labels = context_saved["labels"].astype(np.int64)
    context_probabilities = context_saved["probabilities"].astype(np.float64)

    records, preprocessing = load_released_split(
        data_dir, split="test", window_size=49
    )
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    tokens = np.asarray(
        [
            [TOKEN_TO_INDEX[residue] for residue in record.sequence]
            for record in records
        ],
        dtype=np.int64,
    )
    if not np.array_equal(labels[matched_indices], context_labels):
        raise RuntimeError(
            "Context prediction labels do not match the released test order"
        )
    expected_indices_hash = str(
        context_result["validated_test_indices_sha256"]
    )
    if int_array_sha256(matched_indices) != expected_indices_hash:
        raise RuntimeError("Matched test indices do not match the context report")

    # Preserve the released notebook's float64 pandas-style normalization
    # before TensorFlow casts model inputs internally.
    aaindex_raw = np.loadtxt(data_dir / "aaindex31.txt", dtype=np.float64)
    aaindex_min = aaindex_raw.min(axis=1, keepdims=True)
    aaindex_lookup = (
        (aaindex_raw - aaindex_min)
        / (aaindex_raw.max(axis=1, keepdims=True) - aaindex_min)
    ).T
    aaindex_lookup = np.concatenate(
        [aaindex_lookup, np.zeros((1, 31), dtype=np.float64)],
        axis=0,
    )
    aaindex_inputs = aaindex_lookup[tokens]
    one_hot_inputs = np.eye(21, dtype=np.float64)[tokens]

    # Import legacy Keras only inside this fresh subprocess. This avoids
    # contaminating the PyTorch/Transformers process used for the context model.
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    try:
        import tf_keras
    except ImportError as error:
        raise ImportError(
            "Install tf-keras==2.19.0 before running this comparison"
        ) from error

    model = tf_keras.models.load_model(model_path, compile=False)
    paper_outputs = model.predict(
        [aaindex_inputs, one_hot_inputs, tokens],
        batch_size=args.batch_size,
        verbose=1,
    )
    if paper_outputs.ndim != 2 or paper_outputs.shape != (len(records), 2):
        raise RuntimeError(
            f"Unexpected MMUbiPred output shape: {paper_outputs.shape}"
        )
    paper_probabilities = np.asarray(
        paper_outputs[:, 1], dtype=np.float64
    )
    paper_full_metrics = compute_metrics(
        labels, paper_probabilities, threshold=0.5
    )
    verify_full_replication(paper_full_metrics)

    paper_matched_probabilities = paper_probabilities[matched_indices]
    paper_matched_metrics = compute_metrics(
        context_labels, paper_matched_probabilities, threshold=0.5
    )
    context_matched_metrics = compute_metrics(
        context_labels, context_probabilities, threshold=0.5
    )
    if context_matched_metrics != context_result["fixed_threshold"]:
        raise RuntimeError(
            "Recomputed context metrics do not match the frozen result"
        )
    comparison_metrics = (
        "mcc",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "auroc",
        "auprc",
    )
    differences = {
        metric: (
            float(context_matched_metrics[metric])
            - float(paper_matched_metrics[metric])
        )
        for metric in comparison_metrics
    }
    is_test_informed_exploration = bool(
        context_result.get("historical_test_informed_exploration", False)
    )
    results = {
        "status": (
            "valid matched-cohort post-test exploratory comparison"
            if is_test_informed_exploration
            else "valid matched-cohort historical-test comparison"
        ),
        "comparison_valid": True,
        "historical_test_informed_exploration": (
            is_test_informed_exploration
        ),
        "confirmatory_claim_allowed": not is_test_informed_exploration,
        "primary_threshold": 0.5,
        "matched_test_indices_sha256": expected_indices_hash,
        "matched_support": context_matched_metrics["support"],
        "matched_coverage": {
            "retained": len(matched_indices),
            "processed_full_test": len(records),
            "fraction": len(matched_indices) / len(records),
        },
        "released_mmubipred_model": {
            "path": str(model_path),
            "sha256": actual_model_hash,
        },
        "released_test_preprocessing": {
            name: vars(report) for name, report in preprocessing.items()
        },
        "paper_model_full_test_verification": paper_full_metrics,
        "paper_model_matched_test": paper_matched_metrics,
        "context_model_matched_test": context_matched_metrics,
        "context_minus_paper_matched": differences,
        "context_checkpoint_sha256": context_result["checkpoint_sha256"],
        "context_predictions_sha256": sha256_file(
            context_predictions_path
        ),
    }
    np.savez_compressed(
        output_predictions_path,
        dataset_indices=matched_indices,
        labels=context_labels,
        paper_probabilities=paper_matched_probabilities,
        context_probabilities=context_probabilities,
    )
    # The JSON file is the one-time completion guard and is written last.
    write_json(output_path, results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
