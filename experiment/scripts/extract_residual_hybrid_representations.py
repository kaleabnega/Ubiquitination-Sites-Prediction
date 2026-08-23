#!/usr/bin/env python3
"""Extract frozen hidden representations for post hoc visualization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from ubipred.data import (  # noqa: E402
    SiteDataset,
    load_normalized_aaindex,
    make_loader,
)
from ubipred.engine import load_checkpoint_model_state  # noqa: E402
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.metrics import write_json  # noqa: E402
from ubipred.model import build_model  # noqa: E402
from ubipred.visualization import predict_with_representations  # noqa: E402


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


def aligned(
    labels: np.ndarray,
    probabilities: np.ndarray,
    indices: np.ndarray,
    expected_labels: np.ndarray,
    expected_probabilities: np.ndarray,
    expected_indices: np.ndarray,
    name: str,
) -> np.ndarray:
    order = np.argsort(indices)
    labels = labels[order].astype(np.int64)
    probabilities = probabilities[order].astype(np.float64)
    indices = indices[order].astype(np.int64)
    if not np.array_equal(indices, expected_indices):
        raise RuntimeError(f"{name} representation indices are misaligned")
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError(f"{name} representation labels are misaligned")
    if not np.allclose(
        probabilities,
        expected_probabilities,
        rtol=1e-6,
        atol=1e-6,
    ):
        maximum = float(np.max(np.abs(probabilities - expected_probabilities)))
        raise RuntimeError(
            f"{name} frozen probabilities changed (maximum error {maximum:.3g})"
        )
    return order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hybrid-run-dir", type=Path, required=True)
    parser.add_argument("--context-run-dir", type=Path, required=True)
    parser.add_argument("--test-sequence-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--allow-historical-test-visualization",
        action="store_true",
        help="Acknowledge post hoc feature extraction on an inspected cohort.",
    )
    args = parser.parse_args()
    if not args.allow_historical_test_visualization:
        parser.error(
            "Representation extraction requires "
            "--allow-historical-test-visualization"
        )

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hybrid_run_dir = resolve_project_path(args.hybrid_run_dir)
    context_run_dir = resolve_project_path(args.context_run_dir)
    sequence_cache_path = resolve_project_path(args.test_sequence_cache)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    representations_path = output_dir / "matched_hidden_representations.npz"
    manifest_path = output_dir / "representation_manifest.json"
    if representations_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing visualization artifact"
        )

    local_checkpoint_path = hybrid_run_dir / "best.pt"
    hybrid_metrics_path = hybrid_run_dir / "residual_hybrid_test_metrics.json"
    hybrid_predictions_path = (
        hybrid_run_dir / "residual_hybrid_test_predictions.npz"
    )
    context_checkpoint_path = context_run_dir / "best.pt"
    context_metrics_path = context_run_dir / "context_test_metrics.json"
    matched_comparison_path = (
        context_run_dir / "matched_mmubipred_comparison.json"
    )
    required_paths = (
        config_path,
        local_checkpoint_path,
        hybrid_metrics_path,
        hybrid_predictions_path,
        context_checkpoint_path,
        context_metrics_path,
        matched_comparison_path,
        sequence_cache_path,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen visualization input: {path}")

    hybrid_metrics = json.loads(hybrid_metrics_path.read_text(encoding="utf-8"))
    context_metrics = json.loads(
        context_metrics_path.read_text(encoding="utf-8")
    )
    matched_comparison = json.loads(
        matched_comparison_path.read_text(encoding="utf-8")
    )
    if (
        not bool(hybrid_metrics.get("comparison_valid"))
        or not bool(matched_comparison.get("comparison_valid"))
    ):
        raise RuntimeError("The matched residual-hybrid comparison is invalid")
    if bool(context_metrics.get("historical_test_informed_exploration", False)):
        raise ValueError("The visualization requires the frozen epoch-7 context run")
    if sha256_file(context_checkpoint_path) != str(
        context_metrics["checkpoint_sha256"]
    ):
        raise ValueError(
            "The frozen Long-Context Expert checkpoint checksum changed"
        )
    if sha256_file(local_checkpoint_path) != str(
        hybrid_metrics["local_refit"]["checkpoint_sha256"]
    ):
        raise ValueError("The frozen Short-Range Expert checkpoint checksum changed")
    if sequence_cache_sha256(sequence_cache_path) != str(
        context_metrics["test_context_cache"]["sha256"]
    ):
        raise ValueError("The historical-test sequence cache checksum changed")

    with np.load(hybrid_predictions_path, allow_pickle=False) as saved:
        required_arrays = {
            "dataset_indices",
            "labels",
            "paper_probabilities",
            "local_probabilities",
            "context_probabilities",
            "residual_probabilities",
        }
        if not required_arrays.issubset(saved.files):
            raise ValueError("Saved residual-hybrid predictions are incomplete")
        matched_indices = saved["dataset_indices"].astype(np.int64)
        labels = saved["labels"].astype(np.int64)
        paper_probabilities = saved["paper_probabilities"].astype(np.float64)
        local_probabilities = saved["local_probabilities"].astype(np.float64)
        context_probabilities = saved["context_probabilities"].astype(np.float64)
        residual_probabilities = saved["residual_probabilities"].astype(np.float64)
    expected_index_hash = str(hybrid_metrics["matched_test_indices_sha256"])
    if int_array_sha256(matched_indices) != expected_index_hash:
        raise RuntimeError("Matched historical-test indices changed")
    if expected_index_hash != str(
        matched_comparison["matched_test_indices_sha256"]
    ):
        raise RuntimeError("Matched comparison and hybrid index hashes differ")

    data_dir = resolve_project_path(config["data_dir"])
    all_records, _ = load_released_split(data_dir, split="test", window_size=49)
    all_labels = np.asarray([record.label for record in all_records], dtype=np.int64)
    if not np.array_equal(all_labels[matched_indices], labels):
        raise RuntimeError("Saved labels differ from the released test records")
    matched_records = [all_records[int(index)] for index in matched_indices]

    local_checkpoint = torch.load(
        local_checkpoint_path, map_location="cpu", weights_only=False
    )
    local_metadata = local_checkpoint.get("metadata")
    local_model_config = dict(config["local_expert"]["model"])
    if (
        not isinstance(local_metadata, dict)
        or local_checkpoint.get("state_dict_scope") != "trainable_parameters"
        or local_metadata.get("model") != local_model_config
    ):
        raise ValueError("The frozen Short-Range Expert checkpoint is incompatible")
    local_model = build_model(
        aaindex_lookup=load_normalized_aaindex(data_dir / "aaindex31.txt"),
        window_size=49,
        model_config=local_model_config,
    )
    load_checkpoint_model_state(local_model, local_checkpoint)
    local_loader = make_loader(
        SiteDataset(matched_records),
        indices=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=0,
    )

    sequences, _ = load_sequence_cache(sequence_cache_path)
    validated_indices, contexts, context_report = build_validated_contexts(
        all_records,
        sequences,
        context_window_size=int(config["context_window_size"]),
    )
    context_indices = np.asarray(validated_indices, dtype=np.int64)
    if not np.array_equal(context_indices, matched_indices):
        raise RuntimeError("Reconstructed context cohort differs from matched indices")
    context_checkpoint = torch.load(
        context_checkpoint_path, map_location="cpu", weights_only=False
    )
    context_metadata = context_checkpoint.get("metadata")
    context_model_config = dict(config["context_expert"]["model"])
    final_context = dict(config["final_context_refit"])
    if (
        not isinstance(context_metadata, dict)
        or context_checkpoint.get("state_dict_scope") != "trainable_parameters"
        or context_metadata.get("model") != context_model_config
        or context_metadata.get("experiment") != final_context["experiment_name"]
        or int(context_checkpoint.get("refit_epochs", -1))
        != int(final_context["epochs"])
    ):
        raise ValueError("The frozen Long-Context Expert checkpoint is incompatible")
    context_model = build_model(
        aaindex_lookup=load_normalized_aaindex(data_dir / "aaindex31.txt"),
        window_size=int(config["context_window_size"]),
        model_config=context_model_config,
    )
    load_checkpoint_model_state(context_model, context_checkpoint)
    context_dataset = LongContextSiteDataset(
        matched_records,
        contexts,
        original_indices=matched_indices,
    )
    context_loader = make_loader(
        context_dataset,
        indices=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} matched_records={len(labels)}", flush=True)
    local_model.to(device)
    (
        observed_local_labels,
        observed_local_probabilities,
        observed_local_indices,
        local_representations,
    ) = predict_with_representations(local_model, local_loader, device)
    local_order = aligned(
        observed_local_labels,
        observed_local_probabilities,
        observed_local_indices,
        labels,
        local_probabilities,
        np.arange(len(labels), dtype=np.int64),
        "Short-Range Expert",
    )
    local_representations = local_representations[local_order].astype(np.float32)
    local_model.to("cpu")
    del local_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    context_model.to(device)
    (
        observed_context_labels,
        observed_context_probabilities,
        observed_context_indices,
        context_representations,
    ) = predict_with_representations(context_model, context_loader, device)
    context_order = aligned(
        observed_context_labels,
        observed_context_probabilities,
        observed_context_indices,
        labels,
        context_probabilities,
        matched_indices,
        "Long-Context Expert",
    )
    context_representations = context_representations[context_order].astype(
        np.float32
    )
    if local_representations.shape != (len(labels), 6):
        raise RuntimeError("Unexpected Short-Range representation shape")
    if context_representations.shape != (len(labels), 256):
        raise RuntimeError("Unexpected Long-Context representation shape")

    np.savez_compressed(
        representations_path,
        dataset_indices=matched_indices,
        labels=labels,
        paper_probabilities=paper_probabilities,
        local_probabilities=local_probabilities,
        context_probabilities=context_probabilities,
        residual_probabilities=residual_probabilities,
        local_representations=local_representations,
        context_representations=context_representations,
    )
    manifest = {
        "status": "frozen matched-cohort representations extracted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": (
            "post hoc visualization only; no training, model selection, "
            "threshold selection, or metric reevaluation"
        ),
        "cohort": {
            "records": len(labels),
            "negative": int(np.sum(labels == 0)),
            "positive": int(np.sum(labels == 1)),
            "matched_indices_sha256": expected_index_hash,
        },
        "representations": {
            "local": "ReLU output of the six-unit Short-Range fusion layer",
            "local_shape": list(local_representations.shape),
            "context": (
                "256-dimensional Long-Context classifier penultimate GELU output"
            ),
            "context_shape": list(context_representations.shape),
        },
        "context_validation": context_report,
        "source_artifacts": {
            "config_sha256": sha256_file(config_path),
            "local_checkpoint_sha256": sha256_file(local_checkpoint_path),
            "context_checkpoint_sha256": sha256_file(context_checkpoint_path),
            "test_sequence_cache_sha256": sequence_cache_sha256(
                sequence_cache_path
            ),
            "hybrid_metrics_sha256": sha256_file(hybrid_metrics_path),
            "hybrid_predictions_sha256": sha256_file(hybrid_predictions_path),
        },
        "output": {
            "path": str(representations_path),
            "sha256": sha256_file(representations_path),
        },
        "historical_test_accessed": True,
        "confirmatory_claim_allowed": False,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
