#!/usr/bin/env python3
"""Evaluate the frozen OOF residual fusion as a post-test exploration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.data import (  # noqa: E402
    SiteDataset,
    load_normalized_aaindex,
    make_loader,
)
from ubipred.engine import load_checkpoint_model_state, predict  # noqa: E402
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.metrics import compute_metrics, write_json  # noqa: E402
from ubipred.model import build_model  # noqa: E402
from ubipred.stacking import ResidualStacker  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-run-dir", type=Path)
    parser.add_argument("--hybrid-run-dir", type=Path)
    parser.add_argument("--context-run-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--allow-post-test-exploration",
        action="store_true",
        help="Acknowledge that this is not a confirmatory test.",
    )
    parser.add_argument(
        "--allow-locked-test",
        action="store_true",
        help="Acknowledge use of the released historical test.",
    )
    args = parser.parse_args()
    if not args.allow_post_test_exploration:
        parser.error(
            "Residual evaluation requires --allow-post-test-exploration"
        )
    if not args.allow_locked_test:
        parser.error("Residual evaluation requires --allow-locked-test")

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hybrid_config = dict(config["exploratory_residual_refit"])
    development_run_dir = resolve_project_path(
        args.development_run_dir or str(config["output_dir"])
    )
    hybrid_run_dir = resolve_project_path(
        args.hybrid_run_dir or str(hybrid_config["output_dir"])
    )
    context_run_dir = resolve_project_path(
        args.context_run_dir or str(hybrid_config["context_run_dir"])
    )
    output_path = hybrid_run_dir / "residual_hybrid_test_metrics.json"
    output_predictions_path = (
        hybrid_run_dir / "residual_hybrid_test_predictions.npz"
    )
    output_table_path = hybrid_run_dir / "residual_hybrid_test_predictions.tsv"
    if output_path.exists():
        raise FileExistsError(
            "Refusing to repeat the completed hybrid test evaluation: "
            f"{output_path}"
        )

    local_checkpoint_path = hybrid_run_dir / "best.pt"
    local_manifest_path = hybrid_run_dir / "refit_manifest.json"
    local_summary_path = hybrid_run_dir / "refit_summary.json"
    stabilized_summary_path = development_run_dir / "stabilized_summary.json"
    for required in (
        local_checkpoint_path,
        local_manifest_path,
        local_summary_path,
        stabilized_summary_path,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Missing hybrid artifact: {required}")

    local_manifest = json.loads(
        local_manifest_path.read_text(encoding="utf-8")
    )
    if local_manifest.get("candidate") != "mmubipred_context_residual_hybrid":
        raise ValueError("Refit manifest is not for the residual hybrid")
    if not bool(local_manifest.get("historical_test_informed_exploration")):
        raise ValueError("Hybrid manifest is not marked exploratory")
    local_checkpoint = torch.load(
        local_checkpoint_path, map_location="cpu", weights_only=False
    )
    local_metadata = local_checkpoint.get("metadata")
    if not isinstance(local_metadata, dict):
        raise TypeError("Local checkpoint metadata is missing")
    if local_checkpoint.get("state_dict_scope") != "trainable_parameters":
        raise ValueError("Expected compact local-expert checkpoint")
    if (
        local_metadata.get("experiment")
        != hybrid_config["experiment_name"]
    ):
        raise ValueError("Local checkpoint experiment identity changed")

    stabilized_summary = json.loads(
        stabilized_summary_path.read_text(encoding="utf-8")
    )
    if not bool(stabilized_summary.get("comparison_valid")):
        raise RuntimeError("Corrected development comparison is invalid")
    if (
        sha256_file(stabilized_summary_path)
        != local_metadata["stabilized_summary_sha256"]
    ):
        raise ValueError("Corrected development summary changed after refit")
    stacker_values = stabilized_summary.get(
        "final_stacker_for_future_refit"
    )
    if not isinstance(stacker_values, dict):
        raise TypeError("Corrected summary is missing the final stacker")
    if stacker_values != local_metadata.get("final_stacker"):
        raise ValueError("Frozen residual stacker changed after local refit")
    stacker = ResidualStacker(
        intercept=float(stacker_values["intercept"]),
        local_weight=float(stacker_values["local_weight"]),
        context_weight=float(stacker_values["context_weight"]),
        l2_strength=float(stacker_values["l2_strength"]),
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
                "Run the frozen seven-epoch context and matched-MMUbiPred "
                f"evaluation first; missing {required}"
            )
    context_metrics_result = json.loads(
        context_metrics_path.read_text(encoding="utf-8")
    )
    # Metrics produced before the epoch-15 exploration was added do not have
    # this flag. Missing therefore means the legacy frozen run, not an
    # exploratory run. The checkpoint hash, experiment identity, and exact
    # seven-epoch duration are verified immediately below.
    if bool(
        context_metrics_result.get(
            "historical_test_informed_exploration", False
        )
    ):
        raise ValueError(
            "Residual hybrid must use the frozen seven-epoch context model"
        )
    if (
        sha256_file(context_checkpoint_path)
        != context_metrics_result["checkpoint_sha256"]
    ):
        raise ValueError("Frozen context checkpoint checksum changed")
    if (
        context_metrics_result["checkpoint_sha256"]
        != local_metadata.get("context_checkpoint_sha256")
    ):
        raise ValueError(
            "Context checkpoint differs from the one frozen at local refit"
        )
    context_checkpoint = torch.load(
        context_checkpoint_path, map_location="cpu", weights_only=False
    )
    final_context_config = dict(config["final_context_refit"])
    if (
        context_checkpoint.get("metadata", {}).get("experiment")
        != final_context_config["experiment_name"]
        or int(context_checkpoint.get("refit_epochs", -1))
        != int(final_context_config["epochs"])
    ):
        raise ValueError("Context component is not the frozen primary refit")

    matched_result = json.loads(
        matched_comparison_path.read_text(encoding="utf-8")
    )
    if not bool(matched_result.get("comparison_valid")):
        raise RuntimeError("Saved matched MMUbiPred comparison is invalid")
    matched_saved = np.load(matched_predictions_path)
    matched_indices = matched_saved["dataset_indices"].astype(np.int64)
    labels = matched_saved["labels"].astype(np.int64)
    paper_probabilities = matched_saved["paper_probabilities"].astype(
        np.float64
    )
    context_probabilities = matched_saved[
        "context_probabilities"
    ].astype(np.float64)
    if int_array_sha256(matched_indices) != str(
        matched_result["matched_test_indices_sha256"]
    ):
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
        raise RuntimeError("Saved context probabilities changed")

    data_dir = resolve_project_path(config["data_dir"])
    all_records, preprocessing = load_released_split(
        data_dir, split="test", window_size=49
    )
    all_labels = np.asarray(
        [record.label for record in all_records], dtype=np.int64
    )
    if not np.array_equal(all_labels[matched_indices], labels):
        raise RuntimeError("Released test labels do not match saved predictions")
    matched_records = [all_records[index] for index in matched_indices]
    local_dataset = SiteDataset(matched_records)
    local_loader = make_loader(
        local_dataset,
        indices=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=0,
    )
    local_model_config = dict(config["local_expert"]["model"])
    if local_metadata.get("model") != local_model_config:
        raise ValueError("Local model configuration changed")
    local_model = build_model(
        aaindex_lookup=load_normalized_aaindex(data_dir / "aaindex31.txt"),
        window_size=49,
        model_config=local_model_config,
    )
    load_checkpoint_model_state(local_model, local_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model.to(device)
    print(
        f"device={device} matched_test_records={len(matched_records)}",
        flush=True,
    )
    (
        local_labels,
        local_probabilities,
        local_indices,
        local_diagnostics,
    ) = predict(local_model, local_loader, device)
    order = np.argsort(local_indices)
    local_labels = local_labels[order].astype(np.int64)
    local_probabilities = local_probabilities[order].astype(np.float64)
    local_indices = local_indices[order].astype(np.int64)
    local_diagnostics = local_diagnostics[order]
    if not np.array_equal(
        local_indices, np.arange(len(matched_records), dtype=np.int64)
    ):
        raise RuntimeError("Local expert prediction order is incomplete")
    if not np.array_equal(local_labels, labels):
        raise RuntimeError("Local expert labels do not match the cohort")

    residual_probabilities = stacker.predict_proba(
        local_probabilities, context_probabilities
    )
    threshold = float(hybrid_config["reporting_threshold"])
    if threshold != 0.5:
        raise ValueError("The frozen primary reporting threshold must be 0.5")
    paper_metrics = compute_metrics(labels, paper_probabilities, threshold)
    context_metrics = compute_metrics(
        labels, context_probabilities, threshold
    )
    local_metrics = compute_metrics(labels, local_probabilities, threshold)
    residual_metrics = compute_metrics(
        labels, residual_probabilities, threshold
    )
    assert_metrics_match(
        paper_metrics,
        matched_result["paper_model_matched_test"],
        "released MMUbiPred",
    )
    assert_metrics_match(
        context_metrics,
        matched_result["context_model_matched_test"],
        "frozen context",
    )

    results = {
        "status": (
            "valid matched-cohort post-test exploratory residual-hybrid "
            "comparison"
        ),
        "comparison_valid": True,
        "historical_test_informed_exploration": True,
        "confirmatory_claim_allowed": False,
        "primary_threshold": threshold,
        "matched_test_indices_sha256": int_array_sha256(matched_indices),
        "matched_support": residual_metrics["support"],
        "matched_coverage": matched_result["matched_coverage"],
        "architecture": {
            "local_expert": "PyTorch MMUbiPred-compatible 49-residue model",
            "context_expert": (
                "frozen seven-epoch 257-residue LoRA-ESM2 model"
            ),
            "fusion": (
                "nonnegative residual logit stacker fitted only to corrected "
                "training OOF predictions"
            ),
            "stacker": stacker_values,
        },
        "development_evidence": {
            "candidate_decision": stabilized_summary[
                "amended_candidate_decision"
            ],
            "residual_minus_context_fixed_threshold": stabilized_summary[
                "stack_minus_context_fixed_threshold"
            ],
            "independent_test_accessed": stabilized_summary[
                "independent_test_accessed"
            ],
        },
        "local_refit": {
            "checkpoint": str(local_checkpoint_path),
            "checkpoint_sha256": sha256_file(local_checkpoint_path),
            "seed": local_metadata["seed"],
            "epochs": local_metadata["refit_epochs"],
            "validated_training_records": local_metadata[
                "validated_training_records"
            ],
        },
        "context_refit": {
            "checkpoint": str(context_checkpoint_path),
            "checkpoint_sha256": context_metrics_result[
                "checkpoint_sha256"
            ],
            "epochs": final_context_config["epochs"],
        },
        "released_mmubipred_matched_test": paper_metrics,
        "local_expert_matched_test": local_metrics,
        "context_expert_matched_test": context_metrics,
        "residual_hybrid_matched_test": residual_metrics,
        "residual_minus_released_mmubipred": metric_differences(
            residual_metrics, paper_metrics
        ),
        "residual_minus_local_expert": metric_differences(
            residual_metrics, local_metrics
        ),
        "residual_minus_context_expert": metric_differences(
            residual_metrics, context_metrics
        ),
        "released_test_preprocessing": {
            name: vars(report) for name, report in preprocessing.items()
        },
    }
    np.savez_compressed(
        output_predictions_path,
        dataset_indices=matched_indices,
        labels=labels,
        paper_probabilities=paper_probabilities,
        local_probabilities=local_probabilities,
        context_probabilities=context_probabilities,
        residual_probabilities=residual_probabilities,
        local_branch_diagnostics=local_diagnostics,
    )
    with output_table_path.open(
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
                "local_probability",
                "context_probability",
                "residual_probability",
                "residual_prediction_at_0.5",
            ]
        )
        for (
            index,
            label,
            paper_probability,
            local_probability,
            context_probability,
            residual_probability,
        ) in zip(
            matched_indices,
            labels,
            paper_probabilities,
            local_probabilities,
            context_probabilities,
            residual_probabilities,
        ):
            record = all_records[int(index)]
            writer.writerow(
                [
                    int(index),
                    record.protein_id,
                    record.position,
                    int(label),
                    float(paper_probability),
                    float(local_probability),
                    float(context_probability),
                    float(residual_probability),
                    int(residual_probability >= threshold),
                ]
            )

    local_manifest["independent_test_accessed"] = True
    local_manifest["historical_test_evaluation"] = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post-test exploratory residual-hybrid comparison",
        "confirmatory_claim_allowed": False,
        "matched_test_indices_sha256": results[
            "matched_test_indices_sha256"
        ],
        "primary_threshold": threshold,
        "metrics_file": output_path.name,
        "predictions_file": output_predictions_path.name,
        "predictions_table": output_table_path.name,
    }
    write_json(local_manifest_path, local_manifest)
    # The JSON file is the completion guard and is written only after every
    # supporting artifact and the manifest are complete.
    write_json(output_path, results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
