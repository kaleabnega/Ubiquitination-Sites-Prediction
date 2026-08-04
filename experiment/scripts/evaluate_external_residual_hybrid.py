#!/usr/bin/env python3
"""Evaluate the frozen residual hybrid on an existing external cohort."""

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

from ubipred.data import (  # noqa: E402
    SiteDataset,
    load_normalized_aaindex,
    make_loader,
)
from ubipred.engine import load_checkpoint_model_state, predict  # noqa: E402
from ubipred.fasta import SiteRecord  # noqa: E402
from ubipred.metrics import compute_metrics, write_json  # noqa: E402
from ubipred.model import build_model  # noqa: E402
from ubipred.paired_statistics import (  # noqa: E402
    mcnemar_exact,
    paired_cluster_bootstrap,
)
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


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def load_cohort(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "benchmark_index",
        "identifier",
        "canonical_accession",
        "position_one_based",
        "label",
        "window_49",
    }
    if not records or not required.issubset(records[0]):
        raise ValueError("Frozen external cohort is empty or incomplete")
    return records


def metric_differences(
    candidate: dict[str, object], reference: dict[str, object]
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


def require_hash(path: Path, expected: str, role: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"Checksum mismatch for {role}: {path}")


def validate_analysis_contract(config: dict[str, object]) -> None:
    if (
        float(config["threshold"]) != 0.5
        or int(config["bootstrap_replicates"]) != 10000
        or config["bootstrap_unit"] != "canonical_accession"
        or float(config["confidence_level"]) != 0.95
        or bool(config["fusion_weights_or_threshold_may_be_changed"])
        or bool(config["external_labels_may_be_used_for_selection"])
        or bool(config["confirmatory_claim_allowed"])
    ):
        raise ValueError("External residual-hybrid analysis contract changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--cohort-key", required=True)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--source-evaluation-dir", type=Path, required=True)
    parser.add_argument("--development-run-dir", type=Path, required=True)
    parser.add_argument("--hybrid-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", default="replication/MMUbiPred")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--allow-post-primary-exploration",
        action="store_true",
        help="Acknowledge reuse of previously inspected external labels.",
    )
    args = parser.parse_args()
    if not args.allow_post_primary_exploration:
        parser.error(
            "Refusing external residual evaluation without "
            "--allow-post-primary-exploration"
        )

    analysis_config_path = resolve_project_path(args.analysis_config)
    analysis_config = json.loads(
        analysis_config_path.read_text(encoding="utf-8")
    )
    validate_analysis_contract(analysis_config)
    cohort_specs = analysis_config.get("cohorts")
    if not isinstance(cohort_specs, dict) or args.cohort_key not in cohort_specs:
        raise ValueError(f"Unknown frozen cohort key: {args.cohort_key}")
    cohort_spec = dict(cohort_specs[args.cohort_key])

    architecture_config_path = resolve_project_path(
        str(analysis_config["architecture_config"])
    )
    architecture_config = json.loads(
        architecture_config_path.read_text(encoding="utf-8")
    )
    cohort_dir = resolve_project_path(args.cohort_dir)
    source_evaluation_dir = resolve_project_path(args.source_evaluation_dir)
    development_run_dir = resolve_project_path(args.development_run_dir)
    hybrid_run_dir = resolve_project_path(args.hybrid_run_dir)
    output_dir = resolve_project_path(args.output_dir)
    data_dir = resolve_project_path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "residual_hybrid_external_metrics.json"
    predictions_path = output_dir / "residual_hybrid_external_predictions.npz"
    table_path = output_dir / "residual_hybrid_external_predictions.tsv"
    manifest_path = output_dir / "residual_hybrid_external_manifest.json"
    if results_path.exists():
        raise FileExistsError(
            f"Refusing to repeat completed exploratory evaluation: {results_path}"
        )

    cohort_path = cohort_dir / "external_cohort.tsv"
    cohort_lock_path = cohort_dir / "cohort_lock.json"
    source_results_path = source_evaluation_dir / "external_test_metrics.json"
    source_predictions_path = (
        source_evaluation_dir / "external_test_predictions.npz"
    )
    source_manifest_path = (
        source_evaluation_dir / "external_test_manifest.json"
    )
    for required in (
        cohort_path,
        cohort_lock_path,
        source_results_path,
        source_predictions_path,
        source_manifest_path,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Missing frozen external artifact: {required}")

    cohort_lock = json.loads(cohort_lock_path.read_text(encoding="utf-8"))
    source_results = json.loads(source_results_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not bool(source_results.get("comparison_valid")):
        raise RuntimeError("Source external comparison is invalid")
    require_hash(
        cohort_path,
        str(cohort_lock["final_cohort"]["sha256"]),
        "frozen cohort",
    )
    require_hash(
        cohort_lock_path,
        str(source_manifest["cohort_lock_sha256"]),
        "cohort lock used by source inference",
    )
    require_hash(
        cohort_path,
        str(source_manifest["cohort_sha256"]),
        "cohort used by source inference",
    )
    require_hash(
        source_predictions_path,
        str(source_manifest["predictions_sha256"]),
        "source external predictions",
    )

    cohort = load_cohort(cohort_path)
    labels_expected = np.asarray(
        [int(row["label"]) for row in cohort], dtype=np.int64
    )
    accessions_expected = np.asarray(
        [row["canonical_accession"] for row in cohort]
    )
    indices_expected = np.asarray(
        [int(row["benchmark_index"]) for row in cohort], dtype=np.int64
    )
    negative_count = int(np.sum(labels_expected == 0))
    positive_count = int(np.sum(labels_expected == 1))
    unique_accessions = len(set(accessions_expected.tolist()))
    observed_contract = (
        len(cohort),
        negative_count,
        positive_count,
        unique_accessions,
    )
    expected_contract = (
        int(cohort_spec["expected_sites"]),
        int(cohort_spec["expected_negative_sites"]),
        int(cohort_spec["expected_positive_sites"]),
        int(cohort_spec["expected_unique_accessions"]),
    )
    if observed_contract != expected_contract:
        raise ValueError(
            f"External cohort contract changed: {observed_contract} != "
            f"{expected_contract}"
        )

    with np.load(source_predictions_path, allow_pickle=False) as source:
        required_arrays = {
            "benchmark_indices",
            "labels",
            "canonical_accessions",
            "paper_probabilities",
            "context_probabilities",
        }
        if not required_arrays.issubset(source.files):
            raise ValueError("Source external prediction arrays are incomplete")
        benchmark_indices = source["benchmark_indices"].astype(np.int64)
        labels = source["labels"].astype(np.int64)
        canonical_accessions = source["canonical_accessions"].astype(str)
        paper_probabilities = source["paper_probabilities"].astype(np.float64)
        context_probabilities = source["context_probabilities"].astype(np.float64)
    if (
        not np.array_equal(benchmark_indices, indices_expected)
        or not np.array_equal(labels, labels_expected)
        or not np.array_equal(canonical_accessions, accessions_expected)
    ):
        raise RuntimeError("Saved external predictions do not align with the cohort")

    threshold = float(analysis_config["threshold"])
    paper_metrics = compute_metrics(labels, paper_probabilities, threshold)
    context_metrics = compute_metrics(labels, context_probabilities, threshold)
    assert_metrics_match(
        paper_metrics,
        source_results["exact_released_mmubipred"],
        "exact released MMUbiPred",
    )
    assert_metrics_match(
        context_metrics,
        source_results["frozen_context_model"],
        "frozen context model",
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
            raise FileNotFoundError(f"Missing residual-hybrid artifact: {required}")
    local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
    local_summary = json.loads(local_summary_path.read_text(encoding="utf-8"))
    stabilized_summary = json.loads(
        stabilized_summary_path.read_text(encoding="utf-8")
    )
    if (
        local_manifest.get("candidate") != "mmubipred_context_residual_hybrid"
        or local_summary.get("candidate") != "mmubipred_context_residual_hybrid"
        or not bool(stabilized_summary.get("comparison_valid"))
    ):
        raise ValueError("Residual-hybrid provenance is invalid")
    local_checkpoint = torch.load(
        local_checkpoint_path, map_location="cpu", weights_only=False
    )
    local_metadata = local_checkpoint.get("metadata")
    if (
        not isinstance(local_metadata, dict)
        or local_checkpoint.get("state_dict_scope") != "trainable_parameters"
    ):
        raise ValueError("Local residual-hybrid checkpoint is invalid")
    hybrid_config = dict(architecture_config["exploratory_residual_refit"])
    if local_metadata.get("experiment") != hybrid_config["experiment_name"]:
        raise ValueError("Local residual-hybrid checkpoint identity changed")
    require_hash(
        stabilized_summary_path,
        str(local_metadata["stabilized_summary_sha256"]),
        "stabilized development summary",
    )
    stacker_values = stabilized_summary.get("final_stacker_for_future_refit")
    if (
        not isinstance(stacker_values, dict)
        or stacker_values != local_metadata.get("final_stacker")
        or stacker_values != local_summary.get("final_stacker")
    ):
        raise ValueError("Frozen residual stacker changed")
    if (
        str(local_metadata["context_checkpoint_sha256"])
        != str(source_results["source_artifacts"]["context_checkpoint_sha256"])
    ):
        raise ValueError("External context predictions use a different checkpoint")
    stacker = ResidualStacker(
        intercept=float(stacker_values["intercept"]),
        local_weight=float(stacker_values["local_weight"]),
        context_weight=float(stacker_values["context_weight"]),
        l2_strength=float(stacker_values["l2_strength"]),
    )

    site_records = [
        SiteRecord(
            header=row["identifier"],
            protein_id=row["canonical_accession"],
            position=int(row["position_one_based"]) - 1,
            sequence=row["window_49"],
            label=int(row["label"]),
            source=cohort_path.name,
        )
        for row in cohort
    ]
    local_dataset = SiteDataset(site_records)
    local_loader = make_loader(
        local_dataset,
        indices=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=0,
    )
    local_model_config = dict(architecture_config["local_expert"]["model"])
    if local_metadata.get("model") != local_model_config:
        raise ValueError("Local expert architecture configuration changed")
    local_model = build_model(
        aaindex_lookup=load_normalized_aaindex(data_dir / "aaindex31.txt"),
        window_size=49,
        model_config=local_model_config,
    )
    load_checkpoint_model_state(local_model, local_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model.to(device)
    print(
        f"device={device} cohort={args.cohort_key} sites={len(site_records)} "
        "frozen local inference started",
        flush=True,
    )
    local_labels, local_probabilities, local_indices, diagnostics = predict(
        local_model, local_loader, device
    )
    order = np.argsort(local_indices)
    local_indices = local_indices[order].astype(np.int64)
    local_labels = local_labels[order].astype(np.int64)
    local_probabilities = local_probabilities[order].astype(np.float64)
    diagnostics = diagnostics[order]
    if (
        not np.array_equal(local_indices, np.arange(len(site_records)))
        or not np.array_equal(local_labels, labels)
    ):
        raise RuntimeError("Local residual expert predictions are misaligned")

    residual_probabilities = stacker.predict_proba(
        local_probabilities, context_probabilities
    )
    local_metrics = compute_metrics(labels, local_probabilities, threshold)
    residual_metrics = compute_metrics(labels, residual_probabilities, threshold)
    bootstrap = paired_cluster_bootstrap(
        labels,
        residual_probabilities,
        paper_probabilities,
        canonical_accessions,
        threshold=threshold,
        replicates=int(analysis_config["bootstrap_replicates"]),
        seed=int(cohort_spec["bootstrap_seed"]),
        confidence_level=float(analysis_config["confidence_level"]),
        progress_every=500,
    )
    results = {
        "status": "post-primary exploratory external residual-hybrid evaluation completed",
        "analysis_role": analysis_config["analysis_role"],
        "cohort_key": args.cohort_key,
        "comparison_valid": True,
        "post_primary_exploratory": True,
        "confirmatory_claim_allowed": False,
        "external_labels_used_for_model_weight_or_threshold_selection": False,
        "cohort": cohort_lock["final_cohort"],
        "architecture": {
            "local_expert": "frozen five-epoch MMUbiPred-compatible 49-residue refit",
            "context_expert": "frozen seven-epoch 257-residue LoRA-ESM2 saved probabilities",
            "fusion": "frozen nonnegative residual logit stacker",
            "stacker": stacker_values,
            "threshold": threshold,
        },
        "exact_released_mmubipred": paper_metrics,
        "frozen_context_model": context_metrics,
        "frozen_local_expert": local_metrics,
        "residual_hybrid": residual_metrics,
        "residual_hybrid_minus_exact_mmubipred": metric_differences(
            residual_metrics, paper_metrics
        ),
        "residual_hybrid_minus_context": metric_differences(
            residual_metrics, context_metrics
        ),
        "paired_residual_hybrid_vs_exact_mmubipred": {
            "bootstrap": bootstrap,
            "mcnemar_accuracy_descriptive_site_level": {
                **mcnemar_exact(
                    labels,
                    residual_probabilities,
                    paper_probabilities,
                    threshold,
                ),
                "inferential_role": (
                    "descriptive only; protein-cluster bootstrap is the "
                    "uncertainty analysis"
                ),
            },
        },
        "local_diagnostics": {
            "name": str(
                getattr(local_model, "diagnostic_name", "branch_diagnostics")
            ),
            "mean": diagnostics.mean(axis=0).tolist(),
        },
        "source_artifacts": {
            "analysis_config_sha256": sha256_file(analysis_config_path),
            "architecture_config_sha256": sha256_file(architecture_config_path),
            "cohort_lock_sha256": sha256_file(cohort_lock_path),
            "source_external_results_sha256": sha256_file(source_results_path),
            "source_external_predictions_sha256": sha256_file(
                source_predictions_path
            ),
            "local_checkpoint_sha256": sha256_file(local_checkpoint_path),
            "stabilized_summary_sha256": sha256_file(stabilized_summary_path),
            "context_checkpoint_sha256": local_metadata[
                "context_checkpoint_sha256"
            ],
        },
    }
    np.savez_compressed(
        predictions_path,
        benchmark_indices=benchmark_indices,
        labels=labels,
        canonical_accessions=canonical_accessions,
        paper_probabilities=paper_probabilities,
        context_probabilities=context_probabilities,
        local_probabilities=local_probabilities,
        residual_probabilities=residual_probabilities,
        local_diagnostics=diagnostics,
    )
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "benchmark_index",
                "identifier",
                "canonical_accession",
                "position_one_based",
                "label",
                "paper_probability",
                "context_probability",
                "local_probability",
                "residual_probability",
                "residual_prediction_at_0.5",
            ]
        )
        for row, label, paper, context, local, residual in zip(
            cohort,
            labels,
            paper_probabilities,
            context_probabilities,
            local_probabilities,
            residual_probabilities,
        ):
            writer.writerow(
                [
                    row["benchmark_index"],
                    row["identifier"],
                    row["canonical_accession"],
                    row["position_one_based"],
                    int(label),
                    float(paper),
                    float(context),
                    float(local),
                    float(residual),
                    int(residual >= threshold),
                ]
            )
    manifest = {
        "status": "post-primary exploratory evaluation completed once",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": git_revision(),
        "cohort_key": args.cohort_key,
        "external_labels_used_for_selection": False,
        "results_file": results_path.name,
        "predictions_sha256": sha256_file(predictions_path),
        "prediction_table_sha256": sha256_file(table_path),
        "source_artifacts": results["source_artifacts"],
    }
    write_json(manifest_path, manifest)
    write_json(results_path, results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
