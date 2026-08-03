#!/usr/bin/env python3
"""Secondary paired analyses using frozen external predictions only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.external_benchmark import sha256_file  # noqa: E402
from ubipred.ensemble import equal_probability_average  # noqa: E402
from ubipred.metrics import compute_metrics, write_json  # noqa: E402
from ubipred.paired_statistics import (  # noqa: E402
    mcnemar_exact,
    paired_cluster_bootstrap,
)


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


def metric_differences(
    candidate: dict[str, object], reference: dict[str, object]
) -> dict[str, float]:
    return {
        metric: float(candidate[metric]) - float(reference[metric])
        for metric in COMPARISON_METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    cohort_dir = resolve_project_path(args.cohort_dir)
    evaluation_dir = resolve_project_path(args.evaluation_dir)
    lock_path = cohort_dir / "cohort_lock.json"
    cohort_path = cohort_dir / "external_cohort.tsv"
    primary_results_path = evaluation_dir / "external_test_metrics.json"
    predictions_path = evaluation_dir / "external_test_predictions.npz"
    manifest_path = evaluation_dir / "external_test_manifest.json"
    output_path = evaluation_dir / "external_secondary_comparisons.json"
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to repeat completed secondary analysis: {output_path}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    primary_results = json.loads(
        primary_results_path.read_text(encoding="utf-8")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != sha256_file(config_path):
        raise ValueError("Saved evaluation used a different configuration")
    if manifest["cohort_lock_sha256"] != sha256_file(lock_path):
        raise ValueError("Cohort lock changed after external evaluation")
    if manifest["cohort_sha256"] != sha256_file(cohort_path):
        raise ValueError("Frozen external cohort changed")
    if manifest["predictions_sha256"] != sha256_file(predictions_path):
        raise ValueError("Saved external predictions changed")
    if not primary_results.get("comparison_valid", False):
        raise ValueError("Primary external comparison is not valid")
    if lock.get("external_predictions_generated") is not False:
        raise ValueError("Unexpected cohort-lock inference state")

    with cohort_path.open("r", encoding="utf-8", newline="") as handle:
        cohort = list(csv.DictReader(handle, delimiter="\t"))
    saved = np.load(predictions_path, allow_pickle=False)
    labels = np.asarray(saved["labels"], dtype=np.int64)
    paper_probabilities = np.asarray(
        saved["paper_probabilities"], dtype=np.float64
    )
    context_probabilities = np.asarray(
        saved["context_probabilities"], dtype=np.float64
    )
    accessions = np.asarray(saved["canonical_accessions"])
    cohort_labels = np.asarray(
        [int(row["label"]) for row in cohort], dtype=np.int64
    )
    cohort_accessions = np.asarray(
        [row["canonical_accession"] for row in cohort]
    )
    if not (
        np.array_equal(labels, cohort_labels)
        and np.array_equal(accessions, cohort_accessions)
        and paper_probabilities.shape == labels.shape
        and context_probabilities.shape == labels.shape
    ):
        raise ValueError("Saved predictions do not align with the cohort")

    evaluation = dict(config["frozen_evaluation"])
    threshold = float(evaluation["primary_threshold"])
    paper_metrics = compute_metrics(labels, paper_probabilities, threshold)
    context_metrics = compute_metrics(
        labels, context_probabilities, threshold
    )
    if paper_metrics != primary_results["exact_released_mmubipred"]:
        raise ValueError("Recomputed paper metrics differ from saved results")
    if context_metrics != primary_results["frozen_context_model"]:
        raise ValueError("Recomputed context metrics differ from saved results")

    direct_context_was_primary = (
        evaluation.get("primary_candidate")
        == "frozen_seven_epoch_context_model"
    )
    if direct_context_was_primary:
        candidate_name = "equal_probability_mmubipred_context_hybrid"
        candidate_probabilities = equal_probability_average(
            [paper_probabilities, context_probabilities]
        )
        analysis_role = (
            "post-primary exploratory evaluation of the previously defined "
            "50/50 MMUbiPred-context hybrid; it cannot rescue or replace "
            "the failed confirmatory context-versus-MMUbiPred hypothesis"
        )
    else:
        candidate_name = "frozen_context_model"
        candidate_probabilities = context_probabilities
        analysis_role = (
            "predeclared secondary comparator; does not replace the primary "
            "equal-ensemble comparison"
        )
    candidate_metrics = compute_metrics(
        labels, candidate_probabilities, threshold
    )
    bootstrap = paired_cluster_bootstrap(
        labels,
        candidate_probabilities,
        paper_probabilities,
        accessions,
        threshold=threshold,
        replicates=int(evaluation["bootstrap_replicates"]),
        seed=int(evaluation["bootstrap_seed"]),
        confidence_level=float(evaluation["confidence_level"]),
        progress_every=500,
    )
    paired_comparison = {
        "bootstrap": bootstrap,
        "mcnemar_accuracy_descriptive_site_level": {
            **mcnemar_exact(
                labels,
                candidate_probabilities,
                paper_probabilities,
                threshold,
            ),
            "inferential_role": (
                "descriptive only; protein-cluster bootstrap is the "
                "uncertainty analysis"
            ),
        },
    }
    result = {
        "status": "saved-prediction secondary comparison completed",
        "new_model_inference_performed": False,
        "comparison_valid": True,
        "analysis_role": analysis_role,
        "post_primary_exploratory": direct_context_was_primary,
        "confirmatory_claim_allowed": False,
        "multiplicity_note": (
            "intervals and p-values are unadjusted secondary analyses and "
            "must not be presented as a rescued primary claim"
        ),
        "threshold": threshold,
        "support": lock["final_cohort"]["support"],
        "unique_protein_groups": len(np.unique(accessions)),
        "exact_released_mmubipred": paper_metrics,
        "frozen_context_model": context_metrics,
        "secondary_candidate": candidate_name,
        "secondary_candidate_metrics": candidate_metrics,
        "secondary_candidate_minus_exact_mmubipred": metric_differences(
            candidate_metrics, paper_metrics
        ),
        "paired_secondary_candidate_vs_exact_mmubipred": paired_comparison,
        "source_artifacts": {
            "config_sha256": sha256_file(config_path),
            "cohort_lock_sha256": sha256_file(lock_path),
            "cohort_sha256": sha256_file(cohort_path),
            "primary_results_sha256": sha256_file(primary_results_path),
            "predictions_sha256": sha256_file(predictions_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    }
    if direct_context_was_primary:
        result.update(
            {
                "hybrid_architecture": {
                    "fusion": (
                        "equal arithmetic mean of positive-class "
                        "probabilities"
                    ),
                    "paper_weight": 0.5,
                    "context_weight": 0.5,
                    "trained_fusion_parameters": False,
                    "weights_or_threshold_tuned_on_this_cohort": False,
                },
                "equal_probability_hybrid": candidate_metrics,
                "hybrid_minus_exact_mmubipred": metric_differences(
                    candidate_metrics, paper_metrics
                ),
                "paired_hybrid_vs_exact_mmubipred": paired_comparison,
            }
        )
    else:
        result.update(
            {
                "context_minus_exact_mmubipred": metric_differences(
                    context_metrics, paper_metrics
                ),
                "paired_context_vs_exact_mmubipred": paired_comparison,
            }
        )
    write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
