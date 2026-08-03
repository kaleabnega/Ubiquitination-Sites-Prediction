#!/usr/bin/env python3
"""Diagnose whether frozen local and context predictions justify fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.complementarity import analyze_folds, analyze_pair  # noqa: E402


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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def assert_metric_match(
    observed: dict[str, object],
    expected: dict[str, object],
    name: str,
) -> None:
    for metric in (
        "mcc",
        "accuracy",
        "sensitivity",
        "specificity",
        "auroc",
        "auprc",
    ):
        if not np.isclose(
            float(observed[metric]),
            float(expected[metric]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"Saved {name} metric changed: {metric}")


def load_optional_source(name: str, path: Path) -> dict[str, object]:
    saved = np.load(path, allow_pickle=False)
    required = {"labels", "paper_probabilities", "context_probabilities"}
    missing = sorted(required.difference(saved.files))
    if missing:
        raise ValueError(f"{name} is missing prediction arrays: {missing}")
    labels = np.asarray(saved["labels"], dtype=np.int64)
    paper = np.asarray(saved["paper_probabilities"], dtype=np.float64)
    context = np.asarray(saved["context_probabilities"], dtype=np.float64)
    return {
        "name": name,
        "analysis_role": (
            "post-evaluation descriptive saved-prediction analysis; excluded "
            "from architecture, fusion, weight, and threshold selection"
        ),
        "may_change_development_recommendation": False,
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "paper_vs_context": analyze_pair(
            labels,
            paper,
            context,
            threshold=0.5,
            reference_name="exact_released_mmubipred",
            candidate_name="frozen_context",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--saved-prediction-source",
        nargs=2,
        action="append",
        metavar=("NAME", "NPZ_PATH"),
        help=(
            "Optionally add an already-saved historical or external NPZ with "
            "labels, paper_probabilities, and context_probabilities."
        ),
    )
    args = parser.parse_args()

    run_dir = resolve_project_path(args.development_run_dir)
    output_path = resolve_project_path(args.output)
    summary_path = run_dir / "stabilized_summary.json"
    predictions_path = run_dir / "stabilized_oof_predictions.npz"
    for required in (summary_path, predictions_path):
        if not required.exists():
            raise FileNotFoundError(
                "The corrected context-residual OOF run is required; "
                f"missing {required}"
            )
    if output_path.exists():
        raise FileExistsError(
            f"Completed complementarity analysis already exists: {output_path}"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not bool(summary.get("comparison_valid")):
        raise RuntimeError("Corrected development OOF comparison is invalid")
    if summary.get("collapsed_expert_folds"):
        raise RuntimeError("Corrected development run contains a collapsed fold")
    if summary.get("independent_test_accessed") is not False:
        raise RuntimeError("Development summary unexpectedly accessed the test")

    saved = np.load(predictions_path, allow_pickle=False)
    required_arrays = {
        "labels",
        "fold_assignments",
        "local_probabilities",
        "context_probabilities",
        "unweighted_average_probabilities",
        "residual_stack_crossfit_probabilities",
    }
    missing_arrays = sorted(required_arrays.difference(saved.files))
    if missing_arrays:
        raise ValueError(f"Corrected OOF predictions are incomplete: {missing_arrays}")
    labels = np.asarray(saved["labels"], dtype=np.int64)
    folds = np.asarray(saved["fold_assignments"], dtype=np.int64)
    local = np.asarray(saved["local_probabilities"], dtype=np.float64)
    context = np.asarray(saved["context_probabilities"], dtype=np.float64)
    average = np.asarray(
        saved["unweighted_average_probabilities"], dtype=np.float64
    )
    stack = np.asarray(
        saved["residual_stack_crossfit_probabilities"], dtype=np.float64
    )

    local_vs_context = analyze_pair(
        labels,
        local,
        context,
        threshold=0.5,
        reference_name="stabilized_mmubipred_compatible_local",
        candidate_name="long_context_lora_esm2",
    )
    context_vs_stack = analyze_pair(
        labels,
        context,
        stack,
        threshold=0.5,
        reference_name="long_context_lora_esm2",
        candidate_name="crossfit_nonnegative_residual_stack",
    )
    context_vs_average = analyze_pair(
        labels,
        context,
        average,
        threshold=0.5,
        reference_name="long_context_lora_esm2",
        candidate_name="unweighted_probability_average",
    )
    expected_metrics = summary["metrics"]
    assert_metric_match(
        local_vs_context["reference_metrics"],
        expected_metrics["local_expert"]["fixed_threshold"],
        "local expert",
    )
    assert_metric_match(
        local_vs_context["candidate_metrics"],
        expected_metrics["context_expert"]["fixed_threshold"],
        "context expert",
    )
    assert_metric_match(
        context_vs_stack["candidate_metrics"],
        expected_metrics["residual_stack_crossfit"]["fixed_threshold"],
        "residual stack",
    )
    assert_metric_match(
        context_vs_average["candidate_metrics"],
        expected_metrics["unweighted_average"]["fixed_threshold"],
        "unweighted average",
    )

    local_context_folds = analyze_folds(
        labels,
        local,
        context,
        folds,
        threshold=0.5,
        reference_name="stabilized_mmubipred_compatible_local",
        candidate_name="long_context_lora_esm2",
    )
    context_stack_folds = analyze_folds(
        labels,
        context,
        stack,
        folds,
        threshold=0.5,
        reference_name="long_context_lora_esm2",
        candidate_name="crossfit_nonnegative_residual_stack",
    )
    stack_fold_mcc_gains = [
        float(fold["candidate_minus_reference"]["mcc"])
        for fold in context_stack_folds
    ]
    stack_gain = float(context_vs_stack["candidate_minus_reference"]["mcc"])
    frozen_decision = str(summary.get("amended_candidate_decision"))
    actionable_fusion = frozen_decision == "ADVANCE_RESIDUAL_STACK"
    recommendation = (
        "CONSIDER_ONLY_A_PREDECLARED_GATED_FUSION_SCREEN"
        if actionable_fusion
        else "DO_NOT_TRAIN_GATED_FUSION"
    )

    optional_sources: list[dict[str, object]] = []
    for name, raw_path in args.saved_prediction_source or []:
        source_path = resolve_project_path(raw_path)
        if not source_path.exists():
            raise FileNotFoundError(
                f"Optional saved source does not exist: {source_path}"
            )
        optional_sources.append(load_optional_source(name, source_path))

    result = {
        "status": "saved-prediction complementarity analysis completed",
        "analysis_role": (
            "retrospective diagnostic of frozen predictions; no model, fusion "
            "weight, threshold, or architecture was fitted or selected"
        ),
        "new_model_training_or_inference_performed": False,
        "development_comparison_valid": True,
        "development_independent_or_external_test_accessed": False,
        "primary_threshold": 0.5,
        "development": {
            "local_vs_context": local_vs_context,
            "context_vs_crossfit_residual_stack": context_vs_stack,
            "context_vs_unweighted_average": context_vs_average,
            "local_vs_context_by_fold": local_context_folds,
            "context_vs_crossfit_residual_stack_by_fold": context_stack_folds,
        },
        "fusion_diagnostic": {
            "frozen_development_decision": frozen_decision,
            "crossfit_stack_minus_context_mcc": stack_gain,
            "crossfit_stack_mcc_gain_by_fold": stack_fold_mcc_gains,
            "folds_where_stack_mcc_exceeded_context": int(
                np.sum(np.asarray(stack_fold_mcc_gains) > 0.0)
            ),
            "material_low_capacity_fusion_signal": actionable_fusion,
            "recommendation": recommendation,
            "reason": (
                "A flexible gate is justified only after simpler cross-fitted "
                "fusion demonstrates material incremental value over the better "
                "expert. The frozen residual-stack decision is the controlling "
                "development evidence; historical and external diagnostics "
                "cannot reverse it."
            ),
        },
        "optional_post_evaluation_sources": optional_sources,
        "source_artifacts": {
            "development_summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
            "development_predictions": {
                "path": str(predictions_path),
                "sha256": sha256_file(predictions_path),
            },
        },
        "project_git_commit": git_revision(),
    }
    write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
