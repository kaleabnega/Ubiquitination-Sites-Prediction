#!/usr/bin/env python3
"""Apply the predeclared robust-loss development screening rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_candidate(
    candidate: dict[str, object],
    reference: dict[str, object],
    rule: dict[str, object],
) -> dict[str, object]:
    differences = {
        metric: float(candidate[metric]) - float(reference[metric])
        for metric in (
            "fixed_mcc",
            "fixed_accuracy",
            "fixed_sensitivity",
            "fixed_specificity",
            "auroc",
            "auprc",
            "selected_mcc",
        )
    }
    checks = {
        "not_collapsed": not bool(candidate["collapsed"]),
        "minimum_fixed_mcc_gain": differences["fixed_mcc"]
        >= float(rule["minimum_candidate_minus_bce_fixed_mcc"]),
        "noninferior_auroc": differences["auroc"]
        >= float(rule["minimum_candidate_minus_bce_auroc"]),
        "sensitivity_within_margin": differences["fixed_sensitivity"]
        >= -float(rule["maximum_allowed_sensitivity_decrease"]),
        "specificity_within_margin": differences["fixed_specificity"]
        >= -float(rule["maximum_allowed_specificity_decrease"]),
    }
    return {
        "candidate": candidate["model"],
        "candidate_minus_bce": differences,
        "checks": checks,
        "passed": all(checks.values()),
    }


def screen(
    suite: dict[str, object], summary: dict[str, object]
) -> dict[str, object]:
    if summary["benchmark_name"] != suite["benchmark_name"]:
        raise ValueError("Benchmark summary does not match the suite")
    expected_models = [str(item["name"]) for item in suite["models"]]
    runs = list(summary["runs"])
    if len(runs) != len(expected_models):
        raise ValueError("Robust-loss screen is incomplete")
    by_model = {str(run["model"]): run for run in runs}
    if set(by_model) != set(expected_models):
        raise ValueError("Robust-loss screen model set changed")
    if any(int(run["seed"]) != 42 for run in runs):
        raise ValueError("Robust-loss screen must use seed 42 only")
    if len({str(run["validation_indices_sha256"]) for run in runs}) != 1:
        raise ValueError("Robust-loss candidates used different splits")
    reference_name = str(suite["reference_model"])
    reference = by_model[reference_name]
    rule = dict(suite["screening_rule"])
    if (
        rule["primary_metric"] != "fixed_mcc"
        or rule["winner_rule_if_multiple_pass"] != "highest fixed_mcc"
        or bool(rule["collapsed_run_allowed"])
        or bool(rule["released_independent_or_external_test_access"])
    ):
        raise ValueError("Robust-loss screening contract changed")
    candidates = [
        evaluate_candidate(by_model[name], reference, rule)
        for name in expected_models
        if name != reference_name
    ]
    passing = [item for item in candidates if bool(item["passed"])]
    comparison_valid = bool(summary["comparison_valid"])
    if not comparison_valid and not bool(rule["collapsed_run_allowed"]):
        passing = []
    if passing:
        selected = max(
            passing,
            key=lambda item: float(
                by_model[str(item["candidate"])]["fixed_mcc"]
            ),
        )
        decision = "ADVANCE_" + str(selected["candidate"]).upper()
        selected_candidate: str | None = str(selected["candidate"])
    else:
        decision = "STOP_ROBUST_LOSS"
        selected_candidate = None
    return {
        "status": "development-only robust-loss screen completed",
        "comparison_valid": comparison_valid,
        "released_independent_test_accessed": False,
        "external_test_accessed": False,
        "reference_model": reference_name,
        "screening_rule": rule,
        "candidate_results": candidates,
        "decision": decision,
        "selected_candidate": selected_candidate,
        "next_step": (
            rule["next_step_if_passed"]
            if selected_candidate is not None
            else "retain BCE and stop the label-robustness track"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = resolve_project_path(args.output)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite robust-loss decision: {output_path}"
        )
    result = screen(
        read_json(resolve_project_path(args.suite)),
        read_json(resolve_project_path(args.summary)),
    )
    write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
