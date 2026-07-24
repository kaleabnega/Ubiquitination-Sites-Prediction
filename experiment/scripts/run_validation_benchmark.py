#!/usr/bin/env python3
"""Run paired, development-only model comparisons across fixed seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
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


def mean_and_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def portable_output_path(output_dir: Path) -> str:
    """Prefer a repository-relative path but support persistent external roots."""

    try:
        return str(output_dir.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(output_dir)


def flatten_run(
    model_name: str,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    metrics = read_json(output_dir / "validation_metrics.json")
    manifest = read_json(output_dir / "run_manifest.json")
    fixed = metrics["fixed_threshold"]
    selected = metrics["validation_selected_threshold"]
    development = metrics["development_selection"]
    split = manifest["development_split"]
    prediction_diagnostics = metrics.get("prediction_diagnostics", {})
    collapsed = bool(
        prediction_diagnostics.get(
            "single_class_predictions_at_0_5",
            (float(fixed["sensitivity"]) == 1.0 and float(fixed["specificity"]) == 0.0)
            or (
                float(fixed["sensitivity"]) == 0.0
                and float(fixed["specificity"]) == 1.0
            ),
        )
    )
    return {
        "model": model_name,
        "seed": seed,
        "output_dir": portable_output_path(output_dir),
        "validation_indices_sha256": split["validation_indices_sha256"],
        "best_epoch": development["best_epoch"],
        "collapsed": collapsed,
        "fixed_mcc": fixed["mcc"],
        "fixed_accuracy": fixed["accuracy"],
        "fixed_sensitivity": fixed["sensitivity"],
        "fixed_specificity": fixed["specificity"],
        "auroc": fixed["auroc"],
        "auprc": fixed["auprc"],
        "selected_threshold": selected["threshold"],
        "selected_mcc": selected["mcc"],
    }


def summarize(
    benchmark_name: str,
    reference_model: str,
    runs: list[dict[str, object]],
) -> dict[str, object]:
    metric_names = (
        "fixed_mcc",
        "fixed_accuracy",
        "fixed_sensitivity",
        "fixed_specificity",
        "auroc",
        "auprc",
        "selected_mcc",
    )
    models = sorted({str(run["model"]) for run in runs})
    aggregates: dict[str, object] = {}
    for model_name in models:
        model_runs = [run for run in runs if run["model"] == model_name]
        aggregates[model_name] = {
            "seeds": [int(run["seed"]) for run in model_runs],
            "metrics": {
                metric: mean_and_std(
                    [float(run[metric]) for run in model_runs]
                )
                for metric in metric_names
            },
        }

    by_model_seed = {
        (str(run["model"]), int(run["seed"])): run for run in runs
    }
    paired_differences: dict[str, object] = {}
    for model_name in models:
        if model_name == reference_model:
            continue
        common_seeds = sorted(
            seed
            for candidate, seed in by_model_seed
            if candidate == model_name
            and (reference_model, seed) in by_model_seed
        )
        if not common_seeds:
            continue
        paired_differences[model_name] = {
            "reference_model": reference_model,
            "seeds": common_seeds,
            "candidate_minus_reference": {
                metric: mean_and_std(
                    [
                        float(by_model_seed[(model_name, seed)][metric])
                        - float(by_model_seed[(reference_model, seed)][metric])
                        for seed in common_seeds
                    ]
                )
                for metric in metric_names
            },
        }

    return {
        "benchmark_name": benchmark_name,
        "evaluation_scope": "released training data only; independent test locked",
        "reference_model": reference_model,
        "completed_runs": len(runs),
        "collapsed_runs": [
            {"model": run["model"], "seed": run["seed"]}
            for run in runs
            if bool(run.get("collapsed", False))
        ],
        "comparison_valid": not any(
            bool(run.get("collapsed", False)) for run in runs
        ),
        "runs": runs,
        "aggregates": aggregates,
        "paired_differences": paired_differences,
    }


def validate_paired_splits(runs: list[dict[str, object]]) -> None:
    hashes_by_seed: dict[int, set[str]] = {}
    for run in runs:
        hashes_by_seed.setdefault(int(run["seed"]), set()).add(
            str(run["validation_indices_sha256"])
        )
    mismatches = {
        seed: hashes for seed, hashes in hashes_by_seed.items() if len(hashes) > 1
    }
    if mismatches:
        raise RuntimeError(f"Models used different validation splits: {mismatches}")


def write_tsv(path: Path, runs: list[dict[str, object]]) -> None:
    if not runs:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(runs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a completed run directory instead of retraining it.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override the suite output root, including with a Google Drive path.",
    )
    args = parser.parse_args()

    suite_path = resolve_project_path(args.suite)
    suite = read_json(suite_path)
    benchmark_name = str(suite["benchmark_name"])
    reference_model = str(suite["reference_model"])
    seeds = [int(seed) for seed in suite["seeds"]]
    models = list(suite["models"])
    output_root = (
        resolve_project_path(args.output_root)
        if args.output_root is not None
        else resolve_project_path(
            suite.get(
                "output_root",
                f"experiment/outputs/benchmarks/{benchmark_name}",
            )
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []

    for seed in seeds:
        for model_specification in models:
            model_name = str(model_specification["name"])
            config_path = resolve_project_path(model_specification["config"])
            output_dir = output_root / model_name / f"seed_{seed}"
            completed = (
                (output_dir / "validation_metrics.json").exists()
                and (output_dir / "run_manifest.json").exists()
            )
            if completed and args.resume:
                print(f"resume model={model_name} seed={seed}", flush=True)
            else:
                print(f"run model={model_name} seed={seed}", flush=True)
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        str(PROJECT_ROOT / "experiment" / "scripts" / "train.py"),
                        "--config",
                        str(config_path),
                        "--seed",
                        str(seed),
                        "--output-dir",
                        str(output_dir),
                        "--development-only",
                    ],
                    cwd=PROJECT_ROOT,
                    check=True,
                )

            runs.append(flatten_run(model_name, seed, output_dir))
            validate_paired_splits(runs)
            summary = summarize(benchmark_name, reference_model, runs)
            write_json(output_root / "summary.json", summary)
            write_tsv(output_root / "runs.tsv", runs)
            print(
                f"completed={len(runs)} "
                f"model={model_name} seed={seed} "
                f"mcc={float(runs[-1]['fixed_mcc']):.5f}",
                flush=True,
            )

    print(json.dumps(summarize(benchmark_name, reference_model, runs), indent=2))


if __name__ == "__main__":
    main()
