#!/usr/bin/env python3
"""Combine selected model runs from two development benchmark summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_validation_benchmark import (
    read_json,
    summarize,
    validate_paired_splits,
    write_json,
    write_tsv,
)


def selected_runs(
    summary_path: Path,
    model_name: str,
) -> list[dict[str, object]]:
    summary = read_json(summary_path)
    runs = [run for run in summary["runs"] if run["model"] == model_name]
    if not runs:
        raise ValueError(f"No runs for {model_name!r} in {summary_path}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reference_runs = selected_runs(args.reference_summary, args.reference_model)
    candidate_runs = selected_runs(args.candidate_summary, args.candidate_model)
    runs = sorted(
        reference_runs + candidate_runs,
        key=lambda run: (int(run["seed"]), str(run["model"])),
    )
    reference_seeds = {int(run["seed"]) for run in reference_runs}
    candidate_seeds = {int(run["seed"]) for run in candidate_runs}
    if reference_seeds != candidate_seeds:
        raise ValueError(
            f"Seed mismatch: reference={sorted(reference_seeds)} "
            f"candidate={sorted(candidate_seeds)}"
        )
    validate_paired_splits(runs)
    summary = summarize(args.benchmark_name, args.reference_model, runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    write_tsv(args.output_dir / "runs.tsv", runs)
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
