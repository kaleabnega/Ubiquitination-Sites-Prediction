from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "experiment" / "scripts" / "run_validation_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_validation_benchmark", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import {SCRIPT_PATH}")
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)

MERGE_PATH = PROJECT_ROOT / "experiment" / "scripts" / "merge_validation_benchmarks.py"
MERGE_SPEC = importlib.util.spec_from_file_location(
    "merge_validation_benchmarks", MERGE_PATH
)
if MERGE_SPEC is None or MERGE_SPEC.loader is None:
    raise RuntimeError(f"Could not import {MERGE_PATH}")
merge = importlib.util.module_from_spec(MERGE_SPEC)
MERGE_SPEC.loader.exec_module(merge)


def run_record(model: str, seed: int, mcc: float) -> dict[str, object]:
    return {
        "model": model,
        "seed": seed,
        "validation_indices_sha256": f"split-{seed}",
        "collapsed": False,
        "fixed_mcc": mcc,
        "fixed_accuracy": mcc,
        "fixed_sensitivity": mcc,
        "fixed_specificity": mcc,
        "auroc": mcc,
        "auprc": mcc,
        "selected_mcc": mcc,
    }


class BenchmarkTests(unittest.TestCase):
    def test_paired_candidate_difference(self) -> None:
        runs = [
            run_record("baseline", 1, 0.40),
            run_record("candidate", 1, 0.45),
            run_record("baseline", 2, 0.50),
            run_record("candidate", 2, 0.60),
        ]
        benchmark.validate_paired_splits(runs)
        summary = benchmark.summarize("test", "baseline", runs)
        difference = summary["paired_differences"]["candidate"][
            "candidate_minus_reference"
        ]["fixed_mcc"]["mean"]
        self.assertAlmostEqual(difference, 0.075)

    def test_split_mismatch_is_rejected(self) -> None:
        runs = [
            run_record("baseline", 1, 0.40),
            run_record("candidate", 1, 0.45),
        ]
        runs[1]["validation_indices_sha256"] = "different-split"
        with self.assertRaises(RuntimeError):
            benchmark.validate_paired_splits(runs)

    def test_collapsed_run_invalidates_comparison(self) -> None:
        runs = [run_record("baseline", 1, 0.0), run_record("candidate", 1, 0.5)]
        runs[0]["collapsed"] = True
        summary = benchmark.summarize("test", "baseline", runs)
        self.assertFalse(summary["comparison_valid"])
        self.assertEqual(
            summary["collapsed_runs"], [{"model": "baseline", "seed": 1}]
        )

    def test_merge_selects_requested_model(self) -> None:
        payload = {
            "runs": [run_record("baseline", 1, 0.4), run_record("candidate", 1, 0.5)]
        }
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"
            summary_path.write_text(json.dumps(payload), encoding="utf-8")
            selected = merge.selected_runs(summary_path, "candidate")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["model"], "candidate")

    def test_external_output_path_is_supported(self) -> None:
        external = Path("/tmp") / "persistent-benchmark" / "seed_42"
        self.assertEqual(benchmark.portable_output_path(external), str(external))

    def test_repository_output_path_is_relative(self) -> None:
        internal = PROJECT_ROOT / "experiment" / "outputs" / "seed_42"
        self.assertEqual(
            benchmark.portable_output_path(internal),
            "experiment/outputs/seed_42",
        )


if __name__ == "__main__":
    unittest.main()
