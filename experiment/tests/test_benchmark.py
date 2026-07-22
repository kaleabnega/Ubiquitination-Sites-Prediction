from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "experiment" / "scripts" / "run_validation_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_validation_benchmark", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import {SCRIPT_PATH}")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def run_record(model: str, seed: int, mcc: float) -> dict[str, object]:
    return {
        "model": model,
        "seed": seed,
        "validation_indices_sha256": f"split-{seed}",
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


if __name__ == "__main__":
    unittest.main()
