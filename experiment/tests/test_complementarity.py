from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.complementarity import analyze_folds, analyze_pair  # noqa: E402


class ComplementarityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        self.reference = np.asarray([0.1, 0.9, 0.8, 0.2])
        self.candidate = np.asarray([0.2, 0.1, 0.4, 0.9])

    def test_reports_unique_corrections_and_oracle_ceiling(self) -> None:
        result = analyze_pair(
            self.labels,
            self.reference,
            self.candidate,
            reference_name="local",
            candidate_name="context",
        )

        paired = result["paired_correctness"]
        self.assertEqual(paired["both_correct"], 1)
        self.assertEqual(paired["reference_only_correct"], 1)
        self.assertEqual(paired["candidate_only_correct"], 2)
        self.assertEqual(paired["both_wrong"], 0)
        self.assertAlmostEqual(paired["disagreement_fraction"], 0.75)
        oracle = result["label_informed_oracle_upper_bound"]
        self.assertEqual(oracle["accuracy"], 1.0)
        self.assertAlmostEqual(oracle["gain_over_best_component_accuracy"], 0.25)
        self.assertFalse(oracle["deployable"])

    def test_class_conditioning_distinguishes_error_types(self) -> None:
        result = analyze_pair(self.labels, self.reference, self.candidate)
        negative = result["class_conditioned_paired_correctness"]["negative"]
        positive = result["class_conditioned_paired_correctness"]["positive"]

        self.assertEqual(negative["candidate_only_correct"], 1)
        self.assertEqual(negative["reference_only_correct"], 0)
        self.assertEqual(positive["candidate_only_correct"], 1)
        self.assertEqual(positive["reference_only_correct"], 1)

    def test_fold_analysis_keeps_each_held_out_partition_separate(self) -> None:
        folds = np.asarray([0, 1, 0, 1], dtype=np.int64)
        results = analyze_folds(
            self.labels,
            self.reference,
            self.candidate,
            folds,
        )

        self.assertEqual([result["fold"] for result in results], [0, 1])
        self.assertEqual(
            [result["support"]["total"] for result in results], [2, 2]
        )

    def test_rejects_invalid_or_misaligned_vectors(self) -> None:
        with self.assertRaises(ValueError):
            analyze_pair(self.labels, self.reference[:-1], self.candidate)
        with self.assertRaises(ValueError):
            analyze_pair(self.labels, self.reference, np.asarray([0.1, 0.2, 0.3, 2.0]))


if __name__ == "__main__":
    unittest.main()
