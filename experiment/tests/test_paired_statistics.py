from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.paired_statistics import (  # noqa: E402
    mcnemar_exact,
    paired_stratified_bootstrap,
)


class PairedStatisticsTests(unittest.TestCase):
    def test_identical_predictions_have_zero_differences(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        probabilities = np.asarray([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])
        result = paired_stratified_bootstrap(
            labels,
            probabilities,
            probabilities,
            threshold=0.5,
            replicates=20,
            seed=7,
            confidence_level=0.95,
        )
        for metric in result["metrics"].values():
            self.assertEqual(metric["difference"], 0.0)
            self.assertEqual(metric["confidence_interval"], [0.0, 0.0])
            self.assertEqual(
                metric["paired_bootstrap_two_sided_p_value"], 1.0
            )

    def test_mcnemar_counts_paired_disagreements(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        candidate = np.asarray([0.1, 0.1, 0.9, 0.9])
        reference = np.asarray([0.8, 0.1, 0.2, 0.9])
        result = mcnemar_exact(labels, candidate, reference, 0.5)
        self.assertEqual(result["candidate_only_correct"], 2)
        self.assertEqual(result["reference_only_correct"], 0)
        self.assertEqual(result["discordant_total"], 2)


if __name__ == "__main__":
    unittest.main()
