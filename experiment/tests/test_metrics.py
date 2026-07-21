from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.metrics import compute_metrics  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_known_confusion_matrix(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.1, 0.8, 0.7, 0.2])
        metrics = compute_metrics(labels, probabilities, threshold=0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["sensitivity"], 0.5)
        self.assertAlmostEqual(metrics["specificity"], 0.5)


if __name__ == "__main__":
    unittest.main()
