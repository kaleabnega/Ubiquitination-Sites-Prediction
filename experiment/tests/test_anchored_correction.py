import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ubipred.anchored_correction import AnchoredCorrection, fit_anchored_correction


class AnchoredCorrectionTests(unittest.TestCase):
    def test_zero_parameters_preserve_baseline(self):
        baseline = np.array([0.0, 0.1, 0.7, 1.0])
        correction = AnchoredCorrection(0.0, 0.0, 0.1)
        np.testing.assert_allclose(
            correction.predict_proba(baseline, np.full(4, 0.5)),
            np.clip(baseline, 1e-6, 1 - 1e-6),
        )

    def test_context_coefficient_is_bounded(self):
        labels = np.tile([0, 1], 50)
        baseline = np.full(100, 0.5)
        context = np.where(labels == 1, 0.95, 0.05)
        fitted = fit_anchored_correction(labels, baseline, context)
        self.assertGreater(fitted.context_weight, 0)
        self.assertLessEqual(fitted.context_weight, 1)
        self.assertGreater(np.mean(fitted.predict_proba(baseline, context)[labels == 1]), 0.5)

    def test_intercept_only_has_no_context_effect(self):
        labels = np.tile([0, 1], 20)
        baseline = np.where(labels == 1, 0.8, 0.2)
        correction = fit_anchored_correction(labels, baseline, 1 - baseline,
                                             allow_context=False)
        self.assertEqual(correction.context_weight, 0.0)
        np.testing.assert_allclose(
            correction.predict_proba(baseline, baseline),
            correction.predict_proba(baseline, 1 - baseline),
        )

    def test_rejects_misaligned_or_invalid_inputs(self):
        with self.assertRaises(ValueError):
            fit_anchored_correction(np.array([0, 1]), np.array([0.2]), np.array([0.2, 0.8]))
        with self.assertRaises(ValueError):
            AnchoredCorrection(0, 0, 0.1).predict_proba(np.array([1.2]), np.array([0.5]))


if __name__ == "__main__":
    unittest.main()
