from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "scripts"))

from analyze_robust_loss_screen import screen  # noqa: E402
from ubipred.losses import (  # noqa: E402
    AsymmetricNegativeLabelSmoothingLoss,
    BinaryGeneralizedCrossEntropyLoss,
    build_binary_loss,
)


class RobustLossTests(unittest.TestCase):
    def test_bce_factory_matches_pytorch(self) -> None:
        logits = torch.tensor([-1.0, 0.5, 2.0])
        labels = torch.tensor([0.0, 1.0, 0.0])
        observed = build_binary_loss({"name": "bce"})(logits, labels)
        expected = F.binary_cross_entropy_with_logits(logits, labels)
        torch.testing.assert_close(observed, expected)

    def test_asymmetric_smoothing_changes_only_negative_targets(self) -> None:
        logits = torch.tensor([-0.5, 1.5])
        labels = torch.tensor([0.0, 1.0])
        observed = AsymmetricNegativeLabelSmoothingLoss(0.05)(
            logits, labels
        )
        expected = F.binary_cross_entropy_with_logits(
            logits, torch.tensor([0.05, 1.0])
        )
        torch.testing.assert_close(observed, expected)

    def test_gce_suppresses_gradient_from_confidently_wrong_example(self) -> None:
        bce_logit = torch.tensor([8.0], requires_grad=True)
        gce_logit = torch.tensor([8.0], requires_grad=True)
        label = torch.tensor([0.0])
        F.binary_cross_entropy_with_logits(bce_logit, label).backward()
        BinaryGeneralizedCrossEntropyLoss(0.7)(gce_logit, label).backward()
        self.assertLess(
            abs(float(gce_logit.grad)), abs(float(bce_logit.grad))
        )

    def test_loss_configurations_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            build_binary_loss({"name": "bce", "q": 0.7})
        with self.assertRaises(ValueError):
            build_binary_loss(
                {
                    "name": "asymmetric_negative_label_smoothing",
                    "negative_target": 0.5,
                }
            )
        with self.assertRaises(ValueError):
            build_binary_loss(
                {"name": "generalized_cross_entropy", "q": 0.0}
            )

    def test_predeclared_screen_advances_only_material_gain(self) -> None:
        suite = {
            "benchmark_name": "screen",
            "reference_model": "bce",
            "models": [
                {"name": "bce"},
                {"name": "smooth"},
                {"name": "gce"},
            ],
            "screening_rule": {
                "primary_metric": "fixed_mcc",
                "minimum_candidate_minus_bce_fixed_mcc": 0.01,
                "minimum_candidate_minus_bce_auroc": 0.0,
                "maximum_allowed_sensitivity_decrease": 0.03,
                "maximum_allowed_specificity_decrease": 0.03,
                "collapsed_run_allowed": False,
                "winner_rule_if_multiple_pass": "highest fixed_mcc",
                "released_independent_or_external_test_access": False,
                "next_step_if_passed": "confirm",
            },
        }

        def run(
            name: str,
            mcc: float,
            auroc: float,
            sensitivity: float,
            specificity: float,
        ) -> dict[str, object]:
            return {
                "model": name,
                "seed": 42,
                "validation_indices_sha256": "same",
                "collapsed": False,
                "fixed_mcc": mcc,
                "fixed_accuracy": 0.75,
                "fixed_sensitivity": sensitivity,
                "fixed_specificity": specificity,
                "auroc": auroc,
                "auprc": 0.85,
                "selected_mcc": mcc,
            }

        summary = {
            "benchmark_name": "screen",
            "comparison_valid": True,
            "runs": [
                run("bce", 0.55, 0.86, 0.70, 0.82),
                run("smooth", 0.565, 0.861, 0.68, 0.83),
                run("gce", 0.555, 0.87, 0.72, 0.82),
            ],
        }
        result = screen(suite, summary)
        self.assertEqual(result["decision"], "ADVANCE_SMOOTH")
        self.assertEqual(result["selected_candidate"], "smooth")


if __name__ == "__main__":
    unittest.main()
