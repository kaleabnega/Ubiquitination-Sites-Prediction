from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.model import MMUbiPredCompatible, UbiFusionNet  # noqa: E402


class ModelTests(unittest.TestCase):
    def test_output_and_gate_shapes(self) -> None:
        model = UbiFusionNet(
            aaindex_lookup=np.zeros((21, 31), dtype=np.float32),
            window_size=49,
            embedding_dim=32,
            branch_dim=64,
            conv_channels=8,
            transformer_heads=4,
            transformer_layers=1,
            transformer_ff_dim=64,
            dropout=0.0,
        )
        tokens = torch.zeros((3, 49), dtype=torch.long)
        tokens[:, 24] = 11  # K in ARNDCQEGHILKMFPSTWYV-
        logits, gates = model(tokens, return_gates=True)
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertEqual(tuple(gates.shape), (3, 3))
        torch.testing.assert_close(gates.sum(dim=1), torch.ones(3))

    def test_no_context_ablation_has_zero_context_gate(self) -> None:
        model = UbiFusionNet(
            aaindex_lookup=np.zeros((21, 31), dtype=np.float32),
            window_size=49,
            embedding_dim=32,
            branch_dim=64,
            conv_channels=8,
            transformer_heads=4,
            transformer_layers=1,
            transformer_ff_dim=64,
            dropout=0.0,
            active_branches=("one_hot", "aaindex"),
        )
        tokens = torch.zeros((2, 49), dtype=torch.long)
        tokens[:, 24] = 11
        _, gates = model(tokens, return_gates=True)
        torch.testing.assert_close(gates[:, 0], torch.zeros(2))
        torch.testing.assert_close(gates.sum(dim=1), torch.ones(2))

    def test_mmubipred_compatible_outputs(self) -> None:
        model = MMUbiPredCompatible(
            aaindex_lookup=np.zeros((21, 31), dtype=np.float32),
            window_size=49,
        )
        tokens = torch.zeros((2, 49), dtype=torch.long)
        tokens[:, 24] = 11
        logits, branch_probabilities = model(tokens, return_gates=True)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(branch_probabilities.shape), (2, 3))
        self.assertTrue(torch.all(branch_probabilities >= 0))
        self.assertTrue(torch.all(branch_probabilities <= 1))
        regularization = model.regularization_loss()
        self.assertGreater(float(regularization.detach()), 0.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor([0.0, 1.0])
        ) + regularization
        loss.backward()
        self.assertIsNotNone(model.fusion_output.weight.grad)


if __name__ == "__main__":
    unittest.main()
