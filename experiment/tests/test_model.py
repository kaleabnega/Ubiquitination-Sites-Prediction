from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.model import UbiFusionNet  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
