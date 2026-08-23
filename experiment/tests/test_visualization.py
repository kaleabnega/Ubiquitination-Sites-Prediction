from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.model import LongContextLoRAESM2, MMUbiPredCompatible  # noqa: E402
from ubipred.visualization import (  # noqa: E402
    expert_correctness_masks,
    positional_log2_enrichment,
    predict_with_representations,
    radial_band_log2_enrichment,
    stratified_visualization_indices,
)


class FakeBackbone(torch.nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TokenDataset(Dataset):
    def __init__(self, count: int = 4) -> None:
        self.tokens = torch.zeros((count, 49), dtype=torch.long)
        self.tokens[:, 24] = 11
        self.labels = torch.tensor(
            [index % 2 for index in range(count)], dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": self.tokens[index],
            "label": self.labels[index],
            "index": torch.tensor(index, dtype=torch.long),
        }


class VisualizationTests(unittest.TestCase):
    def test_local_representation_is_post_relu_fusion_vector(self) -> None:
        model = MMUbiPredCompatible(
            aaindex_lookup=np.zeros((21, 31), dtype=np.float32),
            window_size=49,
        )
        loader = DataLoader(TokenDataset(), batch_size=2, shuffle=False)
        labels, probabilities, indices, hidden = predict_with_representations(
            model, loader, torch.device("cpu")
        )
        self.assertEqual(hidden.shape, (4, 6))
        self.assertTrue(np.all(hidden >= 0.0))
        self.assertEqual(labels.tolist(), [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(indices.tolist(), [0, 1, 2, 3])
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_context_representation_is_penultimate_256_vector(self) -> None:
        model = LongContextLoRAESM2(
            backbone=FakeBackbone(hidden_size=16),
            backbone_hidden_size=16,
            residue_token_lookup=list(range(4, 24)) + [0],
            cls_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            window_size=49,
            classifier_hidden_dim=256,
            dropout=0.0,
            local_radius=24,
        )
        loader = DataLoader(TokenDataset(), batch_size=2, shuffle=False)
        _, _, _, hidden = predict_with_representations(
            model, loader, torch.device("cpu")
        )
        self.assertEqual(hidden.shape, (4, 256))

    def test_visualization_sample_is_balanced_and_reproducible(self) -> None:
        labels = np.asarray([0] * 70 + [1] * 80, dtype=np.int64)
        first = stratified_visualization_indices(labels, 100, seed=42)
        second = stratified_visualization_indices(labels, 100, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 100)
        self.assertEqual(int(np.sum(labels[first] == 0)), 50)
        self.assertEqual(int(np.sum(labels[first] == 1)), 50)

    def test_positional_enrichment_recovers_group_specific_residues(self) -> None:
        sequences = ["AKA", "AKA", "GKG", "GKG"]
        numerator = np.asarray([True, True, False, False])
        denominator = ~numerator
        enrichment = positional_log2_enrichment(
            sequences, numerator, denominator
        )
        amino_acids = tuple("ACDEFGHIKLMNPQRSTVWY")
        self.assertGreater(enrichment[amino_acids.index("A"), 0], 0.0)
        self.assertLess(enrichment[amino_acids.index("G"), 0], 0.0)
        self.assertAlmostEqual(enrichment[amino_acids.index("K"), 1], 0.0)

    def test_radial_enrichment_excludes_center_and_aggregates_bands(self) -> None:
        sequences = ["AAKAA", "AAKAA", "GGKGG", "GGKGG"]
        numerator = np.asarray([True, True, False, False])
        denominator = ~numerator
        enrichment = radial_band_log2_enrichment(
            sequences,
            numerator,
            denominator,
            bands=((1, 1), (2, 2)),
        )
        amino_acids = tuple("ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(enrichment.shape, (20, 2))
        self.assertTrue(np.all(enrichment[amino_acids.index("A")] > 0.0))
        self.assertTrue(np.all(enrichment[amino_acids.index("G")] < 0.0))

    def test_correctness_masks_form_expected_partition(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        local = np.asarray([0.1, 0.9, 0.9, 0.1])
        context = np.asarray([0.1, 0.9, 0.1, 0.9])
        masks = expert_correctness_masks(labels, local, context)
        self.assertEqual(
            {name: np.flatnonzero(mask).tolist() for name, mask in masks.items()},
            {
                "Both correct": [0],
                "Short-Range only correct": [2],
                "Long-Context only correct": [3],
                "Both wrong": [1],
            },
        )


if __name__ == "__main__":
    unittest.main()
