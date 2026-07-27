from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.context import (  # noqa: E402
    CONTEXT_PADDING_INDEX,
    LongContextSiteDataset,
    build_validated_contexts,
    centered_sequence_window,
)
from ubipred.fasta import SiteRecord  # noqa: E402
from ubipred.stacking import fit_residual_stacker  # noqa: E402


class ContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sequence = "A" * 50 + "K" + "C" * 70
        released = centered_sequence_window(self.sequence, 51, 49)
        self.record = SiteRecord(
            header="P12345|51",
            protein_id="P12345",
            position=51,
            sequence=released,
            label=1,
            source="test.fasta",
        )

    def test_centered_window_pads_sequence_boundaries(self) -> None:
        self.assertEqual(centered_sequence_window("KAA", 1, 5), "--KAA")
        self.assertEqual(centered_sequence_window("AAK", 3, 5), "AAK--")

    def test_validation_requires_released_window_match(self) -> None:
        indices, contexts, report = build_validated_contexts(
            [self.record], {"P12345": self.sequence}, context_window_size=257
        )
        self.assertEqual(indices, [0])
        self.assertEqual(len(contexts[0]), 257)
        self.assertEqual(contexts[0][128], "K")
        self.assertEqual(report["validated_fraction"], 1.0)

        changed = list(self.sequence)
        changed[30] = "C"
        _, _, mismatch = build_validated_contexts(
            [self.record], {"P12345": "".join(changed)}, 257
        )
        self.assertEqual(mismatch["reason_counts"], {"released_window_mismatch": 1})

    def test_long_context_dataset_keeps_unknown_and_padding_distinct(self) -> None:
        context = list(centered_sequence_window(self.sequence, 51, 257))
        context[100] = "X"
        dataset = LongContextSiteDataset([self.record], ["".join(context)])
        self.assertEqual(int(dataset.tokens[0, 100]), 20)
        self.assertEqual(int(dataset.tokens[0, 0]), CONTEXT_PADDING_INDEX)

    def test_residual_stacker_has_nonnegative_weights(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        local = np.asarray([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        context = np.asarray([0.2, 0.3, 0.45, 0.55, 0.7, 0.8])
        stacker = fit_residual_stacker(labels, local, context)
        self.assertGreaterEqual(stacker.local_weight, 0.0)
        self.assertGreaterEqual(stacker.context_weight, 0.0)
        probabilities = stacker.predict_proba(local, context)
        self.assertTrue(np.all((probabilities > 0) & (probabilities < 1)))


if __name__ == "__main__":
    unittest.main()
