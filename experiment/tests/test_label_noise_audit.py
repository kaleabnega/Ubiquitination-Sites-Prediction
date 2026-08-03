from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "scripts"))

from audit_label_noise import (  # noqa: E402
    conflicting_key_summary,
    cross_source_disagreement,
    negative_to_positive_proximity,
    protein_label_composition,
)
from ubipred.external_benchmark import ExternalSite  # noqa: E402
from ubipred.fasta import SiteRecord  # noqa: E402


def window(residue: str) -> str:
    return residue * 24 + "K" + residue * 24


class LabelNoiseAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            SiteRecord("P1|10", "P1", 10, window("A"), 0, "negative"),
            SiteRecord("P1|20", "P1", 20, window("C"), 1, "positive"),
            SiteRecord("P2|30", "P2", 30, window("D"), 0, "negative"),
        ]

    def test_internal_composition_and_proximity(self) -> None:
        composition = protein_label_composition(self.records)
        self.assertEqual(composition["mixed_label_proteins"], 1)
        self.assertEqual(
            composition["negative_records_on_mixed_label_proteins"], 1
        )
        proximity = negative_to_positive_proximity(self.records)
        self.assertEqual(
            proximity["negative_records_with_positive_on_same_protein"], 1
        )
        self.assertEqual(
            proximity["nearest_positive_distance_cumulative"]["at_most_10"],
            1,
        )

    def test_conflicting_site_and_cross_source_direction(self) -> None:
        conflicting = self.records + [
            SiteRecord("P1|10", "P1", 10, window("A"), 1, "positive")
        ]
        summary = conflicting_key_summary(
            conflicting, lambda record: (record.protein_id, record.position)
        )
        self.assertEqual(summary["conflicting_keys"], 1)

        external_site = ExternalSite(
            benchmark_index=0,
            identifier="P1_HUMAN_11",
            source_protein_id="P1_HUMAN",
            accession_hint="P1",
            species="HUMAN",
            position=11,
            window_21="A" * 10 + "K" + "A" * 10,
            label=1,
        )
        disagreement = cross_source_disagreement(
            self.records,
            [(external_site, "P1", window("A"))],
            training_label=0,
            external_label=1,
        )
        self.assertEqual(disagreement["exact_site_matches"], 1)
        self.assertEqual(disagreement["exact_21mer_matches"], 1)
        self.assertEqual(disagreement["exact_49mer_matches"], 1)


if __name__ == "__main__":
    unittest.main()
