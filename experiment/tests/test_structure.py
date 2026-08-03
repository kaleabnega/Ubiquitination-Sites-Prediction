from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.structure import (  # noqa: E402
    compact_prediction_metadata,
    confidence_category,
    parse_confidence_document,
    select_canonical_prediction,
    sequence_sha256,
)


class StructureTests(unittest.TestCase):
    def test_selects_exact_canonical_instead_of_isoform(self) -> None:
        sequence = "MAKAA"
        predictions = [
            {
                "uniprotAccession": "P1-2",
                "uniprotSequence": sequence,
                "uniprotStart": 1,
                "uniprotEnd": 5,
                "modelEntityId": "AF-P1-2-F1",
            },
            {
                "uniprotAccession": "P1",
                "uniprotSequence": sequence,
                "uniprotStart": 1,
                "uniprotEnd": 5,
                "modelEntityId": "AF-P1-F1",
                "latestVersion": 6,
            },
        ]

        selected, reason = select_canonical_prediction(
            "P1", sequence, predictions
        )

        self.assertEqual(reason, "available")
        self.assertEqual(selected["modelEntityId"], "AF-P1-F1")

    def test_rejects_sequence_or_fragment_mismatch(self) -> None:
        selected, reason = select_canonical_prediction(
            "P1",
            "MAKAA",
            [
                {
                    "uniprotAccession": "P1",
                    "uniprotSequence": "MAKA",
                    "uniprotStart": 1,
                    "uniprotEnd": 4,
                }
            ],
        )
        self.assertIsNone(selected)
        self.assertEqual(reason, "exact_full_sequence_not_returned")

    def test_parses_confidence_and_categories(self) -> None:
        confidence = parse_confidence_document(
            [
                {"residueNumber": 1, "confidenceScore": 49.9},
                {"residueNumber": 2, "confidenceScore": 70.0},
                {"residueNumber": 3, "confidenceScore": 90.0},
            ]
        )
        self.assertEqual(confidence, {1: 49.9, 2: 70.0, 3: 90.0})
        self.assertEqual(confidence_category(confidence[1]), "very_low")
        self.assertEqual(confidence_category(confidence[2]), "confident")
        self.assertEqual(confidence_category(confidence[3]), "very_high")

        parallel_arrays = parse_confidence_document(
            {
                "residueNumber": [1, 2, 3],
                "confidenceScore": [49.9, 70.0, 90.0],
                "confidenceCategory": ["D", "M", "H"],
            }
        )
        self.assertEqual(parallel_arrays, confidence)

    def test_compact_metadata_records_sequence_identity(self) -> None:
        sequence = "MAKAA"
        compact = compact_prediction_metadata(
            "P1",
            sequence,
            {
                "modelEntityId": "AF-P1-F1",
                "globalMetricValue": 81.5,
                "latestVersion": 6,
                "plddtDocUrl": "https://example.test/confidence.json",
            },
        )
        self.assertEqual(compact["sequence_length"], 5)
        self.assertEqual(compact["sequence_sha256"], sequence_sha256(sequence))
        self.assertEqual(compact["status"], "available")


if __name__ == "__main__":
    unittest.main()
