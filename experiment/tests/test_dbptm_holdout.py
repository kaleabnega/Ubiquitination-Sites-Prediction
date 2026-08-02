from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "scripts"))

from normalize_dbptm_benchmark import (  # noqa: E402
    RawRecord,
    normalize_records,
    parse_fasta_text,
)


class DbptmHoldoutTests(unittest.TestCase):
    def test_fasta_parser_and_normalization_are_fail_closed(self) -> None:
        text = (
            ">P12345_HUMAN_11\r\nAAAAAAAAAAKAAAAAAAAAA\r\n"
            ">P99999_HUMAN_12\nAAAAAAAAAAAAAAAAAAAAA\n"
        )
        records = parse_fasta_text(text, 1, "positive.fasta")
        retained, audit = normalize_records(records)
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].identifier, "P12345_HUMAN_11")
        self.assertEqual(audit["exclusion_reason_counts"], {"non_lysine_center": 1})

    def test_conflicting_or_repeated_identifiers_are_all_excluded(self) -> None:
        records = [
            RawRecord(
                "P12345_HUMAN_11",
                "AAAAAAAAAAKAAAAAAAAAA",
                1,
                "positive.fasta",
                0,
            ),
            RawRecord(
                "P12345_HUMAN_11",
                "CCCCCCCCCCKCCCCCCCCCC",
                0,
                "negative.fasta",
                0,
            ),
        ]
        retained, audit = normalize_records(records)
        self.assertEqual(retained, [])
        self.assertEqual(audit["duplicate_identifiers"], 1)
        self.assertEqual(
            audit["exclusion_reason_counts"], {"duplicate_identifier": 2}
        )


if __name__ == "__main__":
    unittest.main()
