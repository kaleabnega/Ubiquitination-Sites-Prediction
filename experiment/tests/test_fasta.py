from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.fasta import (  # noqa: E402
    SiteRecord,
    center_crop,
    iter_fasta,
    prepare_records,
)


def lysine_centered_sequence(residue: str = "A") -> str:
    sequence = [residue] * 81
    sequence[40] = "K"
    return "".join(sequence)


class FastaTests(unittest.TestCase):
    def test_iter_fasta_supports_wrapped_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.fasta"
            path.write_text(">P12345|41\nAAAA\nKKKK\n", encoding="utf-8")
            self.assertEqual(
                list(iter_fasta(path)), [("P12345|41", "AAAAKKKK")]
            )

    def test_center_crop_matches_authors_indices(self) -> None:
        sequence = "".join(chr(65 + (index % 26)) for index in range(81))
        self.assertEqual(center_crop(sequence, 49), sequence[16:65])

    def test_prepare_records_filters_unsupported_residue(self) -> None:
        valid = SiteRecord(
            header="P1|41",
            protein_id="P1",
            position=41,
            sequence=lysine_centered_sequence(),
            label=1,
            source="test",
        )
        invalid_sequence = list(lysine_centered_sequence())
        invalid_sequence[10] = "B"
        invalid = SiteRecord(
            header="P2|41",
            protein_id="P2",
            position=41,
            sequence="".join(invalid_sequence),
            label=1,
            source="test",
        )

        records, report = prepare_records([valid, invalid])
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0].sequence), 49)
        self.assertEqual(records[0].sequence[24], "K")
        self.assertEqual(report.dropped_unsupported, 1)


if __name__ == "__main__":
    unittest.main()
