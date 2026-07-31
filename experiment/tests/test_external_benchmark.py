from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.context import centered_sequence_window  # noqa: E402
from ubipred.external_benchmark import (  # noqa: E402
    ExternalSite,
    build_released_leakage_sets,
    leakage_reasons,
    load_external_benchmark,
    parse_mmseqs_homology_hit,
    parse_external_identifier,
    sha256_file,
    validate_external_site,
)
from ubipred.fasta import SiteRecord  # noqa: E402


class ExternalBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sequence = "A" * 50 + "K" + "C" * 70
        self.window_21 = centered_sequence_window(self.sequence, 51, 21)
        self.site = ExternalSite(
            benchmark_index=0,
            identifier="P12345_HUMAN_51",
            source_protein_id="P12345_HUMAN",
            accession_hint="P12345",
            species="HUMAN",
            position=51,
            window_21=self.window_21,
            label=1,
        )

    def test_identifier_and_csv_contract(self) -> None:
        self.assertEqual(
            parse_external_identifier("P12345_HUMAN_51"),
            ("P12345", "P12345_HUMAN", "HUMAN", 51),
        )
        self.assertEqual(
            parse_external_identifier("1433B_HUMAN_10"),
            ("1433B", "1433B_HUMAN", "HUMAN", 10),
        )
        with self.assertRaises(ValueError):
            parse_external_identifier("P12345|51")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Uniprot", "Seq", "Label"])
                writer.writerow(
                    ["P12345_HUMAN_51", self.window_21, 1]
                )
            records = load_external_benchmark(path)
            self.assertEqual(records, [self.site])

    def test_released_benchmark_is_the_frozen_2077_site_source(self) -> None:
        path = PROJECT_ROOT / "replication" / "MMUbiPred" / "benchmark.csv"
        records = load_external_benchmark(path)
        self.assertEqual(len(records), 2077)
        self.assertEqual(
            sum(record.label == 1 for record in records), 753
        )
        self.assertEqual(
            len({record.source_protein_id for record in records}), 1692
        )
        self.assertEqual(
            sha256_file(path),
            "c303df3ab199b3aa3d4c754aa3cd255343740298c11b1c03f55b8f2c4474ab9f",
        )

    def test_position_is_one_based_and_anchor_must_match(self) -> None:
        validated, reason = validate_external_site(
            self.site, self.sequence, "P12345"
        )
        self.assertEqual(reason, "validated")
        self.assertIsNotNone(validated)
        assert validated is not None
        self.assertEqual(validated.window_49[24], "K")
        self.assertEqual(validated.context_257[128], "K")

        shifted = ExternalSite(
            **{**self.site.__dict__, "position": 50}
        )
        invalid, reason = validate_external_site(
            shifted, self.sequence, "P12345"
        )
        self.assertIsNone(invalid)
        self.assertEqual(reason, "uniprot_center_is_not_lysine")

    def test_mmseqs_hit_uses_shorter_full_protein_denominator(self) -> None:
        hit = parse_mmseqs_homology_hit(
            "Q1\tT1\t31\t0.81\t0.42\t100\t200\t1e-20\n"
        )
        self.assertAlmostEqual(hit.global_identity_to_shorter, 0.31)
        self.assertAlmostEqual(hit.shorter_sequence_coverage, 0.81)

        reverse = parse_mmseqs_homology_hit(
            "Q2\tT2\t61\t0.45\t0.82\t300\t200\t1e-30\n"
        )
        self.assertAlmostEqual(
            reverse.global_identity_to_shorter, 0.305
        )
        self.assertAlmostEqual(reverse.shorter_sequence_coverage, 0.82)

    def test_conservative_released_overlap_detection(self) -> None:
        released_window = centered_sequence_window(self.sequence, 51, 49)
        train_record = SiteRecord(
            header="P12345|50",
            protein_id="P12345",
            position=50,
            sequence=released_window,
            label=1,
            source="train.fasta",
        )
        validated, _ = validate_external_site(
            self.site, self.sequence, "P12345"
        )
        assert validated is not None
        reasons = leakage_reasons(
            validated,
            build_released_leakage_sets([train_record], []),
        )
        self.assertIn("training_accession_overlap", reasons)
        self.assertIn("released_site_overlap", reasons)
        self.assertIn("released_21mer_overlap", reasons)
        self.assertIn("released_49mer_overlap", reasons)


if __name__ == "__main__":
    unittest.main()
