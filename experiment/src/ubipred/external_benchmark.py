"""Parsing, validation, and leakage audits for site-level external benchmarks."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .context import CONTEXT_RESIDUES, centered_sequence_window
from .fasta import ALPHABET_SET, SiteRecord, center_crop


BENCHMARK_ID_PATTERN = re.compile(
    r"^(?P<identifier_prefix>[^_]+)_(?P<species>[^_]+)_"
    r"(?P<position>[1-9][0-9]*)$"
)


@dataclass(frozen=True)
class ExternalSite:
    """One labelled external site with a one-based UniProt coordinate."""

    benchmark_index: int
    identifier: str
    source_protein_id: str
    accession_hint: str
    species: str
    position: int
    window_21: str
    label: int


@dataclass(frozen=True)
class ValidatedExternalSite:
    """An external site whose supplied anchor matches the retrieved protein."""

    site: ExternalSite
    canonical_accession: str
    window_49: str
    context_257: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_external_identifier(identifier: str) -> tuple[str, str, str, int]:
    match = BENCHMARK_ID_PATTERN.fullmatch(identifier.strip())
    if match is None:
        raise ValueError(f"Invalid external benchmark identifier: {identifier}")
    prefix = match.group("identifier_prefix")
    species = match.group("species")
    return prefix, f"{prefix}_{species}", species, int(match.group("position"))


def load_external_benchmark(path: str | Path) -> list[ExternalSite]:
    records: list[ExternalSite] = []
    identifiers: set[str] = set()
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Uniprot", "Seq", "Label"]:
            raise ValueError(
                "Expected benchmark columns exactly: Uniprot, Seq, Label"
            )
        for index, row in enumerate(reader):
            identifier = row["Uniprot"].strip()
            if identifier in identifiers:
                raise ValueError(f"Duplicate benchmark identifier: {identifier}")
            identifiers.add(identifier)
            (
                accession_hint,
                source_protein_id,
                species,
                position,
            ) = parse_external_identifier(identifier)
            window = row["Seq"].strip().upper()
            if len(window) != 21 or window[10] != "K":
                raise ValueError(
                    f"Benchmark row {index} is not a lysine-centred 21-mer"
                )
            label = int(row["Label"])
            if label not in (0, 1):
                raise ValueError(f"Benchmark row {index} has invalid label")
            records.append(
                ExternalSite(
                    benchmark_index=index,
                    identifier=identifier,
                    source_protein_id=source_protein_id,
                    accession_hint=accession_hint,
                    species=species,
                    position=position,
                    window_21=window,
                    label=label,
                )
            )
    if not records:
        raise ValueError("External benchmark is empty")
    return records


def validate_external_site(
    site: ExternalSite,
    protein_sequence: str | None,
    canonical_accession: str | None,
    context_window_size: int = 257,
) -> tuple[ValidatedExternalSite | None, str]:
    """Validate the published one-based position and reconstruct both windows."""

    if protein_sequence is None or canonical_accession is None:
        return None, "protein_identifier_not_resolved"
    sequence = protein_sequence.strip().upper()
    try:
        observed_21 = centered_sequence_window(
            sequence, site.position, window_size=21
        )
        window_49 = centered_sequence_window(
            sequence, site.position, window_size=49
        )
        context = centered_sequence_window(
            sequence, site.position, window_size=context_window_size
        )
    except ValueError:
        return None, "position_out_of_range"
    if sequence[site.position - 1] != "K":
        return None, "uniprot_center_is_not_lysine"
    if observed_21 != site.window_21:
        return None, "published_21mer_mismatch"
    if any(residue not in ALPHABET_SET for residue in window_49):
        return None, "unsupported_49mer_residue"
    if any(
        residue not in set(CONTEXT_RESIDUES) | {"-"} for residue in context
    ):
        return None, "unsupported_context_residue"
    return ValidatedExternalSite(
        site, canonical_accession, window_49, context
    ), "validated"


def build_released_leakage_sets(
    training_records: Sequence[SiteRecord],
    historical_test_records: Sequence[SiteRecord],
) -> dict[str, set[object]]:
    """Build conservative accession, site, and exact-window exclusion sets."""

    all_records = list(training_records) + list(historical_test_records)
    return {
        "training_accessions": {
            record.protein_id for record in training_records
        },
        "historical_test_accessions": {
            record.protein_id for record in historical_test_records
        },
        "released_sites": {
            (record.protein_id, record.position)
            for record in all_records
            if record.position is not None
        },
        "released_21mers": {
            center_crop(record.sequence, 21) for record in all_records
        },
        "released_49mers": {record.sequence for record in all_records},
    }


def leakage_reasons(
    validated: ValidatedExternalSite,
    leakage_sets: Mapping[str, set[object]],
) -> list[str]:
    """Return every exact leakage reason; an empty result is eligible."""

    site = validated.site
    reasons: list[str] = []
    canonical_accession = validated.canonical_accession
    if canonical_accession in leakage_sets["training_accessions"]:
        reasons.append("training_accession_overlap")
    if canonical_accession in leakage_sets["historical_test_accessions"]:
        reasons.append("historical_test_accession_overlap")
    # Released PLMD headers are zero-based; external positions are one-based.
    if (
        canonical_accession,
        site.position - 1,
    ) in leakage_sets["released_sites"]:
        reasons.append("released_site_overlap")
    if site.window_21 in leakage_sets["released_21mers"]:
        reasons.append("released_21mer_overlap")
    if validated.window_49 in leakage_sets["released_49mers"]:
        reasons.append("released_49mer_overlap")
    return reasons


def label_counts(records: Iterable[ValidatedExternalSite]) -> dict[str, int]:
    labels = [record.site.label for record in records]
    return {
        "negative": labels.count(0),
        "positive": labels.count(1),
        "total": len(labels),
    }
