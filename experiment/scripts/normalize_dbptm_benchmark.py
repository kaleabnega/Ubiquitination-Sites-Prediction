#!/usr/bin/env python3
"""Download, audit, and normalize the official dbPTM ubiquitination benchmark."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.external_benchmark import parse_external_identifier  # noqa: E402
from ubipred.fasta import ALPHABET_SET  # noqa: E402


SOURCE_URL = (
    "https://biomics.lab.nycu.edu.tw/dbPTM/download/benchmark/"
    "Ubiquitination.tgz"
)
SOURCE_SHA256 = (
    "8b74e4fbc7b36ed50c015a98a6c17a891d438982e39bb40b7ad127ab8a3c6547"
)
EXPECTED_MEMBERS = {
    "Ubiquitination/Ubiquitination_pos.fasta": (1, 9767),
    "Ubiquitination/Ubiquitination_neg.fasta": (0, 8579),
}


@dataclass(frozen=True)
class RawRecord:
    identifier: str
    sequence: str
    label: int
    source_member: str
    source_index: int


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_atomic(url: str, destination: Path) -> dict[str, str]:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(
            "wb", dir=destination.parent, delete=False
        ) as handle:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    handle.write(block)
            temporary = Path(handle.name)
        provenance = {
            key: value
            for key, value in response.headers.items()
            if key.lower()
            in {"etag", "last-modified", "content-length", "content-type"}
        }
    os.replace(temporary, destination)
    return provenance


def parse_fasta_text(
    text: str, label: int, source_member: str
) -> list[RawRecord]:
    records: list[RawRecord] = []
    identifier: str | None = None
    sequence_parts: list[str] = []

    def finish() -> None:
        if identifier is None:
            return
        records.append(
            RawRecord(
                identifier=identifier,
                sequence="".join(sequence_parts).upper(),
                label=label,
                source_member=source_member,
                source_index=len(records),
            )
        )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            finish()
            identifier = line[1:].strip()
            sequence_parts = []
        else:
            if identifier is None:
                raise ValueError(f"Sequence precedes header in {source_member}")
            sequence_parts.append(line)
    finish()
    return records


def load_archive(path: Path) -> list[RawRecord]:
    records: list[RawRecord] = []
    with tarfile.open(path, "r:gz") as archive:
        member_names = {member.name for member in archive.getmembers() if member.isfile()}
        if member_names != set(EXPECTED_MEMBERS):
            raise ValueError(
                f"Unexpected dbPTM archive members: {sorted(member_names)}"
            )
        for member_name, (label, expected_count) in EXPECTED_MEMBERS.items():
            extracted = archive.extractfile(member_name)
            if extracted is None:
                raise FileNotFoundError(member_name)
            member_records = parse_fasta_text(
                extracted.read().decode("utf-8"), label, member_name
            )
            if len(member_records) != expected_count:
                raise ValueError(
                    f"{member_name} contains {len(member_records)} records; "
                    f"expected {expected_count}"
                )
            records.extend(member_records)
    return records


def normalize_records(
    records: list[RawRecord],
) -> tuple[list[RawRecord], dict[str, object]]:
    by_identifier: dict[str, list[RawRecord]] = defaultdict(list)
    for record in records:
        by_identifier[record.identifier].append(record)
    duplicate_identifiers = {
        identifier
        for identifier, grouped in by_identifier.items()
        if len(grouped) > 1
    }
    reasons: Counter[str] = Counter()
    retained: list[RawRecord] = []
    for record in records:
        if record.identifier in duplicate_identifiers:
            reasons["duplicate_identifier"] += 1
            continue
        try:
            parse_external_identifier(record.identifier)
        except ValueError:
            reasons["invalid_identifier"] += 1
            continue
        if len(record.sequence) != 21:
            reasons["invalid_window_length"] += 1
            continue
        if record.sequence[10] != "K":
            reasons["non_lysine_center"] += 1
            continue
        if any(residue not in ALPHABET_SET for residue in record.sequence):
            reasons["unsupported_residue"] += 1
            continue
        retained.append(record)
    support = Counter(record.label for record in retained)
    audit = {
        "raw_records": len(records),
        "raw_positive": sum(record.label == 1 for record in records),
        "raw_negative": sum(record.label == 0 for record in records),
        "raw_unique_identifiers": len(by_identifier),
        "duplicate_identifiers": len(duplicate_identifiers),
        "duplicate_identifier_examples": sorted(duplicate_identifiers)[:50],
        "exclusion_reason_counts": dict(sorted(reasons.items())),
        "normalized_records": len(retained),
        "normalized_positive": support[1],
        "normalized_negative": support[0],
        "normalized_unique_entry_names": len(
            {parse_external_identifier(record.identifier)[1] for record in retained}
        ),
    }
    return retained, audit


def write_csv(path: Path, records: list[RawRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Uniprot", "Seq", "Label"])
        for record in records:
            writer.writerow(
                [record.identifier, record.sequence, record.label]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source-url", default=SOURCE_URL)
    args = parser.parse_args()

    archive_path = args.archive.resolve()
    output_path = args.output.resolve()
    audit_path = args.audit.resolve()
    response_provenance: dict[str, str] = {}
    if not archive_path.exists():
        response_provenance = download_atomic(
            args.source_url, archive_path
        )
    observed_source_hash = sha256_file(archive_path)
    if observed_source_hash != SOURCE_SHA256:
        raise ValueError(
            "Official dbPTM benchmark checksum changed; refusing to normalize"
        )
    raw_records = load_archive(archive_path)
    normalized, audit = normalize_records(raw_records)
    write_csv(output_path, normalized)
    payload = {
        "status": "official dbPTM benchmark normalized; no model inference",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "url": args.source_url,
            "archive_path": str(archive_path),
            "sha256": observed_source_hash,
            "response_provenance": response_provenance,
        },
        "normalization": audit,
        "normalized_csv": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
        },
        "external_predictions_generated": False,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
