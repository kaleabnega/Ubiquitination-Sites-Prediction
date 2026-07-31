#!/usr/bin/env python3
"""Resumably resolve benchmark entry names to canonical UniProt proteins."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.external_benchmark import (  # noqa: E402
    ExternalSite,
    load_external_benchmark,
)

sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "scripts"))
from fetch_uniprot_context import make_session  # noqa: E402


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{identifier}.fasta"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def response_provenance(response: requests.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower()
        in {"x-uniprot-release", "x-uniprot-release-date", "x-total-results"}
    }


def fetch_entry_name_batch(
    session: requests.Session,
    entry_names: list[str],
    timeout_seconds: int,
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """Return source entry name -> (canonical accession, sequence)."""

    query = " OR ".join(f"id:{entry_name}" for entry_name in entry_names)
    response = session.get(
        UNIPROT_SEARCH_URL,
        params={
            "query": f"({query})",
            "format": "tsv",
            "fields": "accession,id,sequence",
            "size": 500,
            "includeIsoform": "false",
        },
        timeout=timeout_seconds,
    )
    # A bad historical identifier must not invalidate every other item. The
    # caller falls back to individual resolution when a whole batch is 400.
    if response.status_code == 400:
        return {}, response_provenance(response)
    response.raise_for_status()
    resolved: dict[str, tuple[str, str]] = {}
    for line in response.text.splitlines()[1:]:
        accession, entry_name, sequence = line.split("\t", maxsplit=2)
        resolved[entry_name] = (accession, sequence.strip().upper())
    return resolved, response_provenance(response)


def parse_fasta_response(
    response: requests.Response,
) -> tuple[str, str, str] | None:
    if response.status_code in (400, 404):
        return None
    response.raise_for_status()
    lines = response.text.splitlines()
    if not lines or not lines[0].startswith(">"):
        return None
    header_parts = lines[0][1:].split()[0].split("|")
    if len(header_parts) < 3:
        return None
    accession, entry_name = header_parts[1], header_parts[2]
    sequence = "".join(
        line.strip() for line in lines[1:] if line.strip()
    ).upper()
    if not sequence:
        return None
    return accession, entry_name, sequence


def fetch_one(
    session: requests.Session,
    site: ExternalSite,
    timeout_seconds: int,
) -> tuple[tuple[str, str] | None, dict[str, str]]:
    """Try the full entry name, then its accession-like historical prefix."""

    provenance: dict[str, str] = {}
    for identifier in (site.source_protein_id, site.accession_hint):
        response = session.get(
            UNIPROT_ENTRY_URL.format(identifier=identifier),
            timeout=timeout_seconds,
        )
        provenance.update(response_provenance(response))
        parsed = parse_fasta_response(response)
        if parsed is not None:
            accession, _, sequence = parsed
            return (accession, sequence), provenance
    return None, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", default="replication/MMUbiPred/benchmark.csv"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--pause-seconds", type=float, default=0.1)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 100:
        raise ValueError("batch-size must be between 1 and 100")

    benchmark_path = resolve_project_path(args.benchmark)
    output_path = resolve_project_path(args.output)
    records = load_external_benchmark(benchmark_path)
    representative = {
        record.source_protein_id: record for record in records
    }
    required = sorted(representative)

    if output_path.exists():
        cache = json.loads(output_path.read_text(encoding="utf-8"))
        if int(cache.get("schema_version", 0)) != 2:
            raise ValueError(
                "Existing external cache predates canonical entry-name "
                "resolution. Delete it and rerun this cell."
            )
    else:
        cache = {
            "schema_version": 2,
            "source": "UniProt REST API",
            "lookup_key": "published_UniProt_entry_name",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sequences": {},
            "canonical_accessions": {},
            "not_found": [],
            "response_provenance": {},
        }
    sequences = cache.get("sequences")
    canonical_accessions = cache.get("canonical_accessions")
    if not isinstance(sequences, dict) or not isinstance(
        canonical_accessions, dict
    ):
        raise TypeError("External cache dictionaries are malformed")
    known_not_found = set(map(str, cache.get("not_found", [])))
    missing = [
        entry_name
        for entry_name in required
        if entry_name not in sequences and entry_name not in known_not_found
    ]
    print(
        f"required_entry_names={len(required)} "
        f"cached={len(required) - len(missing)} missing={len(missing)}",
        flush=True,
    )

    session = make_session()
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start : start + args.batch_size]
        resolved, provenance = fetch_entry_name_batch(
            session, batch, timeout_seconds=args.timeout_seconds
        )
        for entry_name in batch:
            item = resolved.get(entry_name)
            if item is None:
                item, single_provenance = fetch_one(
                    session,
                    representative[entry_name],
                    args.timeout_seconds,
                )
                provenance.update(single_provenance)
            if item is None:
                known_not_found.add(entry_name)
            else:
                canonical_accession, sequence = item
                sequences[entry_name] = sequence
                canonical_accessions[entry_name] = canonical_accession
        cache["not_found"] = sorted(known_not_found)
        cache["response_provenance"] = provenance
        cache["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        cache["required_entry_names"] = len(required)
        cache["retrieved_entry_names"] = sum(
            entry_name in sequences for entry_name in required
        )
        write_atomic(output_path, cache)
        print(
            f"batch={start // args.batch_size + 1} "
            f"requested={len(batch)} "
            f"cached={cache['retrieved_entry_names']}/{len(required)} "
            f"not_found={len(known_not_found)}",
            flush=True,
        )
        if args.pause_seconds:
            time.sleep(args.pause_seconds)


if __name__ == "__main__":
    main()
