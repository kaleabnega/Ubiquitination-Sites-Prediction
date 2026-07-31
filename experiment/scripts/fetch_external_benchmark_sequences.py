#!/usr/bin/env python3
"""Resumably retrieve UniProt proteins required by an external benchmark."""

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

from ubipred.external_benchmark import load_external_benchmark  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "scripts"))
from fetch_uniprot_context import fetch_batch, make_session  # noqa: E402


UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"


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


def fetch_one(
    session: requests.Session, accession: str, timeout_seconds: int
) -> tuple[str | None, dict[str, str]]:
    response = session.get(
        UNIPROT_ENTRY_URL.format(accession=accession),
        timeout=timeout_seconds,
    )
    if response.status_code == 404:
        return None, {}
    response.raise_for_status()
    lines = [
        line.strip()
        for line in response.text.splitlines()
        if line and not line.startswith(">")
    ]
    sequence = "".join(lines).upper()
    provenance = {
        key: value
        for key, value in response.headers.items()
        if key.lower() in {"x-uniprot-release", "x-uniprot-release-date"}
    }
    return sequence or None, provenance


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
    required = sorted({record.accession for record in records})

    if output_path.exists():
        cache = json.loads(output_path.read_text(encoding="utf-8"))
        if int(cache.get("schema_version", 0)) != 1:
            raise ValueError("Existing cache has an unsupported schema")
    else:
        cache = {
            "schema_version": 1,
            "source": "UniProt REST API",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sequences": {},
            "not_found": [],
            "response_provenance": {},
        }
    sequences = cache.get("sequences")
    if not isinstance(sequences, dict):
        raise TypeError("Cache sequences must be a dictionary")
    known_not_found = set(map(str, cache.get("not_found", [])))
    missing = [
        accession
        for accession in required
        if accession not in sequences and accession not in known_not_found
    ]
    print(
        f"required_accessions={len(required)} "
        f"cached={len(required) - len(missing)} missing={len(missing)}",
        flush=True,
    )

    session = make_session()
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start : start + args.batch_size]
        retrieved, provenance = fetch_batch(
            session, batch, timeout_seconds=args.timeout_seconds
        )
        # Batch queries return primary accessions. Resolve aliases and obsolete
        # identifiers individually only when the requested key was not returned.
        for accession in batch:
            if accession in retrieved:
                sequences[accession] = retrieved[accession]
                continue
            sequence, single_provenance = fetch_one(
                session, accession, args.timeout_seconds
            )
            provenance.update(single_provenance)
            if sequence is None:
                known_not_found.add(accession)
            else:
                sequences[accession] = sequence
        cache["not_found"] = sorted(known_not_found)
        cache["response_provenance"] = provenance
        cache["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        cache["required_accessions"] = len(required)
        cache["retrieved_accessions"] = sum(
            accession in sequences for accession in required
        )
        write_atomic(output_path, cache)
        print(
            f"batch={start // args.batch_size + 1} "
            f"requested={len(batch)} "
            f"cached={cache['retrieved_accessions']}/{len(required)} "
            f"not_found={len(known_not_found)}",
            flush=True,
        )
        if args.pause_seconds:
            time.sleep(args.pause_seconds)


if __name__ == "__main__":
    main()
