#!/usr/bin/env python3
"""Fetch and validate full UniProt sequences for a released data split."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.context import build_validated_contexts  # noqa: E402
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.metrics import write_json  # noqa: E402


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_cache_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def make_session() -> requests.Session:
    retry = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers["User-Agent"] = (
        "MMUbiPred-Context-Residual/1.0 "
        "(research sequence retrieval; https://github.com/kaleabnega/"
        "Ubiquitination-Sites-Prediction)"
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_batch(
    session: requests.Session, accessions: list[str], timeout_seconds: int
) -> tuple[dict[str, str], dict[str, str]]:
    query = " OR ".join(f"accession:{accession}" for accession in accessions)
    response = session.get(
        UNIPROT_SEARCH_URL,
        params={
            "query": f"({query})",
            "format": "tsv",
            "fields": "accession,sequence",
            "size": 500,
            "includeIsoform": "false",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    sequences: dict[str, str] = {}
    lines = response.text.splitlines()
    for line in lines[1:]:
        accession, sequence = line.split("\t", maxsplit=1)
        sequences[accession] = sequence.strip().upper()
    provenance = {
        key: value
        for key, value in response.headers.items()
        if key.lower()
        in {
            "x-uniprot-release",
            "x-uniprot-release-date",
            "x-total-results",
        }
    }
    return sequences, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="replication/MMUbiPred")
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
    )
    parser.add_argument(
        "--allow-locked-test",
        action="store_true",
        help="Required acknowledgement when retrieving test-site context.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--context-window-size", type=int, default=257)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--pause-seconds", type=float, default=0.1)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.batch_size > 100:
        raise ValueError("batch-size must be between 1 and 100")
    if args.split == "test" and not args.allow_locked_test:
        parser.error(
            "Refusing to access independent-test site identifiers without "
            "--allow-locked-test"
        )

    data_dir = resolve_project_path(args.data_dir)
    output = resolve_project_path(args.output)
    validation_report = (
        resolve_project_path(args.validation_report)
        if args.validation_report is not None
        else output.with_name("context_validation.json")
    )
    records, preprocessing = load_released_split(
        data_dir, split=args.split, window_size=49
    )
    required_accessions = sorted({record.protein_id for record in records})

    if output.exists():
        cache = json.loads(output.read_text(encoding="utf-8"))
        if int(cache.get("schema_version", 0)) != 1:
            raise ValueError("Existing cache has an unsupported schema")
    else:
        cache = {
            "schema_version": 1,
            "source": UNIPROT_SEARCH_URL,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sequences": {},
            "response_provenance": {},
        }
    sequences = cache["sequences"]
    if not isinstance(sequences, dict):
        raise TypeError("Existing cache sequences field is not a dictionary")
    missing = [
        accession for accession in required_accessions if accession not in sequences
    ]
    print(
        f"required_accessions={len(required_accessions)} "
        f"cached={len(required_accessions) - len(missing)} missing={len(missing)}",
        flush=True,
    )

    session = make_session()
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start : start + args.batch_size]
        retrieved, provenance = fetch_batch(
            session, batch, timeout_seconds=args.timeout_seconds
        )
        sequences.update(retrieved)
        cache["response_provenance"] = provenance
        cache["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        cache["required_accessions"] = len(required_accessions)
        cache["retrieved_accessions"] = sum(
            accession in sequences for accession in required_accessions
        )
        write_cache_atomic(output, cache)
        print(
            f"batch={start // args.batch_size + 1} "
            f"requested={len(batch)} returned={len(retrieved)} "
            f"cached={cache['retrieved_accessions']}/{len(required_accessions)}",
            flush=True,
        )
        if args.pause_seconds:
            time.sleep(args.pause_seconds)

    valid_indices, _, context_report = build_validated_contexts(
        records,
        {str(key): str(value) for key, value in sequences.items()},
        context_window_size=args.context_window_size,
    )
    report = {
        "split": args.split,
        "source": UNIPROT_SEARCH_URL,
        "sequence_cache": str(output),
        "required_unique_accessions": len(required_accessions),
        "retrieved_unique_accessions": sum(
            accession in sequences for accession in required_accessions
        ),
        "preprocessing": {
            key: vars(value) for key, value in preprocessing.items()
        },
        "context_validation": context_report,
        "valid_indices_sha256": hashlib.sha256(
            np.asarray(valid_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "independent_test_accessed": args.split == "test",
    }
    write_json(validation_report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.split == "test":
        print("Released independent-test site context was accessed.")
    else:
        print("Released independent test set was not accessed.")


if __name__ == "__main__":
    main()
