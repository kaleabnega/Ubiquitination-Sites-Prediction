#!/usr/bin/env python3
"""Fetch exact AlphaFold DB metadata and target-site pLDDT resumably."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.context import (  # noqa: E402
    build_validated_contexts,
    load_sequence_cache,
    sequence_cache_sha256,
)
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.structure import (  # noqa: E402
    compact_prediction_metadata,
    parse_confidence_document,
    select_canonical_prediction,
)


API_TEMPLATE = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
THREAD_LOCAL = threading.local()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = (
        "Ubiquitination-Sites-Prediction/structure-feasibility-v1 "
        "(academic research)"
    )
    return session


def session_for_thread() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = make_session()
        THREAD_LOCAL.session = session
    return session


def fetch_accession(
    accession: str,
    sequence: str,
    positions: list[int],
    timeout_seconds: int,
) -> dict[str, object]:
    session = session_for_thread()
    response = session.get(
        API_TEMPLATE.format(accession=accession), timeout=timeout_seconds
    )
    if response.status_code == 404:
        return {"status": "not_found", "accession": accession}
    response.raise_for_status()
    predictions = response.json()
    if not isinstance(predictions, list):
        raise TypeError("AlphaFold prediction API did not return a list")
    selected, reason = select_canonical_prediction(
        accession, sequence, predictions
    )
    if selected is None:
        return {"status": reason, "accession": accession}
    metadata = compact_prediction_metadata(accession, sequence, selected)
    confidence_url = metadata.get("plddtDocUrl")
    if not confidence_url:
        return {**metadata, "status": "confidence_url_missing"}
    confidence_response = session.get(
        str(confidence_url), timeout=timeout_seconds
    )
    confidence_response.raise_for_status()
    confidence = parse_confidence_document(confidence_response.json())
    site_confidence = {
        str(position): confidence[position]
        for position in positions
        if position in confidence
    }
    metadata["site_confidence"] = site_confidence
    metadata["requested_site_positions"] = positions
    metadata["confidence_positions_available"] = len(confidence)
    metadata["status"] = (
        "available"
        if len(site_confidence) == len(positions)
        else "site_confidence_incomplete"
    )
    return metadata


def entry_complete(entry: object, positions: list[int]) -> bool:
    if not isinstance(entry, dict):
        return False
    status = str(entry.get("status", ""))
    if status in {
        "not_found",
        "canonical_accession_not_returned",
        "exact_full_sequence_not_returned",
        "confidence_url_missing",
        "site_confidence_incomplete",
    }:
        return True
    site_confidence = entry.get("site_confidence")
    return status == "available" and isinstance(site_confidence, dict) and all(
        str(position) in site_confidence for position in positions
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="replication/MMUbiPred")
    parser.add_argument("--sequence-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-window-size", type=int, default=257)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 12:
        raise ValueError("max-workers must be between 1 and 12")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")

    data_dir = resolve_project_path(args.data_dir)
    sequence_cache_path = resolve_project_path(args.sequence_cache)
    output_path = resolve_project_path(args.output)
    records, _ = load_released_split(data_dir, split="train", window_size=49)
    sequences, sequence_metadata = load_sequence_cache(sequence_cache_path)
    valid_indices, _, validation = build_validated_contexts(
        records,
        sequences,
        context_window_size=args.context_window_size,
    )
    positions_by_accession: dict[str, set[int]] = {}
    for index in valid_indices:
        record = records[index]
        if record.position is None:
            raise AssertionError("Validated record lost its position")
        positions_by_accession.setdefault(record.protein_id, set()).add(
            record.position + 1
        )
    required = sorted(positions_by_accession)
    sequence_hash = sequence_cache_sha256(sequence_cache_path)
    required_hash = hashlib.sha256("\n".join(required).encode()).hexdigest()

    if output_path.exists():
        cache = json.loads(output_path.read_text(encoding="utf-8"))
        if int(cache.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported AlphaFold cache schema")
        if cache.get("sequence_cache_sha256") != sequence_hash:
            raise ValueError("UniProt sequence cache changed since AFDB retrieval")
        if cache.get("required_accessions_sha256") != required_hash:
            raise ValueError("Required accession set changed since AFDB retrieval")
    else:
        cache = {
            "schema_version": 1,
            "source": "AlphaFold Protein Structure Database API",
            "api_template": API_TEMPLATE,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sequence_cache_sha256": sequence_hash,
            "sequence_cache_metadata": sequence_metadata,
            "required_accessions_sha256": required_hash,
            "entries": {},
        }
    entries = cache.get("entries")
    if not isinstance(entries, dict):
        raise TypeError("AlphaFold cache entries must be a dictionary")
    pending = [
        accession
        for accession in required
        if not entry_complete(
            entries.get(accession),
            sorted(positions_by_accession[accession]),
        )
    ]
    print(
        f"context_valid_sites={len(valid_indices)}/{len(records)} "
        f"required_accessions={len(required)} "
        f"complete={len(required) - len(pending)} pending={len(pending)}",
        flush=True,
    )

    unresolved: list[dict[str, str]] = []
    completed_since_write = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                fetch_accession,
                accession,
                sequences[accession],
                sorted(positions_by_accession[accession]),
                args.timeout_seconds,
            ): accession
            for accession in pending
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            accession = futures[future]
            try:
                entries[accession] = future.result()
                completed_since_write += 1
            except Exception as error:  # network failures remain retryable
                unresolved.append(
                    {"accession": accession, "error": repr(error)}
                )
            if (
                completed_since_write >= args.checkpoint_every
                or completed_count == len(futures)
            ):
                cache["updated_at_utc"] = dt.datetime.now(
                    dt.timezone.utc
                ).isoformat()
                cache["required_accessions"] = len(required)
                cache["completed_accessions"] = sum(
                    entry_complete(
                        entries.get(value),
                        sorted(positions_by_accession[value]),
                    )
                    for value in required
                )
                write_json_atomic(output_path, cache)
                print(
                    f"processed={completed_count}/{len(pending)} "
                    f"cached={cache['completed_accessions']}/{len(required)} "
                    f"retryable_errors={len(unresolved)}",
                    flush=True,
                )
                completed_since_write = 0

    final_pending = [
        accession
        for accession in required
        if not entry_complete(
            entries.get(accession),
            sorted(positions_by_accession[accession]),
        )
    ]
    print(
        json.dumps(
            {
                "status": (
                    "retrieval_complete" if not final_pending else "resume_required"
                ),
                "required_accessions": len(required),
                "completed_accessions": len(required) - len(final_pending),
                "pending_accessions": len(final_pending),
                "retryable_error_examples": unresolved[:20],
                "context_validation": validation,
                "released_independent_test_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if final_pending:
        raise RuntimeError(
            f"{len(final_pending)} accessions remain retryable; rerun the same command"
        )


if __name__ == "__main__":
    main()
