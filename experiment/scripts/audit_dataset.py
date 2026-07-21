#!/usr/bin/env python3
"""Audit released MMUbiPred artifacts without ML dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "experiment" / "src"
sys.path.insert(0, str(SRC_ROOT))

from ubipred.fasta import load_released_split  # noqa: E402


RELEASED_FILES = [
    "Positive_90_percent_training_set_DeepUBI.fasta",
    "Negative_90_percent_training_set_DeepUBI.fasta",
    "Positive_10_percent_independent_test_set_DeepUBI.fasta",
    "Negative_10_percent_independent_test_set_DeepUBI.fasta",
    "aaindex31.txt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_counts(records) -> dict[str, int]:
    counts = Counter(record.label for record in records)
    return {
        "negative": counts[0],
        "positive": counts[1],
        "total": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "replication" / "MMUbiPred",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window-size", type=int, default=49)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    train, train_reports = load_released_split(
        data_dir, split="train", window_size=args.window_size
    )
    test, test_reports = load_released_split(
        data_dir, split="test", window_size=args.window_size
    )

    train_proteins = {record.protein_id for record in train}
    test_proteins = {record.protein_id for record in test}
    train_sites = {(record.protein_id, record.position) for record in train}
    test_sites = {(record.protein_id, record.position) for record in test}
    train_windows = {record.sequence for record in train}
    test_windows = {record.sequence for record in test}

    train_labels_by_window: dict[str, set[int]] = {}
    for record in train:
        train_labels_by_window.setdefault(record.sequence, set()).add(record.label)

    try:
        displayed_data_dir = str(data_dir.relative_to(PROJECT_ROOT))
    except ValueError:
        displayed_data_dir = str(data_dir)

    audit = {
        "data_dir": displayed_data_dir,
        "window_size": args.window_size,
        "train": {
            "counts": label_counts(train),
            "reports": {
                name: asdict(report) for name, report in train_reports.items()
            },
            "unique_proteins": len(train_proteins),
            "unique_sites": len(train_sites),
            "unique_windows": len(train_windows),
            "windows_with_conflicting_labels": sum(
                len(labels) > 1 for labels in train_labels_by_window.values()
            ),
        },
        "test": {
            "counts": label_counts(test),
            "reports": {
                name: asdict(report) for name, report in test_reports.items()
            },
            "unique_proteins": len(test_proteins),
            "unique_sites": len(test_sites),
            "unique_windows": len(test_windows),
        },
        "overlap": {
            "proteins": len(train_proteins & test_proteins),
            "sites": len(train_sites & test_sites),
            "cropped_49_residue_windows": len(train_windows & test_windows),
        },
        "artifacts": {
            filename: {
                "bytes": (data_dir / filename).stat().st_size,
                "sha256": sha256(data_dir / filename),
            }
            for filename in RELEASED_FILES
        },
    }

    rendered = json.dumps(audit, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
