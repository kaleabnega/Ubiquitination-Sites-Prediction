#!/usr/bin/env python3
"""Freeze the training-only AlphaFold site-coverage feasibility report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.context import (  # noqa: E402
    build_validated_contexts,
    load_sequence_cache,
    sequence_cache_sha256,
)
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.structure import confidence_category, sequence_sha256  # noqa: E402


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def score_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "minimum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metadata-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_dir = resolve_project_path(config["data_dir"])
    sequence_cache_path = resolve_project_path(config["sequence_cache"])
    metadata_cache_path = resolve_project_path(args.metadata_cache)
    output_dir = resolve_project_path(args.output_dir or config["output_dir"])
    report_path = output_dir / "structure_feasibility.json"
    features_path = output_dir / "site_structure_confidence.npz"
    if report_path.exists():
        raise FileExistsError(
            "Completed structural feasibility report already exists; "
            "refusing to overwrite it"
        )

    records, preprocessing = load_released_split(
        data_dir, split="train", window_size=49
    )
    sequences, _ = load_sequence_cache(sequence_cache_path)
    valid_indices, _, context_validation = build_validated_contexts(
        records,
        sequences,
        context_window_size=int(config["context_window_size"]),
    )
    cache = json.loads(metadata_cache_path.read_text(encoding="utf-8"))
    if int(cache.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported AlphaFold metadata-cache schema")
    if cache.get("sequence_cache_sha256") != sequence_cache_sha256(
        sequence_cache_path
    ):
        raise ValueError("AlphaFold and UniProt sequence caches do not match")
    entries = cache.get("entries")
    if not isinstance(entries, dict):
        raise TypeError("AlphaFold cache entries must be a dictionary")

    reason_counts: Counter[str] = Counter()
    reason_by_label = {0: Counter(), 1: Counter()}
    score_by_label: dict[int, list[float]] = {0: [], 1: []}
    category_counts: Counter[str] = Counter()
    resolution_methods_by_site: Counter[str] = Counter()
    eligible_indices: list[int] = []
    eligible_labels: list[int] = []
    accessions: list[str] = []
    positions: list[int] = []
    site_plddt: list[float] = []
    global_plddt: list[float] = []
    issue_examples: list[dict[str, object]] = []

    for dataset_index in valid_indices:
        record = records[dataset_index]
        if record.position is None:
            raise AssertionError("Validated record lost its position")
        position = record.position + 1
        entry = entries.get(record.protein_id)
        if not isinstance(entry, dict):
            reason = "metadata_not_cached"
            score = None
        elif entry.get("status") != "available":
            reason = "afdb_" + str(entry.get("status", "unknown"))
            score = None
        elif entry.get("sequence_sha256") != sequence_sha256(
            sequences[record.protein_id]
        ):
            raise ValueError(
                f"AFDB sequence changed for {record.protein_id}"
            )
        else:
            confidence = entry.get("site_confidence")
            if not isinstance(confidence, dict) or str(position) not in confidence:
                reason = "site_confidence_missing"
                score = None
            else:
                score = float(confidence[str(position)])
                reason = "eligible"
        reason_counts[reason] += 1
        reason_by_label[record.label][reason] += 1
        if score is None:
            if len(issue_examples) < 50:
                issue_examples.append(
                    {
                        "dataset_index": dataset_index,
                        "header": record.header,
                        "reason": reason,
                    }
                )
            continue
        eligible_indices.append(dataset_index)
        eligible_labels.append(record.label)
        accessions.append(record.protein_id)
        positions.append(position)
        site_plddt.append(score)
        global_plddt.append(float(entry["globalMetricValue"]))
        resolution_methods_by_site[
            str(entry.get("resolution_method", "phase1_direct_accession"))
        ] += 1
        score_by_label[record.label].append(score)
        category_counts[confidence_category(score)] += 1

    total_by_label = Counter(records[index].label for index in valid_indices)
    eligible_by_label = Counter(eligible_labels)
    coverage = len(eligible_indices) / max(len(valid_indices), 1)
    positive_coverage = eligible_by_label[1] / max(total_by_label[1], 1)
    negative_coverage = eligible_by_label[0] / max(total_by_label[0], 1)
    checks = {
        "site_coverage": coverage >= float(config["minimum_site_coverage"]),
        "positive_site_coverage": positive_coverage
        >= float(config["minimum_positive_site_coverage"]),
        "negative_site_coverage": negative_coverage
        >= float(config["minimum_negative_site_coverage"]),
    }
    training_authorized = all(checks.values())
    resolution_methods_by_accession = Counter(
        str(entry.get("resolution_method", "phase1_direct_accession"))
        for entry in entries.values()
        if isinstance(entry, dict) and entry.get("status") == "available"
    )
    high_confidence = float(config["high_confidence_plddt"])
    report = {
        "status": (
            "structure feasibility passed; residual screen may be designed"
            if training_authorized
            else "structure feasibility failed; do not train the residual"
        ),
        "training_authorized": training_authorized,
        "coverage_checks": checks,
        "coverage": {
            "context_valid_sites": len(valid_indices),
            "structure_eligible_sites": len(eligible_indices),
            "fraction": coverage,
            "positive": {
                "eligible": eligible_by_label[1],
                "total": total_by_label[1],
                "fraction": positive_coverage,
            },
            "negative": {
                "eligible": eligible_by_label[0],
                "total": total_by_label[0],
                "fraction": negative_coverage,
            },
            "eligible_unique_accessions": len(set(accessions)),
        },
        "site_plddt": {
            "overall": score_summary(site_plddt),
            "positive": score_summary(score_by_label[1]),
            "negative": score_summary(score_by_label[0]),
            "category_counts": dict(category_counts),
            "high_confidence_threshold": high_confidence,
            "fraction_at_least_high_confidence": (
                float(np.mean(np.asarray(site_plddt) >= high_confidence))
                if site_plddt
                else 0.0
            ),
        },
        "reason_counts": dict(reason_counts),
        "identifier_resolution": {
            "methods_by_site": dict(resolution_methods_by_site),
            "methods_by_unique_accession": dict(
                resolution_methods_by_accession
            ),
            "acceptance_rule": "exact complete sequence only",
        },
        "reason_counts_by_label": {
            "negative": dict(reason_by_label[0]),
            "positive": dict(reason_by_label[1]),
        },
        "issue_examples": issue_examples,
        "low_confidence_policy": config["low_confidence_policy"],
        "preprocessing": {
            name: vars(value) for name, value in preprocessing.items()
        },
        "context_validation": context_validation,
        "source_artifacts": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "sequence_cache": {
                "path": str(sequence_cache_path),
                "sha256": sequence_cache_sha256(sequence_cache_path),
            },
            "alphafold_metadata_cache": {
                "path": str(metadata_cache_path),
                "sha256": sha256_file(metadata_cache_path),
            },
        },
        "labels_used_for_model_fitting_or_selection": False,
        "released_independent_or_external_test_accessed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_features = features_path.with_name(features_path.name + ".tmp")
    with temporary_features.open("wb") as handle:
        np.savez_compressed(
            handle,
            dataset_indices=np.asarray(eligible_indices, dtype=np.int64),
            labels=np.asarray(eligible_labels, dtype=np.int64),
            accessions=np.asarray(accessions),
            one_based_positions=np.asarray(positions, dtype=np.int64),
            site_plddt=np.asarray(site_plddt, dtype=np.float32),
            global_plddt=np.asarray(global_plddt, dtype=np.float32),
        )
    os.replace(temporary_features, features_path)
    report["site_feature_artifact"] = {
        "path": str(features_path),
        "sha256": sha256_file(features_path),
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Released independent and external test sets were not accessed.")


if __name__ == "__main__":
    main()
