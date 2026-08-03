#!/usr/bin/env python3
"""Audit asymmetric label-noise evidence without fitting or scoring a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Hashable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.context import load_sequence_cache  # noqa: E402
from ubipred.external_benchmark import (  # noqa: E402
    ExternalSite,
    load_external_benchmark,
    validate_external_site,
)
from ubipred.fasta import (  # noqa: E402
    SiteRecord,
    center_crop,
    load_released_split,
)
def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def label_support(labels: Sequence[int]) -> dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {
        "negative": counts[0],
        "positive": counts[1],
        "total": len(labels),
    }


def conflicting_key_summary(
    records: Sequence[SiteRecord],
    key: Callable[[SiteRecord], Hashable | None],
) -> dict[str, object]:
    grouped: dict[Hashable, list[SiteRecord]] = defaultdict(list)
    for record in records:
        value = key(record)
        if value is not None:
            grouped[value].append(record)
    conflicts = {
        value: values
        for value, values in grouped.items()
        if len({record.label for record in values}) > 1
    }
    repeated = {
        value: values for value, values in grouped.items() if len(values) > 1
    }
    examples = []
    ordered_conflicts = sorted(
        conflicts.items(), key=lambda item: str(item[0])
    )
    for value, values in ordered_conflicts[:20]:
        examples.append(
            {
                "key": str(value),
                "records": [
                    {
                        "header": record.header,
                        "label": record.label,
                        "source": record.source,
                    }
                    for record in values[:10]
                ],
            }
        )
    return {
        "unique_keys": len(grouped),
        "repeated_keys": len(repeated),
        "conflicting_keys": len(conflicts),
        "records_on_conflicting_keys": sum(
            len(values) for values in conflicts.values()
        ),
        "examples": examples,
    }


def protein_label_composition(
    records: Sequence[SiteRecord],
) -> dict[str, object]:
    by_protein: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        by_protein[record.protein_id][record.label] += 1
    only_negative = {
        protein for protein, counts in by_protein.items() if counts[0] and not counts[1]
    }
    only_positive = {
        protein for protein, counts in by_protein.items() if counts[1] and not counts[0]
    }
    mixed = set(by_protein) - only_negative - only_positive
    return {
        "unique_proteins": len(by_protein),
        "negative_only_proteins": len(only_negative),
        "positive_only_proteins": len(only_positive),
        "mixed_label_proteins": len(mixed),
        "negative_records_on_mixed_label_proteins": sum(
            counts[0]
            for protein, counts in by_protein.items()
            if protein in mixed
        ),
        "positive_records_on_mixed_label_proteins": sum(
            counts[1]
            for protein, counts in by_protein.items()
            if protein in mixed
        ),
    }


def negative_to_positive_proximity(
    records: Sequence[SiteRecord],
) -> dict[str, object]:
    positive_positions: dict[str, list[int]] = defaultdict(list)
    for record in records:
        if record.label == 1 and record.position is not None:
            positive_positions[record.protein_id].append(record.position)
    distances = []
    missing_position = 0
    no_positive_on_protein = 0
    for record in records:
        if record.label != 0:
            continue
        if record.position is None:
            missing_position += 1
            continue
        candidates = positive_positions.get(record.protein_id)
        if not candidates:
            no_positive_on_protein += 1
            continue
        distances.append(min(abs(record.position - value) for value in candidates))
    thresholds = (1, 2, 5, 10, 25, 50, 100)
    return {
        "negative_records_with_positive_on_same_protein": len(distances),
        "negative_records_without_positive_on_same_protein": no_positive_on_protein,
        "negative_records_missing_position": missing_position,
        "nearest_positive_distance_cumulative": {
            f"at_most_{threshold}": sum(
                distance <= threshold for distance in distances
            )
            for threshold in thresholds
        },
        "median_nearest_distance": (
            sorted(distances)[len(distances) // 2] if distances else None
        ),
    }


def validate_external_records(
    records: Sequence[ExternalSite],
    sequences: dict[str, str],
    canonical_accessions: dict[str, str],
) -> tuple[list[tuple[ExternalSite, str, str]], dict[str, int]]:
    validated: list[tuple[ExternalSite, str, str]] = []
    reasons: Counter[str] = Counter()
    for record in records:
        result, reason = validate_external_site(
            record,
            sequences.get(record.source_protein_id),
            canonical_accessions.get(record.source_protein_id),
        )
        reasons[reason] += 1
        if result is not None:
            validated.append(
                (result.site, result.canonical_accession, result.window_49)
            )
    return validated, dict(sorted(reasons.items()))


def cross_source_disagreement(
    training_records: Sequence[SiteRecord],
    external_records: Sequence[tuple[ExternalSite, str, str]],
    *,
    training_label: int,
    external_label: int,
) -> dict[str, object]:
    selected_training = [
        record for record in training_records if record.label == training_label
    ]
    selected_external = [
        item for item in external_records if item[0].label == external_label
    ]
    external_sites = {
        (canonical_accession, site.position - 1)
        for site, canonical_accession, _ in selected_external
    }
    external_21mers = {site.window_21 for site, _, _ in selected_external}
    external_49mers = {window_49 for _, _, window_49 in selected_external}
    site_matches = [
        record
        for record in selected_training
        if record.position is not None
        and (record.protein_id, record.position) in external_sites
    ]
    window_21_matches = [
        record
        for record in selected_training
        if center_crop(record.sequence, 21) in external_21mers
    ]
    window_49_matches = [
        record
        for record in selected_training
        if record.sequence in external_49mers
    ]
    example_records = []
    for record in site_matches[:20]:
        example_records.append(
            {
                "training_header": record.header,
                "training_label": record.label,
                "external_label": external_label,
            }
        )
    return {
        "training_label": training_label,
        "external_label": external_label,
        "training_records_considered": len(selected_training),
        "validated_external_records_considered": len(selected_external),
        "exact_site_matches": len(site_matches),
        "unique_exact_sites": len(
            {(record.protein_id, record.position) for record in site_matches}
        ),
        "exact_21mer_matches": len(window_21_matches),
        "exact_49mer_matches": len(window_49_matches),
        "fraction_of_training_class_with_exact_site_conflict": (
            len(site_matches) / max(len(selected_training), 1)
        ),
        "exact_site_examples": example_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="replication/MMUbiPred")
    parser.add_argument("--external-benchmark", type=Path, required=True)
    parser.add_argument("--external-sequence-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_dir = resolve_project_path(args.data_dir)
    external_benchmark_path = resolve_project_path(args.external_benchmark)
    external_cache_path = resolve_project_path(args.external_sequence_cache)
    output_path = resolve_project_path(args.output)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed label-noise audit: {output_path}"
        )
    train, train_preprocessing = load_released_split(
        data_dir, split="train", window_size=49
    )
    external = load_external_benchmark(external_benchmark_path)
    sequences, external_metadata = load_sequence_cache(external_cache_path)
    canonical_accessions = external_metadata.get("canonical_accessions")
    if not isinstance(canonical_accessions, dict):
        raise ValueError("External cache lacks canonical accession mappings")
    canonical_accessions = {
        str(key): str(value) for key, value in canonical_accessions.items()
    }
    validated_external, validation_reasons = validate_external_records(
        external, sequences, canonical_accessions
    )

    audit = {
        "status": "training-label noise audit completed; no model fitted",
        "scope": {
            "model_predictions_used": False,
            "historical_test_labels_used_for_method_selection": False,
            "external_labels_used_only_for_cross_source_conflict_audit": True,
            "audit_does_not_prove_unobserved_negatives_are_true_positives": True,
        },
        "released_training": {
            "support": label_support([record.label for record in train]),
            "preprocessing": {
                key: vars(value) for key, value in train_preprocessing.items()
            },
            "exact_key_audits": {
                "site": conflicting_key_summary(
                    train,
                    lambda record: (
                        (record.protein_id, record.position)
                        if record.position is not None
                        else None
                    ),
                ),
                "window_21": conflicting_key_summary(
                    train, lambda record: center_crop(record.sequence, 21)
                ),
                "window_49": conflicting_key_summary(
                    train, lambda record: record.sequence
                ),
            },
            "protein_label_composition": protein_label_composition(train),
            "negative_to_positive_proximity": negative_to_positive_proximity(
                train
            ),
            "terminal_padding": {
                "negative_records": sum(
                    record.label == 0 and "-" in record.sequence
                    for record in train
                ),
                "positive_records": sum(
                    record.label == 1 and "-" in record.sequence
                    for record in train
                ),
            },
        },
        "released_independent_test_accessed": False,
        "external_cross_source_evidence": {
            "source": {
                "benchmark_path": str(external_benchmark_path),
                "benchmark_sha256": sha256_file(external_benchmark_path),
                "sequence_cache_path": str(external_cache_path),
                "sequence_cache_sha256": sha256_file(external_cache_path),
            },
            "raw_support": label_support([record.label for record in external]),
            "validated_support": label_support(
                [site.label for site, _, _ in validated_external]
            ),
            "validation_reason_counts": validation_reasons,
            "training_negative_vs_external_positive": (
                cross_source_disagreement(
                    train,
                    validated_external,
                    training_label=0,
                    external_label=1,
                )
            ),
            "training_positive_vs_external_negative": (
                cross_source_disagreement(
                    train,
                    validated_external,
                    training_label=1,
                    external_label=0,
                )
            ),
        },
        "artifacts": {
            path.name: sha256_file(path)
            for path in (
                data_dir / "Positive_90_percent_training_set_DeepUBI.fasta",
                data_dir / "Negative_90_percent_training_set_DeepUBI.fasta",
            )
        },
    }
    write_json(output_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
