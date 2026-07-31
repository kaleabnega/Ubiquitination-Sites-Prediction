#!/usr/bin/env python3
"""Create and freeze a leakage-audited external dbPTM/PTMGPT2 cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.context import load_sequence_cache, sequence_cache_sha256  # noqa: E402
from ubipred.external_benchmark import (  # noqa: E402
    build_released_leakage_sets,
    label_counts,
    leakage_reasons,
    load_external_benchmark,
    parse_mmseqs_homology_hit,
    sha256_file,
    validate_external_site,
)
from ubipred.fasta import load_released_split  # noqa: E402


EXPECTED_BENCHMARK_SHA256 = (
    "c303df3ab199b3aa3d4c754aa3cd255343740298c11b1c03f55b8f2c4474ab9f"
)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_fasta(path: Path, sequences: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for accession in sorted(sequences):
            handle.write(f">{accession}\n")
            sequence = sequences[accession]
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def hash_string_values(values: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(values) + "\n").encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--benchmark", default="replication/MMUbiPred/benchmark.csv"
    )
    parser.add_argument("--external-sequence-cache", type=Path, required=True)
    parser.add_argument("--training-sequence-cache", type=Path, required=True)
    parser.add_argument("--released-model", type=Path, required=True)
    parser.add_argument("--context-run-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir", default="replication/MMUbiPred"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mmseqs", default="mmseqs")
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    benchmark_path = resolve_project_path(args.benchmark)
    external_cache_path = resolve_project_path(args.external_sequence_cache)
    training_cache_path = resolve_project_path(args.training_sequence_cache)
    released_model_path = resolve_project_path(args.released_model)
    context_run_dir = resolve_project_path(args.context_run_dir)
    data_dir = resolve_project_path(args.data_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "external_cohort.tsv"
    lock_path = output_dir / "cohort_lock.json"
    if final_path.exists() or lock_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an already frozen external cohort"
        )
    if sha256_file(benchmark_path) != EXPECTED_BENCHMARK_SHA256:
        raise ValueError("External benchmark checksum does not match protocol")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["source_benchmark_sha256"] != EXPECTED_BENCHMARK_SHA256:
        raise ValueError("Configured benchmark checksum changed")
    if (
        sha256_file(released_model_path)
        != config["released_mmubipred_sha256"]
    ):
        raise ValueError("Released MMUbiPred checksum does not match protocol")
    context_artifacts = {
        name: context_run_dir / name
        for name in ("best.pt", "refit_manifest.json", "refit_summary.json")
    }
    for path in context_artifacts.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen context artifact: {path}")
    protocol = dict(config["cohort_protocol"])
    if (
        float(protocol["maximum_sequence_identity"]) != 0.30
        or float(protocol["minimum_shorter_sequence_coverage"]) != 0.80
        or not bool(protocol["exclude_training_accessions"])
        or not bool(protocol["exclude_historical_test_accessions"])
        or not bool(protocol["exclude_exact_released_windows"])
        or protocol["homology_tool"] != "MMseqs2"
    ):
        raise ValueError("Cohort protocol differs from the frozen contract")

    benchmark = load_external_benchmark(benchmark_path)
    external_sequences, external_metadata = load_sequence_cache(
        external_cache_path
    )
    external_canonical_accessions = external_metadata.get(
        "canonical_accessions"
    )
    if not isinstance(external_canonical_accessions, dict):
        raise ValueError(
            "External sequence cache lacks canonical accession mappings"
        )
    training_sequences, training_metadata = load_sequence_cache(
        training_cache_path
    )
    train_records, train_preprocessing = load_released_split(
        data_dir, split="train", window_size=49
    )
    test_records, test_preprocessing = load_released_split(
        data_dir, split="test", window_size=49
    )
    leakage_sets = build_released_leakage_sets(
        train_records, test_records
    )

    validation_reasons: Counter[str] = Counter()
    leakage_reason_counts: Counter[str] = Counter()
    validated = []
    pre_homology = []
    for site in benchmark:
        record, reason = validate_external_site(
            site,
            external_sequences.get(site.source_protein_id),
            external_canonical_accessions.get(site.source_protein_id),
            context_window_size=int(protocol["context_window_size"]),
        )
        validation_reasons[reason] += 1
        if record is None:
            continue
        validated.append(record)
        reasons = leakage_reasons(record, leakage_sets)
        leakage_reason_counts.update(reasons)
        if not reasons:
            pre_homology.append(record)

    candidate_accessions = {
        record.canonical_accession for record in pre_homology
    }
    candidate_sequences: dict[str, str] = {}
    for record in pre_homology:
        sequence = external_sequences[record.site.source_protein_id]
        previous = candidate_sequences.setdefault(
            record.canonical_accession, sequence
        )
        if previous != sequence:
            raise ValueError(
                "One canonical accession resolved to conflicting sequences: "
                f"{record.canonical_accession}"
            )
    reference_accessions = {
        record.protein_id for record in train_records
    }
    reference_sequences = {
        accession: training_sequences[accession]
        for accession in reference_accessions
        if accession in training_sequences
    }
    mapping_fraction = len(validated) / len(benchmark)
    if mapping_fraction < float(protocol["minimum_mapping_fraction"]):
        raise RuntimeError(
            f"Only {mapping_fraction:.4f} of benchmark sites mapped; "
            "below the frozen minimum"
        )
    reference_fraction = len(reference_sequences) / len(reference_accessions)
    if reference_fraction < float(
        protocol["minimum_reference_sequence_fraction"]
    ):
        raise RuntimeError(
            f"Only {reference_fraction:.4f} of PLMD training proteins have "
            "reference sequences; below the frozen minimum"
        )
    reference_fasta = output_dir / "plmd_training_proteins.fasta"
    candidate_fasta = output_dir / "pre_homology_external_proteins.fasta"
    nonhomologous_fasta = output_dir / "nonhomologous_external_proteins.fasta"
    raw_alignments_path = output_dir / "mmseqs_alignments.tsv"
    qualifying_alignments_path = (
        output_dir / "qualifying_homology_alignments.tsv"
    )
    write_fasta(reference_fasta, reference_sequences)
    write_fasta(candidate_fasta, candidate_sequences)

    search_identity = float(
        protocol["homology_search_minimum_local_identity"]
    )
    sensitivity = float(protocol["homology_search_sensitivity"])
    maximum_evalue = float(protocol["homology_search_maximum_evalue"])
    with tempfile.TemporaryDirectory(prefix="ubipred_mmseqs_") as temporary:
        temporary_root = Path(temporary)
        temporary_alignments = temporary_root / "alignments.tsv"
        temporary_work = temporary_root / "work"
        command = [
            args.mmseqs,
            "easy-search",
            str(candidate_fasta),
            str(reference_fasta),
            str(temporary_alignments),
            str(temporary_work),
            "--min-seq-id",
            str(search_identity),
            "-s",
            str(sensitivity),
            "-e",
            str(maximum_evalue),
            "--max-seqs",
            "20000",
            "--format-output",
            "query,target,nident,qcov,tcov,qlen,tlen,evalue",
        ]
        print("$", " ".join(command), flush=True)
        # Inherit stdout/stderr so Colab receives MMseqs2 progress and any
        # failure diagnostic immediately through the notebook's live runner.
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise RuntimeError(
                "MMseqs2 homology search failed with exit status "
                f"{completed.returncode}"
            )
        shutil.copyfile(temporary_alignments, raw_alignments_path)

    minimum_global_identity = float(protocol["maximum_sequence_identity"])
    minimum_shorter_coverage = float(
        protocol["minimum_shorter_sequence_coverage"]
    )
    raw_alignment_count = 0
    qualifying_alignment_count = 0
    homologous_accessions: set[str] = set()
    with raw_alignments_path.open("r", encoding="utf-8") as source, (
        qualifying_alignments_path.open("w", encoding="utf-8")
    ) as destination:
        destination.write(
            "query\ttarget\tidentical_residues\tquery_coverage\t"
            "target_coverage\tquery_length\ttarget_length\tevalue\t"
            "global_identity_to_shorter\tshorter_sequence_coverage\n"
        )
        for line in source:
            if not line.strip():
                continue
            raw_alignment_count += 1
            hit = parse_mmseqs_homology_hit(line)
            if (
                hit.global_identity_to_shorter
                < minimum_global_identity
                or hit.shorter_sequence_coverage
                < minimum_shorter_coverage
            ):
                continue
            qualifying_alignment_count += 1
            homologous_accessions.add(hit.query)
            destination.write(
                f"{hit.query}\t{hit.target}\t{hit.identical_residues}\t"
                f"{hit.query_coverage}\t{hit.target_coverage}\t"
                f"{hit.query_length}\t{hit.target_length}\t{hit.evalue}\t"
                f"{hit.global_identity_to_shorter}\t"
                f"{hit.shorter_sequence_coverage}\n"
            )
    retained_accessions = candidate_accessions - homologous_accessions
    write_fasta(
        nonhomologous_fasta,
        {
            accession: candidate_sequences[accession]
            for accession in retained_accessions
        },
    )
    final_records = [
        record
        for record in pre_homology
        if record.canonical_accession in retained_accessions
    ]
    if not final_records:
        raise RuntimeError("Homology filtering removed the entire cohort")
    minimum_size = int(protocol["minimum_final_sites"])
    if len(final_records) < minimum_size:
        raise RuntimeError(
            f"Final cohort has {len(final_records)} sites; "
            f"protocol requires at least {minimum_size}"
        )
    final_support = label_counts(final_records)
    minimum_per_class = int(protocol["minimum_final_sites_per_class"])
    if min(final_support["negative"], final_support["positive"]) < minimum_per_class:
        raise RuntimeError(
            f"Final class support {final_support} is below the frozen "
            f"minimum of {minimum_per_class} per class"
        )

    fieldnames = [
        "benchmark_index",
        "identifier",
        "source_protein_id",
        "canonical_accession",
        "species",
        "position_one_based",
        "label",
        "window_21",
        "window_49",
        "context_257",
        "protein_sequence_sha256",
    ]
    with final_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in final_records:
            site = record.site
            writer.writerow(
                {
                    "benchmark_index": site.benchmark_index,
                    "identifier": site.identifier,
                    "source_protein_id": site.source_protein_id,
                    "canonical_accession": record.canonical_accession,
                    "species": site.species,
                    "position_one_based": site.position,
                    "label": site.label,
                    "window_21": site.window_21,
                    "window_49": record.window_49,
                    "context_257": record.context_257,
                    "protein_sequence_sha256": hashlib.sha256(
                        external_sequences[site.source_protein_id].encode(
                            "ascii"
                        )
                    ).hexdigest(),
                }
            )

    tool_version = subprocess.run(
        [args.mmseqs, "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    final_accessions = sorted(retained_accessions)
    lock = {
        "status": "external cohort frozen; model inference not performed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "external_predictions_generated": False,
        "labels_used_for_model_or_threshold_selection": False,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": sha256_file(benchmark_path),
            "raw_sites": len(benchmark),
        },
        "sequence_caches": {
            "external": {
                "path": str(external_cache_path),
                "sha256": sequence_cache_sha256(external_cache_path),
                "metadata": external_metadata,
            },
            "plmd_training": {
                "path": str(training_cache_path),
                "sha256": sequence_cache_sha256(training_cache_path),
                "metadata": training_metadata,
            },
        },
        "frozen_model_artifacts": {
            "released_mmubipred": {
                "path": str(released_model_path),
                "sha256": sha256_file(released_model_path),
            },
            "context_checkpoint": {
                "path": str(context_artifacts["best.pt"]),
                "sha256": sha256_file(context_artifacts["best.pt"]),
            },
            "context_refit_manifest": {
                "path": str(context_artifacts["refit_manifest.json"]),
                "sha256": sha256_file(
                    context_artifacts["refit_manifest.json"]
                ),
            },
            "context_refit_summary": {
                "path": str(context_artifacts["refit_summary.json"]),
                "sha256": sha256_file(
                    context_artifacts["refit_summary.json"]
                ),
            },
        },
        "released_preprocessing": {
            "training": {
                key: vars(value)
                for key, value in train_preprocessing.items()
            },
            "historical_test": {
                key: vars(value)
                for key, value in test_preprocessing.items()
            },
        },
        "validation": {
            "validated_before_leakage_filter": len(validated),
            "validated_fraction": mapping_fraction,
            "reason_counts": dict(sorted(validation_reasons.items())),
        },
        "exact_leakage_filter": {
            "sites_before_filter": len(validated),
            "sites_after_filter": len(pre_homology),
            "reason_counts": dict(sorted(leakage_reason_counts.items())),
        },
        "homology_filter": {
            "tool": "MMseqs2",
            "command": command,
            "version_output_first_line": (
                (tool_version.stdout or tool_version.stderr)
                .strip()
                .splitlines()[0]
                if (tool_version.stdout or tool_version.stderr).strip()
                else "unknown"
            ),
            "search_minimum_local_identity": search_identity,
            "search_sensitivity": sensitivity,
            "search_maximum_evalue": maximum_evalue,
            "global_identity_definition": (
                "identical aligned residues divided by shorter full-protein "
                "length"
            ),
            "minimum_global_identity": minimum_global_identity,
            "minimum_shorter_sequence_coverage": minimum_shorter_coverage,
            "raw_alignment_count": raw_alignment_count,
            "qualifying_alignment_count": qualifying_alignment_count,
            "homologous_external_accessions": len(homologous_accessions),
            "raw_alignments": {
                "path": str(raw_alignments_path),
                "sha256": sha256_file(raw_alignments_path),
            },
            "qualifying_alignments": {
                "path": str(qualifying_alignments_path),
                "sha256": sha256_file(qualifying_alignments_path),
            },
            "reference_unique_accessions": len(reference_accessions),
            "reference_sequences_available": len(reference_sequences),
            "reference_sequence_fraction": reference_fraction,
            "candidate_unique_accessions": len(candidate_accessions),
            "retained_unique_accessions": len(retained_accessions),
            "retained_accessions_sha256": hash_string_values(
                final_accessions
            ),
        },
        "final_cohort": {
            "path": str(final_path),
            "sha256": sha256_file(final_path),
            "support": final_support,
            "unique_accessions": len(
                {record.canonical_accession for record in final_records}
            ),
            "benchmark_indices_sha256": hashlib.sha256(
                ",".join(
                    str(record.site.benchmark_index)
                    for record in final_records
                ).encode("ascii")
            ).hexdigest(),
        },
        "frozen_evaluation": config["frozen_evaluation"],
    }
    write_json(lock_path, lock)
    print(json.dumps(lock, indent=2, sort_keys=True))
    print("External cohort frozen. No model predictions were generated.")


if __name__ == "__main__":
    main()
