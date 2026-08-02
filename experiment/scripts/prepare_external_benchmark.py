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
    canonical_line_sha256,
    label_counts,
    leakage_reasons,
    load_external_benchmark,
    parse_mmseqs_homology_hit,
    sha256_file,
    validate_external_site,
)
from ubipred.fasta import load_released_split  # noqa: E402


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
    parser.add_argument("--feasibility-reference-dir", type=Path)
    parser.add_argument("--prior-external-benchmark", type=Path)
    parser.add_argument("--prior-external-sequence-cache", type=Path)
    parser.add_argument("--prior-external-cohort-dir", type=Path)
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
    feasibility_reference_dir = (
        resolve_project_path(args.feasibility_reference_dir)
        if args.feasibility_reference_dir is not None
        else None
    )
    prior_external_benchmark_path = (
        resolve_project_path(args.prior_external_benchmark)
        if args.prior_external_benchmark is not None
        else None
    )
    prior_external_cache_path = (
        resolve_project_path(args.prior_external_sequence_cache)
        if args.prior_external_sequence_cache is not None
        else None
    )
    prior_external_cohort_dir = (
        resolve_project_path(args.prior_external_cohort_dir)
        if args.prior_external_cohort_dir is not None
        else None
    )
    data_dir = resolve_project_path(args.data_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "external_cohort.tsv"
    lock_path = output_dir / "cohort_lock.json"
    if final_path.exists() or lock_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an already frozen external cohort"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_benchmark_sha256 = str(config["source_benchmark_sha256"])
    if sha256_file(benchmark_path) != expected_benchmark_sha256:
        raise ValueError("External benchmark checksum does not match protocol")
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
        or not bool(protocol.get("require_alignment_backtrace", False))
    ):
        raise ValueError("Cohort protocol differs from the frozen contract")
    configured_amendment = config.get("feasibility_amendment")
    if configured_amendment is not None and (
        bool(
            configured_amendment[
                "external_predictions_inspected_before_amendment"
            ]
        )
        or int(configured_amendment["amended_minimum_final_sites"])
        != int(protocol["minimum_final_sites"])
        or int(
            configured_amendment[
                "amended_minimum_final_sites_per_class"
            ]
        )
        != int(protocol["minimum_final_sites_per_class"])
    ):
        raise ValueError("Feasibility amendment differs from the protocol")

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

    exclude_prior_external = bool(
        protocol.get("exclude_prior_external_source", False)
    )
    prior_external_sequences: dict[str, str] = {}
    prior_external_metadata: dict[str, object] = {}
    prior_canonical_map: dict[str, object] = {}
    prior_source_protein_ids: set[str] = set()
    prior_identifiers: set[str] = set()
    prior_windows_21: set[str] = set()
    prior_canonical_accessions: set[str] = set()
    prior_cohort_accessions: set[str] = set()
    prior_cohort_sites: set[tuple[str, int]] = set()
    prior_cohort_windows_49: set[str] = set()
    if exclude_prior_external:
        if (
            prior_external_benchmark_path is None
            or prior_external_cache_path is None
            or prior_external_cohort_dir is None
        ):
            raise ValueError(
                "Prior external benchmark, sequence cache, and cohort are "
                "required by this protocol"
            )
        prior_records = load_external_benchmark(
            prior_external_benchmark_path
        )
        prior_source_protein_ids = {
            record.source_protein_id for record in prior_records
        }
        prior_identifiers = {record.identifier for record in prior_records}
        prior_windows_21 = {record.window_21 for record in prior_records}
        (
            prior_external_sequences,
            prior_external_metadata,
        ) = load_sequence_cache(prior_external_cache_path)
        raw_prior_canonical_map = prior_external_metadata.get(
            "canonical_accessions"
        )
        if not isinstance(raw_prior_canonical_map, dict):
            raise ValueError(
                "Prior external cache lacks canonical accession mappings"
            )
        prior_canonical_map = raw_prior_canonical_map
        prior_canonical_accessions = {
            str(accession) for accession in prior_canonical_map.values()
        }
        prior_cohort_path = (
            prior_external_cohort_dir / "external_cohort.tsv"
        )
        with prior_cohort_path.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            prior_cohort_rows = list(
                csv.DictReader(handle, delimiter="\t")
            )
        if not prior_cohort_rows:
            raise ValueError("Prior external cohort is empty")
        prior_cohort_accessions = {
            row["canonical_accession"] for row in prior_cohort_rows
        }
        prior_cohort_sites = {
            (row["canonical_accession"], int(row["position_one_based"]))
            for row in prior_cohort_rows
        }
        prior_cohort_windows_49 = {
            row["window_49"] for row in prior_cohort_rows
        }

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
        if exclude_prior_external:
            site = record.site
            if site.source_protein_id in prior_source_protein_ids:
                reasons.append("prior_external_source_protein_overlap")
            if site.identifier in prior_identifiers:
                reasons.append("prior_external_identifier_overlap")
            if site.window_21 in prior_windows_21:
                reasons.append("prior_external_21mer_overlap")
            if record.canonical_accession in prior_canonical_accessions:
                reasons.append("prior_external_canonical_accession_overlap")
            if record.canonical_accession in prior_cohort_accessions:
                reasons.append("prior_frozen_cohort_accession_overlap")
            if (
                record.canonical_accession,
                site.position,
            ) in prior_cohort_sites:
                reasons.append("prior_frozen_cohort_site_overlap")
            if record.window_49 in prior_cohort_windows_49:
                reasons.append("prior_frozen_cohort_49mer_overlap")
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
    homology_reference_sequences = dict(reference_sequences)
    if exclude_prior_external:
        for source_protein_id, sequence in prior_external_sequences.items():
            canonical_accession = prior_canonical_map.get(source_protein_id)
            if canonical_accession is None:
                continue
            canonical_accession = str(canonical_accession)
            previous = homology_reference_sequences.setdefault(
                canonical_accession, sequence
            )
            if previous != sequence:
                raise ValueError(
                    "Conflicting homology-reference sequences for "
                    f"{canonical_accession}"
                )
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
    reference_fasta = output_dir / "homology_reference_proteins.fasta"
    candidate_fasta = output_dir / "pre_homology_external_proteins.fasta"
    nonhomologous_fasta = output_dir / "nonhomologous_external_proteins.fasta"
    raw_alignments_path = output_dir / "mmseqs_alignments.tsv"
    qualifying_alignments_path = (
        output_dir / "qualifying_homology_alignments.tsv"
    )
    write_fasta(reference_fasta, homology_reference_sequences)
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
            "-a",
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
    maximum_identical_residues = 0
    maximum_global_identity_observed = 0.0
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
            maximum_identical_residues = max(
                maximum_identical_residues, hit.identical_residues
            )
            maximum_global_identity_observed = max(
                maximum_global_identity_observed,
                hit.global_identity_to_shorter,
            )
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
    if raw_alignment_count and maximum_identical_residues == 0:
        raise RuntimeError(
            "MMseqs2 reported alignments but every nident value was zero. "
            "Refusing to freeze the cohort because alignment-derived "
            "identity statistics are invalid. Verify that the search used "
            "the -a alignment-backtrace option."
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
    final_support = label_counts(final_records)
    minimum_size = int(protocol["minimum_final_sites"])
    minimum_per_class = int(protocol["minimum_final_sites_per_class"])
    minimum_unique_accessions = int(
        protocol.get("minimum_final_unique_accessions", 1)
    )
    final_unique_accessions = len(
        {record.canonical_accession for record in final_records}
    )
    feasibility_path = output_dir / "cohort_feasibility.json"
    feasibility = {
        "status": "pre-inference cohort feasibility audit; cohort not frozen",
        "external_predictions_generated": False,
        "labels_used_for_model_or_threshold_selection": False,
        "validated_sites": len(validated),
        "sites_after_exact_leakage_filter": len(pre_homology),
        "homology_filter": {
            "alignment_backtrace_requested": True,
            "raw_alignment_count": raw_alignment_count,
            "maximum_identical_residues": maximum_identical_residues,
            "maximum_global_identity_observed": (
                maximum_global_identity_observed
            ),
            "qualifying_alignment_count": qualifying_alignment_count,
            "homologous_external_accessions": len(homologous_accessions),
            "retained_unique_accessions": len(retained_accessions),
        },
        "prospective_final_cohort": {
            "sites": len(final_records),
            "unique_accessions": final_unique_accessions,
            "support": final_support,
        },
        "configured_minimums": {
            "sites": minimum_size,
            "sites_per_class": minimum_per_class,
            "unique_accessions": minimum_unique_accessions,
        },
        "passes_configured_minimums": (
            len(final_records) >= minimum_size
            and final_support["negative"] >= minimum_per_class
            and final_support["positive"] >= minimum_per_class
            and final_unique_accessions >= minimum_unique_accessions
        ),
        "artifacts": {
            "raw_alignments_sha256": sha256_file(raw_alignments_path),
            "qualifying_alignments_sha256": sha256_file(
                qualifying_alignments_path
            ),
        },
    }
    amendment = config.get("feasibility_amendment")
    if amendment is not None:
        if feasibility_reference_dir is None:
            raise ValueError(
                "The amended protocol requires --feasibility-reference-dir"
            )
        reference_raw = feasibility_reference_dir / "mmseqs_alignments.tsv"
        reference_qualifying = (
            feasibility_reference_dir / "qualifying_homology_alignments.tsv"
        )
        for reference_path in (reference_raw, reference_qualifying):
            if not reference_path.exists():
                raise FileNotFoundError(
                    f"Missing v2 feasibility artifact: {reference_path}"
                )
        if (
            sha256_file(reference_raw)
            != amendment["v2_raw_alignments_sha256"]
            or sha256_file(reference_qualifying)
            != amendment["v2_qualifying_alignments_sha256"]
        ):
            raise RuntimeError(
                "The preserved v2 feasibility artifacts changed"
            )
        reference_canonical_hashes = {
            "raw_alignments": canonical_line_sha256(reference_raw),
            "qualifying_alignments": canonical_line_sha256(
                reference_qualifying
            ),
        }
        observed_canonical_hashes = {
            "raw_alignments": canonical_line_sha256(raw_alignments_path),
            "qualifying_alignments": canonical_line_sha256(
                qualifying_alignments_path
            ),
        }
        if observed_canonical_hashes != reference_canonical_hashes:
            raise RuntimeError(
                "V3 alignment content differs from the preserved v2 "
                "feasibility audit; refusing to freeze"
            )
        observed_amendment_basis = {
            "prospective_sites": len(final_records),
            "prospective_negative_sites": final_support["negative"],
            "prospective_positive_sites": final_support["positive"],
            "prospective_unique_accessions": feasibility[
                "prospective_final_cohort"
            ]["unique_accessions"],
        }
        expected_amendment_basis = {
            key: amendment[key] for key in observed_amendment_basis
        }
        if observed_amendment_basis != expected_amendment_basis:
            raise RuntimeError(
                "Observed cohort differs from the frozen feasibility "
                "amendment; refusing to freeze v3"
            )
        feasibility["feasibility_amendment_verification"] = {
            "verified": True,
            "reference_directory": str(feasibility_reference_dir),
            "reference_byte_hashes_match_v2": True,
            "order_independent_alignment_hashes": (
                observed_canonical_hashes
            ),
            "prospective_counts_match_v2": True,
        }
    write_json(feasibility_path, feasibility)
    print(json.dumps(feasibility, indent=2, sort_keys=True), flush=True)
    if not final_records:
        raise RuntimeError("Homology filtering removed the entire cohort")
    if len(final_records) < minimum_size:
        raise RuntimeError(
            f"Final cohort has {len(final_records)} sites; "
            f"protocol requires at least {minimum_size}"
        )
    if min(final_support["negative"], final_support["positive"]) < minimum_per_class:
        raise RuntimeError(
            f"Final class support {final_support} is below the frozen "
            f"minimum of {minimum_per_class} per class"
        )
    if final_unique_accessions < minimum_unique_accessions:
        raise RuntimeError(
            f"Final cohort has {final_unique_accessions} unique proteins; "
            f"protocol requires at least {minimum_unique_accessions}"
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
            **(
                {
                    "prior_external": {
                        "path": str(prior_external_cache_path),
                        "sha256": sequence_cache_sha256(
                            prior_external_cache_path
                        ),
                        "metadata": prior_external_metadata,
                    }
                }
                if exclude_prior_external
                and prior_external_cache_path is not None
                else {}
            ),
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
            "prior_external_source_excluded": exclude_prior_external,
            **(
                {
                    "prior_external_benchmark": {
                        "path": str(prior_external_benchmark_path),
                        "sha256": sha256_file(
                            prior_external_benchmark_path
                        ),
                        "raw_sites": len(prior_records),
                    },
                    "prior_external_frozen_cohort": {
                        "path": str(prior_cohort_path),
                        "sha256": sha256_file(prior_cohort_path),
                        "sites": len(prior_cohort_rows),
                    },
                }
                if exclude_prior_external
                and prior_external_benchmark_path is not None
                else {}
            ),
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
            "alignment_backtrace_requested": True,
            "global_identity_definition": (
                "identical aligned residues divided by shorter full-protein "
                "length"
            ),
            "minimum_global_identity": minimum_global_identity,
            "minimum_shorter_sequence_coverage": minimum_shorter_coverage,
            "raw_alignment_count": raw_alignment_count,
            "maximum_identical_residues": maximum_identical_residues,
            "maximum_global_identity_observed": (
                maximum_global_identity_observed
            ),
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
            "homology_reference_unique_accessions": len(
                homology_reference_sequences
            ),
            "prior_external_homology_references_added": (
                len(homology_reference_sequences)
                - len(reference_sequences)
            ),
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
        "feasibility_amendment": configured_amendment,
    }
    write_json(lock_path, lock)
    print(json.dumps(lock, indent=2, sort_keys=True))
    print("External cohort frozen. No model predictions were generated.")


if __name__ == "__main__":
    main()
