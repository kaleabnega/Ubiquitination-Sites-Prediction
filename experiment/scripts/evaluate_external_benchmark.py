#!/usr/bin/env python3
"""One-time evaluation of frozen models on the external dbPTM benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.context import LongContextSiteDataset  # noqa: E402
from ubipred.data import (  # noqa: E402
    TOKEN_TO_INDEX,
    load_normalized_aaindex,
    make_loader,
)
from ubipred.engine import load_checkpoint_model_state, predict  # noqa: E402
from ubipred.ensemble import equal_probability_average  # noqa: E402
from ubipred.external_benchmark import sha256_file  # noqa: E402
from ubipred.fasta import SiteRecord  # noqa: E402
from ubipred.metrics import compute_metrics, write_json  # noqa: E402
from ubipred.model import build_model  # noqa: E402
from ubipred.paired_statistics import (  # noqa: E402
    mcnemar_exact,
    paired_cluster_bootstrap,
)


COMPARISON_METRICS = (
    "mcc",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "auroc",
    "auprc",
)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_cohort(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    if not records:
        raise ValueError("Frozen external cohort is empty")
    required = {
        "benchmark_index",
        "identifier",
        "canonical_accession",
        "position_one_based",
        "label",
        "window_49",
        "context_257",
    }
    if not required.issubset(records[0]):
        raise ValueError("Frozen cohort columns are incomplete")
    return records


def metric_differences(
    candidate: dict[str, object], reference: dict[str, object]
) -> dict[str, float]:
    return {
        metric: float(candidate[metric]) - float(reference[metric])
        for metric in COMPARISON_METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--context-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir", default="replication/MMUbiPred"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--allow-external-test",
        action="store_true",
        help="Required acknowledgement for one-time external inference.",
    )
    args = parser.parse_args()
    if not args.allow_external_test:
        parser.error(
            "Refusing external inference without --allow-external-test"
        )

    config_path = resolve_project_path(args.config)
    cohort_dir = resolve_project_path(args.cohort_dir)
    model_path = resolve_project_path(args.model_file)
    context_run_dir = resolve_project_path(args.context_run_dir)
    output_dir = resolve_project_path(args.output_dir)
    data_dir = resolve_project_path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "external_test_metrics.json"
    predictions_path = output_dir / "external_test_predictions.npz"
    predictions_table_path = output_dir / "external_test_predictions.tsv"
    manifest_path = output_dir / "external_test_manifest.json"
    if results_path.exists():
        raise FileExistsError(
            f"Refusing to repeat completed external evaluation: {results_path}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    evaluation = dict(config["frozen_evaluation"])
    if (
        evaluation["fusion"]
        != "equal arithmetic mean of positive-class probabilities"
        or float(evaluation["paper_weight"]) != 0.5
        or float(evaluation["context_weight"]) != 0.5
        or float(evaluation["primary_threshold"]) != 0.5
        or evaluation.get("bootstrap_unit") != "canonical_accession"
        or not bool(evaluation["external_labels_may_not_select_weights_or_threshold"])
    ):
        raise ValueError("Frozen external evaluation contract changed")

    lock_path = cohort_dir / "cohort_lock.json"
    cohort_path = cohort_dir / "external_cohort.tsv"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("external_predictions_generated") is not False:
        raise ValueError("Cohort lock is not pre-inference")
    if lock["config"]["sha256"] != sha256_file(config_path):
        raise ValueError("Config changed after cohort lock")
    if lock["final_cohort"]["sha256"] != sha256_file(cohort_path):
        raise ValueError("Frozen cohort checksum changed")
    cohort = load_cohort(cohort_path)
    if len(cohort) != int(lock["final_cohort"]["support"]["total"]):
        raise ValueError("Frozen cohort row count changed")

    expected_model_hash = str(config["released_mmubipred_sha256"])
    if sha256_file(model_path) != expected_model_hash:
        raise ValueError("Released MMUbiPred model checksum mismatch")
    frozen_artifacts = lock.get("frozen_model_artifacts")
    if not isinstance(frozen_artifacts, dict):
        raise ValueError("Cohort lock does not freeze model artifacts")
    if (
        frozen_artifacts["released_mmubipred"]["sha256"]
        != expected_model_hash
    ):
        raise ValueError("Released model differs from the cohort lock")
    context_checkpoint_path = context_run_dir / "best.pt"
    context_manifest_path = context_run_dir / "refit_manifest.json"
    context_summary_path = context_run_dir / "refit_summary.json"
    for required in (
        context_checkpoint_path,
        context_manifest_path,
        context_summary_path,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Missing context artifact: {required}")
    locked_context_hashes = {
        context_checkpoint_path: frozen_artifacts["context_checkpoint"][
            "sha256"
        ],
        context_manifest_path: frozen_artifacts["context_refit_manifest"][
            "sha256"
        ],
        context_summary_path: frozen_artifacts["context_refit_summary"][
            "sha256"
        ],
    }
    for path, expected_hash in locked_context_hashes.items():
        if sha256_file(path) != expected_hash:
            raise ValueError(
                f"Frozen context artifact changed after cohort lock: {path}"
            )
    checkpoint = torch.load(
        context_checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint_metadata = checkpoint.get("metadata")
    context_contract = dict(config["context_model"])
    if (
        not isinstance(checkpoint_metadata, dict)
        or checkpoint_metadata.get("experiment")
        != context_contract["experiment_name"]
        or int(checkpoint.get("refit_epochs", -1))
        != int(context_contract["refit_epochs"])
        or checkpoint.get("state_dict_scope") != "trainable_parameters"
    ):
        raise ValueError("Context checkpoint does not match frozen contract")

    site_records = [
        SiteRecord(
            header=row["identifier"],
            protein_id=row["canonical_accession"],
            position=int(row["position_one_based"]) - 1,
            sequence=row["window_49"],
            label=int(row["label"]),
            source=cohort_path.name,
        )
        for row in cohort
    ]
    contexts = [row["context_257"] for row in cohort]
    dataset = LongContextSiteDataset(
        site_records,
        contexts,
        original_indices=list(range(len(site_records))),
    )
    loader = make_loader(
        dataset,
        indices=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=0,
    )
    context_config_path = resolve_project_path(context_contract["config"])
    context_config = json.loads(
        context_config_path.read_text(encoding="utf-8")
    )
    model_config = dict(context_config["context_expert"]["model"])
    if checkpoint_metadata.get("model") != model_config:
        raise ValueError("Context architecture configuration changed")
    context_model = build_model(
        aaindex_lookup=load_normalized_aaindex(data_dir / "aaindex31.txt"),
        window_size=int(
            config["cohort_protocol"]["context_window_size"]
        ),
        model_config=model_config,
    )
    load_checkpoint_model_state(context_model, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_model.to(device)
    print(
        f"device={device} external_records={len(site_records)} "
        "context inference started",
        flush=True,
    )
    labels, context_probabilities, indices, diagnostics = predict(
        context_model, loader, device
    )
    order = np.argsort(indices)
    labels = labels[order].astype(np.int64)
    context_probabilities = context_probabilities[order].astype(np.float64)
    diagnostics = diagnostics[order]
    if not np.array_equal(indices[order], np.arange(len(site_records))):
        raise RuntimeError("Context prediction order changed")
    expected_labels = np.asarray(
        [record.label for record in site_records], dtype=np.int64
    )
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError("Context prediction labels changed")
    diagnostic_name = str(
        getattr(
            context_model,
            "diagnostic_name",
            "branch_diagnostics",
        )
    )

    tokens = np.asarray(
        [
            [TOKEN_TO_INDEX[residue] for residue in record.sequence]
            for record in site_records
        ],
        dtype=np.int64,
    )
    aaindex_raw = np.loadtxt(
        data_dir / "aaindex31.txt", dtype=np.float64
    )
    aaindex_min = aaindex_raw.min(axis=1, keepdims=True)
    aaindex_lookup = (
        (aaindex_raw - aaindex_min)
        / (aaindex_raw.max(axis=1, keepdims=True) - aaindex_min)
    ).T
    aaindex_lookup = np.concatenate(
        [aaindex_lookup, np.zeros((1, 31), dtype=np.float64)], axis=0
    )
    aaindex_inputs = aaindex_lookup[tokens]
    one_hot_inputs = np.eye(21, dtype=np.float64)[tokens]
    context_model.to("cpu")
    del context_model, loader, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # The released model is small and CPU inference avoids TensorFlow and
    # PyTorch competing for GPU memory in the same process.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    try:
        import tf_keras
    except ImportError as error:
        raise ImportError(
            "Install tf-keras==2.19.0 for the released H5 model"
        ) from error
    paper_model = tf_keras.models.load_model(model_path, compile=False)
    paper_outputs = paper_model.predict(
        [aaindex_inputs, one_hot_inputs, tokens],
        batch_size=args.batch_size,
        verbose=1,
    )
    if paper_outputs.shape != (len(site_records), 2):
        raise RuntimeError(
            f"Unexpected released-model output shape: {paper_outputs.shape}"
        )
    paper_probabilities = np.asarray(
        paper_outputs[:, 1], dtype=np.float64
    )
    ensemble_probabilities = equal_probability_average(
        [paper_probabilities, context_probabilities]
    )
    threshold = float(evaluation["primary_threshold"])
    paper_metrics = compute_metrics(
        labels, paper_probabilities, threshold
    )
    context_metrics = compute_metrics(
        labels, context_probabilities, threshold
    )
    ensemble_metrics = compute_metrics(
        labels, ensemble_probabilities, threshold
    )

    canonical_accessions = np.asarray(
        [row["canonical_accession"] for row in cohort]
    )
    bootstrap = paired_cluster_bootstrap(
        labels,
        ensemble_probabilities,
        paper_probabilities,
        canonical_accessions,
        threshold=threshold,
        replicates=int(evaluation["bootstrap_replicates"]),
        seed=int(evaluation["bootstrap_seed"]),
        confidence_level=float(evaluation["confidence_level"]),
        progress_every=500,
    )
    results = {
        "status": "frozen external dbPTM/PTMGPT2 evaluation completed",
        "comparison_valid": True,
        "external_validation_claim_allowed": True,
        "historical_test_informed_architecture": True,
        "external_labels_used_for_model_or_threshold_selection": False,
        "cohort": lock["final_cohort"],
        "architecture": {
            "primary_candidate": evaluation["primary_candidate"],
            "fusion": evaluation["fusion"],
            "paper_weight": 0.5,
            "context_weight": 0.5,
            "threshold": threshold,
        },
        "exact_released_mmubipred": paper_metrics,
        "frozen_context_model": context_metrics,
        "equal_probability_ensemble": ensemble_metrics,
        "ensemble_minus_exact_mmubipred": metric_differences(
            ensemble_metrics, paper_metrics
        ),
        "ensemble_minus_context": metric_differences(
            ensemble_metrics, context_metrics
        ),
        "paired_ensemble_vs_exact_mmubipred": {
            "bootstrap": bootstrap,
            "mcnemar_accuracy_descriptive_site_level": {
                **mcnemar_exact(
                    labels,
                    ensemble_probabilities,
                    paper_probabilities,
                    threshold,
                ),
                "inferential_role": (
                    "descriptive only; protein-cluster bootstrap is the "
                    "primary uncertainty analysis"
                ),
            },
        },
        "context_diagnostics": {
            "name": diagnostic_name,
            "mean": diagnostics.mean(axis=0).tolist(),
        },
        "source_artifacts": {
            "cohort_lock_sha256": sha256_file(lock_path),
            "released_model_sha256": sha256_file(model_path),
            "context_checkpoint_sha256": sha256_file(
                context_checkpoint_path
            ),
        },
    }
    np.savez_compressed(
        predictions_path,
        benchmark_indices=np.asarray(
            [int(row["benchmark_index"]) for row in cohort],
            dtype=np.int64,
        ),
        labels=labels,
        canonical_accessions=canonical_accessions,
        paper_probabilities=paper_probabilities,
        context_probabilities=context_probabilities,
        equal_ensemble_probabilities=ensemble_probabilities,
    )
    with predictions_table_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "benchmark_index",
                "identifier",
                "canonical_accession",
                "position_one_based",
                "label",
                "paper_probability",
                "context_probability",
                "ensemble_probability",
                "ensemble_prediction_at_0.5",
            ]
        )
        for row, label, paper, context, ensemble in zip(
            cohort,
            labels,
            paper_probabilities,
            context_probabilities,
            ensemble_probabilities,
        ):
            writer.writerow(
                [
                    row["benchmark_index"],
                    row["identifier"],
                    row["canonical_accession"],
                    row["position_one_based"],
                    int(label),
                    float(paper),
                    float(context),
                    float(ensemble),
                    int(ensemble >= threshold),
                ]
            )
    manifest = {
        "status": "external evaluation completed exactly once",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": git_revision(),
        "config_sha256": sha256_file(config_path),
        "cohort_lock_sha256": sha256_file(lock_path),
        "cohort_sha256": sha256_file(cohort_path),
        "external_labels_used_for_selection": False,
        "predictions_sha256": sha256_file(predictions_path),
        "prediction_table_sha256": sha256_file(predictions_table_path),
    }
    write_json(manifest_path, manifest)
    # Written last: this is the completion guard.
    write_json(results_path, results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
