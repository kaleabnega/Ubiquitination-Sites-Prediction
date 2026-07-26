#!/usr/bin/env python3
"""Evaluate a selected checkpoint once on the locked MMUbiPred test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "experiment" / "src"
sys.path.insert(0, str(SRC_ROOT))

from ubipred.data import SiteDataset, load_normalized_aaindex, make_loader  # noqa: E402
from ubipred.engine import load_checkpoint_model_state, predict  # noqa: E402
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.metrics import compute_metrics, write_json  # noqa: E402
from ubipred.model import build_model  # noqa: E402


PAPER_BASELINE = {
    "mcc": 0.5458386140298045,
    "accuracy": 0.7725035719955549,
    "sensitivity": 0.7498020585906572,
    "specificity": 0.8067729083665338,
    "confusion_matrix": [[4050, 970], [1896, 5682]],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_locked_outputs_absent(run_dir: Path) -> None:
    locked_outputs = [
        run_dir / "locked_test_metrics.json",
        run_dir / "locked_test_predictions.npz",
        run_dir / "locked_test_predictions.tsv",
    ]
    existing_outputs = [path for path in locked_outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to repeat the locked-test evaluation because output "
            f"already exists: {existing_outputs[0]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "replication" / "MMUbiPred",
    )
    parser.add_argument(
        "--allow-locked-test",
        action="store_true",
        help="Required acknowledgement that this command accesses the final test set.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--require-full-refit",
        action="store_true",
        help="Reject checkpoints not explicitly marked as complete-training refits.",
    )
    args = parser.parse_args()

    if not args.allow_locked_test:
        parser.error(
            "Refusing to access the independent test set without "
            "--allow-locked-test"
        )

    run_dir = args.run_dir.resolve()
    data_dir = args.data_dir.resolve()
    ensure_locked_outputs_absent(run_dir)

    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    if args.require_full_refit and metadata.get("training_scope") != (
        "complete_released_training_set"
    ):
        raise ValueError(
            "Checkpoint is not marked as a complete released-training-set refit"
        )
    manifest_path = run_dir / "run_manifest.json"
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if bool(manifest.get("locked_test_accessed", False)):
            raise RuntimeError(
                "Run manifest already records independent-test access"
            )
    window_size = int(metadata["window_size"])
    aaindex_path = data_dir / "aaindex31.txt"
    actual_aaindex_sha = sha256(aaindex_path)
    if actual_aaindex_sha != metadata["aaindex_sha256"]:
        raise ValueError("AAindex checksum does not match the training artifact")

    records, preprocessing_reports = load_released_split(
        data_dir, split="test", window_size=window_size
    )
    dataset = SiteDataset(records)
    loader = make_loader(
        dataset,
        indices=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=0,
    )

    model = build_model(
        aaindex_lookup=load_normalized_aaindex(aaindex_path),
        window_size=window_size,
        model_config=metadata["model_config"],
    )
    load_checkpoint_model_state(model, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    labels, probabilities, indices, branch_diagnostics = predict(
        model, loader, device
    )
    branch_names = list(getattr(model, "branch_names", ()))
    diagnostic_name = str(
        getattr(model, "diagnostic_name", "branch_diagnostics")
    )
    selected_threshold = float(checkpoint["validation_selected_threshold"])
    fixed_metrics = compute_metrics(labels, probabilities, threshold=0.5)
    selected_metrics = compute_metrics(
        labels, probabilities, threshold=selected_threshold
    )
    comparison = {
        metric: float(fixed_metrics[metric]) - float(PAPER_BASELINE[metric])
        for metric in ["mcc", "accuracy", "sensitivity", "specificity"]
    }
    results = {
        "status": "locked independent test evaluated",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "device": str(device),
        "preprocessing": {
            name: asdict(report) for name, report in preprocessing_reports.items()
        },
        "fixed_threshold": fixed_metrics,
        "validation_selected_threshold": selected_metrics,
        "paper_baseline": PAPER_BASELINE,
        "fixed_threshold_difference_from_paper": comparison,
        "branch_names": branch_names,
        "branch_diagnostic_name": diagnostic_name,
        f"mean_{diagnostic_name}": branch_diagnostics.mean(axis=0).tolist(),
    }
    write_json(run_dir / "locked_test_metrics.json", results)
    np.savez_compressed(
        run_dir / "locked_test_predictions.npz",
        labels=labels,
        probabilities=probabilities,
        dataset_indices=indices,
        branch_diagnostics=branch_diagnostics,
    )

    with (run_dir / "locked_test_predictions.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "protein_id",
                "position",
                "label",
                "probability",
                "prediction_at_0.5",
                *[f"{name}_{diagnostic_name}" for name in branch_names],
            ]
        )
        for label, probability, index, diagnostic_values in zip(
            labels, probabilities, indices, branch_diagnostics
        ):
            record = records[int(index)]
            writer.writerow(
                [
                    record.protein_id,
                    record.position,
                    int(label),
                    float(probability),
                    int(probability >= 0.5),
                    *[float(value) for value in diagnostic_values],
                ]
            )

    if manifest is not None:
        manifest["locked_test_accessed"] = True
        manifest["locked_test_evaluation"] = {
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": results["checkpoint_sha256"],
            "primary_threshold": 0.5,
            "secondary_threshold_selected_on_development": selected_threshold,
            "metrics_file": "locked_test_metrics.json",
            "predictions_file": "locked_test_predictions.npz",
            "predictions_table": "locked_test_predictions.tsv",
        }
        write_json(manifest_path, manifest)

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
