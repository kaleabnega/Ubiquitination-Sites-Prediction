#!/usr/bin/env python3
"""Train UbiFusionNet using only the released MMUbiPred training samples."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "experiment" / "src"
sys.path.insert(0, str(SRC_ROOT))

from ubipred.data import (  # noqa: E402
    SiteDataset,
    load_normalized_aaindex,
    make_loader,
    make_train_validation_indices,
    seed_everything,
)
from ubipred.engine import predict, refit_model, train_model  # noqa: E402
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.metrics import compute_metrics, select_mcc_threshold, write_json  # noqa: E402
from ubipred.model import build_model  # noqa: E402


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "experiment" / "configs" / "ubifusion_v1.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Select and report on development data without a full-data refit.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)

    seed = int(config["seed"])
    window_size = int(config["window_size"])
    data_dir = resolve_project_path(config["data_dir"])
    output_dir = resolve_project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    records, preprocessing_reports = load_released_split(
        data_dir, split="train", window_size=window_size
    )
    train_indices, validation_indices = make_train_validation_indices(
        records,
        validation_fraction=float(config["validation_fraction"]),
        seed=seed,
        strategy=str(config["split_strategy"]),
    )
    dataset = SiteDataset(records)
    train_loader = make_loader(
        dataset,
        indices=train_indices,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        seed=seed,
    )
    validation_loader = make_loader(
        dataset,
        indices=validation_indices,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        seed=seed,
    )

    aaindex_path = data_dir / "aaindex31.txt"
    aaindex_lookup = load_normalized_aaindex(aaindex_path)
    model = build_model(
        aaindex_lookup=aaindex_lookup,
        window_size=window_size,
        model_config=config["model"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_groups = {records[index].protein_id for index in train_indices}
    validation_groups = {records[index].protein_id for index in validation_indices}
    train_labels = np.asarray([records[index].label for index in train_indices])
    validation_labels = np.asarray(
        [records[index].label for index in validation_indices]
    )

    source_files = [
        "Positive_90_percent_training_set_DeepUBI.fasta",
        "Negative_90_percent_training_set_DeepUBI.fasta",
        "aaindex31.txt",
    ]
    manifest = {
        "config": config,
        "config_path": str(args.config.resolve()),
        "project_git_commit": git_revision(PROJECT_ROOT),
        "upstream_git_commit": git_revision(data_dir),
        "device": str(device),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: package_version(name)
            for name in ["numpy", "scikit-learn", "torch"]
        },
        "preprocessing": {
            name: asdict(report) for name, report in preprocessing_reports.items()
        },
        "development_split": {
            "strategy": config["split_strategy"],
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "train_positive": int(train_labels.sum()),
            "train_negative": int((train_labels == 0).sum()),
            "validation_positive": int(validation_labels.sum()),
            "validation_negative": int((validation_labels == 0).sum()),
            "train_proteins": len(train_groups),
            "validation_proteins": len(validation_groups),
            "protein_overlap": len(train_groups & validation_groups),
            "train_indices_sha256": hashlib.sha256(
                np.asarray(train_indices, dtype=np.int64).tobytes()
            ).hexdigest(),
            "validation_indices_sha256": hashlib.sha256(
                np.asarray(validation_indices, dtype=np.int64).tobytes()
            ).hexdigest(),
        },
        "artifacts": {
            filename: {
                "bytes": (data_dir / filename).stat().st_size,
                "sha256": sha256(data_dir / filename),
            }
            for filename in source_files
        },
        "locked_test_accessed": False,
        "development_only": args.development_only,
    }
    write_json(output_dir / "run_manifest.json", manifest)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"device={device}")
    print(f"records={len(records)} train={len(train_indices)} val={len(validation_indices)}")
    print(f"parameters={parameter_count} trainable={trainable_parameter_count}")

    checkpoint_metadata = {
        "experiment_name": config["experiment_name"],
        "window_size": window_size,
        "model_config": config["model"],
        "aaindex_sha256": sha256(aaindex_path),
        "project_git_commit": manifest["project_git_commit"],
        "upstream_git_commit": manifest["upstream_git_commit"],
        "branch_names": list(getattr(model, "branch_names", ())),
        "branch_diagnostic_name": str(
            getattr(model, "diagnostic_name", "branch_diagnostics")
        ),
    }
    gradient_clip_value = config.get("gradient_clip_norm", 1.0)
    gradient_clip_norm = (
        None if gradient_clip_value is None else float(gradient_clip_value)
    )
    development_summary = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        output_dir=output_dir,
        epochs=int(config["epochs"]),
        patience=int(config["early_stopping_patience"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        optimizer_name=str(config.get("optimizer", "adamw")),
        optimizer_epsilon=float(config.get("optimizer_epsilon", 1e-8)),
        gradient_clip_norm=gradient_clip_norm,
        use_amp=bool(config["use_amp"]),
        checkpoint_metadata=checkpoint_metadata,
    )

    labels, probabilities, indices, branch_diagnostics = predict(
        model, validation_loader, device
    )
    selected_threshold, _ = select_mcc_threshold(labels, probabilities)
    diagnostic_name = str(
        getattr(model, "diagnostic_name", "branch_diagnostics")
    )
    diagnostic_key = f"mean_{diagnostic_name}"
    fixed_predictions = probabilities >= 0.5
    validation_results = {
        "fixed_threshold": compute_metrics(labels, probabilities, threshold=0.5),
        "validation_selected_threshold": compute_metrics(
            labels, probabilities, threshold=selected_threshold
        ),
        "branch_names": list(getattr(model, "branch_names", ())),
        "branch_diagnostic_name": diagnostic_name,
        diagnostic_key: branch_diagnostics.mean(axis=0).tolist(),
        "prediction_diagnostics": {
            "probability_mean": float(probabilities.mean()),
            "probability_standard_deviation": float(probabilities.std()),
            "predicted_positive_fraction_at_0_5": float(
                fixed_predictions.mean()
            ),
            "single_class_predictions_at_0_5": bool(
                fixed_predictions.all() or (~fixed_predictions).all()
            ),
        },
        "development_selection": development_summary,
    }
    write_json(output_dir / "validation_metrics.json", validation_results)
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        labels=labels,
        probabilities=probabilities,
        dataset_indices=indices,
        branch_diagnostics=branch_diagnostics,
    )

    if args.development_only:
        training_summary = {
            "development_selection": development_summary,
            "full_training_refit": None,
        }
        write_json(output_dir / "training_summary.json", training_summary)
        manifest["final_refit"] = {
            "performed": False,
            "reason": "development-only benchmark run",
            "test_accessed": False,
        }
        write_json(output_dir / "run_manifest.json", manifest)
        print(json.dumps(validation_results, indent=2, sort_keys=True))
        print("Development-only run completed; full-data refit was skipped.")
        print("Locked independent test set was not accessed.")
        return

    # The development split selects the epoch count and reporting threshold.
    # Reinitialize and refit on every released training sample so the final
    # comparison uses the complete authors' training set.
    model.to("cpu")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    seed_everything(seed)
    full_train_loader = make_loader(
        dataset,
        indices=None,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        seed=seed,
    )
    final_model = build_model(
        aaindex_lookup=aaindex_lookup,
        window_size=window_size,
        model_config=config["model"],
    )
    refit_summary = refit_model(
        model=final_model,
        train_loader=full_train_loader,
        device=device,
        output_dir=output_dir,
        epochs=int(development_summary["best_epoch"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        optimizer_name=str(config.get("optimizer", "adamw")),
        optimizer_epsilon=float(config.get("optimizer_epsilon", 1e-8)),
        gradient_clip_norm=gradient_clip_norm,
        use_amp=bool(config["use_amp"]),
        validation_selected_threshold=selected_threshold,
        checkpoint_metadata=checkpoint_metadata,
    )
    training_summary = {
        "development_selection": development_summary,
        "full_training_refit": refit_summary,
    }
    write_json(output_dir / "training_summary.json", training_summary)
    manifest["final_refit"] = {
        "samples": len(records),
        "positive": int(sum(record.label for record in records)),
        "negative": int(sum(record.label == 0 for record in records)),
        "epochs_selected_on_development_split": int(
            development_summary["best_epoch"]
        ),
        "test_accessed": False,
    }
    write_json(output_dir / "run_manifest.json", manifest)

    print(json.dumps(validation_results, indent=2, sort_keys=True))
    print(json.dumps(training_summary, indent=2, sort_keys=True))
    print("Locked independent test set was not accessed.")


if __name__ == "__main__":
    main()
