#!/usr/bin/env python3
"""Refit a frozen development-selected candidate on all released training data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "experiment" / "src"
sys.path.insert(0, str(SRC_ROOT))

from ubipred.data import (  # noqa: E402
    SiteDataset,
    load_normalized_aaindex,
    make_loader,
    seed_everything,
)
from ubipred.engine import refit_model  # noqa: E402
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.metrics import write_json  # noqa: E402
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


def _require_close(actual: object, expected: object, name: str) -> None:
    if not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"Frozen selection mismatch for {name}: "
            f"expected {expected}, received {actual}"
        )


def validate_frozen_selection(
    config: dict[str, object],
    selection_manifest: dict[str, object],
    selection_metrics: dict[str, object],
) -> None:
    """Reject any development artifact that differs from the frozen decision."""

    contract = config["selection_contract"]
    if not isinstance(contract, dict):
        raise TypeError("selection_contract must be an object")
    development_config = selection_manifest["config"]
    if not isinstance(development_config, dict):
        raise TypeError("Development run manifest config must be an object")

    for key in ["experiment_name", "seed"]:
        if development_config[key] != contract[key]:
            raise ValueError(
                f"Frozen selection mismatch for {key}: "
                f"expected {contract[key]}, received {development_config[key]}"
            )
    if development_config["model"] != config["model"]:
        raise ValueError("Frozen development and final-refit model configs differ")
    if not bool(selection_manifest.get("development_only", False)):
        raise ValueError("Selection artifact is not a development-only run")
    if bool(selection_manifest.get("locked_test_accessed", False)):
        raise ValueError("Selection artifact records independent-test access")

    development_split = selection_manifest["development_split"]
    if not isinstance(development_split, dict):
        raise TypeError("development_split must be an object")
    if (
        development_split["validation_indices_sha256"]
        != contract["validation_indices_sha256"]
    ):
        raise ValueError("Frozen development validation hash does not match")

    selection = selection_metrics["development_selection"]
    if not isinstance(selection, dict):
        raise TypeError("development_selection must be an object")
    for key in ["epochs_completed", "best_epoch"]:
        if int(selection[key]) != int(contract[key]):
            raise ValueError(
                f"Frozen selection mismatch for {key}: "
                f"expected {contract[key]}, received {selection[key]}"
            )
    _require_close(
        selection["validation_mcc_at_0_5"],
        contract["validation_mcc_at_0_5"],
        "validation_mcc_at_0_5",
    )
    _require_close(
        selection["validation_selected_threshold"],
        contract["validation_selected_threshold"],
        "validation_selected_threshold",
    )
    diagnostics = selection_metrics["prediction_diagnostics"]
    if not isinstance(diagnostics, dict):
        raise TypeError("prediction_diagnostics must be an object")
    if (
        bool(diagnostics["single_class_predictions_at_0_5"])
        != bool(contract["single_class_predictions_at_0_5"])
    ):
        raise ValueError("Frozen prediction-collapse diagnostic differs")

    if int(config["refit_epochs"]) != int(contract["best_epoch"]):
        raise ValueError("Final refit epochs must equal the frozen best epoch")
    _require_close(
        config["validation_selected_threshold"],
        contract["validation_selected_threshold"],
        "final reporting threshold",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the complete-training refit after its latest finished epoch.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)

    selection_dir = args.selection_run_dir.resolve()
    selection_manifest_path = selection_dir / "run_manifest.json"
    selection_metrics_path = selection_dir / "validation_metrics.json"
    if not selection_manifest_path.exists() or not selection_metrics_path.exists():
        raise FileNotFoundError(
            "The complete development run_manifest.json and "
            "validation_metrics.json are required"
        )
    selection_manifest = json.loads(
        selection_manifest_path.read_text(encoding="utf-8")
    )
    selection_metrics = json.loads(
        selection_metrics_path.read_text(encoding="utf-8")
    )
    validate_frozen_selection(config, selection_manifest, selection_metrics)

    output_dir = resolve_project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if (output_dir / "best.pt").exists() and (
        output_dir / "refit_summary.json"
    ).exists():
        if manifest_path.exists():
            existing_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            print(
                "Completed full-data refit already exists; training was skipped."
            )
            print(
                "Locked independent test accessed:",
                bool(existing_manifest.get("locked_test_accessed", False)),
            )
            return
        raise FileNotFoundError("Completed refit lacks run_manifest.json")
    if not args.resume and (output_dir / "refit_resume.pt").exists():
        raise RuntimeError(
            "An incomplete refit exists. Rerun with --resume or choose a new "
            "output directory."
        )

    seed = int(config["seed"])
    window_size = int(config["window_size"])
    data_dir = resolve_project_path(config["data_dir"])
    seed_everything(seed)

    records, preprocessing_reports = load_released_split(
        data_dir, split="train", window_size=window_size
    )
    expected_samples = int(config["expected_full_training_samples"])
    if len(records) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} complete-training records, "
            f"received {len(records)}"
        )
    dataset = SiteDataset(records)
    train_loader = make_loader(
        dataset,
        indices=None,
        batch_size=int(config["batch_size"]),
        shuffle=True,
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
    project_commit = git_revision(PROJECT_ROOT)
    upstream_commit = git_revision(data_dir)

    source_files = [
        "Positive_90_percent_training_set_DeepUBI.fasta",
        "Negative_90_percent_training_set_DeepUBI.fasta",
        "aaindex31.txt",
    ]
    selection_provenance = {
        "run_dir": str(selection_dir),
        "manifest_sha256": sha256(selection_manifest_path),
        "metrics_sha256": sha256(selection_metrics_path),
        "validation_indices_sha256": config["selection_contract"][
            "validation_indices_sha256"
        ],
        "best_epoch": int(config["refit_epochs"]),
        "validation_selected_threshold": float(
            config["validation_selected_threshold"]
        ),
    }
    manifest = {
        "config": config,
        "config_path": str(args.config.resolve()),
        "project_git_commit": project_commit,
        "upstream_git_commit": upstream_commit,
        "device": str(device),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: package_version(name)
            for name in ["numpy", "scikit-learn", "torch", "transformers", "peft"]
        },
        "preprocessing": {
            name: asdict(report) for name, report in preprocessing_reports.items()
        },
        "selection_provenance": selection_provenance,
        "complete_training_set": {
            "samples": len(records),
            "positive": int(sum(record.label for record in records)),
            "negative": int(sum(record.label == 0 for record in records)),
        },
        "artifacts": {
            filename: {
                "bytes": (data_dir / filename).stat().st_size,
                "sha256": sha256(data_dir / filename),
            }
            for filename in source_files
        },
        "final_refit": {
            "performed": False,
            "status": "in_progress",
            "epochs": int(config["refit_epochs"]),
        },
        "locked_test_accessed": False,
    }
    write_json(manifest_path, manifest)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"device={device}")
    print(f"complete_training_records={len(records)}")
    print(f"parameters={parameter_count} trainable={trainable_parameter_count}")
    print(f"frozen_refit_epochs={int(config['refit_epochs'])}")
    print("independent_test_accessed=False")

    checkpoint_metadata = {
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "window_size": window_size,
        "model_config": config["model"],
        "aaindex_sha256": sha256(aaindex_path),
        "project_git_commit": project_commit,
        "upstream_git_commit": upstream_commit,
        "branch_names": list(getattr(model, "branch_names", ())),
        "branch_diagnostic_name": str(
            getattr(model, "diagnostic_name", "branch_diagnostics")
        ),
        "training_scope": "complete_released_training_set",
        "selection_provenance": selection_provenance,
    }
    gradient_clip_value = config.get("gradient_clip_norm", 1.0)
    gradient_clip_norm = (
        None if gradient_clip_value is None else float(gradient_clip_value)
    )
    summary = refit_model(
        model=model,
        train_loader=train_loader,
        device=device,
        output_dir=output_dir,
        epochs=int(config["refit_epochs"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        optimizer_name=str(config.get("optimizer", "adamw")),
        optimizer_epsilon=float(config.get("optimizer_epsilon", 1e-8)),
        gradient_clip_norm=gradient_clip_norm,
        gradient_accumulation_steps=int(
            config.get("gradient_accumulation_steps", 1)
        ),
        use_amp=bool(config["use_amp"]),
        validation_selected_threshold=float(
            config["validation_selected_threshold"]
        ),
        checkpoint_metadata=checkpoint_metadata,
        resume=args.resume,
    )
    manifest["final_refit"] = {
        "performed": True,
        "status": "completed",
        **summary,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Full-data refit completed.")
    print("Locked independent test accessed: False")


if __name__ == "__main__":
    main()
