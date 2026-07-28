#!/usr/bin/env python3
"""Continue the frozen epoch-7 context refit to epoch 15 for exploration."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.context import (  # noqa: E402
    LongContextSiteDataset,
    build_validated_contexts,
    load_sequence_cache,
    sequence_cache_sha256,
)
from ubipred.data import (  # noqa: E402
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def int_array_sha256(values: list[int] | np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype=np.int64).tobytes()
    ).hexdigest()


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the exploratory extension after its latest epoch.",
    )
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen_config = dict(config["final_context_refit"])
    extension_config = dict(config["exploratory_epoch_extension"])
    source_run_dir = resolve_project_path(
        args.source_run_dir or str(frozen_config["output_dir"])
    )
    output_dir = resolve_project_path(
        args.output_dir or str(extension_config["output_dir"])
    )
    if source_run_dir == output_dir:
        raise ValueError(
            "Exploratory output must not overwrite the frozen epoch-7 run"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_best_path = source_run_dir / "best.pt"
    source_resume_path = source_run_dir / "refit_resume.pt"
    source_summary_path = source_run_dir / "refit_summary.json"
    source_manifest_path = source_run_dir / "refit_manifest.json"
    for required_path in (
        source_best_path,
        source_resume_path,
        source_summary_path,
        source_manifest_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Missing frozen epoch-7 artifact: {required_path}"
            )

    source_best = torch.load(
        source_best_path, map_location="cpu", weights_only=False
    )
    source_resume = torch.load(
        source_resume_path, map_location="cpu", weights_only=False
    )
    source_summary = json.loads(
        source_summary_path.read_text(encoding="utf-8")
    )
    source_metadata = source_best.get("metadata")
    if not isinstance(source_metadata, dict):
        raise TypeError("Frozen source checkpoint metadata is missing")
    source_epochs = int(frozen_config["epochs"])
    target_epochs = int(extension_config["target_epochs"])
    if int(extension_config["source_epochs"]) != source_epochs:
        raise ValueError("Exploratory source duration does not match frozen refit")
    if int(source_best.get("refit_epochs", -1)) != source_epochs:
        raise ValueError("Frozen source checkpoint has an unexpected duration")
    if int(source_summary.get("epochs", -1)) != source_epochs:
        raise ValueError("Frozen source summary has an unexpected duration")
    if int(source_resume.get("completed_epoch", -1)) != source_epochs:
        raise ValueError("Frozen source resume state is not at epoch 7")
    if source_resume.get("metadata") != source_metadata:
        raise ValueError("Frozen source checkpoint metadata is inconsistent")
    if source_metadata.get("experiment") != frozen_config["experiment_name"]:
        raise ValueError("Source is not the frozen context-only refit")
    if target_epochs <= source_epochs:
        raise ValueError("Exploratory target must exceed the frozen duration")
    if target_epochs != 15:
        raise ValueError("This recorded exploration is frozen at epoch 15")
    if len(source_resume.get("history", [])) != source_epochs:
        raise ValueError("Frozen source training history is incomplete")
    for name, value in source_best["model_state_dict"].items():
        if not torch.equal(value, source_resume["model_state_dict"][name]):
            raise ValueError(
                f"Frozen checkpoint and resume state differ for {name}"
            )

    source_best_hash = sha256_file(source_best_path)
    source_resume_hash = sha256_file(source_resume_path)
    source_manifest_hash = sha256_file(source_manifest_path)

    data_dir = resolve_project_path(config["data_dir"])
    sequence_cache_path = resolve_project_path(config["sequence_cache"])
    all_records, preprocessing = load_released_split(
        data_dir, split="train", window_size=49
    )
    sequences, cache_metadata = load_sequence_cache(sequence_cache_path)
    original_indices, contexts, context_report = build_validated_contexts(
        all_records,
        sequences,
        context_window_size=int(config["context_window_size"]),
    )
    if context_report["validated_fraction"] < float(
        config["minimum_validated_fraction"]
    ):
        raise RuntimeError(
            "Validated context coverage is below the configured minimum: "
            f"{context_report['validated_fraction']:.4f}"
        )
    records = [all_records[index] for index in original_indices]
    dataset = LongContextSiteDataset(
        records,
        contexts,
        original_indices=original_indices,
    )
    valid_indices_hash = int_array_sha256(original_indices)
    cache_hash = sequence_cache_sha256(sequence_cache_path)
    if (
        source_metadata["validated_training_indices_sha256"]
        != valid_indices_hash
    ):
        raise ValueError("Eligible training records changed since epoch 7")
    if source_metadata["context_sequence_cache_sha256"] != cache_hash:
        raise ValueError("Training sequence cache changed since epoch 7")

    target_metadata = {
        **source_metadata,
        "experiment": str(extension_config["experiment_name"]),
        "refit_epochs": target_epochs,
        "exploratory_extension": {
            "historical_test_informed": True,
            "only_changed_factor": "training duration",
            "source_epochs": source_epochs,
            "target_epochs": target_epochs,
            "source_best_sha256": source_best_hash,
            "source_resume_sha256": source_resume_hash,
            "source_manifest_sha256": source_manifest_hash,
        },
    }

    completed_summary_path = output_dir / "refit_summary.json"
    completed_checkpoint_path = output_dir / "best.pt"
    if (
        args.resume
        and completed_summary_path.exists()
        and completed_checkpoint_path.exists()
    ):
        completed_checkpoint = torch.load(
            completed_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if completed_checkpoint.get("metadata") != target_metadata:
            raise ValueError(
                "Existing exploratory checkpoint has different provenance"
            )
        completed = json.loads(
            completed_summary_path.read_text(encoding="utf-8")
        )
        print(json.dumps(completed, indent=2, sort_keys=True))
        print("Completed epoch-15 exploration already exists; training skipped.")
        return

    manifest = {
        "experiment": str(extension_config["experiment_name"]),
        "status": "post-test exploratory epoch extension",
        "candidate": "context_only",
        "project_git_commit": git_revision(),
        "config_path": str(config_path),
        "source_frozen_run": {
            "path": str(source_run_dir),
            "epochs": source_epochs,
            "best_checkpoint_sha256": source_best_hash,
            "resume_checkpoint_sha256": source_resume_hash,
            "manifest_sha256": source_manifest_hash,
        },
        "extension": {
            "historical_test_informed": True,
            "confirmatory_claim_allowed": False,
            "only_changed_factor": "training duration",
            "source_epochs": source_epochs,
            "target_epochs": target_epochs,
            "epochs_added": target_epochs - source_epochs,
        },
        "released_training_records_before_context_validation": len(all_records),
        "released_training_records_after_context_validation": len(records),
        "context_validation": context_report,
        "validated_training_indices_sha256": valid_indices_hash,
        "sequence_cache": {
            "path": str(sequence_cache_path),
            "sha256": cache_hash,
            "metadata": cache_metadata,
        },
        "preprocessing": {
            name: vars(report) for name, report in preprocessing.items()
        },
        "model": source_metadata["model"],
        "training": source_metadata["training"],
        "seed": int(source_metadata["seed"]),
        "reporting_threshold": 0.5,
        "validation_or_test_used_during_extension_training": False,
        "historical_test_informed_design": True,
        "independent_test_accessed": False,
    }
    write_json(output_dir / "refit_manifest.json", manifest)

    del source_best
    del source_resume
    seed = int(source_metadata["seed"])
    context_config = dict(config["context_expert"])
    seed_everything(seed)
    train_loader = make_loader(
        dataset,
        indices=None,
        batch_size=int(context_config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        seed=seed,
    )
    model = build_model(
        aaindex_lookup=load_normalized_aaindex(data_dir / "aaindex31.txt"),
        window_size=int(config["context_window_size"]),
        model_config=dict(context_config["model"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} records={len(records)} "
        f"continuing_epoch={source_epochs + 1} target_epoch={target_epochs}",
        flush=True,
    )
    clip = context_config.get("gradient_clip_norm", 1.0)
    summary = refit_model(
        model=model,
        train_loader=train_loader,
        device=device,
        output_dir=output_dir,
        epochs=target_epochs,
        learning_rate=float(context_config["learning_rate"]),
        weight_decay=float(context_config["weight_decay"]),
        optimizer_name=str(context_config.get("optimizer", "adamw")),
        optimizer_epsilon=float(
            context_config.get("optimizer_epsilon", 1e-8)
        ),
        gradient_clip_norm=None if clip is None else float(clip),
        gradient_accumulation_steps=int(
            context_config.get("gradient_accumulation_steps", 1)
        ),
        use_amp=bool(context_config.get("use_amp", False)),
        validation_selected_threshold=0.5,
        checkpoint_metadata=target_metadata,
        resume=True,
        compact_checkpoint=True,
        resume_source_path=source_resume_path,
        resume_source_metadata=source_metadata,
    )
    summary.update(
        {
            "status": "post-test exploratory epoch extension",
            "historical_test_informed": True,
            "confirmatory_claim_allowed": False,
            "source_epochs": source_epochs,
            "target_epochs": target_epochs,
            "epochs_added": target_epochs - source_epochs,
            "source_best_sha256": source_best_hash,
            "source_resume_sha256": source_resume_hash,
            "independent_test_accessed_during_training": False,
        }
    )
    write_json(output_dir / "refit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Exploratory epoch-15 extension completed.")
    print("No validation or independent-test examples were loaded during training.")


if __name__ == "__main__":
    main()
