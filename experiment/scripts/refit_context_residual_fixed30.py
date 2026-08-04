#!/usr/bin/env python3
"""Full-data fixed-30-epoch refit for the exploratory residual hybrid."""

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
    SiteDataset,
    load_normalized_aaindex,
    make_loader,
    seed_everything,
)
from ubipred.engine import refit_model  # noqa: E402
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.fixed30 import (  # noqa: E402
    FIXED_EPOCHS,
    PROTOCOL_VERSION,
    fixed30_training_arguments,
    validate_fixed30_protocol,
)
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


def indices_sha256(indices: list[int] | np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes()
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


def refit_expert(
    *,
    expert_name: str,
    expert_config: dict[str, object],
    dataset: torch.utils.data.Dataset,
    aaindex_lookup: np.ndarray,
    window_size: int,
    output_dir: Path,
    device: torch.device,
    num_workers: int,
    config_hash: str,
    oof_summary_hash: str,
    context_cache_hash: str,
    validated_indices_hash: str,
    resume: bool,
) -> dict[str, object]:
    expert_seed = int(expert_config["seed"])
    model_config = dict(expert_config["model"])
    checkpoint_metadata = {
        "experiment": "mmubipred_context_residual_fixed30_exploratory_v1",
        "protocol_version": PROTOCOL_VERSION,
        "component": expert_name,
        "analysis_role": "post-test exploratory fixed-duration rebuild",
        "confirmatory_claim_allowed": False,
        "seed": expert_seed,
        "fixed_epochs": FIXED_EPOCHS,
        "early_stopping": False,
        "validation_or_test_used_during_refit": False,
        "window_size": window_size,
        "model": model_config,
        "validated_training_records": len(dataset),
        "validated_training_indices_sha256": validated_indices_hash,
        "context_sequence_cache_sha256": (
            context_cache_hash if expert_name == "context" else None
        ),
        "oof_summary_sha256": oof_summary_hash,
        "config_sha256": config_hash,
        "training": {
            key: expert_config.get(key)
            for key in (
                "batch_size",
                "learning_rate",
                "weight_decay",
                "optimizer",
                "optimizer_epsilon",
                "gradient_clip_norm",
                "gradient_accumulation_steps",
                "use_amp",
            )
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_summary_path = output_dir / "refit_summary.json"
    completed_checkpoint_path = output_dir / "best.pt"
    if resume and completed_summary_path.exists() and completed_checkpoint_path.exists():
        checkpoint = torch.load(
            completed_checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("metadata") != checkpoint_metadata:
            raise ValueError(f"Completed {expert_name} checkpoint metadata mismatch")
        completed = json.loads(completed_summary_path.read_text(encoding="utf-8"))
        print(f"full_refit expert={expert_name} already complete; skipped", flush=True)
        return completed

    seed_everything(expert_seed)
    loader = make_loader(
        dataset,
        indices=None,
        batch_size=int(expert_config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        seed=expert_seed,
    )
    model = build_model(
        aaindex_lookup=aaindex_lookup,
        window_size=window_size,
        model_config=model_config,
    )
    print(
        f"full_refit expert={expert_name} device={device} records={len(dataset)} "
        f"fixed_epochs={FIXED_EPOCHS} early_stopping=False",
        flush=True,
    )
    summary = refit_model(
        model=model,
        train_loader=loader,
        device=device,
        output_dir=output_dir,
        epochs=FIXED_EPOCHS,
        validation_selected_threshold=0.5,
        checkpoint_metadata=checkpoint_metadata,
        resume=resume,
        compact_checkpoint=True,
        **fixed30_training_arguments(expert_config),
    )
    summary.update(
        {
            "component": expert_name,
            "fixed_epochs": FIXED_EPOCHS,
            "early_stopping": False,
            "validation_or_test_used_during_refit": False,
            "independent_or_external_test_accessed": False,
        }
    )
    write_json(completed_summary_path, summary)
    model.to("cpu")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-post-test-exploration", action="store_true")
    args = parser.parse_args()
    if not args.allow_post_test_exploration:
        parser.error(
            "Acknowledge this post-test fixed-duration rebuild with "
            "--allow-post-test-exploration"
        )

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_fixed30_protocol(config)
    development_run_dir = resolve_project_path(
        args.development_run_dir or str(config["output_dir"])
    )
    output_dir = resolve_project_path(
        args.output_dir or str(development_run_dir / "full_refit")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_summary_path = development_run_dir / "summary.json"
    if not oof_summary_path.exists():
        raise FileNotFoundError(
            f"Complete all ten expert-fold runs first: {oof_summary_path}"
        )
    oof_summary = json.loads(oof_summary_path.read_text(encoding="utf-8"))
    if not bool(oof_summary.get("comparison_valid")):
        raise RuntimeError("The fixed30 OOF comparison contains a collapsed expert")
    if int(oof_summary.get("fixed_epochs_per_expert_per_fold", -1)) != FIXED_EPOCHS:
        raise RuntimeError("The OOF summary is not the fixed30 protocol")
    if bool(oof_summary.get("early_stopping", True)):
        raise RuntimeError("The OOF summary unexpectedly used early stopping")
    final_stacker = dict(oof_summary["final_stacker_for_future_refit"])

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
    local_dataset = SiteDataset(records)
    context_dataset = LongContextSiteDataset(records, contexts)
    cache_hash = sequence_cache_sha256(sequence_cache_path)
    valid_indices_hash = indices_sha256(original_indices)
    config_hash = sha256_file(config_path)
    oof_summary_hash = sha256_file(oof_summary_path)
    manifest = {
        "experiment": str(config["experiment_name"]),
        "protocol_version": PROTOCOL_VERSION,
        "analysis_role": str(config["analysis_role"]),
        "project_git_commit": git_revision(),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "development_run_dir": str(development_run_dir),
        "oof_summary": {
            "path": str(oof_summary_path),
            "sha256": oof_summary_hash,
        },
        "fixed_epochs_per_component": FIXED_EPOCHS,
        "early_stopping": False,
        "training_records": len(records),
        "validated_training_indices_sha256": valid_indices_hash,
        "context_validation": context_report,
        "sequence_cache": {
            "path": str(sequence_cache_path),
            "sha256": cache_hash,
            "metadata": cache_metadata,
        },
        "preprocessing": {
            name: vars(report) for name, report in preprocessing.items()
        },
        "fusion": {
            "rule": "nonnegative residual logit stacker fitted to fixed30 OOF predictions",
            "parameters": final_stacker,
            "reporting_threshold": float(config["reporting_threshold"]),
        },
        "historical_test_informed_exploration": True,
        "confirmatory_claim_allowed": False,
        "validation_or_test_used_during_refit": False,
        "independent_or_external_test_accessed": False,
    }
    manifest_path = output_dir / "full_refit_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "config_sha256",
            "validated_training_indices_sha256",
            "fixed_epochs_per_component",
        ):
            if existing.get(key) != manifest.get(key):
                raise ValueError(f"Existing full-refit manifest changed: {key}")
        if existing.get("oof_summary", {}).get("sha256") != oof_summary_hash:
            raise ValueError("Existing full-refit OOF summary hash changed")
        existing["last_resume_git_commit"] = git_revision()
        write_json(manifest_path, existing)
    else:
        write_json(manifest_path, manifest)

    aaindex_lookup = load_normalized_aaindex(data_dir / "aaindex31.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    component_summaries: dict[str, object] = {}
    for expert_name, dataset, window_size in (
        ("local", local_dataset, 49),
        ("context", context_dataset, int(config["context_window_size"])),
    ):
        component_summaries[expert_name] = refit_expert(
            expert_name=expert_name,
            expert_config=dict(config[f"{expert_name}_expert"]),
            dataset=dataset,
            aaindex_lookup=aaindex_lookup,
            window_size=window_size,
            output_dir=output_dir / expert_name,
            device=device,
            num_workers=int(config["num_workers"]),
            config_hash=config_hash,
            oof_summary_hash=oof_summary_hash,
            context_cache_hash=cache_hash,
            validated_indices_hash=valid_indices_hash,
            resume=args.resume,
        )

    final_summary = {
        "experiment": str(config["experiment_name"]),
        "status": "fixed30 residual-hybrid full-data refit completed",
        "analysis_role": str(config["analysis_role"]),
        "fixed_epochs_per_component": FIXED_EPOCHS,
        "early_stopping": False,
        "training_records": len(records),
        "components": {
            name: {
                "checkpoint": str(output_dir / name / "best.pt"),
                "checkpoint_sha256": sha256_file(output_dir / name / "best.pt"),
                "summary": summary,
            }
            for name, summary in component_summaries.items()
        },
        "residual_stacker": final_stacker,
        "reporting_threshold": float(config["reporting_threshold"]),
        "historical_test_informed_exploration": True,
        "confirmatory_claim_allowed": False,
        "independent_or_external_test_accessed": False,
    }
    write_json(output_dir / "hybrid_refit_summary.json", final_summary)
    print(json.dumps(final_summary, indent=2, sort_keys=True))
    print("Released independent and external test sets were not accessed.")


if __name__ == "__main__":
    main()
