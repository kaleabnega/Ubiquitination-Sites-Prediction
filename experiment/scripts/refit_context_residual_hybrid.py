#!/usr/bin/env python3
"""Exploratory full-data Short-Range refit for the context-residual hybrid."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.context import (  # noqa: E402
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


def derive_local_refit_plan(
    development_run_dir: Path,
    fold_count: int,
) -> dict[str, object]:
    selected_seeds: list[int] = []
    selected_best_epochs: list[int] = []
    for fold in range(fold_count):
        stabilized_dir = (
            development_run_dir
            / "folds"
            / f"fold_{fold}"
            / "local_stabilized"
        )
        selection_path = stabilized_dir / "selection.json"
        if not selection_path.exists():
            raise FileNotFoundError(
                f"Missing stabilized local selection: {selection_path}"
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if int(selection.get("fold", -1)) != fold:
            raise ValueError(f"Stale local selection for fold {fold}")
        if bool(selection.get("outer_fold_accessed_for_selection", True)):
            raise ValueError(
                f"Fold {fold} local initialization used its outer fold"
            )
        seed = int(selection["selected_initialization_seed"])
        candidates = [
            candidate
            for candidate in selection["candidates"]
            if int(candidate["initialization_seed"]) == seed
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Fold {fold} does not identify exactly one selected candidate"
            )
        candidate = candidates[0]
        if bool(candidate.get("inner_collapsed", True)):
            raise ValueError(f"Fold {fold} selected a collapsed Short-Range model")
        best_epoch = int(
            candidate["development_selection"]["best_epoch"]
        )
        selected_seeds.append(seed)
        selected_best_epochs.append(best_epoch)

    seed_counts = Counter(selected_seeds)
    selected_seed = min(
        seed for seed, count in seed_counts.items() if count == max(seed_counts.values())
    )
    median_epoch = statistics.median(selected_best_epochs)
    if int(median_epoch) != median_epoch:
        raise RuntimeError(
            "The median local best epoch is not an integer; freeze an explicit "
            "rounding rule before refitting"
        )
    return {
        "fold_selected_initialization_seeds": selected_seeds,
        "fold_selected_best_epochs": selected_best_epochs,
        "selected_seed": selected_seed,
        "selected_epochs": int(median_epoch),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-post-test-exploration",
        action="store_true",
        help=(
            "Acknowledge that residual fusion is being revisited after the "
            "historical test was inspected."
        ),
    )
    args = parser.parse_args()
    if not args.allow_post_test_exploration:
        parser.error(
            "Refusing the post-test hybrid refit without "
            "--allow-post-test-exploration"
        )

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hybrid_config = dict(config["exploratory_residual_refit"])
    if not bool(hybrid_config.get("historical_test_informed")):
        raise ValueError("Exploratory residual refit must be test-informed")
    development_run_dir = resolve_project_path(
        args.development_run_dir or str(config["output_dir"])
    )
    output_dir = resolve_project_path(
        args.output_dir or str(hybrid_config["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stabilized_summary_path = development_run_dir / "stabilized_summary.json"
    if not stabilized_summary_path.exists():
        raise FileNotFoundError(
            f"Missing corrected five-fold summary: {stabilized_summary_path}"
        )
    stabilized_summary = json.loads(
        stabilized_summary_path.read_text(encoding="utf-8")
    )
    if not bool(stabilized_summary.get("comparison_valid")):
        raise RuntimeError("The corrected development comparison is invalid")
    if stabilized_summary.get("amended_candidate_decision") != "ADVANCE_CONTEXT_ONLY":
        raise RuntimeError(
            "This exploratory follow-up expects the frozen "
            "ADVANCE_CONTEXT_ONLY decision"
        )
    final_stacker = stabilized_summary.get("final_stacker_for_future_refit")
    if not isinstance(final_stacker, dict):
        raise TypeError("The corrected summary has no final residual stacker")
    for key in ("intercept", "local_weight", "context_weight", "l2_strength"):
        if key not in final_stacker:
            raise ValueError(f"Final residual stacker is missing {key}")
    if float(final_stacker["local_weight"]) < 0 or float(
        final_stacker["context_weight"]
    ) < 0:
        raise ValueError("Final residual stacker weights must be nonnegative")

    context_run_dir = resolve_project_path(
        hybrid_config["context_run_dir"]
    )
    context_checkpoint_path = context_run_dir / "best.pt"
    if not context_checkpoint_path.exists():
        raise FileNotFoundError(
            "Frozen Long-Context Expert checkpoint is missing: "
            f"{context_checkpoint_path}"
        )
    context_checkpoint = torch.load(
        context_checkpoint_path, map_location="cpu", weights_only=False
    )
    context_metadata = context_checkpoint.get("metadata")
    if not isinstance(context_metadata, dict):
        raise TypeError("Frozen Long-Context Expert checkpoint metadata is missing")
    final_context_config = dict(config["final_context_refit"])
    if (
        context_metadata.get("experiment")
        != final_context_config["experiment_name"]
        or int(context_checkpoint.get("refit_epochs", -1))
        != int(final_context_config["epochs"])
        or context_metadata.get("model")
        != config["context_expert"]["model"]
    ):
        raise ValueError(
            "Residual hybrid requires the frozen primary seven-epoch "
            "Long-Context checkpoint"
        )
    context_checkpoint_hash = sha256_file(context_checkpoint_path)

    refit_plan = derive_local_refit_plan(
        development_run_dir, int(config["outer_folds"])
    )
    selected_seed = int(refit_plan["selected_seed"])
    selected_epochs = int(refit_plan["selected_epochs"])

    data_dir = resolve_project_path(config["data_dir"])
    sequence_cache_path = resolve_project_path(config["sequence_cache"])
    all_records, preprocessing = load_released_split(
        data_dir, split="train", window_size=49
    )
    sequences, cache_metadata = load_sequence_cache(sequence_cache_path)
    original_indices, _, context_report = build_validated_contexts(
        all_records,
        sequences,
        context_window_size=int(config["context_window_size"]),
    )
    if context_report["validated_fraction"] < float(
        config["minimum_validated_fraction"]
    ):
        raise RuntimeError(
            "Validated context coverage is below the configured minimum"
        )
    records = [all_records[index] for index in original_indices]
    dataset = SiteDataset(records)
    local_config = dict(config["local_expert"])
    model_config = dict(local_config["model"])
    summary_hash = sha256_file(stabilized_summary_path)
    indices_hash = int_array_sha256(original_indices)
    checkpoint_metadata = {
        "experiment": str(hybrid_config["experiment_name"]),
        "candidate": "mmubipred_compatible_local_expert",
        "role": "local_component_of_context_residual_hybrid",
        "seed": selected_seed,
        "refit_epochs": selected_epochs,
        "model": model_config,
        "validated_training_records": len(records),
        "validated_training_indices_sha256": indices_hash,
        "context_sequence_cache_sha256": sequence_cache_sha256(
            sequence_cache_path
        ),
        "stabilized_summary_sha256": summary_hash,
        "final_stacker": final_stacker,
        "context_checkpoint_sha256": context_checkpoint_hash,
        "training": {
            key: local_config.get(key)
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
        "historical_test_informed_exploration": True,
    }
    manifest = {
        "experiment": str(hybrid_config["experiment_name"]),
        "status": "post-test exploratory local full-data refit",
        "candidate": "mmubipred_context_residual_hybrid",
        "confirmatory_claim_allowed": False,
        "project_git_commit": git_revision(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "development_run_dir": str(development_run_dir),
        "stabilized_summary": {
            "path": str(stabilized_summary_path),
            "sha256": summary_hash,
            "amended_candidate_decision": "ADVANCE_CONTEXT_ONLY",
            "residual_mcc_gain_over_context": stabilized_summary[
                "stack_minus_context_fixed_threshold"
            ]["mcc"],
        },
        "local_refit_plan": {
            **refit_plan,
            "seed_selection_rule": hybrid_config[
                "local_seed_selection_rule"
            ],
            "epoch_selection_rule": hybrid_config[
                "local_epoch_selection_rule"
            ],
        },
        "final_stacker": final_stacker,
        "fusion_rule": hybrid_config["fusion_rule"],
        "context_run_dir": str(context_run_dir),
        "context_checkpoint": {
            "path": str(context_checkpoint_path),
            "sha256": context_checkpoint_hash,
            "experiment": context_metadata["experiment"],
            "epochs": context_checkpoint["refit_epochs"],
        },
        "released_training_records_before_context_validation": len(
            all_records
        ),
        "released_training_records_after_context_validation": len(records),
        "context_validation": context_report,
        "validated_training_indices_sha256": indices_hash,
        "sequence_cache": {
            "path": str(sequence_cache_path),
            "sha256": checkpoint_metadata[
                "context_sequence_cache_sha256"
            ],
            "metadata": cache_metadata,
        },
        "preprocessing": {
            name: vars(report) for name, report in preprocessing.items()
        },
        "model": model_config,
        "training": checkpoint_metadata["training"],
        "reporting_threshold": float(
            hybrid_config["reporting_threshold"]
        ),
        "validation_or_test_used_during_refit": False,
        "historical_test_informed_exploration": True,
        "independent_test_accessed": False,
    }
    manifest_path = output_dir / "refit_manifest.json"
    completed_summary_path = output_dir / "refit_summary.json"
    completed_checkpoint_path = output_dir / "best.pt"
    if (
        args.resume
        and completed_summary_path.exists()
        and completed_checkpoint_path.exists()
    ):
        checkpoint = torch.load(
            completed_checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("metadata") != checkpoint_metadata:
            raise ValueError(
                "Completed Short-Range checkpoint does not match this protocol"
            )
        completed = json.loads(
            completed_summary_path.read_text(encoding="utf-8")
        )
        print(json.dumps(completed, indent=2, sort_keys=True))
        print("Completed exploratory Short-Range refit already exists; skipped.")
        print("Released independent test set was not accessed.")
        return

    write_json(manifest_path, manifest)
    seed_everything(selected_seed)
    loader = make_loader(
        dataset,
        indices=None,
        batch_size=int(local_config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        seed=selected_seed,
    )
    model = build_model(
        aaindex_lookup=load_normalized_aaindex(data_dir / "aaindex31.txt"),
        window_size=49,
        model_config=model_config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} records={len(records)} local_epochs={selected_epochs} "
        f"local_seed={selected_seed}",
        flush=True,
    )
    clip = local_config.get("gradient_clip_norm")
    summary = refit_model(
        model=model,
        train_loader=loader,
        device=device,
        output_dir=output_dir,
        epochs=selected_epochs,
        learning_rate=float(local_config["learning_rate"]),
        weight_decay=float(local_config["weight_decay"]),
        optimizer_name=str(local_config.get("optimizer", "adam")),
        optimizer_epsilon=float(
            local_config.get("optimizer_epsilon", 1e-7)
        ),
        gradient_clip_norm=None if clip is None else float(clip),
        gradient_accumulation_steps=int(
            local_config.get("gradient_accumulation_steps", 1)
        ),
        use_amp=bool(local_config.get("use_amp", False)),
        validation_selected_threshold=0.5,
        checkpoint_metadata=checkpoint_metadata,
        resume=args.resume,
        compact_checkpoint=True,
    )
    summary.update(
        {
            "candidate": "mmubipred_context_residual_hybrid",
            "component_refitted": "mmubipred_compatible_local_expert",
            "local_refit_plan": refit_plan,
            "final_stacker": final_stacker,
            "context_model_retrained": False,
            "historical_test_informed_exploration": True,
            "confirmatory_claim_allowed": False,
            "independent_test_accessed": False,
        }
    )
    write_json(completed_summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Short-Range Expert refit completed; Long-Context Expert was not retrained.")
    print("Released independent test set was not accessed.")


if __name__ == "__main__":
    main()
