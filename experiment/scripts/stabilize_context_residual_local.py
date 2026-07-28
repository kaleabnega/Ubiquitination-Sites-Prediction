#!/usr/bin/env python3
"""Stabilize only the cheap local OOF expert without retraining ESM-2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "experiment" / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "experiment" / "scripts"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from run_context_residual_cv import (  # noqa: E402
    finish_oof_summary,
    indices_sha256,
    resolve_project_path,
    training_arguments,
)
from ubipred.context import (  # noqa: E402
    build_validated_contexts,
    load_sequence_cache,
    sequence_cache_sha256,
)
from ubipred.data import (  # noqa: E402
    SiteDataset,
    load_normalized_aaindex,
    make_loader,
    make_train_validation_indices,
    seed_everything,
)
from ubipred.engine import (  # noqa: E402
    load_checkpoint_model_state,
    predict,
    train_model,
)
from ubipred.fasta import load_released_split  # noqa: E402
from ubipred.metrics import compute_metrics, write_json  # noqa: E402
from ubipred.model import build_model  # noqa: E402


def is_collapsed(metrics: dict[str, object]) -> bool:
    sensitivity = float(metrics["sensitivity"])
    specificity = float(metrics["specificity"])
    return (sensitivity == 1.0 and specificity == 0.0) or (
        sensitivity == 0.0 and specificity == 1.0
    )


def load_candidate_result(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if "inner_fixed_threshold" not in result:
        raise ValueError(f"Incomplete candidate result: {path}")
    return result


def candidate_sort_key(result: dict[str, object]) -> tuple[float, float, int]:
    metrics = result["inner_fixed_threshold"]
    if not isinstance(metrics, dict):
        raise TypeError("inner_fixed_threshold must be a dictionary")
    return (
        -float(metrics["mcc"]),
        -float(metrics["auprc"]),
        int(result["initialization_seed"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--fold",
        type=int,
        action="append",
        help="Stabilize only this fold; repeat for multiple folds.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = resolve_project_path(args.output_dir or config["output_dir"])
    data_dir = resolve_project_path(config["data_dir"])
    sequence_cache_path = resolve_project_path(config["sequence_cache"])
    seed = int(config["seed"])
    fold_count = int(config["outer_folds"])
    requested_folds = (
        sorted(set(args.fold)) if args.fold is not None else list(range(fold_count))
    )
    if any(fold < 0 or fold >= fold_count for fold in requested_folds):
        raise ValueError(f"fold must be between 0 and {fold_count - 1}")
    stabilization = dict(config["local_stabilization"])
    initialization_seeds = [
        int(value) for value in stabilization["initialization_seeds"]
    ]
    if len(initialization_seeds) < 2 or len(set(initialization_seeds)) != len(
        initialization_seeds
    ):
        raise ValueError("local_stabilization requires unique multiple seeds")

    all_records, _ = load_released_split(data_dir, split="train", window_size=49)
    sequences, _ = load_sequence_cache(sequence_cache_path)
    original_indices, _, context_report = build_validated_contexts(
        all_records,
        sequences,
        context_window_size=int(config["context_window_size"]),
    )
    if context_report["validated_fraction"] < float(
        config["minimum_validated_fraction"]
    ):
        raise RuntimeError("Validated context coverage fell below the frozen minimum")
    records = [all_records[index] for index in original_indices]
    dataset = SiteDataset(records)
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    groups = np.asarray([record.protein_id for record in records])
    splitter = StratifiedGroupKFold(
        n_splits=fold_count, shuffle=True, random_state=seed
    )
    folds = list(splitter.split(np.arange(len(records)), labels, groups))
    fold_assignments = np.full(len(records), -1, dtype=np.int64)
    for fold, (_, validation_indices) in enumerate(folds):
        fold_assignments[validation_indices] = fold

    aaindex_lookup = load_normalized_aaindex(data_dir / "aaindex31.txt")
    local_config = dict(config["local_expert"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_hash = sequence_cache_sha256(sequence_cache_path)
    print(
        f"device={device} protocol=local_multi_initialization "
        f"seeds={initialization_seeds} folds={requested_folds}",
        flush=True,
    )

    for fold in requested_folds:
        outer_train_indices, outer_validation_indices = folds[fold]
        outer_training_records = [
            records[index] for index in outer_train_indices
        ]
        inner_train_relative, inner_validation_relative = (
            make_train_validation_indices(
                outer_training_records,
                validation_fraction=float(
                    local_config["inner_validation_fraction"]
                ),
                seed=seed + fold,
                strategy="protein_grouped",
            )
        )
        inner_train_indices = outer_train_indices[inner_train_relative]
        inner_validation_indices = outer_train_indices[
            inner_validation_relative
        ]
        inner_train_hash = indices_sha256(inner_train_indices)
        inner_validation_hash = indices_sha256(inner_validation_indices)
        stabilized_dir = (
            output_dir / "folds" / f"fold_{fold}" / "local_stabilized"
        )
        candidate_results: list[dict[str, object]] = []

        for initialization_seed in initialization_seeds:
            candidate_dir = (
                stabilized_dir
                / "candidates"
                / f"seed_{initialization_seed}"
            )
            result_path = candidate_dir / "candidate_selection.json"
            if result_path.exists() and args.resume:
                result = load_candidate_result(result_path)
                if (
                    int(result["fold"]) != fold
                    or int(result["initialization_seed"]) != initialization_seed
                    or str(result["inner_train_indices_sha256"])
                    != inner_train_hash
                    or str(result["inner_validation_indices_sha256"])
                    != inner_validation_hash
                ):
                    raise ValueError(
                        f"Stale stabilized candidate result: {result_path}"
                    )
                print(
                    f"fold={fold} seed={initialization_seed} reused "
                    f"inner_mcc={float(result['inner_fixed_threshold']['mcc']):.5f}",
                    flush=True,
                )
                candidate_results.append(result)
                continue

            seed_everything(initialization_seed)
            train_loader = make_loader(
                dataset,
                indices=inner_train_indices,
                batch_size=int(local_config["batch_size"]),
                shuffle=True,
                num_workers=int(config["num_workers"]),
                seed=initialization_seed,
            )
            inner_loader = make_loader(
                dataset,
                indices=inner_validation_indices,
                batch_size=int(local_config["batch_size"]),
                shuffle=False,
                num_workers=int(config["num_workers"]),
                seed=initialization_seed,
            )
            model = build_model(
                aaindex_lookup=aaindex_lookup,
                window_size=49,
                model_config=dict(local_config["model"]),
            )
            metadata = {
                "experiment": "mmubipred_context_residual_v1",
                "protocol_amendment": "local_multi_initialization_stabilization",
                "fold": fold,
                "base_split_seed": seed,
                "initialization_seed": initialization_seed,
                "model": local_config["model"],
                "training": training_arguments(local_config),
                "outer_train_indices_sha256": indices_sha256(
                    outer_train_indices
                ),
                "outer_validation_indices_sha256": indices_sha256(
                    outer_validation_indices
                ),
                "inner_train_indices_sha256": indices_sha256(
                    inner_train_indices
                ),
                "inner_validation_indices_sha256": indices_sha256(
                    inner_validation_indices
                ),
            }
            completed_summary_path = candidate_dir / "development_summary.json"
            completed_checkpoint_path = candidate_dir / "development_best.pt"
            if (
                args.resume
                and completed_summary_path.exists()
                and completed_checkpoint_path.exists()
            ):
                development = json.loads(
                    completed_summary_path.read_text(encoding="utf-8")
                )
                completed_checkpoint = torch.load(
                    completed_checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if completed_checkpoint["metadata"] != metadata:
                    raise ValueError(
                        f"Completed candidate metadata changed: {candidate_dir}"
                    )
                load_checkpoint_model_state(model, completed_checkpoint)
                model.to(device)
            else:
                development = train_model(
                    model=model,
                    train_loader=train_loader,
                    validation_loader=inner_loader,
                    device=device,
                    output_dir=candidate_dir,
                    checkpoint_metadata=metadata,
                    resume=args.resume,
                    **training_arguments(local_config),
                )
            inner_labels, inner_probabilities, inner_indices, _ = predict(
                model, inner_loader, device
            )
            inner_metrics = compute_metrics(
                inner_labels, inner_probabilities, threshold=0.5
            )
            result = {
                "fold": fold,
                "initialization_seed": initialization_seed,
                "development_selection": development,
                "inner_fixed_threshold": inner_metrics,
                "inner_collapsed": is_collapsed(inner_metrics),
                "inner_train_indices_sha256": inner_train_hash,
                "inner_validation_indices_sha256": inner_validation_hash,
                "inner_indices_sha256": indices_sha256(
                    np.sort(inner_indices)
                ),
                "outer_fold_accessed_for_selection": False,
            }
            write_json(result_path, result)
            candidate_results.append(result)
            print(
                f"fold={fold} seed={initialization_seed} "
                f"inner_mcc={float(inner_metrics['mcc']):.5f} "
                f"collapsed={result['inner_collapsed']}",
                flush=True,
            )
            model.to("cpu")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        eligible = [
            result
            for result in candidate_results
            if not bool(result["inner_collapsed"])
        ]
        if not eligible:
            raise RuntimeError(
                f"Every local initialization collapsed on inner fold {fold}"
            )
        selected = sorted(eligible, key=candidate_sort_key)[0]
        selected_seed = int(selected["initialization_seed"])
        selection = {
            "fold": fold,
            "protocol": (
                "select non-collapsed initialization by maximum inner fixed "
                "MCC; break ties by AUPRC then ascending seed"
            ),
            "initialization_seeds": initialization_seeds,
            "candidates": candidate_results,
            "selected_initialization_seed": selected_seed,
            "outer_fold_accessed_for_selection": False,
        }
        write_json(stabilized_dir / "selection.json", selection)

        seed_everything(selected_seed)
        selected_model = build_model(
            aaindex_lookup=aaindex_lookup,
            window_size=49,
            model_config=dict(local_config["model"]),
        )
        checkpoint = torch.load(
            stabilized_dir
            / "candidates"
            / f"seed_{selected_seed}"
            / "development_best.pt",
            map_location="cpu",
            weights_only=False,
        )
        load_checkpoint_model_state(selected_model, checkpoint)
        selected_model.to(device)
        outer_loader = make_loader(
            dataset,
            indices=outer_validation_indices,
            batch_size=int(local_config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            seed=selected_seed,
        )
        outer_labels, outer_probabilities, outer_indices, diagnostics = predict(
            selected_model, outer_loader, device
        )
        order = np.argsort(outer_indices)
        outer_indices = outer_indices[order]
        outer_labels = outer_labels[order]
        outer_probabilities = outer_probabilities[order]
        diagnostics = diagnostics[order]
        outer_metrics = compute_metrics(
            outer_labels, outer_probabilities, threshold=0.5
        )
        np.savez_compressed(
            stabilized_dir / "outer_predictions.npz",
            dataset_indices=outer_indices,
            labels=outer_labels,
            probabilities=outer_probabilities,
            diagnostics=diagnostics,
            context_cache_sha256=np.asarray(""),
            selected_initialization_seed=np.asarray(selected_seed),
        )
        write_json(
            stabilized_dir / "outer_metrics.json",
            {
                "selection": selection,
                "outer_fixed_threshold": outer_metrics,
                "outer_collapsed": is_collapsed(outer_metrics),
                "outer_threshold_was_not_selected": True,
                "outer_samples": len(outer_labels),
                "outer_indices_sha256": indices_sha256(outer_indices),
            },
        )
        print(
            f"fold={fold} selected_seed={selected_seed} "
            f"outer_mcc@0.5={float(outer_metrics['mcc']):.5f} "
            f"outer_collapsed={is_collapsed(outer_metrics)}",
            flush=True,
        )
        selected_model.to("cpu")
        del selected_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    completed = all(
        (
            output_dir
            / "folds"
            / f"fold_{fold}"
            / "local_stabilized"
            / "outer_predictions.npz"
        ).exists()
        and (
            output_dir
            / "folds"
            / f"fold_{fold}"
            / "context"
            / "outer_predictions.npz"
        ).exists()
        for fold in range(fold_count)
    )
    if completed:
        summary = finish_oof_summary(
            output_dir=output_dir,
            records=records,
            fold_assignments=fold_assignments,
            l2_strength=float(config["stacker_l2_strength"]),
            local_directory_name="local_stabilized",
            summary_filename="stabilized_summary.json",
            predictions_filename="stabilized_oof_predictions.npz",
        )
        summary["protocol_amendment"] = {
            "reason": "original local fold 4 predicted one class",
            "context_models_retrained": False,
            "local_initialization_seeds": initialization_seeds,
            "selection_uses_outer_fold": False,
            "sequence_cache_sha256": cache_hash,
        }
        write_json(output_dir / "stabilized_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "Requested stabilized local folds completed; summary awaits all "
            "five stabilized local predictions.",
            flush=True,
        )
    print("Existing context predictions were reused; independent test not accessed.")


if __name__ == "__main__":
    main()
