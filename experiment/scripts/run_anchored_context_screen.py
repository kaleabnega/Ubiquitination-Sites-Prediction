#!/usr/bin/env python3
"""Development-only exact-architecture anchor plus bounded context correction.

The released checkpoint supplies architecture, not in-sample training scores.
Fresh models trained on protein-grouped folds supply all anchor OOF scores.
Only prior train-cohort and context OOF artifacts are read; no test artifact is
opened by any stage of this script.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment/src"))
from ubipred.anchored_correction import fit_anchored_correction  # noqa: E402
from ubipred.benchmark_sources import atomic_json, digest, object_digest  # noqa: E402
from ubipred.fasta import ALPHABET  # noqa: E402
from ubipred.metrics import compute_metrics  # noqa: E402


def read(path: Path):
    return json.loads(path.read_text())


def checked_file(path: Path, expected: str) -> None:
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"Missing or changed artifact: {path}")


def load_contract(args):
    config = read(args.config)
    if (config["schema_version"] != 1 or config["outer_folds"] != 5
            or config["epochs_per_seed"] != 15
            or config["initialization_seeds"] != [42, 123, 2026]
            or config["released_test_access"] != "forbidden"):
        raise ValueError("The registered development protocol has changed")
    prior = args.prior_run_dir
    prior_protocol = read(prior / "protocol.json")
    prior_summary = read(prior / "development_summary.json")
    benchmark = prior_protocol["config"]["benchmark"]
    if benchmark not in config["supported_benchmarks"] or benchmark != prior_summary["benchmark"]:
        raise ValueError("Unsupported or inconsistent attached benchmark")
    if prior_summary["independent_test_accessed"] or not prior_summary["comparison_valid"]:
        raise ValueError("Prior development evidence is invalid")
    checked_file(prior / "train_cohort.json", prior_summary["contract"]["cohort_sha256"])
    checked_file(prior / "train_preparation.json", prior_summary["contract"]["preparation_sha256"])
    checked_file(prior / "fold_assignments.json", prior_summary["contract"]["fold_sha256"])
    if digest(prior / "protocol.json") != prior_summary["contract"]["protocol_sha256"]:
        raise ValueError("Prior protocol changed")
    if digest(args.released_model) != prior_protocol["baseline_sha256"]:
        raise ValueError("Released dataset-specific checkpoint does not match prior run")
    if digest(args.aaindex) != prior_protocol["aaindex_sha256"]:
        raise ValueError("AAindex table differs from the prior run")
    rows = read(prior / "train_cohort.json")
    assignments = np.asarray(read(prior / "fold_assignments.json"), dtype=np.int64)
    if len(assignments) != len(rows) or sorted(np.unique(assignments).tolist()) != list(range(5)):
        raise ValueError("Five-fold assignments are missing or misaligned")
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    groups = np.asarray([row["accession"] for row in rows])
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Training labels must be binary")
    for fold in range(5):
        held = assignments == fold
        if set(groups[held]) & set(groups[~held]):
            raise ValueError(f"Protein leakage in fold {fold}")
        if np.unique(labels[held]).size != 2:
            raise ValueError(f"Fold {fold} has only one class")
    context = np.full(len(rows), np.nan, dtype=np.float64)
    for fold in range(5):
        relative = f"folds/fold_{fold}/context/outer_predictions.npz"
        checked_file(prior / relative, prior_summary["fold_predictions_sha256"][relative])
        with np.load(prior / relative) as saved:
            indices = np.asarray(saved["indices"], dtype=np.int64)
            if (not np.array_equal(indices, np.where(assignments == fold)[0])
                    or not np.array_equal(saved["labels"], labels[indices])):
                raise ValueError(f"Context OOF alignment changed in fold {fold}")
            context[indices] = saved["probabilities"]
    if not np.isfinite(context).all() or ((context < 0) | (context > 1)).any():
        raise ValueError("Context OOF predictions are incomplete")
    contract = {
        "config_sha256": digest(args.config),
        "prior_protocol_sha256": digest(prior / "protocol.json"),
        "prior_development_sha256": digest(prior / "development_summary.json"),
        "cohort_sha256": digest(prior / "train_cohort.json"),
        "fold_sha256": digest(prior / "fold_assignments.json"),
        "released_model_sha256": digest(args.released_model),
        "script_sha256": digest(Path(__file__)),
        "correction_sha256": digest(ROOT / "experiment/src/ubipred/anchored_correction.py"),
        "benchmark": benchmark,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "contract.json"
    if contract_path.exists():
        if read(contract_path) != contract:
            raise ValueError("Existing output uses another protocol; choose a new output directory")
    else:
        atomic_json(contract_path, contract)
    return config, rows, labels, groups, assignments, context, contract


def keras_module():
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    import tensorflow as tf
    import tf_keras
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    return tf, tf_keras


def encode(rows, aaindex_path):
    raw = np.loadtxt(aaindex_path, dtype=np.float64)
    denominator = raw.max(1, keepdims=True) - raw.min(1, keepdims=True)
    if raw.shape[0] != 31 or np.any(denominator == 0):
        raise ValueError("Invalid AAindex table")
    normalized = ((raw - raw.min(1, keepdims=True)) / denominator).T
    lookup = np.concatenate([normalized, np.zeros((1, 31))]).astype(np.float32)
    tokens = np.asarray([[ALPHABET.index(a) for a in r["window_49"]] for r in rows],
                        dtype=np.int32)
    if tokens.shape != (len(rows), 49):
        raise ValueError("Published 49-residue windows are required")
    return [lookup[tokens], np.eye(21, dtype=np.float32)[tokens], tokens]


def fresh_model(keras, released_model, config, seed):
    keras.utils.set_random_seed(seed)
    model = keras.models.clone_model(released_model)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config["learning_rate"], epsilon=config["optimizer_epsilon"]),
        loss="binary_crossentropy",
    )
    return model


def inner_partition(outer: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                    config, fold: int):
    fraction = config["inner_validation_fraction"]
    n_splits = round(1 / fraction)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=42 + fold)
    train_local, val_local = next(splitter.split(outer, labels[outer], groups[outer]))
    train, val = outer[np.sort(train_local)], outer[np.sort(val_local)]
    if (set(groups[train]) & set(groups[val]) or np.unique(labels[train]).size != 2
            or np.unique(labels[val]).size != 2):
        raise ValueError(f"Invalid inner split in fold {fold}")
    return train, val


def predict(model, inputs, indices, batch_size):
    selected = [array[indices] for array in inputs]
    output = model.predict(selected, batch_size=batch_size, verbose=0)
    if output.shape != (len(indices), 2) or not np.isfinite(output).all():
        raise ValueError("Exact-architecture anchor returned invalid predictions")
    if not np.allclose(output.sum(1), 1, atol=1e-5):
        raise ValueError("Anchor predictions do not sum to one")
    return output[:, 1].astype(np.float64)


def train_candidate(args, config, released_model, keras, inputs, labels, train, val,
                    fold, seed, contract):
    folder = args.output_dir / "folds" / f"fold_{fold}" / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    candidate_contract = contract | {
        "fold": fold, "seed": seed,
        "inner_train_sha256": object_digest(train.tolist()),
        "inner_validation_sha256": object_digest(val.tolist()),
    }
    contract_file = folder / "contract.json"
    if contract_file.exists() and read(contract_file) != candidate_contract:
        raise ValueError(f"Candidate contract changed: {folder}")
    if not contract_file.exists():
        atomic_json(contract_file, candidate_contract)
    summary_path = folder / "candidate_summary.json"
    if summary_path.exists():
        summary = read(summary_path)
        checked_file(folder / summary["best_checkpoint"], summary["best_checkpoint_sha256"])
        return summary
    state_path = folder / "state.json"
    state = read(state_path) if state_path.exists() else {
        "completed_epoch": 0, "history": [], "best_epoch": 0,
        "best_mcc": -1.0, "best_auprc": -1.0,
        "best_checkpoint": None, "best_checkpoint_sha256": None,
        "latest_checkpoint": None, "latest_checkpoint_sha256": None,
    }
    if state["completed_epoch"]:
        checked_file(folder / state["latest_checkpoint"], state["latest_checkpoint_sha256"])
        model = keras.models.load_model(folder / state["latest_checkpoint"], compile=True)
        if int(model.optimizer.iterations) <= 0:
            raise ValueError("Resumed model lost optimizer state")
    else:
        model = fresh_model(keras, released_model, config, seed)
    if state["best_checkpoint"]:
        checked_file(folder / state["best_checkpoint"], state["best_checkpoint_sha256"])
    batch_size = config["batch_size"]
    for epoch in range(state["completed_epoch"] + 1, config["epochs_per_seed"] + 1):
        # A fixed epoch-specific permutation allows a resumed run to replay the
        # same epoch ordering, rather than depending on process-level RNG state.
        keras.utils.set_random_seed(seed * 1000 + epoch)
        order = np.random.default_rng(seed * 1000 + epoch).permutation(train)
        fit_inputs = [array[order] for array in inputs]
        model.fit(fit_inputs, np.eye(2, dtype=np.float32)[labels[order]],
                  batch_size=batch_size, epochs=1, shuffle=False, verbose=0)
        probability = predict(model, inputs, val, batch_size)
        metrics = compute_metrics(labels[val], probability, 0.5)
        print(f"fold={fold} seed={seed} epoch={epoch:02d} "
              f"inner_mcc={metrics['mcc']:.5f} inner_auprc={metrics['auprc']:.5f}",
              flush=True)
        improved = (metrics["mcc"] > state["best_mcc"] + 1e-12 or
                    (abs(metrics["mcc"] - state["best_mcc"]) <= 1e-12 and
                     metrics["auprc"] > state["best_auprc"] + 1e-12))
        old_best = state["best_checkpoint"]
        if improved:
            best_name = f"best_epoch_{epoch:02d}.h5"
            model.save(folder / best_name, include_optimizer=False)
            state["best_epoch"] = epoch
            state["best_mcc"] = metrics["mcc"]
            state["best_auprc"] = metrics["auprc"]
            state["best_checkpoint"] = best_name
            state["best_checkpoint_sha256"] = digest(folder / best_name)
        latest_name = f"latest_{epoch % 2}.h5"
        temporary = folder / "latest_pending.h5"
        model.save(temporary, include_optimizer=True)
        os.replace(temporary, folder / latest_name)
        state["latest_checkpoint"] = latest_name
        state["latest_checkpoint_sha256"] = digest(folder / latest_name)
        state["completed_epoch"] = epoch
        state["history"].append({"epoch": epoch, "inner_mcc": metrics["mcc"],
                                 "inner_auprc": metrics["auprc"]})
        atomic_json(state_path, state)
        if improved and old_best and old_best != state["best_checkpoint"]:
            (folder / old_best).unlink(missing_ok=True)
    summary = candidate_contract | {
        "best_epoch": state["best_epoch"],
        "best_mcc": state["best_mcc"],
        "best_auprc": state["best_auprc"],
        "best_checkpoint": state["best_checkpoint"],
        "best_checkpoint_sha256": state["best_checkpoint_sha256"],
        "epochs_completed": state["completed_epoch"],
    }
    atomic_json(summary_path, summary)
    keras.backend.clear_session()
    del model
    gc.collect()
    return summary


def run_fold(args, config, rows, labels, groups, assignments, contract):
    fold = args.fold
    if fold is None or fold not in range(5):
        raise ValueError("--fold must be 0, 1, 2, 3, or 4")
    folder = args.output_dir / "folds" / f"fold_{fold}"
    prediction_path = folder / "anchor_outer_predictions.npz"
    selection_path = folder / "selection.json"
    held = np.where(assignments == fold)[0]
    if selection_path.exists() and prediction_path.exists():
        selection = read(selection_path)
        checked_file(prediction_path, selection["prediction_sha256"])
        with np.load(prediction_path) as saved:
            if (not np.array_equal(saved["indices"], held)
                    or not np.array_equal(saved["labels"], labels[held])):
                raise ValueError(f"Completed fold {fold} alignment changed")
        print(f"fold={fold}: completed anchor predictions reused", flush=True)
        return
    tf, keras = keras_module()
    inputs = encode(rows, args.aaindex)
    released_model = keras.models.load_model(args.released_model, compile=False)
    if [tuple(t.shape) for t in released_model.inputs] != [
            (None, 49, 31), (None, 49, 21), (None, 49)]:
        raise ValueError("The released model has unexpected input shapes")
    outer = np.where(assignments != fold)[0]
    train, val = inner_partition(outer, labels, groups, config, fold)
    candidates = [train_candidate(args, config, released_model, keras, inputs, labels,
                                  train, val, fold, seed, contract)
                  for seed in config["initialization_seeds"]]
    viable = [candidate for candidate in candidates if candidate["best_mcc"] > 0]
    if not viable:
        raise RuntimeError(f"All exact-anchor candidates collapsed in fold {fold}")
    chosen = max(viable, key=lambda item: (item["best_mcc"], item["best_auprc"],
                                           -item["best_epoch"], -item["seed"]))
    best_path = folder / f"seed_{chosen['seed']}" / chosen["best_checkpoint"]
    checked_file(best_path, chosen["best_checkpoint_sha256"])
    model = keras.models.load_model(best_path, compile=False)
    probability = predict(model, inputs, held, config["batch_size"])
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_path, indices=held, labels=labels[held],
                        probabilities=probability)
    selection = {"fold": fold, "seed": chosen["seed"],
                 "best_epoch": chosen["best_epoch"], "inner_mcc": chosen["best_mcc"],
                 "outer_metrics": compute_metrics(labels[held], probability, 0.5),
                 "prediction_sha256": digest(prediction_path),
                 "released_test_accessed": False}
    atomic_json(selection_path, selection)
    print(json.dumps(selection, indent=2), flush=True)
    keras.backend.clear_session()
    del model, released_model
    gc.collect()


def analyze(args, config, labels, assignments, context, contract):
    anchor = np.full(len(labels), np.nan)
    selection = []
    for fold in range(5):
        folder = args.output_dir / "folds" / f"fold_{fold}"
        item = read(folder / "selection.json")
        checked_file(folder / "anchor_outer_predictions.npz", item["prediction_sha256"])
        with np.load(folder / "anchor_outer_predictions.npz") as saved:
            indices = np.asarray(saved["indices"], dtype=np.int64)
            if (not np.array_equal(indices, np.where(assignments == fold)[0])
                    or not np.array_equal(saved["labels"], labels[indices])):
                raise ValueError(f"Anchor OOF alignment changed in fold {fold}")
            anchor[indices] = saved["probabilities"]
        selection.append(item)
    if not np.isfinite(anchor).all():
        raise ValueError("Exact-anchor OOF predictions incomplete")
    corrected = np.full(len(labels), np.nan)
    calibrated = np.full(len(labels), np.nan)
    by_fold = []
    for fold in range(5):
        train, held = assignments != fold, assignments == fold
        model = fit_anchored_correction(labels[train], anchor[train], context[train],
                                        l2_strength=config["correction"]["l2_strength"])
        calibration = fit_anchored_correction(labels[train], anchor[train], context[train],
                                              l2_strength=config["correction"]["l2_strength"],
                                              allow_context=False)
        corrected[held] = model.predict_proba(anchor[held], context[held])
        calibrated[held] = calibration.predict_proba(anchor[held], context[held])
        identity_mcc = compute_metrics(labels[held], anchor[held], 0.5)["mcc"]
        corrected_mcc = compute_metrics(labels[held], corrected[held], 0.5)["mcc"]
        by_fold.append({"fold": fold, "anchor_mcc": identity_mcc,
                        "corrected_mcc": corrected_mcc,
                        "corrected_minus_anchor_mcc": corrected_mcc - identity_mcc,
                        "correction": model.as_dict()})
    metrics = {
        "exact_architecture_anchor": compute_metrics(labels, anchor, 0.5),
        "intercept_only": compute_metrics(labels, calibrated, 0.5),
        "anchored_context_correction": compute_metrics(labels, corrected, 0.5),
        "prior_context": compute_metrics(labels, context, 0.5),
    }
    identity = metrics["exact_architecture_anchor"]
    calibration = metrics["intercept_only"]
    candidate = metrics["anchored_context_correction"]
    rule = config["advance_rule"]
    checks = {
        "mcc_over_anchor": candidate["mcc"] - identity["mcc"] >= rule["minimum_oof_mcc_gain_over_anchor"],
        "mcc_over_intercept_only": candidate["mcc"] - calibration["mcc"] >= rule["minimum_oof_mcc_gain_over_intercept_only"],
        "positive_folds": sum(x["corrected_minus_anchor_mcc"] > 0 for x in by_fold) >= rule["minimum_positive_fold_count"],
        "auroc_noninferior": candidate["auroc"] - identity["auroc"] >= -rule["maximum_auroc_decrease"],
        "sensitivity_noninferior": candidate["sensitivity"] - identity["sensitivity"] >= -rule["maximum_sensitivity_decrease"],
        "specificity_noninferior": candidate["specificity"] - identity["specificity"] >= -rule["maximum_specificity_decrease"],
    }
    final_correction = fit_anchored_correction(labels, anchor, context,
                         l2_strength=config["correction"]["l2_strength"])
    report = {
        "status": "ADVANCE_TO_REFIT_PLANNING" if all(checks.values()) else "STOP_NO_MATERIAL_DEVELOPMENT_GAIN",
        "analysis_role": config["analysis_role"],
        "benchmark": contract["benchmark"], "contract": contract,
        "released_independent_test_accessed": False,
        "comparison_target": "fresh exact-architecture anchor OOF, not the released full-training checkpoint",
        "metrics": metrics, "by_fold": by_fold, "checks": checks,
        "full_oof_correction_if_advanced": final_correction.as_dict(),
        "caveat": "Second-stage cross-fitting over first-level OOF scores is not fully nested end-to-end validation.",
        "next_step": "Do not run released-test inference from this screen. Confirm the method and refit plan before any post-test exploratory assessment.",
    }
    atomic_json(args.output_dir / "development_screen.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preflight", "fold", "analyze"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prior-run-dir", type=Path, required=True)
    parser.add_argument("--released-model", type=Path, required=True)
    parser.add_argument("--aaindex", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    config, rows, labels, groups, assignments, context, contract = load_contract(args)
    if args.stage == "preflight":
        tf, keras = keras_module()
        released = keras.models.load_model(args.released_model, compile=False)
        clone = fresh_model(keras, released, config, 42)
        if clone.count_params() != released.count_params():
            raise ValueError("Cloned architecture parameter count changed")
        if all(np.array_equal(new.numpy(), old.numpy()) for new, old in
               zip(clone.trainable_weights, released.trainable_weights)):
            raise ValueError("Cloned architecture inherited the released trained weights")
        inputs = encode(rows[:8], args.aaindex)
        sample_labels = np.eye(2, dtype=np.float32)[labels[:8]]
        clone.train_on_batch(inputs, sample_labels)
        with tempfile.TemporaryDirectory(dir=args.output_dir) as temp:
            path = Path(temp) / "resume_smoke.h5"
            clone.save(path, include_optimizer=True)
            resumed = keras.models.load_model(path, compile=True)
            if int(resumed.optimizer.iterations) != int(clone.optimizer.iterations):
                raise ValueError("The Keras checkpoint did not preserve optimizer iterations")
        print(json.dumps({"status": "preflight_passed", "benchmark": contract["benchmark"],
                          "training_records": len(rows), "parameters": clone.count_params(),
                          "tensorflow": tf.__version__, "test_accessed": False}, indent=2))
    elif args.stage == "fold":
        run_fold(args, config, rows, labels, groups, assignments, contract)
    else:
        analyze(args, config, labels, assignments, context, contract)


if __name__ == "__main__":
    main()
