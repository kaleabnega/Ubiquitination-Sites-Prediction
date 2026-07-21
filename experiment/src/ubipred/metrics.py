"""Consistent binary metrics for validation and locked test evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    labels: Iterable[int] | np.ndarray,
    probabilities: Iterable[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, object]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_probability = np.asarray(probabilities, dtype=np.float64)
    if y_true.shape != y_probability.shape:
        raise ValueError(
            f"labels and probabilities must have the same shape: "
            f"{y_true.shape} != {y_probability.shape}"
        )

    predictions = (y_probability >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity_denominator = true_negative + false_positive
    specificity = (
        true_negative / specificity_denominator if specificity_denominator else 0.0
    )

    return {
        "threshold": float(threshold),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_probability)),
        "auprc": float(average_precision_score(y_true, y_probability)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "support": {
            "negative": int((y_true == 0).sum()),
            "positive": int((y_true == 1).sum()),
            "total": int(y_true.size),
        },
    }


def select_mcc_threshold(
    labels: Iterable[int] | np.ndarray,
    probabilities: Iterable[float] | np.ndarray,
    minimum: float = 0.05,
    maximum: float = 0.95,
    steps: int = 181,
) -> tuple[float, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_probability = np.asarray(probabilities, dtype=np.float64)
    best_threshold = 0.5
    best_mcc = float("-inf")
    for threshold in np.linspace(minimum, maximum, steps):
        predictions = (y_probability >= threshold).astype(np.int64)
        mcc = matthews_corrcoef(y_true, predictions)
        if mcc > best_mcc:
            best_threshold = float(threshold)
            best_mcc = float(mcc)
    return best_threshold, best_mcc


def write_json(path: str | Path, payload: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
