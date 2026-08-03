"""Descriptive diagnostics for aligned binary model predictions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ubipred.paired_statistics import _compute_metrics, mcnemar_exact


def _as_aligned_vectors(
    labels: Sequence[int] | np.ndarray,
    reference_probabilities: Sequence[float] | np.ndarray,
    candidate_probabilities: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    reference = np.asarray(reference_probabilities, dtype=np.float64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    if y.ndim != 1 or reference.shape != y.shape or candidate.shape != y.shape:
        raise ValueError("labels and both probability arrays must be aligned vectors")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("labels must contain only zero and one")
    for name, probabilities in (
        ("reference", reference),
        ("candidate", candidate),
    ):
        if not np.all(np.isfinite(probabilities)):
            raise ValueError(f"{name} probabilities must be finite")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError(f"{name} probabilities must lie within [0, 1]")
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("both classes are required")
    return y, reference, candidate


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _correctness_counts(
    labels: np.ndarray,
    reference_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> dict[str, int | float]:
    reference_correct = reference_predictions == labels
    candidate_correct = candidate_predictions == labels
    both_correct = int(np.sum(reference_correct & candidate_correct))
    reference_only = int(np.sum(reference_correct & ~candidate_correct))
    candidate_only = int(np.sum(~reference_correct & candidate_correct))
    both_wrong = int(np.sum(~reference_correct & ~candidate_correct))
    total = len(labels)
    discordant = reference_only + candidate_only
    return {
        "both_correct": both_correct,
        "reference_only_correct": reference_only,
        "candidate_only_correct": candidate_only,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "disagreement_fraction": float(discordant / total),
        "candidate_fraction_of_resolved_disagreements": (
            float(candidate_only / discordant) if discordant else 0.5
        ),
        "candidate_minus_reference_unique_correct": candidate_only - reference_only,
    }


def _metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Add confusion/support fields to the dependency-light metric core."""

    result: dict[str, object] = {
        "threshold": float(threshold),
        **_compute_metrics(labels, probabilities, threshold),
    }
    predictions = probabilities >= threshold
    negative = labels == 0
    positive = labels == 1
    true_negative = int(np.sum(~predictions & negative))
    false_positive = int(np.sum(predictions & negative))
    false_negative = int(np.sum(~predictions & positive))
    true_positive = int(np.sum(predictions & positive))
    result["confusion_matrix"] = [
        [true_negative, false_positive],
        [false_negative, true_positive],
    ]
    result["support"] = {
        "negative": int(np.sum(negative)),
        "positive": int(np.sum(positive)),
        "total": int(len(labels)),
    }
    return result


def analyze_pair(
    labels: Sequence[int] | np.ndarray,
    reference_probabilities: Sequence[float] | np.ndarray,
    candidate_probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
) -> dict[str, object]:
    """Measure complementary errors without fitting a fusion model."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie within [0, 1]")
    labels_array, reference, candidate = _as_aligned_vectors(
        labels, reference_probabilities, candidate_probabilities
    )
    reference_predictions = (reference >= threshold).astype(np.int64)
    candidate_predictions = (candidate >= threshold).astype(np.int64)
    reference_correct = reference_predictions == labels_array
    candidate_correct = candidate_predictions == labels_array
    correctness = _correctness_counts(
        labels_array, reference_predictions, candidate_predictions
    )
    class_conditioned: dict[str, object] = {}
    for label, name in ((0, "negative"), (1, "positive")):
        mask = labels_array == label
        class_conditioned[name] = {
            "support": int(np.sum(mask)),
            **_correctness_counts(
                labels_array[mask],
                reference_predictions[mask],
                candidate_predictions[mask],
            ),
        }

    reference_metrics = _metrics(labels_array, reference, threshold)
    candidate_metrics = _metrics(labels_array, candidate, threshold)
    oracle_correct = reference_correct | candidate_correct
    oracle_accuracy = float(np.mean(oracle_correct))
    best_component_accuracy = max(
        float(reference_metrics["accuracy"]),
        float(candidate_metrics["accuracy"]),
    )
    available_oracle_gain = oracle_accuracy - best_component_accuracy

    return {
        "reference_name": reference_name,
        "candidate_name": candidate_name,
        "threshold": float(threshold),
        "support": reference_metrics["support"],
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_minus_reference": {
            metric: float(candidate_metrics[metric])
            - float(reference_metrics[metric])
            for metric in (
                "mcc",
                "accuracy",
                "sensitivity",
                "specificity",
                "precision",
                "f1",
                "auroc",
                "auprc",
            )
        },
        "paired_correctness": correctness,
        "class_conditioned_paired_correctness": class_conditioned,
        "dependence": {
            "probability_pearson": _safe_correlation(reference, candidate),
            "negative_probability_pearson": _safe_correlation(
                reference[labels_array == 0], candidate[labels_array == 0]
            ),
            "positive_probability_pearson": _safe_correlation(
                reference[labels_array == 1], candidate[labels_array == 1]
            ),
            "correctness_pearson": _safe_correlation(
                reference_correct.astype(np.float64),
                candidate_correct.astype(np.float64),
            ),
            "mean_absolute_probability_gap": float(
                np.mean(np.abs(reference - candidate))
            ),
        },
        "label_informed_oracle_upper_bound": {
            "accuracy": oracle_accuracy,
            "gain_over_best_component_accuracy": available_oracle_gain,
            "both_wrong_fraction": float(correctness["both_wrong"] / len(labels_array)),
            "deployable": False,
            "interpretation": (
                "Upper bound that uses the true label to choose whichever "
                "expert is correct; it is not a trainable or reportable model."
            ),
        },
        "mcnemar_site_level_descriptive": mcnemar_exact(
            labels_array,
            candidate,
            reference,
            threshold,
        ),
    }


def analyze_folds(
    labels: Sequence[int] | np.ndarray,
    reference_probabilities: Sequence[float] | np.ndarray,
    candidate_probabilities: Sequence[float] | np.ndarray,
    fold_assignments: Sequence[int] | np.ndarray,
    *,
    threshold: float = 0.5,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
) -> list[dict[str, object]]:
    """Return the same descriptive analysis separately for each fold."""

    labels_array, reference, candidate = _as_aligned_vectors(
        labels, reference_probabilities, candidate_probabilities
    )
    folds = np.asarray(fold_assignments, dtype=np.int64)
    if folds.shape != labels_array.shape or np.any(folds < 0):
        raise ValueError("fold assignments must be aligned nonnegative integers")
    results: list[dict[str, object]] = []
    for fold in sorted(np.unique(folds).tolist()):
        mask = folds == fold
        result = analyze_pair(
            labels_array[mask],
            reference[mask],
            candidate[mask],
            threshold=threshold,
            reference_name=reference_name,
            candidate_name=candidate_name,
        )
        results.append({"fold": int(fold), **result})
    return results
