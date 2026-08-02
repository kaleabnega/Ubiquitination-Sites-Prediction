"""Paired uncertainty estimates for frozen model predictions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import binomtest


DEFAULT_METRICS = (
    "mcc",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "auroc",
    "auprc",
)


def _ranking_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    """Compute ROC AUC and average precision with exact tie grouping."""

    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if not positive_count or not negative_count:
        raise ValueError("Both classes are required for ranking metrics")
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_scores = probabilities[order]
    sorted_labels = labels[order]
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    cumulative_positive = np.cumsum(sorted_labels == 1)[group_ends]
    cumulative_total = group_ends + 1
    cumulative_negative = cumulative_total - cumulative_positive

    true_positive_rate = cumulative_positive / positive_count
    false_positive_rate = cumulative_negative / negative_count
    roc_y = np.r_[0.0, true_positive_rate]
    roc_x = np.r_[0.0, false_positive_rate]
    auroc = float(
        np.sum(
            (roc_y[1:] + roc_y[:-1])
            * (roc_x[1:] - roc_x[:-1])
            / 2.0
        )
    )
    recall_increments = np.diff(np.r_[0.0, true_positive_rate])
    precision = cumulative_positive / cumulative_total
    auprc = float(np.sum(recall_increments * precision))
    return auroc, auprc


def _compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    positive = labels == 1
    negative = ~positive
    true_positive = int(np.sum(predictions & positive))
    false_positive = int(np.sum(predictions & negative))
    true_negative = int(np.sum(~predictions & negative))
    false_negative = int(np.sum(~predictions & positive))
    total = len(labels)
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else 0.0
    )
    denominator = np.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = (
        (
            true_positive * true_negative
            - false_positive * false_negative
        )
        / denominator
        if denominator
        else 0.0
    )
    auroc, auprc = _ranking_metrics(labels, probabilities)
    return {
        "mcc": float(mcc),
        "accuracy": float((true_positive + true_negative) / total),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "auroc": auroc,
        "auprc": auprc,
    }


def mcnemar_exact(
    labels: np.ndarray,
    candidate_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    candidate_correct = (
        np.asarray(candidate_probabilities) >= threshold
    ).astype(np.int64) == labels
    reference_correct = (
        np.asarray(reference_probabilities) >= threshold
    ).astype(np.int64) == labels
    candidate_only = int(np.sum(candidate_correct & ~reference_correct))
    reference_only = int(np.sum(~candidate_correct & reference_correct))
    discordant = candidate_only + reference_only
    p_value = (
        float(
            binomtest(
                min(candidate_only, reference_only),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "candidate_only_correct": candidate_only,
        "reference_only_correct": reference_only,
        "discordant_total": discordant,
        "two_sided_exact_p_value": p_value,
    }


def paired_stratified_bootstrap(
    labels: np.ndarray,
    candidate_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    *,
    threshold: float,
    replicates: int,
    seed: int,
    confidence_level: float,
    metrics: Sequence[str] = DEFAULT_METRICS,
    progress_every: int | None = None,
) -> dict[str, object]:
    """Bootstrap paired metric differences while preserving class support."""

    labels = np.asarray(labels, dtype=np.int64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    reference = np.asarray(reference_probabilities, dtype=np.float64)
    if labels.ndim != 1 or candidate.shape != labels.shape:
        raise ValueError("Candidate predictions and labels must align")
    if reference.shape != labels.shape:
        raise ValueError("Reference predictions and labels must align")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Both classes are required for stratified bootstrap")

    point_candidate = _compute_metrics(labels, candidate, threshold)
    point_reference = _compute_metrics(labels, reference, threshold)
    metric_names = tuple(metrics)
    differences = {
        metric: np.empty(replicates, dtype=np.float64)
        for metric in metric_names
    }
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        indices = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        sampled_labels = labels[indices]
        candidate_metrics = _compute_metrics(
            sampled_labels, candidate[indices], threshold
        )
        reference_metrics = _compute_metrics(
            sampled_labels, reference[indices], threshold
        )
        for metric in metric_names:
            differences[metric][replicate] = (
                float(candidate_metrics[metric])
                - float(reference_metrics[metric])
            )
        if (
            progress_every is not None
            and progress_every > 0
            and (replicate + 1) % progress_every == 0
        ):
            print(
                f"paired bootstrap completed={replicate + 1}/{replicates}",
                flush=True,
            )

    alpha = 1.0 - confidence_level
    result: dict[str, object] = {}
    for metric in metric_names:
        values = differences[metric]
        nonpositive = (np.count_nonzero(values <= 0.0) + 1) / (
            replicates + 1
        )
        nonnegative = (np.count_nonzero(values >= 0.0) + 1) / (
            replicates + 1
        )
        result[metric] = {
            "candidate": float(point_candidate[metric]),
            "reference": float(point_reference[metric]),
            "difference": (
                float(point_candidate[metric])
                - float(point_reference[metric])
            ),
            "confidence_interval": [
                float(np.quantile(values, alpha / 2.0)),
                float(np.quantile(values, 1.0 - alpha / 2.0)),
            ],
            "paired_bootstrap_two_sided_p_value": min(
                1.0, 2.0 * min(nonpositive, nonnegative)
            ),
        }
    return {
        "method": (
            "paired stratified nonparametric bootstrap of candidate-minus-"
            "reference metric differences"
        ),
        "replicates": replicates,
        "seed": seed,
        "confidence_level": confidence_level,
        "metrics": result,
    }


def paired_cluster_bootstrap(
    labels: np.ndarray,
    candidate_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    groups: np.ndarray,
    *,
    threshold: float,
    replicates: int,
    seed: int,
    confidence_level: float,
    metrics: Sequence[str] = DEFAULT_METRICS,
    progress_every: int | None = None,
) -> dict[str, object]:
    """Bootstrap paired differences by resampling whole protein groups."""

    labels = np.asarray(labels, dtype=np.int64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    reference = np.asarray(reference_probabilities, dtype=np.float64)
    groups = np.asarray(groups)
    if labels.ndim != 1 or candidate.shape != labels.shape:
        raise ValueError("Candidate predictions and labels must align")
    if reference.shape != labels.shape or groups.shape != labels.shape:
        raise ValueError("Reference predictions and groups must align")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("Both classes are required for cluster bootstrap")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two protein groups are required")
    group_indices = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }

    point_candidate = _compute_metrics(labels, candidate, threshold)
    point_reference = _compute_metrics(labels, reference, threshold)
    metric_names = tuple(metrics)
    differences = {
        metric: np.empty(replicates, dtype=np.float64)
        for metric in metric_names
    }
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        for _ in range(100):
            sampled_groups = rng.choice(
                unique_groups, size=len(unique_groups), replace=True
            )
            indices = np.concatenate(
                [group_indices[group] for group in sampled_groups]
            )
            sampled_labels = labels[indices]
            if np.any(sampled_labels == 0) and np.any(sampled_labels == 1):
                break
        else:
            raise RuntimeError(
                "Unable to draw a cluster bootstrap sample with both classes"
            )
        candidate_metrics = _compute_metrics(
            sampled_labels, candidate[indices], threshold
        )
        reference_metrics = _compute_metrics(
            sampled_labels, reference[indices], threshold
        )
        for metric in metric_names:
            differences[metric][replicate] = (
                float(candidate_metrics[metric])
                - float(reference_metrics[metric])
            )
        if (
            progress_every is not None
            and progress_every > 0
            and (replicate + 1) % progress_every == 0
        ):
            print(
                f"paired cluster bootstrap completed={replicate + 1}/"
                f"{replicates}",
                flush=True,
            )

    alpha = 1.0 - confidence_level
    result: dict[str, object] = {}
    for metric in metric_names:
        values = differences[metric]
        nonpositive = (np.count_nonzero(values <= 0.0) + 1) / (
            replicates + 1
        )
        nonnegative = (np.count_nonzero(values >= 0.0) + 1) / (
            replicates + 1
        )
        result[metric] = {
            "candidate": float(point_candidate[metric]),
            "reference": float(point_reference[metric]),
            "difference": (
                float(point_candidate[metric])
                - float(point_reference[metric])
            ),
            "confidence_interval": [
                float(np.quantile(values, alpha / 2.0)),
                float(np.quantile(values, 1.0 - alpha / 2.0)),
            ],
            "paired_bootstrap_two_sided_p_value": min(
                1.0, 2.0 * min(nonpositive, nonnegative)
            ),
        }
    return {
        "method": (
            "paired nonparametric cluster bootstrap of candidate-minus-"
            "reference metric differences; whole canonical-accession "
            "protein groups resampled with replacement"
        ),
        "resampling_unit": "canonical_accession",
        "unique_groups": int(len(unique_groups)),
        "replicates": replicates,
        "seed": seed,
        "confidence_level": confidence_level,
        "metrics": result,
    }
