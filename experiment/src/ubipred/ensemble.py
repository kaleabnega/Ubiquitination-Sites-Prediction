"""Parameter-free probability fusion for frozen prediction vectors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def equal_probability_average(
    probability_vectors: Sequence[np.ndarray],
) -> np.ndarray:
    """Return an equal-weight average after strict alignment validation."""

    if len(probability_vectors) < 2:
        raise ValueError("At least two probability vectors are required")
    arrays = [
        np.asarray(probabilities, dtype=np.float64)
        for probabilities in probability_vectors
    ]
    expected_shape = arrays[0].shape
    if len(expected_shape) != 1:
        raise ValueError("Probability vectors must be one-dimensional")
    for probabilities in arrays:
        if probabilities.shape != expected_shape:
            raise ValueError("Probability vectors must have identical shapes")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Probability vectors must contain finite values")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("Probabilities must lie within [0, 1]")
    return np.mean(np.stack(arrays, axis=0), axis=0)
