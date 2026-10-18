"""Bounded context corrections that leave the baseline logit coefficient fixed at one."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector")
    return result


def _logit(probabilities: np.ndarray) -> np.ndarray:
    probabilities = _vector(probabilities, "probabilities")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def _context_feature(baseline: np.ndarray, context: np.ndarray) -> np.ndarray:
    # Fixed scaling bounds the maximum per-site contribution of this feature.
    return np.clip(_logit(context) - _logit(baseline), -4.0, 4.0) / 4.0


@dataclass(frozen=True)
class AnchoredCorrection:
    intercept: float
    context_weight: float
    l2_strength: float

    def predict_proba(self, baseline: np.ndarray, context: np.ndarray) -> np.ndarray:
        baseline = _vector(baseline, "baseline")
        context = _vector(context, "context")
        if baseline.shape != context.shape:
            raise ValueError("baseline and context predictions must be aligned")
        score = (
            _logit(baseline)
            + self.intercept
            + self.context_weight * _context_feature(baseline, context)
        )
        return expit(score)

    def as_dict(self) -> dict[str, float]:
        return {
            "intercept": self.intercept,
            "context_weight": self.context_weight,
            "l2_strength": self.l2_strength,
        }


def fit_anchored_correction(
    labels: np.ndarray,
    baseline: np.ndarray,
    context: np.ndarray,
    *,
    l2_strength: float = 0.1,
    allow_context: bool = True,
) -> AnchoredCorrection:
    """Fit BCE on development predictions only; the baseline coefficient stays one.

    The intercept is bounded to +/-0.5 logits and the context coefficient to
    [0, 1]. A zero context coefficient exactly recovers intercept-only fitting;
    zero intercept also recovers the untouched baseline.
    """
    y = _vector(labels, "labels")
    p = _vector(baseline, "baseline")
    q = _vector(context, "context")
    if y.shape != p.shape or y.shape != q.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("aligned binary labels and probabilities are required")
    if y.size == 0 or np.unique(y).size != 2:
        raise ValueError("both classes are required")
    if l2_strength <= 0:
        raise ValueError("l2_strength must be positive")
    base_logit = _logit(p)
    feature = _context_feature(p, q) if allow_context else np.zeros_like(p)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, weight = parameters
        score = base_logit + intercept + weight * feature
        probability = expit(score)
        loss = np.mean(np.logaddexp(0.0, score) - y * score)
        loss += 0.5 * l2_strength * (intercept**2 + weight**2)
        error = probability - y
        gradient = np.array([
            np.mean(error) + l2_strength * intercept,
            np.mean(error * feature) + l2_strength * weight,
        ])
        return float(loss), gradient

    bounds = [(-0.5, 0.5), (0.0, 1.0 if allow_context else 0.0)]
    optimized = minimize(objective, np.zeros(2), jac=True, bounds=bounds,
                         method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-12})
    if not optimized.success:
        raise RuntimeError(f"Anchored correction optimizer failed: {optimized.message}")
    return AnchoredCorrection(
        intercept=float(optimized.x[0]),
        context_weight=float(optimized.x[1]),
        l2_strength=l2_strength,
    )
