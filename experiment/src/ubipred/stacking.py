"""Low-capacity, nonnegative residual stacking for two expert logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


def probability_to_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class ResidualStacker:
    intercept: float
    local_weight: float
    context_weight: float
    l2_strength: float

    def predict_proba(
        self, local_probabilities: np.ndarray, context_probabilities: np.ndarray
    ) -> np.ndarray:
        logits = (
            self.intercept
            + self.local_weight * probability_to_logit(local_probabilities)
            + self.context_weight * probability_to_logit(context_probabilities)
        )
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))

    def as_dict(self) -> dict[str, float]:
        return {
            "intercept": self.intercept,
            "local_weight": self.local_weight,
            "context_weight": self.context_weight,
            "l2_strength": self.l2_strength,
        }


def fit_residual_stacker(
    labels: Sequence[int] | np.ndarray,
    local_probabilities: np.ndarray,
    context_probabilities: np.ndarray,
    l2_strength: float = 0.01,
    max_iterations: int = 100,
) -> ResidualStacker:
    """Fit BCE stacking with nonnegative weights and a Short-Range Expert prior."""

    if l2_strength < 0:
        raise ValueError("l2_strength must be nonnegative")
    y = torch.as_tensor(np.asarray(labels), dtype=torch.float64)
    local = torch.as_tensor(
        probability_to_logit(local_probabilities), dtype=torch.float64
    )
    context = torch.as_tensor(
        probability_to_logit(context_probabilities), dtype=torch.float64
    )
    if y.ndim != 1 or y.shape != local.shape or y.shape != context.shape:
        raise ValueError("labels and both probability arrays must be aligned vectors")

    intercept = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    raw_weights = torch.nn.Parameter(
        torch.tensor([0.5413248546, -6.0], dtype=torch.float64)
    )
    optimizer = torch.optim.LBFGS(
        [intercept, raw_weights],
        lr=0.5,
        max_iter=max_iterations,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        weights = torch.nn.functional.softplus(raw_weights)
        logits = intercept + weights[0] * local + weights[1] * context
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        prior = (weights[0] - 1.0).square() + weights[1].square()
        loss = bce + 0.5 * l2_strength * prior
        loss.backward()
        return loss

    optimizer.step(closure)
    weights = torch.nn.functional.softplus(raw_weights).detach().cpu().numpy()
    return ResidualStacker(
        intercept=float(intercept.detach()),
        local_weight=float(weights[0]),
        context_weight=float(weights[1]),
        l2_strength=float(l2_strength),
    )
