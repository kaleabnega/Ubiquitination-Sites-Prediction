"""Binary objectives for controlled label-robustness experiments."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class AsymmetricNegativeLabelSmoothingLoss(nn.Module):
    """Smooth only observed negatives toward the positive class."""

    def __init__(self, negative_target: float = 0.05) -> None:
        super().__init__()
        if not 0.0 < negative_target < 0.5:
            raise ValueError("negative_target must lie strictly within (0, 0.5)")
        self.negative_target = float(negative_target)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        targets = torch.where(
            labels > 0.5,
            torch.ones_like(labels),
            torch.full_like(labels, self.negative_target),
        )
        return F.binary_cross_entropy_with_logits(logits, targets)


class BinaryGeneralizedCrossEntropyLoss(nn.Module):
    """Generalized cross entropy using the probability of the given label."""

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        if not 0.0 < q <= 1.0:
            raise ValueError("q must lie within (0, 1]")
        self.q = float(q)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        positive_probabilities = torch.sigmoid(logits)
        label_probabilities = torch.where(
            labels > 0.5,
            positive_probabilities,
            1.0 - positive_probabilities,
        )
        label_probabilities = label_probabilities.clamp(
            min=torch.finfo(logits.dtype).eps,
            max=1.0,
        )
        return ((1.0 - label_probabilities.pow(self.q)) / self.q).mean()


def normalized_loss_config(
    config: Mapping[str, object] | None,
) -> dict[str, object]:
    if config is None:
        return {"name": "bce"}
    normalized = dict(config)
    if "name" not in normalized:
        raise ValueError("Loss configuration requires a name")
    return normalized


def build_binary_loss(
    config: Mapping[str, object] | None,
) -> nn.Module:
    normalized = normalized_loss_config(config)
    name = str(normalized["name"])
    if name == "bce":
        if set(normalized) != {"name"}:
            raise ValueError("BCE loss does not accept additional parameters")
        return nn.BCEWithLogitsLoss()
    if name == "asymmetric_negative_label_smoothing":
        if set(normalized) != {"name", "negative_target"}:
            raise ValueError(
                "Asymmetric smoothing requires only negative_target"
            )
        return AsymmetricNegativeLabelSmoothingLoss(
            negative_target=float(normalized["negative_target"])
        )
    if name == "generalized_cross_entropy":
        if set(normalized) != {"name", "q"}:
            raise ValueError("Generalized cross entropy requires only q")
        return BinaryGeneralizedCrossEntropyLoss(q=float(normalized["q"]))
    raise ValueError(f"Unsupported binary loss: {name}")
