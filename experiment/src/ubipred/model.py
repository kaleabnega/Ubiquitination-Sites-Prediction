"""UbiFusionNet v1: feature-level gated fusion on the MMUbiPred inputs."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class UbiFusionNet(nn.Module):
    """Three feature extractors fused with sample-dependent branch gates."""

    def __init__(
        self,
        aaindex_lookup: np.ndarray | torch.Tensor,
        window_size: int = 49,
        embedding_dim: int = 64,
        branch_dim: int = 128,
        conv_channels: int = 32,
        conv_kernels: Sequence[int] = (3, 5, 7, 11),
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ff_dim: int = 192,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        if embedding_dim % transformer_heads != 0:
            raise ValueError("embedding_dim must be divisible by transformer_heads")
        if any(kernel % 2 == 0 for kernel in conv_kernels):
            raise ValueError("All convolution kernels must be odd")

        aaindex_tensor = torch.as_tensor(aaindex_lookup, dtype=torch.float32)
        if tuple(aaindex_tensor.shape) != (21, 31):
            raise ValueError(
                f"Expected AAindex lookup shape (21, 31), got {aaindex_tensor.shape}"
            )

        self.window_size = window_size
        self.center_index = window_size // 2
        self.padding_index = 20

        # Contextual amino-acid branch.
        self.token_embedding = nn.Embedding(
            num_embeddings=21,
            embedding_dim=embedding_dim,
            padding_idx=self.padding_index,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, window_size, embedding_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.context_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Exact one-hot identities processed at several motif scales.
        self.convolutions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        21,
                        conv_channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(conv_channels),
                    nn.GELU(),
                )
                for kernel in conv_kernels
            ]
        )
        total_conv_channels = conv_channels * len(conv_kernels)
        self.conv_residual = ResidualConvBlock(total_conv_channels, dropout=dropout)
        self.conv_projection = nn.Sequential(
            nn.Linear(total_conv_channels * 2, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # The same normalized 31 AAindex properties used by MMUbiPred.
        self.aaindex_embedding = nn.Embedding.from_pretrained(
            aaindex_tensor,
            freeze=True,
            padding_idx=self.padding_index,
        )
        self.biochemical_gru = nn.GRU(
            input_size=31,
            hidden_size=branch_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.biochemical_projection = nn.Sequential(
            nn.Linear(branch_dim * 2, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Feature-level fusion: gates differ for every candidate lysine.
        self.gate = nn.Sequential(
            nn.Linear(branch_dim * 3, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, 3),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(branch_dim),
            nn.Linear(branch_dim, branch_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim // 2, 1),
        )

    @staticmethod
    def _masked_mean(features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = valid_mask.unsqueeze(-1).to(features.dtype)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (features * weights).sum(dim=1) / denominator

    def forward(
        self, tokens: torch.Tensor, return_gates: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2 or tokens.shape[1] != self.window_size:
            raise ValueError(
                f"Expected token shape (batch, {self.window_size}), got {tokens.shape}"
            )

        valid_mask = tokens.ne(self.padding_index)

        contextual = self.token_embedding(tokens) + self.position_embedding
        contextual = self.transformer(
            contextual,
            src_key_padding_mask=~valid_mask,
        )
        contextual_vector = self.context_projection(
            torch.cat(
                [
                    contextual[:, self.center_index, :],
                    self._masked_mean(contextual, valid_mask),
                ],
                dim=-1,
            )
        )

        one_hot = F.one_hot(tokens, num_classes=21).to(contextual.dtype)
        one_hot = one_hot.transpose(1, 2)
        multiscale = torch.cat(
            [convolution(one_hot) for convolution in self.convolutions], dim=1
        )
        multiscale = self.conv_residual(multiscale)
        conv_vector = self.conv_projection(
            torch.cat(
                [multiscale.amax(dim=-1), multiscale.mean(dim=-1)], dim=-1
            )
        )

        biochemical = self.aaindex_embedding(tokens)
        biochemical, _ = self.biochemical_gru(biochemical)
        biochemical_vector = self.biochemical_projection(
            torch.cat(
                [
                    biochemical[:, self.center_index, :],
                    self._masked_mean(biochemical, valid_mask),
                ],
                dim=-1,
            )
        )

        branches = torch.stack(
            [contextual_vector, conv_vector, biochemical_vector], dim=1
        )
        gates = torch.softmax(
            self.gate(
                torch.cat(
                    [contextual_vector, conv_vector, biochemical_vector], dim=-1
                )
            ),
            dim=-1,
        )
        fused = (branches * gates.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(fused).squeeze(-1)

        if return_gates:
            return logits, gates
        return logits


def build_model(
    aaindex_lookup: np.ndarray,
    window_size: int,
    model_config: dict[str, object],
) -> UbiFusionNet:
    return UbiFusionNet(
        aaindex_lookup=aaindex_lookup,
        window_size=window_size,
        embedding_dim=int(model_config["embedding_dim"]),
        branch_dim=int(model_config["branch_dim"]),
        conv_channels=int(model_config["conv_channels"]),
        conv_kernels=tuple(int(value) for value in model_config["conv_kernels"]),
        transformer_heads=int(model_config["transformer_heads"]),
        transformer_layers=int(model_config["transformer_layers"]),
        transformer_ff_dim=int(model_config["transformer_ff_dim"]),
        dropout=float(model_config["dropout"]),
    )
