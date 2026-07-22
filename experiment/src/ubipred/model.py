"""Proposed and MMUbiPred-compatible models on the released inputs."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _keras_glorot_uniform(module: nn.Linear | nn.Conv1d) -> None:
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _keras_he_normal(module: nn.Linear | nn.Conv1d) -> None:
    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _initialize_keras_lstm(lstm: nn.LSTM) -> None:
    """Approximate Keras LSTM defaults, including the unit forget bias."""

    hidden_size = lstm.hidden_size
    for name, parameter in lstm.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(parameter)
        elif "weight_hh" in name:
            nn.init.orthogonal_(parameter)
        elif "bias" in name:
            nn.init.zeros_(parameter)
    with torch.no_grad():
        lstm.bias_ih_l0[hidden_size : hidden_size * 2].fill_(1.0)


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

    branch_names = ("context", "one_hot", "aaindex")
    diagnostic_name = "branch_gates"

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
        active_branches: Sequence[str] = ("context", "one_hot", "aaindex"),
        fusion: str = "gated",
    ) -> None:
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        if embedding_dim % transformer_heads != 0:
            raise ValueError("embedding_dim must be divisible by transformer_heads")
        if any(kernel % 2 == 0 for kernel in conv_kernels):
            raise ValueError("All convolution kernels must be odd")
        if branch_dim % 2 != 0:
            raise ValueError("branch_dim must be even")
        active_branches = tuple(active_branches)
        if not active_branches or len(set(active_branches)) != len(active_branches):
            raise ValueError("active_branches must contain unique branch names")
        unknown_branches = set(active_branches) - set(self.branch_names)
        if unknown_branches:
            raise ValueError(f"Unknown branches: {sorted(unknown_branches)}")
        if fusion not in {"gated", "mean"}:
            raise ValueError("fusion must be 'gated' or 'mean'")

        aaindex_tensor = torch.as_tensor(aaindex_lookup, dtype=torch.float32)
        if tuple(aaindex_tensor.shape) != (21, 31):
            raise ValueError(
                f"Expected AAindex lookup shape (21, 31), got {aaindex_tensor.shape}"
            )

        self.window_size = window_size
        self.center_index = window_size // 2
        self.padding_index = 20
        self.active_branches = active_branches
        self.fusion = fusion

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
        if fusion == "gated" and len(active_branches) > 1:
            self.gate: nn.Module | None = nn.Sequential(
                nn.Linear(branch_dim * len(active_branches), branch_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(branch_dim, len(active_branches)),
            )
        else:
            self.gate = None
        self.classifier = nn.Sequential(
            nn.LayerNorm(branch_dim),
            nn.Linear(branch_dim, branch_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim // 2, 1),
        )

        branch_modules = {
            "context": [
                self.token_embedding,
                self.transformer,
                self.context_projection,
            ],
            "one_hot": [
                self.convolutions,
                self.conv_residual,
                self.conv_projection,
            ],
            "aaindex": [
                self.aaindex_embedding,
                self.biochemical_gru,
                self.biochemical_projection,
            ],
        }
        for branch_name, modules in branch_modules.items():
            if branch_name not in active_branches:
                for module in modules:
                    for parameter in module.parameters():
                        parameter.requires_grad = False
        if "context" not in active_branches:
            self.position_embedding.requires_grad = False

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

        branch_vectors: dict[str, torch.Tensor] = {}

        if "context" in self.active_branches:
            contextual = self.token_embedding(tokens) + self.position_embedding
            contextual = self.transformer(
                contextual,
                src_key_padding_mask=~valid_mask,
            )
            branch_vectors["context"] = self.context_projection(
                torch.cat(
                    [
                        contextual[:, self.center_index, :],
                        self._masked_mean(contextual, valid_mask),
                    ],
                    dim=-1,
                )
            )

        if "one_hot" in self.active_branches:
            one_hot = F.one_hot(tokens, num_classes=21).to(
                self.classifier[1].weight.dtype
            )
            one_hot = one_hot.transpose(1, 2)
            multiscale = torch.cat(
                [convolution(one_hot) for convolution in self.convolutions], dim=1
            )
            multiscale = self.conv_residual(multiscale)
            branch_vectors["one_hot"] = self.conv_projection(
                torch.cat(
                    [multiscale.amax(dim=-1), multiscale.mean(dim=-1)], dim=-1
                )
            )

        if "aaindex" in self.active_branches:
            biochemical = self.aaindex_embedding(tokens)
            biochemical, _ = self.biochemical_gru(biochemical)
            branch_vectors["aaindex"] = self.biochemical_projection(
                torch.cat(
                    [
                        biochemical[:, self.center_index, :],
                        self._masked_mean(biochemical, valid_mask),
                    ],
                    dim=-1,
                )
            )

        active_vectors = [branch_vectors[name] for name in self.active_branches]
        branches = torch.stack(active_vectors, dim=1)
        if self.gate is None:
            active_gates = torch.full(
                (tokens.shape[0], len(self.active_branches)),
                1.0 / len(self.active_branches),
                dtype=branches.dtype,
                device=branches.device,
            )
        else:
            active_gates = torch.softmax(
                self.gate(torch.cat(active_vectors, dim=-1)), dim=-1
            )
        fused = (branches * active_gates.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(fused).squeeze(-1)

        gate_by_name = {
            name: active_gates[:, index]
            for index, name in enumerate(self.active_branches)
        }
        gates = torch.stack(
            [
                gate_by_name.get(name, torch.zeros_like(active_gates[:, 0]))
                for name in self.branch_names
            ],
            dim=1,
        )

        if return_gates:
            return logits, gates
        return logits


class MMUbiPredCompatible(nn.Module):
    """PyTorch reimplementation of the released three-branch topology.

    The layer dimensions, activations, dropout rates, score-level fusion, and
    documented L1 penalties follow the released training notebook. Framework
    initializers and numerical kernels may differ from the original Keras run.
    """

    branch_names = ("aaindex", "one_hot", "embedding")
    diagnostic_name = "branch_positive_probabilities"

    def __init__(
        self,
        aaindex_lookup: np.ndarray | torch.Tensor,
        window_size: int = 49,
        l1_coefficient: float = 1e-4,
    ) -> None:
        super().__init__()
        if window_size != 49:
            raise ValueError("The released MMUbiPred topology requires window_size=49")
        aaindex_tensor = torch.as_tensor(aaindex_lookup, dtype=torch.float32)
        if tuple(aaindex_tensor.shape) != (21, 31):
            raise ValueError(
                f"Expected AAindex lookup shape (21, 31), got {aaindex_tensor.shape}"
            )

        self.window_size = window_size
        self.l1_coefficient = float(l1_coefficient)
        self.aaindex_embedding = nn.Embedding.from_pretrained(
            aaindex_tensor, freeze=True, padding_idx=20
        )
        self.aaindex_lstm = nn.LSTM(31, 64, batch_first=True)
        self.aaindex_dropout_1 = nn.Dropout(0.3)
        self.aaindex_dense = nn.Linear(window_size * 64, 32)
        self.aaindex_dropout_2 = nn.Dropout(0.3)
        self.aaindex_output = nn.Linear(32, 2)

        self.one_hot_conv = nn.Conv1d(21, 16, kernel_size=3, padding=1)
        self.one_hot_pool = nn.MaxPool1d(kernel_size=2)
        self.one_hot_dense = nn.Linear(24 * 16, 512)
        self.one_hot_dropout = nn.Dropout(0.5)
        self.one_hot_output = nn.Linear(512, 2)

        self.sequence_embedding = nn.Embedding(23, 21)
        self.embedding_conv_1 = nn.Conv1d(21, 256, kernel_size=3)
        self.embedding_pool_1 = nn.MaxPool1d(kernel_size=2)
        self.embedding_dropout_1 = nn.Dropout(0.4)
        self.embedding_conv_2 = nn.Conv1d(256, 256, kernel_size=7)
        self.embedding_pool_2 = nn.MaxPool1d(kernel_size=2)
        self.embedding_dropout_2 = nn.Dropout(0.4)
        self.embedding_dense_1 = nn.Linear(8 * 256, 768)
        self.embedding_dropout_3 = nn.Dropout(0.5)
        self.embedding_dense_2 = nn.Linear(768, 256)
        self.embedding_dropout_4 = nn.Dropout(0.5)
        self.embedding_output = nn.Linear(256, 2)

        self.fusion_dense = nn.Linear(6, 6)
        self.fusion_output = nn.Linear(6, 2)
        self._l1_modules = (
            self.aaindex_dense,
            self.aaindex_output,
            self.one_hot_conv,
            self.one_hot_dense,
            self.one_hot_output,
        )
        self._initialize_like_released_keras_model()

    def _initialize_like_released_keras_model(self) -> None:
        _initialize_keras_lstm(self.aaindex_lstm)
        for module in (
            self.aaindex_dense,
            self.aaindex_output,
            self.one_hot_conv,
            self.one_hot_dense,
            self.one_hot_output,
            self.embedding_output,
            self.fusion_dense,
            self.fusion_output,
        ):
            _keras_glorot_uniform(module)
        nn.init.uniform_(self.sequence_embedding.weight, -0.05, 0.05)
        for module in (
            self.embedding_conv_1,
            self.embedding_conv_2,
            self.embedding_dense_1,
            self.embedding_dense_2,
        ):
            _keras_he_normal(module)

    def regularization_loss(self) -> torch.Tensor:
        penalty = sum(module.weight.abs().sum() for module in self._l1_modules)
        return penalty * self.l1_coefficient

    def forward(
        self, tokens: torch.Tensor, return_gates: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2 or tokens.shape[1] != self.window_size:
            raise ValueError(
                f"Expected token shape (batch, {self.window_size}), got {tokens.shape}"
            )

        aaindex = self.aaindex_embedding(tokens)
        aaindex, _ = self.aaindex_lstm(aaindex)
        aaindex = self.aaindex_dropout_1(aaindex).flatten(start_dim=1)
        aaindex = self.aaindex_dropout_2(F.relu(self.aaindex_dense(aaindex)))
        aaindex_probabilities = torch.softmax(self.aaindex_output(aaindex), dim=-1)

        one_hot = F.one_hot(tokens, num_classes=21).to(aaindex.dtype).transpose(1, 2)
        one_hot = self.one_hot_pool(F.relu(self.one_hot_conv(one_hot)))
        one_hot = one_hot.flatten(start_dim=1)
        one_hot = self.one_hot_dropout(F.relu(self.one_hot_dense(one_hot)))
        one_hot_probabilities = torch.softmax(self.one_hot_output(one_hot), dim=-1)

        embedded = self.sequence_embedding(tokens).transpose(1, 2)
        embedded = self.embedding_pool_1(F.relu(self.embedding_conv_1(embedded)))
        embedded = self.embedding_dropout_1(embedded)
        embedded = self.embedding_pool_2(F.relu(self.embedding_conv_2(embedded)))
        embedded = self.embedding_dropout_2(embedded).flatten(start_dim=1)
        embedded = self.embedding_dropout_3(F.relu(self.embedding_dense_1(embedded)))
        embedded = self.embedding_dropout_4(F.relu(self.embedding_dense_2(embedded)))
        embedding_probabilities = torch.softmax(self.embedding_output(embedded), dim=-1)

        branch_probabilities = torch.cat(
            [aaindex_probabilities, one_hot_probabilities, embedding_probabilities],
            dim=-1,
        )
        fused = F.relu(self.fusion_dense(branch_probabilities))
        class_logits = self.fusion_output(fused)
        binary_logits = class_logits[:, 1] - class_logits[:, 0]
        diagnostics = torch.stack(
            [
                aaindex_probabilities[:, 1],
                one_hot_probabilities[:, 1],
                embedding_probabilities[:, 1],
            ],
            dim=1,
        )
        if return_gates:
            return binary_logits, diagnostics
        return binary_logits


def build_model(
    aaindex_lookup: np.ndarray,
    window_size: int,
    model_config: dict[str, object],
) -> nn.Module:
    architecture = str(model_config.get("architecture", "ubifusion_v1"))
    if architecture == "mmubipred_compatible":
        return MMUbiPredCompatible(
            aaindex_lookup=aaindex_lookup,
            window_size=window_size,
            l1_coefficient=float(model_config.get("l1_coefficient", 1e-4)),
        )
    if architecture != "ubifusion_v1":
        raise ValueError(f"Unknown architecture: {architecture}")
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
        active_branches=tuple(
            str(value)
            for value in model_config.get(
                "active_branches", ["context", "one_hot", "aaindex"]
            )
        ),
        fusion=str(model_config.get("fusion", "gated")),
    )
