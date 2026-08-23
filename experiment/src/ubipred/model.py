"""Proposed and MMUbiPred-compatible models on the released inputs."""

from __future__ import annotations

import json
from pathlib import Path
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


class ESM2CrossFusion(nn.Module):
    """Frozen ESM-2 representations fused with local and biochemical features.

    The protein language model supplies contextual residue representations.
    A dilated convolution branch retains explicit local motif information, and
    an AAindex branch retains the same physicochemical inputs used by MMUbiPred.
    Sample-dependent gates and a residual projection combine the three views.
    """

    branch_names = ("esm2", "local_motif", "aaindex")
    diagnostic_name = "branch_gates"
    alphabet = "ARNDCQEGHILKMFPSTWYV-"

    def __init__(
        self,
        aaindex_lookup: np.ndarray | torch.Tensor,
        backbone: nn.Module,
        residue_token_lookup: Sequence[int],
        cls_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        window_size: int = 49,
        branch_dim: int = 128,
        conv_channels: int = 32,
        conv_kernels: Sequence[int] = (3, 5, 7),
        conv_dilations: Sequence[int] = (1, 2, 3),
        dropout: float = 0.25,
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 0,
        preserve_backbone_trainability: bool = False,
        aaindex_gru_hidden_dim: int = 0,
        auxiliary_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        if len(conv_kernels) != len(conv_dilations) or not conv_kernels:
            raise ValueError("conv_kernels and conv_dilations must have equal length")
        if any(kernel % 2 == 0 for kernel in conv_kernels):
            raise ValueError("All convolution kernels must be odd")
        if any(dilation <= 0 for dilation in conv_dilations):
            raise ValueError("All convolution dilations must be positive")
        if branch_dim <= 0 or conv_channels <= 0:
            raise ValueError("branch_dim and conv_channels must be positive")
        if unfreeze_last_n_layers < 0:
            raise ValueError("unfreeze_last_n_layers cannot be negative")
        if preserve_backbone_trainability and unfreeze_last_n_layers:
            raise ValueError(
                "Cannot request explicit layer unfreezing while preserving "
                "backbone trainability"
            )
        if aaindex_gru_hidden_dim < 0:
            raise ValueError("aaindex_gru_hidden_dim cannot be negative")
        if auxiliary_loss_weight < 0:
            raise ValueError("auxiliary_loss_weight cannot be negative")

        aaindex_tensor = torch.as_tensor(aaindex_lookup, dtype=torch.float32)
        if tuple(aaindex_tensor.shape) != (21, 31):
            raise ValueError(
                f"Expected AAindex lookup shape (21, 31), got {aaindex_tensor.shape}"
            )
        residue_ids = torch.as_tensor(residue_token_lookup, dtype=torch.long)
        if tuple(residue_ids.shape) != (21,):
            raise ValueError(
                "residue_token_lookup must contain the 20 amino acids and padding"
            )

        hidden_size = int(getattr(getattr(backbone, "config", None), "hidden_size", 0))
        if hidden_size <= 0:
            raise ValueError("backbone.config.hidden_size must be a positive integer")

        self.window_size = window_size
        self.center_index = window_size // 2
        self.padding_index = 20
        self.cls_token_id = int(cls_token_id)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.esm_backbone = backbone
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        self._last_auxiliary_logits: torch.Tensor | None = None
        self.register_buffer("residue_token_lookup", residue_ids, persistent=True)

        if not preserve_backbone_trainability:
            for parameter in self.esm_backbone.parameters():
                parameter.requires_grad = not freeze_backbone
        if (
            not preserve_backbone_trainability
            and freeze_backbone
            and unfreeze_last_n_layers
        ):
            encoder = getattr(self.esm_backbone, "encoder", None)
            layers = getattr(encoder, "layer", None)
            if layers is None:
                raise ValueError(
                    "Cannot unfreeze layers: backbone has no encoder.layer collection"
                )
            if unfreeze_last_n_layers > len(layers):
                raise ValueError(
                    "unfreeze_last_n_layers exceeds the number of backbone layers"
                )
            for layer in layers[-unfreeze_last_n_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        self.esm_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.local_convolutions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        20,
                        conv_channels,
                        kernel_size=kernel,
                        dilation=dilation,
                        padding=dilation * (kernel // 2),
                        bias=False,
                    ),
                    nn.BatchNorm1d(conv_channels),
                    nn.GELU(),
                )
                for kernel, dilation in zip(conv_kernels, conv_dilations)
            ]
        )
        local_channels = conv_channels * len(conv_kernels)
        self.local_residual = ResidualConvBlock(local_channels, dropout=dropout)
        self.local_projection = nn.Sequential(
            nn.Linear(local_channels * 2, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.aaindex_embedding = nn.Embedding.from_pretrained(
            aaindex_tensor,
            freeze=True,
            padding_idx=self.padding_index,
        )
        self.aaindex_gru: nn.GRU | None
        if aaindex_gru_hidden_dim:
            self.aaindex_gru = nn.GRU(
                input_size=31,
                hidden_size=aaindex_gru_hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            aaindex_projection_input = aaindex_gru_hidden_dim * 4
        else:
            self.aaindex_gru = None
            aaindex_projection_input = 31 * 2
        self.aaindex_projection = nn.Sequential(
            nn.Linear(aaindex_projection_input, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
        )

        concatenated_dim = branch_dim * len(self.branch_names)
        self.gate = nn.Sequential(
            nn.Linear(concatenated_dim, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, len(self.branch_names)),
        )
        self.residual_fusion = nn.Linear(concatenated_dim, branch_dim)
        self.fusion_norm = nn.LayerNorm(branch_dim)
        self.classifier = nn.Sequential(
            nn.Linear(branch_dim, branch_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim // 2, 1),
        )
        self.auxiliary_heads = nn.ModuleList()
        if self.auxiliary_loss_weight > 0:
            self.auxiliary_heads.extend(
                nn.Sequential(
                    nn.LayerNorm(branch_dim),
                    nn.Dropout(dropout),
                    nn.Linear(branch_dim, 1),
                )
                for _ in self.branch_names
            )

    @classmethod
    def from_pretrained(
        cls,
        aaindex_lookup: np.ndarray | torch.Tensor,
        pretrained_model_name: str,
        **kwargs: object,
    ) -> "ESM2CrossFusion":
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "ESM2CrossFusion requires transformers; install "
                "experiment/requirements-colab.txt"
            ) from error

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        backbone = AutoModel.from_pretrained(pretrained_model_name)
        special_ids = {
            "cls_token_id": tokenizer.cls_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if any(value is None for value in special_ids.values()):
            raise ValueError("The selected tokenizer is missing CLS, EOS, or PAD IDs")

        residue_ids = [
            int(tokenizer.convert_tokens_to_ids(residue))
            for residue in cls.alphabet[:-1]
        ]
        if tokenizer.unk_token_id is not None and any(
            token_id == tokenizer.unk_token_id for token_id in residue_ids
        ):
            raise ValueError("The selected tokenizer does not support the canonical alphabet")
        residue_ids.append(int(tokenizer.pad_token_id))
        return cls(
            aaindex_lookup=aaindex_lookup,
            backbone=backbone,
            residue_token_lookup=residue_ids,
            **special_ids,
            **kwargs,
        )

    @staticmethod
    def _masked_mean(features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = valid_mask.unsqueeze(-1).to(features.dtype)
        return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _esm_inputs(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove terminal gap tokens and return ESM IDs, masks, and site positions."""

        valid_mask = tokens.ne(self.padding_index)
        lengths = valid_mask.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("Every sequence window must contain at least one residue")
        if torch.any(~valid_mask[:, self.center_index]):
            raise ValueError("The central candidate site cannot be padding")

        batch_size = tokens.shape[0]
        esm_ids = torch.full(
            (batch_size, self.window_size + 2),
            self.pad_token_id,
            dtype=torch.long,
            device=tokens.device,
        )
        attention_mask = torch.zeros_like(esm_ids)
        esm_ids[:, 0] = self.cls_token_id
        attention_mask[:, 0] = 1

        compact_positions = valid_mask.cumsum(dim=1)
        batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1)
        batch_indices = batch_indices.expand_as(tokens)[valid_mask]
        residue_positions = compact_positions[valid_mask]
        mapped_tokens = self.residue_token_lookup[tokens[valid_mask]]
        esm_ids[batch_indices, residue_positions] = mapped_tokens
        attention_mask[batch_indices, residue_positions] = 1

        eos_positions = lengths + 1
        esm_ids[torch.arange(batch_size, device=tokens.device), eos_positions] = (
            self.eos_token_id
        )
        attention_mask[
            torch.arange(batch_size, device=tokens.device), eos_positions
        ] = 1
        center_positions = valid_mask[:, : self.center_index].sum(dim=1) + 1
        return esm_ids, attention_mask, center_positions

    def train(self, mode: bool = True) -> "ESM2CrossFusion":
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.esm_backbone.parameters()):
            self.esm_backbone.eval()
        return self

    def forward(
        self, tokens: torch.Tensor, return_gates: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2 or tokens.shape[1] != self.window_size:
            raise ValueError(
                f"Expected token shape (batch, {self.window_size}), got {tokens.shape}"
            )

        valid_mask = tokens.ne(self.padding_index)
        esm_ids, attention_mask, center_positions = self._esm_inputs(tokens)
        backbone_trainable = any(
            parameter.requires_grad for parameter in self.esm_backbone.parameters()
        )
        with torch.set_grad_enabled(self.training and backbone_trainable):
            esm_features = self.esm_backbone(
                input_ids=esm_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
        residue_features = esm_features[:, 1 : self.window_size + 1, :]
        compact_lengths = valid_mask.sum(dim=1, keepdim=True)
        compact_positions = torch.arange(
            1, self.window_size + 1, device=tokens.device
        ).unsqueeze(0)
        compact_valid_mask = compact_positions <= compact_lengths
        esm_vector = self.esm_projection(
            torch.cat(
                [
                    esm_features[batch_indices, center_positions],
                    self._masked_mean(residue_features, compact_valid_mask),
                ],
                dim=-1,
            )
        )

        one_hot = F.one_hot(tokens, num_classes=21)[..., :20].to(esm_vector.dtype)
        one_hot = one_hot.transpose(1, 2)
        local_features = torch.cat(
            [convolution(one_hot) for convolution in self.local_convolutions], dim=1
        )
        local_features = self.local_residual(local_features)
        mask_1d = valid_mask.unsqueeze(1)
        masked_local = local_features.masked_fill(~mask_1d, float("-inf"))
        local_max = masked_local.amax(dim=-1)
        local_mean = (
            local_features * mask_1d.to(local_features.dtype)
        ).sum(dim=-1) / valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        local_vector = self.local_projection(torch.cat([local_max, local_mean], dim=-1))

        aaindex = self.aaindex_embedding(tokens)
        if self.aaindex_gru is not None:
            aaindex, _ = self.aaindex_gru(aaindex)
        aaindex_vector = self.aaindex_projection(
            torch.cat(
                [
                    aaindex[:, self.center_index, :],
                    self._masked_mean(aaindex, valid_mask),
                ],
                dim=-1,
            )
        )

        vectors = [esm_vector, local_vector, aaindex_vector]
        if self.training and self.auxiliary_loss_weight > 0:
            self._last_auxiliary_logits = torch.stack(
                [
                    head(vector).squeeze(-1)
                    for head, vector in zip(self.auxiliary_heads, vectors)
                ],
                dim=1,
            )
        concatenated = torch.cat(vectors, dim=-1)
        gates = torch.softmax(self.gate(concatenated), dim=-1)
        stacked = torch.stack(vectors, dim=1)
        weighted = (stacked * gates.unsqueeze(-1)).sum(dim=1)
        fused = self.fusion_norm(weighted + self.residual_fusion(concatenated))
        logits = self.classifier(fused).squeeze(-1)
        if return_gates:
            return logits, gates
        return logits

    def auxiliary_loss(self, labels: torch.Tensor) -> torch.Tensor:
        """Return equally weighted branch supervision after a training forward."""

        if self.auxiliary_loss_weight == 0:
            return labels.new_zeros(())
        if self._last_auxiliary_logits is None:
            raise RuntimeError(
                "auxiliary_loss requires a preceding training forward pass"
            )
        auxiliary_logits = self._last_auxiliary_logits
        self._last_auxiliary_logits = None
        targets = labels.unsqueeze(1).expand_as(auxiliary_logits)
        return self.auxiliary_loss_weight * F.binary_cross_entropy_with_logits(
            auxiliary_logits, targets
        )


class LoRAESM2Hybrid(ESM2CrossFusion):
    """LoRA-adapted ESM-2 fused with explicit motif and AAindex experts."""

    branch_names = ("esm2_context", "local_motif", "aaindex")

    @classmethod
    def from_pretrained(
        cls,
        aaindex_lookup: np.ndarray | torch.Tensor,
        pretrained_model_name: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: Sequence[str],
        **kwargs: object,
    ) -> "LoRAESM2Hybrid":
        if lora_rank <= 0 or lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not lora_target_modules:
            raise ValueError("At least one LoRA target module is required")
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "LoRAESM2Hybrid requires transformers and peft; install "
                "experiment/requirements-colab.txt"
            ) from error

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        base_backbone = AutoModel.from_pretrained(
            pretrained_model_name, add_pooling_layer=False
        )
        special_ids = {
            "cls_token_id": tokenizer.cls_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if any(value is None for value in special_ids.values()):
            raise ValueError("The selected tokenizer is missing CLS, EOS, or PAD IDs")
        residue_ids = [
            int(tokenizer.convert_tokens_to_ids(residue))
            for residue in cls.alphabet[:-1]
        ]
        if tokenizer.unk_token_id is not None and any(
            token_id == tokenizer.unk_token_id for token_id in residue_ids
        ):
            raise ValueError("The selected tokenizer does not support the canonical alphabet")
        residue_ids.append(int(tokenizer.pad_token_id))

        lora_config = LoraConfig(
            task_type="FEATURE_EXTRACTION",
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=[str(value) for value in lora_target_modules],
            bias="none",
        )
        try:
            backbone = get_peft_model(base_backbone, lora_config)
        except ImportError as error:
            if "incompatible version of torchao" in str(error).lower():
                raise ImportError(
                    "Colab's optional torchao package is incompatible with PEFT. "
                    "Run `%pip uninstall -y torchao` before training; this "
                    "experiment does not use TorchAO."
                ) from error
            raise

        return cls(
            aaindex_lookup=aaindex_lookup,
            backbone=backbone,
            residue_token_lookup=residue_ids,
            preserve_backbone_trainability=True,
            freeze_backbone=False,
            **special_ids,
            **kwargs,
        )


class CenterLoRAESM2(nn.Module):
    """Task-adapted ESM-2 classifier focused on the candidate lysine.

    LoRA adapters are attached by :meth:`from_pretrained`; this module only
    supplies the site-aware input conversion and a compact classification head.
    It deliberately excludes the hand-crafted branches from ESM2-CrossFusion
    so the experiment isolates the effect of task-adapting the protein model.
    """

    branch_names = ("esm2_center",)
    diagnostic_name = "component_weights"
    alphabet = "ARNDCQEGHILKMFPSTWYV-"

    def __init__(
        self,
        backbone: nn.Module,
        backbone_hidden_size: int,
        residue_token_lookup: Sequence[int],
        cls_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        window_size: int = 49,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        if backbone_hidden_size <= 0 or classifier_hidden_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        residue_ids = torch.as_tensor(residue_token_lookup, dtype=torch.long)
        if residue_ids.ndim != 1 or residue_ids.shape[0] < 21:
            raise ValueError(
                "residue_token_lookup must contain the 20 amino acids and padding"
            )

        self.window_size = window_size
        self.center_index = window_size // 2
        self.padding_index = int(residue_ids.shape[0] - 1)
        self.cls_token_id = int(cls_token_id)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.esm_backbone = backbone
        self.register_buffer("residue_token_lookup", residue_ids, persistent=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(backbone_hidden_size),
            nn.Linear(backbone_hidden_size, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: Sequence[str],
        gradient_checkpointing: bool = False,
        tokenizer_do_lower_case: bool | None = None,
        tokenizer_use_fast: bool | None = None,
        tokenizer_vocab_filename: str | None = None,
        backbone_model_type: str | None = None,
        extra_residues: Sequence[str] = (),
        **kwargs: object,
    ) -> "CenterLoRAESM2":
        if lora_rank <= 0 or lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not lora_target_modules:
            raise ValueError("At least one LoRA target module is required")
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "CenterLoRAESM2 requires transformers and peft; install "
                "experiment/requirements-colab.txt"
            ) from error

        if tokenizer_vocab_filename is not None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as error:
                raise ImportError(
                    "Direct protein vocabulary loading requires huggingface_hub"
                ) from error
            vocabulary_path = hf_hub_download(
                repo_id=pretrained_model_name,
                filename=str(tokenizer_vocab_filename),
            )
            vocabulary_tokens = (
                Path(vocabulary_path).read_text(encoding="utf-8").splitlines()
            )
            if len(vocabulary_tokens) != len(set(vocabulary_tokens)):
                raise ValueError("The downloaded tokenizer vocabulary is not unique")
            vocabulary = {
                token: token_id
                for token_id, token in enumerate(vocabulary_tokens)
            }
            required_tokens = {
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                *cls.alphabet[:-1],
                *extra_residues,
            }
            missing_tokens = sorted(required_tokens - vocabulary.keys())
            if missing_tokens:
                raise ValueError(
                    f"The downloaded tokenizer vocabulary is missing: {missing_tokens}"
                )
            special_ids = {
                "cls_token_id": vocabulary["[CLS]"],
                "eos_token_id": vocabulary["[SEP]"],
                "pad_token_id": vocabulary["[PAD]"],
            }
            residue_ids = [
                vocabulary[residue]
                for residue in (*cls.alphabet[:-1], *extra_residues)
            ]
            residue_ids.append(vocabulary["[PAD]"])
        else:
            tokenizer_kwargs = {}
            if tokenizer_do_lower_case is not None:
                tokenizer_kwargs["do_lower_case"] = bool(tokenizer_do_lower_case)
            if tokenizer_use_fast is not None:
                tokenizer_kwargs["use_fast"] = bool(tokenizer_use_fast)
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name, **tokenizer_kwargs
            )
            end_token_id = tokenizer.eos_token_id
            if end_token_id is None:
                end_token_id = getattr(tokenizer, "sep_token_id", None)
            special_ids = {
                "cls_token_id": tokenizer.cls_token_id,
                "eos_token_id": end_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if any(value is None for value in special_ids.values()):
                raise ValueError(
                    "The selected tokenizer is missing CLS, end/SEP, or PAD IDs"
                )
            residue_ids = [
                int(tokenizer.convert_tokens_to_ids(residue))
                for residue in (*cls.alphabet[:-1], *extra_residues)
            ]
            if tokenizer.unk_token_id is not None and any(
                token_id == tokenizer.unk_token_id for token_id in residue_ids
            ):
                raise ValueError(
                    "The selected tokenizer does not support the canonical alphabet"
                )
            residue_ids.append(int(tokenizer.pad_token_id))

        if backbone_model_type is None:
            base_backbone = AutoModel.from_pretrained(
                pretrained_model_name, add_pooling_layer=False
            )
        elif backbone_model_type == "bert":
            from transformers import BertConfig, BertModel

            from huggingface_hub import hf_hub_download

            configuration_path = hf_hub_download(
                repo_id=pretrained_model_name,
                filename="config.json",
            )
            configuration_values = json.loads(
                Path(configuration_path).read_text(encoding="utf-8")
            )
            backbone_config = BertConfig.from_dict(configuration_values)
            base_backbone = BertModel.from_pretrained(
                pretrained_model_name,
                config=backbone_config,
                add_pooling_layer=False,
            )
        else:
            raise ValueError(
                f"Unsupported explicit backbone model type: {backbone_model_type}"
            )
        if gradient_checkpointing:
            gradient_checkpointing_enable = getattr(
                base_backbone, "gradient_checkpointing_enable", None
            )
            enable_input_require_grads = getattr(
                base_backbone, "enable_input_require_grads", None
            )
            if not callable(gradient_checkpointing_enable):
                raise ValueError(
                    "The selected backbone does not support gradient checkpointing"
                )
            gradient_checkpointing_enable()
            if callable(enable_input_require_grads):
                enable_input_require_grads()
        hidden_size = int(base_backbone.config.hidden_size)
        lora_config = LoraConfig(
            task_type="FEATURE_EXTRACTION",
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=[str(value) for value in lora_target_modules],
            bias="none",
        )
        try:
            backbone = get_peft_model(base_backbone, lora_config)
        except ImportError as error:
            if "incompatible version of torchao" in str(error).lower():
                raise ImportError(
                    "Colab's optional torchao package is incompatible with PEFT. "
                    "Run `%pip uninstall -y torchao` before training; this "
                    "experiment does not use TorchAO."
                ) from error
            raise

        return cls(
            backbone=backbone,
            backbone_hidden_size=hidden_size,
            residue_token_lookup=residue_ids,
            **special_ids,
            **kwargs,
        )

    def _esm_inputs(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_mask = tokens.ne(self.padding_index)
        lengths = valid_mask.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("Every sequence window must contain at least one residue")
        if torch.any(~valid_mask[:, self.center_index]):
            raise ValueError("The central candidate site cannot be padding")

        batch_size = tokens.shape[0]
        esm_ids = torch.full(
            (batch_size, self.window_size + 2),
            self.pad_token_id,
            dtype=torch.long,
            device=tokens.device,
        )
        attention_mask = torch.zeros_like(esm_ids)
        esm_ids[:, 0] = self.cls_token_id
        attention_mask[:, 0] = 1

        compact_positions = valid_mask.cumsum(dim=1)
        batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1)
        batch_indices = batch_indices.expand_as(tokens)[valid_mask]
        residue_positions = compact_positions[valid_mask]
        esm_ids[batch_indices, residue_positions] = self.residue_token_lookup[
            tokens[valid_mask]
        ]
        attention_mask[batch_indices, residue_positions] = 1

        row_indices = torch.arange(batch_size, device=tokens.device)
        eos_positions = lengths + 1
        esm_ids[row_indices, eos_positions] = self.eos_token_id
        attention_mask[row_indices, eos_positions] = 1
        center_positions = valid_mask[:, : self.center_index].sum(dim=1) + 1
        return esm_ids, attention_mask, center_positions

    def forward(
        self, tokens: torch.Tensor, return_gates: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2 or tokens.shape[1] != self.window_size:
            raise ValueError(
                f"Expected token shape (batch, {self.window_size}), got {tokens.shape}"
            )
        esm_ids, attention_mask, center_positions = self._esm_inputs(tokens)
        features = self.esm_backbone(
            input_ids=esm_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
        logits = self.classifier(
            features[batch_indices, center_positions]
        ).squeeze(-1)
        if return_gates:
            component_weights = torch.ones(
                (tokens.shape[0], 1), dtype=logits.dtype, device=logits.device
            )
            return logits, component_weights
        return logits


class CenterLoRAProtBERT(CenterLoRAESM2):
    """ProtBERT-BFD backbone with the unchanged v1 central-residue head."""

    branch_names = ("protbert_center",)


class MultiScaleCenterLoRAESM2(CenterLoRAESM2):
    """Fuse candidate-centred ESM-2 context at multiple motif scales.

    The LoRA-adapted backbone and input conversion are identical to
    :class:`CenterLoRAESM2`. The only experimental change is the classification
    head: it combines the central residue with masked mean pools over compact
    residue windows around that site. A shared projection keeps the additional
    task-specific parameter count small, while a learned softmax gate exposes
    the relative contribution of each scale.
    """

    diagnostic_name = "component_weights"

    def __init__(
        self,
        backbone: nn.Module,
        backbone_hidden_size: int,
        residue_token_lookup: Sequence[int],
        cls_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        window_size: int = 49,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.3,
        pooling_radii: Sequence[int] = (2, 5),
    ) -> None:
        radii = tuple(int(radius) for radius in pooling_radii)
        if not radii or any(radius <= 0 for radius in radii):
            raise ValueError("pooling_radii must contain positive integers")
        if tuple(sorted(set(radii))) != radii:
            raise ValueError("pooling_radii must be unique and increasing")
        if any(radius > window_size // 2 for radius in radii):
            raise ValueError("pooling radii cannot extend beyond the input window")

        super().__init__(
            backbone=backbone,
            backbone_hidden_size=backbone_hidden_size,
            residue_token_lookup=residue_token_lookup,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            window_size=window_size,
            classifier_hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )
        self.pooling_radii = radii
        self.branch_names = ("esm2_center",) + tuple(
            f"esm2_radius_{radius}" for radius in radii
        )
        self.component_projection = nn.Sequential(
            nn.LayerNorm(backbone_hidden_size),
            nn.Linear(backbone_hidden_size, classifier_hidden_dim),
            nn.GELU(),
        )
        self.component_gate = nn.Linear(classifier_hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    @staticmethod
    def _centered_mean_pool(
        features: torch.Tensor,
        center_positions: torch.Tensor,
        residue_lengths: torch.Tensor,
        radius: int,
    ) -> torch.Tensor:
        offsets = torch.arange(
            -radius,
            radius + 1,
            dtype=center_positions.dtype,
            device=center_positions.device,
        )
        positions = center_positions.unsqueeze(1) + offsets.unsqueeze(0)
        valid = positions.ge(1) & positions.le(residue_lengths.unsqueeze(1))
        safe_positions = positions.clamp(min=1, max=features.shape[1] - 2)
        batch_indices = torch.arange(
            features.shape[0], device=features.device
        ).unsqueeze(1)
        pooled_features = features[batch_indices, safe_positions]
        valid_weights = valid.unsqueeze(-1).to(features.dtype)
        return (pooled_features * valid_weights).sum(dim=1) / valid_weights.sum(
            dim=1
        ).clamp_min(1)

    def forward(
        self, tokens: torch.Tensor, return_gates: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2 or tokens.shape[1] != self.window_size:
            raise ValueError(
                f"Expected token shape (batch, {self.window_size}), got {tokens.shape}"
            )
        esm_ids, attention_mask, center_positions = self._esm_inputs(tokens)
        features = self.esm_backbone(
            input_ids=esm_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
        residue_lengths = attention_mask.sum(dim=1) - 2
        components = [features[batch_indices, center_positions]]
        components.extend(
            self._centered_mean_pool(
                features=features,
                center_positions=center_positions,
                residue_lengths=residue_lengths,
                radius=radius,
            )
            for radius in self.pooling_radii
        )
        projected = self.component_projection(torch.stack(components, dim=1))
        component_weights = torch.softmax(
            self.component_gate(projected).squeeze(-1), dim=1
        )
        fused = (projected * component_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(fused).squeeze(-1)
        if return_gates:
            return logits, component_weights
        return logits


class LongContextLoRAESM2(CenterLoRAESM2):
    """Target-aware ESM-2 expert over local and distal protein context.

    A single LoRA-adapted backbone supplies the central lysine, a radius-24
    local pool, and a masked global context pool. Their projected
    representations are concatenated rather than competitively gated, so the
    Long-Context Expert cannot silently discard one scale during early training.
    """

    branch_names = ("esm2_center", "esm2_radius_24", "esm2_global")
    diagnostic_name = "component_norm_fractions"

    def __init__(
        self,
        backbone: nn.Module,
        backbone_hidden_size: int,
        residue_token_lookup: Sequence[int],
        cls_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        window_size: int = 257,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.3,
        local_radius: int = 24,
        head_learning_rate_multiplier: float = 3.0,
    ) -> None:
        if local_radius <= 0 or local_radius > window_size // 2:
            raise ValueError("local_radius must fit inside the context window")
        if head_learning_rate_multiplier <= 0:
            raise ValueError("head_learning_rate_multiplier must be positive")
        super().__init__(
            backbone=backbone,
            backbone_hidden_size=backbone_hidden_size,
            residue_token_lookup=residue_token_lookup,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            window_size=window_size,
            classifier_hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )
        self.local_radius = int(local_radius)
        self.head_learning_rate_multiplier = float(head_learning_rate_multiplier)
        self.component_projection = nn.Sequential(
            nn.LayerNorm(backbone_hidden_size),
            nn.Linear(backbone_hidden_size, classifier_hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_hidden_dim * 3),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim * 3, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def optimizer_parameter_groups(
        self, base_learning_rate: float
    ) -> list[dict[str, object]]:
        backbone_parameters = [
            parameter
            for parameter in self.esm_backbone.parameters()
            if parameter.requires_grad
        ]
        head_parameters = [
            parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("esm_backbone.")
        ]
        return [
            {"params": backbone_parameters, "lr": base_learning_rate},
            {
                "params": head_parameters,
                "lr": base_learning_rate * self.head_learning_rate_multiplier,
            },
        ]

    @staticmethod
    def _masked_global_mean(
        features: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        residue_mask = attention_mask.clone()
        residue_mask[:, 0] = 0
        residue_lengths = residue_mask.sum(dim=1) - 1
        row_indices = torch.arange(features.shape[0], device=features.device)
        eos_positions = attention_mask.sum(dim=1) - 1
        residue_mask[row_indices, eos_positions] = 0
        weights = residue_mask.unsqueeze(-1).to(features.dtype)
        return (features * weights).sum(dim=1) / residue_lengths.clamp_min(1).unsqueeze(
            1
        )

    def forward(
        self, tokens: torch.Tensor, return_gates: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2 or tokens.shape[1] != self.window_size:
            raise ValueError(
                f"Expected token shape (batch, {self.window_size}), got {tokens.shape}"
            )
        esm_ids, attention_mask, center_positions = self._esm_inputs(tokens)
        features = self.esm_backbone(
            input_ids=esm_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
        residue_lengths = attention_mask.sum(dim=1) - 2
        components = torch.stack(
            [
                features[batch_indices, center_positions],
                MultiScaleCenterLoRAESM2._centered_mean_pool(
                    features,
                    center_positions,
                    residue_lengths,
                    self.local_radius,
                ),
                self._masked_global_mean(features, attention_mask),
            ],
            dim=1,
        )
        projected = self.component_projection(components)
        logits = self.classifier(projected.flatten(start_dim=1)).squeeze(-1)
        component_norms = projected.norm(dim=-1).clamp_min(1e-8)
        diagnostics = component_norms / component_norms.sum(
            dim=1, keepdim=True
        )
        if return_gates:
            return logits, diagnostics
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
    if architecture == "esm2_crossfusion_v1":
        return ESM2CrossFusion.from_pretrained(
            aaindex_lookup=aaindex_lookup,
            pretrained_model_name=str(model_config["pretrained_model_name"]),
            window_size=window_size,
            branch_dim=int(model_config["branch_dim"]),
            conv_channels=int(model_config["conv_channels"]),
            conv_kernels=tuple(int(value) for value in model_config["conv_kernels"]),
            conv_dilations=tuple(
                int(value) for value in model_config["conv_dilations"]
            ),
            dropout=float(model_config["dropout"]),
            freeze_backbone=bool(model_config.get("freeze_backbone", True)),
            unfreeze_last_n_layers=int(
                model_config.get("unfreeze_last_n_layers", 0)
            ),
        )
    if architecture == "lora_esm2_hybrid_v1":
        return LoRAESM2Hybrid.from_pretrained(
            aaindex_lookup=aaindex_lookup,
            pretrained_model_name=str(model_config["pretrained_model_name"]),
            window_size=window_size,
            branch_dim=int(model_config["branch_dim"]),
            conv_channels=int(model_config["conv_channels"]),
            conv_kernels=tuple(int(value) for value in model_config["conv_kernels"]),
            conv_dilations=tuple(
                int(value) for value in model_config["conv_dilations"]
            ),
            aaindex_gru_hidden_dim=int(
                model_config.get("aaindex_gru_hidden_dim", 0)
            ),
            auxiliary_loss_weight=float(
                model_config.get("auxiliary_loss_weight", 0.0)
            ),
            dropout=float(model_config["dropout"]),
            lora_rank=int(model_config["lora_rank"]),
            lora_alpha=int(model_config["lora_alpha"]),
            lora_dropout=float(model_config["lora_dropout"]),
            lora_target_modules=tuple(
                str(value) for value in model_config["lora_target_modules"]
            ),
        )
    if architecture == "center_lora_esm2_v1":
        return CenterLoRAESM2.from_pretrained(
            pretrained_model_name=str(model_config["pretrained_model_name"]),
            window_size=window_size,
            classifier_hidden_dim=int(model_config["classifier_hidden_dim"]),
            dropout=float(model_config["dropout"]),
            lora_rank=int(model_config["lora_rank"]),
            lora_alpha=int(model_config["lora_alpha"]),
            lora_dropout=float(model_config["lora_dropout"]),
            lora_target_modules=tuple(
                str(value) for value in model_config["lora_target_modules"]
            ),
        )
    if architecture == "center_lora_protbert_v1":
        return CenterLoRAProtBERT.from_pretrained(
            pretrained_model_name=str(model_config["pretrained_model_name"]),
            window_size=window_size,
            classifier_hidden_dim=int(model_config["classifier_hidden_dim"]),
            dropout=float(model_config["dropout"]),
            lora_rank=int(model_config["lora_rank"]),
            lora_alpha=int(model_config["lora_alpha"]),
            lora_dropout=float(model_config["lora_dropout"]),
            lora_target_modules=tuple(
                str(value) for value in model_config["lora_target_modules"]
            ),
            gradient_checkpointing=bool(
                model_config.get("gradient_checkpointing", False)
            ),
            tokenizer_do_lower_case=bool(
                model_config.get("tokenizer_do_lower_case", False)
            ),
            tokenizer_use_fast=bool(
                model_config.get("tokenizer_use_fast", False)
            ),
            tokenizer_vocab_filename=(
                str(model_config["tokenizer_vocab_filename"])
                if model_config.get("tokenizer_vocab_filename") is not None
                else None
            ),
            backbone_model_type=(
                str(model_config["backbone_model_type"])
                if model_config.get("backbone_model_type") is not None
                else None
            ),
        )
    if architecture == "center_lora_esm2_multiscale_v2":
        return MultiScaleCenterLoRAESM2.from_pretrained(
            pretrained_model_name=str(model_config["pretrained_model_name"]),
            window_size=window_size,
            classifier_hidden_dim=int(model_config["classifier_hidden_dim"]),
            dropout=float(model_config["dropout"]),
            pooling_radii=tuple(
                int(value) for value in model_config["pooling_radii"]
            ),
            lora_rank=int(model_config["lora_rank"]),
            lora_alpha=int(model_config["lora_alpha"]),
            lora_dropout=float(model_config["lora_dropout"]),
            lora_target_modules=tuple(
                str(value) for value in model_config["lora_target_modules"]
            ),
        )
    if architecture == "long_context_lora_esm2_v1":
        return LongContextLoRAESM2.from_pretrained(
            pretrained_model_name=str(model_config["pretrained_model_name"]),
            window_size=window_size,
            classifier_hidden_dim=int(model_config["classifier_hidden_dim"]),
            dropout=float(model_config["dropout"]),
            local_radius=int(model_config.get("local_radius", 24)),
            head_learning_rate_multiplier=float(
                model_config.get("head_learning_rate_multiplier", 3.0)
            ),
            extra_residues=("X", "B", "U", "Z", "O"),
            lora_rank=int(model_config["lora_rank"]),
            lora_alpha=int(model_config["lora_alpha"]),
            lora_dropout=float(model_config["lora_dropout"]),
            lora_target_modules=tuple(
                str(value) for value in model_config["lora_target_modules"]
            ),
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
