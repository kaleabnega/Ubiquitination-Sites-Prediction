"""Tensor encodings and leakage-aware development splits."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from .fasta import ALPHABET, SiteRecord


TOKEN_TO_INDEX = {residue: index for index, residue in enumerate(ALPHABET)}


def load_normalized_aaindex(path: str | Path) -> np.ndarray:
    """Return the paper's normalized AAindex lookup with a zero padding row."""

    raw = np.loadtxt(path, dtype=np.float32)
    if raw.shape != (31, 20):
        raise ValueError(f"Expected AAindex shape (31, 20), received {raw.shape}")

    row_min = raw.min(axis=1, keepdims=True)
    row_range = raw.max(axis=1, keepdims=True) - row_min
    if np.any(row_range == 0):
        raise ValueError("AAindex contains a constant property row")

    normalized = (raw - row_min) / row_range
    lookup = normalized.T
    lookup = np.concatenate([lookup, np.zeros((1, 31), dtype=np.float32)], axis=0)
    return lookup.astype(np.float32, copy=False)


class SiteDataset(Dataset):
    def __init__(self, records: Sequence[SiteRecord]) -> None:
        if not records:
            raise ValueError("SiteDataset requires at least one record")
        self.records = list(records)
        self.tokens = torch.tensor(
            [[TOKEN_TO_INDEX[residue] for residue in record.sequence] for record in records],
            dtype=torch.long,
        )
        self.labels = torch.tensor(
            [record.label for record in records], dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": self.tokens[index],
            "label": self.labels[index],
            "index": torch.tensor(index, dtype=torch.long),
        }


def make_train_validation_indices(
    records: Sequence[SiteRecord],
    validation_fraction: float,
    seed: int,
    strategy: str = "protein_grouped",
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")

    labels = np.asarray([record.label for record in records], dtype=np.int64)
    indices = np.arange(len(records))

    if strategy == "stratified":
        train_indices, validation_indices = train_test_split(
            indices,
            test_size=validation_fraction,
            stratify=labels,
            random_state=seed,
        )
    elif strategy == "protein_grouped":
        groups = np.asarray([record.protein_id for record in records])
        n_splits = max(2, round(1.0 / validation_fraction))
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        train_indices, validation_indices = next(
            splitter.split(indices, labels, groups=groups)
        )
        train_groups = set(groups[train_indices])
        validation_groups = set(groups[validation_indices])
        overlap = train_groups & validation_groups
        if overlap:
            raise AssertionError(f"Protein leakage in grouped split: {len(overlap)} groups")
    else:
        raise ValueError("strategy must be 'protein_grouped' or 'stratified'")

    return np.sort(train_indices), np.sort(validation_indices)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(
    dataset: Dataset,
    indices: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    selected: Dataset = Subset(dataset, indices.tolist()) if indices is not None else dataset
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )
