"""Full-protein context retrieval artifacts and target-centred encodings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .fasta import ALPHABET, SiteRecord


CONTEXT_RESIDUES = ALPHABET[:-1] + "XBUZO"
CONTEXT_TOKEN_TO_INDEX = {
    residue: index for index, residue in enumerate(CONTEXT_RESIDUES)
}
CONTEXT_PADDING_INDEX = len(CONTEXT_RESIDUES)


@dataclass(frozen=True)
class ContextValidation:
    """Validation result for one released site against a UniProt sequence."""

    valid: bool
    reason: str
    context: str | None


def centered_sequence_window(
    sequence: str,
    position: int,
    window_size: int,
    padding: str = "-",
) -> str:
    """Return an odd, fixed-size window around a one-based sequence position."""

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if not 1 <= position <= len(sequence):
        raise ValueError(
            f"position {position} is outside a sequence of length {len(sequence)}"
        )
    radius = window_size // 2
    center = position - 1
    start = center - radius
    stop = center + radius + 1
    left_padding = padding * max(0, -start)
    right_padding = padding * max(0, stop - len(sequence))
    window = left_padding + sequence[max(0, start) : min(len(sequence), stop)]
    window += right_padding
    if len(window) != window_size or window[radius] != sequence[center]:
        raise AssertionError("Internal target-centred window error")
    return window


def validate_and_build_context(
    record: SiteRecord,
    protein_sequence: str | None,
    context_window_size: int,
) -> ContextValidation:
    """Require the current UniProt sequence to reproduce the released 49-mer."""

    if protein_sequence is None:
        return ContextValidation(False, "accession_not_retrieved", None)
    if record.position is None:
        return ContextValidation(False, "missing_site_position", None)
    sequence = protein_sequence.strip().upper()
    # The released MMUbiPred/PLMD headers store Python-style zero-based
    # coordinates (for example, P16144|1020 is UniProt residue 1021). Convert
    # explicitly before using the one-based UniProt sequence helper.
    uniprot_position = record.position + 1
    try:
        released_window = centered_sequence_window(
            sequence, uniprot_position, len(record.sequence)
        )
        context = centered_sequence_window(
            sequence, uniprot_position, context_window_size
        )
    except ValueError:
        return ContextValidation(False, "position_out_of_range", None)
    if sequence[uniprot_position - 1] != "K":
        return ContextValidation(False, "uniprot_center_is_not_lysine", None)
    if released_window != record.sequence:
        return ContextValidation(False, "released_window_mismatch", None)
    unsupported = sorted(set(context) - set(CONTEXT_RESIDUES) - {"-"})
    if unsupported:
        return ContextValidation(
            False,
            "unsupported_context_residues:" + "".join(unsupported),
            None,
        )
    return ContextValidation(True, "validated", context)


def load_sequence_cache(path: str | Path) -> tuple[dict[str, str], dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) not in (1, 2):
        raise ValueError("Unsupported UniProt sequence-cache schema")
    sequences = payload.get("sequences")
    if not isinstance(sequences, dict):
        raise TypeError("Sequence cache must contain a sequences dictionary")
    normalized = {
        str(accession): str(sequence).upper()
        for accession, sequence in sequences.items()
    }
    metadata = {
        key: value for key, value in payload.items() if key != "sequences"
    }
    return normalized, metadata


def sequence_cache_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_validated_contexts(
    records: Sequence[SiteRecord],
    sequences: Mapping[str, str],
    context_window_size: int,
) -> tuple[list[int], list[str], dict[str, object]]:
    valid_indices: list[int] = []
    contexts: list[str] = []
    reason_counts: dict[str, int] = {}
    issue_examples: list[dict[str, object]] = []
    for index, record in enumerate(records):
        result = validate_and_build_context(
            record,
            sequences.get(record.protein_id),
            context_window_size=context_window_size,
        )
        reason_counts[result.reason] = reason_counts.get(result.reason, 0) + 1
        if result.valid:
            valid_indices.append(index)
            if result.context is None:
                raise AssertionError("A valid context cannot be empty")
            contexts.append(result.context)
        elif len(issue_examples) < 50:
            issue_examples.append(
                {
                    "dataset_index": index,
                    "header": record.header,
                    "reason": result.reason,
                }
            )
    report = {
        "total_records": len(records),
        "validated_records": len(valid_indices),
        "excluded_records": len(records) - len(valid_indices),
        "validated_fraction": len(valid_indices) / max(len(records), 1),
        "reason_counts": reason_counts,
        "issue_examples": issue_examples,
        "context_window_size": context_window_size,
        "released_header_coordinate_system": "zero_based",
        "uniprot_coordinate_system": "one_based",
    }
    return valid_indices, contexts, report


class LongContextSiteDataset(Dataset):
    """Fixed 257-style contexts with explicit protein-language-model unknowns."""

    def __init__(
        self,
        records: Sequence[SiteRecord],
        contexts: Sequence[str],
        original_indices: Sequence[int] | None = None,
    ) -> None:
        if not records or len(records) != len(contexts):
            raise ValueError("records and contexts must be non-empty and aligned")
        window_size = len(contexts[0])
        if window_size % 2 == 0 or any(len(value) != window_size for value in contexts):
            raise ValueError("All context windows must have one identical odd length")
        if any(value[window_size // 2] != "K" for value in contexts):
            raise ValueError("Every long context must be lysine-centred")
        self.records = list(records)
        self.window_size = window_size
        padding_index = CONTEXT_PADDING_INDEX
        self.tokens = torch.tensor(
            [
                [
                    padding_index
                    if residue == "-"
                    else CONTEXT_TOKEN_TO_INDEX[residue]
                    for residue in context
                ]
                for context in contexts
            ],
            dtype=torch.long,
        )
        self.labels = torch.tensor(
            [record.label for record in records], dtype=torch.float32
        )
        indices = (
            list(range(len(records)))
            if original_indices is None
            else [int(index) for index in original_indices]
        )
        if len(indices) != len(records):
            raise ValueError("original_indices must align with records")
        self.original_indices = torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": self.tokens[index],
            "label": self.labels[index],
            "index": self.original_indices[index],
        }
