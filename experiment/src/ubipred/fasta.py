"""FASTA parsing and preprocessing matching the released MMUbiPred notebook."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Sequence


ALPHABET = "ARNDCQEGHILKMFPSTWYV-"
ALPHABET_SET = frozenset(ALPHABET)
ORIGINAL_WINDOW_SIZE = 81
DEFAULT_WINDOW_SIZE = 49


@dataclass(frozen=True)
class SiteRecord:
    """One lysine-centred labelled sequence window."""

    header: str
    protein_id: str
    position: int | None
    sequence: str
    label: int
    source: str


@dataclass(frozen=True)
class PreprocessingReport:
    raw_count: int
    kept_count: int
    dropped_unsupported: int
    invalid_length: int
    invalid_center: int


def _parse_header(header: str) -> tuple[str, int | None]:
    parts = header.split("|")
    protein_id = parts[0].strip()
    position: int | None = None
    if len(parts) > 1:
        try:
            position = int(parts[-1])
        except ValueError:
            position = None
    return protein_id, position


def iter_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield `(header, sequence)` pairs without requiring Biopython."""

    fasta_path = Path(path)
    header: str | None = None
    chunks: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header = line[1:].strip()
                chunks = []
            else:
                if header is None:
                    raise ValueError(
                        f"Sequence before FASTA header in {fasta_path} "
                        f"at line {line_number}"
                    )
                chunks.append(line)

    if header is not None:
        yield header, "".join(chunks).upper()


def load_labelled_fasta(path: str | Path, label: int) -> list[SiteRecord]:
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, received {label}")

    fasta_path = Path(path)
    records: list[SiteRecord] = []
    for header, sequence in iter_fasta(fasta_path):
        protein_id, position = _parse_header(header)
        records.append(
            SiteRecord(
                header=header,
                protein_id=protein_id,
                position=position,
                sequence=sequence,
                label=label,
                source=fasta_path.name,
            )
        )
    return records


def center_crop(sequence: str, window_size: int = DEFAULT_WINDOW_SIZE) -> str:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if len(sequence) < window_size:
        raise ValueError(
            f"Cannot crop window of {window_size} from sequence of {len(sequence)}"
        )
    start = (len(sequence) - window_size) // 2
    cropped = sequence[start : start + window_size]
    if len(cropped) != window_size:
        raise AssertionError("Internal crop-length error")
    return cropped


def prepare_records(
    records: Sequence[SiteRecord],
    window_size: int = DEFAULT_WINDOW_SIZE,
    expected_input_size: int = ORIGINAL_WINDOW_SIZE,
) -> tuple[list[SiteRecord], PreprocessingReport]:
    """Apply the authors' alphabet filter and centered 81-to-49 crop.

    The released notebook silently drops sequences containing residues outside
    `ARNDCQEGHILKMFPSTWYV-`. We preserve that behavior and report every drop.
    """

    kept: list[SiteRecord] = []
    dropped_unsupported = 0
    invalid_length = 0
    invalid_center = 0

    for record in records:
        if len(record.sequence) != expected_input_size:
            invalid_length += 1
            continue
        if any(residue not in ALPHABET_SET for residue in record.sequence):
            dropped_unsupported += 1
            continue
        if record.sequence[expected_input_size // 2] != "K":
            invalid_center += 1
            continue

        cropped = center_crop(record.sequence, window_size=window_size)
        if cropped[window_size // 2] != "K":
            invalid_center += 1
            continue
        kept.append(replace(record, sequence=cropped))

    report = PreprocessingReport(
        raw_count=len(records),
        kept_count=len(kept),
        dropped_unsupported=dropped_unsupported,
        invalid_length=invalid_length,
        invalid_center=invalid_center,
    )
    return kept, report


def load_released_split(
    data_dir: str | Path,
    split: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> tuple[list[SiteRecord], dict[str, PreprocessingReport]]:
    """Load the exact released train or independent-test FASTA pair."""

    root = Path(data_dir)
    filenames = {
        "train": (
            "Positive_90_percent_training_set_DeepUBI.fasta",
            "Negative_90_percent_training_set_DeepUBI.fasta",
        ),
        "test": (
            "Positive_10_percent_independent_test_set_DeepUBI.fasta",
            "Negative_10_percent_independent_test_set_DeepUBI.fasta",
        ),
    }
    if split not in filenames:
        raise ValueError(f"split must be one of {sorted(filenames)}, received {split}")

    positive_name, negative_name = filenames[split]
    positive_raw = load_labelled_fasta(root / positive_name, label=1)
    negative_raw = load_labelled_fasta(root / negative_name, label=0)
    positive, positive_report = prepare_records(positive_raw, window_size=window_size)
    negative, negative_report = prepare_records(negative_raw, window_size=window_size)

    return positive + negative, {
        "positive": positive_report,
        "negative": negative_report,
    }


def records_to_dicts(records: Iterable[SiteRecord]) -> list[dict[str, object]]:
    return [
        {
            "header": record.header,
            "protein_id": record.protein_id,
            "position": record.position,
            "sequence": record.sequence,
            "label": record.label,
            "source": record.source,
        }
        for record in records
    ]
