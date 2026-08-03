"""AlphaFold DB metadata selection and site-confidence validation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.strip().upper().encode("ascii")).hexdigest()


def select_canonical_prediction(
    accession: str,
    sequence: str,
    predictions: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object] | None, str]:
    """Select only an exact, full-length canonical AFDB prediction."""

    normalized_sequence = sequence.strip().upper()
    canonical = [
        dict(prediction)
        for prediction in predictions
        if str(prediction.get("uniprotAccession", "")) == accession
    ]
    if not canonical:
        return None, "canonical_accession_not_returned"
    exact = []
    for prediction in canonical:
        predicted_sequence = str(
            prediction.get("uniprotSequence")
            or prediction.get("sequence")
            or ""
        ).upper()
        start = int(prediction.get("uniprotStart", prediction.get("sequenceStart", 0)))
        end = int(prediction.get("uniprotEnd", prediction.get("sequenceEnd", 0)))
        if (
            predicted_sequence == normalized_sequence
            and start == 1
            and end == len(normalized_sequence)
        ):
            exact.append(prediction)
    if not exact:
        return None, "exact_full_sequence_not_returned"
    preferred_id = f"AF-{accession}-F1"
    exact.sort(
        key=lambda prediction: (
            str(prediction.get("modelEntityId", "")) != preferred_id,
            bool(prediction.get("isComplex", False)),
            -int(prediction.get("latestVersion", 0)),
        )
    )
    return exact[0], "available"


def select_exact_prediction_alias(
    accession: str,
    sequence: str,
    predictions: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object] | None, str]:
    """Select an exact full-length prediction returned under an accession alias.

    The queried accession remains the dataset identifier. An alias is accepted only
    when the complete predicted sequence and full-length coordinates are identical.
    """

    selected, reason = select_canonical_prediction(
        accession, sequence, predictions
    )
    if selected is not None:
        return selected, reason

    normalized_sequence = sequence.strip().upper()
    exact_aliases: list[dict[str, object]] = []
    for prediction in predictions:
        prediction_accession = str(
            prediction.get("uniprotAccession", "")
        )
        if not prediction_accession or prediction_accession == accession:
            continue
        predicted_sequence = str(
            prediction.get("uniprotSequence")
            or prediction.get("sequence")
            or ""
        ).upper()
        start = int(
            prediction.get("uniprotStart", prediction.get("sequenceStart", 0))
        )
        end = int(
            prediction.get("uniprotEnd", prediction.get("sequenceEnd", 0))
        )
        if (
            predicted_sequence == normalized_sequence
            and start == 1
            and end == len(normalized_sequence)
        ):
            exact_aliases.append(dict(prediction))
    if not exact_aliases:
        return None, reason
    exact_aliases.sort(
        key=lambda prediction: (
            bool(prediction.get("isComplex", False)),
            -int(prediction.get("latestVersion", 0)),
            str(prediction.get("uniprotAccession", "")),
            str(prediction.get("modelEntityId", "")),
        )
    )
    return exact_aliases[0], "available_via_afdb_alias"


def select_exact_uniprot_primary(
    accession: str,
    sequence: str,
    results: Sequence[Mapping[str, object]],
) -> tuple[str | None, str]:
    """Resolve an accession only through a unique exact-sequence UniProt result."""

    normalized_sequence = sequence.strip().upper()
    exact_accessions: set[str] = set()
    for result in results:
        primary = str(result.get("primaryAccession", ""))
        sequence_payload = result.get("sequence")
        if isinstance(sequence_payload, Mapping):
            returned_sequence = str(sequence_payload.get("value", ""))
        else:
            returned_sequence = str(sequence_payload or "")
        if primary and returned_sequence.strip().upper() == normalized_sequence:
            exact_accessions.add(primary)
    if accession in exact_accessions:
        return accession, "same_primary_exact_sequence"
    if len(exact_accessions) == 1:
        return next(iter(exact_accessions)), "mapped_primary_exact_sequence"
    if not exact_accessions:
        return None, "uniprot_exact_sequence_not_returned"
    return None, "uniprot_exact_sequence_mapping_ambiguous"


def compact_prediction_metadata(
    accession: str,
    sequence: str,
    prediction: Mapping[str, object],
) -> dict[str, object]:
    fields = (
        "modelEntityId",
        "entryId",
        "providerId",
        "toolUsed",
        "modelCreatedDate",
        "sequenceVersionDate",
        "globalMetricValue",
        "latestVersion",
        "uniprotStart",
        "uniprotEnd",
        "pdbUrl",
        "cifUrl",
        "bcifUrl",
        "plddtDocUrl",
        "paeDocUrl",
    )
    return {
        "status": "available",
        "accession": accession,
        "sequence_length": len(sequence),
        "sequence_sha256": sequence_sha256(sequence),
        "model_uniprot_accession": prediction.get("uniprotAccession"),
        **{field: prediction.get(field) for field in fields},
    }


def parse_confidence_document(payload: object) -> dict[int, float]:
    """Parse the AFDB per-residue confidence JSON into one-based scores."""

    if isinstance(payload, dict):
        positions = payload.get("residueNumber", payload.get("residue_number"))
        scores = payload.get("confidenceScore", payload.get("confidence_score"))
        if isinstance(positions, list) and isinstance(scores, list):
            if len(positions) != len(scores):
                raise ValueError("AlphaFold confidence arrays do not align")
            records = [
                {"residueNumber": position, "confidenceScore": score}
                for position, score in zip(positions, scores)
            ]
        else:
            records = payload.get("confidence") or payload.get("residues")
    else:
        records = payload
    if not isinstance(records, list):
        raise TypeError("AlphaFold confidence document must contain a list")
    confidence: dict[int, float] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("AlphaFold confidence records must be dictionaries")
        position = record.get("residueNumber", record.get("residue_number"))
        score = record.get("confidenceScore", record.get("confidence_score"))
        if position is None or score is None:
            raise ValueError("AlphaFold confidence record is missing position or score")
        one_based_position = int(position)
        value = float(score)
        if (
            one_based_position <= 0
            or not 0.0 <= value <= 100.0
            or one_based_position in confidence
        ):
            raise ValueError("Invalid AlphaFold residue confidence value")
        confidence[one_based_position] = value
    return confidence


def confidence_category(score: float) -> str:
    if score >= 90.0:
        return "very_high"
    if score >= 70.0:
        return "confident"
    if score >= 50.0:
        return "low"
    return "very_low"
