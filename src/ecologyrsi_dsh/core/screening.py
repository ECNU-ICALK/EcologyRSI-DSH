"""Canonical contracts for replayable two-stage candidate screening."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import digest


SCREENING_SCHEMA_V1 = "ecologyrsi-dsh.candidate-screening/1"
SCREENING_SCHEMA_V2 = "ecologyrsi-dsh.candidate-screening/2"
FORMAL_SELECTION_SCHEMA_V1 = "ecologyrsi-dsh.formal-selection-cohort/1"
FORMAL_SELECTION_SCHEMA_V2 = "ecologyrsi-dsh.formal-selection-cohort/2"
SCREENED_OUT_SCHEMA_V1 = "ecologyrsi-dsh.candidate-screened-out/1"


def screening_record_digest(payload: Mapping[str, Any]) -> str:
    normalized = {
        name: payload[name]
        for name in (
            "generation",
            "candidate_id",
            "score",
            "passed",
            "constraint_violations",
            "origin_count",
            "prediction_cell_count",
            "cohort_digest",
        )
    }
    return digest(normalized)


def screening_cohort_digest(records: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(records, key=lambda item: str(item["candidate_id"]))
    return digest(ordered)


__all__ = [
    "FORMAL_SELECTION_SCHEMA_V1",
    "FORMAL_SELECTION_SCHEMA_V2",
    "SCREENED_OUT_SCHEMA_V1",
    "SCREENING_SCHEMA_V1",
    "SCREENING_SCHEMA_V2",
    "screening_cohort_digest",
    "screening_record_digest",
]
