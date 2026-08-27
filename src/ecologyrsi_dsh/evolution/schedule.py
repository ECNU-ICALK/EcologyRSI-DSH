"""Origin-first execution schedule for Top-2 adaptive epochs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SCHEDULE_SCHEMA_VERSION = "ecologyrsi-dsh.top2-adaptive-epoch-schedule/1"
OPTIMIZATION_PROTOCOL = "top2_adaptive_epoch@1"

_FIELDS = frozenset(
    {
        "schema_version",
        "screening_origin_count",
        "finalist_count",
        "formal_origin_count_per_finalist",
        "local_batch_origin_count",
        "max_local_edits_per_batch",
        "selection_holdout_origin_count",
        "local_evaluation_mode",
    }
)


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be greater than zero")
    return value


@dataclass(frozen=True, slots=True)
class OptimizationSchedule:
    """Frozen scientific and execution schedule expressed in forecast origins."""

    schema_version: str
    screening_origin_count: int
    finalist_count: int
    formal_origin_count_per_finalist: int
    local_batch_origin_count: int
    max_local_edits_per_batch: int
    selection_holdout_origin_count: int
    local_evaluation_mode: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEDULE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SCHEDULE_SCHEMA_VERSION!r}"
            )
        if self.screening_origin_count != 64:
            raise ValueError("screening_origin_count must be 64")
        if self.finalist_count != 2:
            raise ValueError("finalist_count must be 2")
        if self.local_evaluation_mode != "prequential":
            raise ValueError("local_evaluation_mode must be 'prequential'")
        formal = _positive_integer(
            self.formal_origin_count_per_finalist,
            "formal_origin_count_per_finalist",
        )
        batch = _positive_integer(
            self.local_batch_origin_count,
            "local_batch_origin_count",
        )
        edits = self.max_local_edits_per_batch
        if isinstance(edits, bool) or not isinstance(edits, int):
            raise TypeError("max_local_edits_per_batch must be an integer")
        holdout = _positive_integer(
            self.selection_holdout_origin_count,
            "selection_holdout_origin_count",
        )
        if formal % batch:
            raise ValueError(
                "local_batch_origin_count must divide "
                "formal_origin_count_per_finalist"
            )
        if not 0 <= edits <= 5:
            raise ValueError("max_local_edits_per_batch must be between 0 and 5")
        if holdout < 169:
            raise ValueError("selection_holdout_origin_count must be at least 169")

    @classmethod
    def default(cls) -> "OptimizationSchedule":
        return cls(
            schema_version=SCHEDULE_SCHEMA_VERSION,
            screening_origin_count=64,
            finalist_count=2,
            formal_origin_count_per_finalist=500,
            local_batch_origin_count=50,
            max_local_edits_per_batch=2,
            selection_holdout_origin_count=169,
            local_evaluation_mode="prequential",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OptimizationSchedule":
        if not isinstance(value, Mapping):
            raise TypeError("optimization_schedule must be an object")
        supplied = frozenset(value)
        missing = sorted(_FIELDS - supplied)
        unexpected = sorted(supplied - _FIELDS)
        if missing:
            raise ValueError(
                "optimization_schedule missing fields: " + ", ".join(missing)
            )
        if unexpected:
            raise ValueError(
                "optimization_schedule unexpected fields: "
                + ", ".join(unexpected)
            )
        return cls(**{field: value[field] for field in _FIELDS})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "screening_origin_count": self.screening_origin_count,
            "finalist_count": self.finalist_count,
            "formal_origin_count_per_finalist": (
                self.formal_origin_count_per_finalist
            ),
            "local_batch_origin_count": self.local_batch_origin_count,
            "max_local_edits_per_batch": self.max_local_edits_per_batch,
            "selection_holdout_origin_count": (
                self.selection_holdout_origin_count
            ),
            "local_evaluation_mode": self.local_evaluation_mode,
        }

    @property
    def batch_count(self) -> int:
        return self.formal_origin_count_per_finalist // self.local_batch_origin_count

    @property
    def max_local_edits_per_finalist(self) -> int:
        return self.batch_count * self.max_local_edits_per_batch

    def generation_execution_budget(
        self, *, cells_per_origin: int
    ) -> dict[str, int]:
        cells = _positive_integer(cells_per_origin, "cells_per_origin")
        screening = self.screening_origin_count * 4
        formal = self.formal_origin_count_per_finalist * self.finalist_count
        holdout = self.selection_holdout_origin_count * (
            self.finalist_count + 1
        )
        total = screening + formal + holdout
        return {
            "screening_candidate_origins": screening,
            "formal_candidate_origins": formal,
            "holdout_candidate_origins": holdout,
            "total_candidate_origins": total,
            "total_scoring_cells": total * cells,
        }

    def run_execution_budget(
        self,
        planned_generations: int,
        *,
        cells_per_origin: int,
    ) -> dict[str, int]:
        generations = _positive_integer(
            planned_generations, "planned_generations"
        )
        per_generation = self.generation_execution_budget(
            cells_per_origin=cells_per_origin
        )
        return {
            key: value * generations
            for key, value in per_generation.items()
        }

    def required_unique_origins(self, planned_generations: int) -> int:
        generations = _positive_integer(
            planned_generations, "planned_generations"
        )
        return self.formal_origin_count_per_finalist + generations * (
            self.screening_origin_count + self.selection_holdout_origin_count
        )

    def planned_origin_occurrences(self, planned_generations: int) -> int:
        """Return executable origin occurrences, including deterministic reuse.

        The historical method name is retained internally for now, but the
        public contract is occurrence-based: a reused source origin is a new
        execution occurrence, not a new independent source.
        """

        return self.required_unique_origins(planned_generations)
