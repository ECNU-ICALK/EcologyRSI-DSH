"""Origin-first execution schedule for Top-2 adaptive epochs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .parameters import PARAMETER_RULES


LEGACY_SCHEDULE_SCHEMA_VERSION = (
    "ecologyrsi-dsh.top2-adaptive-epoch-schedule/1"
)
SCHEDULE_SCHEMA_VERSION = "ecologyrsi-dsh.top2-adaptive-epoch-schedule/2"
ISOLATED_SCHEDULE_SCHEMA_VERSION = "ecologyrsi-dsh.top2-adaptive-epoch-schedule/3"
TRAINING_SCHEDULE_SCHEMA_VERSION = "ecologyrsi-dsh.top2-adaptive-epoch-schedule/4"
OPTIMIZATION_PROTOCOL = "top2_adaptive_epoch@1"
QUICK_OPTIMIZATION_PROTOCOL = "quick_adaptive_epoch@1"
ADAPTIVE_PROTOCOLS = (OPTIMIZATION_PROTOCOL, QUICK_OPTIMIZATION_PROTOCOL)
QUICK_SCHEDULE_SCHEMA_VERSION = "ecologyrsi-dsh.quick-adaptive-epoch-schedule/1"
PREQUENTIAL_LOCAL_EVALUATION_MODE = "prequential"
PAIRED_LOCAL_EVALUATION_MODE = "paired_champion_challenger"

_SCHEDULE_MODES = {
    LEGACY_SCHEDULE_SCHEMA_VERSION: PREQUENTIAL_LOCAL_EVALUATION_MODE,
    SCHEDULE_SCHEMA_VERSION: PAIRED_LOCAL_EVALUATION_MODE,
    ISOLATED_SCHEDULE_SCHEMA_VERSION: PAIRED_LOCAL_EVALUATION_MODE,
    TRAINING_SCHEDULE_SCHEMA_VERSION: PAIRED_LOCAL_EVALUATION_MODE,
    QUICK_SCHEDULE_SCHEMA_VERSION: PREQUENTIAL_LOCAL_EVALUATION_MODE,
}

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
        expected_mode = _SCHEDULE_MODES.get(self.schema_version)
        if expected_mode is None:
            raise ValueError(
                "schema_version must be one of "
                f"{tuple(_SCHEDULE_MODES)!r}"
            )
        if type(self.screening_origin_count) is not int or self.screening_origin_count != (0 if self.quick else 64):
            raise ValueError("screening_origin_count differs from protocol")
        if type(self.finalist_count) is not int or self.finalist_count != (1 if self.quick else 2):
            raise ValueError("finalist_count differs from protocol")
        if self.local_evaluation_mode != expected_mode:
            raise ValueError(
                "local_evaluation_mode must be "
                f"{expected_mode!r} for schema_version {self.schema_version!r}"
            )
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
        if self.schema_version in {ISOLATED_SCHEDULE_SCHEMA_VERSION, TRAINING_SCHEDULE_SCHEMA_VERSION, QUICK_SCHEDULE_SCHEMA_VERSION} and batch < 2:
            raise ValueError("isolated adaptation batches require at least two origins")
        if formal % batch:
            raise ValueError(
                "local_batch_origin_count must divide "
                "formal_origin_count_per_finalist"
            )
        if not 0 <= edits <= PARAMETER_RULES["max_local_edits_per_batch"]["maximum"]:
            raise ValueError("max_local_edits_per_batch must be between 0 and 5")
        minimum_holdout = (PARAMETER_RULES["selection_holdout_origin_count"]["minimum"]
                           if self.schema_version == TRAINING_SCHEDULE_SCHEMA_VERSION or self.quick else 169)
        if holdout < minimum_holdout:
            raise ValueError(f"selection_holdout_origin_count must be at least {minimum_holdout}")

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
            local_evaluation_mode=PAIRED_LOCAL_EVALUATION_MODE,
        )

    @classmethod
    def for_new_run(cls) -> "OptimizationSchedule":
        """Repeated training epochs, with independent evaluation kept outside search."""
        return replace(cls.default(), schema_version=QUICK_SCHEDULE_SCHEMA_VERSION,
                       screening_origin_count=0, finalist_count=1,
                       local_evaluation_mode=PREQUENTIAL_LOCAL_EVALUATION_MODE,
                       formal_origin_count_per_finalist=PARAMETER_RULES["formal_origin_count"]["default"],
                       local_batch_origin_count=PARAMETER_RULES["local_batch_origin_count"]["default"],
                       selection_holdout_origin_count=PARAMETER_RULES["selection_holdout_origin_count"]["default"])

    @property
    def quick(self) -> bool:
        return self.schema_version == QUICK_SCHEDULE_SCHEMA_VERSION

    @classmethod
    def for_comparison_run(cls) -> "OptimizationSchedule":
        return replace(cls.for_new_run(), schema_version=TRAINING_SCHEDULE_SCHEMA_VERSION,
                       screening_origin_count=64, finalist_count=2,
                       local_evaluation_mode=PAIRED_LOCAL_EVALUATION_MODE)

    def execution_plan(self, generations: int, *, cells_per_origin: int, native: bool = True) -> dict[str, Any]:
        replicas = 1 if self.quick or not native else 2
        return {
            "schema_version": "ecologyrsi-dsh.execution-plan/1",
            "protocol": self.protocol,
            "schedule": self.to_dict(),
            "generations": generations,
            "holdout_inference_replicas": replicas,
            "cells_per_origin": cells_per_origin,
            "generation_budget": self.generation_execution_budget(cells_per_origin=cells_per_origin, holdout_inference_replicas=replicas),
            "run_budget": self.run_execution_budget(generations, cells_per_origin=cells_per_origin, holdout_inference_replicas=replicas),
            "required_unique_origins": self.required_unique_origins(generations),
            "training_replay": True,
            "fresh_epoch_holdout": self.quick,
            "qualification": "exploratory_only" if self.quick else "comparison_requires_certification_gates",
            "comparison_evidence": "complete_pair_practical_delta_cell_nonregression" if self.quick else "paired_time_blocks_and_inference_replicas",
            # The per-call values are the frozen NATIVE_SAMPLE_OPERATION_MAX_TOKENS;
            # the reported thresholds abort a runaway child at twice one full
            # response, well above the 4087..6347 spent by observed successful
            # planners and the 1277..3080 spent by observed critics. Keep them
            # equal to the plugin's SAMPLE_STAGE_LIMITS.
            "output_tokens_per_llm_call": {"planner": 16384, "critic": 8192},
            "sample_execution_limits": {"prediction_tool_calls_per_attempt": 2,
                "planner_steps": 10, "critic_steps": 4,
                "planner_reported_output_threshold": 32768, "critic_reported_output_threshold": 16384},
        }

    @property
    def protocol(self) -> str:
        return QUICK_OPTIMIZATION_PROTOCOL if self.quick else OPTIMIZATION_PROTOCOL

    @property
    def exploratory_local_comparison(self) -> bool:
        """Small training batches screen edits; epoch evidence confirms retention."""
        return self.schema_version == TRAINING_SCHEDULE_SCHEMA_VERSION

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
        return (
            self.max_local_edit_decisions_per_finalist
            * self.max_local_edits_per_batch
        )

    @property
    def max_local_edit_decisions_per_finalist(self) -> int:
        if self.local_evaluation_mode == PAIRED_LOCAL_EVALUATION_MODE:
            return self.batch_count - 1
        return self.batch_count

    def generation_execution_budget(
        self, *, cells_per_origin: int, holdout_inference_replicas: int = 1
    ) -> dict[str, int]:
        cells = _positive_integer(cells_per_origin, "cells_per_origin")
        replicas = _positive_integer(holdout_inference_replicas, "holdout_inference_replicas")
        if self.quick:
            replicas = 1
        screening = self.screening_origin_count * 4
        if self.local_evaluation_mode == PAIRED_LOCAL_EVALUATION_MODE:
            formal_per_finalist = self.local_batch_origin_count + (
                2
                * (self.batch_count - 1)
                * self.local_batch_origin_count
            )
            formal = formal_per_finalist * self.finalist_count
        else:
            formal = self.formal_origin_count_per_finalist * self.finalist_count
        holdout = replicas * self.selection_holdout_origin_count * (
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
        holdout_inference_replicas: int = 1,
    ) -> dict[str, int]:
        generations = _positive_integer(
            planned_generations, "planned_generations"
        )
        per_generation = self.generation_execution_budget(
            cells_per_origin=cells_per_origin,
            holdout_inference_replicas=holdout_inference_replicas,
        )
        return {
            key: value * generations
            for key, value in per_generation.items()
        }

    def required_unique_origins(self, planned_generations: int) -> int:
        generations = _positive_integer(
            planned_generations, "planned_generations"
        )
        if self.schema_version == TRAINING_SCHEDULE_SCHEMA_VERSION:
            return self.formal_origin_count_per_finalist + self.screening_origin_count + self.selection_holdout_origin_count
        return self.formal_origin_count_per_finalist + generations * (
            self.screening_origin_count + self.selection_holdout_origin_count
        )

    def planned_origin_occurrences(self, planned_generations: int) -> int:
        """Count training uses separately from distinct source observations."""
        if self.quick:
            return self.required_unique_origins(planned_generations) + (planned_generations - 1) * self.formal_origin_count_per_finalist
        if self.schema_version == TRAINING_SCHEDULE_SCHEMA_VERSION:
            return self.required_unique_origins(planned_generations) * planned_generations
        return self.required_unique_origins(planned_generations)
