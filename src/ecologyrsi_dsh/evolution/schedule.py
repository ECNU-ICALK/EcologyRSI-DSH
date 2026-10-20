"""The single frozen joint-plugin schedule, expressed in forecast origins."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..core.agent_prediction import PREDICTION_TOOL_CALL_BUDGET
from . import joint_policy

# Identifiers remain importable while callers are migrated; none is accepted.
LEGACY_SCHEDULE_SCHEMA_VERSION = "ecologyrsi-dsh.top2-adaptive-epoch-schedule/1"
SCHEDULE_SCHEMA_VERSION = "ecologyrsi-dsh.top2-adaptive-epoch-schedule/2"
ISOLATED_SCHEDULE_SCHEMA_VERSION = "ecologyrsi-dsh.top2-adaptive-epoch-schedule/3"
TRAINING_SCHEDULE_SCHEMA_VERSION = "ecologyrsi-dsh.top2-adaptive-epoch-schedule/4"
QUICK_SCHEDULE_SCHEMA_VERSION = "ecologyrsi-dsh.quick-adaptive-epoch-schedule/1"
EVIDENCE_GUIDED_SCHEDULE_SCHEMA = "ecologyrsi-dsh.evidence-guided-epoch-schedule/1"
OPTIMIZATION_PROTOCOL = "top2_adaptive_epoch@1"
QUICK_OPTIMIZATION_PROTOCOL = "quick_adaptive_epoch@1"
EVIDENCE_GUIDED_PROTOCOL = "evidence_guided_epoch@1"
PREQUENTIAL_PERFORMANCE_POLICY = "defer_performance_to_paired_epoch@1"
PREQUENTIAL_LOCAL_EVALUATION_MODE = "prequential"
PAIRED_LOCAL_EVALUATION_MODE = "paired_champion_challenger"
ADAPTIVE_PROTOCOLS = (joint_policy.UNIFIED_PROTOCOL,)

_FIELDS = frozenset((
    "schema_version", "screening_origin_count", "finalist_count",
    "formal_origin_count_per_finalist", "local_batch_origin_count",
    "max_local_edits_per_batch", "selection_holdout_origin_count",
    "local_evaluation_mode",
))


def _positive_integer(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class OptimizationSchedule:
    schema_version: str
    screening_origin_count: int
    finalist_count: int
    formal_origin_count_per_finalist: int
    local_batch_origin_count: int
    max_local_edits_per_batch: int
    selection_holdout_origin_count: int
    local_evaluation_mode: str

    def __post_init__(self) -> None:
        if self.schema_version != joint_policy.UNIFIED_SCHEDULE_SCHEMA:
            raise ValueError("Only the joint_plugin_epoch@2 schedule is supported")
        if self.local_evaluation_mode != PAIRED_LOCAL_EVALUATION_MODE:
            raise ValueError("joint_plugin_epoch@2 requires paired local evaluation")
        expected = {
            "screening_origin_count": 10,
            "finalist_count": 1,
            "formal_origin_count_per_finalist": 35,
            "local_batch_origin_count": 25,
            "max_local_edits_per_batch": 2,
            "selection_holdout_origin_count": 50,
        }
        for name, value in expected.items():
            if type(getattr(self, name)) is not int or getattr(self, name) != value:
                raise ValueError(f"joint_plugin_epoch@2 freezes {name}={value}")

    @classmethod
    def for_new_run(cls) -> OptimizationSchedule:
        return cls(joint_policy.UNIFIED_SCHEDULE_SCHEMA, 10, 1, 35, 25, 2, 50,
                   PAIRED_LOCAL_EVALUATION_MODE)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OptimizationSchedule:
        if not isinstance(value, Mapping) or set(value) != _FIELDS:
            raise ValueError("optimization_schedule fields differ from its contract")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in sorted(_FIELDS)}

    @property
    def protocol(self) -> str:
        return joint_policy.UNIFIED_PROTOCOL

    @property
    def unified(self) -> bool:
        return True

    @property
    def joint(self) -> bool:
        return True

    @property
    def race(self) -> bool:
        return True

    @property
    def quick(self) -> bool:
        return True

    @property
    def exploratory_local_comparison(self) -> bool:
        return False

    @property
    def batch_count(self) -> int:
        return 2

    def batch_origin_count(self, index: int) -> int:
        if type(index) is not int or index not in (0, 1):
            raise ValueError("joint batch index must be 0 or 1")
        return self.screening_origin_count if index == 0 else self.local_batch_origin_count

    @property
    def max_local_edit_decisions_per_finalist(self) -> int:
        return 1

    @property
    def max_local_edits_per_finalist(self) -> int:
        return self.max_local_edits_per_batch

    def generation_execution_budget(
        self, *, cells_per_origin: int, holdout_inference_replicas: int = 1,
    ) -> dict[str, int]:
        cells = _positive_integer(cells_per_origin, "cells_per_origin")
        _positive_integer(holdout_inference_replicas, "holdout_inference_replicas")
        # Extra holdout inference is conditional and already reserved once.
        return {
            "base_candidate_origins": 200,
            "recheck_candidate_origins": 100,
            "screening_candidate_origins": 50,
            "formal_candidate_origins": 50,
            "holdout_candidate_origins": 100,
            "total_candidate_origins": 300,
            "total_scoring_cells": 300 * cells,
        }

    def run_execution_budget(
        self, planned_generations: int, *, cells_per_origin: int,
        holdout_inference_replicas: int = 1,
    ) -> dict[str, int]:
        generations = _positive_integer(planned_generations, "planned_generations")
        return {key: value * generations for key, value in self.generation_execution_budget(
            cells_per_origin=cells_per_origin,
            holdout_inference_replicas=holdout_inference_replicas,
        ).items()}

    def required_unique_origins(self, planned_generations: int) -> int:
        return 85 * _positive_integer(planned_generations, "planned_generations")

    def planned_origin_occurrences(self, planned_generations: int) -> int:
        return self.required_unique_origins(planned_generations)

    def execution_plan(self, generations: int, *, cells_per_origin: int, native: bool = True) -> dict[str, Any]:
        if native is not True:
            raise ValueError("Only native execution is supported")
        return {
            "schema_version": "ecologyrsi-dsh.execution-plan/1",
            "protocol": self.protocol,
            "recheck_policy": "joint-conditional-paired-recheck@1",
            "maximum_rechecks_per_generation": 1,
            "maximum_holdout_inference_replicas": 2,
            "budget_includes_recheck_reserve": True,
            "schedule": self.to_dict(),
            "generations": generations,
            "holdout_inference_replicas": 1,
            "cells_per_origin": cells_per_origin,
            "generation_budget": self.generation_execution_budget(cells_per_origin=cells_per_origin),
            "run_budget": self.run_execution_budget(generations, cells_per_origin=cells_per_origin),
            "required_unique_origins": self.required_unique_origins(generations),
            "training_replay": False,
            "screening_includes_incumbent": True,
            "screening_reused_as_diagnosis": True,
            "fresh_epoch_holdout": True,
            "qualification": "exploratory_only",
            "comparison_evidence": "complete_pair_practical_delta_cell_nonregression",
            "output_tokens_per_llm_call": {"planner": 32768, "critic": 8192},
            "sample_execution_limits": {
                "prediction_tool_calls_per_attempt": PREDICTION_TOOL_CALL_BUDGET,
                "planner_steps": 10, "critic_steps": 4,
                "planner_reported_output_threshold": None,
                "critic_reported_output_threshold": None,
            },
        }
