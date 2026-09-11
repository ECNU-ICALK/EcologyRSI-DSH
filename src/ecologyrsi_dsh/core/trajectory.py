"""Immutable identities for Top-2 adaptive finalist trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping

from .models import (
    JsonObject,
    _enum,
    _integer,
    _text,
    canonical_json,
    digest,
    utc_now,
)
from .search_policy import PAIRED_EXECUTION_QUALIFICATION


class EvaluationPhase(str, Enum):
    SCREENING = "screening"
    FORMAL_BATCH = "formal_batch"
    HOLDOUT = "holdout"
    VALIDATION = "validation"
    FINAL_TEST = "final_test"


class FormalBatchArm(str, Enum):
    CHAMPION = "champion"
    CHALLENGER = "challenger"


class FormalBatchComparisonDecision(str, Enum):
    INITIAL_CHAMPION = "initial_champion"
    CHALLENGER_PROMOTED = "challenger_promoted"
    CHAMPION_RETAINED = "champion_retained"


class RevisionStatus(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FINAL = "final"


class TrajectoryStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"


class LocalEditProposalDecision(str, Enum):
    KEEP = "keep"
    MUTATE = "mutate"


class LocalEditOutcome(str, Enum):
    KEPT = "kept"
    APPLIED = "applied"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class RevisionAdvanceReason(str, Enum):
    KEPT = "kept"
    LOCAL_EDIT_APPLIED = "local_edit_applied"
    LOCAL_EDIT_REJECTED = "local_edit_rejected"
    PREQUENTIAL_SAFETY_ROLLBACK = "prequential_safety_rollback"


class HoldoutArm(str, Enum):
    FINALIST_1 = "finalist_1"
    FINALIST_2 = "finalist_2"
    INCUMBENT = "incumbent"


def _sha256(value: Any, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _deep_freeze_json(value: Any, name: str) -> Any:
    """Validate, detach, and recursively freeze a JSON value."""

    import json

    try:
        detached = json.loads(canonical_json(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be JSON-compatible") from exc

    def freeze(item: Any) -> Any:
        if isinstance(item, dict):
            return MappingProxyType({str(key): freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(detached)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("score must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("score must be finite")
    return result


@dataclass(frozen=True, slots=True)
class OriginOccurrence:
    """One executable forecast-origin occurrence.

    A source origin may be reused after the eligible population is exhausted;
    ``cycle_index`` makes that repeat explicit so it can never be mistaken for
    an additional independent observation.
    """

    occurrence_id: str
    source_origin_id: str
    cycle_index: int
    origin_timestamp: int
    maturity_digest: str
    cohort_role: str
    generation: int
    candidate_id: str
    revision_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "occurrence_id", _sha256(self.occurrence_id, "occurrence_id"))
        object.__setattr__(self, "source_origin_id", _text(self.source_origin_id, "source_origin_id"))
        object.__setattr__(self, "cycle_index", _integer(self.cycle_index, "cycle_index", minimum=0))
        object.__setattr__(self, "origin_timestamp", _integer(self.origin_timestamp, "origin_timestamp"))
        object.__setattr__(self, "maturity_digest", _sha256(self.maturity_digest, "maturity_digest"))
        object.__setattr__(self, "cohort_role", _text(self.cohort_role, "cohort_role"))
        object.__setattr__(self, "generation", _integer(self.generation, "generation", minimum=0))
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "revision_id", _text(self.revision_id, "revision_id"))

    @classmethod
    def from_source(
        cls,
        *,
        source_origin_id: str,
        cycle_index: int,
        origin_timestamp: int,
        maturity_digest: str,
        cohort_role: str,
        generation: int,
        candidate_id: str,
        revision_id: str,
    ) -> "OriginOccurrence":
        identity = {
            "source_origin_id": source_origin_id,
            "cycle_index": cycle_index,
            "origin_timestamp": origin_timestamp,
            "maturity_digest": maturity_digest,
            "cohort_role": cohort_role,
            "generation": generation,
            "candidate_id": candidate_id,
            "revision_id": revision_id,
        }
        return cls(occurrence_id=digest(identity), **identity)

    def to_dict(self) -> JsonObject:
        return {
            "occurrence_id": self.occurrence_id,
            "source_origin_id": self.source_origin_id,
            "cycle_index": self.cycle_index,
            "origin_timestamp": self.origin_timestamp,
            "maturity_digest": self.maturity_digest,
            "cohort_role": self.cohort_role,
            "generation": self.generation,
            "candidate_id": self.candidate_id,
            "revision_id": self.revision_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OriginOccurrence":
        result = cls(**dict(value))
        expected = cls.from_source(
            source_origin_id=result.source_origin_id,
            cycle_index=result.cycle_index,
            origin_timestamp=result.origin_timestamp,
            maturity_digest=result.maturity_digest,
            cohort_role=result.cohort_role,
            generation=result.generation,
            candidate_id=result.candidate_id,
            revision_id=result.revision_id,
        ).occurrence_id
        if result.occurrence_id != expected:
            raise ValueError("occurrence_id does not match occurrence identity")
        return result


@dataclass(frozen=True, slots=True)
class EvaluationScope:
    run_id: str
    generation: int
    candidate_id: str
    candidate_revision_id: str
    phase: EvaluationPhase
    cohort_digest: str
    origin_count: int
    batch_index: int | None = None
    holdout_arm: HoldoutArm | None = None
    formal_batch_arm: FormalBatchArm | None = None
    inference_replica: int = 0

    def __post_init__(self) -> None:
        if type(self.inference_replica) is not int or not 0 <= self.inference_replica < 2:
            raise ValueError("inference_replica must be 0 or 1")
        for name in ("run_id", "candidate_id", "candidate_revision_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "generation", _integer(self.generation, "generation", minimum=0)
        )
        object.__setattr__(self, "phase", _enum(self.phase, EvaluationPhase, "phase"))
        object.__setattr__(
            self,
            "cohort_digest",
            _sha256(self.cohort_digest, "cohort_digest"),
        )
        object.__setattr__(
            self,
            "origin_count",
            _integer(self.origin_count, "origin_count", minimum=1),
        )
        if self.batch_index is not None:
            object.__setattr__(
                self,
                "batch_index",
                _integer(self.batch_index, "batch_index", minimum=0),
            )
        if self.holdout_arm is not None:
            object.__setattr__(
                self,
                "holdout_arm",
                _enum(self.holdout_arm, HoldoutArm, "holdout_arm"),
            )
        if self.formal_batch_arm is not None:
            object.__setattr__(
                self,
                "formal_batch_arm",
                _enum(
                    self.formal_batch_arm,
                    FormalBatchArm,
                    "formal_batch_arm",
                ),
            )
        if self.phase in {EvaluationPhase.SCREENING, EvaluationPhase.VALIDATION, EvaluationPhase.FINAL_TEST}:
            if (
                self.batch_index is not None
                or self.holdout_arm is not None
                or self.formal_batch_arm is not None
            ):
                raise ValueError(
                    f"{self.phase.value} scope cannot have batch_index, holdout_arm, "
                    "or formal_batch_arm"
                )
        elif self.phase is EvaluationPhase.FORMAL_BATCH:
            if self.batch_index is None:
                raise ValueError("formal_batch scope requires batch_index")
            if self.holdout_arm is not None:
                raise ValueError("formal_batch scope cannot have holdout_arm")
        elif self.phase is EvaluationPhase.HOLDOUT:
            if self.holdout_arm is None:
                raise ValueError("holdout scope requires holdout_arm")
            if self.batch_index is not None:
                raise ValueError("holdout scope cannot have batch_index")
            if self.formal_batch_arm is not None:
                raise ValueError("holdout scope cannot have formal_batch_arm")

    @property
    def scope_key(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> JsonObject:
        value: JsonObject = {
            "run_id": self.run_id,
            "generation": self.generation,
            "candidate_id": self.candidate_id,
            "candidate_revision_id": self.candidate_revision_id,
            "phase": self.phase.value,
            "cohort_digest": self.cohort_digest,
            "origin_count": self.origin_count,
            "inference_replica": self.inference_replica,
            "batch_index": self.batch_index,
            "holdout_arm": (
                self.holdout_arm.value if self.holdout_arm is not None else None
            ),
        }
        if self.formal_batch_arm is not None:
            value["formal_batch_arm"] = self.formal_batch_arm.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationScope":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class CandidateRevision:
    revision_id: str
    run_id: str
    generation: int
    candidate_id: str
    genome: Mapping[str, Any]
    genome_digest: str
    behavior_digest: str
    mutation_digest: str
    parent_revision_id: str | None = None
    source_batch_index: int | None = None
    status: RevisionStatus = RevisionStatus.CREATED
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("revision_id", "run_id", "candidate_id", "created_at"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "generation", _integer(self.generation, "generation", minimum=0)
        )
        if self.parent_revision_id is not None:
            object.__setattr__(
                self,
                "parent_revision_id",
                _text(self.parent_revision_id, "parent_revision_id"),
            )
        if self.source_batch_index is not None:
            object.__setattr__(
                self,
                "source_batch_index",
                _integer(
                    self.source_batch_index,
                    "source_batch_index",
                    minimum=0,
                ),
            )
        if (self.parent_revision_id is None) != (self.source_batch_index is None):
            raise ValueError(
                "parent_revision_id and source_batch_index must both be set or both be null"
            )
        object.__setattr__(self, "genome", _deep_freeze_json(self.genome, "genome"))
        for name in ("genome_digest", "behavior_digest", "mutation_digest"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(
            self, "status", _enum(self.status, RevisionStatus, "status")
        )

    def identity_dict(self) -> JsonObject:
        return {
            "revision_id": self.revision_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "candidate_id": self.candidate_id,
            "parent_revision_id": self.parent_revision_id,
            "source_batch_index": self.source_batch_index,
            "genome": _thaw_json(self.genome),
            "genome_digest": self.genome_digest,
            "behavior_digest": self.behavior_digest,
            "mutation_digest": self.mutation_digest,
        }

    @property
    def revision_digest(self) -> str:
        return digest(self.identity_dict())

    def to_dict(self) -> JsonObject:
        return {
            **self.identity_dict(),
            "revision_digest": self.revision_digest,
            "status": self.status.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateRevision":
        data = dict(value)
        supplied_digest = data.pop("revision_digest", None)
        revision = cls(**data)
        if supplied_digest is not None and supplied_digest != revision.revision_digest:
            raise ValueError("revision_digest does not match revision identity")
        return revision


@dataclass(frozen=True, slots=True)
class FormalTrajectory:
    trajectory_id: str
    run_id: str
    generation: int
    candidate_id: str
    initial_revision_id: str
    batch_count: int
    status: TrajectoryStatus = TrajectoryStatus.PENDING
    final_revision_id: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in (
            "trajectory_id",
            "run_id",
            "candidate_id",
            "initial_revision_id",
            "created_at",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "generation", _integer(self.generation, "generation", minimum=0)
        )
        object.__setattr__(
            self, "batch_count", _integer(self.batch_count, "batch_count", minimum=1)
        )
        object.__setattr__(
            self, "status", _enum(self.status, TrajectoryStatus, "status")
        )
        if self.final_revision_id is not None:
            object.__setattr__(
                self,
                "final_revision_id",
                _text(self.final_revision_id, "final_revision_id"),
            )
        if self.status is TrajectoryStatus.COMPLETED and self.final_revision_id is None:
            raise ValueError("completed trajectory requires final_revision_id")

    def to_dict(self) -> JsonObject:
        return {
            "trajectory_id": self.trajectory_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "candidate_id": self.candidate_id,
            "initial_revision_id": self.initial_revision_id,
            "batch_count": self.batch_count,
            "status": self.status.value,
            "final_revision_id": self.final_revision_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalTrajectory":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class FormalBatch:
    batch_id: str
    trajectory_id: str
    run_id: str
    generation: int
    candidate_id: str
    revision_id: str
    revision_candidate_id: str
    batch_index: int
    batch_count: int
    cohort_digest: str
    origin_count: int
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in (
            "batch_id",
            "trajectory_id",
            "run_id",
            "candidate_id",
            "revision_id",
            "revision_candidate_id",
            "created_at",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "generation", _integer(self.generation, "generation", minimum=0)
        )
        object.__setattr__(
            self, "batch_index", _integer(self.batch_index, "batch_index", minimum=0)
        )
        object.__setattr__(
            self, "batch_count", _integer(self.batch_count, "batch_count", minimum=1)
        )
        if self.batch_index >= self.batch_count:
            raise ValueError("batch_index must be less than batch_count")
        if self.revision_candidate_id != self.candidate_id:
            raise ValueError("revision candidate must match formal batch candidate")
        object.__setattr__(
            self,
            "cohort_digest",
            _sha256(self.cohort_digest, "cohort_digest"),
        )
        object.__setattr__(
            self,
            "origin_count",
            _integer(self.origin_count, "origin_count", minimum=1),
        )

    def to_dict(self) -> JsonObject:
        return {
            "batch_id": self.batch_id,
            "trajectory_id": self.trajectory_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "candidate_id": self.candidate_id,
            "revision_id": self.revision_id,
            "revision_candidate_id": self.revision_candidate_id,
            "batch_index": self.batch_index,
            "batch_count": self.batch_count,
            "cohort_digest": self.cohort_digest,
            "origin_count": self.origin_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalBatch":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class BatchEvaluation:
    evaluation_id: str
    scope: EvaluationScope
    score: float
    passed: bool
    metrics: Mapping[str, Any]
    evaluator_digest: str
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evaluation_id", _text(self.evaluation_id, "evaluation_id")
        )
        if isinstance(self.scope, Mapping):
            object.__setattr__(self, "scope", EvaluationScope.from_dict(self.scope))
        if not isinstance(self.scope, EvaluationScope):
            raise TypeError("scope must be an EvaluationScope")
        if self.scope.phase is not EvaluationPhase.FORMAL_BATCH:
            raise ValueError("BatchEvaluation scope phase must be formal_batch")
        object.__setattr__(self, "score", _score(self.score))
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool")
        object.__setattr__(self, "metrics", _deep_freeze_json(self.metrics, "metrics"))
        object.__setattr__(
            self,
            "evaluator_digest",
            _sha256(self.evaluator_digest, "evaluator_digest"),
        )
        object.__setattr__(self, "created_at", _text(self.created_at, "created_at"))

    def to_dict(self) -> JsonObject:
        return {
            "evaluation_id": self.evaluation_id,
            "scope": self.scope.to_dict(),
            "score": self.score,
            "passed": self.passed,
            "metrics": _thaw_json(self.metrics),
            "evaluator_digest": self.evaluator_digest,
            "created_at": self.created_at,
        }

    @property
    def evaluation_digest(self) -> str:
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BatchEvaluation":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class FormalBatchComparison:
    comparison_id: str
    run_id: str
    generation: int
    candidate_id: str
    batch_index: int
    cohort_digest: str
    champion_before_revision_id: str
    challenger_revision_id: str
    champion_evaluation_id: str
    challenger_evaluation_id: str
    champion_evaluation_digest: str
    challenger_evaluation_digest: str
    champion_score: float
    challenger_score: float
    score_delta: float
    comparison_contract_digest: str
    safety_gate_passed: bool
    cell_regression_gate_passed: bool
    minimum_score_delta: float
    decision: FormalBatchComparisonDecision
    champion_after_revision_id: str
    reason: str
    created_at: str = field(default_factory=utc_now)
    paired_execution_qualification: str | None = None

    def __post_init__(self) -> None:
        if self.paired_execution_qualification not in (None, PAIRED_EXECUTION_QUALIFICATION):
            raise ValueError("unsupported paired execution qualification contract")
        for name in (
            "comparison_id",
            "run_id",
            "candidate_id",
            "champion_before_revision_id",
            "challenger_revision_id",
            "champion_evaluation_id",
            "challenger_evaluation_id",
            "champion_after_revision_id",
            "reason",
            "created_at",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "generation",
            _integer(self.generation, "generation", minimum=0),
        )
        object.__setattr__(
            self,
            "batch_index",
            _integer(self.batch_index, "batch_index", minimum=0),
        )
        for name in (
            "cohort_digest",
            "champion_evaluation_digest",
            "challenger_evaluation_digest",
            "comparison_contract_digest",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in ("champion_score", "challenger_score", "score_delta"):
            object.__setattr__(self, name, _score(getattr(self, name)))
        object.__setattr__(
            self,
            "minimum_score_delta",
            _score(self.minimum_score_delta),
        )
        if self.minimum_score_delta < 0:
            raise ValueError("minimum_score_delta must be non-negative")
        for name in ("safety_gate_passed", "cell_regression_gate_passed"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        object.__setattr__(
            self,
            "decision",
            _enum(
                self.decision,
                FormalBatchComparisonDecision,
                "decision",
            ),
        )
        expected_delta = self.challenger_score - self.champion_score
        if not math.isclose(
            self.score_delta,
            expected_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("score_delta must equal challenger_score - champion_score")
        if self.decision is FormalBatchComparisonDecision.INITIAL_CHAMPION:
            if self.batch_index != 0:
                raise ValueError("initial_champion comparison must use batch 0")
            if not (
                self.champion_before_revision_id == self.challenger_revision_id
                == self.champion_after_revision_id
                and self.champion_evaluation_id == self.challenger_evaluation_id
                and self.champion_evaluation_digest
                == self.challenger_evaluation_digest
                and self.champion_score == self.challenger_score
                and self.score_delta == 0.0
            ):
                raise ValueError(
                    "initial_champion must reuse one revision and evaluation"
                )
        elif self.decision is FormalBatchComparisonDecision.CHALLENGER_PROMOTED:
            if self.champion_after_revision_id != self.challenger_revision_id:
                raise ValueError(
                    "challenger_promoted champion_after must be the challenger"
                )
            if self.champion_before_revision_id == self.challenger_revision_id:
                raise ValueError("challenger_promoted requires a distinct challenger")
        elif self.champion_after_revision_id != self.champion_before_revision_id:
            raise ValueError(
                "champion_retained champion_after must be the original champion"
            )

    def to_dict(self) -> JsonObject:
        return {
            "comparison_id": self.comparison_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "candidate_id": self.candidate_id,
            "batch_index": self.batch_index,
            "cohort_digest": self.cohort_digest,
            "champion_before_revision_id": self.champion_before_revision_id,
            "challenger_revision_id": self.challenger_revision_id,
            "champion_evaluation_id": self.champion_evaluation_id,
            "challenger_evaluation_id": self.challenger_evaluation_id,
            "champion_evaluation_digest": self.champion_evaluation_digest,
            "challenger_evaluation_digest": self.challenger_evaluation_digest,
            "champion_score": self.champion_score,
            "challenger_score": self.challenger_score,
            "score_delta": self.score_delta,
            "comparison_contract_digest": self.comparison_contract_digest,
            "safety_gate_passed": self.safety_gate_passed,
            "cell_regression_gate_passed": self.cell_regression_gate_passed,
            "minimum_score_delta": self.minimum_score_delta,
            "decision": self.decision.value,
            "champion_after_revision_id": self.champion_after_revision_id,
            "reason": self.reason,
            "created_at": self.created_at,
            **({"paired_execution_qualification": self.paired_execution_qualification}
               if self.paired_execution_qualification is not None else {}),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalBatchComparison":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class TrajectoryRevisionActivation:
    activation_id: str
    run_id: str
    generation: int
    candidate_id: str
    batch_index: int
    from_revision_id: str
    to_revision_id: str
    reason: RevisionAdvanceReason
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in (
            "activation_id",
            "run_id",
            "candidate_id",
            "from_revision_id",
            "to_revision_id",
            "created_at",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "generation", _integer(self.generation, "generation", minimum=0)
        )
        object.__setattr__(
            self, "batch_index", _integer(self.batch_index, "batch_index", minimum=0)
        )
        object.__setattr__(
            self,
            "reason",
            _enum(self.reason, RevisionAdvanceReason, "reason"),
        )

    def to_dict(self) -> JsonObject:
        return {
            "activation_id": self.activation_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "candidate_id": self.candidate_id,
            "batch_index": self.batch_index,
            "from_revision_id": self.from_revision_id,
            "to_revision_id": self.to_revision_id,
            "reason": self.reason.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrajectoryRevisionActivation":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class GenerationHoldout:
    holdout_id: str
    run_id: str
    generation: int
    cohort_digest: str
    origin_count: int
    arm_bindings: Mapping[str, Mapping[str, str]]
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("holdout_id", "run_id", "created_at"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "generation", _integer(self.generation, "generation", minimum=0)
        )
        object.__setattr__(
            self,
            "cohort_digest",
            _sha256(self.cohort_digest, "cohort_digest"),
        )
        object.__setattr__(
            self,
            "origin_count",
            _integer(self.origin_count, "origin_count", minimum=1),
        )
        if not isinstance(self.arm_bindings, Mapping):
            raise TypeError("arm_bindings must be a mapping")
        expected = {arm.value for arm in HoldoutArm}
        if set(self.arm_bindings) not in (expected, {"finalist_1", "incumbent"}):
            raise ValueError("holdout requires exactly three holdout arms")
        normalized: dict[str, Mapping[str, str]] = {}
        identities: set[tuple[str, str]] = set()
        for arm in map(HoldoutArm, self.arm_bindings):
            raw = self.arm_bindings[arm.value]
            if not isinstance(raw, Mapping) or set(raw) != {
                "candidate_id",
                "candidate_revision_id",
            }:
                raise ValueError("each holdout arm requires candidate and revision")
            candidate_id = _text(raw["candidate_id"], "candidate_id")
            revision_id = _text(
                raw["candidate_revision_id"], "candidate_revision_id"
            )
            identities.add((candidate_id, revision_id))
            normalized[arm.value] = MappingProxyType(
                {
                    "candidate_id": candidate_id,
                    "candidate_revision_id": revision_id,
                }
            )
        if len(identities) != len(self.arm_bindings):
            raise ValueError("holdout arms require unique candidate revisions")
        object.__setattr__(self, "arm_bindings", MappingProxyType(normalized))

    def to_dict(self) -> JsonObject:
        return {
            "holdout_id": self.holdout_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "cohort_digest": self.cohort_digest,
            "origin_count": self.origin_count,
            "arm_bindings": _thaw_json(self.arm_bindings),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationHoldout":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class HoldoutEvaluation:
    evaluation_id: str
    scope: EvaluationScope
    score: float
    passed: bool
    metrics: Mapping[str, Any]
    evaluator_digest: str
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evaluation_id", _text(self.evaluation_id, "evaluation_id")
        )
        if isinstance(self.scope, Mapping):
            object.__setattr__(self, "scope", EvaluationScope.from_dict(self.scope))
        if not isinstance(self.scope, EvaluationScope):
            raise TypeError("scope must be an EvaluationScope")
        if self.scope.phase is not EvaluationPhase.HOLDOUT:
            raise ValueError("HoldoutEvaluation scope phase must be holdout")
        object.__setattr__(self, "score", _score(self.score))
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool")
        object.__setattr__(self, "metrics", _deep_freeze_json(self.metrics, "metrics"))
        object.__setattr__(
            self,
            "evaluator_digest",
            _sha256(self.evaluator_digest, "evaluator_digest"),
        )
        object.__setattr__(self, "created_at", _text(self.created_at, "created_at"))

    def to_dict(self) -> JsonObject:
        return {
            "evaluation_id": self.evaluation_id,
            "scope": self.scope.to_dict(),
            "score": self.score,
            "passed": self.passed,
            "metrics": _thaw_json(self.metrics),
            "evaluator_digest": self.evaluator_digest,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HoldoutEvaluation":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class GenerationComparison:
    comparison_id: str
    run_id: str
    generation: int
    cohort_digest: str
    holdout_evaluations: tuple[HoldoutEvaluation, ...]
    selected_candidate_id: str
    selected_revision_id: str
    gate_results: Mapping[str, Any]
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in (
            "comparison_id",
            "run_id",
            "selected_candidate_id",
            "selected_revision_id",
            "created_at",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "generation", _integer(self.generation, "generation", minimum=0)
        )
        object.__setattr__(
            self,
            "cohort_digest",
            _sha256(self.cohort_digest, "cohort_digest"),
        )
        raw_evaluations = self.holdout_evaluations
        if not isinstance(raw_evaluations, (list, tuple)):
            raise TypeError("holdout_evaluations must be an array")
        evaluations = tuple(
            HoldoutEvaluation.from_dict(item) if isinstance(item, Mapping) else item
            for item in raw_evaluations
        )
        if not all(isinstance(item, HoldoutEvaluation) for item in evaluations):
            raise TypeError("holdout_evaluations must contain HoldoutEvaluation")
        arms = {item.scope.holdout_arm for item in evaluations}
        if len(arms) != len(evaluations) or arms not in (set(HoldoutArm), {HoldoutArm.FINALIST_1, HoldoutArm.INCUMBENT}):
            raise ValueError("comparison requires exactly three holdout arms")
        if any(item.scope.cohort_digest != self.cohort_digest for item in evaluations):
            raise ValueError("comparison evaluations must use the same cohort")
        if any(
            item.scope.run_id != self.run_id
            or item.scope.generation != self.generation
            for item in evaluations
        ):
            raise ValueError("comparison evaluation scope does not match run generation")
        selected = [
            item
            for item in evaluations
            if item.scope.candidate_id == self.selected_candidate_id
            and item.scope.candidate_revision_id == self.selected_revision_id
        ]
        if len(selected) != 1:
            raise ValueError("selected revision must identify one holdout arm")
        object.__setattr__(self, "holdout_evaluations", evaluations)
        object.__setattr__(
            self,
            "gate_results",
            _deep_freeze_json(self.gate_results, "gate_results"),
        )

    @property
    def comparison_digest(self) -> str:
        return digest(self.identity_dict())

    def identity_dict(self) -> JsonObject:
        return {
            "comparison_id": self.comparison_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "cohort_digest": self.cohort_digest,
            "holdout_evaluations": [
                item.to_dict() for item in self.holdout_evaluations
            ],
            "selected_candidate_id": self.selected_candidate_id,
            "selected_revision_id": self.selected_revision_id,
            "gate_results": _thaw_json(self.gate_results),
        }

    def to_dict(self) -> JsonObject:
        return {
            **self.identity_dict(),
            "comparison_digest": self.comparison_digest,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationComparison":
        data = dict(value)
        supplied_digest = data.pop("comparison_digest", None)
        comparison = cls(**data)
        if supplied_digest is not None and supplied_digest != comparison.comparison_digest:
            raise ValueError("comparison_digest does not match comparison identity")
        return comparison
