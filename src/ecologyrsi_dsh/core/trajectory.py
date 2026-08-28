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


class EvaluationPhase(str, Enum):
    SCREENING = "screening"
    FORMAL_BATCH = "formal_batch"
    HOLDOUT = "holdout"


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

    def __post_init__(self) -> None:
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
        if self.phase is EvaluationPhase.SCREENING:
            if self.batch_index is not None or self.holdout_arm is not None:
                raise ValueError("screening scope cannot have batch_index or holdout_arm")
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

    @property
    def scope_key(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> JsonObject:
        return {
            "run_id": self.run_id,
            "generation": self.generation,
            "candidate_id": self.candidate_id,
            "candidate_revision_id": self.candidate_revision_id,
            "phase": self.phase.value,
            "cohort_digest": self.cohort_digest,
            "origin_count": self.origin_count,
            "batch_index": self.batch_index,
            "holdout_arm": (
                self.holdout_arm.value if self.holdout_arm is not None else None
            ),
        }

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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BatchEvaluation":
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
        if set(self.arm_bindings) != expected:
            raise ValueError("holdout requires exactly three holdout arms")
        normalized: dict[str, Mapping[str, str]] = {}
        identities: set[tuple[str, str]] = set()
        for arm in HoldoutArm:
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
        if len(identities) != len(HoldoutArm):
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
        if {item.scope.holdout_arm for item in evaluations} != set(HoldoutArm):
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
