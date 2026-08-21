"""Small immutable data contracts used by the evolution core.

These contracts are deliberately boring.  They are the durable boundary
between the UI/DSH adapter and the director, and can later be replaced by a
versioned wire schema without changing the state machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, TypeVar


JsonObject = dict[str, Any]
_EnumT = TypeVar("_EnumT", bound=Enum)


def utc_now() -> str:
    """Return a sortable UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible values deterministically."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be JSON-compatible and finite") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    # Validate once at the boundary.  This also rejects NaN/Infinity.
    canonical_json(result)
    return result


def _enum(value: Any, enum_type: type[_EnumT], name: str) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(repr(item.value) for item in enum_type)
        raise ValueError(f"{name} must be one of {allowed}") from exc


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CandidateStatus(str, Enum):
    SPAWNED = "spawned"
    EVALUATED = "evaluated"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    FAILED = "failed"
    DUPLICATE = "duplicate"


class PromotionDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class InterventionKind(str, Enum):
    GUIDANCE = "guidance"
    PARAMETER_OVERRIDE = "parameter_override"
    CONSTRAINT = "constraint"
    PARENT_SELECTION = "parent_selection"


class ExpertUncertaintyType(str, Enum):
    """Bounded reasons for asking an expert without pausing evolution."""

    SCIENTIFIC_ASSUMPTION = "scientific_assumption"
    DATA_INTERPRETATION = "data_interpretation"
    MODEL_SELECTION = "model_selection"
    TRADEOFF = "tradeoff"
    GOVERNANCE_BOUNDARY = "governance_boundary"


@dataclass(frozen=True, slots=True)
class TaskManifest:
    """Frozen root of a run.

    ``budget`` accepts either a simple candidate count or the documented
    mapping form.  ``candidates_per_generation`` defaults to one for backward
    compatibility and is capped at eight for the synchronous local runner.
    When a generation limit is present, the candidate limit is expanded as
    needed to reserve every slot in every generation. Optional token limits and
    per-wave reservations are also normalized here so the immutable manifest
    digest covers the effective execution budget.
    """

    task_id: str
    objective: str
    domain_pack: str
    visible_datasets: tuple[str, ...] = ()
    budget: int | Mapping[str, Any] = 1
    seed: int = 0
    seed_policy: str = "fixed"
    policy_version: str = "policy@1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "objective", _text(self.objective, "objective"))
        object.__setattr__(self, "domain_pack", _text(self.domain_pack, "domain_pack"))
        datasets = tuple(_text(item, "visible_datasets item") for item in self.visible_datasets)
        object.__setattr__(self, "visible_datasets", datasets)
        if isinstance(self.budget, Mapping):
            budget = _mapping(self.budget, "budget")
            candidates_per_generation = budget.get("candidates_per_generation", 1)
            _integer(
                candidates_per_generation,
                "budget.candidates_per_generation",
                minimum=1,
            )
            if candidates_per_generation > 8:
                raise ValueError("budget.candidates_per_generation must be <= 8")
            budget["candidates_per_generation"] = candidates_per_generation
            max_generations = budget.get("max_generations")
            if max_generations is not None:
                _integer(
                    max_generations,
                    "budget.max_generations",
                    minimum=1,
                )
            if "max_candidates" not in budget:
                budget["max_candidates"] = int(max_generations or 1) * int(
                    candidates_per_generation
                )
            max_candidates = budget["max_candidates"]
            _integer(max_candidates, "budget.max_candidates", minimum=1)
            for name in ("token_limit", "token_reservation_per_wave"):
                if name in budget:
                    budget[name] = _integer(
                        budget[name], f"budget.{name}", minimum=0
                    )
            if max_generations is not None:
                required_candidates = int(max_generations) * int(
                    candidates_per_generation
                )
                if max_candidates < required_candidates:
                    budget["max_candidates"] = required_candidates
        else:
            _integer(self.budget, "budget", minimum=1)
            budget = {
                "max_candidates": self.budget,
                "candidates_per_generation": 1,
            }
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        object.__setattr__(self, "seed_policy", _text(self.seed_policy, "seed_policy"))
        object.__setattr__(self, "policy_version", _text(self.policy_version, "policy_version"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    @property
    def max_candidates(self) -> int:
        return int(self.budget["max_candidates"])

    @property
    def max_generations(self) -> int:
        """Return the generation budget, falling back to candidates."""

        return int(self.budget.get("max_generations", self.max_candidates))

    @property
    def candidates_per_generation(self) -> int:
        return int(self.budget.get("candidates_per_generation", 1))

    @property
    def token_limit(self) -> int:
        """Return zero when durable usage is accounting-only."""

        return int(self.budget.get("token_limit", 0))

    @property
    def token_reservation_per_wave(self) -> int:
        """Return the frozen cap for one submitted gateway call."""

        return int(self.budget.get("token_reservation_per_wave", 0))

    # Friendly aliases for callers that use the shorter domain/dataset names.
    @property
    def domain(self) -> str:
        return self.domain_pack

    @property
    def dataset(self) -> str | None:
        return self.visible_datasets[0] if self.visible_datasets else None

    def to_dict(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "domain_pack": self.domain_pack,
            "visible_datasets": list(self.visible_datasets),
            "budget": dict(self.budget),
            "seed": self.seed,
            "seed_policy": self.seed_policy,
            "policy_version": self.policy_version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskManifest":
        return cls(**dict(value))

    @property
    def digest(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class Run:
    run_id: str
    task_id: str
    task_manifest_digest: str
    status: RunStatus = RunStatus.CREATED
    generation: int = 0
    session_id: str | None = None
    best_candidate_id: str | None = None
    selection_incumbent_id: str | None = None
    validated_candidate_id: str | None = None
    final_test_candidate_id: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "task_manifest_digest", _text(self.task_manifest_digest, "task_manifest_digest"))
        object.__setattr__(self, "status", _enum(self.status, RunStatus, "status"))
        object.__setattr__(self, "generation", _integer(self.generation, "generation", minimum=0))
        if self.session_id is not None:
            object.__setattr__(self, "session_id", _text(self.session_id, "session_id"))
        for name in (
            "best_candidate_id",
            "selection_incumbent_id",
            "validated_candidate_id",
            "final_test_candidate_id",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        object.__setattr__(self, "created_at", _text(self.created_at, "created_at"))

    def to_dict(self) -> JsonObject:
        result: JsonObject = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task_manifest_digest": self.task_manifest_digest,
            "status": self.status.value,
            "generation": self.generation,
            "session_id": self.session_id,
            "best_candidate_id": self.best_candidate_id,
            "created_at": self.created_at,
        }
        for name in (
            "selection_incumbent_id",
            "validated_candidate_id",
            "final_test_candidate_id",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Run":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class Proposal:
    proposal_id: str
    run_id: str
    generation: int
    title: str
    changes: Mapping[str, Any] = field(default_factory=dict)
    parent_candidate_id: str | None = None
    rationale: str = ""
    # Optional, bounded model-design trace.  Older event records do not have
    # this field and continue to replay because the default is an empty map.
    # The trace is advisory: executable parameter changes remain in
    # ``changes`` and are still checked by the host boundary.
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposal_id", _text(self.proposal_id, "proposal_id"))
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(self, "generation", _integer(self.generation, "generation", minimum=0))
        object.__setattr__(self, "title", _text(self.title, "title"))
        object.__setattr__(self, "changes", _mapping(self.changes, "changes"))
        if self.parent_candidate_id is not None:
            object.__setattr__(self, "parent_candidate_id", _text(self.parent_candidate_id, "parent_candidate_id"))
        if not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string")
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        object.__setattr__(self, "created_at", _text(self.created_at, "created_at"))

    def identity_dict(self) -> JsonObject:
        result = {
            "run_id": self.run_id,
            "generation": self.generation,
            "title": self.title,
            "changes": dict(self.changes),
            "parent_candidate_id": self.parent_candidate_id,
        }
        # Preserve historical proposal digests when no model-design trace is
        # present.  New autonomous proposals include the trace in their
        # identity so it is covered by the audit digest.
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result

    @property
    def digest(self) -> str:
        return digest(self.identity_dict())

    def to_dict(self) -> JsonObject:
        return {
            **self.identity_dict(),
            "proposal_id": self.proposal_id,
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Proposal":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    run_id: str
    proposal_id: str
    generation: int
    slot_index: int = 0
    status: CandidateStatus = CandidateStatus.SPAWNED
    evaluation_id: str | None = None
    promotion_id: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("candidate_id", "run_id", "proposal_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "generation", _integer(self.generation, "generation", minimum=0))
        object.__setattr__(self, "slot_index", _integer(self.slot_index, "slot_index", minimum=0))
        object.__setattr__(self, "status", _enum(self.status, CandidateStatus, "status"))
        for name in ("evaluation_id", "promotion_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        object.__setattr__(self, "created_at", _text(self.created_at, "created_at"))

    def to_dict(self) -> JsonObject:
        return {
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
            "proposal_id": self.proposal_id,
            "generation": self.generation,
            "slot_index": self.slot_index,
            "status": self.status.value,
            "evaluation_id": self.evaluation_id,
            "promotion_id": self.promotion_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Candidate":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A fitted candidate artifact bound to one immutable data snapshot."""

    artifact_id: str
    run_id: str
    candidate_id: str
    model_id: str
    dataset_digest: str
    training_partition: str
    training_rows: int
    parameters: Mapping[str, Any] = field(default_factory=dict)
    learned_parameters: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in (
            "artifact_id",
            "run_id",
            "candidate_id",
            "model_id",
            "dataset_digest",
            "training_partition",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "training_rows",
            _integer(self.training_rows, "training_rows", minimum=1),
        )
        object.__setattr__(self, "parameters", _mapping(self.parameters, "parameters"))
        object.__setattr__(
            self,
            "learned_parameters",
            _mapping(self.learned_parameters, "learned_parameters"),
        )
        object.__setattr__(self, "metrics", _mapping(self.metrics, "metrics"))
        object.__setattr__(self, "created_at", _text(self.created_at, "created_at"))

    def identity_dict(self) -> JsonObject:
        return {
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "model_id": self.model_id,
            "dataset_digest": self.dataset_digest,
            "training_partition": self.training_partition,
            "training_rows": self.training_rows,
            "parameters": dict(self.parameters),
            "learned_parameters": dict(self.learned_parameters),
            "metrics": dict(self.metrics),
        }

    @property
    def digest(self) -> str:
        return digest(self.identity_dict())

    def to_dict(self) -> JsonObject:
        return {
            **self.identity_dict(),
            "artifact_id": self.artifact_id,
            "artifact_digest": self.digest,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelArtifact":
        data = dict(value)
        data.pop("artifact_digest", None)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class Evaluation:
    evaluation_id: str
    run_id: str
    candidate_id: str
    score: float
    passed: bool
    metrics: Mapping[str, Any] = field(default_factory=dict)
    partition: str = "development"
    evaluator_digest: str = "evaluator@local"
    artifact_digest: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("evaluation_id", "run_id", "candidate_id", "partition", "evaluator_digest"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be a number")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")
        object.__setattr__(self, "score", float(self.score))
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool")
        object.__setattr__(self, "metrics", _mapping(self.metrics, "metrics"))
        if self.artifact_digest is not None:
            object.__setattr__(
                self,
                "artifact_digest",
                _text(self.artifact_digest, "artifact_digest"),
            )
        object.__setattr__(self, "created_at", _text(self.created_at, "created_at"))

    def to_dict(self) -> JsonObject:
        return {
            "evaluation_id": self.evaluation_id,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "score": self.score,
            "passed": self.passed,
            "metrics": dict(self.metrics),
            "partition": self.partition,
            "evaluator_digest": self.evaluator_digest,
            "artifact_digest": self.artifact_digest,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Evaluation":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class HumanIntervention:
    """Append-only human guidance that may affect one later proposal."""

    intervention_id: str
    run_id: str
    kind: InterventionKind
    message: str
    created_by: str
    parameter_overrides: Mapping[str, Any] = field(default_factory=dict)
    target_candidate_id: str | None = None
    applied_proposal_id: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("intervention_id", "run_id", "message", "created_by"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "kind", _enum(self.kind, InterventionKind, "kind"))
        object.__setattr__(
            self,
            "parameter_overrides",
            _mapping(self.parameter_overrides, "parameter_overrides"),
        )
        for name in ("target_candidate_id", "applied_proposal_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        object.__setattr__(self, "created_at", _text(self.created_at, "created_at"))

    def to_dict(self) -> JsonObject:
        return {
            "intervention_id": self.intervention_id,
            "run_id": self.run_id,
            "kind": self.kind.value,
            "message": self.message,
            "created_by": self.created_by,
            "parameter_overrides": dict(self.parameter_overrides),
            "target_candidate_id": self.target_candidate_id,
            "applied_proposal_id": self.applied_proposal_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HumanIntervention":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ExpertConsultation:
    """A model-authored, non-blocking question for asynchronous expert review."""

    consultation_id: str
    run_id: str
    generation: int
    uncertainty_type: ExpertUncertaintyType
    question: str
    context: str
    fallback_assumption: str
    requested_expertise: tuple[str, ...]
    options: tuple[str, ...]
    confidence: float
    requested_by_model_id: str
    candidate_id: str | None = None
    non_blocking: bool = True
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name, maximum in (
            ("consultation_id", 500),
            ("run_id", 500),
            ("question", 2000),
            ("context", 4000),
            ("fallback_assumption", 2000),
            ("requested_by_model_id", 500),
            ("created_at", 100),
        ):
            value = _text(getattr(self, name), name)
            if len(value) > maximum:
                raise ValueError(f"{name} is too long")
            object.__setattr__(self, name, value)
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise TypeError("generation must be an integer")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        object.__setattr__(
            self,
            "uncertainty_type",
            _enum(self.uncertainty_type, ExpertUncertaintyType, "uncertainty_type"),
        )
        for name, maximum_items, maximum_length in (
            ("requested_expertise", 8, 160),
            ("options", 8, 500),
        ):
            raw = getattr(self, name)
            if not isinstance(raw, (list, tuple)):
                raise TypeError(f"{name} must be an array")
            if len(raw) > maximum_items:
                raise ValueError(f"{name} contains too many items")
            values: list[str] = []
            for item in raw:
                value = _text(item, f"{name} item")
                if len(value) > maximum_length:
                    raise ValueError(f"{name} item is too long")
                if value not in values:
                    values.append(value)
            object.__setattr__(self, name, tuple(values))
        if not self.requested_expertise:
            raise ValueError("requested_expertise must contain at least one item")
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise TypeError("confidence must be a number")
        if not math.isfinite(float(self.confidence)) or not 0 <= float(
            self.confidence
        ) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", float(self.confidence))
        if self.candidate_id is not None:
            candidate_id = _text(self.candidate_id, "candidate_id")
            if len(candidate_id) > 500:
                raise ValueError("candidate_id is too long")
            object.__setattr__(self, "candidate_id", candidate_id)
        if self.non_blocking is not True:
            raise ValueError("expert consultations must be non-blocking")

    def to_dict(self) -> JsonObject:
        return {
            "consultation_id": self.consultation_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "uncertainty_type": self.uncertainty_type.value,
            "question": self.question,
            "context": self.context,
            "fallback_assumption": self.fallback_assumption,
            "requested_expertise": list(self.requested_expertise),
            "options": list(self.options),
            "confidence": self.confidence,
            "requested_by_model_id": self.requested_by_model_id,
            "candidate_id": self.candidate_id,
            "non_blocking": self.non_blocking,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExpertConsultation":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ExpertConsultationAnswer:
    """An expert answer that may be consumed by one later research iteration."""

    answer_id: str
    run_id: str
    consultation_id: str
    answer: str
    answered_by: str
    selected_option: str | None = None
    effective_generation: int | None = None
    applied_generation: int | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name, maximum in (
            ("answer_id", 500),
            ("run_id", 500),
            ("consultation_id", 500),
            ("answer", 4000),
            ("answered_by", 120),
            ("created_at", 100),
        ):
            value = _text(getattr(self, name), name)
            if len(value) > maximum:
                raise ValueError(f"{name} is too long")
            object.__setattr__(self, name, value)
        if self.selected_option is not None:
            selected = _text(self.selected_option, "selected_option")
            if len(selected) > 500:
                raise ValueError("selected_option is too long")
            object.__setattr__(self, "selected_option", selected)
        for name in ("effective_generation", "applied_generation"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or null")
        if self.applied_generation is not None:
            if self.effective_generation is None:
                raise ValueError("applied answer requires an effective generation")
            if self.applied_generation < self.effective_generation:
                raise ValueError("answer cannot be applied before its effective generation")

    def to_dict(self) -> JsonObject:
        return {
            "answer_id": self.answer_id,
            "run_id": self.run_id,
            "consultation_id": self.consultation_id,
            "answer": self.answer,
            "answered_by": self.answered_by,
            "selected_option": self.selected_option,
            "effective_generation": self.effective_generation,
            "applied_generation": self.applied_generation,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExpertConsultationAnswer":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class Promotion:
    promotion_id: str
    run_id: str
    candidate_id: str
    decision: PromotionDecision
    reason: str
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("promotion_id", "run_id", "candidate_id", "reason"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "decision", _enum(self.decision, PromotionDecision, "decision"))
        object.__setattr__(self, "created_at", _text(self.created_at, "created_at"))

    def to_dict(self) -> JsonObject:
        return {
            "promotion_id": self.promotion_id,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "decision": self.decision.value,
            "reason": self.reason,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Promotion":
        return cls(**dict(value))
