"""Immutable contracts for model-directed research and generation reflection."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.models import JsonObject, canonical_json, digest, utc_now


AUTONOMOUS_RESEARCH_PROTOCOL = "dsh-model-search-reflect@1"
SEARCH_PLAN_SCHEMA_VERSION = "ecologyrsi-dsh.research-search-plan/1"
GENERATION_REFLECTION_SCHEMA_VERSION = "ecologyrsi-dsh.generation-reflection/1"
RESEARCH_SYNTHESIS_SCHEMA_VERSION = "ecologyrsi-dsh.research-synthesis/1"

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_DIRECTION_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}")
CANDIDATE_MUTATION_AXES = frozenset(
    {
        "scientific_parameter",
        "registered_predictor",
        "instruction_profile",
    }
)
_MUTATION_DIRECTIONS_BY_AXIS = {
    "scientific_parameter": frozenset({"increase", "decrease"}),
    "registered_predictor": frozenset({"select"}),
    "instruction_profile": frozenset({"select"}),
}
_FORBIDDEN_KEYS = frozenset(
    {
        "code",
        "command",
        "entrypoint",
        "module",
        "script",
        "shell",
        "source_code",
    }
)


def _text(value: Any, name: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    result = " ".join(value.split())
    if len(result) > maximum:
        raise ValueError(f"{name} is too long")
    return result


def _optional_digest(value: Any, name: str) -> str | None:
    if value is None:
        return None
    result = _text(value, name, maximum=64)
    if _DIGEST_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return result


def _generation(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("generation must be a non-negative integer")
    return value


def normalize_search_queries(
    value: Any,
    *,
    name: str = "search_queries",
    minimum_items: int = 1,
) -> tuple[str, ...]:
    """Validate the only model-controlled input accepted by online retrieval."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    if not minimum_items <= len(value) <= 6:
        raise ValueError(f"{name} must contain between {minimum_items} and 6 items")
    result: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        query = _text(raw, f"{name}[{index}]", maximum=180)
        if any(ord(character) < 32 for character in query):
            raise ValueError(f"{name}[{index}] contains a control character")
        identity = query.casefold()
        if identity in seen:
            raise ValueError(f"{name} contains duplicate queries")
        seen.add(identity)
        result.append(query)
    return tuple(result)


def _text_tuple(
    value: Any,
    name: str,
    *,
    minimum_items: int = 0,
    maximum_items: int = 16,
    maximum_length: int = 500,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    if not minimum_items <= len(value) <= maximum_items:
        raise ValueError(
            f"{name} must contain between {minimum_items} and {maximum_items} items"
        )
    result = tuple(
        _text(item, f"{name} item", maximum=maximum_length) for item in value
    )
    if len({item.casefold() for item in result}) != len(result):
        raise ValueError(f"{name} contains duplicate items")
    return result


_CANDIDATE_OUTCOME_FIELDS = frozenset(
    {
        "rank",
        "candidate_id",
        "direction_id",
        "direction_digest",
        "slot_index",
        "status",
        "score",
        "eligible",
        "classification",
        "selection_reason",
        "mutation_operations",
        "behavior_digest",
    }
)


def _normalize_candidate_outcomes(value: Any) -> tuple[Mapping[str, Any], ...]:
    """Validate the Host-owned, rank-ordered candidate/reflection binding."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("canonical_candidate_outcomes must be an array")
    if len(value) > 8:
        raise ValueError("canonical_candidate_outcomes contains too many items")
    outcomes: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    direction_ids: set[str] = set()
    direction_digests: set[str] = set()
    slot_indices: set[int] = set()
    ranked: list[int] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _CANDIDATE_OUTCOME_FIELDS:
            raise ValueError(
                f"canonical_candidate_outcomes[{index}] fields do not match "
                "the contract"
            )
        rank = raw.get("rank")
        if rank is not None and (
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 1
        ):
            raise ValueError(
                f"canonical_candidate_outcomes[{index}].rank must be a "
                "positive integer or null"
            )
        score = raw.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError(
                f"canonical_candidate_outcomes[{index}].score must be a "
                "finite number or null"
            )
        if (rank is None) != (score is None):
            raise ValueError(
                "canonical candidate rank and score must both be present or null"
            )
        slot_index = raw.get("slot_index")
        if (
            isinstance(slot_index, bool)
            or not isinstance(slot_index, int)
            or slot_index < 0
        ):
            raise ValueError(
                f"canonical_candidate_outcomes[{index}].slot_index must be a "
                "non-negative integer"
            )
        eligible = raw.get("eligible")
        if not isinstance(eligible, bool):
            raise TypeError(
                f"canonical_candidate_outcomes[{index}].eligible must be a bool"
            )
        candidate_id = _text(
            raw.get("candidate_id"),
            f"canonical_candidate_outcomes[{index}].candidate_id",
            maximum=500,
        )
        direction_id = _text(
            raw.get("direction_id"),
            f"canonical_candidate_outcomes[{index}].direction_id",
            maximum=80,
        ).casefold()
        if _DIRECTION_ID_RE.fullmatch(direction_id) is None:
            raise ValueError("candidate outcome direction_id syntax is invalid")
        direction_digest = _optional_digest(
            raw.get("direction_digest"),
            f"canonical_candidate_outcomes[{index}].direction_digest",
        )
        behavior_digest = _optional_digest(
            raw.get("behavior_digest"),
            f"canonical_candidate_outcomes[{index}].behavior_digest",
        )
        if direction_digest is None or behavior_digest is None:
            raise ValueError(
                "canonical candidate direction and behavior digests are required"
            )
        operations = raw.get("mutation_operations")
        if (
            isinstance(operations, (str, bytes))
            or not isinstance(operations, Sequence)
            or len(operations) != 1
            or not isinstance(operations[0], Mapping)
        ):
            raise ValueError(
                "canonical candidate mutation_operations must contain exactly "
                "one object"
            )
        # Round-trip through the canonical encoder to reject non-JSON and
        # non-finite values while severing references to mutable proposal data.
        canonical_operations = json.loads(canonical_json(list(operations)))
        outcome = {
            "rank": rank,
            "candidate_id": candidate_id,
            "direction_id": direction_id,
            "direction_digest": direction_digest,
            "slot_index": slot_index,
            "status": _text(
                raw.get("status"),
                f"canonical_candidate_outcomes[{index}].status",
                maximum=80,
            ),
            "score": float(score) if score is not None else None,
            "eligible": eligible,
            "classification": _text(
                raw.get("classification"),
                f"canonical_candidate_outcomes[{index}].classification",
                maximum=160,
            ),
            "selection_reason": _text(
                raw.get("selection_reason"),
                f"canonical_candidate_outcomes[{index}].selection_reason",
                maximum=240,
            ),
            "mutation_operations": canonical_operations,
            "behavior_digest": behavior_digest,
        }
        if candidate_id in candidate_ids:
            raise ValueError("canonical candidate outcomes contain duplicate candidates")
        if direction_id in direction_ids or direction_digest in direction_digests:
            raise ValueError("canonical candidate outcomes contain duplicate directions")
        if slot_index in slot_indices:
            raise ValueError("canonical candidate outcomes contain duplicate slots")
        candidate_ids.add(candidate_id)
        direction_ids.add(direction_id)
        direction_digests.add(direction_digest)
        slot_indices.add(slot_index)
        if rank is not None:
            ranked.append(rank)
        outcomes.append(outcome)
    if sorted(ranked) != list(range(1, len(ranked) + 1)):
        raise ValueError("canonical candidate outcome ranks must be contiguous")
    expected_order = sorted(
        outcomes,
        key=lambda item: (
            item["rank"] is None,
            item["rank"] if item["rank"] is not None else 0,
            item["slot_index"],
            item["candidate_id"],
        ),
    )
    if outcomes != expected_order:
        raise ValueError("canonical candidate outcomes must be ordered by rank")
    return tuple(outcomes)


def _candidate_outcomes_to_json(
    outcomes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return json.loads(canonical_json(list(outcomes)))


@dataclass(frozen=True, slots=True)
class CandidateDirection:
    """One model-authored hypothesis that a candidate slot must implement."""

    direction_id: str
    title: str
    hypothesis: str
    target_weakness: str
    capability_focus: str
    mutation_axis: str
    mutation_target: str
    evidence_refs: tuple[str, ...]
    expected_tradeoff: str
    success_criterion: str
    mutation_direction: str
    _legacy_mutation_direction_omitted: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        direction_id = _text(self.direction_id, "direction_id", maximum=80).casefold()
        if _DIRECTION_ID_RE.fullmatch(direction_id) is None:
            raise ValueError("direction_id must use lowercase identifier syntax")
        object.__setattr__(self, "direction_id", direction_id)
        for name, maximum in (
            ("title", 240),
            ("hypothesis", 1600),
            ("target_weakness", 800),
            ("capability_focus", 240),
            ("expected_tradeoff", 1000),
            ("success_criterion", 1000),
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, maximum=maximum),
            )
        mutation_axis = _text(
            self.mutation_axis,
            "mutation_axis",
            maximum=80,
        )
        if mutation_axis not in CANDIDATE_MUTATION_AXES:
            raise ValueError("unsupported candidate mutation_axis")
        object.__setattr__(self, "mutation_axis", mutation_axis)
        object.__setattr__(
            self,
            "mutation_target",
            _text(self.mutation_target, "mutation_target", maximum=160),
        )
        mutation_direction = _text(
            self.mutation_direction,
            "mutation_direction",
            maximum=40,
        )
        if mutation_direction not in _MUTATION_DIRECTIONS_BY_AXIS[mutation_axis]:
            raise ValueError("mutation_direction is incompatible with mutation_axis")
        object.__setattr__(self, "mutation_direction", mutation_direction)
        object.__setattr__(
            self,
            "evidence_refs",
            _text_tuple(
                self.evidence_refs,
                "evidence_refs",
                maximum_items=16,
                maximum_length=160,
            ),
        )

    @property
    def direction_digest(self) -> str:
        return digest(self.identity_dict())

    def identity_dict(self) -> JsonObject:
        identity = {
            "direction_id": self.direction_id,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "target_weakness": self.target_weakness,
            "capability_focus": self.capability_focus,
            "mutation_axis": self.mutation_axis,
            "mutation_target": self.mutation_target,
            "evidence_refs": list(self.evidence_refs),
            "expected_tradeoff": self.expected_tradeoff,
            "success_criterion": self.success_criterion,
        }
        if not self._legacy_mutation_direction_omitted:
            identity["mutation_direction"] = self.mutation_direction
        return identity

    def to_dict(self) -> JsonObject:
        return {**self.identity_dict(), "direction_digest": self.direction_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateDirection:
        if not isinstance(value, Mapping):
            raise TypeError("candidate direction must be an object")
        data = dict(value)
        supplied_digest = data.pop("direction_digest", None)
        required = {
            "direction_id",
            "title",
            "hypothesis",
            "target_weakness",
            "capability_focus",
            "mutation_axis",
            "mutation_target",
            "mutation_direction",
            "evidence_refs",
            "expected_tradeoff",
            "success_criterion",
        }
        if set(data) != required:
            raise ValueError("candidate direction fields do not match the contract")
        item = cls(**data)
        if supplied_digest is not None and supplied_digest != item.direction_digest:
            raise ValueError("candidate direction digest mismatch")
        return item

    @classmethod
    def from_legacy_dict(cls, value: Mapping[str, Any]) -> CandidateDirection:
        """Replay a pre-directionality candidate hypothesis by its old digest."""

        if not isinstance(value, Mapping):
            raise TypeError("candidate direction must be an object")
        data = dict(value)
        supplied_digest = data.pop("direction_digest", None)
        required = {
            "direction_id",
            "title",
            "hypothesis",
            "target_weakness",
            "capability_focus",
            "mutation_axis",
            "mutation_target",
            "evidence_refs",
            "expected_tradeoff",
            "success_criterion",
        }
        if set(data) != required:
            raise ValueError("legacy candidate direction fields do not match the contract")
        if supplied_digest != digest(data):
            raise ValueError("legacy candidate direction digest mismatch")
        placeholder = (
            "select"
            if data.get("mutation_axis") in {"registered_predictor", "instruction_profile"}
            else "increase"
        )
        item = cls(mutation_direction=placeholder, **data)
        object.__setattr__(item, "_legacy_mutation_direction_omitted", True)
        object.__setattr__(item, "mutation_direction", "legacy_unspecified")
        if item.direction_digest != supplied_digest:
            raise ValueError("legacy candidate direction normalization mismatch")
        return item


def normalize_candidate_directions(
    value: Any,
    *,
    minimum_items: int = 1,
    maximum_items: int = 8,
    exact_items: int | None = None,
    allowed_mutation_targets: Mapping[str, Collection[str]] | None = None,
) -> tuple[CandidateDirection, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("candidate_directions must be an array")
    if exact_items is not None and len(value) != exact_items:
        raise ValueError(
            f"candidate_directions must contain exactly {exact_items} items"
        )
    if not minimum_items <= len(value) <= maximum_items:
        raise ValueError(
            f"candidate_directions must contain between {minimum_items} and "
            f"{maximum_items} items"
        )
    directions = tuple(
        item
        if isinstance(item, CandidateDirection)
        else CandidateDirection.from_dict(item)
        for item in value
    )
    identities = [item.direction_id for item in directions]
    if len(set(identities)) != len(identities):
        raise ValueError("candidate direction ids must be unique")
    hypotheses = [item.hypothesis.casefold() for item in directions]
    if len(set(hypotheses)) != len(hypotheses):
        raise ValueError("candidate direction hypotheses must be distinct")
    allowed = None
    if allowed_mutation_targets is not None:
        unknown_axes = set(allowed_mutation_targets) - CANDIDATE_MUTATION_AXES
        if unknown_axes:
            raise ValueError(
                "allowed_mutation_targets contains unsupported axes: "
                + ", ".join(sorted(unknown_axes))
            )
        allowed = {
            axis: {
                str(item).strip()
                for item in targets
                if isinstance(item, str) and str(item).strip()
            }
            for axis, targets in allowed_mutation_targets.items()
        }
    for index, direction in enumerate(directions):
        if allowed is None:
            continue
        targets = allowed.get(direction.mutation_axis, set())
        if direction.mutation_target not in targets:
            raise ValueError(
                f"candidate_directions[{index}] selects unavailable "
                f"{direction.mutation_axis} target: {direction.mutation_target}"
            )
    return directions


@dataclass(frozen=True, slots=True)
class GenerationSearchPlan:
    """A strategy-model search request executed by the bounded Host retriever."""

    run_id: str
    generation: int
    search_queries: tuple[str, ...]
    focus_areas: tuple[str, ...]
    rationale: str
    source_analysis_digest: str | None = None
    source_reflection_digest: str | None = None
    model_id: str | None = None
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SEARCH_PLAN_SCHEMA_VERSION
    search_plan_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id", maximum=500))
        object.__setattr__(self, "generation", _generation(self.generation))
        object.__setattr__(
            self,
            "search_queries",
            normalize_search_queries(self.search_queries),
        )
        object.__setattr__(
            self,
            "focus_areas",
            _text_tuple(
                self.focus_areas,
                "focus_areas",
                minimum_items=1,
                maximum_items=8,
                maximum_length=240,
            ),
        )
        object.__setattr__(
            self,
            "rationale",
            _text(self.rationale, "rationale", maximum=4000),
        )
        for name in ("source_analysis_digest", "source_reflection_digest"):
            object.__setattr__(
                self,
                name,
                _optional_digest(getattr(self, name), name),
            )
        if self.model_id is not None:
            object.__setattr__(
                self,
                "model_id",
                _text(self.model_id, "model_id", maximum=500),
            )
        object.__setattr__(
            self,
            "created_at",
            _text(self.created_at, "created_at", maximum=100),
        )
        if self.schema_version != SEARCH_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported generation search-plan schema")
        expected = digest(self.identity_dict())
        if self.search_plan_digest and self.search_plan_digest != expected:
            raise ValueError("generation search-plan digest mismatch")
        object.__setattr__(self, "search_plan_digest", expected)

    def identity_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generation": self.generation,
            "search_queries": list(self.search_queries),
            "focus_areas": list(self.focus_areas),
            "rationale": self.rationale,
            "source_analysis_digest": self.source_analysis_digest,
            "source_reflection_digest": self.source_reflection_digest,
            "model_id": self.model_id,
            "created_at": self.created_at,
        }

    def to_dict(self) -> JsonObject:
        return {**self.identity_dict(), "search_plan_digest": self.search_plan_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GenerationSearchPlan:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class GenerationReflection:
    """Batch-level model reflection used to seed the next research cycle."""

    run_id: str
    generation: int
    analysis_digest: str
    summary: str
    lessons: tuple[str, ...]
    recommended_search_queries: tuple[str, ...]
    candidate_directions: tuple[CandidateDirection, ...]
    stop_recommendation: str
    canonical_candidate_outcomes: tuple[Mapping[str, Any], ...] = ()
    model_id: str | None = None
    created_at: str = field(default_factory=utc_now)
    schema_version: str = GENERATION_REFLECTION_SCHEMA_VERSION
    reflection_digest: str = ""
    _legacy_candidate_outcomes_omitted: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id", maximum=500))
        object.__setattr__(self, "generation", _generation(self.generation))
        analysis_digest = _optional_digest(self.analysis_digest, "analysis_digest")
        assert analysis_digest is not None
        object.__setattr__(self, "analysis_digest", analysis_digest)
        object.__setattr__(
            self,
            "summary",
            _text(self.summary, "summary", maximum=6000),
        )
        object.__setattr__(
            self,
            "lessons",
            _text_tuple(
                self.lessons,
                "lessons",
                minimum_items=1,
                maximum_items=16,
                maximum_length=800,
            ),
        )
        object.__setattr__(
            self,
            "recommended_search_queries",
            normalize_search_queries(self.recommended_search_queries),
        )
        directions = normalize_candidate_directions(
            self.candidate_directions,
            minimum_items=2,
            maximum_items=8,
        )
        object.__setattr__(self, "candidate_directions", directions)
        stop = _text(
            self.stop_recommendation,
            "stop_recommendation",
            maximum=40,
        )
        if stop not in {"continue", "converged", "insufficient_evidence"}:
            raise ValueError("unsupported generation stop recommendation")
        object.__setattr__(self, "stop_recommendation", stop)
        object.__setattr__(
            self,
            "canonical_candidate_outcomes",
            _normalize_candidate_outcomes(self.canonical_candidate_outcomes),
        )
        if self.model_id is not None:
            object.__setattr__(
                self,
                "model_id",
                _text(self.model_id, "model_id", maximum=500),
            )
        object.__setattr__(
            self,
            "created_at",
            _text(self.created_at, "created_at", maximum=100),
        )
        if self.schema_version != GENERATION_REFLECTION_SCHEMA_VERSION:
            raise ValueError("unsupported generation reflection schema")
        expected = digest(self.identity_dict())
        if self.reflection_digest and self.reflection_digest != expected:
            raise ValueError("generation reflection digest mismatch")
        object.__setattr__(self, "reflection_digest", expected)

    def identity_dict(self) -> JsonObject:
        identity = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generation": self.generation,
            "analysis_digest": self.analysis_digest,
            "summary": self.summary,
            "lessons": list(self.lessons),
            "recommended_search_queries": list(self.recommended_search_queries),
            "candidate_directions": [
                item.to_dict() for item in self.candidate_directions
            ],
            "stop_recommendation": self.stop_recommendation,
            "model_id": self.model_id,
            "created_at": self.created_at,
        }
        if not self._legacy_candidate_outcomes_omitted:
            identity["canonical_candidate_outcomes"] = _candidate_outcomes_to_json(
                self.canonical_candidate_outcomes
            )
        return identity

    def to_dict(self) -> JsonObject:
        return {**self.identity_dict(), "reflection_digest": self.reflection_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GenerationReflection:
        if not isinstance(value, Mapping):
            raise TypeError("generation reflection must be an object")
        required = {
            "schema_version",
            "run_id",
            "generation",
            "analysis_digest",
            "summary",
            "lessons",
            "recommended_search_queries",
            "candidate_directions",
            "stop_recommendation",
            "canonical_candidate_outcomes",
            "model_id",
            "created_at",
            "reflection_digest",
        }
        if set(value) != required:
            raise ValueError("generation reflection fields do not match the contract")
        return cls(**dict(value))

    @classmethod
    def from_legacy_dict(cls, value: Mapping[str, Any]) -> GenerationReflection:
        """Replay the exact pre-outcome reflection shape without weakening writes."""

        if not isinstance(value, Mapping):
            raise TypeError("generation reflection must be an object")
        required = {
            "schema_version",
            "run_id",
            "generation",
            "analysis_digest",
            "summary",
            "lessons",
            "recommended_search_queries",
            "candidate_directions",
            "stop_recommendation",
            "model_id",
            "created_at",
            "reflection_digest",
        }
        if set(value) != required:
            raise ValueError("legacy generation reflection fields do not match the contract")
        supplied_digest = value.get("reflection_digest")
        if supplied_digest != digest(
            {key: item for key, item in value.items() if key != "reflection_digest"}
        ):
            raise ValueError("legacy generation reflection digest mismatch")
        arguments = dict(value)
        arguments["reflection_digest"] = ""
        arguments["candidate_directions"] = tuple(
            CandidateDirection.from_dict(direction)
            if "mutation_direction" in direction
            else CandidateDirection.from_legacy_dict(direction)
            for direction in arguments["candidate_directions"]
        )
        item = cls(canonical_candidate_outcomes=(), **arguments)
        object.__setattr__(item, "_legacy_candidate_outcomes_omitted", True)
        if digest(item.identity_dict()) != supplied_digest:
            raise ValueError("legacy generation reflection normalization mismatch")
        object.__setattr__(item, "reflection_digest", supplied_digest)
        return item


def validate_research_synthesis(
    value: Any,
    *,
    candidate_count: int,
    allowed_evidence_refs: set[str],
    allowed_mutation_targets: Mapping[str, Collection[str]] | None = None,
) -> dict[str, Any]:
    """Validate research output and bind every current candidate to one direction."""

    if not isinstance(value, Mapping):
        raise TypeError("research synthesis must be an object")
    if set(value) != {"schema_version", "summary", "evidence", "candidate_directions"}:
        raise ValueError("research synthesis fields do not match the contract")
    if value.get("schema_version") != RESEARCH_SYNTHESIS_SCHEMA_VERSION:
        raise ValueError("unsupported research synthesis schema")
    summary = _text(value.get("summary"), "research summary", maximum=12_000)
    raw_evidence = value.get("evidence")
    if isinstance(raw_evidence, (str, bytes)) or not isinstance(raw_evidence, Sequence):
        raise TypeError("research evidence must be an array")
    if len(raw_evidence) > 64:
        raise ValueError("research evidence contains too many items")
    evidence: list[dict[str, str]] = []
    for index, raw in enumerate(raw_evidence):
        if not isinstance(raw, Mapping) or set(raw) != {
            "evidence_ref",
            "finding",
            "relevance",
        }:
            raise ValueError(f"research evidence[{index}] fields are invalid")
        evidence_ref = _text(
            raw.get("evidence_ref"),
            f"research evidence[{index}].evidence_ref",
            maximum=160,
        )
        if evidence_ref not in allowed_evidence_refs:
            raise ValueError("research synthesis cited evidence outside the frozen snapshot")
        evidence.append(
            {
                "evidence_ref": evidence_ref,
                "finding": _text(
                    raw.get("finding"),
                    f"research evidence[{index}].finding",
                    maximum=1600,
                ),
                "relevance": _text(
                    raw.get("relevance"),
                    f"research evidence[{index}].relevance",
                    maximum=1000,
                ),
            }
        )
    directions = normalize_candidate_directions(
        value.get("candidate_directions"),
        exact_items=candidate_count,
        allowed_mutation_targets=allowed_mutation_targets,
    )
    for direction in directions:
        if any(reference not in allowed_evidence_refs for reference in direction.evidence_refs):
            raise ValueError("candidate direction cited evidence outside the frozen snapshot")
    return {
        "schema_version": RESEARCH_SYNTHESIS_SCHEMA_VERSION,
        "summary": summary,
        "evidence": evidence,
        "candidate_directions": [item.to_dict() for item in directions],
    }


def reject_executable_fields(value: Any, *, path: str = "$") -> None:
    """Fail closed if a reflection/search payload attempts executable content."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in _FORBIDDEN_KEYS:
                raise ValueError(f"forbidden executable field at {path}.{key}")
            reject_executable_fields(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            reject_executable_fields(item, path=f"{path}[{index}]")


__all__ = [
    "AUTONOMOUS_RESEARCH_PROTOCOL",
    "CANDIDATE_MUTATION_AXES",
    "CandidateDirection",
    "GENERATION_REFLECTION_SCHEMA_VERSION",
    "GenerationReflection",
    "GenerationSearchPlan",
    "RESEARCH_SYNTHESIS_SCHEMA_VERSION",
    "SEARCH_PLAN_SCHEMA_VERSION",
    "normalize_candidate_directions",
    "normalize_search_queries",
    "reject_executable_fields",
    "validate_research_synthesis",
]
