"""Immutable knowledge records persisted in the evolution event ledger."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..core.models import JsonObject, canonical_json, digest, utc_now

_EXECUTION_STATUSES = frozenset(
    {"adopted", "available_not_selected", "research_only", "metadata_only"}
)
_EVIDENCE_CATALOG_FIELDS = frozenset(
    {
        "knowledge_id",
        "title",
        "summary",
        "source_url",
        "source_kind",
        "source_authority",
        "execution_status",
        "capability_kind",
        "capability_id",
        "capability_ids",
        "parameter_hints",
        "evidence_digest",
    }
)
_EVIDENCE_CATALOG_MAX_ITEMS = 24
_EVIDENCE_CATALOG_MAX_BYTES = 48_000
_EVIDENCE_SUMMARY_MAX_CHARS = 1_600
_EVIDENCE_CAPABILITY_IDS_MAX_ITEMS = 16
_EVIDENCE_PARAMETER_HINTS_MAX_ITEMS = 16
_KNOWLEDGE_CONTEXT_MAX_BYTES = 64_000
_RETRIEVAL_STATUSES = frozenset(
    {
        "catalog_only",
        "catalog_and_online",
        "catalog_online_empty",
        "catalog_online_fallback",
    }
)
_RETRIEVAL_WARNING_MAX_ITEMS = 16


def _text(value: Any, name: str, *, maximum: int = 4000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    result = value.strip()
    if len(result) > maximum:
        raise ValueError(f"{name} is too long")
    return result


def _optional_text(value: Any, name: str, *, maximum: int = 4000) -> str | None:
    if value is None:
        return None
    return _text(value, name, maximum=maximum)


def _text_tuple(value: Any, name: str, *, maximum_items: int = 64) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be an array")
    if len(value) > maximum_items:
        raise ValueError(f"{name} contains too many items")
    return tuple(_text(item, f"{name} item", maximum=500) for item in value)


def _truncated_text(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    return value[: maximum - 3].rstrip() + "..."


def _evidence_identity(card: KnowledgeCard) -> JsonObject:
    return {
        "knowledge_id": card.knowledge_id,
        "title": card.title,
        "summary": _truncated_text(
            card.abstract_summary or card.summary,
            _EVIDENCE_SUMMARY_MAX_CHARS,
        ),
        "source_url": card.source_url,
        "source_kind": card.source_kind,
        "source_authority": card.source_authority,
        "execution_status": card.execution_status,
        "capability_kind": card.capability_kind,
        "capability_id": card.capability_id,
        "capability_ids": list(
            card.capability_ids[:_EVIDENCE_CAPABILITY_IDS_MAX_ITEMS]
        ),
        "parameter_hints": list(
            card.parameter_hints[:_EVIDENCE_PARAMETER_HINTS_MAX_ITEMS]
        ),
    }


@dataclass(frozen=True, slots=True)
class KnowledgeCard:
    """One curated or online knowledge item and its local execution decision."""

    knowledge_id: str
    title: str
    summary: str
    source_url: str
    source_kind: str
    source_authority: str
    execution_status: str
    selection_reason: str
    algorithm_tags: tuple[str, ...] = ()
    capability_kind: str | None = None
    capability_id: str | None = None
    capability_ids: tuple[str, ...] = ()
    parameter_hints: tuple[str, ...] = ()
    publication_year: int | None = None
    cited_by_count: int | None = None
    abstract_summary: str | None = None
    content_digest: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "knowledge_id",
            "title",
            "summary",
            "source_url",
            "source_kind",
            "source_authority",
            "selection_reason",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if not self.source_url.startswith("https://"):
            raise ValueError("knowledge source_url must use https")
        if self.execution_status not in _EXECUTION_STATUSES:
            raise ValueError(f"unknown knowledge execution_status: {self.execution_status}")
        object.__setattr__(
            self, "algorithm_tags", _text_tuple(self.algorithm_tags, "algorithm_tags")
        )
        object.__setattr__(
            self, "parameter_hints", _text_tuple(self.parameter_hints, "parameter_hints")
        )
        object.__setattr__(
            self, "capability_kind", _optional_text(self.capability_kind, "capability_kind")
        )
        object.__setattr__(
            self, "capability_id", _optional_text(self.capability_id, "capability_id")
        )
        object.__setattr__(
            self, "capability_ids", _text_tuple(self.capability_ids, "capability_ids")
        )
        object.__setattr__(
            self,
            "abstract_summary",
            _optional_text(self.abstract_summary, "abstract_summary", maximum=2000),
        )
        object.__setattr__(
            self,
            "content_digest",
            _optional_text(self.content_digest, "content_digest", maximum=64),
        )
        if self.content_digest is not None and (
            len(self.content_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.content_digest)
        ):
            raise ValueError("content_digest must be a lowercase SHA-256 digest")
        for name in ("publication_year", "cited_by_count"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or null")

    @property
    def executable(self) -> bool:
        return self.execution_status == "adopted"

    def to_dict(self) -> JsonObject:
        result: JsonObject = {
            "knowledge_id": self.knowledge_id,
            "title": self.title,
            "summary": self.summary,
            "source_url": self.source_url,
            "source_kind": self.source_kind,
            "source_authority": self.source_authority,
            "execution_status": self.execution_status,
            "executable": self.executable,
            "selection_reason": self.selection_reason,
            "algorithm_tags": list(self.algorithm_tags),
            "capability_kind": self.capability_kind,
            "capability_id": self.capability_id,
            "parameter_hints": list(self.parameter_hints),
            "publication_year": self.publication_year,
            "cited_by_count": self.cited_by_count,
        }
        # Keep the legacy card shape stable so historical snapshot digests replay.
        if self.abstract_summary is not None:
            result["abstract_summary"] = self.abstract_summary
        if self.content_digest is not None:
            result["content_digest"] = self.content_digest
        if self.capability_ids:
            result["capability_ids"] = list(self.capability_ids)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KnowledgeCard:
        data = dict(value)
        data.pop("executable", None)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class KnowledgeSnapshot:
    """Frozen knowledge shared by all candidates in one generation."""

    run_id: str
    generation: int
    query_terms: tuple[str, ...]
    cards: tuple[KnowledgeCard, ...]
    online_enabled: bool
    provider: str
    retrieval_status: str
    warnings: tuple[str, ...] = ()
    retrieved_at: str = field(default_factory=utc_now)
    snapshot_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("generation must be a non-negative integer")
        object.__setattr__(self, "query_terms", _text_tuple(self.query_terms, "query_terms"))
        if not isinstance(self.cards, (list, tuple)) or len(self.cards) > 32:
            raise ValueError("cards must contain at most 32 items")
        object.__setattr__(
            self,
            "cards",
            tuple(
                item if isinstance(item, KnowledgeCard) else KnowledgeCard.from_dict(item)
                for item in self.cards
            ),
        )
        if not isinstance(self.online_enabled, bool):
            raise TypeError("online_enabled must be a bool")
        object.__setattr__(self, "provider", _text(self.provider, "provider"))
        object.__setattr__(
            self, "retrieval_status", _text(self.retrieval_status, "retrieval_status")
        )
        object.__setattr__(self, "warnings", _text_tuple(self.warnings, "warnings"))
        object.__setattr__(self, "retrieved_at", _text(self.retrieved_at, "retrieved_at"))
        expected = digest(self.identity_dict())
        if self.snapshot_digest and self.snapshot_digest != expected:
            raise ValueError("knowledge snapshot digest mismatch")
        object.__setattr__(self, "snapshot_digest", expected)

    @property
    def executable_cards(self) -> tuple[KnowledgeCard, ...]:
        return tuple(item for item in self.cards if item.executable)

    def identity_dict(self) -> JsonObject:
        return {
            "schema_version": "ecologyrsi-dsh.knowledge-snapshot/1",
            "run_id": self.run_id,
            "generation": self.generation,
            "query_terms": list(self.query_terms),
            "cards": [item.to_dict() for item in self.cards],
            "online_enabled": self.online_enabled,
            "provider": self.provider,
            "retrieval_status": self.retrieval_status,
            "warnings": list(self.warnings),
            "retrieved_at": self.retrieved_at,
        }

    def to_dict(self) -> JsonObject:
        return {**self.identity_dict(), "snapshot_digest": self.snapshot_digest}

    def proposal_context(self) -> JsonObject:
        """Return bounded evidence; executable code and raw documents are excluded."""

        adopted = [
            {
                "knowledge_id": item.knowledge_id,
                "title": item.title,
                "summary": item.summary,
                "source_url": item.source_url,
                "capability_kind": item.capability_kind,
                "capability_id": item.capability_id,
                "parameter_hints": list(item.parameter_hints),
                "selection_reason": item.selection_reason,
            }
            for item in self.executable_cards[:16]
        ]
        research = [
            {"knowledge_id": item.knowledge_id, "title": item.title}
            for item in self.cards
            if not item.executable
        ][:8]
        evidence_catalog: list[JsonObject] = []
        for item in self.cards[:_EVIDENCE_CATALOG_MAX_ITEMS]:
            identity = _evidence_identity(item)
            evidence = {**identity, "evidence_digest": digest(identity)}
            if (
                len(canonical_json([*evidence_catalog, evidence]).encode("utf-8"))
                > _EVIDENCE_CATALOG_MAX_BYTES
            ):
                break
            evidence_catalog.append(evidence)
        return {
            "snapshot_digest": self.snapshot_digest,
            "query_terms": list(self.query_terms),
            "retrieval": {
                "online_enabled": self.online_enabled,
                "status": self.retrieval_status,
                "provider": self.provider,
                "warnings": list(self.warnings[:_RETRIEVAL_WARNING_MAX_ITEMS]),
            },
            "adopted_knowledge": adopted,
            "research_only_knowledge": research,
            "evidence_catalog": evidence_catalog,
            "safety_boundary": {
                "external_code_execution": False,
                "parameter_ranges_are_fixed": True,
                "scientific_gate_is_not_mutable": True,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KnowledgeSnapshot:
        data = dict(value)
        data.pop("schema_version", None)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class KnowledgeAssessment:
    """Non-causal round-end assessment of knowledge-guided search."""

    run_id: str
    generation: int
    snapshot_digest: str
    outcome: str
    conclusion: str
    next_action: str
    selected_candidate_id: str | None = None
    score_delta: float | None = None
    non_causal: bool = True
    created_at: str = field(default_factory=utc_now)
    assessment_digest: str = ""

    def __post_init__(self) -> None:
        for name in ("run_id", "snapshot_digest", "outcome", "conclusion", "next_action"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("generation must be a non-negative integer")
        object.__setattr__(
            self,
            "selected_candidate_id",
            _optional_text(self.selected_candidate_id, "selected_candidate_id"),
        )
        if self.score_delta is not None and (
            isinstance(self.score_delta, bool)
            or not isinstance(self.score_delta, (int, float))
            or not math.isfinite(float(self.score_delta))
        ):
            raise ValueError("score_delta must be finite or null")
        if not isinstance(self.non_causal, bool) or not self.non_causal:
            raise ValueError("knowledge assessment must remain non-causal")
        object.__setattr__(self, "created_at", _text(self.created_at, "created_at"))
        expected = digest(self.identity_dict())
        if self.assessment_digest and self.assessment_digest != expected:
            raise ValueError("knowledge assessment digest mismatch")
        object.__setattr__(self, "assessment_digest", expected)

    def identity_dict(self) -> JsonObject:
        return {
            "schema_version": "ecologyrsi-dsh.knowledge-assessment/1",
            "run_id": self.run_id,
            "generation": self.generation,
            "snapshot_digest": self.snapshot_digest,
            "outcome": self.outcome,
            "conclusion": self.conclusion,
            "next_action": self.next_action,
            "selected_candidate_id": self.selected_candidate_id,
            "score_delta": self.score_delta,
            "non_causal": self.non_causal,
            "created_at": self.created_at,
        }

    def to_dict(self) -> JsonObject:
        return {**self.identity_dict(), "assessment_digest": self.assessment_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KnowledgeAssessment:
        data = dict(value)
        data.pop("schema_version", None)
        return cls(**data)


def validate_knowledge_context(value: Any) -> dict[str, Any] | None:
    """Validate a proposal-facing knowledge summary without accepting code."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("knowledge_snapshot must be an object")
    allowed = {
        "snapshot_digest",
        "query_terms",
        "retrieval",
        "adopted_knowledge",
        "research_only_knowledge",
        "evidence_catalog",
        "safety_boundary",
    }
    if set(value) - allowed:
        raise ValueError("knowledge_snapshot contains unsupported fields")
    encoded = canonical_json(value).encode("utf-8")
    if len(encoded) > _KNOWLEDGE_CONTEXT_MAX_BYTES:
        raise ValueError("knowledge_snapshot is too large")
    boundary = value.get("safety_boundary")
    if not isinstance(boundary, Mapping) or boundary.get("external_code_execution") is not False:
        raise ValueError("knowledge_snapshot must prohibit external code execution")
    retrieval = value.get("retrieval")
    if retrieval is not None:
        if not isinstance(retrieval, Mapping):
            raise TypeError("knowledge_snapshot.retrieval must be an object")
        if set(retrieval) != {
            "online_enabled",
            "status",
            "provider",
            "warnings",
        }:
            raise ValueError(
                "knowledge_snapshot.retrieval fields do not match the retrieval schema"
            )
        if not isinstance(retrieval["online_enabled"], bool):
            raise TypeError("knowledge_snapshot.retrieval.online_enabled must be a bool")
        status = _text(
            retrieval["status"],
            "knowledge_snapshot.retrieval.status",
            maximum=64,
        )
        if status not in _RETRIEVAL_STATUSES:
            raise ValueError("knowledge_snapshot.retrieval.status is unknown")
        _text(
            retrieval["provider"],
            "knowledge_snapshot.retrieval.provider",
            maximum=500,
        )
        warnings = retrieval["warnings"]
        if not isinstance(warnings, list):
            raise TypeError("knowledge_snapshot.retrieval.warnings must be an array")
        _text_tuple(
            warnings,
            "knowledge_snapshot.retrieval.warnings",
            maximum_items=_RETRIEVAL_WARNING_MAX_ITEMS,
        )
    adopted = value.get("adopted_knowledge", [])
    if not isinstance(adopted, list) or len(adopted) > 16:
        raise ValueError("adopted_knowledge must be a bounded array")
    if "evidence_catalog" not in value:
        return dict(value)
    catalog = value["evidence_catalog"]
    if not isinstance(catalog, list) or len(catalog) > _EVIDENCE_CATALOG_MAX_ITEMS:
        raise ValueError("evidence_catalog must be an array with at most 24 items")
    if len(canonical_json(catalog).encode("utf-8")) > _EVIDENCE_CATALOG_MAX_BYTES:
        raise ValueError("evidence_catalog is too large")
    knowledge_ids: set[str] = set()
    for index, raw_item in enumerate(catalog):
        name = f"evidence_catalog[{index}]"
        if not isinstance(raw_item, Mapping):
            raise TypeError(f"{name} must be an object")
        if set(raw_item) != _EVIDENCE_CATALOG_FIELDS:
            raise ValueError(f"{name} fields do not match the evidence schema")
        knowledge_id = _text(
            raw_item["knowledge_id"], f"{name}.knowledge_id", maximum=4000
        )
        if knowledge_id in knowledge_ids:
            raise ValueError("evidence_catalog knowledge_id values must be unique")
        knowledge_ids.add(knowledge_id)
        _text(raw_item["title"], f"{name}.title", maximum=4000)
        _text(
            raw_item["summary"],
            f"{name}.summary",
            maximum=_EVIDENCE_SUMMARY_MAX_CHARS,
        )
        source_url = _text(
            raw_item["source_url"], f"{name}.source_url", maximum=4000
        )
        if not source_url.startswith("https://"):
            raise ValueError(f"{name}.source_url must use https")
        _text(raw_item["source_kind"], f"{name}.source_kind", maximum=4000)
        _text(
            raw_item["source_authority"],
            f"{name}.source_authority",
            maximum=4000,
        )
        execution_status = _text(
            raw_item["execution_status"],
            f"{name}.execution_status",
            maximum=64,
        )
        if execution_status not in _EXECUTION_STATUSES:
            raise ValueError(f"{name}.execution_status is unknown")
        for field_name in ("capability_kind", "capability_id"):
            _optional_text(
                raw_item[field_name],
                f"{name}.{field_name}",
                maximum=4000,
            )
        capability_ids = raw_item["capability_ids"]
        if not isinstance(capability_ids, list):
            raise TypeError(f"{name}.capability_ids must be an array")
        _text_tuple(
            capability_ids,
            f"{name}.capability_ids",
            maximum_items=_EVIDENCE_CAPABILITY_IDS_MAX_ITEMS,
        )
        hints = raw_item["parameter_hints"]
        if not isinstance(hints, list):
            raise TypeError(f"{name}.parameter_hints must be an array")
        _text_tuple(
            hints,
            f"{name}.parameter_hints",
            maximum_items=_EVIDENCE_PARAMETER_HINTS_MAX_ITEMS,
        )
        evidence_digest = _text(
            raw_item["evidence_digest"],
            f"{name}.evidence_digest",
            maximum=64,
        )
        if len(evidence_digest) != 64 or any(
            character not in "0123456789abcdef" for character in evidence_digest
        ):
            raise ValueError(f"{name}.evidence_digest must be a lowercase SHA-256 digest")
        identity = {
            field_name: raw_item[field_name]
            for field_name in _EVIDENCE_CATALOG_FIELDS
            if field_name != "evidence_digest"
        }
        if evidence_digest != digest(identity):
            raise ValueError(f"{name}.evidence_digest mismatch")
    return dict(value)


__all__ = [
    "KnowledgeAssessment",
    "KnowledgeCard",
    "KnowledgeSnapshot",
    "validate_knowledge_context",
]
