"""Frozen per-generation research advice for autonomous evolution."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.models import JsonObject, canonical_json, digest, utc_now
from .algorithms import PredictorAdoption

RESEARCH_ITERATION_VERSION = "ecologyrsi-dsh.research-iteration/1"

_STATUSES = frozenset(
    {
        "initial_frozen",
        "model_generated",
        "host_fallback",
        "unavailable",
        "recovered_existing_proposal",
    }
)
_FORBIDDEN_PLAN_FIELDS = frozenset(
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
_SECURITY_BOUNDARY = {
    "advisory_output_only": True,
    "registered_host_capabilities_only": True,
    "model_generated_code_execution": False,
    "dynamic_imports": False,
    "shell_execution": False,
}
_HISTORICAL_PROVENANCE_COUNT_FIELDS = frozenset(
    {
        "scanned_run_count",
        "compatible_source_run_count",
        "included_source_run_count",
        "available_generation_count",
        "included_generation_count",
        "omitted_generation_summaries",
        "omitted_detail_count",
        "max_scanned_runs",
        "max_source_runs",
        "max_generation_summaries",
        "max_serialized_bytes",
    }
)
_HISTORICAL_PROVENANCE_FIELDS = (
    _HISTORICAL_PROVENANCE_COUNT_FIELDS
    | {"source_digest", "history_cutoff_seq"}
)


def _text(value: Any, name: str, *, maximum: int = 4000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    result = value.strip()
    if len(result) > maximum:
        raise ValueError(f"{name} is too long")
    return result


def _optional_text(
    value: Any,
    name: str,
    *,
    maximum: int = 4000,
) -> str | None:
    if value is None:
        return None
    return _text(value, name, maximum=maximum)


def _bounded_plan_value(value: Any, name: str, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > 4000:
            raise ValueError(f"{name} contains an overlong string")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number")
        return value
    if depth >= 6:
        raise ValueError(f"{name} is nested too deeply")
    if isinstance(value, Mapping):
        if len(value) > 128:
            raise ValueError(f"{name} contains too many fields")
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise ValueError(f"{name} contains an invalid field name")
            key = raw_key.strip()
            if key.casefold() in _FORBIDDEN_PLAN_FIELDS:
                raise ValueError(
                    f"{name} contains forbidden executable field: {key}"
                )
            result[key] = _bounded_plan_value(
                item,
                f"{name}.{key}",
                depth=depth + 1,
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) > 128:
            raise ValueError(f"{name} contains too many items")
        return [
            _bounded_plan_value(item, f"{name} item", depth=depth + 1)
            for item in value
        ]
    raise TypeError(f"{name} must contain only JSON-compatible values")


def _bounded_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("research iteration plan must be an object")
    result = _bounded_plan_value(value, "research iteration plan")
    assert isinstance(result, dict)
    if len(canonical_json(result)) > 24_000:
        raise ValueError("research iteration plan exceeds the bounded contract")
    return result


def _historical_provenance(value: Any) -> dict[str, Any] | None:
    """Freeze only the host-owned aggregate lineage for prior-run lessons."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("historical_provenance must be an object or null")
    unknown = set(value) - _HISTORICAL_PROVENANCE_FIELDS
    if unknown:
        raise ValueError(
            "historical_provenance contains unsupported fields: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    required = _HISTORICAL_PROVENANCE_FIELDS - {"history_cutoff_seq"}
    missing = required - set(value)
    if missing:
        raise ValueError(
            "historical_provenance is missing fields: "
            + ", ".join(sorted(missing))
        )
    source_digest = _text(
        value["source_digest"],
        "historical_provenance.source_digest",
        maximum=64,
    )
    if len(source_digest) != 64 or any(
        character not in "0123456789abcdef" for character in source_digest
    ):
        raise ValueError("historical_provenance.source_digest must be sha256 hex")
    result: dict[str, Any] = {"source_digest": source_digest}
    for name in sorted(_HISTORICAL_PROVENANCE_COUNT_FIELDS):
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(
                f"historical_provenance.{name} must be a non-negative integer"
            )
        result[name] = raw
    cutoff = value.get("history_cutoff_seq")
    if cutoff is not None:
        if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 0:
            raise ValueError(
                "historical_provenance.history_cutoff_seq must be a "
                "non-negative integer"
            )
        result["history_cutoff_seq"] = cutoff
    if result["compatible_source_run_count"] > result["scanned_run_count"]:
        raise ValueError("historical_provenance compatible run count is invalid")
    if (
        result["included_source_run_count"]
        > result["compatible_source_run_count"]
    ):
        raise ValueError("historical_provenance included run count is invalid")
    if (
        result["included_generation_count"]
        > result["available_generation_count"]
    ):
        raise ValueError("historical_provenance generation count is invalid")
    return result


@dataclass(frozen=True, slots=True)
class ResearchIteration:
    """One replayable research result shared by a generation's candidates."""

    run_id: str
    generation: int
    status: str
    plan: Mapping[str, Any]
    prediction_model_adoption: Mapping[str, Any]
    knowledge_snapshot_digest: str
    source_analysis_digest: str | None = None
    historical_provenance: Mapping[str, Any] | None = None
    source_assessment_digest: str | None = None
    previous_next_action: str | None = None
    model_id: str | None = None
    pending_consultation_ids: tuple[str, ...] = ()
    expert_answer_ids: tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_now)
    schema_version: str = RESEARCH_ITERATION_VERSION
    iteration_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id", maximum=500))
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("generation must be a non-negative integer")
        status = _text(self.status, "status", maximum=100)
        if status not in _STATUSES:
            raise ValueError("unsupported research iteration status")
        object.__setattr__(self, "status", status)
        if self.schema_version != RESEARCH_ITERATION_VERSION:
            raise ValueError("unsupported research iteration version")
        plan = _bounded_plan(self.plan)
        object.__setattr__(self, "plan", plan)
        adoption = PredictorAdoption.from_dict(self.prediction_model_adoption)
        if adoption.plan_digest != digest(plan):
            raise ValueError("research iteration adoption does not match its plan")
        object.__setattr__(
            self,
            "prediction_model_adoption",
            adoption.to_dict(),
        )
        object.__setattr__(
            self,
            "knowledge_snapshot_digest",
            _text(
                self.knowledge_snapshot_digest,
                "knowledge_snapshot_digest",
                maximum=500,
            ),
        )
        for name in ("source_analysis_digest", "source_assessment_digest"):
            object.__setattr__(
                self,
                name,
                _optional_text(getattr(self, name), name, maximum=500),
            )
        object.__setattr__(
            self,
            "historical_provenance",
            _historical_provenance(self.historical_provenance),
        )
        object.__setattr__(
            self,
            "previous_next_action",
            _optional_text(
                self.previous_next_action,
                "previous_next_action",
                maximum=1000,
            ),
        )
        object.__setattr__(
            self,
            "model_id",
            _optional_text(self.model_id, "model_id", maximum=500),
        )
        for name in ("pending_consultation_ids", "expert_answer_ids"):
            raw = getattr(self, name)
            if not isinstance(raw, (list, tuple)):
                raise TypeError(f"{name} must be an array")
            if len(raw) > 16:
                raise ValueError(f"{name} contains too many items")
            values = tuple(
                _text(item, f"{name} item", maximum=500) for item in raw
            )
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicate items")
            object.__setattr__(self, name, values)
        if self.status in {"host_fallback", "unavailable"} and self.expert_answer_ids:
            raise ValueError("fallback research iterations cannot consume expert answers")
        object.__setattr__(
            self,
            "created_at",
            _text(self.created_at, "created_at", maximum=100),
        )
        expected = digest(self.identity_dict())
        if self.iteration_digest and self.iteration_digest != expected:
            raise ValueError("research iteration digest mismatch")
        object.__setattr__(self, "iteration_digest", expected)

    def identity_dict(self) -> JsonObject:
        result: JsonObject = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generation": self.generation,
            "status": self.status,
            "plan": dict(self.plan),
            "prediction_model_adoption": dict(self.prediction_model_adoption),
            "knowledge_snapshot_digest": self.knowledge_snapshot_digest,
            "source_analysis_digest": self.source_analysis_digest,
            "source_assessment_digest": self.source_assessment_digest,
            "previous_next_action": self.previous_next_action,
            "model_id": self.model_id,
            "security_boundary": dict(_SECURITY_BOUNDARY),
            "created_at": self.created_at,
        }
        # Keep old /1 event digests stable: legacy iterations did not include
        # either field, so empty values remain omitted during replay.
        if self.pending_consultation_ids:
            result["pending_consultation_ids"] = list(
                self.pending_consultation_ids
            )
        if self.expert_answer_ids:
            result["expert_answer_ids"] = list(self.expert_answer_ids)
        if self.historical_provenance is not None:
            result["historical_provenance"] = dict(self.historical_provenance)
        return result

    def to_dict(self) -> JsonObject:
        return {**self.identity_dict(), "iteration_digest": self.iteration_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResearchIteration:
        if not isinstance(value, Mapping):
            raise TypeError("research iteration must be an object")
        data = dict(value)
        boundary = data.pop("security_boundary", None)
        if boundary != _SECURITY_BOUNDARY:
            raise ValueError("research iteration security boundary is invalid")
        return cls(**data)


__all__ = ["RESEARCH_ITERATION_VERSION", "ResearchIteration"]
