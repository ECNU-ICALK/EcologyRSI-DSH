"""Immutable diagnostic and hypothesis records used by the native research loop."""
from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping
from typing import Any
import unicodedata
from .identity import canonical_text, content_id


def _text(value: Any, name: str, *, optional: bool = False, limit: int = 4000) -> str:
    if optional and (value is None or (type(value) is str and not value.strip())):
        return ""
    if type(value) is not str or not value.strip() or len(value) > limit:
        raise ValueError(f"{name}_required")
    return unicodedata.normalize("NFC", value)


def _refs(value: Any, name: str, *, optional: bool = True) -> tuple[str, ...]:
    if value is None and optional:
        return ()
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name}_tuple_required")
    result = tuple(_text(item, name, limit=512) for item in value)
    if not optional and not result:
        raise ValueError(f"{name}_required")
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate_{name}")
    # Mutable/ephemeral aliases make evidence impossible to audit.
    if any(item.lower() in {"latest", "current", "head"} for item in result):
        raise ValueError(f"floating_{name}_forbidden")
    return result


def _thaw(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _thaw(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


class _Contract:
    @property
    def canonical_json(self) -> str:
        return canonical_text(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self)


@dataclass(frozen=True, slots=True)
class DiagnosticReport(_Contract):
    task_id: str
    program_id: str
    comparison_id: str | None
    weak_cells: tuple[str, ...]
    failure_patterns: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    possible_causes: tuple[str, ...]
    unresolved_questions: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "program_id", _text(self.program_id, "program_id"))
        if self.comparison_id is not None:
            object.__setattr__(self, "comparison_id", _text(self.comparison_id, "comparison_id"))
        for name in ("weak_cells", "failure_patterns", "possible_causes", "unresolved_questions"):
            object.__setattr__(self, name, _refs(getattr(self, name), name))
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs, "evidence_refs", optional=False))

    @property
    def report_id(self) -> str:
        return content_id("ecologyrsi/diagnostic-report@1", self.to_dict())


@dataclass(frozen=True, slots=True)
class HypothesisProposal(_Contract):
    proposal_id: str
    parent_program_id: str
    problem: str
    hypothesis: str
    evidence_refs: tuple[str, ...]
    change_kind: str
    change_spec_ref: str
    expected_observation: str
    falsification_condition: str
    applicability_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("proposal_id", "parent_program_id", "problem", "hypothesis", "change_kind", "change_spec_ref", "expected_observation", "falsification_condition"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if "code" in self.change_spec_ref.lower() or "\n" in self.change_spec_ref:
            raise ValueError("unbounded_change_spec_forbidden")
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs, "evidence_refs", optional=False))
        object.__setattr__(self, "applicability_conditions", _refs(self.applicability_conditions, "applicability_conditions", optional=False))

    @property
    def canonical_id(self) -> str:
        return content_id("ecologyrsi/hypothesis@1", {
            "parent_program_id": self.parent_program_id, "problem": self.problem,
            "hypothesis": self.hypothesis, "evidence_refs": self.evidence_refs,
            "change_kind": self.change_kind, "change_spec_ref": self.change_spec_ref,
            "expected_observation": self.expected_observation,
            "falsification_condition": self.falsification_condition,
            "applicability_conditions": self.applicability_conditions,
        })
