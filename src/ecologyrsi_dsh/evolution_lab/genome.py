"""Immutable candidate plugin versions and bounded mutation application."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from ..core.models import canonical_json, digest
from .capabilities import CapabilityRegistry


_MUTATION_AXES = frozenset({"skill", "tool", "workflow", "algorithm"})


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    result = deepcopy(dict(value))
    canonical_json(result)
    return result


@dataclass(frozen=True, slots=True)
class Mutation:
    axis: str
    target: str
    value: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.axis not in _MUTATION_AXES:
            raise ValueError(f"unsupported mutation axis: {self.axis}")
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("mutation target must be a non-empty string")
        object.__setattr__(self, "value", _object(self.value, "mutation value"))

    def to_dict(self) -> dict[str, Any]:
        return {"axis": self.axis, "target": self.target, "value": deepcopy(dict(self.value))}


@dataclass(frozen=True, slots=True)
class PluginGenome:
    domain: str
    skills: Mapping[str, Mapping[str, Any]]
    tools: Mapping[str, tuple[str, ...]]
    workflow: Mapping[str, Any]
    algorithm: Mapping[str, Any]
    parent_digest: str | None = None
    mutation_axis: str | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValueError("genome domain must be non-empty")
        skills = {str(role): _object(value, f"skills.{role}") for role, value in self.skills.items()}
        tools = {str(role): tuple(sorted({str(tool) for tool in values})) for role, values in self.tools.items()}
        workflow = _object(self.workflow, "workflow")
        algorithm = _object(self.algorithm, "algorithm")
        if self.mutation_axis is not None and self.mutation_axis not in _MUTATION_AXES:
            raise ValueError("invalid genome mutation axis")
        object.__setattr__(self, "skills", skills)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "workflow", workflow)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "evidence_refs", tuple(sorted({str(item) for item in self.evidence_refs})))

    @property
    def digest(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ecologyrsi-dsh.plugin-genome/1",
            "domain": self.domain,
            "skills": {role: deepcopy(dict(value)) for role, value in sorted(self.skills.items())},
            "tools": {role: list(values) for role, values in sorted(self.tools.items())},
            "workflow": deepcopy(dict(self.workflow)),
            "algorithm": deepcopy(dict(self.algorithm)),
            "parent_digest": self.parent_digest,
            "mutation_axis": self.mutation_axis,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PluginGenome":
        if not isinstance(value, Mapping):
            raise TypeError("genome must be an object")
        return cls(
            domain=str(value.get("domain") or ""),
            skills=value.get("skills") or {},
            tools={role: tuple(items) for role, items in (value.get("tools") or {}).items()},
            workflow=value.get("workflow") or {},
            algorithm=value.get("algorithm") or {},
            parent_digest=value.get("parent_digest"),
            mutation_axis=value.get("mutation_axis"),
            evidence_refs=tuple(value.get("evidence_refs") or ()),
        )


class MutationPlanner:
    """Apply one registry-validated capability change to a parent Genome."""

    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

    def apply(self, parent: PluginGenome, mutation: Mutation) -> PluginGenome:
        value = deepcopy(dict(mutation.value))
        skills = deepcopy(dict(parent.skills))
        tools = {role: list(values) for role, values in parent.tools.items()}
        workflow = deepcopy(dict(parent.workflow))
        algorithm = deepcopy(dict(parent.algorithm))

        if mutation.axis == "skill":
            role = mutation.target
            capability_id = str(value.get("capability_id") or "")
            self.registry.require(capability_id, "skill", role=role, domain=parent.domain)
            parameters = value.get("parameters") or {}
            self.registry.validate_parameters(capability_id, parameters, kind="skill")
            skills[role] = {"capability_id": capability_id, "parameters": parameters}
        elif mutation.axis == "tool":
            role = mutation.target
            requested = tuple(sorted({str(item) for item in value.get("enabled") or ()}))
            if not requested:
                raise ValueError("tool mutation cannot disable every tool")
            for capability_id in requested:
                self.registry.require(capability_id, "tool", role=role, domain=parent.domain)
            tools[role] = list(requested)
        elif mutation.axis == "workflow":
            capability_id = str(value.get("capability_id") or workflow.get("capability_id") or "")
            self.registry.require(capability_id, "workflow", domain=parent.domain)
            parameters = value.get("parameters") or {}
            self.registry.validate_parameters(capability_id, parameters, kind="workflow")
            workflow = {"capability_id": capability_id, "parameters": parameters}
        elif mutation.axis == "algorithm":
            capability_id = str(value.get("capability_id") or algorithm.get("capability_id") or "")
            self.registry.require(capability_id, "algorithm", domain=parent.domain)
            parameters = value.get("parameters") or {}
            self.registry.validate_parameters(capability_id, parameters, kind="algorithm")
            algorithm = {"capability_id": capability_id, "parameters": parameters}

        child = PluginGenome(
            domain=parent.domain,
            skills=skills,
            tools={role: tuple(values) for role, values in tools.items()},
            workflow=workflow,
            algorithm=algorithm,
            parent_digest=parent.digest,
            mutation_axis=mutation.axis,
            evidence_refs=parent.evidence_refs,
        )
        if child.digest == parent.digest:
            raise ValueError("mutation does not change the parent Genome")
        return child
