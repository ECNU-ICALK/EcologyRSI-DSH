"""Safe, versioned capability registry for the external evolution plugin."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

from ..core.models import canonical_json, digest


_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*@[0-9]+$")
_KINDS = frozenset({"skill", "tool", "workflow", "algorithm"})
_FORBIDDEN_ARTIFACT_KEYS = frozenset({"code", "command", "shell", "python", "module", "import", "url"})


def _reject_executable_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = {str(key).lower() for key in value} & _FORBIDDEN_ARTIFACT_KEYS
        if forbidden:
            raise ValueError("generated executable artifacts require a reviewed Host adapter")
        for item in value.values():
            _reject_executable_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_executable_fields(item)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    value = value.strip()
    if name.endswith("_id") and _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must use a versioned id such as name@1")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    kind: str
    capability_id: str
    roles: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    parameters: Mapping[str, Mapping[str, Any]] = None  # type: ignore[assignment]
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unsupported capability kind: {self.kind}")
        _text(self.capability_id, "capability_id")
        roles = tuple(sorted({_text(item, "role") for item in self.roles}))
        domains = tuple(sorted({_text(item, "domain") for item in self.domains}))
        parameters = dict(self.parameters or {})
        metadata = dict(self.metadata or {})
        canonical_json(metadata)
        for name, contract in parameters.items():
            _text(name, "parameter name")
            if not isinstance(contract, Mapping):
                raise TypeError(f"parameter contract {name} must be an object")
            minimum = contract.get("minimum")
            maximum = contract.get("maximum")
            if minimum is not None and maximum is not None and _finite(minimum, name) > _finite(maximum, name):
                raise ValueError(f"parameter contract {name} has inverted bounds")
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "domains", domains)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "metadata", metadata)

    @property
    def digest(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "capability_id": self.capability_id,
            "roles": list(self.roles),
            "domains": list(self.domains),
            "parameters": {key: dict(value) for key, value in sorted(self.parameters.items())},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CapabilitySpec":
        if not isinstance(value, Mapping):
            raise TypeError("capability must be an object")
        return cls(
            kind=_text(value.get("kind"), "kind"),
            capability_id=_text(value.get("capability_id"), "capability_id"),
            roles=tuple(value.get("roles") or ()),
            domains=tuple(value.get("domains") or ()),
            parameters=value.get("parameters") or {},
            metadata=value.get("metadata") or {},
        )


@dataclass(frozen=True, slots=True)
class CapabilityProposal:
    kind: str
    capability_id: str
    roles: tuple[str, ...]
    artifact: Mapping[str, Any]
    status: str = "pending"

    def validate(self) -> "CapabilityProposal":
        if self.kind not in _KINDS:
            raise ValueError("proposal kind is not registered")
        _text(self.capability_id, "capability_id")
        if not self.roles:
            raise ValueError("proposal requires at least one role")
        if not isinstance(self.artifact, Mapping):
            raise TypeError("proposal artifact must be an object")
        _reject_executable_fields(self.artifact)
        canonical_json(dict(self.artifact))
        if self.status not in {"pending", "approved", "rejected"}:
            raise ValueError("invalid proposal status")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "kind": self.kind,
            "capability_id": self.capability_id,
            "roles": list(self.roles),
            "artifact": dict(self.artifact),
            "status": self.status,
        }


class CapabilityRegistry:
    def __init__(self, capabilities: list[CapabilitySpec] | tuple[CapabilitySpec, ...] = ()) -> None:
        self._items: dict[tuple[str, str], CapabilitySpec] = {}
        for item in capabilities:
            self.register(item)

    def register(self, capability: CapabilitySpec) -> CapabilitySpec:
        key = (capability.kind, capability.capability_id)
        if key in self._items and self._items[key].digest != capability.digest:
            raise ValueError(f"capability id is already registered with different content: {key}")
        self._items[key] = capability
        return capability

    def require(self, capability_id: str, kind: str, *, role: str | None = None, domain: str | None = None) -> CapabilitySpec:
        item = self._items.get((kind, capability_id))
        if item is None:
            raise ValueError(f"unknown {kind} capability: {capability_id}")
        if role and item.roles and role not in item.roles:
            raise ValueError(f"{capability_id} is not available to role {role}")
        if domain and item.domains and domain not in item.domains:
            raise ValueError(f"{capability_id} is not available to domain {domain}")
        return item

    def compatible(self, capability_id: str, kind: str, role: str, domain: str) -> bool:
        try:
            self.require(capability_id, kind, role=role, domain=domain)
        except (TypeError, ValueError):
            return False
        return True

    def validate_parameters(self, capability_id: str, values: Mapping[str, Any], *, kind: str = "skill") -> None:
        item = self.require(capability_id, kind)
        unknown = set(values) - set(item.parameters)
        if unknown:
            raise ValueError(f"unknown parameters for {capability_id}: {sorted(unknown)}")
        for name, value in values.items():
            contract = item.parameters[name]
            numeric = _finite(value, name)
            if contract.get("integer") and numeric != int(numeric):
                raise ValueError(f"parameter {name} must be an integer")
            if contract.get("minimum") is not None and numeric < _finite(contract["minimum"], name):
                raise ValueError(f"parameter {name} is below its minimum")
            if contract.get("maximum") is not None and numeric > _finite(contract["maximum"], name):
                raise ValueError(f"parameter {name} is above its maximum")

    def list(self, kind: str | None = None) -> tuple[CapabilitySpec, ...]:
        values = self._items.values() if kind is None else (item for (entry_kind, _), item in self._items.items() if entry_kind == kind)
        return tuple(sorted(values, key=lambda item: (item.kind, item.capability_id)))
