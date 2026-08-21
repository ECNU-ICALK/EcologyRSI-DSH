"""Stable, credential-free identities for models bound to a run."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..core.models import digest


MODEL_BINDING_SCHEMA_VERSION = "ecologyrsi-dsh.model-binding/1"
HOST_PARAMETER_GENERATOR_ID = "host_parameter_generator@1"
RULE_JUDGE_ID = "rule_judge@1"

_MODEL_ROLE_ALIASES = {
    "strategy": "propose",
    "policy": "propose",
    "planner": "propose",
    "proposer": "propose",
    "review": "judge",
    "reviewer": "judge",
    "critic": "judge",
}
_MODEL_ROLES = frozenset({"propose", "judge"})

_BUILTIN_MODEL_SPECS: dict[str, dict[str, Any]] = {
    HOST_PARAMETER_GENERATOR_ID: {
        "implementation": "StrategyRouterDSHAdapter.bounded-parameter-generator/1",
        "roles": ["propose"],
    },
    RULE_JUDGE_ID: {
        "implementation": "EvaluatorRegistry.scientific-rule-judge/1",
        "roles": ["judge"],
    },
}


def builtin_model_configuration_digest(model_id: str) -> str | None:
    """Return the fixed implementation digest for a built-in model."""

    spec = _BUILTIN_MODEL_SPECS.get(model_id)
    if spec is None:
        return None
    return digest(
        {
            "schema_version": MODEL_BINDING_SCHEMA_VERSION,
            "model_id": model_id,
            **spec,
        }
    )


def canonical_model_role(value: Any) -> str:
    """Return the shared role name used by discovery, API, and execution."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("role must be a non-empty string")
    role = value.strip().casefold()
    role = _MODEL_ROLE_ALIASES.get(role, role)
    if role not in _MODEL_ROLES:
        raise ValueError(f"unsupported model role: {role}")
    return role


def canonical_model_roles(values: Iterable[Any]) -> tuple[str, ...]:
    """Normalize a non-empty role collection without silently widening it."""

    if isinstance(values, (str, bytes)):
        values = (values,)
    try:
        roles = tuple(canonical_model_role(item) for item in values)
    except TypeError as exc:
        raise ValueError("roles must be a non-empty array") from exc
    if not roles:
        raise ValueError("roles must be a non-empty array")
    if len(set(roles)) != len(roles):
        raise ValueError("model roles must not contain duplicates")
    return roles


def model_supports_role(model: Mapping[str, Any], role: str) -> bool:
    """Check canonical catalog roles; role-less legacy entries support both."""

    raw_roles = model.get("roles")
    if not isinstance(raw_roles, (list, tuple)) or not raw_roles:
        return True
    try:
        return canonical_model_role(role) in canonical_model_roles(raw_roles)
    except ValueError:
        return False


__all__ = [
    "HOST_PARAMETER_GENERATOR_ID",
    "MODEL_BINDING_SCHEMA_VERSION",
    "RULE_JUDGE_ID",
    "builtin_model_configuration_digest",
    "canonical_model_role",
    "canonical_model_roles",
    "model_supports_role",
]
