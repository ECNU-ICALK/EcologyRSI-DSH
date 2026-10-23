"""One bounded online-retrieval policy for bootstrap and Agent searches."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


RETRIEVAL_POLICY_LIMITS = {
    "dynamic_max_calls_per_stage": (8, 24),
    "max_queries_per_call": (8, 12),
    "max_query_chars": (320, 600),
    "max_results": (16, 32),
}
DEFAULT_RETRIEVAL_POLICY = {name: limits[0] for name, limits in RETRIEVAL_POLICY_LIMITS.items()}


def validate_retrieval_policy(value: Any) -> dict[str, int]:
    """Validate a partial user policy without silently accepting misspellings."""
    if not isinstance(value, Mapping) or set(value) - set(RETRIEVAL_POLICY_LIMITS):
        raise ValueError("retrieval_policy must contain only supported policy fields")
    result: dict[str, int] = {}
    for name, item in value.items():
        maximum = RETRIEVAL_POLICY_LIMITS[name][1]
        if isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= maximum:
            raise ValueError(f"retrieval_policy.{name} must be an integer between 1 and {maximum}")
        result[name] = item
    return result


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    online_enabled: bool = False
    dynamic_max_calls_per_stage: int = 8
    max_queries_per_call: int = 8
    max_query_chars: int = 320
    max_results: int = 16

    def to_dict(self) -> dict[str, bool | int]:
        return asdict(self)


def resolve_retrieval_policy(
    metadata: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> RetrievalPolicy:
    """Resolve the shared switch; the environment remains an operator override."""
    configured = metadata.get("knowledge_online_enabled", False)
    online = configured if isinstance(configured, bool) else False
    environment = (os.environ if environ is None else environ).get("ECOLOGYRSI_KNOWLEDGE_ONLINE")
    if environment is not None:
        online = environment.strip().casefold() in {"1", "true", "yes", "on"}
    policy = validate_retrieval_policy(metadata.get("retrieval_policy", {}))
    return RetrievalPolicy(online_enabled=online, **policy)
