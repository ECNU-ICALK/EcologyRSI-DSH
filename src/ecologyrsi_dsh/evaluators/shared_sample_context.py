"""Frozen native prompt profile and candidate-independent stage identity."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from ..core.models import digest
ORIGIN_SHARED_CONTEXT_PROFILE = "origin_shared_context@1"
SIBLING_STAGE_CONTEXT_SCHEMA = "ecologyrsi-dsh.sibling-sample-stage-context/1"

def normalized_sample_planner_prompt_profile(
    value: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    """Validate the frozen opt-in profile without changing legacy runs."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("sample_planner_prompt_profile must be an object")
    if set(value) != {"version"}:
        raise ValueError(
            "sample_planner_prompt_profile must define exactly version"
        )
    version = value.get("version")
    if version != ORIGIN_SHARED_CONTEXT_PROFILE:
        raise ValueError(
            "sample_planner_prompt_profile.version must be "
            + ORIGIN_SHARED_CONTEXT_PROFILE
        )
    return {"version": ORIGIN_SHARED_CONTEXT_PROFILE}


def sibling_stage_context_digest(
    *,
    task_manifest_digest: str,
    generation: int,
    frozen_contract_digests: Mapping[str, str],
) -> str:
    """Digest the candidate-independent context shared by sibling genomes."""

    if not isinstance(task_manifest_digest, str) or len(task_manifest_digest) != 64:
        raise ValueError("task_manifest_digest must be a SHA-256 digest")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("generation must be a non-negative integer")
    if not isinstance(frozen_contract_digests, Mapping) or not frozen_contract_digests:
        raise ValueError("frozen_contract_digests must be a non-empty object")
    normalized: dict[str, str] = {}
    for name, value in frozen_contract_digests.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("sibling stage contract digests must be SHA-256 values")
        normalized[name] = value
    return digest(
        {
            "schema_version": SIBLING_STAGE_CONTEXT_SCHEMA,
            "task_manifest_digest": task_manifest_digest,
            "generation": generation,
            "frozen_contract_digests": {
                name: normalized[name] for name in sorted(normalized)
            },
        }
    )
