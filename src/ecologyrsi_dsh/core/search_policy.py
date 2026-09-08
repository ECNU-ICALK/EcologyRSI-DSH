"""Immutable opt-in search guard. Absent metadata preserves historical decisions."""

from collections.abc import Mapping
from typing import Any


SEARCH_GUARD_POLICY = "practical_delta_cell_noninferiority_paired_blocks@1"
LOCAL_PAIRED_BLOCK_MINIMUM = 3
PAIRED_EXECUTION_QUALIFICATION = "paired_strict_agent_chain@1"


def guarded_search(metadata: Mapping[str, Any]) -> bool:
    value = metadata.get("search_guard_policy")
    if value is not None and value != SEARCH_GUARD_POLICY:
        raise ValueError(f"search_guard_policy must be {SEARCH_GUARD_POLICY}")
    return value == SEARCH_GUARD_POLICY


def local_challenger_policy(metadata: Mapping[str, Any]) -> dict[str, Any]:
    guarded = guarded_search(metadata)
    runtime = metadata.get("host_runtime_build")
    positive_only = (
        not guarded
        and metadata.get("execution_protocol") == "dsh_native_plugin_evolution@1"
        and isinstance(runtime, Mapping)
        and runtime.get("evolution_runtime_schema") == "ecologyrsi-dsh.evolution-runtime/3"
    )
    return {
        "minimum_score_delta": 1e-12 if positive_only else 0.005,
        "cell_regression_blocks": not positive_only,
        "require_paired_evidence": guarded,
        "require_paired_strict_chain": guarded,
    }


def paired_execution_qualification_required(
    metadata: Mapping[str, Any], value: Any, *, new_decision: bool = False
) -> bool:
    """Version decision evidence without changing a frozen task's old replay.

    Writers must mark new guarded decisions. Readers preserve unmarked history;
    a marked decision always binds the exact qualification contract.
    """
    guarded = guarded_search(metadata)
    if value is not None and value != PAIRED_EXECUTION_QUALIFICATION:
        raise ValueError("unsupported paired execution qualification contract")
    if value is not None and not guarded:
        raise ValueError("paired execution qualification requires guarded search")
    if new_decision and guarded and value != PAIRED_EXECUTION_QUALIFICATION:
        raise ValueError("new guarded comparison requires paired execution qualification")
    return value == PAIRED_EXECUTION_QUALIFICATION
