"""The frozen scientific policy for current joint evolution runs."""
from collections.abc import Mapping
from typing import Any

SEARCH_GUARD_POLICY = "practical_delta_cell_noninferiority_paired_blocks@1"
LOCAL_PAIRED_BLOCK_MINIMUM = 3
PAIRED_EXECUTION_QUALIFICATION = "paired_strict_agent_chain@1"


def guarded_search(metadata: Mapping[str, Any]) -> bool:
    if metadata.get("search_guard_policy") != SEARCH_GUARD_POLICY:
        raise ValueError(f"search_guard_policy must be {SEARCH_GUARD_POLICY}")
    return True


def local_challenger_policy(metadata: Mapping[str, Any]) -> dict[str, Any]:
    guarded_search(metadata)
    if metadata.get("local_comparison_policy") is not None:
        raise ValueError("current runs require the paired scientific comparison")
    profile = metadata.get("fitness_profile", {})
    return {
        "minimum_score_delta": profile.get("selection_minimum_score_delta", 0.005),
        # This is the local selection tolerance. Independent certification
        # retains its own paired-bootstrap per-cell gate.
        "cell_regression_tolerance": profile.get("selection_cell_regression_tolerance", 1e-12),
        "cell_regression_blocks": True,
        "require_paired_evidence": True,
        "require_paired_strict_chain": True,
    }


def paired_execution_qualification_required(
    metadata: Mapping[str, Any], value: Any, *, new_decision: bool = False,
) -> bool:
    """Require the same qualification for writes and same-version replay."""
    guarded_search(metadata)
    if value != PAIRED_EXECUTION_QUALIFICATION:
        raise ValueError("guarded comparison requires paired execution qualification")
    return True
