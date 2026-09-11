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
    from ..evolution.schedule import OptimizationSchedule
    raw_schedule = metadata.get("optimization_schedule")
    local_policy = metadata.get("local_comparison_policy")
    if local_policy not in (None, "exploratory_paired_point_comparison"):
        raise ValueError("unsupported local comparison policy")
    exploratory = local_policy == "exploratory_paired_point_comparison"
    if exploratory and (raw_schedule is None
            or not OptimizationSchedule.from_dict(raw_schedule).exploratory_local_comparison):
        raise ValueError("exploratory local comparison requires a training epoch schedule")
    runtime = metadata.get("host_runtime_build")
    positive_only = (
        not guarded
        and metadata.get("execution_protocol") == "dsh_native_plugin_evolution@1"
        and isinstance(runtime, Mapping)
        and runtime.get("evolution_runtime_schema") == "ecologyrsi-dsh.evolution-runtime/3"
    )
    profile = metadata.get("fitness_profile", {})
    return {
        "minimum_score_delta": (
            1e-12 if positive_only else profile.get("selection_minimum_score_delta", 0.005)
        ),
        # A *selection* tolerance on the per-cell skill delta against the
        # incumbent -- not the certification gate. Certification uses
        # `per_cell_noninferiority@1`, whose boundary is derived per cell from
        # the paired bootstrap, and `evaluators/generation_comparison.py` labels
        # its own check with `no_regression_scope` for the same reason.
        #
        # The 1e-12 fallback stays 1e-12 deliberately. Every live run carries a
        # fitness profile, so this only applies to the legacy runtime-v3
        # positive-delta path, and moving it would change what that path decided
        # on archived runs. This returned mapping is also splatted into
        # `validate_formal_batch_comparison(**policy)`, so it is a keyword
        # bundle, not a place to add descriptive keys -- the Agent-facing wording
        # belongs in the mutation contract catalog instead.
        "cell_regression_tolerance": profile.get("selection_cell_regression_tolerance", 1e-12),
        "cell_regression_blocks": not positive_only,
        # Local small-batch edits are exploratory. Complete paired execution,
        # practical gain and cell noninferiority still apply; the unchanged
        # epoch gate supplies the cross-day statistical confirmation.
        "require_paired_evidence": guarded and not exploratory,
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
