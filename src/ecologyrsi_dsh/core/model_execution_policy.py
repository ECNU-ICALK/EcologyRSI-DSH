"""Frozen, bounded execution policy for newly created research runs."""
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

# The ceiling every sample operation budget is validated against. A nine-cell
# planner response carries prediction-tool arguments, reasoning and the bounded
# decision object in one output budget, so it is bounded above the single-answer
# stages, exactly as research synthesis already is.
SAMPLE_OPERATION_MIN_MAX_TOKENS = 512
SAMPLE_OPERATION_MAX_MAX_TOKENS = 32768

# Shared by run creation and the native adapter. Nine-cell planner and repair
# responses need room for tool arguments and the final structured decision.
# This bounds a single response; it is not a cumulative run quota.
NATIVE_SAMPLE_OPERATION_MAX_TOKENS = MappingProxyType({
    "sample.planner": 32768,
    "sample.repair": 32768,
    "sample.critic": 8192,
})

RESEARCH_EXECUTION_POLICY_KEY = "research_execution_policy"
RESEARCH_EXECUTION_POLICY = MappingProxyType({
    "schema_version": "ecologyrsi-dsh.research-execution-policy/1",
    "synthesis_context_format": "compact@1",
    "synthesis_report_format": "concise@2",
    "synthesis_max_output_tokens": 16384,
    "retry_identical_exhausted_request": False,
})


def research_execution_policy(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read and validate the current frozen research policy."""
    if RESEARCH_EXECUTION_POLICY_KEY not in metadata:
        return None
    value = metadata[RESEARCH_EXECUTION_POLICY_KEY]
    if (not isinstance(value, Mapping)
            or set(value) != set(RESEARCH_EXECUTION_POLICY)
            or any(type(value[key]) is not type(expected) or value[key] != expected
                   for key, expected in RESEARCH_EXECUTION_POLICY.items())):
        raise ValueError("invalid frozen research execution policy")
    return dict(value)


LOCAL_EDIT_CONTEXT_POLICY = "cohort_scoped_edit_evidence@2"
