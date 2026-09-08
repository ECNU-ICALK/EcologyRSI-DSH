"""Frozen, bounded execution policy for newly created research runs."""
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

# Shared by run creation and the native adapter. Multi-tool, nine-cell Agent
# inference needs the same output room in both paths; never keep API-local
# defaults that silently override the adapter's tested execution contract.
NATIVE_SAMPLE_OPERATION_MAX_TOKENS = MappingProxyType({
    "sample.planner": 8192,
    "sample.repair": 8192,
    "sample.critic": 4096,
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
    """Read a frozen policy without rewriting its recorded format version.

    Replay must remain readable when a new run changes a writing policy. New
    creation uses RESEARCH_EXECUTION_POLICY; native binding checks decide
    whether an older run can execute against the installed runtime.
    """
    if RESEARCH_EXECUTION_POLICY_KEY not in metadata:
        return None
    value = metadata[RESEARCH_EXECUTION_POLICY_KEY]
    if (not isinstance(value, Mapping)
            or set(value) != set(RESEARCH_EXECUTION_POLICY)
            or any(type(value[key]) is not type(expected) or (
                       value[key] not in {"concise@1", "concise@2"}
                       if key == "synthesis_report_format" else value[key] != expected)
                   for key, expected in RESEARCH_EXECUTION_POLICY.items())):
        raise ValueError("invalid frozen research execution policy")
    return dict(value)
