"""Frozen, bounded execution policy for newly created research runs."""
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

# The ceiling every sample operation budget is validated against. A nine-cell
# planner response carries prediction-tool arguments, reasoning and the bounded
# decision object in one output budget, so it is bounded above the single-answer
# stages, exactly as research synthesis already is.
SAMPLE_OPERATION_MIN_MAX_TOKENS = 512
SAMPLE_OPERATION_MAX_MAX_TOKENS = 16384

# Shared by run creation and the native adapter. Multi-tool, nine-cell Agent
# inference needs the same output room in both paths; never keep API-local
# defaults that silently override the adapter's tested execution contract.
#
# 8192 was measured as too tight: in run:010d5ca2-84a2-4955-9eeb-ee4d82446d49
# one sample.plan child ended at turn/end max-tokens on exactly 8192 output
# tokens without reaching structured_output, failing the whole run, while its
# eleven successful siblings spent 4087..6347. The planner and the repair that
# replays its prompt share the raised budget.
#
# The critic's own 4096 was measured as too tight the same way: in
# run:472dc541-8b45-4150-8328-6addf8ae5e48 a sample.critic child on
# newapi/glm-5.2 stopped at length on exactly 4096 output tokens holding one
# unfinished reasoning block and no structured_output, while its siblings
# submitted at 1277 and 3080. On routes whose model metadata advertises no
# reasoning tier the runtime cannot request "off", so review reasoning shares
# this budget with the decision object it must still emit; 8192 leaves the
# largest observed submission its measured room instead of 1016 tokens.
NATIVE_SAMPLE_OPERATION_MAX_TOKENS = MappingProxyType({
    "sample.planner": 16384,
    "sample.repair": 16384,
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
