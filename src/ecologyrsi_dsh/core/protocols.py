"""Capability contracts for frozen sample-agent protocols."""

from __future__ import annotations


STRICT_ORIGIN_PROTOCOLS = frozenset(
    {
        "dsh-strict-origin-bundle@4",
    }
)
CONCURRENT_ORIGIN_PROTOCOL = "dsh-strict-origin-bundle@4"


def is_strict_origin_protocol(value: object) -> bool:
    return isinstance(value, str) and value in STRICT_ORIGIN_PROTOCOLS


def supports_concurrent_origins(value: object) -> bool:
    return value == CONCURRENT_ORIGIN_PROTOCOL


def supports_two_stage_screening(value: object) -> bool:
    return value == CONCURRENT_ORIGIN_PROTOCOL


def require_current_run_contract(task: object) -> None:
    """Reject unsupported run identities before creation or ledger replay."""
    from ..evolution.schedule import OptimizationSchedule
    from ..version import __version__
    from .model_preflight import AUDIT_METADATA_KEY, AUDIT_SCHEMA
    from .search_policy import SEARCH_GUARD_POLICY, PAIRED_EXECUTION_QUALIFICATION
    metadata = getattr(task, "metadata", {})
    if (metadata.get("execution_protocol") != "dsh_native_plugin_evolution@1"
            or metadata.get("optimization_protocol") != "joint_plugin_epoch@2"
            or getattr(task, "domain_pack", None) != "greenhouse_environment@1"
            or getattr(task, "candidates_per_generation", None) != 4
            or metadata.get("sample_agent_protocol") != CONCURRENT_ORIGIN_PROTOCOL):
        raise ValueError("only the current native joint_plugin_epoch@2 run contract is supported")
    if (metadata.get("require_model_contract_preflight") is not True
            or metadata.get(AUDIT_METADATA_KEY) != AUDIT_SCHEMA):
        raise ValueError("current native runs require the four-role model preflight audit")
    if (metadata.get("search_guard_policy") != SEARCH_GUARD_POLICY
            or metadata.get("paired_execution_qualification") != PAIRED_EXECUTION_QUALIFICATION
            or metadata.get("samples_per_update") is not None
            or metadata.get("local_comparison_policy") is not None):
        raise ValueError("current runs require the frozen guarded paired scientific policy")
    OptimizationSchedule.from_dict(metadata.get("optimization_schedule"))
    build = metadata.get("host_runtime_build", {})
    if (build.get("package_version") != __version__
            or build.get("evolution_runtime_schema") != "ecologyrsi-dsh.evolution-runtime/3"):
        raise ValueError("run requires the same package and runtime identity")
