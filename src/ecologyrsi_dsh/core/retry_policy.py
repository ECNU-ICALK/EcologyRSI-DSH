"""Immutable public contracts for durable gateway retry decisions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class GatewayRetryPolicy:
    """The host-owned public values for one retry circuit."""

    circuit_code: str
    suggested_action: str
    public_reason: str
    retry_limit: int = 6


GATEWAY_RETRY_SCHEMA_VERSION = "ecologyrsi-dsh.gateway-retry-scheduled/2"
GATEWAY_RETRY_LIMIT = 6
GATEWAY_RETRY_EPOCH_SECONDS = 30 * 60
GATEWAY_RETRY_MAX_DELAY_SECONDS = 60 * 60
GATEWAY_RETRY_POLICIES: Mapping[str, GatewayRetryPolicy] = MappingProxyType(
    {
        "model_gateway": GatewayRetryPolicy(
            "gateway_retry_circuit_open",
            "check_gateway_then_resume",
            "模型网关暂时不可用，已安排有界延迟重试。",
        ),
        "dsh_native_runtime": GatewayRetryPolicy(
            "dsh_runtime_retry_circuit_open",
            "check_dsh_runtime_then_resume",
            "DSH 智能体运行时暂时不可用，已安排有界延迟重试。",
            retry_limit=12,
        ),
        "research_timeout": GatewayRetryPolicy(
            "research_timeout_retry_circuit_open",
            "check_gateway_then_resume",
            "研究阶段模型请求超时，已安排有界延迟重试。",
        ),
        "sample_result_persistence": GatewayRetryPolicy(
            "sample_persistence_retry_circuit_open",
            "check_persistence_then_resume",
            "样本结果持久化暂时不可用，已安排有界延迟重试。",
        ),
    }
)
GATEWAY_RETRY_CLASSES = frozenset(GATEWAY_RETRY_POLICIES)
GATEWAY_CIRCUIT_CODES = frozenset(
    policy.circuit_code for policy in GATEWAY_RETRY_POLICIES.values()
)


def retry_policy(retry_class: str) -> GatewayRetryPolicy:
    """Return the immutable public contract for a known retry class."""

    policy = GATEWAY_RETRY_POLICIES.get(retry_class)
    if policy is None:
        raise ValueError("unknown gateway retry class")
    return policy


__all__ = [
    "GATEWAY_CIRCUIT_CODES",
    "GATEWAY_RETRY_CLASSES",
    "GATEWAY_RETRY_EPOCH_SECONDS",
    "GATEWAY_RETRY_LIMIT",
    "GATEWAY_RETRY_MAX_DELAY_SECONDS",
    "GATEWAY_RETRY_POLICIES",
    "GATEWAY_RETRY_SCHEMA_VERSION",
    "GatewayRetryPolicy",
    "retry_policy",
]
