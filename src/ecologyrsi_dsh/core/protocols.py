"""Capability contracts for frozen sample-agent protocols."""

from __future__ import annotations


STRICT_ORIGIN_PROTOCOLS = frozenset(
    {
        "dsh-strict-origin-bundle@3",
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
