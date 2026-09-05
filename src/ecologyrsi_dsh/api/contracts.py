"""Versioned builders for public API response contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.ledger import CommandReceipt


def build_command_receipt(
    receipt: CommandReceipt,
    *,
    response: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Serialize durable command metadata without request or secret material."""

    payload: dict[str, object] = {
        "command_id": receipt.command_key,
        "status": receipt.status,
        "command_kind": receipt.command_kind,
        "run_id": receipt.resource_run_id or receipt.run_id,
        "created_at": receipt.created_at,
        "completed_at": receipt.completed_at,
    }
    selected = response if response is not None else receipt.response
    if selected is not None:
        payload["response"] = dict(selected)
    return payload


__all__ = ["build_command_receipt"]
