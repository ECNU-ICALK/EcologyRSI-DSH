"""Small, redacted projections for asynchronous command receipts."""

from __future__ import annotations

from typing import Any

from ..core.ledger import CommandReceipt


def compact_command_receipt_payload(receipt: CommandReceipt) -> dict[str, Any]:
    """Project only durable command metadata; never include request assets."""

    return {
        "command_id": receipt.command_key,
        "status": receipt.status,
        "command_kind": receipt.command_kind,
        "run_id": receipt.resource_run_id or receipt.run_id,
        "created_at": receipt.created_at,
        "completed_at": receipt.completed_at,
    }


__all__ = ["compact_command_receipt_payload"]
