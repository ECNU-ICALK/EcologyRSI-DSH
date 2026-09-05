"""Small, redacted projections for asynchronous command receipts."""

from __future__ import annotations

from typing import Any

from ..core.ledger import CommandReceipt
from .contracts import build_command_receipt


def compact_command_receipt_payload(receipt: CommandReceipt) -> dict[str, Any]:
    """Project only durable command metadata; never include request assets."""

    return build_command_receipt(receipt)


__all__ = ["compact_command_receipt_payload"]
