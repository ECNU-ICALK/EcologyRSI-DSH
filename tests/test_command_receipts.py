from __future__ import annotations

import unittest

from ecologyrsi_dsh.api.command_receipts import compact_command_receipt_payload
from ecologyrsi_dsh.core.ledger import CommandReceipt


class CommandReceiptProjectionTests(unittest.TestCase):
    def test_compact_projection_has_no_request_or_training_asset(self) -> None:
        receipt = CommandReceipt(
            command_key="cmd-1",
            run_id="run-1",
            resource_run_id="run-1",
            command_kind="control:resume",
            request_digest="a" * 64,
            request={"include_training_asset": True},
            start_seq=4,
            status="pending",
            response=None,
            created_at="2026-09-05T00:00:00Z",
            completed_at=None,
        )
        self.assertEqual(
            compact_command_receipt_payload(receipt),
            {
                "command_id": "cmd-1",
                "status": "pending",
                "command_kind": "control:resume",
                "run_id": "run-1",
                "created_at": "2026-09-05T00:00:00Z",
                "completed_at": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
