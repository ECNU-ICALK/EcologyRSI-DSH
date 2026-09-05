from __future__ import annotations

import unittest

from ecologyrsi_dsh.api.contracts import build_command_receipt
from ecologyrsi_dsh.api.errors import ErrorCode, error_code_for_exception, error_payload
from ecologyrsi_dsh.core.ledger import CommandReceipt


class ApiContractsTests(unittest.TestCase):
    def test_draining_timeout_has_dedicated_error_code(self) -> None:
        class DrainingTimeoutError(TimeoutError):
            pass

        self.assertEqual(
            error_code_for_exception(DrainingTimeoutError("drain deadline")),
            ErrorCode.DRAINING_TIMEOUT,
        )

    def test_error_payload_has_stable_code_and_retryability(self) -> None:
        payload = error_payload(
            ErrorCode.PROVIDER_QUEUE_TIMEOUT,
            retryable=True,
            command_id="cmd-1",
        )
        self.assertEqual(payload["error_code"], "provider_queue_timeout")
        self.assertTrue(payload["retryable"])
        self.assertEqual(payload["command_id"], "cmd-1")
        self.assertIsInstance(payload["error"], str)

    def test_error_payload_omits_optional_command_id(self) -> None:
        payload = error_payload(ErrorCode.NOT_FOUND, retryable=False)
        self.assertEqual(payload, {
            "error": "resource not found",
            "error_code": "not_found",
            "retryable": False,
        })

    def test_command_receipt_builder_is_redacted_and_stable(self) -> None:
        receipt = CommandReceipt(
            command_key="cmd-1",
            run_id="run-1",
            resource_run_id="run-1",
            command_kind="control:pause",
            request_digest="digest",
            request={"token": "secret"},
            start_seq=4,
            status="pending",
            response={"status": "paused"},
            created_at="2026-09-05T00:00:00+00:00",
            completed_at=None,
        )
        payload = build_command_receipt(receipt)
        self.assertEqual(payload["command_id"], "cmd-1")
        self.assertEqual(payload["run_id"], "run-1")
        self.assertEqual(payload["status"], "pending")
        self.assertEqual(payload["response"], {"status": "paused"})
        self.assertNotIn("request", payload)


if __name__ == "__main__":
    unittest.main()
