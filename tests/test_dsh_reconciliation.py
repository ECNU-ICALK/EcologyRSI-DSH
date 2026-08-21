from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ecologyrsi_dsh.api.dsh_tools import DshToolService
from ecologyrsi_dsh.core.ledger import EventLedger


def _request(request_id: str) -> dict[str, str]:
    return {
        "request_id": request_id,
        "run_id": "run-reconcile",
        "parent_session_id": "role-host-session",
        "role": "candidate-proposer",
        "stage": "candidate.propose",
        "item_digest": "a" * 64,
        "idempotency_key": "candidate-1",
    }


class DshReconciliationTests(unittest.TestCase):
    def test_launch_attempt_is_durable_and_never_reused_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.sqlite3"
            ledger = EventLedger(path)
            ledger.append("run-reconcile", "RunCreated", {"legacy": True})
            first_service = DshToolService(ledger)

            first = first_service.allocate_child_reservation(_request("request-1"))
            exact_retry = first_service.allocate_child_reservation(_request("request-1"))
            second = first_service.allocate_child_reservation(_request("request-2"))
            self.assertEqual(first, exact_retry)
            self.assertEqual(first["launch"]["launch_attempt"], 1)
            self.assertEqual(second["launch"]["launch_attempt"], 2)
            self.assertNotEqual(
                first["launch"]["reservation_id"],
                second["launch"]["reservation_id"],
            )
            ledger.close()

            recovered_ledger = EventLedger(path)
            recovered_service = DshToolService(recovered_ledger)
            third = recovered_service.allocate_child_reservation(_request("request-3"))
            self.assertEqual(third["launch"]["launch_attempt"], 3)
            self.assertNotIn(
                third["launch"]["reservation_id"],
                {
                    first["launch"]["reservation_id"],
                    second["launch"]["reservation_id"],
                },
            )
            recovered_ledger.close()

    def test_request_id_cannot_be_reused_for_a_different_child(self) -> None:
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        ledger.append("run-reconcile", "RunCreated", {"legacy": True})
        service = DshToolService(ledger)
        service.allocate_child_reservation(_request("request-1"))
        changed = _request("request-1")
        changed["item_digest"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "reused with different input"):
            service.allocate_child_reservation(changed)


if __name__ == "__main__":
    unittest.main()
