from __future__ import annotations

import unittest

from ecologyrsi_dsh.api.dsh_tools import (
    DshToolAdmissionClosedError,
    DshToolService,
)
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import digest


class DshCancelRaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = EventLedger()
        self.ledger.append("run-1", "RunCreated", {"test": True})
        self.service = DshToolService(self.ledger)

    def tearDown(self) -> None:
        self.ledger.close()

    def _identity(self) -> dict:
        return {
            "run_id": "run-1",
            "role": "researcher",
            "stage": "generation.research",
            "run_state_revision": 3,
            "stage_attempt": 1,
            "ledger_expected_revision": self.ledger.latest_seq(),
            "session_id": "child-1",
            "idempotency_key": "research-1",
            "child_reservation_id": "reservation-1",
            "activation_lease_id": "lease-1",
            "genome_digest": "a" * 64,
            "compiled_behavior_digest": "b" * 64,
            "phenotype_instance_digest": "c" * 64,
        }

    def test_closed_run_fence_rejects_late_structured_result_and_new_stage(self) -> None:
        self.service.open_admission("run-1", 3, 1)
        self.service.close_run_admissions("run-1")
        structured = {
            "schema_version": "ecology-research-result@1",
            "summary": "late",
            "evidence": [],
        }
        with self.assertRaises(DshToolAdmissionClosedError):
            self.service.accept_structured(
                {
                    "identity": self._identity(),
                    "output_schema_id": "ecology-research-result@1",
                    "structured": structured,
                    "result_digest": digest(structured),
                }
            )
        with self.assertRaises(DshToolAdmissionClosedError):
            self.service.open_admission("run-1", 4, 1)

        self.service.open_run_admissions("run-1")
        self.service.open_admission("run-1", 4, 1)


if __name__ == "__main__":
    unittest.main()
