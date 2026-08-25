from __future__ import annotations

import unittest

from ecologyrsi_dsh.api.dsh_tools import (
    DshToolAdmissionClosedError,
    DshToolService,
)
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import digest


def _research_skill_evidence() -> dict:
    return {
        "schema_version": "ecologyrsi-dsh.skill-invocation-evidence/1",
        "stage": "generation.research",
        "skill_name": "autonomous-ecology-research",
        "call_count": 1,
        "successful_call_count": 1,
        "call_seq": 1,
        "result_seq": 2,
        "first_tool_call_verified": True,
        "next_tool_name": "structured_output",
        "next_tool_call_seq": 3,
        "order_verified": True,
        "source": "dsh_session_event_log",
    }


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
        fence = self.service.open_admission(
            "run-1",
            3,
            1,
            role="researcher",
            stage="generation.research",
            idempotency_key="research-1",
        )
        self.service.allocate_child_reservation(
            {
                "request_id": "cancel-race-reservation",
                "run_id": "run-1",
                "parent_session_id": "research-host",
                "role": "researcher",
                "stage": "generation.research",
                "run_state_revision": 3,
                "stage_attempt": 1,
                "admission_id": fence.admission_id,
                "timeout_ms": 1_000,
                "item_digest": "d" * 64,
                "idempotency_key": "research-1",
            }
        )
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
                    "skill_invocation_evidence": _research_skill_evidence(),
                    "admission_id": fence.admission_id,
                }
            )
        with self.assertRaises(DshToolAdmissionClosedError):
            self.service.open_admission("run-1", 4, 1)

        self.service.open_run_admissions("run-1")
        self.service.open_admission("run-1", 4, 1)


if __name__ == "__main__":
    unittest.main()
