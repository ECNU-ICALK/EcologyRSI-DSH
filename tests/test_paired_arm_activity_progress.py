import unittest
from types import SimpleNamespace as NS

from ecologyrsi_dsh.api.projection import _formal_batch_live_projection


def event(seq, kind, payload):
    return NS(seq=seq, kind=kind, payload=payload, created_at=f"event-{seq}")


class PairedArmActivityProgressTests(unittest.TestCase):
    def setup_pair(self, same_revision=False):
        revision = "parent" if same_revision else "child"
        batch = {"generation": 0, "candidate_id": "candidate", "batch_index": 1,
                 "revision_id": revision, "cohort_digest": "cohort", "origin_count": 2}
        sealed_parent = NS(scope=NS(generation=0, candidate_id="candidate", batch_index=1,
            cohort_digest="cohort", candidate_revision_id="parent", scope_key="scope:parent"))
        old_launch = {"stage": "sample.plan", "idempotency_key": "parent-request",
                      "reservation_id": "parent-reservation", "sample_member_digests": ["a" * 64]}
        new_launch = {**old_launch, "idempotency_key": "child-request", "reservation_id": "child-reservation"}
        checkpoint = {"evaluation_phase": "formal_batch", "formal_batch_index": 1,
            "holdout_arm": None, "candidate_revision_id": revision,
            "cohort_digest": "cohort", "execution_scope_digest": "scope:child"}
        events = [event(1, "FormalBatchStarted", {"batch": batch}),
            event(2, "DshChildLaunchReserved", {"launch": old_launch}),
            event(3, "DshStructuredResultAccepted", {"identity": {
                "stage": "sample.plan", "idempotency_key": "parent-request",
                "child_reservation_id": "parent-reservation"}}),
            event(4, "EvaluationSampleResultsStarted", {"candidate_id": "candidate",
                "revision": "child-results", "checkpoint": checkpoint}),
            event(5, "DshChildLaunchReserved", {"launch": new_launch})]
        return NS(events=tuple(events), run=NS(generation=0, status=NS(value="running")),
            task_manifest=NS(metadata={"prediction_cells_per_origin": 1, "sample_concurrency": 4}),
            formal_batch_evaluations=(sealed_parent,))

    def test_sealed_parent_keeps_child_live_without_importing_parent_receipts(self):
        for same_revision in (False, True):
            with self.subTest(same_revision=same_revision):
                result = _formal_batch_live_projection(self.setup_pair(same_revision))
                self.assertIsNotNone(result)
                self.assertEqual(result["in_flight_batches"], 1)
                self.assertEqual(result["gateway_request_count"], 1)
                self.assertEqual(result["remote_completed_origins"], 0)
                self.assertEqual(result["completed_origins"], 0)

    def test_child_heartbeat_is_scoped_and_late_parent_receipt_does_not_settle_it(self):
        state = self.setup_pair()
        state.events += (event(6, "DshStructuredResultAccepted", {"identity": {
            "stage": "sample.plan", "idempotency_key": "parent-request",
            "child_reservation_id": "parent-reservation"}}),
            event(7, "EvaluationProgressRecorded", {"schema_version": "ecologyrsi-dsh.evaluation-progress/3",
                "candidate_id": "candidate", "revision": "child-results", "role": "planner",
                "progress_id": 1, "completed_samples": 1, "succeeded_samples": 1,
                "failed_samples": 0, "total_samples": 2, "in_flight_batches": 1, "queued_batches": 0}))
        result = _formal_batch_live_projection(state)
        self.assertEqual(result["remote_completed_origins"], 0)
        self.assertEqual(result["completed_origins"], 1)
        self.assertEqual(result["succeeded_samples"], 1)

    def test_sealed_child_and_paused_run_do_not_claim_live_activity(self):
        state = self.setup_pair()
        state.formal_batch_evaluations += (NS(scope=NS(generation=0, candidate_id="candidate",
            batch_index=1, cohort_digest="cohort", candidate_revision_id="child", scope_key="scope:child")),)
        self.assertIsNone(_formal_batch_live_projection(state))
        state = self.setup_pair()
        state.run.status.value = "paused"
        self.assertIsNone(_formal_batch_live_projection(state))


if __name__ == "__main__":
    unittest.main()
