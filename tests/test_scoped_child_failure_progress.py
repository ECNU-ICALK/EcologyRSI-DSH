import unittest
from types import SimpleNamespace

from ecologyrsi_dsh.api.projection import _origin_live_projection_after


def event(seq, kind, payload):
    return SimpleNamespace(seq=seq, kind=kind, payload=payload, created_at=f"event-{seq}")


def failed(seq, stage, code="structured_child_output_budget_exhausted"):
    return event(seq, "DshChildExecutionFailed", {"identity": {
        "stage": stage, "child_reservation_id": f"failed-{seq}"}, "error_code": code})


class ScopedChildFailureProgressTests(unittest.TestCase):
    def project(self, failures, accepted=False):
        started = event(10, "FormalBatchStarted", {})
        launch = event(11, "DshChildLaunchReserved", {"launch": {
            "stage": "sample.plan", "idempotency_key": "origin-1", "reservation_id": "plan-1",
            "sample_member_digests": ["a" * 64]}})
        events = [failed(9, "sample.critic"), started, launch]
        if accepted:
            events.append(event(12, "DshStructuredResultAccepted", {"identity": {
                "stage": "sample.plan", "idempotency_key": "origin-1", "child_reservation_id": "plan-1"}}))
        events.extend(failures)
        state = SimpleNamespace(events=tuple(events), task_manifest=SimpleNamespace(metadata={
            "prediction_cells_per_origin": 1, "sample_concurrency": 4}))
        return _origin_live_projection_after(state, started, 2)

    def test_failed_review_is_visible_without_completing_active_prediction(self):
        result = self.project([failed(13, "sample.critic")])
        self.assertEqual(result["child_execution_failed_request_count"], 1)
        self.assertEqual(result["remote_completed_origins"], 0)
        self.assertEqual(result["in_flight_batches"], 1)
        self.assertEqual(result["gateway_request_count"], 1)

    def test_review_failure_does_not_erase_accepted_prediction_or_count_old_failures(self):
        result = self.project([failed(13, "sample.critic", "structured_child_model_error")], accepted=True)
        self.assertEqual(result["child_execution_failed_request_count"], 1)
        self.assertEqual(result["structured_child_model_error_count"], 1)
        self.assertEqual(result["remote_completed_origins"], 1)
        self.assertEqual(result["in_flight_batches"], 0)

    def test_unrelated_stage_failure_remains_excluded(self):
        result = self.project([failed(13, "generation.reflect"), failed(14, "candidate.local_edit")])
        self.assertEqual(result["child_execution_failed_request_count"], 0)
        self.assertEqual(result["event_seq"], 11)


if __name__ == "__main__":
    unittest.main()
