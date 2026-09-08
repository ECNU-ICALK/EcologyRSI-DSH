"""Concurrent finalist progress is a sum of distinct Host scoring scopes."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.api.projection import _adaptive_progress_projection
from ecologyrsi_dsh.core.models import CandidateRole


class ParallelFormalProgressTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.now = datetime.now(timezone.utc) - timedelta(minutes=1)
        self.state = NS(
            run=NS(run_id="run:parallel", generation=0, status=NS(value="running")),
            task_manifest=NS(max_generations=1, metadata={
                "optimization_protocol": "top2_adaptive_epoch@1",
                "optimization_schedule": {"screening_origin_count": 1,
                    "formal_origin_count_per_finalist": 10,
                    "selection_holdout_origin_count": 3, "local_batch_origin_count": 5},
                "prediction_cells_per_origin": 9,
            }),
            candidates=tuple(NS(candidate_id=c, generation=0, role=CandidateRole.SEARCH)
                             for c in ("a", "b")),
            candidate_screening_events=tuple(NS(payload={"generation": 0,
                "candidate_id": str(i), "origin_count": 1}) for i in range(4)),
            formal_batch_evaluations=(), holdout_evaluations=(), formal_batches=(), events=(),
        )

    def event(self, kind, payload):
        seq = len(self.events) + 1
        self.events.append(NS(seq=seq, kind=kind, payload=payload,
                              created_at=(self.now + timedelta(seconds=seq)).isoformat()))
        self.state.events = tuple(self.events)

    def start(self, candidate, *, result_revision=None, revision_id=None, scope=None, new_batch=True):
        result_revision = result_revision or f"results:{candidate}"
        revision_id = revision_id or f"revision:{candidate}"
        if new_batch:
            self.event("FormalBatchStarted", {"batch": {"generation": 0,
                "candidate_id": candidate, "revision_id": revision_id, "batch_index": 0,
                "cohort_digest": "cohort", "origin_count": 5}})
            self.state.formal_batches += (NS(generation=0, candidate_id=candidate, batch_index=0),)
        self.event("EvaluationSampleResultsStarted", {"candidate_id": candidate,
            "revision": result_revision, "checkpoint": {"evaluation_phase": "formal_batch",
                "formal_batch_index": 0, "holdout_arm": None,
                "candidate_revision_id": revision_id, "cohort_digest": "cohort",
                **({"execution_scope_digest": scope} if scope else {})}})

    def result(self, candidate, index, *, revision=None):
        self.event("EvaluationSampleResultBatchRecorded", {"run_id": self.state.run.run_id,
            "candidate_id": candidate, "revision": revision or f"results:{candidate}",
            "batch_index": index, "record_count": 9,
            "sample_ids": [f"{candidate}:origin:{index}:cell:{n}" for n in range(9)]})

    def heartbeat(self, candidate, good, bad, *, revision=None):
        self.event("EvaluationProgressRecorded", {"schema_version": "ecologyrsi-dsh.evaluation-progress/3",
            "candidate_id": candidate, "revision": revision or f"results:{candidate}",
            "role": "planner", "progress_id": len(self.events), "completed_samples": good + bad,
            "succeeded_samples": good, "failed_samples": bad, "total_samples": 5,
            "in_flight_batches": 0, "queued_batches": 0})

    def test_interleaved_lanes_count_once_across_resume_and_sealing(self):
        self.start("a")
        self.start("b")
        self.result("a", 1)
        self.heartbeat("a", 1, 0)
        self.result("b", 1)
        self.heartbeat("b", 0, 1)
        p = _adaptive_progress_projection(self.state)
        self.assertEqual((p["formal_completed_origins"], p["run_completed_origins"]), (2, 6))
        self.assertEqual((p["phase_succeeded_origins"], p["phase_failed_origins"]), (1, 1))
        self.assertEqual(p["phase_outcome_scope"], "formal_batch")

        self.event("RunPaused", {})
        self.event("EvaluationSampleResultsResumed", {"candidate_id": "a", "revision": "results:a"})
        self.result("a", 1)  # Repeated archive identity is not another origin.
        self.state.run.status.value = "paused"
        self.assertEqual(_adaptive_progress_projection(self.state)["formal_completed_origins"], 2)
        self.event("RunResumed", {})
        self.state.run.status.value = "running"
        self.result("a", 2)  # Durable result can precede its next heartbeat.
        p = _adaptive_progress_projection(self.state)
        self.assertEqual(p["formal_completed_origins"], 3)
        self.assertFalse(p["phase_outcomes_verified"])
        self.assertIsNone(p["phase_failed_origins"])
        self.heartbeat("a", 2, 0)
        p = _adaptive_progress_projection(self.state)
        self.assertEqual((p["phase_succeeded_origins"], p["phase_failed_origins"]), (2, 1))
        self.assertEqual(p["estimated_remaining_seconds"], p["run_estimated_remaining_seconds"])

        for index in (3, 4, 5):
            self.result("a", index)
        self.heartbeat("a", 5, 0)
        before = _adaptive_progress_projection(self.state)
        self.state.formal_batch_evaluations = (NS(scope=NS(generation=0, candidate_id="a",
            candidate_revision_id="revision:a", cohort_digest="cohort", batch_index=0, origin_count=5),
            metrics={"sample_execution": {"succeeded_origin_samples": 5, "failed_origin_samples": 0}}),)
        after = _adaptive_progress_projection(self.state)
        self.assertEqual(before["run_completed_origins"], after["run_completed_origins"])
        self.assertEqual(after["formal_completed_origins"], 6)
        self.assertEqual(len(after["formal_live_scopes"]), 1)

    def test_sealed_champion_does_not_hide_live_challenger_or_recount_old_revision(self):
        self.start("a", scope="scope:champion")
        self.state.formal_batch_evaluations = (NS(scope=NS(generation=0, candidate_id="a",
            candidate_revision_id="revision:a", cohort_digest="cohort", batch_index=0,
            origin_count=5, scope_key="scope:champion"),
            metrics={"sample_execution": {"succeeded_origin_samples": 4, "failed_origin_samples": 1}}),)
        self.start("a", result_revision="results:challenger", revision_id="revision:challenger",
                   scope="scope:challenger", new_batch=False)
        self.result("a", 1, revision="results:challenger")
        self.heartbeat("a", 1, 0, revision="results:challenger")
        self.result("a", 2, revision="results:a")  # Late old-revision evidence.
        p = _adaptive_progress_projection(self.state)
        self.assertEqual(p["formal_completed_origins"], 6)
        self.assertEqual((p["phase_succeeded_origins"], p["phase_failed_origins"]), (5, 1))
        self.assertEqual(p["formal_live_scopes"][0]["execution_scope_digest"], "scope:challenger")

    def test_repeated_archive_does_not_inflate_rolling_rate(self):
        self.start("a")
        self.start("b")
        self.result("a", 1)
        self.result("b", 1)
        self.result("a", 2)
        with patch("ecologyrsi_dsh.api.projection.datetime", wraps=datetime) as clock:
            clock.now.return_value = self.now + timedelta(minutes=1)
            before = _adaptive_progress_projection(self.state)
            self.result("a", 1)
            self.result("b", 1)
            after = _adaptive_progress_projection(self.state)
        self.assertEqual(before["samples_per_minute"], after["samples_per_minute"])
        self.assertEqual(before["run_completed_origins"], after["run_completed_origins"])
        self.assertEqual(before["run_estimated_remaining_seconds"], after["run_estimated_remaining_seconds"])

    def test_run_resume_excludes_old_sealed_lane_from_rate_window(self):
        self.start("a")
        self.start("b")
        self.result("a", 1)
        self.result("a", 2)
        self.event("RunResumed", {})
        # Lane b resumes while lane a's prior results remain durable progress.
        self.event("EvaluationSampleResultsResumed", {"candidate_id": "b", "revision": "results:b"})
        self.result("b", 1)
        first_new = self.events[-1].created_at
        self.result("b", 2)
        self.result("b", 3)
        fixed_now = self.now + timedelta(minutes=1)
        with patch("ecologyrsi_dsh.api.projection.datetime", wraps=datetime) as clock:
            clock.now.return_value = fixed_now
            progress = _adaptive_progress_projection(self.state)
        expected = round(2 / ((fixed_now - datetime.fromisoformat(first_new)).total_seconds() / 60), 3)
        self.assertEqual(progress["samples_per_minute"], expected)
        self.assertEqual(progress["formal_completed_origins"], 5)

    def test_completed_rate_is_bound_to_completion_not_the_read_clock(self):
        self.start("a")
        self.result("a", 1)
        self.result("a", 2)
        self.event("RunCompleted", {})
        self.state.run.status.value = "completed"
        with patch("ecologyrsi_dsh.api.projection.datetime", wraps=datetime) as clock:
            clock.now.return_value = self.now + timedelta(minutes=1)
            before = _adaptive_progress_projection(self.state)
            clock.now.return_value = self.now + timedelta(days=1)
            after = _adaptive_progress_projection(self.state)
        self.assertGreater(before["samples_per_minute"], 0)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
