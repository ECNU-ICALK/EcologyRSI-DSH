"""Quality-preserving runtime fixes for the September 22 live-run review."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import patch
import time

from ecologyrsi_dsh import EventLedger, TaskManifest
from ecologyrsi_dsh.application.runtime import ApplicationRuntime
from ecologyrsi_dsh.application.formal_trajectory import (
    _check_behavior_revisit, _local_edit_policy_rejection_reason,
    _local_edit_proposal, _trajectory_score_history,
)
from ecologyrsi_dsh.api.projection import (
    _adaptive_progress_projection, _dsh_runtime_projection,
    _host_activity_projection, _runtime_failure_projection,
)
from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
from ecologyrsi_dsh.core.model_execution_policy import LOCAL_EDIT_CONTEXT_POLICY, NATIVE_SAMPLE_OPERATION_MAX_TOKENS
from ecologyrsi_dsh.core.trajectory import LocalEditOutcome
from ecologyrsi_dsh.evolution.local_edits import LocalEditProposal, LocalEditResult
from ecologyrsi_dsh.execution.host_activity import HostActivityMonitor, validate_host_interruption
from ecologyrsi_dsh.execution.sample_admission import RunSampleAdmission
from tests import test_formal_trajectory as trajectory_fixtures
from tests.test_local_edits import _parent, _context


def policy_state(**kwargs):
    return NS(task_manifest=NS(metadata={"local_edit_context_policy": LOCAL_EDIT_CONTEXT_POLICY}), **kwargs)


def proposal(revisit=None):
    return LocalEditProposal(decision="mutate", operations=({"op": "set_bounded_parameter", "name": "history_steps", "value": 8},),
        evidence_refs=("metric:overall",), expected_effect_cells=("air_temperature@1h",), risk_cells=(), revisit=revisit)


def event(seq, kind, payload, at=None):
    return NS(seq=seq, kind=kind, payload=payload, created_at=(at or datetime.now(timezone.utc)).isoformat())


class EditorEvidenceTests(TestCase):
    def test_coverage_does_not_make_different_cohorts_comparable(self):
        state = trajectory_fixtures.FormalTrajectoryTests._scored_state({0: .1, 1: -.2, 2: .8})
        state.task_manifest = policy_state().task_manifest
        result = _trajectory_score_history(state, "candidate:regress", 2)
        self.assertIsNone(result["best_observed"])
        self.assertTrue(all(r["coverage_complete"] and not r["comparable"] for r in result["score_history"]))
        state.task_manifest.metadata.clear()
        self.assertEqual(_trajectory_score_history(state, "candidate:regress", 2)["best_observed"]["score"], .8)

    def test_native_editor_receives_the_frozen_evidence_policy(self):
        parent = _parent()
        context = _context(parent)
        revision = NS(revision_id=context.candidate_revision_id, genome=parent.to_dict(),
                      candidate_id=context.candidate_id, behavior_digest=parent.behavior_digest)
        evaluation = NS(score=.1, passed=False, metrics={"sample_execution": {"coverage_pass": True}}, scope=NS(scope_key="scope"))
        state = policy_state(run=NS(run_id=context.run_id), events=(NS(seq=1),),
            candidate_identity_binding=lambda _: {}, batch_evaluation_for=lambda *_: evaluation,
            revision=lambda _: revision, formal_batches=(), local_edit_outcomes=(),
            local_edit_proposal_for=lambda *_: None, candidate_revisions=(revision,), batch_comparison_for=lambda *_: None)
        services = NS(dsh_native_runtime=object(), ledger=NS(latest_seq=lambda: 1))
        answer = LocalEditProposal("keep", (), context.allowed_evidence_refs[:1], (), ())
        with patch("ecologyrsi_dsh.application.formal_trajectory.DshStructuredRoleRuntime") as runtime:
            runtime.return_value.run.return_value = answer.to_dict()
            self.assertEqual(_local_edit_proposal(services, state, NS(candidate_id=context.candidate_id), NS(batch_index=0), context), answer)
            sent = runtime.return_value.run.call_args.kwargs["context"]
        self.assertFalse(sent["decision_policy"]["host_rolls_back_a_clear_score_regression"])
        self.assertFalse(sent["decision_policy"]["cross_batch_scores_measure_edit_effect"])
        self.assertIsNone(sent["best_observed"])
        self.assertEqual(sent["recent_behavior_states"][0]["behavior_digest"], parent.behavior_digest)
        self.assertEqual(sent["batch_evidence"]["score"], .1)

    def test_revisit_requires_reason_but_crash_replay_does_not(self):
        child = _parent()
        result = LocalEditResult(LocalEditOutcome.APPLIED, (), child, "a" * 64)
        prior = NS(candidate_id="c", behavior_digest=child.behavior_digest, source_batch_index=None)
        state = policy_state(candidate_revisions=(prior,))
        check = lambda p: _check_behavior_revisit(state, "c", p, result, batch_index=3)
        self.assertEqual(check(proposal()).rejection_reason, "repeated_behavior_requires_revisit_reason")
        for reason in ("safety_recovery", "new_batch_evidence", "paired_recheck"):
            self.assertIs(check(proposal({"reason": reason, "justification": "The supplied evidence supports this revisit."})), result)
        prior.source_batch_index = 3  # Child durable; decision not yet committed.
        self.assertIs(check(proposal()), result)
        prior.source_batch_index = 2
        state.task_manifest.metadata.clear()
        self.assertIs(check(proposal()), result)

    def test_revisit_schema_round_trip_and_bounded_rationale(self):
        p = proposal()
        self.assertNotIn("revisit", p.to_dict())
        self.assertEqual(LocalEditProposal.from_dict(p.to_dict()), p)
        p = proposal({"reason": "new_batch_evidence", "justification": "Different errors in the new cohort."})
        self.assertEqual(LocalEditProposal.from_dict(p.to_dict()), p)
        for value in ({}, {"reason": "score_went_down", "justification": "bad comparison"},
                      {"reason": "new_batch_evidence", "justification": " "},
                      {"reason": "new_batch_evidence", "justification": "x" * 401}):
            with self.assertRaises(ValueError):
                proposal(value)

    def test_missing_reason_can_be_repaired_without_bypassing_other_rejections(self):
        p = proposal({"reason": "new_batch_evidence", "justification": "The new cohort identifies a different target error."})
        row = {"outcome": "rejected", "candidate_revision_id": "r", "decision": "mutate",
               "operations": list(p.operations), "reason": "repeated_behavior_requires_revisit_reason"}
        with patch("ecologyrsi_dsh.application.formal_trajectory._recent_local_edit_history", return_value=[row]):
            self.assertIsNone(_local_edit_policy_rejection_reason(policy_state(), "c", 2, "r", p))
            row["reason"] = "invalid_parameter"
            self.assertEqual(_local_edit_policy_rejection_reason(policy_state(), "c", 2, "r", p), "duplicate_recent_rejected_bundle")


class HostInterruptionTests(TestCase):
    def monitor(self, callback=None):
        clocks = [100., 1_700_000_000.]
        monitor = HostActivityMonitor(clock=lambda: clocks[0], wall_clock=lambda: clocks[1], on_gap=callback)
        return monitor, clocks

    def test_slow_model_is_not_a_host_gap_and_wall_clock_catches_suspend(self):
        recorded = []
        monitor, clocks = self.monitor(recorded.append)
        for _ in range(120):
            clocks[0] += 5
            clocks[1] += 5
            self.assertEqual(monitor.observe(), 0)
        clocks[1] += 3600  # Platforms whose monotonic clock excludes sleep.
        self.assertEqual(monitor.observe(), 1)
        self.assertEqual(monitor.observe(), 1)
        self.assertEqual(len(recorded), 1)
        self.assertEqual(validate_host_interruption(recorded[0])["gap_seconds"], 3600)

    def test_persistence_failure_does_not_replace_failure_or_leak_admission(self):
        callback = []
        def persist(payload):
            callback.append(payload)
            if len(callback) == 1:
                raise OSError("synthetic ledger contention")
        monitor, clocks = self.monitor(persist)
        admission = RunSampleAdmission(host_activity=monitor)
        error = DshNativeRuntimeUnavailableError(error_code="structured_role_operational_timeout", status_code=503)
        with self.assertLogs("ecologyrsi_dsh.execution.host_activity", level="ERROR"):
            with self.assertRaises(DshNativeRuntimeUnavailableError) as raised:
                with admission.admit("r", 8):
                    clocks[0] += 3600
                    clocks[1] += 3600
                    raise error
        self.assertIs(raised.exception, error)
        self.assertEqual(admission.snapshot("r")["active"], 0)
        self.assertEqual(admission.snapshot("r")["adaptive_limit"], 8)
        monitor.observe()
        self.assertEqual(callback[0], callback[1])

    def test_only_confirmed_provider_congestion_reduces_limit_across_gap(self):
        cases = [(False, "structured_role_operational_timeout", None, 4),
                 (True, "structured_role_operational_timeout", None, 8),
                 (True, "provider_unavailable", 503, 4),
                 (True, "provider_rate_limited", 429, 4),
                 (False, "dsh_native_runtime_transport_error", None, 8)]
        for gap, code, provider_status, expected in cases:
            with self.subTest(gap=gap, code=code):
                monitor, clocks = self.monitor()
                admission = RunSampleAdmission(host_activity=monitor)
                with self.assertRaises(DshNativeRuntimeUnavailableError):
                    with admission.admit("r", 8):
                        if gap:
                            clocks[0] += 100
                            clocks[1] += 100
                        raise DshNativeRuntimeUnavailableError(error_code=code, status_code=503, provider_status=provider_status)
                self.assertEqual(admission.snapshot("r")["adaptive_limit"], expected)
                self.assertEqual(admission.snapshot("r")["active"], 0)

    def test_gap_is_idempotent_replayable_and_does_not_change_run_status(self):
        with EventLedger() as ledger:
            services = ApplicationRuntime(ledger)
            services.director.start_evolution(TaskManifest(task_id="gap", objective="observe", domain_pack="crop-soil-water@toy"), run_id="r")
            clocks = [100., time.time()]
            monitor = HostActivityMonitor(clock=lambda: clocks[0], wall_clock=lambda: clocks[1], on_gap=services.host_activity._on_gap)
            clocks[0] += 120
            clocks[1] += 120
            monitor.observe()
            events = ledger.events_by_kind("r", "HostExecutionInterrupted")
            self.assertEqual(len(events), 1)
            services.host_activity._on_gap(dict(events[0].payload))
            self.assertEqual(len(ledger.events_by_kind("r", "HostExecutionInterrupted")), 1)
            state = services.director.state("r")
            self.assertEqual(state.run.status.value, "running")
            self.assertEqual(_host_activity_projection(state)["observed_gap_seconds"], 120)
            self.assertEqual(state.candidates, ())


class RuntimeProjectionTests(TestCase):
    def test_resume_excludes_old_origins_and_old_phase_fallback_from_eta(self):
        now = datetime.now(timezone.utc)
        def origin(seq, at):
            return event(seq, "EvaluationSampleResultBatchRecorded", {"generation": 0, "candidate_id": "c", "record_count": 9}, at)
        state = NS(task_manifest=NS(metadata={"optimization_protocol": "top2_adaptive_epoch@1",
            "optimization_schedule": {"screening_origin_count": 64, "formal_origin_count_per_finalist": 500,
                "selection_holdout_origin_count": 169, "local_batch_origin_count": 50},
            "prediction_cells_per_origin": 9, "sample_agent_protocol": "dsh-strict-origin-bundle@4"}, max_generations=1),
            run=NS(generation=0, status=NS(value="running")), formal_batch_evaluations=(), holdout_evaluations=(), formal_batches=(),
            candidate_screening_events=(event(0, "CandidateScreened", {"generation": 0, "candidate_id": "c", "origin_count": 0}, now - timedelta(hours=6)),), candidates=(NS(candidate_id="c", generation=0),), events=(
                origin(1, now - timedelta(hours=5)), origin(2, now - timedelta(hours=4)),
                event(3, "HostExecutionInterrupted", {}, now - timedelta(seconds=30)),
                origin(4, now - timedelta(seconds=20))))
        progress = _adaptive_progress_projection(state)
        self.assertIsNone(progress["estimated_remaining_seconds"])
        self.assertEqual(progress["eta_status"], "collecting_after_resume")
        state.events += (origin(5, now - timedelta(seconds=10)),)
        progress = _adaptive_progress_projection(state)
        self.assertEqual(progress["eta_status"], "estimated")
        self.assertGreater(progress["samples_per_minute"], 1)

    def test_untyped_timeout_remains_visible_as_latest_diagnostic(self):
        state = NS(task_manifest=NS(metadata={"strategy_model_id": "route"}), events=(event(1, "DshChildExecutionFailed", {
            "error_code": "structured_role_operational_timeout", "identity": {"stage": "sample.plan"}}),))
        failure = _runtime_failure_projection(state)
        self.assertEqual(failure["error_code"], "structured_role_operational_timeout")
        self.assertIsNone(failure["provider_status"])
        self.assertTrue(failure["retryable"])

    def test_stage_usage_deduplicates_snapshots_and_sums_to_total(self):
        def usage(seq, session, stage, total, kind="DshSessionUsageRecorded"):
            return event(seq, kind, {"identity": {"session_id": session, "stage": stage, "child_reservation_id": session},
                "settlement": "succeeded", "usage_complete": True,
                "session_metrics": {"session_id": session, "provider_usage": {"available": True, "totals": {"total_tokens": total}}}})
        state = NS(run=NS(session_id="root"), events=(
            event(1, "DshRuntimeBound", {"execution_protocol": "native", "capabilities_digest": "a" * 64, "preset_ids": []}),
            usage(2, "s1", "sample.plan", 100), usage(3, "s1", "sample.plan", 120),
            usage(4, "s1", "sample.plan", 110, "DshStructuredResultAccepted"), usage(5, "s2", "sample.critic", 20)))
        usage = _dsh_runtime_projection(state)["provider_usage"]
        self.assertEqual(usage["total_tokens"], 140)
        self.assertEqual(usage["by_stage"]["sample.plan"], {"session_count": 1, "total_tokens": 120})
        self.assertEqual(sum(r["total_tokens"] for r in usage["by_stage"].values()), usage["total_tokens"])

    def test_quality_limits_remain_unchanged(self):
        self.assertEqual(dict(NATIVE_SAMPLE_OPERATION_MAX_TOKENS), {
            "sample.planner": 16384, "sample.repair": 16384, "sample.critic": 8192})
