from types import SimpleNamespace as NS
import unittest

from ecologyrsi_dsh.api.evidence_projection import candidate_score_evidence, evidence_funnel
from ecologyrsi_dsh.evolution.diagnosis import diagnose_evidence, failure_category
from ecologyrsi_dsh.core.models import digest


def state(**values):
    return NS(task_manifest=NS(metadata={"optimization_protocol": "quick_adaptive_epoch@1"}),
              candidates=[], events=[], formal_batch_evaluations=[], formal_batch_comparisons=[],
              local_edit_outcomes=[], generation_comparisons=[], **values)


class EvidenceProjectionTests(unittest.TestCase):
    def test_host_resolution_and_execution_receipts_are_counted_without_claiming_behavior_difference(self):
        resolution = {"schema_version": "ecologyrsi-dsh.mutation-effect-resolution/1", "operations": [{"path": "parameter:x"}]}
        resolution["resolution_digest"] = digest(resolution)
        s = state()
        s.candidates = [NS(candidate_id="c1", proposal_id="p1"), NS(candidate_id="c2", proposal_id="p2")]
        s.proposal = lambda proposal_id: NS(metadata={"effect_resolution": resolution} if proposal_id == "p1" else {})
        def receipt(status):
            row = {"schema_version": "ecologyrsi-dsh.edit-effect-receipt/1", "status": status,
                   "scope_digest": status, "checks": [], "qualification": "behavior_effect_only_not_scientific_improvement"}
            return {**row, "receipt_digest": digest(row)}
        measurements = [receipt(status) for status in ("passed", "failed", "inconclusive", "not_exercised")]
        s.events = [NS(kind="CandidateScreeningRecorded", payload={"candidate_id": "c1", "metrics": {
            "sample_execution": {"mutation_effect_receipt": row}, "sample_execution_trace_archive": "private-compressed-trace"}}) for row in measurements]
        s.formal_batch_evaluations = [NS(scope=NS(candidate_id="c1"), metrics={"reused_screening_evidence": True,
            "sample_execution": {"mutation_effect_receipt": measurements[0]}})]
        report = evidence_funnel(s)
        self.assertEqual(report["stages"]["structural_validity"]["numerator"], 1)
        self.assertEqual(report["stages"]["structural_validity"]["denominator"], 2)
        self.assertIsNone(report["generated_attempts"])
        self.assertIsNone(report["stages"]["behavior_change"]["rate"])
        observed = report["execution_effect_observations"]
        self.assertEqual(observed["recorded"], 4, "reused screening evidence is not a fifth receipt")
        self.assertTrue(all(observed[key] == 1 for key in ("passed", "failed", "inconclusive", "not_exercised")))
        self.assertFalse(observed["controlled_behavior_change_verified"])
        self.assertNotIn("private-compressed-trace", repr(report))

    def test_screening_event_and_reused_evaluation_never_publish_compressed_trace(self):
        from ecologyrsi_dsh.api.events import EventEndpointsMixin
        from ecologyrsi_dsh.api.projection import _public_evaluation_metrics
        metrics = {"skill_score": .1, "sample_execution_trace_archive": "private-compressed-trace",
                   "sample_execution_records": [{"reasoning": "private-reasoning"}],
                   "promotion_block_evidence": {"raw": "private-blocks"}}
        event = NS(seq=1, event_id="screening-event", run_id="run", kind="CandidateScreeningRecorded",
                   payload={"candidate_id": "c1", "metrics": metrics}, created_at="2026-10-02T00:00:00Z")
        for public in (EventEndpointsMixin._event_json(event), _public_evaluation_metrics(metrics)):
            self.assertNotIn("private-", repr(public))
            self.assertNotIn("sample_execution_trace_archive", repr(public))

    def test_legacy_run_with_explicit_null_schedule_has_an_empty_funnel(self):
        s = state()
        s.task_manifest.metadata["optimization_schedule"] = None
        report = evidence_funnel(s)
        self.assertEqual(report["stages"]["local_pair_completion"]["denominator"], 0)
        self.assertIsNone(report["stages"]["local_acceptance"]["rate"])

    def test_guided_budget_progress_charges_five_screening_arms_but_not_reused_warmup(self):
        from ecologyrsi_dsh.api.projection import _adaptive_progress_projection, _screening_candidates
        from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
        s = state()
        s.task_manifest = NS(metadata={"optimization_protocol": "evidence_guided_epoch@1",
            "sample_agent_mode": "dsh_native_agent", "optimization_schedule": OptimizationSchedule.for_evidence_guided_run().to_dict()}, max_generations=4)
        s.run = NS(generation=1, status=NS(value="completed"))
        s.formal_batches = ()
        s.candidate_screening_events = tuple(NS(payload={"generation": 0, "candidate_id": "c"+str(i), "origin_count": 10}) for i in range(5))
        s.formal_batch_evaluations = [NS(scope=NS(generation=0, origin_count=n), metrics=m) for n, m in [(10, {"reused_screening_evidence": True}), (25, {}), (25, {})]]
        s.holdout_evaluations = tuple(NS(scope=NS(generation=0, origin_count=50)) for _ in range(2))
        s.events = (NS(kind="GenerationComparisonRecorded", payload={"comparison": {"generation": 0}}),)
        s.generation_cohort_for = lambda _: NS(revision_bindings={"historic-incumbent": "r"})
        report = _adaptive_progress_projection(s)
        self.assertEqual(report["completed_origins"], 200)
        self.assertEqual(report["run_total_origins"], 800)
        s.candidates = [NS(candidate_id="historic-incumbent", generation=0), NS(candidate_id="unrelated", generation=0)]
        self.assertEqual([item.candidate_id for item in _screening_candidates(s, 1)], ["historic-incumbent"])

    def test_applied_is_not_a_verified_success_and_unknown_denominators_stay_null(self):
        s = state()
        s.candidates = [NS(candidate_id="c", role="search"), NS(candidate_id="seed", role="incumbent_control")]
        s.local_edit_outcomes = [{"outcome": "applied"}]
        s.events = [NS(kind="CandidateScreenedOut", payload={"candidate_id": "c", "reason": "outside_preregistered_quick_trajectory"})]
        report = evidence_funnel(s)
        self.assertEqual(report["registered_candidates"], 1)
        self.assertEqual(report["budget_unallocated_candidates"], 1)
        self.assertEqual(report["applied_edits"], 1)
        self.assertFalse(report["applied_is_verified_gain"])
        self.assertIsNone(report["stages"]["structural_validity"]["denominator"])
        self.assertIsNone(report["stages"]["local_acceptance"]["rate"])
        self.assertIsNone(report["stages"]["independent_certification"]["numerator"])

    def test_score_is_not_parent_gain_without_recorded_comparison(self):
        result = candidate_score_evidence(state(), NS(candidate_id="child", generation=0), NS(score=-.02, metrics={}))
        self.assertEqual(result["score_vs_baseline"], -.02)
        self.assertIsNone(result["delta_vs_parent"])
        self.assertIsNone(result["delta_vs_incumbent"])

    def test_same_holdout_candidate_gain_and_selected_gain_are_distinct(self):
        s = state()
        def arm(name, candidate, score, cohort="cohort"):
            return NS(score=score, evaluator_digest="e", scope=NS(holdout_arm=name,
                cohort_digest=cohort, candidate_id=candidate, candidate_revision_id=candidate+"-r"))
        incumbent, challenger = arm("incumbent", "seed", -.03), arm("finalist_1", "child", .02)
        comparison = NS(generation=0, comparison_id="paired", holdout_evaluations=[incumbent, challenger],
                        gate_results={"delta_to_incumbent": 0.0})
        s.generation_comparisons = [comparison]
        result = candidate_score_evidence(s, NS(candidate_id="child", generation=0), NS(score=.02, metrics={}))
        self.assertAlmostEqual(result["delta_vs_incumbent"], .05)
        self.assertEqual(result["selected_gain"], 0)
        challenger.scope.cohort_digest = "different"
        result = candidate_score_evidence(s, NS(candidate_id="child", generation=0), NS(score=.02, metrics={}))
        self.assertIsNone(result["delta_vs_incumbent"])

    def test_failure_taxonomy_does_not_call_evidence_shortage_performance_loss(self):
        self.assertEqual(failure_category("structured_child_model_error"), "operational")
        self.assertEqual(failure_category("challenger_safety_gate_failed"), "invalid_change")
        self.assertEqual(failure_category("insufficient_paired_evidence"), "insufficient_evidence")
        self.assertEqual(failure_category("cell_regression"), "performance")
        self.assertEqual(failure_category("independent_review_not_accepted"), "unclassified")

    def test_initial_champion_is_not_a_successful_local_pair(self):
        s = state()
        s.formal_batch_comparisons = [NS(candidate_id="c", batch_index=0, challenger_revision_id="r0", decision="initial_champion")]
        report = evidence_funnel(s)
        self.assertEqual(report["stages"]["local_pair_completion"]["denominator"], 0)
        self.assertIsNone(candidate_score_evidence(s, NS(candidate_id="c", generation=0), None)["delta_vs_parent"])

    def test_inconclusive_certification_is_not_counted_as_a_valid_scientific_failure(self):
        s = state()
        s.events = [NS(kind="FormalStageCompleted", payload={"stage": "validation", "outcome": "inconclusive", "candidate_id": "c"})]
        report = evidence_funnel(s)
        self.assertEqual(report["certification_inconclusive"], 1)
        self.assertEqual(report["stages"]["independent_certification"]["denominator"], 0)
        self.assertIsNone(report["stages"]["independent_certification"]["rate"])

    def test_guided_diagnosis_targets_paired_regression_without_reinterpreting_legacy_events(self):
        task = NS(task_id="t", digest="task-digest", metadata={"optimization_protocol": "evidence_guided_epoch@1"})
        comparison = NS(comparison_id="comparison-1", gate_results={"arms": {"finalist_1": {
            "cell_deltas": {"co2_concentration@6h": -.04, "air_temperature@1h": .02},
            "search_failures": ["cell_regression", "independent_review_not_accepted"]}}})
        report = diagnose_evidence(task, None, comparison, NS(snapshot_digest="knowledge"))
        self.assertEqual(report.weak_cells, ("co2_concentration@6h",))
        self.assertIn("comparison:comparison-1", report.evidence_refs)
        self.assertIn("cell_regression", report.failure_patterns)
        task.metadata = {"optimization_protocol": "quick_adaptive_epoch@1"}
        legacy = diagnose_evidence(task, None, comparison, NS(snapshot_digest="knowledge"))
        self.assertEqual(legacy.weak_cells, ())
        self.assertNotIn("comparison:comparison-1", legacy.evidence_refs)


if __name__ == "__main__":
    unittest.main()
