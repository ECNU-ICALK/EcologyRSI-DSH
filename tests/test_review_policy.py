"""Advisory reviews cannot replace Host gates or change evidence purpose."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from ecologyrsi_dsh.core.models import Evaluation, digest
from ecologyrsi_dsh.core.review_policy import build_host_gate_summary, build_review_policy
from ecologyrsi_dsh.core.trajectory import HoldoutArm
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evaluators.generation_comparison import build_generation_comparison
from tests.test_run_parameter_consistency import small_holdout


class PurposeBoundReviewTests(unittest.TestCase):
    def setUp(self):
        self.candidate = small_holdout(HoldoutArm.FINALIST_1, .2)
        self.incumbent = small_holdout(HoldoutArm.INCUMBENT, .1)
        self.task = SimpleNamespace(metadata={"optimization_schedule": OptimizationSchedule.for_new_run().to_dict()})

    def decide(self, *, accepted=False, candidate=None, status="completed", policy=None):
        item = candidate or self.candidate
        review = {"candidate_id": item.scope.candidate_id,
                  "candidate_revision_id": item.scope.candidate_revision_id,
                  "evaluation_scope_digest": item.scope.scope_key,
                  "judge_status": status, "judge_accepted": accepted,
                  "review_policy": policy or build_review_policy(self.task, item)}
        return build_generation_comparison(
            run_id=item.scope.run_id, generation=0, cohort_digest=item.scope.cohort_digest,
            holdout_evaluations=(item, self.incumbent), quick_experiment=True,
            finalist_reviews={"finalist_1": review})

    def test_unsupported_certification_veto_does_not_block_exploration(self):
        decision = self.decide(accepted=False)
        gate = decision.gate_results["arms"]["finalist_1"]
        self.assertTrue(gate["search_eligible"])
        self.assertFalse(gate["judge_advisory_accepted"])
        self.assertTrue(gate["review_gate_pass"])
        self.assertFalse(gate["certification_eligible"])
        self.assertIsNone(decision.gate_results["certification_selected_arm"])

    def test_positive_review_cannot_override_real_cell_regression(self):
        metrics = self.candidate.to_dict()["metrics"]
        metrics["targets"][0]["skill_score"] = .05
        candidate = replace(self.candidate, metrics=metrics)
        gate = self.decide(accepted=True, candidate=candidate).gate_results["arms"]["finalist_1"]
        self.assertFalse(gate["search_eligible"])
        self.assertIn("cell_regression", gate["search_failures"])

    def test_unavailable_review_still_fails_closed(self):
        gate = self.decide(status="unavailable").gate_results["arms"]["finalist_1"]
        self.assertFalse(gate["search_eligible"])

    def test_other_scope_or_purpose_cannot_reuse_advisory_envelope(self):
        for field, value in (("purpose", "certification_candidate_evidence"), ("binding", {})):
            with self.subTest(field=field):
                policy = build_review_policy(self.task, self.candidate)
                policy[field] = value
                policy["policy_digest"] = digest({k: v for k, v in policy.items() if k != "policy_digest"})
                with self.assertRaisesRegex(ValueError, "purpose or evidence binding"):
                    self.decide(policy=policy)

    def test_modified_policy_digest_is_rejected(self):
        policy = build_review_policy(self.task, self.candidate)
        policy["instructions"] = []
        with self.assertRaises(ValueError):
            self.decide(policy=policy)

    def test_review_receives_actual_paired_delta_and_applicable_host_gates(self):
        # Production judges the canonical Evaluation; pairing must not depend
        # on that record having HoldoutEvaluation's convenient .scope property.
        item = self.candidate
        canonical = Evaluation(evaluation_id=item.evaluation_id, run_id=item.scope.run_id,
            candidate_id=item.scope.candidate_id, candidate_revision_id=item.scope.candidate_revision_id,
            evaluation_scope=item.scope.to_dict(), score=item.score, passed=item.passed,
            metrics=item.to_dict()["metrics"], evaluator_digest=item.evaluator_digest)
        summary = build_host_gate_summary(self.task, canonical, self.incumbent)
        self.assertAlmostEqual(summary["score_delta"], .1)
        self.assertAlmostEqual(summary["worst_cell_delta"], .1)
        self.assertTrue(summary["exploration_gates_pass"])
        self.assertTrue(summary["applicable_exploration_gates"]["complete_paired_scoring_evidence"])
        self.assertEqual(summary["certification_applicability"], "not_applicable_to_exploration")
        self.assertEqual(build_review_policy(self.task, canonical, self.incumbent)["host_gate_summary"], summary)

    def test_review_context_rejects_other_cohort_or_evaluator(self):
        for reference in (
            replace(self.incumbent, scope=replace(self.incumbent.scope, cohort_digest="f" * 64), metrics=self.incumbent.to_dict()["metrics"]),
            replace(self.incumbent, evaluator_digest="e" * 64, metrics=self.incumbent.to_dict()["metrics"]),
        ):
            with self.subTest(reference=reference.evaluator_digest), self.assertRaisesRegex(ValueError, "same frozen cohort"):
                build_host_gate_summary(self.task, self.candidate, reference)

    def test_hard_effect_failure_blocks_selection_but_soft_or_unexercised_does_not(self):
        for semantics, status, rejected in (
            ("hard", "failed", True), ("hard", "passed", False),
            ("hard", "inconclusive", False), ("hard", "not_exercised", False),
            ("soft", "inconclusive", False), ("soft", "not_exercised", False),
        ):
            with self.subTest(semantics=semantics, status=status):
                receipt = {"schema_version": "ecologyrsi-dsh.edit-effect-receipt/1",
                    "contract_id": "a" * 64, "scope_digest": self.candidate.scope.scope_key,
                    "child_genome_digest": "b" * 64, "status": status,
                    "checks": [{"predicate_id": "candidate_model_evidence" if semantics == "hard" else "guidance_behavior_changed",
                                "semantics": semantics, "status": status}],
                    "qualification": "behavior_effect_only_not_scientific_improvement"}
                receipt["receipt_digest"] = digest(receipt)
                metrics = self.candidate.to_dict()["metrics"]
                metrics["sample_execution"]["mutation_effect_receipt"] = receipt
                candidate = replace(self.candidate, metrics=metrics)
                gate = self.decide(accepted=True, candidate=candidate).gate_results["arms"]["finalist_1"]
                self.assertEqual(gate["search_eligible"], not rejected)
                self.assertEqual("hard_mutation_effect_failed" in gate["search_failures"], rejected)
                self.assertEqual("hard_mutation_effect_failed" in gate["certification_failures"], rejected)
                host = build_host_gate_summary(self.task, candidate, self.incumbent)
                self.assertEqual(host["applicable_exploration_gates"]["hard_mutation_effect"], not rejected)
