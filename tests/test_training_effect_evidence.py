"""Training behavior failures remain binding without becoming holdout data."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from ecologyrsi_dsh.core.models import TaskManifest, digest
from ecologyrsi_dsh.core.state import validate_generation_comparison_binding
from ecologyrsi_dsh.core.training_effect_evidence import finalist_training_effect_evidence
from ecologyrsi_dsh.core.review_policy import build_host_gate_summary
from ecologyrsi_dsh.core.trajectory import BatchEvaluation, EvaluationPhase, FormalBatchArm, GenerationHoldout, HoldoutArm
from ecologyrsi_dsh.evaluators.generation_comparison import build_generation_comparison
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from tests.test_run_parameter_consistency import small_holdout


class TrainingEffectEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.candidate = small_holdout(HoldoutArm.FINALIST_1, .2)
        self.incumbent = small_holdout(HoldoutArm.INCUMBENT, .1)
        self.items = (self.candidate, self.incumbent)
        self.schedule = OptimizationSchedule.for_evidence_guided_run()
        self.task = TaskManifest(task_id="task:effects", objective="frozen training effect constraints",
            domain_pack="greenhouse_environment@1", metadata={
                "execution_protocol": "dsh_native_plugin_evolution@1",
                "host_runtime_build": {"evolution_runtime_schema": "ecologyrsi-dsh.evolution-runtime/3"},
                "optimization_schedule": self.schedule.to_dict()})

    def receipt(self, *, failed=True):
        row = {"schema_version": "ecologyrsi-dsh.edit-effect-receipt/1", "scope_digest": "a" * 64,
            "contract_id": "b" * 64, "child_genome_digest": "c" * 64,
            "checks": [{"predicate_id": "candidate_model_evidence", "semantics": "hard",
                        "status": "failed" if failed else "passed"}]}
        return {**row, "receipt_digest": digest(row)}

    def screening(self, *, revision=None, generation=0, receipt=True):
        scope = self.candidate.scope
        return SimpleNamespace(event_id="event:screening", run_id=scope.run_id, payload={
            "generation": generation, "candidate_id": scope.candidate_id,
            "candidate_revision_id": revision or scope.candidate_revision_id,
            "record_digest": "d" * 64,
            "metrics": {"sample_execution": {"mutation_effect_receipt": self.receipt()} if receipt else {}}})

    def batch(self, *, revision=None, failed=False, generation=0):
        scope = replace(self.candidate.scope, candidate_revision_id=revision or self.candidate.scope.candidate_revision_id,
            generation=generation, phase=EvaluationPhase.FORMAL_BATCH, holdout_arm=None,
            batch_index=1, formal_batch_arm=FormalBatchArm.CHAMPION)
        return BatchEvaluation(evaluation_id="eval:training", scope=scope, score=.2, passed=True,
            evaluator_digest=self.candidate.evaluator_digest,
            metrics={"sample_execution": {"mutation_effect_receipt": self.receipt(failed=failed)}})

    def compare(self, screenings=(), batches=()):
        return build_generation_comparison(run_id=self.candidate.scope.run_id, generation=0,
            cohort_digest=self.candidate.scope.cohort_digest, holdout_evaluations=self.items,
            quick_experiment=True, experiment_protocol=self.schedule.protocol,
            training_effect_evidence=finalist_training_effect_evidence(self.items,
                screening_events=screenings, batch_evaluations=batches))

    def validate(self, comparison, screenings=(), batches=()):
        holdout = GenerationHoldout(holdout_id="holdout:effects", run_id=comparison.run_id, generation=0,
            cohort_digest=comparison.cohort_digest, origin_count=50,
            arm_bindings={item.scope.holdout_arm.value: {"candidate_id": item.scope.candidate_id,
                "candidate_revision_id": item.scope.candidate_revision_id} for item in self.items})
        validate_generation_comparison_binding(self.task, comparison.run_id, holdout, {}, comparison,
            persisted_evaluations={item.scope.holdout_arm: item for item in self.items},
            persisted_screening_events=screenings, persisted_batch_evaluations=batches)

    def test_screening_hard_failure_blocks_high_holdout_and_replay_cannot_drop_it(self):
        screening = self.screening()
        comparison = self.compare((screening,))
        self.assertEqual(comparison.selected_candidate_id, self.incumbent.scope.candidate_id)
        self.assertIn("hard_mutation_effect_failed", comparison.gate_results["arms"]["finalist_1"]["search_failures"])
        evidence = comparison.gate_results["training_effect_evidence"]["finalist_1"]
        self.assertEqual(evidence["sources"][0]["source_kind"], "screening")
        summary = build_host_gate_summary(self.task, self.candidate, self.incumbent,
            training_effect_evidence=comparison.gate_results["training_effect_evidence"])
        self.assertFalse(summary["applicable_exploration_gates"]["hard_mutation_effect"])
        self.assertEqual(summary["training_effect_evidence"]["sources"][0]["source_kind"], "screening")
        self.assertNotIn("mutation_effect_receipt", comparison.holdout_evaluations[0].metrics["sample_execution"])
        self.validate(comparison, (screening,))
        with self.assertRaisesRegex(ValueError, "sealed training effect evidence"):
            self.validate(self.compare(), (screening,))

    def test_repaired_child_is_not_tainted_by_parent_or_other_generation(self):
        parents = (self.screening(revision="revision:parent"), self.screening(generation=1))
        child = self.batch(failed=False)
        comparison = self.compare(parents, (child,))
        self.assertEqual(comparison.selected_candidate_id, self.candidate.scope.candidate_id)
        self.assertEqual(len(comparison.gate_results["training_effect_evidence"]["finalist_1"]["sources"]), 1)
        self.validate(comparison, parents, (child,))

    def test_rollback_checks_parent_own_evidence_not_the_rejected_child(self):
        child_failure = self.batch(revision="revision:rejected-child", failed=True)
        healthy = self.compare(batches=(child_failure,))
        self.assertEqual(healthy.selected_candidate_id, self.candidate.scope.candidate_id)
        parent_failure = self.batch(failed=True)
        blocked = self.compare(batches=(parent_failure,))
        self.assertEqual(blocked.selected_candidate_id, self.incumbent.scope.candidate_id)
        self.validate(blocked, batches=(parent_failure,))

    def test_legacy_without_receipts_is_byte_identical_and_replayable(self):
        legacy = self.compare()
        with_old_training = self.compare((self.screening(receipt=False),))
        self.assertEqual(legacy.identity_dict(), with_old_training.identity_dict())
        self.assertNotIn("training_effect_evidence", legacy.gate_results)
        self.validate(legacy, (self.screening(receipt=False),))


if __name__ == "__main__":
    unittest.main()
