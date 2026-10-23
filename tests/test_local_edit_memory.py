from copy import deepcopy
from types import SimpleNamespace
import unittest

from ecologyrsi_dsh.core.immutable import freeze_json
from ecologyrsi_dsh.core.models import canonical_json, digest
from ecologyrsi_dsh.core.trajectory import (
    BatchEvaluation, EvaluationScope, FormalBatchComparisonDecision,
)
from ecologyrsi_dsh.evolution.local_edit_memory import (
    MAX_BYTES, MAX_RECORDS, POLICY, POLICY_KEY, build_local_edit_memory,
    enabled, local_edit_memory_context,
)


class MemoryState:
    def __init__(self):
        self.run = SimpleNamespace(run_id="run:memory")
        self.formal_batch_evaluations = []
        self.formal_batch_comparisons = []
        self.revisions, self.proposals = {}, {}

    def revision(self, key):
        return self.revisions[key]

    def local_edit_proposal_for(self, candidate, batch):
        return self.proposals.get((candidate, batch))

    @property
    def independent_evaluations(self):
        raise AssertionError("independent evaluation must never enter edit memory")


def add_pair(state, generation, *, failed=False, operation=None, index=0):
    cid = f"candidate:{generation}:{index}"
    run_id = state.run.run_id
    for suffix in ("parent", "child"):
        revision = SimpleNamespace(
            revision_id=f"{cid}:{suffix}", run_id=run_id, generation=generation,
            candidate_id=cid, genome_digest=digest([cid, suffix]),
            parent_revision_id=f"{cid}:parent" if suffix == "child" else None,
            source_batch_index=0 if suffix == "child" else None,
        )
        state.revisions[revision.revision_id] = revision
    metrics = {key: digest(key) for key in (
        "dataset_digest", "split_manifest_digest_sha256", "evaluation_index_digest",
        "baseline_profile_digest", "baseline_metrics_digest", "feedback_update_cohort_digest",
    )}
    metrics.update(sample_execution_coverage=1.0, observed_labels=["PRIVATE_LABEL"],
                   sample_execution={"attempted_origin_samples": 25,
                       "succeeded_origin_samples": 25, "failed_origin_samples": 0,
                       "scoring_fallback_examples": 0, "strict_agent_chain_pass": True})
    evaluations = []
    for arm, suffix, score in (("champion", "parent", .1), ("challenger", "child", .02)):
        arm_metrics = deepcopy(metrics)
        if failed and arm == "challenger":
            arm_metrics["sample_execution"].update(succeeded_origin_samples=24,
                failed_origin_samples=1, scoring_fallback_examples=1)
            arm_metrics["sample_execution_coverage"] = 224 / 225
        evaluation = BatchEvaluation(
            evaluation_id=f"{cid}:{arm}:evaluation",
            scope=EvaluationScope(run_id=run_id, generation=generation, candidate_id=cid,
                candidate_revision_id=f"{cid}:{suffix}", phase="formal_batch", batch_index=1,
                cohort_digest=digest([generation, index]), origin_count=25, formal_batch_arm=arm),
            score=score, passed=False, metrics=arm_metrics, evaluator_digest=digest("evaluator"),
            created_at="2026-10-21T00:00:00Z",
        )
        evaluations.append(evaluation)
        state.formal_batch_evaluations.append(evaluation)
    left, right = evaluations
    comparison = SimpleNamespace(
        comparison_id=f"{cid}:comparison", run_id=run_id, generation=generation,
        candidate_id=cid, batch_index=1, cohort_digest=left.scope.cohort_digest,
        champion_before_revision_id=f"{cid}:parent", challenger_revision_id=f"{cid}:child",
        champion_evaluation_id=left.evaluation_id, challenger_evaluation_id=right.evaluation_id,
        champion_evaluation_digest=left.evaluation_digest, challenger_evaluation_digest=right.evaluation_digest,
        score_delta=-.08, decision=FormalBatchComparisonDecision.CHAMPION_RETAINED,
        reason="challenger_safety_gate_failed" if failed else "challenger_score_regressed",
        safety_gate_passed=not failed, cell_regression_gate_passed=False,
    )
    state.formal_batch_comparisons.append(comparison)
    state.proposals[(cid, 0)] = freeze_json({"proposal": {"operations": [operation or {
        "op": "select_instruction_template", "role": "sample-planner",
        "instruction_template_id": "sample-planner-horizon-split-model@1",
    }]}})
    return comparison


class LocalEditMemoryTests(unittest.TestCase):
    def test_missing_frozen_policy_keeps_old_context_empty_without_reading_history(self):
        self.assertEqual(local_edit_memory_context(SimpleNamespace(), generation=2), {})
        self.assertFalse(enabled({}))
        with self.assertRaises(ValueError):
            enabled({POLICY_KEY: "unknown"})

    def test_opted_in_context_contains_only_prior_generation_memory(self):
        state = MemoryState()
        state.task_manifest = SimpleNamespace(metadata={POLICY_KEY: POLICY})
        add_pair(state, 0)
        add_pair(state, 1)
        result = local_edit_memory_context(state, generation=1)
        self.assertEqual(set(result), {"prior_generation_edit_memory"})
        self.assertEqual(len(result["prior_generation_edit_memory"]["records"]), 1)
        self.assertTrue(enabled(state.task_manifest.metadata))

    def test_prior_pair_evidence_is_bounded_advisory_not_labels_or_absolute_scores(self):
        state = MemoryState()
        add_pair(state, 0)
        add_pair(state, 1, failed=True)
        result = build_local_edit_memory(state, generation=2)
        self.assertEqual([row["generation"] for row in result["records"]], [1, 0])
        self.assertIsNone(result["records"][0]["score_delta"])
        self.assertFalse(result["records"][0]["complete_paired_scoring"])
        self.assertEqual(result["records"][1]["score_delta"], -.08)
        self.assertTrue(result["advisory_only"])
        self.assertFalse(result["scientific_improvement_established"])
        text = canonical_json(result)
        self.assertNotIn("PRIVATE_LABEL", text)
        for row in result["records"]:
            self.assertNotIn("champion_score", row)
            self.assertNotIn("challenger_score", row)
        self.assertEqual(result["memory_digest"], digest({k: v for k, v in result.items() if k != "memory_digest"}))

    def test_current_future_foreign_run_and_initial_comparisons_are_excluded(self):
        state = MemoryState()
        initial = add_pair(state, 0)
        initial.decision = FormalBatchComparisonDecision.INITIAL_CHAMPION
        foreign = add_pair(state, 1)
        foreign.run_id = "run:other"
        add_pair(state, 2)
        add_pair(state, 3)
        self.assertEqual(build_local_edit_memory(state, generation=2)["records"], [])

    def test_mismatched_evaluation_binding_is_not_used(self):
        for field, value in (("champion_evaluation_digest", digest("tampered")),
                             ("cohort_digest", digest("other-cohort")),
                             ("champion_before_revision_id", "wrong-revision")):
            with self.subTest(field=field):
                state = MemoryState()
                comparison = add_pair(state, 0)
                setattr(comparison, field, value)
                self.assertEqual(build_local_edit_memory(state, generation=1)["records"], [])

    def test_different_scoring_bindings_and_missing_coverage_hide_performance_delta(self):
        for mutation in ("baseline", "coverage", "boolean_count", "origins"):
            with self.subTest(mutation=mutation):
                state = MemoryState()
                comparison = add_pair(state, 0)
                right = state.formal_batch_evaluations[-1]
                values = right.to_dict()
                if mutation == "baseline":
                    values["metrics"]["baseline_metrics_digest"] = digest("different subset")
                elif mutation == "coverage":
                    del values["metrics"]["sample_execution_coverage"]
                elif mutation == "boolean_count":
                    values["metrics"]["sample_execution"]["failed_origin_samples"] = False
                else:
                    values["scope"]["origin_count"] = 24
                right = BatchEvaluation.from_dict(values)
                state.formal_batch_evaluations[-1] = right
                comparison.challenger_evaluation_digest = right.evaluation_digest
                row = build_local_edit_memory(state, generation=1)["records"][0]
                self.assertFalse(row["complete_paired_scoring"])
                self.assertIsNone(row["score_delta"])

    def test_revision_lineage_and_missing_proposal_cannot_fabricate_edit_memory(self):
        for mutation in ("lineage", "run", "proposal"):
            state = MemoryState()
            comparison = add_pair(state, 0)
            child = state.revision(comparison.challenger_revision_id)
            if mutation == "lineage":
                child.parent_revision_id = "other-parent"
            elif mutation == "run":
                child.run_id = "run:other"
            else:
                state.proposals.clear()
            self.assertEqual(build_local_edit_memory(state, generation=1)["records"], [])

    def test_operation_diversity_size_bound_and_authored_content_is_not_transferred(self):
        state = MemoryState()
        for generation in range(12):
            add_pair(state, generation)
        add_pair(state, 0, index=1, operation={"op": "author_skill_program", "role": "sample-planner",
            "skill_program": {"steps": [{"guidance": "PRIVATE_INSTRUCTION" * 2000}]}})
        result = build_local_edit_memory(state, generation=12)
        self.assertLessEqual(len(result["records"]), MAX_RECORDS)
        self.assertLessEqual(len(canonical_json(result).encode()), MAX_BYTES)
        self.assertEqual(result["records"][1]["operations"][0]["op"], "author_skill_program")
        self.assertNotIn("PRIVATE_INSTRUCTION", canonical_json(result))

    def test_input_order_is_irrelevant_and_invalid_generation_is_rejected(self):
        state = MemoryState()
        add_pair(state, 0)
        add_pair(state, 1)
        before = build_local_edit_memory(state, generation=2)
        state.formal_batch_evaluations.reverse()
        state.formal_batch_comparisons.reverse()
        self.assertEqual(before, build_local_edit_memory(state, generation=2))
        for invalid in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                build_local_edit_memory(state, generation=invalid)


if __name__ == "__main__":
    unittest.main()
