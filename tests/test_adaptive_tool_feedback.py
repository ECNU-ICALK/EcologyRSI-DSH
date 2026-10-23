"""Exercise the actual adaptive summary used by webpage evolution."""
from types import MappingProxyType, SimpleNamespace as NS
from unittest.mock import patch
import unittest

from ecologyrsi_dsh.application.formal_trajectory import _durable_batch_metrics
from ecologyrsi_dsh.application.generation_execution import _build_adaptive_analysis
from ecologyrsi_dsh.core.trajectory import HoldoutArm
from ecologyrsi_dsh.evolution.agent_policy import (
    ADAPTIVE_TOOL_FEEDBACK_KEY, ADAPTIVE_TOOL_FEEDBACK_POLICY,
    adaptive_tool_feedback_enabled, build_agent_policy,
)
from ecologyrsi_dsh.evolution.context import safe_aggregate_feedback
from tests import test_tool_experience_durability as tool_fixture


class AdaptiveToolFeedbackTests(unittest.TestCase):
    def analysis(self, enabled=True, attribution=None):
        candidate = NS(candidate_id="candidate:test", generation=0, slot_index=0)
        revision = NS(candidate_id=candidate.candidate_id, revision_id="revision:test",
                      revision_digest="a" * 64, genome_digest="b" * 64, behavior_digest="c" * 64)
        tool_rows = tool_fixture.ToolExperienceDurabilityTests().evidence()
        metrics = _durable_batch_metrics({"sample_execution": {"agent_tool_performance": tool_rows,
            **({"candidate_agent_attribution": attribution} if attribution else {})}})
        # Persisted BatchEvaluation metrics recursively freeze nested mappings.
        for row in metrics["sample_execution"]["agent_tool_performance"]:
            row["parameters"] = MappingProxyType(row["parameters"])
        def batch(phase, generation=0, cid=candidate.candidate_id):
            return NS(scope=NS(candidate_id=cid, generation=generation, phase=phase,
                               scope_key="d" * 64, batch_index=1, cohort_digest="b" * 64,
                               candidate_revision_id=revision.revision_id), metrics=metrics)
        state = NS(run=NS(run_id="run:adaptive-feedback"),
            task_manifest=NS(metadata={**({ADAPTIVE_TOOL_FEEDBACK_KEY: ADAPTIVE_TOOL_FEEDBACK_POLICY} if enabled else {}),
                **({"candidate_attribution_policy": "sealed_unique_origin_attribution@1"} if attribution else {})}),
            candidates=(candidate,), screening_for=lambda *args: None, revision=lambda _: revision,
            formal_batch_evaluations=(batch("formal_batch"), batch("holdout"), batch("validation"),
                                      batch("formal_batch", generation=1), batch("formal_batch", cid="other")))
        def holdout(arm, cid, revision_id):
            return NS(scope=NS(holdout_arm=arm, candidate_id=cid, candidate_revision_id=revision_id),
                      score=0.1, metrics={"sample_execution": {"agent_tool_performance": [{
                          "source_phase": "training_feedback", "tool_id": "HOLDOUT_MUST_NOT_LEAK"}]}})
        comparison = NS(holdout_evaluations=(
            holdout(HoldoutArm.FINALIST_1, candidate.candidate_id, revision.revision_id),
            holdout(HoldoutArm.INCUMBENT, "incumbent", "revision:incumbent")),
            selected_candidate_id="incumbent", selected_revision_id="revision:incumbent",
            gate_results={"selected_arm": "incumbent", "arms": {
                "finalist_1": {"eligible": False, "search_eligible": False, "passed": False},
                "incumbent": {}}})
        # Lineage summaries do not participate in the feedback transport path.
        with patch("ecologyrsi_dsh.application.generation_execution._outer_mutation_evidence", return_value={}), \
             patch("ecologyrsi_dsh.application.generation_execution._local_edit_trajectory_evidence", return_value={}):
            return _build_adaptive_analysis(state, 0, comparison, (candidate,), "incumbent")

    def test_real_adaptive_summary_transports_sealed_tool_errors_to_next_policy(self):
        analysis = self.analysis()
        payload = safe_aggregate_feedback(analysis.to_dict(), name="prior adaptive analysis")
        policy = build_agent_policy(genome_digest="e" * 64, profile={}, parameters={},
                                    previous_analysis=payload, generation=1)
        rows = policy["experience"]["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mae"], 2)
        self.assertEqual(rows[0]["agent_adjustment_mae_gain"], -4)
        self.assertEqual(rows[0]["source_scope_digest"], "d" * 64)
        self.assertEqual(rows[0]["source_phase"], "formal_batch")
        self.assertIsNone(analysis.selected_candidate_id)
        self.assertFalse(analysis.ranking[0]["eligible"])
        with self.assertRaisesRegex(ValueError, "must precede"):
            build_agent_policy(genome_digest="e" * 64, profile={}, parameters={},
                               previous_analysis=payload, generation=0)

    def test_historical_adaptive_analysis_keeps_original_shape(self):
        analysis = self.analysis(enabled=False)
        self.assertNotIn("agent_tool_performance", analysis.ranking[0])
        policy = build_agent_policy(genome_digest="e" * 64, profile={}, parameters={},
                                    previous_analysis=analysis.to_dict(), generation=1)
        self.assertEqual(policy["experience"]["rows"], [])

    def test_unknown_feedback_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported adaptive_tool_feedback_policy"):
            adaptive_tool_feedback_enabled({ADAPTIVE_TOOL_FEEDBACK_KEY: "unknown@1"})


if __name__ == "__main__":
    unittest.main()
