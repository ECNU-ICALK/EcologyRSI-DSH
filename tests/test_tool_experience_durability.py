"""Exercise the post-score -> durable batch -> next-generation feedback path."""
import unittest
from types import SimpleNamespace

from ecologyrsi_dsh.application.formal_trajectory import _durable_batch_metrics
from ecologyrsi_dsh.evolution.agent_policy import (
    build_agent_policy, prior_candidate_tool_experience, summarize_agent_tools,
)


class ToolExperienceDurabilityTests(unittest.TestCase):
    def evidence(self, partition="training_feedback", identity=None):
        records, scores = [], []
        for index, observed, predicted in ((0, 10.0, 12.0), (1, 20.0, 30.0)):
            sample_id = f"s:{index}"
            scores.append({"sample_id": sample_id, "partition": partition,
                           "target": "air_temperature", "horizon_hours": 6,
                           "observed": observed, "predicted": predicted})
            records.append({"sample_id": sample_id, "attempt_trace": [{
                "outcome": "accepted", "model_evidence": [{
                    "tool_id": "candidate-model", "version": "1",
                    "parameters": {}, "status": "completed",
                    "tool_predicted": observed - 2, "used_as_evidence": True,
                }],
            }]})
        return summarize_agent_tools(records, scores, source_phase="formal_batch", candidate_identity=identity)

    def test_frozen_candidate_identity_qualifies_parameter_overrides(self):
        identity = {"predictor_id": "greenhouse-baseline-aligned-ridge@2", "parameters_digest": "c" * 64}
        # candidate-model does not accept overrides. Use real empty tool args.
        scores = [{"sample_id": "s", "partition": "training_feedback", "observed": 10,
                   "predicted": 12, "target": "air_temperature", "horizon_hours": 6}]
        tool = {"tool_id": "candidate-model", "version": "1", "parameters": {},
                "status": "completed", "tool_predicted": 8, "used_as_evidence": True}
        records = [{"sample_id": "s", "attempt_trace": [{"outcome": "accepted", "model_evidence": [tool]}]}]
        row = summarize_agent_tools(records, scores, candidate_identity=identity)[0]
        self.assertEqual(row["resolved_predictor_id"], identity["predictor_id"])
        self.assertEqual(row["effective_parameters_digest"], "c" * 64)
        self.assertEqual(row["parameters"], {})
        self.assertEqual(row["parameters_semantics"], "request_overrides_only")
        self.assertEqual(row["observation_unit"], "tool_calls_per_scoring_cell_including_retries")
        self.assertIn("not_causal", row["adjustment_gain_semantics"])
        legacy = summarize_agent_tools(records, scores)[0]
        self.assertNotIn("effective_parameters_digest", legacy)

    def test_effective_parameter_receipts_keep_distinct_model_groups(self):
        scores = [{"sample_id": "s", "partition": "training_feedback", "observed": 10,
                   "predicted": 12, "target": "air_temperature", "horizon_hours": 6}]
        tools = [{"tool_id": "registered", "version": "1", "parameters": {}, "status": "completed",
                  "tool_predicted": 8, "effective_parameters_digest": value}
                 for value in ("d" * 64, "e" * 64, "invalid")]
        records = [{"sample_id": "s", "attempt_trace": [{"outcome": "accepted", "model_evidence": tools}]}]
        rows = summarize_agent_tools(records, scores, candidate_identity={"predictor_id": "parent"})
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["effective_parameters_digest"] for row in rows}, {"d" * 64, "e" * 64, None})
        self.assertEqual(next(row for row in rows if row["effective_parameters_digest"] is None)[
            "effective_parameters_source"], "unverified")

    def policy(self, durable):
        candidate = SimpleNamespace(candidate_id="c", generation=0)
        evaluation = SimpleNamespace(scope=SimpleNamespace(
            candidate_id="c", generation=0, phase="formal_batch", scope_key="a" * 64),
            metrics=durable)
        holdout = {"sample_execution": {"agent_tool_performance": [{
            "tool_id": "candidate-model", "source_phase": "holdout", "mae": 999,
        }]}}
        rows = prior_candidate_tool_experience(
            SimpleNamespace(formal_batch_evaluations=(evaluation,)), candidate, holdout)
        return build_agent_policy(genome_digest="b" * 64, profile={}, parameters={},
            generation=1, previous_analysis={"generation": 0,
                "ranking": [{"agent_tool_performance": rows}]})

    def test_scored_tool_errors_reach_next_generation_through_durable_batch(self):
        raw = self.evidence()
        metrics = {"objective_score": -0.1, "sample_execution": {
            "agent_tool_performance": raw, "batch_plan": {"private": True}}}
        durable = _durable_batch_metrics(metrics)
        rows = self.policy(durable)["experience"]["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n"], 2)
        self.assertEqual(rows[0]["mae"], 2)
        self.assertEqual(rows[0]["agent_adjustment_mae_gain"], -4)
        self.assertEqual(rows[0]["source_scope_digest"], "a" * 64)
        self.assertEqual(durable["objective_score"], -0.1)
        self.assertNotIn("batch_plan", durable["sample_execution"])
        self.assertEqual(durable["sample_execution"]["agent_tool_performance"], raw)

    def test_missing_historical_experience_is_not_synthesized(self):
        durable = _durable_batch_metrics({"sample_execution": {"coverage": 1}})
        self.assertNotIn("agent_tool_performance", durable["sample_execution"])
        self.assertEqual(self.policy(durable)["experience"]["rows"], [])

    def test_independent_partition_and_holdout_experience_stay_excluded(self):
        self.assertEqual(self.evidence(partition="validation"), [])
        hidden = [{**self.evidence()[0], "source_phase": "holdout", "mae": 999}]
        durable = _durable_batch_metrics({"sample_execution": {"agent_tool_performance": hidden}})
        self.assertEqual(self.policy(durable)["experience"]["rows"], [])


if __name__ == "__main__":
    unittest.main()
