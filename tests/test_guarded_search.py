"""Guarded search is opt-in, reconstructible, and rejects weak paired evidence."""
from dataclasses import replace
import unittest

from ecologyrsi_dsh.core.models import TaskManifest, digest
from ecologyrsi_dsh.core.search_policy import SEARCH_GUARD_POLICY, local_challenger_policy
from ecologyrsi_dsh.core.state import uses_positive_delta_search_protocol
from ecologyrsi_dsh.core.trajectory import BatchEvaluation, EvaluationScope, EvaluationPhase, FormalBatchArm
from ecologyrsi_dsh.evolution.champion_challenger import assess_local_challenger
from ecologyrsi_dsh.evolution.promotion import build_promotion_block_evidence, assess_promotion_improvement
from ecologyrsi_dsh.evaluators.objectives import DEFAULT_TARGET_WEIGHTS, OBJECTIVE_AGGREGATION_VERSION
from ecologyrsi_dsh.application.formal_trajectory import _durable_batch_metrics, _local_edit_evidence_metrics


def evaluation(arm, gain, blocks=3):
    rows = [{"target": target, "horizon_hours": horizon, "origin_timestamp": day * 24,
             "observed": 0., "predicted": 1. - gain, "baseline": 1., "normalization_scale": 1., "status": "succeeded"}
            for day in range(blocks) for target in DEFAULT_TARGET_WEIGHTS for horizon in (1,6,24)]
    evidence = build_promotion_block_evidence(rows, horizons=(1,6,24), target_weights=DEFAULT_TARGET_WEIGHTS,
        dataset_digest="d"*64, split_manifest_digest_sha256="e"*64)
    metrics = {"objective_aggregation_version": OBJECTIVE_AGGREGATION_VERSION,
        "sample_execution_coverage_pass": True,
        "sample_execution": {"strict_agent_chain_pass": True, "coverage_pass": True,
            "attempted_origin_samples": blocks, "succeeded_origin_samples": blocks,
            "complete_origin_agent_chains": blocks, "failed_origin_samples": 0,
            "failed_examples": 0, "scoring_fallback_examples": 0},
        "objective_target_weights": DEFAULT_TARGET_WEIGHTS, "objective_horizons": [1,6,24],
        "baseline_profile_digest": "b"*64, "evaluation_index_digest": "c"*64,
        "dataset_digest": "d"*64, "split_manifest_digest_sha256": "e"*64,
        "promotion_block_evidence": evidence,
        "targets": [{"target": target, "horizon_hours": horizon, "skill_score": gain}
                    for target in DEFAULT_TARGET_WEIGHTS for horizon in (1,6,24)]}
    return BatchEvaluation(evaluation_id="eval:"+arm.value, score=gain, passed=True, metrics=metrics,
        evaluator_digest="f"*64, scope=EvaluationScope(run_id="run:g",generation=0,candidate_id="c",
            candidate_revision_id="rev:"+arm.value, phase=EvaluationPhase.FORMAL_BATCH,
            cohort_digest=digest("same-cohort"),origin_count=blocks,batch_index=1,formal_batch_arm=arm))


class GuardedSearchTests(unittest.TestCase):
    def assess(self, gain=.02, blocks=3):
        return assess_local_challenger(evaluation(FormalBatchArm.CHAMPION,0,blocks),
            evaluation(FormalBatchArm.CHALLENGER,gain,blocks),challenger_safety_gate_passed=True,
            **local_challenger_policy({"search_guard_policy": SEARCH_GUARD_POLICY}))

    def test_old_frozen_v3_decisions_and_new_guard_are_distinct(self):
        old=TaskManifest(task_id="t",objective="o",domain_pack="g",metadata={
            "execution_protocol":"dsh_native_plugin_evolution@1",
            "host_runtime_build":{"evolution_runtime_schema":"ecologyrsi-dsh.evolution-runtime/3"}})
        new=replace(old,metadata={**old.metadata,"search_guard_policy":SEARCH_GUARD_POLICY})
        self.assertTrue(uses_positive_delta_search_protocol(old))
        self.assertFalse(uses_positive_delta_search_protocol(new))
        self.assertEqual(local_challenger_policy(old.metadata)["minimum_score_delta"],1e-12)
        self.assertEqual(TaskManifest.from_dict(new.to_dict()).digest,new.digest)
        self.assertNotEqual(old.digest,new.digest)
        with self.assertRaises(ValueError): replace(old,metadata={"search_guard_policy":"unknown"})

    def test_tiny_positive_enters_probation_without_replacing_champion(self):
        result=self.assess(.00135548)
        self.assertEqual(result.reason,"probation_below_practical_delta")
        self.assertEqual(result.champion_after_revision_id,"rev:champion")

    def test_paired_three_blocks_needed_and_same_result_repeats(self):
        self.assertEqual(self.assess(blocks=2).reason,"probation_insufficient_evidence")
        first=self.assess()
        self.assertEqual(first.reason,"challenger_improved")
        self.assertEqual(first,self.assess())
        # Local exploration evidence must not weaken the independent promotion gate.
        cert=assess_promotion_improvement(evaluation(FormalBatchArm.CHALLENGER,.02),evaluation(FormalBatchArm.CHAMPION,0))
        self.assertFalse(cert["improved"])
        self.assertEqual(cert["reason_code"],"insufficient_evidence")

    def test_any_cell_regression_blocks_even_when_overall_gain_is_positive(self):
        challenger=evaluation(FormalBatchArm.CHALLENGER,.03)
        metrics=challenger.to_dict()["metrics"]
        metrics["targets"][-1]["skill_score"]=-.01
        result=assess_local_challenger(evaluation(FormalBatchArm.CHAMPION,0),replace(challenger,metrics=metrics),
            challenger_safety_gate_passed=True, require_paired_evidence=True)
        self.assertEqual(result.reason,"challenger_cell_regression")

    def test_missing_or_tampered_blocks_cannot_promote(self):
        for tamper in (False,True):
            challenger=evaluation(FormalBatchArm.CHALLENGER,.03)
            metrics=challenger.to_dict()["metrics"]
            if tamper: metrics["promotion_block_evidence"]["blocks"][0]["cells"][0]["eligible"]=99
            else: metrics.pop("promotion_block_evidence")
            result=assess_local_challenger(evaluation(FormalBatchArm.CHAMPION,0),replace(challenger,metrics=metrics),
                challenger_safety_gate_passed=True,require_paired_evidence=True)
            self.assertEqual(result.reason,"probation_invalid_block_evidence")

    def test_empty_day_block_cannot_count_toward_three_blocks(self):
        champion=evaluation(FormalBatchArm.CHAMPION,0)
        challenger=evaluation(FormalBatchArm.CHALLENGER,.03)
        changed=[]
        for original in (champion, challenger):
            metrics=original.to_dict()["metrics"]
            evidence=metrics["promotion_block_evidence"]
            for cell in evidence["blocks"][0]["cells"]:
                for key in ("eligible", "succeeded", "candidate_squared_error_sum", "baseline_squared_error_sum", "normalized_reward_sum"):
                    cell[key]=0
            evidence["evidence_digest"]=digest({k:v for k,v in evidence.items() if k != "evidence_digest"})
            changed.append(replace(original,metrics=metrics))
        result=assess_local_challenger(*changed,challenger_safety_gate_passed=True,require_paired_evidence=True)
        self.assertEqual(result.reason,"probation_incomplete_paired_blocks")

    def test_block_statistics_are_durable_but_not_sent_to_local_editor(self):
        metrics=evaluation(FormalBatchArm.CHALLENGER,.03).metrics
        self.assertIn("promotion_block_evidence",_durable_batch_metrics(metrics))
        self.assertNotIn("promotion_block_evidence",_local_edit_evidence_metrics(metrics))

if __name__ == "__main__": unittest.main()
