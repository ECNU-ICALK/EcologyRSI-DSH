"""Formal statistics are computed from genuine block sufficient statistics."""
from dataclasses import replace
import math
import unittest

from ecologyrsi_dsh.core.models import Evaluation, digest
from ecologyrsi_dsh.evaluators.fitness import FitnessProfile
from ecologyrsi_dsh.evaluators.formal_evidence import assess_independent_replica, certification_policy
from ecologyrsi_dsh.evaluators.objectives import DEFAULT_TARGET_WEIGHTS, OBJECTIVE_AGGREGATION_VERSION, skill_score
from ecologyrsi_dsh.evolution.promotion import build_promotion_block_evidence


def scored_evaluation(errors=None, *, days=14, bad_cell=False):
    errors = errors or [.6] * days
    rows = []
    cells = []
    for target in DEFAULT_TARGET_WEIGHTS:
        for horizon in (1, 6, 24):
            per_day = [1.1] * days if bad_cell and target == "air_temperature" and horizon == 6 else errors
            for day, error in enumerate(per_day):
                for hour in range(6):
                    rows.append(dict(target=target, horizon_hours=horizon, origin_timestamp=day * 24 + hour,
                                     observed=0., predicted=error, baseline=1., normalization_scale=1.))
            rmse = math.sqrt(sum(error ** 2 for error in per_day) / days)
            cells.append(dict(target=target, horizon_hours=horizon, n=days * 6,
                              skill_score=skill_score(rmse, 1.), sample_execution_coverage=1.))
    evidence = build_promotion_block_evidence(rows, horizons=(1, 6, 24), target_weights=DEFAULT_TARGET_WEIGHTS,
                                             dataset_digest="d" * 64, split_manifest_digest_sha256="e" * 64)
    score = sum(cell["skill_score"] * DEFAULT_TARGET_WEIGHTS[cell["target"]] / 3 for cell in cells)
    return Evaluation(evaluation_id="formal:evaluation", run_id="formal:run", candidate_id="formal:candidate",
                      artifact_digest="a" * 64, evaluator_digest="f" * 64, score=score, passed=True,
                      metrics={"scientific_pass": True, "sample_execution_coverage": 1.,
                               "objective_weight_coverage": 1., "targets": cells,
                               "sample_execution": {"successful_agent_provenance_pass": True},
                               "objective_aggregation_version": OBJECTIVE_AGGREGATION_VERSION,
                               "objective_target_weights": dict(DEFAULT_TARGET_WEIGHTS), "objective_horizons": [1, 6, 24],
                               "baseline_profile_digest": "b" * 64, "promotion_block_evidence": evidence})


class FormalEvidenceTests(unittest.TestCase):
    def assess(self, evaluation, *, profile=None, baseline="b" * 64):
        profile = profile or FitnessProfile()
        return assess_independent_replica(evaluation, profile, policy=certification_policy(profile),
                                          baseline_profile_digest=baseline)

    def test_production_blocks_supply_real_formal_statistics(self):
        result = self.assess(scored_evaluation())
        self.assertEqual(result["outcome"], "passed")
        self.assertTrue(result["formal_confirmation"])
        self.assertGreater(result["formal_score_lcb"], 0)
        self.assertEqual(result["paired_block_count"], 14)
        self.assertEqual(result["valid_three_day_start_count"], 12)
        self.assertEqual(result["certification_scope"], "point_prediction")
        self.assertEqual(result["uq_status"], "not_requested")
        self.assertFalse(result["uq_pass"])

    def test_zero_gain_and_single_cell_regression_cannot_certify(self):
        for evaluation in (scored_evaluation([1.] * 14), scored_evaluation(bad_cell=True)):
            with self.subTest(score=evaluation.score):
                self.assertEqual(self.assess(evaluation)["outcome"], "failed")

    def test_noisy_positive_point_is_not_positive_confidence(self):
        evaluation = scored_evaluation([.1] * 7 + [1.3] * 7)
        self.assertGreater(evaluation.score, .005)
        result = self.assess(evaluation)
        self.assertEqual(result["outcome"], "failed")
        self.assertLess(result["formal_score_lcb"], 0)

    def test_requested_interval_claim_never_downgrades_to_point_only(self):
        result = self.assess(scored_evaluation(), profile=FitnessProfile(require_predictive_intervals=True))
        self.assertEqual(result["outcome"], "inconclusive")
        self.assertEqual(result["certification_scope"], "point_and_interval")
        self.assertFalse(result["formal_confirmation"])

    def test_small_cohort_or_repeated_days_do_not_buy_evidence(self):
        evaluation = scored_evaluation(days=3)
        self.assertEqual(self.assess(evaluation)["outcome"], "inconclusive")
        metrics = evaluation.to_dict()["metrics"]
        raw = metrics["promotion_block_evidence"]
        raw["blocks"][1]["origin_block_index"] = raw["blocks"][0]["origin_block_index"]
        raw["evidence_digest"] = digest({k: v for k, v in raw.items() if k != "evidence_digest"})
        self.assertEqual(self.assess(replace(evaluation, metrics=metrics))["outcome"], "inconclusive")

    def test_binding_score_count_and_existing_scientific_guards_remain_strict(self):
        evaluation = scored_evaluation()
        self.assertEqual(self.assess(evaluation, baseline="c" * 64)["outcome"], "inconclusive")
        self.assertEqual(self.assess(replace(evaluation, score=.99))["outcome"], "inconclusive")
        self.assertEqual(self.assess(replace(evaluation, passed=False))["outcome"], "failed")
        metrics = evaluation.to_dict()["metrics"]
        metrics["targets"][0]["n"] += 1
        self.assertEqual(self.assess(replace(evaluation, metrics=metrics))["outcome"], "inconclusive")

    def test_deterministic_replay_reuses_statistics_not_extra_observations(self):
        evaluation = scored_evaluation([.3, .8] * 7)
        self.assertEqual(self.assess(evaluation), self.assess(evaluation))

    def test_claim_policy_cannot_be_changed_after_opening(self):
        profile = FitnessProfile()
        policy = certification_policy(profile)
        policy["minimum_paired_blocks"] = 2
        with self.assertRaisesRegex(ValueError, "changed after freezing"):
            assess_independent_replica(scored_evaluation(), profile, policy=policy,
                                       baseline_profile_digest="b" * 64)

    def test_cell_summary_cannot_hide_a_scored_regression(self):
        evaluation = scored_evaluation(bad_cell=True)
        metrics = evaluation.to_dict()["metrics"]
        for row in metrics["targets"]:
            if row["skill_score"] < 0:
                row["skill_score"] = .2
        result = self.assess(replace(evaluation, metrics=metrics))
        self.assertEqual(result["outcome"], "inconclusive")
        self.assertIn("formal_cell_skill_mismatch", result["failures"])
