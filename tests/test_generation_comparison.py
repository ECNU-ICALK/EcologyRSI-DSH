from __future__ import annotations

import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.trajectory import (
    EvaluationPhase,
    EvaluationScope,
    HoldoutArm,
    HoldoutEvaluation,
)
from ecologyrsi_dsh.evaluators.objectives import OBJECTIVE_AGGREGATION_VERSION
from ecologyrsi_dsh.evolution.promotion import (
    PROMOTION_BLOCK_EVIDENCE_VERSION,
    PROMOTION_SCORE_DEFINITION,
)
from ecologyrsi_dsh.evaluators.generation_comparison import build_generation_comparison
from ecologyrsi_dsh.evaluators.fitness import FitnessProfile, SelectionAssessment


TARGETS = ("air_temperature", "relative_humidity", "co2_concentration")
HORIZONS = (1, 6, 24)


def _block_evidence(skill: float) -> dict:
    blocks = []
    for index in range(8):
        cells = [
            {
                "target": target,
                "horizon_hours": horizon,
                "eligible": 1,
                "succeeded": 1,
                "candidate_squared_error_sum": (1.0 - skill) ** 2,
                "baseline_squared_error_sum": 1.0,
                "normalized_reward_sum": skill,
            }
            for target in TARGETS
            for horizon in HORIZONS
        ]
        blocks.append(
            {
                "block_id": digest({"block": index}),
                "origin_block_index": index,
                "cells": cells,
            }
        )
    body = {
        "schema_version": PROMOTION_BLOCK_EVIDENCE_VERSION,
        "block_hours": 24,
        "objective_aggregation_version": OBJECTIVE_AGGREGATION_VERSION,
        "score_definition": PROMOTION_SCORE_DEFINITION,
        "target_weights": {target: 1 / 3 for target in TARGETS},
        "horizons": list(HORIZONS),
        "block_count": len(blocks),
        "blocks": blocks,
    }
    return {**body, "evidence_digest": digest(body)}


def _evaluation(
    arm: HoldoutArm,
    candidate: str,
    revision: str,
    score: float,
    passed: bool = True,
    skill: float = 0.2,
):
    scope = EvaluationScope(
        run_id="run:comparison",
        generation=0,
        candidate_id=candidate,
        candidate_revision_id=revision,
        phase=EvaluationPhase.HOLDOUT,
        cohort_digest=digest({"cohort": "holdout"}),
        origin_count=169,
        holdout_arm=arm,
    )
    return HoldoutEvaluation(
        evaluation_id=f"holdout:{arm.value}",
        scope=scope,
        score=score,
        passed=passed,
        metrics={
            "constraint_violations": 0,
            "sample_execution_coverage_pass": True,
            "objective_weight_coverage": 1.0,
            "objective_aggregation_version": OBJECTIVE_AGGREGATION_VERSION,
            "objective_target_weights": {target: 1 / 3 for target in TARGETS},
            "objective_horizons": list(HORIZONS),
            "baseline_profile_digest": "b" * 64,
            "evaluation_index_digest": "c" * 64,
            "dataset_digest": "d" * 64,
            "split_manifest_digest_sha256": "e" * 64,
            "targets": [
                {
                    "target": target,
                    "horizon_hours": horizon,
                    "skill_score": skill,
                    "sample_execution_coverage": 1.0,
                }
                for target in TARGETS
                for horizon in HORIZONS
            ],
            "promotion_block_evidence": _block_evidence(skill),
        },
        evaluator_digest=digest({"evaluator": "test"}),
    )


class GenerationComparisonTests(unittest.TestCase):
    def test_selects_best_eligible_finalist_and_records_delta(self):
        evaluations = (
            _evaluation(HoldoutArm.FINALIST_1, "candidate:a", "revision:a", 0.4, skill=0.2),
            _evaluation(HoldoutArm.FINALIST_2, "candidate:b", "revision:b", 0.7, skill=0.3),
            _evaluation(HoldoutArm.INCUMBENT, "candidate:inc", "revision:inc", 0.6, skill=0.1),
        )
        comparison = build_generation_comparison(
            run_id="run:comparison",
            generation=0,
            cohort_digest=evaluations[0].scope.cohort_digest,
            holdout_evaluations=evaluations,
            incumbent_candidate_id="candidate:inc",
        )
        self.assertEqual(comparison.selected_candidate_id, "candidate:b")
        self.assertEqual(comparison.selected_revision_id, "revision:b")
        self.assertAlmostEqual(comparison.gate_results["delta_to_incumbent"], 0.1)

    def test_falls_back_to_incumbent_when_finalists_fail(self):
        evaluations = (
            _evaluation(HoldoutArm.FINALIST_1, "candidate:a", "revision:a", 0.9, False, skill=0.2),
            _evaluation(HoldoutArm.FINALIST_2, "candidate:b", "revision:b", 0.8, True, skill=0.2),
            _evaluation(HoldoutArm.INCUMBENT, "candidate:inc", "revision:inc", 0.5, skill=0.1),
        )
        evaluations = tuple(
            item if item.scope.holdout_arm is not HoldoutArm.FINALIST_2 else HoldoutEvaluation(
                evaluation_id=item.evaluation_id,
                scope=item.scope,
                score=item.score,
                passed=item.passed,
                metrics={
                    **item.to_dict()["metrics"],
                    "constraint_violations": 1,
                },
                evaluator_digest=item.evaluator_digest,
            )
            for item in evaluations
        )
        comparison = build_generation_comparison(
            run_id="run:comparison",
            generation=0,
            cohort_digest=evaluations[0].scope.cohort_digest,
            holdout_evaluations=evaluations,
            incumbent_candidate_id="candidate:inc",
        )
        self.assertEqual(comparison.selected_candidate_id, "candidate:inc")
        self.assertEqual(comparison.selected_revision_id, "revision:inc")

    def test_retains_incumbent_below_practical_delta(self):
        evaluations = (
            _evaluation(
                HoldoutArm.FINALIST_1,
                "candidate:a",
                "revision:a",
                0.604,
                skill=0.3,
            ),
            _evaluation(
                HoldoutArm.FINALIST_2,
                "candidate:b",
                "revision:b",
                0.59,
                skill=0.2,
            ),
            _evaluation(
                HoldoutArm.INCUMBENT,
                "candidate:inc",
                "revision:inc",
                0.6,
                skill=0.1,
            ),
        )
        comparison = build_generation_comparison(
            run_id="run:comparison",
            generation=0,
            cohort_digest=evaluations[0].scope.cohort_digest,
            holdout_evaluations=evaluations,
            incumbent_candidate_id="candidate:inc",
        )
        self.assertEqual(comparison.selected_candidate_id, "candidate:inc")
        self.assertEqual(comparison.gate_results["eligible_finalist_count"], 0)

    def test_shared_max_t_family_rejects_noisy_sibling_and_keeps_stable_one(self):
        evaluations = (
            _evaluation(HoldoutArm.FINALIST_1, "candidate:noisy", "revision:noisy", 0.8, skill=0.2),
            _evaluation(HoldoutArm.FINALIST_2, "candidate:stable", "revision:stable", 0.7, skill=0.3),
            _evaluation(HoldoutArm.INCUMBENT, "candidate:inc", "revision:inc", 0.5, skill=0.1),
        )
        profile = FitnessProfile(exploratory_resamples=100)
        assessments = (
            SelectionAssessment(
                "candidate:noisy", "exploratory_adaptive_data", "unstable_or_below_delta",
                8, 6, "a" * 64, 0.3, -0.1, False,
            ),
            SelectionAssessment(
                "candidate:stable", "exploratory_adaptive_data", "selection_only",
                8, 6, "a" * 64, 0.2, 0.1, True,
            ),
        )
        with patch(
            "ecologyrsi_dsh.evaluators.generation_comparison.assess_generation_selection",
            return_value=assessments,
        ) as assess:
            comparison = build_generation_comparison(
                run_id="run:comparison",
                generation=0,
                cohort_digest=evaluations[0].scope.cohort_digest,
                holdout_evaluations=evaluations,
                incumbent_candidate_id="candidate:inc",
                fitness_profile=profile,
            )

        self.assertEqual(assess.call_count, 1)
        self.assertIs(assess.call_args.args[2], profile)
        self.assertEqual(comparison.selected_candidate_id, "candidate:stable")
        self.assertFalse(
            comparison.gate_results["arms"]["finalist_1"]["promotion_assessment"]
            ["primary_selection_gate"]
        )


if __name__ == "__main__":
    unittest.main()
