from __future__ import annotations

import unittest

from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.trajectory import (
    EvaluationPhase,
    EvaluationScope,
    HoldoutArm,
    HoldoutEvaluation,
)
from ecologyrsi_dsh.evaluators.generation_comparison import build_generation_comparison


def _evaluation(arm: HoldoutArm, candidate: str, revision: str, score: float, passed: bool = True):
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
        metrics={"constraint_violations": 0, "sample_execution_coverage_pass": True},
        evaluator_digest=digest({"evaluator": "test"}),
    )


class GenerationComparisonTests(unittest.TestCase):
    def test_selects_best_eligible_finalist_and_records_delta(self):
        evaluations = (
            _evaluation(HoldoutArm.FINALIST_1, "candidate:a", "revision:a", 0.4),
            _evaluation(HoldoutArm.FINALIST_2, "candidate:b", "revision:b", 0.7),
            _evaluation(HoldoutArm.INCUMBENT, "candidate:inc", "revision:inc", 0.6),
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
            _evaluation(HoldoutArm.FINALIST_1, "candidate:a", "revision:a", 0.9, False),
            _evaluation(HoldoutArm.FINALIST_2, "candidate:b", "revision:b", 0.8, True),
            _evaluation(HoldoutArm.INCUMBENT, "candidate:inc", "revision:inc", 0.5),
        )
        evaluations = tuple(
            item if item.scope.holdout_arm is not HoldoutArm.FINALIST_2 else HoldoutEvaluation(
                evaluation_id=item.evaluation_id,
                scope=item.scope,
                score=item.score,
                passed=item.passed,
                metrics={"constraint_violations": 1},
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


if __name__ == "__main__":
    unittest.main()
