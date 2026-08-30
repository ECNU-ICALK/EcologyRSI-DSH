from __future__ import annotations

import unittest

from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.trajectory import (
    BatchEvaluation,
    EvaluationPhase,
    EvaluationScope,
    FormalBatchArm,
    FormalBatchComparisonDecision,
)
from ecologyrsi_dsh.evolution.champion_challenger import (
    LOCAL_CELL_REGRESSION_TOLERANCE,
    LOCAL_MINIMUM_SCORE_DELTA,
    assess_local_challenger,
    local_challenger_safety_reason,
)


TARGETS = ("air_temperature", "relative_humidity")
HORIZONS = (1, 6)


def _metrics(skill: float) -> dict:
    return {
        "objective_aggregation_version": "weighted-skill@1",
        "objective_target_weights": {target: 0.5 for target in TARGETS},
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
            }
            for target in TARGETS
            for horizon in HORIZONS
        ],
    }


def _evaluation(
    *,
    arm: FormalBatchArm,
    revision_id: str,
    score: float,
    skill: float,
    metrics: dict | None = None,
    evaluator_digest: str | None = None,
    cohort_digest: str | None = None,
) -> BatchEvaluation:
    return BatchEvaluation(
        evaluation_id=f"evaluation:{arm.value}:{revision_id}",
        scope=EvaluationScope(
            run_id="run:local-comparison",
            generation=2,
            candidate_id="candidate:a",
            candidate_revision_id=revision_id,
            phase=EvaluationPhase.FORMAL_BATCH,
            cohort_digest=cohort_digest or digest({"cohort": 3}),
            origin_count=50,
            batch_index=3,
            formal_batch_arm=arm,
        ),
        score=score,
        passed=True,
        metrics=metrics if metrics is not None else _metrics(skill),
        evaluator_digest=evaluator_digest or digest({"evaluator": "formal"}),
        created_at="2026-08-30T00:00:00Z",
    )


class ChampionChallengerSelectionTests(unittest.TestCase):
    def test_v2_safety_requires_explicit_valid_execution_counts(self) -> None:
        valid = {
            "constraint_violations": 0,
            "sample_execution_coverage_pass": True,
            "sample_execution": {
                "attempted_origin_samples": 50,
                "succeeded_origin_samples": 50,
                "minimum_coverage": 0.95,
                "coverage_pass": True,
                "strict_agent_chain_pass": True,
            },
        }
        self.assertIsNone(local_challenger_safety_reason(valid))

        invalid_cases = {
            "missing_constraint_count": {
                key: value
                for key, value in valid.items()
                if key != "constraint_violations"
            },
            "missing_attempted_count": {
                **valid,
                "sample_execution": {
                    key: value
                    for key, value in valid["sample_execution"].items()
                    if key != "attempted_origin_samples"
                },
            },
            "zero_attempted_count": {
                **valid,
                "sample_execution": {
                    **valid["sample_execution"],
                    "attempted_origin_samples": 0,
                },
            },
            "succeeded_exceeds_attempted": {
                **valid,
                "sample_execution": {
                    **valid["sample_execution"],
                    "succeeded_origin_samples": 51,
                },
            },
            "minimum_coverage_out_of_range": {
                **valid,
                "sample_execution": {
                    **valid["sample_execution"],
                    "minimum_coverage": 1.1,
                },
            },
        }
        for name, metrics in invalid_cases.items():
            with self.subTest(name=name):
                self.assertIsNotNone(local_challenger_safety_reason(metrics))

    def test_negative_challenger_can_replace_more_negative_champion(self) -> None:
        champion = _evaluation(
            arm=FormalBatchArm.CHAMPION,
            revision_id="revision:champion",
            score=-0.6,
            skill=-0.6,
        )
        challenger = _evaluation(
            arm=FormalBatchArm.CHALLENGER,
            revision_id="revision:challenger",
            score=-0.4,
            skill=-0.4,
        )

        result = assess_local_challenger(
            champion,
            challenger,
            challenger_safety_gate_passed=True,
        )

        self.assertEqual(
            result.decision,
            FormalBatchComparisonDecision.CHALLENGER_PROMOTED,
        )
        self.assertEqual(result.reason, "challenger_improved")
        self.assertEqual(result.champion_after_revision_id, "revision:challenger")
        self.assertAlmostEqual(result.score_delta, 0.2)

    def test_exact_practical_delta_is_not_enough(self) -> None:
        champion = _evaluation(
            arm=FormalBatchArm.CHAMPION,
            revision_id="revision:champion",
            score=0.2,
            skill=0.2,
        )
        challenger = _evaluation(
            arm=FormalBatchArm.CHALLENGER,
            revision_id="revision:challenger",
            score=0.205,
            skill=0.205,
        )

        result = assess_local_challenger(
            champion,
            challenger,
            challenger_safety_gate_passed=True,
        )

        self.assertEqual(LOCAL_MINIMUM_SCORE_DELTA, 0.005)
        self.assertEqual(
            result.decision,
            FormalBatchComparisonDecision.CHAMPION_RETAINED,
        )
        self.assertEqual(result.reason, "below_practical_delta")

    def test_any_cell_regression_beyond_tolerance_retains_champion(self) -> None:
        champion = _evaluation(
            arm=FormalBatchArm.CHAMPION,
            revision_id="revision:champion",
            score=0.2,
            skill=0.2,
        )
        challenger_metrics = _metrics(0.3)
        challenger_metrics["targets"][0]["skill_score"] = (
            0.2 - LOCAL_CELL_REGRESSION_TOLERANCE * 2
        )
        challenger = _evaluation(
            arm=FormalBatchArm.CHALLENGER,
            revision_id="revision:challenger",
            score=0.3,
            skill=0.3,
            metrics=challenger_metrics,
        )

        result = assess_local_challenger(
            champion,
            challenger,
            challenger_safety_gate_passed=True,
        )

        self.assertFalse(result.cell_regression_gate_passed)
        self.assertEqual(result.reason, "challenger_cell_regression")

    def test_contract_mismatch_has_priority_over_other_failures(self) -> None:
        champion = _evaluation(
            arm=FormalBatchArm.CHAMPION,
            revision_id="revision:champion",
            score=0.2,
            skill=0.2,
        )
        challenger_metrics = _metrics(0.3)
        challenger_metrics["targets"] = []
        challenger = _evaluation(
            arm=FormalBatchArm.CHALLENGER,
            revision_id="revision:challenger",
            score=0.3,
            skill=0.3,
            metrics=challenger_metrics,
            evaluator_digest=digest({"evaluator": "different"}),
        )

        result = assess_local_challenger(
            champion,
            challenger,
            challenger_safety_gate_passed=False,
        )

        self.assertEqual(result.reason, "incompatible_comparison_contract")
        self.assertEqual(len(result.comparison_contract_digest), 64)

    def test_incomplete_grids_fail_on_the_named_side(self) -> None:
        complete_champion = _evaluation(
            arm=FormalBatchArm.CHAMPION,
            revision_id="revision:champion",
            score=0.2,
            skill=0.2,
        )
        complete_challenger = _evaluation(
            arm=FormalBatchArm.CHALLENGER,
            revision_id="revision:challenger",
            score=0.3,
            skill=0.3,
        )
        incomplete_champion_metrics = _metrics(0.2)
        incomplete_champion_metrics["targets"] = incomplete_champion_metrics[
            "targets"
        ][:-1]
        incomplete_challenger_metrics = _metrics(0.3)
        incomplete_challenger_metrics["targets"][0]["skill_score"] = "not-finite"

        champion_result = assess_local_challenger(
            _evaluation(
                arm=FormalBatchArm.CHAMPION,
                revision_id="revision:champion",
                score=0.2,
                skill=0.2,
                metrics=incomplete_champion_metrics,
            ),
            complete_challenger,
            challenger_safety_gate_passed=True,
        )
        challenger_result = assess_local_challenger(
            complete_champion,
            _evaluation(
                arm=FormalBatchArm.CHALLENGER,
                revision_id="revision:challenger",
                score=0.3,
                skill=0.3,
                metrics=incomplete_challenger_metrics,
            ),
            challenger_safety_gate_passed=True,
        )

        self.assertEqual(champion_result.reason, "champion_evaluation_incomplete")
        self.assertEqual(
            challenger_result.reason,
            "challenger_evaluation_incomplete",
        )

    def test_safety_failure_and_lower_score_retain_champion(self) -> None:
        champion = _evaluation(
            arm=FormalBatchArm.CHAMPION,
            revision_id="revision:champion",
            score=0.2,
            skill=0.2,
        )
        better = _evaluation(
            arm=FormalBatchArm.CHALLENGER,
            revision_id="revision:better",
            score=0.3,
            skill=0.3,
        )
        lower = _evaluation(
            arm=FormalBatchArm.CHALLENGER,
            revision_id="revision:lower",
            score=0.1,
            skill=0.2,
        )

        unsafe = assess_local_challenger(
            champion,
            better,
            challenger_safety_gate_passed=False,
        )
        worse = assess_local_challenger(
            champion,
            lower,
            challenger_safety_gate_passed=True,
        )

        self.assertEqual(unsafe.reason, "challenger_safety_gate_failed")
        self.assertEqual(worse.reason, "below_practical_delta")
        self.assertEqual(
            unsafe.champion_after_revision_id,
            champion.scope.candidate_revision_id,
        )


if __name__ == "__main__":
    unittest.main()
