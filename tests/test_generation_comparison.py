from __future__ import annotations

import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.models import TaskManifest, digest
from ecologyrsi_dsh.core.state import validate_generation_comparison_binding
from ecologyrsi_dsh.core.trajectory import (
    EvaluationPhase,
    EvaluationScope,
    GenerationComparison,
    GenerationHoldout,
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
            "sample_execution": {
                "attempted_origin_samples": 169,
                "succeeded_origin_samples": 169,
                "minimum_coverage": 0.95,
                "coverage_pass": True,
                "strict_agent_chain_pass": True,
            },
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
    def test_runtime_v2_legacy_gate_shape_remains_replayable(self):
        evaluations = (
            _evaluation(
                HoldoutArm.FINALIST_1,
                "candidate:a",
                "revision:a",
                0.7,
                skill=0.3,
            ),
            _evaluation(
                HoldoutArm.FINALIST_2,
                "candidate:b",
                "revision:b",
                0.65,
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
        current = build_generation_comparison(
            run_id="run:comparison",
            generation=0,
            cohort_digest=evaluations[0].scope.cohort_digest,
            holdout_evaluations=evaluations,
            incumbent_candidate_id="candidate:inc",
        )
        current_gates = current.to_dict()["gate_results"]
        legacy_arms = {}
        finalist_fields = {
            "passed",
            "constraint_violations",
            "overall_coverage",
            "coverage_pass",
            "eligible",
            "score",
            "complete_objective_grid",
            "no_cell_regression",
            "worst_cell_delta",
            "cell_deltas",
            "promotion_assessment",
            "stability_lower_bound",
            "failures",
        }
        for arm in (HoldoutArm.FINALIST_1, HoldoutArm.FINALIST_2):
            source = current_gates["arms"][arm.value]
            legacy_arms[arm.value] = {
                key: source[key] for key in finalist_fields
            }
            legacy_arms[arm.value]["eligible"] = source[
                "certification_eligible"
            ]
            legacy_arms[arm.value]["failures"] = source[
                "certification_failures"
            ]
        incumbent_source = current_gates["arms"][HoldoutArm.INCUMBENT.value]
        legacy_arms[HoldoutArm.INCUMBENT.value] = {
            key: incumbent_source[key]
            for key in (
                "passed",
                "constraint_violations",
                "overall_coverage",
                "coverage_pass",
                "eligible",
                "score",
            )
        }
        legacy_gates = {
            key: current_gates[key]
            for key in (
                "schema_version",
                "eligible_finalist_count",
                "selected_arm",
                "selected_score",
                "incumbent_score",
                "delta_to_incumbent",
                "incumbent_gate_pass",
                "fitness_profile_digest",
                "selection_rule",
                "challenger_promotion_allowed",
            )
        }
        legacy_gates["arms"] = legacy_arms
        legacy = GenerationComparison(
            comparison_id=current.comparison_id,
            run_id=current.run_id,
            generation=current.generation,
            cohort_digest=current.cohort_digest,
            holdout_evaluations=current.holdout_evaluations,
            selected_candidate_id=current.selected_candidate_id,
            selected_revision_id=current.selected_revision_id,
            gate_results=legacy_gates,
            created_at=current.created_at,
        )
        holdout = GenerationHoldout(
            holdout_id="holdout:v2-legacy",
            run_id=current.run_id,
            generation=current.generation,
            cohort_digest=current.cohort_digest,
            origin_count=169,
            arm_bindings={
                item.scope.holdout_arm.value: {
                    "candidate_id": item.scope.candidate_id,
                    "candidate_revision_id": item.scope.candidate_revision_id,
                }
                for item in evaluations
            },
        )
        task = TaskManifest(
            task_id="runtime-v2-legacy-comparison",
            objective="replay the frozen v2 gate shape",
            domain_pack="greenhouse_environment@1",
            visible_datasets=("dataset",),
            budget={"max_candidates": 4, "max_generations": 1},
            metadata={
                "execution_protocol": "dsh_native_plugin_evolution@1",
                "host_runtime_build": {
                    "evolution_runtime_schema": (
                        "ecologyrsi-dsh.evolution-runtime/2"
                    )
                },
            },
        )

        validate_generation_comparison_binding(
            task,
            current.run_id,
            holdout,
            {"exploration_only": False},
            legacy,
            persisted_evaluations={
                item.scope.holdout_arm: item for item in evaluations
            },
        )

    def test_positive_delta_search_selects_negative_score_without_certifying_it(self):
        evaluations = (
            _evaluation(
                HoldoutArm.FINALIST_1,
                "candidate:a",
                "revision:a",
                -0.39,
                passed=False,
                skill=0.2,
            ),
            _evaluation(
                HoldoutArm.FINALIST_2,
                "candidate:b",
                "revision:b",
                -0.45,
                passed=False,
                skill=0.2,
            ),
            _evaluation(
                HoldoutArm.INCUMBENT,
                "candidate:inc",
                "revision:inc",
                -0.4,
                passed=False,
                skill=0.1,
            ),
        )

        comparison = build_generation_comparison(
            run_id="run:comparison",
            generation=0,
            cohort_digest=evaluations[0].scope.cohort_digest,
            holdout_evaluations=evaluations,
            incumbent_candidate_id="candidate:inc",
            positive_delta_search=True,
        )

        gate = comparison.gate_results["arms"]["finalist_1"]
        self.assertEqual(comparison.selected_revision_id, "revision:a")
        self.assertTrue(gate["search_eligible"])
        self.assertFalse(gate["certification_eligible"])
        self.assertEqual(
            comparison.gate_results["selection_policy"],
            "positive_delta_search@1",
        )
        self.assertIsNone(
            comparison.gate_results["certification_selected_arm"]
        )

    def test_positive_delta_search_records_cell_regression_as_certification_risk(self):
        finalist = _evaluation(
            HoldoutArm.FINALIST_1,
            "candidate:a",
            "revision:a",
            0.61,
            skill=0.3,
        )
        rows = finalist.to_dict()["metrics"]["targets"]
        rows[0]["skill_score"] = -0.2
        finalist = self._with_target_rows(finalist, rows)
        evaluations = (
            finalist,
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
            positive_delta_search=True,
        )

        gate = comparison.gate_results["arms"]["finalist_1"]
        self.assertEqual(comparison.selected_revision_id, "revision:a")
        self.assertTrue(gate["search_eligible"])
        self.assertFalse(gate["certification_eligible"])
        self.assertFalse(gate["no_cell_regression"])
        self.assertIn("cell_regression", gate["certification_failures"])

    def test_positive_delta_search_ranks_by_largest_delta(self):
        evaluations = (
            _evaluation(
                HoldoutArm.FINALIST_1,
                "candidate:a",
                "revision:a",
                0.61,
                skill=0.2,
            ),
            _evaluation(
                HoldoutArm.FINALIST_2,
                "candidate:b",
                "revision:b",
                0.72,
                skill=0.3,
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
            positive_delta_search=True,
        )

        self.assertEqual(comparison.selected_revision_id, "revision:b")
        self.assertEqual(comparison.gate_results["search_eligible_finalist_count"], 2)

    def test_positive_delta_search_tie_uses_stability_before_candidate_id(self):
        evaluations = (
            _evaluation(
                HoldoutArm.FINALIST_1,
                "candidate:a",
                "revision:a",
                0.7,
                skill=0.2,
            ),
            _evaluation(
                HoldoutArm.FINALIST_2,
                "candidate:z",
                "revision:z",
                0.7,
                skill=0.3,
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
            positive_delta_search=True,
        )

        self.assertEqual(comparison.selected_revision_id, "revision:z")

    def test_search_winner_can_be_certification_eligible_without_being_certified(self):
        evaluations = (
            _evaluation(
                HoldoutArm.FINALIST_1,
                "candidate:larger-delta",
                "revision:larger-delta",
                0.8,
                skill=0.3,
            ),
            _evaluation(
                HoldoutArm.FINALIST_2,
                "candidate:more-stable",
                "revision:more-stable",
                0.7,
                skill=0.3,
            ),
            _evaluation(
                HoldoutArm.INCUMBENT,
                "candidate:inc",
                "revision:inc",
                0.6,
                skill=0.1,
            ),
        )
        assessments = (
            SelectionAssessment(
                "candidate:larger-delta",
                "exploratory_adaptive_data",
                "selection_only",
                8,
                6,
                "a" * 64,
                0.2,
                0.05,
                True,
            ),
            SelectionAssessment(
                "candidate:more-stable",
                "exploratory_adaptive_data",
                "selection_only",
                8,
                6,
                "a" * 64,
                0.1,
                0.09,
                True,
            ),
        )
        with patch(
            "ecologyrsi_dsh.evaluators.generation_comparison.assess_generation_selection",
            return_value=assessments,
        ):
            comparison = build_generation_comparison(
                run_id="run:comparison",
                generation=0,
                cohort_digest=evaluations[0].scope.cohort_digest,
                holdout_evaluations=evaluations,
                incumbent_candidate_id="candidate:inc",
                positive_delta_search=True,
            )

        self.assertEqual(comparison.selected_candidate_id, "candidate:larger-delta")
        self.assertEqual(
            comparison.gate_results["certification_selected_arm"],
            HoldoutArm.FINALIST_2.value,
        )
        self.assertEqual(
            comparison.gate_results["selected_search_certification_status"],
            "certification_eligible_not_selected",
        )

    def test_non_integer_or_negative_constraint_count_fails_closed(self):
        for invalid_count in (0.5, -1, None):
            with self.subTest(constraint_violations=invalid_count):
                finalist = _evaluation(
                    HoldoutArm.FINALIST_1,
                    "candidate:a",
                    "revision:a",
                    0.8,
                    skill=0.3,
                )
                metrics = finalist.to_dict()["metrics"]
                if invalid_count is None:
                    metrics.pop("constraint_violations")
                else:
                    metrics["constraint_violations"] = invalid_count
                finalist = HoldoutEvaluation(
                    evaluation_id=finalist.evaluation_id,
                    scope=finalist.scope,
                    score=finalist.score,
                    passed=finalist.passed,
                    metrics=metrics,
                    evaluator_digest=finalist.evaluator_digest,
                    created_at=finalist.created_at,
                )
                evaluations = (
                    finalist,
                    _evaluation(
                        HoldoutArm.FINALIST_2,
                        "candidate:b",
                        "revision:b",
                        0.4,
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
                    positive_delta_search=True,
                )

                gate = comparison.gate_results["arms"]["finalist_1"]
                self.assertFalse(gate["search_eligible"])
                self.assertFalse(gate["certification_eligible"])
                self.assertGreater(gate["constraint_violations"], 0)
                self.assertIn("constraint_violations", gate["search_failures"])

    @staticmethod
    def _with_target_rows(
        evaluation: HoldoutEvaluation,
        rows: list[dict],
    ) -> HoldoutEvaluation:
        metrics = evaluation.to_dict()["metrics"]
        metrics["targets"] = rows
        return HoldoutEvaluation(
            evaluation_id=evaluation.evaluation_id,
            scope=evaluation.scope,
            score=evaluation.score,
            passed=evaluation.passed,
            metrics=metrics,
            evaluator_digest=evaluation.evaluator_digest,
            created_at=evaluation.created_at,
        )

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

    def test_exploration_only_generation_cannot_promote_a_challenger(self):
        evaluations = (
            _evaluation(
                HoldoutArm.FINALIST_1,
                "candidate:a",
                "revision:a",
                0.8,
                skill=0.4,
            ),
            _evaluation(
                HoldoutArm.FINALIST_2,
                "candidate:b",
                "revision:b",
                0.7,
                skill=0.3,
            ),
            _evaluation(
                HoldoutArm.INCUMBENT,
                "candidate:inc",
                "revision:inc",
                0.5,
                skill=0.1,
            ),
        )

        comparison = build_generation_comparison(
            run_id="run:comparison",
            generation=0,
            cohort_digest=evaluations[0].scope.cohort_digest,
            holdout_evaluations=evaluations,
            incumbent_candidate_id="candidate:inc",
            challenger_promotion_allowed=False,
        )

        self.assertEqual(comparison.selected_candidate_id, "candidate:inc")
        self.assertEqual(comparison.gate_results["eligible_finalist_count"], 0)
        for arm in ("finalist_1", "finalist_2"):
            self.assertFalse(comparison.gate_results["arms"][arm]["eligible"])
            self.assertIn(
                "screening_exploration_only",
                comparison.gate_results["arms"][arm]["failures"],
            )

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

    def test_shared_incomplete_reported_grid_cannot_define_its_own_expected_grid(self):
        evaluations = tuple(
            self._with_target_rows(item, [item.to_dict()["metrics"]["targets"][0]])
            for item in (
                _evaluation(
                    HoldoutArm.FINALIST_1,
                    "candidate:a",
                    "revision:a",
                    0.65,
                    skill=0.2,
                ),
                _evaluation(
                    HoldoutArm.FINALIST_2,
                    "candidate:b",
                    "revision:b",
                    0.7,
                    skill=0.3,
                ),
                _evaluation(
                    HoldoutArm.INCUMBENT,
                    "candidate:inc",
                    "revision:inc",
                    0.6,
                    skill=0.1,
                ),
            )
        )

        comparison = build_generation_comparison(
            run_id="run:comparison",
            generation=0,
            cohort_digest=evaluations[0].scope.cohort_digest,
            holdout_evaluations=evaluations,
            incumbent_candidate_id="candidate:inc",
        )

        self.assertEqual(comparison.selected_candidate_id, "candidate:inc")
        for arm in ("finalist_1", "finalist_2"):
            gate = comparison.gate_results["arms"][arm]
            self.assertFalse(gate["complete_objective_grid"])
            self.assertFalse(gate["eligible"])
            self.assertIn("objective_grid_incomplete", gate["failures"])

    def test_duplicate_reported_cell_cannot_overwrite_into_a_complete_grid(self):
        finalist = _evaluation(
            HoldoutArm.FINALIST_2,
            "candidate:b",
            "revision:b",
            0.7,
            skill=0.3,
        )
        rows = finalist.to_dict()["metrics"]["targets"]
        finalist = self._with_target_rows(finalist, [*rows, dict(rows[0])])
        evaluations = (
            _evaluation(
                HoldoutArm.FINALIST_1,
                "candidate:a",
                "revision:a",
                0.4,
                skill=0.2,
            ),
            finalist,
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

        gate = comparison.gate_results["arms"]["finalist_2"]
        self.assertFalse(gate["complete_objective_grid"])
        self.assertFalse(gate["eligible"])
        self.assertEqual(comparison.selected_candidate_id, "candidate:inc")


if __name__ == "__main__":
    unittest.main()
