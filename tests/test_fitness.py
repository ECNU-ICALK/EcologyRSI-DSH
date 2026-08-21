from __future__ import annotations

import unittest

from ecologyrsi_dsh.core.models import Evaluation, TaskManifest, digest
from ecologyrsi_dsh.evaluators.fitness import (
    EXPLORATORY_EVIDENCE_CLASS,
    FitnessProfile,
    assess_generation_selection,
    build_fitness_assessment,
    build_formal_fitness_assessment,
    build_moving_block_resample_indices,
    fitness_ranking_key,
)
from ecologyrsi_dsh.evaluators.objectives import normalized_absolute_error_reward


TARGETS = ("air_temperature", "relative_humidity", "co2_concentration")
HORIZONS = (1, 6, 24)


def _task() -> TaskManifest:
    return TaskManifest(
        task_id="fitness",
        objective="greenhouse forecast",
        domain_pack="greenhouse_environment@1",
        visible_datasets=("agc_cucumber_2018",),
        metadata={"execution_protocol": "dsh_native_plugin_evolution@1"},
    )


def _cells(skill: float, *, bad_cell: float | None = None) -> list[dict]:
    rows = []
    for index, (target, horizon) in enumerate(
        (item for target in TARGETS for item in ((target, h) for h in HORIZONS))
    ):
        rows.append(
            {
                "target": target,
                "horizon_hours": horizon,
                "skill_score": bad_cell if index == 0 and bad_cell is not None else skill,
                "sample_execution_coverage": 1.0,
                "n": 50,
                "paired_block_count": 8,
            }
        )
    return rows


def _block_evidence(scores: tuple[float, ...], indices: tuple[int, ...] | None = None) -> dict:
    indices = indices or tuple(range(len(scores)))
    blocks = []
    for index, score in zip(indices, scores):
        cells = [
            {
                "target": target,
                "horizon_hours": 1,
                "eligible": 1,
                "succeeded": 1,
                "candidate_squared_error_sum": (1.0 - score) ** 2,
                "baseline_squared_error_sum": 1.0,
                "normalized_reward_sum": score,
            }
            for target in TARGETS
        ]
        blocks.append(
            {
                "block_id": digest({"calendar_day": index}),
                "origin_block_index": index,
                "cells": cells,
            }
        )
    body = {
        "schema_version": "paired_24h_objective_sufficient_statistics@1",
        "block_hours": 24,
        "maximum_blocks": 128,
        "objective_aggregation_version": "weighted_task_skill_reward@2",
        "score_definition": "coverage_penalized_weighted_rmse_skill@2",
        "target_weights": {target: 1 / 3 for target in TARGETS},
        "horizons": [1],
        "block_count": len(blocks),
        "blocks": blocks,
    }
    return {**body, "evidence_digest": digest(body)}


def _evaluation(name: str, score: float, block_scores: tuple[float, ...]) -> Evaluation:
    return Evaluation(
        evaluation_id=f"evaluation:{name}",
        run_id="run:fitness",
        candidate_id=f"candidate:{name}",
        score=score,
        passed=True,
        evaluator_digest="f" * 64,
        metrics={
            "scientific_pass": True,
            "constraint_violations": 0,
            "objective_weight_coverage": 1.0,
            "targets": _cells(score),
            "objective_aggregation_version": "weighted_task_skill_reward@2",
            "objective_target_weights": {target: 1 / 3 for target in TARGETS},
            "objective_horizons": [1],
            "baseline_profile_digest": "b" * 64,
            "evaluation_index_digest": "c" * 64,
            "dataset_digest": "d" * 64,
            "split_manifest_digest_sha256": "e" * 64,
            "promotion_block_evidence": _block_evidence(block_scores),
        },
    )


class FitnessTests(unittest.TestCase):
    def test_point_estimate_and_interval_use_identical_block_ids(self) -> None:
        profile = FitnessProfile.from_task(_task()).with_overrides(
            exploratory_resamples=100
        )
        incumbent = _evaluation("same-spine-old", 0.1, (0.0,) * 8)
        candidate = _evaluation("same-spine-new", 0.2, (0.1,) * 8)
        result = assess_generation_selection((candidate,), incumbent, profile)[0]
        evidence = candidate.metrics["promotion_block_evidence"]
        expected = digest(
            {
                "indices": list(range(8)),
                "block_ids": [block["block_id"] for block in evidence["blocks"]],
            }
        )
        self.assertEqual(result.paired_block_ids_digest, expected)

    def test_existing_sample_reward_values_are_unchanged(self) -> None:
        self.assertEqual(
            normalized_absolute_error_reward([2.0], [1.0], 2.0),
            (0.5, 0.5),
        )
        self.assertEqual(
            normalized_absolute_error_reward([1.0], [5.0], 2.0),
            (-2.0, -1.0),
        )

    def test_bootstrap_draws_contiguous_three_day_blocks(self) -> None:
        draws = build_moving_block_resample_indices(
            tuple(range(8)), resamples=25, seed_material="fixed"
        )
        self.assertTrue(draws)
        for draw in draws:
            for offset in range(0, 6, 3):
                chunk = draw[offset : offset + 3]
                self.assertEqual(chunk, tuple(range(chunk[0], chunk[0] + len(chunk))))

    def test_total_blocks_without_four_legal_three_day_starts_are_insufficient(self) -> None:
        profile = FitnessProfile.from_task(_task())
        incumbent = _evaluation("old-gap", 0.1, (0.0,) * 8)
        candidate = _evaluation("new-gap", 0.2, (0.1,) * 8)
        for evaluation in (incumbent, candidate):
            evidence = _block_evidence((0.0,) * 8, (0, 1, 2, 10, 11, 12, 20, 21))
            evaluation.metrics["promotion_block_evidence"] = evidence
        result = assess_generation_selection((candidate,), incumbent, profile)[0]
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertFalse(result.primary_selection_gate)

    def test_selection_bound_is_labeled_exploratory_not_confidence(self) -> None:
        profile = FitnessProfile.from_task(_task())
        incumbent = _evaluation("old", 0.1, (0.0,) * 8)
        candidate = _evaluation("new", 0.2, (0.1,) * 8)
        result = assess_generation_selection((candidate,), incumbent, profile)[0]
        self.assertEqual(result.evidence_class, EXPLORATORY_EVIDENCE_CLASS)
        self.assertNotIn("confidence", result.to_dict())
        self.assertFalse(result.formal_confirmation)

    def test_max_t_prevents_noise_winner_from_passing(self) -> None:
        profile = FitnessProfile.from_task(_task()).with_overrides(
            exploratory_resamples=500
        )
        incumbent = _evaluation("old-noise", 0.1, (0.0,) * 8)
        noisy = _evaluation(
            "noise", 0.2, (0.7, -0.5, 0.6, -0.4, 0.7, -0.5, 0.6, -0.4)
        )
        stable = _evaluation("stable", 0.15, (0.2,) * 8)
        results = assess_generation_selection((noisy, stable), incumbent, profile)
        by_id = {item.candidate_id: item for item in results}
        self.assertFalse(by_id[noisy.candidate_id].primary_selection_gate)
        self.assertTrue(by_id[stable.candidate_id].primary_selection_gate)

    def test_efficiency_cannot_outrank_scientific_regression(self) -> None:
        profile = FitnessProfile.from_task(_task())
        incumbent = _evaluation("incumbent", 0.5, (0.0,) * 8)
        regressed = _evaluation("fast-regression", 0.6, (0.1,) * 8)
        regressed.metrics["targets"] = _cells(0.1, bad_cell=-0.1)
        robust = _evaluation("slow-robust", 0.55, (0.05,) * 8)
        bad = build_fitness_assessment(
            regressed, incumbent, {"latency_ms": 1}, profile
        )
        good = build_fitness_assessment(
            robust, incumbent, {"latency_ms": 10_000}, profile
        )
        self.assertGreater(fitness_ranking_key(good), fitness_ranking_key(bad))

    def test_execution_policy_quality_is_a_separate_lower_order_fitness_track(self) -> None:
        profile = FitnessProfile.from_task(_task())
        incumbent = _evaluation("policy-incumbent", 0.4, (0.0,) * 8)
        reliable = _evaluation("policy-reliable", 0.5, (0.1,) * 8)
        noisy = _evaluation("policy-noisy", 0.5, (0.1,) * 8)
        reliable.metrics["sample_execution"] = {
            "eligible_examples": 10,
            "failed_examples": 0,
            "retry_count": 0,
            "repair_count": 0,
            "critic_outcome_counts": {"accepted": 10},
        }
        noisy.metrics["sample_execution"] = {
            "eligible_examples": 10,
            "failed_examples": 0,
            "retry_count": 5,
            "repair_count": 0,
            "critic_outcome_counts": {"accepted": 10, "rejected": 10},
        }

        good = build_fitness_assessment(reliable, incumbent, {}, profile)
        bad = build_fitness_assessment(noisy, incumbent, {}, profile)

        self.assertEqual(good.primary_score, bad.primary_score)
        self.assertGreater(good.execution_policy_score, bad.execution_policy_score)
        self.assertGreater(fitness_ranking_key(good), fitness_ranking_key(bad))

    def test_average_improvement_with_one_bad_cell_fails_robustness_gate(self) -> None:
        profile = FitnessProfile.from_task(_task())
        incumbent = _evaluation("cell-old", 0.4, (0.0,) * 8)
        incumbent.metrics["targets"] = _cells(0.0)
        candidate = _evaluation("cell-new", 0.5, (0.1,) * 8)
        candidate.metrics["targets"] = _cells(0.2, bad_cell=-0.01)
        assessment = build_fitness_assessment(candidate, incumbent, {}, profile)
        self.assertGreater(assessment.primary_delta, 0)
        self.assertLess(assessment.robustness_min_cell_delta, 0)
        self.assertFalse(assessment.robustness_pass)

    def test_formal_gate_uses_frozen_baseline_and_requires_point_and_uq(self) -> None:
        candidate = _evaluation("formal", 0.2, (0.1,) * 14)
        candidate.metrics.update(
            {
                "objective_weight_coverage": 0.96,
                "formal_score": 0.1,
                "formal_score_lcb": 0.02,
                "formal_valid_three_day_start_count": 10,
                "formal_baseline_uq_artifact_digest": "9" * 64,
                "paired_interval_score_delta_ucb": 0.04,
                "targets": [
                    {
                        **cell,
                        "n": 80,
                        "paired_block_count": 14,
                        "sample_execution_coverage": 0.95,
                        "interval_coverage_lcb": 0.86,
                    }
                    for cell in _cells(0.01)
                ],
            }
        )
        baseline = {
            "artifact_digest": "9" * 64,
            "policy_id": "cellwise_time_block_calibrated_residual@1",
            "alpha": 0.1,
        }
        result = build_formal_fitness_assessment(
            candidate, baseline, FitnessProfile.from_task(_task())
        )
        self.assertEqual(result.status, "passed")
        self.assertTrue(result.point_pass)
        self.assertTrue(result.uq_pass)

        candidate.metrics["formal_score_lcb"] = -0.001
        candidate.metrics["paired_interval_score_delta_ucb"] = -1.0
        failed = build_formal_fitness_assessment(
            candidate, baseline, FitnessProfile.from_task(_task())
        )
        self.assertEqual(failed.status, "rejected")
        self.assertFalse(failed.point_pass)
        self.assertIn("formal_score_lcb_nonpositive", failed.failures)

    def test_missing_formal_evidence_is_inconclusive(self) -> None:
        candidate = _evaluation("formal-missing", 0.2, (0.1,) * 14)
        result = build_formal_fitness_assessment(
            candidate,
            {
                "artifact_digest": "9" * 64,
                "policy_id": "cellwise_time_block_calibrated_residual@1",
                "alpha": 0.1,
            },
            FitnessProfile.from_task(_task()),
        )
        self.assertEqual(result.status, "inconclusive")
        self.assertFalse(result.formal_confirmation)


if __name__ == "__main__":
    unittest.main()
