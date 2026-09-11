from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import ecologyrsi_dsh
from ecologyrsi_dsh.core.models import Evaluation, TaskManifest, canonical_json, digest
from ecologyrsi_dsh.evaluators.fitness import (
    EXPLORATORY_EVIDENCE_CLASS,
    FitnessProfile,
    assess_generation_selection,
    build_fitness_assessment,
    build_formal_fitness_assessment,
    build_moving_block_resample_indices,
    fitness_ranking_key,
)
from ecologyrsi_dsh.evaluators.objectives import OBJECTIVE_AGGREGATION_VERSION
from ecologyrsi_dsh.evolution.promotion import (
    PROMOTION_BLOCK_EVIDENCE_VERSION,
    PROMOTION_SCORE_DEFINITION,
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
                "horizon_hours": horizon,
                "eligible": 1,
                "succeeded": 1,
                "candidate_squared_error_sum": (1.0 - score) ** 2,
                "baseline_squared_error_sum": 1.0,
                "normalized_reward_sum": score,
            }
            for target in TARGETS
            for horizon in HORIZONS
        ]
        blocks.append(
            {
                "block_id": digest({"calendar_day": index}),
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
            "objective_aggregation_version": OBJECTIVE_AGGREGATION_VERSION,
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
    def test_default_sample_budget_can_reach_every_selection_gate(self) -> None:
        self.assertEqual(
            FitnessProfile.from_task(_task()).minimum_balanced_samples_per_update(),
            1_521,
        )
        self.assertEqual(
            FitnessProfile.from_task(_task()).minimum_balanced_origins_per_update(),
            169,
        )
        self.assertEqual(FitnessProfile.from_task(_task()).prediction_cell_count, 9)

    def test_frozen_profile_digest_must_match_the_profile_in_use(self) -> None:
        profile = FitnessProfile()
        task_data = _task().to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "fitness_profile": profile.to_dict(),
            "fitness_profile_digest": "0" * 64,
        }

        with self.assertRaisesRegex(ValueError, "does not match"):
            FitnessProfile.from_task(TaskManifest.from_dict(task_data))

        task_data["metadata"]["fitness_profile_digest"] = profile.profile_digest
        self.assertEqual(
            FitnessProfile.from_task(TaskManifest.from_dict(task_data)), profile
        )

    def test_missing_or_sparse_block_evidence_fails_closed(self) -> None:
        profile = FitnessProfile.from_task(_task())
        candidate = _evaluation("missing-blocks", 0.2, (0.1,) * 8)
        candidate = replace(candidate, metrics={k: v for k, v in candidate.metrics.items() if k != "promotion_block_evidence"})

        missing = build_fitness_assessment(candidate, None, {}, profile)

        self.assertFalse(missing.validity_pass)
        self.assertIn("promotion_block_evidence_invalid", missing.validity_failures)
        self.assertTrue(
            any(
                failure.startswith("cell_blocks_insufficient:")
                for failure in missing.validity_failures
            )
        )

        sparse = _evaluation("sparse-blocks", 0.2, (0.1,) * 7)
        sparse_assessment = build_fitness_assessment(sparse, None, {}, profile)
        self.assertFalse(sparse_assessment.validity_pass)
        self.assertIn("paired_blocks_insufficient", sparse_assessment.validity_failures)

    def test_missing_objective_cells_serialize_finite_failure_diagnostics(self) -> None:
        profile = FitnessProfile.from_task(_task())
        candidate = _evaluation("missing-cells", 0.2, (0.1,) * 8)
        candidate = replace(candidate, metrics={k: v for k, v in candidate.metrics.items() if k != "targets"})

        assessment = build_fitness_assessment(candidate, None, {}, profile)
        payload = assessment.to_dict()

        self.assertFalse(assessment.validity_pass)
        self.assertIsNone(payload["robustness_min_cell_delta"])
        self.assertIsNone(payload["robustness_lower_quartile_cell_delta"])
        canonical_json(payload)

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
        evidence = _block_evidence((0.0,) * 8, (0, 1, 2, 10, 11, 12, 20, 21))
        incumbent, candidate = (
            replace(item, metrics={**item.metrics, "promotion_block_evidence": evidence})
            for item in (incumbent, candidate)
        )
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
        regressed = replace(regressed, metrics={**regressed.metrics, "targets": _cells(0.1, bad_cell=-0.1)})
        robust = _evaluation("slow-robust", 0.55, (0.05,) * 8)
        bad = build_fitness_assessment(
            regressed, incumbent, {"latency_ms": 1}, profile
        )
        good = build_fitness_assessment(
            robust, incumbent, {"latency_ms": 10_000}, profile
        )
        self.assertGreater(fitness_ranking_key(good), fitness_ranking_key(bad))

    def test_primary_score_breaks_ties_before_operational_fitness(self) -> None:
        profile = FitnessProfile.from_task(_task())
        higher = _evaluation("a-higher", 0.3, (0.1,) * 8)
        lower = _evaluation("z-lower", 0.2, (0.1,) * 8)
        higher, lower = (
            replace(item, metrics={**item.metrics, "targets": _cells(0.1)})
            for item in (higher, lower)
        )
        higher_assessment = build_fitness_assessment(higher, None, {}, profile)
        lower_assessment = build_fitness_assessment(lower, None, {}, profile)

        self.assertEqual(
            higher_assessment.robustness_min_cell_delta,
            lower_assessment.robustness_min_cell_delta,
        )
        self.assertGreater(
            fitness_ranking_key(higher_assessment),
            fitness_ranking_key(lower_assessment),
        )

    def test_execution_policy_quality_is_a_separate_lower_order_fitness_track(self) -> None:
        profile = FitnessProfile.from_task(_task())
        incumbent = _evaluation("policy-incumbent", 0.4, (0.0,) * 8)
        reliable = _evaluation("policy-reliable", 0.5, (0.1,) * 8)
        noisy = _evaluation("policy-noisy", 0.5, (0.1,) * 8)
        reliable = replace(reliable, metrics={**reliable.metrics, "sample_execution": {
            "eligible_examples": 10,
            "failed_examples": 0,
            "retry_count": 0,
            "repair_count": 0,
            "critic_outcome_counts": {"accepted": 10},
        }})
        noisy = replace(noisy, metrics={**noisy.metrics, "sample_execution": {
            "eligible_examples": 10,
            "failed_examples": 0,
            "retry_count": 5,
            "repair_count": 0,
            "critic_outcome_counts": {"accepted": 10, "rejected": 10},
        }})

        good = build_fitness_assessment(reliable, incumbent, {}, profile)
        bad = build_fitness_assessment(noisy, incumbent, {}, profile)

        self.assertEqual(good.primary_score, bad.primary_score)
        self.assertGreater(good.execution_policy_score, bad.execution_policy_score)
        self.assertGreater(fitness_ranking_key(good), fitness_ranking_key(bad))

    def test_average_improvement_with_one_bad_cell_fails_robustness_gate(self) -> None:
        profile = FitnessProfile.from_task(_task())
        incumbent = _evaluation("cell-old", 0.4, (0.0,) * 8)
        incumbent = replace(incumbent, metrics={**incumbent.metrics, "targets": _cells(0.0)})
        candidate = _evaluation("cell-new", 0.5, (0.1,) * 8)
        candidate = replace(candidate, metrics={**candidate.metrics, "targets": _cells(0.2, bad_cell=-0.01)})
        assessment = build_fitness_assessment(candidate, incumbent, {}, profile)
        self.assertGreater(assessment.primary_delta, 0)
        self.assertLess(assessment.robustness_min_cell_delta, 0)
        self.assertFalse(assessment.robustness_pass)

    def test_formal_gate_uses_frozen_baseline_and_requires_point_and_uq(self) -> None:
        candidate = _evaluation("formal", 0.2, (0.1,) * 14)
        candidate = replace(candidate, metrics={**candidate.metrics, **{
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
            }})
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

        candidate = replace(candidate, metrics={**candidate.metrics, "formal_score_lcb": -0.001})
        candidate = replace(candidate, metrics={**candidate.metrics, "paired_interval_score_delta_ucb": -1.0})
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

    def test_formal_evidence_fields_have_no_production_writer_yet(self) -> None:
        """Pin the formal layer's unfed state so a fixture cannot hide it.

        ``test_formal_gate_uses_frozen_baseline_and_requires_point_and_uq`` above
        hand-builds every metric the formal gate reads, which made a fully
        unimplemented layer look exercised. None of the five top-level required
        metrics, nor the per-cell ``interval_coverage_lcb``, is written anywhere
        in ``src/``: the formal path needs holdout scoring and calibrated
        prediction intervals that do not exist yet.

        Fail-closed is therefore the correct behaviour, and this test asserts it
        directly. When a real producer lands, this test is what will fail and
        tell whoever wrote it to come update the claim.
        """

        source = Path(ecologyrsi_dsh.__file__).parent
        unwritten = (
            "formal_score",
            "formal_score_lcb",
            "formal_valid_three_day_start_count",
            "formal_baseline_uq_artifact_digest",
            "paired_interval_score_delta_ucb",
            "interval_coverage_lcb",
        )
        producers = {name: [] for name in unwritten}
        for path in sorted(source.rglob("*.py")):
            if path.name == "fitness.py":
                continue  # The reader, not a writer.
            text = path.read_text(encoding="utf-8")
            for name in unwritten:
                if f'"{name}"' in text:
                    producers[name].append(path.name)
        self.assertEqual(
            {name: files for name, files in producers.items() if files}, {}
        )

        # And so the gate refuses rather than assuming the best.
        candidate = _evaluation("formal-unfed", 0.2, (0.1,) * 14)
        result = build_formal_fitness_assessment(
            candidate,
            {
                "artifact_digest": "9" * 64,
                "policy_id": "cellwise_time_block_calibrated_residual@1",
                "alpha": 0.1,
            },
            FitnessProfile.from_task(_task()),
        )
        self.assertFalse(result.formal_confirmation)
        self.assertEqual(result.status, "inconclusive")


if __name__ == "__main__":
    unittest.main()
