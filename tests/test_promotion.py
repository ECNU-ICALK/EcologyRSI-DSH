from __future__ import annotations

import unittest

from ecologyrsi_dsh.core.models import Evaluation, digest
from ecologyrsi_dsh.evolution.promotion import (
    assess_promotion_improvement,
    build_promotion_block_evidence,
)


TARGETS = ("air_temperature", "relative_humidity", "co2_concentration")


def _block_evidence(scores: tuple[float, ...]) -> dict:
    blocks = []
    for index, score in enumerate(scores, start=1):
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
        blocks.append({"block_id": f"{index:064x}", "cells": cells})
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


def _evaluation(
    candidate: str,
    score: float,
    *,
    version: str | None = None,
    baseline_digest: str = "b" * 64,
    block_scores: tuple[float, ...] = (),
) -> Evaluation:
    metrics = {}
    if version is not None:
        metrics = {
            "objective_aggregation_version": version,
            "objective_target_weights": {target: 1 / 3 for target in TARGETS},
            "objective_horizons": [1],
            "baseline_profile_digest": baseline_digest,
            "evaluation_index_digest": "c" * 64,
            "dataset_digest": "d" * 64,
            "split_manifest_digest_sha256": "e" * 64,
            "promotion_block_evidence": _block_evidence(block_scores),
        }
    return Evaluation(
        evaluation_id=f"evaluation:{candidate}",
        run_id="run:promotion",
        candidate_id=f"candidate:{candidate}",
        score=score,
        passed=True,
        metrics=metrics,
        evaluator_digest="f" * 64,
    )


class PromotionPolicyTests(unittest.TestCase):
    def test_three_or_seven_blocks_are_insufficient_and_fail_closed(self) -> None:
        for count in (3, 7):
            incumbent = _evaluation(
                f"old-{count}", 0.4, version="weighted_task_skill_reward@2", block_scores=(0.0,) * count
            )
            candidate = _evaluation(
                f"new-{count}", 0.5, version="weighted_task_skill_reward@2", block_scores=(0.1,) * count
            )
            assessment = assess_promotion_improvement(
                candidate,
                incumbent,
                execution_protocol="dsh_native_plugin_evolution@1",
            )
            self.assertFalse(assessment["improved"])
            self.assertEqual(assessment["reason_code"], "insufficient_evidence")

    def test_adaptively_reused_selection_rows_cannot_produce_confirmatory_promotion(self) -> None:
        incumbent = _evaluation(
            "old-adaptive", 0.4, version="weighted_task_skill_reward@2", block_scores=(0.0,) * 8
        )
        candidate = _evaluation(
            "new-adaptive", 0.5, version="weighted_task_skill_reward@2", block_scores=(0.1,) * 8
        )
        assessment = assess_promotion_improvement(
            candidate,
            incumbent,
            execution_protocol="dsh_native_plugin_evolution@1",
        )
        self.assertEqual(assessment["evidence_class"], "exploratory_adaptive_data")
        self.assertTrue(assessment["selection_only"])
        self.assertFalse(assessment["validated"])
        self.assertFalse(assessment["confirmed"])

    def test_legacy_confidence_pass_cannot_validate_a_new_protocol_candidate(self) -> None:
        incumbent = _evaluation(
            "old-legacy-field", 0.4, version="weighted_task_skill_reward@2", block_scores=(0.0,) * 8
        )
        candidate = _evaluation(
            "new-legacy-field", 0.5, version="weighted_task_skill_reward@2", block_scores=(0.1,) * 8
        )
        candidate.metrics["confidence_pass"] = True
        assessment = assess_promotion_improvement(
            candidate,
            incumbent,
            execution_protocol="dsh_native_plugin_evolution@1",
        )
        self.assertFalse(assessment["validated"])
        self.assertFalse(assessment["confirmed"])

    def test_legacy_evaluations_keep_strict_epsilon_policy(self) -> None:
        incumbent = _evaluation("old", 0.8)

        rejected = assess_promotion_improvement(
            _evaluation("tiny", 0.8 + 0.5e-12), incumbent
        )
        approved = assess_promotion_improvement(
            _evaluation("larger", 0.8 + 2e-12), incumbent
        )

        self.assertFalse(rejected["improved"])
        self.assertTrue(approved["improved"])
        self.assertEqual(rejected["minimum_score_delta"], 1e-12)

    def test_v2_requires_practical_score_improvement(self) -> None:
        blocks = (0.0, 0.0, 0.0)
        incumbent = _evaluation("old", 0.4, version="weighted_task_skill_reward@2", block_scores=blocks)
        assessment = assess_promotion_improvement(
            _evaluation("new", 0.404, version="weighted_task_skill_reward@2", block_scores=(0.1, 0.1, 0.1)),
            incumbent,
        )

        self.assertFalse(assessment["improved"])
        self.assertEqual(assessment["reason_code"], "below_practical_delta")
        self.assertEqual(assessment["minimum_score_delta"], 0.005)

    def test_v2_fails_closed_on_baseline_profile_mismatch(self) -> None:
        incumbent = _evaluation(
            "old", 0.4, version="weighted_task_skill_reward@2", block_scores=(0.0,) * 4
        )
        assessment = assess_promotion_improvement(
            _evaluation(
                "new",
                0.5,
                version="weighted_task_skill_reward@2",
                baseline_digest="a" * 64,
                block_scores=(0.1,) * 4,
            ),
            incumbent,
        )

        self.assertFalse(assessment["comparable"])
        self.assertFalse(assessment["improved"])
        self.assertEqual(assessment["reason_code"], "incompatible_scoring_contract")

    def test_v2_requires_complete_digest_and_objective_contracts(self) -> None:
        incumbent = _evaluation(
            "old", 0.4, version="weighted_task_skill_reward@2", block_scores=(0.0,) * 4
        )
        missing_digest = _evaluation(
            "missing", 0.5, version="weighted_task_skill_reward@2", block_scores=(0.1,) * 4
        )
        incumbent.metrics.pop("dataset_digest")
        missing_digest.metrics.pop("dataset_digest")
        self.assertFalse(
            assess_promotion_improvement(missing_digest, incumbent)["comparable"]
        )

        incumbent = _evaluation(
            "old-weights", 0.4, version="weighted_task_skill_reward@2", block_scores=(0.0,) * 4
        )
        changed_weights = _evaluation(
            "new-weights", 0.5, version="weighted_task_skill_reward@2", block_scores=(0.1,) * 4
        )
        evidence = changed_weights.metrics["promotion_block_evidence"]
        evidence["target_weights"] = {
            "air_temperature": 1.0,
            "relative_humidity": 0.0,
            "co2_concentration": 0.0,
        }
        evidence["evidence_digest"] = digest(
            {key: value for key, value in evidence.items() if key != "evidence_digest"}
        )
        assessment = assess_promotion_improvement(changed_weights, incumbent)
        self.assertFalse(assessment["comparable"])
        self.assertFalse(assessment["improved"])

    def test_v2_uses_deterministic_paired_block_confidence_interval(self) -> None:
        incumbent = _evaluation(
            "old", 0.4, version="weighted_task_skill_reward@2", block_scores=(0.0,) * 6
        )
        positive = assess_promotion_improvement(
            _evaluation(
                "positive",
                0.42,
                version="weighted_task_skill_reward@2",
                block_scores=(0.02,) * 6,
            ),
            incumbent,
        )
        unstable = assess_promotion_improvement(
            _evaluation(
                "unstable",
                0.42,
                version="weighted_task_skill_reward@2",
                block_scores=(0.08, 0.08, 0.08, -0.05, -0.05, -0.05),
            ),
            incumbent,
        )

        self.assertTrue(positive["improved"])
        self.assertGreater(positive["confidence_interval_95"][0], 0.0)
        self.assertFalse(unstable["improved"])
        self.assertEqual(unstable["reason_code"], "confidence_interval_crosses_zero")
        self.assertLessEqual(unstable["confidence_interval_95"][0], 0.0)

    def test_legacy_evaluator_mismatch_and_v2_block_mismatch_fail_closed(self) -> None:
        legacy_incumbent = _evaluation("legacy-old", 0.4)
        legacy_candidate = _evaluation("legacy-new", 0.5)
        object.__setattr__(legacy_candidate, "evaluator_digest", "a" * 64)
        self.assertFalse(
            assess_promotion_improvement(legacy_candidate, legacy_incumbent)[
                "comparable"
            ]
        )

        incumbent = _evaluation(
            "old", 0.4, version="weighted_task_skill_reward@2", block_scores=(0.0,) * 4
        )
        candidate = _evaluation(
            "new", 0.5, version="weighted_task_skill_reward@2", block_scores=(0.1,) * 4
        )
        evidence = candidate.metrics["promotion_block_evidence"]
        evidence["blocks"].pop()
        evidence["block_count"] = len(evidence["blocks"])
        evidence["evidence_digest"] = digest(
            {key: value for key, value in evidence.items() if key != "evidence_digest"}
        )
        assessment = assess_promotion_improvement(candidate, incumbent)
        self.assertFalse(assessment["comparable"])
        self.assertEqual(assessment["reason_code"], "mismatched_block_identities")

        short_ids = _evaluation(
            "short-ids", 0.5, version="weighted_task_skill_reward@2", block_scores=(0.1,) * 4
        )
        short_incumbent = _evaluation(
            "short-old", 0.4, version="weighted_task_skill_reward@2", block_scores=(0.0,) * 4
        )
        for evaluation in (short_ids, short_incumbent):
            evidence = evaluation.metrics["promotion_block_evidence"]
            for index, block in enumerate(evidence["blocks"]):
                block["block_id"] = str(index)
            evidence["evidence_digest"] = digest(
                {
                    key: value
                    for key, value in evidence.items()
                    if key != "evidence_digest"
                }
            )
        self.assertFalse(
            assess_promotion_improvement(short_ids, short_incumbent)["comparable"]
        )

    def test_block_evidence_uses_origin_and_sufficient_statistics(self) -> None:
        rows = []
        for target in TARGETS:
            rows.append(
                {
                    "target": target,
                    "horizon_hours": 1,
                    "origin_timestamp": 23,
                    "target_timestamp": 24,
                    "observed": 2.0,
                    "predicted": 1.0,
                    "baseline": 0.0,
                    "normalization_scale": 2.0,
                    "sample_execution_status": "succeeded",
                }
            )
            rows.append(
                {
                    "target": target,
                    "horizon_hours": 1,
                    "origin_timestamp": 24,
                    "target_timestamp": 25,
                    "observed": 2.0,
                    "predicted": 0.0,
                    "baseline": 0.0,
                    "normalization_scale": 2.0,
                    "sample_execution_status": "failed",
                    "scoring_fallback": "failed",
                }
            )
        evidence = build_promotion_block_evidence(
            rows,
            horizons=(1,),
            target_weights={target: 1 / 3 for target in TARGETS},
            dataset_digest="d" * 64,
            split_manifest_digest_sha256="e" * 64,
        )

        self.assertEqual(evidence["block_count"], 2)
        self.assertLessEqual(evidence["block_count"], 128)
        first_cell = evidence["blocks"][0]["cells"][0]
        self.assertEqual(first_cell["eligible"], 1)
        self.assertEqual(first_cell["succeeded"], 1)
        self.assertAlmostEqual(first_cell["candidate_squared_error_sum"], 0.25)
        second_cell = evidence["blocks"][1]["cells"][0]
        self.assertEqual(second_cell["eligible"], 1)
        self.assertEqual(second_cell["succeeded"], 0)


if __name__ == "__main__":
    unittest.main()
