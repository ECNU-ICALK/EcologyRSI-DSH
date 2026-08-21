from __future__ import annotations

import json
import math
import os
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.datasets import DatasetRegistry, DatasetSeries
from ecologyrsi_dsh.evaluation import (
    EXOGENOUS_RIDGE_MODEL_ID,
    GREENHOUSE_EVALUATOR_ID,
    GREENHOUSE_MULTIHORIZON_EVALUATOR_ID,
    GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID,
    GREENHOUSE_ROLLING_PREDICTOR_ID,
    HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
    TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
    TOY_EVALUATOR_ID,
    EvaluationBundle,
    EvaluatorRegistry,
)
from ecologyrsi_dsh.greenhouse import FeatureSpec
from ecologyrsi_dsh.knowledge.algorithms import (
    compile_algorithm_spec,
    resolve_predictor_adoption,
)
from ecologyrsi_dsh.models import (
    Candidate,
    Evaluation,
    ModelArtifact,
    Proposal,
    TaskManifest,
)
from ecologyrsi_dsh.splits import IndexRange
from ecologyrsi_dsh.evaluators.registry import _aggregate_greenhouse_objective

DATASET_ID = "agc_cucumber_2018"
SPLIT_DIGEST = "s" * 64


class _DatasetStub:
    def __init__(self, series: DatasetSeries) -> None:
        self._series = series

    def series(
        self,
        dataset_id: str,
        episode_id: str | None = None,
        *,
        expected_dataset_digest: str | None = None,
        expected_split_manifest_digest: str | None = None,
    ) -> DatasetSeries:
        if dataset_id != DATASET_ID:
            raise KeyError(dataset_id)
        if episode_id is not None and episode_id != self._series.episode_id:
            raise KeyError(episode_id)
        if (
            expected_dataset_digest is not None
            and expected_dataset_digest != self._series.digest
        ):
            raise ValueError("数据集快照发生漂移")
        if (
            expected_split_manifest_digest is not None
            and expected_split_manifest_digest
            != self._series.split_manifest_digest_sha256
        ):
            raise ValueError("时间分区快照发生漂移")
        return self._series


class _JudgeGatewayStub:
    def __init__(self, parameter_override: dict) -> None:
        self.parameter_override = parameter_override

    def judge(self, model_id: str, proposal: dict, metrics: dict) -> dict:
        return {
            "accepted": True,
            "guidance": "将建议传入下一轮。",
            "parameter_override": self.parameter_override,
        }


def _series(
    *, tail_marker: float = 0.0, feedback_missing: bool = False
) -> DatasetSeries:
    timestamps = tuple(range(100))
    temperature: list[float | None] = [
        21.0 + 0.03 * index + 0.2 * math.sin(index / 4) for index in timestamps
    ]
    humidity: list[float | None] = [
        70.0 + 2.0 * math.sin(index / 5) for index in timestamps
    ]
    co2: list[float | None] = [
        720.0 + 15.0 * math.sin(index / 3) for index in timestamps
    ]
    temperature[5] = None
    temperature[35] = None
    temperature[40] = float("nan")
    if feedback_missing:
        for index in range(31, 60):
            temperature[index] = None
    # The fit/feedback evaluator must not read the embargo row or any later
    # development/gate-tail values.  Variants deliberately disagree there.
    temperature[30] = tail_marker
    for values in (temperature, humidity, co2):
        for index in range(60, 100):
            values[index] = tail_marker
    return DatasetSeries(
        schema="ecologyrsi-dsh.dataset-series/1",
        dataset_id=DATASET_ID,
        domain_id="greenhouse_cucumber_2018",
        episode_id=f"{DATASET_ID}:TeamA",
        digest="d" * 64,
        timestamps=timestamps,
        values={
            "air_temperature": tuple(temperature),
            "relative_humidity": tuple(humidity),
            "co2_concentration": tuple(co2),
        },
        partitions={
            "training_fit": IndexRange(0, 30),
            "training_feedback": IndexRange(31, 60),
            "development": IndexRange(80, 80),
        },
        features={},
        split_manifest_digest_sha256=SPLIT_DIGEST,
    )


def _cohort_series() -> DatasetSeries:
    timestamps = tuple(range(220))
    temperature: list[float | None] = [
        22.0 + 0.015 * index + 0.3 * math.sin(index / 7) for index in timestamps
    ]
    humidity: list[float | None] = [
        68.0 + 1.5 * math.sin(index / 9) for index in timestamps
    ]
    co2: list[float | None] = [
        710.0 + 12.0 * math.sin(index / 5) for index in timestamps
    ]
    temperature[140] = None
    return DatasetSeries(
        schema="ecologyrsi-dsh.dataset-series/1",
        dataset_id=DATASET_ID,
        domain_id="greenhouse_cucumber_2018",
        episode_id=f"{DATASET_ID}:TeamA",
        digest="d" * 64,
        timestamps=timestamps,
        values={
            "air_temperature": tuple(temperature),
            "relative_humidity": tuple(humidity),
            "co2_concentration": tuple(co2),
        },
        partitions={
            "training_fit": IndexRange(0, 100),
            # Index 100 is an embargo row and must not enter model history.
            "training_feedback": IndexRange(101, 200),
            "development": IndexRange(200, 220),
        },
        features={},
        split_manifest_digest_sha256=SPLIT_DIGEST,
    )


def _task() -> TaskManifest:
    return TaskManifest(
        task_id="greenhouse-evaluation-test",
        objective="评估温室环境预测",
        domain_pack="greenhouse_cucumber_2018",
        visible_datasets=(DATASET_ID,),
        budget={"max_candidates": 1},
        metadata={
            "evaluator_id": GREENHOUSE_EVALUATOR_ID,
            "prediction_model_id": GREENHOUSE_ROLLING_PREDICTOR_ID,
            "episode_id": f"{DATASET_ID}:TeamA",
            "dataset_digest": "d" * 64,
            "split_manifest_digest": SPLIT_DIGEST,
        },
    )


def _candidate_and_proposal() -> tuple[Candidate, Proposal]:
    proposal = Proposal(
        proposal_id="proposal:evaluation:1",
        run_id="run:evaluation",
        generation=0,
        title="温室环境参数方案",
        changes={"blend": 0.5, "window": 3, "bias_scale": 0.5},
    )
    candidate = Candidate(
        candidate_id="candidate:evaluation:1",
        run_id=proposal.run_id,
        proposal_id=proposal.proposal_id,
        generation=0,
    )
    return candidate, proposal


def _nested_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _nested_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _nested_values(item)
    else:
        yield value


class GreenhouseEvaluationTests(unittest.TestCase):
    def test_weighted_objective_includes_non_equal_task_counts_and_missing_cells(self):
        rows = [
            {
                "target": "air_temperature",
                "horizon_hours": 1,
                "n": 10,
                "skill_score": 0.2,
                "normalized_mean_reward": 0.1,
            },
            {
                "target": "air_temperature",
                "horizon_hours": 6,
                "n": 10,
                "skill_score": 0.4,
                "normalized_mean_reward": 0.3,
            },
            {
                "target": "relative_humidity",
                "horizon_hours": 1,
                "n": 1,
                "skill_score": 0.6,
                "normalized_mean_reward": 0.5,
            },
            {
                "target": "relative_humidity",
                "horizon_hours": 6,
                "n": 1,
                "skill_score": 0.8,
                "normalized_mean_reward": 0.7,
            },
            {
                "target": "co2_concentration",
                "horizon_hours": 1,
                "n": 1,
                "skill_score": 0.0,
                "normalized_mean_reward": 0.0,
            },
        ]
        aggregate = _aggregate_greenhouse_objective(rows, (1, 6))
        self.assertAlmostEqual(aggregate["weighted_skill_score"], 1.0 / 6.0)
        self.assertAlmostEqual(
            aggregate["weighted_normalized_mean_reward"], 0.6 / 6.0
        )
        self.assertEqual(aggregate["objective_task_count"], 6)
        self.assertEqual(aggregate["objective_missing_task_count"], 1)
        self.assertAlmostEqual(aggregate["objective_weight_coverage"], 5.0 / 6.0)

    def test_weighted_objective_honors_target_weights(self):
        rows = [
            {
                "target": "air_temperature",
                "horizon_hours": 1,
                "n": 1,
                "skill_score": 1.0,
                "normalized_mean_reward": 1.0,
            },
            {
                "target": "relative_humidity",
                "horizon_hours": 1,
                "n": 1,
                "skill_score": 0.0,
                "normalized_mean_reward": 0.0,
            },
            {
                "target": "co2_concentration",
                "horizon_hours": 1,
                "n": 1,
                "skill_score": 0.0,
                "normalized_mean_reward": 0.0,
            },
        ]
        with patch.dict(
            "ecologyrsi_dsh.evaluators.registry.GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS",
            {
                "air_temperature": 0.6,
                "relative_humidity": 0.3,
                "co2_concentration": 0.1,
            },
            clear=True,
        ):
            aggregate = _aggregate_greenhouse_objective(rows, (1,))
        self.assertAlmostEqual(aggregate["weighted_skill_score"], 0.6)
        self.assertAlmostEqual(aggregate["weighted_normalized_mean_reward"], 0.6)

    def test_weighted_objective_rejects_invalid_weights_and_horizons(self):
        with patch.dict(
            "ecologyrsi_dsh.evaluators.registry.GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS",
            {"air_temperature": -1.0},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                _aggregate_greenhouse_objective([], (1,))
        with self.assertRaises(ValueError):
            _aggregate_greenhouse_objective([], (1, 1))

    def evaluate(self, series: DatasetSeries):
        candidate, proposal = _candidate_and_proposal()
        registry = EvaluatorRegistry(_DatasetStub(series))  # type: ignore[arg-type]
        return registry.evaluate(_task(), candidate, proposal)

    def evaluate_parameters(
        self,
        series: DatasetSeries,
        changes: dict,
        *,
        suffix: str,
        prediction_model_id: str = GREENHOUSE_ROLLING_PREDICTOR_ID,
        evaluator_id: str = GREENHOUSE_EVALUATOR_ID,
    ):
        task_data = _task().to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "prediction_model_id": prediction_model_id,
            "evaluator_id": evaluator_id,
        }
        proposal = Proposal(
            proposal_id=f"proposal:cohort:{suffix}",
            run_id="run:cohort",
            generation=0,
            title="公共评测 cohort 测试",
            changes=changes,
        )
        candidate = Candidate(
            candidate_id=f"candidate:cohort:{suffix}",
            run_id=proposal.run_id,
            proposal_id=proposal.proposal_id,
            generation=0,
        )
        return EvaluatorRegistry(_DatasetStub(series)).evaluate_scientific(  # type: ignore[arg-type]
            TaskManifest.from_dict(task_data), candidate, proposal
        )

    @staticmethod
    def baseline_summary(metrics: dict) -> tuple:
        return tuple(
            (
                item["target"],
                item["horizon_hours"],
                item["n"],
                item["eligible_rows"],
                item["missing_or_nonfinite_rows"],
                item["baseline_mae"],
                item["baseline_rmse"],
                item["baseline_normalized_rmse"],
            )
            for item in metrics["targets"]
        )

    def test_missing_values_are_skipped_and_reported_without_preview_misalignment(
        self,
    ) -> None:
        bundle = self.evaluate(_series())
        metrics = bundle.evaluation.metrics
        targets = {item["target"]: item for item in metrics["targets"]}

        self.assertEqual(bundle.artifact.metrics["air_temperature_n"], 25)
        self.assertEqual(
            bundle.artifact.metrics["air_temperature_missing_or_nonfinite_rows"], 2
        )
        self.assertEqual(targets["air_temperature"]["eligible_rows"], 12)
        self.assertEqual(targets["air_temperature"]["n"], 12)
        self.assertEqual(targets["air_temperature"]["missing_or_nonfinite_rows"], 0)
        self.assertEqual(metrics["missing_or_nonfinite_rows"], 0)
        self.assertEqual(bundle.artifact.metrics["training_partition_rows"], 30)
        self.assertEqual(bundle.artifact.metrics["fit_passes_completed"], 1)
        self.assertFalse(bundle.artifact.metrics["iterative_epoch_training"])
        self.assertEqual(
            bundle.artifact.metrics["training_eligible_examples"],
            bundle.artifact.metrics["training_used_examples"]
            + bundle.artifact.metrics["training_skipped_examples"],
        )
        self.assertEqual(metrics["evaluation_partition_rows"], 29)
        self.assertEqual(metrics["evaluation_eligible_examples"], metrics["n"])
        self.assertEqual(metrics["evaluation_used_examples"], metrics["n"])
        self.assertTrue(math.isfinite(bundle.evaluation.score))

        preview = metrics["prediction_preview"]
        expected_targets = (
            "air_temperature",
            "relative_humidity",
            "co2_concentration",
        )
        preview_counts = {
            target: sum(row["target"] == target for row in preview)
            for target in expected_targets
        }
        self.assertEqual({row["target"] for row in preview}, set(expected_targets))
        self.assertEqual(preview_counts, {target: 12 for target in expected_targets})
        self.assertLessEqual(len(preview), 48)
        self.assertTrue(all(count <= 16 for count in preview_counts.values()))
        self.assertTrue(all(31 <= row["timestamp"] < 60 for row in preview))
        self.assertTrue(all(math.isfinite(row["observed"]) for row in preview))
        self.assertTrue(all(math.isfinite(row["predicted"]) for row in preview))
        self.assertEqual(
            [row["target"] for row in preview],
            [
                target
                for target in expected_targets
                for _ in range(preview_counts[target])
            ],
        )
        for target in expected_targets:
            timestamps = [
                row["timestamp"] for row in preview if row["target"] == target
            ]
            self.assertEqual(timestamps, sorted(timestamps))
        air_timestamps = {
            row["timestamp"] for row in preview if row["target"] == "air_temperature"
        }
        self.assertTrue({35, 36, 40, 41}.isdisjoint(air_timestamps))

    def test_split_manifest_digest_uses_explicit_sha256_field(self) -> None:
        metrics = self.evaluate(_series()).evaluation.metrics
        self.assertEqual(metrics["split_manifest_digest_sha256"], SPLIT_DIGEST)
        self.assertNotIn("split_manifest_digest", metrics)

    def test_rolling_windows_share_feedback_cohort_and_baseline(self) -> None:
        series = _cohort_series()
        shortest = self.evaluate_parameters(
            series,
            {"blend": 0.5, "window": 1, "bias_scale": 0.5},
            suffix="rolling-1",
        ).evaluation.metrics
        longest = self.evaluate_parameters(
            series,
            {"blend": 0.5, "window": 48, "bias_scale": 0.5},
            suffix="rolling-48",
        ).evaluation.metrics

        self.assertEqual(shortest["n"], longest["n"])
        self.assertEqual(shortest["missing_or_nonfinite_rows"], 2)
        self.assertEqual(
            shortest["missing_or_nonfinite_rows"],
            longest["missing_or_nonfinite_rows"],
        )
        self.assertEqual(
            shortest["baseline_normalized_rmse"],
            longest["baseline_normalized_rmse"],
        )
        self.assertEqual(
            self.baseline_summary(shortest), self.baseline_summary(longest)
        )
        self.assertEqual(
            shortest["evaluation_index_digest"],
            longest["evaluation_index_digest"],
        )
        self.assertEqual(
            shortest["baseline_metrics_digest"],
            longest["baseline_metrics_digest"],
        )
        self.assertEqual(len(shortest["evaluation_index_digest"]), 64)
        self.assertEqual(
            shortest["evaluation_cohort"]["minimum_history_hours"],
            48,
        )
        self.assertFalse(shortest["evaluation_cohort"]["embargo_rows_visible"])

    def test_ridge_history_steps_share_feedback_cohort_and_baseline(self) -> None:
        series = _cohort_series()
        shallow = self.evaluate_parameters(
            series,
            {"history_steps": 1, "ridge_alpha": 0.01, "residual_scale": 1.0},
            suffix="ridge-1",
            prediction_model_id=EXOGENOUS_RIDGE_MODEL_ID,
            evaluator_id=GREENHOUSE_MULTIHORIZON_EVALUATOR_ID,
        ).evaluation.metrics
        deep = self.evaluate_parameters(
            series,
            {"history_steps": 12, "ridge_alpha": 0.01, "residual_scale": 1.0},
            suffix="ridge-12",
            prediction_model_id=EXOGENOUS_RIDGE_MODEL_ID,
            evaluator_id=GREENHOUSE_MULTIHORIZON_EVALUATOR_ID,
        ).evaluation.metrics

        self.assertEqual(shallow["n"], deep["n"])
        self.assertEqual(
            shallow["baseline_normalized_rmse"],
            deep["baseline_normalized_rmse"],
        )
        self.assertEqual(self.baseline_summary(shallow), self.baseline_summary(deep))
        self.assertEqual(
            shallow["evaluation_index_digest"],
            deep["evaluation_index_digest"],
        )
        self.assertEqual(
            shallow["baseline_metrics_digest"],
            deep["baseline_metrics_digest"],
        )
        self.assertEqual(
            shallow["evaluation_cohort"]["minimum_history_steps"],
            12,
        )

    def test_evaluator_rejects_task_snapshot_drift_before_fitting(self) -> None:
        candidate, proposal = _candidate_and_proposal()
        task_data = _task().to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "dataset_digest": "0" * 64,
        }
        registry = EvaluatorRegistry(_DatasetStub(_series()))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "数据集快照发生漂移"):
            registry.evaluate(TaskManifest.from_dict(task_data), candidate, proposal)

    def test_embargo_development_and_gate_tail_do_not_affect_evaluation(self) -> None:
        ordinary = self.evaluate(_series(tail_marker=0.0))
        sentinel = self.evaluate(_series(tail_marker=999999.0))

        self.assertEqual(ordinary.evaluation.score, sentinel.evaluation.score)
        self.assertEqual(ordinary.evaluation.metrics, sentinel.evaluation.metrics)
        self.assertEqual(
            ordinary.artifact.learned_parameters, sentinel.artifact.learned_parameters
        )
        self.assertNotIn(999999.0, _nested_values(sentinel.evaluation.metrics))
        self.assertNotIn("development", sentinel.evaluation.metrics)
        self.assertNotIn("gate", sentinel.evaluation.metrics)

    def test_target_with_no_usable_feedback_rows_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "no usable rows for air_temperature"):
            self.evaluate(_series(feedback_missing=True))

    def test_judge_parameter_override_is_validated_and_persisted(self) -> None:
        candidate, proposal = _candidate_and_proposal()
        task_data = _task().to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "judge_model_id": "judge-main",
        }
        task = TaskManifest.from_dict(task_data)
        valid = EvaluatorRegistry(
            _DatasetStub(_series()),  # type: ignore[arg-type]
            _JudgeGatewayStub({"window": 4}),  # type: ignore[arg-type]
        ).evaluate(task, candidate, proposal)
        self.assertEqual(
            valid.evaluation.metrics["judge_parameter_override"],
            {"window": 4},
        )

        invalid = EvaluatorRegistry(
            _DatasetStub(_series()),  # type: ignore[arg-type]
            _JudgeGatewayStub({"window": 99}),  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(ValueError, "outside the allowed range"):
            invalid.evaluate(task, candidate, proposal)

    def test_rule_judge_persists_derived_scientific_pass_metric(self) -> None:
        candidate, proposal = _candidate_and_proposal()
        artifact = ModelArtifact(
            artifact_id="artifact:scientific-pass-fallback",
            run_id=candidate.run_id,
            candidate_id=candidate.candidate_id,
            model_id=GREENHOUSE_ROLLING_PREDICTOR_ID,
            dataset_digest="d" * 64,
            training_partition="training_fit",
            training_rows=1,
        )
        scientific = Evaluation(
            evaluation_id="evaluation:scientific-pass-fallback",
            run_id=candidate.run_id,
            candidate_id=candidate.candidate_id,
            score=-0.2,
            passed=False,
            metrics={"skill_score": -0.2},
            partition="training_feedback",
            evaluator_digest="fixed-evaluator",
            artifact_digest=artifact.digest,
        )
        registry = EvaluatorRegistry(_DatasetStub(_series()))  # type: ignore[arg-type]

        judged = registry.apply_judge(
            _task(),
            proposal,
            EvaluationBundle(artifact=artifact, evaluation=scientific),
        ).evaluation

        self.assertIs(judged.metrics["scientific_pass"], False)
        self.assertIs(judged.metrics["judge_accepted"], False)
        self.assertFalse(judged.passed)

    def test_predictor_and_evaluator_catalog_exposes_server_compatibility(self) -> None:
        registry = EvaluatorRegistry(_DatasetStub(_series()))  # type: ignore[arg-type]
        predictors = {item["id"]: item for item in registry.predictor_catalog()}
        evaluators = {item["id"]: item for item in registry.catalog()}
        self.assertEqual(registry.default_predictor(DATASET_ID), EXOGENOUS_RIDGE_MODEL_ID)
        self.assertEqual(
            registry.default_evaluator(DATASET_ID),
            GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID,
        )
        self.assertEqual(
            len(predictors[EXOGENOUS_RIDGE_MODEL_ID]["configuration_digest"]), 64
        )
        self.assertEqual(
            evaluators[GREENHOUSE_MULTIHORIZON_EVALUATOR_ID]["horizons_hours"],
            [1, 6, 24],
        )
        self.assertEqual(evaluators[TOY_EVALUATOR_ID]["prediction_task_count"], 1)
        self.assertEqual(
            evaluators[GREENHOUSE_EVALUATOR_ID]["prediction_task_count"], 3
        )
        self.assertEqual(
            evaluators[GREENHOUSE_MULTIHORIZON_EVALUATOR_ID][
                "prediction_task_count"
            ],
            9,
        )
        for evaluator in evaluators.values():
            self.assertEqual(
                evaluator["minimum_samples_per_update"],
                evaluator["prediction_task_count"],
            )
        self.assertEqual(
            evaluators[GREENHOUSE_MULTIHORIZON_EVALUATOR_ID]["prediction_model_ids"],
            [
                EXOGENOUS_RIDGE_MODEL_ID,
                TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
            ],
        )
        self.assertEqual(
            evaluators[GREENHOUSE_MULTIHORIZON_EVALUATOR_ID][
                "configuration_digest"
            ],
            "4ace89b5759e802b3c66980fc09f0ffe4238f46b96e3708a12a4a806d9c59624",
        )
        profile = registry.objective_profile(GREENHOUSE_MULTIHORIZON_EVALUATOR_ID)
        self.assertEqual(profile["objective_aggregation_version"], "weighted_task_skill_reward@2")
        self.assertEqual(profile["baseline_profile_version"], "fit_selected_persistence_or_seasonal_24h@1")
        self.assertEqual(profile["minimum_practical_score_delta"], 0.005)
        self.assertEqual(profile["confidence_method"], "paired_moving_block_bootstrap@1")
        self.assertEqual(profile["block_hours"], 24)
        self.assertEqual(profile["bootstrap_resamples"], 1000)
        self.assertEqual(profile["maximum_blocks"], 128)
        self.assertEqual(profile["minimum_paired_blocks"], 4)
        self.assertEqual(profile["normalization_scale_method"], "training_fit_std_floor@1")
        self.assertEqual(profile["objective_component_bound"], 1.0)
        self.assertEqual(profile["baseline_selection_tolerance"], 1e-12)
        self.assertEqual(
            evaluators[GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID][
                "prediction_model_ids"
            ],
            [
                EXOGENOUS_RIDGE_MODEL_ID,
                TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
                HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
            ],
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            registry.validate_binding(
                DATASET_ID,
                GREENHOUSE_MULTIHORIZON_EVALUATOR_ID,
                HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
            )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            registry.validate_binding(
                DATASET_ID,
                GREENHOUSE_MULTIHORIZON_EVALUATOR_ID,
                GREENHOUSE_ROLLING_PREDICTOR_ID,
            )

    def test_targetwise_ridge_runs_full_evaluator_with_co2_persistence(self) -> None:
        parameters = {
            "history_steps": 3,
            "ridge_alpha": 0.01,
            "air_temperature_residual_scale": 1.0,
            "relative_humidity_residual_scale": 1.0,
            "co2_concentration_residual_scale": 0.0,
        }

        bundle = self.evaluate_parameters(
            _cohort_series(),
            parameters,
            suffix="targetwise-ridge",
            prediction_model_id=TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
            evaluator_id=GREENHOUSE_MULTIHORIZON_EVALUATOR_ID,
        )

        self.assertEqual(
            bundle.artifact.model_id, TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
        )
        self.assertEqual(bundle.artifact.parameters, parameters)
        self.assertEqual(
            bundle.evaluation.metrics["prediction_model_id"],
            TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
        )
        co2_results = [
            item
            for item in bundle.evaluation.metrics["targets"]
            if item["target"] == "co2_concentration"
        ]
        for item in bundle.evaluation.metrics["targets"]:
            self.assertAlmostEqual(
                item["mean_reward"], item["baseline_mae"] - item["mae"]
            )
            self.assertEqual(
                item["normalized_mean_reward"] < 0,
                item["mean_reward"] < 0,
            )
            self.assertGreaterEqual(item["negative_reward_fraction"], 0.0)
            self.assertLessEqual(item["negative_reward_fraction"], 1.0)
        self.assertEqual(
            bundle.evaluation.metrics["reward_definition"],
            "absolute_error_improvement_vs_fit_selected_baseline@2",
        )
        self.assertTrue(bundle.evaluation.metrics["positive_reward_is_better"])
        self.assertEqual(
            bundle.evaluation.metrics["baseline_selection_partition"],
            "training_fit",
        )
        self.assertEqual(
            bundle.evaluation.metrics["baseline_profile_digest"],
            bundle.evaluation.metrics["baseline_profile"]["digest"],
        )
        self.assertEqual(
            bundle.evaluation.metrics["promotion_block_evidence"]["block_hours"],
            24,
        )
        self.assertTrue(bundle.sample_results)
        self.assertTrue(
            all(
                item["baseline_id"] in {"persistence", "seasonal_24h"}
                and "model_reference_baseline" in item
                and -1.0 <= item["normalized_reward"] <= 1.0
                for item in bundle.sample_results
            )
        )
        self.assertTrue(co2_results)
        self.assertTrue(
            all(math.isclose(item["skill_score"], 0.0) for item in co2_results)
        )
        self.assertEqual(bundle.evaluation.metrics["objective_task_count"], 9)
        self.assertEqual(bundle.evaluation.metrics["objective_missing_task_count"], 0)
        self.assertAlmostEqual(
            bundle.evaluation.metrics["objective_weight_coverage"], 1.0
        )
        self.assertAlmostEqual(
            bundle.evaluation.metrics["overall_reward"],
            bundle.evaluation.metrics["weighted_normalized_mean_reward"],
        )

    def test_horizon_targetwise_ridge_routes_cell_specific_parameters(self) -> None:
        parameters = {
            "history_steps": 3,
            "ridge_alpha": 0.01,
            "air_temperature_1h_residual_scale": 1.0,
            "air_temperature_6h_residual_scale": 1.0,
            "air_temperature_24h_residual_scale": 1.0,
            "relative_humidity_1h_residual_scale": 1.0,
            "relative_humidity_6h_residual_scale": 1.0,
            "relative_humidity_24h_residual_scale": 1.0,
            "co2_concentration_1h_residual_scale": 0.0,
            "co2_concentration_6h_residual_scale": 1.0,
            "co2_concentration_24h_residual_scale": 1.0,
        }

        bundle = self.evaluate_parameters(
            _cohort_series(),
            parameters,
            suffix="horizon-targetwise-ridge",
            prediction_model_id=HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
            evaluator_id=GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID,
        )

        self.assertEqual(
            bundle.artifact.model_id,
            HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
        )
        self.assertEqual(bundle.artifact.parameters, parameters)
        co2_by_horizon = {
            item["horizon_hours"]: item
            for item in bundle.evaluation.metrics["targets"]
            if item["target"] == "co2_concentration"
        }
        self.assertEqual(set(co2_by_horizon), {1, 6, 24})
        self.assertTrue(math.isclose(co2_by_horizon[1]["skill_score"], 0.0))

    def test_algorithm_spec_predictor_adoption_controls_runtime_routing(self) -> None:
        registry = EvaluatorRegistry(_DatasetStub(_series()))  # type: ignore[arg-type]
        ridge_schema = {
            "history_steps": {"type": "integer", "minimum": 1, "maximum": 12},
            "ridge_alpha": {"type": "number", "minimum": 0.0001, "maximum": 1.0},
            "residual_scale": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        }
        rolling_schema = {
            "blend": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "window": {"type": "integer", "minimum": 1, "maximum": 48},
            "bias_scale": {"type": "number", "minimum": 0.0, "maximum": 2.0},
        }
        task_data = _task().to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "prediction_model_id": EXOGENOUS_RIDGE_MODEL_ID,
            "prediction_model_digest": registry.predictor_configuration_digest(
                EXOGENOUS_RIDGE_MODEL_ID
            ),
            "evaluator_id": GREENHOUSE_EVALUATOR_ID,
            "runtime_component_catalog": {
                "prediction_models": [
                    {
                        "id": EXOGENOUS_RIDGE_MODEL_ID,
                        "dataset_ids": [DATASET_ID],
                        "configuration_digest": registry.predictor_configuration_digest(
                            EXOGENOUS_RIDGE_MODEL_ID
                        ),
                        "parameter_schemas": ridge_schema,
                    },
                    {
                        "id": GREENHOUSE_ROLLING_PREDICTOR_ID,
                        "dataset_ids": [DATASET_ID],
                        "configuration_digest": registry.predictor_configuration_digest(
                            GREENHOUSE_ROLLING_PREDICTOR_ID
                        ),
                        "parameter_schemas": rolling_schema,
                    },
                ],
                "evaluators": [],
                "selected_prediction_model_id": EXOGENOUS_RIDGE_MODEL_ID,
                "selected_evaluator_id": GREENHOUSE_EVALUATOR_ID,
            },
        }
        task = TaskManifest.from_dict(task_data)
        plan = {
            "prediction_model": {
                "id": GREENHOUSE_ROLLING_PREDICTOR_ID,
                "name": "registered rolling residual predictor",
            }
        }
        adoption = resolve_predictor_adoption(task, plan)
        proposal = Proposal(
            proposal_id="proposal:runtime-predictor-adoption",
            run_id="run:runtime-predictor-adoption",
            generation=0,
            title="采用研究计划中的宿主 predictor",
            changes={"blend": 0.5, "window": 3, "bias_scale": 0.5},
            metadata={
                "plan": plan,
                "prediction_model_adoption": adoption.to_dict(),
            },
        )
        candidate = Candidate(
            candidate_id="candidate:runtime-predictor-adoption",
            run_id=proposal.run_id,
            proposal_id=proposal.proposal_id,
            generation=0,
        )

        spec = compile_algorithm_spec(task, proposal, None)
        bundle = registry.evaluate_scientific(
            task,
            candidate,
            proposal,
            algorithm_spec=spec,
        )

        self.assertEqual(spec.adapter_id, GREENHOUSE_ROLLING_PREDICTOR_ID)
        self.assertEqual(bundle.artifact.model_id, GREENHOUSE_ROLLING_PREDICTOR_ID)

    def test_evaluator_digest_versions_the_common_cohort_semantics(self) -> None:
        """Old runs must fail closed instead of mixing two score cohorts."""

        registry = EvaluatorRegistry(_DatasetStub(_series()))  # type: ignore[arg-type]
        implementations = {
            "toy_time_forward@1": ("toy-forward-split/3", "toy-forward-split/2"),
            GREENHOUSE_EVALUATOR_ID: (
                "greenhouse-one-hour-forward/7",
                "greenhouse-one-hour-forward/6",
            ),
            GREENHOUSE_MULTIHORIZON_EVALUATOR_ID: (
                "greenhouse-multihorizon-forward/6",
                "greenhouse-multihorizon-forward/5",
            ),
            GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID: (
                "greenhouse-multihorizon-forward/7",
                "greenhouse-multihorizon-forward/6",
            ),
        }
        for item in registry.catalog():
            implementation, previous_implementation = implementations[item["id"]]
            versioned_payload = {
                "evaluator_id": item["id"],
                "implementation": implementation,
                "evaluation_partition": item["evaluation_partition"],
                "prediction_model_ids": item["prediction_model_ids"],
                "horizons_hours": item["horizons_hours"],
            }
            if item.get("objective_profile") == "greenhouse_equal_weight_skill@1":
                scoring_contract = registry.objective_profile(item["id"])
                versioned_payload["scoring_contract"] = {
                    key: value
                    for key, value in scoring_contract.items()
                    if key != "id"
                }
            self.assertEqual(item["configuration_digest"], digest(versioned_payload))
            legacy_payload = {
                **versioned_payload,
                "implementation": previous_implementation,
            }
            self.assertNotEqual(item["configuration_digest"], digest(legacy_payload))

    def test_bounded_ridge_feedback_uses_balanced_cohort_and_full_fit(self) -> None:
        series = _cohort_series()
        task_data = _task().to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "prediction_model_id": EXOGENOUS_RIDGE_MODEL_ID,
            "evaluator_id": GREENHOUSE_MULTIHORIZON_EVALUATOR_ID,
            "samples_per_update": 18,
        }
        task = TaskManifest.from_dict(task_data)

        def evaluate(*, suffix: str, generation: int, history_steps: int):
            proposal = Proposal(
                proposal_id=f"proposal:bounded-ridge:{suffix}",
                run_id="run:bounded-ridge",
                generation=generation,
                title="bounded ridge feedback cohort",
                changes={
                    "history_steps": history_steps,
                    "ridge_alpha": 0.1,
                    "residual_scale": 1.0,
                },
            )
            candidate = Candidate(
                candidate_id=f"candidate:bounded-ridge:{suffix}",
                run_id=proposal.run_id,
                proposal_id=proposal.proposal_id,
                generation=generation,
            )
            return EvaluatorRegistry(_DatasetStub(series)).evaluate_scientific(  # type: ignore[arg-type]
                task, candidate, proposal
            )

        first = evaluate(suffix="first", generation=0, history_steps=3)
        sibling = evaluate(suffix="sibling", generation=0, history_steps=6)
        next_generation = evaluate(suffix="next", generation=1, history_steps=3)
        metrics = first.evaluation.metrics

        self.assertEqual(metrics["evaluation_selected_examples"], 18)
        self.assertEqual(metrics["evaluation_eligible_examples"], 18)
        self.assertEqual(metrics["evaluation_used_examples"], 18)
        self.assertEqual(metrics["evaluation_skipped_examples"], 0)
        self.assertEqual(metrics["sample_execution_coverage"], 1.0)
        self.assertGreater(metrics["evaluation_available_examples"], 18)
        self.assertEqual(
            metrics["evaluation_deferred_examples"],
            metrics["evaluation_available_examples"] - 18,
        )
        self.assertEqual({item["eligible_rows"] for item in metrics["targets"]}, {2})
        self.assertTrue(
            all(
                item["partition_eligible_rows"] >= item["available_rows"]
                >= item["eligible_rows"]
                for item in metrics["targets"]
            )
        )
        self.assertEqual(first.artifact.metrics["training_partition_rows"], 100)
        self.assertGreater(first.artifact.metrics["training_used_examples"], 18)
        self.assertEqual(
            metrics["feedback_update_cohort_digest"],
            sibling.evaluation.metrics["feedback_update_cohort_digest"],
        )
        self.assertEqual(
            metrics["evaluation_index_digest"],
            sibling.evaluation.metrics["evaluation_index_digest"],
        )
        self.assertNotEqual(
            metrics["feedback_update_cohort_digest"],
            next_generation.evaluation.metrics["feedback_update_cohort_digest"],
        )
        self.assertEqual(
            next_generation.evaluation.metrics["feedback_update_window_offset"], 18
        )

    def test_legacy_manifest_keeps_full_feedback_cohort(self) -> None:
        series = _cohort_series()
        task_data = _task().to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "prediction_model_id": EXOGENOUS_RIDGE_MODEL_ID,
            "evaluator_id": GREENHOUSE_MULTIHORIZON_EVALUATOR_ID,
        }
        proposal = Proposal(
            proposal_id="proposal:legacy-full-feedback",
            run_id="run:legacy-full-feedback",
            generation=0,
            title="legacy full feedback cohort",
            changes={
                "history_steps": 3,
                "ridge_alpha": 0.1,
                "residual_scale": 1.0,
            },
        )
        candidate = Candidate(
            candidate_id="candidate:legacy-full-feedback",
            run_id=proposal.run_id,
            proposal_id=proposal.proposal_id,
            generation=0,
        )
        bundle = EvaluatorRegistry(_DatasetStub(series)).evaluate_scientific(  # type: ignore[arg-type]
            TaskManifest.from_dict(task_data), candidate, proposal
        )

        metrics = bundle.evaluation.metrics
        self.assertNotIn("samples_per_update", metrics)
        self.assertNotIn("feedback_update_cohort_digest", metrics)
        self.assertNotIn("update_window", metrics["evaluation_cohort"])
        self.assertGreater(metrics["evaluation_used_examples"], 18)
        self.assertEqual(
            metrics["evaluation_eligible_examples"],
            sum(item["eligible_rows"] for item in metrics["targets"]),
        )
        self.assertEqual(bundle.artifact.metrics["training_partition_rows"], 100)

    def test_bounded_rolling_windows_share_the_same_generation_cohort(self) -> None:
        series = _cohort_series()
        task_data = _task().to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "samples_per_update": 15,
        }
        task = TaskManifest.from_dict(task_data)

        def evaluate(suffix: str, window: int):
            proposal = Proposal(
                proposal_id=f"proposal:bounded-rolling:{suffix}",
                run_id="run:bounded-rolling",
                generation=0,
                title="bounded rolling cohort",
                changes={"blend": 0.5, "window": window, "bias_scale": 0.5},
            )
            candidate = Candidate(
                candidate_id=f"candidate:bounded-rolling:{suffix}",
                run_id=proposal.run_id,
                proposal_id=proposal.proposal_id,
                generation=0,
            )
            return EvaluatorRegistry(_DatasetStub(series)).evaluate_scientific(  # type: ignore[arg-type]
                task, candidate, proposal
            )

        short = evaluate("short", 1)
        long = evaluate("long", 12)
        metrics = short.evaluation.metrics
        self.assertEqual(metrics["evaluation_selected_examples"], 15)
        self.assertEqual(metrics["evaluation_eligible_examples"], 15)
        self.assertEqual({item["eligible_rows"] for item in metrics["targets"]}, {5})
        self.assertEqual(metrics["sample_execution_coverage"], 1.0)
        self.assertEqual(short.artifact.metrics["training_partition_rows"], 100)
        self.assertGreater(short.artifact.metrics["training_used_examples"], 15)
        self.assertEqual(
            metrics["feedback_update_cohort_digest"],
            long.evaluation.metrics["feedback_update_cohort_digest"],
        )
        self.assertEqual(
            metrics["evaluation_index_digest"],
            long.evaluation.metrics["evaluation_index_digest"],
        )

    def test_exogenous_ridge_multihorizon_metrics_are_complete_and_finite(self) -> None:
        timestamps = tuple(range(240))
        signal = [math.sin(2 * math.pi * index / 24) for index in timestamps]

        def driven(initial: float, gain: float) -> tuple[float, ...]:
            values = [initial]
            for index in range(len(timestamps) - 1):
                values.append(values[-1] + gain * signal[index])
            return tuple(values)

        series = DatasetSeries(
            schema="ecologyrsi-dsh.dataset-series/1",
            dataset_id=DATASET_ID,
            domain_id="greenhouse_cucumber_2018",
            episode_id=f"{DATASET_ID}:TeamA",
            digest="d" * 64,
            timestamps=timestamps,
            values={
                "air_temperature": driven(22.0, 0.35),
                "relative_humidity": driven(70.0, 0.7),
                "co2_concentration": driven(700.0, 12.0),
                "ventilation_setpoint": tuple(signal),
            },
            partitions={
                "training_fit": IndexRange(0, 140),
                "training_feedback": IndexRange(140, 240),
                "development": IndexRange(240, 240),
            },
            features={
                "air_temperature": FeatureSpec(
                    "air_temperature", "温室气温", "environment", "degC"
                ),
                "relative_humidity": FeatureSpec(
                    "relative_humidity", "相对湿度", "environment", "percent"
                ),
                "co2_concentration": FeatureSpec(
                    "co2_concentration", "二氧化碳浓度", "environment", "ppm"
                ),
                "ventilation_setpoint": FeatureSpec(
                    "ventilation_setpoint", "通风设定值", "action", "percent"
                ),
            },
            split_manifest_digest_sha256=SPLIT_DIGEST,
        )
        task_data = _task().to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "prediction_model_id": EXOGENOUS_RIDGE_MODEL_ID,
            "evaluator_id": GREENHOUSE_MULTIHORIZON_EVALUATOR_ID,
        }
        task = TaskManifest.from_dict(task_data)
        proposal = Proposal(
            proposal_id="proposal:ridge:1",
            run_id="run:ridge",
            generation=0,
            title="外生变量岭回归方案",
            changes={
                "history_steps": 3,
                "ridge_alpha": 0.01,
                "residual_scale": 1.0,
            },
        )
        candidate = Candidate(
            candidate_id="candidate:ridge:1",
            run_id=proposal.run_id,
            proposal_id=proposal.proposal_id,
            generation=0,
        )
        bundle = EvaluatorRegistry(_DatasetStub(series)).evaluate_scientific(  # type: ignore[arg-type]
            task, candidate, proposal
        )
        metrics = bundle.evaluation.metrics
        self.assertEqual(bundle.artifact.model_id, EXOGENOUS_RIDGE_MODEL_ID)
        self.assertEqual(len(metrics["targets"]), 9)
        self.assertEqual(
            {item["horizon_hours"] for item in metrics["targets"]},
            {1, 6, 24},
        )
        self.assertEqual(len(metrics["horizons"]), 3)
        self.assertLessEqual(len(metrics["prediction_preview"]), 48)
        self.assertEqual(bundle.artifact.metrics["training_partition_rows"], 140)
        self.assertGreater(bundle.artifact.metrics["training_used_examples"], 140)
        self.assertEqual(bundle.artifact.metrics["fit_method"], "closed_form_ridge")
        self.assertEqual(bundle.artifact.metrics["fit_passes_completed"], 1)
        self.assertFalse(bundle.artifact.metrics["iterative_epoch_training"])
        self.assertEqual(metrics["evaluation_partition_rows"], 100)
        self.assertEqual(metrics["evaluation_used_examples"], metrics["n"])
        self.assertEqual(
            metrics["evaluation_eligible_examples"],
            metrics["evaluation_used_examples"]
            + metrics["evaluation_skipped_examples"],
        )
        self.assertTrue(
            all(
                row["target_timestamp"] - row["origin_timestamp"]
                == row["horizon_hours"]
                for row in metrics["prediction_preview"]
            )
        )
        self.assertTrue(
            all(
                math.isfinite(float(value))
                for item in metrics["targets"]
                for key, value in item.items()
                if key not in {"target", "unit"}
            )
        )
        json.dumps(bundle.artifact.to_dict(), allow_nan=False)
        json.dumps(bundle.evaluation.to_dict(), allow_nan=False)

    @unittest.skipUnless(
        os.environ.get("ECOLOGYRSI_TEST_REAL_DATA") == "1",
        "需设置 ECOLOGYRSI_TEST_REAL_DATA=1",
    )
    def test_optional_real_greenhouse_evaluation_is_finite_and_digest_bound(
        self,
    ) -> None:
        datasets = DatasetRegistry()
        evaluator = EvaluatorRegistry(datasets)
        for dataset_id in ("agc_cucumber_2018", "agc_tomato_2019"):
            with self.subTest(dataset_id=dataset_id):
                task = TaskManifest(
                    task_id=f"real-evaluation:{dataset_id}",
                    objective="评估真实温室环境预测",
                    domain_pack="greenhouse_real_observation",
                    visible_datasets=(dataset_id,),
                    budget={"max_candidates": 1},
                    metadata={
                        "evaluator_id": GREENHOUSE_EVALUATOR_ID,
                        "prediction_model_id": GREENHOUSE_ROLLING_PREDICTOR_ID,
                        "episode_id": datasets.series(dataset_id).episode_id,
                        "dataset_digest": datasets.series(dataset_id).digest,
                        "split_manifest_digest": datasets.series(
                            dataset_id
                        ).split_manifest_digest_sha256,
                    },
                )
                proposal = Proposal(
                    proposal_id=f"proposal:{dataset_id}:1",
                    run_id=f"run:{dataset_id}",
                    generation=0,
                    title="真实温室评测方案",
                    changes={"blend": 0.5, "window": 6, "bias_scale": 0.5},
                )
                candidate = Candidate(
                    candidate_id=f"candidate:{dataset_id}:1",
                    run_id=proposal.run_id,
                    proposal_id=proposal.proposal_id,
                    generation=0,
                )
                bundle = evaluator.evaluate(task, candidate, proposal)
                self.assertTrue(math.isfinite(bundle.evaluation.score))
                self.assertGreater(bundle.evaluation.metrics["n"], 0)
                self.assertEqual(
                    len(bundle.evaluation.metrics["split_manifest_digest_sha256"]),
                    64,
                )
                json.dumps(bundle.evaluation.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
