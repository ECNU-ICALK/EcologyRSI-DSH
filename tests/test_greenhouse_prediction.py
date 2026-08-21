from __future__ import annotations

import json
import math
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.datasets import DatasetSeries
from ecologyrsi_dsh.greenhouse import FeatureSpec
from ecologyrsi_dsh.greenhouse_prediction import (
    EXOGENOUS_RIDGE_MODEL_ID,
    HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
    TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
    ExogenousRidgeConfig,
    HorizonTargetwiseExogenousRidgeConfig,
    TargetwiseExogenousRidgeConfig,
    fit_predict_exogenous_ridge,
    predict_fitted_exogenous_ridge,
)
from ecologyrsi_dsh.splits import IndexRange


def _feature(name: str, role: str) -> FeatureSpec:
    return FeatureSpec(name, name, role, "unit")


def _series(
    timestamps: tuple[int, ...],
    values: dict[str, tuple[float | None, ...]],
    roles: dict[str, str],
    *,
    fit: tuple[int, int],
    feedback: tuple[int, int],
) -> DatasetSeries:
    return DatasetSeries(
        schema="ecologyrsi-dsh.dataset-series/1",
        dataset_id="agc_cucumber_2018",
        domain_id="greenhouse_cucumber_2018",
        episode_id="agc_cucumber_2018:test",
        digest="d" * 64,
        timestamps=timestamps,
        values=values,
        partitions={
            "training_fit": IndexRange(*fit),
            "training_feedback": IndexRange(*feedback),
            "development": IndexRange(feedback[1], feedback[1]),
        },
        features={name: _feature(name, role) for name, role in roles.items()},
        split_manifest_digest_sha256="s" * 64,
    )


def _run(
    series: DatasetSeries,
    *,
    horizons: tuple[int, ...] = (1,),
    history_steps: int = 1,
) -> dict:
    return fit_predict_exogenous_ridge(
        series,
        targets=("air_temperature",),
        horizons=horizons,
        config=ExogenousRidgeConfig(
            history_steps=history_steps,
            ridge_alpha=0.01,
            residual_scale=1.0,
        ),
    )


class GreenhousePredictionTests(unittest.TestCase):
    def test_feedback_prediction_can_be_deferred_until_tool_invocation(self) -> None:
        timestamps = tuple(range(72))
        target = tuple(20.0 + math.sin(index / 4.0) for index in timestamps)
        action = tuple(math.cos(index / 7.0) for index in timestamps)
        series = _series(
            timestamps,
            {"air_temperature": target, "ventilation": action},
            {"air_temperature": "environment", "ventilation": "action"},
            fit=(0, 48),
            feedback=(48, 72),
        )
        config = ExogenousRidgeConfig(
            history_steps=3,
            ridge_alpha=0.01,
            residual_scale=0.8,
        )
        eager = fit_predict_exogenous_ridge(
            series,
            targets=("air_temperature",),
            horizons=(1,),
            config=config,
        )
        deferred = fit_predict_exogenous_ridge(
            series,
            targets=("air_temperature",),
            horizons=(1,),
            config=config,
            defer_prediction_partitions=("training_feedback",),
        )
        eager_row = next(
            row
            for row in eager["prediction_rows"]
            if row["partition"] == "training_feedback"
        )
        deferred_row = next(
            row
            for row in deferred["prediction_rows"]
            if row["partition"] == "training_feedback"
            and row["origin_timestamp"] == eager_row["origin_timestamp"]
        )

        self.assertNotIn("predicted", deferred_row)
        self.assertEqual(
            deferred["deferred_prediction_partitions"], ["training_feedback"]
        )
        predicted = predict_fitted_exogenous_ridge(
            target=deferred_row["target"],
            horizon_hours=deferred_row["horizon_hours"],
            baseline=deferred_row["baseline"],
            label_free_context=deferred_row["label_free_context"],
            models=deferred["models"],
            config=config,
        )
        self.assertAlmostEqual(predicted, eager_row["predicted"], places=12)

    def test_targetwise_ridge_uses_registered_persistence_fallback_for_co2(
        self,
    ) -> None:
        timestamps = tuple(range(140))
        signal = tuple(math.sin(index / 5.0) for index in timestamps)
        air = [20.0]
        humidity = [70.0]
        co2 = [700.0]
        for index in range(len(timestamps) - 1):
            air.append(air[-1] + 0.8 * signal[index])
            humidity.append(humidity[-1] - 1.2 * signal[index])
            co2.append(co2[-1] + 20.0 * signal[index])
        series = _series(
            timestamps,
            {
                "air_temperature": tuple(air),
                "relative_humidity": tuple(humidity),
                "co2_concentration": tuple(co2),
                "ventilation": signal,
            },
            {
                "air_temperature": "environment",
                "relative_humidity": "environment",
                "co2_concentration": "environment",
                "ventilation": "action",
            },
            fit=(0, 85),
            feedback=(85, 140),
        )

        result = fit_predict_exogenous_ridge(
            series,
            targets=(
                "air_temperature",
                "relative_humidity",
                "co2_concentration",
            ),
            horizons=(1,),
            config=TargetwiseExogenousRidgeConfig(
                history_steps=3,
                ridge_alpha=0.01,
                air_temperature_residual_scale=1.0,
                relative_humidity_residual_scale=1.0,
                co2_concentration_residual_scale=0.0,
            ),
        )

        self.assertEqual(result["model_id"], TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID)
        feedback_rows = [
            item
            for item in result["prediction_rows"]
            if item["partition"] == "training_feedback"
        ]
        co2_rows = [
            item for item in feedback_rows if item["target"] == "co2_concentration"
        ]
        air_rows = [
            item for item in feedback_rows if item["target"] == "air_temperature"
        ]
        self.assertTrue(co2_rows)
        self.assertTrue(
            all(item["predicted"] == item["baseline"] for item in co2_rows)
        )
        self.assertTrue(all(item["used_target_persistence"] for item in co2_rows))
        self.assertTrue(
            all(
                item["label_free_context"]["predictor_state"][
                    "used_target_persistence"
                ]
                for item in co2_rows
            )
        )
        self.assertFalse(
            any(item["used_zero_residual_fallback"] for item in co2_rows)
        )
        self.assertTrue(
            any(
                not math.isclose(item["predicted"], item["baseline"])
                for item in air_rows
            )
        )

    def test_horizon_targetwise_ridge_isolates_persistence_to_one_cell(self) -> None:
        timestamps = tuple(range(220))
        signal = tuple(
            math.sin(index / 5.0) + 0.3 * math.cos(index / 11.0)
            for index in timestamps
        )
        co2 = [700.0]
        for index in range(len(timestamps) - 1):
            co2.append(co2[-1] + 18.0 * signal[index])
        series = _series(
            timestamps,
            {
                "co2_concentration": tuple(co2),
                "ventilation": signal,
            },
            {
                "co2_concentration": "environment",
                "ventilation": "action",
            },
            fit=(0, 140),
            feedback=(140, 220),
        )
        config = HorizonTargetwiseExogenousRidgeConfig(
            history_steps=3,
            ridge_alpha=0.01,
            air_temperature_1h_residual_scale=1.0,
            air_temperature_6h_residual_scale=1.0,
            air_temperature_24h_residual_scale=1.0,
            relative_humidity_1h_residual_scale=1.0,
            relative_humidity_6h_residual_scale=1.0,
            relative_humidity_24h_residual_scale=1.0,
            co2_concentration_1h_residual_scale=0.0,
            co2_concentration_6h_residual_scale=1.0,
            co2_concentration_24h_residual_scale=1.0,
        )

        result = fit_predict_exogenous_ridge(
            series,
            targets=("co2_concentration",),
            horizons=(1, 6, 24),
            config=config,
        )

        self.assertEqual(
            result["model_id"], HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
        )
        feedback_rows = [
            row
            for row in result["prediction_rows"]
            if row["partition"] == "training_feedback"
        ]
        one_hour = [row for row in feedback_rows if row["horizon_hours"] == 1]
        longer = [row for row in feedback_rows if row["horizon_hours"] in {6, 24}]
        self.assertTrue(one_hour)
        self.assertTrue(longer)
        self.assertTrue(
            all(row["predicted"] == row["baseline"] for row in one_hour)
        )
        self.assertTrue(all(row["used_target_persistence"] for row in one_hour))
        self.assertTrue(
            any(
                not math.isclose(row["predicted"], row["baseline"])
                for row in longer
            )
        )
        self.assertFalse(any(row["used_target_persistence"] for row in longer))
        with self.assertRaisesRegex(ValueError, "not registered"):
            config.residual_scale_for("co2_concentration", 2)

    def test_horizon_targetwise_scale_mapping_has_no_cross_cell_aliases(self) -> None:
        expected = {
            ("air_temperature", 1): 0.11,
            ("air_temperature", 6): 0.12,
            ("air_temperature", 24): 0.13,
            ("relative_humidity", 1): 0.21,
            ("relative_humidity", 6): 0.22,
            ("relative_humidity", 24): 0.23,
            ("co2_concentration", 1): 0.31,
            ("co2_concentration", 6): 0.32,
            ("co2_concentration", 24): 0.33,
        }
        config = HorizonTargetwiseExogenousRidgeConfig(
            history_steps=3,
            ridge_alpha=0.01,
            air_temperature_1h_residual_scale=0.11,
            air_temperature_6h_residual_scale=0.12,
            air_temperature_24h_residual_scale=0.13,
            relative_humidity_1h_residual_scale=0.21,
            relative_humidity_6h_residual_scale=0.22,
            relative_humidity_24h_residual_scale=0.23,
            co2_concentration_1h_residual_scale=0.31,
            co2_concentration_6h_residual_scale=0.32,
            co2_concentration_24h_residual_scale=0.33,
        )

        for (target, horizon), scale in expected.items():
            with self.subTest(target=target, horizon=horizon):
                self.assertEqual(config.residual_scale_for(target, horizon), scale)

    def test_exact_timestamp_offsets_skip_gaps_for_every_horizon(self) -> None:
        timestamps = (0, 1, 2, 4, 5, 6, 10, 11, 12, 14, 15, 16)
        target = tuple(20.0 + timestamp for timestamp in timestamps)
        series = _series(
            timestamps,
            {"air_temperature": target},
            {"air_temperature": "environment"},
            fit=(0, 6),
            feedback=(6, 12),
        )

        result = _run(series, horizons=(1, 2))
        rows = result["prediction_rows"]
        one_hour_pairs = {
            (row["origin_timestamp"], row["timestamp"])
            for row in rows
            if row["horizon_hours"] == 1
        }
        two_hour_pairs = {
            (row["origin_timestamp"], row["timestamp"])
            for row in rows
            if row["horizon_hours"] == 2
        }

        self.assertEqual(
            one_hour_pairs,
            {(0, 1), (1, 2), (4, 5), (5, 6), (10, 11), (11, 12), (14, 15), (15, 16)},
        )
        self.assertEqual(
            two_hour_pairs,
            {(0, 2), (2, 4), (4, 6), (10, 12), (12, 14), (14, 16)},
        )
        self.assertTrue(
            all(row["timestamp"] - row["origin_timestamp"] == row["horizon_hours"] for row in rows)
        )

    def test_lags_and_forward_fill_never_cross_partition_boundary(self) -> None:
        timestamps = tuple(range(12))
        target = tuple(20.0 + index for index in timestamps)
        action: list[float | None] = [1.0] * 6 + [None, None, 2.0, None, None, None]
        series = _series(
            timestamps,
            {"air_temperature": target, "heating_setpoint": tuple(action)},
            {"air_temperature": "environment", "heating_setpoint": "action"},
            fit=(0, 6),
            feedback=(6, 12),
        )

        result = _run(series, history_steps=2)
        feedback_rows = [
            row for row in result["prediction_rows"] if row["partition"] == "training_feedback"
        ]

        # Origin 6 cannot use target lag timestamp 5 from training_fit.
        self.assertEqual(feedback_rows[0]["origin_timestamp"], 7)
        self.assertEqual(feedback_rows[0]["timestamp"], 8)
        sample_context = feedback_rows[0]["label_free_context"]
        self.assertEqual(
            sample_context["schema_version"],
            "ecologyrsi-dsh.label-free-sample-context/1",
        )
        self.assertEqual(len(sample_context["history_window"]), 2)
        self.assertTrue(sample_context["feature_snapshot"])
        self.assertNotIn("observed", json.dumps(sample_context).casefold())
        model = result["models"][0]
        action_stat = next(
            item for item in model["feature_statistics"] if item.get("source_feature") == "heating_setpoint"
        )
        self.assertEqual(action_stat["median"], 1.0)
        self.assertEqual(action_stat["mean"], 1.0)

    def test_feedback_sentinels_do_not_change_fit_metadata_or_feature_selection(self) -> None:
        timestamps = tuple(range(80))
        target = tuple(20.0 + math.sin(index / 5.0) for index in timestamps)
        fit_signal = [float(index % 3) for index in range(40)]
        ordinary_signal = tuple(fit_signal + [0.0] * 40)
        sentinel_signal = tuple(fit_signal + [999999.0] * 40)
        sparse = tuple([None] * 35 + [1.0] * 5 + [777777.0] * 40)
        roles = {
            "air_temperature": "environment",
            "outside_temperature": "outside_weather",
            "feedback_only": "action",
        }

        ordinary = _run(
            _series(
                timestamps,
                {
                    "air_temperature": target,
                    "outside_temperature": ordinary_signal,
                    "feedback_only": sparse,
                },
                roles,
                fit=(0, 40),
                feedback=(40, 80),
            )
        )
        sentinel = _run(
            _series(
                timestamps,
                {
                    "air_temperature": target,
                    "outside_temperature": sentinel_signal,
                    "feedback_only": sparse,
                },
                roles,
                fit=(0, 40),
                feedback=(40, 80),
            )
        )

        ordinary_model = ordinary["models"][0]
        sentinel_model = sentinel["models"][0]
        for field in (
            "selected_exogenous_features",
            "feature_names",
            "feature_statistics",
            "intercept",
            "coefficients",
            "fit_digest_sha256",
        ):
            self.assertEqual(ordinary_model[field], sentinel_model[field])
        self.assertNotIn("feedback_only", ordinary_model["selected_exogenous_features"])

    def test_collinear_and_constant_features_fit_and_remain_strict_json(self) -> None:
        timestamps = tuple(range(100))
        target = tuple(18.0 + 0.03 * index + math.sin(index / 7.0) for index in timestamps)
        signal = tuple(math.sin(index / 4.0) for index in timestamps)
        constant = tuple(5.0 for _ in timestamps)
        series = _series(
            timestamps,
            {
                "air_temperature": target,
                "signal_a": signal,
                "signal_b": signal,
                "constant": constant,
            },
            {
                "air_temperature": "environment",
                "signal_a": "outside_weather",
                "signal_b": "action",
                "constant": "crop",
            },
            fit=(0, 60),
            feedback=(60, 100),
        )

        result = _run(series, history_steps=3)
        model = result["models"][0]

        self.assertEqual(result["model_id"], EXOGENOUS_RIDGE_MODEL_ID)
        self.assertEqual(model["status"], "fitted")
        constant_stat = next(
            item for item in model["feature_statistics"] if item.get("source_feature") == "constant"
        )
        self.assertTrue(constant_stat["constant"])
        self.assertEqual(constant_stat["scale"], 1.0)
        json.dumps(result, allow_nan=False)

    def test_origin_external_signal_materially_improves_over_persistence(self) -> None:
        timestamps = tuple(range(240))
        external = tuple(math.sin(2.0 * math.pi * index / 17.0) for index in timestamps)
        target_values = [20.0]
        for index in range(len(timestamps) - 1):
            target_values.append(target_values[-1] + 2.5 * external[index])
        series = _series(
            timestamps,
            {
                "air_temperature": tuple(target_values),
                "ventilation_setpoint": external,
            },
            {
                "air_temperature": "environment",
                "ventilation_setpoint": "action",
            },
            fit=(0, 140),
            feedback=(140, 240),
        )

        result = _run(series)
        feedback = [
            row for row in result["prediction_rows"] if row["partition"] == "training_feedback"
        ]
        model_rmse = math.sqrt(
            sum((row["predicted"] - row["observed"]) ** 2 for row in feedback) / len(feedback)
        )
        baseline_rmse = math.sqrt(
            sum((row["baseline"] - row["observed"]) ** 2 for row in feedback) / len(feedback)
        )

        self.assertIn("ventilation_setpoint", result["models"][0]["selected_exogenous_features"])
        self.assertLess(model_rmse, baseline_rmse * 0.1)

    def test_forward_fill_limits_use_elapsed_hours_and_solver_failure_falls_back(self) -> None:
        timestamps = tuple(range(0, 202, 2))
        target = tuple(20.0 + math.sin(timestamp / 8.0) for timestamp in timestamps)
        action = tuple(
            float(timestamp) if timestamp % 10 == 0 else None for timestamp in timestamps
        )
        crop = tuple(1.0 if timestamp == 0 else None for timestamp in timestamps)
        series = _series(
            timestamps,
            {
                "air_temperature": target,
                "heating_setpoint": action,
                "cumulative_leaves": crop,
            },
            {
                "air_temperature": "environment",
                "heating_setpoint": "action",
                "cumulative_leaves": "crop",
            },
            fit=(0, 90),
            feedback=(90, 101),
        )

        with patch(
            "ecologyrsi_dsh.evaluators.greenhouse_prediction._solve_with_partial_pivoting",
            side_effect=ArithmeticError("synthetic solver failure"),
        ):
            result = _run(series, horizons=(2,))

        model = result["models"][0]
        self.assertEqual(model["status"], "fallback_zero_residual")
        self.assertEqual(model["fallback_reason"], "numerical_fit_failure")
        self.assertGreater(model["prediction_fallback_rows"], 0)
        self.assertTrue(
            all(row["predicted"] == row["baseline"] for row in result["prediction_rows"])
        )
        # Re-run normally to inspect fit-only availability learned after
        # elapsed-time, rather than row-count, forward filling.
        ordinary = _run(series, horizons=(2,))
        stats = {
            item.get("source_feature"): item
            for item in ordinary["models"][0]["feature_statistics"]
        }
        # Origins are timestamps 0..176 (89 rows). Action values cover four
        # two-hour origins per ten-hour block; crop value covers through 168h.
        self.assertAlmostEqual(stats["heating_setpoint"]["coverage"], 72 / 89)
        self.assertAlmostEqual(stats["cumulative_leaves"]["coverage"], 85 / 89)
        json.dumps(result, allow_nan=False)

    def test_parameter_bounds_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "history_steps"):
            ExogenousRidgeConfig(history_steps=0, ridge_alpha=0.1, residual_scale=1.0)
        with self.assertRaisesRegex(ValueError, "ridge_alpha"):
            ExogenousRidgeConfig(history_steps=1, ridge_alpha=0.0, residual_scale=1.0)
        with self.assertRaisesRegex(ValueError, "residual_scale"):
            ExogenousRidgeConfig(history_steps=1, ridge_alpha=0.1, residual_scale=1.1)


if __name__ == "__main__":
    unittest.main()
