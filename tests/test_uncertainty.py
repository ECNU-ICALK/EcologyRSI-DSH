from __future__ import annotations

import unittest

from ecologyrsi_dsh.evaluators.uncertainty import (
    UQ_POLICY_ID,
    apply_calibrated_intervals,
    calibrate_residual_policy,
    summarize_interval_evidence,
)


def _rows(errors: list[float], *, point_offset: float = 0.0) -> list[dict]:
    return [
        {
            "target": "air_temperature",
            "horizon_hours": 1,
            "target_timestamp": 1000 + index,
            "partition": "calibration_uq",
            "observed": 20.0,
            "predicted": 20.0 + point_offset + error,
            "normalization_scale": 2.0,
        }
        for index, error in enumerate(errors)
    ]


class CalibratedResidualTests(unittest.TestCase):
    def test_finite_sample_quantile_and_artifact_are_deterministic(self) -> None:
        rows = _rows([0, 1, 2, 3, 4, 5, 6, 7, 8])
        first = calibrate_residual_policy(
            rows,
            point_artifact_digest="a" * 64,
            data_protocol_digest="b" * 64,
            alpha=0.1,
        )
        second = calibrate_residual_policy(
            list(reversed(rows)),
            point_artifact_digest="a" * 64,
            data_protocol_digest="b" * 64,
            alpha=0.1,
        )
        cell = first["cells"][0]
        self.assertEqual(first, second)
        self.assertEqual(first["policy_id"], UQ_POLICY_ID)
        self.assertEqual(cell["quantile_index"], 9)
        self.assertEqual(cell["normalized_residual_quantile"], 4.0)
        self.assertEqual(cell["half_width"], 8.0)

    def test_candidate_and_baseline_keep_separate_residual_artifacts(self) -> None:
        candidate = calibrate_residual_policy(
            _rows([0.5] * 9),
            point_artifact_digest="a" * 64,
            data_protocol_digest="b" * 64,
        )
        baseline = calibrate_residual_policy(
            _rows([2.0] * 9),
            point_artifact_digest="c" * 64,
            data_protocol_digest="b" * 64,
        )
        self.assertNotEqual(candidate["artifact_digest"], baseline["artifact_digest"])
        self.assertEqual(candidate["cells"][0]["half_width"], 0.5)
        self.assertEqual(baseline["cells"][0]["half_width"], 2.0)

    def test_calibration_rejects_non_uq_rows_and_scale_drift(self) -> None:
        wrong_partition = _rows([1.0] * 9)
        wrong_partition[0]["partition"] = "model_selection"
        with self.assertRaisesRegex(ValueError, "calibration_uq"):
            calibrate_residual_policy(
                wrong_partition,
                point_artifact_digest="a" * 64,
                data_protocol_digest="b" * 64,
            )
        scale_drift = _rows([1.0] * 9)
        scale_drift[-1]["normalization_scale"] = 3.0
        with self.assertRaisesRegex(ValueError, "scale"):
            calibrate_residual_policy(
                scale_drift,
                point_artifact_digest="a" * 64,
                data_protocol_digest="b" * 64,
            )

    def test_application_and_winkler_score_use_exact_formula(self) -> None:
        artifact = calibrate_residual_policy(
            _rows([1.0] * 9),
            point_artifact_digest="a" * 64,
            data_protocol_digest="b" * 64,
        )
        formal = [
            {
                "target": "air_temperature",
                "horizon_hours": 1,
                "target_timestamp": 2000,
                "observed": 24.0,
                "predicted": 20.0,
                "normalization_scale": 2.0,
            }
        ]
        bounded = apply_calibrated_intervals(formal, artifact)
        self.assertEqual(bounded[0]["prediction_lower"], 19.0)
        self.assertEqual(bounded[0]["prediction_upper"], 21.0)
        summary = summarize_interval_evidence(bounded, alpha=0.1)
        cell = summary["cells"][0]
        self.assertEqual(cell["coverage"], 0.0)
        # width/scale + (2/alpha)*(observed-upper)/scale = 1 + 30
        self.assertEqual(cell["normalized_interval_score"], 31.0)


if __name__ == "__main__":
    unittest.main()
