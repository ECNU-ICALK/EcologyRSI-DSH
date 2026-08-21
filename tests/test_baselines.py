from __future__ import annotations

from dataclasses import replace
import unittest

from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.data.contracts import DatasetSeries
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.evaluators.baselines import (
    apply_baseline_profile,
    fit_baseline_profile,
)


def _series(values: tuple[float | None, ...]) -> DatasetSeries:
    return DatasetSeries(
        schema="test",
        dataset_id="test-dataset",
        domain_id="greenhouse",
        episode_id="episode-1",
        digest="dataset-digest",
        timestamps=tuple(range(len(values))),
        values={
            "air_temperature": values,
            "relative_humidity": values,
            "co2_concentration": values,
        },
        partitions={
            "training_fit": IndexRange(0, 72),
            "training_feedback": IndexRange(72, len(values)),
        },
        features={},
        split_manifest_digest_sha256="split-digest",
    )


class StrongBaselineTests(unittest.TestCase):
    def test_fit_selects_seasonal_when_it_beats_persistence(self) -> None:
        values = tuple(float(index % 24) for index in range(96))
        profile = fit_baseline_profile(
            _series(values),
            targets=("air_temperature",),
            horizons=(1, 6, 24),
        )

        self.assertEqual(
            [cell["baseline_id"] for cell in profile["cells"]],
            ["seasonal_24h", "seasonal_24h", "persistence"],
        )
        self.assertEqual(profile["selection_partition"], "training_fit")
        self.assertEqual(len(profile["digest"]), 64)

    def test_fit_tie_and_future_seasonal_reference_choose_persistence(self) -> None:
        values = tuple(5.0 for _ in range(96))
        profile = fit_baseline_profile(
            _series(values),
            targets=("air_temperature",),
            horizons=(1, 25),
        )

        cells = {
            cell["horizon_hours"]: cell for cell in profile["cells"]
        }
        self.assertEqual(cells[1]["baseline_id"], "persistence")
        self.assertEqual(cells[25]["baseline_id"], "persistence")
        self.assertEqual(cells[25]["seasonal_status"], "not_causal")

    def test_profile_uses_training_fit_only(self) -> None:
        fit = [float(index % 24) for index in range(72)]
        first = fit + [1000.0] * 24
        second = fit + [-1000.0] * 24

        left = fit_baseline_profile(
            _series(tuple(first)), targets=("air_temperature",), horizons=(1,)
        )
        right = fit_baseline_profile(
            _series(tuple(second)), targets=("air_temperature",), horizons=(1,)
        )

        self.assertEqual(left, right)

    def test_apply_preserves_model_reference_and_marks_missing_fallback(self) -> None:
        raw = list(float(index % 24) for index in range(96))
        raw[72] = None  # seasonal reference for target timestamp 96 is unavailable
        series = _series(tuple(raw))
        profile = fit_baseline_profile(
            series, targets=("air_temperature",), horizons=(1,)
        )
        self.assertEqual(profile["cells"][0]["baseline_id"], "seasonal_24h")
        rows = [
            {
                "target": "air_temperature",
                "horizon_hours": 1,
                "timestamp": 96,
                "target_timestamp": 96,
                "origin_timestamp": 95,
                "baseline": 23.0,
            }
        ]

        applied = apply_baseline_profile(series, rows, profile)

        self.assertEqual(applied[0]["model_reference_baseline"], 23.0)
        self.assertEqual(applied[0]["baseline"], 23.0)
        self.assertEqual(applied[0]["baseline_id"], "persistence")
        self.assertEqual(applied[0]["baseline_fallback"], "seasonal_reference_missing")
        self.assertEqual(applied[0]["baseline_profile_digest"], profile["digest"])

    def test_failed_execution_cannot_receive_positive_reward(self) -> None:
        series = _series(tuple(float(index % 24) for index in range(96)))
        profile = fit_baseline_profile(
            series, targets=("air_temperature",), horizons=(1,)
        )
        rows = [
            {
                "target": "air_temperature",
                "horizon_hours": 1,
                "timestamp": 73,
                "target_timestamp": 73,
                "origin_timestamp": 72,
                "observed": 0.0,
                "predicted": 0.0,
                "baseline": 0.0,
                "sample_execution_status": "failed",
                "scoring_fallback": "registered_fallback",
            }
        ]

        applied = apply_baseline_profile(series, rows, profile)

        self.assertEqual(applied[0]["baseline_id"], "seasonal_24h")
        self.assertEqual(applied[0]["baseline"], 1.0)
        self.assertEqual(applied[0]["predicted"], 1.0)
        self.assertEqual(applied[0]["failed_reward_policy"], "nonpositive")

    def test_apply_rejects_tampered_or_cross_dataset_profile(self) -> None:
        series = _series(tuple(float(index % 24) for index in range(96)))
        profile = fit_baseline_profile(
            series, targets=("air_temperature",), horizons=(1,)
        )
        profile["cells"][0]["baseline_id"] = "persistence"

        with self.assertRaisesRegex(ValueError, "digest does not match"):
            apply_baseline_profile(series, (), profile)

        tampered_contract = fit_baseline_profile(
            series, targets=("air_temperature",), horizons=(1,)
        )
        tampered_contract["selection_partition"] = "development"
        tampered_contract["selection_rule"] = "external_holdout_best"
        tampered_contract["digest"] = digest(
            {
                key: value
                for key, value in tampered_contract.items()
                if key != "digest"
            }
        )
        with self.assertRaisesRegex(ValueError, "canonical fit selection"):
            apply_baseline_profile(series, (), tampered_contract)

        valid_profile = fit_baseline_profile(
            series, targets=("air_temperature",), horizons=(1,)
        )
        with self.assertRaisesRegex(ValueError, "dataset digest"):
            apply_baseline_profile(
                replace(series, digest="different-dataset"), (), valid_profile
            )


if __name__ == "__main__":
    unittest.main()
