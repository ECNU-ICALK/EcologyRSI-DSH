from __future__ import annotations

import math
import unittest

from ecologyrsi_dsh.evaluators.objectives import (
    DEFAULT_TARGET_WEIGHTS,
    aggregate_greenhouse_objective,
    normalized_absolute_error_reward,
    skill_score,
)


class ObjectiveKernelTests(unittest.TestCase):
    def test_reward_sign_and_bounds(self) -> None:
        raw, bounded = normalized_absolute_error_reward([2.0], [1.0], 2.0)
        self.assertEqual(raw, 0.5)
        self.assertEqual(bounded, 0.5)

        raw, bounded = normalized_absolute_error_reward([1.0], [5.0], 2.0)
        self.assertEqual(raw, -2.0)
        self.assertEqual(bounded, -1.0)

    def test_skill_is_unit_rescaling_invariant_and_bounded(self) -> None:
        self.assertEqual(skill_score(0.5, 1.0), 0.5)
        self.assertEqual(skill_score(5.0, 10.0), 0.5)
        self.assertEqual(skill_score(3.0, 1.0), -1.0)
        self.assertEqual(skill_score(0.0, 0.0), 0.0)
        self.assertEqual(skill_score(0.1, 0.0), -1.0)

    def test_partial_coverage_is_penalized_once(self) -> None:
        rows = [
            {
                "target": target,
                "horizon_hours": 1,
                "n": 8,
                "eligible_rows": 10,
                "skill_score": 0.5,
                "normalized_mean_reward": 0.25,
                "objective_quality": 0.8,
            }
            for target in DEFAULT_TARGET_WEIGHTS
        ]

        result = aggregate_greenhouse_objective(
            rows,
            (1,),
            target_weights=DEFAULT_TARGET_WEIGHTS,
        )

        self.assertAlmostEqual(result["weighted_skill_score"], 0.2)
        self.assertAlmostEqual(result["weighted_normalized_mean_reward"], 0.0)
        self.assertAlmostEqual(result["objective_effective_weight_coverage"], 0.8)

    def test_missing_cell_keeps_its_denominator_weight(self) -> None:
        rows = [
            {
                "target": "air_temperature",
                "horizon_hours": 1,
                "n": 1,
                "eligible_rows": 1,
                "skill_score": 1.0,
                "normalized_mean_reward": 1.0,
            }
        ]

        result = aggregate_greenhouse_objective(
            rows,
            (1,),
            target_weights=DEFAULT_TARGET_WEIGHTS,
        )

        self.assertAlmostEqual(result["weighted_skill_score"], -1.0 / 3.0)
        self.assertAlmostEqual(
            result["weighted_normalized_mean_reward"], -1.0 / 3.0
        )
        self.assertEqual(result["objective_missing_task_count"], 2)
        self.assertAlmostEqual(result["objective_weight_coverage"], 1.0 / 3.0)

    def test_row_order_does_not_change_explicit_weights(self) -> None:
        rows = [
            {
                "target": target,
                "horizon_hours": horizon,
                "n": 100 if target == "air_temperature" else 1,
                "eligible_rows": 100 if target == "air_temperature" else 1,
                "skill_score": value,
                "normalized_mean_reward": value / 2,
            }
            for target, value in (
                ("air_temperature", 0.9),
                ("relative_humidity", 0.3),
                ("co2_concentration", 0.0),
            )
            for horizon in (1, 6)
        ]

        forward = aggregate_greenhouse_objective(
            rows, (1, 6), target_weights=DEFAULT_TARGET_WEIGHTS
        )
        reverse = aggregate_greenhouse_objective(
            list(reversed(rows)), (1, 6), target_weights=DEFAULT_TARGET_WEIGHTS
        )

        self.assertAlmostEqual(forward["weighted_skill_score"], 0.4)
        self.assertEqual(forward, reverse)

    def test_duplicate_unknown_and_invalid_inputs_fail_closed(self) -> None:
        row = {
            "target": "air_temperature",
            "horizon_hours": 1,
            "n": 1,
            "skill_score": 0.2,
            "normalized_mean_reward": 0.1,
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_greenhouse_objective(
                [row, dict(row)], (1,), target_weights=DEFAULT_TARGET_WEIGHTS
            )
        with self.assertRaisesRegex(ValueError, "unknown target"):
            aggregate_greenhouse_objective(
                [{**row, "target": "yield"}],
                (1,),
                target_weights=DEFAULT_TARGET_WEIGHTS,
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_greenhouse_objective(
                [], (1, 1), target_weights=DEFAULT_TARGET_WEIGHTS
            )
        with self.assertRaisesRegex(ValueError, "weights"):
            aggregate_greenhouse_objective(
                [],
                (1,),
                target_weights={
                    "air_temperature": math.nan,
                    "relative_humidity": 0.5,
                    "co2_concentration": 0.5,
                },
            )


if __name__ == "__main__":
    unittest.main()
