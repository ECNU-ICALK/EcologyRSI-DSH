from __future__ import annotations

import unittest

from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule


class OptimizationScheduleTests(unittest.TestCase):
    def test_default_schedule_uses_origin_units(self) -> None:
        schedule = OptimizationSchedule.default()

        self.assertEqual(schedule.formal_origin_count_per_finalist, 500)
        self.assertEqual(schedule.local_batch_origin_count, 50)
        self.assertEqual(schedule.batch_count, 10)
        self.assertEqual(schedule.max_local_edits_per_batch, 2)
        self.assertEqual(schedule.max_local_edits_per_finalist, 20)
        self.assertEqual(schedule.selection_holdout_origin_count, 169)
        self.assertEqual(
            schedule.generation_execution_budget(cells_per_origin=9),
            {
                "screening_candidate_origins": 256,
                "formal_candidate_origins": 1000,
                "holdout_candidate_origins": 507,
                "total_candidate_origins": 1763,
                "total_scoring_cells": 15867,
            },
        )
        self.assertEqual(
            schedule.run_execution_budget(5, cells_per_origin=9)[
                "total_candidate_origins"
            ],
            8815,
        )
        self.assertEqual(
            schedule.run_execution_budget(5, cells_per_origin=9)[
                "total_scoring_cells"
            ],
            79335,
        )
        self.assertEqual(schedule.required_unique_origins(5), 1665)

    def test_schedule_rejects_invalid_values(self) -> None:
        cases = [
            ({"local_batch_origin_count": 64}, "must divide"),
            ({"max_local_edits_per_batch": 0}, "between 1 and 5"),
            ({"max_local_edits_per_batch": 6}, "between 1 and 5"),
            ({"selection_holdout_origin_count": 168}, "at least 169"),
            ({"formal_origin_count_per_finalist": True}, "must be an integer"),
        ]
        for patch, message in cases:
            with self.subTest(patch=patch):
                value = {**OptimizationSchedule.default().to_dict(), **patch}
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    OptimizationSchedule.from_dict(value)

    def test_schedule_rejects_unknown_missing_and_changed_policy_fields(self) -> None:
        default = OptimizationSchedule.default().to_dict()

        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            OptimizationSchedule.from_dict({**default, "samples_per_update": 4500})
        missing = dict(default)
        missing.pop("local_batch_origin_count")
        with self.assertRaisesRegex(ValueError, "missing fields"):
            OptimizationSchedule.from_dict(missing)
        for field, changed in (
            ("schema_version", "legacy@1"),
            ("screening_origin_count", 63),
            ("finalist_count", 3),
            ("local_evaluation_mode", "rescore"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    OptimizationSchedule.from_dict({**default, field: changed})

    def test_budget_inputs_reject_bool_and_non_positive_values(self) -> None:
        schedule = OptimizationSchedule.default()

        for value in (True, 0, -1):
            with self.subTest(cells_per_origin=value):
                with self.assertRaisesRegex((TypeError, ValueError), "cells_per_origin"):
                    schedule.generation_execution_budget(cells_per_origin=value)
        for value in (True, 0, -1):
            with self.subTest(planned_generations=value):
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "planned_generations"
                ):
                    schedule.run_execution_budget(value, cells_per_origin=9)


if __name__ == "__main__":
    unittest.main()
