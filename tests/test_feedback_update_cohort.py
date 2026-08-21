from __future__ import annotations

import math
import unittest

from ecologyrsi_dsh.evaluators.registry import _select_feedback_update_cohort


def _rows(*, rows_per_task: int = 120) -> list[dict]:
    rows = []
    for target in ("air_temperature", "relative_humidity", "co2_concentration"):
        for horizon in (1, 6, 24):
            for index in range(rows_per_task):
                rows.append(
                    {
                        "partition": "training_feedback",
                        "target": target,
                        "horizon_hours": horizon,
                        "origin_timestamp": 10_000 + index,
                        "timestamp": 10_000 + index + horizon,
                        "observed": float(index),
                        "baseline": float(index - 1),
                        "predicted": float(index - 0.5),
                    }
                )
    return rows


def _identities(rows: list[dict]) -> list[tuple]:
    return [
        (
            row["target"],
            row["horizon_hours"],
            row["origin_timestamp"],
            row["timestamp"],
        )
        for row in rows
    ]


class FeedbackUpdateCohortTests(unittest.TestCase):
    def select(self, rows: list[dict], generation: int, limit: int = 500):
        return _select_feedback_update_cohort(
            rows,
            generation=generation,
            samples_per_update=limit,
            dataset_digest="d" * 64,
            split_manifest_digest="s" * 64,
        )

    def test_exact_limit_is_stratified_and_rotates_without_early_reuse(self) -> None:
        population = _rows()
        first, first_evidence = self.select(population, 0)
        second, second_evidence = self.select(population, 1)

        self.assertEqual(len(first), 500)
        self.assertEqual(len(second), 500)
        self.assertEqual(first_evidence["window_offset"], 0)
        self.assertEqual(second_evidence["window_offset"], 500)
        self.assertEqual(first_evidence["population_count"], 1080)
        self.assertEqual(
            first_evidence["population_digest"],
            second_evidence["population_digest"],
        )
        self.assertEqual(first_evidence["deferred_count"], 580)
        first_counts = [item["selected_count"] for item in first_evidence["tasks"]]
        second_counts = [item["selected_count"] for item in second_evidence["tasks"]]
        self.assertLessEqual(max(first_counts) - min(first_counts), 1)
        self.assertLessEqual(max(second_counts) - min(second_counts), 1)
        self.assertTrue(set(_identities(first)).isdisjoint(_identities(second)))

    def test_same_generation_digest_ignores_candidate_values_and_input_order(self) -> None:
        rows = _rows(rows_per_task=8)
        candidate_variant = [
            {
                **row,
                "observed": float(row["observed"]) + 99.0,
                "predicted": float(row["predicted"]) - 42.0,
                "label_free_context": {"candidate_parameter": math.pi},
            }
            for row in reversed(rows)
        ]

        selected_a, evidence_a = self.select(rows, 2, limit=17)
        selected_b, evidence_b = self.select(candidate_variant, 2, limit=17)

        self.assertEqual(_identities(selected_a), _identities(selected_b))
        self.assertEqual(evidence_a["cohort_digest"], evidence_b["cohort_digest"])
        self.assertEqual(evidence_a, evidence_b)

    def test_full_population_is_stable_and_digest_repeats_across_generations(self) -> None:
        rows = _rows(rows_per_task=2)
        first, first_evidence = self.select(rows, 0, limit=500)
        later, later_evidence = self.select(rows, 9, limit=500)

        self.assertEqual(_identities(first), _identities(later))
        self.assertEqual(first_evidence["selected_count"], len(rows))
        self.assertEqual(first_evidence["window_offset"], 0)
        self.assertEqual(later_evidence["window_offset"], 0)
        self.assertEqual(
            first_evidence["cohort_digest"], later_evidence["cohort_digest"]
        )

    def test_repeated_computation_is_resume_idempotent(self) -> None:
        rows = _rows(rows_per_task=70)
        first = self.select(rows, 3, limit=113)
        replay = self.select(rows, 3, limit=113)
        self.assertEqual(first, replay)

    def test_window_wraps_without_duplicates_inside_the_update(self) -> None:
        rows = _rows()
        selected, evidence = self.select(rows, 2)
        self.assertEqual(evidence["window_offset"], 1000)
        self.assertTrue(evidence["window_wraps"])
        self.assertEqual(len(selected), 500)
        self.assertEqual(len(set(_identities(selected))), 500)
        counts = [item["selected_count"] for item in evidence["tasks"]]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_7125_population_wraps_on_update_15_without_reuse_before_cycle(self) -> None:
        population = _rows(rows_per_task=792)[:-3]
        self.assertEqual(len(population), 7_125)

        selections = [
            self.select(population, generation)[0] for generation in range(15)
        ]
        fifteenth, evidence = self.select(population, 14)
        flattened = [identity for rows in selections for identity in _identities(rows)]

        self.assertEqual(evidence["window_offset"], 7_000)
        self.assertTrue(evidence["window_wraps"])
        self.assertEqual(len(fifteenth), 500)
        self.assertEqual(len(set(_identities(fifteenth))), 500)
        self.assertEqual(len(set(flattened[:7_125])), 7_125)
        self.assertEqual(flattened[7_125:], flattened[:375])

    def test_exhausted_short_tasks_are_not_padded_before_global_wrap(self) -> None:
        population = [
            {
                "partition": "training_feedback",
                "target": target,
                "horizon_hours": 1,
                "origin_timestamp": index,
                "timestamp": index + 1,
                "observed": float(index),
                "baseline": float(index),
                "predicted": float(index),
            }
            for target, count in (
                ("air_temperature", 6),
                ("relative_humidity", 2),
                ("co2_concentration", 1),
            )
            for index in range(count)
        ]

        first, _first_evidence = self.select(population, 0, limit=4)
        second, second_evidence = self.select(population, 1, limit=4)

        self.assertTrue(set(_identities(first)).isdisjoint(_identities(second)))
        self.assertEqual(
            {
                item["target"]: item["selected_count"]
                for item in second_evidence["tasks"]
            },
            {
                "air_temperature": 3,
                "co2_concentration": 0,
                "relative_humidity": 1,
            },
        )
        self.assertFalse(second_evidence["window_wraps"])

    def test_invalid_limit_and_duplicate_identity_fail_closed(self) -> None:
        rows = _rows(rows_per_task=1)
        with self.assertRaisesRegex(ValueError, "samples_per_update"):
            self.select(rows, 0, limit=0)
        with self.assertRaisesRegex(ValueError, "duplicate sample identities"):
            self.select(rows + [dict(rows[0])], 0, limit=3)


if __name__ == "__main__":
    unittest.main()
