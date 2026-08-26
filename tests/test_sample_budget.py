"""Contract tests for complete-origin sample budgets."""

from __future__ import annotations

import unittest

from ecologyrsi_dsh.core.sample_budget import (
    complete_origin_count,
    scoring_cell_budget,
)


class SampleBudgetTests(unittest.TestCase):
    def test_complete_origin_budget_rejects_silent_truncation(self) -> None:
        self.assertEqual(complete_origin_count(4_500, 9), 500)
        with self.assertRaisesRegex(ValueError, "4,501.*9.*4,500"):
            complete_origin_count(4_501, 9)

    def test_scoring_budget_is_derived_from_complete_origins(self) -> None:
        self.assertEqual(scoring_cell_budget(500, 9), 4_500)
        self.assertEqual(scoring_cell_budget(169, 9), 1_521)
