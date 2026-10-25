from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import patch

from ecologyrsi_dsh.api.projection import (
    _adaptive_progress_projection,
    _execution_diagnostics,
)


class ExecutedProgressTests(TestCase):
    def state(self):
        def evaluation(generation, origins, **metrics):
            return NS(scope=NS(generation=generation, origin_count=origins), metrics=metrics)

        events = []
        for generation in range(2):
            for kind, payload in (
                ("FormalSelectionCohortFrozen", {"generation": generation}),
                ("HoldoutArmStarted", {"generation": generation}),
                ("ScreeningRepeatabilityRecorded", {"generation": generation,
                    "scope": {"origin_count": 10}}),
                ("GenerationComparisonRecorded", {"comparison": {"generation": generation}}),
            ):
                events.append(NS(kind=kind, payload=payload, seq=len(events) + 1,
                    created_at="2026-10-25T00:00:00+00:00"))
        return NS(
            task_manifest=NS(metadata={
                "optimization_protocol": "top2_adaptive_epoch@1",
                "optimization_schedule": {
                    "screening_origin_count": 10,
                    "formal_origin_count_per_finalist": 20,
                    "selection_holdout_origin_count": 50,
                    "local_batch_origin_count": 10,
                }, "prediction_cells_per_origin": 9,
            }, max_generations=2),
            run=NS(generation=2, status=NS(value="completed")),
            candidates=(), artifacts=(), evaluations=(), proposals=(), formal_batches=(),
            candidate_screening_events=tuple(NS(payload={"generation": g,
                "candidate_id": f"c:{g}:{i}", "origin_count": 10})
                for g in range(2) for i in range(4)),
            formal_batch_evaluations=tuple(
                evaluation(g, 10) for g in range(2) for _ in range(2)
            ) + tuple(evaluation(g, 10, reused_screening_evidence=True,
                additional_prediction_executions=0) for g in range(2)),
            holdout_evaluations=tuple(
                evaluation(g, 50) for g in range(2) for _ in range(2)),
            holdout_rechecks=(evaluation(0, 10),),
            events=tuple(events),
        )

    def test_two_generations_distinguish_executions_from_closed_budget(self):
        # Only 170 executions in the current generation; 70 unused slots
        # close its budget. Prior generation also actually ran a 10-origin recheck.
        with patch("ecologyrsi_dsh.api.projection.build_repeatability_projection",
                   return_value={"total_origins": 10, "completed_origins": 10,
                                 "status": "completed"}):
            progress = _adaptive_progress_projection(self.state())
        self.assertEqual(progress["completed_origins"], 240)
        self.assertEqual(progress["run_completed_origins"], 480)
        self.assertEqual(progress["terminal_skipped_origins"], 70)
        self.assertEqual(progress["executed_origins"], 170)
        self.assertEqual(progress["settled_origins"], 170)
        self.assertEqual(progress["run_executed_origins"], 350)
        self.assertEqual(progress["epoch_progress_percent"], 100)

    def test_diagnostics_include_aa_but_exclude_reused_evidence(self):
        with patch("ecologyrsi_dsh.api.projection._adaptive_progress_projection",
                   return_value={"completed_origins": 240, "executed_origins": 170}):
            diagnostics = _execution_diagnostics(self.state())
        self.assertEqual(diagnostics["repeatability_settled_origins"], 20)
        self.assertEqual(diagnostics["adaptive_all_settled_origins"], 350)
        self.assertEqual(diagnostics["adaptive_scored_cells"], 3150)
        self.assertEqual(diagnostics["adaptive_live_completed_origins"], 170)

    def test_missing_replica_evidence_does_not_turn_budget_into_executions(self):
        with patch("ecologyrsi_dsh.api.projection._sealed_holdout_execution_multiplier", return_value=None):
            progress = _adaptive_progress_projection(self.state())
        self.assertFalse(progress["execution_counts_known"])
        self.assertFalse(progress["run_execution_counts_known"])
        self.assertIsNone(progress["executed_origins"])
        self.assertIsNone(progress["run_executed_origins"])
        self.assertIsNone(progress["settled_origins"])
        self.assertEqual(progress["epoch_progress_percent"], 100)
