import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from ecologyrsi_dsh import EventLedger, EvolutionDirector, FakeDSHAdapter, TaskManifest
from ecologyrsi_dsh.api.projection import _run_execution_progress


class AdaptiveReflectionProgressTests(unittest.TestCase):
    def project(self, *, reflection_status="started", run_status="running", retry=None):
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            director.start_evolution(TaskManifest(
                task_id="reflection-progress", objective="Show active reflection",
                domain_pack="crop-soil-water@toy", visible_datasets=("generated-toy-series@1",),
                budget={"max_candidates": 8, "max_generations": 2, "candidates_per_generation": 4}, seed=3,
            ), run_id="run:reflection-progress")
            state = director.state("run:reflection-progress")
        events = (
            SimpleNamespace(seq=10, kind="EvolutionStageRecorded", created_at="2026-10-21T04:14:35+00:00",
                            payload={"generation": 0, "stage": "reflection", "status": reflection_status,
                                     "candidate_id": None}),
            SimpleNamespace(seq=11, kind="DshChildLaunchReserved", created_at="2026-10-21T04:14:36+00:00",
                            payload={"launch": {"stage": "generation.reflect", "role": "reflector",
                                                "reservation_id": "reflection-child", "launch_attempt": 1}}),
        )
        state = replace(state, events=state.events + events,
                        run=replace(state.run, status=type(state.run.status)(run_status)))
        adaptive = {"generation": 0, "evaluation_phase": "decision", "epoch_progress_percent": 100.0,
                    "current_candidate_id": None, "run_completed_origins": 300, "run_total_origins": 600,
                    "completed_samples": 300, "total_samples": 300}
        with patch("ecologyrsi_dsh.api.projection._adaptive_progress_projection", return_value=adaptive), \
             patch("ecologyrsi_dsh.api.projection._gateway_retry_projection", return_value=retry):
            return _run_execution_progress(state)

    def test_active_reflection_keeps_model_activity_after_evaluation_finishes(self):
        progress = self.project()
        self.assertEqual(progress["current_stage"], "reflection")
        self.assertEqual(progress["phase"], "reflection")
        self.assertIsNone(progress["current_candidate_id"])
        self.assertEqual(progress["progress_percent"], 50.0)
        self.assertEqual(progress["stage_progress"]["completed_samples"], 300)
        self.assertEqual(progress["dsh_activity"]["dsh_stage"], "generation.reflect")
        self.assertEqual(progress["dsh_activity"]["unresolved_child_count"], 1)

    def test_completed_reflection_does_not_revive_old_stage(self):
        progress = self.project(reflection_status="completed")
        self.assertEqual(progress["current_stage"], "decision")
        self.assertIsNone(progress["dsh_activity"])

    def test_terminal_run_hides_reflection_activity(self):
        progress = self.project(run_status="completed")
        self.assertEqual(progress["phase"], "completed")
        self.assertIsNone(progress["current_stage"])
        self.assertIsNone(progress["dsh_activity"])

    def test_paused_run_preserves_context_without_live_activity(self):
        progress = self.project(run_status="paused")
        self.assertEqual(progress["phase"], "paused")
        self.assertEqual(progress["current_stage"], "reflection")
        self.assertIsNone(progress["dsh_activity"])


if __name__ == "__main__":
    unittest.main()
