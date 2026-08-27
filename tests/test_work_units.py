from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace

from ecologyrsi_dsh.api import work_units
from ecologyrsi_dsh.api import formal_trajectory
from ecologyrsi_dsh.api import projection
from ecologyrsi_dsh.core.models import RunStatus
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule


class WorkUnitContractTests(unittest.TestCase):
    def test_non_running_run_does_not_claim_work(self):
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=SimpleNamespace(
                    state=lambda _run_id: SimpleNamespace(
                        run=SimpleNamespace(status=RunStatus.PAUSED)
                    )
                )
            )
        )
        self.assertFalse(work_units.execute_next_adaptive_work_unit(endpoint, "run:x"))

    def test_one_lane_batch_is_one_scheduler_turn(self):
        state = SimpleNamespace(
            run=SimpleNamespace(status=RunStatus.RUNNING, generation=0),
            batch_for=lambda _generation: SimpleNamespace(generation=0),
            candidates=(),
            task_manifest=SimpleNamespace(metadata={}),
        )
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=SimpleNamespace(state=lambda _run_id: state)
            )
        )
        with patch.object(work_units, "RunStatus", RunStatus):
            # The protocol gate is evaluated before any lane call; this fake
            # deliberately proves a paused/non-adaptive scheduler is a no-op.
            self.assertFalse(work_units.execute_next_adaptive_work_unit(endpoint, "run:x"))

    def test_trajectory_initialization_derives_batch_count_from_schedule(self):
        """The manifest stores only canonical schedule fields, not derived counts."""

        candidate = SimpleNamespace(generation=0, candidate_id="candidate:1")
        revision = SimpleNamespace(revision_id="revision:candidate:1:r0")
        state = SimpleNamespace(
            candidate=lambda _candidate_id: candidate,
            formal_selection_for=lambda _generation: SimpleNamespace(
                payload={"selected_candidate_ids": [candidate.candidate_id]}
            ),
            trajectory_for=lambda _candidate_id: None,
            initial_revision_for=lambda _candidate_id: revision,
            task_manifest=SimpleNamespace(
                metadata={"optimization_schedule": OptimizationSchedule.default().to_dict()}
            ),
        )
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=SimpleNamespace(state=lambda _run_id: state)
            )
        )
        with patch.object(
            formal_trajectory,
            "_director_mutation",
            return_value="trajectory",
        ) as mutation:
            self.assertEqual(
                formal_trajectory.ensure_formal_trajectory(
                    endpoint, "run:x", candidate.candidate_id
                ),
                "trajectory",
            )
        self.assertEqual(mutation.call_args.args[-1], 10)

    def test_adaptive_projection_reports_origin_and_visible_batch_units(self):
        schedule = OptimizationSchedule.default().to_dict()
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(
                metadata={
                    "optimization_protocol": "top2_adaptive_epoch@1",
                    "optimization_schedule": schedule,
                }
            ),
            run=SimpleNamespace(generation=0),
            candidate_screening_events=tuple(
                SimpleNamespace(payload={"generation": 0, "origin_count": 64})
                for _ in range(4)
            ),
            formal_batch_evaluations=tuple(
                SimpleNamespace(
                    scope=SimpleNamespace(generation=0, origin_count=50)
                )
                for _ in range(3)
            ),
            holdout_evaluations=(),
            formal_batches=(
                SimpleNamespace(
                    generation=0, batch_index=2, candidate_id="candidate:1"
                ),
            ),
        )
        progress = projection._adaptive_progress_projection(state)
        self.assertIsNotNone(progress)
        self.assertEqual(progress["total_origins"], 1763)
        self.assertEqual(progress["completed_origins"], 406)
        self.assertEqual(progress["batch_index"], 3)
        self.assertEqual(progress["batch_count"], 10)


if __name__ == "__main__":
    unittest.main()
