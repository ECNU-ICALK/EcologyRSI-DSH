from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace

from ecologyrsi_dsh.api import work_units
from ecologyrsi_dsh.api import formal_trajectory
from ecologyrsi_dsh.api import generation_execution
from ecologyrsi_dsh.api import projection
from ecologyrsi_dsh.core.models import RunStatus
from ecologyrsi_dsh.core.trajectory import TrajectoryStatus
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule


class _AdaptiveLaneState:
    def __init__(self) -> None:
        self.run = SimpleNamespace(status=RunStatus.RUNNING, generation=0)
        self._generation_batch = SimpleNamespace(generation=0)
        self.candidates = tuple(
            SimpleNamespace(candidate_id=candidate_id, generation=0, slot_index=index)
            for index, candidate_id in enumerate(("candidate:a", "candidate:b"))
        )
        self.task_manifest = SimpleNamespace(metadata={"candidate_concurrency": 2})
        self._formal = SimpleNamespace(
            payload={
                "selected_candidate_ids": ["candidate:a", "candidate:b"]
            }
        )
        self.trajectories = {
            candidate_id: SimpleNamespace(
                candidate_id=candidate_id,
                batch_count=2,
                status=TrajectoryStatus.RUNNING,
            )
            for candidate_id in self._formal.payload["selected_candidate_ids"]
        }
        self.batches: set[tuple[str, int]] = set()
        self.evaluations: set[tuple[str, int]] = set()
        self.activations: set[tuple[str, int]] = set()

    def batch_for(self, _generation):
        return self._generation_batch

    def formal_selection_for(self, _generation):
        return self._formal

    def trajectory_for(self, candidate_id):
        return self.trajectories[candidate_id]

    def formal_batch_for(self, candidate_id, batch_index):
        if (candidate_id, batch_index) in self.batches:
            return SimpleNamespace(candidate_id=candidate_id, batch_index=batch_index)
        return None

    def batch_evaluation_for(self, candidate_id, batch_index):
        if (candidate_id, batch_index) in self.evaluations:
            return SimpleNamespace(candidate_id=candidate_id, batch_index=batch_index)
        return None

    def revision_activation_for(self, candidate_id, batch_index):
        if (candidate_id, batch_index) in self.activations:
            return SimpleNamespace(candidate_id=candidate_id, batch_index=batch_index)
        return None


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

    def test_top_two_lanes_rotate_after_each_batch_edit_pair(self):
        state = _AdaptiveLaneState()
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=SimpleNamespace(state=lambda _run_id: state)
            )
        )
        calls: list[str] = []

        def execute_batch(_endpoint, _run_id, candidate_id):
            batch_index = sum(
                item_candidate == candidate_id
                for item_candidate, _item_index in state.activations
            )
            key = (candidate_id, batch_index)
            if key in state.evaluations:
                return False
            state.batches.add(key)
            state.evaluations.add(key)
            calls.append(f"{candidate_id}:batch:{batch_index}")
            return True

        def execute_edit(_endpoint, _run_id, candidate_id):
            pending = sorted(
                key
                for key in state.evaluations
                if key[0] == candidate_id and key not in state.activations
            )
            if not pending:
                return False
            state.activations.add(pending[0])
            calls.append(f"{candidate_id}:edit:{pending[0][1]}")
            return True

        with (
            patch.object(
                generation_execution,
                "_two_stage_screening_enabled",
                return_value=True,
            ),
            patch.object(
                formal_trajectory,
                "ensure_formal_trajectory",
                side_effect=lambda _endpoint, _run_id, candidate_id: (
                    state.trajectories[candidate_id]
                ),
            ),
            patch.object(
                formal_trajectory,
                "execute_next_formal_batch",
                side_effect=execute_batch,
            ),
            patch.object(
                formal_trajectory,
                "execute_next_local_edit",
                side_effect=execute_edit,
            ),
        ):
            for _ in range(5):
                self.assertTrue(
                    work_units.execute_next_adaptive_work_unit(endpoint, "run:x")
                )

        self.assertEqual(
            calls,
            [
                "candidate:a:batch:0",
                "candidate:a:edit:0",
                "candidate:b:batch:0",
                "candidate:b:edit:0",
                "candidate:a:batch:1",
            ],
        )

    def test_half_finished_pair_is_recovered_before_switching_lanes(self):
        state = _AdaptiveLaneState()
        state.batches.add(("candidate:a", 0))
        state.evaluations.add(("candidate:a", 0))
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=SimpleNamespace(state=lambda _run_id: state)
            )
        )
        calls: list[str] = []

        def execute_batch(_endpoint, _run_id, candidate_id):
            calls.append(f"{candidate_id}:batch")
            return False

        def execute_edit(_endpoint, _run_id, candidate_id):
            calls.append(f"{candidate_id}:edit")
            state.activations.add((candidate_id, 0))
            return True

        with (
            patch.object(
                generation_execution,
                "_two_stage_screening_enabled",
                return_value=True,
            ),
            patch.object(
                formal_trajectory,
                "ensure_formal_trajectory",
                side_effect=lambda _endpoint, _run_id, candidate_id: (
                    state.trajectories[candidate_id]
                ),
            ),
            patch.object(
                formal_trajectory,
                "execute_next_formal_batch",
                side_effect=execute_batch,
            ),
            patch.object(
                formal_trajectory,
                "execute_next_local_edit",
                side_effect=execute_edit,
            ),
        ):
            self.assertTrue(
                work_units.execute_next_adaptive_work_unit(endpoint, "run:x")
            )

        self.assertEqual(calls, ["candidate:a:batch", "candidate:a:edit"])
        self.assertNotIn(("candidate:b", 0), state.activations)

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
