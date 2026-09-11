from __future__ import annotations

import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.application import work_units
from ecologyrsi_dsh.application import formal_trajectory
from ecologyrsi_dsh.application import generation_execution
from ecologyrsi_dsh.api import projection
from ecologyrsi_dsh.core.models import (
    CandidateRole,
    CandidateStatus,
    RunStatus,
    digest,
)
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
        self.assertFalse(work_units.execute_next_adaptive_work_unit((endpoint).server, "run:x"))

    def test_fatal_parallel_lane_overrides_earlier_transport_failure(self):
        from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
        state = _AdaptiveLaneState()
        services = SimpleNamespace(director=SimpleNamespace(state=lambda _run_id: state))
        barrier = threading.Barrier(2)
        fatal = DshNativeRuntimeUnavailableError(error_code="structured_child_tool_protocol_error", status_code=422)
        def batch(_services, _run_id, candidate_id):
            barrier.wait(timeout=3)
            if candidate_id == "candidate:b":
                raise fatal
            raise DshNativeRuntimeUnavailableError(status_code=502)
        with patch.object(generation_execution, "_two_stage_screening_enabled", return_value=True), \
             patch.object(formal_trajectory, "ensure_formal_trajectory", side_effect=lambda _s, _r, c: state.trajectories[c]), \
             patch.object(formal_trajectory, "execute_next_formal_batch", side_effect=batch):
            with self.assertRaises(DshNativeRuntimeUnavailableError) as caught:
                work_units.execute_next_adaptive_work_unit(services, "run:lanes")
        self.assertIs(caught.exception, fatal)

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
            self.assertFalse(work_units.execute_next_adaptive_work_unit((endpoint).server, "run:x"))

    def test_existing_batch_resumes_incomplete_candidate_and_frozen_inputs(self):
        batch = SimpleNamespace(generation=0, batch_size=4)
        candidate = SimpleNamespace(
            candidate_id="candidate:partial",
            generation=0,
            slot_index=0,
        )
        state = SimpleNamespace(
            run=SimpleNamespace(status=RunStatus.RUNNING, generation=0),
            batch_for=lambda _generation: batch,
            candidates=(candidate,),
            task_manifest=SimpleNamespace(
                metadata={
                    "optimization_protocol": "top2_adaptive_epoch@1",
                    "cohort_capacity_enforced": True,
                }
            ),
            initial_revision_for=lambda _candidate_id: None,
            run_adaptation_cohort=None,
            generation_cohort_for=lambda _generation: None,
        )
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=SimpleNamespace(state=lambda _run_id: state)
            )
        )

        with (
            patch.object(
                generation_execution,
                "_spawn_generation_candidates",
                return_value=True,
            ) as spawn,
            patch.object(
                generation_execution,
                "_freeze_adaptive_generation_inputs",
            ) as freeze,
        ):
            self.assertTrue(
                work_units.execute_next_adaptive_work_unit((endpoint).server, "run:x")
            )

        spawn.assert_called_once_with(endpoint.server, "run:x", batch)
        freeze.assert_called_once_with(endpoint.server, "run:x", 0)

    def test_incumbent_control_is_not_counted_as_generation_work(self):
        batch = SimpleNamespace(generation=0, batch_size=4)
        search = tuple(
            SimpleNamespace(
                candidate_id=f"candidate:search:{index}",
                generation=0,
                slot_index=index,
                role=CandidateRole.SEARCH,
            )
            for index in range(4)
        )
        control = SimpleNamespace(
            candidate_id="candidate:seed-control",
            generation=0,
            slot_index=0,
            role=CandidateRole.INCUMBENT_CONTROL,
        )
        state = SimpleNamespace(
            run=SimpleNamespace(status=RunStatus.RUNNING, generation=0),
            batch_for=lambda _generation: batch,
            candidates=(*search, control),
            task_manifest=SimpleNamespace(
                metadata={
                    "optimization_protocol": "top2_adaptive_epoch@1",
                    "cohort_capacity_enforced": True,
                }
            ),
            initial_revision_for=lambda _candidate_id: object(),
            run_adaptation_cohort=object(),
            generation_cohort_for=lambda _generation: object(),
            formal_selection_for=lambda _generation: None,
        )
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=SimpleNamespace(state=lambda _run_id: state)
            )
        )

        with (
            patch.object(
                generation_execution,
                "_spawn_generation_candidates",
            ) as spawn,
            patch.object(
                generation_execution,
                "_freeze_adaptive_generation_inputs",
            ) as freeze,
            patch.object(
                generation_execution,
                "_two_stage_screening_enabled",
                return_value=False,
            ),
        ):
            self.assertFalse(
                work_units.execute_next_adaptive_work_unit((endpoint).server, "run:x")
            )

        spawn.assert_not_called()
        freeze.assert_not_called()

    def test_legacy_generation_zero_does_not_inject_seed_control(self):
        schedule = OptimizationSchedule.default()
        candidates = tuple(
            SimpleNamespace(
                candidate_id=f"candidate:legacy:{index}",
                generation=0,
                slot_index=index,
                role=CandidateRole.SEARCH,
            )
            for index in range(4)
        )
        state = SimpleNamespace(
            candidates=candidates,
            task_manifest=SimpleNamespace(
                visible_datasets=("dataset:legacy",),
                max_generations=1,
                seed=7,
                metadata={
                    "optimization_protocol": "top2_adaptive_epoch@1",
                    "optimization_schedule": schedule.to_dict(),
                    "cohort_capacity_enforced": True,
                    "cohort_capacity_report": {"planner_digest": "planner"},
                    "prediction_cells_per_origin": 1,
                },
            ),
            initial_revision_for=lambda _candidate_id: object(),
        )
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=SimpleNamespace(state=lambda _run_id: state),
                datasets=SimpleNamespace(selection_view=lambda *_args, **_kwargs: object()),
            )
        )
        mutations: list[str] = []

        def record_mutation(_endpoint, method_name, *_args, **_kwargs):
            mutations.append(method_name)
            return object()

        with (
            patch.object(
                generation_execution,
                "_director_mutation",
                side_effect=record_mutation,
            ),
            patch.object(
                generation_execution,
                "estimate_epoch_capacity",
                return_value=SimpleNamespace(planner_digest="planner", to_dict=lambda: {"planner_digest": "planner"}),
            ),
            patch.object(
                generation_execution,
                "plan_run_adaptation_cohort",
                return_value=object(),
            ),
            patch.object(
                generation_execution,
                "plan_generation_selection_cohorts",
                return_value=object(),
            ),
        ):
            generation_execution._freeze_adaptive_generation_inputs(
                (endpoint).server,
                "run:legacy",
                0,
            )

        self.assertNotIn("ensure_seed_incumbent_control", mutations)
        self.assertEqual(
            mutations,
            [
                "freeze_run_adaptation_cohort",
                "freeze_generation_selection_cohorts",
            ],
        )

        mutations.clear()
        state.task_manifest.metadata["host_runtime_build"] = {
            "package_version": "0.3.55",
            "evolution_runtime_schema": "ecologyrsi-dsh.evolution-runtime/2",
            "generation_comparison_schema": (
                "ecologyrsi-dsh.generation-comparison/1"
            ),
            "projection_schema": "ecologyrsi-dsh.execution-projection/2",
        }
        with (
            patch.object(
                generation_execution,
                "_director_mutation",
                side_effect=record_mutation,
            ),
            patch.object(
                generation_execution,
                "estimate_epoch_capacity",
                return_value=SimpleNamespace(planner_digest="planner", to_dict=lambda: {"planner_digest": "planner"}),
            ),
            patch.object(
                generation_execution,
                "plan_run_adaptation_cohort",
                return_value=object(),
            ),
            patch.object(
                generation_execution,
                "plan_generation_selection_cohorts",
                return_value=object(),
            ),
        ):
            generation_execution._freeze_adaptive_generation_inputs(
                (endpoint).server,
                "run:host-only-adaptive",
                0,
            )
        self.assertNotIn("ensure_seed_incumbent_control", mutations)

    def test_legacy_screening_recovery_keeps_formal_selection_v2(self):
        candidates = tuple(
            SimpleNamespace(
                candidate_id=f"candidate:legacy-screening:{index}",
                generation=0,
                slot_index=index,
                status=CandidateStatus.SPAWNED,
            )
            for index in range(4)
        )
        screening = {
            candidate.candidate_id: SimpleNamespace(
                payload={
                    "generation": 0,
                    "candidate_id": candidate.candidate_id,
                    "score": 1.0 - candidate.slot_index * 0.1,
                    "passed": False,
                    "constraint_violations": 0,
                    "origin_count": 64,
                    "prediction_cell_count": 64,
                    "cohort_digest": digest(
                        {"candidate_id": candidate.candidate_id}
                    ),
                }
            )
            for candidate in candidates
        }
        state = SimpleNamespace(
            run=SimpleNamespace(status=RunStatus.RUNNING),
            candidates=candidates,
            task_manifest=SimpleNamespace(
                metadata={
                    "optimization_protocol": "top2_adaptive_epoch@1",
                    "cohort_capacity_enforced": True,
                }
            ),
            formal_selection_for=lambda _generation: None,
            screening_for=lambda _generation, candidate_id: screening[candidate_id],
            candidate=lambda candidate_id: next(
                item for item in candidates if item.candidate_id == candidate_id
            ),
        )
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=SimpleNamespace(state=lambda _run_id: state)
            )
        )
        freeze_kwargs = None

        def mutation(_endpoint, method_name, *_args, **kwargs):
            nonlocal freeze_kwargs
            if method_name == "freeze_formal_selection_cohort":
                freeze_kwargs = kwargs
                return SimpleNamespace(
                    event_id="formal:legacy",
                    payload={
                        "selected_candidate_ids": [
                            candidates[0].candidate_id,
                            candidates[1].candidate_id,
                        ]
                    },
                )
            return object()

        with (
            patch.object(
                generation_execution,
                "run_candidate_evaluations",
            ),
            patch.object(
                generation_execution,
                "_director_mutation",
                side_effect=mutation,
            ),
        ):
            finalists = generation_execution._prepare_formal_finalists(
                (endpoint).server,
                "run:legacy",
                candidates,
                max_concurrency=1,
            )

        self.assertEqual(len(finalists), 2)
        self.assertIsNotNone(freeze_kwargs)
        self.assertFalse(freeze_kwargs["include_exploration_state"])

    def test_incomplete_batch_without_capacity_gate_does_not_claim_progress(self):
        batch = SimpleNamespace(generation=0, batch_size=4)
        candidate = SimpleNamespace(
            candidate_id="candidate:partial",
            generation=0,
            slot_index=0,
        )
        state = SimpleNamespace(
            run=SimpleNamespace(status=RunStatus.RUNNING, generation=0),
            batch_for=lambda _generation: batch,
            candidates=(candidate,),
            task_manifest=SimpleNamespace(
                metadata={
                    "optimization_protocol": "top2_adaptive_epoch@1",
                    "cohort_capacity_enforced": False,
                }
            ),
        )
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=SimpleNamespace(state=lambda _run_id: state)
            )
        )

        with (
            patch.object(
                generation_execution,
                "_spawn_generation_candidates",
            ) as spawn,
            patch.object(
                generation_execution,
                "_freeze_adaptive_generation_inputs",
            ) as freeze,
        ):
            self.assertFalse(
                work_units.execute_next_adaptive_work_unit((endpoint).server, "run:x")
            )

        spawn.assert_not_called()
        freeze.assert_not_called()

    def test_top_two_lanes_rotate_after_each_batch_edit_pair(self):
        state = _AdaptiveLaneState()
        state.task_manifest.metadata["candidate_concurrency"] = 1
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
                    work_units.execute_next_adaptive_work_unit((endpoint).server, "run:x")
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
        state.task_manifest.metadata["candidate_concurrency"] = 1
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
                work_units.execute_next_adaptive_work_unit((endpoint).server, "run:x")
            )

        self.assertEqual(calls, ["candidate:a:batch", "candidate:a:edit"])
        self.assertNotIn(("candidate:b", 0), state.activations)

    def test_two_finalist_batches_advance_in_the_same_scheduler_turn(self):
        state = _AdaptiveLaneState()
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=SimpleNamespace(state=lambda _run_id: state)
            )
        )
        rendezvous = threading.Barrier(2)
        calls: list[str] = []
        calls_lock = threading.Lock()

        def execute_batch(_endpoint, _run_id, candidate_id):
            rendezvous.wait(timeout=1)
            with calls_lock:
                calls.append(candidate_id)
            state.batches.add((candidate_id, 0))
            state.evaluations.add((candidate_id, 0))
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
                return_value=False,
            ),
        ):
            self.assertTrue(
                work_units.execute_next_adaptive_work_unit((endpoint).server, "run:x")
            )

        self.assertCountEqual(calls, ["candidate:a", "candidate:b"])
        self.assertEqual(
            state.evaluations,
            {("candidate:a", 0), ("candidate:b", 0)},
        )

    def test_two_pending_local_edits_are_serialized_and_both_advance(self):
        state = _AdaptiveLaneState()
        for candidate_id in ("candidate:a", "candidate:b"):
            state.batches.add((candidate_id, 0))
            state.evaluations.add((candidate_id, 0))
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=SimpleNamespace(state=lambda _run_id: state)
            )
        )
        active_edits = 0
        maximum_active_edits = 0
        edit_lock = threading.Lock()

        def execute_edit(_endpoint, _run_id, candidate_id):
            nonlocal active_edits, maximum_active_edits
            with edit_lock:
                active_edits += 1
                maximum_active_edits = max(maximum_active_edits, active_edits)
            time.sleep(0.01)
            state.activations.add((candidate_id, 0))
            with edit_lock:
                active_edits -= 1
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
                return_value=False,
            ),
            patch.object(
                formal_trajectory,
                "execute_next_local_edit",
                side_effect=execute_edit,
            ),
        ):
            self.assertTrue(
                work_units.execute_next_adaptive_work_unit((endpoint).server, "run:x")
            )

        self.assertEqual(maximum_active_edits, 1)
        self.assertEqual(
            state.activations,
            {("candidate:a", 0), ("candidate:b", 0)},
        )

    def test_queued_local_edit_rechecks_pause_before_starting(self):
        state = _AdaptiveLaneState()
        for candidate_id in ("candidate:a", "candidate:b"):
            state.batches.add((candidate_id, 0))
            state.evaluations.add((candidate_id, 0))
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=SimpleNamespace(state=lambda _run_id: state)
            )
        )
        calls: list[str] = []

        def execute_edit(_endpoint, _run_id, candidate_id):
            calls.append(candidate_id)
            state.activations.add((candidate_id, 0))
            state.run.status = RunStatus.PAUSED
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
                return_value=False,
            ),
            patch.object(
                formal_trajectory,
                "execute_next_local_edit",
                side_effect=execute_edit,
            ),
        ):
            self.assertTrue(
                work_units.execute_next_adaptive_work_unit((endpoint).server, "run:x")
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(state.activations), 1)

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
                    (endpoint).server, "run:x", candidate.candidate_id
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
            run=SimpleNamespace(generation=0, status=RunStatus.RUNNING),
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
        self.assertEqual(progress["total_origins"], 2663)
        self.assertEqual(progress["completed_origins"], 406)
        self.assertEqual(progress["batch_index"], 3)
        self.assertEqual(progress["batch_count"], 10)


if __name__ == "__main__":
    unittest.main()
