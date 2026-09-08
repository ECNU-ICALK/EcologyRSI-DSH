from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ecologyrsi_dsh.application import generation_execution
from ecologyrsi_dsh.integrations.dsh_tools import DshToolAdmissionClosedError
from ecologyrsi_dsh.application.generation_execution import _candidate_signature
from ecologyrsi_dsh.core.models import Evaluation, ModelArtifact, RunStatus
from ecologyrsi_dsh.core.trajectory import (
    EvaluationPhase,
    EvaluationScope,
    FormalBatchComparisonDecision,
    HoldoutArm,
    TrajectoryStatus,
)
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule


_FIXTURE_DIR = Path(__file__).with_name("fixtures")


def _load_top2_screening_golden() -> tuple[tuple[SimpleNamespace, ...], dict, list[str]]:
    payload = json.loads(
        (_FIXTURE_DIR / "top2_screening_golden.json").read_text(encoding="utf-8")
    )
    candidates = tuple(SimpleNamespace(**item) for item in payload["candidates"])
    return candidates, payload["screening"], payload["expected_finalist_ids"]


class _Director:
    def __init__(self, candidate_concurrency: int | None) -> None:
        self.status = RunStatus.RUNNING
        metadata = (
            {}
            if candidate_concurrency is None
            else {"candidate_concurrency": candidate_concurrency}
        )
        self.task_manifest = SimpleNamespace(metadata=metadata)

    def state(self, _run_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            run=SimpleNamespace(status=self.status),
            task_manifest=self.task_manifest,
        )

    def run_status(self, _run_id: str) -> RunStatus:
        return self.status


class CandidateParallelEvaluationTests(unittest.TestCase):
    def test_paired_holdout_uses_retained_final_champion(self) -> None:
        retained_champion_id = "revision:finalist:champion"
        rejected_challenger_id = "revision:finalist:challenger"
        trajectory = SimpleNamespace(
            candidate_id="candidate:finalist",
            status=TrajectoryStatus.COMPLETED,
            batch_count=10,
            initial_revision_id="revision:finalist:r0",
            final_revision_id=retained_champion_id,
        )
        final_comparison = SimpleNamespace(
            decision=FormalBatchComparisonDecision.CHAMPION_RETAINED,
            challenger_revision_id=rejected_challenger_id,
            champion_after_revision_id=retained_champion_id,
        )
        state = SimpleNamespace(
            batch_comparison_for=lambda candidate_id, batch_index: (
                final_comparison
                if candidate_id == trajectory.candidate_id and batch_index == 9
                else None
            )
        )

        revision_id = generation_execution._trajectory_holdout_revision_id(
            state,
            trajectory,
            OptimizationSchedule.default(),
        )

        self.assertEqual(revision_id, retained_champion_id)
        self.assertNotEqual(revision_id, rejected_challenger_id)

        mismatched = SimpleNamespace(
            **{
                **vars(trajectory),
                "final_revision_id": rejected_challenger_id,
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            "paired trajectory final revision is not the durable champion",
        ):
            generation_execution._trajectory_holdout_revision_id(
                state,
                mismatched,
                OptimizationSchedule.default(),
            )
        missing_comparison_state = SimpleNamespace(
            batch_comparison_for=lambda _candidate_id, _batch_index: None
        )
        with self.assertRaisesRegex(
            ValueError,
            "paired trajectory final revision is not the durable champion",
        ):
            generation_execution._trajectory_holdout_revision_id(
                missing_comparison_state,
                trajectory,
                OptimizationSchedule.default(),
            )

    def test_hot_control_callbacks_use_constant_size_status_query_only(self) -> None:
        director = Mock()
        director.run_status.return_value = RunStatus.RUNNING
        director.state.side_effect = AssertionError(
            "hot control callback must not replay run state"
        )

        # A deterministic high-call-count assertion replaces a timing benchmark:
        # every decision performs exactly one fixed-size status lookup regardless
        # of event-stream length, and never asks for a projection replay.
        iterations = 1_000
        for _ in range(iterations):
            self.assertEqual(
                generation_execution._sample_run_control(director, "run:hot"),
                "running",
            )
            self.assertTrue(
                generation_execution._run_admission_open(director, "run:hot")
            )
            self.assertTrue(
                generation_execution._sample_publication_open(
                    director, "run:hot"
                )
            )

        self.assertEqual(director.run_status.call_count, iterations * 3)
        director.state.assert_not_called()

        director.run_status.return_value = RunStatus.PAUSED
        self.assertEqual(
            generation_execution._sample_run_control(director, "run:hot"),
            "paused",
        )
        self.assertFalse(
            generation_execution._run_admission_open(director, "run:hot")
        )
        self.assertTrue(
            generation_execution._sample_publication_open(director, "run:hot")
        )

        director.run_status.return_value = RunStatus.COMPLETED
        self.assertEqual(
            generation_execution._sample_run_control(director, "run:hot"),
            "cancelled",
        )
        self.assertFalse(
            generation_execution._sample_publication_open(director, "run:hot")
        )

    def test_holdout_recovery_requires_exact_revision_scope_and_artifact(self) -> None:
        scope = EvaluationScope(
            run_id="run:holdout-recovery",
            generation=0,
            candidate_id="candidate:finalist",
            candidate_revision_id="revision:finalist:batch:10",
            phase=EvaluationPhase.HOLDOUT,
            cohort_digest="a" * 64,
            origin_count=169,
            holdout_arm=HoldoutArm.FINALIST_1,
        )
        artifact = ModelArtifact(
            artifact_id="artifact:finalist:holdout",
            run_id=scope.run_id,
            candidate_id=scope.candidate_id,
            candidate_revision_id=scope.candidate_revision_id,
            evaluation_scope_digest=scope.scope_key,
            model_id="greenhouse-targetwise-ridge@1",
            dataset_digest="b" * 64,
            training_partition="training_fit",
            training_rows=500,
        )
        evaluation = Evaluation(
            evaluation_id="evaluation:finalist:holdout",
            run_id=scope.run_id,
            candidate_id=scope.candidate_id,
            candidate_revision_id=scope.candidate_revision_id,
            evaluation_scope=scope.to_dict(),
            score=0.2,
            passed=True,
            metrics={"constraint_violations": 0},
            evaluator_digest="c" * 64,
            artifact_digest=artifact.digest,
        )

        self.assertTrue(
            generation_execution._canonical_holdout_outcome_complete(
                artifact, evaluation, scope
            )
        )
        wrong_scope = dict(scope.to_dict())
        wrong_scope["phase"] = EvaluationPhase.SCREENING.value
        screening_evaluation = Evaluation(
            **{
                **evaluation.to_dict(),
                "evaluation_scope": wrong_scope,
            }
        )
        self.assertFalse(
            generation_execution._canonical_holdout_outcome_complete(
                artifact, screening_evaluation, scope
            )
        )
        recovered = generation_execution._holdout_from_canonical_evaluation(
            evaluation, scope
        )
        self.assertEqual(recovered.scope, scope)
        self.assertEqual(recovered.score, evaluation.score)
        self.assertEqual(recovered.metrics, evaluation.metrics)

    def test_top2_screening_selection_is_a_locked_outer_contract(self) -> None:
        candidates, records, expected_ids = _load_top2_screening_golden()

        selected = generation_execution._select_screening_finalists(
            candidates,
            records,
            top_k=2,
        )

        self.assertEqual(
            [candidate.candidate_id for candidate in selected],
            expected_ids,
        )

    def test_pause_admission_closure_keeps_candidate_evaluation_recoverable(
        self,
    ) -> None:
        error = DshToolAdmissionClosedError("run admission is closed")

        self.assertTrue(generation_execution._recoverable_evaluation_error(error))

    def test_unsupported_tool_protocol_is_propagated_to_run_owner(self) -> None:
        from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
        error = DshNativeRuntimeUnavailableError(
            error_code="structured_child_tool_protocol_error", status_code=422)
        self.assertTrue(generation_execution._recoverable_evaluation_error(error))
        self.assertFalse(generation_execution._recoverable_evaluation_error(ValueError("bad prediction")))

    def test_screening_freezes_deterministic_top_two_by_evidence(self) -> None:
        candidates = tuple(
            SimpleNamespace(slot_index=index, candidate_id=f"candidate-{index}")
            for index in range(4)
        )
        screening = {
            "candidate-0": {"score": 0.8, "constraint_violations": 1},
            "candidate-1": {"score": 0.5, "constraint_violations": 0},
            "candidate-2": {
                "score": 0.7,
                "passed": False,
                "constraint_violations": 0,
            },
            "candidate-3": {"score": 0.7, "constraint_violations": 0},
        }

        selected = generation_execution._select_screening_finalists(
            candidates,
            screening,
            top_k=2,
        )

        self.assertEqual(
            [candidate.candidate_id for candidate in selected],
            ["candidate-2", "candidate-3"],
        )

    def test_candidate_signature_distinguishes_agent_behavior(self) -> None:
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(
                visible_datasets=("dataset",),
                metadata={
                    "prediction_model_id": "predictor@1",
                    "prediction_model_digest": "p" * 64,
                    "dataset_digest": "d" * 64,
                    "evaluator_digest": "e" * 64,
                },
            )
        )
        first = SimpleNamespace(
            changes={"ridge_alpha": 0.2},
            metadata={"behavior_digest": "a" * 64},
        )
        second = SimpleNamespace(
            changes={"ridge_alpha": 0.2},
            metadata={"behavior_digest": "b" * 64},
        )

        self.assertNotEqual(
            _candidate_signature(state, first),
            _candidate_signature(state, second),
        )

    def test_stage_ledger_writes_are_serialized_across_candidate_threads(self) -> None:
        lock = threading.RLock()
        state_lock = threading.Lock()
        active = 0
        maximum_active = 0

        class Director:
            def record_evolution_stage(self, *_args: object, **_kwargs: object) -> None:
                nonlocal active, maximum_active
                with state_lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.03)
                with state_lock:
                    active -= 1

        endpoint = SimpleNamespace(
            server=SimpleNamespace(director=Director(), mutation_lock=lock)
        )
        threads = [
            threading.Thread(
                target=generation_execution._record_stage,
                args=(endpoint.server, "run-1", 0, "evaluation", "started"),
                kwargs={"candidate_id": f"candidate-{index}"},
            )
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)

        self.assertEqual(maximum_active, 1)

    def test_generation_integration_overlaps_four_sibling_candidates(self) -> None:
        director = _Director(candidate_concurrency=4)
        endpoint = SimpleNamespace(server=SimpleNamespace(director=director))
        candidates = tuple(
            SimpleNamespace(slot_index=index, candidate_id=f"candidate-{index}")
            for index in range(4)
        )
        barrier = threading.Barrier(4, timeout=3)
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def evaluate(_endpoint: object, _run_id: str, _candidate_id: str) -> None:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            barrier.wait()
            with lock:
                active -= 1

        with patch.object(generation_execution, "_evaluate_candidate", side_effect=evaluate):
            generation_execution._evaluate_generation_candidates(
                (endpoint).server,
                "run-1",
                candidates,
            )

        self.assertEqual(maximum_active, 4)

    def test_pause_stops_queued_candidate_after_admitted_workers_drain(self) -> None:
        director = _Director(candidate_concurrency=2)
        endpoint = SimpleNamespace(server=SimpleNamespace(director=director))
        candidates = tuple(
            SimpleNamespace(slot_index=index, candidate_id=f"candidate-{index}")
            for index in range(3)
        )
        peer_started = threading.Event()
        paused = threading.Event()
        started: list[str] = []

        def evaluate(_endpoint: object, _run_id: str, candidate_id: str) -> None:
            started.append(candidate_id)
            if candidate_id == "candidate-0":
                self.assertTrue(peer_started.wait(3))
                director.status = RunStatus.PAUSED
                paused.set()
            elif candidate_id == "candidate-1":
                peer_started.set()
                self.assertTrue(paused.wait(3))
            else:
                self.fail("queued candidate started after the run paused")

        with patch.object(generation_execution, "_evaluate_candidate", side_effect=evaluate):
            generation_execution._evaluate_generation_candidates(
                (endpoint).server,
                "run-1",
                candidates,
            )

        self.assertEqual(set(started), {"candidate-0", "candidate-1"})

    def test_manifest_without_candidate_concurrency_remains_serial(self) -> None:
        director = _Director(candidate_concurrency=None)
        endpoint = SimpleNamespace(server=SimpleNamespace(director=director))
        candidates = tuple(
            SimpleNamespace(slot_index=index, candidate_id=f"candidate-{index}")
            for index in range(2)
        )
        caller = threading.get_ident()
        observed_threads: list[int] = []

        with patch.object(
            generation_execution,
            "_evaluate_candidate",
            side_effect=lambda *_args: observed_threads.append(threading.get_ident()),
        ):
            generation_execution._evaluate_generation_candidates(
                (endpoint).server,
                "run-1",
                candidates,
            )

        self.assertEqual(observed_threads, [caller, caller])

    def test_legacy_manifest_with_null_candidate_concurrency_remains_serial(self) -> None:
        director = _Director(candidate_concurrency=None)
        director.task_manifest.metadata["candidate_concurrency"] = None
        endpoint = SimpleNamespace(server=SimpleNamespace(director=director))
        candidates = tuple(
            SimpleNamespace(slot_index=index, candidate_id=f"candidate-{index}")
            for index in range(2)
        )
        caller = threading.get_ident()
        observed_threads: list[int] = []

        with patch.object(
            generation_execution,
            "_evaluate_candidate",
            side_effect=lambda *_args: observed_threads.append(threading.get_ident()),
        ):
            generation_execution._evaluate_generation_candidates(
                (endpoint).server,
                "run-1",
                candidates,
            )

        self.assertEqual(observed_threads, [caller, caller])


if __name__ == "__main__":
    unittest.main()
