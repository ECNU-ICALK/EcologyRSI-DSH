from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as _Namespace
from unittest.mock import patch

from ecologyrsi_dsh import EventLedger, EvolutionDirector, TaskManifest
from tests.native_fixtures import NativeRoleStub, native_manifest, start_native_run, spawn_native_candidate
from ecologyrsi_dsh.api.projection import (
    _adaptive_progress_projection,
    _adaptive_trajectory_projection,
    _active_scoped_evaluation_progress,
    _dsh_activity_projection,
    _evolution_evidence_projection,
    _evaluation_progress_projection,
    _evaluation_progress_rates,
    _execution_diagnostics,
    _gateway_retry_projection,
    _projection_json,
    _public_evaluation_metrics,
    _run_failure_projection,
    _rounds_projection,
    _screening_progress_projection,
)
from ecologyrsi_dsh.api.shared import _event_type
from ecologyrsi_dsh.core.models import (
    CandidateRole,
    digest,
)
from ecologyrsi_dsh.core.trajectory import FormalBatchArm
from ecologyrsi_dsh.evolution.schedule import (
    OptimizationSchedule,
)


def _task(*, max_candidates: int = 1, max_generations: int = 1) -> TaskManifest:
    return native_manifest(task_id="projection-trace", generations=max_generations)


def _skill_evidence(stage: str, skill_name: str) -> dict:
    return {
        "schema_version": "ecologyrsi-dsh.skill-invocation-evidence/1",
        "stage": stage,
        "skill_name": skill_name,
        "call_count": 1,
        "successful_call_count": 1,
        "call_seq": 1,
        "result_seq": 2,
        "first_tool_call_verified": True,
        "next_tool_name": "structured_output",
        "next_tool_call_seq": 3,
        "order_verified": True,
        "source": "dsh_session_event_log",
    }


def _origin_members(label: str) -> list[str]:
    return sorted(digest(f"{label}:cell:{index}") for index in range(9))


def SimpleNamespace(**values):
    """Projection-only states explicitly expose frozen candidate bindings."""
    result = _Namespace(**values)
    if 'task_manifest' in values and 'run' in values:
        for key in ('candidates','formal_batches','formal_batch_evaluations','holdout_evaluations','candidate_screening_events'):
            if not hasattr(result,key): setattr(result,key,())
        for index,candidate in enumerate(result.candidates):
            if not hasattr(candidate,'candidate_id'): candidate.candidate_id=f'candidate:{index}'
    if 'task_manifest' in values and 'run' in values and not hasattr(result, 'generation_cohort_for'):
        result.generation_cohort_for = lambda generation: _Namespace(revision_bindings={
            c.candidate_id: 'revision:' + c.candidate_id for c in getattr(result,'candidates',())
            if getattr(c,'generation',generation) == generation})
    return result


class ExecutionProjectionTests(unittest.TestCase):
    def test_zero_execution_race_views_do_not_repeat_physical_work(self) -> None:
        def evaluated(generation, count, *, reuse=False, failed=0):
            return SimpleNamespace(
                # Shared cohort/candidate does not make real parent/challenger
                # executions duplicates. Only the explicit reuse is omitted.
                scope=SimpleNamespace(generation=generation, origin_count=count,
                    candidate_id="candidate:shared", cohort_digest="same-cohort"),
                metrics={"sample_execution": {
                    "succeeded_origin_samples": count - failed,
                    "failed_origin_samples": failed, "failed_examples": failed * 9,
                }, **({"reused_screening_evidence": True,
                       "additional_prediction_executions": 0} if reuse else {})},
            )

        state = SimpleNamespace(
            artifacts=(), evaluations=(), events=(), proposals=(), candidates=(),
            formal_batch_evaluations=tuple(item for generation in range(2) for item in (
                evaluated(generation, 10, reuse=True, failed=1),
                evaluated(generation, 25), evaluated(generation, 25, failed=1),
            )),
            holdout_evaluations=tuple(evaluated(generation, 50)
                for generation in range(2) for _ in range(2)),
            candidate_screening_events=tuple(SimpleNamespace(payload={
                "generation": generation, "origin_count": 10,
            }) for generation in range(2) for _ in range(5)),
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{"prediction_cells_per_origin": 9}},
                max_generations=2),
            run=SimpleNamespace(generation=2, status=SimpleNamespace(value="completed")),
        )
        with patch("ecologyrsi_dsh.api.projection._adaptive_progress_projection",
                return_value={"completed_origins": 600}):
            result = _execution_diagnostics(state)
        self.assertEqual(result["adaptive_evaluation_count"], 10)
        self.assertEqual(result["adaptive_settled_origins"], 300)
        self.assertEqual(result["adaptive_all_settled_origins"], 400)
        self.assertEqual(result["adaptive_scored_cells"], 3600)
        self.assertEqual(result["candidate_work_items"], 3600)
        self.assertEqual(result["adaptive_succeeded_origins"], 298)
        self.assertEqual(result["adaptive_failed_origins"], 2)
        self.assertEqual(result["adaptive_failed_scoring_cells"], 18)
        self.assertEqual(result["adaptive_live_completed_origins"], 600)
        self.assertEqual(result["execution_evidence_status"], "recorded")
        # Conditional confirmation repeats the same origins in two real arms.
        # It is extra physical work, not extra independent observations.
        state.holdout_rechecks = tuple(evaluated(generation, 50)
            for generation in range(2) for _ in range(2))
        with patch("ecologyrsi_dsh.api.projection._adaptive_progress_projection",
                return_value={"completed_origins": 600}):
            repeated = _execution_diagnostics(state)
        self.assertEqual(repeated["adaptive_all_settled_origins"], 600)
        self.assertEqual(repeated["adaptive_scored_cells"], 5400)
        self.assertEqual(repeated["candidate_work_items"], 5400)
        self.assertEqual(repeated["adaptive_evaluation_count"], 14)
        self.assertEqual(repeated["adaptive_succeeded_origins"], 498)
        self.assertEqual(repeated["adaptive_failed_origins"], 2)
        self.assertEqual(repeated["adaptive_failed_scoring_cells"], 18)
        self.assertTrue(repeated["adaptive_execution_counts_known"])
        self.assertTrue(repeated["adaptive_outcome_counts_known"])

    @staticmethod
    def _sealed_replica_state():
        from dataclasses import replace
        from ecologyrsi_dsh.core.trajectory import HoldoutArm
        from tests.test_run_parameter_consistency import small_holdout
        evaluation = small_holdout(HoldoutArm.INCUMBENT, .1)
        events = []
        for index in range(2):
            scope = replace(evaluation.scope, inference_replica=index)
            revision = f"results:{index}"
            events.append(SimpleNamespace(seq=index * 2 + 1,
                kind="EvaluationSampleResultsStarted", payload={
                    "run_id": scope.run_id, "generation": scope.generation,
                    "candidate_id": scope.candidate_id, "revision": revision,
                    "checkpoint": {"execution_scope_digest": scope.scope_key,
                        "evaluation_phase": "holdout", "candidate_revision_id": scope.candidate_revision_id,
                        "cohort_digest": scope.cohort_digest, "holdout_arm": scope.holdout_arm.value,
                        "inference_replica": index, "sample_count": 450}}))
            events.append(SimpleNamespace(seq=index * 2 + 2,
                kind="EvaluationSampleResultBatchRecorded", payload={
                    "candidate_id": scope.candidate_id, "revision": revision,
                    "record_count": 450, "batch_index": 1,
                    "sample_ids": [f"same-environment-cell:{i}" for i in range(450)]}))
        events.append(SimpleNamespace(seq=5, kind="HoldoutEvaluationRecorded",
            payload={"evaluation": evaluation.to_dict()}))
        return SimpleNamespace(artifacts=(), evaluations=(), events=tuple(events), proposals=(), candidates=(),
            formal_batch_evaluations=(), holdout_evaluations=(evaluation,), holdout_rechecks=(),
            candidate_screening_events=(), task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "prediction_cells_per_origin": 9, "optimization_protocol": "joint_plugin_epoch@2",
                "sample_agent_mode": "dsh_native_agent"}}, max_generations=1),
            run=SimpleNamespace(generation=1, status=SimpleNamespace(value="completed")))



    def test_only_explicit_integer_zero_formal_reuse_is_omitted(self) -> None:
        state = SimpleNamespace(
            artifacts=(), evaluations=(), events=(), proposals=(), candidates=(),
            formal_batch_evaluations=(), holdout_evaluations=(),
            candidate_screening_events=(),
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{"prediction_cells_per_origin": 9}},
                max_generations=1),
            run=SimpleNamespace(generation=1, status=SimpleNamespace(value="completed")),
        )
        # Legacy evidence without the complete reuse marker keeps its original
        # count. Malformed race evidence is rejected by state replay, not here.
        cases = (
            ({}, 10),
            ({"reused_screening_evidence": True}, 10),
            ({"additional_prediction_executions": 0}, 10),
            ({"reused_screening_evidence": False, "additional_prediction_executions": 0}, 10),
            ({"reused_screening_evidence": True, "additional_prediction_executions": 5}, 10),
            ({"reused_screening_evidence": True, "additional_prediction_executions": False}, 10),
            ({"reused_screening_evidence": True, "additional_prediction_executions": 0.0}, 10),
            ({"reused_screening_evidence": True, "additional_prediction_executions": 0}, 0),
        )
        for metrics, expected in cases:
            with self.subTest(metrics=metrics), patch(
                    "ecologyrsi_dsh.api.projection._adaptive_progress_projection", return_value=None):
                evaluation = SimpleNamespace(scope=SimpleNamespace(origin_count=10), metrics=metrics)
                state.formal_batch_evaluations = (evaluation,)
                formal = _execution_diagnostics(state)
                self.assertEqual(formal["adaptive_all_settled_origins"], expected)
                self.assertEqual(formal["adaptive_evaluation_count"], 1)
                state.formal_batch_evaluations = ()
                state.holdout_evaluations = (evaluation,)
                self.assertEqual(_execution_diagnostics(state)["adaptive_all_settled_origins"], 10)
                state.holdout_evaluations = ()

    def test_adaptive_round_shows_shared_advisory_review_without_judge_receipts(self) -> None:
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{"optimization_protocol": "joint_plugin_epoch@2"}}),
            run=SimpleNamespace(status=SimpleNamespace(value="completed")),
            generation_comparisons=(SimpleNamespace(generation=0),),
            candidate_screening_events=(SimpleNamespace(payload={"candidate_id": "c:screened"}),),
        )
        row = {"generation": 1, "generation_reflection": {"reflection_id": "r:1"},
            "stages": {"judge": "running", "decision": "pending"},
            "candidates": [{"candidate_id": "c:screened", "stages": {
                "evaluation": "pending", "judge": "pending", "decision": "pending"}}]}
        with patch("ecologyrsi_dsh.api.projection.rounds", return_value=[row]):
            projected = _rounds_projection(state)[0]
        self.assertEqual(projected["adaptive_completion"]["review_scope"], "generation_advisory")
        self.assertTrue(projected["adaptive_completion"]["comparison_recorded"])
        self.assertEqual(projected["stages"]["judge"], "completed")
        self.assertEqual(projected["candidates"][0]["stages"], {
            "evaluation": "completed", "judge": "completed", "decision": "recorded"})

    def test_terminal_adaptive_progress_preserves_completed_generation(self) -> None:
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "optimization_protocol": "joint_plugin_epoch@2",
                "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
            }}, max_generations=2),
            run=SimpleNamespace(generation=1, status=SimpleNamespace(value="completed")),
            candidates=(), formal_batches=(),
            candidate_screening_events=tuple(SimpleNamespace(
                payload={"generation": 0, "candidate_id": f"c:{i}", "origin_count": 10}
            ) for i in range(5)),
            formal_batch_evaluations=tuple(SimpleNamespace(
                scope=SimpleNamespace(generation=0, origin_count=25)
            ) for _ in range(2)),
            holdout_evaluations=tuple(SimpleNamespace(
                scope=SimpleNamespace(generation=0, origin_count=50)
            ) for _ in range(2)),
            events=(SimpleNamespace(kind="GenerationComparisonRecorded",
                payload={"comparison": {"generation": 0}}),),
        )
        progress = _adaptive_progress_projection(state)
        self.assertEqual(progress["generation"], 0)
        self.assertEqual(progress["completed_origins"], 300)
        self.assertEqual(progress["run_completed_origins"], 300)
        self.assertEqual(progress["run_total_origins"], 600)
        self.assertEqual(progress["epoch_progress_percent"], 100)
        self.assertEqual(progress["terminal_skipped_origins"], 100)
        self.assertIsNone(progress["current_candidate_id"])
        # A later failed generation must not inherit the prior completion.
        state.run.status.value = "failed"
        state.events += (SimpleNamespace(kind="GenerationBatchStarted",
            payload={"batch": {"generation": 1}}),)
        self.assertIsNone(_adaptive_progress_projection(state))

    def test_sealed_screening_is_not_aborted_but_next_revision_remains_partial(self) -> None:
        candidate = SimpleNamespace(candidate_id="c:screened", proposal_id="p:screened",
            generation=0, role=CandidateRole.SEARCH, status=SimpleNamespace(value="failed"))
        state = SimpleNamespace(
            artifacts=(), evaluations=(), formal_batch_evaluations=(), holdout_evaluations=(),
            proposals=(), candidates=(candidate,),
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{"prediction_cells_per_origin": 9}}, max_generations=1),
            candidate_screening_events=(SimpleNamespace(payload={"origin_count": 64}),),
            run=SimpleNamespace(generation=1, status=SimpleNamespace(value="completed")),
            events=(
                SimpleNamespace(seq=1, kind="EvaluationSampleResultsStarted",
                    payload={"candidate_id": candidate.candidate_id, "revision": "screen"}),
                SimpleNamespace(seq=2, kind="EvaluationSampleResultsRecorded",
                    payload={"candidate_id": candidate.candidate_id, "revision": "screen"}),
            ),
        )
        with patch("ecologyrsi_dsh.api.projection._evaluation_progress_projection",
                return_value={"completed_samples": 2, "total_samples": 5}), patch(
                "ecologyrsi_dsh.api.projection._evaluation_batch_progress_projection", return_value=None):
            sealed = _execution_diagnostics(state)
            self.assertEqual(sealed["partial_evaluation_aborted_candidate_count"], 0)
            self.assertEqual(sealed["adaptive_all_settled_origins"], 64)
            self.assertEqual(sealed["adaptive_scored_cells"], 576)
            self.assertEqual(sealed["candidate_work_items"], 576)
            self.assertEqual(sealed["execution_evidence_status"], "recorded")
            state.events += (SimpleNamespace(seq=3, kind="EvaluationSampleResultsStarted",
                payload={"candidate_id": candidate.candidate_id, "revision": "new"}),)
            unfinished = _execution_diagnostics(state)
            self.assertEqual(unfinished["partial_evaluation_aborted_candidate_count"], 1)
            self.assertEqual(unfinished["execution_evidence_status"], "aborted_partial")

    def test_seed_control_holdout_progress_is_not_recounted_as_screening(
        self,
    ) -> None:
        candidate_id = "candidate:seed-control"
        revision_id = "revision:seed-control:0"
        cohort_digest = "c" * 64
        events = (
            SimpleNamespace(
                seq=100,
                kind="HoldoutArmStarted",
                payload={
                    "generation": 0,
                    "holdout_arm": "incumbent",
                    "candidate_id": candidate_id,
                    "candidate_revision_id": revision_id,
                    "cohort_digest": cohort_digest,
                    "origin_count": 169,
                },
                created_at="2026-08-30T01:00:00+00:00",
            ),
            SimpleNamespace(
                seq=101,
                kind="EvaluationSampleResultsStarted",
                payload={
                    "candidate_id": candidate_id,
                    "revision": "sample-results:seed-holdout",
                    "checkpoint": {
                        "evaluation_phase": "holdout",
                        "formal_batch_index": None,
                        "holdout_arm": "incumbent",
                        "candidate_revision_id": revision_id,
                        "cohort_digest": cohort_digest,
                        "sample_count": 169,
                    },
                },
                created_at="2026-08-30T01:00:01+00:00",
            ),
            SimpleNamespace(
                seq=102,
                kind="EvaluationProgressRecorded",
                payload={
                    "schema_version": "ecologyrsi-dsh.evaluation-progress/3",
                    "candidate_id": candidate_id,
                    "role": "planner",
                    "revision": "sample-results:seed-holdout",
                    "progress_id": 1,
                    "completed_samples": 12,
                    "total_samples": 169,
                    "succeeded_samples": 12,
                    "failed_samples": 0,
                },
                created_at="2026-08-30T01:00:02+00:00",
            ),
        )
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(
                metadata={**dict(native_manifest().metadata), **{
                    "optimization_protocol": "joint_plugin_epoch@2",
                    "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
                    "prediction_cells_per_origin": 9,
                }},
                max_generations=1,
            ),
            run=SimpleNamespace(
                run_id="run:seed-holdout-progress",
                generation=0,
                status=SimpleNamespace(value="running"),
            ),
            candidate_screening_events=(),
            formal_batch_evaluations=(),
            holdout_evaluations=(),
            formal_batches=(),
            candidates=(
                SimpleNamespace(
                    candidate_id=candidate_id,
                    generation=0,
                    role=CandidateRole.INCUMBENT_CONTROL,
                ),
            ),
            events=events,
        )

        progress = _adaptive_progress_projection(state)

        self.assertIsNotNone(progress)
        self.assertEqual(progress["evaluation_phase"], "holdout")
        self.assertEqual(progress["screening_completed_origins"], 0)
        self.assertEqual(progress["holdout_completed_origins"], 12)
        self.assertEqual(progress["current_candidate_id"], candidate_id)

    def test_seed_control_holdout_is_adaptive_evidence_not_partial_candidate_work(
        self,
    ) -> None:
        control = SimpleNamespace(
            candidate_id="candidate:seed-control",
            proposal_id="proposal:seed-control",
            role=CandidateRole.INCUMBENT_CONTROL,
            status=SimpleNamespace(value="spawned"),
        )
        holdout = SimpleNamespace(
            scope=SimpleNamespace(origin_count=169),
            metrics={
                "sample_execution": {
                    "succeeded_origin_samples": 169,
                    "failed_origin_samples": 0,
                    "failed_examples": 0,
                }
            },
        )
        state = SimpleNamespace(
            artifacts=(),
            evaluations=(),
            formal_batch_evaluations=(),
            holdout_evaluations=(holdout,),
            events=(),
            proposals=(
                SimpleNamespace(
                    proposal_id=control.proposal_id,
                    metadata={**dict(native_manifest().metadata), **{"proposal_source": "seed_incumbent_control"}},
                ),
            ),
            candidates=(control,),
            task_manifest=SimpleNamespace(
                metadata={**dict(native_manifest().metadata), **{}},
                max_generations=1,
                budget={"max_generations": 1},
            ),
            run=SimpleNamespace(
                status=SimpleNamespace(value="completed"),
                generation=1,
            ),
        )

        with patch(
            "ecologyrsi_dsh.api.projection._adaptive_progress_projection",
            return_value=None,
        ):
            diagnostics = _execution_diagnostics(state)

        self.assertEqual(diagnostics["proposal_sources"], {})
        self.assertEqual(diagnostics["live_evaluation_candidate_count"], 0)
        self.assertEqual(diagnostics["adaptive_evaluation_count"], 1)
        self.assertEqual(diagnostics["execution_evidence_status"], "recorded")


    def test_evolution_evidence_requires_committed_champion_identity(
        self,
    ) -> None:
        incumbent = SimpleNamespace(
            scope=SimpleNamespace(
                holdout_arm=SimpleNamespace(value="incumbent"),
                candidate_id="candidate:legacy-incumbent",
                candidate_revision_id="revision:legacy-incumbent",
                cohort_digest="c" * 64,
            ),
            score=0.1,
            passed=True,
        )
        comparison = SimpleNamespace(
            generation=0,
            cohort_digest="c" * 64,
            selected_candidate_id=incumbent.scope.candidate_id,
            selected_revision_id=incumbent.scope.candidate_revision_id,
            holdout_evaluations=(incumbent,),
            gate_results={"selected_arm": "incumbent", "arms": {}},
        )
        base = dict(
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{}}),
            generation_comparisons=(comparison,),
            formal_selection_for=lambda _generation: None,
            analysis_for=lambda _generation: None,
            candidates=(),
            initial_revision_for=lambda _candidate_id: None,
            revision=lambda _revision_id: SimpleNamespace(
                genome_digest="l" * 64,
                behavior_digest="m" * 64,
            ),
        )
        crash_state = SimpleNamespace(
            **base,
            effective_revision_for=lambda _generation: None,
            events=(),
        )

        crash_evidence = _evolution_evidence_projection(crash_state)

        self.assertIsNone(crash_evidence["global_champion"])
        self.assertFalse(crash_evidence["generation_decisions"][0]["committed"])

        legacy_state = SimpleNamespace(
            **base,
            effective_revision_for=lambda generation: (
                incumbent.scope.candidate_revision_id if generation == 0 else None
            ),
            events=(
                SimpleNamespace(
                    kind="GenerationChampionSelected",
                    payload={
                        "generation": 0,
                        "selected_candidate_id": incumbent.scope.candidate_id,
                        "selected_revision_id": incumbent.scope.candidate_revision_id,
                    },
                ),
            ),
        )

        legacy_evidence = _evolution_evidence_projection(legacy_state)

        self.assertFalse(legacy_evidence["global_champion"]["is_initial_seed"])
        self.assertEqual(legacy_evidence["protocol_scope"], "global_incumbent_v2")
        self.assertEqual(
            legacy_evidence["global_champion"]["source"],
            "generation_holdout",
        )

    def test_retained_incumbent_keeps_its_original_promotion_generation(self) -> None:
        def retained(generation: int, score: float) -> SimpleNamespace:
            evaluation = SimpleNamespace(
                scope=SimpleNamespace(
                    holdout_arm=SimpleNamespace(value="incumbent"),
                    candidate_id="candidate:champion",
                    candidate_revision_id="revision:champion",
                    cohort_digest=("a" if generation == 0 else "b") * 64,
                ),
                score=score,
                passed=True,
            )
            return SimpleNamespace(
                generation=generation,
                cohort_digest=evaluation.scope.cohort_digest,
                selected_candidate_id=evaluation.scope.candidate_id,
                selected_revision_id=evaluation.scope.candidate_revision_id,
                holdout_evaluations=(evaluation,),
                gate_results={"selected_arm": "incumbent", "arms": {}},
            )

        comparisons = (retained(0, 0.2), retained(1, 0.3))
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{}}),
            generation_comparisons=comparisons,
            formal_selection_for=lambda _generation: None,
            analysis_for=lambda _generation: None,
            candidates=(),
            initial_revision_for=lambda _candidate_id: None,
            revision=lambda _revision_id: SimpleNamespace(
                genome_digest="g" * 64,
                behavior_digest="b" * 64,
            ),
            effective_revision_for=lambda generation: (
                "revision:champion" if generation in {0, 1} else None
            ),
            events=tuple(
                SimpleNamespace(
                    kind="GenerationChampionSelected",
                    payload={
                        "generation": generation,
                        "selected_candidate_id": "candidate:champion",
                        "selected_revision_id": "revision:champion",
                    },
                )
                for generation in (0, 1)
            ),
        )

        evidence = _evolution_evidence_projection(state)

        self.assertEqual(evidence["global_champion"]["generation"], 1)
        self.assertEqual(evidence["global_champion"]["absolute_score"], 0.3)

    def test_holdout_progress_uses_durable_batches_when_heartbeat_is_missing(self) -> None:
        revision = "revision:holdout"
        candidate_id = "candidate:holdout"
        cohort_digest = "a" * 64
        started = SimpleNamespace(
            seq=10,
            kind="HoldoutArmStarted",
                payload={
                    "candidate_id": candidate_id,
                    "revision": revision,
                    "checkpoint": {
                    "evaluation_phase": "holdout",
                    "formal_batch_index": None,
                    "holdout_arm": "finalist_1",
                    "candidate_revision_id": revision,
                    "cohort_digest": cohort_digest,
                },
            },
        )
        events = (
            started,
            SimpleNamespace(
                seq=11,
                kind="EvaluationSampleResultsStarted",
                payload={
                    "candidate_id": candidate_id,
                    "revision": revision,
                    "checkpoint": {
                        "evaluation_phase": "holdout",
                        "formal_batch_index": None,
                        "holdout_arm": "finalist_1",
                        "candidate_revision_id": revision,
                        "cohort_digest": cohort_digest,
                    },
                },
            ),
            SimpleNamespace(
                seq=12,
                kind="EvaluationSampleResultBatchRecorded",
                payload={
                    "candidate_id": candidate_id,
                    "revision": revision,
                    "record_count": 9,
                },
            ),
        )
        state = SimpleNamespace(
            events=events,
            task_manifest=SimpleNamespace(
                metadata={**dict(native_manifest().metadata), **{"prediction_cells_per_origin": 9}}
            ),
        )

        progress = _active_scoped_evaluation_progress(
            state,
            started=started,
            candidate_id=candidate_id,
            evaluation_phase="holdout",
            origin_total=2,
            holdout_arm="finalist_1",
            candidate_revision_id=revision,
            cohort_digest=cohort_digest,
        )

        self.assertIsNotNone(progress)
        self.assertEqual(progress["completed_samples"], 1)
        self.assertEqual(progress["total_samples"], 2)

    def test_adaptive_events_have_stable_public_timeline_types(self) -> None:
        self.assertEqual(
            {
                kind: _event_type(kind)
                for kind in (
                    "CandidateScreeningRecorded",
                    "FormalSelectionCohortFrozen",
                    "CandidateScreenedOut",
                    "FormalBatchStarted",
                    "FormalBatchEvaluated",
                    "FormalBatchCompared",
                    "HoldoutArmStarted",
                    "HoldoutRecheckStarted",
                    "GenerationComparisonRecorded",
                )
            },
            {
                "CandidateScreeningRecorded": "candidate.screening_recorded",
                "FormalSelectionCohortFrozen": "formal.selection_cohort_frozen",
                "CandidateScreenedOut": "candidate.screened_out",
                "FormalBatchStarted": "formal.batch_started",
                "FormalBatchEvaluated": "formal.batch_evaluated",
                "FormalBatchCompared": "formal.batch_compared",
                "HoldoutArmStarted": "holdout.arm_started",
                "HoldoutRecheckStarted": "holdout.recheck_started",
                "GenerationComparisonRecorded": "generation.comparison_recorded",
            },
        )

    def test_formal_batch_comparison_has_truthful_public_timeline_payload(self) -> None:
        from ecologyrsi_dsh.api.events import EventEndpointsMixin

        comparison = {
            "comparison_id": "comparison:candidate:a:3",
            "generation": 0,
            "candidate_id": "candidate:a",
            "batch_index": 3,
            "cohort_digest": "a" * 64,
            "champion_before_revision_id": "revision:a:champion",
            "challenger_revision_id": "revision:a:challenger",
            "champion_score": -0.4,
            "challenger_score": -0.5,
            "score_delta": -0.1,
            "minimum_score_delta": 0.005,
            "safety_gate_passed": True,
            "cell_regression_gate_passed": True,
            "decision": "champion_retained",
            "champion_after_revision_id": "revision:a:champion",
            "reason": "below_practical_delta",
            "created_at": "private-comparison-timestamp",
        }
        event = SimpleNamespace(
            seq=17,
            event_id="event:formal-batch-compared",
            run_id="run:projection",
            kind="FormalBatchCompared",
            payload={"comparison": comparison},
            created_at="2026-08-30T00:00:00+00:00",
        )

        public_event = EventEndpointsMixin._event_json(event)

        self.assertEqual(public_event["type"], "formal.batch_compared")
        self.assertEqual(
            public_event["payload"]["message"],
            "同 cohort 冠军—挑战者比较已完成。",
        )
        self.assertEqual(
            public_event["payload"]["decision"],
            "champion_retained",
        )
        self.assertEqual(
            public_event["payload"]["champion_after_revision_id"],
            "revision:a:champion",
        )
        self.assertEqual(
            public_event["payload"]["reason"],
            "below_practical_delta",
        )
        self.assertNotIn("created_at", public_event["payload"])


    def test_adaptive_progress_waits_for_evaluation_evidence(self) -> None:
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "optimization_protocol": "joint_plugin_epoch@2",
                "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "two_stage_evaluation_enabled": True,
                "sample_reflection_policy": "candidate_aggregate_post_score@1",
                "sample_concurrency": 64,
            }}),
            run=SimpleNamespace(generation=0, status=SimpleNamespace(value="running")),
            candidate_screening_events=(),
            formal_batch_evaluations=(),
            holdout_evaluations=(),
            formal_batches=(),
            candidates=(),
            events=(
                SimpleNamespace(
                    seq=1,
                    kind="EvolutionStageRecorded",
                    payload={"generation": 0, "stage": "search", "status": "started"},
                    created_at="2026-08-27T16:23:07+00:00",
                ),
            ),
        )

        self.assertIsNone(_adaptive_progress_projection(state))

    def test_screening_progress_counts_planner_as_terminal_for_aggregate_reflection(
        self,
    ) -> None:
        events = (
            SimpleNamespace(
                seq=10,
                kind="GenerationBatchStarted",
                payload={"batch": {"generation": 0}},
                created_at="2026-08-26T05:59:59+00:00",
            ),
            SimpleNamespace(
                seq=11,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:origin-a",
                    "reservation_id": "reservation-plan-a",
                    "sample_member_digests": _origin_members("origin-a"),
                }},
                created_at="2026-08-26T06:00:00+00:00",
            ),
            SimpleNamespace(
                seq=12,
                kind="DshStructuredResultAccepted",
                payload={"identity": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:origin-a",
                    "child_reservation_id": "reservation-plan-a",
                }},
                created_at="2026-08-26T06:00:01+00:00",
            ),
        )
        state = SimpleNamespace(
            run=SimpleNamespace(generation=0, status=SimpleNamespace(value="running")),
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "two_stage_evaluation_enabled": True,
                "sample_reflection_policy": "candidate_aggregate_post_score@1",
                "sample_concurrency": 64,
            }}),
            candidates=tuple(SimpleNamespace(generation=0) for _ in range(5)),
            events=events,
        )

        progress = _screening_progress_projection(state)

        self.assertEqual(progress["completed_samples"], 1)
        self.assertEqual(progress["succeeded_samples"], 1)
        self.assertEqual(progress["failed_samples"], 0)
        self.assertEqual(progress["in_flight_batches"], 0)
        self.assertEqual(progress["awaiting_submission_batches"], 49)

    def test_screening_progress_counts_complete_prediction_tool_for_aggregate_reflection(
        self,
    ) -> None:
        events = (
            SimpleNamespace(
                seq=10,
                kind="GenerationBatchStarted",
                payload={"batch": {"generation": 0}},
                created_at="2026-08-26T05:59:59+00:00",
            ),
            SimpleNamespace(
                seq=11,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:origin-a",
                    "reservation_id": "reservation-plan-a",
                    "sample_member_digests": _origin_members("origin-a"),
                }},
                created_at="2026-08-26T06:00:00+00:00",
            ),
            SimpleNamespace(
                seq=12,
                kind="DshPredictionToolExecuted",
                payload={
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:origin-a",
                    "prediction_count": 9,
                },
                created_at="2026-08-26T06:00:01+00:00",
            ),
        )
        state = SimpleNamespace(
            run=SimpleNamespace(generation=0, status=SimpleNamespace(value="running")),
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "two_stage_evaluation_enabled": True,
                "sample_reflection_policy": "candidate_aggregate_post_score@1",
                "sample_concurrency": 64,
            }}),
            candidates=tuple(SimpleNamespace(generation=0) for _ in range(5)),
            events=events,
        )

        progress = _screening_progress_projection(state)

        self.assertEqual(progress["completed_samples"], 0)
        self.assertEqual(progress["succeeded_samples"], 0)
        self.assertEqual(progress["prediction_tool_output_origins"], 1)
        self.assertEqual(progress["prediction_tool_pending_origins"], 1)
        self.assertEqual(progress["in_flight_batches"], 1)

    def test_adaptive_progress_includes_live_screening_tool_receipts(self) -> None:
        schedule = {
            "screening_origin_count": 64,
            "formal_origin_count_per_finalist": 500,
            "selection_holdout_origin_count": 169,
            "local_batch_origin_count": 50,
        }
        events = (
            SimpleNamespace(
                seq=10,
                kind="GenerationBatchStarted",
                payload={"batch": {"generation": 0}},
                created_at="2026-08-26T05:59:59+00:00",
            ),
            SimpleNamespace(
                seq=11,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:origin-a",
                    "reservation_id": "reservation-plan-a",
                    "sample_member_digests": _origin_members("origin-a"),
                }},
                created_at="2026-08-26T06:00:00+00:00",
            ),
            SimpleNamespace(
                seq=12,
                kind="DshPredictionToolExecuted",
                payload={
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:origin-a",
                    "prediction_count": 9,
                },
                created_at="2026-08-26T06:00:01+00:00",
            ),
        )
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "optimization_protocol": "joint_plugin_epoch@2",
                "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "two_stage_evaluation_enabled": True,
                "sample_reflection_policy": "candidate_aggregate_post_score@1",
                "sample_concurrency": 64,
            }}),
            run=SimpleNamespace(generation=0, status=SimpleNamespace(value="running")),
            candidate_screening_events=(),
            formal_batch_evaluations=(),
            holdout_evaluations=(),
            formal_batches=(),
            candidates=tuple(SimpleNamespace(generation=0) for _ in range(5)),
            events=events,
        )

        progress = _adaptive_progress_projection(state)

        self.assertEqual(progress["evaluation_phase"], "screening")
        self.assertEqual(progress["progress_kind"], "waiting")
        self.assertEqual(progress["screening_completed_origins"], 0)
        self.assertEqual(progress["completed_origins"], 0)
        self.assertEqual(progress["settled_origins"], 0)
        self.assertNotIn("succeeded_samples", progress)
        self.assertNotIn("failed_samples", progress)
        self.assertEqual(progress["in_flight_batches"], 1)
        self.assertEqual(progress["in_flight_requests"], 1)
        self.assertEqual(progress["awaiting_submission_batches"], 49)
        self.assertEqual(progress["awaiting_settlement_batches"], 0)
        self.assertEqual(progress["prediction_tool_output_origins"], 1)
        self.assertEqual(progress["prediction_tool_pending_origins"], 1)
        self.assertEqual(progress["gateway_request_count"], 1)
        self.assertEqual(progress["configured_concurrency"], 64)

    def test_adaptive_screening_rate_uses_durable_origin_result_batches(self) -> None:
        schedule = {
            "screening_origin_count": 64,
            "formal_origin_count_per_finalist": 500,
            "selection_holdout_origin_count": 169,
            "local_batch_origin_count": 50,
        }
        candidate_id = "candidate:screening-rate"
        revision = "revision:screening-rate"
        now = datetime.now(timezone.utc).replace(microsecond=0)
        events = (
            SimpleNamespace(
                seq=10,
                kind="GenerationBatchStarted",
                payload={"batch": {"generation": 0}},
                created_at=(now - timedelta(seconds=42)).isoformat(),
            ),
            SimpleNamespace(
                seq=11,
                kind="EvaluationSampleResultsStarted",
                payload={
                    "candidate_id": candidate_id,
                    "revision": revision,
                },
                created_at=(now - timedelta(seconds=41)).isoformat(),
            ),
            SimpleNamespace(
                seq=12,
                kind="EvaluationSampleResultBatchRecorded",
                payload={
                    "candidate_id": candidate_id,
                    "revision": revision,
                    "record_count": 9,
                },
                created_at=(now - timedelta(seconds=31)).isoformat(),
            ),
            SimpleNamespace(
                seq=13,
                kind="EvaluationProgressRecorded",
                payload={
                    "schema_version": "ecologyrsi-dsh.evaluation-progress/3",
                    "revision": revision,
                    "progress_id": 1,
                    "progress_kind": "completed_batch",
                    "candidate_id": candidate_id,
                    "role": "planner",
                    "model_id": "model:test",
                    "batch_index": 1,
                    "batch_count": 64,
                    "batch_size": 1,
                    "completed_samples": 1,
                    "total_samples": 64,
                    "succeeded_samples": 1,
                    "failed_samples": 0,
                    "in_flight_batches": 0,
                    "queued_batches": 0,
                },
                created_at=(now - timedelta(seconds=30)).isoformat(),
            ),
            SimpleNamespace(
                seq=14,
                kind="EvaluationSampleResultBatchRecorded",
                payload={
                    "candidate_id": candidate_id,
                    "revision": revision,
                    "record_count": 9,
                },
                created_at=(now - timedelta(seconds=1)).isoformat(),
            ),
            SimpleNamespace(
                seq=15,
                kind="EvaluationProgressRecorded",
                payload={
                    "schema_version": "ecologyrsi-dsh.evaluation-progress/3",
                    "revision": revision,
                    "progress_id": 2,
                    "progress_kind": "completed_batch",
                    "candidate_id": candidate_id,
                    "role": "planner",
                    "model_id": "model:test",
                    "batch_index": 2,
                    "batch_count": 64,
                    "batch_size": 1,
                    "completed_samples": 2,
                    "total_samples": 64,
                    "succeeded_samples": 2,
                    "failed_samples": 0,
                    "in_flight_batches": 0,
                    "queued_batches": 0,
                },
                created_at=now.isoformat(),
            ),
        )
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "optimization_protocol": "joint_plugin_epoch@2",
                "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "two_stage_evaluation_enabled": True,
                "sample_reflection_policy": "candidate_aggregate_post_score@1",
                "sample_concurrency": 64,
                "prediction_cells_per_origin": 9,
            }}),
            run=SimpleNamespace(
                generation=0,
                status=SimpleNamespace(value="running"),
            ),
            candidate_screening_events=(),
            formal_batch_evaluations=(),
            holdout_evaluations=(),
            formal_batches=(),
            candidates=(
                SimpleNamespace(candidate_id=candidate_id, generation=0),
                SimpleNamespace(candidate_id="candidate:1", generation=0),
                SimpleNamespace(candidate_id="candidate:2", generation=0),
                SimpleNamespace(candidate_id="candidate:3", generation=0),
            ),
            events=events,
        )

        progress = _adaptive_progress_projection(state)

        self.assertEqual(progress["evaluation_phase"], "screening")
        self.assertEqual(progress["completed_origins"], 2)
        self.assertGreater(progress["samples_per_minute"], 0)
        self.assertGreater(progress["estimated_remaining_seconds"], 0)
        self.assertEqual(
            progress["throughput_semantics"],
            "host_settled_origins_rolling_32_durable_result_boundaries",
        )

    def test_adaptive_rate_resets_after_restart_resume_boundary(self) -> None:
        schedule = {
            "screening_origin_count": 64,
            "formal_origin_count_per_finalist": 500,
            "selection_holdout_origin_count": 169,
            "local_batch_origin_count": 50,
        }
        candidate_id = "candidate:restart-rate"
        now = datetime.now(timezone.utc).replace(microsecond=0)

        def event(seq: int, kind: str, at: datetime, payload: dict) -> SimpleNamespace:
            return SimpleNamespace(
                seq=seq,
                kind=kind,
                payload=payload,
                created_at=at.isoformat(),
            )

        events = (
            event(
                10,
                "GenerationBatchStarted",
                now - timedelta(hours=12, seconds=20),
                {"generation": 0, "batch": {"generation": 0}},
            ),
            event(
                11,
                "EvaluationSampleResultBatchRecorded",
                now - timedelta(hours=12, seconds=10),
                {"generation": 0, "candidate_id": candidate_id, "record_count": 9},
            ),
            event(
                12,
                "EvaluationSampleResultBatchRecorded",
                now - timedelta(hours=12),
                {"generation": 0, "candidate_id": candidate_id, "record_count": 9},
            ),
            event(
                13,
                "EvaluationSampleResultsResumed",
                now - timedelta(seconds=20),
                {"generation": 0, "candidate_id": candidate_id, "record_count": 0},
            ),
            event(
                14,
                "EvaluationSampleResultBatchRecorded",
                now - timedelta(seconds=10),
                {"generation": 0, "candidate_id": candidate_id, "record_count": 9},
            ),
            event(
                15,
                "EvaluationSampleResultBatchRecorded",
                now,
                {"generation": 0, "candidate_id": candidate_id, "record_count": 9},
            ),
        )
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "optimization_protocol": "joint_plugin_epoch@2",
                "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "two_stage_evaluation_enabled": True,
                "sample_reflection_policy": "candidate_aggregate_post_score@1",
                "sample_concurrency": 64,
                "prediction_cells_per_origin": 9,
            }}, max_generations=5),
            run=SimpleNamespace(
                generation=0,
                status=SimpleNamespace(value="running"),
            ),
            formal_batch_evaluations=(),
            holdout_evaluations=(),
            formal_batches=(),
            candidate_screening_events=(
                SimpleNamespace(
                    payload={
                        "generation": 0,
                        "candidate_id": candidate_id,
                        "origin_count": 0,
                    },
                    created_at=now.isoformat(),
                ),
            ),
            candidates=tuple(
                SimpleNamespace(candidate_id=item, generation=0)
                for item in (candidate_id, "candidate:1", "candidate:2", "candidate:3")
            ),
            events=events,
        )

        progress = _adaptive_progress_projection(state)

        self.assertIsNotNone(progress)
        self.assertGreater(progress["samples_per_minute"], 1.0)
        self.assertLess(progress["estimated_remaining_seconds"], 100_000)
        self.assertGreater(
            progress["run_estimated_remaining_seconds"],
            progress["estimated_remaining_seconds"],
        )

    def test_screening_progress_separates_sparse_repair_from_primary_origin(
        self,
    ) -> None:
        schedule = {
            "screening_origin_count": 64,
            "formal_origin_count_per_finalist": 500,
            "selection_holdout_origin_count": 169,
            "local_batch_origin_count": 50,
        }
        # Six cells proves the projection follows task metadata rather than
        # assuming the historical 3-target x 3-horizon bundle.
        primary_members = _origin_members("primary-a")[:6]
        repair_members = primary_members[:1]
        events = (
            SimpleNamespace(
                seq=10,
                kind="GenerationBatchStarted",
                payload={"batch": {"generation": 0}},
                created_at="2026-08-28T06:00:00+00:00",
            ),
            SimpleNamespace(
                seq=11,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:primary-a",
                    "reservation_id": "reservation-primary-a",
                    "sample_member_digests": primary_members,
                }},
                created_at="2026-08-28T06:00:01+00:00",
            ),
            SimpleNamespace(
                seq=12,
                kind="DshStructuredResultAccepted",
                payload={"identity": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:primary-a",
                    "child_reservation_id": "reservation-primary-a",
                }},
                created_at="2026-08-28T06:00:02+00:00",
            ),
            SimpleNamespace(
                seq=13,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:repair-a",
                    "reservation_id": "reservation-repair-a",
                    "sample_member_digests": repair_members,
                }},
                created_at="2026-08-28T06:00:03+00:00",
            ),
            SimpleNamespace(
                seq=14,
                kind="DshStructuredResultAccepted",
                payload={"identity": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:repair-a",
                    "child_reservation_id": "reservation-repair-a",
                }},
                created_at="2026-08-28T06:00:04+00:00",
            ),
            SimpleNamespace(
                seq=15,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:repair-active",
                    "reservation_id": "reservation-repair-active",
                    "sample_member_digests": primary_members[1:3],
                }},
                created_at="2026-08-28T06:00:05+00:00",
            ),
        )
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "optimization_protocol": "joint_plugin_epoch@2",
                "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "two_stage_evaluation_enabled": True,
                "sample_reflection_policy": "candidate_aggregate_post_score@1",
                "sample_concurrency": 64,
                "prediction_cells_per_origin": 6,
            }}),
            run=SimpleNamespace(generation=0, status=SimpleNamespace(value="running")),
            candidate_screening_events=(),
            formal_batch_evaluations=(),
            holdout_evaluations=(),
            formal_batches=(),
            candidates=tuple(SimpleNamespace(generation=0) for _ in range(5)),
            events=events,
        )

        screening = _screening_progress_projection(state)

        self.assertEqual(screening["completed_samples"], 1)
        self.assertEqual(screening["remote_completed_origins"], 1)
        self.assertEqual(screening["completed_repair_waves"], 1)
        self.assertEqual(screening["active_repair_waves"], 1)
        self.assertEqual(screening["primary_gateway_request_count"], 1)
        self.assertEqual(screening["repair_gateway_request_count"], 2)
        self.assertEqual(screening["gateway_request_count"], 3)
        self.assertEqual(screening["in_flight_batches"], 1)
        self.assertEqual(screening["awaiting_submission_batches"], 49)

        adaptive = _adaptive_progress_projection(state)

        self.assertEqual(adaptive["completed_origins"], 0)
        self.assertEqual(adaptive["remote_completed_origins"], 1)
        self.assertEqual(adaptive["awaiting_settlement_batches"], 1)
        self.assertEqual(adaptive["completed_repair_waves"], 1)
        self.assertEqual(adaptive["active_repair_waves"], 1)
        self.assertEqual(adaptive["in_flight_requests"], 1)

    def test_adaptive_screening_counts_host_settled_failed_origins(self) -> None:
        schedule = {
            "screening_origin_count": 64,
            "formal_origin_count_per_finalist": 500,
            "selection_holdout_origin_count": 169,
            "local_batch_origin_count": 50,
        }
        heartbeat = SimpleNamespace(
            seq=20,
            kind="EvaluationProgressRecorded",
            payload={
                "schema_version": "ecologyrsi-dsh.evaluation-progress/3",
                "revision": "revision:screening-a",
                "progress_id": 4,
                "progress_kind": "completed_batch",
                "candidate_id": "candidate:screening-a",
                "role": "planner",
                "model_id": "model:test",
                "batch_index": 4,
                "batch_count": 10,
                "batch_size": 1,
                "completed_samples": 4,
                "total_samples": 10,
                "succeeded_samples": 0,
                "failed_samples": 4,
                "gateway_request_count": 4,
                "in_flight_batches": 0,
                "queued_batches": 0,
            },
            created_at="2026-08-28T02:00:00+00:00",
        )
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "optimization_protocol": "joint_plugin_epoch@2",
                "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
                "sample_concurrency": 64,
            }}),
            run=SimpleNamespace(
                generation=0,
                status=SimpleNamespace(value="running"),
            ),
            candidate_screening_events=(),
            formal_batch_evaluations=(),
            holdout_evaluations=(),
            formal_batches=(),
            candidates=(SimpleNamespace(
                candidate_id="candidate:screening-a",
                generation=0,
            ),),
            events=(heartbeat,),
        )

        progress = _adaptive_progress_projection(state)

        self.assertEqual(progress["evaluation_phase"], "screening")
        self.assertEqual(progress["completed_samples"], 4)
        self.assertEqual(progress["settled_origins"], 4)
        self.assertEqual(progress["succeeded_samples"], 0)
        self.assertEqual(progress["failed_samples"], 4)
        self.assertTrue(progress["outcomes_verified"])
        self.assertEqual(progress["awaiting_submission_batches"], 46)
        self.assertEqual(progress["updated_at"], heartbeat.created_at)

    def test_adaptive_progress_uses_frozen_phase_boundary_after_early_failure(
        self,
    ) -> None:
        schedule = {
            "screening_origin_count": 64,
            "formal_origin_count_per_finalist": 500,
            "selection_holdout_origin_count": 169,
            "local_batch_origin_count": 50,
        }
        screening_events = tuple(
            SimpleNamespace(
                payload={
                    "generation": 0,
                    "candidate_id": f"candidate:{index}",
                    "origin_count": 10,
                },
                created_at=f"2026-08-28T02:0{index}:00+00:00",
            )
            for index in range(3)
        )
        events = (
            SimpleNamespace(
                seq=1,
                kind="GenerationBatchStarted",
                payload={"batch": {"generation": 0}},
                created_at="2026-08-28T02:00:00+00:00",
            ),
            SimpleNamespace(
                seq=2,
                kind="FormalSelectionCohortFrozen",
                payload={"generation": 0},
                created_at="2026-08-28T02:04:00+00:00",
            ),
        )
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(
                metadata={**dict(native_manifest().metadata), **{
                    "optimization_protocol": "joint_plugin_epoch@2",
                    "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
                    "sample_concurrency": 64,
                }}
            ),
            run=SimpleNamespace(
                generation=0,
                status=SimpleNamespace(value="running"),
            ),
            candidate_screening_events=screening_events,
            formal_batch_evaluations=(),
            holdout_evaluations=(),
            formal_batches=(),
            candidates=tuple(
                SimpleNamespace(
                    candidate_id=f"candidate:{index}",
                    generation=0,
                )
                for index in range(5)
            ),
            events=events,
        )

        progress = _adaptive_progress_projection(state)

        self.assertEqual(progress["evaluation_phase"], "formal_batch")
        self.assertEqual(progress["screening_completed_origins"], 30)
        self.assertEqual(progress["screening_skipped_origins"], 20)
        self.assertEqual(progress["terminal_skipped_origins"], 20)
        self.assertEqual(progress["completed_origins"], 50)
        self.assertNotIn("awaiting_submission_batches", progress)

    def test_adaptive_progress_includes_live_formal_origin_receipts(self) -> None:
        schedule = {
            "screening_origin_count": 64,
            "formal_origin_count_per_finalist": 500,
            "selection_holdout_origin_count": 169,
            "local_batch_origin_count": 50,
        }
        members = [f"{index:064x}" for index in range(1, 10)]
        events = (
            SimpleNamespace(
                seq=10,
                kind="FormalBatchStarted",
                payload={"batch": {
                    "generation": 0,
                    "candidate_id": "candidate:formal",
                    "batch_index": 1,
                    "batch_count": 2,
                    "origin_count": 25,
                    "cohort_digest": "a" * 64,
                }},
                created_at="2026-08-28T00:00:00+00:00",
            ),
            SimpleNamespace(
                seq=11,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:formal-origin-a",
                    "reservation_id": "reservation-formal-a",
                    "sample_member_digests": members,
                }},
                created_at="2026-08-28T00:00:01+00:00",
            ),
            SimpleNamespace(
                seq=12,
                kind="DshPredictionToolExecuted",
                payload={
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:formal-origin-a",
                    "prediction_count": 9,
                },
                created_at="2026-08-28T00:00:02+00:00",
            ),
        )
        screening_events = tuple(
            SimpleNamespace(
                payload={"generation": 0, "origin_count": 10},
                created_at=f"2026-08-28T00:0{index}:00+00:00",
            )
            for index in range(5)
        )
        prior_formal = SimpleNamespace(
            scope=SimpleNamespace(
                generation=0,
                candidate_id="candidate:formal",
                batch_index=1,
                cohort_digest="b" * 64,
                origin_count=25,
            ),
            created_at="2026-08-28T00:04:00+00:00",
        )
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "optimization_protocol": "joint_plugin_epoch@2",
                "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "two_stage_evaluation_enabled": True,
                "sample_reflection_policy": "candidate_aggregate_post_score@1",
                "sample_concurrency": 64,
                "prediction_cells_per_origin": 9,
            }}),
            run=SimpleNamespace(generation=0, status=SimpleNamespace(value="running")),
            candidate_screening_events=screening_events,
            formal_batch_evaluations=(prior_formal,),
            holdout_evaluations=(),
            formal_batches=(SimpleNamespace(
                generation=0,
                candidate_id="candidate:formal",
                batch_index=1,
            ),),
            candidates=(),
            events=events,
        )

        progress = _adaptive_progress_projection(
            state,
            {
                "limit": 64,
                "adaptive_limit": 16,
                "active": 16,
                "waiting": 7,
                "congestion_events": 3,
            },
        )

        self.assertEqual(progress["evaluation_phase"], "formal_batch")
        self.assertEqual(progress["completed_origins"], 75)
        self.assertEqual(progress["settled_origins"], 75)
        self.assertNotIn("succeeded_samples", progress)
        self.assertNotIn("failed_samples", progress)
        self.assertEqual(progress["in_flight_batches"], 1)
        self.assertEqual(progress["in_flight_requests"], 1)
        self.assertEqual(progress["awaiting_submission_batches"], 24)
        self.assertEqual(progress["awaiting_settlement_batches"], 0)
        self.assertEqual(progress["prediction_tool_pending_origins"], 1)
        self.assertEqual(
            progress["completed_origins"],
            progress["settled_origins"]
            + progress["awaiting_settlement_batches"],
        )
        self.assertEqual(progress["gateway_request_count"], 1)
        self.assertEqual(progress["updated_at"], events[-1].created_at)
        self.assertGreater(progress["samples_per_minute"], 0)
        self.assertGreater(progress["estimated_remaining_seconds"], 0)
        self.assertEqual(progress["adaptive_admission_limit"], 16)
        self.assertEqual(progress["admission_active"], 16)
        self.assertEqual(progress["admission_waiting"], 7)
        self.assertEqual(progress["admission_congestion_events"], 3)

    def test_adaptive_progress_includes_live_holdout_origin_receipts(self) -> None:
        schedule = {
            "screening_origin_count": 64,
            "formal_origin_count_per_finalist": 500,
            "selection_holdout_origin_count": 169,
            "local_batch_origin_count": 50,
        }
        members = [f"{index:064x}" for index in range(1, 10)]
        events = (
            SimpleNamespace(
                seq=100,
                kind="HoldoutArmStarted",
                payload={
                    "generation": 0,
                    "holdout_arm": "finalist_1",
                    "candidate_id": "candidate:finalist",
                    "origin_count": 50,
                },
                created_at="2026-08-28T01:00:00+00:00",
            ),
            SimpleNamespace(
                seq=101,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:holdout-origin-a",
                    "reservation_id": "reservation-holdout-a",
                    "sample_member_digests": members,
                }},
                created_at="2026-08-28T01:00:01+00:00",
            ),
            SimpleNamespace(
                seq=102,
                kind="DshPredictionToolExecuted",
                payload={
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:holdout-origin-a",
                    "prediction_count": 9,
                },
                created_at="2026-08-28T01:00:02+00:00",
            ),
        )
        screening_events = tuple(
            SimpleNamespace(
                payload={"generation": 0, "origin_count": 10},
                created_at=f"2026-08-28T00:0{index}:00+00:00",
            )
            for index in range(5)
        )
        formal_evaluations = tuple(
            SimpleNamespace(
                scope=SimpleNamespace(
                    generation=0,
                    candidate_id=f"candidate:{index}",
                    batch_index=1,
                    cohort_digest=f"{index + 1:064x}",
                    origin_count=25,
                ),
                created_at=f"2026-08-28T00:{10 + index:02d}:00+00:00",
            )
            for index in range(2)
        )
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "optimization_protocol": "joint_plugin_epoch@2",
                "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
                "sample_concurrency": 64,
                "prediction_cells_per_origin": 9,
            }}),
            run=SimpleNamespace(generation=0, status=SimpleNamespace(value="running")),
            candidate_screening_events=screening_events,
            formal_batch_evaluations=formal_evaluations,
            holdout_evaluations=(),
            formal_batches=(SimpleNamespace(
                generation=0,
                candidate_id="candidate:1",
                batch_index=9,
            ),),
            candidates=(),
            events=events,
        )

        progress = _adaptive_progress_projection(state)

        self.assertEqual(progress["evaluation_phase"], "holdout")
        self.assertEqual(progress["completed_origins"], 100)
        self.assertEqual(progress["holdout_completed_origins"], 0)
        self.assertEqual(progress["settled_origins"], 100)
        self.assertEqual(progress["awaiting_settlement_batches"], 0)
        self.assertEqual(progress["prediction_tool_pending_origins"], 1)
        self.assertEqual(
            progress["completed_origins"],
            progress["settled_origins"]
            + progress["awaiting_settlement_batches"],
        )
        self.assertEqual(progress["holdout_arm"], "finalist_1")
        self.assertEqual(progress["current_candidate_id"], "candidate:finalist")
        self.assertIsNone(progress["batch_index"])
        self.assertIsNone(progress["batch_count"])
        self.assertEqual(progress["awaiting_submission_batches"], 49)
        self.assertEqual(progress["in_flight_requests"], 1)

    def test_adaptive_active_scopes_use_host_settlement_heartbeat(self) -> None:
        schedule = {
            "screening_origin_count": 64,
            "formal_origin_count_per_finalist": 500,
            "selection_holdout_origin_count": 169,
            "local_batch_origin_count": 50,
        }
        screening_events = tuple(
            SimpleNamespace(
                payload={"generation": 0, "origin_count": 10},
                created_at=f"2026-08-28T00:0{index}:00+00:00",
            )
            for index in range(5)
        )
        for phase, origin_total, succeeded, failed in (
            ("formal_batch", 25, 0, 1),
            ("holdout", 50, 1, 0),
        ):
            with self.subTest(phase=phase):
                candidate_id = f"candidate:{phase}"
                revision_id = f"revision:{phase}"
                cohort_digest = digest(f"cohort:{phase}")
                if phase == "formal_batch":
                    boundary = SimpleNamespace(
                        seq=10,
                        kind="FormalBatchStarted",
                        payload={"batch": {
                            "generation": 0,
                            "candidate_id": candidate_id,
                            "revision_id": revision_id,
                            "batch_index": 1,
                            "batch_count": 2,
                            "origin_count": origin_total,
                            "cohort_digest": cohort_digest,
                        }},
                        created_at="2026-08-28T01:00:00+00:00",
                    )
                    formal_batch_index = 1
                    holdout_arm = None
                    formal_count = 1
                    expected_completed = 76
                    expected_scope = "active_formal_batch"
                    expected_awaiting_submission = 23
                else:
                    boundary = SimpleNamespace(
                        seq=10,
                        kind="HoldoutArmStarted",
                        payload={
                            "generation": 0,
                            "candidate_id": candidate_id,
                            "candidate_revision_id": revision_id,
                            "holdout_arm": "finalist_1",
                            "origin_count": origin_total,
                            "cohort_digest": cohort_digest,
                        },
                        created_at="2026-08-28T01:00:00+00:00",
                    )
                    formal_batch_index = None
                    holdout_arm = "finalist_1"
                    formal_count = 2
                    expected_completed = 101
                    expected_scope = "active_holdout_arm"
                    expected_awaiting_submission = 48
                revision_start = SimpleNamespace(
                    seq=11,
                    kind="EvaluationSampleResultsStarted",
                    payload={
                        "candidate_id": candidate_id,
                        "revision": f"sample-revision:{phase}",
                        "checkpoint": {
                            "evaluation_phase": phase,
                            "formal_batch_index": formal_batch_index,
                            "holdout_arm": holdout_arm,
                            "candidate_revision_id": revision_id,
                            "cohort_digest": cohort_digest,
                        },
                    },
                    created_at="2026-08-28T01:00:01+00:00",
                )
                child_events = []
                for offset, label in enumerate(("a", "b")):
                    key = f"run:test:sample.plan:{phase}:{label}"
                    reservation = f"reservation:{phase}:{label}"
                    child_events.extend((
                        SimpleNamespace(
                            seq=12 + offset * 2,
                            kind="DshChildLaunchReserved",
                            payload={"launch": {
                                "stage": "sample.plan",
                                "idempotency_key": key,
                                "reservation_id": reservation,
                                "sample_member_digests": _origin_members(
                                    f"{phase}:{label}"
                                ),
                            }},
                            created_at=f"2026-08-28T01:00:0{2 + offset * 2}+00:00",
                        ),
                        SimpleNamespace(
                            seq=13 + offset * 2,
                            kind="DshStructuredResultAccepted",
                            payload={"identity": {
                                "stage": "sample.plan",
                                "idempotency_key": key,
                                "child_reservation_id": reservation,
                            }},
                            created_at=f"2026-08-28T01:00:0{3 + offset * 2}+00:00",
                        ),
                    ))
                heartbeat = SimpleNamespace(
                    seq=16,
                    kind="EvaluationProgressRecorded",
                    payload={
                        "schema_version": "ecologyrsi-dsh.evaluation-progress/3",
                        "revision": f"sample-revision:{phase}",
                        "progress_id": 1,
                        "progress_kind": "completed_batch",
                        "candidate_id": candidate_id,
                        "role": "planner",
                        "model_id": "model:test",
                        "batch_index": 1,
                        "batch_count": origin_total,
                        "batch_size": 1,
                        "completed_samples": 1,
                        "total_samples": origin_total,
                        "succeeded_samples": succeeded,
                        "failed_samples": failed,
                        "gateway_request_count": 2,
                        "in_flight_batches": 0,
                        "queued_batches": 0,
                    },
                    created_at="2026-08-28T01:00:06+00:00",
                )
                formal_evaluations = tuple(
                    SimpleNamespace(
                        scope=SimpleNamespace(
                            generation=0,
                            candidate_id=f"candidate:{index}",
                            batch_index=1,
                            cohort_digest=f"{index + 1:064x}",
                            origin_count=25,
                        ),
                        created_at=f"2026-08-28T00:{10 + index:02d}:00+00:00",
                    )
                    for index in range(formal_count)
                )
                state = SimpleNamespace(
                    task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                        "optimization_protocol": "joint_plugin_epoch@2",
                        "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
                        "sample_concurrency": 64,
                        "prediction_cells_per_origin": 9,
                    }}),
                    run=SimpleNamespace(
                        generation=0,
                        status=SimpleNamespace(value="running"),
                    ),
                    candidate_screening_events=screening_events,
                    formal_batch_evaluations=formal_evaluations,
                    holdout_evaluations=(),
                    formal_batches=(SimpleNamespace(
                        generation=0,
                        candidate_id=candidate_id,
                        batch_index=formal_batch_index or 9,
                    ),),
                    events=(boundary, revision_start, *child_events, heartbeat),
                )

                progress = _adaptive_progress_projection(state)

                self.assertEqual(progress["completed_origins"], expected_completed)
                self.assertEqual(progress["settled_origins"], expected_completed)
                self.assertEqual(progress["remote_completed_origins"], 2)
                self.assertEqual(progress["awaiting_settlement_batches"], 1)
                self.assertEqual(
                    progress["awaiting_submission_batches"],
                    expected_awaiting_submission,
                )
                self.assertEqual(progress["succeeded_samples"], succeeded)
                self.assertEqual(progress["failed_samples"], failed)
                self.assertTrue(progress["outcomes_verified"])
                self.assertEqual(progress["outcome_scope"], expected_scope)
                self.assertEqual(progress["updated_at"], heartbeat.created_at)

    def test_dsh_activity_uses_unresolved_formal_child_without_stage_event(self) -> None:
        launch = SimpleNamespace(
            seq=20,
            kind="DshChildLaunchReserved",
            payload={"launch": {
                "stage": "candidate.local_edit",
                "role": "candidate-proposer",
                "launch_attempt": 1,
                "reservation_id": "reservation-local-edit",
            }},
            created_at="2026-08-28T00:00:00+00:00",
        )

        activity = _dsh_activity_projection(
            SimpleNamespace(events=(launch,)),
            current_stage="evaluation",
            run_status="running",
        )

        self.assertEqual(activity["state"], "model_running")
        self.assertEqual(activity["dsh_stage"], "candidate.local_edit")
        self.assertEqual(activity["role"], "candidate-proposer")

    def test_dsh_activity_counts_parallel_local_edit_and_unresolved_samples(self) -> None:
        def launch(seq: int, stage: str) -> SimpleNamespace:
            return SimpleNamespace(
                seq=seq,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": stage,
                    "role": "candidate-proposer" if stage == "candidate.local_edit" else "sample-planner",
                    "launch_attempt": 1,
                    "reservation_id": f"private-reservation:{seq}",
                    "context": {"private": "never project model context"},
                }},
                created_at=f"2026-08-28T00:00:{seq:02d}+00:00",
            )

        events = [
            launch(20, "candidate.local_edit"),
            launch(21, "sample.plan"),
            launch(22, "sample.plan"),
            launch(23, "sample.plan"),
            launch(24, "sample.repair"),
            SimpleNamespace(seq=25, kind="DshStructuredResultAccepted",
                payload={"identity": {"child_reservation_id": "private-reservation:21"}},
                created_at="2026-08-28T00:00:25+00:00"),
            SimpleNamespace(seq=26, kind="DshChildExecutionFailed",
                payload={"identity": {"child_reservation_id": "private-reservation:24"}},
                created_at="2026-08-28T00:00:26+00:00"),
        ]
        activity = _dsh_activity_projection(
            SimpleNamespace(events=events), current_stage="evaluation", run_status="running"
        )
        self.assertEqual(activity["dsh_stage"], "sample.plan")
        self.assertEqual(activity["unresolved_child_count"], 3)
        self.assertEqual(activity["unresolved_stage_counts"], [
            {"dsh_stage": "candidate.local_edit", "count": 1},
            {"dsh_stage": "sample.plan", "count": 2},
        ])
        self.assertNotIn("private-reservation", str(activity))
        self.assertNotIn("model context", str(activity))
        self.assertIsNone(_dsh_activity_projection(
            SimpleNamespace(events=events), current_stage="evaluation", run_status="paused"
        ))


    def test_trajectory_operations_keep_non_scalar_subjects_and_name_the_axis(
        self,
    ) -> None:
        # The row used to keep a fixed op/name/value/target/program_id
        # whitelist, which is the shape of a scalar parameter move and of
        # nothing else. An instruction or workflow edit therefore reached the
        # process UI as a bare `{"op": ...}` and rendered untargeted.
        evaluation = SimpleNamespace(
            score=-0.2,
            passed=True,
            created_at="2026-08-28T00:00:02+00:00",
            metrics={"sample_execution_coverage": 1.0},
        )
        batch = SimpleNamespace(
            generation=0,
            candidate_id="candidate:axes",
            batch_index=0,
            batch_count=4,
            origin_count=50,
            revision_id="revision:axes:0",
            cohort_digest="a" * 64,
            created_at="2026-08-28T00:00:00+00:00",
        )
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{}}),
            local_edit_proposals=(
                {
                    "candidate_id": "candidate:axes",
                    "batch_index": 0,
                    "decision": "mutate",
                    "operations": [
                        {
                            "op": "select_instruction_template",
                            "role": "sample-planner",
                            "instruction_template_id": (
                                "sample-planner-anomaly-aware@1"
                            ),
                        },
                        {
                            "op": "set_bounded_workflow_parameter",
                            "name": "max_attempts",
                            "value": 3,
                        },
                    ],
                },
            ),
            local_edit_outcomes=(),
            trajectory_revision_activations=(),
            formal_batches=(batch,),
            formal_trajectories=(
                SimpleNamespace(
                    generation=0,
                    candidate_id="candidate:axes",
                    status=SimpleNamespace(value="running"),
                    initial_revision_id="revision:axes:0",
                    final_revision_id=None,
                    batch_count=4,
                ),
            ),
            batch_evaluation_for=lambda candidate_id, batch_index: evaluation,
        )

        operations = _adaptive_trajectory_projection(state)[0]["batches"][0][
            "operations"
        ]

        self.assertEqual(
            operations[0],
            {
                "op": "select_instruction_template",
                "role": "sample-planner",
                "instruction_template_id": "sample-planner-anomaly-aware@1",
                "axis": "instruction_profile",
                # Joined in from the operator's own target resolver, so the UI
                # never has to know the per-operation field names.
                "target": "sample-planner-anomaly-aware@1",
            },
        )
        self.assertEqual(
            operations[1],
            {
                "op": "set_bounded_workflow_parameter",
                "name": "max_attempts",
                "value": 3,
                "axis": "workflow_parameter",
                "target": "max_attempts",
            },
        )

    def test_paired_trajectory_projects_durable_comparison_and_next_challenger(self) -> None:
        champion = SimpleNamespace(
            score=-0.6,
            passed=True,
            created_at="2026-08-30T00:00:01+00:00",
            metrics={"sample_execution_coverage": 1.0},
        )
        challenger = SimpleNamespace(
            score=-0.4,
            passed=True,
            created_at="2026-08-30T00:00:02+00:00",
            metrics={
                "sample_execution_coverage": 1.0,
                "sample_execution": {
                    "attempted_origin_samples": 50,
                    "succeeded_origin_samples": 50,
                    "failed_origin_samples": 0,
                    "failed_examples": 0,
                    "scoring_fallback_examples": 0,
                },
            },
        )
        batch = SimpleNamespace(
            generation=0,
            candidate_id="candidate:paired",
            batch_index=1,
            batch_count=10,
            origin_count=50,
            revision_id="revision:paired:challenger-1",
            cohort_digest="b" * 64,
            created_at="2026-08-30T00:00:00+00:00",
        )
        comparison = SimpleNamespace(
            decision=SimpleNamespace(value="challenger_promoted"),
            reason="challenger_improved",
            champion_before_revision_id="revision:paired:champion-0",
            challenger_revision_id="revision:paired:challenger-1",
            champion_score=-0.6,
            challenger_score=-0.4,
            score_delta=0.2,
            minimum_score_delta=0.005,
            safety_gate_passed=True,
            cell_regression_gate_passed=True,
            champion_after_revision_id="revision:paired:challenger-1",
            created_at="2026-08-30T00:00:03+00:00",
        )
        activation = SimpleNamespace(
            candidate_id="candidate:paired",
            batch_index=1,
            from_revision_id="revision:paired:challenger-1",
            to_revision_id="revision:paired:challenger-2",
            reason=SimpleNamespace(value="local_edit_applied"),
            created_at="2026-08-30T00:00:04+00:00",
        )

        def evaluation_for(
            _candidate_id: str,
            _batch_index: int,
            arm: FormalBatchArm | None = None,
        ) -> SimpleNamespace:
            return champion if arm is FormalBatchArm.CHAMPION else challenger

        state = SimpleNamespace(
            task_manifest=SimpleNamespace(
                metadata={**dict(native_manifest().metadata), **{
                    "optimization_schedule": OptimizationSchedule.for_new_run().to_dict()
                }}
            ),
            local_edit_proposals=(
                {
                    "candidate_id": "candidate:paired",
                    "batch_index": 1,
                    "proposal": {
                        "decision": "mutate",
                        "operations": [
                            {
                                "op": "set_bounded_parameter",
                                "name": "ridge_alpha",
                                "value": 0.2,
                            }
                        ],
                    },
                },
            ),
            local_edit_outcomes=(
                {
                    "candidate_id": "candidate:paired",
                    "batch_index": 1,
                    "outcome": "applied",
                    "active_revision_id": "revision:paired:challenger-2",
                },
            ),
            trajectory_revision_activations=(activation,),
            formal_batches=(batch,),
            formal_trajectories=(
                SimpleNamespace(
                    generation=0,
                    candidate_id="candidate:paired",
                    status=SimpleNamespace(value="running"),
                    initial_revision_id="revision:paired:champion-0",
                    final_revision_id=None,
                    batch_count=10,
                ),
            ),
            batch_evaluation_for=evaluation_for,
            batch_comparison_for=lambda candidate_id, batch_index: (
                comparison
                if candidate_id == "candidate:paired" and batch_index == 1
                else None
            ),
        )

        lanes = _adaptive_trajectory_projection(state)

        self.assertEqual(lanes[0]["completed_batch_count"], 1)
        self.assertEqual(
            lanes[0]["strategy_label"],
            "同 cohort 冠军—挑战者配对策略",
        )
        row = lanes[0]["batches"][0]
        self.assertEqual(
            row["champion_before_revision_id"],
            "revision:paired:champion-0",
        )
        self.assertEqual(
            row["challenger_revision_id"],
            "revision:paired:challenger-1",
        )
        self.assertEqual(row["champion_score"], -0.6)
        self.assertEqual(row["challenger_score"], -0.4)
        self.assertEqual(row["score_delta"], 0.2)
        self.assertEqual(row["minimum_score_delta"], 0.005)
        self.assertEqual(row["comparison_decision"], "challenger_promoted")
        self.assertEqual(row["comparison_reason"], "challenger_improved")
        self.assertEqual(
            row["champion_after_revision_id"],
            "revision:paired:challenger-1",
        )
        self.assertEqual(
            row["next_challenger_revision_id"],
            "revision:paired:challenger-2",
        )
        self.assertEqual(
            row["next_challenger_operations"][0]["name"],
            "ridge_alpha",
        )
        self.assertEqual(
            row["score_comparability"],
            "same_batch_cohort_paired_comparison",
        )
        self.assertEqual(
            lanes[0]["score_comparability"],
            "same_batch_cohort_paired_comparison",
        )

    def test_dsh_activity_ignores_unresolved_child_before_resume_boundary(self) -> None:
        launch = SimpleNamespace(
            seq=20,
            kind="DshChildLaunchReserved",
            payload={"launch": {
                "stage": "sample.plan",
                "role": "sample-planner",
                "launch_attempt": 1,
                "reservation_id": "reservation-before-pause",
            }},
            created_at="2026-08-28T00:00:00+00:00",
        )
        resumed = SimpleNamespace(
            seq=21,
            kind="RunResumed",
            payload={},
            created_at="2026-08-28T00:01:00+00:00",
        )

        activity = _dsh_activity_projection(
            SimpleNamespace(events=(launch, resumed)),
            current_stage="evaluation",
            run_status="running",
        )

        self.assertIsNone(activity)

    def test_native_retry_wait_clears_on_its_structured_success_contract(self) -> None:
        retry = SimpleNamespace(
            seq=10,
            kind="GatewayRetryScheduled",
            payload={
                "schema_version": "ecologyrsi-dsh.gateway-retry-scheduled/2",
                "generation": 0,
                "stage": "generation",
                "continuity_reset_contract": "dsh_structured_success@1",
            },
            created_at="2026-08-27T06:00:00+00:00",
        )
        success = SimpleNamespace(
            seq=11,
            kind="DshStructuredResultAccepted",
            payload={"identity": {"stage": "sample.plan"}},
            created_at="2026-08-27T06:00:01+00:00",
        )
        state = SimpleNamespace(
            run=SimpleNamespace(generation=0, status=SimpleNamespace(value="running")),
            events=(retry, success),
        )

        self.assertIsNone(_gateway_retry_projection(state))

    def test_screening_progress_retires_launches_before_retry_boundary(self) -> None:
        events = (
            SimpleNamespace(
                seq=10,
                kind="GenerationBatchStarted",
                payload={"batch": {"generation": 0}},
                created_at="2026-08-26T05:59:59+00:00",
            ),
            SimpleNamespace(
                seq=11,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:stale-a",
                    "reservation_id": "reservation-stale-a",
                    "sample_member_digests": _origin_members("stale-a"),
                }},
                created_at="2026-08-26T06:00:00+00:00",
            ),
            SimpleNamespace(
                seq=12,
                kind="GatewayRetryScheduled",
                payload={"generation": 0},
                created_at="2026-08-26T06:00:01+00:00",
            ),
            SimpleNamespace(
                seq=13,
                kind="RunPaused",
                payload={},
                created_at="2026-08-26T06:00:03+00:00",
            ),
            SimpleNamespace(
                seq=14,
                kind="RunResumed",
                payload={},
                created_at="2026-08-26T06:00:04+00:00",
            ),
            SimpleNamespace(
                seq=15,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:current-b",
                    "reservation_id": "reservation-current-b",
                    "sample_member_digests": _origin_members("current-b"),
                }},
                created_at="2026-08-26T06:00:02+00:00",
            ),
        )
        state = SimpleNamespace(
            run=SimpleNamespace(generation=0, status=SimpleNamespace(value="running")),
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "two_stage_evaluation_enabled": True,
                "sample_reflection_policy": "candidate_aggregate_post_score@1",
                "sample_concurrency": 64,
            }}),
            candidates=tuple(SimpleNamespace(generation=0) for _ in range(5)),
            events=events,
        )

        progress = _screening_progress_projection(state)

        self.assertEqual(progress["completed_samples"], 0)
        self.assertEqual(progress["in_flight_batches"], 1)
        self.assertEqual(progress["queued_batches"], 0)
        self.assertEqual(progress["awaiting_submission_batches"], 49)

    def test_screening_progress_retires_durably_failed_child_launch(self) -> None:
        events = (
            SimpleNamespace(
                seq=10,
                kind="GenerationBatchStarted",
                payload={"batch": {"generation": 0}},
                created_at="2026-08-26T05:59:59+00:00",
            ),
            SimpleNamespace(
                seq=11,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:failed-a",
                    "reservation_id": "reservation-failed-a",
                    "sample_member_digests": _origin_members("failed-a"),
                }},
                created_at="2026-08-26T06:00:00+00:00",
            ),
            SimpleNamespace(
                seq=12,
                kind="DshPredictionToolExecuted",
                payload={
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:failed-a",
                    "prediction_count": 9,
                },
                created_at="2026-08-26T06:00:01+00:00",
            ),
            SimpleNamespace(
                seq=13,
                kind="DshChildExecutionFailed",
                payload={
                    "schema_version": "ecologyrsi-dsh.child-execution-failed/1",
                    "identity": {
                        "stage": "sample.plan",
                        "idempotency_key": "run:test:sample.plan:failed-a",
                        "child_reservation_id": "reservation-failed-a",
                    },
                    "error_code": "structured_child_model_error",
                },
                created_at="2026-08-26T06:00:02+00:00",
            ),
        )
        state = SimpleNamespace(
            run=SimpleNamespace(
                generation=0,
                status=SimpleNamespace(value="running"),
            ),
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "two_stage_evaluation_enabled": True,
                "sample_reflection_policy": "candidate_aggregate_post_score@1",
                "sample_concurrency": 64,
            }}),
            candidates=tuple(SimpleNamespace(generation=0) for _ in range(5)),
            events=events,
        )

        progress = _screening_progress_projection(state)

        self.assertEqual(progress["completed_samples"], 0)
        self.assertEqual(progress["succeeded_samples"], 0)
        self.assertEqual(progress["failed_samples"], 0)
        self.assertEqual(progress["prediction_tool_output_origins"], 1)
        self.assertEqual(progress["prediction_tool_pending_origins"], 0)
        self.assertEqual(progress["child_execution_failed_request_count"], 1)
        self.assertEqual(progress["structured_child_model_error_count"], 1)
        self.assertEqual(progress["in_flight_batches"], 0)
        self.assertEqual(progress["queued_batches"], 0)
        self.assertEqual(progress["awaiting_submission_batches"], 50)

    def test_screening_progress_honors_concurrency_above_legacy_eight(
        self,
    ) -> None:
        events = [
            SimpleNamespace(
                seq=10,
                kind="GenerationBatchStarted",
                payload={"batch": {"generation": 0}},
                created_at="2026-08-26T06:00:00+00:00",
            )
        ]
        events.extend(
            SimpleNamespace(
                seq=11 + index,
                kind="DshChildLaunchReserved",
                payload={
                    "launch": {
                        "stage": "sample.plan",
                        "idempotency_key": f"run:test:sample.plan:origin-{index}",
                        "reservation_id": f"reservation-plan-{index}",
                    }
                },
                created_at=f"2026-08-26T06:00:{index + 1:02d}+00:00",
            )
            for index in range(10)
        )
        state = SimpleNamespace(
            run=SimpleNamespace(
                generation=0,
                status=SimpleNamespace(value="running"),
            ),
            task_manifest=SimpleNamespace(
                metadata={**dict(native_manifest().metadata), **{
                    "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                    "two_stage_evaluation_enabled": True,
                    "sample_concurrency": 64,
                }}
            ),
            candidates=tuple(SimpleNamespace(generation=0) for _ in range(5)),
            events=tuple(events),
        )

        progress = _screening_progress_projection(state)

        self.assertEqual(progress["configured_concurrency"], 64)
        self.assertEqual(progress["in_flight_batches"], 10)
        self.assertEqual(progress["queued_batches"], 0)
        self.assertEqual(progress["awaiting_submission_batches"], 40)

    def test_screening_progress_counts_one_origin_across_plan_and_critic(self) -> None:
        events = (
            SimpleNamespace(
                seq=10,
                kind="GenerationBatchStarted",
                payload={"batch": {"generation": 0}},
                created_at="2026-08-26T06:00:00+00:00",
            ),
            SimpleNamespace(
                seq=11,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:origin-a",
                    "reservation_id": "reservation-plan-a",
                    "sample_member_digests": _origin_members("origin-a"),
                }},
                created_at="2026-08-26T06:00:01+00:00",
            ),
            SimpleNamespace(
                seq=12,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.critic",
                    "idempotency_key": "run:test:sample.critic:origin-a",
                    "reservation_id": "reservation-critic-a",
                    "sample_member_digests": _origin_members("origin-a"),
                }},
                created_at="2026-08-26T06:00:02+00:00",
            ),
        )
        state = SimpleNamespace(
            run=SimpleNamespace(
                generation=0,
                status=SimpleNamespace(value="running"),
            ),
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "two_stage_evaluation_enabled": True,
                "sample_reflection_policy": "candidate_aggregate_post_score@1",
                "sample_concurrency": 64,
            }}),
            candidates=tuple(SimpleNamespace(generation=0) for _ in range(5)),
            events=events,
        )

        progress = _screening_progress_projection(state)

        self.assertEqual(progress["completed_samples"], 0)
        self.assertEqual(progress["in_flight_batches"], 1)
        self.assertEqual(progress["queued_batches"], 0)
        self.assertEqual(progress["awaiting_submission_batches"], 49)

    def test_prediction_receipt_retires_correlated_critic_launch(self) -> None:
        members = _origin_members("origin-a")
        events = (
            SimpleNamespace(
                seq=10,
                kind="GenerationBatchStarted",
                payload={"batch": {"generation": 0}},
                created_at="2026-08-26T06:00:00+00:00",
            ),
            SimpleNamespace(
                seq=11,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:origin-a",
                    "reservation_id": "reservation-plan-a",
                    "sample_member_digests": members,
                }},
                created_at="2026-08-26T06:00:01+00:00",
            ),
            SimpleNamespace(
                seq=12,
                kind="DshPredictionToolExecuted",
                payload={
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:sample.plan:origin-a",
                    "prediction_count": 9,
                },
                created_at="2026-08-26T06:00:02+00:00",
            ),
            SimpleNamespace(
                seq=13,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.critic",
                    "idempotency_key": "run:test:sample.critic:origin-a",
                    "reservation_id": "reservation-critic-a",
                    "sample_member_digests": members,
                }},
                created_at="2026-08-26T06:00:03+00:00",
            ),
        )
        state = SimpleNamespace(
            run=SimpleNamespace(
                generation=0,
                status=SimpleNamespace(value="running"),
            ),
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "two_stage_evaluation_enabled": True,
                "sample_reflection_policy": "candidate_aggregate_post_score@1",
                "sample_concurrency": 64,
            }}),
            candidates=tuple(SimpleNamespace(generation=0) for _ in range(5)),
            events=events,
        )

        progress = _screening_progress_projection(state)

        self.assertEqual(progress["completed_samples"], 0)
        self.assertEqual(progress["prediction_tool_pending_origins"], 1)
        self.assertEqual(progress["in_flight_batches"], 1)
        self.assertEqual(progress["awaiting_submission_batches"], 49)

    def test_screening_progress_deduplicates_retried_origin(self) -> None:
        members = _origin_members("origin-a")
        events = (
            SimpleNamespace(
                seq=10,
                kind="GenerationBatchStarted",
                payload={"batch": {"generation": 0}},
                created_at="2026-08-26T06:00:00+00:00",
            ),
            SimpleNamespace(
                seq=11,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:attempt-a:sample.plan:origin-a",
                    "reservation_id": "reservation-plan-a",
                    "sample_member_digests": members,
                }},
                created_at="2026-08-26T06:00:01+00:00",
            ),
            SimpleNamespace(
                seq=12,
                kind="DshPredictionToolExecuted",
                payload={
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:attempt-a:sample.plan:origin-a",
                    "prediction_count": 9,
                },
                created_at="2026-08-26T06:00:02+00:00",
            ),
            SimpleNamespace(
                seq=13,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:attempt-b:sample.plan:origin-a",
                    "reservation_id": "reservation-plan-b",
                    "sample_member_digests": members,
                }},
                created_at="2026-08-26T06:00:05+00:00",
            ),
            SimpleNamespace(
                seq=16,
                kind="DshPredictionToolExecuted",
                payload={
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:attempt-b:sample.plan:origin-a",
                    "prediction_count": 9,
                },
                created_at="2026-08-26T06:00:06+00:00",
            ),
        )
        state = SimpleNamespace(
            run=SimpleNamespace(
                generation=0,
                status=SimpleNamespace(value="running"),
            ),
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "two_stage_evaluation_enabled": True,
                "sample_reflection_policy": "candidate_aggregate_post_score@1",
                "sample_concurrency": 64,
            }}),
            candidates=tuple(SimpleNamespace(generation=0) for _ in range(5)),
            events=events,
        )

        progress = _screening_progress_projection(state)

        self.assertEqual(progress["completed_samples"], 0)
        self.assertEqual(progress["succeeded_samples"], 0)
        self.assertEqual(progress["prediction_tool_output_origins"], 1)
        self.assertEqual(progress["prediction_tool_pending_origins"], 1)
        self.assertEqual(progress["in_flight_batches"], 1)
        self.assertEqual(progress["queued_batches"], 0)
        self.assertEqual(progress["awaiting_submission_batches"], 49)

    def test_completed_origin_retires_later_retry_launch(self) -> None:
        members = _origin_members("origin-a")
        events = (
            SimpleNamespace(
                seq=10,
                kind="GenerationBatchStarted",
                payload={"batch": {"generation": 0}},
                created_at="2026-08-26T06:00:00+00:00",
            ),
            SimpleNamespace(
                seq=11,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:attempt-a:sample.plan:origin-a",
                    "reservation_id": "reservation-plan-a",
                    "sample_member_digests": members,
                }},
                created_at="2026-08-26T06:00:01+00:00",
            ),
            SimpleNamespace(
                seq=12,
                kind="DshPredictionToolExecuted",
                payload={
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:attempt-a:sample.plan:origin-a",
                    "prediction_count": 9,
                },
                created_at="2026-08-26T06:00:02+00:00",
            ),
            SimpleNamespace(
                seq=13,
                kind="RunResumed",
                payload={},
                created_at="2026-08-26T06:00:03+00:00",
            ),
            SimpleNamespace(
                seq=14,
                kind="DshChildLaunchReserved",
                payload={"launch": {
                    "stage": "sample.plan",
                    "idempotency_key": "run:test:attempt-b:sample.plan:origin-a",
                    "reservation_id": "reservation-plan-b",
                    "sample_member_digests": members,
                }},
                created_at="2026-08-26T06:00:04+00:00",
            ),
        )
        state = SimpleNamespace(
            run=SimpleNamespace(
                generation=0,
                status=SimpleNamespace(value="running"),
            ),
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "two_stage_evaluation_enabled": True,
                "sample_reflection_policy": "candidate_aggregate_post_score@1",
                "sample_concurrency": 64,
            }}),
            candidates=tuple(SimpleNamespace(generation=0) for _ in range(5)),
            events=events,
        )

        progress = _screening_progress_projection(state)

        self.assertEqual(progress["completed_samples"], 0)
        self.assertEqual(progress["prediction_tool_pending_origins"], 0)
        self.assertEqual(progress["in_flight_batches"], 1)
        self.assertEqual(progress["queued_batches"], 0)
        self.assertEqual(progress["awaiting_submission_batches"], 49)

    def test_evaluation_progress_projects_configured_concurrency_up_to_128(
        self,
    ) -> None:
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{"sample_concurrency": 128}}),
            events=(
                SimpleNamespace(
                    seq=20,
                    kind="EvaluationProgressRecorded",
                    payload={
                        "candidate_id": "candidate:projection-concurrency",
                        "role": "planner",
                        "completed_samples": 1,
                        "total_samples": 10,
                        "in_flight_batches": 3,
                        "queued_batches": 0,
                    },
                    created_at="2026-08-26T06:00:00+00:00",
                ),
            ),
        )

        progress = _evaluation_progress_projection(
            state,
            "candidate:projection-concurrency",
        )

        self.assertEqual(progress["configured_concurrency"], 128)
        self.assertEqual(progress["awaiting_submission_batches"], 6)



    def test_completed_run_hides_recovered_failed_stage(self) -> None:
        failed_stage = SimpleNamespace(
            seq=10,
            kind="EvolutionStageRecorded",
            payload={
                "generation": 0,
                "stage": "reflection",
                "status": "failed",
                "attempt": 3,
                "public_error": "temporary structured response failure",
            },
            created_at="2026-08-25T14:36:12+00:00",
        )
        completed = SimpleNamespace(
            seq=20,
            kind="RunCompleted",
            payload={},
            created_at="2026-08-25T14:39:02+00:00",
        )
        state = SimpleNamespace(
            run=SimpleNamespace(generation=1, status=SimpleNamespace(value="completed")),
            events=(failed_stage, completed),
        )

        failure_reason, projected_stage = _run_failure_projection(state)

        self.assertIsNone(failure_reason)
        self.assertIsNone(projected_stage)

    def test_dsh_activity_projects_running_retry_and_host_validation(self) -> None:
        stage_started = SimpleNamespace(
            seq=10,
            kind="EvolutionStageRecorded",
            payload={"stage": "evaluation", "status": "started"},
            created_at="2026-08-25T13:44:02+00:00",
        )
        launch = SimpleNamespace(
            seq=20,
            kind="DshChildLaunchReserved",
            payload={
                "launch": {
                    "stage": "sample.critic",
                    "role": "sample-critic",
                    "launch_attempt": 2,
                    "reservation_id": "reservation-critic-2",
                }
            },
            created_at="2026-08-25T13:54:17+00:00",
        )
        running_state = SimpleNamespace(events=(stage_started, launch))

        running = _dsh_activity_projection(
            running_state,
            current_stage="evaluation",
            run_status="running",
        )

        self.assertEqual(running["state"], "model_retry_running")
        self.assertEqual(running["dsh_stage"], "sample.critic")
        self.assertEqual(running["role"], "sample-critic")
        self.assertEqual(running["launch_attempt"], 2)
        self.assertEqual(running["started_at"], launch.created_at)
        self.assertNotIn("reservation_id", running)

        remote = {"kind": "streaming", "log_revision": 40, "updated_at": "2026-08-25T13:54:20+00:00"}
        telemetry = SimpleNamespace(seq=21, kind="DshSessionUsageRecorded", created_at="2026-08-25T13:54:25+00:00",
                                    payload={"identity": {"child_reservation_id": "reservation-critic-2"},
                                             "session_metrics": {"activity": remote}})
        unrelated = SimpleNamespace(seq=22, kind="DshSessionUsageRecorded", created_at="2026-08-25T14:10:00+00:00",
                                    payload={"identity": {"child_reservation_id": "another"}, "session_metrics": {}})
        observed = _dsh_activity_projection(SimpleNamespace(events=(stage_started, launch, telemetry, unrelated)),
                                           current_stage="evaluation", run_status="running")
        self.assertEqual(observed["remote_activity"], remote)
        self.assertEqual(observed["updated_at"], remote["updated_at"])

        accepted = SimpleNamespace(
            seq=21,
            kind="DshStructuredResultAccepted",
            payload={
                "identity": {
                    "stage": "sample.critic",
                    "child_reservation_id": "reservation-critic-2",
                }
            },
            created_at="2026-08-25T13:54:41+00:00",
        )
        validating = _dsh_activity_projection(
            SimpleNamespace(events=(stage_started, launch, accepted)),
            current_stage="evaluation",
            run_status="running",
        )

        self.assertEqual(validating["state"], "host_validating")
        self.assertEqual(validating["updated_at"], accepted.created_at)

    def test_dsh_activity_projects_provider_slot_wait_without_fabricated_progress(
        self,
    ) -> None:
        stage_started = SimpleNamespace(
            seq=10,
            kind="EvolutionStageRecorded",
            payload={"stage": "research", "status": "started"},
            created_at="2026-08-25T13:38:27+00:00",
        )

        activity = _dsh_activity_projection(
            SimpleNamespace(events=(stage_started,)),
            current_stage="research",
            run_status="running",
        )

        self.assertEqual(activity["state"], "waiting_for_model_slot")
        self.assertEqual(activity["evolution_stage"], "research")
        self.assertIsNone(activity["dsh_stage"])
        self.assertNotIn("progress_percent", activity)

    def test_dsh_activity_maps_search_plan_to_search_stage(self) -> None:
        stage_started = SimpleNamespace(
            seq=10,
            kind="EvolutionStageRecorded",
            payload={"stage": "search", "status": "started"},
            created_at="2026-08-25T13:38:27+00:00",
        )
        launch = SimpleNamespace(
            seq=11,
            kind="DshChildLaunchReserved",
            payload={
                "launch": {
                    "stage": "generation.search-plan",
                    "role": "coordinator",
                    "launch_attempt": 1,
                    "reservation_id": "reservation-search-1",
                }
            },
            created_at="2026-08-25T13:38:31+00:00",
        )

        activity = _dsh_activity_projection(
            SimpleNamespace(events=(stage_started, launch)),
            current_stage="search",
            run_status="running",
        )

        self.assertEqual(activity["state"], "model_running")
        self.assertEqual(activity["evolution_stage"], "search")
        self.assertEqual(activity["dsh_stage"], "generation.search-plan")
        self.assertEqual(activity["role"], "coordinator")

    def test_dsh_activity_is_hidden_after_run_is_terminal(self) -> None:
        activity = _dsh_activity_projection(
            SimpleNamespace(events=()),
            current_stage=None,
            run_status="completed",
        )

        self.assertIsNone(activity)



    def test_candidate_local_edit_structured_result_replays(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = start_native_run(director, task=_task(),run_id='run:projection-candidate-local-edit').run.run_id
            structured = {
                "schema_version": "ecology-local-edit@1",
                "decision": "mutate",
                "operations": [
                    {
                        "op": "select_registered_pipeline",
                        "predictor_id": "greenhouse-horizon-targetwise-ridge@1",
                    }
                ],
                "evidence_refs": ["batch:score", "batch:metrics"],
                "expected_effect_cells": ["air_temperature@6h"],
                "risk_cells": ["relative_humidity@6h"],
            }
            ledger.append(
                run_id,
                "DshStructuredResultAccepted",
                {
                    "schema_version": "ecologyrsi-dsh.structured-result-accepted/1",
                    "identity": {
                        "run_id": run_id,
                        "role": "candidate-proposer",
                        "stage": "candidate.local_edit",
                        "session_id": "dsh-child-local-editor-1",
                    },
                    "output_schema_id": "ecology-local-edit@1",
                    "result_digest": digest(structured),
                    "structured": structured,
                    "skill_invocation_evidence": _skill_evidence(
                        "candidate.local_edit",
                        "bounded-plugin-experiment",
                    ),
                },
            )

            state = director.state(run_id)

        self.assertEqual(state.events[-1].kind, "DshStructuredResultAccepted")
        self.assertEqual(
            state.events[-1].payload["identity"]["stage"],
            "candidate.local_edit",
        )

    def test_candidate_review_skill_evidence_replays(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = start_native_run(director, task=_task(),run_id='run:projection-candidate-review').run.run_id
            structured = {
                "schema_version": "ecology-generation-review@1",
                "accepted": False,
                "rationale": "The supplied candidate evidence is insufficient.",
                "flags": ["insufficient_evidence"],
            }
            ledger.append(
                run_id,
                "DshStructuredResultAccepted",
                {
                    "schema_version": "ecologyrsi-dsh.structured-result-accepted/1",
                    "identity": {
                        "run_id": run_id,
                        "role": "generation-judge",
                        "stage": "generation.judge",
                        "session_id": "dsh-child-candidate-review-1",
                    },
                    "output_schema_id": "ecology-generation-review@1",
                    "result_digest": digest(structured),
                    "structured": structured,
                    "skill_invocation_evidence": _skill_evidence(
                        "generation.judge",
                        "candidate-scientific-review",
                    ),
                },
            )

            state = director.state(run_id)

        self.assertEqual(state.events[-1].kind, "DshStructuredResultAccepted")
        self.assertEqual(
            state.events[-1].payload["skill_invocation_evidence"]["skill_name"],
            "candidate-scientific-review",
        )



    def test_public_metrics_hide_private_sample_and_promotion_evidence(self) -> None:
        public = _public_evaluation_metrics(
            {
                "score": 0.2,
                "sample_execution_records": [{"sample_id": "private"}],
                "sample_execution_trace_archive": "private",
                "promotion_block_evidence": {"blocks": [{"cells": []}]},
            }
        )

        self.assertEqual(public, {"score": 0.2})

    def test_public_metrics_tolerate_malformed_historical_control(self) -> None:
        public = _public_evaluation_metrics(
            {
                "generation_control_evaluations": [
                    {
                        "comparison_role": "search_parent",
                        "candidate_id": "candidate:legacy-parent",
                        "evaluation": {
                            "candidate_id": "candidate:legacy-parent",
                            "score": 0.2,
                            "metrics": {
                                "sample_execution": {
                                    "strict_agent_chain_pass": True,
                                }
                            },
                        },
                    }
                ]
            }
        )

        self.assertEqual(len(public["generation_controls"]), 1)
        self.assertIsNone(
            public["generation_controls"][0]["evaluation_cohort_digest"]
        )










    def test_v2_progress_rejects_batch_larger_than_completed_samples(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = start_native_run(director, task=_task(),run_id='run:projection-v2-progress-validation').run.run_id
            candidate = spawn_native_candidate(director, run_id)
            with self.assertRaisesRegex(ValueError, "batch size"):
                director.record_evaluation_progress(
                    run_id,
                    generation=candidate.generation,
                    proposal_id=candidate.proposal_id,
                    candidate_id=candidate.candidate_id,
                    progress={
                        "role": "planner",
                        "model_id": "newapi/glm-5.2",
                        "batch_index": 1,
                        "batch_count": 2,
                        "batch_size": 2,
                        "completed_samples": 1,
                        "total_samples": 4,
                        "succeeded_samples": 1,
                        "failed_samples": 0,
                    },
                )

    def test_failed_run_exposes_reason_and_latest_failed_stage(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = start_native_run(director, task=_task(max_candidates=1),run_id='run:projection-failed').run.run_id
            director.record_evolution_stage(
                run_id,
                generation=0,
                stage="proposal",
                status="failed",
                public_error="TimeoutError: strategy request timed out",
            )
            director.fail_run(run_id, "remote strategy API timed out")

            projection = _projection_json(director.state(run_id))

            self.assertEqual(projection["status"], "failed")
            self.assertEqual(
                projection["failure_reason"], "remote strategy API timed out"
            )
            self.assertEqual(projection["failed_stage"]["generation"], 0)
            self.assertEqual(projection["failed_stage"]["stage"], "proposal")
            self.assertIn("TimeoutError", projection["failed_stage"]["public_error"])




    def test_stopped_runs_do_not_project_unfinished_candidates_as_running(
        self,
    ) -> None:
        cases = (
            ("paused", "paused", "paused", "retained_partial"),
            ("cancelled", "cancelled", "not_recorded", "aborted_partial"),
            ("failed", "failed", "not_recorded", "aborted_partial"),
            ("completed", "completed", "not_recorded", "aborted_partial"),
        )
        for label, expected_phase, expected_stage, expected_evidence in cases:
            with self.subTest(status=label), EventLedger() as ledger:
                director = EvolutionDirector(ledger, NativeRoleStub())
                run_id = f"run:stopped-stage-projection:{label}"
                start_native_run(director, task=_task(),run_id=run_id)
                candidate = spawn_native_candidate(director, run_id)
                director.record_evolution_stage(
                    run_id,
                    generation=0,
                    proposal_id=candidate.proposal_id,
                    candidate_id=candidate.candidate_id,
                    stage="evaluation",
                    status="started",
                )
                director.record_evaluation_progress(
                    run_id,
                    generation=0,
                    proposal_id=candidate.proposal_id,
                    candidate_id=candidate.candidate_id,
                    progress={
                        "role": "planner",
                        "model_id": "projection-test-model",
                        "batch_index": 1,
                        "batch_count": 2,
                        "batch_size": 1,
                        "completed_samples": 1,
                        "total_samples": 2,
                        "succeeded_samples": 1,
                        "failed_samples": 0,
                    },
                )
                live = _projection_json(director.state(run_id))
                self.assertEqual(live["execution_progress"]["phase"], "evaluation")
                self.assertEqual(
                    live["candidates"][0]["execution"]["stages"]["evaluation"],
                    "running",
                )

                if label == "paused":
                    director.pause_run(run_id, reason="operator pause")
                elif label == "cancelled":
                    director.cancel_run(run_id, "operator cancel")
                elif label == "failed":
                    director.fail_run(run_id, "runtime failed")
                else:
                    director.complete_run(run_id)
                projection = _projection_json(director.state(run_id))

                candidate_view = projection["candidates"][0]
                expected_candidate_status = (
                    "paused" if label == "paused" else "aborted"
                )
                self.assertEqual(candidate_view["status"], expected_candidate_status)
                self.assertEqual(
                    candidate_view["execution"]["status"],
                    expected_candidate_status,
                )
                self.assertEqual(
                    candidate_view["execution"]["stages"]["evaluation"],
                    expected_stage,
                )
                self.assertNotIn(
                    "running",
                    candidate_view["execution"]["stages"].values(),
                )
                self.assertEqual(
                    projection["execution_progress"]["phase"],
                    expected_phase,
                )
                self.assertEqual(
                    projection["execution_progress"]["current_stage"],
                    "evaluation" if label == "paused" else None,
                )
                self.assertEqual(
                    projection["execution_diagnostics"]["execution_evidence_status"],
                    expected_evidence,
                )
                round_view = projection["rounds"][0]
                self.assertNotIn("running", round_view["stages"].values())
                self.assertNotIn(
                    "running",
                    round_view["candidates"][0]["stages"].values(),
                )
                if label == "paused":
                    director.resume_run(run_id)
                    resumed = _projection_json(director.state(run_id))
                    self.assertEqual(resumed["execution_progress"]["phase"], "evaluation")
                    self.assertEqual(resumed["candidates"][0]["status"], "evaluating")
                    self.assertEqual(
                        resumed["candidates"][0]["execution"]["stages"][
                            "evaluation"
                        ],
                        "running",
                    )

    def test_recent_evaluation_progress_rates_use_a_bounded_window(self) -> None:
        events = [
            SimpleNamespace(
                seq=index,
                created_at=f"2026-08-19T00:{index:02d}:00+00:00",
                payload={
                    "completed_samples": index * 10,
                    "gateway_request_count": index * 2,
                },
            )
            for index in range(1, 13)
        ]

        sample_rate, gateway_rate = _evaluation_progress_rates(
            events, events[-1]
        )

        self.assertEqual(sample_rate, 10.0)
        self.assertEqual(gateway_rate, 2.0)

    def test_evaluation_progress_rates_accept_a_short_history(self) -> None:
        events = [
            SimpleNamespace(
                seq=index,
                created_at=f"2026-08-19T00:00:0{index}+00:00",
                payload={
                    "completed_samples": index,
                    "gateway_request_count": index,
                },
            )
            for index in range(1, 4)
        ]

        sample_rate, gateway_rate = _evaluation_progress_rates(
            events, events[-1]
        )

        self.assertEqual(sample_rate, 60.0)
        self.assertEqual(gateway_rate, 60.0)

    def test_evaluation_progress_rates_exclude_pre_resume_downtime(self) -> None:
        events = [
            SimpleNamespace(
                seq=10,
                payload={"completed_samples": 0, "gateway_request_count": 0},
                created_at="2026-08-26T06:00:00+00:00",
            ),
            SimpleNamespace(
                seq=30,
                payload={"completed_samples": 10, "gateway_request_count": 10},
                created_at="2026-08-26T06:10:00+00:00",
            ),
            SimpleNamespace(
                seq=40,
                payload={"completed_samples": 20, "gateway_request_count": 20},
                created_at="2026-08-26T06:11:00+00:00",
            ),
        ]

        sample_rate, gateway_rate = _evaluation_progress_rates(
            events,
            events[-1],
            after_seq=20,
        )

        self.assertEqual(sample_rate, 10.0)
        self.assertEqual(gateway_rate, 10.0)


    def test_projection_exposes_isolated_legacy_revision_aggregate(self) -> None:
        """An isolated revision is visible without reviving its counters."""

        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = start_native_run(director, task=_task(max_candidates=1),run_id='run:projection-superseded-revision').run.run_id
            candidate = spawn_native_candidate(director, run_id)
            director.record_evolution_stage(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                stage="evaluation",
                status="started",
            )
            legacy_revision = "revision:legacy-uncheckpointed"
            director.start_evaluation_sample_results(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                revision=legacy_revision,
            )
            director.record_evaluation_progress(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                progress={
                    "role": "planner",
                    "model_id": "newapi/glm-5.2",
                    "batch_index": 8,
                    "batch_count": 12,
                    "batch_size": 9,
                    "completed_samples": 72,
                    "total_samples": 100,
                    "succeeded_samples": 70,
                    "failed_samples": 2,
                },
            )
            active_revision = "revision:checkpointed-replacement"
            director.start_evaluation_sample_results(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                revision=active_revision,
                checkpoint={
                    "schema_version": "ecologyrsi-dsh.sample-checkpoint/1",
                    "cohort_digest": "a" * 64,
                    "execution_context_digest": "b" * 64,
                    "sample_count": 100,
                },
                supersedes_revision=legacy_revision,
                resume_disposition="legacy_revision_without_checkpoint",
            )
            director.record_evaluation_progress(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                revision=active_revision,
                progress={
                    "role": "planner",
                    "model_id": "newapi/glm-5.2",
                    "batch_index": 0,
                    "batch_count": 12,
                    "batch_size": 0,
                    "completed_samples": 0,
                    "total_samples": 100,
                    "succeeded_samples": 0,
                    "failed_samples": 0,
                    "gateway_request_count": 0,
                    "adaptive_split_trigger_count": 0,
                    "adaptive_split_count": 0,
                    "adaptive_split_max_depth": 0,
                    "adaptive_split_recovered_samples": 0,
                    "adaptive_split_failed_samples": 0,
                    "progress_id": 1,
                    "progress_kind": "waiting",
                    "in_flight_batches": 2,
                    "queued_batches": 0,
                },
            )

            projection = _projection_json(director.state(run_id))

        candidate_execution = projection["candidates"][0]["execution"]
        active_progress = candidate_execution["stage_progress"]
        superseded = candidate_execution["superseded_sample_revision"]
        self.assertEqual(active_progress["revision"], active_revision)
        self.assertEqual(active_progress["completed_samples"], 0)
        self.assertEqual(superseded["revision"], legacy_revision)
        self.assertEqual(superseded["resume_disposition"], "legacy_revision_without_checkpoint")
        self.assertEqual(superseded["completed_samples"], 72)
        self.assertEqual(superseded["succeeded_samples"], 70)
        self.assertEqual(superseded["failed_samples"], 2)
        self.assertEqual(superseded["total_samples"], 100)
        self.assertEqual(
            projection["execution_progress"]["superseded_sample_revision"],
            superseded,
        )
        self.assertNotIn("sample_id", str(superseded))
        self.assertNotIn("observed", str(superseded))

    def test_projection_omits_superseded_summary_for_normal_resume(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = start_native_run(director, task=_task(max_candidates=1),run_id='run:projection-normal-resume').run.run_id
            candidate = spawn_native_candidate(director, run_id)
            revision = "revision:matching-checkpoint"
            checkpoint = {
                "schema_version": "ecologyrsi-dsh.sample-checkpoint/1",
                "cohort_digest": "a" * 64,
                "execution_context_digest": "b" * 64,
                "sample_count": 4,
            }
            director.start_evaluation_sample_results(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                revision=revision,
                checkpoint=checkpoint,
            )
            director.prepare_evaluation_sample_checkpoint(
                run_id,
                generation=candidate.generation,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                checkpoint=checkpoint,
            )

            execution = _projection_json(director.state(run_id))["candidates"][0][
                "execution"
            ]

        self.assertIsNone(execution["superseded_sample_revision"])








if __name__ == "__main__":
    unittest.main()

class NativeReplicaProgressTests(unittest.TestCase):
    def test_joint_progress_counts_screening_reuse_and_paired_batch(self):
        schedule = OptimizationSchedule.for_new_run()
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
                'optimization_protocol': 'joint_plugin_epoch@2',
                'optimization_schedule': OptimizationSchedule.for_new_run().to_dict(),
                'prediction_cells_per_origin': 9,
            }}),
            run=SimpleNamespace(generation=0, status=SimpleNamespace(value='running')),
            candidates=(), candidate_screening_events=(),
            formal_batch_evaluations=(), holdout_evaluations=(), formal_batches=(),
            generation_cohort_for=lambda _generation: None,
            events=(SimpleNamespace(seq=1, kind='FormalSelectionCohortFrozen',
                payload={'generation': 0}, created_at='2026-10-05T17:19:00+00:00'),),
        )
        for index in (0, 1):
            with self.subTest(batch_index=index):
                state.formal_batches = (SimpleNamespace(
                    generation=0, candidate_id='candidate:joint', batch_index=index),)
                progress = _adaptive_progress_projection(state)
                self.assertEqual(progress['evaluation_phase'], 'formal_batch')
                self.assertEqual(progress['current_batch'], index + 1)
                self.assertEqual(progress['batch_count'], 2)
                self.assertEqual(progress['formal_total_origins'], 50)
                self.assertEqual(progress['total_origins'], 300)


    def test_native_progress_uses_the_same_execution_budget_as_capacity(self):
        schedule = OptimizationSchedule.for_new_run()
        state = SimpleNamespace(task_manifest=SimpleNamespace(metadata={**dict(native_manifest().metadata), **{
            'sample_agent_mode':'dsh_native_agent', 'optimization_protocol':'joint_plugin_epoch@2',
            'optimization_schedule':OptimizationSchedule.for_new_run().to_dict(), 'prediction_cells_per_origin':9}}),
            run=SimpleNamespace(generation=0,status=SimpleNamespace(value='running')), candidates=(),
            candidate_screening_events=(), formal_batch_evaluations=(), holdout_evaluations=(), formal_batches=(),
            events=(SimpleNamespace(seq=1,kind='HoldoutArmStarted', created_at='2026-09-08T00:00:00+00:00',
                payload={'generation':0,'holdout_arm':'incumbent','candidate_id':'c','origin_count':169}),))
        progress = _adaptive_progress_projection(state)
        self.assertEqual(progress['total_origins'], 300)
        self.assertEqual(progress['holdout_total_origins'], 200)
        self.assertEqual(progress['total_origins'], schedule.generation_execution_budget(
            cells_per_origin=9,holdout_inference_replicas=1)['total_candidate_origins'])
        # Durable holdout batches remain visible during a pause or terminal
        # stop, but those states do not predict a future completion time.
        state.events += tuple(SimpleNamespace(seq=i + 2, kind='EvaluationSampleResultBatchRecorded',
            created_at=f'2026-09-08T00:00:0{i + 1}+00:00', payload={
                'candidate_id':'c','revision':'holdout-results','batch_index':i + 1,
                'record_count':9,'sample_ids':[f'origin-{i}:cell-{n}' for n in range(9)]})
            for i in range(2))
        for status in ('paused', 'cancelled', 'failed', 'completed'):
            state.run.status.value = status
            stopped = _adaptive_progress_projection(state)
            self.assertIsNone(stopped['estimated_remaining_seconds'])
            self.assertIsNone(stopped['run_estimated_remaining_seconds'])
