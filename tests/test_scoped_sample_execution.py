from __future__ import annotations

import unittest
from types import SimpleNamespace

from ecologyrsi_dsh import EventLedger, EvolutionDirector, FakeDSHAdapter, TaskManifest
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.screening import screening_cohort_digest
from ecologyrsi_dsh.core.sample_results import (
    build_sample_results,
    sample_result_batch_event_payload,
    sample_results_completion_payload,
)
from ecologyrsi_dsh.core.trajectory import (
    BatchEvaluation,
    CandidateRevision,
    EvaluationPhase,
    EvaluationScope,
    LocalEditOutcome,
    RevisionAdvanceReason,
    RevisionStatus,
)
from ecologyrsi_dsh.data.greenhouse import FeatureSpec
from ecologyrsi_dsh.data.registry import DatasetSeries
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.evaluators.epoch_cohorts import (
    plan_generation_selection_cohorts,
    plan_run_adaptation_cohort,
)
from ecologyrsi_dsh.evaluators.greenhouse_prediction import (
    ExogenousRidgeConfig,
    MAX_EXOGENOUS_RIDGE_HISTORY_STEPS,
    fit_predict_exogenous_ridge,
)
from ecologyrsi_dsh.evaluators.registry import _select_planned_evaluation_cohort
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule


def _sha(label: str) -> str:
    return digest({"label": label})


class ScopedSampleExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = EventLedger()
        self.director = EvolutionDirector(
            self.ledger, FakeDSHAdapter(max_proposals=20)
        )
        self.schedule = OptimizationSchedule.from_dict(
            {
                **OptimizationSchedule.default().to_dict(),
                "formal_origin_count_per_finalist": 100,
                "local_batch_origin_count": 10,
            }
        )
        task = TaskManifest(
            task_id="scoped-sample-execution",
            objective="isolate every adaptive evaluation scope",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={
                "max_generations": 1,
                "candidates_per_generation": 4,
                "max_candidates": 4,
            },
            seed=7,
            metadata={
                "episode_id": "episode:scoped-sample-execution",
                "optimization_protocol": "top2_adaptive_epoch@1",
                "optimization_schedule": self.schedule.to_dict(),
                "prediction_cells_per_origin": 9,
            },
        )
        self.run_id = "run:scoped-sample-execution"
        self.director.start_evolution(task, run_id=self.run_id)
        self.candidates = tuple(
            self.director.propose_and_spawn(self.run_id) for _ in range(4)
        )
        self.revisions: dict[str, CandidateRevision] = {}
        for index, candidate in enumerate(self.candidates):
            revision = CandidateRevision(
                revision_id=f"revision:scope:{index}:0",
                run_id=self.run_id,
                generation=0,
                candidate_id=candidate.candidate_id,
                genome={"parameters": {"slot": index}},
                genome_digest=_sha(f"genome:{index}"),
                behavior_digest=_sha(f"behavior:{index}"),
                mutation_digest=_sha(f"mutation:{index}"),
                status=RevisionStatus.ACTIVE,
            )
            self.director.create_candidate_revision(self.run_id, revision)
            self.revisions[candidate.candidate_id] = revision
        dataset = SimpleNamespace(
            dataset_id="generated-toy-series@1",
            episode_id="episode:scoped-sample-execution",
            timestamps=tuple(range(3200)),
            partitions={"model_selection": IndexRange(0, 3200)},
        )
        adaptation = plan_run_adaptation_cohort(
            dataset, schedule=self.schedule, seed=7
        )
        self.generation_cohorts = plan_generation_selection_cohorts(
            dataset,
            schedule=self.schedule,
            generation=0,
            adaptation=adaptation,
            seed=7,
        )
        self.director.freeze_run_adaptation_cohort(self.run_id, adaptation)
        self.director.freeze_generation_selection_cohorts(
            self.run_id, self.generation_cohorts
        )
        for candidate in self.candidates:
            self.director.record_candidate_screening(
                self.run_id,
                candidate_id=candidate.candidate_id,
                generation=0,
                score=1.0 - candidate.slot_index * 0.1,
                passed=True,
                constraint_violations=0,
                origin_count=self.schedule.screening_origin_count,
                prediction_cell_count=self.schedule.screening_origin_count * 9,
                cohort_digest=self.generation_cohorts.screening.cohort_digest,
            )
        records = [
            event.payload
            for event in self.director.state(self.run_id).candidate_screening_events
        ]
        self.finalist = self.candidates[0]
        self.director.freeze_formal_selection_cohort(
            self.run_id,
            generation=0,
            selected_candidate_ids=[
                self.candidates[0].candidate_id,
                self.candidates[1].candidate_id,
            ],
            screening_digest=screening_cohort_digest(records),
        )
        self.director.start_formal_trajectory(
            self.run_id,
            self.finalist.candidate_id,
            self.revisions[self.finalist.candidate_id].revision_id,
            self.schedule.batch_count,
        )

    def tearDown(self) -> None:
        self.ledger.close()

    @staticmethod
    def _checkpoint(scope: EvaluationScope) -> dict[str, object]:
        return {
            "schema_version": "ecologyrsi-dsh.sample-checkpoint/2",
            "candidate_revision_id": scope.candidate_revision_id,
            "evaluation_phase": scope.phase.value,
            "formal_batch_index": scope.batch_index,
            "holdout_arm": None,
            "cohort_digest": scope.cohort_digest,
            "execution_scope_digest": scope.scope_key,
            "sample_cohort_digest": _sha(f"samples:{scope.scope_key}"),
            "execution_context_digest": _sha(f"context:{scope.scope_key}"),
            "sample_count": scope.origin_count * 9,
        }

    @staticmethod
    def _sample_rows(candidate_id: str, count: int) -> tuple[dict, ...]:
        return build_sample_results(
            candidate_id,
            [
                {
                    "sample_index": index,
                    "sample_id": f"sample:{index}",
                    "target": "air_temperature",
                    "unit": "degC",
                    "horizon_hours": 1,
                    "origin_timestamp": index,
                    "target_timestamp": index + 1,
                    "observed": 21.0,
                    "predicted": 20.5,
                    "baseline": 20.0,
                    "sample_execution_status": "succeeded",
                    "sample_execution_attempts": 1,
                    "sample_execution_retry_count": 0,
                }
                for index in range(1, count + 1)
            ],
        )

    def test_checkpoints_are_isolated_by_revision_phase_and_batch(self) -> None:
        candidate = self.finalist
        revision = self.revisions[candidate.candidate_id]
        adaptation = self.director.state(self.run_id).run_adaptation_cohort
        assert adaptation is not None

        first_batch = adaptation.batches[0]
        started = self.director.start_formal_batch(
            self.run_id,
            candidate.candidate_id,
            revision.revision_id,
            0,
        )
        self.assertEqual(started.cohort_digest, first_batch.cohort.cohort_digest)
        self.assertNotEqual(started.cohort_digest, first_batch.batch_digest)
        first_scope = EvaluationScope(
            run_id=self.run_id,
            generation=0,
            candidate_id=candidate.candidate_id,
            candidate_revision_id=revision.revision_id,
            phase=EvaluationPhase.FORMAL_BATCH,
            cohort_digest=first_batch.cohort.cohort_digest,
            origin_count=first_batch.origin_count,
            batch_index=0,
        )
        prepared = self.director.prepare_evaluation_sample_checkpoint(
            self.run_id,
            generation=0,
            proposal_id=candidate.proposal_id,
            candidate_id=candidate.candidate_id,
            scope=first_scope,
            checkpoint=self._checkpoint(first_scope),
        )
        resumed = self.director.prepare_evaluation_sample_checkpoint(
            self.run_id,
            generation=0,
            proposal_id=candidate.proposal_id,
            candidate_id=candidate.candidate_id,
            scope=first_scope,
            checkpoint=self._checkpoint(first_scope),
        )
        self.assertEqual(resumed["revision"], prepared["revision"])
        self.assertTrue(resumed["resumed"])

        evaluation = BatchEvaluation(
            evaluation_id="evaluation:scope:0",
            scope=first_scope,
            score=0.5,
            passed=True,
            metrics={"rmse": 1.0},
            evaluator_digest=_sha("evaluator"),
        )
        rows = self._sample_rows(
            candidate.candidate_id,
            first_scope.origin_count * 9,
        )
        self.director.record_evaluation_sample_result_batch(
            self.run_id,
            sample_result_batch_event_payload(
                self.run_id,
                candidate.candidate_id,
                rows,
                revision=prepared["revision"],
                batch_index=1,
            ),
        )
        self.director.record_formal_batch_evaluation(
            self.run_id,
            evaluation,
            sample_results=sample_results_completion_payload(
                run_id=self.run_id,
                evaluation_id=evaluation.evaluation_id,
                candidate_id=candidate.candidate_id,
                rows=rows,
                revision=prepared["revision"],
            ),
        )
        self.director.record_local_edit_proposal(
            self.run_id,
            {
                "proposal_id": "local:scope:0",
                "candidate_id": candidate.candidate_id,
                "batch_index": 0,
                "evidence_scope_digest": first_scope.scope_key,
                "decision": "keep",
                "operations": [],
            },
        )
        self.director.decide_local_edit(
            self.run_id,
            {
                "proposal_id": "local:scope:0",
                "candidate_id": candidate.candidate_id,
                "batch_index": 0,
                "outcome": LocalEditOutcome.KEPT.value,
                "active_revision_id": revision.revision_id,
            },
        )
        self.director.advance_trajectory_revision(
            self.run_id,
            candidate.candidate_id,
            0,
            revision.revision_id,
            RevisionAdvanceReason.KEPT,
        )

        second_batch = adaptation.batches[1]
        self.director.start_formal_batch(
            self.run_id,
            candidate.candidate_id,
            revision.revision_id,
            1,
        )
        second_scope = EvaluationScope(
            run_id=self.run_id,
            generation=0,
            candidate_id=candidate.candidate_id,
            candidate_revision_id=revision.revision_id,
            phase=EvaluationPhase.FORMAL_BATCH,
            cohort_digest=second_batch.cohort.cohort_digest,
            origin_count=second_batch.origin_count,
            batch_index=1,
        )
        second = self.director.prepare_evaluation_sample_checkpoint(
            self.run_id,
            generation=0,
            proposal_id=candidate.proposal_id,
            candidate_id=candidate.candidate_id,
            scope=second_scope,
            checkpoint=self._checkpoint(second_scope),
        )

        self.assertNotEqual(second["revision"], prepared["revision"])
        self.assertFalse(second["resumed"])


class FrozenOriginSelectionTests(unittest.TestCase):
    def test_planner_origins_are_complete_for_maximum_ridge_history(self) -> None:
        timestamps = tuple(range(180))
        targets = (
            "air_temperature",
            "relative_humidity",
            "co2_concentration",
        )
        series = DatasetSeries(
            schema="ecologyrsi-dsh.dataset-series/1",
            dataset_id="dataset:planner-evaluator-contract",
            domain_id="greenhouse",
            episode_id="episode:planner-evaluator-contract",
            digest="d" * 64,
            timestamps=timestamps,
            values={
                target: tuple(float(index) for index in timestamps)
                for target in targets
            },
            partitions={
                "training_fit": IndexRange(0, 70),
                "training_feedback": IndexRange(80, 180),
                "model_selection": IndexRange(80, 180),
                "development": IndexRange(180, 180),
            },
            features={
                target: FeatureSpec(target, target, "environment", "unit")
                for target in targets
            },
            split_manifest_digest_sha256="s" * 64,
        )
        schedule = OptimizationSchedule.from_dict(
            {
                **OptimizationSchedule.default().to_dict(),
                "formal_origin_count_per_finalist": 20,
                "local_batch_origin_count": 10,
            }
        )
        planned = plan_run_adaptation_cohort(
            series, schedule=schedule, seed=3
        ).batches[0].cohort
        generated = fit_predict_exogenous_ridge(
            series,
            targets=targets,
            horizons=(1, 6, 24),
            config=ExogenousRidgeConfig(
                history_steps=3,
                ridge_alpha=0.1,
                residual_scale=1.0,
            ),
            evaluation_history_steps=MAX_EXOGENOUS_RIDGE_HISTORY_STEPS,
            defer_prediction_partitions=("training_feedback",),
        )
        feedback_rows = [
            row
            for row in generated["prediction_rows"]
            if row["partition"] == "training_feedback"
        ]

        selected, evidence = _select_planned_evaluation_cohort(
            feedback_rows, planned
        )

        self.assertEqual(planned.origins[0].origin_index, 91)
        self.assertEqual(len(selected), planned.origin_count * 9)
        self.assertEqual(evidence["prediction_cells_per_origin"], 9)

    def test_evaluator_executes_exact_frozen_origins_in_planner_order(self) -> None:
        schedule = OptimizationSchedule.from_dict(
            {
                **OptimizationSchedule.default().to_dict(),
                "formal_origin_count_per_finalist": 100,
                "local_batch_origin_count": 10,
            }
        )
        dataset = SimpleNamespace(
            dataset_id="dataset:scope-selection",
            episode_id="episode:scope-selection",
            timestamps=tuple(range(1000)),
            partitions={"model_selection": IndexRange(0, 1000)},
        )
        planned = plan_run_adaptation_cohort(
            dataset, schedule=schedule, seed=3
        ).batches[0].cohort
        tasks = tuple(
            (target, horizon)
            for target in ("air_temperature", "co2_concentration", "relative_humidity")
            for horizon in (1, 6, 24)
        )
        rows = [
            {
                "partition": "training_feedback",
                "target": target,
                "horizon_hours": horizon,
                "origin_timestamp": origin.origin_timestamp,
                "target_timestamp": origin.origin_timestamp + horizon,
                "timestamp": origin.origin_timestamp + horizon,
            }
            for origin in reversed(planned.origins)
            for target, horizon in reversed(tasks)
        ]
        rows.extend(
            {
                "partition": "training_feedback",
                "target": target,
                "horizon_hours": horizon,
                "origin_timestamp": 999_999,
                "target_timestamp": 999_999 + horizon,
                "timestamp": 999_999 + horizon,
            }
            for target, horizon in tasks
        )

        selected, evidence = _select_planned_evaluation_cohort(rows, planned)

        self.assertEqual(len(selected), planned.origin_count * 9)
        self.assertEqual(evidence["cohort_digest"], planned.cohort_digest)
        self.assertEqual(evidence["selected_origin_count"], planned.origin_count)
        self.assertEqual(evidence["samples_per_update"], planned.origin_count)
        self.assertEqual(evidence["prediction_cell_budget"], len(selected))
        self.assertEqual(evidence["prediction_cells_per_origin"], 9)
        self.assertEqual(evidence["population_origin_count"], planned.origin_count + 1)
        self.assertEqual(evidence["population_count"], len(rows))
        self.assertEqual(evidence["selected_count"], len(selected))
        self.assertEqual(evidence["deferred_count"], 9)
        self.assertEqual(len(evidence["population_digest"]), 64)
        self.assertIsInstance(evidence["window_offset"], int)
        self.assertIsInstance(evidence["window_cycle"], int)
        self.assertIsInstance(evidence["window_wraps"], bool)
        self.assertEqual(
            tuple(dict.fromkeys(row["origin_timestamp"] for row in selected)),
            tuple(origin.origin_timestamp for origin in planned.origins),
        )
        self.assertNotIn(999_999, {row["origin_timestamp"] for row in selected})

    def test_wrapped_origins_materialize_distinct_sample_occurrences(self) -> None:
        schedule = OptimizationSchedule.from_dict(
            {
                **OptimizationSchedule.default().to_dict(),
                "formal_origin_count_per_finalist": 20,
                "local_batch_origin_count": 10,
            }
        )
        dataset = SimpleNamespace(
            dataset_id="dataset:wrapped-selection",
            episode_id="episode:wrapped-selection",
            timestamps=tuple(range(50)),
            partitions={"model_selection": IndexRange(0, 50)},
        )
        planned = plan_run_adaptation_cohort(
            dataset, schedule=schedule, seed=3
        ).cohort
        self.assertGreater(len({origin.origin_id for origin in planned.origins}), 0)
        self.assertGreater(
            len(planned.origins), len({origin.origin_id for origin in planned.origins})
        )
        tasks = tuple(
            (target, horizon)
            for target in ("air_temperature", "co2_concentration", "relative_humidity")
            for horizon in (1, 6, 24)
        )
        rows = [
            {
                "partition": "training_feedback",
                "target": target,
                "horizon_hours": horizon,
                "origin_timestamp": origin.origin_timestamp,
                "target_timestamp": origin.origin_timestamp + horizon,
                "timestamp": origin.origin_timestamp + horizon,
            }
            for origin in {item.origin_timestamp: item for item in planned.origins}.values()
            for target, horizon in tasks
        ]

        selected, _evidence = _select_planned_evaluation_cohort(rows, planned)

        self.assertEqual(len(selected), planned.origin_count * len(tasks))
        self.assertEqual(
            len({row["sample_index"] for row in selected}), len(selected)
        )
        self.assertEqual(
            {row["cohort_origin_occurrence"] for row in selected}, {0, 1}
        )

    def test_evaluator_rejects_partial_origin_vector(self) -> None:
        schedule = OptimizationSchedule.from_dict(
            {
                **OptimizationSchedule.default().to_dict(),
                "formal_origin_count_per_finalist": 100,
                "local_batch_origin_count": 10,
            }
        )
        dataset = SimpleNamespace(
            dataset_id="dataset:partial-selection",
            episode_id="episode:partial-selection",
            timestamps=tuple(range(1000)),
            partitions={"model_selection": IndexRange(0, 1000)},
        )
        planned = plan_run_adaptation_cohort(
            dataset, schedule=schedule, seed=3
        ).batches[0].cohort
        rows = [
            {
                "partition": "training_feedback",
                "target": "air_temperature",
                "horizon_hours": 1,
                "origin_timestamp": origin.origin_timestamp,
                "target_timestamp": origin.origin_timestamp + 1,
            }
            for origin in planned.origins
        ]

        with self.assertRaisesRegex(ValueError, "complete prediction vector"):
            _select_planned_evaluation_cohort(rows, planned)
