from dataclasses import replace
from types import SimpleNamespace
import unittest
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.trajectory import FormalBatchArm
from ecologyrsi_dsh.data.greenhouse import FeatureSpec
from ecologyrsi_dsh.data.registry import DatasetSeries
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.evaluators.epoch_cohorts import plan_run_adaptation_cohort
from ecologyrsi_dsh.evaluators.greenhouse_prediction import (ExogenousRidgeConfig,
    MAX_EXOGENOUS_RIDGE_HISTORY_STEPS, fit_predict_exogenous_ridge)
from ecologyrsi_dsh.evaluators.pipeline import _select_planned_evaluation_cohort
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from tests import test_duplicate_race_checkpoint as fixtures


class ScopedSampleExecutionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.DuplicateRaceCheckpointTests('test_both_formal_arms_use_all_callbacks_and_replay_preserves_duplicate')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_checkpoints_isolate_paired_revisions_and_resume_exactly(self):
        f=self.fixture
        scopes=[f.scope(arm) for arm in FormalBatchArm]
        opened=[]
        for scope in scopes:
            session=f.session(scope)
            item=session.prepare_checkpoint(f.checkpoint(scope))
            opened.append(item)
            self.assertFalse(item['resumed'])
        self.assertNotEqual(opened[0]['revision'],opened[1]['revision'])
        for scope,item in zip(scopes,opened):
            recovered=f.session(scope).prepare_checkpoint(f.checkpoint(scope))
            self.assertTrue(recovered['resumed'])
            self.assertEqual(recovered['revision'],item['revision'])

    def test_checkpoint_rejects_unfrozen_batch_revision_and_cohort(self):
        f=self.fixture; scope=f.scope(FormalBatchArm.CHAMPION)
        for bad in (replace(scope,batch_index=0), replace(scope,candidate_revision_id='unknown'),
                    replace(scope,cohort_digest=digest('other')),replace(scope,origin_count=24)):
            with self.subTest(scope=bad.to_dict()),self.assertRaises((ValueError, KeyError)):
                f.session(bad).prepare_checkpoint(f.checkpoint(bad))
        self.assertFalse(any(e.kind=='EvaluationSampleResultsStarted' for e in f.director.state(f.run_id).events))

    def test_unsealed_partial_results_are_not_other_arm_results(self):
        f=self.fixture; parent=f.scope(FormalBatchArm.CHAMPION); child=f.scope(FormalBatchArm.CHALLENGER)
        parent_session=f.session(parent); parent_session.prepare_checkpoint(f.checkpoint(parent))
        rows=[dict(sample_id=f'p:{h}',sample_index=i+1,target='air_temperature',horizon_hours=h,
              origin_timestamp=1,target_timestamp=1+h,observed=0.,predicted=0.,baseline=1.)
              for i,h in enumerate((1,6,24))]
        parent_session.record_results(rows)
        self.assertEqual(f.session(child).prepare_checkpoint(f.checkpoint(child))['rows'],())
        self.assertEqual(len(f.session(parent).prepare_checkpoint(f.checkpoint(parent))['rows']),3)


class FrozenOriginSelectionTests(unittest.TestCase):
    def test_planner_origins_are_complete_for_maximum_ridge_history(self) -> None:
        timestamps = tuple(range(1800))
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
                "training_feedback": IndexRange(80, 1800),
                "model_selection": IndexRange(80, 1800),
                "development": IndexRange(1800, 1800),
            },
            features={
                target: FeatureSpec(target, target, "environment", "unit")
                for target in targets
            },
            split_manifest_digest_sha256="s" * 64,
        )
        schedule = OptimizationSchedule.for_new_run()
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
        schedule = OptimizationSchedule.for_new_run()
        dataset = SimpleNamespace(
            dataset_id="dataset:scope-selection",
            episode_id="episode:scope-selection",
            timestamps=tuple(range(3000)),
            partitions={"model_selection": IndexRange(0, 3000)},
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

    def test_insufficient_origin_population_is_rejected_without_wrapping(self):
        dataset=SimpleNamespace(dataset_id='dataset:short',episode_id='episode:short',
            timestamps=tuple(range(50)),partitions={'model_selection':IndexRange(0,50)})
        with self.assertRaises(ValueError):
            plan_run_adaptation_cohort(dataset,schedule=OptimizationSchedule.for_new_run(),seed=3)

    def test_evaluator_rejects_partial_origin_vector(self) -> None:
        schedule = OptimizationSchedule.for_new_run()
        dataset = SimpleNamespace(
            dataset_id="dataset:partial-selection",
            episode_id="episode:partial-selection",
            timestamps=tuple(range(3000)),
            partitions={"model_selection": IndexRange(0, 3000)},
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
