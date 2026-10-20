"""Current joint capacity gates are value-blind and precede model work."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.evaluators.epoch_cohorts import estimate_epoch_capacity, CohortCapacityError
from ecologyrsi_dsh.evaluators.greenhouse_prediction import COHORT_HISTORY_HOURS
from ecologyrsi_dsh.evolution.evidence_capacity import (
    _cohort_evidence, require_guarded_cohort_evidence_capacity,
)
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from tests.test_epoch_cohort_planning import dataset_fixture
from tests.native_http_fixture import NativeBindingFixture


class GuardedCohortCapacityTests(unittest.TestCase):
    def test_fixed_schedule_cannot_shrink_scientific_blocks(self):
        schedule = OptimizationSchedule.for_new_run()
        for field, value in (("local_batch_origin_count", 10),
                             ("formal_origin_count_per_finalist", 20),
                             ("selection_holdout_origin_count", 10)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(schedule, **{field: value})

    def test_report_is_value_blind_digest_bound_and_checks_each_generation(self):
        arguments = dict(schedule=OptimizationSchedule.for_new_run(), planned_generations=3,
                         seed=7, history_steps=COHORT_HISTORY_HOURS)
        first = require_guarded_cohort_evidence_capacity(
            dataset=dataset_fixture(2000, timestamp_gap_at=450), **arguments)
        changed = require_guarded_cohort_evidence_capacity(
            dataset=dataset_fixture(2000, timestamp_gap_at=450, changed_labels=True), **arguments)
        self.assertEqual(first, changed)
        self.assertEqual(len(first["selection_holdouts"]), 3)
        self.assertEqual({row["generation"] for row in first["formal_batches"]}, {0, 1, 2})
        self.assertTrue(all(row["day_block_count"] >= row["minimum_day_blocks"]
                            for row in first["formal_batches"] + first["selection_holdouts"]))
        self.assertEqual(first["report_digest"], digest({k: v for k, v in first.items() if k != "report_digest"}))

    def test_repeated_origin_occurrences_cannot_inflate_day_evidence(self):
        cohort = SimpleNamespace(cohort_digest="a" * 64, origin_count=50,
            origins=[SimpleNamespace(origin_timestamp=24)] * 50)
        evidence = _cohort_evidence(cohort, minimum=3)
        self.assertEqual(evidence["day_block_count"], 1)
        self.assertFalse(evidence["sufficient"])
        with self.assertRaises(CohortCapacityError):
            require_guarded_cohort_evidence_capacity(dataset=dataset_fixture(100),
                schedule=OptimizationSchedule.for_new_run(), planned_generations=1, seed=7,
                history_steps=COHORT_HISTORY_HOURS)


class GuardedCohortAPIAdmissionTests(NativeBindingFixture):
    def test_joint_capacity_counts_base_and_conditional_pairs_without_duplicate_screening(self):
        schedule = OptimizationSchedule.for_new_run()
        with patch.object(self.server.datasets, "selection_view", return_value=dataset_fixture(2000)), \
             patch.object(self.server.datasets, "partition_summary", return_value={}):
            status, report = self._post_path("/evolution-capacity", {
                "dataset_id": "agc_cucumber_2018", "episode_id": "agc_cucumber_2018:TeamA",
                "optimization_schedule": schedule.to_dict(),
                "planned_generations": 4,
            })
        self.assertEqual(status, 200, report)
        self.assertTrue(report["sufficient"], report)
        self.assertEqual(report["required_unique_origins"], 340)
        self.assertEqual(report["candidate_origin_executions_per_generation"], 300)
        self.assertEqual(report["candidate_origin_executions_for_run"], 1200)
        self.assertEqual(report["scoring_cells_for_run"], 10800)
        self.assertEqual(report["execution_plan"], schedule.execution_plan(4, cells_per_origin=9))
        self.assertEqual(report["execution_plan"]["run_budget"]["base_candidate_origins"], 800)
        self.assertEqual(report["execution_plan"]["run_budget"]["recheck_candidate_origins"], 400)
        self.assertEqual(report["holdout_inference_replicas"], 1)

    def test_joint_api_freezes_schedule_mutation_bounds_and_rejects_retired_shapes(self):
        state, _ = self.create("joint-fixed-schedule")
        metadata = state.task_manifest.metadata
        self.assertEqual(metadata["optimization_schedule"], OptimizationSchedule.for_new_run().to_dict())
        self.assertEqual(metadata["mutation_policy"]["candidate_maximum_operations"], 4)
        self.assertEqual(metadata["mutation_policy"]["local_maximum_operations"], 2)
        self.assertEqual(metadata["mutation_policy"]["mutation_operator_id"], "joint-plugin-mutation@1")
        for schema in ("ecologyrsi-dsh.top2-adaptive-epoch-schedule/3",
                       "ecologyrsi-dsh.quick-adaptive-epoch-schedule/1",
                       "ecologyrsi-dsh.evidence-guided-epoch-schedule/1"):
            invalid = {**OptimizationSchedule.for_new_run().to_dict(), "schema_version": schema}
            with patch("ecologyrsi_dsh.api.handler.run_preflight") as paid:
                for route in ("/model-preflight", "/runs"):
                    status, result = self._post_path(route, self.body(schema, optimization_schedule=invalid))
                    self.assertEqual(status, 400, result)
                paid.assert_not_called()
        for index, budget in enumerate((
            {"max_generations": 51, "max_candidates": 204, "candidates_per_generation": 4},
            {"max_generations": 1, "max_candidates": 4, "candidates_per_generation": 3},
            {"max_generations": 1, "max_candidates": 257, "candidates_per_generation": 4},
        )):
            with patch("ecologyrsi_dsh.api.handler.run_preflight") as paid:
                for route in ("/model-preflight", "/runs"):
                    status, result = self._post_path(route, self.body(f"bad-budget-{index}", budget=budget))
                    self.assertEqual(status, 400, result)
                paid.assert_not_called()
        self.assertEqual(len(self.native.created), 1)

    def test_preview_and_preflight_agree_at_native_history_capacity_boundary(self):
        fixture = dataset_fixture(1056)
        base = self.server.datasets.selection_view("agc_cucumber_2018", "agc_cucumber_2018:TeamA")
        view = replace(base, timestamps=fixture.timestamps, partitions=fixture.partitions,
                       values={name: tuple(values[:1056]) for name, values in base.values.items()})
        schedule = OptimizationSchedule.for_new_run()
        expected = estimate_epoch_capacity(view, schedule=schedule, planned_generations=5, seed=0,
                                          history_steps=COHORT_HISTORY_HOURS)
        self.assertFalse(expected.sufficient)
        self.assertEqual(expected.max_feasible_generations, 4)
        preview = {"dataset_id": view.dataset_id, "episode_id": view.episode_id,
                   "optimization_schedule": schedule.to_dict(), "planned_generations": 5}
        body = self.body("history-capacity-boundary", seed=0,
                         budget={"max_generations": 5, "max_candidates": 20, "candidates_per_generation": 4})
        with patch.object(self.server.datasets, "selection_view", return_value=view), \
             patch.object(self.server.datasets, "partition_summary", return_value={}), \
             patch("ecologyrsi_dsh.api.handler.run_preflight") as paid:
            status, report = self._post_path("/evolution-capacity", preview)
            self.assertEqual(status, 200, report)
            self.assertFalse(report["sufficient"])
            self.assertEqual(report["origin_history_alignment_hours"], COHORT_HISTORY_HOURS)
            self.assertEqual(report["available_eligible_origins"], expected.available_eligible_origins)
            self.assertEqual(report["max_feasible_generations"], 4)
            for route in ("/model-preflight", "/runs"):
                status, rejected = self._post_path(route, body)
                self.assertEqual(status, 400, rejected)
                self.assertEqual(rejected["error_code"], "insufficient_cohort_capacity", rejected)
            paid.assert_not_called()
            status, report = self._post_path("/evolution-capacity", {**preview, "planned_generations": 4})
            self.assertEqual(status, 200, report)
            self.assertTrue(report["sufficient"])
            evidence = require_guarded_cohort_evidence_capacity(
                dataset=view, schedule=schedule, planned_generations=4, seed=0,
                profile=self.server.evaluators.fitness_profile("greenhouse_multihorizon_time_forward@4", view.dataset_id),
                history_steps=COHORT_HISTORY_HOURS)
            self.assertEqual(report["guarded_evidence"], evidence)
            status, checked = self._post_path("/model-preflight", {**body, "check_only": True,
                "budget": {"max_generations": 4, "max_candidates": 16, "candidates_per_generation": 4}})
            self.assertEqual(status, 200, checked)
            self.assertTrue(checked["passed"], checked)
            self.assertFalse(checked["pending"], checked)
            self.assertEqual(len(checked["receipts"]), 4)
            paid.assert_not_called()
        self.assertEqual(self.native.created, [])
