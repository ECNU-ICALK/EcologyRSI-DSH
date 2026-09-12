"""Guard admission uses selected day identities before launching model work."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.evaluators.greenhouse_prediction import BASELINE_ALIGNED_RIDGE_MODEL_ID
from ecologyrsi_dsh.evolution.evidence_capacity import (
    guarded_cohort_evidence_capacity, require_guarded_cohort_evidence_capacity,
)
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.integrations.dsh_native_runtime import DSH_NATIVE_EXECUTION_PROTOCOL
from tests.test_epoch_cohort_planning import dataset_fixture
from tests import test_runtime_integration as runtime_test_helpers


def schedule(batch):
    return replace(OptimizationSchedule.default(),
                   schema_version="ecologyrsi-dsh.top2-adaptive-epoch-schedule/3",
                   formal_origin_count_per_finalist=batch * 2, local_batch_origin_count=batch)


class GuardedCohortCapacityTests(unittest.TestCase):
    def test_small_batches_are_rejected_even_with_sufficient_total_origins(self):
        data = dataset_fixture(2000)
        report = guarded_cohort_evidence_capacity(data, schedule=schedule(10), planned_generations=1, seed=7)
        self.assertEqual([row["day_block_count"] for row in report["formal_batches"]], [2, 2])
        self.assertFalse(report["sufficient"])
        with self.assertRaisesRegex(ValueError, "实际 cohort 仅覆盖 2 个日块.*请增大局部批次"):
            require_guarded_cohort_evidence_capacity(dataset=data, schedule=schedule(10), planned_generations=1, seed=7)

    def test_report_is_value_blind_digest_bound_and_all_generations_are_checked(self):
        first = require_guarded_cohort_evidence_capacity(
            dataset=dataset_fixture(2000, timestamp_gap_at=450), schedule=schedule(72), planned_generations=3, seed=7)
        changed = require_guarded_cohort_evidence_capacity(
            dataset=dataset_fixture(2000, timestamp_gap_at=450, changed_labels=True),
            schedule=schedule(72), planned_generations=3, seed=7)
        self.assertEqual(first, changed)
        self.assertEqual(len(first["selection_holdouts"]), 3)
        self.assertTrue(all(row["day_block_count"] >= 3 for row in first["formal_batches"]))
        self.assertTrue(all(row["day_block_count"] >= 8 for row in first["selection_holdouts"]))
        self.assertEqual(first["report_digest"], digest({key: value for key, value in first.items() if key != "report_digest"}))

    def test_repeated_origins_cannot_inflate_holdout_day_evidence(self):
        legacy_schedule = replace(OptimizationSchedule.default(),
                                  formal_origin_count_per_finalist=144, local_batch_origin_count=72)
        # The historical cycling planner can repeat 169 origin occurrences
        # from a four-day source; only distinct real day buckets count.
        data = dataset_fixture(100)
        report = guarded_cohort_evidence_capacity(data, schedule=legacy_schedule, planned_generations=1, seed=7)
        self.assertEqual(report["selection_holdouts"][0]["origin_count"], 169)
        self.assertEqual(report["selection_holdouts"][0]["day_block_count"], 4)
        with self.assertRaisesRegex(ValueError, "选择留出 cohort 仅覆盖 4 个日块.*至少需要 8"):
            require_guarded_cohort_evidence_capacity(dataset=data, schedule=legacy_schedule, planned_generations=1, seed=7)


class GuardedCohortAPIAdmissionTests(unittest.TestCase):
    setUp = runtime_test_helpers.RuntimeIntegrationTests.setUp
    tearDown = runtime_test_helpers.RuntimeIntegrationTests.tearDown
    request = runtime_test_helpers.RuntimeIntegrationTests.request

    def test_capacity_preview_enforces_the_same_day_blocks_and_counts_inference_replicas(self):
        from ecologyrsi_dsh.evaluators.epoch_cohorts import estimate_epoch_capacity
        data = dataset_fixture(2000)
        with patch.object(self.server.datasets, "selection_view", return_value=data):
            status, rejected = self.request("/evolution-capacity", "POST", {
                "dataset_id": "agc_cucumber_2018", "episode_id": None,
                "optimization_schedule": schedule(10).to_dict(), "planned_generations": 1,
            })
            self.assertEqual(status, 200)
            self.assertFalse(rejected["sufficient"])
            self.assertIn("请增大局部批次", rejected["rejection_reason"])
            valid = schedule(72)
            status, report = self.request("/evolution-capacity", "POST", {
                "dataset_id": "agc_cucumber_2018", "episode_id": None,
                "optimization_schedule": valid.to_dict(), "planned_generations": 1,
            })
        self.assertEqual(status, 200)
        self.assertTrue(report["sufficient"])
        self.assertTrue(report["guarded_evidence"]["sufficient"])
        original = estimate_epoch_capacity(data, schedule=valid, planned_generations=1, seed=0)
        self.assertEqual(report["required_unique_origins"], original.required_unique_origins)
        self.assertEqual(report["holdout_inference_replicas"], 2)
        self.assertEqual(report["candidate_origin_executions_per_generation"],
                         original.candidate_origin_executions_per_generation + 3 * 169)
        self.assertEqual(report["scoring_cells_for_run"],
                         report["candidate_origin_executions_for_run"] * 9)
        self.assertEqual(report["planner_digest"], digest({key: report[key] for key in original.identity_dict()}))

    def test_small_guarded_run_rejected_before_any_model_stage_and_legacy_remains_accepted(self):
        description = {"descriptor": {"runnable": True, "display_name_zh": "测试温室序列",
                                      "domain_id": "greenhouse_test", "adapter_id": "greenhouse_test"},
                       "readiness": {"ready": True}}
        series = SimpleNamespace(digest="d" * 64, split_manifest_digest_sha256="b" * 64,
                                 episode_id="agc_cucumber_2018:test")
        view = SimpleNamespace(dataset_id="agc_cucumber_2018", episode_id=series.episode_id,
                               timestamps=tuple(range(2000)), partitions={"model_selection": IndexRange(0, 2000)},
                               data_protocol_digest="c" * 64, selection_view_digest="e" * 64)
        native = Mock()
        native.capabilities.return_value = {
            "schema_version": "ecology-agent-runtime-capabilities/1", "ready": True,
            "root_services": {"required": ["agents"], "missing": [], "declared": True},
            "presets": [{"preset_id": preset, "declared": True, "preset_mountable": True,
                         "tool_surface_verified": True, "route_resolvable": True,
                         "live_agent_service_ready": True, "first_call_verified": False}
                        for preset in ("ecology-coordinator-v5", "ecology-researcher-v12",
                                       "ecology-candidate-proposer-v4", "ecology-sample-planner-v9",
                                       "ecology-sample-critic-v5", "ecology-generation-judge-v8")],
            "live_agent_service_ready": True, "first_call_verified": False,
        }
        self.server.dsh_native_runtime = native
        body = {"execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                "domain_pack_id": "greenhouse_environment@1", "dataset_id": "agc_cucumber_2018",
                "strategy_model_id": "stub/strategy", "review_model_id": "stub/review",
                "autonomous_mode": True, "prediction_model_id": BASELINE_ALIGNED_RIDGE_MODEL_ID,
                "budget": {"max_generations": 1, "max_candidates": 4, "candidates_per_generation": 4},
                "optimization_schedule": schedule(10).to_dict(),
                "auto_advance": 0, "idempotency_key": "guard-small-blocks"}
        with (patch.object(self.server.datasets, "describe", return_value=description),
              patch.object(self.server.datasets, "series", return_value=series),
              patch.object(self.server.datasets, "selection_view", return_value=view)):
            with patch("ecologyrsi_dsh.api.handler.run_preflight") as paid_preflight:
                status, rejected = self.request("/model-preflight", "POST", body)
                self.assertEqual(status, 400, rejected)
                self.assertIn("请增大局部批次", rejected["error"])
                paid_preflight.assert_not_called()
            status, rejected = self.request("/runs", "POST", body)
            self.assertEqual(status, 400, rejected)
            self.assertIn("请增大局部批次", rejected["error"])
            native.run_stage.assert_not_called()
            body.update(prediction_model_id="greenhouse-exogenous-ridge@1", idempotency_key="legacy-small-blocks")
            status, created = self.request("/runs", "POST", body)
            self.assertEqual(status, 201, created)
            state = self.server.director.state(created["projection"]["run_id"])
            self.assertNotIn("guarded_cohort_evidence_capacity", state.task_manifest.metadata)
            body.update(prediction_model_id=BASELINE_ALIGNED_RIDGE_MODEL_ID,
                        optimization_schedule=OptimizationSchedule.for_comparison_run().to_dict(), idempotency_key="guard-valid-blocks")
            status, created = self.request("/runs", "POST", body)
            self.assertEqual(status, 201, created)
            state = self.server.director.state(created["projection"]["run_id"])
            report = state.task_manifest.metadata["guarded_cohort_evidence_capacity"]
            self.assertTrue(report["sufficient"])
            self.assertTrue(all(row["day_block_count"] >= 2 for row in report["formal_batches"]))
            self.assertTrue(all(row["day_block_count"] >= 8 for row in report["selection_holdouts"]))
            self.assertEqual(state.task_manifest.metadata["sample_budget_class"], "selection_eligible")
            self.assertEqual(state.task_manifest.metadata["minimum_selection_origin_samples_per_update"], 40)
            self.assertEqual(state.task_manifest.metadata["optimization_schedule"], OptimizationSchedule.for_comparison_run().to_dict())
            # Explicit budgets must survive normalization and receive the same
            # rejection before preflight as before actual run creation.
            invalid_budgets = (
                {"max_generations": 51, "max_candidates": 204, "candidates_per_generation": 4},
                {"max_generations": 1, "max_candidates": 4, "candidates_per_generation": 3},
                {"max_generations": 1, "max_candidates": 257, "candidates_per_generation": 4},
            )
            for index, invalid in enumerate(invalid_budgets):
                body.update(budget=invalid, idempotency_key=f"invalid-new-budget-{index}")
                with patch("ecologyrsi_dsh.api.handler.run_preflight") as preflight:
                    for route in ("/model-preflight", "/runs"):
                        status, rejected = self.request(route, "POST", body)
                        self.assertEqual(status, 400, rejected)
                    preflight.assert_not_called()
        native.run_stage.assert_not_called()
