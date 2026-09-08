"""Dataset task definitions drive evaluation, admission and browser bindings."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.data.adapters import (
    CUCUMBER_2018, TOMATO_2019, DATASET_ADAPTERS, dataset_adapter,
)
from ecologyrsi_dsh.data.greenhouse import feature_specs
from ecologyrsi_dsh.data.registry import DatasetRegistry
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.evaluators.epoch_cohorts import plan_run_adaptation_cohort
from ecologyrsi_dsh.evaluators.registry import EvaluatorRegistry, EXOGENOUS_RIDGE_MODEL_ID
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.core.models import TaskManifest, digest
from ecologyrsi_dsh.core.immutable import thaw_json
from tests.test_evaluation import _cohort_series, _DatasetStub, _task, _candidate_and_proposal


class DatasetTaskAdapterTests(unittest.TestCase):
    def test_independent_contracts_preserve_different_label_units_and_semantics(self):
        self.assertNotEqual(CUCUMBER_2018.contract()["contract_digest"], TOMATO_2019.contract()["contract_digest"])
        for adapter, unit in ((CUCUMBER_2018, "kg_m2_cumulative"), (TOMATO_2019, "kg_m2_harvest")):
            self.assertEqual(feature_specs(adapter.domain_id)["marketable_yield"].unit, unit)
            self.assertNotIn("marketable_yield", adapter.target_names)
            self.assertNotIn("fruit_tss", adapter.target_names)
            contract = adapter.contract()
            contract["targets"][0]["unit"] = "tampered"
            self.assertEqual(adapter.contract()["targets"][0]["unit"], "degC")

    def test_unregistered_datasets_and_invalid_grids_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "no prediction task adapter"):
            dataset_adapter("unknown")
        for changes in ({"targets": ()}, {"horizons_hours": (True,)},
                        {"horizons_hours": (1, 1)}, {"targets": CUCUMBER_2018.targets[:1]},
                        {"minimum_coverage": 1.1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(CUCUMBER_2018, **changes)

    def test_adapter_rejects_cross_dataset_and_unit_mismatch(self):
        source = _cohort_series()
        CUCUMBER_2018.validate_series(source)
        with self.assertRaisesRegex(ValueError, "dataset/domain"):
            TOMATO_2019.validate_series(source)
        features = {**source.features, "air_temperature": replace(source.features["air_temperature"], unit="K")}
        with self.assertRaisesRegex(ValueError, "unit mismatch"):
            CUCUMBER_2018.validate_series(replace(source, features=features))

    def test_catalog_admission_and_evaluator_identity_are_dataset_owned(self):
        datasets = DatasetRegistry()
        catalog = datasets.catalog()["datasets"]
        self.assertEqual({d["dataset_id"] for d in catalog if d["training_selectable"]}, set(DATASET_ADAPTERS))
        registry = EvaluatorRegistry(datasets)
        identities = []
        for adapter in DATASET_ADAPTERS.values():
            profile = registry.fitness_profile(adapter.evaluator_id, adapter.dataset_id)
            self.assertEqual(profile.expected_targets, adapter.target_names)
            self.assertEqual(profile.expected_horizons, adapter.horizons_hours)
            objective = registry.objective_profile(adapter.evaluator_id, adapter.dataset_id)
            self.assertEqual(objective["target_weights"], adapter.target_weights)
            identities.append(registry.evaluator_configuration_digest(adapter.evaluator_id, adapter.dataset_id))
        self.assertNotEqual(*identities)

    def test_selection_cannot_override_adapter_targets_or_horizons(self):
        registry = DatasetRegistry()
        for kwargs in ({"target_names": ("marketable_yield",)}, {"horizons": (12,)}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "match the dataset adapter"):
                registry.selection_view(CUCUMBER_2018.dataset_id, **kwargs)

    def test_evaluator_uses_adapter_grid_weights_bounds_and_thresholds(self):
        adapter = replace(CUCUMBER_2018, horizons_hours=(1, 6),
                          targets=(replace(CUCUMBER_2018.targets[0], weight=0.8),
                                   replace(CUCUMBER_2018.targets[1], weight=0.2)),
                          minimum_skill=0.2, minimum_coverage=0.95, selection_minimum_coverage=0.95)
        data = _task().to_dict()
        data["metadata"].update(evaluator_id=adapter.evaluator_id,
                                prediction_model_id=EXOGENOUS_RIDGE_MODEL_ID,
                                dataset_task=adapter.contract())
        candidate, proposal = _candidate_and_proposal()
        proposal = replace(proposal, changes={"history_steps": 3, "ridge_alpha": 0.1, "residual_scale": 0.5})
        registry = EvaluatorRegistry(_DatasetStub(_cohort_series()))
        with patch("ecologyrsi_dsh.data.adapters.DATASET_ADAPTERS", {adapter.dataset_id: adapter}):
            profile = registry.fitness_profile(adapter.evaluator_id, adapter.dataset_id)
            self.assertEqual(profile.prediction_cell_count, 4)
            result = registry.evaluate_scientific(TaskManifest.from_dict(data), candidate, proposal)
            metrics = result.evaluation.metrics
            self.assertEqual(metrics["dataset_task"], adapter.contract())
            self.assertEqual(metrics["objective_target_weights"], adapter.target_weights)
            self.assertEqual({(row["target"], row["horizon_hours"]) for row in metrics["targets"]},
                             {(t, h) for t in adapter.target_names for h in adapter.horizons_hours})
            self.assertEqual(result.evaluation.evaluator_digest,
                             registry.evaluator_configuration_digest(adapter.evaluator_id, adapter.dataset_id))
            gates = registry.objective_profile(adapter.evaluator_id, adapter.dataset_id)["hard_gates"]
            self.assertEqual(gates[0]["threshold"], 0.2)
            self.assertEqual(gates[-1]["threshold"], 0.95)

    def test_changed_contract_cannot_reuse_frozen_task(self):
        data = _task().to_dict()
        data["metadata"]["dataset_task"] = TOMATO_2019.contract()
        candidate, proposal = _candidate_and_proposal()
        with self.assertRaisesRegex(ValueError, "frozen dataset task"):
            EvaluatorRegistry(_DatasetStub(_cohort_series())).evaluate_scientific(
                TaskManifest.from_dict(data), candidate, proposal)

    def test_cohort_purge_uses_adapter_maximum_horizon(self):
        adapter = replace(CUCUMBER_2018, horizons_hours=(1, 6))
        dataset = SimpleNamespace(dataset_id=adapter.dataset_id, episode_id=adapter.dataset_id + ":test",
                                  timestamps=tuple(range(2000)), partitions={"model_selection": IndexRange(0, 2000)})
        with patch("ecologyrsi_dsh.evaluators.epoch_cohorts.DATASET_ADAPTERS", {adapter.dataset_id: adapter}):
            cohort = plan_run_adaptation_cohort(dataset, schedule=OptimizationSchedule.for_new_run(), seed=0)
        self.assertEqual(cohort.cohort.maximum_horizon, 6)
        self.assertTrue(all(o.maximum_target_timestamp - o.origin_timestamp == 6 for o in cohort.origins))
        for previous, following in zip(cohort.batches, cohort.batches[1:]):
            self.assertGreater(following.cohort.origins[0].origin_timestamp,
                               previous.cohort.origins[-1].maximum_target_timestamp)

    def test_promotion_keeps_adapter_practical_delta_and_rejects_contract_tampering(self):
        from tests.test_promotion import _evaluation
        from ecologyrsi_dsh.evaluators.objectives import OBJECTIVE_AGGREGATION_VERSION
        from ecologyrsi_dsh.evolution.promotion import assess_promotion_improvement

        adapter = replace(CUCUMBER_2018, selection_minimum_score_delta=0.02)
        def evaluation(name, score):
            result = _evaluation(name, score, version=OBJECTIVE_AGGREGATION_VERSION, block_scores=(score,) * 10)
            metrics = thaw_json(result.metrics)
            metrics["dataset_task"] = adapter.contract()
            evidence = metrics["promotion_block_evidence"]
            evidence["dataset_task"] = adapter.contract()
            evidence["evidence_digest"] = digest({k: v for k, v in evidence.items() if k != "evidence_digest"})
            return replace(result, metrics=metrics)
        old, new = evaluation("old", 0.4), evaluation("new", 0.41)
        with patch("ecologyrsi_dsh.data.adapters.DATASET_ADAPTERS", {adapter.dataset_id: adapter}):
            report = assess_promotion_improvement(new, old)
            self.assertTrue(report["comparable"])
            self.assertFalse(report["improved"])
            self.assertEqual(report["minimum_score_delta"], 0.02)
            changed = thaw_json(new.metrics)
            changed["dataset_task"]["selection_minimum_score_delta"] = 0.001
            self.assertFalse(assess_promotion_improvement(replace(new, metrics=changed), old)["comparable"])
