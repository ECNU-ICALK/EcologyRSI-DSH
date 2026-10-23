"""Research must see the same defaults that a real pipeline switch installs."""
from copy import deepcopy
from dataclasses import replace
import unittest

from ecologyrsi_dsh.core.prediction_policy import RUNTIME_PREDICTION_POLICY, prediction_usage
from ecologyrsi_dsh.evolution.genome import EcologyEvolutionPluginGenome, apply_genome_mutation, materialize_seed_genome
from ecologyrsi_dsh.evolution.strategies import _mutation_contract_catalog, _predictor_selection_options
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from tests.test_autonomous_search_reflection_cycle import _task
from tests.test_evolution_genome import _initialization, _mutation_context


class PredictorSelectionContextTests(unittest.TestCase):
    def setUp(self):
        self.registry = current_program_registry()
        self.ids = ["greenhouse-baseline-aligned-ridge@1", "greenhouse-horizon-targetwise-ridge@1"]
        catalog = [{"id": pid, "parameter_schemas": {
            name: {k: v for k, v in contract.items() if k != "default"}
            for name, contract in self.registry.program("predictors", pid)["parameters"].items()}}
            for pid in self.ids]
        base = _task()
        self.task = replace(base, metadata={**base.metadata,
            "prediction_selection_policy": RUNTIME_PREDICTION_POLICY,
            "runtime_component_catalog": {"prediction_models": catalog}})
        self.parent = materialize_seed_genome(
            self.registry.seed_template("greenhouse-baseline-aligned-cellwise-default@1"),
            _initialization(task_manifest_digest=self.task.digest))

    def test_delivered_defaults_equal_real_mutation_for_zero_and_nonzero_models(self):
        task_before, parent_before = deepcopy(self.task.to_dict()), self.parent.to_dict()
        catalog = _mutation_contract_catalog(self.task, self.parent)
        options = catalog["prediction_selection"]["predictor_options"]
        for option in options:
            with self.subTest(predictor=option["id"]):
                child = apply_genome_mutation(self.parent,
                    {"schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": [
                        {"op": "select_registered_pipeline", "predictor_id": option["id"]}]},
                    _mutation_context(self.parent), self.registry)
                installed = child.scientific_program["parameter_overrides"]
                shown = option["on_selection"]
                self.assertEqual(shown["installed_parameters"], installed)
                self.assertEqual(shown["predictor_ref"], child.scientific_program["predictor_ref"])
                self.assertEqual(shown["registry_catalog_digest"], self.parent.to_dict()["runtime_binding"]["registry_catalog_digest"])
                self.assertEqual(shown["default_tool_usage"], prediction_usage(option["id"], installed))
        self.assertEqual(options[0]["on_selection"]["default_tool_usage"]["mode"], "baseline_only")
        self.assertTrue(any(value > 0 for key, value in
            options[1]["on_selection"]["installed_parameters"].items() if "residual_scale" in key))
        self.assertEqual(self.task.to_dict(), task_before)
        self.assertEqual(self.parent.to_dict(), parent_before)

    def test_defaults_are_not_borrowed_from_a_different_frozen_registry(self):
        value = self.parent.to_dict()
        value["runtime_binding"]["registry_catalog_digest"] = "0" * 64
        value.pop("genome_id")
        value.pop("genome_digest")
        altered = EcologyEvolutionPluginGenome.from_dict(value)
        with self.assertRaisesRegex(ValueError, "registry catalog binding mismatch"):
            _predictor_selection_options(self.task, altered)

    def test_unknown_catalog_entry_does_not_get_invented_defaults(self):
        task = replace(self.task, metadata={**self.task.metadata,
            "runtime_component_catalog": {"prediction_models": [
                {"id": "unknown-model@1", "parameter_schemas": {"scale": {"minimum": 0}}}]}})
        self.assertNotIn("on_selection", _predictor_selection_options(task, self.parent)[0])


if __name__ == "__main__":
    unittest.main()
