"""Real definitions, temporal availability and missing-resource semantics."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from ecologyrsi_dsh.data.adapters import adapter_from_definition, CUCUMBER_2018
from ecologyrsi_dsh.data.definitions import DATASET_DEFINITIONS, load_dataset_definitions
from ecologyrsi_dsh.data.greenhouse import GreenhouseDatasetAdapter, _derive_tomato_resource_totals


class DatasetDefinitionTests(unittest.TestCase):
    def test_shipped_files_drive_fields_units_and_scoring(self):
        for name, definition in load_dataset_definitions().items():
            adapter = adapter_from_definition(definition)
            self.assertEqual(adapter.dataset_id, name)
            self.assertEqual(adapter.target_names, tuple(t["name"] for t in definition["prediction_task"]["targets"]))
            self.assertIn("bias", adapter.diagnostic_metrics)
            loader = adapter.loader(Path("unused"))
            for feature in definition["features"]:
                self.assertEqual(loader.features[feature["name"]].to_dict(), feature)

    def test_changing_source_mapping_invalidates_frozen_definition(self):
        definition = deepcopy(DATASET_DEFINITIONS[CUCUMBER_2018.dataset_id])
        definition["features"][0]["aliases"] = ["different_column"]
        changed = adapter_from_definition(definition)
        self.assertNotEqual(changed.contract()["contract_digest"], CUCUMBER_2018.contract()["contract_digest"])

    def test_unsupported_scoring_metrics_are_rejected(self):
        for changes in ({"primary_metric": "invented"}, {"diagnostic_metrics": ("invented",)},
                        {"diagnostic_metrics": ("mae", "mae")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(CUCUMBER_2018, **changes)

    def test_bad_definition_identity_unit_and_low_frequency_target_fail_closed(self):
        original = deepcopy(DATASET_DEFINITIONS[CUCUMBER_2018.dataset_id])
        def check(definition):
            with tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                (root / "task.json").write_text(json.dumps(definition))
                (root / "catalog.json").write_text(json.dumps({"datasets": [{
                    "dataset_id": original["dataset_id"], "domain_id": original["domain_id"],
                    "adapter_id": original["prediction_task"]["adapter_id"],
                    "runnable": True, "adapter_definition": "task.json"}]}))
                with self.assertRaises(ValueError):
                    load_dataset_definitions(root / "catalog.json")
        broken = deepcopy(original)
        broken["dataset_id"] = "wrong"
        check(broken)
        broken = deepcopy(original)
        broken["features"][0]["unit"] = "K"
        check(broken)
        broken = deepcopy(original)
        broken["prediction_task"]["targets"] = [{"name": "marketable_yield", "unit": "kg_m2_cumulative"}]
        check(broken)

    def test_definition_cannot_escape_catalog_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "catalog.json"
            path.write_text(json.dumps({"datasets": [{"adapter_definition": "../task.json"}]}))
            with self.assertRaisesRegex(ValueError, "catalog directory"):
                load_dataset_definitions(path)

    def test_daily_date_label_is_sparse_and_available_next_day(self):
        loader = GreenhouseDatasetAdapter("tomato", "greenhouse_tomato_2019", ".")
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "Resources.csv"
            path.write_text("time,Irr\n2020-01-01,8\n43832,9\n")
            rows = loader._read_features(path, {"irrigation_water": ("Irr",)}, ("time",),
                                        source_interval="daily", availability_delay_hours=24)
        self.assertEqual(rows, {43832 * 24: {"irrigation_water": 8.0}, 43833 * 24: {"irrigation_water": 9.0}})

    def test_missing_resource_is_unknown_not_zero(self):
        values = {"electricity_peak_use": [2., None, 3.], "electricity_offpeak_use": [None, 1., 2.],
                  "irrigation_water": [8., 9., None], "drain_water": [None, 3., 2.]}
        _derive_tomato_resource_totals(values)
        self.assertEqual(values["electricity_use"], [None, None, 5.])
        self.assertEqual(values["water_use"], [None, 6., None])
