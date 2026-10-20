"""Public native API boundaries; retired direct gateway/demo paths stay absent."""
from __future__ import annotations

import json
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ecologyrsi_dsh.integrations.model_directory import ModelConnection, ModelDirectory
from tests import test_dsh_native_runtime as native


class RuntimeIntegrationTests(unittest.TestCase):
    setUp = native.DshNativeHTTPGateTests.setUp
    tearDown = native.DshNativeHTTPGateTests.tearDown

    def request(self, path, method="GET", body=None):
        request = Request(self.base + path, method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_catalog_exposes_current_models_without_credentials(self):
        self.server.model_directory = ModelDirectory((ModelConnection(
            model_id="provider/deepseek-v4.1-flash-1017", gateway_url="https://example.invalid/v1",
            model="deepseek-v4.1-flash-1017", token="private-test-credential"),))
        status, catalog = self.request("/catalog")
        self.assertEqual(status, 200)
        self.assertNotIn("private-test-credential", json.dumps(catalog))
        self.assertNotIn("generated-toy-series@1", {item["id"] for item in catalog["datasets"]})
        model = catalog["dsh_models"][0]
        self.assertEqual(set(model["roles"]), {"propose", "judge"})
        self.assertTrue(model["execution_available"])
        self.assertFalse(model["authentication_verified"])
        self.assertEqual(catalog["runtime_evaluator_id"], "greenhouse_multihorizon_time_forward@4")

    def test_samples_are_paginated_and_formal_partitions_are_closed(self):
        path = "/datasets/agc_cucumber_2018?partition=training_fit&limit=3"
        status, first = self.request(path)
        self.assertEqual(status, 200, first)
        self.assertEqual(len(first["rows"]), 3)
        status, second = self.request(path + "&offset=3")
        self.assertEqual(status, 200)
        self.assertNotEqual(first["rows"][0]["timestamp"], second["rows"][0]["timestamp"])
        for partition in ("validation", "final_test", "development", "external_holdout"):
            status, _ = self.request("/datasets/agc_cucumber_2018?partition=" + partition)
            self.assertEqual(status, 403, partition)

    def test_current_api_rejects_legacy_creation_before_persisting_run(self):
        self.server.dsh_native_runtime = native._FakeNativeRuntime()
        body = {"execution_protocol": "dsh_native_plugin_evolution@1", "dataset_id": "agc_cucumber_2018",
            "strategy_model_id": "dsh/strategy", "review_model_id": "dsh/review", "start": False,
            "rounds": 1, "knowledge_online_enabled": False}
        for extra in ({"optimization_protocol": "quick_experiment@1"},
                      {"domain_pack_id": "crop_soil_water"},
                      {"allow_host_fallback": True}, {"require_model_contract_preflight": False},
                      {"budget": {"max_generations": 1, "token_limit": 500}},
                      {"strategy_model_id": ""}):
            with self.subTest(extra=extra):
                status, payload = self.request("/runs", "POST", {**body, **extra})
                self.assertEqual(status, 400, payload)
        self.assertEqual(self.server.ledger.run_ids(), ())

    def test_retired_gateway_and_export_routes_are_absent(self):
        for path in ("/models/verify", "/verify-models", "/runs/import"):
            status, _ = self.request(path, "POST", {})
            self.assertEqual(status, 404, path)
