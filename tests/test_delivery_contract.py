"""Current native HTTP delivery and durable command-receipt contracts."""
from __future__ import annotations

import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from ecologyrsi_dsh.api.handler import EvolutionHTTPServer
from tests import native_http_fixture, test_runtime_integration
from tests.http_poll_fixture import wait_for_command


class DeliveryContractTests(unittest.TestCase):
    def setUp(self):
        native_http_fixture.NativeBindingFixture.setUp(self)

    tearDown = native_http_fixture.NativeBindingFixture.tearDown
    def request(self, path, method="GET", body=None):
        status, result = test_runtime_integration.RuntimeIntegrationTests.request(self, path, method, body)
        return wait_for_command(self, result) if status == 202 else (status, result)

    def create(self, key="create-key", **overrides):
        body = native_http_fixture.NativeBindingFixture.body(key, **overrides)
        status, result = self.request("/runs", "POST", body)
        self.assertEqual(status, 201, result)
        return result["projection"]["run_id"], result, body

    def test_health_reports_current_scientific_boundary(self):
        status, result = self.request("/health")
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        self.assertEqual(result["evaluation_partition"], "training_feedback")
        self.assertEqual(result["scientific_scope"], "historical_replay_prediction_non_causal")
        self.assertEqual(result["package_version"], "0.11.0")

    def test_health_is_available_without_model_directory_or_ledger_access(self):
        with patch.object(self.server.model_directory, "catalog", side_effect=AssertionError("directory read")), self.server.ledger._lock:
            status, result = self.request("/health")
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])

    def test_create_and_control_receipts_survive_restart(self):
        run_id, created, body = self.create("restart-create")
        self.assertEqual(self.request("/runs", "POST", body), (200, created))
        path = "/runs/" + run_id.replace(":", "%3A") + "/control"
        cancel = {"action": "cancel", "idempotency_key": "restart-cancel"}
        status, cancelled = self.request(path, "POST", cancel)
        self.assertEqual(status, 200, cancelled)
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.server = EvolutionHTTPServer(("127.0.0.1", 0), Path(self.directory.name) / "events.sqlite3")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/api"
        self.assertEqual(self.request(path, "POST", cancel), (200, cancelled))
        self.assertEqual(self.request("/runs", "POST", body), (200, created))

    def test_manual_advance_route_is_removed(self):
        run_id, _, _ = self.create()
        status, _ = self.request("/runs/" + run_id.replace(":", "%3A") + "/advance", "POST", {"steps": 1})
        self.assertEqual(status, 404)
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_unknown_dataset_does_not_create_run(self):
        status, result = self.request("/runs", "POST", native_http_fixture.NativeBindingFixture.body("unknown", dataset_id="field-observations-2026"))
        self.assertEqual(status, 404, result)
        self.assertEqual(self.server.ledger.run_ids(), ())

    def test_invalid_budget_and_advance_are_rejected_before_write(self):
        for i, extra in enumerate((
            {"auto_advance": 1}, {"auto_advance": -1}, {"auto_advance": 1.5},
            {"auto_advance": None}, {"budget": {"max_generations": "bad"}},
            {"budget": {"max_generations": 1, "candidates_per_generation": 2}},
        )):
            status, result = self.request("/runs", "POST", native_http_fixture.NativeBindingFixture.body(f"bad-{i}", **extra))
            self.assertEqual(status, 400, result)
        self.assertEqual(self.server.ledger.run_ids(), ())
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_invalid_control_does_not_leave_pending_receipt(self):
        run_id, _, _ = self.create()
        path = "/runs/" + run_id.replace(":", "%3A") + "/control"
        body = {"action": "rewind", "idempotency_key": "bad-action"}
        for _ in range(2):
            status, result = self.request(path, "POST", body)
            self.assertEqual(status, 400, result)
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_completed_control_receipt_replays_when_terminal(self):
        run_id, _, _ = self.create()
        path = "/runs/" + run_id.replace(":", "%3A") + "/control"
        body = {"action": "cancel", "idempotency_key": "terminal-cancel"}
        status, result = self.request(path, "POST", body)
        self.assertEqual(status, 200, result)
        self.assertEqual(result["projection"]["status"], "cancelled")
        self.assertEqual(self.request(path, "POST", body), (200, result))
