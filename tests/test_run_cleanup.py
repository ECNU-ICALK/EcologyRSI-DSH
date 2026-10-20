from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from tests import native_http_fixture as native_fixture
from tests.http_poll_fixture import wait_for_command
from ecologyrsi_dsh.api import auto_progress as auto_progress_module
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import canonical_json, digest
from ecologyrsi_dsh.api.handler import EvolutionHTTPServer, EvolutionRequestHandler


class RunCleanupHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        native_fixture.NativeBindingFixture.setUp(self)
        self.base = self.base.removesuffix("/api")

    tearDown = native_fixture.NativeBindingFixture.tearDown

    def request(
        self, path: str, method: str = "GET", body: dict | None = None
    ) -> tuple[int, dict]:
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self.base + path,
            data=encoded,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=3) as response:
                status, payload = response.status, json.loads(response.read())
            if status == 202:
                return wait_for_command(self, payload)
            return status, payload
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def create(self, key: str) -> tuple[str, dict]:
        body = native_fixture.NativeBindingFixture.body(key)
        status, payload = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 201, payload)
        return str(payload["projection"]["run_id"]), body

    def test_archive_is_terminal_only_hidden_by_default_and_reversible(self) -> None:
        run_id, _body = self.create("archive-lifecycle")
        run_path = "/api/runs/" + quote(run_id, safe="")

        status, rejected = self.request(run_path + "/archive", "POST", {})
        self.assertEqual(status, 400)
        self.assertIn("终态", rejected["error"])
        status, rejected = self.request(
            run_path, "DELETE", {"confirm_run_id": run_id}
        )
        self.assertEqual(status, 400)
        self.assertIn("终态", rejected["error"])

        status, _cancelled = self.request(
            run_path + "/control", "POST", {"action": "cancel"}
        )
        self.assertEqual(status, 200)
        status, archived = self.request(run_path + "/archive", "POST", {})
        self.assertEqual(status, 200)
        self.assertTrue(archived["projection"]["archived"])
        self.assertIsNotNone(archived["projection"]["archived_at"])

        status, visible = self.request("/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual(visible["runs"], [])
        self.assertEqual(visible["archived_count"], 1)
        status, history = self.request("/api/runs?include_archived=true")
        self.assertEqual(status, 200)
        self.assertEqual([item["run_id"] for item in history["runs"]], [run_id])
        self.assertTrue(history["runs"][0]["archived"])

        status, restored = self.request(run_path + "/restore", "POST", {})
        self.assertEqual(status, 200)
        self.assertFalse(restored["projection"]["archived"])
        status, visible = self.request("/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual([item["run_id"] for item in visible["runs"]], [run_id])

        status, summary = self.request("/api/runs?view=summary")
        self.assertEqual(status, 200)
        self.assertEqual(summary["view"], "summary")
        self.assertEqual([item["run_id"] for item in summary["runs"]], [run_id])
        self.assertEqual(
            summary["runs"][0]["schema_version"],
            "ecologyrsi-dsh.browser-run-summary/2",
        )
        self.assertNotIn("rounds", summary["runs"][0])
        self.assertNotIn("candidates", summary["runs"][0])
        self.assertNotIn("configuration", summary["runs"][0])
        self.assertNotIn("budget", summary["runs"][0])

    def test_permanent_delete_requires_archive_exact_confirmation_and_is_scoped(self) -> None:
        run_id, create_body = self.create("purge-target")
        other_run_id, _other_body = self.create("purge-other")
        run_path = "/api/runs/" + quote(run_id, safe="")
        status, _cancelled = self.request(
            run_path + "/control",
            "POST",
            {"action": "cancel", "idempotency_key": "purge-cancel"},
        )
        self.assertEqual(status, 200)

        create_receipt = self.server.ledger.command_receipt("create:purge-target")
        self.assertIsNotNone(create_receipt)
        self.assertEqual(create_receipt.resource_run_id, run_id)
        before_events = self.server.ledger.count(run_id)
        before_commands = self.server.ledger.command_count()

        status, rejected = self.request(
            run_path, "DELETE", {"confirm_run_id": run_id}
        )
        self.assertEqual(status, 400)
        self.assertIn("归档", rejected["error"])
        self.assertEqual(self.server.ledger.count(run_id), before_events)

        status, _archived = self.request(run_path + "/archive", "POST", {})
        self.assertEqual(status, 200)
        for invalid_body in ({}, {"confirm_run_id": "wrong"}, {"confirm_run_id": run_id, "force": True}):
            status, rejected = self.request(run_path, "DELETE", invalid_body)
            self.assertEqual(status, 400)
            self.assertIn("confirm_run_id", rejected["error"])
        self.assertEqual(self.server.ledger.count(run_id), before_events)

        status, deleted = self.request(
            run_path, "DELETE", {"confirm_run_id": run_id}
        )
        self.assertEqual(status, 200)
        self.assertTrue(deleted["permanently_deleted"])
        self.assertEqual(deleted["deleted"]["events"], before_events)
        self.assertEqual(deleted["deleted"]["command_receipts"], 2)
        self.assertEqual(self.server.ledger.count(run_id), 0)
        self.assertIsNone(self.server.ledger.command_receipt("create:purge-target"))
        self.assertEqual(self.server.ledger.command_count(), before_commands - 2)
        self.assertGreater(self.server.ledger.count(other_run_id), 0)

        status, _missing = self.request(run_path)
        self.assertEqual(status, 404)
        status, listing = self.request("/api/runs?include_archived=true")
        self.assertEqual(status, 200)
        self.assertEqual([item["run_id"] for item in listing["runs"]], [other_run_id])

        status, recreated = self.request("/api/runs", "POST", create_body)
        self.assertEqual(status, 201)
        self.assertNotEqual(recreated["projection"]["run_id"], run_id)

    def test_pending_create_receipt_remains_owned_for_cleanup(self) -> None:
        body = native_fixture.NativeBindingFixture.body("interrupted-create")
        with patch.object(self.server.director, "record_model_contract_preflight",
                          side_effect=RuntimeError("injected audit persistence failure")):
            status, failed = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 400, failed)
        run_id = self.server.ledger.run_ids()[0]
        receipt = self.server.ledger.command_receipt("create:interrupted-create")
        self.assertEqual(receipt.status, "pending")
        self.assertEqual(receipt.resource_run_id, run_id)
        path = "/api/runs/" + quote(run_id, safe="")
        self.assertEqual(self.request(path + "/control", "POST", {"action": "cancel"})[0], 200)
        self.assertEqual(self.request(path + "/archive", "POST", {})[0], 200)
        status, deleted = self.request(path, "DELETE", {"confirm_run_id": run_id})
        self.assertEqual(status, 200, deleted)
        self.assertEqual(self.server.ledger.command_count(), 0)

    def test_purge_fences_active_worker_and_reused_run_id_incarnation(self) -> None:
        run_id = "run:purge-incarnation-fence"
        path = "/api/runs/" + quote(run_id, safe="")
        body = native_fixture.NativeBindingFixture.body("incarnation-old", run_id=run_id)
        status, created = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 201, created)
        incarnation = self.server.director.state(run_id).events[0].seq
        lease = self.server.generation_lock(run_id)
        self.assertTrue(lease.acquire(blocking=False))
        try:
            self.server.director.cancel_run(run_id)
            self.assertEqual(self.request(path + "/archive", "POST", {})[0], 200)
            status, blocked = self.request(path, "DELETE", {"confirm_run_id": run_id})
            self.assertEqual(status, 409, blocked)
            self.assertGreater(self.server.ledger.count(run_id), 0)
        finally:
            lease.release()
        work_item = (run_id, incarnation)
        self.server.auto_progress._defer_failure(work_item, "stale failure")
        self.assertEqual(self.request(path, "DELETE", {"confirm_run_id": run_id})[0], 200)
        self.assertNotIn(work_item, self.server.auto_progress._deferred_failures)
        self.assertFalse(lease.acquire(blocking=False))
        status, recreated = self.request("/api/runs", "POST", {**body, "idempotency_key": "incarnation-new"})
        self.assertEqual(status, 201, recreated)
        self.assertIsNot(self.server.generation_lock(run_id), lease)
        events = self.server.ledger.events(run_id)
        with patch.object(auto_progress_module, "execute_generation") as execute:
            keep_running = self.server.auto_progress._run_one_generation(run_id, expected_incarnation=incarnation)
        self.assertFalse(keep_running)
        execute.assert_not_called()
        self.assertEqual(self.server.ledger.events(run_id), events)


class RunCleanupSchemaTests(unittest.TestCase):
    def test_legacy_database_is_rejected_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite3"
            with sqlite3.connect(path) as connection:
                connection.executescript("PRAGMA user_version=4; CREATE TABLE legacy_marker (value TEXT);")
            with self.assertRaisesRegex(RuntimeError, "schema"):
                EventLedger(path)
            with sqlite3.connect(path) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 4)
                self.assertIsNotNone(connection.execute("SELECT name FROM sqlite_master WHERE name='legacy_marker'").fetchone())


if __name__ == "__main__":
    unittest.main()
