"""Current-version crash recovery, ownership and immutable dataset boundaries."""
from pathlib import Path
from dataclasses import replace
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import quote

from ecologyrsi_dsh.api.handler import EvolutionHTTPServer
from ecologyrsi_dsh.core.ledger import EventLedger
from tests import native_http_fixture, test_dsh_native_runtime, test_runtime_integration
from tests.http_poll_fixture import wait_for_command


class SchemaBoundaryTests(unittest.TestCase):
    def test_old_schemas_fail_without_migration(self):
        for version in (1, 3, 5, 8):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as root:
                path = Path(root) / "ledger.db"
                with sqlite3.connect(path) as db:
                    db.executescript(f"PRAGMA user_version={version}; CREATE TABLE marker(value TEXT); INSERT INTO marker VALUES('preserve');")
                with self.assertRaisesRegex(RuntimeError, "unsupported"):
                    EventLedger(path)
                with sqlite3.connect(path) as db:
                    self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], version)
                    self.assertEqual(db.execute("SELECT value FROM marker").fetchone()[0], "preserve")


class RecoveryAndLineageTests(unittest.TestCase):
    def setUp(self):
        native_http_fixture.NativeBindingFixture.setUp(self)
        self.db_path = Path(self.directory.name) / "events.sqlite3"

    tearDown = native_http_fixture.NativeBindingFixture.tearDown

    def request(self, path, method="GET", body=None):
        status, result = test_runtime_integration.RuntimeIntegrationTests.request(self, path, method, body)
        return wait_for_command(self, result) if status == 202 else (status, result)

    def restart(self):
        datasets = self.server.datasets
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.server = EvolutionHTTPServer(("127.0.0.1", 0), self.db_path)
        self.server.datasets = datasets
        self.server.dsh_native_runtime = test_dsh_native_runtime._FakeNativeRuntime()
        scheduling = patch.object(self.server.auto_progress, "schedule_if_enabled", return_value=False)
        scheduling.start()
        self.addCleanup(scheduling.stop)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/api"

    def body(self, key, **extra):
        return native_http_fixture.NativeBindingFixture.body(key, **extra)

    def create(self, key):
        status, result = self.request("/runs", "POST", self.body(key))
        self.assertEqual(status, 201, result)
        return result["projection"]["run_id"], result["projection"]

    def test_persistent_database_has_one_sidecar_owner(self):
        with self.assertRaisesRegex(RuntimeError, "already owns this database"):
            EvolutionHTTPServer(("127.0.0.1", 0), self.db_path)
        self.restart()
        self.assertEqual(self.request("/health")[0], 200)

    def test_partial_create_audit_recovers_same_run_without_duplication(self):
        body = self.body("partial-audit")
        with patch.object(self.server.director, "record_model_contract_preflight", side_effect=RuntimeError("crash boundary")):
            status, _ = self.request("/runs", "POST", body)
        self.assertEqual(status, 400)
        before = self.server.ledger.run_ids()
        self.assertEqual(len(before), 1)
        self.assertEqual(self.server.ledger.command_receipt("create:partial-audit").status, "pending")
        status, result = self.request("/runs", "POST", body)
        self.assertEqual(status, 200, result)
        self.assertEqual(self.server.ledger.run_ids(), before)
        self.assertEqual(self.server.ledger.pending_command_keys(), ())
        kinds = [event.kind for event in self.server.ledger.events(before[0])]
        self.assertEqual(kinds.count("ModelContractPreflightRecorded"), 1)

    def test_unstarted_create_can_resume_after_restart(self):
        body = self.body("partial-start", start=True)
        with patch.object(self.server.director, "start_run", side_effect=RuntimeError("crash boundary")):
            status, _ = self.request("/runs", "POST", body)
        self.assertEqual(status, 400)
        before = self.server.ledger.run_ids()
        self.restart()
        self.assertEqual(self.server.ledger.command_receipt("create:partial-start").status, "pending")
        status, result = self.request("/runs", "POST", body)
        self.assertEqual(status, 200, result)
        self.assertEqual(result["projection"]["status"], "running")
        self.assertEqual(self.server.ledger.run_ids(), before)

    def test_restart_reconciles_cancel_but_never_misattributed_pause(self):
        run_id, _ = self.create("pending-controls")
        for action in ("pause", "cancel"):
            self.server.ledger.begin_command(f"{run_id}:{action}", run_id, f"control:{action}", {"action": action})
        self.server.director.cancel_run(run_id, "observed cancel")
        self.restart()
        self.assertEqual(self.server.ledger.command_receipt(f"{run_id}:cancel").status, "completed")
        self.assertEqual(self.server.ledger.command_receipt(f"{run_id}:pause").status, "pending")

    def test_unbound_or_missing_create_receipts_are_not_sealed(self):
        for name in ("unbound", "missing"):
            self.server.ledger.begin_command(name, "create", "create_run", {"idempotency_key": name})
        self.server.ledger.bind_command_resource_run("missing", "run:absent")
        self.restart()
        self.assertCountEqual(self.server.ledger.pending_command_keys(), ["unbound", "missing"])

    def test_page_and_run_share_frozen_dataset_identity(self):
        _, projection = self.create("dataset-identity")
        query = ("/datasets/agc_cucumber_2018?partition=training_fit&offset=0&limit=3"
                 "&episode_id=" + quote(projection["dataset"]["episode_id"], safe="")
                 + "&expected_dataset_digest=" + projection["dataset_digest"]
                 + "&expected_split_manifest_digest=" + projection["dataset"]["split_manifest_digest"])
        status, page = self.request(query)
        self.assertEqual(status, 200, page)
        self.assertEqual(page["dataset_digest_sha256"], projection["dataset_digest"])
        status, result = self.request(query.replace(projection["dataset_digest"], "0" * 64))
        self.assertEqual(status, 400, result)

    def test_restart_detects_dataset_drift_before_start(self):
        run_id, _ = self.create("dataset-drift")
        self.restart()
        view = self.server.datasets.selection_view("agc_cucumber_2018", episode_id="agc_cucumber_2018:TeamA")
        with patch.object(self.server.datasets, "selection_view", return_value=replace(view, dataset_digest="0" * 64)):
            status, result = self.request("/runs/" + quote(run_id, safe="") + "/control", "POST", {"action": "start"})
        self.assertEqual(status, 400, result)
        self.assertEqual(self.server.director.state(run_id).run.status.value, "created")
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_public_create_cannot_inject_causal_scope_or_model_alias(self):
        for extra in ({"scientific_scope": "causal_effect"}, {"task_manifest": {}}, {"policy_model_id": "other"}):
            status, result = self.request("/runs", "POST", self.body("bad", **extra))
            self.assertEqual(status, 400, result)
        self.assertEqual(self.server.ledger.run_ids(), ())
