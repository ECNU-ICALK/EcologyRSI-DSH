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

from ecologyrsi_dsh.api import auto_progress as auto_progress_module
from ecologyrsi_dsh.ledger import EventLedger
from ecologyrsi_dsh.models import canonical_json, digest
from ecologyrsi_dsh.server import EvolutionHTTPServer, EvolutionRequestHandler


class RunCleanupHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.server = EvolutionHTTPServer(
            ("127.0.0.1", 0), Path(self.directory.name) / "events.sqlite3"
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.directory.cleanup()

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
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def create(self, key: str) -> tuple[str, dict]:
        body = {
            "domain_pack_id": "crop_soil_water",
            "dataset_id": "generated-toy-series@1",
            "budget": {"max_generations": 2, "max_candidates": 2},
            "auto_advance": 0,
            "idempotency_key": key,
        }
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
            "ecologyrsi-dsh.browser-run-summary/1",
        )
        self.assertNotIn("rounds", summary["runs"][0])
        self.assertNotIn("candidates", summary["runs"][0])
        self.assertIn("configuration", summary["runs"][0])
        self.assertIn("budget", summary["runs"][0])

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

    def test_failed_auto_advance_keeps_pending_create_receipt_owned_for_cleanup(self) -> None:
        body = {
            "domain_pack_id": "crop_soil_water",
            "dataset_id": "generated-toy-series@1",
            "budget": {"max_generations": 2, "max_candidates": 2},
            "auto_advance": 1,
            "idempotency_key": "failed-auto-advance-owner",
        }
        with patch.object(
            EvolutionRequestHandler,
            "_advance_run",
            side_effect=RuntimeError("injected auto advance failure"),
        ):
            status, failed = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 400, failed)
        self.assertEqual(failed["command_status"], "等待恢复")

        run_id = self.server.ledger.run_ids()[0]
        receipt = self.server.ledger.command_receipt(
            "create:failed-auto-advance-owner"
        )
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.status, "pending")
        self.assertEqual(receipt.resource_run_id, run_id)

        run_path = "/api/runs/" + quote(run_id, safe="")
        status, _cancelled = self.request(
            run_path + "/control", "POST", {"action": "cancel"}
        )
        self.assertEqual(status, 200)
        status, _archived = self.request(run_path + "/archive", "POST", {})
        self.assertEqual(status, 200)
        status, deleted = self.request(
            run_path, "DELETE", {"confirm_run_id": run_id}
        )
        self.assertEqual(status, 200)
        self.assertEqual(deleted["deleted"]["command_receipts"], 1)
        self.assertEqual(self.server.ledger.command_count(), 0)

    def test_purge_fences_active_worker_and_reused_run_id_incarnation(self) -> None:
        run_id = "run:purge-incarnation-fence"
        run_path = "/api/runs/" + quote(run_id, safe="")
        old_body = {
            "run_id": run_id,
            "domain_pack_id": "crop_soil_water",
            "dataset_id": "generated-toy-series@1",
            "rounds": 2,
            "candidates_per_generation": 1,
            "max_candidates": 2,
            "auto_progress": True,
            "auto_advance": 0,
            "start": False,
            "idempotency_key": "purge-incarnation-old",
        }
        proposal_started = threading.Event()
        release_proposal = threading.Event()
        proposal_returned = threading.Event()
        propose = self.server.strategy_router.propose

        def blocked_propose(*args, **kwargs):
            proposal_started.set()
            release_proposal.wait(timeout=5)
            result = propose(*args, **kwargs)
            proposal_returned.set()
            return result

        with patch.object(
            self.server.strategy_router,
            "propose",
            side_effect=blocked_propose,
        ):
            try:
                status, created = self.request("/api/runs", "POST", old_body)
                self.assertEqual(status, 201, created)
                old_state = self.server.director.state(run_id)
                old_incarnation = old_state.events[0].seq
                old_lease = self.server.generation_lock(run_id)

                status, started = self.request(
                    run_path + "/action", "POST", {"action": "start"}
                )
                self.assertEqual(status, 200, started)
                self.assertTrue(proposal_started.wait(timeout=2))

                status, cancelled = self.request(
                    run_path + "/control", "POST", {"action": "cancel"}
                )
                self.assertEqual(status, 200, cancelled)
                status, archived = self.request(run_path + "/archive", "POST", {})
                self.assertEqual(status, 200, archived)

                status, blocked = self.request(
                    run_path, "DELETE", {"confirm_run_id": run_id}
                )
                self.assertEqual(status, 409, blocked)
                self.assertIn("generation", blocked["error"])
                self.assertGreater(self.server.ledger.count(run_id), 0)
            finally:
                release_proposal.set()

            self.assertTrue(proposal_returned.wait(timeout=3))
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if old_lease.acquire(blocking=False):
                    old_lease.release()
                    break
                time.sleep(0.01)
            else:
                self.fail("background generation did not release its lease")

            old_work_item = (run_id, old_incarnation)
            self.server.auto_progress._defer_failure(
                old_work_item, "stale failure from deleted incarnation"
            )
            status, deleted = self.request(
                run_path, "DELETE", {"confirm_run_id": run_id}
            )
            self.assertEqual(status, 200, deleted)
            self.assertEqual(self.server.ledger.count(run_id), 0)
            with self.server._generation_locks_guard:
                self.assertNotIn(run_id, self.server._generation_locks)
            self.assertFalse(old_lease.acquire(blocking=False))
            self.assertNotIn(
                old_work_item,
                self.server.auto_progress._deferred_failures,
            )

            new_body = {
                **old_body,
                "rounds": 1,
                "max_candidates": 1,
                "auto_progress": False,
                "start": True,
                "idempotency_key": "purge-incarnation-new",
            }
            status, recreated = self.request("/api/runs", "POST", new_body)
            self.assertEqual(status, 201, recreated)
            new_lease = self.server.generation_lock(run_id)
            self.assertIsNot(new_lease, old_lease)

            before = [event.kind for event in self.server.ledger.events(run_id)]
            with patch.object(auto_progress_module, "execute_generation") as execute:
                keep_running = self.server.auto_progress._run_one_generation(
                    run_id,
                    expected_incarnation=old_incarnation,
                )
            self.assertFalse(keep_running)
            execute.assert_not_called()
            self.assertEqual(
                [event.kind for event in self.server.ledger.events(run_id)],
                before,
            )
            self.assertEqual(before, ["RunCreated", "RunStarted"])


class RunCleanupMigrationTests(unittest.TestCase):
    def test_v4_create_receipt_is_structurally_bound_and_purged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite3"
            run_id = "run:legacy-create-owner"
            request_body = {"run_id": run_id, "idempotency_key": "legacy"}
            response_body = {"projection": {"run_id": run_id, "status": "cancelled"}}
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE evolution_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE http_command_receipts (
                    command_key TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    command_kind TEXT NOT NULL,
                    request_digest TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    start_seq INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'completed')),
                    response_json TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );
                PRAGMA user_version = 4;
                """
            )
            connection.execute(
                """
                INSERT INTO evolution_events
                    (event_id, run_id, kind, payload_json, created_at)
                VALUES (?, ?, 'RunCancelled', '{}', '2026-08-17T00:00:00+00:00')
                """,
                ("legacy-event", run_id),
            )
            connection.execute(
                """
                INSERT INTO http_command_receipts
                    (command_key, run_id, command_kind, request_digest,
                     request_json, start_seq, status, response_json,
                     created_at, completed_at)
                VALUES (?, 'create', 'create_run', ?, ?, 0, 'completed', ?, ?, ?)
                """,
                (
                    "legacy-create",
                    digest(request_body),
                    canonical_json(request_body),
                    canonical_json(response_body),
                    "2026-08-17T00:00:00+00:00",
                    "2026-08-17T00:00:01+00:00",
                ),
            )
            connection.commit()
            connection.close()

            ledger = EventLedger(path)
            self.addCleanup(ledger.close)
            receipt = ledger.command_receipt("legacy-create")
            self.assertIsNotNone(receipt)
            self.assertEqual(receipt.resource_run_id, run_id)
            ledger.archive_run(run_id)
            deleted = ledger.purge_run(
                run_id, confirmation=run_id, terminal_status="cancelled"
            )
            self.assertEqual(deleted["command_receipts"], 1)
            self.assertEqual(ledger.count(run_id), 0)
            self.assertEqual(ledger.command_count(), 0)


if __name__ == "__main__":
    unittest.main()
