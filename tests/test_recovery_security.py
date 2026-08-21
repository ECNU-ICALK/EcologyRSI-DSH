from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ecologyrsi_dsh.server import EvolutionHTTPServer, EvolutionRequestHandler
from ecologyrsi_dsh.ledger import EventLedger, SCHEMA_VERSION
from ecologyrsi_dsh.models import canonical_json, digest


class LedgerMigrationTests(unittest.TestCase):
    def test_v1_command_receipt_migrates_with_durable_start_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite3"
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
                    status TEXT NOT NULL CHECK (status IN ('pending', 'completed')),
                    response_json TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );
                PRAGMA user_version = 1;
                """
            )
            request = {"steps": 1}
            connection.execute(
                "INSERT INTO evolution_events "
                "(event_id, run_id, kind, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                ("event-before", "run:legacy", "RunStarted", "{}", "2026-08-16T00:00:00+00:00"),
            )
            connection.execute(
                "INSERT INTO http_command_receipts "
                "(command_key, run_id, command_kind, request_digest, request_json, "
                "status, response_json, created_at, completed_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', NULL, ?, NULL)",
                (
                    "legacy-advance",
                    "run:legacy",
                    "advance",
                    digest(request),
                    canonical_json(request),
                    "2026-08-16T00:00:01+00:00",
                ),
            )
            connection.execute(
                "INSERT INTO evolution_events "
                "(event_id, run_id, kind, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                ("event-after", "run:other", "RunStarted", "{}", "2026-08-16T00:00:02+00:00"),
            )
            connection.commit()
            connection.close()

            ledger = EventLedger(path)
            self.addCleanup(ledger.close)
            receipt = ledger.command_receipt("legacy-advance")

            self.assertEqual(ledger.schema_version, SCHEMA_VERSION)
            self.assertIsNotNone(receipt)
            self.assertEqual(receipt.start_seq, 1)
            self.assertIsNone(
                ledger.begin_command(
                    "legacy-advance",
                    "run:legacy",
                    "advance",
                    request,
                    resume_pending=True,
                )
            )

    def test_v3_migrates_to_current_model_verification_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-v3.sqlite3"
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
                CREATE TABLE model_verifications (
                    model_id TEXT PRIMARY KEY,
                    configuration_digest TEXT NOT NULL,
                    credential_fingerprint TEXT NOT NULL,
                    verified INTEGER NOT NULL CHECK (verified IN (0, 1)),
                    connection_state TEXT NOT NULL,
                    last_checked_at TEXT,
                    last_error TEXT,
                    updated_at TEXT NOT NULL
                );
                INSERT INTO model_verifications (
                    model_id, configuration_digest, credential_fingerprint,
                    verified, connection_state, last_checked_at, last_error,
                    updated_at
                ) VALUES (
                    'legacy/model', 'legacy-configuration', 'legacy-credential',
                    1, 'available', '2026-08-16T00:00:00+00:00',
                    'GatewayResponseError: Authorization: Bearer migrated-audit-secret',
                    '2026-08-16T00:00:00+00:00'
                );
                PRAGMA user_version = 3;
                """
            )
            connection.commit()
            connection.close()

            ledger = EventLedger(path)
            self.addCleanup(ledger.close)

            self.assertEqual(SCHEMA_VERSION, 7)
            self.assertEqual(ledger.schema_version, 7)
            inspection = sqlite3.connect(path)
            try:
                columns = inspection.execute(
                    "PRAGMA table_info(model_verifications)"
                ).fetchall()
            finally:
                inspection.close()
            self.assertEqual(
                [(row[1], row[2], row[3], row[5]) for row in columns],
                [
                    ("model_id", "TEXT", 1, 1),
                    ("configuration_digest", "TEXT", 1, 2),
                    ("credential_fingerprint", "TEXT", 1, 3),
                    ("verified", "INTEGER", 1, 0),
                    ("connection_state", "TEXT", 1, 0),
                    ("last_checked_at", "TEXT", 0, 0),
                    ("last_error", "TEXT", 0, 0),
                    ("updated_at", "TEXT", 1, 0),
                ],
            )

            migrated = ledger.model_verification(
                "legacy/model",
                "legacy-configuration",
                "legacy-credential",
            )
            self.assertIsNotNone(migrated)
            self.assertTrue(migrated["verified"])
            self.assertEqual(migrated["state"], "available")
            self.assertEqual(migrated["last_error"], "GatewayResponseError")
            inspection = sqlite3.connect(path)
            try:
                stored_error = inspection.execute(
                    "SELECT last_error FROM model_verifications WHERE model_id = ?",
                    ("legacy/model",),
                ).fetchone()[0]
            finally:
                inspection.close()
            self.assertEqual(stored_error, "GatewayResponseError")

            ledger.record_model_verification(
                "provider/model",
                "configuration-digest",
                "credential-fingerprint",
                verified=True,
                state="available",
                last_checked_at="2026-08-17T00:00:00+00:00",
                last_error=None,
            )
            restored = ledger.model_verification(
                "provider/model",
                "configuration-digest",
                "credential-fingerprint",
            )
            self.assertIsNotNone(restored)
            self.assertTrue(restored["verified"])
            self.assertEqual(restored["state"], "available")

    def test_v3_verification_without_credential_fingerprint_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-v3-without-fingerprint.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE model_verifications (
                    model_id TEXT PRIMARY KEY,
                    configuration_digest TEXT NOT NULL,
                    verified INTEGER NOT NULL CHECK (verified IN (0, 1)),
                    connection_state TEXT NOT NULL,
                    last_checked_at TEXT,
                    last_error TEXT,
                    updated_at TEXT NOT NULL
                );
                INSERT INTO model_verifications (
                    model_id, configuration_digest, verified, connection_state,
                    last_checked_at, last_error, updated_at
                ) VALUES (
                    'legacy/model', 'legacy-configuration', 1, 'available',
                    '2026-08-16T00:00:00+00:00', NULL,
                    '2026-08-16T00:00:00+00:00'
                );
                PRAGMA user_version = 3;
                """
            )
            connection.commit()
            connection.close()

            ledger = EventLedger(path)
            self.addCleanup(ledger.close)

            self.assertEqual(ledger.schema_version, 7)
            inspection = sqlite3.connect(path)
            try:
                columns = inspection.execute(
                    "PRAGMA table_info(model_verifications)"
                ).fetchall()
                row_count = inspection.execute(
                    "SELECT COUNT(*) FROM model_verifications"
                ).fetchone()[0]
            finally:
                inspection.close()
            self.assertEqual(
                [row[1] for row in columns[:3]],
                [
                    "model_id",
                    "configuration_digest",
                    "credential_fingerprint",
                ],
            )
            self.assertEqual([row[5] for row in columns[:3]], [1, 2, 3])
            self.assertEqual(row_count, 0)
            self.assertIsNone(
                ledger.model_verification(
                    "legacy/model",
                    "legacy-configuration",
                    "any-current-credential",
                )
            )


class RecoveryAndLineageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self._directory.name) / "events.sqlite3"
        self.server = EvolutionHTTPServer(
            ("127.0.0.1", 0), self.db_path
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self._directory.cleanup()

    def test_frozen_model_alias_conflict_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "策略模型字段冲突"):
            EvolutionRequestHandler._validate_frozen_model_aliases(
                {
                    "strategy_model_id": "strategy-api",
                    "policy_model_id": "different-api",
                }
            )

    def request(
        self,
        path: str,
        method: str = "GET",
        body: dict | None = None,
    ) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_persistent_database_has_one_sidecar_owner(self) -> None:
        other_db = Path(self._directory.name) / "single-owner.sqlite3"
        first = EvolutionHTTPServer(("127.0.0.1", 0), other_db)
        try:
            with self.assertRaisesRegex(RuntimeError, "already owns this database"):
                EvolutionHTTPServer(("127.0.0.1", 0), other_db)
        finally:
            first.close()

        replacement = EvolutionHTTPServer(("127.0.0.1", 0), other_db)
        replacement.close()

    @staticmethod
    def create_body(key: str, *, auto_advance: int = 1, seed: int = 7) -> dict:
        return {
            "domain_pack_id": "crop_soil_water",
            "dataset_id": "generated-toy-series@1",
            "budget": {"max_generations": 2, "max_candidates": 2},
            "auto_advance": auto_advance,
            "seed": seed,
            "idempotency_key": key,
        }

    def test_proposal_failure_is_a_visible_terminal_run(self) -> None:
        body = self.create_body("proposal-failure", auto_advance=1)
        with patch.object(
            self.server.director,
            "request_proposal",
            side_effect=RuntimeError("invalid model response"),
        ):
            status, failed = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 201, failed)
        self.assertEqual(failed["projection"]["status"], "failed")
        self.assertEqual(self.server.ledger.pending_command_keys(), ())
        receipt = self.server.ledger.command_receipt("create:proposal-failure")
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.status, "completed")

        status, replayed = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 200, replayed)
        self.assertEqual(replayed, failed)

    def test_partial_create_receipt_resumes_with_same_key(self) -> None:
        body = self.create_body("resume-create", auto_advance=1)
        with patch.object(
            EvolutionRequestHandler,
            "_advance_run",
            side_effect=RuntimeError("simulated process boundary"),
        ):
            status, interrupted = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 400, interrupted)
        self.assertEqual(interrupted["command_status"], "等待恢复")
        self.assertEqual(
            self.server.ledger.pending_command_keys(), ("create:resume-create",)
        )

        status, recovered = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 200, recovered)
        self.assertEqual(recovered["projection"]["candidates_count"], 1)
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_restart_seals_only_bound_replayable_create_receipts(self) -> None:
        body = self.create_body("restart-seal-create", auto_advance=1)
        with patch.object(
            EvolutionRequestHandler,
            "_advance_run",
            side_effect=RuntimeError("simulated response boundary"),
        ):
            status, interrupted = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 400, interrupted)
        self.assertEqual(interrupted["command_status"], "等待恢复")

        run_id = self.server.ledger.run_ids()[0]
        run_path = f"/api/runs/{quote(run_id, safe='')}"
        status, cancelled = self.request(
            run_path + "/control", "POST", {"action": "cancel"}
        )
        self.assertEqual(status, 200, cancelled)
        self.assertEqual(cancelled["projection"]["status"], "cancelled")

        self.assertIsNone(
            self.server.ledger.begin_command(
                "create:unbound-recovery-control",
                "create",
                "create_run",
                {"idempotency_key": "unbound-recovery-control"},
            )
        )
        self.assertIsNone(
            self.server.ledger.begin_command(
                "create:missing-recovery-control",
                "create",
                "create_run",
                {"idempotency_key": "missing-recovery-control"},
            )
        )
        self.server.ledger.bind_command_resource_run(
            "create:missing-recovery-control", "run:missing-recovery-control"
        )
        self.assertIsNone(
            self.server.ledger.begin_command(
                "control:pending-recovery-control",
                run_id,
                "control:pause",
                {"action": "pause"},
            )
        )

        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.server = EvolutionHTTPServer(("127.0.0.1", 0), self.db_path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

        receipt = self.server.ledger.command_receipt("create:restart-seal-create")
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.status, "completed")
        self.assertEqual(receipt.resource_run_id, run_id)
        self.assertEqual(receipt.response["projection"]["status"], "cancelled")
        self.assertCountEqual(
            self.server.ledger.pending_command_keys(),
            [
                "control:pending-recovery-control",
                "create:missing-recovery-control",
                "create:unbound-recovery-control",
            ],
        )

        status, replayed = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 200, replayed)
        self.assertEqual(replayed, receipt.response)

    def test_restart_keeps_partial_synchronous_create_resumable(self) -> None:
        body = self.create_body("restart-resume-create", auto_advance=1)
        with patch.object(
            EvolutionRequestHandler,
            "_advance_run",
            side_effect=RuntimeError("simulated process boundary"),
        ):
            status, interrupted = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 400, interrupted)

        run_id = self.server.ledger.run_ids()[0]
        self.assertEqual(self.server.director.state(run_id).run.status.value, "running")
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.server = EvolutionHTTPServer(("127.0.0.1", 0), self.db_path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

        receipt = self.server.ledger.command_receipt("create:restart-resume-create")
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.status, "pending")
        status, recovered = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 200, recovered)
        self.assertEqual(recovered["projection"]["candidates_count"], 1)
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_restart_keeps_bound_unstarted_create_resumable(self) -> None:
        body = self.create_body("restart-resume-start", auto_advance=0)
        with patch.object(
            self.server.director,
            "start_run",
            side_effect=RuntimeError("simulated process boundary"),
        ):
            status, interrupted = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 400, interrupted)

        run_id = self.server.ledger.run_ids()[0]
        self.assertEqual(self.server.director.state(run_id).run.status.value, "created")
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.server = EvolutionHTTPServer(("127.0.0.1", 0), self.db_path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

        receipt = self.server.ledger.command_receipt("create:restart-resume-start")
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.status, "pending")
        status, recovered = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 200, recovered)
        self.assertEqual(recovered["projection"]["status"], "running")
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_finalization_event_gap_is_repaired_on_same_key_retry(self) -> None:
        body = {
            "domain_pack_id": "crop_soil_water",
            "dataset_id": "generated-toy-series@1",
            "slot": "residual_water_stress",
            "budget": {"max_generations": 1, "max_candidates": 1},
            "auto_advance": 0,
            "idempotency_key": "finalization-gap",
        }
        status, created = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 201, created)
        run_id = str(created["projection"]["run_id"])
        run_path = f"/api/runs/{quote(run_id, safe='')}"
        advance = {"steps": 1, "idempotency_key": "finalization-gap-advance"}
        original_append = self.server.ledger.append

        def fail_completion(
            event_run_id,
            kind,
            payload,
            *,
            event_id=None,
            created_at=None,
        ):
            if kind == "RunCompleted":
                raise RuntimeError("simulated process boundary")
            return original_append(
                event_run_id,
                kind,
                payload,
                event_id=event_id,
                created_at=created_at,
            )

        with patch.object(self.server.ledger, "append", side_effect=fail_completion):
            status, interrupted = self.request(run_path + "/advance", "POST", advance)
        self.assertEqual(status, 400, interrupted)
        self.assertEqual(interrupted["command_status"], "等待恢复")
        self.assertEqual(
            self.server.ledger.pending_command_keys(),
            (f"{run_id}:finalization-gap-advance",),
        )

        status, recovered = self.request(run_path + "/advance", "POST", advance)
        self.assertEqual(status, 200, recovered)
        self.assertEqual(recovered["projection"]["status"], "completed")
        self.assertEqual(self.server.ledger.pending_command_keys(), ())
        self.assertEqual(
            sum(
                event.kind == "RunCompleted"
                for event in self.server.ledger.events(run_id)
            ),
            1,
        )

    def test_spawned_candidate_is_reconciled_instead_of_duplicated(self) -> None:
        status, created = self.request(
            "/api/runs", "POST", self.create_body("partial-spawn-run", auto_advance=0)
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        advance = {"steps": 1, "idempotency_key": "resume-spawned"}
        key = f"{run_id}:resume-spawned"
        self.assertIsNone(
            self.server.ledger.begin_command(key, run_id, "advance", advance)
        )
        spawned = self.server.director.propose_and_spawn(run_id)

        status, recovered = self.request(
            f"/api/runs/{quote(run_id, safe='')}/advance", "POST", advance
        )
        self.assertEqual(status, 200, recovered)
        self.assertEqual(recovered["projection"]["candidates_count"], 1)
        self.assertEqual(
            recovered["projection"]["candidates"][0]["candidate_id"],
            spawned.candidate_id,
        )
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_orphan_proposal_recovers_exact_human_intervention_receipt(self) -> None:
        status, created = self.request(
            "/api/runs", "POST", self.create_body("receipt-recovery", auto_advance=1)
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        run_path = f"/api/runs/{quote(run_id, safe='')}"
        status, _paused = self.request(
            run_path + "/control", "POST", {"action": "pause"}
        )
        self.assertEqual(status, 200)
        status, _recorded = self.request(
            run_path + "/interventions",
            "POST",
            {
                "kind": "parameter_override",
                "message": "将平滑系数固定为恢复测试值。",
                "parameter_overrides": {"alpha": 0.42},
                "created_by": "恢复测试研究员",
            },
        )
        self.assertEqual(status, 201)
        status, _resumed = self.request(
            run_path + "/control", "POST", {"action": "resume"}
        )
        self.assertEqual(status, 200)

        original_append = self.server.ledger.append

        def crash_before_receipt(
            event_run_id,
            kind,
            payload,
            *,
            event_id=None,
            created_at=None,
        ):
            if kind == "HumanInterventionApplied":
                raise KeyboardInterrupt("simulated hard process exit")
            return original_append(
                event_run_id,
                kind,
                payload,
                event_id=event_id,
                created_at=created_at,
            )

        with patch.object(
            self.server.ledger,
            "append",
            side_effect=crash_before_receipt,
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.server.director.propose_and_spawn(run_id)

        interrupted = self.server.director.state(run_id)
        self.assertEqual(len(interrupted.pending_interventions), 1)
        proposal_event = next(
            event
            for event in reversed(interrupted.events)
            if event.kind == "ProposalSubmitted"
        )
        embedded = proposal_event.payload["intervention_receipts"][0]
        self.assertEqual(embedded["application_status"], "enforced")
        self.assertEqual(embedded["result_values"], {"alpha": 0.42})

        status, recovered = self.request(
            run_path + "/advance",
            "POST",
            {"steps": 1, "idempotency_key": "recover-exact-receipt"},
        )
        self.assertEqual(status, 200, recovered)
        projection = recovered["projection"]
        intervention = projection["interventions"][0]
        self.assertEqual(intervention["application_status"], "enforced")
        self.assertEqual(intervention["result_values"], {"alpha": 0.42})
        self.assertEqual(projection["candidates"][0]["changes"]["alpha"], 0.42)

    def test_training_page_run_and_artifact_share_one_snapshot(self) -> None:
        status, page = self.request(
            "/api/datasets/generated-toy-series%401/samples"
            "?partition=training_fit&offset=0&limit=3"
        )
        self.assertEqual(status, 200, page)
        status, created = self.request(
            "/api/runs", "POST", self.create_body("lineage-snapshot", seed=987)
        )
        self.assertEqual(status, 201, created)
        projection = created["projection"]
        self.assertEqual(projection["dataset_digest"], page["dataset_digest_sha256"])
        self.assertEqual(
            projection["dataset"]["episode_id"], page["dataset"]["episode_id"]
        )
        self.assertEqual(
            projection["artifacts"][0]["dataset_digest"],
            page["dataset_digest_sha256"],
        )
        frozen_query = (
            "/api/datasets/generated-toy-series%401/samples"
            "?partition=training_fit&offset=0&limit=3"
            f"&episode_id={quote(projection['dataset']['episode_id'], safe='')}"
            f"&expected_dataset_digest={projection['dataset_digest']}"
            "&expected_split_manifest_digest="
            f"{projection['dataset']['split_manifest_digest']}"
        )
        status, frozen_page = self.request(frozen_query)
        self.assertEqual(status, 200, frozen_page)
        self.assertEqual(frozen_page["dataset_digest_sha256"], projection["dataset_digest"])
        status, drifted_page = self.request(
            frozen_query.replace(
                f"expected_dataset_digest={projection['dataset_digest']}",
                "expected_dataset_digest=" + "0" * 64,
            )
        )
        self.assertEqual(status, 400, drifted_page)
        self.assertIn("数据集快照发生漂移", drifted_page["error"])

    def test_restarted_service_rejects_changed_dataset_snapshot(self) -> None:
        status, created = self.request(
            "/api/runs",
            "POST",
            self.create_body("restart-dataset-drift", auto_advance=0),
        )
        self.assertEqual(status, 201, created)
        run_id = created["projection"]["run_id"]
        frozen_digest = created["projection"]["dataset_digest"]

        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self.server = EvolutionHTTPServer(("127.0.0.1", 0), self.db_path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        current = self.server.datasets.series("generated-toy-series@1")
        drifted = replace(current, digest="0" * 64)

        with patch.object(self.server.datasets, "series", return_value=drifted):
            status, rejected = self.request(
                f"/api/runs/{quote(run_id, safe='')}/advance",
                "POST",
                {"steps": 1},
            )
        self.assertEqual(status, 400, rejected)
        self.assertIn("数据集快照发生漂移", rejected["error"])
        self.assertEqual(frozen_digest, created["projection"]["dataset"]["digest"])
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_client_cannot_claim_a_causal_scope(self) -> None:
        body = {
            "task_manifest": {
                "task_id": "scope-spoof",
                "objective": "historical prediction only",
                "domain_pack": "crop_soil_water",
                "visible_datasets": ["generated-toy-series@1"],
                "budget": 1,
                "seed": 0,
                "seed_policy": "fixed",
                "policy_version": "policy@1",
                "metadata": {
                    "evaluation_partition": "validation",
                    "scientific_scope": "causal_effect_estimate",
                },
            },
            "auto_advance": 0,
            "idempotency_key": "scope-spoof",
        }
        status, created = self.request("/api/runs", "POST", body)
        self.assertEqual(status, 201, created)
        self.assertEqual(
            created["projection"]["scientific_scope"],
            "prediction_demo_non_causal",
        )


if __name__ == "__main__":
    unittest.main()
