"""A tiny SQLite event ledger.

The domain event stream is append-only during normal operation.  A run
projection can always be rebuilt from ``events(run_id)``; no mutable
``current_best`` file is needed.  Run archival is stored outside that stream,
and an explicitly confirmed administrative purge is the only operation that
removes run-owned events.  SQLite is used instead of a service or ORM so the
same code works in tests and in a single-process local deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

from .models import canonical_json, digest, utc_now
from .redaction import public_error_summary


SCHEMA_VERSION = 7


class CommandInProgressError(RuntimeError):
    """Raised when another worker already owns an idempotent command."""


class ConcurrentRunMutationError(RuntimeError):
    """Raised when a state-derived event loses its compare-and-swap race."""


@dataclass(frozen=True, slots=True)
class Event:
    seq: int
    event_id: str
    run_id: str
    kind: str
    payload: dict[str, Any]
    created_at: str


@dataclass(frozen=True, slots=True)
class CommandReceipt:
    command_key: str
    run_id: str
    resource_run_id: str | None
    command_kind: str
    request_digest: str
    request: dict[str, Any]
    start_seq: int
    status: str
    response: dict[str, Any] | None
    created_at: str
    completed_at: str | None


class EventLedger:
    """Append and read events from a local SQLite database."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            parent = Path(self.path).expanduser().resolve().parent
            parent.mkdir(parents=True, exist_ok=True)
        # The optional HTTP adapter serves requests on worker threads.  A
        # re-entrant lock keeps the single local connection safe there while
        # preserving the same lightweight implementation for tests.
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current_version = int(
                self._connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if current_version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema {current_version} is newer than supported {SCHEMA_VERSION}"
                )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evolution_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_evolution_events_run_seq "
                "ON evolution_events(run_id, seq)"
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS http_command_receipts (
                    command_key TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    command_kind TEXT NOT NULL,
                    request_digest TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    resource_run_id TEXT,
                    start_seq INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'completed')),
                    response_json TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_http_command_receipts_run "
                "ON http_command_receipts(run_id, created_at)"
            )
            receipt_columns = {
                str(row["name"])
                for row in self._connection.execute(
                    "PRAGMA table_info(http_command_receipts)"
                ).fetchall()
            }
            if "start_seq" not in receipt_columns:
                self._connection.execute(
                    "ALTER TABLE http_command_receipts "
                    "ADD COLUMN start_seq INTEGER NOT NULL DEFAULT 0"
                )
                # Older receipts predate the durable command cursor.  ISO-8601 UTC
                # timestamps are sortable, so this reconstructs the last event
                # that was visible when each legacy receipt was claimed.
                self._connection.execute(
                    """
                    UPDATE http_command_receipts
                    SET start_seq = COALESCE((
                        SELECT MAX(e.seq)
                        FROM evolution_events AS e
                        WHERE e.created_at <= http_command_receipts.created_at
                    ), 0)
                    """
                )
            if "resource_run_id" not in receipt_columns:
                self._connection.execute(
                    "ALTER TABLE http_command_receipts "
                    "ADD COLUMN resource_run_id TEXT"
                )
            self._backfill_command_resource_run_ids()
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_http_command_receipts_resource_run "
                "ON http_command_receipts(resource_run_id, created_at)"
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_archives (
                    run_id TEXT PRIMARY KEY,
                    archived_at TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_run_archives_time "
                "ON run_archives(archived_at, run_id)"
            )
            self._ensure_model_verifications_schema()
            self._sanitize_model_verification_errors()
            self._ensure_scientific_exposure_schema()
            if current_version < SCHEMA_VERSION:
                self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def _ensure_scientific_exposure_schema(self) -> None:
        """Create append-only selection/formal exposure identity tables."""

        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS adaptive_evidence_exposures (
                evidence_digest TEXT PRIMARY KEY,
                first_run_id TEXT NOT NULL,
                fitness_profile_digest TEXT NOT NULL,
                evidence_class TEXT NOT NULL
                    CHECK (evidence_class = 'exploratory_adaptive_data'),
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS formal_holdout_exposures (
                holdout_exposure_key TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                token_digest TEXT,
                candidate_id TEXT,
                artifact_digest TEXT,
                genome_digest TEXT,
                partition_digest TEXT,
                analysis_plan_digest TEXT,
                state TEXT NOT NULL CHECK (state IN ('reserved', 'opened', 'sealed')),
                idempotency_key TEXT NOT NULL UNIQUE,
                reserved_at TEXT NOT NULL,
                opened_at TEXT,
                sealed_at TEXT
            )
            """
        )
        formal_columns = {
            str(row["name"])
            for row in self._connection.execute(
                "PRAGMA table_info(formal_holdout_exposures)"
            ).fetchall()
        }
        for name in (
            "token_digest",
            "candidate_id",
            "artifact_digest",
            "genome_digest",
            "partition_digest",
            "analysis_plan_digest",
        ):
            if name not in formal_columns:
                self._connection.execute(
                    f"ALTER TABLE formal_holdout_exposures ADD COLUMN {name} TEXT"
                )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS formal_analysis_families (
                holdout_exposure_key TEXT NOT NULL,
                objective_family_digest TEXT NOT NULL,
                analysis_plan_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (holdout_exposure_key, objective_family_digest),
                FOREIGN KEY (holdout_exposure_key)
                    REFERENCES formal_holdout_exposures(holdout_exposure_key)
            )
            """
        )

    def _backfill_command_resource_run_ids(self) -> None:
        """Bind legacy receipts to their concrete run without text matching."""

        self._connection.execute(
            """
            UPDATE http_command_receipts
            SET resource_run_id = run_id
            WHERE resource_run_id IS NULL AND command_kind <> 'create_run'
            """
        )
        rows = self._connection.execute(
            """
            SELECT command_key, run_id, command_kind, request_json, response_json
            FROM http_command_receipts
            WHERE resource_run_id IS NULL
            """
        ).fetchall()
        bindings: list[tuple[str, str]] = []
        for row in rows:
            resource_run_id = self._receipt_resource_run_id(row)
            if resource_run_id is None and str(row["command_kind"]) == "create_run":
                resource_run_id = self._created_run_id_for_receipt(row)
            if resource_run_id is not None:
                bindings.append((resource_run_id, str(row["command_key"])))
        if bindings:
            self._connection.executemany(
                """
                UPDATE http_command_receipts
                SET resource_run_id = ?
                WHERE command_key = ? AND resource_run_id IS NULL
                """,
                bindings,
            )

    def _create_model_verifications_table(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE model_verifications (
                model_id TEXT NOT NULL,
                configuration_digest TEXT NOT NULL,
                credential_fingerprint TEXT NOT NULL,
                verified INTEGER NOT NULL CHECK (verified IN (0, 1)),
                connection_state TEXT NOT NULL
                    CHECK (connection_state IN ('configured', 'available', 'error')),
                last_checked_at TEXT,
                last_error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (
                    model_id, configuration_digest, credential_fingerprint
                )
            )
            """
        )

    def _ensure_model_verifications_schema(self) -> None:
        """Create or transactionally upgrade the credential-bound trust table."""

        rows = self._connection.execute(
            "PRAGMA table_info(model_verifications)"
        ).fetchall()
        if not rows:
            self._create_model_verifications_table()
            return
        columns = {str(row["name"]): row for row in rows}
        expected_primary_key = {
            "model_id": 1,
            "configuration_digest": 2,
            "credential_fingerprint": 3,
        }
        required_columns = {
            "model_id",
            "configuration_digest",
            "credential_fingerprint",
            "verified",
            "connection_state",
            "last_checked_at",
            "last_error",
            "updated_at",
        }
        primary_key_matches = all(
            name in columns and int(columns[name]["pk"]) == position
            for name, position in expected_primary_key.items()
        ) and all(
            int(row["pk"]) == 0 or str(row["name"]) in expected_primary_key
            for row in rows
        )
        if required_columns.issubset(columns) and primary_key_matches:
            return

        # SQLite cannot alter a primary key in place.  The unique temporary
        # name keeps the migration atomic without risking an unrelated backup
        # table left by an operator.
        legacy_name = f"model_verifications_legacy_{uuid4().hex}"
        self._connection.execute(
            f'ALTER TABLE model_verifications RENAME TO "{legacy_name}"'
        )
        self._create_model_verifications_table()

        # Rows without a credential fingerprint are intentionally not restored:
        # they cannot prove that the verified credential is still the one in use.
        identity_columns = {
            "model_id",
            "configuration_digest",
            "credential_fingerprint",
            "verified",
        }
        if identity_columns.issubset(columns):
            state_expression = (
                "CASE WHEN connection_state IN ('configured', 'available', 'error') "
                "THEN connection_state ELSE 'configured' END"
                if "connection_state" in columns
                else "'configured'"
            )
            checked_expression = (
                "last_checked_at" if "last_checked_at" in columns else "NULL"
            )
            error_expression = "last_error" if "last_error" in columns else "NULL"
            updated_expression = (
                "COALESCE(updated_at, CURRENT_TIMESTAMP)"
                if "updated_at" in columns
                else "CURRENT_TIMESTAMP"
            )
            self._connection.execute(
                f"""
                INSERT OR REPLACE INTO model_verifications (
                    model_id, configuration_digest, credential_fingerprint,
                    verified, connection_state, last_checked_at, last_error,
                    updated_at
                )
                SELECT
                    model_id, configuration_digest, credential_fingerprint,
                    verified, {state_expression}, {checked_expression},
                    {error_expression}, {updated_expression}
                FROM "{legacy_name}"
                WHERE typeof(model_id) = 'text' AND trim(model_id) <> ''
                    AND typeof(configuration_digest) = 'text'
                    AND trim(configuration_digest) <> ''
                    AND typeof(credential_fingerprint) = 'text'
                    AND trim(credential_fingerprint) <> ''
                    AND verified IN (0, 1)
                """
            )
        self._connection.execute(f'DROP TABLE "{legacy_name}"')

    def _sanitize_model_verification_errors(self) -> None:
        """Remove credential-like text from mutable legacy health records."""

        rows = self._connection.execute(
            """
            SELECT model_id, configuration_digest, credential_fingerprint, last_error
            FROM model_verifications
            WHERE last_error IS NOT NULL
            """
        ).fetchall()
        updates: list[tuple[str | None, str, str, str]] = []
        for row in rows:
            raw_error = str(row["last_error"])
            safe_error = public_error_summary(raw_error)
            if safe_error != raw_error:
                updates.append(
                    (
                        safe_error,
                        str(row["model_id"]),
                        str(row["configuration_digest"]),
                        str(row["credential_fingerprint"]),
                    )
                )
        if updates:
            self._connection.executemany(
                """
                UPDATE model_verifications
                SET last_error = ?
                WHERE model_id = ? AND configuration_digest = ?
                    AND credential_fingerprint = ?
                """,
                updates,
            )

    def append(
        self,
        run_id: str,
        kind: str,
        payload: Mapping[str, Any],
        *,
        event_id: str | None = None,
        created_at: str | None = None,
        expected_run_seq: int | None = None,
    ) -> Event:
        """Append one event, returning an existing event for duplicate IDs.

        The optional event ID makes retries idempotent.  Payloads are validated
        before touching SQLite, keeping malformed events out of the ledger.
        """

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("kind must be a non-empty string")
        payload_dict = dict(payload)
        payload_json = canonical_json(payload_dict)
        normalized_payload = json.loads(payload_json)
        event_id = event_id or str(uuid4())
        created_at = created_at or utc_now()
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        if not isinstance(created_at, str) or not created_at.strip():
            raise ValueError("created_at must be a non-empty string")
        if expected_run_seq is not None and (
            isinstance(expected_run_seq, bool)
            or not isinstance(expected_run_seq, int)
            or expected_run_seq < 0
        ):
            raise ValueError("expected_run_seq must be a non-negative integer")

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if expected_run_seq is not None:
                    current = int(
                        self._connection.execute(
                            """
                            SELECT COALESCE(MAX(seq), 0) AS seq
                            FROM evolution_events WHERE run_id = ?
                            """,
                            (run_id.strip(),),
                        ).fetchone()["seq"]
                    )
                    if current != expected_run_seq:
                        raise ConcurrentRunMutationError(
                            f"run {run_id.strip()} changed while appending {kind.strip()}"
                        )
                self._connection.execute(
                """
                INSERT OR IGNORE INTO evolution_events
                    (event_id, run_id, kind, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_id, run_id.strip(), kind.strip(), payload_json, created_at),
                )
                row = self._connection.execute(
                """
                SELECT seq, event_id, run_id, kind, payload_json, created_at
                FROM evolution_events WHERE event_id = ?
                """,
                (event_id,),
                ).fetchone()
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        if row is None:  # pragma: no cover - defensive against a corrupt DB
            raise RuntimeError("event insert did not produce a row")
        event = self._row_to_event(row)
        if (
            event.run_id != run_id.strip()
            or event.kind != kind.strip()
            or event.payload != normalized_payload
        ):
            raise ValueError(f"event_id already belongs to a different event: {event_id}")
        return event

    def append_many(
        self,
        run_id: str,
        entries: Sequence[
            tuple[str, Mapping[str, Any], str | None]
        ],
        *,
        created_at: str | None = None,
        expected_run_seq: int | None = None,
    ) -> tuple[Event, ...]:
        """Atomically append an ordered group of idempotent run events."""

        run_id = self._required_text(run_id, "run_id")
        if not entries:
            raise ValueError("entries must not be empty")
        occurred_at = created_at or utc_now()
        if not isinstance(occurred_at, str) or not occurred_at.strip():
            raise ValueError("created_at must be a non-empty string")
        if expected_run_seq is not None and (
            isinstance(expected_run_seq, bool)
            or not isinstance(expected_run_seq, int)
            or expected_run_seq < 0
        ):
            raise ValueError("expected_run_seq must be a non-negative integer")
        normalized: list[tuple[str, dict[str, Any], str, str]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise TypeError(
                    f"entries[{index}] must be a (kind, payload, event_id) tuple"
                )
            kind, payload, event_id = entry
            kind = self._required_text(kind, f"entries[{index}].kind")
            payload_dict = dict(payload)
            payload_json = canonical_json(payload_dict)
            resolved_event_id = event_id or str(uuid4())
            resolved_event_id = self._required_text(
                resolved_event_id, f"entries[{index}].event_id"
            )
            normalized.append(
                (kind, payload_dict, payload_json, resolved_event_id)
            )

        appended: list[Event] = []
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if expected_run_seq is not None:
                    current = int(
                        self._connection.execute(
                            """
                            SELECT COALESCE(MAX(seq), 0) AS seq
                            FROM evolution_events WHERE run_id = ?
                            """,
                            (run_id,),
                        ).fetchone()["seq"]
                    )
                    if current != expected_run_seq:
                        raise ConcurrentRunMutationError(
                            f"run {run_id} changed while appending event batch"
                        )
                for kind, payload_dict, payload_json, event_id in normalized:
                    self._connection.execute(
                        """
                        INSERT OR IGNORE INTO evolution_events
                            (event_id, run_id, kind, payload_json, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (event_id, run_id, kind, payload_json, occurred_at),
                    )
                    row = self._connection.execute(
                        """
                        SELECT seq, event_id, run_id, kind, payload_json, created_at
                        FROM evolution_events WHERE event_id = ?
                        """,
                        (event_id,),
                    ).fetchone()
                    if row is None:  # pragma: no cover - defensive DB guard
                        raise RuntimeError("event insert did not produce a row")
                    event = self._row_to_event(row)
                    if (
                        event.run_id != run_id
                        or event.kind != kind
                        or event.payload != payload_dict
                    ):
                        raise ValueError(
                            "event_id already belongs to a different event: "
                            + event_id
                        )
                    appended.append(event)
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return tuple(appended)

    def events(
        self,
        run_id: str | None = None,
        *,
        after_seq: int = 0,
    ) -> tuple[Event, ...]:
        """Return events in append order, optionally scoped to a run."""

        if isinstance(after_seq, bool) or not isinstance(after_seq, int) or after_seq < 0:
            raise ValueError("after_seq must be a non-negative integer")
        with self._lock:
            if run_id is None:
                rows = self._connection.execute(
                """
                SELECT seq, event_id, run_id, kind, payload_json, created_at
                FROM evolution_events WHERE seq > ? ORDER BY seq
                """,
                (after_seq,),
                ).fetchall()
            else:
                rows = self._connection.execute(
                """
                SELECT seq, event_id, run_id, kind, payload_json, created_at
                FROM evolution_events WHERE run_id = ? AND seq > ? ORDER BY seq
                """,
                (run_id, after_seq),
                ).fetchall()
        return tuple(self._row_to_event(row) for row in rows)

    def iter_events(self, run_id: str | None = None, *, after_seq: int = 0) -> Iterator[Event]:
        yield from self.events(run_id, after_seq=after_seq)

    def count(self, run_id: str | None = None) -> int:
        with self._lock:
            if run_id is None:
                row = self._connection.execute("SELECT COUNT(*) AS n FROM evolution_events").fetchone()
            else:
                row = self._connection.execute(
                    "SELECT COUNT(*) AS n FROM evolution_events WHERE run_id = ?", (run_id,)
                ).fetchone()
        return int(row["n"])

    def latest_seq(self) -> int:
        """Return the global append-only event cursor."""

        with self._lock:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(seq), 0) AS seq FROM evolution_events"
            ).fetchone()
        return int(row["seq"])

    def run_ids(self, *, include_archived: bool = True) -> tuple[str, ...]:
        """Return distinct run IDs in first-seen order.

        Existing internal callers retain the complete audit view.  List APIs
        opt into ``include_archived=False`` so archived runs disappear from the
        normal work queue without losing their evidence.
        """

        if not isinstance(include_archived, bool):
            raise ValueError("include_archived must be a bool")
        with self._lock:
            if include_archived:
                rows = self._connection.execute(
                    "SELECT run_id, MIN(seq) AS first_seq FROM evolution_events "
                    "GROUP BY run_id ORDER BY first_seq"
                ).fetchall()
            else:
                rows = self._connection.execute(
                    """
                    SELECT e.run_id, MIN(e.seq) AS first_seq
                    FROM evolution_events AS e
                    WHERE NOT EXISTS (
                        SELECT 1 FROM run_archives AS a WHERE a.run_id = e.run_id
                    )
                    GROUP BY e.run_id
                    ORDER BY first_seq
                    """
                ).fetchall()
        return tuple(str(row["run_id"]) for row in rows)

    def archived_at(self, run_id: str) -> str | None:
        run_id = self._required_text(run_id, "run_id")
        with self._lock:
            row = self._connection.execute(
                "SELECT archived_at FROM run_archives WHERE run_id = ?", (run_id,)
            ).fetchone()
        return str(row["archived_at"]) if row is not None else None

    def archived_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS n
                FROM run_archives AS a
                WHERE EXISTS (
                    SELECT 1 FROM evolution_events AS e WHERE e.run_id = a.run_id
                )
                """
            ).fetchone()
        return int(row["n"])

    def archive_run(self, run_id: str) -> str:
        """Mark an existing run as archived without changing its event stream."""

        run_id = self._required_text(run_id, "run_id")
        archived_at = utc_now()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                exists = self._connection.execute(
                    "SELECT 1 FROM evolution_events WHERE run_id = ? LIMIT 1",
                    (run_id,),
                ).fetchone()
                if exists is None:
                    raise KeyError(f"unknown run: {run_id}")
                self._connection.execute(
                    "INSERT OR IGNORE INTO run_archives(run_id, archived_at) VALUES (?, ?)",
                    (run_id, archived_at),
                )
                row = self._connection.execute(
                    "SELECT archived_at FROM run_archives WHERE run_id = ?", (run_id,)
                ).fetchone()
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        if row is None:  # pragma: no cover - guarded by the transaction
            raise RuntimeError("run archive did not produce a row")
        return str(row["archived_at"])

    def restore_run(self, run_id: str) -> bool:
        """Remove a run's archive marker, returning whether one existed."""

        run_id = self._required_text(run_id, "run_id")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                exists = self._connection.execute(
                    "SELECT 1 FROM evolution_events WHERE run_id = ? LIMIT 1",
                    (run_id,),
                ).fetchone()
                if exists is None:
                    raise KeyError(f"unknown run: {run_id}")
                cursor = self._connection.execute(
                    "DELETE FROM run_archives WHERE run_id = ?", (run_id,)
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return cursor.rowcount > 0

    def purge_run(
        self,
        run_id: str,
        *,
        confirmation: str,
        terminal_status: str,
    ) -> dict[str, int]:
        """Permanently remove one archived terminal run and owned receipts.

        The caller must both prove the projected terminal status and repeat the
        exact run ID.  Requiring the archive marker makes destructive cleanup a
        deliberate second step after the reversible default action.
        """

        run_id = self._required_text(run_id, "run_id")
        if confirmation != run_id:
            raise ValueError("confirmation must exactly match run_id")
        if terminal_status not in {"completed", "cancelled", "failed"}:
            raise ValueError("only a terminal run can be permanently deleted")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                event_row = self._connection.execute(
                    "SELECT COUNT(*) AS n FROM evolution_events WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                event_count = int(event_row["n"])
                if event_count == 0:
                    raise KeyError(f"unknown run: {run_id}")
                archive_row = self._connection.execute(
                    "SELECT 1 FROM run_archives WHERE run_id = ?", (run_id,)
                ).fetchone()
                if archive_row is None:
                    raise ValueError("运行必须先归档，才能永久删除")

                receipt_rows = self._connection.execute(
                    """
                    SELECT command_key, run_id, command_kind, request_json,
                           response_json, resource_run_id
                    FROM http_command_receipts
                    """
                ).fetchall()
                creation_key = self._run_creation_idempotency_key(run_id)
                receipt_keys = [
                    str(row["command_key"])
                    for row in receipt_rows
                    if row["resource_run_id"] == run_id
                    or row["run_id"] == run_id
                    or self._receipt_resource_run_id(row) == run_id
                    or (
                        creation_key is not None
                        and str(row["command_kind"]) == "create_run"
                        and self._receipt_idempotency_key(row) == creation_key
                    )
                ]
                if receipt_keys:
                    self._connection.executemany(
                        "DELETE FROM http_command_receipts WHERE command_key = ?",
                        ((key,) for key in receipt_keys),
                    )
                archive_cursor = self._connection.execute(
                    "DELETE FROM run_archives WHERE run_id = ?", (run_id,)
                )
                event_cursor = self._connection.execute(
                    "DELETE FROM evolution_events WHERE run_id = ?", (run_id,)
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return {
            "events": event_cursor.rowcount,
            "command_receipts": len(receipt_keys),
            "archive_markers": archive_cursor.rowcount,
        }

    @property
    def schema_version(self) -> int:
        with self._lock:
            return int(self._connection.execute("PRAGMA user_version").fetchone()[0])

    def integrity_check(self) -> str:
        with self._lock:
            row = self._connection.execute("PRAGMA integrity_check").fetchone()
        return str(row[0])

    def begin_command(
        self,
        command_key: str,
        run_id: str,
        command_kind: str,
        request: Mapping[str, Any],
        *,
        resume_pending: bool = False,
    ) -> CommandReceipt | None:
        """Claim a command or return its previously completed receipt.

        ``None`` means this caller owns the newly persisted pending claim.  A
        completed receipt is safe to return after a process restart.  A
        pending claim is never stolen automatically because the prior process
        may have written domain events before it stopped.  The single-process
        HTTP adapter may explicitly resume it while holding its mutation lock.
        """

        command_key = self._required_text(command_key, "command_key")
        run_id = self._required_text(run_id, "run_id")
        command_kind = self._required_text(command_kind, "command_kind")
        request_dict = dict(request)
        request_json = canonical_json(request_dict)
        request_digest = digest(request_dict)
        created_at = utc_now()
        resource_run_id = None if command_kind == "create_run" else run_id

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT * FROM http_command_receipts WHERE command_key = ?",
                    (command_key,),
                ).fetchone()
                if row is None:
                    start_seq = int(
                        self._connection.execute(
                            "SELECT COALESCE(MAX(seq), 0) AS seq FROM evolution_events"
                        ).fetchone()["seq"]
                    )
                    self._connection.execute(
                        """
                        INSERT INTO http_command_receipts
                            (command_key, run_id, command_kind, request_digest,
                             request_json, resource_run_id, start_seq, status,
                             response_json, created_at, completed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?, NULL)
                        """,
                        (
                            command_key,
                            run_id,
                            command_kind,
                            request_digest,
                            request_json,
                            resource_run_id,
                            start_seq,
                            created_at,
                        ),
                    )
                    self._connection.commit()
                    return None
                receipt = self._row_to_command_receipt(row)
                if (
                    receipt.run_id != run_id
                    or receipt.command_kind != command_kind
                    or receipt.request_digest != request_digest
                ):
                    raise ValueError("idempotency key already belongs to a different command")
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

        if receipt.status == "pending" and not resume_pending:
            raise CommandInProgressError(
                "idempotent command is pending; run doctor before retrying"
            )
        if receipt.status == "pending":
            return None
        return receipt

    def abandon_command(self, command_key: str) -> None:
        """Remove a new pending claim that wrote no domain events.

        Callers must establish the no-write condition while holding their
        mutation lock.  A completed receipt is immutable and is never removed.
        """

        command_key = self._required_text(command_key, "command_key")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT status FROM http_command_receipts WHERE command_key = ?",
                    (command_key,),
                ).fetchone()
                if row is None:
                    self._connection.commit()
                    return
                if str(row["status"]) != "pending":
                    raise ValueError("only a pending command can be abandoned")
                self._connection.execute(
                    "DELETE FROM http_command_receipts WHERE command_key = ? AND status = 'pending'",
                    (command_key,),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def complete_command(
        self,
        command_key: str,
        response: Mapping[str, Any],
    ) -> CommandReceipt:
        """Persist the immutable response for a claimed command."""

        command_key = self._required_text(command_key, "command_key")
        response_dict = dict(response)
        response_json = canonical_json(response_dict)
        response_run_id = self._resource_run_id_from_payload(response_dict)
        completed_at = utc_now()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT * FROM http_command_receipts WHERE command_key = ?",
                    (command_key,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown command claim: {command_key}")
                receipt = self._row_to_command_receipt(row)
                bound_run_id = response_run_id or receipt.resource_run_id
                if bound_run_id is None and receipt.command_kind != "create_run":
                    bound_run_id = receipt.run_id
                if (
                    receipt.resource_run_id is not None
                    and bound_run_id is not None
                    and receipt.resource_run_id != bound_run_id
                ):
                    raise ValueError("command receipt belongs to a different run")
                if receipt.status == "completed":
                    if receipt.response != response_dict:
                        raise ValueError("completed command response is immutable")
                    if receipt.resource_run_id is None and bound_run_id is not None:
                        self._connection.execute(
                            """
                            UPDATE http_command_receipts
                            SET resource_run_id = ?
                            WHERE command_key = ? AND resource_run_id IS NULL
                            """,
                            (bound_run_id, command_key),
                        )
                    self._connection.commit()
                    row = self._connection.execute(
                        "SELECT * FROM http_command_receipts WHERE command_key = ?",
                        (command_key,),
                    ).fetchone()
                    if row is None:  # pragma: no cover - guarded by the transaction
                        raise RuntimeError("completed command receipt disappeared")
                    return self._row_to_command_receipt(row)
                self._connection.execute(
                    """
                    UPDATE http_command_receipts
                    SET status = 'completed', response_json = ?, completed_at = ?,
                        resource_run_id = COALESCE(resource_run_id, ?)
                    WHERE command_key = ? AND status = 'pending'
                    """,
                    (response_json, completed_at, bound_run_id, command_key),
                )
                row = self._connection.execute(
                    "SELECT * FROM http_command_receipts WHERE command_key = ?",
                    (command_key,),
                ).fetchone()
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        if row is None:  # pragma: no cover - guarded by the transaction
            raise RuntimeError("command completion did not produce a row")
        return self._row_to_command_receipt(row)

    def bind_command_resource_run(
        self, command_key: str, run_id: str
    ) -> CommandReceipt:
        """Idempotently bind a claimed command to the run it created or changed."""

        command_key = self._required_text(command_key, "command_key")
        run_id = self._required_text(run_id, "run_id")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT * FROM http_command_receipts WHERE command_key = ?",
                    (command_key,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown command claim: {command_key}")
                current = row["resource_run_id"]
                if current is not None and str(current) != run_id:
                    raise ValueError("command receipt belongs to a different run")
                if current is None:
                    self._connection.execute(
                        """
                        UPDATE http_command_receipts
                        SET resource_run_id = ?
                        WHERE command_key = ? AND resource_run_id IS NULL
                        """,
                        (run_id, command_key),
                    )
                    row = self._connection.execute(
                        "SELECT * FROM http_command_receipts WHERE command_key = ?",
                        (command_key,),
                    ).fetchone()
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        if row is None:  # pragma: no cover - guarded by the transaction
            raise RuntimeError("command binding did not produce a row")
        return self._row_to_command_receipt(row)

    def command_receipt(self, command_key: str) -> CommandReceipt | None:
        command_key = self._required_text(command_key, "command_key")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM http_command_receipts WHERE command_key = ?",
                (command_key,),
            ).fetchone()
        return self._row_to_command_receipt(row) if row is not None else None

    def command_count(self, *, status: str | None = None) -> int:
        with self._lock:
            if status is None:
                row = self._connection.execute(
                    "SELECT COUNT(*) AS n FROM http_command_receipts"
                ).fetchone()
            else:
                if status not in {"pending", "completed"}:
                    raise ValueError("status must be pending or completed")
                row = self._connection.execute(
                    "SELECT COUNT(*) AS n FROM http_command_receipts WHERE status = ?",
                    (status,),
                ).fetchone()
        return int(row["n"])

    def pending_command_keys(self) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT command_key FROM http_command_receipts "
                "WHERE status = 'pending' ORDER BY created_at, command_key"
            ).fetchall()
        return tuple(str(row["command_key"]) for row in rows)

    def model_verification(
        self,
        model_id: str,
        configuration_digest: str,
        credential_fingerprint: str,
    ) -> dict[str, Any] | None:
        """Load verification only for the exact callable and credential."""

        model_id = self._required_text(model_id, "model_id")
        configuration_digest = self._required_text(
            configuration_digest, "configuration_digest"
        )
        credential_fingerprint = self._required_text(
            credential_fingerprint, "credential_fingerprint"
        )
        with self._lock:
            row = self._connection.execute(
                """
                SELECT verified, connection_state, last_checked_at, last_error, updated_at
                FROM model_verifications
                WHERE model_id = ? AND configuration_digest = ?
                    AND credential_fingerprint = ?
                """,
                (model_id, configuration_digest, credential_fingerprint),
            ).fetchone()
        if row is None:
            return None
        return {
            "verified": bool(row["verified"]),
            "state": str(row["connection_state"]),
            "last_checked_at": (
                str(row["last_checked_at"])
                if row["last_checked_at"] is not None
                else None
            ),
            "last_error": (
                public_error_summary(str(row["last_error"]))
                if row["last_error"] is not None
                else None
            ),
            "updated_at": str(row["updated_at"]),
        }

    def record_model_verification(
        self,
        model_id: str,
        configuration_digest: str,
        credential_fingerprint: str,
        *,
        verified: bool | None,
        state: str,
        last_checked_at: str | None,
        last_error: str | None,
    ) -> None:
        """Persist trust and health without letting stale health revoke trust.

        ``verified=None`` is a connection-health update.  It atomically keeps
        any existing explicit verification decision made by another process.
        A boolean is reserved for the result of an explicit verification.
        """

        model_id = self._required_text(model_id, "model_id")
        configuration_digest = self._required_text(
            configuration_digest, "configuration_digest"
        )
        credential_fingerprint = self._required_text(
            credential_fingerprint, "credential_fingerprint"
        )
        if verified is not None and not isinstance(verified, bool):
            raise ValueError("verified must be a boolean or None")
        if state not in {"configured", "available", "error"}:
            raise ValueError("state must be configured, available, or error")
        if last_checked_at is not None:
            last_checked_at = self._required_text(last_checked_at, "last_checked_at")
        if last_error is not None:
            last_error = public_error_summary(
                self._required_text(last_error, "last_error")
            )
        updated_at = utc_now()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    """
                    INSERT INTO model_verifications (
                        model_id, configuration_digest, credential_fingerprint,
                        verified, connection_state, last_checked_at, last_error,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(
                        model_id, configuration_digest, credential_fingerprint
                    ) DO UPDATE SET
                        verified = CASE
                            WHEN ? THEN excluded.verified
                            ELSE model_verifications.verified
                        END,
                        connection_state = excluded.connection_state,
                        last_checked_at = excluded.last_checked_at,
                        last_error = excluded.last_error,
                        updated_at = excluded.updated_at
                    """,
                    (
                        model_id,
                        configuration_digest,
                        credential_fingerprint,
                        int(bool(verified)),
                        state,
                        last_checked_at,
                        last_error,
                        updated_at,
                        int(verified is not None),
                    ),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "EventLedger":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        payload = json.loads(row["payload_json"])
        if not isinstance(payload, dict):  # pragma: no cover - guarded by append
            raise ValueError("event payload must be an object")
        return Event(
            seq=int(row["seq"]),
            event_id=str(row["event_id"]),
            run_id=str(row["run_id"]),
            kind=str(row["kind"]),
            payload=payload,
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _required_text(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    @classmethod
    def _resource_run_id_from_payload(cls, payload: Mapping[str, Any]) -> str | None:
        """Read a run ID only from documented response/request object fields."""

        candidates: list[Any] = [payload]
        candidates.extend(
            payload.get(key) for key in ("projection", "run_projection")
        )
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            value = candidate.get("run_id", candidate.get("id"))
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @classmethod
    def _receipt_resource_run_id(cls, row: sqlite3.Row) -> str | None:
        """Resolve old create receipts structurally, never with SQL LIKE."""

        response_json = row["response_json"]
        if response_json is not None:
            try:
                response = json.loads(response_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("command response JSON is malformed") from exc
            if not isinstance(response, dict):
                raise ValueError("command response must be an object")
            response_run_id = cls._resource_run_id_from_payload(response)
            if response_run_id is not None:
                return response_run_id

        try:
            request = json.loads(row["request_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("command request JSON is malformed") from exc
        if not isinstance(request, dict):
            raise ValueError("command request must be an object")
        request_run_id = cls._resource_run_id_from_payload(request)
        if request_run_id is not None:
            return request_run_id
        if str(row["command_kind"]) != "create_run":
            run_id = str(row["run_id"]).strip()
            return run_id or None
        return None

    @staticmethod
    def _receipt_idempotency_key(row: sqlite3.Row) -> str | None:
        try:
            request = json.loads(row["request_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("command request JSON is malformed") from exc
        if not isinstance(request, dict):
            raise ValueError("command request must be an object")
        value = request.get("idempotency_key")
        return value.strip() if isinstance(value, str) and value.strip() else None

    def _run_creation_idempotency_key(self, run_id: str) -> str | None:
        row = self._connection.execute(
            """
            SELECT payload_json
            FROM evolution_events
            WHERE run_id = ? AND kind = 'RunCreated'
            ORDER BY seq
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("RunCreated payload JSON is malformed") from exc
        if not isinstance(payload, dict):
            raise ValueError("RunCreated payload must be an object")
        manifest = payload.get("task_manifest")
        metadata = manifest.get("metadata") if isinstance(manifest, dict) else None
        value = metadata.get("idempotency_key") if isinstance(metadata, dict) else None
        return value.strip() if isinstance(value, str) and value.strip() else None

    def _created_run_id_for_receipt(self, row: sqlite3.Row) -> str | None:
        idempotency_key = self._receipt_idempotency_key(row)
        if idempotency_key is None:
            return None
        event_rows = self._connection.execute(
            """
            SELECT run_id, payload_json
            FROM evolution_events
            WHERE kind = 'RunCreated'
            ORDER BY seq
            """
        ).fetchall()
        matches = [
            str(event_row["run_id"])
            for event_row in event_rows
            if self._event_creation_idempotency_key(event_row) == idempotency_key
        ]
        if len(matches) > 1:
            raise ValueError("create command idempotency key belongs to multiple runs")
        return matches[0] if matches else None

    @staticmethod
    def _event_creation_idempotency_key(row: sqlite3.Row) -> str | None:
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("RunCreated payload JSON is malformed") from exc
        if not isinstance(payload, dict):
            raise ValueError("RunCreated payload must be an object")
        manifest = payload.get("task_manifest")
        metadata = manifest.get("metadata") if isinstance(manifest, dict) else None
        value = metadata.get("idempotency_key") if isinstance(metadata, dict) else None
        return value.strip() if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _row_to_command_receipt(row: sqlite3.Row) -> CommandReceipt:
        request = json.loads(row["request_json"])
        if not isinstance(request, dict):  # pragma: no cover - database invariant
            raise ValueError("command request must be an object")
        response_json = row["response_json"]
        response = json.loads(response_json) if response_json is not None else None
        if response is not None and not isinstance(response, dict):  # pragma: no cover
            raise ValueError("command response must be an object")
        return CommandReceipt(
            command_key=str(row["command_key"]),
            run_id=str(row["run_id"]),
            resource_run_id=(
                str(row["resource_run_id"])
                if row["resource_run_id"] is not None
                else None
            ),
            command_kind=str(row["command_kind"]),
            request_digest=str(row["request_digest"]),
            request=request,
            start_seq=int(row["start_seq"]),
            status=str(row["status"]),
            response=response,
            created_at=str(row["created_at"]),
            completed_at=(str(row["completed_at"]) if row["completed_at"] is not None else None),
        )
