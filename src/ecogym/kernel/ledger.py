"""Append-only SQLite ledger with a hash chain and a sealed-exposure registry.

Each event stores its run, kind, UTC time and JSON body. ``hash`` chains the
event to its predecessor (``sha256(prev + canonical event)``), so ``verify``
detects any edited, removed or reordered row. The ``exposures`` table is the
one place that records which method has seen which sealed partition; its
primary key makes "evaluate once" atomic across processes.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .canonical import canonical_json, jsonable

SCHEMA_VERSION = "ecogym.ledger/1"
_GENESIS = "0" * 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    at TEXT NOT NULL,
    body TEXT NOT NULL,
    prev TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS events_by_run ON events (run_id, seq);
CREATE TABLE IF NOT EXISTS exposures (
    key TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    stage TEXT NOT NULL,
    method TEXT NOT NULL,
    run_id TEXT NOT NULL,
    at TEXT NOT NULL
);
"""


class LedgerError(RuntimeError):
    """The ledger is corrupt or an append was refused."""


@dataclass(frozen=True, slots=True)
class Event:
    seq: int
    run_id: str
    kind: str
    at: str
    body: dict[str, Any]
    hash: str


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _event_hash(prev: str, run_id: str, kind: str, at: str, body: str) -> str:
    material = prev + "\n" + canonical_json({"run_id": run_id, "kind": kind, "at": at, "body": json.loads(body)})
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class Ledger:
    """One SQLite file holding any number of runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, isolation_level=None, timeout=30.0)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        row = self._db.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone()
        if row is None:
            self._db.execute("INSERT INTO meta (key, value) VALUES ('schema', ?)", (SCHEMA_VERSION,))
        elif row[0] != SCHEMA_VERSION:
            raise LedgerError(f"{self.path} uses ledger schema {row[0]}, expected {SCHEMA_VERSION}")

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    def append(self, run_id: str, kind: str, body: Any) -> Event:
        text = canonical_json(jsonable(body))
        at = _now()
        with self._transaction() as db:
            row = db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else _GENESIS
            event_hash = _event_hash(prev, run_id, kind, at, text)
            cursor = db.execute(
                "INSERT INTO events (run_id, kind, at, body, prev, hash) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, kind, at, text, prev, event_hash),
            )
            seq = int(cursor.lastrowid)
        return Event(seq, run_id, kind, at, json.loads(text), event_hash)

    def events(self, run_id: str | None = None, kind: str | None = None) -> list[Event]:
        query = "SELECT seq, run_id, kind, at, body, hash FROM events"
        clauses, values = [], []
        if run_id is not None:
            clauses.append("run_id = ?")
            values.append(run_id)
        if kind is not None:
            clauses.append("kind = ?")
            values.append(kind)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY seq"
        return [
            Event(seq, run, event_kind, at, json.loads(body), event_hash)
            for seq, run, event_kind, at, body, event_hash in self._db.execute(query, values)
        ]

    def runs(self) -> list[str]:
        rows = self._db.execute("SELECT run_id, MIN(seq) FROM events GROUP BY run_id ORDER BY MIN(seq)")
        return [run_id for run_id, _ in rows]

    def verify(self) -> int:
        """Re-derive the hash chain; return the number of events checked."""

        prev = _GENESIS
        count = 0
        for seq, run_id, kind, at, body, stored_prev, stored_hash in self._db.execute(
            "SELECT seq, run_id, kind, at, body, prev, hash FROM events ORDER BY seq"
        ):
            if stored_prev != prev or _event_hash(prev, run_id, kind, at, body) != stored_hash:
                raise LedgerError(f"ledger hash chain is broken at event {seq}")
            prev = stored_hash
            count += 1
        return count

    def claim_exposure(self, key: str, *, task: str, stage: str, method: str, run_id: str) -> bool:
        """Record a sealed exposure; False if this key was already claimed."""

        try:
            with self._transaction() as db:
                db.execute(
                    "INSERT INTO exposures (key, task, stage, method, run_id, at) VALUES (?, ?, ?, ?, ?, ?)",
                    (key, task, stage, method, run_id, _now()),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def exposures(self, *, task: str | None = None, method: str | None = None) -> list[dict[str, str]]:
        rows = self._db.execute("SELECT key, task, stage, method, run_id, at FROM exposures ORDER BY at")
        result = [
            dict(zip(("key", "task", "stage", "method", "run_id", "at"), row, strict=True)) for row in rows
        ]
        return [
            item
            for item in result
            if (task is None or item["task"] == task) and (method is None or item["method"] == method)
        ]


__all__ = ["SCHEMA_VERSION", "Event", "Ledger", "LedgerError"]
