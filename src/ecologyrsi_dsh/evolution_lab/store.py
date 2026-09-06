"""Small durable SQLite ledger owned by the external evolution plugin."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..core.models import utc_now
from .evaluator import EvaluationReport, PromotionDecision
from .genome import PluginGenome


class EvolutionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS plugin_genomes (
                    digest TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS plugin_evaluations (
                    digest TEXT NOT NULL,
                    cohort_id TEXT NOT NULL,
                    report TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (digest, cohort_id),
                    FOREIGN KEY (digest) REFERENCES plugin_genomes(digest)
                );
                CREATE TABLE IF NOT EXISTS plugin_transitions (
                    idempotency_key TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    from_digest TEXT,
                    to_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def put_genome(self, genome: PluginGenome, status: str = "exploratory") -> None:
        payload = json.dumps(genome.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO plugin_genomes(digest,payload,status,created_at) VALUES(?,?,?,?)",
                (genome.digest, payload, status, utc_now()),
            )

    def set_status(self, digest: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE plugin_genomes SET status=? WHERE digest=?", (status, digest))

    def get_genome(self, digest: str) -> PluginGenome:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM plugin_genomes WHERE digest=?", (digest,)).fetchone()
        if row is None:
            raise KeyError(f"unknown Genome: {digest}")
        return PluginGenome.from_dict(json.loads(row["payload"]))

    def get_status(self, digest: str) -> str:
        with self._connect() as connection:
            row = connection.execute("SELECT status FROM plugin_genomes WHERE digest=?", (digest,)).fetchone()
        if row is None:
            raise KeyError(f"unknown Genome: {digest}")
        return str(row["status"])

    def put_evaluation(self, genome: PluginGenome, cohort_id: str, report: EvaluationReport, decision: PromotionDecision) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO plugin_evaluations(digest,cohort_id,report,decision,created_at) VALUES(?,?,?,?,?)",
                (genome.digest, cohort_id, json.dumps(report.to_dict(), sort_keys=True), json.dumps(decision.to_dict(), sort_keys=True), utc_now()),
            )
            if decision.status == "certification_eligible":
                connection.execute("UPDATE plugin_genomes SET status='eligible' WHERE digest=?", (genome.digest,))
            elif decision.status == "search_winner":
                connection.execute("UPDATE plugin_genomes SET status='search_winner' WHERE digest=?", (genome.digest,))

    def record_transition(self, action: str, from_digest: str | None, to_digest: str, idempotency_key: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO plugin_transitions(idempotency_key,action,from_digest,to_digest,created_at) VALUES(?,?,?,?,?)",
                (idempotency_key, action, from_digest, to_digest, utc_now()),
            )
        return cursor.rowcount == 1

    def incumbent_digest(self) -> str | None:
        with self._connect() as connection:
            row = connection.execute("SELECT digest FROM plugin_genomes WHERE status='incumbent' ORDER BY rowid DESC LIMIT 1").fetchone()
        return None if row is None else str(row["digest"])

    def latest_digest(self) -> str:
        with self._connect() as connection:
            row = connection.execute("SELECT digest FROM plugin_genomes ORDER BY rowid DESC LIMIT 1").fetchone()
        if row is None:
            raise KeyError("no Genome has been recorded")
        return str(row["digest"])
