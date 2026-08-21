"""Durable, cross-run scientific evidence exposure registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from .ledger import EventLedger
from .models import utc_now
from .models import digest


class FormalExposureAlreadyUsed(RuntimeError):
    """A raw formal holdout was already reserved, opened, or sealed."""


@dataclass(frozen=True, slots=True)
class FormalStageToken:
    holdout_exposure_key: str
    token_digest: str
    run_id: str
    stage: str
    candidate_id: str
    artifact_digest: str
    genome_digest: str
    partition_digest: str
    objective_family_digest: str
    analysis_plan_digest: str
    idempotency_key: str


def raw_holdout_exposure_key(
    *,
    dataset_digest: str,
    split_manifest_digest: str,
    episode_id: str,
    stage: str,
    stage_partition_digest: str,
) -> str:
    """Raw uniqueness intentionally excludes objective and statistical choices."""

    return digest(
        {
            "dataset_digest": _sha256(dataset_digest, "dataset_digest"),
            "split_manifest_digest": _sha256(
                split_manifest_digest, "split_manifest_digest"
            ),
            "episode_id": _text(episode_id, "episode_id"),
            "stage": _text(stage, "stage"),
            "stage_partition_digest": _sha256(
                stage_partition_digest, "stage_partition_digest"
            ),
        }
    )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _sha256(value: Any, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


class ScientificExposureRegistry:
    """Records selection exposure without granting any formal evidence status."""

    def __init__(self, ledger: EventLedger) -> None:
        if not isinstance(ledger, EventLedger):
            raise TypeError("ledger must be an EventLedger")
        self.ledger = ledger

    def record_adaptive_evidence(
        self,
        *,
        run_id: str,
        evidence_digest: str,
        fitness_profile_digest: str,
    ) -> dict[str, Any]:
        run_id = _text(run_id, "run_id")
        evidence_digest = _sha256(evidence_digest, "evidence_digest")
        fitness_profile_digest = _sha256(
            fitness_profile_digest, "fitness_profile_digest"
        )
        created_at = utc_now()
        with self.ledger._lock:
            self.ledger._connection.execute("BEGIN IMMEDIATE")
            try:
                self.ledger._connection.execute(
                    """
                    INSERT OR IGNORE INTO adaptive_evidence_exposures (
                        evidence_digest, first_run_id, fitness_profile_digest,
                        evidence_class, created_at
                    ) VALUES (?, ?, ?, 'exploratory_adaptive_data', ?)
                    """,
                    (evidence_digest, run_id, fitness_profile_digest, created_at),
                )
                row = self.ledger._connection.execute(
                    """
                    SELECT evidence_digest, first_run_id, fitness_profile_digest,
                           evidence_class, created_at
                    FROM adaptive_evidence_exposures WHERE evidence_digest = ?
                    """,
                    (evidence_digest,),
                ).fetchone()
                self.ledger._connection.commit()
            except Exception:
                self.ledger._connection.rollback()
                raise
        if row is None:
            raise RuntimeError("adaptive exposure insert did not produce a row")
        return {
            "evidence_digest": str(row["evidence_digest"]),
            "first_run_id": str(row["first_run_id"]),
            "fitness_profile_digest": str(row["fitness_profile_digest"]),
            "evidence_class": str(row["evidence_class"]),
            "created_at": str(row["created_at"]),
            "formal_confirmation": False,
        }

    def formal_exposure(self, holdout_exposure_key: str) -> dict[str, Any] | None:
        key = _sha256(holdout_exposure_key, "holdout_exposure_key")
        with self.ledger._lock:
            row = self.ledger._connection.execute(
                "SELECT * FROM formal_holdout_exposures WHERE holdout_exposure_key = ?",
                (key,),
            ).fetchone()
        return dict(row) if row is not None else None

    def reserve_formal_stage(
        self,
        *,
        raw_holdout_key: str,
        objective_family_digest: str,
        plan_digest: str,
        idempotency_key: str,
        run_id: str,
        stage: str,
        candidate_id: str,
        artifact_digest: str,
        genome_digest: str,
        partition_digest: str,
    ) -> FormalStageToken:
        raw_key = _sha256(raw_holdout_key, "raw_holdout_key")
        objective = _sha256(objective_family_digest, "objective_family_digest")
        plan = _sha256(plan_digest, "plan_digest")
        run_id = _text(run_id, "run_id")
        stage = _text(stage, "stage")
        if stage not in {"validation", "final_test"}:
            raise ValueError("formal stage must be validation or final_test")
        candidate_id = _text(candidate_id, "candidate_id")
        artifact = _sha256(artifact_digest, "artifact_digest")
        genome = _sha256(genome_digest, "genome_digest")
        partition = _sha256(partition_digest, "partition_digest")
        idempotency = _text(idempotency_key, "idempotency_key")
        token_body = {
            "holdout_exposure_key": raw_key,
            "run_id": run_id,
            "stage": stage,
            "candidate_id": candidate_id,
            "artifact_digest": artifact,
            "genome_digest": genome,
            "partition_digest": partition,
            "objective_family_digest": objective,
            "analysis_plan_digest": plan,
            "idempotency_key": idempotency,
        }
        token = FormalStageToken(
            **token_body,
            token_digest=digest(token_body),
        )
        reserved_at = utc_now()
        with self.ledger._lock:
            self.ledger._connection.execute("BEGIN IMMEDIATE")
            try:
                existing = self.ledger._connection.execute(
                    "SELECT * FROM formal_holdout_exposures WHERE holdout_exposure_key = ?",
                    (raw_key,),
                ).fetchone()
                if existing is not None:
                    if (
                        str(existing["idempotency_key"]) == idempotency
                        and str(existing["token_digest"]) == token.token_digest
                    ):
                        self.ledger._connection.commit()
                        return token
                    raise FormalExposureAlreadyUsed(
                        "raw formal holdout has already been reserved"
                    )
                self.ledger._connection.execute(
                    """
                    INSERT INTO formal_holdout_exposures (
                        holdout_exposure_key, run_id, stage, token_digest,
                        candidate_id, artifact_digest, genome_digest,
                        partition_digest, analysis_plan_digest, state,
                        idempotency_key, reserved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?)
                    """,
                    (
                        raw_key, run_id, stage, token.token_digest, candidate_id,
                        artifact, genome, partition, plan, idempotency, reserved_at,
                    ),
                )
                self.ledger._connection.execute(
                    """
                    INSERT INTO formal_analysis_families (
                        holdout_exposure_key, objective_family_digest,
                        analysis_plan_digest, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (raw_key, objective, plan, reserved_at),
                )
                self.ledger._connection.commit()
            except Exception:
                self.ledger._connection.rollback()
                raise
        return token

    def _validate_token_row(
        self, token: FormalStageToken, row: Any
    ) -> None:
        expected = {
            "holdout_exposure_key": token.holdout_exposure_key,
            "run_id": token.run_id,
            "stage": token.stage,
            "token_digest": token.token_digest,
            "candidate_id": token.candidate_id,
            "artifact_digest": token.artifact_digest,
            "genome_digest": token.genome_digest,
            "partition_digest": token.partition_digest,
            "analysis_plan_digest": token.analysis_plan_digest,
            "idempotency_key": token.idempotency_key,
        }
        if row is None or any(str(row[name]) != value for name, value in expected.items()):
            raise ValueError("formal stage token binding mismatch")

    def open_formal_stage(
        self,
        token: FormalStageToken,
        *,
        run_id: str | None = None,
        candidate_id: str | None = None,
        artifact_digest: str | None = None,
        genome_digest: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(token, FormalStageToken):
            raise TypeError("token must be a FormalStageToken")
        supplied = {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "artifact_digest": artifact_digest,
            "genome_digest": genome_digest,
        }
        for name, value in supplied.items():
            if value is not None and value != getattr(token, name):
                raise ValueError(f"formal stage token binding mismatch: {name}")
        opened_at = utc_now()
        with self.ledger._lock:
            self.ledger._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self.ledger._connection.execute(
                    "SELECT * FROM formal_holdout_exposures WHERE holdout_exposure_key = ?",
                    (token.holdout_exposure_key,),
                ).fetchone()
                self._validate_token_row(token, row)
                if str(row["state"]) != "reserved":
                    raise ValueError("formal stage token is single-use and already opened")
                self.ledger._connection.execute(
                    """
                    UPDATE formal_holdout_exposures
                    SET state = 'opened', opened_at = ?
                    WHERE holdout_exposure_key = ? AND state = 'reserved'
                    """,
                    (opened_at, token.holdout_exposure_key),
                )
                self.ledger._connection.commit()
            except Exception:
                self.ledger._connection.rollback()
                raise
        return self.formal_exposure(token.holdout_exposure_key) or {}

    _T = TypeVar("_T")

    def with_formal_stage(
        self,
        token: FormalStageToken,
        reader: Callable[[], _T],
    ) -> _T:
        """Open atomically before the first callback can read any metric."""

        self.open_formal_stage(token)
        try:
            return reader()
        except Exception:
            self.seal_formal_stage(token, outcome="failed")
            raise

    def seal_formal_stage(
        self, token: FormalStageToken, *, outcome: str
    ) -> dict[str, Any]:
        outcome = _text(outcome, "outcome")
        sealed_at = utc_now()
        with self.ledger._lock:
            self.ledger._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self.ledger._connection.execute(
                    "SELECT * FROM formal_holdout_exposures WHERE holdout_exposure_key = ?",
                    (token.holdout_exposure_key,),
                ).fetchone()
                self._validate_token_row(token, row)
                if str(row["state"]) not in {"reserved", "opened"}:
                    raise ValueError("formal stage exposure is already sealed")
                self.ledger._connection.execute(
                    """
                    UPDATE formal_holdout_exposures
                    SET state = 'sealed', sealed_at = ?
                    WHERE holdout_exposure_key = ?
                    """,
                    (sealed_at, token.holdout_exposure_key),
                )
                self.ledger._connection.commit()
            except Exception:
                self.ledger._connection.rollback()
                raise
        result = self.formal_exposure(token.holdout_exposure_key) or {}
        result["outcome"] = outcome
        return result


__all__ = [
    "FormalExposureAlreadyUsed",
    "FormalStageToken",
    "ScientificExposureRegistry",
    "raw_holdout_exposure_key",
]
