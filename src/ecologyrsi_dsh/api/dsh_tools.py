"""Authenticated DSH role-tool boundary with fail-closed admission."""

from __future__ import annotations

import math
import re
import sqlite3
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Any, Iterator
from uuid import uuid4

from ..core.ledger import ConcurrentRunMutationError
from ..core.models import canonical_json, digest
from ..knowledge.retrieval import (
    assess_dynamic_search_quality,
    merge_dynamic_search_results,
    normalize_dynamic_search_result,
    search_openalex_metadata,
)


ROLE_TOOLS: dict[str, frozenset[str]] = {
    "coordinator": frozenset({"web_search"}),
    "researcher": frozenset({"web_search"}),
    "candidate-proposer": frozenset({"web_search"}),
    "sample-planner": frozenset(
        {"web_search", "ecology_execute_prediction_tool"}
    ),
    "sample-critic": frozenset({"web_search"}),
    "generation-judge": frozenset({"web_search"}),
}

_PREDICTION_TOOL_NAME = "ecology_execute_prediction_tool"
_RETRIEVAL_TOOL_NAME = "web_search"
_RETRIEVAL_MAX_CALLS_PER_STAGE = 3
_RETRIEVAL_MAX_QUERIES = 4
_RETRIEVAL_QUERY_MAX_CHARS = 180
_RETRIEVAL_IDEMPOTENCY_MAX_CHARS = 120
_RETRIEVAL_TECHNICAL_FAILURES = frozenset(
    {
        "primary_provider_unavailable",
        "primary_provider_error",
        "primary_timeout",
        "primary_malformed",
    }
)
_STRUCTURED_STAGE_CONTRACTS: dict[str, tuple[str, str]] = {
    "generation.research": ("researcher", "ecology-research-result@1"),
    "generation.search-plan": (
        "researcher",
        "ecology-research-search-plan@1",
    ),
    "generation.research-synthesis": (
        "researcher",
        "ecology-research-synthesis@1",
    ),
    "generation.reflect": (
        "generation-judge",
        "ecology-generation-reflection@1",
    ),
    "candidate.propose": ("candidate-proposer", "ecology-genome-mutation@1"),
    "candidate.local_edit": (
        "candidate-proposer",
        "ecology-local-edit@1",
    ),
    "generation.judge": ("generation-judge", "ecology-generation-review@1"),
    "sample.plan": ("sample-planner", "ecology-sample-decisions@1"),
    "sample.critic": ("sample-critic", "ecology-sample-review@1"),
    "sample.reflect": ("sample-critic", "ecology-sample-reflection@1"),
}
_GENOME_IDENTITY_DIGEST_FIELDS = (
    "genome_digest",
    "compiled_behavior_digest",
    "phenotype_instance_digest",
)

MAX_STRUCTURED_STAGE_TIMEOUT_MS = 1_800_000

_STRUCTURED_STAGE_SKILLS: dict[str, frozenset[str]] = {
    "generation.research": frozenset({"autonomous-ecology-research"}),
    "generation.search-plan": frozenset({"autonomous-ecology-research"}),
    "generation.research-synthesis": frozenset({"autonomous-ecology-research"}),
    "generation.reflect": frozenset({"batch-scientific-reflection"}),
    "candidate.propose": frozenset({"bounded-plugin-experiment"}),
    "candidate.local_edit": frozenset({"bounded-plugin-experiment"}),
    "generation.judge": frozenset({"candidate-scientific-review"}),
    "sample.plan": frozenset(
        {
            "origin-vector-forecasting-balanced",
            "origin-vector-forecasting-anomaly-aware",
            "origin-vector-forecasting-horizon-aware",
        }
    ),
    "sample.critic": frozenset({"origin-vector-review"}),
    "sample.reflect": frozenset({"origin-vector-review"}),
}

_IDENTITY_FIELDS = frozenset(
    {
        "run_id",
        "role",
        "stage",
        "run_state_revision",
        "stage_attempt",
        "ledger_expected_revision",
        "session_id",
        "idempotency_key",
        "child_reservation_id",
        "activation_lease_id",
        "genome_digest",
        "compiled_behavior_digest",
        "phenotype_instance_digest",
    }
)
_MODEL_BLOCKED_FIELDS = _IDENTITY_FIELDS


def _strict_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int/float coercions."""

    return canonical_json(left) == canonical_json(right)


def _skill_invocation_evidence(
    value: Any,
    *,
    stage: str,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "stage",
        "skill_name",
        "call_count",
        "successful_call_count",
        "call_seq",
        "result_seq",
        "first_tool_call_verified",
        "next_tool_name",
        "next_tool_call_seq",
        "order_verified",
        "source",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("DSH Skill invocation evidence has an invalid shape")
    allowed_skills = _STRUCTURED_STAGE_SKILLS.get(stage)
    expected_next_tool = (
        _PREDICTION_TOOL_NAME if stage == "sample.plan" else "structured_output"
    )
    if (
        value.get("schema_version")
        != "ecologyrsi-dsh.skill-invocation-evidence/1"
        or value.get("stage") != stage
        or value.get("skill_name") not in (allowed_skills or frozenset())
        or value.get("call_count") != 1
        or value.get("successful_call_count") != 1
        or value.get("first_tool_call_verified") is not True
        or value.get("next_tool_name") != expected_next_tool
        or value.get("order_verified") is not True
        or value.get("source") != "dsh_session_event_log"
    ):
        raise ValueError("DSH Skill invocation evidence violates the stage contract")
    for name in ("call_seq", "result_seq", "next_tool_call_seq"):
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"DSH Skill invocation evidence {name} is invalid")
    if not value["call_seq"] < value["result_seq"] < value["next_tool_call_seq"]:
        raise ValueError("DSH Skill invocation evidence order is invalid")
    return deepcopy(dict(value))


class DshToolAuthorizationError(PermissionError):
    error_code = "dsh_tool_authorization_failed"


class DshToolAdmissionClosedError(RuntimeError):
    error_code = "dsh_tool_admission_closed"


class DshPredictionBindingClosedError(DshToolAdmissionClosedError):
    error_code = "dsh_prediction_binding_closed"


class DshToolOperationalTimeoutError(RuntimeError):
    error_code = "structured_role_operational_timeout"


class DshStructuredResultPersistenceError(RuntimeError):
    """A retryable failure to durably accept an already-produced result.

    The caller still owns the same immutable structured output and tool
    receipt. Exposing this as a narrow public control error lets the native
    runtime retry that write without converting an infrastructure incident
    into a failed scientific prediction.
    """

    error_code = "structured_result_persistence_unavailable"


@dataclass(slots=True)
class AdmissionFence:
    run_id: str
    run_state_revision: int
    stage_attempt: int
    state: str
    admission_id: str
    role: str | None = None
    stage: str | None = None
    idempotency_key: str | None = None
    timeout_ms: int | None = None
    deadline_monotonic_ms: float | None = None
    lock: Any = field(default_factory=RLock, repr=False, compare=False)


@dataclass(slots=True)
class DshPredictionToolBinding:
    """One frozen Planner wave and its single executable vector tool."""

    run_id: str
    stage_attempt: int
    idempotency_key: str
    wave_digest: str
    tool_id: str
    sample_ids: tuple[str, ...]
    executor: Callable[[], Mapping[str, Any]] = field(repr=False)
    _result: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _receipt: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _session_calls: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def _materialize_result(self) -> dict[str, Any]:
        raw = self.executor()
        if not isinstance(raw, Mapping) or set(raw) != set(self.sample_ids):
            raise ValueError(
                "registered vector tool must return every frozen sample exactly once"
            )
        outputs: list[dict[str, Any]] = []
        for sample_id in self.sample_ids:
            item = raw[sample_id]
            if not isinstance(item, Mapping) or set(item) != {
                "predicted",
                "metadata",
            }:
                raise ValueError(
                    "registered vector tool output must contain predicted and metadata"
                )
            predicted = item.get("predicted")
            if (
                isinstance(predicted, bool)
                or not isinstance(predicted, (int, float))
                or not math.isfinite(float(predicted))
            ):
                raise ValueError("registered vector tool prediction must be finite")
            metadata = item.get("metadata")
            if not isinstance(metadata, Mapping):
                raise TypeError("registered vector tool metadata must be an object")
            normalized = {
                "sample_id": sample_id,
                "predicted": float(predicted),
                "metadata": deepcopy(dict(metadata)),
            }
            digest(normalized)
            outputs.append(normalized)
        result_body = {
            "schema_version": "ecologyrsi-dsh.prediction-tool-result/1",
            "tool_id": self.tool_id,
            "wave_digest": self.wave_digest,
            "prediction_unit": "forecast_origin_with_target_horizon_vector",
            "prediction_count": len(outputs),
            "outputs": outputs,
        }
        return {**result_body, "output_digest": digest(result_body)}

    def request_digest(self) -> str:
        return digest(
            {
                "tool_name": _PREDICTION_TOOL_NAME,
                "run_id": self.run_id,
                "stage": "sample.plan",
                "stage_attempt": self.stage_attempt,
                "idempotency_key": self.idempotency_key,
                "arguments": {
                    "tool_id": self.tool_id,
                    "wave_digest": self.wave_digest,
                },
            }
        )

    def event_payload(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "ecologyrsi-dsh.dsh-prediction-tool-executed/1",
            "stage": "sample.plan",
            "stage_attempt": self.stage_attempt,
            "idempotency_key": self.idempotency_key,
            "tool_id": self.tool_id,
            "wave_digest": self.wave_digest,
            "sample_ids": list(self.sample_ids),
            "prediction_count": len(self.sample_ids),
            "request_digest": self.request_digest(),
            "output_digest": result["output_digest"],
            "execution_owner": "dsh_agent_tool_call",
        }

    def restore_recorded_result(
        self,
        *,
        event_id: str,
        event_seq: int,
        event_payload: Mapping[str, Any],
    ) -> None:
        """Rebuild a deterministic tool result and bind its durable receipt."""

        with self._lock:
            if self._result is not None or self._receipt is not None:
                raise RuntimeError("prediction tool result is already materialized")
            result = self._materialize_result()
            expected_payload = self.event_payload(result)
            if dict(event_payload) != expected_payload:
                raise ValueError("recorded prediction-tool result no longer reproduces")
            self._result = result
            self._receipt = {
                "event_id": event_id,
                "event_seq": event_seq,
                "request_digest": expected_payload["request_digest"],
                "output_digest": result["output_digest"],
                "execution_owner": "dsh_agent_tool_call",
            }

    def execute(self, arguments: Mapping[str, Any], *, session_id: str) -> dict[str, Any]:
        if set(arguments) != {"tool_id", "wave_digest"}:
            raise ValueError(
                "prediction tool arguments must contain tool_id and wave_digest only"
            )
        if arguments.get("tool_id") != self.tool_id:
            raise DshToolAuthorizationError("prediction tool id is outside the frozen wave")
        if arguments.get("wave_digest") != self.wave_digest:
            raise DshToolAuthorizationError("prediction wave digest does not match")
        with self._lock:
            if self._session_calls.get(session_id, 0) != 0:
                raise DshToolAuthorizationError(
                    "the Planner must call the vector prediction tool exactly once"
                )
            self._session_calls[session_id] = 1
            if self._result is None:
                self._result = self._materialize_result()
            return deepcopy(self._result)

    def set_receipt(self, receipt: Mapping[str, Any]) -> None:
        with self._lock:
            self._receipt = deepcopy(dict(receipt))

    def require_session_call(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            if self._session_calls.get(session_id) != 1 or self._result is None:
                raise DshToolAuthorizationError(
                    "sample.plan requires one vector prediction tool call by its DSH child"
                )
            if self._receipt is None:
                raise RuntimeError("prediction tool execution was not durably recorded")
            return deepcopy(self._receipt)

    def prediction_bundle(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            if self._result is None or self._receipt is None:
                raise RuntimeError("DSH prediction tool did not complete")
            audit = {
                "execution_owner": "dsh_agent_tool_call",
                "dsh_tool_event_id": self._receipt["event_id"],
                "dsh_tool_output_digest": self._result["output_digest"],
            }
            return {
                str(item["sample_id"]): {
                    "predicted": float(item["predicted"]),
                    "metadata": {**deepcopy(item["metadata"]), **audit},
                }
                for item in self._result["outputs"]
            }

    def audit_receipt(self) -> dict[str, Any]:
        with self._lock:
            if self._receipt is None:
                raise RuntimeError("DSH prediction tool receipt is unavailable")
            return deepcopy(self._receipt)


def _assert_finite_json_shape(value: Any, *, label_free: bool, path: str = "$") -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json_shape(item, label_free=label_free, path=f"{path}[{index}]")
        return
    if not isinstance(value, Mapping):
        return
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("DSH tool argument keys must be strings")
        normalized = key.casefold().replace("-", "_")
        if normalized in _MODEL_BLOCKED_FIELDS:
            raise DshToolAuthorizationError(
                f"model arguments cannot override Host identity at {path}.{key}"
            )
        if label_free and any(
            token in normalized.split("_")
            for token in ("observed", "observation", "label", "ground", "truth")
        ):
            raise DshToolAuthorizationError(
                f"planner arguments must be label-free at {path}.{key}"
            )
        _assert_finite_json_shape(item, label_free=label_free, path=f"{path}.{key}")


def _dsh_session_metrics(value: Any, *, session_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "session_id",
        "context_pressure",
        "provider_usage",
    }:
        raise ValueError("DSH session metrics have an invalid shape")
    if value.get("schema_version") != "ecologyrsi-dsh.dsh-session-metrics/1":
        raise ValueError("unsupported DSH session metrics schema")
    if value.get("session_id") != session_id:
        raise DshToolAuthorizationError("DSH session metrics identity mismatch")

    pressure = value.get("context_pressure")
    if not isinstance(pressure, Mapping):
        raise TypeError("DSH context pressure must be an object")
    if pressure.get("available") is True:
        expected = {
            "available",
            "source",
            "measurement",
            "log_revision",
            "baseline_kind",
            "total_tokens",
            "surface_tokens",
        }
        if set(pressure) != expected:
            raise ValueError("DSH context pressure has an invalid shape")
        if pressure.get("source") != "dsh_token_meter" or pressure.get(
            "measurement"
        ) != "current_context_pressure":
            raise ValueError("DSH context pressure source is invalid")
        if pressure.get("baseline_kind") not in {"none", "estimated", "usage"}:
            raise ValueError("DSH context pressure baseline is invalid")
        for name in ("log_revision", "total_tokens", "surface_tokens"):
            item = pressure.get(name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"DSH context pressure {name} is invalid")
    elif dict(pressure) != {"available": False, "source": "dsh_token_meter"}:
        raise ValueError("unavailable DSH context pressure is invalid")

    usage = value.get("provider_usage")
    if not isinstance(usage, Mapping):
        raise TypeError("DSH provider usage must be an object")
    if usage.get("available") is True:
        if set(usage) != {"available", "source", "measurement", "totals"}:
            raise ValueError("DSH provider usage has an invalid shape")
        if usage.get("source") != "dsh_session_projection_token_usage" or usage.get(
            "measurement"
        ) != "cumulative_provider_reported_usage":
            raise ValueError("DSH provider usage source is invalid")
        totals = usage.get("totals")
        fields = {
            "uncached_input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "total_tokens",
        }
        if not isinstance(totals, Mapping) or set(totals) != fields:
            raise ValueError("DSH provider usage totals have an invalid shape")
        for name in fields:
            item = totals.get(name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"DSH provider usage {name} is invalid")
        if totals["total_tokens"] != sum(
            int(totals[name]) for name in fields - {"total_tokens"}
        ):
            raise ValueError("DSH provider total token count is inconsistent")
    elif dict(usage) != {
        "available": False,
        "source": "dsh_session_projection_token_usage",
    }:
        raise ValueError("unavailable DSH provider usage is invalid")
    return deepcopy(dict(value))


class DshToolService:
    def __init__(
        self,
        ledger: Any,
        *,
        retrieval_fallback: Callable[..., Mapping[str, Any]] | None = None,
        monotonic_ms: Callable[[], float] | None = None,
    ) -> None:
        self.ledger = ledger
        self._fences: dict[tuple[str, int, int], AdmissionFence] = {}
        self._run_admission: dict[str, str] = {}
        self._admission_registry_lock = RLock()
        self._prediction_bindings: dict[
            tuple[str, int, str], DshPredictionToolBinding
        ] = {}
        self._prediction_lock = Lock()
        # Reservation mutations are serialized per deterministic key.  A
        # bounded stripe set avoids a process-wide bottleneck and does not
        # retain one lock per unbounded stream of requests.
        self._launch_locks = tuple(Lock() for _ in range(32))
        self._retrieval_lock = Lock()
        self._retrieval_fallback = retrieval_fallback or search_openalex_metadata
        self._monotonic_ms = monotonic_ms or (
            lambda: time.monotonic_ns() / 1_000_000
        )

    @staticmethod
    def _prediction_event_id(run_id: str, idempotency_key: str) -> str:
        return (
            f"{run_id}:dsh-prediction-tool:"
            f"{digest({'idempotency_key': idempotency_key, 'tool_name': _PREDICTION_TOOL_NAME})}"
        )

    @staticmethod
    def _structured_event_id(run_id: str, stage: str, idempotency_key: str) -> str:
        return (
            f"{run_id}:dsh-structured:"
            f"{digest({'stage': stage, 'idempotency_key': idempotency_key})}"
        )

    @staticmethod
    def _retrieval_event_id(
        run_id: str,
        role: str,
        stage: str,
        stage_attempt: int,
        stage_idempotency_key: str,
        idempotency_key: str,
    ) -> str:
        return (
            f"{run_id}:dsh-retrieval:"
            f"{digest({'role': role, 'stage': stage, 'stage_attempt': stage_attempt, 'stage_idempotency_key': stage_idempotency_key, 'retrieval_key': idempotency_key})}"
        )

    def _event_by_id(self, run_id: str, event_id: str) -> Any | None:
        return self.ledger.event_by_id(event_id, run_id=run_id)

    def _launch_lock_for(self, key: str) -> Lock:
        return self._launch_locks[int(digest(key)[:8], 16) % len(self._launch_locks)]

    @contextmanager
    def bind_prediction_tool(
        self,
        *,
        run_id: str,
        stage_attempt: int,
        idempotency_key: str,
        wave_digest: str,
        tool_id: str,
        sample_ids: tuple[str, ...],
        executor: Callable[[], Mapping[str, Any]],
    ) -> Iterator[DshPredictionToolBinding]:
        """Bind one ephemeral executable to one Host-authenticated Planner wave."""

        if not run_id or not idempotency_key or not tool_id:
            raise ValueError("prediction tool binding identity must be non-empty")
        if stage_attempt < 1:
            raise ValueError("prediction tool stage_attempt must be positive")
        if len(wave_digest) != 64 or any(
            character not in "0123456789abcdef" for character in wave_digest
        ):
            raise ValueError("prediction tool wave_digest must be SHA-256")
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("prediction tool sample_ids must be non-empty and unique")
        if not callable(executor):
            raise TypeError("prediction tool executor must be callable")
        key = (run_id, stage_attempt, idempotency_key)
        binding = DshPredictionToolBinding(
            run_id=run_id,
            stage_attempt=stage_attempt,
            idempotency_key=idempotency_key,
            wave_digest=wave_digest,
            tool_id=tool_id,
            sample_ids=sample_ids,
            executor=executor,
        )
        with self._prediction_lock:
            if key in self._prediction_bindings:
                raise RuntimeError("prediction tool binding is already active")
            self._prediction_bindings[key] = binding
        try:
            prior = self._event_by_id(
                run_id,
                self._prediction_event_id(run_id, idempotency_key),
            )
            if prior is not None:
                if prior.kind != "DshPredictionToolExecuted":
                    raise ValueError("prediction-tool event identity was reused")
                binding.restore_recorded_result(
                    event_id=prior.event_id,
                    event_seq=prior.seq,
                    event_payload=prior.payload,
                )
            yield binding
        finally:
            with self._prediction_lock:
                if self._prediction_bindings.get(key) is binding:
                    self._prediction_bindings.pop(key, None)

    def open_admission(
        self,
        run_id: str,
        run_state_revision: int,
        stage_attempt: int,
        *,
        role: str | None = None,
        stage: str | None = None,
        idempotency_key: str | None = None,
    ) -> AdmissionFence:
        binding = (role, stage, idempotency_key)
        if any(value is not None for value in binding):
            if not all(isinstance(value, str) and value.strip() for value in binding):
                raise ValueError("structured admission binding must be non-empty text")
        key = (run_id, run_state_revision, stage_attempt)
        with self._admission_registry_lock:
            if self._run_admission.get(run_id, "open") != "open":
                raise DshToolAdmissionClosedError("run admission is closed")
            fence = self._fences.get(key)
            if fence is None:
                fence = AdmissionFence(
                    run_id,
                    run_state_revision,
                    stage_attempt,
                    "open",
                    f"admission-{uuid4()}",
                    role=role,
                    stage=stage,
                    idempotency_key=idempotency_key,
                )
                self._fences[key] = fence
        with fence.lock:
            if fence.state != "open":
                raise DshToolAdmissionClosedError(
                    "admission fence is permanently closed"
                )
            existing_binding = (fence.role, fence.stage, fence.idempotency_key)
            if all(value is None for value in existing_binding) and all(
                value is not None for value in binding
            ):
                fence.role = role
                fence.stage = stage
                fence.idempotency_key = idempotency_key
            elif any(value is not None for value in binding) and existing_binding != binding:
                raise DshToolOperationalTimeoutError(
                    "structured admission binding mismatch"
                )
        return fence

    def close_admission(
        self, run_id: str, run_state_revision: int, stage_attempt: int
    ) -> AdmissionFence:
        key = (run_id, run_state_revision, stage_attempt)
        with self._admission_registry_lock:
            fence = self._fences.get(key)
            if fence is None:
                fence = AdmissionFence(
                    run_id,
                    run_state_revision,
                    stage_attempt,
                    "closed",
                    f"admission-{uuid4()}",
                )
                self._fences[key] = fence
        with fence.lock:
            fence.state = "closed"
        return fence

    def close_run_admissions(self, run_id: str) -> tuple[AdmissionFence, ...]:
        with self._admission_registry_lock:
            self._run_admission[run_id] = "closed"
            fences = tuple(
                fence
                for key, fence in self._fences.items()
                if key[0] == run_id
            )
        closed: list[AdmissionFence] = []
        for fence in fences:
            with fence.lock:
                if fence.state != "open":
                    continue
                fence.state = "closed"
                closed.append(fence)
        return tuple(closed)

    def open_run_admissions(self, run_id: str) -> None:
        with self._admission_registry_lock:
            self._run_admission[run_id] = "open"

    def allocate_child_reservation(self, request: Mapping[str, Any]) -> dict[str, Any]:
        expected_fields = {
            "request_id",
            "run_id",
            "parent_session_id",
            "role",
            "stage",
            "run_state_revision",
            "stage_attempt",
            "admission_id",
            "timeout_ms",
            "item_digest",
            "idempotency_key",
        }
        optional_fields = {"sample_member_digests"}
        if not isinstance(request, Mapping) or set(request) not in (
            expected_fields,
            expected_fields | optional_fields,
        ):
            raise ValueError("child reservation request has an invalid shape")
        for name in (
            "request_id",
            "run_id",
            "parent_session_id",
            "role",
            "stage",
            "admission_id",
            "idempotency_key",
        ):
            if not isinstance(request[name], str) or not request[name].strip():
                raise ValueError(f"child reservation {name} must be non-empty text")
        for name in ("run_state_revision", "stage_attempt"):
            value = request[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"child reservation {name} must be a non-negative integer"
                )
        timeout_ms = request["timeout_ms"]
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or timeout_ms < 1
        ):
            raise ValueError("child reservation timeout_ms must be a positive integer")
        if timeout_ms > MAX_STRUCTURED_STAGE_TIMEOUT_MS:
            raise DshToolOperationalTimeoutError(
                "structured role timeout exceeds the protocol ceiling"
            )
        item_digest = request["item_digest"]
        if (
            not isinstance(item_digest, str)
            or len(item_digest) != 64
            or any(character not in "0123456789abcdef" for character in item_digest)
        ):
            raise ValueError("child reservation item_digest must be a SHA-256 digest")
        sample_member_digests = request.get("sample_member_digests")
        if sample_member_digests is not None:
            if not str(request["stage"]).startswith("sample."):
                raise ValueError(
                    "sample member correlation is valid only for sample stages"
                )
            if (
                not isinstance(sample_member_digests, list)
                or not 1 <= len(sample_member_digests) <= 128
                or any(
                    not isinstance(member, str)
                    or len(member) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in member
                    )
                    for member in sample_member_digests
                )
                or sample_member_digests != sorted(set(sample_member_digests))
            ):
                raise ValueError(
                    "child reservation sample_member_digests must be sorted "
                    "unique SHA-256 digests"
                )
        run_id = str(request["run_id"])
        fence_key = (
            run_id,
            int(request["run_state_revision"]),
            int(request["stage_attempt"]),
        )
        with self._admission_registry_lock:
            run_admission = self._run_admission.get(run_id, "open")
            fence = self._fences.get(fence_key)
        if run_admission != "open":
            raise DshToolAdmissionClosedError("run admission is closed")
        if fence is None:
            raise DshToolOperationalTimeoutError(
                "structured admission is unavailable"
            )
        with fence.lock:
            if fence.state != "open":
                raise DshToolOperationalTimeoutError(
                    "structured admission is closed"
                )
            if (
                fence.admission_id != request["admission_id"]
                or fence.role != request["role"]
                or fence.stage != request["stage"]
                or fence.idempotency_key != request["idempotency_key"]
            ):
                raise DshToolOperationalTimeoutError(
                    "structured admission binding mismatch"
                )
            if fence.timeout_ms is None:
                fence.timeout_ms = timeout_ms
                fence.deadline_monotonic_ms = self._monotonic_ms() + timeout_ms
            elif fence.timeout_ms != timeout_ms:
                raise DshToolOperationalTimeoutError(
                    "structured admission timeout mismatch"
                )
            frozen_deadline = fence.deadline_monotonic_ms
        if frozen_deadline is None:  # pragma: no cover - guarded assignment above
            raise RuntimeError("structured admission deadline was not armed")
        if self.ledger.latest_run_seq(run_id) == 0:
            raise DshToolAuthorizationError("unknown child reservation run")
        event_id = f"{run_id}:dsh-child-reservation:{digest({'request_id': request['request_id']})}"
        business_key_digest = digest(
            {
                "parent_session_id": request["parent_session_id"],
                "role": request["role"],
                "stage": request["stage"],
                "item_digest": item_digest,
                "idempotency_key": request["idempotency_key"],
            }
        )
        request_contract_digest = digest(dict(request))
        with self._launch_lock_for(business_key_digest):
            while True:
                prior = self._event_by_id(run_id, event_id)
                if prior is not None:
                    if (
                        prior.kind != "DshChildLaunchReserved"
                        or prior.payload.get("request_id") != request["request_id"]
                        or prior.payload.get("parent_session_id")
                        != request["parent_session_id"]
                        or prior.payload.get("business_key_digest")
                        != business_key_digest
                        or prior.payload.get("request_contract_digest")
                        != request_contract_digest
                    ):
                        raise ValueError(
                            "child reservation request_id was reused with different input"
                        )
                    launch = prior.payload["launch"]
                    break
                if self._monotonic_ms() >= frozen_deadline:
                    raise DshToolOperationalTimeoutError(
                        "structured role deadline expired"
                    )
                matching = [
                    event.payload["launch"]
                    for event in self.ledger.events_by_kind(
                        run_id, "DshChildLaunchReserved"
                    )
                    if event.payload.get("business_key_digest") == business_key_digest
                ]
                launch_attempt = max(
                    (int(item["launch_attempt"]) for item in matching),
                    default=0,
                ) + 1
                reservation_digest = digest(
                    {
                        "run_id": run_id,
                        "role": request["role"],
                        "stage": request["stage"],
                        "business_key_digest": business_key_digest,
                        "launch_attempt": launch_attempt,
                        "request_id": request["request_id"],
                    }
                )
                reservation_id = f"reservation-{reservation_digest}"
                launch = {
                    "reservation_id": reservation_id,
                    "run_id": run_id,
                    "stage": request["stage"],
                    "role": request["role"],
                    "item_digest": item_digest,
                    "idempotency_key": request["idempotency_key"],
                    "launch_attempt": launch_attempt,
                }
                if sample_member_digests is not None:
                    launch["sample_member_digests"] = list(
                        sample_member_digests
                    )
                try:
                    @contextmanager
                    def commit_guard(inserted: bool) -> Iterator[None]:
                        if not inserted:
                            yield
                            return
                        with fence.lock:
                            if (
                                fence.state != "open"
                                or fence.admission_id != request["admission_id"]
                                or fence.timeout_ms != timeout_ms
                                or fence.deadline_monotonic_ms is None
                                or self._monotonic_ms()
                                >= fence.deadline_monotonic_ms
                            ):
                                raise DshToolOperationalTimeoutError(
                                    "structured role deadline expired"
                                )
                            yield

                    self.ledger.append(
                        run_id,
                        "DshChildLaunchReserved",
                        {
                            "schema_version": "ecologyrsi-dsh.child-launch-reserved/1",
                            "request_id": request["request_id"],
                            "parent_session_id": request["parent_session_id"],
                            "business_key_digest": business_key_digest,
                            "request_contract_digest": request_contract_digest,
                            "launch": launch,
                        },
                        event_id=event_id,
                        expected_run_seq=self.ledger.latest_run_seq(run_id),
                        commit_guard=commit_guard,
                    )
                except ConcurrentRunMutationError:
                    continue
                break
        return {
            "accepted": True,
            "admission_id": fence.admission_id,
            "timeout_ms": timeout_ms,
            "launch": dict(launch),
            "ledger_expected_revision": self.ledger.latest_seq(),
        }

    def record_child_failure(self, request: Mapping[str, Any]) -> dict[str, Any]:
        expected_fields = {"run_id", "stage", "idempotency_key", "error_code"}
        if not isinstance(request, Mapping) or set(request) != expected_fields:
            raise ValueError("child failure request has an invalid shape")
        for name in ("run_id", "stage", "idempotency_key", "error_code"):
            if not isinstance(request[name], str) or not request[name].strip():
                raise ValueError(f"child failure {name} must be non-empty text")
        error_code = str(request["error_code"]).strip()
        if len(error_code) > 80 or not error_code.replace("_", "").isalnum():
            raise ValueError("child failure error_code must be normalized text")
        run_id = str(request["run_id"])
        lock_key = digest(
            {
                "run_id": run_id,
                "stage": request["stage"],
                "idempotency_key": request["idempotency_key"],
            }
        )
        with self._launch_lock_for(lock_key):
            while True:
                launch_event = next(
                    (
                        event
                        for event in reversed(
                            self.ledger.events_by_kind(
                                run_id, "DshChildLaunchReserved"
                            )
                        )
                        if isinstance(event.payload.get("launch"), Mapping)
                        and event.payload["launch"].get("stage") == request["stage"]
                        and event.payload["launch"].get("idempotency_key")
                        == request["idempotency_key"]
                    ),
                    None,
                )
                if launch_event is None:
                    return {"accepted": False, "reason": "launch_not_found"}
                launch = launch_event.payload["launch"]
                reservation_id = str(launch.get("reservation_id") or "")
                settled_events = self.ledger.events_by_kind(
                    run_id, "DshStructuredResultAccepted"
                ) + self.ledger.events_by_kind(run_id, "DshChildExecutionFailed")
                if any(
                    isinstance(event.payload.get("identity"), Mapping)
                    and event.payload["identity"].get("child_reservation_id")
                    == reservation_id
                    for event in settled_events
                ):
                    return {"accepted": True, "already_settled": True}
                event_id = (
                    f"{run_id}:dsh-child-failed:"
                    f"{digest({'reservation_id': reservation_id})}"
                )
                try:
                    self.ledger.append(
                        run_id,
                        "DshChildExecutionFailed",
                        {
                            "schema_version": (
                                "ecologyrsi-dsh.child-execution-failed/1"
                            ),
                            "identity": {
                                "child_reservation_id": reservation_id,
                                "stage": request["stage"],
                                "idempotency_key": request["idempotency_key"],
                            },
                            "error_code": error_code,
                        },
                        event_id=event_id,
                        expected_run_seq=self.ledger.latest_run_seq(run_id),
                    )
                except ConcurrentRunMutationError:
                    # Sample completions and sibling failures can append at
                    # high frequency. Re-read so failure settlement is never
                    # lost merely because another child won this CAS slot.
                    continue
                break
        return {"accepted": True, "already_settled": False}

    def execute(self, tool_name: str, envelope: Mapping[str, Any]) -> dict[str, Any]:
        if set(envelope) != {"identity", "arguments"}:
            raise ValueError("DSH tool envelope must contain identity and arguments only")
        identity = envelope["identity"]
        arguments = envelope["arguments"]
        if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS:
            raise DshToolAuthorizationError("invalid Host-bound DSH tool identity")
        if not isinstance(arguments, Mapping):
            raise TypeError("DSH tool arguments must be an object")
        role = self._authorize_identity(
            identity,
            tool_name=tool_name,
            allow_ledger_advance=True,
        )
        _assert_finite_json_shape(arguments, label_free=role == "sample-planner")
        if tool_name != _PREDICTION_TOOL_NAME:
            raise DshToolAuthorizationError("unsupported DSH role tool")
        if identity.get("stage") != "sample.plan":
            raise DshToolAuthorizationError(
                "prediction tool is available only during sample.plan"
            )
        key = (
            str(identity["run_id"]),
            int(identity["stage_attempt"]),
            str(identity["idempotency_key"]),
        )
        with self._prediction_lock:
            binding = self._prediction_bindings.get(key)
        if binding is None:
            raise DshPredictionBindingClosedError(
                "no active Host prediction tool is bound to this Planner wave"
            )
        result = binding.execute(arguments, session_id=str(identity["session_id"]))
        request_digest = binding.request_digest()
        event_id = self._prediction_event_id(
            str(identity["run_id"]), str(identity["idempotency_key"])
        )
        event_payload = binding.event_payload(result)
        with self._prediction_lock:
            prior = self._event_by_id(str(identity["run_id"]), event_id)
            if prior is not None:
                if prior.kind != "DshPredictionToolExecuted" or prior.payload != event_payload:
                    raise ValueError("prediction-tool idempotency key was reused")
                event = prior
            else:
                event = self.ledger.append(
                    str(identity["run_id"]),
                    "DshPredictionToolExecuted",
                    event_payload,
                    event_id=event_id,
                )
        receipt = {
            "event_id": event.event_id,
            "event_seq": event.seq,
            "request_digest": request_digest,
            "output_digest": result["output_digest"],
            "execution_owner": "dsh_agent_tool_call",
        }
        binding.set_receipt(receipt)
        return {
            "accepted": True,
            "tool_name": tool_name,
            "request_digest": request_digest,
            "event_id": event.event_id,
            **result,
        }

    @staticmethod
    def _retrieval_arguments(value: Any) -> tuple[tuple[str, ...], str]:
        if not isinstance(value, Mapping) or set(value) != {
            "queries",
            "retrieval_key",
        }:
            raise ValueError("dynamic retrieval arguments have an invalid shape")
        raw_queries = value.get("queries")
        if not isinstance(raw_queries, list) or not (
            1 <= len(raw_queries) <= _RETRIEVAL_MAX_QUERIES
        ):
            raise ValueError("dynamic retrieval requires one to four queries")
        queries: list[str] = []
        seen: set[str] = set()
        for raw_query in raw_queries:
            if not isinstance(raw_query, str):
                raise ValueError("dynamic retrieval queries must be text")
            query = " ".join(raw_query.split())
            if not query or len(query) > _RETRIEVAL_QUERY_MAX_CHARS:
                raise ValueError("dynamic retrieval query is empty or too long")
            normalized = query.casefold()
            if normalized not in seen:
                seen.add(normalized)
                queries.append(query)
        raw_key = value.get("retrieval_key")
        if (
            not isinstance(raw_key, str)
            or not 1 <= len(raw_key) <= _RETRIEVAL_IDEMPOTENCY_MAX_CHARS
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", raw_key) is None
        ):
            raise ValueError("dynamic retrieval idempotency key is invalid")
        return tuple(queries), raw_key

    @staticmethod
    def _retrieval_response(payload: Mapping[str, Any]) -> dict[str, Any]:
        result = payload.get("result")
        if not isinstance(result, Mapping):
            raise ValueError("recorded dynamic retrieval result is invalid")
        if payload.get("result_digest") != digest(result):
            raise ValueError("recorded dynamic retrieval result digest mismatch")
        return {
            **deepcopy(dict(result)),
            "provider_route": payload.get("provider_route"),
            "fallback_reason": payload.get("fallback_reason"),
            "result_digest": payload.get("result_digest"),
        }

    def _recorded_retrieval(
        self,
        *,
        identity: Mapping[str, Any],
        queries: tuple[str, ...],
        retrieval_idempotency_key: str,
    ) -> Any | None:
        run_id = str(identity["run_id"])
        event_id = self._retrieval_event_id(
            run_id,
            str(identity["role"]),
            str(identity["stage"]),
            int(identity["stage_attempt"]),
            str(identity["idempotency_key"]),
            retrieval_idempotency_key,
        )
        prior = self._event_by_id(run_id, event_id)
        if prior is None:
            return None
        if prior.kind != "DshRetrievalExecuted":
            raise ValueError("dynamic retrieval event identity was reused")
        payload = prior.payload
        if (
            not isinstance(payload, Mapping)
            or payload.get("query_digest") != digest(list(queries))
            or payload.get("queries") != list(queries)
        ):
            raise ValueError("dynamic retrieval idempotency key was reused")
        recorded_identity = payload.get("identity")
        if not isinstance(recorded_identity, Mapping) or any(
            recorded_identity.get(name) != identity.get(name)
            for name in (
                "run_id",
                "role",
                "stage",
                "run_state_revision",
                "stage_attempt",
                "idempotency_key",
                "genome_digest",
                "compiled_behavior_digest",
                "phenotype_instance_digest",
            )
        ):
            raise DshToolAuthorizationError("dynamic retrieval replay identity mismatch")
        return prior

    def replay_retrieval(self, envelope: Mapping[str, Any]) -> dict[str, Any] | None:
        """Replay one completed dynamic retrieval before any network access."""

        if not isinstance(envelope, Mapping) or set(envelope) != {
            "identity",
            "arguments",
        }:
            raise ValueError("dynamic retrieval replay envelope has an invalid shape")
        identity = envelope.get("identity")
        if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS:
            raise DshToolAuthorizationError("invalid Host-bound retrieval identity")
        queries, retrieval_key = self._retrieval_arguments(envelope.get("arguments"))
        self._authorize_identity(
            identity,
            tool_name=_RETRIEVAL_TOOL_NAME,
            allow_ledger_advance=True,
        )
        prior = self._recorded_retrieval(
            identity=identity,
            queries=queries,
            retrieval_idempotency_key=retrieval_key,
        )
        return None if prior is None else self._retrieval_response(prior.payload)

    def complete_retrieval(self, envelope: Mapping[str, Any]) -> dict[str, Any]:
        """Assess DSH primary evidence, fall back when needed, and persist once."""

        if not isinstance(envelope, Mapping) or set(envelope) not in (
            {"identity", "arguments", "primary_result"},
            {"identity", "arguments", "primary_error_code"},
            {
                "identity",
                "arguments",
                "primary_result",
                "primary_error_code",
            },
        ):
            raise ValueError("dynamic retrieval completion envelope has an invalid shape")
        identity = envelope.get("identity")
        arguments = envelope.get("arguments")
        if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS:
            raise DshToolAuthorizationError("invalid Host-bound retrieval identity")
        queries, retrieval_key = self._retrieval_arguments(arguments)
        self._authorize_identity(
            identity,
            tool_name=_RETRIEVAL_TOOL_NAME,
            allow_ledger_advance=True,
        )
        with self._retrieval_lock:
            prior = self._recorded_retrieval(
                identity=identity,
                queries=queries,
                retrieval_idempotency_key=retrieval_key,
            )
            if prior is not None:
                return self._retrieval_response(prior.payload)
            stage_events = [
                event
                for event in self.ledger.events(str(identity["run_id"]))
                if event.kind == "DshRetrievalExecuted"
                and isinstance(event.payload.get("identity"), Mapping)
                and event.payload["identity"].get("stage") == identity["stage"]
                and event.payload["identity"].get("stage_attempt")
                == identity["stage_attempt"]
                and event.payload["identity"].get("idempotency_key")
                == identity["idempotency_key"]
            ]
            if len(stage_events) >= _RETRIEVAL_MAX_CALLS_PER_STAGE:
                raise DshToolAuthorizationError(
                    "dynamic retrieval stage call budget is exhausted"
                )

            primary: dict[str, Any] | None = None
            primary_error_code = envelope.get("primary_error_code")
            if (
                primary_error_code is not None
                and primary_error_code not in _RETRIEVAL_TECHNICAL_FAILURES
            ):
                raise ValueError("dynamic retrieval primary error code is invalid")
            if "primary_result" in envelope:
                try:
                    primary = normalize_dynamic_search_result(
                        envelope.get("primary_result")
                    )
                except (TypeError, ValueError):
                    primary_error_code = "primary_malformed"
            elif primary_error_code not in _RETRIEVAL_TECHNICAL_FAILURES:
                raise ValueError("dynamic retrieval primary error code is invalid")

            if primary is None:
                primary_quality = assess_dynamic_search_quality(
                    queries,
                    {"sources": [], "truncated": False},
                )
                fallback_reason = str(primary_error_code)
                primary_quality["fallback_reason"] = fallback_reason
            else:
                primary_quality = assess_dynamic_search_quality(queries, primary)
                fallback_reason = (
                    str(primary_error_code)
                    if primary_error_code is not None
                    else primary_quality["fallback_reason"]
                )
                primary_quality["fallback_reason"] = fallback_reason
                primary_quality["sufficient"] = fallback_reason is None

            fallback: dict[str, Any] | None = None
            if fallback_reason is not None:
                try:
                    fallback = normalize_dynamic_search_result(
                        self._retrieval_fallback(queries, limit=8)
                    )
                except Exception:  # noqa: BLE001 - isolated optional provider
                    fallback = {"sources": [], "truncated": False}
                provider_route = "dsh_primary_then_openalex_fallback"
            else:
                provider_route = "dsh_primary"
            result = merge_dynamic_search_results(primary, fallback)
            result_digest = digest(result)
            payload = {
                "schema_version": "ecologyrsi-dsh.retrieval-executed/1",
                "execution_owner": "dsh_agent_web_search",
                "identity": deepcopy(dict(identity)),
                "retrieval_idempotency_key": retrieval_key,
                "query_digest": digest(list(queries)),
                "queries": list(queries),
                "provider_route": provider_route,
                "fallback_reason": fallback_reason,
                "primary_quality": deepcopy(dict(primary_quality)),
                "result": deepcopy(result),
                "result_digest": result_digest,
            }
            event = self.ledger.append(
                str(identity["run_id"]),
                "DshRetrievalExecuted",
                payload,
                event_id=self._retrieval_event_id(
                    str(identity["run_id"]),
                    str(identity["role"]),
                    str(identity["stage"]),
                    int(identity["stage_attempt"]),
                    str(identity["idempotency_key"]),
                    retrieval_key,
                ),
            )
            return self._retrieval_response(event.payload)

    @staticmethod
    def _validate_identity_static(
        identity: Mapping[str, Any],
        *,
        tool_name: str | None = None,
        expected_role: str | None = None,
    ) -> str:
        role = str(identity.get("role") or "")
        if tool_name is not None and tool_name not in ROLE_TOOLS.get(role, frozenset()):
            raise DshToolAuthorizationError("tool is not allowed for the bound role")
        if expected_role is not None and role != expected_role:
            raise DshToolAuthorizationError("structured result role does not match its stage")
        for name in (
            "run_id",
            "role",
            "stage",
            "session_id",
            "idempotency_key",
            "child_reservation_id",
            "activation_lease_id",
        ):
            if not isinstance(identity.get(name), str) or not str(identity[name]).strip():
                raise DshToolAuthorizationError(f"invalid Host identity field: {name}")
        for name in (
            "genome_digest",
            "compiled_behavior_digest",
            "phenotype_instance_digest",
        ):
            value = identity.get(name)
            if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise DshToolAuthorizationError(f"invalid Host digest field: {name}")
        for name in (
            "run_state_revision",
            "stage_attempt",
            "ledger_expected_revision",
        ):
            value = identity.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DshToolAuthorizationError(f"invalid Host revision field: {name}")
        return role

    def _validate_recorded_planner_binding(self, event: Any) -> None:
        invalid_message = "recorded sample.plan prediction-tool binding is invalid"
        payload = event.payload
        identity = payload.get("identity")
        structured = payload.get("structured")
        receipt = payload.get("required_tool_receipt")
        receipt_fields = {
            "event_id",
            "event_seq",
            "request_digest",
            "output_digest",
            "execution_owner",
        }
        if (
            not isinstance(identity, Mapping)
            or not isinstance(structured, Mapping)
            or not isinstance(receipt, Mapping)
            or set(receipt) != receipt_fields
            or not isinstance(receipt.get("event_id"), str)
            or not receipt["event_id"]
            or isinstance(receipt.get("event_seq"), bool)
            or not isinstance(receipt.get("event_seq"), int)
            or receipt["event_seq"] < 0
            or receipt.get("execution_owner") != "dsh_agent_tool_call"
            or any(
                not isinstance(receipt.get(name), str)
                or len(receipt[name]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in receipt[name]
                )
                for name in ("request_digest", "output_digest")
            )
        ):
            raise ValueError(invalid_message)

        run_id = str(identity["run_id"])
        idempotency_key = str(identity["idempotency_key"])
        expected_event_id = self._prediction_event_id(run_id, idempotency_key)
        if receipt["event_id"] != expected_event_id:
            raise ValueError(invalid_message)
        tool_event = self._event_by_id(run_id, receipt["event_id"])
        if (
            tool_event is None
            or tool_event.kind != "DshPredictionToolExecuted"
            or tool_event.seq != receipt["event_seq"]
            or tool_event.seq >= event.seq
        ):
            raise ValueError(invalid_message)

        tool_payload = tool_event.payload
        tool_fields = {
            "schema_version",
            "stage",
            "stage_attempt",
            "idempotency_key",
            "tool_id",
            "wave_digest",
            "sample_ids",
            "prediction_count",
            "request_digest",
            "output_digest",
            "execution_owner",
        }
        if not isinstance(tool_payload, Mapping) or set(tool_payload) != tool_fields:
            raise ValueError(invalid_message)
        sample_ids = tool_payload.get("sample_ids")
        if (
            tool_payload.get("schema_version")
            != "ecologyrsi-dsh.dsh-prediction-tool-executed/1"
            or tool_payload.get("stage") != "sample.plan"
            or tool_payload.get("execution_owner") != "dsh_agent_tool_call"
            or isinstance(tool_payload.get("stage_attempt"), bool)
            or not isinstance(tool_payload.get("stage_attempt"), int)
            or tool_payload["stage_attempt"] < 0
            or not isinstance(tool_payload.get("idempotency_key"), str)
            or not tool_payload["idempotency_key"]
            or not isinstance(tool_payload.get("tool_id"), str)
            or not tool_payload["tool_id"]
            or not isinstance(sample_ids, list)
            or not sample_ids
            or not all(isinstance(item, str) and item for item in sample_ids)
            or len(sample_ids) != len(set(sample_ids))
            or isinstance(tool_payload.get("prediction_count"), bool)
            or not isinstance(tool_payload.get("prediction_count"), int)
            or tool_payload["prediction_count"] != len(sample_ids)
            or any(
                not isinstance(tool_payload.get(name), str)
                or len(tool_payload[name]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in tool_payload[name]
                )
                for name in ("wave_digest", "request_digest", "output_digest")
            )
            or tool_payload["stage_attempt"] != identity["stage_attempt"]
            or tool_payload["idempotency_key"] != identity["idempotency_key"]
            or tool_payload["wave_digest"] != structured.get("wave_digest")
            or receipt["request_digest"] != tool_payload["request_digest"]
            or receipt["output_digest"] != tool_payload["output_digest"]
        ):
            raise ValueError(invalid_message)
        expected_request_digest = digest(
            {
                "tool_name": _PREDICTION_TOOL_NAME,
                "run_id": run_id,
                "stage": "sample.plan",
                "stage_attempt": tool_payload["stage_attempt"],
                "idempotency_key": tool_payload["idempotency_key"],
                "arguments": {
                    "tool_id": tool_payload["tool_id"],
                    "wave_digest": tool_payload["wave_digest"],
                },
            }
        )
        if tool_payload["request_digest"] != expected_request_digest:
            raise ValueError(invalid_message)

        decisions = structured.get("decisions")
        if not isinstance(decisions, list) or not all(
            isinstance(item, Mapping)
            and isinstance(item.get("sample_id"), str)
            and item.get("sample_id")
            and item.get("next_tool") == tool_payload["tool_id"]
            for item in decisions
        ):
            raise ValueError(invalid_message)
        decision_ids = [str(item["sample_id"]) for item in decisions]
        if len(decision_ids) != len(sample_ids) or set(decision_ids) != set(
            sample_ids
        ):
            raise ValueError(invalid_message)

    def _validate_recorded_structured_result(
        self,
        event: Any,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        payload = event.payload
        required_fields = {
            "schema_version",
            "identity",
            "output_schema_id",
            "result_digest",
            "structured",
            "skill_invocation_evidence",
        }
        if (
            not isinstance(payload, Mapping)
            or not required_fields.issubset(payload)
            or set(payload)
            - (required_fields | {"session_metrics", "required_tool_receipt"})
        ):
            raise ValueError("recorded structured result has an invalid shape")
        if (
            payload.get("schema_version")
            != "ecologyrsi-dsh.structured-result-accepted/1"
        ):
            raise ValueError("recorded structured result has an unsupported version")
        identity = payload.get("identity")
        if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS:
            raise ValueError("recorded structured result identity is invalid")
        self._validate_identity_static(identity)
        stage = identity.get("stage")
        expected_contract = _STRUCTURED_STAGE_CONTRACTS.get(stage)
        if expected_contract is None or (
            identity.get("role"), payload.get("output_schema_id")
        ) != expected_contract:
            raise ValueError("recorded structured result stage contract mismatch")
        if identity.get("run_id") != event.run_id:
            raise ValueError("recorded structured result run identity mismatch")
        _skill_invocation_evidence(
            payload.get("skill_invocation_evidence"),
            stage=str(stage),
        )
        if "session_metrics" in payload:
            _dsh_session_metrics(
                payload["session_metrics"],
                session_id=str(identity["session_id"]),
            )
        structured = payload.get("structured")
        if (
            not isinstance(structured, Mapping)
            or payload.get("result_digest") != digest(structured)
        ):
            raise ValueError("recorded structured result digest mismatch")
        if stage == "sample.plan":
            self._validate_recorded_planner_binding(event)
        elif "required_tool_receipt" in payload:
            raise ValueError(
                "recorded non-Planner structured result has a tool receipt"
            )
        return identity, structured

    def _authorize_identity(
        self,
        identity: Mapping[str, Any],
        *,
        tool_name: str | None = None,
        expected_role: str | None = None,
        allow_ledger_advance: bool = False,
        require_open_fence: bool = True,
    ) -> str:
        role = self._validate_identity_static(
            identity,
            tool_name=tool_name,
            expected_role=expected_role,
        )
        run_id = str(identity["run_id"])
        if not self.ledger.events(run_id):
            raise DshToolAuthorizationError("unknown DSH tool run")
        expected_ledger_revision = int(identity["ledger_expected_revision"])
        current_ledger_revision = self.ledger.latest_seq()
        if (
            expected_ledger_revision > current_ledger_revision
            or (not allow_ledger_advance and expected_ledger_revision != current_ledger_revision)
        ):
            raise DshToolAuthorizationError("ledger expected revision is stale")
        if require_open_fence:
            fence_key = (
                run_id,
                int(identity["run_state_revision"]),
                int(identity["stage_attempt"]),
            )
            with self._admission_registry_lock:
                fence = self._fences.get(fence_key)
            if fence is None:
                raise DshToolAdmissionClosedError("stage admission is closed")
            with fence.lock:
                if fence.state != "open":
                    raise DshToolAdmissionClosedError(
                        "stage admission is closed"
                    )
        return role

    def accept_structured(self, envelope: Mapping[str, Any]) -> dict[str, Any]:
        required_fields = {
            "identity",
            "output_schema_id",
            "structured",
            "result_digest",
            "skill_invocation_evidence",
            "admission_id",
        }
        if "admission_id" not in envelope:
            raise ValueError("DSH structured envelope requires admission_id")
        if not required_fields.issubset(envelope) or set(envelope) - (
            required_fields | {"session_metrics"}
        ):
            raise ValueError("DSH structured envelope has an invalid shape")
        identity = envelope["identity"]
        structured = envelope["structured"]
        output_schema_id = envelope["output_schema_id"]
        supplied_digest = envelope["result_digest"]
        admission_id = envelope["admission_id"]
        if not isinstance(admission_id, str) or not admission_id.strip():
            raise ValueError("admission_id must be non-empty text")
        if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS:
            raise DshToolAuthorizationError("invalid Host-bound DSH structured identity")
        if not isinstance(structured, Mapping):
            raise TypeError("DSH structured result must be an object")
        if not isinstance(output_schema_id, str) or not output_schema_id.strip():
            raise ValueError("output_schema_id must be non-empty text")
        expected = _STRUCTURED_STAGE_CONTRACTS.get(str(identity.get("stage") or ""))
        if expected is None or output_schema_id != expected[1]:
            raise DshToolAuthorizationError("structured result schema is not allowed for its stage")
        actual_digest = digest(structured)
        if supplied_digest != actual_digest:
            raise ValueError("DSH structured result digest mismatch")
        skill_evidence = _skill_invocation_evidence(
            envelope["skill_invocation_evidence"],
            stage=str(identity.get("stage") or ""),
        )
        payload = {
            "schema_version": "ecologyrsi-dsh.structured-result-accepted/1",
            "identity": dict(identity),
            "output_schema_id": output_schema_id,
            "result_digest": actual_digest,
            "structured": deepcopy(dict(structured)),
            "skill_invocation_evidence": skill_evidence,
        }
        if "session_metrics" in envelope:
            payload["session_metrics"] = _dsh_session_metrics(
                envelope["session_metrics"],
                session_id=str(identity["session_id"]),
            )
        self._validate_identity_static(identity, expected_role=expected[0])
        event_id = self._structured_event_id(
            str(identity["run_id"]),
            str(identity["stage"]),
            str(identity["idempotency_key"]),
        )

        def prior_receipt(
            *, allow_recorded_tool_receipt: bool = False
        ) -> dict[str, Any] | None:
            prior = self._event_by_id(str(identity["run_id"]), event_id)
            if prior is None:
                return None
            if prior.kind != "DshStructuredResultAccepted":
                raise ValueError("structured-result idempotency key was reused")
            self._validate_recorded_structured_result(prior)
            matches_payload = _strict_json_equal(prior.payload, payload)
            if not matches_payload and allow_recorded_tool_receipt:
                recorded_payload = dict(prior.payload)
                recorded_payload.pop("required_tool_receipt", None)
                matches_payload = _strict_json_equal(
                    recorded_payload,
                    payload,
                )
            if not matches_payload:
                raise ValueError("structured-result idempotency key was reused")
            return {
                "accepted": True,
                "result_digest": actual_digest,
                "event_id": prior.event_id,
                "event_seq": prior.seq,
            }

        receipt = prior_receipt(
            allow_recorded_tool_receipt=identity.get("stage") == "sample.plan"
        )
        if receipt is not None:
            return receipt

        required_tool_receipt: dict[str, Any] | None = None
        if identity.get("stage") == "sample.plan":
            key = (
                str(identity["run_id"]),
                int(identity["stage_attempt"]),
                str(identity["idempotency_key"]),
            )
            with self._prediction_lock:
                binding = self._prediction_bindings.get(key)
            if binding is None:
                raise DshToolAdmissionClosedError(
                    "sample.plan has no active prediction tool binding"
                )
            if structured.get("wave_digest") != binding.wave_digest:
                raise DshToolAuthorizationError(
                    "sample.plan result does not match its prediction wave"
                )
            decisions = structured.get("decisions")
            if not isinstance(decisions, list):
                raise DshToolAuthorizationError("sample.plan decisions are missing")
            decision_ids = [
                item.get("sample_id") if isinstance(item, Mapping) else None
                for item in decisions
            ]
            if (
                len(decision_ids) != len(binding.sample_ids)
                or set(decision_ids) != set(binding.sample_ids)
                or any(
                    not isinstance(item, Mapping)
                    or item.get("next_tool") != binding.tool_id
                    for item in decisions
                )
            ):
                raise DshToolAuthorizationError(
                    "sample.plan must submit one frozen-tool decision per prediction"
                )
            required_tool_receipt = binding.require_session_call(
                str(identity["session_id"])
            )
        if required_tool_receipt is not None:
            payload["required_tool_receipt"] = required_tool_receipt
        fence_key = (
            str(identity["run_id"]),
            int(identity["run_state_revision"]),
            int(identity["stage_attempt"]),
        )
        with self._admission_registry_lock:
            fence = self._fences.get(fence_key)
        if fence is None:
            raise DshToolOperationalTimeoutError(
                "structured admission is unavailable"
            )

        def require_matching_fence() -> None:
            if (
                fence.admission_id != admission_id
                or fence.role != identity["role"]
                or fence.stage != identity["stage"]
                or fence.idempotency_key != identity["idempotency_key"]
                or fence.timeout_ms is None
                or fence.deadline_monotonic_ms is None
            ):
                raise DshToolOperationalTimeoutError(
                    "structured admission binding mismatch"
                )

        with fence.lock:
            require_matching_fence()
        receipt = prior_receipt()
        if receipt is not None:
            return receipt
        with fence.lock:
            require_matching_fence()
            fence_open = fence.state == "open"
            deadline_open = self._monotonic_ms() < fence.deadline_monotonic_ms
        if not fence_open or not deadline_open:
            # A concurrent exact commit may have linearized after the first
            # lookup.  Re-read before rejecting; a conflicting payload still
            # fails idempotency validation.
            receipt = prior_receipt()
            if receipt is not None:
                return receipt
            if not fence_open:
                raise DshToolAdmissionClosedError("stage admission is closed")
            raise DshToolOperationalTimeoutError("structured role deadline expired")
        self._authorize_identity(
            identity,
            expected_role=expected[0],
            allow_ledger_advance=True,
            require_open_fence=False,
        )
        with fence.lock:
            require_matching_fence()
            fence_open = fence.state == "open"
            deadline_open = self._monotonic_ms() < fence.deadline_monotonic_ms
        if not fence_open or not deadline_open:
            receipt = prior_receipt()
            if receipt is not None:
                return receipt
            if not fence_open:
                raise DshToolAdmissionClosedError("stage admission is closed")
            raise DshToolOperationalTimeoutError("structured role deadline expired")

        @contextmanager
        def commit_guard(inserted: bool) -> Iterator[None]:
            if not inserted:
                yield
                return
            with fence.lock:
                require_matching_fence()
                if fence.state != "open":
                    raise DshToolAdmissionClosedError(
                        "stage admission is closed"
                    )
                if self._monotonic_ms() >= fence.deadline_monotonic_ms:
                    raise DshToolOperationalTimeoutError(
                        "structured result deadline expired"
                    )
                yield

        # The child has already completed its prediction-tool call. A brief
        # SQLite/control-plane failure at this point must never be translated
        # into a scientific sample failure. Retry only the idempotent append
        # with the same event ID and exact payload; no tool or child work is
        # repeated here. The native caller performs a second bounded retry if
        # this local window remains unavailable.
        for persistence_attempt in range(2):
            try:
                event = self.ledger.append(
                    str(identity["run_id"]),
                    "DshStructuredResultAccepted",
                    payload,
                    event_id=event_id,
                    commit_guard=commit_guard,
                )
                break
            except ValueError:
                # A competing writer may have won after every prior lookup.
                # Re-read through the durable validator so malformed winners
                # fail closed and valid conflicts retain the structured
                # idempotency error.
                receipt = prior_receipt()
                if receipt is not None:
                    return receipt
                raise
            except (sqlite3.OperationalError, ConcurrentRunMutationError) as exc:
                # The insert may have committed before the caller observed its
                # transient failure. Prefer its exact durable receipt before
                # attempting the same append again.
                receipt = prior_receipt()
                if receipt is not None:
                    return receipt
                if persistence_attempt == 1:
                    raise DshStructuredResultPersistenceError(
                        "structured result persistence is temporarily unavailable"
                    ) from exc
                time.sleep(0.01)
        else:  # pragma: no cover - loop always breaks or raises
            raise AssertionError("structured persistence retry was not resolved")
        self._validate_recorded_structured_result(event)
        if not _strict_json_equal(event.payload, payload):
            raise ValueError("structured-result idempotency key was reused")
        return {
            "accepted": True,
            "result_digest": actual_digest,
            "event_id": event.event_id,
            "event_seq": event.seq,
        }

    def replay_structured_result(
        self,
        *,
        run_id: str,
        stage: str,
        role: str,
        stage_attempt: int,
        idempotency_key: str,
        output_schema_id: str,
        identity_digests: Mapping[str, str],
    ) -> dict[str, Any] | None:
        """Return one previously accepted result without launching another child."""

        event_id = self._structured_event_id(run_id, stage, idempotency_key)
        prior = self._event_by_id(run_id, event_id)
        if prior is None:
            return None
        if prior.kind != "DshStructuredResultAccepted":
            raise ValueError("structured-result event identity was reused")
        payload = prior.payload
        expected_contract = _STRUCTURED_STAGE_CONTRACTS.get(stage)
        if expected_contract != (role, output_schema_id):
            raise DshToolAuthorizationError(
                "structured replay role/schema does not match its stage"
            )
        identity, structured = self._validate_recorded_structured_result(prior)
        expected_identity = {
            "run_id": run_id,
            "stage": stage,
            "role": role,
            "stage_attempt": stage_attempt,
            "idempotency_key": idempotency_key,
        }
        if any(
            not _strict_json_equal(identity.get(name), value)
            for name, value in expected_identity.items()
        ):
            raise DshToolAuthorizationError("structured replay identity mismatch")
        if set(identity_digests) != set(_GENOME_IDENTITY_DIGEST_FIELDS) or any(
            identity.get(name) != identity_digests.get(name)
            for name in _GENOME_IDENTITY_DIGEST_FIELDS
        ):
            raise DshToolAuthorizationError(
                "structured replay genome identity mismatch"
            )
        if payload.get("output_schema_id") != output_schema_id:
            raise DshToolAuthorizationError("structured replay schema mismatch")
        if stage == "sample.plan":
            key = (run_id, stage_attempt, idempotency_key)
            with self._prediction_lock:
                binding = self._prediction_bindings.get(key)
            if binding is not None:
                if structured.get("wave_digest") != binding.wave_digest:
                    raise DshToolAuthorizationError(
                        "sample.plan replay does not match its prediction wave"
                    )
                if not _strict_json_equal(
                    payload.get("required_tool_receipt"),
                    binding.audit_receipt(),
                ):
                    raise ValueError(
                        "sample.plan replay prediction-tool receipt mismatch"
                    )
        return deepcopy(dict(structured))


__all__ = [
    "AdmissionFence",
    "DshPredictionBindingClosedError",
    "DshToolAdmissionClosedError",
    "DshToolAuthorizationError",
    "DshPredictionToolBinding",
    "DshToolService",
    "ROLE_TOOLS",
]
