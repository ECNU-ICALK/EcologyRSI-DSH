"""Authenticated DSH role-tool boundary with fail-closed admission."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterator

from ..core.ledger import ConcurrentRunMutationError
from ..core.models import digest
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

_STRUCTURED_STAGE_SKILLS: dict[str, frozenset[str]] = {
    "generation.research": frozenset({"autonomous-ecology-research"}),
    "generation.search-plan": frozenset({"autonomous-ecology-research"}),
    "generation.research-synthesis": frozenset({"autonomous-ecology-research"}),
    "generation.reflect": frozenset({"batch-scientific-reflection"}),
    "candidate.propose": frozenset({"bounded-plugin-experiment"}),
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


@dataclass(frozen=True, slots=True)
class AdmissionFence:
    run_id: str
    run_state_revision: int
    stage_attempt: int
    state: str


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
    ) -> None:
        self.ledger = ledger
        self._fences: dict[tuple[str, int, int], AdmissionFence] = {}
        self._run_admission: dict[str, str] = {}
        self._prediction_bindings: dict[
            tuple[str, int, str], DshPredictionToolBinding
        ] = {}
        self._prediction_lock = Lock()
        self._launch_lock = Lock()
        self._retrieval_lock = Lock()
        self._retrieval_fallback = retrieval_fallback or search_openalex_metadata

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
        self, run_id: str, run_state_revision: int, stage_attempt: int
    ) -> AdmissionFence:
        key = (run_id, run_state_revision, stage_attempt)
        if self._run_admission.get(run_id, "open") != "open":
            raise DshToolAdmissionClosedError("run admission is closed")
        prior = self._fences.get(key)
        if prior is not None and prior.state != "open":
            raise DshToolAdmissionClosedError("admission fence is permanently closed")
        fence = AdmissionFence(run_id, run_state_revision, stage_attempt, "open")
        self._fences[key] = fence
        return fence

    def close_admission(
        self, run_id: str, run_state_revision: int, stage_attempt: int
    ) -> AdmissionFence:
        key = (run_id, run_state_revision, stage_attempt)
        fence = AdmissionFence(run_id, run_state_revision, stage_attempt, "closed")
        self._fences[key] = fence
        return fence

    def close_run_admissions(self, run_id: str) -> tuple[AdmissionFence, ...]:
        self._run_admission[run_id] = "closed"
        closed: list[AdmissionFence] = []
        for key, prior in tuple(self._fences.items()):
            if key[0] != run_id or prior.state != "open":
                continue
            fence = AdmissionFence(key[0], key[1], key[2], "closed")
            self._fences[key] = fence
            closed.append(fence)
        return tuple(closed)

    def open_run_admissions(self, run_id: str) -> None:
        self._run_admission[run_id] = "open"

    def allocate_child_reservation(self, request: Mapping[str, Any]) -> dict[str, Any]:
        expected_fields = {
            "request_id",
            "run_id",
            "parent_session_id",
            "role",
            "stage",
            "item_digest",
            "idempotency_key",
        }
        if not isinstance(request, Mapping) or set(request) != expected_fields:
            raise ValueError("child reservation request has an invalid shape")
        for name in (
            "request_id",
            "run_id",
            "parent_session_id",
            "role",
            "stage",
            "idempotency_key",
        ):
            if not isinstance(request[name], str) or not request[name].strip():
                raise ValueError(f"child reservation {name} must be non-empty text")
        item_digest = request["item_digest"]
        if (
            not isinstance(item_digest, str)
            or len(item_digest) != 64
            or any(character not in "0123456789abcdef" for character in item_digest)
        ):
            raise ValueError("child reservation item_digest must be a SHA-256 digest")
        run_id = str(request["run_id"])
        if self._run_admission.get(run_id, "open") != "open":
            raise DshToolAdmissionClosedError("run admission is closed")
        if not self.ledger.events(run_id):
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
        with self._launch_lock:
            while True:
                events = self.ledger.events(run_id)
                prior = self._event_by_id(run_id, event_id)
                if prior is not None:
                    if (
                        prior.kind != "DshChildLaunchReserved"
                        or prior.payload.get("request_id") != request["request_id"]
                        or prior.payload.get("parent_session_id")
                        != request["parent_session_id"]
                        or prior.payload.get("business_key_digest")
                        != business_key_digest
                    ):
                        raise ValueError(
                            "child reservation request_id was reused with different input"
                        )
                    launch = prior.payload["launch"]
                    break
                matching = [
                    event.payload["launch"]
                    for event in events
                    if event.kind == "DshChildLaunchReserved"
                    and event.payload.get("business_key_digest")
                    == business_key_digest
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
                try:
                    self.ledger.append(
                        run_id,
                        "DshChildLaunchReserved",
                        {
                            "schema_version": "ecologyrsi-dsh.child-launch-reserved/1",
                            "request_id": request["request_id"],
                            "parent_session_id": request["parent_session_id"],
                            "business_key_digest": business_key_digest,
                            "launch": launch,
                        },
                        event_id=event_id,
                        expected_run_seq=events[-1].seq,
                    )
                except ConcurrentRunMutationError:
                    continue
                break
        return {
            "accepted": True,
            "launch": dict(launch),
            "ledger_expected_revision": self.ledger.latest_seq(),
        }

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
            raise DshToolAdmissionClosedError(
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

    def _authorize_identity(
        self,
        identity: Mapping[str, Any],
        *,
        tool_name: str | None = None,
        expected_role: str | None = None,
        allow_ledger_advance: bool = False,
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
        fence_key = (
            run_id,
            int(identity["run_state_revision"]),
            int(identity["stage_attempt"]),
        )
        fence = self._fences.get(fence_key)
        if fence is None or fence.state != "open":
            raise DshToolAdmissionClosedError("stage admission is closed")
        return role

    def accept_structured(self, envelope: Mapping[str, Any]) -> dict[str, Any]:
        required_fields = {
            "identity",
            "output_schema_id",
            "structured",
            "result_digest",
            "skill_invocation_evidence",
        }
        if not required_fields.issubset(envelope) or set(envelope) - (
            required_fields | {"session_metrics"}
        ):
            raise ValueError("DSH structured envelope has an invalid shape")
        identity = envelope["identity"]
        structured = envelope["structured"]
        output_schema_id = envelope["output_schema_id"]
        supplied_digest = envelope["result_digest"]
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
        payload = {
            "schema_version": "ecologyrsi-dsh.structured-result-accepted/1",
            "identity": dict(identity),
            "output_schema_id": output_schema_id,
            "result_digest": actual_digest,
            "structured": deepcopy(dict(structured)),
            "skill_invocation_evidence": skill_evidence,
        }
        if required_tool_receipt is not None:
            payload["required_tool_receipt"] = required_tool_receipt
        if "session_metrics" in envelope:
            payload["session_metrics"] = _dsh_session_metrics(
                envelope["session_metrics"],
                session_id=str(identity["session_id"]),
            )
        event_id = self._structured_event_id(
            str(identity["run_id"]),
            str(identity["stage"]),
            str(identity["idempotency_key"]),
        )
        prior = self._event_by_id(str(identity["run_id"]), event_id)
        if prior is not None:
            if prior.kind != "DshStructuredResultAccepted" or prior.payload != payload:
                raise ValueError("structured-result idempotency key was reused")
            return {
                "accepted": True,
                "result_digest": actual_digest,
                "event_id": prior.event_id,
                "event_seq": prior.seq,
            }
        self._authorize_identity(
            identity,
            expected_role=expected[0],
            allow_ledger_advance=True,
        )
        event = self.ledger.append(
            str(identity["run_id"]),
            "DshStructuredResultAccepted",
            payload,
            event_id=event_id,
        )
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
        required_fields = {
            "schema_version",
            "identity",
            "output_schema_id",
            "result_digest",
            "structured",
            "skill_invocation_evidence",
        }
        if not isinstance(payload, Mapping) or not required_fields.issubset(payload):
            raise ValueError("recorded structured result has an invalid shape")
        if set(payload) - (
            required_fields | {"session_metrics", "required_tool_receipt"}
        ):
            raise ValueError("recorded structured result has an invalid shape")
        if (
            payload.get("schema_version")
            != "ecologyrsi-dsh.structured-result-accepted/1"
        ):
            raise ValueError("recorded structured result has an unsupported version")
        expected_contract = _STRUCTURED_STAGE_CONTRACTS.get(stage)
        if expected_contract != (role, output_schema_id):
            raise DshToolAuthorizationError(
                "structured replay role/schema does not match its stage"
            )
        identity = payload.get("identity")
        if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS:
            raise ValueError("recorded structured result identity is invalid")
        expected_identity = {
            "run_id": run_id,
            "stage": stage,
            "role": role,
            "stage_attempt": stage_attempt,
            "idempotency_key": idempotency_key,
        }
        if any(
            identity.get(name) != value
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
        _skill_invocation_evidence(
            payload.get("skill_invocation_evidence"),
            stage=stage,
        )
        structured = payload.get("structured")
        result_digest = payload.get("result_digest")
        if not isinstance(structured, Mapping) or result_digest != digest(structured):
            raise ValueError("recorded structured result digest mismatch")
        if stage == "sample.plan":
            key = (run_id, stage_attempt, idempotency_key)
            with self._prediction_lock:
                binding = self._prediction_bindings.get(key)
            if binding is None:
                raise DshToolAdmissionClosedError(
                    "sample.plan replay has no active prediction tool binding"
                )
            if structured.get("wave_digest") != binding.wave_digest:
                raise DshToolAuthorizationError(
                    "sample.plan replay does not match its prediction wave"
                )
            if payload.get("required_tool_receipt") != binding.audit_receipt():
                raise ValueError(
                    "sample.plan replay prediction-tool receipt mismatch"
                )
        return deepcopy(dict(structured))


__all__ = [
    "AdmissionFence",
    "DshToolAdmissionClosedError",
    "DshToolAuthorizationError",
    "DshPredictionToolBinding",
    "DshToolService",
    "ROLE_TOOLS",
]
