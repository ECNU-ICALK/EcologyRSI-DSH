"""Authenticated DSH role-tool boundary with fail-closed admission."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import Any

from ..core.ledger import ConcurrentRunMutationError
from ..core.models import digest


ROLE_TOOLS: dict[str, frozenset[str]] = {
    "coordinator": frozenset({"ecology_get_run_context"}),
    "researcher": frozenset(
        {"ecology_get_run_context", "ecology_get_research_evidence"}
    ),
    "candidate-proposer": frozenset(
        {
            "ecology_get_run_context",
            "ecology_get_research_evidence",
            "ecology_get_generation_summary",
        }
    ),
    "sample-planner": frozenset(
        {
            "ecology_get_run_context",
            "ecology_get_sample_wave",
            "ecology_execute_prediction_tool",
            "ecology_submit_sample_decisions",
        }
    ),
    "sample-critic": frozenset(
        {
            "ecology_get_sample_wave",
            "ecology_get_prediction_summary",
            "ecology_submit_sample_review",
        }
    ),
    "generation-judge": frozenset({"ecology_get_generation_summary"}),
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
        context_provider: Callable[[str, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
    ) -> None:
        self.ledger = ledger
        self.context_provider = context_provider
        self._fences: dict[tuple[str, int, int], AdmissionFence] = {}
        self._run_admission: dict[str, str] = {}
        self._receipts: dict[tuple[str, str, str], tuple[str, dict[str, Any]]] = {}
        self._launch_lock = Lock()

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
                prior = next(
                    (event for event in events if event.event_id == event_id),
                    None,
                )
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
        role = self._authorize_identity(identity, tool_name=tool_name)
        _assert_finite_json_shape(arguments, label_free=role == "sample-planner")
        request_digest = digest(
            {"tool_name": tool_name, "identity": dict(identity), "arguments": arguments}
        )
        receipt_key = (str(identity["run_id"]), str(identity["idempotency_key"]), tool_name)
        prior = self._receipts.get(receipt_key)
        if prior is not None:
            if prior[0] != request_digest:
                raise ValueError("idempotency key was reused with different tool input")
            return deepcopy(prior[1])
        if self.context_provider is not None:
            payload = dict(self.context_provider(tool_name, identity, arguments))
        else:
            payload = {
                "accepted": True,
                "tool_name": tool_name,
                "request_digest": request_digest,
            }
        payload.setdefault("accepted", True)
        payload.setdefault("request_digest", request_digest)
        self._receipts[receipt_key] = (request_digest, deepcopy(payload))
        return payload

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
        expected_by_stage = {
            "generation.research": ("researcher", "ecology-research-result@1"),
            "candidate.propose": ("candidate-proposer", "ecology-genome-mutation@1"),
            "generation.judge": ("generation-judge", "ecology-generation-review@1"),
            "sample.plan": ("sample-planner", "ecology-sample-decisions@1"),
            "sample.critic": ("sample-critic", "ecology-sample-review@1"),
        }
        expected = expected_by_stage.get(str(identity.get("stage") or ""))
        if expected is None or output_schema_id != expected[1]:
            raise DshToolAuthorizationError("structured result schema is not allowed for its stage")
        actual_digest = digest(structured)
        if supplied_digest != actual_digest:
            raise ValueError("DSH structured result digest mismatch")
        payload = {
            "schema_version": "ecologyrsi-dsh.structured-result-accepted/1",
            "identity": dict(identity),
            "output_schema_id": output_schema_id,
            "result_digest": actual_digest,
            "structured": deepcopy(dict(structured)),
        }
        if "session_metrics" in envelope:
            payload["session_metrics"] = _dsh_session_metrics(
                envelope["session_metrics"],
                session_id=str(identity["session_id"]),
            )
        event_id = (
            f"{identity['run_id']}:dsh-structured:"
            f"{digest({'stage': identity['stage'], 'idempotency_key': identity['idempotency_key']})}"
        )
        prior = next(
            (
                event
                for event in self.ledger.events(str(identity["run_id"]))
                if event.event_id == event_id
            ),
            None,
        )
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


__all__ = [
    "AdmissionFence",
    "DshToolAdmissionClosedError",
    "DshToolAuthorizationError",
    "DshToolService",
    "ROLE_TOOLS",
]
