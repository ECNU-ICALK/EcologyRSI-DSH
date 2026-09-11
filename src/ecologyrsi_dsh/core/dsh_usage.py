"""Provider usage contracts, independent of scientific result acceptance."""
from __future__ import annotations
from datetime import datetime
from collections.abc import Mapping
from typing import Any

USAGE_SCHEMA = "ecologyrsi-dsh.session-usage/1"
USAGE_EVENT = "DshSessionUsageRecorded"


def active_session_ids(events) -> set[str]:
    """A terminal child never becomes active again because telemetry arrived late."""
    active: dict[str, str] = {}
    terminal_reservations: set[str] = set()
    terminal_sessions: set[str] = set()
    for event in events:
        if event.kind not in {USAGE_EVENT, "DshStructuredResultAccepted", "DshChildExecutionFailed"}:
            continue
        identity = event.payload.get("identity", {})
        reservation = identity.get("child_reservation_id")
        session = identity.get("session_id")
        if event.kind == USAGE_EVENT:
            if event.payload.get("settlement") == "active":
                active[reservation] = session
            else:
                terminal_reservations.add(reservation)
                terminal_sessions.add(session)
        elif event.kind in {"DshStructuredResultAccepted", "DshChildExecutionFailed"}:
            if reservation:
                terminal_reservations.add(reservation)
            if session:
                terminal_sessions.add(session)
    return {session for reservation, session in active.items()
            if session and reservation not in terminal_reservations and session not in terminal_sessions}


def validate_session_usage(payload: Any, *, run_id: str) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version", "identity", "session_metrics", "settlement", "usage_complete"
    } or payload.get("schema_version") != USAGE_SCHEMA:
        raise ValueError("DSH usage record has an invalid shape")
    identity = payload["identity"]
    fields = {"run_id", "stage", "idempotency_key", "child_reservation_id", "session_id"}
    if not isinstance(identity, Mapping) or set(identity) != fields or any(
        not isinstance(identity[k], str) or not identity[k] or len(identity[k]) > 1000 for k in fields
    ) or identity["run_id"] != run_id:
        raise ValueError("DSH usage identity is invalid")
    if payload["settlement"] not in {"active", "succeeded", "failed", "cancelled", "timed_out"}:
        raise ValueError("DSH usage settlement is invalid")
    if not isinstance(payload["usage_complete"], bool):
        raise ValueError("DSH usage completeness must be a bool")
    validate_session_metrics(payload["session_metrics"], session_id=identity["session_id"])
    if payload["usage_complete"] and (payload["settlement"] == "active" or not payload["session_metrics"]["provider_usage"]["available"]):
        raise ValueError("active or unavailable usage cannot be complete")


def check_usage_binding(payload: Mapping, events: Any) -> None:
    events = tuple(events)
    identity = payload["identity"]
    launch = next((e.payload["launch"] for e in events
                   if e.kind == "DshChildLaunchReserved"
                   and e.payload["launch"].get("reservation_id") == identity["child_reservation_id"]), None)
    if launch is None or any(launch.get(k) != identity[k] for k in ("run_id", "stage", "idempotency_key")):
        raise ValueError("DSH usage has no matching durable child reservation")
    for event in events:
        if event.kind not in {USAGE_EVENT, "DshStructuredResultAccepted"}:
            continue
        previous = event.payload
        bound = previous.get("identity", {})
        if bound.get("child_reservation_id") != identity["child_reservation_id"]:
            if bound.get("session_id") == identity["session_id"]:
                raise ValueError("DSH usage session belongs to another reservation")
            continue
        if bound.get("session_id") != identity["session_id"]:
            raise ValueError("DSH usage child session identity changed")
        if event.kind != USAGE_EVENT:
            continue
        old = previous["session_metrics"]["provider_usage"]
        new = payload["session_metrics"]["provider_usage"]
        if old["available"] and (not new["available"] or any(new["totals"][k] < v for k, v in old["totals"].items())):
            raise ValueError("DSH cumulative usage cannot decrease")
        if previous["settlement"] != "active" and payload["settlement"] != previous["settlement"]:
            raise ValueError("DSH settled usage cannot change its outcome")
        if previous["usage_complete"] and not payload["usage_complete"]:
            raise ValueError("DSH complete usage cannot become incomplete")


def session_usage_projection(events: Any) -> tuple[dict, dict]:
    metrics_by_session = {}
    reservations = set()
    observed = {}
    for event in events:
        payload = event.payload
        if event.kind == "DshChildLaunchReserved":
            reservations.add(payload["launch"]["reservation_id"])
        if event.kind not in {USAGE_EVENT, "DshStructuredResultAccepted"}:
            continue
        metrics = payload.get("session_metrics")
        if not isinstance(metrics, Mapping):
            continue
        session_id = metrics["session_id"]
        previous = metrics_by_session.get(session_id)
        # Accepted results may carry an older cumulative snapshot than an
        # independently recorded usage update. Never sum snapshots or regress.
        new_total = metrics["provider_usage"].get("totals", {}).get("total_tokens", -1)
        old_total = previous[1]["provider_usage"].get("totals", {}).get("total_tokens", -1) if previous else -2
        if new_total >= old_total:
            metrics_by_session[session_id] = (event.seq, metrics)
        if event.kind == USAGE_EVENT:
            observed[payload["identity"]["child_reservation_id"]] = payload
    settled = sum(p["settlement"] != "active" for p in observed.values())
    # Interrupted streams can omit their last provider usage receipt even
    # when previous messages were fully metered. Never call these complete.
    complete = sum(p["usage_complete"] and p["settlement"] not in {"cancelled", "timed_out"}
                   for p in observed.values())
    return metrics_by_session, {
        "scope": "dsh_child_provider_usage", "launched_session_count": len(reservations),
        "observed_session_count": len(observed), "settled_session_count": settled,
        "complete_session_count": complete, "unobserved_session_count": len(reservations - set(observed)),
        "complete": bool(reservations) and complete == len(reservations),
        "includes_retrieval_provider_usage": False,
    }

def validate_session_metrics(value: Any, *, session_id: str) -> None:
    if not isinstance(value, Mapping) or set(value) - {"activity"} != {
        "schema_version",
        "session_id",
        "context_pressure",
        "provider_usage",
    }:
        raise ValueError("DSH session metrics have an invalid shape")
    if "activity" in value:
        activity = value["activity"]
        if not isinstance(activity, Mapping) or set(activity) != {"kind", "log_revision", "updated_at"}:
            raise ValueError("DSH session activity has an invalid shape")
        if activity["kind"] not in {"streaming", "retrying", "tool", "waiting", "settling"}:
            raise ValueError("DSH session activity kind is invalid")
        revision = activity["log_revision"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("DSH session activity revision is invalid")
        stamp = activity["updated_at"]
        if not isinstance(stamp, str) or len(stamp) > 40:
            raise ValueError("DSH session activity timestamp is invalid")
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("activity timestamp requires timezone")
        except ValueError as exc:
            raise ValueError("DSH session activity timestamp is invalid") from exc
    if value.get("schema_version") != "ecologyrsi-dsh.dsh-session-metrics/1":
        raise ValueError("unsupported DSH session metrics schema")
    if value.get("session_id") != session_id:
        raise ValueError("DSH session metrics identity mismatch")

    pressure = value.get("context_pressure")
    if not isinstance(pressure, Mapping):
        raise ValueError("DSH context pressure must be an object")
    if pressure.get("available") is True:
        if set(pressure) != {
            "available",
            "source",
            "measurement",
            "log_revision",
            "baseline_kind",
            "total_tokens",
            "surface_tokens",
        }:
            raise ValueError("DSH context pressure has an invalid shape")
        if (
            pressure.get("source") != "dsh_token_meter"
            or pressure.get("measurement") != "current_context_pressure"
            or pressure.get("baseline_kind") not in {"none", "estimated", "usage"}
        ):
            raise ValueError("DSH context pressure semantics are invalid")
        for name in ("log_revision", "total_tokens", "surface_tokens"):
            item = pressure.get(name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"DSH context pressure {name} is invalid")
    elif dict(pressure) != {"available": False, "source": "dsh_token_meter"}:
        raise ValueError("unavailable DSH context pressure is invalid")

    usage = value.get("provider_usage")
    if not isinstance(usage, Mapping):
        raise ValueError("DSH provider usage must be an object")
    if usage.get("available") is True:
        if set(usage) != {"available", "source", "measurement", "totals"}:
            raise ValueError("DSH provider usage has an invalid shape")
        if (
            usage.get("source") != "dsh_session_projection_token_usage"
            or usage.get("measurement") != "cumulative_provider_reported_usage"
        ):
            raise ValueError("DSH provider usage semantics are invalid")
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
