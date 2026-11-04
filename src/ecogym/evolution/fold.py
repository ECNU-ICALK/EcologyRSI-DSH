"""Fold an evolution run's ledger events into one readable record.

Round 0 runs the starting harness on every training task. Each later round
has an evolver session that reviews the evidence and proposes candidates,
the candidates' and the incumbent's episodes, one comparison per candidate
and at most one adoption. The loop's state (incumbent, its unbiased episodes,
the versions already tried) is recomputed from this record, so a run can be
resumed after any finished round; an adoption takes effect only when its
round finishes (a round cut short reruns with the old incumbent).
``owner`` is the process that runs the run (named by ``evolve_resumed`` as
soon as a process continues it, then by each ``round_started``);
``stop_reason`` says why a stopped or failed run stopped. An adoption's check
on a sealed period (``adoption_checked``) is attached to its round as
``sealed_check``, an error never replacing a result. A run ended before its
last generation finishes with ``ended_early``. ``evolve_finished`` is final:
only verification events count after it (a process still running the run
cannot change its final version or reopen it).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..kernel.ledger import Event

START = "evolve_started"
EVENT_KINDS = frozenset({
    START,
    "round_started",
    "evolver_finished",
    "evolver_note",
    "expert_question",
    "candidate_proposed",
    "candidate_rejected",
    "episodes_finished",
    "candidate_compared",
    "harness_adopted",
    "round_finished",
    "evolve_finished",
    "evolve_failed",
    "holdout_verified",
    "sealed_task",  # one task's sealed result, recorded as soon as it is judged
    "sealed_verified",
    "adoption_checked",  # an adopted version against the one it replaced, on the training tasks' validation period
    "outage_started",  # a model service stopped answering: no session starts until it answers again
    "outage_ended",
    "evolve_resumed",  # who continued a stopped run (a person, or the service by itself) and why it had stopped
})
SCORED_STATUSES = ("finished", "timeout")  # the agent ran; its submission counts, whatever its quality
AFTER_FINISH = frozenset({"holdout_verified", "sealed_task", "sealed_verified", "adoption_checked"})


def fold(events: Sequence[Event]) -> dict[str, Any]:
    if not events or events[0].kind != START:
        raise ValueError("an evolution run starts with evolve_started")
    start = events[0]
    view: dict[str, Any] = {
        "run_id": start.run_id,
        "started_at": start.at,
        "updated_at": events[-1].at,
        "status": "running",
        **start.body,
        "rounds": [],
        "notes": [],
        "questions": [],
        "incumbent": start.body["seed"]["digest"],
        "adoptions": [],
        "verification": {"holdout": None, "sealed": {}},
        "last_seq": events[-1].seq,
    }
    current: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None  # the current round's adoption, applied when the round finishes
    for event in events[1:]:
        body = event.body
        if view["status"] == "finished" and event.kind not in AFTER_FINISH:
            view["ignored_after_finish"] = view.get("ignored_after_finish", 0) + 1
            continue
        if event.kind == "adoption_checked":  # attached to the round that adopted, whenever the check ran
            view["adoption_checks"] = [*view.get("adoption_checks", []), body]
            target = next((item for item in reversed(view["rounds"]) if item["generation"] == body.get("generation")
                           and item["finished"]), None)
            held = target.get("sealed_check") if target is not None else None
            if target is not None and not (body.get("error") and held and not held.get("error")):
                target["sealed_check"] = body
            continue
        if event.kind in ("outage_started", "outage_ended"):
            if event.kind == "outage_started":
                view["waiting"] = {"since": event.at, **body}
            else:
                view.pop("waiting", None)
            if current is not None:
                current.setdefault("outages", []).append({"at": event.at, "kind": event.kind, **body})
            continue
        if event.kind == "evolve_resumed":  # the resuming process owns the run from here, before its first round
            view.setdefault("resumes", []).append({"at": event.at, **body})
            _running(view)
            if body.get("owner"):
                view["owner"] = body["owner"]
            continue
        if event.kind == "round_started":
            view.pop("waiting", None)
            _running(view)  # resumed by a version that logged no evolve_resumed
            current = {"generation": body["generation"], "incumbent": body["incumbent"], "started_at": event.at,
                       "evolver": None, "notes": [], "candidates": {}, "episodes": [], "adopted": None,
                       "finished": False, "guidance": body.get("guidance")}
            pending = None
            view["rounds"].append(current)
            view["owner"] = body.get("owner")
        elif event.kind == "evolver_note":
            view["notes"].append({"at": event.at, **body})
            if current is not None:
                current["notes"].append({"at": event.at, **body})
        elif event.kind == "expert_question":
            view["questions"].append({"id": f"{event.run_id}#{event.seq}", "at": event.at, **body})
        elif current is None:
            if event.kind in ("evolve_finished", "evolve_failed", "holdout_verified", "sealed_verified"):
                _closing(view, event)
            continue
        elif event.kind == "evolver_finished":
            if current["evolver"] is not None:  # a session the service cut short, reviewed again (``rerun``)
                current.setdefault("earlier_evolvers", []).append(current["evolver"])
            current["evolver"] = body
        elif event.kind in ("candidate_proposed", "candidate_rejected"):
            status = "proposed" if event.kind == "candidate_proposed" else body.get("status", "rejected")
            current["candidates"][body["name"]] = {**body, "status": status, "decision": None}
        elif event.kind == "episodes_finished":
            current["episodes"].extend(body["episodes"])
        elif event.kind == "candidate_compared":
            entry = current["candidates"].setdefault(body["name"], {"name": body["name"], "status": "proposed"})
            entry["decision"] = body["decision"]
            entry["missing"] = body.get("missing") or []  # training tasks without usable episodes
            if body["decision"]["adopted"]:
                entry["status"] = "passed"
            elif entry["status"] == "proposed":
                entry["status"] = "not_adopted"
        elif event.kind == "harness_adopted":
            pending = body
        elif event.kind == "round_finished":
            current["finished"] = True
            current["finished_at"] = event.at
            current["incumbent_after"] = body["incumbent"]
            current["outcome"] = body.get("outcome")
            view["incumbent"] = body["incumbent"]
            if pending is not None and body.get("adopted") == pending["name"]:
                current["adopted"] = pending["name"]
                current["candidates"][pending["name"]]["status"] = "adopted"
                view["adoptions"].append({"generation": pending["generation"], "name": pending["name"],
                                          "digest": pending["digest"], "delta": pending.get("delta")})
            pending = None
        else:
            _closing(view, event)
    for round_ in view["rounds"]:
        round_["candidates"] = [round_["candidates"][name] for name in sorted(round_["candidates"])]
    return view


def _going_on(view: dict[str, Any], status: str) -> None:
    """A stopped or failed run goes on (``running``) or is ended (``finished``): why it stopped becomes history."""

    if view["status"] in ("failed", "stopped"):
        view.setdefault("interruptions", []).append(view.pop("error", None))
        view.pop("stop_reason", None)
    view["status"] = status


def _running(view: dict[str, Any]) -> None:
    _going_on(view, "running")


def _closing(view: dict[str, Any], event: Event) -> None:
    if event.kind == "evolve_finished":
        _going_on(view, "finished")
        view["finished_at"] = event.at
        view["result"] = event.body
        view["ended_early"] = bool(event.body.get("ended_early"))
    elif event.kind == "evolve_failed":
        view["status"] = "stopped" if event.body.get("stopped") else "failed"
        view["error"] = event.body.get("error")
        view["stop_reason"] = event.body.get("reason")  # user | shutdown | service_down | None (other failures)
        view.pop("waiting", None)
    elif event.kind == "holdout_verified":
        view["verification"]["holdout"] = {"at": event.at, **event.body}
    elif event.kind == "sealed_verified":
        view["verification"]["sealed"][event.body["stage"]] = {"at": event.at, **event.body}


def evolver_sessions(view: dict[str, Any]) -> list[dict[str, Any]]:
    """Every evolver session's record, those reviewed again after an outage included."""

    return [item for round_ in view["rounds"]
            for item in [*round_.get("earlier_evolvers", []), *([round_["evolver"]] if round_.get("evolver") else [])]]


def summary(view: dict[str, Any]) -> dict[str, Any]:
    rounds = view["rounds"]
    episodes = [item for round_ in rounds for item in round_["episodes"]]
    sessions = [*episodes, *evolver_sessions(view)]
    return {
        "run_id": view["run_id"], "label": view.get("label", ""), "status": view["status"],
        "started_at": view["started_at"], "updated_at": view["updated_at"], "finished_at": view.get("finished_at"),
        "suite": view["suite"], "seed": view["seed"], "incumbent": view["incumbent"],
        "generations": view["config"]["generations"],
        "rounds_finished": sum(1 for round_ in rounds if round_["finished"] and round_["generation"] > 0),
        "adoptions": len(view["adoptions"]), "episodes": len(episodes),
        "candidates": sum(len(round_["candidates"]) for round_ in rounds),
        "tokens": sum(item.get("tokens") or 0 for item in sessions),
        "metered_tokens": sum(item.get("metered_tokens") or 0 for item in sessions),
        "verified": {"holdout": view["verification"]["holdout"] is not None,
                     "sealed": sorted(view["verification"]["sealed"])},
        "ended_early": bool(view.get("ended_early")),
        "error": (view.get("error") or "")[:300] or None,
    }


__all__ = ["EVENT_KINDS", "SCORED_STATUSES", "START", "evolver_sessions", "fold", "summary"]
