"""Fold an evolution run's ledger events into one readable record.

Round 0 runs the starting harness on every training task. Each later round
has an evolver session that reviews the evidence and proposes candidates,
the candidates' and the incumbent's episodes, one comparison per candidate
and at most one adoption. The loop's state (incumbent, its unbiased episodes,
the versions already tried) is recomputed from this record, so a run can be
resumed after any finished round; an adoption takes effect only when its
round finishes (a round cut short reruns with the old incumbent).
``owner`` is the process that runs the latest round.
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
})
SCORED_STATUSES = ("finished", "timeout")  # the agent ran; its submission counts, whatever its quality


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
    }
    current: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None  # the current round's adoption, applied when the round finishes
    for event in events[1:]:
        body = event.body
        if event.kind == "round_started":
            if view["status"] in ("failed", "stopped"):  # resumed after an interruption
                view.setdefault("interruptions", []).append(view.pop("error", None))
                view["status"] = "running"
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
            current["evolver"] = body
        elif event.kind in ("candidate_proposed", "candidate_rejected"):
            status = "proposed" if event.kind == "candidate_proposed" else body.get("status", "rejected")
            current["candidates"][body["name"]] = {**body, "status": status, "decision": None}
        elif event.kind == "episodes_finished":
            current["episodes"].extend(body["episodes"])
        elif event.kind == "candidate_compared":
            entry = current["candidates"].setdefault(body["name"], {"name": body["name"], "status": "proposed"})
            entry["decision"] = body["decision"]
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


def _closing(view: dict[str, Any], event: Event) -> None:
    if event.kind == "evolve_finished":
        view["status"] = "finished"
        view["finished_at"] = event.at
        view["result"] = event.body
    elif event.kind == "evolve_failed":
        view["status"] = "stopped" if event.body.get("stopped") else "failed"
        view["error"] = event.body.get("error")
    elif event.kind == "holdout_verified":
        view["verification"]["holdout"] = {"at": event.at, **event.body}
    elif event.kind == "sealed_verified":
        view["verification"]["sealed"][event.body["stage"]] = {"at": event.at, **event.body}


def summary(view: dict[str, Any]) -> dict[str, Any]:
    rounds = view["rounds"]
    episodes = [item for round_ in rounds for item in round_["episodes"]]
    sessions = [*episodes, *(round_["evolver"] for round_ in rounds if round_.get("evolver"))]
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
        "error": (view.get("error") or "")[:300] or None,
    }


__all__ = ["EVENT_KINDS", "SCORED_STATUSES", "START", "fold", "summary"]
