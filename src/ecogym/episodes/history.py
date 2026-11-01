"""Fold an episode's ledger events into one record (for listings, evidence and export)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from ..kernel.ledger import Event, Ledger

START = "episode_started"


def fold_episode(events: Sequence[Event]) -> dict[str, Any]:
    if not events or events[0].kind != START:
        raise ValueError("an episode starts with episode_started")
    start = events[0]
    record: dict[str, Any] = {
        "episode": start.run_id,
        "started_at": start.at,
        "updated_at": events[-1].at,
        "status": "running",
        **{key: start.body.get(key) for key in ("task_id", "label", "harness", "model", "agent", "minutes", "tokens",
                                                "limits", "context", "root", "visible_rows", "boundaries", "dsh",
                                                "start", "replace_margin")},
        "evaluations": [],
        "submissions": [],
        "notes": [],
        "questions": [],
        "scored": None,
        "finished": None,
    }
    record["context"] = record["context"] or {}
    for event in events[1:]:
        body = {"at": event.at, **event.body}
        if event.kind == "episode_evaluated":
            record["evaluations"].append(body)
        elif event.kind == "episode_submitted":
            record["submissions"].append(body)
        elif event.kind == "episode_note":
            record["notes"].append(body)
        elif event.kind == "expert_question":
            record["questions"].append({"id": f"{event.run_id}#{event.seq}", **body})
        elif event.kind == "episode_scored":
            record["scored"] = event.body
        elif event.kind == "episode_finished":
            record["finished"] = event.body
            record["status"] = event.body.get("status", "finished")
    return record


def episode_ids(ledger: Ledger) -> list[str]:
    return ledger.runs(first=START)


def episode_records(ledger: Ledger, ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    return [fold_episode(ledger.events(run_id)) for run_id in (episode_ids(ledger) if ids is None else ids)]


def summary(record: dict[str, Any]) -> dict[str, Any]:
    """The listing fields of one episode record."""

    finished, scored = record.get("finished") or {}, record.get("scored") or {}
    best = max((item["local"]["score"] for item in record["submissions"] if item.get("accepted")), default=None)
    return {
        "episode": record["episode"], "task_id": record["task_id"], "label": record.get("label") or "",
        "status": record["status"], "started_at": record["started_at"], "updated_at": record["updated_at"],
        "harness": record.get("harness") or {}, "model": record.get("model"), "context": record["context"],
        "minutes": record.get("minutes"),
        "score": scored.get("score"), "interval": scored.get("interval"), "best_local": best,
        "submissions": len(record["submissions"]), "evaluations": len(record["evaluations"]),
        "notes": len(record["notes"]), "questions": len(record["questions"]),
        "seconds": finished.get("seconds"), "tokens": finished.get("tokens"),
        "metered_tokens": finished.get("metered_tokens"),
    }


__all__ = ["START", "episode_ids", "episode_records", "fold_episode", "summary"]
