"""Fold a run's ledger events into one readable record."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..kernel.ledger import Event


def fold(events: Sequence[Event]) -> dict[str, Any]:
    if not events or events[0].kind != "run_started":
        raise ValueError("a run starts with run_started")
    start = events[0]
    run: dict[str, Any] = {
        "run_id": start.run_id,
        "started_at": start.at,
        "status": "running",
        **start.body,
        "generations": [],
        "sealed": [],
        "agent_exchanges": 0,
    }
    current: dict[str, Any] | None = None
    for event in events[1:]:
        body = event.body
        if event.kind == "generation_started":
            current = {
                "generation": body["generation"],
                "incumbent": body["incumbent"],
                "cohorts": {"screening": body["screening"], "holdout": body["holdout"]},
                "proposed": [],
                "rejected": [],
                "screened": None,
                "comparison": None,
                "promoted": False,
            }
            run["generations"].append(current)
        elif event.kind == "candidate_proposed" and current is not None:
            current["proposed"].append(body)
        elif event.kind == "candidate_rejected" and current is not None:
            current["rejected"].append(body)
        elif event.kind == "candidates_screened" and current is not None:
            current["screened"] = body
        elif event.kind == "holdout_compared" and current is not None:
            current["comparison"] = body
        elif event.kind == "incumbent_promoted" and current is not None:
            current["promoted"] = True
        elif event.kind == "generation_finished" and current is not None:
            current["incumbent_after"] = body["incumbent"]
        elif event.kind == "run_finished":
            run["status"] = "finished"
            run["finished_at"] = event.at
            run["result"] = body
        elif event.kind == "run_failed":
            run["status"] = "failed"
            run["error"] = body.get("error")
        elif event.kind == "sealed_evaluated":
            run["sealed"].append(body)
        elif event.kind == "agent_exchange":
            run["agent_exchanges"] += 1
    return run


__all__ = ["fold"]
