"""Where a run's starting version came from: the runs that evolved it, newest first.

A run can start from a version an earlier run adopted (``ecogym evolve run --harness <digest>``). Its own record then
shows no change from its start, though the start already holds the earlier run's gains; the panel says so. The
adoptions are followed back one run at a time: the run that adopted the version, the comparison that adopted it,
and that run's independent verification when the version is the run's final one. Only adoptions that took effect
count (a round cut short adopts nothing, see ``fold``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..kernel.ledger import Ledger
from .fold import fold

LIMIT = 5  # runs followed back


def _mean(values: Iterable[Any]) -> float | None:
    found = [float(value) for value in values if isinstance(value, (int, float))]
    return float(np.mean(found)) if found else None


def _verification(body: dict[str, Any] | None) -> dict[str, Any] | None:
    """The seed's and the final version's means, the gain with its interval and the verdict of one verification."""

    if not body:
        return None
    decision = body.get("decision") or {}
    means = body.get("means") or {}
    return {"seed": means.get("seed"), "final": means.get("final"), "delta": decision.get("delta"),
            "interval": decision.get("interval"), "reason": decision.get("reason"),
            "tasks": len(decision.get("tasks") or body.get("tasks") or [])}


def _step(view: dict[str, Any], digest: str) -> dict[str, Any] | None:
    """How the run ``view`` adopted ``digest``; None when it never did (or the adoption did not take effect)."""

    adoption = next((item for item in view["adoptions"] if item["digest"] == digest), None)
    if adoption is None:
        return None
    round_ = next((item for item in reversed(view["rounds"]) if item["generation"] == adoption["generation"]
                   and item.get("adopted") == adoption["name"]), {})
    candidate = next((item for item in round_.get("candidates", []) if item["name"] == adoption["name"]), {})
    decision = candidate.get("decision") or {}
    tasks = decision.get("tasks") or []
    final = view["incumbent"] == digest
    verification = view["verification"]
    return {"run_id": view["run_id"], "label": view.get("label", ""), "started_at": view["started_at"],
            "suite": (view.get("suite") or {}).get("id"),
            "seed": {key: view["seed"].get(key) for key in ("name", "digest")},
            "generation": adoption["generation"], "name": adoption["name"], "digest": digest,
            "summary": candidate.get("summary", ""),
            "before": _mean(item.get("incumbent") for item in tasks),
            "after": _mean(item.get("candidate") for item in tasks),
            "delta": decision.get("delta"), "interval": decision.get("interval"),
            "better": sum(1 for item in tasks if (item.get("delta") or 0) > 0), "tasks": len(tasks),
            "final": final,
            "holdout": _verification(verification.get("holdout")) if final else None,
            "sealed": {stage: _verification(body) for stage, body in (verification.get("sealed") or {}).items()}
            if final else {}}


def ancestry(ledger: Ledger, digest: str, *, exclude: str = "") -> list[dict[str, Any]]:
    """The runs that evolved ``digest`` (a run's starting version), newest first; the entry after a run's is about
    that run's own starting version. ``exclude`` is the run asking, never its own ancestor."""

    adopters: dict[str, list[str]] = {}
    for event in ledger.events(kind="harness_adopted"):
        runs = adopters.setdefault(str(event.body.get("digest")), [])
        if event.run_id != exclude and event.run_id not in runs:
            runs.append(event.run_id)
    chain: list[dict[str, Any]] = []
    seen = {exclude}
    while len(chain) < LIMIT:
        step = None
        for run_id in adopters.get(digest, []):
            if run_id in seen:
                continue
            try:
                step = _step(fold(ledger.events(run_id)), digest)
            except (ValueError, KeyError):
                continue
            if step is not None:
                seen.add(run_id)
                break
        if step is None:
            break
        chain.append(step)
        digest = str(step["seed"]["digest"])
    return chain


__all__ = ["ancestry"]
