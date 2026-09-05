"""Compact execution progress projection."""

from __future__ import annotations

from typing import Any, Mapping


def build_progress_projection(state: Any, *, admission: Mapping[str, object] | None = None) -> dict[str, object]:
    from .projection import _run_execution_progress

    progress = dict(_run_execution_progress(state, admission))
    lifecycle = ("local_waiting", "admission_waiting", "provider_queued", "provider_active", "workflow_running", "persisting", "completed", "failed", "retry_waiting", "cancelled", "draining")
    for key in lifecycle:
        progress.setdefault(key, 0)
    for candidate in getattr(state, "candidates", ()):
        status = getattr(getattr(candidate, "status", None), "value", "")
        if status in {"promoted", "evaluated", "rejected", "duplicate", "screened_out"}:
            progress["completed"] += 1
        elif status == "failed":
            progress["failed"] += 1
        elif status == "cancelled":
            progress["cancelled"] += 1
        else:
            progress["workflow_running"] += 1
    progress["planned"] = sum(int(progress.get(key, 0) or 0) for key in lifecycle)
    return progress
