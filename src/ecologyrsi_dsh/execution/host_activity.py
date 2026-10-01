"""Observe host scheduling gaps without guessing that every gap is sleep.

A dedicated heartbeat keeps a slow model request distinct from a suspended or
stalled host. Diagnostics never extend deadlines or authorize late results.
"""
from __future__ import annotations

import math
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable

HOST_INTERRUPTION_EVENT = "HostExecutionInterrupted"


def validate_host_interruption(payload):
    if (not isinstance(payload, dict)
            or set(payload) != {"schema_version", "started_at", "resumed_at", "gap_seconds", "evidence"}
            or payload["schema_version"] != "ecologyrsi-dsh.host-interruption/1"
            or payload["evidence"] != "host_heartbeat_gap"
            or type(payload["gap_seconds"]) not in (int, float)
            or not math.isfinite(payload["gap_seconds"])
            or payload["gap_seconds"] < 30):
        raise ValueError("invalid host interruption")
    dates = [datetime.fromisoformat(payload[k]) for k in ("started_at", "resumed_at")]
    if any(d.tzinfo is None for d in dates) or dates[1] <= dates[0]:
        raise ValueError("invalid host interruption interval")
    return payload


class HostActivityMonitor:
    def __init__(self, *, clock=time.monotonic, wall_clock=time.time,
                 on_gap: Callable[[dict], None] | None = None):
        self._clock, self._wall = clock, wall_clock
        self._last = clock()
        self._last_wall = wall_clock()
        self._epoch = 0
        self._on_gap = on_gap
        self._lock = threading.Lock()
        self._persist_lock = threading.Lock()
        self._pending: list[dict] = []
        self._stop = threading.Event()
        self._thread = None

    def observe(self) -> int:
        event = None
        with self._lock:
            now, wall = self._clock(), self._wall()
            elapsed = now - self._last
            wall_elapsed = wall - self._last_wall
            # Wall time also catches platforms whose monotonic clock excludes
            # suspend. Clock corrections can be scheduling-gap evidence only;
            # this deliberately does not claim OS-confirmed sleep.
            gap = max(elapsed, wall_elapsed)
            if gap >= 30:
                self._epoch += 1
                event = {"schema_version": "ecologyrsi-dsh.host-interruption/1",
                         "started_at": datetime.fromtimestamp(self._last_wall, timezone.utc).isoformat(),
                         "resumed_at": datetime.fromtimestamp(max(wall, self._last_wall + elapsed), timezone.utc).isoformat(),
                         "gap_seconds": round(gap, 3), "evidence": "host_heartbeat_gap"}
                self._pending.append(event)
            self._last, self._last_wall = now, wall
            epoch = self._epoch
        self._persist_pending()
        return epoch

    def _persist_pending(self):
        # Observational persistence must neither replace an execution error nor
        # leak an admission slot. Retry the same payload after a transient
        # ledger failure; the composition root supplies an idempotent event ID.
        if not self._persist_lock.acquire(blocking=False):
            return
        try:
            while True:
                with self._lock:
                    if not self._pending:
                        return
                    event = self._pending[0]
                try:
                    if self._on_gap is not None:
                        self._on_gap(event)
                except Exception:
                    logging.getLogger(__name__).exception("Host heartbeat persistence failed; will retry")
                    return
                with self._lock:
                    self._pending.pop(0)
        finally:
            self._persist_lock.release()

    def start(self):
        if self._thread is not None:
            return
        def heartbeat():
            while not self._stop.wait(5):
                try:
                    self.observe()
                except Exception:
                    # An observational write must not interrupt model work.
                    logging.getLogger(__name__).exception("Host heartbeat persistence failed")
        self._thread = threading.Thread(target=heartbeat, name="ecology-host-heartbeat", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
