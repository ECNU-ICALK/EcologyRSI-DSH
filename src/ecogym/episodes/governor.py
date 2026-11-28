"""Host-wide concurrency governor for sandboxed predictor judge subprocesses.

Outer evolution loops limit concurrent episodes, while each episode's local tools,
template screening, template attribution and sealed verification can all launch
judge subprocesses concurrently. A shared host-level governor caps active judge
subprocesses across the whole process so low-load settings actually hold.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager


class SlotCancelled(RuntimeError):
    """Raised when a caller's cancellation event fires while waiting for a compute slot."""


def _default_limit() -> int:
    raw = os.environ.get("ECOGYM_MAX_JUDGES", "").strip()
    if raw.isascii() and raw.isdigit() and int(raw) > 0:
        return int(raw)
    cpus = os.cpu_count() or 2
    return max(2, min(4, cpus))


class ComputeGovernor:
    """Reconfigurable, cancel-aware host concurrency gate for judge subprocesses."""

    def __init__(self, limit: int | None = None) -> None:
        self._cond = threading.Condition()
        self._capacity = max(1, int(limit)) if limit is not None else _default_limit()
        self._in_use = 0
        self._peak = 0

    @property
    def capacity(self) -> int:
        with self._cond:
            return self._capacity

    @property
    def in_use(self) -> int:
        with self._cond:
            return self._in_use

    @property
    def peak(self) -> int:
        with self._cond:
            return self._peak

    def configure(self, limit: int) -> None:
        """Update the concurrent slot cap dynamically (never blocks active holders)."""

        cap = max(1, int(limit))
        with self._cond:
            self._capacity = cap
            self._cond.notify_all()

    def reset_peak(self) -> None:
        with self._cond:
            self._peak = self._in_use

    @contextmanager
    def slot(self, cancel: threading.Event | None = None) -> Iterator[None]:
        with self._cond:
            while self._in_use >= self._capacity:
                if cancel is not None and cancel.is_set():
                    raise SlotCancelled("cancelled while waiting for a judge slot")
                self._cond.wait(timeout=0.2)
            if cancel is not None and cancel.is_set():
                raise SlotCancelled("cancelled while acquiring a judge slot")
            self._in_use += 1
            if self._in_use > self._peak:
                self._peak = self._in_use
        try:
            yield
        finally:
            with self._cond:
                self._in_use -= 1
                self._cond.notify()


_GOVERNOR = ComputeGovernor()


def judge_governor() -> ComputeGovernor:
    """Return the process-wide judge concurrency governor."""

    return _GOVERNOR


def configure_judge_limit(limit: int) -> None:
    """Set the host-wide judge cap unless pinned by ``ECOGYM_MAX_JUDGES``."""

    raw = os.environ.get("ECOGYM_MAX_JUDGES", "").strip()
    if raw.isascii() and raw.isdigit() and int(raw) > 0:
        _GOVERNOR.configure(int(raw))
        return
    _GOVERNOR.configure(limit)


__all__ = ["ComputeGovernor", "SlotCancelled", "configure_judge_limit", "judge_governor"]
