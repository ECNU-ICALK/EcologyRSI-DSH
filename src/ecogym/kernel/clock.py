"""Session budgets in awake time, and keeping the machine awake while agents run.

A Mac that sleeps (a closed lid, idle sleep) stops every agent, model call and judge, but the wall clock keeps
running, so a budget measured on it is spent while nothing can work: on 2026-10-30 six episodes of a run were
awake for only 100 to 224 seconds of their 20 minutes, two of them ended with only the naive template, and
the evolver of the last round was stopped after less than two minutes awake. ``now()`` is the wall clock at import
plus the monotonic time since, and the monotonic clock stops while the machine sleeps (``mach_absolute_time``
on macOS, ``CLOCK_MONOTONIC`` on Linux): deadlines taken from it count only the time the machine was awake.
Its values look like wall-clock times but fall behind after every sleep, so show a deadline with ``wall()``
and record event times with ``time.time()``.

``awake()`` also keeps a Mac awake while a run or a session is going (``caffeinate -i -s``): no idle sleep,
and on AC power no sleep at all, a closed lid included (the 2026-10-30 run slept on AC with the lid closed).
On battery a closed lid still sleeps the machine; the awake clock is what keeps budgets fair then.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

_WALL, _MONO = time.time(), time.monotonic()
CAFFEINATE = Path("/usr/bin/caffeinate")


def now() -> float:
    """Seconds on the awake clock: the wall clock at import plus the time the machine has been awake since."""

    return _WALL + (time.monotonic() - _MONO)


def wall(moment: float) -> float:
    """The wall-clock time an awake-clock ``moment`` falls on if the machine stays awake until then."""

    return time.time() + (moment - now())


def asleep_since(wall_start: float, awake_start: float) -> float:
    """Seconds the machine slept since ``wall_start`` (``time.time()``) and ``awake_start`` (``now()``)."""

    return max(0.0, (time.time() - wall_start) - (now() - awake_start))


@contextlib.contextmanager
def awake() -> Iterator[None]:
    """Hold off sleep until the block ends (macOS ``caffeinate``; nothing elsewhere or if it is missing).

    ``-i`` stops idle sleep, ``-s`` all system sleep while on AC power. ``-w`` ties the assertions to this
    process as well, so they never outlive a process that dies abruptly. Nested blocks are harmless.
    """

    process = None
    if sys.platform == "darwin" and CAFFEINATE.exists():
        with contextlib.suppress(OSError):
            process = subprocess.Popen([str(CAFFEINATE), "-i", "-s", "-w", str(os.getpid())],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield
    finally:
        if process is not None:
            process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)


__all__ = ["asleep_since", "awake", "now", "wall"]
