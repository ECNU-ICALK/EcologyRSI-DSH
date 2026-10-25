"""Best-effort, process-local idle-sleep protection for active evaluation work.

macOS defaults to enabled; ECOLOGYRSI_PREVENT_IDLE_SLEEP=0 disables it. Only
idle system sleep is inhibited: display sleep and explicit sleep stay available.
No power setting is changed and no assertion survives the owning Host process.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
from functools import wraps
import logging
import os
import subprocess
import sys
import threading


class IdleSleepInhibitor:
    def __init__(self, *, enabled=None, platform=None, popen=None, interval=1.0):
        configured = os.environ.get("ECOLOGYRSI_PREVENT_IDLE_SLEEP", "1").strip().lower()
        self.enabled = ((sys.platform if platform is None else platform) == "darwin"
                        and (configured not in {"0", "false", "off", "no"}
                             if enabled is None else bool(enabled)))
        self._popen = subprocess.Popen if popen is None else popen
        self._interval = interval
        self._lock = threading.RLock()
        self._leases = {}
        self._process = None
        self._thread = None
        self._stop = threading.Event()
        self._unavailable = False
        if self.enabled:
            atexit.register(self.close)

    @staticmethod
    def _active(predicate):
        try:
            return bool(predicate())
        except Exception:
            # This optional convenience must not fail or extend scientific work.
            logging.getLogger(__name__).warning("Cannot confirm active work; releasing idle-sleep protection")
            return False

    @contextmanager
    def hold(self, should_hold=lambda: True):
        token = object()
        try:
            if self.enabled and not self._stop.is_set() and self._active(should_hold):
                with self._lock:
                    if not self._stop.is_set():
                        self._leases[token] = should_hold
                        self._ensure_process_locked()
                        if self._thread is None and not self._unavailable:
                            self._thread = threading.Thread(target=self._watch,
                                name="ecology-idle-sleep", daemon=True)
                            try:
                                self._thread.start()
                            except Exception:
                                self._unavailable = True
                                self._thread = None
                                self._stop_process_locked()
                                logging.getLogger(__name__).warning("Idle-sleep monitor unavailable; continuing evaluation")
            yield
        finally:
            with self._lock:
                self._leases.pop(token, None)
                if not self._leases:
                    self._stop_process_locked()

    def _ensure_process_locked(self):
        if self._unavailable or self._stop.is_set():
            return
        try:
            if self._process is not None and self._process.poll() is None:
                return
            self._process = self._popen(
                ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True,
            )
        except Exception:
            self._process = None
            self._unavailable = True
            logging.getLogger(__name__).warning("Idle-sleep protection unavailable; continuing evaluation")

    def _stop_process_locked(self):
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=0.5)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=0.5)
            except Exception:
                logging.getLogger(__name__).warning("Idle-sleep process cleanup did not finish")

    def refresh(self):
        # Read the ledger outside our lock so a pause/shutdown never inverts
        # the evaluation's lock order. Tokens fence late callbacks and resumes.
        with self._lock:
            current = tuple(self._leases.items())
        expired = [token for token, predicate in current if not self._active(predicate)]
        with self._lock:
            for token in expired:
                self._leases.pop(token, None)
            if self._leases:
                self._ensure_process_locked()
            else:
                self._stop_process_locked()

    def _watch(self):
        while not self._stop.wait(self._interval):
            self.refresh()

    def close(self):
        self._stop.set()
        with self._lock:
            self._leases.clear()
            self._stop_process_locked()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        atexit.unregister(self.close)


def protect_generation(function):
    """Cover synchronous CLI/HTTP generations and adaptive worker turns."""
    @wraps(function)
    def protected(services, run_id, *args, **kwargs):
        guard = getattr(services, "idle_sleep", None)
        if guard is None:
            return function(services, run_id, *args, **kwargs)
        with guard.hold(lambda: services.ledger.latest_run_lifecycle_kind(run_id)
                        in {"RunStarted", "RunResumed"}):
            return function(services, run_id, *args, **kwargs)
    return protected


def protect_independent_evaluation(function):
    """The parent training run may already be completed during validation."""
    @wraps(function)
    def protected(service, *args, **kwargs):
        guard = getattr(service.server, "idle_sleep", None)
        if guard is None:
            return function(service, *args, **kwargs)
        with guard.hold(lambda: not service.stopping.is_set()):
            return function(service, *args, **kwargs)
    return protected
