"""Idle-sleep assertions follow active work, without changing evaluation results."""
import os
import subprocess
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ecologyrsi_dsh.execution.idle_sleep import IdleSleepInhibitor


class FakeProcess:
    def __init__(self, *, ignores_terminate=False):
        self.returncode = None
        self.ignores_terminate = ignores_terminate
        self.stopped = threading.Event()
        self.terminations = 0
        self.kills = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminations += 1
        if not self.ignores_terminate:
            self.returncode = 0
            self.stopped.set()

    def kill(self):
        self.kills += 1
        self.returncode = -9
        self.stopped.set()

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake-caffeinate", timeout)
        return self.returncode


class IdleSleepTests(unittest.TestCase):
    def guard(self, **kwargs):
        children = []
        def spawn(*_args, **_kwargs):
            child = FakeProcess(**kwargs)
            children.append(child)
            return child
        factory = Mock(side_effect=spawn)
        guard = IdleSleepInhibitor(enabled=True, platform="darwin", popen=factory, interval=.01)
        self.addCleanup(guard.close)
        return guard, children, factory

    def test_parallel_work_shares_assertion_until_last_owner_finishes(self):
        guard, children, factory = self.guard()
        entered = threading.Barrier(3)
        release = [threading.Event(), threading.Event()]
        def work(index):
            with guard.hold():
                entered.wait(timeout=2)
                release[index].wait(timeout=2)
        workers = [threading.Thread(target=work, args=(index,)) for index in range(2)]
        for worker in workers:
            worker.start()
        try:
            entered.wait(timeout=2)
            self.assertEqual(len(children), 1)
            self.assertEqual(factory.call_args.args[0],
                             ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
            self.assertTrue(factory.call_args.kwargs["start_new_session"])
            release[0].set()
            workers[0].join(timeout=2)
            self.assertIsNone(children[0].poll())
        finally:
            for event in release:
                event.set()
            for worker in workers:
                worker.join(timeout=2)
        self.assertEqual(children[0].terminations, 1)
        self.assertIsNotNone(children[0].poll())

    def test_pause_releases_while_call_waits_and_late_release_cannot_stop_resume(self):
        guard, children, _factory = self.guard()
        active = [True]
        old = guard.hold(lambda: active[0])
        old.__enter__()
        try:
            active[0] = False
            self.assertTrue(children[0].stopped.wait(timeout=2))
            active[0] = True
            with guard.hold(lambda: active[0]):
                self.assertEqual(len(children), 2)
                old.__exit__(None, None, None)
                self.assertIsNone(children[1].poll())
            self.assertEqual(children[1].terminations, 1)
        finally:
            old.__exit__(None, None, None)

    def test_disabled_platform_and_unavailable_binary_leave_work_unchanged(self):
        factory = Mock(side_effect=OSError("missing optional executable"))
        for platform, enabled in (("linux", True), ("win32", True), ("darwin", False)):
            guard = IdleSleepInhibitor(platform=platform, enabled=enabled, popen=factory)
            with guard.hold():
                pass
            guard.close()
        with patch.dict(os.environ, {"ECOLOGYRSI_PREVENT_IDLE_SLEEP": "0"}):
            guard = IdleSleepInhibitor(platform="darwin", popen=factory)
            with guard.hold():
                pass
            guard.close()
        factory.assert_not_called()
        guard = IdleSleepInhibitor(platform="darwin", enabled=True, popen=factory)
        self.addCleanup(guard.close)
        sentinel = RuntimeError("original evaluation failure")
        with self.assertLogs("ecologyrsi_dsh.execution.idle_sleep", level="WARNING"):
            with self.assertRaises(RuntimeError) as caught:
                with guard.hold():
                    raise sentinel
        self.assertIs(caught.exception, sentinel)
        self.assertEqual(factory.call_count, 1)

    def test_shutdown_kills_unresponsive_child_without_masking_result(self):
        guard, children, factory = self.guard(ignores_terminate=True)
        with guard.hold():
            guard.close()
            self.assertEqual(children[0].kills, 1)
            with guard.hold():
                pass
        self.assertEqual(factory.call_count, 1)
        self.assertFalse(guard._thread.is_alive())

    def test_generation_and_adaptive_public_entrypoints_hold_only_active_work(self):
        from ecologyrsi_dsh.application import generation_execution as execution
        from ecologyrsi_dsh.application.work_units import execute_next_adaptive_work_unit
        from ecologyrsi_dsh.core.models import RunStatus
        guard, children, _factory = self.guard()
        state = SimpleNamespace(run=SimpleNamespace(status=RunStatus.RUNNING, generation=0),
                                batch_for=lambda _generation: None)
        services = SimpleNamespace(idle_sleep=guard,
            ledger=SimpleNamespace(latest_run_lifecycle_kind=Mock(return_value="RunStarted")),
            director=SimpleNamespace(state=Mock(return_value=state)))
        def spawn(*_args):
            self.assertIsNone(children[-1].poll())
            return ()
        with patch.object(execution, "complete_if_budget_exhausted", return_value=state), \
                patch.object(execution, "start_generation_batch", return_value=object()), \
                patch.object(execution, "_spawn_generation_candidates", side_effect=spawn), \
                patch("ecologyrsi_dsh.evolution.batches.start_generation_batch", return_value=object()):
            self.assertIs(execution.execute_generation(services, "run:test"), state)
            self.assertFalse(execute_next_adaptive_work_unit(services, "run:test"))
            self.assertEqual(len(children), 2)
            self.assertTrue(all(child.stopped.is_set() for child in children))
            state.run.status = RunStatus.PAUSED
            services.ledger.latest_run_lifecycle_kind.return_value = "RunPaused"
            execution.execute_generation(services, "run:test")
            execute_next_adaptive_work_unit(services, "run:test")
            self.assertEqual(len(children), 2)

    def test_independent_worker_holds_despite_completed_training_and_releases_on_error(self):
        from tests.test_independent_evaluation import IndependentJobTests
        for fail in (False, True):
            with self.subTest(fail=fail):
                fixture = IndependentJobTests()
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                guard, children, _factory = self.guard()
                fixture.server.idle_sleep = guard
                def evaluate(*_args):
                    self.assertIsNone(children[0].poll())
                    if fail:
                        raise RuntimeError("synthetic worker failure")
                    return fixture.server.dsh_native_runtime.execute_sample()
                with patch.object(fixture.service, "_evaluate", side_effect=evaluate), \
                        patch("ecologyrsi_dsh.application.independent_evaluation.logging"):
                    fixture.service.start(fixture.fixture.run_id, "validation")
                    fixture.service.workers[0].join(timeout=3)
                self.assertEqual(len(children), 1)
                self.assertTrue(children[0].stopped.is_set())
                self.assertFalse(fixture.service.workers[0].is_alive())

    def test_http_shutdown_releases_assertion_before_waiting_for_workers(self):
        from ecologyrsi_dsh.api.handler import EvolutionHTTPServer
        guard, children, _factory = self.guard()
        def worker_close():
            self.assertTrue(children[0].stopped.is_set())
            return False
        server = SimpleNamespace(idle_sleep=guard, host_activity=Mock(),
            independent_evaluation=SimpleNamespace(close=worker_close),
            auto_progress=SimpleNamespace(close=worker_close), server_close=Mock())
        with guard.hold():
            EvolutionHTTPServer.close(server)
        server.server_close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
