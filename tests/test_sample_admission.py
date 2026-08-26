from __future__ import annotations

import threading
import time
import unittest

from ecologyrsi_dsh.api.sample_admission import RunSampleAdmission
from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError


class RunSampleAdmissionTests(unittest.TestCase):
    def _exercise_limit(
        self,
        *,
        limit: int,
        worker_groups: tuple[int, ...],
        error_callers: frozenset[int] = frozenset(),
    ) -> tuple[RunSampleAdmission, int, list[type[BaseException]]]:
        admission = RunSampleAdmission()
        release = threading.Event()
        start = threading.Barrier(sum(worker_groups) + 1)
        active_lock = threading.Lock()
        active = 0
        maximum_active = 0
        errors: list[type[BaseException]] = []
        threads: list[threading.Thread] = []

        def run(caller: int) -> None:
            nonlocal active, maximum_active
            start.wait()
            try:
                with admission.admit("run:test", limit):
                    with active_lock:
                        active += 1
                        maximum_active = max(maximum_active, active)
                    try:
                        release.wait(3)
                        if caller in error_callers:
                            raise RuntimeError("synthetic admitted-body cancellation")
                    finally:
                        with active_lock:
                            active -= 1
            except RuntimeError as exc:
                errors.append(type(exc))

        caller = 0
        for group, count in enumerate(worker_groups):
            for _ in range(count):
                caller += 1
                thread = threading.Thread(
                    target=run,
                    args=(caller,),
                    name=f"candidate-{group}-origin-{caller}",
                )
                thread.start()
                threads.append(thread)

        start.wait()
        deadline = time.monotonic() + 3
        expected_active = min(limit, 8)
        expected_waiting = sum(worker_groups) - expected_active
        while time.monotonic() < deadline:
            snapshot = admission.snapshot("run:test")
            if (
                snapshot["active"] == expected_active
                and snapshot["waiting"] == expected_waiting
            ):
                break
            time.sleep(0.005)

        snapshot = admission.snapshot("run:test")
        self.assertEqual(snapshot["active"], expected_active)
        self.assertEqual(snapshot["waiting"], expected_waiting)
        self.assertEqual(maximum_active, expected_active)

        release.set()
        for thread in threads:
            thread.join(3)
            self.assertFalse(thread.is_alive())
        final_snapshot = admission.snapshot("run:test")
        self.assertEqual(final_snapshot["limit"], limit)
        self.assertEqual(final_snapshot["active"], 0)
        self.assertEqual(final_snapshot["waiting"], 0)
        return admission, maximum_active, errors

    def test_exact_limit_eight_across_three_worker_groups(self) -> None:
        admission, maximum_active, errors = self._exercise_limit(
            limit=8,
            worker_groups=(3, 3, 3),
            error_callers=frozenset({1, 9}),
        )

        self.assertEqual(maximum_active, 8)
        self.assertEqual(errors, [RuntimeError, RuntimeError])
        self.assertEqual(
            admission.snapshot("run:test"),
            {
                "limit": 8,
                "adaptive_limit": 8,
                "active": 0,
                "waiting": 0,
                "congestion_events": 0,
            },
        )

    def test_exact_limit_sixty_four_with_sixty_five_callers(self) -> None:
        _admission, maximum_active, errors = self._exercise_limit(
            limit=64,
            worker_groups=(65,),
        )

        self.assertEqual(maximum_active, 8)
        self.assertEqual(errors, [])

    def test_configured_sixty_four_uses_additive_ramp_without_a_burst(self) -> None:
        admission = RunSampleAdmission()

        with admission.admit("run:adaptive", 64):
            snapshot = admission.snapshot("run:adaptive")

        self.assertEqual(snapshot["limit"], 64)
        self.assertEqual(snapshot["adaptive_limit"], 8)

        for _ in range(7):
            with admission.admit("run:adaptive", 64):
                pass
        self.assertEqual(
            admission.snapshot("run:adaptive")["adaptive_limit"],
            9,
        )

    def test_replayed_successes_cannot_exponentially_jump_to_configured_limit(
        self,
    ) -> None:
        admission = RunSampleAdmission()

        for _ in range(204):
            with admission.admit("run:replayed", 64):
                pass

        self.assertEqual(
            admission.snapshot("run:replayed")["adaptive_limit"],
            22,
        )

    def test_retryable_provider_failure_reduces_adaptive_limit_once(self) -> None:
        admission = RunSampleAdmission()
        for _ in range(8):
            with admission.admit("run:congested", 64):
                pass
        self.assertEqual(
            admission.snapshot("run:congested")["adaptive_limit"],
            9,
        )

        with self.assertRaises(DshNativeRuntimeUnavailableError):
            with admission.admit("run:congested", 64):
                raise DshNativeRuntimeUnavailableError(
                    "provider unavailable",
                    error_code="dsh_native_runtime_http_error",
                    status_code=502,
                )

        snapshot = admission.snapshot("run:congested")
        self.assertEqual(snapshot["limit"], 64)
        self.assertEqual(snapshot["adaptive_limit"], 4)
        self.assertEqual(snapshot["congestion_events"], 1)

    def test_overlapping_failure_reduces_after_success_grows_window(self) -> None:
        admission = RunSampleAdmission()
        for _ in range(sum(range(8, 16))):
            with admission.admit("run:mixed", 64):
                pass

        for _ in range(15):
            with admission.admit("run:mixed", 64):
                pass

        successful = admission.admit("run:mixed", 64)
        failing = admission.admit("run:mixed", 64)
        successful.__enter__()
        failing.__enter__()
        successful.__exit__(None, None, None)
        self.assertEqual(
            admission.snapshot("run:mixed")["adaptive_limit"],
            17,
        )

        error = DshNativeRuntimeUnavailableError(
            "provider unavailable",
            error_code="dsh_native_runtime_http_error",
            status_code=502,
        )
        self.assertFalse(
            failing.__exit__(type(error), error, error.__traceback__)
        )

        snapshot = admission.snapshot("run:mixed")
        self.assertEqual(snapshot["adaptive_limit"], 8)
        self.assertEqual(snapshot["congestion_events"], 1)

    def test_exact_non_divisible_limit_three_with_eight_callers(self) -> None:
        _admission, maximum_active, errors = self._exercise_limit(
            limit=3,
            worker_groups=(3, 3, 2),
        )

        self.assertEqual(maximum_active, 3)
        self.assertEqual(errors, [])

    def test_run_limit_is_frozen_after_first_admission(self) -> None:
        admission = RunSampleAdmission()

        with admission.admit("run:frozen", 8):
            with self.assertRaisesRegex(ValueError, "frozen admission limit"):
                with admission.admit("run:frozen", 3):
                    self.fail("mismatched run limit must not be admitted")

        self.assertEqual(
            admission.snapshot("run:frozen"),
            {
                "limit": 8,
                "adaptive_limit": 8,
                "active": 0,
                "waiting": 0,
                "congestion_events": 0,
            },
        )
        self.assertEqual(
            admission.snapshot("run:unseen"),
            {
                "limit": 0,
                "adaptive_limit": 0,
                "active": 0,
                "waiting": 0,
                "congestion_events": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
