from __future__ import annotations

import threading
import time
import unittest

from ecologyrsi_dsh.api.sample_admission import RunSampleAdmission


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
        expected_waiting = sum(worker_groups) - limit
        while time.monotonic() < deadline:
            snapshot = admission.snapshot("run:test")
            if snapshot["active"] == limit and snapshot["waiting"] == expected_waiting:
                break
            time.sleep(0.005)

        snapshot = admission.snapshot("run:test")
        self.assertEqual(snapshot["active"], limit)
        self.assertEqual(snapshot["waiting"], expected_waiting)
        self.assertEqual(maximum_active, limit)

        release.set()
        for thread in threads:
            thread.join(3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(
            admission.snapshot("run:test"),
            {"limit": limit, "active": 0, "waiting": 0},
        )
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
            {"limit": 8, "active": 0, "waiting": 0},
        )

    def test_exact_limit_sixty_four_with_sixty_five_callers(self) -> None:
        _admission, maximum_active, errors = self._exercise_limit(
            limit=64,
            worker_groups=(65,),
        )

        self.assertEqual(maximum_active, 64)
        self.assertEqual(errors, [])

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
            {"limit": 8, "active": 0, "waiting": 0},
        )
        self.assertEqual(
            admission.snapshot("run:unseen"),
            {"limit": 0, "active": 0, "waiting": 0},
        )


if __name__ == "__main__":
    unittest.main()
