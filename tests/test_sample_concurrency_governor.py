from __future__ import annotations

import threading
import time
import unittest

from ecologyrsi_dsh.execution.sample_admission import RunSampleAdmission
from ecologyrsi_dsh.evaluators.sample_execution import _strict_origin_worker_count


class SampleConcurrencyGovernorTests(unittest.TestCase):
    def test_two_finalist_lanes_share_one_cold_start_window(self) -> None:
        admission = RunSampleAdmission()
        release = threading.Event()
        start = threading.Barrier(101)
        threads: list[threading.Thread] = []

        def origin() -> None:
            start.wait()
            with admission.admit("run:two-finalists", 64):
                release.wait(3)

        for lane in range(2):
            for index in range(50):
                thread = threading.Thread(
                    target=origin,
                    name=f"finalist-{lane}-origin-{index}",
                )
                thread.start()
                threads.append(thread)
        start.wait()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            snapshot = admission.snapshot("run:two-finalists")
            if snapshot["active"] == 8 and snapshot["waiting"] == 92:
                break
            time.sleep(0.005)

        snapshot = admission.snapshot("run:two-finalists")
        self.assertEqual(snapshot["limit"], 64)
        self.assertEqual(snapshot["active"], 8)
        self.assertEqual(snapshot["waiting"], 92)

        release.set()
        for thread in threads:
            thread.join(3)
            self.assertFalse(thread.is_alive())

    def test_returned_failed_outcomes_do_not_raise_admission_capacity(self):
        admission = RunSampleAdmission()
        for _ in range(50):
            with admission.admit("run:unhealthy", 64) as outcome:
                outcome.healthy = False
        self.assertEqual(admission.snapshot("run:unhealthy")["adaptive_limit"], 8)
        for _ in range(8):
            with admission.admit("run:unhealthy", 64):
                pass
        self.assertEqual(admission.snapshot("run:unhealthy")["adaptive_limit"], 9)

    def test_candidate_concurrency_does_not_divide_origin_workers(self) -> None:
        self.assertEqual(
            _strict_origin_worker_count(
                pending_origin_count=50,
                sample_concurrency=64,
                candidate_concurrency=4,
            ),
            50,
        )
        self.assertEqual(
            _strict_origin_worker_count(
                pending_origin_count=100,
                sample_concurrency=64,
                candidate_concurrency=2,
            ),
            64,
        )


if __name__ == "__main__":
    unittest.main()
