from __future__ import annotations

import threading
import time
import unittest

from ecologyrsi_dsh.execution.sample_admission import RunSampleAdmission
from ecologyrsi_dsh.evaluators.sample_execution import _strict_origin_worker_count


class SampleConcurrencyGovernorTests(unittest.TestCase):
    def test_two_finalist_lanes_share_the_full_run_limit(self) -> None:
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
            if snapshot["active"] == 64 and snapshot["waiting"] == 36:
                break
            time.sleep(0.005)

        snapshot = admission.snapshot("run:two-finalists")
        self.assertEqual(snapshot["limit"], 64)
        self.assertEqual(snapshot["active"], 64)
        self.assertEqual(snapshot["waiting"], 36)

        release.set()
        for thread in threads:
            thread.join(3)
            self.assertFalse(thread.is_alive())

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
