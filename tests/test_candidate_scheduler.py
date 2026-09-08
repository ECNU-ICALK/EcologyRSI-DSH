from __future__ import annotations

import threading
import unittest

from ecologyrsi_dsh.application.candidate_scheduler import (
    CandidateEvaluationTask,
    run_candidate_evaluations,
)


class CandidateSchedulerTests(unittest.TestCase):
    def test_candidate_ids_and_slots_must_both_be_unique(self) -> None:
        cases = (
            (
                (
                    CandidateEvaluationTask(0, "candidate-0"),
                    CandidateEvaluationTask(1, "candidate-0"),
                ),
                "candidate ids",
            ),
            (
                (
                    CandidateEvaluationTask(0, "candidate-0"),
                    CandidateEvaluationTask(0, "candidate-1"),
                ),
                "slot indices",
            ),
        )
        for tasks, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                run_candidate_evaluations(
                    tasks,
                    max_concurrency=2,
                    evaluate=lambda _candidate_id: None,
                    admission_open=lambda: True,
                )

    def test_four_candidates_overlap_in_one_bounded_batch(self) -> None:
        barrier = threading.Barrier(4, timeout=3)
        lock = threading.Lock()
        active = 0
        maximum_active = 0
        evaluated: list[str] = []

        def evaluate(candidate_id: str) -> None:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
                evaluated.append(candidate_id)
            barrier.wait()
            with lock:
                active -= 1

        tasks = tuple(
            CandidateEvaluationTask(slot_index=index, candidate_id=f"candidate-{index}")
            for index in range(4)
        )

        run_candidate_evaluations(
            tasks,
            max_concurrency=4,
            evaluate=evaluate,
            admission_open=lambda: True,
        )

        self.assertEqual(maximum_active, 4)
        self.assertEqual(set(evaluated), {f"candidate-{index}" for index in range(4)})

    def test_first_error_stops_refill_and_drains_admitted_sibling(self) -> None:
        second_started = threading.Event()
        first_raised = threading.Event()
        release_second = threading.Event()
        started: list[str] = []
        failures: list[BaseException] = []

        def evaluate(candidate_id: str) -> None:
            started.append(candidate_id)
            if candidate_id == "candidate-0":
                self.assertTrue(second_started.wait(3))
                first_raised.set()
                raise RuntimeError("candidate-0 retryable failure")
            if candidate_id == "candidate-1":
                second_started.set()
                self.assertTrue(release_second.wait(3))
                return
            self.fail("queued candidate started after the scheduler observed an error")

        tasks = tuple(
            CandidateEvaluationTask(slot_index=index, candidate_id=f"candidate-{index}")
            for index in range(3)
        )

        def run() -> None:
            try:
                run_candidate_evaluations(
                    tasks,
                    max_concurrency=2,
                    evaluate=evaluate,
                    admission_open=lambda: True,
                )
            except BaseException as exc:  # capture the scheduler result in this thread
                failures.append(exc)

        scheduler = threading.Thread(target=run)
        scheduler.start()
        self.assertTrue(first_raised.wait(3))
        self.assertTrue(scheduler.is_alive(), "scheduler returned before its admitted sibling drained")
        self.assertNotIn("candidate-2", started)
        release_second.set()
        scheduler.join(3)

        self.assertFalse(scheduler.is_alive())
        self.assertEqual([str(item) for item in failures], ["candidate-0 retryable failure"])
        self.assertEqual(set(started), {"candidate-0", "candidate-1"})

    def test_closed_admission_stops_queued_candidate(self) -> None:
        admission = {"open": True}
        peer_started = threading.Event()
        close_admission = threading.Event()
        started: list[str] = []

        def evaluate(candidate_id: str) -> None:
            started.append(candidate_id)
            if candidate_id == "candidate-0":
                self.assertTrue(peer_started.wait(3))
                admission["open"] = False
                close_admission.set()
                return
            if candidate_id == "candidate-1":
                peer_started.set()
                self.assertTrue(close_admission.wait(3))
                return
            self.fail("candidate started after admission closed")

        tasks = tuple(
            CandidateEvaluationTask(slot_index=index, candidate_id=f"candidate-{index}")
            for index in range(3)
        )

        run_candidate_evaluations(
            tasks,
            max_concurrency=2,
            evaluate=evaluate,
            admission_open=lambda: admission["open"],
        )

        self.assertEqual(set(started), {"candidate-0", "candidate-1"})

    def test_legacy_serial_mode_uses_the_calling_thread(self) -> None:
        calling_thread = threading.get_ident()
        worker_threads: list[int] = []
        tasks = (
            CandidateEvaluationTask(slot_index=0, candidate_id="candidate-0"),
            CandidateEvaluationTask(slot_index=1, candidate_id="candidate-1"),
        )

        run_candidate_evaluations(
            tasks,
            max_concurrency=1,
            evaluate=lambda _candidate_id: worker_threads.append(threading.get_ident()),
            admission_open=lambda: True,
        )

        self.assertEqual(worker_threads, [calling_thread, calling_thread])


if __name__ == "__main__":
    unittest.main()
