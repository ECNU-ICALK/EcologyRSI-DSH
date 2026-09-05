from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from ecologyrsi_dsh import (
    CandidateStatus,
    Evaluation,
    EventLedger,
    EvolutionDirector,
    FakeDSHAdapter,
    PromotionDecision,
    RunStatus,
    TaskManifest,
    ToyCropSoilWater,
)
from ecologyrsi_dsh.core.models import digest


def manifest(max_candidates: int = 3) -> TaskManifest:
    return TaskManifest(
        task_id="toy-forecast",
        objective="predict soil water",
        domain_pack="crop-soil-water@toy",
        visible_datasets=("toy-dataset@1",),
        budget={"max_candidates": max_candidates},
        seed=11,
    )


class CoreTests(unittest.TestCase):
    def test_task_manifest_digest_is_stable(self) -> None:
        first = manifest()
        second = TaskManifest.from_dict(first.to_dict())
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.max_candidates, 3)
        self.assertEqual(first.domain, "crop-soil-water@toy")

    def test_task_manifest_expands_conflicting_candidate_budget(self) -> None:
        task = TaskManifest(
            task_id="complete-generation-budget",
            objective="reserve every candidate slot in every generation",
            domain_pack="crop-soil-water@toy",
            budget={
                "max_generations": 10,
                "candidates_per_generation": 3,
                "max_candidates": 3,
            },
        )

        self.assertEqual(task.max_generations, 10)
        self.assertEqual(task.candidates_per_generation, 3)
        self.assertEqual(task.max_candidates, 30)
        self.assertEqual(task.budget["max_candidates"], 30)

    def test_ledger_duplicate_event_id_is_idempotent(self) -> None:
        ledger = EventLedger()
        first = ledger.append("run:1", "Example", {"value": 1}, event_id="event:1")
        second = ledger.append("run:1", "Example", {"value": 1}, event_id="event:1")
        self.assertEqual(first, second)
        self.assertEqual(ledger.count("run:1"), 1)
        self.assertEqual(ledger.events("run:1")[0].payload, {"value": 1})
        with self.assertRaisesRegex(ValueError, "different event"):
            ledger.append("run:1", "Example", {"value": 2}, event_id="event:1")
        ledger.close()

    def test_latest_run_seq_uses_the_run_tail_without_loading_events(self) -> None:
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        first = ledger.append("run:a", "Example", {"value": 1})
        ledger.append("run:b", "Example", {"value": 2})
        latest_a = ledger.append("run:a", "Example", {"value": 3})
        latest_global = ledger.append("run:b", "Example", {"value": 4})

        with patch.object(
            ledger,
            "events",
            side_effect=AssertionError("run stream must not be loaded"),
        ):
            self.assertEqual(ledger.latest_run_seq("run:a"), latest_a.seq)
            self.assertEqual(ledger.latest_run_seq("run:b"), latest_global.seq)
            self.assertEqual(ledger.latest_run_seq("run:missing"), 0)

        self.assertLess(first.seq, latest_a.seq)

    def test_lifecycle_tail_is_covered_and_ignores_non_lifecycle_history(self) -> None:
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        ledger.append("run:lifecycle", "RunCreated", {})
        for index in range(200):
            ledger.append("run:lifecycle", "SampleCheckpointed", {"index": index})
        ledger.append("run:lifecycle", "RunStarted", {"session_id": "session:1"})
        for index in range(200, 400):
            ledger.append("run:lifecycle", "ModelUsageRecorded", {"index": index})

        with patch.object(
            ledger,
            "events",
            side_effect=AssertionError("run stream must not be loaded"),
        ):
            self.assertEqual(
                ledger.latest_run_lifecycle_kind("run:lifecycle"),
                "RunStarted",
            )
            self.assertIsNone(ledger.latest_run_lifecycle_kind("run:missing"))

        plan = ledger._connection.execute(  # noqa: SLF001 - verify hot-path index
            """
            EXPLAIN QUERY PLAN
            SELECT kind FROM evolution_events
            WHERE run_id = ? AND kind IN (
                'RunCreated', 'RunStarted', 'RunPaused', 'RunResumed',
                'RunCancelled', 'RunFailed', 'RunCompleted'
            )
            ORDER BY seq DESC LIMIT 1
            """,
            ("run:lifecycle",),
        ).fetchall()
        self.assertIn(
            "USING COVERING INDEX idx_evolution_events_run_lifecycle_seq",
            " ".join(str(row["detail"]) for row in plan),
        )

    def test_director_run_status_matches_lifecycle_without_replay(self) -> None:
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger)
        run_id = "run:indexed-status"
        director.create_run(manifest(), run_id=run_id)
        self.assertIs(director.run_status(run_id), RunStatus.CREATED)
        director.start_run(run_id)
        self.assertIs(director.run_status(run_id), RunStatus.RUNNING)
        director.pause_run(run_id)
        self.assertIs(director.run_status(run_id), RunStatus.PAUSED)
        director.resume_run(run_id)
        self.assertIs(director.run_status(run_id), RunStatus.RUNNING)
        director.cancel_run(run_id)

        with patch.object(
            ledger,
            "events",
            side_effect=AssertionError("status lookup must not replay events"),
        ):
            self.assertIs(director.run_status(run_id), RunStatus.CANCELLED)
            with self.assertRaisesRegex(KeyError, "unknown run"):
                director.run_status("run:missing")

    def test_state_replay_is_cached_until_run_tail_changes(self) -> None:
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger)
        run_id = "run:state-cache"
        director.create_run(manifest(1), run_id=run_id)

        with patch(
            "ecologyrsi_dsh.core.director.project_run_state",
            wraps=__import__(
                "ecologyrsi_dsh.core.state", fromlist=["project_run_state"]
            ).project_run_state,
        ) as replay:
            first = director.state(run_id)
            second = director.state(run_id)

        self.assertIs(first, second)
        self.assertEqual(replay.call_count, 1)

        with patch(
            "ecologyrsi_dsh.core.director.project_run_state",
            wraps=__import__(
                "ecologyrsi_dsh.core.state", fromlist=["project_run_state"]
            ).project_run_state,
        ) as replay_after_append:
            director.start_run(run_id)
            updated = director.state(run_id)

        self.assertIsNot(updated, first)
        self.assertEqual(replay_after_append.call_count, 1)
        self.assertIs(updated.run.status, RunStatus.RUNNING)

    def test_candidate_identity_source_has_one_durable_spawn_event(self) -> None:
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=1))
        director.start_evolution(manifest(1), run_id="run:spawn-identity")
        proposal = director.request_proposal("run:spawn-identity")
        candidate = director.spawn_candidate(
            "run:spawn-identity",
            proposal,
            candidate_id="candidate:stable",
        )
        spawned = next(
            event
            for event in ledger.events("run:spawn-identity")
            if event.kind == "CandidateSpawned"
        )
        self.assertEqual(
            spawned.event_id,
            "candidate-spawned:"
            + digest(
                {
                    "run_id": "run:spawn-identity",
                    "candidate_id": candidate.candidate_id,
                }
            ),
        )

        ledger.append(
            "run:spawn-identity",
            "CandidateSpawned",
            spawned.payload,
            event_id="injected-second-spawn",
        )
        with self.assertRaisesRegex(ValueError, "multiple spawn events"):
            director.state("run:spawn-identity")

    def test_ledger_compares_idempotency_after_json_normalization(self) -> None:
        ledger = EventLedger()
        first = ledger.append(
            "run:json-normalization",
            "GenerationAnalyzed",
            {"ranking": ({"candidate_id": "candidate-1"},)},
            event_id="event:json-normalization",
        )
        second = ledger.append(
            "run:json-normalization",
            "GenerationAnalyzed",
            {"ranking": [{"candidate_id": "candidate-1"}]},
            event_id="event:json-normalization",
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first.payload,
            {"ranking": [{"candidate_id": "candidate-1"}]},
        )
        ledger.close()

    def test_ledger_idempotency_preserves_json_scalar_types(self) -> None:
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        conflicting_values = (
            ("bool-versus-int", True, 1),
            ("int-versus-float", 2, 2.0),
            ("float-versus-bool", 0.0, False),
        )

        for label, recorded, retry in conflicting_values:
            with self.subTest(label=label):
                event_id = f"event:json-type:{label}"
                ledger.append(
                    "run:json-types",
                    "Example",
                    {"nested": {"value": recorded}},
                    event_id=event_id,
                )

                with self.assertRaisesRegex(ValueError, "different event"):
                    ledger.append(
                        "run:json-types",
                        "Example",
                        {"nested": {"value": retry}},
                        event_id=event_id,
                    )

        first = ledger.append(
            "run:json-types",
            "Example",
            {"outer": {"alpha": 1, "beta": 2}, "stable": True},
            event_id="event:json-type:key-order",
        )
        reordered = ledger.append(
            "run:json-types",
            "Example",
            {"stable": True, "outer": {"beta": 2, "alpha": 1}},
            event_id="event:json-type:key-order",
        )

        self.assertEqual(reordered, first)

        ledger.append_many(
            "run:json-types",
            (("Example", {"nested": {"value": 3}}, "event:batch-json-type"),),
        )
        with self.assertRaisesRegex(ValueError, "different event"):
            ledger.append_many(
                "run:json-types",
                (
                    (
                        "Example",
                        {"nested": {"value": 3.0}},
                        "event:batch-json-type",
                    ),
                ),
            )

    def test_legacy_run_has_empty_structured_screening_state(self) -> None:
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=1))
        state = director.start_evolution(manifest(1), run_id="run:legacy-screening")

        self.assertEqual(state.candidate_screening_events, ())
        self.assertEqual(state.formal_selection_events, ())
        self.assertEqual(state.screened_out_events, ())
        self.assertIsNone(state.screening_for(0, "candidate:missing"))
        self.assertIsNone(state.formal_selection_for(0))

    def test_full_loop_replays_from_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "events.sqlite3"
            ledger = EventLedger(db)
            director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=2))
            state = director.start_evolution(manifest(2), run_id="run:replay")
            self.assertIs(state.run.status, RunStatus.RUNNING)

            toy = ToyCropSoilWater(seed=11)
            for _ in range(2):
                candidate = director.propose_and_spawn("run:replay")
                proposal = director.state("run:replay").proposal(candidate.proposal_id)
                evaluation = toy.evaluate_candidate("run:replay", candidate, proposal)
                director.evaluate_and_decide(evaluation)
            director.complete_run("run:replay")
            expected = director.state("run:replay")
            ledger.close()

            replay_ledger = EventLedger(db)
            replayed = EvolutionDirector(replay_ledger, FakeDSHAdapter()).replay("run:replay")
            self.assertEqual(replayed.run, expected.run)
            self.assertEqual(replayed.task_manifest.digest, expected.task_manifest.digest)
            self.assertEqual(len(replayed.proposals), 2)
            self.assertEqual(len(replayed.evaluations), 2)
            self.assertTrue(all(item.status is CandidateStatus.PROMOTED for item in replayed.candidates))
            self.assertTrue(
                all(item.decision in (PromotionDecision.APPROVED, PromotionDecision.REJECTED) for item in replayed.promotions)
            )
            self.assertEqual(len(replayed.events), replay_ledger.count("run:replay"))
            replay_ledger.close()

    def test_time_forward_toy_splits_and_repeatability(self) -> None:
        first = ToyCropSoilWater(seed=4, days=30)
        second = ToyCropSoilWater(seed=4, days=30)
        self.assertEqual(first.observations, second.observations)
        self.assertEqual(tuple(first.splits), ("train", "validation", "test"))
        self.assertEqual(tuple(len(items) for items in first.splits.values()), (9, 8, 5))
        self.assertLess(first.splits["train"][-1].day, first.splits["validation"][0].day)
        self.assertLess(first.splits["validation"][-1].day, first.splits["test"][0].day)
        parameters = {"alpha": 0.35, "window": 5, "water_threshold": 0.4}
        self.assertEqual(first.score(parameters, "validation"), second.score(parameters, "validation"))

    def test_invalid_transition_and_budget_are_rejected(self) -> None:
        director = EvolutionDirector(EventLedger(), FakeDSHAdapter(max_proposals=2))
        director.create_run(manifest(1), run_id="run:errors")
        with self.assertRaises(RuntimeError):
            director.propose_and_spawn("run:errors")
        director.start_run("run:errors")
        first = director.propose_and_spawn("run:errors")
        with self.assertRaisesRegex(RuntimeError, "budget"):
            director.propose_and_spawn("run:errors")
        evaluation = Evaluation(
            evaluation_id="evaluation:one",
            run_id="run:errors",
            candidate_id=first.candidate_id,
            score=0.5,
            passed=True,
        )
        director.evaluate_and_decide(evaluation)
        with self.assertRaises(RuntimeError):
            director.start_run("run:errors")

    def test_cancelled_terminal_state_wins_a_stale_background_failure(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        director.start_evolution(manifest(1), run_id="run:cancel-race")
        failure_ready = threading.Event()
        release_failure = threading.Event()
        failure_errors: list[BaseException] = []
        original_append = ledger.append

        def delayed_append(*args, **kwargs):
            if len(args) > 1 and args[1] == "RunFailed":
                failure_ready.set()
                release_failure.wait(timeout=2)
            return original_append(*args, **kwargs)

        def fail_in_background() -> None:
            try:
                director.fail_run("run:cancel-race", "stale background failure")
            except BaseException as exc:  # noqa: BLE001 - assert thread result
                failure_errors.append(exc)

        with patch.object(ledger, "append", side_effect=delayed_append):
            worker = threading.Thread(target=fail_in_background)
            worker.start()
            self.assertTrue(failure_ready.wait(timeout=1))
            director.cancel_run("run:cancel-race")
            release_failure.set()
            worker.join(timeout=2)

        state = director.state("run:cancel-race")
        self.assertEqual(state.run.status, RunStatus.CANCELLED)
        self.assertNotIn("RunFailed", [event.kind for event in state.events])
        self.assertEqual(len(failure_errors), 1)
        self.assertIn("cancelled", str(failure_errors[0]))
        ledger.close()

    def test_replay_never_regresses_after_the_first_terminal_event(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        director.start_evolution(manifest(1), run_id="run:sticky-terminal")
        director.cancel_run("run:sticky-terminal")
        ledger.append("run:sticky-terminal", "RunFailed", {"reason": "late failure"})
        ledger.append(
            "run:sticky-terminal",
            "RunStarted",
            {"session_id": "late-session"},
        )
        ledger.append(
            "run:sticky-terminal",
            "GenerationAdvanced",
            {"generation": 1},
        )

        replayed = director.replay("run:sticky-terminal").run
        self.assertEqual(replayed.status, RunStatus.CANCELLED)
        self.assertEqual(replayed.generation, 0)
        ledger.close()


if __name__ == "__main__":
    unittest.main()
