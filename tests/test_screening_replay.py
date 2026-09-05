from __future__ import annotations

from dataclasses import replace
import importlib
import unittest

from ecologyrsi_dsh import EventLedger, EvolutionDirector, FakeDSHAdapter, TaskManifest
from ecologyrsi_dsh.core.ledger import Event
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.state import project_run_state


_SCREENING_FIELDS = (
    "generation",
    "candidate_id",
    "score",
    "passed",
    "constraint_violations",
    "origin_count",
    "prediction_cell_count",
    "cohort_digest",
)


def _record_digest(payload: dict[str, object]) -> str:
    return digest({name: payload[name] for name in _SCREENING_FIELDS})


class ScreeningReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = EventLedger()
        self.addCleanup(self.ledger.close)
        self.director = EvolutionDirector(self.ledger, FakeDSHAdapter(max_proposals=4))
        self.run_id = "run:screening-replay"
        task = TaskManifest(
            task_id="strict-v4-screening",
            objective="validate two-stage screening replay",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("toy-dataset@1",),
            budget={"max_candidates": 4, "candidates_per_generation": 4},
            seed=11,
            metadata={
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "sample_budget_class": "selection_eligible",
                "prediction_cells_per_origin": 3,
                "two_stage_evaluation_enabled": True,
            },
        )
        self.director.start_evolution(task, run_id=self.run_id)
        self.candidates = tuple(
            self.director.propose_and_spawn(self.run_id) for _ in range(4)
        )
        scores = (0.9, 0.8, 0.7, 0.6)
        passed = (False, True, True, True)
        screening_events: list[Event] = []
        for index, candidate in enumerate(self.candidates):
            payload: dict[str, object] = {
                "schema_version": "ecologyrsi-dsh.candidate-screening/2",
                "generation": 0,
                "candidate_id": candidate.candidate_id,
                "score": scores[index],
                "passed": passed[index],
                "constraint_violations": 0,
                "origin_count": 64,
                "prediction_cell_count": 192,
                "cohort_digest": digest(
                    {"phase": "screening", "candidate_id": candidate.candidate_id}
                ),
            }
            payload["record_digest"] = _record_digest(payload)
            screening_events.append(
                self.ledger.append(
                    self.run_id,
                    "CandidateScreeningRecorded",
                    payload,
                    event_id=(
                        f"{self.run_id}:generation:0:screening:{candidate.candidate_id}"
                    ),
                )
            )
        self.screening_events = tuple(screening_events)
        screening_digest = digest(
            [
                event.payload
                for event in sorted(
                    self.screening_events,
                    key=lambda item: str(item.payload["candidate_id"]),
                )
            ]
        )
        self.selected_ids = tuple(
            candidate.candidate_id for candidate in self.candidates[:2]
        )
        self.formal_event = self.ledger.append(
            self.run_id,
            "FormalSelectionCohortFrozen",
            {
                "schema_version": "ecologyrsi-dsh.formal-selection-cohort/2",
                "generation": 0,
                "selected_candidate_ids": list(self.selected_ids),
                "screening_digest": screening_digest,
            },
            event_id=f"{self.run_id}:generation:0:formal-selection",
        )
        for candidate in self.candidates[2:]:
            self.ledger.append(
                self.run_id,
                "CandidateScreenedOut",
                {
                    "schema_version": "ecologyrsi-dsh.candidate-screened-out/1",
                    "candidate_id": candidate.candidate_id,
                    "generation": 0,
                    "formal_selection_event_id": self.formal_event.event_id,
                    "reason": "not_selected_by_screening_top_k",
                },
                event_id=(
                    f"{self.run_id}:generation:0:screened-out:{candidate.candidate_id}"
                ),
            )
        self.events = self.ledger.events(self.run_id)

    def _replace_event(
        self,
        events: tuple[Event, ...],
        target: Event,
        payload: dict[str, object],
    ) -> tuple[Event, ...]:
        return tuple(
            replace(event, payload=dict(payload)) if event.event_id == target.event_id else event
            for event in events
        )

    def test_digest_helpers_use_canonical_screening_fields_and_candidate_order(self) -> None:
        try:
            screening = importlib.import_module("ecologyrsi_dsh.core.screening")
        except ModuleNotFoundError as exc:
            self.fail(f"screening contracts module is missing: {exc}")
        records = [
            {
                "generation": 0,
                "candidate_id": "candidate-b",
                "score": 0.5,
                "passed": False,
                "constraint_violations": 1,
                "origin_count": 64,
                "prediction_cell_count": 192,
                "cohort_digest": "b" * 64,
                "ignored": "not part of the record identity",
            },
            {
                "generation": 0,
                "candidate_id": "candidate-a",
                "score": 0.75,
                "passed": True,
                "constraint_violations": 0,
                "origin_count": 64,
                "prediction_cell_count": 192,
                "cohort_digest": "a" * 64,
            },
        ]

        self.assertEqual(
            screening.screening_record_digest(records[0]),
            "ff1343b8a13ccd46d168533631d9d3188cd03aa296fd68576b5fd2ed5f4a15fb",
        )
        self.assertEqual(
            screening.screening_cohort_digest(records),
            "9788671d8f6eb5dc2db6f1d97a72e99874be3835b4e2f890b24749a07ad2471f",
        )

    def test_new_writes_use_versioned_screening_contracts(self) -> None:
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=3))
        task = TaskManifest(
            task_id="strict-v4-writer",
            objective="write versioned screening contracts",
            domain_pack="crop-soil-water@toy",
            budget={"max_candidates": 3, "candidates_per_generation": 3},
            metadata={
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "sample_budget_class": "selection_eligible",
                "prediction_cells_per_origin": 3,
            },
        )
        director.start_evolution(task, run_id="run:screening-writer")
        candidates = tuple(
            director.propose_and_spawn("run:screening-writer") for _ in range(3)
        )
        written = tuple(
            director.record_candidate_screening(
                "run:screening-writer",
                candidate_id=candidate.candidate_id,
                generation=0,
                score=0.9 - index * 0.1,
                passed=index != 0,
                constraint_violations=0,
                origin_count=64,
                prediction_cell_count=192,
                cohort_digest=digest({"candidate_id": candidate.candidate_id}),
            )
            for index, candidate in enumerate(candidates)
        )
        for event in written:
            self.assertEqual(
                event.payload["schema_version"],
                "ecologyrsi-dsh.candidate-screening/2",
            )
            self.assertEqual(event.payload["record_digest"], _record_digest(event.payload))
        formal_digest = digest(
            [event.payload for event in sorted(written, key=lambda item: item.payload["candidate_id"])]
        )
        formal = director.freeze_formal_selection_cohort(
            "run:screening-writer",
            generation=0,
            selected_candidate_ids=tuple(item.candidate_id for item in candidates[:2]),
            screening_digest=formal_digest,
        )
        self.assertEqual(
            formal.payload["schema_version"],
            "ecologyrsi-dsh.formal-selection-cohort/2",
        )
        director.screen_out_candidate(
            "run:screening-writer",
            candidates[2].candidate_id,
            generation=0,
            formal_selection_event_id=formal.event_id,
        )
        screened_out = ledger.events("run:screening-writer")[-1]
        self.assertEqual(
            screened_out.payload["schema_version"],
            "ecologyrsi-dsh.candidate-screened-out/1",
        )

    def test_all_failed_screening_is_frozen_as_an_exploration_generation(self) -> None:
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=4))
        run_id = "run:screening-exploration"
        director.start_evolution(
            TaskManifest(
                task_id="screening-exploration",
                objective="classify an all-failed screening generation",
                domain_pack="crop-soil-water@toy",
                budget={"max_candidates": 4, "candidates_per_generation": 4},
            ),
            run_id=run_id,
        )
        candidates = tuple(director.propose_and_spawn(run_id) for _ in range(4))
        screening_events = tuple(
            director.record_candidate_screening(
                run_id,
                candidate_id=candidate.candidate_id,
                generation=0,
                score=-0.2 - index * 0.1,
                passed=False,
                constraint_violations=0,
                origin_count=64,
                prediction_cell_count=192,
                cohort_digest=digest({"candidate": candidate.candidate_id}),
            )
            for index, candidate in enumerate(candidates)
        )
        formal = director.freeze_formal_selection_cohort(
            run_id,
            generation=0,
            selected_candidate_ids=[
                candidates[0].candidate_id,
                candidates[1].candidate_id,
            ],
            screening_digest=digest(
                [
                    event.payload
                    for event in sorted(
                        screening_events,
                        key=lambda item: str(item.payload["candidate_id"]),
                    )
                ]
            ),
            include_exploration_state=True,
        )

        self.assertEqual(
            formal.payload["schema_version"],
            "ecologyrsi-dsh.formal-selection-cohort/3",
        )
        self.assertEqual(formal.payload["screening_pass_count"], 0)
        self.assertTrue(formal.payload["exploration_only"])
        self.assertEqual(formal.payload["consecutive_exploration_generations"], 1)
        replayed = director.replay(run_id).formal_selection_for(0)
        self.assertEqual(replayed.payload, formal.payload)

        events = ledger.events(run_id)
        inflated = tuple(
            replace(
                event,
                payload={
                    **event.payload,
                    "consecutive_exploration_generations": 2,
                },
            )
            if event.event_id == formal.event_id
            else event
            for event in events
        )
        with self.assertRaisesRegex(ValueError, "exploration state"):
            project_run_state(inflated)

        boolean_count = tuple(
            replace(
                event,
                payload={**event.payload, "screening_pass_count": False},
            )
            if event.event_id == formal.event_id
            else event
            for event in events
        )
        with self.assertRaisesRegex(ValueError, "exploration state"):
            project_run_state(boolean_count)

    def test_normal_replay_preserves_selected_ids_and_failed_diagnostic(self) -> None:
        state = project_run_state(self.events)

        formal = state.formal_selection_for(0)
        self.assertIsNotNone(formal)
        assert formal is not None
        self.assertEqual(tuple(formal.payload["selected_candidate_ids"]), self.selected_ids)
        failed_diagnostic = state.screening_for(0, self.selected_ids[0])
        self.assertIsNotNone(failed_diagnostic)
        assert failed_diagnostic is not None
        self.assertIs(failed_diagnostic.payload["passed"], False)
        self.assertEqual(len(state.candidate_screening_events), 4)
        self.assertEqual(len(state.formal_selection_events), 1)
        self.assertEqual(len(state.screened_out_events), 2)

    def test_replay_rejects_screening_generation_forgery(self) -> None:
        event = self.screening_events[0]
        payload = {**event.payload, "generation": 1}
        if "record_digest" in payload:
            payload["record_digest"] = _record_digest(payload)
        events_with_wrong_generation = self._replace_event(self.events, event, payload)

        with self.assertRaisesRegex(ValueError, "screening generation"):
            project_run_state(events_with_wrong_generation)

    def test_replay_rejects_candidate_ownership_forgery(self) -> None:
        candidate_id = self.selected_ids[0]
        spawned = next(
            event
            for event in self.events
            if event.kind == "CandidateSpawned"
            and event.payload["candidate"]["candidate_id"] == candidate_id
        )
        payload = {
            **spawned.payload,
            "candidate": {
                **spawned.payload["candidate"],
                "run_id": "run:forged-owner",
            },
        }
        events_with_wrong_owner = self._replace_event(self.events, spawned, payload)

        with self.assertRaisesRegex(ValueError, "candidate ownership"):
            project_run_state(events_with_wrong_owner)

    def test_replay_rejects_malformed_cohort_digest(self) -> None:
        event = self.screening_events[0]
        payload = {**event.payload, "cohort_digest": "not-a-digest"}
        if "record_digest" in payload:
            payload["record_digest"] = _record_digest(payload)
        events_with_malformed_cohort_digest = self._replace_event(
            self.events, event, payload
        )

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            project_run_state(events_with_malformed_cohort_digest)

    def test_replay_rejects_formal_selection_without_screening(self) -> None:
        missing_id = self.selected_ids[0]
        events_selecting_unscreened_candidate = tuple(
            event
            for event in self.events
            if not (
                event.kind == "CandidateScreeningRecorded"
                and event.payload["candidate_id"] == missing_id
            )
        )

        with self.assertRaisesRegex(ValueError, "missing screening"):
            project_run_state(events_selecting_unscreened_candidate)

    def test_replay_rejects_screening_out_selected_candidate(self) -> None:
        screened_out = next(
            event for event in self.events if event.kind == "CandidateScreenedOut"
        )
        payload = {**screened_out.payload, "candidate_id": self.selected_ids[0]}
        events_screening_out_selected_candidate = self._replace_event(
            self.events, screened_out, payload
        )

        with self.assertRaisesRegex(ValueError, "selected candidate"):
            project_run_state(events_screening_out_selected_candidate)

    def test_replay_rejects_conflicting_screening_identity(self) -> None:
        event = self.screening_events[0]
        payload = {**event.payload, "score": float(event.payload["score"]) - 0.1}
        if "record_digest" in payload:
            payload["record_digest"] = _record_digest(payload)
        conflicting = replace(
            event,
            seq=self.events[-1].seq + 1,
            event_id=f"{event.event_id}:forged-conflict",
            payload=payload,
        )
        events_with_same_id_and_different_payload = (*self.events, conflicting)

        with self.assertRaisesRegex(ValueError, "conflicting screening"):
            project_run_state(events_with_same_id_and_different_payload)

    def test_director_calls_are_idempotent_and_changed_payloads_conflict(self) -> None:
        original = self.screening_events[0]
        retry = self.director.record_candidate_screening(
            self.run_id,
            candidate_id=str(original.payload["candidate_id"]),
            generation=0,
            score=float(original.payload["score"]),
            passed=bool(original.payload["passed"]),
            constraint_violations=int(original.payload["constraint_violations"]),
            origin_count=int(original.payload["origin_count"]),
            prediction_cell_count=int(original.payload["prediction_cell_count"]),
            cohort_digest=str(original.payload["cohort_digest"]),
        )
        self.assertEqual(retry, original)
        formal_retry = self.director.freeze_formal_selection_cohort(
            self.run_id,
            generation=0,
            selected_candidate_ids=self.selected_ids,
            screening_digest=str(self.formal_event.payload["screening_digest"]),
        )
        self.assertEqual(formal_retry, self.formal_event)
        screened_out_id = self.candidates[2].candidate_id
        screened_out_screening = self.screening_events[2]
        screening_retry_after_status_change = (
            self.director.record_candidate_screening(
                self.run_id,
                candidate_id=screened_out_id,
                generation=0,
                score=float(screened_out_screening.payload["score"]),
                passed=bool(screened_out_screening.payload["passed"]),
                constraint_violations=int(
                    screened_out_screening.payload["constraint_violations"]
                ),
                origin_count=int(screened_out_screening.payload["origin_count"]),
                prediction_cell_count=int(
                    screened_out_screening.payload["prediction_cell_count"]
                ),
                cohort_digest=str(screened_out_screening.payload["cohort_digest"]),
            )
        )
        self.assertEqual(screening_retry_after_status_change, screened_out_screening)
        screened_out_retry = self.director.screen_out_candidate(
            self.run_id,
            screened_out_id,
            generation=0,
            formal_selection_event_id=self.formal_event.event_id,
        )
        self.assertEqual(screened_out_retry.candidate_id, screened_out_id)

        with self.assertRaisesRegex(ValueError, "different event"):
            self.director.record_candidate_screening(
                self.run_id,
                candidate_id=str(original.payload["candidate_id"]),
                generation=0,
                score=float(original.payload["score"]) - 0.1,
                passed=bool(original.payload["passed"]),
                constraint_violations=int(original.payload["constraint_violations"]),
                origin_count=int(original.payload["origin_count"]),
                prediction_cell_count=int(original.payload["prediction_cell_count"]),
                cohort_digest=str(original.payload["cohort_digest"]),
            )
        with self.assertRaisesRegex(ValueError, "different event"):
            self.director.freeze_formal_selection_cohort(
                self.run_id,
                generation=0,
                selected_candidate_ids=tuple(reversed(self.selected_ids)),
                screening_digest=str(self.formal_event.payload["screening_digest"]),
            )
        with self.assertRaisesRegex(ValueError, "different event"):
            self.director.screen_out_candidate(
                self.run_id,
                screened_out_id,
                generation=0,
                formal_selection_event_id="forged-formal-event",
            )

    def test_screen_out_without_screening_evidence_commits_no_event(self) -> None:
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=3))
        task = TaskManifest(
            task_id="screened-out-precommit-validation",
            objective="reject missing screening before append",
            domain_pack="crop-soil-water@toy",
            budget={"max_candidates": 3, "candidates_per_generation": 3},
            metadata={
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "sample_budget_class": "selection_eligible",
                "prediction_cells_per_origin": 1,
            },
        )
        run_id = "run:screened-out-precommit-validation"
        director.start_evolution(task, run_id=run_id)
        candidates = tuple(director.propose_and_spawn(run_id) for _ in range(3))
        screening = tuple(
            director.record_candidate_screening(
                run_id,
                candidate_id=candidate.candidate_id,
                generation=0,
                score=0.9 - index * 0.1,
                passed=True,
                constraint_violations=0,
                origin_count=64,
                prediction_cell_count=64,
                cohort_digest=digest({"candidate_id": candidate.candidate_id}),
            )
            for index, candidate in enumerate(candidates[:2])
        )
        formal = director.freeze_formal_selection_cohort(
            run_id,
            generation=0,
            selected_candidate_ids=tuple(
                candidate.candidate_id for candidate in candidates[:2]
            ),
            screening_digest=digest(
                [
                    event.payload
                    for event in sorted(
                        screening,
                        key=lambda event: str(event.payload["candidate_id"]),
                    )
                ]
            ),
        )
        event_count = ledger.count(run_id)

        with self.assertRaisesRegex(ValueError, "missing screening"):
            director.screen_out_candidate(
                run_id,
                candidates[2].candidate_id,
                generation=0,
                formal_selection_event_id=formal.event_id,
            )

        self.assertEqual(ledger.count(run_id), event_count)

    def test_historical_v1_stream_still_replays(self) -> None:
        v1_screening_payloads: dict[str, dict[str, object]] = {}
        converted: list[Event] = []
        for event in self.events:
            payload = dict(event.payload)
            if event.kind == "CandidateScreeningRecorded":
                payload["schema_version"] = "ecologyrsi-dsh.candidate-screening/1"
                payload.pop("record_digest", None)
                v1_screening_payloads[str(payload["candidate_id"])] = payload
            elif event.kind == "CandidateScreenedOut":
                payload.pop("schema_version", None)
            converted.append(replace(event, payload=payload))
        legacy_digest = digest(
            [v1_screening_payloads[key] for key in sorted(v1_screening_payloads)]
        )
        historical_v1_events = tuple(
            replace(
                event,
                payload={
                    **event.payload,
                    "schema_version": "ecologyrsi-dsh.formal-selection-cohort/1",
                    "screening_digest": legacy_digest,
                },
            )
            if event.kind == "FormalSelectionCohortFrozen"
            else event
            for event in converted
        )

        state = project_run_state(historical_v1_events)

        self.assertEqual(
            tuple(state.formal_selection_for(0).payload["selected_candidate_ids"]),
            self.selected_ids,
        )
        self.assertEqual(
            state.screening_for(0, self.selected_ids[0]).payload["schema_version"],
            "ecologyrsi-dsh.candidate-screening/1",
        )


if __name__ == "__main__":
    unittest.main()
