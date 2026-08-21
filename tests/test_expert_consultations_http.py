from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ecologyrsi_dsh import ExpertConsultation, TaskManifest
from ecologyrsi_dsh.application.config import bind_toy_dataset
from ecologyrsi_dsh.api.events import EventEndpointsMixin
from ecologyrsi_dsh.core.ledger import ConcurrentRunMutationError, Event
from ecologyrsi_dsh.server import EvolutionHTTPServer


class ExpertConsultationHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.server = EvolutionHTTPServer(
            ("127.0.0.1", 0),
            Path(self._directory.name) / "events.sqlite3",
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        task = bind_toy_dataset(
            TaskManifest(
                task_id="expert-consultation-http",
                objective="exercise asynchronous expert collaboration",
                domain_pack="crop-soil-water@toy",
                visible_datasets=("generated-toy-series@1",),
                budget={"max_candidates": 3, "max_generations": 3},
                seed=7,
            ),
            required=True,
        )
        self.run_id = self.server.director.start_evolution(
            task,
            run_id="run:expert-consultation-http",
        ).run.run_id
        self.consultation_id = "expert-consultation:http-1"
        self.server.director.record_expert_consultation(
            ExpertConsultation(
                consultation_id=self.consultation_id,
                run_id=self.run_id,
                generation=0,
                uncertainty_type="scientific_assumption",
                question="Should water balance remain a hard constraint?",
                context="Aggregate dry-period errors remain ambiguous.",
                fallback_assumption="Keep the registered hard constraint.",
                requested_expertise=("crop hydrology",),
                options=("keep hard constraint", "request more evidence"),
                confidence=0.46,
                requested_by_model_id="strategy-model",
            )
        )

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.close()
        self._directory.cleanup()

    def request(
        self,
        path: str,
        method: str = "GET",
        body: dict | None = None,
    ) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=3) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    @property
    def run_path(self) -> str:
        return quote(self.run_id, safe="")

    @property
    def answer_path(self) -> str:
        consultation = quote(self.consultation_id, safe="")
        return (
            f"/api/runs/{self.run_path}/expert-consultations/"
            f"{consultation}/answer"
        )

    def answer_body(self, *, key: str = "expert-answer-http") -> dict:
        return {
            "answer": "Keep the hard constraint for the next search round.",
            "selected_option": "keep hard constraint",
            "answered_by": "domain-expert-7",
            "idempotency_key": key,
        }

    def test_running_answer_is_non_blocking_projected_and_idempotent(self) -> None:
        status, initial = self.request(f"/api/runs/{self.run_path}")
        self.assertEqual(status, 200)
        pending = initial["projection"]["expert_consultations"]
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["status"], "pending")
        self.assertEqual(pending[0]["source_generation"], 1)
        self.assertIs(pending[0]["non_blocking"], True)

        with patch.object(
            self.server.auto_progress,
            "schedule_if_enabled",
            wraps=self.server.auto_progress.schedule_if_enabled,
        ) as schedule:
            status, answered = self.request(
                self.answer_path,
                "POST",
                self.answer_body(),
            )
        self.assertEqual(status, 201, answered)
        schedule.assert_not_called()
        projection = answered["projection"]
        self.assertEqual(projection["status"], "running")
        consultation = projection["expert_consultations"][0]
        self.assertEqual(consultation["status"], "answered")
        self.assertEqual(consultation["answer"], self.answer_body()["answer"])
        self.assertEqual(consultation["answered_by"], "domain-expert-7")
        self.assertEqual(consultation["effective_generation"], 2)
        self.assertIsNone(consultation["applied_generation"])

        repeat_status, repeated = self.request(
            self.answer_path,
            "POST",
            self.answer_body(),
        )
        self.assertEqual(repeat_status, 200, repeated)
        state = self.server.director.state(self.run_id)
        self.assertEqual(
            sum(
                event.kind == "ExpertConsultationAnswered"
                for event in state.events
            ),
            1,
        )

        event_status, events = self.request(f"/api/runs/{self.run_path}/events")
        self.assertEqual(event_status, 200)
        answer_event = next(
            item
            for item in events["events"]
            if item["kind"] == "ExpertConsultationAnswered"
        )
        self.assertEqual(
            answer_event["event_type"],
            "expert_consultation.answered",
        )
        self.assertEqual(answer_event["payload"]["effective_generation"], 2)
        self.assertIs(answer_event["payload"]["audit_only"], False)

    def test_terminal_answer_is_retained_for_audit_without_effective_round(self) -> None:
        self.server.director.complete_run(self.run_id)

        status, answered = self.request(
            self.answer_path,
            "POST",
            self.answer_body(key="terminal-expert-answer"),
        )

        self.assertEqual(status, 201, answered)
        projection = answered["projection"]
        self.assertEqual(projection["status"], "completed")
        consultation = projection["expert_consultations"][0]
        self.assertEqual(consultation["status"], "answered")
        self.assertIsNone(consultation["effective_generation"])
        self.assertIsNone(consultation["applied_generation"])
        events = self.request(f"/api/runs/{self.run_path}/events")[1]["events"]
        answer_event = next(
            item
            for item in events
            if item["kind"] == "ExpertConsultationAnswered"
        )
        self.assertIs(answer_event["payload"]["audit_only"], True)

    def test_invalid_option_and_unknown_field_do_not_write_an_answer(self) -> None:
        invalid = self.answer_body(key="invalid-option")
        invalid["selected_option"] = "disable every constraint"
        status, payload = self.request(self.answer_path, "POST", invalid)
        self.assertEqual(status, 400, payload)

        unknown = self.answer_body(key="unknown-answer-field")
        unknown["parameter_overrides"] = {"alpha": 1}
        status, payload = self.request(self.answer_path, "POST", unknown)
        self.assertEqual(status, 400, payload)

        structured = self.answer_body(key="structured-answer")
        structured["answer"] = {"parameter_overrides": {"alpha": 1}}
        status, payload = self.request(self.answer_path, "POST", structured)
        self.assertEqual(status, 400, payload)

        structured_actor = self.answer_body(key="structured-actor")
        structured_actor["answered_by"] = ["expert", "admin"]
        status, payload = self.request(
            self.answer_path,
            "POST",
            structured_actor,
        )
        self.assertEqual(status, 400, payload)
        self.assertIsNone(
            self.server.director.state(self.run_id).answer_for_consultation(
                self.consultation_id
            )
        )
        self.assertEqual(self.server.ledger.pending_command_keys(), ())

    def test_projection_and_event_redact_credential_like_text(self) -> None:
        second_id = "expert-consultation:redaction"
        self.server.director.record_expert_consultation(
            ExpertConsultation(
                consultation_id=second_id,
                run_id=self.run_id,
                generation=0,
                uncertainty_type="data_interpretation",
                question="api_key=not-public",
                context="password=not-public",
                fallback_assumption="Use aggregate-only evidence.",
                requested_expertise=("data governance",),
                options=(),
                confidence=0.3,
                requested_by_model_id="strategy-model",
            )
        )
        second_path = (
            f"/api/runs/{self.run_path}/expert-consultations/"
            f"{quote(second_id, safe='')}/answer"
        )
        status, _payload = self.request(
            second_path,
            "POST",
            {
                "answer": "api_key=also-not-public",
                "answered_by": "data-governance-expert",
                "idempotency_key": "redacted-expert-answer",
            },
        )
        self.assertEqual(status, 201)
        projection = self.request(f"/api/runs/{self.run_path}")[1]["projection"]
        item = next(
            row
            for row in projection["expert_consultations"]
            if row["consultation_id"] == second_id
        )
        self.assertEqual(item["question"], "[REDACTED]")
        self.assertEqual(item["context"], "[REDACTED]")
        self.assertEqual(item["answer"], "[REDACTED]")

        events = self.request(f"/api/runs/{self.run_path}/events")[1]["events"]
        event = next(
            row
            for row in events
            if row["kind"] == "ExpertConsultationRequested"
            and row["payload"]["consultation_id"] == second_id
        )
        self.assertEqual(event["payload"]["question"], "[REDACTED]")
        self.assertEqual(event["payload"]["context"], "[REDACTED]")
        answered_event = next(
            row
            for row in events
            if row["kind"] == "ExpertConsultationAnswered"
            and row["payload"]["consultation_id"] == second_id
        )
        self.assertEqual(answered_event["payload"]["answer"], "[REDACTED]")

    def test_applied_event_has_public_alias_and_display_generation(self) -> None:
        projected = EventEndpointsMixin._event_json(
            Event(
                seq=8,
                event_id="event:expert-consultation-applied",
                run_id=self.run_id,
                kind="ExpertConsultationApplied",
                payload={
                    "consultation_id": self.consultation_id,
                    "answer_id": "expert-answer:1",
                    "generation": 1,
                    "research_iteration_digest": "sha256:iteration",
                },
                created_at="2026-08-20T00:00:00Z",
            )
        )

        self.assertEqual(projected["event_type"], "expert_consultation.applied")
        self.assertEqual(projected["payload"]["applied_generation"], 2)
        self.assertEqual(
            projected["payload"]["research_iteration_digest"],
            "sha256:iteration",
        )

    def test_cas_retry_recomputes_effective_generation(self) -> None:
        original = self.server.director.answer_expert_consultation
        raced = False

        def answer_after_generation_race(answer):
            nonlocal raced
            if not raced:
                raced = True
                state = self.server.director.state(self.run_id)
                self.server.ledger.append(
                    self.run_id,
                    "GenerationAdvanced",
                    {"generation": state.run.generation + 1},
                    event_id=f"{self.run_id}:test-generation-race",
                    expected_run_seq=state.events[-1].seq,
                )
                raise ConcurrentRunMutationError("simulated generation race")
            return original(answer)

        with patch.object(
            self.server.director,
            "answer_expert_consultation",
            side_effect=answer_after_generation_race,
        ):
            status, payload = self.request(
                self.answer_path,
                "POST",
                self.answer_body(key="cas-retry-answer"),
            )

        self.assertEqual(status, 201, payload)
        self.assertEqual(payload["projection"]["generation"], 1)
        consultation = payload["projection"]["expert_consultations"][0]
        # Internal generation 2 is the next unopened generation after the race;
        # the public projection presents it as round 3.
        self.assertEqual(consultation["effective_generation"], 3)


if __name__ == "__main__":
    unittest.main()
