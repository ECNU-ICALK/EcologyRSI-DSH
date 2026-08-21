from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import (
    ExpertConsultation,
    ExpertConsultationAnswer,
    RunStatus,
    TaskManifest,
    digest,
)
from ecologyrsi_dsh.evolution.strategies import FakeDSHAdapter
from ecologyrsi_dsh.knowledge.algorithms import resolve_predictor_adoption
from ecologyrsi_dsh.knowledge.research_iteration import ResearchIteration
from ecologyrsi_dsh.knowledge.retrieval import retrieve_generation_knowledge


class ExpertConsultationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.ledger = EventLedger(Path(self.directory.name) / "events.sqlite3")
        self.director = EvolutionDirector(self.ledger, FakeDSHAdapter())
        self.task = TaskManifest(
            task_id="expert-consultations",
            objective="improve a bounded ecological forecast",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={
                "max_candidates": 3,
                "max_generations": 3,
                "candidates_per_generation": 1,
            },
            metadata={
                "domain": "toy",
                "autonomous_mode": True,
                "strategy_id": "autonomous_model@1",
                "strategy_model_id": "research-model",
                "prediction_model_id": "toy-rolling-water@1",
                "evaluator_id": "toy_time_forward@1",
                "knowledge_online_enabled": False,
            },
        )
        self.run_id = self.director.start_evolution(
            self.task,
            run_id="run:expert-consultations",
        ).run.run_id

    def tearDown(self) -> None:
        self.ledger.close()
        self.directory.cleanup()

    def consultation(
        self,
        *,
        consultation_id: str = "consultation:water-threshold",
        generation: int = 0,
        question: str = "Which conservative soil-water threshold is defensible?",
        non_blocking: bool = True,
    ) -> ExpertConsultation:
        return ExpertConsultation(
            consultation_id=consultation_id,
            run_id=self.run_id,
            generation=generation,
            uncertainty_type="scientific_assumption",
            question=question,
            context=(
                "The aggregate soil-moisture skill is unstable at long horizons; "
                "no sample rows or labels are included."
            ),
            fallback_assumption="retain the registered conservative default",
            requested_expertise=("soil physics", "crop water stress"),
            options=("retain default", "use a lower conservative bound"),
            confidence=0.42,
            requested_by_model_id="research-model",
            non_blocking=non_blocking,
            created_at="2026-08-20T01:00:00+00:00",
        )

    def answer(
        self,
        *,
        consultation_id: str = "consultation:water-threshold",
        answer_id: str = "answer:water-threshold",
        effective_generation: int | None = 1,
        selected_option: str | None = "retain default",
    ) -> ExpertConsultationAnswer:
        return ExpertConsultationAnswer(
            answer_id=answer_id,
            run_id=self.run_id,
            consultation_id=consultation_id,
            answer="Retain the default until a sensitivity analysis is available.",
            answered_by="domain-expert",
            selected_option=selected_option,
            effective_generation=effective_generation,
            created_at="2026-08-20T02:00:00+00:00",
        )

    def record_knowledge(self):
        state = self.director.state(self.run_id)
        snapshot = retrieve_generation_knowledge(state)
        self.ledger.append(
            self.run_id,
            "GenerationKnowledgeRetrieved",
            {"knowledge_snapshot": snapshot.to_dict()},
            event_id=(
                f"{self.run_id}:generation:{state.run.generation}:"
                "expert-test-knowledge"
            ),
        )
        return snapshot

    def research_iteration(
        self,
        *,
        generation: int,
        knowledge_snapshot_digest: str,
        expert_answer_ids: tuple[str, ...] = (),
    ) -> ResearchIteration:
        plan = {
            "strategy": {
                "id": "autonomous_model@1",
                "rationale": "retain a bounded registered search",
            }
        }
        return ResearchIteration(
            run_id=self.run_id,
            generation=generation,
            status="model_generated",
            plan=plan,
            prediction_model_adoption=resolve_predictor_adoption(
                self.task,
                plan,
            ).to_dict(),
            knowledge_snapshot_digest=knowledge_snapshot_digest,
            model_id="research-model",
            expert_answer_ids=expert_answer_ids,
            created_at=f"2026-08-20T0{generation + 3}:00:00+00:00",
        )

    def test_running_consultation_is_pending_non_blocking_and_replayable(self) -> None:
        manifest_digest = self.task.digest
        consultation = self.consultation()

        recorded = self.director.record_expert_consultation(consultation)
        repeated = self.director.record_expert_consultation(consultation)
        state = self.director.state(self.run_id)

        self.assertEqual(recorded, consultation)
        self.assertEqual(repeated, consultation)
        self.assertIs(state.run.status, RunStatus.RUNNING)
        self.assertEqual(state.task_manifest.digest, manifest_digest)
        self.assertEqual(
            state.consultation(consultation.consultation_id),
            consultation,
        )
        self.assertEqual(state.pending_expert_consultations, (consultation,))
        self.assertFalse(any(event.kind == "RunPaused" for event in state.events))
        self.assertEqual(
            sum(event.kind == "ExpertConsultationRequested" for event in state.events),
            1,
        )

        conflicting_data = consultation.to_dict()
        conflicting_data["question"] = "A conflicting question reusing the same ID"
        with self.assertRaises(ValueError):
            self.director.record_expert_consultation(
                ExpertConsultation.from_dict(conflicting_data)
            )

        self.director.advance_generation(self.run_id)
        advanced = self.director.state(self.run_id)
        self.assertIs(advanced.run.status, RunStatus.RUNNING)
        self.assertEqual(advanced.run.generation, 1)
        self.assertEqual(advanced.pending_expert_consultations, (consultation,))

        replayed = EvolutionDirector(
            self.ledger,
            FakeDSHAdapter(),
        ).replay(self.run_id)
        self.assertEqual(
            replayed.consultation(consultation.consultation_id),
            consultation,
        )
        self.assertEqual(replayed.pending_expert_consultations, (consultation,))

    def test_consultation_contract_rejects_blocking_and_unknown_privileged_fields(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            self.consultation(non_blocking=False)

        consultation_data = self.consultation().to_dict()
        for field, value in (
            ("blocking", True),
            ("requested_capabilities", ["hidden.read"]),
            ("ground_truth", [0.7]),
        ):
            with self.subTest(field=field):
                malformed = {**consultation_data, field: value}
                with self.assertRaises(TypeError):
                    ExpertConsultation.from_dict(malformed)

        answer_data = self.answer().to_dict()
        with self.assertRaises(TypeError):
            ExpertConsultationAnswer.from_dict(
                {**answer_data, "tool_permissions": ["network.any"]}
            )

    def test_answer_is_available_only_later_and_consumed_exactly_once(self) -> None:
        consultation = self.director.record_expert_consultation(self.consultation())
        answer = self.director.answer_expert_consultation(self.answer())
        answered_state = self.director.state(self.run_id)

        self.assertEqual(answered_state.pending_expert_consultations, ())
        self.assertEqual(answered_state.available_expert_answers(0), ())
        self.assertEqual(answered_state.available_expert_answers(1), (answer,))
        self.assertIsNone(
            answered_state.answer_for_consultation(
                consultation.consultation_id
            ).applied_generation
        )

        self.director.advance_generation(self.run_id)
        knowledge = self.record_knowledge()
        iteration = self.research_iteration(
            generation=1,
            knowledge_snapshot_digest=knowledge.snapshot_digest,
            expert_answer_ids=(answer.answer_id,),
        )
        next_consultation = self.consultation(
            consultation_id="consultation:next-round",
            generation=1,
            question="Should the next round prioritize stability or mean skill?",
        )

        recorded_iteration = self.director.record_research_iteration(
            iteration,
            expert_consultation=next_consultation,
        )
        state = self.director.state(self.run_id)

        self.assertEqual(recorded_iteration, iteration)
        self.assertEqual(
            state.research_iteration_for(1).expert_answer_ids,
            (answer.answer_id,),
        )
        applied_answer = state.answer_for_consultation(consultation.consultation_id)
        self.assertIsNotNone(applied_answer)
        assert applied_answer is not None
        self.assertEqual(applied_answer.applied_generation, 1)
        self.assertEqual(state.available_expert_answers(1), ())
        self.assertEqual(state.pending_expert_consultations, (next_consultation,))

        generation_events = [
            event
            for event in state.events
            if event.kind
            in {
                "GenerationResearchIterated",
                "ExpertConsultationApplied",
                "ExpertConsultationRequested",
            }
            and event.seq > answered_state.events[-1].seq
        ]
        self.assertEqual(
            [event.kind for event in generation_events],
            [
                "GenerationResearchIterated",
                "ExpertConsultationApplied",
                "ExpertConsultationRequested",
            ],
        )
        applied_payload = generation_events[1].payload
        self.assertEqual(
            applied_payload["consultation_id"],
            consultation.consultation_id,
        )
        self.assertEqual(applied_payload["answer_id"], answer.answer_id)
        self.assertEqual(applied_payload["generation"], 1)
        self.assertEqual(
            applied_payload["research_iteration_digest"],
            iteration.iteration_digest,
        )

        repeated = self.director.record_research_iteration(
            iteration,
            expert_consultation=next_consultation,
        )
        repeated_state = self.director.state(self.run_id)
        self.assertEqual(repeated, iteration)
        self.assertEqual(
            sum(
                event.kind == "ExpertConsultationApplied"
                and event.payload.get("answer_id") == answer.answer_id
                for event in repeated_state.events
            ),
            1,
        )
        self.assertEqual(
            sum(
                event.kind == "ExpertConsultationRequested"
                and event.payload.get("consultation", {}).get("consultation_id")
                == next_consultation.consultation_id
                for event in repeated_state.events
            ),
            1,
        )

        replayed = EvolutionDirector(
            self.ledger,
            FakeDSHAdapter(),
        ).replay(self.run_id)
        self.assertEqual(
            replayed.answer_for_consultation(
                consultation.consultation_id
            ).applied_generation,
            1,
        )
        self.assertEqual(replayed.available_expert_answers(1), ())

    def test_empty_expert_answer_ids_preserve_legacy_iteration_digest(self) -> None:
        plan = {"strategy": {"id": "autonomous_model@1"}}
        adoption = resolve_predictor_adoption(self.task, plan).to_dict()
        legacy_identity = {
            "schema_version": "ecologyrsi-dsh.research-iteration/1",
            "run_id": self.run_id,
            "generation": 0,
            "status": "model_generated",
            "plan": plan,
            "prediction_model_adoption": adoption,
            "knowledge_snapshot_digest": "k" * 64,
            "source_analysis_digest": None,
            "source_assessment_digest": None,
            "previous_next_action": None,
            "model_id": "research-model",
            "security_boundary": {
                "advisory_output_only": True,
                "registered_host_capabilities_only": True,
                "model_generated_code_execution": False,
                "dynamic_imports": False,
                "shell_execution": False,
            },
            "created_at": "2026-08-20T03:00:00+00:00",
        }
        legacy_digest = digest(legacy_identity)

        replayed = ResearchIteration.from_dict(
            {**legacy_identity, "iteration_digest": legacy_digest}
        )

        self.assertEqual(replayed.expert_answer_ids, ())
        self.assertEqual(replayed.iteration_digest, legacy_digest)
        self.assertNotIn("expert_answer_ids", replayed.to_dict())

    def test_terminal_run_accepts_late_answer_as_audit_only(self) -> None:
        consultation = self.director.record_expert_consultation(self.consultation())
        self.director.complete_run(self.run_id)

        answer = self.answer(effective_generation=None, selected_option=None)
        recorded = self.director.answer_expert_consultation(answer)
        repeated = self.director.answer_expert_consultation(answer)
        state = self.director.state(self.run_id)

        self.assertEqual(recorded, answer)
        self.assertEqual(repeated, answer)
        self.assertIs(state.run.status, RunStatus.COMPLETED)
        self.assertEqual(
            state.answer_for_consultation(consultation.consultation_id),
            answer,
        )
        self.assertIsNone(recorded.effective_generation)
        self.assertIsNone(recorded.applied_generation)
        self.assertEqual(state.available_expert_answers(1), ())
        self.assertFalse(
            any(event.kind == "ExpertConsultationApplied" for event in state.events)
        )
        self.assertEqual(
            sum(event.kind == "ExpertConsultationAnswered" for event in state.events),
            1,
        )


if __name__ == "__main__":
    unittest.main()
