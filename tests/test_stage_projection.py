from __future__ import annotations

import unittest

from ecologyrsi_dsh import (
    EventLedger,
    EvolutionDirector,
    TaskManifest,
)
from ecologyrsi_dsh.knowledge.research_iteration import ResearchIteration
from tests.native_fixtures import NativeRoleStub, native_manifest, start_native_run, spawn_native_candidate
from ecologyrsi_dsh.knowledge.retrieval import retrieve_generation_knowledge
from ecologyrsi_dsh.presentation.reporting import rounds, run_summary
from ecologyrsi_dsh.api.projection import (
    _candidate_selection_disposition,
    _projection_json,
)


def task() -> TaskManifest:
    return native_manifest(task_id="stage-projection-task", generations=2)


def stage_payload(*, stage: str = "proposal", status: str = "started") -> dict:
    return {
        "generation": 0,
        "proposal_id": None,
        "candidate_id": None,
        "stage": stage,
        "status": status,
        "attempt": 1,
        "public_error": None,
    }


class EvolutionStageProjectionTests(unittest.TestCase):
    def test_candidate_projection_exposes_selection_disposition(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = "run:candidate-selection-disposition"
            start_native_run(director, task=task(), run_id=run_id)
            candidate = spawn_native_candidate(director, run_id)
            projection = _projection_json(director.state(run_id))

        projected = next(
            item
            for item in projection["candidates"]
            if item["candidate_id"] == candidate.candidate_id
        )
        self.assertEqual(projected["selection_disposition"], "pending")

    def test_candidate_selection_disposition_separates_incumbent_and_search_roles(
        self,
    ) -> None:
        self.assertEqual(
            _candidate_selection_disposition(
                candidate_id="c-current",
                incumbent_id="c-current",
                promotion_decision="approved",
                search_parent_id="c-parent",
                candidate_status="promoted",
            ),
            "incumbent",
        )
        self.assertEqual(
            _candidate_selection_disposition(
                candidate_id="c-parent",
                incumbent_id="c-current",
                promotion_decision=None,
                search_parent_id="c-parent",
                candidate_status="evaluated",
            ),
            "search_parent",
        )
        self.assertEqual(
            _candidate_selection_disposition(
                candidate_id="c-old",
                incumbent_id="c-current",
                promotion_decision="approved",
                search_parent_id=None,
                candidate_status="promoted",
            ),
            "promoted_historical",
        )
        self.assertEqual(
            _candidate_selection_disposition(
                candidate_id="c-failed",
                incumbent_id="c-current",
                promotion_decision=None,
                search_parent_id=None,
                candidate_status="failed",
            ),
            "failed",
        )

    def test_round_projects_completed_native_dsh_research_and_reflection(
        self,
    ) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = "run:native-research-round-summary"
            start_native_run(director, task=task(), run_id=run_id)
            state = director.state(run_id)
            snapshot = retrieve_generation_knowledge(state)
            ledger.append(
                run_id,
                "GenerationKnowledgeRetrieved",
                {"knowledge_snapshot": snapshot.to_dict()},
            )
            plan = {
                "status": "model_generated",
                "prediction_model": {"id": "greenhouse-horizon-targetwise-ridge@1"},
                "dsh_research_summary": (
                    "Prioritize the weak 24-hour target while keeping the "
                    "registered predictor boundary."
                ),
                "dsh_research_evidence": [
                    {
                        "finding": "The preceding 24-hour skill was negative.",
                        "relevance": "Test a small registered parameter delta.",
                        "evidence_digest": "a" * 64,
                    }
                ],
                "dsh_evolution_reflection": {
                    "schema_version": "ecologyrsi-dsh.evolution-reflection/2",
                    "avoid_behaviors": [
                        {
                            "behavior_digest": "b" * 64,
                            "prediction_model_id": "greenhouse-horizon-targetwise-ridge@1",
                            "parameters_digest": "c" * 64,
                            "reason": "scientific_gate_failed",
                            "source_run_id": "run:prior",
                            "source_generation": 0,
                        }
                    ],
                    "active_unresolved": [
                        {
                            "failure_code": "scientific_gate_failed",
                            "next_action": "improve_24h_skill",
                        }
                    ],
                    "policy": {
                        "exact_failed_behavior_replay": "reject_and_retry_once",
                        "host_enforced": True,
                    },
                },
            }
            iteration = ResearchIteration(
                run_id=run_id,
                generation=0,
                status="model_generated",
                plan=plan,
                knowledge_snapshot_digest=snapshot.snapshot_digest,
                model_id="dsh-researcher",
            )
            director.record_research_iteration(iteration)

            projected = rounds(director.state(run_id))[0]["research_iteration"]

            self.assertEqual(projected["analysis_summary"]["status"], "completed")
            self.assertEqual(
                projected["analysis_summary"]["summary"],
                plan["dsh_research_summary"],
            )
            self.assertEqual(
                projected["analysis_summary"]["source"], "dsh_native_research"
            )
            self.assertEqual(
                projected["analysis_summary"]["key_findings"][0]["finding"],
                "The preceding 24-hour skill was negative.",
            )
            self.assertEqual(
                projected["final_plan"]["status"], "ready_for_host_compilation"
            )
            self.assertEqual(
                projected["evolution_reflection"]["avoid_behavior_count"], 1
            )
            self.assertEqual(
                projected["evolution_reflection"]["active_unresolved"][0][
                    "next_action"
                ],
                "improve_24h_skill",
            )

    def _candidate_stage_views(
        self,
        director: EvolutionDirector,
        run_id: str,
        candidate_id: str,
    ) -> tuple[dict[str, str], dict[str, str]]:
        state = director.state(run_id)
        round_view = next(
            item
            for round_item in rounds(state)
            for item in round_item["candidates"]
            if item["candidate_id"] == candidate_id
        )["stages"]
        candidate_view = next(
            item
            for item in _projection_json(state)["candidates"]
            if item["id"] == candidate_id
        )["execution"]["stages"]
        self.assertEqual(round_view, candidate_view)
        return round_view, candidate_view

    def test_round_projects_only_bounded_native_research_fields(self) -> None:
        from ecologyrsi_dsh.presentation.reporting import _research_iteration_summary
        plan = {
            "dsh_research_summary": "s" * 3000,
            "dsh_research_evidence": [{"finding": "f" * 1200, "source": "public source",
                "evidence_digest": "a" * 64, "private_reasoning": "MUST_NOT_PROJECT"}] * 12,
            "candidate_directions": [{"direction_id": "d1", "title": "bounded direction",
                "private_reasoning": "MUST_NOT_PROJECT"}] * 12,
            "private_reasoning": "MUST_NOT_PROJECT",
        }
        iteration = ResearchIteration(run_id="run:bounded-summary", generation=0,
            status="model_generated", plan=plan, knowledge_snapshot_digest="b" * 64,
            model_id="fixture/strategy")
        projected = _research_iteration_summary(iteration)
        self.assertEqual(projected["iteration_digest"], iteration.iteration_digest)
        self.assertEqual(projected["knowledge_snapshot_digest"], "b" * 64)
        self.assertEqual(projected["model_id"], "fixture/strategy")
        self.assertEqual(projected["analysis_summary"]["source"], "dsh_native_research")
        self.assertEqual(len(projected["analysis_summary"]["summary"]), 2000)
        self.assertEqual(len(projected["analysis_summary"]["key_findings"]), 8)
        self.assertEqual(len(projected["analysis_summary"]["key_findings"][0]["finding"]), 1000)
        self.assertEqual(len(projected["candidate_directions"]), 8)
        self.assertEqual(projected["final_plan"]["status"], "ready_for_host_compilation")
        self.assertEqual(projected["final_plan"]["implementation_mode"], "registered_host_components_only")
        self.assertNotIn("MUST_NOT_PROJECT", str(projected))
        self.assertNotIn("private_reasoning", str(projected))

    def test_replay_rejects_unknown_stage_and_status(self) -> None:
        invalid_cases = (
            (stage_payload(stage="deployment"), "unknown evolution stage"),
            (stage_payload(status="queued"), "unknown evolution stage status"),
        )
        for payload, expected_error in invalid_cases:
            with self.subTest(payload=payload):
                with EventLedger() as ledger:
                    director = EvolutionDirector(ledger, NativeRoleStub())
                    start_native_run(director, task=task(), run_id="run:invalid-stage")
                    ledger.append(
                        "run:invalid-stage",
                        "EvolutionStageRecorded",
                        payload,
                    )
                    with self.assertRaisesRegex(ValueError, expected_error):
                        director.replay("run:invalid-stage")

    def test_latest_recorded_status_overrides_inference_without_mutating_core(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            start_native_run(director, task=task(), run_id="run:recorded-stage")
            candidate = spawn_native_candidate(director, "run:recorded-stage")
            proposal = director.state("run:recorded-stage").proposal(
                candidate.proposal_id
            )
            core_before = director.state("run:recorded-stage")

            first = director.record_evolution_stage(
                "run:recorded-stage",
                generation=0,
                proposal_id=proposal.proposal_id,
                candidate_id=candidate.candidate_id,
                stage="proposal",
                status="completed",
            )
            duplicate = director.record_evolution_stage(
                "run:recorded-stage",
                generation=0,
                proposal_id=proposal.proposal_id,
                candidate_id=candidate.candidate_id,
                stage="proposal",
                status="completed",
            )
            self.assertEqual(duplicate, first)

            recorded = (
                ("proposal", "started", 2),
                ("candidate", "failed", 1),
                ("training", "completed", 1),
                ("evaluation", "started", 1),
                ("judge", "failed", 1),
                ("decision", "completed", 1),
            )
            for stage, status, attempt in recorded:
                director.record_evolution_stage(
                    "run:recorded-stage",
                    generation=0,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate.candidate_id,
                    stage=stage,
                    status=status,
                    attempt=attempt,
                    public_error="可公开失败摘要" if status == "failed" else None,
                )

            replayed = director.replay("run:recorded-stage")
            self.assertEqual(replayed.run, core_before.run)
            self.assertEqual(replayed.proposals, core_before.proposals)
            self.assertEqual(replayed.candidates, core_before.candidates)
            self.assertEqual(replayed.artifacts, core_before.artifacts)
            self.assertEqual(replayed.evaluations, core_before.evaluations)
            self.assertEqual(replayed.promotions, core_before.promotions)

            projected = rounds(replayed)
            self.assertEqual(len(projected), 1)
            self.assertEqual(
                projected[0]["stages"],
                {
                    "proposal": "running",
                    "candidate": "failed",
                    "training": "completed",
                    "evaluation": "running",
                    "judge": "failed",
                    "decision": "completed",
                },
            )
            self.assertEqual(sum(event.kind == "EvolutionStageRecorded" for event in replayed.events), 7)

    def test_started_stage_is_visible_before_proposal_exists(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            start_native_run(director, task=task(), run_id="run:stage-only")
            director.record_evolution_stage(
                "run:stage-only",
                generation=0,
                stage="proposal",
                status="started",
            )

            projected = rounds(director.replay("run:stage-only"))
            self.assertEqual(len(projected), 1)
            self.assertEqual(projected[0]["generation"], 1)
            self.assertIsNone(projected[0]["proposal_id"])
            self.assertEqual(projected[0]["stages"]["proposal"], "running")
            self.assertEqual(projected[0]["stages"]["candidate"], "pending")
            self.assertEqual(
                run_summary(director.replay("run:stage-only"))["round_count"], 1
            )

    def test_cancelled_run_rejects_late_stage_observations(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = "run:cancelled-stage-fence"
            start_native_run(director, task=task(), run_id=run_id)
            director.cancel_run(run_id, "operator cancelled")
            event_count = ledger.count(run_id)

            with self.assertRaisesRegex(RuntimeError, "is cancelled; expected running"):
                director.record_evolution_stage(
                    run_id,
                    generation=0,
                    stage="evaluation",
                    status="failed",
                    public_error="late worker completion",
                )

            self.assertEqual(ledger.count(run_id), event_count)
            self.assertEqual(director.state(run_id).run.status.value, "cancelled")

    def test_gateway_retry_heartbeat_is_projected_as_live_wait(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = "run:gateway-retry-heartbeat"
            start_native_run(director, task=task(), run_id=run_id)
            state = director.state(run_id)
            director.schedule_gateway_retry_or_pause(run_id,
                run_incarnation=state.events[0].seq, generation=0, stage="research",
                retry_class="dsh_native_runtime", failure_id="a" * 64,
                attempt_anchor_seq=state.events[-1].seq, delay_seconds=15.,
                last_error_code="provider_unavailable")
            progress = _projection_json(director.replay(run_id))["execution_progress"]
            self.assertEqual(progress["phase"], "gateway_retry")
            self.assertTrue(progress["retry_wait"]["waiting"])
            self.assertEqual(progress["retry_wait"]["attempt"], 1)

    def test_gateway_retry_wait_clears_when_the_same_stage_restarts(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = "run:gateway-retry-restarted"
            start_native_run(director, task=task(), run_id=run_id)
            state = director.state(run_id)
            director.schedule_gateway_retry_or_pause(run_id,
                run_incarnation=state.events[0].seq, generation=0, stage="research",
                retry_class="dsh_native_runtime", failure_id="a" * 64,
                attempt_anchor_seq=state.events[-1].seq, delay_seconds=15.,
                last_error_code="provider_unavailable")
            director.record_evolution_stage(
                run_id,
                generation=0,
                stage="research",
                status="started",
                attempt=2,
            )

            progress = _projection_json(director.replay(run_id))["execution_progress"]

            self.assertNotEqual(progress["phase"], "gateway_retry")
            self.assertIsNone(progress.get("retry_wait"))

    def test_unscoped_stage_does_not_override_a_completed_candidate_stage(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            start_native_run(director, task=task(), run_id="run:scoped-stage")
            candidate = spawn_native_candidate(director, "run:scoped-stage")
            director.record_evolution_stage(
                "run:scoped-stage",
                generation=0,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                stage="proposal",
                status="completed",
            )
            director.record_evolution_stage(
                "run:scoped-stage",
                generation=0,
                stage="proposal",
                status="started",
                attempt=2,
            )

            projected = rounds(director.replay("run:scoped-stage"))[0]
            self.assertEqual(projected["stages"]["proposal"], "completed")
            self.assertEqual(
                projected["candidates"][0]["stages"]["proposal"], "completed"
            )

    def test_missing_candidate_id_is_scoped_by_proposal_id(self) -> None:
        """Proposal-scoped stage events must not bleed into sibling rows."""

        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            start_native_run(director, task=task(), run_id="run:proposal-scoped-stage")
            first = spawn_native_candidate(director, "run:proposal-scoped-stage")
            second = spawn_native_candidate(director, "run:proposal-scoped-stage")
            director.record_evolution_stage(
                "run:proposal-scoped-stage",
                generation=0,
                proposal_id=first.proposal_id,
                candidate_id=first.candidate_id,
                stage="proposal",
                status="completed",
            )
            # A stage can identify its proposal before a candidate is known;
            # the first candidate must retain its own status.
            director.record_evolution_stage(
                "run:proposal-scoped-stage",
                generation=0,
                proposal_id=second.proposal_id,
                stage="proposal",
                status="started",
            )

            projected = rounds(director.replay("run:proposal-scoped-stage"))[0]
            by_candidate = {
                item["candidate_id"]: item for item in projected["candidates"]
            }
            self.assertEqual(
                by_candidate[first.candidate_id]["stages"]["proposal"],
                "completed",
            )
            self.assertEqual(
                by_candidate[second.candidate_id]["stages"]["proposal"],
                "running",
            )

    def test_failed_candidate_without_stage_event_has_one_stage_projection(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = "run:failed-stage-invariant"
            start_native_run(director, task=task(), run_id=run_id)
            candidate = spawn_native_candidate(director, run_id)
            director.fail_candidate(run_id, candidate.candidate_id, "训练失败")

            stages, _ = self._candidate_stage_views(
                director, run_id, candidate.candidate_id
            )

        self.assertEqual(
            stages,
            {
                "proposal": "completed",
                "candidate": "completed",
                "training": "failed",
                "evaluation": "skipped",
                "judge": "skipped",
                "decision": "skipped",
            },
        )
        self.assertFalse({"pending", "running"}.intersection(stages.values()))

    def test_failed_candidate_preserves_scoped_failed_stage_in_both_views(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = "run:scoped-failed-stage-invariant"
            start_native_run(director, task=task(), run_id=run_id)
            candidate = spawn_native_candidate(director, run_id)
            director.record_evolution_stage(
                run_id,
                generation=0,
                proposal_id=candidate.proposal_id,
                candidate_id=candidate.candidate_id,
                stage="evaluation",
                status="failed",
                public_error="评测失败",
            )
            director.fail_candidate(run_id, candidate.candidate_id, "评测失败")

            stages, _ = self._candidate_stage_views(
                director, run_id, candidate.candidate_id
            )

        self.assertEqual(stages["evaluation"], "failed")
        self.assertEqual(stages["training"], "completed")
        self.assertEqual(stages["judge"], "skipped")
        self.assertEqual(stages["decision"], "skipped")
        self.assertFalse({"pending", "running"}.intersection(stages.values()))

    def test_duplicate_candidate_has_one_skipped_stage_projection(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = "run:duplicate-stage-invariant"
            start_native_run(director, task=task(), run_id=run_id)
            original = spawn_native_candidate(director, run_id)
            duplicate = spawn_native_candidate(director, run_id)
            director.mark_candidate_duplicate(
                run_id,
                duplicate.candidate_id,
                original.candidate_id,
            )

            stages, _ = self._candidate_stage_views(
                director, run_id, duplicate.candidate_id
            )

        self.assertEqual(
            {name: stages[name] for name in ("training", "evaluation", "judge", "decision")},
            {
                "training": "skipped",
                "evaluation": "skipped",
                "judge": "skipped",
                "decision": "skipped",
            },
        )

    def test_proposal_scoped_stage_is_identical_in_both_views(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = "run:legacy-proposal-stage-invariant"
            start_native_run(director, task=task(), run_id=run_id)
            candidate = spawn_native_candidate(director, run_id)
            director.record_evolution_stage(
                run_id,
                generation=0,
                proposal_id=candidate.proposal_id,
                stage="training",
                status="started",
            )

            stages, _ = self._candidate_stage_views(
                director, run_id, candidate.candidate_id
            )

        self.assertEqual(stages["training"], "running")

    def test_run_summary_exposes_all_frozen_runtime_bindings(self) -> None:
        configured = native_manifest(task_id="summary-bindings", metadata={
            "strategy_digest": "1" * 64, "prediction_model_digest": "2" * 64,
            "evaluator_digest": "3" * 64, "policy_model_digest": "4" * 64,
            "judge_model_digest": "5" * 64, "split_manifest_digest": "6" * 64,
        })
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            start_native_run(director, task=configured, run_id="run:summary-bindings")

            summary = run_summary(director.state("run:summary-bindings"))

        for field, expected in (
            ("strategy_digest", "1" * 64),
            ("prediction_model_digest", "2" * 64),
            ("evaluator_digest", "3" * 64),
            ("policy_model_digest", "4" * 64),
            ("judge_model_digest", "5" * 64),
            ("split_manifest_digest", "6" * 64),
        ):
            self.assertEqual(summary[field], expected)

    def test_new_candidate_without_worker_events_keeps_evaluation_unknown(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            run_id = "run:waiting-worker-events"
            start_native_run(director, task=task(), run_id=run_id)
            candidate = spawn_native_candidate(director, run_id)
            stages, _ = self._candidate_stage_views(director, run_id, candidate.candidate_id)
            progress = _projection_json(director.state(run_id))["execution_progress"]
        self.assertEqual(stages["proposal"], "completed")
        self.assertEqual(stages["candidate"], "completed")
        self.assertEqual(stages["evaluation"], "pending")
        self.assertIsNone(progress["overall_progress_percent"])


if __name__ == "__main__":
    unittest.main()
