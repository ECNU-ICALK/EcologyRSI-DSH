from __future__ import annotations

import unittest

from ecologyrsi_dsh import (
    Evaluation,
    EventLedger,
    EvolutionDirector,
    FakeDSHAdapter,
    TaskManifest,
)
from ecologyrsi_dsh.knowledge.algorithms import resolve_predictor_adoption
from ecologyrsi_dsh.knowledge.research_iteration import ResearchIteration
from ecologyrsi_dsh.knowledge.retrieval import retrieve_generation_knowledge
from ecologyrsi_dsh.reporting import rounds, run_summary, training_assets
from ecologyrsi_dsh.server import _projection_json


def task() -> TaskManifest:
    return TaskManifest(
        task_id="stage-projection-task",
        objective="predict soil water",
        domain_pack="crop-soil-water@toy",
        visible_datasets=("generated-toy-series@1",),
        budget={"max_candidates": 2, "max_generations": 2},
        seed=11,
    )


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

    def test_round_projects_bounded_research_iteration_summary(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = "run:research-round-summary"
            director.start_evolution(task(), run_id=run_id)
            state = director.state(run_id)
            snapshot = retrieve_generation_knowledge(state)
            ledger.append(
                run_id,
                "GenerationKnowledgeRetrieved",
                {"knowledge_snapshot": snapshot.to_dict()},
            )
            plan = {
                "prediction_model": {"id": "toy-rolling-water@1"},
                "algorithm_blueprint": {
                    "schema_version": "ecologyrsi-dsh.algorithm-blueprint/1",
                    "pipeline_id": "toy-rolling-water@1",
                    "operator_ids": ["toy.history_window@1"],
                    "parameter_names": ["alpha"],
                    "evidence_refs": ["knowledge:predictor", "knowledge:paper"],
                    "rationale": "not needed in the compact blueprint projection",
                },
                "algorithm_synthesis": {
                    "schema_version": "ecologyrsi-dsh.algorithm-synthesis/1",
                    "pipeline_id": "toy-rolling-water@1",
                    "evidence_refs": ["knowledge:paper"],
                    "parameter_focus": ["alpha"],
                    "rationale": "Translate frozen evidence into bounded alpha search.",
                },
                "research": [
                    {
                        "title": "Bounded smoothing study",
                        "source": "example journal",
                        "finding": "Recent observations can improve short-horizon forecasts.",
                        "relevance": "Supports testing alpha within the registered bounds.",
                    }
                ],
                "private_reasoning": "must not be projected",
            }
            adoption = resolve_predictor_adoption(state.task_manifest, plan)
            iteration = ResearchIteration(
                run_id=run_id,
                generation=0,
                status="model_generated",
                plan=plan,
                prediction_model_adoption=adoption.to_dict(),
                knowledge_snapshot_digest=snapshot.snapshot_digest,
                model_id="policy-main",
            )
            director.record_research_iteration(iteration)

            projected = rounds(director.state(run_id))[0]["research_iteration"]

            self.assertEqual(projected["status"], "model_generated")
            self.assertEqual(projected["iteration_digest"], iteration.iteration_digest)
            self.assertEqual(
                projected["knowledge_snapshot_digest"], snapshot.snapshot_digest
            )
            self.assertEqual(projected["model_id"], "policy-main")
            self.assertEqual(
                projected["predictor_adoption"]["adopted_id"],
                "toy-rolling-water@1",
            )
            self.assertEqual(
                projected["algorithm_synthesis"]["evidence_refs"],
                ["knowledge:paper"],
            )
            self.assertEqual(
                projected["analysis_summary"],
                {
                    "schema_version": "ecologyrsi-dsh.literature-analysis-summary/1",
                    "status": "completed",
                    "summary": "Translate frozen evidence into bounded alpha search.",
                    "evidence_refs": ["knowledge:paper"],
                    "key_findings": [
                        {
                            "title": "Bounded smoothing study",
                            "source": "example journal",
                            "finding": (
                                "Recent observations can improve short-horizon forecasts."
                            ),
                            "relevance": (
                                "Supports testing alpha within the registered bounds."
                            ),
                        }
                    ],
                    "source": "model_research_plan",
                },
            )
            self.assertEqual(
                projected["final_plan"],
                {
                    "schema_version": "ecologyrsi-dsh.final-implementation-plan/1",
                    "status": "ready_for_host_compilation",
                    "predictor_id": "toy-rolling-water@1",
                    "pipeline_id": "toy-rolling-water@1",
                    "operator_ids": ["toy.history_window@1"],
                    "parameter_names": ["alpha"],
                    "parameter_focus": ["alpha"],
                    "rationale": (
                        "Translate frozen evidence into bounded alpha search."
                    ),
                    "implementation_mode": "registered_host_components_only",
                    "validation_sequence": [
                        "compile_registered_ir",
                        "training_fit_smoke_test",
                        "training_feedback_evaluation",
                        "independent_model_review",
                    ],
                },
            )
            self.assertNotIn("private_reasoning", str(projected))

    def test_replay_rejects_unknown_stage_and_status(self) -> None:
        invalid_cases = (
            (stage_payload(stage="deployment"), "unknown evolution stage"),
            (stage_payload(status="queued"), "unknown evolution stage status"),
        )
        for payload, expected_error in invalid_cases:
            with self.subTest(payload=payload):
                with EventLedger() as ledger:
                    director = EvolutionDirector(ledger, FakeDSHAdapter())
                    director.start_evolution(task(), run_id="run:invalid-stage")
                    ledger.append(
                        "run:invalid-stage",
                        "EvolutionStageRecorded",
                        payload,
                    )
                    with self.assertRaisesRegex(ValueError, expected_error):
                        director.replay("run:invalid-stage")

    def test_latest_recorded_status_overrides_inference_without_mutating_core(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            director.start_evolution(task(), run_id="run:recorded-stage")
            candidate = director.propose_and_spawn("run:recorded-stage")
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
            receipts = training_assets(replayed)[0]["episode"]["event_receipts"]
            self.assertIn("EvolutionStageRecorded", {item["kind"] for item in receipts})

    def test_started_stage_is_visible_before_proposal_exists(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            director.start_evolution(task(), run_id="run:stage-only")
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
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = "run:cancelled-stage-fence"
            director.start_evolution(task(), run_id=run_id)
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
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = "run:gateway-retry-heartbeat"
            director.start_evolution(task(), run_id=run_id)
            ledger.append(
                run_id,
                "GatewayRetryScheduled",
                {
                    "generation": 0,
                    "retry_at": "2026-08-19T12:00:15+00:00",
                    "delay_seconds": 15.0,
                    "attempt": 2,
                    "error_code": "gateway_response_error",
                    "reason": "网关请求已完成本地重试，等待服务端队列恢复后再次调用",
                },
            )
            progress = _projection_json(director.replay(run_id))["execution_progress"]
            self.assertEqual(progress["phase"], "gateway_retry")
            self.assertTrue(progress["retry_wait"]["waiting"])
            self.assertEqual(progress["retry_wait"]["attempt"], 2)

    def test_unscoped_stage_does_not_override_a_completed_candidate_stage(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            director.start_evolution(task(), run_id="run:scoped-stage")
            candidate = director.propose_and_spawn("run:scoped-stage")
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
        """Legacy proposal-only events must not bleed into sibling rows."""

        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            director.start_evolution(task(), run_id="run:proposal-scoped-stage")
            first = director.propose_and_spawn("run:proposal-scoped-stage")
            second = director.propose_and_spawn("run:proposal-scoped-stage")
            director.record_evolution_stage(
                "run:proposal-scoped-stage",
                generation=0,
                proposal_id=first.proposal_id,
                candidate_id=first.candidate_id,
                stage="proposal",
                status="completed",
            )
            # Simulate an old event shape that carried only the second
            # proposal id.  The first candidate must retain its own status.
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
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = "run:failed-stage-invariant"
            director.start_evolution(task(), run_id=run_id)
            candidate = director.propose_and_spawn(run_id)
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
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = "run:scoped-failed-stage-invariant"
            director.start_evolution(task(), run_id=run_id)
            candidate = director.propose_and_spawn(run_id)
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
            director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=2))
            run_id = "run:duplicate-stage-invariant"
            director.start_evolution(task(), run_id=run_id)
            original = director.propose_and_spawn(run_id)
            duplicate = director.propose_and_spawn(run_id)
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

    def test_legacy_proposal_only_stage_is_scoped_identically(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            run_id = "run:legacy-proposal-stage-invariant"
            director.start_evolution(task(), run_id=run_id)
            candidate = director.propose_and_spawn(run_id)
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
        configured = TaskManifest(
            task_id="summary-bindings",
            objective="audit frozen runtime bindings",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget=1,
            seed=13,
            metadata={
                "strategy_id": "parameter_sweep@1",
                "strategy_digest": "strategy-digest",
                "prediction_model_id": "toy-rolling-water@1",
                "prediction_model_digest": "prediction-digest",
                "evaluator_id": "toy_time_forward@1",
                "evaluator_digest": "evaluator-digest",
                "policy_model_id": "host_parameter_generator@1",
                "policy_model_digest": "policy-digest",
                "judge_model_id": "rule_judge@1",
                "judge_model_digest": "judge-digest",
                "dataset_digest": "dataset-digest",
                "split_manifest_digest": "split-digest",
            },
        )
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            director.start_evolution(configured, run_id="run:summary-bindings")

            summary = run_summary(director.state("run:summary-bindings"))

        for field, expected in (
            ("strategy_digest", "strategy-digest"),
            ("prediction_model_digest", "prediction-digest"),
            ("evaluator_digest", "evaluator-digest"),
            ("policy_model_digest", "policy-digest"),
            ("judge_model_digest", "judge-digest"),
            ("split_manifest_digest", "split-digest"),
        ):
            self.assertEqual(summary[field], expected)

    def test_legacy_run_without_stage_events_keeps_inferred_statuses(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            director.start_evolution(task(), run_id="run:legacy-stage-fallback")
            candidate = director.propose_and_spawn("run:legacy-stage-fallback")
            director.evaluate_and_decide(
                Evaluation(
                    evaluation_id="evaluation:legacy-stage-fallback",
                    run_id="run:legacy-stage-fallback",
                    candidate_id=candidate.candidate_id,
                    score=0.8,
                    passed=True,
                )
            )

            projected = rounds(director.replay("run:legacy-stage-fallback"))
            self.assertEqual(
                projected[0]["stages"],
                {
                    "proposal": "completed",
                    "candidate": "completed",
                    "training": "pending",
                    "evaluation": "completed",
                    "judge": "not_recorded",
                    "decision": "completed",
                },
            )


if __name__ == "__main__":
    unittest.main()
