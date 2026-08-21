from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ecologyrsi_dsh import (
    CandidateStatus,
    Evaluation,
    EventLedger,
    EvolutionDirector,
    FakeDSHAdapter,
    ModelArtifact,
    Proposal,
    StrategyRouterDSHAdapter,
    TaskManifest,
)
from ecologyrsi_dsh.api.generation_execution import (
    _apply_candidate_judge,
    _generation_judges_should_retry,
    complete_if_budget_exhausted,
    execute_generation,
)
from ecologyrsi_dsh.evaluators.registry import EvaluationBundle
from ecologyrsi_dsh.integrations.model_gateway import (
    GatewayConfigurationError,
    GatewayResponseError,
)
from ecologyrsi_dsh.knowledge.algorithm_smoke import AlgorithmSmokeError
from ecologyrsi_dsh.server import EvolutionRequestHandler


class _UnavailableJudge:
    def apply_judge(self, task, proposal, bundle):
        raise TimeoutError("judge timed out")


class _UnavailableProposalGateway:
    def catalog(self):
        return []

    def propose(self, model_id, context, allowed_parameters):
        raise GatewayResponseError("policy model request failed: TimeoutError")


class _RepeatingAdapter:
    def open_session(self, run, task):
        return f"repeat:{run.run_id}"

    def propose(
        self,
        run,
        task,
        session_id,
        *,
        parent_candidate_id=None,
        parent_context=None,
        interventions=None,
        batch_context=None,
    ):
        return Proposal(
            proposal_id=f"proposal:{run.run_id}:{run.generation}",
            run_id=run.run_id,
            generation=run.generation,
            title="same executable parameters",
            changes={"alpha": 0.4, "window": 5, "water_threshold": 0.4},
            parent_candidate_id=parent_candidate_id,
        )

    def close_session(self, session_id):
        return None


class _DeterministicEvaluator:
    def evaluate_scientific(
        self, task, candidate, proposal, *, on_training_complete=None
    ):
        if on_training_complete is not None:
            on_training_complete()
        artifact = ModelArtifact(
            artifact_id=f"artifact:{candidate.candidate_id}",
            run_id=candidate.run_id,
            candidate_id=candidate.candidate_id,
            model_id="test-predictor",
            dataset_digest="dataset-digest",
            training_partition="training_fit",
            training_rows=10,
            parameters=proposal.changes,
        )
        evaluation = Evaluation(
            evaluation_id=f"evaluation:{candidate.candidate_id}",
            run_id=candidate.run_id,
            candidate_id=candidate.candidate_id,
            score=-0.1,
            passed=False,
            metrics={"scientific_pass": False, "skill_score": -0.1},
            partition="validation",
            evaluator_digest="fixed-evaluator",
            artifact_digest=artifact.digest,
        )
        return EvaluationBundle(artifact=artifact, evaluation=evaluation)

    def apply_judge(self, task, proposal, bundle):
        data = bundle.evaluation.to_dict()
        data["metrics"] = {
            **dict(bundle.evaluation.metrics),
            "judge_model_id": "rule-judge",
            "judge_accepted": False,
            "judge_guidance": "scientific gate failed",
            "judge_parameter_override": {},
        }
        return EvaluationBundle(
            artifact=bundle.artifact,
            evaluation=Evaluation.from_dict(data),
        )


class _PassingDeterministicEvaluator(_DeterministicEvaluator):
    def evaluate_scientific(
        self, task, candidate, proposal, *, on_training_complete=None
    ):
        bundle = super().evaluate_scientific(
            task,
            candidate,
            proposal,
            on_training_complete=on_training_complete,
        )
        data = bundle.evaluation.to_dict()
        data.update(
            {
                "score": 0.5,
                "passed": True,
                "metrics": {"scientific_pass": True, "skill_score": 0.5},
            }
        )
        return EvaluationBundle(
            artifact=bundle.artifact,
            evaluation=Evaluation.from_dict(data),
        )

    def apply_judge(self, task, proposal, bundle):
        data = bundle.evaluation.to_dict()
        data["passed"] = True
        data["metrics"] = {
            **dict(bundle.evaluation.metrics),
            "judge_model_id": "rule-judge",
            "judge_accepted": True,
            "judge_guidance": "accepted",
            "judge_parameter_override": {},
        }
        return EvaluationBundle(
            artifact=bundle.artifact,
            evaluation=Evaluation.from_dict(data),
        )


class _FailingScientificEvaluator:
    def evaluate_scientific(
        self, task, candidate, proposal, *, on_training_complete=None
    ):
        raise RuntimeError("scientific evaluator failed")


class _RecoveringJudgmentEvaluator(_DeterministicEvaluator):
    def __init__(self) -> None:
        self.available = False
        self.scientific_calls = 0
        self.judge_calls = 0

    def evaluate_scientific(
        self, task, candidate, proposal, *, on_training_complete=None
    ):
        self.scientific_calls += 1
        return super().evaluate_scientific(
            task,
            candidate,
            proposal,
            on_training_complete=on_training_complete,
        )

    def apply_judge(self, task, proposal, bundle):
        self.judge_calls += 1
        if not self.available:
            raise TimeoutError("judge timed out")
        return super().apply_judge(task, proposal, bundle)


class _PermanentUnavailableJudgmentEvaluator(_DeterministicEvaluator):
    def apply_judge(self, task, proposal, bundle):
        raise GatewayResponseError(
            "judge authentication failed",
            retryable=False,
            status_code=401,
            error_code="authentication_failed",
        )


class _InvalidJudgmentEvaluator(_DeterministicEvaluator):
    def apply_judge(self, task, proposal, bundle):
        raise ValueError("judge result violates its deterministic contract")


class _MisconfiguredJudgmentEvaluator(_DeterministicEvaluator):
    def apply_judge(self, task, proposal, bundle):
        raise GatewayConfigurationError("judge model does not allow the judge role")


class _OverwritingJudgmentEvaluator(_DeterministicEvaluator):
    def apply_judge(self, task, proposal, bundle):
        data = bundle.evaluation.to_dict()
        data.update(
            {
                "passed": True,
                "metrics": {
                    "scientific_pass": True,
                    "judge_model_id": "custom-judge",
                    "judge_accepted": True,
                    "judge_guidance": "accept",
                    "judge_parameter_override": {},
                },
            }
        )
        return EvaluationBundle(
            artifact=bundle.artifact,
            evaluation=Evaluation.from_dict(data),
        )


class JudgePersistenceTests(unittest.TestCase):
    def test_autonomous_request_gets_effective_default_budget(self) -> None:
        handler = object.__new__(EvolutionRequestHandler)
        handler._bind_runtime_task = lambda manifest, **_kwargs: manifest
        manifest = handler._task_from_request(
            {
                "dataset_id": "generated-toy-series@1",
                "autonomous_mode": True,
                "strategy_model_id": "strategy-model",
                "review_model_id": "review-model",
            }
        )
        self.assertEqual(
            manifest.budget,
            {
                "max_generations": 5,
                "candidates_per_generation": 4,
                "max_candidates": 20,
            },
        )
        self.assertEqual(manifest.metadata["budget_profile"], "autonomous_default_5x4")

    def test_judge_error_preserves_scientific_score_and_fails_closed(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            task = TaskManifest(
                task_id="judge-persistence",
                objective="preserve scientific evidence",
                domain_pack="crop-soil-water@toy",
                visible_datasets=("generated-toy-series@1",),
                budget={"max_candidates": 1},
                metadata={"judge_model_id": "remote-judge"},
            )
            run_id = director.start_evolution(
                task, run_id="run:judge-persistence"
            ).run.run_id
            candidate = director.propose_and_spawn(run_id)
            state = director.state(run_id)
            proposal = state.proposal(candidate.proposal_id)
            artifact = ModelArtifact(
                artifact_id="artifact:judge-persistence",
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                model_id="test-predictor",
                dataset_digest="dataset-digest",
                training_partition="training_fit",
                training_rows=10,
            )
            director.record_artifact(artifact)
            scientific = Evaluation(
                evaluation_id="evaluation:judge-persistence",
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                score=0.123,
                passed=True,
                metrics={"skill_score": 0.123},
                partition="training_feedback",
                evaluator_digest="fixed-evaluator",
                artifact_digest=artifact.digest,
            )
            director.record_evaluation(scientific)
            endpoint = SimpleNamespace(
                server=SimpleNamespace(
                    director=director,
                    evaluators=_UnavailableJudge(),
                )
            )

            _apply_candidate_judge(
                endpoint,
                director.state(run_id),
                proposal,
                artifact,
                scientific,
            )

            state = director.state(run_id)
            persisted = state.evaluation_for(candidate.candidate_id)
            self.assertIsNotNone(persisted)
            self.assertEqual(persisted.score, 0.123)
            self.assertTrue(persisted.metrics["scientific_pass"])
            self.assertEqual(persisted.metrics["judge_status"], "unavailable")
            self.assertEqual(persisted.metrics["judge_failure_class"], "transient")
            self.assertEqual(persisted.metrics["judge_error_code"], "TimeoutError")
            self.assertFalse(persisted.passed)
            self.assertIs(
                state.candidate(candidate.candidate_id).status,
                CandidateStatus.EVALUATED,
            )
            self.assertNotIn("CandidateFailed", {event.kind for event in state.events})
            self.assertEqual(
                [
                    event.payload["status"]
                    for event in state.events
                    if event.kind == "EvolutionStageRecorded"
                    and event.payload["stage"] == "judge"
                ],
                ["started", "failed"],
            )

    def test_custom_judge_cannot_replace_scientific_gate_result(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            task = TaskManifest(
                task_id="custom-judge-scientific-pass",
                objective="preserve the scientific gate result",
                domain_pack="crop-soil-water@toy",
                visible_datasets=("generated-toy-series@1",),
                budget={
                    "max_candidates": 1,
                    "max_generations": 1,
                    "candidates_per_generation": 1,
                },
                metadata={"judge_model_id": "custom-judge"},
            )
            run_id = director.start_evolution(
                task, run_id="run:custom-judge-scientific-pass"
            ).run.run_id
            endpoint = SimpleNamespace(
                server=SimpleNamespace(
                    director=director,
                    evaluators=_OverwritingJudgmentEvaluator(),
                )
            )

            state = execute_generation(endpoint, run_id)

            self.assertEqual(state.run.status.value, "completed")
            self.assertEqual(len(state.evaluations), 1)
            persisted = state.evaluations[0]
            self.assertIs(persisted.metrics["scientific_pass"], False)
            self.assertIs(persisted.metrics["judge_accepted"], True)
            self.assertFalse(persisted.passed)
            self.assertIs(state.candidates[0].status, CandidateStatus.REJECTED)

    def test_remote_proposal_failure_falls_back_without_losing_prior_slots(self) -> None:
        with EventLedger() as ledger:
            adapter = StrategyRouterDSHAdapter(
                gateway=_UnavailableProposalGateway(),  # type: ignore[arg-type]
                max_proposals=3,
            )
            director = EvolutionDirector(ledger, adapter)
            task = TaskManifest(
                task_id="proposal-slot-fallback",
                objective="保留已生成槽位并回退失败的远程提案",
                domain_pack="greenhouse-climate@1",
                visible_datasets=("generated-toy-series@1",),
                budget={
                    "max_candidates": 3,
                    "max_generations": 1,
                    "candidates_per_generation": 3,
                },
                metadata={
                    "strategy_id": "dsh_authenticated@1",
                    "policy_model_id": "policy-main",
                    "domain": "greenhouse",
                    "prediction_model_id": "greenhouse-rolling-residual@1",
                    "evaluator_id": "greenhouse_time_forward@1",
                    "prediction_model_digest": "predictor-digest",
                    "evaluator_digest": "fixed-evaluator",
                },
            )
            run_id = director.start_evolution(
                task, run_id="run:proposal-slot-fallback"
            ).run.run_id
            endpoint = SimpleNamespace(
                server=SimpleNamespace(
                    director=director,
                    evaluators=_DeterministicEvaluator(),
                )
            )

            state = execute_generation(endpoint, run_id)

            self.assertEqual(state.run.status.value, "completed")
            self.assertEqual(len(state.candidates), 3)
            ordered = sorted(state.candidates, key=lambda item: item.slot_index)
            self.assertEqual([item.slot_index for item in ordered], [0, 1, 2])
            self.assertEqual(
                state.proposal(ordered[0].proposal_id).changes,
                {"blend": 1.0, "window": 24, "bias_scale": 0.0},
            )
            self.assertEqual(
                state.proposal(ordered[1].proposal_id).changes,
                {"blend": 0.93, "window": 25, "bias_scale": 0.0},
            )
            fallback = state.proposal(ordered[2].proposal_id)
            self.assertEqual(
                fallback.changes,
                {"blend": 0.87, "window": 24, "bias_scale": 0.0},
            )
            self.assertEqual(fallback.metadata["host_fallback"]["slot_index"], 2)
            self.assertNotIn("RunFailed", {event.kind for event in state.events})
            fallback_stages = [
                event.payload
                for event in state.events
                if event.kind == "EvolutionStageRecorded"
                and event.payload.get("proposal_id") == fallback.proposal_id
                and event.payload.get("stage") == "proposal"
            ]
            self.assertEqual(
                [(item["status"], item["attempt"]) for item in fallback_stages],
                [("failed", 1), ("completed", 2)],
            )

    def test_equivalent_candidate_is_deduplicated_across_generations(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, _RepeatingAdapter())
            task = TaskManifest(
                task_id="cross-generation-deduplication",
                objective="do not repeat identical work",
                domain_pack="crop-soil-water@toy",
                visible_datasets=("generated-toy-series@1",),
                budget={
                    "max_candidates": 2,
                    "max_generations": 2,
                    "candidates_per_generation": 1,
                },
                metadata={
                    "prediction_model_digest": "predictor-digest",
                    "evaluator_digest": "fixed-evaluator",
                },
            )
            run_id = director.start_evolution(
                task, run_id="run:cross-generation-deduplication"
            ).run.run_id
            endpoint = SimpleNamespace(
                server=SimpleNamespace(
                    director=director,
                    evaluators=_PassingDeterministicEvaluator(),
                )
            )

            first_state = execute_generation(endpoint, run_id)
            self.assertEqual(first_state.run.generation, 1)
            evaluation_events_before = tuple(
                event
                for event in first_state.events
                if event.kind == "EvaluationRecorded"
            )
            advanced_events_before = tuple(
                event
                for event in first_state.events
                if event.kind == "GenerationAdvanced"
            )
            final_state = execute_generation(endpoint, run_id)

            ordered = sorted(final_state.candidates, key=lambda item: item.generation)
            self.assertIs(ordered[0].status, CandidateStatus.PROMOTED)
            self.assertIs(ordered[1].status, CandidateStatus.DUPLICATE)
            self.assertEqual(len(final_state.artifacts), 1)
            self.assertEqual(len(final_state.evaluations), 1)
            self.assertEqual(final_state.run.status.value, "completed")
            self.assertEqual(final_state.run.generation, 1)
            self.assertEqual(final_state.run.best_candidate_id, ordered[0].candidate_id)
            self.assertNotIn("RunFailed", {event.kind for event in final_state.events})
            evaluation_events_after = tuple(
                event
                for event in final_state.events
                if event.kind == "EvaluationRecorded"
            )
            self.assertEqual(evaluation_events_after, evaluation_events_before)
            advanced = [
                event
                for event in final_state.events
                if event.kind == "GenerationAdvanced"
            ]
            self.assertEqual(tuple(advanced), advanced_events_before)
            self.assertEqual([event.payload["generation"] for event in advanced], [1])
            self.assertIsNotNone(final_state.analysis_for(1))
            self.assertTrue(
                any(
                    event.kind == "GenerationChampionSelected"
                    and event.payload["generation"] == 1
                    for event in final_state.events
                )
            )
            completion = next(
                event
                for event in reversed(final_state.events)
                if event.kind == "RunCompleted"
            )
            self.assertEqual(
                completion.payload["termination_reason"],
                "search_space_converged_all_candidates_duplicate",
            )
            duplicate_event = next(
                event
                for event in final_state.events
                if event.kind == "CandidateMarkedDuplicate"
            )
            self.assertEqual(
                duplicate_event.payload["duplicate_of_candidate_id"],
                ordered[0].candidate_id,
            )

    def test_duplicate_convergence_completion_recovers_without_advancing(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, _RepeatingAdapter())
            task = TaskManifest(
                task_id="duplicate-convergence-recovery",
                objective="recover a missing convergence completion event",
                domain_pack="crop-soil-water@toy",
                visible_datasets=("generated-toy-series@1",),
                budget={
                    "max_candidates": 3,
                    "max_generations": 3,
                    "candidates_per_generation": 1,
                },
                metadata={
                    "prediction_model_digest": "predictor-digest",
                    "evaluator_digest": "fixed-evaluator",
                },
            )
            run_id = director.start_evolution(
                task, run_id="run:duplicate-convergence-recovery"
            ).run.run_id
            endpoint = SimpleNamespace(
                server=SimpleNamespace(
                    director=director,
                    evaluators=_PassingDeterministicEvaluator(),
                )
            )

            execute_generation(endpoint, run_id)
            baseline = director.state(run_id)
            baseline_evaluations = tuple(
                event for event in baseline.events if event.kind == "EvaluationRecorded"
            )
            baseline_advances = tuple(
                event for event in baseline.events if event.kind == "GenerationAdvanced"
            )
            with patch.object(
                director,
                "complete_run",
                side_effect=RuntimeError("simulated completion write crash"),
            ), self.assertRaisesRegex(RuntimeError, "simulated completion"):
                execute_generation(endpoint, run_id)

            interrupted = director.state(run_id)
            self.assertEqual(interrupted.run.status.value, "running")
            self.assertEqual(interrupted.run.generation, 1)
            self.assertEqual(
                tuple(
                    event
                    for event in interrupted.events
                    if event.kind == "EvaluationRecorded"
                ),
                baseline_evaluations,
            )
            self.assertEqual(
                tuple(
                    event
                    for event in interrupted.events
                    if event.kind == "GenerationAdvanced"
                ),
                baseline_advances,
            )
            self.assertEqual(
                [event.kind for event in interrupted.events][-1],
                "GenerationChampionSelected",
            )
            self.assertIsNotNone(interrupted.analysis_for(1))

            recovered = complete_if_budget_exhausted(endpoint, run_id)

            self.assertEqual(recovered.run.status.value, "completed")
            self.assertEqual(recovered.run.generation, 1)
            self.assertEqual(
                tuple(
                    event
                    for event in recovered.events
                    if event.kind == "EvaluationRecorded"
                ),
                baseline_evaluations,
            )
            self.assertEqual(
                tuple(
                    event
                    for event in recovered.events
                    if event.kind == "GenerationAdvanced"
                ),
                baseline_advances,
            )
            completion = next(
                event
                for event in reversed(recovered.events)
                if event.kind == "RunCompleted"
            )
            self.assertEqual(
                completion.payload["termination_reason"],
                "search_space_converged_all_candidates_duplicate",
            )

    def test_generation_without_any_evaluation_fails_without_advancing(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            task = TaskManifest(
                task_id="generation-evidence-missing",
                objective="stop when all candidate evaluations fail",
                domain_pack="crop-soil-water@toy",
                visible_datasets=("generated-toy-series@1",),
                budget={
                    "max_candidates": 2,
                    "max_generations": 2,
                    "candidates_per_generation": 2,
                },
            )
            run_id = director.start_evolution(
                task, run_id="run:generation-evidence-missing"
            ).run.run_id
            endpoint = SimpleNamespace(
                server=SimpleNamespace(
                    director=director,
                    evaluators=_FailingScientificEvaluator(),
                )
            )

            state = execute_generation(endpoint, run_id)

            self.assertEqual(state.run.status.value, "failed")
            self.assertEqual(state.run.generation, 0)
            self.assertEqual(len(state.candidates), 2)
            self.assertTrue(
                all(item.status is CandidateStatus.FAILED for item in state.candidates)
            )
            self.assertEqual(state.evaluations, ())
            self.assertNotIn(
                "GenerationAdvanced", {event.kind for event in state.events}
            )
            failure = next(
                event for event in reversed(state.events) if event.kind == "RunFailed"
            )
            self.assertIn("generation_evidence_missing", failure.payload["reason"])
            batch_failure = next(
                event
                for event in reversed(state.events)
                if event.kind == "EvolutionStageRecorded"
                and event.payload["stage"] == "decision"
                and event.payload["status"] == "failed"
                and event.payload["candidate_id"] is None
            )
            self.assertIn(
                "generation_evidence_missing", batch_failure.payload["public_error"]
            )

    def test_failed_algorithm_attempt_is_not_counted_as_generation_evidence(
        self,
    ) -> None:
        def failing_smoke(spec, task, proposal, *, attempt, failure_feedback):
            del spec, task, proposal, attempt, failure_feedback
            raise AlgorithmSmokeError(
                "smoke_registered_tool_failed",
                "registered smoke tool failed",
            )

        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            task = TaskManifest(
                task_id="failed-algorithm-is-not-evaluation",
                objective="do not advance an epoch without scientific evaluation",
                domain_pack="crop-soil-water@toy",
                visible_datasets=("generated-toy-series@1",),
                budget={
                    "max_candidates": 1,
                    "max_generations": 2,
                    "candidates_per_generation": 1,
                },
            )
            run_id = director.start_evolution(
                task, run_id="run:failed-algorithm-is-not-evaluation"
            ).run.run_id
            endpoint = SimpleNamespace(
                server=SimpleNamespace(
                    director=director,
                    evaluators=_DeterministicEvaluator(),
                    algorithm_smoke_runner=failing_smoke,
                )
            )

            state = execute_generation(endpoint, run_id)

            self.assertEqual(state.run.status.value, "failed")
            self.assertEqual(state.run.generation, 0)
            self.assertEqual(state.evaluations, ())
            self.assertTrue(
                any(item.status == "failed" for item in state.algorithm_attempts)
            )
            self.assertNotIn(
                "GenerationAdvanced", {event.kind for event in state.events}
            )
            failure = next(
                event for event in reversed(state.events) if event.kind == "RunFailed"
            )
            self.assertIn("generation_evidence_missing", failure.payload["reason"])

    def test_all_temporarily_unavailable_judges_retry_without_retraining(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            task = TaskManifest(
                task_id="generation-judges-recover",
                objective="retry temporarily unavailable judges",
                domain_pack="crop-soil-water@toy",
                visible_datasets=("generated-toy-series@1",),
                budget={
                    "max_candidates": 2,
                    "max_generations": 1,
                    "candidates_per_generation": 2,
                },
                metadata={"judge_model_id": "remote-judge"},
            )
            run_id = director.start_evolution(
                task, run_id="run:generation-judges-recover"
            ).run.run_id
            evaluator = _RecoveringJudgmentEvaluator()
            endpoint = SimpleNamespace(
                server=SimpleNamespace(director=director, evaluators=evaluator)
            )

            with self.assertRaises(GatewayResponseError) as raised:
                execute_generation(endpoint, run_id)

            self.assertTrue(raised.exception.retryable)
            self.assertEqual(
                raised.exception.error_code, "generation_judges_unavailable"
            )
            waiting = director.state(run_id)
            self.assertEqual(waiting.run.status.value, "running")
            self.assertEqual(waiting.run.generation, 0)
            self.assertEqual(evaluator.scientific_calls, 2)
            self.assertEqual(evaluator.judge_calls, 2)
            self.assertEqual(len(waiting.promotions), 0)
            self.assertNotIn("RunFailed", {event.kind for event in waiting.events})
            self.assertTrue(
                all(
                    evaluation.metrics["judge_failure_class"] == "transient"
                    for evaluation in waiting.evaluations
                )
            )

            evaluator.available = True
            recovered = execute_generation(endpoint, run_id)

            self.assertEqual(recovered.run.status.value, "completed")
            self.assertEqual(recovered.run.generation, 1)
            self.assertEqual(evaluator.scientific_calls, 2)
            self.assertEqual(evaluator.judge_calls, 4)
            self.assertEqual(len(recovered.promotions), 2)
            self.assertTrue(
                all(
                    evaluation.metrics["judge_status"] == "completed"
                    for evaluation in recovered.evaluations
                )
            )

    def test_all_permanently_unavailable_judges_fail_without_advancing(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            task = TaskManifest(
                task_id="generation-judges-unavailable",
                objective="stop when all judges are unavailable",
                domain_pack="crop-soil-water@toy",
                visible_datasets=("generated-toy-series@1",),
                budget={
                    "max_candidates": 2,
                    "max_generations": 2,
                    "candidates_per_generation": 2,
                },
                metadata={"judge_model_id": "remote-judge"},
            )
            run_id = director.start_evolution(
                task, run_id="run:generation-judges-unavailable"
            ).run.run_id
            endpoint = SimpleNamespace(
                server=SimpleNamespace(
                    director=director,
                    evaluators=_PermanentUnavailableJudgmentEvaluator(),
                )
            )

            state = execute_generation(endpoint, run_id)

            self.assertEqual(state.run.status.value, "failed")
            self.assertEqual(state.run.generation, 0)
            self.assertEqual(len(state.evaluations), 2)
            self.assertTrue(
                all(
                    evaluation.metrics["judge_status"] == "unavailable"
                    for evaluation in state.evaluations
                )
            )
            self.assertTrue(
                all(
                    evaluation.metrics["judge_failure_class"] == "permanent"
                    for evaluation in state.evaluations
                )
            )
            self.assertNotIn(
                "GenerationAdvanced", {event.kind for event in state.events}
            )
            failure = next(
                event for event in reversed(state.events) if event.kind == "RunFailed"
            )
            self.assertIn("generation_judges_unavailable", failure.payload["reason"])
            batch_failure = next(
                event
                for event in reversed(state.events)
                if event.kind == "EvolutionStageRecorded"
                and event.payload["stage"] == "decision"
                and event.payload["status"] == "failed"
                and event.payload["candidate_id"] is None
            )
            self.assertIn(
                "generation_judges_unavailable",
                batch_failure.payload["public_error"],
            )

    def test_invalid_judge_result_is_permanent_and_does_not_retry(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            task = TaskManifest(
                task_id="generation-judge-invalid-result",
                objective="stop on a deterministic judge contract error",
                domain_pack="crop-soil-water@toy",
                visible_datasets=("generated-toy-series@1",),
                budget={
                    "max_candidates": 1,
                    "max_generations": 1,
                    "candidates_per_generation": 1,
                },
            )
            run_id = director.start_evolution(
                task, run_id="run:generation-judge-invalid-result"
            ).run.run_id
            endpoint = SimpleNamespace(
                server=SimpleNamespace(
                    director=director,
                    evaluators=_InvalidJudgmentEvaluator(),
                )
            )

            state = execute_generation(endpoint, run_id)

            self.assertEqual(state.run.status.value, "failed")
            self.assertEqual(len(state.evaluations), 1)
            self.assertEqual(
                state.evaluations[0].metrics["judge_failure_class"], "permanent"
            )
            self.assertFalse(_generation_judges_should_retry(state, 0))

    def test_judge_configuration_error_remains_permanent(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            task = TaskManifest(
                task_id="generation-judge-misconfigured",
                objective="fail a permanently misconfigured judge",
                domain_pack="crop-soil-water@toy",
                visible_datasets=("generated-toy-series@1",),
                budget={
                    "max_candidates": 1,
                    "max_generations": 1,
                    "candidates_per_generation": 1,
                },
                metadata={"judge_model_id": "proposal-only-model"},
            )
            run_id = director.start_evolution(
                task, run_id="run:generation-judge-misconfigured"
            ).run.run_id
            endpoint = SimpleNamespace(
                server=SimpleNamespace(
                    director=director,
                    evaluators=_MisconfiguredJudgmentEvaluator(),
                )
            )

            state = execute_generation(endpoint, run_id)

            self.assertEqual(state.run.status.value, "failed")
            self.assertEqual(len(state.evaluations), 1)
            self.assertEqual(
                state.evaluations[0].metrics["judge_failure_class"], "permanent"
            )
            self.assertEqual(
                state.evaluations[0].metrics["judge_error_code"],
                "GatewayConfigurationError",
            )
            self.assertIn("RunFailed", {event.kind for event in state.events})


if __name__ == "__main__":
    unittest.main()
