from __future__ import annotations

import unittest
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

from ecologyrsi_dsh.api.generation_execution import _evaluate_candidate
from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import (
    Evaluation,
    ExpertConsultation,
    ExpertConsultationAnswer,
    ModelArtifact,
    Proposal,
    TaskManifest,
    canonical_json,
    digest,
)
from ecologyrsi_dsh.data.registry import DatasetRegistry
from ecologyrsi_dsh.evaluators.registry import EvaluatorRegistry
from ecologyrsi_dsh.evolution.analysis import (
    CROSS_GENERATION_EXPERIENCE_MAX_BYTES,
    GenerationAnalysis,
    build_cross_generation_experience,
)
from ecologyrsi_dsh.evolution.batches import (
    ResearchResponseContractError,
    finalize_generation_batch,
    start_generation_batch,
)
from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter
from ecologyrsi_dsh.integrations.model_gateway import GatewayResponseError
from ecologyrsi_dsh.knowledge.algorithm_ir import registered_algorithm_blueprint
from ecologyrsi_dsh.knowledge.algorithms import (
    AlgorithmAttempt,
    compile_algorithm_spec,
    resolve_predictor_adoption,
)
from ecologyrsi_dsh.knowledge.research_iteration import ResearchIteration
from ecologyrsi_dsh.knowledge.retrieval import retrieve_generation_knowledge
from ecologyrsi_dsh.server import _projection_json


class _ResearchGateway:
    def __init__(
        self,
        *,
        fail_research: bool = False,
        requested_id: str = "unregistered-transformer@9",
    ) -> None:
        self.fail_research = fail_research
        self.requested_id = requested_id
        self.research_contexts: list[dict] = []
        self.proposal_contexts: list[dict] = []

    def catalog(self) -> list[dict]:
        return []

    def research_plan(self, _model_id: str, context: dict) -> dict:
        if self.fail_research:
            raise AssertionError("a frozen generation must not be researched again")
        self.research_contexts.append(context)
        return {
            "status": "model_generated",
            "prediction_model": {"id": self.requested_id},
            "strategy": {"id": "autonomous_model@1"},
            "research": [
                {
                    "title": "bounded residual forecasting",
                    "url": "https://example.invalid/research",
                }
            ],
            "generation_marker": context["generation"],
        }

    def propose(
        self,
        _model_id: str,
        context: dict,
        _allowed_parameters: dict,
    ) -> dict:
        self.proposal_contexts.append(context)
        if "blend" in _allowed_parameters:
            parameters = {"blend": 0.72, "window": 10, "bias_scale": 0.9}
        elif "history_steps" in _allowed_parameters:
            parameters = {
                "history_steps": 6,
                "ridge_alpha": 0.05,
                "residual_scale": 0.75,
            }
        else:
            parameters = {
                "alpha": 0.45,
                "window": 5,
                "water_threshold": 0.4,
            }
        return {
            "parameters": parameters,
            "rationale": "use aggregate failure feedback",
        }


class _PartialResearchGateway(_ResearchGateway):
    def research_plan(self, _model_id: str, context: dict) -> dict:
        self.research_contexts.append(context)
        return {
            "strategy": {
                "id": "autonomous_model@1",
                "rationale": "retain the active predictor and refine search guidance",
            }
        }


class _GuardrailOverrideResearchGateway(_PartialResearchGateway):
    def research_plan(self, model_id: str, context: dict) -> dict:
        result = super().research_plan(model_id, context)
        result["historical_parameter_guardrails"] = {
            "policy": "untrusted_model_override",
            "protected_parameter_evidence": [],
        }
        return result


class _ContractFailureResearchGateway(_ResearchGateway):
    def research_plan(self, _model_id: str, context: dict) -> dict:
        self.research_contexts.append(context)
        raise GatewayResponseError("invalid research response contract")


class _DegradedResearchGateway(_ResearchGateway):
    def research_plan(self, _model_id: str, context: dict) -> dict:
        self.research_contexts.append(context)
        return {
            "algorithm_synthesis_degradation": {
                "schema_version": (
                    "ecologyrsi-dsh.algorithm-synthesis-degradation/1"
                ),
                "reason_code": "no_compatible_executable_evidence",
                "rationale": "no compatible evidence in this frozen snapshot",
            }
        }


class _BlockingResearchGateway(_ResearchGateway):
    def __init__(self) -> None:
        super().__init__()
        self.request_started = Event()
        self.release_request = Event()

    def research_plan(self, model_id: str, context: dict) -> dict:
        self.request_started.set()
        if not self.release_request.wait(timeout=5):
            raise TimeoutError("test research request was not released")
        return super().research_plan(model_id, context)


class _ExpertConsultingResearchGateway(_ResearchGateway):
    def research_plan(self, model_id: str, context: dict) -> dict:
        result = super().research_plan(model_id, context)
        result["expert_consultation"] = {
            "uncertainty_type": "scientific_assumption",
            "question": "Which conservative soil-water bound should be retained?",
            "context": "Aggregate long-horizon skill remains unstable.",
            "fallback_assumption": "Retain the registered conservative default.",
            "requested_expertise": ["soil physics"],
            "options": ["retain default", "lower the bound"],
            "confidence": 0.41,
            "non_blocking": True,
        }
        return result


def _task(*, candidates_per_generation: int) -> TaskManifest:
    return TaskManifest(
        task_id="research-iteration",
        objective="improve soil-water forecasts from generation feedback",
        domain_pack="crop-soil-water@toy",
        visible_datasets=("generated-toy-series@1",),
        budget={
            "max_candidates": candidates_per_generation * 2,
            "max_generations": 2,
            "candidates_per_generation": candidates_per_generation,
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


def _greenhouse_switch_task() -> TaskManifest:
    dataset_id = "agc_cucumber_2018"
    ridge_schemas = {
        "history_steps": {"type": "integer", "minimum": 1, "maximum": 12},
        "ridge_alpha": {"type": "number", "minimum": 0.0001, "maximum": 1.0},
        "residual_scale": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }
    rolling_schemas = {
        "blend": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "window": {"type": "integer", "minimum": 1, "maximum": 48},
        "bias_scale": {"type": "number", "minimum": 0.0, "maximum": 2.0},
    }
    return TaskManifest(
        task_id="research-predictor-switch",
        objective="adopt a compatible registered predictor",
        domain_pack="greenhouse_environment@1",
        visible_datasets=(dataset_id,),
        budget={
            "max_candidates": 1,
            "max_generations": 1,
            "candidates_per_generation": 1,
        },
        metadata={
            "domain": "greenhouse",
            "autonomous_mode": True,
            "strategy_id": "autonomous_model@1",
            "strategy_model_id": "research-model",
            "prediction_model_id": "greenhouse-exogenous-ridge@1",
            "prediction_model_digest": "ridge-digest",
            "evaluator_id": "greenhouse_time_forward@1",
            "dataset_digest": "d" * 64,
            "split_manifest_digest": "s" * 64,
            "knowledge_online_enabled": False,
            "runtime_component_catalog": {
                "prediction_models": [
                    {
                        "id": "greenhouse-exogenous-ridge@1",
                        "dataset_ids": [dataset_id],
                        "configuration_digest": "ridge-digest",
                        "parameter_schemas": ridge_schemas,
                    },
                    {
                        "id": "greenhouse-rolling-residual@1",
                        "dataset_ids": [dataset_id],
                        "configuration_digest": "rolling-digest",
                        "parameter_schemas": rolling_schemas,
                    },
                ],
                "evaluators": [],
                "selected_prediction_model_id": "greenhouse-exogenous-ridge@1",
                "selected_evaluator_id": "greenhouse_time_forward@1",
            },
        },
    )


def _historical_experience_task(
    *,
    evaluator_id: str = "greenhouse_multihorizon_time_forward@1",
    evaluator_digest: str = "e" * 64,
    split_manifest_digest: str = "s" * 64,
    samples_per_update: int | None = None,
) -> TaskManifest:
    data = _greenhouse_switch_task().to_dict()
    runtime_catalog = dict(data["metadata"]["runtime_component_catalog"])
    runtime_catalog["selected_evaluator_id"] = evaluator_id
    data["metadata"] = {
        **data["metadata"],
        "evaluator_id": evaluator_id,
        "evaluator_digest": evaluator_digest,
        "dataset_digest": "d" * 64,
        "split_manifest_digest": split_manifest_digest,
        "evaluation_partition": "training_feedback",
        "scientific_scope": "historical_replay_prediction_non_causal",
        "runtime_component_catalog": runtime_catalog,
    }
    if samples_per_update is not None:
        data["metadata"]["samples_per_update"] = samples_per_update
    return TaskManifest.from_dict(data)


def _historical_state(
    task: TaskManifest,
    run_id: str,
    analyses: tuple[GenerationAnalysis, ...],
    *,
    created_seq: int,
) -> SimpleNamespace:
    events = [SimpleNamespace(kind="RunCreated", seq=created_seq, payload={})]
    events.extend(
        SimpleNamespace(
            kind="GenerationAnalyzed",
            seq=created_seq + index + 1,
            payload={"analysis": analysis.to_dict()},
        )
        for index, analysis in enumerate(analyses)
    )
    return SimpleNamespace(
        task_manifest=task,
        run=SimpleNamespace(run_id=run_id),
        events=tuple(events),
        generation_analyses=analyses,
        research_iterations=(),
        proposals=(),
        algorithm_attempts=(),
    )


def _guardrail_analysis(
    run_id: str,
    generation: int,
    *,
    cohort_digest: str,
    cell_sample_count: int,
    skill_score: float = 0.12,
    constraint_violations: int = 0,
    value: float = 0.0,
    population_count: int = 1000,
    population_digest: str = "f" * 64,
    selected_count: int | None = None,
    window_offset: int | None = None,
) -> GenerationAnalysis:
    candidate_id = f"candidate:guardrail:{run_id}:{generation}"
    resolved_selected_count = selected_count or cell_sample_count
    resolved_window_offset = (
        generation * resolved_selected_count
        if window_offset is None
        else window_offset
    )
    return GenerationAnalysis(
        run_id=run_id,
        generation=generation,
        candidate_count=1,
        eligible_count=1 if constraint_violations == 0 and skill_score >= 0 else 0,
        outcome=(
            "promoted"
            if constraint_violations == 0 and skill_score >= 0
            else "no_eligible_candidate"
        ),
        selected_candidate_id=(
            candidate_id
            if constraint_violations == 0 and skill_score >= 0
            else None
        ),
        champion_candidate_id=candidate_id,
        ranking=(
            {
                "candidate_id": candidate_id,
                "score": skill_score,
                "constraint_violations": constraint_violations,
                "evaluation_cohort_digest": cohort_digest,
                "evaluation_cohort_window": {
                    "evaluation_cohort_digest": cohort_digest,
                    "feedback_update_cohort_digest": cohort_digest,
                    "schema_version": "ecologyrsi-dsh.feedback-update-cohort/1",
                    "selection_policy": (
                        "target_horizon_interleaved_rotating_window@1"
                    ),
                    "population_count": population_count,
                    "population_digest": population_digest,
                    "selected_count": resolved_selected_count,
                    "window_offset": resolved_window_offset,
                },
                "parameters": {
                    "co2_concentration_1h_residual_scale": value,
                    "residual_scale": 0.5,
                },
                "target_skill_scores": [
                    {
                        "target": "co2_concentration",
                        "horizon_hours": 1,
                        "skill_score": skill_score,
                        "n": cell_sample_count,
                        "observed": "must-not-enter-guardrails",
                    }
                ],
            },
        ),
    )


class ResearchIterationTests(unittest.TestCase):
    def test_model_plan_contract_failure_is_retryable_boundary(self) -> None:
        gateway = _ResearchGateway()
        with EventLedger() as ledger:
            adapter = StrategyRouterDSHAdapter(gateway=gateway)  # type: ignore[arg-type]
            director = EvolutionDirector(ledger, adapter)
            run_id = "run:invalid-model-research-plan"
            director.start_evolution(
                _task(candidates_per_generation=1),
                run_id=run_id,
            )
            invalid_result = {
                "status": "model_generated",
                "model_id": "research-model",
                "plan": {"implementation": {"code": "not executable"}},
            }

            with (
                patch.object(
                    adapter,
                    "research_iteration",
                    return_value=invalid_result,
                ),
                self.assertRaises(ResearchResponseContractError) as caught,
            ):
                start_generation_batch(director, run_id)

            self.assertIsInstance(caught.exception.__cause__, ValueError)
            state = director.state(run_id)
            stages = [
                event
                for event in state.events
                if event.kind == "EvolutionStageRecorded"
                and event.payload.get("stage") == "research"
            ]
            self.assertEqual(
                [event.payload["status"] for event in stages],
                ["started", "failed"],
            )
            self.assertFalse(
                any(event.kind == "GenerationResearchIterated" for event in state.events)
            )

    def test_host_research_persistence_value_error_is_not_retryable_contract(self) -> None:
        gateway = _ResearchGateway()
        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            run_id = "run:host-research-state-error"
            director.start_evolution(
                _task(candidates_per_generation=1),
                run_id=run_id,
            )
            host_error = ValueError("host state is invalid")

            with (
                patch.object(
                    director,
                    "record_research_iteration",
                    side_effect=host_error,
                ),
                self.assertRaises(ValueError) as caught,
            ):
                start_generation_batch(director, run_id)

            self.assertIs(caught.exception, host_error)
            self.assertNotIsInstance(
                caught.exception,
                ResearchResponseContractError,
            )

    def test_remote_research_request_is_visible_while_gateway_is_blocked(
        self,
    ) -> None:
        gateway = _BlockingResearchGateway()
        errors: list[BaseException] = []
        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            run_id = "run:research-progress"
            director.start_evolution(
                _task(candidates_per_generation=1),
                run_id=run_id,
            )

            def run_batch() -> None:
                try:
                    start_generation_batch(director, run_id)
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            worker = Thread(target=run_batch, daemon=True)
            worker.start()
            try:
                self.assertTrue(gateway.request_started.wait(timeout=2))
                active = _projection_json(director.state(run_id))
                stage_events = [
                    event
                    for event in director.state(run_id).events
                    if event.kind == "EvolutionStageRecorded"
                ]

                self.assertEqual(active["execution_progress"]["phase"], "research")
                self.assertEqual(
                    active["execution_progress"]["current_stage"], "research"
                )
                self.assertEqual(
                    active["execution_diagnostics"]["remote_strategy_calls"], 1
                )
                self.assertEqual(
                    active["execution_diagnostics"]["remote_strategy_successes"],
                    0,
                )
                self.assertEqual(
                    active["execution_diagnostics"]["remote_research_calls"], 1
                )
                self.assertEqual(
                    active["execution_diagnostics"]["remote_research_successes"],
                    0,
                )
                self.assertEqual(
                    active["execution_diagnostics"]["remote_strategy_status"],
                    "running",
                )
                self.assertEqual(active["rounds"][0]["timing"]["status"], "running")
                self.assertEqual(
                    active["rounds"][0]["timing"]["started_at"],
                    stage_events[0].created_at,
                )
                self.assertEqual(len(stage_events), 1)
                self.assertEqual(stage_events[0].payload["stage"], "research")
                self.assertEqual(stage_events[0].payload["status"], "started")
                self.assertNotIn("context", stage_events[0].payload)
                self.assertNotIn("prompt", str(stage_events[0].payload).casefold())
            finally:
                gateway.release_request.set()
                worker.join(timeout=5)

            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            state = director.state(run_id)
            knowledge = next(
                event
                for event in state.events
                if event.kind == "GenerationKnowledgeRetrieved"
            )
            research_started = next(
                event
                for event in state.events
                if event.kind == "EvolutionStageRecorded"
                and event.payload.get("stage") == "research"
                and event.payload.get("status") == "started"
            )
            research_iterated = next(
                event
                for event in state.events
                if event.kind == "GenerationResearchIterated"
            )
            research_completed = next(
                event
                for event in state.events
                if event.kind == "EvolutionStageRecorded"
                and event.payload.get("stage") == "research"
                and event.payload.get("status") == "completed"
            )
            batch_started = next(
                event
                for event in state.events
                if event.kind == "GenerationBatchStarted"
            )
            self.assertLess(
                knowledge.seq,
                research_started.seq,
            )
            self.assertLess(research_started.seq, research_iterated.seq)
            self.assertLess(research_iterated.seq, research_completed.seq)
            self.assertLess(research_completed.seq, batch_started.seq)

    def test_strict_research_failure_records_redacted_failed_stage(self) -> None:
        task_data = _task(candidates_per_generation=1).to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "remote_fallback_policy": "fail_run",
        }
        gateway = _ContractFailureResearchGateway()
        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            run_id = "run:research-progress-failed"
            director.start_evolution(
                TaskManifest.from_dict(task_data),
                run_id=run_id,
            )

            with self.assertRaises(GatewayResponseError):
                start_generation_batch(director, run_id)

            state = director.state(run_id)
            stages = [
                event
                for event in state.events
                if event.kind == "EvolutionStageRecorded"
                and event.payload.get("stage") == "research"
            ]
            projection = _projection_json(state)

            self.assertEqual(
                [event.payload["status"] for event in stages],
                ["started", "failed"],
            )
            self.assertEqual(projection["execution_progress"]["phase"], "research")
            self.assertEqual(projection["failed_stage"]["stage"], "research")
            self.assertNotIn("invalid research response", str(stages[-1].payload))
            self.assertFalse(
                any(
                    event.kind
                    in {"GenerationResearchIterated", "GenerationBatchStarted"}
                    for event in state.events
                )
            )

    def test_replay_closes_started_research_after_iteration_was_persisted(
        self,
    ) -> None:
        gateway = _ResearchGateway(fail_research=True)
        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            run_id = "run:research-progress-recovery"
            director.start_evolution(
                _task(candidates_per_generation=1),
                run_id=run_id,
            )
            state = director.state(run_id)
            snapshot = retrieve_generation_knowledge(state)
            ledger.append(
                run_id,
                "GenerationKnowledgeRetrieved",
                {"knowledge_snapshot": snapshot.to_dict()},
            )
            director.record_evolution_stage(
                run_id,
                generation=0,
                stage="research",
                status="started",
            )
            plan = {"prediction_model": {"id": "toy-rolling-water@1"}}
            director.record_research_iteration(
                ResearchIteration(
                    run_id=run_id,
                    generation=0,
                    status="model_generated",
                    plan=plan,
                    prediction_model_adoption=resolve_predictor_adoption(
                        state.task_manifest,
                        plan,
                    ).to_dict(),
                    knowledge_snapshot_digest=snapshot.snapshot_digest,
                    model_id="research-model",
                )
            )

            start_generation_batch(director, run_id)
            recovered = director.state(run_id)

            self.assertEqual(gateway.research_contexts, [])
            self.assertEqual(
                [
                    event.payload["status"]
                    for event in recovered.events
                    if event.kind == "EvolutionStageRecorded"
                    and event.payload.get("stage") == "research"
                ],
                ["started", "completed"],
            )
            self.assertIsNotNone(recovered.batch_for(0))

    def test_strict_research_contract_failure_blocks_without_host_fallback(
        self,
    ) -> None:
        task_data = _task(candidates_per_generation=1).to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "remote_fallback_policy": "fail_run",
        }
        task = TaskManifest.from_dict(task_data)
        gateway = _ContractFailureResearchGateway()
        adapter = StrategyRouterDSHAdapter(gateway=gateway)  # type: ignore[arg-type]
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, adapter)
            director.start_evolution(task, run_id="run:strict-contract-failure")
            state = director.state("run:strict-contract-failure")
            knowledge = retrieve_generation_knowledge(state).proposal_context()

            with self.assertRaisesRegex(
                GatewayResponseError,
                "invalid research response contract",
            ):
                adapter.research_iteration(
                    run=state.run,
                    task=state.task_manifest,
                    previous_generation_analysis=None,
                    knowledge_snapshot=knowledge,
                    previous_knowledge_assessment=None,
                    current_plan={
                        "prediction_model": {"id": "toy-rolling-water@1"}
                    },
                )

    def test_research_contract_fallback_is_explicit_when_allowed(self) -> None:
        task_data = _task(candidates_per_generation=1).to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "remote_fallback_policy": "record_and_continue",
        }
        task = TaskManifest.from_dict(task_data)
        gateway = _ContractFailureResearchGateway()
        adapter = StrategyRouterDSHAdapter(gateway=gateway)  # type: ignore[arg-type]
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, adapter)
            director.start_evolution(task, run_id="run:audited-contract-fallback")
            state = director.state("run:audited-contract-fallback")
            result = adapter.research_iteration(
                run=state.run,
                task=state.task_manifest,
                previous_generation_analysis=None,
                knowledge_snapshot=(
                    retrieve_generation_knowledge(state).proposal_context()
                ),
                previous_knowledge_assessment=None,
                current_plan={
                    "prediction_model": {"id": "toy-rolling-water@1"}
                },
            )

        diagnostics = result["plan"]["fallback_diagnostics"]
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(
            diagnostics["algorithm_synthesis_status"],
            "not_refreshed_due_gateway_error",
        )
        self.assertFalse(diagnostics["retryable"])

    def test_explicit_synthesis_degradation_does_not_inherit_old_synthesis(
        self,
    ) -> None:
        gateway = _DegradedResearchGateway()
        adapter = StrategyRouterDSHAdapter(gateway=gateway)  # type: ignore[arg-type]
        task = _task(candidates_per_generation=1)
        current_plan = {
            "prediction_model": {"id": "toy-rolling-water@1"},
            "algorithm_blueprint": {
                **registered_algorithm_blueprint("toy-rolling-water@1"),
                "evidence_refs": ["knowledge:old"],
            },
            "algorithm_synthesis": {
                "schema_version": "ecologyrsi-dsh.algorithm-synthesis/1",
                "pipeline_id": "toy-rolling-water@1",
                "evidence_refs": ["knowledge:old"],
                "parameter_focus": ["alpha"],
                "rationale": "old synthesis",
            },
        }
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, adapter)
            director.start_evolution(task, run_id="run:explicit-degradation")
            state = director.state("run:explicit-degradation")
            result = adapter.research_iteration(
                run=state.run,
                task=state.task_manifest,
                previous_generation_analysis=None,
                knowledge_snapshot=(
                    retrieve_generation_knowledge(state).proposal_context()
                ),
                previous_knowledge_assessment=None,
                current_plan=current_plan,
            )

        self.assertIn("algorithm_synthesis_degradation", result["plan"])
        self.assertNotIn("algorithm_blueprint", result["plan"])
        self.assertNotIn("algorithm_synthesis", result["plan"])
        self.assertEqual(
            result["plan"]["prediction_model"],
            current_plan["prediction_model"],
        )

    def test_current_workflow_rejects_predictor_switch_without_blueprint(self) -> None:
        task_data = _greenhouse_switch_task().to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "model_workflow": "research_compile_evolve@1",
        }
        task = TaskManifest.from_dict(task_data)

        adoption = resolve_predictor_adoption(
            task,
            {"prediction_model": {"id": "greenhouse-rolling-residual@1"}},
        )

        self.assertEqual(adoption.status, "research_only")
        self.assertEqual(adoption.adopted_id, "greenhouse-exogenous-ridge@1")
        self.assertEqual(
            adoption.reason,
            "requested_predictor_requires_registered_algorithm_blueprint",
        )

    def test_partial_iteration_inherits_predictor_and_uses_its_parameter_schema(
        self,
    ) -> None:
        gateway = _PartialResearchGateway()
        adapter = StrategyRouterDSHAdapter(gateway=gateway)  # type: ignore[arg-type]
        task = _greenhouse_switch_task()
        current_plan = {
            "prediction_model": {"id": "greenhouse-rolling-residual@1"},
            "strategy": {"id": "autonomous_model@1"},
        }
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, adapter)
            director.start_evolution(task, run_id="run:partial-plan-inheritance")
            state = director.state("run:partial-plan-inheritance")
            knowledge = retrieve_generation_knowledge(state).proposal_context()

            result = adapter.research_iteration(
                run=state.run,
                task=state.task_manifest,
                previous_generation_analysis=None,
                knowledge_snapshot=knowledge,
                previous_knowledge_assessment=None,
                current_plan=current_plan,
            )

        self.assertEqual(
            set(gateway.research_contexts[0]["allowed_parameter_schemas"]),
            {"blend", "window", "bias_scale"},
        )
        self.assertEqual(
            result["plan"]["prediction_model"]["id"],
            "greenhouse-rolling-residual@1",
        )
        self.assertEqual(
            resolve_predictor_adoption(task, result["plan"]).adopted_id,
            "greenhouse-rolling-residual@1",
        )

    def test_generation_shares_one_iteration_and_restart_replays_it(self) -> None:
        gateway = _ResearchGateway()
        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            task = _task(candidates_per_generation=2)
            director.start_evolution(task, run_id="run:shared-research")

            first_batch = start_generation_batch(director, "run:shared-research")
            second_batch = start_generation_batch(director, "run:shared-research")
            proposals = [
                director.request_proposal(
                    "run:shared-research",
                    generation_batch=first_batch,
                    slot_index=slot,
                    consume_interventions=False,
                )
                for slot in range(2)
            ]
            state = director.state("run:shared-research")
            iteration = state.research_iteration_for(0)

            self.assertEqual(first_batch, second_batch)
            self.assertIsNotNone(iteration)
            assert iteration is not None
            self.assertEqual(len(gateway.research_contexts), 1)
            self.assertEqual(
                {item.metadata["research_iteration_digest"] for item in proposals},
                {iteration.iteration_digest},
            )
            self.assertTrue(
                all(
                    item.metadata["prediction_model_adoption"]["status"]
                    == "research_only"
                    for item in proposals
                )
            )
            self.assertEqual(
                compile_algorithm_spec(task, proposals[0], state.knowledge_for(0)).adapter_id,
                "toy-rolling-water@1",
            )
            self.assertEqual(
                sum(
                    event.kind == "GenerationResearchIterated"
                    for event in state.events
                ),
                1,
            )

            replay_gateway = _ResearchGateway(fail_research=True)
            replayed = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(  # type: ignore[arg-type]
                    gateway=replay_gateway
                ),
            )
            replayed_batch = start_generation_batch(
                replayed, "run:shared-research"
            )

            self.assertEqual(replayed_batch, first_batch)
            self.assertEqual(replay_gateway.research_contexts, [])
            self.assertEqual(
                replayed.state("run:shared-research")
                .research_iteration_for(0)
                .iteration_digest,  # type: ignore[union-attr]
                iteration.iteration_digest,
            )

    def test_next_research_iteration_applies_answers_and_records_new_question(
        self,
    ) -> None:
        gateway = _ExpertConsultingResearchGateway()
        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            run_id = "run:expert-research-context"
            director.start_evolution(
                _task(candidates_per_generation=1),
                run_id=run_id,
            )
            for consultation_id, question in (
                ("consultation:pending", "Should the long horizon stay enabled?"),
                ("consultation:answered", "Which threshold is defensible?"),
            ):
                director.record_expert_consultation(
                    ExpertConsultation(
                        consultation_id=consultation_id,
                        run_id=run_id,
                        generation=0,
                        uncertainty_type="scientific_assumption",
                        question=question,
                        context="Only aggregate validation evidence is included.",
                        fallback_assumption="Retain the registered default.",
                        requested_expertise=("soil physics",),
                        options=("retain default", "use conservative bound"),
                        confidence=0.45,
                        requested_by_model_id="research-model",
                        created_at=(
                            "2026-08-20T01:00:00+00:00"
                            if consultation_id.endswith("pending")
                            else "2026-08-20T01:01:00+00:00"
                        ),
                    )
                )
            director.answer_expert_consultation(
                ExpertConsultationAnswer(
                    answer_id="answer:threshold",
                    run_id=run_id,
                    consultation_id="consultation:answered",
                    answer="Retain the default until sensitivity evidence is stable.",
                    answered_by="domain-expert",
                    selected_option="retain default",
                    effective_generation=1,
                    created_at="2026-08-20T02:00:00+00:00",
                )
            )
            ledger.append(run_id, "GenerationAdvanced", {"generation": 1})

            start_generation_batch(director, run_id)
            state = director.state(run_id)
            iteration = state.research_iteration_for(1)
            assert iteration is not None

            collaboration = gateway.research_contexts[0]["expert_collaboration"]
            self.assertEqual(
                [
                    item["consultation_id"]
                    for item in collaboration["pending_consultations"]
                ],
                ["consultation:pending"],
            )
            self.assertEqual(
                [item["answer_id"] for item in collaboration["available_answers"]],
                ["answer:threshold"],
            )
            self.assertNotIn("answered_by", collaboration["available_answers"][0])
            self.assertTrue(
                collaboration["policy"]["continue_with_fallback_when_unanswered"]
            )
            self.assertEqual(
                iteration.pending_consultation_ids,
                ("consultation:pending",),
            )
            self.assertEqual(iteration.expert_answer_ids, ("answer:threshold",))
            self.assertEqual(
                state.answer_for_consultation(
                    "consultation:answered"
                ).applied_generation,  # type: ignore[union-attr]
                1,
            )
            new_consultations = [
                item
                for item in state.expert_consultations
                if item.generation == 1
            ]
            self.assertEqual(len(new_consultations), 1)
            self.assertTrue(new_consultations[0].non_blocking)
            self.assertEqual(
                new_consultations[0].requested_by_model_id,
                "research-model",
            )
            atomic_kinds = [
                event.kind
                for event in state.events
                if event.kind
                in {
                    "GenerationResearchIterated",
                    "ExpertConsultationApplied",
                    "ExpertConsultationRequested",
                }
            ][-3:]
            self.assertEqual(
                atomic_kinds,
                [
                    "GenerationResearchIterated",
                    "ExpertConsultationApplied",
                    "ExpertConsultationRequested",
                ],
            )

    def test_unavailable_research_does_not_consume_available_expert_answer(
        self,
    ) -> None:
        gateway = _ContractFailureResearchGateway()
        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            run_id = "run:expert-answer-unavailable"
            director.start_evolution(
                _task(candidates_per_generation=1),
                run_id=run_id,
            )
            director.record_expert_consultation(
                ExpertConsultation(
                    consultation_id="consultation:unavailable",
                    run_id=run_id,
                    generation=0,
                    uncertainty_type="model_selection",
                    question="Should the registered baseline be retained?",
                    context="The remote research gateway may be unavailable.",
                    fallback_assumption="Retain the registered baseline.",
                    requested_expertise=("ecological forecasting",),
                    options=("retain baseline",),
                    confidence=0.5,
                    requested_by_model_id="research-model",
                )
            )
            director.answer_expert_consultation(
                ExpertConsultationAnswer(
                    answer_id="answer:unavailable",
                    run_id=run_id,
                    consultation_id="consultation:unavailable",
                    answer="Retain the baseline.",
                    answered_by="domain-expert",
                    selected_option="retain baseline",
                    effective_generation=1,
                )
            )
            ledger.append(run_id, "GenerationAdvanced", {"generation": 1})

            start_generation_batch(director, run_id)
            state = director.state(run_id)
            iteration = state.research_iteration_for(1)
            assert iteration is not None

            self.assertEqual(iteration.status, "unavailable")
            self.assertEqual(iteration.expert_answer_ids, ())
            self.assertEqual(
                [item.answer_id for item in state.available_expert_answers(1)],
                ["answer:unavailable"],
            )
            self.assertFalse(
                any(
                    event.kind == "ExpertConsultationApplied"
                    for event in state.events
                )
            )

    def test_next_generation_research_receives_all_frozen_feedback(self) -> None:
        gateway = _ResearchGateway()
        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            director.start_evolution(
                _task(candidates_per_generation=1),
                run_id="run:research-feedback",
            )
            first_batch = start_generation_batch(
                director, "run:research-feedback"
            )
            proposal = director.request_proposal(
                "run:research-feedback",
                generation_batch=first_batch,
                slot_index=0,
            )
            candidate = director.spawn_candidate(
                "run:research-feedback", proposal, slot_index=0
            )
            compiled = compile_algorithm_spec(
                director.state("run:research-feedback").task_manifest,
                proposal,
                director.state("run:research-feedback").knowledge_for(0),
            )
            director.record_algorithm_attempt(
                AlgorithmAttempt(
                    run_id=candidate.run_id,
                    generation=0,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate.candidate_id,
                    phase="compile",
                    attempt=1,
                    status="passed",
                    algorithm_spec_digest=compiled.spec_digest,
                    algorithm_spec=compiled.to_dict(),
                )
            )
            director.record_algorithm_attempt(
                AlgorithmAttempt(
                    run_id=candidate.run_id,
                    generation=0,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate.candidate_id,
                    phase="debug",
                    attempt=1,
                    status="failed",
                    algorithm_spec_digest=compiled.spec_digest,
                    evidence={
                        "stage": "training_fit_smoke",
                        "failure_feedback": {
                            "retryable": True,
                            "exception_type": "TimeoutError",
                        },
                    },
                    failure_code="smoke_tool_timeout",
                    public_error="TimeoutError: bounded smoke timeout",
                )
            )
            director.record_algorithm_attempt(
                AlgorithmAttempt(
                    run_id=candidate.run_id,
                    generation=0,
                    proposal_id=proposal.proposal_id,
                    candidate_id=candidate.candidate_id,
                    phase="debug",
                    attempt=2,
                    status="passed",
                    algorithm_spec_digest=compiled.spec_digest,
                    evidence={"stage": "training_fit_smoke", "passed": True},
                )
            )
            artifact = director.record_artifact(
                ModelArtifact(
                    artifact_id="artifact:research-feedback",
                    run_id=candidate.run_id,
                    candidate_id=candidate.candidate_id,
                    model_id="toy-rolling-water@1",
                    dataset_digest=compiled.dataset_digest,
                    training_partition="training_fit",
                    training_rows=32,
                    parameters=proposal.changes,
                )
            )
            director.record_evaluation(
                Evaluation(
                    evaluation_id="evaluation:research-feedback",
                    run_id=candidate.run_id,
                    candidate_id=candidate.candidate_id,
                    score=-0.2,
                    passed=False,
                    partition="training_feedback",
                    evaluator_digest="toy-evaluator@feedback",
                    artifact_digest=artifact.digest,
                    metrics={
                        "scientific_pass": False,
                        "judge_status": "completed",
                        "judge_accepted": False,
                        "constraint_violations": 0,
                        "targets": [
                            {
                                "target": "soil_moisture",
                                "horizon_hours": 24,
                                "unit": "fraction",
                                "skill_score": -0.2,
                            }
                        ],
                        "horizons": [
                            {"horizon_hours": 168, "skill_score": -0.3}
                        ],
                        "sample_execution": {
                            "eligible_examples": 10,
                            "attempted_examples": 10,
                            "succeeded_examples": 8,
                            "failed_examples": 2,
                            "skipped_examples": 2,
                            "coverage": 0.8,
                            "minimum_coverage": 0.8,
                            "coverage_pass": True,
                            "failure_counts": {"tool_timeout": 2},
                        },
                        "sample_execution_records": [
                            {
                                "sample_id": "secret-row",
                                "observed": 0.7,
                                "predicted": 0.2,
                                "labels": ["never-forward"],
                            }
                        ],
                    },
                )
            )
            analysis = finalize_generation_batch(
                director, "run:research-feedback"
            )
            assessment = director.state(
                "run:research-feedback"
            ).knowledge_assessment_for(0)
            self.assertIsNotNone(assessment)
            assert assessment is not None

            director.advance_generation("run:research-feedback")
            start_generation_batch(director, "run:research-feedback")
            state = director.state("run:research-feedback")
            iteration = state.research_iteration_for(1)
            context = gateway.research_contexts[-1]

            self.assertEqual(len(gateway.research_contexts), 2)
            self.assertEqual(
                context["previous_generation_analysis"]["analysis_digest"],
                analysis.analysis_digest,
            )
            previous_analysis = context["previous_generation_analysis"]
            self.assertEqual(
                previous_analysis["algorithm_failures"][0]["failure_code"],
                "smoke_tool_timeout",
            )
            self.assertEqual(
                previous_analysis["sample_failures"][0]["failure_counts"],
                {"tool_timeout": 2},
            )
            self.assertEqual(
                previous_analysis["target_weaknesses"][0]["target"],
                "soil_moisture",
            )
            self.assertNotIn("secret-row", str(previous_analysis))
            self.assertNotIn("never-forward", str(previous_analysis))
            self.assertEqual(
                context["previous_knowledge_assessment"]["next_action"],
                assessment.next_action,
            )
            self.assertEqual(
                context["knowledge_snapshot"]["snapshot_digest"],
                state.knowledge_for(1).snapshot_digest,  # type: ignore[union-attr]
            )
            self.assertIsNotNone(iteration)
            assert iteration is not None
            self.assertEqual(iteration.source_analysis_digest, analysis.analysis_digest)
            self.assertEqual(
                iteration.source_assessment_digest,
                assessment.assessment_digest,
            )
            self.assertEqual(iteration.previous_next_action, assessment.next_action)

    def test_cross_generation_experience_replays_and_archives_resolved_issues(
        self,
    ) -> None:
        task_data = _task(candidates_per_generation=1).to_dict()
        task_data["budget"] = {
            "max_candidates": 3,
            "max_generations": 3,
            "candidates_per_generation": 1,
        }
        task = TaskManifest.from_dict(task_data)
        gateway = _PartialResearchGateway()
        run_id = "run:cross-generation-experience"
        blueprint = {
            **registered_algorithm_blueprint("toy-rolling-water@1"),
            "evidence_refs": ["knowledge:bounded-synthesis"],
            "rationale": "use one frozen bounded synthesis",
        }
        synthesis = {
            "schema_version": "ecologyrsi-dsh.algorithm-synthesis/1",
            "pipeline_id": "toy-rolling-water@1",
            "evidence_refs": ["knowledge:bounded-synthesis"],
            "parameter_focus": ["alpha"],
            "rationale": "test the registered smoothing direction",
        }
        plan_zero = {
            "prediction_model": {"id": "toy-rolling-water@1"},
            "algorithm_blueprint": blueprint,
            "algorithm_synthesis": synthesis,
            "strategy": {"id": "autonomous_model@1", "steps": ["diagnose"]},
        }
        plan_one = {
            **plan_zero,
            "strategy": {"id": "autonomous_model@1", "steps": ["repair"]},
        }
        analysis_zero = GenerationAnalysis(
            run_id=run_id,
            generation=0,
            candidate_count=1,
            eligible_count=0,
            outcome="no_eligible_candidate",
            ranking=(
                {
                    "candidate_id": "candidate:experience:0",
                    "score": -0.4,
                    "parameters": {
                        "alpha": 0.2,
                        "window": 3,
                        "water_threshold": 0.35,
                    },
                },
            ),
            target_weaknesses=(
                {
                    "target": "soil_moisture",
                    "horizon_hours": 24,
                    "unit": "fraction",
                    "median_skill_score": -0.2,
                    "evidence_count": 1,
                },
                {
                    "target": "co2_concentration",
                    "horizon_hours": 24,
                    "unit": "ppm",
                    "median_skill_score": -0.4,
                    "evidence_count": 1,
                },
            ),
            horizon_weaknesses=(
                {
                    "horizon_hours": 168,
                    "median_skill_score": -0.3,
                    "evidence_count": 1,
                },
            ),
            algorithm_failures=(
                {
                    "phase": "debug",
                    "stage": "training_fit_smoke",
                    "failure_code": "smoke_tool_timeout",
                    "retryable": True,
                    "error_type": "TimeoutError",
                    "attempt_count": 1,
                    "details": {"raw": "secret-row"},
                },
            ),
            sample_failures=(
                {
                    "failed": 2,
                    "failure_counts": {"tool_timeout": 2},
                    "repair_count": 2,
                    "recovered_examples": 1,
                    "repair_tool_outcomes": {
                        "bounded-projection-repair": {
                            "completed": 1,
                            "failed": 1,
                        }
                    },
                    "sample_execution_records": [
                        {"sample_id": "secret-row", "observed": 0.7}
                    ],
                },
            ),
        )
        analysis_one = GenerationAnalysis(
            run_id=run_id,
            generation=1,
            candidate_count=1,
            eligible_count=1,
            outcome="promoted",
            selected_candidate_id="candidate:experience:1",
            champion_candidate_id="candidate:experience:1",
            ranking=(
                {
                    "candidate_id": "candidate:experience:1",
                    "score": 0.1,
                    "parameters": {
                        "alpha": 0.35,
                        "window": 5,
                        "water_threshold": 0.4,
                    },
                },
            ),
            target_weaknesses=(
                {
                    "target": "soil_moisture",
                    "horizon_hours": 24,
                    "unit": "fraction",
                    "median_skill_score": 0.1,
                    "evidence_count": 1,
                },
                {
                    "target": "co2_concentration",
                    "horizon_hours": 24,
                    "unit": "ppm",
                    "median_skill_score": -0.1,
                    "evidence_count": 1,
                },
            ),
            horizon_weaknesses=(
                {
                    "horizon_hours": 168,
                    "median_skill_score": 0.05,
                    "evidence_count": 1,
                },
            ),
        )

        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            director.start_evolution(task, run_id=run_id)
            ledger.append(
                run_id,
                "ProposalSubmitted",
                {
                    "proposal": Proposal(
                        proposal_id="proposal:experience:0",
                        run_id=run_id,
                        generation=0,
                        title="generation zero",
                        changes={"alpha": 0.2, "window": 3, "water_threshold": 0.35},
                        metadata={"plan": plan_zero},
                    ).to_dict()
                },
            )
            ledger.append(
                run_id,
                "GenerationAnalyzed",
                {"analysis": analysis_zero.to_dict()},
            )
            ledger.append(run_id, "GenerationAdvanced", {"generation": 1})
            ledger.append(
                run_id,
                "ProposalSubmitted",
                {
                    "proposal": Proposal(
                        proposal_id="proposal:experience:1",
                        run_id=run_id,
                        generation=1,
                        title="generation one",
                        changes={"alpha": 0.35, "window": 5, "water_threshold": 0.4},
                        metadata={"plan": plan_one},
                    ).to_dict()
                },
            )
            ledger.append(
                run_id,
                "GenerationAnalyzed",
                {"analysis": analysis_one.to_dict()},
            )
            ledger.append(run_id, "GenerationAdvanced", {"generation": 2})

            before_replay = build_cross_generation_experience(
                director.state(run_id), 2
            )
            replayed = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            after_replay = build_cross_generation_experience(
                replayed.replay(run_id), 2
            )

            self.assertEqual(after_replay, before_replay)
            self.assertEqual(
                after_replay["window"]["included_generations"], [0, 1]
            )
            self.assertEqual(
                [item["improved"] for item in after_replay["generations"]],
                [False, False],
            )
            self.assertEqual(
                after_replay["generations"][1]["modifications"][
                    "algorithm_blueprint_status"
                ],
                "inherited",
            )
            self.assertEqual(
                after_replay["generations"][1]["modifications"][
                    "algorithm_synthesis_status"
                ],
                "inherited",
            )
            self.assertEqual(
                after_replay["generations"][1]["modifications"][
                    "algorithm_synthesis_parameter_focus"
                ],
                ["alpha"],
            )
            synthesis_effects = [
                item["algorithm_synthesis_effect"]
                for item in after_replay["generations"]
            ]
            self.assertEqual(
                [item["observed_improvement"] for item in synthesis_effects],
                [False, False],
            )
            self.assertEqual(
                synthesis_effects[1]["selection_interpretation"],
                "selected_without_comparable_score_baseline",
            )
            self.assertEqual(
                {item["synthesis_digest"] for item in synthesis_effects},
                {digest(synthesis)},
            )
            self.assertTrue(
                all(item["causal_attribution"] is False for item in synthesis_effects)
            )
            active = after_replay["active_unresolved"]
            archived = after_replay["resolved_archived"]
            self.assertTrue(
                any(
                    item["kind"] == "weak_target"
                    and item["identity"].get("target") == "co2_concentration"
                    and item["status"] == "active_unresolved"
                    for item in active
                )
            )
            self.assertTrue(
                {
                    "algorithm_failure",
                    "sample_failure",
                    "weak_target",
                    "weak_horizon",
                }
                <= {item["kind"] for item in archived}
            )
            self.assertTrue(
                all(item["status"] == "resolved_archived" for item in archived)
            )
            self.assertNotIn("secret-row", canonical_json(after_replay))
            self.assertLessEqual(
                len(canonical_json(after_replay).encode("utf-8")),
                CROSS_GENERATION_EXPERIENCE_MAX_BYTES,
            )

            start_generation_batch(replayed, run_id)
            context = gateway.research_contexts[-1]
            iteration = replayed.state(run_id).research_iteration_for(2)

            self.assertEqual(context["cross_generation_experience"], after_replay)
            self.assertIsNotNone(iteration)
            assert iteration is not None
            self.assertEqual(
                iteration.plan["prediction_model"], plan_one["prediction_model"]
            )
            self.assertEqual(
                iteration.plan["algorithm_blueprint"],
                plan_one["algorithm_blueprint"],
            )
            self.assertEqual(
                iteration.plan["algorithm_synthesis"],
                plan_one["algorithm_synthesis"],
            )

    def test_completed_archived_run_enters_new_run_research_context(self) -> None:
        historical_task = _historical_experience_task(samples_per_update=9)
        current_task = _historical_experience_task(samples_per_update=500)
        gateway = _PartialResearchGateway()
        historical_run_id = "run:historical-experience-source"
        current_run_id = "run:historical-experience-current"
        analysis = GenerationAnalysis(
            run_id=historical_run_id,
            generation=0,
            candidate_count=1,
            eligible_count=1,
            outcome="promoted",
            selected_candidate_id="candidate:historical:0",
            champion_candidate_id="candidate:historical:0",
            ranking=(
                {
                    "candidate_id": "candidate:historical:0",
                    "score": 0.42,
                    "parameters": {
                        "history_steps": 6,
                        "ridge_alpha": 0.05,
                        "residual_scale": 0.75,
                    },
                },
            ),
            target_weaknesses=(
                {
                    "target": "co2_concentration",
                    "horizon_hours": 1,
                    "median_skill_score": -0.4,
                    "evidence_count": 3,
                },
            ),
            sample_failures=(
                {
                    "failed": 1,
                    "failure_counts": {"gateway_timeout": 1},
                    "sample_execution_records": [
                        {
                            "sample_id": "must-not-cross-runs",
                            "observed": 1000.0,
                            "predicted": 800.0,
                        }
                    ],
                },
            ),
        )

        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            director.start_evolution(historical_task, run_id=historical_run_id)
            ledger.append(
                historical_run_id,
                "GenerationAnalyzed",
                {"analysis": analysis.to_dict()},
            )
            director.cancel_run(historical_run_id, "historical run closed")
            ledger.archive_run(historical_run_id)
            director.start_evolution(current_task, run_id=current_run_id)
            cutoff = director.state(current_run_id).events[0].seq

            start_generation_batch(director, current_run_id)
            context = gateway.research_contexts[-1][
                "cross_generation_experience"
            ]
            sources = context["historical_generations"]

            self.assertEqual(len(sources), 1)
            source = sources[0]
            self.assertEqual(source["source_run_id"], historical_run_id)
            self.assertEqual(source["source_generation"], 0)
            self.assertEqual(
                source["source_analysis_digest"], analysis.analysis_digest
            )
            self.assertEqual(source["compatibility_scope"], "directional_only")
            self.assertEqual(source["history_cutoff_seq"], cutoff)
            self.assertNotIn("best_score", source)
            self.assertEqual(
                source["modifications"]["candidate_parameter_sets"][0],
                {
                    "history_steps": 6,
                    "residual_scale": 0.75,
                    "ridge_alpha": 0.05,
                },
            )
            self.assertNotIn("must-not-cross-runs", canonical_json(context))
            provenance = context["historical_provenance"]
            self.assertEqual(provenance["history_cutoff_seq"], cutoff)
            self.assertEqual(provenance["source_digest"], digest(sources))
            iteration = director.state(current_run_id).research_iteration_for(0)
            self.assertIsNotNone(iteration)
            assert iteration is not None
            self.assertIsNone(iteration.source_analysis_digest)
            self.assertEqual(iteration.historical_provenance, provenance)
            projected_provenance = _projection_json(
                director.state(current_run_id)
            )["rounds"][0]["research_iteration"]["historical_provenance"]
            self.assertEqual(
                projected_provenance["source_digest"],
                provenance["source_digest"],
            )
            self.assertEqual(
                projected_provenance["included_generation_count"],
                1,
            )
            self.assertNotIn("max_serialized_bytes", projected_provenance)
            self.assertLessEqual(
                len(
                    canonical_json(
                        {
                            "historical_generations": sources,
                            "historical_provenance": provenance,
                        }
                    ).encode("utf-8")
                ),
                8 * 1024,
            )

            replayed = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            replayed_experience = build_cross_generation_experience(
                replayed.state(current_run_id),
                0,
                historical_states=(replayed.state(historical_run_id),),
                history_cutoff_seq=cutoff,
            )
            self.assertEqual(replayed_experience, context)
            replayed_iteration = replayed.state(current_run_id).research_iteration_for(0)
            self.assertIsNotNone(replayed_iteration)
            assert replayed_iteration is not None
            self.assertEqual(
                replayed_iteration.historical_provenance,
                provenance,
            )

    def test_historical_parameter_guardrails_require_replicated_cell_evidence(
        self,
    ) -> None:
        historical_task = _historical_experience_task(
            evaluator_id="greenhouse_multihorizon_time_forward@1",
            evaluator_digest="1" * 64,
        )
        current_state = _historical_state(
            historical_task,
            "run:guardrail-current",
            (),
            created_seq=100,
        )

        one_sample_state = _historical_state(
            historical_task,
            "run:guardrail-one-sample",
            (
                _guardrail_analysis(
                    "run:guardrail-one-sample",
                    0,
                    cohort_digest="a" * 64,
                    cell_sample_count=1,
                ),
            ),
            created_seq=10,
        )
        one_sample_experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=(one_sample_state,),
            history_cutoff_seq=100,
        )
        self.assertEqual(
            one_sample_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )
        self.assertEqual(len(one_sample_experience["historical_generations"]), 1)

        repeated_same_cohort_state = _historical_state(
            historical_task,
            "run:guardrail-same-cohort",
            tuple(
                _guardrail_analysis(
                    "run:guardrail-same-cohort",
                    generation,
                    cohort_digest="a" * 64,
                    cell_sample_count=20,
                    window_offset=0,
                )
                for generation in range(2)
            ),
            created_seq=20,
        )
        repeated_same_cohort_experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=(repeated_same_cohort_state,),
            history_cutoff_seq=100,
        )
        self.assertEqual(
            repeated_same_cohort_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )
        self.assertEqual(
            len(repeated_same_cohort_experience["historical_generations"]),
            2,
        )

        replicated_state = _historical_state(
            historical_task,
            "run:guardrail-replicated",
            (
                _guardrail_analysis(
                    "run:guardrail-replicated",
                    0,
                    cohort_digest="a" * 64,
                    cell_sample_count=20,
                    skill_score=0.12,
                ),
                _guardrail_analysis(
                    "run:guardrail-replicated",
                    1,
                    cohort_digest="b" * 64,
                    cell_sample_count=21,
                    skill_score=0.08,
                ),
            ),
            created_seq=30,
        )
        experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=(replicated_state,),
            history_cutoff_seq=100,
        )
        guardrails = experience["historical_parameter_guardrails"]
        self.assertEqual(
            guardrails["policy"],
            "preserve_verified_target_horizon_parameters_without_new_aggregate_evidence",
        )
        self.assertEqual(
            guardrails["evidence_requirements"],
            {
                "minimum_independent_cohort_count": 2,
                "minimum_cell_sample_count_per_cohort": 20,
                "minimum_total_cell_sample_count": 40,
                "same_cohort_generations_count_once": True,
                "requires_pairwise_non_overlapping_cohorts": True,
                "requires_nonnegative_skill_in_every_observation": True,
                "requires_zero_constraint_violations_in_every_observation": True,
            },
        )
        protected = guardrails["protected_parameter_evidence"]
        self.assertEqual(
            protected,
            [
                {
                    "parameter": "co2_concentration_1h_residual_scale",
                    "value": 0.0,
                    "target": "co2_concentration",
                    "horizon_hours": 1,
                    "skill_score": 0.08,
                    "source_run_id": "run:guardrail-replicated",
                    "source_generation": 1,
                    "independent_cohort_count": 2,
                    "total_cell_sample_count": 41,
                    "cohort_evidence": [
                        {
                            "evaluation_cohort_digest": "a" * 64,
                            "cell_sample_count": 20,
                            "minimum_skill_score": 0.12,
                            "schema_version": "ecologyrsi-dsh.feedback-update-cohort/1",
                            "selection_policy": "target_horizon_interleaved_rotating_window@1",
                            "feedback_update_cohort_digest": "a" * 64,
                            "population_count": 1000,
                            "population_digest": "f" * 64,
                            "selected_count": 20,
                            "window_offset": 0,
                            "source_run_id": "run:guardrail-replicated",
                            "source_generation": 0,
                        },
                        {
                            "evaluation_cohort_digest": "b" * 64,
                            "cell_sample_count": 21,
                            "minimum_skill_score": 0.08,
                            "schema_version": "ecologyrsi-dsh.feedback-update-cohort/1",
                            "selection_policy": "target_horizon_interleaved_rotating_window@1",
                            "feedback_update_cohort_digest": "b" * 64,
                            "population_count": 1000,
                            "population_digest": "f" * 64,
                            "selected_count": 21,
                            "window_offset": 21,
                            "source_run_id": "run:guardrail-replicated",
                            "source_generation": 1,
                        },
                    ],
                    "policy": "preserve_verified_target_horizon_parameters_without_new_aggregate_evidence",
                }
            ],
        )
        self.assertNotIn("must-not-enter-guardrails", canonical_json(experience))

        underpowered_cohort_state = _historical_state(
            historical_task,
            "run:guardrail-underpowered",
            (
                _guardrail_analysis(
                    "run:guardrail-underpowered",
                    0,
                    cohort_digest="a" * 64,
                    cell_sample_count=20,
                ),
                _guardrail_analysis(
                    "run:guardrail-underpowered",
                    1,
                    cohort_digest="b" * 64,
                    cell_sample_count=19,
                ),
            ),
            created_seq=40,
        )
        underpowered_experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=(underpowered_cohort_state,),
            history_cutoff_seq=100,
        )
        self.assertEqual(
            underpowered_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )

        conflicting_values_state = _historical_state(
            historical_task,
            "run:guardrail-conflicting-values",
            tuple(
                _guardrail_analysis(
                    "run:guardrail-conflicting-values",
                    generation,
                    cohort_digest=cohort * 64,
                    cell_sample_count=20,
                    value=value,
                )
                for generation, (cohort, value) in enumerate(
                    (("a", 0.0), ("b", 0.0), ("c", 0.5), ("d", 0.5))
                )
            ),
            created_seq=50,
        )
        conflicting_values_experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=(conflicting_values_state,),
            history_cutoff_seq=100,
        )
        self.assertEqual(
            conflicting_values_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )

        overlapping_windows_state = _historical_state(
            historical_task,
            "run:guardrail-overlapping-windows",
            (
                _guardrail_analysis(
                    "run:guardrail-overlapping-windows",
                    0,
                    cohort_digest="a" * 64,
                    cell_sample_count=20,
                    population_count=600,
                    selected_count=500,
                    window_offset=0,
                ),
                _guardrail_analysis(
                    "run:guardrail-overlapping-windows",
                    1,
                    cohort_digest="b" * 64,
                    cell_sample_count=20,
                    population_count=600,
                    selected_count=500,
                    window_offset=500,
                ),
            ),
            created_seq=60,
        )
        overlapping_windows_experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=(overlapping_windows_state,),
            history_cutoff_seq=100,
        )
        self.assertEqual(
            overlapping_windows_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )
        self.assertEqual(
            len(overlapping_windows_experience["historical_generations"]),
            2,
        )

        different_populations_state = _historical_state(
            historical_task,
            "run:guardrail-different-populations",
            (
                _guardrail_analysis(
                    "run:guardrail-different-populations",
                    0,
                    cohort_digest="a" * 64,
                    cell_sample_count=20,
                    population_digest="c" * 64,
                    window_offset=0,
                ),
                _guardrail_analysis(
                    "run:guardrail-different-populations",
                    1,
                    cohort_digest="b" * 64,
                    cell_sample_count=20,
                    population_digest="d" * 64,
                    window_offset=20,
                ),
            ),
            created_seq=70,
        )
        different_populations_experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=(different_populations_state,),
            history_cutoff_seq=100,
        )
        self.assertEqual(
            different_populations_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )

    def test_historical_parameter_guardrails_reject_conflicts_and_binding_drift(
        self,
    ) -> None:
        historical_task = _historical_experience_task(
            evaluator_id="greenhouse_multihorizon_time_forward@1",
            evaluator_digest="1" * 64,
        )
        current_state = _historical_state(
            historical_task,
            "run:guardrail-current",
            (),
            created_seq=100,
        )

        for label, second_skill, second_violations in (
            ("negative-skill", -0.01, 0),
            ("constraint-violation", 0.1, 1),
        ):
            with self.subTest(label=label):
                source_run_id = f"run:guardrail-{label}"
                historical_state = _historical_state(
                    historical_task,
                    source_run_id,
                    (
                        _guardrail_analysis(
                            source_run_id,
                            0,
                            cohort_digest="a" * 64,
                            cell_sample_count=20,
                        ),
                        _guardrail_analysis(
                            source_run_id,
                            1,
                            cohort_digest="b" * 64,
                            cell_sample_count=20,
                            skill_score=second_skill,
                            constraint_violations=second_violations,
                        ),
                    ),
                    created_seq=10,
                )
                experience = build_cross_generation_experience(
                    current_state,
                    0,
                    historical_states=(historical_state,),
                    history_cutoff_seq=100,
                )
                self.assertEqual(
                    experience["historical_parameter_guardrails"][
                        "protected_parameter_evidence"
                    ],
                    [],
                )

        replicated_state = _historical_state(
            historical_task,
            "run:guardrail-replicated",
            tuple(
                _guardrail_analysis(
                    "run:guardrail-replicated",
                    generation,
                    cohort_digest=cohort * 64,
                    cell_sample_count=20,
                )
                for generation, cohort in enumerate(("a", "b"))
            ),
            created_seq=20,
        )

        changed_evaluator_state = _historical_state(
            _historical_experience_task(
                evaluator_id="greenhouse_multihorizon_time_forward@2",
                evaluator_digest="2" * 64,
            ),
            "run:guardrail-changed-evaluator",
            (),
            created_seq=101,
        )
        changed_evaluator_experience = build_cross_generation_experience(
            changed_evaluator_state,
            0,
            historical_states=(replicated_state,),
            history_cutoff_seq=101,
        )
        self.assertEqual(
            changed_evaluator_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )
        self.assertEqual(
            len(changed_evaluator_experience["historical_generations"]),
            2,
        )

        changed_dataset_state = _historical_state(
            _historical_experience_task(
                evaluator_id="greenhouse_multihorizon_time_forward@1",
                evaluator_digest="1" * 64,
                split_manifest_digest="x" * 64,
            ),
            "run:guardrail-changed-dataset",
            (),
            created_seq=102,
        )
        changed_dataset_experience = build_cross_generation_experience(
            changed_dataset_state,
            0,
            historical_states=(replicated_state,),
            history_cutoff_seq=102,
        )
        self.assertEqual(
            changed_dataset_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )
        self.assertEqual(
            changed_dataset_experience["historical_generations"],
            [],
        )

    def test_historical_experience_round_robins_runs_within_byte_budget(
        self,
    ) -> None:
        task = _historical_experience_task()
        historical_states = []
        source_ids = []
        for run_index in range(4):
            run_id = f"run:balanced-history:{run_index}"
            source_ids.append(run_id)
            analyses = tuple(
                GenerationAnalysis(
                    run_id=run_id,
                    generation=generation,
                    candidate_count=1,
                    eligible_count=0,
                    outcome="no_eligible_candidate",
                    ranking=(
                        {
                            "candidate_id": f"candidate:{run_index}:{generation}",
                            "score": float(generation) / 10,
                            "parameters": {
                                "history_steps": generation + 1,
                                "ridge_alpha": 0.1,
                            },
                        },
                    ),
                    target_weaknesses=tuple(
                        {
                            "target": f"target_{run_index}_{generation}_{item}",
                            "horizon_hours": item + 1,
                            "median_skill_score": -0.5,
                            "evidence_count": 2,
                        }
                        for item in range(8)
                    ),
                    algorithm_failures=tuple(
                        {
                            "phase": "debug",
                            "stage": f"stage-{item}",
                            "failure_code": f"failure-{run_index}-{generation}-{item}",
                            "attempt_count": 1,
                        }
                        for item in range(6)
                    ),
                    sample_failures=(
                        {
                            "failed": 1,
                            "failure_counts": {"gateway_timeout": 1},
                            "sample_execution_records": [
                                {"observed": f"raw-marker-{run_index}-{generation}"}
                            ],
                        },
                    ),
                )
                for generation in range(8)
            )
            historical_states.append(
                _historical_state(
                    task,
                    run_id,
                    analyses,
                    created_seq=run_index * 100 + 1,
                )
            )
        current_state = _historical_state(
            task,
            "run:balanced-history:current",
            (),
            created_seq=1000,
        )

        experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=tuple(historical_states),
            history_cutoff_seq=1000,
        )
        sources = experience["historical_generations"]
        source_counts = {
            run_id: sum(item["source_run_id"] == run_id for item in sources)
            for run_id in source_ids
        }
        historical_envelope = {
            "historical_generations": sources,
            "historical_parameter_guardrails": experience[
                "historical_parameter_guardrails"
            ],
            "historical_provenance": experience["historical_provenance"],
        }

        self.assertEqual(set(item["source_run_id"] for item in sources), set(source_ids))
        self.assertTrue(any(count >= 2 for count in source_counts.values()))
        self.assertLessEqual(
            len(canonical_json(historical_envelope).encode("utf-8")),
            8 * 1024,
        )
        self.assertLessEqual(
            len(canonical_json(experience).encode("utf-8")),
            CROSS_GENERATION_EXPERIENCE_MAX_BYTES,
        )
        self.assertNotIn("raw-marker", canonical_json(experience))

    def test_historical_guardrails_are_host_owned_in_both_model_contexts(
        self,
    ) -> None:
        gateway = _GuardrailOverrideResearchGateway()
        adapter = StrategyRouterDSHAdapter(gateway=gateway)  # type: ignore[arg-type]
        task = _greenhouse_switch_task()
        guardrails = {
            "policy": "preserve_verified_target_horizon_parameters_without_new_aggregate_evidence",
            "evidence_requirements": {
                "minimum_independent_cohort_count": 2,
                "minimum_cell_sample_count_per_cohort": 20,
                "minimum_total_cell_sample_count": 40,
                "same_cohort_generations_count_once": True,
                "requires_pairwise_non_overlapping_cohorts": True,
                "requires_nonnegative_skill_in_every_observation": True,
                "requires_zero_constraint_violations_in_every_observation": True,
            },
            "protected_parameter_evidence": [
                {
                    "parameter": "residual_scale",
                    "value": 0.0,
                    "target": "co2_concentration",
                    "horizon_hours": 1,
                    "skill_score": 0.2,
                    "source_run_id": "run:source",
                    "source_generation": 2,
                    "independent_cohort_count": 2,
                    "total_cell_sample_count": 40,
                    "cohort_evidence": [
                        {
                            "evaluation_cohort_digest": "a" * 64,
                            "cell_sample_count": 20,
                            "minimum_skill_score": 0.2,
                            "schema_version": "ecologyrsi-dsh.feedback-update-cohort/1",
                            "selection_policy": "target_horizon_interleaved_rotating_window@1",
                            "feedback_update_cohort_digest": "a" * 64,
                            "population_count": 100,
                            "population_digest": "f" * 64,
                            "selected_count": 20,
                            "window_offset": 0,
                            "source_run_id": "run:source",
                            "source_generation": 1,
                        },
                        {
                            "evaluation_cohort_digest": "b" * 64,
                            "cell_sample_count": 20,
                            "minimum_skill_score": 0.2,
                            "schema_version": "ecologyrsi-dsh.feedback-update-cohort/1",
                            "selection_policy": "target_horizon_interleaved_rotating_window@1",
                            "feedback_update_cohort_digest": "b" * 64,
                            "population_count": 100,
                            "population_digest": "f" * 64,
                            "selected_count": 20,
                            "window_offset": 20,
                            "source_run_id": "run:source",
                            "source_generation": 2,
                        },
                    ],
                    "policy": "preserve_verified_target_horizon_parameters_without_new_aggregate_evidence",
                }
            ],
        }
        experience = {"historical_parameter_guardrails": guardrails}
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, adapter)
            run_id = "run:guardrail-contexts"
            director.start_evolution(task, run_id=run_id)
            state = director.state(run_id)
            knowledge = retrieve_generation_knowledge(state).proposal_context()
            result = adapter.research_iteration(
                run=state.run,
                task=state.task_manifest,
                previous_generation_analysis=None,
                knowledge_snapshot=knowledge,
                previous_knowledge_assessment=None,
                current_plan={},
                cross_generation_experience=experience,
            )

            self.assertEqual(
                gateway.research_contexts[-1]["historical_parameter_guardrails"],
                guardrails,
            )
            self.assertEqual(
                result["plan"]["historical_parameter_guardrails"],
                guardrails,
            )
            iteration = ResearchIteration(
                run_id=run_id,
                generation=0,
                status="model_generated",
                plan=result["plan"],
                prediction_model_adoption=resolve_predictor_adoption(
                    state.task_manifest,
                    result["plan"],
                ).to_dict(),
                knowledge_snapshot_digest=knowledge["snapshot_digest"],
                model_id="research-model",
            )
            proposal = adapter.propose(
                state.run,
                state.task_manifest,
                state.run.session_id,  # type: ignore[arg-type]
                batch_context={
                    "generation": 0,
                    "slot_index": 0,
                    "batch_size": 1,
                    "round_parent_candidate_id": None,
                    "previous_generation_analysis": None,
                    "knowledge_snapshot": knowledge,
                    "knowledge_snapshot_digest": knowledge["snapshot_digest"],
                    "research_iteration": iteration.to_dict(),
                    "frozen_runtime_binding": None,
                    "context_digest": "guardrail-context",
                },
            )
            self.assertEqual(proposal.changes["residual_scale"], 0.0)
            self.assertEqual(
                proposal.metadata["historical_parameter_guardrail_enforcement"],
                {
                    "policy": "preserve_verified_target_horizon_parameters_without_new_aggregate_evidence",
                    "protected_parameter_names": ["residual_scale"],
                    "overridden_parameter_names": ["residual_scale"],
                },
            )

        proposal_context = gateway.proposal_contexts[-1]
        self.assertEqual(
            proposal_context["historical_parameter_guardrails"],
            guardrails,
        )
        self.assertEqual(
            proposal_context["autonomous_plan"][
                "historical_parameter_guardrails"
            ],
            guardrails,
        )

    def test_evaluator_version_history_is_directional_and_cutoff_is_strict(
        self,
    ) -> None:
        historical_task = _historical_experience_task(
            evaluator_id="greenhouse_multihorizon_time_forward@1",
            evaluator_digest="1" * 64,
        )
        mismatched_split_task = _historical_experience_task(
            evaluator_id="greenhouse_multihorizon_time_forward@1",
            evaluator_digest="1" * 64,
            split_manifest_digest="x" * 64,
        )
        current_task = _historical_experience_task(
            evaluator_id="greenhouse_multihorizon_time_forward@2",
            evaluator_digest="2" * 64,
        )
        same_id_drift_task = _historical_experience_task(
            evaluator_id="greenhouse_multihorizon_time_forward@2",
            evaluator_digest="3" * 64,
        )
        gateway = _PartialResearchGateway()

        def analyzed(run_id: str, score: float) -> GenerationAnalysis:
            return GenerationAnalysis(
                run_id=run_id,
                generation=0,
                candidate_count=1,
                eligible_count=0,
                outcome="no_eligible_candidate",
                ranking=(
                    {
                        "candidate_id": f"candidate:{run_id}",
                        "score": score,
                        "parameters": {
                            "history_steps": 3,
                            "ridge_alpha": 0.1,
                            "residual_scale": 0.5,
                        },
                    },
                ),
                target_weaknesses=(
                    {
                        "target": "co2_concentration",
                        "horizon_hours": 1,
                        "median_skill_score": -0.7,
                        "evidence_count": 2,
                    },
                ),
            )

        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            compatible_id = "run:directional-compatible"
            same_id_drift_id = "run:directional-same-id-drift"
            wrong_split_id = "run:directional-wrong-split"
            late_analysis_id = "run:directional-late-analysis"
            current_id = "run:directional-current"
            for run_id, task in (
                (compatible_id, historical_task),
                (same_id_drift_id, same_id_drift_task),
                (wrong_split_id, mismatched_split_task),
                (late_analysis_id, historical_task),
            ):
                director.start_evolution(task, run_id=run_id)
            ledger.append(
                compatible_id,
                "GenerationAnalyzed",
                {"analysis": analyzed(compatible_id, 0.8).to_dict()},
            )
            ledger.append(
                same_id_drift_id,
                "GenerationAnalyzed",
                {"analysis": analyzed(same_id_drift_id, 0.75).to_dict()},
            )
            ledger.append(
                wrong_split_id,
                "GenerationAnalyzed",
                {"analysis": analyzed(wrong_split_id, 0.9).to_dict()},
            )
            director.start_evolution(current_task, run_id=current_id)
            current_state = director.state(current_id)
            cutoff = current_state.events[0].seq
            ledger.append(
                late_analysis_id,
                "GenerationAnalyzed",
                {"analysis": analyzed(late_analysis_id, 0.95).to_dict()},
            )

            experience = build_cross_generation_experience(
                current_state,
                0,
                historical_states=tuple(
                    director.state(run_id)
                    for run_id in (
                        compatible_id,
                        same_id_drift_id,
                        wrong_split_id,
                        late_analysis_id,
                        current_id,
                    )
                ),
                history_cutoff_seq=cutoff,
            )
            sources = experience["historical_generations"]

            self.assertEqual(len(sources), 2)
            self.assertEqual(
                {item["source_run_id"] for item in sources},
                {compatible_id, same_id_drift_id},
            )
            self.assertTrue(
                all(
                    item["compatibility_scope"] == "directional_only"
                    for item in sources
                )
            )
            self.assertEqual(
                experience["historical_parameter_guardrails"][
                    "protected_parameter_evidence"
                ],
                [],
            )

            def keys(value: object) -> list[str]:
                if isinstance(value, dict):
                    return [
                        *(str(name) for name in value),
                        *(
                            nested
                            for item in value.values()
                            for nested in keys(item)
                        ),
                    ]
                if isinstance(value, list):
                    return [nested for item in value for nested in keys(item)]
                return []

            for source in sources:
                source_keys = [name.casefold() for name in keys(source)]
                self.assertFalse(
                    any(
                        "score" in name or "delta" in name
                        for name in source_keys
                    )
                )
                self.assertNotIn("improved", source_keys)
                self.assertEqual(
                    source["weak_targets"][0]["target"],
                    "co2_concentration",
                )
                self.assertNotIn(
                    "median_skill_score", source["weak_targets"][0]
                )

    def test_windowed_experience_compares_scores_only_within_same_cohort(
        self,
    ) -> None:
        task_data = _task(candidates_per_generation=1).to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "samples_per_update": 500,
        }
        task = TaskManifest.from_dict(task_data)
        synthesis = {
            "schema_version": "ecologyrsi-dsh.algorithm-synthesis/1",
            "pipeline_id": "greenhouse-exogenous-ridge@1",
            "evidence_refs": ["knowledge:windowed-experience"],
            "parameter_focus": ["ridge_alpha"],
            "rationale": "refine one bounded feedback cohort",
        }
        plan = {
            "prediction_model": {"id": "greenhouse-exogenous-ridge@1"},
            "algorithm_synthesis": synthesis,
        }
        first_cohort = "a" * 64
        next_cohort = "b" * 64
        analyses = tuple(
            GenerationAnalysis(
                run_id="run:windowed-experience",
                generation=generation,
                candidate_count=1,
                eligible_count=1,
                outcome="promoted",
                selected_candidate_id=f"candidate:windowed:{generation}",
                champion_candidate_id=f"candidate:windowed:{generation}",
                ranking=(
                    {
                        "candidate_id": f"candidate:windowed:{generation}",
                        "score": score,
                        "evaluation_cohort_digest": cohort_digest,
                        "parameters": {"ridge_alpha": 0.1 + generation / 10},
                    },
                ),
            )
            for generation, score, cohort_digest in (
                (0, 0.2, first_cohort),
                (1, 0.5, first_cohort),
                (2, 0.1, next_cohort),
            )
        )
        state = SimpleNamespace(
            task_manifest=task,
            generation_analyses=analyses,
            research_iterations=tuple(
                SimpleNamespace(
                    generation=generation,
                    plan=plan,
                    prediction_model_adoption={},
                )
                for generation in range(3)
            ),
            proposals=(),
            algorithm_attempts=(),
        )

        experience = build_cross_generation_experience(state, 3)
        initial, comparable, changed = experience["generations"]

        self.assertEqual(initial["score_comparison"], "no_previous_generation")
        self.assertFalse(initial["improved"])
        self.assertEqual(comparable["score_comparison"], "same_cohort")
        self.assertAlmostEqual(
            comparable["best_score_delta_vs_previous_generation"], 0.3
        )
        self.assertTrue(comparable["improved"])
        self.assertEqual(
            changed["score_comparison"], "different_cohort_not_compared"
        )
        self.assertNotIn("best_score_delta_vs_previous_generation", changed)
        self.assertFalse(changed["improved"])
        self.assertEqual(changed["best_score"], 0.1)
        self.assertEqual(changed["evaluation_cohort_digest"], next_cohort)
        synthesis_effect = changed["algorithm_synthesis_effect"]
        self.assertFalse(synthesis_effect["observed_improvement"])
        self.assertEqual(
            synthesis_effect["score_comparison"],
            "different_cohort_not_compared",
        )
        self.assertEqual(
            synthesis_effect["selection_interpretation"],
            "current_cohort_batch_champion",
        )

    def test_experience_compares_formal_candidates_not_unaccepted_batch_maxima(
        self,
    ) -> None:
        task = _task(candidates_per_generation=2)
        synthesis = {
            "schema_version": "ecologyrsi-dsh.algorithm-synthesis/1",
            "pipeline_id": "greenhouse-exogenous-ridge@1",
            "evidence_refs": ["knowledge:formal-comparison"],
            "parameter_focus": ["ridge_alpha"],
            "rationale": "compare only the formally selected candidate",
        }
        plan = {
            "prediction_model": {"id": "greenhouse-exogenous-ridge@1"},
            "algorithm_synthesis": synthesis,
        }

        def analysis(
            generation: int,
            selected_score: float,
            unaccepted_score: float,
        ) -> GenerationAnalysis:
            selected_id = f"candidate:formal:{generation}"
            return GenerationAnalysis(
                run_id="run:formal-comparison",
                generation=generation,
                candidate_count=2,
                eligible_count=1,
                outcome="promoted",
                selected_candidate_id=selected_id,
                champion_candidate_id=selected_id,
                ranking=(
                    {
                        "candidate_id": f"candidate:unaccepted:{generation}",
                        "score": unaccepted_score,
                        "eligible": False,
                        "parameters": {"ridge_alpha": 9.0},
                    },
                    {
                        "candidate_id": selected_id,
                        "score": selected_score,
                        "eligible": True,
                        "parameters": {"ridge_alpha": 0.1 + generation / 10},
                    },
                ),
            )

        analyses = (
            analysis(0, 0.6, 0.95),
            analysis(1, 0.7, 0.8),
            # Deliberately inconsistent historical evidence: even if an old
            # event says promoted, a negative formal delta is not improvement.
            analysis(2, 0.65, 0.99),
        )
        state = SimpleNamespace(
            task_manifest=task,
            generation_analyses=analyses,
            research_iterations=tuple(
                SimpleNamespace(
                    generation=generation,
                    plan=plan,
                    prediction_model_adoption={},
                )
                for generation in range(3)
            ),
            proposals=(),
            algorithm_attempts=(),
        )

        first, improved, regressed = build_cross_generation_experience(
            state, 3
        )["generations"]

        self.assertEqual(first["comparison_score"], 0.6)
        self.assertEqual(first["best_score"], 0.6)
        self.assertEqual(first["batch_highest_observed_score"], 0.95)
        self.assertFalse(first["improved"])

        self.assertAlmostEqual(
            improved["best_score_delta_vs_previous_generation"], 0.1
        )
        self.assertTrue(improved["improved"])
        self.assertEqual(improved["comparison_candidate_role"], "champion")
        self.assertEqual(improved["batch_highest_observed_score"], 0.8)
        improved_effect = improved["algorithm_synthesis_effect"]
        self.assertEqual(improved_effect["best_score"], 0.7)
        self.assertEqual(improved_effect["batch_highest_observed_score"], 0.8)
        self.assertTrue(improved_effect["observed_improvement"])

        self.assertAlmostEqual(
            regressed["best_score_delta_vs_previous_generation"], -0.05
        )
        self.assertFalse(regressed["improved"])
        regressed_effect = regressed["algorithm_synthesis_effect"]
        self.assertFalse(regressed_effect["observed_improvement"])
        self.assertEqual(
            regressed_effect["selection_interpretation"],
            "selected_without_observed_score_improvement",
        )

    def test_cross_generation_experience_enforces_capacity_limits(self) -> None:
        analyses = []
        for generation in range(30):
            analyses.append(
                GenerationAnalysis(
                    run_id="run:experience-capacity",
                    generation=generation,
                    candidate_count=1,
                    eligible_count=0,
                    outcome="no_eligible_candidate",
                    ranking=(
                        {
                            "candidate_id": f"candidate:capacity:{generation}",
                            "score": -float(generation + 1),
                            "parameters": {
                                "alpha": 0.2,
                                "window": generation + 1,
                                "water_threshold": 0.35,
                            },
                        },
                    ),
                    target_weaknesses=tuple(
                        {
                            "target": f"target-{generation}-{index}",
                            "horizon_hours": index + 1,
                            "median_skill_score": -1.0,
                            "evidence_count": 1,
                        }
                        for index in range(8)
                    ),
                    horizon_weaknesses=tuple(
                        {
                            "horizon_hours": generation * 10 + index,
                            "median_skill_score": -1.0,
                            "evidence_count": 1,
                        }
                        for index in range(8)
                    ),
                    algorithm_failures=tuple(
                        {
                            "phase": "debug",
                            "stage": f"stage-{index}",
                            "failure_code": f"failure-{generation}-{index}",
                            "attempt_count": 1,
                        }
                        for index in range(16)
                    ),
                    sample_failures=(
                        {
                            "failed": 32,
                            "failure_counts": {
                                f"sample-failure-{generation}-{index}": 2
                                for index in range(16)
                            },
                        },
                    ),
                )
            )
        state = SimpleNamespace(
            generation_analyses=tuple(analyses),
            research_iterations=(),
            proposals=(),
        )

        experience = build_cross_generation_experience(state, 30)

        self.assertEqual(experience["window"]["available_generation_count"], 30)
        self.assertEqual(experience["window"]["scanned_generation_count"], 24)
        self.assertTrue(experience["window"]["history_truncated"])
        self.assertLessEqual(len(experience["generations"]), 6)
        self.assertLessEqual(len(experience["active_unresolved"]), 16)
        self.assertLessEqual(len(experience["resolved_archived"]), 16)
        self.assertGreater(
            experience["capacity"]["omitted_generation_summaries"], 0
        )
        self.assertGreater(experience["capacity"]["omitted_active_issues"], 0)
        self.assertLessEqual(
            len(canonical_json(experience).encode("utf-8")),
            CROSS_GENERATION_EXPERIENCE_MAX_BYTES,
        )

    def test_partial_generation_recovers_plan_without_remote_research(self) -> None:
        task = _task(candidates_per_generation=1)
        with EventLedger() as ledger:
            first = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=_ResearchGateway()),  # type: ignore[arg-type]
            )
            first.start_evolution(task, run_id="run:partial-research")
            snapshot = retrieve_generation_knowledge(
                first.state("run:partial-research")
            )
            ledger.append(
                "run:partial-research",
                "GenerationKnowledgeRetrieved",
                {"knowledge_snapshot": snapshot.to_dict()},
            )
            legacy_plan = {
                "prediction_model": {"id": "unregistered-transformer@9"},
                "research": [],
            }
            first.submit_proposal(
                Proposal(
                    proposal_id="proposal:legacy-partial",
                    run_id="run:partial-research",
                    generation=0,
                    title="legacy partial proposal",
                    changes={
                        "alpha": 0.4,
                        "window": 5,
                        "water_threshold": 0.4,
                    },
                    metadata={"plan": legacy_plan},
                )
            )

            replay_gateway = _ResearchGateway(fail_research=True)
            replayed = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(  # type: ignore[arg-type]
                    gateway=replay_gateway
                ),
            )
            start_generation_batch(replayed, "run:partial-research")
            iteration = replayed.state(
                "run:partial-research"
            ).research_iteration_for(0)

            self.assertIsNotNone(iteration)
            assert iteration is not None
            self.assertEqual(iteration.status, "recovered_existing_proposal")
            self.assertEqual(iteration.plan, legacy_plan)
            self.assertEqual(replay_gateway.research_contexts, [])

    def test_research_adoption_changes_schema_then_runs_full_pipeline(self) -> None:
        switch_gateway = _ResearchGateway(
            requested_id="greenhouse-rolling-residual@1"
        )
        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(  # type: ignore[arg-type]
                    gateway=switch_gateway
                ),
            )
            switch_task = _greenhouse_switch_task()
            director.start_evolution(switch_task, run_id="run:research-switch")
            batch = start_generation_batch(director, "run:research-switch")
            proposal = director.request_proposal(
                "run:research-switch", generation_batch=batch, slot_index=0
            )
            adoption = proposal.metadata["prediction_model_adoption"]

            self.assertEqual(adoption["status"], "adopted")
            self.assertEqual(
                adoption["adopted_id"], "greenhouse-rolling-residual@1"
            )
            self.assertEqual(set(proposal.changes), {"blend", "window", "bias_scale"})
            self.assertEqual(
                compile_algorithm_spec(
                    switch_task,
                    proposal,
                    director.state("run:research-switch").knowledge_for(0),
                ).adapter_id,
                "greenhouse-rolling-residual@1",
            )

        datasets = DatasetRegistry()
        evaluators = EvaluatorRegistry(datasets)
        series = datasets.series("generated-toy-series@1")
        predictor_digest = evaluators.predictor_configuration_digest(
            "toy-rolling-water@1"
        )
        task_data = _task(candidates_per_generation=1).to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "dataset_digest": series.digest,
            "split_manifest_digest": series.split_manifest_digest_sha256,
            "prediction_model_digest": predictor_digest,
            "runtime_component_catalog": {
                "prediction_models": [
                    {
                        "id": "toy-rolling-water@1",
                        "dataset_ids": ["generated-toy-series@1"],
                        "configuration_digest": predictor_digest,
                        "parameter_schemas": {
                            "alpha": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "window": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 24,
                            },
                            "water_threshold": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                    }
                ],
                "evaluators": [],
                "selected_prediction_model_id": "toy-rolling-water@1",
                "selected_evaluator_id": "toy_time_forward@1",
            },
        }
        full_task = TaskManifest.from_dict(task_data)
        full_gateway = _ResearchGateway(requested_id="toy-rolling-water@1")
        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(  # type: ignore[arg-type]
                    gateway=full_gateway
                ),
            )
            director.start_evolution(full_task, run_id="run:research-full")
            batch = start_generation_batch(director, "run:research-full")
            proposal = director.request_proposal(
                "run:research-full", generation_batch=batch, slot_index=0
            )
            candidate = director.spawn_candidate(
                "run:research-full", proposal, slot_index=0
            )
            endpoint = SimpleNamespace(
                server=SimpleNamespace(
                    director=director,
                    evaluators=evaluators,
                )
            )

            _evaluate_candidate(
                endpoint, "run:research-full", candidate.candidate_id
            )
            state = director.state("run:research-full")
            attempts = state.algorithm_attempts_for(candidate.candidate_id)
            debug = next(item for item in attempts if item.phase == "debug")

            self.assertEqual(
                proposal.metadata["prediction_model_adoption"]["status"],
                "adopted",
            )
            self.assertEqual(
                [(item.phase, item.status) for item in attempts],
                [("compile", "passed"), ("debug", "passed")],
            )
            self.assertTrue(debug.evidence["passed"])
            self.assertEqual(debug.evidence["smoke"]["status"], "passed")
            self.assertIsNotNone(state.artifact_for(candidate.candidate_id))
            self.assertIsNotNone(state.evaluation_for(candidate.candidate_id))
            completed_stages = {
                event.payload["stage"]
                for event in state.events
                if event.kind == "EvolutionStageRecorded"
                and event.payload.get("status") == "completed"
            }
            self.assertTrue({"training", "evaluation", "judge"} <= completed_stages)

    def test_iteration_rejects_executable_plan_fields(self) -> None:
        gateway = _ResearchGateway()
        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            director.start_evolution(
                _task(candidates_per_generation=1),
                run_id="run:research-security",
            )
            start_generation_batch(director, "run:research-security")
            iteration = director.state(
                "run:research-security"
            ).research_iteration_for(0)
            self.assertIsNotNone(iteration)
            assert iteration is not None
            payload = iteration.to_dict()
            payload["plan"] = {
                **payload["plan"],
                "implementation": {"source_code": "print('not allowed')"},
            }

            with self.assertRaisesRegex(ValueError, "forbidden executable field"):
                ResearchIteration.from_dict(payload)

    def test_iteration_rejects_untrusted_historical_provenance_fields(self) -> None:
        gateway = _ResearchGateway()
        with EventLedger() as ledger:
            director = EvolutionDirector(
                ledger,
                StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
            )
            director.start_evolution(
                _task(candidates_per_generation=1),
                run_id="run:research-provenance-security",
            )
            start_generation_batch(director, "run:research-provenance-security")
            iteration = director.state(
                "run:research-provenance-security"
            ).research_iteration_for(0)
            self.assertIsNotNone(iteration)
            assert iteration is not None
            payload = iteration.to_dict()
            payload["historical_provenance"] = {
                **payload["historical_provenance"],
                "api_key": "must-not-be-persisted",
            }

            with self.assertRaisesRegex(ValueError, "unsupported fields"):
                ResearchIteration.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
