from __future__ import annotations

import unittest
from types import SimpleNamespace

from ecologyrsi_dsh.api.dsh_tools import DshToolService
from ecologyrsi_dsh.api.generation_execution import _apply_candidate_judge
from ecologyrsi_dsh.core.director import EvolutionDirector, _task_for_proposal_predictor
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import (
    Evaluation,
    ModelArtifact,
    TaskManifest,
    canonical_json,
    digest,
)
from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter
from ecologyrsi_dsh.integrations.dsh_structured_roles import DshStructuredRoleRuntime
from ecologyrsi_dsh.knowledge.algorithms import (
    compile_algorithm_spec,
    resolve_predictor_adoption,
)
from ecologyrsi_dsh.knowledge.research_iteration import ResearchIteration


def _native_task() -> TaskManifest:
    return TaskManifest(
        task_id="structured-native",
        objective="predict greenhouse climate",
        domain_pack="greenhouse_environment@1",
        visible_datasets=("agc_cucumber_2018",),
        budget={"max_candidates": 2, "candidates_per_generation": 1},
        metadata={
            "execution_protocol": "dsh_native_plugin_evolution@1",
            "seed_genome_template_id": "greenhouse-default@1",
            "prediction_model_id": "greenhouse-horizon-targetwise-ridge@1",
            "evaluator_id": "greenhouse_multihorizon_time_forward@2",
            "strategy_id": "autonomous_model@1",
            "dataset_digest": "2" * 64,
            "dataset_snapshot_set_digest": "2" * 64,
            "split_manifest_digest": "3" * 64,
            "data_protocol_digest": "4" * 64,
            "stage_policy_digest": "5" * 64,
            "evaluator_digest": "6" * 64,
            "fitness_profile_digest": "7" * 64,
            "security_kernel_digest": "8" * 64,
            "selection_reviewer_program_digest": "9" * 64,
            "required_capability_digest": "a" * 64,
            "resolved_policy_route_digest": "b" * 64,
            "resolved_review_route_digest": "c" * 64,
            "resolved_policy_route_config_digest": "b" * 64,
            "resolved_review_route_config_digest": "c" * 64,
            "preset_content_digest": "d" * 64,
            "standing_tool_surface_digest": "e" * 64,
            "evaluation_cohort_digest": "f" * 64,
        },
    )


class _NativeRuntime:
    def __init__(self, structured: dict, *, wrong_digest: bool = False) -> None:
        self.structured = structured
        self.wrong_digest = wrong_digest
        self.requests: list[dict] = []

    def run_stage(self, request: dict) -> dict:
        self.requests.append(request)
        return {
            "structured": self.structured,
            "result_digest": "0" * 64 if self.wrong_digest else digest(self.structured),
        }


class DshStructuredRoleTests(unittest.TestCase):
    def test_result_digest_is_verified_before_host_use(self) -> None:
        runtime = _NativeRuntime({"schema_version": "x", "value": 1}, wrong_digest=True)
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            DshStructuredRoleRuntime(runtime).run(
                run_id="run:digest",
                stage="candidate.propose",
                role="candidate-proposer",
                context={"safe": True},
                output_schema_id="ecology-genome-mutation@1",
                run_state_revision=1,
                stage_attempt=1,
                ledger_expected_revision=1,
                idempotency_key="digest-1",
            )

    def test_native_research_freezes_failed_cross_run_parameters_for_proposer(self) -> None:
        runtime = _NativeRuntime(
            {
                "schema_version": "ecology-research-result@1",
                "summary": "Change direction after the prior scientific-gate failure.",
                "evidence": [],
            }
        )
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger, adapter)
        run_id = "run:native-cross-run-reflection"
        task = _native_task()
        director.create_run(task, run_id=run_id)
        director.start_run(run_id)
        state = director.state(run_id)
        parent = state.materialized_seed_genome()
        failed_parameters = dict(
            parent.to_dict()["scientific_program"]["parameter_overrides"]
        )
        failed_parameters["ridge_alpha"] = 0.15
        plan = adapter.research_plan(
            "strategy-model",
            run=state.run,
            task=task,
            previous_generation_analysis=None,
            knowledge_snapshot=None,
            previous_knowledge_assessment=None,
            current_plan={},
            cross_generation_experience={
                "schema_version": "ecologyrsi-dsh.cross-generation-experience/2",
                "generations": [],
                "historical_generations": [
                    {
                        "source_run_id": "run:prior-failed",
                        "source_generation": 0,
                        "outcome": "no_eligible_candidate",
                        "improved": False,
                        "common_failures": ["scientific_gate_failed"],
                        "modifications": {
                            "adopted_predictor_id": parent.to_dict()[
                                "scientific_program"
                            ]["predictor_ref"]["id"],
                            "candidate_parameter_sets": [failed_parameters],
                        },
                    }
                ],
                "active_unresolved": [],
            },
            parent_genome=parent.to_dict(),
            run_state_revision=state.events[-1].seq,
            stage_attempt=1,
            ledger_expected_revision=ledger.latest_seq(),
        )

        avoid = plan["dsh_evolution_reflection"]["avoid_parameter_sets"]
        self.assertEqual(len(avoid), 1)
        self.assertEqual(avoid[0]["parameters"], failed_parameters)
        self.assertEqual(avoid[0]["reason"], "scientific_gate_failed")
        self.assertEqual(avoid[0]["source_run_id"], "run:prior-failed")

    def test_native_research_retry_uses_distinct_idempotency_key(self) -> None:
        runtime = _NativeRuntime(
            {
                "schema_version": "ecology-research-result@1",
                "summary": "Retry after a transient DSH restart.",
                "evidence": [],
            }
        )
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger, adapter)
        task = _native_task()
        run_id = "run:native-research-retry-key"
        director.create_run(task, run_id=run_id)
        director.start_run(run_id)
        state = director.state(run_id)
        parent = state.materialized_seed_genome()

        for stage_attempt in (1, 2):
            adapter.research_plan(
                "strategy-model",
                run=state.run,
                task=task,
                previous_generation_analysis=None,
                knowledge_snapshot=None,
                previous_knowledge_assessment=None,
                current_plan={},
                parent_genome=parent.to_dict(),
                run_state_revision=state.events[-1].seq,
                stage_attempt=stage_attempt,
                ledger_expected_revision=ledger.latest_seq(),
            )

        keys = [request["idempotency_key"] for request in runtime.requests]
        self.assertEqual(len(keys), 2)
        self.assertNotEqual(keys[0], keys[1])
        self.assertTrue(keys[0].endswith(":attempt:1"))
        self.assertTrue(keys[1].endswith(":attempt:2"))

    def test_native_proposer_applies_only_host_registered_mutation(self) -> None:
        runtime = _NativeRuntime(
            {
                "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                "operations": [],
            }
        )
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger, adapter)
        run_id = "run:structured-native"
        task = _native_task()
        director.create_run(task, run_id=run_id)
        director.start_run(run_id)
        state = director.state(run_id)
        parent = state.materialized_seed_genome()
        canonical_parent = canonical_json(parent.to_dict())
        stage_digests = {
            "research_iteration_digest": "1" * 64,
            "knowledge_snapshot_digest": "2" * 64,
        }

        proposal = adapter.propose(
            state.run,
            task,
            state.run.session_id or "",
            batch_context={
                "generation": 0,
                "slot_index": 0,
                "batch_size": 1,
                "context_digest": "3" * 64,
                "parent_genome_digest": parent.genome_digest,
                "parent_genome_canonical_json": canonical_parent,
                "stage_context_digests": stage_digests,
                "run_state_revision": state.events[-1].seq,
                "stage_attempt": 1,
                "ledger_expected_revision": ledger.latest_seq(),
            },
        )

        child = proposal.metadata["evolution_genome_canonical_json"]
        self.assertNotEqual(child, canonical_parent)
        self.assertEqual(proposal.metadata["execution_protocol"], "dsh_native_plugin_evolution@1")
        self.assertEqual(proposal.metadata["proposal_source"], "dsh_native_agent")
        self.assertTrue(proposal.metadata["remote_strategy_called"])
        self.assertTrue(proposal.metadata["remote_strategy_succeeded"])
        self.assertEqual(runtime.requests[0]["request"]["output_schema_id"], "ecology-genome-mutation@1")
        self.assertEqual(runtime.requests[0]["request"]["context"]["parent_genome"]["genome_digest"], parent.genome_digest)

    def test_native_proposer_receives_actionable_reflection_not_only_digests(self) -> None:
        runtime = _NativeRuntime(
            {
                "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                "operations": [],
            }
        )
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger, adapter)
        run_id = "run:native-reflection"
        task = _native_task()
        director.create_run(task, run_id=run_id)
        director.start_run(run_id)
        state = director.state(run_id)
        parent = state.materialized_seed_genome()
        previous_analysis = {
            "analysis_digest": "a" * 64,
            "outcome": "no_eligible_candidate",
            "common_failures": ["scientific_gate_failed"],
            "next_generation_focus": "优先改善1小时室内气温预测",
            "next_search_direction": ["不要重复 ridge_alpha=0.15"],
            "ranking": [
                {
                    "candidate_id": "candidate:failed",
                    "classification": "scientific_gate_failed",
                    "score": -0.155,
                    "parameters": {
                        "ridge_alpha": 0.15,
                        "history_steps": 8,
                        "co2_concentration_1h_residual_scale": 0.4,
                    },
                }
            ],
        }
        plan = {
            "status": "model_generated",
            "dsh_research_summary": (
                "The preceding mutation failed the scientific gate; explore a "
                "different registered parameter direction."
            ),
            "dsh_research_evidence": [
                {
                    "finding": "The previous ridge configuration did not improve the baseline.",
                    "relevance": "Avoid an exact replay of the failed mutation.",
                }
            ],
        }
        iteration = ResearchIteration(
            run_id=run_id,
            generation=0,
            status="model_generated",
            plan=plan,
            prediction_model_adoption=resolve_predictor_adoption(
                task,
                plan,
            ).to_dict(),
            knowledge_snapshot_digest="2" * 64,
            source_analysis_digest=previous_analysis["analysis_digest"],
            previous_next_action="改变未验证参数，不要重复失败候选。",
        )

        adapter.propose(
            state.run,
            task,
            state.run.session_id or "",
            batch_context={
                "generation": 0,
                "slot_index": 0,
                "batch_size": 1,
                "previous_generation_analysis": previous_analysis,
                "knowledge_snapshot_digest": "2" * 64,
                "research_iteration": iteration.to_dict(),
                "context_digest": "3" * 64,
                "parent_genome_digest": parent.genome_digest,
                "parent_genome_canonical_json": canonical_json(parent.to_dict()),
                "stage_context_digests": {
                    "research_iteration_digest": iteration.iteration_digest,
                    "knowledge_snapshot_digest": "2" * 64,
                },
                "run_state_revision": state.events[-1].seq,
                "stage_attempt": 1,
                "ledger_expected_revision": ledger.latest_seq(),
            },
        )

        context = runtime.requests[0]["request"]["context"]
        self.assertEqual(
            context["research_iteration"]["iteration_digest"],
            iteration.iteration_digest,
        )
        reflection = context["evolution_reflection"]
        self.assertEqual(
            reflection["previous_generation_analysis"]["common_failures"],
            ["scientific_gate_failed"],
        )
        self.assertEqual(
            reflection["research_summary"],
            plan["dsh_research_summary"],
        )
        self.assertEqual(
            reflection["previous_next_action"],
            iteration.previous_next_action,
        )

    def test_native_proposer_retries_then_rejects_an_exact_failed_behavior(self) -> None:
        repeated_mutation = {
            "schema_version": "ecologyrsi-dsh.genome-mutation/1",
            "operations": [
                {
                    "op": "set_bounded_parameter",
                    "name": "ridge_alpha",
                    "value": 0.15,
                }
            ],
        }
        runtime = _NativeRuntime(repeated_mutation)
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger, adapter)
        run_id = "run:native-duplicate-reflection"
        task = _native_task()
        director.create_run(task, run_id=run_id)
        director.start_run(run_id)
        state = director.state(run_id)
        parent = state.materialized_seed_genome()
        failed_parameters = dict(
            parent.to_dict()["scientific_program"]["parameter_overrides"]
        )
        failed_parameters["ridge_alpha"] = 0.15
        plan = {
            "status": "model_generated",
            "dsh_evolution_reflection": {
                "schema_version": "ecologyrsi-dsh.evolution-reflection/1",
                "avoid_parameter_sets": [
                    {
                        "prediction_model_id": parent.to_dict()["scientific_program"][
                            "predictor_ref"
                        ]["id"],
                        "parameters": failed_parameters,
                        "reason": "scientific_gate_failed",
                    }
                ],
            },
        }
        iteration = ResearchIteration(
            run_id=run_id,
            generation=0,
            status="model_generated",
            plan=plan,
            prediction_model_adoption=resolve_predictor_adoption(task, plan).to_dict(),
            knowledge_snapshot_digest="2" * 64,
        )

        with self.assertRaisesRegex(ValueError, "repeats a previously failed behavior"):
            adapter.propose(
                state.run,
                task,
                state.run.session_id or "",
                batch_context={
                    "generation": 0,
                    "slot_index": 0,
                    "batch_size": 1,
                    "knowledge_snapshot_digest": "2" * 64,
                    "research_iteration": iteration.to_dict(),
                    "context_digest": "3" * 64,
                    "parent_genome_digest": parent.genome_digest,
                    "parent_genome_canonical_json": canonical_json(parent.to_dict()),
                    "stage_context_digests": {
                        "research_iteration_digest": iteration.iteration_digest,
                        "knowledge_snapshot_digest": "2" * 64,
                    },
                    "run_state_revision": state.events[-1].seq,
                    "stage_attempt": 1,
                    "ledger_expected_revision": ledger.latest_seq(),
                },
            )

        self.assertEqual(len(runtime.requests), 2)
        retry_reflection = runtime.requests[1]["request"]["context"][
            "evolution_reflection"
        ]
        self.assertEqual(
            retry_reflection["host_rejections"][0]["reason"],
            "exact_failed_behavior_replay",
        )

    def test_native_proposal_uses_verified_genome_predictor_boundary(self) -> None:
        runtime = _NativeRuntime(
            {
                "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                "operations": [],
            }
        )
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )
        task_data = _native_task().to_dict()
        task_data["metadata"]["prediction_model_id"] = "greenhouse-exogenous-ridge@1"
        task = TaskManifest.from_dict(task_data)
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger, adapter)
        director.create_run(task, run_id="run:native-genome-boundary")
        director.start_run("run:native-genome-boundary")
        state = director.state("run:native-genome-boundary")
        parent = state.materialized_seed_genome()
        proposal = adapter.propose(
            state.run,
            task,
            state.run.session_id or "",
            batch_context={
                "generation": 0,
                "slot_index": 0,
                "batch_size": 1,
                "context_digest": "3" * 64,
                "parent_genome_digest": parent.genome_digest,
                "parent_genome_canonical_json": canonical_json(parent.to_dict()),
                "stage_context_digests": {
                    "research_iteration_digest": "1" * 64,
                    "knowledge_snapshot_digest": "2" * 64,
                },
                "run_state_revision": state.events[-1].seq,
                "stage_attempt": 1,
                "ledger_expected_revision": ledger.latest_seq(),
            },
        )

        effective = _task_for_proposal_predictor(task, proposal)
        self.assertEqual(
            effective.metadata["prediction_model_id"],
            "greenhouse-horizon-targetwise-ridge@1",
        )
        compiled = compile_algorithm_spec(task, proposal, None)
        self.assertEqual(
            compiled.adapter_id,
            "greenhouse-horizon-targetwise-ridge@1",
        )

    def test_native_judge_cannot_override_a_scientific_failure(self) -> None:
        runtime = _NativeRuntime(
            {
                "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                "operations": [],
            }
        )
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger, adapter)
        task = _native_task()
        run_id = "run:native-judge"
        director.create_run(task, run_id=run_id)
        director.start_run(run_id)
        state = director.state(run_id)
        parent = state.materialized_seed_genome()
        proposal = adapter.propose(
            state.run,
            task,
            state.run.session_id or "",
            batch_context={
                "generation": 0,
                "slot_index": 0,
                "batch_size": 1,
                "context_digest": "3" * 64,
                "parent_genome_digest": parent.genome_digest,
                "parent_genome_canonical_json": canonical_json(parent.to_dict()),
                "stage_context_digests": {
                    "research_iteration_digest": "1" * 64,
                    "knowledge_snapshot_digest": "2" * 64,
                },
                "run_state_revision": state.events[-1].seq,
                "stage_attempt": 1,
                "ledger_expected_revision": ledger.latest_seq(),
            },
        )
        director.submit_proposal(proposal)
        candidate = director.spawn_candidate(run_id, proposal, slot_index=0)
        artifact = ModelArtifact(
            artifact_id="artifact:native-judge",
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            model_id="registered-predictor",
            dataset_digest="2" * 64,
            training_partition="training_fit",
            training_rows=10,
        )
        director.record_artifact(artifact)
        scientific = Evaluation(
            evaluation_id="evaluation:native-judge",
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            score=-0.2,
            passed=False,
            metrics={"scientific_pass": False, "skill_score": -0.2},
            evaluator_digest="6" * 64,
            artifact_digest=artifact.digest,
        )
        director.record_evaluation(scientific)
        runtime.structured = {
            "schema_version": "ecology-generation-review@1",
            "accepted": True,
            "rationale": "advisory acceptance",
            "flags": [],
        }
        legacy_evaluators = SimpleNamespace(
            apply_judge=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("legacy judge was used")
            )
        )
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=director,
                ledger=ledger,
                dsh_native_runtime=runtime,
                dsh_tools=DshToolService(ledger),
                evaluators=legacy_evaluators,
            )
        )

        _apply_candidate_judge(
            endpoint,
            director.state(run_id),
            proposal,
            artifact,
            scientific,
        )

        judged = director.state(run_id).evaluation_for(candidate.candidate_id)
        self.assertIsNotNone(judged)
        self.assertFalse(judged.passed)
        self.assertTrue(judged.metrics["judge_accepted"], judged.metrics)
        self.assertEqual(runtime.requests[-1]["stage"], "generation.judge")


if __name__ == "__main__":
    unittest.main()
