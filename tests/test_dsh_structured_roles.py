from __future__ import annotations

import json
import re
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
from ecologyrsi_dsh.evolution.genome import EcologyEvolutionPluginGenome
from ecologyrsi_dsh.evolution.strategies import (
    StrategyRouterDSHAdapter,
    _native_evolution_reflection_from_experience,
)
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


class _SequencedNativeRuntime(_NativeRuntime):
    def __init__(self, structured_results: list[dict]) -> None:
        if not structured_results:
            raise ValueError("structured_results must be non-empty")
        super().__init__(structured_results[0])
        self.structured_results = list(structured_results)

    def run_stage(self, request: dict) -> dict:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.structured_results) - 1)
        structured = self.structured_results[index]
        return {"structured": structured, "result_digest": digest(structured)}


class DshStructuredRoleTests(unittest.TestCase):
    def test_runtime_request_carries_the_server_issued_admission_id(self) -> None:
        structured = {
            "schema_version": "ecology-research-result@1",
            "summary": "bounded evidence",
            "evidence": [],
        }
        native = _NativeRuntime(structured)
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        ledger.append("run:admission-id", "RunCreated", {"test": True})
        service = DshToolService(ledger)
        runtime = DshStructuredRoleRuntime(native, admission=service)

        self.assertEqual(
            runtime.run(
                run_id="run:admission-id",
                stage="generation.research",
                role="researcher",
                context={"question": "bounded"},
                output_schema_id="ecology-research-result@1",
                run_state_revision=3,
                stage_attempt=1,
                ledger_expected_revision=ledger.latest_seq(),
                idempotency_key="research-admission-1",
                identity_digests={},
            ),
            structured,
        )
        admission_id = native.requests[0]["admission_id"]
        self.assertIsInstance(admission_id, str)
        self.assertTrue(re.fullmatch(r"admission-[0-9a-f-]{36}", admission_id))

    def test_native_proposer_retries_a_no_effect_mutation(self) -> None:
        runtime = _SequencedNativeRuntime(
            [
                {
                    "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                    "operations": [
                        {
                            "op": "select_registered_pipeline",
                            "predictor_id": "greenhouse-horizon-targetwise-ridge@1",
                        }
                    ],
                },
                {
                    "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                    "operations": [
                        {
                            "op": "set_bounded_parameter",
                            "name": "ridge_alpha",
                            "value": 0.2,
                        }
                    ],
                },
            ]
        )
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        admission = DshToolService(ledger)
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: DshStructuredRoleRuntime(
                runtime,
                admission=admission,
            ),
        )
        director = EvolutionDirector(ledger, adapter)
        task = _native_task()
        run_id = "run:native-invalid-mutation-retry"
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
                "batch_size": 1,
                "slot_index": 0,
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

        self.assertEqual(proposal.changes["ridge_alpha"], 0.2)
        self.assertEqual(len(runtime.requests), 2)
        self.assertEqual(
            [request["stage_attempt"] for request in runtime.requests],
            [1, 2],
        )
        retry_reflection = runtime.requests[1]["request"]["context"][
            "evolution_reflection"
        ]
        self.assertEqual(
            retry_reflection["host_rejections"],
            [
                {
                    "reason": "invalid_mutation_contract",
                    "rejection_code": "mutation_has_no_effect",
                    "validation_detail": (
                        "mutation operation select_registered_pipeline does not "
                        "change the pipeline"
                    ),
                    "rejected_operations": runtime.structured_results[0][
                        "operations"
                    ],
                }
            ],
        )
        mutation_contract = runtime.requests[1]["request"]["context"][
            "mutation_contract"
        ]
        self.assertEqual(
            mutation_contract["bounded_parameters"]["ridge_alpha"],
            {"type": "number", "minimum": 0.0001, "maximum": 1.0},
        )
        self.assertEqual(
            mutation_contract["policy"],
            "single_axis_reject_out_of_bounds_without_clamping",
        )
        self.assertEqual(mutation_contract["maximum_operations"], 1)
        self.assertEqual(
            mutation_contract["mutation_operator_id"],
            "bounded-trust-region-mutation@2",
        )

    def test_native_proposer_retries_a_non_compiling_predictor_binding(self) -> None:
        incompatible_mutation = {
            "schema_version": "ecologyrsi-dsh.genome-mutation/1",
            "operations": [
                {
                    "op": "select_registered_pipeline",
                    "predictor_id": "greenhouse-rolling-residual@1",
                }
            ],
        }
        compatible_mutation = {
            "schema_version": "ecologyrsi-dsh.genome-mutation/1",
            "operations": [
                {
                    "op": "select_registered_pipeline",
                    "predictor_id": "greenhouse-targetwise-ridge@1",
                }
            ],
        }
        runtime = _SequencedNativeRuntime(
            [incompatible_mutation, compatible_mutation]
        )
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        admission = DshToolService(ledger)
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: DshStructuredRoleRuntime(
                runtime,
                admission=admission,
            ),
        )
        director = EvolutionDirector(ledger, adapter)
        task = _native_task()
        run_id = "run:native-compile-retry"
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
                "batch_size": 1,
                "slot_index": 0,
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

        self.assertEqual(
            EcologyEvolutionPluginGenome.from_dict(
                json.loads(proposal.metadata["evolution_genome_canonical_json"])
            ).scientific_program["predictor_ref"]["id"],
            "greenhouse-targetwise-ridge@1",
        )
        self.assertEqual(len(runtime.requests), 2)
        rejection = runtime.requests[1]["request"]["context"][
            "evolution_reflection"
        ]["host_rejections"][0]
        self.assertEqual(
            rejection,
            {
                "reason": "compiled_behavior_validation_failed",
                "rejection_code": "predictor_evaluator_incompatible",
                "validation_detail": (
                    "predictor and evaluator bindings are incompatible"
                ),
                "rejected_operations": incompatible_mutation["operations"],
            },
        )

    def test_native_proposer_retries_a_sibling_behavior_before_evaluation(self) -> None:
        first_mutation = {
            "schema_version": "ecologyrsi-dsh.genome-mutation/1",
            "operations": [
                {
                    "op": "set_bounded_parameter",
                    "name": "ridge_alpha",
                    "value": 0.2,
                }
            ],
        }
        distinct_mutation = {
            "schema_version": "ecologyrsi-dsh.genome-mutation/1",
            "operations": [
                {
                    "op": "set_bounded_parameter",
                    "name": "ridge_alpha",
                    "value": 0.3,
                }
            ],
        }
        invalid_mutation = {
            "schema_version": "ecologyrsi-dsh.genome-mutation/1",
            "operations": [
                {
                    "op": "set_bounded_parameter",
                    "name": "not_a_registered_parameter",
                    "value": 0.5,
                }
            ],
        }
        runtime = _SequencedNativeRuntime(
            [
                first_mutation,
                first_mutation,
                invalid_mutation,
                distinct_mutation,
            ]
        )
        adapter = StrategyRouterDSHAdapter(
            gateway=object(), native_runtime_provider=lambda: runtime
        )
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger, adapter)
        task = _native_task()
        run_id = "run:native-sibling-retry"
        director.create_run(task, run_id=run_id)
        director.start_run(run_id)
        state = director.state(run_id)
        parent = state.materialized_seed_genome()
        shared = {
            "generation": 0,
            "batch_size": 2,
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
        }
        first = adapter.propose(
            state.run,
            task,
            state.run.session_id or "",
            batch_context={**shared, "slot_index": 0},
        )
        first_genome = first.metadata["evolution_genome_canonical_json"]
        first_behavior_digest = first.metadata["behavior_digest"]

        try:
            second = adapter.propose(
                state.run,
                task,
                state.run.session_id or "",
                batch_context={
                    **shared,
                    "slot_index": 1,
                    "sibling_candidate_behaviors": [
                        {
                            "slot_index": 0,
                            "prediction_model_id": (
                                "greenhouse-horizon-targetwise-ridge@1"
                            ),
                            "parameters": dict(first.changes),
                            "parameters_digest": digest(dict(first.changes)),
                            "genome_digest": first.metadata["genome_digest"],
                            "behavior_digest": first_behavior_digest,
                        }
                    ],
                },
            )
        except ValueError as exc:
            self.fail(f"validated sibling context was rejected: {exc}")

        self.assertNotEqual(
            second.metadata["evolution_genome_canonical_json"], first_genome
        )
        self.assertEqual(len(runtime.requests), 4)
        retry_reflection = runtime.requests[3]["request"]["context"][
            "evolution_reflection"
        ]
        self.assertEqual(
            retry_reflection["host_rejections"][0]["reason"],
            "sibling_behavior_duplicate",
        )
        self.assertEqual(
            retry_reflection["host_rejections"][1],
            {
                "reason": "invalid_mutation_contract",
                "rejection_code": "mutation_validation_failed",
                "validation_detail": (
                    "unregistered predictor parameter: "
                    "not_a_registered_parameter"
                ),
                "rejected_operations": invalid_mutation["operations"],
            },
        )

    def test_native_proposer_allows_historical_parameters_with_new_agent_behavior(
        self,
    ) -> None:
        first_mutation = {
            "schema_version": "ecologyrsi-dsh.genome-mutation/1",
            "operations": [
                {
                    "op": "set_bounded_parameter",
                    "name": "ridge_alpha",
                    "value": 0.2,
                }
            ],
        }
        agent_mutation = {
            "schema_version": "ecologyrsi-dsh.genome-mutation/1",
            "operations": [
                {
                    "op": "set_instruction_parameter",
                    "role": "sample-planner",
                    "name": "confidence_threshold",
                    "value": 0.8,
                },
            ],
        }
        runtime = _SequencedNativeRuntime([first_mutation, agent_mutation])
        adapter = StrategyRouterDSHAdapter(
            gateway=object(), native_runtime_provider=lambda: runtime
        )
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        director = EvolutionDirector(ledger, adapter)
        task = _native_task()
        run_id = "run:native-distinct-agent-behavior"
        director.create_run(task, run_id=run_id)
        director.start_run(run_id)
        state = director.state(run_id)
        parent = state.materialized_seed_genome()
        shared = {
            "generation": 0,
            "batch_size": 2,
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
        }
        first = adapter.propose(
            state.run,
            task,
            state.run.session_id or "",
            batch_context={**shared, "slot_index": 0},
        )
        plan = {
            "status": "model_generated",
            "dsh_evolution_reflection": {
                "schema_version": "ecologyrsi-dsh.evolution-reflection/2",
                "avoid_behaviors": [
                    {
                        "behavior_digest": first.metadata["behavior_digest"],
                        "prediction_model_id": (
                            "greenhouse-horizon-targetwise-ridge@1"
                        ),
                        "parameters_digest": digest(dict(first.changes)),
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
        second = adapter.propose(
            state.run,
            task,
            state.run.session_id or "",
            batch_context={
                **shared,
                "slot_index": 1,
                "knowledge_snapshot_digest": "2" * 64,
                "research_iteration": iteration.to_dict(),
                "stage_context_digests": {
                    "research_iteration_digest": iteration.iteration_digest,
                    "knowledge_snapshot_digest": "2" * 64,
                },
            },
        )

        self.assertEqual(
            second.changes,
            dict(parent.scientific_program["parameter_overrides"]),
        )
        self.assertNotEqual(
            second.metadata["behavior_digest"], first.metadata["behavior_digest"]
        )
        self.assertEqual(len(runtime.requests), 2)
        self.assertEqual(
            runtime.requests[1]["request"]["context"]["evolution_reflection"][
                "host_rejections"
            ],
            [],
        )

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
                "schema_version": "ecologyrsi-dsh.cross-generation-experience/3",
                "generations": [],
                "historical_generations": [
                    {
                        "source_run_id": "run:prior-failed",
                        "source_generation": 0,
                        "outcome": "no_eligible_candidate",
                        "improved": False,
                        "common_failures": ["scientific_gate_failed"],
                        "modifications": {
                            "candidate_parameter_sets": [failed_parameters],
                            "candidate_behaviors": [
                                {
                                    "behavior_digest": "a" * 64,
                                    "prediction_model_id": parent.to_dict()[
                                        "scientific_program"
                                    ]["predictor_ref"]["id"],
                                    "parameters_digest": digest(failed_parameters),
                                    "classification": "scientific_gate_failed",
                                }
                            ],
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

        avoid = plan["dsh_evolution_reflection"]["avoid_behaviors"]
        self.assertEqual(len(avoid), 1)
        self.assertEqual(avoid[0]["behavior_digest"], "a" * 64)
        self.assertEqual(avoid[0]["parameters_digest"], digest(failed_parameters))
        self.assertEqual(avoid[0]["reason"], "scientific_gate_failed")
        self.assertEqual(avoid[0]["source_run_id"], "run:prior-failed")

    def test_insufficient_diagnostic_evidence_is_not_a_hard_behavior_ban(
        self,
    ) -> None:
        reflection = _native_evolution_reflection_from_experience(
            {
                "schema_version": "ecologyrsi-dsh.cross-generation-experience/3",
                "generations": [],
                "historical_generations": [
                    {
                        "source_run_id": "run:diagnostic",
                        "source_generation": 0,
                        "outcome": "no_eligible_candidate",
                        "improved": False,
                        "common_failures": ["scientific_gate_failed"],
                        "gate_result": {
                            "candidate_count": 2,
                            "eligible_count": 0,
                            "outcome": "no_eligible_candidate",
                            "insufficient_evidence": True,
                            "constraint_failure_count": 0,
                            "judge_disagreement_count": 0,
                        },
                        "modifications": {
                            "candidate_behaviors": [
                                {
                                    "behavior_digest": "a" * 64,
                                    "prediction_model_id": (
                                        "greenhouse-horizon-targetwise-ridge@1"
                                    ),
                                    "parameters_digest": "b" * 64,
                                    "classification": "scientific_gate_failed",
                                }
                            ]
                        },
                    }
                ],
                "active_unresolved": [],
            },
            current_run_id="run:current",
        )

        self.assertEqual(reflection["avoid_behaviors"], [])

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
                "operations": [
                    {
                        "op": "set_bounded_parameter",
                        "name": "ridge_alpha",
                        "value": 0.2,
                    }
                ],
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
                "operations": [
                    {
                        "op": "set_bounded_parameter",
                        "name": "ridge_alpha",
                        "value": 0.2,
                    }
                ],
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
            context["research_iteration"]["source_digest"],
            iteration.iteration_digest,
        )
        reflection = context["evolution_reflection"]
        self.assertTrue(
            reflection["previous_generation_signals"]["scientific_gate_failed"]
        )
        self.assertTrue(reflection["research_signals"]["has_research_summary"])
        self.assertTrue(reflection["research_signals"]["has_previous_action"])
        self.assertNotIn("research_summary", reflection)
        self.assertNotIn("previous_next_action", reflection)

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
        failed_genome = parent.to_dict()
        failed_genome["scientific_program"]["parameter_overrides"] = failed_parameters
        for field_name in ("genome_id", "genome_digest", "behavior_digest"):
            failed_genome.pop(field_name, None)
        failed_behavior_digest = EcologyEvolutionPluginGenome.from_dict(
            failed_genome
        ).behavior_digest
        plan = {
            "status": "model_generated",
            "dsh_evolution_reflection": {
                "schema_version": "ecologyrsi-dsh.evolution-reflection/2",
                "avoid_behaviors": [
                    {
                        "behavior_digest": failed_behavior_digest,
                        "prediction_model_id": parent.to_dict()["scientific_program"][
                            "predictor_ref"
                        ]["id"],
                        "parameters_digest": digest(failed_parameters),
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

        self.assertEqual(len(runtime.requests), 4)
        retry_reflection = runtime.requests[3]["request"]["context"][
            "evolution_reflection"
        ]
        self.assertEqual(
            [item["reason"] for item in retry_reflection["host_rejections"]],
            ["exact_failed_behavior_replay"] * 3,
        )

    def test_native_proposal_uses_verified_genome_predictor_boundary(self) -> None:
        runtime = _NativeRuntime(
            {
                "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                "operations": [
                    {
                        "op": "set_bounded_parameter",
                        "name": "co2_concentration_1h_residual_scale",
                        "value": 0.1,
                    }
                ],
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
                "operations": [
                    {
                        "op": "set_bounded_parameter",
                        "name": "ridge_alpha",
                        "value": 0.2,
                    }
                ],
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
