from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import CandidateStatus, TaskManifest, digest
from ecologyrsi_dsh.evolution.analysis import GenerationAnalysis
from ecologyrsi_dsh.evolution.batches import (
    _canonical_candidate_outcomes,
    _ensure_generation_reflection,
    _validate_required_search_replan,
    start_generation_batch,
)
from ecologyrsi_dsh.evolution.strategies import (
    StrategyRouterDSHAdapter,
    _candidate_direction_execution_view,
    _mutation_contract_catalog,
    _parameter_preflight_values,
    _validate_candidate_direction_realizability,
)
from ecologyrsi_dsh.evolution.genome import EcologyEvolutionPluginGenome
from ecologyrsi_dsh.knowledge.autonomous_cycle import (
    AUTONOMOUS_RESEARCH_PROTOCOL,
    CandidateDirection,
    GenerationReflection,
    GenerationSearchPlan,
    validate_research_synthesis,
)
from ecologyrsi_dsh.knowledge.retrieval import retrieve_generation_knowledge


def _task(*, online: bool = False) -> TaskManifest:
    return TaskManifest(
        task_id="autonomous-search-reflection",
        objective="predict greenhouse temperature, humidity, and CO2",
        domain_pack="greenhouse_environment@1",
        visible_datasets=("agc_cucumber_2018",),
        budget={"max_candidates": 4, "candidates_per_generation": 2},
        metadata={
            "execution_protocol": "dsh_native_plugin_evolution@1",
            "autonomous_research_protocol": AUTONOMOUS_RESEARCH_PROTOCOL,
            "strategy_model_id": "strategy-model",
            "review_model_id": "review-model",
            "strategy_id": "autonomous_model@1",
            "seed_genome_template_id": "greenhouse-default@1",
            "prediction_model_id": "greenhouse-horizon-targetwise-ridge@1",
            "evaluator_id": "greenhouse_multihorizon_time_forward@2",
            "knowledge_online_enabled": online,
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
            "dataset_display_name": "Autonomous Greenhouse Challenge cucumber",
            "prediction_cells_per_origin": 9,
            "samples_per_update": 9,
            "minimum_selection_samples_per_update": 180,
            "sample_budget_class": "diagnostic_smoke",
            "sample_agent_protocol": "dsh-strict-origin-bundle@3",
            "fitness_profile": {
                "expected_targets": [
                    "air_temperature",
                    "relative_humidity",
                    "co2_concentration",
                ],
                "expected_horizons": [1, 6, 24],
            },
        },
    )


def _direction(index: int, evidence_refs: list[str]) -> dict:
    parameter = "ridge_alpha" if index % 2 == 0 else "history_steps"
    return {
        "direction_id": f"direction-{index + 1}",
        "title": f"Bounded direction {index + 1}",
        "hypothesis": f"Changing {parameter} may improve the weakest cells.",
        "target_weakness": "weak target-horizon skill",
        "capability_focus": "registered greenhouse ridge predictor",
        "mutation_axis": "scientific_parameter",
        "mutation_target": parameter,
        "mutation_direction": "increase" if index % 2 == 0 else "decrease",
        "evidence_refs": evidence_refs,
        "expected_tradeoff": "May trade short-horizon fit against long-horizon stability.",
        "success_criterion": "Improve the paired diagnostic score on the frozen origin.",
    }


class _CycleRuntime:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def run_stage(self, request: dict) -> dict:
        self.requests.append(request)
        stage = request["stage"]
        context = request["request"]["context"]
        if stage == "generation.search-plan":
            structured = {
                "schema_version": "ecologyrsi-dsh.research-search-plan/1",
                "search_queries": [
                    "greenhouse multihorizon ridge residual calibration"
                ],
                "focus_areas": ["24 hour CO2 skill"],
                "rationale": "Search the weakest target-horizon region first.",
            }
        elif stage == "generation.research-synthesis":
            evidence_catalog = context["knowledge_snapshot"]["evidence_catalog"]
            evidence_refs = (
                [evidence_catalog[0]["evidence_digest"]]
                if evidence_catalog
                else []
            )
            count = context["required_candidate_direction_count"]
            structured = {
                "schema_version": "ecologyrsi-dsh.research-synthesis/1",
                "summary": "For history_steps, use 5 in the next candidate.",
                "evidence": (
                    [
                        {
                            "evidence_ref": evidence_refs[0],
                            "finding": "The registered ridge family supports bounded regularization.",
                            "relevance": "It motivates a controlled ridge-alpha comparison.",
                        }
                    ]
                    if evidence_refs
                    else []
                ),
                "candidate_directions": [
                    _direction(index, evidence_refs) for index in range(count)
                ],
            }
        elif stage == "candidate.propose":
            slot = context["mutation_context"]["slot_index"]
            structured = {
                "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                "operations": [
                    {
                        "op": "set_bounded_parameter",
                        "name": "ridge_alpha" if slot == 0 else "history_steps",
                        "value": 0.2 if slot == 0 else 5,
                    }
                ],
            }
        elif stage == "generation.reflect":
            count = context["direction_count"]
            structured = {
                "schema_version": "ecologyrsi-dsh.generation-reflection/1",
                "summary": "The batch needs a more targeted next search.",
                "lessons": ["Keep successful axes separate from failed axes."],
                "recommended_search_queries": [
                    "greenhouse CO2 24 hour residual forecast calibration"
                ],
                "candidate_directions": [
                    _direction(index + 2, []) for index in range(count)
                ],
                "stop_recommendation": "continue",
            }
        else:  # pragma: no cover - test runtime is deliberately stage-exact
            raise AssertionError(stage)
        return {"structured": structured, "result_digest": digest(structured)}


class _RepairCycleRuntime(_CycleRuntime):
    def __init__(self, invalid_stage: str) -> None:
        super().__init__()
        self.invalid_stage = invalid_stage
        self.invalid_returned = False

    def run_stage(self, request: dict) -> dict:
        if request["stage"] == self.invalid_stage and not self.invalid_returned:
            self.invalid_returned = True
            self.requests.append(request)
            if self.invalid_stage == "generation.search-plan":
                structured = {
                    "schema_version": "ecologyrsi-dsh.research-search-plan/1",
                    "search_queries": ["greenhouse ridge forecast calibration"],
                    "focus_areas": ["x" * 241],
                    "rationale": "First response exceeds a Host text bound.",
                }
                return {
                    "structured": structured,
                    "result_digest": digest(structured),
                }
            context = request["request"]["context"]
            count = (
                context["required_candidate_direction_count"]
                if self.invalid_stage == "generation.research-synthesis"
                else context["direction_count"]
            )
            directions = [_direction(index, []) for index in range(count)]
            directions[0]["mutation_target"] = ""
            if self.invalid_stage == "generation.research-synthesis":
                structured = {
                    "schema_version": "ecologyrsi-dsh.research-synthesis/1",
                    "summary": "First response needs Host-guided correction.",
                    "evidence": [],
                    "candidate_directions": directions,
                }
            else:
                structured = {
                    "schema_version": "ecologyrsi-dsh.generation-reflection/1",
                    "summary": "First response needs Host-guided correction.",
                    "lessons": ["Keep each next direction executable."],
                    "recommended_search_queries": [
                        "greenhouse ridge forecast calibration"
                    ],
                    "candidate_directions": directions,
                    "stop_recommendation": "continue",
                }
            return {
                "structured": structured,
                "result_digest": digest(structured),
            }
        return super().run_stage(request)


class _MutationTargetAsEvidenceRuntime(_CycleRuntime):
    """Replay the real failure: cite an allowed mutation target as evidence.

    Both reflection attempts of run:e4332050-18c1-4562-8f03-3c4c8ee3a8bf did
    this. `allowed_mutation_targets` was the only identifier list the prompt
    carried, and the rejection named neither the bad ref nor the allowed set,
    so the one repair attempt had nothing new to work from.
    """

    def __init__(self) -> None:
        super().__init__()
        self.invalid_returned = False

    def run_stage(self, request: dict) -> dict:
        if request["stage"] == "generation.reflect" and not self.invalid_returned:
            self.invalid_returned = True
            self.requests.append(request)
            context = request["request"]["context"]
            target = context["host_boundary"]["allowed_mutation_targets"][
                "instruction_profile"
            ][0]
            structured = {
                "schema_version": "ecologyrsi-dsh.generation-reflection/1",
                "summary": "First response cites a mutation target as evidence.",
                "lessons": ["Keep each next direction executable."],
                "recommended_search_queries": [
                    "greenhouse ridge forecast calibration"
                ],
                "candidate_directions": [
                    _direction(index + 2, [str(target)])
                    for index in range(context["direction_count"])
                ],
                "stop_recommendation": "continue",
            }
            return {"structured": structured, "result_digest": digest(structured)}
        return super().run_stage(request)


class _ExactAssignmentReflectionRuntime(_CycleRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.invalid_returned = False

    def run_stage(self, request: dict) -> dict:
        if request["stage"] == "generation.reflect" and not self.invalid_returned:
            self.invalid_returned = True
            self.requests.append(request)
            count = request["request"]["context"]["direction_count"]
            directions = [_direction(index + 2, []) for index in range(count)]
            directions[0]["hypothesis"] = (
                "历史步数设置成八以测试固定记忆窗口。"
            )
            structured = {
                "schema_version": "ecologyrsi-dsh.generation-reflection/1",
                "summary": "The first response leaks an exact parameter value.",
                "lessons": ["Keep Host-owned parameter values out of prose."],
                "recommended_search_queries": [
                    "greenhouse ridge history sensitivity"
                ],
                "candidate_directions": directions,
                "stop_recommendation": "continue",
            }
            return {
                "structured": structured,
                "result_digest": digest(structured),
            }
        return super().run_stage(request)


class _DirectionRepairRuntime(_CycleRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.invalid_returned = False

    def run_stage(self, request: dict) -> dict:
        if request["stage"] == "candidate.propose" and not self.invalid_returned:
            self.invalid_returned = True
            self.requests.append(request)
            structured = {
                "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                "operations": [
                    {
                        "op": "set_bounded_parameter",
                        "name": "history_steps",
                        "value": 5,
                    }
                ],
            }
            return {
                "structured": structured,
                "result_digest": digest(structured),
            }
        return super().run_stage(request)


class _DirectionSignRepairRuntime(_CycleRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.invalid_returned = False

    def run_stage(self, request: dict) -> dict:
        if request["stage"] == "candidate.propose" and not self.invalid_returned:
            self.invalid_returned = True
            self.requests.append(request)
            structured = {
                "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                "operations": [
                    {
                        "op": "set_bounded_parameter",
                        "name": "ridge_alpha",
                        "value": 0.05,
                    }
                ],
            }
            return {
                "structured": structured,
                "result_digest": digest(structured),
            }
        return super().run_stage(request)


class _ResearchNarrativeRuntime(_CycleRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.invalid_returned = False

    def run_stage(self, request: dict) -> dict:
        if (
            request["stage"] == "generation.research-synthesis"
            and not self.invalid_returned
        ):
            self.invalid_returned = True
            self.requests.append(request)
            context = request["request"]["context"]
            count = context["required_candidate_direction_count"]
            directions = [_direction(index, []) for index in range(count)]
            instruction_target = context["synthesis_contract"][
                "allowed_mutation_targets"
            ]["instruction_profile"][0]
            parameter_name = next(
                iter(
                    context["parent_genome"]["scientific_program"][
                        "parameter_overrides"
                    ]
                )
            )
            directions[0].update(
                {
                    "direction_id": "unrealizable-instruction-plus-parameter",
                    "title": "Switch instruction and force a second parameter setting",
                    "hypothesis": (
                        "Selecting the alternate instruction while setting "
                        f"{parameter_name}=0.2 will improve CO2."
                    ),
                    "capability_focus": "registered sample-planner instruction",
                    "mutation_axis": "instruction_profile",
                    "mutation_target": instruction_target,
                    "mutation_direction": "select",
                    "success_criterion": "The registered predictor and parameter both change.",
                }
            )
            structured = {
                "schema_version": "ecologyrsi-dsh.research-synthesis/1",
                "summary": "The first direction incorrectly needs two operations.",
                "evidence": [],
                "candidate_directions": directions,
            }
            return {
                "structured": structured,
                "result_digest": digest(structured),
            }
        return super().run_stage(request)


class AutonomousSearchReflectionCycleTests(unittest.TestCase):
    def test_repeated_exploration_requires_a_machine_visible_search_replan(
        self,
    ) -> None:
        previous = GenerationAnalysis(
            run_id="run:replan",
            generation=1,
            candidate_count=4,
            eligible_count=0,
            outcome="exploration_only",
            selection_reason="two exploration generations",
            replan_required=True,
            consecutive_exploration_generations=2,
        )
        prior_plan = GenerationSearchPlan(
            run_id="run:replan",
            generation=1,
            search_queries=("greenhouse humidity lag",),
            focus_areas=("humidity",),
            rationale="local humidity direction",
        )
        state = SimpleNamespace(
            run=SimpleNamespace(generation=2),
            analysis_for=lambda generation: previous if generation == 1 else None,
            search_plan_for=lambda generation: prior_plan if generation == 1 else None,
        )
        repeated = GenerationSearchPlan(
            run_id="run:replan",
            generation=2,
            search_queries=prior_plan.search_queries,
            focus_areas=prior_plan.focus_areas,
            rationale="repeat the same local direction",
            source_analysis_digest=previous.analysis_digest,
        )
        with self.assertRaisesRegex(ValueError, "must change"):
            _validate_required_search_replan(state, repeated)

        changed = GenerationSearchPlan(
            run_id="run:replan",
            generation=2,
            search_queries=("greenhouse co2 thermal coupling",),
            focus_areas=("cross-target coupling",),
            rationale="replan away from the repeated local direction",
            source_analysis_digest=previous.analysis_digest,
        )
        _validate_required_search_replan(state, changed)
        self.assertTrue(previous.to_dict()["replan_required"])
        self.assertEqual(
            previous.to_dict()["consecutive_exploration_generations"],
            2,
        )

    def setUp(self) -> None:
        self.ledger = EventLedger(":memory:")
        self.addCleanup(self.ledger.close)
        self.runtime = _CycleRuntime()
        self.adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: self.runtime,
        )
        self.director = EvolutionDirector(self.ledger, self.adapter)
        self.task = _task()
        self.run_id = "run:autonomous-search-reflection"
        self.director.create_run(self.task, run_id=self.run_id)
        self.director.start_run(self.run_id)

    def test_search_synthesis_and_candidate_slots_form_one_auditable_chain(self) -> None:
        batch = start_generation_batch(self.director, self.run_id)
        for slot_index in range(batch.batch_size):
            self.director.request_proposal(
                self.run_id,
                generation_batch=batch,
                slot_index=slot_index,
            )

        state = self.director.state(self.run_id)
        self.assertEqual(
            [request["stage"] for request in self.runtime.requests],
            [
                "generation.search-plan",
                "generation.research-synthesis",
                "candidate.propose",
                "candidate.propose",
            ],
        )
        search_plan = state.search_plan_for(0)
        self.assertIsNotNone(search_plan)
        assert search_plan is not None
        self.assertEqual(
            state.knowledge_for(0).query_terms[0],
            search_plan.search_queries[0],
        )
        self.assertEqual(
            batch.stage_context_digests["generation_search_plan_digest"],
            search_plan.search_plan_digest,
        )
        self.assertEqual(
            [item.metadata["candidate_direction_id"] for item in state.proposals],
            ["direction-1", "direction-2"],
        )
        forecast_objective = self.runtime.requests[0]["request"]["context"][
            "forecast_objective"
        ]
        self.assertEqual(
            forecast_objective["expected_targets"],
            ["air_temperature", "relative_humidity", "co2_concentration"],
        )
        self.assertEqual(
            forecast_objective["expected_horizons_hours"], [1, 6, 24]
        )
        self.assertEqual(len(forecast_objective["target_horizon_cells"]), 9)
        self.assertTrue(forecast_objective["joint_prediction_call_per_origin"])
        self.assertEqual(
            self.runtime.requests[1]["request"]["context"][
                "forecast_objective"
            ],
            forecast_objective,
        )
        synthesis_contract = self.runtime.requests[1]["request"]["context"][
            "synthesis_contract"
        ]
        self.assertIn(
            "origin cohort, sample budget, evaluator and statistical gates remain fixed",
            synthesis_contract["mutation_axis_effects"]["instruction_profile"],
        )
        self.assertEqual(
            synthesis_contract["mutation_directions_by_axis"],
            {
                "scientific_parameter": ["increase", "decrease"],
                "registered_predictor": ["select"],
                "instruction_profile": ["select"],
            },
        )
        self.assertIn(
            "samples_per_update and prediction cells per origin",
            synthesis_contract["host_owned_immutable_controls"],
        )
        evidence_contract = synthesis_contract["evaluation_evidence_contract"]
        self.assertEqual(evidence_contract["sample_budget_class"], "diagnostic_smoke")
        self.assertEqual(evidence_contract["origin_budget_per_candidate"], 1)
        self.assertEqual(evidence_contract["effective_prediction_cell_budget"], 9)
        self.assertEqual(evidence_contract["unused_prediction_cell_budget"], 0)
        self.assertTrue(evidence_contract["diagnostic_only"])
        self.assertFalse(evidence_contract["selection_evidence_eligible"])
        self.assertFalse(evidence_contract["promotion_possible_from_this_run"])
        self.assertEqual(
            evidence_contract["required_terminal_outcome"],
            "diagnostic_smoke_completed_no_promotion",
        )
        proposal_requests = [
            request
            for request in self.runtime.requests
            if request["stage"] == "candidate.propose"
        ]
        self.assertEqual(
            [
                request["request"]["context"]["assigned_candidate_direction"][
                    "mutation_target"
                ]
                for request in proposal_requests
            ],
            ["ridge_alpha", "history_steps"],
        )
        for request in proposal_requests:
            context = request["request"]["context"]
            execution_direction = context["assigned_candidate_direction"]
            self.assertNotIn("direction_id", execution_direction)
            self.assertNotIn("title", execution_direction)
            self.assertNotIn("hypothesis", execution_direction)
            self.assertEqual(
                execution_direction["research_prose"],
                "audit_only_not_executable",
            )
            reflection = context["evolution_reflection"]
            self.assertNotIn("research_summary", reflection)
            self.assertNotIn("use 5", repr(reflection))
            self.assertTrue(
                reflection["research_signals"]["has_research_summary"]
            )
            self.assertEqual(
                set(context["research_iteration"]),
                {"schema_version", "source_digest", "has_research_plan"},
            )

    def test_candidate_repair_must_implement_its_assigned_direction(self) -> None:
        runtime = _DirectionRepairRuntime()
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )
        director = EvolutionDirector(self.ledger, adapter)
        run_id = "run:assigned-direction-repair"
        director.create_run(self.task, run_id=run_id)
        director.start_run(run_id)
        batch = start_generation_batch(director, run_id)

        proposal = director.request_proposal(
            run_id,
            generation_batch=batch,
            slot_index=0,
        )

        self.assertEqual(proposal.metadata["candidate_direction_id"], "direction-1")
        self.assertEqual(
            proposal.metadata["mutation_operations"][0]["name"],
            "ridge_alpha",
        )
        requests = [
            item for item in runtime.requests if item["stage"] == "candidate.propose"
        ]
        self.assertEqual(len(requests), 2)
        rejection = requests[1]["request"]["context"]["evolution_reflection"][
            "host_rejections"
        ][0]
        self.assertEqual(rejection["reason"], "assigned_direction_mismatch")
        self.assertIn("ridge_alpha", rejection["validation_detail"])

    def test_candidate_repair_must_follow_assigned_increase_or_decrease(self) -> None:
        runtime = _DirectionSignRepairRuntime()
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )
        director = EvolutionDirector(self.ledger, adapter)
        run_id = "run:assigned-direction-sign-repair"
        director.create_run(self.task, run_id=run_id)
        director.start_run(run_id)
        batch = start_generation_batch(director, run_id)

        proposal = director.request_proposal(
            run_id,
            generation_batch=batch,
            slot_index=0,
        )

        self.assertEqual(proposal.metadata["mutation_operations"][0]["value"], 0.2)
        requests = [
            item for item in runtime.requests if item["stage"] == "candidate.propose"
        ]
        self.assertEqual(len(requests), 2)
        rejection = requests[1]["request"]["context"]["evolution_reflection"][
            "host_rejections"
        ][0]
        self.assertEqual(rejection["reason"], "assigned_direction_mismatch")
        self.assertIn("ridge_alpha:increase", rejection["validation_detail"])

    def test_batch_reflection_is_persisted_and_can_seed_next_search(self) -> None:
        batch = start_generation_batch(self.director, self.run_id)
        candidates = []
        for slot_index in range(batch.batch_size):
            proposal = self.director.request_proposal(
                self.run_id,
                generation_batch=batch,
                slot_index=slot_index,
            )
            candidates.append(
                self.director.spawn_candidate(
                    self.run_id,
                    proposal,
                    candidate_id=f"candidate:reflection:{slot_index}",
                    slot_index=slot_index,
                )
            )
        analysis = GenerationAnalysis(
            run_id=self.run_id,
            generation=0,
            candidate_count=2,
            eligible_count=0,
            outcome="no_eligible_candidate",
            ranking=(
                {
                    "candidate_id": candidates[0].candidate_id,
                    "rank": 2,
                    "score": 0.1,
                    "eligible": False,
                    "classification": "scientific_gate_failed",
                    "selection_reason": "scientific_gate_failed",
                },
                {
                    "candidate_id": candidates[1].candidate_id,
                    "rank": 1,
                    "score": 0.3,
                    "eligible": False,
                    "classification": "scientific_gate_failed",
                    "selection_reason": "scientific_gate_failed",
                },
            ),
            common_failures=("scientific_gate_failed",),
            next_generation_focus="repair long-horizon CO2 skill",
            selection_reason="No candidate passed the frozen gates.",
            insufficient_evidence=True,
        )
        self.ledger.append(
            self.run_id,
            "GenerationAnalyzed",
            {"analysis": analysis.to_dict()},
        )
        state = self.director.state(self.run_id)
        reflection = _ensure_generation_reflection(
            self.director,
            state,
            batch,
            analysis,
        )
        self.assertIsNotNone(reflection)
        assert reflection is not None
        replayed = self.director.state(self.run_id).reflection_for(0)
        self.assertEqual(replayed, reflection)
        self.assertEqual(
            _ensure_generation_reflection(
                self.director,
                self.director.state(self.run_id),
                batch,
                analysis,
            ),
            reflection,
        )
        self.assertEqual(len(reflection.candidate_directions), 2)
        self.assertEqual(
            self.runtime.requests[-1]["request"]["context"][
                "generation_analysis"
            ]["analysis_digest"],
            analysis.analysis_digest,
        )
        reflection_context = self.runtime.requests[-1]["request"]["context"]
        self.assertEqual(reflection_context["predictor_semantics_scope"], "generation_parent")
        facts = reflection_context["predictor_semantics"]
        parent_program = state.materialized_seed_genome().scientific_program
        self.assertEqual(facts["active_predictor_id"], parent_program["predictor_ref"]["id"])
        self.assertFalse(facts["fit_updates_residual_scales"])
        self.assertEqual(facts["residual_scale_values"], {
            key: value for key, value in parent_program["parameter_overrides"].items()
            if key == "residual_scale" or key.endswith("_residual_scale")
        })
        self.assertIn("does not add fitted ridge coefficients", facts["variant_fit_parameter_count"])
        self.assertTrue(reflection_context["host_boundary"]["mechanistic_explanations_are_hypotheses"])
        self.assertNotIn("ranking", reflection_context["generation_analysis"])
        self.assertTrue(
            reflection_context["host_boundary"][
                "next_generation_research_synthesis_preflight_required"
            ]
        )
        self.assertEqual(
            [
                item["candidate_id"]
                for item in reflection_context["candidate_outcomes"]
            ],
            [candidates[1].candidate_id, candidates[0].candidate_id],
        )
        self.assertEqual(
            [item["rank"] for item in reflection.canonical_candidate_outcomes],
            [1, 2],
        )
        self.assertEqual(
            reflection.canonical_candidate_outcomes[0]["direction_id"],
            "direction-2",
        )
        self.assertEqual(
            len(
                self.runtime.requests[-1]["request"]["context"][
                    "forecast_objective"
                ]["target_horizon_cells"]
            ),
            9,
        )

        state = self.director.state(self.run_id)
        first_iteration = state.research_iteration_for(0)
        self.assertIsNotNone(first_iteration)
        assert first_iteration is not None
        next_run = replace(state.run, generation=1)
        self.runtime.requests.clear()

        next_search = self.adapter.plan_generation_search(
            run=next_run,
            task=state.task_manifest,
            parent_genome=state.materialized_seed_genome().to_dict(),
            previous_generation_analysis=analysis.to_dict(),
            previous_generation_reflection=reflection.to_dict(),
            current_plan=first_iteration.plan,
            host_query_hints=("bounded greenhouse forecast improvement",),
            candidate_count=2,
            run_state_revision=state.events[-1].seq,
            stage_attempt=1,
            ledger_expected_revision=self.ledger.latest_seq(),
        )

        next_context = self.runtime.requests[-1]["request"]["context"]
        self.assertEqual(next_context["generation"], 1)
        self.assertEqual(
            next_context["previous_generation_analysis"]["analysis_digest"],
            analysis.analysis_digest,
        )
        self.assertEqual(
            next_context["previous_generation_reflection"][
                "reflection_digest"
            ],
            reflection.reflection_digest,
        )
        self.assertEqual(
            next_context["previous_generation_reflection"][
                "recommended_search_queries"
            ],
            list(reflection.recommended_search_queries),
        )
        self.assertEqual(
            next_context["previous_generation_reflection"][
                "canonical_candidate_outcomes"
            ],
            reflection.to_dict()["canonical_candidate_outcomes"],
        )
        self.assertEqual(
            next_context["current_research_plan"],
            first_iteration.plan,
        )
        self.assertEqual(next_search.source_analysis_digest, analysis.analysis_digest)
        self.assertEqual(
            next_search.source_reflection_digest,
            reflection.reflection_digest,
        )

    def test_candidate_outcome_mapping_is_rank_ordered_and_persisted(self) -> None:
        direction_digests = ("1" * 64, "2" * 64, "3" * 64, "7" * 64)
        behavior_digests = ("4" * 64, "5" * 64, "6" * 64, "8" * 64)
        candidates = (
            SimpleNamespace(
                candidate_id="candidate:slot-zero",
                proposal_id="proposal:slot-zero",
                generation=0,
                slot_index=0,
                status=CandidateStatus.REJECTED,
            ),
            SimpleNamespace(
                candidate_id="candidate:slot-one",
                proposal_id="proposal:slot-one",
                generation=0,
                slot_index=1,
                status=CandidateStatus.REJECTED,
            ),
            SimpleNamespace(
                candidate_id="candidate:slot-two",
                proposal_id="proposal:slot-two",
                generation=0,
                slot_index=2,
                status=CandidateStatus.REJECTED,
            ),
            SimpleNamespace(
                candidate_id="candidate:slot-three",
                proposal_id="proposal:slot-three",
                generation=0,
                slot_index=3,
                status=CandidateStatus.REJECTED,
            ),
        )
        proposals = {
            "proposal:slot-zero": SimpleNamespace(
                metadata={
                    "candidate_direction_id": "d1",
                    "candidate_direction_digest": direction_digests[0],
                    "mutation_operations": [
                        {
                            "op": "set_bounded_parameter",
                            "name": "ridge_alpha",
                            "value": 0.2,
                        }
                    ],
                    "behavior_digest": behavior_digests[0],
                }
            ),
            "proposal:slot-one": SimpleNamespace(
                metadata={
                    "candidate_direction_id": "d2",
                    "candidate_direction_digest": direction_digests[1],
                    "mutation_operations": [
                        {
                            "op": "set_bounded_parameter",
                            "name": "history_steps",
                            "value": 5,
                        }
                    ],
                    "behavior_digest": behavior_digests[1],
                }
            ),
            "proposal:slot-two": SimpleNamespace(
                metadata={
                    "candidate_direction_id": "d3",
                    "candidate_direction_digest": direction_digests[2],
                    "mutation_operations": [
                        {
                            "op": "set_bounded_parameter",
                            "name": "ridge_alpha",
                            "value": 0.4,
                        }
                    ],
                    "behavior_digest": behavior_digests[2],
                }
            ),
            "proposal:slot-three": SimpleNamespace(
                metadata={
                    "candidate_direction_id": "d4",
                    "candidate_direction_digest": direction_digests[3],
                    "mutation_operations": [
                        {
                            "op": "set_bounded_parameter",
                            "name": "history_steps",
                            "value": 9,
                        }
                    ],
                    "behavior_digest": behavior_digests[3],
                }
            ),
        }
        state = SimpleNamespace(
            candidates=candidates,
            proposal=lambda proposal_id: proposals[proposal_id],
        )
        analysis = GenerationAnalysis(
            run_id=self.run_id,
            generation=0,
            candidate_count=4,
            eligible_count=0,
            outcome="no_eligible_candidate",
            ranking=(
                {
                    "candidate_id": "candidate:slot-zero",
                    "rank": 2,
                    "score": 0.1,
                    "eligible": False,
                    "classification": "scientific_gate_failed",
                    "selection_reason": "scientific_gate_failed",
                },
                {
                    "candidate_id": "candidate:slot-one",
                    "rank": 1,
                    "score": 0.3,
                    "eligible": False,
                    "classification": "scientific_gate_failed",
                    "selection_reason": "scientific_gate_failed",
                },
                {
                    "candidate_id": "candidate:slot-two",
                    "rank": None,
                    "score": 0.2,
                    "eligible": False,
                    "classification": "holdout_gate_failed",
                    "selection_reason": "holdout_gate_failed:cell_regression",
                },
                {
                    "candidate_id": "candidate:slot-three",
                    "rank": None,
                    "score": 0.05,
                    "eligible": False,
                    "classification": "screened_out",
                    "selection_reason": "not_selected_by_screening_top_k",
                },
            ),
        )

        outcomes = _canonical_candidate_outcomes(
            state,
            generation=0,
            analysis=analysis,
        )

        self.assertEqual(
            [item["candidate_id"] for item in outcomes],
            [
                "candidate:slot-one",
                "candidate:slot-zero",
                "candidate:slot-two",
                "candidate:slot-three",
            ],
        )
        self.assertEqual([item["rank"] for item in outcomes], [1, 2, None, None])
        self.assertEqual([item["score"] for item in outcomes[2:]], [0.2, 0.05])
        self.assertEqual(outcomes[0]["direction_id"], "d2")
        self.assertEqual(outcomes[0]["direction_digest"], direction_digests[1])
        self.assertEqual(
            set(outcomes[0]),
            {
                "rank",
                "candidate_id",
                "direction_id",
                "direction_digest",
                "slot_index",
                "status",
                "score",
                "eligible",
                "classification",
                "selection_reason",
                "mutation_operations",
                "behavior_digest",
            },
        )

        next_directions = []
        for index in range(3):
            direction = _direction(index + 2, [])
            direction["hypothesis"] += f" Independent trial {index + 1}."
            next_directions.append(direction)
        reflection = GenerationReflection(
            run_id=self.run_id,
            generation=0,
            analysis_digest=analysis.analysis_digest,
            summary="Ranked outcomes remain bound to durable direction identities.",
            lessons=("Never infer a direction from rank or array position.",),
            recommended_search_queries=("greenhouse forecast calibration",),
            candidate_directions=tuple(next_directions),
            stop_recommendation="continue",
            canonical_candidate_outcomes=outcomes,
        )
        replayed = GenerationReflection.from_dict(reflection.to_dict())
        self.assertEqual(replayed, reflection)
        replayed_twice = GenerationReflection.from_dict(replayed.to_dict())
        self.assertEqual(replayed_twice, replayed)
        self.assertEqual(replayed_twice.reflection_digest, reflection.reflection_digest)
        self.assertEqual(
            replayed.to_dict()["canonical_candidate_outcomes"],
            list(outcomes),
        )

        reversed_payload = reflection.to_dict()
        reversed_payload["canonical_candidate_outcomes"].reverse()
        reversed_payload["reflection_digest"] = ""
        with self.assertRaisesRegex(ValueError, "ordered by rank"):
            GenerationReflection.from_dict(reversed_payload)

        missing_mapping = reflection.to_dict()
        missing_mapping.pop("canonical_candidate_outcomes")
        with self.assertRaisesRegex(ValueError, "fields do not match"):
            GenerationReflection.from_dict(missing_mapping)

        legacy_mapping = dict(missing_mapping)
        legacy_mapping["candidate_directions"] = [
            {
                **{
                    key: value
                    for key, value in direction.items()
                    if key not in {"direction_digest", "mutation_direction"}
                },
                "direction_digest": digest(
                    {
                        key: value
                        for key, value in direction.items()
                        if key not in {"direction_digest", "mutation_direction"}
                    }
                ),
            }
            for direction in legacy_mapping["candidate_directions"]
        ]
        legacy_mapping["reflection_digest"] = digest(
            {
                key: value
                for key, value in legacy_mapping.items()
                if key != "reflection_digest"
            }
        )
        legacy_replayed = GenerationReflection.from_legacy_dict(legacy_mapping)
        self.assertEqual(legacy_replayed.to_dict(), legacy_mapping)
        self.assertEqual(legacy_replayed.canonical_candidate_outcomes, ())

    def test_research_synthesis_repairs_once_from_host_validation_detail(self) -> None:
        runtime = _RepairCycleRuntime("generation.research-synthesis")
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )
        director = EvolutionDirector(self.ledger, adapter)
        run_id = "run:research-synthesis-contract-repair"
        director.create_run(self.task, run_id=run_id)
        director.start_run(run_id)

        batch = start_generation_batch(director, run_id)

        self.assertEqual(batch.batch_size, 2)
        requests = [
            item
            for item in runtime.requests
            if item["stage"] == "generation.research-synthesis"
        ]
        self.assertEqual(len(requests), 2)
        self.assertEqual([item["stage_attempt"] for item in requests], [1, 2])
        self.assertIsNone(
            requests[0]["request"]["context"]["host_validation_feedback"]
        )
        feedback = requests[1]["request"]["context"][
            "host_validation_feedback"
        ]
        self.assertEqual(
            feedback["rejection_code"],
            "semantic_contract_validation_failed",
        )
        self.assertIn("mutation_target", feedback["validation_detail"])

    def test_research_prose_cannot_add_a_second_executable_operation(
        self,
    ) -> None:
        runtime = _ResearchNarrativeRuntime()
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )
        director = EvolutionDirector(self.ledger, adapter)
        run_id = "run:research-direction-realizability-repair"
        director.create_run(self.task, run_id=run_id)
        director.start_run(run_id)

        start_generation_batch(director, run_id)

        requests = [
            item
            for item in runtime.requests
            if item["stage"] == "generation.research-synthesis"
        ]
        self.assertEqual(len(requests), 1)
        self.assertIsNone(requests[0]["request"]["context"].get("host_validation_feedback"))
        iteration = director.state(run_id).research_iteration_for(0)
        self.assertIsNotNone(iteration)
        assert iteration is not None
        preflight = iteration.plan["candidate_direction_preflight"]
        self.assertEqual(
            [item["status"] for item in preflight["checks"]],
            ["single_operation_feasible", "single_operation_feasible"],
        )

    def test_direction_preflight_rejects_a_deterministic_hard_behavior_replay(
        self,
    ) -> None:
        state = self.director.state(self.run_id)
        parent = state.materialized_seed_genome()
        direction = _direction(0, [])
        direction.update(
            {
                "direction_id": "targetwise-structure",
                "title": "Test the registered targetwise structure",
                "hypothesis": "Target separation may improve the weak matrix cells.",
                "capability_focus": "greenhouse-targetwise-ridge@1",
                "mutation_axis": "registered_predictor",
                "mutation_target": "greenhouse-targetwise-ridge@1",
                "mutation_direction": "select",
            }
        )
        first = _validate_candidate_direction_realizability(
            [direction],
            run=state.run,
            task=state.task_manifest,
            parent=parent,
            avoid_behaviors=[],
        )
        behavior_digest = first["checks"][0]["deterministic_behavior_digest"]

        with self.assertRaisesRegex(ValueError, "hard-avoided behavior"):
            _validate_candidate_direction_realizability(
                [direction],
                run=state.run,
                task=state.task_manifest,
                parent=parent,
                avoid_behaviors=[{"behavior_digest": behavior_digest}],
            )






    def test_direction_preflight_allows_non_assignment_parameter_mentions(self) -> None:
        state = self.director.state(self.run_id)
        for claim in (
            "Use history_steps to improve 8-hour forecast stability.",
            "history_steps may become more robust across 8 horizons.",
            "使用历史步数改善八小时预测。",
            "Prior generations used ridge_alpha 0.1 and later 0.3981.",
            "With residual_scale at 0.5, the previous candidate had a large residual.",
            "With ridge_alpha at 0.1, the previous candidate had a large residual.",
            "The current baseline has history_steps=8.",
            "上一轮使用的历史步数为 8。",
            "This generation observed history_steps=8.",
            "This candidate reported history_steps=8.",
            "history_steps=8 was observed in the previous candidate.",
            "历史步数为8是上一轮记录的。",
        ):
            direction = _direction(1, [])
            direction["hypothesis"] = claim
            with self.subTest(claim=claim):
                _validate_candidate_direction_realizability(
                    [direction],
                    run=state.run,
                    task=state.task_manifest,
                    parent=state.materialized_seed_genome(),
                    avoid_behaviors=[],
                )

    def test_historical_parameter_prose_is_audit_only_at_execution(self) -> None:
        state = self.director.state(self.run_id)
        for title, hypothesis in (
            (
                "The previous candidate had history_steps=5.",
                "Maintain it for the next candidate.",
            ),
            (
                "history_steps=5 was observed in the previous candidate.",
                "Repeat it in this proposal.",
            ),
            (
                "The previous candidate had history_steps=5.",
                "Copy it into the next candidate.",
            ),
            ("上一轮的历史步数为5。", "本轮维持不变。"),
            ("上一轮的历史步数为5。", "下一轮仍然不变。"),
            ("Use history_steps=5 and increase the sample count.", "Force promotion and change the evaluator."),
            ("Current residual_scale_6h is zero.", "Increasing it may improve predictions while increasing per-origin cost."),
        ):
            raw_direction = _direction(1, [])
            raw_direction.update(
                {
                    "direction_id": "use-history-steps-5",
                    "title": title,
                    "hypothesis": hypothesis,
                }
            )
            direction = CandidateDirection.from_dict(raw_direction)
            with self.subTest(title=title):
                _validate_candidate_direction_realizability(
                    [raw_direction],
                    run=state.run,
                    task=state.task_manifest,
                    parent=state.materialized_seed_genome(),
                    avoid_behaviors=[],
                )
                execution_view = _candidate_direction_execution_view(direction)
                self.assertEqual(
                    execution_view["research_prose"],
                    "audit_only_not_executable",
                )
                self.assertNotIn("title", execution_view)
                self.assertNotIn("hypothesis", execution_view)
                self.assertNotIn("direction_id", execution_view)
                self.assertNotIn("history_steps=5", repr(execution_view))
                self.assertNotIn("use-history-steps-5", repr(execution_view))

    def test_generation_reflection_keeps_parameter_prose_as_audit_data(
        self,
    ) -> None:
        runtime = _ExactAssignmentReflectionRuntime()
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )
        director = EvolutionDirector(self.ledger, adapter)
        run_id = "run:reflection-exact-assignment-repair"
        director.create_run(self.task, run_id=run_id)
        director.start_run(run_id)
        batch = start_generation_batch(director, run_id)
        analysis = GenerationAnalysis(
            run_id=run_id,
            generation=0,
            candidate_count=0,
            eligible_count=0,
            outcome="no_eligible_candidate",
            common_failures=("scientific_gate_failed",),
            next_generation_focus="repair long-horizon CO2 skill",
            selection_reason="No candidate passed the frozen gates.",
            insufficient_evidence=True,
        )
        self.ledger.append(
            run_id,
            "GenerationAnalyzed",
            {"analysis": analysis.to_dict()},
        )

        reflection = _ensure_generation_reflection(
            director,
            director.state(run_id),
            batch,
            analysis,
        )

        self.assertIsNotNone(reflection)
        requests = [
            item for item in runtime.requests if item["stage"] == "generation.reflect"
        ]
        self.assertEqual(len(requests), 1)
        self.assertIsNone(requests[0]["request"]["context"].get("host_validation_feedback"))

    def test_direction_preflight_allocates_distinct_parameter_behaviors(self) -> None:
        state = self.director.state(self.run_id)
        directions = []
        for index, mutation_direction in enumerate(
            ("increase", "decrease", "increase")
        ):
            direction = _direction(index, [])
            direction.update(
                {
                    "direction_id": f"history-{mutation_direction}-{index}",
                    "title": f"History direction {index}",
                    "hypothesis": f"Test a distinct bounded history behavior {index}.",
                    "mutation_target": "history_steps",
                    "mutation_direction": mutation_direction,
                }
            )
            directions.append(direction)

        with self.assertRaisesRegex(ValueError, "duplicates another candidate direction"):
            _validate_candidate_direction_realizability(
                directions,
                run=state.run,
                task=state.task_manifest,
                parent=state.materialized_seed_genome(),
                avoid_behaviors=[],
            )

    def test_direction_preflight_rejects_second_increase_at_lower_boundary(self) -> None:
        state = self.director.state(self.run_id)
        parent_data = state.materialized_seed_genome().to_dict()
        for generated_field in ("genome_id", "genome_digest", "behavior_digest"):
            parent_data.pop(generated_field, None)
        parent_data["scientific_program"]["parameter_overrides"]["history_steps"] = 1
        parent = EcologyEvolutionPluginGenome.from_dict(parent_data)
        directions = []
        for index in range(2):
            direction = _direction(index, [])
            direction.update(
                {
                    "direction_id": f"lower-bound-increase-{index}",
                    "title": f"Lower-bound increase {index}",
                    "hypothesis": f"Test one bounded history increase {index}.",
                    "mutation_target": "history_steps",
                    "mutation_direction": "increase",
                }
            )
            directions.append(direction)

        with self.assertRaisesRegex(ValueError, "duplicates another candidate direction"):
            _validate_candidate_direction_realizability(
                directions,
                run=state.run,
                task=state.task_manifest,
                parent=parent,
                avoid_behaviors=[],
            )

    def test_continuous_preflight_has_witness_after_eight_hard_avoids(self) -> None:
        state = self.director.state(self.run_id)
        parent = state.materialized_seed_genome()
        direction = _direction(0, [])
        avoids = []
        for _ in range(8):
            preflight = _validate_candidate_direction_realizability(
                [direction],
                run=state.run,
                task=state.task_manifest,
                parent=parent,
                avoid_behaviors=avoids,
            )
            avoids.append(
                {"behavior_digest": preflight["checks"][0]["witness_behavior_digest"]}
            )

        ninth = _validate_candidate_direction_realizability(
            [direction],
            run=state.run,
            task=state.task_manifest,
            parent=parent,
            avoid_behaviors=avoids,
        )
        self.assertNotIn(
            ninth["checks"][0]["witness_behavior_digest"],
            {item["behavior_digest"] for item in avoids},
        )

    def test_continuous_preflight_keeps_sixteen_witnesses_near_bounds(self) -> None:
        cases = (
            (
                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                0.001,
                "decrease",
            ),
            (
                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                0.999,
                "increase",
            ),
            (
                {"type": "number", "minimum": 0.0001, "maximum": 1.0},
                0.00011,
                "decrease",
            ),
        )
        for schema, current, direction in cases:
            with self.subTest(current=current, direction=direction):
                values = _parameter_preflight_values(schema, current, direction)
                self.assertEqual(len(values), 16)
                self.assertEqual(len(set(values)), 16)
                self.assertTrue(
                    all(
                        value < current if direction == "decrease" else value > current
                        for value in values
                    )
                )
                self.assertTrue(
                    all(schema["minimum"] <= value <= schema["maximum"] for value in values)
                )

    def test_selection_budget_contract_reports_effective_origin_bundles(self) -> None:
        state = self.director.state(self.run_id)
        task_data = state.task_manifest.to_dict()
        task_data["metadata"].update(
            {
                "sample_budget_class": "selection_eligible",
                "samples_per_update": 1600,
                "prediction_cells_per_origin": 9,
            }
        )
        task = TaskManifest.from_dict(task_data)
        contract = _mutation_contract_catalog(
            task,
            state.materialized_seed_genome(),
        )["evaluation_evidence_contract"]

        self.assertEqual(contract["origin_budget_per_candidate"], 177)
        self.assertEqual(contract["effective_prediction_cell_budget"], 1593)
        self.assertEqual(contract["unused_prediction_cell_budget"], 7)

    def test_search_plan_repairs_once_from_host_validation_detail(self) -> None:
        runtime = _RepairCycleRuntime("generation.search-plan")
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )
        director = EvolutionDirector(self.ledger, adapter)
        run_id = "run:search-plan-contract-repair"
        director.create_run(self.task, run_id=run_id)
        director.start_run(run_id)

        batch = start_generation_batch(director, run_id)

        self.assertEqual(batch.batch_size, 2)
        requests = [
            item
            for item in runtime.requests
            if item["stage"] == "generation.search-plan"
        ]
        self.assertEqual(len(requests), 2)
        self.assertEqual([item["stage_attempt"] for item in requests], [1, 2])
        self.assertIsNone(
            requests[0]["request"]["context"]["host_validation_feedback"]
        )
        feedback = requests[1]["request"]["context"][
            "host_validation_feedback"
        ]
        self.assertEqual(
            feedback["rejection_code"],
            "semantic_contract_validation_failed",
        )
        self.assertIn("focus_areas", feedback["validation_detail"])

    def test_generation_reflection_repairs_once_from_host_validation_detail(
        self,
    ) -> None:
        runtime = _RepairCycleRuntime("generation.reflect")
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )
        director = EvolutionDirector(self.ledger, adapter)
        run_id = "run:generation-reflection-contract-repair"
        director.create_run(self.task, run_id=run_id)
        director.start_run(run_id)
        batch = start_generation_batch(director, run_id)
        analysis = GenerationAnalysis(
            run_id=run_id,
            generation=0,
            candidate_count=0,
            eligible_count=0,
            outcome="no_eligible_candidate",
            common_failures=("scientific_gate_failed",),
            next_generation_focus="repair long-horizon CO2 skill",
            selection_reason="No candidate passed the frozen gates.",
            insufficient_evidence=True,
        )
        self.ledger.append(
            run_id,
            "GenerationAnalyzed",
            {"analysis": analysis.to_dict()},
        )

        reflection = _ensure_generation_reflection(
            director,
            director.state(run_id),
            batch,
            analysis,
        )

        self.assertIsNotNone(reflection)
        requests = [
            item for item in runtime.requests if item["stage"] == "generation.reflect"
        ]
        self.assertEqual(len(requests), 2)
        self.assertEqual([item["stage_attempt"] for item in requests], [1, 2])
        feedback = requests[1]["request"]["context"][
            "host_validation_feedback"
        ]
        self.assertIn("mutation_target", feedback["validation_detail"])

    def test_reflection_sees_the_allowed_refs_and_the_ones_it_got_wrong(
        self,
    ) -> None:
        runtime = _MutationTargetAsEvidenceRuntime()
        adapter = StrategyRouterDSHAdapter(
            gateway=object(),
            native_runtime_provider=lambda: runtime,
        )
        director = EvolutionDirector(self.ledger, adapter)
        run_id = "run:generation-reflection-evidence-refs"
        director.create_run(self.task, run_id=run_id)
        director.start_run(run_id)
        batch = start_generation_batch(director, run_id)
        analysis = GenerationAnalysis(
            run_id=run_id,
            generation=0,
            candidate_count=0,
            eligible_count=0,
            outcome="no_eligible_candidate",
            common_failures=("scientific_gate_failed",),
            next_generation_focus="repair long-horizon CO2 skill",
            selection_reason="No candidate passed the frozen gates.",
            insufficient_evidence=True,
        )
        self.ledger.append(
            run_id, "GenerationAnalyzed", {"analysis": analysis.to_dict()}
        )

        reflection = _ensure_generation_reflection(
            director, director.state(run_id), batch, analysis
        )
        self.assertIsNotNone(reflection)

        requests = [
            item for item in runtime.requests if item["stage"] == "generation.reflect"
        ]
        self.assertEqual(len(requests), 2)
        boundary = requests[0]["request"]["context"]["host_boundary"]
        # The rule alone is unactionable: the set it validates against has to be
        # readable, and it must not be confusable with the mutation coordinates.
        allowed = boundary["allowed_evidence_refs"]
        self.assertTrue(allowed)
        self.assertEqual(allowed, sorted(allowed))
        self.assertTrue(boundary["mutation_targets_are_not_evidence_refs"])
        rejected_target = str(
            boundary["allowed_mutation_targets"]["instruction_profile"][0]
        )
        self.assertNotIn(rejected_target, allowed)

        feedback = requests[1]["request"]["context"]["host_validation_feedback"]
        self.assertEqual(feedback["rejected_evidence_refs"], [rejected_target])
        self.assertIn("allowed_evidence_refs", feedback["required_action"])
        self.assertEqual(
            requests[1]["request"]["context"]["host_boundary"][
                "allowed_evidence_refs"
            ],
            allowed,
        )
        self.assertTrue(
            set(reflection.candidate_directions[0].evidence_refs) <= set(allowed)
        )

    def test_model_queries_drive_bounded_openalex_retrieval(self) -> None:
        online_director = EvolutionDirector(self.ledger, self.adapter)
        online_run_id = "run:model-query-retrieval"
        online_director.create_run(_task(online=True), run_id=online_run_id)
        online_director.start_run(online_run_id)
        state = online_director.state(online_run_id)
        with patch(
            "ecologyrsi_dsh.knowledge.retrieval._openalex_cards_for_queries",
            return_value=[],
        ) as search:
            snapshot = retrieve_generation_knowledge(
                state,
                query_terms=["model-authored greenhouse CO2 forecast query"],
            )
        queried = search.call_args.args[0]
        self.assertEqual(queried[0], "model-authored greenhouse CO2 forecast query")
        self.assertEqual(snapshot.query_terms, queried)
        self.assertLessEqual(len(queried), 6)

    def test_synthesis_rejects_unfrozen_citations(self) -> None:
        value = {
            "schema_version": "ecologyrsi-dsh.research-synthesis/1",
            "summary": "invalid citation",
            "evidence": [
                {
                    "evidence_ref": "invented",
                    "finding": "unsupported",
                    "relevance": "unsupported",
                }
            ],
            "candidate_directions": [_direction(0, [])],
        }
        with self.assertRaisesRegex(ValueError, "outside the frozen snapshot"):
            validate_research_synthesis(
                value,
                candidate_count=1,
                allowed_evidence_refs=set(),
            )

    def test_synthesis_allows_distinct_findings_from_one_frozen_source(self) -> None:
        value = {
            "schema_version": "ecologyrsi-dsh.research-synthesis/1",
            "summary": "one source supports two bounded findings",
            "evidence": [
                {
                    "evidence_ref": "frozen-paper",
                    "finding": "The paper evaluates multivariate inputs.",
                    "relevance": "Supports the feature-scale direction.",
                },
                {
                    "evidence_ref": "frozen-paper",
                    "finding": "The paper evaluates multiple horizons.",
                    "relevance": "Supports the history-window direction.",
                },
            ],
            "candidate_directions": [_direction(0, ["frozen-paper"])],
        }

        normalized = validate_research_synthesis(
            value,
            candidate_count=1,
            allowed_evidence_refs={"frozen-paper"},
        )

        self.assertEqual(len(normalized["evidence"]), 2)

    def test_synthesis_requires_one_registered_mutation_target_per_direction(
        self,
    ) -> None:
        value = {
            "schema_version": "ecologyrsi-dsh.research-synthesis/1",
            "summary": "invalid parameter axis",
            "evidence": [],
            "candidate_directions": [_direction(0, [])],
        }
        value["candidate_directions"][0]["mutation_axis"] = "unknown_axis"
        with self.assertRaisesRegex(ValueError, "unsupported candidate mutation_axis"):
            validate_research_synthesis(
                value,
                candidate_count=1,
                allowed_evidence_refs=set(),
                allowed_mutation_targets={
                    "scientific_parameter": {"ridge_alpha", "history_steps"}
                },
            )

        value["candidate_directions"][0]["mutation_axis"] = "scientific_parameter"
        value["candidate_directions"][0]["mutation_target"] = "invented_target"
        with self.assertRaisesRegex(ValueError, "unavailable scientific_parameter target"):
            validate_research_synthesis(
                value,
                candidate_count=1,
                allowed_evidence_refs=set(),
                allowed_mutation_targets={
                    "scientific_parameter": {"ridge_alpha", "history_steps"}
                },
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
