"""New synthesis format remains scientifically complete and locally realizable."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.model_execution_policy import RESEARCH_EXECUTION_POLICY
from ecologyrsi_dsh.core.models import Run, digest
from ecologyrsi_dsh.evolution.genome import materialize_seed_genome, parameter_trust_region_neighborhood
from ecologyrsi_dsh.evolution.batches import start_generation_batch
from ecologyrsi_dsh.evolution.research_context import compact_research_context, research_report_size_diagnostics
from ecologyrsi_dsh.evolution.strategies import (
    StrategyRouterDSHAdapter, _forecast_objective_context, _mutation_contract_catalog,
    _predictor_semantics, _validate_candidate_direction_realizability,
)
from ecologyrsi_dsh.knowledge.autonomous_cycle import GenerationSearchPlan, validate_research_synthesis
from ecologyrsi_dsh.knowledge.models import KnowledgeCard, KnowledgeSnapshot
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from tests.test_autonomous_search_reflection_cycle import _task
from tests.test_evolution_genome import _initialization


def minimal_aligned_synthesis():
    directions = []
    for index, (horizon, target) in enumerate(((1, "temperature"), (6, "humidity"), (24, "CO2"), (1, "humidity"))):
        directions.append({
            "direction_id": f"direction-{index + 1}", "title": f"{horizon}h {target} correction",
            "hypothesis": f"Enabling {horizon}h residual correction may improve {target} skill.",
            "target_weakness": f"Uncertain {target} residual benefit at {horizon}h.",
            "capability_focus": "Registered aligned ridge",
            "mutation_axis": "scientific_parameter", "mutation_target": f"residual_scale_{horizon}h",
            "mutation_direction": "increase", "evidence_refs": [],
            "expected_tradeoff": "Shared correction can worsen other targets.",
            "success_criterion": "Compare paired cell skill and the full forecast matrix.",
        })
    return {"schema_version": "ecologyrsi-dsh.research-synthesis/1",
            "summary": "Test horizon corrections across the full matrix. Distinct same-axis values remain proposer-owned.",
            "evidence": [], "candidate_directions": directions}


class ResearchContextTests(unittest.TestCase):
    def setUp(self):
        self.task = replace(_task(), metadata={**_task().metadata,
            "prediction_model_id": "greenhouse-baseline-aligned-ridge@1",
            "evaluator_id": "greenhouse_multihorizon_time_forward@3",
            "seed_genome_template_id": "greenhouse-baseline-aligned-default@1"})
        self.run = Run(run_id="run:research-context", task_id=self.task.task_id, task_manifest_digest=self.task.digest)
        registry = current_program_registry()
        self.parent = materialize_seed_genome(registry.seed_template("greenhouse-baseline-aligned-default@1"),
                                              _initialization(task_manifest_digest=self.task.digest))
        self.schemas = StrategyRouterDSHAdapter._GREENHOUSE_ALIGNED_RIDGE_SCHEMAS
        self.search = GenerationSearchPlan(run_id=self.run.run_id, generation=0,
            search_queries=("registered greenhouse forecasting",), focus_areas=("full matrix",), rationale="Test causal residuals.")
        card = KnowledgeCard(knowledge_id="frozen-paper", title="Frozen evidence", summary="Metadata supports a hypothesis only.",
            source_url="https://example.org/paper", source_kind="paper", source_authority="metadata",
            execution_status="metadata_only", selection_reason="Relevant to residual forecasting.")
        self.knowledge = KnowledgeSnapshot(run_id=self.run.run_id, generation=0, query_terms=self.search.search_queries,
            cards=(card,), online_enabled=False, provider="frozen", retrieval_status="catalog_only").proposal_context()
        self.context = {
            "generation": 0, "objective": self.task.objective, "forecast_objective": _forecast_objective_context(self.task),
            "parent_genome": self.parent.to_dict(), "parent_plan": {}, "knowledge_snapshot": self.knowledge,
            "generation_search_plan": self.search.to_dict(), "required_candidate_direction_count": 4,
            "synthesis_contract": {**_mutation_contract_catalog(self.task, self.parent),
                                   "predictor_semantics": _predictor_semantics(self.parent),
                                   "prior_failure_advice": [{"reason": "scientific_gate_failed", "behavior_digest": "a" * 64}]},
            "cross_generation_experience": {"capacity": {"max_generation_summaries": 6},
                "historical_generations": [{"common_failures": ["insufficient_evidence"], "gate_result": {"insufficient_evidence": True}}]},
        }

    def test_four_zero_seed_directions_are_concise_and_have_distinct_legal_behaviors(self):
        output = minimal_aligned_synthesis()
        normalized = validate_research_synthesis(output, candidate_count=4, allowed_evidence_refs=set(),
            allowed_mutation_targets=self.context["synthesis_contract"]["allowed_mutation_targets"])
        research_report_size_diagnostics(normalized)
        checks = _validate_candidate_direction_realizability(normalized["candidate_directions"],
            run=self.run, task=self.task, parent=self.parent, avoid_behaviors=[])["checks"]
        self.assertEqual(len({check["witness_behavior_digest"] for check in checks}), 4)
        self.assertEqual(checks[0]["mutation_target"], checks[3]["mutation_target"])
        self.assertLess(len(json.dumps(output, separators=(",", ":"))), 3000)
        duplicate = deepcopy(output)
        duplicate["candidate_directions"][3]["hypothesis"] = duplicate["candidate_directions"][0]["hypothesis"]
        with self.assertRaisesRegex(ValueError, "hypotheses must be distinct"):
            validate_research_synthesis(duplicate, candidate_count=4, allowed_evidence_refs=set())

    def test_compaction_preserves_evidence_scientific_boundaries_and_failure_advice(self):
        original = deepcopy(self.context)
        compact = compact_research_context(self.context, parent=self.parent, parameter_schemas=self.schemas,
                                            policy=RESEARCH_EXECUTION_POLICY)
        self.assertEqual(self.context, original)
        self.assertEqual(compact["source_context_digest"], digest(original))
        self.assertEqual(compact["knowledge_snapshot"]["evidence_catalog"], self.knowledge["evidence_catalog"])
        self.assertEqual(compact["parent_program"]["scientific_program"], self.parent.to_dict()["scientific_program"])
        self.assertEqual(compact["parent_program"]["agent_program"], self.parent.to_dict()["agent_program"])
        self.assertEqual(compact["synthesis_contract"]["prior_failure_advice"], original["synthesis_contract"]["prior_failure_advice"])
        self.assertEqual(compact["cross_generation_experience"]["historical_generations"], original["cross_generation_experience"]["historical_generations"])
        self.assertEqual(compact["forecast_objective"]["expected_targets"], original["forecast_objective"]["expected_targets"])
        self.assertEqual(compact["forecast_objective"]["expected_horizons_hours"], [1, 6, 24])
        table = compact["synthesis_contract"]["scientific_parameter_boundaries"]
        for name, row in table["rows_by_parameter"].items():
            fields = dict(zip(table["columns"], row))
            step = parameter_trust_region_neighborhood(name=name, previous=self.parent.scientific_program["parameter_overrides"][name], contract=self.schemas[name])
            self.assertEqual(fields["one_step_minimum"], step["minimum"])
            self.assertEqual(fields["one_step_maximum"], step["maximum"])
            self.assertTrue(fields["agent_policy_mutation_allowed"])

    def test_new_policy_only_changes_synthesis_context_budget_and_report_bounds(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                task = replace(self.task, metadata={**self.task.metadata,
                    **({"research_execution_policy": dict(RESEARCH_EXECUTION_POLICY)} if enabled else {})})
                parent = materialize_seed_genome(current_program_registry().seed_template("greenhouse-baseline-aligned-default@1"),
                                                _initialization(task_manifest_digest=task.digest))
                captured = []
                def invoke(**request):
                    captured.append(request)
                    output = minimal_aligned_synthesis()
                    if enabled:
                        output["summary"] = "x" * 1362
                    return output
                adapter = StrategyRouterDSHAdapter(gateway=object())
                with patch.object(adapter, "_native_runtime", return_value=SimpleNamespace(run=invoke)):
                    result = adapter.research_plan("offline", run=self.run, task=task, parent_genome=parent.to_dict(),
                        knowledge_snapshot=self.knowledge, generation_search_plan=self.search.to_dict(), candidate_count=4)
                self.assertEqual(len(captured), 1)
                self.assertEqual(captured[0]["max_tokens"], 16384 if enabled else 8192)
                self.assertEqual("research_execution_policy" in captured[0]["context"], enabled)
                self.assertEqual("parent_genome" in captured[0]["context"], not enabled)
                self.assertEqual(len(result["candidate_direction_preflight"]["checks"]), 4)
                if enabled:
                    self.assertEqual(result["dsh_research_summary"], "x" * 1362)
                    self.assertEqual(result["report_size_advisories"], [{"field": "summary", "actual": 1362, "recommended_maximum": 1200}])

    def test_concise_limits_apply_after_unchanged_historical_scientific_schema(self):
        output = minimal_aligned_synthesis()
        output["summary"] = "x" * 1201
        normalized = validate_research_synthesis(output, candidate_count=4, allowed_evidence_refs=set())
        self.assertEqual(research_report_size_diagnostics(normalized), [{"field": "summary", "actual": 1201, "recommended_maximum": 1200}])
        self.assertEqual(normalized["summary"], output["summary"])
        output["summary"] = "x" * 12001
        with self.assertRaises(ValueError):
            validate_research_synthesis(output, candidate_count=4, allowed_evidence_refs=set())

    def test_same_axis_proposer_sees_occupied_value_and_can_repair_in_a_fresh_child(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), EventLedger(":memory:") as ledger:
                requests = []
                def run_stage(request):
                    requests.append(request)
                    stage, context = request["stage"], request["request"]["context"]
                    if stage == "generation.search-plan":
                        structured = {"schema_version": "ecologyrsi-dsh.research-search-plan/1",
                            "search_queries": ["greenhouse causal forecasting"], "focus_areas": ["full matrix"],
                            "rationale": "Compare registered residual corrections."}
                    elif stage == "generation.research-synthesis":
                        structured = minimal_aligned_synthesis()
                    elif stage == "candidate.propose":
                        slot = context["mutation_context"]["slot_index"]
                        # Deliberately return the duplicate once. A fresh child
                        # must receive the rejected value, rather than an opaque
                        # digest it cannot use to choose a replacement.
                        value = .05 if slot == 3 and context["evolution_reflection"]["host_rejections"] else .1
                        structured = {"schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": [
                            {"op": "set_bounded_parameter", "name": f"residual_scale_{(1, 6, 24, 1)[slot]}h", "value": value}]}
                    else:
                        raise AssertionError(stage)
                    return {"structured": structured, "result_digest": digest(structured)}
                task = replace(self.task, budget={"max_candidates": 4, "candidates_per_generation": 4},
                    metadata={**self.task.metadata,
                              **({"research_execution_policy": dict(RESEARCH_EXECUTION_POLICY)} if enabled else {})})
                adapter = StrategyRouterDSHAdapter(gateway=object(), native_runtime_provider=lambda: SimpleNamespace(run_stage=run_stage))
                director = EvolutionDirector(ledger, adapter)
                run_id = f"run:proposal-exclusions-{enabled}"
                director.create_run(task, run_id=run_id)
                director.start_run(run_id)
                batch = start_generation_batch(director, run_id)
                for slot in range(4):
                    proposal = director.request_proposal(run_id, generation_batch=batch, slot_index=slot)
                    director.spawn_candidate(run_id, proposal, slot_index=slot)
                final_requests = [request["request"]["context"] for request in requests
                                  if request["stage"] == "candidate.propose"
                                  and request["request"]["context"]["mutation_context"]["slot_index"] == 3]
                self.assertEqual(len(final_requests), 2)
                contract = final_requests[0]["mutation_contract"]
                rejection = final_requests[1]["evolution_reflection"]["host_rejections"][0]
                self.assertEqual(rejection["reason"], "sibling_behavior_duplicate")
                self.assertEqual("sibling_parameter_exclusions" in contract, enabled)
                self.assertEqual("rejected_operations" in rejection, enabled)
                if enabled:
                    self.assertEqual([item["value"] for item in contract["sibling_parameter_exclusions"]], [.1])
                    self.assertEqual(contract["sibling_parameter_exclusions"][0]["slot_index"], 0)
                    self.assertEqual(rejection["rejected_operations"][0]["value"], .1)
                self.assertEqual(proposal.changes["residual_scale_1h"], .05)
