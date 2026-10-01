"""Skill mutations must reach native planning, receipts and bounded exploration."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.evaluators.skill_program import (
    SKILL_POLICY_ID, execute_skill_program, seed_skill_program, validate_skill_program,
)
from ecologyrsi_dsh.evaluators.sample_execution import _validated_result, _attempt_trace_entry
from ecologyrsi_dsh.evolution.diversity import (
    DIVERSITY_POLICY, FAMILY_AXES, direction_schedule, validate_direction_schedule,
    preregister_candidate, summarize_behavior, exploration_archive,
)
from ecologyrsi_dsh.evolution.genome import apply_genome_mutation, EcologyEvolutionPluginGenome
from ecologyrsi_dsh.evolution.mutation_specs import mutation_coordinates
from ecologyrsi_dsh.evolution.workflow_ir import (
    resolve_candidate_agent_profile,
    compile_dsh_workflow_spec,
)
from ecologyrsi_dsh.knowledge.autonomous_cycle import CANDIDATE_MUTATION_AXES
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from tests.test_authored_directive import _seed_genome, _mutation_context, _directive
from tests import test_agent_owned_prediction as agents
from tests import test_research_context as research
from tests import test_autonomous_search_reflection_cycle as cycle
from tests.test_evolution_genome import _initialization
from ecologyrsi_dsh.evolution.genome import materialize_seed_genome
from ecologyrsi_dsh.evolution.strategies import (
    StrategyRouterDSHAdapter,
    _deterministic_fallback_directions,
)


def skill_child(program=None):
    parent = _seed_genome()
    child = apply_genome_mutation(parent, {"schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": [
        {"op": "author_skill_program", "role": "sample-planner", "skill_program": program or seed_skill_program()}]},
        _mutation_context(parent), current_program_registry())
    return parent, child


class SkillProgramTests(unittest.TestCase):
    def test_mutation_compiles_exports_and_round_trips_without_changing_legacy_parent(self):
        parent, child = skill_child()
        profile = resolve_candidate_agent_profile(child, current_program_registry())
        self.assertNotIn("skill_program", resolve_candidate_agent_profile(parent, current_program_registry()))
        self.assertEqual(profile["skill_program_digest"], digest(seed_skill_program()))
        self.assertNotEqual(parent.behavior_digest, child.behavior_digest)
        self.assertEqual(EcologyEvolutionPluginGenome.from_dict(child.to_dict()).to_dict(), child.to_dict())
        self.assertEqual(mutation_coordinates({"op": "author_skill_program", "role": "sample-planner"}),
                         ("skill_program", SKILL_POLICY_ID, "skill-program:sample-planner"))

    def test_program_ref_cannot_be_forged(self):
        _, child = skill_child()
        data = child.to_dict()
        execution = data["agent_program"]["candidate_execution_program"]
        profile = next(p for p in execution["role_profiles"] if p["role"] == "sample-planner")
        profile["skill_policy_ref"]["catalog_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            compile_dsh_workflow_spec(execution["workflow_template_ref"], execution["workflow_overrides"], execution["role_profiles"], current_program_registry())

    def test_rejects_unregistered_code_oversize_duplicate_and_empty_programs(self):
        valid = seed_skill_program()
        cases = [{"steps": []}, {**valid, "code": "arbitrary"}, {**valid, "schema_version": "wrong"},
                 {"steps": valid["steps"] * 2}]
        for field, value in (("module_id", "arbitrary@1"), ("when", "future_label"), ("guidance", "x" * 361)):
            case = deepcopy(valid); case["steps"][0][field] = value; cases.append(case)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_skill_program(case)

    def test_numeric_timestamps_are_sorted_and_future_scoring_fields_are_ignored(self):
        context = {"history_window": [4., 1., 3.], "causal_provenance": {"history_timestamps": [3, 1, 2]}}
        receipt = execute_skill_program(seed_skill_program(), context, 1)
        output = receipt["steps"][0]["output"]
        self.assertEqual(output["latest_value"], 4.)
        self.assertEqual(output["recent_change"], 1.)
        self.assertEqual(receipt, execute_skill_program(seed_skill_program(), {**context, "observed": 999, "future": [99]}, 1))

    def test_conditional_workflow_activates_only_applicable_steps(self):
        program = seed_skill_program("horizon-routing@1")
        program["steps"][0]["when"] = "long_horizon"
        self.assertEqual(execute_skill_program(program, {}, 1)["steps"][0]["status"], "skipped")
        self.assertEqual(execute_skill_program(program, {}, 24)["steps"][0]["output"]["branch"], "long")
        missing = execute_skill_program(seed_skill_program(), {"history_window": [1., None, 3.],
            "causal_provenance": {"history_timestamps": [1, 2, 3]}}, 1)["steps"][0]["output"]
        self.assertIsNone(missing["recent_change"])

    def test_missing_provenance_does_not_invent_a_trend(self):
        output = execute_skill_program(seed_skill_program(), {"history_window": [1, 2]}, 1)["steps"][0]["output"]
        self.assertFalse(output["timestamp_order_verified"])
        self.assertIsNone(output["latest_value"])


class SkillNativeExecutionTests(unittest.TestCase):
    setup_agent = agents.AgentOwnedPredictionTests.setup_agent
    predict = agents.AgentOwnedPredictionTests.predict

    def test_skill_reaches_real_native_context_and_receipt_without_extra_model_calls(self):
        _, child = skill_child()
        profile = resolve_candidate_agent_profile(child, current_program_registry())
        def policy(context, call):
            receipt = context["context"]["skill_execution"][context["samples"][0]["sample_id"]]
            self.assertEqual(receipt["program_digest"], profile["skill_program_digest"])
            self.assertEqual(receipt["steps"][0]["output"]["latest_value"], 20.)
            call("candidate-model", "model")
            return agents.result(context, method="model", refs=("model",))
        adapter, _, native, ledger = self.setup_agent(policy)
        plan = adapter.plan_batch({"candidate_agent_profile": profile})
        outcome = self.predict(adapter, plan)
        self.assertIsNone(outcome.error, str(outcome.error))
        validated = _validated_result(outcome.result)
        trace = _attempt_trace_entry(1, validated["agent_decisions"], validated["tool_calls"], outcome="accepted")
        self.assertEqual(len(trace["skill_evidence"]), 1)
        self.assertEqual(len(trace["model_evidence"]), 1)
        self.assertIn("behavior_signature", trace["skill_evidence"][0])
        self.assertEqual(len(native.requests), 1)
        self.assertEqual(len(ledger.events_by_kind("run-agent", "DshPredictionToolExecuted")), 1)

    def test_skill_receipts_survive_full_executor_with_strict_agent_provenance(self):
        _, child = skill_child()
        profile = resolve_candidate_agent_profile(child, current_program_registry())
        def policy(context, call):
            call("candidate-model", "model")
            return agents.result(context, method="model", refs=("model",))
        adapter, _, _, _ = self.setup_agent(policy)
        row = agents._request("origin-10").to_dict()
        row.update(origin_timestamp=10, target_timestamp=11, observed=21.)
        row["label_free_context"]["causal_provenance"].update(
            origin_cutoff_timestamp=10, latest_context_timestamp=10, history_timestamps=[9, 10])
        batch = agents.CollaborativeSampleExecutor(adapter).execute(
            [row], context={"run_id": "run-agent", "candidate_id": "candidate-1", "dataset_digest": "d" * 64,
                "partition": "training_feedback", "algorithm_id": "registered-predictor", "algorithm_version": "1",
                "sample_concurrency": 1, "candidate_agent_profile": profile},
            target_bounds={"air_temperature": {"minimum": -20., "maximum": 80.}},
            policy=agents.SampleExecutionPolicy(), algorithm_id="registered-predictor", algorithm_version="1")
        self.assertTrue(batch.summary["strict_agent_chain_pass"])
        self.assertTrue(batch.records[0]["attempt_trace"][0]["skill_evidence"])


class DiversityPolicyTests(unittest.TestCase):
    def test_skill_training_revises_skills_and_rejects_parameter_drift(self):
        from ecologyrsi_dsh.application.formal_trajectory import _local_edit_context
        from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
        from ecologyrsi_dsh.evolution.local_edits import (
            LocalEditProposal,
            apply_or_reject_local_edit_bundle,
        )
        _, parent = skill_child()
        task = cycle._task()
        task = replace(task, metadata={**task.metadata, "evolution_diversity_policy": DIVERSITY_POLICY,
                                      "optimization_schedule": OptimizationSchedule.default().to_dict()})
        revision = NS(revision_id="r0", genome=parent.to_dict(), genome_digest=parent.genome_digest)
        state = NS(run=NS(run_id="run:authored-directive"), task_manifest=task,
            proposal=lambda _: NS(metadata={"candidate_direction": {"mutation_axis": "skill_program"}}),
            batch_evaluation_for=lambda *_: NS(scope=NS(scope_key="a" * 64)))
        candidate = NS(candidate_id="c1", proposal_id="p1", generation=0)
        context = _local_edit_context(state, candidate, revision, NS(batch_index=0))
        self.assertEqual({axis for axis, targets in context.allowed_mutation_targets.items() if targets}, {"skill_program"})
        def edit(operation):
            proposal = LocalEditProposal(decision="mutate", operations=(operation,), evidence_refs=("batch:score",),
                expected_effect_cells=("air_temperature@1h",), risk_cells=())
            return apply_or_reject_local_edit_bundle(parent, proposal, context, current_program_registry())
        changed = edit({"op": "author_skill_program", "role": "sample-planner", "skill_program": seed_skill_program("trend-check@1")})
        self.assertEqual(changed.outcome.value, "applied")
        self.assertNotEqual(changed.child.behavior_digest, parent.behavior_digest)
        self.assertEqual(edit({"op": "set_bounded_parameter", "name": "ridge_alpha", "value": .2}).outcome.value, "rejected")

    def test_native_research_to_prompt_and_skill_proposals_replays(self):
        task = cycle._task()
        task = replace(task, metadata={**task.metadata, "evolution_diversity_policy": DIVERSITY_POLICY})
        class Runtime(cycle._CycleRuntime):
            def run_stage(self, request):
                context = request["request"]["context"]
                if request["stage"] == "generation.research-synthesis":
                    contract = context["synthesis_contract"]
                    directions = _deterministic_fallback_directions(contract["allowed_mutation_targets"], 2,
                        family_schedule=contract["diversity_schedule"])
                    directions[0].update(mutation_axis="instruction_directive",
                        mutation_target=contract["allowed_mutation_targets"]["instruction_directive"][0], mutation_direction="author")
                    structured = {"schema_version": "ecologyrsi-dsh.research-synthesis/1",
                        "summary": "Test prompt and skill strategies.", "evidence": [], "candidate_directions": directions}
                elif request["stage"] == "candidate.propose":
                    axis = context["assigned_candidate_direction"]["mutation_axis"]
                    operation = ({"op": "author_role_directive", "role": "sample-planner", "authored_directive": _directive()}
                        if axis == "instruction_directive" else
                        {"op": "author_skill_program", "role": "sample-planner", "skill_program": seed_skill_program()})
                    structured = {"schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": [operation]}
                else:
                    return super().run_stage(request)
                self.requests.append(request)
                return {"structured": structured, "result_digest": digest(structured)}
        runtime = Runtime()
        with cycle.EventLedger(":memory:") as ledger:
            adapter = StrategyRouterDSHAdapter(gateway=object(), native_runtime_provider=lambda: runtime)
            director = cycle.EvolutionDirector(ledger, adapter)
            director.create_run(task, run_id="run:diversity-chain")
            director.start_run("run:diversity-chain")
            batch = cycle.start_generation_batch(director, "run:diversity-chain")
            for slot in range(batch.batch_size):
                try:
                    director.request_proposal("run:diversity-chain", generation_batch=batch, slot_index=slot)
                except ValueError as exc:
                    self.fail(f"{exc}: {runtime.requests[-1]['request']['context']['evolution_reflection']['host_rejections']}")
            state = director.state("run:diversity-chain")
            self.assertEqual([p.metadata["candidate_direction"]["mutation_axis"] for p in state.proposals],
                             ["instruction_directive", "skill_program"])
            self.assertEqual([r["stage"] for r in runtime.requests].count("candidate.propose"), 2)
            self.assertEqual(state.proposals[1].metadata["diversity_schedule"]["required_family_by_slot"], ["prompt", "skill"])

    def test_partial_candidate_batch_uses_the_frozen_primary_family(self):
        targets = {axes[0]: ("registered",) for axes in FAMILY_AXES.values()}
        schedule = direction_schedule({"evolution_diversity_policy": DIVERSITY_POLICY}, targets, 1, 3)
        proposals = {family: NS(metadata={"candidate_direction": {"mutation_axis": FAMILY_AXES[family][0]},
                                         "diversity_schedule": schedule}) for family in schedule["required_family_by_slot"]}
        candidates = [NS(slot_index=i, proposal_id=family) for i, family in enumerate(proposals)]
        chosen = preregister_candidate(candidates, proposals.__getitem__, {"evolution_diversity_policy": DIVERSITY_POLICY}, 1)
        self.assertEqual(chosen.proposal_id, "skill")

    def test_synthesis_repairs_parameter_only_output_into_executable_families(self):
        fixture = research.ResearchContextTests()
        fixture.setUp()
        task = replace(fixture.task, metadata={**fixture.task.metadata, "evolution_diversity_policy": DIVERSITY_POLICY})
        parent = materialize_seed_genome(current_program_registry().seed_template("greenhouse-baseline-aligned-default@1"),
                                        _initialization(task_manifest_digest=task.digest))
        adapter = StrategyRouterDSHAdapter(gateway=object())
        calls = []
        def invoke(**request):
            calls.append(request)
            if len(calls) == 1:
                return research.minimal_aligned_synthesis()
            contract = request["context"]["synthesis_contract"]
            directions = _deterministic_fallback_directions(contract["allowed_mutation_targets"], 4,
                family_schedule=contract["diversity_schedule"], parent=parent)
            return {"schema_version": "ecologyrsi-dsh.research-synthesis/1", "summary": "Test different executable families.",
                    "evidence": [], "candidate_directions": directions}
        with patch.object(adapter, "_native_runtime", return_value=NS(run=invoke)):
            plan = adapter.research_plan("offline", run=fixture.run, task=task, parent_genome=parent.to_dict(),
                knowledge_snapshot=fixture.knowledge, generation_search_plan=fixture.search.to_dict(), candidate_count=4)
        self.assertEqual(len(calls), 2)
        self.assertIn("prompt", calls[1]["context"]["host_validation_feedback"]["validation_detail"])
        self.assertEqual(len(plan["candidate_direction_preflight"]["checks"]), 4)
        self.assertEqual(plan["diversity_schedule"]["required_family_by_slot"], list(FAMILY_AXES))

    def test_archive_uses_only_prior_complete_training_revisions_and_is_bounded(self):
        _, child = skill_child()
        revision = NS(revision_id="r1", genome=child.to_dict())
        def batch(generation=0, phase="formal_batch", coverage=True, violations=0):
            return NS(scope=NS(generation=generation, phase=NS(value=phase), batch_index=0,
                candidate_id="c1", candidate_revision_id="r1", scope_key="scope"), score=.1,
                metrics={"constraint_violations": violations, "sample_execution": {"coverage_pass": coverage,
                    "observed_behavior": {"method_counts": {"model": 9}, "signature": "b" * 64}}})
        state = NS(task_manifest=NS(metadata={"evolution_diversity_policy": DIVERSITY_POLICY}),
            candidate=lambda _: NS(proposal_id="p1"), revision=lambda _: revision,
            proposal=lambda _: NS(metadata={"candidate_direction": {"mutation_axis": "skill_program"}}))
        for invalid in (batch(generation=1), batch(phase="holdout"), batch(coverage=False), batch(violations=1)):
            state.formal_batch_evaluations = [invalid]
            self.assertEqual(exploration_archive(state, 1)["entries"], [])
        state.formal_batch_evaluations = [batch()]
        archive = exploration_archive(state, 1)
        self.assertEqual(archive["entries"][0]["components"]["skill_program"], seed_skill_program())
        self.assertIn("not_certified", archive["entries"][0]["qualification"])
        self.assertLess(len(json.dumps(archive).encode()), 6500)
        later_failed = batch(coverage=False)
        later_failed.scope.batch_index = 1
        state.formal_batch_evaluations = [batch(), later_failed]
        self.assertEqual(exploration_archive(state, 1)["entries"], archive["entries"])

    def test_every_family_gets_the_primary_slot_without_increasing_batch_size(self):
        targets = {axes[0]: ("registered",) for axes in FAMILY_AXES.values()}
        schedules = [direction_schedule({"evolution_diversity_policy": DIVERSITY_POLICY}, targets, g, 4) for g in range(4)]
        self.assertEqual([s["required_family_by_slot"][0] for s in schedules], list(FAMILY_AXES))
        for schedule in schedules:
            self.assertEqual(set(schedule["required_family_by_slot"]), set(FAMILY_AXES))
            with self.assertRaisesRegex(ValueError, "must explore"):
                validate_direction_schedule([{"mutation_axis": "scientific_parameter"}] * 4, schedule)

    def test_preregistration_rotates_by_family_and_legacy_stays_on_slot_zero(self):
        proposals = {family: NS(metadata={"candidate_direction": {"mutation_axis": axes[0]}}) for family, axes in FAMILY_AXES.items()}
        candidates = [NS(slot_index=i, proposal_id=family) for i, family in enumerate(reversed(FAMILY_AXES))]
        for generation, family in enumerate(FAMILY_AXES):
            chosen = preregister_candidate(candidates, proposals.__getitem__, {"evolution_diversity_policy": DIVERSITY_POLICY}, generation)
            self.assertEqual(chosen.proposal_id, family)
            self.assertEqual(preregister_candidate(candidates, proposals.__getitem__, {}, generation).slot_index, 0)

    def test_empty_and_unknown_policy_are_rejected(self):
        with self.assertRaises(ValueError):
            direction_schedule({"evolution_diversity_policy": "unknown"}, {}, 0, 4)
        with self.assertRaises(ValueError):
            preregister_candidate([], lambda _: None, {}, 0)

    def test_source_prose_does_not_count_as_observed_diversity(self):
        records = [{"attempt_trace": [{"model_evidence": [{"tool_id": "candidate-model"}]}]}]
        rows = [{"sample_id": "a", "predicted": 1., "agent_prediction": {"method": "model"}}]
        before = summarize_behavior(records, rows)
        after = summarize_behavior([{**records[0], "genome_digest": "new", "rationale": "different words"}], rows)
        self.assertEqual(before["signature"], after["signature"])
        self.assertFalse(before["novelty_verified"])
        different = deepcopy(records); different[0]["attempt_trace"][0]["model_evidence"].append({"tool_id": "persistence"})
        self.assertNotEqual(before["signature"], summarize_behavior(different, rows)["signature"])

    def test_plugin_schemas_admit_every_advertised_candidate_axis(self):
        root = Path(__file__).resolve().parents[1] / "integrations/dsh_ecology_plugin/schemas"
        for name in ("research-synthesis", "generation-reflection"):
            schema = json.loads((root / f"{name}.schema.json").read_text())
            self.assertEqual(set(schema["properties"]["candidate_directions"]["items"]["properties"]["mutation_axis"]["enum"]), set(CANDIDATE_MUTATION_AXES))
        for name in ("genome-mutation", "local-edit"):
            schema = json.loads((root / f"{name}.schema.json").read_text())
            operations = {item["properties"]["op"].get("const") for item in schema["properties"]["operations"]["items"]["oneOf"]}
            self.assertTrue({"author_skill_program", "author_role_directive", "author_feature_recipe"} <= operations)
