from types import SimpleNamespace
import unittest

from ecologyrsi_dsh.core.local_edit_events import validate_local_effect_event
from ecologyrsi_dsh.core.models import canonical_json, digest
from ecologyrsi_dsh.evaluators.skill_program import skill_preflight_programs
from ecologyrsi_dsh.evolution.effect_contracts import (
    build_edit_effect_contract, check_edit_effect_contract, observe_edit_effects,
    resolve_mutation_effects, verify_runtime_effects, hard_effect_failure, mutation_effect_cells,
)
from ecologyrsi_dsh.evolution.genome import apply_genome_mutation
from ecologyrsi_dsh.evolution.local_edits import LocalEditProposal, apply_local_edit_bundle
from ecologyrsi_dsh.evolution.workflow_ir import resolve_candidate_agent_profile
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from ecologyrsi_dsh.integrations.prediction_binding import DshPredictionToolBinding
from tests.test_evolution_genome import _mutation_context, _seed_genome
from tests.test_local_edits import _context, _parent


def mutate(parent, operation, **kwargs):
    return apply_genome_mutation(parent, {
        "schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": [operation],
    }, _mutation_context(parent), current_program_registry(), **kwargs)


def local_result():
    parent = _parent()
    proposal = LocalEditProposal(
        decision="mutate", operations=({"op": "set_bounded_parameter", "name": "ridge_alpha", "value": 0.2},),
        evidence_refs=("metric:overall",), expected_effect_cells=("air_temperature@1h",), risk_cells=(),
    )
    context = _context(parent)
    return parent, proposal, context, apply_local_edit_bundle(parent, proposal, context, current_program_registry())


def accepted_trace(*, tools=(), skills=(), attempt=1):
    return {"attempt_trace": [{"attempt": attempt, "outcome": "accepted", "model_evidence": list(tools),
                              "skill_evidence": list(skills), "selected_tool": {"output_digest": "a" * 64}}]}


class EffectContractTests(unittest.TestCase):
    def test_outer_mutation_rechecks_actual_host_floor_before_inference(self):
        parent = _seed_genome()
        operation = {"op": "set_bounded_workflow_parameter", "name": "max_attempts", "value": 4}
        child = mutate(parent, operation)
        resolution = resolve_mutation_effects(parent, child, [operation], current_program_registry())
        with self.assertRaisesRegex(ValueError, "effective_noop"):
            verify_runtime_effects(resolution, {"sample_max_attempts_floor": 5})
        with self.assertRaisesRegex(ValueError, "effective_noop"):
            mutate(parent, operation, runtime_constraints={"sample_max_attempts_floor": 5})
        operation["value"] = 6
        child = mutate(parent, operation, runtime_constraints={"sample_max_attempts_floor": 5})
        resolution = resolve_mutation_effects(parent, child, [operation], current_program_registry(),
                                             runtime_constraints={"sample_max_attempts_floor": 5})
        self.assertEqual(resolution["operations"][0]["before_effective_value"], 5)
        self.assertEqual(resolution["operations"][0]["after_effective_value"], 6)

    def test_unconsumed_workflow_parameter_and_always_critic_threshold_are_noops(self):
        parent = _seed_genome()
        with self.assertRaisesRegex(ValueError, "no execution consumer"):
            mutate(parent, {"op": "set_bounded_workflow_parameter", "name": "max_concurrent", "value": 5})
        threshold = {"op": "set_instruction_parameter", "role": "sample-planner",
                     "name": "confidence_threshold", "value": 0.6}
        with self.assertRaisesRegex(ValueError, "effective_noop"):
            mutate(parent, threshold, runtime_constraints={"remote_critic_policy": None})
        child = mutate(parent, threshold, runtime_constraints={"remote_critic_policy": {
            "version": "uncertain_or_failure@1", "min_planner_confidence": 0.5,
        }})
        self.assertNotEqual(child.behavior_digest, parent.behavior_digest)

    def test_reset_restores_baseline_without_changing_tools_or_scientific_parameters(self):
        parent = _seed_genome()
        directives = {"anchor": "candidate_model", "blend_rule": "mean", "tool_plan": [{"tool_id": "ecology_execute_prediction_tool", "purpose": "discrepancy_check"}],
                      "rationale": "Use the current model evidence."}
        for component, operation in (
            ("authored_directive", {"op": "author_role_directive", "role": "sample-planner", "authored_directive": directives}),
            ("skill_program", {"op": "author_skill_program", "role": "sample-planner", "skill_program": skill_preflight_programs()[0]}),
        ):
            with self.subTest(component=component):
                authored = mutate(parent, operation)
                reset = mutate(authored, {"op": "reset_role_component", "role": "sample-planner", "component": component})
                self.assertEqual(reset.scientific_program, parent.scientific_program)
                self.assertEqual(resolve_candidate_agent_profile(reset, current_program_registry()),
                                 resolve_candidate_agent_profile(parent, current_program_registry()))
                self.assertIn(component, resolve_candidate_agent_profile(authored, current_program_registry()))
                with self.assertRaisesRegex(ValueError, "does not change"):
                    mutate(reset, {"op": "reset_role_component", "role": "sample-planner", "component": component})
        with self.assertRaisesRegex(ValueError, "resettable"):
            mutate(parent, {"op": "reset_role_component", "role": "sample-planner", "component": "enabled_tool_ids"})

    def test_local_contract_is_bound_to_training_evidence_and_immutable_genomes(self):
        parent, proposal, context, result = local_result()
        self.assertEqual(result.effect_contract["parent_revision_id"], context.candidate_revision_id)
        self.assertEqual(result.effect_contract["affected_cells"], sorted(context.allowed_effect_cells))
        self.assertNotEqual(result.effect_contract["affected_cells"], list(proposal.expected_effect_cells))
        parent_revision = SimpleNamespace(revision_id=context.candidate_revision_id, genome=parent.to_dict())
        revision = SimpleNamespace(parent_revision_id=context.candidate_revision_id, genome=result.child.to_dict())
        event_proposal = {"operations": list(proposal.operations), "proposal": proposal.to_dict(),
                          "evidence_scope_digest": context.evidence_scope_digest}
        payload = {"outcome": "applied", "effect_contract": result.effect_contract, "effect_resolution": result.effect_resolution}
        validate_local_effect_event(payload, revision=revision, parent_revision=parent_revision, proposal=event_proposal, allowed_effect_cells=context.allowed_effect_cells)
        forged = {**event_proposal, "evidence_scope_digest": "f" * 64}
        with self.assertRaisesRegex(ValueError, "evidence"):
            validate_local_effect_event(payload, revision=revision, parent_revision=parent_revision, proposal=forged, allowed_effect_cells=context.allowed_effect_cells)
        with self.assertRaisesRegex(ValueError, "training evidence"):
            build_edit_effect_contract(parent_revision_id="r", evidence_scope_digest="e" * 64,
                                       evidence_refs=["holdout:score"], affected_cells=["air_temperature@1h"],
                                       resolution=result.effect_resolution, source_phase="holdout")

    def test_direct_predictions_do_not_prove_scientific_parameter_consumption(self):
        _, _, _, result = local_result()
        metadata = {"genome_digest": result.child.genome_digest, "effect_contract": result.effect_contract,
                    "effect_resolution": result.effect_resolution, "mutation_operations": list(result.operations)}
        common = dict(scope_digest="b" * 64, phase="formal_batch", parameters=result.child.scientific_program["parameter_overrides"])
        receipt = observe_edit_effects(metadata, [accepted_trace()], **common)
        self.assertEqual(receipt["status"], "failed")
        tool = {"tool_id": "candidate-model", "status": "completed", "used_as_evidence": True, "dsh_tool_event_id": "event:1"}
        self.assertEqual(observe_edit_effects(metadata, [accepted_trace(tools=[tool])], **common)["status"], "passed")
        for change in ({"status": "failed"}, {"used_as_evidence": False}):
            self.assertEqual(observe_edit_effects(metadata, [accepted_trace(tools=[{**tool, **change}])], **common)["status"], "failed")
        self.assertIsNone(observe_edit_effects(metadata, [accepted_trace(tools=[tool])], **{**common, "phase": "holdout"}))
        metadata["evolution_genome_canonical_json"] = canonical_json(result.child.to_dict())
        tool = {**tool, "tool_id": result.child.scientific_program["predictor_ref"]["id"],
                "effective_parameters_digest": digest(dict(common["parameters"]))}
        self.assertEqual(observe_edit_effects(metadata, [accepted_trace(tools=[tool])], **common)["status"], "passed")
        self.assertEqual(observe_edit_effects(metadata, [accepted_trace(tools=[{**tool, "tool_id": "other-model"}])], **common)["status"], "failed")

    def test_unused_retry_capacity_is_not_an_exercised_effect(self):
        parent = _seed_genome()
        operation = {"op": "set_bounded_workflow_parameter", "name": "max_attempts", "value": 4}
        child = mutate(parent, operation)
        resolution = resolve_mutation_effects(parent, child, [operation], current_program_registry())
        contract = build_edit_effect_contract(parent_revision_id="r", evidence_scope_digest="a" * 64,
            evidence_refs=["batch:score"], affected_cells=["air_temperature@1h"], resolution=resolution)
        metadata = {"genome_digest": child.genome_digest, "effect_contract": contract, "effect_resolution": resolution,
                    "mutation_operations": [operation]}
        common = dict(scope_digest="b" * 64, phase="formal_batch", parameters={})
        for depth in (1, 3):
            self.assertEqual(observe_edit_effects(metadata, [accepted_trace(attempt=depth)], **common)["status"], "not_exercised")
        self.assertEqual(observe_edit_effects(metadata, [accepted_trace(attempt=4)], **common)["status"], "passed")

    def test_skill_trigger_receipt_is_not_prompt_compliance(self):
        parent = _seed_genome()
        operation = {"op": "author_skill_program", "role": "sample-planner", "skill_program": skill_preflight_programs()[0]}
        child = mutate(parent, operation)
        resolution = resolve_mutation_effects(parent, child, [operation], current_program_registry())
        contract = build_edit_effect_contract(parent_revision_id="r", evidence_scope_digest="a" * 64,
            evidence_refs=["batch:score"], affected_cells=["air_temperature@1h"], resolution=resolution)
        metadata = {"genome_digest": child.genome_digest, "effect_contract": contract, "effect_resolution": resolution,
                    "mutation_operations": [operation]}
        for triggered in ("not_triggered", "triggered"):
            receipt = observe_edit_effects(metadata, [accepted_trace(skills=[{"trigger_status": triggered, "output_digest": "c" * 64}])],
                                           scope_digest="b" * 64, phase="formal_batch", parameters={})
            self.assertEqual(receipt["status"], "inconclusive")
            checks = {row["predicate_id"]: row for row in receipt["checks"]}
            self.assertEqual(checks["skill_triggered"]["status"], "passed" if triggered == "triggered" else "not_exercised")
            self.assertEqual(checks["guidance_behavior_changed"]["status"], "inconclusive")

    def test_unknown_unreferenced_or_cross_child_observations_cannot_pass(self):
        _, _, _, result = local_result()
        common = dict(scope_digest="a" * 64, child_genome_digest=result.child.genome_digest)
        with self.assertRaisesRegex(ValueError, "receipt references"):
            check_edit_effect_contract(result.effect_contract, **common,
                observations={"candidate_model_evidence": {"eligible": 1, "matched": 1, "evidence_refs": []}})
        with self.assertRaisesRegex(ValueError, "another child"):
            check_edit_effect_contract(result.effect_contract, **{**common, "child_genome_digest": "f" * 64}, observations={})
        with self.assertRaisesRegex(ValueError, "unknown"):
            check_edit_effect_contract(result.effect_contract, **common, observations={"configuration_changed": True})

    def test_hard_failure_blocks_using_frozen_receipts_but_not_soft_unknown_effects(self):
        from ecologyrsi_dsh.core.immutable import freeze_json
        _, _, _, result = local_result()
        metadata = {"genome_digest": result.child.genome_digest, "effect_resolution": result.effect_resolution,
                    "mutation_operations": list(result.operations)}
        receipt = observe_edit_effects(metadata, [accepted_trace()], scope_digest="b" * 64,
                                       phase="screening", parameters=result.child.scientific_program["parameter_overrides"])
        metrics = freeze_json({"sample_execution": {"mutation_effect_receipt": receipt}})
        self.assertEqual(hard_effect_failure(metrics), "hard_mutation_effect_failed")
        for semantics, status in (("soft", "inconclusive"), ("hard", "not_exercised"), ("soft", "failed")):
            body = {**receipt, "checks": [{"predicate_id": "guidance_behavior_changed", "semantics": semantics, "status": status}]}
            body.pop("receipt_digest")
            body["receipt_digest"] = digest(body)
            self.assertIsNone(hard_effect_failure(freeze_json({"sample_execution": {"mutation_effect_receipt": body}})))
        bad = {**receipt, "receipt_digest": "0" * 64}
        with self.assertRaisesRegex(ValueError, "identity"):
            hard_effect_failure({"sample_execution": {"mutation_effect_receipt": bad}})

    def test_specific_parameter_effect_domain_is_host_computed(self):
        cells = ("air_temperature@1h", "air_temperature@24h", "co2_concentration@24h")
        self.assertEqual(mutation_effect_cells([{"op": "set_bounded_parameter", "name": "air_temperature_24h_residual_scale"}], cells),
                         ("air_temperature@24h",))
        self.assertEqual(mutation_effect_cells([{"op": "set_bounded_parameter", "name": "residual_scale_24h"}], cells),
                         ("air_temperature@24h", "co2_concentration@24h"))
        self.assertEqual(mutation_effect_cells([{"op": "author_role_directive"}], cells), tuple(sorted(cells)))

    def test_native_proposer_retries_host_masked_edit_before_freezing_candidate(self):
        from ecologyrsi_dsh import EventLedger, EvolutionDirector
        from ecologyrsi_dsh.evolution.execution_plan import DerivedExecutionPlan
        from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter
        from tests.test_dsh_structured_roles import _native_task, _SequencedNativeRuntime
        runtime = _SequencedNativeRuntime([
            {"schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": [
                {"op": "set_bounded_workflow_parameter", "name": "max_attempts", "value": 4}]},
            {"schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": [
                {"op": "set_bounded_workflow_parameter", "name": "max_attempts", "value": 6}]},
        ])
        adapter = StrategyRouterDSHAdapter(gateway=object(), native_runtime_provider=lambda: runtime)
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, adapter)
            task = _native_task()
            director.create_run(task, run_id="effect-floor")
            director.start_run("effect-floor")
            state = director.state("effect-floor")
            parent = state.materialized_seed_genome()
            proposal = adapter.propose(state.run, task, state.run.session_id, batch_context={
                "generation": 0, "slot_index": 0, "batch_size": 1, "context_digest": "a" * 64,
                "parent_genome_digest": parent.genome_digest,
                "parent_genome_canonical_json": canonical_json(parent.to_dict()),
                "stage_context_digests": {"research_iteration_digest": "b" * 64, "knowledge_snapshot_digest": "c" * 64},
                "derived_execution_plan": DerivedExecutionPlan(None, None, sample_max_attempts=5).to_dict(),
            })
        self.assertEqual(len(runtime.requests), 2)
        rejection = runtime.requests[1]["request"]["context"]["evolution_reflection"]["host_rejections"][0]
        self.assertEqual(rejection["rejection_code"], "mutation_has_no_effect")
        self.assertEqual(proposal.metadata["effect_resolution"]["operations"][0]["before_effective_value"], 5)

    def test_mean_formula_is_bound_to_actual_cited_sample_outputs_and_replay(self):
        events = []

        def persist(payload):
            event = SimpleNamespace(event_id=f"tool:{len(events)}", payload=payload)
            events.append(event)
            return event

        args = dict(run_id="r", stage_attempt=1, idempotency_key="wave:1", wave_digest="a" * 64,
                    sample_ids=["a", "b"], catalog=[{"tool_id": "one", "version": "1"}, {"tool_id": "two", "version": "1"}],
                    executor=lambda tool_id, _: {sample: {"predicted": value} for sample, value in
                        zip(("a", "b"), ((10, 20) if tool_id == "one" else (30, 40)))})
        binding = DshPredictionToolBinding(**args, mean_blend_required=True)
        for tool_id in ("one", "two"):
            binding.execute({"tool_id": tool_id, "call_id": tool_id, "parameters": {}, "wave_digest": "a" * 64},
                            session_id="s", persist=persist)
        rows = [{"sample_id": sample, "method": "blend", "predicted": value,
                 "evidence_call_ids": ["one", "two"]} for sample, value in (("a", 20), ("b", 30))]
        binding.validate_formulas(rows)
        replay = DshPredictionToolBinding(**args, mean_blend_required=True)
        for event in events:
            replay.restore(event)
        replay.validate_formulas(rows)
        wrong = [{**rows[0], "predicted": 30}, rows[1]]
        with self.assertRaisesRegex(ValueError, "does not equal"):
            replay.validate_formulas(wrong)
        with self.assertRaisesRegex(ValueError, "another sample"):
            replay.validate_formulas([{**rows[0], "sample_id": "foreign"}])
        legacy = DshPredictionToolBinding(**args)
        legacy.validate_formulas(wrong)

    def test_frozen_formula_policy_survives_local_rebinding_without_upgrading_legacy(self):
        from ecologyrsi_dsh.evolution.agent_policy import build_agent_policy, rebind_agent_policy
        kwargs = dict(genome_digest="a" * 64, profile={}, parameters={})
        legacy = build_agent_policy(**kwargs, generation=0, previous_analysis=None)
        new = build_agent_policy(**kwargs, generation=0, previous_analysis=None,
                                 prediction_formula_policy="mean-referenced-tools@1")
        self.assertNotIn("prediction_formula_policy", rebind_agent_policy(legacy, **kwargs)["inference"])
        self.assertEqual(rebind_agent_policy(new, **kwargs)["inference"]["prediction_formula_policy"], "mean-referenced-tools@1")


if __name__ == "__main__":
    unittest.main()
