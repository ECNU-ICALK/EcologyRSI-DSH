"""Mechanism admission, real-output alignment, and old-contract compatibility."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.screening import screening_record_digest
from ecologyrsi_dsh.core.screening import screening_cohort_digest
from ecologyrsi_dsh.core.trajectory import CandidateRevision, RevisionStatus
from ecologyrsi_dsh.core.training_race import race_bindings
from ecologyrsi_dsh.evolution.branch_archive import build_branch_archive, candidate_mechanism_evidence
from ecologyrsi_dsh.evolution.genome import EcologyEvolutionPluginGenome
from ecologyrsi_dsh.core.models import canonical_json
from ecologyrsi_dsh.core.state import project_run_state
from ecologyrsi_dsh.core.immutable import thaw_json
from ecologyrsi_dsh.evolution.context import safe_aggregate_feedback
from ecologyrsi_dsh.knowledge.research_iteration import validate_research_plan
from ecologyrsi_dsh.api.projection import _public_evaluation_metrics
from ecologyrsi_dsh.evolution import exploration_policy as policy, mechanisms
from ecologyrsi_dsh.evolution.behavior_comparison import (
    build_execution_behavior, compare_generation_behaviors, unpack_behavior, pack_behavior,
)
from ecologyrsi_dsh.evolution.batches import start_generation_batch
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter, _candidate_direction_execution_view
from ecologyrsi_dsh.knowledge.autonomous_cycle import CandidateDirection
from ecologyrsi_dsh.evaluators.epoch_cohorts import plan_run_adaptation_cohort, plan_generation_selection_cohorts
from ecologyrsi_dsh.data.adapters import dataset_adapter
from tests.test_evolution_genome import _seed_genome
from tests.test_epoch_cohort_planning import dataset_fixture
from tests.test_autonomous_search_reflection_cycle import _task, _direction, _CycleRuntime


def hypothesis(mechanism, catalog):
    return {"schema_version": mechanisms.HYPOTHESIS_SCHEMA, "mechanism_id": mechanism,
            "input_refs": [catalog["input_refs"][0]], "target_cells": [catalog["target_cells"][0]],
            "expected_observation_id": "paired_error_decreases",
            "falsification_rule_id": "no_paired_error_improvement"}


class MechanismRuntime(_CycleRuntime):
    def run_stage(self, request):
        stage, context = request["stage"], request["request"]["context"]
        if stage in {"generation.research-synthesis", "generation.reflect"}:
            response = super().run_stage(request)
            portfolio = (context["synthesis_contract"]["candidate_portfolio"]
                         if stage == "generation.research-synthesis" else context["candidate_portfolio"])
            catalog = portfolio["mechanism_catalog"]
            for i, direction in enumerate(response["structured"]["candidate_directions"]):
                mechanism = portfolio["slot_requirements"][i]["allowed_mechanism_ids"][0]
                if i == 3:
                    mechanism = "parameter_adjustment"
                axis = next(axis for axis, kind in catalog["mechanism_by_axis"].items() if kind == mechanism)
                if mechanism == "agent_decision":
                    axis = "instruction_profile"
                if mechanism == "execution_control":
                    axis = "workflow_parameter"
                target = catalog["axes"][axis][0]
                if axis == "scientific_parameter":
                    target = "ridge_alpha" if i != 3 else "history_steps"
                direction.update(mutation_axis=axis, mutation_target=target,
                                 hypothesis=f"Test {mechanism} on training cells in slot {i}.",
                                 mutation_direction="select" if axis == "instruction_profile" else "decrease" if target == "history_steps" else "increase",
                                 mechanism_hypothesis=hypothesis(mechanism, catalog))
            response["result_digest"] = digest(response["structured"])
            return response
        if stage == "candidate.propose":
            self.requests.append(request)
            direction = context["assigned_candidate_direction"]
            axis, target = direction["mutation_axis"], direction["mutation_target"]
            if axis == "instruction_profile":
                op = {"op": "select_instruction_template", "role": "sample-planner", "instruction_template_id": target}
            elif axis == "workflow_parameter":
                current = context["parent_genome"]["agent_program"]["candidate_execution_program"]["workflow_overrides"][target]
                op = {"op": "set_bounded_workflow_parameter", "name": target, "value": current + 1}
            else:
                op = {"op": "set_bounded_parameter", "name": target, "value": 5 if target == "history_steps" else .2}
            structured = {"schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": [op]}
            return {"structured": structured, "result_digest": digest(structured)}
        return super().run_stage(request)


class MechanismTests(unittest.TestCase):
    def test_old_direction_bytes_and_new_hypothesis_immutability(self):
        old = CandidateDirection.from_dict(_direction(0, []))
        self.assertNotIn("mechanism_hypothesis", old.to_dict())
        task = replace(_task(), metadata={**_task().metadata, policy.POLICY_KEY: policy.POLICY})
        portfolio = mechanisms.build_candidate_portfolio(task=task, parent=_seed_genome(), generation=0, count=4,
                                                         targets={"scientific_parameter": ["ridge_alpha"]})
        self.assertEqual(portfolio["minimum_distinct_mechanisms"], 1)
        self.assertEqual(portfolio["availability_reason"], "fewer_executable_mechanisms")
        value = _direction(0, [])
        value["mechanism_hypothesis"] = hypothesis("parameter_adjustment", portfolio["mechanism_catalog"])
        new = CandidateDirection.from_dict(value)
        frozen = new.direction_digest
        value["mechanism_hypothesis"]["input_refs"].append("history:invented")
        self.assertEqual(new.direction_digest, frozen)
        self.assertIn("mechanism_hypothesis", _candidate_direction_execution_view(new))
        self.assertNotIn("hypothesis", _candidate_direction_execution_view(new))
        bad = new.to_dict()
        bad.pop("direction_digest")
        bad["mechanism_hypothesis"]["mechanism_id"] = "agent_decision"
        with self.assertRaises(ValueError):
            CandidateDirection.from_dict(bad)

    def test_research_to_proposal_and_replay_with_three_mechanisms(self):
        runtime = MechanismRuntime()
        adapter = StrategyRouterDSHAdapter(gateway=object(), native_runtime_provider=lambda: runtime)
        with EventLedger(":memory:") as ledger:
            director = EvolutionDirector(ledger, adapter)
            base = _task()
            task = replace(base, budget={"max_candidates": 4, "candidates_per_generation": 4},
                           metadata={**base.metadata, policy.POLICY_KEY: policy.POLICY,
                                     "optimization_protocol": "joint_plugin_epoch@2",
                                     "episode_id": "episode:fixture",
                                     "optimization_schedule": OptimizationSchedule.for_new_run().to_dict(),
                                     "evolution_diversity_policy": "joint_behavior_search@1"})
            director.create_run(task, run_id="run:mechanisms")
            director.start_run("run:mechanisms")
            with patch("ecologyrsi_dsh.knowledge.deepseek_search.is_deepseek_search_configured", return_value=False):
                batch = start_generation_batch(director, "run:mechanisms")
                proposals = [director.request_proposal("run:mechanisms", generation_batch=batch, slot_index=i) for i in range(4)]
            tags = {p.metadata["mechanism_realization"]["mechanism_ids"][0] for p in proposals}
            self.assertEqual(len(tags), 3)
            self.assertEqual(len(director.replay("run:mechanisms").proposals), 4)
            bad = replace(proposals[0], proposal_id="proposal:tampered",
                          metadata={**proposals[0].metadata, "mechanism_realization": {}})
            with self.assertRaisesRegex(ValueError, "realization receipt"):
                director.submit_proposal(bad)
            self.assertEqual(len(director.state("run:mechanisms").proposals), 4)
            # Continue through all five screening arms and durable comparison,
            # using deterministic fixture predictions without any remote calls.
            candidates = []
            for i, proposal in enumerate(proposals):
                candidate = director.spawn_candidate("run:mechanisms", proposal, slot_index=i)
                candidates.append(candidate)
                genome = EcologyEvolutionPluginGenome.from_dict(
                    json.loads(proposal.metadata["evolution_genome_canonical_json"]))
                director.create_candidate_revision("run:mechanisms", CandidateRevision(
                    revision_id=f"revision:{candidate.candidate_id}", run_id="run:mechanisms", generation=0,
                    candidate_id=candidate.candidate_id, genome=genome.to_dict(), genome_digest=genome.genome_digest,
                    behavior_digest=genome.behavior_digest, mutation_digest=genome.lineage["mutation_digest"],
                    status=RevisionStatus.ACTIVE))
            control = director.ensure_seed_incumbent_control("run:mechanisms")
            fixture = BehaviorTests()
            fixture.setUp()
            fixture.base = task
            fixture.metadata = dict(task.metadata)
            dataset = replace(dataset_fixture(3000), dataset_id=task.dataset)
            schedule = OptimizationSchedule.for_new_run()
            adaptation = plan_run_adaptation_cohort(dataset, schedule=schedule, seed=task.seed)
            planned = replace(plan_generation_selection_cohorts(dataset, schedule=schedule, generation=0,
                adaptation=adaptation, seed=task.seed), revision_bindings=race_bindings(director.state("run:mechanisms"), 0))
            fixture.planned = planned
            director.freeze_run_adaptation_cohort("run:mechanisms", adaptation)
            director.freeze_generation_selection_cohorts("run:mechanisms", planned)
            for i, candidate in enumerate([*candidates, control]):
                director.record_candidate_screening("run:mechanisms", candidate_id=candidate.candidate_id,
                    candidate_revision_id=planned.revision_bindings[candidate.candidate_id], generation=0,
                    score=.2, passed=False, constraint_violations=0, origin_count=10, prediction_cell_count=90,
                    cohort_digest=planned.screening.cohort_digest,
                    metrics={"constraint_violations": 0, "sample_execution": {
                        "coverage_pass": True,
                        "execution_behavior": fixture.profile(candidate.candidate_id, delta=i)}})
            event = director.record_generation_behavior_comparison("run:mechanisms", generation=0)
            self.assertEqual(len(event.payload["pairs"]), 10)
            self.assertEqual(director.record_generation_behavior_comparison("run:mechanisms", generation=0).event_id, event.event_id)
            replayed = director.replay("run:mechanisms")
            self.assertEqual(replayed.generation_behavior_events[0].payload, event.payload)
            corrupted = thaw_json(event.payload)
            corrupted["pairs"][0]["near_outputs"] = not corrupted["pairs"][0]["near_outputs"]
            corrupted["comparison_digest"] = digest({k: v for k, v in corrupted.items() if k != "comparison_digest"})
            with self.assertRaisesRegex(ValueError, "does not match executed evidence"):
                project_run_state(tuple(replace(e, payload=corrupted) if e.event_id == event.event_id else e
                                        for e in ledger.events("run:mechanisms")))
            records = [e.payload for e in replayed.candidate_screening_events]
            director.freeze_formal_selection_cohort("run:mechanisms", generation=0,
                selected_candidate_ids=[candidates[0].candidate_id], screening_digest=screening_cohort_digest(records))
            archive = build_branch_archive(director.replay("run:mechanisms"), 1)
            self.assertGreaterEqual(len(archive["mechanism_counts"]), 3)
            self.assertLessEqual(len(canonical_json(archive["entries"]).encode()), 6000)
            self.assertTrue(all(e["qualification"] == "screening_only_not_independent_validation" for e in archive["entries"]))
            self.assertEqual(validate_research_plan({"exploration_archive": archive})["exploration_archive"], archive)
            self.assertEqual(safe_aggregate_feedback({"exploration_archive": archive}, name="next_generation")["exploration_archive"], archive)
            self.assertTrue(any("workflow_overrides" in e["components"] for e in archive["entries"]))
            evidence = candidate_mechanism_evidence(replayed, candidates[0], proposals[0])
            self.assertEqual(evidence["comparison_status"], "measured")
            self.assertNotIn("payload", canonical_json(evidence))
            reflection = adapter.reflect_generation(
                run=replayed.run, task=task, parent_genome=json.loads(batch.parent_genome_canonical_json),
                generation_analysis={"analysis_digest": digest("fixture-analysis"), "outcome": "no_eligible_candidate"},
                knowledge_snapshot=None, research_iteration=None, candidate_outcomes=[], direction_count=4,
                run_state_revision=0, stage_attempt=1, ledger_expected_revision=ledger.latest_seq())
            self.assertEqual(len({d.mechanism_hypothesis["mechanism_id"] for d in reflection.candidate_directions}), 3)


class BehaviorTests(unittest.TestCase):
    def setUp(self):
        base = _task()
        self.metadata = {**base.metadata, policy.POLICY_KEY: policy.POLICY}
        schedule = OptimizationSchedule.for_new_run()
        dataset = dataset_fixture(3000)
        adaptation = plan_run_adaptation_cohort(dataset, schedule=schedule, seed=0)
        self.planned = plan_generation_selection_cohorts(dataset, schedule=schedule, generation=0, adaptation=adaptation, seed=0)
        self.base = base

    def profile(self, cid, *, delta=0, failed=False, changed_labels=False, different_actions=False):
        cohort = self.planned.screening
        task = replace(self.base, metadata={**self.metadata,
            "_planned_evaluation_cohort": cohort.to_dict(),
            "_evaluation_scope": {"phase": "screening", "candidate_id": cid,
                "candidate_revision_id": (self.planned.revision_bindings or {}).get(cid, f"revision:{cid}"),
                "cohort_digest": cohort.cohort_digest}})
        adapter = dataset_adapter(task.dataset)
        rows, records = [], []
        for i, origin in enumerate(cohort.origins):
            for target in adapter.target_names:
                for horizon in adapter.horizons_hours:
                    sid = f"{cid}:{i}:{target}:{horizon}"
                    rows.append({"sample_id": sid, "target": target, "horizon_hours": horizon,
                        "origin_timestamp": origin.origin_timestamp, "cohort_origin_occurrence": origin.reuse_index,
                        "predicted": 10 + delta, "observed": 999 if changed_labels else 1,
                        "sample_execution_status": "failed" if failed else "succeeded",
                        "agent_prediction": {"method": "fixture", "tools": [
                            {"tool_id": "other" if different_actions else "candidate-model",
                             "status": "completed", "used_as_evidence": True}]}})
                    records.append({"sample_id": sid, "attempt_trace": [{
                        "outcome": "failed" if failed else "accepted",
                        "model_evidence": [{"tool_id": "other" if different_actions else "candidate-model",
                                            "status": "completed", "used_as_evidence": True}]}]})
        scales = {t: {"scale": 10, "source_partition": "training_fit"} for t in adapter.target_names}
        return build_execution_behavior(task, records, rows, scales)

    def compare(self, **changes):
        ids = ["a", "b", "c", "d", "incumbent"]
        planned = replace(self.planned, revision_bindings={cid: f"revision:{cid}" for cid in ids})
        records = {}
        for cid in ids:
            profile = self.profile(cid, **changes.get(cid, {}))
            record = {"schema_version": "ecologyrsi-dsh.candidate-screening/3", "generation": 0,
                      "candidate_id": cid, "candidate_revision_id": f"revision:{cid}",
                      "score": .1, "passed": True, "constraint_violations": 0,
                      "origin_count": 10, "prediction_cell_count": 90,
                      "cohort_digest": planned.screening.cohort_digest,
                      "metrics": {"sample_execution": {"execution_behavior": profile}}}
            record["record_digest"] = screening_record_digest(record)
            records[cid] = record
        return compare_generation_behaviors(planned, records, metadata=self.metadata), planned, records

    def test_candidate_ids_and_labels_do_not_create_output_diversity(self):
        self.assertEqual(self.profile("a"), self.profile("a", changed_labels=True))
        result, _, _ = self.compare()
        self.assertEqual(len(result["pairs"]), 10)
        self.assertTrue(all(p["status"] == "measured" and p["near_outputs"] for p in result["pairs"]))
        self.assertEqual(len(result["groups"]), 1)
        self.assertFalse(result["scientific_gain_established"])

    def test_paths_outputs_and_failures_remain_separate(self):
        result, _, _ = self.compare(b={"different_actions": True}, c={"delta": 2}, d={"failed": True})
        pairs = {(p["left"], p["right"]): p for p in result["pairs"]}
        self.assertTrue(pairs["a", "b"]["near_outputs"])
        self.assertGreater(pairs["a", "b"]["changed_action_cells"], 0)
        self.assertFalse(pairs["a", "c"]["near_outputs"])
        self.assertEqual(pairs["a", "d"]["status"], "unverified")
        self.assertAlmostEqual(next(iter(pairs["a", "c"]["distances"].values()))["prediction_distance"], .2)

    def test_scope_tampering_and_missing_arm_fail_closed(self):
        _, planned, records = self.compare()
        with self.assertRaisesRegex(ValueError, "every frozen"):
            compare_generation_behaviors(planned, list(records.values())[:-1], metadata=self.metadata)
        body = unpack_behavior(records["a"]["metrics"]["sample_execution"]["execution_behavior"])
        body["revision_id"] = "another"
        records["a"]["metrics"]["sample_execution"]["execution_behavior"] = pack_behavior(body)
        with self.assertRaisesRegex(ValueError, "another screening"):
            compare_generation_behaviors(planned, records, metadata=self.metadata)
        archive = self.profile("a")
        archive["uncompressed_bytes"] = 2_000_000
        with self.assertRaises(ValueError):
            unpack_behavior(archive)

    def test_formal_partition_cannot_produce_exploration_evidence(self):
        task = replace(self.base, metadata={**self.metadata, "_evaluation_scope": {"phase": "validation"}})
        self.assertIsNone(build_execution_behavior(task, [], [], {}))

    def test_raw_vectors_stay_out_of_public_and_model_feedback(self):
        from ecologyrsi_dsh.application.formal_trajectory import _durable_batch_metrics
        metrics = {"objective_score": .1, "sample_execution": {"coverage": 1., "execution_behavior": self.profile("a")}}
        self.assertIn("execution_behavior", _durable_batch_metrics(metrics)["sample_execution"])
        self.assertNotIn("execution_behavior", _public_evaluation_metrics(metrics)["sample_execution"])
        self.assertNotIn("execution_behavior", canonical_json(safe_aggregate_feedback(metrics, name="feedback")))

    def test_native_evaluator_produces_replayable_actual_output_vectors(self):
        from ecologyrsi_dsh.core.models import Candidate, Proposal, TaskManifest
        from ecologyrsi_dsh.core.trajectory import EvaluationScope, EvaluationPhase
        from ecologyrsi_dsh.core.sample_results import build_sample_results
        from ecologyrsi_dsh.evaluators.epoch_cohorts import PlannedCohort, PlannedOrigin
        from ecologyrsi_dsh.evaluators.registry import EvaluatorRegistry, GREENHOUSE_MULTIHORIZON_EVALUATOR_V3_ID
        from ecologyrsi_dsh.evaluators.greenhouse_prediction import BASELINE_ALIGNED_RIDGE_MODEL_ID
        from tests.test_baseline_aligned_ridge import periodic_series
        from tests.test_evaluation import _DatasetStub, _DshOriginRuntimeStub, _dsh_prediction_tool_binder
        from tests.test_effect_contracts import mutate
        from ecologyrsi_dsh.evolution.effect_contracts import resolve_mutation_effects
        from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
        series, runtime = periodic_series(), _DshOriginRuntimeStub()
        parent = mutate(_seed_genome(), {"op": "select_registered_pipeline",
                                        "predictor_id": BASELINE_ALIGNED_RIDGE_MODEL_ID})
        operation = {"op": "set_bounded_parameter", "name": "residual_scale_24h", "value": .1}
        child = mutate(parent, operation)
        resolution = resolve_mutation_effects(parent, child, [operation], current_program_registry())
        task = TaskManifest(task_id="measured-native", objective="test actual native execution evidence",
            domain_pack="greenhouse_environment@1", visible_datasets=(series.dataset_id,),
            budget={"max_candidates": 1}, metadata={
                policy.POLICY_KEY: policy.POLICY, "episode_id": series.episode_id,
                "mutation_effect_scope_policy": "per_predicate_impact@1",
                "prediction_model_id": BASELINE_ALIGNED_RIDGE_MODEL_ID,
                "evaluator_id": GREENHOUSE_MULTIHORIZON_EVALUATOR_V3_ID,
                "dataset_digest": series.digest, "split_manifest_digest": series.split_manifest_digest_sha256,
                "sample_agent_mode": "dsh_native_agent", "sample_agent_protocol": "dsh-strict-origin-bundle@3",
                "sample_agent_batch_size": 9, "sample_concurrency": 1, "prediction_cells_per_origin": 9,
                "strategy_model_id": "dsh/strategy", "review_model_id": "dsh/review",
            })
        proposal = Proposal(proposal_id="proposal:measured", run_id="run:measured", generation=0,
                            title="measured", changes=thaw_json(child.scientific_program["parameter_overrides"]),
                            metadata={"genome_digest": child.genome_digest,
                                      "mutation_operations": [operation],
                                      "mutation_effect_resolution": resolution,
                                      "evolution_genome_canonical_json": canonical_json(child.to_dict())})
        candidate = Candidate(candidate_id="candidate:measured", run_id=proposal.run_id,
                              proposal_id=proposal.proposal_id, generation=0)
        cohort = PlannedCohort(role="screening", maximum_horizon=24, origins=(
            PlannedOrigin(digest("origin"), series.dataset_id, series.episode_id, 200, 200, 224, digest("maturity")),))
        scope = EvaluationScope(run_id=candidate.run_id, generation=0, candidate_id=candidate.candidate_id,
            candidate_revision_id="revision:measured", phase=EvaluationPhase.SCREENING,
            cohort_digest=cohort.cohort_digest, origin_count=1)
        registry = EvaluatorRegistry(_DatasetStub(series), object(), dsh_runtime_provider=lambda: runtime,
            dsh_revision_provider=lambda _: {"run_state_revision": 1, "ledger_expected_revision": 1},
            dsh_identity_provider=lambda *_: {"genome_digest": child.genome_digest, "compiled_behavior_digest": "b"*64,
                                             "phenotype_instance_digest": "c"*64},
            dsh_prediction_tool_binder=_dsh_prediction_tool_binder)
        saved = {}
        def capture(rows):
            saved.update({r["sample_id"]: r for r in build_sample_results(candidate.candidate_id, rows)})
        first = registry.evaluate_scientific(task, candidate, proposal, scope=scope, cohort=cohort, on_sample_results=capture)
        archive = first.evaluation.metrics["sample_execution"]["execution_behavior"]
        body = unpack_behavior(archive)
        self.assertEqual((archive["status"], archive["successful_cell_count"]), ("measured", 9))
        self.assertTrue(all(v["tool_predicted"] is not None for v in body["vectors"]))
        effect = first.evaluation.metrics["sample_execution"]["mutation_effect_receipt"]
        check = effect["checks"][0]
        self.assertEqual((check["eligible"], check["matched"], check["status"]), (3, 3, "passed"))
        self.assertEqual(set(check["affected_cells"]), {
            "air_temperature@24h", "relative_humidity@24h", "co2_concentration@24h"})
        runtime.requests.clear()
        replay = registry.evaluate_scientific(task, candidate, proposal, scope=scope, cohort=cohort,
                                             on_sample_checkpoint=lambda _: {"rows": list(saved.values())})
        self.assertEqual(runtime.requests, [])
        self.assertEqual(archive, replay.evaluation.metrics["sample_execution"]["execution_behavior"])
        self.assertEqual(first.evaluation.score, replay.evaluation.score)
        # Historical checkpoints retain predictions and chain attestations,
        # not the complete attempt trace. Never invent successful effect
        # observations from those missing traces on recovery.
        recovered = replay.evaluation.metrics["sample_execution"]["mutation_effect_receipt"]
        self.assertEqual(recovered["status"], "inconclusive")
        self.assertEqual(recovered["reason"], "incomplete_effect_trace")
        self.assertEqual(recovered["checks"][0]["eligible"], 0)
