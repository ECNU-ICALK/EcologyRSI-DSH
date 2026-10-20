from tests.native_fixtures import start_native_run
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import canonical_json, digest
from ecologyrsi_dsh.evolution import novelty
from ecologyrsi_dsh.evolution.batches import start_generation_batch
from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter, _validate_candidate_direction_realizability
from ecologyrsi_dsh.evaluators.authored_directive import AUTHORED_DIRECTIVE_POLICY_ID
from ecologyrsi_dsh.knowledge.autonomous_cycle import GenerationSearchPlan
from ecologyrsi_dsh.knowledge.models import KnowledgeSnapshot
from ecologyrsi_dsh.knowledge.research_iteration import ResearchIteration, validate_research_plan
from tests.test_dsh_structured_roles import _native_task, _SequencedNativeRuntime
from tests.test_autonomous_search_reflection_cycle import _task, _CycleRuntime, _direction


def parameter(value):
    return {"schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": [
        {"op": "set_bounded_parameter", "name": "ridge_alpha", "value": value}]}


class CandidateNoveltyTests(unittest.TestCase):
    def setup_run(self, outputs, *, enabled=True, task=None):
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        runtime = _SequencedNativeRuntime(outputs)
        base = task or _native_task()
        task = replace(base, budget={"max_candidates": 12, "max_generations": 3, "candidates_per_generation": 4},
                       metadata={**base.metadata, **({novelty.CANDIDATE_NOVELTY_POLICY_KEY:
                           novelty.CANDIDATE_NOVELTY_POLICY} if enabled else {})})
        adapter = StrategyRouterDSHAdapter(gateway=object(), native_runtime_provider=lambda: runtime,
            candidate_novelty_provider=lambda run_id, generation: novelty.build_archive(self.novelty_state(director), generation))
        director = EvolutionDirector(ledger, adapter)
        self.fixture_generation = 0
        start_native_run(director, task=task, run_id="run:novelty")
        return ledger, director, adapter, runtime

    def novelty_state(self, director):
        # This unit seam supplies a later immutable state to proposal novelty;
        # it does not bypass the production scientific advance-generation gate.
        state = director.state("run:novelty")
        return replace(state, run=replace(state.run, generation=self.fixture_generation))

    def proposal_context(self, director, direction=None):
        state = self.novelty_state(director)
        parent = state.materialized_seed_genome()
        plan = {}
        if state.task_manifest.metadata.get("autonomous_research_protocol"):
            plan["candidate_directions"] = [direction or _direction(0, [])]
        if novelty.enabled(state.task_manifest.metadata):
            plan[novelty.ARCHIVE_KEY] = novelty.archive_reference(novelty.build_archive(state, state.run.generation))
        iteration = ResearchIteration(run_id=state.run.run_id, generation=state.run.generation,
            status="model_generated", plan=plan,

            knowledge_snapshot_digest="2" * 64)
        return {"generation": state.run.generation, "slot_index": 0, "batch_size": 1,
                "context_digest": "3" * 64, "parent_genome_digest": parent.genome_digest,
                "parent_genome_canonical_json": canonical_json(parent.to_dict()),
                "research_iteration": iteration.to_dict(), "knowledge_snapshot_digest": "2" * 64,
                "stage_context_digests": {"research_iteration_digest": iteration.iteration_digest,
                                          "knowledge_snapshot_digest": "2" * 64},
                "run_state_revision": state.events[-1].seq, "stage_attempt": 1,
                "ledger_expected_revision": director.ledger.latest_seq()}

    def propose(self, director, adapter, *, parent=None, context=None):
        state = self.novelty_state(director)
        return adapter.propose(state.run, state.task_manifest, state.run.session_id,
                               parent_candidate_id=parent, batch_context=context or self.proposal_context(director))

    def persist_first_and_advance(self, director, adapter):
        proposal = self.propose(director, adapter)
        director.submit_proposal(proposal)
        candidate = director.spawn_candidate("run:novelty", proposal)
        director.fail_candidate("run:novelty", candidate.candidate_id, "fixture: no evaluation requested")
        self.fixture_generation = 1
        return candidate, proposal

    def test_prior_generation_duplicate_is_repaired_before_submission_or_slot_consumption(self):
        ledger, director, adapter, runtime = self.setup_run([parameter(.2), parameter(.2), parameter(.3)])
        original, first = self.persist_first_and_advance(director, adapter)
        before = ledger.latest_seq()
        second = self.propose(director, adapter, parent=original.candidate_id)
        self.assertEqual(ledger.latest_seq(), before)  # rejected output never entered the ledger
        self.assertEqual(second.changes["ridge_alpha"], .3)
        self.assertNotEqual(second.metadata["behavior_digest"], first.metadata["behavior_digest"])
        self.assertEqual(len(runtime.requests), 3)
        rejection = runtime.requests[-1]["request"]["context"]["evolution_reflection"]["host_rejections"][0]
        self.assertEqual(rejection["reason"], "prior_generation_behavior_duplicate")
        self.assertEqual(rejection["prior_candidate_id"], original.candidate_id)
        self.assertEqual(rejection["rejected_operations"], parameter(.2)["operations"])
        self.assertNotEqual(runtime.requests[-2]["stage_attempt"], runtime.requests[-1]["stage_attempt"])
        state = director.state("run:novelty")
        self.assertEqual(second.generation, 1)
        self.assertEqual(len(state.proposals), 1)
        self.assertEqual(len(state.candidates), 1)
        self.assertFalse(state.evaluations)  # no score reuse or hidden evaluation

    def test_exhausted_duplicate_repairs_do_not_commit_a_candidate(self):
        ledger, director, adapter, runtime = self.setup_run([parameter(.2)])
        original, _ = self.persist_first_and_advance(director, adapter)
        before = ledger.latest_seq()
        with self.assertRaisesRegex(ValueError, "prior-generation behavior after bounded retry"):
            self.propose(director, adapter, parent=original.candidate_id)
        self.assertEqual(len(runtime.requests), 5)  # first proposal, then initial attempt + three bounded repairs
        self.assertEqual(ledger.latest_seq(), before)
        self.assertEqual(len(director.state("run:novelty").candidates), 1)

    def test_disabled_optional_novelty_has_no_archive_context(self):
        _, director, adapter, runtime = self.setup_run([parameter(.2)], enabled=False)
        adapter._candidate_novelty_provider = lambda *args: self.fail("historical run must not request novelty history")
        original, first = self.persist_first_and_advance(director, adapter)
        second = self.propose(director, adapter, parent=original.candidate_id)
        self.assertEqual(second.metadata["behavior_digest"], first.metadata["behavior_digest"])
        self.assertEqual(len(runtime.requests), 2)
        self.assertNotIn(novelty.ARCHIVE_KEY, runtime.requests[-1]["request"]["context"]["mutation_contract"])

    def test_same_scientific_parameters_with_distinct_agent_program_are_allowed(self):
        agent = {"schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": [
            {"op": "select_instruction_template", "role": "sample-planner",
             "instruction_template_id": "sample-planner-horizon-aware@1"}]}
        other_agent = deepcopy(agent)
        other_agent["operations"][0]["instruction_template_id"] = "sample-planner-model-anchored@1"
        _, director, adapter, runtime = self.setup_run([agent, other_agent])
        original, first = self.persist_first_and_advance(director, adapter)
        second = self.propose(director, adapter, parent=original.candidate_id)
        self.assertEqual(len(runtime.requests), 2)
        self.assertEqual(second.changes, first.changes)
        self.assertNotEqual(second.metadata["behavior_digest"], first.metadata["behavior_digest"])

    def test_archive_is_stable_after_current_generation_appends_and_director_restart(self):
        ledger, director, adapter, _ = self.setup_run([parameter(.2), parameter(.3)])
        original, _ = self.persist_first_and_advance(director, adapter)
        before = novelty.build_archive(self.novelty_state(director), 1)
        second = self.propose(director, adapter, parent=original.candidate_id)
        replayed = EvolutionDirector(ledger, adapter).state("run:novelty")
        current = replace(original, candidate_id='candidate:current', proposal_id=second.proposal_id, generation=1)
        replayed = replace(replayed, run=replace(replayed.run,generation=1),
            proposals=(*replayed.proposals,second), candidates=(*replayed.candidates,current))
        self.assertEqual(before, novelty.build_archive(replayed, 1))
        self.assertEqual(len(before["entries"]), 1)
        self.assertNotIn("score", canonical_json(before))
        self.assertNotIn("observed", canonical_json(before))

    def test_missing_or_changed_reference_fails_before_remote_call(self):
        _, director, adapter, runtime = self.setup_run([parameter(.2)])
        context = self.proposal_context(director)
        for reference in (None, {"archive_digest": "0" * 64}):
            altered = deepcopy(context)
            altered["research_iteration"]["plan"][novelty.ARCHIVE_KEY] = reference
            # Rebuild the enclosing trusted iteration identity, leaving only
            # the archive-reference mismatch for the proposer to reject.
            iteration = ResearchIteration(run_id="run:novelty", generation=0, status="model_generated",
                plan=altered["research_iteration"]["plan"],
                knowledge_snapshot_digest="2" * 64)
            altered["research_iteration"] = iteration.to_dict()
            altered["stage_context_digests"]["research_iteration_digest"] = iteration.iteration_digest
            with self.assertRaisesRegex(ValueError, "frozen research reference"):
                self.propose(director, adapter, context=altered)
        self.assertFalse(runtime.requests)

    def test_real_research_batch_persists_only_reference_and_rebuilds_full_archive(self):
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        runtime = _CycleRuntime()
        task = _task()
        task = replace(task, metadata={**task.metadata, novelty.CANDIDATE_NOVELTY_POLICY_KEY: novelty.CANDIDATE_NOVELTY_POLICY})
        adapter = StrategyRouterDSHAdapter(gateway=object(), native_runtime_provider=lambda: runtime,
            candidate_novelty_provider=lambda rid, generation: novelty.build_archive(director.state(rid), generation))
        director = EvolutionDirector(ledger, adapter)
        start_native_run(director,task=task,run_id="run:research-novelty")
        batch = start_generation_batch(director, "run:research-novelty")
        state = director.state("run:research-novelty")
        plan = state.research_iteration_for(0).plan
        reference = novelty.archive_reference(novelty.build_archive(state, 0))
        self.assertEqual(plan[novelty.ARCHIVE_KEY], reference)
        self.assertNotIn("entries", plan[novelty.ARCHIVE_KEY])
        validate_research_plan(plan)
        proposal = director.request_proposal("run:research-novelty", generation_batch=batch, slot_index=0)
        director.spawn_candidate("run:research-novelty", proposal, slot_index=0)
        self.assertEqual(novelty.archive_reference(novelty.build_archive(director.state("run:research-novelty"), 0)), reference)
        self.assertEqual(runtime.requests[-1]["request"]["context"]["mutation_contract"][novelty.ARCHIVE_KEY]["archive_digest"], reference["archive_digest"])

    def test_joint_research_keeps_history_directional_and_leaves_hard_novelty_to_proposer(self):
        first_mutation = {"schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": [
            {"op": "select_instruction_template", "role": "sample-planner",
             "instruction_template_id": "sample-planner-horizon-aware@1"}]}
        direction = {**_direction(0, []), "mutation_axis": "instruction_profile",
                     "mutation_target": "sample-planner-horizon-aware@1", "mutation_direction": "select"}
        repeated = {"schema_version": "ecologyrsi-dsh.research-synthesis/1", "summary": "Test a planner template.",
                    "evidence": [], "candidate_directions": [direction]}
        corrected = deepcopy(repeated)
        corrected["candidate_directions"][0]["mutation_target"] = "sample-planner-model-anchored@1"
        ledger, director, adapter, runtime = self.setup_run([first_mutation, repeated, corrected], task=_task())
        proposal = self.propose(director, adapter, context=self.proposal_context(director, direction))
        director.submit_proposal(proposal)
        candidate = director.spawn_candidate("run:novelty", proposal)
        director.fail_candidate("run:novelty", candidate.candidate_id, "fixture: no evaluation requested")
        self.fixture_generation = 1
        state = self.novelty_state(director)
        search = GenerationSearchPlan(run_id=state.run.run_id, generation=1,
            search_queries=("greenhouse causal forecasting",), focus_areas=("planner",), rationale="Change one planner strategy.")
        knowledge = KnowledgeSnapshot(run_id=state.run.run_id, generation=1, query_terms=search.search_queries,
            cards=(), online_enabled=False, provider="frozen", retrieval_status="catalog_only")
        archive = novelty.build_archive(state, 1)
        before = ledger.latest_seq()
        plan = adapter.research_plan("fixture-model", run=state.run, task=state.task_manifest,
            parent_genome=state.materialized_seed_genome().to_dict(), generation_search_plan=search.to_dict(),
            knowledge_snapshot=knowledge.proposal_context(), candidate_count=1,
            cross_generation_experience={novelty.ARCHIVE_KEY: novelty.archive_reference(archive)})
        self.assertEqual(plan["candidate_directions"][0]["mutation_target"], "sample-planner-horizon-aware@1")
        self.assertEqual(len(runtime.requests), 2)
        self.assertIsNone(runtime.requests[-1]["request"]["context"].get("host_validation_feedback"))
        self.assertEqual(plan[novelty.ARCHIVE_KEY], novelty.archive_reference(archive))
        self.assertEqual(ledger.latest_seq(), before)

    def test_prior_history_does_not_exhaust_open_directive_authoring_space(self):
        _, director, _, _ = self.setup_run([parameter(.2)])
        state = director.state("run:novelty")
        direction = {**_direction(0, []), "mutation_axis": "instruction_directive",
                     "mutation_target": AUTHORED_DIRECTIVE_POLICY_ID, "mutation_direction": "author"}
        kwargs = {"run": state.run, "task": state.task_manifest,
                  "parent": state.materialized_seed_genome()}
        sampled = []
        # Exhaust the finite Host witness set, which cannot represent every
        # legal authored directive the model may actually submit.
        for _ in range(32):
            try:
                result = _validate_candidate_direction_realizability([direction], avoid_behaviors=sampled, **kwargs)
            except ValueError as error:
                self.assertIn("hard-avoided behavior", str(error))
                break
            sampled.append({"behavior_digest": result["checks"][0]["witness_behavior_digest"]})
        else:
            self.fail("fixture should exhaust the finite directive witness set")
        self.assertTrue(sampled)
        result = _validate_candidate_direction_realizability([direction], avoid_behaviors=[],
            prior_generation_behaviors=sampled, **kwargs)
        self.assertEqual(result["checks"][0]["status"], "single_operation_feasible")

    def test_complete_history_does_not_exhaust_continuous_parameter_witnesses(self):
        _, director, _, _ = self.setup_run([parameter(.2)])
        state = director.state("run:novelty")
        parent = state.materialized_seed_genome()
        direction = _direction(0, [])
        sampled = []
        # Fill the previous fixed witness window without changing the frozen
        # task/parent identity; the actual check below uses the new policy.
        with patch("ecologyrsi_dsh.evolution.strategies.novelty.enabled", return_value=False):
            for _ in range(16):
                result = _validate_candidate_direction_realizability([direction], run=state.run, task=state.task_manifest,
                    parent=parent, avoid_behaviors=sampled)
                sampled.append({"behavior_digest": result["checks"][0]["witness_behavior_digest"]})
        result = _validate_candidate_direction_realizability([direction], run=state.run, task=state.task_manifest,
            parent=parent, avoid_behaviors=[], prior_generation_behaviors=sampled)
        self.assertNotIn(result["checks"][0]["witness_behavior_digest"], {row["behavior_digest"] for row in sampled})


if __name__ == "__main__":
    unittest.main()
