"""Budget, isolation, frozen race identities, and resumable paired execution."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from ecologyrsi_dsh import EventLedger, EvolutionDirector, FakeDSHAdapter
from ecologyrsi_dsh.application import generation_execution as execution, formal_trajectory
from ecologyrsi_dsh.core.models import Proposal, canonical_json, digest
from ecologyrsi_dsh.core.trajectory import CandidateRevision, RevisionStatus
from ecologyrsi_dsh.core.training_race import race_bindings
from ecologyrsi_dsh.evolution.genome import apply_genome_mutation, GenomeMutationContextV1, MUTATION_SCHEMA_VERSION
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evolution.workflow_ir import resolve_candidate_agent_profile
from ecologyrsi_dsh.evaluators.epoch_cohorts import (
    plan_run_adaptation_cohort, plan_generation_selection_cohorts,
    estimate_epoch_capacity, GenerationCohorts,
)
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from tests.test_dsh_structured_roles import _native_task
from tests.test_epoch_cohort_planning import dataset_fixture
from tests import test_formal_trajectory as fixtures


class EvidenceGuidedPlanTests(unittest.TestCase):
    def test_fixed_cost_and_fresh_value_blind_cohorts(self):
        schedule = OptimizationSchedule.for_evidence_guided_run()
        plan = schedule.execution_plan(4, cells_per_origin=9)
        self.assertEqual(plan['generation_budget'], dict(screening_candidate_origins=50,
            formal_candidate_origins=50, holdout_candidate_origins=100,
            total_candidate_origins=200, total_scoring_cells=1800))
        self.assertEqual(plan['run_budget']['total_candidate_origins'], 800)
        self.assertEqual(plan['required_unique_origins'], 340)
        self.assertEqual(plan['qualification'], 'exploratory_only')
        plans = []
        for labels in (False, True):
            dataset = dataset_fixture(3000, changed_labels=labels)
            initial = plan_run_adaptation_cohort(dataset, schedule=schedule, seed=0)
            cohorts = [plan_generation_selection_cohorts(dataset, schedule=schedule,
                generation=g, adaptation=initial, seed=0) for g in range(4)]
            ids = []
            prior_maturity = -1
            for c in cohorts:
                self.assertEqual(GenerationCohorts.from_dict(c.to_dict()), c)
                self.assertEqual([b.origin_count for b in c.adaptation.batches], [10, 25])
                self.assertEqual(c.screening.origin_ids, c.adaptation.batches[0].origin_ids)
                for stage in (c.screening, c.adaptation.batches[1].cohort, c.holdout):
                    self.assertGreater(stage.origins[0].origin_timestamp, prior_maturity)
                    prior_maturity = stage.origins[-1].maximum_target_timestamp
                    ids.extend(stage.origin_ids)
            self.assertEqual(len(set(ids)), 340)
            self.assertTrue(estimate_epoch_capacity(dataset, schedule=schedule,
                planned_generations=4, seed=0).sufficient)
            plans.append(cohorts)
        self.assertEqual(*plans)
        from ecologyrsi_dsh.evolution.evidence_capacity import guarded_cohort_evidence_capacity
        capacity = guarded_cohort_evidence_capacity(dataset_fixture(3000), schedule=schedule,
            planned_generations=4, seed=0)
        self.assertEqual(len(capacity['formal_batches']), 8)
        self.assertEqual({r['generation'] for r in capacity['formal_batches']}, {0, 1, 2, 3})
        self.assertTrue(all(r['minimum_day_blocks'] == 3 for r in capacity['formal_batches'] if r['batch_index'] == 1))
        with self.assertRaises(ValueError):
            replace(schedule, local_batch_origin_count=10)


class EvidenceGuidedRaceTests(unittest.TestCase):
    def setUp(self):
        self.ledger = EventLedger()
        self.addCleanup(self.ledger.close)
        self.director = EvolutionDirector(self.ledger, FakeDSHAdapter())
        self.schedule = OptimizationSchedule.for_evidence_guided_run()
        base = _native_task()
        self.task = replace(base, budget={'max_generations': 4, 'candidates_per_generation': 4, 'max_candidates': 16},
            metadata={**base.metadata, 'optimization_protocol': self.schedule.protocol,
                'optimization_schedule': self.schedule.to_dict(), 'prediction_cells_per_origin': 1,
                'fitness_profile': {'expected_targets': ['air_temperature'], 'expected_horizons': [1]},
                'episode_id': 'episode:race'})
        self.run_id = 'run:evidence-guided'
        self.director.start_evolution(self.task, run_id=self.run_id)
        self.control = self.director.ensure_seed_incumbent_control(self.run_id)
        parent = self.director.state(self.run_id).materialized_seed_genome()
        registry = current_program_registry()
        self.candidates = []
        for i in range(4):
            context = GenomeMutationContextV1(self.run_id, 0, i, i, None, parent.genome_digest,
                *('a'*64,)*4, 'test-mutation@1')
            child = apply_genome_mutation(parent, {'schema_version': MUTATION_SCHEMA_VERSION,
                'operations': [{'op':'set_bounded_parameter','name':'ridge_alpha','value':.2+i*.1}]}, context, registry)
            proposal = Proposal(proposal_id=f'proposal:race:{i}', run_id=self.run_id, generation=0,
                title=f'race {i}', rationale='bounded fixture', changes=dict(child.scientific_program['parameter_overrides']),
                metadata={'execution_protocol':'dsh_native_plugin_evolution@1',
                    'genome_digest':child.genome_digest,'behavior_digest':child.behavior_digest,
                    'evolution_genome_canonical_json':canonical_json(child.to_dict()),
                    'candidate_agent_profile':resolve_candidate_agent_profile(child, registry)})
            self.director.submit_proposal(proposal)
            c = self.director.spawn_candidate(self.run_id, proposal, slot_index=i)
            self.candidates.append(c)
            self.director.create_candidate_revision(self.run_id, CandidateRevision(
                revision_id=f'revision:race:{i}:r0', run_id=self.run_id, generation=0,
                candidate_id=c.candidate_id, genome=child.to_dict(), genome_digest=child.genome_digest,
                behavior_digest=child.behavior_digest, mutation_digest=child.lineage['mutation_digest'],
                status=RevisionStatus.ACTIVE))
        dataset = dataset_fixture(3000)
        dataset = SimpleNamespace(dataset_id=self.task.dataset, episode_id='episode:race',
            timestamps=dataset.timestamps, partitions=dataset.partitions)
        a = plan_run_adaptation_cohort(dataset, schedule=self.schedule, seed=self.task.seed)
        self.cohorts = replace(plan_generation_selection_cohorts(dataset, schedule=self.schedule,
            generation=0, adaptation=a, seed=self.task.seed), revision_bindings=race_bindings(self.director.state(self.run_id), 0))
        self.director.freeze_run_adaptation_cohort(self.run_id, a)
        self.director.freeze_generation_selection_cohorts(self.run_id, self.cohorts)
        self.evaluator = fixtures._PairedLaneEvaluator('revision:race:0:r0')
        self.services = SimpleNamespace(director=self.director, ledger=self.ledger, evaluators=self.evaluator)

    def _screen(self, candidate, *, extra_metrics=None):
        state = self.director.state(self.run_id)
        revision = self.cohorts.revision_bindings[candidate.candidate_id]
        scope = SimpleNamespace(candidate_revision_id=revision, batch_index=0,
            formal_batch_arm=None, origin_count=10)
        metrics = self.evaluator.evaluate_scientific(None, None, None, scope=scope).evaluation.metrics
        metrics['screening_evaluator_digest'] = 'race-fixture@1'
        if extra_metrics:
            metrics.update(extra_metrics)
        return self.director.record_candidate_screening(self.run_id,
            candidate_id=candidate.candidate_id, candidate_revision_id=revision, generation=0,
            score=.2, passed=False, constraint_violations=0, origin_count=10,
            prediction_cell_count=10, cohort_digest=self.cohorts.screening.cohort_digest, metrics=metrics)

    def test_five_arms_freeze_and_resume_without_rescoring_warmup(self):
        # Selection requires all five arms, including the control, to be sealed.
        for c in (*self.candidates, self.control):
            self._screen(c)
        self.evaluator.calls.clear()
        chosen = execution._prepare_formal_finalists(self.services, self.run_id, self.candidates, max_concurrency=1)
        self.assertEqual([c.candidate_id for c in chosen], [self.candidates[0].candidate_id])
        cid = chosen[0].candidate_id
        with patch.object(formal_trajectory, 'EvaluationSession', fixtures._NoopScopedCallbacks), \
             patch.object(formal_trajectory, '_local_edit_context', side_effect=fixtures.PairedFormalTrajectoryTests._local_context), \
             patch.object(formal_trajectory, '_local_edit_proposal', return_value=replace(fixtures.PairedFormalTrajectoryTests._mutate_proposal(), operations=({'op':'set_bounded_parameter','name':'ridge_alpha','value':.7},))):
            self.assertTrue(formal_trajectory.execute_next_formal_batch(self.services, self.run_id, cid))
            self.assertEqual(self.evaluator.calls, [])
            self.assertTrue(formal_trajectory.execute_next_formal_batch(self.services, self.run_id, cid))
            self.assertTrue(formal_trajectory.execute_next_local_edit(self.services, self.run_id, cid))
            for _ in range(4):
                formal_trajectory.execute_next_formal_batch(self.services, self.run_id, cid)
        state = self.director.replay(self.run_id)
        self.assertEqual(len(self.evaluator.calls), 2)
        self.assertEqual(state.trajectory_for(cid).status.value, 'completed')
        self.assertTrue(state.batch_evaluation_for(cid, 0).metrics['reused_screening_evidence'])
        self.assertEqual(state.trajectory_for(cid).final_revision_id, 'revision:race:0:r0')

    def test_wrong_revision_is_rejected_before_scoring(self):
        c = self.candidates[0]
        with self.assertRaisesRegex(ValueError, 'frozen revision'):
            self.director.record_candidate_screening(self.run_id, candidate_id=c.candidate_id,
                candidate_revision_id='revision:race:1:r0', generation=0, score=.9, passed=True,
                constraint_violations=0, origin_count=10, prediction_cell_count=10,
                cohort_digest=self.cohorts.screening.cohort_digest, metrics={})

    def test_hard_effect_contradiction_loses_race_and_local_admission(self):
        from ecologyrsi_dsh.evolution.champion_challenger import local_challenger_safety_reason
        from ecologyrsi_dsh.core.immutable import freeze_json
        receipt = {'schema_version': 'ecologyrsi-dsh.edit-effect-receipt/1',
                   'checks': [{'semantics': 'hard', 'status': 'failed'}]}
        receipt['receipt_digest'] = digest(receipt)
        metrics = {'sample_execution': {'mutation_effect_receipt': receipt}}
        self.assertEqual(local_challenger_safety_reason(freeze_json(metrics)), 'hard_mutation_effect_failed')
        self._screen(self.candidates[0], extra_metrics=metrics)
        for c in (*self.candidates[1:], self.control):
            self._screen(c)
        chosen = execution._prepare_formal_finalists(self.services, self.run_id, self.candidates, max_concurrency=1)
        self.assertEqual([c.candidate_id for c in chosen], [self.candidates[1].candidate_id])
        self.assertEqual(list(self.director.replay(self.run_id).formal_selection_for(0).payload['selected_candidate_ids']),
                         [self.candidates[1].candidate_id])

    def test_diagnostic_reuse_rejects_score_or_evidence_rewriting(self):
        from ecologyrsi_dsh.core.training_race import validate_race_diagnostic_reuse
        from ecologyrsi_dsh.core.trajectory import BatchEvaluation, EvaluationScope, EvaluationPhase
        from ecologyrsi_dsh.core.immutable import thaw_json
        c = self.candidates[0]
        self._screen(c)
        source = self.director.state(self.run_id).screening_for(0, c.candidate_id)
        metrics = {**thaw_json(source.payload['metrics']), 'source_screening_event_id': source.event_id,
                   'reused_screening_evidence': True, 'additional_prediction_executions': 0}
        scope = EvaluationScope(run_id=self.run_id, generation=0, candidate_id=c.candidate_id,
            candidate_revision_id=self.cohorts.revision_bindings[c.candidate_id],
            phase=EvaluationPhase.FORMAL_BATCH, batch_index=0, origin_count=10,
            cohort_digest=self.cohorts.adaptation.batches[0].cohort.cohort_digest)
        evaluation = BatchEvaluation(evaluation_id='reuse-check', scope=scope, score=source.payload['score'],
            passed=source.payload['passed'], metrics=metrics,
            evaluator_digest=digest({'evaluator': metrics['screening_evaluator_digest']}))
        validate_race_diagnostic_reuse(evaluation, source)
        for edited in (replace(evaluation, score=.9, metrics=metrics), replace(evaluation, metrics={**metrics, 'additional_prediction_executions': 10})):
            with self.assertRaisesRegex(ValueError, 'cannot alter'):
                validate_race_diagnostic_reuse(edited, source)

    def test_later_race_only_admits_the_bound_historical_incumbent(self):
        from ecologyrsi_dsh.core.models import CandidateStatus
        from ecologyrsi_dsh.core.trajectory import EvaluationScope, EvaluationPhase
        from tests.test_scoped_sample_execution import ScopedSampleExecutionTests
        state = self.director.state(self.run_id)
        incumbent = replace(self.control, status=CandidateStatus.SCREENED_OUT)
        planned = replace(self.cohorts, generation=1)
        later_state = replace(state, run=replace(state.run, generation=1),
                              generation_selection_cohorts=(self.cohorts, planned))
        scope = EvaluationScope(run_id=self.run_id, generation=1, candidate_id=incumbent.candidate_id,
            candidate_revision_id=planned.revision_bindings[incumbent.candidate_id], phase=EvaluationPhase.SCREENING,
            cohort_digest=planned.screening.cohort_digest, origin_count=10)
        checkpoint = {**ScopedSampleExecutionTests._checkpoint(scope), 'sample_count': 10}
        self.director._validate_evaluation_scope(later_state, incumbent, scope)
        self.assertTrue(self.director._sample_checkpoint_generation_allowed(later_state, incumbent, 1, checkpoint, scope))
        self.assertTrue(self.director._sample_checkpoint_candidate_status_allowed(later_state, incumbent, checkpoint, generation=1))
        # The same whitelist must hold when appending/resuming sample batches,
        # not just at the initial checkpoint admission boundary.
        start = self.ledger.append(self.run_id, 'EvaluationSampleResultsStarted', {
            'run_id': self.run_id, 'candidate_id': incumbent.candidate_id,
            'proposal_id': incumbent.proposal_id, 'generation': 1,
            'revision': 'sample-revision:later-race', 'checkpoint': checkpoint})
        later_state = replace(later_state, events=(*state.events, start))
        actual_start, batches, completion = self.director._sample_result_revision_events(
            later_state, candidate_id=incumbent.candidate_id, revision='sample-revision:later-race')
        self.assertEqual(actual_start.event_id, start.event_id)
        self.assertFalse(batches)
        self.assertIsNone(completion)
        for changed in ({'candidate_revision_id': 'another-revision'}, {'cohort_digest': 'b'*64}, {'evaluation_phase': 'formal_batch'}):
            self.assertFalse(self.director._sample_checkpoint_candidate_status_allowed(
                later_state, incumbent, {**checkpoint, **changed}, generation=1))

    def test_historical_incumbent_uses_current_host_reliability_plan(self):
        from ecologyrsi_dsh.evolution.analysis import GenerationAnalysis
        from ecologyrsi_dsh.evolution.execution_plan import derive_execution_plan
        state = self.director.state(self.run_id)
        analysis = GenerationAnalysis(run_id=self.run_id, generation=0, candidate_count=4,
            eligible_count=0, outcome='no_eligible_candidate', sample_failures=({
                'attempted': 10, 'failed': 3, 'coverage_pass': False,
                'failure_counts': {'tool_timeout': 2, 'constraint_rejected': 1}},))
        state = replace(state, generation_analyses=(analysis,))
        revision_id = self.cohorts.revision_bindings[self.control.candidate_id]
        candidate, proposal, compiled = execution._holdout_replay_inputs(
            state, self.control, 1, revision_id, self.task)
        expected = derive_execution_plan(analysis).to_dict()
        self.assertEqual(expected['sample_max_attempts'], 6)
        self.assertEqual(dict(proposal.metadata['derived_execution_plan']), expected)
        self.assertEqual(dict(compiled.derived_execution_plan), expected)
        self.assertEqual(proposal.metadata['genome_digest'], state.revision(revision_id).genome_digest)
        self.assertEqual(candidate.generation, 1)
        self.assertEqual(proposal.metadata['agent_policy']['inference']['prediction_formula_policy'],
                         'mean-referenced-tools@1')

    def test_selection_requires_the_control_receipt(self):
        from ecologyrsi_dsh.core.screening import screening_cohort_digest
        for c in self.candidates:
            self._screen(c)
        records = [e.payload for e in self.director.state(self.run_id).candidate_screening_events]
        with self.assertRaisesRegex(ValueError, 'five race arms'):
            self.director.freeze_formal_selection_cohort(self.run_id, generation=0,
                selected_candidate_ids=[self.candidates[0].candidate_id],
                screening_digest=screening_cohort_digest(records))

    def test_scoring_entry_point_evaluates_all_five_once(self):
        calls = []
        def evaluate(task, candidate, proposal, **kwargs):
            scope = kwargs['scope']
            calls.append((candidate.candidate_id, scope.cohort_digest, scope.origin_count))
            result = self.evaluator.evaluate_scientific(task, candidate, proposal,
                scope=replace(scope, phase='formal_batch', batch_index=0))
            result.evaluation.metrics['sample_execution']['prediction_cell_count'] = 10
            result.evaluation.metrics['feedback_update_cohort_digest'] = scope.cohort_digest
            return result
        evaluator = SimpleNamespace(evaluate_scientific=evaluate)
        services = SimpleNamespace(director=self.director, ledger=self.ledger, evaluators=evaluator)
        with patch.object(execution, 'EvaluationSession', fixtures._NoopScopedCallbacks):
            selected = execution._prepare_formal_finalists(services, self.run_id,
                self.candidates, max_concurrency=1)
            execution._prepare_formal_finalists(services, self.run_id, self.candidates, max_concurrency=1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(calls), 5)
        self.assertEqual(sum(row[2] for row in calls), 50)
        self.assertEqual({row[1] for row in calls}, {self.cohorts.screening.cohort_digest})


if __name__ == '__main__':
    unittest.main()
