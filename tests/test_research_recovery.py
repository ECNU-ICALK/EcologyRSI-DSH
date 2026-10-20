"""Durable research correction reuses submissions, never completed evaluations."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.immutable import freeze_json
from ecologyrsi_dsh.evolution.research_recovery import research_checkpoint
from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter, _mutation_contract_catalog
from ecologyrsi_dsh.evaluators.skill_program import skill_preflight_programs
from tests import test_research_context as context_tests
from tests.test_joint_plugin_evolution import mutate


class ResearchRecoveryTests(unittest.TestCase):
    def setUp(self):
        fixture = context_tests.ResearchContextTests()
        fixture.setUp()
        self.run, self.task, self.parent = fixture.run, fixture.task, fixture.parent
        self.search, self.knowledge = fixture.search, fixture.knowledge
        self.raw = context_tests.minimal_aligned_synthesis()

    def checkpoint(self, structured=None):
        raw = deepcopy(self.raw if structured is None else structured)
        return dict(run_id=self.run.run_id, generation=self.run.generation,
            parent_genome_digest=self.parent.genome_digest,
            search_plan_digest=self.search.search_plan_digest,
            knowledge_snapshot_digest=self.knowledge['snapshot_digest'],
            event_id='event:submitted', event_seq=10,
            result_digest=digest(raw), structured=raw)

    def invoke(self, checkpoint, runtime):
        adapter = StrategyRouterDSHAdapter(gateway=object())
        with patch.object(adapter, '_native_runtime', return_value=SimpleNamespace(run=runtime)):
            return adapter.research_iteration(run=self.run, task=self.task,
                parent_genome=self.parent.to_dict(), knowledge_snapshot=self.knowledge,
                generation_search_plan=self.search.to_dict(), candidate_count=4,
                previous_generation_analysis=None, previous_knowledge_assessment=None, current_plan={},
                research_recovery=checkpoint, stage_attempt=2)

    def test_valid_submission_finishes_host_commit_without_another_model_call(self):
        def forbidden(**request):
            self.fail('valid checkpoint must be revalidated locally without a provider request')
        result = self.invoke(freeze_json(self.checkpoint()), forbidden)
        self.assertEqual(result['plan']['dsh_research_summary'], self.raw['summary'])
        self.assertEqual(len(result['plan']['candidate_direction_preflight']['checks']), 4)

    def test_rejected_checkpoint_is_corrected_not_silently_rewritten_or_regenerated(self):
        self.raw['candidate_directions'][2].update(mutation_axis='skill_program',
            mutation_target='causal_planner_skills@1', mutation_direction='revise')
        checkpoint = self.checkpoint()
        before = deepcopy(checkpoint)
        calls = []
        def repair(**request):
            calls.append(request)
            corrected = deepcopy(self.raw)
            corrected['candidate_directions'][2]['mutation_direction'] = 'author'
            return corrected
        result = self.invoke(checkpoint, repair)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['stage_attempt'], 5)
        self.assertEqual(checkpoint, before)
        context = calls[0]['context']
        self.assertEqual(context['rejected_research_synthesis'], self.raw)
        self.assertIn('requires author', context['host_validation_feedback']['validation_detail'])
        self.assertEqual(result['plan']['candidate_directions'][2]['mutation_direction'], 'author')

    def test_stale_or_tampered_checkpoint_fails_before_provider_call(self):
        for key, value in [('run_id', 'run:other'), ('generation', 1),
                           ('parent_genome_digest', 'a' * 64),
                           ('search_plan_digest', 'b' * 64),
                           ('knowledge_snapshot_digest', 'c' * 64),
                           ('result_digest', 'd' * 64)]:
            checkpoint = {**self.checkpoint(), key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'checkpoint'):
                self.invoke(checkpoint, lambda **kw: self.fail('stale checkpoint called provider'))

    def test_ledger_recovery_scopes_submission_after_both_freezes(self):
        with EventLedger(':memory:') as ledger:
            ledger.append(self.run.run_id, 'GenerationSearchPlanned', {'search_plan': self.search.to_dict()})
            frozen = ledger.append(self.run.run_id, 'GenerationKnowledgeRetrieved', {
                'knowledge_snapshot': {'generation': 0}})
            identity = dict(run_id=self.run.run_id, stage='generation.research-synthesis', role='researcher',
                genome_digest=self.parent.genome_digest, run_state_revision=frozen.seq,
                idempotency_key=f'{self.run.run_id}:generation:0:generation.research-synthesis:attempt:2')
            payload = dict(identity=identity, output_schema_id='ecology-research-synthesis@1',
                structured=self.raw, result_digest=digest(self.raw))
            accepted = ledger.append(self.run.run_id, 'DshStructuredResultAccepted', payload)
            # Later requests never submitted. Their text/failure is not a new result.
            ledger.append(self.run.run_id, 'DshChildExecutionFailed', {'error_code': 'structured_child_output_budget_exhausted'})
            state = SimpleNamespace(run=self.run, events=ledger.events(self.run.run_id),
                search_plan_for=lambda gen: self.search,
                knowledge_for=lambda gen: SimpleNamespace(snapshot_digest=self.knowledge['snapshot_digest']))
            checkpoint = research_checkpoint(state, self.parent)
            self.assertEqual(checkpoint['event_id'], accepted.event_id)
            checkpoint['structured']['summary'] = 'A mutable copy, not a ledger edit.'
            self.assertEqual(accepted.payload['structured']['summary'], self.raw['summary'])
            for field, value in [('genome_digest', 'f' * 64), ('role', 'sample-planner'),
                                 ('idempotency_key', f'{self.run.run_id}:generation:1:generation.research-synthesis:attempt:2'),
                                 ('run_state_revision', frozen.seq - 1)]:
                changed = deepcopy(payload)
                changed['identity'][field] = value
                event = replace(accepted, payload=changed)
                with self.subTest(field=field):
                    self.assertIsNone(research_checkpoint(SimpleNamespace(**{**vars(state),
                        'events': (*state.events[:2], event)}), self.parent))
            failed = replace(state.events[-1], payload={'identity': identity,
                'error_code': 'structured_child_execution_budget_exhausted'})
            self.assertIsNone(research_checkpoint(SimpleNamespace(**{**vars(state),
                'events': (*state.events[:-1], failed)}), self.parent))
            ledger.append(self.run.run_id, 'GenerationKnowledgeRetrieved', {'knowledge_snapshot': {'generation': 0}})
            state.events = ledger.events(self.run.run_id)
            self.assertIsNone(research_checkpoint(state, self.parent))

    def test_authoring_contract_tracks_actual_parent_and_does_not_mutate_catalog(self):
        parent = mutate(self.parent, [dict(op='author_skill_program', role='sample-planner',
            skill_program=skill_preflight_programs()[0])])
        for genome, expected in [(self.parent, 'author'), (parent, 'revise'), (self.parent, 'author')]:
            contract = _mutation_contract_catalog(self.task, genome)
            self.assertEqual(contract['mutation_directions_by_axis']['skill_program'], [expected])
            state = contract['parent_authoring_state']
            self.assertEqual(state['genome_digest'], genome.genome_digest)
            self.assertEqual(state['axes']['skill_program']['present'], expected == 'revise')

    def test_an_invalid_saved_submission_does_not_enable_unbounded_semantic_retries(self):
        invalid = {**self.raw, 'summary': 'x' * 4001}
        calls = []
        def still_invalid(**request):
            calls.append(request)
            return deepcopy(invalid)
        with self.assertRaises(ValueError):
            self.invoke(self.checkpoint(invalid), still_invalid)
        self.assertEqual(len(calls), 2)
        self.assertEqual([c['stage_attempt'] for c in calls], [5, 6])


if __name__ == '__main__':
    unittest.main()
