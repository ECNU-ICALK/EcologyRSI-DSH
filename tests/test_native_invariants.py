from dataclasses import replace
from types import SimpleNamespace
import unittest
from ecologyrsi_dsh.core.models import Proposal
from ecologyrsi_dsh import EventLedger, EvolutionDirector
from ecologyrsi_dsh.evolution.analysis import GenerationAnalysis
from ecologyrsi_dsh.evolution.diagnosis import diagnose_generation, hypothesis_for_proposal
from ecologyrsi_dsh.evolution.strategies import _native_evolution_reflection_from_experience
from tests.test_core import manifest
from tests.native_fixtures import native_manifest, NativeRoleStub


class ResearchReviewTests(unittest.TestCase):

    def test_only_candidate_scientific_failures_become_advice_never_a_global_ban(self):
        behaviors = [{'behavior_digest': str(i)*64, 'classification': label} for i, label in enumerate(('eligible', 'execution_failed', 'judge_unavailable', 'scientific_gate_failed'), 1)]
        reflection = _native_evolution_reflection_from_experience({'generations': [{'outcome': 'no_eligible_candidate', 'modifications': {'candidate_behaviors': behaviors}}]}, current_run_id='run')
        self.assertEqual(reflection['avoid_behaviors'], [])
        self.assertEqual([r['behavior_digest'] for r in reflection['review_behaviors']], ['4'*64])
        self.assertFalse(reflection['policy']['host_enforced'])

    def test_diagnosis_and_hypothesis_bind_visible_evidence_and_actual_changes(self):
        analysis = GenerationAnalysis('run', 0, 1, 0, 'no_eligible_candidate',
            common_failures=('execution_failed',), insufficient_evidence=True,
            target_weaknesses=({'target': 'humidity', 'horizon_hours': 24, 'median_skill_score': -.2},))
        state = SimpleNamespace(run=SimpleNamespace(generation=1), task_manifest=manifest(2),
            analysis_for=lambda _: analysis, comparison_for=lambda _: None)
        diagnostic = diagnose_generation(state, SimpleNamespace(snapshot_digest='a'*64))
        self.assertEqual(diagnostic.weak_cells, ('humidity@24h',))
        self.assertIn('insufficient_evidence', diagnostic.failure_patterns)
        self.assertIn('analysis:'+analysis.analysis_digest, diagnostic.evidence_refs)
        self.assertTrue(any('不是科学模型失效' in cause for cause in diagnostic.possible_causes))
        direction = {'target_weakness':'humidity@24h', 'hypothesis':'longer history may help', 'mutation_axis':'scientific_parameter', 'success_criterion':'paired loss decreases', 'evidence_refs':['knowledge:'+'a'*64]}
        proposal = Proposal('proposal', 'run', 1, 'test', {'history_steps': 12}, metadata={'candidate_direction': direction})
        evidence = hypothesis_for_proposal(proposal, diagnostic)
        changed = hypothesis_for_proposal(replace(proposal, changes={'history_steps': 24}), diagnostic)
        self.assertNotEqual(evidence['hypothesis']['change_spec_ref'], changed['hypothesis']['change_spec_ref'])
        self.assertEqual(evidence['change_spec']['parameters'], {'history_steps': 12})
        self.assertEqual(evidence['evidence_class'], 'exploratory')


class ProposalCommitReviewTests(unittest.TestCase):
    def test_pause_between_provider_response_and_commit_fences_the_proposal(self):
        from tests.test_genome_replay import GenomeReplayTests
        fixture = GenomeReplayTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        run_id = fixture._start_new("run:proposal-race")
        proposal = fixture._proposal(run_id)
        fixture.director.pause_run(run_id)
        before = fixture.ledger.count()
        with self.assertRaises(RuntimeError):
            fixture.director.submit_proposal(proposal)
        self.assertEqual(fixture.ledger.count(), before)
        self.assertFalse(fixture.director.state(run_id).proposals)


class FrozenNativeContractReviewTests(unittest.TestCase):
    def test_research_plan_remains_bound_to_its_recorded_identity(self):
        from ecologyrsi_dsh.knowledge.research_iteration import ResearchIteration

        task = native_manifest()
        plan = {"prediction_model": {"id": "greenhouse-horizon-targetwise-ridge@1"}, "notes": ["initial"]}
        iteration = ResearchIteration(
            run_id="frozen-research", generation=0, status="model_generated",
            plan=plan,
            knowledge_snapshot_digest="a" * 64,
        )
        original = iteration.iteration_digest
        plan["notes"].append("caller edit")
        exported = iteration.to_dict()
        exported["plan"]["notes"].append("export edit")
        self.assertEqual(iteration.plan["notes"], ["initial"])
        self.assertEqual(ResearchIteration.from_dict(iteration.to_dict()).iteration_digest, original)
        with self.assertRaises(TypeError):
            iteration.plan["prediction_model"]["id"] = "changed@1"

    def test_task_budget_nested_metadata_and_cached_events_are_read_only(self):
        from ecologyrsi_dsh.core.models import canonical_json, digest
        metadata = {'nested': {'values': [1, 2]}}
        task = native_manifest(task_id='freeze', metadata=metadata)
        original = task.digest
        metadata['nested']['values'].append(3)
        self.assertEqual(task.digest, original)
        wire = task.to_dict()
        wire['metadata']['nested']['values'].append(9)
        self.assertEqual(task.digest, original)
        self.assertEqual(original, digest(task.to_dict()))
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            director.create_run(task, run_id='frozen-root')
            state = director.state('frozen-root')
            operations = [
                lambda: state.task_manifest.budget.update(max_candidates=999),
                lambda: state.task_manifest.metadata['nested']['values'].append(8),
                lambda: state.events[0].payload['task_manifest']['metadata']['nested'].clear(),
            ]
            for operation in operations:
                with self.assertRaisesRegex(TypeError, 'frozen_json'):
                    operation()
            self.assertEqual(director.state('frozen-root').task_manifest.digest, original)
            self.assertEqual(canonical_json(task.to_dict()), canonical_json(state.task_manifest.to_dict()))

    def test_evaluation_and_proposal_exports_are_detached_mutable_copies(self):
        from ecologyrsi_dsh.core.models import Evaluation
        proposal = Proposal('p', 'r', 0, 'probe', {'weights': [1, 2]}, metadata={'nested': {'value': 3}})
        exported = proposal.to_dict(); exported['changes']['weights'][0] = 9
        exported['metadata']['nested']['value'] = 4
        self.assertEqual(proposal.changes['weights'], [1, 2])
        self.assertEqual(proposal.metadata['nested']['value'], 3)
        with self.assertRaises(TypeError): proposal.changes['weights'] *= 2
        evaluation = Evaluation('e', 'r', 'c', .2, True, metrics={'cell': {'values': [1]}})
        copied = evaluation.to_dict(); copied['metrics']['cell']['values'].clear()
        self.assertEqual(evaluation.metrics['cell']['values'], [1])
