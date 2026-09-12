import json
import threading
from copy import deepcopy
from types import SimpleNamespace
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.parse import quote

from tests import test_http as http_helpers
from ecologyrsi_dsh.application.config import bind_toy_dataset
from ecologyrsi_dsh.application.read_cache import ReadSectionCache
from ecologyrsi_dsh.core.models import TaskManifest
from ecologyrsi_dsh.presentation.training_assets import training_assets


class WorkspaceHTTPTests(unittest.TestCase):
    def test_candidate_list_defers_details_and_artifacts_until_selected(self):
        from ecologyrsi_dsh.api.projection import _candidate_projection
        run, candidate = self.seed()
        with patch('ecologyrsi_dsh.api.workspaces._artifact_projection', side_effect=AssertionError('eager artifact')):
            status, payload = self.request('/api/runs/' + quote(run) + '?view=candidates')
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload['projection']['artifacts'], [])
        self.assertFalse(payload['projection']['candidates'][0]['details_loaded'])
        status, payload = self.request('/api/runs/' + quote(run) + '?view=candidate&candidate_id=' + quote(candidate))
        self.assertEqual(status, 200, payload)
        detail = payload['projection']['candidate']
        self.assertTrue(detail['details_loaded'])
        self.assertEqual(detail['candidate_id'], candidate)
        state = self.server.director.state(run)
        self.assertEqual(detail, {**_candidate_projection(state, state.candidate(candidate)), 'details_loaded': True})
        self.assertEqual(self.request('/api/runs/' + quote(run) + '?view=candidate&candidate_id=foreign')[0], 404)

    setUp = http_helpers.HTTPContractTests.setUp
    tearDown = http_helpers.HTTPContractTests.tearDown
    request = http_helpers.HTTPContractTests.request

    def seed(self):
        task = bind_toy_dataset(TaskManifest(task_id='workspace', objective='loading test',
            domain_pack='crop_soil_water', visible_datasets=('generated-toy-series@1',),
            budget=1, seed=7, metadata={'evaluation_partition': 'validation'}), required=True)
        run = self.server.director.start_evolution(task, run_id='run:workspace')
        candidate = self.server.director.propose_and_spawn(run.run.run_id)
        return run.run.run_id, candidate.candidate_id

    def test_overview_never_builds_hidden_collections(self):
        run, _ = self.seed()
        with (patch('ecologyrsi_dsh.api.projection.training_assets', side_effect=AssertionError('heavy assets')),
             patch('ecologyrsi_dsh.api.projection._candidate_projection', side_effect=AssertionError('heavy candidate')),
             patch('ecologyrsi_dsh.api.projection._rounds_projection', side_effect=AssertionError('heavy round'))):
            status, value = self.request('/api/runs/' + quote(run) + '?view=overview')
        self.assertEqual(status, 200, value)
        for key in ('candidates', 'rounds', 'artifacts', 'training_assets', 'algorithm_attempts'):
            self.assertNotIn(key, value['projection'])
        self.assertEqual(value['projection']['candidates_count'], 1)

    def test_completed_overview_survives_list_reads_without_restoring_full_state(self):
        run, _ = self.seed()
        self.server.director.complete_run(run)
        path = '/api/runs/' + quote(run) + '?view=overview'
        status, first = self.request(path)
        self.assertEqual(status, 200, first)
        self.assertEqual(self.request('/api/runs?view=summary')[0], 200)
        with patch.object(self.server.director, 'state', side_effect=AssertionError('full replay')):
            status, second = self.request(path)
        self.assertEqual(status, 200, second)
        self.assertEqual(first, second)

    def test_completed_overview_rebuilds_after_corruption_source_or_revision_change(self):
        from ecologyrsi_dsh.api.workspaces import overview_projection
        run, _ = self.seed()
        self.server.director.complete_run(run)
        path = '/api/runs/' + quote(run) + '?view=overview'
        self.assertEqual(self.request(path)[0], 200)
        cache = self.server.director._projection_checkpoints
        identity = overview_projection.__module__ + '.' + overview_projection.__qualname__
        cache_path = cache._summary_path(run, identity)
        for field, value in [('revision', -1), ('version', 'other-source'), ('sha256', 'corrupt')]:
            raw = json.loads(cache_path.read_text()); raw[field] = value
            cache_path.write_text(json.dumps(raw))
            with patch.object(self.server.director, 'state', wraps=self.server.director.state) as replay:
                status, result = self.request(path)
            self.assertEqual(status, 200, result)
            replay.assert_called_once_with(run)

    def test_completed_workspaces_restore_separate_snapshots_without_event_replay(self):
        from ecologyrsi_dsh.core.projection_checkpoint import ProjectionCheckpoints
        run, candidate = self.seed()
        self.server.director.complete_run(run)
        paths = ['/api/runs/' + quote(run) + '?view=' + view
                 for view in ('overview', 'process', 'training', 'candidates', 'collaboration')]
        paths.append('/api/runs/' + quote(run) + '?view=asset&candidate_id=' + quote(candidate))
        paths.append('/api/runs/' + quote(run) + '/events?tail=60')
        paths.append('/api/runs/' + quote(run) + '/samples?candidate_id=' + quote(candidate) + '&offset=0&limit=25')
        first = [self.request(path) for path in paths]
        self.assertTrue(all(status == 200 for status, _ in first), first)
        # Fresh cache objects model a restart; no in-memory state or section
        # may be needed to read a validated completed workspace.
        with (patch.object(self.server.director, '_projection_checkpoints', ProjectionCheckpoints(self.server.ledger)),
              patch.object(self.server, 'workspace_cache', ReadSectionCache()),
              patch.object(self.server.director, 'state', side_effect=AssertionError('full replay'))):
            second = [self.request(path) for path in paths]
        self.assertEqual(first, second)

    def test_running_overview_does_not_reuse_completed_snapshot_path(self):
        run, _ = self.seed()
        path = '/api/runs/' + quote(run) + '?view=overview'
        self.assertEqual(self.request(path)[0], 200)
        with patch.object(self.server.director, 'state', wraps=self.server.director.state) as replay:
            self.assertEqual(self.request(path)[0], 200)
        replay.assert_called_once_with(run)

    def test_cold_run_picker_never_replays_history_or_builds_scientific_evidence(self):
        run, candidate = self.seed()
        self.server.director.complete_run(run)
        with (patch.object(self.server.director, 'state', side_effect=AssertionError('cold full replay')),
              patch.object(self.server.ledger, 'events', side_effect=AssertionError('decoded event stream'))):
            status, result = self.request('/api/runs?view=summary&limit=5')
        self.assertEqual(status, 200, result)
        row = result['runs'][0]
        self.assertEqual((row['run_id'], row['status'], row['candidates_count']), (run, 'completed', 1))
        self.assertTrue(row['has_evolution_progress'])
        self.assertEqual(row['projection_revision'], self.server.ledger.latest_run_seq(run))
        for field in ('configuration', 'budget', 'outcome', 'best_candidate_score', 'model_contract_preflight'):
            self.assertNotIn(field, row)

    def test_capacity_read_does_not_wait_for_an_unrelated_run_mutation(self):
        from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
        before = self.server.ledger.latest_seq()
        with self.server.mutation_lock:
            status, result = self.request('/api/evolution-capacity', 'POST', {
                'dataset_id': 'generated-toy-series@1', 'episode_id': None,
                'optimization_schedule': OptimizationSchedule.default().to_dict(),
                'planned_generations': 1,
            })
        self.assertEqual(status, 200, result)
        self.assertEqual(self.server.ledger.latest_seq(), before)

    def test_picker_rejects_hidden_evaluation_partition_without_reading_results(self):
        run, candidate = self.seed()
        self.server.ledger.append(run, 'EvaluationRecorded', {'evaluation': {
            'candidate_id': candidate, 'partition': 'hidden', 'metrics': {'secret': 'not public'}}})
        status, result = self.request('/api/runs?view=summary&limit=5')
        self.assertEqual(status, 400, result)
        self.assertNotIn('not public', json.dumps(result))

    def test_workspaces_have_separate_collections_and_identity(self):
        run, _ = self.seed()
        for view, included, excluded in (
            ('process', 'rounds', 'training_assets'),
            ('candidates', 'artifacts', 'rounds'),
            ('training', 'training_assets', 'candidates'),
            ('collaboration', 'interventions', 'training_assets'),
        ):
            status, result = self.request('/api/runs/' + quote(run) + '?view=' + view)
            self.assertEqual(status, 200, result)
            self.assertEqual(result['view'], view)
            self.assertEqual(result['projection']['run_id'], run)
            self.assertIn(included, result['projection'])
            self.assertNotIn(excluded, result['projection'])

    def test_process_builds_only_summary_without_hidden_evidence_work(self):
        from contextlib import ExitStack
        from ecologyrsi_dsh.api.projection import _candidate_projection
        run, candidate_id = self.seed()
        state = self.server.director.state(run)
        complete = _candidate_projection(state, state.candidate(candidate_id))
        hidden = {'metrics', 'inference_trace', 'algorithm_execution', 'model_plan', 'genome'}
        with ExitStack() as stack:
            for name in ('_algorithm_execution_projection', '_public_inference_trace',
                         '_public_evaluation_metrics', '_safe_plan_value'):
                stack.enter_context(patch('ecologyrsi_dsh.api.projection.' + name,
                    side_effect=AssertionError('summary attempted hidden detail construction')))
            status, payload = self.request('/api/runs/' + quote(run) + '?view=process')
        self.assertEqual(status, 200, payload)
        summary = payload['projection']['candidate_summaries'][0]
        self.assertEqual(summary, {k: v for k, v in complete.items() if k not in hidden})

    def test_collaboration_contains_parent_options_without_loading_candidate_detail(self):
        from ecologyrsi_dsh.core.models import Evaluation, Promotion
        run, candidate = self.seed()
        director = self.server.director
        director.record_evaluation(Evaluation(evaluation_id='evaluation:parent', run_id=run,
            candidate_id=candidate, score=.5, passed=True, metrics={}, partition='validation'))
        director.decide_promotion(Promotion(promotion_id='promotion:parent', run_id=run,
            candidate_id=candidate, decision='approved', reason='eligible test parent'))
        status, payload = self.request('/api/runs/' + quote(run) + '?view=collaboration')
        self.assertEqual(status, 200, payload)
        self.assertEqual([c['id'] for c in payload['projection']['intervention_candidates']], [candidate])
        self.assertNotIn('candidate_summaries', payload['projection'])
        self.assertNotIn('candidates', payload['projection'])

    def test_training_summary_defers_trace_but_detail_keeps_exact_identity(self):
        run, candidate = self.seed()
        with patch('ecologyrsi_dsh.presentation.training_assets.build_training_trajectory', side_effect=AssertionError('trace')):
            status, result = self.request('/api/runs/' + quote(run) + '?view=training')
        self.assertEqual(status, 200, result)
        summary = result['projection']['training_assets'][0]
        self.assertFalse(summary['details_loaded'])
        self.assertNotIn('episode', summary)
        status, result = self.request('/api/runs/' + quote(run) + '?view=asset&candidate_id=' + quote(candidate))
        self.assertEqual(status, 200, result)
        detail = result['projection']['training_asset']
        expected = training_assets(self.server.director.state(run), candidate_id=candidate)[0]
        self.assertEqual(detail, json.loads(json.dumps(expected)))
        self.assertEqual(detail['sample_id'], summary['sample_id'])
        self.assertIn('episode', detail)

    def test_workspace_query_rejects_ambiguous_or_foreign_detail(self):
        run, _ = self.seed()
        for query in ('view=process&view=training', 'view=unknown', 'view=overview&candidate_id=x'):
            status, _ = self.request('/api/runs/' + quote(run) + '?' + query)
            self.assertEqual(status, 400)
        status, _ = self.request('/api/runs/' + quote(run) + '?view=asset&candidate_id=foreign')
        self.assertEqual(status, 404)

    def test_event_tail_reuses_validated_stream_without_ledger_decode(self):
        run, _ = self.seed()
        self.server.director.state(run)
        original = self.server.ledger.events
        def guarded(run_id=None, **kwargs):
            self.assertNotEqual(kwargs.get('after_seq'), 0, 'tail decoded the full ledger again')
            return original(run_id, **kwargs)
        with patch.object(self.server.ledger, 'events', side_effect=guarded):
            status, result = self.request('/api/runs/' + quote(run) + '/events?tail=2')
        self.assertEqual(status, 200, result)
        self.assertEqual(len(result['events']), 2)
        self.assertTrue(result['truncated'])


class ReadSectionCacheTests(unittest.TestCase):
    def test_screening_sample_evidence_requires_the_exact_phase_cohort_and_candidate(self):
        from ecologyrsi_dsh.api.events import _screening_completion_matches
        candidate = SimpleNamespace(candidate_id='c', proposal_id='p', generation=0)
        start = SimpleNamespace(payload={'generation': 0, 'proposal_id': 'p',
            'checkpoint': {'evaluation_phase': 'screening', 'cohort_digest': 'cohort'}})
        completed = SimpleNamespace(payload={'evaluation_id': 'screening-evaluation:c:0', 'record_count': 9})
        evidence = SimpleNamespace(kind='CandidateScreeningRecorded', payload={
            'candidate_id': 'c', 'generation': 0, 'cohort_digest': 'cohort', 'prediction_cell_count': 9})
        state = SimpleNamespace(events=[evidence])
        self.assertTrue(_screening_completion_matches(state, candidate, start, completed))
        for key, value in [('candidate_id', 'other'), ('generation', 1), ('cohort_digest', 'other'), ('prediction_cell_count', 8)]:
            altered = deepcopy(evidence); altered.payload[key] = value
            self.assertFalse(_screening_completion_matches(SimpleNamespace(events=[altered]), candidate, start, completed))
        altered = deepcopy(start); altered.payload['checkpoint']['evaluation_phase'] = 'formal_batch'
        self.assertFalse(_screening_completion_matches(state, candidate, altered, completed))
        altered = deepcopy(completed); altered.payload['evaluation_id'] = 'unbound'
        self.assertFalse(_screening_completion_matches(state, candidate, start, altered))

    def test_duplicate_reads_share_one_build_without_sharing_mutable_results(self):
        cache = ReadSectionCache()
        started, release = threading.Event(), threading.Event()
        calls = []
        def build():
            calls.append(1); started.set(); release.wait(2)
            return {'rows': [1]}
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(cache.get_or_build, ('run', 1, 'training'), build)
            self.assertTrue(started.wait(1))
            second = pool.submit(cache.get_or_build, ('run', 1, 'training'), build)
            release.set()
            a, b = first.result(), second.result()
        a['rows'].append(2)
        self.assertEqual(b, {'rows': [1]})
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.get_or_build(('run', 2, 'training'), lambda: {'revision': 2}), {'revision': 2})

    def test_cache_enforces_byte_limit_and_does_not_cache_errors(self):
        cache = ReadSectionCache(max_bytes=30, max_entries=2)
        for i in range(4):
            cache.get_or_build(i, lambda: {'data': 'x' * 10})
        self.assertLessEqual(cache._bytes, 30)
        self.assertLessEqual(len(cache._values), 2)
        def failure(): raise ValueError('unavailable')
        with self.assertRaises(ValueError): cache.get_or_build('failed', failure)
        self.assertEqual(cache.get_or_build('failed', lambda: {'ok': True}), {'ok': True})
