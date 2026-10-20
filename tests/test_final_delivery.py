"""Regression checks for release-critical scientific and accounting boundaries."""
from copy import deepcopy
from types import SimpleNamespace
import os
import unittest

from ecologyrsi_dsh.core.dsh_usage import validate_session_usage, check_usage_binding, session_usage_projection
from tests.dsh_usage_fixture import usage


def event(kind, payload, seq=1):
    return SimpleNamespace(kind=kind, payload=payload, seq=seq)


class UsageTests(unittest.TestCase):
    def test_activity_is_bounded_metadata_and_does_not_change_billing(self):
        body = usage(10)
        body['session_metrics']['activity'] = {
            'kind': 'streaming', 'log_revision': 20, 'updated_at': '2026-09-09T00:00:00Z',
        }
        validate_session_usage(body, run_id='r')
        self.assertEqual(body['session_metrics']['provider_usage']['totals']['total_tokens'], 10)
        for key, value in [('kind', 'arbitrary'), ('log_revision', True), ('updated_at', 'yesterday')]:
            bad = deepcopy(body); bad['session_metrics']['activity'][key] = value
            with self.assertRaises(ValueError):
                validate_session_usage(bad, run_id='r')
        body['session_metrics']['activity']['text'] = 'not allowed'
        with self.assertRaises(ValueError):
            validate_session_usage(body, run_id='r')

    def test_failed_calls_count_without_scientific_acceptance_and_snapshots_do_not_sum(self):
        launch = event('DshChildLaunchReserved', {'launch': dict(run_id='r', stage='s', idempotency_key='k', reservation_id='c')})
        active = usage()
        failed = usage(20, 'failed', True)
        validate_session_usage(failed, run_id='r')
        check_usage_binding(failed, [launch, event('DshSessionUsageRecorded', active)])
        events = [launch, event('DshSessionUsageRecorded', active, 2), event('DshSessionUsageRecorded', failed, 3),
                  event('DshStructuredResultAccepted', active, 4)]
        metrics, coverage = session_usage_projection(events)
        self.assertEqual(metrics['session'][1]['provider_usage']['totals']['total_tokens'], 20)
        self.assertTrue(coverage['complete'])
        self.assertFalse(coverage['includes_retrieval_provider_usage'])
        with self.assertRaises(ValueError):
            check_usage_binding(active, events)
        forged = deepcopy(failed); forged['identity']['session_id'] = 'other'
        with self.assertRaises(ValueError):
            check_usage_binding(forged, events)
        with self.assertRaises(ValueError):
            check_usage_binding(failed, [])

    def test_interrupted_stream_is_not_complete_even_with_legacy_flag(self):
        launch = event('DshChildLaunchReserved', {'launch': {'reservation_id': 'c'}})
        _, coverage = session_usage_projection([launch, event('DshSessionUsageRecorded', usage(20, 'cancelled', True))])
        self.assertEqual(coverage['complete_session_count'], 0)
        self.assertFalse(coverage['complete'])

    def test_unobserved_and_active_usage_is_incomplete(self):
        launch = event('DshChildLaunchReserved', {'launch': {'reservation_id': 'c'}})
        _, coverage = session_usage_projection([launch])
        self.assertFalse(coverage['complete'])
        self.assertEqual(coverage['unobserved_session_count'], 1)
        with self.assertRaises(ValueError):
            validate_session_usage(usage(10, 'active', True), run_id='r')
        for malformed in ({}, {'identity': []}, usage(-1)):
            with self.assertRaises((ValueError, TypeError)):
                validate_session_usage(malformed, run_id='r')


class UsageServiceTests(unittest.TestCase):
    def test_durable_usage_survives_admission_close_and_service_restart(self):
        from ecologyrsi_dsh.core.ledger import EventLedger
        from ecologyrsi_dsh.integrations.dsh_tools import DshToolService
        from tests.test_dsh_tool_contracts import _reservation_request
        with EventLedger() as ledger:
            ledger.append('run:tool-test', 'RunCreated', {'test': True})
            service = DshToolService(ledger)
            fence = service.open_admission('run:tool-test', 3, 2, role='researcher',
                       stage='generation.search-plan', idempotency_key='deadline-result')
            reservation = service.allocate_child_reservation({**_reservation_request(fence.admission_id), "stage": "generation.search-plan"})
            body = usage(10, 'cancelled', True)
            body['identity'].update(run_id='run:tool-test', stage='generation.search-plan',
                idempotency_key='deadline-result', child_reservation_id=reservation['launch']['reservation_id'])
            service.close_run_admissions('run:tool-test')
            self.assertTrue(service.record_session_usage(body)['accepted'])
            restarted = DshToolService(ledger)
            self.assertTrue(restarted.record_session_usage(body)['already_recorded'])
            self.assertEqual(len(ledger.events_by_kind('run:tool-test', 'DshSessionUsageRecorded')), 1)
            for identity in ([], None, 'bad'):
                bad = deepcopy(body); bad['identity'] = identity
                with self.assertRaises(ValueError): restarted.record_session_usage(bad)


class DatasetProtocolTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("ECOLOGYRSI_TEST_REAL_DATA") == "1", "requires prepared AGC data")
    def test_frozen_fit_page_uses_fit_boundary_and_rejects_protocol_drift(self):
        from ecologyrsi_dsh.data.registry import DatasetRegistry
        registry = DatasetRegistry()
        view = registry.selection_view('agc_cucumber_2018', 'agc_cucumber_2018:AiCU')
        page = registry.sample(view.dataset_id, episode_id=view.episode_id, limit=100,
                    expected_data_protocol_digest=view.data_protocol_digest)
        self.assertEqual(page['total'], view.partitions['calibration_fit'].size)
        self.assertTrue(all(row['index'] < view.partitions['calibration_fit'].end for row in page['rows']))
        with self.assertRaises(ValueError):
            registry.sample(view.dataset_id, expected_data_protocol_digest='0' * 64)
        with self.assertRaises(PermissionError):
            registry.sample(view.dataset_id, partition='final', expected_data_protocol_digest=view.data_protocol_digest)


class UsageReplayTests(unittest.TestCase):
    def test_full_replay_never_compares_an_early_snapshot_to_future_usage(self):
        from ecologyrsi_dsh import EventLedger, EvolutionDirector
        from tests.native_fixtures import NativeRoleStub, start_native_run
        from ecologyrsi_dsh.core.state import project_run_state
        from ecologyrsi_dsh.integrations.dsh_tools import DshToolService
        from tests.test_dsh_tool_contracts import _reservation_request
        from tests.test_research_iteration import _task
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            start_native_run(director, task=_task(), run_id='run:tool-test')
            service = DshToolService(ledger)
            fence = service.open_admission('run:tool-test', 3, 2, role='researcher',
                       stage='generation.search-plan', idempotency_key='deadline-result')
            reservation = service.allocate_child_reservation({**_reservation_request(fence.admission_id), "stage": "generation.search-plan"})
            for amount, settlement, complete in [(10, 'active', False), (20, 'succeeded', True)]:
                body = usage(amount, settlement, complete)
                body['identity'].update(run_id='run:tool-test', stage='generation.search-plan',
                    idempotency_key='deadline-result', child_reservation_id=reservation['launch']['reservation_id'])
                service.record_session_usage(body)
                # Exercise batched replay, not just the append-time validator.
                state = project_run_state(ledger.events('run:tool-test'))
                self.assertEqual(state.events[-1].payload['settlement'], settlement)
            metrics, coverage = session_usage_projection(state.events)
            self.assertTrue(coverage['complete'])
            self.assertEqual(metrics['session'][1]['provider_usage']['totals']['total_tokens'], 20)
