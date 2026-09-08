"""Regression checks for release-critical scientific and accounting boundaries."""
from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace
import os
import unittest

from ecologyrsi_dsh.core.dsh_usage import validate_session_usage, check_usage_binding, session_usage_projection
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evaluators.epoch_cohorts import (
    plan_run_adaptation_cohort, plan_generation_selection_cohorts,
    estimate_epoch_capacity, RunAdaptationCohort, GenerationCohorts, CohortCapacityError,
)
from tests.test_epoch_cohort_planning import dataset_fixture
from ecologyrsi_dsh.evolution.batches import _research_contract_fallback_enabled


class IsolatedCohortsTests(unittest.TestCase):
    def test_time_purge_cross_day_and_round_trip(self):
        schedule = replace(OptimizationSchedule.for_new_run(),
                           formal_origin_count_per_finalist=20, local_batch_origin_count=10)
        data = dataset_fixture(1200, timestamp_gap_at=400)
        adaptation = plan_run_adaptation_cohort(data, schedule=schedule, seed=7)
        groups = [b.cohort for b in adaptation.batches]
        for generation in range(2):
            cohorts = plan_generation_selection_cohorts(data, schedule=schedule,
                        generation=generation, adaptation=adaptation, seed=7)
            self.assertEqual(GenerationCohorts.from_dict(cohorts.to_dict()), cohorts)
            groups.extend([cohorts.screening, cohorts.holdout])
        for left, right in zip(groups, groups[1:]):
            self.assertLess(left.origins[-1].maximum_target_timestamp, right.origins[0].origin_timestamp)
            a = {o.origin_timestamp + h for o in left.origins for h in (1, 6, 24)}
            b = {o.origin_timestamp + h for o in right.origins for h in (1, 6, 24)}
            self.assertFalse(a & b)
        for batch in adaptation.batches:
            self.assertGreaterEqual(batch.cohort.origins[-1].origin_timestamp - batch.cohort.origins[0].origin_timestamp, 24)
        self.assertEqual(RunAdaptationCohort.from_dict(adaptation.to_dict()), adaptation)
        changed = dataset_fixture(1200, changed_labels=True, timestamp_gap_at=400)
        self.assertEqual(plan_run_adaptation_cohort(changed, schedule=schedule, seed=7), adaptation)

    def test_capacity_refuses_reuse_and_default_fits_realistic_partition(self):
        data = dataset_fixture(803)
        schedule = OptimizationSchedule.for_new_run()
        report = estimate_epoch_capacity(data, schedule=schedule, planned_generations=1, seed=7)
        self.assertTrue(report.sufficient)
        self.assertEqual(report.max_feasible_generations, 1)
        self.assertEqual(report.reused_origin_occurrences, 0)
        report = estimate_epoch_capacity(data, schedule=schedule, planned_generations=2, seed=7)
        self.assertFalse(report.sufficient)
        adaptation = plan_run_adaptation_cohort(data, schedule=schedule, seed=7)
        with self.assertRaises(CohortCapacityError):
            plan_generation_selection_cohorts(data, schedule=schedule, generation=1, adaptation=adaptation, seed=7)

    def test_legacy_identity_is_stable(self):
        schedule = OptimizationSchedule.default()
        cohort = plan_run_adaptation_cohort(dataset_fixture(800), schedule=schedule, seed=7)
        self.assertTrue(cohort.planner_schema.endswith('/1'))
        self.assertEqual(RunAdaptationCohort.from_dict(cohort.to_dict()), cohort)

    def test_native_never_uses_incomplete_host_fallback(self):
        for flag in (False, True):
            state = SimpleNamespace(task_manifest=SimpleNamespace(metadata={
                'execution_protocol': 'dsh_native_plugin_evolution@1',
                'allow_host_fallback': flag,
                'host_runtime_build': {'evolution_runtime_schema': 'ecologyrsi-dsh.evolution-runtime/3'},
            }))
            self.assertFalse(_research_contract_fallback_enabled(state))


def usage(total=10, settlement='active', complete=False):
    return {'schema_version': 'ecologyrsi-dsh.session-usage/1',
        'identity': dict(run_id='r', stage='s', idempotency_key='k', child_reservation_id='c', session_id='session'),
        'session_metrics': {'schema_version': 'ecologyrsi-dsh.dsh-session-metrics/1', 'session_id': 'session',
            'context_pressure': {'available': False, 'source': 'dsh_token_meter'},
            'provider_usage': {'available': True, 'source': 'dsh_session_projection_token_usage',
                'measurement': 'cumulative_provider_reported_usage',
                'totals': dict(uncached_input_tokens=total, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0, total_tokens=total)}},
        'settlement': settlement, 'usage_complete': complete}


def event(kind, payload, seq=1):
    return SimpleNamespace(kind=kind, payload=payload, seq=seq)


class UsageTests(unittest.TestCase):
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
                       stage='generation.research', idempotency_key='deadline-result')
            reservation = service.allocate_child_reservation(_reservation_request(fence.admission_id))
            body = usage(10, 'cancelled', True)
            body['identity'].update(run_id='run:tool-test', stage='generation.research',
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
        from ecologyrsi_dsh import EventLedger, EvolutionDirector, FakeDSHAdapter
        from ecologyrsi_dsh.core.state import project_run_state
        from ecologyrsi_dsh.integrations.dsh_tools import DshToolService
        from tests.test_dsh_tool_contracts import _reservation_request
        from tests.test_research_iteration import _task
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, FakeDSHAdapter())
            director.start_evolution(_task(candidates_per_generation=1), run_id='run:tool-test')
            service = DshToolService(ledger)
            fence = service.open_admission('run:tool-test', 3, 2, role='researcher',
                       stage='generation.research', idempotency_key='deadline-result')
            reservation = service.allocate_child_reservation(_reservation_request(fence.admission_id))
            for amount, settlement, complete in [(10, 'active', False), (20, 'succeeded', True)]:
                body = usage(amount, settlement, complete)
                body['identity'].update(run_id='run:tool-test', stage='generation.research',
                    idempotency_key='deadline-result', child_reservation_id=reservation['launch']['reservation_id'])
                service.record_session_usage(body)
                # Exercise batched replay, not just the append-time validator.
                state = project_run_state(ledger.events('run:tool-test'))
                self.assertEqual(state.events[-1].payload['settlement'], settlement)
            metrics, coverage = session_usage_projection(state.events)
            self.assertTrue(coverage['complete'])
            self.assertEqual(metrics['session'][1]['provider_usage']['totals']['total_tokens'], 20)
