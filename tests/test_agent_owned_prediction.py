"""Acceptance of autonomous inference, numerical ownership and evidence isolation."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from threading import Lock
import unittest

from ecologyrsi_dsh.core.agent_prediction import validate_predictions, validate_prediction_receipt
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.evaluators.dsh_sample_adapter import DshSampleCollaborationAdapter
from ecologyrsi_dsh.evaluators.sample_execution import CollaborativeSampleExecutor, SampleExecutionPolicy
from ecologyrsi_dsh.integrations.dsh_tools import DshToolService
from ecologyrsi_dsh.integrations.dsh_structured_roles import DshStructuredRoleRuntime
from ecologyrsi_dsh.integrations.prediction_binding import DshPredictionToolBinding
from tests.test_dsh_sample_execution import _request, _constant_forecast_bundle, _skill_evidence


def result(context, value=23.25, *, method='direct', refs=()):
    return {'schema_version': 'ecology-sample-predictions@2', 'wave_digest': context['wave_digest'],
            'decisions': [{'sample_id': s['sample_id'], 'predicted': value, 'confidence': .95,
                           'reason_code': 'agent_'+method, 'method': method, 'evidence_call_ids': list(refs)}
                          for s in context['samples']]}


class AdmittedAgent:
    """Transport double; lifecycle, calls, results and replay use production code."""
    def __init__(self, service, ledger, policy):
        self.service, self.ledger, self.policy = service, ledger, policy
        self.requests = []
        self._request_lock = Lock()

    def run_stage(self, request):
        with self._request_lock:
            self.requests.append(request)
            request_number = len(self.requests)
        stage = request['stage']
        context = request['request']['context']
        reservation = self.service.allocate_child_reservation({
            'request_id': 'agent-'+str(request_number), 'run_id': request['run_id'],
            'parent_session_id': 'test-host', 'role': request['request']['role'], 'stage': stage,
            'run_state_revision': request['run_state_revision'], 'stage_attempt': request['stage_attempt'],
            'admission_id': request['admission_id'], 'timeout_ms': 10000,
            'item_digest': request['request']['context_digest'], 'idempotency_key': request['idempotency_key'],
        })
        identity = {
            'run_id': request['run_id'], 'role': request['request']['role'], 'stage': stage,
            'run_state_revision': request['run_state_revision'], 'stage_attempt': request['stage_attempt'],
            'ledger_expected_revision': self.ledger.latest_seq(), 'session_id': 'agent-'+str(request_number),
            'idempotency_key': request['idempotency_key'], 'child_reservation_id': reservation['launch']['reservation_id'],
            'activation_lease_id': 'lease-'+str(request_number), **request['request']['identity_digests'],
        }
        def call(tool_id, call_id, parameters=None):
            return self.service.execute('ecology_execute_prediction_tool', {'identity': identity, 'arguments': {
                'tool_id': tool_id, 'call_id': call_id, 'parameters': parameters or {}, 'wave_digest': context['wave_digest'],
            }})
        if stage == 'sample.plan':
            structured = self.policy(context, call)
        elif stage == 'sample.critic':
            structured = {'schema_version': 'ecology-sample-review@2', 'wave_digest': context['wave_digest'],
                          'decisions': [{'sample_id': s['sample_id'], 'action': 'accept',
                                         'reason_code': 'accept_prediction', 'confidence': .9} for s in context['samples']]}
        else:
            raise AssertionError(stage)
        self.service.accept_structured({
            'identity': identity, 'output_schema_id': request['request']['output_schema_id'],
            'structured': structured, 'result_digest': digest(structured),
            'skill_invocation_evidence': _skill_evidence(stage), 'admission_id': request['admission_id'],
        })
        return {'structured': structured, 'result_digest': digest(structured)}


class AgentOwnedPredictionTests(unittest.TestCase):
    def test_origin_failure_drains_and_persists_completed_sibling(self):
        import threading
        import time
        from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
        second_started, first_failed = threading.Event(), threading.Event()
        failure = DshNativeRuntimeUnavailableError(
            error_code='structured_child_model_error', status_code=503)
        def policy(c, call):
            if c['samples'][0]['origin_timestamp'] == 10:
                self.assertTrue(second_started.wait(3))
                first_failed.set()
                raise failure
            second_started.set()
            self.assertTrue(first_failed.wait(3))
            time.sleep(.04)
            return result(c)
        adapter, plan, native, ledger = self.setup_agent(policy)
        rows = []
        for index, name in enumerate(('sample-fails', 'sample-succeeds')):
            row = _request(name).to_dict()
            origin = 10 + index * 10
            row.update(origin_timestamp=origin, target_timestamp=origin+1, observed=21.0)
            row['label_free_context']['causal_provenance'].update(
                origin_cutoff_timestamp=origin, latest_context_timestamp=origin,
                history_timestamps=[origin-1, origin])
            rows.append(row)
        publications = []
        with self.assertRaises(DshNativeRuntimeUnavailableError):
            CollaborativeSampleExecutor(adapter).execute(
                rows, context={'run_id': 'run-agent', 'candidate_id': 'candidate-1',
                    'dataset_digest': 'd'*64, 'partition': 'training_feedback',
                    'algorithm_id': 'registered-predictor', 'algorithm_version': '1',
                    'sample_concurrency': 2},
                target_bounds={'air_temperature': {'minimum': -20.0, 'maximum': 80.0}},
                algorithm_id='registered-predictor', algorithm_version='1',
                result_callback=publications.extend)
        self.assertEqual([row['origin_timestamp'] for row in publications], [20])
        self.assertEqual(publications[0]['sample_execution_status'], 'succeeded')

    def test_output_budget_failure_is_penalized_without_repeating_tools(self):
        from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
        def policy(c, call):
            call('candidate-model', 'fit')
            raise DshNativeRuntimeUnavailableError(
                error_code='structured_child_output_budget_exhausted', status_code=422)
        adapter, plan, native, ledger = self.setup_agent(policy)
        from ecologyrsi_dsh.evaluators.sample_execution import CollaborativeSampleExecutor
        row = _request('sample-budget').to_dict()
        row['observed'] = 21.0
        publications = []
        CollaborativeSampleExecutor(adapter).execute(
                (row,), context={'run_id': 'run-agent', 'candidate_id': 'candidate-1',
                    'dataset_digest': 'd'*64, 'partition': 'training_feedback',
                    'algorithm_id': 'registered-predictor', 'algorithm_version': '1'},
                target_bounds={'air_temperature': {'minimum': -20.0, 'maximum': 80.0}},
                algorithm_id='registered-predictor', algorithm_version='1',
                result_callback=publications.extend)
        self.assertEqual(len(publications), 1)
        self.assertEqual(publications[0]['sample_execution_status'], 'failed')
        self.assertEqual(publications[0]['scoring_fallback'], 'failure_non_improvement_penalty')
        self.assertEqual(len(native.requests), 1)
        self.assertEqual(len(ledger.events_by_kind('run-agent', 'DshPredictionToolExecuted')), 1)
        self.assertFalse(ledger.events_by_kind('run-agent', 'DshStructuredResultAccepted'))

    def setup_agent(self, policy, *, tool=None):
        ledger = EventLedger(); self.addCleanup(ledger.close)
        ledger.append('run-agent', 'RunCreated', {'test': True})
        service = DshToolService(ledger)
        native = AdmittedAgent(service, ledger, policy)
        adapter = DshSampleCollaborationAdapter(
            run_id='run-agent', runtime_provider=lambda: DshStructuredRoleRuntime(native, admission=service),
            revision_provider=lambda _: {'run_state_revision': ledger.latest_seq(), 'ledger_expected_revision': ledger.latest_seq()},
            identity_digests={'genome_digest': 'a'*64, 'compiled_behavior_digest': 'b'*64, 'phenotype_instance_digest': 'c'*64},
            strategy_model_id='dsh/strategy', review_model_id='dsh/review',
            forecast_bundle_tool=tool or _constant_forecast_bundle(21.5), prediction_tool_binder=service.bind_prediction_tool,
            remote_critic_policy={'version': 'uncertain_or_failure@1', 'min_planner_confidence': .8},
            sample_reflection_policy='candidate_aggregate_post_score@1',
        )
        plan = adapter.plan_batch({'run_id': 'run-agent', 'candidate_id': 'candidate-1', 'algorithm_id': 'registered-predictor', 'algorithm_version': '1'})
        return adapter, plan, native, ledger

    def predict(self, adapter, plan, request=None):
        return adapter.predict_samples((request or _request('sample-1'),), (plan,), attempts=(1,))[0]

    def test_direct_prediction_has_no_model_call_and_replays_without_agent(self):
        def forbidden(_): raise AssertionError('optional predictor must not run')
        adapter, plan, native, ledger = self.setup_agent(lambda c, call: result(c), tool=forbidden)
        self.assertEqual(plan['forecast_value_source'], 'agent_final_structured_prediction')
        self.assertIn('optionally_calls_tools', plan['routing_policy'])
        self.assertIn('persistence', [tool['tool_id'] for tool in plan['tools']])
        outcome = self.predict(adapter, plan)
        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.result['predicted'], 23.25)
        self.assertEqual([s['tool_id'] for s in outcome.result['tool_calls']], ['agent-final-prediction'])
        replay = self.predict(adapter, plan)
        self.assertEqual(replay.result, outcome.result)
        self.assertEqual(len(native.requests), 1)
        self.assertFalse(ledger.events_by_kind('run-agent', 'DshPredictionToolExecuted'))
        accepted = ledger.events_by_kind('run-agent', 'DshStructuredResultAccepted')[0]
        validate_prediction_receipt(accepted.payload['structured'], accepted.payload['required_tool_receipt'],
                                    event_lookup=lambda key: ledger.event_by_id(key), identity=accepted.payload['identity'], before_seq=accepted.seq)

    def test_two_model_results_are_evidence_and_agent_owns_final_value(self):
        def policy(c, call):
            first = call('candidate-model', 'ridge')
            second = call('persistence', 'baseline')
            self.assertEqual(first['outputs'][0]['predicted'], 21.5)
            self.assertEqual(second['outputs'][0]['predicted'], 20.0)
            return result(c, 20.75, method='blend', refs=('ridge', 'baseline'))
        adapter, plan, native, ledger = self.setup_agent(policy)
        outcome = self.predict(adapter, plan)
        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.result['predicted'], 20.75)
        self.assertEqual(len(ledger.events_by_kind('run-agent', 'DshPredictionToolExecuted')), 2)
        self.assertEqual(len(outcome.result['tool_calls']), 3)
        from ecologyrsi_dsh.evaluators.sample_execution import _attempt_trace_entry
        trace = _attempt_trace_entry(1, outcome.result['agent_decisions'],
                                     outcome.result['tool_calls'], outcome='accepted')
        self.assertEqual(trace['selected_tool']['tool_id'], 'agent-final-prediction')
        self.assertEqual(self.predict(adapter, plan).result, outcome.result)
        self.assertEqual(len(native.requests), 1)

    def test_tool_failure_can_be_followed_by_an_alternative_and_adjustment(self):
        def broken(_): raise ArithmeticError('failed fit')
        def policy(c, call):
            self.assertEqual(call('candidate-model', 'failed')['status'], 'failed')
            self.assertEqual(call('persistence', 'ok')['status'], 'completed')
            return result(c, 20.4, method='adjusted', refs=('ok',))
        adapter, plan, _, ledger = self.setup_agent(policy, tool=broken)
        outcome = self.predict(adapter, plan)
        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.result['predicted'], 20.4)
        self.assertEqual([s['status'] for s in outcome.result['tool_calls']], ['failed', 'completed', 'completed'])

    def test_host_rejection_returns_to_agent_instead_of_silent_clipping(self):
        def policy(c, call):
            attempt = c['samples'][0]['attempt']
            if attempt > 1:
                feedback = c['samples'][0]['failure_feedback']
                self.assertTrue(feedback)
                self.assertNotIn('observed', str(feedback))
            return result(c, 999 if attempt == 1 else 22.75)
        adapter, plan, native, _ = self.setup_agent(policy)
        request = _request('retry-sample')
        first = self.predict(adapter, plan, request)
        # Adapter preserves the Agent's number. Physical validation belongs to executor.
        self.assertEqual(first.result['predicted'], 999)
        retry_plan = {**plan, 'sample_retry_feedback': [{'attempt': 1, 'failure_class': 'constraint_rejected',
                       'retryable': True, 'previous_prediction': 999, 'tool_ids': ['agent-final-prediction']}]}
        second = adapter.predict_samples((request,), (retry_plan,), attempts=(2,))[0]
        self.assertIsNone(second.error)
        self.assertEqual(second.result['predicted'], 22.75)
        self.assertEqual(len(native.requests), 2)

    def test_context_contains_causal_numbers_and_never_the_scoring_label(self):
        adapter, plan, native, _ = self.setup_agent(lambda c, call: result(c))
        self.predict(adapter, plan)
        context = native.requests[0]['request']['context']
        ref = context['samples'][0]['context_ref']
        self.assertEqual(context['context']['origin_contexts'][ref]['history_window'], [19.0, 20.0])
        self.assertNotIn("'observed':", str(context))
        self.assertNotIn('proposed_prediction', str(context))

    def test_foreign_or_failed_tool_evidence_is_rejected(self):
        for refs in [('missing',), ('failed',)]:
            with self.subTest(refs=refs):
                def policy(c, call):
                    if refs == ('failed',): call('candidate-model', 'failed', {'bad': 1})
                    return result(c, method='model', refs=refs)
                adapter, plan, _, _ = self.setup_agent(policy)
                self.assertIsNotNone(self.predict(adapter, plan).error)

    def test_six_calls_budget_counts_failures_and_same_call_is_idempotent(self):
        calls = []
        def policy(c, call):
            for i in range(6):
                self.assertEqual(call('candidate-model', str(i))['remaining_calls'], 5-i)
            self.assertEqual(call('candidate-model', '0')['remaining_calls'], 0)
            with self.assertRaisesRegex(ValueError, 'budget'): call('candidate-model', '7')
            with self.assertRaisesRegex(ValueError, 'reused'): call('candidate-model', '0', {'x': 1})
            return result(c, method='model', refs=('0',))
        def tool(reqs):
            calls.append(True)
            return _constant_forecast_bundle(21.5)(reqs)
        adapter, plan, _, _ = self.setup_agent(policy, tool=tool)
        self.assertIsNone(self.predict(adapter, plan).error)
        self.assertEqual(len(calls), 6)

    def test_validation_rejects_nonfinite_duplicate_missing_and_bad_blend(self):
        context = {'wave_digest': 'a'*64, 'samples': [{'sample_id': 'one'}]}
        for value in (float('nan'), float('inf'), True):
            with self.assertRaises(ValueError): validate_predictions(result(context, value), ['one'], wave_digest='a'*64)
        for change in ({'decisions': []}, {'wave_digest': 'b'*64}):
            with self.assertRaises(ValueError): validate_predictions({**result(context), **change}, ['one'], wave_digest='a'*64)
        with self.assertRaises(ValueError): validate_predictions(result(context, method='blend', refs=('one',)), ['one'], wave_digest='a'*64)
