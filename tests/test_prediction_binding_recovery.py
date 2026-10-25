"""A replacement session explicitly reacquires already-paid frozen evidence."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from ecologyrsi_dsh.core.agent_prediction import (
    EVIDENCE_FORMULA_POLICY, PredictionContractViolation, validate_prediction_receipt,
)
from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
from ecologyrsi_dsh.integrations.dsh_structured_roles import DshStructuredRoleRuntime
from ecologyrsi_dsh.integrations.dsh_tools import DshToolService
from ecologyrsi_dsh.integrations.prediction_binding import DshPredictionToolBinding
from tests import test_agent_owned_prediction as helpers


class PredictionBindingRecoveryTests(unittest.TestCase):
    setup_agent = helpers.AgentOwnedPredictionTests.setup_agent
    predict = helpers.AgentOwnedPredictionTests.predict

    def test_full_durable_budget_recovers_in_new_session_and_receipt_replays(self):
        contexts = []

        def policy(context, call):
            contexts.append(deepcopy(context))
            contract = context['context']['prediction_contract']
            if len(contexts) == 1:
                self.assertEqual(contract['remaining_calls'], 4)
                self.assertEqual(contract['recoverable_call_requests'], [])
                for index in range(4):
                    call('candidate-model', f'original-{index}')
                raise DshNativeRuntimeUnavailableError(
                    error_code='structured_child_model_error', status_code=503)
            self.assertEqual(contract['remaining_calls'], 0)
            requests = contract['recoverable_call_requests']
            self.assertEqual(len(requests), 4)
            self.assertNotIn('predicted', str(requests))
            self.assertEqual(context['wave_digest'], contexts[0]['wave_digest'])
            with self.assertRaisesRegex(ValueError, 'budget'):
                call('candidate-model', 'invented-new-id')
            request = requests[0]
            response = call(request['tool_id'], request['call_id'], request['parameters'])
            self.assertEqual(response['remaining_calls'], 0)
            return helpers.result(context, response['outputs'][0]['predicted'],
                                  method='model', refs=[response['call_id']])

        adapter, plan, native, ledger = self.setup_agent(policy)
        plan['decision_context']['agent_policy'] = {'inference': {
            'prediction_formula_policy': EVIDENCE_FORMULA_POLICY}}
        with self.assertRaises(DshNativeRuntimeUnavailableError):
            self.predict(adapter, plan)
        before = ledger.events_by_kind('run-agent', 'DshPredictionToolExecuted')
        self.assertEqual(len(before), 4)
        service = DshToolService(ledger)
        native.service = service
        adapter._prediction_tool_binder = service.bind_prediction_tool
        adapter._decision_client.runtime_provider = lambda: DshStructuredRoleRuntime(native, admission=service)
        outcome = self.predict(adapter, plan)
        self.assertIsNone(outcome.error)
        self.assertEqual(len(ledger.events_by_kind('run-agent', 'DshPredictionToolExecuted')), 4)
        self.assertEqual(native.requests[0]['idempotency_key'], native.requests[1]['idempotency_key'])
        accepted = ledger.events_by_kind('run-agent', 'DshStructuredResultAccepted')[0]
        receipt = accepted.payload['required_tool_receipt']
        self.assertEqual(receipt['schema_version'], 'ecologyrsi-dsh.agent-prediction-receipt/3')
        self.assertEqual([call['call_id'] for call in receipt['calls']], ['original-0'])
        validate_prediction_receipt(accepted.payload['structured'], receipt,
            event_lookup=ledger.event_by_id, identity=accepted.payload['identity'], before_seq=accepted.seq)
        self.assertEqual(self.predict(adapter, plan).result, outcome.result)
        self.assertEqual(len(native.requests), 2)

    def test_listing_recovery_requests_does_not_grant_visibility_or_mutate_calls(self):
        events = []
        def persist(payload):
            event = SimpleNamespace(event_id=f'tool:{len(events)}', payload=payload)
            events.append(event)
            return event
        def binding(wave='a'*64):
            return DshPredictionToolBinding(run_id='r', stage_attempt=1, idempotency_key='k',
                wave_digest=wave, sample_ids=['s'], budget=1,
                catalog=[{'tool_id': 'model', 'version': '1', 'parameters': {}}],
                executor=lambda *_: {'s': {'predicted': 12.5}}, formula_policy=EVIDENCE_FORMULA_POLICY)
        original = binding()
        arguments = {'tool_id': 'model', 'call_id': 'original', 'parameters': {}, 'wave_digest': 'a'*64}
        original.execute(arguments, session_id='old', persist=persist)
        restored = binding()
        restored.restore(events[0])
        listed = restored.recovery_context()
        structured = helpers.result({'wave_digest': 'a'*64, 'samples': [{'sample_id': 's'}]},
                                    12.5, method='model', refs=['original'])
        with self.assertRaises(PredictionContractViolation):
            restored.final_receipt(structured, session_id='new')
        listed['recoverable_call_requests'][0]['parameters']['unexpected'] = 1
        self.assertEqual(restored.recovery_context()['recoverable_call_requests'], [arguments])
        with self.assertRaisesRegex(ValueError, 'reused'):
            restored.execute({**arguments, 'parameters': {'changed': 1}}, session_id='new', persist=persist)
        with self.assertRaisesRegex(ValueError, 'outside'):
            restored.execute({**arguments, 'wave_digest': 'b'*64}, session_id='new', persist=persist)
        with self.assertRaisesRegex(ValueError, 'another wave'):
            binding('b'*64).restore(events[0])
        restored.execute(arguments, session_id='new', persist=persist)
        self.assertEqual(len(events), 1)
        self.assertEqual(restored.final_receipt(structured, session_id='new')['calls'][0]['event_id'], 'tool:0')
        with self.assertRaises(PredictionContractViolation):
            restored.final_receipt(structured, session_id='unrelated-session')

    def test_changed_candidate_identity_cannot_recover_previous_wave(self):
        contexts = []
        def policy(context, call):
            contexts.append(context)
            contract = context['context']['prediction_contract']
            self.assertEqual(contract['remaining_calls'], 4)
            self.assertEqual(contract['recoverable_call_requests'], [])
            response = call('candidate-model', 'fit')
            if len(contexts) == 1:
                raise DshNativeRuntimeUnavailableError(
                    error_code='structured_child_model_error', status_code=503)
            return helpers.result(context, response['outputs'][0]['predicted'], method='model', refs=['fit'])
        adapter, plan, native, ledger = self.setup_agent(policy)
        with self.assertRaises(DshNativeRuntimeUnavailableError):
            self.predict(adapter, plan)
        adapter._decision_client.identity_digests['genome_digest'] = 'd'*64
        self.assertIsNone(self.predict(adapter, plan).error)
        self.assertNotEqual(contexts[0]['wave_digest'], contexts[1]['wave_digest'])
        self.assertEqual(len(ledger.events_by_kind('run-agent', 'DshPredictionToolExecuted')), 2)


if __name__ == '__main__':
    unittest.main()
