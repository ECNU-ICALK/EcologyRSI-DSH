"""Host-proven model identities remove alias votes without rewriting old evidence."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from ecologyrsi_dsh.core.agent_prediction import (
    EVIDENCE_FORMULA_POLICY, PredictionContractViolation,
    unique_prediction_evidence, validate_tool_event,
)
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.integrations.prediction_binding import DshPredictionToolBinding


class CanonicalComputationTests(unittest.TestCase):
    proof = {'schema_version': 'ecologyrsi-dsh.canonical-computation/1',
             'model_id': 'registered-model@1', 'parameters_digest': 'b' * 64,
             'fit_resource_key': 'c' * 64}

    def binding(self, *, tagged=True):
        events = []
        def execute(tool, _parameters):
            metadata = {'canonical_computation': {
                **self.proof, 'model_id': ('other-model@1' if tool == 'other' else 'registered-model@1')
            }} if tagged else {}
            return {sid: {'predicted': value, 'metadata': metadata}
                    for sid, value in (('a', 10.), ('b', 40.))}
        binding = DshPredictionToolBinding(run_id='r', stage_attempt=1, idempotency_key='k',
            wave_digest='a' * 64, sample_ids=['a', 'b'], executor=execute, budget=8,
            catalog=[{'tool_id': tool, 'version': '1', 'parameters': {}}
                     for tool in ('candidate-model', 'registered-model@1', 'other')],
            formula_policy=EVIDENCE_FORMULA_POLICY)
        def persist(payload):
            event = SimpleNamespace(event_id=f'event:{len(events)}', payload=payload)
            events.append(event)
            return event
        def call(tool, call_id):
            return binding.execute({'tool_id': tool, 'call_id': call_id,
                'wave_digest': 'a' * 64, 'parameters': {}}, session_id='s', persist=persist)
        return binding, events, call

    def test_alias_gets_one_vote_but_equal_distinct_models_keep_their_vote(self):
        binding, events, call = self.binding()
        call('candidate-model', 'candidate')
        duplicate = call('registered-model@1', 'alias')
        self.assertNotIn('mean_formula', duplicate)
        response = call('other', 'other')
        self.assertEqual(response['mean_formula']['evidence_call_ids'], ['candidate', 'other'])
        for event in events:
            validate_tool_event(event.payload)
        row = {'sample_id': 'a', 'method': 'blend', 'predicted': 10.,
               'evidence_call_ids': ['candidate', 'alias']}
        with self.assertRaises(PredictionContractViolation) as error:
            binding.validate_formulas([row])
        self.assertEqual(error.exception.diagnostics[0]['code'], 'duplicate_computation_evidence')
        binding.validate_formulas([{**row, 'evidence_call_ids': ['candidate', 'other']}])

    def test_distinct_fit_parameters_model_wave_and_samples_are_not_deduplicated(self):
        binding, _events, call = self.binding()
        call('candidate-model', 'first')
        first, _ = binding.calls['first']
        for changed in ('fit_resource_key', 'parameters_digest', 'model_id', 'wave_digest', 'sample_ids'):
            with self.subTest(changed=changed):
                second = deepcopy(first)
                if changed in ('wave_digest', 'sample_ids'):
                    second[changed] = 'd' * 64 if changed == 'wave_digest' else ['different']
                else:
                    for row in second['result']['outputs']:
                        row['metadata']['canonical_computation'][changed] = (
                            'different-model@1' if changed == 'model_id' else 'd' * 64)
                calls = {'first': (first, 'event:1'), 'second': (second, 'event:2')}
                self.assertEqual(unique_prediction_evidence(calls), ['first', 'second'])

    def test_old_aliases_and_mixed_restore_keep_historical_evidence_valid(self):
        legacy, events, call = self.binding(tagged=False)
        call('candidate-model', 'candidate')
        response = call('registered-model@1', 'alias')
        self.assertEqual(response['mean_formula']['evidence_call_ids'], ['candidate', 'alias'])
        restored, _new_events, _ = self.binding()
        for event in events:
            restored.restore(event)
        self.assertEqual(unique_prediction_evidence(restored.calls), ['candidate', 'alias'])
        # Adding a new proven result cannot retroactively infer old alias identity.
        new, _, call_new = self.binding()
        call_new('registered-model@1', 'new')
        mixed = {**restored.calls, **new.calls}
        mixed['new'] = (mixed['new'][0], 'event:new')
        self.assertEqual(unique_prediction_evidence(mixed), ['candidate', 'alias', 'new'])
        restored.validate_formulas([{'sample_id': 'a', 'method': 'blend', 'predicted': 10.,
                                    'evidence_call_ids': ['candidate', 'alias']}])

    def test_partial_or_malformed_host_proof_is_rejected_but_old_metadata_is_optional(self):
        binding, _, call = self.binding()
        call('candidate-model', 'first')
        original = binding.calls['first'][0]
        for change in ('missing', 'wrong_digest', 'different_proof'):
            with self.subTest(change=change):
                payload = deepcopy(original)
                payload['result']['outputs'][0]['metadata'] = deepcopy(payload['result']['outputs'][0]['metadata'])
                proof = payload['result']['outputs'][0]['metadata']['canonical_computation']
                if change == 'missing':
                    del payload['result']['outputs'][0]['metadata']['canonical_computation']
                elif change == 'wrong_digest':
                    proof['parameters_digest'] = 'invalid'
                else:
                    proof['parameters_digest'] = 'd' * 64
                payload['output_digest'] = digest(payload['result'])
                with self.assertRaisesRegex(ValueError, 'canonical computation proof'):
                    validate_tool_event(payload)
        for metadata in (None, {}, 'historical-metadata'):
            payload = deepcopy(original)
            for row in payload['result']['outputs']:
                row['metadata'] = metadata
            payload['output_digest'] = digest(payload['result'])
            validate_tool_event(payload)


if __name__ == '__main__':
    unittest.main()
