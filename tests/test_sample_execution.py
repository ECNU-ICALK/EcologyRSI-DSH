"""Native sample boundaries: labels, physical limits, failures and trace integrity."""
from __future__ import annotations

import base64
import json
import unittest
import zlib
from dataclasses import replace

from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.evaluators.sample_execution import (
    CollaborativeSampleExecutor, SampleExecutionPolicy, SampleExecutionContractError,
    SampleRepairRequired, _fallback_scoring_row, _host_critic_result,
    _validated_result, bounded_sample_execution_records, encode_sample_execution_trace,
)
from tests.test_dsh_sample_execution import _request
from tests.test_native_output_repair import execute, _SampleRuntime


class NativeSampleBoundaryTests(unittest.TestCase):
    def test_adapter_is_explicit_and_legacy_protocols_are_rejected_before_prediction(self):
        with self.assertRaisesRegex(ValueError, "explicit native"):
            CollaborativeSampleExecutor()
        class Legacy:
            def plan_batch(self, context):
                return {'sample_agent_protocol': 'sample_feedback@1'}
            def predict_sample(self, *args, **kwargs):
                raise AssertionError('legacy predictor must never execute')
        with self.assertRaisesRegex(ValueError, 'dsh-strict-origin-bundle@4'):
            CollaborativeSampleExecutor(Legacy()).execute([], context={}, target_bounds={},
                algorithm_id='registered', algorithm_version='1')

    def test_policy_rejects_bool_nonfinite_and_unknown_limits(self):
        for field, value in (('max_attempts', True), ('max_attempts', 0),
                             ('minimum_coverage', float('nan')), ('minimum_coverage', 1.1),
                             ('retry_backoff_seconds', -1)):
            with self.subTest(field=field, value=value), self.assertRaises((TypeError, ValueError)):
                SampleExecutionPolicy(**{field: value})
        with self.assertRaises(ValueError):
            SampleExecutionPolicy.from_mapping({'unlimited_retries': True})

    def test_request_rejects_nested_label_aliases_and_never_serializes_precomputed_prediction(self):
        request = _request('sample-safe')
        for key in ('observed', 'Label', 'ground_truth', 'actual-value', 'proposedPrediction'):
            with self.subTest(key=key), self.assertRaises((TypeError, ValueError)):
                replace(request, label_free_context={'nested': [{key: 123}]})
        for field in ('observed', 'predicted', 'proposed_prediction'):
            self.assertNotIn(field, request.to_dict())
        for value in (True, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                replace(request, baseline=value)

    def test_response_requires_finite_prediction_and_completed_evidence(self):
        valid = {'predicted': 22., 'agent_decisions': [{'role': 'planner', 'decision': 'predict', 'status': 'completed'}],
                 'tool_calls': [{'tool_id': 'agent-final-prediction', 'version': '1', 'status': 'completed'}]}
        self.assertEqual(_validated_result(valid)['predicted'], 22.)
        for change in ({'predicted': True}, {'predicted': float('nan')}, {'agent_decisions': []},
                       {'tool_calls': []}, {'private_reasoning': 'must not be captured'}):
            with self.subTest(change=change), self.assertRaises(SampleExecutionContractError):
                _validated_result({**valid, **change})

    def test_physical_critic_rejects_without_clipping_or_replacing_prediction(self):
        request = _request('physical')
        output = {'predicted': request.maximum + 1,
                  'agent_decisions': [{'role': 'planner', 'decision': 'predict', 'status': 'completed'}],
                  'tool_calls': [{'tool_id': 'agent-final-prediction', 'version': '1', 'status': 'completed'}]}
        with self.assertRaises(SampleRepairRequired) as caught:
            _host_critic_result(output, request)
        self.assertEqual(caught.exception.previous_prediction, output['predicted'])
        self.assertEqual(output['predicted'], request.maximum + 1)
        self.assertEqual(caught.exception.tool_calls[-1]['status'], 'rejected')

    def test_failed_prediction_cannot_gain_by_dropping_rows_or_beating_baseline(self):
        request = replace(_request('failed'), baseline=20., proposed_prediction=None)
        row = {'observed': 30., 'origin_timestamp': 10, 'target': request.target}
        failure = {'attempts': 2, 'retry_count': 1, 'failure': {'failure_class': 'contract'}}
        scored = _fallback_scoring_row(row, request, failure)
        self.assertEqual(scored['observed'], 30.)
        self.assertEqual(scored['predicted'], 20.)
        self.assertEqual(scored['sample_execution_status'], 'failed')
        self.assertEqual(scored['scoring_fallback'], 'failure_non_improvement_penalty')
        worse = _fallback_scoring_row(row, replace(request, proposed_prediction=10.), failure)
        self.assertEqual(worse['predicted'], 10.)

    def test_native_trace_roundtrip_retains_every_cell_and_digest(self):
        batch = execute(_SampleRuntime(), origins=(10, 20))
        archive = encode_sample_execution_trace(batch.records)
        raw = zlib.decompress(base64.b64decode(archive['payload']))
        decoded = json.loads(raw)
        self.assertEqual(len(decoded), 18)
        self.assertEqual(archive['record_count'], 18)
        self.assertEqual(len(raw), archive['uncompressed_bytes'])
        self.assertEqual(digest(decoded), archive['trace_digest'])
        self.assertEqual(decoded, list(batch.records))
        self.assertTrue(batch.summary['strict_agent_chain_pass'])

    def test_bounded_preview_preserves_late_failure_without_claiming_full_trace(self):
        records = [{'sample_id': str(i), 'status': 'succeeded'} for i in range(80)]
        records[-1]['status'] = 'failed'
        preview = bounded_sample_execution_records(records, limit=4)
        self.assertEqual(len(preview), 4)
        self.assertEqual(preview[-1], records[-1])
        self.assertEqual(len(records), 80)
        for limit in (0, True, -1):
            with self.assertRaises(ValueError):
                bounded_sample_execution_records(records, limit=limit)
