"""Ephemeral capability catalog for one autonomous sample Agent wave."""
from collections.abc import Mapping
from copy import deepcopy
from threading import RLock
import math
from time import monotonic

from ..core.agent_prediction import PREDICTION_TOOL_CALL_BUDGET, validate_predictions
from ..core.models import digest


class DshPredictionToolBinding:
    def __init__(self, *, run_id, stage_attempt, idempotency_key, wave_digest, sample_ids, catalog, executor):
        self.run_id = run_id
        self.stage_attempt = stage_attempt
        self.idempotency_key = idempotency_key
        self.wave_digest = wave_digest
        self.sample_ids = tuple(sample_ids)
        self.catalog = {item['tool_id']: dict(item) for item in catalog}
        self.executor = executor
        self.calls = {}
        self._predictions = {}
        self._session_calls = {}
        self._lock = RLock()

    def restore(self, event):
        from ..core.agent_prediction import validate_tool_event
        p = event.payload
        validate_tool_event(p)
        if p['wave_digest'] != self.wave_digest or p['sample_ids'] != list(self.sample_ids):
            raise ValueError('recorded tool call belongs to another wave')
        self.calls[p['call_id']] = (deepcopy(p), event.event_id)
        if p['result']['status'] == 'completed':
            self._predictions[self._computation_key(p['arguments'])] = (
                deepcopy(p['result']['outputs']), event.event_id
            )

    def _computation_key(self, arguments):
        tool = self.catalog[arguments['tool_id']]
        parameters = {
            name: spec['default'] for name, spec in tool.get('parameters', {}).items()
            if isinstance(spec, Mapping) and 'default' in spec
        }
        parameters.update(arguments['parameters'])
        # This cache belongs to exactly one immutable wave/executor. No reuse
        # across origins, revisions, fitting data or independent Agent replicas.
        return digest({'wave_digest': self.wave_digest, 'tool_id': arguments['tool_id'],
                       'version': tool['version'], 'parameters': parameters})

    def execute(self, arguments, *, session_id, persist):
        if set(arguments) != {'tool_id', 'wave_digest', 'call_id', 'parameters'}:
            raise ValueError('tool requires tool_id, wave_digest, call_id and parameters')
        if arguments['wave_digest'] != self.wave_digest or arguments['tool_id'] not in self.catalog:
            raise ValueError('tool is outside this Agent wave capability catalog')
        call_id = arguments['call_id']
        if not isinstance(call_id, str) or not 1 <= len(call_id) <= 80 or not isinstance(arguments['parameters'], Mapping):
            raise ValueError('prediction tool call identity/parameters are invalid')
        with self._lock:
            if call_id in self.calls:
                p, event_id = self.calls[call_id]
                if p['arguments'] != arguments:
                    raise ValueError('prediction call_id was reused with different arguments')
            else:
                if len(self.calls) >= PREDICTION_TOOL_CALL_BUDGET:
                    raise ValueError('prediction tool call budget exhausted')
                result = {'tool_id': arguments['tool_id'], 'call_id': call_id, 'wave_digest': self.wave_digest}
                started = monotonic()
                computation_key = self._computation_key(arguments)
                cached = self._predictions.get(computation_key)
                try:
                    raw = (
                        {row['sample_id']: row for row in cached[0]} if cached is not None
                        else self.executor(arguments['tool_id'], dict(arguments['parameters']))
                    )
                    if not isinstance(raw, Mapping) or set(raw) != set(self.sample_ids):
                        raise ValueError('tool must return every sample exactly once')
                    outputs = []
                    for sample_id in self.sample_ids:
                        item = raw[sample_id]
                        value = item['predicted']
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                            raise ValueError('tool prediction must be finite')
                        outputs.append({'sample_id': sample_id, 'predicted': float(value), 'metadata': dict(item.get('metadata', {}))})
                    result.update(status='completed', outputs=outputs)
                    if cached is not None:
                        result['reused_from_event_id'] = cached[1]
                except Exception as exc:
                    from ..evaluators.sample_execution import SampleExecutionControlError
                    if isinstance(exc, SampleExecutionControlError):
                        raise
                    # A failed capability is evidence for the Agent to choose another path.
                    result.update(status='failed', error_code=type(exc).__name__)
                result['elapsed_ms'] = round((monotonic() - started) * 1000, 3)
                p = {
                    'schema_version': 'ecologyrsi-dsh.dsh-prediction-tool-executed/2',
                    'stage': 'sample.plan', 'stage_attempt': self.stage_attempt,
                    'idempotency_key': self.idempotency_key, 'tool_id': arguments['tool_id'],
                    'wave_digest': self.wave_digest, 'sample_ids': list(self.sample_ids),
                    'prediction_count': len(self.sample_ids), 'call_id': call_id,
                    'arguments': deepcopy(dict(arguments)), 'request_digest': digest(arguments),
                    'result': result, 'output_digest': digest(result), 'execution_owner': 'dsh_agent_tool_call',
                }
                event = persist(p)
                event_id = event.event_id
                self.calls[call_id] = (p, event_id)
                if result['status'] == 'completed':
                    self._predictions.setdefault(computation_key, (deepcopy(result['outputs']), event_id))
            self._session_calls.setdefault(session_id, set()).add(call_id)
            return {'accepted': True, 'event_id': event_id, 'output_digest': p['output_digest'],
                    'remaining_calls': max(0, PREDICTION_TOOL_CALL_BUDGET - len(self.calls)), **deepcopy(p['result'])}

    def final_receipt(self, structured, *, session_id):
        rows = validate_predictions(structured, self.sample_ids, wave_digest=self.wave_digest)
        with self._lock:
            visible = self._session_calls.get(session_id, set())
            successful = {key for key in visible if self.calls[key][0]['result']['status'] == 'completed'}
            if any(set(row['evidence_call_ids']) - successful for row in rows):
                raise ValueError('Agent cites tool evidence it did not receive or that failed')
            return {
                'schema_version': 'ecologyrsi-dsh.agent-prediction-receipt/2',
                'wave_digest': self.wave_digest, 'sample_ids': list(self.sample_ids),
                'result_digest': digest(structured),
                'calls': [{'call_id': key, 'event_id': event_id, 'output_digest': p['output_digest']}
                          for key, (p, event_id) in self.calls.items() if key in visible],
            }

    def public_trace(self, sample_id=None, evidence_call_ids=()):
        """Bounded per-cell evidence, distinct from every call made in the wave."""
        with self._lock:
            trace = []
            for p, event_id in self.calls.values():
                item = {
                    'tool_id': p['tool_id'], 'version': str(self.catalog[p['tool_id']]['version']),
                    'status': p['result']['status'], 'input_digest': p['request_digest'],
                    'output_digest': p['output_digest'], 'execution_owner': 'dsh_agent_tool_call',
                    'dsh_tool_event_id': event_id, 'dsh_tool_output_digest': p['output_digest'],
                    'elapsed_ms': p['result'].get('elapsed_ms', 0),
                    'call_id': p['call_id'], 'parameters': deepcopy(p['arguments']['parameters']),
                    'used_as_evidence': p['call_id'] in evidence_call_ids,
                }
                output = next((row for row in p['result'].get('outputs', ()) if row['sample_id'] == sample_id), None)
                if output is not None:
                    item['tool_predicted'] = output['predicted']
                    item['parameters'] = deepcopy(output.get('metadata', {}).get('parameters', item['parameters']))
                trace.append(item)
            return trace
