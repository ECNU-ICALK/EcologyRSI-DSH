"""Deterministic Agent policy doubles using the production tool binding."""
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from ecologyrsi_dsh.integrations.prediction_binding import DshPredictionToolBinding
from ecologyrsi_dsh.core.models import digest

active_binding = ContextVar('test_agent_prediction_binding')

@contextmanager
def agent_binding(**arguments):
    binding = DshPredictionToolBinding(**arguments)
    token = active_binding.set(binding)
    try:
        yield binding
    finally:
        active_binding.reset(token)


def model_result(context):
    binding = active_binding.get()
    return binding.execute(
        {'tool_id': 'candidate-model', 'wave_digest': context['wave_digest'], 'call_id': 'model-1', 'parameters': {}},
        session_id='test-agent',
        persist=lambda payload: SimpleNamespace(event_id='test-tool:'+digest(payload)),
    )


def prediction_rows(context, result):
    if result['status'] != 'completed':
        raise RuntimeError('test Agent terminates after tool failure')
    return [dict(sample_id=row['sample_id'], predicted=row['predicted'], confidence=0.9,
                 reason_code='agent_model', method='model', evidence_call_ids=[result['call_id']])
            for row in result['outputs']]
