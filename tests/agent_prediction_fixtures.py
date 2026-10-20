"""Deterministic Agent policy doubles using the production tool binding."""
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from ecologyrsi_dsh.integrations.prediction_binding import DshPredictionToolBinding
from ecologyrsi_dsh.core.models import digest

active_binding = ContextVar('test_agent_prediction_binding')

@contextmanager
def agent_binding(*, prediction_tool_call_budget=None, **arguments):
    # Stands in for DshToolService.bind_prediction_tool, so it translates the
    # per-run execution budget the same way the real binder does.
    binding = DshPredictionToolBinding(budget=prediction_tool_call_budget, **arguments)
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


class TestActivationAuthority:
    """Explicit native-state double; envelopes cannot create their own leases.

    Integration tests separately exercise the actual rc.2 registry and HTTP
    attestation. These policy doubles record the session they create here.
    """
    def __init__(self):
        from threading import Lock
        self._lock = Lock()
        self._identities = {}

    def register(self, identity, parent_session_id):
        with self._lock:
            self._identities[identity["session_id"]] = (dict(identity), parent_session_id)

    def __call__(self, identity, *, parent_session_id, timeout):
        assert 0 < timeout <= 10
        with self._lock:
            return self._identities.get(identity["session_id"]) == (dict(identity), parent_session_id)
