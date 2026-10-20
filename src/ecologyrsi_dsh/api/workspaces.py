"""Independent browser workspaces; full audit data is loaded explicitly."""
from .projection import (
    _assert_http_scope, _artifact_projection, _candidate_projection,
    _rounds_projection, _search_candidates, _projection_json,
)

WORKSPACE_VIEWS = frozenset({'overview', 'process', 'candidates', 'candidate'})


def overview_projection(state):
    return _projection_json(state, overview_only=True)


def workspace_section(state, view, candidate_id=None):
    _assert_http_scope(state)
    if view == 'overview':
        return {}
    if view == 'process':
        # Execution cards need current identity, score and progress, while the
        # metrics, model internals and sample traces belong to candidate detail.
        return {
            'rounds': _rounds_projection(state),
            'candidate_summaries': [
                _candidate_projection(state, c, summary_only=True)
                for c in reversed(_search_candidates(state))
            ],
        }
    if view == 'candidates':
        summaries = []
        for candidate in reversed(_search_candidates(state)):
            item = _candidate_projection(state, candidate, summary_only=True)
            evaluation = state.evaluation_for(candidate.candidate_id)
            item['metrics'] = {
                key: evaluation.metrics[key] for key in (
                    'skill_score', 'rmse', 'n', 'constraint_violations',
                    'scientific_pass', 'judge_accepted', 'judge_status',
                ) if evaluation is not None and key in evaluation.metrics
            }
            item['details_loaded'] = False
            summaries.append(item)
        return {
            'candidates': summaries,
            'artifacts': [],
        }
    if view == 'candidate':
        candidate = next((c for c in _search_candidates(state) if c.candidate_id == candidate_id), None)
        if candidate is None:
            raise KeyError('unknown candidate')
        return {
            'candidate': {**_candidate_projection(state, candidate), 'details_loaded': True},
            'artifacts': [_artifact_projection(state, a) for a in reversed(state.artifacts)
                          if a.candidate_id == candidate_id],
        }
    raise ValueError('unsupported workspace view')
