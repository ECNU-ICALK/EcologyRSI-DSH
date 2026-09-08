"""Independent browser workspaces; full audit data is loaded explicitly."""
from .projection import (
    _assert_http_scope, _artifact_projection, _candidate_projection,
    _expert_consultation_projection, _intervention_projection,
    _rounds_projection, _search_candidates, _projection_json,
)
from ..presentation.training_assets import training_assets

WORKSPACE_VIEWS = frozenset({'overview', 'process', 'candidates', 'training', 'collaboration', 'asset'})


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
        return {
            'candidates': [_candidate_projection(state, c) for c in reversed(_search_candidates(state))],
            'artifacts': [_artifact_projection(state, a) for a in reversed(state.artifacts)],
        }
    if view == 'training':
        return {'training_assets': training_assets(state, summary_only=True)}
    if view == 'collaboration':
        return {
            'interventions': [_intervention_projection(state, i) for i in reversed(state.interventions)],
            'expert_consultations': [_expert_consultation_projection(state, i) for i in reversed(state.expert_consultations)],
            'intervention_candidates': [
                {'id': c.candidate_id, 'candidate_id': c.candidate_id,
                 'generation': c.generation + 1, 'title': state.proposal(c.proposal_id).title}
                for c in reversed(_search_candidates(state))
                if c.status.value in {'promoted', 'rejected'} and state.promotion_for(c.candidate_id) is not None
            ],
        }
    if view == 'asset':
        if not candidate_id or not any(c.candidate_id == candidate_id for c in _search_candidates(state)):
            raise KeyError('unknown training asset candidate')
        return {'training_asset': training_assets(state, candidate_id=candidate_id)[0]}
    raise ValueError('unsupported workspace view')
