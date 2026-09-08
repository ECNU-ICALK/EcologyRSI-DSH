"""Frozen Agent policy and post-score experience for subsequent generations.

Only aggregate training-feedback evidence crosses this boundary. The policy is
fixed before inference and participates in the runtime phenotype identity.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
from ..core.models import digest

POLICY_SCHEMA = 'ecologyrsi-dsh.agent-policy/1'


def summarize_agent_tools(records, scoring_rows, *, source_phase='training_feedback'):
    scores = {row.get('sample_id'): row for row in scoring_rows}
    groups = {}
    for record in records:
        row = scores.get(record.get('sample_id'))
        if row is None or row.get('partition') != 'training_feedback':
            continue
        observed, final = row.get('observed'), row.get('predicted')
        if type(observed) not in (int, float) or not math.isfinite(observed):
            continue
        for attempt in record.get('attempt_trace', ()):
            for tool in attempt.get('model_evidence', ()):
                params = tool.get('parameters', {})
                identity = {'tool_id': tool['tool_id'], 'version': tool['version'],
                            'parameters': dict(params), 'target': row['target'], 'horizon_hours': row['horizon_hours']}
                key = digest(identity)
                group = groups.setdefault(key, {**identity, 'parameters_digest': digest(params), 'source_phase': source_phase,
                    'evidence_kind': 'raw_tool_output', 'calls': 0, 'failed': 0, 'n': 0,
                    '_elapsed_ms': 0., 'cited': 0, 'adjusted_n': 0, '_se': 0., '_ae': 0., '_delta': 0.})
                group['calls'] += 1
                group['_elapsed_ms'] += float(tool.get('elapsed_ms', 0))
                value = tool.get('tool_predicted')
                if tool.get('status') != 'completed' or type(value) not in (int, float) or not math.isfinite(value):
                    group['failed'] += 1
                    continue
                group['n'] += 1
                group['_se'] += (value-observed)**2
                group['_ae'] += abs(value-observed)
                if tool.get('used_as_evidence'):
                    group['cited'] += 1
                    if attempt.get('outcome') == 'accepted' and type(final) in (int, float) and math.isfinite(final):
                        group['adjusted_n'] += 1
                        group['_delta'] += abs(value-observed)-abs(final-observed)
    result = []
    for _, group in sorted(groups.items()):
        n = group['n']; adjusted = group['adjusted_n']
        se, ae, delta = (group.pop(key) for key in ('_se', '_ae', '_delta'))
        elapsed = group.pop('_elapsed_ms')
        result.append({**group, 'mean_tool_latency_ms': elapsed / group['calls'], 'rmse': math.sqrt(se/n) if n else None,
                       'mae': ae/n if n else None,
                       'agent_adjustment_mae_gain': delta/adjusted if adjusted else None})
    return sorted(result, key=lambda row: (-row['n'], row['parameters_digest'], row['tool_id'], row['target'], row['horizon_hours']))[:64]


def prior_candidate_tool_experience(state, candidate, metrics):
    """Use adaptation batches, never normalized holdout rows, as tool experience."""
    def tool_rows(values):
        summary = values.get('sample_execution')
        rows = summary.get('agent_tool_performance') if isinstance(summary, Mapping) else None
        return rows if isinstance(rows, (list, tuple)) else ()
    result = []
    for evaluation in getattr(state, 'formal_batch_evaluations', ()):
        scope = evaluation.scope
        if (scope.candidate_id != candidate.candidate_id or scope.generation != candidate.generation
                or getattr(scope.phase, 'value', scope.phase) != 'formal_batch'):
            continue
        rows = tool_rows(evaluation.metrics)
        result.extend({**dict(row), 'source_scope_digest': scope.scope_key}
                      for row in rows if isinstance(row, Mapping) and row.get('source_phase') == 'formal_batch')
    if not result:
        rows = tool_rows(metrics)
        result = [dict(row) for row in rows if isinstance(row, Mapping)
                  and row.get('source_phase') == 'training_feedback']
    return result[-64:]


def build_agent_policy(*, genome_digest, profile, parameters, previous_analysis, generation):
    experience = []
    if isinstance(previous_analysis, Mapping):
        source_generation = previous_analysis.get('generation')
        # Reject same/future-generation feedback rather than silently leaking it.
        if source_generation is not None and (type(source_generation) is not int or source_generation >= generation):
            raise ValueError('Agent experience must precede this generation')
        for candidate in previous_analysis.get('ranking', ())[:8]:
            for row in candidate.get('agent_tool_performance', ())[:32]:
                if (isinstance(row, Mapping) and row.get('evidence_kind') == 'raw_tool_output'
                        and row.get('source_phase') in ('formal_batch', 'training_feedback')):
                    experience.append(deepcopy(dict(row)))
                if len(experience) == 32:
                    break
            if len(experience) == 32:
                break
    body = {'schema_version': POLICY_SCHEMA, 'genome_digest': genome_digest,
            'profile': deepcopy(dict(profile)), 'default_model_parameters': deepcopy(dict(parameters)),
            'experience': {'scope': 'prior_generation_training_feedback_aggregates',
                'source_analysis_digest': digest(previous_analysis) if previous_analysis else None,
                'rows': experience},
            'inference': {'prediction_owner': 'sample_agent', 'max_tool_calls_per_attempt': 6,
                          'model_parameter_selection': 'agent_within_registered_bounds',
                          'critic_protocol': 'ecology-sample-review@2', 'holdout_replicates': 2}}
    return {**body, 'policy_digest': digest(body)}


def validate_agent_policy(value):
    if not isinstance(value, Mapping) or value.get('schema_version') != POLICY_SCHEMA:
        raise ValueError('Agent policy schema mismatch')
    body = {key: item for key, item in value.items() if key != 'policy_digest'}
    if value.get('policy_digest') != digest(body):
        raise ValueError('Agent policy digest mismatch')
    return value


def rebind_agent_policy(policy, *, genome_digest, profile, parameters):
    """Local edits change policy source while preserving frozen prior experience."""
    base = build_agent_policy(genome_digest=genome_digest, profile=profile, parameters=parameters,
                              previous_analysis=None, generation=0)
    if policy is not None:
        validate_agent_policy(policy)
        base["experience"] = deepcopy(dict(policy["experience"]))
    base["policy_digest"] = digest({key: value for key, value in base.items() if key != "policy_digest"})
    return base
