"""Frozen Agent policy and post-score experience for subsequent generations.

Only aggregate training-feedback evidence crosses this boundary. The policy is
fixed before inference and participates in the runtime phenotype identity.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
from ..core.agent_prediction import (
    MAX_PREDICTION_CALLS,
    PREDICTION_METHODS,
    PREDICTION_TOOL_CALL_BUDGET,
)
from ..core.models import digest
from ..core.immutable import thaw_json

POLICY_SCHEMA = 'ecologyrsi-dsh.agent-policy/1'
ADAPTIVE_TOOL_FEEDBACK_KEY = 'adaptive_tool_feedback_policy'
ADAPTIVE_TOOL_FEEDBACK_POLICY = 'sealed_adaptation_tools@1'


def adaptive_tool_feedback_enabled(metadata):
    value = metadata.get(ADAPTIVE_TOOL_FEEDBACK_KEY)
    if value not in (None, ADAPTIVE_TOOL_FEEDBACK_POLICY):
        raise ValueError('unsupported adaptive_tool_feedback_policy')
    return value == ADAPTIVE_TOOL_FEEDBACK_POLICY


def summarize_agent_tools(records, scoring_rows, *, source_phase='training_feedback', candidate_identity=None):
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
                if isinstance(candidate_identity, Mapping):
                    # Tool arguments are request overrides, not a complete model
                    # configuration. In particular candidate-model accepts no
                    # overrides and resolves to this frozen candidate's model.
                    alias = tool['tool_id'] == 'candidate-model'
                    effective = (candidate_identity.get('parameters_digest') if alias and not params
                                 else tool.get('effective_parameters_digest'))
                    valid_digest = (isinstance(effective, str) and len(effective) == 64
                                    and all(c in '0123456789abcdef' for c in effective))
                    identity.update({
                        'parameter_identity_schema': 'ecologyrsi-dsh.tool-parameter-identity/1',
                        'parameters_semantics': 'request_overrides_only',
                        'observation_unit': 'tool_calls_per_scoring_cell_including_retries',
                        'adjustment_gain_semantics': 'cited_accepted_attempts_observational_not_causal',
                        'resolved_predictor_id': candidate_identity.get('predictor_id') if alias else tool['tool_id'],
                        'effective_parameters_digest': effective if valid_digest else None,
                        'effective_parameters_source': ('frozen_candidate' if alias and not params else 'tool_receipt')
                            if valid_digest else 'unverified',
                    })
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


def prior_candidate_attribution(state, candidate):
    """Latest sealed adaptation report, never holdout or independent labels.

    Keep the entire target/horizon grid of one revision. Mixing revisions or
    truncating cells by call frequency would hide adverse local-edit evidence.
    """
    from ..evaluators.candidate_attribution import enabled, validated_summary, METRIC_KEY
    if (not enabled(state.task_manifest.metadata)
            or state.task_manifest.metadata.get("evaluation_partition", "training_feedback") != "training_feedback"):
        return None
    selected = None
    selected_index = -1
    for evaluation in getattr(state, 'formal_batch_evaluations', ()):
        scope = evaluation.scope
        index = getattr(scope, 'batch_index', None)
        if (scope.candidate_id != candidate.candidate_id or scope.generation != candidate.generation
                or getattr(scope.phase, 'value', scope.phase) != 'formal_batch'
                or type(index) is not int or index <= 0 or index < selected_index):
            continue
        summary = evaluation.metrics.get('sample_execution', {}).get(METRIC_KEY)
        checked = validated_summary(summary, phases={'formal_batch'})
        if checked is None or checked.get('cohort_digest') != scope.cohort_digest:
            continue
        body = {key: value for key, value in checked.items() if key != 'summary_digest'}
        body.update(source_scope_digest=scope.scope_key, source_generation=scope.generation,
            source_candidate_id=scope.candidate_id, source_candidate_revision_id=scope.candidate_revision_id,
            sealed_summary_digest=checked['summary_digest'])
        selected = {**body, 'summary_digest': digest(body)}
        selected_index = index
    return selected


def build_agent_policy(*, genome_digest, profile, parameters, previous_analysis, generation,
                       prediction_tool_call_budget=None, prediction_formula_policy=None, joint=False):
    budget = PREDICTION_TOOL_CALL_BUDGET if prediction_tool_call_budget is None else prediction_tool_call_budget
    if type(budget) is not int or not 1 <= budget <= MAX_PREDICTION_CALLS:
        raise ValueError('prediction tool call budget must be a positive int within the receipt cap')
    if prediction_formula_policy not in (None, "mean-referenced-tools@1", "evidence-mean-or-model@1"):
        raise ValueError("unsupported frozen prediction formula policy")
    if prediction_formula_policy == "evidence-mean-or-model@1":
        if not set(allowed_prediction_methods(profile)) & {"model", "blend"}:
            raise ValueError("frozen prediction formula requires model or blend methods")
        directive = profile.get("authored_directive")
        if isinstance(directive, Mapping) and directive.get("blend_rule") not in ("none", "mean"):
            raise ValueError("frozen prediction formula conflicts with authored blend_rule")
    experience = []
    attribution = []
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
    if isinstance(previous_analysis, Mapping):
        from ..evaluators.candidate_attribution import validated_summary, METRIC_KEY
        for candidate in previous_analysis.get('ranking', ())[:8]:
            checked = validated_summary(candidate.get(METRIC_KEY), phases={'formal_batch'})
            if checked is None:
                continue
            source_generation = checked.get('source_generation')
            if type(source_generation) is not int or source_generation >= generation:
                raise ValueError('Candidate attribution must precede this generation')
            # Whole 9-cell reports, never the most frequently invoked tools.
            attribution.append(checked)
            if len(attribution) == 4:
                break
    body = {'schema_version': POLICY_SCHEMA, 'genome_digest': genome_digest,
            'profile': deepcopy(dict(profile)), 'default_model_parameters': deepcopy(dict(parameters)),
            'experience': {'scope': 'prior_generation_training_feedback_aggregates',
                'source_analysis_digest': digest(previous_analysis) if previous_analysis else None,
                'rows': experience},
            'inference': {'prediction_owner': 'sample_agent',
                          'max_tool_calls_per_attempt': budget,
                          'model_parameter_selection': 'agent_within_registered_bounds',
                          'critic_protocol': 'ecology-sample-review@2', 'holdout_replicates': 2}}
    if attribution:
        body['experience']['candidate_attribution'] = attribution
    if isinstance(previous_analysis, Mapping):
        for candidate in previous_analysis.get('ranking', ())[:8]:
            noise = candidate.get('same_genome_repeatability')
            if not isinstance(noise, Mapping):
                continue
            counts = (noise.get('paired_origin_count'), noise.get('changed_prediction_count'))
            drift = noise.get('absolute_score_drift')
            if (not all(type(v) is int and v >= 0 for v in counts)
                    or type(drift) not in (int, float) or not math.isfinite(drift) or drift < 0):
                raise ValueError('same-genome repeatability must contain finite aggregate evidence')
            body['experience']['same_genome_repeatability'] = {
                'paired_origin_count': counts[0], 'changed_prediction_count': counts[1],
                'absolute_score_drift': drift,
                'interpretation': 'One A/A pair measures realized Agent drift, not a confidence bound or causal attribution; no new independent observations; never replace the selection score with the better repeat.',
            }
            break
    if joint:
        from .joint_policy import INFERENCE_POLICY
        body['inference']['tool_selection_policy'] = INFERENCE_POLICY
    if prediction_formula_policy is not None:
        body['inference']['prediction_formula_policy'] = prediction_formula_policy
    return {**body, 'policy_digest': digest(body)}


def validate_agent_policy(value):
    if not isinstance(value, Mapping) or value.get('schema_version') != POLICY_SCHEMA:
        raise ValueError('Agent policy schema mismatch')
    body = {key: item for key, item in value.items() if key != 'policy_digest'}
    if value.get('policy_digest') != digest(body):
        raise ValueError('Agent policy digest mismatch')
    return value


def frozen_prediction_tool_call_budget(policy):
    """The execution budget this run froze, not the current default.

    Both the enforcing binding and the contract shown to the Agent must read the
    same per-run value; a mismatch would either promise calls the binding
    refuses, or hide calls it would allow.
    """
    if isinstance(policy, Mapping):
        inference = policy.get('inference')
        if isinstance(inference, Mapping):
            budget = inference.get('max_tool_calls_per_attempt')
            if type(budget) is int and 1 <= budget <= MAX_PREDICTION_CALLS:
                return budget
    return PREDICTION_TOOL_CALL_BUDGET


def allowed_prediction_methods(profile):
    """The methods this candidate's directive admits, or all of them.

    Read defensively from the archived profile for the same reason the budget
    above is: the contract shown to the Agent and the check applied to its
    answer must come from one value, and a candidate that authored no directive
    must be validated exactly as it was before authoring existed.
    """
    if isinstance(profile, Mapping):
        methods = profile.get('allowed_prediction_methods')
        if isinstance(methods, (list, tuple)) and methods:
            permitted = frozenset(str(item) for item in methods)
            if permitted <= PREDICTION_METHODS:
                return tuple(sorted(permitted))
    return tuple(sorted(PREDICTION_METHODS))


def rebind_agent_policy(policy, *, genome_digest, profile, parameters):
    """Local edits change policy source while preserving frozen prior experience.

    The execution budget is frozen per run, so it is carried over from the
    supplied policy rather than re-read from the current default: an archived
    artifact has to rebind to the budget it was scored under, or raising
    PREDICTION_TOOL_CALL_BUDGET would make every archived run fail the frozen
    Agent policy check in evaluators/registry.py.
    """
    frozen_budget = None
    if policy is not None:
        validate_agent_policy(policy)
        frozen_budget = frozen_prediction_tool_call_budget(policy)
    base = build_agent_policy(genome_digest=genome_digest, profile=profile, parameters=parameters,
                              previous_analysis=None, generation=0,
                              prediction_tool_call_budget=frozen_budget,
                              joint=bool(policy and policy.get("inference", {}).get("tool_selection_policy") == "autonomous-plugin-tools@1"),
                              prediction_formula_policy=(policy.get("inference", {}).get("prediction_formula_policy")
                                                         if policy is not None else None))
    if policy is not None:
        base["experience"] = deepcopy(dict(policy["experience"]))
    base["policy_digest"] = digest({key: value for key, value in base.items() if key != "policy_digest"})
    return base
