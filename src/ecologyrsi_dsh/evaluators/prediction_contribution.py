"""Label-free observations of tool outputs and final predictions, never scores.

One row is one scoring cell, not one billable API/tool call. Missing receipts
remain unknown. Equality is descriptive and does not establish causal use.
"""
from collections.abc import Mapping
import json
from math import isfinite, isclose
from ..core.models import digest
from ..core.immutable import thaw_json
from ..core.prediction_policy import CONTRIBUTION_POLICY, LEGACY_CONTRIBUTION_POLICY, PREVIOUS_CONTRIBUTION_POLICY

LEGACY_SCHEMA = 'ecologyrsi-dsh.prediction-contribution/1'
PREVIOUS_SCHEMA = 'ecologyrsi-dsh.prediction-contribution/2'
SCHEMA = 'ecologyrsi-dsh.prediction-contribution/3'
_SCHEMAS = {LEGACY_CONTRIBUTION_POLICY: LEGACY_SCHEMA, PREVIOUS_CONTRIBUTION_POLICY: PREVIOUS_SCHEMA,
            CONTRIBUTION_POLICY: SCHEMA}


def compiled_candidate_identity(metadata, parameters, *, policy=CONTRIBUTION_POLICY):
    """Identify the Host-compiled model; never infer identity from tool names."""
    if policy != CONTRIBUTION_POLICY:
        return None
    raw = metadata.get('evolution_genome_canonical_json') if isinstance(metadata, Mapping) else None
    if not isinstance(raw, str) or not isinstance(parameters, Mapping):
        return None
    scientific = json.loads(raw)['scientific_program']
    return {'predictor_id': scientific['predictor_ref']['id'],
            'parameters_digest': digest(thaw_json(
                {'feature_recipe': scientific['feature_recipe']} if 'feature_recipe' in scientific else parameters))}


def _equivalent_candidate_tool(tool, accepted, identity):
    if not isinstance(identity, Mapping) or tool.get('tool_id') != identity.get('predictor_id'):
        return False
    if not tool.get('call_id') or not identity.get('parameters_digest'):
        return False
    # Scoring rows deliberately contain a small tool view. Join it back to the
    # accepted attempt's Host-normalized receipt by sample/call/value, so an
    # alternate parameterization or an earlier abandoned attempt cannot count.
    return any(receipt.get('tool_id') == tool.get('tool_id')
               and receipt.get('call_id') == tool['call_id']
               and receipt.get('status') == 'completed'
               and receipt.get('dsh_tool_event_id')
               and receipt.get('effective_parameters_digest') == identity['parameters_digest']
               and _number(receipt.get('tool_predicted')) == _number(tool.get('tool_predicted'))
               for receipt in (accepted or {}).get('model_evidence', ()) if isinstance(receipt, Mapping))


def _input_identities(context):
    """Separate observed inputs from the candidate's mutable configuration.

    Retain actual feature values and causal provenance, including the recipe
    read contract: a different feature/read set is not a controlled comparison.
    In particular, a baseline-only branch with no prepared residual features
    cannot prove the same inputs as an active branch merely by sharing history.
    No future observation, score, or inferred missing value enters either hash.
    """
    observations = thaw_json(context)
    treatment = observations.pop('predictor_state', None)
    if not any(key != 'schema_version' for key in observations):
        return None, None
    reads = {
        name: treatment[name]
        for name in ('feature_recipe_digest', 'feature_recipe_anchor',
                     'required_timestamp_offsets', 'cohort_history_hours')
        if isinstance(treatment, Mapping) and name in treatment
    }
    observation_digest = digest({'context': observations, 'read_contract': reads})
    treatment_digest = digest(treatment) if isinstance(treatment, Mapping) and treatment else None
    return observation_digest, treatment_digest


def _number(value):
    return float(value) if type(value) in (int, float) and isfinite(value) else None


def summarize_contribution(records, rows, *, cohort_digest=None, source_phase=None, review_policy=None,
                           policy=CONTRIBUTION_POLICY, candidate_identity=None):
    if policy not in _SCHEMAS:
        raise ValueError('unsupported prediction contribution policy')
    legacy = policy == LEGACY_CONTRIBUTION_POLICY
    identity = candidate_identity if policy == CONTRIBUTION_POLICY else None
    threshold = _number(review_policy.get("min_planner_confidence")) if isinstance(review_policy, Mapping) else None
    by_id = {r.get('sample_id'): r for r in records}
    groups = {}
    for row in rows:
        final, baseline = _number(row.get('predicted')), _number(row.get('baseline'))
        if final is None or baseline is None:
            continue
        key = (str(row.get('target')), int(row.get('horizon_hours', 1)))
        group = groups.setdefault(key, dict(target=key[0], horizon_hours=key[1], cells=0,
            candidate_tool_cells=0, candidate_cited_cells=0, model_equals_baseline=0,
            final_equals_model=0, final_equals_baseline=0, trace_available_cells=0,
            critic_cells=0, skill_cells=0, _model_delta=0., _final_delta=0., _confidence=[],
            _inputs=[], _outputs=[], _treatments=[]))
        group['cells'] += 1
        group['final_equals_baseline'] += isclose(final, baseline, rel_tol=0, abs_tol=1e-12)
        info = row.get('agent_prediction', {})
        accepted = [a for a in by_id.get(row.get('sample_id'), {}).get('attempt_trace', ())
                    if isinstance(a, Mapping) and a.get('outcome') == 'accepted']
        tools = [t for t in info.get('tools', ()) if isinstance(t, Mapping)
                 and t.get('status') == 'completed' and _number(t.get('tool_predicted')) is not None
                 and (t.get('tool_id') == 'candidate-model'
                      or _equivalent_candidate_tool(t, accepted[-1] if accepted else None, identity))]
        confidence = _number(info.get('confidence'))
        if confidence is not None:
            group['_confidence'].append(confidence)
        # Several calls on a cell are not independent cells. Observe the last
        # cited candidate output, or the last completed output if none is cited.
        cited = [t for t in tools if t.get('used_as_evidence') is True]
        if tools:
            tool = (cited or tools)[-1]
            value = float(tool['tool_predicted'])
            group['candidate_tool_cells'] += 1
            group['candidate_cited_cells'] += bool(cited)
            group['model_equals_baseline'] += isclose(value, baseline, rel_tol=0, abs_tol=1e-12)
            group['final_equals_model'] += isclose(value, final, rel_tol=0, abs_tol=1e-12)
            group['_model_delta'] += abs(value-baseline)
            group['_final_delta'] += abs(final-value)
            # @1 must retain its original context hash and output bytes. @2
            # freezes observed inputs separately from candidate configuration.
            context = row.get('label_free_context')
            if isinstance(context, Mapping) and context:
                ident = (key[0], key[1], row.get('origin_timestamp'))
                input_digest, treatment_digest = (
                    (digest(thaw_json(context)), None) if legacy else _input_identities(context)
                )
                if input_digest is not None:
                    group['_inputs'].append((ident, input_digest, baseline))
                    group['_outputs'].append((ident, value))
                if treatment_digest is not None:
                    group['_treatments'].append((ident, treatment_digest))
        if accepted:
            trace = accepted[-1]
            group['trace_available_cells'] += 1
            group['critic_cells'] += bool(trace.get('critic_decisions'))
            group['skill_cells'] += bool(trace.get('skill_evidence'))
    cells = []
    for _, group in sorted(groups.items()):
        n = group['candidate_tool_cells']
        confidence = group.pop('_confidence')
        inputs, outputs = group.pop('_inputs'), group.pop('_outputs')
        treatments = group.pop('_treatments')
        model_delta, final_delta = group.pop('_model_delta'), group.pop('_final_delta')
        cells.append({**group, 'model_baseline_mean_absolute_delta': model_delta / n if n else None,
            'final_model_mean_absolute_delta': final_delta / n if n else None,
            'branch_evidence_complete': group['trace_available_cells'] == group['cells'],
            'critic_confidence_threshold': threshold,
            'confidence_below_threshold_cells': sum(value < threshold for value in confidence) if threshold is not None else None,
            'confidence_observed_cells': len(confidence),
            'confidence_min': min(confidence) if confidence else None,
            'confidence_max': max(confidence) if confidence else None,
            'input_vector_digest': digest(sorted(inputs)) if len(inputs) == group['cells'] else None,
            'tool_vector_digest': digest(sorted(outputs)) if len(outputs) == group['cells'] else None,
            **({} if legacy else {
                'observation_vector_digest': digest(sorted(inputs)) if len(inputs) == group['cells'] else None,
                'treatment_vector_digest': digest(sorted(treatments)) if len(treatments) == group['cells'] else None,
            })})
    body = {'schema_version': _SCHEMAS[policy], 'policy': policy,
        'source_phase': source_phase, 'cohort_digest': cohort_digest, 'cells': cells,
        'unit': 'scored_cell', 'scientific_gain_established': False,
        'interpretation': 'output_equality_is_not_causal_attribution; absent_trace_is_unknown'}
    if identity is not None:
        body['candidate_model_identity'] = thaw_json(identity)
    return {**body, 'summary_digest': digest(body)}


def paired_tool_effect(parent, child):
    """Compare raw candidate-tool vectors only when all input identities match."""
    if (not parent or not child or parent.get('schema_version') not in _SCHEMAS.values()
            or child.get('schema_version') not in _SCHEMAS.values() or not parent.get('cohort_digest')
            or parent.get('cohort_digest') != child.get('cohort_digest')):
        return {'status': 'unverified', 'reason': 'same_cohort_required'}
    if parent['schema_version'] != child['schema_version']:
        return {'status': 'unverified', 'reason': 'same_diagnostic_version_required'}
    expected_policy = next(policy for policy, schema in _SCHEMAS.items() if schema == parent['schema_version'])
    if parent['schema_version'] != LEGACY_SCHEMA and (
            parent.get('policy') != expected_policy or child.get('policy') != expected_policy):
        return {'status': 'unverified', 'reason': 'same_diagnostic_version_required'}
    before = {(c['target'], c['horizon_hours']): c for c in parent['cells']}
    after = {(c['target'], c['horizon_hours']): c for c in child['cells']}
    if not before or before.keys() != after.keys():
        return {'status': 'unverified', 'reason': 'cell_coverage_mismatch'}
    changed = []
    for key, cell in before.items():
        other = after[key]
        if parent['schema_version'] != LEGACY_SCHEMA and (
                cell.get('observation_vector_digest') != cell.get('input_vector_digest')
                or other.get('observation_vector_digest') != other.get('input_vector_digest')):
            return {'status': 'unverified', 'reason': 'complete_matching_inputs_required'}
        if (not cell.get('input_vector_digest') or cell['input_vector_digest'] != other.get('input_vector_digest')
                or not cell.get('tool_vector_digest') or not other.get('tool_vector_digest')):
            return {'status': 'unverified', 'reason': 'complete_matching_inputs_required'}
        if cell['tool_vector_digest'] != other['tool_vector_digest']:
            changed.append(f'{key[0]}@{key[1]}h')
    return {'status': 'measured', 'changed_cells': changed, 'compared_cells': len(before),
            'scientific_gain_established': False, 'cohort_digest': parent['cohort_digest']}
