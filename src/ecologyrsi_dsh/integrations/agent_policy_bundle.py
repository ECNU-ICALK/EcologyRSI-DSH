"""Portable Agent inference policy, with explicit training/runtime requirements."""
from collections.abc import Mapping
from pathlib import Path
import json
from ..core.models import digest
from ..evolution.agent_policy import validate_agent_policy
from ..version import __version__

SCHEMA = 'ecologyrsi-dsh.agent-policy-bundle/1'


def export_policy_bundle(artifact):
    learned = artifact.learned_parameters
    policy = learned.get('agent_policy')
    validate_agent_policy(policy)
    body = {'schema_version': SCHEMA, 'package_version': __version__,
            'policy': dict(policy), 'runtime_contract': dict(learned['runtime_contract']),
            'training_data_digest': learned['training_data_digest'],
            'default_model': {'tool_id': artifact.model_id, 'parameters': dict(artifact.parameters),
                'fit': {key: learned[key] for key in ('models', 'baseline_profile') if key in learned}},
            'optional_tool_catalog': learned['optional_tool_catalog'],
            'requirements': {'training_partition': 'training_fit', 'prediction_owner': 'sample_agent',
                             'new_origin_requires_causal_history': True,
                             'provider_credentials': 'supplied_by_runtime_never_in_bundle'},
            'source_artifact_digest': artifact.digest}
    # Detach immutable application objects without exporting the run's ledger or credentials.
    body = json.loads(json.dumps(body, default=lambda value: dict(value) if isinstance(value, Mapping) else list(value)))
    return {**body, 'bundle_digest': digest(body)}


def load_policy_bundle(path):
    bundle = json.loads(Path(path).read_text())
    validate_policy_bundle(bundle)
    return bundle


def validate_policy_bundle(bundle):
    if not isinstance(bundle, Mapping) or bundle.get('schema_version') != SCHEMA:
        raise ValueError('unsupported Agent policy bundle')
    if bundle.get('package_version') != __version__:
        raise ValueError('Agent policy bundle requires its exact runtime package version')
    if bundle.get('bundle_digest') != digest({k: v for k, v in bundle.items() if k != 'bundle_digest'}):
        raise ValueError('Agent policy bundle digest mismatch')
    validate_agent_policy(bundle['policy'])
    return bundle


def create_policy_adapter(bundle, series, *, runtime_contract, **runtime_bindings):
    """Load a policy for new causal origins with the ordinary native Agent chain.

    The caller supplies Host-admitted runtime bindings, never raw provider tokens
    in the exported policy. No fixed model output replaces the Agent's result.
    """
    from ..evaluators.agent_model_tools import AgentModelTools, _CONFIGS
    from ..evaluators.dsh_sample_adapter import DshSampleCollaborationAdapter
    from ..evaluators.greenhouse_prediction import predict_fitted_exogenous_ridge
    validate_policy_bundle(bundle)
    if dict(runtime_contract) != bundle['runtime_contract']:
        raise ValueError('deployed Agent runtime contract differs from exported policy')
    model = bundle['default_model']
    config = _CONFIGS[model['tool_id']].from_mapping(model['parameters'])
    fit = model['fit']
    bank = AgentModelTools(series, targets=sorted({m['target'] for m in fit['models']}),
                          horizons=sorted({m['horizon_hours'] for m in fit['models']}))
    if bank.training_digest != bundle['training_data_digest']:
        raise ValueError('training data differs from the policy model recipe')
    if bank.catalog() != bundle['optional_tool_catalog']:
        raise ValueError('registered numerical tool contract differs from exported policy')
    def default_tool(requests):
        result = {}
        for request in requests:
            baseline, context = bank._origin_input(request, config, fit)
            result[request.sample_id] = {'predicted': predict_fitted_exogenous_ridge(
                target=request.target, horizon_hours=request.horizon_hours, baseline=baseline,
                label_free_context=context, models=fit['models'], config=config),
                'metadata': {'model_id': model['tool_id'], 'parameters': model['parameters'], 'fit_digest': digest(fit['models'])}}
        return result
    adapter = DshSampleCollaborationAdapter(
        strategy_model_id=runtime_contract['strategy_model_id'], review_model_id=runtime_contract['review_model_id'],
        forecast_bundle_tool=default_tool, prediction_tool_catalog=bank.catalog(),
        prediction_tool_executor=bank.execute, **runtime_bindings)
    return adapter, {'candidate_agent_profile': bundle['policy']['profile'],
                     'candidate_parameters': model['parameters'], 'agent_policy': bundle['policy'],
                     'tool_experience': bundle['policy']['experience']['rows']}
