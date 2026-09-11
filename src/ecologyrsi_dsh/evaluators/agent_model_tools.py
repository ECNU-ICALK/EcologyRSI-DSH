"""Optional numerical capabilities with restart-safe fits and per-origin inference.

Memory capacity is an LRU policy, not an exploration quota. The Agent's frozen
per-attempt call budget bounds exploration regardless of concurrent origin order.
"""
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from threading import RLock, BoundedSemaphore
import json
import os
import tempfile

from . import greenhouse_prediction as ridge
from .feature_recipe import recipe_grammar
from ..core.models import digest
from .sample_execution import SampleExecutionCancelledError, SampleExecutionPausedError

_CONFIGS = {
    'greenhouse-exogenous-ridge@1': ridge.ExogenousRidgeConfig,
    'greenhouse-baseline-aligned-ridge@1': ridge.BaselineAlignedRidgeConfig,
    'greenhouse-targetwise-ridge@1': ridge.TargetwiseExogenousRidgeConfig,
    'greenhouse-horizon-targetwise-ridge@1': ridge.HorizonTargetwiseExogenousRidgeConfig,
    'greenhouse-recipe-ridge@1': ridge.RecipeRidgeConfig,
}
_FIT_WORKERS = BoundedSemaphore(2)
# Residual scaling is applied after the fit, so it must not fragment the fit
# cache. The scalar path pins every ``*residual_scale*`` field to this value;
# the recipe path rewrites its per-horizon map to the same constant.
_FIT_IDENTITY_RESIDUAL_SCALE = .5


def _defaults(config_type, horizons=(1, 6, 24)):
    if config_type is ridge.RecipeRidgeConfig:
        # Target-history core only: it compiles against any dataset, so the tool
        # default never depends on a column this ecology may not have. Exogenous
        # terms are the Agent's (or the adapter's) explicit choice.
        return ridge.seed_recipe_parameters(horizons=horizons)
    values = {'history_steps': 6, 'ridge_alpha': 0.1}
    values.update({name: _FIT_IDENTITY_RESIDUAL_SCALE for name in config_type.__dataclass_fields__
                   if 'residual_scale' in name and not name.startswith('_')})
    return config_type.from_mapping(values).to_dict()


class AgentModelTools:
    CACHE_CAPACITY = 8

    def __init__(self, series, *, targets, horizons, control=None, cache_dir=None, default_fit=None, default_config=None):
        self.series, self.targets, self.horizons = series, tuple(targets), tuple(horizons)
        self.control = control
        fit_range = series.partitions['training_fit']
        self.training_digest = digest({'timestamps': series.timestamps[fit_range.start:fit_range.end],
            'values': {name: values[fit_range.start:fit_range.end] for name, values in series.values.items()},
            'features': {name: getattr(feature, 'role', None) for name, feature in series.features.items()}})
        self._fits = OrderedDict()
        self._lock = RLock()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._index = {t: i for i, t in enumerate(series.timestamps)}
        self._filled = None
        self._plans = {}
        if default_fit is not None and default_config is not None and all(m['status'] != 'baseline_only' for m in default_fit['models']):
            tool_id = next(name for name, cls in _CONFIGS.items() if type(default_config) is cls)
            key, _ = self._fit_identity(tool_id, default_config)
            self._fits[key] = {k: default_fit[k] for k in ('models', 'baseline_profile') if k in default_fit}

    def catalog(self):
        entries = []
        for name, config in _CONFIGS.items():
            if config is ridge.RecipeRidgeConfig:
                # The recipe tool advertises the primitive whitelist itself, so
                # the Agent reads the same bounds the host enforces instead of
                # guessing from a scalar range that does not apply.
                entries.append({
                    'tool_id': name, 'version': '1',
                    'purpose': 'Ridge over a declarative feature recipe; the host compiles and bounds every primitive',
                    'parameters': {'feature_recipe': {
                        'default': _defaults(config, self.horizons)['feature_recipe'],
                        'type': 'object', 'grammar': recipe_grammar(),
                    }},
                })
                continue
            entries.append({
                'tool_id': name, 'version': '1',
                'purpose': 'Optional ridge fitted only on training_fit; parameters are selected by the Agent',
                'parameters': {key: {'default': value, 'minimum': 1 if key == 'history_steps' else .0001 if key == 'ridge_alpha' else 0,
                                     'maximum': 12 if key == 'history_steps' else 1,
                                     'type': 'integer' if key == 'history_steps' else 'number'}
                               for key, value in _defaults(config, self.horizons).items()},
            })
        return entries

    def _check_control(self):
        if self.control is None:
            return
        state = self.control()
        if state == 'paused':
            raise SampleExecutionPausedError('Agent model fit paused')
        if state != 'running':
            raise SampleExecutionCancelledError('Agent model fit cancelled')

    def _fit_identity(self, tool_id, config):
        values = {key: _FIT_IDENTITY_RESIDUAL_SCALE if 'residual_scale' in key else value
                  for key, value in config.to_dict().items()}
        if 'feature_recipe' in values:
            recipe = dict(values['feature_recipe'])
            recipe.pop('per_cell', None)
            recipe['per_horizon'] = {str(h): {'residual_scale': _FIT_IDENTITY_RESIDUAL_SCALE}
                                     for h in sorted(self.horizons)}
            values['feature_recipe'] = recipe
        identity = {'schema': 'agent-ridge-fit/3', 'training_digest': self.training_digest,
                    'training_range': [self.series.partitions['training_fit'].start, self.series.partitions['training_fit'].end],
                    'targets': self.targets, 'horizons': self.horizons, 'tool_id': tool_id, 'parameters': values}
        return digest(identity), _CONFIGS[tool_id].from_mapping(values)

    @contextmanager
    def _worker(self):
        while not _FIT_WORKERS.acquire(timeout=.1):
            self._check_control()
        try:
            self._check_control()
            yield
        finally:
            _FIT_WORKERS.release()

    def _load_or_fit(self, key, fit_config):
        path = self.cache_dir / (key + '.json') if self.cache_dir is not None else None
        if path is not None and path.exists():
            entry = json.loads(path.read_text())
            if entry.get('key') != key or entry.get('content_digest') != digest(entry.get('fit')):
                raise ValueError('cached Agent fit identity mismatch')
            return entry['fit']
        with self._worker():
            fitted = ridge.fit_predict_exogenous_ridge(
                self.series, targets=self.targets, horizons=self.horizons,
                config=fit_config, prediction_partitions=(), check_control=self._check_control,
            )
        fit = {name: fitted[name] for name in ('models', 'baseline_profile') if name in fitted}
        self._check_control()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=key + '.', suffix='.tmp', dir=path.parent)
            try:
                with os.fdopen(fd, 'w') as stream:
                    json.dump({'key': key, 'content_digest': digest(fit), 'fit': fit}, stream, allow_nan=False)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return fit

    def _plan(self, config, target, horizon_hours):
        """Compile (and memoize) the recipe read plan for one cell.

        The plan is the single source of the required offsets, so this path and
        the batch fit path cannot drift into reading different history.
        """
        if not isinstance(config, ridge.RecipeRidgeConfig):
            return None
        key = (config.recipe.digest, target, horizon_hours)
        if key not in self._plans:
            self._plans[key] = ridge.compile_feature_plan(
                config.recipe, series=self.series, target=target,
                horizon_hours=horizon_hours, allowed_roles=ridge._ALLOWED_EXOGENOUS_ROLES,
                max_history_hours=ridge.COHORT_HISTORY_HOURS,
            )
        return self._plans[key]

    def _origin_input(self, request, config, fit):
        selected = self.series.partitions['training_feedback']
        origin = self._index.get(request.origin_timestamp)
        if origin is None or not selected.start <= origin < selected.end:
            raise ValueError('tool origin is outside the frozen feedback partition')
        plan = self._plan(config, request.target, request.horizon_hours)
        offsets = (plan.required_timestamp_offsets if plan is not None
                   else tuple(-lag for lag in range(config.history_steps)))
        lag_indices = [self._index.get(request.origin_timestamp + offset) for offset in offsets]
        if any(i is None or not selected.start <= i <= origin for i in lag_indices):
            raise ValueError('tool requires complete causal history')
        lags = tuple(ridge._finite_value(self.series.values[request.target][i]) for i in lag_indices)
        if any(v is None for v in lags):
            raise ValueError('tool history contains missing target values')
        sample = ridge._BaseSample(origin, origin, lags, tuple(self.series.timestamps[i] for i in lag_indices), 0., lags[0])
        visible = {time: index for time, index in self._index.items()
                   if index <= origin and any(r.start <= index < r.end for name, r in self.series.partitions.items() if name in {'training_fit', 'training_feedback'})} \
            if 'baseline_profile' in fit or plan is not None else {}
        if 'baseline_profile' in fit:
            sample = ridge._align_sample_baselines(self.series, request.target, request.horizon_hours,
                       (sample,), fit['baseline_profile'], visible)[0]
        model = next(m for m in fit['models'] if m['target'] == request.target and m['horizon_hours'] == request.horizon_hours)
        stats = [ridge._FeatureStatistic(source_feature=None, **v) if 'source_feature' not in v else ridge._FeatureStatistic(**v)
                 for v in model.get('feature_statistics', [])]
        # Shared causal preparation is independent of candidate model parameters.
        if self._filled is None:
            self._filled = ridge._causal_forward_fill(self.series, selected, ridge._external_feature_roles(self.series))
        coefs = model.get('coefficients', {})
        coefficients = [model.get('intercept', 0.), *coefs.values()] if coefs else []
        rows, _ = ridge._predict_rows(self.series, 'training_feedback', request.target, request.horizon_hours,
            (sample,), self._filled, stats, coefficients, config, defer_prediction=True,
            plan=plan, index_by_timestamp=visible)
        return rows[0]['baseline'], rows[0]['label_free_context']

    def execute(self, requests, tool_id, parameters):
        self._check_control()
        config = _CONFIGS[tool_id].from_mapping({**_defaults(_CONFIGS[tool_id], self.horizons), **parameters})
        key, fit_config = self._fit_identity(tool_id, config)
        with self._lock:
            if key not in self._fits:
                self._fits[key] = self._load_or_fit(key, fit_config)
                while len(self._fits) > self.CACHE_CAPACITY:
                    self._fits.popitem(last=False)
            self._fits.move_to_end(key)
            fit = self._fits[key]
            output = {}
            for request in requests:
                self._check_control()
                baseline, context = self._origin_input(request, config, fit)
                value = ridge.predict_fitted_exogenous_ridge(target=request.target, horizon_hours=request.horizon_hours,
                    baseline=baseline, label_free_context=context, models=fit['models'], config=config)
                output[request.sample_id] = {'predicted': value, 'metadata': {'model_id': tool_id,
                    'parameters': config.to_dict(), 'fit_digest': digest(fit['models']),
                    'fit_resource_key': key, 'training_partition': 'training_fit'}}
            return output
