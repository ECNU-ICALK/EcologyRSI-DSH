from dataclasses import replace
import unittest

from ecologyrsi_dsh.evaluators.agent_model_tools import AgentModelTools
from ecologyrsi_dsh.evaluators.epoch_cohorts import _eligible_origins
from ecologyrsi_dsh.evaluators.feature_recipe import (
    compile_feature_plan, recipe_grammar, servable_recipe_grammar)
from ecologyrsi_dsh.evaluators.greenhouse_prediction import (
    COHORT_HISTORY_HOURS, MAX_EXOGENOUS_RIDGE_HISTORY_STEPS, SEED_RECIPE_HISTORY_HOURS)
from ecologyrsi_dsh.evaluators.sample_execution import SamplePredictionRequest
from tests.test_baseline_aligned_ridge import periodic_series


class AgentModelToolsTests(unittest.TestCase):
    def request(self, series):
        return SamplePredictionRequest(sample_id='one', candidate_id='candidate', dataset_digest=series.digest,
            partition='training_feedback', target='air_temperature', unit='degC', horizon_hours=1,
            origin_timestamp=205, target_timestamp=206, baseline=20, proposed_prediction=None,
            minimum=-20, maximum=80, algorithm_id='optional', algorithm_version='1', label_free_context={})

    def test_models_are_optional_lazy_and_scale_changes_reuse_fit(self):
        series = periodic_series()
        bank = AgentModelTools(series, targets=('air_temperature','relative_humidity','co2_concentration'), horizons=(1,6,24))
        self.assertEqual(len(bank._fits), 0)
        request = self.request(series)
        for item in bank.catalog():
            output = bank.execute((request,), item['tool_id'], {})['one']
            self.assertIsInstance(output['predicted'], float)
            self.assertNotIn('observed', str(output))
            self.assertEqual(output['metadata']['training_partition'], 'training_fit')
        self.assertEqual(len(bank._fits), len(bank.catalog()))
        bank.execute((request,), 'greenhouse-exogenous-ridge@1', {'residual_scale': 0})
        bank.execute((request,), 'greenhouse-baseline-aligned-ridge@1', {'residual_scale_1h': .3, 'residual_scale_6h': .7})
        self.assertEqual(len(bank._fits), len(bank.catalog()))
        for params in ({'ridge_alpha': -1}, {'history_steps': 20}, {'observed': 100}, {'residual_scale': True}):
            with self.assertRaises((ValueError, TypeError)):
                bank.execute((request,), 'greenhouse-exogenous-ridge@1', params)

    def test_recipe_residual_scale_changes_reuse_the_same_fit(self):
        """Residual scaling is applied after the fit, so it must not refit."""
        series = periodic_series()
        bank = AgentModelTools(series, targets=('air_temperature',), horizons=(1,))
        base = bank.catalog()
        recipe = next(item for item in base
                      if item['tool_id'] == 'greenhouse-recipe-ridge@1'
                      )['parameters']['feature_recipe']['default']
        bank.execute((self.request(series),), 'greenhouse-recipe-ridge@1',
                     {'feature_recipe': recipe})
        self.assertEqual(len(bank._fits), 1)
        rescaled = {**recipe, 'per_horizon': {'1': {'residual_scale': 0.9}}}
        bank.execute((self.request(series),), 'greenhouse-recipe-ridge@1',
                     {'feature_recipe': rescaled})
        self.assertEqual(len(bank._fits), 1)
        # A structural change is a different model and must refit.
        restructured = {**recipe, 'features': [
            *recipe['features'], {'op': 'rolling_std', 'w': 6}]}
        bank.execute((self.request(series),), 'greenhouse-recipe-ridge@1',
                     {'feature_recipe': restructured})
        self.assertEqual(len(bank._fits), 2)

    def test_recipe_grammar_is_advertised_instead_of_scalar_bounds(self):
        series = periodic_series()
        bank = AgentModelTools(series, targets=('air_temperature',), horizons=(1, 6, 24))
        entry = next(item for item in bank.catalog()
                     if item['tool_id'] == 'greenhouse-recipe-ridge@1')
        self.assertEqual(list(entry['parameters']), ['feature_recipe'])
        grammar = entry['parameters']['feature_recipe']['grammar']
        self.assertIn('seasonal_reference', grammar['allowed_ops'])
        self.assertEqual(grammar['allowed_ops']['seasonal_reference']['parameters'],
                         ['period'])
        # Bounds live in a flat sibling map so the grammar stays inside the
        # sample-contract depth fence when it travels in plan.tools[*].
        self.assertEqual(grammar['op_parameters']['seasonal_reference.period'],
                         {'kind': 'integer', 'choices': [24, 168]})
        self.assertEqual(grammar['numeric_bounds_enforced_by'], 'host')
        seed = entry['parameters']['feature_recipe']['default']
        # The reachability guarantee: the seed already carries the reads the
        # selected seasonal baseline uses, so no trust-region step is needed.
        self.assertIn({'op': 'seasonal_reference', 'period': 24}, seed['features'])
        self.assertIn({'op': 'target_lag', 'k': 24}, seed['features'])
        self.assertEqual(sorted(seed['per_horizon']), ['1', '24', '6'])

    def test_a_recipe_origin_near_the_partition_head_has_no_causal_history(self):
        """The failure the advertised depth used to hide, pinned at one origin.

        The seed recipe reaches 24 hours back, so every origin in the first day
        of ``training_feedback`` is unservable no matter how good the fit is.
        Origin selection is a deterministic rolling window, so these origins are
        *always planned* -- in production this raised once per origin instead of
        once at the call, which is what made the run look like a tool failure.
        """
        series = periodic_series()
        head = self.request(series)
        # Partition offset 12: past the 12-hour scalar depth, short of the 24
        # the recipe needs. The default fixture origin (205, offset 37) clears
        # both, which is why no test used to reach this branch.
        head = replace(head, origin_timestamp=180, target_timestamp=181)
        bank = AgentModelTools(series, targets=('air_temperature',), horizons=(1,))
        recipe = next(item for item in bank.catalog()
                      if item['tool_id'] == 'greenhouse-recipe-ridge@1'
                      )['parameters']['feature_recipe']['default']
        with self.assertRaisesRegex(ValueError, 'tool requires complete causal history'):
            bank.execute((head,), 'greenhouse-recipe-ridge@1', {'feature_recipe': recipe})
        # The scalar tools only read ``history_steps`` lags, so the same origin
        # is fine for them: the defect was the recipe's depth, not the origin.
        self.assertIsInstance(
            bank.execute((head,), 'greenhouse-exogenous-ridge@1', {})['one']['predicted'], float)

    def test_the_recipe_tool_is_withheld_below_the_depth_the_cohort_serves(self):
        """A run only advertises what its frozen origin set can serve.

        Compiling against the 48-hour package ceiling while the cohort
        guaranteed the run's alignment meant a 12-hour run offered the Agent a
        seed recipe it accepted and then could not execute at any origin inside
        the first day. Deciding it once here costs the Agent no call.
        """
        series = periodic_series()
        shallow = AgentModelTools(series, targets=('air_temperature',), horizons=(1,),
                                  origin_history_alignment=MAX_EXOGENOUS_RIDGE_HISTORY_STEPS)
        self.assertNotIn('greenhouse-recipe-ridge@1',
                         [item['tool_id'] for item in shallow.catalog()])
        # Withholding one tool must not withhold the others.
        self.assertEqual(len(shallow.catalog()) + 1,
                         len(AgentModelTools(series, targets=('air_temperature',),
                                             horizons=(1,)).catalog()))
        # One short of the threshold is still withheld: the seed reaches 24 hours
        # back and an alignment counts the origin as one of its observations, so
        # 24 serves only 23. Treating the two units as one advertised a seed the
        # cohort could not serve at its own head-most planned origin.
        self.assertNotIn('greenhouse-recipe-ridge@1', [
            item['tool_id'] for item in
            AgentModelTools(series, targets=('air_temperature',), horizons=(1,),
                            origin_history_alignment=SEED_RECIPE_HISTORY_HOURS - 1).catalog()])
        deep = AgentModelTools(series, targets=('air_temperature',), horizons=(1,),
                               origin_history_alignment=SEED_RECIPE_HISTORY_HOURS)
        entry = next(item for item in deep.catalog()
                     if item['tool_id'] == 'greenhouse-recipe-ridge@1')
        # And what it does advertise is clamped to that same depth, so the Agent
        # cannot author a deeper recipe out of the published grammar either.
        reach = SEED_RECIPE_HISTORY_HOURS - 1
        grammar = entry['parameters']['feature_recipe']['grammar']
        self.assertEqual(grammar['servable_history_hours'], reach)
        self.assertEqual(grammar['max_lag_hours'], reach)
        self.assertEqual(grammar['op_parameters']['target_lag.k']['maximum'], reach)
        self.assertEqual(grammar['op_parameters']['seasonal_reference.period']['choices'], [24])
        # A depth that admits no seasonal period at all withholds the whole
        # primitive rather than publishing an empty range to discover by call.
        narrow = servable_recipe_grammar(12)
        self.assertNotIn('seasonal_reference', narrow['allowed_ops'])
        self.assertNotIn('seasonal_reference.period', narrow['op_parameters'])
        self.assertEqual(narrow['op_parameters']['target_lag.k']['maximum'], 11)
        # No alignment means no clamp: the unbounded grammar is unchanged.
        self.assertEqual(servable_recipe_grammar(None), recipe_grammar())

    def test_every_origin_the_cohort_plans_can_serve_the_recipe_it_advertises(self):
        """The planner and the tool must count history in the same unit.

        ``_eligible_origins`` guarantees lags ``range(alignment)`` -- the origin
        counts as one observation -- while a compiled plan's
        ``max_history_hours`` is hours *reached back*. Reading the alignment as a
        reach let the seed recipe pass ``_servable`` at 24 and then fail at
        exactly the one origin the planner puts at the partition head, which is
        the shape that turns one cell into a non-retryable run failure. Assert
        the whole planned cohort, not a sampled origin: an off-by-one is only
        ever visible at the head.
        """
        series = periodic_series()
        for alignment in (SEED_RECIPE_HISTORY_HOURS, COHORT_HISTORY_HOURS):
            bank = AgentModelTools(series, targets=('air_temperature',), horizons=(1,),
                                   origin_history_alignment=alignment)
            recipe = next(item for item in bank.catalog()
                          if item['tool_id'] == 'greenhouse-recipe-ridge@1'
                          )['parameters']['feature_recipe']['default']
            planned, _gaps = _eligible_origins(series, horizons=(1,), history_steps=alignment)
            self.assertTrue(planned)
            for origin in planned:
                request = replace(self.request(series),
                                  origin_timestamp=origin.origin_timestamp,
                                  target_timestamp=origin.origin_timestamp + 1)
                served = bank.execute((request,), 'greenhouse-recipe-ridge@1',
                                      {'feature_recipe': recipe})['one']
                self.assertIsInstance(served['predicted'], float)

    def test_a_recipe_at_the_advertised_reach_compiles_and_one_hour_deeper_does_not(self):
        """The bound itself, at the exact hour where it used to be wrong."""
        series = periodic_series()
        grammar = servable_recipe_grammar(COHORT_HISTORY_HOURS)
        reach = grammar['servable_history_hours']
        self.assertEqual(reach, COHORT_HISTORY_HOURS - 1)
        for lag, servable in ((reach, True), (reach + 1, False)):
            recipe = {'schema_version': 'ecologyrsi-dsh.feature-recipe/1',
                      'features': [{'op': 'target_lag', 'k': lag}],
                      'model': {'kind': 'ridge', 'alpha': .1, 'anchor': 'fit_selected_baseline'},
                      'per_horizon': {'1': {'residual_scale': .5}}}
            if servable:
                plan = compile_feature_plan(recipe, series=series, target='air_temperature',
                                            horizon_hours=1,
                                            origin_history_alignment=COHORT_HISTORY_HOURS)
                self.assertEqual(plan.max_history_hours, reach)
            else:
                with self.assertRaisesRegex(ValueError, f'guarantees {reach}h'):
                    compile_feature_plan(recipe, series=series, target='air_temperature',
                                         horizon_hours=1,
                                         origin_history_alignment=COHORT_HISTORY_HOURS)

    def test_changing_future_labels_cannot_change_tool_prediction(self):
        series = periodic_series()
        future = {name: tuple(value + 1000 if index > 205 and value is not None else value
                              for index, value in enumerate(values)) for name, values in series.values.items()}
        changed = replace(series, values=future)
        for tool_id in ('greenhouse-exogenous-ridge@1', 'greenhouse-baseline-aligned-ridge@1'):
            outputs = []
            for source in (series, changed):
                bank = AgentModelTools(source, targets=('air_temperature',), horizons=(1,))
                outputs.append(bank.execute((self.request(source),), tool_id, {})['one']['predicted'])
            self.assertEqual(*outputs)

    def test_zero_scale_reuses_fitted_model_without_requiring_residual_features(self):
        from tempfile import TemporaryDirectory
        from ecologyrsi_dsh.evaluators.greenhouse_prediction import BaselineAlignedRidgeConfig
        series = periodic_series()
        request = self.request(series)
        with TemporaryDirectory() as directory:
            bank = AgentModelTools(series, targets=('air_temperature',), horizons=(1,), cache_dir=directory)
            tool = 'greenhouse-baseline-aligned-ridge@1'
            fitted = bank.execute((request,), tool, {})['one']
            zero = bank.execute((request,), tool, {'residual_scale_1h': 0})['one']
            config = BaselineAlignedRidgeConfig.from_mapping(zero['metadata']['parameters'])
            expected, _ = bank._origin_input(request, config, next(iter(bank._fits.values())))
            self.assertEqual(zero['predicted'], expected)
            self.assertEqual(len(bank._fits), 1)
            self.assertEqual(bank.execute((request,), tool, {})['one'], fitted)
            restored = AgentModelTools(series, targets=('air_temperature',), horizons=(1,), cache_dir=directory)
            self.assertEqual(restored.execute((request,), tool, {'residual_scale_1h': 0})['one'], zero)

    def test_an_empty_call_executes_the_candidate_genome_not_the_package_default(self):
        """The evolved parameters must reach the number the Agent submits.

        A ``scientific_parameter`` edit only changes this candidate's config. If an
        empty ``parameters`` object selected the package defaults instead, every
        such edit would be invisible to the score no matter how the Agent behaved.
        """
        from ecologyrsi_dsh.evaluators.greenhouse_prediction import (
            ExogenousRidgeConfig, fit_predict_exogenous_ridge)
        series = periodic_series()
        request = self.request(series)
        evolved = ExogenousRidgeConfig(9, .4, .25)
        fit = fit_predict_exogenous_ridge(series, targets=('air_temperature',), horizons=(1,), config=evolved)
        tool = 'greenhouse-exogenous-ridge@1'
        bank = AgentModelTools(series, targets=('air_temperature',), horizons=(1,),
                               default_fit=fit, default_config=evolved)
        entry = next(item for item in bank.catalog() if item['tool_id'] == tool)
        self.assertEqual({name: spec['default'] for name, spec in entry['parameters'].items()},
                         evolved.to_dict())
        self.assertIn("candidate's own evolved model", entry['purpose'])
        # The other tools keep advertising the package defaults.
        other = next(item for item in bank.catalog() if item['tool_id'] == 'greenhouse-targetwise-ridge@1')
        self.assertNotIn('evolved model', other['purpose'])

        # Selecting the candidate default costs no additional fit: it is exactly
        # the fit the batch backend already computed and pre-seeded.
        self.assertEqual(len(bank._fits), 1)
        empty = bank.execute((request,), tool, {})['one']
        self.assertEqual(len(bank._fits), 1)
        self.assertEqual(empty['metadata']['parameters'], evolved.to_dict())
        expected = next(row['predicted'] for row in fit['prediction_rows']
                        if row['origin_timestamp'] == 205 and row['partition'] == 'training_feedback')
        self.assertAlmostEqual(empty['predicted'], expected)
        # An explicit parameter still overrides the candidate default per field.
        override = bank.execute((request,), tool, {'ridge_alpha': .8})['one']
        self.assertEqual(override['metadata']['parameters'],
                         {**evolved.to_dict(), 'ridge_alpha': .8})
        # Without a genome the bank falls back to the package defaults.
        plain = AgentModelTools(series, targets=('air_temperature',), horizons=(1,))
        self.assertEqual(plain.execute((request,), tool, {})['one']['metadata']['parameters'],
                         ExogenousRidgeConfig.from_mapping(
                             {'history_steps': 6, 'ridge_alpha': .1, 'residual_scale': .5}).to_dict())
