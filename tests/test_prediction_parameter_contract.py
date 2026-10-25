"""Parameter mistakes observed in run 62a55a62, without its private dataset."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.evaluators.agent_model_tools import AgentModelTools
from ecologyrsi_dsh.evaluators.sample_execution import SamplePredictionRequest
from tests.test_baseline_aligned_ridge import periodic_series


class PredictionParameterContractTests(unittest.TestCase):
    def setUp(self):
        self.bank = AgentModelTools(periodic_series(), targets=('air_temperature',), horizons=(1,))
        self.catalog = {item['tool_id']: item for item in self.bank.catalog()}
        self.recipe = self.catalog['greenhouse-recipe-ridge@1']['parameters']['feature_recipe']['default']
        self.request = SamplePredictionRequest(
            sample_id='one', candidate_id='candidate', dataset_digest=self.bank.series.digest,
            partition='training_feedback', target='air_temperature', unit='degC',
            horizon_hours=1, origin_timestamp=205, target_timestamp=206, baseline=20,
            proposed_prediction=None, minimum=-20, maximum=80, algorithm_id='optional',
            algorithm_version='1', label_free_context={},
        )

    def test_observed_cross_tool_parameters_remain_rejected_before_fitting(self):
        # These five parameter-name patterns account for eight actual failures.
        cases = [
            ((90468, 90593, 93793), 'greenhouse-recipe-ridge@1',
             {'feature_recipe': self.recipe, 'ridge_alpha': .1}, 'unsupported ridge_alpha'),
            ((91150, 93104), 'greenhouse-recipe-ridge@1',
             {'feature_recipe': self.recipe, 'history_steps': 6, 'ridge_alpha': .1},
             'unsupported history_steps, ridge_alpha'),
            ((93375,), 'greenhouse-recipe-ridge@1', {'history_steps': 6},
             'unsupported history_steps'),
            ((91975,), 'greenhouse-targetwise-ridge@1',
             {'history_steps': 6, 'ridge_alpha': .1, 'residual_scale': .5},
             'unsupported residual_scale'),
            ((92241,), 'greenhouse-horizon-targetwise-ridge@1',
             {'history_steps': 6, 'ridge_alpha': .1, 'air_temperature_residual_scale': .5,
              'co2_concentration_residual_scale': .5, 'relative_humidity_residual_scale': .5},
             'unsupported air_temperature_residual_scale'),
        ]
        with patch.object(self.bank, '_load_or_fit') as fit:
            for events, tool, parameters, error in cases:
                with self.subTest(events=events):
                    self.assertTrue(set(parameters) - set(self.catalog[tool]['parameters']))
                    with self.assertRaisesRegex(ValueError, error):
                        self.bank.execute((self.request,), tool, parameters)
            fit.assert_not_called()

    def test_empty_recipe_is_not_the_default_request(self):
        # Actual failure 93657: {} selects defaults; {'feature_recipe': {}} replaces them.
        with patch.object(self.bank, '_load_or_fit') as fit:
            with self.assertRaisesRegex(TypeError, 'feature_recipe.features must be a list'):
                self.bank.execute((self.request,), 'greenhouse-recipe-ridge@1', {'feature_recipe': {}})
            fit.assert_not_called()
        default = self.bank.execute((self.request,), 'greenhouse-recipe-ridge@1', {})['one']
        explicit = self.bank.execute((self.request,), 'greenhouse-recipe-ridge@1',
                                     {'feature_recipe': deepcopy(self.recipe)})['one']
        self.assertEqual(default['predicted'], explicit['predicted'])

    def test_complete_recipe_and_tool_specific_scalar_names_execute(self):
        recipe = deepcopy(self.recipe)
        recipe['model']['alpha'] = .3
        result = self.bank.execute((self.request,), 'greenhouse-recipe-ridge@1',
                                   {'feature_recipe': recipe})['one']
        self.assertEqual(result['metadata']['parameters']['feature_recipe']['model']['alpha'], .3)
        for tool, name in (
            ('greenhouse-targetwise-ridge@1', 'air_temperature_residual_scale'),
            ('greenhouse-horizon-targetwise-ridge@1', 'air_temperature_1h_residual_scale'),
        ):
            with self.subTest(tool=tool):
                self.assertIn(name, self.catalog[tool]['parameters'])
                output = self.bank.execute((self.request,), tool, {name: .3})['one']
                self.assertEqual(output['metadata']['parameters'][name], .3)


if __name__ == '__main__':
    unittest.main()
