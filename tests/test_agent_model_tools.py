from dataclasses import replace
import unittest

from ecologyrsi_dsh.evaluators.agent_model_tools import AgentModelTools
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
        self.assertEqual(len(bank._fits), 4)
        bank.execute((request,), 'greenhouse-exogenous-ridge@1', {'residual_scale': 0})
        bank.execute((request,), 'greenhouse-baseline-aligned-ridge@1', {'residual_scale_1h': .3, 'residual_scale_6h': .7})
        self.assertEqual(len(bank._fits), 4)
        for params in ({'ridge_alpha': -1}, {'history_steps': 20}, {'observed': 100}, {'residual_scale': True}):
            with self.assertRaises((ValueError, TypeError)):
                bank.execute((request,), 'greenhouse-exogenous-ridge@1', params)

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
