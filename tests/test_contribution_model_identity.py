from copy import deepcopy
import json
import unittest

from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.prediction_policy import PREVIOUS_CONTRIBUTION_POLICY
from ecologyrsi_dsh.evaluators.prediction_contribution import compiled_candidate_identity, summarize_contribution
import test_prediction_contribution as contribution_fixture


class ContributionModelIdentityTests(unittest.TestCase):
    def fixture(self):
        parameters = {'history_steps': 6, 'residual_scale_6h': .1}
        metadata = {'evolution_genome_canonical_json': json.dumps({'scientific_program': {
            'predictor_ref': {'id': 'registered-ridge@1'}}})}
        identity = compiled_candidate_identity(metadata, parameters)
        tool = {'tool_id': 'registered-ridge@1', 'status': 'completed', 'call_id': 'call-1',
                'parameters': parameters, 'tool_predicted': 10., 'used_as_evidence': True}
        receipt = {**tool, 'effective_parameters_digest': digest(parameters), 'dsh_tool_event_id': 'event-1'}
        rows = contribution_fixture.ContributionTests().rows()
        rows[0]['agent_prediction']['tools'] = [tool]
        records = [{'sample_id': rows[0]['sample_id'], 'attempt_trace': [
            {'outcome': 'accepted', 'model_evidence': [receipt]}]}]
        return identity, records, rows

    def test_exact_registered_model_receipt_counts_without_mutating_data(self):
        identity, records, rows = self.fixture()
        before = deepcopy((identity, records, rows))
        result = summarize_contribution(records, rows, candidate_identity=identity)
        self.assertEqual(result['cells'][0]['candidate_tool_cells'], 1)
        self.assertEqual(result['cells'][0]['candidate_cited_cells'], 1)
        self.assertEqual(result['candidate_model_identity'], identity)
        self.assertEqual((identity, records, rows), before)
        self.assertFalse(result['scientific_gain_established'])
        rows[0]['observed'] = -1000.
        self.assertEqual(result, summarize_contribution(records, rows, candidate_identity=identity))

    def test_missing_or_mismatched_receipts_are_not_inferred_as_equivalent(self):
        for field, value in [('effective_parameters_digest', '0' * 64), ('dsh_tool_event_id', None),
                             ('call_id', 'another-call'), ('tool_predicted', 99.),
                             ('tool_id', 'another-model@1'), ('status', 'failed')]:
            with self.subTest(field=field):
                identity, records, rows = self.fixture()
                records[0]['attempt_trace'][0]['model_evidence'][0][field] = value
                result = summarize_contribution(records, rows, candidate_identity=identity)
                self.assertEqual(result['cells'][0]['candidate_tool_cells'], 0)
        identity, records, rows = self.fixture()
        records[0]['attempt_trace'][0]['outcome'] = 'critic_revise'
        self.assertEqual(summarize_contribution(records, rows, candidate_identity=identity)
                         ['cells'][0]['candidate_tool_cells'], 0)
        self.assertEqual(summarize_contribution([], rows, candidate_identity=identity)
                         ['cells'][0]['candidate_tool_cells'], 0)

    def test_v2_keeps_exact_old_digest_and_does_not_recognize_registered_aliases(self):
        old = summarize_contribution([], contribution_fixture.ContributionTests().rows(), cohort_digest='a',
                                     policy=PREVIOUS_CONTRIBUTION_POLICY)
        self.assertEqual(old['summary_digest'], '5bb16ca46b851c8035394dbd940647cc696a037096a5aca32ede41988482170f')
        identity, records, rows = self.fixture()
        result = summarize_contribution(records, rows, candidate_identity=identity, policy=PREVIOUS_CONTRIBUTION_POLICY)
        self.assertEqual(result['cells'][0]['candidate_tool_cells'], 0)
        self.assertNotIn('candidate_model_identity', result)
        self.assertIsNone(compiled_candidate_identity(
            {'evolution_genome_canonical_json': 'old unused text'}, {}, policy=PREVIOUS_CONTRIBUTION_POLICY))

    def test_recipe_identity_uses_the_frozen_recipe_not_scalar_overrides(self):
        recipe = {'anchor': 'baseline', 'terms': []}
        metadata = {'evolution_genome_canonical_json': json.dumps({'scientific_program': {
            'predictor_ref': {'id': 'recipe@1'}, 'feature_recipe': recipe}})}
        identity = compiled_candidate_identity(metadata, {'ridge_alpha': .1})
        self.assertEqual(identity, {'predictor_id': 'recipe@1', 'parameters_digest': digest({'feature_recipe': recipe})})


if __name__ == '__main__':
    unittest.main()
