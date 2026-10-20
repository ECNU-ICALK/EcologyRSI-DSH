import unittest
from copy import deepcopy
from dataclasses import replace
from ecologyrsi_dsh.evaluators.prediction_contribution import summarize_contribution, paired_tool_effect
from ecologyrsi_dsh.core.prediction_policy import CONTRIBUTION_POLICY, LEGACY_CONTRIBUTION_POLICY
from ecologyrsi_dsh.evolution.local_edits import validate_local_edit_proposal
from tests import test_local_edits as fixture

class ContributionTests(unittest.TestCase):
    def rows(self):
        return [dict(sample_id='s:1',target='relative_humidity',horizon_hours=6,origin_timestamp=30,
            predicted=11.,baseline=10.,observed=999.,label_free_context={'history_window':[10.,11.]},
            agent_prediction={'confidence':.8,'tools':[dict(tool_id='candidate-model',status='completed',tool_predicted=10.,used_as_evidence=True)]})]
    def test_baseline_only_tool_can_be_cited_without_final_copy_or_scientific_gain(self):
        rows=self.rows(); summary=summarize_contribution([],rows,cohort_digest='a',source_phase='formal_batch');cell=summary['cells'][0]
        self.assertEqual((cell['candidate_tool_cells'],cell['candidate_cited_cells'],cell['model_equals_baseline'],cell['final_equals_model']),(1,1,1,0))
        self.assertEqual(cell['trace_available_cells'],0);self.assertFalse(summary['scientific_gain_established'])
        changed=deepcopy(rows);changed[0]['observed']=-999.
        self.assertEqual(summary,summarize_contribution([],changed,cohort_digest='a',source_phase='formal_batch'))

    def test_v1_summary_retains_the_prechange_digest_and_shape(self):
        summary = summarize_contribution([], self.rows(), cohort_digest='a', policy=LEGACY_CONTRIBUTION_POLICY)
        self.assertEqual(summary['schema_version'], 'ecologyrsi-dsh.prediction-contribution/1')
        self.assertEqual(summary['policy'], 'prediction_contribution@1')
        # Recorded from the pre-fix implementation. This covers the complete
        # summary, including its old full-context input identity semantics.
        self.assertEqual(summary['summary_digest'], '8f587cef0842e459eb93c64062c239f339fc5aa221ce3f8046aec971e00c906a')
        self.assertNotIn('treatment_vector_digest', summary['cells'][0])
        self.assertNotIn('observation_vector_digest', summary['cells'][0])

    def test_v2_separates_treatment_and_rejects_mixed_diagnostic_versions(self):
        rows = self.rows()
        rows[0]['label_free_context']['predictor_state'] = {'ridge_alpha': .1}
        before = summarize_contribution([], rows, cohort_digest='a')
        changed = deepcopy(rows)
        changed[0]['label_free_context']['predictor_state']['ridge_alpha'] = .2
        changed[0]['agent_prediction']['tools'][0]['tool_predicted'] = 10.2
        after = summarize_contribution([], changed, cohort_digest='a')
        self.assertEqual(before['schema_version'], 'ecologyrsi-dsh.prediction-contribution/2')
        a, b = before['cells'][0], after['cells'][0]
        self.assertEqual(a['observation_vector_digest'], b['observation_vector_digest'])
        self.assertEqual(a['input_vector_digest'], a['observation_vector_digest'])
        self.assertNotEqual(a['treatment_vector_digest'], b['treatment_vector_digest'])
        self.assertEqual(paired_tool_effect(before, after)['changed_cells'], ['relative_humidity@6h'])
        legacy = summarize_contribution([], rows, cohort_digest='a', policy=LEGACY_CONTRIBUTION_POLICY)
        self.assertEqual(paired_tool_effect(legacy, before), {
            'status': 'unverified', 'reason': 'same_diagnostic_version_required'})
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            summarize_contribution([], rows, policy='prediction_contribution@99')

    def test_recipe_read_contract_and_missing_observations_remain_unverified(self):
        rows = self.rows()
        rows[0]['label_free_context']['predictor_state'] = {
            'feature_recipe_digest': 'a' * 64, 'required_timestamp_offsets': [0, -1],
            'feature_recipe_anchor': 'baseline', 'cohort_history_hours': 48}
        before = summarize_contribution([], rows, cohort_digest='a')
        for name, value in (('feature_recipe_digest', 'b' * 64),
                            ('required_timestamp_offsets', [0, -24])):
            changed = deepcopy(rows)
            changed[0]['label_free_context']['predictor_state'][name] = value
            self.assertEqual(paired_tool_effect(before, summarize_contribution(
                [], changed, cohort_digest='a'))['status'], 'unverified')
        rows[0]['label_free_context'] = {'predictor_state': {'ridge_alpha': .1}}
        self.assertIsNone(summarize_contribution([], rows)['cells'][0]['observation_vector_digest'])
    def test_paired_raw_output_requires_complete_identical_inputs(self):
        rows=self.rows();before=summarize_contribution([],rows,cohort_digest='a')
        changed=deepcopy(rows);changed[0]['agent_prediction']['tools'][0]['tool_predicted']=10.2
        after=summarize_contribution([],changed,cohort_digest='a')
        self.assertEqual(paired_tool_effect(before,after)['changed_cells'],['relative_humidity@6h'])
        for field,value in (('cohort_digest','b'),('cohort_digest',None)):
            self.assertEqual(paired_tool_effect(before,{**after,field:value})['status'],'unverified')
        changed[0].pop('label_free_context');self.assertEqual(paired_tool_effect(before,summarize_contribution([],changed,cohort_digest='a'))['status'],'unverified')
    def test_replayed_contribution_is_json_safe_for_local_editor(self):
        import json
        from ecologyrsi_dsh.core.immutable import freeze_json
        from ecologyrsi_dsh.application.formal_trajectory import _local_edit_contribution
        summary = summarize_contribution([], self.rows(), cohort_digest='a')
        metrics = freeze_json({'sample_execution': {'prediction_contribution': summary}})
        detached = _local_edit_contribution(metrics)
        self.assertEqual(json.loads(json.dumps(detached)), summary)
        detached['cells'][0]['cells'] = 999
        self.assertEqual(metrics['sample_execution']['prediction_contribution']['cells'][0]['cells'], 1)

    def test_durable_metrics_preserve_diagnostics_for_editor_and_paired_replay(self):
        import json
        from ecologyrsi_dsh.core.immutable import freeze_json
        from ecologyrsi_dsh.application.formal_trajectory import (
            _durable_batch_metrics, _local_edit_contribution, _local_edit_evidence_metrics,
        )
        from ecologyrsi_dsh.evolution.diversity import summarize_behavior

        rows = self.rows()
        changed = deepcopy(rows)
        changed[0]['agent_prediction']['tools'][0]['tool_predicted'] = 10.2
        summaries = []
        for source_rows in (rows, changed):
            contribution = summarize_contribution([], source_rows, cohort_digest='a', source_phase='formal_batch')
            behavior = summarize_behavior([], source_rows)
            metrics = freeze_json({'objective_score': .25, 'scientific_pass': False,
                'sample_execution': {'prediction_contribution': contribution,
                    'observed_behavior': behavior, 'batch_plan': {'private': 'omitted'}}})
            durable = _durable_batch_metrics(metrics)
            replayed = freeze_json(json.loads(json.dumps(durable)))
            self.assertEqual(_local_edit_contribution(replayed), contribution)
            self.assertEqual(json.loads(json.dumps(durable['sample_execution']['observed_behavior'])), behavior)
            self.assertEqual(durable['objective_score'], .25)
            self.assertFalse(durable['scientific_pass'])
            self.assertNotIn('batch_plan', durable['sample_execution'])
            # Full durable diagnostics do not expand the editor's bounded metrics.
            compact = _local_edit_evidence_metrics(replayed)['sample_execution']
            self.assertNotIn('prediction_contribution', compact)
            self.assertNotIn('observed_behavior', compact)
            summaries.append(_local_edit_contribution(replayed))
            durable['sample_execution']['prediction_contribution']['cells'][0]['cells'] = 999
            self.assertEqual(metrics['sample_execution']['prediction_contribution']['cells'][0]['cells'], 1)
        self.assertEqual(paired_tool_effect(*summaries)['changed_cells'], ['relative_humidity@6h'])
        historical = _durable_batch_metrics({'sample_execution': {'coverage': 1.}})
        self.assertEqual(historical['sample_execution'], {'coverage': 1.})

    def test_missing_tools_and_accepted_branch_trace(self):
        rows=self.rows();rows[0]['agent_prediction']['tools']=[]
        records=[{'sample_id':'s:1','attempt_trace':[{'outcome':'failed','skill_evidence':[{}]}, {'outcome':'accepted','critic_decisions':[{'role':'remote_critic_agent'}],'skill_evidence':[]}]}]
        cell=summarize_contribution(records,rows)['cells'][0]
        self.assertEqual(cell['critic_cells'],1);self.assertEqual(cell['skill_cells'],0)
        self.assertIsNone(cell['model_baseline_mean_absolute_delta']);self.assertEqual(cell['candidate_tool_cells'],0)
        threshold=summarize_contribution(records,rows,review_policy={'min_planner_confidence':.55})['cells'][0]
        self.assertEqual(threshold['confidence_below_threshold_cells'],0)
        self.assertEqual(threshold['confidence_observed_cells'],1)

class PredictorContributionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_baseline_aligned_ridge import periodic_series, run
        from ecologyrsi_dsh.evaluators.greenhouse_prediction import (
            BaselineAlignedRidgeConfig, BaselineAlignedCellwiseRidgeConfig,
        )
        cls.series = periodic_series()
        cls.config = BaselineAlignedCellwiseRidgeConfig.from_v1(
            BaselineAlignedRidgeConfig(6, .1, .2).to_dict())
        cls.before = cls._tool_rows(run(cls.series, config=cls.config)['prediction_rows'])
        cls.after = cls._tool_rows(run(cls.series, config=replace(
            cls.config, relative_humidity_6h_residual_scale=.1))['prediction_rows'])
        cls.zero = cls._tool_rows(run(cls.series, config=replace(
            cls.config, relative_humidity_6h_residual_scale=0.))['prediction_rows'])

    @staticmethod
    def _tool_rows(rows):
        origins = sorted({r['origin_timestamp'] for r in rows if r['partition'] == 'training_feedback'})[:5]
        selected = deepcopy([r for r in rows if r['partition'] == 'training_feedback' and r['origin_timestamp'] in origins])
        for row in selected:
            row['sample_id'] = f"{row['target']}:{row['horizon_hours']}:{row['origin_timestamp']}"
            row['agent_prediction'] = {'tools': [{'tool_id': 'candidate-model',
                'status': 'completed', 'tool_predicted': row['predicted'], 'used_as_evidence': True}]}
        return selected

    def test_real_cellwise_edit_is_measured_without_changing_scientific_rows(self):
        untouched = deepcopy((self.before, self.after))
        summaries = [summarize_contribution([], rows, cohort_digest='frozen-cohort')
                     for rows in (self.before, self.after)]
        self.assertEqual(len(self.before), 45)
        self.assertEqual(paired_tool_effect(*summaries), {
            'status': 'measured', 'changed_cells': ['relative_humidity@6h'], 'compared_cells': 9,
            'scientific_gain_established': False, 'cohort_digest': 'frozen-cohort'})
        for a, b in zip(*[s['cells'] for s in summaries]):
            self.assertEqual(a['observation_vector_digest'], b['observation_vector_digest'])
            self.assertNotEqual(a['treatment_vector_digest'], b['treatment_vector_digest'])
        self.assertEqual((self.before, self.after), untouched)
        legacy = [summarize_contribution([], rows, cohort_digest='frozen-cohort',
                   policy=LEGACY_CONTRIBUTION_POLICY) for rows in (self.before, self.after)]
        self.assertEqual(paired_tool_effect(*legacy), {
            'status': 'unverified', 'reason': 'complete_matching_inputs_required'})

    def test_real_observation_or_read_set_drift_is_not_a_controlled_pair(self):
        before = summarize_contribution([], self.before, cohort_digest='frozen-cohort')
        for field in ('history_window', 'feature_snapshot', 'causal_provenance'):
            changed = deepcopy(self.after)
            context = changed[0]['label_free_context']
            if field == 'history_window':
                context[field][0] += 1
            elif field == 'feature_snapshot':
                context[field][0]['value'] += 1
            else:
                context[field]['history_timestamps'][0] -= 1
            result = paired_tool_effect(before, summarize_contribution([], changed, cohort_digest='frozen-cohort'))
            self.assertEqual(result, {'status': 'unverified', 'reason': 'complete_matching_inputs_required'}, field)
        # The zero-residual branch supplies no prepared residual features.
        # Conservatively report missing common inputs, not a fabricated effect.
        self.assertEqual(paired_tool_effect(before, summarize_contribution(
            [], self.zero, cohort_digest='frozen-cohort'))['status'], 'unverified')

    def test_pipeline_uses_frozen_diagnostic_version_with_identical_scores(self):
        from ecologyrsi_dsh.core.models import Candidate, Proposal, TaskManifest
        from ecologyrsi_dsh.core.prediction_policy import CELLWISE_EVALUATOR_ID
        from ecologyrsi_dsh.evaluators.greenhouse_prediction import BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID
        from tests.test_evaluation import _native_registry, _scope_cohort, _profile_metadata
        from tests.native_fixtures import native_manifest
        registry = _native_registry(self.series)
        proposal = Proposal(proposal_id='p:diagnostics', run_id='r:diagnostics', generation=0,
                            title='diagnostic version isolation', changes=self.config.to_dict(), metadata=_profile_metadata())
        candidate = Candidate(candidate_id='c:diagnostics', run_id=proposal.run_id,
                              proposal_id=proposal.proposal_id, generation=0)
        bundles=[]
        for policy, version in ((LEGACY_CONTRIBUTION_POLICY, 1), (CONTRIBUTION_POLICY, 2)):
            task = native_manifest(metadata={
                    'episode_id': self.series.episode_id, 'evaluator_id': CELLWISE_EVALUATOR_ID,
                    'prediction_model_id': BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID,
                    'dataset_digest': self.series.digest,
                    'split_manifest_digest': self.series.split_manifest_digest_sha256,
                    'behavior_diagnostics_policy': policy, 'data_protocol_digest': 'f'*64,
                    'sample_remote_critic_policy': {'version': 'uncertain_or_failure@1', 'min_planner_confidence': .5}})
            bundle = registry.evaluate_scientific(task, candidate, proposal, **_scope_cohort(self.series, candidate))
            summary = bundle.evaluation.metrics['sample_execution']['prediction_contribution']
            self.assertEqual(summary['policy'], policy)
            self.assertEqual(summary['schema_version'], f'ecologyrsi-dsh.prediction-contribution/{version}')
            bundles.append(bundle)
        self.assertEqual(bundles[0].evaluation.score, bundles[1].evaluation.score)
        self.assertEqual(bundles[0].evaluation.passed, bundles[1].evaluation.passed)
        self.assertEqual(bundles[0].artifact.learned_parameters, bundles[1].artifact.learned_parameters)

class TargetedLocalEditTests(unittest.TestCase):
    def test_single_cell_edit_and_old_policy_unchanged(self):
        parent=fixture._parent();base=fixture._context(parent)
        name='co2_concentration_24h_residual_scale'
        context=replace(base,search_policy='cellwise_training_evidence@1',allowed_mutation_targets={'scientific_parameter':(name,'ridge_alpha')},parameter_schemas={name:{'minimum':0,'maximum':1},'ridge_alpha':{'minimum':.0001,'maximum':1}})
        proposal={'schema_version':'ecology-local-edit@1','decision':'mutate','operations':[{'op':'set_bounded_parameter','name':name,'value':.1}], 'evidence_refs':['metric:co2:24'],'expected_effect_cells':['co2_concentration@24h'],'risk_cells':[]}
        self.assertEqual(len(validate_local_edit_proposal(proposal,context).operations),1)
        with self.assertRaisesRegex(ValueError,'one target-horizon'):
            validate_local_edit_proposal({**proposal,'operations':[{'op':'set_bounded_parameter','name':'ridge_alpha','value':.2}]},context)
        mixed={**proposal,'operations':[*proposal['operations'],{'op':'set_bounded_parameter','name':'ridge_alpha','value':.2}]}
        with self.assertRaisesRegex(ValueError,'exactly one'):validate_local_edit_proposal(mixed,context)
        self.assertEqual(len(validate_local_edit_proposal(mixed,replace(context,search_policy=None)).operations),2)
        self.assertNotIn('search_policy',base.to_dict())


if __name__=='__main__':unittest.main()
