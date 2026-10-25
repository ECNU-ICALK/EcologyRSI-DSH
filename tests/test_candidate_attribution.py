"""Whole-grid training attribution, receipt binding, and feedback delivery."""
from copy import deepcopy
from types import SimpleNamespace as NS
import math
import unittest

from ecologyrsi_dsh.core.immutable import freeze_json
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.evaluators.candidate_attribution import (
    POLICY, POLICY_KEY, METRIC_KEY, enabled, summarize_candidate_attribution,
)
from ecologyrsi_dsh.evolution.agent_policy import (
    build_agent_policy, prior_candidate_attribution, rebind_agent_policy,
)
from ecologyrsi_dsh.evolution.context import safe_aggregate_feedback


class CandidateAttributionTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {POLICY_KEY: POLICY, 'dataset_task': {
            'targets': [{'name':t} for t in ('air_temperature','relative_humidity','co2_concentration')],
            'horizons_hours':[1,6,24]}}
        self.identity = {'predictor_id':'registered@1','parameters_digest':'a'*64}

    def evidence(self):
        records, rows = [], []
        for target in self.metadata['dataset_task']['targets']:
            for horizon in (1,6,24):
                for origin, observed, final in ((24,10.,12.),(48,20.,30.)):
                    sample = f'{target["name"]}:{horizon}:{origin}'
                    tool = {'call_id':'call:1','tool_id':'candidate-model','parameters':{},
                        'status':'completed','tool_predicted':observed-2.,'used_as_evidence':True}
                    rows.append({'sample_id':sample,'partition':'training_feedback','target':target['name'],
                        'horizon_hours':horizon,'origin_timestamp':origin,'observed':observed,'predicted':final,
                        'status':'succeeded','agent_prediction':{'tools':[tool]}})
                    records.append({'sample_id':sample,'predicted':final,'attempt_trace':[
                        {'outcome':'rejected','model_evidence':[{**tool,'tool_predicted':900.}]},
                        {'outcome':'accepted','model_evidence':[{**tool,'dsh_tool_event_id':'event:1'}]}]})
        return records, rows

    def summary(self, records=None, rows=None, **kwargs):
        if records is None: records,rows=self.evidence()
        return summarize_candidate_attribution(records,rows,metadata=self.metadata,
            source_phase=kwargs.get('phase','formal_batch'),candidate_identity=self.identity,cohort_digest='b'*64)

    def test_distinct_origins_and_retries_preserve_every_cell(self):
        records,rows=self.evidence()
        expected=self.summary(records,rows)
        actual=self.summary(records+deepcopy(records),rows+deepcopy(rows)*10)
        self.assertEqual(expected,actual)
        self.assertEqual(len(actual['cells']),9)
        for c in actual['cells']:
            self.assertEqual((c['distinct_origins'],c['paired_origins']),(2,2))
            self.assertEqual(c['candidate_raw_mae'],2)
            self.assertEqual(c['candidate_raw_rmse'],2)
            self.assertEqual(c['final_mae'],6)
            self.assertAlmostEqual(c['final_rmse'],math.sqrt(52))
            self.assertEqual(c['adjustment_mae_gain'],-4)
            self.assertEqual(c['adjustment_harmed_origins'],1)
            self.assertEqual(c['paired_24h_blocks'],2)
            self.assertEqual(c['worst_24h_adjustment_mae_gain'],-8)
        self.assertIn('not_causal',actual['interpretation'])
        self.assertFalse(actual['scientific_gain_established'])

    def test_receipt_binding_ignores_abandoned_or_unverified_tools(self):
        records,rows=self.evidence()
        records[0]['attempt_trace'][-1]['model_evidence'][0]['dsh_tool_event_id']=None
        # Same output of a different parameterization cannot pose as candidate.
        rows[1]['agent_prediction']['tools'][0]['tool_id']='registered@1'
        receipt=records[1]['attempt_trace'][-1]['model_evidence'][0]
        receipt['tool_id']='registered@1';receipt['effective_parameters_digest']='z'*64
        c=self.summary(records,rows)['cells'][0]
        self.assertEqual(c['paired_origins'],0)
        self.assertEqual(c['missing_candidate_origins'],2)
        self.assertIsNone(c['candidate_raw_mae'])
        self.assertEqual(c['all_successful_final_mae'],6)
        receipt['effective_parameters_digest']=self.identity['parameters_digest']
        self.assertEqual(self.summary(records,rows)['cells'][0]['paired_origins'],1)

    def test_conflicting_duplicate_origins_are_unknown_not_extra_samples(self):
        records,rows=self.evidence();dup=deepcopy(rows[0]);dup['predicted']=99
        result=self.summary(records,rows+[dup])['cells'][0]
        self.assertEqual(result['distinct_origins'],2)
        self.assertEqual(result['ambiguous_origins'],1)
        self.assertEqual(result['paired_origins'],1)

    def test_failed_and_nontraining_partitions_do_not_supply_label_feedback(self):
        records,rows=self.evidence()
        rows[0]['status']='failed';rows[1]['scoring_fallback']='fallback'
        c=self.summary(records,rows)['cells'][0]
        self.assertEqual(c['successful_origins'],0)
        self.assertEqual(c['paired_origins'],0)
        for partition in ('validation','final_test','training_holdout'):
            for row in rows:row['partition']=partition
            report=self.summary(records,rows)
            self.assertTrue(all(c['distinct_origins']==0 for c in report['cells']))
        for phase in ('holdout','validation','final_test'):
            self.assertIsNone(self.summary(phase=phase))
        # Even malformed/adapted rows cannot bypass the frozen partition.
        original_metadata=self.metadata
        self.metadata={**self.metadata,'evaluation_partition':'validation'}
        self.assertIsNone(self.summary())
        self.metadata=original_metadata
        self.assertIsNone(summarize_candidate_attribution([],[],metadata={},source_phase='formal_batch',candidate_identity=None))
        with self.assertRaisesRegex(ValueError,'unsupported'):
            enabled({POLICY_KEY:'unknown'})

    def delivered(self):
        candidate=NS(candidate_id='candidate:1',generation=0)
        def batch(index,phase='formal_batch',generation=0,cid='candidate:1',summary=None):
            return NS(scope=NS(candidate_id=cid,generation=generation,phase=phase,batch_index=index,
                cohort_digest='b'*64,scope_key=str(index)*64,candidate_revision_id=f'revision:{index}'),
                metrics=freeze_json({'sample_execution':{METRIC_KEY:summary or self.summary()}}))
        state=NS(task_manifest=NS(metadata=self.metadata),formal_batch_evaluations=(
            batch(0),batch(1),batch(2),batch(3,phase='holdout'),batch(3,generation=1),batch(4,cid='other')))
        report=prior_candidate_attribution(state,candidate)
        self.assertEqual(report['source_candidate_revision_id'],'revision:2')
        self.assertEqual(len(report['cells']),9)
        return report,state,candidate

    def test_frozen_report_survives_actual_aggregate_boundary_and_agent_policy(self):
        report,state,candidate=self.delivered()
        # More frequent legacy calls never crowd out any attribution cell.
        frequent=[{'evidence_kind':'raw_tool_output','source_phase':'formal_batch','n':999}]*32
        analysis=safe_aggregate_feedback({'generation':0,'ranking':[
            {'agent_tool_performance':frequent,METRIC_KEY:report}]},name='generation analysis')
        self.assertEqual(analysis['ranking'][0][METRIC_KEY],report)
        policy=build_agent_policy(genome_digest='c'*64,profile={},parameters={},generation=1,previous_analysis=analysis)
        delivered=policy['experience']['candidate_attribution'][0]
        self.assertEqual(len(delivered['cells']),9)
        self.assertEqual(delivered['cells'][0]['adjustment_mae_gain'],-4)
        self.assertEqual(len(policy['experience']['rows']),32)
        rebound=rebind_agent_policy(policy,genome_digest='d'*64,profile={},parameters={})
        self.assertEqual(rebound['experience'],policy['experience'])
        state.task_manifest.metadata={}
        self.assertIsNone(prior_candidate_attribution(state,candidate))
        historical=build_agent_policy(genome_digest='c'*64,profile={},parameters={},generation=1,previous_analysis={'generation':0})
        self.assertNotIn('candidate_attribution',historical['experience'])
        altered=deepcopy(report);altered['source_generation']=1
        altered['summary_digest']=digest({k:v for k,v in altered.items() if k!='summary_digest'})
        with self.assertRaisesRegex(ValueError,'precede'):
            build_agent_policy(genome_digest='c'*64,profile={},parameters={},generation=1,
                previous_analysis={'generation':0,'ranking':[{METRIC_KEY:altered}]})

    def test_durable_actual_adaptive_and_local_editor_boundaries_preserve_grid(self):
        from ecologyrsi_dsh.application.formal_trajectory import _durable_batch_metrics, _local_edit_attribution
        from tests.test_adaptive_tool_feedback import AdaptiveToolFeedbackTests
        summary=self.summary()
        metrics=freeze_json({'sample_execution':{METRIC_KEY:summary},
            'prediction_formula_policy':'evidence-mean-or-model@1',
            'comparison_retrieval_snapshot_digest':'f'*64})
        durable=_durable_batch_metrics(metrics)
        self.assertEqual(durable['prediction_formula_policy'],'evidence-mean-or-model@1')
        self.assertEqual(durable['comparison_retrieval_snapshot_digest'],'f'*64)
        self.assertEqual(_local_edit_attribution(freeze_json(durable)),summary)
        analysis=AdaptiveToolFeedbackTests().analysis(attribution=summary)
        transported=safe_aggregate_feedback(analysis.to_dict(),name='actual adaptive analysis')
        report=transported['ranking'][0][METRIC_KEY]
        self.assertEqual(len(report['cells']),9)
        self.assertEqual(report['sealed_summary_digest'],summary['summary_digest'])
        policy=build_agent_policy(genome_digest='c'*64,profile={},parameters={},generation=1,previous_analysis=transported)
        self.assertEqual(policy['experience']['candidate_attribution'][0],report)
        from ecologyrsi_dsh.evolution.batches import _adaptive_reflection_analysis
        reflection=_adaptive_reflection_analysis(NS(task_manifest=NS(metadata=self.metadata)),analysis)
        safe_reflection=safe_aggregate_feedback(reflection,name='generation reflection')
        self.assertEqual(safe_reflection['candidate_attribution'][0],report)

    def test_repeatability_drift_reaches_policy_once_without_claiming_confidence(self):
        evidence={'paired_origin_count':10,'changed_prediction_count':82,'absolute_score_drift':.03}
        policy=build_agent_policy(genome_digest='c'*64,profile={},parameters={},generation=1,
            previous_analysis={'generation':0,'ranking':[{'same_genome_repeatability':evidence}]*4})
        delivered=policy['experience']['same_genome_repeatability']
        self.assertEqual(delivered['absolute_score_drift'],.03)
        self.assertIn('not a confidence bound',delivered['interpretation'])
        self.assertIn('no new independent observations',delivered['interpretation'])

    def test_retained_revision_attribution_excludes_later_rejected_challenger(self):
        _, state, candidate = self.delivered()
        champion = state.formal_batch_evaluations[1]
        challenger = state.formal_batch_evaluations[2]
        # Both arms are measured on the same latest cohort. The rejected
        # challenger is persisted last, exactly as in the paired workflow.
        champion.scope.batch_index = challenger.scope.batch_index = 2
        state.formal_batch_evaluations = (champion, challenger)
        state.trajectory_for = lambda _cid: NS(final_revision_id='revision:1')
        report = prior_candidate_attribution(state, candidate)
        self.assertEqual(report['source_candidate_revision_id'], 'revision:1')
        self.assertEqual(report['sealed_summary_digest'], self.summary()['summary_digest'])
        self.assertEqual(len(report['cells']), 9)
        self.assertIsNone(prior_candidate_attribution(
            state, candidate, candidate_revision_id='revision:without-training-evidence'))
        self.assertEqual(prior_candidate_attribution(
            state, candidate, candidate_revision_id='revision:2')['source_candidate_revision_id'], 'revision:2')

    def test_historical_misbound_report_cannot_teach_next_policy_as_retained_revision(self):
        report, _, _ = self.delivered()
        row = {METRIC_KEY: report, 'final_revision_id': 'revision:1'}
        original = deepcopy(row)
        policy = build_agent_policy(genome_digest='c'*64, profile={}, parameters={}, generation=1,
            previous_analysis={'generation': 0, 'ranking': [row]})
        self.assertNotIn('candidate_attribution', policy['experience'])
        self.assertEqual(row, original)  # Archived evidence is never rewritten.

    def test_paired_reflection_keeps_one_full_aa_diagnostic_and_bound_attribution(self):
        from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
        from ecologyrsi_dsh.evolution.batches import _adaptive_reflection_analysis
        from ecologyrsi_dsh.evaluators.repeatability import POLICY_KEY as AA_KEY, POLICY as AA_POLICY
        report, _, _ = self.delivered()
        noise = {'paired_origin_count': 10, 'paired_prediction_count': 90,
            'changed_prediction_count': 84, 'absolute_score_drift': .02163476387433745,
            'reference_score': .054304800786523076, 'repeat_score': .032670036912185625,
            'independent_observation_count_multiplier': 1, 'selection_score_replaced': False,
            'interpretation': 'One A/A pair measures realized Agent drift, not a confidence bound or a causal attribution; do not select the better repeat.',
            'cells': [{'target': t, 'horizon_hours': h, 'skill_delta': -.01}
                      for t in ('air_temperature', 'relative_humidity', 'co2_concentration') for h in (1, 6, 24)]}
        ranking = [{METRIC_KEY: report, 'candidate_id': 'candidate:1', 'final_revision_id': 'revision:1', 'score': .02},
                   {METRIC_KEY: report, 'candidate_id': 'candidate:2', 'final_revision_id': 'revision:2', 'score': .01}]
        analysis = NS(generation=0, ranking=ranking, to_dict=lambda: {'generation': 0, 'ranking': deepcopy(ranking)})
        incumbent = NS(scope=NS(holdout_arm=NS(value='incumbent'), candidate_id='incumbent',
                              candidate_revision_id='revision:incumbent'), score=.03)
        comparison = NS(holdout_evaluations=(incumbent,), selected_candidate_id='incumbent',
            selected_revision_id='revision:incumbent', comparison_digest='d'*64,
            gate_results={'selected_arm': 'incumbent', 'delta_to_incumbent': 0})
        state = NS(task_manifest=NS(metadata={**self.metadata, AA_KEY: AA_POLICY,
            'optimization_protocol': 'joint_plugin_epoch@2',
            'optimization_schedule': OptimizationSchedule.default().to_dict()}),
            events=(NS(kind='ScreeningRepeatabilityRecorded', payload={'generation': 0, 'summary': freeze_json(noise)}),),
            comparison_for=lambda _gen: comparison,
            revision=lambda rid: NS(candidate_id='incumbent', revision_id=rid,
                revision_digest='a'*64, genome_digest='b'*64, behavior_digest='c'*64))
        reflection = safe_aggregate_feedback(_adaptive_reflection_analysis(state, analysis), name='reflection')
        self.assertEqual(reflection['same_genome_repeatability'], noise)
        self.assertEqual(len(reflection['same_genome_repeatability']['cells']), 9)
        self.assertNotIn('ranking', reflection)
        self.assertEqual(reflection['candidate_attribution'], [report])
        rows = reflection['adaptive_epoch_evidence']['candidate_results']
        self.assertEqual([r['score'] for r in rows], [.02, .01])
        self.assertTrue(all('same_genome_repeatability' not in r for r in rows))
        # A legacy run does not acquire evidence or need an event collection.
        state.task_manifest.metadata.pop(AA_KEY)
        del state.events
        self.assertNotIn('same_genome_repeatability', _adaptive_reflection_analysis(state, analysis))

    def test_local_editor_attributes_rejected_metrics_to_their_actual_revision(self):
        from ecologyrsi_dsh.application.formal_trajectory import _local_edit_attribution
        from ecologyrsi_dsh.evaluators.candidate_attribution import validated_summary
        summary = self.summary()
        metrics = freeze_json({'sample_execution': {METRIC_KEY: summary}})
        scope = NS(scope_key='e'*64, candidate_revision_id='revision:rejected')
        report = _local_edit_attribution(metrics, scope=scope, current_revision_id='revision:retained')
        self.assertEqual(report['source_candidate_revision_id'], 'revision:rejected')
        self.assertEqual(report['current_candidate_revision_id'], 'revision:retained')
        self.assertEqual(report['revision_relation'], 'different_evaluated_revision')
        self.assertEqual(report['cells'], summary['cells'])
        self.assertEqual(validated_summary(report), report)
        self.assertEqual(_local_edit_attribution(metrics, scope=scope,
            current_revision_id='revision:rejected')['revision_relation'], 'current_candidate_revision')

    def test_formula_policy_is_frozen_rebound_and_rejects_incompatible_directives(self):
        kwargs=dict(genome_digest='c'*64,parameters={},generation=0,previous_analysis=None,
            prediction_formula_policy='evidence-mean-or-model@1')
        p=build_agent_policy(profile={},**kwargs)
        q=rebind_agent_policy(p,genome_digest='d'*64,profile={},parameters={})
        self.assertEqual(q['inference']['prediction_formula_policy'],'evidence-mean-or-model@1')
        for profile in ({'allowed_prediction_methods':['direct']},
                        {'authored_directive':{'blend_rule':'confidence_weighted'}}):
            with self.assertRaises(ValueError):build_agent_policy(profile=profile,**kwargs)
        for rule in ('none','mean'):
            build_agent_policy(profile={'authored_directive':{'blend_rule':rule}},**kwargs)


if __name__=='__main__':unittest.main()
