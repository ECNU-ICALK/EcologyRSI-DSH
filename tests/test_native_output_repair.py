"""Native planner truncation consumes the existing origin attempt allowance."""
import unittest
from dataclasses import replace

from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError
from ecologyrsi_dsh.core.sample_results import build_sample_results
from ecologyrsi_dsh.evaluators.dsh_sample_adapter import DshSampleCollaborationAdapter
from ecologyrsi_dsh.evaluators.sample_execution import CollaborativeSampleExecutor, SampleExecutionPolicy
from tests.test_dsh_sample_execution import (
    _SampleRuntime, _request, _constant_forecast_bundle, _fake_agent_prediction_binder,
)


class TruncatedPlanner(_SampleRuntime):
    def __init__(self, failures=1, code='structured_child_output_budget_exhausted'):
        super().__init__()
        self.failures, self.code = failures, code

    def run_stage(self, request):
        if request['stage'] == 'sample.plan' and self.failures:
            self.requests.append(request)
            self.failures -= 1
            raise DshNativeRuntimeUnavailableError(
                'No structured prediction was submitted.', error_code=self.code,
                status_code=422, failure_domain='execution_contract',
            )
        return super().run_stage(request)


def execute(runtime, max_attempts=3, publications=None, *, origins=(10,), checkpoint_rows=()):
    adapter = DshSampleCollaborationAdapter(
        run_id='run-output-repair', runtime_provider=lambda: runtime,
        revision_provider=lambda _: {'run_state_revision': 7, 'ledger_expected_revision': 11},
        identity_digests={'genome_digest': 'a'*64, 'compiled_behavior_digest': 'b'*64,
                          'phenotype_instance_digest': 'c'*64},
        strategy_model_id='dsh/strategy', review_model_id='dsh/review',
        forecast_bundle_tool=_constant_forecast_bundle(21.5),
        prediction_tool_binder=_fake_agent_prediction_binder,
        remote_critic_policy={'version': 'uncertain_or_failure@1', 'min_planner_confidence': .5},
        sample_reflection_policy='candidate_aggregate_post_score@1',
        sample_concurrency=1,
    )
    rows=[]
    for origin in origins:
        for target in ('air_temperature', 'relative_humidity', 'co2_concentration'):
            for horizon in (1, 6, 24):
                request=replace(_request(f'{target}-{horizon}-{origin}'), target=target, horizon_hours=horizon,
                                origin_timestamp=origin, target_timestamp=origin+horizon)
                context = dict(request.label_free_context)
                context['causal_provenance'] = {**context['causal_provenance'],
                    'origin_cutoff_timestamp': origin, 'latest_context_timestamp': origin,
                    'history_timestamps': [origin-1, origin]}
                rows.append({**request.to_dict(), 'label_free_context': context, 'observed': 21.0})
    return CollaborativeSampleExecutor(adapter).execute(
        rows, context={'run_id': 'run-output-repair', 'candidate_id': 'candidate-1',
                       'dataset_digest': 'd'*64, 'partition': 'training_feedback',
                       'algorithm_id': 'registered-predictor', 'algorithm_version': '1'},
        target_bounds={t: {'minimum': -20., 'maximum': 80.} for t in
                       ('air_temperature', 'relative_humidity', 'co2_concentration')},
        algorithm_id='registered-predictor', algorithm_version='1',
        policy=SampleExecutionPolicy(max_attempts=max_attempts),
        result_callback=(lambda rows: publications.append(rows)) if publications is not None else None,
        checkpoint_callback=lambda _: {'rows': tuple(checkpoint_rows)},
    )


class NativeOutputRepairTests(unittest.TestCase):
    def test_output_fault_resume_executes_only_missing_complete_origin(self):
        class FailedSecondOrigin(_SampleRuntime):
            planners = 0
            def run_stage(self, request):
                if request['stage'] == 'sample.plan':
                    self.planners += 1
                    if self.planners == 2:
                        self.requests.append(request)
                        raise DshNativeRuntimeUnavailableError(
                            error_code='structured_output_attempt_limit_exceeded',
                            failure_domain='execution_contract', status_code=422)
                return super().run_stage(request)
        runtime = FailedSecondOrigin(); publications = []
        with self.assertRaises(DshNativeRuntimeUnavailableError):
            execute(runtime, origins=(10, 20), publications=publications)
        self.assertEqual([len(rows) for rows in publications], [9])
        persisted = build_sample_results('candidate-1', publications[0])
        self.assertTrue(all(row['sample_agent_chain']['complete'] for row in persisted))
        resumed_runtime = _SampleRuntime(); resumed_publications = []
        resumed = execute(resumed_runtime, origins=(10, 20), checkpoint_rows=persisted,
                          publications=resumed_publications)
        self.assertEqual(len(resumed_runtime.requests), 1)
        self.assertEqual(resumed.summary['checkpoint_resumed_examples'], 9)
        self.assertEqual(resumed.summary['complete_agent_chains'], 18)
        self.assertEqual(len(resumed.scoring_rows), 18)
        self.assertEqual([len(rows) for rows in resumed_publications], [9])
        old_ids = {row['sample_id'] for row in persisted}
        new_ids = {row['sample_id'] for row in resumed_publications[0]}
        self.assertFalse(old_ids & new_ids)
        self.assertTrue(resumed.summary['strict_agent_chain_pass'])

    def test_one_repair_changes_identity_and_feedback_without_changing_limits(self):
        runtime=TruncatedPlanner(); publications=[]
        batch=execute(runtime, max_attempts=2, publications=publications)
        self.assertEqual(len(runtime.requests), 2)
        first, second=runtime.requests
        self.assertNotEqual(first['idempotency_key'], second['idempotency_key'])
        self.assertNotEqual(first['stage_attempt'], second['stage_attempt'])
        self.assertEqual(first['request']['max_tokens'], second['request']['max_tokens'])
        before, after=[r['request']['context'] for r in runtime.requests]
        self.assertEqual([s['sample_id'] for s in before['samples']], [s['sample_id'] for s in after['samples']])
        self.assertEqual(before['available_tools'], after['available_tools'])
        self.assertEqual(before['context']['prediction_contract'], after['context']['prediction_contract'])
        self.assertNotIn('output_repair_contract', before['context'])
        self.assertIn('output_repair_contract', after['context'])
        self.assertTrue(all(s['attempt']==2 and s['failure_feedback'][0]['failure_class']=='output_budget_exhausted'
                            for s in after['samples']))
        self.assertEqual(len(batch.scoring_rows), 9)
        self.assertTrue(all(r['status']=='succeeded' and r['attempts']==2 and r['retry_count']==1 for r in batch.records))
        self.assertEqual([len(p) for p in publications], [9])
        self.assertTrue(batch.summary['strict_agent_chain_pass'])
        replay=TruncatedPlanner(); execute(replay, max_attempts=2)
        self.assertEqual([r['idempotency_key'] for r in runtime.requests], [r['idempotency_key'] for r in replay.requests])

    def test_second_exhaustion_fails_closed_without_scoring_even_with_spare_attempts(self):
        runtime=TruncatedPlanner(failures=3); publications=[]
        with self.assertRaises(DshNativeRuntimeUnavailableError):execute(runtime, publications=publications)
        self.assertEqual(len(runtime.requests), 2)
        self.assertEqual(publications, [])

    def test_single_attempt_policy_does_not_gain_an_extra_call(self):
        runtime=TruncatedPlanner(); publications=[]
        with self.assertRaises(DshNativeRuntimeUnavailableError):execute(runtime, max_attempts=1, publications=publications)
        self.assertEqual(len(runtime.requests), 1)
        self.assertEqual(publications, [])

    def test_other_execution_contract_faults_are_not_reclassified(self):
        for code in ('structured_child_execution_budget_exhausted', 'structured_child_output_schema_invalid'):
            with self.subTest(code=code):
                runtime=TruncatedPlanner(code=code); publications=[]
                with self.assertRaises(DshNativeRuntimeUnavailableError):execute(runtime, publications=publications)
                self.assertEqual(len(runtime.requests), 1)
                self.assertEqual(publications, [])
