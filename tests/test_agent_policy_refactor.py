from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.trajectory import EvaluationScope, EvaluationPhase, HoldoutArm, HoldoutEvaluation
from ecologyrsi_dsh.evolution.agent_policy import build_agent_policy, summarize_agent_tools, prior_candidate_tool_experience
from ecologyrsi_dsh.evaluators.agent_stability import replica_summary, stability_evidence, paired_stability_gate
from ecologyrsi_dsh.evaluators.agent_model_tools import AgentModelTools
from ecologyrsi_dsh.evaluators.sample_execution import SampleExecutionPausedError, _validated_result
from tests import test_agent_model_tools as model_fixtures
from tests import test_agent_owned_prediction as agent_fixtures
from tests.test_agent_owned_prediction import result
from tests.test_baseline_aligned_ridge import periodic_series


class AgentPolicyRefactorTests(unittest.TestCase):
    def test_critic_revision_and_uncertainty_return_to_agent(self):
        fixture = agent_fixtures.AgentOwnedPredictionTests()
        self.addCleanup(fixture.doCleanups)
        for action in ('revise', 'uncertain'):
            adapter, plan, native, _ = fixture.setup_agent(lambda c, call: {**result(c), 'decisions': [{**row, 'confidence': .5} for row in result(c)['decisions']]})
            original = adapter._decision_client.sample_decide
            def review(model, **kwargs):
                if kwargs['role'] != 'critic':
                    return original(model, **kwargs)
                sample = kwargs['samples'][0]
                self.assertIn('history_window', sample['causal_context'])
                self.assertNotIn('observed', str(sample))
                return {'decisions': [{'sample_id': sample['sample_id'], 'action': action,
                    'reason_code': 'insufficient_context', 'confidence': .8}]}
            adapter._decision_client.sample_decide = review
            outcome = fixture.predict(adapter, plan)
            self.assertIsNone(outcome.result)
            self.assertEqual(outcome.error.failure_class, 'critic_' + action)
            self.assertTrue(outcome.error.retryable)
            self.assertEqual(outcome.error.previous_prediction, 23.25)
            self.assertIsNone(outcome.error.requested_tool_id)

    def test_raw_tool_error_and_agent_adjustment_are_separate(self):
        calls = [{'tool_id': 'ridge', 'version': '1', 'parameters': {'ridge_alpha': .1},
                  'status': 'completed', 'tool_predicted': 10., 'used_as_evidence': True}]
        records = [{'sample_id': 'one', 'attempt_trace': [{'outcome': 'accepted', 'model_evidence': calls}]}]
        rows = [{'sample_id': 'one', 'partition': 'training_feedback', 'target': 'temperature',
                 'horizon_hours': 1, 'predicted': 12., 'observed': 13.}]
        evidence = summarize_agent_tools(records, rows)
        self.assertEqual(evidence[0]['mae'], 3.)
        self.assertEqual(evidence[0]['agent_adjustment_mae_gain'], 2.)
        self.assertEqual(summarize_agent_tools(records, [{**rows[0], 'partition': 'model_selection'}]), [])
        args = dict(genome_digest='a'*64, profile={}, parameters={}, generation=2)
        baseline = build_agent_policy(**args, previous_analysis=None)
        learned = build_agent_policy(**args, previous_analysis={'generation': 1,
                       'ranking': [{'agent_tool_performance': evidence}]})
        self.assertNotEqual(baseline['policy_digest'], learned['policy_digest'])
        self.assertEqual(learned['experience']['rows'], evidence)
        with self.assertRaisesRegex(ValueError, 'precede'):
            build_agent_policy(**args, previous_analysis={'generation': 2})

    def test_formal_experience_survives_final_holdout_without_using_holdout_labels(self):
        tool = {'tool_id': 'ridge', 'evidence_kind': 'raw_tool_output', 'source_phase': 'formal_batch', 'mae': 3.}
        candidate = SimpleNamespace(candidate_id='c', generation=1)
        formal = SimpleNamespace(scope=SimpleNamespace(candidate_id='c', generation=1,
            phase='formal_batch', scope_key='a'*64), metrics={'sample_execution': {'agent_tool_performance': [tool]}})
        hidden = {**tool, 'source_phase': 'holdout', 'mae': 999.}
        final_metrics = {'sample_execution': {'agent_tool_performance': [hidden]}}
        rows = prior_candidate_tool_experience(SimpleNamespace(formal_batch_evaluations=(formal,)), candidate, final_metrics)
        self.assertEqual(rows, [{**tool, 'source_scope_digest': 'a'*64}])
        policy = build_agent_policy(genome_digest='b'*64, profile={}, parameters={}, generation=2,
            previous_analysis={'generation': 1, 'ranking': [{'agent_tool_performance': [*rows, hidden]}]})
        self.assertEqual(policy['experience']['rows'], rows)
        self.assertEqual(prior_candidate_tool_experience(SimpleNamespace(), candidate, final_metrics), [])

    def test_cache_eviction_and_restart_do_not_remove_model_capabilities(self):
        series = periodic_series()
        request = model_fixtures.AgentModelToolsTests().request(series)
        with TemporaryDirectory() as tmp:
            bank = AgentModelTools(series, targets=('air_temperature',), horizons=(1,), cache_dir=tmp)
            first = bank.execute((request,), 'greenhouse-exogenous-ridge@1', {'history_steps': 1})['one']
            for history in range(2, 11):
                bank.execute((request,), 'greenhouse-exogenous-ridge@1', {'history_steps': history})
            self.assertEqual(len(bank._fits), bank.CACHE_CAPACITY)
            restored = AgentModelTools(series, targets=('air_temperature',), horizons=(1,), cache_dir=tmp)
            with patch('ecologyrsi_dsh.evaluators.greenhouse_prediction.fit_predict_exogenous_ridge', side_effect=AssertionError('fit should be durable')):
                self.assertEqual(restored.execute((request,), 'greenhouse-exogenous-ridge@1', {'history_steps': 1})['one'], first)
            files = list(Path(tmp).glob('*.json'))
            self.assertEqual(len(files), 10)

    def test_model_fit_checks_pause_between_target_horizon_tasks(self):
        states = iter(['running', 'running', 'paused'])
        bank = AgentModelTools(periodic_series(), targets=('air_temperature',), horizons=(1,6,24), control=lambda: next(states))
        request = model_fixtures.AgentModelToolsTests().request(bank.series)
        with self.assertRaises(SampleExecutionPausedError):
            bank.execute((request,), 'greenhouse-exogenous-ridge@1', {})
        self.assertFalse(bank._fits)

    def test_hot_admission_and_retrieval_counts_decode_no_event_payload(self):
        ledger = EventLedger(); self.addCleanup(ledger.close)
        identity = {'run_id': 'r', 'stage': 'sample.plan', 'stage_attempt': 1, 'idempotency_key': 'key'}
        for i in range(200):
            ledger.append('r', 'Noise', {'payload': 'x'*1000})
        ledger.append('r', 'DshRetrievalExecuted', {'identity': identity})
        with patch.object(ledger, '_row_to_event', side_effect=AssertionError('unexpected event decoding')):
            self.assertTrue(ledger.run_exists('r'))
            self.assertEqual(ledger.retrieval_stage_count(identity), 1)
            self.assertFalse(ledger.run_exists('missing'))

    def test_event_replay_cannot_count_as_an_independent_replica(self):
        def arm(candidate, scores):
            scope = EvaluationScope('r', 1, candidate, 'rev:'+candidate, EvaluationPhase.HOLDOUT,
                                    'b'*64, 169, holdout_arm=HoldoutArm.INCUMBENT)
            rows = [replica_summary(replace(scope, inference_replica=i), SimpleNamespace(
                score=score, passed=True, metrics={'sample_execution': {'coverage': 1.}})) for i, score in enumerate(scores)]
            evaluation = HoldoutEvaluation('evaluation:'+candidate, scope, scores[0], True,
                {'agent_inference_stability': stability_evidence(rows)}, 'c'*64)
            return HoldoutEvaluation.from_dict(evaluation.to_dict())
        incumbent, candidate = arm('i', [.1,.11]), arm('c', [.15,.16])
        self.assertTrue(paired_stability_gate(candidate, incumbent, minimum_delta=.01)['passed'])
        unstable = arm('c', [.15,.10])
        self.assertFalse(paired_stability_gate(unstable, incumbent, minimum_delta=.01)['passed'])
        # Check the actual persisted immutable contract, including tampering
        # and replay presented as a second independent inference.
        metrics = candidate.to_dict()['metrics']
        metrics['agent_inference_stability']['replicas'][1]['score'] += .01
        tampered = replace(candidate, metrics=metrics)
        self.assertFalse(paired_stability_gate(tampered, incumbent, minimum_delta=.01)['passed'])
        rows = candidate.to_dict()['metrics']['agent_inference_stability']['replicas']
        rows[1]['scope_digest'] = rows[0]['scope_digest']
        candidate = replace(candidate, metrics={'agent_inference_stability': stability_evidence(rows)})
        self.assertFalse(paired_stability_gate(candidate, incumbent, minimum_delta=.01)['passed'])

    def test_exported_policy_loads_and_executes_through_the_native_agent(self):
        from ecologyrsi_dsh.integrations.agent_policy_bundle import export_policy_bundle, load_policy_bundle, create_policy_adapter
        from ecologyrsi_dsh.evaluators.greenhouse_prediction import ExogenousRidgeConfig, fit_predict_exogenous_ridge
        from ecologyrsi_dsh.integrations.dsh_structured_roles import DshStructuredRoleRuntime
        import json
        series = periodic_series()
        config = ExogenousRidgeConfig(6,.1,.5)
        fit = fit_predict_exogenous_ridge(series, targets=('air_temperature',), horizons=(1,), config=config)
        bank = AgentModelTools(series, targets=('air_temperature',), horizons=(1,))
        policy = build_agent_policy(genome_digest='a'*64, profile={}, parameters=config.to_dict(), previous_analysis=None, generation=0)
        contract = {'strategy_model_id':'dsh/strategy', 'review_model_id':'dsh/review'}
        artifact = SimpleNamespace(learned_parameters={'models':fit['models'], 'agent_policy':policy,
            'runtime_contract':contract, 'training_data_digest':bank.training_digest, 'optional_tool_catalog':bank.catalog()},
            model_id='greenhouse-exogenous-ridge@1', parameters=config.to_dict(), digest='c'*64)
        bundle = export_policy_bundle(artifact)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'policy.json';path.write_text(json.dumps(bundle))
            bundle = load_policy_bundle(path)
        fixture = agent_fixtures.AgentOwnedPredictionTests();self.addCleanup(fixture.doCleanups)
        def agent(c, call):
            output = call('candidate-model','loaded-model')
            return result(c, output['outputs'][0]['predicted'], method='model', refs=('loaded-model',))
        original, _, native, ledger = fixture.setup_agent(agent)
        bindings = dict(run_id='run-agent', runtime_provider=lambda: DshStructuredRoleRuntime(native,admission=native.service),
            revision_provider=original._decision_client.revision_provider,
            identity_digests=original._decision_client.identity_digests,
            prediction_tool_binder=native.service.bind_prediction_tool,
            remote_critic_policy={'version':'uncertain_or_failure@1','min_planner_confidence':.8},
            sample_reflection_policy='candidate_aggregate_post_score@1')
        adapter, context = create_policy_adapter(bundle, series, runtime_contract=contract, **bindings)
        request = model_fixtures.AgentModelToolsTests().request(series)
        outcome = adapter.predict_samples((request,), (adapter.plan_batch(context),), attempts=(1,))[0]
        self.assertIsNone(outcome.error)
        expected = next(r['predicted'] for r in fit['prediction_rows'] if r['origin_timestamp']==205 and r['partition']=='training_feedback')
        self.assertAlmostEqual(outcome.result['predicted'], expected)
        self.assertEqual(outcome.result['tool_calls'][0]['call_id'], 'loaded-model')
        altered = replace(series, values={**series.values, 'air_temperature': tuple(v+1 if i<100 else v for i,v in enumerate(series.values['air_temperature']))})
        with self.assertRaisesRegex(ValueError, 'training data differs'):
            create_policy_adapter(bundle, altered, runtime_contract=contract, **bindings)
