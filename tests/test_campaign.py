"""Guardrails must not mistake cost accounting for scientific progress."""
import unittest
from types import SimpleNamespace
from ecologyrsi_dsh.api.auto_progress import _failure_diagnostics
from ecologyrsi_dsh.core.errors import DshNativeRuntimeUnavailableError, dsh_native_runtime_retryable
from ecologyrsi_dsh.evolution.strategies import _predictor_semantics
from ecologyrsi_dsh.application.campaign import CampaignLimits, observe_progress, pause_reason, progress_signature, utc_timestamp


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.limits = CampaignLimits(1000, 200, 60)

    def reason(self, p, now=100, last=90):
        return pause_reason(p, self.limits, now=now, last_progress_at=last)

    def test_only_running_runs_are_controlled(self):
        for status in ('paused', 'completed', 'failed', 'cancelled', 'created'):
            self.assertIsNone(self.reason({'status':status}, now=1001))

    def test_unknown_usage_is_not_zero_or_a_hard_limit(self):
        for value in (None, True, '200', 199):
            self.assertIsNone(self.reason({'status':'running','token_usage_available':True,'tokens_used':value}))
        self.assertIsNone(self.reason({'status':'running','token_usage_available':False,'tokens_used':201}))
        self.assertEqual(self.reason({'status':'running','token_usage_available':True,'tokens_used':200}),
                         'campaign_reported_token_threshold')

    def test_deadline_applies_when_usage_unknown(self):
        self.assertEqual(self.reason({'status':'running'},now=1000), 'campaign_deadline')

    def test_usage_and_revision_do_not_reset_stall_clock(self):
        p={'status':'running','projection_revision':1,'tokens_used':100}
        self.assertEqual(progress_signature(p),progress_signature({**p,'projection_revision':3,'tokens_used':200}))
        self.assertEqual(self.reason(p,now=150,last=90), 'campaign_no_completed_work')

    def test_accepted_calls_and_origins_advance_progress(self):
        p={'execution_progress':{'stage_progress':{'completed_origins':1}}}
        self.assertNotEqual(progress_signature({}),progress_signature(p))
        self.assertNotEqual(progress_signature({}),progress_signature({'dsh_runtime':{'skill_invocation':{'verified_call_count':1}}}))

    def test_limits_and_timezones_are_required(self):
        for value in (True,0,-1,1.5):
            with self.assertRaises(ValueError):CampaignLimits(1000,value,60)
        with self.assertRaises(ValueError):utc_timestamp('2026-09-07T00:00:00')
        self.assertEqual(utc_timestamp('2026-09-07T08:00:00+08:00'),utc_timestamp('2026-09-07T00:00:00Z'))

    def test_maintenance_pause_does_not_count_as_running_stall(self):
        p = {'status': 'running'}
        saved = observe_progress(p, {}, now=100)
        saved = observe_progress({'status': 'paused'}, saved, now=120)
        saved = observe_progress(p, saved, now=500)
        self.assertEqual(saved['last_progress_at'], 500)
        self.assertIsNone(self.reason(p, now=500, last=saved['last_progress_at']))
        self.assertEqual(self.reason(p, now=560, last=saved['last_progress_at']),
                         'campaign_no_completed_work')

    def test_watcher_restart_and_usage_do_not_extend_running_stall(self):
        p = {'status': 'running'}
        saved = observe_progress(p, {}, now=100)
        saved = observe_progress({**p, 'tokens_used': 100, 'projection_revision': 99},
                                 dict(saved), now=150)
        self.assertEqual(saved['last_progress_at'], 100)
        self.assertEqual(self.reason(p, now=160, last=saved['last_progress_at']),
                         'campaign_no_completed_work')

    def test_resume_never_extends_deadline_or_token_limit(self):
        for p, now, expected in (
            ({'status': 'running'}, 1000, 'campaign_deadline'),
            ({'status': 'running', 'tokens_used': 200, 'token_usage_available': True},
             500, 'campaign_reported_token_threshold'),
        ):
            saved = observe_progress(p, {'observed_status': 'paused'}, now=now)
            self.assertEqual(self.reason(p, now=now, last=saved['last_progress_at']), expected)

    def test_research_failure_is_not_mislabeled_as_screening(self):
        state=SimpleNamespace(run=SimpleNamespace(generation=0),
                              task_manifest=SimpleNamespace(metadata={'optimization_protocol':'top2_adaptive_epoch@1'}))
        for stage in ('preflight','search','research','reflection','generation.judge'):
            _, context=_failure_diagnostics(state,RuntimeError(),stage=stage)
            self.assertEqual(context['stage'],stage)
            self.assertEqual(context['work_unit_kind'],stage)

    def test_model_contract_and_output_budget_are_nonretryable(self):
        for code in ('structured_child_tool_protocol_error','structured_child_output_budget_exhausted'):
            error=DshNativeRuntimeUnavailableError(error_code=code,status_code=422)
            self.assertFalse(dsh_native_runtime_retryable(error))
            state=SimpleNamespace(run=SimpleNamespace(generation=0),
                                  task_manifest=SimpleNamespace(metadata={}))
            actual, context=_failure_diagnostics(state,error,stage='research')
            self.assertEqual(actual,code)
            self.assertEqual(context['failure_domain'],'model_execution')
            untrusted=RuntimeError();untrusted.error_code=code
            self.assertNotEqual(_failure_diagnostics(state,untrusted,stage='research')[0],code)

    def test_predictor_facts_follow_the_parent_without_inventing_scale_fitting(self):
        for predictor, parameter, scope in (
            ('greenhouse-exogenous-ridge@1','residual_scale','shared across targets and horizons'),
            ('greenhouse-targetwise-ridge@1','relative_humidity_residual_scale','one scale per target, shared across horizons'),
            ('greenhouse-horizon-targetwise-ridge@1','relative_humidity_24h_residual_scale','one scale per target and horizon'),
        ):
            parent=SimpleNamespace(scientific_program={'predictor_ref':{'id':predictor},'parameter_overrides':{parameter:0.4}})
            facts=_predictor_semantics(parent)
            self.assertEqual(facts['residual_scale_scope'],scope)
            self.assertEqual(facts['residual_scale_values'],{parameter:0.4})
            self.assertFalse(facts['fit_updates_residual_scales'])
        other=SimpleNamespace(scientific_program={'predictor_ref':{'id':'toy-rolling-water@1'},'parameter_overrides':{}})
        self.assertNotIn('fit_scope',_predictor_semantics(other))
