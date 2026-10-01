"""Regression checks for the September 2026 evidence and lifecycle review."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.api.auto_progress import _failure_diagnostics
from ecologyrsi_dsh.api.errors import public_error_payload
from ecologyrsi_dsh.core.errors import FrozenRuntimeBindingDriftError, safe_binding_diagnostics
from ecologyrsi_dsh.core.ledger import ConcurrentRunMutationError, EventLedger
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.evolution.genome import mutation_policy_contract
from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter
from ecologyrsi_dsh.evolution.promotion import _paired_bootstrap_interval, _validated_evidence
from ecologyrsi_dsh.evaluators.objectives import OBJECTIVE_AGGREGATION_VERSION
from tests.test_promotion import _evaluation


class ReviewRegressionTests(unittest.TestCase):
    def test_confidence_level_controls_actual_bootstrap_quantiles(self):
        current = _validated_evidence(_evaluation('new', .1, version=OBJECTIVE_AGGREGATION_VERSION,
            block_scores=(-.2, .01, .1, .4, .2, .05, -.1, .3)))
        control = _validated_evidence(_evaluation('old', 0., version=OBJECTIVE_AGGREGATION_VERSION,
            block_scores=(0.,) * 8))
        ids = list(current['blocks'])
        wide = _paired_bootstrap_interval(current, control, ids, seed_material='confidence', confidence_level=.95)
        narrow = _paired_bootstrap_interval(current, control, ids, seed_material='confidence', confidence_level=.90)
        self.assertLess(wide[0], narrow[0])
        self.assertGreater(wide[1], narrow[1])
        for invalid in (0., 1., float('nan')):
            with self.assertRaises(ValueError):
                _paired_bootstrap_interval(current, control, ids, seed_material='confidence', confidence_level=invalid)

    def test_numerical_mutation_policy_changes_strategy_identity(self):
        before = StrategyRouterDSHAdapter.configuration_digest('autonomous_model@1')
        self.assertEqual(mutation_policy_contract()['maximum_normalized_step'], .15)
        with patch('ecologyrsi_dsh.evolution.genome.TRUST_REGION_MAX_NORMALIZED_STEP', .35):
            self.assertNotEqual(StrategyRouterDSHAdapter.configuration_digest('autonomous_model@1'), before)

    def test_safe_drift_diagnostics_survive_error_and_failure_projection(self):
        error = FrozenRuntimeBindingDriftError('进化策略实现', expected_digest='a' * 64, current_digest='b' * 64)
        self.assertEqual(public_error_payload(error)['binding_drift']['expected_digest'], 'a' * 64)
        state = SimpleNamespace(run=SimpleNamespace(generation=0), task_manifest=SimpleNamespace(metadata={}))
        code, context = _failure_diagnostics(state, error, stage='preflight')
        self.assertEqual(code, 'frozen_runtime_binding_drift')
        self.assertEqual(context['binding_drift'], error.diagnostics)
        unsafe = FrozenRuntimeBindingDriftError('https://secret.invalid/token', expected_digest='secret', current_digest='b' * 64)
        self.assertNotIn('secret', str(public_error_payload(unsafe)))
        self.assertEqual(safe_binding_diagnostics({'binding_label': [], 'secret': 'value'}), {})

    def test_late_append_cannot_revive_deleted_or_recreated_run(self):
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        run_id = 'run:incarnation-fence'
        original = ledger.append(run_id, 'RunCreated', {'test': True})
        ledger.archive_run(run_id)
        ledger.purge_run(run_id, confirmation=run_id, terminal_status='completed')
        for recreate in (False, True):
            if recreate:
                replacement = ledger.append(run_id, 'RunCreated', {'test': True})
                self.assertNotEqual(replacement.seq, original.seq)
            with self.assertRaises(ConcurrentRunMutationError):
                ledger.append(run_id, 'FormalStageCompleted', {'outcome': 'passed'},
                    expected_run_created_seq=original.seq)
            self.assertFalse(any(e.kind == 'FormalStageCompleted' for e in ledger.events_after(run_id)))


class EvidenceProjectionTests(unittest.TestCase):
    def test_last_identity_ignores_newer_untrained_proposal(self):
        from ecologyrsi_dsh.api.projection import _last_evidence_candidate_id
        state = SimpleNamespace(candidates=(SimpleNamespace(candidate_id='trained'), SimpleNamespace(candidate_id='proposal')),
            events=(SimpleNamespace(kind='EvaluationProgressRecorded', payload={'candidate_id': 'trained'}),
                    SimpleNamespace(kind='CandidateSpawned', payload={'candidate_id': 'proposal'})))
        self.assertEqual(_last_evidence_candidate_id(state), 'trained')
        state.events = (SimpleNamespace(kind='EvaluationRecorded', payload={'evaluation': {'candidate_id': 'trained'}}),)
        self.assertEqual(_last_evidence_candidate_id(state), 'trained')
        state.events = state.events[1:]
        self.assertIsNone(_last_evidence_candidate_id(state))

    def test_audit_deduplicates_cumulative_usage_and_records_stop_evidence(self):
        from scripts.analyze_evolution_campaign import analyze
        def event(kind, payload, seq):
            return SimpleNamespace(kind=kind, payload=payload, seq=seq, event_id=str(seq), run_id='run:audit', created_at='2026-09-22T00:00:00Z')
        def usage(total):
            return {'session_metrics': {'session_id': 's1', 'provider_usage': {'totals': {'total_tokens': total}}},
                    'identity': {'session_id': 's1', 'child_reservation_id': 'r1', 'stage': 'sample.plan'},
                    'settlement': 'completed', 'usage_complete': True}
        events = [event('RunCreated', {'task_manifest': {'metadata': {}, 'visible_datasets': ['toy']}}, 1),
                  event('DshChildLaunchReserved', {'launch': {'reservation_id': 'r1'}}, 2),
                  event('DshSessionUsageRecorded', usage(100), 3),
                  event('DshSessionUsageRecorded', usage(110), 4),
                  event('DshStructuredResultAccepted', usage(90), 5),
                  event('RunFailed', {'error_code': 'frozen_runtime_binding_drift', 'failure_context': {'stage': 'preflight'}}, 6)]
        report = analyze(events)
        self.assertEqual(report['reported_tokens'], 110)
        self.assertEqual(report['tokens_by_stage'], {'sample.plan': 110})
        self.assertEqual(report['latest_stop']['seq'], 6)
        self.assertEqual(report['latest_stop']['stage'], 'preflight')
        self.assertEqual(report['independent_evaluation_results'], [])
        self.assertTrue(report['usage_coverage']['complete'])

    def test_quick_exclusion_reason_survives_later_generic_ranking(self):
        from ecologyrsi_dsh.api.projection import _candidate_projection
        candidate = SimpleNamespace(candidate_id='unused', proposal_id='p', generation=0,
            status=SimpleNamespace(value='screened_out'), slot_index=3, created_at='now')
        proposal = SimpleNamespace(proposal_id='p', parent_candidate_id=None, title='proposal', rationale='', changes={}, metadata={})
        reason = 'outside_preregistered_quick_trajectory'
        state = SimpleNamespace(run=SimpleNamespace(),
            proposal=lambda _: proposal, evaluation_for=lambda _: None, promotion_for=lambda _: None,
            analysis_for=lambda _: SimpleNamespace(ranking=[{'candidate_id': 'unused', 'selection_reason': 'missing_evaluation'}]),
            events=(SimpleNamespace(kind='CandidateScreenedOut', payload={'candidate_id': 'unused', 'reason': reason}),))
        with patch('ecologyrsi_dsh.api.projection._candidate_execution_projection', return_value={}):
            self.assertEqual(_candidate_projection(state, candidate, summary_only=True)['selection_reason'], reason)
