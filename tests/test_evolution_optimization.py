"""Regressions from the September campaign: execution, selection and read models."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from ecologyrsi_dsh.application.formal_trajectory import _local_edit_execution_reason
from ecologyrsi_dsh.api.projection import _workspace_revisions
from ecologyrsi_dsh.core.dsh_usage import active_session_ids
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.search_policy import SEARCH_GUARD_POLICY, local_challenger_policy
from ecologyrsi_dsh.core.trajectory import FormalBatchArm, LocalEditOutcome
from ecologyrsi_dsh.evaluators.fitness import FitnessProfile
from ecologyrsi_dsh.evaluators.generation_comparison import _gate, _cell_gate
from ecologyrsi_dsh.evolution.champion_challenger import assess_local_challenger
from ecologyrsi_dsh.evolution.local_edits import LocalEditContext, LocalEditProposal, apply_or_reject_local_edit_bundle
from ecologyrsi_dsh.integrations.prediction_binding import DshPredictionToolBinding
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from tests.test_local_edits import _parent, _context
from tests.test_champion_challenger import _evaluation, _metrics


class EvolutionOptimizationTests(unittest.TestCase):
    def test_non_executable_uncertainty_mutation_cannot_create_revision(self):
        parent = _parent()
        raw = _context(parent).to_dict()
        raw['allowed_mutation_targets']['uncertainty_policy'] = ['cellwise_time_block_calibrated_residual@1']
        proposal = LocalEditProposal(decision='mutate', operations=({
            'op': 'select_registered_uncertainty_policy',
            'program_id': 'cellwise_time_block_calibrated_residual@1',
        },), evidence_refs=('metric:overall',), expected_effect_cells=('air_temperature@1h',), risk_cells=())
        outcome = apply_or_reject_local_edit_bundle(parent, proposal, LocalEditContext(**raw), current_program_registry())
        self.assertEqual(outcome.outcome, LocalEditOutcome.REJECTED)
        self.assertIsNone(outcome.child)
        self.assertIn('uncertainty_policy_not_executable', outcome.rejection_reason)

    def test_target_horizon_edit_cannot_claim_other_horizon_and_requires_evidence(self):
        parent = _parent()
        raw = _context(parent).to_dict()
        name = 'co2_concentration_1h_residual_scale'
        raw['allowed_mutation_targets']['scientific_parameter'].append(name)
        raw['parameter_schemas'][name] = {'minimum': 0., 'maximum': 1.}
        proposal = LocalEditProposal(decision='mutate', operations=({
            'op': 'set_bounded_parameter', 'name': name, 'value': .05,
        },), evidence_refs=('metric:overall',), expected_effect_cells=('co2_concentration@24h',), risk_cells=())
        for edited, reason in ((proposal, 'effect cells'), (replace(proposal, evidence_refs=()), 'requires training evidence')):
            result = apply_or_reject_local_edit_bundle(parent, edited, LocalEditContext(**raw), current_program_registry())
            self.assertEqual(result.outcome, LocalEditOutcome.REJECTED)
            self.assertIsNone(result.child)
            self.assertIn(reason, result.rejection_reason)

    def test_frozen_tolerance_controls_local_and_epoch_cell_gates(self):
        champion = _evaluation(arm=FormalBatchArm.CHAMPION, revision_id='old', score=.2, skill=.2)
        metrics = _metrics(.3)
        metrics['targets'][0]['skill_score'] = .194
        challenger = _evaluation(arm=FormalBatchArm.CHALLENGER, revision_id='new', score=.27, skill=.3, metrics=metrics)
        profile = FitnessProfile(selection_cell_regression_tolerance=.01)
        policy = local_challenger_policy({'fitness_profile': profile.to_dict(), 'search_guard_policy': SEARCH_GUARD_POLICY})
        # Test the numeric contract independently of the stronger paired-block gate.
        def assess(tolerance):
            return assess_local_challenger(champion, challenger, challenger_safety_gate_passed=True,
                minimum_score_delta=policy['minimum_score_delta'], cell_regression_tolerance=tolerance)
        permissive = assess(policy['cell_regression_tolerance'])
        strict = assess(.005)
        self.assertEqual(permissive.champion_after_revision_id, 'new')
        self.assertEqual(strict.champion_after_revision_id, 'old')
        self.assertNotEqual(permissive.comparison_contract_digest, strict.comparison_contract_digest)
        expected = {(row['target'], row['horizon_hours']) for row in metrics['targets']}
        gate = _cell_gate(challenger, champion, expected, profile)
        self.assertTrue(gate['no_regression'])
        self.assertFalse(_cell_gate(challenger, champion, expected, profile.with_overrides(selection_cell_regression_tolerance=.005))['no_regression'])
        self.assertNotEqual(profile.profile_digest, profile.with_overrides(selection_cell_regression_tolerance=.005).profile_digest)

    def test_weight_coverage_does_not_hide_execution_failures(self):
        item = SimpleNamespace(passed=True, score=.3, metrics={
            'constraint_violations': 0, 'objective_weight_coverage': 1.,
            'sample_execution': {'coverage_pass': True, 'coverage': 1.,
                                 'attempted_origin_samples': 100, 'succeeded_origin_samples': 60},
        })
        result = _gate(item)
        self.assertEqual(result['overall_coverage'], .6)
        self.assertFalse(result['eligible'])
        self.assertIsNotNone(_local_edit_execution_reason({'sample_execution': {
            'failed_origin_samples': 2, 'coverage_pass': True, 'successful_agent_provenance_pass': True,
        }}))

    def test_valid_success_receipts_do_not_promote_an_incomplete_pair(self):
        champion = _evaluation(arm=FormalBatchArm.CHAMPION, revision_id='old', score=.1, skill=.1)
        challenger = _evaluation(arm=FormalBatchArm.CHALLENGER, revision_id='new', score=.4, skill=.4)
        def with_receipts(evaluation):
            return replace(evaluation, metrics={**evaluation.to_dict()['metrics'], 'sample_execution': {
                'successful_agent_provenance_pass': True, 'strict_agent_chain_pass': False,
                'attempted_origin_samples': 50, 'succeeded_origin_samples': 49, 'failed_origin_samples': 1,
            }})
        result = assess_local_challenger(with_receipts(champion), with_receipts(challenger),
            challenger_safety_gate_passed=True, require_paired_strict_chain=True)
        self.assertEqual(result.champion_after_revision_id, 'old')
        self.assertEqual(result.reason, 'paired_scoring_evidence_incomplete')

    def test_equivalent_requests_reuse_computation_with_distinct_session_receipts(self):
        ledger = EventLedger()
        self.addCleanup(ledger.close)
        computations = []
        def execute(tool, parameters):
            computations.append((tool, parameters))
            return {'s': {'predicted': 20., 'metadata': {}}}
        def binding(wave='w'):
            return DshPredictionToolBinding(run_id='r', stage_attempt=1, idempotency_key='k', wave_digest=wave,
                sample_ids=['s'], catalog=[{'tool_id': 'ridge', 'version': '1', 'parameters': {'alpha': {'default': .1}}}], executor=execute)
        def call(current, call_id, parameters=None):
            return current.execute({'tool_id': 'ridge', 'call_id': call_id, 'wave_digest': current.wave_digest,
                'parameters': parameters or {}}, session_id='agent', persist=lambda p: ledger.append('r', 'DshPredictionToolExecuted', p))
        current = binding()
        first = call(current, 'one')
        second = call(current, 'two', {'alpha': .1})
        self.assertEqual(len(computations), 1)
        self.assertNotEqual(first['event_id'], second['event_id'])
        self.assertEqual(second['reused_from_event_id'], first['event_id'])
        structured = {'schema_version': 'ecology-sample-predictions@2', 'wave_digest': 'w',
                      'decisions': [{'sample_id': 's', 'predicted': 20., 'confidence': .9,
                        'method': 'model', 'reason_code': 'agent_model', 'evidence_call_ids': ['two']}]}
        self.assertEqual(len(current.final_receipt(structured, session_id='agent')['calls']), 2)
        with self.assertRaisesRegex(ValueError, 'did not receive'):
            current.final_receipt(structured, session_id='other-agent')
        restored = binding()
        for event in ledger.events('r'):
            restored.restore(event)
        call(restored, 'two', {'alpha': .1})
        self.assertEqual(len(computations), 1)
        with self.assertRaisesRegex(ValueError, 'budget exhausted'):
            call(restored, 'three')
        call(binding('new-attempt'), 'four', {'alpha': .2})
        call(binding('other-origin'), 'five')
        self.assertEqual(len(computations), 3)

    def test_usage_only_updates_preserve_workspace_revision_and_terminal_session(self):
        def event(seq, kind, settlement=None):
            return SimpleNamespace(seq=seq, kind=kind, payload={
                'identity': {'child_reservation_id': 'child', 'session_id': 'session'},
                'settlement': settlement,
            })
        events = [event(1, 'CandidateSpawned'), event(2, 'DshSessionUsageRecorded', 'active')]
        state = SimpleNamespace(events=events)
        self.assertEqual(_workspace_revisions(state)['candidates'], 1)
        self.assertEqual(active_session_ids(events), {'session'})
        events.extend([event(3, 'DshStructuredResultAccepted'), event(4, 'DshSessionUsageRecorded', 'active')])
        self.assertFalse(active_session_ids(events))
        self.assertEqual(_workspace_revisions(state)['candidates'], 3)
        events.append(event(5, 'CandidateEvaluated'))
        self.assertEqual(_workspace_revisions(state)['training'], 5)
