from dataclasses import replace
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.core.models import TaskManifest, digest
from ecologyrsi_dsh.evaluators.agent_model_tools import _defaults, _CONFIGS
from ecologyrsi_dsh.evaluators.strategy_program import seed_strategy_program, execute_causal_tool
from ecologyrsi_dsh.evolution import program_preflight as preflight, joint_policy
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from tests import test_candidate_novelty as novelty_tests
from tests.test_dsh_structured_roles import _native_task
from tests.test_joint_plugin_evolution import program_op
from tests.test_dsh_sample_execution import _request


def seasonal(period=24):
    p = seed_strategy_program()
    p['calls'] = [{'name': 'season', 'tool_id': 'causal-seasonal@1', 'parameters': {'period': period}}]
    p['expression'] = 'season'
    return p


def task(enabled=True):
    base = _native_task()
    return replace(base, metadata={**base.metadata, 'optimization_protocol': joint_policy.UNIFIED_PROTOCOL,
        'origin_history_alignment_hours': 48,
        **({preflight.POLICY_KEY: preflight.POLICY} if enabled else {})})


class ProgramPreflightTests(unittest.TestCase):
    def check(self, program, *, enabled=True, history=6, alignment=48):
        t = task(enabled)
        t = replace(t, metadata={**t.metadata, 'origin_history_alignment_hours': alignment})
        predictor = t.metadata['prediction_model_id']
        params = {**_defaults(_CONFIGS[predictor]), 'history_steps': history}
        return preflight.validate_program_inputs(t, predictor_id=predictor, parameters=params,
            role_profiles=[{'role': 'sample-planner', 'strategy_program': program}])

    def test_seasonal_failure_detected_from_exact_prompt_history_not_alignment(self):
        with self.assertRaisesRegex(preflight.CandidateProgramPreflightError, r'missing causal offsets \[-23\]'):
            self.check(seasonal())
        # The unchanged runtime guard agrees with the input-contract check.
        request = _request('causal')
        with self.assertRaisesRegex(ValueError, 'outside available causal history'):
            execute_causal_tool([request], 'causal-seasonal@1', {'period':24})

    def test_all_horizons_checked_and_unused_calls_are_still_eager(self):
        p = seasonal(12)
        p['expression'] = '0 if horizon <= 6 else season'
        with self.assertRaisesRegex(preflight.CandidateProgramPreflightError, r'@1h'):
            self.check(p)
        # Both short horizons fit a 6-point window when the period is six.
        receipt = self.check(seasonal(6))
        self.assertEqual(receipt['programs'][0]['checked_cell_count'], 9)

    def test_valid_contract_digest_is_small_and_does_not_read_dataset_or_model(self):
        with patch('ecologyrsi_dsh.data.registry.DatasetRegistry', side_effect=AssertionError('no data reads')):
            receipt = self.check(seasonal(1))
        self.assertEqual(receipt['status'], 'input_contract_checked')
        self.assertFalse(receipt['label_access'])
        self.assertEqual(receipt['receipt_digest'], digest({k:v for k,v in receipt.items() if k!='receipt_digest'}))
        self.assertLess(len(str(receipt)), 1400)

    def test_historical_manifest_keeps_no_preflight_or_upgrade(self):
        self.assertIsNone(self.check(seasonal(), enabled=False))
        old = task(False)
        self.assertEqual(TaskManifest.from_dict(old.to_dict()).digest, old.digest)
        self.assertNotEqual(old.digest, task(True).digest)
        with self.assertRaisesRegex(ValueError, 'candidate_executability_policy'):
            replace(old, metadata={**old.metadata, preflight.POLICY_KEY:'unknown'})

    def test_window_and_model_history_have_distinct_sources(self):
        p = seed_strategy_program()
        with self.assertRaisesRegex(preflight.CandidateProgramPreflightError, 'needs 6 history values'):
            self.check(p, history=3)
        p['calls'] = [{'name':'m','tool_id':'greenhouse-exogenous-ridge@1', 'parameters':{'history_steps':12}}]
        p['expression'] = 'm'
        self.assertEqual(self.check(p, history=3)['status'], 'input_contract_checked')
        with self.assertRaisesRegex(preflight.CandidateProgramPreflightError, 'beyond the frozen alignment'):
            self.check(p, history=3, alignment=6)

    def test_real_proposer_repairs_invalid_candidate_before_any_slot_is_committed(self):
        helper = novelty_tests.CandidateNoveltyTests(methodName='runTest')
        self.addCleanup(helper.doCleanups)
        outputs = [{'schema_version':'ecologyrsi-dsh.genome-mutation/1','operations':[program_op(p)]}
                   for p in (seasonal(), seasonal(1))]
        ledger, director, adapter, runtime = helper.setup_run(outputs, task=task())
        before = ledger.latest_seq()
        proposal = helper.propose(director, adapter)
        self.assertEqual(ledger.latest_seq(), before)
        self.assertEqual(len(runtime.requests), 2)
        rejection = runtime.requests[1]['request']['context']['evolution_reflection']['host_rejections'][0]
        self.assertEqual(rejection['rejection_code'], 'candidate_program_not_executable')
        self.assertIn('missing causal offsets', rejection['validation_detail'])
        self.assertNotEqual(runtime.requests[0]['stage_attempt'], runtime.requests[1]['stage_attempt'])
        director.submit_proposal(proposal)
        candidate = director.spawn_candidate('run:novelty', proposal)
        self.assertEqual(len(director.state('run:novelty').candidates), 1)
        self.assertFalse(director.state('run:novelty').evaluations)
        self.assertIsNotNone(candidate)

    def test_exhausted_proposal_repair_does_not_create_scoring_or_candidate(self):
        helper = novelty_tests.CandidateNoveltyTests(methodName='runTest')
        self.addCleanup(helper.doCleanups)
        output={'schema_version':'ecologyrsi-dsh.genome-mutation/1','operations':[program_op(seasonal())]}
        ledger, director, adapter, runtime = helper.setup_run([output], task=task())
        before=ledger.latest_seq()
        with self.assertRaisesRegex(ValueError, 'does not compile after bounded retry'):
            helper.propose(director, adapter)
        self.assertEqual(len(runtime.requests), 4)
        self.assertEqual(ledger.latest_seq(), before)
        self.assertFalse(director.state('run:novelty').candidates)

    def test_local_edit_rejects_inapplicable_program_without_creating_child(self):
        from tests.test_local_edits import _parent, _context
        from ecologyrsi_dsh.evolution.local_edits import LocalEditProposal, apply_or_reject_local_edit_bundle
        from ecologyrsi_dsh.core.trajectory import LocalEditOutcome
        parent = _parent()
        before = parent.to_dict()
        context = _context(parent)
        context = replace(context, allowed_mutation_targets={
            **context.allowed_mutation_targets, 'strategy_program': ('bounded_strategy_program@1',)})
        proposal = LocalEditProposal(decision='mutate', operations=[program_op(seasonal())],
            evidence_refs=['metric:overall'], expected_effect_cells=['air_temperature@1h'], risk_cells=[])
        rejected = apply_or_reject_local_edit_bundle(parent, proposal, context,
            current_program_registry(), task=task())
        self.assertEqual(rejected.outcome, LocalEditOutcome.REJECTED)
        self.assertIn('missing causal offsets', rejected.rejection_reason)
        self.assertIsNone(rejected.child)
        self.assertEqual(parent.to_dict(), before)
        repaired = replace(proposal, operations=[program_op(seasonal(1))])
        accepted = apply_or_reject_local_edit_bundle(parent, repaired, context,
            current_program_registry(), task=task())
        self.assertEqual(accepted.outcome, LocalEditOutcome.APPLIED, accepted.rejection_reason)
        self.assertIsNotNone(accepted.child)

    def test_recipe_program_uses_authored_recipe_instead_of_scalar_parameters(self):
        from tests.test_workflow_ir import _recipe_task, _seed
        from tests.test_joint_plugin_evolution import mutate
        from ecologyrsi_dsh.evolution.workflow_ir import compile_plugin_behavior
        base = _recipe_task()
        t = replace(base, metadata={**base.metadata, preflight.POLICY_KEY: preflight.POLICY,
                                   'origin_history_alignment_hours':48})
        genome = mutate(_seed(t, template_id='greenhouse-recipe-default@1'), [program_op(seasonal(1))])
        compiled = compile_plugin_behavior(genome, t, None, current_program_registry())
        receipt = compiled.feature_training_spec['candidate_executability']
        self.assertEqual(receipt['status'], 'input_contract_checked')
        self.assertEqual(receipt['programs'][0]['checked_cell_count'], 9)
        self.assertEqual(receipt['parameters_digest'], digest({'feature_recipe':
            genome.to_dict()['scientific_program']['feature_recipe']}))


if __name__ == '__main__':
    unittest.main()
