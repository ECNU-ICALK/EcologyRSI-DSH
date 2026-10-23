"""Prospective arithmetic contracts use real evidence and preserve legacy runs."""
import json
from dataclasses import replace
from types import SimpleNamespace
import unittest

from ecologyrsi_dsh.core.agent_prediction import (
    EVIDENCE_FORMULA_POLICY, PredictionContractViolation, prediction_formula_contract, validate_prediction_receipt,
)
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.integrations.prediction_binding import DshPredictionToolBinding
from tests import test_agent_owned_prediction as agent_helpers


class EvidenceFormulaTests(unittest.TestCase):
    setup_agent = agent_helpers.AgentOwnedPredictionTests.setup_agent
    predict = agent_helpers.AgentOwnedPredictionTests.predict

    @staticmethod
    def freeze(plan):
        plan['decision_context']['agent_policy'] = {'inference': {
            'prediction_formula_policy': EVIDENCE_FORMULA_POLICY}}

    def test_exact_model_copy_and_mean_preview_accept_and_replay_without_model_call(self):
        for method in ('model', 'blend'):
            with self.subTest(method=method):
                def policy(context, call):
                    contract = context['context']['prediction_contract']
                    self.assertEqual(contract['allowed_methods'], ['blend', 'model'])
                    self.assertEqual(contract['formula_policy'], EVIDENCE_FORMULA_POLICY)
                    first = call('candidate-model', 'fit')
                    if method == 'model':
                        return agent_helpers.result(context, first['outputs'][0]['predicted'], method=method, refs=['fit'])
                    preview = call('persistence', 'base')['mean_formula']
                    self.assertEqual(preview['evidence_call_ids'], ['fit', 'base'])
                    self.assertEqual(preview['weight_per_call'], .5)
                    return agent_helpers.result(context, preview['outputs'][0]['predicted'],
                                                method=method, refs=preview['evidence_call_ids'])
                adapter, plan, native, ledger = self.setup_agent(policy)
                self.freeze(plan)
                outcome = self.predict(adapter, plan)
                self.assertIsNone(outcome.error)
                self.assertEqual(outcome.result['predicted'], 21.5 if method == 'model' else 20.75)
                self.assertEqual(outcome.result['tool_calls'][-1]['formula_status'],
                                 'model_copy_passed' if method == 'model' else 'passed')
                accepted = ledger.events_by_kind('run-agent', 'DshStructuredResultAccepted')[0]
                receipt = accepted.payload['required_tool_receipt']
                self.assertEqual(receipt['schema_version'], 'ecologyrsi-dsh.agent-prediction-receipt/3')
                self.assertEqual(receipt['formula_policy'], EVIDENCE_FORMULA_POLICY)
                validate_prediction_receipt(accepted.payload['structured'], receipt,
                    event_lookup=lambda key: ledger.event_by_id(key), identity=accepted.payload['identity'], before_seq=accepted.seq)
                def replay_event(key):
                    event = ledger.event_by_id(key)
                    return SimpleNamespace(payload=event.payload, kind=event.kind, seq=event.seq)
                validate_prediction_receipt(accepted.payload['structured'], receipt,
                    event_lookup=replay_event, identity=accepted.payload['identity'], before_seq=accepted.seq)
                self.assertEqual(self.predict(adapter, plan).result, outcome.result)
                self.assertEqual(len(native.requests), 1)
                # Re-signing the submitted result cannot make false arithmetic replay.
                forged = json.loads(json.dumps(accepted.payload['structured']))
                forged['decisions'][0]['predicted'] += 1
                forged_receipt = {**receipt, 'result_digest': digest(forged)}
                with self.assertRaises(PredictionContractViolation):
                    validate_prediction_receipt(forged, forged_receipt,
                        event_lookup=lambda key: ledger.event_by_id(key), identity=accepted.payload['identity'], before_seq=accepted.seq)

    def test_arbitrary_numbers_forbidden_methods_and_ambiguous_model_refs_reject_without_substitution(self):
        for method, value, refs in (
            ('model', 25., ['fit']), ('blend', 22., ['fit', 'base']),
            ('model', 20.75, ['fit', 'base']), ('adjusted', 20.75, ['fit']),
            ('direct', 20.75, []), ('model', 20.75, ['invented']),
        ):
            with self.subTest(method=method, refs=refs):
                def policy(context, call):
                    call('candidate-model', 'fit'); call('persistence', 'base')
                    return agent_helpers.result(context, value, method=method, refs=refs)
                adapter, plan, _, ledger = self.setup_agent(policy)
                self.freeze(plan)
                outcome = self.predict(adapter, plan)
                self.assertIsNotNone(outcome.error)
                self.assertIsNone(outcome.result)
                self.assertFalse(ledger.events_by_kind('run-agent', 'DshStructuredResultAccepted'))

    def test_failed_tool_does_not_create_implicit_fallback(self):
        def broken(_): raise ArithmeticError('synthetic failed model')
        for explicit_alternative in (False, True):
            with self.subTest(explicit_alternative=explicit_alternative):
                def policy(context, call):
                    self.assertEqual(call('candidate-model', 'fit')['status'], 'failed')
                    if explicit_alternative:
                        actual = call('persistence', 'base')
                        return agent_helpers.result(context, actual['outputs'][0]['predicted'], method='model', refs=['base'])
                    return agent_helpers.result(context, 20., method='model', refs=['fit'])
                adapter, plan, _, ledger = self.setup_agent(policy, tool=broken)
                self.freeze(plan)
                outcome = self.predict(adapter, plan)
                self.assertEqual(outcome.error is None, explicit_alternative)
                self.assertEqual(bool(ledger.events_by_kind('run-agent', 'DshStructuredResultAccepted')), explicit_alternative)

    def test_legacy_receipt_and_unrestricted_predictions_are_not_upgraded(self):
        def policy(context, call):
            response = call('candidate-model', 'fit')
            self.assertNotIn('mean_formula', response)
            return agent_helpers.result(context, 99., method='model', refs=['fit'])
        adapter, plan, native, ledger = self.setup_agent(policy)
        outcome = self.predict(adapter, plan)
        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.result['predicted'], 99.)
        accepted = ledger.events_by_kind('run-agent', 'DshStructuredResultAccepted')[0]
        receipt = accepted.payload['required_tool_receipt']
        self.assertEqual(receipt['schema_version'], 'ecologyrsi-dsh.agent-prediction-receipt/2')
        self.assertNotIn('formula_policy', receipt)
        def replay_event(key):
            event = ledger.event_by_id(key)
            return SimpleNamespace(payload=event.payload, kind=event.kind, seq=event.seq)
        validate_prediction_receipt(accepted.payload['structured'], receipt,
            event_lookup=replay_event, identity=accepted.payload['identity'], before_seq=accepted.seq)
        self.assertEqual(self.predict(adapter, plan).result, outcome.result)
        self.assertEqual(len(native.requests), 1)

    def test_duplicate_computation_is_not_an_extra_vote_but_equal_distinct_tools_are(self):
        events = []
        executions = []
        def execute(tool, parameters):
            executions.append((tool, parameters))
            return {sid: {'predicted': value} for sid, value in (('a', 10.1234567890123), ('b', 40.1234567890123))}
        def persist(payload):
            event = SimpleNamespace(event_id=f'tool:{len(events)}', payload=payload)
            events.append(event)
            return event
        binding = DshPredictionToolBinding(run_id='r', stage_attempt=1, idempotency_key='k',
            wave_digest='a'*64, sample_ids=['a', 'b'], executor=execute, budget=6,
            catalog=[{'tool_id': tool, 'version': '1', 'parameters': {'scale': {'default': 1}}}
                     for tool in ('first', 'different')], formula_policy=EVIDENCE_FORMULA_POLICY)
        def call(tool, call_id, parameters):
            return binding.execute({'tool_id': tool, 'call_id': call_id, 'parameters': parameters,
                                    'wave_digest': 'a'*64}, session_id='s', persist=persist)
        call('first', 'one', {})
        duplicate = call('first', 'same', {'scale': 1})
        self.assertNotIn('mean_formula', duplicate)
        self.assertEqual(len(executions), 1)
        equal_but_distinct = call('different', 'other', {})
        self.assertEqual(equal_but_distinct['mean_formula']['evidence_call_ids'], ['one', 'other'])
        self.assertEqual(len(executions), 2)
        row = {'sample_id': 'a', 'method': 'blend', 'predicted': 10.1234567890123,
               'evidence_call_ids': ['one', 'same']}
        with self.assertRaises(PredictionContractViolation) as error:
            binding.validate_formulas([row])
        self.assertEqual(error.exception.diagnostics[0]['code'], 'duplicate_computation_evidence')
        binding.validate_formulas([{**row, 'evidence_call_ids': ['one', 'other']}])
        with self.assertRaises(PredictionContractViolation):
            binding.validate_formulas([{**row, 'sample_id': 'b', 'evidence_call_ids': ['one', 'other']}])
        with self.assertRaises(PredictionContractViolation):
            binding.validate_formulas([{**row, 'sample_id': 'foreign', 'evidence_call_ids': ['one', 'other']}])


class FormulaAuthoringTests(unittest.TestCase):
    metadata = {'prediction_formula_policy': EVIDENCE_FORMULA_POLICY}

    @staticmethod
    def operation(rule):
        return {'op': 'author_role_directive', 'role': 'sample-planner', 'authored_directive': {
            'anchor': 'candidate_model', 'blend_rule': rule,
            'tool_plan': [{'tool_id': 'ecology_execute_prediction_tool', 'purpose': 'candidate_model_baseline'}],
            'rationale': 'Use measured numerical evidence with an auditable combination.'}}

    def test_new_grammar_and_witnesses_filter_incompatible_rules_without_changing_legacy(self):
        from ecologyrsi_dsh.evolution.strategies import _directive_authoring_contract, _directive_preflight_operations
        from ecologyrsi_dsh.application.formal_trajectory import _legal_directive_grammar
        from tests.test_local_edits import _parent
        parent = _parent()
        legacy = _directive_authoring_contract(parent)
        frozen = _directive_authoring_contract(parent, metadata=self.metadata)
        self.assertIn('confidence_weighted', legacy['blend_rule']['choices'])
        self.assertEqual(frozen['blend_rule']['choices'], ['mean', 'none'])
        self.assertEqual(frozen['blend_rule.allowed_methods'], {'none': ['model'], 'mean': ['model', 'blend']})
        self.assertEqual(_legal_directive_grammar(SimpleNamespace(genome=parent.to_dict()), metadata=self.metadata), frozen)
        self.assertEqual({op['authored_directive']['blend_rule'] for op in
            _directive_preflight_operations(parent, 'author', metadata=self.metadata)}, {'mean', 'none'})
        self.assertIn('confidence_weighted', {op['authored_directive']['blend_rule'] for op in
            _directive_preflight_operations(parent, 'author')})

    def test_proposer_repairs_conflicting_directive_inside_existing_bounded_retry(self):
        from ecologyrsi_dsh import EventLedger, EvolutionDirector
        from ecologyrsi_dsh.core.models import canonical_json
        from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter
        from tests.test_dsh_structured_roles import _native_task, _SequencedNativeRuntime
        runtime = _SequencedNativeRuntime([{'schema_version': 'ecologyrsi-dsh.genome-mutation/1',
            'operations': [self.operation(rule)]} for rule in ('confidence_weighted', 'mean')])
        adapter = StrategyRouterDSHAdapter(gateway=object(), native_runtime_provider=lambda: runtime)
        task = _native_task()
        task = replace(task, metadata={**task.metadata, **self.metadata})
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, adapter)
            director.create_run(task, run_id='formula-authoring')
            director.start_run('formula-authoring')
            state = director.state('formula-authoring')
            parent = state.materialized_seed_genome()
            proposal = adapter.propose(state.run, task, state.run.session_id, batch_context={
                'generation': 0, 'slot_index': 0, 'batch_size': 1, 'context_digest': 'a'*64,
                'parent_genome_digest': parent.genome_digest, 'parent_genome_canonical_json': canonical_json(parent.to_dict()),
                'stage_context_digests': {'research_iteration_digest': 'b'*64, 'knowledge_snapshot_digest': 'c'*64}})
        self.assertEqual(len(runtime.requests), 2)
        context = runtime.requests[1]['request']['context']
        rejection = context['evolution_reflection']['host_rejections'][0]
        self.assertIn('blend_rule', rejection['validation_detail'])
        self.assertEqual(proposal.metadata['agent_policy']['inference']['prediction_formula_policy'], EVIDENCE_FORMULA_POLICY)
        self.assertEqual(proposal.metadata['candidate_agent_profile']['authored_directive']['blend_rule'], 'mean')

    def test_local_editor_repairs_formula_conflict_and_apply_rejects_bypass(self):
        from ecologyrsi_dsh.application.formal_trajectory import _validated_local_edit_response
        from ecologyrsi_dsh.evolution.local_edits import LocalEditContext, LocalEditProposal, apply_or_reject_local_edit_bundle
        from ecologyrsi_dsh.core.trajectory import LocalEditOutcome
        from ecologyrsi_dsh.evaluators.authored_directive import AUTHORED_DIRECTIVE_POLICY_ID
        from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
        from tests.test_local_edits import _parent, _context
        def proposal(rule):
            return LocalEditProposal(decision='mutate', operations=(self.operation(rule),),
                evidence_refs=('metric:overall',), expected_effect_cells=('air_temperature@1h',), risk_cells=())
        calls = []
        def run(**request):
            calls.append(request)
            return proposal('confidence_weighted' if len(calls) == 1 else 'mean').to_dict()
        accepted = _validated_local_edit_response(SimpleNamespace(run=run),
            context={'prediction_formula_contract': prediction_formula_contract(self.metadata)}, idempotency_key='local:formula')
        self.assertEqual(accepted.operations[0]['authored_directive']['blend_rule'], 'mean')
        self.assertEqual(len(calls), 2)
        self.assertIn('host_validation_feedback', calls[1]['context'])
        parent = _parent()
        context_data = _context(parent).to_dict()
        context_data['allowed_mutation_targets']['instruction_directive'] = [AUTHORED_DIRECTIVE_POLICY_ID]
        context = LocalEditContext(**context_data)
        rejected = apply_or_reject_local_edit_bundle(parent, proposal('confidence_weighted'), context,
            current_program_registry(), task=SimpleNamespace(metadata=self.metadata))
        self.assertEqual(rejected.outcome, LocalEditOutcome.REJECTED)
        self.assertIn('blend_rule', rejected.rejection_reason)
        self.assertIsNone(rejected.child)
        legacy = apply_or_reject_local_edit_bundle(parent, proposal('confidence_weighted'), context,
            current_program_registry(), task=SimpleNamespace(metadata={}))
        self.assertEqual(legacy.outcome, LocalEditOutcome.APPLIED)


if __name__ == '__main__':
    unittest.main()
