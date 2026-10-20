"""Joint plugin admission, numerical execution, causal evidence and budget replay."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from ecologyrsi_dsh.core.models import canonical_json, digest
from ecologyrsi_dsh.core.trajectory import CandidateRevision
from ecologyrsi_dsh.core.agent_prediction import validate_tool_event
from ecologyrsi_dsh.evaluators.strategy_program import (
    seed_strategy_program, validate_strategy_program, execute_strategy_program, execute_causal_tool,
)
from ecologyrsi_dsh.evolution import joint_policy
from ecologyrsi_dsh.evolution.genome import EcologyEvolutionPluginGenome, apply_genome_mutation, TRUST_REGION_MUTATION_OPERATOR_ID
from ecologyrsi_dsh.evolution.workflow_ir import resolve_candidate_agent_profile
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evolution.effect_contracts import resolve_mutation_effects, observe_edit_effects
from ecologyrsi_dsh.evolution.strategies import _inherited_strategy_context
from ecologyrsi_dsh.integrations.prediction_binding import DshPredictionToolBinding
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from tests.test_evolution_genome import _seed_genome, _mutation_context
from tests.test_dsh_sample_execution import _request
from tests import test_agent_owned_prediction as agent_tests
from tests.test_agent_owned_prediction import result
from tests.test_effect_contracts import accepted_trace
from tests import test_evidence_guided_epoch as race_tests


def program():
    value = seed_strategy_program()
    value['calls'][1]['parameters']['window'] = 2
    return value


def mutate(parent, operations, operator=joint_policy.MUTATION_OPERATOR):
    return apply_genome_mutation(parent, {'schema_version': 'ecologyrsi-dsh.genome-mutation/1', 'operations': operations},
        _mutation_context(parent, mutation_operator_id=operator), current_program_registry())


def program_op(value=None):
    return {'op': 'author_strategy_program', 'role': 'sample-planner', 'strategy_program': value or program()}


class JointPluginTests(unittest.TestCase):
    def test_persisted_revision_program_replays_without_mutating_frozen_source(self):
        child = mutate(_seed_genome(), [program_op()])
        revision = CandidateRevision(
            revision_id='revision:joint:r0', run_id='run:joint', generation=0,
            candidate_id='candidate:joint', genome=child.to_dict(),
            genome_digest=child.genome_digest, behavior_digest=child.behavior_digest,
            mutation_digest=child.lineage['mutation_digest'],
        )
        # The same immutable nested mappings are reconstructed on ledger replay
        # and passed to the screening/formal-lane compiler in production.
        revision = CandidateRevision.from_dict(revision.to_dict())
        before = revision.to_dict()
        restored = EcologyEvolutionPluginGenome.from_dict(dict(revision.genome))
        self.assertEqual(restored.to_dict(), child.to_dict())
        frozen_program = next(row['strategy_program'] for row in
            revision.genome['agent_program']['candidate_execution_program']['role_profiles']
            if 'strategy_program' in row)
        actual = execute_strategy_program(frozen_program, [_request('cell')],
            lambda *args: {'cell': {'predicted': 20.}})
        expected = execute_strategy_program(program(), [_request('cell')],
            lambda *args: {'cell': {'predicted': 20.}})
        self.assertEqual(actual, expected)
        editable = validate_strategy_program(frozen_program)
        editable['calls'][1]['parameters']['damping'] = .5
        self.assertEqual(revision.to_dict(), before)

    def test_joint_genome_compiles_and_inherits_all_guidance_without_altering_parent(self):
        parent = _seed_genome(); before = parent.to_dict()
        ops = [program_op(), {'op': 'set_bounded_parameter', 'name': 'ridge_alpha', 'value': .2}]
        child = mutate(parent, ops)
        self.assertEqual(parent.to_dict(), before)
        profile = resolve_candidate_agent_profile(child, current_program_registry())
        self.assertEqual(profile['strategy_program'], program())
        self.assertEqual(profile['strategy_program_digest'], digest(program()))
        task = SimpleNamespace(metadata={'optimization_protocol': joint_policy.PROTOCOL})
        inherited = _inherited_strategy_context(child, task)['inherited_search_strategy']
        self.assertEqual(inherited['memory'], program()['memory'])
        self.assertEqual(inherited['proposal_prompt'], program()['proposal_prompt'])
        self.assertEqual(_inherited_strategy_context(child, SimpleNamespace(metadata={})), {})
        resolution = resolve_mutation_effects(parent, child, ops, current_program_registry())
        self.assertEqual(len(resolution['operations']), 2)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            mutate(parent, ops, TRUST_REGION_MUTATION_OPERATOR_ID)
        with self.assertRaisesRegex(ValueError, 'joint mutation policy'):
            mutate(parent, [program_op()], TRUST_REGION_MUTATION_OPERATOR_ID)

    def test_joint_duplicate_coordinate_cannot_walk_past_trust_region(self):
        parent = _seed_genome()
        with self.assertRaisesRegex(ValueError, 'repeat a coordinate'):
            mutate(parent, [{'op':'set_bounded_parameter','name':'ridge_alpha','value':v} for v in (.2, .3)])
        with self.assertRaises(ValueError):
            mutate(parent, [program_op()]*5)
        with self.assertRaises(ValueError):
            mutate(parent, [{'op':'set_bounded_parameter','name':'ridge_alpha','value':100.}])

    def test_program_changes_numerical_output_and_records_only_executed_branch_consumption(self):
        request = _request('cell')
        def executor(tool, params):
            if tool == 'persistence':return {'cell': {'predicted':20.}}
            return execute_causal_tool([request], tool, params)
        first = execute_strategy_program(program(), [request], executor)['cell']
        modified = program();modified['calls'][1]['parameters']['damping'] = .4
        second = execute_strategy_program(modified, [request], executor)['cell']
        self.assertAlmostEqual(first['predicted'], 20.1)
        self.assertAlmostEqual(second['predicted'], 20.4)
        self.assertEqual([r['used_in_expression'] for r in first['metadata']['internal_calls']], [False, True])
        self.assertNotEqual(first['metadata']['program_digest'], second['metadata']['program_digest'])
        self.assertEqual(len(first['metadata']['internal_calls']), 2)

    def test_program_grammar_rejects_unsafe_unbounded_and_invalid_parameters(self):
        for expr in ['__import__("os")', 'recent.__class__', '[x for x in recent]', 'recent ** 999',
                     'future + recent', 'float("nan")', '1e99', '1' * 1000]:
            with self.subTest(expr=expr[:40]), self.assertRaises(ValueError):
                validate_strategy_program({**program(), 'expression': expr})
        for params in ({'window':100}, {'window':True}, {'unexpected':1}):
            bad=program();bad['calls'][1]['parameters']=params
            with self.assertRaises(ValueError):validate_strategy_program(bad)
        bad=program();bad['calls'][0]['parameters']={'hidden':1}
        with self.assertRaises(ValueError):validate_strategy_program(bad)
        for expr in ['recent / 0', 'horizon > 0']:
            with self.assertRaises((ValueError, ZeroDivisionError)):
                execute_strategy_program({**program(),'expression':expr}, [_request('cell')],
                                         lambda *args:{'cell':{'predicted':20.}})
        with self.assertRaises(ValueError):
            execute_strategy_program(program(), [_request('cell')], lambda *args:{'cell':{'predicted':float('nan')}})

    def test_causal_calculation_refuses_future_missing_and_insufficient_history(self):
        request = _request('cell')
        for times, history in [([9,11],[19,20]), ([9,9],[19,20]), ([8,9],[19,20]), ([9,10],[None,20])]:
            context=deepcopy(request.label_free_context)
            context['history_window']=history;context['causal_provenance']['history_timestamps']=times
            with self.assertRaises(ValueError):
                execute_causal_tool([replace(request,label_free_context=context)], 'causal-trend@1', {'window':2})
        with self.assertRaises(ValueError):execute_causal_tool([request], 'causal-seasonal@1', {'period':24})
        self.assertEqual(execute_causal_tool([request], 'causal-mean@1', {'window':2})['cell']['predicted'],19.5)

    def test_cost_reservation_failed_execution_and_replay_cannot_bypass_budget(self):
        catalog=[{'tool_id':'candidate-program','version':'1','parameters':{},'call_cost':3}]
        events=[]; executions=[]
        def execute(*args):
            executions.append(args)
            return execute_strategy_program(program(),[_request('cell')], lambda *args:{'cell':{'predicted':20.}})
        kwargs=dict(run_id='r',stage_attempt=1,idempotency_key='i',wave_digest='a'*64,
                    sample_ids=['cell'],catalog=catalog,executor=execute,budget=4)
        binding=DshPredictionToolBinding(**kwargs)
        def persist(payload):
            event=SimpleNamespace(event_id='e:'+str(len(events)),payload=payload);events.append(event);return event
        args={'tool_id':'candidate-program','wave_digest':'a'*64,'call_id':'one','parameters':{}}
        response=binding.execute(args,session_id='s',persist=persist)
        self.assertEqual(response['remaining_calls'],1)
        self.assertEqual(events[0].payload['result']['charged_calls'],3)
        binding.execute(args,session_id='s',persist=persist)
        self.assertEqual(len(executions),1)
        replay=DshPredictionToolBinding(**kwargs);replay.restore(events[0])
        replay.execute(args,session_id='s2',persist=persist)
        self.assertEqual(len(executions),1)
        with self.assertRaisesRegex(ValueError,'budget exhausted'):
            replay.execute({**args,'call_id':'two'},session_id='s2',persist=persist)
        forged=deepcopy(events[0].payload);forged['result']['charged_calls']=0;forged['output_digest']=digest(forged['result'])
        with self.assertRaisesRegex(ValueError,'call cost'):validate_tool_event(forged)
        def broken(*args):raise ValueError('invalid calculation')
        failed=DshPredictionToolBinding(**{**kwargs,'executor':broken})
        failed.execute(args,session_id='s',persist=persist)
        self.assertEqual(failed._spent(),3)
        self.assertEqual(events[-1].payload['result']['status'],'failed')

    def test_only_matching_cited_program_proves_activation(self):
        parent=_seed_genome();child=mutate(parent,[program_op()])
        resolution=resolve_mutation_effects(parent,child,[program_op()],current_program_registry())
        metadata={'genome_digest':child.genome_digest,'effect_resolution':resolution,
                  'mutation_operations':[program_op()], 'evolution_genome_canonical_json':canonical_json(child.to_dict())}
        common=dict(scope_digest='b'*64, phase='formal_batch', parameters=child.scientific_program['parameter_overrides'])
        tool={'tool_id':'candidate-program','status':'completed','used_as_evidence':True,
              'program_digest':digest(program()),'dsh_tool_event_id':'event:1'}
        receipt=observe_edit_effects(metadata,[accepted_trace(tools=[tool])],**common)
        self.assertEqual(receipt['status'],'passed')
        for change in ({'program_digest':'f'*64},{'used_as_evidence':False},{'status':'failed'}):
            with self.subTest(change=change):
                self.assertEqual(observe_edit_effects(metadata,[accepted_trace(tools=[{**tool,**change}])],**common)['status'],'failed')

    def test_joint_proposals_and_reflection_survive_next_generation_context(self):
        from tests import test_autonomous_search_reflection_cycle as cycle
        original_task = cycle._task
        original_runtime = cycle._CycleRuntime
        def task(**kwargs):
            base=original_task(**kwargs)
            return replace(base,metadata={**base.metadata,'optimization_protocol':joint_policy.PROTOCOL})
        class Runtime(original_runtime):
            def run_stage(self, request):
                response=super().run_stage(request)
                if request['stage']=='candidate.propose':
                    self_contract=request['request']['context']['mutation_contract']
                    assert self_contract['maximum_operations']==4
                    response['structured']['operations'].append(program_op())
                    response['result_digest']=digest(response['structured'])
                return response
        fixture=cycle.AutonomousSearchReflectionCycleTests()
        self.addCleanup(fixture.doCleanups)
        def compact_analysis(state, analysis):
            # This fixture supplies synthetic ranking evidence to test the wire
            # contract. JointRaceTests independently verifies real race sealing.
            value=analysis.to_dict()
            value.pop('ranking',None)
            return value
        with patch.object(cycle,'_task',task), patch.object(cycle,'_CycleRuntime',Runtime), \
             patch('ecologyrsi_dsh.evolution.batches._adaptive_reflection_analysis',compact_analysis):
            fixture.setUp()
            fixture.test_batch_reflection_is_persisted_and_can_seed_next_search()
        state=fixture.director.state(fixture.run_id)
        reflection=state.reflection_for(0)
        self.assertTrue(all(len(row['mutation_operations'])==2 for row in reflection.canonical_candidate_outcomes))
        self.assertTrue(all(row['mutation_operator_id']==joint_policy.MUTATION_OPERATOR for row in reflection.canonical_candidate_outcomes))
        legacy=reflection.to_dict()
        for row in legacy['canonical_candidate_outcomes']:
            row.pop('mutation_operator_id')
        with self.assertRaisesRegex(ValueError,'frozen operator limit'):
            type(reflection).from_dict(legacy)

    def test_guidance_only_revision_does_not_claim_changed_computation(self):
        parent=mutate(_seed_genome(),[program_op()])
        guidance=program();guidance['planner_prompt']='Assess long horizon errors more carefully.'
        child=mutate(parent,[program_op(guidance)])
        resolution=resolve_mutation_effects(parent,child,[program_op(guidance)],current_program_registry())
        metadata={'genome_digest':child.genome_digest,'effect_resolution':resolution,
                  'mutation_operations':[program_op(guidance)]}
        tool={'tool_id':'candidate-program','status':'completed','used_as_evidence':True,
              'program_digest':digest(guidance),'dsh_tool_event_id':'event:1'}
        receipt=observe_edit_effects(metadata,[accepted_trace(tools=[tool])],scope_digest='b'*64,
                                    phase='formal_batch',parameters=child.scientific_program['parameter_overrides'])
        self.assertNotEqual(receipt['status'],'passed')
        args=([_request('cell')], lambda *args:{'cell':{'predicted':20.}})
        old=execute_strategy_program(program(),*args)['cell']['metadata']
        new=execute_strategy_program(guidance,*args)['cell']['metadata']
        self.assertEqual(old['calculation_digest'],new['calculation_digest'])
        self.assertNotEqual(old['program_digest'],new['program_digest'])

    def test_joint_budget_and_local_bounds_are_distinct_from_legacy(self):
        schedule=OptimizationSchedule.for_joint_run()
        self.assertTrue(schedule.race)
        self.assertEqual(schedule.execution_plan(4,cells_per_origin=9)['run_budget']['total_candidate_origins'],800)
        self.assertEqual(schedule.generation_execution_budget(cells_per_origin=9),
            dict(screening_candidate_origins=50,formal_candidate_origins=50,holdout_candidate_origins=100,
                 total_candidate_origins=200,total_scoring_cells=1800))
        self.assertEqual(schedule.max_local_edits_per_batch,2)
        self.assertEqual(OptimizationSchedule.for_evidence_guided_run().max_local_edits_per_batch,1)
        with self.assertRaises(ValueError):replace(schedule,max_local_edits_per_batch=3)

    def test_agent_owned_program_prediction_persists_and_replays(self):
        # Use production child admission, ledger and binding with a transport double.
        fixture=agent_tests.AgentOwnedPredictionTests();self.addCleanup(fixture.doCleanups)
        def policy(context,call):
            output=call('candidate-program','program')
            self.assertEqual(output['status'],'completed')
            return result(context,value=output['outputs'][0]['predicted'],method='model',refs=['program'])
        adapter,plan,native,ledger=fixture.setup_agent(policy,tool=lambda requests:{r.sample_id:{'predicted':23.} for r in requests})
        value=program();value['calls'][0]['tool_id']='candidate-model';value['expression']='recent + trend / 100'
        plan['decision_context']['candidate_agent_profile']={'strategy_program':value}
        plan['decision_context']['agent_policy']={'inference':{'tool_selection_policy':joint_policy.INFERENCE_POLICY}}
        plan=adapter.plan_batch(plan['decision_context'])
        outcome=fixture.predict(adapter,plan)
        self.assertIsNone(outcome.error)
        self.assertAlmostEqual(outcome.result['predicted'],23.201)
        trace=outcome.result['tool_calls']
        self.assertTrue(any(t.get('program_digest')==digest(value) for t in trace))
        self.assertTrue(any(t.get('candidate_model_consumed') is True for t in trace))
        self.assertEqual(fixture.predict(adapter,plan).result['predicted'],outcome.result['predicted'])
        self.assertEqual(len(native.requests),1)
        self.assertEqual(len(ledger.events_by_kind('run-agent','DshPredictionToolExecuted')),1)


class JointRaceTests(race_tests.EvidenceGuidedRaceTests):
    """Run every existing five-arm seal/restart/replay invariant under the new contract."""
    def setUp(self):
        with patch.object(OptimizationSchedule,'for_evidence_guided_run',OptimizationSchedule.for_joint_run):
            super().setUp()
