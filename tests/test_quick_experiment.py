"""Retired schedules cannot be smuggled into a current frozen comparison."""
import unittest
from dataclasses import replace
from ecologyrsi_dsh.core.state import validate_generation_comparison_binding
from ecologyrsi_dsh.core.trajectory import GenerationHoldout, HoldoutArm
from ecologyrsi_dsh.evaluators.generation_comparison import build_generation_comparison
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from tests.native_fixtures import native_manifest
from tests.test_run_parameter_consistency import small_holdout

class JointProtocolBindingTests(unittest.TestCase):
    def test_current_comparison_is_recomputed_and_old_protocols_rejected(self):
        values=(small_holdout(HoldoutArm.FINALIST_1,.2),small_holdout(HoldoutArm.INCUMBENT,.1))
        scope=values[0].scope
        kwargs=dict(run_id=scope.run_id,generation=0,cohort_digest=scope.cohort_digest,
                    holdout_evaluations=values,require_paired_strict_chain=True)
        decision=build_generation_comparison(**kwargs)
        self.assertEqual(decision.gate_results['experiment_protocol'],'joint_plugin_epoch@2')
        self.assertIsNone(decision.gate_results['certification_selected_arm'])
        holdout=GenerationHoldout(holdout_id='holdout:binding',run_id=scope.run_id,generation=0,
            cohort_digest=scope.cohort_digest,origin_count=50,arm_bindings={
                i.scope.holdout_arm.value:dict(candidate_id=i.scope.candidate_id,
                candidate_revision_id=i.scope.candidate_revision_id) for i in values})
        from ecologyrsi_dsh.evaluators.fitness import FitnessProfile
        profile=FitnessProfile()
        task=native_manifest(metadata={"fitness_profile":profile.to_dict(),"fitness_profile_digest":profile.profile_digest})
        validate_generation_comparison_binding(task,scope.run_id,holdout,{},decision,
            persisted_evaluations={i.scope.holdout_arm:i for i in values})
        with self.assertRaises(ValueError):
            validate_generation_comparison_binding(task,scope.run_id,holdout,{},
                replace(decision,selected_candidate_id='candidate:forged',gate_results=decision.to_dict()['gate_results']),
                persisted_evaluations={i.scope.holdout_arm:i for i in values})
        for protocol in ('quick_adaptive_epoch@1','evidence_guided_epoch@1','joint_plugin_epoch@1'):
            with self.subTest(protocol=protocol),self.assertRaises(ValueError):
                build_generation_comparison(**kwargs,experiment_protocol=protocol)
            changed=OptimizationSchedule.for_new_run().to_dict();changed['protocol']=protocol
            with self.assertRaises(ValueError):OptimizationSchedule.from_dict(changed)
