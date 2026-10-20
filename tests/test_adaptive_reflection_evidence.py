"""Current joint reflection preserves measured revisions and excludes raw data."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest

from ecologyrsi_dsh.application.generation_execution import _build_adaptive_analysis, _local_edit_trajectory_evidence
from ecologyrsi_dsh.core.models import CandidateStatus, CandidateRole, canonical_json, digest
from ecologyrsi_dsh.core.trajectory import CandidateRevision, RevisionStatus, HoldoutArm, FormalBatchComparisonDecision, RevisionAdvanceReason
from ecologyrsi_dsh.evolution.batches import _adaptive_reflection_analysis, _canonical_candidate_outcomes
from ecologyrsi_dsh.evolution.context import safe_aggregate_feedback
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evaluators.generation_comparison import build_generation_comparison
from tests.native_fixtures import native_manifest
from tests.test_run_parameter_consistency import small_holdout


class AdaptiveReflectionEvidenceTests(unittest.TestCase):
    def fixture(self, *, retain=False):
        pair=(small_holdout(HoldoutArm.FINALIST_1,.09 if retain else .2),small_holdout(HoldoutArm.INCUMBENT,.1))
        comparison=build_generation_comparison(run_id=pair[0].scope.run_id,generation=0,
            cohort_digest=pair[0].scope.cohort_digest,holdout_evaluations=pair,require_paired_strict_chain=True)
        finalist=pair[0].scope.candidate_id
        r0='revision:initial'
        final=pair[0].scope.candidate_revision_id
        revisions={}
        for item in pair:
            scope=item.scope;genome={'test_fixture':scope.candidate_revision_id}
            revisions[scope.candidate_revision_id]=CandidateRevision(revision_id=scope.candidate_revision_id,
                run_id=scope.run_id,generation=0,candidate_id=scope.candidate_id,genome=genome,
                genome_digest=digest(genome),behavior_digest=digest({'behavior':genome}),
                mutation_digest=digest({'mutation':genome}),status=RevisionStatus.FINAL)
        candidates=tuple(NS(candidate_id=finalist if i==0 else f'candidate:{i}',
            proposal_id=f'proposal:{i}',generation=0,slot_index=i,role=CandidateRole.SEARCH,
            status=CandidateStatus.EVALUATED if i==0 else CandidateStatus.SCREENED_OUT) for i in range(4))
        proposals={c.proposal_id:NS(metadata={'candidate_direction_id':f'direction:{i}',
            'candidate_direction_digest':digest({'direction':i}),
            'mutation_operations':[{'op':'set_bounded_parameter','name':'ridge_alpha','value':.2}],
            'behavior_digest':digest({'behavior':i})}) for i,c in enumerate(candidates)}
        comparisons=tuple(NS(candidate_id=finalist,batch_index=i,
            champion_before_revision_id=r0,challenger_revision_id=r0 if i==0 else final,
            champion_after_revision_id=r0 if i==0 else final,
            decision=FormalBatchComparisonDecision.INITIAL_CHAMPION if i==0 else FormalBatchComparisonDecision.CHALLENGER_PROMOTED,
            reason='initial_champion' if i==0 else 'challenger_improved',score_delta=0 if i==0 else .02,
            minimum_score_delta=.005,safety_gate_passed=True,cell_regression_gate_passed=True,
            raw_predictions=[123],labels=[456],rationale='private-model-prose',created_at='private-time') for i in range(2))
        state=NS(run=NS(run_id=pair[0].scope.run_id),task_manifest=native_manifest(),candidates=candidates,
            formal_batches=tuple(NS(candidate_id=finalist,batch_index=i,revision_id=r0 if i==0 else final) for i in range(2)),
            formal_batch_comparisons=comparisons,
            local_edit_proposals=({'candidate_id':finalist,'batch_index':0,'proposal':{'decision':'mutate',
                'operations':[{'op':'set_bounded_parameter','name':'history_steps','value':'private-mutation-value'}]}},),
            local_edit_outcomes=({'candidate_id':finalist,'batch_index':0,'outcome':'applied','active_revision_id':final},),
            trajectory_revision_activations=(NS(candidate_id=finalist,batch_index=0,from_revision_id=r0,
                to_revision_id=final,reason=RevisionAdvanceReason.LOCAL_EDIT_APPLIED),),
            proposal=lambda pid:proposals[pid],revision=lambda rid:revisions[rid],
            trajectory_for=lambda cid:NS(initial_revision_id=r0,final_revision_id=final,batch_count=2) if cid==finalist else None,
            comparison_for=lambda gen:comparison if gen==0 else None,
            screening_for=lambda gen,cid:NS(payload={'score':.08,'passed':False,'constraint_violations':0}) if cid!=finalist else None)
        return state,comparison,candidates,pair

    def test_current_analysis_and_reflection_preserve_gate_revision_and_redaction(self):
        for retain in (False,True):
            with self.subTest(retain=retain):
                state,comparison,candidates,pair=self.fixture(retain=retain)
                analysis=_build_adaptive_analysis(state,0,comparison,(candidates[0],),pair[1].scope.candidate_id)
                self.assertEqual(analysis.eligible_count,0 if retain else 1)
                self.assertEqual(analysis.selected_candidate_id,None if retain else candidates[0].candidate_id)
                if retain: self.assertIn('1 个 finalist',analysis.next_generation_focus)
                self.assertNotIn('Top-2',analysis.selection_reason+analysis.next_generation_focus)
                self.assertNotIn('3 臂',analysis.selection_reason)
                self.assertEqual(len(analysis.ranking),4)
                finalist=next(r for r in analysis.ranking if r['candidate_id']==candidates[0].candidate_id)
                self.assertEqual(finalist['final_revision_id'],pair[0].scope.candidate_revision_id)
                self.assertEqual(len(finalist['comparison_gate']['cell_deltas']),9)
                local=finalist['local_edit_evidence']
                self.assertEqual(local['comparison_count'],2)
                self.assertEqual(local['included_batch_count'],1)
                self.assertEqual(len(local['revision_chain']),3)
                outcomes=_canonical_candidate_outcomes(state,generation=0,analysis=analysis)
                self.assertEqual(len(outcomes),4)
                reflection=_adaptive_reflection_analysis(state,analysis)
                self.assertEqual(reflection,_adaptive_reflection_analysis(state,analysis))
                evidence=reflection['adaptive_epoch_evidence']
                self.assertEqual(evidence['selected']['revision_id'],comparison.selected_revision_id)
                self.assertEqual(evidence['incumbent']['revision_id'],pair[1].scope.candidate_revision_id)
                self.assertEqual(len(evidence['candidate_results']),4)
                serialized=canonical_json(reflection)
                for forbidden in ('raw_predictions','labels','created_at','rationale','private-model-prose','private-mutation-value','private-time'):
                    self.assertNotIn(forbidden,serialized)
                safe_aggregate_feedback(reflection,name='current joint reflection')

    def test_incomplete_history_or_unmeasured_final_revision_cannot_feed_reflection(self):
        state,_,candidates,pair=self.fixture()
        args=(candidates[0].candidate_id,pair[0].scope.candidate_revision_id)
        for altered in (
            NS(**{**vars(state),'formal_batches':state.formal_batches[:1]}),
            NS(**{**vars(state),'formal_batch_comparisons':state.formal_batch_comparisons[:1]}),
            NS(**{**vars(state),'local_edit_proposals':()}),
            NS(**{**vars(state),'local_edit_proposals':(*state.local_edit_proposals,
                {**state.local_edit_proposals[0],'batch_index':1})}),
        ):
            with self.assertRaises(RuntimeError):_local_edit_trajectory_evidence(altered,*args)
        with self.assertRaises(RuntimeError):_local_edit_trajectory_evidence(state,args[0],'revision:unmeasured')
