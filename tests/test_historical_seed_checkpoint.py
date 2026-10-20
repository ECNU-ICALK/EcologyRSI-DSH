"""Historical SPAWNED controls must traverse every durable checkpoint callback."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from ecologyrsi_dsh.core.models import CandidateStatus, RunStatus
from ecologyrsi_dsh.core.trajectory import GenerationHoldout, EvaluationScope, EvaluationPhase, HoldoutArm
from ecologyrsi_dsh.core.sample_results import build_sample_results
from ecologyrsi_dsh.application.generation_execution import EvaluationSession
from tests import test_evidence_guided_epoch as race_fixtures
from tests import test_scoped_sample_execution as scoped_fixtures


class HistoricalSeedCheckpointTests(unittest.TestCase):
    setUp = race_fixtures.EvidenceGuidedRaceTests.setUp

    def later_fixture(self):
        original_state = self.director.state
        seed = self.control
        seed_revision = original_state(self.run_id).initial_revision_for(seed.candidate_id).revision_id
        holdout = GenerationHoldout(holdout_id='holdout:later', run_id=self.run_id,
            generation=1, cohort_digest='f'*64, origin_count=50, arm_bindings={
                'incumbent': {'candidate_id':seed.candidate_id,'candidate_revision_id':seed_revision},
                'finalist_1': {'candidate_id':self.candidates[0].candidate_id,'candidate_revision_id':'revision:race:0:r0'}})
        # Freeze only the orchestration fixture; all checkpoint writes and their
        # event replay below use the real director, ledger and callbacks.
        def later_state(run_id):
            state = original_state(run_id)
            return replace(state, run=replace(state.run, generation=1), generation_holdouts=(holdout,))
        self.addCleanup(patch.stopall)
        patch.object(self.director, 'state', side_effect=later_state).start()
        scope = EvaluationScope(run_id=self.run_id, generation=1, candidate_id=seed.candidate_id,
            candidate_revision_id=seed_revision, phase=EvaluationPhase.HOLDOUT,
            cohort_digest=holdout.cohort_digest, origin_count=50, holdout_arm=HoldoutArm.INCUMBENT)
        checkpoint = {**scoped_fixtures.ScopedSampleExecutionTests._checkpoint(scope), 'holdout_arm':'incumbent', 'sample_count':50}
        return seed, scope, checkpoint

    def session(self, seed, scope):
        return EvaluationSession(self.services, run_id=self.run_id, generation=1,
            proposal_id=seed.proposal_id, candidate_id=seed.candidate_id, scope=scope)

    def test_seed_all_callbacks_pause_resume_and_completion(self):
        seed, scope, checkpoint = self.later_fixture()
        session = self.session(seed, scope)
        self.assertFalse(session.prepare_checkpoint(checkpoint)['resumed'])
        rows = [dict(sample_id=f'sample:{i}',sample_index=i+1,target='air_temperature',
            horizon_hours=1,origin_timestamp=i,target_timestamp=i+1,observed=2.,predicted=1.5,baseline=1.)
            for i in range(50)]
        session.record_results(rows[:9])
        session.record_model_usage([dict(call_id='call:seed',logical_call_digest='c'*64,
            role='planner',model_id='test/model',outcome='succeeded',usage_reported=True,
            http_attempts=1,prompt_tokens=10,completion_tokens=2,total_tokens=12)])
        session.record_progress(dict(role='planner',model_id='test/model',batch_index=1,batch_count=2,
            batch_size=9,completed_samples=9,total_samples=50,succeeded_samples=9,failed_samples=0,
            progress_id=1,progress_kind='completed_batch',in_flight_batches=0,queued_batches=0))
        self.director.pause_run(self.run_id)
        resumed = self.session(seed, scope)
        prepared = resumed.prepare_checkpoint(checkpoint)
        self.assertTrue(prepared['resumed'])
        self.assertEqual(len(prepared['rows']), 9)
        self.assertEqual(prepared['revision'], session.revision)
        self.director.resume_run(self.run_id)
        resumed.record_results(rows[9:])
        completion = resumed.completion_payload('evaluation:seed', build_sample_results(seed.candidate_id, rows))
        self.director._validate_sample_results_completion_identity(self.director.state(self.run_id),
            run_id=self.run_id,candidate_id=seed.candidate_id,evaluation_id="evaluation:seed",payload=completion)
        self.ledger.append(self.run_id,'EvaluationSampleResultsRecorded',completion)
        replayed = self.director.replay(self.run_id)
        self.assertEqual(replayed.candidate(seed.candidate_id).status, CandidateStatus.SPAWNED)
        self.assertEqual(sum(e.kind == 'EvaluationSampleResultsStarted' for e in replayed.events),1)
        self.assertEqual(sum(e.kind == 'ModelUsageRecorded' for e in replayed.events),1)
        with self.assertRaises(ValueError):
            resumed.record_results(rows[:1])

    def test_start_and_prepare_reject_wrong_frozen_contract(self):
        seed, scope, checkpoint = self.later_fixture()
        changes = ({'candidate_revision_id':'revision:race:1:r0'}, {'cohort_digest':'e'*64},
            {'holdout_arm':'finalist_1'}, {'sample_count':49}, {'execution_scope_digest':'d'*64},
            {'schema_version':'ecologyrsi-dsh.sample-checkpoint/1'})
        for changed in changes:
            for method in ('prepare_evaluation_sample_checkpoint','start_evaluation_sample_results'):
                with self.subTest(changed=changed,method=method), self.assertRaises((ValueError,KeyError)):
                    kwargs = dict(generation=1,proposal_id=seed.proposal_id,candidate_id=seed.candidate_id,
                        checkpoint={**checkpoint,**changed},scope=scope)
                    if method.startswith('start_'): kwargs['revision']='bad-revision'
                    getattr(self.director,method)(self.run_id,**kwargs)
        self.assertFalse(any(e.kind=='EvaluationSampleResultsStarted' for e in self.director.replay(self.run_id).events))

    def test_status_generation_and_arm_are_not_broadly_relaxed(self):
        seed, scope, checkpoint = self.later_fixture()
        state = self.director.state(self.run_id)
        allowed = self.director._sample_checkpoint_candidate_status_allowed
        self.assertTrue(allowed(state,seed,checkpoint,generation=1))
        for candidate in (replace(seed,generation=2), replace(seed,status=CandidateStatus.FAILED),
                          replace(seed,status=CandidateStatus.DUPLICATE), self.candidates[0]):
            self.assertFalse(allowed(state,candidate,checkpoint,generation=1))
        for changed in ({'evaluation_phase':'formal_batch'}, {'holdout_arm':'finalist_1'},
                        {'candidate_revision_id':'unbound'}):
            self.assertFalse(allowed(state,seed,{**checkpoint,**changed},generation=1))
        self.assertFalse(allowed(replace(state,generation_holdouts=()),seed,checkpoint,generation=1))


if __name__ == '__main__':
    unittest.main()
