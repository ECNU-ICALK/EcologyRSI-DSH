"""A previously frozen race winner keeps exact admission after deduplication.

The fixture preserves CandidateMarkedDuplicate and uses real ledger writes and
replay for the checkpoint callbacks; no model/provider requests are made.
"""
import unittest
from dataclasses import replace
from unittest.mock import patch

from ecologyrsi_dsh.application import formal_trajectory, generation_execution
from ecologyrsi_dsh.application.generation_execution import EvaluationSession
from ecologyrsi_dsh.core.models import CandidateStatus, Evaluation, ModelArtifact, digest
from ecologyrsi_dsh.core.finalist_review import finalist_review_evidence
from ecologyrsi_dsh.core.training_effect_evidence import finalist_training_effect_evidence
from ecologyrsi_dsh.core.sample_results import build_sample_results
from ecologyrsi_dsh.core.trajectory import (
    BatchEvaluation, EvaluationPhase, EvaluationScope, FormalBatchArm,
    GenerationHoldout, HoldoutArm, HoldoutEvaluation,
)
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule
from ecologyrsi_dsh.evaluators.generation_comparison import build_generation_comparison
from ecologyrsi_dsh.evaluators.fitness import FitnessProfile
from tests import test_evidence_guided_epoch as race_fixtures
from tests import test_formal_trajectory as trajectory_fixtures
from tests.scoped_checkpoint_fixture import checkpoint as scoped_checkpoint


class DuplicateRaceCheckpointTests(unittest.TestCase):
    _screen = race_fixtures.EvidenceGuidedRaceTests._screen

    def setUp(self):
        race_fixtures.EvidenceGuidedRaceTests.setUp(self)
        self.finalist = self.director.mark_candidate_duplicate(
            self.run_id, self.candidates[0].candidate_id,
            self.candidates[1].candidate_id,
        )
        for candidate in (*self.candidates, self.control):
            self._screen(candidate)
        chosen = generation_execution._prepare_formal_finalists(
            self.services, self.run_id, self.candidates, max_concurrency=1,
        )
        self.assertEqual(chosen[0].candidate_id, self.finalist.candidate_id)
        with patch.object(formal_trajectory, '_local_edit_context',
                          side_effect=trajectory_fixtures.PairedFormalTrajectoryTests._local_context), \
             patch.object(formal_trajectory, '_local_edit_proposal', return_value=replace(
                 trajectory_fixtures.PairedFormalTrajectoryTests._mutate_proposal(),
                 operations=({'op': 'set_bounded_parameter', 'name': 'ridge_alpha', 'value': .7},))):
            for _ in range(2):
                formal_trajectory.execute_next_formal_batch(
                    self.services, self.run_id, self.finalist.candidate_id)
            formal_trajectory.execute_next_local_edit(
                self.services, self.run_id, self.finalist.candidate_id)
        state = self.director.state(self.run_id)
        child = state.revision_activation_for(self.finalist.candidate_id, 0).to_revision_id
        self.batch = self.director.start_formal_batch(
            self.run_id, self.finalist.candidate_id, child, 1)

    def scope(self, arm):
        state = self.director.state(self.run_id)
        revision = (state.batch_comparison_for(self.finalist.candidate_id, 0).champion_after_revision_id
                    if arm is FormalBatchArm.CHAMPION else self.batch.revision_id)
        return EvaluationScope(
            run_id=self.run_id, generation=0, candidate_id=self.finalist.candidate_id,
            candidate_revision_id=revision, phase=EvaluationPhase.FORMAL_BATCH,
            cohort_digest=self.batch.cohort_digest, origin_count=25,
            batch_index=1, formal_batch_arm=arm,
        )

    @staticmethod
    def checkpoint(scope):
        return {**scoped_checkpoint(scope),
                'holdout_arm': scope.holdout_arm.value if scope.holdout_arm else None,
                'inference_replica': scope.inference_replica,
                'sample_count': scope.origin_count * 9}

    def session(self, scope):
        return EvaluationSession(
            self.services, run_id=self.run_id, generation=scope.generation,
            candidate_id=self.finalist.candidate_id,
            proposal_id=self.finalist.proposal_id, scope=scope,
        )

    def exercise_callbacks(self, scope):
        """Cover prepare/start, result batches, usage, progress, and resume."""
        checkpoint = self.checkpoint(scope)
        session = self.session(scope)
        self.assertFalse(session.prepare_checkpoint(checkpoint)['resumed'])
        rows = [dict(sample_id=f'sample:{i}:{target}:{h}', sample_index=i*9+j*3+k+1,
                     target=target, horizon_hours=h, origin_timestamp=i,
                     target_timestamp=i+h, observed=2., predicted=1.5, baseline=1.)
                for i in range(scope.origin_count)
                for j,target in enumerate(('air_temperature','relative_humidity','co2_concentration'))
                for k,h in enumerate((1,6,24))]
        session.record_results(rows[:9])
        session.record_progress(dict(
            role='planner', model_id='test/model', batch_index=1, batch_count=2,
            batch_size=1, completed_samples=1, total_samples=scope.origin_count,
            succeeded_samples=1, failed_samples=0, progress_id=1,
            progress_kind='completed_batch', in_flight_batches=0, queued_batches=0,
        ))
        self.director.pause_run(self.run_id)
        resumed = self.session(scope)
        prepared = resumed.prepare_checkpoint(checkpoint)
        self.assertTrue(prepared['resumed'])
        self.assertEqual(prepared['revision'], session.revision)
        self.assertEqual(len(prepared['rows']), 9)
        self.director.resume_run(self.run_id)
        resumed.record_results(rows[9:])
        return resumed, build_sample_results(self.finalist.candidate_id, rows)

    def complete_paired_batch(self):
        for arm in FormalBatchArm:
            scope = self.scope(arm)
            session, rows = self.exercise_callbacks(scope)
            result = self.evaluator.evaluate_scientific(None, None, None, scope=scope,
                cohort=self.cohorts.adaptation.batches[1].cohort).evaluation
            evaluation = BatchEvaluation(
                evaluation_id=f'test-paired:{arm.value}', scope=scope,
                score=result.score, passed=result.passed, metrics=result.metrics,
                evaluator_digest=digest({'evaluator': result.evaluator_digest}),
            )
            self.director.record_formal_batch_evaluation(
                self.run_id, evaluation,
                sample_results=session.completion_payload(evaluation.evaluation_id, rows),
            )
            with self.assertRaises(ValueError):
                session.record_results([dict(sample_id='late', sample_index=26,
                    target='air_temperature', horizon_hours=1, origin_timestamp=26,
                    target_timestamp=27, observed=2., predicted=1.5, baseline=1.)])
        formal_trajectory.execute_next_formal_batch(
            self.services, self.run_id, self.finalist.candidate_id)

    def holdout_scope(self):
        self.complete_paired_batch()
        state = self.director.state(self.run_id)
        trajectory = state.trajectory_for(self.finalist.candidate_id)
        self.assertEqual(trajectory.status.value, 'completed')
        holdout = self.director.freeze_generation_holdout(
            self.run_id, 0, self.cohorts.holdout.cohort_digest, {
                'finalist_1': {'candidate_id': self.finalist.candidate_id,
                               'candidate_revision_id': trajectory.final_revision_id},
                'incumbent': {'candidate_id': self.control.candidate_id,
                              'candidate_revision_id': state.initial_revision_for(self.control.candidate_id).revision_id},
            },
        )
        return EvaluationScope(
            run_id=self.run_id, generation=0, candidate_id=self.finalist.candidate_id,
            candidate_revision_id=trajectory.final_revision_id,
            phase=EvaluationPhase.HOLDOUT, cohort_digest=holdout.cohort_digest,
            origin_count=50, holdout_arm=HoldoutArm.FINALIST_1,
        )

    def test_both_formal_arms_use_all_callbacks_and_replay_preserves_duplicate(self):
        self.complete_paired_batch()
        state = self.director.replay(self.run_id)
        self.assertEqual(state.candidate(self.finalist.candidate_id).status, CandidateStatus.DUPLICATE)
        self.assertEqual(state.trajectory_for(self.finalist.candidate_id).status.value, 'completed')
        self.assertEqual(sum(e.kind == 'EvaluationSampleResultsStarted' for e in state.events), 2)

    def test_finalist_holdout_comparison_selection_and_completion(self):
        scope = self.holdout_scope()
        session, rows = self.exercise_callbacks(scope)
        evaluation = HoldoutEvaluation(
            evaluation_id='test-holdout:finalist', scope=scope,
            score=.4, passed=False, metrics={}, evaluator_digest='a' * 64,
        )
        self.director.record_holdout_evaluation(
            self.run_id, evaluation,
            sample_results=session.completion_payload(evaluation.evaluation_id, rows),
        )
        state = self.director.state(self.run_id)
        incumbent = HoldoutEvaluation(
            evaluation_id='test-holdout:incumbent', scope=replace(scope,
                candidate_id=self.control.candidate_id,
                candidate_revision_id=state.initial_revision_for(self.control.candidate_id).revision_id,
                holdout_arm=HoldoutArm.INCUMBENT),
            score=.5, passed=False, metrics={}, evaluator_digest='a' * 64,
        )
        self.director.record_holdout_evaluation(self.run_id, incumbent)
        # Publish the exact fitted revision and then its separate durable judge
        # result. Duplicate admission does not bypass independent final review.
        self.assertEqual(state.candidate(self.finalist.candidate_id).status, CandidateStatus.DUPLICATE)
        revision = state.revision(scope.candidate_revision_id)
        program = revision.genome["scientific_program"]
        artifact = ModelArtifact(
            artifact_id="artifact:duplicate-finalist", run_id=self.run_id,
            candidate_id=self.finalist.candidate_id,
            model_id=program["predictor_ref"]["id"], dataset_digest="d" * 64,
            training_partition="calibration_fit", training_rows=50,
            parameters=program["parameter_overrides"],
            candidate_revision_id=scope.candidate_revision_id,
            evaluation_scope_digest=scope.scope_key,
        )
        self.director.record_artifact(artifact)
        scientific = Evaluation(
            evaluation_id=evaluation.evaluation_id, run_id=self.run_id,
            candidate_id=self.finalist.candidate_id, score=evaluation.score,
            passed=evaluation.passed, metrics=evaluation.metrics,
            evaluator_digest=evaluation.evaluator_digest, artifact_digest=artifact.digest,
            candidate_revision_id=scope.candidate_revision_id,
            evaluation_scope=scope.to_dict(),
        )
        self.director.record_evaluation(scientific)
        self.director.record_judgment(replace(scientific, metrics={
            **scientific.metrics, "judge_status": "completed", "judge_accepted": False,
            "judge_model_id": state.task_manifest.metadata["review_model_id"],
            "judge_result_digest": "b" * 64,
        }))
        state = self.director.replay(self.run_id)
        items = (evaluation, incumbent)
        comparison = build_generation_comparison(
            run_id=self.run_id, generation=0, cohort_digest=scope.cohort_digest,
            holdout_evaluations=(evaluation, incumbent),
            incumbent_candidate_id=self.control.candidate_id,
            fitness_profile=FitnessProfile.from_task(state.task_manifest),
            quick_experiment=True, experiment_protocol=self.schedule.protocol, require_paired_strict_chain=True,
            finalist_reviews=finalist_review_evidence(items,
                {item.candidate_id: item for item in state.evaluations},
                state.task_manifest.metadata["review_model_id"]),
            training_effect_evidence=finalist_training_effect_evidence(items,
                screening_events=state.candidate_screening_events,
                batch_evaluations=state.formal_batch_evaluations),
        )
        self.director.record_generation_comparison(self.run_id, comparison)
        self.director.select_generation_champion(self.run_id, 0,
            comparison.selected_revision_id, comparison.comparison_digest)
        self.director.complete_run(self.run_id, termination_reason='fixture_completed')
        state = self.director.replay(self.run_id)
        self.assertEqual(state.candidate(self.finalist.candidate_id).status, CandidateStatus.EVALUATED)
        self.assertEqual(sum(event.kind == 'CandidateMarkedDuplicate'
            and event.payload['candidate_id'] == self.finalist.candidate_id
            for event in state.events), 1)
        self.assertEqual(state.evaluation_for(self.finalist.candidate_id).metrics['judge_status'], 'completed')
        self.assertIsNotNone(state.holdout_evaluation_for(0, HoldoutArm.FINALIST_1))
        self.assertEqual(state.effective_revision_for(0), incumbent.scope.candidate_revision_id)
        self.assertEqual(state.run.status.value, 'completed')

    def test_public_entries_reject_wrong_revision_arm_cohort_and_count(self):
        scope = self.scope(FormalBatchArm.CHAMPION)
        checkpoint = self.checkpoint(scope)
        changes = (
            {'candidate_revision_id': self.batch.revision_id},
            {'cohort_digest': 'f' * 64}, {'sample_count': 24},
            {'formal_batch_index': 0}, {'holdout_arm': 'finalist_1'},
            {'execution_scope_digest': self.scope(FormalBatchArm.CHALLENGER).scope_key},
            {'schema_version': 'ecologyrsi-dsh.sample-checkpoint/1'},
        )
        for change in changes:
            for method in ('prepare_evaluation_sample_checkpoint', 'start_evaluation_sample_results'):
                with self.subTest(change=change, method=method), self.assertRaises((ValueError, KeyError)):
                    kwargs = dict(generation=0, proposal_id=self.finalist.proposal_id,
                        candidate_id=self.finalist.candidate_id,
                        checkpoint={**checkpoint, **change}, scope=scope)
                    if method.startswith('start_'):
                        kwargs['revision'] = 'bad-revision'
                    getattr(self.director, method)(self.run_id, **kwargs)
        self.assertFalse(any(e.kind == 'EvaluationSampleResultsStarted'
                             for e in self.director.replay(self.run_id).events))

    def test_helper_rejects_unfrozen_candidate_status_and_foreign_revision(self):
        state = self.director.state(self.run_id)
        scope = self.scope(FormalBatchArm.CHAMPION)
        checkpoint = self.checkpoint(scope)
        allowed = self.director._sample_checkpoint_candidate_status_allowed
        self.assertTrue(allowed(state, self.finalist, checkpoint, generation=0))
        for candidate in (replace(self.finalist, status=CandidateStatus.FAILED),
                          replace(self.finalist, generation=1),
                          replace(self.candidates[1], status=CandidateStatus.DUPLICATE)):
            self.assertFalse(allowed(state, candidate, checkpoint, generation=0))
        for changed_state in (
            replace(state, formal_selection_events=()),
            replace(state, formal_trajectories=()),
            replace(state, generation_selection_cohorts=()),
        ):
            self.assertFalse(allowed(changed_state, self.finalist, checkpoint, generation=0))
        for changes in ({'candidate_revision_id': 'revision:race:1:r0'},
                        {'formal_batch_index': 99}, {'execution_scope_digest': 'b' * 64},
                        {'sample_count': 24}, {'cohort_digest': self.cohorts.screening.cohort_digest}):
            self.assertFalse(allowed(state, self.finalist, {**checkpoint, **changes}, generation=0))

    def test_holdout_requires_completed_selected_revision_and_finalist_arm(self):
        scope = self.holdout_scope()
        state = self.director.state(self.run_id)
        checkpoint = self.checkpoint(scope)
        allowed = self.director._sample_checkpoint_candidate_status_allowed
        self.assertTrue(allowed(state, self.finalist, checkpoint, generation=0))
        for changes in ({'holdout_arm': 'incumbent'}, {'holdout_arm': 'finalist_2'},
                        {'candidate_revision_id': self.batch.revision_id},
                        {'cohort_digest': 'f' * 64}, {'execution_scope_digest': 'b' * 64},
                        {'sample_count': 49}):
            self.assertFalse(allowed(state, self.finalist, {**checkpoint, **changes}, generation=0))
        self.assertFalse(allowed(replace(state, generation_holdouts=()),
                                 self.finalist, checkpoint, generation=0))

    def test_adopted_historical_duplicate_requires_prior_champion_and_new_control(self):
        original_scope = self.holdout_scope()
        original_state = self.director.state
        revision_id = original_scope.candidate_revision_id
        holdout = GenerationHoldout(
            holdout_id='holdout:next-race', run_id=self.run_id, generation=1,
            cohort_digest='f' * 64, origin_count=50, arm_bindings={
                'incumbent': {'candidate_id': self.finalist.candidate_id,
                              'candidate_revision_id': revision_id},
                'finalist_1': {'candidate_id': self.candidates[1].candidate_id,
                               'candidate_revision_id': 'revision:race:1:r0'},
            },
        )
        planned = replace(self.cohorts, generation=1,
            revision_bindings={**self.cohorts.revision_bindings,
                               self.finalist.candidate_id: revision_id})

        # Only the next generation's orchestration records are a state fixture.
        # Checkpoint callbacks below still use real writes/replay, as in the
        # historical SPAWNED-seed regression.
        def later_state(run_id):
            state = original_state(run_id)
            return replace(state, run=replace(state.run, generation=1),
                generation_selection_cohorts=(*state.generation_selection_cohorts, planned),
                generation_holdouts=(*state.generation_holdouts, holdout),
                effective_revision_bindings=({'generation': 0,
                    'selected_candidate_id': self.finalist.candidate_id,
                    'selected_revision_id': revision_id, 'comparison_digest': 'a' * 64},))

        scope = replace(original_scope, generation=1, cohort_digest=holdout.cohort_digest,
                        holdout_arm=HoldoutArm.INCUMBENT)
        checkpoint = self.checkpoint(scope)
        state = later_state(self.run_id)
        allowed = self.director._sample_checkpoint_candidate_status_allowed
        self.assertTrue(allowed(state, self.finalist, checkpoint, generation=1))
        self.assertFalse(allowed(replace(state, effective_revision_bindings=()),
                                 self.finalist, checkpoint, generation=1))
        self.assertFalse(allowed(replace(state, generation_selection_cohorts=(self.cohorts,)),
                                 self.finalist, checkpoint, generation=1))
        self.assertFalse(allowed(state, self.finalist,
            {**checkpoint, 'holdout_arm': 'finalist_1'}, generation=1))
        self.assertFalse(allowed(state, self.finalist,
            {**checkpoint, 'candidate_revision_id': self.batch.revision_id}, generation=1))
        with patch.object(self.director, 'state', side_effect=later_state):
            self.exercise_callbacks(scope)
        self.assertEqual(self.director.replay(self.run_id).candidate(self.finalist.candidate_id).status,
                         CandidateStatus.DUPLICATE)


class JointDuplicateRaceCheckpointTests(DuplicateRaceCheckpointTests):
    """Exercise the currently deployed joint_plugin_epoch@2 contract too."""

    def setUp(self):
        super().setUp()

    def test_conditional_holdout_recheck_has_separate_checkpoint(self):
        scope = self.holdout_scope()
        revisions = []
        for replica in (0, 1):
            scoped = replace(scope, inference_replica=replica)
            session, rows = self.exercise_callbacks(scoped)
            revisions.append(session.revision)
            evaluation = HoldoutEvaluation(
                evaluation_id=f'test-holdout:{replica}', scope=scoped,
                score=.4, passed=False, metrics={}, evaluator_digest='a' * 64,
            )
            self.director.record_holdout_evaluation(
                self.run_id, evaluation,
                sample_results=session.completion_payload(evaluation.evaluation_id, rows),
            )
        self.assertNotEqual(*revisions)
        state = self.director.replay(self.run_id)
        self.assertIsNotNone(state.holdout_recheck_for(0, HoldoutArm.FINALIST_1))


if __name__ == '__main__':
    unittest.main()
