from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Event as ThreadEvent, Lock
from types import SimpleNamespace
from unittest.mock import patch

from ecologyrsi_dsh.application import formal_trajectory, generation_execution
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.sample_results import build_sample_results
from ecologyrsi_dsh.core.state import (
    project_run_state,
)
from ecologyrsi_dsh.core.trajectory import (
    BatchEvaluation,
    CandidateRevision,
    EvaluationPhase,
    EvaluationScope,
    FormalBatchArm,
    FormalBatchComparison,
    LocalEditOutcome,
    RevisionAdvanceReason,
    TrajectoryStatus,
)
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule


def _sha(label: str) -> str:
    return digest({"label": label})


def _paired_metrics(*, score: float, batch_index: int) -> dict:
    return {
        "objective_aggregation_version": "paired-replay-objective@1",
        "objective_target_weights": {"air_temperature": 1.0},
        "objective_horizons": [1],
        "baseline_profile_digest": _sha("paired-baseline-profile"),
        "evaluation_index_digest": _sha(f"paired-index:{batch_index}"),
        "dataset_digest": _sha("paired-dataset"),
        "split_manifest_digest_sha256": _sha("paired-split"),
        "constraint_violations": 0,
        "sample_execution_coverage_pass": True,
        "sample_execution": {
            "attempted_origin_samples": 10,
            "succeeded_origin_samples": 10,
            "coverage_pass": True,
            "minimum_coverage": 0.95,
            "strict_agent_chain_pass": True,
        },
        "targets": [
            {
                "target": "air_temperature",
                "horizon_hours": 1,
                "skill_score": score,
            }
        ],
    }


def _paired_contract_digest(metrics: dict, evaluator_digest: str) -> str:
    contract = {
        "objective_aggregation_version": metrics[
            "objective_aggregation_version"
        ],
        "objective_target_weights": metrics["objective_target_weights"],
        "objective_horizons": metrics["objective_horizons"],
        "baseline_profile_digest": metrics["baseline_profile_digest"],
        "evaluation_index_digest": metrics["evaluation_index_digest"],
        "dataset_digest": metrics["dataset_digest"],
        "split_manifest_digest_sha256": metrics[
            "split_manifest_digest_sha256"
        ],
    }
    return digest(
        {
            "champion": contract,
            "challenger": contract,
            "champion_evaluator_digest": evaluator_digest,
            "challenger_evaluator_digest": evaluator_digest,
        }
    )


class TrajectoryEventReplayTests(unittest.TestCase):
    def setUp(self):
        pass

    def _paired_comparison_case(self, run_id, *, reject_initial_challenger=False,
                                stop_after_proposal=False, stop_before_pair_evaluation=False):
        from tests import test_formal_trajectory as fixtures
        fixture = fixtures.PairedFormalTrajectoryTests('test_reused_screening_then_paired_child_regression_retains_parent')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.ledger, self.director = fixture.ledger, fixture.director
        services, run_id, cid = fixture.services, fixture.run_id, fixture.cid
        with fixture.patches():
            fixture.run_formal(); fixture.run_formal()
            state = self.director.state(run_id)
            initial = state.initial_revision_for(cid)
            warmup = state.batch_evaluation_for(cid, 0)
            captured = {}
            real_mutation = formal_trajectory._director_mutation
            class BeforeChild(Exception): pass
            def capture_child(services, method, *args, **kwargs):
                if method == 'create_candidate_revision':
                    captured['child'] = args[1]
                    if stop_after_proposal or reject_initial_challenger:
                        raise BeforeChild()
                return real_mutation(services, method, *args, **kwargs)
            with patch.object(formal_trajectory, '_director_mutation', side_effect=capture_child):
                try:
                    formal_trajectory.execute_next_local_edit(services, run_id, cid)
                except BeforeChild:
                    pass
            state = self.director.state(run_id)
            proposal = dict(next(e.payload for e in state.events if e.kind=='LocalEditProposalRecorded' and e.payload['candidate_id']==cid))
            case = SimpleNamespace(run_id=run_id, candidate=state.candidate(cid), initial=initial,
                warmup=warmup, proposal_payload=proposal, legal_child=captured['child'])
            if stop_after_proposal:
                return case
            if reject_initial_challenger:
                decision = dict(proposal_id=proposal['proposal_id'], candidate_id=cid, batch_index=0,
                    outcome='rejected', active_revision_id=initial.revision_id)
                self.director.decide_local_edit(run_id, decision)
                self.director.advance_trajectory_revision(run_id, cid, 0, initial.revision_id,
                    RevisionAdvanceReason.LOCAL_EDIT_REJECTED)
                case.decision_payload = decision
                return case
            case.challenger = captured['child']
            case.decision_payload = dict(next(o for o in state.local_edit_outcomes if o['candidate_id']==cid and o['batch_index']==0))
            batch = self.director.start_formal_batch(run_id, cid, case.challenger.revision_id, 1)
            evaluations=[]
            for arm, revision in ((FormalBatchArm.CHAMPION,initial), (FormalBatchArm.CHALLENGER,case.challenger)):
                scope = EvaluationScope(run_id=run_id,generation=0,candidate_id=cid,
                    candidate_revision_id=revision.revision_id, phase=EvaluationPhase.FORMAL_BATCH,
                    cohort_digest=batch.cohort_digest,origin_count=25,batch_index=1,formal_batch_arm=arm)
                result=fixture.evaluator.evaluate_scientific(None,None,None,scope=scope,
                    cohort=fixture.cohorts.adaptation.batches[1].cohort).evaluation
                evaluation=BatchEvaluation(evaluation_id='paired:'+arm.value,scope=scope,score=result.score,
                    passed=result.passed,metrics=result.metrics,evaluator_digest=digest(result.evaluator_digest))
                evaluations.append(evaluation)
                if not stop_before_pair_evaluation:
                    self.director.record_formal_batch_evaluation(run_id,evaluation)
            case.champion_evaluation,case.challenger_evaluation=evaluations
            if not stop_before_pair_evaluation:
                with patch.object(formal_trajectory,'_director_mutation',side_effect=lambda _s,_m,_r,value:value):
                    case.comparison=formal_trajectory._record_challenger_comparison(services,run_id,*evaluations)
            return case

    def test_same_revision_retry_ignores_timestamp_but_not_identity(self):
        case=self._paired_comparison_case('retry',stop_after_proposal=True)
        first=self.director.create_candidate_revision(case.run_id,case.legal_child)
        recovered=self.director.create_candidate_revision(case.run_id,CandidateRevision.from_dict({**first.to_dict(),'created_at':'2026-10-20T00:00:00Z'}))
        self.assertEqual(first,recovered)
        with self.assertRaises(ValueError):
            self.director.create_candidate_revision(case.run_id,CandidateRevision.from_dict({**first.to_dict(),'behavior_digest':'c'*64}))

    @staticmethod
    def _paired_checkpoint(scope: EvaluationScope) -> dict:
        return {
            "schema_version": "ecologyrsi-dsh.sample-checkpoint/2",
            "candidate_revision_id": scope.candidate_revision_id,
            "evaluation_phase": scope.phase.value,
            "formal_batch_index": scope.batch_index,
            "holdout_arm": None,
            "cohort_digest": scope.cohort_digest,
            "execution_scope_digest": scope.scope_key,
            "sample_cohort_digest": _sha("paired-checkpoint-samples"),
            "execution_context_digest": _sha(scope.scope_key),
            "sample_count": scope.origin_count * 9,
        }


    def test_paired_checkpoint_resumes_and_seals_each_revision_independently(self) -> None:
        case = self._paired_comparison_case(
            "run:paired-checkpoint", stop_before_pair_evaluation=True
        )
        services = SimpleNamespace(director=self.director, ledger=self.ledger)
        checkpoint_revisions = []
        for evaluation in (case.champion_evaluation, case.challenger_evaluation):
            scope = evaluation.scope
            checkpoint = self._paired_checkpoint(scope)
            kwargs = dict(
                run_id=case.run_id, generation=0,
                proposal_id=case.candidate.proposal_id,
                candidate_id=case.candidate.candidate_id, scope=scope,
            )
            callbacks = generation_execution.EvaluationSession(services, **kwargs)
            opened = callbacks.prepare_checkpoint(checkpoint)
            self.assertFalse(opened["resumed"])
            rows = [
                {"sample_id": f"sample:{i}", "sample_index": i + 1, "target": target,
                 "horizon_hours": horizon, "origin_timestamp": origin,
                 "target_timestamp": origin + horizon,
                 "observed": 2.0, "predicted": 1.5, "baseline": 1.0}
                for i, (origin, target, horizon) in enumerate(
                    (origin, target, horizon)
                    for origin in range(scope.origin_count)
                    for target in ("air_temperature", "relative_humidity", "co2_concentration")
                    for horizon in (1, 6, 24)
                )
            ]
            callbacks.record_results(rows[:45])
            self.director.pause_run(case.run_id)
            self.director.resume_run(case.run_id)
            resumed = generation_execution.EvaluationSession(services, **kwargs)
            prepared = resumed.prepare_checkpoint(checkpoint)
            self.assertTrue(prepared["resumed"])
            self.assertEqual(prepared["revision"], opened["revision"])
            self.assertEqual(len(prepared["rows"]), 45)
            resumed.record_results(rows[45:])
            self.director.record_formal_batch_evaluation(
                case.run_id, evaluation,
                sample_results=resumed.completion_payload(
                    evaluation.evaluation_id,
                    build_sample_results(case.candidate.candidate_id, rows),
                ),
            )
            checkpoint_revisions.append(opened["revision"])
        self.assertEqual(len(set(checkpoint_revisions)), 2)
        replayed = project_run_state(self.ledger.events(case.run_id))
        self.assertEqual(replayed.batch_evaluation_for(
            case.candidate.candidate_id, 1, FormalBatchArm.CHAMPION
        ).scope.candidate_revision_id, case.initial.revision_id)
        self.assertEqual(replayed.batch_evaluation_for(
            case.candidate.candidate_id, 1, FormalBatchArm.CHALLENGER
        ).scope.candidate_revision_id, case.challenger.revision_id)
        self.assertEqual(sum(e.kind == "EvaluationSampleResultsRecorded"
                             for e in replayed.events), 2)


    def test_paired_checkpoint_rejects_swapped_arms_and_unfrozen_cohort(self) -> None:
        case = self._paired_comparison_case(
            "run:paired-checkpoint-forged", stop_before_pair_evaluation=True
        )
        champion = case.champion_evaluation.scope
        challenger = case.challenger_evaluation.scope
        for invalid in (
            replace(champion, candidate_revision_id=case.challenger.revision_id),
            replace(challenger, candidate_revision_id=case.initial.revision_id),
            replace(champion, formal_batch_arm=None),
            replace(champion, cohort_digest=_sha("unfrozen-cohort")),
            replace(champion, origin_count=champion.origin_count + 1),
            replace(challenger, batch_index=0),
        ):
            with self.subTest(scope=invalid.to_dict()), self.assertRaises(ValueError):
                self.director.prepare_evaluation_sample_checkpoint(
                    case.run_id, generation=0,
                    candidate_id=case.candidate.candidate_id,
                    proposal_id=case.candidate.proposal_id,
                    checkpoint=self._paired_checkpoint(invalid), scope=invalid,
                )
        self.assertFalse(any(e.kind == "EvaluationSampleResultsStarted"
                             for e in self.director.state(case.run_id).events))


    @staticmethod
    def _forged_promotion(case: SimpleNamespace) -> FormalBatchComparison:
        return FormalBatchComparison.from_dict(
            {
                **case.comparison.to_dict(),
                "safety_gate_passed": False,
                "cell_regression_gate_passed": False,
                "decision": "challenger_promoted",
                "champion_after_revision_id": case.challenger.revision_id,
                "reason": "challenger_improved",
            }
        )


    @staticmethod
    def _forged_local_child(
        case: SimpleNamespace,
        *,
        revision_id: str,
        parent_revision_id: str,
        source_batch_index: int,
    ) -> CandidateRevision:
        return CandidateRevision.from_dict({**{k:v for k,v in case.legal_child.to_dict().items() if k!='revision_digest'}, 'revision_id':revision_id,
            'parent_revision_id':parent_revision_id,'source_batch_index':source_batch_index})


    def test_director_rejects_forged_host_owned_comparison_decision(self) -> None:
        case = self._paired_comparison_case("run:paired-forged-director")

        with self.assertRaisesRegex(ValueError, "host-owned assessment"):
            self.director.record_formal_batch_comparison(
                case.run_id,
                self._forged_promotion(case),
            )

        self.assertIsNone(
            self.director.state(case.run_id).batch_comparison_for(
                case.candidate.candidate_id,
                1,
            )
        )


    def test_raw_ledger_replay_rejects_forged_comparison_decision(self) -> None:
        case = self._paired_comparison_case("run:paired-forged-replay")
        self.director.record_formal_batch_comparison(
            case.run_id,
            case.comparison,
        )
        forged = self._forged_promotion(case)
        forged_events = tuple(
            replace(
                event,
                payload={"comparison": forged.to_dict()},
            )
            if event.kind == "FormalBatchCompared"
            and event.payload["comparison"]["batch_index"] == 1
            else event
            for event in self.ledger.events(case.run_id)
        )

        with self.assertRaisesRegex(ValueError, "host-owned assessment"):
            project_run_state(forged_events)


    def test_paired_child_must_descend_from_selected_champion(self) -> None:
        case = self._paired_comparison_case("run:paired-child-parent-director")
        forged = self._forged_local_child(
            case,
            revision_id="revision:paired:wrong-parent",
            parent_revision_id=case.challenger.revision_id,
            source_batch_index=0,
        )
        before = self.ledger.count(case.run_id)

        with self.assertRaisesRegex(ValueError, "selected champion"):
            self.director.create_candidate_revision(case.run_id, forged)

        self.assertEqual(self.ledger.count(case.run_id), before)


    def test_raw_replay_rejects_paired_child_with_wrong_parent(self) -> None:
        case = self._paired_comparison_case("run:paired-child-parent-replay")
        forged = self._forged_local_child(
            case,
            revision_id="revision:paired:wrong-parent-raw",
            parent_revision_id=case.challenger.revision_id,
            source_batch_index=0,
        )
        self.ledger.append(
            case.run_id,
            "CandidateRevisionCreated",
            {"revision": forged.to_dict()},
            event_id=f"{case.run_id}:forged-wrong-parent",
        )

        with self.assertRaisesRegex(ValueError, "selected champion"):
            project_run_state(tuple(self.ledger.events(case.run_id)))


    def test_paired_child_cannot_arrive_after_local_decision(self) -> None:
        case = self._paired_comparison_case(
            "run:paired-late-child-director",
            reject_initial_challenger=True,
        )
        late_child = self._forged_local_child(
            case,
            revision_id="revision:paired:late-after-rejection",
            parent_revision_id=case.initial.revision_id,
            source_batch_index=0,
        )
        before = self.ledger.count(case.run_id)

        with self.assertRaisesRegex(ValueError, "before local decision"):
            self.director.create_candidate_revision(case.run_id, late_child)

        self.assertEqual(self.ledger.count(case.run_id), before)


    def test_raw_replay_rejects_child_after_local_decision(self) -> None:
        case = self._paired_comparison_case(
            "run:paired-late-child-replay",
            reject_initial_challenger=True,
        )
        late_child = self._forged_local_child(
            case,
            revision_id="revision:paired:late-after-rejection-raw",
            parent_revision_id=case.initial.revision_id,
            source_batch_index=0,
        )
        self.ledger.append(
            case.run_id,
            "CandidateRevisionCreated",
            {"revision": late_child.to_dict()},
            event_id=f"{case.run_id}:forged-late-child",
        )

        with self.assertRaisesRegex(ValueError, "before local decision"):
            project_run_state(tuple(self.ledger.events(case.run_id)))


    def test_concurrent_paired_child_creation_keeps_only_one_challenger(self) -> None:
        case = self._paired_comparison_case(
            "run:paired-concurrent-child",
            stop_after_proposal=True,
        )
        children = tuple(
            self._forged_local_child(
                case,
                revision_id=f"revision:paired:concurrent:{index}",
                parent_revision_id=case.initial.revision_id,
                source_batch_index=0,
            )
            for index in range(2)
        )
        original_state = self.director.state
        barrier = Barrier(2, timeout=3)
        counter_lock = Lock()
        synchronized_calls = 0

        def synchronized_state(run_id: str):
            nonlocal synchronized_calls
            state = original_state(run_id)
            should_wait = False
            if run_id == case.run_id:
                with counter_lock:
                    if synchronized_calls < 2:
                        synchronized_calls += 1
                        should_wait = True
            if should_wait:
                barrier.wait()
            return state

        def create(child: CandidateRevision):
            try:
                return self.director.create_candidate_revision(
                    case.run_id,
                    child,
                )
            except ValueError as exc:
                return exc

        with (
            patch.object(
                self.director,
                "state",
                side_effect=synchronized_state,
            ),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            results = tuple(pool.map(create, children))

        self.assertEqual(
            sum(isinstance(item, CandidateRevision) for item in results),
            1,
        )
        self.assertEqual(sum(isinstance(item, ValueError) for item in results), 1)
        persisted = [
            item
            for item in original_state(case.run_id).candidate_revisions
            if item.candidate_id == case.candidate.candidate_id
            and item.source_batch_index == 0
        ]
        self.assertEqual(len(persisted), 1)


    def test_concurrent_child_and_rejection_cannot_both_commit(self) -> None:
        case = self._paired_comparison_case(
            "run:paired-concurrent-child-rejection",
            stop_after_proposal=True,
        )
        child = self._forged_local_child(
            case,
            revision_id="revision:paired:concurrent-before-rejection",
            parent_revision_id=case.initial.revision_id,
            source_batch_index=0,
        )
        decision_payload = {
            "proposal_id": case.proposal_payload["proposal_id"],
            "candidate_id": case.candidate.candidate_id,
            "batch_index": 0,
            "outcome": LocalEditOutcome.REJECTED.value,
            "active_revision_id": case.initial.revision_id,
        }
        original_state = self.director.state
        original_append = self.ledger.append
        barrier = Barrier(2, timeout=3)
        counter_lock = Lock()
        child_committed = ThreadEvent()
        synchronized_calls = 0

        def synchronized_state(run_id: str):
            nonlocal synchronized_calls
            state = original_state(run_id)
            should_wait = False
            if run_id == case.run_id:
                with counter_lock:
                    if synchronized_calls < 2:
                        synchronized_calls += 1
                        should_wait = True
            if should_wait:
                barrier.wait()
            return state

        def ordered_append(run_id, kind, payload, **kwargs):
            if kind == "LocalEditDecided":
                self.assertTrue(child_committed.wait(3))
            event = original_append(run_id, kind, payload, **kwargs)
            if kind == "CandidateRevisionCreated":
                child_committed.set()
            return event

        def create_child():
            try:
                return self.director.create_candidate_revision(
                    case.run_id,
                    child,
                )
            except ValueError as exc:
                return exc

        def reject_child():
            try:
                return self.director.decide_local_edit(
                    case.run_id,
                    decision_payload,
                )
            except ValueError as exc:
                return exc

        with (
            patch.object(
                self.director,
                "state",
                side_effect=synchronized_state,
            ),
            patch.object(self.ledger, "append", side_effect=ordered_append),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            futures = (pool.submit(create_child), pool.submit(reject_child))
            results = tuple(future.result(timeout=5) for future in futures)

        self.assertEqual(
            sum(isinstance(item, CandidateRevision) for item in results),
            1,
        )
        self.assertEqual(sum(isinstance(item, ValueError) for item in results), 1)
        state = original_state(case.run_id)
        self.assertIsNone(
            next(
                (
                    item
                    for item in state.local_edit_outcomes
                    if item.get("candidate_id") == case.candidate.candidate_id
                    and item.get("batch_index") == 0
                ),
                None,
            )
        )


    def test_completed_paired_proposal_retry_is_idempotent(self) -> None:
        case = self._paired_comparison_case("run:paired-proposal-retry-completed")
        self.director.record_formal_batch_comparison(
            case.run_id,
            case.comparison,
        )
        self.director.complete_formal_trajectory(
            case.run_id,
            case.candidate.candidate_id,
            case.comparison.champion_after_revision_id,
        )
        existing = next(
            event
            for event in self.ledger.events(case.run_id)
            if event.kind == "LocalEditProposalRecorded"
        )
        before = self.ledger.count(case.run_id)

        try:
            retried = self.director.record_local_edit_proposal(
                case.run_id,
                case.proposal_payload,
            )
        except ValueError as exc:
            self.fail(f"exact completed proposal retry was rejected: {exc}")

        self.assertEqual(retried.event_id, existing.event_id)
        self.assertEqual(self.ledger.count(case.run_id), before)


    def test_completed_paired_decision_retry_is_idempotent(self) -> None:
        case = self._paired_comparison_case("run:paired-decision-retry-completed")
        self.director.record_formal_batch_comparison(
            case.run_id,
            case.comparison,
        )
        self.director.complete_formal_trajectory(
            case.run_id,
            case.candidate.candidate_id,
            case.comparison.champion_after_revision_id,
        )
        existing = next(
            event
            for event in self.ledger.events(case.run_id)
            if event.kind == "LocalEditDecided"
        )
        before = self.ledger.count(case.run_id)

        try:
            retried = self.director.decide_local_edit(
                case.run_id,
                case.decision_payload,
            )
        except ValueError as exc:
            self.fail(f"exact completed decision retry was rejected: {exc}")

        self.assertEqual(retried.event_id, existing.event_id)
        self.assertEqual(self.ledger.count(case.run_id), before)


    def test_raw_replay_accepts_exact_local_retries_after_completion(self) -> None:
        case = self._paired_comparison_case("run:paired-raw-retry-completed")
        self.director.record_formal_batch_comparison(
            case.run_id,
            case.comparison,
        )
        self.director.complete_formal_trajectory(
            case.run_id,
            case.candidate.candidate_id,
            case.comparison.champion_after_revision_id,
        )
        proposal_event = next(
            event
            for event in self.ledger.events(case.run_id)
            if event.kind == "LocalEditProposalRecorded"
        )
        decision_event = next(
            event
            for event in self.ledger.events(case.run_id)
            if event.kind == "LocalEditDecided"
        )
        self.ledger.append(
            case.run_id,
            proposal_event.kind,
            proposal_event.payload,
            event_id=f"{proposal_event.event_id}:late-retry",
        )
        self.ledger.append(
            case.run_id,
            decision_event.kind,
            decision_event.payload,
            event_id=f"{decision_event.event_id}:late-retry",
        )

        try:
            replayed = project_run_state(
                tuple(self.ledger.events(case.run_id))
            )
        except ValueError as exc:
            self.fail(f"exact raw local retry was rejected: {exc}")

        self.assertIs(
            replayed.trajectory_for(case.candidate.candidate_id).status,
            TrajectoryStatus.COMPLETED,
        )


    def test_raw_replay_rejects_local_proposal_operation_bypasses(self) -> None:
        case = self._paired_comparison_case(
            "run:paired-proposal-operation-bypass",
            stop_after_proposal=True,
        )
        state = self.director.state(case.run_id)
        schedule = OptimizationSchedule.from_dict(
            state.task_manifest.metadata["optimization_schedule"]
        )
        invalid_operations = (
            [],
            [
                {"op": "forged"}
                for _ in range(schedule.max_local_edits_per_batch + 1)
            ],
        )

        for operations in invalid_operations:
            with self.subTest(operation_count=len(operations)):
                forged_events = tuple(
                    replace(
                        event,
                        payload={**event.payload, "proposal": {**event.payload["proposal"], "operations": operations}},
                    )
                    if event.kind == "LocalEditProposalRecorded"
                    else event
                    for event in self.ledger.events(case.run_id)
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "proposal evidence",
                ):
                    project_run_state(forged_events)


    def test_paired_mutate_proposal_cannot_be_marked_kept(self) -> None:
        case = self._paired_comparison_case(
            "run:paired-mutate-kept-director",
            stop_after_proposal=True,
        )
        kept_payload = {
            "proposal_id": case.proposal_payload["proposal_id"],
            "candidate_id": case.candidate.candidate_id,
            "batch_index": 0,
            "outcome": LocalEditOutcome.KEPT.value,
            "active_revision_id": case.initial.revision_id,
        }
        before = self.ledger.count(case.run_id)

        with self.assertRaisesRegex(ValueError, "proposal decision"):
            self.director.decide_local_edit(case.run_id, kept_payload)

        self.assertEqual(self.ledger.count(case.run_id), before)


    def test_raw_replay_rejects_mutate_to_kept_outcome(self) -> None:
        case = self._paired_comparison_case(
            "run:paired-mutate-kept-replay",
            stop_after_proposal=True,
        )
        self.ledger.append(
            case.run_id,
            "LocalEditDecided",
            {
                "proposal_id": case.proposal_payload["proposal_id"],
                "candidate_id": case.candidate.candidate_id,
                "batch_index": 0,
                "outcome": LocalEditOutcome.KEPT.value,
                "active_revision_id": case.initial.revision_id,
            },
            event_id=f"{case.run_id}:forged-mutate-kept",
        )

        with self.assertRaisesRegex(ValueError, "proposal decision"):
            project_run_state(tuple(self.ledger.events(case.run_id)))


    def test_completion_rejects_child_bound_to_rejected_outcome(self) -> None:
        case = self._paired_comparison_case(
            "run:paired-orphan-child-completion"
        )
        self.director.record_formal_batch_comparison(
            case.run_id,
            case.comparison,
        )
        state = self.director.state(case.run_id)
        forged_outcomes = tuple(
            {
                **item,
                "outcome": LocalEditOutcome.REJECTED.value,
                "active_revision_id": case.initial.revision_id,
            }
            if item.get("candidate_id") == case.candidate.candidate_id
            and item.get("batch_index") == 0
            else item
            for item in state.local_edit_outcomes
        )
        forged_activations = tuple(
            replace(
                item,
                to_revision_id=case.initial.revision_id,
                reason=RevisionAdvanceReason.LOCAL_EDIT_REJECTED,
            )
            if item.candidate_id == case.candidate.candidate_id
            and item.batch_index == 0
            else item
            for item in state.trajectory_revision_activations
        )
        forged_state = replace(
            state,
            local_edit_outcomes=forged_outcomes,
            trajectory_revision_activations=forged_activations,
        )

        with (
            patch.object(self.director, "state", return_value=forged_state),
            self.assertRaisesRegex(ValueError, "child/outcome binding"),
        ):
            self.director.complete_formal_trajectory(
                case.run_id,
                case.candidate.candidate_id,
                case.comparison.champion_after_revision_id,
            )


    def test_completion_rejects_mutate_proposal_bound_to_kept_outcome(self) -> None:
        case = self._paired_comparison_case(
            "run:paired-mutate-kept-completion"
        )
        self.director.record_formal_batch_comparison(
            case.run_id,
            case.comparison,
        )
        state = self.director.state(case.run_id)
        forged_outcomes = tuple(
            {
                **item,
                "outcome": LocalEditOutcome.KEPT.value,
                "active_revision_id": case.initial.revision_id,
            }
            if item.get("candidate_id") == case.candidate.candidate_id
            and item.get("batch_index") == 0
            else item
            for item in state.local_edit_outcomes
        )
        forged_activations = tuple(
            replace(
                item,
                to_revision_id=case.initial.revision_id,
                reason=RevisionAdvanceReason.KEPT,
            )
            if item.candidate_id == case.candidate.candidate_id
            and item.batch_index == 0
            else item
            for item in state.trajectory_revision_activations
        )
        forged_state = replace(
            state,
            candidate_revisions=tuple(
                item
                for item in state.candidate_revisions
                if item.revision_id != case.challenger.revision_id
            ),
            local_edit_outcomes=forged_outcomes,
            trajectory_revision_activations=forged_activations,
        )

        with (
            patch.object(self.director, "state", return_value=forged_state),
            self.assertRaisesRegex(ValueError, "child/outcome binding"),
        ):
            self.director.complete_formal_trajectory(
                case.run_id,
                case.candidate.candidate_id,
                case.comparison.champion_after_revision_id,
            )


    def test_paired_final_batch_rejects_local_proposal_and_child(self) -> None:
        proposal_case = self._paired_comparison_case(
            "run:paired-final-proposal-director"
        )
        self.director.record_formal_batch_comparison(
            proposal_case.run_id,
            proposal_case.comparison,
        )
        final_payload = {
                            'proposal_id': "local:paired:final",
                            'candidate_id': proposal_case.candidate.candidate_id,
                            'batch_index': 1,
                            'evidence_scope_digest': proposal_case.challenger_evaluation.scope.scope_key,
                            "proposal": {"schema_version": "ecology-local-edit@1", "decision": "keep", "operations": [], "evidence_refs": ["batch:score"], "expected_effect_cells": [], "risk_cells": []}
                        }
        before = self.ledger.count(proposal_case.run_id)
        with self.assertRaisesRegex(ValueError, "final batch"):
            self.director.record_local_edit_proposal(
                proposal_case.run_id,
                final_payload,
            )
        self.assertEqual(self.ledger.count(proposal_case.run_id), before)

        child_case = self._paired_comparison_case(
            "run:paired-final-child-director"
        )
        self.director.record_formal_batch_comparison(
            child_case.run_id,
            child_case.comparison,
        )
        final_child = self._forged_local_child(
            child_case,
            revision_id="revision:paired:post-final",
            parent_revision_id=(
                child_case.comparison.champion_after_revision_id
            ),
            source_batch_index=1,
        )
        before = self.ledger.count(child_case.run_id)
        with self.assertRaisesRegex(ValueError, "final batch"):
            self.director.create_candidate_revision(
                child_case.run_id,
                final_child,
            )
        self.assertEqual(self.ledger.count(child_case.run_id), before)


    def test_raw_replay_rejects_final_batch_local_artifacts(self) -> None:
        proposal_case = self._paired_comparison_case(
            "run:paired-final-proposal-replay"
        )
        self.director.record_formal_batch_comparison(
            proposal_case.run_id,
            proposal_case.comparison,
        )
        self.ledger.append(
            proposal_case.run_id,
            "LocalEditProposalRecorded",
            {
                'proposal_id': "local:paired:final:raw",
                'candidate_id': proposal_case.candidate.candidate_id,
                'batch_index': 1,
                'evidence_scope_digest': proposal_case.challenger_evaluation.scope.scope_key,
                "proposal": {"schema_version": "ecology-local-edit@1", "decision": "keep", "operations": [], "evidence_refs": ["batch:score"], "expected_effect_cells": [], "risk_cells": []}
            },
            event_id=f"{proposal_case.run_id}:forged-final-proposal",
        )
        with self.assertRaisesRegex(ValueError, "final batch"):
            project_run_state(tuple(self.ledger.events(proposal_case.run_id)))

        child_case = self._paired_comparison_case(
            "run:paired-final-child-replay"
        )
        self.director.record_formal_batch_comparison(
            child_case.run_id,
            child_case.comparison,
        )
        final_child = self._forged_local_child(
            child_case,
            revision_id="revision:paired:post-final-raw",
            parent_revision_id=(
                child_case.comparison.champion_after_revision_id
            ),
            source_batch_index=1,
        )
        self.ledger.append(
            child_case.run_id,
            "CandidateRevisionCreated",
            {"revision": final_child.to_dict()},
            event_id=f"{child_case.run_id}:forged-final-child",
        )
        with self.assertRaisesRegex(ValueError, "final batch"):
            project_run_state(tuple(self.ledger.events(child_case.run_id)))


    def test_paired_evaluations_and_comparison_replay_by_explicit_arm(self) -> None:
        case = self._paired_comparison_case("run:paired-replay")
        comparison = case.comparison
        run_id = case.run_id
        candidate = case.candidate
        initial = case.initial
        champion_evaluation = case.champion_evaluation
        challenger_evaluation = case.challenger_evaluation
        mismatched = FormalBatchComparison.from_dict(
            {**comparison.to_dict(), "cohort_digest": _sha("wrong-cohort")}
        )
        with self.assertRaisesRegex(ValueError, "cohort"):
            self.director.record_formal_batch_comparison(run_id, mismatched)
        bad_digest = FormalBatchComparison.from_dict(
            {
                **comparison.to_dict(),
                "champion_evaluation_digest": _sha("wrong-evaluation"),
            }
        )
        with self.assertRaisesRegex(ValueError, "digest"):
            self.director.record_formal_batch_comparison(run_id, bad_digest)

        recorded = self.director.record_formal_batch_comparison(
            run_id,
            comparison,
        )
        recovered = self.director.record_formal_batch_comparison(
            run_id,
            comparison,
        )
        self.assertEqual(recovered, recorded)
        conflicting = FormalBatchComparison.from_dict(
            {**comparison.to_dict(), "reason": "below_practical_delta"}
        )
        with self.assertRaisesRegex(ValueError, "different comparison"):
            self.director.record_formal_batch_comparison(run_id, conflicting)

        replayed = self.director.replay(run_id)
        self.assertEqual(
            replayed.batch_evaluation_for(
                candidate.candidate_id,
                1,
                FormalBatchArm.CHAMPION,
            ),
            champion_evaluation,
        )
        self.assertEqual(
            replayed.batch_evaluation_for(
                candidate.candidate_id,
                1,
                FormalBatchArm.CHALLENGER,
            ),
            challenger_evaluation,
        )
        self.assertEqual(
            replayed.batch_evaluation_for(candidate.candidate_id, 1),
            challenger_evaluation,
        )
        self.assertEqual(
            replayed.batch_comparison_for(candidate.candidate_id, 1),
            comparison,
        )
        self.assertEqual(
            replayed.trajectory_champion_revision_id(candidate.candidate_id),
            initial.revision_id,
        )
        self.assertEqual(
            sum(event.kind == "FormalBatchCompared" for event in replayed.events),
            2,
        )


if __name__ == "__main__":
    unittest.main()
