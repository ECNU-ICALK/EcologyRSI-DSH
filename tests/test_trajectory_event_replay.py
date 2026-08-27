from __future__ import annotations

import unittest
from types import SimpleNamespace

from ecologyrsi_dsh import EventLedger, EvolutionDirector, FakeDSHAdapter, TaskManifest
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.screening import screening_cohort_digest
from ecologyrsi_dsh.core.trajectory import (
    BatchEvaluation,
    CandidateRevision,
    EvaluationPhase,
    EvaluationScope,
    GenerationComparison,
    HoldoutArm,
    HoldoutEvaluation,
    RevisionAdvanceReason,
    RevisionStatus,
)
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.evaluators.epoch_cohorts import (
    plan_generation_selection_cohorts,
    plan_run_adaptation_cohort,
)
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule


def _sha(label: str) -> str:
    return digest({"label": label})


class TrajectoryEventReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = EventLedger()
        self.director = EvolutionDirector(
            self.ledger, FakeDSHAdapter(max_proposals=20)
        )
        schedule = OptimizationSchedule.from_dict(
            {
                **OptimizationSchedule.default().to_dict(),
                "formal_origin_count_per_finalist": 100,
                "local_batch_origin_count": 10,
            }
        )
        self.schedule = schedule
        task = TaskManifest(
            task_id="trajectory-replay",
            objective="exercise adaptive trajectory replay",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={
                "max_generations": 1,
                "candidates_per_generation": 4,
                "max_candidates": 4,
            },
            seed=7,
            metadata={
                "episode_id": "episode:trajectory-replay",
                "optimization_protocol": "top2_adaptive_epoch@1",
                "optimization_schedule": schedule.to_dict(),
            },
        )
        self.run_id = "run:trajectory-replay"
        self.director.start_evolution(task, run_id=self.run_id)
        self.candidates = tuple(
            self.director.propose_and_spawn(self.run_id) for _ in range(4)
        )
        self.revisions = {}
        for revision_index, candidate in enumerate(self.candidates):
            revision = CandidateRevision(
                revision_id=f"revision:{revision_index}:0",
                run_id=self.run_id,
                generation=0,
                candidate_id=candidate.candidate_id,
                genome={"parameters": {"slot": revision_index}},
                genome_digest=_sha(f"genome:{revision_index}"),
                behavior_digest=_sha(f"behavior:{revision_index}"),
                mutation_digest=_sha(f"mutation:{revision_index}"),
                status=RevisionStatus.ACTIVE,
            )
            self.director.create_candidate_revision(self.run_id, revision)
            self.revisions[candidate.candidate_id] = revision
        dataset = SimpleNamespace(
            dataset_id="generated-toy-series@1",
            episode_id="episode:trajectory-replay",
            timestamps=tuple(range(3200)),
            partitions={"model_selection": IndexRange(0, 3200)},
        )
        adaptation = plan_run_adaptation_cohort(
            dataset, schedule=self.schedule, seed=7
        )
        self.generation_cohorts = plan_generation_selection_cohorts(
            dataset,
            schedule=self.schedule,
            generation=0,
            adaptation=adaptation,
            seed=7,
        )
        self.director.freeze_run_adaptation_cohort(self.run_id, adaptation)
        self.director.freeze_generation_selection_cohorts(
            self.run_id, self.generation_cohorts
        )

    def tearDown(self) -> None:
        self.ledger.close()

    def test_adaptive_screening_requires_initial_revision_r0(self) -> None:
        run_id = "run:trajectory-missing-r0"
        task = TaskManifest(
            task_id="trajectory-missing-r0",
            objective="reject screening before R0",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={
                "max_generations": 1,
                "candidates_per_generation": 4,
                "max_candidates": 4,
            },
            seed=11,
            metadata={
                "optimization_protocol": "top2_adaptive_epoch@1",
                "optimization_schedule": self.schedule.to_dict(),
            },
        )
        self.director.start_evolution(task, run_id=run_id)
        candidate = self.director.propose_and_spawn(run_id)

        with self.assertRaisesRegex(ValueError, "initial revision R0"):
            self.director.record_candidate_screening(
                run_id,
                candidate_id=candidate.candidate_id,
                generation=0,
                score=0.5,
                passed=True,
                constraint_violations=0,
                origin_count=64,
                prediction_cell_count=64,
                cohort_digest=_sha("missing-r0-screening"),
            )

    def _freeze_top2(self) -> tuple:
        cohort = self.generation_cohorts.screening.cohort_digest
        for candidate in self.candidates:
            self.director.record_candidate_screening(
                self.run_id,
                candidate_id=candidate.candidate_id,
                generation=0,
                score=1.0 - candidate.slot_index * 0.1,
                passed=True,
                constraint_violations=0,
                origin_count=64,
                prediction_cell_count=64,
                cohort_digest=cohort,
            )
        state = self.director.state(self.run_id)
        records = [event.payload for event in state.candidate_screening_events]
        finalists = self.candidates[:2]
        self.director.freeze_formal_selection_cohort(
            self.run_id,
            generation=0,
            selected_candidate_ids=[item.candidate_id for item in finalists],
            screening_digest=screening_cohort_digest(records),
        )
        return finalists

    def _complete_lane(self, candidate) -> None:
        revision = self.revisions[candidate.candidate_id]
        self.director.start_formal_trajectory(
            self.run_id,
            candidate.candidate_id,
            revision.revision_id,
            self.schedule.batch_count,
        )
        for batch_index in range(self.schedule.batch_count):
            adaptation = self.director.state(self.run_id).run_adaptation_cohort
            assert adaptation is not None
            cohort = adaptation.batches[batch_index].cohort.cohort_digest
            batch = self.director.start_formal_batch(
                self.run_id,
                candidate.candidate_id,
                revision.revision_id,
                batch_index,
            )
            scope = EvaluationScope(
                run_id=self.run_id,
                generation=0,
                candidate_id=candidate.candidate_id,
                candidate_revision_id=revision.revision_id,
                phase=EvaluationPhase.FORMAL_BATCH,
                cohort_digest=cohort,
                origin_count=10,
                batch_index=batch_index,
            )
            self.director.record_formal_batch_evaluation(
                self.run_id,
                BatchEvaluation(
                    evaluation_id=f"evaluation:{candidate.slot_index}:{batch_index}",
                    scope=scope,
                    score=0.5 + batch_index / 100,
                    passed=True,
                    metrics={"rmse": 1.0},
                    evaluator_digest=_sha("evaluator"),
                ),
            )
            proposal_id = f"local:{candidate.slot_index}:{batch_index}"
            self.director.record_local_edit_proposal(
                self.run_id,
                {
                    "proposal_id": proposal_id,
                    "candidate_id": candidate.candidate_id,
                    "batch_index": batch_index,
                    "evidence_scope_digest": scope.scope_key,
                    "decision": "keep",
                    "operations": [],
                },
            )
            self.director.decide_local_edit(
                self.run_id,
                {
                    "proposal_id": proposal_id,
                    "candidate_id": candidate.candidate_id,
                    "batch_index": batch_index,
                    "outcome": "kept",
                    "active_revision_id": revision.revision_id,
                },
            )
            self.director.advance_trajectory_revision(
                self.run_id,
                candidate.candidate_id,
                batch_index,
                revision.revision_id,
                RevisionAdvanceReason.KEPT,
            )
            self.assertEqual(batch.batch_index, batch_index)
        self.director.complete_formal_trajectory(
            self.run_id,
            candidate.candidate_id,
            revision.revision_id,
        )

    def test_complete_two_lane_replay_selects_effective_revision(self) -> None:
        finalists = self._freeze_top2()
        for candidate in finalists:
            self._complete_lane(candidate)

        incumbent = self.candidates[2]
        arm_bindings = {
            HoldoutArm.FINALIST_1.value: {
                "candidate_id": finalists[0].candidate_id,
                "candidate_revision_id": self.revisions[
                    finalists[0].candidate_id
                ].revision_id,
            },
            HoldoutArm.FINALIST_2.value: {
                "candidate_id": finalists[1].candidate_id,
                "candidate_revision_id": self.revisions[
                    finalists[1].candidate_id
                ].revision_id,
            },
            HoldoutArm.INCUMBENT.value: {
                "candidate_id": incumbent.candidate_id,
                "candidate_revision_id": self.revisions[
                    incumbent.candidate_id
                ].revision_id,
            },
        }
        holdout = self.director.freeze_generation_holdout(
            self.run_id,
            0,
            _sha("holdout"),
            arm_bindings,
        )
        evaluations = []
        for index, arm in enumerate(HoldoutArm):
            binding = arm_bindings[arm.value]
            evaluation = HoldoutEvaluation(
                evaluation_id=f"holdout-evaluation:{arm.value}",
                scope=EvaluationScope(
                    run_id=self.run_id,
                    generation=0,
                    candidate_id=binding["candidate_id"],
                    candidate_revision_id=binding["candidate_revision_id"],
                    phase=EvaluationPhase.HOLDOUT,
                    cohort_digest=holdout.cohort_digest,
                    origin_count=169,
                    holdout_arm=arm,
                ),
                score=0.9 - index * 0.1,
                passed=True,
                metrics={"rmse": 1.0 + index},
                evaluator_digest=_sha("evaluator"),
            )
            self.director.record_holdout_evaluation(self.run_id, evaluation)
            evaluations.append(evaluation)
        comparison = GenerationComparison(
            comparison_id="comparison:0",
            run_id=self.run_id,
            generation=0,
            cohort_digest=holdout.cohort_digest,
            holdout_evaluations=tuple(evaluations),
            selected_candidate_id=finalists[0].candidate_id,
            selected_revision_id=self.revisions[finalists[0].candidate_id].revision_id,
            gate_results={"host_gates_passed": True},
        )
        self.director.record_generation_comparison(self.run_id, comparison)
        self.director.select_generation_champion(
            self.run_id,
            0,
            comparison.selected_revision_id,
            comparison.comparison_digest,
        )

        state = self.director.replay(self.run_id)
        self.assertEqual(
            state.trajectory_for(finalists[0].candidate_id).final_revision_id,
            self.revisions[finalists[0].candidate_id].revision_id,
        )
        self.assertEqual(
            state.formal_batch_for(finalists[0].candidate_id, 0).revision_id,
            self.revisions[finalists[0].candidate_id].revision_id,
        )
        self.assertEqual(
            state.comparison_for(0).selected_revision_id,
            comparison.selected_revision_id,
        )
        self.assertEqual(
            state.effective_revision_for(0), comparison.selected_revision_id
        )

    def test_trajectory_cannot_start_before_top2_freeze(self) -> None:
        candidate = self.candidates[0]
        with self.assertRaisesRegex(ValueError, "Top 2"):
            self.director.start_formal_trajectory(
                self.run_id,
                candidate.candidate_id,
                self.revisions[candidate.candidate_id].revision_id,
                self.schedule.batch_count,
            )

    def test_batch_index_cannot_skip_and_activation_is_exactly_once(self) -> None:
        candidate = self._freeze_top2()[0]
        revision = self.revisions[candidate.candidate_id]
        self.director.start_formal_trajectory(
            self.run_id,
            candidate.candidate_id,
            revision.revision_id,
            self.schedule.batch_count,
        )
        with self.assertRaisesRegex(ValueError, "next batch index"):
            self.director.start_formal_batch(
                self.run_id,
                candidate.candidate_id,
                revision.revision_id,
                1,
            )


if __name__ == "__main__":
    unittest.main()
