from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ecologyrsi_dsh import EventLedger, EvolutionDirector, FakeDSHAdapter, TaskManifest
from ecologyrsi_dsh.api import formal_trajectory, generation_execution
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.screening import screening_cohort_digest
from ecologyrsi_dsh.core.trajectory import (
    BatchEvaluation,
    CandidateRevision,
    EvaluationPhase,
    EvaluationScope,
    FormalBatchArm,
    FormalBatchComparison,
    FormalBatchComparisonDecision,
    GenerationComparison,
    HoldoutArm,
    HoldoutEvaluation,
    LocalEditOutcome,
    RevisionAdvanceReason,
    RevisionStatus,
)
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.evaluators.epoch_cohorts import (
    plan_generation_selection_cohorts,
    plan_run_adaptation_cohort,
)
from ecologyrsi_dsh.evolution.local_edits import LocalEditProposal, LocalEditResult
from ecologyrsi_dsh.evolution.schedule import OptimizationSchedule


def _sha(label: str) -> str:
    return digest({"label": label})


class TrajectoryEventReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = EventLedger()
        self.director = EvolutionDirector(
            self.ledger, FakeDSHAdapter(max_proposals=20)
        )
        schedule_value = OptimizationSchedule.default().to_dict()
        schedule_value.update(
            schema_version="ecologyrsi-dsh.top2-adaptive-epoch-schedule/1",
            local_evaluation_mode="prequential",
            formal_origin_count_per_finalist=100,
            local_batch_origin_count=10,
        )
        schedule = OptimizationSchedule.from_dict(schedule_value)
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
                "prediction_cells_per_origin": 9,
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

    def test_candidate_revision_retry_ignores_only_created_at(self) -> None:
        candidate = self.candidates[0]
        parent = self.revisions[candidate.candidate_id]
        first = CandidateRevision(
            revision_id=f"revision:{candidate.candidate_id}:batch:1",
            run_id=self.run_id,
            generation=0,
            candidate_id=candidate.candidate_id,
            parent_revision_id=parent.revision_id,
            source_batch_index=0,
            genome={"parameters": {"slot": 100}},
            genome_digest=_sha("child-genome"),
            behavior_digest=_sha("child-behavior"),
            mutation_digest=_sha("child-mutation"),
            status=RevisionStatus.ACTIVE,
            created_at="2026-08-28T00:00:00+00:00",
        )
        persisted = self.director.create_candidate_revision(self.run_id, first)
        replay = CandidateRevision.from_dict(
            {
                **first.to_dict(),
                "created_at": "2026-08-28T00:01:00+00:00",
            }
        )

        recovered = self.director.create_candidate_revision(self.run_id, replay)

        self.assertEqual(recovered, persisted)
        self.assertEqual(recovered.created_at, first.created_at)
        self.assertEqual(
            sum(
                event.kind == "CandidateRevisionCreated"
                and event.payload["revision"]["revision_id"] == first.revision_id
                for event in self.director.state(self.run_id).events
            ),
            1,
        )

    def test_local_edit_recovery_reuses_persisted_child_without_reauthoring(
        self,
    ) -> None:
        """Resume the exact crash window between child creation and edit decision."""

        candidate = self._freeze_top2()[0]
        parent = self.revisions[candidate.candidate_id]
        self.director.start_formal_trajectory(
            self.run_id,
            candidate.candidate_id,
            parent.revision_id,
            self.schedule.batch_count,
        )
        batch = self.director.start_formal_batch(
            self.run_id,
            candidate.candidate_id,
            parent.revision_id,
            0,
        )
        scope = EvaluationScope(
            run_id=self.run_id,
            generation=0,
            candidate_id=candidate.candidate_id,
            candidate_revision_id=parent.revision_id,
            phase=EvaluationPhase.FORMAL_BATCH,
            cohort_digest=batch.cohort_digest,
            origin_count=batch.origin_count,
            batch_index=0,
        )
        self.director.record_formal_batch_evaluation(
            self.run_id,
            BatchEvaluation(
                evaluation_id=f"evaluation:{candidate.candidate_id}:0",
                scope=scope,
                score=0.5,
                passed=True,
                metrics={
                    "constraint_violations": 0,
                    "sample_execution": {"coverage_pass": True},
                },
                evaluator_digest=_sha("local-edit-recovery-evaluator"),
            ),
        )
        proposal = LocalEditProposal(
            decision="mutate",
            operations=(
                {
                    "op": "set_bounded_parameter",
                    "name": "ridge_alpha",
                    "value": 0.5,
                },
            ),
            evidence_refs=("batch:score",),
            expected_effect_cells=("air_temperature@1h",),
            risk_cells=(),
        )
        self.director.record_local_edit_proposal(
            self.run_id,
            {
                "proposal_id": f"local-edit:{candidate.candidate_id}:0",
                "candidate_id": candidate.candidate_id,
                "batch_index": 0,
                "evidence_scope_digest": scope.scope_key,
                "proposal": proposal.to_dict(),
            },
        )
        child_payload = {"parameters": {"slot": 100}}
        child_genome_digest = _sha("recovered-child-genome")
        child_behavior_digest = _sha("recovered-child-behavior")
        child_mutation_digest = _sha("recovered-child-mutation")
        child_revision_id = f"revision:{candidate.candidate_id}:batch:1"
        persisted_child = CandidateRevision(
            revision_id=child_revision_id,
            run_id=self.run_id,
            generation=0,
            candidate_id=candidate.candidate_id,
            parent_revision_id=parent.revision_id,
            source_batch_index=0,
            genome=child_payload,
            genome_digest=child_genome_digest,
            behavior_digest=child_behavior_digest,
            mutation_digest=child_mutation_digest,
            status=RevisionStatus.ACTIVE,
            created_at="2026-08-28T00:00:00+00:00",
        )
        self.director.create_candidate_revision(self.run_id, persisted_child)
        child = SimpleNamespace(
            to_dict=lambda: child_payload,
            genome_digest=child_genome_digest,
            behavior_digest=child_behavior_digest,
            lineage={"mutation_digest": child_mutation_digest},
        )
        context = SimpleNamespace(
            candidate_revision_id=parent.revision_id,
            evidence_scope_digest=scope.scope_key,
        )
        endpoint = SimpleNamespace(server=SimpleNamespace(director=self.director))

        with (
            patch.object(
                formal_trajectory,
                "_local_edit_context",
                return_value=context,
            ),
            patch.object(formal_trajectory, "_local_edit_proposal") as authored,
            patch.object(
                formal_trajectory.EcologyEvolutionPluginGenome,
                "from_dict",
                return_value=object(),
            ),
            patch.object(
                formal_trajectory,
                "apply_or_reject_local_edit_bundle",
                return_value=LocalEditResult(
                    LocalEditOutcome.APPLIED,
                    tuple(proposal.operations),
                    child,
                    digest(proposal.to_dict()),
                ),
            ),
        ):
            progressed = formal_trajectory.execute_next_local_edit(
                endpoint,
                self.run_id,
                candidate.candidate_id,
            )

        self.assertTrue(progressed)
        authored.assert_not_called()
        state = self.director.state(self.run_id)
        outcome = next(
            item
            for item in state.local_edit_outcomes
            if item["candidate_id"] == candidate.candidate_id
            and item["batch_index"] == 0
        )
        self.assertEqual(outcome["outcome"], LocalEditOutcome.APPLIED.value)
        self.assertEqual(outcome["active_revision_id"], child_revision_id)
        activation = state.revision_activation_for(candidate.candidate_id, 0)
        self.assertIsNotNone(activation)
        self.assertEqual(activation.to_revision_id, child_revision_id)
        self.assertEqual(
            sum(
                event.kind == "CandidateRevisionCreated"
                and event.payload["revision"]["revision_id"] == child_revision_id
                for event in state.events
            ),
            1,
        )
        self.assertEqual(
            sum(
                event.kind == "LocalEditDecided"
                and event.payload["candidate_id"] == candidate.candidate_id
                and event.payload["batch_index"] == 0
                for event in state.events
            ),
            1,
        )
        self.assertEqual(
            sum(
                event.kind == "TrajectoryRevisionAdvanced"
                and event.payload["activation"]["candidate_id"]
                == candidate.candidate_id
                and event.payload["activation"]["batch_index"] == 0
                for event in state.events
            ),
            1,
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

    def test_holdout_arm_started_is_idempotent_across_real_state_replay(self) -> None:
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
            _sha("idempotent-holdout"),
            arm_bindings,
        )
        arm = HoldoutArm.FINALIST_1
        endpoint = SimpleNamespace(
            server=SimpleNamespace(
                director=self.director,
                ledger=self.ledger,
                evaluators=Mock(),
            )
        )
        endpoint.server.evaluators.evaluate_scientific.side_effect = RuntimeError(
            "stop after durable start"
        )
        cohort = SimpleNamespace(
            cohort_digest=holdout.cohort_digest,
            origin_count=holdout.origin_count,
        )
        started_event_id = (
            f"{self.run_id}:generation:0:holdout:{arm.value}:started"
        )

        with patch.object(
            generation_execution,
            "_holdout_replay_inputs",
            return_value=(finalists[0], object(), object()),
        ):
            for _ in range(2):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "stop after durable start",
                ):
                    generation_execution._execute_adaptive_holdout_arm(
                        endpoint,
                        self.run_id,
                        0,
                        arm,
                        arm_bindings[arm.value],
                        cohort,
                    )
                replayed = self.director.replay(self.run_id)
                starts = [
                    event
                    for event in replayed.events
                    if event.event_id == started_event_id
                ]
                self.assertEqual(len(starts), 1)
                self.assertEqual(starts[0].kind, "HoldoutArmStarted")
                self.assertEqual(
                    starts[0].payload["cohort_digest"],
                    holdout.cohort_digest,
                )

        self.assertEqual(
            endpoint.server.evaluators.evaluate_scientific.call_count,
            2,
        )

    def test_generation_holdout_retry_preserves_first_created_at(self) -> None:
        finalists = self._freeze_top2()
        for candidate in finalists:
            self._complete_lane(candidate)
        incumbent = self.candidates[2]
        bindings = {
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
        first_time = datetime(2026, 8, 28, 0, 0, tzinfo=timezone.utc)
        retry_time = datetime(2026, 8, 28, 0, 1, tzinfo=timezone.utc)
        with patch("ecologyrsi_dsh.core.models.datetime") as clock:
            clock.now.return_value = first_time
            first = self.director.freeze_generation_holdout(
                self.run_id,
                0,
                self.generation_cohorts.holdout.cohort_digest,
                bindings,
            )
            clock.now.return_value = retry_time
            recovered = self.director.freeze_generation_holdout(
                self.run_id,
                0,
                self.generation_cohorts.holdout.cohort_digest,
                bindings,
            )

        self.assertEqual(recovered, first)
        self.assertEqual(recovered.created_at, first_time.isoformat(timespec="milliseconds"))
        self.assertEqual(
            sum(
                event.kind == "GenerationHoldoutFrozen"
                for event in self.director.state(self.run_id).events
            ),
            1,
        )

    def test_screened_out_incumbent_holdout_opens_scoped_checkpoint(self) -> None:
        finalists = self._freeze_top2()
        for candidate in finalists:
            self._complete_lane(candidate)

        incumbent = self.candidates[2]
        bindings = {
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
            self.generation_cohorts.holdout.cohort_digest,
            bindings,
        )
        binding = bindings[HoldoutArm.INCUMBENT.value]
        scope = EvaluationScope(
            run_id=self.run_id,
            generation=0,
            candidate_id=incumbent.candidate_id,
            candidate_revision_id=binding["candidate_revision_id"],
            phase=EvaluationPhase.HOLDOUT,
            cohort_digest=holdout.cohort_digest,
            origin_count=holdout.origin_count,
            holdout_arm=HoldoutArm.INCUMBENT,
        )
        checkpoint = {
            "schema_version": "ecologyrsi-dsh.sample-checkpoint/2",
            "candidate_revision_id": scope.candidate_revision_id,
            "evaluation_phase": scope.phase.value,
            "formal_batch_index": None,
            "holdout_arm": HoldoutArm.INCUMBENT.value,
            "cohort_digest": scope.cohort_digest,
            "execution_scope_digest": scope.scope_key,
            "sample_cohort_digest": _sha("incumbent-holdout-samples"),
            "execution_context_digest": _sha("incumbent-holdout-context"),
            "sample_count": scope.origin_count * 9,
        }

        prepared = self.director.prepare_evaluation_sample_checkpoint(
            self.run_id,
            generation=0,
            proposal_id=incumbent.proposal_id,
            candidate_id=incumbent.candidate_id,
            checkpoint=checkpoint,
            scope=scope,
        )

        self.assertFalse(prepared["resumed"])
        self.assertEqual(len(prepared["rows"]), 0)

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

    def test_paired_evaluations_and_comparison_replay_by_explicit_arm(self) -> None:
        run_id = "run:paired-replay"
        schedule = OptimizationSchedule.from_dict(
            {
                **OptimizationSchedule.default().to_dict(),
                "formal_origin_count_per_finalist": 20,
                "local_batch_origin_count": 10,
            }
        )
        task = TaskManifest(
            task_id="paired-replay",
            objective="exercise paired comparison replay",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={
                "max_generations": 1,
                "candidates_per_generation": 4,
                "max_candidates": 4,
            },
            seed=17,
            metadata={
                "episode_id": "episode:paired-replay",
                "optimization_protocol": "top2_adaptive_epoch@1",
                "optimization_schedule": schedule.to_dict(),
                "prediction_cells_per_origin": 9,
            },
        )
        self.director.start_evolution(task, run_id=run_id)
        candidates = tuple(
            self.director.propose_and_spawn(run_id) for _ in range(4)
        )
        revisions = []
        for index, candidate in enumerate(candidates):
            revision = CandidateRevision(
                revision_id=f"revision:paired:{index}:0",
                run_id=run_id,
                generation=0,
                candidate_id=candidate.candidate_id,
                genome={"parameters": {"slot": index}},
                genome_digest=_sha(f"paired-genome:{index}"),
                behavior_digest=_sha(f"paired-behavior:{index}"),
                mutation_digest=_sha(f"paired-mutation:{index}"),
                status=RevisionStatus.ACTIVE,
            )
            self.director.create_candidate_revision(run_id, revision)
            revisions.append(revision)
        dataset = SimpleNamespace(
            dataset_id="generated-toy-series@1",
            episode_id="episode:paired-replay",
            timestamps=tuple(range(1200)),
            partitions={"model_selection": IndexRange(0, 1200)},
        )
        adaptation = plan_run_adaptation_cohort(
            dataset,
            schedule=schedule,
            seed=17,
        )
        cohorts = plan_generation_selection_cohorts(
            dataset,
            schedule=schedule,
            generation=0,
            adaptation=adaptation,
            seed=17,
        )
        self.director.freeze_run_adaptation_cohort(run_id, adaptation)
        self.director.freeze_generation_selection_cohorts(run_id, cohorts)
        for candidate in candidates:
            self.director.record_candidate_screening(
                run_id,
                candidate_id=candidate.candidate_id,
                generation=0,
                score=1.0 - candidate.slot_index * 0.1,
                passed=True,
                constraint_violations=0,
                origin_count=64,
                prediction_cell_count=64,
                cohort_digest=cohorts.screening.cohort_digest,
            )
        screening_records = [
            event.payload
            for event in self.director.state(run_id).candidate_screening_events
        ]
        finalists = candidates[:2]
        self.director.freeze_formal_selection_cohort(
            run_id,
            generation=0,
            selected_candidate_ids=[item.candidate_id for item in finalists],
            screening_digest=screening_cohort_digest(screening_records),
        )

        candidate = finalists[0]
        initial = revisions[0]
        self.director.start_formal_trajectory(
            run_id,
            candidate.candidate_id,
            initial.revision_id,
            schedule.batch_count,
        )
        batch0 = self.director.start_formal_batch(
            run_id,
            candidate.candidate_id,
            initial.revision_id,
            0,
        )
        warmup = BatchEvaluation(
            evaluation_id="evaluation:paired:warmup",
            scope=EvaluationScope(
                run_id=run_id,
                generation=0,
                candidate_id=candidate.candidate_id,
                candidate_revision_id=initial.revision_id,
                phase=EvaluationPhase.FORMAL_BATCH,
                cohort_digest=batch0.cohort_digest,
                origin_count=batch0.origin_count,
                batch_index=0,
                formal_batch_arm=FormalBatchArm.CHAMPION,
            ),
            score=0.2,
            passed=True,
            metrics={"targets": []},
            evaluator_digest=_sha("paired-evaluator"),
        )
        self.director.record_formal_batch_evaluation(run_id, warmup)
        initial_comparison = FormalBatchComparison(
            comparison_id=f"comparison:{candidate.candidate_id}:0",
            run_id=run_id,
            generation=0,
            candidate_id=candidate.candidate_id,
            batch_index=0,
            cohort_digest=batch0.cohort_digest,
            champion_before_revision_id=initial.revision_id,
            challenger_revision_id=initial.revision_id,
            champion_evaluation_id=warmup.evaluation_id,
            challenger_evaluation_id=warmup.evaluation_id,
            champion_evaluation_digest=warmup.evaluation_digest,
            challenger_evaluation_digest=warmup.evaluation_digest,
            champion_score=warmup.score,
            challenger_score=warmup.score,
            score_delta=0.0,
            comparison_contract_digest=_sha("paired-contract"),
            safety_gate_passed=True,
            cell_regression_gate_passed=True,
            minimum_score_delta=0.005,
            decision=FormalBatchComparisonDecision.INITIAL_CHAMPION,
            champion_after_revision_id=initial.revision_id,
            reason="initial_champion",
        )
        self.director.record_formal_batch_comparison(run_id, initial_comparison)

        proposal_id = f"local:{candidate.candidate_id}:0"
        self.director.record_local_edit_proposal(
            run_id,
            {
                "proposal_id": proposal_id,
                "candidate_id": candidate.candidate_id,
                "batch_index": 0,
                "evidence_scope_digest": warmup.scope.scope_key,
                "decision": "mutate",
                "operations": [
                    {
                        "op": "set_bounded_parameter",
                        "name": "ridge_alpha",
                        "value": 0.5,
                    }
                ],
            },
        )
        challenger = CandidateRevision(
            revision_id=f"revision:{candidate.candidate_id}:challenger:1",
            run_id=run_id,
            generation=0,
            candidate_id=candidate.candidate_id,
            parent_revision_id=initial.revision_id,
            source_batch_index=0,
            genome={"parameters": {"slot": 100}},
            genome_digest=_sha("paired-child-genome"),
            behavior_digest=_sha("paired-child-behavior"),
            mutation_digest=_sha("paired-child-mutation"),
            status=RevisionStatus.ACTIVE,
        )
        self.director.create_candidate_revision(run_id, challenger)
        self.director.decide_local_edit(
            run_id,
            {
                "proposal_id": proposal_id,
                "candidate_id": candidate.candidate_id,
                "batch_index": 0,
                "outcome": "applied",
                "active_revision_id": challenger.revision_id,
            },
        )
        self.director.advance_trajectory_revision(
            run_id,
            candidate.candidate_id,
            0,
            challenger.revision_id,
            RevisionAdvanceReason.LOCAL_EDIT_APPLIED,
        )

        batch1 = self.director.start_formal_batch(
            run_id,
            candidate.candidate_id,
            challenger.revision_id,
            1,
        )
        champion_evaluation = BatchEvaluation(
            evaluation_id="evaluation:paired:champion:1",
            scope=EvaluationScope(
                run_id=run_id,
                generation=0,
                candidate_id=candidate.candidate_id,
                candidate_revision_id=initial.revision_id,
                phase=EvaluationPhase.FORMAL_BATCH,
                cohort_digest=batch1.cohort_digest,
                origin_count=batch1.origin_count,
                batch_index=1,
                formal_batch_arm=FormalBatchArm.CHAMPION,
            ),
            score=0.4,
            passed=True,
            metrics={"targets": []},
            evaluator_digest=_sha("paired-evaluator"),
        )
        challenger_evaluation = BatchEvaluation(
            evaluation_id="evaluation:paired:challenger:1",
            scope=EvaluationScope(
                run_id=run_id,
                generation=0,
                candidate_id=candidate.candidate_id,
                candidate_revision_id=challenger.revision_id,
                phase=EvaluationPhase.FORMAL_BATCH,
                cohort_digest=batch1.cohort_digest,
                origin_count=batch1.origin_count,
                batch_index=1,
                formal_batch_arm=FormalBatchArm.CHALLENGER,
            ),
            score=0.3,
            passed=True,
            metrics={"targets": []},
            evaluator_digest=_sha("paired-evaluator"),
        )
        self.director.record_formal_batch_evaluation(
            run_id,
            champion_evaluation,
        )
        self.director.record_formal_batch_evaluation(
            run_id,
            challenger_evaluation,
        )
        comparison = FormalBatchComparison(
            comparison_id=f"comparison:{candidate.candidate_id}:1",
            run_id=run_id,
            generation=0,
            candidate_id=candidate.candidate_id,
            batch_index=1,
            cohort_digest=batch1.cohort_digest,
            champion_before_revision_id=initial.revision_id,
            challenger_revision_id=challenger.revision_id,
            champion_evaluation_id=champion_evaluation.evaluation_id,
            challenger_evaluation_id=challenger_evaluation.evaluation_id,
            champion_evaluation_digest=champion_evaluation.evaluation_digest,
            challenger_evaluation_digest=challenger_evaluation.evaluation_digest,
            champion_score=champion_evaluation.score,
            challenger_score=challenger_evaluation.score,
            score_delta=-0.1,
            comparison_contract_digest=_sha("paired-contract"),
            safety_gate_passed=True,
            cell_regression_gate_passed=False,
            minimum_score_delta=0.005,
            decision=FormalBatchComparisonDecision.CHAMPION_RETAINED,
            champion_after_revision_id=initial.revision_id,
            reason="challenger_cell_regression",
        )
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
