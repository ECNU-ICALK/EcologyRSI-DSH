from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from threading import Barrier, Event as ThreadEvent, Lock
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ecologyrsi_dsh import EventLedger, EvolutionDirector, FakeDSHAdapter, TaskManifest
from ecologyrsi_dsh.api import formal_trajectory, generation_execution
from ecologyrsi_dsh.core.models import CandidateRole, digest
from ecologyrsi_dsh.core.screening import (
    screening_cohort_digest,
    screening_record_digest,
)
from ecologyrsi_dsh.core.state import (
    project_run_state,
    uses_global_incumbent_protocol,
    uses_positive_delta_search_protocol,
)
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
    TrajectoryStatus,
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

    def test_runtime_v2_and_v3_freeze_distinct_search_policies(self) -> None:
        base = self.director.state(self.run_id).task_manifest.to_dict()
        metadata = {
            **dict(base["metadata"]),
            "execution_protocol": "dsh_native_plugin_evolution@1",
            "host_runtime_build": {
                "package_version": "0.3.55",
                "evolution_runtime_schema": "ecologyrsi-dsh.evolution-runtime/2",
                "generation_comparison_schema": (
                    "ecologyrsi-dsh.generation-comparison/1"
                ),
                "projection_schema": "ecologyrsi-dsh.execution-projection/2",
            },
        }
        runtime_v2 = TaskManifest.from_dict({**base, "metadata": metadata})
        self.assertTrue(uses_global_incumbent_protocol(runtime_v2))
        self.assertFalse(uses_positive_delta_search_protocol(runtime_v2))

        metadata["host_runtime_build"] = {
            **metadata["host_runtime_build"],
            "evolution_runtime_schema": "ecologyrsi-dsh.evolution-runtime/3",
        }
        runtime_v3 = TaskManifest.from_dict({**base, "metadata": metadata})
        self.assertTrue(uses_global_incumbent_protocol(runtime_v3))
        self.assertTrue(uses_positive_delta_search_protocol(runtime_v3))

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
        with self.assertRaisesRegex(ValueError, "comparison digest"):
            self.director.select_generation_champion(
                self.run_id,
                0,
                comparison.selected_revision_id,
                _sha("different-comparison"),
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

    def test_closeout_resumes_after_three_durable_holdout_arms_without_rerun(
        self,
    ) -> None:
        """A crash after holdout must only append the missing decision events."""

        class _InjectedCrash(RuntimeError):
            pass

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
        evaluations = []
        for arm in HoldoutArm:
            binding = bindings[arm.value]
            evaluation = HoldoutEvaluation(
                evaluation_id=f"holdout-evaluation:{arm.value}",
                scope=EvaluationScope(
                    run_id=self.run_id,
                    generation=0,
                    candidate_id=binding["candidate_id"],
                    candidate_revision_id=binding["candidate_revision_id"],
                    phase=EvaluationPhase.HOLDOUT,
                    cohort_digest=holdout.cohort_digest,
                    origin_count=holdout.origin_count,
                    holdout_arm=arm,
                ),
                score=0.4,
                passed=False,
                metrics={"constraint_violations": 0},
                evaluator_digest=_sha("closeout-evaluator"),
            )
            self.director.record_holdout_evaluation(self.run_id, evaluation)
            evaluations.append(evaluation)
        comparison = GenerationComparison(
            comparison_id="comparison:closeout-resume",
            run_id=self.run_id,
            generation=0,
            cohort_digest=holdout.cohort_digest,
            holdout_evaluations=tuple(evaluations),
            selected_candidate_id=incumbent.candidate_id,
            selected_revision_id=self.revisions[incumbent.candidate_id].revision_id,
            gate_results={"host_gates_passed": False},
        )
        endpoint = SimpleNamespace(
            server=SimpleNamespace(director=self.director, ledger=self.ledger)
        )
        real_mutation = generation_execution._director_mutation

        def commit_selection_then_crash(endpoint, method_name, *args, **kwargs):
            result = real_mutation(endpoint, method_name, *args, **kwargs)
            if method_name == "select_generation_champion":
                raise _InjectedCrash("after durable champion selection")
            return result

        with (
            patch.object(
                generation_execution,
                "_execute_adaptive_holdout_arm",
                side_effect=lambda *_args: self.director.state(
                    self.run_id
                ).holdout_evaluation_for(0, _args[3]),
            ) as execute_holdout,
            patch.object(
                generation_execution,
                "build_generation_comparison",
                return_value=comparison,
            ),
            patch.object(
                generation_execution,
                "_director_mutation",
                side_effect=commit_selection_then_crash,
            ),
        ):
            with self.assertRaisesRegex(
                _InjectedCrash,
                "after durable champion selection",
            ):
                generation_execution._finalize_adaptive_generation(
                    endpoint,
                    self.run_id,
                    SimpleNamespace(generation=0),
                )

        self.assertEqual(execute_holdout.call_count, 3)
        first_state = self.director.replay(self.run_id)
        self.assertEqual(len(first_state.holdout_evaluations), 3)
        self.assertEqual(
            sum(
                event.kind == "GenerationComparisonRecorded"
                for event in first_state.events
            ),
            1,
        )
        self.assertEqual(
            sum(
                event.kind == "GenerationChampionSelected"
                for event in first_state.events
            ),
            1,
        )

        self.director = EvolutionDirector(
            self.ledger,
            FakeDSHAdapter(max_proposals=20),
        )
        endpoint.server.director = self.director
        with (
            patch.object(
                generation_execution,
                "_execute_adaptive_holdout_arm",
                side_effect=lambda *_args: self.director.state(
                    self.run_id
                ).holdout_evaluation_for(0, _args[3]),
            ) as retry_holdout,
            patch.object(
                generation_execution,
                "_build_adaptive_analysis",
                side_effect=_InjectedCrash("stop after closeout"),
            ),
        ):
            with self.assertRaisesRegex(_InjectedCrash, "stop after closeout"):
                generation_execution._finalize_adaptive_generation(
                    endpoint,
                    self.run_id,
                    SimpleNamespace(generation=0),
                )

        replayed = self.director.replay(self.run_id)
        self.assertEqual(retry_holdout.call_count, 3)
        self.assertEqual(len(replayed.holdout_evaluations), 3)
        self.assertEqual(
            sum(
                event.kind == "GenerationComparisonRecorded"
                for event in replayed.events
            ),
            1,
        )
        self.assertEqual(
            sum(
                event.kind == "GenerationChampionSelected"
                for event in replayed.events
            ),
            1,
        )

    def test_holdout_finalists_must_bind_completed_trajectory_revisions(self) -> None:
        finalists = self._freeze_top2()
        for candidate in finalists:
            self._complete_lane(candidate)

        finalist = finalists[0]
        final_revision = self.revisions[finalist.candidate_id]
        orphan = CandidateRevision(
            revision_id=f"revision:{finalist.candidate_id}:orphan",
            run_id=self.run_id,
            generation=0,
            candidate_id=finalist.candidate_id,
            parent_revision_id=final_revision.revision_id,
            source_batch_index=0,
            genome={"parameters": {"slot": 404}},
            genome_digest=_sha("orphan-finalist-genome"),
            behavior_digest=_sha("orphan-finalist-behavior"),
            mutation_digest=_sha("orphan-finalist-mutation"),
            status=RevisionStatus.ACTIVE,
        )
        self.director.create_candidate_revision(self.run_id, orphan)
        incumbent = self.candidates[2]
        valid_bindings = {
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
        forged_bindings = {
            arm: dict(binding) for arm, binding in valid_bindings.items()
        }
        forged_bindings[HoldoutArm.FINALIST_1.value][
            "candidate_revision_id"
        ] = orphan.revision_id
        before = self.ledger.count(self.run_id)

        with self.assertRaisesRegex(ValueError, "trajectory final revision"):
            self.director.freeze_generation_holdout(
                self.run_id,
                0,
                _sha("holdout-final-binding"),
                forged_bindings,
            )
        self.assertEqual(self.ledger.count(self.run_id), before)

        holdout = self.director.freeze_generation_holdout(
            self.run_id,
            0,
            _sha("holdout-final-binding"),
            valid_bindings,
        )
        forged_holdout = {
            **holdout.to_dict(),
            "arm_bindings": forged_bindings,
        }
        forged_events = tuple(
            replace(event, payload={"holdout": forged_holdout})
            if event.kind == "GenerationHoldoutFrozen"
            else event
            for event in self.ledger.events(self.run_id)
        )
        with self.assertRaisesRegex(ValueError, "trajectory final revision"):
            project_run_state(forged_events)

    def test_runtime_v2_generation_zero_holdout_requires_materialized_seed_control(
        self,
    ) -> None:
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
        state = self.director.state(self.run_id)
        task = TaskManifest.from_dict(
            {
                **state.task_manifest.to_dict(),
                "metadata": {
                    **dict(state.task_manifest.metadata),
                    "execution_protocol": "dsh_native_plugin_evolution@1",
                    "host_runtime_build": {
                        "package_version": "0.3.55",
                        "evolution_runtime_schema": (
                            "ecologyrsi-dsh.evolution-runtime/2"
                        ),
                        "generation_comparison_schema": (
                            "ecologyrsi-dsh.generation-comparison/1"
                        ),
                        "projection_schema": (
                            "ecologyrsi-dsh.execution-projection/2"
                        ),
                    },
                },
            }
        )
        v2_state = Mock(wraps=state)
        v2_state.task_manifest = task
        v2_state.formal_trajectories = state.formal_trajectories
        v2_state.candidates = state.candidates
        v2_state.candidate_revisions = state.candidate_revisions
        v2_state.materialized_seed_genome_canonical_json = None
        v2_state.generation_holdout_for = state.generation_holdout_for
        v2_state.formal_selection_for = state.formal_selection_for
        v2_state.revision = state.revision

        with patch.object(self.director, "state", return_value=v2_state):
            with self.assertRaisesRegex(ValueError, "seed incumbent control"):
                self.director.freeze_generation_holdout(
                    self.run_id,
                    0,
                    _sha("runtime-v2-seed-binding"),
                    bindings,
                )

        legacy_holdout = self.director.freeze_generation_holdout(
            self.run_id,
            0,
            _sha("runtime-v2-seed-binding"),
            bindings,
        )
        self.assertEqual(legacy_holdout.generation, 0)

    def test_runtime_v2_later_holdout_requires_prior_effective_champion(self) -> None:
        state = self.director.state(self.run_id)
        task = TaskManifest.from_dict(
            {
                **state.task_manifest.to_dict(),
                "metadata": {
                    **dict(state.task_manifest.metadata),
                    "execution_protocol": "dsh_native_plugin_evolution@1",
                    "host_runtime_build": {
                        "package_version": "0.3.55",
                        "evolution_runtime_schema": (
                            "ecologyrsi-dsh.evolution-runtime/2"
                        ),
                        "generation_comparison_schema": (
                            "ecologyrsi-dsh.generation-comparison/1"
                        ),
                        "projection_schema": (
                            "ecologyrsi-dsh.execution-projection/2"
                        ),
                    },
                },
            }
        )
        finalists = tuple(
            SimpleNamespace(
                candidate_id=f"candidate:g1:{index}",
                generation=1,
                role=CandidateRole.SEARCH,
            )
            for index in range(2)
        )
        finalist_revisions = tuple(
            CandidateRevision(
                revision_id=f"revision:g1:{index}",
                run_id=self.run_id,
                generation=1,
                candidate_id=candidate.candidate_id,
                genome={"slot": index},
                genome_digest=_sha(f"g1-genome:{index}"),
                behavior_digest=_sha(f"g1-behavior:{index}"),
                mutation_digest=_sha(f"g1-mutation:{index}"),
                status=RevisionStatus.FINAL,
            )
            for index, candidate in enumerate(finalists)
        )
        trajectories = tuple(
            SimpleNamespace(
                generation=1,
                candidate_id=candidate.candidate_id,
                status=TrajectoryStatus.COMPLETED,
                final_revision_id=revision.revision_id,
            )
            for candidate, revision in zip(finalists, finalist_revisions)
        )
        prior_revision = self.revisions[self.candidates[0].candidate_id]
        wrong_incumbent = self.candidates[2]
        wrong_revision = self.revisions[wrong_incumbent.candidate_id]
        all_candidates = (*state.candidates, *finalists)
        all_revisions = (*state.candidate_revisions, *finalist_revisions)
        revision_by_id = {item.revision_id: item for item in all_revisions}
        v2_state = SimpleNamespace(
            task_manifest=task,
            generation_holdout_for=lambda _generation: None,
            formal_trajectories=trajectories,
            formal_selection_for=lambda _generation: SimpleNamespace(
                payload={
                    "selected_candidate_ids": [
                        item.candidate_id for item in finalists
                    ]
                }
            ),
            revision=lambda revision_id: revision_by_id[revision_id],
            candidates=all_candidates,
            candidate_revisions=all_revisions,
            materialized_seed_genome_canonical_json=None,
            effective_revision_for=lambda generation: (
                prior_revision.revision_id if generation == 0 else None
            ),
            events=(
                SimpleNamespace(
                    kind="GenerationChampionSelected",
                    payload={"generation": 0},
                ),
            ),
        )
        bindings = {
            HoldoutArm.FINALIST_1.value: {
                "candidate_id": finalists[0].candidate_id,
                "candidate_revision_id": finalist_revisions[0].revision_id,
            },
            HoldoutArm.FINALIST_2.value: {
                "candidate_id": finalists[1].candidate_id,
                "candidate_revision_id": finalist_revisions[1].revision_id,
            },
            HoldoutArm.INCUMBENT.value: {
                "candidate_id": wrong_incumbent.candidate_id,
                "candidate_revision_id": wrong_revision.revision_id,
            },
        }

        with patch.object(self.director, "state", return_value=v2_state):
            with self.assertRaisesRegex(ValueError, "prior effective champion"):
                self.director.freeze_generation_holdout(
                    self.run_id,
                    1,
                    _sha("generation-one-holdout"),
                    bindings,
                )

    def test_exploration_comparison_must_durably_retain_incumbent(self) -> None:
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
            _sha("exploration-comparison"),
            bindings,
        )
        evaluations = []
        for arm in HoldoutArm:
            binding = bindings[arm.value]
            evaluation = HoldoutEvaluation(
                evaluation_id=f"evaluation:exploration:{arm.value}",
                scope=EvaluationScope(
                    run_id=self.run_id,
                    generation=0,
                    candidate_id=binding["candidate_id"],
                    candidate_revision_id=binding["candidate_revision_id"],
                    phase=EvaluationPhase.HOLDOUT,
                    cohort_digest=holdout.cohort_digest,
                    origin_count=holdout.origin_count,
                    holdout_arm=arm,
                ),
                score=0.7 if arm is HoldoutArm.FINALIST_1 else 0.5,
                passed=True,
                metrics={"constraint_violations": 0},
                evaluator_digest=_sha("exploration-evaluator"),
            )
            self.director.record_holdout_evaluation(self.run_id, evaluation)
            evaluations.append(evaluation)
        invalid = GenerationComparison(
            comparison_id="comparison:forged-exploration",
            run_id=self.run_id,
            generation=0,
            cohort_digest=holdout.cohort_digest,
            holdout_evaluations=tuple(evaluations),
            selected_candidate_id=finalists[0].candidate_id,
            selected_revision_id=self.revisions[
                finalists[0].candidate_id
            ].revision_id,
            gate_results={
                "selected_arm": HoldoutArm.FINALIST_1.value,
                "challenger_promotion_allowed": True,
            },
        )
        changed_evaluations = list(evaluations)
        changed_evaluations[0] = HoldoutEvaluation.from_dict(
            {
                **changed_evaluations[0].to_dict(),
                "score": changed_evaluations[0].score + 0.01,
            }
        )
        changed_score = GenerationComparison(
            comparison_id="comparison:changed-score",
            run_id=self.run_id,
            generation=0,
            cohort_digest=holdout.cohort_digest,
            holdout_evaluations=tuple(changed_evaluations),
            selected_candidate_id=incumbent.candidate_id,
            selected_revision_id=self.revisions[incumbent.candidate_id].revision_id,
            gate_results={"selected_arm": HoldoutArm.INCUMBENT.value},
        )
        with self.assertRaisesRegex(ValueError, "durable holdout evidence"):
            self.director.record_generation_comparison(
                self.run_id,
                changed_score,
            )

        other_run_evaluations = tuple(
            HoldoutEvaluation.from_dict(
                {
                    **evaluation.to_dict(),
                    "scope": {
                        **evaluation.scope.to_dict(),
                        "run_id": "run:other",
                    },
                }
            )
            for evaluation in evaluations
        )
        cross_run = GenerationComparison(
            comparison_id="comparison:cross-run",
            run_id="run:other",
            generation=0,
            cohort_digest=holdout.cohort_digest,
            holdout_evaluations=other_run_evaluations,
            selected_candidate_id=other_run_evaluations[-1].scope.candidate_id,
            selected_revision_id=(
                other_run_evaluations[-1].scope.candidate_revision_id
            ),
            gate_results={"selected_arm": HoldoutArm.INCUMBENT.value},
        )
        with self.assertRaisesRegex(ValueError, "another run"):
            self.director.record_generation_comparison(self.run_id, cross_run)

        last = self.ledger.events(self.run_id)[-1]
        forged_changed_event = replace(
            last,
            seq=last.seq + 1,
            event_id="forged:changed-score-comparison",
            kind="GenerationComparisonRecorded",
            payload={"comparison": changed_score.to_dict()},
        )
        with self.assertRaisesRegex(ValueError, "durable holdout evidence"):
            project_run_state((*self.ledger.events(self.run_id), forged_changed_event))

        state = self.director.state(self.run_id)
        v2_task = TaskManifest.from_dict(
            {
                **state.task_manifest.to_dict(),
                "metadata": {
                    **dict(state.task_manifest.metadata),
                    "execution_protocol": "dsh_native_plugin_evolution@1",
                    "host_runtime_build": {
                        "package_version": "0.3.55",
                        "evolution_runtime_schema": (
                            "ecologyrsi-dsh.evolution-runtime/2"
                        ),
                        "generation_comparison_schema": (
                            "ecologyrsi-dsh.generation-comparison/1"
                        ),
                        "projection_schema": (
                            "ecologyrsi-dsh.execution-projection/2"
                        ),
                    },
                },
            }
        )
        canonical_state = Mock(wraps=state)
        canonical_state.task_manifest = v2_task
        canonical_state.comparison_for = state.comparison_for
        canonical_state.generation_holdout_for = state.generation_holdout_for
        canonical_state.holdout_evaluation_for = state.holdout_evaluation_for
        canonical_state.formal_selection_for = state.formal_selection_for
        with patch.object(self.director, "state", return_value=canonical_state):
            with self.assertRaisesRegex(ValueError, "deterministic Host comparison"):
                self.director.record_generation_comparison(self.run_id, invalid)

        exploration_state = Mock(wraps=state)
        exploration_state.formal_selection_for = lambda _generation: SimpleNamespace(
            payload={
                "schema_version": "ecologyrsi-dsh.formal-selection-cohort/3",
                "screening_pass_count": 0,
                "exploration_only": True,
                "consecutive_exploration_generations": 1,
            }
        )

        with patch.object(self.director, "state", return_value=exploration_state):
            with self.assertRaisesRegex(ValueError, "exploration comparison"):
                self.director.record_generation_comparison(self.run_id, invalid)

        self.ledger.append(
            self.run_id,
            "GenerationComparisonRecorded",
            {"comparison": invalid.to_dict()},
            event_id=f"{self.run_id}:generation:0:comparison",
        )
        events = list(self.ledger.events(self.run_id))
        screening_payloads = []
        for event in events:
            if event.kind == "CandidateScreeningRecorded":
                screening_payload = {**event.payload, "passed": False}
                if "record_digest" in screening_payload:
                    screening_payload["record_digest"] = screening_record_digest(
                        screening_payload
                    )
                screening_payloads.append(screening_payload)
        forged_screening_digest = screening_cohort_digest(screening_payloads)
        screening_by_candidate = {
            str(payload["candidate_id"]): payload
            for payload in screening_payloads
        }
        forged_events = tuple(
            replace(
                event,
                payload=screening_by_candidate[str(event.payload["candidate_id"])],
            )
            if event.kind == "CandidateScreeningRecorded"
            else replace(
                event,
                payload={
                    "schema_version": "ecologyrsi-dsh.formal-selection-cohort/3",
                    "generation": 0,
                    "selected_candidate_ids": list(
                        event.payload["selected_candidate_ids"]
                    ),
                    "screening_digest": forged_screening_digest,
                    "screening_pass_count": 0,
                    "exploration_only": True,
                    "consecutive_exploration_generations": 1,
                },
            )
            if event.kind == "FormalSelectionCohortFrozen"
            else event
            for event in events
        )
        with self.assertRaisesRegex(ValueError, "exploration comparison"):
            project_run_state(forged_events)

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

    def _paired_comparison_case(
        self,
        run_id: str,
        *,
        reject_initial_challenger: bool = False,
        stop_after_proposal: bool = False,
    ) -> SimpleNamespace:
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
        warmup_metrics = _paired_metrics(score=0.2, batch_index=0)
        paired_evaluator_digest = _sha("paired-evaluator")
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
            metrics=warmup_metrics,
            evaluator_digest=paired_evaluator_digest,
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
            comparison_contract_digest=_paired_contract_digest(
                warmup_metrics,
                paired_evaluator_digest,
            ),
            safety_gate_passed=True,
            cell_regression_gate_passed=True,
            minimum_score_delta=0.005,
            decision=FormalBatchComparisonDecision.INITIAL_CHAMPION,
            champion_after_revision_id=initial.revision_id,
            reason="initial_champion",
        )
        self.director.record_formal_batch_comparison(run_id, initial_comparison)

        proposal_id = f"local:{candidate.candidate_id}:0"
        proposal_payload = {
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
        }
        self.director.record_local_edit_proposal(run_id, proposal_payload)
        if stop_after_proposal:
            return SimpleNamespace(
                run_id=run_id,
                candidate=candidate,
                initial=initial,
                warmup=warmup,
                proposal_payload=proposal_payload,
            )
        if reject_initial_challenger:
            decision_payload = {
                "proposal_id": proposal_id,
                "candidate_id": candidate.candidate_id,
                "batch_index": 0,
                "outcome": "rejected",
                "active_revision_id": initial.revision_id,
            }
            self.director.decide_local_edit(run_id, decision_payload)
            self.director.advance_trajectory_revision(
                run_id,
                candidate.candidate_id,
                0,
                initial.revision_id,
                RevisionAdvanceReason.LOCAL_EDIT_REJECTED,
            )
            return SimpleNamespace(
                run_id=run_id,
                candidate=candidate,
                initial=initial,
                warmup=warmup,
                proposal_payload=proposal_payload,
                decision_payload=decision_payload,
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
        decision_payload = {
            "proposal_id": proposal_id,
            "candidate_id": candidate.candidate_id,
            "batch_index": 0,
            "outcome": "applied",
            "active_revision_id": challenger.revision_id,
        }
        self.director.decide_local_edit(run_id, decision_payload)
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
        champion_metrics = _paired_metrics(score=0.4, batch_index=1)
        challenger_metrics = _paired_metrics(score=0.3, batch_index=1)
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
            metrics=champion_metrics,
            evaluator_digest=paired_evaluator_digest,
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
            metrics=challenger_metrics,
            evaluator_digest=paired_evaluator_digest,
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
            comparison_contract_digest=_paired_contract_digest(
                champion_metrics,
                paired_evaluator_digest,
            ),
            safety_gate_passed=True,
            cell_regression_gate_passed=False,
            minimum_score_delta=0.005,
            decision=FormalBatchComparisonDecision.CHAMPION_RETAINED,
            champion_after_revision_id=initial.revision_id,
            reason="challenger_cell_regression",
        )
        return SimpleNamespace(
            run_id=run_id,
            candidate=candidate,
            initial=initial,
            challenger=challenger,
            champion_evaluation=champion_evaluation,
            challenger_evaluation=challenger_evaluation,
            comparison=comparison,
            proposal_payload=proposal_payload,
            decision_payload=decision_payload,
        )

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
        return CandidateRevision(
            revision_id=revision_id,
            run_id=case.run_id,
            generation=0,
            candidate_id=case.candidate.candidate_id,
            parent_revision_id=parent_revision_id,
            source_batch_index=source_batch_index,
            genome={"parameters": {"slot": 999}},
            genome_digest=_sha(f"{revision_id}:genome"),
            behavior_digest=_sha(f"{revision_id}:behavior"),
            mutation_digest=_sha(f"{revision_id}:mutation"),
            status=RevisionStatus.ACTIVE,
        )

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
                        payload={**event.payload, "operations": operations},
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
            "proposal_id": "local:paired:final",
            "candidate_id": proposal_case.candidate.candidate_id,
            "batch_index": 1,
            "evidence_scope_digest": (
                proposal_case.challenger_evaluation.scope.scope_key
            ),
            "decision": "keep",
            "operations": [],
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
                "proposal_id": "local:paired:final:raw",
                "candidate_id": proposal_case.candidate.candidate_id,
                "batch_index": 1,
                "evidence_scope_digest": (
                    proposal_case.challenger_evaluation.scope.scope_key
                ),
                "decision": "keep",
                "operations": [],
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
