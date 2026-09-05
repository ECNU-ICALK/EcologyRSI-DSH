from __future__ import annotations

import unittest
from types import SimpleNamespace

from ecologyrsi_dsh.api.generation_execution import (
    _build_adaptive_analysis,
    _local_edit_trajectory_evidence,
)
from ecologyrsi_dsh.core.models import (
    CandidateRole,
    CandidateStatus,
    canonical_json,
    digest,
)
from ecologyrsi_dsh.core.trajectory import (
    CandidateRevision,
    EvaluationPhase,
    EvaluationScope,
    FormalBatchComparisonDecision,
    GenerationComparison,
    HoldoutArm,
    HoldoutEvaluation,
    RevisionAdvanceReason,
    RevisionStatus,
)
from ecologyrsi_dsh.evolution.batches import (
    _adaptive_reflection_analysis,
    _canonical_candidate_outcomes,
)
from ecologyrsi_dsh.evolution.context import safe_aggregate_feedback
from ecologyrsi_dsh.evolution.schedule import (
    LEGACY_SCHEDULE_SCHEMA_VERSION,
    OPTIMIZATION_PROTOCOL,
    PREQUENTIAL_LOCAL_EVALUATION_MODE,
    OptimizationSchedule,
)


class AdaptiveReflectionEvidenceTests(unittest.TestCase):
    maxDiff = None

    def test_legacy_prequential_reflection_shape_is_unchanged(self) -> None:
        candidate_id = "candidate:legacy"
        final_revision_id = "revision:legacy:r1"
        schedule = OptimizationSchedule.default().to_dict()
        schedule.update(
            schema_version=LEGACY_SCHEDULE_SCHEMA_VERSION,
            local_evaluation_mode=PREQUENTIAL_LOCAL_EVALUATION_MODE,
        )
        state = SimpleNamespace(
            task_manifest=SimpleNamespace(
                metadata={"optimization_schedule": schedule}
            ),
            formal_batches=(
                SimpleNamespace(
                    candidate_id=candidate_id,
                    batch_index=0,
                    revision_id="revision:legacy:r0",
                ),
            ),
            local_edit_proposals=(
                {
                    "candidate_id": candidate_id,
                    "batch_index": 0,
                    "proposal": {
                        "decision": "mutate",
                        "operations": [
                            {
                                "op": "set_bounded_parameter",
                                "name": "ridge_alpha",
                                "value": 0.2,
                            }
                        ],
                    },
                },
            ),
            local_edit_outcomes=(
                {
                    "candidate_id": candidate_id,
                    "batch_index": 0,
                    "outcome": "applied",
                    "active_revision_id": final_revision_id,
                },
            ),
            trajectory_revision_activations=(
                SimpleNamespace(
                    candidate_id=candidate_id,
                    batch_index=0,
                    from_revision_id="revision:legacy:r0",
                    to_revision_id=final_revision_id,
                    reason=RevisionAdvanceReason.LOCAL_EDIT_APPLIED,
                ),
            ),
            trajectory_for=lambda requested_id: (
                SimpleNamespace(
                    initial_revision_id="revision:legacy:r0",
                    final_revision_id=final_revision_id,
                    batch_count=1,
                )
                if requested_id == candidate_id
                else None
            ),
        )

        evidence = _local_edit_trajectory_evidence(
            state,
            candidate_id,
            final_revision_id,
        )

        self.assertNotIn("local_evaluation_mode", evidence)
        self.assertNotIn("recent_comparisons", evidence)
        self.assertEqual(
            set(evidence["batches"][0]),
            {
                "batch_index",
                "batch_number",
                "evaluated_revision_id",
                "decision",
                "operation_category",
                "operation_count",
                "outcome",
                "outcome_reason",
                "from_revision_id",
                "active_revision_id",
                "advance_reason",
            },
        )

    def _revision(self, candidate_id: str, revision_id: str, marker: str) -> CandidateRevision:
        genome = {"schema_version": "test@1", "marker": marker}
        return CandidateRevision(
            revision_id=revision_id,
            run_id="run:adaptive-reflection",
            generation=0,
            candidate_id=candidate_id,
            genome=genome,
            genome_digest=digest(genome),
            behavior_digest=digest({"behavior": marker}),
            mutation_digest=digest({"mutation": marker}),
            status=RevisionStatus.FINAL,
        )

    def _holdout(
        self,
        arm: HoldoutArm,
        candidate_id: str,
        revision_id: str,
        score: float,
    ) -> HoldoutEvaluation:
        return HoldoutEvaluation(
            evaluation_id=f"holdout:{arm.value}",
            scope=EvaluationScope(
                run_id="run:adaptive-reflection",
                generation=0,
                candidate_id=candidate_id,
                candidate_revision_id=revision_id,
                phase=EvaluationPhase.HOLDOUT,
                cohort_digest="c" * 64,
                origin_count=169,
                holdout_arm=arm,
            ),
            score=score,
            passed=True,
            metrics={"constraint_violations": 0},
            evaluator_digest="e" * 64,
        )

    def test_analysis_and_reflection_preserve_gate_and_revision_evidence(self) -> None:
        candidates = tuple(
            SimpleNamespace(
                candidate_id=f"candidate:{index}",
                proposal_id=f"proposal:{index}",
                generation=0,
                slot_index=index,
                status=(
                    CandidateStatus.EVALUATED
                    if index in {0, 3}
                    else CandidateStatus.SCREENED_OUT
                ),
            )
            for index in range(4)
        )
        finalist_ids = (candidates[0].candidate_id, candidates[3].candidate_id)
        finalist_revisions = {
            finalist_ids[0]: self._revision(
                finalist_ids[0], "revision:failed:final", "failed-final"
            ),
            finalist_ids[1]: self._revision(
                finalist_ids[1], "revision:winner:final", "winner-final"
            ),
        }
        incumbent_revision = self._revision(
            "candidate:incumbent",
            "revision:incumbent",
            "incumbent",
        )
        revisions = {
            item.revision_id: item
            for item in (*finalist_revisions.values(), incumbent_revision)
        }
        proposals = {
            candidate.proposal_id: SimpleNamespace(
                metadata={
                    "candidate_direction_id": f"direction-{candidate.slot_index}",
                    "candidate_direction_digest": digest(
                        {"direction": candidate.slot_index}
                    ),
                    "mutation_operations": [
                        {
                            "op": "set_bounded_parameter",
                            "name": "ridge_alpha",
                            "value": 0.1 + candidate.slot_index,
                        }
                    ],
                    "behavior_digest": digest(
                        {"outer_behavior": candidate.slot_index}
                    ),
                }
            )
            for candidate in candidates
        }
        control = SimpleNamespace(
            candidate_id="candidate:seed-control",
            proposal_id="proposal:seed-control",
            generation=0,
            slot_index=0,
            status=CandidateStatus.SPAWNED,
            role=CandidateRole.INCUMBENT_CONTROL,
        )
        proposals[control.proposal_id] = SimpleNamespace(metadata={})
        formal_batches = []
        local_edit_proposals = []
        local_edit_outcomes = []
        activations = []
        formal_batch_comparisons = []
        trajectories = {}
        for candidate_id in finalist_ids:
            final_revision_id = finalist_revisions[candidate_id].revision_id
            initial_revision_id = f"revision:{candidate_id}:r0"
            champion_revision_id = initial_revision_id
            scheduled_revision_id = initial_revision_id
            trajectories[candidate_id] = SimpleNamespace(
                initial_revision_id=initial_revision_id,
                final_revision_id=final_revision_id,
                batch_count=10,
            )
            for batch_index in range(10):
                formal_batches.append(
                    SimpleNamespace(
                        candidate_id=candidate_id,
                        batch_index=batch_index,
                        revision_id=scheduled_revision_id,
                    )
                )
                champion_before_revision_id = champion_revision_id
                if batch_index == 0:
                    decision = FormalBatchComparisonDecision.INITIAL_CHAMPION
                    reason = "initial_champion"
                    score_delta = 0.0
                elif batch_index == 2:
                    decision = FormalBatchComparisonDecision.CHAMPION_RETAINED
                    reason = "below_practical_delta"
                    score_delta = -0.01
                else:
                    decision = FormalBatchComparisonDecision.CHALLENGER_PROMOTED
                    reason = "challenger_improved"
                    score_delta = 0.02
                    champion_revision_id = scheduled_revision_id
                formal_batch_comparisons.append(
                    SimpleNamespace(
                        candidate_id=candidate_id,
                        batch_index=batch_index,
                        champion_before_revision_id=champion_before_revision_id,
                        challenger_revision_id=scheduled_revision_id,
                        champion_after_revision_id=champion_revision_id,
                        decision=decision,
                        reason=reason,
                        score_delta=score_delta,
                        minimum_score_delta=0.005,
                        safety_gate_passed=True,
                        cell_regression_gate_passed=True,
                        # These fields emulate data that must never enter the
                        # bounded generation-reflection envelope.
                        raw_predictions=[0.1, 0.2],
                        labels=[0.0, 1.0],
                        created_at="2026-08-30T00:00:00Z",
                        rationale="unbounded model-authored prose",
                    )
                )
                if batch_index == 9:
                    continue
                next_revision_id = (
                    final_revision_id
                    if batch_index == 8
                    else f"revision:{candidate_id}:challenger:{batch_index + 1}"
                )
                operation_name = "history_steps" if batch_index == 1 else "ridge_alpha"
                local_edit_proposals.append(
                    {
                        "candidate_id": candidate_id,
                        "batch_index": batch_index,
                        "proposal": {
                            "decision": "mutate",
                            "operations": [
                                {
                                    "op": "set_bounded_parameter",
                                    "name": operation_name,
                                    "value": (
                                        "secret-rejected-operation-value"
                                        if batch_index == 1
                                        else 0.2 + batch_index / 100
                                    ),
                                }
                            ],
                        },
                    }
                )
                local_edit_outcomes.append(
                    {
                        "candidate_id": candidate_id,
                        "batch_index": batch_index,
                        "outcome": "applied",
                        "active_revision_id": next_revision_id,
                    }
                )
                activations.append(
                    SimpleNamespace(
                        candidate_id=candidate_id,
                        batch_index=batch_index,
                        from_revision_id=champion_revision_id,
                        to_revision_id=next_revision_id,
                        reason=RevisionAdvanceReason.LOCAL_EDIT_APPLIED,
                    )
                )
                scheduled_revision_id = next_revision_id

        failed_holdout = self._holdout(
            HoldoutArm.FINALIST_1,
            finalist_ids[0],
            finalist_revisions[finalist_ids[0]].revision_id,
            0.54,
        )
        winner_holdout = self._holdout(
            HoldoutArm.FINALIST_2,
            finalist_ids[1],
            finalist_revisions[finalist_ids[1]].revision_id,
            0.62,
        )
        incumbent_holdout = self._holdout(
            HoldoutArm.INCUMBENT,
            incumbent_revision.candidate_id,
            incumbent_revision.revision_id,
            0.50,
        )
        cell_deltas = {
            f"{target}@{horizon}h": 0.02
            for target in (
                "air_temperature",
                "relative_humidity",
                "co2_concentration",
            )
            for horizon in (1, 6, 24)
        }

        def finalist_gate(*, eligible: bool, no_regression: bool) -> dict:
            return {
                "passed": True,
                "constraint_violations": 0,
                "overall_coverage": 1.0,
                "coverage_pass": True,
                "eligible": eligible,
                "complete_objective_grid": True,
                "no_cell_regression": no_regression,
                "worst_cell_delta": 0.02 if no_regression else -0.08,
                "cell_deltas": cell_deltas,
                "promotion_assessment": {
                    "evidence_class": "selection_eligible",
                    "status": "selection_only",
                    "paired_block_count": 169,
                    "valid_three_day_start_count": 167,
                    "paired_block_ids_digest": "b" * 64,
                    "primary_delta": 0.12,
                    "selection_stability_floor": 0.04,
                    "primary_selection_gate": True,
                },
                "stability_lower_bound": 0.04,
                "failures": [] if no_regression else ["cell_regression"],
            }

        gate_results = {
            "schema_version": "ecologyrsi-dsh.generation-comparison/1",
            "arms": {
                HoldoutArm.FINALIST_1.value: finalist_gate(
                    eligible=False, no_regression=False
                ),
                HoldoutArm.FINALIST_2.value: finalist_gate(
                    eligible=True, no_regression=True
                ),
                HoldoutArm.INCUMBENT.value: {
                    "passed": True,
                    "constraint_violations": 0,
                    "overall_coverage": 1.0,
                    "coverage_pass": True,
                    "eligible": True,
                    "score": incumbent_holdout.score,
                },
            },
            "eligible_finalist_count": 1,
            "selected_arm": HoldoutArm.FINALIST_2.value,
            "selected_score": winner_holdout.score,
            "incumbent_score": incumbent_holdout.score,
            "delta_to_incumbent": winner_holdout.score - incumbent_holdout.score,
            "incumbent_gate_pass": True,
        }
        comparison = GenerationComparison(
            comparison_id="comparison:adaptive-reflection",
            run_id="run:adaptive-reflection",
            generation=0,
            cohort_digest="c" * 64,
            holdout_evaluations=(
                failed_holdout,
                winner_holdout,
                incumbent_holdout,
            ),
            selected_candidate_id=winner_holdout.scope.candidate_id,
            selected_revision_id=winner_holdout.scope.candidate_revision_id,
            gate_results=gate_results,
        )
        screenings = {
            candidates[1].candidate_id: {
                "score": 0.41,
                "passed": True,
                "constraint_violations": 0,
            },
            candidates[2].candidate_id: {
                "score": 0.39,
                "passed": False,
                "constraint_violations": 0,
            },
        }
        state = SimpleNamespace(
            run=SimpleNamespace(run_id="run:adaptive-reflection"),
            task_manifest=SimpleNamespace(
                metadata={
                    "optimization_protocol": OPTIMIZATION_PROTOCOL,
                    "optimization_schedule": OptimizationSchedule.default().to_dict(),
                }
            ),
            candidates=(*candidates, control),
            formal_batches=tuple(formal_batches),
            formal_batch_comparisons=tuple(formal_batch_comparisons),
            local_edit_proposals=tuple(local_edit_proposals),
            local_edit_outcomes=tuple(local_edit_outcomes),
            trajectory_revision_activations=tuple(activations),
            proposal=lambda proposal_id: proposals[proposal_id],
            revision=lambda revision_id: revisions[revision_id],
            trajectory_for=lambda candidate_id: trajectories.get(candidate_id),
            comparison_for=lambda generation: comparison if generation == 0 else None,
            screening_for=lambda generation, candidate_id: (
                SimpleNamespace(payload=screenings[candidate_id])
                if generation == 0 and candidate_id in screenings
                else None
            ),
        )

        analysis = _build_adaptive_analysis(
            state,
            0,
            comparison,
            (candidates[0], candidates[3]),
            incumbent_revision.candidate_id,
        )

        self.assertEqual(analysis.selected_candidate_id, candidates[3].candidate_id)
        self.assertEqual(analysis.eligible_count, 1)
        self.assertEqual(len(analysis.ranking), 4)
        winner = analysis.ranking[0]
        failed = next(
            row
            for row in analysis.ranking
            if row["candidate_id"] == candidates[0].candidate_id
        )
        screened = [
            row
            for row in analysis.ranking
            if row["selection_status"] == "screened_out"
        ]
        self.assertEqual(winner["rank"], 1)
        self.assertEqual(winner["selection_reason"], "generation_holdout_winner")
        self.assertEqual(winner["final_revision_id"], "revision:winner:final")
        self.assertEqual(
            winner["final_revision_digest"],
            finalist_revisions[finalist_ids[1]].revision_digest,
        )
        self.assertEqual(len(winner["comparison_gate"]["cell_deltas"]), 9)
        self.assertEqual(winner["outer_mutation_evidence"]["kind"], "outer_generation_mutation")
        self.assertEqual(winner["local_edit_evidence"]["kind"], "batch_local_edits")
        self.assertEqual(winner["local_edit_evidence"]["included_batch_count"], 9)
        self.assertEqual(len(winner["local_edit_evidence"]["batches"]), 9)
        self.assertEqual(winner["local_edit_evidence"]["comparison_count"], 10)
        self.assertEqual(len(winner["local_edit_evidence"]["revision_chain"]), 11)
        comparison_history = winner["local_edit_evidence"]["recent_comparisons"]
        self.assertEqual(len(comparison_history), 10)
        accepted = next(item for item in comparison_history if item["batch_index"] == 1)
        rejected = next(item for item in comparison_history if item["batch_index"] == 2)
        self.assertEqual(accepted["decision"], "challenger_promoted")
        self.assertEqual(accepted["reason"], "challenger_improved")
        self.assertEqual(rejected["decision"], "champion_retained")
        self.assertEqual(rejected["reason"], "below_practical_delta")
        self.assertEqual(rejected["score_delta"], -0.01)
        self.assertIn("history_steps", rejected["rejected_operation_targets"])
        self.assertEqual(
            rejected["champion_before_revision_id"],
            rejected["champion_after_revision_id"],
        )
        self.assertNotEqual(
            rejected["challenger_revision_id"],
            rejected["champion_after_revision_id"],
        )
        self.assertEqual(
            rejected["next_mutation_parent_revision_id"],
            rejected["champion_after_revision_id"],
        )
        self.assertEqual(failed["rank"], None)
        self.assertEqual(failed["score"], 0.54)
        self.assertFalse(failed["eligible"])
        self.assertIn("cell_regression", failed["failure_reasons"])
        self.assertIn("cell_regression", failed["selection_reason"])
        self.assertEqual([row["score"] for row in screened], [0.41, 0.39])
        self.assertTrue(all(row["rank"] is None for row in screened))
        self.assertTrue(screened[0]["scientific_pass"])
        self.assertEqual(screened[0]["classification"], "screened_out_lower_rank")
        self.assertFalse(screened[1]["scientific_pass"])
        self.assertEqual(screened[1]["classification"], "screening_gate_failed")

        outcomes = _canonical_candidate_outcomes(
            state,
            generation=0,
            analysis=analysis,
        )
        self.assertEqual(len(outcomes), 4)
        self.assertEqual(outcomes[0]["candidate_id"], candidates[3].candidate_id)
        self.assertEqual(outcomes[1]["score"], 0.54)
        self.assertIsNone(outcomes[1]["rank"])
        self.assertEqual([item["score"] for item in outcomes[2:]], [0.41, 0.39])

        reflection = _adaptive_reflection_analysis(state, analysis)
        replayed = _adaptive_reflection_analysis(state, analysis)
        self.assertEqual(reflection, replayed)
        evidence = reflection["adaptive_epoch_evidence"]
        self.assertEqual(evidence["selected"]["revision_id"], "revision:winner:final")
        self.assertEqual(evidence["incumbent"]["revision_id"], "revision:incumbent")
        self.assertEqual(len(evidence["candidate_results"]), 4)
        reflected_lane = next(
            item
            for item in evidence["trajectory_comparisons"]
            if item["candidate_id"] == candidates[3].candidate_id
        )
        reflected_rejection = next(
            item
            for item in reflected_lane["recent_comparisons"]
            if item["batch_index"] == 2
        )
        self.assertEqual(reflected_rejection["reason"], "below_practical_delta")
        self.assertEqual(reflected_rejection["score_delta"], -0.01)
        self.assertIn(
            "history_steps",
            reflected_rejection["rejected_operation_targets"],
        )
        serialized_reflection = canonical_json(reflection)
        for forbidden_field in (
            "raw_predictions",
            "labels",
            "created_at",
            "timestamps",
            "rationale",
        ):
            self.assertNotIn(f'"{forbidden_field}"', serialized_reflection)
        self.assertNotIn("secret-rejected-operation-value", serialized_reflection)
        self.assertNotIn("unbounded model-authored prose", serialized_reflection)
        self.assertIsNotNone(
            safe_aggregate_feedback(
                reflection,
                name="adaptive reflection test evidence",
            )
        )
        legacy_schedule = OptimizationSchedule.default().to_dict()
        legacy_schedule.update(
            schema_version=LEGACY_SCHEDULE_SCHEMA_VERSION,
            local_evaluation_mode=PREQUENTIAL_LOCAL_EVALUATION_MODE,
        )
        legacy_reflection_state = SimpleNamespace(
            **{
                **vars(state),
                "task_manifest": SimpleNamespace(
                    metadata={
                        "optimization_protocol": OPTIMIZATION_PROTOCOL,
                        "optimization_schedule": legacy_schedule,
                    }
                ),
            }
        )
        legacy_reflection = _adaptive_reflection_analysis(
            legacy_reflection_state,
            analysis,
        )
        self.assertIn("created_at", legacy_reflection)
        self.assertNotIn(
            "trajectory_comparisons",
            legacy_reflection["adaptive_epoch_evidence"],
        )

        search_only_gate = {
            **gate_results["arms"][HoldoutArm.FINALIST_2.value],
            "eligible": True,
            "search_eligible": True,
            "certification_eligible": False,
            "search_failures": [],
            "certification_failures": ["scientific_gate_failed"],
        }
        search_only_comparison = GenerationComparison(
            comparison_id="comparison:adaptive-reflection:search-only",
            run_id="run:adaptive-reflection",
            generation=0,
            cohort_digest="c" * 64,
            holdout_evaluations=(
                failed_holdout,
                winner_holdout,
                incumbent_holdout,
            ),
            selected_candidate_id=winner_holdout.scope.candidate_id,
            selected_revision_id=winner_holdout.scope.candidate_revision_id,
            gate_results={
                **gate_results,
                "selection_policy": "positive_delta_search@1",
                "certification_selected_arm": None,
                "selected_search_certification_status": "search_only",
                "arms": {
                    **gate_results["arms"],
                    HoldoutArm.FINALIST_2.value: search_only_gate,
                },
            },
        )
        search_only_state = SimpleNamespace(
            **{
                **vars(state),
                "comparison_for": lambda generation: (
                    search_only_comparison if generation == 0 else None
                ),
            }
        )
        search_only = _build_adaptive_analysis(
            search_only_state,
            0,
            search_only_comparison,
            (candidates[0], candidates[3]),
            incumbent_revision.candidate_id,
        )
        self.assertEqual(search_only.outcome, "search_version_advanced")
        self.assertEqual(
            search_only.selected_candidate_id,
            winner_holdout.scope.candidate_id,
        )
        self.assertIsNone(search_only.champion_candidate_id)
        self.assertEqual(
            search_only.incumbent_after_candidate_id,
            incumbent_revision.candidate_id,
        )
        self.assertEqual(
            search_only.ranking[0]["selection_reason"],
            "next_round_search_version",
        )

        incumbent_gate_results = {
            **gate_results,
            "arms": {
                **gate_results["arms"],
                HoldoutArm.FINALIST_2.value: finalist_gate(
                    eligible=False,
                    no_regression=False,
                ),
            },
            "eligible_finalist_count": 0,
            "selected_arm": HoldoutArm.INCUMBENT.value,
            "selected_score": incumbent_holdout.score,
            "delta_to_incumbent": 0.0,
        }
        incumbent_comparison = GenerationComparison(
            comparison_id="comparison:adaptive-reflection:no-improvement",
            run_id="run:adaptive-reflection",
            generation=0,
            cohort_digest="c" * 64,
            holdout_evaluations=(
                failed_holdout,
                winner_holdout,
                incumbent_holdout,
            ),
            selected_candidate_id=incumbent_holdout.scope.candidate_id,
            selected_revision_id=incumbent_holdout.scope.candidate_revision_id,
            gate_results=incumbent_gate_results,
        )
        incumbent_state = SimpleNamespace(
            **{
                **vars(state),
                "comparison_for": lambda generation: (
                    incumbent_comparison if generation == 0 else None
                ),
            }
        )
        no_improvement = _build_adaptive_analysis(
            incumbent_state,
            0,
            incumbent_comparison,
            (candidates[0], candidates[3]),
            incumbent_revision.candidate_id,
        )
        self.assertIsNone(no_improvement.selected_candidate_id)
        self.assertEqual(no_improvement.outcome, "no_improvement")
        self.assertEqual(
            no_improvement.search_parent_candidate_id,
            incumbent_revision.candidate_id,
        )
        self.assertEqual(
            no_improvement.incumbent_after_candidate_id,
            incumbent_revision.candidate_id,
        )
        self.assertIsNone(no_improvement.champion_candidate_id)
        self.assertIn("incumbent", no_improvement.next_search_direction[0])
        self.assertNotIn("冠军", no_improvement.next_search_direction[0])
        self.assertTrue(all(row["rank"] is None for row in no_improvement.ranking))
        finalist_rows = [
            row
            for row in no_improvement.ranking
            if row.get("holdout_arm")
            in {HoldoutArm.FINALIST_1.value, HoldoutArm.FINALIST_2.value}
        ]
        self.assertTrue(
            all(
                row["selection_reason"].startswith("holdout_gate_failed:")
                for row in finalist_rows
            )
        )
        no_improvement_reflection = _adaptive_reflection_analysis(
            incumbent_state,
            no_improvement,
        )
        self.assertEqual(
            no_improvement_reflection["adaptive_epoch_evidence"]["selected"][
                "revision_id"
            ],
            incumbent_revision.revision_id,
        )

        exploration_comparison = GenerationComparison(
            comparison_id="comparison:adaptive-reflection:exploration",
            run_id="run:adaptive-reflection",
            generation=0,
            cohort_digest="c" * 64,
            holdout_evaluations=(
                failed_holdout,
                winner_holdout,
                incumbent_holdout,
            ),
            selected_candidate_id=incumbent_holdout.scope.candidate_id,
            selected_revision_id=incumbent_holdout.scope.candidate_revision_id,
            gate_results={
                **incumbent_gate_results,
                "challenger_promotion_allowed": False,
            },
        )
        exploration_state = SimpleNamespace(
            **{
                **vars(state),
                "comparison_for": lambda generation: (
                    exploration_comparison if generation == 0 else None
                ),
                "formal_selection_for": lambda generation: (
                    SimpleNamespace(
                        payload={
                            "schema_version": (
                                "ecologyrsi-dsh.formal-selection-cohort/3"
                            ),
                            "exploration_only": True,
                            "consecutive_exploration_generations": 2,
                        }
                    )
                    if generation == 0
                    else None
                ),
            }
        )
        exploration = _build_adaptive_analysis(
            exploration_state,
            0,
            exploration_comparison,
            (candidates[0], candidates[3]),
            incumbent_revision.candidate_id,
        )
        self.assertEqual(exploration.outcome, "exploration_only")
        self.assertTrue(exploration.replan_required)
        self.assertEqual(exploration.consecutive_exploration_generations, 2)
        self.assertTrue(exploration.to_dict()["replan_required"])

        positive_exploration_comparison = GenerationComparison(
            comparison_id="comparison:adaptive-reflection:positive-exploration",
            run_id="run:adaptive-reflection",
            generation=0,
            cohort_digest="c" * 64,
            holdout_evaluations=(
                failed_holdout,
                winner_holdout,
                incumbent_holdout,
            ),
            selected_candidate_id=winner_holdout.scope.candidate_id,
            selected_revision_id=winner_holdout.scope.candidate_revision_id,
            gate_results={
                **search_only_comparison.to_dict()["gate_results"],
                "challenger_promotion_allowed": False,
            },
        )
        positive_exploration_state = SimpleNamespace(
            **{
                **vars(exploration_state),
                "comparison_for": lambda generation: (
                    positive_exploration_comparison if generation == 0 else None
                ),
            }
        )
        positive_exploration = _build_adaptive_analysis(
            positive_exploration_state,
            0,
            positive_exploration_comparison,
            (candidates[0], candidates[3]),
            incumbent_revision.candidate_id,
        )
        self.assertEqual(
            positive_exploration.search_parent_candidate_id,
            winner_holdout.scope.candidate_id,
        )
        self.assertTrue(positive_exploration.replan_required)
        self.assertIn("重新生成候选方向", positive_exploration.next_generation_focus)
        self.assertIn("下一轮搜索版本", positive_exploration.selection_reason)
        self.assertNotIn("仅保留探索证据", positive_exploration.selection_reason)


if __name__ == "__main__":
    unittest.main()
