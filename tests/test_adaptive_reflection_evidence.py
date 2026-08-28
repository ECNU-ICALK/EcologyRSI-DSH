from __future__ import annotations

import unittest
from types import SimpleNamespace

from ecologyrsi_dsh.api.generation_execution import _build_adaptive_analysis
from ecologyrsi_dsh.core.models import CandidateStatus, digest
from ecologyrsi_dsh.core.trajectory import (
    CandidateRevision,
    EvaluationPhase,
    EvaluationScope,
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
from ecologyrsi_dsh.evolution.schedule import OPTIMIZATION_PROTOCOL


class AdaptiveReflectionEvidenceTests(unittest.TestCase):
    maxDiff = None

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
        formal_batches = []
        local_edit_proposals = []
        local_edit_outcomes = []
        activations = []
        trajectories = {}
        for candidate_id in finalist_ids:
            final_revision_id = finalist_revisions[candidate_id].revision_id
            current_revision_id = f"revision:{candidate_id}:r0"
            trajectories[candidate_id] = SimpleNamespace(
                initial_revision_id=current_revision_id,
                final_revision_id=final_revision_id,
                batch_count=10,
            )
            for batch_index in range(10):
                next_revision_id = (
                    final_revision_id
                    if batch_index == 9
                    else f"revision:{candidate_id}:r{batch_index + 1}"
                )
                formal_batches.append(
                    SimpleNamespace(
                        candidate_id=candidate_id,
                        batch_index=batch_index,
                        revision_id=current_revision_id,
                    )
                )
                local_edit_proposals.append(
                    {
                        "candidate_id": candidate_id,
                        "batch_index": batch_index,
                        "proposal": {
                            "decision": "mutate",
                            "operations": [
                                {
                                    "op": "set_bounded_parameter",
                                    "name": "ridge_alpha",
                                    "value": 0.2 + batch_index / 100,
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
                        from_revision_id=current_revision_id,
                        to_revision_id=next_revision_id,
                        reason=RevisionAdvanceReason.LOCAL_EDIT_APPLIED,
                    )
                )
                current_revision_id = next_revision_id

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
                metadata={"optimization_protocol": OPTIMIZATION_PROTOCOL}
            ),
            candidates=candidates,
            formal_batches=tuple(formal_batches),
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
        self.assertEqual(winner["local_edit_evidence"]["included_batch_count"], 10)
        self.assertEqual(len(winner["local_edit_evidence"]["batches"]), 10)
        self.assertEqual(len(winner["local_edit_evidence"]["revision_chain"]), 11)
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
        self.assertIsNotNone(
            safe_aggregate_feedback(
                reflection,
                name="adaptive reflection test evidence",
            )
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


if __name__ == "__main__":
    unittest.main()
