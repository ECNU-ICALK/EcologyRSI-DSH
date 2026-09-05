from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import (
    Candidate,
    CandidateRole,
    CandidateStatus,
    Evaluation,
    HumanIntervention,
    InterventionKind,
    ModelArtifact,
    Proposal,
    Run,
    TaskManifest,
    digest,
)
from ecologyrsi_dsh.evolution.batches import (
    finalize_generation_batch,
    start_generation_batch,
)
from ecologyrsi_dsh.evolution.analysis import (
    GENERATION_CONTROL_EVALUATION_SCHEMA,
    GENERATION_CONTROL_POLICY,
    GenerationAnalysis,
    GenerationBatch,
    _rank_key,
    build_generation_analysis,
    generation_control_evaluations,
)
from ecologyrsi_dsh.evolution.strategies import FakeDSHAdapter, StrategyRouterDSHAdapter
from ecologyrsi_dsh.evaluators.fitness import (
    EXPLORATORY_EVIDENCE_CLASS,
    FitnessAssessment,
)
from ecologyrsi_dsh.presentation.reporting import rounds


class _CapturingPolicyGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, dict]] = []

    def catalog(self) -> list[dict]:
        return []

    def propose(
        self,
        model_id: str,
        context: dict,
        allowed_parameters: dict,
    ) -> dict:
        self.calls.append((model_id, context, allowed_parameters))
        return {
            "parameters": {"alpha": 0.44},
            "rationale": "基于父候选聚合证据做局部修改。",
        }


def _task(strategy_id: str, *, policy_model_id: str | None = None) -> TaskManifest:
    metadata = {"strategy_id": strategy_id, "domain": "toy"}
    if policy_model_id is not None:
        metadata["policy_model_id"] = policy_model_id
    return TaskManifest(
        task_id="feedback-loop",
        objective="根据已完成评测改进下一轮参数",
        domain_pack="crop-soil-water@toy",
        visible_datasets=("generated-toy-series@1",),
        budget={"max_candidates": 3, "max_generations": 3},
        metadata=metadata,
    )


def _batched_task() -> TaskManifest:
    return TaskManifest(
        task_id="batched-feedback-loop",
        objective="在正式门禁未通过时继续有界搜索",
        domain_pack="crop-soil-water@toy",
        visible_datasets=("generated-toy-series@1",),
        budget={
            "max_candidates": 4,
            "max_generations": 2,
            "candidates_per_generation": 2,
        },
        metadata={"strategy_id": "parameter_sweep@1", "domain": "toy"},
    )


def _parent_context(*, score: float, improvement: float) -> dict:
    return {
        "candidate_id": "candidate:parent",
        "status": "promoted",
        "proposal_id": "proposal:parent",
        "proposal_parameters": {
            "alpha": 0.5,
            "window": 8,
            "water_threshold": 0.4,
        },
        "artifact": {
            "training_partition": "training_fit",
            "training_rows": 30,
            "metrics": {"training_rmse": 0.2},
        },
        "evaluation": {
            "score": score,
            "passed": False,
            "partition": "training_feedback",
            "metrics": {"improvement": improvement, "rmse": 0.3},
        },
        "judge": {
            "accepted": False,
            "guidance": "收窄搜索窗口。",
            "parameter_override": {"window": 4},
        },
    }


class EvolutionFeedbackLoopTests(unittest.TestCase):
    def test_incumbent_control_is_isolated_from_analysis_advance_and_next_batch_budget(
        self,
    ) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        try:
            run_id = director.start_evolution(
                _batched_task(), run_id="run:budget-neutral-control"
            ).run.run_id
            control_proposal = Proposal(
                proposal_id="proposal:budget-neutral-control",
                run_id=run_id,
                generation=0,
                title="budget-neutral control",
                changes={"alpha": 0.25},
                metadata={"candidate_role": CandidateRole.INCUMBENT_CONTROL.value},
            )
            director.submit_proposal(control_proposal)
            director.spawn_candidate(
                run_id,
                control_proposal,
                candidate_id="candidate:budget-neutral-control",
                role=CandidateRole.INCUMBENT_CONTROL,
            )

            batch, _candidates = self._record_batch(
                director,
                run_id,
                (0.3, 0.2),
                judge_acceptances=(False, False),
            )
            analysis = finalize_generation_batch(director, run_id)

            self.assertEqual(batch.batch_size, 2)
            self.assertEqual(analysis.candidate_count, 2)
            director.advance_generation(run_id)
            next_batch = start_generation_batch(director, run_id)
            self.assertEqual(next_batch.batch_size, 2)
        finally:
            ledger.close()

    def test_exploratory_evaluated_row_sorts_ahead_of_unevaluated_duplicate(
        self,
    ) -> None:
        evaluated = {
            "candidate_id": "candidate:evaluated",
            "slot_index": 0,
            "score": -0.1,
            "evidence_class": "exploratory_adaptive_data",
            "validity_pass": False,
            "primary_selection_gate": False,
            "primary_selection_stability_floor": -0.2,
            "robustness_min_cell_delta": -0.3,
            "uq_pass_or_not_required": True,
            "interval_score": 0.1,
            "efficiency_score": 0.0,
            "complexity": 1.0,
        }
        duplicate = {
            "candidate_id": "candidate:duplicate",
            "slot_index": 1,
            "score": None,
            "scientific_pass": False,
            "judge_accepted": False,
            "constraint_violations": 0,
        }

        ranked = sorted((duplicate, evaluated), key=_rank_key, reverse=True)

        self.assertEqual(ranked[0]["candidate_id"], "candidate:evaluated")

    def test_diagnostic_exploratory_rows_rank_by_primary_score_not_candidate_id(
        self,
    ) -> None:
        common = {
            "slot_index": 0,
            "evidence_class": "exploratory_adaptive_data",
            "validity_pass": False,
            "primary_selection_gate": False,
            "primary_selection_stability_floor": None,
            "robustness_min_cell_delta": -1.0,
            "robustness_lower_quartile_cell_delta": -0.1,
            "uq_pass_or_not_required": True,
            "interval_score": None,
            "execution_policy_score": 1.0,
            "efficiency_score": 0.0,
            "complexity": 0.0,
        }
        higher = {
            **common,
            "candidate_id": "candidate:a-higher",
            "score": 0.02,
            "primary_score": 0.02,
        }
        lower = {
            **common,
            "candidate_id": "candidate:z-lower",
            "score": -0.15,
            "primary_score": -0.15,
        }

        ranked = sorted((lower, higher), key=_rank_key, reverse=True)

        self.assertEqual(ranked[0]["candidate_id"], "candidate:a-higher")

    def test_unevaluated_duplicate_has_no_scientific_rank(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        try:
            run_id = director.start_evolution(
                _batched_task(), run_id="run:duplicate-unranked"
            ).run.run_id
            batch = start_generation_batch(director, run_id)
            first_proposal = director.request_proposal(
                run_id,
                generation_batch=batch,
                slot_index=0,
                consume_interventions=False,
            )
            first = director.spawn_candidate(
                run_id, first_proposal, slot_index=0
            )
            artifact = ModelArtifact(
                artifact_id="artifact:duplicate-unranked:0",
                run_id=run_id,
                candidate_id=first.candidate_id,
                model_id="toy-rolling-water@1",
                dataset_digest="dataset-digest",
                training_partition="training_fit",
                training_rows=30,
                parameters=first_proposal.changes,
            )
            director.record_artifact(artifact)
            director.record_evaluation(
                Evaluation(
                    evaluation_id="evaluation:duplicate-unranked:0",
                    run_id=run_id,
                    candidate_id=first.candidate_id,
                    score=-0.1,
                    passed=False,
                    metrics={
                        "scientific_pass": False,
                        "judge_status": "completed",
                        "judge_accepted": False,
                        "constraint_violations": 0,
                        "skill_score": -0.1,
                    },
                    partition="training_feedback",
                    evaluator_digest="toy-evaluator@1",
                    artifact_digest=artifact.digest,
                )
            )
            second_proposal = director.request_proposal(
                run_id,
                generation_batch=batch,
                slot_index=1,
                consume_interventions=False,
            )
            second = director.spawn_candidate(
                run_id, second_proposal, slot_index=1
            )
            director.mark_candidate_duplicate(
                run_id, second.candidate_id, first.candidate_id
            )

            analysis = finalize_generation_batch(director, run_id)

            by_id = {row["candidate_id"]: row for row in analysis.ranking}
            self.assertEqual(by_id[first.candidate_id]["rank"], 1)
            self.assertIsNone(by_id[second.candidate_id]["rank"])
            self.assertEqual(
                analysis.ranking[-1]["candidate_id"], second.candidate_id
            )
        finally:
            ledger.close()

    @staticmethod
    def _record_batch(
        director: EvolutionDirector,
        run_id: str,
        scores: tuple[float, float],
        *,
        first_scientific_only: bool = False,
        all_eligible: bool = False,
        cohort_digests: tuple[str, str] | None = None,
        judge_acceptances: tuple[bool, bool] | None = None,
    ) -> tuple[object, tuple[object, object]]:
        batch = start_generation_batch(director, run_id)
        candidates = []
        for slot_index, score in enumerate(scores):
            proposal = director.request_proposal(
                run_id,
                generation_batch=batch,
                slot_index=slot_index,
                consume_interventions=False,
            )
            candidate = director.spawn_candidate(
                run_id,
                proposal,
                slot_index=slot_index,
            )
            artifact = ModelArtifact(
                artifact_id=f"artifact:{run_id}:{batch.generation}:{slot_index}",
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                model_id="toy-rolling-water@1",
                dataset_digest="dataset-digest",
                training_partition="training_fit",
                training_rows=30,
                parameters=proposal.changes,
                learned_parameters={},
                metrics={"training_rmse": 0.2},
            )
            director.record_artifact(artifact)
            metrics = {
                "scientific_pass": bool(
                    all_eligible or (first_scientific_only and slot_index == 0)
                ),
                "constraint_violations": 0,
                "skill_score": score,
                "targets": [
                    {
                        "target": "soil_water",
                        "horizon_hours": 1,
                        "unit": "fraction",
                        "skill_score": score,
                        "n": 20,
                        "prediction_preview": [{"observed": 0.4, "predicted": 0.5}],
                    }
                ],
            }
            if cohort_digests is not None:
                metrics["evaluation_index_digest"] = cohort_digests[slot_index]
                metrics["feedback_update_cohort_digest"] = cohort_digests[
                    slot_index
                ]
                metrics["evaluation_cohort"] = {
                    "update_window": {
                        "schema_version": (
                            "ecologyrsi-dsh.feedback-update-cohort/1"
                        ),
                        "selection_policy": (
                            "target_horizon_interleaved_rotating_window@1"
                        ),
                        "population_count": 1000,
                        "population_digest": "f" * 64,
                        "selected_count": 500,
                        "window_offset": batch.generation * 500 % 1000,
                        "cohort_digest": cohort_digests[slot_index],
                    }
                }
            if judge_acceptances is not None:
                metrics.update(
                    {
                        "judge_status": "completed",
                        "judge_accepted": judge_acceptances[slot_index],
                    }
                )
            elif all_eligible:
                metrics.update(
                    {"judge_status": "completed", "judge_accepted": True}
                )
            elif not (first_scientific_only and slot_index == 0):
                metrics.update(
                    {"judge_status": "completed", "judge_accepted": False}
                )
            director.record_evaluation(
                Evaluation(
                    evaluation_id=f"evaluation:{run_id}:{batch.generation}:{slot_index}",
                    run_id=run_id,
                    candidate_id=candidate.candidate_id,
                    score=score,
                    passed=bool(
                        all_eligible or (first_scientific_only and slot_index == 0)
                    ),
                    metrics=metrics,
                    partition="training_feedback",
                    evaluator_digest="toy-evaluator@1",
                    artifact_digest=artifact.digest,
                )
            )
            candidates.append(candidate)
        return batch, (candidates[0], candidates[1])

    @staticmethod
    def _windowed_batched_task() -> TaskManifest:
        data = _batched_task().to_dict()
        data["metadata"] = {
            **data["metadata"],
            "samples_per_update": 500,
        }
        return TaskManifest.from_dict(data)

    def test_native_generation_ranking_preserves_candidate_slot_identity(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        try:
            run_id = director.start_evolution(
                _batched_task(), run_id="run:native-slot-identity"
            ).run.run_id
            batch, candidates = self._record_batch(
                director,
                run_id,
                (-0.2, -0.1),
                judge_acceptances=(False, False),
            )

            state = director.state(run_id)
            task_data = state.task_manifest.to_dict()
            task_data["metadata"] = {
                **task_data["metadata"],
                "execution_protocol": "dsh_native_plugin_evolution@1",
                "fitness_profile": {
                    "expected_targets": ["soil_water"],
                    "expected_horizons": [1],
                },
            }

            class NativeAnalysisState:
                task_manifest = TaskManifest.from_dict(task_data)

                def __getattr__(self, name: str) -> object:
                    return getattr(state, name)

            analysis = build_generation_analysis(NativeAnalysisState(), batch)

            by_id = {row["candidate_id"]: row for row in analysis.ranking}
            self.assertEqual(by_id[candidates[0].candidate_id]["slot_index"], 0)
            self.assertEqual(by_id[candidates[1].candidate_id]["slot_index"], 1)
        finally:
            ledger.close()

    def test_native_generation_analysis_serializes_missing_cell_diagnostics(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        try:
            run_id = director.start_evolution(
                _batched_task(), run_id="run:native-missing-cells"
            ).run.run_id
            batch, _candidates = self._record_batch(
                director,
                run_id,
                (0.2, 0.1),
                all_eligible=True,
            )
            state = director.state(run_id)
            task_data = state.task_manifest.to_dict()
            task_data["metadata"] = {
                **task_data["metadata"],
                "execution_protocol": "dsh_native_plugin_evolution@1",
                "fitness_profile": {
                    "expected_targets": ["unreported_target"],
                    "expected_horizons": [1],
                },
            }

            class NativeAnalysisState:
                task_manifest = TaskManifest.from_dict(task_data)

                def __getattr__(self, name: str) -> object:
                    return getattr(state, name)

            analysis = build_generation_analysis(NativeAnalysisState(), batch)

            self.assertEqual(analysis.outcome, "no_improvement")
            self.assertTrue(
                all(
                    row["robustness_min_cell_delta"] is None
                    and row["robustness_lower_quartile_cell_delta"] is None
                    for row in analysis.ranking
                )
            )
        finally:
            ledger.close()

    def test_native_diagnostic_budget_cannot_create_a_champion(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        try:
            run_id = director.start_evolution(
                _batched_task(), run_id="run:native-diagnostic-no-champion"
            ).run.run_id
            batch, candidates = self._record_batch(
                director,
                run_id,
                (0.8, 0.7),
                all_eligible=True,
            )
            state = director.state(run_id)
            task_data = state.task_manifest.to_dict()
            task_data["metadata"] = {
                **task_data["metadata"],
                "execution_protocol": "dsh_native_plugin_evolution@1",
                "sample_budget_class": "diagnostic_smoke",
                "fitness_profile": {
                    "expected_targets": ["soil_water"],
                    "expected_horizons": [1],
                },
            }

            class DiagnosticAnalysisState:
                task_manifest = TaskManifest.from_dict(task_data)

                def __getattr__(self, name: str) -> object:
                    return getattr(state, name)

            def passing_assessment(evaluation, *_args, **_kwargs):
                score = float(evaluation.score)
                return FitnessAssessment(
                    candidate_id=evaluation.candidate_id,
                    evidence_class=EXPLORATORY_EVIDENCE_CLASS,
                    validity_pass=True,
                    validity_failures=(),
                    primary_score=score,
                    primary_delta=score,
                    primary_selection_gate=True,
                    primary_selection_stability_floor=0.1,
                    robustness_pass=True,
                    robustness_min_cell_delta=0.1,
                    robustness_lower_quartile_cell_delta=0.1,
                    uq_pass_or_not_required=True,
                    interval_score=None,
                    execution_policy_score=1.0,
                    efficiency_score=0.0,
                    complexity=0.0,
                    slot_index=int(evaluation.metrics.get("slot_index", 0)),
                )

            with patch(
                "ecologyrsi_dsh.evolution.analysis.build_fitness_assessment",
                side_effect=passing_assessment,
            ):
                analysis = build_generation_analysis(
                    DiagnosticAnalysisState(), batch
                )

            self.assertEqual(analysis.outcome, "no_improvement")
            self.assertIsNone(analysis.champion_candidate_id)
            self.assertIsNone(analysis.incumbent_after_candidate_id)
            self.assertEqual(
                analysis.search_parent_candidate_id,
                candidates[0].candidate_id,
            )
            self.assertEqual(
                analysis.ranking[0]["selection_reason"],
                "diagnostic_smoke_search_parent_only",
            )
            self.assertIn("不生成冠军", analysis.selection_reason)
        finally:
            ledger.close()

    def test_second_diagnostic_generation_uses_prior_search_parent_without_controls(
        self,
    ) -> None:
        run_id = "run:diagnostic-two-generations"
        task = TaskManifest(
            task_id="diagnostic-two-generations",
            objective="exercise two diagnostic generations",
            domain_pack="crop-soil-water@toy",
            visible_datasets=("generated-toy-series@1",),
            budget={
                "max_candidates": 2,
                "max_generations": 2,
                "candidates_per_generation": 1,
            },
            metadata={
                "strategy_id": "parameter_sweep@1",
                "execution_protocol": "dsh_native_plugin_evolution@1",
                "sample_agent_protocol": "dsh-strict-origin-bundle@3",
                "sample_budget_class": "diagnostic_smoke",
                "samples_per_update": 1,
                "prediction_cells_per_origin": 1,
                "fitness_profile": {
                    "expected_targets": ["soil_water"],
                    "expected_horizons": [1],
                    "selection_minimum_cell_samples": 1,
                    "selection_minimum_paired_blocks": 1,
                    "selection_minimum_valid_three_day_starts": 1,
                    "moving_block_days": 1,
                    "exploratory_resamples": 10,
                },
            },
        )
        first_candidate = Candidate(
            candidate_id="candidate:diagnostic:first",
            run_id=run_id,
            proposal_id="proposal:diagnostic:first",
            generation=0,
            slot_index=0,
            status=CandidateStatus.REJECTED,
        )
        second_candidate = Candidate(
            candidate_id="candidate:diagnostic:second",
            run_id=run_id,
            proposal_id="proposal:diagnostic:second",
            generation=1,
            slot_index=0,
            status=CandidateStatus.EVALUATED,
        )
        proposals = {
            first_candidate.proposal_id: Proposal(
                proposal_id=first_candidate.proposal_id,
                run_id=run_id,
                generation=0,
                title="first diagnostic proposal",
                changes={"alpha": 0.2},
            ),
            second_candidate.proposal_id: Proposal(
                proposal_id=second_candidate.proposal_id,
                run_id=run_id,
                generation=1,
                title="second diagnostic proposal",
                changes={"alpha": 0.35},
                parent_candidate_id=first_candidate.candidate_id,
            ),
        }

        def evaluation(candidate: Candidate, score: float, cohort: str) -> Evaluation:
            return Evaluation(
                evaluation_id=f"evaluation:{candidate.candidate_id}",
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                score=score,
                passed=True,
                metrics={
                    "scientific_pass": True,
                    "judge_status": "completed",
                    "judge_accepted": True,
                    "constraint_violations": 0,
                    "skill_score": score,
                    "evaluation_index_digest": cohort,
                    "feedback_update_cohort_digest": cohort,
                    "sample_execution": {
                        "strict_agent_contract": True,
                        "strict_agent_chain_pass": True,
                        "attempted_examples": 1,
                        "attempted_origin_samples": 1,
                        "prediction_cells_per_origin": 1,
                        "complete_agent_chains": 1,
                        "host_route_bypass_count": 0,
                    },
                    "targets": [
                        {
                            "target": "soil_water",
                            "horizon_hours": 1,
                            "unit": "fraction",
                            "skill_score": score,
                            "n": 1,
                        }
                    ],
                },
                partition="training_feedback",
                evaluator_digest="toy-evaluator@1",
                artifact_digest="a" * 64,
            )

        evaluations = {
            first_candidate.candidate_id: evaluation(
                first_candidate, 0.01, "a" * 64
            ),
            second_candidate.candidate_id: evaluation(
                second_candidate, 0.02, "b" * 64
            ),
        }
        first_analysis = GenerationAnalysis(
            run_id=run_id,
            generation=0,
            candidate_count=1,
            eligible_count=1,
            outcome="no_improvement",
            selected_candidate_id=first_candidate.candidate_id,
            search_parent_candidate_id=first_candidate.candidate_id,
            ranking=(
                {
                    "candidate_id": first_candidate.candidate_id,
                    "score": evaluations[first_candidate.candidate_id].score,
                    "constraint_violations": 0,
                },
            ),
        )
        second_batch = GenerationBatch(
            run_id=run_id,
            generation=1,
            batch_size=1,
            task_manifest_digest=task.digest,
            parent_candidate_id=first_candidate.candidate_id,
            previous_analysis_digest=first_analysis.analysis_digest,
        )

        class DiagnosticState:
            task_manifest = task
            run = SimpleNamespace(run_id=run_id, best_candidate_id=None)
            candidates = (first_candidate, second_candidate)

            @staticmethod
            def candidate(candidate_id: str) -> Candidate:
                return next(
                    item
                    for item in DiagnosticState.candidates
                    if item.candidate_id == candidate_id
                )

            @staticmethod
            def proposal(proposal_id: str) -> Proposal:
                return proposals[proposal_id]

            @staticmethod
            def evaluation_for(candidate_id: str) -> Evaluation | None:
                return evaluations.get(candidate_id)

            @staticmethod
            def analysis_for(generation: int) -> GenerationAnalysis | None:
                return first_analysis if generation == 0 else None

            @staticmethod
            def algorithm_attempts_for(_candidate_id: str) -> tuple[object, ...]:
                return ()

        analysis = build_generation_analysis(DiagnosticState(), second_batch)

        self.assertEqual(
            second_batch.parent_candidate_id,
            first_candidate.candidate_id,
        )
        self.assertEqual(analysis.generation, 1)
        self.assertEqual(analysis.outcome, "no_improvement")
        self.assertIsNone(analysis.champion_candidate_id)
        self.assertEqual(
            analysis.search_parent_candidate_id,
            second_candidate.candidate_id,
        )
        self.assertEqual(
            analysis.ranking[0]["selection_reason"],
            "diagnostic_smoke_search_parent_only",
        )

    def test_changed_update_window_keeps_formal_incumbent_and_uses_search_parent(
        self,
    ) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        first_digest = "a" * 64
        second_digest = "b" * 64
        try:
            run_id = director.start_evolution(
                self._windowed_batched_task(), run_id="run:windowed-champion"
            ).run.run_id
            _first_batch, first_candidates = self._record_batch(
                director,
                run_id,
                (0.8, 0.7),
                all_eligible=True,
                cohort_digests=(first_digest, first_digest),
            )
            first = finalize_generation_batch(director, run_id)
            self.assertEqual(first.champion_candidate_id, first_candidates[0].candidate_id)
            self.assertEqual(
                first.ranking[0]["evaluation_cohort_window"]["selected_count"],
                500,
            )

            director.advance_generation(run_id)
            _second_batch, second_candidates = self._record_batch(
                director,
                run_id,
                (0.4, 0.3),
                all_eligible=True,
                cohort_digests=(second_digest, second_digest),
            )
            second = finalize_generation_batch(director, run_id)

            self.assertEqual(second.outcome, "no_improvement")
            self.assertIsNone(second.champion_candidate_id)
            self.assertEqual(second.search_parent_candidate_id, second_candidates[0].candidate_id)
            self.assertEqual(
                second.ranking[0]["selection_reason"],
                "cohort_changed_search_parent_only",
            )
            self.assertEqual(second.parameter_effects, ())
            self.assertIn("未比较跨窗口原始分数", second.selection_reason)
            self.assertEqual(
                director.state(run_id).run.best_candidate_id,
                first_candidates[0].candidate_id,
            )
        finally:
            ledger.close()

    def test_same_update_window_retains_better_prior_search_parent(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        cohort_digest = "a" * 64
        try:
            run_id = director.start_evolution(
                self._windowed_batched_task(), run_id="run:search-elite"
            ).run.run_id
            _first_batch, first_candidates = self._record_batch(
                director,
                run_id,
                (-0.65, -0.70),
                cohort_digests=(cohort_digest, cohort_digest),
                judge_acceptances=(False, False),
            )
            first = finalize_generation_batch(director, run_id)
            self.assertEqual(
                first.search_parent_candidate_id,
                first_candidates[0].candidate_id,
            )

            director.advance_generation(run_id)
            _second_batch, second_candidates = self._record_batch(
                director,
                run_id,
                (-0.66, -0.72),
                cohort_digests=(cohort_digest, cohort_digest),
                judge_acceptances=(False, False),
            )
            second = finalize_generation_batch(director, run_id)

            self.assertEqual(
                second.ranking[0]["candidate_id"],
                second_candidates[0].candidate_id,
            )
            self.assertEqual(
                second.search_parent_candidate_id,
                first_candidates[0].candidate_id,
            )
            self.assertIn("保留该父方案以避免跨轮退化", second.selection_reason)
        finally:
            ledger.close()

    def test_strict_generation_uses_same_cohort_control_and_retains_parent(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        try:
            run_id = director.start_evolution(
                self._windowed_batched_task(), run_id="run:strict-control"
            ).run.run_id
            _first_batch, first_candidates = self._record_batch(
                director,
                run_id,
                (0.40, 0.30),
                all_eligible=True,
                cohort_digests=("a" * 64, "a" * 64),
            )
            first = finalize_generation_batch(director, run_id)
            parent_id = first_candidates[0].candidate_id
            self.assertEqual(first.champion_candidate_id, parent_id)

            director.advance_generation(run_id)
            second_batch, second_candidates = self._record_batch(
                director,
                run_id,
                (0.39, 0.38),
                all_eligible=True,
                cohort_digests=("b" * 64, "b" * 64),
            )
            state = director.state(run_id)
            strict_summary = {
                "strict_agent_contract": True,
                "strict_agent_chain_pass": True,
                "attempted_examples": 1,
                "attempted_origin_samples": 1,
                "prediction_cells_per_origin": 1,
                "complete_agent_chains": 1,
                "host_route_bypass_count": 0,
            }
            control_data = state.evaluation_for(parent_id).to_dict()
            control_data["evaluation_id"] += ":generation-1-control"
            control_metrics = dict(control_data["metrics"])
            control_metrics.update(
                {
                    "evaluation_index_digest": "b" * 64,
                    "feedback_update_cohort_digest": "b" * 64,
                    "sample_execution": strict_summary,
                }
            )
            control_data["metrics"] = control_metrics
            controls = [
                {
                    "schema_version": GENERATION_CONTROL_EVALUATION_SCHEMA,
                    "generation": 1,
                    "comparison_role": "search_parent_and_formal_elite",
                    "candidate_id": parent_id,
                    "evaluation": control_data,
                }
            ]
            overrides = {}
            for candidate in second_candidates:
                evaluation_data = state.evaluation_for(candidate.candidate_id).to_dict()
                metrics = dict(evaluation_data["metrics"])
                metrics["sample_execution"] = strict_summary
                if candidate.slot_index == 0:
                    metrics.update(
                        {
                            "generation_control_policy": GENERATION_CONTROL_POLICY,
                            "generation_control_evaluations": controls,
                            "generation_control_evidence_digest": digest(controls),
                        }
                    )
                evaluation_data["metrics"] = metrics
                overrides[candidate.candidate_id] = Evaluation.from_dict(
                    evaluation_data
                )
            task_data = state.task_manifest.to_dict()
            task_data["metadata"] = {
                **task_data["metadata"],
                "execution_protocol": "dsh_native_plugin_evolution@1",
                "sample_agent_protocol": "dsh-strict-origin-bundle@3",
                "sample_budget_class": "selection_eligible",
                "minimum_selection_samples_per_update": 1,
                "minimum_selection_origin_samples_per_update": 1,
                "prediction_cells_per_origin": 1,
                "fitness_profile": {
                    "expected_targets": ["soil_water"],
                    "expected_horizons": [1],
                    "selection_minimum_cell_samples": 1,
                    "selection_minimum_paired_blocks": 1,
                    "selection_minimum_valid_three_day_starts": 1,
                    "moving_block_days": 1,
                    "exploratory_resamples": 10,
                },
            }

            class StrictAnalysisState:
                task_manifest = TaskManifest.from_dict(task_data)

                def evaluation_for(self, candidate_id: str):
                    return overrides.get(candidate_id, state.evaluation_for(candidate_id))

                def __getattr__(self, name: str) -> object:
                    return getattr(state, name)

            strict_state = StrictAnalysisState()
            resolved = generation_control_evaluations(strict_state, second_batch)
            analysis = build_generation_analysis(strict_state, second_batch)

            self.assertEqual(resolved["search_parent"][0], parent_id)
            self.assertEqual(resolved["formal_elite"][0], parent_id)
            self.assertEqual(analysis.outcome, "no_improvement")
            self.assertIsNone(analysis.champion_candidate_id)
            self.assertEqual(analysis.search_parent_candidate_id, parent_id)
            self.assertTrue(
                all(
                    row["search_parent_selection_gate"] is False
                    for row in analysis.ranking
                )
            )
            self.assertIn("保留该父方案以避免跨轮退化", analysis.selection_reason)
        finally:
            ledger.close()

    def test_windowed_generation_rejects_mismatched_sibling_cohorts(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        try:
            run_id = director.start_evolution(
                self._windowed_batched_task(), run_id="run:mismatched-window"
            ).run.run_id
            self._record_batch(
                director,
                run_id,
                (0.4, 0.3),
                all_eligible=True,
                cohort_digests=("a" * 64, "b" * 64),
            )
            with self.assertRaisesRegex(
                RuntimeError, "different bounded evaluation cohorts"
            ):
                finalize_generation_batch(director, run_id)
        finally:
            ledger.close()

    def test_search_parent_is_separate_from_formal_incumbent(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        try:
            run_id = director.start_evolution(
                _batched_task(), run_id="run:search-parent"
            ).run.run_id
            _batch, candidates = self._record_batch(
                director,
                run_id,
                (-0.1, -0.4),
                first_scientific_only=True,
            )
            analysis = finalize_generation_batch(director, run_id)

            self.assertEqual(analysis.outcome, "no_eligible_candidate")
            self.assertEqual(analysis.eligible_count, 0)
            self.assertEqual(
                analysis.search_parent_candidate_id,
                candidates[0].candidate_id,
            )
            self.assertIsNone(director.state(run_id).run.best_candidate_id)
            row = next(
                item
                for item in analysis.ranking
                if item["candidate_id"] == candidates[0].candidate_id
            )
            self.assertEqual(row["classification"], "judge_unavailable")
            self.assertTrue(row["scientific_pass"])
            self.assertFalse(row["eligible"])
            self.assertEqual(row["parameters"], {"alpha": 0.2, "water_threshold": 0.35, "window": 3})
            self.assertEqual(row["target_skill_scores"][0]["skill_score"], -0.1)
            self.assertEqual(row["target_skill_scores"][0]["n"], 20)
            self.assertNotIn("prediction_preview", str(row))

            director.advance_generation(run_id)
            next_batch = start_generation_batch(director, run_id)
            self.assertEqual(next_batch.parent_candidate_id, candidates[0].candidate_id)
            self.assertIsNone(director.state(run_id).run.best_candidate_id)
        finally:
            ledger.close()

    def test_gate_failing_search_parent_uses_objective_evidence_before_judge(
        self,
    ) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        try:
            run_id = director.start_evolution(
                _batched_task(), run_id="run:objective-search-parent"
            ).run.run_id
            _batch, candidates = self._record_batch(
                director,
                run_id,
                (0.37, 0.33),
                judge_acceptances=(False, True),
            )

            analysis = finalize_generation_batch(director, run_id)

            self.assertEqual(analysis.outcome, "no_eligible_candidate")
            self.assertEqual(
                analysis.search_parent_candidate_id,
                candidates[0].candidate_id,
            )
            self.assertEqual(
                analysis.ranking[0]["candidate_id"],
                candidates[0].candidate_id,
            )
            self.assertEqual(analysis.ranking[0]["score"], 0.37)
            self.assertFalse(analysis.ranking[0]["judge_accepted"])
            self.assertTrue(analysis.ranking[1]["judge_accepted"])
            projected = rounds(director.state(run_id))[0]
            self.assertEqual(
                projected["candidate_id"], candidates[0].candidate_id
            )
            self.assertEqual(projected["score"], 0.37)
        finally:
            ledger.close()

    def test_search_parent_is_objective_while_promotion_still_requires_judge(
        self,
    ) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        try:
            run_id = director.start_evolution(
                _batched_task(), run_id="run:judge-filtered-promotion"
            ).run.run_id
            _batch, candidates = self._record_batch(
                director,
                run_id,
                (0.60, 0.40),
                all_eligible=True,
                judge_acceptances=(False, True),
            )

            analysis = finalize_generation_batch(director, run_id)

            self.assertEqual(analysis.eligible_count, 1)
            self.assertEqual(
                analysis.selected_candidate_id,
                candidates[1].candidate_id,
            )
            self.assertEqual(
                analysis.champion_candidate_id,
                candidates[1].candidate_id,
            )
            self.assertEqual(
                analysis.search_parent_candidate_id,
                candidates[0].candidate_id,
            )
            self.assertEqual(
                analysis.ranking[0]["candidate_id"],
                candidates[0].candidate_id,
            )
            self.assertFalse(analysis.ranking[0]["judge_accepted"])
            self.assertTrue(analysis.ranking[1]["judge_accepted"])
        finally:
            ledger.close()

    def test_parameter_effects_accumulate_unique_configurations_across_rounds(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        try:
            run_id = director.start_evolution(
                _batched_task(), run_id="run:cross-round-effects"
            ).run.run_id
            self._record_batch(director, run_id, (-0.4, -0.2))
            first = finalize_generation_batch(director, run_id)
            self.assertEqual(first.parameter_effects, ())

            director.advance_generation(run_id)
            self._record_batch(director, run_id, (-0.1, 0.05))
            second = finalize_generation_batch(director, run_id)

            self.assertTrue(second.parameter_effects)
            self.assertTrue(
                all(item["evidence_count"] >= 3 for item in second.parameter_effects)
            )
            self.assertTrue(
                all(
                    item["interpretation"]
                    == "cross_generation_observational_association_non_causal"
                    for item in second.parameter_effects
                )
            )
            self.assertFalse(second.insufficient_evidence)
        finally:
            ledger.close()

    def test_parameter_sweep_changes_one_parent_parameter_and_inherits_the_rest(self) -> None:
        run = Run(
            run_id="run:parent-sweep",
            task_id="feedback-loop",
            task_manifest_digest="digest",
            generation=0,
            session_id="strategy-dsh:run:parent-sweep",
        )
        adapter = StrategyRouterDSHAdapter()
        parent = _parent_context(score=0.4, improvement=0.1)
        parent["judge"]["parameter_override"] = {}
        task = _task("parameter_sweep@1")
        proposal = adapter.propose(
            run,
            task,
            adapter.open_session(run, task),
            parent_candidate_id="candidate:parent",
            parent_context=parent,
        )

        differences = {
            name
            for name, value in proposal.changes.items()
            if value != parent["proposal_parameters"][name]
        }
        self.assertEqual(differences, {"alpha"})
        self.assertEqual(proposal.changes["window"], 8)
        self.assertEqual(proposal.changes["water_threshold"], 0.4)

    def test_empty_constraint_is_rejected(self) -> None:
        run = Run(
            run_id="run:invalid-constraint",
            task_id="feedback-loop",
            task_manifest_digest="digest",
            generation=0,
            session_id="strategy-dsh:run:invalid-constraint",
        )
        task = _task("parameter_sweep@1")
        adapter = StrategyRouterDSHAdapter()
        with self.assertRaisesRegex(ValueError, "must be a non-empty string"):
            adapter.propose(
                run,
                task,
                adapter.open_session(run, task),
                interventions={"constraints": [None]},
            )

    def test_parent_context_identity_and_completion_are_enforced(self) -> None:
        run = Run(
            run_id="run:invalid-parent-context",
            task_id="feedback-loop",
            task_manifest_digest="digest",
            generation=0,
            session_id="strategy-dsh:run:invalid-parent-context",
        )
        task = _task("parameter_sweep@1")
        parent = _parent_context(score=0.4, improvement=0.1)
        parent["status"] = "running"
        adapter = StrategyRouterDSHAdapter()
        with self.assertRaisesRegex(ValueError, "completed candidate"):
            adapter.propose(
                run,
                task,
                adapter.open_session(run, task),
                parent_candidate_id="candidate:parent",
                parent_context=parent,
            )

        parent["status"] = "promoted"
        with self.assertRaisesRegex(ValueError, "does not match"):
            adapter.propose(
                run,
                task,
                adapter.open_session(run, task),
                parent_candidate_id="candidate:different",
                parent_context=parent,
            )

    def test_parent_sweep_moves_inward_from_parameter_boundary(self) -> None:
        run = Run(
            run_id="run:boundary-parent",
            task_id="feedback-loop",
            task_manifest_digest="digest",
            generation=0,
            session_id="strategy-dsh:run:boundary-parent",
        )
        task = _task("parameter_sweep@1")
        parent = _parent_context(score=0.4, improvement=0.1)
        parent["proposal_parameters"]["alpha"] = 0.95
        parent["judge"]["parameter_override"] = {}
        adapter = StrategyRouterDSHAdapter()
        proposal = adapter.propose(
            run,
            task,
            adapter.open_session(run, task),
            parent_candidate_id="candidate:parent",
            parent_context=parent,
        )
        self.assertLess(proposal.changes["alpha"], 0.95)

    def test_opposite_sweep_slots_remain_distinct_at_parameter_boundary(self) -> None:
        schemas = {
            "blend": {"type": "number", "minimum": 0.1, "maximum": 1.0},
            "window": {"type": "integer", "minimum": 1, "maximum": 48},
            "bias_scale": {"type": "number", "minimum": 0.0, "maximum": 2.0},
        }
        parent = {"blend": 0.9, "window": 24, "bias_scale": 0.0}

        positive = StrategyRouterDSHAdapter._sweep_around_parent(
            parent, schemas, 4
        )
        blocked_negative = StrategyRouterDSHAdapter._sweep_around_parent(
            parent, schemas, 5
        )

        self.assertEqual(positive["bias_scale"], 0.2)
        self.assertEqual(blocked_negative["bias_scale"], 0.4)
        self.assertNotEqual(positive, blocked_negative)

    def test_adaptive_strategy_uses_parent_parameters_and_evaluation_signal(self) -> None:
        run = Run(
            run_id="run:adaptive-feedback",
            task_id="feedback-loop",
            task_manifest_digest="digest",
            generation=1,
            session_id="strategy-dsh:run:adaptive-feedback",
        )
        task = _task("adaptive_local@1")

        positive_adapter = StrategyRouterDSHAdapter()
        positive = positive_adapter.propose(
            run,
            task,
            positive_adapter.open_session(run, task),
            parent_candidate_id="candidate:parent",
            parent_context=_parent_context(score=0.8, improvement=0.2),
        )
        negative_adapter = StrategyRouterDSHAdapter()
        negative = negative_adapter.propose(
            run,
            task,
            negative_adapter.open_session(run, task),
            parent_candidate_id="candidate:parent",
            parent_context=_parent_context(score=-0.8, improvement=-0.2),
        )

        self.assertNotEqual(positive.changes, negative.changes)
        self.assertLess(positive.changes["alpha"], 0.5)
        self.assertGreater(negative.changes["alpha"], 0.5)
        self.assertEqual(positive.changes["window"], 3)
        self.assertEqual(negative.changes["window"], 5)
        self.assertIn("已完成父候选", positive.rationale)
        self.assertIn("评审参数建议", positive.rationale)

    def test_authenticated_strategy_receives_redacted_parent_and_human_context(self) -> None:
        ledger = EventLedger()
        gateway = _CapturingPolicyGateway()
        director = EvolutionDirector(
            ledger,
            StrategyRouterDSHAdapter(gateway=gateway),  # type: ignore[arg-type]
        )
        try:
            task = _task("dsh_authenticated@1", policy_model_id="policy-main")
            run_id = director.start_evolution(
                task, run_id="run:authenticated-feedback"
            ).run.run_id
            first_proposal = director.submit_proposal(
                Proposal(
                    proposal_id="proposal:seed-parent",
                    run_id=run_id,
                    generation=0,
                    title="初始父候选",
                    changes={
                        "alpha": 0.5,
                        "window": 8,
                        "water_threshold": 0.4,
                    },
                )
            )
            parent = director.spawn_candidate(run_id, first_proposal)
            artifact = ModelArtifact(
                artifact_id="artifact:seed-parent",
                run_id=run_id,
                candidate_id=parent.candidate_id,
                model_id="toy-rolling-water@1",
                dataset_digest="dataset-digest",
                training_partition="training_fit",
                training_rows=36,
                parameters=first_proposal.changes,
                learned_parameters={"bias": 0.02},
                metrics={"training_rmse": 0.12},
            )
            director.record_artifact(artifact)
            director.evaluate_and_decide(
                Evaluation(
                    evaluation_id="evaluation:seed-parent",
                    run_id=run_id,
                    candidate_id=parent.candidate_id,
                    score=0.72,
                    passed=True,
                    metrics={
                        "rmse": 0.18,
                        "improvement": 0.08,
                        "prediction_preview": [
                            {"observed": 0.4, "predicted": 0.41}
                        ],
                        "judge_model_id": "judge-main",
                        "judge_accepted": True,
                        "judge_guidance": "继续缩短窗口。",
                        "judge_parameter_override": {"window": 7},
                    },
                    partition="training_feedback",
                    evaluator_digest="toy-evaluator@1",
                    artifact_digest=artifact.digest,
                )
            )
            director.advance_generation(run_id)
            director.pause_run(run_id)
            for intervention in (
                HumanIntervention(
                    intervention_id="guidance-1",
                    run_id=run_id,
                    kind=InterventionKind.GUIDANCE,
                    message="优先降低短期波动。",
                    created_by="研究者",
                ),
                HumanIntervention(
                    intervention_id="constraint-1",
                    run_id=run_id,
                    kind=InterventionKind.CONSTRAINT,
                    message="不得修改固定科学门禁。",
                    created_by="研究者",
                ),
                HumanIntervention(
                    intervention_id="parent-1",
                    run_id=run_id,
                    kind=InterventionKind.PARENT_SELECTION,
                    message="选择已完成候选继续演化。",
                    created_by="研究者",
                    target_candidate_id=parent.candidate_id,
                ),
            ):
                director.record_intervention(intervention)
            director.resume_run(run_id)

            proposal = director.request_proposal(
                run_id, parent_candidate_id=parent.candidate_id
            )
            self.assertEqual(
                proposal.changes,
                {"alpha": 0.44, "window": 7, "water_threshold": 0.4},
            )
            self.assertIn("未修改宿主科学门禁", proposal.rationale)
            self.assertEqual(director.state(run_id).pending_interventions, ())

            _, context, _ = gateway.calls[0]
            self.assertEqual(
                context["parent"]["proposal_parameters"],
                first_proposal.changes,
            )
            self.assertEqual(
                context["parent"]["artifact"]["metrics"]["training_rmse"],
                0.12,
            )
            self.assertEqual(context["parent"]["evaluation"]["score"], 0.72)
            self.assertNotIn(
                "prediction_preview",
                context["parent"]["evaluation"]["metrics"],
            )
            self.assertEqual(
                context["parent"]["judge"]["parameter_override"],
                {"window": 7},
            )
            self.assertEqual(
                context["human_input"]["guidance"], "优先降低短期波动。"
            )
            self.assertEqual(
                context["human_input"]["constraints"],
                ["不得修改固定科学门禁。"],
            )
            self.assertTrue(context["host_boundary"]["scientific_gate_is_not_mutable"])
        finally:
            ledger.close()

    def test_parent_selection_rejects_an_incomplete_candidate(self) -> None:
        ledger = EventLedger()
        director = EvolutionDirector(ledger, FakeDSHAdapter())
        try:
            run_id = director.start_evolution(
                _task("parameter_sweep@1"), run_id="run:incomplete-parent"
            ).run.run_id
            candidate = director.propose_and_spawn(run_id)
            director.pause_run(run_id)
            with self.assertRaisesRegex(ValueError, "completed promotion decision"):
                director.record_intervention(
                    HumanIntervention(
                        intervention_id="invalid-parent",
                        run_id=run_id,
                        kind=InterventionKind.PARENT_SELECTION,
                        message="这个候选尚未完成。",
                        created_by="研究者",
                        target_candidate_id=candidate.candidate_id,
                    )
                )
        finally:
            ledger.close()


if __name__ == "__main__":
    unittest.main()
