from __future__ import annotations
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from ecologyrsi_dsh.core.models import TaskManifest, canonical_json, digest
from ecologyrsi_dsh.evolution.analysis import CROSS_GENERATION_EXPERIENCE_MAX_BYTES, GenerationAnalysis, build_cross_generation_experience
from ecologyrsi_dsh.knowledge.algorithms import AlgorithmAttempt
from ecologyrsi_dsh.knowledge.research_iteration import ResearchIteration
from tests.native_fixtures import native_manifest

def _task(*, candidates_per_generation=4):
    if candidates_per_generation != 4:
        raise ValueError("current run has four candidate slots")
    return native_manifest()

def _greenhouse_switch_task():
    return native_manifest(metadata={"prediction_model_id":"greenhouse-horizon-targetwise-ridge@1",
                                    "evaluator_id":"greenhouse_multihorizon_time_forward@4"})

def _historical_experience_task(
    *,
    evaluator_id: str = "greenhouse_multihorizon_time_forward@1",
    evaluator_digest: str = "e" * 64,
    split_manifest_digest: str = "s" * 64,
    samples_per_update: int | None = None,
) -> TaskManifest:
    data = _greenhouse_switch_task().to_dict()
    runtime_catalog = dict(data["metadata"].get("runtime_component_catalog", {}))
    runtime_catalog["selected_evaluator_id"] = evaluator_id
    data["metadata"] = {
        **data["metadata"],
        "evaluator_id": evaluator_id,
        "evaluator_digest": evaluator_digest,
        "dataset_digest": "d" * 64,
        "split_manifest_digest": split_manifest_digest,
        "evaluation_partition": "training_feedback",
        "scientific_scope": "historical_replay_prediction_non_causal",
        "runtime_component_catalog": runtime_catalog,
    }
    if samples_per_update is not None:
        data["metadata"]["samples_per_update"] = samples_per_update
    return TaskManifest.from_dict(data)

def _historical_state(
    task: TaskManifest,
    run_id: str,
    analyses: tuple[GenerationAnalysis, ...],
    *,
    created_seq: int,
) -> SimpleNamespace:
    events = [SimpleNamespace(kind="RunCreated", seq=created_seq, payload={})]
    events.extend(
        SimpleNamespace(
            kind="GenerationAnalyzed",
            seq=created_seq + index + 1,
            payload={"analysis": analysis.to_dict()},
        )
        for index, analysis in enumerate(analyses)
    )
    return SimpleNamespace(
        task_manifest=task,
        run=SimpleNamespace(run_id=run_id),
        events=tuple(events),
        generation_analyses=analyses,
        research_iterations=(),
        proposals=(),
        algorithm_attempts=(),
    )

def _guardrail_analysis(
    run_id: str,
    generation: int,
    *,
    cohort_digest: str,
    cell_sample_count: int,
    skill_score: float = 0.12,
    constraint_violations: int = 0,
    value: float = 0.0,
    population_count: int = 1000,
    population_digest: str = "f" * 64,
    selected_count: int | None = None,
    window_offset: int | None = None,
) -> GenerationAnalysis:
    candidate_id = f"candidate:guardrail:{run_id}:{generation}"
    resolved_selected_count = selected_count or cell_sample_count
    resolved_window_offset = (
        generation * resolved_selected_count
        if window_offset is None
        else window_offset
    )
    return GenerationAnalysis(
        run_id=run_id,
        generation=generation,
        candidate_count=1,
        eligible_count=1 if constraint_violations == 0 and skill_score >= 0 else 0,
        outcome=(
            "promoted"
            if constraint_violations == 0 and skill_score >= 0
            else "no_eligible_candidate"
        ),
        selected_candidate_id=(
            candidate_id
            if constraint_violations == 0 and skill_score >= 0
            else None
        ),
        champion_candidate_id=candidate_id,
        ranking=(
            {
                "candidate_id": candidate_id,
                "score": skill_score,
                "constraint_violations": constraint_violations,
                "evaluation_cohort_digest": cohort_digest,
                "evaluation_cohort_window": {
                    "evaluation_cohort_digest": cohort_digest,
                    "feedback_update_cohort_digest": cohort_digest,
                    "schema_version": "ecologyrsi-dsh.feedback-update-cohort/1",
                    "selection_policy": (
                        "target_horizon_interleaved_rotating_window@1"
                    ),
                    "population_count": population_count,
                    "population_digest": population_digest,
                    "selected_count": resolved_selected_count,
                    "window_offset": resolved_window_offset,
                },
                "parameters": {
                    "co2_concentration_1h_residual_scale": value,
                    "residual_scale": 0.5,
                },
                "target_skill_scores": [
                    {
                        "target": "co2_concentration",
                        "horizon_hours": 1,
                        "skill_score": skill_score,
                        "n": cell_sample_count,
                        "observed": "must-not-enter-guardrails",
                    }
                ],
            },
        ),
    )

class ResearchIterationTests(unittest.TestCase):

    def test_historical_parameter_guardrails_require_replicated_cell_evidence(
        self,
    ) -> None:
        historical_task = _historical_experience_task(
            evaluator_id="greenhouse_multihorizon_time_forward@1",
            evaluator_digest="1" * 64,
        )
        current_state = _historical_state(
            historical_task,
            "run:guardrail-current",
            (),
            created_seq=100,
        )

        one_sample_state = _historical_state(
            historical_task,
            "run:guardrail-one-sample",
            (
                _guardrail_analysis(
                    "run:guardrail-one-sample",
                    0,
                    cohort_digest="a" * 64,
                    cell_sample_count=1,
                ),
            ),
            created_seq=10,
        )
        one_sample_experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=(one_sample_state,),
            history_cutoff_seq=100,
        )
        self.assertEqual(
            one_sample_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )
        self.assertEqual(len(one_sample_experience["historical_generations"]), 1)

        repeated_same_cohort_state = _historical_state(
            historical_task,
            "run:guardrail-same-cohort",
            tuple(
                _guardrail_analysis(
                    "run:guardrail-same-cohort",
                    generation,
                    cohort_digest="a" * 64,
                    cell_sample_count=20,
                    window_offset=0,
                )
                for generation in range(2)
            ),
            created_seq=20,
        )
        repeated_same_cohort_experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=(repeated_same_cohort_state,),
            history_cutoff_seq=100,
        )
        self.assertEqual(
            repeated_same_cohort_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )
        self.assertEqual(
            len(repeated_same_cohort_experience["historical_generations"]),
            2,
        )

        replicated_state = _historical_state(
            historical_task,
            "run:guardrail-replicated",
            (
                _guardrail_analysis(
                    "run:guardrail-replicated",
                    0,
                    cohort_digest="a" * 64,
                    cell_sample_count=20,
                    skill_score=0.12,
                ),
                _guardrail_analysis(
                    "run:guardrail-replicated",
                    1,
                    cohort_digest="b" * 64,
                    cell_sample_count=21,
                    skill_score=0.08,
                ),
            ),
            created_seq=30,
        )
        experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=(replicated_state,),
            history_cutoff_seq=100,
        )
        guardrails = experience["historical_parameter_guardrails"]
        self.assertEqual(
            guardrails["policy"],
            "preserve_verified_target_horizon_parameters_without_new_aggregate_evidence",
        )
        self.assertEqual(
            guardrails["evidence_requirements"],
            {
                "minimum_independent_cohort_count": 2,
                "minimum_cell_sample_count_per_cohort": 20,
                "minimum_total_cell_sample_count": 40,
                "same_cohort_generations_count_once": True,
                "requires_pairwise_non_overlapping_cohorts": True,
                "requires_nonnegative_skill_in_every_observation": True,
                "requires_zero_constraint_violations_in_every_observation": True,
            },
        )
        protected = guardrails["protected_parameter_evidence"]
        self.assertEqual(
            protected,
            [
                {
                    "parameter": "co2_concentration_1h_residual_scale",
                    "value": 0.0,
                    "target": "co2_concentration",
                    "horizon_hours": 1,
                    "skill_score": 0.08,
                    "source_run_id": "run:guardrail-replicated",
                    "source_generation": 1,
                    "independent_cohort_count": 2,
                    "total_cell_sample_count": 41,
                    "cohort_evidence": [
                        {
                            "evaluation_cohort_digest": "a" * 64,
                            "cell_sample_count": 20,
                            "minimum_skill_score": 0.12,
                            "schema_version": "ecologyrsi-dsh.feedback-update-cohort/1",
                            "selection_policy": "target_horizon_interleaved_rotating_window@1",
                            "feedback_update_cohort_digest": "a" * 64,
                            "population_count": 1000,
                            "population_digest": "f" * 64,
                            "selected_count": 20,
                            "window_offset": 0,
                            "source_run_id": "run:guardrail-replicated",
                            "source_generation": 0,
                        },
                        {
                            "evaluation_cohort_digest": "b" * 64,
                            "cell_sample_count": 21,
                            "minimum_skill_score": 0.08,
                            "schema_version": "ecologyrsi-dsh.feedback-update-cohort/1",
                            "selection_policy": "target_horizon_interleaved_rotating_window@1",
                            "feedback_update_cohort_digest": "b" * 64,
                            "population_count": 1000,
                            "population_digest": "f" * 64,
                            "selected_count": 21,
                            "window_offset": 21,
                            "source_run_id": "run:guardrail-replicated",
                            "source_generation": 1,
                        },
                    ],
                    "policy": "preserve_verified_target_horizon_parameters_without_new_aggregate_evidence",
                }
            ],
        )
        self.assertNotIn("must-not-enter-guardrails", canonical_json(experience))

        underpowered_cohort_state = _historical_state(
            historical_task,
            "run:guardrail-underpowered",
            (
                _guardrail_analysis(
                    "run:guardrail-underpowered",
                    0,
                    cohort_digest="a" * 64,
                    cell_sample_count=20,
                ),
                _guardrail_analysis(
                    "run:guardrail-underpowered",
                    1,
                    cohort_digest="b" * 64,
                    cell_sample_count=19,
                ),
            ),
            created_seq=40,
        )
        underpowered_experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=(underpowered_cohort_state,),
            history_cutoff_seq=100,
        )
        self.assertEqual(
            underpowered_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )

        conflicting_values_state = _historical_state(
            historical_task,
            "run:guardrail-conflicting-values",
            tuple(
                _guardrail_analysis(
                    "run:guardrail-conflicting-values",
                    generation,
                    cohort_digest=cohort * 64,
                    cell_sample_count=20,
                    value=value,
                )
                for generation, (cohort, value) in enumerate(
                    (("a", 0.0), ("b", 0.0), ("c", 0.5), ("d", 0.5))
                )
            ),
            created_seq=50,
        )
        conflicting_values_experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=(conflicting_values_state,),
            history_cutoff_seq=100,
        )
        self.assertEqual(
            conflicting_values_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )

        overlapping_windows_state = _historical_state(
            historical_task,
            "run:guardrail-overlapping-windows",
            (
                _guardrail_analysis(
                    "run:guardrail-overlapping-windows",
                    0,
                    cohort_digest="a" * 64,
                    cell_sample_count=20,
                    population_count=600,
                    selected_count=500,
                    window_offset=0,
                ),
                _guardrail_analysis(
                    "run:guardrail-overlapping-windows",
                    1,
                    cohort_digest="b" * 64,
                    cell_sample_count=20,
                    population_count=600,
                    selected_count=500,
                    window_offset=500,
                ),
            ),
            created_seq=60,
        )
        overlapping_windows_experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=(overlapping_windows_state,),
            history_cutoff_seq=100,
        )
        self.assertEqual(
            overlapping_windows_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )
        self.assertEqual(
            len(overlapping_windows_experience["historical_generations"]),
            2,
        )

        different_populations_state = _historical_state(
            historical_task,
            "run:guardrail-different-populations",
            (
                _guardrail_analysis(
                    "run:guardrail-different-populations",
                    0,
                    cohort_digest="a" * 64,
                    cell_sample_count=20,
                    population_digest="c" * 64,
                    window_offset=0,
                ),
                _guardrail_analysis(
                    "run:guardrail-different-populations",
                    1,
                    cohort_digest="b" * 64,
                    cell_sample_count=20,
                    population_digest="d" * 64,
                    window_offset=20,
                ),
            ),
            created_seq=70,
        )
        different_populations_experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=(different_populations_state,),
            history_cutoff_seq=100,
        )
        self.assertEqual(
            different_populations_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )

    def test_historical_parameter_guardrails_reject_conflicts_and_binding_drift(
        self,
    ) -> None:
        historical_task = _historical_experience_task(
            evaluator_id="greenhouse_multihorizon_time_forward@1",
            evaluator_digest="1" * 64,
        )
        current_state = _historical_state(
            historical_task,
            "run:guardrail-current",
            (),
            created_seq=100,
        )

        for label, second_skill, second_violations in (
            ("negative-skill", -0.01, 0),
            ("constraint-violation", 0.1, 1),
        ):
            with self.subTest(label=label):
                source_run_id = f"run:guardrail-{label}"
                historical_state = _historical_state(
                    historical_task,
                    source_run_id,
                    (
                        _guardrail_analysis(
                            source_run_id,
                            0,
                            cohort_digest="a" * 64,
                            cell_sample_count=20,
                        ),
                        _guardrail_analysis(
                            source_run_id,
                            1,
                            cohort_digest="b" * 64,
                            cell_sample_count=20,
                            skill_score=second_skill,
                            constraint_violations=second_violations,
                        ),
                    ),
                    created_seq=10,
                )
                experience = build_cross_generation_experience(
                    current_state,
                    0,
                    historical_states=(historical_state,),
                    history_cutoff_seq=100,
                )
                self.assertEqual(
                    experience["historical_parameter_guardrails"][
                        "protected_parameter_evidence"
                    ],
                    [],
                )

        replicated_state = _historical_state(
            historical_task,
            "run:guardrail-replicated",
            tuple(
                _guardrail_analysis(
                    "run:guardrail-replicated",
                    generation,
                    cohort_digest=cohort * 64,
                    cell_sample_count=20,
                )
                for generation, cohort in enumerate(("a", "b"))
            ),
            created_seq=20,
        )

        changed_evaluator_state = _historical_state(
            _historical_experience_task(
                evaluator_id="greenhouse_multihorizon_time_forward@2",
                evaluator_digest="2" * 64,
            ),
            "run:guardrail-changed-evaluator",
            (),
            created_seq=101,
        )
        changed_evaluator_experience = build_cross_generation_experience(
            changed_evaluator_state,
            0,
            historical_states=(replicated_state,),
            history_cutoff_seq=101,
        )
        self.assertEqual(
            changed_evaluator_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )
        self.assertEqual(
            len(changed_evaluator_experience["historical_generations"]),
            2,
        )

        changed_dataset_state = _historical_state(
            _historical_experience_task(
                evaluator_id="greenhouse_multihorizon_time_forward@1",
                evaluator_digest="1" * 64,
                split_manifest_digest="x" * 64,
            ),
            "run:guardrail-changed-dataset",
            (),
            created_seq=102,
        )
        changed_dataset_experience = build_cross_generation_experience(
            changed_dataset_state,
            0,
            historical_states=(replicated_state,),
            history_cutoff_seq=102,
        )
        self.assertEqual(
            changed_dataset_experience["historical_parameter_guardrails"][
                "protected_parameter_evidence"
            ],
            [],
        )
        self.assertEqual(
            changed_dataset_experience["historical_generations"],
            [],
        )

    def test_historical_experience_round_robins_runs_within_byte_budget(
        self,
    ) -> None:
        task = _historical_experience_task()
        historical_states = []
        source_ids = []
        for run_index in range(4):
            run_id = f"run:balanced-history:{run_index}"
            source_ids.append(run_id)
            analyses = tuple(
                GenerationAnalysis(
                    run_id=run_id,
                    generation=generation,
                    candidate_count=1,
                    eligible_count=0,
                    outcome="no_eligible_candidate",
                    ranking=(
                        {
                            "candidate_id": f"candidate:{run_index}:{generation}",
                            "score": float(generation) / 10,
                            "parameters": {
                                "history_steps": generation + 1,
                                "ridge_alpha": 0.1,
                            },
                        },
                    ),
                    target_weaknesses=tuple(
                        {
                            "target": f"target_{run_index}_{generation}_{item}",
                            "horizon_hours": item + 1,
                            "median_skill_score": -0.5,
                            "evidence_count": 2,
                        }
                        for item in range(8)
                    ),
                    algorithm_failures=tuple(
                        {
                            "phase": "debug",
                            "stage": f"stage-{item}",
                            "failure_code": f"failure-{run_index}-{generation}-{item}",
                            "attempt_count": 1,
                        }
                        for item in range(6)
                    ),
                    sample_failures=(
                        {
                            "failed": 1,
                            "failure_counts": {"gateway_timeout": 1},
                            "sample_execution_records": [
                                {"observed": f"raw-marker-{run_index}-{generation}"}
                            ],
                        },
                    ),
                )
                for generation in range(8)
            )
            historical_states.append(
                _historical_state(
                    task,
                    run_id,
                    analyses,
                    created_seq=run_index * 100 + 1,
                )
            )
        current_state = _historical_state(
            task,
            "run:balanced-history:current",
            (),
            created_seq=1000,
        )

        experience = build_cross_generation_experience(
            current_state,
            0,
            historical_states=tuple(historical_states),
            history_cutoff_seq=1000,
        )
        sources = experience["historical_generations"]
        source_counts = {
            run_id: sum(item["source_run_id"] == run_id for item in sources)
            for run_id in source_ids
        }
        historical_envelope = {
            "historical_generations": sources,
            "historical_parameter_guardrails": experience[
                "historical_parameter_guardrails"
            ],
            "historical_provenance": experience["historical_provenance"],
        }

        self.assertEqual(set(item["source_run_id"] for item in sources), set(source_ids))
        self.assertTrue(any(count >= 2 for count in source_counts.values()))
        self.assertLessEqual(
            len(canonical_json(historical_envelope).encode("utf-8")),
            8 * 1024,
        )
        self.assertLessEqual(
            len(canonical_json(experience).encode("utf-8")),
            CROSS_GENERATION_EXPERIENCE_MAX_BYTES,
        )
        self.assertNotIn("raw-marker", canonical_json(experience))

    def test_windowed_experience_compares_scores_only_within_same_cohort(
        self,
    ) -> None:
        task_data = _task(candidates_per_generation=4).to_dict()
        task_data["metadata"] = {
            **task_data["metadata"],
            "samples_per_update": 500,
        }
        task = TaskManifest.from_dict(task_data)
        synthesis = {
            "schema_version": "ecologyrsi-dsh.algorithm-synthesis/1",
            "pipeline_id": "greenhouse-exogenous-ridge@1",
            "evidence_refs": ["knowledge:windowed-experience"],
            "parameter_focus": ["ridge_alpha"],
            "rationale": "refine one bounded feedback cohort",
        }
        plan = {
            "prediction_model": {"id": "greenhouse-exogenous-ridge@1"},
            "algorithm_synthesis": synthesis,
        }
        first_cohort = "a" * 64
        next_cohort = "b" * 64
        analyses = tuple(
            GenerationAnalysis(
                run_id="run:windowed-experience",
                generation=generation,
                candidate_count=1,
                eligible_count=1,
                outcome="promoted",
                selected_candidate_id=f"candidate:windowed:{generation}",
                champion_candidate_id=f"candidate:windowed:{generation}",
                ranking=(
                    {
                        "candidate_id": f"candidate:windowed:{generation}",
                        "score": score,
                        "evaluation_cohort_digest": cohort_digest,
                        "parameters": {"ridge_alpha": 0.1 + generation / 10},
                    },
                ),
            )
            for generation, score, cohort_digest in (
                (0, 0.2, first_cohort),
                (1, 0.5, first_cohort),
                (2, 0.1, next_cohort),
            )
        )
        state = SimpleNamespace(
            task_manifest=task,
            generation_analyses=analyses,
            research_iterations=tuple(
                SimpleNamespace(
                    generation=generation,
                    plan=plan,

                )
                for generation in range(3)
            ),
            proposals=(),
            algorithm_attempts=(),
        )

        experience = build_cross_generation_experience(state, 3)
        initial, comparable, changed = experience["generations"]

        self.assertEqual(initial["score_comparison"], "no_previous_generation")
        self.assertFalse(initial["improved"])
        self.assertEqual(comparable["score_comparison"], "same_cohort")
        self.assertAlmostEqual(
            comparable["best_score_delta_vs_previous_generation"], 0.3
        )
        self.assertTrue(comparable["improved"])
        self.assertEqual(
            changed["score_comparison"], "different_cohort_not_compared"
        )
        self.assertNotIn("best_score_delta_vs_previous_generation", changed)
        self.assertFalse(changed["improved"])
        self.assertEqual(changed["best_score"], 0.1)
        self.assertEqual(changed["evaluation_cohort_digest"], next_cohort)

    def test_experience_compares_formal_candidates_not_unaccepted_batch_maxima(
        self,
    ) -> None:
        task = _task(candidates_per_generation=4)
        synthesis = {
            "schema_version": "ecologyrsi-dsh.algorithm-synthesis/1",
            "pipeline_id": "greenhouse-exogenous-ridge@1",
            "evidence_refs": ["knowledge:formal-comparison"],
            "parameter_focus": ["ridge_alpha"],
            "rationale": "compare only the formally selected candidate",
        }
        plan = {
            "prediction_model": {"id": "greenhouse-exogenous-ridge@1"},
            "algorithm_synthesis": synthesis,
        }

        def analysis(
            generation: int,
            selected_score: float,
            unaccepted_score: float,
        ) -> GenerationAnalysis:
            selected_id = f"candidate:formal:{generation}"
            return GenerationAnalysis(
                run_id="run:formal-comparison",
                generation=generation,
                candidate_count=2,
                eligible_count=1,
                outcome="promoted",
                selected_candidate_id=selected_id,
                champion_candidate_id=selected_id,
                ranking=(
                    {
                        "candidate_id": f"candidate:unaccepted:{generation}",
                        "score": unaccepted_score,
                        "evaluation_cohort_digest": "c" * 64,
                        "eligible": False,
                        "parameters": {"ridge_alpha": 9.0},
                    },
                    {
                        "candidate_id": selected_id,
                        "score": selected_score,
                        "evaluation_cohort_digest": "c" * 64,
                        "eligible": True,
                        "parameters": {"ridge_alpha": 0.1 + generation / 10},
                    },
                ),
            )

        analyses = (
            analysis(0, 0.6, 0.95),
            analysis(1, 0.7, 0.8),
            # Deliberately inconsistent historical evidence: even if an old
            # event says promoted, a negative formal delta is not improvement.
            analysis(2, 0.65, 0.99),
        )
        state = SimpleNamespace(
            task_manifest=task,
            generation_analyses=analyses,
            research_iterations=tuple(
                SimpleNamespace(
                    generation=generation,
                    plan=plan,

                )
                for generation in range(3)
            ),
            proposals=(),
            algorithm_attempts=(),
        )

        first, improved, regressed = build_cross_generation_experience(
            state, 3
        )["generations"]

        self.assertEqual(first["comparison_score"], 0.6)
        self.assertEqual(first["best_score"], 0.6)
        self.assertEqual(first["batch_highest_observed_score"], 0.95)
        self.assertFalse(first["improved"])

        self.assertAlmostEqual(
            improved["best_score_delta_vs_previous_generation"], 0.1
        )
        self.assertTrue(improved["improved"])
        self.assertEqual(improved["comparison_candidate_role"], "champion")
        self.assertEqual(improved["batch_highest_observed_score"], 0.8)
        self.assertAlmostEqual(
            regressed["best_score_delta_vs_previous_generation"], -0.05
        )
        self.assertFalse(regressed["improved"])

    def test_cross_generation_experience_enforces_capacity_limits(self) -> None:
        analyses = []
        for generation in range(30):
            analyses.append(
                GenerationAnalysis(
                    run_id="run:experience-capacity",
                    generation=generation,
                    candidate_count=1,
                    eligible_count=0,
                    outcome="no_eligible_candidate",
                    ranking=(
                        {
                            "candidate_id": f"candidate:capacity:{generation}",
                            "score": -float(generation + 1),
                            "parameters": {
                                "alpha": 0.2,
                                "window": generation + 1,
                                "water_threshold": 0.35,
                            },
                        },
                    ),
                    target_weaknesses=tuple(
                        {
                            "target": f"target-{generation}-{index}",
                            "horizon_hours": index + 1,
                            "median_skill_score": -1.0,
                            "evidence_count": 1,
                        }
                        for index in range(8)
                    ),
                    horizon_weaknesses=tuple(
                        {
                            "horizon_hours": generation * 10 + index,
                            "median_skill_score": -1.0,
                            "evidence_count": 1,
                        }
                        for index in range(8)
                    ),
                    algorithm_failures=tuple(
                        {
                            "phase": "debug",
                            "stage": f"stage-{index}",
                            "failure_code": f"failure-{generation}-{index}",
                            "attempt_count": 1,
                        }
                        for index in range(16)
                    ),
                    sample_failures=(
                        {
                            "failed": 32,
                            "failure_counts": {
                                f"sample-failure-{generation}-{index}": 2
                                for index in range(16)
                            },
                        },
                    ),
                )
            )
        state = SimpleNamespace(
            generation_analyses=tuple(analyses),
            research_iterations=(),
            proposals=(),
        )

        experience = build_cross_generation_experience(state, 30)

        self.assertEqual(experience["window"]["available_generation_count"], 30)
        self.assertEqual(experience["window"]["scanned_generation_count"], 24)
        self.assertTrue(experience["window"]["history_truncated"])
        self.assertLessEqual(len(experience["generations"]), 6)
        self.assertLessEqual(len(experience["active_unresolved"]), 16)
        self.assertLessEqual(len(experience["resolved_archived"]), 16)
        self.assertGreater(
            experience["capacity"]["omitted_generation_summaries"], 0
        )
        self.assertGreater(experience["capacity"]["omitted_active_issues"], 0)
        self.assertLessEqual(
            len(canonical_json(experience).encode("utf-8")),
            CROSS_GENERATION_EXPERIENCE_MAX_BYTES,
        )

    def test_cross_generation_experience_compacts_realistic_nine_cell_tool_rows(
        self,
    ) -> None:
        targets = ("air_temperature", "relative_humidity", "co2_concentration")
        horizons = (1, 6, 24)

        def ranking(generation: int) -> tuple[dict, ...]:
            return tuple(
                {
                    "candidate_id": f"candidate:{generation}:{candidate_index}",
                    "score": -0.1 * (generation + candidate_index + 1),
                    "classification": "scientific_gate_failed",
                    "parameters": {
                        "history_steps": 6 + generation,
                        "ridge_alpha": 0.1,
                    },
                    "tool_performance": [
                        {
                            "tool_id": f"greenhouse-ridge-{candidate_index}",
                            "version": "1",
                            "target": target,
                            "horizon_hours": horizon,
                            "selected": 1,
                            "completed": 1,
                            "failed": 0,
                            "rejected": 0,
                            "critic_accept": 1,
                            "critic_repair": 0,
                            "critic_failed": 0,
                            "final_accept": 1,
                            "recovered": 0,
                            "n": 1,
                            "mae": 12.3456789,
                            "rmse": 12.3456789,
                            "baseline_mae": 10.1234567,
                            "baseline_rmse": 10.1234567,
                        }
                        for target in targets
                        for horizon in horizons
                    ],
                }
                for candidate_index in range(2)
            )

        analyses = tuple(
            GenerationAnalysis(
                run_id="run:realistic-tool-capacity",
                generation=generation,
                candidate_count=2,
                eligible_count=0,
                outcome="no_eligible_candidate",
                ranking=ranking(generation),
                common_failures=("scientific_gate_failed",),
                insufficient_evidence=True,
            )
            for generation in range(2)
        )
        state = SimpleNamespace(
            generation_analyses=analyses,
            research_iterations=(),
            proposals=(),
            algorithm_attempts=(),
        )

        experience = build_cross_generation_experience(state, 2)

        self.assertEqual(experience["window"]["included_generations"], [0, 1])
        self.assertTrue(
            all(
                item["gate_result"]["insufficient_evidence"]
                for item in experience["generations"]
            )
        )
        self.assertGreater(
            experience["capacity"]["omitted_generation_details"], 0
        )
        self.assertLessEqual(
            len(canonical_json(experience).encode("utf-8")),
            CROSS_GENERATION_EXPERIENCE_MAX_BYTES,
        )
