from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ecologyrsi_dsh.api.generation_execution import (
    _evaluate_generation_controls,
    _generation_evidence_failure,
    complete_if_budget_exhausted,
)
from ecologyrsi_dsh.core.models import (
    Candidate,
    CandidateStatus,
    Evaluation,
    ModelArtifact,
    Proposal,
    RunStatus,
    TaskManifest,
    digest,
)
from ecologyrsi_dsh.data.registry import DatasetRegistry
from ecologyrsi_dsh.evaluators.registry import EvaluationBundle, EvaluatorRegistry
from ecologyrsi_dsh.knowledge.algorithms import AlgorithmSpec


def _spec(run_id: str, proposal_id: str, generation: int) -> AlgorithmSpec:
    return AlgorithmSpec(
        run_id=run_id,
        generation=generation,
        proposal_id=proposal_id,
        algorithm_id="registered-ridge",
        algorithm_version="1",
        adapter_id="greenhouse-exogenous-ridge@1",
        adapter_version="1",
        evaluator_id="greenhouse_multihorizon_time_forward@2",
        evaluator_version="2",
        strategy_id="research_compile_evolve@1",
        tool_ids=("greenhouse-exogenous-ridge@1",),
        parameters={"ridge_alpha": 1.0},
        visible_datasets=("dataset",),
    )


class GenerationControlExecutionTests(unittest.TestCase):
    @staticmethod
    def _strict_evidence_state(
        *,
        control_evidence: list[dict] | None = None,
        control_digest: str | None = None,
        sample_budget_class: str = "selection_eligible",
        attempted_examples: int = 1,
        attempted_origins: int = 1,
    ) -> object:
        run_id = "run:control-gate"
        parent = Candidate(
            candidate_id="candidate:control-parent",
            run_id=run_id,
            proposal_id="proposal:control-parent",
            generation=0,
            status=CandidateStatus.PROMOTED,
        )
        current = Candidate(
            candidate_id="candidate:control-current",
            run_id=run_id,
            proposal_id="proposal:control-current",
            generation=1,
            slot_index=0,
            status=CandidateStatus.EVALUATED,
        )
        strict_summary = {
            "strict_agent_contract": True,
            "strict_agent_chain_pass": True,
            "attempted_examples": attempted_examples,
            "attempted_origin_samples": attempted_origins,
            "prediction_cells_per_origin": 1,
            "complete_agent_chains": 1,
            "host_route_bypass_count": 0,
        }
        metrics = {
            "scientific_pass": True,
            "judge_status": "completed",
            "evaluation_index_digest": "c" * 64,
            "sample_execution": strict_summary,
        }
        if control_evidence is not None:
            metrics.update(
                {
                    "generation_control_policy": (
                        "same_cohort_search_parent_and_formal_elite@1"
                    ),
                    "generation_control_evaluations": control_evidence,
                    "generation_control_evidence_digest": (
                        control_digest
                        if control_digest is not None
                        else digest(control_evidence)
                    ),
                }
            )
        current_evaluation = Evaluation(
            evaluation_id="evaluation:control-current",
            run_id=run_id,
            candidate_id=current.candidate_id,
            score=0.2,
            passed=True,
            metrics=metrics,
            partition="training_feedback",
            evaluator_digest="evaluator@1",
            artifact_digest="a" * 64,
        )
        task = TaskManifest(
            task_id="strict-control-gate",
            objective="require complete same-cohort control evidence",
            domain_pack="greenhouse_environment@1",
            visible_datasets=("dataset",),
            budget={"max_candidates": 4, "max_generations": 2},
            metadata={
                "sample_agent_protocol": "dsh-strict-origin-bundle@3",
                "sample_budget_class": sample_budget_class,
                "samples_per_update": 1,
                "minimum_selection_samples_per_update": 1,
                "minimum_selection_origin_samples_per_update": 1,
                "prediction_cells_per_origin": 1,
            },
        )
        candidate_map = {
            parent.candidate_id: parent,
            current.candidate_id: current,
        }
        evaluations = {current.candidate_id: current_evaluation}
        batch = SimpleNamespace(
            generation=1,
            parent_candidate_id=parent.candidate_id,
        )

        class State:
            task_manifest = task
            run = SimpleNamespace(
                run_id=run_id,
                best_candidate_id=parent.candidate_id,
            )
            candidates = tuple(candidate_map.values())

            @staticmethod
            def batch_for(generation: int):
                return batch if generation == 1 else None

            @staticmethod
            def candidate(candidate_id: str):
                return candidate_map[candidate_id]

            @staticmethod
            def evaluation_for(candidate_id: str):
                return evaluations.get(candidate_id)

        return State()

    def test_complete_diagnostic_chain_can_feed_the_next_diagnostic_generation(
        self,
    ) -> None:
        state = self._strict_evidence_state(
            sample_budget_class="diagnostic_smoke",
        )

        self.assertIsNone(_generation_evidence_failure(state, 1))

    def test_incomplete_diagnostic_chain_cannot_advance(self) -> None:
        state = self._strict_evidence_state(
            sample_budget_class="diagnostic_smoke",
            attempted_examples=0,
            attempted_origins=0,
        )

        failure = _generation_evidence_failure(state, 1)

        self.assertIsNotNone(failure)
        self.assertIn("generation_strict_agent_evidence_incomplete", failure)

    def test_generation_gate_rejects_missing_control_evidence(self) -> None:
        state = self._strict_evidence_state()

        failure = _generation_evidence_failure(state, 1)

        self.assertIsNotNone(failure)
        self.assertIn("generation_control_evidence_invalid", failure)

    def test_generation_gate_rejects_tampered_control_evidence(self) -> None:
        state_without_controls = self._strict_evidence_state()
        current_evaluation = state_without_controls.evaluation_for(
            "candidate:control-current"
        )
        self.assertIsNotNone(current_evaluation)
        control_data = current_evaluation.to_dict()
        control_data.update(
            {
                "evaluation_id": "evaluation:control-parent-replay",
                "candidate_id": "candidate:control-parent",
            }
        )
        controls = [
            {
                "schema_version": (
                    "ecologyrsi-dsh.generation-control-evaluation/1"
                ),
                "generation": 1,
                "comparison_role": "search_parent_and_formal_elite",
                "candidate_id": "candidate:control-parent",
                "evaluation": control_data,
            }
        ]
        state = self._strict_evidence_state(
            control_evidence=controls,
            control_digest="0" * 64,
        )

        failure = _generation_evidence_failure(state, 1)

        self.assertIsNotNone(failure)
        self.assertIn("generation_control_evidence_invalid", failure)

    def test_parent_and_elite_are_replayed_once_when_they_are_the_same_candidate(
        self,
    ) -> None:
        run_id = "run:controls"
        parent_proposal = Proposal(
            proposal_id="proposal:parent",
            run_id=run_id,
            generation=0,
            title="parent",
            changes={"ridge_alpha": 1.0},
        )
        current_proposal = Proposal(
            proposal_id="proposal:current",
            run_id=run_id,
            generation=1,
            title="current",
            changes={"ridge_alpha": 1.05},
            parent_candidate_id="candidate:parent",
        )
        parent = Candidate(
            candidate_id="candidate:parent",
            run_id=run_id,
            proposal_id=parent_proposal.proposal_id,
            generation=0,
            status=CandidateStatus.PROMOTED,
        )
        current = Candidate(
            candidate_id="candidate:current",
            run_id=run_id,
            proposal_id=current_proposal.proposal_id,
            generation=1,
            slot_index=0,
        )
        compiled = {
            parent.candidate_id: _spec(run_id, parent.proposal_id, 0).to_dict(),
            current.candidate_id: _spec(run_id, current.proposal_id, 1).to_dict(),
        }
        proposals = {
            parent.proposal_id: parent_proposal,
            current.proposal_id: current_proposal,
        }
        candidates = {
            parent.candidate_id: parent,
            current.candidate_id: current,
        }
        task = TaskManifest(
            task_id="strict-controls",
            objective="compare on one cohort",
            domain_pack="greenhouse_environment@1",
            visible_datasets=("dataset",),
            budget={"max_candidates": 4, "max_generations": 2},
            metadata={
                "sample_agent_protocol": "dsh-strict-origin-bundle@3",
                "sample_budget_class": "selection_eligible",
            },
        )
        batch = SimpleNamespace(
            generation=1,
            parent_candidate_id=parent.candidate_id,
        )

        class State:
            task_manifest = task
            run = SimpleNamespace(best_candidate_id=parent.candidate_id)

            @staticmethod
            def batch_for(generation: int):
                return batch if generation == 1 else None

            @staticmethod
            def candidate(candidate_id: str):
                return candidates[candidate_id]

            @staticmethod
            def proposal(proposal_id: str):
                return proposals[proposal_id]

            @staticmethod
            def compiled_algorithm_for(candidate_id: str):
                return compiled[candidate_id]

        registry = EvaluatorRegistry(DatasetRegistry(), object())
        endpoint = SimpleNamespace(server=SimpleNamespace(evaluators=registry))
        seen = []

        def evaluate(_task, candidate, proposal, **kwargs):
            seen.append((candidate, proposal, kwargs["algorithm_spec"]))
            artifact = ModelArtifact(
                artifact_id="artifact:control",
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                model_id="registered-ridge",
                dataset_digest="d" * 64,
                training_partition="training_fit",
                training_rows=10,
                parameters=proposal.changes,
            )
            evaluation = Evaluation(
                evaluation_id="evaluation:control",
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                score=0.2,
                passed=True,
                metrics={
                    "scientific_pass": True,
                    "evaluation_index_digest": "e" * 64,
                    "sample_execution": {
                        "strict_agent_contract": True,
                        "strict_agent_chain_pass": True,
                        "host_route_bypass_count": 0,
                    },
                },
                partition="training_feedback",
                evaluator_digest="evaluator@1",
                artifact_digest=artifact.digest,
            )
            return EvaluationBundle(artifact, evaluation)

        with patch.object(registry, "evaluate_scientific", side_effect=evaluate):
            controls = _evaluate_generation_controls(
                endpoint,
                State(),
                current,
                on_model_usage=lambda _receipts: None,
                on_sample_control=lambda: "running",
            )

        self.assertEqual(len(seen), 1)
        replay_candidate, replay_proposal, replay_spec = seen[0]
        self.assertEqual(replay_candidate.generation, 1)
        self.assertEqual(replay_proposal.generation, 1)
        self.assertEqual(replay_spec.generation, 1)
        self.assertEqual(replay_spec.parameters, {"ridge_alpha": 1.0})
        self.assertEqual(
            controls[0]["comparison_role"],
            "search_parent_and_formal_elite",
        )
        self.assertNotIn(
            "prediction_preview", controls[0]["evaluation"]["metrics"]
        )

    def test_diagnostic_budget_completion_has_explicit_non_promotion_outcome(
        self,
    ) -> None:
        run_id = "run:diagnostic-complete"
        task = TaskManifest(
            task_id="diagnostic-complete",
            objective="exercise the complete chain without selecting a champion",
            domain_pack="greenhouse_environment@1",
            visible_datasets=("dataset",),
            budget={"max_candidates": 2, "max_generations": 2},
            metadata={
                "sample_agent_protocol": "dsh-strict-origin-bundle@3",
                "sample_budget_class": "diagnostic_smoke",
            },
        )
        running = SimpleNamespace(
            run=SimpleNamespace(
                run_id=run_id,
                status=RunStatus.RUNNING,
                generation=2,
                best_candidate_id=None,
            ),
            task_manifest=task,
            candidates=(),
            events=(),
        )
        completed = SimpleNamespace(
            run=SimpleNamespace(
                run_id=run_id,
                status=RunStatus.COMPLETED,
                generation=2,
                best_candidate_id=None,
            ),
            task_manifest=task,
            candidates=(),
            events=(),
        )

        class Director:
            completion: dict | None = None

            def state(self, requested_run_id: str):
                self.assert_run_id(requested_run_id)
                return completed if self.completion is not None else running

            @staticmethod
            def assert_run_id(requested_run_id: str) -> None:
                if requested_run_id != run_id:
                    raise AssertionError(requested_run_id)

            def complete_run(self, requested_run_id: str, **kwargs) -> None:
                self.assert_run_id(requested_run_id)
                self.completion = kwargs

        director = Director()
        endpoint = SimpleNamespace(server=SimpleNamespace(director=director))

        result = complete_if_budget_exhausted(endpoint, run_id, running)

        self.assertIs(result, completed)
        self.assertEqual(
            director.completion,
            {
                "termination_reason": "diagnostic_smoke_completed_no_promotion",
                "outcome": "diagnostic_smoke_completed",
            },
        )


if __name__ == "__main__":
    unittest.main()
