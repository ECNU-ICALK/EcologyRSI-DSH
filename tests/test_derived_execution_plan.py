from __future__ import annotations

import unittest

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import Proposal, TaskManifest
from ecologyrsi_dsh.evolution.analysis import GenerationAnalysis
from ecologyrsi_dsh.evolution.batches import start_generation_batch
from ecologyrsi_dsh.evolution.execution_plan import (
    DerivedExecutionPlan,
    derive_execution_plan,
)
from ecologyrsi_dsh.evolution.strategies import StrategyRouterDSHAdapter
from ecologyrsi_dsh.knowledge.algorithms import (
    AlgorithmCompileError,
    AlgorithmSpec,
    compile_algorithm_spec,
)


def _task() -> TaskManifest:
    return TaskManifest(
        task_id="derived-execution-plan",
        objective="predict soil water",
        domain_pack="crop-soil-water@toy",
        visible_datasets=("generated-toy-series@1",),
        budget={
            "max_candidates": 2,
            "max_generations": 2,
            "candidates_per_generation": 1,
        },
        metadata={
            "domain": "toy",
            "strategy_id": "adaptive_local@1",
            "prediction_model_id": "toy-rolling-water@1",
            "evaluator_id": "toy_time_forward@1",
        },
    )


def _analysis() -> GenerationAnalysis:
    return GenerationAnalysis(
        run_id="run:derived-plan",
        generation=0,
        candidate_count=1,
        eligible_count=0,
        outcome="no_eligible_candidate",
        sample_failures=(
            {
                "candidate_id": "candidate:aggregate-only",
                "attempted": 10,
                "succeeded": 7,
                "failed": 3,
                "coverage": 0.7,
                "minimum_coverage": 0.8,
                "coverage_pass": False,
                "failure_counts": {
                    "tool_timeout": 2,
                    "constraint_rejected": 1,
                },
                # Unknown fields demonstrate that the derivation is an
                # allowlisted aggregate projection, not a raw-record copy.
                "sample_execution_records": [
                    {"sample_id": "must-not-propagate", "observed": 99.0}
                ],
            },
        ),
    )


class DerivedExecutionPlanTests(unittest.TestCase):
    def test_pure_derivation_routes_failures_without_copying_samples(self) -> None:
        analysis = _analysis()
        first = derive_execution_plan(analysis)
        second = derive_execution_plan(analysis)

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.source_analysis_digest, analysis.analysis_digest)
        self.assertEqual(first.sample_max_attempts, 6)
        self.assertEqual(first.plan_max_attempts, 5)
        self.assertEqual(first.retry_backoff_seconds, 2.0)
        self.assertEqual(
            first.repair_sequence[0], "bounded-persistence-fallback"
        )
        self.assertEqual(
            [item["failure_class"] for item in first.failure_profile],
            ["timeout", "constraint_rejected"],
        )
        serialized = str(first.to_dict())
        self.assertNotIn("must-not-propagate", serialized)
        self.assertNotIn("observed", serialized)

    def test_plan_digest_rejects_tampering(self) -> None:
        data = derive_execution_plan(_analysis()).to_dict()
        data["sample_max_attempts"] = 7
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            DerivedExecutionPlan.from_dict(data)

    def test_recovered_failures_and_repair_success_keep_the_registered_route(self) -> None:
        analysis = GenerationAnalysis(
            run_id="run:recovered-repairs",
            generation=0,
            candidate_count=1,
            eligible_count=0,
            outcome="no_eligible_candidate",
            sample_failures=(
                {
                    "attempted": 10,
                    "failed": 0,
                    "coverage_pass": True,
                    "repair_count": 4,
                    "failure_counts": {
                        "constraint_rejected": 4,
                        "timeout": 2,
                    },
                    "recovered_by_failure_class": {
                        "constraint_rejected": 4,
                        "timeout": 2,
                    },
                    "repair_tool_outcomes": {
                        "bounded-projection-repair": {"completed": 4}
                    },
                },
            ),
        )

        plan = derive_execution_plan(analysis)

        self.assertEqual(plan.sample_max_attempts, 3)
        self.assertEqual(plan.plan_max_attempts, 3)
        self.assertEqual(plan.retry_backoff_seconds, 0.0)
        self.assertEqual(plan.repair_sequence[0], "bounded-projection-repair")

    def test_director_freezes_host_plan_in_proposal_and_algorithm_spec(self) -> None:
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, StrategyRouterDSHAdapter())
            director.start_evolution(_task(), run_id="run:initial-plan")
            batch = start_generation_batch(director, "run:initial-plan")
            proposal = director.request_proposal(
                "run:initial-plan", generation_batch=batch, slot_index=0
            )
            proposal_plan = DerivedExecutionPlan.from_dict(
                proposal.metadata["derived_execution_plan"]
            )
            self.assertIsNone(proposal_plan.source_generation)

            state = director.state("run:initial-plan")
            spec = compile_algorithm_spec(
                state.task_manifest,
                proposal,
                state.knowledge_for(0),
            )
            replayed = AlgorithmSpec.from_dict(spec.to_dict())
            self.assertEqual(
                replayed.derived_execution_plan,
                proposal_plan.to_dict(),
            )
            self.assertEqual(replayed.spec_digest, spec.spec_digest)

    def test_compiler_rejects_plan_from_wrong_generation(self) -> None:
        proposal = Proposal(
            proposal_id="proposal:wrong-plan-scope",
            run_id="run:wrong-plan-scope",
            generation=1,
            title="wrong plan scope",
            changes={"alpha": 0.4, "window": 5, "water_threshold": 0.4},
            metadata={"derived_execution_plan": derive_execution_plan(None).to_dict()},
        )
        with self.assertRaises(AlgorithmCompileError) as caught:
            compile_algorithm_spec(_task(), proposal, None)
        self.assertEqual(
            caught.exception.code,
            "derived_execution_plan_scope_mismatch",
        )


if __name__ == "__main__":
    unittest.main()
