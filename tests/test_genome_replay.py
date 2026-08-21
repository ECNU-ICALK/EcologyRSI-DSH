from __future__ import annotations

import json
import unittest

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.models import (
    Evaluation,
    ModelArtifact,
    Promotion,
    PromotionDecision,
    Proposal,
    TaskManifest,
    canonical_json,
)
from ecologyrsi_dsh.evolution.genome import (
    GenomeMutationContextV1,
    apply_genome_mutation,
)
from ecologyrsi_dsh.evolution.analysis import GenerationAnalysis
from ecologyrsi_dsh.evolution.batches import start_generation_batch
from ecologyrsi_dsh.evolution.strategies import FakeDSHAdapter
from ecologyrsi_dsh.knowledge.algorithms import AlgorithmAttempt
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry


def _new_task() -> TaskManifest:
    return TaskManifest(
        task_id="genome-replay",
        objective="predict greenhouse climate",
        domain_pack="greenhouse_environment@1",
        visible_datasets=("agc_cucumber_2018",),
        budget={"max_candidates": 4, "candidates_per_generation": 2},
        metadata={
            "execution_protocol": "dsh_native_plugin_evolution@1",
            "seed_genome_template_id": "greenhouse-default@1",
            "prediction_model_id": "greenhouse-horizon-targetwise-ridge@1",
            "evaluator_id": "greenhouse_multihorizon_time_forward@2",
            "strategy_id": "autonomous_model@1",
            "dataset_digest": "2" * 64,
            "dataset_snapshot_set_digest": "2" * 64,
            "split_manifest_digest": "3" * 64,
            "data_protocol_digest": "4" * 64,
            "stage_policy_digest": "5" * 64,
            "evaluator_digest": "6" * 64,
            "fitness_profile_digest": "7" * 64,
            "security_kernel_digest": "8" * 64,
            "selection_reviewer_program_digest": "9" * 64,
            "required_capability_digest": "a" * 64,
            "resolved_policy_route_digest": "b" * 64,
            "resolved_review_route_digest": "c" * 64,
            "resolved_policy_route_config_digest": "b" * 64,
            "resolved_review_route_config_digest": "c" * 64,
            "preset_content_digest": "d" * 64,
            "standing_tool_surface_digest": "e" * 64,
            "evaluation_cohort_digest": "f" * 64,
        },
    )


def _legacy_task() -> TaskManifest:
    return TaskManifest(
        task_id="legacy-replay",
        objective="predict water",
        domain_pack="crop-soil-water@toy",
        visible_datasets=("generated-toy-series@1",),
        budget={"max_candidates": 1},
    )


class GenomeReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = EventLedger()
        self.director = EvolutionDirector(self.ledger, FakeDSHAdapter())

    def tearDown(self) -> None:
        self.ledger.close()

    def _start_new(self, run_id: str = "run:genome-replay") -> str:
        self.director.create_run(_new_task(), run_id=run_id)
        self.director.start_run(run_id)
        return run_id

    def _proposal(self, run_id: str, *, slot_index: int = 0) -> Proposal:
        state = self.director.state(run_id)
        parent = state.materialized_seed_genome()
        context = GenomeMutationContextV1(
            run_id=run_id,
            generation=0,
            slot_index=slot_index,
            slot_seed=100 + slot_index,
            parent_candidate_id=None,
            parent_genome_digest=parent.genome_digest,
            generation_batch_digest="1" * 64,
            research_iteration_digest="2" * 64,
            knowledge_snapshot_digest="3" * 64,
            mutation_budget_digest="4" * 64,
            mutation_operator_id="bounded-single-parent-mutation@1",
        )
        child = apply_genome_mutation(
            parent,
            {"schema_version": "ecologyrsi-dsh.genome-mutation/1", "operations": []},
            context,
            current_program_registry(),
        )
        return Proposal(
            proposal_id=f"proposal:genome:{slot_index}",
            run_id=run_id,
            generation=0,
            title=f"bounded child {slot_index}",
            changes=dict(child.scientific_program["parameter_overrides"]),
            metadata={
                "execution_protocol": "dsh_native_plugin_evolution@1",
                "evolution_genome_canonical_json": canonical_json(child.to_dict()),
                "genome_digest": child.genome_digest,
                "behavior_digest": child.behavior_digest,
            },
        )

    def _spawn(self, run_id: str, *, slot_index: int = 0):
        proposal = self._proposal(run_id, slot_index=slot_index)
        self.director.submit_proposal(proposal)
        return self.director.spawn_candidate(
            run_id,
            proposal,
            candidate_id=f"candidate:genome:{slot_index}",
            slot_index=slot_index,
        )

    def _record_artifact_and_evaluation(self, run_id: str, candidate_id: str):
        artifact = ModelArtifact(
            artifact_id=f"artifact:{candidate_id}",
            run_id=run_id,
            candidate_id=candidate_id,
            model_id="greenhouse-horizon-targetwise-ridge@1",
            dataset_digest="2" * 64,
            training_partition="training_fit",
            training_rows=10,
        )
        self.director.record_artifact(artifact)
        evaluation = Evaluation(
            evaluation_id=f"evaluation:{candidate_id}",
            run_id=run_id,
            candidate_id=candidate_id,
            score=0.2,
            passed=False,
            evaluator_digest="6" * 64,
            artifact_digest=artifact.digest,
        )
        self.director.record_evaluation(evaluation)
        return artifact, evaluation

    def test_new_protocol_rejects_all_none_genome_chain(self) -> None:
        run_id = self._start_new()
        proposal = Proposal(
            proposal_id="proposal:none-chain",
            run_id=run_id,
            generation=0,
            title="invalid empty chain",
            changes={},
            metadata={
                "execution_protocol": "dsh_native_plugin_evolution@1",
                "evolution_genome_canonical_json": None,
                "genome_digest": None,
                "behavior_digest": None,
            },
        )
        with self.assertRaisesRegex((TypeError, ValueError), "genome|canonical"):
            self.director.submit_proposal(proposal)

    def test_legacy_protocol_accepts_missing_genome_fields(self) -> None:
        self.director.start_evolution(_legacy_task(), run_id="run:legacy")
        proposal = Proposal(
            proposal_id="proposal:legacy",
            run_id="run:legacy",
            generation=0,
            title="historical proposal",
            changes={"alpha": 0.5},
        )
        self.director.submit_proposal(proposal)
        candidate = self.director.spawn_candidate("run:legacy", proposal)
        self.assertIsNone(
            self.director.state("run:legacy").candidate_identity_binding(
                candidate.candidate_id
            )
        )

    def test_materialized_seed_canonical_json_precedes_first_generation_batch(
        self,
    ) -> None:
        run_id = self._start_new()
        state = self.director.state(run_id)
        seed = state.materialized_seed_genome()
        kinds = [event.kind for event in state.events]

        self.assertEqual(kinds[:3], ["RunCreated", "RunSeedGenomeMaterialized", "RunStarted"])
        self.assertEqual(seed.genome_digest, state.parent_genome_for_generation(0).genome_digest)
        self.assertNotIn("GenerationBatchStarted", kinds[:2])

    def test_restart_uses_persisted_seed_after_catalog_change(self) -> None:
        run_id = "run:restart-seed"
        self.director.create_run(_new_task(), run_id=run_id)
        before = self.director.state(run_id).materialized_seed_genome().to_dict()
        changed_registry = current_program_registry().with_program_override(
            "predictors",
            "greenhouse-horizon-targetwise-ridge@1",
            {"version": "future-registry/999"},
        )
        self.assertNotEqual(
            changed_registry.catalog_digest,
            current_program_registry().catalog_digest,
        )

        restarted = EvolutionDirector(self.ledger, FakeDSHAdapter())
        self.assertEqual(
            restarted.state(run_id).materialized_seed_genome().to_dict(), before
        )

    def test_crash_between_run_created_and_seed_materialized_recovers_identically(
        self,
    ) -> None:
        run_id = "run:partial-initialization"
        run, payload = self.director.prepare_run_creation(_new_task(), run_id=run_id)
        self.ledger.append(run_id, "RunCreated", payload, event_id=f"{run_id}:created")
        with self.assertRaisesRegex(RuntimeError, "seed|initial"):
            self.director.start_run(run_id)

        recovered = self.director.recover_run_initialization(run_id)
        state = self.director.state(run_id)
        expected = json.loads(payload["genome_initialization"]["expected_seed_canonical_json"])
        self.assertEqual(recovered.to_dict(), expected)
        self.assertEqual(state.materialized_seed_genome().to_dict(), expected)

    def test_partial_initialization_cannot_start_generation(self) -> None:
        run_id = "run:partial-batch"
        _run, payload = self.director.prepare_run_creation(_new_task(), run_id=run_id)
        self.ledger.append(run_id, "RunCreated", payload)

        with self.assertRaisesRegex(RuntimeError, "running|seed|initial"):
            start_generation_batch(self.director, run_id)

    def test_generation_batch_freezes_complete_parent_genome_and_stage_context(self) -> None:
        run_id = self._start_new("run:frozen-parent")
        batch = start_generation_batch(self.director, run_id)
        state = self.director.state(run_id)
        parent = state.materialized_seed_genome()

        self.assertEqual(batch.parent_genome_digest, parent.genome_digest)
        self.assertEqual(
            json.loads(batch.parent_genome_canonical_json), parent.to_dict()
        )
        self.assertEqual(
            state.parent_genome_for_generation(0).genome_digest,
            parent.genome_digest,
        )
        self.assertEqual(
            batch.stage_context_digests["registry_catalog_digest"],
            current_program_registry().catalog_digest,
        )

    def test_next_generation_research_resolves_search_parent_before_batch_exists(
        self,
    ) -> None:
        run_id = self._start_new("run:next-research-parent")
        start_generation_batch(self.director, run_id)
        candidates = [
            self._spawn(run_id, slot_index=slot_index)
            for slot_index in range(2)
        ]
        for candidate in candidates:
            artifact, evaluation = self._record_artifact_and_evaluation(
                run_id, candidate.candidate_id
            )
            self.director.decide_promotion(
                Promotion(
                    promotion_id=f"promotion:{candidate.candidate_id}",
                    run_id=run_id,
                    candidate_id=candidate.candidate_id,
                    decision=PromotionDecision.REJECTED,
                    reason="test search parent",
                )
            )
        analysis = GenerationAnalysis(
            run_id=run_id,
            generation=0,
            candidate_count=2,
            eligible_count=0,
            outcome="no_eligible_candidate",
            search_parent_candidate_id=candidates[0].candidate_id,
        )
        self.ledger.append(
            run_id,
            "GenerationAnalyzed",
            {"analysis": analysis.to_dict()},
        )

        self.director.advance_generation(run_id)
        state = self.director.state(run_id)
        self.assertIsNone(state.batch_for(1))
        parent_candidate_id = analysis.search_parent_candidate_id
        self.assertIsNotNone(parent_candidate_id)
        self.assertEqual(
            state.parent_genome_for_generation(1).genome_digest,
            state.persisted_genome_for(parent_candidate_id).genome_digest,
        )

    def test_proposal_does_not_require_candidate_dependent_instance_digest(self) -> None:
        run_id = self._start_new()
        proposal = self._proposal(run_id)
        self.assertNotIn("phenotype_instance_digest", proposal.metadata)
        self.director.submit_proposal(proposal)
        candidate = self.director.spawn_candidate(run_id, proposal, slot_index=0)
        binding = self.director.state(run_id).candidate_identity_binding(
            candidate.candidate_id
        )
        self.assertTrue(binding["phenotype_instance_digest"])
        self.assertTrue(binding["compiled_behavior_digest"])

    def test_first_algorithm_attempt_requires_candidate_identity_binding(self) -> None:
        run_id = self._start_new("run:algorithm-binding")
        candidate = self._spawn(run_id)
        attempt = AlgorithmAttempt(
            run_id=run_id,
            generation=0,
            proposal_id=candidate.proposal_id,
            candidate_id=candidate.candidate_id,
            phase="compile",
            attempt=1,
            status="failed",
            failure_code="bounded_compile_failure",
            public_error="compile failed",
        )
        self.director.record_algorithm_attempt(attempt)
        event = next(
            event
            for event in self.director.state(run_id).events
            if event.kind == "AlgorithmAttemptRecorded"
        )
        self.assertEqual(
            event.payload["identity_binding"],
            self.director.state(run_id).candidate_identity_binding(
                candidate.candidate_id
            ),
        )

        tampered = dict(event.payload)
        tampered.pop("identity_binding")
        other_run = self._start_new("run:algorithm-binding-tampered")
        other_candidate = self._spawn(other_run)
        tampered["algorithm_attempt"] = {
            **attempt.to_dict(),
            "run_id": other_run,
            "proposal_id": other_candidate.proposal_id,
            "candidate_id": other_candidate.candidate_id,
        }
        self.ledger.append(other_run, "AlgorithmAttemptRecorded", tampered)
        with self.assertRaisesRegex(ValueError, "identity binding"):
            self.director.state(other_run)

    def test_behavior_identical_siblings_dedupe_by_compiled_behavior_and_cohort(
        self,
    ) -> None:
        run_id = self._start_new()
        first = self._spawn(run_id, slot_index=0)
        second = self._spawn(run_id, slot_index=1)
        state = self.director.state(run_id)
        self.assertNotEqual(
            state.persisted_genome_for(first.candidate_id).genome_digest,
            state.persisted_genome_for(second.candidate_id).genome_digest,
        )
        self.assertEqual(
            state.candidate_duplicate_signature(first.candidate_id),
            state.candidate_duplicate_signature(second.candidate_id),
        )

    def test_promotion_rejects_matching_genome_but_wrong_phenotype_instance(self) -> None:
        run_id = self._start_new()
        candidate = self._spawn(run_id)
        artifact, evaluation = self._record_artifact_and_evaluation(
            run_id, candidate.candidate_id
        )
        binding = dict(
            self.director.state(run_id).candidate_identity_binding(candidate.candidate_id)
        )
        binding["phenotype_instance_digest"] = "0" * 64
        promotion = Promotion(
            promotion_id="promotion:wrong-instance",
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            decision=PromotionDecision.REJECTED,
            reason="tampered instance binding",
        )
        self.ledger.append(
            run_id,
            "PromotionDecided",
            {
                "promotion": promotion.to_dict(),
                "identity_binding": binding,
                "evaluation_id": evaluation.evaluation_id,
                "evaluation_digest": "1" * 64,
                "artifact_digest": artifact.digest,
            },
        )
        with self.assertRaisesRegex(ValueError, "phenotype|identity binding"):
            self.director.state(run_id)

    def test_projected_legacy_genome_cannot_be_promoted_or_resumed_without_migration_seed(
        self,
    ) -> None:
        run_id = self._start_new()
        proposal = self._proposal(run_id)
        metadata = dict(proposal.metadata)
        projected = {
            "schema_version": "ecologyrsi-dsh.legacy-genome-projection/1",
            "projected": True,
            "inheritable": False,
        }
        metadata["evolution_genome_canonical_json"] = canonical_json(projected)
        invalid = Proposal(
            **{
                **proposal.to_dict(),
                "proposal_id": "proposal:projected-legacy",
                "metadata": metadata,
            }
        )
        with self.assertRaisesRegex(ValueError, "genome|schema|migration"):
            self.director.submit_proposal(invalid)


if __name__ == "__main__":
    unittest.main()
