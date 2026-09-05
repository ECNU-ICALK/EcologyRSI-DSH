from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.api.dsh_tools import DshToolService
from ecologyrsi_dsh.api.projection import _candidate_projection
from ecologyrsi_dsh.core.ledger import EventLedger
from ecologyrsi_dsh.core.state import RunState
from ecologyrsi_dsh.core.models import (
    CandidateStatus,
    Evaluation,
    ModelArtifact,
    Promotion,
    PromotionDecision,
    Proposal,
    TaskManifest,
    canonical_json,
    digest,
)
from ecologyrsi_dsh.evolution.genome import (
    GenomeMutationContextV1,
    apply_genome_mutation,
)
from ecologyrsi_dsh.evolution.workflow_ir import resolve_candidate_agent_profile
from ecologyrsi_dsh.evolution.analysis import (
    GenerationAnalysis,
    build_cross_generation_experience,
)
from ecologyrsi_dsh.core.trajectory import (
    EvaluationPhase,
    EvaluationScope,
    GenerationComparison,
    HoldoutArm,
    HoldoutEvaluation,
)
from ecologyrsi_dsh.evolution.batches import (
    _generation_parent_candidate_id,
    _generation_parent_genome,
    _search_parent_from_analysis,
    start_generation_batch,
)
from ecologyrsi_dsh.evolution.schedule import OPTIMIZATION_PROTOCOL
from ecologyrsi_dsh.evolution.strategies import (
    FakeDSHAdapter,
    _native_evolution_reflection_from_experience,
)
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

    def test_adaptive_next_epoch_uses_frozen_effective_final_revision(self) -> None:
        run_id = self._start_new("run:effective-final-parent")
        seed = self.director.state(run_id).materialized_seed_genome()
        final = apply_genome_mutation(
            seed,
            {
                "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                "operations": [
                    {
                        "op": "set_bounded_parameter",
                        "name": "ridge_alpha",
                        "value": 0.35,
                    }
                ],
            },
            GenomeMutationContextV1(
                run_id=run_id,
                generation=0,
                slot_index=2,
                slot_seed=102,
                parent_candidate_id=None,
                parent_genome_digest=seed.genome_digest,
                generation_batch_digest="1" * 64,
                research_iteration_digest="2" * 64,
                knowledge_snapshot_digest="3" * 64,
                mutation_budget_digest="4" * 64,
                mutation_operator_id="bounded-single-parent-mutation@1",
            ),
            current_program_registry(),
        )
        incumbent_id = "candidate:screened-incumbent"
        failed_finalist_id = "candidate:failed-finalist"
        revision = SimpleNamespace(
            revision_id="revision:incumbent:final",
            candidate_id=incumbent_id,
            genome=final.to_dict(),
            genome_digest=final.genome_digest,
        )
        candidates = {
            incumbent_id: SimpleNamespace(status=CandidateStatus.SCREENED_OUT),
            failed_finalist_id: SimpleNamespace(status=CandidateStatus.REJECTED),
        }
        analysis = SimpleNamespace(
            search_parent_candidate_id=incumbent_id,
            ranking=(
                {
                    "candidate_id": incumbent_id,
                    "score": 0.8,
                    "constraint_violations": 0,
                },
                {
                    "candidate_id": failed_finalist_id,
                    "score": 0.7,
                    "constraint_violations": 0,
                },
            ),
        )

        def persisted_genome_for(candidate_id: str):
            self.fail(
                "automatic adaptive parenting must not reload the candidate R0: "
                + candidate_id
            )

        state = SimpleNamespace(
            run=SimpleNamespace(
                generation=1,
                best_candidate_id=failed_finalist_id,
            ),
            task_manifest=SimpleNamespace(
                metadata={"optimization_protocol": OPTIMIZATION_PROTOCOL}
            ),
            pending_interventions=(),
            effective_revision_for=lambda generation: (
                revision.revision_id if generation == 0 else None
            ),
            revision=lambda revision_id: (
                revision
                if revision_id == revision.revision_id
                else (_ for _ in ()).throw(KeyError(revision_id))
            ),
            analysis_for=lambda generation: analysis if generation == 0 else None,
            candidate=lambda candidate_id: candidates[candidate_id],
            evaluation_for=lambda candidate_id: (
                object() if candidate_id == failed_finalist_id else None
            ),
            persisted_genome_for=persisted_genome_for,
            batch_for=lambda generation: None,
        )

        # The legacy completed-candidate resolver skips the screened-out
        # incumbent and would incorrectly choose the failed finalist.
        self.assertEqual(
            _search_parent_from_analysis(state, analysis), failed_finalist_id
        )
        self.assertEqual(_generation_parent_candidate_id(state), incumbent_id)
        parent = _generation_parent_genome(state, incumbent_id)
        self.assertEqual(parent.genome_digest, final.genome_digest)
        self.assertNotEqual(parent.genome_digest, seed.genome_digest)
        replay_parent = RunState.parent_genome_for_generation(state, 1)
        self.assertEqual(replay_parent.genome_digest, final.genome_digest)

    def _proposal(self, run_id: str, *, slot_index: int = 0) -> Proposal:
        state = self.director.state(run_id)
        generation = state.run.generation
        parent = state.parent_genome_for_generation(generation)
        parent_candidate_id = (
            state.analysis_for(generation - 1).search_parent_candidate_id
            if generation > 0 and state.analysis_for(generation - 1) is not None
            else None
        )
        context = GenomeMutationContextV1(
            run_id=run_id,
            generation=generation,
            slot_index=slot_index,
            slot_seed=100 + slot_index,
            parent_candidate_id=parent_candidate_id,
            parent_genome_digest=parent.genome_digest,
            generation_batch_digest="1" * 64,
            research_iteration_digest="2" * 64,
            knowledge_snapshot_digest="3" * 64,
            mutation_budget_digest="4" * 64,
            mutation_operator_id="bounded-single-parent-mutation@1",
        )
        child = apply_genome_mutation(
            parent,
            {
                "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                "operations": [
                    {
                        "op": "set_bounded_parameter",
                        "name": "ridge_alpha",
                        "value": 0.2 + generation / 10,
                    }
                ],
            },
            context,
            current_program_registry(),
        )
        return Proposal(
            proposal_id=f"proposal:genome:{generation}:{slot_index}",
            run_id=run_id,
            generation=generation,
            title=f"bounded child {slot_index}",
            changes=dict(child.scientific_program["parameter_overrides"]),
            parent_candidate_id=parent_candidate_id,
            metadata={
                "execution_protocol": "dsh_native_plugin_evolution@1",
                "evolution_genome_canonical_json": canonical_json(child.to_dict()),
                "genome_digest": child.genome_digest,
                "behavior_digest": child.behavior_digest,
                "candidate_agent_profile": resolve_candidate_agent_profile(
                    child,
                    current_program_registry(),
                ),
            },
        )

    def _spawn(self, run_id: str, *, slot_index: int = 0):
        proposal = self._proposal(run_id, slot_index=slot_index)
        self.director.submit_proposal(proposal)
        return self.director.spawn_candidate(
            run_id,
            proposal,
            candidate_id=(
                f"candidate:genome:{self.director.state(run_id).run.generation}:"
                f"{slot_index}"
            ),
            slot_index=slot_index,
        )

    def test_cross_generation_experience_keeps_full_behavior_identity(self) -> None:
        run_id = self._start_new("run:behavior-experience")
        candidate = self._spawn(run_id)
        proposal = self.director.state(run_id).proposal(candidate.proposal_id)
        analysis = GenerationAnalysis(
            run_id=run_id,
            generation=0,
            candidate_count=1,
            eligible_count=0,
            outcome="no_eligible_candidate",
            search_parent_candidate_id=candidate.candidate_id,
            ranking=(
                {
                    "candidate_id": candidate.candidate_id,
                    "classification": "scientific_gate_failed",
                    "parameters": dict(proposal.changes),
                },
            ),
            insufficient_evidence=True,
        )
        self.ledger.append(
            run_id,
            "GenerationAnalyzed",
            {"analysis": analysis.to_dict()},
        )

        experience = build_cross_generation_experience(
            self.director.state(run_id), 1
        )
        behavior = experience["generations"][0]["modifications"][
            "candidate_behaviors"
        ][0]

        self.assertEqual(behavior["behavior_digest"], proposal.metadata["behavior_digest"])
        self.assertEqual(behavior["parameters_digest"], digest(dict(proposal.changes)))
        self.assertEqual(behavior["classification"], "scientific_gate_failed")
        self.assertTrue(
            experience["generations"][0]["gate_result"][
                "insufficient_evidence"
            ]
        )
        reflection = _native_evolution_reflection_from_experience(
            experience,
            current_run_id=run_id,
        )
        self.assertEqual(reflection["avoid_behaviors"], [])

    def test_dynamic_retrieval_event_replays_and_rejects_digest_tampering(self) -> None:
        run_id = self._start_new("run:retrieval-replay")
        service = DshToolService(
            self.ledger,
            retrieval_fallback=lambda _queries, limit=8: {
                "sources": [],
                "truncated": False,
            },
        )
        service.open_admission(run_id, 3, 1)
        identity = {
            "run_id": run_id,
            "role": "researcher",
            "stage": "generation.research",
            "run_state_revision": 3,
            "stage_attempt": 1,
            "ledger_expected_revision": self.ledger.latest_seq(),
            "session_id": "session:retrieval-replay",
            "idempotency_key": "research-result",
            "child_reservation_id": "reservation:retrieval-replay",
            "activation_lease_id": "lease:retrieval-replay",
            "genome_digest": "a" * 64,
            "compiled_behavior_digest": "b" * 64,
            "phenotype_instance_digest": "c" * 64,
        }
        service.complete_retrieval(
            {
                "identity": identity,
                "arguments": {
                    "queries": ["greenhouse temperature forecasting"],
                    "retrieval_key": "replay-evidence",
                },
                "primary_result": {
                    "sources": [
                        {
                            "url": "https://example.org/temperature",
                            "title": "Greenhouse temperature forecasting",
                        },
                        {
                            "url": "https://example.net/model",
                            "title": "Protected crop temperature model",
                        },
                    ],
                    "truncated": False,
                },
            }
        )
        self.assertEqual(self.director.state(run_id).run.run_id, run_id)

        corrupt_run = self._start_new("run:retrieval-replay-corrupt")
        recorded = next(
            event
            for event in self.ledger.events(run_id)
            if event.kind == "DshRetrievalExecuted"
        )
        payload = json.loads(json.dumps(recorded.payload))
        payload["identity"]["run_id"] = corrupt_run
        payload["result_digest"] = "0" * 64
        self.ledger.append(corrupt_run, "DshRetrievalExecuted", payload)
        with self.assertRaisesRegex(ValueError, "retrieval.*digest"):
            self.director.state(corrupt_run)

    def test_native_approval_uses_recorded_adaptive_champion_not_legacy_policy(
        self,
    ) -> None:
        run_id = self._start_new("run:native-selection-authority")
        first = self._spawn(run_id)
        first_artifact = ModelArtifact(
            artifact_id="artifact:native-selection:first",
            run_id=run_id,
            candidate_id=first.candidate_id,
            model_id="greenhouse-horizon-targetwise-ridge@1",
            dataset_digest="2" * 64,
            training_partition="training_fit",
            training_rows=10,
        )
        self.director.record_artifact(first_artifact)
        self.director.record_evaluation(
            Evaluation(
                evaluation_id="evaluation:native-selection:first",
                run_id=run_id,
                candidate_id=first.candidate_id,
                score=0.2,
                passed=True,
                evaluator_digest="6" * 64,
                artifact_digest=first_artifact.digest,
            )
        )
        with self.assertRaisesRegex(ValueError, "recorded generation champion"):
            self.director.decide_promotion(
                Promotion(
                    promotion_id="promotion:native-selection:premature",
                    run_id=run_id,
                    candidate_id=first.candidate_id,
                    decision=PromotionDecision.APPROVED,
                    reason="must not bypass unified generation selection",
                )
            )
        first_analysis = GenerationAnalysis(
            run_id=run_id,
            generation=0,
            candidate_count=1,
            eligible_count=1,
            outcome="promoted",
            selected_candidate_id=first.candidate_id,
            champion_candidate_id=first.candidate_id,
            incumbent_after_candidate_id=first.candidate_id,
            search_parent_candidate_id=first.candidate_id,
            ranking=(
                {
                    "candidate_id": first.candidate_id,
                    "primary_selection_gate": True,
                },
            ),
        )
        self.ledger.append(
            run_id,
            "GenerationAnalyzed",
            {"analysis": first_analysis.to_dict()},
        )
        self.director.decide_promotion(
            Promotion(
                promotion_id="promotion:native-selection:first",
                run_id=run_id,
                candidate_id=first.candidate_id,
                decision=PromotionDecision.APPROVED,
                reason="recorded first-generation adaptive champion",
            )
        )
        self.director.advance_generation(run_id)

        second = self._spawn(run_id)
        second_artifact = ModelArtifact(
            artifact_id="artifact:native-selection:second",
            run_id=run_id,
            candidate_id=second.candidate_id,
            model_id="greenhouse-horizon-targetwise-ridge@1",
            dataset_digest="2" * 64,
            training_partition="training_fit",
            training_rows=10,
        )
        self.director.record_artifact(second_artifact)
        self.director.record_evaluation(
            Evaluation(
                evaluation_id="evaluation:native-selection:second",
                run_id=run_id,
                candidate_id=second.candidate_id,
                score=0.3,
                passed=True,
                metrics={
                    "objective_aggregation_version": "weighted_task_skill_reward@3"
                },
                evaluator_digest="6" * 64,
                artifact_digest=second_artifact.digest,
            )
        )
        second_analysis = GenerationAnalysis(
            run_id=run_id,
            generation=1,
            candidate_count=1,
            eligible_count=1,
            outcome="promoted",
            selected_candidate_id=second.candidate_id,
            champion_candidate_id=second.candidate_id,
            incumbent_before_candidate_id=first.candidate_id,
            incumbent_after_candidate_id=second.candidate_id,
            search_parent_candidate_id=second.candidate_id,
            ranking=(
                {
                    "candidate_id": second.candidate_id,
                    "primary_selection_gate": True,
                    "selection_status": "selection_only",
                },
            ),
        )
        self.ledger.append(
            run_id,
            "GenerationAnalyzed",
            {"analysis": second_analysis.to_dict()},
        )

        promotion = self.director.decide_promotion(
            Promotion(
                promotion_id="promotion:native-selection:second",
                run_id=run_id,
                candidate_id=second.candidate_id,
                decision=PromotionDecision.APPROVED,
                reason="recorded max-T adaptive champion",
            )
        )

        self.assertIs(promotion.decision, PromotionDecision.APPROVED)
        self.assertEqual(
            self.director.state(run_id).run.best_candidate_id,
            second.candidate_id,
        )

    def test_runtime_v3_approval_allows_certified_champion_to_differ_from_search_version(
        self,
    ) -> None:
        run_id = self._start_new("run:v3-split-search-and-certification")
        search_candidate = self._spawn(run_id, slot_index=0)
        certified_candidate = self._spawn(run_id, slot_index=1)
        artifact = ModelArtifact(
            artifact_id="artifact:v3-certified",
            run_id=run_id,
            candidate_id=certified_candidate.candidate_id,
            model_id="greenhouse-horizon-targetwise-ridge@1",
            dataset_digest="2" * 64,
            training_partition="training_fit",
            training_rows=10,
        )
        self.director.record_artifact(artifact)
        self.director.record_evaluation(
            Evaluation(
                evaluation_id="evaluation:v3-certified",
                run_id=run_id,
                candidate_id=certified_candidate.candidate_id,
                score=0.3,
                passed=True,
                evaluator_digest="6" * 64,
                artifact_digest=artifact.digest,
            )
        )
        cohort_digest = digest({"cohort": "v3-split"})

        def holdout(
            arm: HoldoutArm,
            candidate_id: str,
            revision_id: str,
            score: float,
        ) -> HoldoutEvaluation:
            return HoldoutEvaluation(
                evaluation_id=f"holdout:v3:{arm.value}",
                scope=EvaluationScope(
                    run_id=run_id,
                    generation=0,
                    candidate_id=candidate_id,
                    candidate_revision_id=revision_id,
                    phase=EvaluationPhase.HOLDOUT,
                    cohort_digest=cohort_digest,
                    origin_count=10,
                    holdout_arm=arm,
                ),
                score=score,
                passed=True,
                metrics={"constraint_violations": 0},
                evaluator_digest="6" * 64,
            )

        search_holdout = holdout(
            HoldoutArm.FINALIST_1,
            search_candidate.candidate_id,
            "revision:v3:search",
            0.31,
        )
        certified_holdout = holdout(
            HoldoutArm.FINALIST_2,
            certified_candidate.candidate_id,
            "revision:v3:certified",
            0.30,
        )
        incumbent_holdout = holdout(
            HoldoutArm.INCUMBENT,
            "candidate:v3:incumbent",
            "revision:v3:incumbent",
            0.20,
        )
        comparison = GenerationComparison(
            comparison_id="comparison:v3-split",
            run_id=run_id,
            generation=0,
            cohort_digest=cohort_digest,
            holdout_evaluations=(
                search_holdout,
                certified_holdout,
                incumbent_holdout,
            ),
            selected_candidate_id=search_candidate.candidate_id,
            selected_revision_id=search_holdout.scope.candidate_revision_id,
            gate_results={
                "selection_policy": "positive_delta_search@1",
                "selected_arm": HoldoutArm.FINALIST_1.value,
                "certification_selected_arm": HoldoutArm.FINALIST_2.value,
                "arms": {
                    HoldoutArm.FINALIST_1.value: {
                        "search_eligible": True,
                        "certification_eligible": False,
                    },
                    HoldoutArm.FINALIST_2.value: {
                        "search_eligible": True,
                        "certification_eligible": True,
                    },
                    HoldoutArm.INCUMBENT.value: {
                        "search_eligible": True,
                        "certification_eligible": True,
                    },
                },
            },
        )
        analysis = GenerationAnalysis(
            run_id=run_id,
            generation=0,
            candidate_count=2,
            eligible_count=2,
            outcome="promoted",
            selected_candidate_id=search_candidate.candidate_id,
            champion_candidate_id=certified_candidate.candidate_id,
            incumbent_after_candidate_id=certified_candidate.candidate_id,
            search_parent_candidate_id=search_candidate.candidate_id,
            ranking=(
                {
                    "candidate_id": search_candidate.candidate_id,
                    "search_eligible": True,
                    "certification_eligible": False,
                    "primary_selection_gate": False,
                },
                {
                    "candidate_id": certified_candidate.candidate_id,
                    "search_eligible": True,
                    "certification_eligible": True,
                    "primary_selection_gate": True,
                },
            ),
        )
        real_state = self.director.state(run_id)
        runtime_v3_task = TaskManifest.from_dict(
            {
                **real_state.task_manifest.to_dict(),
                "metadata": {
                    **dict(real_state.task_manifest.metadata),
                    "host_runtime_build": {
                        "package_version": "0.3.55",
                        "evolution_runtime_schema": (
                            "ecologyrsi-dsh.evolution-runtime/3"
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
        v3_state = Mock(wraps=real_state)
        v3_state.task_manifest = runtime_v3_task
        v3_state.run = real_state.run
        v3_state.candidates = real_state.candidates
        v3_state.promotions = real_state.promotions
        v3_state.events = real_state.events
        v3_state.generation_comparisons = (comparison,)
        v3_state.analysis_for = lambda generation: analysis if generation == 0 else None
        v3_state.comparison_for = (
            lambda generation: comparison if generation == 0 else None
        )
        v3_state.effective_revision_for = lambda generation: (
            comparison.selected_revision_id if generation == 0 else None
        )

        search_projection = _candidate_projection(v3_state, search_candidate)
        certified_projection = _candidate_projection(v3_state, certified_candidate)
        self.assertTrue(search_projection["search_version_selected"])
        self.assertFalse(search_projection["certification_selected"])
        self.assertEqual(
            search_projection["certification_status"],
            "search_version_not_certified",
        )
        self.assertFalse(certified_projection["search_version_selected"])
        self.assertTrue(certified_projection["certification_selected"])

        with patch.object(self.director, "state", return_value=v3_state):
            with self.assertRaisesRegex(
                ValueError,
                "strict-certification decision",
            ):
                self.director.decide_promotion(
                    Promotion(
                        promotion_id="promotion:v3-certified:tampered-rejection",
                        run_id=run_id,
                        candidate_id=certified_candidate.candidate_id,
                        decision=PromotionDecision.REJECTED,
                        reason="tampered ledger decision must fail closed",
                    )
                )
            promotion = self.director.decide_promotion(
                Promotion(
                    promotion_id="promotion:v3-certified",
                    run_id=run_id,
                    candidate_id=certified_candidate.candidate_id,
                    decision=PromotionDecision.APPROVED,
                    reason="strict certification winner may differ from search parent",
                )
            )

        self.assertIs(promotion.decision, PromotionDecision.APPROVED)

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

if __name__ == "__main__":
    unittest.main()
