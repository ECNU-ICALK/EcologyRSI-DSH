from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests.native_fixtures import native_manifest, NativeRoleStub, start_native_run
from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.integrations.dsh_tools import DshToolService
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
from ecologyrsi_dsh.evolution.batches import (
    _generation_parent_candidate_id,
    _generation_parent_genome,
    start_generation_batch,
)
from ecologyrsi_dsh.evolution.strategies import (
    _native_evolution_reflection_from_experience,
)
from ecologyrsi_dsh.knowledge.algorithms import AlgorithmAttempt
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry


def _new_task() -> TaskManifest:
    return native_manifest(task_id="genome-replay", generations=2, metadata={
            "execution_protocol": "dsh_native_plugin_evolution@1",
            "seed_genome_template_id": "greenhouse-default@2",
            "prediction_model_id": "greenhouse-horizon-targetwise-ridge@1",
            "evaluator_id": "greenhouse_multihorizon_time_forward@4",
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
        })


class GenomeReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = EventLedger()
        self.director = EvolutionDirector(self.ledger, NativeRoleStub())

    def tearDown(self) -> None:
        self.ledger.close()

    def _start_new(self, run_id: str = "run:genome-replay") -> str:
        start_native_run(self.director, task=_new_task(), run_id=run_id)
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
                metadata={"optimization_protocol": "joint_plugin_epoch@2"}
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
        self.assertEqual(_generation_parent_candidate_id(state), incumbent_id)
        parent = _generation_parent_genome(state, incumbent_id)
        self.assertEqual(parent.genome_digest, final.genome_digest)
        self.assertNotEqual(parent.genome_digest, seed.genome_digest)
        replay_parent = RunState.parent_genome_for_generation(state, 1)
        self.assertEqual(replay_parent.genome_digest, final.genome_digest)

    def _proposal(
        self,
        run_id: str,
        *,
        slot_index: int = 0,
        extra_metadata: dict | None = None,
    ) -> Proposal:
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
                **(extra_metadata or {}),
            },
        )

    def _spawn(
        self,
        run_id: str,
        *,
        slot_index: int = 0,
        extra_metadata: dict | None = None,
    ):
        proposal = self._proposal(
            run_id,
            slot_index=slot_index,
            extra_metadata=extra_metadata,
        )
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

    def _direction(self) -> dict:
        return {
            "direction_id": "direction-1",
            "title": "Raise ridge regularization on the weak CO2 cells",
            "hypothesis": (
                "More regularization steadies the 24-hour CO2 horizon at a "
                "small cost to the 1-hour fit."
            ),
            "target_weakness": "24h CO2 skill collapses on humid origins",
            "capability_focus": "registered greenhouse ridge predictor",
            "mutation_axis": "scientific_parameter",
            "mutation_target": "ridge_alpha",
            "mutation_direction": "increase",
            "evidence_refs": ["a" * 64],
            "expected_tradeoff": "short-horizon fit for long-horizon stability",
            "success_criterion": "the paired diagnostic score improves",
        }

    def test_mutation_explanation_shows_the_parent_value_axis_and_hypothesis(
        self,
    ) -> None:
        # The candidate card used to carry `changes` alone: the child's own
        # absolute overrides, with no parent value beside them, no axis, and no
        # statement of what the model hoped the move would buy. An expert asked
        # to confirm a direction could read that alpha is 0.2 and nothing else.
        run_id = self._start_new("run:mutation-explanation")
        candidate = self._spawn(
            run_id,
            extra_metadata={
                "mutation_operations": [
                    {
                        "op": "set_bounded_parameter",
                        "name": "ridge_alpha",
                        "value": 0.2,
                    }
                ],
                "candidate_direction": self._direction(),
            },
        )

        explanation = _candidate_projection(
            self.director.state(run_id),
            candidate,
        )["mutation_explanation"]

        self.assertTrue(explanation["available"])
        self.assertTrue(explanation["parent_genome_available"])
        self.assertEqual(explanation["parent_source"], "generation_parent_genome")
        self.assertEqual(
            explanation["mutation_operator_id"],
            "bounded-single-parent-mutation@1",
        )
        # Previous → new for the one leaf that moved, not the eleven overrides
        # the child happens to carry.
        self.assertEqual(
            explanation["diff"],
            [
                {
                    "axis": "scientific_program",
                    "path": "parameter_overrides.ridge_alpha",
                    "name": "ridge_alpha",
                    "previous_value": 0.1,
                    "new_value": 0.2,
                    "change_kind": "changed",
                }
            ],
        )
        self.assertEqual(explanation["diff_count"], 1)
        self.assertEqual(
            explanation["operations"],
            [
                {
                    "op": "set_bounded_parameter",
                    # The axis is joined in from the contract catalog, so the
                    # card can say "scientific parameter" instead of leaving an
                    # expert to recognize the operation name.
                    "axis": "scientific_parameter",
                    "detail": {"name": "ridge_alpha", "value": 0.2},
                }
            ],
        )
        self.assertEqual(explanation["operation_count"], 1)
        direction = explanation["direction"]
        self.assertEqual(direction["mutation_target"], "ridge_alpha")
        self.assertEqual(direction["mutation_direction"], "increase")
        self.assertIn("24-hour CO2 horizon", direction["hypothesis"])
        self.assertEqual(direction["evidence_refs"], ["a" * 64])
        # Recomputed from the parent, because only its digest is persisted.
        # "It raised alpha" and "it raised alpha out of these eleven knobs" are
        # different claims, and only the second is reviewable.
        allowed = explanation["allowed_mutation_targets"]
        self.assertIn("ridge_alpha", allowed["scientific_parameter"])
        self.assertIn("history_steps", allowed["scientific_parameter"])
        self.assertIn("instruction_profile", allowed)

    def test_mutation_explanation_covers_a_non_parameter_axis(self) -> None:
        # An instruction-profile move leaves `parameter_overrides` untouched, so
        # `changes` reported it as no change at all -- the whole reason the
        # non-scalar axes looked inert on the web.
        run_id = self._start_new("run:mutation-explanation-instruction")
        state = self.director.state(run_id)
        parent = state.parent_genome_for_generation(0)
        operations = [
            {
                "op": "select_instruction_template",
                "role": "sample-planner",
                "instruction_template_id": "sample-planner-anomaly-aware@1",
            }
        ]
        child = apply_genome_mutation(
            parent,
            {
                "schema_version": "ecologyrsi-dsh.genome-mutation/1",
                "operations": operations,
            },
            GenomeMutationContextV1(
                run_id=run_id,
                generation=0,
                slot_index=0,
                slot_seed=100,
                parent_candidate_id=None,
                parent_genome_digest=parent.genome_digest,
                generation_batch_digest="1" * 64,
                research_iteration_digest="2" * 64,
                knowledge_snapshot_digest="3" * 64,
                mutation_budget_digest="4" * 64,
                mutation_operator_id="bounded-single-parent-mutation@1",
            ),
            current_program_registry(),
        )
        proposal = Proposal(
            proposal_id="proposal:instruction-axis",
            run_id=run_id,
            generation=0,
            title="instruction profile child",
            changes=dict(child.scientific_program["parameter_overrides"]),
            metadata={
                "execution_protocol": "dsh_native_plugin_evolution@1",
                "evolution_genome_canonical_json": canonical_json(child.to_dict()),
                "genome_digest": child.genome_digest,
                "behavior_digest": child.behavior_digest,
                "candidate_agent_profile": resolve_candidate_agent_profile(
                    child,
                    current_program_registry(),
                ),
                "mutation_operations": operations,
            },
        )
        self.director.submit_proposal(proposal)
        candidate = self.director.spawn_candidate(
            run_id,
            proposal,
            candidate_id="candidate:instruction-axis",
            slot_index=0,
        )

        projection = _candidate_projection(self.director.state(run_id), candidate)
        explanation = projection["mutation_explanation"]

        self.assertEqual(
            projection["changes"],
            dict(parent.scientific_program["parameter_overrides"]),
        )
        self.assertEqual(explanation["operations"][0]["axis"], "instruction_profile")
        diff_by_name = {row["name"]: row for row in explanation["diff"]}
        template = diff_by_name["id"]
        self.assertEqual(
            template["path"],
            "candidate_execution_program.role_profiles.sample-planner"
            ".instruction_template_ref.id",
        )
        self.assertEqual(template["axis"], "agent_program")
        self.assertEqual(template["previous_value"], "sample-planner-balanced@1")
        self.assertEqual(template["new_value"], "sample-planner-anomaly-aware@1")
        # The operator clears the candidate's instruction parameters, and that
        # clearing is a real behavior change the expert should see named.
        self.assertEqual(diff_by_name["confidence_threshold"]["change_kind"], "removed")
        self.assertEqual(diff_by_name["confidence_threshold"]["previous_value"], 0.5)


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

    def test_native_approval_rejects_analysis_without_frozen_paired_comparison(self):
        run_id = self._start_new("run:selection-authority")
        candidate = self._spawn(run_id)
        self._record_artifact_and_evaluation(run_id, candidate.candidate_id)
        analysis = GenerationAnalysis(run_id=run_id, generation=0,
            candidate_count=1, eligible_count=1, outcome="promoted",
            selected_candidate_id=candidate.candidate_id, champion_candidate_id=candidate.candidate_id)
        self.ledger.append(run_id, "GenerationAnalyzed", {"analysis": analysis.to_dict()})
        before = self.ledger.count()
        with self.assertRaisesRegex(ValueError, "recorded generation comparison"):
            self.director.decide_promotion(Promotion("promotion:forged", run_id,
                candidate.candidate_id, PromotionDecision.APPROVED, "missing paired evidence"))
        self.assertEqual(self.ledger.count(), before)


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

        self.assertEqual(kinds[:4], ["RunCreated", "RunSeedGenomeMaterialized", "ModelContractPreflightRecorded", "RunStarted"])
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

        restarted = EvolutionDirector(self.ledger, NativeRoleStub())
        self.assertEqual(
            restarted.state(run_id).materialized_seed_genome().to_dict(), before
        )

    def test_crash_between_run_created_and_seed_materialized_recovers_identically(
        self,
    ) -> None:
        run_id = "run:partial-initialization"
        run, payload = self.director.prepare_run_creation(_new_task(), run_id=run_id)
        self.ledger.append(run_id, "RunCreated", payload, event_id=f"{run_id}:created")
        with self.assertRaisesRegex(RuntimeError, "seed|initial|preflight"):
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
        with patch("ecologyrsi_dsh.evolution.batches._ensure_generation_research_iteration", return_value=None):
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
