from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from tests import test_genome_replay as native_fixture
from ecologyrsi_dsh.core.artifact_identity import (
    ARTIFACT_EVENT_V2, EVALUATION_EVENT_V2, FORMAL_STAGE_V2,
    build_artifact_revision_binding, validate_artifact_revision_binding,
)
from ecologyrsi_dsh.core.models import CandidateStatus, Evaluation, ModelArtifact, canonical_json, digest
from ecologyrsi_dsh.core.state import RunStateReducer, project_run_state
from ecologyrsi_dsh.core.trajectory import (
    CandidateRevision, EvaluationPhase, EvaluationScope, GenerationHoldout, HoldoutArm,
)
from ecologyrsi_dsh.evolution.genome import GenomeMutationContextV1, apply_genome_mutation
from ecologyrsi_dsh.evolution.workflow_ir import resolve_candidate_agent_profile
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry


class ArtifactRevisionIdentityTests(unittest.TestCase):
    """Exercise commands and replay against a frozen holdout predecessor state.

    The candidate's native compilation is real. Trajectory scheduling is covered
    separately; its completed revisions and frozen holdout are injected here so
    identity failures are independent of score/ranking decisions.
    """

    def setUp(self):
        self.native = native_fixture.GenomeReplayTests()
        self.native.setUp()
        self.addCleanup(self.native.tearDown)
        self.director, self.ledger = self.native.director, self.native.ledger
        self.run_id = self.native._start_new()
        seed = self.director.state(self.run_id).materialized_seed_genome()
        self.r0_genome = self._mutate(seed, ridge_alpha=0.15)
        proposal = self.native._proposal(self.run_id)
        proposal = replace(proposal,
            changes=dict(self.r0_genome.scientific_program["parameter_overrides"]),
            metadata={**proposal.metadata,
                "evolution_genome_canonical_json": canonical_json(self.r0_genome.to_dict()),
                "genome_digest": self.r0_genome.genome_digest,
                "behavior_digest": self.r0_genome.behavior_digest,
                "candidate_agent_profile": resolve_candidate_agent_profile(
                    self.r0_genome, current_program_registry()),
            })
        self.director.submit_proposal(proposal)
        self.candidate = self.director.spawn_candidate(
            self.run_id, proposal, candidate_id="candidate:revision-identity", slot_index=0)
        self.r1_genome = self._mutate(self.r0_genome, history_steps=7, ridge_alpha=0.2)
        self.r0 = self._revision("revision:r0", self.r0_genome)
        self.r1 = self._revision("revision:r1", self.r1_genome,
                                 parent_revision_id=self.r0.revision_id, source_batch_index=0)
        self.holdout = GenerationHoldout(
            holdout_id="holdout:identity", run_id=self.run_id, generation=0,
            cohort_digest="c" * 64, origin_count=10,
            arm_bindings={
                HoldoutArm.FINALIST_1.value: {"candidate_id": self.candidate.candidate_id,
                    "candidate_revision_id": self.r1.revision_id},
                HoldoutArm.FINALIST_2.value: {"candidate_id": "candidate:other",
                    "candidate_revision_id": "revision:other"},
                HoldoutArm.INCUMBENT.value: {"candidate_id": "candidate:control",
                    "candidate_revision_id": "revision:control"},
            })
        self.scope = EvaluationScope(
            run_id=self.run_id, generation=0, candidate_id=self.candidate.candidate_id,
            candidate_revision_id=self.r1.revision_id, phase=EvaluationPhase.HOLDOUT,
            cohort_digest=self.holdout.cohort_digest, origin_count=10,
            holdout_arm=HoldoutArm.FINALIST_1)
        fitted = {"coefficients": [0.2, 0.7], "intercept": 0.1}
        fitted["fit_digest_sha256"] = digest(fitted)
        self.artifact = ModelArtifact(
            artifact_id="artifact:r1", run_id=self.run_id,
            candidate_id=self.candidate.candidate_id,
            candidate_revision_id=self.r1.revision_id,
            evaluation_scope_digest=self.scope.scope_key,
            model_id=self.r1_genome.scientific_program["predictor_ref"]["id"],
            dataset_digest="2" * 64, training_partition="training_fit", training_rows=10,
            parameters=dict(self.r1_genome.scientific_program["parameter_overrides"]),
            learned_parameters={"models": [fitted]})
        self.evaluation = Evaluation(
            evaluation_id="evaluation:r1", run_id=self.run_id,
            candidate_id=self.candidate.candidate_id, candidate_revision_id=self.r1.revision_id,
            evaluation_scope=self.scope.to_dict(), artifact_digest=self.artifact.digest,
            evaluator_digest="6" * 64, score=0.4, passed=True)
        self.prefix = self.ledger.events(self.run_id)
        self.reducer = self._predecessor()

    def _mutate(self, parent, **parameters):
        return apply_genome_mutation(parent, {
            "schema_version": "ecologyrsi-dsh.genome-mutation/1",
            "operations": [{"op": "set_bounded_parameter", "name": key, "value": value}
                           for key, value in parameters.items()],
        }, GenomeMutationContextV1(
            run_id=self.run_id, generation=0, slot_index=0, slot_seed=17,
            parent_candidate_id=None, parent_genome_digest=parent.genome_digest,
            generation_batch_digest="1" * 64, research_iteration_digest="2" * 64,
            knowledge_snapshot_digest="3" * 64, mutation_budget_digest="4" * 64,
            mutation_operator_id="bounded-single-parent-mutation@1"), current_program_registry())

    def _revision(self, revision_id, genome, **kwargs):
        return CandidateRevision(revision_id=revision_id, run_id=self.run_id, generation=0,
            candidate_id=self.candidate.candidate_id, genome=genome.to_dict(),
            genome_digest=genome.genome_digest, behavior_digest=genome.behavior_digest,
            mutation_digest=digest(genome.to_dict()), **kwargs)

    def _predecessor(self):
        reducer = RunStateReducer(self.prefix[0])
        reducer.apply(self.prefix[1:])
        reducer.candidate_revisions.update({r.revision_id: r for r in (self.r0, self.r1)})
        reducer.generation_holdouts[0] = self.holdout
        return reducer

    def _state(self, _run_id):
        self.reducer.apply(tuple(event for event in self.ledger.events(self.run_id)
                                 if event.seq > self.reducer.events[-1].seq))
        return self.reducer.snapshot()

    def _record(self):
        with patch.object(self.director, "state", side_effect=self._state):
            self.director.record_artifact(self.artifact)
            self.director.record_evaluation(self.evaluation)
            self.director.record_artifact(self.artifact)
            self.director.record_evaluation(self.evaluation)
        return self.ledger.events(self.run_id)[len(self.prefix):]

    def test_r1_artifact_and_evaluation_preserve_r0_provenance_and_replay(self):
        tail = self._record()
        self.assertEqual(len(tail), 2)
        artifact_event, evaluation_event = tail
        self.assertEqual(artifact_event.payload["schema_version"], ARTIFACT_EVENT_V2)
        self.assertEqual(evaluation_event.payload["schema_version"], EVALUATION_EVENT_V2)
        original = self.reducer.snapshot().candidate_identity_binding(self.candidate.candidate_id)
        self.assertEqual(artifact_event.payload["proposal_identity_binding"], original)
        binding = artifact_event.payload["artifact_revision_binding"]
        self.assertEqual(binding["genome_digest"], self.r1.genome_digest)
        self.assertNotEqual(binding["genome_digest"], original["genome_digest"])
        self.assertEqual(dict(self.artifact.parameters)["history_steps"], 7)
        self.assertEqual(dict(self.artifact.parameters)["ridge_alpha"], 0.2)
        self.assertEqual(evaluation_event.payload["artifact_revision_binding"], binding)
        replay = self._predecessor()
        replay.apply(tail)
        self.assertEqual(replay.snapshot().artifact_revision_binding(self.artifact.artifact_id), binding)
        self.assertEqual(replay.snapshot().evaluation_for(self.candidate.candidate_id), self.evaluation)
        from ecologyrsi_dsh.api.projection import _artifact_projection
        public = _artifact_projection(replay.snapshot(), self.artifact)
        self.assertEqual(public["actual_revision_label"], "R1")
        self.assertEqual(public["identity_status"], "revision_verified")
        self.assertEqual(public["artifact_revision_binding"], binding)
        self.assertEqual(public["proposal_identity_binding"], original)

    def test_artifact_replay_rejects_cross_revision_scope_artifact_and_proposal_tamper(self):
        event = self._record()[0]
        cases = [
            ("artifact_revision_binding", "candidate_revision_id", "revision:r0"),
            ("artifact_revision_binding", "artifact_id", "artifact:other"),
            ("artifact_revision_binding", "evaluation_scope_digest", "0" * 64),
            ("artifact_revision_binding", "learned_parameters_digest", "0" * 64),
            ("proposal_identity_binding", "compiled_behavior_digest", "0" * 64),
            ("artifact", "candidate_revision_id", "revision:r0"),
            ("artifact", "evaluation_scope_digest", "0" * 64),
            ("artifact", "artifact_id", "artifact:other"),
        ]
        for section, field, value in cases:
            with self.subTest(section=section, field=field):
                payload = json.loads(canonical_json(event.payload))
                payload[section][field] = value
                with self.assertRaises(ValueError):
                    self._predecessor().apply((replace(event, payload=payload),))

    def test_new_artifact_rejects_r0_parameters_bad_fitted_digest_and_unknown_scope(self):
        changed_model = json.loads(canonical_json(self.artifact.learned_parameters))
        changed_model["models"][0]["coefficients"][0] += 1
        cases = [
            replace(self.artifact, parameters=self.r0_genome.scientific_program["parameter_overrides"]),
            replace(self.artifact, learned_parameters=changed_model),
            replace(self.artifact, evaluation_scope_digest="0" * 64),
            replace(self.artifact, model_id="predictor:wrong"),
        ]
        for artifact in cases:
            with self.subTest(artifact=artifact.digest):
                with patch.object(self.director, "state", side_effect=self._state):
                    with self.assertRaises(ValueError):
                        self.director.record_artifact(artifact)
        self.assertEqual(self.ledger.events(self.run_id), self.prefix)

    def test_binding_recomputation_rejects_changed_revision_and_unknown_fields(self):
        binding = build_artifact_revision_binding(self.artifact, self.r1, self.scope)
        for revision in [replace(self.r1, genome=self.r1_genome.to_dict(), genome_digest="0" * 64),
                         replace(self.r1, genome=self.r1_genome.to_dict(), behavior_digest="0" * 64), self.r0]:
            with self.assertRaises(ValueError):
                validate_artifact_revision_binding(binding, artifact=self.artifact,
                                                   revision=revision, scope=self.scope)
        with self.assertRaises(ValueError):
            validate_artifact_revision_binding({**binding, "ignored": "tamper"},
                artifact=self.artifact, revision=self.r1, scope=self.scope)

    def test_evaluation_replay_rejects_revision_scope_artifact_and_version_drift(self):
        artifact_event, event = self._record()
        for field, value in [("candidate_revision_id", "revision:r0"),
                             ("artifact_digest", "0" * 64)]:
            with self.subTest(field=field):
                payload = json.loads(canonical_json(event.payload))
                payload["evaluation"][field] = value
                # Re-signing the envelope does not permit another revision.
                if field == "candidate_revision_id":
                    payload["evaluation"]["evaluation_scope"][field] = value
                payload["evaluation_digest"] = digest(payload["evaluation"])
                reducer = self._predecessor()
                with self.assertRaises(ValueError):
                    reducer.apply((artifact_event, replace(event, payload=payload)))
        payload = json.loads(canonical_json(event.payload))
        payload["evaluation"]["evaluation_scope"]["cohort_digest"] = "0" * 64
        payload["evaluation_digest"] = digest(payload["evaluation"])
        with self.assertRaises(ValueError):
            self._predecessor().apply((artifact_event, replace(event, payload=payload)))
        payload = {"evaluation": self.evaluation.to_dict(),
                   "identity_binding": event.payload["proposal_identity_binding"],
                   "artifact_digest": self.artifact.digest,
                   "evaluation_digest": digest(self.evaluation.to_dict())}
        with self.assertRaisesRegex(ValueError, "version"):
            self._predecessor().apply((artifact_event, replace(event, payload=payload)))

    def test_legacy_v1_r1_events_still_replay_without_claiming_verified_v2(self):
        tail = self._record()
        legacy = []
        for event in tail:
            payload = json.loads(canonical_json(event.payload))
            payload.pop("schema_version")
            payload.pop("artifact_revision_binding")
            payload["identity_binding"] = payload.pop("proposal_identity_binding")
            legacy.append(replace(event, payload=payload))
        reducer = self._predecessor()
        reducer.apply(tuple(legacy))
        self.assertEqual(reducer.snapshot().artifact_for(self.candidate.candidate_id), self.artifact)
        self.assertIsNone(reducer.snapshot().artifact_revision_binding(self.artifact.artifact_id))
        from ecologyrsi_dsh.api.projection import _artifact_projection
        public = _artifact_projection(reducer.snapshot(), self.artifact)
        self.assertEqual(public["actual_revision_label"], "R1")
        self.assertEqual(public["identity_status"], "legacy_unverified")

    def test_non_revision_native_artifact_retains_v1_full_ledger_replay(self):
        artifact = replace(self.artifact, candidate_revision_id=None, evaluation_scope_digest=None)
        evaluation = replace(self.evaluation, candidate_revision_id=None, evaluation_scope=None,
                             artifact_digest=artifact.digest)
        self.director.record_artifact(artifact)
        self.director.record_evaluation(evaluation)
        state = project_run_state(self.ledger.events(self.run_id))
        self.assertEqual(state.artifact_for(self.candidate.candidate_id), artifact)
        self.assertNotIn("schema_version", state.events[-2].payload)

    def test_validation_and_final_test_freeze_effective_r1_genome(self):
        self._record()
        self.reducer.run = replace(self.reducer.run,
            selection_incumbent_id=self.candidate.candidate_id,
            validated_candidate_id=self.candidate.candidate_id)
        self.reducer.candidates[self.candidate.candidate_id] = replace(
            self.reducer.candidates[self.candidate.candidate_id], status=CandidateStatus.PROMOTED)
        self.reducer.task = replace(self.reducer.task,
            metadata={**self.reducer.task.metadata, "episode_id": "episode:identity"})
        for stage in ("validation", "final_test"):
            with self.subTest(stage=stage):
                with patch.object(self.director, "state", side_effect=self._state):
                    token = self.director.reserve_formal_stage(self.run_id, stage=stage,
                        candidate_id=self.candidate.candidate_id,
                        objective_family_digest="a" * 64, analysis_plan_digest="b" * 64,
                        partition_digest=digest(stage), idempotency_key=f"formal:{stage}")
                self.assertEqual(token.genome_digest, self.r1.genome_digest)
                event = self.ledger.events(self.run_id)[-1]
                self.assertEqual(event.payload["schema_version"], FORMAL_STAGE_V2)
                self.assertEqual(event.payload["candidate_revision_id"], self.r1.revision_id)
                self._state(self.run_id)
                payload = {**event.payload, "genome_digest": self.r0.genome_digest}
                replay = self._predecessor()
                replay.run = replace(replay.run, validated_candidate_id=self.candidate.candidate_id)
                previous = self.ledger.events(self.run_id)[len(self.prefix):-1]
                replay.apply(previous)
                with self.assertRaisesRegex(ValueError, "actual genome"):
                    replay.apply((replace(event, payload=payload),))
                downgraded = dict(payload)
                for field in ("schema_version", "candidate_revision_id", "artifact_revision_binding"):
                    downgraded.pop(field)
                replay = self._predecessor()
                replay.run = replace(replay.run, validated_candidate_id=self.candidate.candidate_id)
                replay.apply(previous)
                with self.assertRaisesRegex(ValueError, "downgrade"):
                    replay.apply((replace(event, payload=downgraded),))

    def test_judgment_cannot_rebind_v2_revision_or_overwrite_scientific_score(self):
        self._record()
        bad_scope = {**self.scope.to_dict(), "candidate_revision_id": self.r0.revision_id}
        wrong_revision = replace(self.evaluation, candidate_revision_id=self.r0.revision_id,
                                 evaluation_scope=bad_scope)
        with patch.object(self.director, "state", side_effect=self._state):
            with self.assertRaisesRegex(ValueError, "scientific evaluation fields"):
                self.director.record_judgment(wrong_revision)
            judged = replace(self.evaluation, metrics={"judge_status": "completed", "judge_accepted": True})
            self.director.record_judgment(judged)
            self._state(self.run_id)
        event = self.ledger.events(self.run_id)[-1]
        for changed in (wrong_revision, replace(judged, score=0.9)):
            with self.subTest(score=changed.score, revision=changed.candidate_revision_id):
                payload = {**event.payload, "evaluation": changed.to_dict(),
                           "evaluation_digest": digest(changed.to_dict())}
                replay = self._predecessor()
                replay.apply(self.ledger.events(self.run_id)[len(self.prefix):-1])
                with self.assertRaises(ValueError):
                    replay.apply((replace(event, payload=payload),))

    def test_legacy_r0_token_for_actual_r1_refuses_reopen_without_mutating_exposure(self):
        from ecologyrsi_dsh.core.exposure_registry import ScientificExposureRegistry, raw_holdout_exposure_key
        self._record()
        self.reducer.run = replace(self.reducer.run, validated_candidate_id=self.candidate.candidate_id)
        self.reducer.task = replace(self.reducer.task,
            metadata={**self.reducer.task.metadata, "episode_id": "episode:identity"})
        raw_key = raw_holdout_exposure_key(dataset_digest="2" * 64,
            split_manifest_digest="3" * 64, episode_id="episode:identity",
            stage="final_test", stage_partition_digest="c" * 64)
        registry = ScientificExposureRegistry(self.ledger)
        registry.reserve_formal_stage(raw_holdout_key=raw_key,
            objective_family_digest="a" * 64, plan_digest="b" * 64,
            idempotency_key="formal:legacy", run_id=self.run_id, stage="final_test",
            candidate_id=self.candidate.candidate_id, artifact_digest=self.artifact.digest,
            genome_digest=self.r0.genome_digest, partition_digest="c" * 64)
        before = registry.formal_exposure(raw_key)
        events_before = self.ledger.events(self.run_id)
        with patch.object(self.director, "state", side_effect=self._state):
            with self.assertRaisesRegex(ValueError, "legacy_formal_revision_conflict"):
                self.director.reserve_formal_stage(self.run_id, stage="final_test",
                    candidate_id=self.candidate.candidate_id,
                    objective_family_digest="a" * 64, analysis_plan_digest="b" * 64,
                    partition_digest="c" * 64, idempotency_key="formal:legacy")
        self.assertEqual(registry.formal_exposure(raw_key), before)
        self.assertEqual(self.ledger.events(self.run_id), events_before)

    def test_legacy_correct_r0_formal_token_retry_preserves_original_event(self):
        artifact = replace(self.artifact, candidate_revision_id=None, evaluation_scope_digest=None)
        self.director.record_artifact(artifact)
        self.reducer.run = replace(self.reducer.run, validated_candidate_id=self.candidate.candidate_id)
        self.reducer.task = replace(self.reducer.task,
            metadata={**self.reducer.task.metadata, "episode_id": "episode:identity"})
        request = dict(stage="final_test", candidate_id=self.candidate.candidate_id,
            objective_family_digest="a" * 64, analysis_plan_digest="b" * 64,
            partition_digest="c" * 64, idempotency_key="formal:legacy")
        with patch.object(self.director, "state", side_effect=self._state):
            first = self.director.reserve_formal_stage(self.run_id, **request)
            original = self.ledger.events(self.run_id)
            second = self.director.reserve_formal_stage(self.run_id, **request)
        self.assertEqual(first, second)
        self.assertEqual(original, self.ledger.events(self.run_id))
        self.assertNotIn("schema_version", original[-1].payload)
        from ecologyrsi_dsh.core.exposure_registry import ScientificExposureRegistry, raw_holdout_exposure_key
        other_key = raw_holdout_exposure_key(dataset_digest="2" * 64,
            split_manifest_digest="3" * 64, episode_id="episode:identity",
            stage="final_test", stage_partition_digest="d" * 64)
        with patch.object(self.director, "state", side_effect=self._state):
            with self.assertRaisesRegex(ValueError, "already frozen"):
                self.director.reserve_formal_stage(self.run_id, **{**request, "partition_digest": "d" * 64})
        self.assertIsNone(ScientificExposureRegistry(self.ledger).formal_exposure(other_key))


if __name__ == "__main__":
    unittest.main()
