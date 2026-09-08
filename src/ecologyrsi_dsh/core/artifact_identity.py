"""Effective fitted-artifact identity, separate from proposal compilation.

An adaptive R1 keeps its R0 proposal/compilation provenance. It does not claim
that compilation's phenotype digest as the identity of the newly fitted model.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .models import Evaluation, ModelArtifact, canonical_json, digest
from .trajectory import CandidateRevision, EvaluationPhase, EvaluationScope, HoldoutArm
from ..evolution.genome import EcologyEvolutionPluginGenome

ARTIFACT_EVENT_V2 = "ecologyrsi-dsh.artifact-recorded/2"
EVALUATION_EVENT_V2 = "ecologyrsi-dsh.evaluation-recorded/2"
ARTIFACT_REVISION_BINDING = "ecologyrsi-dsh.artifact-revision-binding/1"
FORMAL_STAGE_V2 = "ecologyrsi-dsh.formal-stage-frozen/2"


def resolve_artifact_scope(
    artifact: ModelArtifact, *, holdouts: Iterable[Any] = (),
    evaluations: Iterable[Any] = (),
) -> EvaluationScope:
    """Resolve only a previously frozen holdout or recorded scored scope."""
    scopes = [item.scope for item in evaluations]
    for holdout in holdouts:
        for arm, binding in holdout.arm_bindings.items():
            if binding["candidate_id"] != artifact.candidate_id:
                continue
            scopes.append(EvaluationScope(
                run_id=holdout.run_id, generation=holdout.generation,
                candidate_id=binding["candidate_id"],
                candidate_revision_id=binding["candidate_revision_id"],
                phase=EvaluationPhase.HOLDOUT, cohort_digest=holdout.cohort_digest,
                origin_count=holdout.origin_count, holdout_arm=HoldoutArm(arm),
            ))
    matches = {scope.scope_key: scope for scope in scopes
               if scope.scope_key == artifact.evaluation_scope_digest
               and scope.run_id == artifact.run_id
               and scope.candidate_id == artifact.candidate_id
               and scope.candidate_revision_id == artifact.candidate_revision_id}
    if len(matches) != 1:
        raise ValueError("artifact evaluation scope is not a frozen or scored revision cohort")
    return next(iter(matches.values()))


def build_artifact_revision_binding(
    artifact: ModelArtifact, revision: CandidateRevision, scope: EvaluationScope,
) -> dict[str, Any]:
    genome = EcologyEvolutionPluginGenome.from_dict(dict(revision.genome))
    if (revision.genome_digest != genome.genome_digest
            or revision.behavior_digest != genome.behavior_digest):
        raise ValueError("artifact revision genome or behavior digest mismatch")
    if (artifact.run_id != revision.run_id or artifact.candidate_id != revision.candidate_id
            or artifact.candidate_revision_id != revision.revision_id
            or scope.run_id != artifact.run_id or scope.candidate_id != artifact.candidate_id
            or scope.candidate_revision_id != revision.revision_id
            or scope.generation != revision.generation
            or artifact.evaluation_scope_digest != scope.scope_key):
        raise ValueError("artifact revision or evaluation scope binding mismatch")
    program = genome.scientific_program
    if artifact.model_id != program["predictor_ref"]["id"]:
        raise ValueError("artifact predictor does not match its actual revision")
    if dict(artifact.parameters) != dict(program["parameter_overrides"]):
        raise ValueError("artifact parameters do not match its actual revision")
    models = artifact.learned_parameters.get("models")
    if models is not None:
        if not isinstance(models, (list, tuple)) or not models:
            raise ValueError("artifact fitted models must be a nonempty array")
        for model in models:
            if not isinstance(model, Mapping) or model.get("fit_digest_sha256") != digest(
                {key: value for key, value in model.items() if key != "fit_digest_sha256"}
            ):
                raise ValueError("artifact fitted model digest mismatch")
    result = {
        "schema_version": ARTIFACT_REVISION_BINDING,
        "artifact_id": artifact.artifact_id,
        "run_id": artifact.run_id, "candidate_id": artifact.candidate_id,
        "candidate_revision_id": revision.revision_id,
        "revision_digest": revision.revision_digest, "generation": revision.generation,
        "genome_digest": genome.genome_digest, "behavior_digest": genome.behavior_digest,
        "predictor_id": artifact.model_id,
        "parameters_digest": digest(artifact.parameters),
        "learned_parameters_digest": digest(artifact.learned_parameters),
        "evaluation_scope": scope.to_dict(), "evaluation_scope_digest": scope.scope_key,
        "artifact_digest": artifact.digest,
    }
    return {**result, "binding_digest": digest(result)}


def validate_artifact_revision_binding(
    value: Any, *, artifact: ModelArtifact, revision: CandidateRevision,
    scope: EvaluationScope,
) -> dict[str, Any]:
    expected = build_artifact_revision_binding(artifact, revision, scope)
    if not isinstance(value, Mapping) or canonical_json(value) != canonical_json(expected):
        raise ValueError("artifact revision binding does not match the actual fitted artifact")
    return expected


def validate_evaluation_artifact_binding(
    evaluation: Evaluation, artifact: ModelArtifact, binding: Mapping[str, Any],
) -> None:
    if (evaluation.run_id != artifact.run_id or evaluation.candidate_id != artifact.candidate_id
            or evaluation.artifact_digest != artifact.digest
            or evaluation.candidate_revision_id != artifact.candidate_revision_id
            or evaluation.evaluation_scope_digest != artifact.evaluation_scope_digest
            or canonical_json(evaluation.evaluation_scope) != canonical_json(binding["evaluation_scope"])):
        raise ValueError("evaluation does not match the effective artifact revision and scope")


__all__ = ["ARTIFACT_EVENT_V2", "EVALUATION_EVENT_V2", "FORMAL_STAGE_V2", "ARTIFACT_REVISION_BINDING",
           "resolve_artifact_scope", "build_artifact_revision_binding", "validate_artifact_revision_binding",
           "validate_evaluation_artifact_binding"]
