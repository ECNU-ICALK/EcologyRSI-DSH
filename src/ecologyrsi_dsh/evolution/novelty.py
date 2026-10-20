"""Opt-in, run-local proposal novelty, independent of evaluation scores.

The complete archive is reconstructed from durable candidates and revisions.
Only its reference enters the bounded research plan; current-generation
siblings keep using their existing proposal repair contract.
"""
from collections.abc import Mapping

from ..core.models import digest
from .genome import EcologyEvolutionPluginGenome, deep_thaw_json

CANDIDATE_NOVELTY_POLICY_KEY = "candidate_novelty_policy"
CANDIDATE_NOVELTY_POLICY = "run_behavior_novelty@1"
ARCHIVE_KEY = "candidate_novelty_archive"
ARCHIVE_SCHEMA = "ecologyrsi-dsh.candidate-novelty-archive/1"
REFERENCE_SCHEMA = "ecologyrsi-dsh.candidate-novelty-reference/1"


def enabled(metadata):
    policy = metadata.get(CANDIDATE_NOVELTY_POLICY_KEY)
    if policy not in (None, CANDIDATE_NOVELTY_POLICY):
        raise ValueError("unsupported candidate_novelty_policy")
    return policy == CANDIDATE_NOVELTY_POLICY


def build_archive(state, generation):
    """Include all earlier search behaviors, never labels or holdout results.

    Filtering by generation makes this snapshot stable as the current batch
    gains candidates, and reconstructible after the run advances or restarts.
    A digest identifies executable behavior, not a successful experiment.
    """
    if type(generation) is not int or not 0 <= generation <= state.run.generation:
        raise ValueError("novelty archive generation is outside the run")
    entries = {}
    candidates = {c.candidate_id: c for c in state.candidates
                  if c.generation < generation
                  and getattr(c.role, "value", c.role) == "search"}

    def add(genome, candidate, revision_id=None):
        scientific = genome.to_dict()["scientific_program"]
        entries.setdefault(genome.behavior_digest, {
            "behavior_digest": genome.behavior_digest,
            "candidate_id": candidate.candidate_id,
            "revision_id": revision_id,
            "generation": candidate.generation,
            "predictor_id": scientific["predictor_ref"]["id"],
            "parameters": scientific["parameter_overrides"],
        })

    for candidate in sorted(candidates.values(), key=lambda c: (c.generation, c.candidate_id)):
        add(state.persisted_genome_for(candidate.candidate_id), candidate)
    for revision in sorted(state.candidate_revisions, key=lambda r: r.revision_id):
        candidate = candidates.get(revision.candidate_id)
        if candidate is not None and revision.generation < generation:
            add(EcologyEvolutionPluginGenome.from_dict(deep_thaw_json(revision.genome)),
                candidate, revision.revision_id)
    body = {"schema_version": ARCHIVE_SCHEMA, "policy": CANDIDATE_NOVELTY_POLICY,
            "run_id": state.run.run_id, "generation": generation,
            "task_manifest_digest": state.run.task_manifest_digest,
            "entries": [entries[key] for key in sorted(entries)]}
    return {**body, "archive_digest": digest(body)}


def validate_archive(archive, *, run, task):
    if not isinstance(archive, Mapping):
        raise ValueError("candidate novelty archive is missing")
    body = {key: value for key, value in archive.items() if key != "archive_digest"}
    if (set(body) != {"schema_version", "policy", "run_id", "generation", "task_manifest_digest", "entries"}
            or body["schema_version"] != ARCHIVE_SCHEMA
            or body["policy"] != CANDIDATE_NOVELTY_POLICY
            or body["run_id"] != run.run_id or body["generation"] != run.generation
            or body["task_manifest_digest"] != task.digest
            or archive.get("archive_digest") != digest(body)):
        raise ValueError("candidate novelty archive identity mismatch")
    return archive


def archive_reference(archive):
    return {"schema_version": REFERENCE_SCHEMA,
            **{key: archive[key] for key in ("policy", "run_id", "generation", "task_manifest_digest", "archive_digest")},
            "behavior_count": len(archive["entries"])}


def context_view(archive):
    """Bound model-facing examples; Host still checks every archived digest."""
    examples = sorted(archive["entries"], key=lambda e: (e["generation"], e["behavior_digest"]), reverse=True)[:8]
    return {**archive_reference(archive), "recent_examples": examples,
            "host_checks_complete_archive": True,
            "rule": "Propose a distinct complete executable behavior within the frozen direction and mutation bounds. "
                    "Prior scores are neither transferred nor used to exclude a parameter axis. "
                    "A registered pipeline default already tried needs a legal related change when the contract permits one."}
