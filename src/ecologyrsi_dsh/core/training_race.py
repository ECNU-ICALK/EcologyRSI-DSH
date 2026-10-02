"""Host-owned identities for a frozen five-arm training race."""
from .models import CandidateRole, canonical_json, digest
from .immutable import thaw_json
from ..evolution.effect_contracts import hard_effect_failure


def race_bindings(state, generation):
    bindings = {c.candidate_id: state.initial_revision_for(c.candidate_id).revision_id
                for c in state.candidates
                if c.generation == generation and c.role is CandidateRole.SEARCH}
    if generation:
        revision_id = state.effective_revision_for(generation - 1)
        if revision_id is None:
            raise ValueError("race requires the prior frozen incumbent")
        bindings[state.revision(revision_id).candidate_id] = revision_id
    else:
        control = next(c for c in state.candidates if c.role is CandidateRole.INCUMBENT_CONTROL)
        bindings[control.candidate_id] = state.initial_revision_for(control.candidate_id).revision_id
    if len(bindings) != 5:
        raise ValueError("race requires four search candidates and one incumbent")
    return bindings


def validate_race_plan(planned, schedule, *, candidates, revisions, prior_revision_id, previous):
    """Used by both writes and replay; never relax scope validation globally."""
    adaptation = planned.adaptation
    if adaptation is None or not schedule.race:
        raise ValueError("race is missing its generation adaptation plan")
    current = {c.candidate_id for c in candidates.values()
               if c.generation == planned.generation and c.role is CandidateRole.SEARCH}
    bindings = dict(planned.revision_bindings or {})
    if len(current) != 4 or len(bindings) != 5 or not current < bindings.keys():
        raise ValueError("race bindings must contain the four candidates and incumbent")
    for candidate_id, revision_id in bindings.items():
        revision = revisions.get(revision_id)
        if revision is None or revision.candidate_id != candidate_id:
            raise ValueError("race revision binding mismatch")
        if candidate_id in current:
            if revision.parent_revision_id is not None:
                raise ValueError("race candidates must bind frozen R0")
        elif planned.generation:
            if revision_id != prior_revision_id:
                raise ValueError("race control is not the prior incumbent")
        elif candidates[candidate_id].role is not CandidateRole.INCUMBENT_CONTROL or revision.parent_revision_id is not None:
            raise ValueError("initial race control must be the seed incumbent")
    if (adaptation.dataset_id != planned.dataset_id
            or adaptation.episode_id != planned.episode_id
            or adaptation.seed != planned.seed
            or adaptation.adaptation_digest != planned.adaptation_digest
            or adaptation.batch_digests != planned.adaptation_batch_digests
            or adaptation.origin_count != 35 or len(adaptation.batches) != 2
            or [b.origin_count for b in adaptation.batches] != [10, 25]
            or planned.screening.origin_count != 10
            or planned.screening.shared_candidate_count != 5
            or planned.holdout.origin_count != 50 or planned.holdout.shared_arm_count != 2
            or adaptation.batches[0].origin_ids != planned.screening.origin_ids):
        raise ValueError("race cohort differs from frozen 10/25/50 plan")
    stages = (planned.screening, adaptation.batches[1].cohort, planned.holdout)
    for left, right in zip(stages, stages[1:]):
        if right.origins[0].origin_timestamp <= max(o.maximum_target_timestamp for o in left.origins):
            raise ValueError("race stages must be target-time purged")
    all_origins = (*adaptation.origins, *planned.holdout.origins)
    if len({o.origin_id for o in all_origins}) != 85 or any(o.reuse_index for o in all_origins):
        raise ValueError("race must use 85 fresh origins")
    for prior in previous:
        if prior.generation >= planned.generation:
            continue
        if stages[0].origins[0].origin_timestamp <= max(o.maximum_target_timestamp for o in prior.holdout.origins):
            raise ValueError("race generations must be chronological and purged")


def race_screening_revision(planned, candidate_id):
    if planned is None or not planned.revision_bindings:
        return None
    return planned.revision_bindings.get(candidate_id)


def validate_race_selection(planned, records, selected, candidates):
    by_id = {r['candidate_id']: r for r in records}
    if set(by_id) != set(planned.revision_bindings):
        raise ValueError("all five race arms must be sealed before selecting or exposing feedback")
    searches = [c for c in candidates if c.generation == planned.generation
                and c.role is CandidateRole.SEARCH]
    ranked = sorted(searches, key=lambda c: (bool(hard_effect_failure(by_id[c.candidate_id].get('metrics', {}))),
        by_id[c.candidate_id]['constraint_violations'],
        -by_id[c.candidate_id]['score'], c.slot_index, c.candidate_id))
    if list(selected) != [ranked[0].candidate_id]:
        raise ValueError("race finalist must follow the frozen same-cohort ranking")


def validate_race_diagnostic_reuse(evaluation, source):
    if source is None or evaluation.scope.batch_index != 0:
        raise ValueError("diagnostic reuse requires sealed screening evidence")
    expected = {**source.payload['metrics'], 'source_screening_event_id': source.event_id,
                'reused_screening_evidence': True, 'additional_prediction_executions': 0}
    if (evaluation.scope.candidate_revision_id != source.payload['candidate_revision_id']
            or evaluation.score != source.payload['score']
            or evaluation.passed != source.payload['passed']
            or canonical_json(thaw_json(evaluation.metrics)) != canonical_json(thaw_json(expected))
            or evaluation.evaluator_digest != digest({'evaluator': expected['screening_evaluator_digest']})):
        raise ValueError("diagnostic reuse cannot alter or manufacture screening evidence")
