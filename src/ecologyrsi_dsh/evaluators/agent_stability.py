"""Independent inference replicas are different evidence from event replay."""
from collections.abc import Mapping
from dataclasses import replace
import math
from ..core.models import digest
from ..core.immutable import thaw_json

REPLICA_COUNT = 2
SCHEMA = 'ecologyrsi-dsh.agent-inference-stability/1'


def capacity_with_inference_replicas(report, schedule):
    """Add repeat execution cost while preserving unique observation capacity."""
    cells = report.scoring_cells_per_generation // report.candidate_origin_executions_per_generation
    budget = schedule.generation_execution_budget(cells_per_origin=cells,
                                                  holdout_inference_replicas=REPLICA_COUNT)
    extra_origins = budget['total_candidate_origins'] - report.candidate_origin_executions_per_generation
    result = replace(report,
        candidate_origin_executions_per_generation=budget['total_candidate_origins'],
        scoring_cells_per_generation=budget['total_scoring_cells'],
        candidate_origin_executions_for_run=report.candidate_origin_executions_for_run + extra_origins * report.planned_generations,
        scoring_cells_for_run=report.scoring_cells_for_run + extra_origins * cells * report.planned_generations,
    ).to_dict()
    result['holdout_inference_replicas'] = 1 if schedule.quick else REPLICA_COUNT
    return result


def replica_summary(scope, evaluation):
    execution = evaluation.metrics.get('sample_execution', {})
    return {'replica_index': scope.inference_replica, 'scope_digest': scope.scope_key,
            'cohort_digest': scope.cohort_digest, 'candidate_revision_id': scope.candidate_revision_id,
            'score': float(evaluation.score), 'passed': bool(evaluation.passed),
            'coverage': float(execution.get('coverage', 0)),
            'origin_count': scope.origin_count}


def stability_evidence(replicas):
    rows = [dict(row) for row in replicas]
    body = {'schema_version': SCHEMA, 'kind': 'independent_agent_inference', 'replicas': rows,
            'independent_observation_count_multiplier': 1}
    return {**body, 'evidence_digest': digest(body)}


def paired_stability_gate(candidate, incumbent, *, minimum_delta):
    """Conservative certification: each paired repetition must retain its gain.

    Does not count replicas as independent time blocks or replace the existing
    time-block confidence interval. Search may continue while certification fails.
    """
    def validate(evaluation):
        evidence = evaluation.metrics.get('agent_inference_stability')
        if not isinstance(evidence, Mapping) or evidence.get('schema_version') != SCHEMA:
            return None
        # Persisted holdout metrics are deeply frozen. Hash their unchanged
        # JSON representation, including the nested replica rows.
        evidence = thaw_json(evidence)
        if evidence.get('evidence_digest') != digest({k: v for k, v in evidence.items() if k != 'evidence_digest'}):
            return None
        rows = evidence.get('replicas', ())
        if len(rows) != REPLICA_COUNT or {r.get('replica_index') for r in rows} != set(range(REPLICA_COUNT)):
            return None
        if len({r.get('scope_digest') for r in rows}) != REPLICA_COUNT:
            return None
        for row in rows:
            if (row.get('cohort_digest') != evaluation.scope.cohort_digest
                or row.get('candidate_revision_id') != evaluation.scope.candidate_revision_id
                or row.get('origin_count') != evaluation.scope.origin_count
                or row.get('passed') is not True or type(row.get('score')) not in (int, float)
                or not math.isfinite(row['score']) or row.get('coverage', 0) < .95):
                return None
        return {row['replica_index']: row for row in rows}
    left, right = validate(candidate), validate(incumbent)
    if left is None or right is None:
        return {'passed': False, 'reason': 'independent_inference_evidence_incomplete', 'paired_deltas': []}
    deltas = [left[i]['score'] - right[i]['score'] for i in range(REPLICA_COUNT)]
    passed = min(deltas) > minimum_delta
    return {'passed': passed, 'reason': 'passed' if passed else 'gain_not_stable_across_agent_inference',
            'paired_deltas': deltas, 'minimum_paired_delta': min(deltas), 'replica_count': REPLICA_COUNT}
