#!/usr/bin/env python3
"""Read a campaign's durable ledger; report only bounded aggregate evidence.

No model calls, ledger mutations, hidden reasoning or raw held-out data access.
Partial screening scores are exploratory and never promote a candidate.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import statistics

from ecologyrsi_dsh.core.dsh_usage import session_usage_projection
from ecologyrsi_dsh.core.ledger import Event
from ecologyrsi_dsh.core.sample_results import decode_sample_result_batch
from ecologyrsi_dsh.core.errors import safe_binding_diagnostics
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.redaction import public_error_summary


def metrics(rows):
    errors = [r['predicted'] - r['observed'] for r in rows]
    base_errors = [r['baseline'] - r['observed'] for r in rows]
    rmse = math.sqrt(statistics.fmean(e * e for e in errors))
    baseline_rmse = math.sqrt(statistics.fmean(e * e for e in base_errors))
    return dict(n=len(rows), rmse=rmse, mae=statistics.fmean(abs(e) for e in errors),
                bias=statistics.fmean(errors), baseline_rmse=baseline_rmse,
                rmse_gain_vs_baseline=1-rmse/baseline_rmse if baseline_rmse else None,
                mean_normalized_reward=statistics.fmean(r['normalized_reward'] for r in rows))


def analyze(events):
    usages, coverage = session_usage_projection(events)
    usage = sum(m['provider_usage'].get('totals', {}).get('total_tokens', 0) for _, m in usages.values())
    created = next(e for e in events if e.kind == 'RunCreated')
    manifest = created.payload['task_manifest']
    metadata = manifest['metadata']
    session_stages = {e.payload['session_metrics']['session_id']: e.payload.get('identity', {}).get('stage', 'unknown')
                      for e in events if e.kind in ('DshSessionUsageRecorded', 'DshStructuredResultAccepted')
                      and e.payload.get('session_metrics')}
    tokens_by_stage = Counter()
    for session_id, (_, measurement) in usages.items():
        tokens_by_stage[session_stages.get(session_id, 'unknown')] += measurement['provider_usage'].get('totals', {}).get('total_tokens', 0)
    cells = {}
    candidates = {}
    checkpoints = {}
    checkpoint_identities = {}
    scopes = {}
    stages = defaultdict(list)
    for e in events:
        p = e.payload
        if e.kind == 'CandidateSpawned':
            c = p['candidate']
            candidates[c['candidate_id']] = dict(slot=c['slot_index'], role=c.get('role', 'challenger'))
        elif e.kind == 'EvaluationSampleResultsStarted':
            checkpoint = p['checkpoint']
            scope = (checkpoint['cohort_digest'], p['generation'], checkpoint['evaluation_phase'], checkpoint.get('formal_batch_index'))
            checkpoints[p['candidate_id'], p['revision']] = scope
            checkpoint_identities[p['candidate_id'], p['revision']] = {
                key: checkpoint.get(key) for key in
                ('candidate_revision_id', 'execution_scope_digest', 'holdout_arm')
            }
            scopes[scope] = dict(cohort_digest=scope[0], generation=scope[1], phase=scope[2], batch_index=scope[3])
            if p.get('supersedes_revision'):
                cells = {key: row for key, row in cells.items()
                         if not (key[0] == p['candidate_id'] and key[2] == p['supersedes_revision'])}
        elif e.kind == 'DshSessionUsageRecorded':
            stages[p['identity']['stage']].append(p)
        elif e.kind == 'EvaluationSampleResultBatchRecorded':
            scope = checkpoints[p['candidate_id'], p['revision']]
            for row in decode_sample_result_batch(p):
                key = (p['candidate_id'], scope, p['revision'], row['sample_id'])
                cells[key] = row
    grouped = defaultdict(list)
    for (cid, cohort, revision, _), r in cells.items():
        grouped[cid, cohort, revision].append(r)
    rolls = []
    for (cid, cohort, revision), rows in grouped.items():
        success = [r for r in rows if r['status'] == 'succeeded']
        by_cell = defaultdict(list)
        for r in success:
            by_cell[r['target'], r['horizon_hours']].append(r)
        rolls.append(dict(candidate_id=cid, **candidates.get(cid, {}), **scopes[cohort], revision=revision,
                          **checkpoint_identities[cid, revision],
                          statuses=dict(Counter(r['status'] for r in rows)),
                          retry_count=sum(r['retry_count'] for r in rows),
                          cells=[dict(target=t, horizon_hours=h, **metrics(rs)) for (t, h), rs in sorted(by_cell.items())]))
    # Compare only complete common origins within exactly the same frozen
    # cohort. Scores on different incomplete prefixes are never ranked.
    paired = []
    by_cohort = defaultdict(dict)
    for (cid, cohort, revision), rows in grouped.items():
        by_origin = defaultdict(list)
        for r in rows:
            if r['status'] == 'succeeded': by_origin[r['origin_timestamp']].append(r)
        n_cells = metadata['prediction_cells_per_origin']
        by_cohort[cohort][(cid, revision)] = {o: rs for o, rs in by_origin.items()
            if len({(r['target'], r['horizon_hours']) for r in rs}) == n_cells and len(rs) == n_cells}
    for cohort, candidates_by_origin in by_cohort.items():
        if len(candidates_by_origin) < 2: continue
        common = set.intersection(*(set(rows) for rows in candidates_by_origin.values()))
        if not common: continue
        results = []
        for (cid, revision), origins in candidates_by_origin.items():
            by_cell = defaultdict(list)
            for o in sorted(common):
                for r in origins[o]: by_cell[r['target'], r['horizon_hours']].append(r)
            results.append(dict(candidate_id=cid, revision=revision, **candidates.get(cid, {}),
                                **checkpoint_identities[cid, revision],
                                cells=[dict(target=t, horizon_hours=h, **metrics(rs)) for (t,h),rs in sorted(by_cell.items())]))
        paired.append(dict(**scopes[cohort], common_complete_origins=len(common),
                           origin_first=min(common), origin_last=max(common), candidates=results))
    statuses = {'RunCreated':'created','RunStarted':'running','RunResumed':'running','RunPaused':'paused',
                'RunFailed':'failed','RunCancelled':'cancelled','RunCompleted':'completed'}
    status = next(statuses[e.kind] for e in reversed(events) if e.kind in statuses)
    terminal = next((e for e in reversed(events) if e.kind in ('RunFailed', 'RunPaused', 'RunCancelled', 'RunCompleted')), None)
    formal_results = [dict(stage=e.payload.get('stage'), outcome=e.payload.get('outcome'),
                           event_id=e.event_id, seq=e.seq) for e in events if e.kind == 'FormalStageCompleted']
    return dict(run_id=created.run_id, status=status, last_seq=events[-1].seq,
                run_created_seq=created.seq, task_manifest_digest=digest(manifest),
                budget=manifest.get('budget'), optimization_protocol=metadata.get('optimization_protocol'),
                latest_stop=(dict(kind=terminal.kind, seq=terminal.seq, event_id=terminal.event_id,
                    created_at=terminal.created_at, error_code=terminal.payload.get('error_code'),
                    reason=public_error_summary(str(terminal.payload.get('reason') or '')),
                    stage=terminal.payload.get('failure_context', {}).get('stage'),
                    binding_drift=safe_binding_diagnostics(terminal.payload.get('failure_context', {}).get('binding_drift')))
                    if terminal else None),
                independent_evaluation_results=formal_results,
                dataset_id=metadata.get('dataset_id') or manifest['visible_datasets'][0], episode_id=metadata.get('episode_id'),
                configuration={k:metadata.get(k) for k in ('strategy_model_id','review_model_id','prediction_model_id','optimization_schedule','data_protocol_digest','derived_execution_budget')},
                event_counts=dict(Counter(e.kind for e in events)), reported_tokens=usage, usage_coverage=coverage,
                tokens_by_stage=dict(tokens_by_stage), token_accounting='maximum_cumulative_snapshot_per_session_not_billing',
                scoring_rows=len(cells), statuses=dict(Counter(r['status'] for r in cells.values())),
                public_failures=[dict(time=e.created_at, kind=e.kind,
                    code=e.payload.get('error_code') or e.payload.get('code'),
                    stage=e.payload.get('identity',{}).get('stage') or e.payload.get('stage'))
                    for e in events if e.kind in ('DshChildExecutionFailed','RunFailed','RunPaused')],
                stages={stage:dict(observed_sessions=len({p['identity']['session_id'] for p in ps})) for stage,ps in stages.items()},
                candidates=rolls, paired_partial_comparisons=paired,
                evidence_scope='exploratory_committed_run_results_not_independent_validation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path,nargs='?')
    parser.add_argument('--db',type=Path,required=True)
    parser.add_argument('--run-id',action='append',default=[])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if bool(args.manifest) == bool(args.run_id):
        parser.error('provide either a campaign manifest or one or more --run-id arguments')
    config=json.loads(args.manifest.read_text()) if args.manifest else {'runs': [{'run_id': rid, 'label': rid} for rid in args.run_id]}
    reports=[]
    with sqlite3.connect(args.db.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        db.row_factory=sqlite3.Row
        for run in config['runs']:
            events=[Event(seq=r['seq'],event_id=r['event_id'],run_id=r['run_id'],kind=r['kind'],
                          payload=json.loads(r['payload_json']),created_at=r['created_at'])
                    for r in db.execute('SELECT * FROM evolution_events WHERE run_id=? ORDER BY seq',(run['run_id'],))]
            if not events:
                reports.append(dict(run_id=run['run_id'],error='run_not_created'))
                continue
            reports.append(dict(label=run['label'], **analyze(events)))
    result=dict(observed_at=datetime.now(timezone.utc).isoformat(),runs=reports)
    out=args.output or (args.manifest.parent/'analysis.json' if args.manifest else None)
    if out is None:
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({'analysis_path':str(out),'runs':[{k:r.get(k) for k in ('label','status','reported_tokens','scoring_rows','statuses')} for r in reports]},ensure_ascii=False))


if __name__=='__main__': main()
