# Champion–Challenger Adaptive Trajectory Design

## Status

Approved concept: preserve the best revision, use unsuccessful revisions as evidence, and replace the best revision only after a same-cohort comparison.

This specification defines the durable state, execution order, compatibility boundary, UI semantics, and acceptance tests for that behavior.

## Problem

The current adaptive formal lane advances from the last schema-valid local edit. A local edit can therefore become the next batch's parent even when its evaluation did not improve model quality. The safety guard checks coverage, strict execution, and constraint violations, but it does not compare the edited revision with the previous best revision on the same cohort. At the end of the lane, holdout evaluates the last revision rather than the best validated revision.

Adjacent batch scores cannot safely choose a best revision because each batch uses a different cohort. A revision must be compared with the lane's current champion on the same frozen cohort and scoring contract.

## Goals

1. Keep one durable champion revision for each finalist lane.
2. Treat each local edit as a challenger, not as an immediate champion replacement.
3. Evaluate champion and challenger on the same batch cohort before selecting either revision.
4. Replace the champion only when the challenger clears the paired local selection gate.
5. Retain rejected challenger evidence and expose it to the next local-edit proposal and generation reflection.
6. Generate every new challenger from the selected champion.
7. Bind final holdout to the lane champion, never to an unvalidated last edit.
8. Keep existing `top2_adaptive_epoch@1` runs replayable and resumable with their original semantics.
9. Make the process UI distinguish “challenger generated” from “promoted to lane champion.”

## Non-goals

- This change does not alter the generation-level Top-2 screening policy.
- This change does not weaken the final three-arm holdout comparison against the global incumbent.
- This change does not select revisions by comparing scores from different cohorts.
- This change does not add unregistered mutation targets or allow arbitrary code edits.
- This change does not rewrite events from an already-started legacy run.

## Terminology

- **Lane champion**: the best validated revision within one finalist's adaptive formal lane.
- **Challenger**: a child revision proposed from the current lane champion and scheduled for the next paired batch.
- **Warm-up batch**: the first formal batch, in which the initial revision is evaluated once and becomes the initial lane champion.
- **Paired batch**: a later formal batch in which champion and challenger are independently evaluated on the same frozen cohort.
- **Local selection gate**: the deterministic host-owned rule that decides whether a challenger replaces the champion.
- **Global incumbent**: the revision retained across generations and included as the third arm of final holdout.

## Selected approach

Use a versioned champion–challenger schedule with one warm-up batch followed by paired batches. Preserve the existing formal-batch evaluation as the challenger-side record and add a durable champion-side evaluation plus a durable comparison decision.

This approach is preferred over two alternatives:

- Rejecting every negative score would be cheap but scientifically incorrect: a negative challenger can still improve on a more negative champion, and scores from different cohorts cannot identify that improvement.
- Rolling back after comparing adjacent batch scores would reuse existing events but would make a causal claim across different cohorts.

Same-cohort paired evaluation costs more executions, but it is the smallest auditable design that can truthfully maintain a best revision.

## Version and compatibility boundary

`optimization_protocol` remains `top2_adaptive_epoch@1`; the frozen optimization schedule selects the local trajectory semantics.

- Legacy schedule schema: `ecologyrsi-dsh.top2-adaptive-epoch-schedule/1`, with `local_evaluation_mode="prequential"`.
- New schedule schema: `ecologyrsi-dsh.top2-adaptive-epoch-schedule/2`, with `local_evaluation_mode="paired_champion_challenger"`.

`OptimizationSchedule.from_dict` accepts both exact schemas. `OptimizationSchedule.default()` returns schema v2 for new runs. Existing manifests and ledger events containing schema v1 continue through the legacy execution path without reinterpretation.

The schedule remains immutable after run creation. A run cannot switch local evaluation modes while it is active or on resume.

## Durable model

### Batch roles

Add `FormalBatchArm` with:

- `champion`
- `challenger`

Extend `EvaluationScope` with optional `formal_batch_arm`. It is required for schema-v2 formal-batch evaluations, absent for schema-v1 formal-batch evaluations, and forbidden for screening and holdout scopes.

### Paired evaluations

When champion and challenger are distinct, both evaluations use `BatchEvaluation` and the same:

- run, generation, candidate, and batch index;
- cohort digest and origin count;
- dataset, split, objective, and evaluator contracts.

They differ only in `candidate_revision_id`, `formal_batch_arm`, predictions, and derived metrics.

State indexing becomes `(candidate_id, batch_index, formal_batch_arm)`. The existing two-argument `batch_evaluation_for(candidate_id, batch_index)` remains a legacy-compatible alias: it returns the unarmed legacy evaluation for schema v1; for schema v2 it returns the scheduled-revision evaluation, which is the champion in the warm-up batch and the challenger in later batches. New schema-v2 code uses an explicit arm.

When champion and challenger have the same revision identity, one `BatchEvaluation` is sufficient. `FormalBatchComparison` references that evaluation as both sides; no second model execution or synthetic duplicate evaluation is recorded.

### Comparison decision

Add immutable `FormalBatchComparison` with:

- `comparison_id`, run, generation, candidate, and batch index;
- cohort digest;
- champion-before revision ID;
- challenger revision ID;
- champion and challenger evaluation IDs and digests;
- champion score, challenger score, and `score_delta`;
- comparison-contract digest;
- safety and cell-regression gate results;
- minimum score delta;
- decision: `initial_champion`, `challenger_promoted`, or `champion_retained`;
- champion-after revision ID;
- reason code and creation time.

The event is `FormalBatchCompared`. Replay rejects duplicate keys with conflicting payloads, mismatched cohorts, mismatched candidate identities, or a decision whose champion-after binding is inconsistent.

### Trajectory state

The lane champion is derived from durable evidence:

- before batch 0: `initial_revision_id`;
- after any completed batch: `FormalBatchComparison.champion_after_revision_id`;
- after the last batch: `FormalTrajectory.final_revision_id`, which must equal the last comparison's champion-after revision.

No mutable “best score” cache is authoritative. Projections may expose a derived `champion_revision_id`.

Existing `CandidateRevision` objects remain immutable. Rejected challengers remain in the ledger and revision collection for audit and reflection.

## Local selection gate

Batch 0 records `initial_champion` after one evaluation and performs no paired comparison.

For each later batch, the challenger replaces the champion only when every condition holds:

1. Champion and challenger evaluations use the same cohort and complete scoring contract.
2. Both have complete, finite objective grids.
3. Challenger execution passes coverage, strict-chain, and constraint guards.
4. Challenger has no target/horizon cell regression relative to the champion beyond `1e-12`.
5. `challenger.score - champion.score > 0.005`.

The local `0.005` threshold reuses the practical improvement delta already used by promotion policy. The microbatch gate does not require the eight-block bootstrap used by final holdout because a 50-origin batch cannot supply that evidence. Statistical promotion remains the responsibility of the final holdout.

A challenger may replace a more negative champion when it satisfies all relative-improvement conditions. A negative sign alone neither promotes nor rejects a challenger.

Reason codes are deterministic and ordered:

1. `incompatible_comparison_contract`
2. `champion_evaluation_incomplete`
3. `challenger_evaluation_incomplete`
4. `challenger_safety_gate_failed`
5. `challenger_cell_regression`
6. `below_practical_delta`
7. `challenger_improved`

## Execution state machine

### Warm-up batch

1. Start batch 0 with `initial_revision_id` as both evaluated revision and initial champion.
2. Evaluate it once with arm `champion`.
3. Record `FormalBatchCompared(decision="initial_champion")`.
4. Unless batch 0 is also the last batch, ask the local editor for a mutation using the warm-up evidence.
5. Validate the mutation and create a challenger child from the champion.
6. Schedule that challenger for batch 1.

### Paired batch

1. Resolve champion-before from the previous comparison.
2. Resolve the scheduled challenger from the previous local-edit decision.
3. Freeze one formal batch cohort.
4. Evaluate champion and challenger independently on that same cohort. If they are the same revision because the previous proposal was kept or rejected, evaluate once and reuse the immutable evaluation evidence for comparison without duplicating model execution.
5. Record `FormalBatchCompared` using the host-owned local selection gate.
6. Set champion-after to the challenger only for `challenger_promoted`; otherwise retain champion-before.
7. Unless this is the last batch, call the local editor with paired evidence and rejected-challenger history, then create the next challenger from champion-after.
8. The next batch evaluates that challenger against champion-after.

### Final batch

The last batch records its comparison and completes the trajectory immediately with champion-after. It must not call the local editor, create a child revision, record an unvalidated local edit, or schedule another challenger.

### Failure and recovery

Every boundary is idempotent:

- a completed arm evaluation is never repeated;
- a recorded comparison is never recomputed from changed code or live data;
- a recorded local-edit proposal is never re-authored;
- a created challenger is never recreated under another identity;
- resume advances from the first missing durable boundary.

If champion evaluation fails, the work unit remains retryable and no comparison is recorded. If challenger evaluation completes but fails any required local-gate evidence, comparison records `champion_retained`; the run does not fail. Infrastructure failures remain separate from model scores under the existing recovery policy.

## Local editor evidence

For schema v2, the local editor receives:

- champion-before and challenger revision summaries;
- both same-cohort scores and bounded aggregate metrics;
- the comparison decision, score delta, gate results, and reason code;
- recent accepted and rejected challenger operations;
- the selected champion-after state as the only mutation parent.

Rejected challenger information remains evidence only. It cannot change the mutation parent identity.

The existing operation bounds, registered target catalog, duplicate-rejected-bundle rule, and schema validation remain in force.

## Holdout and next generation

Each finalist holdout arm binds to `FormalTrajectory.final_revision_id`, now guaranteed to be the lane champion. The third arm remains the global incumbent revision. Existing holdout eligibility, paired-block evidence, confidence interval, constraint, coverage, and no-cell-regression gates remain unchanged.

If neither finalist passes final holdout, the global incumbent remains both `incumbent_after_candidate_id` and `search_parent_candidate_id`. Failed challenger evidence is still included in generation reflection so the next generation can choose a different mutation direction around the retained incumbent.

## Budget and capacity reporting

For `B` batches of `N` origins per finalist:

- unique formal origins remain `B × N`;
- execution occurrences per finalist become `N + 2 × (B - 1) × N` when every paired batch has a distinct challenger;
- when champion and challenger identities are equal, that batch uses `N` occurrences rather than `2N`;
- maximum local-edit decisions per finalist become `B - 1`.

Capacity projections report the conservative distinct-challenger upper bound. UI copy identifies the extra work as paired champion/challenger validation, not additional independent source data.

## API and UI projection

Each adaptive trajectory row exposes:

- `champion_before_revision_id`;
- `challenger_revision_id`;
- `champion_score` and `challenger_score`;
- `score_delta` and `minimum_score_delta`;
- comparison decision and reason;
- `champion_after_revision_id`;
- the next challenger operation, if one was generated after that comparison.

UI labels become:

- `初始冠军已冻结`
- `挑战者已晋升为轨迹冠军`
- `挑战者未改善，继续使用原冠军`
- `下一挑战版本已生成，等待同 cohort 对照验证`
- `局部提案未通过宿主校验`

The generic label `已应用` is not used for schema-v2 trajectory selection. Legacy schema-v1 rows keep the existing wording and are visibly marked `旧版连续更新策略`.

## Testing strategy

### Model and replay tests

- Validate arm rules in `EvaluationScope`.
- Replay paired evaluations and comparisons deterministically.
- Reject conflicting duplicate comparisons and mismatched revision/cohort bindings.
- Replay a legacy v1 run unchanged.

### Selection tests

- Promote a challenger only when same-cohort delta exceeds `0.005` and every gate passes.
- Retain the champion for a lower score, equal score, insufficient delta, safety failure, incomplete evidence, contract mismatch, or cell regression.
- Allow a negative challenger to replace a more negative champion when relative gates pass.
- Never compare adjacent different-cohort scores.

### Execution and recovery tests

- Warm-up evaluates one revision and schedules one challenger.
- Paired batches evaluate both distinct revisions on one cohort.
- A rejected challenger does not become the next mutation parent.
- A promoted challenger becomes the next mutation parent.
- Same-identity champion/challenger reuses one evaluation.
- Last batch creates no unvalidated child and completes with champion-after.
- Resume after each event boundary performs only the missing work.

### Holdout tests

- Holdout binds each finalist arm to the last durable lane champion.
- Rejected final challengers never reach holdout.
- Global incumbent is retained when both lane champions fail holdout gates.

### Projection and browser tests

- Display paired scores, delta, champion decision, and next-challenger status.
- Never render schema-v2 local-edit creation as `已应用`.
- Preserve readable legacy v1 projections.

### Regression verification

- Run trajectory, replay, execution projection, work-unit, auto-progress, schedule, capacity, adaptive reflection, HTTP, and smoke suites.
- Run the complete Python and browser test suites before completion.

## Acceptance criteria

1. No schema-v2 challenger becomes lane champion without a durable same-cohort comparison decision.
2. Every schema-v2 local edit is created from the current durable champion.
3. A retained champion remains the next comparison baseline after any unsuccessful challenger.
4. Unsuccessful challenger evidence is present in the next local-editor context and generation reflection.
5. Final holdout references the last durable champion and no later unvalidated revision exists.
6. Legacy schema-v1 runs replay and resume without event reinterpretation.
7. UI wording cannot imply that generating a challenger means accepting its performance.
8. Capacity reporting includes the paired-evaluation upper bound.
