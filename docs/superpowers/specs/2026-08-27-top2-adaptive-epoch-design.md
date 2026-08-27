# Top-2 Adaptive Epoch Evolution Design

## 1. Purpose

Replace the current one-shot formal evaluation of each Top-2 finalist with a
bounded, restart-safe, batch-adaptive trajectory while preserving the existing
outer selection behavior:

```text
4 candidates -> same 64-origin screening cohort -> existing Top-2 selection
             -> two independent 10 x 50 adaptive trajectories
             -> F1/F2/incumbent on the same 169-origin selection holdout
             -> one deterministic champion decision and global reflection
```

This design is intentionally incompatible with existing executable run state.
Old runs are archived as read-only evidence before deployment and are never
replayed by the new projector.

## 2. Locked product decisions

- Keep the current four-candidate generation, 64-origin screening, Top-2
  ranking, tie-break, and screened-out behavior unchanged.
- Interpret `500` as the number of complete forecast origins processed by
  **each** Top-2 finalist, never as scoring cells and never as a run-wide total.
- Split each finalist's 500 origins into 10 sequential local batches of 50.
- Allow zero through `max_local_edits_per_batch` local edits after each batch;
  the default is 2 and the configurable range is 1 through 5.
- A local edit may target any Host-registered mutable genome location, not only
  a numeric hyperparameter. Unregistered paths, executable code, evaluator
  policy, data partitions, budgets, credentials, and model routing are
  immutable.
- Use the forward/prequential implementation for the first delivery. Do not
  run a full parent shadow on every local batch. Batch conclusions are
  provisional; only the generation holdout decides promotion.
- After batch 10, evaluate both final revisions and the previous champion on
  the same 169-origin holdout.
- Keep per-run sample concurrency at 64 by default, configurable through 128.
  The two finalist lanes share that run limit; they do not each receive 64.
- Do not add CO2-specific or target/horizon-specific model families in this
  change.

## 3. Terminology and units

| Term | Exact meaning |
|---|---|
| forecast origin | One prediction timestamp returning the complete target x horizon vector |
| scoring cell | One target x horizon value at one forecast origin |
| cells per origin | Evaluator-derived; currently 3 targets x 3 horizons = 9 |
| screening cohort | The same 64 complete origins used by all four initial candidates |
| adaptive cohort | The same ordered 500 complete origins used independently by both finalists |
| adaptive epoch | One finalist's 500-origin trajectory inside one outer generation; a generation has two parallel adaptive epochs |
| local batch | One 50-origin slice of the adaptive cohort |
| selection holdout | 169 origins unseen by screening and local updates, used to select the next champion |
| formal validation/final test | Locked partitions used only after the whole evolution run, not the per-generation holdout |
| candidate-origin execution | One candidate revision evaluated at one complete origin |

The API, manifest, state, and UI use origin counts as primary values. Scoring
cell counts are derived display and accounting values only. Formal batch
indices are zero-based (`0..9`) in state, events, retry keys, and digests; the
UI renders them as human-facing batches `1..10`.

## 4. Current behavior to retain

The following behavior remains semantically unchanged:

1. A generation batch creates four sibling candidates from the same incumbent
   and frozen generation context.
2. All four candidates are compiled before scientific evaluation.
   Immediately after each successful compile, the Host materializes an
   immutable initial revision `R0`; this adds identity only and does not change
   the compiled behavior or screening order.
3. All four candidates use the same 64-origin screening cohort.
4. The existing deterministic screening ranking selects exactly two IDs.
5. The formal selection cohort is frozen in the ledger.
6. The two unselected candidates become `screened_out`.
7. Only a completed generation decision may advance the generation.

Golden tests must record the selected candidate IDs before the refactor and
prove they remain identical after the refactor for the same screening records.

## 5. Behavior to replace

Delete the current behavior in which `_phase_task_manifest(..., "formal")`
constructs one 500-origin task and `_evaluate_candidate()` binds one immutable
algorithm to the entire formal pass.

Replace it with two `FormalTrajectory` instances. A trajectory owns one outer
candidate and an immutable revision lineage:

```text
initial revision R0
  B1 uses R0 -> aggregate evidence -> keep/apply/reject/safety-revert
             -> explicitly activate the revision for B2
  B2 uses activated revision -> aggregate evidence -> next activation
  ...
  B10 uses R9 -> aggregate evidence -> explicitly activate final revision
  holdout uses the trajectory's final revision
```

An edit generated from batch `Bi` can only affect `Bi+1`. The edit generated
after `B10` receives its first fair evaluation on the generation holdout.
Results from an already observed batch are never recomputed with a new
revision.

## 6. Frozen run schedule

The web request keeps the existing DSH transport/runtime identity
`execution_protocol="dsh_native_plugin_evolution@1"` and adds the incompatible
optimization behavior identity `optimization_protocol="top2_adaptive_epoch@1"`.
It carries one exact schedule object:

```json
{
  "execution_protocol": "dsh_native_plugin_evolution@1",
  "optimization_protocol": "top2_adaptive_epoch@1",
  "optimization_schedule": {
    "schema_version": "ecologyrsi-dsh.top2-adaptive-epoch-schedule/1",
    "screening_origin_count": 64,
    "finalist_count": 2,
    "formal_origin_count_per_finalist": 500,
    "local_batch_origin_count": 50,
    "max_local_edits_per_batch": 2,
    "selection_holdout_origin_count": 169,
    "local_evaluation_mode": "prequential"
  }
}
```

`screening_origin_count`, `finalist_count`, and `local_evaluation_mode` are
visible read-only policy values in the parameter page. The other four values
are editable within Host-owned bounds.

Validation rules:

- all values are integers and JSON booleans are rejected as integers;
- `formal_origin_count_per_finalist >= 100`;
- `10 <= local_batch_origin_count <= formal_origin_count_per_finalist`;
- formal origins must be exactly divisible by local batch origins;
- `1 <= max_local_edits_per_batch <= 5`;
- selection holdout origins must be at least the evaluator's frozen minimum,
  currently 169;
- `candidates_per_generation == 4` and `finalist_count == 2` for this protocol;
- the dataset planner must prove enough eligible origins exist before run
  creation;
- the complete object is included in the task manifest digest and is immutable
  after the run receipt is created.

Public request and manifest code no longer accept `samples_per_update` as the
source of truth. Evaluation scopes carry origin counts directly.

## 7. Cohort planning and scientific exposure

For generation `g`, freeze these identities before their first use:

- `S_g`: 64 screening origins shared by all four candidates;
- `A`: 500 adaptation origins, divided into ten 50-origin batches and shared
  by the two finalist lanes; later generations revisit the same training
  members but execute them again under their own generation/revision scopes;
- `H_g`: 169 or more fresh selection-holdout origins shared by F1, F2, and the
  previous champion.

Within a generation, `S_g`, `A`, and `H_g` are mutually disjoint by planned
occurrence. The planner consumes eligible source origins in deterministic
causal order; when the source population is exhausted it wraps to the first
origin and increments that occurrence's `reuse_index`. Thus a repeated source
timestamp is a new auditable occurrence, never an accidental duplicate within
one cohort. Batch order still respects the evaluator's causal
timestamp/maximum-horizon maturity rules, and cohort selection depends only on
identity and timestamp metadata, never labels or predictions.

`H_g` is a model-selection holdout because its aggregate result guides later
generations. It is not presented as final validation. Raw holdout labels,
predictions, and timestamps never enter the local editor or global proposer.
Only the post-decision aggregate comparison enters the next generation's
reflection.

Reusing a source origin means reusing only its frozen input vector and batch
order. Predictions, sample-result events, model usage, and local-edit feedback
are never reused: each planned occurrence receives a fresh sample identity and
remains scoped to its generation/revision. For a frozen planned generation
count `G`, formal origins per finalist `F`, and selection holdout `H`, the exact
planned-occurrence formulas are:

```text
planned origin occurrences           = F + G * (64 + H)
candidate-origin executions / gen   = 4 * 64 + 2 * F + 3 * H
candidate-origin executions / run   = G * (4 * 64 + 2 * F + 3 * H)
scoring cells / run                 = executions / run * cells_per_origin
```

`G` is the generation limit already frozen by the existing outer
max-generation/candidate-budget policy; the new protocol never starts a
partial four-candidate generation.

Run creation requires at least one eligible causal origin. The capacity report
shows planned occurrences, available source origins, the deterministic reuse
policy, and the number of occurrences that will wrap. Only an empty eligible
population blocks creation; the planner never silently truncates a cohort or
overlaps screening and holdout occurrences.

## 8. Local edit contract

The outer generation candidate proposer remains single-axis and unchanged.
Batch-local editing uses a separate DSH role and contract so increasing the
local edit limit cannot change the four initial candidates.

Input to `candidate.local_edit`:

- run, generation, candidate, trajectory, revision, and batch identities;
- parent genome and behavior digests;
- Host-generated mutation catalog and per-axis trust-region contracts;
- the current batch's aggregate 3 x 3 metrics, coverage, constraint status,
  and bounded weakness identifiers;
- prior local decisions as digests and bounded outcome categories;
- `maximum_operations` frozen from the schedule.

Output:

```json
{
  "schema_version": "ecology-local-edit@1",
  "decision": "keep",
  "operations": [],
  "evidence_refs": [],
  "expected_effect_cells": [],
  "risk_cells": []
}
```

For `decision="mutate"`, operations contain 1 through K Host-registered atomic
edits. The Host rejects duplicate or conflicting paths and applies the bundle
atomically. Each operation retains its existing per-axis trust-region bound.
A failed schema repair, compile, or smoke check records `rejected` and keeps the
parent revision. `KEEP` is valid and does not create an identical revision.

The wire-level model field is a proposal decision (`keep|mutate`). The durable
Host outcome is a separate enum (`kept|applied|rejected`); `applied` means only
that validation/compile/smoke succeeded, not that the edit scientifically
improved the model. Transport/provider failures produce no Host outcome and
remain scheduler retries.

No free-form model rationale is an executable input. Audit explanations are
derived from bounded coordinates and metric references.

## 9. Immutable execution model

Add these core identities:

- `CandidateRevision`: immutable genome/behavior identity and parent lineage;
- `FormalTrajectory`: one selected candidate's ordered revision/batch process;
- `FormalBatch`: one candidate, revision, batch index, cohort digest, and
  restart checkpoint;
- `BatchEvaluation`: aggregate metrics for one formal batch;
- `TrajectoryRevisionActivation`: the exact revision selected for the next
  batch or final holdout after a kept, applied, rejected, or safety-revert
  decision;
- `GenerationHoldout`: one frozen cohort and its three evidence arms;
- `GenerationComparison`: paired F1/F2/incumbent evidence and deterministic
  champion decision.

`Candidate` remains the outer four-way search identity. It is never overwritten
when a local edit occurs. Artifacts, evaluations, sample checkpoints, model
usage, and retry authority are scoped by:

```text
(run_id, generation, candidate_id, candidate_revision_id,
 evaluation_phase, batch_index, cohort_digest)
```

The seed genome is materialized as the generation-0 incumbent revision. A
promotion binds the outer candidate ID, its final revision ID, and the exact
generation-comparison digest.

Do not add a second outer candidate lifecycle. A finalist remains outer status
`spawned` while its `FormalTrajectory` carries active progress. After all
holdout evidence is bound, both finalists receive final-revision evaluations
and become `evaluated`; the selected finalist then becomes `promoted` and the
other `rejected`. If the incumbent is retained, both current finalists become
`rejected` and the incumbent's prior outer status is not rewritten.

## 10. Events and replay

Retain the current outer screening events:

- `CandidateScreeningRecorded`
- `FormalSelectionCohortFrozen`
- `CandidateScreenedOut`

Add the new formal-stage events:

- `RunAdaptationCohortFrozen`
- `GenerationCohortsFrozen`
- `CandidateRevisionCreated`
- `FormalTrajectoryStarted`
- `FormalBatchStarted`
- `FormalBatchEvaluated`
- `LocalEditProposed`
- `LocalEditDecided`
- `TrajectoryRevisionAdvanced`
- `FormalTrajectoryCompleted`
- `GenerationHoldoutFrozen`
- `HoldoutEvaluationRecorded`
- `GenerationCompared`
- `CandidateEffectiveRevisionFrozen`
- `GenerationChampionSelected`

Sample-result and model-usage events include the complete evaluation scope.
Event IDs are deterministic from run/generation/candidate/revision/batch/arm.
Replaying the same event is idempotent; a conflicting payload with the same ID
fails closed.

The projector validates monotonic batch order, exact cohort digests, one active
revision per batch, no edit applied to its evidence batch, exactly two formal
trajectories, exactly three holdout roles, and no champion decision before all
three arms complete.

After every completed batch there must be exactly one
`TrajectoryRevisionAdvanced` event. A model proposal has wire decision
`keep|mutate`; the Host records the distinct outcome `kept|applied|rejected`.
Infrastructure or incomplete-coverage failures remain retry/pause conditions
and cannot be converted into scientific decisions. An empirical absolute
physical-constraint failure may skip the editor and activate the most recent
previously safe revision with reason `safety_revert`. For display batch 10 the
strict order is: evaluate internal batch 9, decide/apply/keep/reject or safety
revert, freeze the resulting active revision, complete the trajectory, then
freeze the generation holdout.

When a finalist wins, its effective candidate identity is the selected final
revision—not the outer proposal's initial genome. The frozen effective revision
and identity binding become the only parent genome source for the next
generation, judge, artifact/evaluation binding, promotion, and formal-stage
seal. Every generation writes one effective-revision binding; retaining the
incumbent binds the same already frozen revision to the new comparison digest.

## 11. Work-unit scheduler and recovery

Replace the assumption that one worker invocation must advance a complete
generation with `execute_next_work_unit()`.

Work-unit kinds are:

1. prepare generation and spawn four candidates;
2. run/finalize four-way screening;
3. freeze Top 2 and screen out the other two;
4. start two formal trajectories;
5. execute the next missing batch for each eligible lane, in parallel;
6. create/decide the next local edit for each completed lane;
7. freeze and run the three holdout arms;
8. compare, reflect, select/retain champion, and advance generation.

Each invocation executes one scheduler work-unit wave. A formal wave may carry
one independently durable lane sub-unit for each eligible finalist; a holdout
wave may carry independently durable arm sub-units. The run lease covers the
wave, while every sub-unit has its own scope, events, retry key, and result. If
lane A commits and lane B fails, replay retains A and the next selector returns
only B. This preserves parallel lanes without allowing two unrelated workers
to mutate one run concurrently.

The wave returns:

```python
WorkUnitResult(
    state_changed: bool,
    generation_advanced: bool,
    run_terminal: bool,
    retry_required: bool,
)
```

Recovery rules:

- an incomplete batch resumes the same revision and cohort and evaluates only
  missing complete origins;
- a completed batch with missing analysis rebuilds deterministically from
  stored sample results;
- a persisted proposal with no decision reruns only pure Host validation,
  compilation, and smoke checks;
- one completed finalist never reruns because the sibling is incomplete;
- holdout arms recover independently and their frozen revision/cohort cannot
  change;
- pause closes admission for new origins/batches and drains current complete
  origins; cancel prevents any new admission;
- infrastructure failure retries or pauses; it never becomes a scientific
  rollback.

## 12. Concurrency and queue policy

Use the existing two-level admission architecture; do not create a second,
competing limiter:

- Python `RunSampleAdmission` is the run-level authority with configured
  default 64 and maximum 128;
- DSH `provider-stage-gate` is the provider/model-route authority with physical
  maximum 128.

Every actual provider request must pass both. Screening candidates and formal
lanes share these governors. Candidate-level concurrency cannot multiply the
sample limit. Only origins from the current durable work unit may enter the
queue; future local batches are not pre-enqueued.

The gateway retry key includes candidate ID, revision ID, phase, batch index,
and holdout arm so parallel finalist failures cannot overwrite each other.
The run-level sample concurrency is the only user-configurable in-flight
source of truth. Candidate concurrency only selects lane work; origin-wave
size only bounds queue construction; provider-route 128 is an internal
physical ceiling.

## 13. Generation comparison

After both trajectories complete, run these arms on the exact same `H_g`:

- `finalist_a_final_revision`;
- `finalist_b_final_revision`;
- `incumbent_revision_before_generation_g`.

The three arms always produce three independently scoped evidence and usage
records, even when two arms have the same genome or behavior digest. No
cross-arm prediction cache may reduce the fixed `3 * H` comparison budget.

Required hard gates:

- complete 3 x 3 target/horizon matrix;
- coverage at least 0.95 in the overall result and every required cell;
- zero physical-constraint violations;
- same cohort, evaluator, baseline, and scoring profile for all arms;
- enough valid paired blocks for the frozen evaluator policy.

Default promotion gates:

- overall paired score improvement greater than 0.005;
- the configured paired stability/confidence lower bound is greater than 0;
- no target/horizon cell has practical skill regression below -0.01.

If both finalists pass, rank by stability lower bound, overall delta, worst-cell
delta, then deterministic candidate ID. If one passes, select it. If neither
passes, retain the incumbent. The independent judge explains evidence but
cannot override Host gates.

The global reflection records F1/F2 vs incumbent 3 x 3 deltas, applied/kept/
rejected local edit bundles, hard failures, the deterministic decision, and
bounded next-generation focus/avoid coordinates.

## 14. Parameter page and projection

The parameter page groups controls into:

1. outer search: max generations, four candidates, candidate concurrency, total
   candidate budget, and visible read-only `64 -> Top 2` policy;
2. finalist adaptation: 500 formal origins, 50 local-batch origins, maximum 2
   edits per batch, 169 selection-holdout origins;
3. throughput: gateway origin-wave limit 64 and sample concurrency 64/128;
4. reproducibility: seed and online retrieval controls;
5. scientific policy: visible coverage, completeness, confidence, physical
   constraint, improvement, and no-regression gates.

The dynamic per-generation budget for defaults is:

```text
screening              4 x 64  =   256 candidate-origins
two adaptive lanes     2 x 500 = 1,000 candidate-origins
three holdout arms     3 x 169 =   507 candidate-origins
total                            1,763 candidate-origins
scoring cells           1,763 x 9 = 15,867
```

For a five-generation run, planned execution is therefore 8,815
candidate-origins and 79,335 scoring cells. Source identities are not
multiplied as “unique data”: later generations reuse frozen source vectors with
explicit occurrence indices, and the UI shows both current-generation progress
and whole-run planned progress.

The process page shows generation, screening, finalist lane A/B, batch index,
active revision, origin completion, applied/kept/rejected edits, holdout arms,
actual in-flight requests, bounded queue, provider wait/backoff, last durable
activity, throughput, and ETA. Percent completion is derived from durable
candidate-origin work, not model prose or synthetic stage percentages.

## 15. Incompatible cutover

Before deployment:

1. stop automatic admission and archive/cancel active old runs;
2. stop 8777 and 8848 only during the deployment window;
3. copy the current SQLite/event store to a timestamped read-only archive and
   export static run projections for human access;
4. start the new version with a fresh event database and new schema identity;
5. do not implement old event replay or old run resume in the new projector;
6. preserve outer screening/Top-2 source, but delete old one-shot formal-500,
   `samples_per_update` public conversion, old formal progress projection, and
   obsolete protocol branches;
7. start one small deterministic smoke run before a full default run.

## 16. Acceptance criteria

- The same frozen screening fixtures select the same Top-2 IDs as before.
- Each selected candidate has exactly ten ordered 50-origin batches.
- A local edit never affects the batch that produced its evidence.
- Each local bundle contains zero through the configured maximum operations;
  default 2 and maximum 5 are enforced by the Host.
- Unregistered/conflicting/oversized edits are rejected atomically.
- Both lanes share the same 500-origin/batch identities but never share model
  feedback or revision state.
- Actual in-flight sample requests never exceed the run's configured 64 and
  never exceed the service route cap.
- A crash at every event boundary resumes without duplicate origins, revisions,
  local decisions, holdout arms, or usage accounting.
- F1, F2, and incumbent holdout evidence use the same 169-origin digest.
- Hard-gate or practical-regression failure retains the previous champion.
- The UI distinguishes origins, candidate-origin executions, and scoring cells.
- Old runs are available only through the archived static export and cannot be
  resumed by the new service.
