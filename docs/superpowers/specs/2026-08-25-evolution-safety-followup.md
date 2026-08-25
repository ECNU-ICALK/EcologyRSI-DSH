# Evolution Safety Follow-up Design

## Context

The runtime-stability work removed the observed infinite research retry and
ten-minute critic stall from the controlled DSH-native diagnostic path. Two
fixed runs completed with explicit diagnostic-only, no-promotion outcomes.
The follow-up review nevertheless found three runtime races and one formal
scientific-boundary defect that must be closed before treating the current
branch as stable:

1. A negated diagnostic list can incorrectly absorb a later affirmative
   passive clause, for example `No eligibility, and promotion is claimed.`
2. The structured-role timeout only bounds child creation. A child result that
   settles after the deadline may still be admitted and persisted, and a child
   that ignores abort may wait forever.
3. Pause/cancel can snapshot an empty pending-start set, after which a stage
   that was awaiting schema or reservation data can still launch a child.
4. `execute_formal_stage` accepts an arbitrary aggregate mapping whose only
   scientific assertion is `{"outcome": "passed"}`.
5. A native `pause`/`cancel` request can hold the server-wide mutation lock
   while DSH quiescence or the generation barrier waits. One pending pause
   then makes unrelated archive/create requests hit the proxy's 30-second
   timeout and appear as a service outage.
6. The DSH-compatible schema subset discards static string-pattern and length
   constraints. A sample reflection can therefore persist a wrong wave digest
   or a prediction-cell ID before Python detects the Host-identity mismatch and
   fails the whole candidate.
7. Retryable gateway failure is deferred before the ordinary worker retry
   limit and re-enters with that local counter reset. The durable attempt count
   controls only backoff, so a run can remain scheduled forever at the
   five-minute delay cap.

The review also found that the legacy promotion helper advertises a paired
moving-block bootstrap while sampling independent single-day blocks in hash
order. The active DSH-native diagnostic path uses a different, correct
contiguous-block selector, but the mislabeled helper must be corrected before
it is reused for a promotion decision.

Operational evidence from the restarted production service also showed that
the web console's current default full run is unsuitable for an interactive
health check: 177 origins × four candidates under 60-second provider pacing is
at least 11.8 hours per generation and about 59 hours for five generations,
before model latency and retries. The already supported one-origin/nine-cell
budget needs an explicit diagnostic-only UI preset and honest evidence labels.

## Goals

- Reject affirmative eligibility, gate, selection, or promotion assertions
  even when they follow a negated clause.
- Make every structured-role timeout a hard end-to-end deadline covering child
  creation, child result settlement, validation, and persistence admission.
- Close a run's child-launch fence synchronously before pause/cancel drains
  outstanding work.
- Bind every sample-stage structured result to Host-owned wave and sample
  identities before launch and before any accepted receipt can be persisted.
- Convert unbounded gateway defer into a durable, finite, checkpoint-preserving
  circuit pause with an explicit operator recovery action.
- Keep diagnostic runs structurally unable to reserve or pass a formal stage.
- Do not allow a formal pass until a host-derived canonical formal assessment
  and its frozen UQ bindings are integrated.
- Make the legacy promotion bootstrap operate on ordered, contiguous calendar
  blocks and report the method it actually executes.
- Keep unrelated mutations responsive while a run is draining after pause or
  cancel.
- Make quick diagnostic launch the explicit web-console default, keep full
  selection-grade evolution opt-in, and show honest pacing-cost and scientific
  scope labels for both.
- Rebuild the isolated runtime and verify new cucumber and tomato diagnostic
  runs through terminal acceptance.

## Non-goals

- This change does not expose validation or final-test rows.
- It does not claim that the full formal UQ pipeline is integrated.
- It does not add complexity/identifiability gates or redesign the greenhouse
  predictor family.
- It does not compare scores across datasets or frozen cohorts.
- It does not add synthetic progress percentages or billing-token claims.
- It does not trust a caller-supplied mode or budget-class label; the server
  continues deriving scientific eligibility from numeric budget evidence.
- It does not rewrite model output, broaden the DSH schema dialect, or migrate
  a historically accepted invalid structured receipt.
- It does not automatically resume a circuit-paused run or classify a
  transient gateway outage as candidate scientific failure.

## Design

### 1. Diagnostic claim scoping

Shared negation is accepted only when the complete coordinated phrase is one
bounded assertion. A suffix containing its own finite affirmative predicate is
a new claim scope and is rejected. Regression controls cover both affirmative
passive clauses and legitimate shared-negation lists.

### 2. End-to-end structured-role deadline

One deadline owns the entire child lifecycle. Child creation and child result
settlement race the same deadline. Once it expires, the operation is classified
as an operational timeout, the abort signal is sent, late success/failure is
drained without admission, and no result is persisted. A child that ignores
abort cannot keep the stage promise pending.

### 3. Pause/cancel launch fence

Pause/cancel first closes a per-run launch fence synchronously. Every launch
path performs a synchronous admission check after its last preparatory await
and registers itself as pending without an intervening await. Draining then
waits for a stable pending set. Releasing an earlier schema/reservation await
after pause cannot start a child or workflow.

### 4. Host-bound sample structured contracts

The Host specializes the already cloned schema after validating the structured
stage context and before launching any child or Workflow. All sample stages
receive an exact `wave_digest` const. Plan and critic decisions receive the
exact outer sample-ID enum; reflection receives the single outer
`context.sample.sample_id` const and must never substitute a prediction-cell
ID. Prompts state the same copy-exactly rule.

A schema-rejected or missing sample result is not persisted. It may consume the
existing bounded fresh-reservation retry under the same absolute stage
deadline. Control, timeout, authorization, infrastructure, and persistence
errors are not output retries. Python retains its final fail-closed contract
checks, and an already durable invalid receipt is never ignored or repaired in
place.

### 5. Persistent bounded gateway circuit

Gateway defer is a durable state decision, not an in-memory timer followed by a
best-effort heartbeat. Under an optimistic SQLite sequence check, the director
either appends the next scoped `GatewayRetryScheduled` event or atomically
transitions the run to `RunPaused` with code
`gateway_retry_circuit_open`. A stable failure identity prevents duplicate
workers from counting the same logical failure twice.

The model-gateway breaker opens after six consecutive orchestration-level
failures or 30 minutes in one epoch, whichever occurs first. Durable progress,
generation advance, or explicit resume starts a new epoch; stage-start events
do not. Circuit pause preserves generation, candidate, and sample checkpoints,
does not create candidate/run failure, and cannot wake itself. Explicit resume
retries the same checkpoint and the UI explains
`check_gateway_then_resume`. Provider response bodies and raw exception text
remain private.

### 6. Formal stage fail-closed boundary

Formal reservation requires a selection-eligible budget and a legitimate
locked selection incumbent. Diagnostic smoke runs are rejected before any
formal exposure is reserved. The current generic mapping evaluator cannot
produce a passing formal result. Until a host-owned evaluator, frozen baseline
UQ artifact, and canonical `FormalFitnessAssessment` are wired together, a
formal pass remains explicitly unavailable. Event replay also refuses a
`passed` completion that lacks the canonical assessment bindings.

### 7. Explicit quick diagnostic and full-run cost boundary

The web console presents two explicit presets. Quick diagnostic launches one
generation, one candidate, one origin/nine prediction cells, concurrency one,
and max-candidates one. It is labeled diagnostic-only and cannot claim a
champion, formal best, selection, validation, final test, or promotion. Full
selection-grade evolution remains available only as an explicit choice and
retains the existing five generations, four candidates per generation, 1600
cells per update, and max-candidates twenty.

The console derives a lower-bound duration estimate from the selected origin
count, candidate count, generation count, and the fixed 60-second provider
pacing. It explains that model latency and retries add to that bound. The mode
is UI convenience only: launch requests send numeric budgets, while the server
derives `sample_budget_class` and enforces formal eligibility. Task 6's formal
fail-closed boundary is a prerequisite.

### 8. Ordered moving-block bootstrap

Validated promotion evidence retains a unique integer
`origin_block_index`. Candidate and incumbent indices must match exactly.
Bootstrap draws use non-circular contiguous three-day blocks, never cross a
calendar gap, and truncate to the paired cohort size. Evidence with too few
paired days or legal starts fails closed. Returned metadata records block
length and legal-start count in addition to the versioned method.

### 9. Bounded global mutation-lock scope

The HTTP dispatcher treats native pause/cancel as potentially long-running
control operations, just as it already treats generation advancement. The
outer server-wide lock is not held across DSH quiescence or a per-run
generation barrier. The action path still acquires that lock around the short
durable pause/cancel event boundary, preserving coherent append-only state
without blocking unrelated runs.

## Acceptance

- All new regression tests demonstrate RED before implementation and GREEN
  after implementation.
- Late structured-role success persists zero results; late rejection and an
  abort-ignoring child both return the operational-timeout classification.
- A schema-blocked stage released after pause starts zero children and the
  control operation returns without waiting for a stage timeout.
- Wrong sample wave/outer IDs are rejected by the specialized child schema,
  write zero accepted receipts on the first attempt, and can recover only via
  the bounded fresh-reservation sample retry.
- A permanently unavailable gateway reaches a durable circuit pause within six
  logical failures or 30 minutes; restart preserves the decision, explicit
  resume restarts the epoch at the same checkpoint, and duplicate workers do
  not double-count one failure.
- Diagnostic formal reservation and `lambda: {"outcome": "passed"}` are
  rejected without setting validated/final-test candidate state.
- The default web launch request is exactly 1 × 1 × 9 with both concurrencies
  and max-candidates set to one; the full preset requires an explicit choice,
  retains 5 × 4 × 1600 / 20, and both show their pacing lower bound.
- Quick diagnostic projections and UI copy make no selection, champion,
  formal-best, validation, final-test, or promotion claim. No caller-trusted
  eligibility flag is sent; the server derives eligibility from numeric budgets.
- Moving-block draws contain only contiguous calendar blocks and never bridge
  a time gap.
- While pause/cancel is blocked in drain, another thread can acquire the global
  mutation lock and unrelated archive/create commands remain responsive.
- Full Python, Node, browser, source verification, and diff checks pass.
- Fresh cucumber and tomato one-origin/nine-cell diagnostics terminate with
  `diagnostic_smoke_completed_no_promotion`, use DSH-owned prediction tools,
  expose interpretable stage activity, and pass the native acceptance script.
