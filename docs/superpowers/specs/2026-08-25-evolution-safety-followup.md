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

The review also found that the legacy promotion helper advertises a paired
moving-block bootstrap while sampling independent single-day blocks in hash
order. The active DSH-native diagnostic path uses a different, correct
contiguous-block selector, but the mislabeled helper must be corrected before
it is reused for a promotion decision.

## Goals

- Reject affirmative eligibility, gate, selection, or promotion assertions
  even when they follow a negated clause.
- Make every structured-role timeout a hard end-to-end deadline covering child
  creation, child result settlement, validation, and persistence admission.
- Close a run's child-launch fence synchronously before pause/cancel drains
  outstanding work.
- Keep diagnostic runs structurally unable to reserve or pass a formal stage.
- Do not allow a formal pass until a host-derived canonical formal assessment
  and its frozen UQ bindings are integrated.
- Make the legacy promotion bootstrap operate on ordered, contiguous calendar
  blocks and report the method it actually executes.
- Rebuild the isolated runtime and verify new cucumber and tomato diagnostic
  runs through terminal acceptance.

## Non-goals

- This change does not expose validation or final-test rows.
- It does not claim that the full formal UQ pipeline is integrated.
- It does not add complexity/identifiability gates or redesign the greenhouse
  predictor family.
- It does not compare scores across datasets or frozen cohorts.
- It does not add synthetic progress percentages or billing-token claims.

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

### 4. Formal stage fail-closed boundary

Formal reservation requires a selection-eligible budget and a legitimate
locked selection incumbent. Diagnostic smoke runs are rejected before any
formal exposure is reserved. The current generic mapping evaluator cannot
produce a passing formal result. Until a host-owned evaluator, frozen baseline
UQ artifact, and canonical `FormalFitnessAssessment` are wired together, a
formal pass remains explicitly unavailable. Event replay also refuses a
`passed` completion that lacks the canonical assessment bindings.

### 5. Ordered moving-block bootstrap

Validated promotion evidence retains a unique integer
`origin_block_index`. Candidate and incumbent indices must match exactly.
Bootstrap draws use non-circular contiguous three-day blocks, never cross a
calendar gap, and truncate to the paired cohort size. Evidence with too few
paired days or legal starts fails closed. Returned metadata records block
length and legal-start count in addition to the versioned method.

## Acceptance

- All new regression tests demonstrate RED before implementation and GREEN
  after implementation.
- Late structured-role success persists zero results; late rejection and an
  abort-ignoring child both return the operational-timeout classification.
- A schema-blocked stage released after pause starts zero children and the
  control operation returns without waiting for a stage timeout.
- Diagnostic formal reservation and `lambda: {"outcome": "passed"}` are
  rejected without setting validated/final-test candidate state.
- Moving-block draws contain only contiguous calendar blocks and never bridge
  a time gap.
- Full Python, Node, browser, source verification, and diff checks pass.
- Fresh cucumber and tomato one-origin/nine-cell diagnostics terminate with
  `diagnostic_smoke_completed_no_promotion`, use DSH-owned prediction tools,
  expose interpretable stage activity, and pass the native acceptance script.

