# Task 3 report — pause/cancel launch fence

## What changed

- Added a persistent synchronous per-run launch fence to the provider-stage
  gate and the pending-launch registry. Closing a run also aborts its queued
  provider records; reopening is explicit.
- Direct, continuable, and Workflow starts now perform the same synchronous
  admission check immediately before registering work. The pending record and
  its settlement promise exist before any external starter is invoked.
- Workflow launch uses a pending placeholder before `workflowEngine.start()`
  and publishes the returned handle to the active set without an intervening
  await. Re-entrant pause/cancel therefore observes and cancels the launch.
- Provider and pending-start drains loop until the closed run has a stable
  empty record set. Quiescence keeps child/Workflow cancellation ahead of the
  provider drain, so a provider operation waiting on child settlement does not
  create a lock-order cycle.
- Pause/cancel transition runtime status and close the launch fence
  synchronously before the first await. Per-run control drains are serialized,
  so a queued cancel cannot be overwritten by an older pause.
- Resume waits for the complete pause drain before reopening launch admission.
  A completed cancel is terminal and cannot reopen its fence.

## Files

- `integrations/dsh_ecology_plugin/lib/runtime/controller.js`
- `integrations/dsh_ecology_plugin/lib/runtime/provider-stage-gate.js`
- `integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js`
- `integrations/dsh_ecology_plugin/lib/runtime/workflows.js`
- `integrations/dsh_ecology_plugin/test/cancel_race.test.mjs`
- `integrations/dsh_ecology_plugin/test/launch_fence.test.mjs`
- `integrations/dsh_ecology_plugin/test/provider_stage_gate.test.mjs`
- `integrations/dsh_ecology_plugin/test/workflow_lifecycle.test.mjs`
- `.superpowers/sdd/2026-08-25-evolution-safety-followup/task-3-report.md`

## RED evidence

The first deterministic RED run used explicit schema and control-drain
barriers:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/launch_fence.test.mjs \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs \
  integrations/dsh_ecology_plugin/test/provider_stage_gate.test.mjs \
  integrations/dsh_ecology_plugin/test/workflow_lifecycle.test.mjs
```

```text
tests 18
pass 12
fail 6
duration_ms 58.082666
```

The six failures matched the production defects:

```text
resume waits for the pause drain before reopening launch admission
  actual: resume settled before the explicit drain release

cancelled runs cannot reopen their launch fence
  Missing expected rejection

pause closes the fence before a schema-blocked direct child can launch
  actual childStarts: 1
  expected childStarts: 0

cancel closes the fence before a schema-blocked Workflow can launch
  actual workflowStarts: 1
  expected workflowStarts: 0

a closed provider launch fence rejects future work until explicitly reopened
  TypeError: gate.closeRun is not a function

reentrant drain observes pending work before the external starter returns
  actual pending size: 0
  expected pending size: 1
```

A second TDD cycle covered the Workflow publication placeholder and overlapping
control operations:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs \
  integrations/dsh_ecology_plugin/test/workflow_lifecycle.test.mjs
```

```text
tests 13
pass 11
fail 2
duration_ms 48.195292
```

```text
cancel queued during pause drain cannot be overwritten by the older control
  actual drains before pause release: 2
  expected: 1

Workflow launch is registered before engine start can reenter drain
  TypeError: starts.startWorkflow is not a function
```

## GREEN evidence

Focused launch, control, stage-runner, and structured-role tests:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/launch_fence.test.mjs \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs \
  integrations/dsh_ecology_plugin/test/provider_stage_gate.test.mjs \
  integrations/dsh_ecology_plugin/test/workflow_lifecycle.test.mjs \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs \
  integrations/dsh_ecology_plugin/test/structured_roles.test.mjs
```

```text
tests 62
pass 62
fail 0
duration_ms 492.54
```

Complete plugin Node suite, including the proxy security script:

```bash
node --test integrations/dsh_ecology_plugin/test/*.mjs
```

```text
tests 112
pass 112
fail 0
duration_ms 641.652084
```

The following also completed with exit code 0 and no diagnostics:

```bash
node --check integrations/dsh_ecology_plugin/lib/runtime/controller.js
node --check integrations/dsh_ecology_plugin/lib/runtime/provider-stage-gate.js
node --check integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js
node --check integrations/dsh_ecology_plugin/lib/runtime/workflows.js
node --check integrations/dsh_ecology_plugin/test/cancel_race.test.mjs
node --check integrations/dsh_ecology_plugin/test/provider_stage_gate.test.mjs
node --check integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
node --check integrations/dsh_ecology_plugin/test/workflow_lifecycle.test.mjs
node --check integrations/dsh_ecology_plugin/test/launch_fence.test.mjs
git diff --check
```

## Self-review

- The two production launch paths are direct one-shot child creation through
  `PendingChildStarts.start()` and sample-planner Workflow creation through
  `startHomogeneousWorkflow()`. The generic continuable branch is not called by
  production today, but it uses the same fence and has an explicit closed-run
  regression.
- Sidecar reservation and schema resolution remain preparatory awaits. After
  schema returns, both launch paths reach the common synchronous fence check
  and pending registration before their external start call. Closed admission
  preserves the stable `provider_stage_admission_closed` code and is not a
  retryable model failure.
- The local pending-launch fence is authoritative even when tests inject a
  simplified provider gate. The production provider gate also rejects new
  records early and rechecks after its own queue/cooldown awaits.
- No mutex or fence is held across schema, child result, Workflow result,
  disposal, or provider-drain awaits. Quiescence closes the fence first, then
  cancels active/pending work, then drains provider records, and only then
  revokes child bindings.
- The barrier race gives the structured stage a 60-second watchdog but the
  pause/cancel tests finish in milliseconds after schema release. Thus control
  completion is caused by the fence rejection, not by stage timeout.
- Pause completion reopens only through an explicit resume after drain. A
  pause followed by a queued cancel stays closed and ends cancelled; durable
  Workflow reconciliation still starts only fresh remaining work and never
  revives an old Workflow.
- The complete plugin Node suite passed, and no Python service, database,
  production port, current run, or external state was touched.

## Concerns

- No known launch-admission race remains in the audited child, continuable, or
  Workflow entry points.
- The initial submission left a liveness boundary when an already registered
  child start or active Workflow ignored cancellation. Fix Round 1 below closes
  that reviewed gap by joining owner finalization rather than the raw external
  promise.

## Fix Round 1/5

### Review findings resolved

- Added an explicit per-run control transition matrix and lifecycle epoch.
  Pause and resume cannot overwrite `cancelling` or `cancelled`; a terminal
  cancel advances intent synchronously, runs even when the preceding drain
  rejected, and can be retried idempotently after a failed drain. Resume is a
  queued lifecycle operation, so a later terminal cancel supersedes it without
  an admission-open interval.
- `startRun()` now captures the lifecycle epoch before awaiting role-agent
  creation. Only the unchanged `created`/`running` epoch may open admission.
  A completion made stale by pause or cancel quiesces the newly published
  handles, disposing them for terminal cancellation, and a stale creation
  failure no longer deletes the terminal registry state.
- Pending one-shot and Workflow launches now expose a lifecycle-finalization
  settlement in addition to their raw external start/result promises. Task 2's
  hard-deadline `finish()` releases a control drain even when start, result, or
  disposal ignores abort; late result and cleanup failures stay detached and
  rejection-drained. Workflow cancel and disposal are memoized for cleanup
  idempotency.
- Active Workflow entries and their pending placeholder share one lifecycle.
  Runner quiescence repeatedly scans both sets before provider drain and again
  after it, covering the synchronous `workflowEngine.start()` re-entrant
  publication gap without awaiting an unbounded raw Workflow result.

### Round 1 RED evidence

Controller transition and stale-start repros, before implementation:

```bash
node --test integrations/dsh_ecology_plugin/test/cancel_race.test.mjs
```

```text
tests 14
pass 9
fail 5

terminal cancel rejects a queued pause and resume without reopening admission
  actual outcomes: fulfilled, fulfilled, fulfilled

queued terminal cancel still drains after a prior pause drain rejects
  actual outcomes: rejected, rejected

terminal cancel supersedes an already queued resume without an open window
  actual resume outcome: fulfilled

stale startRun completion after pause/cancel cannot reopen and is quiesced
  actual final call: open
```

The child finalization repro failed with the exact leaked drain:

```bash
node --test --test-name-pattern='hard deadline finalization releases' \
  integrations/dsh_ecology_plugin/test/structured_roles.test.mjs
```

```text
tests 1
pass 0
fail 1
actual drainOutcome: still-pending
expected: drained
```

The Workflow hard-deadline and re-entrant publication repros also both failed:

```bash
node --test --test-name-pattern='^sample planner hard deadline releases' \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
node --test --test-name-pattern='^reentrant Workflow quiescence' \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
```

```text
tests 1; pass 0; fail 1; actual drainOutcome: still-pending
tests 1; pass 0; fail 1; re-entrant cleanup was not one lifecycle
```

Thus all eight newly encoded reviewer/adversarial cases were observed failing
for the intended production reasons before the runtime changes.

### Round 1 GREEN evidence

Focused controller, launch-fence, provider, pending-start, Workflow, runner,
and Task 2 deadline coverage:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/launch_fence.test.mjs \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs \
  integrations/dsh_ecology_plugin/test/provider_stage_gate.test.mjs \
  integrations/dsh_ecology_plugin/test/workflow_lifecycle.test.mjs \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs \
  integrations/dsh_ecology_plugin/test/structured_roles.test.mjs
```

```text
tests 74
pass 74
fail 0
duration_ms 569.067584
```

Complete plugin Node suite, including proxy security:

```bash
node --test integrations/dsh_ecology_plugin/test/*.mjs
```

```text
tests 124
pass 124
fail 0
duration_ms 661.338667
```

All changed production and test JavaScript files also passed `node --check`.
`git diff --check` completed with exit code 0 and no diagnostics.

### Round 1 self-review

- Close-before-await remains intact: pause/cancel mutate intent and close the
  fence synchronously; their first asynchronous step is the queued drain.
- Direct, continuable, and Workflow starts still perform synchronous admission
  and pending registration before invoking the external starter. No await was
  inserted in that atomic admission/publication boundary.
- Child/Workflow cancellation still precedes provider-record draining, and a
  second stable lifecycle scan follows provider drain. This preserves the
  prior lock-order ruling while covering re-entrant publication.
- Task 2 keeps one absolute deadline and the same operational-timeout code.
  The finalization signal changes only what control drain joins; it does not
  admit a late result, extend a deadline, or surface a private cleanup error.
- Workflow cancel/dispose and one-shot dispose are single-flight, so the stage
  owner and concurrent control drain can safely request the same cleanup.
- Durable Workflow reconciliation, provider pacing/cooldown, child-binding
  revocation, and structured error classifications were not changed.
- No Python service, database, production port, current run, or external state
  was touched.

### Round 1 concerns

- Role-host quiescence still relies on the underlying retained Agent's
  `waitForIdle()` contract. This round bounds structured child and Workflow
  launch lifecycles; it does not add a new timeout around the role-host manager.
