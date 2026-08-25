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

## Fix Round 2/5

### Review findings resolved

- Python durable restoration now transmits the ledger status and an explicit
  `{source, status}` restore provenance. The Node registry accepts `paused`
  only for the exact `runtime-restore:<run_id>` command whose provenance says
  `python_durable_ledger/paused`; ordinary `created` runs still cannot resume.
  A real `RuntimeController` plus `NativeStageRunner` reconciliation test proves
  that a restored paused run completes controller reconstruction with admission
  closed, rejects pre-resume stage work, and opens only after resume.
- Nonterminal controls now have explicit queued tokens and a visible
  `resuming` state. Exact concurrent resumes share the same Promise, a
  synchronous resume-then-pause executes the later pause, and an exact failed
  pause is permitted to re-drain. Rejection rollback distinguishes a failed
  preceding pause from a completed pause, preventing `pausing`/`resuming`
  wedges while preserving terminal epoch supremacy.
- `startRun()` is one exact-identity flight with a finalization token and an
  explicit role-host state (`creating`, `ready`, or `failed`). All role-host
  creations settle before the primary creation failure is chosen. Pause,
  cancel, and resume join start finalization and stale cleanup; no successful
  or failed late start may reopen admission before cleanup finishes.
- Start failure closes admission, marks hosts failed, best-effort disposes all
  retained hosts, and deletes an unchanged registry entry in `finally`,
  including a restored-paused entry whose reconstruction itself failed.
  A pause racing the failure retains the safe paused record, while resume is
  rejected with `runtime_role_hosts_incomplete`. Private cleanup failures do
  not replace the primary creation/setup error.
- `RoleAgentManager.quiesceRun()` repeatedly joins pending creations, rescans
  newly published handles, attempts idle/flush and disposal with all-settled
  semantics, and removes disposed registry handles even when one cleanup
  rejects. Role creation and persisted-role resume preserve their primary
  setup error when best-effort private disposal also fails.
- Workflow cancel/dispose memoization installs its in-flight sentinel before
  invoking the external cleanup function. Synchronous re-entry therefore
  observes and joins the same operation instead of invoking cleanup twice.

### Round 2 RED evidence

The initial deterministic controller, role-host, Workflow, and real-fence
repros were run together before production changes:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/agent_lifecycle.test.mjs \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs \
  integrations/dsh_ecology_plugin/test/launch_fence.test.mjs \
  integrations/dsh_ecology_plugin/test/workflow_lifecycle.test.mjs
```

```text
tests 37
pass 26
fail 11
```

The failures matched the reviewed defects: paused restore was rejected by the
real registry, ordinary control state had no queued `resuming` representation,
pause retry and identical resume single-flight were absent, start failure left
an open/active run or exposed private cleanup, pause could finish before a
pending start, concurrent starts duplicated work, and synchronous Workflow
cleanup re-entry invoked the external operation twice.

The two Python restoration assertions also failed before the handler change:

```bash
uv run --with pytest --frozen python -m pytest -q \
  tests/test_dsh_native_runtime.py \
  -k 'frozen_native_run_is_recreated_after_dsh_process_restart or resume_restores_a_paused_native_run_after_dsh_restart'
```

```text
2 failed
```

Two final adversarial TDD cycles caught additional atomicity edges after the
first GREEN pass. The first covered queued rollback:

```bash
node --test --test-name-pattern='queued resume rejected' \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs
```

```text
tests 1
pass 0
fail 1
actual final status: pausing
expected final status: paused
```

The resume rejection rollback now recognizes that the preceding pause drain
completed before host creation failed; the same repro passes and retains the
closed, durable `paused` state.

The second proved that an unsuperseded restored-paused start failure must delete
the incomplete reconstruction instead of leaving a durable-looking paused
record:

```bash
node --test --test-name-pattern='failed restored-paused' \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs
```

```text
tests 1
pass 0
fail 1
actual registry entry: paused
expected registry entry: null
```

The failure cleanup now deletes any unchanged start identity regardless of its
initial open/paused state, while lifecycle-epoch changes still preserve a
superseding pause or terminal cancel.

### Round 2 GREEN evidence

Focused control, role-host, real launch-fence, and Workflow lifecycle coverage:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/agent_lifecycle.test.mjs \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs \
  integrations/dsh_ecology_plugin/test/launch_fence.test.mjs \
  integrations/dsh_ecology_plugin/test/workflow_lifecycle.test.mjs
```

```text
tests 40
pass 40
fail 0
duration_ms 58.8695
```

Complete plugin Node suite, including the proxy security script:

```bash
node --test integrations/dsh_ecology_plugin/test/*.mjs
```

```text
tests 137
pass 137
fail 0
duration_ms 652.591458
```

Targeted Python restoration, reconciliation, and cancel-race coverage:

```bash
uv run --with pytest --frozen python -m pytest -o addopts='' -q \
  tests/test_dsh_native_runtime.py \
  tests/test_dsh_reconciliation.py \
  tests/test_dsh_cancel_race.py
```

```text
22 passed in 10.09s
```

All changed JavaScript files passed `node --check`; the changed Python handler
and test passed `python -m py_compile`; `git diff --check` exited 0 without
diagnostics.

### Round 2 self-review

- Restored paused startup closes the same provider/child admission fence as a
  live pause and never calls open during role-host reconstruction. The exact
  provenance/idempotency fence does not broaden resume from normal `created`.
- Control calls mutate intent and close admission synchronously. Queue tokens
  are ordered, exact duplicates join one Promise, rejected drains remain
  retryable only under the same pause/cancel binding, and terminal cancellation
  still wins through its separate epoch.
- Start success can open only after every role-host creation is fulfilled and
  only if its captured lifecycle epoch/status is unchanged. Both stale-success
  and failure paths join all pending publication and cleanup before controls
  may reopen; failed host sets are never resumable.
- Role-host cleanup uses no fail-fast aggregate. Idle, flush, and disposal are
  attempted for every published handle, while primary start/setup errors remain
  the public failure. Workflow cleanup retains the earlier hard-deadline and
  detached-late-cleanup behavior.
- Provider ordering/cooldown, stable child/Workflow drain, terminal cancel,
  lifecycle epochs, deadline finalization, reconciliation, and public error
  classification were exercised by the unchanged full suite.
- No Python service, database, production port, current run, or external state
  was touched.

### Round 2 concerns

- No known correctness race remains in the reviewed restore, nonterminal
  control, start failure, concurrent start, or cleanup memoization paths.
- As in Round 1, role-host quiescence joins the underlying Agent
  `waitForIdle()` contract without adding a new timeout; this round makes its
  multi-host cleanup complete and failure-atomic but does not change that Host
  API boundary.

## Fix Round 3/5

### Review findings resolved

- Live readiness is now derived from the complete role-host state of every
  nonterminal registry run instead of a sticky controller-wide bit. A durable
  restored-paused run therefore reports live Agent service readiness after all
  six role hosts are ready while its launch fence remains closed. Any active
  run whose hosts are creating, failed, or unknown makes the global capability
  fail closed, and an empty preset catalog cannot manufacture readiness through
  a vacuous successful start. Cancelling/cancelled tombstones are excluded;
  cancelling one of two ready runs retains readiness through the other, and
  cancelling the final ready run clears it.
- The real capability path is covered with the production-shaped six-preset
  catalog, exact tool surfaces, mountable standing keys, resolvable routes, and
  real `RoleAgentManager` creation. A loopback test fixture exposes the actual
  Node runtime routes. Python tests consume those routes through the real
  `DshNativeAgentRuntimeClient`, and the real Python HTTP handler is exercised
  across create, pause, Node-process restart, durable paused reconstruction,
  live-capability validation, resume, and status.
- Terminal start rejection now precedes exact single-flight/idempotency handling
  in `RuntimeController` and precedes idempotency-key comparison in
  `RuntimeRunRegistry`. Both `cancelling` and `cancelled` reject every start,
  including an exact retry whose raw start and control keys collide. The
  cancellation tombstone remains authoritative after a failed start clears its
  lifecycle start token.
- Ordinary `created` runs remain outside the restored-paused resume path; the
  existing regression still rejects that transition without opening admission.

### Round 3 RED evidence

The deterministic restored-paused and exact-key terminal retry repros failed
before the production changes:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/launch_fence.test.mjs \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs
```

```text
tests 30
pass 27
fail 3

registry terminal tombstones reject an exact same-key start
  Missing expected exception

terminal cancel dominates an exact same-key retry after failed start cleanup
  actual retry outcome: fulfilled
  expected: rejected

real controller reconciles a durable paused restore with admission closed until resume
  actual controller.liveReady: false
  expected: true
```

The real Python client rejected the real Node restored-paused capability at the
same production gate used before `runtime.resume`:

```bash
uv run --with pytest --frozen python -m pytest -o addopts='' -q \
  tests/test_dsh_native_runtime.py \
  -k 'python_resume_handshake_accepts_real_node_restored_paused_hosts'
```

```text
1 failed, 19 deselected
DshNativeRuntimeUnavailableError: dsh_native_runtime_not_ready
```

A mutation audit changed the multi-run readiness reduction from `every` to
`some`; the deterministic incomplete-second-host assertion failed with actual
`true` versus expected `false`. Restoring `every` returned the test to GREEN,
proving that one healthy run cannot mask another nonterminal incomplete run.
An additional empty-catalog RED failed with actual live readiness `true` versus
expected `false`; requiring a nonempty catalog closed that vacuous path.

### Round 3 GREEN evidence

Focused launch, controller, registry, capability, and cross-language coverage:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs \
  integrations/dsh_ecology_plugin/test/launch_fence.test.mjs
```

```text
tests 33
pass 33
fail 0
```

Complete plugin Node suite, including proxy security:

```bash
node --test integrations/dsh_ecology_plugin/test/*.mjs
```

```text
tests 142
pass 142
fail 0
duration_ms 698.09725
```

Targeted Python restoration, real Python/Node handshake, reconciliation, and
cancel-race coverage:

```bash
uv run --with pytest --frozen python -m pytest -o addopts='' -q \
  tests/test_dsh_native_runtime.py \
  tests/test_dsh_reconciliation.py \
  tests/test_dsh_cancel_race.py
```

```text
24 passed in 11.28s
```

All changed JavaScript files passed `node --check`; the changed Python test
passed `python -m py_compile`; `git diff --check` exited 0 without diagnostics.

### Round 3 self-review

- Host service readiness and launch admission are deliberately independent.
  A complete paused host set is live, but only resume opens its launch fence.
- Global readiness requires every nonterminal run to have a complete host set.
  This produces a conservative transient false while another run is still
  creating, then becomes true only after that start fully settles. It never
  derives readiness from a terminal tombstone or a different healthy run alone.
- Controller and registry terminal checks execute before exact-key handling.
  The failed-start/cancel repro blocks the first role creation, installs a
  same-key cancelling tombstone, releases the exact failure, waits for start
  token cleanup and terminal cancellation, then proves both the in-flight and
  post-cleanup retries reject without a second creation or fence open.
- No FIFO provider ordering, lifecycle epoch, single-flight, failure-atomic
  cleanup, deadline finalization, stable drain, durable reconciliation, or
  ordinary-created behavior was changed. The unchanged complete Node suite and
  relevant Python suite exercise those paths.
- No Python service, database, production port, current run, package install,
  or external state was touched.

### Round 3 concerns

- The intentionally fail-closed global capability can be temporarily false
  while a concurrent run is constructing its host set; Python checks it after
  its own synchronous create/restore completes, so this does not reopen the
  reviewed handshake race.
- The pre-existing role-host `waitForIdle()` boundary remains unbounded, as
  documented in Rounds 1 and 2; this round does not alter that Host API contract.

## Fix Round 4/5

### Review findings resolved

- Resume now requires controller-local role-host authority, not merely a shared
  registry row. A locally finalized host set must be exactly `ready`; an
  `unknown`, `failed`, or orphaned `creating` lifecycle rejects with
  `runtime_role_hosts_incomplete`, retains the durable `paused` record, and
  explicitly keeps launch admission closed. The only incomplete state allowed
  to enter the queued resume path is `creating` with that controller's live
  start token; after its finalization the same exact `ready` requirement is
  enforced before the registry can become `running` or the fence can open.
- Failed-start deletion is now a compare-and-delete against the exact frozen
  registry record returned to that start. A same-key `cancelling` or
  `cancelled` record written by another controller is a different authoritative
  generation and can never be deleted by the failing controller's unchanged
  local epoch. Retries remain rejected both while cancellation is draining and
  after its terminal tombstone is installed.
- The two deterministic regressions use two controllers and one injected
  `RuntimeRunRegistry`. The resume repro first establishes a real finalized
  restored-paused lifecycle on the owning controller, then proves a second
  controller with unknown/orphaned-creating hosts cannot borrow that readiness.
  The cleanup repro blocks the first role creation, installs an external
  same-key cancellation, releases the primary failure, and proves no retry can
  create a second host or open admission.
- Existing control-order tests now establish their ready lifecycle through the
  real `startRun()` path instead of treating a test-injected registry record as
  evidence that role hosts exist. Their pause/resume, exact single-flight,
  close-before-drain, and FIFO assertions are otherwise unchanged.

### Round 4 RED evidence

Both shared-registry repros failed deterministically before the controller
change:

```bash
node --test \
  --test-name-pattern='second controller cannot resume|another controller.s same-key cancel' \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs
```

```text
tests 2
pass 0
fail 2

a second controller cannot resume registry-only paused hosts
  actual: fulfilled
  expected: rejected

failed start cleanup preserves another controller's same-key cancel tombstone
  actual retry outcome: fulfilled
  expected: rejected
```

The first failure proved that a registry-only paused row could become running
and open the second controller's fence without local hosts. The second proved
that matching a local epoch plus idempotency key allowed failed-start cleanup
to erase an externally written cancelling record and admit a new start.

### Round 4 GREEN evidence

The two exact shared-registry regressions passed:

```bash
node --test \
  --test-name-pattern='second controller cannot resume|another controller.s same-key cancel' \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs
```

```text
tests 2
pass 2
fail 0
```

Focused controller, registry, launch-fence, restored-handshake, live-readiness,
and control-order coverage passed:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs \
  integrations/dsh_ecology_plugin/test/launch_fence.test.mjs
```

```text
tests 35
pass 35
fail 0
```

The complete plugin Node suite, including proxy security, passed:

```bash
node --test integrations/dsh_ecology_plugin/test/*.mjs
```

```text
tests 144
pass 144
fail 0
```

The relevant Python restoration, real Python/Node handshake, reconciliation,
and cancel-race suite passed:

```bash
uv run --with pytest --frozen python -m pytest -o addopts='' -q \
  tests/test_dsh_native_runtime.py \
  tests/test_dsh_reconciliation.py \
  tests/test_dsh_cancel_race.py
```

```text
24 passed
```

The changed JavaScript files passed `node --check`, and `git diff --check`
exited 0 without diagnostics.

### Round 4 self-review

- A registry status is durable control intent; it is not proof that the current
  controller owns live role hosts. The resume gate therefore reads only its
  own lifecycle and fails closed before changing a registry-only paused row.
- A genuinely live start remains resumable while creation is pending: resume
  joins its finalization and opens only after `hosts === "ready"`. Existing
  success, failure, stale cleanup, and restored-paused handshake tests cover
  each branch, including the real six-preset Python/Node path.
- Frozen registry object identity acts as the cleanup generation token. Every
  transition produces a new object, so external cancelling/cancelled state,
  even with the same raw idempotency key, supersedes the failed start and is
  preserved without relying on another controller's private epoch.
- Terminal start rejection remains before local single-flight and registry
  idempotency handling. The new race proves both cancelling and cancelled
  retries reject, while unchanged ordinary failed starts still delete their
  own untouched record.
- No provider FIFO/cooldown, lifecycle epoch ordering, live-readiness
  reduction, stable child/Workflow drain, deadline finalization, durable
  reconciliation, or ordinary-created behavior was changed. The full Node and
  relevant Python suites exercise those adjacent paths.
- No Python service, database, production port, current run, package install,
  or external state was touched.

### Round 4 concerns

- No known correctness gap remains in the reviewed registry-authority resume
  or cross-controller failed-start cleanup paths.
- The pre-existing role-host `waitForIdle()` boundary remains unbounded, as
  documented in Rounds 1 through 3; this round does not alter that Host API
  contract.

## Fix Round 5/5

### Review finding resolved

- Every registry-created run generation now has an opaque, registry-owned
  identity held in private `WeakMap` metadata. It cannot be selected through a
  request field or reconstructed by a caller. `transition()` and `refresh()`
  publish new immutable records with the same identity, while a start after an
  actual deletion receives a fresh identity.
- The same private metadata retains an immutable clone of the complete original
  start payload. An active same-key request with any changed top-level or nested
  field now rejects with `runtime_start_conflict` and leaves the current record
  untouched. An exact replay returns the current record in the same generation;
  a foreign controller returns only an idempotent receipt and neither creates
  duplicate role hosts nor claims live readiness. If controls or stages changed
  the current row's receipt, the replay response still echoes the original start
  idempotency key required by the Python client without rolling registry state
  back.
- Controller host lifecycle state now carries `hostGeneration`, and every live
  start token carries its accepted registry generation. Global live readiness,
  stage admission before and after the external stage, resume admission before
  and after start finalization, and both start/resume fence-open paths require
  `hosts === "ready"` and exact equality with the current registry generation.
- Successful role creation checks generation ownership before setting `ready`.
  A superseded completion disposes/quiesces its private hosts and never opens
  admission. The same controller also rejects claiming a replacement generation
  until its older start token has finalized, preventing run-id cleanup from
  crossing two local host generations.
- Failed-start deletion remains the Round 4 exact-record compare-and-delete.
  Generation identity does not weaken that CAS: a transition or refresh in the
  same generation creates a different exact record, so cancelling/cancelled
  tombstones and other authoritative mutations remain undeletable by stale
  cleanup.

### Round 5 RED evidence

The initial registry and natural two-controller generation suite failed every
new case before implementation:

```bash
node --test \
  --test-name-pattern='registry exact start replay|registry rejects a changed|genuine replacement|exact shared-registry|changed same-key|stale ready hosts' \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs
```

```text
tests 7
pass 0
fail 7

generationOf: TypeError (three registry/generation cases)
changed same-key registry start: Missing expected exception
changed same-key controller start: actual fulfilled; expected rejected
stale-ready replacement resumes: actual fulfilled; expected rejected (two cases)
```

The start-open and stage-admission mutations also both reproduced:

```bash
node --test \
  --test-name-pattern='superseded by a new generation|stale ready host generation rejects stage admission' \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs
```

```text
tests 2
pass 0
fail 2

superseded start: no generation API and would otherwise open
stale stage admission: actual fulfilled; expected rejected
```

Two self-review TDD cycles then caught exact-replay and local-token edge cases.
The replay after a pause transition returned `pause-after-replayed-start-1`
instead of the requested `runtime-restore:run-1`; the unfinished old local
start accepted a replacement replay as fulfilled instead of rejecting it.
Each one-test command failed 0/1 for that exact reason before its minimal fix.

### Round 5 GREEN evidence

All new registry, two-controller success/failure, exact-replay, stale-resume,
superseded-start, and stale-stage cases passed together:

```bash
node --test \
  --test-name-pattern='registry exact start replay|registry rejects a changed|genuine replacement|exact shared-registry|changed same-key|stale ready hosts|superseded by a new generation|stale ready host generation rejects stage admission' \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs
```

```text
tests 9
pass 9
fail 0
```

The complete controller race file passed 39/39, and focused controller plus
launch-fence coverage passed:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/cancel_race.test.mjs \
  integrations/dsh_ecology_plugin/test/launch_fence.test.mjs
```

```text
tests 44
pass 44
fail 0
```

The complete plugin Node suite, including proxy security, passed:

```bash
node --test integrations/dsh_ecology_plugin/test/*.mjs
```

```text
tests 153
pass 153
fail 0
duration_ms 663.536625
```

The relevant Python restoration, real Python/Node handshake, reconciliation,
and cancel-race suite passed:

```bash
uv run --with pytest --frozen python -m pytest -o addopts='' -q \
  tests/test_dsh_native_runtime.py \
  tests/test_dsh_reconciliation.py \
  tests/test_dsh_cancel_race.py
```

```text
24 passed in 11.27s
```

Both changed production JavaScript files and the changed race test passed
`node --check`; `git diff --check` exited 0 without diagnostics.

### Round 5 self-review

- Mutation coverage is direct: changing generation preservation, accepting a
  changed nested model binding, reusing an old generation, treating stale hosts
  as ready, opening before the generation check, admitting a stale stage,
  deleting a transitioned tombstone, duplicating exact-replay hosts, or returning
  the wrong replay key fails at least one deterministic regression.
- The natural shared-registry races establish an actually ready restored-paused
  owner, delete only to model a genuine replacement, block the replacement's
  role creation, and attempt the old owner's resume while the new generation is
  `creating`. Both replacement success and failure keep every inappropriate
  fence closed; success leaves the new paused generation ready only on its owner,
  while ordinary untouched failure still deletes only its own exact record.
- A changed same-key attempt never reaches role creation and leaves the original
  paused row and old ready hosts authoritative. An exact replay is intentionally
  weaker than host ownership: it acknowledges the already accepted command but
  leaves a foreign controller `liveReady === false`.
- Generation identity is process-local control metadata, deliberately absent
  from JSON responses and caller bindings. The durable business identity remains
  the Python-owned frozen payload; a Node process restart constructs a fresh
  registry and a fresh generation while the existing real restore handshake
  recreates and verifies all six role hosts.
- No provider FIFO/cooldown, control ordering, stable child/Workflow drain,
  structured-role deadline, durable reconciliation, or terminal transition
  behavior was broadened. All 153 Node tests and the 24 relevant Python tests
  exercise those adjacent invariants.
- No Python service, database, production port, current run, package install,
  or external state was touched.

### Round 5 concerns

- No known correctness gap remains in the reviewed generation-bound host
  readiness, replay conflict, stale resume, stage admission, or fence-open paths.
- The pre-existing role-host `waitForIdle()` boundary remains unbounded, as
  documented in Rounds 1 through 4; this round does not alter that Host API
  contract.
