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
- A pre-existing liveness boundary remains: if an already registered external
  child-start promise ignores abort forever, or an already active Workflow
  result ignores `cancel()` forever, the corresponding quiescence await can
  still be prolonged. The new schema-blocked launch race creates neither
  resource and returns promptly; changing the cancellation contract for
  already-started non-cooperating resources is outside Task 3's admission-fence
  scope.
