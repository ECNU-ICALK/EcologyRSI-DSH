# Task 2 report — hard structured-role deadline

## What changed

- Added one monotonic absolute deadline for child start, child result settlement,
  admission, and persistence settlement. Every awaited phase races the same
  deadline rather than receiving a fresh per-phase timeout.
- Added the stable `structured_role_operational_timeout` error code with the
  existing `structured role operational timeout` public message. A result that
  rejects after expiry cannot leak as `structured_child_result_failed` or
  expose the child failure.
- Deadline expiry aborts the child, but completion does not depend on the child,
  result promise, or disposer honoring abort.
- Late result success/rejection remains observed by the race without entering
  admission or persistence. A child start that itself arrives after timeout is
  also drained and disposed in the background.
- Disposal and pending-start finish are invoked with rejection handlers but are
  not awaited, so cleanup cannot turn the hard deadline into another unbounded
  wait.

## Files

- `integrations/dsh_ecology_plugin/lib/runtime/structured-roles.js`
- `integrations/dsh_ecology_plugin/test/structured_roles.test.mjs`
- `.superpowers/sdd/2026-08-25-evolution-safety-followup/task-2-report.md`

No `stage-runner.js` production change was needed because it already supplies
the stage timeout to `runStructuredRole`; its module tests exercise that seam.

## RED evidence

Command, after adding the three regressions and before changing production
code:

```bash
node --test integrations/dsh_ecology_plugin/test/structured_roles.test.mjs
```

Result:

```text
tests 8
pass 5
fail 3
duration_ms 420.235709
```

The failures matched the three production defects:

```text
structured role rejects a result that succeeds after its operational deadline
  AssertionError: Missing expected rejection.

structured role classifies a result rejection after the deadline as an operational timeout
  Caught error: Error: structured_child_result_failed

structured role deadline does not wait for abort-ignoring result or disposal
  actual: 'watchdog'
  expected: 'rejected'
```

Thus the late success was admitted, the late rejection leaked the result-phase
classification, and the abort-ignoring result remained pending past the
250-millisecond test watchdog.

## GREEN evidence

Focused structured-role command:

```bash
node --test integrations/dsh_ecology_plugin/test/structured_roles.test.mjs
```

Result:

```text
tests 8
pass 8
fail 0
duration_ms 195.434958
```

Related stage-runner module command:

```bash
node --test integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
```

Result:

```text
tests 14
pass 14
fail 0
duration_ms 83.082208
```

Complete DSH ecology plugin Node test command:

```bash
node --test integrations/dsh_ecology_plugin/test/*.mjs
```

Result:

```text
tests 82
pass 82
fail 0
duration_ms 616.439583
```

`node --check integrations/dsh_ecology_plugin/lib/runtime/structured-roles.js`
and `git diff --check` also completed with exit code 0 and no output.

## Self-review

- The late-success test deliberately spends part of the timeout in child start
  and the rest in result settlement. Both individual phases finish faster than
  `timeoutMs`, while their combined duration exceeds it, so the test rejects an
  incorrect implementation that restarts a full timeout for the result phase.
- The same deadline promise is raced against every asynchronous lifecycle
  phase. Monotonic `performance.now()` checks close gaps before validation,
  admission, persistence, and return.
- Deadline classification is independent of which losing child promise later
  settles. `Promise.race` and explicit background observers prevent late
  rejections from becoming unhandled rejections.
- Admission and persistence counters stay at zero for a late successful result;
  the abort-ignoring test also proves pending-start bookkeeping is finished and
  disposal cannot delay the returned timeout.
- Existing pre-deadline start failure, result failure, missing-result,
  admission-closed, persistence, disposal, and stage-runner behavior remains
  covered by the focused and plugin-wide regression runs.

## Concerns

- No known functional concerns.
- Cleanup is intentionally best-effort after the structured-role outcome: it is
  invoked and rejection-drained, but a non-cooperating disposer cannot be
  awaited without violating the hard deadline.
- The complete plugin Node suite was run; the full repository Python/browser
  suites were outside this focused JavaScript lifecycle task and were not run.
