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

## Fix Round 1/5

### Review findings addressed

The first implementation bounded awaited child work, but its deadline began
after `pendingStarts.start()`, omitted post-operation checks around several
synchronous boundaries, detached cleanup even on the normal path, and could
only stop waiting for persistence rather than prevent a late durable commit.
This round closes all five Important findings:

1. The fixed monotonic deadline and fixed `deadline_unix_ms` are created before
   output-schema cloning and before any child-start work.
2. Every async phase is checked both before and after settlement. Synchronous
   and asynchronous admission results or errors that cross the boundary are
   normalized to `structured_role_operational_timeout`.
3. Result accessors, admission, persistence cloning/local preparation,
   persistence settlement, receipt accessors, final cloning, and response
   construction all have post-operation deadline checks. Persistence is never
   invoked when the pre-persistence clone crosses the deadline.
4. Persistence now has a two-layer fence. Node sends the fixed Host-generated
   wall deadline plus the abort signal and remaining transport timeout. Python
   converts the arrival wall-clock delta once to an injected monotonic deadline
   and enforces it again at the durable commit linearization point.
5. Final receipt access, clone, and return preparation cannot succeed after the
   deadline. Normal completion still awaits `dispose()` and pending-start
   bookkeeping in order; only an expired path detaches rejection-drained,
   bounded cleanup.

The sidecar transport timeout is clamped to its existing 600,000 ms hard limit.
This prevents the 1,800,000 ms research-stage deadline from being rejected as
an invalid transport timeout while leaving the envelope's absolute research
deadline unchanged.

### Durable fence and lock evidence

- `AdmissionFence` objects are stable and own a per-fence re-entrant lock; open,
  close, run-close, and structured acceptance no longer replace a fence object.
- Registry access is short and never calls the ledger while holding the
  registry lock.
- `EventLedger.append(..., commit_guard=...)` determines whether `INSERT OR
  IGNORE` inserted a new row. For a new row, the guard acquires the stable fence,
  rechecks open state and the converted monotonic deadline, and holds the fence
  through SQLite `COMMIT`. An exception rolls the transaction back.
- The resulting commit order is `ledger._lock -> fence.lock -> COMMIT`.
  Admission close takes only `fence.lock`, so close cannot overtake a guarded
  commit and no reverse ledger/fence order exists.
- An exact existing receipt, including the `inserted=False` concurrent path,
  replays without a second durable effect even after expiry. A conflicting
  payload still fails idempotency validation.
- The persisted accepted-event payload intentionally excludes
  `deadline_unix_ms`; the deadline is an authorization envelope field, so exact
  receipt replay remains stable.

### RED evidence

The expanded Node tests were run before the JavaScript fix:

```bash
node --test integrations/dsh_ecology_plugin/test/structured_roles.test.mjs \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
```

```text
tests 29
pass 21
fail 8
duration_ms 364.374167
```

The eight failures demonstrated: no fixed deadline in the stage-runner
envelope; synchronous child start escaping the deadline; admission false being
misclassified; persistence being called after a blocking clone; success after
the final clone and receipt accessor; normal cleanup settling before disposal;
and the late-success boundary returning instead of timing out. The independent
never-settling child-start and late-rejection controls already rejected rather
than hanging.

The initial server-side deadline tests were also RED with the correct Python
3.12 runtime:

```bash
PYTHONPATH=src /Users/jiezhou/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12 \
  -m unittest \
  tests.test_dsh_tool_contracts.DshToolServiceTests.test_structured_result_requires_a_runtime_deadline \
  tests.test_dsh_tool_contracts.DshToolServiceTests.test_expired_structured_result_is_rejected_before_arrival_work \
  tests.test_dsh_tool_contracts.DshToolServiceTests.test_structured_result_deadline_is_rechecked_inside_ledger_precommit \
  -v
```

```text
Ran 3 tests in 1.014s
FAILED (failures=3)
```

The missing deadline was accepted, expired envelopes were rejected only as an
unknown extra-field shape, and no precommit deadline guard was reached.

Two additional self-review regressions were proven RED before their fixes:

```text
test_concurrent_exact_receipt_replays_after_early_lookup_expires ... FAIL
actual: DshToolAdmissionClosedError('structured result deadline expired')
expected: the first exact durable receipt

native stage runner reserves before first child tool and durably persists structured output ... FAIL
assertion: transport timeout must be <= 600000 ms
```

These exposed, respectively, an exact-replay race after an early receipt miss
and the research timeout exceeding the sidecar client's existing hard limit.

### GREEN evidence

Final focused Node run:

```bash
node --test integrations/dsh_ecology_plugin/test/structured_roles.test.mjs \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
```

```text
tests 30
pass 30
fail 0
duration_ms 497.150542
```

Final complete plugin Node run:

```bash
node --test integrations/dsh_ecology_plugin/test/*.mjs
```

```text
tests 90
pass 90
fail 0
duration_ms 709.690417
```

Final related Python run:

```bash
PYTHONPATH=src /Users/jiezhou/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12 \
  -m unittest \
  tests.test_dsh_tool_contracts tests.test_dsh_cancel_race \
  tests.test_dsh_native_runtime tests.test_dsh_sample_execution \
  tests.test_core tests.test_dsh_structured_roles \
  tests.test_dsh_reconciliation tests.test_genome_replay -v
```

```text
Ran 98 tests in 11.718s
OK
```

Fresh static verification completed with exit code 0 and no diagnostics:

```bash
node --check integrations/dsh_ecology_plugin/lib/runtime/structured-roles.js
node --check integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js
node --check integrations/dsh_ecology_plugin/lib/sidecar/client.js
PYTHONPATH=src /Users/jiezhou/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12 \
  -m py_compile src/ecologyrsi_dsh/api/dsh_tools.py \
  src/ecologyrsi_dsh/core/ledger.py tests/test_dsh_tool_contracts.py \
  tests/test_dsh_cancel_race.py tests/test_dsh_native_runtime.py \
  tests/test_dsh_sample_execution.py
git diff --check
```

### Added deterministic coverage

- synchronous child start crossing a 5 ms deadline;
- child start ignoring abort and never settling;
- synchronous and asynchronous admission returning false or rejecting after
  crossing the deadline, all with stable timeout classification;
- persistence clone crossing the deadline with zero persistence calls;
- final return clone and receipt getter crossing the deadline;
- late child success and rejection without equal-timer ordering assumptions;
- normal bounded and unbounded paths retaining pending bookkeeping until
  disposal completes;
- a disposer that never settles still allowing bounded timeout cleanup;
- fixed pre-start `deadline_unix_ms`, signal, and bounded sidecar timeout
  propagation;
- missing and already-expired server deadlines;
- fake-monotonic expiry at ledger precommit with rollback/no accepted event;
- close blocked behind a commit holding the stable fence;
- sequential, early-lookup-racing, and `INSERT OR IGNORE` exact receipt replay
  after expiry, with exactly one accepted event.

### Self-review and concerns

- No known functional concern remains within the reviewed deadline and durable
  acceptance scope.
- Abort remains a transport/resource signal, not the correctness fence. The
  server-side deadline and commit guard prevent a non-cooperating in-flight
  request from creating a late durable event.
- A run without `timeoutMs` intentionally preserves the original unbounded
  cleanup semantics and awaits disposal/bookkeeping. Production structured
  stages supply a finite timeout; only those expired paths use detached cleanup.
- The trusted Node and Python sidecar run on the same host. Python uses wall
  time only once to translate the wire deadline, then uses monotonic time for
  all subsequent authorization checks.
- The complete plugin Node suite and 98 directly related Python tests were run.
  The unrelated full repository/browser suites were not run.
- No production service, live run, or external state was touched.
