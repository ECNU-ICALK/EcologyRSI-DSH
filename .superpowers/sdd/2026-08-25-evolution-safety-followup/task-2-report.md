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

## Fix Round 2/5

### Review findings addressed

This round replaces the wall-clock authorization bridge with a process-local,
one-shot server arm and applies the same absolute lifecycle deadline to the
`sample.plan` Workflow path.

1. `DshStructuredRoleRuntime` now opens a stage fence with its exact
   run/revision/attempt/role/stage/idempotency binding and passes the generated
   `admission_id` to Node. Before any child reservation, Node sends that exact
   admission plus the selected role timeout. Python freezes
   `monotonic_now + timeout_ms` once under the stable fence lock; subsequent
   arms must use the same admission and timeout and can never extend it.
2. Wall time and `deadline_unix_ms` have been removed from structured-result
   authorization. The protocol accepts only integer timeouts from 1 through
   1,800,000 ms. Node config, runtime construction, and Python reservation
   validation all enforce the same ceiling before a platform timer is created.
3. Normal one-shot and Workflow cleanup awaits disposal and bookkeeping. A
   rejected disposer or finisher is drained and cannot replace a successful
   result or the primary structured phase error. Expired cleanup remains
   detached and rejection-observed so a non-cooperating object cannot defeat
   the hard deadline.
4. Python sidecar errors now include a sanitized `error_code`. The Node
   transport allowlists `structured_role_operational_timeout`, so server expiry
   and admission-arm mismatch cannot degrade to `sidecar_rejected` or disclose
   a private error string.
5. `sample.plan` now races Workflow result, admission, persistence, and normal
   cleanup against the same absolute deadline created after the first arm
   receipt. It checks synchronous clone/receipt/return boundaries, aborts and
   cancels on expiry, detaches expired disposal, retains active Workflow
   bookkeeping until normal disposal completes, and reuses the same deadline
   across retries. Schema loading, provider-gate retry waits, and the final
   stage response digest are bounded by that deadline as well.

### Durable admission and lock evidence

- `open_admission` creates one random `admission_id`; the stable fence is keyed
  by run/revision/attempt and binds role, stage, and idempotency key. A process
  restart has no such fence, so an old wire admission fails closed. A fresh Host
  invocation can open a new fence and still replay an already durable exact
  structured result through the existing replay path.
- `/child-reservations` has an exact request shape containing
  `run_state_revision`, `stage_attempt`, `admission_id`, and `timeout_ms`.
  Its durable event stores a request-contract digest for exact request-ID
  replay while retaining the admission-independent business digest, so launch
  attempts remain monotonic after a Host restart. Projection accepts legacy
  events without the new digest but validates it when present.
- Arming holds only `fence.lock` and never calls the ledger. Reservation and
  structured-result commits acquire `ledger._lock` first, then their stable
  fence through `commit_guard(inserted=True)`, and hold it through SQLite
  `COMMIT`. `close_admission` and `close_run_admissions` use the same fence lock,
  so neither close path can overtake an accepted commit.
- An `inserted=False` exact duplicate skips the new-row guard and replays even
  after expiry. Sequential, early-lookup-racing, and `INSERT OR IGNORE` exact
  replays are covered; a conflicting payload after expiry still raises the
  idempotency conflict and never creates another event.
- Accepted structured-event payloads contain no admission or deadline
  metadata. The admission is an authorization envelope, not scientific state.

### RED evidence

Timeout ceiling coverage was added before the implementation:

```bash
node --test integrations/dsh_ecology_plugin/test/config.test.mjs \
  integrations/dsh_ecology_plugin/test/structured_roles.test.mjs \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
```

```text
tests 36
pass 33
fail 3
```

The three failures included a Node `TimeoutOverflowWarning` and proved that
config, direct structured roles, and runtime construction accepted values above
the protocol/platform bound.

Two cleanup tests were RED because a private `finish` rejection replaced both
the successful result and the primary `structured_result_missing` error. After
the cleanup fix, the selected cleanup/bookkeeping group passed 4/4.

Early-arm protocol tests were RED independently at each boundary:

- the Node reservation omitted revision, attempt, admission, and timeout;
- the Python runtime request had no server-issued admission ID;
- four Python tests could not arm a frozen monotonic deadline, reject an
  over-ceiling timeout, or reject a mismatched admission.

The HTTP/Sidecar error-code tests were run before their transport changes:

```text
Node:   tests 1, pass 0, fail 1 (received sidecar_rejected)
Python: Ran 1 test, FAILED (response had no error_code)
```

The six focused Workflow deadline tests were all RED before the Workflow
refactor:

```text
tests 6
pass 0
fail 6
duration_ms 478.239458
```

Both a cancel-ignoring Workflow result and abort-ignoring persistence reached
the 160 ms outer guard instead of the 20 ms stage deadline. Admission crossing
the boundary was misclassified, a clone crossing returned an unstable error,
the final clone still returned success, and normal disposal removed active
bookkeeping early and leaked its private rejection.

The first related Python migration run intentionally exposed every old wire
fixture:

```text
Ran 80 tests
pass 62
fail 7
error 11
```

All failures were then migrated to the strict admission/arm protocol. One
remaining cancellation projection failure identified the new reservation
contract digest as an unrecognized event field; the projection was updated
with strict new-field validation plus legacy-event compatibility.

### GREEN evidence

Focused Workflow deadline run after the implementation:

```text
tests 6
pass 6
fail 0
duration_ms 342.700458
```

The additional cross-retry test confirms that both reservations send the same
frozen server timeout while the second transport request receives only the
remaining local absolute budget.

Complete plugin Node run:

```bash
node --test integrations/dsh_ecology_plugin/test/*.mjs
```

```text
tests 103
pass 103
fail 0
duration_ms 659.201875
```

Related Python regression run:

```bash
PYTHONPATH=src /Users/jiezhou/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12 \
  -m unittest tests.test_dsh_tool_contracts tests.test_dsh_cancel_race \
  tests.test_dsh_native_runtime tests.test_dsh_sample_execution \
  tests.test_core tests.test_dsh_structured_roles \
  tests.test_dsh_reconciliation tests.test_genome_replay -v
```

```text
Ran 106 tests in 12.201s
OK
```

Fresh syntax and whitespace verification completed without diagnostics:

```bash
node --check integrations/dsh_ecology_plugin/lib/runtime/structured-deadline.js
node --check integrations/dsh_ecology_plugin/lib/runtime/routes.js
node --check integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js
node --check integrations/dsh_ecology_plugin/lib/runtime/structured-roles.js
node --check integrations/dsh_ecology_plugin/lib/sidecar/client.js
PYTHONPATH=src /Users/jiezhou/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12 \
  -m py_compile src/ecologyrsi_dsh/api/dsh_tools.py \
  src/ecologyrsi_dsh/api/handler.py src/ecologyrsi_dsh/core/state.py \
  src/ecologyrsi_dsh/integrations/dsh_native_runtime.py \
  src/ecologyrsi_dsh/integrations/dsh_structured_roles.py \
  tests/test_dsh_tool_contracts.py tests/test_dsh_cancel_race.py \
  tests/test_dsh_native_runtime.py tests/test_dsh_reconciliation.py \
  tests/test_dsh_sample_execution.py tests/test_dsh_structured_roles.py
git diff --check
```

### Added deterministic coverage

- one-shot server arm, exact repeat without extension, and admission mismatch;
- wall-clock rollback is never consulted after the server arm;
- protocol ceiling at config, Node runtime, and Python reservation boundaries;
- fixed admission/revision/attempt/timeout propagation before child launch;
- expiry and close checks at reservation and structured durable commit guards;
- close-stage and close-run serialization through commit;
- exact replay after expiry, concurrent early-lookup and insert-ignore races,
  plus expired conflicting-payload rejection;
- HTTP machine error code and Node Sidecar stable-code propagation;
- cancel-ignoring Workflow result, abort-ignoring persistence, admission and
  clone boundary crossings, final return cloning, normal rejected disposal,
  timeout disposal, active bookkeeping, and cross-retry remaining budget;
- normal one-shot disposer/finisher rejection preserving success and the
  primary phase error.

### Self-review and concerns

- No known functional concern remains in the reviewed structured lifecycle and
  durable admission scope.
- The Python admission arm is intentionally in-memory and process-local. A
  sidecar restart invalidates old in-flight wire admissions rather than trying
  to reconstruct a possibly expired monotonic deadline from wall time.
- Abort signals and Workflow cancellation remain resource-reclamation hints.
  Correctness comes from the frozen server deadline and commit guard; a late
  non-cooperating request cannot make a new durable acceptance.
- Exact already-durable receipts remain replayable after expiry. A new durable
  event never bypasses the current fence, deadline, or close state.
- No production service, live run, browser suite, or external state was
  touched. The complete plugin suite and 106 directly related Python tests were
  run; unrelated repository-wide tests were not run.

## Fix Round 3/5

### Review findings addressed

1. `accept_structured` now validates and normalizes the complete
   envelope-derived accepted payload, derives its durable event identity, and
   checks for an exact prior event before consulting process-local admission
   state. This restores response-loss replay after a Python service restart.
   A different structured result or identity under the same event identity
   raises the existing idempotency conflict; malformed or inconsistent schema,
   digest, skill evidence, and session metrics fail during normalization. If
   no durable prior exists, the request continues through prediction-tool
   binding, admission ID, frozen deadline, authorization, and commit guard
   exactly as before.
2. The structured sidecar HTTP boundary no longer treats an arbitrary
   regex-safe exception attribute as public. Its explicit machine-code
   allowlist contains only `structured_role_operational_timeout`; every other
   internal code is omitted from the response.
3. Constructing `DshStructuredRoleRuntime` around a real
   `DshNativeAgentRuntimeClient` without the Host-local admission service now
   fails immediately with `dsh_native_runtime_contract_error`. Both the
   StrategyRouter and sample adapter raw-provider branches therefore reject at
   their provider boundary instead of creating a runtime that later fails on
   missing `admission_id`. Duck-typed `run_stage` test runtimes remain valid,
   while the production server providers continue to pass `self.dsh_tools`.

### Durable replay safety evidence

- The fence-free path can return only an already committed
  `DshStructuredResultAccepted` event whose kind and complete durable payload
  match the normalized retry. The returned `event_id`, `event_seq`, and result
  digest come from that immutable event, and no append is attempted.
- Admission metadata is intentionally not scientific ledger state. A retry
  after process restart therefore does not try to reconstruct or trust the old
  in-memory admission ID; it only acknowledges the exact durable side effect.
- For `sample.plan`, `required_tool_receipt` is Host-produced durable metadata
  that is not present in the wire envelope. The fast path requires the prior
  event to contain that receipt and requires every envelope-derived payload
  field to match after removing only that one recorded field. A new Planner
  result still requires the live prediction binding and armed fence.
- A legacy `DshChildLaunchReserved` event without
  `request_contract_digest` remains readable by projection but is deliberately
  fail-closed for exact reservation retry. The service will not invent the
  missing admission/revision/attempt/timeout contract or treat an unverifiable
  old request as the new strict wire request.

### RED evidence

The restart replay tests were added first:

```bash
PYTHONPATH=src python3.12 -m unittest -v \
  tests.test_dsh_tool_contracts.DshToolServiceTests.test_exact_structured_receipt_replays_after_service_restart \
  tests.test_dsh_tool_contracts.DshToolServiceTests.test_restarted_service_rejects_conflicting_structured_replay \
  tests.test_dsh_tool_contracts.DshToolServiceTests.test_restarted_service_rejects_conflicting_identity_replay
```

```text
Ran 3 tests in 0.007s
FAILED (errors=3)
```

All three reached `structured admission is unavailable` before the durable
receipt lookup. This proved both the lost-response regression and the missing
durable conflict classification.

The HTTP allowlist pair was RED with the approved timeout control already
passing:

```text
Ran 2 tests in 1.065s
FAILED (failures=1)
```

The failing response exposed
`{"error_code": "private_internal_code"}` solely because the string matched
the generic compact-code regex.

The two real-client provider tests were independently RED:

```text
Ran 2 tests in 0.001s
FAILED (failures=2)
```

StrategyRouter returned an admission-free wrapper without raising; the sample
branch reached `run_stage` and reported the deeper `missing admission_id`
contract failure rather than the required Host admission boundary.

### GREEN evidence

The same three restart replay tests passed after the durable fast path:

```text
Ran 3 tests in 0.006s
OK
```

The allowlisted timeout and rejected internal code passed together:

```text
Ran 2 tests in 1.068s
OK
```

Both real raw-provider branches passed with the same immediate error:

```text
Ran 2 tests in 0.000s
OK
```

The directly affected Python modules passed as a focused group:

```bash
PYTHONPATH=src python3.12 -m unittest \
  tests.test_dsh_tool_contracts tests.test_dsh_structured_roles \
  tests.test_dsh_sample_execution tests.test_strategy_router -v
```

```text
Ran 98 tests in 2.586s
OK
```

Complete plugin Node regression:

```bash
node --test integrations/dsh_ecology_plugin/test/*.mjs
```

```text
tests 103
pass 103
fail 0
duration_ms 651.783917
```

Related Python regression, extended with StrategyRouter coverage:

```bash
PYTHONPATH=src python3.12 -m unittest \
  tests.test_dsh_tool_contracts tests.test_dsh_cancel_race \
  tests.test_dsh_native_runtime tests.test_dsh_sample_execution \
  tests.test_core tests.test_dsh_structured_roles \
  tests.test_dsh_reconciliation tests.test_genome_replay \
  tests.test_strategy_router -v
```

```text
Ran 143 tests in 12.870s
OK
```

Syntax, byte-compilation, and whitespace verification completed without
diagnostics:

```bash
node --check integrations/dsh_ecology_plugin/lib/runtime/structured-deadline.js
node --check integrations/dsh_ecology_plugin/lib/runtime/routes.js
node --check integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js
node --check integrations/dsh_ecology_plugin/lib/runtime/structured-roles.js
node --check integrations/dsh_ecology_plugin/lib/sidecar/client.js
PYTHONPATH=src python3.12 -m py_compile \
  src/ecologyrsi_dsh/api/dsh_tools.py \
  src/ecologyrsi_dsh/api/handler.py \
  src/ecologyrsi_dsh/integrations/dsh_structured_roles.py \
  tests/test_dsh_tool_contracts.py \
  tests/test_dsh_sample_execution.py tests/test_strategy_router.py
git diff --check
```

### Self-review and concerns

- No known functional concern remains within the three Round 3 review items.
- Exact durable replay is acknowledgment-only and cannot create a new event.
  Every request without a matching durable event still needs the current
  process-local admission, frozen monotonic deadline, and guarded commit.
- The machine-code allowlist is local to the two structured sidecar routes;
  it does not broaden any other HTTP error surface.
- Raw real-client compatibility now has an explicit secure migration:
  providers must return an admission-bound `DshStructuredRoleRuntime`. There
  is no Node protocol downgrade or implicit HTTP-client admission object.
- No production service, live run, browser suite, or external state was
  touched. The complete plugin suite and 143 related Python tests were run;
  unrelated repository-wide tests were not run.
