# Task 5 report — durable bounded retry circuits

Base: `5bb63e3fdb1030ea844b1d8b30de204b41796e1d`

## Outcome

The former memory-first, best-effort retry path is replaced by a
director-owned durable optimistic-CAS decision. Each retryable orchestration
failure now linearizes as exactly one of:

- a version-2 `GatewayRetryScheduled` event followed by timer installation; or
- a checkpoint-preserving `RunPaused` circuit transition.

The event commit is the scheduling authority. A failed append installs no
timer and causes no new gateway/DSH invocation; the exact pending failure
report is retried until its durable decision is known. An exact duplicate
`failure_id` replays the original decision, while distinct IDs cannot reuse one
attempt anchor to increment the circuit twice.

The four former `_retry_later_error` classes have independent finite scopes:

| Retry class | Pause code | Operator action |
| --- | --- | --- |
| `model_gateway` | `gateway_retry_circuit_open` | `check_gateway_then_resume` |
| `dsh_native_runtime` | `dsh_runtime_retry_circuit_open` | `check_dsh_runtime_then_resume` |
| `research_timeout` | `research_timeout_retry_circuit_open` | `check_gateway_then_resume` |
| `sample_result_persistence` | `sample_persistence_retry_circuit_open` | `check_persistence_then_resume` |

Every class opens after six consecutive logical failures or within the
30-minute epoch boundary. A proposed retry at or beyond the epoch deadline
pauses immediately, so a long `Retry-After` cannot leave a run indefinitely
`running`. Deterministic credential/configuration failures retain their prior
fail-closed behavior.

## Durable and concurrent semantics

- The circuit scope is run incarnation, generation, stage, and retry class.
  `failure_id` is derived from stable Host-owned run/scope/attempt authority;
  exception text, retry time, wall clock, random UUIDs, credentials, and raw
  provider bodies are not identity inputs or persisted fields.
- `schedule_gateway_retry_or_pause()` always replays fresh state and appends
  with `expected_run_seq`. CAS conflicts recompute the decision from current
  durable state. Same-ID uncertain commits return the original schedule/pause
  outcome and event identity.
- Version-2 replay validates exact payload shape, current incarnation and
  generation, `RUNNING` status, monotonic timestamps, exact count/epoch chain,
  unique failure IDs, and authoritative attempt-anchor succession. Legacy
  retry heartbeats remain readable but cannot inflate a new bounded epoch.
- A durable scoped stage completion, generation advance, or explicit resume
  resets the scoped count. A stage-start event does not. DSH runtime, gateway,
  research timeout, and sample persistence counts never mix.
- Threshold decisions lose safely to cancel, terminal transition, manual
  pause, scoped success, or generation advance. Old workers and timer callbacks
  carry decision sequence and breaker epoch and become no-ops after a competing
  transition or resume.
- A retry timer is installed only after the schedule event commits. Timer
  startup failure retains the durable deadline and retries installation; it
  never probes before `retry_at`. Recovery restores the latest unconsumed
  version-2 authority, including a logical attempt whose stage-start/failed
  marker was already written before process restart.
- Circuit resume records the exact origin pause event and breaker epoch,
  clears stale cooldown/timer state, starts the next breaker epoch, and retries
  the same checkpoint. Circuit-paused `start` is rejected in the HTTP control
  layer, director, and state replay, so it cannot bypass the explicit resume
  transition.

## Projection and console behavior

Running projections expose an allowlisted bounded retry summary: class, stage,
epoch, N/N, first/last failure times, safe error code, retry time, and suggested
action. Stage-start does not erase that summary. Circuit-paused projections
expose the stable code/action and checkpoint information while retaining the
legacy `pause_code`/`pause_reason` fields.

Neither public projections nor event payloads expose `failure_id`, attempt
anchor, exception text, provider response body, token, or credential material.
The console recognizes exact code/class/action triples. Its circuit action
sends only `resume` and explains that the operator is retrying the current
checkpoint; it does not issue the ordinary post-resume `/advance`. Manual
pause behavior remains unchanged. DSH runtime has distinct actionable copy.

## RED evidence

Tests were added before each production slice and run in focused isolation.
The deterministic failures observed before implementation included:

- the director had no atomic retry-or-pause operation and the sixth gateway
  failure remained schedulable instead of producing `RunPaused`;
- duplicate workers, two different IDs using the same attempt anchor, and stale
  CAS state could increment or schedule independently;
- a long retry delay was accepted beyond the 30-minute boundary;
- replay accepted shape-valid retry/pause events with invalid lifecycle state,
  count jumps, reused anchors, duplicate failure IDs, forged threshold pauses,
  generic resume, and `RunStarted` circuit bypasses;
- auto-progress installed memory state before durable commit, retried the
  gateway after append ambiguity, retained old timers across resume, and could
  dispatch early when `Timer.start()` failed;
- restart after a retry-owned stage `started`/`failed` marker selected the
  wrong attempt authority and left the run orphaned;
- the outer DSH runtime failure was misclassified as `model_gateway` because a
  nested gateway exception won classification, and retry backoff mixed classes;
- HTTP `start` called the director for a circuit-paused run, and the console's
  paused action emitted both resume and `/advance`;
- running/paused public state lacked bounded circuit details and exact DSH
  operator guidance.

Representative RED tests are retained in
`tests/test_gateway_retry_circuit.py`, `tests/test_auto_progress.py`, and
`plugins/ecology_evolution/test/smoke.mjs`. They cover the above failures with
controlled clocks, barriers, fake timers, append faults, restart, and fake
gateway/DSH boundaries rather than production services.

## GREEN evidence

Focused director/state/auto-progress suite:

```bash
.venv/bin/python -m unittest \
  tests.test_gateway_retry_circuit tests.test_auto_progress -v
```

```text
Ran 77 tests in 25.641s
OK
```

Projection, redaction, and HTTP compatibility:

```bash
.venv/bin/python -m unittest \
  tests.test_stage_projection tests.test_public_redaction tests.test_http -v
```

```text
Ran 44 tests in 11.928s
OK
```

Director invariants, core compatibility, and cleanup:

```bash
.venv/bin/python -m unittest \
  tests.test_director_invariants tests.test_core tests.test_run_cleanup -v
```

```text
Ran 32 tests in 2.334s
OK
```

Execution projection and sample-result contract suites:

```bash
.venv/bin/python -m unittest tests.test_execution_projection -v
.venv/bin/python -m unittest tests.test_sample_results_contract -v
```

```text
Ran 37 tests in 0.073s
OK

Ran 17 tests in 1.617s
OK
```

Console smoke coverage:

```bash
node plugins/ecology_evolution/test/smoke.mjs
```

```text
ecology evolution plugin smoke tests passed
```

The end-to-end deterministic outage test proves six failures pause, explicit
resume retries the same checkpoint, and recovery completes one generation with
exactly one candidate. Separate tests prove the DSH runtime circuit, N-1
success reset, second bounded epoch, future/due retry recovery, append and
timer-start faults, duplicate-worker linearization, threshold races, old
worker/timer rejection, and safe projections.

Final lightweight validation on the submitted tree:

```bash
.venv/bin/python -m compileall -q \
  src tests/test_gateway_retry_circuit.py tests/test_auto_progress.py
node --check plugins/ecology_evolution/assets/js/core.js
node --check plugins/ecology_evolution/assets/js/commands.js
node --check plugins/ecology_evolution/assets/js/render_shell.js
node --check plugins/ecology_evolution/assets/js/render_process.js
git diff --check
```

All commands exited 0 with no diagnostics. The local virtual environment does
not contain a `ruff` executable, so no unavailable lint command is represented
as having run. The focused and related suites above were used instead of an
unrelated full-repository expansion during final handoff.

## Files and boundary

- `src/ecologyrsi_dsh/core/director.py`
- `src/ecologyrsi_dsh/core/state.py`
- `src/ecologyrsi_dsh/api/auto_progress.py`
- `src/ecologyrsi_dsh/api/execution.py`
- `src/ecologyrsi_dsh/api/projection.py`
- `src/ecologyrsi_dsh/api/events.py`
- `plugins/ecology_evolution/assets/js/{core,commands,render_shell,render_process}.js`
- `plugins/ecology_evolution/test/smoke.mjs`
- `tests/test_gateway_retry_circuit.py`
- `tests/test_auto_progress.py`
- this report

No plan, spec, progress, Task 3/4 implementation, production service, port,
database, or active run was modified. No automatic half-open probe was added;
explicit resume remains the only operator authorization for a new epoch.

## Fix Round 1/5 — replayable pause evidence and owned error codes

Base: `803a2af4dfdf6a169c2b1c3c05470800f37ea4e5`

- Every circuit pause now persists `pause_trigger` as exactly
  `failure_limit`, `epoch_elapsed`, or `retry_deadline_reaches_epoch`, plus the
  configured epoch seconds, exact epoch deadline, and proposed retry time.
  Replay verifies the selected predicate and its priority, rejecting a
  count-2/limit-6 pause without evidence and inconsistent trigger evidence.
- Retry error codes use one per-class Host-owned allowlist and fixed generic
  mapping at the auto-progress boundary and again at the director append
  boundary. Replay independently rejects unowned codes. The token-shaped
  `sk_live_abc123credential` regression proves the value is absent from the
  SQLite event payload, run projection, and public event export.

The new focused tests were first run against `803a2af4` and produced the
intended RED result: 8 tests, 5 failures and 3 errors. Missing trigger fields,
the accepted forged pause, and the raw token-shaped error code accounted for
all failures. After the minimal implementation the same 8 tests passed.

Final Task 5 focused verification:

```bash
.venv/bin/python -m unittest \
  tests.test_gateway_retry_circuit tests.test_auto_progress -v
```

```text
Ran 82 tests in 26.154s
OK
```

`node plugins/ecology_evolution/test/smoke.mjs`, Python `compileall`, and
`git diff --check` also exited 0. No unrelated full-repository suite was run.

## Fix Round 2/5 — ledger-time trigger evidence

Base: `870fc1f7b485c10d6f23bec0ddaba479bf038844`

Pause replay no longer trusts self-reported failure times. It derives the first
failure from the first active retry event's `created_at` (or the pause event for
a first-failure pause), the last failure from the pause event's `created_at`,
and the proposed retry from that pause time plus a persisted finite delay
bounded to 0..3600 seconds. Retained timestamp fields must exactly equal these
derived values. The two count-2 forged elapsed/deadline tests were RED because
the old replay accepted both; they and the out-of-range-delay control are now
GREEN.

```text
Targeted trigger tests: 6/6 OK
Task 5 focused suites: Ran 85 tests in 26.329s — OK
Python compileall and git diff --check: exit 0
```
