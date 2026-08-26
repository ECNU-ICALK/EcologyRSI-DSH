# Task 4 report — bind sample output before durable acceptance

## What changed

- Each sample-stage output schema clone is specialized synchronously after
  schema loading and before prompt construction or child/Workflow launch.
  `wave_digest` is bound to the exact Host value with `const`; planner and
  critic decision `sample_id` values are bound to the exact ordered Host enum;
  reflection `sample_id` is bound only to outer `context.sample.sample_id`.
- Host sample context now fails locally with `sample_stage_context_invalid`
  when its wave digest is not lowercase SHA-256, its decision sample set is
  empty or above 128 items, or a sample ID is empty, duplicated, or above 240
  characters. Invalid context starts no child/Workflow and persists nothing.
- All three sample prompts explicitly require exact Host identities. The
  reflection prompt forbids using `context.sample.prediction_cells[].sample_id`.
- Direct and Workflow missing/schema-rejected captures use the stable
  `structured_result_missing` code. Only `sample.plan`, `sample.critic`, and
  `sample.reflect` add that code to the existing two-attempt retry budget.
  The existing global `structured_child_model_error` retry is unchanged.
- Planner retry keeps the existing prediction binding: distinct child sessions
  receive a deep copy of the same cached prediction result and durable receipt,
  while the scientific executor runs once.

## Files

- `integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js`
- `integrations/dsh_ecology_plugin/test/stage_runner.test.mjs`
- `integrations/dsh_ecology_plugin/test/launch_fence.test.mjs` (valid sample
  schema/context fixture only; the Task 3 race assertions are unchanged)
- `tests/test_dsh_sample_execution.py`
- `.superpowers/sdd/2026-08-25-evolution-safety-followup/task-4-report.md`

No static schema, schema ID, schema dialect, reason/tool enum, retry count,
deadline, ledger, plan, spec, or progress file changed.

## RED evidence

Command after adding the deterministic tests and before production changes:

```bash
node --check integrations/dsh_ecology_plugin/test/stage_runner.test.mjs && \
node --test integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
```

Result:

```text
tests 29
pass 23
fail 6
duration_ms 421.145459
```

The six failures matched the intended defects:

```text
sample critic retries a normally-ended missing result in a fresh child
  actual wave schema: { type: 'string' }
  expected: { type: 'string', const: '<Host wave>' }

sample reflection binds the outer Host identity and retries one missing result
  Error code: structured_result_missing (no retry)

sample reflection stops after two missing structured outputs without persistence
  actual reservations: 1
  expected reservations: 2

malformed sample Host identities fail locally before child launch or persistence
  actual error code: structured_result_missing
  expected error code: sample_stage_context_invalid

sample planner retries a missing Workflow result with a fresh reservation and session
  Error: structured workflow returned an invalid result batch

sample planner waves execute through the retained DSH Workflow Engine
  actual wave schema: { type: 'string' }
  expected: { type: 'string', const: '<Host wave>' }
```

The existing prediction-binding control was also run before implementation:

```bash
PYTHONPATH=src /Users/jiezhou/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12 \
  -m unittest \
  tests.test_dsh_sample_execution.DshSampleExecutionTests.test_fresh_planner_children_reuse_one_prediction_result_and_receipt \
  -v
```

```text
Ran 1 test in 0.000s
OK
```

This characterization establishes that two fresh planner child session IDs
reuse one result/receipt and invoke the registered scientific executor once.

## GREEN evidence

Focused Node command after implementation and the final clone-isolation
regression:

```bash
node --test integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
```

```text
tests 30
pass 30
fail 0
duration_ms 438.939666
```

The tests prove exact wave/sample binding on direct and Workflow launches,
outer-reflection identity selection, per-run clone isolation, malformed Host
context rejection before launch, fresh reservation/session identity for retry,
one persistence after missing-then-valid, zero persistence after two missing,
and exact non-retry behavior for lifecycle, admission, timeout, authorization,
sidecar, not-accepted, and persistence error codes.

Complete plugin Node suite:

```bash
node --test integrations/dsh_ecology_plugin/test/*.mjs
```

```text
tests 159
pass 159
fail 0
duration_ms 665.415292
```

Relevant Python sample and durable-tool contract suites:

```bash
PYTHONPATH=src /Users/jiezhou/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12 \
  -m unittest tests.test_dsh_sample_execution tests.test_dsh_tool_contracts -v
```

```text
Ran 62 tests in 2.181s
OK
```

The changed JavaScript files passed `node --check`, the changed Python test
passed `python3.12 -m py_compile`, and `git diff --check` completed with exit
code 0 and no diagnostics.

## Retry and replay semantics

- A missing first result has no structured-result persistence call, so there is
  no poison accepted receipt for the runtime replay layer to return. The next
  attempt allocates a fresh durable child reservation under the same stage
  business identity; the allocator supplies launch attempts 1 then 2 and fresh
  reservation/child session identities.
- Planner prediction execution is scoped outside the JavaScript child retry.
  `DshPredictionToolBinding` caches the materialized result and its receipt;
  a fresh child session may make its own authorized once-per-session call but
  receives the cached result without a second scientific execution.
- The retry allowlist is exactly the pre-existing global
  `structured_child_model_error`, plus `structured_result_missing` only for
  sample plan/critic/reflect. Control abort, child start/result phase failure,
  admission closed, operational timeout, authorization, provider/sidecar
  failure, durable not-accepted, and persistence failure are not retried.
  Therefore a possibly committed outcome is never followed by a fresh model
  attempt.

## Residual boundary

The retained DSH schema subset can bind membership with `const`/`enum`, but it
does not enforce decision-array completeness or uniqueness after cardinality
keywords are projected out. The Python/Host expected-contract validation
remains authoritative for those properties; they are intentionally outside
Task 4.

## Fix Round 1/5 — trusted capture and phase provenance

### Review findings addressed

- A `stopReason: "error"` result is classified as a rejected schema capture
  only when no valid structured object exists and Host-owned child session
  events contain a `structured_output` tool call followed by its matching
  `tool/result` with `isError: true`. Direct stages read the live child session;
  Workflow stages use the end-event snapshot or the live session when result
  settlement precedes `workflow/agent-end`. An ordinary error result without
  this evidence remains `structured_child_model_error`.
- Retry no longer trusts public `error.code`. A private runtime module owns a
  `WeakMap` from freshly created phase errors to their immutable local phase.
  The retry gate accepts only trusted `model` phase errors globally and trusted
  `capture` phase errors for the three sample stages.
- Direct and Workflow start, result, admission, persistence, not-accepted,
  child-session, abort, and control boundaries now produce locally classified
  errors. Every external cause is freshly wrapped; its private diagnostic data
  is retained only inside the module-owned weak metadata and cannot forge the
  wrapper's public phase.
- Persistence failures are always the non-retryable
  `structured_result_persist_failed`, even when a failure after a possible
  durable commit carries either public allowlist code. No second reservation,
  child/Workflow start, or persistence call follows that ambiguity.

### RED evidence

The first focused RED run, after adding capture and phase-spoof regressions but
before production changes, was:

```bash
node --check integrations/dsh_ecology_plugin/test/stage_runner.test.mjs && \
node --test integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
```

```text
tests 35
pass 31
fail 4
duration_ms 438.162667
```

The four failures were the two independent defects on both launch paths:

```text
direct sample schema rejection is a bounded missing-capture retry
  actual: structured_child_model_error
  expected: structured_result_missing

Workflow sample schema rejection is a bounded missing-capture retry
  actual: structured_child_model_error
  expected: structured_result_missing

direct phase causes cannot spoof either retry allowlist code
  actual: public start cause escaped with structured_result_missing
  expected: structured_child_start_failed and one reservation/start

Workflow phase causes cannot spoof retry or duplicate a post-commit persist
  actual: public start cause escaped with structured_result_missing
  expected: structured_child_start_failed and one reservation/start
```

A second TDD cycle tightened the Workflow capture test to the rc.6 ordering in
which the live session contains the rejected tool result before the end-event
snapshot is published:

```bash
node --test --test-name-pattern='Workflow sample schema rejection' \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
```

```text
tests 1
pass 0
fail 1
actual: structured_child_model_error
expected: structured_result_missing
duration_ms 54.784208
```

### GREEN evidence

Focused stage, structured-role, Workflow, and launch-fence run:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs \
  integrations/dsh_ecology_plugin/test/structured_roles.test.mjs \
  integrations/dsh_ecology_plugin/test/workflow_lifecycle.test.mjs \
  integrations/dsh_ecology_plugin/test/launch_fence.test.mjs
```

```text
tests 68
pass 68
fail 0
duration_ms 546.245042
```

Complete plugin Node suite:

```bash
node --test integrations/dsh_ecology_plugin/test/*.mjs
```

```text
tests 164
pass 164
fail 0
duration_ms 703.662584
```

Related Python sample and durable-tool contracts:

```bash
PYTHONPATH=src /Users/jiezhou/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12 \
  -m unittest tests.test_dsh_sample_execution tests.test_dsh_tool_contracts -v
```

```text
Ran 62 tests in 2.211s
OK
```

The phase matrix injects both public allowlist codes independently at direct
and Workflow start, result, admission, and persistence boundaries. Every case
uses one reservation/start. Both post-commit persistence cases call persist
exactly once and make zero fresh attempts. Legitimate local missing and model
phases still consume exactly two fresh attempts; rejected captures make zero
structured-result persistence calls.

## Fix Round 2/5 — rc.6 consumed-turn capture classification

### Runtime contract

- Direct sample capture classification now requires the real rc.6 public shape:
  no valid structured object, public `stopReason: "error"`, and a child event
  log whose `foldConsumedWork`-equivalent terminal reason is exactly
  `completed`. A completed consumed turn with no `structured_output` call is an
  ordinary missing capture.
- A rejected `structured_output` call is missing only when the latest call in
  that consumed turn has a later result with the same turn, step, and call ID,
  exact `sourceEventSeqs: [call.seq]`, nested `isError: true`, and exact
  `data.error` identity `ToolArgsError` / `INVALID_ARGS`.
- Provider/model terminal errors, authorization or guard failures, aborted or
  unknown tools, missing/mismatched source linkage, and reused call IDs are
  public `structured_child_model_error` outcomes backed by a private
  `model_terminal` provenance. They are not accepted by either retry allowlist.
  The existing trusted `model` provenance and its global retry remain intact.
- Workflow classification now requires the natural rc.6 combination of an
  exact `workflow/agent-end` child identity with `outcome: "failed"`, a
  completed Workflow result containing `[null]`, and the same child-session
  consumed-turn classification. Top-level Workflow shape alone cannot create a
  missing retry.
- The Round 1 module-owned `WeakMap`, public error-code compatibility, and fresh
  wrapping at every lifecycle and persistence boundary are unchanged.

### RED evidence

The deterministic natural-shape RED command was:

```bash
node --check integrations/dsh_ecology_plugin/test/stage_runner.test.mjs && \
node --test \
  --test-name-pattern='consumed completed|exact INVALID_ARGS|provider terminal|authorization, abort|reused callId|Workflow completed null' \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
```

```text
tests 8
pass 4
fail 4
duration_ms 65.969125
```

The failures were the intended four unsafe branches: direct provider terminal
error, direct authorization/ABORTED/UNKNOWN_TOOL, direct reused call ID with a
mismatched source, and Workflow authorization rejection all produced
`structured_result_missing` instead of one-attempt
`structured_child_model_error`. The completed/no-call and exact INVALID_ARGS
fixtures already exercised the positive missing branch.

A second single-case RED ensured the Workflow classifier accepts only the
natural `[null]` batch, not an arbitrary invalid value:

```bash
node --test --test-name-pattern='non-null-batch shape' \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
```

```text
tests 1
pass 0
fail 1
actual: structured_result_missing
expected: structured_child_model_error
duration_ms 53.745959
```

### GREEN evidence

The same focused command after implementation passed:

```text
tests 8
pass 8
fail 0
duration_ms 66.081166
```

The final focused runtime suite passed after all prior missing-result fixtures
were migrated to the natural rc.6 terminal/outcome shape:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs \
  integrations/dsh_ecology_plugin/test/structured_roles.test.mjs \
  integrations/dsh_ecology_plugin/test/workflow_lifecycle.test.mjs \
  integrations/dsh_ecology_plugin/test/launch_fence.test.mjs
```

```text
tests 76
pass 76
fail 0
duration_ms 541.586209
```

Complete plugin Node suite:

```bash
node --test integrations/dsh_ecology_plugin/test/*.mjs
```

```text
tests 172
pass 172
fail 0
duration_ms 656.309708
```

Related Python sample and durable-tool contracts:

```bash
PYTHONPATH=src /Users/jiezhou/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12 \
  -m unittest tests.test_dsh_sample_execution tests.test_dsh_tool_contracts -v
```

```text
Ran 62 tests in 2.179s
OK
```

Every negative capture-classification case makes one reservation/start and zero
structured-result persistence calls. Both positive missing cases consume the
exact two-attempt budget and still persist zero results when exhausted; existing
missing-then-valid paths persist exactly once.

## Fix Round 3/5 — exact Workflow stop-reason allowlist

### Runtime contract

- The Workflow seam now follows the rc.6 stop-reason union exactly. Top-level
  `cancelled` and defensive legacy `aborted` are freshly wrapped as the trusted
  nonretry `aborted` phase with public code `structured_child_aborted`.
- Top-level `error` and every unknown non-completed token are freshly wrapped
  as the trusted nonretry `model_terminal` phase. The stable public code remains
  `structured_child_model_error`, but its private provenance cannot enter the
  model retry allowlist.
- A Workflow top-level stop token never creates the private retryable `model`
  origin. Local direct-child model failures retain that origin and their
  existing bounded retry; Workflow sample capture-missing retains only the
  Round 2 completed/`[null]` plus failed-child/event-evidence route.
- The absolute-deadline retry test now uses that natural capture-missing route
  instead of the non-existent Workflow `model_error` token.

### RED evidence

The deterministic RED command was:

```bash
node --check integrations/dsh_ecology_plugin/test/stage_runner.test.mjs && \
node --test --test-name-pattern='exact nonretry boundaries|absolute local' \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
```

```text
tests 2
pass 1
fail 1
duration_ms 96.650292
```

The natural capture-missing deadline retry passed. The exact boundary matrix
failed first on rc.6 `cancelled`: the old branch returned
`structured_child_model_error` instead of `structured_child_aborted` and was
eligible for a second reservation/penalty. The same old catch-all branch made
`error` and unknown tokens retryable model origins.

### GREEN evidence

After the minimal stop-reason mapping, the focused provenance and retry set
passed:

```bash
node --test \
  --test-name-pattern='exact nonretry boundaries|absolute local|transient child model failure|exact INVALID_ARGS|failed child no-call' \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
```

```text
tests 6
pass 6
fail 0
duration_ms 96.836916
```

Each `cancelled`, `error`, unknown, and legacy `aborted` case asserts the exact
tuple of one reservation, one Workflow start, zero provider penalties, and zero
persistence calls. The same run proves direct trusted model and natural direct
and Workflow capture-missing retries remain enabled.

Final focused runtime suite:

```bash
node --test \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs \
  integrations/dsh_ecology_plugin/test/structured_roles.test.mjs \
  integrations/dsh_ecology_plugin/test/workflow_lifecycle.test.mjs \
  integrations/dsh_ecology_plugin/test/launch_fence.test.mjs
```

```text
tests 77
pass 77
fail 0
duration_ms 543.867917
```

Complete plugin Node suite:

```bash
node --test integrations/dsh_ecology_plugin/test/*.mjs
```

```text
tests 173
pass 173
fail 0
duration_ms 671.762834
```

Related Python sample and durable-tool contracts:

```bash
PYTHONPATH=src /Users/jiezhou/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12 \
  -m unittest tests.test_dsh_sample_execution tests.test_dsh_tool_contracts -v
```

```text
Ran 62 tests in 2.200s
OK
```

The changed JavaScript files also passed `node --check`; `git diff --check`
completed with no diagnostics.
