# Evolution Safety Follow-up Implementation Plan

**Spec:**
`docs/superpowers/specs/2026-08-25-evolution-safety-followup.md`

## Global Constraints

- Work only in the isolated `codex/evolution-stability` worktree.
- Preserve every existing runtime-stability change and unrelated user change.
- Use TDD for each behavior change and record RED/GREEN commands and output.
- Diagnostic smoke remains one origin × nine prediction cells and can never
  select, validate, final-test, or promote a candidate.
- The web console defaults to an explicit quick-diagnostic preset. Full
  selection-grade evolution is a separate deliberate choice with a visible
  lower-bound duration estimate and scientific-scope warning.
- Formal pass stays disabled until a host-derived canonical assessment and its
  frozen UQ bindings exist; do not simulate formal readiness with caller data.
- Structured-role timeout is a hard end-to-end deadline. No late result may be
  admitted or persisted.
- Pause/cancel closes the launch fence before drain; no child/workflow may
  start after that fence is closed.
- Promotion resampling uses ordered, non-circular contiguous three-day blocks
  and never bridges a calendar gap.
- Production provider pacing remains 60 seconds. Test-only runtimes may use a
  shorter interval only when explicitly isolated and documented.
- Do not expose holdout rows, secrets, reservation identifiers, or raw model
  responses in public projections or test reports.

## Task 0: Keep unrelated mutations responsive during pause/cancel drain

Files:

- `src/ecologyrsi_dsh/api/handler.py`
- `tests/test_http.py`

Add a deterministic failing concurrency test proving that `pause` and
`cancel` do not hold the server-wide mutation lock while DSH quiescence or the
per-run generation barrier is waiting. Keep only the short durable state
boundary under the global lock so archive/create operations for unrelated runs
remain responsive. Confirm the regression against the production symptom:
an old pending pause receipt must never make the proxy time out unrelated
mutations.

## Task 1: Reject affirmative clauses after shared negation

Files:

- `src/ecologyrsi_dsh/evolution/strategies.py`
- `tests/test_autonomous_search_reflection_cycle.py`

Add failing cases for:

- `No eligibility, and promotion is claimed.`
- `No eligibility, and gate passage is asserted.`

Keep legitimate shared-negation enumerations accepted. Change claim scoping so
a coordinated suffix with its own affirmative finite predicate becomes a new
scope and fails closed. Run the focused autonomous-cycle tests.

## Task 2: Enforce a hard structured-role deadline

Files:

- `integrations/dsh_ecology_plugin/lib/runtime/structured-roles.js`
- `integrations/dsh_ecology_plugin/test/structured_roles.test.mjs`
- `integrations/dsh_ecology_plugin/test/stage_runner.test.mjs` when integration
  coverage is needed

Add failing tests for a late successful result, a late rejected result, and a
child result that ignores abort forever. Race child start and result settlement
against one deadline, classify all deadline expirations as operational timeout,
and prevent validation/persistence after expiry. Run focused Node tests.

## Task 3: Close the pause/cancel launch race

Files:

- `integrations/dsh_ecology_plugin/lib/runtime/controller.js`
- `integrations/dsh_ecology_plugin/lib/runtime/provider-stage-gate.js`
- `integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js`
- `integrations/dsh_ecology_plugin/lib/runtime/workflows.js`
- relevant focused Node tests

Add a deterministic failing race test: block schema resolution, begin pause,
release schema, and prove that no child starts and pause returns promptly.
Introduce a synchronous per-run launch fence, re-check admission after the last
preparatory await, and register pending work without another await. Drain to a
stable empty state. Cover both child and workflow launch paths.

## Task 4: Fail closed at the formal scientific boundary

Files:

- `src/ecologyrsi_dsh/core/director.py`
- `src/ecologyrsi_dsh/core/state.py`
- `tests/test_formal_stage_protocol.py`
- any narrowly required director/state test helper

Add failing tests proving diagnostic smoke cannot reserve validation, an
arbitrary `{"outcome": "passed"}` cannot write a passing completion, and
replay rejects an unbound passing formal event. Require selection-eligible
evidence for reservation and keep passing formal execution unavailable until a
host-derived canonical assessment path is implemented. Preserve sealing and
single-exposure semantics on all failures.

## Task 5: Add an explicit quick-diagnostic launch mode

Files:

- `plugins/ecology_evolution/index.html`
- `plugins/ecology_evolution/app.js`
- `plugins/ecology_evolution/assets/js/commands.js`
- `plugins/ecology_evolution/assets/js/catalog.js`
- `plugins/ecology_evolution/assets/js/render_shell.js`
- `plugins/ecology_evolution/assets/js/core.js`
- `plugins/ecology_evolution/assets/js/render_process.js`
- `plugins/ecology_evolution/styles.css`
- `plugins/ecology_evolution/test/smoke.mjs`
- narrowly required API/acceptance tests and documentation

Add failing UI/command tests proving the default launch is one generation,
one candidate, nine cells, concurrency one, and max candidates one. Add a
separate explicit selection-grade preset retaining the existing 5 × 4 × 1600
budget. Display a provider-pacing lower bound (about one minute for the quick
diagnostic and at least 11.8 hours per generation / 59 hours for the default
full preset at 177 origins, four candidates, and 60-second pacing). Treat the
mode as a UI preset only: the server derives `sample_budget_class` from trusted
numeric budgets. Label quick results as diagnostic-only with no champion,
formal-best, selection, validation, or promotion claim. Depend on Task 4's
server-side formal fail-closed boundary before shipping this mode.

## Task 6: Correct the legacy moving-block bootstrap

Files:

- `src/ecologyrsi_dsh/evolution/promotion.py`
- `tests/test_promotion.py`

Add deterministic tests showing every draw consists of contiguous three-day
calendar blocks and that no draw bridges a gap. Retain and validate unique
`origin_block_index` values, require candidate/incumbent index identity, and
sample ordered blocks rather than hash-sorted IDs. Return actual block length
and legal-start count. Run focused promotion and fitness tests.

## Task 7: Full source verification and review

Run the full Python suite, all plugin Node tests, browser smoke tests,
`make verify`, and `git diff --check`. Request an independent whole-branch
review; resolve every Critical or Important finding before runtime packaging.

## Task 8: Rebuild and restart the production-port runtime

Package and install the updated DSH plugin into the existing DSH profile.
Per operator direction, stop all other EcologyRSI project services and run
only ports 8777/8848. Preserve the existing event ledger, confirm sidecar
health, DSH capabilities, plugin revision, and responsive archive/create
commands before creating new work.

## Task 9: Run and analyze fresh controlled evolutions

Run one cucumber and one tomato diagnostic with one generation, one candidate,
and nine cells. Monitor durable stage, child, retry, and progress events until
terminal state. If a run pauses or fails, diagnose from the first causal error,
add a failing regression test, fix, rebuild, and rerun. Both runs must pass
`scripts/dsh_native_e2e_acceptance.py` and end without promotion.

## Task 10: Final verification and handoff

Re-run the complete verification set on the exact code/package used by the
successful runs. Audit run duration, retries, model/tool calls, candidate
metrics, cohort bindings, pending receipts, formal exposure count, and public
projection. Request a final independent code/scientific review and report all
remaining formal-readiness gaps explicitly.
