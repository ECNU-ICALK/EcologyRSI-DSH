# Evolution Runtime Stability Implementation Plan

> Execute in the `codex/evolution-stability` worktree. Use test-driven
> development for behavior changes and verify against isolated DSH-native runs.

## Task 1: Fix diagnostic negation parsing

- Add the exact live English outputs as passing regression cases.
- Keep affirmative and mixed independent-clause controls failing.
- Extend bounded negation recognition in
  `src/ecologyrsi_dsh/evolution/strategies.py`.
- Run the focused autonomous-cycle tests.

Status: completed.

## Task 2: Add a research semantic circuit breaker

- Preserve bounded validation detail on `ResearchResponseContractError`.
- Pause with `research_contract_retry_exhausted` after the inner repair budget.
- Prove no outer `GatewayRetryScheduled` event is written for this error.
- Preserve delayed retries for transport/provider failures.

Status: completed.

## Task 3: Bound and clarify sample critic execution

- Add a critic-specific 180-second timeout configuration.
- Require exact wave/sample identifiers and non-empty structured arguments.
- End a rejected child without prose and use the Host's fresh-child retry path.
- Cover configuration and stage-runner behavior with Node tests.

Status: completed.

## Task 4: Project active DSH work accurately

- Derive provider wait, model execution, model retry, and host validation from
  durable events.
- Map `generation.search-plan` to the broad `search` stage.
- Render stage, role, attempt, elapsed time, and heartbeat without synthetic
  percentages.
- Hide active DSH state after terminal completion.
- Clear a recovered stage failure from completed run projections.
- Cover Python projection and browser rendering behavior.

Status: completed.

## Task 5: Real-run verification

- Build/install the updated plugin in an isolated DSH profile.
- Run one-generation, one-candidate, nine-cell cucumber and tomato diagnostics.
- Inspect contract rejections, retries, evidence binding, tool execution, and
  terminal decision.
- Run the DSH-native acceptance verifier for each completed run.

Status: completed. Cucumber completed in about 7m24s. Tomato paused safely on a
new research phrase, resumed from its checkpoint after the regression fix, then
completed and passed acceptance.

## Task 6: Full regression verification

- Run the complete Python test suite.
- Run every plugin Node test.
- Run the browser smoke test and `git diff --check`.
- Confirm the completed tomato projection no longer exposes stale failure data
  after restarting the isolated service on the final code.

Status: completed. Final verification: 871 Python tests passed (4 skipped),
78 Node plugin tests passed, browser smoke passed, `git diff --check` passed,
source-only delivery verification passed, both real runs passed DSH-native
acceptance after the final plugin reinstall, and the completed projection has no
stale failure or active-child state.
