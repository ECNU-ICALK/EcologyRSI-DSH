# Dynamic DSH Retrieval Routing Implementation Plan

> **For Codex:** Execute this plan task by task. For each behavioral change, add the failing test first, observe the intended failure, implement the minimum production change, and rerun the focused test before moving on.

**Goal:** Add one stage-local `web_search` tool that prefers DSH internal search, automatically falls back to EcologyRSI OpenAlex retrieval on technical failure or deterministic insufficiency, and durably replays completed results.

**Architecture:** The DSH agent plugin owns the single model-facing wrapper and calls inherited `ctx.web.search`. The Python sidecar owns request validation, evidence-quality decisions, OpenAlex fallback, merging, event persistence, and replay. Existing bootstrap retrieval remains compatible and dynamic results remain advisory.

**Tech Stack:** Node.js ESM plugin code and `node:test`; Python 3.11, dataclasses/stdlib HTTP server, pytest/unittest-style test suite; append-only EcologyRSI event ledger; DSH YAML presets.

---

## Task 1: Freeze the Python retrieval contract with failing tests

**Files:**

- Modify: `tests/test_dsh_tool_contracts.py`
- Modify: `tests/test_genome_replay.py`

- [ ] Add tests for sufficient primary evidence, technical fallback, each quantitative insufficiency reason, empty fallback, replay-before-network, conflicting idempotency reuse, per-stage call budget, unauthorized roles, stale fences, and concurrent duplicate completion.
- [ ] Add an event replay test for `DshRetrievalExecuted`, including rejection of a tampered result digest.
- [ ] Run the focused tests and confirm they fail because the retrieval API/event does not yet exist.

## Task 2: Implement normalized quality, fallback, persistence, and replay

**Files:**

- Modify: `src/ecologyrsi_dsh/knowledge/retrieval.py`
- Modify: `src/ecologyrsi_dsh/api/dsh_tools.py`
- Modify: `src/ecologyrsi_dsh/core/state.py`
- Modify: `src/ecologyrsi_dsh/api/server.py` if projection wiring is required there

- [ ] Factor an allowlisted OpenAlex query function from the bootstrap retriever without changing existing bootstrap behavior.
- [ ] Implement bounded query/result normalization, canonical HTTPS URL handling, lexical-overlap metrics, quality assessment, source merge, and result digesting.
- [ ] Extend `DshToolService` with replay and completion operations, frozen-identity authorization, three-call stage budget, idempotency conflict detection, concurrency serialization, and injected fallback retriever for tests.
- [ ] Append a bounded `DshRetrievalExecuted` event and validate it during ledger replay.
- [ ] Add sidecar HTTP endpoints for replay and completion.
- [ ] Rerun the focused Python tests until green, then run the existing DSH tool and retrieval test groups for regression coverage.

## Task 3: Freeze the DSH wrapper behavior with failing Node tests

**Files:**

- Add: `integrations/dsh_ecology_plugin/test/retrieval_tool.test.mjs`
- Modify: `integrations/dsh_ecology_plugin/test/capabilities.test.mjs`
- Modify: `integrations/dsh_ecology_plugin/test/preset_realms.test.mjs`
- Modify: `integrations/dsh_ecology_plugin/test/stage_runner.test.mjs`

- [ ] Test replay hits before any DSH search call.
- [ ] Test primary multi-query collection and completion through the sidecar.
- [ ] Test safe mapping of provider errors and timeouts.
- [ ] Test that explicit abort propagates without completion/fallback.
- [ ] Test that every role exposes `web_search`, while provider-specific tools and `web_fetch` remain absent.
- [ ] Test optional search after Skill and before the required prediction/structured output sequence.
- [ ] Run the focused Node tests and confirm the expected missing-tool failures.

## Task 4: Implement the single model-facing DSH search wrapper

**Files:**

- Add: `integrations/dsh_ecology_plugin/lib/tools/retrieval.js`
- Modify: `integrations/dsh_ecology_plugin/lib/tools/roles.js`
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/child-bindings.js`
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/sidecar-client.js`
- Modify: `integrations/dsh_ecology_plugin/agent-plugin.js`

- [ ] Register `web_search` with the bounded schema and authorize it for all EcologyRSI roles.
- [ ] Inject the inherited DSH `web` service into the agent plugin.
- [ ] Ask the sidecar for replay before primary search.
- [ ] Invoke `ctx.web.search` for normalized queries, preserve deterministic query ordering, and send only bounded results or safe error codes to completion.
- [ ] Propagate cancellation without starting completion/fallback.
- [ ] Return the sidecar's bounded final result to the model.
- [ ] Rerun focused and full plugin Node tests until green.

## Task 5: Update role contracts, presets, and controller expectations

**Files:**

- Modify: `integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js`
- Modify: `integrations/dsh_ecology_plugin/presets/ecology-rsi-*/persona.yml`
- Modify: relevant `integrations/dsh_ecology_plugin/presets/ecology-rsi-*/skills/*.md`
- Modify: `src/ecologyrsi_dsh/engine/runtime_controller.py`
- Modify: `src/ecologyrsi_dsh/api/dsh_tools.py` role-tool mirror
- Modify: associated Python and Node preset/runtime tests

- [ ] Add `web_search` to all frozen role tool surfaces and default required-tool expectations.
- [ ] Rewrite stage response protocols to allow zero to three searches after Skill and before terminal tools without making search mandatory.
- [ ] Tell agents to request evidence by query only and never select a provider.
- [ ] Preserve Planner prediction cardinality, structured-output ordering, hidden-label boundaries, and judge/critic immutability.
- [ ] Run preset scanning, capability, runtime-controller, and stage-runner tests.

## Task 6: Add bounded runtime projection and compatibility coverage

**Files:**

- Modify: `src/ecologyrsi_dsh/api/projection.py`
- Modify: projection/reconciliation tests under `tests/`
- Modify: protocol documentation that still describes search as generation-start-only

- [ ] Project retrieval count, fallback count, route/stage aggregates, and result digests without full source bodies.
- [ ] Verify historical ledgers without retrieval events retain their prior projection.
- [ ] Clarify that bootstrap catalog retrieval remains citation-authoritative and dynamic retrieval is advisory.
- [ ] Run replay, reconciliation, and API projection tests.

## Task 7: End-to-end verification

**Files:**

- Modify only if verification exposes a defect.

- [ ] Run all Python tests.
- [ ] Run all DSH plugin Node tests.
- [ ] Run static compilation/type/syntax checks used by the repository.
- [ ] Build the distributable artifact and run its artifact verification.
- [ ] Run a local sidecar/native DSH smoke covering a replay miss and hit; if external provider credentials are unavailable, verify the automatic fallback path and report that limitation precisely.
- [ ] Review `git diff --check`, the scoped diff, and final test output before claiming completion.
