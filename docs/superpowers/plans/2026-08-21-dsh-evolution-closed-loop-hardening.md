# DSH Evolution Closed-Loop Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make reflection actionable, keep deterministic candidate fitness authoritative, exercise DSH workflows, and expose truthful DSH runtime evidence.

**Architecture:** Python validates and persists a bounded reflection and two-track fitness record. DSH receives the complete validated context, executes fixed role/workflow templates, and returns schema-bound results whose real child identity and provider usage are persisted.

**Tech Stack:** Python 3.10+, unittest, Node.js ESM, DSH 0.1.0-rc.6, Cordis plugin presets, SQLite event ledger.

**Spec:** `docs/superpowers/specs/2026-08-21-dsh-evolution-closed-loop-hardening-design.md`

## Global Constraints

- DSH itself never evolves; only registered plugin-genome fields evolve.
- Python remains the scientific and durable-ledger authority.
- No per-sample Agent token hard limit is introduced.
- Tests are written and observed failing before production edits.
- The workspace has no Git metadata, so commit steps are recorded as unavailable rather than simulated.

---

### Task 1: Actionable reflection bridge

**Files:**
- Modify: `src/ecologyrsi_dsh/evolution/strategies.py`
- Modify: `src/ecologyrsi_dsh/evolution/context.py`
- Modify: `src/ecologyrsi_dsh/evolution/genome.py`
- Test: `tests/test_strategy_router.py`
- Test: `tests/test_plugin_genome.py`

**Interfaces:**
- Consumes: validated `GenerationBatch.research_iteration` and previous aggregate analysis.
- Produces: `candidate.propose` context fields `research_iteration`, `evolution_reflection`, and `prohibited_mutation_digests`.

- [ ] Add a failing test proving historical failure priorities and rejected parameters appear in the DSH proposer context.
- [ ] Run `PYTHONPATH=src .venv/bin/python -m unittest tests.test_strategy_router -v` and confirm the new assertion fails because only digests are present.
- [ ] Build a bounded reflection from existing aggregate summaries and include it in `stage_context` without sample-level labels or holdout data.
- [ ] Add a failing test proving an exact prohibited mutation is rejected without new evidence.
- [ ] Implement Host-side duplicate mutation validation and run the focused tests green.

### Task 2: Two-track scientific and execution-policy fitness

**Files:**
- Modify: `src/ecologyrsi_dsh/evaluators/gateway_sample_adapter.py`
- Modify: `src/ecologyrsi_dsh/evaluators/sample_execution.py`
- Modify: `src/ecologyrsi_dsh/evaluators/fitness.py`
- Modify: `src/ecologyrsi_dsh/api/projection.py`
- Test: `tests/test_sample_execution.py`
- Test: `tests/test_fitness.py`

**Interfaces:**
- Consumes: successful registered-tool prediction plus optional DSH critic recommendation.
- Produces: immutable `raw_candidate` outcome and separate `execution_policy` diagnostics.

- [ ] Add a failing sample-execution test in which the critic requests fallback after a successful prediction and assert the original prediction remains authoritative.
- [ ] Run the focused test and confirm it fails by returning `SampleRepairRequired`.
- [ ] Change successful-review handling so non-accept critic choices are advisory and preserve the raw result.
- [ ] Add explicit raw/repaired score-source fields and a regression test preventing fallback from converting a candidate failure into neutral reward.
- [ ] Run sample-execution and fitness tests green.

### Task 3: Real DSH workflow execution

**Files:**
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/workflows.js`
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js`
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/controller.js`
- Test: `integrations/dsh_ecology_plugin/test/workflow_lifecycle.test.mjs`
- Test: `integrations/dsh_ecology_plugin/test/structured_roles.test.mjs`

**Interfaces:**
- Consumes: pre-reserved homogeneous structured items and coordinator/sample-planner role-host.
- Produces: DSH `WorkflowRun` results with fixed scripts, bounded limits, and quiescent cancellation.

- [ ] Add a failing integration test that expects a sample wave to call `workflowEngine.start()` with structured args and no script interpolation.
- [ ] Run the Node test and confirm no workflow is started.
- [ ] Route sample planner/critic batches through `startHomogeneousWorkflow`, preserving one-shot paths for researcher/proposer/judge.
- [ ] Add cancellation and limit tests, then run the DSH plugin suite green.

### Task 4: Reason codes and semantic review guards

**Files:**
- Modify: `integrations/dsh_ecology_plugin/schemas/sample-decisions.schema.json`
- Modify: `integrations/dsh_ecology_plugin/schemas/sample-review.schema.json`
- Modify: `src/ecologyrsi_dsh/evaluators/dsh_sample_adapter.py`
- Modify: `src/ecologyrsi_dsh/evaluators/gateway_sample_adapter.py`
- Test: `tests/test_sample_execution.py`
- Test: `integrations/dsh_ecology_plugin/test/package_contract.test.mjs`

**Interfaces:**
- Consumes: Host-owned reason-code catalog and confidence threshold.
- Produces: schema-valid, semantically consistent decisions.

- [ ] Add failing tests for unknown reason codes and contradictory high-confidence/low-confidence decisions.
- [ ] Add enum-constrained schemas and include allowed reason codes in stage context.
- [ ] Add Host semantic validation and run Python/Node focused tests green.

### Task 5: Runtime identity and token observability

**Files:**
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/structured-roles.js`
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js`
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/controller.js`
- Modify: `src/ecologyrsi_dsh/api/projection.py`
- Test: `integrations/dsh_ecology_plugin/test/token_semantics.test.mjs`
- Test: `tests/test_stage_projection.py`

**Interfaces:**
- Consumes: actual `SubagentRun.childId`, reservation label, TokenMeter pressure, and SessionProjection token usage.
- Produces: durable actual child UUID, separate label, and observed provider-usage summaries.

- [ ] Add failing tests proving persisted identity uses the real child UUID and usage comes from SessionProjection rather than TokenMeter.
- [ ] Pass child identity into the persistence callback and aggregate token projections per run.
- [ ] Project provider usage with explicit availability/source labels and run focused tests green.

### Task 6: Native-only admission, cancellation, and E2E

**Files:**
- Modify: `src/ecologyrsi_dsh/api/handler.py`
- Modify: `src/ecologyrsi_dsh/api/auto_progress.py`
- Modify: `src/ecologyrsi_dsh/core/state.py`
- Test: `tests/test_dsh_cancel_race.py`
- Test: `tests/test_runtime_integration.py`
- Test: `scripts/dsh_native_e2e_acceptance.py`

**Interfaces:**
- Consumes: autonomous run creation and terminal admission state.
- Produces: native-only autonomous runs and a terminal fence rejecting late events.

- [ ] Add failing tests for autonomous legacy creation and every post-cancel late event.
- [ ] Enforce native protocol at the server boundary and close terminal admission before cleanup.
- [ ] Run the complete Python and Node suites.
- [ ] Start a two-generation browser run and verify generation 2 receives generation 1 reflection, does not repeat the failed mutation, and records real DSH workflow/session evidence.
