# Evolution Quality Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce small effective mutations, eliminate misleading duplicate results, expose truthful DSH evidence, and reduce redundant sample-agent work.

**Architecture:** Python enforces genome and scientific contracts; DSH owns every non-trivial Agent decision and its context/workflow lifecycle. Deterministic single-choice routing is treated as execution policy rather than an Agent decision, and all historical manifests keep frozen semantics.

**Tech Stack:** Python 3.10+, unittest, Node.js ESM/node:test, DSH 0.1.0-rc.6, SQLite event ledger.

**Spec:** `docs/superpowers/specs/2026-08-21-evolution-quality-hardening-design.md`

## Global Constraints

- Do not alter reward, baseline, cohort, promotion, or dataset-partition formulas.
- Do not evolve DSH itself; only the registered plugin genome may change.
- Preserve all existing uncommitted user changes and do not restart the live service until code tests pass.
- Write and observe a failing regression test before each production change.
- Do not fabricate provider usage from TokenMeter pressure.
- Do not change frozen behavior of historical run manifests.

---

### Task 1: Effective mutation contract

**Files:**
- Modify: `src/ecologyrsi_dsh/evolution/genome.py`
- Modify: `src/ecologyrsi_dsh/evolution/strategies.py`
- Modify: `integrations/dsh_ecology_plugin/schemas/genome-mutation.schema.json`
- Test: `tests/test_evolution_genome.py`

**Interfaces:**
- Consumes: `apply_genome_mutation(parent, accepted_mutation, context, registry)`.
- Produces: a changed genome created by exactly one bounded trust-region operation or a `ValueError`.

- [ ] Add tests that pass 5 and 22 operations, an empty operation list, an unchanged parameter, an unchanged policy, and a change-then-revert sequence.
- [ ] Run `LC_ALL=en_US.UTF-8 PYTHONUTF8=1 PYTHONPATH=src .venv/bin/python -m unittest tests.test_evolution_genome -v`; the new cases must fail because the current Host permits them.
- [ ] Enforce `1 <= len(operations) <= 4`, reject per-operation no-ops, and compare the final genome with the parent before constructing provenance.
- [ ] Change the frozen mutation budget and package schema from 16/32 to 4.
- [ ] Re-run the focused tests and the DSH package-contract test.

### Task 2: Scientific ranking and sibling diversity

**Files:**
- Modify: `src/ecologyrsi_dsh/evolution/analysis.py`
- Modify: `src/ecologyrsi_dsh/evolution/strategies.py`
- Test: `tests/test_evolution_feedback_loop.py`
- Test: `tests/test_strategy_router.py`

**Interfaces:**
- Consumes: candidate rows with optional evaluation and the accepted candidates earlier in the same generation.
- Produces: ranks only for finite evaluated candidates and bounded sibling-avoid context for each later proposal.

- [ ] Add a mixed ranking test where a duplicate with no score sorts after evaluated rows and has no rank.
- [ ] Run the focused analysis test and observe the duplicate rank failure.
- [ ] Make evaluation availability the leading rank predicate and leave duplicate/failed rank unset in the public generation analysis.
- [ ] Add a strategy test whose second proposal initially compiles to the first sibling behavior, then returns a distinct mutation on retry.
- [ ] Add sibling behavior and parameter digests to later proposal context and bounded retry validation.
- [ ] Run both focused suites green.

### Task 3: Native research/reflection projection

**Files:**
- Modify: `src/ecologyrsi_dsh/presentation/reporting.py`
- Test: `tests/test_stage_projection.py`

**Interfaces:**
- Consumes: completed native research iterations containing `dsh_research_summary`, `dsh_research_evidence`, and `dsh_evolution_reflection`.
- Produces: completed `analysis_summary` and `final_plan` public projection.

- [ ] Add a reporting test using the actual native plan shape and assert completed status, summary, evidence, and reflection priorities.
- [ ] Run the focused test and observe the current `pending` projection.
- [ ] Normalize native and legacy field names into the existing public response without exposing unrestricted raw context.
- [ ] Re-run reporting and execution-projection tests green.

### Task 4: Workflow token observability

**Files:**
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/agents.js`
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js`
- Test: `integrations/dsh_ecology_plugin/test/token_semantics.test.mjs`
- Test: `integrations/dsh_ecology_plugin/test/stage_runner.test.mjs`

**Interfaces:**
- Consumes: a live DSH child Session or an already captured SessionProjection snapshot.
- Produces: `ecologyrsi-dsh.dsh-session-metrics/1` with provider usage when DSH reported it.

- [ ] Add a workflow test that removes the child from `ctx.sessions` after `workflow.result` but still expects persisted provider usage.
- [ ] Run the Node focused test and observe `provider_usage.available=false`.
- [ ] Allow metrics normalization from a captured session/snapshot and capture it at `workflow/agent-end` or the last public lifecycle point before disposal.
- [ ] Keep absence fail-soft and keep TokenMeter separate.
- [ ] Re-run token and stage-runner tests green.

### Task 5: Strict per-sample agent execution

**Files:**
- Modify: `src/ecologyrsi_dsh/evaluators/gateway_sample_adapter.py`
- Modify: `src/ecologyrsi_dsh/api/handler.py`
- Modify: `README.md`
- Test: `tests/test_sample_execution.py`
- Test: `tests/test_runtime_integration.py`

**Interfaces:**
- Consumes: legal registered tool catalogs and frozen `sample_remote_critic_policy`.
- Produces: one remote Planner and Critic call per effective sample, followed by one post-score remote Reflector call.

- [x] Add a sample-execution test proving a one-tool normal sample still invokes Planner and Critic.
- [x] Add post-score Reflector schema/stage tests and keep labels unavailable to Planner/Critic.
- [x] Disable the deterministic single-tool Host bypass for strict runs.
- [x] Freeze new real runs to one-sample prompts and `always@1` critic policy.
- [x] Persist chain attestations only after reflection so restart cannot skip an unproven call.
- [x] Update README wording and run focused sample/runtime tests.

### Task 6: Integrated verification and native smoke

**Files:**
- Verify: all modified files and existing candidate-parallel scheduler changes.

**Interfaces:**
- Consumes: the five completed task contracts.
- Produces: regression evidence and a bounded native evolution result.

- [ ] Run `LC_ALL=en_US.UTF-8 PYTHONUTF8=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`.
- [ ] Run `node --test integrations/dsh_ecology_plugin/test/*.test.mjs`.
- [ ] Run `node --test plugins/ecology_evolution/test/smoke.mjs`.
- [ ] Run `git diff --check` and a secret/path scan over tracked delivery files.
- [ ] Restart through the documented DSH entrypoint only after tests pass, create a small native run, and verify real DSH child evidence, non-duplicate effective deltas, truthful token availability, completed reflection projection, and terminal/retry state.
