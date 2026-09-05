# Positive Delta Search Version Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让新运行在同 cohort 总体 delta 严格为正时推进下一轮搜索版本，同时保留独立严格认证、完整风险证据和旧运行重放兼容性。

**Architecture:** 以 runtime schema v3 作为行为边界；局部和代际比较各自产生搜索资格与认证诊断，现有 effective revision 绑定搜索版本。Host 编辑验证返回结构化原因，projection/UI 分层展示；研究合同失败使用独立确定性 fallback。

**Tech Stack:** Python 3.10–3.12、stdlib dataclass/unittest、append-only event ledger、Vanilla JavaScript/Node smoke、DSH structured roles。

**Spec:** `docs/superpowers/specs/2026-08-31-positive-delta-search-version-design.md`

## Global Constraints

- 直接在当前 `main` 工作区增量实施，不覆盖现有未提交的 runtime-v2、seed incumbent、projection 和 UI 改动。
- 不迁移、不重写旧运行；无 runtime build、runtime-v2 和 runtime-v3 必须按各自冻结语义重放。
- 搜索提升阈值为严格 `delta > 1e-12`；`delta == 1e-12` 不通过。
- 覆盖率、严格执行链、完整比较合同和物理约束仍是搜索硬门。
- 逐 cell 回归、绝对科学门、统计稳定性和 `0.005` 仅决定 runtime-v3 稳健认证。
- 采用 unittest runner；仓库 `.venv` 不提供 pytest。
- 当前共享主分支不自动提交，避免把既有未提交工作错误分组；完成后由用户决定提交边界。

---

### Task 1: Freeze runtime-v3 policy boundary

**Files:**
- Modify: `src/ecologyrsi_dsh/api/handler.py`
- Modify: `src/ecologyrsi_dsh/core/state.py`
- Test: `tests/test_dsh_native_runtime.py`
- Test: `tests/test_trajectory_event_replay.py`

**Interfaces:**
- Produces: `EVOLUTION_RUNTIME_SCHEMA_V3`, `uses_positive_delta_search_protocol(task)`。
- Preserves: `uses_global_incumbent_protocol(task)` 对 v2/v3 均成立。

- [ ] **Step 1: Write failing compatibility tests**

新增断言：新运行冻结 `/3`；v2 仍被 replay helper 识别为 global incumbent 但不启用 positive-delta；v3 同时启用两者。

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=src .venv/bin/python -m unittest -v tests.test_dsh_native_runtime tests.test_trajectory_event_replay`

Expected: FAIL，因为当前 Host 仍冻结 `/2` 且没有 v3 policy helper。

- [ ] **Step 3: Implement minimal version helpers**

新增 v3 常量和精确 helper；兼容校验只允许当前新建 v3，历史 ledger 的 v2 由 replay task 自身冻结规则验证。

- [ ] **Step 4: Run GREEN**

重复 Step 2，预期 PASS。

### Task 2: Promote any safe positive local challenger

**Files:**
- Modify: `src/ecologyrsi_dsh/evolution/champion_challenger.py`
- Modify: `src/ecologyrsi_dsh/api/formal_trajectory.py`
- Modify: `src/ecologyrsi_dsh/core/director.py`
- Modify: `src/ecologyrsi_dsh/core/state.py`
- Test: `tests/test_champion_challenger.py`
- Test: `tests/test_formal_trajectory.py`
- Test: `tests/test_trajectory_event_replay.py`

**Interfaces:**
- Produces: policy-aware `assess_local_challenger(..., minimum_score_delta, cell_regression_blocks)`。
- Persists: comparison `minimum_score_delta` and diagnostic `cell_regression_gate_passed`。

- [ ] **Step 1: Write failing behavior tests**

覆盖：v3 `+1e-6` 晋级；`+1e-12` 不晋级；总体正但一个 cell 退化仍晋级；v2 相同 cell 回归仍保留 champion；重放伪造 policy 被拒绝。

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=src .venv/bin/python -m unittest -v tests.test_champion_challenger tests.test_formal_trajectory tests.test_trajectory_event_replay`

Expected: FAIL，当前固定使用 `0.005` 和逐 cell veto。

- [ ] **Step 3: Implement policy-aware local comparison**

仅 runtime-v3 采用 epsilon 和诊断性 cell gate；runtime-v2 保持原结果及 reason code。

- [ ] **Step 4: Run GREEN**

重复 Step 2，预期 PASS。

### Task 3: Split generation search selection from strict certification

**Files:**
- Modify: `src/ecologyrsi_dsh/evaluators/generation_comparison.py`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py`
- Modify: `src/ecologyrsi_dsh/evolution/analysis.py`
- Modify: `src/ecologyrsi_dsh/core/state.py`
- Test: `tests/test_generation_comparison.py`
- Test: `tests/test_evolution_feedback_loop.py`
- Test: `tests/test_trajectory_event_replay.py`

**Interfaces:**
- Produces per arm: `search_eligible`, `certification_eligible`, `search_failures`, `certification_failures`。
- Produces comparison: `selection_policy`, `certification_selected_arm`, `selected_search_certification_status`。
- Consumes: runtime-v3 policy helper from Task 1。

- [ ] **Step 1: Write failing comparison tests**

覆盖：负绝对分但 `delta>0` 被选；8/9 cell 改善且一 cell 回归仍被选但未认证；两个正提升选择 delta 最大者；覆盖/约束/完整性失败仍保留 incumbent；v2 仍严格拒绝。

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=src .venv/bin/python -m unittest -v tests.test_generation_comparison tests.test_evolution_feedback_loop tests.test_trajectory_event_replay`

Expected: FAIL，当前 `eligible` 同时绑定科学、cell、稳定性和 `0.005`。

- [ ] **Step 3: Implement dual gates and deterministic ranking**

runtime-v3 用 search gate 选择 `GenerationComparison.selected_revision_id`；strict certification 继续使用旧组合条件。analysis 把 effective revision 明确解释为 search parent。

- [ ] **Step 4: Run GREEN**

重复 Step 2，预期 PASS。

### Task 4: Persist exact Host proposal rejection evidence

**Files:**
- Modify: `src/ecologyrsi_dsh/evolution/local_edits.py`
- Modify: `src/ecologyrsi_dsh/api/formal_trajectory.py`
- Modify: `src/ecologyrsi_dsh/evolution/genome.py`
- Test: `tests/test_local_edits.py`
- Test: `tests/test_formal_trajectory.py`

**Interfaces:**
- Produces: `LocalEditResult.rejection_reason` and bounded legal parameter neighborhood in local editor context。
- Persists: `LocalEditDecided.reason="proposal_host_validation_failed: ..."`。

- [ ] **Step 1: Write failing rejection tests**

以 `co2_concentration_residual_scale: 0→0.8` 复现；断言 child 不存在、reason 包含 `0.8` 和 `0.15`、上下文合法邻域上界为 `0.15`。

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=src .venv/bin/python -m unittest -v tests.test_local_edits tests.test_formal_trajectory`

Expected: FAIL，当前异常被折叠成无 reason 的 `REJECTED`。

- [ ] **Step 3: Implement bounded public rejection details**

捕获 Host `TypeError/ValueError`，生成稳定前缀和最多 240 字符公开详情；不捕获 parent digest invariant。向模型暴露合法邻域但不静默裁剪。

- [ ] **Step 4: Run GREEN**

重复 Step 2，预期 PASS。

### Task 5: Project and render search/certification semantics

**Files:**
- Modify: `src/ecologyrsi_dsh/api/projection.py`
- Modify: `src/ecologyrsi_dsh/api/events.py`
- Modify: `plugins/ecology_evolution/assets/js/render_process.js`
- Modify: `plugins/ecology_evolution/assets/js/render_shell.js`
- Modify: `plugins/ecology_evolution/styles.css`
- Test: `tests/test_execution_projection.py`
- Test: `tests/test_trajectory_public_events.py`
- Test: `plugins/ecology_evolution/test/smoke.mjs`

**Interfaces:**
- Consumes Task 2/3 comparison and rejection fields。
- Produces separate search version, holdout delta, certification status and Host validation UI fields。

- [ ] **Step 1: Write failing projection and smoke assertions**

断言 runtime-v3 文案、搜索版本卡、认证风险、cell regression 警告、具体 Host reject reason；旧 v1/v2 文案保持准确。

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=src .venv/bin/python -m unittest -v tests.test_execution_projection tests.test_trajectory_public_events && node plugins/ecology_evolution/test/smoke.mjs`

Expected: FAIL，当前页面仍以 global champion/eligible 混合展示。

- [ ] **Step 3: Implement projection and UI**

按规格分层渲染，不使用跨 cohort 折线推断效果。

- [ ] **Step 4: Run GREEN**

重复 Step 2，预期 PASS。

### Task 6: Add deterministic research-contract fallback

**Files:**
- Modify: `src/ecologyrsi_dsh/knowledge/research_iteration.py`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py`
- Modify: `src/ecologyrsi_dsh/api/auto_progress.py`
- Test: `tests/test_research_iteration.py`
- Test: `tests/test_auto_progress.py`
- Test: `tests/test_autonomous_search_reflection_cycle.py`

**Interfaces:**
- Produces: Host-built bounded fallback research plan from durable previous-generation evidence。
- Changes: exhausted semantic repair first attempts the same deterministic fallback; only unavailable fallback pauses。

- [ ] **Step 1: Write failing recovery tests**

复现 `research iteration plan exceeds the bounded contract`；断言 fallback 产生合法计划、不重跑已完成工作、重启后 digest 一致；无可用证据时才暂停为 `research_contract_fallback_unavailable`。

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=src .venv/bin/python -m unittest -v tests.test_research_iteration tests.test_auto_progress tests.test_autonomous_search_reflection_cycle`

Expected: FAIL，当前修复预算耗尽后直接 `research_contract_retry_exhausted` 暂停。

- [ ] **Step 3: Implement minimal deterministic fallback**

只使用冻结 mutation catalog 和 durable reflection/weakness evidence；不调用新的模型请求。

- [ ] **Step 4: Run GREEN**

重复 Step 2，预期 PASS。

### Task 7: End-to-end verification

**Files:**
- Verify all modified source, test, spec and plan files。

**Interfaces:**
- Validates all prior tasks together。

- [ ] **Step 1: Run focused Python tests**

Run: `PYTHONPATH=src .venv/bin/python -m unittest -v tests.test_optimization_schedule tests.test_champion_challenger tests.test_generation_comparison tests.test_local_edits tests.test_formal_trajectory tests.test_evolution_feedback_loop tests.test_trajectory_event_replay tests.test_execution_projection tests.test_trajectory_public_events tests.test_research_iteration tests.test_auto_progress tests.test_autonomous_search_reflection_cycle tests.test_dsh_native_runtime`

- [ ] **Step 2: Run browser smoke**

Run: `node plugins/ecology_evolution/test/smoke.mjs`

- [ ] **Step 3: Run repository verification**

Run: `make test && make verify`

- [ ] **Step 4: Inspect final diff and runtime artifacts**

Run: `git status --short && git diff --check && git diff --stat`

确认没有修改 `.runtime` 数据库、没有覆盖用户已有改动、没有未解释的新文件。
