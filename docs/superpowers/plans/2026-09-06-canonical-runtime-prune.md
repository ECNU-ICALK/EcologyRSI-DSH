# Canonical Runtime Prune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收敛 EcologyRSI-DSH 为单一 DSH-native 正式运行链，删除未使用代码和旧协议入口，同时保留科学评测、正向增益搜索、恢复幂等与审计能力。

**Architecture:** 服务端只接受 canonical `/api/ecology-evolution` 路径和 `research_compile_evolve@1` 工作流；历史协议不再作为新运行入口。浏览器只读服务端连续运行投影，不再重复发起旧版自动推进；静态 `demo=1` 继续作为显式演示入口。投影与合同测试只保留实际生产调用的实现。

**Tech Stack:** Python 3.10+ 标准库、SQLite、Node.js 原生测试、Vanilla JavaScript；不新增运行时依赖。

**Spec:** `docs/superpowers/specs/2026-08-31-positive-delta-search-version-design.md` and the approved cleanup boundary in the current conversation.

## Global Constraints

- 不改变同 cohort 比较、正向增益搜索、独立稳健认证、物理约束和恢复幂等语义。
- 不删除 `.runtime` 或历史数据库数据；仅停止旧协议运行入口。
- `demo=1` 静态演示保留，但不得被真实页面默认加载。
- 所有删除必须有调用链证据；没有证据的算法预测器和科学门禁不删除。
- 每个行为变更先写失败测试，再实现，再运行受影响测试和完整回归。

---

### Task 1: 收敛 HTTP 与目录入口

**Files:**
- Modify: `src/ecologyrsi_dsh/api/transport.py`, `src/ecologyrsi_dsh/api/handler.py`, `src/ecologyrsi_dsh/api/catalog.py`, `src/ecologyrsi_dsh/api/run_projection.py`
- Modify: `plugins/ecology_evolution/assets/js/host.js`, `plugins/ecology_evolution/assets/js/core.js`, `plugins/ecology_evolution/plugin.json`
- Test: `tests/test_http.py`, `tests/test_api_contracts.py`, `plugins/ecology_evolution/test/smoke.mjs`

**Interfaces:** canonical API base is `/api/ecology-evolution`; catalog exposes only `research_compile_evolve@1`; configuration uses one canonical candidate-count key and runtime-v3 default.

- [x] 为别名路径、旧工作流和旧字段写失败断言。
- [x] 删除浏览器公开的 `/api/v1`、`/api/v2`、`/api/ecology-evolution/v1` 路径回退及 `legacy_component_search@1` 目录项；`/api` 仅保留为本地 sidecar 内部根，不再写入浏览器清单。
- [x] 生产目录只消费后端 `dsh_models` 和角色字段；显式 `demo=1` 的静态夹具仍允许使用 `models`，不进入真实运行链。
- [x] 将新运行默认协议固定为 DSH-native，历史投影继续只读展示但不能创建新运行。
- [x] 运行 HTTP 合同和前端 smoke 测试。

### Task 2: 删除未接入生产的包装和死代码

**Files:**
- Delete: `src/ecologyrsi_dsh/api/evidence_projection.py`, `src/ecologyrsi_dsh/api/progress_projection.py`
- Modify: `tests/test_projection_contract.py`, `plugins/ecology_evolution/assets/js/catalog.js`

- [x] 先断言实际 projection 直接提供详情和进度字段，而不是导入两个包装模块。
- [x] 删除两个仅被合同测试导入的包装模块，并把测试改为调用生产 projection。
- [x] 删除无调用的 `predictionOriginsPerUpdateMaximum()`。
- [x] 运行 projection 合同测试和前端 smoke。

### Task 3: 移除浏览器旧版自动推进写路径

**Files:**
- Modify: `plugins/ecology_evolution/assets/js/commands.js`, `plugins/ecology_evolution/assets/js/data.js`, `plugins/ecology_evolution/assets/js/app.js`, `plugins/ecology_evolution/test/smoke.mjs`

- [x] 添加失败测试：服务端连续运行时页面不调用 `advance`，只启动只读 monitor。
- [x] 删除旧 `autoAdvance*` timer/retry 写路径及重复的 visibility/selection 清理。
- [x] 保留暂停、恢复、停止命令的幂等收据和超时核对逻辑。
- [x] 运行前端 smoke，并检查演示模式仍可浏览。

### Task 4: 断开旧网关运行入口并保留科学执行核心

**Files:**
- Modify: `src/ecologyrsi_dsh/api/catalog.py`, `src/ecologyrsi_dsh/application/cli.py`, `src/ecologyrsi_dsh/api/runtime.py`, `src/ecologyrsi_dsh/api/handler.py`
- Modify: `tests/test_api_contracts.py`, `tests/test_recoverable_evaluation_integration.py`
- Preserve: `src/ecologyrsi_dsh/evaluators/generation_comparison.py`, `src/ecologyrsi_dsh/evaluators/fitness.py`, `src/ecologyrsi_dsh/evolution/champion_challenger.py`

- [x] 先添加失败断言：无 DSH-native runtime 时不能把旧 gateway workflow 暴露为正式可运行能力。
- [x] 从服务启动和 catalog 移除旧 gateway 作为正式 execution mode；历史评测与内部重试适配器保留，供既有科学回放和单元测试使用，不再作为浏览器新运行入口。
- [x] 删除新运行对 `allow_host_fallback`/`legacy_read_only` 的默认开启，不删除历史事件解释器。
- [x] 运行 API、执行控制、恢复和评测回归测试。

### Task 5: 文档、打包清单与交付验证

**Files:**
- Modify: `README.md`, `plugins/ecology_evolution/README.md`, `MANIFEST.in`, `scripts/verify_delivery.sh`, `scripts/verify_artifacts.py`, `Makefile`
- Test: `tests/test_delivery_scripts.py`, `tests/test_http.py`, `plugins/ecology_evolution/test/smoke.mjs`

- [x] 删除文档中旧 API 前缀、旧工作流和“兼容入口”描述，明确 demo 需显式 `demo=1`。
- [x] 确认打包不包含删除模块、旧运行资产或未使用入口。
- [x] 运行 `node plugins/ecology_evolution/test/smoke.mjs`、`make test-fast`、`make test-integration` 和 `make verify`。
- [x] 检查 `git diff --check`、未提交文件清单及源码/测试引用残留。
