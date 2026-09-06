# AI for AI Plugin Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改 DSH 的前提下，增加可复现、可评测、可回滚的外部插件自进化控制平面。

**Architecture:** 新增 `ecologyrsi_dsh.evolution_lab` 包，维护能力注册表、不可变 PluginGenome、实验报告和 SQLite 账本；通过窄 `ExperimentBackend` 接口适配 DSH。第一版只自动执行已注册或声明式能力，任意生成代码不直接进入 DSH。

**Tech Stack:** Python 3.10+ 标准库、SQLite、unittest；不新增运行时依赖。

**Spec:** `docs/superpowers/specs/2026-09-06-ai-for-ai-plugin-evolution-design.md`

## Global Constraints

- 不修改 DSH HTTP 协议、DSH runtime、权限、数据分区和科学门禁。
- 不允许 Genome 包含任意代码、Shell、网络地址或动态导入。
- 候选只能改变一个主要 mutation axis，保证反馈可归因。
- 同 cohort 比较、可靠性门禁和回滚记录必须持久化。
- 现有测试全部保持通过；新增包必须有独立单元测试和 CLI smoke。

### Task 1: Capability Registry

**Files:**
- Create: `src/ecologyrsi_dsh/evolution_lab/capabilities.py`
- Create: `tests/test_evolution_lab_capabilities.py`

- [x] 写失败测试：未知 Skill、未授权 Tool、越界参数和可执行代码声明必须拒绝。
- [x] 实现不可变 `CapabilitySpec`、`CapabilityRegistry` 和声明式能力提案。
- [x] 为 Skill、Tool、Workflow、Algorithm 提供 digest 和角色/领域兼容性校验。
- [x] 运行能力注册表测试。

### Task 2: Plugin Genome and Mutation Planner

**Files:**
- Create: `src/ecologyrsi_dsh/evolution_lab/genome.py`
- Create: `tests/test_evolution_lab_genome.py`

- [x] 写失败测试：Genome digest 对字段顺序稳定、未知引用拒绝、单次变更超过一个轴拒绝。
- [x] 实现不可变 `PluginGenome`、`Mutation`、`MutationPlanner`。
- [x] 支持 Skill 选择、Skill 参数、Tool Policy、Workflow 参数和 Algorithm 选择五类有限变更。
- [x] 运行 Genome 测试。

### Task 3: Experiment Backend and Evaluator

**Files:**
- Create: `src/ecologyrsi_dsh/evolution_lab/evaluator.py`
- Create: `tests/test_evolution_lab_evaluator.py`

- [x] 写失败测试：不同 cohort 不可直接比较；负分但相对改进只能成为 search winner；物理违规和单元退化阻断晋级。
- [x] 实现 `EvaluationReport`、`PromotionDecision`、硬门禁和多目标排序。
- [x] 实现 `ExperimentBackend` Protocol 与确定性 `CallableBackend`。
- [x] 运行评测测试。

### Task 4: Durable Store and Controller

**Files:**
- Create: `src/ecologyrsi_dsh/evolution_lab/store.py`
- Create: `src/ecologyrsi_dsh/evolution_lab/controller.py`
- Create: `tests/test_evolution_lab_controller.py`

- [x] 写失败测试：候选、评测、晋级和回滚必须可重放，重复写入不能制造重复版本。
- [x] 实现 SQLite 账本和 `EvolutionController`。
- [x] 实现 exploratory、eligible、incumbent、rolled_back 状态转换。
- [x] 运行控制器测试。

### Task 5: DSH Adapter Boundary and CLI

**Files:**
- Create: `src/ecologyrsi_dsh/evolution_lab/adapters.py`
- Create: `src/ecologyrsi_dsh/evolution_lab/__main__.py`
- Create: `plugins/ai_evolution_lab/README.md`
- Create: `plugins/ai_evolution_lab/plugin.json`
- Modify: `pyproject.toml`
- Create: `tests/test_evolution_lab_cli.py`

- [x] 写失败测试：CLI 能创建基线、生成候选、使用确定性 backend 评测并输出 incumbent。
- [x] 实现 DSH adapter 协议和 CLI smoke，不修改 DSH 路由。
- [x] 将独立 CLI 暴露为 `ecologyrsi-ai-evolve` / `python -m ecologyrsi_dsh.evolution_lab`，支持 `--db`、`--cohort` 和 `--demo`。
- [x] 更新额外插件说明，明确 DSH 不变和 Tool adapter 审核边界。
- [x] 运行 CLI smoke。

### Task 6: Regression and Delivery

**Files:**
- Modify: `README.md`
- Modify: `RELEASE-CHECKLIST.md`

- [x] 增加 AI for AI 架构说明和运行示例。
- [x] 运行新增测试、`make test-fast`、`make test-integration`、`make verify`。
- [x] 运行 `git diff --check` 和残留引用扫描。
