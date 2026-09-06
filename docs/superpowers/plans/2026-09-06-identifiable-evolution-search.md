# 可辨识定向自进化实现计划

> **For agentic workers:** 本计划已在当前主工作区按用户“开始解决以上问题”授权直接执行；每个任务均先写回归测试，再实现最小改动。

**Goal:** 修复兼容模型路径的候选同质化和反馈不可归因问题，并把弱目标与参数可辨识性纳入可审计输出。

**Architecture:** 保持 DSH 协议、运行时和正式晋级门禁不变，在策略宿主层增加单轴实验设计，在分析层增加常量轴可辨识性，在投影层输出有限审计字段。

**Tech Stack:** Python 3.12、标准库、unittest；不新增运行时依赖。

**Spec:** `docs/superpowers/specs/2026-09-06-identifiable-evolution-search-design.md`

## Global Constraints

- 正式版本必须继续通过科学门禁、同 cohort 比较和独立评审。
- 模型仍只能提出有界参数；宿主负责投影、去重和执行。
- 审计输出只能包含有限聚合参数，不泄露样本明细或私有错误文本。
- 不修改 DSH HTTP 协议、权限和原生运行时边界。

### Task 1: 可辨识候选设计

**Files**
- Modify: `src/ecologyrsi_dsh/evolution/strategies.py`
- Test: `tests/test_strategy_router.py`

- [x] 多候选兼容模型提案共享参考参数。
- [x] 每个槽位只改变一个参数轴并覆盖不同轴。
- [x] 记录目标焦点、远程变化、宿主投影和兄弟去重证据。
- [x] 回归测试重复多参数模型输出仍生成四个唯一候选。

### Task 2: 弱目标反馈

**Files**
- Modify: `src/ecologyrsi_dsh/evolution/strategies.py`
- Test: `tests/test_strategy_router.py`

- [x] 从上一轮聚合弱点提取 target_focus。
- [x] 将 target_focus 传给策略模型和候选设计审计。
- [x] 回归测试验证 6 小时室内气温弱点被传递。

### Task 3: 参数可辨识性分析

**Files**
- Modify: `src/ecologyrsi_dsh/evolution/analysis.py`
- Create: `tests/test_evolution_analysis.py`

- [x] 常量参数轴输出 unidentifiable，而不是静默丢弃。
- [x] 保持 insufficient_evidence 对不可辨识轴的保护。
- [x] 回归测试验证 constant axis 不产生伪因果关联。

### Task 4: 可解释投影

**Files**
- Modify: `src/ecologyrsi_dsh/presentation/trajectory.py`
- Modify: `src/ecologyrsi_dsh/presentation/reporting.py`
- Test: `tests/test_stage_projection.py`

- [x] 向候选轨迹和轮次候选行输出 bounded search_design_audit。
- [x] 过滤完整参考参数和非聚合内容。
- [x] 运行投影回归测试。

### Task 5: 集成验证

**Files**
- No additional source files.

- [x] 运行策略、分析、投影和全量 Python 测试。
- [x] 运行编译、交付检查和 Node 集成测试。
- [x] 用 4 候选真实温室数据启动一次演化，确认参数轴覆盖、弱目标焦点和晋级安全性。

验证记录：全量 Python 测试 1321 项通过（4 项按环境跳过）；`make verify` 与
`make test-integration` 均通过；真实 `agc_cucumber_2018` 运行完成 4 个候选，
四个候选均通过模型提案、训练、评估和评审阶段，但因科学门禁未通过而全部保留为
搜索证据，没有晋级正式版本。候选审计显示共享参考参数为
`history_steps=3, residual_scale=0.5, ridge_alpha=0.01`，并覆盖
`history_steps`、`ridge_alpha`、`residual_scale` 三个轴；运行结束后识别出最弱目标为
室内气温 6 小时时距，正式晋级状态保持不变。
