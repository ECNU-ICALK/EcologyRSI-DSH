# 正向增益搜索版本与稳健认证分层设计

日期：2026-08-31
状态：用户已批准，作为 `2026-08-30-global-incumbent-evolution-evidence-design.md` 的增量设计

## 1. 目标

新运行必须把“下一轮从哪个版本继续搜索”和“哪个版本已经获得严格科学认证”分成两个结论：

- 只要挑战版本在同一冻结 cohort、同一评分合同下取得严格正的总体增益，并通过完整性、覆盖率、严格执行链和物理约束检查，它就成为下一轮搜索版本；
- `0.005` 实用效应阈值、统计稳定性、绝对科学基线和逐 cell 回归继续记录，但只决定稳健认证，不再阻断搜索版本前进；
- 旧 prequential 运行和现有 runtime-v2 运行保持原始语义，不被重放成新规则。

这使搜索能够积累整体改善，同时保留对温度、湿度、CO2 及 1/6/24 小时单元风险的完整披露。

## 2. 版本与兼容边界

新建 DSH-native 运行冻结 `ecologyrsi-dsh.evolution-runtime/3`。运行时规则如下：

- 无 `host_runtime_build` 的历史运行：按原有审计/兼容路径重放；
- runtime-v2：保留 `delta > 0.005 + 科学门禁 + cell 门禁 + 稳定性` 的旧选择语义；
- runtime-v3：使用本设计的正向增益搜索规则；
- 恢复运行时不得自动改写 runtime schema，也不得重新解释已持久化比较事件。

generation comparison 和 projection 保持现有外层 schema，通过明确的 policy/status 字段扩展，不伪装为旧语义。

## 3. 局部 lane champion

runtime-v3 的同 cohort 局部比较按以下顺序决策：

1. champion 与 challenger 的 run、generation、candidate、batch、cohort、origin 数量、evaluator digest 和评分合同完全一致；
2. 两侧目标 × 时距网格完整且数值有限；
3. challenger 覆盖率、严格执行链和物理约束通过；
4. `score_delta = challenger.score - champion.score > 1e-12`。

满足即将 challenger 设为 lane champion。否则保留 champion。

逐 cell delta 仍计算并持久化：

- `cell_regression_gate_passed` 改为风险诊断，不再是 runtime-v3 的局部阻断项；
- runtime-v2 仍按旧规则把 cell 回归作为阻断项；
- `minimum_score_delta` 持久化实际采用的阈值，使回放能够验证选择规则。

## 4. 代际搜索版本与稳健认证

每个 finalist 相对同一 holdout incumbent 产生两类资格。

### 4.1 搜索资格

runtime-v3 的 `search_eligible` 仅要求：

- 三臂绑定同一 cohort、同一 evaluator 和完整评分合同；
- candidate 与 incumbent 的目标 × 时距网格完整；
- candidate 总体/逐 cell 覆盖达到冻结要求；
- 严格执行链通过；
- 无物理约束违规；
- 总体 `delta > 1e-12`。

绝对 score 可以为负。若两个 finalist 都满足，选择总体 delta 最大者；稳定性下界、最差 cell delta、候选/修订 ID 仅作确定性次级排序。

`exploration_only` 不再阻断 runtime-v3 搜索版本，但继续阻断稳健认证并在页面显示风险。

### 4.2 稳健认证资格

`certification_eligible` 保留旧严格规则：

- 固定科学门禁通过；
- 无约束违规且覆盖完整；
- 目标 × 时距网格完整；
- 最差 cell delta 不低于 `-0.01`；
- max-T/移动区块稳定性通过；
- 总体 delta 大于冻结的 `0.005` 实用效应阈值；
- 非 exploration-only。

comparison 持久化：

- 每臂 `search_eligible`、`certification_eligible`；
- `search_failures`、`certification_failures`；
- 搜索选择 arm/revision；
- 严格认证候选 arm/revision（若存在）；
- 被选搜索版本的认证状态。

`CandidateEffectiveRevisionFrozen` 和后续代 parent 绑定搜索选择结果。网页不得再将它笼统称为“正式最优版本”。

## 5. 宿主提案合法性

局部编辑上下文必须向模型提供基于当前值和 `0.15` 归一化信赖域计算出的合法邻域。模型仍可返回非法值，但 Host 必须：

- 不创建 child revision；
- 返回稳定原因码 `proposal_host_validation_failed`；
- 保存经过截断和公开化处理的具体原因，例如 `normalized trust-region step 0.8 exceeds maximum 0.15`；
- 将原因写入 `LocalEditDecided.reason`，供恢复、反思和网页展示；
- 不静默裁剪模型建议值，因为静默裁剪会改变提案语义。

## 6. 页面语义

页面按以下层次展示：

1. 本 cohort 绝对分：只描述该数据窗口；
2. 同 cohort 局部 delta：决定 lane champion；
3. 同 holdout 代际 delta：决定下一轮搜索版本；
4. 稳健认证：显示科学门禁、稳定性和最差 cell 风险；
5. 提案合法性：与效果校验分列，显示具体 Host 拒绝原因。

文案要求：

- “当前全局冠军”改为“当前下一轮搜索版本”；
- 认证通过时显示“稳健认证通过”；否则显示“已进入下一轮搜索，稳健认证未通过”；
- runtime-v1 显示“旧版连续更新审计，跨 cohort 不可归因”；
- runtime-v2 显示“旧版严格 paired 门禁”；
- runtime-v3 显示“正向增益搜索 + 独立稳健认证”。

## 7. 研究合同可靠性

`research iteration plan exceeds the bounded contract` 与评分选择独立处理。语义修复预算耗尽后：

- 不重放已经完成的 generation 工作；
- 持久化结构化失败详情；
- 使用 Host 拥有的最小合法研究计划作为一次确定性 fallback；
- fallback 只能引用已经冻结的上一代弱点、失败单元和合法 mutation catalog；
- fallback 也无法构造时才暂停，并明确标记 `research_contract_fallback_unavailable`。

## 8. 验收标准

- runtime-v3 中，绝对分为负但同 cohort `delta > 1e-12` 的 challenger 成为局部 champion；
- runtime-v3 中，某 cell 退化但总体 delta 为正的 challenger 进入下一轮搜索，并带风险警告；
- `delta <= 1e-12`、比较合同不一致、网格不完整、覆盖不足、严格链失败或约束违规时保留 incumbent；
- runtime-v2 事件仍按原严格规则重放；
- 两个 finalist 都改善时选择总体 delta 最大者；
- 下一代 parent digest 等于上一代搜索选择 revision digest；
- Host 越界提案持久化具体原因且不创建 child；
- 页面明确区分绝对分、局部 delta、代际 delta、搜索版本和认证状态；
- 研究合同 fallback 不重复已完成工作，并且可在重启后确定性重放。
