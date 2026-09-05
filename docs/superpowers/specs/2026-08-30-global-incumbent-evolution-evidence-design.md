# 全局基准与自进化证据链设计

日期：2026-08-30
状态：已确认，作为 `2026-08-30-champion-challenger-adaptive-trajectory-design.md` 的增量设计

## 1. 目标

现有 champion–challenger 轨迹已经能在同一批次内比较局部候选，但仍需解决四个会让用户感觉“越进化越差”的问题：

1. 第 0 代的 incumbent 可能由首个普通候选兜底，不一定是运行开始时的真实种子；
2. 局部轨迹前进、全局 champion 晋升、历史绝对分数混在一起展示，负分时容易被误解为“已采用更差版本”；
3. 全部候选未通过筛选时，系统仍会挑出分数靠前者继续探索，却没有明确标成“探索轮”；
4. 进化运行跨代码版本或在三臂留出比较中断后，缺少足够明确的兼容性与幂等恢复边界。

本设计的核心原则是：局部探索可以容忍负的绝对分数，但任何全局替换都必须以同一留出集上的三臂比较为依据，并且页面必须把这两件事清楚地区分开。

## 2. 两级 champion

### 2.1 局部轨迹 champion

每个 finalist lane 都维护自己的当前版本。下一版仅与该 lane 的上一版在同一 formal batch 上成对比较。

- 允许绝对分数为负；
- 只有 paired delta 达到局部接受标准，才成为该 lane 下一批的父版本；
- 未达到标准的版本保留为审计记录，不能成为父版本；
- 页面使用“局部接受/局部拒绝”，不使用笼统的“已应用”。

### 2.2 全局 champion

每代结束时，使用完全相同的 generation holdout 对三个臂做比较：

1. finalist 1 的最终局部 champion；
2. finalist 2 的最终局部 champion；
3. 当前全局 incumbent。

只有满足全局晋升约束的 challenger 才能替换 incumbent。否则保留 incumbent。局部接受不等于全局晋升。

## 3. 第 0 代真实种子基准

运行创建时已经持久化 `RunSeedGenomeMaterialized`。新增一个显式、可重放的 incumbent control：

- 它绑定该事件中的规范化 seed genome 与摘要；
- 它不是四个搜索候选之一，不占候选预算，也不参加 finalist 排名；
- 它具有稳定的 candidate/revision 身份，以便复用现有评估、检查点和审计链；
- 第 0 代 holdout 的 incumbent 臂必须引用这个 seed control；
- 后续代 incumbent 臂引用上一代 `GenerationChampionSelected` 选出的有效 revision；
- 若旧运行缺少该 control，不猜测或用普通候选代替：旧 v1 运行继续作为审计记录，新 v2 运行在冻结输入时补建确定性的 seed control。

验收断言：第 0 代三臂比较中的 incumbent genome digest 必须等于 `RunSeedGenomeMaterialized` 的 seed genome digest。

## 4. 筛选失败与探索轮

当四个候选全部未通过 screening 时，本代进入 `exploration_only`：

- 仍可选择相对较优的两个候选开展局部轨迹，以收集方向性证据；
- screening 结果不能显示为成功；
- generation comparison 仍可运行，但 challenger 默认没有全局晋升资格，除非配置明确允许且所有科学与回归约束同时通过；
- 页面显示“本代为探索轮：无候选通过初筛”；
- 连续探索轮计数达到阈值时，下一代强制重新规划搜索方向，而不是沿用失败方向持续微调。

## 5. 证据指标与页面语义

必须分开呈现以下指标：

- `absolute_score`：单次评估绝对分数，只能在相同 cohort 上直接比较；
- `paired_delta`：局部 challenger 相对同批 lane champion 的差值；
- `holdout_delta`：同一代 holdout 上 challenger 相对 incumbent 的差值；
- `historical_score`：历史批次观测值，只用于趋势与审计，不用于跨 cohort 连线或直接排序。

网页端新增三类证据区域：

1. 全局 champion 卡：当前有效版本、来源代次、是否仍为初始种子；
2. 三臂 holdout 决策卡：三个臂的同批结果、资格、回归原因、最终选择；
3. 局部 paired comparison 表：每个 formal batch 的 incumbent、challenger、delta 与接受理由。

旧 v1 事件中的“已应用，待后续验证”改为“已生成下一尝试，尚未同批验证”，避免把生成/试验误写成采用。

## 6. 数据容量与科学结论边界

在当前可用 origin 数不足以覆盖五代完全不重复评估时：

- 网页默认先运行一代受控验证；
- 页面显示 cohort 重用比例与有效 origin 容量；
- 重用数据上的结果标记为工程探索证据，不宣称五代独立科学改进；
- 若要形成多代科学结论，应补充数据，或预先锁定从未参与搜索的最终验证集。

## 7. 运行版本与中断恢复

每个新运行冻结以下兼容信息：

- schedule id/version；
- runtime build identity；
- comparison/projection schema version。

恢复时：

- 同一兼容版本允许继续；
- 不兼容版本拒绝自动继续，并给出明确原因；
- 三个 holdout arm 已完成但 comparison/selection 尚未落盘时，重放 closeout 只补齐缺失事件，不重复运行已完成 arm；
- 每个步骤使用稳定 scope/checkpoint key，确保幂等。

已有 v1 长运行不自动升级为 v2，也不把旧证据重新解释为新协议证据。

## 8. 分阶段落地

1. 恢复安全与运行版本：先确保中断 closeout 可幂等完成，并记录运行构建身份；
2. 真实 seed incumbent：建立独立 control，修正第 0 代三臂基准；
3. 探索轮语义：持久化筛选状态和连续失败计数，控制全局晋升资格；
4. 页面证据链：全局卡、三臂卡、局部 paired 表和旧语义修正；
5. 容量护栏：一代默认值、重用率提示和最终验证集边界；
6. 端到端验证：从网页启动新 v2 运行，确认 seed 基准、局部比较、三臂决策与恢复行为。

## 9. 验收标准

- 第 0 代 incumbent 是真实 seed，而不是任意搜索候选；
- 被局部拒绝的 revision 永不成为下一批父版本；
- 三个 holdout arm 使用完全相同的 cohort；
- 三臂完成后的中断可在不重跑 arm 的情况下补齐 comparison 和 champion selection；
- 全部筛选失败时页面明确显示探索轮，且不伪装为成功；
- 负绝对分数不会再显示成含义不明的“已应用”；
- 页面能回答“当前全局最好是谁、为什么仍保留或替换、比较是否同批”；
- 从网页启动的一代受控运行能够稳定进入并完成相应阶段，事件和页面一致。
