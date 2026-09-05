# 全局基准与自进化证据链实施计划

> 依据：`docs/superpowers/specs/2026-08-30-global-incumbent-evolution-evidence-design.md`

目标：在保留现有 champion–challenger v2 轨迹的基础上，建立真实 seed incumbent、探索轮语义、可恢复的三臂 closeout 和不误导的网页证据链，并从网页启动一代受控自进化验证。

工作方式：直接在当前 `main` 工作区分阶段实施；保留用户已有的 `data.js`、`smoke.mjs` 修改和运行时数据库文件。每个行为先写失败测试并观察失败，再写最小实现，阶段结束运行相关测试并提交。

## 阶段 1：中断恢复与运行构建身份

### 任务 1.1：定位最近 closeout `NameError`

- 对比故障时间附近提交中的 closeout 路径；
- 检查当时版本的未定义符号和当前版本的差异；
- 使用复制的运行数据库或纯状态夹具复现，不修改原数据库；
- 记录根因是代码缺陷、跨版本运行还是二者共同作用。

### 任务 1.2：用测试固定幂等 closeout

- 在 generation execution 测试中构造“三臂已完成、comparison 未记录”的状态；
- 首次执行只补 `GenerationComparisonRecorded` 与 `GenerationChampionSelected`；
- 再次执行不新增重复 holdout、comparison 或 selection 事件；
- 先观察测试失败，再实现最小修复。

### 任务 1.3：冻结 runtime build identity

- 为新运行记录构建身份和协议 schema；
- 恢复时验证兼容性，旧运行保持原协议；
- 添加同版本可恢复、明显不兼容版本拒绝的测试；
- 页面/接口返回可理解的暂停原因。

## 阶段 2：真实 seed incumbent control

### 任务 2.1：定义控制候选身份

- 增加向后兼容的 candidate role，默认 `search`，新增 `incumbent_control`；
- control 使用确定性 candidate/proposal/revision id；
- 从 `RunSeedGenomeMaterialized` 构建 R0，保留 genome 与摘要；
- control 不计入 `max_candidates`、四候选冻结和 finalist 排名。

### 任务 2.2：接入第 0 代三臂比较

- 第 0 代 incumbent binding 必须指向 seed control R0；
- 后续代继续指向上一代 effective revision；
- 删除“取本代第一个非 finalist 候选作为 incumbent”的兜底；
- 添加 genome digest 一致性测试和 control 不污染排名/计数测试。

## 阶段 3：探索轮与晋升约束

### 任务 3.1：持久化 screening 状态

- 冻结 generation 时记录通过数量、是否 `exploration_only`、连续探索轮计数；
- 全失败仍选两个相对候选作局部研究，但公开事件明确标记探索；
- 添加部分通过、全部失败两类测试。

### 任务 3.2：限制全局晋升与触发重新规划

- exploration-only challenger 默认不得晋升全局 champion；
- 达到连续失败阈值时给下一代 planner 明确的 replan 信号；
- champion selection reason 必须说明“保留 incumbent / 探索证据不足”；
- 添加晋升阻断和阈值触发测试。

## 阶段 4：投影与网页证据链

### 任务 4.1：扩展 API projection

- 输出 global champion 摘要；
- 输出 generation 三臂 holdout 决策、共同 cohort、delta、资格和回归原因；
- 输出局部 paired comparison 行；
- 输出 exploration-only、origin 容量和复用率。

### 任务 4.2：更新页面

- 新增全局 champion 卡、三臂决策卡、局部 paired 表；
- 旧 v1 “已应用”改为“已生成下一尝试，尚未同批验证”；
- 绝对分数、paired delta、holdout delta 分栏命名；
- 跨 cohort 历史分数不连成可比较趋势；
- 先补前端 smoke 断言，再实现 UI。

## 阶段 5：数据容量护栏

### 任务 5.1：安全默认值与提示

- 网页默认受控运行一代；
- 显示计划 origin 需求、可用 origin、预计/实际复用率；
- 多代且数据不足时显示“工程探索，不构成独立科学验证”；
- 为锁定最终验证集预留配置与 projection 字段，但不伪造未存在的数据。

## 阶段 6：验证与网页实跑

### 任务 6.1：分层验证

- 运行新增单元/集成测试；
- 运行 Python 全套、Node smoke 和仓库总验证；
- 自查用户已有未提交修改未被覆盖；
- 检查 git diff 只含计划内变更。

### 任务 6.2：网页启动一代自进化

- 启动本地服务并打开网页；
- 使用新运行而不是继续旧 v1 运行；
- 选择一代受控配置并启动；
- 观察事件至少覆盖 seed control、screening 状态和首个轨迹/holdout 阶段；
- 若运行可在合理时间完成，核验完整三臂决策；若依赖外部模型耗时，则保留运行继续并报告已验证到的阶段、运行 id 和未完成项。

## 完成判定

- 所有新增测试先红后绿；
- 第 0 代 seed digest 断言通过；
- closeout 重放无重复事件；
- exploration-only 不会误晋升；
- 页面不再用“已应用”描述未经同批验证的旧步骤；
- 网页端新运行正常启动，页面与事件语义一致；
- 完整验证命令在最终汇报前重新运行并保留最新输出。
