# AI for AI 外部插件自进化设计

## 目标

在不修改 DSH 协议、运行时、权限和审计边界的前提下，增加一个独立的插件自进化控制平面。该控制平面能够把生态数据实验转化为可复现的插件候选版本，比较 Skill、Tool Policy、Workflow 和算法配置的效果，并仅将经过同 cohort 科学评测和可靠性门禁的版本晋级。

## 核心原则

1. DSH 是稳定底座：负责执行、权限、数据边界、事件、投影和审计；自进化逻辑不写入 DSH。
2. 插件版本是不可变 Genome：Skill、Tool、Workflow、Algorithm 和全部 digest 一起版本化。
3. 模型只能提出结构化候选；插件控制平面负责校验、实验、比较、晋级和回滚。
4. 新 Skill 可以以文本/模板候选进入实验池；新 Tool 只能以声明式规格或已审核的 Host adapter 进入执行池，任意代码不得直接进入 DSH。
5. 先过安全与科学硬门禁，再比较同 cohort 改进；绝对负分的候选最多成为搜索父版本，不能直接成为生产 incumbent。

## 架构

```text
生态数据
   ↓
AI Evolution Lab（外部插件）
   ├─ Capability Registry：Skill / Tool / Workflow / Algorithm
   ├─ Plugin Genome：不可变候选版本
   ├─ Mutation Planner：一次只改变一个可归因轴
   ├─ DSH Adapter：调用既有 DSH API，不修改 DSH
   ├─ Evaluator：同 cohort、科学门禁、可靠性和成本指标
   ├─ Experiment Store：候选、证据、晋级、回滚账本
   └─ Promotion Controller：exploratory → eligible → incumbent
   ↓
+DSH 原生运行 / 事件 / 公共投影
```

## Genome

每个候选版本至少包含：

- `skills`：角色到 Skill 模板、版本和有限参数的映射；
- `tools`：角色可用工具、路由顺序、重试和回退策略；
- `workflow`：已注册工作流模板及有界参数；
- `algorithm`：预测器、特征、拟合、不确定性策略和参数；
- `parent_digest`、`cohort_policy_digest`、`capability_digest`；
- `created_by`、`mutation_axis`、`evidence_refs`。

Genome 不允许包含数据分区、科学门禁、权限范围、任意代码或可执行命令。

## 评测和晋级

评测结果由四类指标组成：

- 科学：综合技能分、每个目标/时距的分数、物理约束违规、holdout 稳定性；
- 能力可靠性：Skill 合同通过率、Tool 成功率、无效调用率、修复成功率；
- 运行代价：延迟、重试、模型调用和 Tool 调用次数；
- 证据：cohort digest、样本覆盖、独立评审和失败归因完整性。

晋级采用以下顺序：

1. 结构和安全门禁；
2. 科学硬门禁；
3. 与 incumbent 在相同 cohort 上的实际改进 `delta > practical_delta`；
4. 各目标/时距不得出现不可接受退化；
5. 稳定性和可靠性满足阈值；
6. 记录新 incumbent，并保留父版本作为回滚点。

## DSH Adapter

控制平面只依赖一个窄接口：

```python
class ExperimentBackend(Protocol):
    def evaluate(self, genome: PluginGenome, cohort_id: str) -> EvaluationReport: ...
```

生产适配器通过已有 DSH HTTP API 创建运行、等待终态、读取投影和事件；测试适配器可以使用确定性函数。这样 DSH 不需要知道 Genome、Mutation 或 Promotion。

## 失败归因

每次候选实验都生成 `failure_attribution`：`skill`、`tool`、`workflow`、`algorithm`、`data` 或 `infrastructure`。下一轮 Mutation 只针对归因层生成方向，避免工具错误被误调成算法参数。

## 交付范围

首个版本实现纯 Python、无新增依赖的 Registry、Genome、SQLite 账本、评测决策器、可插拔 Backend 和命令行演示；不修改 DSH HTTP 路由和 DSH runtime。任意新 Tool 的实际代码接入仍需通过已审核的 adapter 注册。
