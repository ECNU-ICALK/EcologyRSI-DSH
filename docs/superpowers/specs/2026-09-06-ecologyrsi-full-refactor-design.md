# 目标架构：EcologyRSI-DSH 整体重构规格

> **这是目标规格，不是当前代码完成声明。** 当前可运行范围以 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) 为准。
> 参考包用独立命名空间 `ecologyrsi_kernel` 避免覆盖上游；目标整合目录仍是 `ecologyrsi_dsh`。迁移时按 [CURRENT_CODE.md](CURRENT_CODE.md) 的模块映射提取，不整包复制成第二个永久内核。
> 以下章节保留全局编号，便于跨文件引用。第 16—17、20—22 节在 [MIGRATION.md](MIGRATION.md)，第 18 节在 [TESTING.md](TESTING.md)，第 19 节在 [CODEX_PLAN.md](CODEX_PLAN.md)。

**已核对的上游基线：** `af265fc198518b5a17258f91cfec1eee8f27bbe9`，核对日期 2026-09-06。不是对执行时本地 HEAD 的假设；先做本地差异检查。

**架构决策：** 模块化单体、一个可信内核、对象层与方法层两类进化、可替换 Batch/DSH 执行后端。先打通严格的实验闭环，再扩展研究策略、技能和受控程序生成。

**验证边界：** 本包数值演示、参考代码和测试已运行；上游完整测试、真实 DSH、真实 AGC 数据、正式确认与部署均未在本次环境运行。本文不能作为这些事项已通过的证据。

## 1. 重构不是换目录：要改变哪几件事

### 1.1 本次源码核对结论

以下只用于定位迁移入口，不代表要机械保留旧实现。

| 位置 | 已核对的事实 | 重构决策 |
|---|---|---|
| `evolution_lab/evaluator.py` | 外部报告可直接带入总分、单元分数与 `stable`；数值、覆盖和报告绑定验证不完整。[S03](SOURCES.md#s03) | 执行器只交预测与运行产物，可信评测器重新计算科学分数与统计证据。 |
| `evolution_lab/genome.py` | 嵌套映射可修改；以含谱系的完整摘要检测无效变异。[S04](SOURCES.md#s04) | 一个统一的不可变行为表示；分离行为、版本、编译和实验身份。 |
| `evolution_lab/store.py` | 评测按候选/cohort 覆盖保存；`:memory:` 在每次新连接下不共享状态。[S05](SOURCES.md#s05) | 评测与运行尝试追加保存，连接生命周期集中管理。 |
| `evolution_lab/controller.py` | 晋级分多次提交；先检查资格再检查幂等；未绑定当前冠军的精确修订。[S06](SOURCES.md#s06) | 带基线修订的原子 CAS 晋级，不再依赖全局 `eligible` 状态。 |
| `evolution_lab/adapters.py` | 真实执行接缝是宿主提供的回调。[S07](SOURCES.md#s07) | 定义完整执行协议和 DSH 映射；真实适配器不能用 smoke 分数替代。 |
| `evolution/genome.py` | 原生路径已有递归冻结、规范化与多种身份。[S08](SOURCES.md#s08) | 迁移经过测试的语义；旧摘要保持原样，新协议使用新命名空间。 |
| `evolution/workflow_ir.py` | 已区分源码、编译行为和运行实例。[S09](SOURCES.md#s09) | 继承这一思想，将科学程序编译与 DSH 运行绑定解耦。 |
| `evaluators/generation_comparison.py` | 原生比较有完整目标集合、覆盖与数值约束，输入是原生持久化评测对象。[S10](SOURCES.md#s10) | 提取通用指标/门禁原语，不把新的两臂比较直接塞入旧的多臂接口。 |
| `core/exposure_registry.py` | 已记录探索性证据并提供正式阶段暴露管理。[S11](SOURCES.md#s11) | 迁移其记录与边界，补足跨运行、跨重命名的数据暴露约束。 |
| `core/director.py` | 生命周期、提案、样本记录、重试和正式阶段等职责集中在约 6,000 行文件中。[S12](SOURCES.md#s12) | 用应用服务、持久化作业和纯状态转换替代大控制器。 |
| 本次核对提交 `af265fc` | 已加入单轴投影、兄弟候选去重信息及弱目标反馈。[S01](SOURCES.md#s01) | 保留并增强，不重复开发“首个单轴搜索”。 |

### 1.2 需要明确纠正的理解

原生路径不是只有 mock，也不是完全没有严谨评测。不能把外部实验台的缺陷推断为原生发布机制已经失效。

不能直接复用一个高层比较函数就声称“统一了评测”：输入对象、比较臂数、逐单元容忍度、覆盖门槛、统计规则都可能不同。迁移时应提取原语，并把协议参数写入不可变合同。[S09](SOURCES.md#s09)[S10](SOURCES.md#s10)

“本轮分数更高”不等于“研究方法进化”。新版应分别报告预测程序变好、搜索成本降低、研究策略跨任务迁移三类证据。

---
## 2. 目标架构：一个内核，两个进化回路

### 2.1 总体流程

```text
用户 / CLI / 现有 DSH Web UI
             │
             ▼
       应用服务 Services
       ├─ RunService              创建、暂停、恢复、取消
       ├─ ExperimentService       安排与执行实验
       ├─ SearchService           选择候选、记录反思、更新研究档案
       ├─ PromotionService        仅更新搜索冠军
       └─ CertificationService    独立验证与发布凭证
             │
     ┌───────┴──────────────────────────┐
     │                                  │
     ▼                                  ▼
科学程序进化                        研究策略进化
ProgramGenome                       ResearchPolicyGenome
模型/特征/拟合/执行技能              提案/诊断/记忆/算子选择
     │                                  │
     └────────结构化提案，不直接执行─────┘
                         │
                         ▼
                  可信内核 Kernel
    合同校验 → 能力编译 → 数据视图 → 宿主评测 → 门禁决策
                         │
                         ▼
                 持久化作业调度 Execution
           状态机 / 预算 / 租约 / 重试 / 取消 / 对账
                   │               │
                   ▼               ▼
             Batch 后端         DSH 后端
            数值训练预测        完整 Agent 执行链
                   └───────┬───────┘
                           ▼
             内容寻址产物 + SQLite 证据账本
```

这里的“双层”是两个不同优化对象，不是必须同时运行两个复杂多智能体系统。第一阶段只启用科学程序进化，研究策略保持固定；随后再用独立任务集验证方法层更新。

### 2.2 固定部分与可变部分

| 部分 | 是否可进化 | 允许改变 | 不允许改变 |
|---|---|---|---|
| 科学程序 | 是 | 已注册模型、特征、拟合方案、不确定性方法、运行时技能与受限工作流 | 数据真值、指标定义、权限上限、评测器实现 |
| 研究策略 | 是，后续启用 | 提案模板、诊断模板、记忆检索、算子权重、预算内实验安排 | 总预算上限、确认集访问策略、通过阈值、审计规则 |
| 工具能力 | 受审核扩展 | 新版本适配器或隔离构建后获准的能力 | 候选直接执行网络代码、导入任意模块、安装宿主依赖 |
| 执行后端 | 工程配置 | Batch、DSH、明确的测试后端 | 伪称运行方式，混合不可比性能数据 |
| 可信内核 | 普通开发流程升级 | 经测试与审核的新协议版本 | 在同一实验中由候选临时修改 |

### 2.3 采用与不采用的架构

**采用模块化单体。** 先解决科学可信性和演化有效性，不把状态一致性问题分散到多个服务。DSH 可仍作为独立进程运行，但 Python 侧只有一个可信业务内核和一套状态真相。

**不采用“完全重写后一次切换”。** 内部可重新设计，对外通过渐进替换迁移。旧运行由旧引擎只读回放，新运行逐步切换。

**不把多 Agent 数量当作创新点。** 调研、提案、诊断可以是同一个模型的不同受控调用；独立评审不取代可执行的科学验证。

---

## 3. 目标目录与依赖规则

保留包名 `ecologyrsi_dsh`，避免无必要地破坏安装器与插件分发。以下是目标模块，不要求 T00 一次创建所有空文件。

```text
src/ecologyrsi_dsh/
  kernel/                    # 纯合同与决策，不访问数据库/网络
    identity.py              # 规范化、深不可变表示、内容身份
    genome.py                # ProgramGenome / ResearchPolicyGenome
    contracts.py             # TaskContract / EvaluationContext / MetricPolicy
    predictions.py           # 预测矩阵合同、严格输入检查
    evidence.py              # EvidenceBundle / ComparisonRecord
    gates.py                 # SearchGate / CertificationGate
    registry.py              # 冻结能力合同与编译权限
    compiler.py              # 与 DSH 无关的可执行计划
    errors.py                # 稳定错误码
  science/
    datasets.py              # 原始数据适配与稳定行身份
    splits.py                # 前向切分、标签到达时间、purge/embargo
    views.py                 # 训练视图/推理视图/评测真值视图
    metrics.py               # 逐单元损失、强基线、权重与单位
    statistics.py            # 配对区块统计、比较证据
    prediction_adapter.py    # 接入现有绿色温室数值预测器
    domains.py               # 预测/模拟任务的领域接口
  research/
    proposals.py             # 可检验假设与受限变更
    operators.py             # 单轴/结构/技能/组合变异
    search.py                # 搜索轮次和算子选择
    repertoire.py            # 互补候选档案，按可比上下文分区
    memory.py                # 结构化实验经验，不是任意历史聊天
    metaevolution.py          # 研究策略的跨任务评估
  execution/
    jobs.py                  # 持久化作业状态与请求身份
    scheduler.py             # 阶段调度、成对执行、公平预算
    budget.py                # 预留、消费、释放与不确定成本
    reconciliation.py        # 外部执行对账、租约与 fencing
    backends.py              # 执行后端 Protocol
  storage/
    database.py              # 连接生命周期与 Unit of Work
    repositories.py          # 实体/评测/比较记录读取与追加
    artifacts.py             # 字节级内容寻址、写后校验
    promotion.py             # 原子搜索冠军事务
    exposure.py              # 样本使用及正式证据暴露
    migrations/
      001_kernel.sql
      002_jobs.sql
      003_research.sql
      004_certification.sql
  adapters/
    dsh_backend.py           # 只负责 DSH 数据/执行协议映射
    batch_backend.py         # 确定性科学计算
    synthetic_backend.py     # 仅测试，不能混入真实晋级
    legacy_import.py         # 旧合同和账本只读导入
  services/
    runs.py
    experiments.py
    search.py
    promotions.py
    certification.py
  application/               # 保留现有 CLI，新增显式 v2 命令
  api/                       # 保留现有 HTTP 框架，增加 v2 资源合同
  presentation/              # 保留现有界面投影与脱敏能力

# 迁移期间保留，最终仅承担旧版只读回放或删除冗余写路径
  core/
  evolution/
  evolution_lab/
  evaluators/
  integrations/

configs/refactor-smoke.json
tests/refactor/              # 新代码全部 unittest 可发现
  __init__.py
  test_identity.py
  test_prediction_contract.py
  test_statistical_gate.py
  test_atomic_promotion.py
  test_jobs.py
  test_batch_backend.py
  test_dsh_backend_contract.py
  test_data_isolation.py
  test_search.py
  test_memory.py
  test_certification.py
  test_metaevolution.py
  test_migration.py
  test_architecture.py
  test_end_to_end.py
scripts/refactor/
  inspect_baseline.py
  migrate.py
  verify.py
  benchmark.py
docs/refactor/
  README.md                  # 本文件
  PROGRESS.md                # 每次执行的真实进展
  BASELINE.md                # 原始测试/源码/接口基线
  DECISIONS.md               # 与本文有差异的架构决策
  BENCHMARK.md               # 实验设计和结果，未运行则不填数值
```

依赖只能朝内：`api/application → services → kernel + research/science/execution/storage`；适配器实现 `execution/backends.py` 的协议，由装配层注入。`kernel` 不得导入 `sqlite3`、HTTP 客户端、DSH 模块或 UI；`research` 不得直接读确认集或直接修改冠军表。

以 AST/导入检查测试这些规则。不能仅在 README 写“解耦”，然后在 `kernel` 里继续 import 旧 `EvolutionDirector`。

---

## 4. 统一身份模型：防止空进化、错误复用与不可比较

### 4.1 五个身份，不再用一个 digest 做所有事

| 身份 | 包含 | 不包含 | 用途 |
|---|---|---|---|
| `behavior_id` | 规范化后的科学程序、执行技能、工作流与生效参数 | 父代、描述、时间戳、评分、运行 ID | 源级无效变异检查 |
| `genome_id` | `behavior_id`、父代列表、变异来源、证据引用、schema | 本次执行的结果 | 谱系与版本追溯 |
| `compiled_id` | 实际编译计划、能力内容摘要、编译器语义、有效运行配置 | 提案说明、候选排序、无效覆盖字段 | 生效行为去重 |
| `context_id` | 数据视图、实际样本、目标网格、评测器、尺度/基线、统计合同、种子、外生运行条件 | 候选程序、父代、候选实际消耗 | 判断两个候选是否可比较 |
| `experiment_id` | `compiled_id + context_id + replicate_id + execution_mode + evaluation_protocol_id` | 调度重试次数 | 实验去重与结果定位 |

另外独立保存 `attempt_id`：同一实验的每一次运行尝试都不覆盖前一次。外部请求还有 `request_id`，用于查询/幂等；调用方不能用“最新插入候选”猜测结果属于谁。

**特别注意：`context_id` 不能包含候选自己的 `compiled_id`。** 否则两个有实际差异的候选永远无法通过“同上下文”检查。另一方面，实际使用的样本集合、训练可见边界与评测版本必须进入上下文，不能只比较 cohort 名字或样本数。

### 4.2 不可变表示

新内核采用规范化 JSON 字符串作为一个最小可靠实现：对象内部不保存可变字典，读取时返回脱离原对象的副本。[身份实现](../src/ecologyrsi_kernel/kernel/identity.py) 给出完整参考模块。

迁移原则：已有 `FrozenJsonObject` 等实现可以复用或提取，不必为了新目录重复造轮子；但 **不能把旧摘要重新按新算法计算后覆盖旧 ID**。旧版和新版身份用不同命名空间，维护显式 `legacy_id → new_id` 映射。[S08](SOURCES.md#s08)

具体要求：拒绝 NaN/Inf、拒绝非字符串键、拒绝 Unicode 规范化后重名的键；布尔值不冒充数值；整数/浮点的归一化由字段 schema 决定，不能全局无区别转换。提示词文本、技能文件、能力实现分别以内容摘要绑定，不只记录一个可随时变化的名称。

### 4.3 编译级去重

```text
提案解析 → 完整 schema 校验 → 源级变更检查 → 权限与能力校验
        → 解析默认值/单位/枚举 → 编译有效计划 → compiled_id 比较
        → 相同则 NoEffectiveChange，不安排收费实验
```

以下都不算有效进化：只改描述、增加父代引用、改变字典顺序、显式填写与默认值相同的参数、修改一个未进入执行计划的字段。

执行后的“预测恰好相同”不一定是空变异，因为不同模型在某组样本上可以输出相同结果。因此区分编译无效变异和行为测量中的无可辨识效果，不用一次输出相同就全局删除候选。

---

## 5. 程序与研究策略的数据模型

### 5.1 ProgramGenome

源合同采用以下结构。示例中的 `*_ref` 必须由宿主解析成存在且内容摘要匹配的对象，不能把字符串当作已安装能力。

```json
{
  "schema_version": "ecologyrsi.program-genome/3",
  "domain": "greenhouse_forecasting",
  "scientific_program": {
    "predictor_ref": "greenhouse-ridge-residual@1",
    "feature_policy_ref": "causal-lag-features@1",
    "fit_policy_ref": "training-only-fit@1",
    "uncertainty_policy_ref": "training-residual-interval@1",
    "parameters": {"ridge_alpha": 0.1, "lag_hours": 24}
  },
  "execution_program": {
    "workflow_ref": "batch-fit-predict@1",
    "skill_refs": [],
    "tool_policy_ref": "registered-science-tools@1"
  },
  "lineage": {
    "parent_genome_ids": [],
    "operator_id": "seed-materializer@1",
    "proposal_id": "seed-greenhouse",
    "evidence_refs": []
  }
}
```

这是目标 schema 示例，不表示这些示例 ID 已在当前仓库登记。Codex 必须先通过 `prediction_adapter.py` 将真实现有能力注册到新内核；未登记 ID 一律拒绝，不得静默切换到默认模型。

`ProgramGenome` 不持有确认集路径、评分阈值、数据库凭据、任意 Python 源码或 DSH 全局管理权限。

### 5.2 ResearchPolicyGenome

将研究方法单独版本化，避免把“预测器升级”和“提案器升级”混成一个不可归因的变异：

```json
{
  "schema_version": "ecologyrsi.research-policy/1",
  "proposal_skill_ref": "single-hypothesis-proposal@1",
  "diagnosis_skill_ref": "paired-failure-diagnosis@1",
  "memory_retrieval_ref": "evidence-filtered-memory@1",
  "operator_weights": {
    "single_axis_parameter": 0.5,
    "feature_change": 0.2,
    "runtime_skill_change": 0.2,
    "compatible_recombination": 0.1
  },
  "screening_fraction": 0.25,
  "max_proposals_per_round": 4,
  "parent_policy_ids": []
}
```

权重与比例是启动配置示例，不是已经验证的最优值。宿主对权重和、范围、允许算子、最大提案数及总预算执行强校验。

方法层更新不能改 `MetricPolicy`、`GatePolicy`、确认集暴露规则或资源总上限。进入同一策略比较的各方法共享外层任务与预算合同。

### 5.3 Proposal：必须是可执行假设

每个候选提案持久化以下字段：

| 字段 | 合同 |
|---|---|
| `proposal_id` | 全局唯一；重试不产生另一个逻辑提案 |
| `parent_genome_id` | 精确父版本，不用“current”这样的浮动引用 |
| `hypothesis` | 说明哪个失败现象与哪个机制相关，不声称已证明因果 |
| `target_cells` | 受合同约束的目标/时距集合 |
| `operator_id` | 冻结注册表内的变异算子 |
| `changes` | 有界结构化变更；默认一次一个可归因因素 |
| `expected_effect` | 方向与需验证的观察量，不是承诺增益 |
| `evidence_refs` | 能实际读取且未越权的数据/轨迹/文献证据 |
| `stop_rule` | 单次试验的预算/失败终止条件 |
| `search_context_id` | 使用了哪一版训练反馈、记忆与知识快照 |

保留已核对提交中的单轴设计和弱目标定位；再增加结构变异与组合变异时，显式记录“这是联合变更”，不能继续用单因素因果解释。[S01](SOURCES.md#s01)

---

## 6. 数据合同：把可见范围做成代码边界

### 6.1 三种数据视图

`TrainingView`：只包含本次允许拟合的数据、当时已到达的标签以及训练专用尺度拟合参数。

`InferenceView`：只包含预测起点时可获得的历史观测、已知外生变量和合同允许的行动计划；不包含未来真值与评测目录。

`TruthView`：只交给可信评测进程/模块，用于计算误差和约束；不可序列化进 LLM 消息、候选上下文或前端搜索投影。

Python 接口隔离不是操作系统级安全隔离。对宿主已注册的可信数值算法，模块边界可作为工程约束；对模型生成代码，必须另设进程/容器、独立权限和不可见的标签挂载。不能用一句“只传了视图”声称已防止同用户进程读文件。

### 6.2 预测单元身份

每个预测单元至少绑定：

```text
source_dataset_id          原始数据来源稳定身份，不是任意导出文件名
source_episode_id          原始温室/实验 episode 身份
source_row_identity        原始位置与内容指纹，派生快照保留血缘
origin_timestamp_utc       带时区的预测起点
target_name                例如 temperature
horizon                    统一单位的预测时距
label_timestamp_utc        真值所属时间
label_available_at_utc     标签实际可供训练/反馈使用的时间
replicate_seed             本次可比较重复的种子
```

`case_id` 由宿主规范化计算。样本集合摘要基于这些稳定身份与具体视图，不依赖用户起的 cohort 名称。

同一原始数据被重新导出、换目录、换运行名称、增加无关列，不应该变成新的独立证据。发生原始数据修订时记录 revision 和共享行血缘；新增快照的所有行也不能自动被认定为“从未看过”。

### 6.3 训练与预测的因果时序

在 `science/splits.py` 实现：

```text
assert_no_future_features(view, origin)
assert_labels_available(training_rows, fit_cutoff)
assert_no_forbidden_overlap(training_manifest, evaluation_manifest, split_policy)
build_forward_splits(episodes, policy)
```

`assert_labels_available` 检查的是 `label_available_at <= fit_cutoff`，不只是 `row.timestamp <= origin`。当同一行需要构造 24 小时标签时，行时间在过去也不代表未来标签已经可用。

标准化、缺失值填充、基线选择、特征筛选和校准也只能使用相应训练视图。禁止整表归一化以后再切分。

分区默认按完整 episode 或连续时间区间预先冻结；对跨边界标签做 purge，对需要额外隔离的协议做 embargo。长度由最大标签跨度、数据采样和依赖结构决定，不能写死“永远 24 小时足够”。

### 6.4 搜索数据复用与独立证据

探索性数据可以复用，但分别报告：`execution_occurrences`、`unique_origins`、`unique_source_rows`、`effective_blocks`。禁止把 repeated occurrence 当作新的独立样本。

确认数据的暴露记录放在跨 run 的权威账本。重命名目标、改变权重、开新 run 或换策略模型都不能重置暴露状态。若用不同文件部署多个独立数据库，必须明确无法保证全局不重用；正式验证应指定唯一 authority，而不是每个 run 自建一个“干净”的数据库。

---

## 7. 评测重构：从“报告可信”改成“产物可验证”

### 7.1 后端只返回预测与执行证据

替代当前 `Backend -> EvaluationReport(scientific_score, stable, ...)`：

```text
ExecutionBackend
  返回 PredictionArtifactRef + ExecutionTraceRef + UsageRecord + RuntimeReceipt

TrustedEvaluator
  校验请求/产物/矩阵/真值绑定
  → 计算逐单元损失与约束
  → 计算基线与聚合指标
  → 形成不可变 EvaluationRecord

ComparisonEngine
  读取同 context 的候选/基线评测
  → 计算配对差与统计区间
  → 执行版本化 GatePolicy
  → 保存 ComparisonRecord
```

现有的 `stable` 输入字段删除。稳定性来源于可信统计函数计算的证据，而不是调用方声明。

### 7.2 预测矩阵严格验证

[预测矩阵实现](../src/ecologyrsi_kernel/kernel/predictions.py) 是完整参考模块，负责：有限数、精确样本集合、重复行、编译身份和上下文一致性。

生产入口另外强制 body/行数/字符串长度上限、确切 schema 字段、标准单位、记录数量与摘要校验。未知字段拒绝，而不是默默忽略；JSON 重复键必须在解析时拒绝。

候选与基线需要逐 `case_id` 对齐。不允许只取双方交集来让失败样本消失；预先约定了缺失策略的协议也必须保留所有失败与分母。

### 7.3 科学指标与优化方向

优先保持当前已验证评分协议作为 `legacy_metric_policy` 的固定版本。新目标可以使用训练尺度归一化的 MAE/RMSE、强基线相对收益与物理约束，但必须另起版本，不与旧原始分数跨协议排名。

通用损失定义：

```text
cell_loss(c) = metric(y_true[c], y_pred[c]) / frozen_scale[c]
aggregate_loss = sum(frozen_weight[c] * cell_loss(c))
paired_gain = incumbent_loss - candidate_loss       # 正数表示候选更好
```

`frozen_scale`、权重与强基线必须在看候选结果之前冻结。基线误差为零、缺失标签、单位不匹配、全常数序列需要明确错误码或预设分母策略，不能临时加一个 epsilon 来“让结果通过”。

物理约束为硬门槛，不通过“精度高”抵消负浓度或禁用行为。效率进化使用非劣性门槛，不能直接把金额与温度误差相加。

### 7.4 配对统计与门禁

`statistics.py` 输出 `PairedEvidence`，至少包含：

```text
candidate_evaluation_id / baseline_evaluation_id
context_id / metric_policy_id / statistical_policy_id
case_ids_digest / unique_origins / effective_blocks
point_gain / gain_ci_lower / gain_ci_upper
per_cell_gain / per_cell_lower_bound
resampling_seed / resample_count / block_definition_digest
alpha_budget_id / analysis_family_id
```

实现顺序：先校验双边完整性和配对身份，再形成按 episode/连续时间分组的配对损失，最后做预先声明的区块统计。共享天气或同一温室批次导致的相关性也应纳入分组合同。

不能把 9 个目标/时距单元当作 9 个独立 episode。数据不足以形成有效区块时返回 `inconclusive`，不是把 `stable=True` 作为默认值。

**搜索门禁**可以是描述性的：在固定探索性集合上，改善达到实用阈值且满足硬约束就进入档案或更新搜索冠军；明确标记它不是独立确认。

**确认门禁**还需：冻结的候选/基线、未暴露的验证计划、预先定义的统计检验与多重比较处理、足够独立数据、完整证据链。连续试到显著为止不属于独立验证。

不要把普通 bootstrap 区间包装成“对任意自适应搜索都有效”。同一确认批次有多个候选/多个指标时，预先冻结比较家族与校正规则；固定样本方案最先实现，序贯方案另起协议。

### 7.5 两条收益路径

| 路径 | 科学条件 | 资源条件 |
|---|---|---|
| `scientific_gain` | 配对科学收益超过实用阈值，逐单元在预设容忍范围内 | 成本/延迟不超资源合同 |
| `efficiency_gain` | 科学性能在预先约定非劣界限内 | 成本、延迟或执行失败率有可测改善 |

技能变化不一定改变预测值，可能减少修复次数和工具失败。这类收益可以被接受，但必须采用效率路径的证据，不人为给“装了新技能”固定加分。

---

## 8. 持久化：追加证据、原子晋级和可恢复副作用

### 8.1 状态与证据分开

删除把 `eligible` 写成候选永久属性的做法。候选是否值得晋级取决于“相对哪个冠军、在哪个上下文、依据哪份评测、使用哪个策略版本”。

核心对象：

| 对象 | 是否允许覆盖 | 说明 |
|---|---|---|
| Genome / CompiledProgram | 否 | 同 ID 不同内容为冲突 |
| ExperimentAttempt | 状态受控更新；结果追加 | 不覆盖旧 attempt 的输入和输出 |
| Evaluation / Comparison | 否 | 同主键不同 payload 拒绝；同实验重跑使用新记录 |
| SearchChampion | 是，仅通过事务与 CAS | 每个 run 一个搜索冠军，带递增 seq |
| ReleaseManifest | 否 | 与 SearchChampion 独立，不由搜索控制器直接写 |
| EvidenceRevocation | 追加 | 发现问题时新增撤销记录，不修改过去的评分 |
| CommandReceipt / Event | 追加 | 幂等命令与可审计事件 |

[基础数据库 schema](../src/ecologyrsi_kernel/storage/schema.sql) 提供晋级纵向切片的基础 DDL，[原子账本实现](../src/ecologyrsi_kernel/storage/ledger.py) 给出对应完整事务函数。后续作业、产物、正式验证和研究档案表通过后续迁移添加，不能另建第二套不相容的晋级库。

### 8.2 连接与事务

`storage/database.py` 统一配置每个连接：`foreign_keys=ON`、`busy_timeout`、明确事务控制。文件库在初始化时设置 WAL，并按部署要求选择 durability；`:memory:` 测试使用存活的同一连接，或明确的共享内存 URI 与 keeper 连接，不能每次连接一个新数据库。[S13](SOURCES.md#s13)[S14](SOURCES.md#s14)

第一阶段只需一个受锁保护的持久化连接或单写入线程；跨进程工作者通过服务提交结果。允许读连接，但所有修改通过统一服务和短事务。

显式事务内不调用 DSH、模型、网络或耗时统计函数。迁移脚本中的 `executescript()` 与业务事务分开，避免隐式提交破坏预期原子性。[S14](SOURCES.md#s14)

### 8.3 正确的晋级顺序

```text
服务端权限校验
 → BEGIN IMMEDIATE
 → 按 run_id + idempotency_key 查旧命令
 → 已存在且请求摘要一致：返回旧响应，不重新检查候选现状
 → 已存在但请求摘要不同：409 IdempotencyConflict
 → 读取当前冠军与 seq
 → 读取宿主生成的 ComparisonRecord 和双边 Evaluation
 → 校验比较基线、上下文、协议、来源、有效期与撤销状态
 → CAS 更新 champion（旧 genome_id + seq 必须匹配）
 → 插入 command receipt + event + 必要的 outbox
 → COMMIT
```

必须用递增 `seq` 防止 A→B→A 后旧证据再次通过，即 ABA 问题。只比较当前 genome ID 不足够。

幂等命令重放返回原来的成功结果，并不再次把当时的候选晋级；原操作后来已被替换或撤销，也仍可查询其历史回执。新的晋级动作必须使用新的命令键和当前证据。

### 8.4 回滚不是“任意候选提升”

实现 `rollback_search_champion(run_id, target_history_entry_id, expected_seq, reason, idempotency_key)`：目标必须来自该 run 的有效冠军历史；记录原因与授权人；事务内核对当前 seq；回滚产生新 seq 和新事件，不删除中间历史。

正式版本回滚走独立的 `ReleaseService`，目标必须是历史获准的 release；不能直接调用搜索回滚替代正式治理。

### 8.5 产物、数据库与 outbox

产物先写到受控临时路径，计算字节级摘要并校验大小，原子发布到内容寻址目录，再在数据库事务中引用。写文件成功、写库失败留下的孤儿由有保留期的清理器处理；不能删除仍被任何历史记录引用的对象。

`artifact_id = SHA256(原始字节)`，语义对象 ID 使用规范化 JSON 的独立命名空间。两者用途不同，不混用。

所有外部执行先在数据库创建作业和 outbox，提交后才发请求。远端返回以前崩溃的情况靠 request ID 对账；数据库原子性不能神奇地保证远端模型“只计费一次”。

哈希链能帮助检查内容变更，但不防止拥有同一数据库写权限的人伪造整条链。真实证据来源需靠服务权限、隔离和必要的宿主签名，而不是仅靠 `report_digest` 字段。

---

## 9. 执行引擎：先把真正的实验跑对，再谈自进化

### 9.1 稳定的后端接口

在 `execution/backends.py` 定义下列接口合同；方法中的 `...` 是 Protocol 的合法接口声明，不是允许生产后端留空。

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    backend_kind: str
    semantic_version: str
    supports_idempotent_submit: bool
    supports_lookup_by_request: bool
    supports_cancel: bool
    supports_batch_prediction: bool

@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    request_id: str
    experiment_id: str
    attempt_id: str
    run_id: str
    lease_epoch: int
    compiled_plan_artifact_id: str
    inference_view_artifact_id: str
    training_view_artifact_id: str | None
    context_id: str
    seed: int
    budget_reservation_id: str
    deadline_epoch: int

@dataclass(frozen=True, slots=True)
class ExecutionHandle:
    request_id: str
    remote_execution_id: str
    attempt_id: str

@dataclass(frozen=True, slots=True)
class ExecutionStatus:
    state: str
    prediction_artifact_id: str | None
    trace_artifact_id: str | None
    usage_artifact_id: str | None
    runtime_receipt_artifact_id: str | None
    error_code: str | None

class ExecutionBackend(Protocol):
    def capabilities(self) -> BackendCapabilities: ...
    def submit(self, request: ExecutionRequest) -> ExecutionHandle: ...
    def lookup(self, request_id: str) -> ExecutionHandle | None: ...
    def poll(self, handle: ExecutionHandle) -> ExecutionStatus: ...
    def cancel(self, handle: ExecutionHandle) -> ExecutionStatus: ...
```

这些 DTO 在落地时增加与[预测矩阵实现](../src/ecologyrsi_kernel/kernel/predictions.py) 同等级的运行时类型/范围检查；类型标注本身不是校验。持久化前使用严格 JSON 解析，所有可变输入在构造时规范化冻结。

`ExecutionRequest` 不能携带 `TruthView`，也不能让后端自行选择评分函数。`deadline_epoch` 是协议字段，实际进程内计时用单调时钟；墙上时钟变化不应延长预算。

### 9.2 两种真实执行模式

**BatchScientificBackend：** 一次创建训练产物，批量预测目标矩阵，适合算法/特征/参数搜索。LLM 主要用于实验前的研究与实验后的诊断，不为每个预测起点重复做同样规划。

**DshAgentBackend：** 运行真实规划、工具调用、修复等链路，适合研究运行时技能和工作流。记录每个动作的有效配置、工具结果和失败，不用一个合成评分函数代替。

二者可以共享科学评测器，但不能跨执行模式直接比较延迟或成本；当比较科学准确率时，也必须预先定义允许的模型输入与信息边界，不让某一模式多看未来信息。

### 9.3 DSH 适配落点

需要核对并迁移的当前入口：

- `src/ecologyrsi_dsh/integrations/dsh_native_runtime.py`
- `src/ecologyrsi_dsh/integrations/dsh_structured_roles.py`
- `src/ecologyrsi_dsh/integrations/model_gateway.py`
- `src/ecologyrsi_dsh/evolution/workflow_ir.py`
- `integrations/dsh_ecology_plugin/lib/` 下的宿主代码
- `tests/test_dsh_native_runtime.py`、`test_dsh_sample_execution.py`、`test_dsh_reconciliation.py`

以上集成入口是核对与迁移方向，测试名称须以本地实际存在文件为准。具体调用签名由 T08 在本地读取并写入 `docs/refactor/DSH_MAPPING.md`。本文不虚构 DSH 的 REST endpoint。[S09](SOURCES.md#s09)[S15](SOURCES.md#s15)

`DSH_MAPPING.md` 必须包含：新方法、实际旧类/函数、真实请求字段、返回字段、错误码、能力支持情况、对应测试。接口不支持 lookup 或远端幂等时，明确返回该能力为 false，不用假实现掩盖。

原生编译器已经提供 `compile_plugin_behavior()` 与 `bind_phenotype_instance()`，并区分 `CompiledEcologyBehaviorSpec` 和 `CompilationInstanceContext`。重构是提取并扩展已有行为/实例分离，不是从零新增这条机制。[S09](SOURCES.md#s09)

新内核接口建议整理为两步：

```text
compile_program(genome, registry_snapshot, compiler_policy) -> CompiledProgram
bind_execution(compiled_program, task_context, backend_profile) -> ExecutionRequest
```

第一步解析有效科学行为和受限 DAG；第二步加入 run/attempt、数据视图、实际模型路由与资源限额。编译器不读实时冠军，不访问数据库；同输入产生同计划。

### 9.4 四类变异必须有行为证据

| 变异 | 集成测试应验证 | 不能接受的替代验证 |
|---|---|---|
| Algorithm | 新预测器/有效参数进入训练，产物 manifest 和受控样本输出相应改变 | 只看配置 JSON 文本不同 |
| Feature | 训练器实际消费新的特征集合，特征血缘与可见时间正确 | 只在提案中写了“增加滞后特征” |
| Skill | 宿主加载新的技能内容摘要，受控失败场景中的动作/诊断发生预期变化 | skill ID 不同就固定加分 |
| Workflow/Tool | 轨迹中节点和工具准入按新计划执行，禁用工具真的不能调用 | 工具列表有变化但执行路径仍相同 |

“受控样本输出改变”只用于精心设计的行为测试，不要求任何参数变更在任何数据上都改变预测。

---

## 10. 作业状态机、取消、重试和预算

### 10.1 不让一个 Director 包办所有状态

运行状态：`CREATED → RUNNING ↔ PAUSED → COMPLETED/CANCELLED/FAILED`。

实验作业状态：

```text
PLANNED → QUEUED → LEASED → SUBMITTING → RUNNING → COLLECTING → SUCCEEDED
                            │              │          │
                            └── UNKNOWN ───┘          └→ INVALID_OUTPUT
                                            ├→ FAILED_INFRA
                                            ├→ FAILED_CANDIDATE
                                            ├→ TIMED_OUT
                                            └→ CANCELLED
```

`UNKNOWN` 表示提交结果不确定，不等于未执行。进入对账流程，不能立刻创建新的收费请求。

以纯函数 `transition_job(state, command, clock_snapshot) -> TransitionResult` 计算允许的状态变化与需要追加的事件；实际持久化由 Unit of Work 完成。时钟与随机数均注入，测试不依赖真实等待。

### 10.2 持久化字段

在 `002_jobs.sql` 新增：

```text
jobs:
  job_id PK, experiment_id, run_id, state, state_seq,
  active_attempt_id, lease_owner, lease_until_epoch, lease_epoch,
  next_action_at_epoch, cancel_requested, created_at_epoch

attempts:
  attempt_id PK, job_id FK, attempt_number,
  request_id UNIQUE, backend_kind, request_payload_digest,
  remote_execution_id, state, result_artifact_id, error_code,
  started_at_epoch, finished_at_epoch

outbox:
  message_id PK, run_id, job_id, event_kind, payload_json,
  request_digest, state, attempts, available_at_epoch

budget_accounts:
  account_id PK, run_id, unit, limit_amount, used_amount, reserved_amount

budget_reservations:
  reservation_id PK, account_id FK, request_id,
  reserved_amount, settled_amount, status
```

金额、token 数、调用数使用明确单位的整数；成本未知不能填 0。美元微单位与 GPU 秒分别建账，不直接相加；单次调用的价格版本写入计费合同。

### 10.3 租约与 fencing

工作者认领作业时原子增加 `lease_epoch`。结果回传必须带当前 `attempt_id + lease_epoch`；旧工作者在租约过期后返回的结果只作为迟到产物存档，不能覆盖新的状态或触发晋级。

恢复时优先 `lookup(request_id)` 查找实际远程运行。后端不支持查询且提交是否成功不明时，作业停在 `UNKNOWN`，报告阻塞；不能承诺“严格只调用一次”。

缓存命中不是新的运行尝试成功，也不是新的独立样本。评测缓存可复用已核验结果，但通过 `EvidenceReused` 记录它的来源，不增加独立证据计数。

### 10.4 失败分类

| 错误 | 是否计入候选能力 | 默认处理 |
|---|---|---|
| Provider 429、可恢复网络断连、短暂服务不可用 | 否 | 退避重试；成本不确定先对账 |
| 凭据无效、权限配置错误 | 否 | 暂停运行，明确需要修复环境 |
| 候选调用禁用工具、输出非法矩阵、算法数值溢出 | 是 | 记录候选失败，按协议允许有限修复 |
| 评分器自身异常、标签缺失、数据合同不完整 | 否，不能给候选打低分代替 | 阻断评测并标记无效实验 |
| 超过候选约定的资源上限 | 通常是 | 按版本化预算合同判定；保留实际资源证据 |
| 用户取消或宿主关闭 | 否 | 取消/可恢复状态，不能伪装成自然完成 |

### 10.5 双层预算

运行级限制：总候选数、科学评测次数、模型调用数、token、墙钟、货币成本。

作业级限制：本次允许工具次数、输出 token、超时、内存与并发。研究策略只能在剩余额度内安排实验，不能自己提高账户上限。

预留与结算事务分别实现：`reserve_budget(request_id, amounts)`、`settle_budget(reservation_id, usage)`、`mark_usage_unknown(reservation_id)`。重复结算不能重复扣费；请求结果未知时不能释放全部额度后继续无限发请求。

---

## 11. 搜索机制：从“反复选冠军”到“可归因的实验搜索”

### 11.1 默认搜索流程

```text
冻结本轮知识快照 + 可见经验 + 研究策略
 → 在当前可比上下文内选父代/互补档案
 → 生成有限数量的假设
 → 编译与去重
 → 小预算同窗初筛
 → 为入围者分配相同追加评测预算
 → 候选与当前冠军同 context 成对评估
 → 可信比较，更新搜索档案
 → 必要时 CAS 更新搜索冠军
 → 记录假设支持/反驳与预算消耗
```

每个候选都保留未通过的原因。不能因为最终没有入围，就删除其失败记录或不计提案成本。

### 11.2 多保真筛选

初筛与精评使用不同 `context_id` 或 fidelity 字段，禁止把不同 fidelity 的原始分数放在同一排行榜。

在同一阶段对所有候选使用共享样本；入围后在新的固定阶段上下文比较。低保真结果只负责分配预算，不作为正式验证。缓存需要包含 fidelity、数据视图与编译身份。

默认先用确定性的逐阶段调度，不在第一版就接入复杂异步 bandit。T10 完成后再比较 successive halving 等策略，且必须保留等预算对照。

### 11.3 互补候选档案

`RepertoireEntry` 最小字段：

```text
entry_id / genome_id / compiled_id
context_id / comparison_family_id
niche_id                     如 overall、long_horizon、low_cost
metric_vector / cost_vector
supporting_evaluation_ids
admission_reason / created_event_id
```

档案按比较家族和上下文分区；不同数据上的分数不能直接互相支配。候选切换到新场景前必须重新评测。

保留有限数量的互补版本，不只保存一个全面冠军。正式发布门禁仍严格。用于正式选择的 niche 必须预先定义；探索性发现的新 niche 标记为探索用途。

### 11.4 变异算子

第一批只实现：

```text
SingleAxisParameterMutation    复用最新单轴设计；范围、步长与默认值规范化
FeaturePolicyMutation          只换一个可执行特征策略
RuntimeSkillMutation           只换一个运行时技能/参数
CompatibleRecombination        组合已有版本，检查数据/维度/工具兼容
```

每个算子必须有：输入 schema、允许字段、前置条件、输出 schema、最大变化幅度、有效变化检查与编译拒绝原因。

保留一条固定随机搜索对照；不能把随机对照关掉以后只展示进化曲线。去重拒绝也计入提案质量统计，但不扣一次不存在的科学评测。

### 11.5 算子收益更新

先用可解释的指数滑动统计或固定分配，不强制复杂强化学习。奖励使用同上下文配对收益与实际成本，不能用各自不同 cohort 上的最高分差。

```text
operator_event = {
  operator_id,
  parent_compiled_id,
  candidate_compiled_id,
  paired_gain,
  evaluation_cost,
  valid_compilation,
  candidate_failure,
  evidence_class
}
```

单轴试验的结果是受该背景条件约束的局部证据；不要把少量相关性输出成普遍因果结论。已存在的不可辨识参数标记继续保留。[S01](SOURCES.md#s01)

---

## 12. 经验与技能：保存“什么条件下有效”，而不是只保存漂亮反思

### 12.1 ExperienceRecord

`research/memory.py` 使用结构化记录：

```json
{
  "schema_version": "ecologyrsi.experience/1",
  "experience_id": "exp-memory-001",
  "domain": "greenhouse_forecasting",
  "task_family": "multihorizon_temperature",
  "observed_failure": "long_horizon_error_after_regime_change",
  "hypothesis": "current features omit a relevant lagged response",
  "intervention_operator": "feature_policy_change",
  "parent_compiled_id": "bound-by-host",
  "candidate_compiled_id": "bound-by-host",
  "comparison_id": "bound-by-host",
  "outcome": "supported_on_search_data",
  "applicability": ["same feature availability", "same target definition"],
  "counterexamples": [],
  "evidence_class": "search",
  "knowledge_snapshot_id": "bound-by-host"
}
```

示例中的绑定值由宿主填充，LLM 不得自行构造一个不存在的 `comparison_id` 来证明观点。

### 12.2 检索顺序

先做权限和数据域过滤，再做任务/失败模式匹配，最后排序。第一版用 SQLite 元数据索引和确定性的匹配函数即可；没有证据说明需要向量库之前，不新增一个服务。

推荐排序依据：同任务族、同可见特征条件、相近失败模式、证据质量、成功和反例数量、最近一次验证时间。只检索成功案例会强化错误偏好，必须保留被反驳与无结论记录。

### 12.3 从经验到技能的晋级

```text
LLM 提炼候选技能
 → 引用的实验真实存在且可见
 → 技能内容版本化
 → 编译进受控运行时
 → 在未参与提炼的开发任务上做有/无技能比较
 → 符合可靠性/成本条件后加入可选技能库
```

搜索阶段可以在新的开发任务上继续改技能，但这些结果仍是开发证据。不能把技能开发时反复看的任务重新称为独立确认集。

科研记忆的条目是待验证的经验，不是系统指令。外部论文、用户文件、工具输出中的“忽略权限”或“上传密钥”等内容不能获得工具授权；工具准入由能力合同与宿主检查决定。

### 12.4 可测量的记忆收益

固定提案和实验预算，比较“无跨轮记忆”和“使用记忆”：有效提案率、重复失败率、达到目标损失的成本、跨任务性能。保存同一任务族的来源与排除关系，避免检索到目标确认任务的历史答案。

---

## 13. 方法层进化：怎样证明系统越来越会做研究

### 13.1 外层评估单位是“任务”，不是单个预测起点

输入：两个冻结的 `ResearchPolicyGenome`。

比较条件：相同任务池、初始科学程序、可见知识快照、初始记忆快照、种子计划和资源上限。

输出：每个策略在每个任务上的搜索产物、搜索成本、未见评估集上的结果、失败类型。方法层不读取对手私有搜索状态，也不在试验中临时换初始资源。

### 13.2 策略 trial 的内部时序

```text
冻结 ResearchPolicyGenome
 → 在 meta-development 任务的 fit/search 视图上运行完整内层搜索
 → 冻结最后提交的科学程序及其构建产物
 → 在该任务的 held-out development 视图评测一次
 → 汇总任务级收益和成本
 → 比较两种研究策略
```

这些 development 结果可用于下一次研究策略调整，因此不能当作最终独立验证。最终方法层结论使用另一个冻结的 meta-confirmation 任务集合，并记录其暴露；独立性不能靠层级改名获得。

### 13.3 元策略的数据合同

```text
MetaEvaluationContract:
  task_set_manifest_id
  initial_program_manifest_id
  initial_memory_snapshot_id
  knowledge_snapshot_id
  policy_candidate_id / policy_baseline_id
  seed_schedule
  per_task_budget_policy_id
  task_level_metric_policy_id
  confirmation_plan_id | null
```

`task_set_manifest_id` 中记录任务数据的原始血缘，防止训练任务和确认任务只是同一温室 episode 的两种描述。

### 13.4 汇总指标

必须同时报告未见任务的平均/中位科学结果、任务级配对差、总成本和失败率。可以补充单位成本增益，但不能只报一个比值。

```text
efficiency = held_out_gain / measured_search_cost
```

成本为零或未知时该比值为缺失，不填无穷大。绝对增益为负时不能用极低成本把它包装成成功。

研究策略晋级依据跨任务结果，不依据它优化过的某一个任务的最高分。不要让元策略直接选择确认指标、反复查询最终任务集或自行扩充预算。

---

## 14. 科学任务扩展：先预测，再研究行动与长期约束

### 14.1 首先形成真正可替换的科学计算接口

在 `science/prediction_adapter.py` 与 `science/domains.py` 定义以下行为合同，实现时复用现有预测器和测试数据，不重写已验证的数值算法：

```text
fit(program_spec, training_view, seed) -> ModelArtifactRef
predict(model_artifact, inference_view) -> PredictionArtifactRef
validate_physical_constraints(predictions, domain_contract) -> ConstraintReport
```

`fit` 的结果保存模型权重/参数、特征 schema、拟合窗口、随机种子、依赖版本、源代码提交和产物摘要。

在线更新必须显式声明：每个预测起点前只用已到达标签更新；所有更新产生可追溯 checkpoint。禁止在没有轨迹记录的情况下修改同一个模型文件。

### 14.2 从有限参数走向程序组合

下一阶段支持一个受限、类型化的科学流水线 IR：

```text
InputSchema
 → CausalFeatureTransform
 → RegisteredForecaster
 → OptionalResidualCorrection
 → RegisteredUncertaintyEstimator
 → PredictionSchema
```

每个节点登记输入/输出维度、单位、可用数据分区、确定性要求和成本级别。编译时检查 DAG 无环、端口类型、节点数/深度上限与能力摘要。需要循环时使用单独受限算子及最大步数，不允许任意跳转。

优先增加可复用的特征/残差/集成算子，再考虑大量独立预测模型。组合产生的新程序依然需要完整训练与对照，不因“机理 + 数据”这样的标签自动获得更高质量评价。

### 14.3 行动条件模型

后续新增 `ActionConditionalForecastTask`，与 `HistoricalForecastTask` 分开：

```text
predict(history, known_exogenous_inputs, planned_actions, horizon)
    -> future_state_distribution
```

必须区分“预测时已知的控制计划”与“事后记录的实际未来动作”。后者直接作为预测输入可能带入未来信息，或形成不能用于新策略评估的关联。

先通过可控模拟器验证行动响应和多步误差，再讨论真实设备控制。历史预测准确不等于干预效果正确，不把离线误差改善直接报告成节能、增产或生态稳定收益。

### 14.4 长期约束

`DomainContract` 增加状态范围、质量/能量平衡、资源预算与允许行动类别。封闭系统条件下，禁止未经声明的外部物质补充；内部循环、光照或温控调节与外部补给分别建模。

长期目标可包含累计约束违规、失稳次数、资源消耗、长期模拟收益及不确定性覆盖。具体物理方程和参数由领域数据与校验确定，不能在代码里凭经验硬填一套“生态真值”。

本轮重构只保留接口和可测模拟任务入口，不连接真实温室设备，也不把尚未验证的模拟器作为真实世界裁判。

---

## 15. 新能力与代码生成：受控扩展，不再永久困在既有参数表

### 15.1 能力注册不只是一张名称表

`CapabilityRecord` 至少包含：

```text
capability_id / kind / semantic_version
implementation_artifact_id / dependency_lock_id
input_schema_id / output_schema_id
allowed_domains / allowed_roles / allowed_data_views
resource_envelope / required_permissions
deterministic_flag / validation_suite_id
review_receipt_id / registry_snapshot_id
```

注册表在实验开始时冻结。允许新能力的引入，但注册表变化创建新的实验 epoch，不在同一 comparison 中偷偷替换同名实现。

能力参数需要按 schema 支持整数、有限实数、布尔、枚举、短文本和结构化小对象，不能全部按浮点处理；同时限制大小、长度、深度和未知字段。[S16](SOURCES.md#s16)

### 15.2 三条能力扩展路径

**路径 A：已注册参数/组合。** 默认启用，成本低，直接编译与实验。

**路径 B：结构化新技能。** 生成提示模板、输入输出规则与可用工具声明；经过编译、行为测试和独立开发任务验证后加入候选库。

**路径 C：生成新科学代码。** 可选研究功能，默认关闭；只进入隔离构建队列，不直接注入 DSH 或当前宿主。

### 15.3 可选代码实验的具体流程

```text
CodeProposal（独立源码产物）
 → 静态合同检查
 → 无密钥、无网络、无宿主写权限的隔离构建
 → 依赖来自审核允许的离线锁定集合
 → 输入输出合同与资源限额测试
 → 合成/开发数据上的科学与数值测试
 → 保存源码/镜像/依赖/测试报告摘要
 → 宿主审核注册为新能力
 → 新 epoch 中按同一评测流程参与竞争
```

路径白名单、AST 检查或禁止几个字段名，不是执行隔离的替代品。容器也不能挂载宿主 Docker socket、模型凭据、完整数据根目录或确认真值；默认禁网，超时后终止整个子进程树，并保留审计。

隔离功能未配置时返回 `CapabilityBuildUnavailable`，不能降级为宿主 `exec()`、`eval()` 或 `subprocess(shell=True)` 执行生成内容。

这条路径提供真正扩展程序空间的能力，但不让优化器进化自己的评分器、权限管理器或正式验证集。

---
