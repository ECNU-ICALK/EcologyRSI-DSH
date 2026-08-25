# EcologyRSI-DSH 项目整体 Review 与方案 B 优化报告

> 评审日期：2026-08-24<br>
> 优化版本：0.3.26<br>
> 评审范围：Python 核心、温室评测、reward/objective、进化晋级、逐样本归档、数据边界、插件与交付验收

## 1. 结论摘要

项目已具备较完整的“可重放进化实验工作台”骨架：有追加式账本、固定数据分区、候选生成、登记预测器、逐样本执行、独立评审、窗口感知晋级和中文可视化插件。工程安全边界和审计性设计明显强于一般研究原型。

但它当前的科学实体仍是“温室环境多时距预测 + 有界参数搜索”，不是完整的农业—生态世界模型。它尚未建模作物生长、产量、冠层状态、土壤水热氮碳、微生物、管理动作后果、长时滚动不确定性和跨 episode 外推。因此，对外定位应区分“已可运行的温室预测进化核心”与“待建设的生态世界模型科学层”。

本次方案 B 已直接实施，重点解决了原 reward 可被弱持续性基线虚高、聚合分母易漂移、失败样本惩罚语义不统一、晋级只要 `1e-12` 即视为改善，以及评分代码难以独立验证的问题。

对最近本地运行的只读审计显示：闭环、候选变异、真实 DSH Agent 调用、联合向量工具和跨代上下文都在工作；尚无候选通过正式科学门禁，因此晋级数为 0，也就没有证据可以声称已产生科学上有效的自进化。历史运行的单轮预算主要为 9、18、200、500 个评分 cell，均低于当前正式选择所需的 1521 个平衡 cell；小预算结果只能作为工程诊断，不能用较快完成或单次正分数推断进化有效。

## 2. 现有项目优势

### 2.1 工程与治理

- 领域对象、追加式事件、SQLite 重放和状态投影构成了清晰的证据链。
- 数据分区明确区分 `training_fit`、`training_feedback`、`development`、`gate` 和 external holdout，浏览器 API 只暴露可见分区。
- 预测器、评测器、模型网关和 judge 都有宿主登记、参数边界和 digest 冻结。
- 逐样本执行支持失败分类、重试、修复、检查点、有界归档和实时进度。
- 网关 URL、Bearer 凭据、重定向、HTTP 明文路由和浏览器投影的安全约束较完整。

### 2.2 时序评测

- 模型只在 `training_fit` 拟合，后续 `training_feedback` 只用于评测和更新证据。
- 支持 3 个目标 × 1/6/24 小时时距，按真实小时戳而非数组位置构造时序样本。
- 已区分原始物理单位误差与用 `training_fit` 尺度归一化的跨目标指标。
- 同轮候选共享固定 cohort，不同窗口的原始分数不直接互比。

## 3. 审查发现与优先级

| 优先级 | 发现 | 风险 | 本次处理 |
|---|---|---|---|
| P0 | `evaluators/registry.py` 存在损坏的重复条件表达式 | Python 源码无法编译，所有评测被阻断 | 已修复并对全部 Python 源文件做编译检查 |
| P0 | 1 小时评测仍引用重命名前的 `skill_score` 局部变量 | 运行到评测时 `NameError` | 已修复为 `objective_score` 并回归验证 |
| P0 | reward 只相对持续性基线 | 在强日周期数据上过度估计候选改善 | 已改为仅用 `training_fit` 选择的持续性/24h 季节性强基线 |
| P0 | 晋级只要分数高 `1e-12` | 数值噪声可被当成真实改善 | DSH-native 已改为 `>0.005` + 同轮 centered max-T；其他当前路径使用配对区块 bootstrap |
| P0 | DSH-native 轮末 max-T 后又被旧 pairwise bootstrap 二次判定 | 两套选择规则可互相冲突，合法冠军无法写入 | Director 只接受账本中统一分析选出的 champion，不再重跑旧规则 |
| P1 | 跨代失败黑名单只按 predictor + 参数 | 相同参数但不同 agent 程序被误杀 | 改为完整 `behavior_digest` 精确匹配，并优先使用最近失败行为 |
| P1 | reward、skill、覆盖率惩罚和聚合混在大型 registry 中 | 难以独立证明边界和不变性 | 已抽取 `evaluators/objectives.py` 并建立纯函数测试 |
| P1 | 失败预测的惩罚基于原持续性参照 | 更换为强基线后可出现正 reward | 评分后置为不优于冻结基线，且失败行不进入模型 RMSE 证据 |
| P1 | 新旧 reward 归档合同未明确分版 | 历史重放与新语义可混淆 | 已新增 v2 reward 标识，decoder 同时接受 v1/v2 |
| P1 | 原区块置信度使用 MAE reward、目标时刻分块和成功行交集 | 与 canonical RMSE-skill 选择分数不一致，失败会改变 cohort 身份 | 已改为按预测起点分块的 RMSE 充分统计量、候选无关 cohort 与严格块身份配对 |
| P1 | 搜索父方案只取当前轮最佳 | 同一评测队列下本轮退化时会丢失跨轮较优探索方案 | 已增加同 cohort、同 evaluator 的跨轮搜索精英保留；不同窗口继续禁止原始分数互比 |
| P1 | baseline/profile 和 evaluator digest 未绑定全部评分常量 | 被篡改或语义漂移时可能继续比较 | 已校验 profile 内容/数据/切分 digest，并把 objective、reward、baseline、门禁和晋级常量纳入评测器 digest |
| P1 | 核心评测、样本执行和 director 文件过大 | 变更耦合、review 困难 | 已抽出 objective、baseline、promotion 三个责任；其余拆分列入后续路线 |
| P1 | 项目名称暗示完整生态世界模型，当前主体是温室环境预测 | 科学声称超出现有证据 | 本报告明确能力边界，建议按第 8 节扩展 |
| P2 | 当前分数是点预测 RMSE 技能，缺少概率评分 | 不能衡量校准度与尾部风险 | 建议新增 quantile/ensemble 预测与 CRPS/WIS |
| P2 | 只有单 episode 内反馈闭环 | 跨温室、跨年、跨作物外推能力不明 | 建议使用 episode-level nested validation 与独立 holdout |

## 4. 方案 B 已实施设计

### 4.1 Reward 合同

对观测 `y`、候选预测 `p`、冻结评分基线 `b` 和目标尺度 `s_t`：

```text
raw_reward        = |b - y| - |p - y|
normalized_reward = clip(raw_reward / s_t, -1, 1)
```

- `raw_reward` 保留物理单位，用于审计。
- `normalized_reward` 是跨温度、湿度、CO2 的学习信号。
- `s_t` 继续使用 `training_fit_std_floor@1`，不使用反馈标签拟合尺度。
- 新 reward 定义为 `absolute_error_improvement_vs_fit_selected_baseline@2`。

### 4.2 强基线选择

每个 `target × horizon` 独立比较：

1. persistence：用预测起点的目标值。
2. seasonal-24h：用目标时刻前 24 小时的值，且必须在预测起点之前可见。
3. 仅在 `training_fit` 中两者都可用的交集行上比较 RMSE。
4. seasonal 只在严格更优时入选，平局保留 persistence。
5. 选择结果、比较样本数、两类 RMSE、数据/分区 digest 共同形成 `baseline_profile_digest`。

强基线在模型预测完成后才应用，因此不会把季节性值偷偷换成模型输入。原持续性参照保存为 `model_reference_baseline`。

### 4.3 Objective 聚合

聚合版本升级为 `weighted_task_skill_reward@3`。三个目标等权，各时距等权，分母由预先声明的完整网格决定，不随候选返回的行数变化；负向 RMSE 技能分使用对称有界比率，避免多个劣于基线两倍以上的候选全部塌缩为 `-1` 而失去跨轮排序信号。

```text
candidate_nrmse = RMSE(p-y) / s_t
baseline_nrmse  = RMSE(b-y) / s_t
cell_skill      = clip(1 - candidate_nrmse / baseline_nrmse, -1, 1)

effective_cell = coverage * observed_component
                 + (1 - coverage) * (-1)
```

缺失 cell 不会从分母消失；部分覆盖只惩罚一次。选择分数仍是加权 skill，reward 作为独立学习信号，不用 reward 替代科学门禁。

### 4.4 晋级可靠性

新版评估与 incumbent 比较前要求以下合同一致：

- evaluation cohort digest；
- evaluator digest；
- objective aggregation version；
- baseline profile digest；
- dataset digest 和 split-manifest digest。

上述 v2 digest 必须是完整的 64 位小写 SHA-256，不能以“两边都缺失”冒充相等；区块 evidence 的目标权重和时距也必须与 evaluation 中冻结的 objective 合同完全一致。

可比后先要求：

```text
candidate_score - incumbent_score > 0.005
```

评测样本按 `origin_timestamp // 24` 投影为有序区块，私有统计计算保留全部区块，公共投影不暴露区块明细。每个 `target × horizon` cell 只保存 `eligible/succeeded`、候选与基线的归一化平方误差和、归一化 reward 和，不保存原始观测、预测或时间戳。失败样本保留在 `eligible` 中形成覆盖惩罚，但不进入候选 RMSE 证据。

候选与 incumbent 必须具有完全一致的区块身份和配置。每次重采样都从上述充分统计量重建与正式选择分数相同的“覆盖惩罚后加权 RMSE-skill”，而不是对 MAE reward 求均值。DSH-native 在不少于 8 个配对区块且具备连续 3 日重采样起点时，对同轮全部候选执行 1,000 次 centered max-T 重采样，并同时要求逐单元不退化；其他当前评测路径使用配对 bootstrap。证据不足一律 fail closed。手工提交的 DSH-native `approved` 只有在账本中已有相同轮次、相同候选的统一 champion 分析时才可写入。

这些选择结果属于自适应探索证据，不等同于独立 holdout 的正式确认。

搜索父方案与正式晋级继续分离：如果相邻轮次使用同一 evaluation cohort 和 evaluator，系统比较去除槽位、候选 ID 等轮内 tie-breaker 后的客观证据，保留较优的跨轮搜索精英；如果 cohort 或 evaluator 不同，则不作跨轮原始分数比较，仍使用当前轮父方案。

### 4.5 DSH 插件式自主研究—实现—评测闭环

新运行冻结 `dsh-model-search-reflect@1`，且只扩展 EcologyRSI 插件与 Python sidecar，不修改 DSH 核心。每代严格执行：

```text
上一代聚合分析 + 整批反思 + 父 Genome
  -> Researcher 生成检索词
  -> Host 受限检索 OpenAlex/内置可信目录并冻结证据
  -> Researcher 基于冻结证据生成 N 套不同方向
  -> Candidate Proposer 每槽实现一个方向
  -> Host 校验、编译和训练 smoke
  -> N 个候选在同一个冻结 training_feedback cohort 上评测
  -> Host 统一排序、稳定性检验、incumbent 比较和选优
  -> Generation Judge 基于聚合结果反思
  -> 反思摘要和建议检索词进入下一代
```

DSH 管理 Agent、子 Agent、上下文、模型路由和 Workflow；sidecar 管理数据权限、允许调用的数值工具、参数边界、评分、晋级和追加式账本。模型可以决定“查什么、提出什么假设、选择哪个登记轴、如何解释结果”，但不能直接读取标签、执行任意代码、改评测器、改数据切分或修改 DSH 框架。

每个研究方向必须给出一个且仅一个父 Genome 当前可执行的登记变异轴与目标：科学参数、已登记预测器或 Planner 指令/Skill 配置；候选 Host 校验还要求实际 mutation 与被分配方向完全一致。检索计划、研究综合或整批反思若形式上是 JSON、但语义不合法（例如检索词/关注方向超长、`mutation_target` 为空、引用未冻结文献或使用未登记目标），Host 会把有界 `validation_detail`、允许的长度/登记范围和被拒输出 digest 交给一个新的 DSH 子 Agent，自修正一次；第二次仍不合法则失败关闭，绝不由 Host 截断或伪造模型方案。

### 4.6 一次起点同时预测九个评测单元

温度、相对湿度、CO₂ 三个目标与 1h、6h、24h 三个时距组成 9 个 cell。现在一个预测起点只执行一条完整 Agent 链：

```text
Planner 一次 -> 登记的岭回归向量工具一次 -> Critic 一次
             -> Host 对九个 cell 评分 -> Reflector 一次
```

Planner 必须显式选择工具；岭回归只是被调用的数值工具，不拥有策略决策。Critic 不能自造预测值，只能接受或要求白名单内修复；Reflector 在真值解封和评分之后运行，只能给下一代建议，不能回写本次预测。九条评分记录和一条共享起点反思必须同时完成后才提交 checkpoint，因此进程恢复不会留下半个起点。

### 4.7 1521 的含义与速度边界

1521 是“评分 cell 数”，不是 1521 次模型预测调用：

```text
169 个连续预测起点 × 每起点 9 个 target×horizon cell = 1521
169 = (8 个 24h 配对区块 - 1) × 24 + 1
```

正式晋级需要至少 8 个配对日区块和连续 3 日 moving-block 起点，所以每个候选至少需要 169 条完整起点链，即通常为 169 次 Planner、169 次向量工具、169 次 Critic 和 169 次 Reflector。`samples_per_update=9` 只运行一个完整起点，可快速验证真实 Agent/工具/持久化链；它可以按配置继续多代，以验收前代反思、检索词和搜索父方案是否进入后代上下文，但始终标记为 `diagnostic_smoke`，不得产生正式晋级。诊断代不会执行只有正式选择才需要的父代同 cohort 控制复评，分析层也不再错误地索取这项不存在的证据。此前“进化很快”的现象主要来自小样本诊断或旧宿主快捷路径；严格协议下不存在零模型调用的闭式岭回归旁路。

### 4.8 选择集、最终测试集与预测器一致性

逐代候选比较使用冻结、轮转的 `training_feedback` cohort，而不是最终测试集。`training_fit` 只用于拟合预测器、缩放器和基线；`development/gate/external/hidden/final` 不参与每代反馈。否则“在测试集合上不断试方案”会把测试集变成训练信号，最终效果不再可信。

本次真实运行还发现并修复了一个关键一致性问题：任务元数据默认预测器曾为 `greenhouse-exogenous-ridge@1`，而种子 Genome 为 `greenhouse-horizon-targetwise-ridge@1`，导致 Host 用错误参数表拒绝模型提出的合法逐时距参数。新运行现在按最终预测器选择匹配的种子模板；旧运行恢复时以账本中冻结的父 Genome 为参数边界，保证真实执行对象、研究方向和 mutation 校验一致。

## 5. 代码结构调整

| 模块 | 职责 |
|---|---|
| `evaluators/objectives.py` | 有界 skill、归一化 reward、固定网格聚合和输入校验 |
| `evaluators/baselines.py` | 仅拟合集基线选择、因果可见性检查、profile digest 和评分后应用 |
| `evolution/promotion.py` | 通用路径的实用差异、区块证据和配对 bootstrap |
| `evaluators/fitness.py` | DSH-native 有效性、稳健性、moving-block max-T 与分层排名 |
| `core/sample_results.py` | 逐样本基线血缘和归一化 reward |
| `evaluators/registry.py` | 编排预测、后置评分基线、评测指标与产物，不再自行实现 objective 算法 |
| `core/director.py` / `evolution/analysis.py` | 统一生成级选择；Director 校验并持久化已选 champion |

## 6. 当前版本边界与可重放性

- 没有数据库 schema 迁移；新字段是 evaluation/sample archive 的可加性字段。
- 新 reward archive 使用 `absolute_error_improvement_vs_fit_selected_baseline@2`。
- 评测器 implementation 版本已提升，digest 同时冻结 objective/reward/baseline、归一化尺度、裁剪边界、硬门禁和晋级置信度常量；旧运行若合同不同会 fail closed，不会静默混用分数。
- baseline profile 在应用时会基于同一数据集的 `training_fit` 重新计算 canonical 选择并逐项比对；重算自报 digest 不能绕过 partition、selection rule 或 cell 内容校验。
- 模型预测路径仍使用原始 persistence 参照，强基线只改变评分语义。

## 7. 验收矩阵

| 验收项 | 预期 |
|---|---|
| 全部 Python 源码编译 | 无 SyntaxError |
| objective 纯函数测试 | 符号、有界性、缩放不变性、缺失分母、覆盖惩罚、错误输入全通过 |
| baseline 测试 | 选择、平局、非因果时距、fit-only、fallback、失败 reward 全通过 |
| promotion 测试 | 0.005 门槛、contract mismatch、pairwise CI、max-T 和 champion 绑定全通过 |
| evaluator 回归 | 1h 和 1/6/24h 评测、反馈聚合、数据隔离通过 |
| 完整 Python 测试 | Python 3.11 下共收集 823 项，819 项通过、4 项按环境跳过；包含真实 AGC 数据测试与源码交付验收 |
| JS 语法、插件 smoke、代理安全 | 源码与发布物验收均通过 |
| 0.3.26 源码 | Python、Node、浏览器 smoke、语法与交付脚本全部通过 |

### 7.1 真实 DSH-native 多代验收

本地追加式账本 `.runtime/final-review.X8lIXg/acceptance.sqlite3` 保留了成功与失败运行，作为故障和修复审计证据。成功运行 `run:0c55d048-c440-481e-bb1d-64ed4a40b05c` 完成 2 代、每代 1 个候选、每候选 1 个预测起点：

- 共完成 2 个 origin、18 个评分 cell，只产生 2 条 `DshPredictionToolExecuted`，证明一次工具调用联合返回 9 个 cell，而不是逐 cell 调 9 次工具。
- 每个候选均有 1 次 Planner、1 次向量工具、1 次 Critic、1 次评分后 Reflector，Host bypass 为 0。
- 两代均持久化 Search、Knowledge、Research、Analysis 和 Reflection；第 2 代搜索计划绑定第 1 代 analysis/reflection digest。
- 两代 objective score 分别为 `0.1115566977` 和 `-0.0076755704`，均因样本与区块证据不足而拒绝。
- 运行终态为 `completed / diagnostic_smoke_completed / diagnostic_smoke_completed_no_promotion`，`best_candidate_id=null`，没有把诊断证据冒充正式冠军。

该运行还真实暴露并验证了两处修复：第 2 代诊断分析不再强制要求仅正式选择才存在的控制复评；检索模型输出超出 Host 长度合同后可进行一次带验证反馈的模型语义重写。此前成功运行 `run:f3d3a249-f7cc-4c5a-9d58-239cfb0f5c92` 另覆盖了 1 代 × 2 候选并行批次。

## 8. 从当前核心到农业—生态世界模型的后续路线

### 阶段 1：评测可靠性补强（1–2 个迭代）

1. 在多个 greenhouse episode 上做 leave-one-episode-out 评估。
2. 将 development 用于限次模型选择，gate/external 只用于最终封存验证。
3. 新增 quantile 或 ensemble 预测，报告 CRPS/WIS、覆盖率和校准曲线。
4. 把 0.005 升级为基于真实运行差异分布预注册的最小效应，而非长期保留固定经验值。

### 阶段 2：多过程状态与物理约束（2–4 个迭代）

1. 状态扩展为室内气候、作物生物量/LAI/发育阶段、根区水分、EC/pH、氮素和能量状态。
2. 动作扩展为通风、加热、遮阳、CO2、灌溉和施肥。
3. 预测器输出多状态转移与守恒残差，reward 分解为预测技能、物理可行性和运行成本；不在单一分数中隐藏硬约束。
4. 接入 GreenLight/WOFOST/AquaCrop 等机理模型时，使用登记 adapter 和参数校准分区，不让策略模型直接修改物理方程。

### 阶段 3：长时滚动与决策（4–6 个迭代）

1. 从单步 teacher-forced 预测扩展为 24h、7d、作物周期滚动，分开报告一步误差和滚动漂移。
2. 引入不确定性传播、状态观测器、反事实不可识别标识和 off-policy 风险评估。
3. 先做决策支持，不直接声称可自主控制；所有动作建议继续经过宿主物理边界与人工治理。

## 9. 工程拆分建议

后续功能不应继续堆入三个超大文件。建议保持公开接口不变，逐步做机械式拆分：

- `evaluators/registry.py`：拆为 catalog/binding、one-hour orchestration、ridge orchestration 和 evaluation assembly。
- `evaluators/sample_execution.py`：拆为 contracts、scheduler、adjudication、checkpoint codec 和 diagnostics。
- `core/director.py`：拆为 evaluation recording、sample-result lifecycle、promotion、generation transition 和 recovery。
- 每次拆分前固定行为测试和 digest，不在同一变更中同时改算法语义。

## 10. 建议的发布判定

0.3.26 可作为“自主检索、自反思与同源多目标进化加固版”进入源码验收，但不应因此将项目宣称为已完成的农业—生态世界模型或生产控制系统。对外最稳妥的表述是：

> 一个具备可重放账本、时间前向评测、强基线 reward、逐样本工具反馈和统计晋级门槛的温室环境预测进化核心，为后续构建多过程农业—生态世界模型提供可审计基础。
