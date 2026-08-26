# EcologyRSI-DSH 0.3.33 可交付版设计

## 1. 目标

本次变更把当前 `0.3.32` 工作树整理为可复现、可安装、可回放、可持续运行的 `0.3.33` 交付版，同时删除有证据证明冗余的实现和发布资产。

交付成功必须同时满足：

1. 当前自进化数据可以从既有 SQLite 事件流完整回放，升级前后的运行身份、代次、候选和预算不漂移。
2. 用户设置“逐样本并发请求数 = 8”时，一个 run 最多向 DSH 提交 8 个样本链请求；DSH 宿主继续对相同 provider 执行全局最多 8 个在飞 stage 请求。
3. 用户界面的“每次更新预测单元数 = 500”表示 500 个完整预测时点；对于当前 3 个目标乘 3 个时距的任务，冻结到运行 manifest 的正式评分预算是 `500 × 9 = 4,500` 个评分单元。
4. 两阶段筛选的筛选记录、正式入围集合和淘汰事件可以被确定性验证和回放。
5. 干净源码、wheel、sdist、DSH 插件 tgz 和完整 delivery 包均可独立校验、安装和启动。
6. 只部署后端 `127.0.0.1:8777` 和前端/DSH 宿主 `127.0.0.1:8848`，不恢复该项目的其他服务。

## 2. 非目标与兼容边界

本次不做以下高风险改造：

- 不拆分 `strategies.py`、`sample_execution.py`、`gateway_sample_adapter.py` 等大型模块；只提取已经存在多份且可由契约测试覆盖的小型公共逻辑。
- 不删除 `dsh-strict-origin-bundle@3`、v3/v6 preset 或 v4/v7 preset。当前持久化数据库和回放测试仍包含这些协议。
- 不删除 `evaluators/uncertainty.py`。虽然当前主运行路径没有调用它，但 wheel/sdist 公共模块契约和测试仍要求该路径存在。
- 不增加 CO₂ 专用模型，也不增加分目标或分时距模型。
- 不改变正式候选的科学排序规则，不把筛选阶段的 `passed` 字段改造成硬淘汰门槛。64 个筛选时点不足以满足正式阶段至少 169 个完整时点的证据门槛，因此该字段只作为诊断证据。
- 不删除任何 `.runtime` 数据库、现有 `dist` 候选产物或用户运行证据，直到新的 `0.3.33` 产物完成验证。
- 不重写既有事件。新校验只约束新事件以及已经携带两阶段事件的运行；没有两阶段事件的历史运行继续按原协议回放。

## 3. 已确认的根因

### 3.1 两阶段事件只写不验

写侧已经产生 `CandidateScreeningRecorded`、`FormalSelectionCohortFrozen` 和 `CandidateScreenedOut`，但投影侧没有把前两类事件恢复为结构化状态，也没有验证代次、候选归属、摘要或集合关系。结果是格式合法但语义矛盾的事件仍可能被回放接受。

### 3.2 Python 调度使用向上取整

每个候选的 worker 数当前按 `ceil(sample_concurrency / candidate_concurrency)` 计算。候选并行后，本地 worker 总数会成为 `C × ceil(S/C)`。当 `S=8,C=3` 时会提交 9 个；当 `S=8,C=5` 时会提交 10 个。DSH 的共享 `ProviderStageGate` 会把相同 provider 的真实在飞 stage 请求压到 8，但多出的请求先进入宿主队列，造成“排队很多”以及进度估算失真。

### 3.3 预算被整除截断

API 目前只验证 `samples_per_update` 是范围内整数。v4 phase manifest 再按每个完整预测时点的评分单元数执行整除，非整倍数会被静默截断。例如 4,501 会变成 4,500，导致请求值、冻结值和页面展示不一致。

### 3.4 筛选进度混淆完成与成功

投影把完成 `sample.reflect` 的样本链直接统计为 `succeeded_samples`，并固定 `failed_samples=0`。结构化调用成功不代表科学预测结果成功，因此页面可能显示错误的成功数量。

### 3.5 发布包不是自包含的

源码配置要求版本化 DSH 插件 tgz，但 `.gitignore` 仍只放行旧版本文件；delivery 构建又排除了所有嵌套 `dist` 目录。干净克隆和解压后的完整交付包可能缺少 CLI 实际查找的插件文件。

### 3.6 发布门禁和文档落后

发布脚本只执行一个宿主 Node 测试，并重复运行 Python 源码验证。README、插件说明、发布清单和截图仍混有“并发 2”“每轮 1,600”“同一 provider 串行”等旧语义。测试源码还使用了 Python 3.12 才支持的 f-string 写法，与 `requires-python >=3.10` 冲突。

## 4. 运行正确性设计

### 4.1 精确的 run 级样本并发

在一轮候选执行开始时创建 run 级样本许可器，容量等于冻结 manifest 的 `sample_concurrency`。许可器在一个完整 origin 样本链进入 DSH 前获取，在该样本链完成、失败或取消时释放。

候选调度仍可并行运行最多 `candidate_concurrency` 个候选，但每个候选的 worker 不再独立决定可提交请求的总量。即使 worker 数按候选分配后多于 8，只有持有 run 级许可的 8 条样本链可以进入 DSH；等待发生在 Python 进程内，不进入 DSH provider 队列。

必须保证：

- 正常完成、异常、超时和取消路径都在 `finally` 语义下释放许可。
- 同一个 run 的筛选阶段和正式阶段复用相同上限语义，但不同阶段不会共享已失效许可。
- `sample_concurrency < candidate_concurrency` 时仍不超过样本并发上限。
- 当前默认 `sample_concurrency=8,candidate_concurrency=4` 的吞吐不下降。
- DSH 的共享 `ProviderStageGate(maxInFlight=8)` 保留，继续作为跨 run、按 provider 的最终物理上限；Python 许可器负责每个 run 的精确准入和减少无效宿主排队。

公开状态中分别展示：

- `run_in_flight`：当前 run 已持有许可的样本链数；
- `provider_in_flight`：DSH provider gate 的实际 stage 在飞数；
- `provider_queued`：已进入 DSH gate 的等待数；
- `local_waiting`：尚未进入 DSH、等待 run 级许可的样本链数。

若现有 API 不能提供全部字段，新增字段必须是向后兼容的可选字段，旧字段继续保留到后续版本。

### 4.2 完整预测时点与评分预算

用户界面继续默认显示 500 个完整预测时点。后端兼容字段 `samples_per_update` 继续保存评分单元总数，以免破坏历史 manifest；创建运行时由任务定义中的 `prediction_cells_per_origin` 计算：

```text
formal_prediction_origins = samples_per_update / prediction_cells_per_origin
```

新运行必须满足：

```text
samples_per_update % prediction_cells_per_origin == 0
```

不满足时 HTTP 创建接口返回 400，错误信息同时给出请求值、每个完整时点的评分单元数和最近的合法值，不再静默取整。历史 manifest 不重新计算。

当前温室任务的固定关系为：

```text
3 个目标 × 3 个预测时距 = 9 个评分单元/完整预测时点
500 个完整预测时点 = 4,500 个评分单元
169 个完整预测时点 = 1,521 个评分单元
```

页面、API 投影和日志必须同时标明“完整预测时点”和“评分单元”，禁止把二者都简称为“样本”。

### 4.3 两阶段筛选事件状态机

`RunState` 增加按代次组织的结构化状态：

- 每个候选一条筛选记录；
- 每代至多一个正式候选冻结集合；
- 每个未入围候选的淘汰状态和原因。

回放 `CandidateScreeningRecorded` 时验证：

- generation 等于候选所属代次；
- candidate 已在该代次登记；
- 同一候选不能出现内容不同的第二条筛选记录；
- origin 数、评分单元数与该运行冻结的 screening manifest 一致；
- cohort digest 是 64 位小写十六进制 SHA-256，且可以从事件的规范化输入重新计算；
- `passed` 只作为诊断字段保存，不参与入围资格过滤。

回放 `FormalSelectionCohortFrozen` 时验证：

- 所有 selected candidate 都来自同一代次且已有筛选记录；
- selected 集合无重复，数量等于该运行配置允许的正式候选数；
- selection digest 可以从按 candidate id 稳定排序后的筛选证据重新计算；
- 重放相同事件幂等，内容冲突则 fail closed。

回放 `CandidateScreenedOut` 时验证：

- 候选属于同一代次且已有筛选记录；
- 正式候选集合已经冻结；
- 该候选不在 selected 集合中；
- 重复相同淘汰事件幂等，理由或代次冲突则拒绝。

在部署前必须使用当前生产 SQLite 的只读副本执行完整 replay。任何历史事件不兼容都阻止部署，不允许通过修改数据库绕过。

### 4.4 筛选结果和进度语义

投影新增或明确以下三类数量：

- `completed_samples`：样本链已产生可解析 reflection；
- `succeeded_samples`：reflection 明确包含成功的科学 outcome；
- `failed_samples`：reflection 明确包含失败 outcome 或样本链终止失败。

没有足够结构化证据判断成功或失败时，只增加 `completed_samples`，不得推断为成功。页面筛选阶段主进度使用完成数，成功/失败作为辅助诊断。

`passed` 在页面上改称“筛选诊断通过”，并说明正式入围按约束违反数、分数和稳定 tie-break 排序，不把 64 点诊断通过当作 169 点正式证据门槛。

## 5. 有边界的代码去重

### 5.1 异常图遍历

在 `core/errors.py` 保留唯一的有界异常图遍历实现，统一处理：

- `__cause__`；
- 未被 cause 覆盖的 `__context__`；
- `ExceptionGroup` 子异常；
- 环检测；
- 最大深度/最大节点数。

model gateway、generation execution、auto progress 和 token/provider 错误分类只消费该遍历器，不再各自维护 while/stack 版本。测试覆盖 cause、context、group、循环和深度边界。

### 5.2 重试策略契约

将 director 与 state 中重复的重试状态、允许动作和边界常量移入单一 `core/retry_policy.py`。写侧和回放侧调用同一纯函数验证，但 state 模块不得反向依赖 director，以免扩大导入环。

### 5.3 协议能力判断

将散落的 `{v3,v4}` 字符串集合判断收敛为无状态 protocol capability helpers，例如：

- strict origin bundle；
- concurrent origin execution；
- two-stage screening support。

helper 只编码能力，不删除任何旧协议分支。v3 与 v4 的 replay fixture 必须分别通过。

### 5.4 不执行的大重构

重复代码扫描发现的递归 value validator、DSH identity/evidence 字段集合等仅在确认调用语义完全一致且测试可证明时合并。若两个调用点的错误消息、容错级别或安全边界不同，本次保留现状，不为减少行数制造共享耦合。

## 6. 删除与保留清单

### 6.1 删除或移出交付物

- 删除已跟踪且无引用的 `.superpowers/sdd/**/task-*-report.md` 临时报告。
- 删除工作树内可重建的 `.cache`、`.pytest_cache`、`.ruff_cache`、`__pycache__`、`*.pyc`、`build` 和 `*.egg-info`；不使用覆盖整个仓库的递归清理命令。
- `docs/superpowers/**` 和陈旧的内部整体 Review 报告保留在开发仓库，但通过公开文档 allowlist 排除出 wheel、sdist 和 delivery。
- 浏览器 `smoke.mjs` 保留在源码测试目录和源码开发包中，但从 wheel 运行时数据文件中移除。
- 删除测试中的未使用局部变量，并把 Python 3.12 专用的多行 f-string 表达式改为 Python 3.10 可编译的预先序列化变量。
- 数据集 MD5 校验调用显式使用 `usedforsecurity=False`，保留上游校验算法但消除安全扫描误报。

### 6.2 必须保留

- 当前 SQLite 数据库、运行事件和检查点；
- v3/v6、v4/v7 preset 及相关 replay 兼容代码；
- `evaluators/uncertainty.py` 及其公共导入路径；
- DSH runtime handshake fixture；
- `LICENSE`、`NOTICE`、README、CHANGELOG 和发布清单；
- 当前 `0.3.32` 产物，直到 `0.3.33` 的校验和、安装和运行验收全部通过。

## 7. 交付与构建设计

### 7.1 版本一致性

所有公开版本位置统一更新为 `0.3.33`，包括 Python version、pyproject、插件 package、插件 manifest、README、CHANGELOG、安装路径、产物名称和校验脚本。

构建验证必须扫描这些位置并拒绝版本不一致。`.gitignore` 放行当前版本化插件 tgz 的通用文件模式，但发布校验只接受与当前版本完全相符的一个插件包。

### 7.2 自包含源码与 delivery

源码归档和完整 delivery 必须包含：

```text
integrations/dsh_ecology_plugin/dist/
  ecologyrsi-dsh-evolution-plugin-0.3.33.tgz
```

delivery 的顶层 `artifacts/` 可以再保存同一 tgz，但两个文件必须字节一致。CLI 按现有嵌套路径即可安装，不引入第二套查找规则。

公开文档使用 allowlist，而不是递归打包整个 `docs`。交付验证必须显式断言内部 specs、plans、SDD 报告和旧 Review 报告不存在。

### 7.3 插件法律文件

独立 DSH npm tgz 必须包含与仓库根目录字节一致的 `LICENSE` 和 `NOTICE`。构建脚本负责复制到 staging，产物验证负责比较内容。

### 7.4 发布门禁

`verify_delivery.sh --source-only` 运行源码门禁；新增独立 artifact-only 路径，只验证已经生成的产物。`build_delivery.sh` 不得重复执行全套源码测试。

源码门禁至少包括：

- Python 当前解释器完整 unittest；
- Python 3.10 对 `src`、`tests`、`scripts` 的 compileall；
- Ruff 未使用导入/变量与语法检查；
- 插件全部 Node tests，而不是只运行 proxy security；
- 浏览器插件 smoke；
- 安全扫描中无未解释的高风险项。

产物门禁至少包括：

- wheel、sdist、DSH tgz、完整 delivery 和 `SHA256SUMS`；
- wheel 隔离安装与 CLI smoke；
- sdist 构建 wheel；
- delivery 解压后按 README 执行 editable install、安装 DSH runtime 并启动 smoke；
- 所有产物版本和源文件同源；
- DSH tgz 含 LICENSE/NOTICE；
- delivery 不含内部文档和缓存。

正式 release 要求 clean HEAD。开发候选允许显式构建 dirty 版本，但必须标注 dirty，且不能命名为最终 `0.3.33`。最终 delivery 增加 `BUILD-INFO.json`，记录 commit、`dirty=false`、构建时间基准和工具版本。

## 8. 文档与页面

README、插件 README、发布清单和页面文案统一使用以下术语：

- 默认每次更新：500 个完整预测时点；
- 当前任务等价预算：4,500 个评分单元；
- 筛选：每候选 64 个完整预测时点；
- 正式评估：Top 2，各 500 个完整预测时点；
- 默认候选并发：4；
- 默认逐样本并发：8；
- DSH 相同 provider 全局最多 8 个在飞 stage 请求，FIFO 排队；
- Python run 级许可器保证单 run 不超过配置值。

删除“并发 2”“每轮 1,600”“六个 v2 preset”“同一 provider 串行”和已移除 Token 上限的操作性说明。历史兼容默认必须放在单独的兼容章节，不能与新建运行默认混写。

部署后重新截取与 README 对应的网页截图；参数截图必须能看到 500、8、4，运行截图必须区分完整预测时点、评分单元、实际在飞和排队。

## 9. 测试驱动实施顺序

每项行为变更执行严格 red-green-refactor：

1. 先增加能够复现当前错误的最小测试并确认失败原因正确。
2. 只实现让该测试通过的最小改动。
3. 运行相邻测试，绿色后再提取公共逻辑。
4. 每个独立任务提交后进行规格符合性和代码质量复审。

关键失败测试包括：

- `S=8,C=3`、`S=3,C=8` 和多个候选取消时，进入 DSH 的 run 级最大并发不超过 S，许可不泄漏；
- 错误 generation、非法 SHA、未筛选候选入围、入围候选被淘汰、冲突幂等事件均使 replay fail closed；
- 4,501 个评分单元在 9 单元/时点任务上返回 400，4,500 正常创建；
- reflection 只有“调用完成”但科学 outcome 失败时，不计入成功；
- 干净源码归档和解压 delivery 均能找到并安装准确版本的 DSH tgz；
- 全部宿主 Node tests 被发布脚本执行；
- Python 3.10 compileall 通过；
- wheel/sdist/delivery 不含内部文档、SDD 报告和 wheel-only smoke 资产；
- 当前生产 SQLite 的只读副本可以在 `0.3.33` 上完整 replay。

## 10. 部署与回滚

实现和产物验证期间不停止当前服务，不修改活动数据库。

部署步骤：

1. 记录 `8777/8848` 当前 PID、版本、数据库路径、活动 run id、最新事件序号和自动推进状态。
2. 复制数据库并在副本上执行 `0.3.33` replay 验证。
3. 暂停自动推进并等待已提交的 DSH stage 到达安全边界；不删除 run，不归档 run。
4. 只停止本项目占用 `8777/8848` 的进程。
5. 使用最终 wheel 和 DSH tgz 启动新进程，继续指向原数据库。
6. 验证健康检查、页面加载、运行列表、当前 run replay、事件序号单调增加和 provider 在飞上限。
7. 恢复自动推进，并观察至少一个完整 origin wave 的提交、完成、失败和排队变化。

回滚条件包括：历史 replay 失败、事件序号回退/冲突、重复提交、provider 在飞超过 8、页面不能恢复当前 run 或新版本健康检查失败。回滚只切换程序版本，继续保留原数据库和升级前产物；禁止回写或删除事件。

## 11. 最终交付物

最终交付至少包含：

- `ecologyrsi_dsh-0.3.33-py3-none-any.whl`；
- `ecologyrsi_dsh-0.3.33.tar.gz`；
- `ecologyrsi-dsh-evolution-plugin-0.3.33.tgz`；
- `ecologyrsi-dsh-0.3.33-delivery.tar.gz`；
- `SHA256SUMS`；
- `BUILD-INFO.json`；
- 修改明细：按运行正确性、代码去重、删除项、构建交付、文档和部署分组列出文件与行为变化；
- 验证明细：记录命令、测试数量、退出码、production DB replay 结果和网页验收结果。

只有全部门禁通过并在 `8777/8848` 完成实际运行验收后，才能把 `0.3.33` 标记为可交付版本。
