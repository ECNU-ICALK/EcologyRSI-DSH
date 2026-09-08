# Changelog

## 0.7.10 — 2026-09-08

- 独立推理副本按执行 scope 隔离检查点、写入权限、用量和结算；恢复时保留原 Agent 工具及参数证据，完成结果原样复用。
- 轮末稳定性检查对已封存的只读证据使用原 JSON 表示计算摘要，修复全部预测完成后决策阶段的序列化异常。
- 暂停及终态保留已完成副本进度，停止无意义的实时 ETA；初筛淘汰、正式完成及缺失详情在网页统一展示。
- 补齐轮末反思方向的缓存类型登记，已完成运行的工作区复用持久快照，避免重启后重复重放大账本。
- 正常网页完成 1702 个预测起点的完整实验；两条候选均被科学门槛拒绝，保留原方案，未宣称独立最终验证或稳定预测提升。详见 `docs/refactor/WEB-EVOLUTION-0.7.10.md`。

## 0.7.9

- 独立评审输出上限统一为 4096 Token；增加 Kimi K3 低推理档位配置示例，避免默认最大推理耗尽评审提交空间。
- 继承 0.7.8 的并发结果保存、样本故障隔离、输出身份约束和请求节流修复。

## 0.7.8

- 网页实跑修复：约束样本输出身份；并发失败时保存已完成的其他结果。
- 将孤立输出额度耗尽纳入样本失败与惩罚评分；环境协议故障继续阻止运行。
- 工具反馈剩余调用次数，默认子任务启动间隔增至六秒，保留供应商自适应退避。

## 0.7.7 — 2026-09-08

- 删除研究自然语言的关键词执行门禁；候选只接受结构化变更坐标，原文保留审阅，不作为额外操作或取值来源。
- 保留受信组件、单一操作、参数范围、变更方向、行为去重及科学门禁；人工干预解析规则移回所属模块。
- 研究预设 v12 与该执行边界一致，继续从真实网页验证自主工具调用和完整预测链。

## 0.7.6 — 2026-09-08

- 修复研究方向校验把 per-origin/per-sample 计算成本和延迟误判为更改宿主样本数量的问题；真实采样变更仍拒绝。
- 延续 0.7.5 的预算、取消、推理配置、报告长度和历史首屏加载修复，继续实际网页验收。

## 0.7.5 — 2026-09-08

- 明确的原生输出预算与工具协议故障上升为运行故障，在等待候选线程退出前启动远端取消，保留最初原因，不把执行环境错误计为候选科学失败。
- DSH 原生推理档位、协议和模型参数配置纳入运行指纹；同名配置变化可被检测，凭据不进入公共目录。
- 研究报告推荐长度改为可审计提示，保持科学内容及硬上限校验；避免仅因摘要略长就暂停整个实验。报告格式升级 concise@2，研究预设升级 v11。
- 历史报告格式按原始版本读取，修复升级后运行列表请求失败、首屏无法加载的问题。
- 真实网页迭代与供应商推理配置验证见 `docs/refactor/WEB-EVOLUTION-0.7.5.md`。

## 0.7.3 — 2026-09-08

- 统一网页创建和原生样本 Agent 的输出预算，修复真实多工具预测被旧 4,096-token 配置截断的问题。
- 输出预算及工具协议故障停止后续样本准入，不再污染科学评分；保留真实错误及用量审计。
- 样本 Planner v8 减少重复试算；修复网页验收、交付预设清单和锁文件版本一致性。
- 真实运行进展与工程验证见 `docs/refactor/WEB-EVOLUTION-0.7.3.md`。

## 0.7.2 — 网页实跑后的工具、修正与进度一致性

- 保留同一 Agent 的一次合法输出参数修正，校验失败来源、顺序及唯一成功提交；修复额度耗尽后报告阶段失败，不再作为网关故障反复执行整段研究。
- 修复零残差参数复用已拟合缓存时错误要求残差特征的问题，仍校验因果基线来源。
- 知识目录补全基线对齐工具与当前评测器；研究上下文明确默认工具参数与最终 Agent 预测的区别。
- 留出阶段按两次独立推理累计真实执行量，独立检查点去重，进度预算与创建时容量一致；重复推理不增加统计独立样本量。
- 本轮尚未评测时显示“等待评测”，避免提前显示“未通过”。更新研究与样本 Agent 预设。

## 0.7.0 — Agent 策略闭环与运行重构

- 研究目标改为 Agent 最终预测质量；默认工具的零残差敏感性只作提示，不再裁剪整个 Agent 的搜索空间。
- 原生起点执行器脱离旧 Gateway 路由调度器；Critic v2 返回接受、修改或证据不足，修改请求回到 Agent 的有界重试。
- 工具调用记录包含调用 ID、参数、逐单元原始值与引用状态，分开统计工具误差和 Agent 调整效果；冻结跨代经验并绑定运行身份。
- 拟合与起点推理分离，训练数据身份驱动持久缓存；缓存使用 LRU 淘汰，工具探索额度由每次 Agent 尝试决定，拟合有并发上限和中途取消检查。
- 正式留出认证运行两次独立 Agent 推理，逐次配对检查收益，重复次数不增加统计独立样本量。
- 新增 Agent 策略包导出和加载接口，包含模型、可选工具配方、历史经验与运行合同；过程和候选页面共用预测方法说明。
- 身份授权与检索预算计数改用 SQLite 索引查询。
- 容量预览与创建共用日块证据检查；页面和合同预算计入独立重复推理，停止后的当前运行保持可见。

## 0.6.0 — Agent 自主预测

- 每个预测起点由 Agent 分析因果数值上下文，自主选择零次或多次工具调用，提交直接、模型、融合或调整后的最终数值。
- 四类登记岭回归支持受限参数调整、按需拟合与候选内缓存；训练仅使用 training_fit。
- 移除强制单模型输出和宿主静默修复；失败进入有界 Agent 重试，工具证据与最终预测分别持久化和校验。
- 评分归属于 Agent 最终结果，网页展示方法、置信度和最终尝试的工具调用；回放复用记录，不重新拟合。
- 原生样本 Planner/修复默认输出预算提高至 8192；启用新版样本协议和 v6 preset，不迁移旧运行协议。

## 0.5.2 — 启动与历史加载优化

- 运行列表按数据库索引读取最小选择信息，不再为首屏、历史分页或列表刷新回放完整运行及科学比较。
- 所选运行概况在后台加载，配置界面先可用；详情读取失败独立显示，较晚响应不能覆盖新的运行选择或创建操作。
- 精简列表合同，不在缺少证据时推断科学结果；保留分区检查、归档过滤、取消空运行识别及稳定分页。
- 数据容量只读检查移出全局写锁；加载中的进度与容量显示等待状态，避免误报零进度或不足。

## 0.5.1 — 主链一致性与按需读取修复

- 完整任务清单与简化请求统一解析执行协议和预测选择字段，冲突绑定在创建前明确拒绝。
- 候选生成后立即从其持久化基因组显示实际预测器和基线状态，避免使用运行初始配置。
- 过程摘要直接构建轻量投影，不再生成随后丢弃的算法详情、推理轨迹、完整基因组及研究计划。
- 过程摘要、候选详情、人工审核候选选项分开存储与加载，防止异步响应混用不同版本的数据；审核选项不再依赖先访问候选页。

## 0.5.0 — 运行中自主选择预测方案

- 移除网页创建时的预测方案配置，统一提交运行中选择策略；服务端冻结四种可用预测器与 v4 共同评测合同。
- 研究和候选生成阶段获得方案目录、是否启用残差模型的语义与可执行切换操作；切换后按实际候选模型继续优化参数。
- 以仅用训练段选定基线作为比较参照，允许按时距启用残差，或切回零残差；零残差跳过拟合且不再记录为求解失败。
- 过程和模型产物展示实际采用的预测器与基线状态；补充创建、研究决策、参数优化和跨模型一致性回归。

## 0.4.0 — 主链收敛与冗余清理

- 删除独立 `ecologyrsi_kernel`、`evolution_lab`、未接入的适配层及其插件、示例和测试脚手架；不再提供 `v2`、`policy-study` 或 `ecologyrsi-ai-evolve` 入口。
- 将主流程仍使用的规范身份与诊断、假设合同归入原生 `core/`；保留原生并发、冻结合同、事件回放和科学资格回归。
- 样本分页只接受当前接口合同并核验运行、候选与分页；移除旧响应解析和失败时的摘要预览回退。无归档明确显示无样本记录，显式演示保留演示标识。
- 清理旧迁移计划和安装包清单，网页、CLI、Python 包与 DSH 插件统一为 0.4.0。

## 0.3.60 — 分区加载与读取性能优化

- 首屏只加载运行概况；过程、候选、协作和训练资产分别按当前工作区读取。
- 完整训练轨迹按单个候选展开，历史运行每页 5 条，过程首次读取最近 60 条事件。
- GET 请求合并，数据描述与样本页短期缓存；服务端按运行修订缓存独立详情，限制缓存容量。
- 隐藏工作区停止重渲染；迟到详情不能覆盖较新的监控状态，区域失败可独立重试。
- 已完成运行复用按代码版本和账本修订校验的持久化概况，避免正常重启首屏恢复完整历史；完成后的吞吐率绑定完成时刻。
- 初筛样本封存结果按对应筛选证据校验，修复已淘汰候选的样本详情误报。

## 0.3.59 — 网页实测与一致性修复

- Complete the adaptive Top-2 decision barrier with durable independent reviews of
  both actual finalist revisions. Bind review evidence to final comparisons and
  verify it during writes and replay; missing or rejected reviews block adoption.

- Share adaptive start spacing across each provider after RPM throttling; coalesce
  a burst into one rate reduction per window. Classify explicit provider concurrency
  limits separately and reduce their physical window instead of the RPM rate.

- 修复网页表单实际提交时丢失预测器与评测器绑定的问题，完整测试真实表单事件、模型预检和创建请求。
- 根据实际冻结 cohort 在创建前检查局部与留出日块容量。
- 基线零修正时阻止不会改变预测的历史窗口／正则参数搜索，并正确路由逐时距弱点反馈。
- 修复排队取消、活动任务排空和等待定时器清理，保持真实并发资源计数。
- 统一创建、实时研究阶段与缺失分数展示，避免制造零分或虚假进度。
- 将原生 max-tokens 终态明确归类为输出预算耗尽，停止相同预算请求的重复重试。
- 新运行冻结简短研究报告、去重上下文与仅综合阶段16384输出上限；明确同轴不同假设可组成合法研究方向，历史运行契约不变。
- 将提案来源与实际产物修订分别绑定，评估、评审及正式验证共同核验真实版本，拒绝身份降级。
- 新的受保护比较同时核验双方执行链和完整成功评分，阻止将运行失败惩罚算作科学收益；比较结果带明确资格版本，旧决定仍按原版本重放。
- 新增真实已封存预测的无重拟合同快照复算，严格核对版本、参数、scope及工具回执。
- 以预留记录与会话的双身份索引消除累计用量入账全历史解析／重试压力，完整事件回放保持相等；修复真实轨迹检查点的嵌套类型恢复。
- 修复并行正式通道总进度、待比较修订编号和慢速历史读取；浏览器验收绑定实际加载的静态文件摘要。

## 0.3.58 — 基线诊断与稳健搜索

- 新增训练段物理隔离、三折前推和共同9格基线/候选诊断，支持安装后的CLI调用。
- 新预测器对齐训练期选择的基底，以1/6/24小时独立倍率控制修正，v3评测身份与零修正种子独立注册。
- 新运行可冻结实际增益、分项非劣与完整配对日块门槛；待复核记录可追溯，历史v3决定不变。
- 新增真实工具/schema传输预检与时效回执；全局操作锁和普通代理超时不受长预检影响。
- 同快照宿主执行对照验证数值身份，真实数据证实关闭有害6h修正能够改善结果，但不宣称已通过独立科学验证。


## 0.3.57 — 稳定进化运行改进

- 区分工具调用被输出为文本、输出预算耗尽与临时模型故障，避免对确定性失败进行相同请求重试。
- 保留研究、检索和预检的实际失败阶段，避免误标为样本筛选错误。
- 为候选提案提供宿主计算的单步参数区间，减少对数信任域边界舍入造成的无效提案。
- 新增多运行巡检：只控制清单内的运行，按已报告用量、截止时间和有效进度停滞请求暂停，保留实验记录。
- 修复配对更新中旧冠军检查点被误拒绝的问题；检查点与最终评分共享版本绑定规则，验证分组恢复和原子封存。
- 暂停后恢复重新计算无进展时长；监控器重启、用量增长不延长运行中的停滞窗口，截止时间和用量阈值保持有效。
- 修复完整一代结束后代数推进引起的 67% 进度回退、已封存筛选证据被误判中止，以及轮末共享建议评审未完成的误报；按统一单位汇总筛选、配对更新和留出评分项，避免重复计数。
- 源码发行包显式包含最终聚合 JSON；不包含运行账本或原始评测记录。
- 轮末独立反思复用宿主模型事实，明确残差尺度不由 ridge 拟合自动学习、尺度布局变化不增加相同特征配置下的拟合系数；自由文本机制解释仍按假设处理。


## 0.3.56 — 2026-09-07

- 完成渐进式模块重构与研究诊断接入；保留原工作台、原生数据/评测链与历史回放。
- 原生研究契约失败明确停止，禁止错误进入缺少候选方向的宿主回退；输出 Schema 绑定冻结证据与合法修改目标，合成阶段限制输出并关闭重复动态检索。
- 独立持久化成功、失败、取消和超时子会话用量；按因果前缀验证整批回放并去重快照；中断流不冒充完整计量，界面显示覆盖率及未计入的检索用量范围。
- 新建运行使用 schedule/3：跨日更新批次、目标时间隔离、禁止循环复用、真实容量检查；默认 200 个更新时点。既有 schedule/1、/2 冻结身份仍可回放。
- 网页数据页按运行冻结的四阶段协议读取实际拟合分区，校验协议漂移。
- 交付归档包含重构审查文档；工程验收与独立科研验证分开记录。


All notable changes to EcologyRSI-DSH are recorded in this file.

## 0.3.55 - 2026-08-28

### Durable champion–challenger adaptive trajectories

- Evaluate each schema-v2 local challenger against the current lane champion
  on the same frozen 50-origin cohort. Persist both evaluation arms, their
  score delta and gates, and the resulting lane champion so a negative score
  can improve a more-negative champion without treating a worse challenger as
  accepted.
- Generate the next bounded proposal from the durable champion. Rejected
  challengers remain reflection evidence but never become mutation parents;
  the final batch creates no unvalidated child, and generation holdout binds
  only the two final lane champions plus the global incumbent.
- Re-derive every host-owned comparison gate during both event writes and
  ledger replay, fail closed when strict-chain or execution-count evidence is
  missing, and reject forged ancestry, post-final local artifacts, or holdout
  bindings that do not name the completed trajectory revision.
- Keep explicit schedule-v1 `prequential` runs replayable while making
  schedule-v2 `paired_champion_challenger` the default for new runs. Report the
  paired formal upper bound as 1,900 candidate-origin occurrences and the full
  default generation as 2,663 occurrences / 23,967 scoring cells without
  inflating the 500 formal unique-origin source capacity.
- Expose durable comparison decisions in public events and trajectory rows.
  The browser now distinguishes frozen, promoted, retained, pending-next, and
  Host-rejected states instead of labeling schema-v2 challenger creation as
  generic “已应用”.

### Origin-scoped recovery and truthful live progress

- Keep a terminal outcome scoped to the single forecast origin that produced
  it. One failed concurrent child can no longer suppress deterministic Host
  repair for unrelated origins; finite physical-range rejections continue
  through the frozen local repair sequence without launching another DSH
  child.
- Treat structured-result persistence as infrastructure, not scientific
  evidence. DSH retries that boundary within a bounded attempt budget, and an
  exhausted persistence failure aborts the work unit instead of writing a
  model-score penalty.
- Derive live throughput and ETA from durable, complete origin result batches
  as soon as two origins settle. Coarser screening/formal/holdout phase
  boundaries remain a replay-safe fallback.

### Delivery

- Align the Python package, browser plugin, Host plugin, lockfile, legal
  metadata, documentation, and packed Host artifact at version 0.3.55.

## 0.3.54 - 2026-08-28

### Deterministic repair and durable adaptive execution

- Split sample retries into three explicit routes. A critic-selected repair
  keeps its critic provenance; a Host physical-range rejection now executes
  the frozen derived repair sequence locally; only a genuine transient failure
  before any prediction/tool evidence may create a fresh Planner child. This
  removes the repeated-predictor loop that exhausted Repair output tokens and
  stalled live 64-origin runs.
- Give a genuine pre-tool Planner retry the same bounded 4,096-token output
  allowance as the original Planner. Tighten the sample Planner preset so it
  submits the registered vector tool result immediately and leaves numeric
  validation and repair to the Host. Publish that changed immutable preset as
  `ecology-sample-planner-v5` so an existing DSH installation upgrades without
  accepting same-ID content drift.
- Validate generation comparison reports against the fitness profile's exact
  target-by-horizon grid. Missing, duplicate, extra, or malformed cells now
  fail closed instead of defining a smaller comparison grid from their own
  incomplete output.
- Recover an adaptive batch after a crash between `GenerationBatchStarted` and
  candidate/cohort freezing, and make candidate revision and holdout-freeze
  replay idempotent while still rejecting real identity or status drift.

### Parameters and monitoring

- Keep full-origin admission at 64 by default and configurable through 128.
  Present the separate per-origin vector capacity as nine atomic scoring cells,
  preventing it from being mistaken for either origin concurrency or nine
  independent model requests.
- Label DSH counts as unfinished child tasks, model termination errors by their
  real scope, and formal/holdout counters as phase totals. Show Provider waiting
  only when the backend exposes an exact gate snapshot.

### Delivery

- Align the Python package, browser plugin, Host plugin, lockfile, legal
  metadata, documentation, and packed Host artifact at version 0.3.54.

## 0.3.52 - 2026-08-28

### Provider-rate and capture recovery

- Pace new structured children at a three-second default interval while
  preserving the configured 64-origin Host admission and 128-origin physical
  ceiling. A sample planner normally uses three model turns, so this prevents a
  recovered run from exhausting a user-RPM window before the first feedback.
- Detect terminal `RATE_LIMIT`/HTTP 429 failures, prefer DSH's bounded
  `providerRetryAfterMs` signal, and use a redacted message value only as a
  fallback. Retry the same Host origin after the absolute cooldown without
  incorrectly reducing the provider concurrency capacity; re-check an
  in-progress spacing wait when a later Retry-After extends it.
- Detect the exact zero-turn DSH child shape produced when a provider rejects a
  request after Session allocation but before any model/tool boundary, then
  back off and retry it in a fresh bounded child.
- Wait for a bounded final Session projection before classifying a missing
  structured result. An exact `INVALID_ARGS` structured-output rejection is
  retried in a fresh child even if DSH later projects another successful call;
  reused call identities and authorization/unknown-tool failures remain
  fail-closed. Local capture retries no longer penalize provider capacity.
- Raise the native `sample.plan` output ceiling from 2,048 to 4,096 tokens after
  live validation found rare truncations after a successful prediction tool.
- Stop presenting Host-derived `queued_batches` as an exact Provider queue in
  the browser. Provider waiting is shown only when an explicit gate snapshot is
  available; Host concurrency, Host waiting, and awaiting submission remain.

### Delivery

- Align the Python package, browser plugin, Host plugin, lockfile, legal
  metadata, documentation, and packed Host artifact at version 0.3.52.

## 0.3.51 - 2026-08-28

### Effective concurrency and durable execution

- Remove the hidden eight-request provider cold-start window. A default
  64-origin run now enters DSH at its configured run limit immediately, while
  the provider-wide ceiling remains 128 and AIMD still reduces the window
  after an observed model or congestion failure.
- Advance both Top-2 finalist lanes in the same scheduler turn when candidate
  concurrency permits it. Their two 50-origin batches share the existing
  run-level 64-request admission budget; local-edit boundaries remain
  serialized and recheck pause/cancel before starting.
- Wait briefly for the DSH Session projection after a child result completes,
  eliminating the race that intermittently reported
  `structured_result_persist_failed`. Treat an exact structured-output
  `INVALID_ARGS` receipt as a missing capture and retry it in a fresh bounded
  child; retain only Sidecar-redacted diagnostics in local logs.

### Adaptive evidence and truthful monitoring

- Build adaptive ranking and selection reasons from the frozen three-arm
  comparison gates instead of candidate slot order. Diagnostic screening or
  failed-holdout scores may remain visible without a formal rank, and an
  incumbent win now remains the next generation's search parent.
- Carry the exact frozen effective final revision into the next generation's
  search, research, batch and proposal contexts. A same-generation
  `screened_out` incumbent may remain the parent after winning the three-arm
  comparison; the system no longer falls back to its original R0 genome or a
  failed finalist.
- Feed the generation reflector the selected final revision and digests, all
  nine target/horizon cell gates, bounded failure reasons, and the complete
  ten-batch revision/edit chain while distinguishing the outer generation
  mutation from within-epoch local edits.
- Derive the live adaptive phase from durable state-machine boundaries. Origins
  that were not executed because a candidate terminated early are shown as
  skipped, not queued forever, so phase and epoch progress cannot remain stuck
  on screening after Top-2 has already been frozen.
- Present full-epoch and current-phase progress separately, including Host
  admission, DSH in-flight work, provider waiting, pending submissions, and
  skipped origins. Treat `screened_out` as terminal, preserve null microbatch
  scores as “waiting”, use current DSH activity to avoid false stalled alarms,
  and show invalid parameter input instead of silently normalizing it.
- Give adaptive screening, formal-batch, holdout and comparison events stable
  public timeline types and Chinese labels instead of grouping them under the
  generic “system event” fallback.
- Remove the three obsolete browser-side parameter normalization helpers now
  that creation and summaries share strict validation.

### Delivery

- Align the Python package, browser plugin, Host plugin, lockfile, legal
  metadata, documentation, and packed Host artifact at version 0.3.51.

## 0.3.50 - 2026-08-28

### DSH child compatibility and cleanup

- Run every `sample.plan` as a direct one-shot structured DSH child and pass
  its 2,048-token output cap through the supported
  `SubagentStartRequest.agentOptions.maxTokens` field while inheriting the
  frozen provider and model from the retained role host.
- Remove the obsolete per-sample Workflow executor, script template, active
  Workflow registry, duplicate cancellation/disposal paths, worker-thread
  preset dependency, and their redundant tests. Rename the current sample
  execution mode to `dsh_native_agent`; this release does not accept the old
  mode name.

### Truthful failure evidence and progress

- Treat prediction-tool execution as intermediate evidence only; a remote
  origin completes only after its structured child result is accepted. Keep
  Host-settled success/failure counts separate from remote completion,
  provider admission, retries, and structured-child request failures.
- Aggregate active screening heartbeats with sealed candidate checkpoints so
  failed origins advance the settled counter instead of leaving total progress
  at zero, without double-counting a sealed candidate.
- Retain deterministic worst-case failure penalties only in the private
  scoring archive. Public sample APIs, candidate previews, live monitoring,
  inference traces, and training trajectories now expose failed rows as having
  no model prediction, error, or reward.
- Canonicalize each local-edit operation bundle and reject an exact bundle that
  was already rejected for the same immutable candidate revision. The original
  proposal remains durable audit evidence, while the Host records
  `duplicate_recent_rejected_bundle` without creating a redundant child
  revision; a changed parent revision remains eligible for reconsideration.

### Delivery

- Align the Python package, browser plugin, Host plugin, lockfile, legal
  metadata, documentation, and packed Host artifact at version 0.3.50.

## 0.3.49 - 2026-08-28

### Durable adaptive scopes

- Stream and resume origin rows, model usage, and progress in screening, every
  formal local batch, and every holdout arm instead of waiting for an entire
  64/50/169-origin scope to return.
- Atomically seal each scope's complete sample checkpoint with its screening,
  formal-batch, or incumbent-holdout result; keep finalist canonical outcomes
  on the existing atomic evaluation boundary and reject late writers.
- Authorize incumbent holdout replay only through the exact frozen arm binding,
  including generation-zero screened-out incumbents and historical champions,
  without weakening screening or formal candidate-state fences.

### Bounded native model traffic

- Freeze a 2,048-token output cap for every DSH-native sample operation and
  carry it through both direct children and Workflow Engine children.
- Replace raw history/feature prompt duplication with a content-addressed
  routing manifest containing bounded target, horizon, range, digest, and Host
  anomaly summaries; the representative payload is 32.8% of its prior size.
- Make the non-native ModelGateway resolve routing-manifest v2 variants
  directly instead of applying defaults that v2 deliberately removed.
- Keep `sample.plan` output schemas stable across origin waves for provider
  prefix-cache reuse while retaining Host-side exact wave, sample-set, tool,
  and decision validation.

### Truthful progress and delivery

- Count only complete task-sized primary waves as completed prediction origins.
  Report sparse repair waves and primary/repair request counts independently in
  the API and browser—including repair-only in-flight states—so repairs cannot
  inflate progress or settlement queues.
- Align the Python package, browser plugin, Host plugin, lockfile, legal
  metadata, and packed Host artifact at version 0.3.49.

## 0.3.48 - 2026-08-28

### Progression and lifecycle recovery

- Replace per-sample full-run projection replays with an O(1) lifecycle lookup
  backed by a partial covering index, keeping 64-way origin execution from
  starving monitor and control traffic as the event ledger grows.
- Keep diagnostics read-only while moving dead retry-timer and orphaned-running
  run repair into scheduler-owned maintenance; reject work-unit success unless
  the durable run sequence actually advances.
- Pause unclassified Host faults at the current checkpoint instead of replaying
  ambiguous work, and durably reconcile native DSH pause/cancel quiescence so a
  Host transition cannot silently leave its DSH run active.

### Adaptive evidence safety

- Record an idempotent start boundary for each finalist/incumbent holdout arm
  and project its live completed, in-flight, queued, and unsubmitted origins
  without waiting for the entire 169-origin arm to settle.
- Separate complete-origin success from prediction-cell coverage. Require the
  frozen origin-integrity threshold and strict agent chain before another local
  mutation, rolling back to the parent revision when a changed lineage violates
  that boundary.
- Reserve whole registered-pipeline selection for the outer four-candidate
  search while retaining bounded parameter and program edits inside each
  finalist's 50-origin local batch; the deterministic Top-2 contract is
  unchanged.

### Observability, evidence size, and delivery

- Show origin success and scoring-cell coverage independently, preserve the
  distinction between Host-settled and remotely completed work, and avoid
  rendering missing ratios as zero.
- Compact `FormalBatchEvaluated` to aggregate decision inputs, immutable
  science/data digests, and one canonical compressed trace archive. Drop the
  duplicate expanded execution records and browser preview; revision-scoped
  checkpoints and the retained archive preserve replay and failure audits.
- Align the Python package, browser plugin, Host plugin, lockfile, legal
  metadata, and packed Host artifact at version 0.3.48.

## 0.3.47 - 2026-08-28

### Adaptive decision safety

- Give the local editor the exact current registered pipeline, bounded
  parameters, workflow profiles, and eight most recent proposal outcomes;
  reject mutations whose executable behavior is unchanged.
- Roll back revisions after constraint-derived sample failures, keep the
  current revision for infrastructure-only coverage loss, and expose each
  safety decision as a distinct durable trajectory outcome.
- Compare both finalists and the incumbent in one centered max-T family with a
  practical-delta threshold, then recover holdout completion only from an exact
  revision, cohort scope, and artifact binding.

### Recovery and observability

- Persist full local-edit proposals before child creation, recover terminal
  command receipts only from matching command evidence, fence terminal
  knowledge writes, and publish stable failure locations down to batch/work
  unit scope.
- Add rolling origin throughput, ETA, admission congestion, live adaptive batch
  projections, and compact monitor responses with best-effort structural
  hydration and event-tail degradation.

### Budget semantics and delivery

- Always expose de-duplicated DSH Session provider usage while removing the
  hidden 100M-token native default. DSH-native manifests reject `token_limit`
  because provider reports are telemetry rather than an atomic reservation
  ledger; the separate sample-gateway hard budget is unchanged.
- Align the Python package, browser plugin, Host plugin, lockfile, legal
  metadata, and packed Host artifact at version 0.3.47.

## 0.3.46 - 2026-08-28

### Adaptive evolution

- Keep the outer four-candidate, deterministic Top-2 selection contract while
  giving each finalist a 500-origin epoch composed of configurable 50-origin
  local batches and zero-to-five Host-bounded edits per batch.
- Alternate complete batch/edit pairs between both finalists, recover an
  interrupted half-pair before switching lanes, and retain a same-cohort
  round-end comparison against both finalists and the incumbent.

### Runtime reliability

- Replace per-origin full run replays with indexed revision lookups and one
  validated immutable candidate-identity cache per run.
- Make health responses independent of the model gateway and event-ledger hot
  path, and raise the sidecar listen backlog to cover the supported 128-request
  sample-concurrency ceiling plus operator traffic.
- Give repeated candidate-spawn attempts a stable event identity and reject a
  second durable spawn source for the same candidate.

### Observability and interface

- Separate total-run progress from current-epoch progress; report settled,
  in-flight, queued, unsubmitted, and settling origins without fabricating
  success or failure counts.
- Add the two-finalist microbatch/revision trajectory table, cohort-safe score
  presentation, active DSH child-stage heartbeat, bounded incremental event
  streaming, and explicit stall diagnostics.
- Treat the health probe as advisory in the browser and retain the last valid
  run view during a transient catalog or run refresh failure.

### Cleanup and delivery

- Remove obsolete managed preset trees and packaging compatibility branches;
  install and verify exactly the six current DSH role presets.
- Align the Python package, browser plugin, Host plugin, lockfile, legal
  metadata, and packed Host artifact at version 0.3.46.

## 0.3.33 - 2026-08-27

### Runtime correctness

- Bound each run to one immutable sample-admission limit: new strict runs default
  to 64 concurrent origin chains, accept 1–128, and share a provider-wide FIFO
  cap of 128 physical DSH stage requests.
- Project the two-stage evaluation accurately: every candidate receives a
  disjoint 64-origin screen, Top 2 are frozen deterministically, and each
  finalist receives the configured 500-origin formal window.

### Replay safety

- Centralize exception classification and retry policy, retire completed screen
  launches, and preserve durable sibling-settlement anchors before requeueing a
  recoverable parallel evaluation.
- Re-admit running-but-idle auto-progress work from replayed scheduler state
  without discarding its existing evaluation checkpoint.

### Cleanup

- Remove duplicate runtime contract logic while retaining explicit compatibility
  behavior for historical manifests and immutable prior preset IDs.
- Separate provider-admission queue counts from origins that have not yet been
  submitted, preventing the browser from presenting both states as one queue.

### Delivery

- Make the packed DSH plugin a self-contained, versioned source artifact and
  verify byte-for-byte agreement with the selected package sources.
- Fail closed on symlinked, traversing, sensitive, special-file, stale, or
  ambiguous release inputs; propagate JavaScript syntax failures and reject
  unexpected delivery-verifier arguments.

### Deployment

- Align Python, browser, manifest, npm-host, lockfile, NOTICE, release paths, and
  the tracked DSH plugin artifact at version 0.3.33.
- Document the twelve installed immutable preset IDs, the six current runtime
  roles, the 64/128 concurrency boundaries, and the 500-origin formal workflow.

## 0.3.32 - 2026-08-26

### Fixed

- Anchor recoverable parallel-evaluation failures after every admitted sibling
  settles, so a successful DSH child cannot supersede another child's retry and
  leave an unfinished run with an idle scheduler.
- Requeue durable auto-progress runs discovered in the running-but-idle state
  during scheduler diagnostics, preserving their existing evaluation checkpoint.

## 0.3.31 - 2026-08-26

### Fixed

- Project two-stage screening progress directly from durable DSH child events,
  so the workbench reports completed, in-flight, and not-yet-submitted origins
  instead of showing a live screening pass as an empty scheduler queue.

## 0.3.30 - 2026-08-26

### Fixed

- Raise the sample critic/reflect operational deadline from 3 to 10 minutes so
  valid provider responses under eight-way load do not occupy a worker until a
  premature timeout prevents origin-chain refill.

## 0.3.29 - 2026-08-26

### Changed

- Replace the provider-wide serial 60-second launch interval with a FIFO
  concurrency gate capped at eight physical requests, and run complete origin
  chains concurrently under the strict `dsh-strict-origin-bundle@4` protocol.
- Evaluate every sibling on a disjoint 64-origin screening window, freeze a
  deterministic Top 2, and use the configured 500-origin window only for the
  formal pass.

### Fixed

- Reset native-runtime retry epochs after durable DSH success without
  invalidating legacy 0.3.27 retry chains during ledger replay.
- Prevent same-generation structured success from leaving a recovered run in
  an endless native-runtime retry cycle.

## 0.3.28 - 2026-08-26

### Changed

- Make the workbench update budget count complete prediction calls instead of
  internal target-by-horizon scoring cells. The default 500 complete
  predictions are converted to 4500 scoring cells for the registered
  nine-cell greenhouse evaluator before the existing backend request is sent.
- Show both complete-prediction and internal-cell counts in parameter previews,
  and express the formal-selection boundary as 169 complete predictions / 1521
  scoring cells.
- Raise the workbench default and invalid-value fallback for per-sample request
  concurrency to 8. Existing runs and the raw backend API defaults are unchanged.

## 0.3.27 - 2026-08-25

### Changed

- Make bounded `web_search` available to every DSH reasoning role after its
  required Skill call, so a stage can retrieve evidence when a question arises
  instead of relying only on the generation bootstrap search plan.
- Keep provider routing Host-owned: use DSH web search first, assess evidence
  quality deterministically, and invoke the bounded OpenAlex metadata adapter
  only after a primary technical failure or quantitative insufficiency.
- Persist and project replayable `DshRetrievalExecuted` receipts while keeping
  dynamic search advisory-only and outside trusted prediction, gate, and
  promotion evidence.
- Publish the retrieval-capable roles under immutable `v4`/`v7` preset IDs;
  retain the prior preset trees and legacy tool profile so an upgrade never
  mutates an already-installed preset in place.

### Fixed

- Preserve partial DSH results across multi-query failures, suppress provider
  error details, propagate explicit cancellation, and serialize duplicate
  completion for one frozen stage identity.
- Give the optional OpenAlex completion its own bounded sidecar deadline and
  require the dynamic retrieval module in npm and delivery artifact checks.

## 0.3.26 - 2026-08-24

### Changed

- Publish Researcher and Generation Judge `v6` Skills that distinguish
  diagnostic ranking from selection evidence and forbid claims that Planner
  instructions can change Host-owned sampling or prediction-tool values.
- Split single-candidate judging from batch reflection into separate Skills,
  and persist one Host-authored rank-to-candidate-to-direction outcome mapping
  for replay and next-generation research.
- Require every scientific-parameter direction to declare `increase` or
  `decrease`, jointly preflight sibling behaviors, and bind the proposer output
  to that declared sign; predictor and instruction directions use `select`.
- Make `mutation_direction` mandatory at both the JSON Schema and Python replay
  boundaries; legacy direction records without it now fail closed.
- Expose the frozen sample class, cells per origin, origin budget, effective
  cell budget, unused remainder, selection eligibility, and required terminal
  outcome as a machine-readable research and reflection contract.

### Fixed

- Reject and semantically retry diagnostic directions whose success criterion
  requires eligibility, gate passage, or promotion, and instruction-profile
  directions that claim sampling or numerical forecast effects, while allowing
  explicit negated invariants such as “RMSE unchanged”.
- Reject exact parameter assignments hidden in direction prose, reserve legal
  behavior witnesses across every sibling axis, and retain additional
  continuous witnesses after all eight hard-avoid slots are occupied.
- Keep negation local to its actual clause when screening Host-owned claims,
  recognize English and Chinese parameter-assignment aliases, and generate 16
  distinct continuous witnesses even when the parent is close to a bound.
- Report non-divisible budgets accurately: the default 1600-cell budget exposes
  177 complete nine-cell origins, 1593 effective cells, and 7 unused cells.
- Give DSH stage requests a timeout long enough for the DSH-owned two-attempt,
  30-minute research deadline, preventing the sidecar from abandoning a live
  Researcher and starting duplicate retries after 11 minutes.

## 0.3.25 - 2026-08-24

### Changed

- Preflight every model-authored candidate direction against the frozen parent,
  registered single-operation mutation contract, compiler, sibling directions,
  and sufficiently supported hard-failure history before proposal generation.
- Publish Researcher and Generation Judge `v5` Skills that require each
  hypothesis to be completely testable by one registered operation.

### Fixed

- Preserve `insufficient_evidence` on current-run cross-generation summaries so
  one-origin diagnostic candidates remain soft research evidence instead of
  becoming permanent behavior bans.
- Compact verbose per-cell tool statistics within the current-run experience
  budget, preventing later research generations from failing before launch.
- Reject predictor-switch directions that also require concurrent parameter
  assignments and return precise Host feedback for a fresh model-authored plan.
## 0.3.24 - 2026-08-24

### Changed

- Give autonomous research stages an independent 30-minute operational
  timeout while retaining the bounded timeout for ordinary structured stages.
- Supply the complete target-horizon objective matrix to search planning,
  evidence synthesis, and generation reflection, and require research and
  judging Skills to preserve all nine greenhouse forecast cells.
- Publish the changed researcher and generation-judge presets under immutable
  `v4` identities and retire their installed `v3` predecessors on upgrade.

### Fixed

- Treat one-origin diagnostic evidence as a soft research lesson rather than
  a permanent failed-behavior ban, so later generations can retest a predictor
  with a distinct parameter or Planner-Skill hypothesis.
- Keep model-provider retry pacing run-local and prevent a slow research turn
  from invalidating otherwise healthy multi-agent evolution runs.

## 0.3.23 - 2026-08-24

### Changed

- Use the registered algorithm capability catalog as the single source of
  predictor/evaluator compatibility for Genome compilation and candidate
  research boundaries.
- Precompile every model-authored Genome mutation before candidate creation;
  invalid predictor/evaluator combinations now enter bounded proposer repair
  instead of terminating the generation.

### Fixed

- Allow the registered targetwise ridge predictor with the multihorizon v2
  evaluator and reject the one-hour rolling predictor from both multihorizon
  evaluators.

## 0.3.22 - 2026-08-24

### Changed

- Require every DSH child to load its role Skill before any prediction or
  structured-output call, and persist the verified Session event ordering as
  replayable evidence.
- Expand autonomous candidate directions from numeric parameters alone to
  three Host-registered axes: scientific parameters, predictor pipelines, and
  Planner instruction/Skill profiles.
- Carry the candidate-selected Planner instruction and Skill profile through
  proposal persistence, Genome compilation, sample context, and DSH execution.
- Release the Skill-bearing role presets under immutable `v3` identities and
  remove superseded `v1`/`v2` presets during installation.

### Fixed

- Reject a candidate whose persisted Agent profile does not match the
  materialized Genome, preventing evaluation under a different Planner Skill.
- Make Skill-call evidence mandatory for accepted and replayed structured
  results, including strict call order for the joint prediction tool.

## 0.3.21 - 2026-08-24

### Changed

- Remove obsolete top-level compatibility modules, v1 role presets, and old
  catalogue fixtures from the source delivery; the current package and DSH
  runtime use only the canonical package layout and v2 presets.
- Extend the native acceptance runner to reattach to an existing `run_id` and
  use a one-hour default deadline appropriate for fully model-backed stages;
  require the configured generation count and verify that each later search
  plan consumes the preceding analysis and reflection digests.
- Allow complete diagnostic Agent chains to continue across the configured
  generation budget so cross-generation learning can be exercised cheaply,
  while keeping diagnostic evidence categorically ineligible for promotion.

### Fixed

- Require same-cohort parent/control replay only for `selection_eligible`
  generations; multi-generation diagnostic smoke runs now retain the strict
  no-promotion boundary without failing because deliberately omitted formal
  control evidence is unavailable.
- Feed invalid autonomous search plans back to one fresh DSH search Agent for
  a single bounded semantic rewrite, including the Host limits for query and
  focus-area lengths; a second invalid result still fails closed and the Host
  never truncates or invents model research content.
- Persist the Host `RunCancelled` boundary and close all sidecar admissions
  before waiting for DSH child-session disposal; a blocked or unavailable DSH
  runtime can no longer leave the scientific run visibly `running`, pending
  idempotent retries resume quiescence without appending a second terminal
  event, and the response still waits for the Host generation lease to drain.
- Resume a failed Critic from the durable Planner result without invoking the
  Planner or its registered vector prediction tool a second time.
- Verify replayed Planner identity, Genome/behavior/instance digests, and the
  persisted prediction-tool output digest before admitting a Critic retry.
- Restrict the prediction tool to `sample.plan`, and add direct event lookup so
  retry recovery does not scan the complete event ledger.
- Make the diagnostic non-promotion boundary explicit in generation analysis
  and terminal outcomes instead of relying indirectly on sparse-sample fitness
  failure.
- Publish only the safe analysis/reflection provenance digests needed to audit
  cross-generation context, and distinguish actual prediction budget from the
  169-origin / 1521-cell selection threshold in the plugin UI.
- Verify every file in the assembled delivery archive for credentials and make
  artifact verification independent of removed compatibility modules.

## 0.3.19 - 2026-08-23

### Changed

- Make `dsh-strict-origin-bundle@3` the sole DSH-native sample protocol and
  remove the obsolete per-cell protocol and all v1 role presets.
- Require every forecast origin to use a real Planner-owned vector-tool call,
  one independent Critic, Host scoring, and one post-score Reflector before
  its checkpoint can become durable.
- Restrict real DSH-native greenhouse runs to the 3-target × 3-horizon
  evaluator so one origin always carries all nine prediction cells.

### Fixed

- Consume the DSH Agent tool result even for a one-cell vector, preventing a
  second Host scalar execution in toy and reduced-grid tests.
- Add vector prediction tools to every reachable DSH-native evaluator path and
  verify durable `DshPredictionToolExecuted` events in end-to-end acceptance.
- Make delivery scripts set a UTF-8 locale in Chinese workspace paths.

## 0.3.18 - 2026-08-23

### Changed

- Pace DSH structured Agent stages and failed-stage cooldowns at 60 seconds by
  default so providers limited to roughly one request per minute do not turn
  valid Planner, Critic, Reflector, or Judge work into false sample failures.
- Retry one transient child-model failure with a fresh reservation and fresh
  DSH child while keeping schema and identity violations fail closed.

### Fixed

- Defer DSH-native feedback prediction until the Sample Planner selects the
  registered ridge tool, so one strict forecast origin now performs one real
  nine-output tool invocation instead of wrapping nine eagerly computed
  values as nine scalar tool calls.
- Propagate retryable DSH 5xx/transport failures across sample workers,
  evaluation, Judge, and generation finalization as resumable remote outages
  instead of recording them as scientific `tool_error` outcomes.
- Preserve immutable historical plugin archives by releasing these fixes as a
  new package instead of replacing `0.3.17`.

## 0.3.17 - 2026-08-23

### Changed

- Give research synthesis and generation reflection one bounded semantic repair
  attempt carrying the Host validation detail and rejected-output digest.
- Require every autonomous candidate direction to name exactly one parameter
  axis registered for the frozen parent Genome, and require the accepted
  mutation to implement that assigned axis.
- Select a predictor-matched seed Genome template for new DSH-native runs and
  derive restored-run mutation boundaries from the persisted parent Genome.

### Fixed

- Prevent valid horizon-targetwise mutations from being rejected against a
  stale exogenous-ridge task boundary.
- Preserve immutable historical plugin archives by releasing these fixes as a
  new package instead of replacing `0.3.16`.

## 0.3.16 - 2026-08-23

### Added

- Add the versioned `dsh-model-search-reflect@1` loop: model-authored search
  plans, Host-frozen OpenAlex/catalog evidence, evidence-bound multi-direction
  synthesis, one direction per candidate slot, and post-generation reflection.
- Aggregate post-score sample-reflection outcome, error-source, and next-action
  counts into the next generation's safe strategy context.

### Changed

- Execute all target × horizon predictions for one forecast origin through one
  strict Planner → registered vector tool → Critic → score → Reflector chain.
- Prioritize strategy-model queries ahead of deterministic Host search hints.
- Return bounded Host mutation-validation details to proposal retries.
- Version the autonomous-search and nine-cell forecast personas as immutable
  `v2` presets while retaining the original `v1` trees for historical replay.
- Select the bundled DSH plugin archive by the exact Python package version,
  so historical archives in a source checkout cannot make installation
  ambiguous.

## 0.3.15 - 2026-08-21

DSH candidate-proposer convergence patch.

### Fixed

- Define genome operations as a concise mutation delta and explicitly forbid
  reconstruction of unchanged parent configuration, preventing the proposer
  from repeatedly reasoning through a full copied genome.
- Keep DSH-native context management and schema-bound structured output as the
  sole agent execution path; no per-sample Token hard cap was introduced.

## 0.3.14 - 2026-08-21

DSH transient-outage recovery patch.

### Fixed

- Keep an autonomous run alive and schedule a visible delayed retry when the
  DSH transport, service, or bounded structured-stage call is temporarily
  unavailable, instead of terminally failing after three rapid replays.
- Give each research-stage retry a distinct deterministic idempotency key while
  preserving the durable generation and attempt identity.
- Distinguish retryable DSH service errors from schema, identity, route, and
  response-contract violations so permanent protocol faults still fail closed.
- Leave a 60-second client response margin beyond DSH's own 600-second
  structured-stage deadline, allowing DSH to own cancellation and return its
  bounded error contract rather than racing a local socket timeout.

### Verified

- Recover a real DSH-native research stage across a Python sidecar restart and
  accept the second durable child launch without Host/model-gateway fallback.
- Pass the complete Python regression suite plus focused native-runtime,
  structured-role, cancellation, and sample-agent tests.

## 0.3.13 - 2026-08-21

Workbench completion-language clarification.

### Changed

- Replace the misleading user-facing “budget exhausted” completion wording
  with “completed the configured evolution scale”, including completed
  generation and candidate counts when available.
- Explain that no candidate passed all evaluation gates while retaining the
  original machine-readable outcome and termination reason for audit and
  replay compatibility.

## 0.3.12 - 2026-08-21

DSH multi-generation workbench reliability patch.

### Fixed

- Require one-shot DSH roles to call `structured_output` in their first response,
  avoiding provider-side analysis narration that can exhaust a stage timeout.
- Raise the bounded DSH web projection proxy response ceiling from 2 MiB to
  16 MiB and reject declared oversize responses cleanly, so real
  multi-generation projections remain refreshable in the workbench.

### Verified

- Retain the 0.3.11 parent-genome continuity fix and validate it through a
  real three-generation DSH-native evolution run.

## 0.3.11 - 2026-08-21

Multi-generation DSH parent-genome continuity patch.

### Fixed

- Resolve a new generation's research parent from the previous generation's
  persisted `search_parent_candidate_id` before the new generation batch has
  been created.
- Preserve the batch-frozen parent genome as the authority after batch
  creation, so restart and replay semantics remain unchanged.
- Add a regression test covering the exact generation-advance gap found by the
  three-generation web run.

## 0.3.10 - 2026-08-21

DSH Host-bound structured-response patch.

### Fixed

- Send the concise first-response `structured_output` instruction in every
  Host-generated one-shot and Workflow child prompt, so the constraint reaches
  both existing and newly installed frozen role presets.
- Restore the `v1` preset definitions to their frozen content instead of
  silently changing an already published preset identity.
- Add a request-boundary regression test that inspects the actual child prompt
  delivered to DSH.

## 0.3.9 - 2026-08-21

Preset-only DSH structured-role response hint (superseded by 0.3.10).

### Changed

- Require researcher, candidate-proposer, sample-planner, sample-critic, and
  generation-judge presets to submit the requested schema in their first
  response without narrating long analysis.
- Add a preset contract test that keeps the concise first-response instruction
  present across every structured role.

## 0.3.8 - 2026-08-21

DSH proposal provenance visibility patch.

### Fixed

- Mark candidate mutations produced by the native DSH candidate-proposer as
  `dsh_native_agent` instead of projecting them as `legacy_unknown`.
- Count native DSH proposal calls and successes in run and round diagnostics.
- Show an explicit DSH-native proposal label in process and candidate views.

## 0.3.7 - 2026-08-21

DSH native child-session metrics replay patch.

### Fixed

- Accept the optional, ingress-validated `session_metrics` field while replaying
  `DshStructuredResultAccepted` events.
- Revalidate exact context-pressure and provider-reported token-usage semantics
  during replay, including Session identity and token-total consistency.
- Add a regression test for the real write-success/read-failure boundary found
  by the web evolution run.

## 0.3.6 - 2026-08-21

DSH rc.6 one-shot child identity compatibility patch.

### Fixed

- Read the published one-shot child Session id from `SubagentRun.id`, matching
  the rc.6 public contract. `childId` remains limited to continuable starts and
  Workflow lifecycle events.

## 0.3.5 - 2026-08-21

Bounded DSH structured-stage diagnostics.

### Added

- Added stable phase codes for child start/result, schema-bound output,
  admission, child Session identity and durable-result persistence failures.
  Provider error text remains behind the redaction boundary.

## 0.3.4 - 2026-08-21

DSH Session observability fail-soft patch.

### Fixed

- Made final TokenMeter and Session Projection snapshots independent and
  fail-soft so a cold/disposed child Session cannot reject an otherwise valid
  structured scientific result.

## 0.3.3 - 2026-08-21

DSH-native runtime observability and catalog correction.

### Fixed

- Read the real flat four-bucket DSH Token Usage projection so cumulative
  provider usage is no longer reported as unavailable.
- Retained the Python sidecar's already-redacted bounded rejection detail in
  DSH runtime diagnostics without exposing tokens or response headers.
- Corrected the catalog and workbench banner to report native DSH
  Agent/Session/Workflow execution instead of the legacy compatibility path.

## 0.3.2 - 2026-08-21

DSH-native closed-loop evolution hardening patch.

### Fixed

- Routed sample-planner waves through the real DSH Workflow Engine and
  persisted real child Session identities and DSH provider-usage projections.
- Added actionable cross-generation reflection and rejected exact replays of
  previously failed compiled behavior.
- Kept raw scientific fitness separate from execution-policy quality while
  preserving Host-owned physical-range repair and terminal authentication
  failures.
- Closed the cancellation race that allowed stale workers to append stage
  observations after a run became terminal.
- Aligned sample reason-code schemas with the Host enum and exposed DSH context
  pressure without introducing a Token hard cap.

## 0.3.1 - 2026-08-21

DSH-native runtime reliability patch.

### Fixed

- Bounded DSH critic inputs to aggregate evidence while preserving independent
  review of every selected sample.
- Removed private sample execution archives from generation-judge context and
  classified deterministic judge contract failures as permanent.
- Prevented unavailable judges from entering an unbounded retry loop or the
  candidate decision path.
- Normalized event payloads through canonical JSON before idempotency checks,
  preventing tuple/list representation drift after a successful append.
- Added a DSH child operational timeout without imposing any Token, context, or
  output-length cap.

## 0.3.0 - 2026-08-21

DSH-native plugin evolution runtime.

### Added

- Immutable plugin genomes, deterministic compilation, DSH role presets,
  structured Agent stages, durable child reservations, restart reconciliation,
  and pause/cancel admission barriers.
- DSH-native sample planning and review with Host-owned numerical tools while
  preserving the existing reward contract for identical predictions.
- Cellwise calibrated residual uncertainty artifacts and formal point/UQ gates.

### Changed

- Agent context, compaction, subagents, workflows, and output length are owned
  by DSH. The workbench no longer accepts a per-sample Token hard limit.
- Public projections distinguish search, validation, runtime, genome, and
  fitness identities without exposing private Agent messages.

## 0.2.2 - 2026-08-20

Reward, baseline, and promotion reliability hardening.

### Added

- A pure, versioned objective kernel with fixed target/horizon denominators,
  bounded skill and reward components, explicit coverage penalties, and strict
  rejection of duplicate or unknown task cells.
- Leakage-safe baseline fitting that selects persistence or a causal 24-hour
  seasonal comparator per target and horizon using only `training_fit`.
- Per-sample baseline provenance, normalized rewards, and compatibility-aware
  decoding for historical persistence-reward archives.
- A shared promotion policy with a practical score delta of `0.005` and a
  deterministic 1,000-resample paired 24-hour origin-block bootstrap that
  reconstructs coverage-penalized RMSE skill from bounded sufficient statistics.

### Changed

- Greenhouse objective aggregation is now `weighted_task_skill_reward@2` and
  reward is `absolute_error_improvement_vs_fit_selected_baseline@2`.
- The one-hour and multi-horizon evaluators apply the selected scoring baseline
  only after model prediction, preserving the original persistence input as
  `model_reference_baseline`.
- Direct, automatic, and manual promotion paths now use the same version-aware
  improvement assessment. Legacy evaluations retain their historical `1e-12`
  rule only under a matching common contract and are not compared directly
  with v2 scoring contracts.
- Evaluator configuration digests now freeze objective, reward, baseline,
  hard-gate, and promotion-confidence constants.

### Fixed

- Repaired a malformed duplicate objective metric expression that prevented the
  Python evaluator registry from importing.
- Repaired stale one-hour objective variable references that failed evaluation
  after the earlier score-field rename.
- Prevented failed sample executions from receiving positive reward against the
  stronger frozen scoring baseline.
- Made evaluation cohort identity independent of candidate execution failures,
  rejected tampered baseline profiles, and removed failed fallbacks and private
  bootstrap evidence from public prediction projections.
- Required complete SHA-256 scoring contracts for v2 comparisons, bound block
  weights and horizons to the frozen objective, and seeded confidence evidence
  from the cohort and both evaluations.

## 0.2.1 - 2026-08-20

Delivery hardening and observable autonomous evolution.

### Added

- Per-sample execution evidence for observed and predicted values, reward,
  agent/tool attempts, retry counts, failure categories, and conservative
  scoring fallback, exposed through bounded APIs and candidate evaluation UI.
- A versioned multi-horizon greenhouse evaluator and a registered ridge
  predictor with independent residual scales for every target at 1, 6, and
  24 hours.  The original evaluator contract and digest remain available for
  replaying historical runs, while new greenhouse runs use the expanded v2
  contract.
- A frozen research-to-execution chain covering literature evidence, research
  plan, algorithm blueprint and synthesis, restricted IR compilation, static
  debug, training smoke, real-sample execution, and cross-generation feedback.
- Feedback-driven planner, predictor, critic, repair-tool, and host-adjudication
  sample loops with bounded retries and microbatch remote calls. Individual
  sample failures are penalized without aborting an otherwise viable candidate.
- Continuous generation progression with bounded workers, durable recovery,
  per-run generation locks, and explicit archive, restore, and delete controls.
- A dedicated parameter-design workspace for generation count, candidates per
  generation, feedback samples per update, gateway microbatch size, sample
  concurrency, candidate and token budgets, seed policy, and knowledge retrieval.
- Release-bound real API acceptance reports with a per-file source manifest and
  verified SHA-256 identities for the wheel, sdist, and complete delivery archive;
  artifact/source equality is checked both before and after the live API run.

### Changed

- Remote model startup no longer depends on a separate browser API probe.
  Runtime calls use longer bounded timeouts, backoff, and retry classification.
- Research-plan calls explicitly reserve up to 8,192 output tokens, and the
  workbench uses one projection monitor for server-managed continuous runs
  with slower polling outside evaluation stages.
- Real API acceptance no longer auto-allows a non-loopback plaintext HTTP
  provider; release evidence defaults to configured HTTPS model routes.
- Real API acceptance now reads the selected evolution ledger in read-only mode
  and binds evidence to its latest matching GLM 5.2 and DeepSeek Flash
  `RunCreated` configuration instead of a stale hard-coded run identity.
- Candidate evaluation now separates live, paused, cancelled, partial, and
  completed sample states and preserves the last coherent snapshot on errors.
- New real autonomous runs use a deterministic, target/horizon-interleaved
  `training_feedback` window of 500 samples per update by default. Sibling
  candidates share the same window, while registered predictors such as ridge
  regression continue to fit on the full `training_fit` partition. Historical
  manifests without `samples_per_update` retain their full-feedback behavior.
- Promotion and cross-generation experience are cohort-aware: scores are
  compared strictly only for identical evaluation cohorts; after a window
  change, the current cohort's eligible champion advances without treating its
  raw score as an improvement over the prior window.
- Release verification now compares wheel, sdist, and complete delivery archive
  bytes with the current source tree, and cross-checks every user-visible and
  handshake version marker, preventing stale artifacts from passing.
- The DSH host bridge is explicitly private and unlicensed for redistribution,
  matching the proprietary root package.
- Isolated release builds pin their setuptools and wheel versions, and source
  distributions plus complete delivery archives include the workspace lockfile.

### Fixed

- Kept terminal run views on the latest persisted round and candidate evidence,
  including completed, failed, cancelled, paused, empty-round, and partially
  evaluated states. Candidate and process views now converge on the same latest
  sample revision without overriding an explicit operator selection.
- Made truncated or malformed proposal and independent-review responses enter
  durable cooldown recovery. Completed scientific evaluations are reused when
  only the final judge is temporarily unavailable, while authentication,
  permission, role, routing, and configuration errors remain terminal.
- Prevented candidate-budget exhaustion from completing a run before the final
  candidate evaluation is sealed, and saturated persistent retry backoff before
  exponentiation so very large retry histories cannot overflow.
- Added a bounded run-list summary projection for browser polling and removed
  the global mutation lock from long-running manual generation advances. This
  keeps pause, cancel, and unrelated runs responsive during remote calls while
  retaining the per-run generation lease.
- Ignored expected client disconnect exceptions at the HTTP server boundary so
  browser refreshes and bounded CLI reads do not emit misleading sidecar
  tracebacks; unexpected request exceptions still use the standard handler.
- Rendered a temporarily unavailable independent judge as an automatic retry
  or unavailable state instead of a rejection, and withheld the provisional
  `judge_accepted=false` metric until a completed judge response exists.
- Derived each evaluator's minimum feedback window from its complete
  target-by-horizon task count (1 for toy, 3 for single-horizon greenhouse,
  and 9 for multi-horizon greenhouse), and reject smaller runs in both the API
  and browser command layer. This prevents deterministic scientific-gate
  failures caused by a feedback window that can never cover every task.
- Kept research-stage timeouts, truncated responses, non-definitive gateway
  response failures, and exhausted semantic corrections alive as durable
  15-to-300-second cooldown retries.  Definite authentication and other 4xx
  failures remain terminal, and a retry releases the autonomous worker instead
  of replaying or failing the complete generation.
- Separated executable-blueprint evidence from literature-synthesis evidence:
  blueprints remain bound to registered predictors, while synthesis may cite
  any item in the frozen generation snapshot. Repeated semantic-contract
  failures now enter a durable cooldown and retry instead of terminating the
  complete evolution run.
- Kept long-running research stages visibly timed from their durable start
  event, restored the create action after asynchronous submission, and exposed
  pending, answered, applied, and audit-only expert-consultation states without
  requiring synchronous expert feedback.
- Unified public-data redaction across event, run, training-asset, trajectory,
  and strategy projections. Credential-like keys are matched after
  case/punctuation folding, and persisted exception diagnostics expose only a
  bounded exception type and error code rather than raw exception messages.
  Legacy model-health rows are scrubbed on open, and HTTP errors cannot expose
  credential-like text, private URLs, or local absolute paths.
- Removed API facade import-order dependencies so events, execution, catalog,
  projection, transport, and handler modules can each be imported in a fresh
  process without importing `server` first.
- Requeued failed terminal-event writes without replaying the completed
  generation, preventing a transient ledger failure from leaving a run marked
  `running` but absent from the autonomous worker queue.
- Limited complete sample-response JSON syntax recovery to one extra top-level
  request per wave; split descendants do not inherit the configurable transport
  retry budget, and progress records the real HTTP attempt count.
- Made the default adaptive splitter reduce malformed causal waves of eight or
  fewer decisions to singletons while retaining the existing bounded split
  floor for larger batches and explicit caller overrides.
- Prevented exhausted planner or critic transport retries from being replayed by
  the sample repair layer, while preserving decoded contract failures as
  eligible for adaptive splitting and repair.
- Mapped remote sample `reason_code` values to a finite host-owned vocabulary so
  a model cannot reflect Authorization data into metrics or compressed traces.
- Prevented stale or out-of-order projection and event responses from replacing
  newer UI state; fixed run-switch and sample-pagination rollback behavior.
- Accepted the observed GLM JSON-mode corruption only when an exact redundant
  object prefix can be removed into one complete object; all nearby malformed,
  multi-object, array-wrapped, truncated, and private-reasoning forms stay rejected.
- Made run selection transactional: an unreadable target no longer clears the
  current run's creation state, sample view, timing, or automatic progression.
- Cleared run-scoped failure banners after a successful context switch and
  kept the run selector readable when status filters and actions share the bar.
- Preserved the complete run outcome and formal-validation boundary on narrow
  screens instead of truncating it behind an ellipsis.
- Hardened the DSH loopback proxy against absolute-URL, path traversal, prefix
  escape, credential forwarding, cookie reflection, and upstream error leaks.
- Added state-recovery and command/idempotency checks for interrupted autonomous
  runs, while keeping historical event schemas replayable.
- Kept public metrics, the retained-score summary, and the green trajectory tied
  to the actual incumbent after every promotion, including a lower raw score on
  a different feedback cohort. The cross-run raw maximum is now observation-only,
  and experience summaries report the formal comparison score separately from
  `batch_highest_observed_score` so an unaccepted candidate cannot imply progress.
- Made hard Token-budget admission deterministic when the configured sample
  concurrency exceeds the number of calls that can be reserved. The coordinator
  now starts the earliest affordable schedules first instead of letting worker
  lock timing choose which samples consume the remaining budget.
- Restored isolated PEP 517 builds so release creation installs the declared
  build backend instead of depending on undeclared packages in the project venv.

### Boundaries

- This remains an unsigned, single-host research delivery candidate. Formal
  development/gate/holdout evaluation, production isolation, plugin signing,
  and physical greenhouse actuation remain outside the release.

## 0.2.0 - 2026-08-16

Deliverable ecology evolution workbench.

### Changed

- Reorganized the Python implementation into application, core, data,
  evolution, evaluator, integration, presentation, and API subpackages while
  retaining thin compatibility modules for the original public imports.
- Split the browser workbench into small dependency-free feature scripts under
  `assets/js`; `app.js` now contains only event binding and startup.

### Added

- Runnable AGC 2018 cucumber and AGC 2019 tomato historical datasets with
  hourly normalization, time-forward splits, embargoes, and safe sample APIs.
- Source-archive provenance auditing with declared size and MD5 verification,
  exposed separately from extracted-data readiness without local path leakage.
- `data audit` and `data fetch` CLI commands for HTTPS download, bounded
  size/MD5 verification, conflict-safe reuse, and guarded ZIP/7z extraction.
- Selectable rolling-residual and exogenous-ridge prediction models, plus a
  frozen compatibility contract between datasets, predictors, and evaluators.
- Leakage-bounded exogenous ridge fitting and 1/6/24-hour greenhouse evaluation
  across temperature, relative humidity, and CO2, including horizon summaries,
  persistence baselines, missing rows, physical-range violations, and previews.
- Bounded parameter sweep, local adaptive, and authenticated DSH proposal
  strategies with separate policy and judge model roles.
- Training artifacts, three-target greenhouse evaluation, baseline comparison,
  score trajectories, redacted event projections, and append-only human input.
- Five Chinese workspaces for setup, training data, evolution progress,
  candidate evaluation, and collaboration/governance.
- Strict incumbent retention: the first passing candidate may establish the
  incumbent; later candidates must improve its score by more than `1e-12`.
  Passing without improvement is rejected, and manual approval cannot bypass
  the invariant.
- One redacted `projection.training_assets` record per candidate, restricted to
  `iterative_positive`, `iterative_negative`, `quarantine`, or `pending`, plus
  a six-stage `projection.rounds` view from proposal through retention decision.
- Chinese training-asset and per-round stage views in the webview workbench,
  with sample browsing for both `training_fit` and `training_feedback` partitions.
- Explicit authenticated-model connection verification; remote policy and judge
  models must pass a minimal Bearer JSON probe before a run can start.
- Unified DSH model directory for the strategy and independent-review API
  selectors, with authenticated and pending-verification subsets, host-session
  ID intersection, and case-insensitive role aliases.  The DSH Web Profile host
  now reads its sanitized `llm.models` directory and passes provider/model
  aliases through the handshake; the server-side directory remains authoritative.
- Deterministic, bounded execution receipts for guidance, parameter overrides,
  numeric constraints, and parent selection; unparsed text is recorded-only.
- Complete redacted evolution episodes with five stages, lineage,
  reproducibility metadata, event receipts, and self-verifying digests.
- Live progress projection, dataset retry and frozen-run context, expandable
  event history, model connection status, and responsive mobile charts.
- Durable started/completed/failed events for proposal, candidate, training,
  evaluation, judge, and decision stages, including restart-safe event IDs.
- A real fit-to-evaluation callback boundary, so live stage polling and failure
  attribution switch from training to feedback scoring at the actual boundary.
- Chinese source-integrity, multi-horizon process, learned-model, and prediction
  detail views with predictor-specific human parameter guidance.
- Observable first-round startup, frozen remote-model re-verification after a
  service restart, and a wrapped training-data toolbar down to narrow viewports.
- Per-generation knowledge snapshots with a curated offline algorithm catalog,
  optional OpenAlex metadata retrieval, explicit host-capability mapping,
  non-executable research-only isolation, and non-causal round-end assessment.
- Chinese knowledge-source, execution-status, snapshot, and next-action views,
  with the same snapshot digest carried into candidate training-asset lineage.
- A local DSH Web Profile Cordis plugin that registers a Chinese sidebar entry,
  embeds the workbench in a DSH overlay, serves its static assets, and proxies
  the EcologyRSI API through the DSH origin.

### Boundaries

- Real greenhouse runs are historical replay and prediction only; they do not
  support counterfactual control or causal claims.
- Lettuce datasets remain catalog-only until their multimodal adapters exist.
- Development, gate, hidden, final, and external holdout samples remain outside
  browser access and the adaptive loop.
- Evolution training assets always declare `formal_training_ready=false` and
  require governance review; they are not formal SFT or DPO datasets.
- DSH integration includes a local Web Profile host plugin and a server-side
  OpenAI-compatible Bearer gateway, but remains unsigned. It is not official
  DSH OAuth, account authentication, plugin signing, or marketplace publication.

## 0.1.0 - 2026-08-16

Initial runnable MVP.

### Added

- Dependency-free Python evolution core with immutable run contracts.
- Append-only SQLite event ledger and deterministic replay projection.
- Narrow DSH adapter protocol with a deterministic local fake adapter.
- Structured parameter proposals; arbitrary candidate code is not executed.
- Toy crop-soil-water fixture with time-forward train, validation, and test splits.
- Local CLI and standard-library HTTP API for create, advance, control, and replay.
- CLI preflight (`doctor`), compact summaries, projection-only export/verify, and
  append-only import/replay of run bundles.
- Browser-native Ecology Evolution plugin shell and development manifest.
- Wheel, source distribution, and complete delivery-archive build checks.

### Boundaries

- The toy domain is an engineering fixture, not a validated scientific model.
- The plugin manifest is SDK-neutral and development-only.
- Hidden/final evaluation, production isolation, signed release governance,
  arbitrary-code evolution, and real DSH integration are not implemented.
