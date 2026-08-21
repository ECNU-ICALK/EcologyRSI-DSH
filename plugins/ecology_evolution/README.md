# 生态模型进化工作台插件

这是一个可由数字科学枢纽（DSH）直接托管的静态 webview 插件，也可作为本地 sidecar 页面运行。前端使用原生 HTML/CSS/JavaScript，不依赖 React、npm 或原 `EcologyRSI/console`；数据、进化、训练评测和 SQLite 账本仍由独立 Python 后端负责。

## 启动

在仓库根目录执行：

```bash
source activate py310
PYTHONPATH=src python -m ecologyrsi_dsh serve \
  --host 127.0.0.1 --port 8765 --db /tmp/ecologyrsi-dsh.sqlite3
```

打开 <http://127.0.0.1:8765/plugins/ecology/evolution/>。页面默认连接同源 `/api`，也可使用 `/api/v1`、DSH 推荐代理前缀 `/api/ecology-evolution` 或 `/api/ecology-evolution/v1`。这些前缀都指向同一后端，不会启动额外前端端口。

没有后端时可显式启动浏览器演示：

```bash
python -m http.server 4173 --directory plugins/ecology_evolution
```

打开 <http://127.0.0.1:4173/?demo=1>。没有 `demo=1` 时，服务不可用会显示错误，不会静默伪造运行实例。

## 六个工作区

- **运行设置**：直接选择可运行训练数据集、策略模型 API 和独立评审模型 API。两个模型下拉框使用同一份 DSH `dsh_models` 登记目录。目录不依赖手工连接预验证；未配置凭据、后端路由被安全策略禁用或职责不匹配的条目会被禁用，真实连通性与 JSON 契约在提案/评审请求中检查。刷新目录时优先保留用户当前选择。真实 AGC 数据优先使用数据集目录中的指定 episode；兼容请求省略时才由服务端确定性回退。预测模型、进化策略和评测器由服务端根据数据集目录与模型研究结果自动绑定。
- **参数设计**：设置进化轮数、每轮候选方案数 K、每轮智能体反馈样本数、DSH Workflow 微批、并发数、候选总预算、随机种子与在线知识检索。上下文和输出长度由 DSH Session/模型路由管理，页面不再提交 `token_limit`。每个样本都经过 DSH planner、宿主登记数值工具和 DSH critic；reward 仍由 Host 按冻结基线计算。
- **训练数据**：绑定当前运行冻结的数据集与 episode，通过“查看分区”在 `training_fit` 与 `training_feedback` 间切换分页样本及中文字段定义，显示来源归档大小/MD5 校验和每候选一条的完整脱敏进化训练轨迹。展开轨迹后可按顺序查看输入上下文、智能体交互、宿主编译、训练预测、评测反馈、优化方向和最终结果；不返回开发、门禁、隐藏、最终或外部留出样本。
- **进化过程**：推进期间实时刷新“知识检索 → 逐代研究 → 能力编译 → 训练/评测 → 轮末决策”进度、真实样本微批位置、实际预测模型/评测器/时距摘要、候选散点和 incumbent 轨迹，并支持展开全部中文脱敏追加式事件。每个真实样本先经过远程 planner 和登记工具；高置信成功结果由宿主物理约束检查后直接采用，只有低置信或工具失败样本调用紧凑远程 critic，失败样本才进入 repair 波次。新真实运行默认冻结 64 条微批（128 仍是协议上限）；明确的输出截断/格式/决定契约错误会在宿主中有界二分到 8 条，认证、权限、路由、配置或已耗尽网关重试的传输错误不会被逐样本放大。进度事件只保留网关请求数、拆分次数/深度和恢复/失败数等聚合诊断。“实际执行口径”显示跨候选累计扫描的训练/反馈分区行次、实际使用和跳过的目标样本、候选产物/评测数、提案来源及 API 成功/回退情况；单次偏差拟合和闭式岭回归显示为 fit pass，并明确“无神经网络 epoch”。候选尚未封存时，页面优先使用实时进度；heartbeat 暂未写入时可从当前 revision 的已持久化样本批次回退计数，不与 heartbeat 重复累加。执行中的部分证据显示为“部分实时证据”（`partial_live`）；暂停保留、已中止保留和混合候选证据分别显示为“已保留部分证据”（`retained_partial`）、“已中止，保留部分证据”（`aborted_partial`）和“部分执行证据”（`mixed_partial`），后三类不使用“实时”描述。
- **候选评测**：比较候选参数、训练子模型、三目标 × 1/6/24 小时时距指标、持续性基线、预测起点/目标时间和训练反馈搜索保留结论。
- **人工协作与治理**：暂停后追加方向建议、参数覆盖、数值约束或父方案选择，并显示每条意见“已强制执行／已应用／仅记录未执行”的收据及权限与门禁边界。研究模型遇到实质性不确定内容时也可主动提出非阻塞专家咨询；专家无需实时在线，可在运行中稍后答复，答复只进入后续轮次。运行结束后仍可补录迟到答复，但只作审计归档，不改写已冻结的计划、候选或结果。

## 插件 API

```text
GET  {base}/health
GET  {base}/catalog
GET  {base}/datasets/{dataset_id}
GET  {base}/datasets/{dataset_id}/samples?partition=training_fit&offset=0&limit=20
GET  {base}/runs
GET  {base}/runs?include_archived=true
GET  {base}/runs/{run_id}
GET  {base}/runs/{run_id}/events?after={cursor}
GET  {base}/runs/{run_id}/samples?candidate_id={candidate_id}&offset=0&limit=50
POST {base}/runs
POST {base}/runs/{run_id}/control
POST {base}/runs/{run_id}/advance
POST {base}/runs/{run_id}/interventions
POST {base}/runs/{run_id}/expert-consultations/{consultation_id}/answer
POST {base}/runs/{run_id}/archive
POST {base}/runs/{run_id}/restore
DELETE {base}/runs/{run_id}
```

候选逐样本接口每页最多返回 200 条，并以完整 cohort 中冻结的 `sample_index` 稳定排序。状态区分尚未启动的 `pending`、执行中的 `running`、完整封口的 `completed`、保留部分结果的 `aborted` 和没有该合同的 `legacy` 运行。Reward 定义为 `|baseline - observed| - |predicted - observed|`，正值代表相对持续性基线更好；失败行会明确标记 `prediction_source=scoring_fallback`，其中的数值是固定评分惩罚，不是模型成功预测。

运行创建请求以 `dataset_id` 为必填首要输入；真实 AGC 运行可同时提交目录返回的 `episode_id`，以冻结训练团队／序列，兼容请求省略时才回退到首个非 Reference 优化 episode。插件不会把研究领域作为用户输入提交，预测模型、策略和评测器均由服务端根据数据集目录与模型研究结果自动绑定。请求还可提交 `strategy_model_id`、`review_model_id`、`autonomous_mode=true`、`rounds`、`budget`、`candidates_per_generation`、`samples_per_update`、`sample_agent_batch_size`、`sample_concurrency`、`token_budget`、`knowledge_online_enabled` 和 `seed_policy`。后端仍接受 `domain`、`domain_pack_id`、`research_domain` 作为旧客户端兼容字段，但它们只能用于一致性校验，不能覆盖数据集目录；如传入值与目录推导结果冲突，服务端拒绝创建并返回明确提示。服务端随后冻结 `dataset_id`、`episode_id`、推导出的 `domain_pack_id`、`strategy_id`、`prediction_model_id`、`evaluator_id`、模型配置 digest、批次预算和种子。工作台创建运行时默认提交 `auto_advance=true`，服务端将其冻结为 `auto_progress=true` 的连续策略并执行首轮；服务端后台默认使用 1 个有界 worker 按 FIFO 串行处理不同运行，以避免并发进化压满上游 API。同一运行始终只执行一个完整轮次并在轮末回到队尾，直到完成预设代数或候选数量、暂停、取消或失败。可用 `ECOLOGYRSI_AUTO_PROGRESS_WORKERS=1..8` 显式调整；设为 2 以上时，一个长超时或 `Retry-After` 不会占用全部 worker，但应按上游容量谨慎调大。浏览器只轮询阶段事件和投影，不再要求用户点击“执行下一轮”。已暂停运行恢复后会重新加入自动队列；每轮生成一个共享父方案和评测条件的小批次，完成全部候选训练反馈后统一分析并最多保留一个冠军。

已完成、已取消或失败的运行可归档，归档后默认从运行列表隐藏，但事件和训练证据仍完整保留；勾选“已归档”可恢复查看或撤销归档。永久删除只允许已归档终态运行，必须精确输入完整 `run_id`；删除会在单个 SQLite 事务内清理该运行的事件、归档标记和关联命令回执，不可恢复。

真实温室可选择 `greenhouse-rolling-residual@1` + 1 小时评测，或让 `greenhouse-exogenous-ridge@1`、`greenhouse-targetwise-ridge@1` 配合 `greenhouse_time_forward@1` 的 1 小时评测或 `greenhouse_multihorizon_time_forward@1` 的 1/6/24 小时评测。目标级岭回归为三项目标分别缩放残差修正，某一目标的缩放系数为 0 时仅该目标回退到持续性预测。岭回归只在 `training_fit` 学习特征处理和系数，在后续 `training_feedback` 与同一时距持续性基线比较；结果变量不会作为外生特征。每个可评分样本由宿主多角色工具运行时独立预测、批评、分类失败和选择修复；允许部分样本耗尽重试预算，但它们会保留在固定评分队列并按不改善原则惩罚。

逐代 research 未明确切换预测器时会继承当前已采用的预测器及其参数 schema，显式切换时不会继承旧 pipeline 的蓝图或 synthesis。在生产 `research_compile_evolve@1` 工作流中，每轮必须重新提交可编译的 `algorithm_blueprint` 和 `algorithm_synthesis`；宿主确认无兼容可执行证据时则必须显式提交 `algorithm_synthesis_degradation`，不能静默省略。蓝图的同代冻结证据引用至少包含一条状态为 `adopted` 或 `available_not_selected`、类型为 predictor 且映射到该 pipeline 的记录。仅提交预测器 ID，或只引用 `research_only` / `metadata_only` 资料，不能编译执行。

`algorithm_synthesis` 必须与 Blueprint 的 pipeline 一致，证据引用必须来自 Blueprint 已引用的同代冻结证据，`parameter_focus` 也只能使用登记参数。本轮有 OpenAlex `metadata_only` 摘要时，Blueprint 和 synthesis 都必须至少引用其中一条；否则至少引用一条 `research_only` 方向证据。宿主把 plan、Blueprint 和 synthesis digest 编译到仅含登记算子的受限 IR；候选依次通过 compile、静态 debug 和 `training_fit` 时间前向 training smoke 后，才进入真实样本的“远程 planner → 登记工具 → 宿主物理约束检查 → 低置信/工具失败时远程 critic → 仅失败样本 repair”契约。synthesis 与同代 compile/debug/评测/晋升结果会按 digest 关联记录，但不作因果归因。若共享系统代理对 OpenAlex 返回 429，且允许该固定 HTTPS 来源直连，可仅设置 `NO_PROXY=api.openalex.org`，不会改变模型 provider 的代理路径。

DSH-native 运行不设逐样本 Token 硬预算。页面只读显示 DSH TokenMeter 的当前上下文压力，累计用量只采信 Session projection 中的 provider usage，二者不互相推算。历史网关运行的硬预算账本仅用于只读回放。

每代 research 都会收到由 SQLite 追加式事件账本重放派生的跨代经验摘要。它最多扫描最近 24 个已分析代、最多展示最近 6 代，汇总修改、synthesis、算法/样本失败、弱目标/时距、修复成效和是否改善；未解决问题与已有后续评测证据支持的已解决问题分别进入 `active_unresolved` 和 `resolved_archived`，各最多 16 项。摘要不含原始样本或预测记录，UTF-8 JSON 硬限制为 16 KiB，超限时确定性裁剪并保留 omitted 计数；相同事件流在服务重启后可派生相同经验。

同一轮的候选共享完全相同的反馈样本窗口，并在该窗口内直接排名；本轮通过科学门禁和独立评审的冠军进入下一轮。跨轮窗口不同时，原始分数不具备严格可比性，因此不要求冠军高于上一窗口的 incumbent 分数；只有两个评测的样本摘要相同，才执行 `candidate_score > incumbent_score + 1e-12` 的严格改善规则。固定窗口运行缺少可验证的样本摘要时禁止晋升，人工或外部批准也不能绕过这些校验。

历史参数 hard guardrail 采用更严格的证据边界：同一目标/时距需要至少两个窗口、每窗不少于 20 个 cell 样本且总数不少于 40，并要求 skill 非负、零约束违规。不同 cohort digest 本身不等于样本独立；宿主要求底层 population digest 相同，并验证冻结环形窗口两两不重叠。字段缺失、population 不同、窗口重叠或同一参数出现多个各自达标的冲突值时均不保护。通过验证的值由宿主在候选参数生成的最后边界强制恢复，远端模型不能覆盖；未达到门槛的结果仍只作为方向性经验。

`projection.training_assets` 与候选一一对应，每个候选只有一条脱敏派生轨迹资产，而不是一条孤立的单步训练样本。轨迹会把输入上下文、智能体检索/提案、宿主编译、训练和预测记录、评测反馈、下一轮优化方向及父子候选关系串成一个有序过程；预测明细采用有限结构化摘要，避免把原始数据或隐藏推理带入浏览器。准入标签只可能是 `iterative_positive`、`iterative_negative`、`quarantine`、`pending`。所有标签都显示 `formal_training_ready=false` 且需要治理审核；这些记录只提供迭代证据，不是可直接用于正式 SFT/DPO 的训练数据。`projection.rounds` 逐轮展示自主调研、宿主能力编译、训练、评测、独立评审和优化决策证据。

人工主动意见必须在暂停态提交。恢复运行后，待处理意见只处理下一条变更提案；可解析建议按固定步长应用，参数覆盖与数值约束由宿主校验，无法唯一解析的文字只记录、不执行。模型主动发起的专家咨询始终为非阻塞：未答复时按咨询中登记的保守假设继续，答复在下一次尚未冻结的研究轮次作为 advisory 上下文使用，并且不能扩大数据、工具或权限边界。每条答复最多消费一次；迟到至终态的答复仅归档。固定数据划分、评测指标和搜索保留规则不会被浏览器、模型或专家答复改写。

## DSH 宿主接入

插件加载后向父窗口发送：

```json
{"type":"plugin.ready","plugin_id":"ecologyrsi.evolution","version":"0.3.15"}
```

宿主通过 `postMessage` 返回。最小兼容合同只要求同源代理地址和短期能力令牌；身份、能力范围和模型目录可选：

```json
{
  "type": "dsh.context",
  "api_base": "/api/ecology-evolution",
  "capability_token": "短期能力令牌",
  "identity": {"subject_id": "researcher-17", "display_name": "研究员甲"},
  "capabilities": [
    "evolution.catalog.read",
    "evolution.run.create",
    "evolution.run.advance",
    "evolution.projection.read",
    "training.data.read",
    "run.control",
    "run.archive",
    "run.delete",
    "intervention.write"
  ],
  "models": [
    {"model_id": "dsh-policy@1", "label": "DSH 策略模型 API", "roles": ["propose"]},
    {"model_id": "dsh-judge@1", "label": "DSH 独立评审模型 API", "roles": ["judge"]}
  ]
}
```

能力令牌只保存在 `host.js` 的页面内存闭包中，请求时作为 Bearer 发送；页面不显示、不导出、不写入本地存储。宿主能力与服务能力的交集只控制页面入口；当前服务端只校验进程级静态 Bearer，没有用户/task/run/session 级 scope，通过该令牌即可访问全部服务 API。宿主模型只会匹配并预选后端已经登记的同名模型，不能从浏览器新增后端未登记的可执行模型。

DSH 部署时由宿主托管 `plugins/ecology_evolution/` 的静态文件，并把同源 `/api/ecology-evolution/*` 代理到已部署的 EcologyRSI Python 服务。纯静态模式只能展示显式 `demo=1` 演示，不能执行真实进化。

策略模型和独立评审模型由 Python 服务端通过 `ECOLOGYRSI_DSH_MODELS_JSON` 配置，并经 OpenAI-compatible Bearer 网关调用。两个下拉框使用同一份 DSH 登记目录，但服务端仍依据 `propose` / `judge` 角色和独立模型标识执行安全校验。New API、GLM 等只要提供兼容 `POST /v1/chat/completions` 的接口即可使用；同一个底层模型若同时承担两个职责，应配置为两个不同的目录 ID（分别声明 `propose` 与 `judge`），以保留独立角色边界：

```json
[
  {"id": "newapi-glm52-strategy", "label": "New API GLM 5.2 · 策略", "gateway_url": "https://new-api.example/v1", "model": "glm-5.2", "api_key_env": "NEW_API_KEY", "roles": ["propose"]},
  {"id": "newapi-glm52-review", "label": "New API GLM 5.2 · 独立评审", "gateway_url": "https://new-api.example/v1", "model": "glm-5.2", "api_key_env": "NEW_API_KEY", "roles": ["judge"]}
]
```

工作台不执行单独的连接预验证。模型具备安全的后端路由、服务端凭据和对应角色即可创建或继续运行；连通性与 JSON 响应契约在真实提案和评审请求中检查。模型请求默认单次等待 900 秒、最多尝试 4 次；瞬时超时、限流和网关失败会使用指数退避、`Retry-After` 和有界失败重试，只记录业务调用诊断，不把已验证的 API 改成未验证。可用 `ECOLOGYRSI_DSH_MODEL_TIMEOUT` 覆盖单次等待时间。浏览器目录只收到脱敏模型标识、角色和连接状态，不会收到密钥或密钥环境变量名。

0.3.0 主执行路径已接入 DSH Agent Loop、Session、compaction、subagent 和 Workflow，Python sidecar 不再直接调用模型。当前交付仍是未签名的本地插件候选，不代表 DSH 市场签名或多租户 OAuth 认证已完成。

## 科学边界

AGC 2018 黄瓜和 AGC 2019 番茄数据仅用于历史回放与 1/6/24 小时时间前向预测，不代表在线控制、反事实实验或因果结论。来源归档完整性与解压数据运行就绪分开报告；生菜数据目前只登记来源，不能启动运行。隐藏评测、最终评测、发布、回滚和物理执行始终由外部治理服务控制。

## 检查

```bash
find . -name '*.js' -exec node --check {} \;
node test/smoke.mjs
```

插件崩溃或断线不会改变 SQLite 事件账本中的真实运行状态。
