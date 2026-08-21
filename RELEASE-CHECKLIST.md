# EcologyRSI-DSH 发布验收清单

适用于当前本地开发交付。所有勾选项都应有命令输出、截图或审核记录；未执行的项目不能记为通过。

## 1. 版本与交付内容

- [ ] `pyproject.toml`、包版本、插件清单、浏览器握手、HTML 页脚、插件 README、宿主插件、`NOTICE` 和变更日志版本一致。
- [ ] `README.md`、完整实施方案、`CHANGELOG.md`、`LICENSE`、`NOTICE` 和本清单存在。
- [ ] `integrations/dsh_ecology_plugin` 包含宿主路由、API 代理和 DSH 侧栏/覆盖层客户端入口。
- [ ] 交付明确标记为 alpha/未签名交付候选/专有许可，不宣称生产可用。
- [ ] 源码包不包含 API key、Bearer token、私有环境变量值、SQLite 运行库、原始数据或用户运行产物。
- [ ] 生菜 2021/2022 明确标记为 `catalog_only`，没有出现在可运行数据集列表。

## 2. Python 3.10 与源码验证

```bash
source activate py310
python --version
PYTHONPATH=src python -m unittest discover -s tests -v
find plugins/ecology_evolution -name '*.js' -exec node --check {} \;
node plugins/ecology_evolution/test/smoke.mjs
find integrations/dsh_ecology_plugin -name '*.js' -exec node --check {} \;
make verify
```

- [ ] Python 版本不低于 3.10，测试使用 `py310` 环境执行。
- [ ] 全部非真实数据单元与 HTTP 集成测试通过。
- [ ] JavaScript 语法检查和插件静态 smoke 测试通过。
- [ ] `make verify` 从源码完成 toy 演示、SQLite 重放、完整测试和插件检查。
- [ ] `doctor` 能检查 SQLite 完整性、事件重放、插件静态文件和可选 task manifest。

## 3. 真实 AGC 数据

```bash
export ECOLOGYRSI_DATA_ROOT=/absolute/path/to/greenhouse
PYTHONPATH=src python -m ecologyrsi_dsh data audit
PYTHONPATH=src python -m ecologyrsi_dsh data fetch agc_cucumber_2018 --archive-only
ECOLOGYRSI_TEST_REAL_DATA=1 PYTHONPATH=src \
  python -m unittest discover -s tests -v
```

- [ ] `/api/catalog` 中 `agc_cucumber_2018` 和 `agc_tomato_2019` 为 ready。
- [ ] 黄瓜加载 6 个 episode、每个约 2754 个小时行；Reference episode 为 `external_holdout`。
- [ ] 番茄加载 6 个 episode、约 3983 个小时行；Reference episode 为 `external_holdout`。
- [ ] 插件列出 5 个非 Reference episode，并按数据集目录确定性绑定一个；样本、运行清单、数据 digest 和产物血缘一致，Reference 不可进入运行。
- [ ] 真实数据集成测试没有因缺少 `ECOLOGYRSI_TEST_REAL_DATA=1` 被跳过。
- [ ] 数据来源、DOI、许可、必需文件模式、字段名称、单位和 digest 可从目录/描述接口核对。
- [ ] 黄瓜和番茄 `_archives` 来源归档分别通过声明大小和 MD5 校验；缺失或不匹配在页面告警且不与解压数据 `ready` 状态混淆。
- [ ] 缺失真实文件时数据集显示 missing，不会被启动表单当成 ready。
- [ ] `data fetch` 复用已校验归档，拒绝大小/MD5 不一致、路径穿越、链接、特殊文件及冲突覆盖；番茄主机具备 `bsdtar`。

## 4. 时间前向与科学边界

- [ ] 分区策略为 `time-forward-embargo/1`：约 30% 拟合、30% 反馈、20% 开发、20% gate。
- [ ] `training_fit` 与 `training_feedback` 之间有 1 小时 embargo；训练/开发和开发/gate 之间各有 24 小时 embargo。
- [ ] 温室训练产物只使用 `training_fit`，评测只使用后续 `training_feedback`。
- [ ] 浏览器样本 API 允许 `training_fit` 和 `training_feedback`，拒绝 `development/gate/external/hidden/test/final`。
- [ ] 评测覆盖气温、相对湿度和 CO2，包含只用 `training_fit` 选择的持续性/24 小时季节性强基线、MAE、RMSE、归一化 RMSE、技能得分、reward、缺失行和物理范围违规数。
- [ ] 外生变量岭回归的特征选择、填补和标准化只使用 `training_fit`，不跨分区前填，不把 outcome 结果变量作为特征，并按精确小时戳生成 1/6/24 小时目标。
- [ ] 多时距评测展示 3 个目标 × 3 个时距的 9 组结果、每时距汇总及预测起点/目标时间；滚动残差模型不能绑定多时距评测器。
- [ ] `passed` 要求固定科学门禁通过并且独立 judge 接受；judge 不能覆盖科学失败，页面不把 `passed` 等同于搜索保留或正式验证。
- [ ] 固定样本窗口的晋升规则通过测试：缺少可验证 evaluation/cohort digest 时禁止晋升；同轮候选 cohort digest 必须一致，否则 fail-closed；同轮只在相同 cohort 内稳定排名。
- [ ] incumbent 与本轮候选的 cohort、评测器、objective、数据/分区和基线 digest 相同时，新版要求 `candidate_score - incumbent_score > 0.005`；候选与 incumbent 必须具有完全一致的 24 小时预测起点块身份，块内保存 RMSE-skill 充分统计量且最多 128 块；配对区块数不少于 4 时，1,000 次 bootstrap 的 95% 置信区间下界还必须大于 0。旧评估回放仅在共同合同匹配时保留 `1e-12` 规则；cohort 改变时不得直接比较跨窗口原始分数。
- [ ] 外部或人工构造的 `approved` 决策不能绕过 `passed`、cohort 完整性和上述晋升校验，违规请求 fail-closed。
- [ ] 投影包含 `causal_interpretation=false`，页面和文档没有把历史回放表述为因果或反事实效果。

## 5. 策略、模型与 DSH 网关

- [ ] 目录展示 `parameter_sweep@1`、`adaptive_local@1`、`dsh_authenticated@1` 和 `autonomous_model@1` 四种策略；其中自主调研策略仍只能编译宿主已登记的有界能力。
- [ ] 目录展示合成、滚动残差、外生变量岭回归预测模型及兼容评测器；不兼容组合在创建运行前 fail-closed。
- [ ] 第二轮候选真实继承父参数；自适应策略消费上一轮指标，DSH 策略消费脱敏父代指标和 judge 建议。
- [ ] 本地两种策略只能使用 `host_parameter_generator@1`。
- [ ] DSH 策略只接受具有 `propose` 角色、安全可执行后端路由和服务端凭据的远程模型。
- [ ] 独立 judge 可选 `rule_judge@1`，或具有 `judge` 角色、安全可执行后端路由和服务端凭据的远程模型。
- [ ] 创建运行时拒绝相同的 `policy_model_id` 和 `judge_model_id`。
- [ ] OpenAI-compatible 网关请求 `/chat/completions`，使用 Bearer、`temperature=0` 和 JSON object 格式。
- [ ] 模型输出中的未知字段、未知参数、非有限值、错误类型和越界参数全部被拒绝。
- [ ] 远程网关只允许 HTTPS；HTTP 仅允许 localhost/回环地址，URL 不含凭据、查询参数或片段。
- [ ] `/api/catalog`、运行投影、事件和浏览器状态不含 token、`api_key_env` 或原始模型响应。
- [ ] 实际联调分别触发 policy 和 judge 调用，并使用不同模型标识和各自 Bearer 凭据。
- [ ] 真实 API 验收报告包含当前源码逐文件清单、脚本摘要和三类发布物摘要；报告保持在发布包外，并另行记录其 SHA-256。
- [ ] 自主温室首轮槽位符合约定：`K=1` 一次远程提案，`K=2` 一个宿主锚点加一次远程提案，`K>=3` 最多两个宿主种子；投影中的 `proposal_source`、远程调用数和 fallback 数与实际请求一致。
- [ ] 工作台和 HTTP 合同不再暴露独立模型连接验证功能；真实提案/评审请求校验 Bearer、连通性和 JSON 响应契约。
- [ ] 目录区分凭据/路由可执行性与最近一次真实请求健康状态；所有 HTTP 重定向被拒绝，目录不暴露网关地址。
- [ ] `/api/catalog` 的 `dsh_models` 是策略与评审两个下拉框的共同远程目录；`authenticated_models` 仅作为旧客户端诊断字段，不是创建或推进门槛；宿主模型 ID 交集不会新增服务端未登记模型。
- [ ] `/api/catalog` 同时核对 `configured_*`、`executable_*` 和 `roles_ready`；只有策略与评审两个职责均有可执行模型时才标记双角色就绪。
- [ ] 未设置 `ECOLOGYRSI_DSH_MODELS_JSON` 时，自动发现能读取 `~/.dsh/settings.yaml` 与权限为 `0600` 的 `~/.dsh/.credentials.yaml`；`ECOLOGYRSI_DSH_DISCOVERY=0` 可关闭，非回环 HTTP 默认阻止，`ECOLOGYRSI_DSH_ALLOW_INSECURE_HTTP_PROVIDERS` 只放行精确匹配的 provider；旧全局开关仅做兼容且不推荐。
- [ ] 验收记录区分“目录/凭据可执行”与“最近一次真实调用成功”；瞬时调用失败不改写用户当前模型选择。
- [ ] 交付说明明确：当前只有服务端 OpenAI-compatible Bearer 网关，不是 DSH 官方 OAuth、账号认证、正式插件认证或市场发布。
- [ ] 原生运行时已绑定时，目录明确返回 `harness_execution=dsh_native_agent_runtime` 且 `official_harness_agent_loop=true`，并验证角色 preset、Session、Workflow 和子智能体实际执行；兼容网关标记仅用于未绑定运行时的历史回放。

## 6. 运行、产物与人工干预

- [ ] 创建运行时冻结数据、分区、领域包、预测模型、策略、评测器、policy、judge、预算、seed、`candidates_per_generation`、`samples_per_update`、`sample_agent_batch_size`、`sample_concurrency`、在线知识设置及全部配置 digest；推进前检测实现或远程配置漂移。
- [ ] 工作台默认 5 轮、每轮 4 个候选、总预算 20、每轮 500 个反馈样本、同一 causal origin wave 内微批上限 64、样本并发 2、逐样本智能体 Token 预算 100,000,000，并提交 `auto_progress=true`；参数页不以样本数除以微批上限伪造请求次数，实际请求数以运行进度为准；服务端自动逐轮推进，浏览器只轮询，不出现要求点击“下一轮”的 `waiting` 状态。
- [ ] 用户调整轮数或每轮候选数时，未手工覆盖的总预算同步为两者乘积；总预算不足时创建被明确阻止，不能提前耗尽却仍声称完成配置轮数。
- [ ] 暂停在当前轮次边界生效，恢复后重新入队；多个连续运行按轮公平交替，重启只恢复未归档运行。
- [ ] 服务重启后，冻结远程 policy/judge 的角色、凭据、目录可用性、执行可用性和配置 digest 仍逐项检查；任一项不匹配时 fail closed。
- [ ] 自动 worker 与显式 `/advance` 在每轮开始前复用同一冻结数据集、策略、预测器、评测器和远程模型绑定校验。
- [ ] 每轮按提案、候选、训练产物、评测、独立评审、搜索保留决策、轮次推进的顺序写入或投影证据。
- [ ] 六个阶段均从真实 `EvolutionStageRecorded` 事件投影 started/completed/failed；中断恢复不重复阶段事件，决策异常也有 failed 记录。
- [ ] `projection.rounds` 逐轮展示提案、候选生成、训练、评测、独立评审、保留决策六阶段及其完成、等待或失败状态。
- [ ] `projection.training_assets` 与候选一一对应，每个候选恰好一条含五阶段、血缘、复现信息、事件收据链和自校验 digest 的脱敏派生资产，并可由同一 SQLite 事件流稳定重放和导出。
- [ ] 训练资产的 `admission.tier` 只出现 `iterative_positive`、`iterative_negative`、`quarantine`、`pending`；实际晋升者归入 `iterative_positive`，其余已完成但未晋升者归入 `iterative_negative`，并保留同窗未改善、跨窗冠军、低排名或 cohort 不可验证等具体原因。
- [ ] 所有训练资产均为 `formal_training_ready=false`、`requires_governance_review=true`，文档和页面明确其不是正式 SFT/DPO 数据。
- [ ] 投影展示候选得分、参数变更、理由、训练产物、指标、搜索保留结果，以及反馈窗口的 cohort digest、offset、cycle、wrap、全分区总量和选中量；跨窗口分数不得画成可直接比较的连续改善曲线。
- [ ] 顶层 `metrics`、当前保留得分和绿色轨迹只绑定实际 incumbent；`best_observed_score` 明确为跨窗口不可直接比较的原始观察值且不驱动当前指标。跨代经验以正式 champion/selected 分数比较，并将 `batch_highest_observed_score` 单独记录。
- [ ] 执行诊断分别展示每候选全量 `training_fit` 拟合行数、本轮 `training_feedback` cohort 的 eligible/available/selected/deferred/used/skipped 目标样本、累计候选工作量、拟合方式、拟合 pass 和轮次耗时；不把 bias/闭式岭回归的一次拟合误称为多 epoch 训练。
- [ ] 候选训练或远程模型失败被记录为 `CandidateFailed`，不会伪造成功分数。
- [ ] 创建、控制、推进和人工意见的 `idempotency_key` 重试不会重复写入。
- [ ] 运行中拒绝提交人工意见；暂停后允许提交。
- [ ] `guidance`、`parameter_override`、`constraint`、`parent_selection` 四类意见均可记录并产生明确执行收据。
- [ ] 可唯一解析的方向建议按固定步长应用；参数覆盖执行领域参数白名单和范围校验；数值约束在覆盖之后强制执行；指定父方案只能引用本运行已有候选。
- [ ] 恢复并推进后意见准确显示“已强制执行／已应用／仅记录（未执行）”，且只处理下一提案，不改写历史候选。
- [ ] 重启服务并使用原 SQLite 文件后，运行、产物、人工意见和当前最佳可以完整重放。
- [ ] SQLite schema v5 保留旧收据升级，补齐 `start_seq`、`resource_run_id` 和归档表；pending 创建/推进命令可按同一幂等键恢复并能随所属运行清理。

## 7. API 与权限

- [ ] `health`、`plugin`、`catalog`、`datasets`、`samples`、`runs`、`events`、`control`、`advance`、`interventions`、`archive`、`restore`、`delete` 主路径可用。
- [ ] `/api` 与声明支持的 `/api/v1` 行为一致。
- [ ] 运行 API 只返回 `browser-run/2` 脱敏投影，不返回完整 `state_snapshot`。
- [ ] 事件接口只返回安全摘要和游标，不返回 task manifest 或原始事件 payload。
- [ ] 回环地址可本地运行；非回环监听未设置 `ECOLOGYRSI_SERVICE_TOKEN` 时拒绝启动。
- [ ] 设置服务令牌后，无 Bearer 或错误 Bearer 的 API 请求返回 401。
- [ ] 发布说明明确当前服务令牌是进程级静态 Bearer，不宣称已实现 task/run/session scope。
- [ ] 跨机器部署位于 HTTPS 反向代理和网络访问控制之后，没有把内置 HTTP 服务直接暴露到公网。

## 8. 中文界面与响应式验收

- [ ] 六个工作区名称为“运行设置、参数设计、训练数据、进化过程、候选评测、人工协作与治理”。
- [ ] 除技术 ID、协议字段和单位外，业务标签、按钮、状态、错误和空状态均为中文。
- [ ] 运行设置以训练数据集为唯一数据边界输入；领域包、episode、预测模型、评测器和进化策略由数据集目录与模型研究结果自动绑定，策略模型 API 与独立评审模型 API 分离，并在不满足条件时禁止启动。
- [ ] 参数设计工作区可控制 `candidates_per_generation`、`samples_per_update`、`sample_agent_batch_size`、`sample_concurrency`、轮数、总候选预算、Token 上限、seed 和在线知识等冻结参数；默认值、范围、联动预算和创建后不可修改状态均清楚显示。
- [ ] 训练数据工作区提供“查看分区”选择器，可在 `training_fit` 与 `training_feedback` 间切换分页样本；开发、门禁、外部留出及其他受限原始数据仍不可选择且 API 拒绝访问。
- [ ] 训练数据工作区展示真实字段、单位、digest、来源归档校验、未就绪数据资产和进化训练资产。
- [ ] 训练资产表明确显示候选、轮次、准入标签、评测得分、judge 和“需治理审核”，页面同时声明其不是正式 SFT/DPO 数据。
- [ ] 推进期间进化曲线、候选、六阶段轮次和事件时间线实时同步；事件可展开全部记录；曲线明确区分同一 cohort 内可比较分数与跨 cohort 仅用于分轮观察的分数。
- [ ] 候选评测可选择候选并查看参数、理由、训练子模型、预测起点/目标时间/时距、逐样本真实值/预测值/reward、分目标多时距指标、cohort 证据和搜索保留结论。
- [ ] 人工协作页要求先暂停，再提交意见；历史展示覆盖参数、目标候选、实际应用提案、执行级别和数值前后值。
- [ ] 页面收到显式 capability 列表时，缺失能力的对应操作入口缺省拒绝，不因目录为空而回退为允许。
- [ ] DSH context 只接受同源父窗口或显式 `parent_origin`；API 只接受清单声明的 `/api`、`/api/v1`、`/api/ecology-evolution`、`/api/ecology-evolution/v1` 及同源/显式 `api_origin`。
- [ ] 宿主身份只用于显示和人工意见归属；宿主 capability 与服务 capability 的交集只约束页面入口；宿主模型只能匹配后端已登记模型。
- [ ] capability token 仅保存在内存，不进入 localStorage、导出摘要或公开插件状态。
- [ ] 进程级 `ECOLOGYRSI_SERVICE_TOKEN` 通过后可访问全部服务 API；发布说明不把前端 capability 交集宣称为服务端用户级 scope。
- [ ] 终态运行可归档并默认隐藏；可恢复查看；永久删除必须先归档、精确确认 `run_id`，并在单事务中只清理该运行的事件和命令回执。
- [ ] 1440px 桌面视口无重叠、截断或页面级横向滚动。
- [ ] 390px 手机视口无重叠、截断或页面级横向滚动；宽表仅在自身容器内滚动。
- [ ] 431–600px 窄屏下训练数据标题与分区/分页工具分行显示，按钮不被页面裁切。
- [ ] 浏览器控制台没有 error/warning，网络请求没有意外 4xx/5xx。
- [ ] 从 DSH `8848` 侧栏点击“生态模型进化”后外层 URL 保持在 DSH，同源静态页面和 `/api/ecology-evolution` 均由 DSH 返回。
- [ ] 只有显式 `demo=1` 才进入浏览器演示模式；后端失败不会自动伪造数据。

## 9. 发布物

```bash
source activate py310
make release
make verify-artifacts
```

- [ ] 构建机已安装 `uv`，wheel、sdist、完整交付压缩包和 `SHA256SUMS` 重新生成。
- [ ] wheel 在无旧项目依赖的干净临时环境中使用 `--no-deps` 安装成功。
- [ ] 安装后的 CLI demo 可运行，安装后的 `serve` 可返回健康状态并托管插件资源。
- [ ] sdist 包含源码、测试、示例、数据目录、插件、脚本和法律文件。
- [ ] 完整交付包包含 wheel、sdist、源码和内部校验和。
- [ ] `dist/SHA256SUMS` 中每个摘要重新计算并通过。
- [ ] 发布记录保存版本、数据 digest、split manifest digest、构建产物 digest、测试证据和审批人。

## 10. 必须保留的限制声明

- [ ] 真实数据结果只表述为历史回放和预测，不表述为因果、反事实或控制收益。
- [ ] 当前训练明确为滚动残差偏差拟合或外生变量岭回归残差拟合，不表述为通用大模型或任意神经网络训练。
- [ ] 当前为单数据集、单个自动绑定 episode、同轮小批次候选和固定反馈样本窗口，不宣称跨地点或跨季节验证；旧 manifest 未冻结 `samples_per_update` 时仍按历史全量反馈语义恢复。
- [ ] development/gate/external 已分区但未进入当前本地搜索保留闭环。
- [ ] 生菜数据仍仅目录登记。
- [ ] 插件仍是未签名 webview 交付候选，不宣称已经上架、完成市场发布或通过特定 DSH 产品认证。
- [ ] DSH 模型接入只表述为服务端 OpenAI-compatible Bearer 网关，不宣称已完成官方 OAuth 或账号认证。
- [ ] 当前同源嵌入和进程级 token 限制已披露；跨域 DSH 接入不宣称已完成。
- [ ] 系统不具备正式发布、回滚、隐藏数据读取或物理设备控制权限。
