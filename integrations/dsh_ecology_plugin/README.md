# EcologyRSI DSH 宿主插件

当前插件版本 `0.8.4`，适配 DeepSeek Harness `0.2.0-rc.2`（2026-10-01 核对 npm `latest`）。

该插件把现有生态模型进化工作台接入 DeepSeek Harness Web Profile：

- 在 DSH 侧栏注册“生态模型进化”入口；
- 在 DSH 覆盖层中加载工作台，不离开 DSH；
- 由 DSH 在 `/plugins/ecology/evolution/` 托管静态资源；
- 由 DSH 将 `/api/ecology-evolution/*` 同源代理到本机 Python 服务；
- 打开工作台时通过 DSH Session Remote 的 `session/modelCatalog` 读取当前已登记
  且可用的模型目录（宿主未安装该 Remote 命名空间时回退到同源 `/api/session/modelCatalog`
  路由），只把 provider、模型 ID、显示名和职责元数据传给 iframe，不读取或转发密钥。

浏览器只需访问 DSH 端口。Python 服务仍在回环地址运行，但不再作为用户入口。
当前交付只启动两个进程：DSH Web Profile 监听 `8848`，EcologyRSI Python
sidecar 监听 `127.0.0.1:8777`；无需再启动独立前端服务或其他项目端口。

自进化创建的角色会话及其子会话仅保留在本地 DSH 日志中，不进入 Workspaces /
Ungrouped 会话列表。安装器同时挂载独立的 `session-visibility` 插件，按角色会话 ID、
生态 preset 和父子关系识别已有及新建记录，不根据工作目录或标题隐藏普通聊天。
识别结果缓存在 DSH session 根目录的 `.ecology-workspace-hidden.json` 中，后续列表
请求直接跳过已识别的日志目录及摘要构建。缓存缺失时首次请求会重建索引；日志不移动、
不删除，按 ID 读取、恢复、计费和自进化工作台继续使用原始记录。

研究、候选提议、样本规划/批评和代际评审均由 DSH Agent Session、受限 preset
与直接、一次性的结构化子 Agent 执行。Python sidecar 只保留科学数值工具、不可变基因组编译
和追加式事件账本。上下文压缩与角色生命周期由 DSH 管理；不设跨调用的逐样本
Token 总预算；sample Planner/Repair 通过 `agentOptions.maxTokens` 限制最多输出
4,096 tokens，Critic 最多输出 2,048 tokens。

Generation Judge preset 内部使用两个职责隔离的 Skill：`candidate-scientific-review`
只审查单个候选的冻结科学证据，`batch-scientific-reflection` 只读取 Host 生成的
rank→candidate→direction 聚合映射并提出建议；后者不能替代下一代 Host 预检，也不拥有选择或晋级权限。

当前六个活动角色 preset 都暴露同一个 `web_search`，安装包只交付这六个当前 ID。
Agent 在必需 Skill 之后、阶段终端工具之前按需提交查询，不指定 provider；工具默认使用
DSH `ctx.web.search`，技术失败或定量证据不足时由 Python sidecar 自动切到 OpenAlex
元数据检索。结果和路由进入追加式事件账本并可重放。插件不挂载 `dsh-tool-web`、
不开放 `web_fetch`，动态结果也不能替代冻结证据、登记预测工具或科学门禁。

安装已打包的运行时：

```bash
ecologyrsi-dsh install-dsh-runtime --profile web
```

安装器使用 `dsh plugin --profile web add --save-exact file:<tgz>`，安装六个当前
不可变 preset ID，并在受管 `cordis.patch.yml` 区块声明六个 `@deepseek-ai/dsh-agent-preset`。
新版 Harness 不再扫描 `.agent-presets`；声明从已安装插件内加载组合与 Skill，目录副本仅保留安装完整性检查。
升级前应结束当前实验；旧运行冻结的 preset ID 不会自动迁移为新身份。新运行使用 `@2` 种子模板，历史 `@1` 模板保持原始内容。

```bash
npm install -g @deepseek-ai/dsh@0.2.0-rc.2
ecologyrsi-dsh install-dsh-runtime --profile web
# 源码目录内：隔离启动真实 Harness，用本地模型替身检查六类角色、结构化输出和恢复
node scripts/verify_dsh_harness.mjs
```

验收脚本可用 `DSH_BIN` 指定待验证的 CLI，不使用用户 API key，也不调用远程模型。

新建严格运行先让 4 个候选共享 64-origin 初筛，再让 Top 2 各执行一个
500-origin adaptive epoch（默认 `10 × 50`，每批最多接受 2 处局部改动）；最后把
两个最终 revision 与 incumbent 放入同一 169-origin holdout。单轮合计 1,763
candidate-origins；默认 9 单元温室任务对应 15,867 个评分单元。候选并发默认 4；
逐样本并发默认 64、可配置 1–128。候选并发允许时，两条 finalist lane 会在同一
调度轮各推进一个 50-origin batch，但共同使用同一个 run 级 64 请求 admission；
同一 provider 的 DSH stage 全局物理在飞上限为 128，观测到拥塞或模型失败后会自适应降载。

Node 宿主插件的 API 代理支持以下配置：

```yaml
config:
  staticRoot: /absolute/path/to/EcologyRSI-DSH/plugins/ecology_evolution
  backendOrigin: http://127.0.0.1:8777
  # 普通结构化阶段 10 分钟；长上下文调研阶段默认 30 分钟
  structuredStageTimeoutMs: 600000
  researchStageTimeoutMs: 1800000
  # 预测规划含多次工具比较与长推理，单独给予 30 分钟
  samplePlannerStageTimeoutMs: 1800000
  # 评分前 sample critic 独立上限 10 分钟
  sampleCriticStageTimeoutMs: 600000
  # 可选：也可以省略此项，直接使用 Node 进程环境变量
  serviceToken: replace-with-runtime-token
```

`researchStageTimeoutMs` 只用于搜索规划和证据综合等 researcher 阶段，
避免大上下文、慢推理模型被普通 10 分钟阶段上限误伤。
`sampleCriticStageTimeoutMs` 只用于评分前 `sample.critic`；默认 10 分钟，以覆盖高并发下正常的长响应，同时仍限制无效结构化输出后的异常长生成；
`samplePlannerStageTimeoutMs` 只用于 `sample.plan`，默认 30 分钟：实跑中仍有生成活动的样本曾在 10 分钟时被中止，累计输出仅约 6900 Token。此时间包含准入排队、所有子调用、重试及持久化，不因重试而重置；达到截止时间仍中止并隔离晚到结果。输出额度、模型、预测工具次数和科学门槛保持不变。
候选提案、评分和反思使用 `structuredStageTimeoutMs`。

`serviceToken` 也可以省略，插件会读取 Node 进程的
`ECOLOGYRSI_SERVICE_TOKEN`。配置后，代理在服务端覆盖 iframe 请求中的
`Authorization`，因此令牌不会出现在 URL、静态 JavaScript 或浏览器存储中。
回环后端未设置服务令牌时保持免令牌兼容；非回环监听仍必须在 Python 服务端设置
同一个 `ECOLOGYRSI_SERVICE_TOKEN`。

策略模型和独立评审模型两个下拉框使用同一份宿主模型目录。浏览器再与 Python
服务端 `dsh_models` 目录按模型 ID、provider/model 或别名取交集；宿主目录不能
新增服务端未登记的可执行模型。服务端仍要求每个远程模型具备安全可执行路由、对应职责和服务端凭据；
工作台不再要求手工连接预验证，连通性与 JSON 响应契约在真实提案/评审请求中检查。若未提供显式
`ECOLOGYRSI_DSH_MODELS_JSON`，后端会从 `~/.dsh/settings.yaml` 与权限为 `0600` 的
`~/.dsh/.credentials.yaml` 自动读取同一份 DSH 目录；可用
`ECOLOGYRSI_DSH_DISCOVERY=0` 关闭，非回环 HTTP 需显式设置
精确 provider 白名单，例如
`ECOLOGYRSI_DSH_ALLOW_INSECURE_HTTP_PROVIDERS=newapi`。该名单使用逗号分隔、
区分大小写且不展开通配符。旧 `ECOLOGYRSI_DSH_ALLOW_INSECURE_HTTP=1` 仍兼容，
但会放行所有自动发现的非回环 HTTP provider，不推荐使用。

`ECOLOGYRSI_SERVICE_TOKEN` 是进程级服务令牌，通过后可访问全部 EcologyRSI API。DSH 上下文中的 capability 列表只用于前端隐藏或禁用操作，不是服务端的用户级 scope 校验；多用户部署需要在可信代理层增加 scoped token 签发与校验。

Python 目录中的 `id` 推荐使用 DSH 的 `provider/model` 形式，例如
`newapi/glm-5.2`；若保留自定义 ID，至少填写相同的 `model` 字段，前端会用宿主
目录公布的原始模型 ID 做别名匹配。策略和评审可以指向同一个 DSH 模型，但运行请求
仍需使用两个不同的目录 ID，以保留独立职责和评审边界。
