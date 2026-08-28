# EcologyRSI DSH 宿主插件

该插件把现有生态模型进化工作台接入 DeepSeek Harness Web Profile：

- 在 DSH 侧栏注册“生态模型进化”入口；
- 在 DSH 覆盖层中加载工作台，不离开 DSH；
- 由 DSH 在 `/plugins/ecology/evolution/` 托管静态资源；
- 由 DSH 将 `/api/ecology-evolution/*` 同源代理到本机 Python 服务；
- 打开工作台时通过 DSH `llm.models` 读取当前已登记且可用的模型目录，只把
  provider、模型 ID、显示名和职责元数据传给 iframe，不读取或转发密钥。

浏览器只需访问 DSH 端口。Python 服务仍在回环地址运行，但不再作为用户入口。

研究、候选提议、样本规划/批评和代际评审均由 DSH Agent Session、受限 preset
与直接、一次性的结构化子 Agent 执行。Python sidecar 只保留科学数值工具、不可变基因组编译
和追加式事件账本。上下文压缩与角色生命周期由 DSH 管理；不设跨调用的逐样本
Token 总预算，每次 sample 子模型通过 `agentOptions.maxTokens` 限制最多输出 2,048 tokens。

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
不可变 preset ID，并写入受管 `cordis.patch.yml` 区块。

新建严格运行先让 4 个候选共享 64-origin 初筛，再让 Top 2 各执行一个
500-origin adaptive epoch（默认 `10 × 50`，每批最多接受 2 处局部改动）；最后把
两个最终 revision 与 incumbent 放入同一 169-origin holdout。单轮合计 1,763
candidate-origins；默认 9 单元温室任务对应 15,867 个评分单元。候选并发默认 4；
逐样本并发默认 64、可配置 1–128；同一 provider 的 DSH stage 全局物理在飞上限为 128。

Node 宿主插件的 API 代理支持以下配置：

```yaml
config:
  staticRoot: /absolute/path/to/EcologyRSI-DSH/plugins/ecology_evolution
  backendOrigin: http://127.0.0.1:8777
  # 普通结构化阶段 10 分钟；长上下文调研阶段默认 30 分钟
  structuredStageTimeoutMs: 600000
  researchStageTimeoutMs: 1800000
  # 评分前 sample critic 独立上限 10 分钟
  sampleCriticStageTimeoutMs: 600000
  # 可选：也可以省略此项，直接使用 Node 进程环境变量
  serviceToken: replace-with-runtime-token
```

`researchStageTimeoutMs` 只用于搜索规划和证据综合等 researcher 阶段，
避免大上下文、慢推理模型被普通 10 分钟阶段上限误伤。
`sampleCriticStageTimeoutMs` 只用于评分前 `sample.critic`；默认 10 分钟，以覆盖高并发下正常的长响应，同时仍限制无效结构化输出后的异常长生成；
样本预测、候选提案、评分和反思使用 `structuredStageTimeoutMs`。

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
