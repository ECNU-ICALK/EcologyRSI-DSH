# EcologyRSI DSH 宿主插件

该插件把现有生态模型进化工作台接入 DeepSeek Harness Web Profile：

- 在 DSH 侧栏注册“生态模型进化”入口；
- 在 DSH 覆盖层中加载工作台，不离开 DSH；
- 由 DSH 在 `/plugins/ecology/evolution/` 托管静态资源；
- 由 DSH 将 `/api/ecology-evolution/*` 同源代理到本机 Python 服务；
- 打开工作台时通过 DSH `llm.models` 读取当前已登记且可用的模型目录，只把
  provider、模型 ID、显示名和职责元数据传给 iframe，不读取或转发密钥。

浏览器只需访问 DSH 端口。Python 服务仍在回环地址运行，但不再作为用户入口。

0.3.0 起，研究、候选提议、样本规划/批评和代际评审均由 DSH Agent
Session、受限 preset、subagent 和 Workflow 执行；Python sidecar 只保留科学数值工具、
不可变基因组编译和追加式事件账本。上下文压缩、输出长度和多智能体生命周期
均交由 DSH 管理，不设逐样本 Token 硬上限。

安装已打包的运行时：

```bash
ecologyrsi-dsh install-dsh-runtime --profile web
```

安装器使用 `dsh plugin --profile web add --save-exact file:<tgz>`，安装六个固定
preset，并写入受管 `cordis.patch.yml` 区块。

Node 宿主插件的 API 代理支持以下配置：

```yaml
config:
  staticRoot: /absolute/path/to/EcologyRSI-DSH/plugins/ecology_evolution
  backendOrigin: http://127.0.0.1:8777
  # 可选：也可以省略此项，直接使用 Node 进程环境变量
  serviceToken: replace-with-runtime-token
```

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
