# GLM 推理能力声明与隔离配置验证

核验日期：2026-10-20。本文对应 `scripts/configure_dsh_reasoning.mjs` 的配置生成修正；本次代码修复与离线回归未更改原服务 settings、模型路由或正在运行的实验 E。经另行授权的隔离实例 G 使用独立配置验证。

## 已确认的问题与修正

旧生成器把所有受支持的 GLM 都声明为不发送 `reasoning_effort`，因此对 GLM-5.2 选择 high 等档位也无法送达该参数；同时缺少 GLM-5.3/Flash 的能力声明。仅补能力列表仍不完整：Harness 0.2.0-rc.2 的默认 effort 属于 provider，不能在单个 model 条目中设置。实际 Pi AI 的 `zai` 分支在没有解析到 effort 时发出 `thinking.type=disabled`，这与 5.3/Flash 的限制冲突。

依据 [Z.AI 官方 Thinking 文档](https://docs.z.ai/guides/capabilities/thinking)：5.2 默认 max，接受若干别名；5.3 和 5.3-FLASH 的 API 仅接受 low/high/max，不能关闭 thinking。Coding Plan 的别名规则不同。生成器针对这些 API 规则生成如下声明；第三方网关是否执行相同语义需要单独验证。

| 模型 | Harness 可选 effort → 实际 effort 参数 | 未配置 provider 默认时 |
|---|---|---|
| GLM-5.2 | off → 无 effort 参数并发 `thinking.disabled`；minimal → minimal；low/medium/high → high；xhigh/max → max | max |
| GLM-5.3 / GLM-5.3-FLASH | low → low；high → high；max → max；不声明 off/minimal/medium/xhigh | max |
| 既有 GLM-5 / 5.1 / 4.7 | 保留原先 off/high 的 thinking 开关合同，不发送 effort 参数 | high |
| 既有 DeepSeek V4 Flash/Pro 及四位日期后缀 | 保留原先 off/low/high 声明 | high |

GLM-5.2 的 `off` 是 Harness 的选项名，并非发送 `reasoning_effort=off`。此实现使用 `thinking.disabled`，不是 `reasoning_effort=none`。测试其中一种 wire 格式，不能替另一种格式或另一模型的网关兼容性背书。

## 显式配置与作用域

生成器克隆输入，仅为选中模型补充能力和兼容字段。endpoint、认证引用、headers、未选中模型和其他配置字段保持不变；测试使用合成值验证这一点。兼容的既有能力子集和 provider 默认保留。显式的错误映射、不同 wire 格式、不兼容默认、选中模型的 `modelOverrides`、畸形 compat 对象均明确报错，由操作者协调，不能静默覆盖。

默认 effort 影响整个 provider。若原 provider 没有默认，只有所有显式模型均被选中且所需默认相同，生成器才补默认。存在未选中模型、未选中覆盖条目，或者不同代际需要不同默认时，会要求使用隔离 provider profile 或显式共同默认。多条 `--route` 一次性处理，与参数顺序无关。未知模型失败关闭，不能根据名字相似性猜能力。

可在维护阶段用原命令接口生成一个新的私有目录（下列仅示例，不能对正在运行的服务热切换）：

```bash
node scripts/configure_dsh_reasoning.mjs \
  --dsh-package /absolute/path/to/dsh/package.json \
  --source /absolute/path/to/isolated-input.yaml \
  --output /absolute/path/to/new-private-output \
  --route isolated/glm-5.2
```

脚本创建独立 `settings.yaml` 和 `settings.patch.yml`，两者权限均为 `0600`，使用排他写入防止覆盖。patch 直接配置 `id: llm-pi-ai`，由 Harness profile 的 Cordis loader 注册适配器；`settings.yaml` 保留给 Python 模型目录读取。输入应为待配置的隔离 profile；不要把生成器的默认注入当成对共享 provider 的无影响修改。脚本仅打印 patch 路径及用法，不打印认证引用或值，也不启动服务。

隔离 G 首次启动发现旧 patch `id: settings / config.path` 在 Harness 0.2.0-rc.2 不生效：该 settings 服务只迁移 `profile.home/settings.yaml`，没有这个 path 配置入口，导致 `NO_ADAPTER`、零外部调用。此问题已改为上述直接插件 overlay，并保留首次失败证据。只调用 `PiAiAdapter.apply` 的单元集成无法发现此类 profile 注册缺口，故增加下一节的完整启动回归。

## 离线验证与证据边界

普通回归：

```bash
node --test integrations/dsh_ecology_plugin/test/configure_reasoning.test.mjs \
  integrations/dsh_ecology_plugin/test/agent_lifecycle.test.mjs
```

实际适配器测试：

```bash
DSH_BIN=/absolute/path/to/dsh node \
  integrations/dsh_ecology_plugin/test/reasoning_payload_harness.mjs \
  /absolute/path/to/reasoning-wire-verification.json
```

后者固定验证已安装 Harness 0.2.0-rc.2，经其公开 `Config`/`apply` 构造真实 `PiAiAdapter`，向随机的 `127.0.0.1` 端口发送请求。假 HTTP 服务记录 JSON 并返回最小 SSE；认证值为本地占位字符串。脚本不读取用户 settings 或凭据，不调用远端模型。17 个实际请求覆盖 5.2 的全部声明档位、5.3/Flash 的默认与全部合法档位、显式 provider 默认保留；另验证 5.3/Flash 的 off 在 HTTP 前拒绝。它证实配置经过真实适配器后形成的请求字段，不能证明远端网关接受字段、降低成本或不损失质量。

完整配置注册回归另行运行：

```bash
DSH_BIN=/absolute/path/to/dsh node \
  integrations/dsh_ecology_plugin/test/reasoning_registration_harness.mjs \
  /absolute/path/to/reasoning-registration-verification.json
```

该测试在临时 `DSH_HOME` 用合成配置执行生成器 CLI，将其产出的 patch 原样传给真实 `runProfile`，通过 Cordis、`ctx.llm.resolveModelInfo` 与 `prepareCall/stream` 验证注册和默认解析，再请求本地假 HTTP。三个模型均成功注册，四次出站请求分别验证 5.2 默认 max/显式 off、5.3 默认 max、Flash 显式 low；另验证 Flash off 在 HTTP 前拒绝，以及私有文件权限和重复生成不覆盖。临时实例关闭并删除，无远端模型调用。该回归覆盖 CLI 到运行时注册边界，不能用直接适配器测试替代。

经授权的两次隔离真实网关探针另存于 `.runtime/optimization-20261020/glm52-reasoning-probe-live.json`。原配置默认组耗时 13.626 秒、716 tokens、九值工具输出正确且无可见正文，严格探针通过；生成器同款 off 组实际发送 `thinking.disabled`、无 effort 参数，耗时 3.801 秒、698 tokens、九值工具输出正确且没有推理输出，但附带 260 字节可见正文，故严格探针 `passed=false`。这说明该网关接受关闭字段，不代表整套生产合同通过。每组仅一次，且生成配置的兼容字段也变化（例如输出额度字段由 `max_completion_tokens` 变为 `max_tokens`），不能把差值归因于单一 effort 或外推吞吐/科学质量。仍须在隔离新实例通过标准四角色 native 预检，才能创建对应实验；不能据此直接切换既有运行。

## 对下一轮角色实验的约束

当前 `lib/runtime/agents.js` 的角色规则为：sample planner/critic 在模型明确支持 off 时关闭可选推理；generation judge 有 low 时使用 low；其他角色采用声明默认，否则回退 high。因此，只修正 5.3/Flash 的能力声明不会自动让 sample 角色使用 low，其默认仍可能是 max。任何新的按角色策略都是另一项实验变化，需要独立登记与验证，不应在 E 中途改动。

最小正确实施路径：

1. 在新的隔离 provider/profile 声明模型能力、wire 格式和默认。先验证 `resolveModelInfo`，再记录实际出站字段，最后对相同配置做真实工具 canary；目录声明不能取代网关实测。
2. 若采用静态按角色策略，将它纳入有版本的 native 源码，使 native content digest 覆盖；若做每次运行可配置策略，则增加规范化 effort policy，冻结到运行配置与 route digest，并传递到角色创建、恢复和请求 waterfall，不能添加摘要之外的 UI 开关。
3. 同一策略必须用于 production 与 canary。`model_canary.py` 的身份已包含 role/stage/preset/route digest；新增策略需改变相应身份，旧回执不能复用。角色子代理应继承或重新解析同一已冻结策略，并以真实 `request/header` 的 resolved config 核验。
4. 新运行验证 low/high/max 与当前 Agent 的质量—成本关系；先做 A/A 估计方差，再做配对样本、按格不退化和成本对照。单次九值工具探针只验证接口可用性，不能支持进化质量结论。

0.10.3 修复了既有原生摘要读取接口的遗漏：Harness 0.2 使用 `settings.describe({redactSecrets: true})`，而非旧 `settings.get()`。真实 Cordis 回归确认修改 reasoning 或 compat 会同时改变原生摘要、Python 路由摘要和四角色预检身份；仅轮换认证不改变身份。配置不存在或有歧义时拒绝继续，不能再把空配置作为有效身份。0.10.2 及此前的该读取路径不具备这一保护，因此升级后的预检必须重新执行。

修复后的冻结边界位于 `lib/runtime/capabilities.js` 的 `inferenceConfigDigest`、Python `integrations/dsh_directory.py` 的 provider execution digest，以及 API 的 create/resume route 校验。模型能力、compat、provider reasoning 均会进入相关摘要。配置改动后应创建新运行和新预检；旧运行若恢复校验因真实绑定变化拒绝，应保留暂停，不绕过摘要。

可复核的 Harness 实现为已安装 `@deepseek-ai/dsh-llm-pi-ai/lib/index.js` 的 `resolveModelReasoning`、`profileOptions`、`resolveReasoningLevel`、`reasoningInfo`、`streamWithSnapshot`，以及 `@earendil-works/pi-ai/dist/api/openai-completions.js` 的 zai thinking 分支。所有适配器验证固定在该版本；升级 Harness 后必须重跑。
