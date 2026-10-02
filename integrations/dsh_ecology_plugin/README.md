# EcologyRSI DSH 原生插件

**版本 0.9.0，固定适配 DeepSeek Harness 0.2.0-rc.2。** 本插件属于本地研究交付，尚未签名或发布到官方插件市场。完整安装、数据准备、实验协议与验收状态以[仓库 README](../../README.md)为准；此文说明宿主集成合同。

## 组件与请求链路

- 在 Harness 侧栏注册“生态模型进化”，以覆盖层加载中文工作台。
- 在 `/plugins/ecology/evolution/` 托管静态网页，将 `/api/ecology-evolution/*` 同源代理至 Python sidecar 的 `/api/*`。
- 通过 Session Remote 的 `session/modelCatalog` 读取脱敏模型目录；缺少该 Remote 命名空间时使用同源 `/api/session/modelCatalog`。浏览器目录不能增加服务端未登记的可执行模型。
- 由 Harness Agent Session 执行研究、提案、预测、条件评审与反思；Python Host 负责数值工具、能力编译、科学门禁和追加式事件账本。

默认只有两个服务：Harness `127.0.0.1:8848` 与 sidecar `127.0.0.1:8777`。浏览器访问 Harness，不需要单独启动前端开发服务。

安装器同时挂载 `session-visibility` 插件，按角色会话、preset 和父子关系隐藏进化会话，避免其混入普通 Workspaces 列表。识别缓存位于 Harness session 根目录的 `.ecology-workspace-hidden.json`；原始日志保留，按 ID 查询、恢复和用量核对不受影响。普通聊天不按标题或工作目录隐藏。

## 安装与模型发现

```bash
npm install --global @deepseek-ai/dsh@0.2.0-rc.2
export DSH_HOME="$PWD/.runtime/dsh-home"
ecologyrsi-dsh install-dsh-runtime --profile web

# Python Host 不会仅凭 DSH_HOME 切换模型配置路径，必须同时设置：
export ECOLOGYRSI_DSH_SETTINGS_FILE="$DSH_HOME/settings.yaml"
export ECOLOGYRSI_DSH_CREDENTIALS_FILE="$DSH_HOME/.credentials.yaml"
```

安装器调用 `dsh plugin --profile web add --save-exact file:<tgz>`，并在受管 `cordis.patch.yml` 区块声明六个 `@deepseek-ai/dsh-agent-preset`。新版 Harness 从已安装插件加载组合和 Skill；目录副本用于安装完整性检查。活动 preset 的唯一清单是 [`preset-manifest.json`](presets/preset-manifest.json)：

| 角色 | 不可变 preset |
|---|---|
| 协调 | `ecology-coordinator-v6` |
| 研究 | `ecology-researcher-v15` |
| 提案 | `ecology-candidate-proposer-v6` |
| 样本预测 | `ecology-sample-planner-v12` |
| 条件样本评审 | `ecology-sample-critic-v6` |
| 候选评审与批次反思 | `ecology-generation-judge-v10` |

未提供 `ECOLOGYRSI_DSH_MODELS_JSON` 时，Host 从指定的 settings 与 credentials 文件发现模型；默认路径分别为 `~/.dsh/settings.yaml` 和 `~/.dsh/.credentials.yaml`。凭据文件须限制为当前用户可读写（`0600`），密钥也可由 provider 的 `apiKeyEnv` 引用服务端环境变量。可用 `ECOLOGYRSI_DSH_DISCOVERY=0` 关闭自动发现。

显式 `ECOLOGYRSI_DSH_MODELS_JSON` 是服务端 JSON 模型目录，不应嵌入前端或提交 Git。模型 ID 推荐使用 `provider/model`，且须有对应职责、可执行路由和凭据。策略与评审连接 ID 必须不同；不同 ID 本身并不证明两个底层模型具有统计独立性。

生产连接应使用 HTTPS。确需在受信任环境连接非回环 HTTP provider 时，Host 支持精确、区分大小写的逗号分隔白名单 `ECOLOGYRSI_DSH_ALLOW_INSECURE_HTTP_PROVIDERS`，不展开通配符。更改路由或模型能力后先收尾旧运行，再重启服务并重新预检，不能在原冻结运行中热换配置。

## 认证与权限边界

| 环境变量 | 用途 |
|---|---|
| `ECOLOGYRSI_DSH_RUNTIME_URL` | Python 调用原生运行时的 Harness 地址，默认部署使用 `http://127.0.0.1:8848` |
| `ECOLOGYRSI_DSH_RUNTIME_TOKEN` | Python → Harness 原生阶段认证 |
| `ECOLOGYRSI_SIDECAR_TOOL_TOKEN` | Harness → Python 登记工具认证 |
| `ECOLOGYRSI_SERVICE_TOKEN` | 可选的进程级 HTTP API 认证，非回环 Host 监听必须配置 |

前两类调用令牌须让两个进程继承相同的对应值；生成和启动示例见根 README。API 服务令牌配置后，代理在服务端覆盖请求的 `Authorization`，不把令牌写入静态资源、URL 或浏览器存储。

进程级服务令牌可访问整个 EcologyRSI API，不是按用户划分的 scope。前端 capability 集合仅控制界面入口；多用户服务需要额外的可信身份与权限层。插件后端代理仅接受回环 origin。

## 阶段与预算

创建真实运行前执行工具和结构化结果预检，预检会产生调用；目录“可用”不等于真实模型已经通过。预检通过后才冻结运行身份。工具调用错误可以在有界范围内纠正，但失败结果、缺少必填字段或伪造输出不能成为成功回执。

| 阶段限制 | 默认值 |
|---|---:|
| 普通结构化阶段 `structuredStageTimeoutMs` | 600,000 ms |
| 研究 `researchStageTimeoutMs` | 1,800,000 ms |
| 样本规划 `samplePlannerStageTimeoutMs` | 1,800,000 ms |
| 条件样本评审 `sampleCriticStageTimeoutMs` | 600,000 ms |
| Planner / Critic 最大步骤 | 10 / 4 |
| Planner / Critic 单次输出额度 | 16,384 / 8,192 tokens |
| Planner / Critic 累计输出回执中止阈值 | 32,768 / 16,384 tokens |

阶段截止时间包含排队、子调用、重试及持久化，不因重试重置。累计回执阈值用于中止失控阶段，不是精确费用硬上限。每次 Planner 尝试最多调用两个登记预测工具，最终提交三目标 × 三时距的九个预测值。

候选并发默认 4，样本并发默认 64、可配置 1–128；实际准入受自适应并发、冷却及 provider 全局上限共同约束。同 provider 的原生阶段物理在飞上限为 128。HTTP 429/503 等有界恢复与科学性能拒绝分别记录。

0.9.0 默认使用快速模式，另有显式证据引导试点和正式对照模式。预测预算由创建时的冻结 schedule 计算，不能用某一模式的示例总量解释全部运行。各模式流程和容量见[实验模式与预算](../../README.md#实验模式与预算)。

## 研究与评审

六个活动角色均可使用 `web_search`，在必需 Skill 之后、阶段终端工具之前按需检索。默认使用 Harness `ctx.web.search`；技术失败或定量证据不足时，sidecar 可回退到 OpenAlex 元数据检索。结果与路由记入账本，外部内容只能作为研究线索，不能覆盖冻结规则或替代登记工具。

Judge preset 包含职责隔离的 `candidate-scientific-review` 与 `batch-scientific-reflection`：前者审查单候选冻结证据，后者消费 Host 生成的排名与候选映射。0.9.0 评审还绑定搜索或认证用途；模型建议不拥有选择、晋级或放宽门槛的权限。

## 升级与验收

升级前暂停相关运行，等待活动子任务收尾，备份账本与本地配置。安装新版本后用独立运行目录创建实验；不可变 preset 不能原地改写，旧运行不自动迁移到新身份。不要把缓存、凭据、账本或原始会话加入插件归档。

```bash
# 从仓库根目录执行；隔离 Harness + 本地模型替身，不调用远程模型。
DSH_BIN="$(command -v dsh)" node scripts/verify_dsh_harness.mjs
make test-integration PYTHON="$PWD/.venv/bin/python"
```

真实 Harness 的本地替身验收用于验证适配器、六类角色、结构化结果与恢复链路。它不等于远程模型四轮完成，也不构成预测性能提升的证据。
