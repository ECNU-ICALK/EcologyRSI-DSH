# EcologyRSI DSH 原生插件

**版本 0.10.11，固定适配 DeepSeek Harness 0.2.0-rc.2。** 本插件属于本地研究交付，尚未签名或发布到官方插件市场。完整安装、数据准备、实验协议与验收状态以[仓库 README](../../README.md)为准；此文说明宿主集成合同。

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
| 研究 | `ecology-researcher-v17` |
| 提案 | `ecology-candidate-proposer-v9` |
| 样本预测 | `ecology-sample-planner-v14` |
| 条件样本评审 | `ecology-sample-critic-v6` |
| 候选评审与批次反思 | `ecology-generation-judge-v10` |

未提供 `ECOLOGYRSI_DSH_MODELS_JSON` 时，Host 从指定的 settings 与 credentials 文件发现模型；默认路径分别为 `~/.dsh/settings.yaml` 和 `~/.dsh/.credentials.yaml`。凭据文件须限制为当前用户可读写（`0600`），密钥也可由 provider 的 `apiKeyEnv` 引用服务端环境变量。可用 `ECOLOGYRSI_DSH_DISCOVERY=0` 关闭自动发现。

显式 `ECOLOGYRSI_DSH_MODELS_JSON` 是服务端 JSON 模型目录，不应嵌入前端或提交 Git。模型 ID 推荐使用 `provider/model`，且须有对应职责、可执行路由和凭据。策略与评审可使用同一个连接 ID；每个角色仍须具备对应权限并通过独立的角色合同预检。不同角色分别绑定 preset、会话和回执，同模型不代表统计独立。

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
| 新运行 Planner / Critic 单次输出额度 | 32,768 / 8,192 tokens |
| 累计输出回执中止阈值 | 无隐式累计 token 配额；回执用于观察 |

阶段截止时间包含排队、子调用、重试及持久化，不因重试重置。执行由冻结步数、工具额度、单次输出上限和绝对期限约束；累计 token 回执不是费用硬上限。历史显式输出预算保持原值。新运行每次 Planner 尝试冻结 4 个预测调用额度，历史运行仍读取各自冻结值。组合程序按内部工具调用加一次计算扣除额度，失败也计费；最终提交三目标 × 三时距的九个预测值。

候选并发默认 4，样本并发默认 64、可配置 1–128；实际准入受自适应并发、冷却及 provider 全局上限共同约束。同 provider 的原生阶段物理在飞上限为 128。HTTP 429/503 等有界恢复与科学性能拒绝分别记录。

新建实验统一使用 `joint_plugin_epoch@2`，融合四候选初筛、单主线配对优化、轮末比较与一次条件复核。每轮基础 200 次预测时点执行，复核预留 100 次，总上限 300 次；唯一起点仍为 85 个。历史协议只按原合同恢复和回放。流程、门槛和容量见[实验模式与预算](../../README.md#实验模式与预算)。

## 样本提交前校验

样本 Planner 的 `structured_output` 在一次性捕获前调用认证的 Host 预校验。Host 复用最终写入的预测合同，检查冻结样本集合、候选允许的方法、会话已收到的成功工具证据，以及已冻结的均值公式。已知可纠正问题只返回有界的错误代码和字段路径，由模型在原有两次提交上限内自行修正。

0.10.8 在原提交次数与工具预算内支持一次有界的补证据修正：仅当样本 Planner 的首次提交收到认证 Host 的 `blend_requires_two_results` 反馈时，才允许在第二次提交前继续调用既有预测工具。运行时以会话内私有授权记录绑定首次拒绝和第二次有效提交，最终轨迹校验检查同一身份、允许的工具顺序及总工具次数；模型构造的错误文本不能授予修正权限。其他角色、普通模式校验错误和权限错误不获得此权限。

预校验不写账本、不读取评分标签、不调用数值工具、不扣工具额度，也不替模型修改预测或补引用。候选摘要由 Host 绑定；会话与激活租约由原生运行时的认证只读接口核实。最终首次写入再次验证，已有精确幂等回执仍可重放。权限、关闭、超时和基础设施错误保持原错误类别，不能充当模型参数纠错，也不能直接当作供应商拥塞。

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

0.10.9 仅修正 Python 工作台的实际执行统计：复用初筛的诊断记录不重复计为预测执行。此版原生运行逻辑、preset、输出 schema、提交次数、工具预算与 0.10.8 相同；版本声明与交付说明更新不代表新的模型性能结论。

0.10.10 补全 Python 工作台对条件复核和有完整来源的旧协议推理副本的实际执行统计；缺少必要证据时保持未知，不复制首份副本的成败结果。此版原生运行逻辑、preset、输出 schema、提交次数、工具预算均与 0.10.8/0.10.9 相同。包版本变化不要求原生执行摘要改变，也不代表新的模型性能结论。

0.10.11 支持同一模型承担策略与评审，保留各角色的权限、独立会话和四角色预检证据。`deepseek-v4.1-flash-1017` 可作为既有服务商下的模型 ID 登记；交付包不包含本机服务商凭据或私有配置。最终验收见[0.10.11 交付说明](../../docs/refactor/FINAL-DELIVERY-0.10.11.md)。
