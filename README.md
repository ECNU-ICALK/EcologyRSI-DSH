# EcologyRSI-DSH

**面向农业生态时序预测的、可审计的循环自进化工作台。**

EcologyRSI-DSH 基于 DeepSeek Harness，让模型根据研究资料和实验反馈提出改进，再由宿主在冻结的数据、预算与评测规则下验证。当前支持温室气温、相对湿度与 CO₂ 的 1／6／24 小时预测，改进范围包括模型参数、已登记预测器、提示词、Skill 程序、受控数值计算和执行策略。

> **交付版本：0.10.11 · 本地研究版／预览**
>
> Python 3.10+ · DeepSeek Harness **0.2.0-rc.2** · 建议 Node.js 24 LTS
>
> 策略与评审可选择同一已授权模型，四角色真实预检仍须分别通过。历史 U 已在冻结 0.10.8 上完成两轮；本版工程验收另行记录，运行完成不代表获得科学收益。

RSI 在这里指 Recursive Self-Improvement：依据已有证据循环提出、执行和检验改进。模型只能修改宿主登记的能力，不能改写评分规则、读取未授权分区或执行任意生成代码。**执行成功、搜索采用和独立认证是三个不同结论。**一次完整运行可以没有合格改进。

[快速启动](#快速启动) · [实验模式与预算](#实验模式与预算) · [数据与科学边界](#数据与科学边界) · [运行与排障](#运行与排障) · [开发与交付验收](#开发与交付验收) · [更新记录](CHANGELOG.md)

## 交付内容

| 组件 | 用途 |
|---|---|
| Python Host | 数据适配、时间分区、能力编译、预测工具、评分、版本选择与 SQLite 事件账本 |
| DSH 原生插件 | 六类 Agent 角色、工具权限、结构化提交、用量记录、网页托管与同源代理 |
| 中文工作台 | 开始运行、运行参数、数据、运行过程、候选方案、人工审核、独立评测七个工作区 |
| 验证与构建工具 | 确定性演示、重放、真实数据测试、隔离 Harness 验收、wheel／sdist／完整交付包 |

新建实验统一使用联合策略进化：提示词、参数、Skill、受控计算与经验记忆可组合改进，四候选与当前最佳版本同组筛选，再进行父子配对、轮末比较和条件复核。旧协议只用于既有运行的读取、恢复和回放，新运行冻结 `joint_plugin_epoch@2`。完整变更见 [CHANGELOG](CHANGELOG.md)。

## 快速启动

以下命令从仓库根目录执行。完整模型运行需要自行配置的模型服务凭据，并会产生模型调用费用；静态演示和确定性工程演示不需要 API key。

### 1. 获取并安装

```bash
git clone https://github.com/ECNU-ICALK/EcologyRSI-DSH.git
cd EcologyRSI-DSH

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
npm install --global @deepseek-ai/dsh@0.2.0-rc.2

# 专用 Harness 目录；后续安装、配置和启动始终使用同一 DSH_HOME。
export DSH_HOME="$PWD/.runtime/dsh-home"
ecologyrsi-dsh install-dsh-runtime --profile web
```

安装器校验固定 Harness 版本、插件内容和当前六个不可变 preset。请使用 Python 3.10 以上的解释器创建虚拟环境；Python 包无第三方运行依赖，发布构建另需 `uv`。番茄数据解压另需系统提供 `bsdtar`。

已有独立配置的 Harness 可以将 `DSH_HOME` 指向该目录。不要在其他实验尚未收尾时覆盖其插件或 preset。

### 2. 准备真实数据

```bash
# 可选：不设置时使用 ~/.ecologyrsi-dsh/data/greenhouse
export ECOLOGYRSI_DATA_ROOT="$HOME/.ecologyrsi-dsh/data/greenhouse"
ecologyrsi-dsh data audit
ecologyrsi-dsh data fetch agc_cucumber_2018
# 需要番茄任务时再下载：
# ecologyrsi-dsh data fetch agc_tomato_2019
ecologyrsi-dsh data audit
```

下载器校验来源归档的大小与 MD5，并检查解压路径；已核验的文件会复用。数据不随仓库或交付包分发。`--archive-only` 只下载并校验归档，尚不足以创建真实运行。

### 3. 配置模型并启动服务

在专用 Harness 中配置可用的策略与评审模型连接。密钥由 Harness 或服务端环境变量保管；不要写入 README、前端源码或 Git。页面只读取脱敏模型目录。

策略与评审是两个职责，可以使用同一目录模型 ID，也可以使用不同模型。同一模型承担两种职责时须具备两种角色权限；同模型不会合并角色合同、预检回执或评审职责，也不产生独立模型判断的统计保证。

例如，接收方可在已有 `pjlab` provider 的 `models` 列表中追加 `deepseek-v4.1-flash-1017`。下面仅示意追加位置，并非完整 `settings.yaml`；保留原 provider URL、API 类型、能力声明和密钥引用，也保留其他模型条目：

```yaml
llm-pi-ai:
  providers:
    pjlab:
      models:
        - id: deepseek-v4.1-flash-1017
```

目录解析为 `pjlab/deepseek-v4.1-flash-1017` 后，创建参数中的两个模型字段可以相同：

```json
{
  "strategy_model_id": "pjlab/deepseek-v4.1-flash-1017",
  "review_model_id": "pjlab/deepseek-v4.1-flash-1017"
}
```

这只是部署可配置路由示例，不表示网关已提供该模型或其能力已验证。本次交付尚未对该新路由执行真实模型预检或新进化；不能继承 GLM、旧 DeepSeek 路由或本地替身的通过结论。接收方须按实际提供方声明配置工具、输出与推理能力，再通过创建前的标准预检。

下面在同一个终端启动两个本地进程。首次启动后可在 Harness 设置中完成模型配置；配置完成后重启这两个进程，让 Host 重新发现目录。

```bash
source .venv/bin/activate
export DSH_HOME="$PWD/.runtime/dsh-home"
export ECOLOGYRSI_DSH_SETTINGS_FILE="$DSH_HOME/settings.yaml"
export ECOLOGYRSI_DSH_CREDENTIALS_FILE="$DSH_HOME/.credentials.yaml"
mkdir -p .runtime

export ECOLOGYRSI_DSH_RUNTIME_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export ECOLOGYRSI_SIDECAR_TOOL_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export ECOLOGYRSI_DSH_RUNTIME_URL="http://127.0.0.1:8848"

ecologyrsi-dsh serve --host 127.0.0.1 --port 8777 \
  --db "$PWD/.runtime/ecologyrsi-dsh.sqlite3" &
ECOLOGYRSI_SIDECAR_PID=$!
trap 'kill "$ECOLOGYRSI_SIDECAR_PID" 2>/dev/null || true' EXIT

dsh web --no-open --host 127.0.0.1 --port 8848
```

打开 **<http://127.0.0.1:8848/>**，点击侧栏的“生态模型进化”入口。浏览器通过同源代理访问 Python Host；`8777` 是内部 sidecar 端口。两个进程必须继承同一组运行时令牌，重启时也应一起更新。

上面的两个文件路径变量让 Python 服务读取专用 Harness 目录；只设置 `DSH_HOME` 不会改变 Python 默认读取的 `~/.dsh/` 路径。显式服务端模型目录、运行时参数和权限说明见 [DSH 插件 README](integrations/dsh_ecology_plugin/README.md)。自定义模型网关应使用 HTTPS，并正确声明模型的工具与推理能力；不能通过忽略版本或响应校验来制造就绪状态。

### 4. 创建第一次运行

1. 在“开始运行”选择就绪数据集、策略模型和评审模型。
2. 在“运行参数”核对联合策略流程、轮数、基础预算、复核预留及数据容量。
3. 点击“创建并开始”。系统先分别核验 researcher、generation-judge、sample-planner、sample-critic 四个角色的真实工具与结构化结果合同，通过后才冻结并创建运行。同一模型承担双角色时仍保留四份角色绑定证据；请勿重复提交。
4. 在“运行过程”查看预测落盘、局部修改、评审及轮末选择；需要干预时先暂停。

数据容量由实际 episode、历史上下文、时距和时间隔离共同决定。表单名义默认轮数为 5，但会根据容量下调，**不能保证每个数据集都能运行 5 轮**。此前 0.10.0 的真实数据记录中，黄瓜 `AiCU` 在当时参数及固定种子下支持旧联合策略最多 3 轮。统一协议每轮仍需 85 个不同评分起点，但新运行必须重新核验容量，不能通过缩短时间隔离绕过不足。

### 无 API 的工程演示

```bash
# 确定性演示：验证编排与账本，不代表真实模型性能。
ecologyrsi-dsh demo --db /tmp/ecologyrsi-demo.sqlite3
ecologyrsi-dsh doctor --db /tmp/ecologyrsi-demo.sqlite3

# 独立静态界面演示，不连接 Host、不写账本、不调用模型。
python -m http.server 4173 --directory plugins/ecology_evolution
```

静态演示地址：<http://127.0.0.1:4173/?demo=1>。真实服务失败时不会自动切换为演示数据。

## 实验模式与预算

新建实验仅提供 **联合策略进化 `joint_plugin_epoch@2`**。四种旧模式的有效机制整合为一条固定流程，页面不再选择模式。四个候选是真实比较的策略方案，不是四种实验模式。

| 阶段 | 冻结流程 | 每轮执行上限 |
|---|---|---:|
| 同组初筛 | 四候选＋轮初最佳版本，各预测相同的 10 个时点；复用结果诊断 | 50 |
| 局部配对 | 一条优化主线；父版本与一次局部修订，各预测 25 个新时点 | 50 |
| 轮末比较 | 候选与轮初最佳版本，各预测另外 50 个新时点 | 100 |
| 条件复核预留 | 满足更新基础条件但收益临界或跨日收益混合时，同一版本、同组数据双臂再推理一次 | 100 |
| 合计 | 基础 200 次＋至多一次复核 100 次，未触发的预留不执行 | **300** |

当前温室任务的一个预测时点（origin）包含三个目标 × 三个时距，即 **9 个评分单元**；上限为每轮 2,700 个单元。其他数据集按其冻结目标和时距计算。执行次数不是唯一样本数，也不是 LLM 请求数。每轮需要 **85 个不同评分起点**，另计历史上下文和目标成熟隔离；重复推理不增加独立观测数。研究、提案、反思、工具和恢复另外计量。

**策略与效率。** 每个候选最多四处关联修改，局部修订最多两处；可联合修改提示词、登记参数、特征、Skill、受控计算和经验记忆。四候选同组初筛后仅深入一条主线，初筛证据不重复执行；连续改进通过跨轮继承实现。每次配对期间版本固定，结果反馈仅用于之后的修改。数值程序最多调用三个登记能力并做有界组合计算，内部调用与组合均计入原工具额度；不执行任意 Python、不安装依赖、不修改 Harness。

**更新与复核规则。** 完整执行、总体实用增益、分项不退化及跨日证据共同约束更新。至少需要三个可核验日块，各块预测覆盖达到冻结门槛，多数日块的收益为正，最差日块退化不超过冻结容差。通过这些基础条件后，若总体增益不超过实用增益阈值的两倍，或存在非正但仍在容差内的日块增益，触发一次双臂复测。首次比较和复测都必须通过，禁止挑选较好的一次；首次已失败或时间覆盖不足时直接保留原版本，不通过重复推理补足时间数据。规则由版本化协议冻结，`execution_plan` 单列基础、预留和总预算。

**历史兼容与独立验证。** `quick_adaptive_epoch@1`、`evidence_guided_epoch@1`、`top2_adaptive_epoch@1` 和 `joint_plugin_epoch@1` 继续按原合同读取、恢复和回放，不改写旧评分与预算；新建接口拒绝旧协议及旧 schedule。搜索晋级不授予独立认证；验证与最终测试使用隔离分区、冻结版本及单独预算，不回流训练。容量不足拒绝创建，不通过复用留出或缩短隔离补足。实际可运行轮数以统一容量预检为准。

## 如何解读结果

| 页面证据 | 可以说明什么 | 不能据此认定什么 |
|---|---|---|
| 执行成功、预测已落盘 | 当前要求的预测向量与角色步骤完整 | 预测性能提升 |
| 相对基线技能分为正 | 当前 cohort 的综合评分高于冻结强基线 | 所有目标／时距均改善 |
| 修改已应用 | 新 revision 可用于后续实验 | 修改已验证有效 |
| 同组父版本／冠军增益 | 同一比较范围的新旧差异 | 跨窗口可直接累计的增益 |
| 搜索采用 | 通过该运行冻结的搜索规则 | 通过独立科学认证 |
| 效果回执为不确定 | 暂无足够证据确认行为契约效果 | 成功或确定失败 |

不同批次的原始分数不能直接连成“修改收益曲线”。本版分别展示基线分、父子增益和冠军增益，并区分执行故障、无效修改、证据不足和性能退化。模型评审只提供绑定用途的建议，不能覆盖宿主门禁。

修改效果契约检查实际生效值、受影响单元和执行证据。部分提示词／Skill 仍属于软约束；置信度加权等未给出明确权重的策略不能宣称已完成公式验证。当前并不保证每次局部修改都能改善预测。

## 数据与科学边界

| 数据集 | 当前能力 | 来源许可 |
|---|---|---|
| `agc_cucumber_2018` | 2018 黄瓜，温室环境历史回放与多时距预测 | CC0-1.0 |
| `agc_tomato_2019` | 2019 番茄，温室环境历史回放与多时距预测 | CC0-1.0 |
| 两个生菜 RGB-D 条目 | 仅目录登记，尚不能创建可执行运行 | CC-BY-4.0 |
| `generated-toy-series@1` | 确定性合成数据，工程回归使用 | 项目测试数据 |

数据来源、文件合同与完整性清单位于 [`datasets/`](datasets/)。页面将文件就绪和来源归档校验分开显示；有目录条目不等于本机已准备数据。

新运行冻结 `time-forward-four-stage@2`，依次划分：

```text
calibration_fit → 时间隔离 → calibration_uq → 时间隔离
     → model_selection → 时间隔离 → validation → 时间隔离 → final_test
```

- `training_fit` 是 `calibration_fit` 的公开别名，仅用于拟合、预处理与强基线选择。
- `training_feedback` 是 `model_selection` 的公开别名，用于搜索训练及轮末比较。
- 不确定性校准、独立验证和最终测试具有各自边界；浏览器不能读取隐藏分区原始行用于调参。
- 每个“目标 × 时距”仅在拟合段比较持续性与 24 小时季节性基线，选择后冻结。
- 评分检查覆盖率、物理范围、相对基线表现及分项退化；严格比较还检查配对证据和推理副本。
- 稀疏产量、能耗及资源记录不计入当前三目标环境预测评分；尚未实现跨 episode 联合认证。

这是离线历史预测研究，不能将预测差异解释成控制动作的因果收益。系统不控制真实温室设备，不授予正式发布权限，也不自动把搜索轨迹认证为训练数据资产。

## 运行与排障

| 现象 | 检查与处理 |
|---|---|
| 没有工作台入口 | 检查 Harness 是否为固定版本、插件是否安装在正在使用的 `DSH_HOME` 和 `web` profile；安装后重启宿主 |
| 数据集不可用 | 执行 `data audit`，检查数据根目录、必需文件和解压依赖 |
| 模型未就绪／预检失败 | 核对两个模型的目录 ID、职责、HTTPS 路由和服务端凭据；查看脱敏错误码 |
| 一段时间没有新事件 | 查看真实子任务、最近活动、配额与输出预算；没有事件不等于模型已经失败 |
| HTTP 429／503 | 等待现有的有界退避、冷却与并发调节，不反复创建运行 |
| 输出额度耗尽／不可恢复协议错误 | 核对失败阶段及实际回执；修正配置或实现后创建新运行，不能伪造结构化结果 |
| 总分为正但未通过 | 核对目标／时距退化、覆盖率、约束及比较证据；不是通过放宽门槛解决的问题 |
| 新协议容量不足 | 减少预登记轮数或换用足够数据；保持样本量与隔离要求 |

暂停后等待在途任务收尾，再做受限人工干预。恢复沿用同一冻结合同和已验收检查点；取消后的记录保留用于复核。归档隐藏终态记录，不等于删除账本。

升级步骤：**暂停 → 等待子任务收尾 → 备份账本和本地配置 → 安装新版本 → 使用独立运行目录创建新实验**。不要在活跃运行中更换模型路由、数据、源码身份或 preset。旧账本需要其对应的固定版本解释，本版不自动迁移历史合同。

```bash
ecologyrsi-dsh doctor --db .runtime/ecologyrsi-dsh.sqlite3
ecologyrsi-dsh summary RUN_ID --db .runtime/ecologyrsi-dsh.sqlite3
ecologyrsi-dsh export-policy RUN_ID CANDIDATE_ID \
  --db .runtime/ecologyrsi-dsh.sqlite3 --output .runtime/agent-policy.json
```

策略包包含冻结的 Agent 策略、模型配置与证据身份，不包含凭据。加载入口为 `ecologyrsi_dsh.integrations.agent_policy_bundle`；部署使用仍须匹配冻结合同与授权边界。

## 架构与接口

```text
浏览器工作台（Harness :8848）
  ├─ DSH 原生 Agent：研究、提案、预测、条件评审、反思
  └─ 同源代理 /api/ecology-evolution
       ↓
Python Host（回环 :8777）
  ├─ 应用编排与唯一运行状态机
  ├─ 数据合同、能力注册表、数值工具与科学评测
  └─ SQLite 追加式事件 → 只读投影、检查点、导出
```

| 目录 | 职责 |
|---|---|
| `src/ecologyrsi_dsh/application/` | CLI、依赖装配、轮次与批次编排 |
| `src/ecologyrsi_dsh/core/` | 事件、回放、运行状态、评审与版本选择 |
| `src/ecologyrsi_dsh/evolution/` | 候选基因组、变更轴、效果契约、协议和预算 |
| `src/ecologyrsi_dsh/evaluators/` | 预测执行、cohort、分项评分与配对比较 |
| `src/ecologyrsi_dsh/data/`、`knowledge/` | 数据适配、分区与冻结研究资料 |
| `src/ecologyrsi_dsh/api/`、`integrations/` | HTTP、脱敏投影与 DSH 连接 |
| `integrations/dsh_ecology_plugin/` | 原生插件、角色清单、Skill、Schema、安装包 |
| `plugins/ecology_evolution/` | 中文网页及按工作区加载的前端模块 |
| `tests/`、`scripts/` | 回归、交付构建、验收与诊断 |

网页 API 前缀为 `/api/ecology-evolution`，由 DSH 代理到 sidecar 的 `/api`。常用相对路径：

| 方法与路径 | 用途 |
|---|---|
| `GET /health`、`GET /catalog` | 服务版本、就绪能力与脱敏目录 |
| `GET /datasets/{id}`、`GET /datasets/{id}/samples` | 数据说明和授权训练分区样本 |
| `POST /evolution-capacity` | 创建前容量与预算检查，不调用模型 |
| `POST /model-preflight` | 真实模型工具与结构化输出预检，有模型用量 |
| `POST /runs` | 幂等创建与可选自动推进 |
| `GET /runs/{id}?view=process` | 过程投影；另有 overview、candidates、candidate 视图 |
| `GET /runs/{id}/events?after={seq}` | 追加式事件增量读取 |
| `POST /runs/{id}/control` | 暂停、恢复与取消等运行控制 |
| `POST /runs/{id}/interventions` | 追加受限人工意见 |
| `POST /runs/{id}/archive`、`POST /runs/{id}/restore` | 归档与恢复终态记录 |

API 客户端应复用网页的预检与冻结创建合同，并使用新的 `idempotency_key` 创建新实验；重试同一创建请求保持原 key。原生运行不接受任意生成代码或通过客户端改写科学门槛。更多宿主协议见 [前端 README](plugins/ecology_evolution/README.md) 与 [原生插件 README](integrations/dsh_ecology_plugin/README.md)。

## 开发与交付验收

```bash
make test-fast PYTHON="$PWD/.venv/bin/python"
make verify PYTHON="$PWD/.venv/bin/python"

# 准备真实 AGC 数据后：
ECOLOGYRSI_TEST_REAL_DATA=1 make test PYTHON="$PWD/.venv/bin/python"

# 使用已安装的固定 Harness，在隔离目录和本地模型替身上验证：
DSH_BIN="$(command -v dsh)" node scripts/verify_dsh_harness.mjs
```

`make verify` 包括确定性演示、账本重放、完整 Python／Node 检查、前端 smoke、版本和打包资源一致性；真实数据及 Harness 检查是否执行取决于环境，须查看实际跳过项。真实 Harness 本地替身测试不使用用户 API key，也不等于真实远程模型验收。

**0.10.11 本版验收：**同模型双角色仍受角色权限、四角色预检与冻结身份约束。本版完整门禁、隔离快照提交、发布产物及摘要统一记录于[本版最终交付索引](docs/refactor/FINAL-DELIVERY-0.10.11.md)；对应回执完成前保持待验收，不将下面的历史测试数量或 U 运行改记为本版新验证。`deepseek-v4.1-flash-1017` 的真实预检与进化尚未执行。

**历史 0.10.10 工程验收：**2026-10-20 的冻结源码通过完整真实数据 Python 2,009 项、Node 338 项，零失败、零跳过；前端、lint、锁文件检查及源码与原生归档一致性通过。506 个受检源码文件在门禁期间未变。该版实际执行诊断纳入已封存条件复核；旧协议推理副本仅在 Host 来源、scope 和原始结果回执匹配时计入额外执行。缺失证据保持未知，并保留零新增诊断不重复计数规则。此修复不改原账本、预算、科学门槛或原生合同，不证明调用成本或预测性能改善。

**历史 U 真实两轮验收（冻结 0.10.8）：**`RunCompleted` 回执 seq6964 确认两轮执行完成。实际 400 次候选／版本起点执行、3,600 格预测，对应 170 个不同环境起点；上限 600 中的 200 次条件预留未触发、未执行。18 组评分独立重算一致，两轮均按原逐格与时间块门槛保留种子，没有新增可接受候选，也没有独立科学认证。G0 复用 R 已暴露的开发起点，G1 虽向前隔离但依赖先前反馈，不能当作两次独立试验或因果收益。

U 的 468 个子任务均结束并结算，其中 466 接受、2 次 critic 尝试失败；20 次供应商内部重试恢复，15 次 Host 合同纠错在原子任务内成功。全部 3,600 格预测封存成功不等于全程无故障：G0 有 10 格咨询降级，G1 无降级。U 已报告 20,889,210 tokens，另有预检 15,068；仍有用量不完整记录，不能换算为完整账单。新增“拒绝后补调用工具”特殊路径在 U 实跑中的覆盖为 0，其验证来自本地回环和 T 轨迹回放，不能借两轮完成扩大覆盖声明。

**前版 0.10.9 展示统计修复：**初筛的零新增执行诊断仍保留来源与记录数，但不再重复累计实际预测时点、评分项、成功/失败次数和样本工作项。真实 R 旧账本只读回放从误显示的 210 / 1,890 修正为 200 / 1,800；预算核销 300（含跳过 100）保持，同起点的父子臂真实执行仍分别计数。95 项定向回归通过，包括多轮、兼容边界和真实事件重放；完整门禁需核对同版本归档的验收回执。本修复不改账本、预测、科学分数、门槛、工具预算或原生运行语义，也不宣称减少模型调用或 token。

**前版 0.10.8 验证状态：**完整真实数据Python 2,005项、Node 338项、前端、lint与锁文件均通过，失败0、跳过0；506个受检源码文件在测试期间未变。15个原生协议场景、7个生产Host回环与T三条真实轨迹受限回放通过，零远端模型调用。前版0.10.7通过Python 2,005项、Node 326项及本地认证回环，但外部T运行在24次起点预测后暂停：模型根据Host反馈补充预测工具，第二次输出有效，最终轨迹检查仍禁止中间工具。该版只针对可信Host语义反馈协调该边界；原工具额度、两次提交上限、绝对期限和科学门槛保持。后续 U 的真实两轮证据按其冻结 0.10.8 身份单独记录，不借用 R 或改记为 0.10.11 新模型验证。

**前版工程回归（2026-10-20）：**0.10.6 隔离源树通过完整真实数据 Python 1,994 项、完整 Node 316 项，均无失败或跳过；前端 smoke、锁文件检查与 lint 通过。真实 Harness 0.2.0-rc.2 的 5 类样本集合校验场景使用 16 次本地 HTTP 请求；另以 17 次本地 HTTP 验证 Host 错误经过原生路由、Python 解码与并发接纳的完整边界，均不调用外部模型。源分发回归发现新增 `.mjs` fixture 缺失，已修正 MANIFEST 并重新全量通过。首次全量的三个旧测试预期已按明确供应商证据契约修正，失败日志保留。R 真实实验继续使用冻结 0.10.5，完整轮末结论见交付验收记录；局部预测、预检和工程测试不替代科学认证。没有执行要求干净工作区的发布构建。模型档位和冻结边界见[推理配置说明](docs/reasoning-configuration-2026-10-20.md)。

**历史工程验收（2026-10-04，0.10.0）：**完整 `make verify` 通过 1,875 项 Python、280 项 Node 测试，零失败、零跳过；包含真实 AGC 数据及隔离 Harness 本地适配器验收。联合策略的多操作编译、五臂比较、数值程序实际执行、计费重放和跨轮反思已通过工程回归。截至当时，远程运行只提供部分阶段证据，新协议尚未完成真实提供方的完整进化与独立科学认证。该历史记录不代表后续版本的验收结果。见[本次实施记录](docs/plugin-joint-evolution-2026-10-03.md)。接收方应在自己的环境重新验收。

### 构建交付包

构建需要 `uv` 和干净的 Git 工作区；先完成源码、文档及内置插件包更新，再提交并构建：

本次 0.10.11 交付使用独立的源码快照。隔离快照须在提交后用于发布构建，其提交、源码清单和产物摘要由本版交付索引记录；这不表示原共享工作区的未提交修改已被提交，也不表示已推送或正式发布。最终构建是否完成以该索引的实际回执为准，本地研究版和未签名插件的限制保持。

```bash
make release PYTHON="$PWD/.venv/bin/python"
make verify-artifacts PYTHON="$PWD/.venv/bin/python"
```

`dist/` 输出 Python wheel、源码 sdist、完整交付归档及 `SHA256SUMS`，归档记录源码与产物校验信息。源码安装所需的原生插件 `.tgz` 随仓库提供。可用 `python -m pip install dist/<wheel-file>.whl` 验证离线交付安装，再执行同一运行时安装命令。完整验收清单见 [RELEASE-CHECKLIST](RELEASE-CHECKLIST.md)。

真实远程链路验收脚本 `scripts/real_api_agent_tool_acceptance.py` 需绑定既有冻结运行与同源发布物，会产生额外调用；它是工程检查，不能作为候选晋级或科学评分证据。

## 界面示意

以下六张图来自仓库保留的静态演示，展示主要工作区布局；并非本次真实运行成绩，也不是 0.9.0 全部界面的验收截图。当前已增加独立评测工作区。截图约束见 [截图清单](docs/screenshots/README.md)。

<details>
<summary>展开演示截图</summary>

![开始运行](docs/screenshots/01-run-settings.jpg)
![运行参数](docs/screenshots/02-parameter-design.jpg)
![训练数据](docs/screenshots/03-training-data.jpg)
![运行过程](docs/screenshots/04-evolution-process.jpg)
![候选方案](docs/screenshots/05-candidate-evaluation.jpg)
![人工审核](docs/screenshots/06-human-governance.jpg)

</details>

## 安全、限制与许可

- API key、认证令牌、本地模型配置、运行数据库及原始会话日志不进入 Git 或交付包；`.runtime/` 和常见凭据文件默认忽略。上传导出物前仍须单独核验内容。
- 两个本地服务默认绑定回环地址。非回环 API 需要服务令牌；当前进程级令牌不是多用户权限体系，前端 capability 列表不等于后端用户授权。
- 对外提供的是脱敏结构化证据，不包括私密推理或原始认证头。外部论文元数据仅作为研究线索，不能直接变成可执行算法。
- 本地单进程与 SQLite 面向研究交付；未完成生产多租户、插件签名、官方市场发布或设备控制验收。
- 源码遵循仓库 [LICENSE](LICENSE) 的授权评估／交付条款，**不是宽松开源许可**；第三方软件、数据及模型服务各自遵循其许可。详情见 [NOTICE](NOTICE)。
