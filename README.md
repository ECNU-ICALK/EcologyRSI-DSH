# EcologyRSI-DSH：面向农业生态的 RSI 工作台

EcologyRSI-DSH 是一个基于 DeepSeek Harness 的 **AI for Ecology** 框架，用来把人工智能真正用到农业和生态研究中。

使用者只需要告诉模型两件事：**手里的数据是什么，以及想解决什么问题**。模型会自己完成资料调研、选择合适的方法、调用已登记的分析工具、训练和评测候选方案，并根据实验结果继续改进。每一步都会留下可查看的记录，研究者可以知道模型改了什么、为什么改、效果是否真的变好。

这里的 RSI 指 Recursive Self-Improvement（循环自进化）。它不是一个只会聊天的助手，而是一套能够围绕研究目标持续做实验、比较证据并改进下一轮方案的流程。它不会直接执行模型生成的代码，也不会把一次偶然的高分当成结论；只有在相同数据条件下确实提高，并通过必要检查的方案，才会进入下一轮。

> 当前交付：`0.3.55` 可交付候选版 · Python 3.10+ · DSH `0.1.0-rc.6` · 本地服务端口 `8777/8848`

## 先看这里

- [项目意义与边界](#为什么做这个工作台)：它解决什么问题，不能做什么。
- [快速启动](#快速启动)：启动服务或打开无后端演示。
- [界面概览](#界面概览)：六个工作区和当前版本的界面证据。
- [架构](#架构)：浏览器、服务端和数据账本如何配合。
- [测试与发布](#测试与发布)：本地验证、真实数据验收和交付边界。

## 它是怎样工作的

工作台把“研究问题”变成一个可以反复验证的闭环：

1. 使用者提供数据、预测目标和基本约束。
2. 模型阅读数据说明和已有研究，提出若干可执行的方案。
3. 服务端在固定的数据分区和评测规则下训练、比较这些方案。
4. 效果有提高的方案进入下一轮，效果变差的方案保留失败原因，不会继续污染后续实验。
5. 研究者随时可以查看过程、暂停运行、补充方向，或恢复到可验证的状态。

模型可以提出新方向，但只能使用宿主已经登记并经过限制的工具；它不能改写 DSH 的权限、评测规则或数据边界。

| 参与者 | 负责什么 | 不负责什么 |
|---|---|---|
| 研究工作台 | 固定数据、生成候选方案、训练和公平比较、保存全过程记录、支持人工确认 | 不执行模型生成的任意代码，不因一次分数变好就跳过检查 |
| DeepSeek Harness | 提供模型会话、工具调用、权限和页面托管 | 不替研究者作科学结论，也不直接控制温室设备 |

这里的“自进化”指系统根据上一轮实验结果提出下一轮改进方向，不是模型不受限制地复制自己。可以改进的内容包括：

- **研究流程**：调研、规划、诊断和复盘的步骤；
- **工具组合**：在什么情况下使用哪些农业生态分析工具；
- **实验安排**：训练、评测和复盘的顺序与参数；
- **模型方法**：开发者已经实现并登记的预测模型和参数范围。

所有变更都会先生成一份固定的“方案版本”，再用相同数据、时间窗口和评测规则比较。只要相对当前使用版本有明确的总体提高，并通过完整性、稳定性和安全检查，就可以成为下一轮继续优化的父版本；搜索阶段暂时领先不等于正式发布。

### 一次实验如何产生下一版本

```text
冻结数据与评测合同
        ↓
生成研究计划和改进建议
        ↓
转换为已登记的模型配置（拒绝任意代码）
        ↓
用同一批样本初筛，再对入围方案做局部实验
        ↓
科学门禁 + 独立评审 + 配对稳定性
        ↓
采用新版本，或保留旧版本并记录原因
```

## 为什么做这个工作台

生态和农业模型研究通常不只是“换一个更大的模型”。真正影响结论能否复现、能否解释的，往往是数据集和观测序列是否固定，时间切分有没有泄漏，候选方案是否在同一条件下比较，网上看到的算法是否真的能在当前数据上执行，以及失败的尝试和人工判断有没有留下证据。过去这些步骤容易散落在脚本、笔记和临时对话中，研究者很难在下一轮准确回答“改了什么、为什么改、效果是否真的变好”。

EcologyRSI-DSH 把模型放在“研究助理”的位置：模型负责提出方向，服务端负责固定数据、训练、评测和保留规则。每个候选方案都在相同的时间区间上比较；本轮总体更好的方案可成为下一轮继续优化的起点，但仍会单独显示是否通过严格认证。这样即使结果变差，也能明确知道是哪一步导致的，并可以暂停、恢复和复盘。

## 研究价值

| 研究问题 | 工作台提供的支撑 |
|---|---|
| 如何让不同方案在同一条件下比较？ | 创建运行时固定数据集、实验批次、时间分区、预测模型/评测器版本、随机种子和检查规则；每轮使用同一份知识记录。 |
| 如何避免把论文摘要或模型幻觉直接变成代码？ | 网上资料只作为可追溯的参考，并且必须对应到服务端已登记的模型能力；未适配的方法只能作为研究线索。 |
| 如何判断“变好”而不是只看一个漂亮分数？ | 使用时间前向反馈集、仅在 `training_fit` 中选择的持续性/24 小时季节性强基线、多时距指标、物理范围约束和独立评审；同轮候选必须在相同样本窗口比较，新版评分还要求实用差异与 24 小时配对区块置信度。 |
| 如何保留失败经验和人的判断？ | 候选、失败样本、人工意见和每轮结论都会写入只追加的记录，支持暂停、干预、重放和导出。 |
| 如何从一个数据集扩展到更多生态问题？ | 数据适配器、预测器、评测器和知识映射采用注册表接口，可逐步接入温室环境、作物水分、产量和机理模型，而不改变进化闭环。 |

当前交付是面向本地研究开发的可交付版本，不是生产控制系统。它采用 Python 3.10、标准库、单进程和 SQLite；不执行模型生成的 Python/Shell 源码，不连接真实温室设备，也不开放隐藏评测或正式发布权限。研究结果应被理解为在明确数据边界内的离线证据，而不是控制收益或跨场景泛化的保证。

## 默认一轮如何执行

当前外层仍是“每轮 4 个候选、筛选后保留 Top 2”的选择逻辑；变化只发生在每个入围候选内部。默认执行合同如下：

| 阶段 | 默认工作量 | 作用 |
|---|---:|---|
| 同窗初筛 | `4 × 64` 次完整预测 | 四个候选方案使用同一批 64 个预测起点，选出前两名 |
| 入围方案的多批次优化 | 最多 `1,900` 次完整预测 | 两个入围方案共享 500 个固定预测起点；每批都让当前版本和新建议在同一批样本上比较 |
| 每轮结束的独立检验 | `3 × 169` 次完整预测 | 两个入围方案的最终版本与上一轮版本使用同一批冻结的独立检验样本重新比较 |
| 单轮合计 | 最多 `2,663` 次完整预测 | 默认 3 个目标 × 3 个预测时距，共 `23,967` 个评分单元 |

一次“完整预测”从一个时间起点出发，同时给出温度、相对湿度和 CO₂ 在 1、6、24 小时后的 9 个结果。系统会重复使用固定的预测起点来完成多批次比较，但每一批都保留样本批次编号，绝不会把不同批次的原始分数直接混在一起排名。

默认同时处理 4 个候选方案、最多 64 个预测起点（可配置 1–128）。每个起点固定产生 9 个结果；两个入围方案可以并行处理，但共享同一运行的总并发额度，服务端还会限制短时间内向模型服务发起的请求数量。

### 这些词是什么意思

README 后面的接口和日志会保留少量固定英文标识。它们对应的功能含义如下：

| 英文标识 | 这里实际指什么 |
|---|---|
| `cohort` | 同一轮、同一时间范围内共同用于比较的一批样本 |
| `genome` / `revision` | 一份固定的方案版本，包含模型方法和参数 |
| `origin` | 一次完整预测的起始时间点 |
| `incumbent` | 当前正在使用、需要被新方案挑战的版本 |
| `holdout` | 只在阶段结束时使用的独立检验样本，不能拿来调参 |
| `cell` | 一个“目标 × 预测时距”的评分项，例如“温度未来 6 小时” |
| `projection` | 服务端整理后提供给页面显示的数据 |
| `digest` | 用来确认数据或方案没有被悄悄改动的校验值 |
| `sidecar` | 只在本机运行、负责保存状态和执行评测的 Python 服务 |

因此，页面里看到“同 cohort 比较”，可以直接理解为“用同一批样本公平比较”；看到“更新 incumbent”，就是“把当前使用版本换成更好的版本”。

## 界面概览

前端以 DSH Web 插件运行。用户只需选择数据集、候选生成模型、独立评审模型和进化轮数；研究领域、训练时间段、预测模型和评测方式由服务端自动确定并固定，不需要在多个内部实现之间做技术选择。

下面六张截图按当前 `0.3.55` 前端的 `?demo=1` 显式演示模式核对，全部是浏览器内合成示例，只用于说明当前界面和预算口径，不代表真实 AGC 评测结果；截图中没有真实路径、运行 ID、令牌或内部网关地址。演示模式不会启动训练、写入账本或调用模型 API。

截图文件、工作区对应关系和重拍约束见 [`docs/screenshots/README.md`](docs/screenshots/README.md)。

![运行设置：选择数据集、策略模型、独立评审模型和轮数](docs/screenshots/01-run-settings.jpg)

运行设置页先检查数据是否齐全、两个模型是否分工明确以及连接配置是否可用，再允许创建任务；它不会在启动时额外探测模型接口。右侧会显示服务端自动选择的内容和可用数据量。

![参数设计：统一设置代数、候选、样本批次、并发和总预算](docs/screenshots/02-parameter-design.jpg)

参数设计页统一配置进化轮数、候选数量、每个入围方案的优化批次、每批最多修改项、轮末独立检验样本量、并发和总预算。页面同时显示“完整预测次数”和“评分单元数”，避免把一次产生 9 个结果的预测误算成 9 次模型请求。模型上下文和输出长度由 DSH 会话统一管理。

![训练数据：数据结构、分区边界与样本预览](docs/screenshots/03-training-data.jpg)

训练数据页把字段单位、训练拟合/训练反馈分区、冻结快照、来源完整性和进化训练资产放在一起，便于在看分数前先确认“用的是什么数据”。

![进化过程：同轮候选、当前最佳方案（incumbent）轨迹与阶段证据](docs/screenshots/04-evolution-process.jpg)

进化过程页按“检索 → 生成方案 → 训练 → 评测 → 独立评审 → 本轮结论”展示证据，并分别显示总进度、当前阶段、服务端排队、未完成任务、待提交请求和最近活动。只有服务端有确切数据时才显示模型服务等待。前两名方案进入多批次优化后，页面逐批显示参数修改和比较结果；绿色线表示实际被保留、用于下一轮的方案。

![候选评测：指标、约束、产物与搜索保留结论](docs/screenshots/05-candidate-evaluation.jpg)

候选评测页同时呈现误差、基线、约束违规、参数变化、训练产物校验值和搜索结论，避免把单个综合分数误读成最终发布结果。

![人工协作与治理：暂停、提交意见和权限边界](docs/screenshots/06-human-governance.jpg)

人工协作与治理页分开显示非阻塞的模型咨询、暂停后才生效的人工意见和仅记录意见，并明确隐藏评测、正式验证和发布权限仍由外部治理服务控制。

六个工作区的阅读顺序是：

1. **运行设置**：确认数据集和两个模型角色，创建并冻结运行清单。
2. **参数设计**：设置 finalist epoch、局部 batch、每批改动数、轮末 holdout、并发和总预算。
3. **训练数据**：核对字段、时间分区、样本预览和数据血缘。
4. **进化过程**：查看知识快照、候选阶段、实时请求、微批 revision 和 incumbent 轨迹。
5. **候选评测**：比较指标、科学门禁、独立评审和保留理由。
6. **人工协作与治理**：暂停后追加方向、参数覆盖、约束或父方案选择，检查哪些意见被执行、哪些只被记录。

## 已实现能力

- 接入 Autonomous Greenhouse Challenge 2018 黄瓜和 2019 番茄真实历史数据，完成 CSV 规范化、小时聚合、字段单位、缺失值处理、快照校验和时间前向分区；来源归档另按官方大小与 MD5 审计。
- 保留确定性合成数据 `generated-toy-series@1`，用于不依赖外部数据的工程回归。
- 提供四类候选生成路径：继承父代的有界参数扫描、消费上一轮指标的局部自适应搜索、服务端 DSH Bearer 网关模型提案，以及由模型输出调研计划、再由宿主编译的 `autonomous_model@1`。
- 每轮先生成可审计的知识快照：内置核验目录离线可用，可选通过 OpenAlex 检索论文元数据；只有映射到本地已注册且由本运行冻结选中的能力才进入候选上下文，在线结果和未安装算法只作为研究线索。
- 提供合成水分预测器、温室滚动残差预测器和外生变量岭回归残差预测器；服务端按数据集与模型自主研究结果自动绑定兼容评测器，可执行 1 小时或 1/6/24 小时多时距评测，目标为室内气温、相对湿度和 CO2 浓度。
- 支持内置规则评审或独立的 OpenAI-compatible 远程评审模型；候选生成模型与评审模型不能使用同一个模型标识。
- 提供中文 DSH webview 插件，包含“运行设置、参数设计、训练数据、进化过程、候选评测、人工协作与治理”六个工作区。
- 通过“查看分区”切换训练拟合/训练反馈样本，并展示字段和单位、来源归档校验、未就绪数据资产、每候选一条的进化训练轨迹、数据与分区校验值、候选参数、训练产物、多时距预测预览、得分轨迹、历史最高得分、真实六阶段状态、评测指标、搜索保留结果和脱敏事件。
- 支持暂停后追加方向建议、参数覆盖、数值边界约束或指定父方案；恢复后只处理下一轮。可唯一解析的方向建议按固定步长应用，参数覆盖与约束由宿主校验，无法唯一解析的文字只记录为“未执行”，历史记录不被改写。
- 支持运行创建、暂停、恢复、取消、逐轮推进、SQLite 重放、摘要、导出、校验和导入。

## 架构

```text
中文 DSH 插件 / HTTP 客户端
              |
              v
      本地 JSON API（脱敏投影）
              |
   +----------+-----------+
   |          |           |
数据注册表  策略路由器   科学评测器
   |          |           |
AGC CSV    本地策略或     训练产物 +
适配与分区  DSH 模型网关  独立评审
   +----------+-----------+
              |
       Evolution Director
              |
       追加式 SQLite 账本
```

浏览器只读取面向界面的脱敏投影。数据凭据、DSH Bearer 凭据、完整任务清单和原始事件载荷均保留在服务端。

后端实现按职责分层，已删除顶层同名兼容模块，业务代码直接从所属子包导入：

```text
src/ecologyrsi_dsh/
  application/    CLI 与配置装配
  core/           领域实体、事件账本、进化状态机
  data/           数据合同、目录、下载准备、温室适配与分区
  evolution/      候选策略与人工干预约束
  evaluators/     预测、科学评测与指标
  integrations/   模型绑定与 OpenAI-compatible 网关
  knowledge/      知识目录、公开元数据检索、能力映射与轮末判断
  presentation/   训练资产、运行投影与导出
  api/            HTTP 端点、命令执行、事件、静态资源和服务生命周期
```

前端不需要打包器，按功能拆分在 `plugins/ecology_evolution/assets/js/`；根目录 `app.js` 只负责事件绑定和启动。维护原则是入口保持薄层、共享合同集中、模块按职责拆分；现有账本、模型网关、评测器和部分前端渲染模块仍偏大，后续功能扩展前应优先继续拆分，而不是向这些文件追加新的职责。

## 快速启动

本项目当前固定使用 DSH `0.1.0-rc.6`。请先安装 Node.js（包含 `npm`）和
Python 3.10 或更高版本，再按以下顺序执行。

### 1. 安装 DSH

```bash
npm install --global @deepseek-ai/dsh@0.1.0-rc.6
dsh --help
```

如果本机已经安装了这个版本，可以跳过本步骤。不要直接省略版本号安装最新预览版，
因为 DSH 仍在快速迭代，本项目的宿主插件和 preset 已按 `0.1.0-rc.6` 的运行时接口冻结。

### 2. 安装 EcologyRSI-DSH

```bash
cd <repo-dir>
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
ecologyrsi-dsh install-dsh-runtime --profile web
```

`install-dsh-runtime` 会把 EcologyRSI 宿主插件和当前六个不可变角色 preset ID
安装到 DSH 的 `web` profile。安装器会校验版本和内容摘要；同一份未发生漂移的
安装可以安全地重复执行。本版本不安装历史 preset，也不承诺旧运行合同兼容。

### 3. 启动服务

下面的命令在一个终端中启动内部 Python sidecar，并以前台方式启动 DSH Web：

```bash
source .venv/bin/activate
mkdir -p .runtime

export ECOLOGYRSI_DSH_RUNTIME_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export ECOLOGYRSI_SIDECAR_TOOL_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export ECOLOGYRSI_DSH_RUNTIME_URL="http://127.0.0.1:8848"

ecologyrsi-dsh serve \
  --host 127.0.0.1 \
  --port 8777 \
  --db "$PWD/.runtime/ecologyrsi-dsh.sqlite3" &
ECOLOGYRSI_SIDECAR_PID=$!
trap 'kill "$ECOLOGYRSI_SIDECAR_PID" 2>/dev/null || true' EXIT

dsh --profile web --port 8848
```

打开 <http://127.0.0.1:8848/>，点击侧栏中的“生态模型进化”。浏览器只访问 DSH
的 `8848` 端口；`8777` 是仅供 DSH 宿主插件调用的本机 sidecar，不是需要单独打开的
前端端口。按 `Ctrl-C` 停止 DSH 后，上述 `trap` 会同时停止本次启动的 sidecar。

页面会读取当前真正可运行的数据集、预测模型、策略、评测器和 DSH 模型，不能运行的
目录项不会进入启动选项。首要选择是“数据集”，研究领域由数据集目录自动推导。

### 可选：无后端静态演示

无后端的静态演示必须显式使用 `demo=1`，且不会写入 SQLite：

```bash
python -m http.server 4173 --directory plugins/ecology_evolution
```

打开 <http://127.0.0.1:4173/?demo=1>。

## 真实数据

建议显式指定可移植的数据根目录，然后先审计、再按需准备数据：

```bash
export ECOLOGYRSI_DATA_ROOT=/path/to/greenhouse
PYTHONPATH=src python -m ecologyrsi_dsh data audit
PYTHONPATH=src python -m ecologyrsi_dsh data fetch agc_cucumber_2018
PYTHONPATH=src python -m ecologyrsi_dsh data fetch agc_tomato_2019
```

`data fetch` 从目录记录的 HTTPS 地址下载到
`$ECOLOGYRSI_DATA_ROOT/<dataset_id>/_archives/`，使用唯一临时文件，完成预期大小和 MD5 校验后才发布归档。默认继续安全解压；`--archive-only` 可只准备归档。已校验归档和已经满足必需文件合同的解压目录会直接复用；已有但校验不一致的归档或解压文件不会被覆盖。ZIP 会拒绝绝对路径、`..` 路径穿越、符号链接和特殊文件；番茄 7z 使用系统 `bsdtar`，同样先检查成员路径和类型。

未设置 `ECOLOGYRSI_DATA_ROOT` 时，程序使用
`~/.ecologyrsi-dsh/data/greenhouse`；显式设置环境变量始终具有最高优先级。

目录下包含数据集标识同名子目录。当前支持自动准备的状态如下：

| 数据集标识 | 内容 | 许可 | 当前状态 |
|---|---|---|---|
| `agc_cucumber_2018` | 2018 黄瓜，6 个温室 episode，每个约 2754 个小时行 | CC0-1.0 | 可运行 |
| `agc_tomato_2019` | 2019 番茄，6 个温室 episode，每个约 3983 个小时行 | CC0-1.0 | 可运行 |
| `agc_lettuce_online_rgbd_2021` | 生菜在线 RGB-D | CC-BY-4.0 | 仅目录登记 |
| `agc_lettuce_timeseries_rgbd_2022` | 生菜时序与 RGB-D | CC-BY-4.0 | 仅目录登记 |

两个生菜条目尚未实现规范化适配器，不能创建运行。服务启动时会按必需文件模式重新计算就绪状态，因此“目录中登记”不等于“本机可运行”。

真实数据说明接口同时返回 `readiness.provenance` 和 `readiness.source_integrity`。后者核对 `<dataset>/_archives/` 中来源归档的文件名、大小与 MD5；当前归档记录为黄瓜 `243eaa9041da23d0c4bf99576715aa44`、番茄 `2a0c7f3332881caef54ca8f4dc60c9a3`。归档缺失或不匹配会在界面明确告警，但不会把已经解压并满足必需文件合同的数据伪装成未就绪；运行就绪与来源可追溯性是两个独立状态。

### 时间前向分区

真实温室 episode 按时间顺序固定划分，所有范围均为左闭右开的行区间：

```text
约 0%       30%       60%          80%          100%
  | training_fit | 1h | training_feedback | 24h | development | 24h | gate |
```

- 前 60% 为训练区，其中前半用于 `training_fit`，后半用于 `training_feedback`，两者之间设置 1 小时 embargo。
- 随后的 20% 为 `development`，与训练区之间设置 24 小时 embargo。
- 最后约 20% 为 `gate`，与开发区之间设置 24 小时 embargo。
- 名称含 `Reference` 的 episode 标记为 `external_holdout`。
- 插件从 5 个非 Reference 优化 episode 中按数据集目录确定性绑定一个；绑定结果进入样本请求、运行清单、数据 digest 和训练产物血缘。当前界面不要求用户单独选择 episode。
- 浏览器样本 API 只允许 `training_fit` 和 `training_feedback`；`development`、`gate`、`external`、`hidden`、`test`、`final` 的原始行一律拒绝。
- 当前本地自适应闭环只在 `training_fit` 拟合，在后续 `training_feedback` 评测；不会使用 `development`、`gate` 或外部留出集调参。

## 模型如何被比较

| 类型 | 内部标识 | 用途（页面会自动选择兼容组合） |
|---|---|---|
| 策略 | `parameter_sweep@1` | 继承已完成父候选参数，同轮按稳定槽位扫描有界维度 |
| 策略 | `adaptive_local@1` | 根据父候选得分、是否通过和相对改进确定调整方向与步长 |
| 策略 | `dsh_authenticated@1` | 服务端 Bearer 网关模型接收脱敏父参数、聚合指标、评审建议和人工意见后提案，宿主继续校验字段与范围 |
| 策略 | `autonomous_model@1` | 服务端模型生成一次结构化调研计划，宿主从已登记能力编译有界参数提案 |
| 预测模型 | `greenhouse-rolling-residual@1` | 使用目标历史窗口与训练拟合分区偏差开展 1 小时预测 |
| 预测模型 | `greenhouse-exogenous-ridge@1` | 使用外气象、设定值、动作、根区等外生特征学习相对持续性基线的残差 |
| 预测模型 | `greenhouse-targetwise-ridge@1` | 分目标缩放岭回归残差修正；单个目标的缩放系数为 0 时独立使用持续性预测 |
| 评测器 | `toy_time_forward@1` | 合成数据验证分区工程评测 |
| 评测器 | `greenhouse_time_forward@1` | 真实温室 1 小时训练拟合/训练反馈时间前向评测 |
| 评测器 | `greenhouse_multihorizon_time_forward@1` | 岭回归专用的 1/6/24 小时训练拟合/训练反馈评测 |
| 候选生成 | `host_parameter_generator@1` | 内置有界参数生成器 |
| 独立评审 | `rule_judge@1` | 内置固定规则门禁 |

模型和评测器必须兼容：滚动残差模型只支持 1 小时评测；外生变量岭回归和目标级岭回归同时兼容 `greenhouse_time_forward@1` 的 1 小时评测与 `greenhouse_multihorizon_time_forward@1` 的 1/6/24 小时评测；合成预测器只支持 toy 评测。工作台会自动对齐可选项，服务端仍在运行写入前再次拒绝不兼容组合。

滚动残差候选修改 `blend`、`window`、`bias_scale`；统一缩放的岭回归候选修改 `history_steps`、`ridge_alpha`、`residual_scale`；目标级岭回归则分别修改空气温度、相对湿度和二氧化碳的残差缩放系数，其中某个目标的系数为 0 时该目标单独使用持续性预测。所有参数都由策略模型在宿主登记范围内提出，不由用户选择。岭回归的特征选择、缺失值填补和标准化只在 `training_fit` 拟合，并按真实小时戳构造 1/6/24 小时目标；结果变量不进入外生特征。评测在后续 `training_feedback` 比较三项目标的 MAE、RMSE、归一化 RMSE、技能得分、缺失行和物理范围违规。评分基线在预测前仅用 `training_fit` 对每个“目标 × 时距”比较持续性与 24 小时季节性基线，同样本 RMSE 更低者被冻结并绑定 digest；模型生成预测仍保留原持续性参照，不受评分基线替换影响。固定科学门禁要求每个目标/时距不劣于冻结强基线、总体技能得分大于 0 且预测不违反目标物理范围；若使用远程 judge，还必须同时得到独立评审接受，候选才算 `passed`。

候选通过基础检查，并不代表它会被保留。同一轮的所有方案必须使用完全相同的评测窗口；服务端先检查数据完整、预测链完整、没有物理约束违规，再比较总体提高。只要新方案相对当前方案有明确的总体提高，就可以成为下一轮搜索版本；稳定性、逐项风险和独立检验结果会单独展示，不能被一次偶然高分掩盖。每轮的完整比较记录都会保存下来，效果变差的方案不会成为后续父版本。

## 网上知识检索与可执行方案筛选

新建 DSH-native 运行把自主研究协议冻结为 `dsh-model-search-reflect@1`。每轮形成下面的可重放闭环：

1. `generation.search-plan`：策略模型读取上一轮指标、批次复盘和当前方案，自主提出最多 6 条检索词与关注问题；这一步不会直接访问网络，也看不到原始反馈行。
2. Host 检索工具：宿主先执行模型检索词，再补充确定性的领域/弱点查询；读取内置核验目录，并在启用联网时查询 OpenAlex 元数据。网络失败只产生告警并回退内置目录。
3. `generation.research-synthesis`：研究模型只能引用本轮已保存的资料，整理出数量匹配、彼此不同且确实可以执行的改进方向。参数必须明确写出增加或减少；服务端会拒绝藏在自然语言里的精确参数赋值。
4. `candidate.propose`：每个候选方案选择一个方向，模型给出初始参数修改；服务端把它转换为已登记的模型配置，并检查目标、范围、步长和是否重复，拒绝任意代码。
5. 四个候选方案先用同一批 64 个预测起点筛选出前两名。随后每个入围方案进行最多 500 个预测起点的多批次优化：每批让当前版本和一个新建议在同一批样本上比较，只有总体确实提高的新建议才会继续接棒。
6. 多批次优化结束后，两个入围方案的最终版本与上一轮版本使用同一批 169 个独立检验起点重新比较。不同批次的原始分数不会直接混排；服务端根据这次统一比较、科学检查和稳定性结果决定下一轮使用哪个版本，并把原因交给下一轮研究。

启动检索不是唯一检索时机。Researcher、Candidate Proposer、Sample Planner、Sample Critic、Generation Judge 和 Coordinator 在各自阶段加载必需 Skill 后，都可以在遇到证据缺口时调用零到三次同一个 `web_search`。模型只提交 1–4 条短查询和阶段内 `retrieval_key`，不能选择 provider：包装器先调用 DSH 内部 `ctx.web.search`；若 provider 不可用、出错，或结果少于 2 个不同 HTTPS 来源、少于 2 个有标题/摘要的证据来源、与查询没有词项重合，Python sidecar 才自动调用 OpenAlex 元数据回退。主结果与回退结果按 URL 去重、最多保留 8 个来源，并以 `DshRetrievalExecuted` 事件持久化；相同阶段检索在恢复时先重放，不重复联网。动态结果只作为推理参考，不能自行进入冻结 `evidence_ref`、改变预测向量、评分、门禁或晋级。

插件不暴露 provider 专用工具，也不开放 `web_fetch`。显式取消会终止当前检索而不发起回退；检索仍必须发生在 Skill 之后、预测/结构化终端工具之前，Sample Planner 的一次登记预测工具调用保持不变。

这里的“实现方案”指把模型建议转换成 EcologyRSI 插件自己的受限模型配置，并调用宿主已登记的科学工具；不会修改 DSH 基本框架，也不会执行模型生成的 Python、Shell、依赖安装或任意网络代码。新增算法必须先由开发者实现、测试并登记，之后才能被模型选择。

在线内容不会被下载为代码，也不能修改参数范围、数据分区、评测器或科学门禁。服务端默认仅使用内置目录；插件创建的运行默认启用 OpenAlex 元数据检索。可用 `ECOLOGYRSI_KNOWLEDGE_ONLINE=0` 在部署层强制关闭，或设置为 `1` 强制开启。每轮最多执行 6 条、每条最多 180 字符的短查询；模型生成的查询拥有最高优先级，Host 的具体弱点词随后执行，宽泛领域词最后兜底。结果够用就停止，空结果才继续，并按 OpenAlex work ID 去重。系统不会把全部领域、弱点和失败词拼成一个长查询。OpenAlex 单次请求默认等待 20 秒，可用 `ECOLOGYRSI_OPENALEX_TIMEOUT` 设置正数秒值；超时、连接错误及 HTTP 408/425/429/5xx 最多额外重试 3 次并短指数退避，存在 `Retry-After` 时在单次 5 秒安全上限内按服务端建议值等待，其他 HTTP 错误立即回退到内置目录。若共享系统代理对 OpenAlex 返回 429，且部署策略允许该固定 HTTPS 来源直连，可在启动服务时仅设置 `NO_PROXY=api.openalex.org`，模型 provider 仍继续使用原代理。每次响应仍只读取最多 1 MB 的元数据，不做启动探活。

## 页面显示的结果

运行投影包含两个面向解释和治理的派生视图：

- 页面按轮展示资料检索、方案生成、训练、评测、独立评审和保留决定，并列出资料来源、实际采用的方法和结论。
- 每个候选方案只有一条脱敏的过程记录，按顺序串起输入背景、模型建议、服务端检查、训练预测、评测反馈、下一步方向和父子版本关系。记录只提供必要的结构化摘要，不返回隐藏评测、原始数据行或模型私有推理。

过程记录会标记为“改进保留”“改进不足”“需要隔离”或“等待决定”。这些记录只是帮助下一轮改进的证据，不能未经审核直接当作正式训练数据。

## DSH 原生 Agent 运行时

0.3.55 新建运行使用 `dsh_native_plugin_evolution@1`：Agent Session、上下文压缩和
一次性结构化 subagent 由 DSH 管理；逐 origin 的 `sample.plan` 使用直接子 Agent。
Python 只提供科学工具与持久账本。安装后直接运行：

```bash
ecologyrsi-dsh install-dsh-runtime --profile web
dsh --profile web --port 8848
```

完整的首次安装、sidecar 启动和令牌配置见上文“快速启动”。浏览器只访问 DSH 的
`8848` 端口，不需要单独启动前端端口；Python sidecar 默认仅监听
`127.0.0.1:8777`。

## DSH OpenAI-compatible 模型目录

推荐分别配置候选生成模型和独立评审模型。插件的两个角色下拉框共同读取后端 `dsh_models` 登记目录；具备安全后端路由、服务端凭据和对应角色的条目可以直接用于运行，角色不匹配、缺少凭据或被 URL 安全策略阻止的条目会禁用。目录通过 `configured_strategy_model_count`、`configured_review_model_count`、`executable_strategy_model_count`、`executable_review_model_count` 和 `roles_ready` 报告运行就绪状态。DSH Web Profile 打开插件时，会通过宿主 `llm.models` 目录把当前已登记的 provider/model 脱敏传入握手。后端 `dsh_models` 是执行配置与调用健康状态的权威目录；仅存在于宿主的模型仍会显示，但会以 `host_route_not_available_to_sidecar` 原因禁用，不能被当前后端选择或执行。密钥只放在服务端环境变量中：

未设置 `ECOLOGYRSI_DSH_MODELS_JSON` 时，Python 服务会自动读取当前用户的 `~/.dsh/settings.yaml` 和权限为 `0600` 的 `~/.dsh/.credentials.yaml`，把 DSH 的 provider/model 目录转换为同一份 `provider/model` ID。设置 `ECOLOGYRSI_DSH_DISCOVERY=0` 可关闭自动发现，设置 `ECOLOGYRSI_DSH_SETTINGS_FILE` 或 `ECOLOGYRSI_DSH_CREDENTIALS_FILE` 可指定文件位置。自动发现只接受 OpenAI-compatible provider；非回环 `http://` 地址会以 `insecure_http_blocked` 原因显示为不可用，交付配置必须使用 HTTPS。创建运行只校验路由、凭据、执行配置和角色，不发送 API 健康探测；真实连通性与 JSON 契约由提案或评审请求检查。这些请求默认单次可等待 900 秒、最多尝试 4 次，对 408/425/429/5xx、超时和瞬时网络故障指数退避，并尊重服务端 `Retry-After`。可用 `ECOLOGYRSI_DSH_MODEL_TIMEOUT`、`ECOLOGYRSI_DSH_MODEL_MAX_ATTEMPTS`、`ECOLOGYRSI_DSH_MODEL_RETRY_BASE_SECONDS` 和 `ECOLOGYRSI_DSH_MODEL_RETRY_MAX_SECONDS` 调整；瞬时失败只记录业务调用诊断，不撤销已持久化的验证状态。

```bash
export ECOLOGYRSI_POLICY_TOKEN='replace-with-server-secret'
export ECOLOGYRSI_JUDGE_TOKEN='replace-with-server-secret'
export ECOLOGYRSI_DSH_MODELS_JSON='[
  {
    "id": "policy-main",
    "label": "DSH 候选生成模型",
    "roles": ["propose"],
    "gateway_url": "https://dsh.example/v1",
    "model": "policy-model-name",
    "api_key_env": "ECOLOGYRSI_POLICY_TOKEN"
  },
  {
    "id": "judge-main",
    "label": "DSH 独立评审模型",
    "roles": ["judge"],
    "gateway_url": "https://dsh.example/v1",
    "model": "judge-model-name",
    "api_key_env": "ECOLOGYRSI_JUDGE_TOKEN"
  }
]'
```

网关调用 `{gateway_url}/chat/completions`，使用 Bearer 认证和 JSON object 响应格式。远程地址必须使用 HTTPS；只有 `localhost` 或回环 IP 可以使用 HTTP。URL 不允许内嵌凭据、查询参数或片段。模型目录和 API 投影会删除密钥及密钥环境变量名。

自主运行必须在目录中配置两个不同的模型连接 ID，并分别声明 `propose` 与 `judge` 角色；单模型配置不属于当前交付合同。

### New API / GLM 5.2 双角色示例

New API 只要提供 OpenAI-compatible 的 HTTPS `/v1` 入口即可接入。下面把同一服务中的 GLM 5.2 分成候选生成和独立评审两个模型连接；请将 `<new-api-host>` 替换为实际部署地址，并确认该部署使用的模型标识确实是 `glm-5.2`（不同网关可能使用别名）。密钥只放在服务端环境变量，不要写入插件或提交到仓库：

```bash
export NEWAPI_GLM52_POLICY_TOKEN='replace-with-newapi-server-secret'
export NEWAPI_GLM52_JUDGE_TOKEN='replace-with-newapi-server-secret'
export ECOLOGYRSI_DSH_MODELS_JSON='[
  {
    "id": "newapi-glm52-policy",
    "label": "New API GLM 5.2 候选生成",
    "roles": ["propose"],
    "gateway_url": "https://<new-api-host>/v1",
    "model": "glm-5.2",
    "api_key_env": "NEWAPI_GLM52_POLICY_TOKEN"
  },
  {
    "id": "newapi-glm52-judge",
    "label": "New API GLM 5.2 独立评审",
    "roles": ["judge"],
    "gateway_url": "https://<new-api-host>/v1",
    "model": "glm-5.2",
    "api_key_env": "NEWAPI_GLM52_JUDGE_TOKEN"
  }
]'
```

创建运行时使用 `strategy_model_id: "newapi-glm52-policy"` 和 `review_model_id: "newapi-glm52-judge"`。两者可以使用同一 New API 账号，但必须保持不同的连接标识和角色；服务端会分别检查执行配置并把无凭据的配置摘要冻结到运行清单。网关会自动请求 `{gateway_url}/chat/completions`，不要把该路径重复写入 `gateway_url`。

工作台不提供独立的模型连接验证步骤。服务端在每次真实模型调用中使用 Bearer 凭据，并严格校验 OpenAI-compatible JSON 响应契约；连接或契约失败会写入脱敏健康状态和运行阶段记录，不暴露凭据或网关地址。运行就绪读取 `execution_available` 和 `roles_ready`。

## 模型自主调研与受限能力编译边界

模型自主工作流默认启用：创建运行只冻结数据、模型路由与预算，不调用模型 API；后台开始首轮后，策略模型才在轮次上下文中返回受限 JSON 计划，内容包括候选团队角色、预测模型族、搜索策略、公开来源摘要、能力采用理由和置信度。每一代只生成并持久化一份共享研究计划，同代候选复用该计划。未明确切换到另一预测器时，系统继承当前已采用的预测器并按它生成参数 schema；显式切换时不会把旧 pipeline 的蓝图或 synthesis 错配到新算法。生产 `research_compile_evolve@1` 每轮都必须重新提交与冻结证据相绑定的 Blueprint 和 synthesis；当宿主确认无兼容可执行证据时，只能返回显式 `algorithm_synthesis_degradation`，不能通过省略字段静默继承。首次漏字段、错误降级或未引用本轮调研证据时，网关会带契约反馈再请求一次；第二次仍不合格则在严格模式中阻断轮次。服务重启时优先重放已冻结计划，不重复发起研究请求。联网调研只读取固定的 OpenAlex 论文元数据和有界摘要；离线时使用内置知识目录。

每次逐代 research 还会收到一个完全由追加式事件账本重放派生的 `cross_generation_experience`。系统最多扫描最近 24 个可用运行，按运行轮转选取最多 8 个来源运行、12 个已分析代的修改、算法综合、算法失败、样本失败、弱目标/时距、修复成效和是否改善；问题分别进入 `active_unresolved` 与 `resolved_archived`，两组各最多 16 项。问题只有在后续同类评测证据中不再出现时才归档，后续没有相应评测证据时仍保持未解决。整个摘要采用聚合白名单，不含原始样本、预测记录或代码，UTF-8 JSON 硬限制为 16 KiB；容量超限时确定性裁剪并记录 omitted 计数。因此第 N 代可以使用不止第 N-1 代的经验，同时相同事件流在重启后会派生相同摘要。

历史结果分为“方向性经验”和“硬参数 guardrail”。小样本、单一 cohort 的结果可进入弱点与修复摘要，但不会冻结后续搜索参数。只有数据集、时间分区、评测器 ID 及 digest 完全一致，且同一 `target × horizon` 在至少 2 个样本窗口中分别有不少于 20 个样本、总样本数不少于 40、每次 skill 均非负且约束违规为 0 时，才进入硬保护候选。两个 cohort digest 不同只表示窗口不同，不代表样本独立；宿主还要求底层 population digest 完全相同，并根据 selected count 和环形 window offset 验证窗口两两不重叠。字段缺失、population 不同、窗口重叠或同一参数有多个分别达标的冲突值都不生成 hard guardrail。同一 cohort 的多代证据只计一次，并取最保守的样本数和 skill。有效保护值会在远端提案、人工覆盖和有界干预完成后由宿主最终恢复，模型上下文中的建议不能修改它。因此 `N=9` 的链路验收每个目标/时距只有 1 个样本，绝不会被标记为已验证硬约束。

这里的“能力编译”是策略模型提出选择和参数方向，宿主将已采用的知识映射、候选参数和冻结数据边界编译成不可变算法 IR。IR 只含宿主登记的特征、拟合、预测和后处理算子；模型不能写入或执行任意 Python、Shell、动态导入、依赖安装或网络工具。在生产 `research_compile_evolve@1` 工作流中，若研究计划要切换预测器，必须同时提交与宿主登记的 pipeline、算子顺序和参数名称完全匹配的 `algorithm_blueprint`；蓝图引用的同代冻结证据中，至少一条必须是 `adopted` 或 `available_not_selected` 的 predictor，并明确映射到该 pipeline。只提交预测器 ID、引用其他 pipeline，或仅有 `research_only` / `metadata_only` 证据，都不能编译执行。

策略模型在该蓝图之上提交 `algorithm_synthesis`，但它只能引用蓝图已经引用的同代冻结证据，只能选择该 pipeline 已登记的 `parameter_focus`，并且 pipeline 必须与蓝图一致；它不能增加算子、参数、依赖或代码。宿主把冻结的 plan、Blueprint 和 synthesis digest 一并编译到受限算法 IR。每个候选随后必须依次通过 compile、静态 debug 和 `training_fit` 内部时间前向 training smoke，确认登记算子、有限输出、物理边界和算子轨迹后，才进入真实样本合同。闭式岭回归等传统模型只作为宿主登记预测工具：每个 origin 必须由远程 Planner 显式选择工具；只有 Planner 表示不确定、工具结果异常或执行失败时才调用远程 Critic，候选完成宿主评分后再做聚合反思。瞬时 smoke 工具故障可重试，确定性失败进入轮末 `algorithm_failures`；同代 synthesis 与 compile/debug/评测/晋升结果的关联进入跨代经验，但明确标记为观察关联而非因果归因。

真实自主运行只使用 `dsh-strict-origin-bundle@4`。一个“智能体样本”表示一个预测起点：远程 Planner 在 DSH 子会话中选择登记的联合向量工具，Host 一次生成温度、相对湿度、CO₂ 在 1、6、24 小时的 9 个结果；不确定或失败时才进入远程 Critic 修复/拒绝路径，随后由 Host 逐单元评分。严格 checkpoint 只在整个向量和当前策略要求的角色动作完整后原子落盘，恢复时只复用能验证该冻结执行策略的 origin。候选级聚合反思可以读取评分后的结果，但无权改写已经持久化的预测。

严格运行只有在冻结统计门槛满足，且已评测 origin 具有完整 Planner/工具证据以及条件触发时的 Critic 证据，才允许晋升。finalist 的冠军与挑战者只在同一个 batch cohort 内配对比较；这次比较决定 lane champion，但不同 batch 的原始分数不能直接比较。每个 revision 的改动数受 `max_local_edits_per_batch` 限制，数值步长和所有结构变更仍受宿主信赖域及能力注册表约束。轮末必须把两条 lane 的耐久冠军与 incumbent 放回同一 holdout，未显著改善时继续保留 incumbent，不用跨窗口原始分数制造“快速进化”。

DSH-native 运行不接收 `token_limit`，上下文压缩和输出长度由 DSH Session 与模型路由统一管理。

覆盖率止损只在固定 cohort 的总体或任一目标/时距“最大可达覆盖率”低于冻结门槛时触发，即使所有尚可恢复和未执行样本全部成功也无法通过才会停止后续远程微批。未执行样本仍生成 `attempts=0` 的明确失败记录，并沿用现有最坏回退参与评分；该止损不缩小分母、不提高分数，也不改变总体和逐任务 80% 门槛。

需要在不创建完整多轮进化运行的情况下验收真实样本链路时，可执行下面的显式工程检查。脚本从 `--db` 指定的账本中只读选择最近一条同时冻结 planner 与 critic 的 `RunCreated` 绑定；可用 `--reference-run-id` 固定某次运行，`--planner`、`--critic` 和对应 digest 仅用于断言账本中的冻结值。它完整拟合 `training_fit`，再从 3 个目标 × 3 个时距各取一个不依赖标签的时间分位点。planner 只选择工具，岭回归仅在被选择后由宿主执行，critic 再接受或指定修复工具；总体和每个预测任务都必须达到 80% 覆盖率。输出明确标记为不可用于科学评分、候选晋级或训练资产。脚本先验证 wheel、sdist、完整交付包、内外部校验和与当前逐文件源码完全同源，再把这些 SHA-256 写入 `release_binding`；网络验收结束后会重新执行并逐字段比较同一绑定，期间任何源码或产物变化都会使验收失败。因此必须先构建并校验当前发布物。验收脚本不会放行非回环明文 HTTP provider，正式交付必须使用 HTTPS。

```bash
make release
RELEASE_PYTHON="$(uv python find --no-project --system '>=3.10')"
"$RELEASE_PYTHON" -B scripts/real_api_agent_tool_acceptance.py \
  --db /tmp/ecologyrsi-dsh-dsh-adapter.sqlite3 \
  --samples-per-task 1 \
  --minimum-coverage 0.8 \
  --dist-dir dist \
  --output dist/ecologyrsi_dsh-0.3.55-real-api-agent-tool-acceptance.json
```

验收无论通过或失败都会原子写入 JSON 报告；省略 `--output` 时默认写到系统临时目录下的
`ecologyrsi-dsh-real-api-agent-tool-acceptance-latest.json`。输出可以放在 `dist/` 的独立 JSON
或项目目录之外，但不能覆盖源码清单成员、输入账本、wheel、sdist、完整交付包或
`SHA256SUMS`；冲突路径只向标准输出返回脱敏失败，不写文件。失败报告只保留模型 ID、请求角色、
耗时、错误分类、重试次数和覆盖率等脱敏诊断，不保存提示词、响应正文、密钥或网关地址；命令
返回非零表示本次真实链路证据不足，并不自动表示 API 凭据失效。验收 JSON 在发布构建之后
生成并保持外置，避免报告递归绑定包含自身的归档；交付记录应另行保存该报告本身的 SHA-256。

因此，自主调研和实现计划是可审计的建议输入，不等于模型已经证明了科学有效性，也不等于正式发布或设备控制授权。候选仍必须经过训练、时间前向评测、独立评审和宿主保留规则；训练资产仍标记为需要治理审核。

## DSH 插件与服务令牌

插件清单位于 `plugins/ecology_evolution/plugin.json`，当前标记为 `delivery-candidate`，但仍是未签名 webview 包。DSH 可直接托管其中的静态前端，不需要另起前端端口；Python 后端保持独立部署，由 DSH 同源代理转发。宿主在收到 `plugin.ready` 后通过 `postMessage` 发送：

```json
{
  "type": "dsh.context",
  "api_base": "/api/ecology-evolution",
  "capability_token": "service-token-from-host",
  "identity": {"subject_id": "researcher-17", "display_name": "研究员甲"},
  "capabilities": ["evolution.projection.read", "evolution.run.create"],
  "models": [{"model_id": "dsh-policy@1", "roles": ["propose"]}]
}
```

非回环监听必须配置服务端能力令牌：

```bash
export ECOLOGYRSI_SERVICE_TOKEN='replace-with-runtime-token'
```

插件只接受同源父窗口，或 URL 中通过 `parent_origin` 明确授权的父窗口；`api_base` 只允许 canonical `/api/ecology-evolution`，并且只接受同源 API 或通过 `api_origin` 明确授权的来源。能力 token 仅保存在宿主适配模块的内存闭包中，不进入导出或公开插件状态。宿主 capability 与服务 capability 的交集只用于控制页面操作入口，不是后端的用户级 scope 授权；当前进程级服务令牌一旦通过，即可访问服务声明的全部 API。模型执行能力以后端 `dsh_models` 目录为准，宿主独有模型不会由前端直接调用。推荐使用 `provider/model` 作为目录 ID。

### 安装到 DSH Web Profile

`integrations/dsh_ecology_plugin/` 是实际的 Cordis 双端宿主插件。Node 端在 DSH 端口托管工作台并代理 EcologyRSI API；浏览器端向 DSH 侧栏注册“生态模型进化”，点击后在全屏覆盖层中打开工作台并完成 `plugin.ready` / `dsh.context` 握手。

```bash
ecologyrsi-dsh install-dsh-runtime --profile web
```

安装器会校验并安装当前版本的打包插件与六个不可变角色 preset ID。安装器还会在
`$DSH_HOME/profiles/web/cordis.patch.yml` 中维护下列完整原生运行时注入。不要手工缩减
`inject` 列表，否则 Planner 工具调用、子 Agent、会话持久化或跨代上下文会变成不可用：

```yaml
- insert:
    - id: ecologyrsi-evolution
      name: '@ecologyrsi/dsh-evolution-plugin'
      inject: [webServer, agents, sessions, tokenMeter, subagents, tools, sessionPersistence, sessionProjections, agentPresets, llm, web]
      config:
        staticRoot: '/path/to/EcologyRSI-DSH/plugins/ecology_evolution'
        backendOrigin: 'http://127.0.0.1:8777'
        # 与 Python 服务的 ECOLOGYRSI_SERVICE_TOKEN 保持一致；
        # 由 Node 代理服务端注入，不会进入 iframe URL 或前端代码。
        serviceToken: 'replace-with-runtime-token'
```

启动 Python 后端和 DSH 后，用户只访问 DSH 地址，例如 <http://127.0.0.1:8848/>。`8777` 是仅供 DSH 同源代理访问的回环后端，不再是用户入口。

如果 Python 服务设置了 `ECOLOGYRSI_SERVICE_TOKEN`，请在上面的宿主配置中设置同值
`serviceToken`，或在启动 DSH 的 Node 进程环境中设置同名变量。代理会覆盖浏览器侧
请求令牌；这样服务令牌不会落入插件 URL、静态 JavaScript 或浏览器存储。

API 请求使用 `Authorization: Bearer ...`。当前后端只比较进程级 `ECOLOGYRSI_SERVICE_TOKEN`：该令牌授予全部服务 API，尚未实现 task/run/session 级令牌签发与 scope 校验。因此不应把服务令牌交给不可信客户端；正式多用户部署需要由可信 DSH 代理签发并校验带 scope 的令牌。本地 HTTP 服务不提供 TLS，非本机部署必须置于受控的 HTTPS 反向代理之后，不能直接暴露到公网。

## HTTP API

sidecar 主前缀是 `/api`；浏览器通过 DSH 同源代理使用 `/api/ecology-evolution`。

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/health` | 无外部扫描的静态存活、版本和科学边界信息 |
| GET | `/api/plugin/ecology_evolution` | 插件能力清单 |
| GET | `/api/catalog` | 可用数据、策略、评测器和脱敏模型目录 |
| GET | `/api/datasets/{id}` | 数据说明、就绪状态和科学边界 |
| GET | `/api/datasets/{id}/samples?partition=training_fit&offset=0&limit=20` | 授权训练样本分页 |
| GET | `/api/runs` | 未归档运行投影列表 |
| GET | `/api/runs?include_archived=true` | 包含已归档运行的历史列表 |
| GET | `/api/runs/{id}` | 单个运行的脱敏投影 |
| GET | `/api/commands/{command_id}` | 查询异步 pause/cancel 等命令收据 |
| GET | `/api/runs/{id}/events?tail=500` 或 `?after={seq}` | 首次读取最近 1–500 条，后续按游标增量读取；不要组合两个参数以免跳过中间事件 |
| GET | `/api/runs/{id}/samples?candidate_id={candidate_id}&offset=0&limit=50` | 候选逐样本结果分页（每页最多 200 条） |
| POST | `/api/runs` | 创建并可选自动推进运行 |
| POST | `/api/runs/{id}/control` | `start/pause/resume/cancel/complete` |
| POST | `/api/runs/{id}/advance` | 推进 1 至 32 轮 |
| POST | `/api/runs/{id}/interventions` | 追加人工意见 |
| POST | `/api/runs/{id}/archive` | 归档终态运行，默认列表隐藏但保留证据 |
| POST | `/api/runs/{id}/restore` | 恢复已归档运行 |
| DELETE | `/api/runs/{id}` | 永久删除已归档终态运行，请求体必须精确确认 `confirm_run_id` |

逐样本接口在评测开始前返回 `status=pending`，执行中返回 `running`，正常封口后返回 `completed`；候选异常终止且已有部分结果时返回 `aborted`。行顺序由完整评测 cohort 的固定 `sample_index` 决定。成功行的 `raw_reward = |baseline - observed| - |predicted - observed|`，正值表示相对冻结评分基线降低了绝对误差；`normalized_reward = clip(raw_reward / training_fit_scale, -1, 1)` 用于跨目标学习信号。失败行在私有评分归档中保留固定最差惩罚，公开接口则返回 `prediction_source=failed_no_model_prediction`、`model_prediction_available=false`、`scoring_penalty_applied=true`，并将 `predicted`、误差和 reward 置空，不能被解释成模型输出。

创建模型自主温室运行的完整显式请求示例（预测模型、搜索策略和评测器由服务端绑定）：

```json
{
  "dataset_id": "agc_cucumber_2018",
  "episode_id": "agc_cucumber_2018:Croperators",
  "execution_protocol": "dsh_native_plugin_evolution@1",
  "optimization_protocol": "top2_adaptive_epoch@1",
  "optimization_schedule": {
    "schema_version": "ecologyrsi-dsh.top2-adaptive-epoch-schedule/2",
    "screening_origin_count": 64,
    "finalist_count": 2,
    "formal_origin_count_per_finalist": 500,
    "local_batch_origin_count": 50,
    "max_local_edits_per_batch": 2,
    "selection_holdout_origin_count": 169,
    "local_evaluation_mode": "paired_champion_challenger"
  },
  "strategy_model_id": "newapi-glm52-policy",
  "review_model_id": "newapi-glm52-judge",
  "rounds": 5,
  "candidate_concurrency": 4,
  "sample_agent_batch_size": 9,
  "sample_concurrency": 64,
  "budget": {
    "max_generations": 5,
    "candidates_per_generation": 4,
    "max_candidates": 20
  },
  "model_workflow": "research_compile_evolve@1",
  "autonomous_mode": true,
  "seed_policy": "fixed",
  "auto_advance": true,
  "idempotency_key": "greenhouse-run-001"
}
```

`ecologyrsi-dsh.top2-adaptive-epoch-schedule/1` 与 `prequential` 仅用于显式读取旧版运行和旧账本；新建运行固定使用上面的 v2 配对合同。

自主运行使用 `auto_advance: true` 进入连续模式：服务端完成一轮后自动排入下一轮，直到达到轮数/候选预算、暂停、取消或失败；页面只轮询真实阶段事件，不需要反复点击“下一轮”。服务默认使用 4 个有界 worker 推进不同运行；同一运行始终只能持有一个世代租约，每执行一代就回到队尾。`ECOLOGYRSI_AUTO_PROGRESS_WORKERS` 可显式配置为 1–8。不同运行和同一运行内的候选可以并行；同一 provider 的 DSH stage 统一经过全局 FIFO 准入，物理在飞上限为 128，失败冷却对该 provider 的全部运行生效。服务重启后会从 SQLite 恢复未归档的连续运行，并在每轮开始前重新校验冻结的数据、预测器、策略、评测器和远程模型绑定；绑定发生漂移时以 `frozen_runtime_binding_drift` 停止运行并提示新建。

默认界面固定使用每轮 4 个候选，5 轮对应 20 个候选总预算。每个提案都记录 `proposal_source`，投影分别统计远程成功、宿主种子和显式宿主回退，不把“未调用 API”显示成“调用完成”。

新建运行不再接受含义含混的 `samples_per_update`；所有工作量都由 `optimization_schedule` 以 prediction origins 表示。外层固定先对 4 个候选各筛选 64 origins，再冻结 Top 2。每个 finalist lane 的 500 个 formal unique-origin 槽位默认分成 10 个 50-origin batch：batch 0 只评测并冻结初始冠军；batch 1–9 从当前 selected champion 生成一个有界挑战者，并让两臂在同一 cohort 上分别执行。通过门禁的挑战者成为新冠军，未改善的挑战者被拒且下一次仍从原冠军生成；末批完成比较后不会再创建一个无法验证的 child。epoch 结束后，两条 lane 的耐久冠军与上一冠军在同一批 169-origin holdout 上重新评测。169 是 169 次完整预测；默认任务内部产生 `169 × 9 = 1,521` 个评分单元，但并不是 1,521 次独立模型请求。

默认 formal 执行上限为 `2 × [50 + 2 × (10 - 1) × 50] = 1,900` candidate-origin occurrences；默认单轮预算为 `4 × 64 + 1,900 + 3 × 169 = 2,663 candidate-origin occurrences`，即 `2,663 × 9 = 23,967` 个评分单元。五代所需的唯一源起点仍为 `500 + 5 × (64 + 169) = 1,665`，因为两条 lane 与 paired arms 复用同一组 formal unique-origin 槽位。候选并发默认 4；逐样本并发默认 64、可配置 1–128；两条 finalist lane 可并行推进但共享 run 级逐样本并发，同一 provider 的物理在飞上限为 128。每轮按预测起点构造候选无关的确定性窗口；可用起点少于计划 occurrence 时会循环复用，但同一比较阶段仍冻结一致的 cohort 身份。

每个 origin 调用一次 Planner，由 Planner 在 DSH 子会话内选择并调用登记的联合向量预测工具；只有不确定或失败时才调用 Critic。Host 将 9 条评分记录原子持久化，候选完成后再执行聚合反思。岭回归可以完整扫描 `training_fit` 拟合参数，但只能作为 Planner 主动调用的注册工具，不能替代智能体决策阶段。

单个 origin 耗尽重试预算不会终止整个候选，但评分后处理保证失败行相对冻结强基线的 reward 不大于 0，不能通过失败或丢样本提高分数。同一 batch 内的配对分数用于 lane champion 选择，不同 batch 分数不直接比较；被拒挑战者只作为下一次受限提案的证据，不会成为父版本。轮末 F1/F2/incumbent 使用同一 holdout 的 centered max-T 选择门禁，并同时要求 0.005 实用差异和一致的评分合同。跨代反思按完整 `behavior_digest` 防止失败行为的精确重放；相同科学参数但不同 agent 程序仍可继续探索。`execution_diagnostics` 分别给出物理分区行数、selected/deferred 评分单元和 origins、eligible/used/skipped 目标、累计候选工作量、拟合 pass、提案来源和轮次耗时。正式 `EvaluationRecorded` 到达前，页面只把可验证的 checkpoint 聚合标为部分进度。

创建合同以 `dataset_id` 为首要输入；策略模型 API、独立评审模型 API 和轮数是另外三个用户输入。Web 界面会提交目录已绑定的 `episode_id`；原始 API 省略时，服务端确定性选择首个可优化 episode。服务端根据数据集目录记录的 `domain_id`、适配器、许可和模型能力推导 `domain_pack` / `research_domain`，再由模型研究结果冻结预测模型、策略和评测器；这些内部绑定不由前端用户手工覆盖。

提交人工意见前必须暂停运行。支持 `guidance`、`parameter_override`、`constraint` 和 `parent_selection`。`guidance` 只有在唯一识别一个允许参数和一个增减方向时才按固定步长应用；`constraint` 只接受唯一的 `<=`/`>=` 数值边界，并在参数覆盖之后由宿主强制执行。歧义、冲突、否定或越出宿主范围的输入会被消费但明确标记为“仅记录（未执行）”。这些操作都不会改写固定评测器、数据分区或门禁规则。

```json
{
  "kind": "parameter_override",
  "message": "将历史步数固定为 6。",
  "created_by": "研究员甲",
  "parameter_overrides": {"history_steps": 6},
  "idempotency_key": "human-input-001"
}
```

## 测试与发布

```bash
make test
find plugins/ecology_evolution -name '*.js' -exec node --check {} \;
node plugins/ecology_evolution/test/smoke.mjs
make verify
```

发布前以本机重新执行上述命令的结果为准；README 不固化会随测试增删变化的断言数量。

`make test` 默认优先使用 `uv` 发现的本机 Python 3.10+，未安装 `uv`
时才回退到项目 `.venv` 或 `python3`；也可以通过 `PYTHON=/path/to/python`
显式指定。在 Apple Silicon 上不要用 x86_64 Anaconda 或 x86_64 虚拟环境的
`pytest` 运行本项目，Rosetta 退出异常会留下无法由普通 `kill`
回收的 `UE` 进程。

在真实 AGC 数据已就绪的机器上，额外启用真实数据集成用例：

```bash
ECOLOGYRSI_TEST_REAL_DATA=1 make test
```

需要验证真实的逐样本智能体工具链时，运行受控验收脚本。脚本从指定账本选择最近一条
同时冻结 planner 与 critic 的 `RunCreated` 绑定；`--planner` / `--critic` 只用于可选断言。脚本不做
API 健康预检，直接按 900 秒请求窗口和 4 次传输重试执行。岭回归使用完整
`training_fit` 拟合，但只有按 `target × horizon` 固定时间分位抽取的小 cohort
进入远程路由；报告固定标记为不可晋级、不可生成训练资产、不可作为科学得分。

当前原生执行协议由 DSH Web Profile 中的 Cordis 插件承载。角色 Agent Session、
上下文压缩、模型路由和结构化子智能体由 DSH 执行；逐 origin `sample.plan`
通过直接一次性子 Agent 执行，并在 Host 接受结构化结果后才计为远端完成；
Python sidecar 只保留科学状态机、评测、幂等结果账本和治理边界。目录在运行时
已绑定时返回 `harness_execution=dsh_native_agent_runtime` 与
`official_harness_agent_loop=true`；当前交付只验收这一原生运行时路径。

```bash
make release
RELEASE_PYTHON="$(uv python find --no-project --system '>=3.10')"
"$RELEASE_PYTHON" -B scripts/real_api_agent_tool_acceptance.py \
  --db /tmp/ecologyrsi-dsh-dsh-adapter.sqlite3 \
  --samples-per-task 1 \
  --dist-dir dist \
  --output dist/ecologyrsi_dsh-0.3.55-real-api-agent-tool-acceptance.json
```

构建 wheel、sdist 和完整交付包需要 `uv`：

```bash
make release
```

命令行 toy 演示、诊断、重放和导出仍可使用：

```bash
PYTHONPATH=src python -m ecologyrsi_dsh demo --db /tmp/ecologyrsi-demo.sqlite3
PYTHONPATH=src python -m ecologyrsi_dsh doctor --db /tmp/ecologyrsi-demo.sqlite3
PYTHONPATH=src python -m ecologyrsi_dsh summary run:demo --db /tmp/ecologyrsi-demo.sqlite3
```

## 科学与交付边界

- AGC 2018/2019 是历史观测日志，本系统当前只能支持离线回放、1/6/24 小时时间前向预测和支持域分析，不能把预测差异解释为控制动作的因果效应或反事实结果。
- 当前“训练”是有界滚动残差偏差拟合或外生变量岭回归残差拟合，不是通用神经网络训练，也不是任意模型代码搜索；进化训练资产也不是已经获准使用的正式 SFT/DPO 数据。
- 对真实 AGC 数据，插件提交目录冻结的 `episode_id`；当前尚未实现跨 episode、跨团队联合评测。
- `development`、`gate`、外部留出、隐藏和最终评测没有进入本地搜索保留闭环；插件也没有正式发布、回滚或实体控制权限。
- DSH 接入包括本地 Web Profile Cordis 宿主插件、受限角色 preset、Session/压缩、
  结构化子智能体、同源静态托管/API 代理与 Python 科学状态 sidecar；
  仍未完成官方 OAuth、插件签名或市场发布。
- 当前静态资源 CSP 只允许同源嵌入；跨域 DSH 宿主需要同源代理或经过审核的 CSP、origin 和令牌适配。
- 单进程锁和 SQLite 适用于本地交付与研究验证，不是多租户、高并发生产架构。

发布前的人工验收项与安全边界见 `RELEASE-CHECKLIST.md`。

## 0.3.55 交付更新

- 自适应进度接口同时提供当前轮 `estimated_remaining_seconds` 与跨全部进化轮次的 `run_estimated_remaining_seconds`；页面优先显示“预计全程剩余”，并在有明显差异时附带当前阶段 ETA，避免多轮运行只按当前轮剩余量低估总耗时。
- 吞吐率继续只使用 Host 已持久化的完整 prediction-origin 结算边界；进程恢复后按候选的恢复边界重置速率窗口，避免把停机时间算进 ETA。

## 0.3.52 交付更新

- 实测故障来自上游 RPM 窗口，而不是“并发 64”参数本身：每个 planner 通常需要 Skill、预测工具、结构化输出三次模型 turn。新版保留宿主并发默认 64、最高 128，同时默认每 3 秒启动一个新 child，避免在首个反馈前用完 RPM。这是启动速率保护，不是把并发上限改成 8。
- 终态 `RATE_LIMIT`/HTTP 429 会优先采用 DSH 的 `providerRetryAfterMs`，并安全兼容消息中的 `retry_after`；新请求会等到冷却结束，且 RPM 限流不再被误当成物理并发不足而把 128 降到 4。
- 结构化结果判定会最多等待 2 秒的完整 Session 终态。若第一次 `structured_output` 精确返回 `INVALID_ARGS`，即使旧 child 稍后又调用成功，也会按 exactly-once 合同开启一个干净的有界重试；授权拒绝、未知工具和重用 call ID 仍严格终止。
- `sample.plan` 的原生输出上限由 2,048 提高到 4,096 Token，解决预测工具已成功但最终结构化输出被截断的少数样本。
- 页面不再把宿主账本推算的 `queued_batches` 冒充为精确的“Provider 等待”；仍展示可核验的宿主并发、宿主等待和当前阶段待提交。
- 0.3.51 的 64-origin Host admission、Top 2 双通道、500-origin 自适应 epoch、最终 revision 继承、耐久进度与页面分层监控保持不变。

## 0.3.51 交付更新

- 配置为 64 的逐样本并发会立即作为本运行的 Host admission 上限进入 DSH，不再被隐藏的 8 请求冷启动窗口长期限速；同一 provider 仍共享 128 个物理在飞上限，并在观察到拥塞或模型失败后自适应降载。
- 候选并发允许时，两条 Top 2 finalist lane 会在同一调度轮各推进一个 50-origin batch，并严格共享运行级 64 请求预算；局部 revision 修改仍串行执行，并在等待后重新检查暂停或取消状态。
- DSH 子任务完成后会等待 Session 投影与结构化结果同步，消除偶发的持久化竞态；明确的结构化输出参数错误按缺失结果在新的有界子任务中重试，日志只保留 sidecar 脱敏信息。
- 轮末排名、入选原因和下一轮父方案改为直接读取冻结的 F1/F2/incumbent 三臂比较门禁，不再依赖候选槽位；聚合反思包含最终 revision、9 个目标/时距门禁、失败摘要和完整 10-batch 局部修改链。
- 下一轮调研、研究综合、候选批次和提案会共同继承上一轮冻结的最终有效 revision，而不是候选初始 R0；即使首轮保留的是同轮 `screened_out` incumbent，也不会错误回退到留出门禁失败的 finalist。
- 页面把全轮、当前阶段、Host 准入、Host 等待、DSH 未终结子任务和待提交工作分层显示；只有精确 gate 快照才标注为 Provider 等待。候选提前终止后未执行的 origins 明确计为“跳过”，`screened_out` 作为终态处理，空微批分数保持“等待评测”，不再显示为 0 或永久排队。
- 删除前端已经失去调用方的旧参数归一化分支；创建、预算摘要和服务端共用当前严格参数合同，不保留旧运行数据兼容路径。

## 0.3.50 交付更新

- `sample.plan` 不再经过当前 DSH 版本不支持输出上限参数的 Workflow bridge，而是直接启动一次性结构化子 Agent；2,048-token 上限通过 DSH 支持的 `agentOptions.maxTokens` 下发，provider/model 继续从冻结角色宿主继承。同步删除旧 Workflow 执行器、脚本模板、双重取消路径和对应冗余测试。
- 预测工具回执只表示宿主工具已经产出向量，只有 `DshStructuredResultAccepted` 才算远端 origin 完成。候选 heartbeat 负责 Host 已结算的成功/失败拆分，避免结构化失败时总进度长期停在 0 或把失败误报为成功。
- 私有账本仍保留固定最差惩罚用于确定性评分门禁，浏览 API、候选预览、实时执行和训练轨迹一律把它显示为“未产生有效预测”，并将预测、误差和 reward 置空。失败请求数、子模型终止错误数和失败 origin 数分开显示，重试请求不会伪装成失败样本。
- 局部编辑现在由 Host 对完整操作包做顺序无关的规范化签名；同一不可变父 revision 上已经被拒绝的完全相同操作包会保留提案审计记录，但直接以 `duplicate_recent_rejected_bundle` 拒绝，不再创建重复子 revision。父 revision 改变后不会被误拦截。
- 当前样本执行模式更名为 `dsh_native_agent`；本版本不读取旧模式名或旧 preset 包，运行数据合同无需向前兼容。

## 0.3.49 交付更新

- screening、每个 50-origin 正式微批和 incumbent holdout 都会在首次请求前打开 revision-scoped checkpoint，逐 origin 持久化行、进度和用量，并把阶段科学结果与全量样本完成事件原子封口。暂停或进程重启只重放未结算 origin，迟到的进度/用量不能写入已完成范围。
- holdout checkpoint 严格绑定冻结的 arm、candidate revision 和 cohort：第 0 代可正确复评 `SCREENED_OUT` incumbent，后续代可复评历史冠军，但不会放宽 screening/formal 的同代、`SPAWNED` 边界。
- DSH-native 每个 sample 子模型调用的输出上限冻结为 2,048 tokens，并同时传递到直接 child 和 Workflow child。新的 routing manifest v2 只远端提供目标、时距、边界、digest 和 Host 异常摘要，不重复发送原始历史/特征行；测试载荷缩至原来的 32.8%，非原生 ModelGateway 也按 v2 直接解析 variant，不再要求已删除的 defaults。
- `sample.plan` 使用跨 wave 稳定的输出 schema，便于 provider 复用前缀缓存；Host 仍严格校验 wave digest、样本 ID 全集、唯一预测工具收据和每个预测的唯一决策。
- 远端进度只把达到任务实际 `prediction_cells_per_origin` 的完整 `sample.plan` 主波计为一个预测时点；更小的 repair wave 单独展示完成/在飞数与主预测/修复请求数，不再虚增 origin 进度或待结算数。

## 0.3.48 交付更新

- 逐 origin 的运行控制不再重放完整事件流，而是通过只包含生命周期事件的 SQLite covering index 执行 O(1) 状态查询；64 路逐样本并发不会再因重复解析整条 run 历史挤占 sidecar，页面监控和控制请求也不再与科学执行争用同一条重放热路径。
- 自动推进的诊断接口保持纯读，调度器自身以低频维护循环修复失效的 retry timer 和仍为 `RUNNING`、却意外失去队列 ownership 的运行；工作单元只有在追加式账本序号真实前进后才算成功，避免“返回成功但没有持久进度”的空转。
- 未分类 Host 异常会在当前检查点安全暂停，不再按进程内猜测无限重放；暂停或失败前先关闭 DSH 写入 admission，并以可恢复的原生 quiescence 状态同步 pause/cancel。短暂的 DSH 控制失败会保留恢复证据并重试，服务重启后也会核对 Host 与 DSH 状态，避免一侧已终止、另一侧仍持续运行。
- holdout 每个 F1/F2/incumbent arm 在发出第一个请求前写入独立的开始边界，页面能够显示当前 arm、已完成 origin、在飞、待提交与待结算数量；崩溃恢复仍要求完全相同的 revision、cohort 和 artifact 绑定，不会把其他阶段结果误作留出证据。
- adaptive batch 新增严格 origin 安全门：评分单元覆盖率和“9 个目标/时距结果全部成功”的完整 origin 成功率分别计算；严格 agent chain 不完整或 origin 成功率低于冻结门槛时禁止继续修改，并在已有父 revision 时执行安全回退。完整 pipeline 的选择保留在轮级 4 候选搜索中，不再作为 50-origin batch 内的局部编辑，从而保持外层 Top 2 逻辑不变、局部 revision 只做有界小改动。
- 进化页同时展示 `origin 成功率` 与 `评分单元覆盖率`，并明确区分 Host 已落盘、远端已完成、在飞和排队状态；空值不会再被显示成 0%，不同 cohort 的 batch 分数仍只用于诊断，不绘制伪趋势。
- `FormalBatchEvaluated` 只保存决策所需的聚合指标、科学/数据校验值和一份规范化压缩轨迹；删除重复的展开执行记录与浏览器预览。逐样本 checkpoint 事件与该压缩归档共同保留可重放、可复现校验链和失败审计能力。

## 0.3.47 交付更新

- 局部编辑现在携带当前 revision 的真实 pipeline、参数与最近 8 次编辑结果；同值/no-op 会被宿主拒绝，约束导致的样本失败会回退到父 revision，provider/覆盖率瞬时不足则保留当前 revision 并停止继续修改，避免把基础设施故障误判为模型退化。
- 轮末 F1/F2/incumbent 共用同一 max-T 多重比较族与实际改善阈值；holdout 恢复必须同时匹配冻结 revision、cohort scope 和 artifact digest，已落盘的 canonical 结果可以补齐缺失事件，不会把其他阶段结果当作留出完成。
- 自动推进失败现在公开稳定的阶段、候选、批次与工作单元位置；命令收据、知识检索终态竞态和删除后同 ID 重建均按追加式账本恢复，不再把无关终态误归到待处理操作。
- 高频页面轮询改用有界 monitor 投影和增量事件尾；轻量状态先提交，候选详情仅在结构变化或低频刷新时补齐，详情/事件瞬时失败不会覆盖新的暂停、失败或终态，隐藏标签页自动降频。
- DSH Session provider 回执累计用量会持续显示，但不是完整、原子的计费账本，因此 DSH-native 运行不接受 `token_limit`，也不再暗中附加 1 亿 token 上限；旧的 sample-gateway 模式仍保留其独立的硬预算合同。
- 运行进度以 prediction origin 为唯一主口径，显示实时 admission 上限、在飞、等待 provider、待提交、待结算、滚动吞吐和 ETA；自适应轨迹区分“已应用待验证”“宿主拒绝”“安全回退”和“安全保持”。

## 0.3.46 交付更新（历史）

- 新建严格运行使用 `top2_adaptive_epoch@1` 与 `dsh-strict-origin-bundle@4`。外层 4 候选筛选与 Top 2 选择不变；每个 finalist 内部改为 500-origin adaptive epoch，默认 `10 × 50`，每批最多 2 处局部修改，最后由 F1/F2/incumbent 在同一 169-origin holdout 上比较。
- 参数页以 prediction origins 为主口径，同时给出评分单元换算和单轮/全程总预算；`samples_per_update` 已从新建合同移除。逐样本并发默认 64、可配置 1–128，provider 全局物理在飞上限为 128。
- 数据容量不足不再阻止创建：冻结 population 按 occurrence 循环复用，并保留可复现窗口身份。跨 cohort 的原始得分仍不得直接作晋升比较。
- 进化页同时展示全程进度、epoch 进度、当前 DSH 子阶段、真实在飞/排队状态、心跳、候选 revision 和微批优化轨迹；事件接口使用最多 500 条的增量 tail，避免长运行反复下载完整账本。
- `/health` 改为不扫描模型目录或账本的静态存活检查；DSH 身份解析使用有界缓存，HTTP backlog 为 256，降低 64/128 并发时健康检查和页面轮询被堵塞的概率。
- 并发候选结算失败时，自动推进以全部已入场兄弟任务完成后的最新账本序号重试；若运行仍为“运行中”而调度器意外空闲，状态轮询会把它重新入队。
- 逐 origin Planner/条件 Critic 以及候选聚合反思均使用有界端到端运行时限；取消和暂停在阶段边界安全生效，不把超时或失败样本计为成功。
- 当时 README 的六张界面图使用 0.3.46 显式演示模式重拍；当前截图清单和复核说明见 [`docs/screenshots/README.md`](docs/screenshots/README.md)。
