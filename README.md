# EcologyRSI-DSH · ecogym

**ecogym 是一个让生态研究智能体自我改进的训练场。** 给一组生态预测任务和数据，DSH 研究智能体按某一版 Harness（指令、技能、代码库、经验、DSH 设置）独立解每个任务：读可见数据、查文献、自己写代码和做实验，最后提交一个预测程序；宿主在它看不到的隐藏时期上逐时刻运行这个程序并打分。评审智能体根据成绩和完整轨迹提出新的 Harness 候选，候选只有在多任务配对比较中可信地变好才被继承；被采纳的版本还要在留出任务和封存时期上各验证一次。过程中积累两类资产：可复用的知识（技能、代码库、经验、专家意见）和完整的任务轨迹（可以导出为多轮 SFT 数据和偏好对）。

DSH 内核不改：进化的只是 DSH 能加载的内容；打分由确定性的宿主代码完成，不交给 LLM。ecogym 本身是 DSH 的一个插件：启动 DSH 就启动了 ecogym，侧栏「生态进化」面板里查看和发起进化，面板的「介绍」页讲整个流程。

> ecogym 0.2.0 · Python 3.11+ · 依赖 numpy · 智能体回合需要 macOS（沙箱）、DeepSeek Harness（`dsh`，已在 0.1.5-rc.2 上验证）和一个可用的模型提供方：DeepSeek 官方接口，或 DSH 设置里任一 `openai-completions` 提供方

[交付内容](#交付内容) · [快速开始](#快速开始) · [在 DSH 中使用](#在-dsh-中使用) · [内循环与外循环](#内循环与外循环) · [评测规则](#评测规则) · [实测](#当前参考成绩与实测) · [训练数据](#训练数据) · [方法竞技](#方法竞技参数级基线) · [代码结构](#代码结构) · [已知限制](#已知限制) · [设计文档](docs/evolution.md) · [前后端架构](docs/architecture.md) · [新增领域](docs/new-domain.md)

## 交付内容

| 部分 | 内容 | 位置 |
|---|---|---|
| Python 包 `ecogym` | 命令行 `ecogym`；JSON 服务；内循环（沙箱回合、模型网关与提供方路由、裁判、续跑）；外循环（评审会话、多任务配对比较、采纳、独立验证）；训练数据导出；方法竞技 | `src/ecogym/` |
| DSH 插件 `@ecogym/dsh-plugin` | 随 DSH 启动和停止服务、同源代理、「生态进化」面板（进化、回合、专家、训练数据、任务、介绍） | `src/ecogym/web/plugin/` |
| 温室领域包 | AGC 2018 黄瓜与 2019 番茄的数据集规格、两张任务卡（十个隔间）、两个任务组（4 或 8 个训练隔间）、专家求解器、领域知识 | `src/ecogym/domains/greenhouse/` |
| 起点 Harness | 研究流程说明、三个技能（实验、特征与模型、从文献到方法）、朴素基线模板与 `ecotk` 代码库 | `src/ecogym/harness/seed/` |
| 文档 | 本文、[设计](docs/evolution.md)（含 2026-10-29 的实测和评测方案检验）、[前后端架构](docs/architecture.md)、[新增领域](docs/new-domain.md)、[变更记录](CHANGELOG.md) | `docs/` |

验证情况：`make test` 的 97 个测试和 `make lint` 通过；2026-10-29 从面板用 `pjlab/deepseek-v4.1-flash` 完整跑了一次进化（4 个训练任务、1 代 × 2 个候选、32 个回合，见[实测](#当前参考成绩与实测)）；面板的每个页面在浏览器里检查过。

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e .
ecogym setup          # 智能体运行时：独立的 Python 3.12 环境，装好 numpy、pandas、scipy、scikit-learn、statsmodels
ecogym dsh install    # 把 ecogym 装进 DSH 的 web profile（见下节）
dsh web               # 启动 DSH，ecogym 随之启动；侧栏点「生态进化」，先看「介绍」
```

数据放在数据根目录 `$ECOGYM_DATA_ROOT`（默认 `~/.ecogym/data`）下。自带的温室领域需要 Autonomous Greenhouse Challenge 数据，解压到 `greenhouse/agc_cucumber_2018/`（[2018 黄瓜](https://doi.org/10.4121/uuid:e4987a7b-04dd-4c89-9b18-883aad30ba9a)）和 `greenhouse/agc_tomato_2019/`（[2019 番茄](https://doi.org/10.4121/uuid:88d22c60-21b3-4ea8-90db-20249a5be2a7)），每个团队一个子目录。

模型写成 `<提供方>/<模型>`。DeepSeek 官方接口（`deepseek-official`，裸的 DeepSeek 模型 id 也指它）的密钥从环境变量 `DEEPSEEK_API_KEY`、`$ECOGYM_CREDENTIALS_FILE` 中的 `DEEPSEEK_API_KEY: …` 行或 DSH 的凭据文件读取；DSH 设置（`~/.dsh/settings.yaml` 的 `llm-pi-ai.providers`）里的其他提供方用各自 `apiKeyEnv` 指向的密钥。真实密钥只留在本机：智能体只拿到本机模型网关的一次性令牌，网关只转发这个会话自己的模型调用和网页搜索。网页搜索走 DeepSeek 的接口，没有 DeepSeek 密钥时关闭。

```bash
ecogym suites                                              # 任务组：训练任务与留出任务、门槛
ecogym models                                              # 可用的模型：DeepSeek 与 DSH 设置里的提供方，及不能用的原因
ecogym episode run greenhouse.agc2018.cucumber.climate@1   # 一个回合：智能体解题，裁判在隐藏期上打分
ecogym evolve run --suite greenhouse.climate@1 --repeats 2 --model pjlab/deepseek-v4.1-flash \
                  --evolver-model pjlab/deepseek-v4.1-flash  # 进化 Harness（默认 3 代 × 2 个候选，20 分钟的回合）
ecogym evolve show evo:<id>                                # 各代候选、提案、比较和采纳
ecogym evolve verify evo:<id> --holdout                    # 留出任务上验证一次
ecogym evolve verify evo:<id> --sealed validation          # 封存时期上验证一次（之后才能 final_test）
ecogym export --out data/sft                               # SFT 线程、偏好对、经验库、回合索引
ecogym expert list                                         # 智能体留给专家的问题；answer / note 回答或给提示
```

这些命令和 DSH 面板做的是同一件事、写同一个账本；面板里能做的（新建、停止、继续、验证、回答专家问题、导出）命令行都能做。开始运行前，宿主检查每个模型的提供方和密钥，并发一个很小的请求：设置里列出但密钥无权调用的模型不能开始运行。

所有运行、回合、曝光和专家意见写入同一个账本 `$ECOGYM_HOME/ledger.sqlite3`（默认 `~/.ecogym`），轨迹写在 `$ECOGYM_HOME/trajectories/`，Harness 版本按内容摘要存放在 `$ECOGYM_HOME/harnesses/`。`--ledger` 可指定其他账本。

## 在 DSH 中使用

ecogym 是 DeepSeek Harness 的插件 `@ecogym/dsh-plugin`（源码在 `src/ecogym/web/plugin/`）。安装一次即可：

```bash
ecogym dsh install                      # 默认装进 web profile；--profile 换 profile，--home 指定 ECOGYM_HOME
ecogym dsh status                       # 装在哪里、DSH 会用哪个 Python 和哪个账本启动服务
dsh web                                 # 或 dsh --profile web；ecogym 服务随 DSH 启动和退出
```

`install` 做两件事：用 `dsh plugin --profile web add link:<插件目录>` 把插件加为 profile 的一层（插件包里的 `cordis.patch.yml` 插入插件行，`dsh plugin --profile web remove @ecogym/dsh-plugin` 会连同这一层一起拿掉）；把启动服务用的 Python、`ECOGYM_HOME` 和当前 shell 里的 `ECOGYM_DATA_ROOT`、`ECOGYM_RUNTIME_PYTHON`、`ECOGYM_DSH_BIN` 写进 `~/.ecogym/dsh-plugin.json`（`--env KEY=VALUE` 可再加）。插件以链接方式安装，改动直接生效：改了 `lib/client.js` 刷新页面即可；改了 Python 代码要重启后端（重启 DSH，或带 `X-Ecogym: 1` 头 POST `/api/ecogym/_restart`）；改了 `lib/index.js` 要重启 DSH。早期 EcologyRSI 插件写在 `cordis.patch.yml` 里的托管区块会被移除并备份，它的工作台依赖的服务已不存在（`--keep-legacy` 保留）。`ecogym dsh uninstall` 移除插件和设置文件。

运行方式：

- DSH 启动时，插件在 DSH 进程里以子进程启动 `python -m ecogym serve --port 0 --exit-with-stdin`，服务令牌只留在 DSH 进程里。浏览器经同源的 `/api/ecogym/*` 访问，请求先过 DSH 自己的 `/api` 检查（Host／Origin 栅栏和浏览器登录 cookie），再由插件带上令牌转发。服务崩溃会自动重启，面板顶部显示后端状态和日志。
- DSH 退出（或子进程管道断开）时，服务先停下正在运行的智能体再退出：进化运行停在最后完成的代，记为“已停止”，在面板里点「继续运行」即从那里接着跑，中断的那一代重新运行。
- 进化只在用户点「新建进化」后开始：选任务组和起点 Harness，设代数、候选数、每个版本每个任务的回合数、每个回合的时间和 token，在「模型（API）」里分别选内循环和评审智能体的模型，表单给出回合数、时长和 token 的上限估计。下拉列表按提供方分组，包括 DeepSeek 官方接口和 DSH 设置里的全部提供方；缺少密钥或接口不受支持的提供方显示原因、不能选，选中模型时面板就测试它能否调用。
- ecogym 的智能体会话各用自己的 DSH 主目录，回合结束后存进轨迹，不出现在 DSH 网页端的会话列表里。

面板（DSH 主栏，跟随 DSH 的深浅色主题）分六部分：

- **进化**：运行列表；运行页有进度和用量、正在运行的回合、训练任务上的隐藏期成绩表（含几个回合的预报平均后的成绩和宿主参考方法）、各代候选的增益与区间、独立验证（留出任务、validation、final_test 的汇总表与逐任务成绩）、各代详情（评审总结和笔记、候选的摘要、提案、逐文件 diff、逐任务增益、每个回合）、在任 Harness 的指令、经验、DSH 设置及与起点的差异、专家建议、运行配置。可以本代结束后停止、立即中止、继续运行和验证。
- **回合**：隐藏期逐格技能、本地检验与提交的时间线、笔记、专家问题、最终回复和工具使用。
- **专家**：待回答的问题与回答、主动提示（全部任务、数据集或任务）。
- **训练数据**：按过滤条件在后台导出 SFT 线程和偏好对（写到账本旁的 `exports/`），查看计数、文件和第一条样本。
- **任务**：任务组的训练与留出任务、采纳门槛，各任务的数据集、目标和时距。
- **介绍**：项目的整体流程；一个训练任务的实际时间分段和谁能看到哪一段；三层评测与采纳门槛；一个回合里有什么；各页面的用途；前后端的分层和服务的全部接口；术语。

不经 DSH 时，`ecogym serve` 提供同样的 JSON API：接口表是 `src/ecogym/service.py` 的 `ROUTES`，`GET /api/routes` 返回同一张表；分层、请求路径、状态放在哪里和每个页面用哪些接口见 [docs/architecture.md](docs/architecture.md)。绑定非回环地址必须设置 `--token`，所有 POST 需要 `X-Ecogym: 1` 头。

## 内循环与外循环

| 概念 | 含义 | 位置 |
|---|---|---|
| Harness | 智能体带进每个任务的全部内容：`instructions.md`、`skills/`、`template/`（起始预测程序与 `ecotk/` 代码库）、`memory/lessons.md`、`harness.toml` 的 `[agent]`（模型、推理强度、网页搜索、子智能体深度、并行工具调用）。内容决定版本摘要。 | `harness/`，起点 `harness/seed/` |
| 回合 Episode | 一个 DSH 智能体在 macOS 沙箱里按一版 Harness 独立解一个任务；工具 `task_info`、`evaluate_local`、`submit`、`note`、`ask_expert`；智能体在剩余时间超过四分之一时就结束的，宿主开新会话请它继续（最多 3 段，一段续跑什么都没做就停）；裁判在隐藏期上逐时刻运行最后一次被接受的提交。 | `episodes/` |
| 任务组 Suite | 训练任务（比较 Harness 版本）与留出任务（只用于最终验证），以及采纳门槛。`<任务卡>:<时段>` 让一张任务卡用在数据集的每个站点或团队上。自带 `greenhouse.climate@1`（训练 4 个隔间、留出 6 个）和 `greenhouse.wide.climate@1`（训练 8 个、留出 2 个，检验力更高，每代回合数加倍）。 | `task/suite.py`，`domains/*/suites/` |
| 进化运行 | 第 0 代跑起点版本；之后每代：评审会话写候选 → 候选与在任版本同时在每个训练任务上运行 → 多任务配对比较 → 采纳。全部状态可从账本重算，`--resume` 从最后一个完成的代继续。一代的全部回合都没有得到模型回复时运行停下（可继续）。 | `evolution/loop.py` |
| 评审会话 | 一个 DSH 智能体读证据（各回合的成绩、时间线、笔记、代码、最终回复，几个回合的预报平均后的成绩，以往候选的改动与结果）并编写候选；宿主独立检查每个候选，包括候选自己设定的模型能否调用。 | `evolution/evolver.py`、`evidence.py` |
| 独立验证 | 起点与最终版本在留出任务上各跑一个回合；它们的预测程序在封存时期上各评一次。 | `evolution/verify.py` |

外循环的细节、统计方法、数据格式和成本见 [docs/evolution.md](docs/evolution.md)。

## 评测规则

- **分区**：每个时段按时间切成 `calibration_fit`、`calibration_uq`、`model_selection`、`validation`、`final_test`，相邻分区之间有隔离期。日历切点在数据集全部非外部时段上统一计算。
- **回合的信息集**：智能体只看到 `model_selection` 开始之前的行（`calibration_fit`、`calibration_uq` 及其间隔），本地检验在 `calibration_uq` 上按裁判协议运行；隐藏期是 `model_selection`。裁判用隐藏期之前的全部行调用 `fit`，然后逐个时间步送入新观测，在每个计分起点调用 `forecast`，程序看不到起点之后的任何值。程序在无网络、只能读自己目录的沙箱里运行。
- **技能**：每格对比在 `calibration_fit` 上选定的朴素基线（持续性或季节值）：`skill = (RMSE_基线 − RMSE) / max(RMSE, RMSE_基线)`。缺失预报按覆盖率扣分（每格最低 −1），预报裁剪到目标的物理范围，总分是逐格加权平均。
- **Harness 的采纳门槛**（任务组的 `[gates]`）：候选 − 在任的平均增益大于 `min_delta`（默认 0.005）；任务组区间的下限大于 0（3 个及以上训练任务时是各任务增益的 t 区间，更少时依次重抽任务、回合和天；双侧 95% 区间只看下限，即单侧 2.5% 的检验，一代有多个候选时按 Bonferroni 调整）；没有任务的损失超过 `task_cap`（默认 0.03）且它自己的 95% 区间整体低于 0，这一条在两个版本在该任务上都至少有 2 个回合时才判断。在任版本的成绩只用它作为在任时、在同一份专家意见下跑的回合；没有得到任何模型回复的回合算服务故障，重试一次，不计入比较。建议每个版本每个任务跑 2 个回合（`--repeats 2`）。
- **独立验证**用同样的平均增益和区间；每个版本在每个任务上只有一个结果，所以不做单任务否决，逐任务成绩照样列出。
- **封存与曝光**：留出任务和封存时期先在账本里登记曝光再运行；每个 Harness 版本在每个留出任务、每个封存分区上只评一次，`final_test` 必须在 `validation` 之后。验证中断或遇到服务故障后，再次验证只补没有结果的任务。方法竞技中的方法同样每个封存分区只评一次。

## 当前参考成绩与实测

宿主内置方法只用 `calibration_fit` 拟合，在各隔间同一批计分起点上的隐藏期（`model_selection`）总技能（智能体回合的 `TASK.md` 与裁判报告里给出的参考值就是这些）：

| 隔间 | 专家模型 `expert` | recipe 种子 | | 隔间 | 专家模型 | recipe 种子 |
|---|---|---|---|---|---|---|
| 黄瓜 2018 AiCU（训练） | +0.129 | +0.088 | | 番茄 2019 AICU（训练） | +0.158 | +0.104 |
| 黄瓜 2018 DeepGreens（训练） | +0.148 | +0.096 | | 番茄 2019 Automatoes（训练） | +0.143 | +0.104 |
| 黄瓜 2018 Croperators（留出） | +0.144 | +0.111 | | 番茄 2019 Digilog（留出） | +0.133 | +0.086 |
| 黄瓜 2018 Sonoma（留出） | +0.116 | +0.065 | | 番茄 2019 IUACAAS（留出） | +0.199 | +0.125 |
| 黄瓜 2018 iGrow（留出） | +0.097 | +0.074 | | 番茄 2019 TheAutomators（留出） | +0.165 | +0.108 |

番茄的专家模型沿用黄瓜结构，未在番茄上调。（表中的“训练”和“留出”按 `greenhouse.climate@1` 划分。）

2026-10-29 的完整运行（`greenhouse.climate@1` 的 4 个训练任务，内循环和评审都用 `pjlab/deepseek-v4.1-flash`，每个回合 8 分钟，1 代 × 2 个候选，每个版本每个任务 2 个回合，共 32 个回合、2,200 万 token、约 45 分钟）：

| 版本 | 4 个训练任务的平均 | 说明 |
|---|---|---|
| 起点 Harness（每任务 4 个回合） | +0.089 | 同一任务上的回合差别很大，例如 AiCU 四个回合 +0.018 至 +0.141 |
| 候选 a / b（每任务 2 个回合） | +0.116 / +0.118 | 模板改为逐格基线残差岭回归加收缩；增益 +0.027 / +0.029，但 4 个任务的区间跨 0，没有采纳 |
| 起点 Harness 4 个回合的预报平均 | +0.132 | 各自写出的程序误差相关性低；两个候选的程序彼此接近，平均后几乎不变 |

在可见数据末尾多开检验窗口不比现在的本地检验更准（与隐藏期的相关 +0.50 对 +0.58），回合内智能体自己的选择已接近事后最优（+0.098 对 +0.103）。细节见 [docs/evolution.md](docs/evolution.md) 第 11、12 节。

## 训练数据

`ecogym export --out DIR` 从账本和轨迹生成：

- `sft.jsonl`：每个会话线程一条多轮样本（system、user、带推理和工具调用的 assistant、tool 结果），附工具定义和元数据（任务、Harness、模型、隐藏期成绩与区间、是否优于参考方法、来自哪次进化）。默认只收正常结束、评分的是被接受的提交（不是从未提交时的自动快照）、隐藏期成绩高于朴素基线的回合，以及提出了有效候选的评审会话；`--all` 全收并用 `meta.passes_filters` 标记。续跑的每段会话是各自的线程。
- `preferences.jsonl`：同一任务上隐藏期成绩差至少 `--margin` 且配对按天自助区间在 0 以上的两个回合，成绩高者为 chosen。默认只配对提示完全相同（同一版 Harness）的回合；`--cross-prompt` 也导出提示不同的对，附负方自己的 `rejected_prompt`。
- `notes.jsonl`（经验库）、`episodes.jsonl`（回合索引）、`manifest.json`（计数与过滤条件）。

会话目录统一替换为 `/workspace`，会话令牌和形如密钥的字符串替换为 `<redacted>`；真实密钥从不进入轨迹（网关只记录请求体）。

## 方法竞技（参数级基线）

方法竞技在固定的求解器（岭回归特征配方、温室专家模型）上搜索参数：每代提案 → 初筛日排名 → 不重叠的留出日上与在任方法配对比较 → 晋级。宿主直接计算全部预测，任何运行都能按记录的修改逐事件重放。它给智能体回合提供参考成绩，也可以单独用来比较人写的方法。

```bash
ecogym list                                                # 数据集、任务、求解器
ecogym describe greenhouse.agc2018.cucumber.climate@1      # 分区、计分起点、逐格基线
ecogym evaluate greenhouse.agc2018.cucumber.climate@1 --preset expert
ecogym run greenhouse.agc2018.cucumber.climate@1 --preset expert --proposer sweep
ecogym runs | show <run> | replay <run> | report <run> --out run.html
ecogym seal <run> --stage validation                       # 封存评测：每个方法一次
```

- **晋级门槛**（在当代留出队列上，候选对在任方法）：总分提升大于 `min_delta`；按天重抽样的配对自助区间下限大于 0；每一格都没有可信的退化（3 天移动块自助区间的下限不大于 0，且点估计退化不超过 3%）。每代把 `model_selection` 的日子按 3 天一组随机分成初筛（约 1/3）和留出（约 2/3），研究工具只能看初筛日。
- **LLM 提案者**（`--proposer llm`）：研究员调用只读研究工具（`overview`、`column_profile`、`residual_profile`、`try_edit`，数据只来自 `calibration_fit` 和当代初筛日）写研究笔记，提案者把笔记变成至多 k 个修改，评审只在有具体理由时剔除。两种传输：OpenAI 兼容接口（`--option base_url=… model=… key_env=…`，可选 `proxy=none`、`fallbacks=[…]`）和 DSH 无头模式（`--option transport=dsh model=provider/model`，DSH 自身工具全部禁用）。
- **封存结果（黄瓜 AiCU，2026-10-28，最终方法为专家模型）**：validation +0.127 [+0.085, +0.167]（406 个起点），final_test +0.130 [+0.097, +0.153]（407 个起点）；CO2 的 6 和 24 小时在后期略差于朴素基线，预报整体偏低。验收记录见 [docs/implementation-2026-10-28.md](docs/implementation-2026-10-28.md)。

服务的方法竞技接口（`/api/runs*`）面板不调用，留给直接使用 `ecogym serve` 的程序（命令行不经服务，直接调用方法竞技）。

## 代码结构

```
src/ecogym/
  kernel/      规范 JSON 与摘要、严格配置解析、SQLite 账本（只追加、哈希链）、密钥解析
  data/        数据集规格、csv_tables 加载器、稠密时间网格 Frame、缓存
  task/        任务卡与 <任务卡>:<时段> 引用、任务组
  protocol/    分区、因果视图、计分起点、基线、队列
  solvers/     求解器注册、岭回归、特征配方
  evaluation/  评分、块自助法、晋级门槛、多任务配对比较、Bench
  harness/     Harness 版本（内容寻址）、检查、diff；起点版本 seed/
  episodes/    内循环：沙箱、模型网关与提供方路由、MCP 工具服务、会话与续跑、裁判、工作区、专家意见、轨迹、训练数据导出
  evolution/   外循环：评审会话与证据、进化循环、独立验证、几个回合预报的平均、账本折叠
  arena/       方法竞技：基因组修改与信赖域、提案者、竞技循环、封存评测、回放
  agents/      方法竞技的 LLM 提案者：传输、研究工具、研究员／提案者／评审
  cli.py  service.py（接口表 ROUTES）  report.py
  web/         DSH 插件安装器；plugin/ 是插件本身：lib/index.js（随 DSH 启停服务、同源代理）、
               lib/client.js（「生态进化」面板，所有调用在它的 Api 表里）、cordis.patch.yml（profile 层）
  domains/greenhouse/   AGC 2018 黄瓜与 2019 番茄、任务卡、两个任务组、专家求解器、领域知识
```

上层只依赖下层，框架代码不出现任何领域名称，这两点由 `tests/ecogym/test_layers.py` 检查。前后端怎样分工见 [docs/architecture.md](docs/architecture.md)。

## 已知限制

- 沙箱只实现了 macOS（`sandbox-exec`）；DSH 只在 0.1.5-rc.2 上验证过。服务是单机单用户的，只监听 127.0.0.1，经 DSH 访问时由 DSH 的登录保护。
- 目前只有温室一个领域。留出隔间与训练隔间共享天气和日历，主要检验向其他控制策略的迁移；向未来时期的迁移由封存时期检验。
- 4 个训练任务时门槛的检验力低：t 区间只有 3 个自由度，2026-10-29 的测试里任务间增益的标准差约 0.028，一个候选的平均增益要超过约 +0.044 才能通过。可以用 `greenhouse.wide.climate@1` 或更多回合，成本线性增加。
- 回合之间的随机差异大，建议 `--repeats 2` 以上；每个版本每个任务只有 1 个回合时，单任务否决不起作用。
- 0.2.0 新增的逻辑（接口表、几个回合预报的平均、续跑规则、多任务门槛、提供方路由）还没有单元测试，目前靠离线检查和真实运行验证。
- 更多见 [docs/evolution.md](docs/evolution.md) 第 9 节。

## 文档

- [Harness 自进化设计](docs/evolution.md)：内循环、外循环、多任务比较、独立验证、训练数据、成本与限制，以及冒烟验收、完整任务组测试和评测方案检验的记录
- [前后端架构](docs/architecture.md)：面板、插件宿主端和服务的分工，请求路径，状态放在哪里，页面与接口，怎样加接口和页面
- [新增领域指南](docs/new-domain.md)：数据集规格、任务卡、任务组、领域求解器
- [温室专家模型研究](docs/greenhouse-forecasting-study-2026-10-28.md)
- [重构方案](docs/refactor-proposal-2026-10-28.md)与[实施记录](docs/implementation-2026-10-28.md)（0.1.0）
- [变更记录](CHANGELOG.md)

## 开发

```bash
make test        # python -m unittest discover -s tests -t .
make lint        # ruff
node --check src/ecogym/web/plugin/lib/client.js   # 面板没有构建步骤，检查语法
```

需要真实数据、沙箱或智能体运行时的测试在条件不满足时跳过。

ecogym 是 EcologyRSI-DSH 0.10.33（标签 `v0.10.33`）的重写，不读取旧系统的运行。`tests/ecogym/golden/` 中的金标准文件由 0.10.33 生成，`test_parity.py` 用它们检查两个数据集的全部列、分区边界、逐格基线、计分起点集合（449／154／683），以及专家模型和 recipe 种子的逐格 RMSE（相差小于 1e-12）。

## 许可

见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
