# EcologyRSI-DSH · ecogym

**ecogym 是一个生态学预测训练场**：基线、专家模型、参数搜索和 LLM 研究智能体在同一套可复现的规则下比较。训练场负责定义任务、控制信息集（不能看未来，不能看封存分区）、用统一的统计门槛判断“是否真的更好”、记录完整来源，并保证每个方法在封存的验证集和最终测试集上各只评一次。

> 当前版本 **ecogym 0.1.0**（取代 EcologyRSI-DSH 0.10.33）· Python 3.11+ · 依赖 numpy
>
> 旧系统的运行记录留在标签 `v0.10.33` 中查看，见[从 0.10.33 迁移](#从-01033-迁移)。

[快速开始](#快速开始) · [核心概念](#核心概念) · [评测规则](#评测规则) · [LLM 研究智能体](#llm-研究智能体) · [网页工作台](#网页工作台) · [代码结构](#代码结构) · [新增领域](docs/ecogym/new-domain.md)

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e .
```

数据放在数据根目录下：`$ECOGYM_DATA_ROOT`，未设置时依次使用 `~/.ecogym/data` 和旧位置 `~/.ecologyrsi-dsh/data`。自带的温室领域需要 Autonomous Greenhouse Challenge 数据，解压到 `greenhouse/agc_cucumber_2018/`（[2018 黄瓜](https://doi.org/10.4121/uuid:e4987a7b-04dd-4c89-9b18-883aad30ba9a)）和 `greenhouse/agc_tomato_2019/`（[2019 番茄](https://doi.org/10.4121/uuid:88d22c60-21b3-4ea8-90db-20249a5be2a7)），每个团队一个子目录。

```bash
ecogym list                                                # 数据集、任务、求解器
ecogym describe greenhouse.agc2018.cucumber.climate@1      # 分区、计分起点、逐格基线
ecogym evaluate greenhouse.agc2018.cucumber.climate@1 --preset expert
ecogym run greenhouse.agc2018.cucumber.climate@1 --preset expert --proposer sweep
ecogym runs                                                # 账本中的运行
ecogym replay <run>                                        # 按记录的修改重新计算并逐事件比对
ecogym report <run> --out run.html                         # 静态 HTML 报告
ecogym seal <run> --stage validation                       # 封存评测：每个方法一次
ecogym serve                                               # 网页工作台 http://127.0.0.1:8850/
```

运行与封存记录写入同一个账本 `$ECOGYM_HOME/ledger.sqlite3`（默认 `~/.ecogym`），`--ledger` 可指定其他文件。

## 核心概念

| 概念 | 含义 | 位置 |
|---|---|---|
| 数据集 Dataset | 多变量时间序列，按时段（团队、站点）组织；变量带角色、单位、有效范围和可得延迟 | `data/`，规格为 TOML |
| 任务卡 Task | 目标与权重、时距（以步计）、历史长度与范围、可用输入角色、基线菜单、分区协议、门槛、种子方法 | `task/`，TOML |
| 协议 Protocol | 时间前向的五个分区、因果读取视图、计分起点、逐格基线、初筛／留出队列 | `protocol/` |
| 求解器 Solver | `fit(拟合起点)` 与 `predict(预报起点)`，参数表登记一次，搜索坐标、校验和提案说明都由它派生 | `solvers/` |
| 基因组 Genome | 求解器引用加完整参数；它的摘要就是一个方法的身份 | `solvers/base.py` |
| 提案者 Proposer | 产生有界修改：扫描、随机、LLM 研究智能体，或人工 | `arena/proposers.py`、`agents/` |
| 竞技循环 Arena | 每代：提案 → 初筛 → 留出比较 → 晋级 | `arena/loop.py` |
| 账本 Ledger | 追加写、带哈希链的 SQLite 事件日志，外加封存曝光登记 | `kernel/ledger.py` |
| 领域包 Domain pack | 数据集规格、任务卡、领域知识、领域求解器 | `domains/` |

## 评测规则

- **分区**：每个时段按时间切成 `calibration_fit`、`calibration_uq`、`model_selection`、`validation`、`final_test`，相邻分区之间有隔离期。日历切点在数据集全部非外部时段上统一计算。方法只在 `calibration_fit` 上拟合；`calibration_uq` 是不参与任何选择的开发期复核；`validation` 和 `final_test` 封存。
- **信息集**：求解器只能通过 `Origins.past(列, k)`（k ≥ 0）读取起点及以前的数据；任务的 `history_scope = "partition"` 时，读数和前向填充都不出当前分区。标签只在拟合数据上可见。
- **计分起点**：一个起点要求每个目标最近 `history_steps` 步都有观测、每个时距的标签都在同一分区内。所有方法在同一组起点上评分。
- **技能**：每格对比在 `calibration_fit` 上选定的朴素基线（持续性或季节值）：`skill = (RMSE_基线 − RMSE) / max(RMSE, RMSE_基线)`。缺失预报按覆盖率扣分。总分是逐格加权平均。
- **晋级门槛**（在当代留出队列上，候选对在任方法）：总分提升大于 `min_delta`（默认 0.005）；按天重抽样的配对自助区间下限大于 0；每一格都没有可信的退化（3 天移动块自助区间的下限不大于 0，且点估计退化不超过 3%）。
- **队列**：每代把 `model_selection` 的日子按 3 天一组随机分成初筛（约 1/3）和留出（约 2/3）。候选在初筛日上排名，最好的一个在不重叠的留出日上与在任方法比较。研究工具只能看初筛日。
- **封存**：`ecogym seal` 先在账本里登记曝光再打开分区；同一方法在同一分区只能评一次，`final_test` 必须在 `validation` 之后。曝光键只取决于数据和方法，改门槛设置不能换来第二次评测。报告会列出该分区累计被多少个方法看过。

## LLM 研究智能体

`--proposer llm` 每代按三个角色工作：研究员调用研究工具分析数据并写研究笔记，提案者把笔记变成至多 k 个具体修改，评审只在有具体理由时剔除修改。所有预测都由宿主根据方法参数计算，智能体只在方法层工作。每次调用、研究笔记和评审决定都记在账本里；一代默认最多 10 次调用。

研究工具：`overview`、`column_profile`、`residual_profile`、`try_edit`。数据只来自 `calibration_fit` 和当代的初筛日，看不到留出日、`calibration_uq` 和封存分区。

两种传输：

```bash
# OpenAI 兼容接口；密钥从 key_env 指定的环境变量读，或从 $ECOGYM_CREDENTIALS_FILE 中的 “名称: 值” 行读
ecogym run greenhouse.agc2018.cucumber.climate@1 --preset expert --proposer llm \
  --option base_url=https://api.example.com/v1 --option model=glm-5.3 --option key_env=MY_API_KEY \
  --option min_interval=20 \
  --option 'fallbacks=[{"base_url": "https://backup.example.com/v1", "key_env": "BACKUP_KEY"}]'

# DeepSeek Harness 无头模式：复用 DSH 的模型配置与凭据；DSH 自身的工具全部禁用，进程在空目录中运行
ecogym run ... --proposer llm --option transport=dsh --option model=provider/model \
  --option dsh_bin=/path/to/dsh --option 'patches=["/path/to/models.patch.yml"]'
```

`ecogym replay` 对 LLM 运行同样有效：它按账本中记录的修改重新拟合、评分和判定，逐事件比对（智能体对话本身不重放）。

## 网页工作台

`ecogym serve` 在 `127.0.0.1:8850` 提供中文工作台和 JSON 接口：新建运行（含 LLM 设置）、运行列表、逐代时间线（研究笔记、候选修改、初筛与留出结果、未通过的格）、种子与最终方法的逐格对比、封存评测按钮和 HTML 报告。绑定非回环地址时必须设置 `--token`；所有 POST 请求需要 `X-Ecogym: 1` 头。

## 代码结构

```
src/ecogym/
  kernel/      规范 JSON 与摘要、严格配置解析、SQLite 账本
  data/        数据集规格、csv_tables 加载器、稠密时间网格 Frame、缓存
  task/        任务卡
  protocol/    分区、因果视图、计分起点、基线、队列
  solvers/     求解器注册、岭回归、特征配方
  evaluation/  评分、块自助法、晋级门槛、Bench
  arena/       基因组修改与信赖域、提案者、竞技循环、封存评测、回放
  agents/      LLM 传输、研究工具、研究员／提案者／评审
  cli.py  service.py  report.py  web/
  domains/greenhouse/   AGC 2018 黄瓜与 2019 番茄、任务卡、专家求解器、领域知识
```

上层只依赖下层，框架代码不出现任何领域名称，这两点由 `tests/ecogym/test_layers.py` 检查。

## 从 0.10.33 迁移

ecogym 是重写，不兼容旧运行。旧系统（Python 宿主、DSH 插件、旧网页）保留在标签 `v0.10.33`：

```bash
git worktree add ../ecologyrsi-v0.10.33 v0.10.33
```

迁移时用金标准对照测试（`tests/ecogym/golden/`，由旧系统生成）保证数值一致：两个数据集全部列逐位一致；各分区边界、逐格基线选择和计分起点集合（449／154／683）完全一致；专家模型（model_selection +0.1292、calibration_uq +0.1769）和 recipe 种子（+0.0878、+0.1179）的逐格 RMSE 与旧系统相差小于 1e-12。

有意的改变：逐样本 LLM 链取消，预测由宿主直接计算；逐格非劣检验的对照从朴素基线改为在任方法；每代队列改为按天分组的初筛／留出划分；预测裁剪到目标的物理范围。

## 开发

```bash
make test        # python -m unittest discover -s tests -t .
make lint        # ruff
```

需要真实数据的测试在数据缺失时跳过。金标准文件由 `tests/ecogym/golden/make_golden.py` 在 `v0.10.33` 检出上生成。

## 许可

见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
