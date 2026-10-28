# 新增领域指南

新增一个生态预测领域只需要写数据和配置，不改框架代码。框架代码里不出现任何数据集、列名或目标名，`tests/ecogym/test_layers.py` 会检查这一点。本指南以仓库自带的温室领域包为例。

## 1. 领域包的结构

```
src/ecogym/domains/<领域名>/
  __init__.py            声明 DATASETS、TASKS；导入领域求解器（可选）
  <数据集>.toml          数据集规格
  tasks/<任务>.toml      任务卡（可以有多个）
  knowledge.md           给研究智能体看的领域知识（可选）
  expert.py              领域求解器（可选）
```

内置领域包在 `ecogym/domains/__init__.py` 的 `BUILTIN` 中登记。外部领域包可以单独发布，在自己的 `pyproject.toml` 中声明入口点，安装后自动加载：

```toml
[project.entry-points."ecogym.domains"]
soil = "my_soil_pack"
```

`__init__.py` 的最小内容：

```python
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = (HERE / "site_x.toml",)
TASKS = (HERE / "tasks" / "moisture.toml",)
```

## 2. 数据集规格

数据放在数据根目录（`$ECOGYM_DATA_ROOT`，默认 `~/.ecogym/data`）下的 `path` 子目录。`csv_tables` 加载器读取“每个时段（站点、团队）一个目录、每个目录若干 CSV 表”的布局，也支持全数据集共享的表（例如一个气象站）。

```toml
id = "site_x_soil"
title = "Site X soil moisture"
source = "https://doi.org/..."
path = "soil/site_x"            # 相对数据根目录
loader = "csv_tables"
step = "1d"                     # "1h" 或 "1d"
episodes = "*/sensors.csv"      # 第一张表；每个匹配目录是一个时段，目录名即时段名
external_episodes = ["reference"]   # 名字含这些子串的时段永不用于优化

[[columns]]
name = "soil_moisture_10cm"     # 规范列名，小写下划线
role = "environment"            # 角色由领域自定；任务卡按角色决定哪些列可作输入
unit = "m3_m3"
label = "10 cm 土壤含水量"
aliases = ["SWC_10"]            # 源文件列名，大小写和标点不敏感
required = true
valid = [0.0, 0.6]              # 超出范围的读数在聚合前丢弃

[[tables]]
file = "sensors.csv"
time = ["timestamp"]            # 时间列别名；数值按 1899-12-30 起的天数解析，也接受常见日期格式
interval_minutes = 30           # 源采样间隔；用于把只有日期的行在一天内展开
columns = ["soil_moisture_10cm"]

[[tables]]
file = "weather.csv"
scope = "dataset"               # 放在数据集根目录、所有时段共享
time = ["date"]
interval_minutes = 1440
delay = 1                       # 可得延迟（步）：日汇总要等这一天结束后才能用
columns = ["precipitation"]
```

要点：

- 第一张表决定每个时段有哪些时间步；其他表只在这些时间步上取值。
- 每张表按数据集步长取均值；`allow_suffix = true` 时别名 `x` 也匹配 `x_1`、`x_2` 等列，并先在列之间取均值。
- 周表可用 `time_format = "iso_week"` 加 `week_year`。
- `[[derived]]` 可以生成源数据没有直接给出的合计（`op = "sum"` 或 `"difference"`）。
- 加载结果按规格摘要和源文件的大小、修改时间缓存在 `$ECOGYM_CACHE_DIR`（默认 `~/.cache/ecogym`）。

## 3. 任务卡

任务卡定义预测什么、能用什么信息、如何判定更好。所有时长都以数据集步长计。

```toml
id = "soil.site_x.moisture@1"
title = "Site X: 10 cm soil moisture at 1, 3 and 7 days"
dataset = "site_x_soil"
episodes = ["plot_a"]
horizons = [1, 3, 7]
history_steps = 60              # 每个计分起点都有 60 步完整的目标历史
history_scope = "partition"     # "partition"：读数和前向填充不出当前分区；"causal"：可读更早的分区
block_steps = 7                 # 自助法的时间块（这里按周）
season_steps = 7                # 季节基线的周期；不超过 history_steps
periods = [7]                   # 配方中季节项可用的周期
knowledge_files = ["../knowledge.md"]

[[targets]]
name = "soil_moisture_10cm"
weight = 1.0
bounds = [0.0, 0.6]             # 预测会被裁剪到物理范围

[exogenous]
roles = ["weather"]             # 可作输入的角色；"outcome" 一类的列不要放进来
min_coverage = 0.2

[exogenous.max_fill_age]        # 每个角色的读数最多前向填充多少步
weather = 2

[protocol]
embargo = 7                     # 分区之间的隔离期，默认等于最大时距

[gates]
min_delta = 0.005

[presets.recipe_seed]
solver = "recipe_ridge@1"
note = "target lags and a weekly reference"

[presets.recipe_seed.params]
alpha = 0.1
scale = 0.5
terms = [
    { op = "target_lag", k = 0 },
    { op = "target_lag", k = 1 },
    { op = "seasonal_reference", period = 7 },
    { op = "exogenous", col = "precipitation", lag = 0 },
]
```

分区规则（四阶段、时间前向）、门槛参数和默认值见 `src/ecogym/task/spec.py`。默认分区为：前 60% 时间用于训练，其中前一半再按 70%／30% 分为 `calibration_fit` 和 `calibration_uq`，后一半为 `model_selection`；随后 20% 为 `validation`，最后为 `final_test`，相邻分区之间留出隔离期。日历切点在数据集全部非外部时段上统一计算。

## 4. 领域求解器（可选）

通用求解器 `recipe_ridge@1` 已能表达滞后、季节参照、滚动统计、日周期相位和外生输入。需要领域结构时，在领域包中登记一个 `SolverSpec`：

```python
from ecogym.solvers.base import Param, SolverSpec, register_solver

def fit(data, ctx, params):        # data：calibration_fit 起点，带标签
    ...
def predict(state, origins, ctx):  # origins：只能用 origins.past(列, k)，k >= 0
    return {cell.key: forecasts for cell in ctx.task.cells}

register_solver(SolverSpec(
    id="soil.bucket", version=1, title="Bucket model", doc="...",
    params=(Param("alpha", "float", 0.5, lo=1e-4, hi=1.0, doc="..."),
            Param("scale", "cells", 1.0, lo=0.0, hi=1.0, doc="...")),
    fit=fit, predict=predict,
))
```

求解器只能通过 `Origins` 读数据：`past(列, k)` 返回起点前 k 步的值，`past_filled(列, k, 最大填充步数)` 返回前向填充后的值，`label(目标, h)` 只在拟合数据上可用。参数表就是提案者能改的全部内容：数值参数自动成为搜索坐标（跨两个数量级以上的正区间按对数尺度），`cells` 类型是每格一个值，`json` 类型由 `validate` 函数校验。温室专家模型 `domains/greenhouse/expert.py` 是一个完整示例，机理项的列名写在任务卡的种子方法参数里，不写在代码里。

## 5. 领域知识

`knowledge_files` 中的 Markdown 会原样放进研究智能体的说明。写事实和已检验的结论（包括无效的尝试），不要写对特定分区结果的猜测。

## 6. 检查

```bash
ecogym list                                   # 新数据集、任务和求解器出现在列表中
ecogym describe soil.site_x.moisture@1        # 分区边界、各阶段计分起点数、逐格选定的基线
ecogym evaluate soil.site_x.moisture@1 --preset recipe_seed
ecogym run soil.site_x.moisture@1 --preset recipe_seed --proposer sweep
python -m unittest tests.ecogym.test_domains  # 领域一致性测试：每个任务卡都能在自己的数据上跑通
```

`describe` 报告某个阶段没有计分起点时，通常是 `history_steps` 太长、数据缺测太多或分区太短；`ProtocolError` 会说明哪一项证据不足。
