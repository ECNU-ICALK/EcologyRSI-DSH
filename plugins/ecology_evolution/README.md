# 生态模型进化工作台

工作台由 DSH 托管网页，Python Host 负责数据分区、执行编排、评分和 SQLite 事件账本。页面使用原生 HTML/CSS/JavaScript。安装与模型连接说明统一维护在[仓库 README](../../README.md)，本次实现与实跑结果见[优化记录](../../docs/refactor/QUICK-OPTIMIZATION-IMPLEMENTATION-20260909.md)。

## 当前默认流程

网页优先选择 2018 黄瓜或 2019 番茄数据集，以及 DSH 目录中的策略模型与评审模型。创建前检查研究、轮次评审、样本 Planner 与样本 Critic 的真实结构化工具协议；通过后冻结模型、数据、执行预算和插件内容指纹。

默认 `quick_adaptive_epoch@1`：5 轮，每轮 100 个训练起点，每批 10 个，轮末用 50 个新起点比较候选和轮初冠军。每轮只训练一条主线，基础执行量为 200 次，五轮为 1000 次。研究、反思、条件 Critic 和故障恢复的模型调用另外计量。正式对照模式须显式选择，默认最多 936 次执行/轮。

每批完整执行后反思，修订仅应用于下一批并标记为待验证。轮末固定最后一个合法修订，与轮初冠军共同评测；完整执行、总体实用增益和分项约束共同决定是否更新工作版本。快速实验不授予统计认证，独立评测使用隔离的时间分区。

一个样本指一个预测起点。Agent 读取截止该起点的因果信息，自行决定直接预测或调用登记模型工具，最终提交温度、湿度、CO₂ 在 1/6/24 小时的九个预测值。预测工具每次 Planner 尝试最多调用 2 次；工具结果是证据，最终数值由 Agent 提交。Planner 最多 10 步、Critic 最多 4 步；单次输出额度分别为 8192/4096 tokens。累计输出回执另有阶段中止阈值，不能解释为精确费用硬上限。

服务 503、429、连接和超时进入有界恢复与路由冷却，已接受的 Planner 和工具结果可以供 Critic 恢复复用。不可恢复的协议、输出预算等故障关闭本次执行准入。执行不完整时不进入后续批次、反思择优或额外评测副本，不以失败罚分证明科学改善。

## 页面与数据加载

- **运行设置**：选择数据集、模型、实验模式和参数；容量与预算读取后端同一份冻结执行计划。
- **训练数据**：按 `training_fit` / `training_feedback` 分区分页读取样本和字段定义。
- **进化过程**：显示当前阶段、有效样本、在途任务、等待恢复、最近故障、真实用量和轮末选择。
- **候选评测**：按需读取所选候选的参数、工具、三目标 × 三时距指标及同 cohort 对照。
- **独立评测**：单独启动隔离数据评测，不将结果混入搜索反馈。
- **人工协作**：暂停后追加方向建议，模型咨询与人工干预各自保留审计收据。

列表和总览只加载摘要；工作区按照自身版本号失效，遥测变化不触发全部候选重载。明细、样本和事件均按需请求，过期响应不得覆盖当前运行或候选。浏览器只展示结构化脱敏证据。

## 本地运行

当前 DSH 主实例页面为 `http://127.0.0.1:8848/plugins/ecology/evolution/`，后端端口为 8777。页面通过同源 `/api/ecology-evolution` 访问服务。完整安装与启动命令见仓库 README。

只调试 Host 页面时可在仓库根目录运行：

```bash
PYTHONPATH=src .venv/bin/python -m ecologyrsi_dsh serve \
  --host 127.0.0.1 --port 8765 --db /tmp/ecologyrsi-dsh.sqlite3
```

未连接原生模型运行时时，真实执行显示不可用。静态演示须显式使用 `?demo=1`，不会在服务失败时自动伪造结果。

## 主要接口

统一前缀 `/api/ecology-evolution`：

```text
GET  /health
GET  /catalog
GET  /datasets/{dataset_id}/samples?partition=training_fit&offset=0&limit=20
GET  /runs?view=summary&offset=0&limit=5
GET  /runs/{run_id}?view=overview
GET  /runs/{run_id}?view=process
GET  /runs/{run_id}?view=candidates
GET  /runs/{run_id}?view=candidate&candidate_id={candidate_id}
GET  /runs/{run_id}/events?after={cursor}
POST /model-preflight
POST /runs
POST /runs/{run_id}/control
POST /runs/{run_id}/advance
```

旧运行保留原始冻结协议和审计记录。更改模型、工具、预算或绑定内容后须创建新实验；历史结果不会被升级为新协议证据。
