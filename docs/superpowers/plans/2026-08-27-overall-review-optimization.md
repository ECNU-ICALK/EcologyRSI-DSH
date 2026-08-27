# Overall Runtime and Evolution Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不保留历史运行、旧数据库和旧协议兼容分支的前提下，把当前温室预测进化工作台收敛为一个可观测、可取消、可恢复、可验证的 v2 运行协议。保持外层“四候选 → 同一 64 个预测起点 → 现有 Top 2 选择”不变，只重构每个 finalist 的 500 起点/10×50 batch 持续进化、运行时调度和证据统计。

**Architecture:** 采用单一严格协议、单一 Host 状态机和单一 provider admission。Host 负责不可变任务清单、起点 occurrence、候选 revision、批次边界、评分和晋级；DSH 负责 Agent Session/Workflow 与模型调用；浏览器只消费脱敏 projection 和异步 command receipt。一个预测起点仍是 3 目标 × 3 时距的完整 9-cell 向量，样本并发只计起点链，不再由候选并发向上取整放大。

**Tech Stack:** Python 3.10+、标准库 dataclass/threading/unittest、SQLite WAL 追加式事件账本、DSH Cordis/ESM、Vanilla JavaScript、Node.js smoke test。新协议使用全新数据库 schema 和全新 preset/catalog；旧数据仅作为离线归档文件，不被新服务打开或回放。

**Spec:** 本文是整体 review 后的实施总计划；Top-2 adaptive epoch 的已有细节继续参考 `docs/superpowers/specs/2026-08-27-top2-adaptive-epoch-design.md`，但以下运行时、容量和交付合同以本文为准。

## Global Constraints

- 外层 Top-2 的排序、tie-break、四候选数量和同一 64 起点屏蔽语义不改。若 `passed` 字段与现有排序语义冲突，必须改字段含义或生成阶段，而不是悄悄改变 Top-2 排序。
- 500、50、169 的单位都是完整预测起点（origin），不是评分 cell；9 个 cell 只由起点展开得到。
- 一个起点的 9-cell 必须作为一个原子结果提交；任何 partial vector 都不能计入成功起点。
- 每个 finalist 的 batch 只影响下一 batch；禁止使用同一 batch 的观测结果修改后又在该 batch 上评分。
- 用户配置的 64 是单 run 起点链逻辑上限，128 是 provider route 的物理上限；两者在 projection 中分别展示，不能互相替代。
- 数据耗尽后允许确定性循环复用，但所有 occurrence 必须带 `source_origin_id + cycle_index`；任何“unique origins”字段不得再表示 occurrence 数。
- 不增加 CO₂ 专用模型，不增加分目标/分时距模型族，不把当前 point forecast 重新包装成因果控制系统。
- 新协议一次性切换：删除旧 schema decoder、旧 one-shot formal 路径、旧 host/gateway fallback、旧 preset 版本和旧浏览器字段。旧 DB 先复制到只读归档目录，再由新服务使用全新 DB。
- 本轮只做 review 和计划，不直接改生产代码。

## 1. Review 结论与证据

### P0：会造成“7% 长时间不动”或控制请求无效

| 问题 | 证据位置 | 影响 | 目标修复 |
|---|---|---|---|
| provider 排队不计入阶段 deadline | [`stage-runner.js:701-779`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js:701)、[`provider-stage-gate.js:109-156`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/integrations/dsh_ecology_plugin/lib/runtime/provider-stage-gate.js:109) | 请求可无限等 FIFO，进度只完成少量起点；阶段超时在拿到 slot 后才开始 | 入队前建立 deadline，排队、reservation、模型调用、持久化共享剩余时间；超时写 `provider_queue_timeout` |
| 取消只拒绝调用方，底层 operation 仍占 active slot | [`provider-stage-gate.js:165-179`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/integrations/dsh_ecology_plugin/lib/runtime/provider-stage-gate.js:165) | UI 看到取消但 provider 槽位不释放，后续请求继续排队 | operation 接受 AbortSignal；不可中断时显式 `draining`，控制接口不等待无限 drain |
| pause/cancel/archive/control 使用短浏览器 timeout，而服务端同步等待 quiesce | [`commands.js:440`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/plugins/ecology_evolution/assets/js/commands.js:440)、[`commands.js:489`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/plugins/ecology_evolution/assets/js/commands.js:489)、[`handler.py:3297-3304`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/api/handler.py:3297) | 原操作已经开始但浏览器收到 timeout，形成“归档失败/取消无效” | 命令先落盘并返回 202 receipt，后台完成 drain；增加 command status 查询 |

### P1：会造成吞吐、统计或维护结果失真

| 问题 | 证据位置 | 影响 | 目标修复 |
|---|---|---|---|
| queued/in-flight 是推算值，不是真实 provider 状态 | [`projection.py:1797-1805`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/api/projection.py:1797) | Python admission 等待、provider FIFO、workflow 执行和 draining 被混成一个“排队”数字 | 由请求生命周期事件汇总真实状态计数，删除 `min(outstanding, configured_concurrency)` 估算 |
| reservation 使用全局 launch lock 并扫描完整事件流 | [`dsh_tools.py:808-842`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/api/dsh_tools.py:808) | 64/128 并发下锁竞争和 O(n) replay 放大，可能看似提交但长时间没有 launch | 按 run+business key 分片锁；SQLite/内存索引直接定位最新 attempt |
| Python admission 与 DSH gate 双重自适应、候选并发还可能形成层级放大 | [`sample_admission.py:94-124`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/api/sample_admission.py:94)、[`sample_execution.py:110-135`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/evaluators/sample_execution.py:110) | 一次错误减半、恢复缓慢；不同候选/不同 run 的实际请求数与页面配置不一致 | Host 只保留 run 配额；provider route 统一控制物理并发；自适应采用窗口 AIMD |
| 筛选记录保存 `passed`，但 Top-2 helper 只按记录、约束和 score 排序 | [`generation_execution.py:108-134`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/api/generation_execution.py:108) | 若 `passed` 被理解为准入门槛，会出现 gate-fail 候选进入 Top-2；若不参与选择，字段会误导 | 保持现有 Top-2 顺序；将字段改为明确的诊断状态，或在记录生成阶段禁止矛盾值，并锁定 golden fixture |
| `required_unique_origins` 实际表示计划 occurrence 数 | [`schedule.py:173-179`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/evolution/schedule.py:173)、[`epoch_cohorts.py:448-495`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/evaluators/epoch_cohorts.py:448) | 数据循环复用后名称与实际科学含义相反，UI/日志易误读 | 改为 `planned_origin_occurrences`，另报 `available_source_origins`、`reused_occurrence_count`、`effective_source_count` |
| 非 9-cell 整倍数的 `samples_per_update` 会在 phase manifest 中整除截断 | [`generation_execution.py:145-164`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/api/generation_execution.py:145) | 用户预算、冻结 manifest 和真实执行数不一致 | 新 API 只接受完整起点数；若保留 cell 输入则显式返回 requested/effective/remainder，禁止静默舍入 |
| 24h block bootstrap 按时间块聚合，未表达循环复用的 source cluster | [`promotion.py:94-188`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/evolution/promotion.py:94) | 同一 source origin 多次复用会增加平方误差权重，却不增加独立证据；置信区间可能过窄 | 统计按 source origin cluster + cycle occurrence 分层，报告 effective N 和 reuse sensitivity |

### P1/P2：交付与工程结构问题

- 最近一天连续落地大量 0.3.27–0.3.33 变更，核心执行、并发、轨迹、晋级和交付同时变化，缺少稳定基线；后续必须以完整测试和虚拟时钟压测作为合并门槛。
- [`projection.py`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/api/projection.py) 在完整投影、summary 等多个位置重复构造 `configuration`；[`handler.py`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/api/handler.py) 同时承担路由、幂等、控制、创建和错误映射。
- [`generation_execution.py`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/api/generation_execution.py)、[`sample_execution.py`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/evaluators/sample_execution.py)、[`registry.py`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/evaluators/registry.py)、[`strategies.py`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/src/ecologyrsi_dsh/evolution/strategies.py) 均为 4,000 行级单体文件，职责和旧协议分支交织。
- Node smoke test 当前在 [`smoke.mjs:2865`](/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH/plugins/ecology_evolution/test/smoke.mjs:2865) 对 preset 目录硬编码，实际还包含 v1/v3/v4/v5，测试期望只包含 v3/v4/v6/v7；这不是运行时数据问题，但会阻断交付验证。
- 完整 Python suite 当前有 1 个已知打包失败：sdist 通过 stale `egg-info/SOURCES.txt` 带入根目录被忽略的中文规划文件；需要清洁构建并重建 SOURCES，而不是继续在 verifier 中增加例外。

## 2. 目标状态机与数据流

```text
RunCreated
  -> RuntimeBindingPending
  -> Running
       -> Screening(4 × 64 origins, Top-2 语义不变)
       -> FinalistTrajectory
            -> BatchStarted(50 origins)
            -> Origin chain × 50
            -> BatchEvidenceRecorded
            -> LocalEditDecided(0..max operations)
            -> RevisionActivated
            -> repeat 10 batches
       -> HoldoutFrozen(F1/F2/incumbent, same 169 origins)
       -> ComparisonRecorded -> ChampionSelected -> GenerationAdvanced
  -> PausedRequested/CancelledRequested
       -> Draining
       -> Paused/Cancelled
```

每个 provider 请求必须经历以下可观测状态，且状态计数守恒：

```text
planned -> local_waiting -> admission_waiting -> provider_queued
        -> provider_active -> workflow_running -> result_persisting
        -> completed | failed | retry_waiting | cancelled | draining
```

### 新的核心实体

```text
OriginOccurrence:
  occurrence_id, source_origin_id, cycle_index, origin_timestamp,
  maturity_digest, cohort_role, generation, candidate_id, revision_id

WorkUnit:
  work_unit_id, run_id, generation, candidate_id, revision_id,
  phase, batch_index, occurrence_ids, status, attempt, lease_owner

RequestLifecycle:
  request_id, provider_route, work_unit_id, queued_at, admitted_at,
  started_at, finished_at, queue_wait_ms, execution_ms, status

CommandReceipt:
  command_id, idempotency_key, command_kind, run_id, status,
  requested_at, accepted_at, completed_at, error_code, retryable
```

## 3. 分阶段实施方案

### Phase 0：建立新基线与切换边界

**Files:** `pyproject.toml`, `src/ecologyrsi_dsh/core/ledger.py`, `tests/fixtures/`, `tests/test_candidate_parallel_evaluation.py`, `scripts/verify_delivery.sh`

- [ ] 新建 `protocol_v2`、ledger schema 8 和全新 DB 初始化路径；启动时若发现旧 schema 或旧 event identity，返回明确错误并提示离线归档。
- [ ] 归档当前 `.runtime/*.sqlite3` 和运行日志到只读目录；归档不进入新服务的读路径。
- [ ] 固定四候选 Top-2 golden fixture：记录 slot、score、constraint、screening digest 和期望 ID；任何后续任务不得改变排序。
- [ ] 固定验收指标：起点进度心跳 ≤30 秒、实际 in-flight ≤ provider 上限、pause/cancel receipt ≤1 秒、每个 batch 可重放、重启后无孤儿 work unit。
- [ ] 先清理 `src/ecologyrsi_dsh.egg-info` 和 build cache，再执行一次干净 `make test`、Node smoke、`make verify`，把基线失败单独登记。

### Phase 1：统一 origin、预算和循环复用合同

**Files:** `src/ecologyrsi_dsh/evolution/schedule.py`, `src/ecologyrsi_dsh/evaluators/epoch_cohorts.py`, `src/ecologyrsi_dsh/core/trajectory.py`, `src/ecologyrsi_dsh/evaluators/registry.py`, `plugins/ecology_evolution/assets/js/catalog.js`, `plugins/ecology_evolution/assets/js/render_shell.js`

- [ ] 用 `planned_origin_occurrences` 替换 `required_unique_origins`；容量报告同时返回 `available_source_origins`、`planned_origin_occurrences`、`reused_occurrence_count`、`cycle_count`、`effective_source_count`。
- [ ] `OriginOccurrence` 采用 `(source_origin_id, cycle_index)` 作为唯一键；同一 source 在不同 cohort 或不同 cycle 不能靠裸 `origin_id` 去重。
- [ ] 规定 deterministic cursor：`cursor = adaptation_count + generation * (screening_count + holdout_count)`，source 按固定 digest 排序后取模；将 cursor、seed 和 policy 写入 manifest。
- [ ] 每个 cohort 内禁止重复 occurrence；允许跨 epoch 复用 source，但必须在指标中做 source-clustered 聚合。重复 source 不提高独立样本数。
- [ ] 预算入口只接受完整 origin 数；9-cell 为内部派生量。所有 API/UI 文案统一“预测起点/评分 cell”两种单位。
- [ ] 增加循环复用敏感性指标：`raw_origin_count`、`unique_source_origin_count`、`effective_origin_count`、`reuse_fraction`、`cluster_weighting`。

**Tests:**

- [ ] property test：任意 eligible 数 >0 时，任意计划轮数都能产生确定性 occurrence，cohort 内无重复 occurrence。
- [ ] property test：相同 seed/manifest 产生相同 digest，不同 cycle 产生不同 occurrence key。
- [ ] test：9-cell 不整倍数请求被拒绝，不能静默截断。
- [ ] test：773 source origins 计划 1665 occurrences 时，报告 892 次复用、source 数仍为 773。

### Phase 2：重做 provider gate、取消和 reservation

**Files:** `integrations/dsh_ecology_plugin/lib/runtime/provider-stage-gate.js`, `integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js`, `integrations/dsh_ecology_plugin/lib/runtime/controller.js`, `src/ecologyrsi_dsh/api/dsh_tools.py`, `src/ecologyrsi_dsh/api/sample_admission.py`, `src/ecologyrsi_dsh/evaluators/sample_execution.py`

- [ ] `ProviderStageGate.run()` 入队前创建 deadline；deadline 覆盖 FIFO 等待、child reservation、模型请求和 structured result 持久化。
- [ ] `operation(signal)` 是唯一调用接口；HTTP/DSH workflow 将 AbortSignal 传到底层。不能中断的请求必须在 record 中转为 `draining`，不能提前把 slot 当成可用。
- [ ] 为每个 request 写入 `queued_at/admitted_at/started_at/finished_at` 和耗时；provider gate 提供 route 级 active/queued/draining snapshot。
- [ ] `drainRun(runId, deadline)` 有硬截止时间；超时返回 drain 状态和数量，不阻塞 HTTP 控制请求。
- [ ] 删除 Python admission 的 provider 自适应职责，只保留 run 的逻辑配额；provider gate 负责 route 的物理上限。run=64、provider=128、candidate concurrency 三者不再相乘。
- [ ] 自适应降速改为最近 32/64 请求的窗口 AIMD：按 429、timeout、5xx、queue wait 计算拥塞，调节事件记录 old/new/window/reason。
- [ ] `dsh_tools.allocate_child_reservation()` 改为 `(run_id, business_key_digest)` 分片锁，新增 reservation 索引表或内存索引；事件流只负责审计，不再做热路径全量扫描。

**Tests:**

- [ ] provider slot 占满时，新请求在 deadline 内返回 `provider_queue_timeout`。
- [ ] cancel 后 queued request 在 1 秒内拒绝；active request 能中断或明确进入 draining。
- [ ] 64 起点并发、128 provider 上限下，实际 active 永不超过上限；多 run 仍共享 route 上限。
- [ ] 事件量扩大 10 倍后 reservation 延迟不按事件总数线性增长；同 business key 仍幂等。
- [ ] 虚拟时钟测试 AIMD 不因一次 429 直接 64→32→16，恢复时间有上界。

### Phase 3：把 auto-progress 改成 durable work-unit scheduler

**Files:** `src/ecologyrsi_dsh/api/auto_progress.py`, `src/ecologyrsi_dsh/api/candidate_scheduler.py`, `src/ecologyrsi_dsh/api/formal_trajectory.py`, `src/ecologyrsi_dsh/api/generation_execution.py`, `src/ecologyrsi_dsh/core/director.py`, `src/ecologyrsi_dsh/core/state.py`

- [ ] `next_work_unit(state)` 只做纯函数选择：screening origin、formal batch origin、local edit、holdout arm、decision barrier；执行一个 work unit 后立即写 checkpoint、释放 lease、重新入队。
- [ ] 一个 batch 仍是 50 origins，但每个 origin chain 完成即写 progress heartbeat；batch 结束后才调用 local editor。进程在 batch 中断时从最后一个 occurrence 继续。
- [ ] 两条 finalist lane 可并行，但每条 lane 的 revision 激活严格串行；local edit 的输出只绑定上一个 batch 的 evidence digest。
- [ ] 失败只重试同一个 work unit/attempt，不重新创建候选或重新消耗已完成起点；超过 deadline 转为结构化 terminal outcome。
- [ ] 浏览器不再推进 server-owned continuous run；只轮询 projection/command status。手动 advance 也复用同一 work-unit lease，不能并发第二个 generation。
- [ ] 为每个 work unit 保存 `lease_owner/lease_expires_at/attempt/checkpoint_digest`，重启 recovery 只认未过期且与 manifest digest 一致的 work unit。

**Tests:**

- [ ] 在第 17/50 个 origin 后强制进程退出，重启后只执行 18–50，不重复提交 1–17。
- [ ] 两条 finalist lane 同时运行时，revision 不交叉；下一 batch 只能读取已激活 revision。
- [ ] auto-progress 多 run FIFO/公平性测试；一个 provider cooldown 不阻塞其他 run。
- [ ] 手动 advance 与后台 worker 同时到达时只有一个 lease 成功，另一个收到可重试的 command receipt。

### Phase 4：收敛候选进化和轮末比较

**Files:** `src/ecologyrsi_dsh/api/formal_trajectory.py`, `src/ecologyrsi_dsh/evolution/local_edits.py`, `src/ecologyrsi_dsh/evaluators/generation_comparison.py`, `src/ecologyrsi_dsh/evolution/promotion.py`, `src/ecologyrsi_dsh/api/generation_execution.py`

- [ ] 保留 `_select_screening_finalists()` 的 Top-2 排序和 golden 结果；将 screening `passed` 明确为诊断字段，或在写入前保证其与既有排序语义一致。
- [ ] `max_local_operations_per_batch` 作为唯一人工参数，默认 2；允许 0 表示 KEEP，允许 1–5 个实际 Host mutation operation。每个 operation 必须有 target、old digest、new digest、affected cells 和 reason code。
- [ ] local editor 只能看到本 batch 的 aggregate metrics/evidence digest，不得看到原始标签、完整预测序列或下一 batch 数据。
- [ ] 每个 batch 产生 `BatchEvidenceRecorded -> LocalEditProposed -> LocalEditDecided -> RevisionActivated` 的不可变链；KEEP 也必须落事件，不能用缺事件表示。
- [ ] 500 起点完成后才做 F1/F2/incumbent 的 169 起点同 cohort holdout；只由轮末 comparison 选择 champion，batch score 不能直接晋级。
- [ ] promotion evidence 按 source origin cluster 聚合；同一 source 的不同 cycle 只作为相关重复观测。bootstrap 的重采样单元为 source cluster 或预先声明的 independent block，并输出 effective N/reuse sensitivity。
- [ ] `generation_comparison.py` 只保留一套 promotion gate：完整 grid、coverage、cell non-regression、practical delta、clustered confidence；删除重复的旧 pairwise/legacy gate。

**Tests:**

- [ ] batch local edit 的 0、1、max 操作边界、越界和重复 target 测试。
- [ ] 同 batch 修改不会影响本 batch 评分；revision digest 链可重放。
- [ ] 三臂 holdout 必须共享 cohort/evaluator/objective contract；source cluster 重采样在循环复用和不复用数据时给出一致的独立样本解释。
- [ ] Top-2 golden test 覆盖 score tie、constraint tie、diagnostic `passed=false`，确认外层排序未被隐式改变。

### Phase 5：API、命令收据和前端状态

**Files:** `src/ecologyrsi_dsh/api/handler.py`, `src/ecologyrsi_dsh/api/execution.py`, `src/ecologyrsi_dsh/api/transport.py`, `src/ecologyrsi_dsh/api/projection.py`, `plugins/ecology_evolution/assets/js/host.js`, `plugins/ecology_evolution/assets/js/commands.js`, `plugins/ecology_evolution/assets/js/data.js`, `plugins/ecology_evolution/assets/js/render_process.js`, `plugins/ecology_evolution/assets/js/catalog.js`

- [ ] `POST /runs`、`POST /runs/{id}/control`、archive/delete 等命令统一返回 `{command_id, status: accepted|completed, run_id, projection_revision}`；长操作默认 202，不等待 DSH drain。
- [ ] 增加 `GET /commands/{command_id}`，返回 command 状态、retryable、retry_after_ms、error_code 和关联 run projection；同 idempotency key 只产生一个 receipt。
- [ ] 错误统一为稳定 code，不让前端解析中文文案：`provider_queue_timeout`、`provider_admission_closed`、`runtime_unavailable`、`draining_timeout`、`invalid_origin_budget`、`stale_manifest` 等。
- [ ] 统一 projection builder，完整 run、summary、events 使用同一 configuration/metrics/progress schema；删除 nullable legacy 字段和重复 aliases。
- [ ] progress 显示 `planned/local_waiting/admission_waiting/provider_queued/provider_active/workflow_running/persisting/completed/failed/retry_waiting/cancelled/draining`，并验证计数总和等于 planned。
- [ ] 前端只保留一个 API client、一个 state store、一个 render scheduler；server auto-progress 运行时停止 browser advance loop，避免双重轮询/竞态。
- [ ] control/archive/delete 请求不再使用 8 秒默认 timeout；改为快速 receipt + command polling。数据读取和命令读取使用不同 timeout。
- [ ] `catalog.js` 和 `render_shell.js` 统一使用新容量字段，显示 source origin 与 occurrence，不再出现“unique origins”误导。

**Tests:**

- [ ] 远端阻塞时 pause/cancel/create 在 1 秒内返回 202；最终状态由 command polling 得到。
- [ ] 浏览器断线后用同 idempotency key 重试返回同一 command；不会重复创建 run 或重复 archive。
- [ ] projection 状态计数守恒，provider snapshot 与 UI 的 active/queued/draining 一致。
- [ ] Node smoke 从单一 preset manifest 读取期望列表，不硬编码 v1/v3/v4/v5/v6/v7 集合。

### Phase 6：拆分单体、删除冗余和建立唯一来源

**目标拆分:**

| 当前文件 | 新职责模块 |
|---|---|
| `api/handler.py` | `routes_read.py`、`routes_commands.py`、`command_receipts.py`、`health.py` |
| `api/projection.py` | `run_projection.py`、`progress_projection.py`、`evidence_projection.py` |
| `api/generation_execution.py` | `generation_orchestrator.py`、`screening.py`、`holdout.py`、`decision_barrier.py` |
| `core/director.py` | `run_lifecycle.py`、`candidate_lifecycle.py`、`trajectory_lifecycle.py` |
| `core/state.py` | `event_replay.py`、`state_indexes.py`、`invariant_checks.py` |
| `evaluators/registry.py` | `binding_registry.py`、`scientific_runner.py`、`judge_runner.py` |
| `evaluators/sample_execution.py` | `origin_chain.py`、`checkpoint_store.py`、`result_publisher.py` |
| `evolution/strategies.py` / `analysis.py` | `proposal_contract.py`、`research_context.py`、`decision_analysis.py` |

**明确删除项（新协议切换时一次性删除）:**

- 旧 `samples_per_update`、cell↔sample 反算和静默舍入分支。
- 旧 one-shot formal-500 执行入口、旧 feedback rotating-window fallback，以及 Host/model gateway fallback（新协议只走 DSH-native）。
- 旧 v1/v3/v4/v5/v6/v7 preset 树、重复 preset catalog 和硬编码版本断言；生成一个 current preset manifest。
- 旧事件 schema decoder、legacy status alias、旧 browser projection 字段；旧 DB 不由新 binary 打开。
- `projection.py` 中重复 configuration 构造、重复错误映射、重复 progress 推算。
- `app.js`/`commands.js` 中 demo 与真实 API 的双重 command 路径；demo 若保留则单独构建入口，不进入生产 bundle。
- `uncertainty.py` 只有在产品明确不交付 UQ 时才连同测试、artifact allow-list 一起删除；不能只删模块留下交付断言。

### Phase 7：交付、测试与部署

**Files:** `Makefile`, `scripts/build_delivery.sh`, `scripts/verify_delivery.sh`, `scripts/verify_artifacts.py`, `scripts/install_dsh_ecology_runtime.mjs`, `plugins/ecology_evolution/test/smoke.mjs`, `.runtime/launch.sh`（改为受控脚本，不提交运行数据）

- [ ] 测试分三层：快速合同/纯函数（<10s）、本地 fake DSH 集成（<60s）、受控 64/128 并发压力与清洁打包（单独 job）。
- [ ] 引入虚拟时钟 fake provider，覆盖 queue timeout、abort、drain、retry、AIMD，不使用真实 10 分钟等待。
- [ ] 增加属性测试：occurrence cycling、digest determinism、cohort partition、request count conservation、idempotency。
- [ ] 清洁临时目录执行 sdist/wheel/plugin 构建；构建前删除 stale egg-info，确保根目录忽略文档不进入 sdist；verifier 不再靠不断追加 exclude 例外。
- [ ] plugin smoke 从 `preset-manifest.json` 派生期望集合，并同时检查 HTML 参数默认值、API v2 字段、command polling 和真实 progress 状态。
- [ ] 增加 `dev-up/dev-down/dev-status/dev-restart/dev-logs`：检测 8777/8848 端口冲突，写 PID/lock，任一进程退出时联动停止；先停新任务，再 drain，最后关 SQLite。
- [ ] 增加 `/health/live` 与 `/health/ready`：ready 同时检查 SQLite integrity、DSH capabilities、plugin manifest digest、数据目录和 provider gate；不要用 HTTP 200 静态页冒充 ready。
- [ ] 增加结构化 JSON 日志和 trace id：贯穿 run、generation、candidate、revision、work unit、origin occurrence、provider request；日志永不输出 token/原始标签。

## 4. 验收门槛

实现完成前必须全部满足：

1. 真实 progress 每 30 秒内至少产生一次可解释 heartbeat；不存在“百分比不变且无 queue/active/retry/deadline 事件”的窗口。
2. 任意时刻 `provider_active + draining <= provider_limit`，任意 run 的 `origin_chain_active <= configured_run_limit`；candidate concurrency 不会放大物理上限。
3. provider 排队、workflow 执行、持久化分别可见；`planned = sum(all lifecycle states)`。
4. pause/cancel/archive/create 的 HTTP receipt ≤1 秒；最终结果可由 command id 查询；相同幂等键不会重复副作用。
5. 取消后新请求全部拒绝；底层可中断请求释放 slot，不可中断请求进入有上限的 draining，绝不无限占用。
6. 一个起点永远完整提交 9-cell；500 起点由 10 个 50 起点 batch 组成；local edit 只能影响下一 batch。
7. 数据循环复用不增加独立样本数；报告 source count、occurrence count、effective N 和 reuse fraction；promotion bootstrap 按 source cluster 解释。
8. 外层四候选 Top-2 golden fixture 与当前结果完全一致；任何内层 revision 变化不改变 Top-2 选择合同。
9. 清洁 `make test`、Node smoke、`make verify` 和受控压测全部通过；测试失败时不得生成可交付 artifact。

## 5. 推荐落地顺序

1. 先做 Phase 0–2：deadline、取消、真实队列、reservation 索引。这四项直接解决 7% 卡住和大量排队。
2. 再做 Phase 3：durable work unit 和重启恢复，确保一次 batch 中断不会从头重跑。
3. 做 Phase 4：把局部进化与轮末比较的统计意义固定下来，尤其是循环复用后的 source-cluster evidence。
4. 做 Phase 5：统一异步命令和前端状态，消除“请求超时但后台已执行”的错觉。
5. 最后做 Phase 6–7：删旧代码、拆单体、清洁打包、supervisor 和交付验收。

每一阶段都先添加失败测试，再实现，再运行对应层级的测试；不要在旧协议和新协议之间做双写/双读。完成 Phase 7 后才允许重新启动 8777/8848 并发起新的真实自进化运行。
