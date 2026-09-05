# EcologyRSI-DSH 架构与冗余治理实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前可运行但职责交叉的 Host、账本、投影、DSH 运行时和浏览器代码收敛为单一状态模型、单一错误合同和单一投影来源，降低维护成本并消除高并发下的隐性性能风险。

**Architecture:** 保留 SQLite 追加事件账本和 DSH-native 执行边界，但把“事件写入/状态回放/领域编排/HTTP 适配/展示投影”分层。所有公共响应由版本化 schema builder 生成；所有错误先转换为 Host-owned `ErrorCode`，再由 HTTP/DSH/浏览器分别映射。provider admission 和 run work-unit scheduler 保持独立，前者只负责 provider 物理并发，后者只负责 run 逻辑预算。

**Tech Stack:** Python 3.10+ 标准库、SQLite WAL、Node.js 原生测试、Vanilla JavaScript；不增加运行时依赖，不引入新的模型族或控制系统。

**Spec:** 本计划基于 2026-09-05 全局代码审查；当前基线为 Python 1290 tests、Node 178 tests、`make verify` 通过。

## Global Constraints

- 不改变四候选 Top-2 排序、同 cohort 比较和 champion/challenger 晋级规则。
- 一个 origin 始终原子地产生完整 9-cell 结果；重复 source 必须通过 `OriginOccurrence` 表达。
- 新服务只读新协议和新 projection；历史事件仅通过显式离线导入工具进入，不在热路径猜测旧字段。
- provider 物理并发上限保持 128；run 逻辑起点预算和 provider 并发不得相乘。
- 浏览器不得推进 server-owned continuous run；浏览器只读 projection 和 command receipt。
- 所有新改动必须先添加失败测试，再实现，再运行受影响的快速测试和完整回归。

## 审查结论

1. `core/director.py`（6126 行）同时承担运行生命周期、种子物化、候选编排、DSH 合同和恢复；`core/state.py`（4785 行）同时承担事件解码、跨域校验和状态索引，任何协议变化都会扩大回归范围。
2. `api/handler.py`（4169 行）把路由、认证、锁、幂等收据、DSH sidecar 和错误映射放在一个请求处理器中；`do_POST` 的控制分支与命令执行分支仍共享大量隐含状态。
3. `api/projection.py`（5972 行）在完整运行投影和 summary 投影中分别构造 configuration（约第 5473、5781 行），并同时输出顶层旧字段和嵌套字段，存在字段漂移风险。
4. `evaluators/sample_execution.py`、`evaluators/gateway_sample_adapter.py`、`evolution/strategies.py`、`evolution/analysis.py` 均为 3800–4700 行，且 `plan_batch`、`predict_sample`、`_finite`、`_sha256`、`_text` 等职责或校验重复。
5. `api/dsh_tools.py` 在 child reservation 热路径持有全局 `_launch_lock` 并反复调用 `ledger.events(run_id)` 全量加载事件；事件量增加后延迟近似按事件数增长。
6. `api/auto_progress.py` 和 `generation_execution.py` 大量捕获 `Exception`/`BaseException`，虽有部分重试分类，但仍可能把编程错误包装成可重试运行故障，导致“暂停后自动重试”或故障原因丢失。
7. `_public_http_error`、`_dsh_sidecar_error`、多个 endpoint 的异常映射并非同一来源；部分路径只有字符串 `error`，没有稳定 `error_code`。
8. `.runtime/` 当前约 569 MB 且全部被忽略，缺少明确的运行数据保留、归档和清理策略；近 14 天约 138 个提交，版本与交付包容易出现源码/归档不同步。
9. 测试数量充分，但没有系统的属性测试、虚拟时钟 provider 压测和复杂度门槛；若不增加这些测试，重构后仍可能在 64/128 并发和长事件流下退化。

## 实施阶段

### Phase 1：冻结公共合同与错误模型

**Files:**
- Create: `src/ecologyrsi_dsh/api/contracts.py`
- Create: `src/ecologyrsi_dsh/api/errors.py`
- Modify: `src/ecologyrsi_dsh/api/shared.py`, `src/ecologyrsi_dsh/api/handler.py`, `src/ecologyrsi_dsh/api/transport.py`, `src/ecologyrsi_dsh/api/execution.py`
- Test: `tests/test_api_contracts.py`, `tests/test_http.py`, `tests/test_public_redaction.py`

**Interfaces:**
- `ErrorCode` 枚举至少包含 `invalid_request`、`not_found`、`command_in_progress`、`runtime_unavailable`、`provider_queue_timeout`、`draining_timeout`、`stale_manifest`、`internal_error`。
- `error_payload(code: ErrorCode, *, retryable: bool, command_id: str | None = None) -> dict[str, object]` 是唯一公共错误序列化入口。
- `build_command_receipt(receipt: CommandReceipt, *, response: Mapping[str, object] | None = None) -> dict[str, object]` 是唯一命令收据序列化入口。

- [ ] 先为每个公开异常场景添加 `error_code` 断言，确认 sidecar、HTTP、command polling 不再依赖中文文案。
- [ ] 将 `_public_http_error`、`_dsh_sidecar_error` 和 `_send_post_error` 改为调用统一 mapper；异常原文只能进入结构化内部日志，不能进入响应。
- [ ] 保留现有状态码语义，但所有 4xx/5xx JSON 都包含稳定 `error_code` 和可选 `retryable`。
- [ ] 运行 `PYTHONPATH=src .venv/bin/python -m unittest tests.test_api_contracts tests.test_http tests.test_public_redaction`。

### Phase 2：建立唯一 projection builder

**Files:**
- Create: `src/ecologyrsi_dsh/api/run_projection.py`
- Create: `src/ecologyrsi_dsh/api/progress_projection.py`
- Create: `src/ecologyrsi_dsh/api/evidence_projection.py`
- Modify: `src/ecologyrsi_dsh/api/projection.py`, `src/ecologyrsi_dsh/api/catalog.py`, `src/ecologyrsi_dsh/api/handler.py`
- Test: `tests/test_projection_contract.py`, `tests/test_execution_projection.py`, `tests/test_http.py`

**Interfaces:**
- `build_configuration(task: TaskManifest, state: RunState) -> dict[str, object]`。
- `build_run_summary(state: RunState) -> dict[str, object]`。
- `build_progress_projection(state: RunState, *, admission: Mapping[str, object] | None) -> dict[str, object]`。
- `build_evidence_projection(state: RunState) -> dict[str, object]`。

- [ ] 先写测试，要求完整 run 和 summary 的 `configuration` 键集合相同，且不再同时输出同义旧字段。
- [ ] 把 70 键完整 configuration 和 26 键 summary configuration 合并到 `build_configuration`，按 projection profile 只裁剪展示层，不重复定义字段。
- [ ] 统一 progress 计数：`planned == local_waiting + admission_waiting + provider_queued + provider_active + workflow_running + persisting + completed + failed + retry_waiting + cancelled + draining`。
- [ ] 让 monitor 只调用 compact builder，详情页再调用 evidence builder；禁止在 monitor 读取训练资产和完整候选证据。
- [ ] 运行 projection、HTTP、Node smoke 全套合同测试。

### Phase 3：拆分账本回放和状态索引

**Files:**
- Create: `src/ecologyrsi_dsh/core/event_replay.py`
- Create: `src/ecologyrsi_dsh/core/state_indexes.py`
- Create: `src/ecologyrsi_dsh/core/invariant_checks.py`
- Modify: `src/ecologyrsi_dsh/core/state.py`, `src/ecologyrsi_dsh/core/director.py`, `src/ecologyrsi_dsh/core/ledger.py`
- Test: `tests/test_state_replay.py`, `tests/test_core.py`, `tests/test_recovery_security.py`

**Interfaces:**
- `replay_events(events: Sequence[Event], *, strict: bool = True) -> RunState`。
- `RunStateIndex.from_events(events: Sequence[Event]) -> RunStateIndex`。
- `assert_state_invariants(state: RunState) -> None`。
- `EventLedger.events_after(run_id: str, seq: int, *, limit: int | None = None) -> tuple[Event, ...]`。

- [ ] 为长事件流添加测试：读取最新状态只能查询 tail 和增量，不得每次 poll 重新加载整个 run。
- [ ] 将 `state.py` 中 DSH skill、retry、trajectory、knowledge 的校验函数分别迁移到领域模块；state 只组合索引和调用 invariant checks。
- [ ] 将 `director.py` 的生命周期写入、候选生命周期和 trajectory 生命周期拆为三个 orchestrator，Director 只负责依赖注入和事务边界。
- [ ] 以 `latest_run_seq`、`event_by_id` 和按 kind/identity 的索引替换热路径全量 `ledger.events`。
- [ ] 运行完整 core、recovery、trajectory、work-unit 测试，并对 10 倍事件量做基准。

### Phase 4：收敛 reservation、provider 和 work-unit 边界

**Files:**
- Create: `src/ecologyrsi_dsh/api/reservation_index.py`
- Create: `integrations/dsh_ecology_plugin/lib/runtime/request-lifecycle.js`
- Modify: `src/ecologyrsi_dsh/api/dsh_tools.py`, `src/ecologyrsi_dsh/api/sample_admission.py`, `src/ecologyrsi_dsh/evaluators/sample_execution.py`, `integrations/dsh_ecology_plugin/lib/runtime/provider-stage-gate.js`, `integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js`
- Test: `tests/test_dsh_reconciliation.py`, `tests/test_sample_admission.py`, `integrations/dsh_ecology_plugin/test/provider_stage_gate.test.mjs`, `integrations/dsh_ecology_plugin/test/stage_runner.test.mjs`

- [ ] 先增加虚拟时钟测试，验证 queue timeout、abort、draining、retry 的 deadline 全部覆盖排队到持久化。
- [ ] 用 `(run_id, business_key_digest)` 分片锁和 reservation index 取代 `_launch_lock + ledger.events(run_id)` 全扫描。
- [ ] 将 request lifecycle 记录统一为 `planned → provider_queued → provider_active → persisting → completed|failed|draining`，Python 和 Node 使用同一字段名。
- [ ] 保证 provider gate 只管理 128 物理上限，run admission 只管理 64 逻辑上限；删除重复自适应调整。
- [ ] 为 work unit 统一 `lease_owner`、`lease_expires_at`、`attempt`、`checkpoint_digest`，重启只恢复未过期且 manifest digest 一致的租约。

### Phase 5：拆分进化编排与执行适配器

**Files:**
- Create: `src/ecologyrsi_dsh/api/screening_orchestrator.py`
- Create: `src/ecologyrsi_dsh/api/holdout_orchestrator.py`
- Create: `src/ecologyrsi_dsh/api/decision_barrier.py`
- Create: `src/ecologyrsi_dsh/evaluators/origin_chain.py`
- Create: `src/ecologyrsi_dsh/evaluators/result_publisher.py`
- Create: `src/ecologyrsi_dsh/evolution/proposal_contract.py`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py`, `src/ecologyrsi_dsh/api/formal_trajectory.py`, `src/ecologyrsi_dsh/evaluators/sample_execution.py`, `src/ecologyrsi_dsh/evaluators/gateway_sample_adapter.py`, `src/ecologyrsi_dsh/evolution/strategies.py`, `src/ecologyrsi_dsh/evolution/analysis.py`
- Test: `tests/test_generation_control_execution.py`, `tests/test_formal_trajectory.py`, `tests/test_work_units.py`, `tests/test_sample_execution.py`

- [ ] 把 screening、finalist batch、holdout、generation comparison 的输入输出定义成 typed dataclass，删除跨阶段共享可变 dict。
- [ ] 将 gateway adapter、DSH adapter、host fallback 收敛为同一 `OriginChainExecutor` 接口；新协议路径只允许一个实际执行器。
- [ ] 将 proposal/analysis 中重复的有限数、整数、digest 和 mutation 校验改为共享验证器。
- [ ] 每个 batch 固定 `BatchEvidenceRecorded → LocalEditProposed → LocalEditDecided → RevisionActivated` 事件链；KEEP 也必须写事件。
- [ ] 验证同 batch 修改不能影响当前 batch 评分，失败重试只重放同一个 work unit，不重复消耗已完成 occurrence。

### Phase 6：统一浏览器状态和运行数据治理

**Files:**
- Create: `plugins/ecology_evolution/assets/js/api_client.js`
- Create: `plugins/ecology_evolution/assets/js/state_store.js`
- Create: `plugins/ecology_evolution/assets/js/render_scheduler.js`
- Modify: `plugins/ecology_evolution/assets/js/commands.js`, `plugins/ecology_evolution/assets/js/data.js`, `plugins/ecology_evolution/assets/js/host.js`, `plugins/ecology_evolution/assets/js/render_process.js`
- Create: `scripts/runtime_gc.py`
- Modify: `README.md`, `.gitignore`, `Makefile`
- Test: `plugins/ecology_evolution/test/smoke.mjs`, `tests/test_run_cleanup.py`, `tests/test_delivery_scripts.py`

- [ ] 浏览器 API client 区分 command timeout 和 data timeout；超时后用 command id 查询，不直接显示失败。
- [ ] 新协议下移除 browser auto-advance 写路径，只保留 server monitor；所有页面使用同一 state store 和 render scheduler。
- [ ] `runtime_gc.py` 按 run 状态、归档时间和最大总容量清理 `.runtime`，清理前生成只读 manifest；默认保留最近 7 天和所有未完成运行。
- [ ] 将 plugin 文件清单、preset manifest 和 delivery verifier 的重复列表改为从一个 manifest 生成。
- [ ] 增加 `dev-up/dev-down/dev-status/dev-restart/dev-logs`，端口冲突、PID、SQLite 关闭和 provider drain 均有明确状态。

### Phase 7：质量门槛与发布节奏

**Files:**
- Create: `tests/property/test_occurrence_properties.py`
- Create: `tests/benchmarks/test_ledger_hotpaths.py`
- Modify: `Makefile`, `scripts/verify_delivery.sh`, `scripts/build_delivery.sh`, `RELEASE-CHECKLIST.md`

- [ ] 增加 occurrence cycling、digest determinism、request-count conservation、idempotency 的属性测试；没有第三方依赖时使用标准库随机种子生成器。
- [ ] 增加虚拟时钟 provider benchmark，记录 64/128 并发、10 倍事件流下的 P50/P95 reservation 与 projection 延迟。
- [ ] `make test-fast` 运行纯函数/合同测试，`make test-integration` 运行 fake DSH，`make test-full` 运行全量，`make verify` 额外执行清洁打包。
- [ ] 发布前只允许一个版本号、一个 preset manifest digest、一个 plugin archive；源码与归档必须由同一构建步骤生成。
- [ ] 复杂度门槛：核心单文件不超过 2500 行；超过时必须说明拆分边界并新增模块级测试。

## 验收标准

1. 完整 projection、summary、monitor 使用同一 configuration builder，字段集合和含义一致。
2. 任意 provider 请求都能查询 queued/active/persisting/draining，且 `planned` 与生命周期计数守恒。
3. 同一 reservation 在 10 倍事件量下的延迟不随事件总量线性增长；同一业务键仍只产生一个有效 launch。
4. pause/cancel/archive/create 在服务端先返回 command receipt，浏览器超时不会误报最终失败；相同幂等键不会产生重复副作用。
5. 程序员异常不会被自动重试器吞掉；所有可重试错误都有稳定 code、attempt 和 deadline。
6. `.runtime` 有上限和可恢复归档策略，清理不影响未完成运行和审计导出。
7. Python、Node、交付构建和基准测试全部通过；Top-2 golden fixture 与现有结果一致。

