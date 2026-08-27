# Top-2 Adaptive Epoch Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在完全保留“每轮 4 个候选、同一 64 时点初筛、现有规则选 Top 2”的前提下，将每个 finalist 的一次性 500 时点评测替换为可恢复的 `10 x 50` 局部持续进化，并以 F1/F2/上一冠军在同一 169 时点上的配对结果决定下一冠军。

**Architecture:** 外层 `Candidate` 与 Top-2 筛选保持不变；Top-2 内部新增不可变 `CandidateRevision`、`FormalTrajectory`、`FormalBatch` 和三臂 `GenerationHoldout`。每条 finalist lane 内部串行、两条 lane 之间并行；每批只产生供下一批使用的有界 local-edit，只有轮末同 cohort 比较可以晋级冠军。自动推进改成一次推进一个 durable work unit，样本请求受 run 级 64 和 provider-route 级 128 两层真实并发约束。

**Tech Stack:** Python 3.10–3.12、stdlib dataclass/Enum/threading/unittest、append-only SQLite event ledger、Vanilla HTML/CSS/JavaScript、Node.js ESM/node:test、DSH Cordis plugin structured roles。仓库当前 `.venv` 不含 pytest，所有 Python 命令必须使用既有 `unittest` runner；不得把安装 pytest 当成隐含前置条件。

**Spec:** `docs/superpowers/specs/2026-08-27-top2-adaptive-epoch-design.md`

## Global Constraints

- 外层行为固定为 `4 candidates -> same 64 origins -> existing deterministic Top 2`；不得改变 `_select_screening_finalists()` 的排序、tie-break 或 screened-out 语义。
- `500`、`50`、`169` 的单位都是完整 forecast origins；当前评分单元仅由 `origin_count * 9` 派生。
- 每个 Top-2 finalist 各自处理 500 origins；两个 finalist 合计 1,000 candidate-origin executions，不得把 500 描述成 run-wide 总量。
- 本计划中的一个 adaptive epoch 是“一个 finalist 在一个 outer generation 内的 500-origin trajectory”；因此每轮并行存在两个 epoch，但外层仍只做一次 Top-2 与一次冠军决策。
- local batch 默认 50，且必须整除 finalist formal origins；默认 10 批。
- 每批 local edit 上限默认 2、可配置 1–5；模型可以返回 `KEEP`，即实际 0 个修改。
- 外层 candidate proposer 仍为当前单轴 trust-region mutation；多操作仅属于新的 `candidate.local_edit` 契约。
- local edit 只影响下一批；禁止用已经观察的同一批重新评分新 revision。
- batch 级结论是 provisional；冠军只能由 F1/F2/incumbent 在相同 holdout 上的 Host 门禁产生。
- 每轮 selection holdout 默认 169，不能与本轮 screening/adaptation 重叠，不能冒充 final validation/test。
- 样本并发默认 64、最大 128；两个 finalist 共享同一 run limit，candidate concurrency 不能乘大实际样本并发。
- 新协议不回放、不恢复旧 run；切换前只读归档旧数据库。唯一例外是与 active runtime import graph 隔离的 offline static-archive reader，它不能构造/推进 run。保留外层筛选源码，删除旧 one-shot formal-500 和相关执行兼容分支。
- raw labels、raw predictions、完整 timestamps、credentials、自由文本 rationale 不得进入 local editor 或下一轮 proposer。
- 不在本计划中增加 CO2 专用模型，也不增加分目标/分时距模型族。
- 每个行为变更必须遵循 RED -> GREEN -> REFACTOR；先运行指定失败测试，再修改生产代码。
- formal batch index 在 state/event/retry key 中统一为 `0..batch_count-1`；页面统一显示为 `1..batch_count`，不得混用。
- 工作目录固定为 `/Users/jiezhou/Desktop/工作/项目申请/农业-生态世界模型/EcologyRSI-DSH-evolution-stability`，不得在相邻 `EcologyRSI-DSH` 目录执行实现。

---

## 文件结构锁定

### 新建的聚焦模块

| 文件 | 单一职责 |
|---|---|
| `src/ecologyrsi_dsh/evolution/schedule.py` | 新协议 schedule、origin 单位校验和预算派生 |
| `src/ecologyrsi_dsh/core/trajectory.py` | revision/trajectory/batch/holdout/comparison 不可变类型 |
| `src/ecologyrsi_dsh/evaluators/epoch_cohorts.py` | run adaptation cohort 和 generation screening/holdout cohort 冻结 |
| `src/ecologyrsi_dsh/evolution/local_edits.py` | local-edit schema、Host 校验、原子应用和 revision 生成 |
| `src/ecologyrsi_dsh/evaluators/generation_comparison.py` | 三臂配对评测、硬门禁和确定性冠军排序 |
| `src/ecologyrsi_dsh/api/formal_trajectory.py` | 两条 10 x 50 lane 的单 work-unit 执行 |
| `src/ecologyrsi_dsh/api/work_units.py` | 从 RunState 选择并执行下一个 durable work unit |

### 现有大文件只保留编排入口

- `api/generation_execution.py` 保留 candidate generation、64 screening、Top-2 冻结和 generation 总入口；formal trajectory 细节移出。
- `evaluators/sample_execution.py` 保留完整 origin 原子提交，但只消费显式 `EvaluationScope`，不再推断 formal 500。
- `evolution/strategies.py` 保留外层候选生成；local edit 仅通过新模块/DSH stage 接入。
- `core/state.py` 只做事件严格回放和索引，不实现科学比较。
- `api/projection.py` 只把 durable state 投影给前端，不重新推断执行状态。

### 旧代码到新代码的迁移矩阵

| 当前代码/行为 | 处理方式 | 新代码/新职责 | 完成任务 |
|---|---|---|---|
| `api/handler.py` 接收 `samples_per_update`，把评分单元反算成预测时点 | 删除公开字段和反算路径 | `OptimizationSchedule` 直接冻结 500/50/169 origin counts；handler 只校验、绑定和写 manifest | Task 1、2、14 |
| `generation_execution._select_screening_finalists()` | 原样保留，并先做 golden fixture | 仍以当前 score、约束和 tie-break 从同一 64-origin screening 选 Top 2 | Task 0、14、15 |
| `_screen_candidate()` 与 `_prepare_formal_finalists()` | 保留外层语义，只改为消费显式 scope/cohort identity | 仍完成四候选初筛、Top-2 冻结和另外两候选 `screened_out` | Task 4、5、6、14 |
| `_phase_task_manifest(..., "formal")` 把一次 formal 绑定成一个 500-origin task | 删除 formal one-shot 分支 | `FormalTrajectory` 把每个 finalist 拆成 10 个顺序 50-origin batch | Task 6、9、14 |
| `_evaluate_candidate()` 用一个不可变算法跑完整个 formal pass | 删除 formal-stage 单次入口；screening 所需路径保留 | `formal_trajectory.execute_next_formal_batch()` 与 `execute_next_local_edit()` 交替推进 revision lineage | Task 8、9、14 |
| `strategies.py` 外层 proposer 的 `maximum_operations=1` | 原样保留并加回归测试 | 新增独立 `candidate.local_edit` 角色，读取每批 1–5 上限，允许 `KEEP=0` | Task 8 |
| evaluator/checkpoint 主要按 candidate 或旧 phase key 索引 | 替换为完整不可变作用域 | `EvaluationScope(run,generation,candidate,revision,phase,batch/arm,cohort)` 统一 artifact、sample、usage 与恢复 | Task 3、4、6 |
| feedback-update cohort 的旋转/wrap 逻辑承担 formal 样本选择 | 新协议不再调用，最终删除无调用兼容分支 | `epoch_cohorts.py` 冻结 run adaptation cohort、每轮 screening 和新 holdout，并做 causal capacity proof | Task 5、14 |
| `RunSampleAdmission` 与 DSH `provider-stage-gate` | 复用并补两条 lane 的峰值测试，不再另造 limiter | 同一 run 的 screening/lane/holdout 共用 64；所有 run 的同 provider route 物理上限 128 | Task 7 |
| `auto_progress._run_one_generation_locked()` 一次占用 worker 直到整轮完成 | 替换 | `next_work_unit()` 纯函数选择下一 durable unit；执行一次即落盘、释放 lease、重新排队 | Task 11 |
| `projection.py`/前端从旧 CandidateStatus 和 `samples_per_update` 推断 formal 进度 | 删除重复推断和旧静态文案 | 后端统一投影两条 trajectory、batch/revision/edit、三臂 holdout 和 candidate-origin 预算 | Task 12、14 |
| champion 只绑定 outer candidate/初始 genome | 替换为 final-revision identity | `CandidateEffectiveRevisionFrozen` 成为下一轮 parent、judge、artifact、promotion 和 seal 的唯一来源 | Task 10 |
| 旧 ledger 可被新 binary 打开/升级 | 删除自动兼容 | 旧库先静态归档；新 binary 在任何 WAL/DDL 前拒绝非 schema 8/event identity 数据库 | Task 13、14、15 |

实现顺序必须沿表中任务推进：先冻结不变的 Top-2 行为，再建立新 schedule/identity/cohort，随后接 evaluator/local edit/trajectory/holdout，最后切换 scheduler/UI/ledger。不得先删旧 formal 路径再补新状态机，否则中间提交无法恢复或验证。

---

### Task 0: 建立可恢复基线并冻结 Top-2 golden behavior

**Files:**
- Read: `src/ecologyrsi_dsh/api/generation_execution.py:72-330`
- Modify: `tests/test_candidate_parallel_evaluation.py`
- Create: `tests/fixtures/top2_screening_golden.json`
- Verify: current Python, Node, browser smoke, delivery checks

**Interfaces:**
- Consumes: 当前 `_select_screening_finalists(candidates, screening, top_k=2)`。
- Produces: 新旧实现共用的 Top-2 golden fixture；后续任务不得修改 fixture 期望 ID。

- [ ] **Step 1: 记录干净工作树和当前版本**

Run:

```bash
git status --short --branch
git rev-parse HEAD
```

Expected: 只有本 spec/plan 文档为未提交文件；记录当前 HEAD。若有用户代码改动，先停止并审查重叠文件，不得覆盖。

- [ ] **Step 2: 为现有筛选结果写 golden test**

先创建 fixture 目录（当前仓库尚无该目录）：

```bash
mkdir -p tests/fixtures
```

在 fixture 中固定四个候选的 `slot_index`、score、constraint violations 和 record digest，并在测试中断言：

```python
class Top2GoldenTests(unittest.TestCase):
    def test_top2_screening_selection_is_a_locked_outer_contract(self):
        candidates, records = load_top2_screening_golden()
        selected = _select_screening_finalists(candidates, records, top_k=2)
        self.assertEqual(
            [item.candidate_id for item in selected],
            ["candidate:golden:1", "candidate:golden:3"],
        )
```

不要在此任务修复或重新解释 `passed` 字段；用户要求选择逻辑不变。

- [ ] **Step 3: 运行 golden test 并确认当前代码为 GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_candidate_parallel_evaluation
```

Expected: PASS。这个测试是后续重构的外层行为护栏。

- [ ] **Step 4: 运行完整基线**

Run:

```bash
make test
node --test integrations/dsh_ecology_plugin/test/*.test.mjs
node plugins/ecology_evolution/test/smoke.mjs
make verify
```

Expected: 全部通过；记录 test count 和耗时。任何基线失败必须先单独诊断，不能归入新协议实现。

- [ ] **Step 5: 提交基线护栏**

```bash
git add tests/test_candidate_parallel_evaluation.py \
  tests/fixtures/top2_screening_golden.json
git commit -m "test: lock existing top2 screening behavior"
```

---

### Task 1: 建立 origin-first OptimizationSchedule 和后端创建契约

**Files:**
- Create: `src/ecologyrsi_dsh/evolution/schedule.py`
- Create: `tests/test_optimization_schedule.py`
- Modify: `src/ecologyrsi_dsh/evolution/__init__.py`
- Modify: `src/ecologyrsi_dsh/api/handler.py:184-205,1390-1500,1700-1780,2337-2505`
- Modify: `tests/test_http.py`
- Modify: `tests/test_runtime_integration.py`
- Modify: `tests/test_dsh_native_runtime.py`

**Interfaces:**
- Produces: `OptimizationSchedule.from_dict(value)`, `.default()`, `.to_dict()`, `.batch_count`, `.max_local_edits_per_finalist`, `.generation_execution_budget(cells_per_origin)`, `.run_execution_budget(planned_generations, cells_per_origin)`, `.required_unique_origins(planned_generations)`。
- Produces: manifest metadata keys `optimization_protocol="top2_adaptive_epoch@1"` and `optimization_schedule` with schema `ecologyrsi-dsh.top2-adaptive-epoch-schedule/1`。
- Removes from public create contract: `samples_per_update`。
- Keeps public `sample_concurrency` as the single run-level in-flight control，omission freezes 64 and explicit values validate as integer 1–128。

- [ ] **Step 1: 写 schedule RED tests**

```python
class OptimizationScheduleTests(unittest.TestCase):
    def test_default_schedule_uses_origin_units(self):
        schedule = OptimizationSchedule.default()
        self.assertEqual(schedule.formal_origin_count_per_finalist, 500)
        self.assertEqual(schedule.local_batch_origin_count, 50)
        self.assertEqual(schedule.batch_count, 10)
        self.assertEqual(schedule.max_local_edits_per_batch, 2)
        self.assertEqual(schedule.selection_holdout_origin_count, 169)
        self.assertEqual(
            schedule.generation_execution_budget(cells_per_origin=9),
            {
                "screening_candidate_origins": 256,
                "formal_candidate_origins": 1000,
                "holdout_candidate_origins": 507,
                "total_candidate_origins": 1763,
                "total_scoring_cells": 15867,
            },
        )
        self.assertEqual(
            schedule.run_execution_budget(5, cells_per_origin=9)[
                "total_candidate_origins"
            ],
            8815,
        )
        self.assertEqual(schedule.required_unique_origins(5), 1665)

    def test_schedule_rejects_invalid_values(self):
        cases = [
            ({"local_batch_origin_count": 64}, "must divide"),
            ({"max_local_edits_per_batch": 0}, "between 1 and 5"),
            ({"max_local_edits_per_batch": 6}, "between 1 and 5"),
            ({"selection_holdout_origin_count": 168}, "at least 169"),
            ({"formal_origin_count_per_finalist": True}, "must be an integer"),
        ]
        for patch, message in cases:
            with self.subTest(patch=patch):
                value = {**OptimizationSchedule.default().to_dict(), **patch}
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    OptimizationSchedule.from_dict(value)
```

- [ ] **Step 2: 运行 RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_optimization_schedule
```

Expected: FAIL，模块尚不存在。

- [ ] **Step 3: 实现不可变 schedule**

实现精确字段：

```python
@dataclass(frozen=True, slots=True)
class OptimizationSchedule:
    schema_version: str
    screening_origin_count: int
    finalist_count: int
    formal_origin_count_per_finalist: int
    local_batch_origin_count: int
    max_local_edits_per_batch: int
    selection_holdout_origin_count: int
    local_evaluation_mode: str

    @property
    def batch_count(self) -> int:
        return self.formal_origin_count_per_finalist // self.local_batch_origin_count

    @property
    def max_local_edits_per_finalist(self) -> int:
        return self.batch_count * self.max_local_edits_per_batch
```

`from_dict()` 使用 exact-field validation；固定 `screening_origin_count=64`、`finalist_count=2`、`local_evaluation_mode="prequential"`，拒绝额外字段和 bool-as-int。

- [ ] **Step 4: 写 HTTP/manifest RED tests**

新增断言：新 create body 的 `optimization_schedule` 被规范化后完整写入 `TaskManifest.metadata`；四个可编辑值任一变化都会改变 manifest digest；`samples_per_update` 请求得到 400：

```python
class NewRunContractTests(unittest.TestCase):
    def test_new_protocol_rejects_samples_per_update(self):
        response = self.client.post_json("/runs", {
            **valid_new_run_request(),
            "samples_per_update": 4500,
        })
        self.assertEqual(response.status, 400)
        self.assertIn("samples_per_update is not supported", response.json["error"])
```

- [ ] **Step 5: 修改 handler 创建与 binding**

- 在 `PLUGIN_MANIFEST.request_schema` 中加入 `optimization_schedule`，移除 `samples_per_update`。
- compact/full-manifest 两条路径都调用同一个 `OptimizationSchedule.from_dict()`。
- 保留 `execution_protocol="dsh_native_plugin_evolution@1"`，因为它标识 DSH sidecar transport/runtime；新增并固定 `optimization_protocol="top2_adaptive_epoch@1"` 标识本次不兼容的进化行为。
- `_bind_runtime_task()` 在本任务校验 evaluator holdout minimum、`candidates_per_generation==4` 并冻结 schedule dict；dataset capacity 必须由 Task 5 的真实 causal cohort planner 校验，不能用 catalog `row_count` 粗算。
- compact/full-manifest 两条创建路径对 `sample_concurrency` 使用同一 validator：省略=64、bool/0/129=400、最大合法128；不得再用 candidate concurrency 把它截断到8。
- 保留 `prediction_cells_per_origin` 作为 evaluator-derived metadata；不再把 500 乘 9 后写回用户字段。
- 只为内部显示写入 `derived_execution_budget`，它不是执行 source of truth。

- [ ] **Step 6: 运行 focused GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_optimization_schedule \
  tests.test_http \
  tests.test_runtime_integration \
  tests.test_dsh_native_runtime
```

Expected: PASS。

- [ ] **Step 7: 提交 schedule contract**

```bash
git add src/ecologyrsi_dsh/evolution/schedule.py \
  src/ecologyrsi_dsh/evolution/__init__.py \
  src/ecologyrsi_dsh/api/handler.py \
  tests/test_optimization_schedule.py tests/test_http.py \
  tests/test_runtime_integration.py tests/test_dsh_native_runtime.py
git commit -m "feat: freeze top2 adaptive epoch schedule"
```

---

### Task 2: 重建参数设计页面和 origin/candidate-origin/cell 预算显示

**Files:**
- Modify: `plugins/ecology_evolution/index.html:120-155`
- Modify: `plugins/ecology_evolution/app.js:55-65,145-180`
- Modify: `plugins/ecology_evolution/assets/js/commands.js:212-315`
- Modify: `plugins/ecology_evolution/assets/js/catalog.js:489-565`
- Modify: `plugins/ecology_evolution/assets/js/core.js:919-990`
- Modify: `plugins/ecology_evolution/assets/js/render_shell.js:104-190`
- Modify: `plugins/ecology_evolution/assets/js/render_process.js:1566-1606`
- Modify: `plugins/ecology_evolution/assets/js/demo.js`
- Modify: `plugins/ecology_evolution/styles.css`
- Modify: `plugins/ecology_evolution/test/smoke.mjs`

**Interfaces:**
- Produces form IDs: `formal-origin-count`, `local-batch-origin-count`, `max-local-edits-per-batch`, `selection-holdout-origin-count`。
- Produces request object exactly matching Task 1 `optimization_schedule`。
- Produces dynamic per-generation budget summary `256 + 1000 + 507 = 1763 candidate-origins = 15867 cells` and whole-run budget from the frozen planned generation limit；five-generation defaults are 8815/79335，unique dataset origins are1665。

- [ ] **Step 1: 写前端 RED assertions**

在 `smoke.mjs` 断言 HTML 默认值、read-only policy 和 payload：

```javascript
assert.equal(valueOf("formal-origin-count"), "500");
assert.equal(valueOf("local-batch-origin-count"), "50");
assert.equal(valueOf("max-local-edits-per-batch"), "2");
assert.equal(valueOf("selection-holdout-origin-count"), "169");
assert.equal(valueOf("sample-concurrency"), "64");
assert.equal(maxOf("sample-concurrency"), "128");
assert.equal(lastCreateBody.execution_protocol, "dsh_native_plugin_evolution@1");
assert.equal(lastCreateBody.optimization_protocol, "top2_adaptive_epoch@1");
assert.deepEqual(lastCreateBody.optimization_schedule, {
  schema_version: "ecologyrsi-dsh.top2-adaptive-epoch-schedule/1",
  screening_origin_count: 64,
  finalist_count: 2,
  formal_origin_count_per_finalist: 500,
  local_batch_origin_count: 50,
  max_local_edits_per_batch: 2,
  selection_holdout_origin_count: 169,
  local_evaluation_mode: "prequential"
});
assert.equal("samples_per_update" in lastCreateBody, false);
```

再覆盖 500/64 不整除、edit=0/6、holdout<169、bool/空字符串，以及 sample concurrency 1/64/128 合法、0/129/bool 非法。此任务不在浏览器用 `row_count` 猜测可用 cohort capacity；Task 5 增加服务端 planner truth 后再接 capacity readiness。

- [ ] **Step 2: 运行 RED**

```bash
node plugins/ecology_evolution/test/smoke.mjs
```

Expected: FAIL，页面字段/请求对象尚未存在。

- [ ] **Step 3: 修改参数布局**

将当前单一 `.parameter-grid` 拆成：

```html
<section class="parameter-group" data-group="outer-search">...</section>
<section class="parameter-group" data-group="finalist-adaptation">...</section>
<section class="parameter-group" data-group="throughput">...</section>
<section class="parameter-group" data-group="scientific-policy">...</section>
```

- 保留最大轮数、候选并发、候选总预算。
- 每轮候选数显示为固定 4；增加只读 `64 时点初筛 -> Top 2`。
- 把“请求微批样本数”改名为“网关 origin wave 上限”。
- 新增四个 adaptation 输入及 help text。
- 科学 coverage/完整9单元/物理约束显示为只读 Host policy。

- [ ] **Step 4: 实现共享规范化和跨字段 validity**

在 `commands.js` 只保留一个 schedule 构造器：

```javascript
function normalizedOptimizationSchedule(values) {
  var formal = strictInteger(values.formal_origin_count, "每个候选更新时点数");
  var batch = strictInteger(values.local_batch_origin_count, "局部 batch 时点数");
  var edits = strictInteger(values.max_local_edits_per_batch, "每批最大局部改动数");
  var holdout = strictInteger(values.selection_holdout_origin_count, "轮末比较时点数");
  if (formal % batch !== 0) { throw new Error("局部 batch 必须整除每个候选的更新时点数"); }
  if (edits < 1 || edits > 5) { throw new Error("每批最大局部改动数必须在 1 到 5 之间"); }
  if (holdout < 169) { throw new Error("轮末比较时点数不得低于 169"); }
  return {
    schema_version: "ecologyrsi-dsh.top2-adaptive-epoch-schedule/1",
    screening_origin_count: 64,
    finalist_count: 2,
    formal_origin_count_per_finalist: formal,
    local_batch_origin_count: batch,
    max_local_edits_per_batch: edits,
    selection_holdout_origin_count: holdout,
    local_evaluation_mode: "prequential"
  };
}
```

不要用 `Math.floor()` 静默修正无效小数；输入无效时阻止创建并聚焦对应字段。

- [ ] **Step 5: 修改 create body、normalization 和冻结回显**

- `createRun()` 发送 `optimization_schedule`，删除 `requestedPredictionOrigins * cellsPerOrigin` 和 `samples_per_update`。
- `normalizeRun()` 从 projection 读取冻结 schedule；页面后续输入改变不得修改 active run card。
- `readiness()` 增加整除、范围和 evaluator minimum 检查；capacity 暂显示“创建时由服务端因果 cohort planner 校验”，Task 5 再替换为服务端精确值。
- 动态摘要分别显示 per-generation origin member roles/candidate-origin executions/scoring cells、whole-run planned executions、真正 unique dataset origins、10 batches、每 finalist 最大20 edits；使用当前表单的 frozen planned generation limit，不把 733 误乘成独立数据。

- [ ] **Step 6: 运行 GREEN 和静态检查**

```bash
node plugins/ecology_evolution/test/smoke.mjs
git diff --check
```

Expected: PASS；默认摘要包含 `每个入围候选 500`、`10 x 50`、`1,763`、`15,867`，不得显示“Top 2 各正式评估 500”的旧静态文案。

- [ ] **Step 7: 提交参数页面**

```bash
git add plugins/ecology_evolution
git commit -m "feat: expose adaptive finalist schedule in parameters"
```

---

### Task 3: 增加不可变 revision/trajectory/evaluation scope 核心模型

**Files:**
- Create: `src/ecologyrsi_dsh/core/trajectory.py`
- Create: `tests/test_trajectory_models.py`
- Modify: `src/ecologyrsi_dsh/core/models.py:92-110,379-540`
- Modify: `src/ecologyrsi_dsh/core/__init__.py`

**Interfaces:**
- Produces enums: `EvaluationPhase`, `RevisionStatus`, `TrajectoryStatus`, `LocalEditProposalDecision(KEEP|MUTATE)`, `LocalEditOutcome(KEPT|APPLIED|REJECTED)`, `RevisionAdvanceReason`, `HoldoutArm`。
- Produces dataclasses: `EvaluationScope`, `CandidateRevision`, `FormalTrajectory`, `FormalBatch`, `BatchEvaluation`, `TrajectoryRevisionActivation`, `GenerationHoldout`, `HoldoutEvaluation`, `GenerationComparison`。
- Changes `ModelArtifact` and `Evaluation` to require `candidate_revision_id` and `evaluation_scope` for the new database schema。

- [ ] **Step 1: 写 identity/validation RED tests**

必须覆盖：

```python
class TrajectoryModelTests(unittest.TestCase):
    def test_revision_identity_covers_parent_genome_and_source_batch(self):
        revision = revision_fixture(
            revision_id="revision:child",
            parent_revision_id="revision:parent",
            source_batch_index=3,
        )
        self.assertEqual(revision.candidate_id, "candidate:a")
        self.assertEqual(revision.source_batch_index, 3)
        self.assertEqual(revision.revision_digest, digest(revision.identity_dict()))

    def test_formal_batch_requires_matching_revision_candidate_and_index(self):
        with self.assertRaisesRegex(ValueError, "revision candidate"):
            FormalBatch.from_dict(invalid_cross_candidate_batch())

    def test_holdout_requires_exactly_two_finalists_and_one_incumbent(self):
        with self.assertRaisesRegex(ValueError, "three holdout arms"):
            GenerationHoldout.from_dict(two_arm_holdout())
```

还要测试 bool index、负 batch、非 SHA digest、batch index 超界、holdout role 重复、comparison cohort 不同、可变 nested mapping 被深拷贝。

- [ ] **Step 2: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_trajectory_models
```

Expected: import failure。

- [ ] **Step 3: 实现精确 scope 和模型**

`EvaluationScope` 固定字段：

```python
@dataclass(frozen=True, slots=True)
class EvaluationScope:
    run_id: str
    generation: int
    candidate_id: str
    candidate_revision_id: str
    phase: EvaluationPhase
    cohort_digest: str
    origin_count: int
    batch_index: int | None = None
    holdout_arm: HoldoutArm | None = None

    @property
    def scope_key(self) -> str:
        return digest(self.to_dict())
```

约束：screening 不带 batch/arm；formal batch 必须带 `batch_index`；holdout 必须带 arm；candidate/revision/phase/cohort 全部进入 digest。

`candidate_revision_id` 在所有 phase 都必填：四个 outer candidate 编译成功后、任何 screening origin 发出前，各自写一次 immutable `R0`。screening scope 绑定该 R0；Top-2 只是在后续把已有 R0 接入 trajectory，不重新创建或改变 R0。这样不改变筛选算法，同时消除 candidate-only artifact/checkpoint 身份。

`CandidateRevision` 保存 canonical genome JSON、genome/behavior/mutation digests、parent revision、source batch 和状态，不保存 raw batch evidence。

- [ ] **Step 4: revision-scope Artifact/Evaluation**

在新 schema 中：

```python
class ModelArtifact:
    candidate_id: str
    candidate_revision_id: str
    evaluation_scope_digest: str

class Evaluation:
    candidate_id: str
    candidate_revision_id: str
    evaluation_scope: Mapping[str, Any]
```

删除只按 `candidate_id` 认为只有一个 artifact/evaluation 的新协议假设；下一任务将替换 RunState lookup。

- [ ] **Step 5: 运行模型 GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_trajectory_models
```

Expected: PASS。

- [ ] **Step 6: 提交核心模型**

```bash
git add src/ecologyrsi_dsh/core/trajectory.py \
  src/ecologyrsi_dsh/core/models.py src/ecologyrsi_dsh/core/__init__.py \
  tests/test_trajectory_models.py
git commit -m "feat: add immutable finalist trajectory models"
```

---

### Task 4: 扩展事件账本、Director 命令和严格 replay 状态机

**Files:**
- Create: `tests/test_trajectory_event_replay.py`
- Modify: `src/ecologyrsi_dsh/core/state.py:1178-1300,1446-2750`
- Modify: `src/ecologyrsi_dsh/core/director.py:1900-2700,3000-3420`
- Modify: `src/ecologyrsi_dsh/api/events.py`
- Modify: `tests/test_director_invariants.py`
- Create: `tests/test_api_events.py`

**Interfaces:**
- Produces Director methods listed below with deterministic event IDs。
- Produces RunState indexes for revisions, trajectories, formal batches, batch evaluations, local proposals/outcomes, revision activations, holdouts, holdout evaluations, and comparisons。
- Consumes Task 3 immutable types and Task 1 schedule。

- [ ] **Step 1: 写正常 replay RED test**

使用一个不绕过产品校验的合法 `formal=100,batch=10` schedule，构造完整 10-batch event fixture。顺序从四候选已有 R0 开始：R0、trajectory start、internal batch0 start/evaluate、local KEEP、revision advance、internal batch1...9、最后一次 revision advance、complete、holdout freeze、三臂 evidence、comparison、effective revision、champion。重放后断言：

```python
state = project_run_state(tuple(events))
self.assertEqual(
    state.trajectory_for("candidate:a").final_revision_id,
    "revision:a:1",
)
self.assertEqual(
    state.batch_for_candidate("candidate:a", 0).revision_id,
    "revision:a:0",
)
self.assertEqual(state.comparison_for(0).selected_revision_id, "revision:a:1")
```

- [ ] **Step 2: 写非法顺序/冲突 RED tests**

逐项证明 replay fail-closed：

- 未冻结 Top 2 就启动 trajectory；
- 未创建 revision 就启动 batch；
- revision candidate 与 trajectory 不同；
- batch index 跳号或同一 lane 并行两个 batch；
- `LocalEditDecided` 的 evidence batch 与新 revision source 不一致；
- edit 生成的新 revision 被错误用于同一 evidence batch；
- batch evaluation 后没有且重复写 `TrajectoryRevisionAdvanced`；
- internal batch 9 未完成 decision/advance 就完成 trajectory 或冻结 holdout；
- trajectory 未完成就冻结 holdout；
- F1/F2/incumbent cohort digest 不同；
- 少于三臂就 comparison；
- comparison 前 champion；
- 相同 event ID 不同 payload。

- [ ] **Step 3: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_trajectory_event_replay
```

Expected: FAIL，新事件尚不支持。

- [ ] **Step 4: 实现 RunState 索引和 lookup**

新增 tuple/index 与精确 lookup：

```python
def revision(self, revision_id: str) -> CandidateRevision: ...
def initial_revision_for(self, candidate_id: str) -> CandidateRevision | None: ...
def trajectory_for(self, candidate_id: str) -> FormalTrajectory | None: ...
def formal_batch_for(self, candidate_id: str, batch_index: int) -> FormalBatch | None: ...
def batch_evaluation_for(self, candidate_id: str, batch_index: int) -> BatchEvaluation | None: ...
def revision_activation_for(self, candidate_id: str, batch_index: int) -> TrajectoryRevisionActivation | None: ...
def generation_holdout_for(self, generation: int) -> GenerationHoldout | None: ...
def holdout_evaluation_for(self, generation: int, arm: HoldoutArm) -> HoldoutEvaluation | None: ...
def comparison_for(self, generation: int) -> GenerationComparison | None: ...
```

实现时不得保留 `artifact_for(candidate_id)`/`evaluation_for(candidate_id)` 作为 formal-stage truth；改成 scope/revision lookup。

- [ ] **Step 5: 实现 Director 命令**

精确方法：

```python
start_formal_trajectory(run_id, candidate_id, initial_revision_id, batch_count)
create_candidate_revision(run_id, revision)
start_formal_batch(run_id, candidate_id, revision_id, batch_index, cohort_digest)
record_formal_batch_evaluation(run_id, evaluation)
record_local_edit_proposal(run_id, proposal_payload)
decide_local_edit(run_id, decision_payload)
advance_trajectory_revision(run_id, candidate_id, batch_index, revision_id, reason)
complete_formal_trajectory(run_id, candidate_id, final_revision_id)
freeze_generation_holdout(run_id, generation, cohort_digest, arm_bindings)
record_holdout_evaluation(run_id, evaluation)
record_generation_comparison(run_id, comparison)
select_generation_champion(run_id, generation, selected_revision_id, comparison_digest)
```

每个命令在 append 前校验 RunState；event ID 包含 run/generation/candidate/revision/batch/arm，重复相同输入返回同一事件。

- [ ] **Step 6: 实现 replay 和私有事件说明**

`state.py` 对每种事件做 exact-field/schema/digest/ownership/order 校验；每个 batch evaluation 后必须恰有一个 revision activation，且 internal batch 9 activation 先于 trajectory completion。`api/events.py` 只返回脱敏摘要，例如 UI batch index、revision short ID、proposal decision/Host outcome，不返回 genome、raw evidence 或 sample rows。

- [ ] **Step 7: 运行 GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_trajectory_event_replay \
  tests.test_director_invariants \
  tests.test_api_events
```

Expected: PASS。

- [ ] **Step 8: 提交事件状态机**

```bash
git add src/ecologyrsi_dsh/core/state.py src/ecologyrsi_dsh/core/director.py \
  src/ecologyrsi_dsh/api/events.py tests/test_trajectory_event_replay.py \
  tests/test_director_invariants.py tests/test_api_events.py
git commit -m "feat: persist adaptive trajectory state machine"
```

---

### Task 5: 冻结 run adaptation cohort 与每轮 screening/holdout cohorts

**Files:**
- Create: `src/ecologyrsi_dsh/evaluators/epoch_cohorts.py`
- Create: `tests/test_epoch_cohort_planning.py`
- Modify: `src/ecologyrsi_dsh/data/splits.py`
- Modify: `src/ecologyrsi_dsh/evaluators/registry.py:150-260,2100-2180`
- Modify: `src/ecologyrsi_dsh/api/handler.py`
- Modify: `src/ecologyrsi_dsh/api/catalog.py`
- Modify: `src/ecologyrsi_dsh/core/director.py`
- Modify: `src/ecologyrsi_dsh/core/state.py`
- Modify: `plugins/ecology_evolution/assets/js/catalog.js`
- Modify: `plugins/ecology_evolution/assets/js/data.js`
- Modify: `tests/test_feedback_update_cohort.py`
- Modify: `tests/test_scientific_exposure_registry.py`
- Modify: `tests/test_http.py`
- Modify: `tests/test_runtime_integration.py`

**Interfaces:**
- Produces pure planners: `plan_run_adaptation_cohort(dataset, schedule, seed) -> RunAdaptationCohort` and `plan_generation_selection_cohorts(dataset, schedule, generation, adaptation, seed) -> GenerationCohorts`。
- Produces idempotent Director wrappers: `freeze_run_adaptation_cohort(run_id, planned)` and `freeze_generation_selection_cohorts(run_id, planned)`；wrappers 只校验并追加 planner 的 immutable identity。
- Produces: `estimate_epoch_capacity(dataset_id, episode_id, schedule, planned_generations) -> CohortCapacityReport` and read-only `POST /evolution-capacity`（与现有 `/runs` 同一 API base）。
- Adds events: `RunAdaptationCohortFrozen`, `GenerationCohortsFrozen`。
- Produces ten stable batch digests shared by both finalist lanes。

- [ ] **Step 1: 写 cohort RED tests**

```python
class EpochCohortPlanningTests(unittest.TestCase):
    def test_default_adaptation_cohort_has_ten_ordered_batches(self):
        schedule = OptimizationSchedule.default()
        adaptation = plan_run_adaptation_cohort(
            dataset_fixture(3200), schedule=schedule, seed=7
        )
        self.assertEqual(adaptation.origin_count, 500)
        self.assertEqual(len(adaptation.batches), 10)
        self.assertEqual([batch.origin_count for batch in adaptation.batches], [50] * 10)
        self.assertEqual(
            len({origin for batch in adaptation.batches for origin in batch.origin_ids}),
            500,
        )

    def test_generation_cohorts_are_disjoint_and_value_blind(self):
        schedule = OptimizationSchedule.default()
        left_data = dataset_fixture(3200)
        right_data = dataset_fixture(3200, changed_labels=True)
        left_adaptation = plan_run_adaptation_cohort(left_data, schedule, seed=7)
        right_adaptation = plan_run_adaptation_cohort(right_data, schedule, seed=7)
        left = plan_generation_selection_cohorts(
            left_data, schedule, generation=2, adaptation=left_adaptation, seed=7
        )
        right = plan_generation_selection_cohorts(
            right_data, schedule, generation=2, adaptation=right_adaptation, seed=7
        )
        self.assertEqual(left_adaptation.identity_dict(), right_adaptation.identity_dict())
        self.assertEqual(left.identity_dict(), right.identity_dict())
        self.assertTrue(set(left.screening.origin_ids).isdisjoint(left.holdout.origin_ids))
        self.assertTrue(set(left.holdout.origin_ids).isdisjoint(left_adaptation.origin_ids))
```

还要覆盖：4 candidates 同 screening digest、2 lanes 同10 batch digest、3 holdout arms 同 digest、generation screening/holdout 均不跨代复用、causal timestamp/最长时距成熟、容量不足显示 required/available/max feasible、禁止 wrap/truncate。默认 G 轮基础身份需求固定为 `500 + G * (64 + 169)`，再由 planner 加上真实 maturity/embargo 约束，不能由前端 row count 推测。增加非默认 `formal=600,batch=60,holdout=200,G=3` 参数化断言，防止把 733/1763 写死。

- [ ] **Step 2: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_epoch_cohort_planning
```

Expected: FAIL。

- [ ] **Step 3: 实现 cohort planner**

- 从 `training_feedback/model_selection` 的公开 identity/timestamp 元数据选取，不读取 label/prediction。
- run 初始化冻结一个 500-origin adaptation cohort；每轮可按固定 seed 对完整时间块做确定性 rotation，但不改变成员。每一轮的两个 finalist 都重新执行完整 500 origins；只复用成员身份，不复用 prediction、sample-result、usage 或 local feedback，因 `generation` 必须进入 scope/event ID。
- 每轮冻结新的 64 screening 和 169 holdout；二者不与 adaptation 或其他 generation holdout 重叠。
- 对每个 batch 记录 `origin_identity_digest`、`origin_count=50`、时间边界和最大 horizon maturity proof；公开投影只含 digest/count。
- 资源不足时抛出结构化 `CohortCapacityError(required, available, max_generations)`。

- [ ] **Step 4: 让 create 与参数页共享同一个 capacity truth**

- 新增只读 `POST /evolution-capacity`，请求只接受 dataset/episode/frozen planned generation limit/optimization schedule，响应包含 eligible、required、max feasible、maturity gaps 和 planner digest。planned generation limit 继续由现有 max-generations/四候选总预算规则产生，禁止启动不足四个候选的尾轮。
- `_bind_runtime_task()` 调用同一个 estimator；capacity 不足时 run receipt 不创建。
- 参数页在 dataset、轮数或 schedule 变化时 debounce 请求该 endpoint；`readiness()` 只消费服务端 report，不读取 `row_count` 推断。
- report digest 写入最终 manifest，并在真正 freeze cohorts 时复算一致性。

- [ ] **Step 5: 将 cohort freeze 接入事件状态**

在四个 candidate 已生成、任何 screening origin 请求发出前，由同一个 `FREEZE_COHORTS` work-unit wave 首轮写 `RunAdaptationCohortFrozen`，每轮写 `GenerationCohortsFrozen`。replay 必须证明 schedule count 与每个 digest/count 一致；这与 Task 11 的 `PREPARE_GENERATION -> FREEZE_COHORTS -> SCREEN` 顺序一致。

- [ ] **Step 6: 禁用新协议的 rotating wrap**

在 `registry.py` 删除新协议对 `_select_feedback_update_cohort()` wrap 结果的依赖；旧函数只留给本次切换前的独立非新协议测试，Task 14 最终删除已无调用的兼容分支。

- [ ] **Step 7: 运行 GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_epoch_cohort_planning \
  tests.test_feedback_update_cohort \
  tests.test_scientific_exposure_registry \
  tests.test_http tests.test_runtime_integration
node plugins/ecology_evolution/test/smoke.mjs
```

Expected: PASS。

- [ ] **Step 8: 提交 cohort planner**

```bash
git add src/ecologyrsi_dsh/evaluators/epoch_cohorts.py \
  src/ecologyrsi_dsh/data/splits.py src/ecologyrsi_dsh/evaluators/registry.py \
  src/ecologyrsi_dsh/api/handler.py src/ecologyrsi_dsh/api/catalog.py \
  src/ecologyrsi_dsh/core/director.py src/ecologyrsi_dsh/core/state.py \
  plugins/ecology_evolution/assets/js/catalog.js \
  plugins/ecology_evolution/assets/js/data.js \
  tests/test_epoch_cohort_planning.py tests/test_feedback_update_cohort.py \
  tests/test_scientific_exposure_registry.py tests/test_http.py \
  tests/test_runtime_integration.py
git commit -m "feat: freeze adaptive and selection cohorts"
```

---

### Task 6: 将 evaluator/sample checkpoint 全面改成 EvaluationScope

**Files:**
- Create: `tests/test_scoped_sample_execution.py`
- Modify: `src/ecologyrsi_dsh/evaluators/registry.py`
- Modify: `src/ecologyrsi_dsh/evaluators/sample_execution.py:730-1150,3000-3420`
- Modify: `src/ecologyrsi_dsh/evaluators/dsh_sample_adapter.py`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py:169-330`
- Modify: `src/ecologyrsi_dsh/core/director.py:3000-3420`
- Modify: `src/ecologyrsi_dsh/core/state.py:2600-2700`
- Modify: `tests/test_sample_execution.py`
- Modify: `tests/test_sample_results_contract.py`
- Modify: `tests/test_dsh_sample_execution.py`

**Interfaces:**
- `EvaluatorRegistry.evaluate_scientific(..., scope: EvaluationScope, algorithm_spec, ...) -> EvaluationBundle`。
- Sample checkpoint/result/model-usage scope key is `EvaluationScope.scope_key`。
- Removes formal execution dependence on public `samples_per_update` and candidate-only checkpoint keys。

- [ ] **Step 1: 写 scope RED tests**

测试同一 candidate 的两个 revision/batch 不串结果：

```python
class ScopedSampleExecutionTests(unittest.TestCase):
    def test_sample_checkpoints_are_isolated_by_revision_phase_and_batch(self):
        director = director_fixture()
        record_complete_origin(
            director, scope=scope("revision:a:1", batch=0), origin="o1"
        )
        self.assertEqual(
            missing_origins(director, scope("revision:a:1", batch=0)), set()
        )
        self.assertEqual(
            missing_origins(director, scope("revision:a:2", batch=1)), {"o1"}
        )
```

覆盖 screening/formal/holdout 同 candidate、并行 finalist、restart、同 scope 重放、相同 event ID 冲突 payload、9-cell origin 原子性、usage 去重。

- [ ] **Step 2: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_scoped_sample_execution
```

Expected: FAIL，现有 key 仍只按 candidate/revision revision string。

- [ ] **Step 3: 修改 evaluator API 和内部预算**

- `evaluate_scientific()` 必须接收 `scope`，从 `scope.origin_count` 选取已冻结 cohort。
- `_screen_candidate()` 在 compile/smoke 成功后先幂等 materialize outer candidate 的 `R0`，再创建绑定 R0 的 screening scope；四候选均如此，Top-2 选择函数和输入记录不变。
- sample executor 不再使用 `task.metadata["samples_per_update"]` 计算 origin count。
- evaluator 仍产生 cell metrics，但结果同时记录 `origin_count` 和 `prediction_cell_count`，两者单位不可互换。
- `ModelArtifact`、`Evaluation` 和回调都绑定 `candidate_revision_id` 与 scope digest。

- [ ] **Step 4: 修改 checkpoint/result/usage events**

所有相关 payload 强制包含：

```python
{
    "candidate_revision_id": scope.candidate_revision_id,
    "evaluation_phase": scope.phase.value,
    "formal_batch_index": scope.batch_index,
    "holdout_arm": None if scope.holdout_arm is None else scope.holdout_arm.value,
    "cohort_digest": scope.cohort_digest,
    "execution_scope_digest": scope.scope_key,
}
```

Director prepare/append/complete 方法先验证 scope 与 trajectory/holdout state 一致。

- [ ] **Step 5: 运行 GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_scoped_sample_execution \
  tests.test_sample_execution \
  tests.test_sample_results_contract \
  tests.test_dsh_sample_execution
```

Expected: PASS。

- [ ] **Step 6: 提交 scoped execution**

```bash
git add src/ecologyrsi_dsh/evaluators/registry.py \
  src/ecologyrsi_dsh/evaluators/sample_execution.py \
  src/ecologyrsi_dsh/evaluators/dsh_sample_adapter.py \
  src/ecologyrsi_dsh/api/generation_execution.py \
  src/ecologyrsi_dsh/core/director.py src/ecologyrsi_dsh/core/state.py \
  tests/test_scoped_sample_execution.py tests/test_sample_execution.py \
  tests/test_sample_results_contract.py tests/test_dsh_sample_execution.py
git commit -m "refactor: scope sample execution by candidate revision"
```

---

### Task 7: 证明并加固现有 run/provider 两层并发与 bounded admission

**Files:**
- Create: `tests/test_sample_concurrency_governor.py`
- Modify: `src/ecologyrsi_dsh/api/sample_admission.py`
- Modify: `src/ecologyrsi_dsh/api/handler.py:371-460`
- Modify: `src/ecologyrsi_dsh/evaluators/sample_execution.py:1040-1140`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py:2248-2310`
- Modify: `src/ecologyrsi_dsh/api/auto_progress.py`
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/provider-stage-gate.js`
- Modify: `integrations/dsh_ecology_plugin/test/provider_stage_gate.test.mjs`
- Modify: `tests/test_candidate_parallel_evaluation.py`
- Modify: `tests/test_sample_admission.py`

**Interfaces:**
- Reuses: `RunSampleAdmission.admit(run_id, limit)` as the only Python run-level authority。
- Reuses: DSH `provider-stage-gate` as the only provider/model-route authority。
- Run-level `sample_concurrency` is the only user-configurable in-flight source of truth: default 64/range 1–128. Provider-route physical limit 128、origin-wave queue bound、candidate concurrency 都是独立内部控制，不能覆盖或相乘 run limit。
- Screening candidates、two formal lanes、holdout arms 使用同一 server-owned `RunSampleAdmission`; different runs share the DSH route gate。

- [ ] **Step 1: 写并发 RED tests**

用 barrier fake sample chain 记录同一 run 的 actual active peak：

```python
class SampleConcurrencyGovernorTests(unittest.TestCase):
    def test_two_finalist_lanes_share_run_limit(self):
        admission = RunSampleAdmission()
        peak = run_blocking_requests(
            admission,
            run_id="run:a",
            lane_request_counts={"candidate:a": 50, "candidate:b": 50},
            run_limit=64,
        )
        self.assertLessEqual(peak.for_run("run:a"), 64)
```

再覆盖 cancellation while waiting、permit release on exception、candidate_concurrency=3 不超限、两个 finalist lane 共用同一 snapshot、current work unit 外任务不入队。Node blocking-provider test 单独证明两个 runs 同 route 的实际 provider calls 不超过128。

- [ ] **Step 2: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_sample_concurrency_governor
node --test integrations/dsh_ecology_plugin/test/provider_stage_gate.test.mjs
```

Expected: 新增的 two-lane/route 端到端断言 FAIL；现有单 run admission 单元测试仍保持 GREEN。

- [ ] **Step 3: 加固现有许可接线**

- 保留 `handler.py` 只创建一个 `RunSampleAdmission` 实例，并把同一个 `admit` callback 传给所有 evaluators。
- 每个 screening/formal/holdout sample chain 在启动 DSH request 前调用该 callback；两个 lane 不创建自己的 admission object。
- DSH stage runner 的每个实际 provider call 继续通过单一 `provider-stage-gate`；route key 必须来自冻结 provider/model。
- 暂停/取消等待者不再启动新 origin；异常路径由两个既有 context/gate 各自释放许可。

- [ ] **Step 4: 删除候选并发的除法分摊**

- 不再计算 `ceil(sample_concurrency / candidate_concurrency)`。
- candidate scheduler 只决定哪些 lane/candidate 可启动；sample executor 每个 request 独立向同一个 `RunSampleAdmission` 申请。
- 两个 50-origin lane 同时运行时只 admission 当前100个 origin；最多64在飞、其余最多36等待，不创建 batch2 任务。

- [ ] **Step 5: 运行 GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_sample_concurrency_governor \
  tests.test_candidate_parallel_evaluation \
  tests.test_sample_admission
node --test integrations/dsh_ecology_plugin/test/provider_stage_gate.test.mjs
```

Expected: PASS，所有 actual active peak assertions 不超过冻结 limit。

- [ ] **Step 6: 提交并发接线与证据**

```bash
git add src/ecologyrsi_dsh/api/sample_admission.py \
  src/ecologyrsi_dsh/api/handler.py \
  src/ecologyrsi_dsh/evaluators/sample_execution.py \
  src/ecologyrsi_dsh/api/generation_execution.py \
  src/ecologyrsi_dsh/api/auto_progress.py \
  integrations/dsh_ecology_plugin/lib/runtime/provider-stage-gate.js \
  integrations/dsh_ecology_plugin/test/provider_stage_gate.test.mjs \
  tests/test_sample_concurrency_governor.py \
  tests/test_candidate_parallel_evaluation.py tests/test_sample_admission.py
git commit -m "test: enforce shared adaptive admission limits"
```

---

### Task 8: 新建独立 candidate.local_edit DSH 契约和 Host 原子应用器

**Files:**
- Create: `src/ecologyrsi_dsh/evolution/local_edits.py`
- Create: `tests/test_local_edits.py`
- Create: `integrations/dsh_ecology_plugin/schemas/local-edit.schema.json`
- Create: `integrations/dsh_ecology_plugin/presets/ecology-local-editor-v1/preset.yml`
- Create: `integrations/dsh_ecology_plugin/presets/ecology-local-editor-v1/agent.cordis.yml`
- Modify: `src/ecologyrsi_dsh/evolution/genome.py:865-1155`
- Modify: `src/ecologyrsi_dsh/evolution/strategies.py:2580-2640,3594-3715`
- Modify: `src/ecologyrsi_dsh/api/dsh_tools.py:60-100`
- Modify: `src/ecologyrsi_dsh/integrations/dsh_structured_roles.py`
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/structured-roles.js`
- Modify: `integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js`
- Modify: `integrations/dsh_ecology_plugin/lib/tools/roles.js`
- Modify: `integrations/dsh_ecology_plugin/test/structured_roles.test.mjs`
- Modify: `integrations/dsh_ecology_plugin/test/stage_runner.test.mjs`
- Modify: `pyproject.toml`
- Modify: `scripts/verify_delivery.sh`
- Modify: `tests/test_dsh_structured_roles.py`
- Modify: `tests/test_evolution_genome.py`

**Interfaces:**
- Adds DSH operation `candidate.local_edit`, role `candidate-local-editor`, output contract `ecology-local-edit@1`。
- Produces: `LocalEditContext`, `LocalEditProposal`, `LocalEditResult`, `validate_local_edit_proposal()`, `apply_local_edit_bundle()`。
- Wire proposal uses `decision=keep|mutate`; durable Host result uses distinct `outcome=kept|applied|rejected`，避免把“技术上已应用”误写成“科学上已改善”。
- Reuses current `_registered_mutation_targets()` and per-operation trust-region validation without changing outer `candidate.propose maximum_operations=1`。

- [ ] **Step 1: 写 Python local-edit RED tests**

必须覆盖：

```python
class LocalEditTests(unittest.TestCase):
    def test_keep_is_valid_and_does_not_create_revision(self):
        result = apply_local_edit_bundle(
            parent_genome(), keep_proposal(), context(maximum=2), registry()
        )
        self.assertEqual(result.outcome, LocalEditOutcome.KEPT)
        self.assertIsNone(result.child)

    def test_two_registered_edits_are_applied_atomically(self):
        parent = parent_genome()
        result = apply_local_edit_bundle(
            parent, two_axis_proposal(), context(maximum=2), registry()
        )
        self.assertEqual(result.outcome, LocalEditOutcome.APPLIED)
        self.assertIsNotNone(result.child)
        self.assertEqual(len(result.operations), 2)
        self.assertEqual(result.child.lineage["parent_genome_digest"], parent.genome_digest)

    def test_edit_count_above_frozen_maximum_is_rejected(self):
        for count in (3, 6):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "maximum_operations"):
                    validate_local_edit_proposal(
                        proposal_with_count(count), context(maximum=2), registry()
                    )
```

还要覆盖 0/1/2/5、duplicate path、conflicting operations、unregistered target、预算/evaluator/data/credential/raw-code path、每操作 trust-region、整个 bundle 原子回滚、compile/smoke failure -> rejected/keep、evidence batch mismatch。

- [ ] **Step 2: 写外层行为保护 RED test**

```python
class OuterCandidateContractTests(unittest.TestCase):
    def test_outer_candidate_proposer_remains_single_operation(self):
        contract = outer_candidate_mutation_contract(task_fixture())
        self.assertEqual(contract["maximum_operations"], 1)
```

这个测试必须贯穿任务，禁止把 `strategies.py` 两处现有 `maximum_operations: 1` 直接替换为页面值。

- [ ] **Step 3: 写 Node schema/role RED tests**

先创建新 preset 目录（当前尚不存在）：

```bash
mkdir -p integrations/dsh_ecology_plugin/presets/ecology-local-editor-v1
```

断言 local editor：

- 只能加载新 schema/preset；
- 输入只含 aggregate metrics、digest 和 Host catalog；
- 不包含 raw observed/predicted/timestamp；
- `decision=keep` 要求空 operations；
- `decision=mutate` 要求 1..K operations；
- idempotency key 含 candidate/revision/batch。

- [ ] **Step 4: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_local_edits tests.test_evolution_genome \
  tests.test_dsh_structured_roles
node --test \
  integrations/dsh_ecology_plugin/test/structured_roles.test.mjs \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
```

Expected: FAIL，新 stage/schema 尚不存在。

- [ ] **Step 5: 抽取共享 operation 应用内核**

在 `genome.py` 抽取私有 `_apply_registered_operations(parent, operations, policy, registry, schemas)`：

- `apply_genome_mutation()` 继续用外层 policy，strict trust-region 时 `min=max=1`；
- `apply_local_edit_bundle()` 用 `min=1,max=context.maximum_operations`；
- 每项仍执行现有 registered path、bounds、direction 和 trust-region 检查；
- 全部操作先在 detached copy 上验证，全部通过后才构造 child genome。

- [ ] **Step 6: 实现 bounded local-edit contract**

输出 exact fields：

```json
{
  "schema_version": "ecology-local-edit@1",
  "decision": "keep|mutate",
  "operations": [],
  "evidence_refs": [],
  "expected_effect_cells": [],
  "risk_cells": []
}
```

JSON Schema 将 `operations.maxItems` 固定为绝对上限5；`evidence_refs` 只能引用 Host 提供的 metric IDs；effect/risk cells 只能使用冻结 target/horizon ID。Host 再用 run schedule 的 `maximum_operations`（1–5）做动态上限校验。禁止自由文本成为操作值。

- [ ] **Step 7: 接入 DSH role/preset/stage**

`dsh_tools.py` 和 Node role catalog 增加 `candidate.local_edit`；使用 fresh Session、最多一次 schema repair、确定性 idempotency key：

```text
{run_id}:candidate.local_edit:{generation}:{candidate_id}:{revision_id}:{batch_index}
```

schema/Host/compile/smoke 拒绝返回结构化 rejection code，由 Host 写 `LocalEditDecided(outcome="rejected")`。DSH transport/provider failure 不写任何 scientific outcome，只由 work-unit retry authority 处理，不能把它当作 KEEP 或 rejected edit。

同步把新 preset/schema 加入 `pyproject.toml` package-data 与 `verify_delivery.sh` source inventory；Node package 已由 `presets/**/*.yml`、`schemas/*.json` glob 收录，但测试必须验证生成的 tgz 确实含这些文件。

- [ ] **Step 8: 运行 GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_local_edits tests.test_evolution_genome \
  tests.test_dsh_structured_roles
node --test \
  integrations/dsh_ecology_plugin/test/structured_roles.test.mjs \
  integrations/dsh_ecology_plugin/test/stage_runner.test.mjs
make verify
```

Expected: PASS；outer proposer golden 仍为1 operation。

- [ ] **Step 9: 提交 local editor**

```bash
git add src/ecologyrsi_dsh/evolution/local_edits.py \
  src/ecologyrsi_dsh/evolution/genome.py \
  src/ecologyrsi_dsh/evolution/strategies.py \
  src/ecologyrsi_dsh/api/dsh_tools.py \
  src/ecologyrsi_dsh/integrations/dsh_structured_roles.py \
  integrations/dsh_ecology_plugin/schemas/local-edit.schema.json \
  integrations/dsh_ecology_plugin/presets/ecology-local-editor-v1 \
  integrations/dsh_ecology_plugin/lib integrations/dsh_ecology_plugin/test \
  pyproject.toml scripts/verify_delivery.sh \
  tests/test_local_edits.py tests/test_evolution_genome.py \
  tests/test_dsh_structured_roles.py
git commit -m "feat: add bounded batch local editor"
```

---

### Task 9: 实现单 finalist 的 10 x 50 FormalTrajectory executor

**Files:**
- Create: `src/ecologyrsi_dsh/api/formal_trajectory.py`
- Create: `tests/test_formal_trajectory_execution.py`
- Create: `tests/test_formal_trajectory_recovery.py`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py:1546-1930,2248-2310`
- Modify: `src/ecologyrsi_dsh/knowledge/algorithms.py`
- Modify: `src/ecologyrsi_dsh/evolution/batches.py`

**Interfaces:**
- Produces: `ensure_formal_trajectory(endpoint, run_id, candidate_id) -> FormalTrajectory`。
- Produces: `execute_next_formal_batch(endpoint, run_id, candidate_id) -> WorkUnitResult`。
- Produces: `execute_next_local_edit(endpoint, run_id, candidate_id) -> WorkUnitResult`。
- Uses Task 6 scoped evaluator and Task 8 local editor。

- [ ] **Step 1: 写前两批合法轨迹 RED test**

测试 schedule fixture 使用合法的 `formal=100,batch=10`，只推进 internal batch0 和 batch1 后断言；不得绕过 Task 1 的产品校验来伪造 `formal=4,batch=2`：

```python
class FormalTrajectoryExecutionTests(unittest.TestCase):
    def test_revision_from_display_batch_one_only_runs_on_display_batch_two(self):
        endpoint = endpoint_fixture(formal=100, batch=10)
        trajectory = run_first_two_batches(
            endpoint, decisions=[mutate_one(), keep()]
        )
        self.assertEqual(
            trajectory.batch(0).revision_id, trajectory.initial_revision_id
        )
        self.assertEqual(
            trajectory.revision_from_batch(0).revision_id,
            trajectory.batch(1).revision_id,
        )
        self.assertNotEqual(
            trajectory.batch(0).revision_id, trajectory.batch(1).revision_id
        )
```

默认配置另断言 exactly 10 batch starts/evaluations，origin counts 均为50，总计500。

- [ ] **Step 2: 写恢复 RED matrix**

在以下边界模拟进程异常并重建 endpoint：

- compiled R0 persisted/trajectory start missing；
- batch started/0、17、49 origins complete；
- 50 origins complete/BatchEvaluation missing；
- analysis complete/local proposal missing；
- proposal persisted/decision missing；
- outcome applied/revision event missing；
- local outcome persisted/revision activation missing；
- lane A complete/lane B batch6；
- internal batch9 local revision created/activation missing；
- internal batch9 activation persisted/trajectory completion missing；
- trajectory complete/holdout freeze missing。

每个用例断言 origin、usage、proposal、decision、revision 不重复。

- [ ] **Step 3: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_formal_trajectory_execution \
  tests.test_formal_trajectory_recovery
```

Expected: FAIL。

- [ ] **Step 4: 把已有 compiled R0 接入 finalist trajectory**

`ensure_formal_trajectory()` 仅在 `FormalSelectionCohortFrozen` 含 candidate 时运行：

1. 获取 Task 6 在 screening 前已写入、并与 compiled genome/behavior digest 一致的 immutable R0；
2. 若 R0 缺失或 digest 不一致则 fail closed，绝不静默重建；
3. 写 `FormalTrajectoryStarted(batch_count=schedule.batch_count, initial_revision_id=R0)`；
4. 重复调用只返回现有 trajectory。

- [ ] **Step 5: 执行一个 batch work unit**

`execute_next_formal_batch()`：

1. 从 state 选择最小未完成 batch index；
2. 冻结 active revision 和 adaptation batch digest；
3. 写 `FormalBatchStarted`；
4. 调用 scoped `evaluate_scientific()` 执行缺失 origins；
5. 从完整结果构造 aggregate 3 x 3 `BatchEvaluation`；
6. coverage/缺样本属于 retry/pause，不得生成 batch scientific result；只有完整 evidence 才检查 absolute physical constraint；
7. 写 `FormalBatchEvaluated`，不在该函数调用 local editor 或改变 active revision。

- [ ] **Step 6: 执行一个 local-edit work unit**

`execute_next_local_edit()` 只消费已完成且尚无 decision 的 batch：

- DSH transport/provider failure 抛给 retry scheduler，不写 KEEP；
- valid KEEP 写 `outcome=kept`，不建 revision；
- valid mutate -> Host validate -> compile -> smoke -> 写 child revision 和 `outcome=applied`；
- schema/semantic/compile/smoke 拒绝 -> `outcome=rejected`，不建 child；
- 随后无论 outcome 都必须写一个 `TrajectoryRevisionAdvanced`，明确下一 batch revision；新 child 只能从下一 batch 生效；
- 若完整 batch 发现 empirical absolute physical violation，跳过 model editor，写 `reason=safety_revert` 并激活最近一个已在先前 batch 通过 absolute gate 的 revision；soft score 变化和 coverage/infrastructure failure 均不能触发 rollback；
- internal batch9 完成 outcome/safety decision 后写 activation，该 active revision 才成为 holdout final revision；KEEP/rejected 沿用 current，APPLIED 使用 child，safety revert 使用 last-safe；
- `FormalTrajectoryCompleted` 必须在 internal batch9 activation 之后、holdout freeze 之前。

- [ ] **Step 7: 让两个 finalist lane 可并行**

`generation_execution.py` 保留现有 `run_candidate_evaluations()`，但一个 scheduler wave 中装入每条 eligible lane 的一个独立 scoped sub-unit；lane 内依赖 state 维持顺序，lane A/B 之间可并行。每条 sub-unit 单独落盘/失败/重试；A成功B失败时下一 wave 只补B。两个 lane 共享 Task 7 governor 和 Task 5 batch digests，但不共享 aggregate feedback/session/revision。

- [ ] **Step 8: 运行 GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_formal_trajectory_execution \
  tests.test_formal_trajectory_recovery \
  tests.test_candidate_parallel_evaluation
```

Expected: PASS。

- [ ] **Step 9: 提交 trajectory executor**

```bash
git add src/ecologyrsi_dsh/api/formal_trajectory.py \
  src/ecologyrsi_dsh/api/generation_execution.py \
  src/ecologyrsi_dsh/knowledge/algorithms.py \
  src/ecologyrsi_dsh/evolution/batches.py \
  tests/test_formal_trajectory_execution.py \
  tests/test_formal_trajectory_recovery.py \
  tests/test_candidate_parallel_evaluation.py
git commit -m "feat: execute adaptive finalist trajectories"
```

---

### Task 10: 实现 F1/F2/incumbent 三臂 holdout 和确定性冠军门禁

**Files:**
- Create: `src/ecologyrsi_dsh/evaluators/generation_comparison.py`
- Create: `tests/test_generation_holdout.py`
- Create: `tests/test_generation_comparison.py`
- Modify: `src/ecologyrsi_dsh/evaluators/fitness.py`
- Modify: `src/ecologyrsi_dsh/evolution/promotion.py`
- Modify: `src/ecologyrsi_dsh/evolution/analysis.py:3270-3500`
- Modify: `src/ecologyrsi_dsh/evolution/batches.py:900-1100`
- Modify: `src/ecologyrsi_dsh/core/models.py`
- Modify: `src/ecologyrsi_dsh/core/state.py:1320-1380,2100-2180`
- Modify: `src/ecologyrsi_dsh/core/director.py:1750-1810,2140-2190,2420-2620`
- Modify: `src/ecologyrsi_dsh/evolution/strategies.py`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py:1040-1200`
- Modify: `tests/test_fitness.py`
- Modify: `tests/test_promotion.py`
- Modify: `tests/test_horizon_feedback.py`

**Interfaces:**
- Produces: `build_generation_comparison(holdout, evaluations, fitness_profile) -> GenerationComparison`。
- Produces: `select_generation_champion(comparison, gates) -> ChampionDecision`。
- Promotion binds `candidate_revision_id` and `comparison_digest`。
- Produces: `CandidateEffectiveRevisionFrozen` and `RunState.effective_revision_for(candidate_id)` as the only next-generation parent source。

- [ ] **Step 1: 写三臂同 cohort RED tests**

```python
class GenerationHoldoutTests(unittest.TestCase):
    def test_holdout_uses_exactly_same_cohort_for_f1_f2_and_incumbent(self):
        holdout = build_holdout_fixture(origin_count=169)
        self.assertEqual(
            {arm.scope.cohort_digest for arm in holdout.arms},
            {holdout.cohort_digest},
        )
        self.assertEqual({arm.scope.origin_count for arm in holdout.arms}, {169})

    def test_comparison_rejects_mixed_cohort_even_when_scores_improve(self):
        with self.assertRaisesRegex(ValueError, "same holdout cohort"):
            build_generation_comparison(
                mixed_cohort_holdout(), mixed_evaluations(), profile()
            )

    def test_next_generation_parent_is_winner_final_revision(self):
        state = completed_generation_with_winner(
            outer_revision="revision:a:0",
            final_revision="revision:a:10",
        )
        effective = state.effective_revision_for("candidate:a")
        self.assertEqual(effective.revision_id, "revision:a:10")
        self.assertEqual(
            next_generation_parent_genome(state).genome_digest,
            effective.genome_digest,
        )
```

- [ ] **Step 2: 写门禁/排序 RED tests**

覆盖以下真值表：

- A 通过/B失败 -> A；
- B通过/A失败 -> B；
- A/B都失败 -> retain incumbent；
- A/B都通过 -> LCB、overall delta、worst-cell delta、candidate ID 顺序；
- overall delta<=0.005、LCB<=0、任一cell delta<-0.01、coverage<0.95、缺9-cell、constraint>0 均不晋级；
- generation0 使用 seed revision incumbent；
- 即使 F revision 与 incumbent digest 相同也执行三个独立 arm scope；禁止跨 arm 复用 prediction/usage，预算始终为 `3 * holdout origins`；
- judge 反对/赞成不能覆盖 Host decision。
- 下一代 parent genome digest 必须等于 winner final revision digest，不能回到 outer proposal 的 initial genome。

- [ ] **Step 3: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_generation_holdout tests.test_generation_comparison
```

Expected: FAIL。

- [ ] **Step 4: 复用现有科学统计而非另造分数**

- 使用现有 `FitnessProfile`、3 x 3 metric aggregation、paired moving-block/max-T helpers。
- 明确同一个 family 中同时比较 F1-vs-C 和 F2-vs-C，控制“二选一”偏差。
- 结果只持久化 aggregate score/delta/coverage/constraint/CI/better-neutral-worse 和 sufficient-stat digest；公共事件不含 block/raw rows。

- [ ] **Step 5: 实现 holdout work 和 champion decision**

三个 arm 可在一个 wave 内并行并独立恢复，每个 arm 都产生独立 sample/usage/evidence events；`GenerationCompared` 仅在三臂 complete 后写入。随后：

```python
if not eligible_finalists:
    selected_revision_id = comparison.incumbent_revision_id
elif len(eligible_finalists) == 1:
    selected_revision_id = eligible_finalists[0].revision_id
else:
    selected_revision_id = min(eligible_finalists, key=deterministic_champion_sort_key).revision_id
```

若 retain incumbent，两个当前 finalist 都写 rejected；若晋级，promotion 绑定 final revision/comparison digest，另一 finalist rejected。

不新增与现有 `CandidateStatus` 竞争的生命周期：trajectory 运行期间 outer candidate 保持 `SPAWNED`，实时状态只看 `TrajectoryStatus`；三臂证据完成后给两个 finalist 绑定各自 final revision 的 Evaluation 并转为 `EVALUATED`，随后 winner `PROMOTED`、另一方 `REJECTED`。retain incumbent 时两个当前 finalist 均 `REJECTED`，不改写旧 incumbent 的历史 candidate status。

- [ ] **Step 6: 冻结 winner 的 effective revision 身份**

在 `GenerationCompared` 之后、promotion/next generation 之前写 `CandidateEffectiveRevisionFrozen`：

```python
{
    "candidate_id": winner_candidate_id,
    "candidate_revision_id": winner_revision_id,
    "genome_digest": winner_revision.genome_digest,
    "behavior_digest": winner_revision.behavior_digest,
    "comparison_digest": comparison.comparison_digest,
}
```

该事件每轮恰好写一次。保留 incumbent 时，本轮仍写新的 generation binding/comparison digest，但 candidate/revision 指向上一轮已冻结的同一 effective revision；generation0 seed 具有显式 seed revision。随后将这些旧的 candidate-root lookup 全部切换到 `effective_revision_for()`：下一轮 parent selection、generation judge、artifact/evaluation binding、promotion、formal-stage seal、generation analysis/reflection。

- [ ] **Step 7: 构建轮末 global reflection**

输出：两个 finalist 相对 incumbent 的9-cell矩阵、每批 edit bundle 的 kept/applied/rejected category、hard failures、selected/retained reason、下一轮 bounded focus/avoid coordinates。只在 champion event 已持久化后让下一轮 proposer 读取。

- [ ] **Step 8: 运行 GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_generation_holdout tests.test_generation_comparison \
  tests.test_fitness tests.test_promotion \
  tests.test_horizon_feedback
```

Expected: PASS。

- [ ] **Step 9: 提交 comparison/promotion**

```bash
git add src/ecologyrsi_dsh/evaluators/generation_comparison.py \
  src/ecologyrsi_dsh/evaluators/fitness.py \
  src/ecologyrsi_dsh/evolution/promotion.py \
  src/ecologyrsi_dsh/evolution/analysis.py \
  src/ecologyrsi_dsh/evolution/batches.py src/ecologyrsi_dsh/core/models.py \
  src/ecologyrsi_dsh/core/state.py src/ecologyrsi_dsh/core/director.py \
  src/ecologyrsi_dsh/evolution/strategies.py \
  src/ecologyrsi_dsh/api/generation_execution.py \
  tests/test_generation_holdout.py tests/test_generation_comparison.py \
  tests/test_fitness.py tests/test_promotion.py tests/test_horizon_feedback.py
git commit -m "feat: compare finalist revisions on generation holdout"
```

---

### Task 11: 用 durable work units 替换“一次 worker 跑完整代”

**Files:**
- Create: `src/ecologyrsi_dsh/api/work_units.py`
- Create: `tests/test_adaptive_work_units.py`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py:2248-2480`
- Modify: `src/ecologyrsi_dsh/api/auto_progress.py:718-1000,1180-1270`
- Modify: `src/ecologyrsi_dsh/api/execution.py`
- Modify: `src/ecologyrsi_dsh/api/handler.py`
- Modify: `tests/test_auto_progress.py`
- Modify: `tests/test_generation_control_execution.py`
- Modify: `tests/test_evolution_feedback_loop.py`

**Interfaces:**
- Produces: `WorkUnitKind`, `ScopedWorkUnit`, `WorkUnitWave`, `ScopedWorkResult`, `WorkUnitResult`。
- Produces: `next_work_unit(state) -> WorkUnitWave | None` as a pure deterministic selector；formal/holdout wave 内每个 lane/arm sub-unit 独立 durable。
- Produces: `execute_next_work_unit(endpoint, run_id) -> WorkUnitResult`。
- `POST /runs/{run_id}/advance` and auto-progress request progress by work unit; only finalization increments generation。

- [ ] **Step 1: 写 pure selector RED table**

用逐渐完整的 RunState fixtures 断言顺序：

```python
class AdaptiveWorkUnitTests(unittest.TestCase):
    def test_next_work_unit_is_deterministic(self):
        cases = [
            ("empty_generation", WorkUnitKind.PREPARE_GENERATION),
            ("candidates_spawned", WorkUnitKind.FREEZE_COHORTS),
            ("cohorts_frozen", WorkUnitKind.SCREEN_CANDIDATES),
            ("screening_complete", WorkUnitKind.FREEZE_TOP2),
            ("top2_frozen", WorkUnitKind.START_TRAJECTORIES),
            ("batch_missing", WorkUnitKind.EXECUTE_FORMAL_BATCH_WAVE),
            ("batch_evaluated", WorkUnitKind.DECIDE_LOCAL_EDIT_WAVE),
            ("trajectories_complete", WorkUnitKind.FREEZE_HOLDOUT),
            ("holdout_incomplete", WorkUnitKind.EXECUTE_HOLDOUT_WAVE),
            ("holdout_complete", WorkUnitKind.COMPARE_GENERATION),
            ("comparison_complete", WorkUnitKind.FINALIZE_GENERATION),
        ]
        for state_name, expected_kind in cases:
            with self.subTest(state_name=state_name):
                self.assertIs(next_work_unit(state_fixture(state_name)).kind, expected_kind)
```

- [ ] **Step 2: 写自动推进 RED tests**

覆盖：一次 worker call 只执行一个 unit；generation 未增长不再被视为失败；unit 后 run 重新入队；两 run 按 unit 公平轮转；同 run lease 唯一；pause/cancel；两个 lane unit 中 A完成/B失败后只补B；比较落盘前 generation 不前移。

- [ ] **Step 3: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_adaptive_work_units tests.test_auto_progress
```

Expected: FAIL，当前 `_run_one_generation_locked()` 仍假设完整 generation。

- [ ] **Step 4: 实现 WorkUnitResult**

```python
@dataclass(frozen=True, slots=True)
class ScopedWorkResult:
    scope_key: str
    state_changed: bool
    retry_required: bool


@dataclass(frozen=True, slots=True)
class WorkUnitResult:
    state_changed: bool
    generation_advanced: bool
    run_terminal: bool
    retry_required: bool
    scoped_results: tuple[ScopedWorkResult, ...] = ()
```

`next_work_unit()` 只读 state，不访问网络或写 ledger。formal wave 按“完成 batch 较少者优先、再按 candidate ID”生成最多两个 lane sub-unit；holdout wave 按 frozen arm order 生成尚未完成的 arm sub-unit。`execute_next_work_unit()` 在单一 per-run lease 内并行 dispatch wave；每个 sub-unit 自己落盘并返回结果，每次返回前重新读取 state。部分成功合法：summary `state_changed=True,retry_required=True`，下一 selector 只返回失败 scope。

- [ ] **Step 5: 重写 auto-progress 循环**

- `_run_one_generation_locked()` 改名 `_run_one_work_unit_locked()`；一次调用只执行一个 wave，不循环吞掉下一 wave。
- 每次 unit 后释放 per-run lease 并重新排队；不同 run 获得轮转机会。
- `generation_advanced=False` 是正常结果。
- 只有 `retry_required` 进入 retry/backoff；Host validation rejection 属于已完成 state change。
- retry authority key 扩展为 `(incarnation,generation,work_kind,candidate_id,revision_id,batch_index,holdout_arm,retry_class)`。
- pending retry 与 active scope 不一致时 fail closed，不允许把 lane A cooldown 应用给 lane B。

- [ ] **Step 6: 保持显式控制语义**

`POST /runs/{run_id}/advance` 提交一个 work-unit admission receipt 并立即返回最新 projection；连续模式自动重排，单步模式只推进一个 durable wave。pause 关闭新 unit/origin admission，已在飞 complete origin 排空；cancel 不再启动 local edit/holdout。

- [ ] **Step 7: 运行 GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_adaptive_work_units tests.test_auto_progress \
  tests.test_generation_control_execution \
  tests.test_evolution_feedback_loop
```

Expected: PASS。

- [ ] **Step 8: 提交 work-unit scheduler**

```bash
git add src/ecologyrsi_dsh/api/work_units.py \
  src/ecologyrsi_dsh/api/generation_execution.py \
  src/ecologyrsi_dsh/api/auto_progress.py \
  src/ecologyrsi_dsh/api/execution.py src/ecologyrsi_dsh/api/handler.py \
  tests/test_adaptive_work_units.py tests/test_auto_progress.py \
  tests/test_generation_control_execution.py \
  tests/test_evolution_feedback_loop.py
git commit -m "feat: advance evolution by durable work unit"
```

---

### Task 12: 投影并渲染两条轨迹、local edits、三臂 holdout 和真实预算

**Files:**
- Modify: `src/ecologyrsi_dsh/api/projection.py:1500-1750,2300-2450,2886-3335`
- Modify: `src/ecologyrsi_dsh/api/events.py`
- Modify: `plugins/ecology_evolution/assets/js/core.js:919-1020`
- Modify: `plugins/ecology_evolution/assets/js/render_process.js:1500-1750`
- Modify: `plugins/ecology_evolution/assets/js/render_shell.js`
- Modify: `plugins/ecology_evolution/assets/js/demo.js`
- Modify: `plugins/ecology_evolution/styles.css`
- Modify: `tests/test_execution_projection.py`
- Modify: `tests/test_stage_projection.py`
- Modify: `plugins/ecology_evolution/test/smoke.mjs`

**Interfaces:**
- Public projection adds `optimization_schedule`, `execution_budget`, `screening`, `formal_trajectories[2]`, `generation_holdout`, `generation_comparison`, `active_work_unit`。
- Current-generation progress denominator uses durable candidate-origin work: defaults `256+1000+507=1763` and derived cells `15867`。Whole-run planned denominator multiplies by frozen planned generations; unique dataset-origin capacity uses `formal + G*(screening+holdout)` instead。
- Projection never emits raw cohort members, raw metrics rows, full genomes, prompts, credentials, or private DSH Session content。

- [ ] **Step 1: 写 backend projection RED tests**

```python
class AdaptiveProjectionTests(unittest.TestCase):
    def test_default_projection_separates_all_budget_units(self):
        projection = project_default_adaptive_run(planned_generations=5)
        self.assertEqual(projection["execution_budget"]["per_generation"], {
            "origin_member_roles": 733,
            "candidate_origin_executions": 1763,
            "derived_scoring_cells": 15867,
        })
        self.assertEqual(projection["execution_budget"]["run_plan"], {
            "planned_generations": 5,
            "unique_dataset_origins": 1665,
            "candidate_origin_executions": 8815,
            "derived_scoring_cells": 79335,
        })
        self.assertEqual(
            [lane["batch_count"] for lane in projection["formal_trajectories"]],
            [10, 10],
        )
        self.assertEqual(projection["generation_holdout"]["arm_count"], 3)
```

增加 partial state：screening 8/256、lane A batch4 17/50、lane B batch3 complete、provider backoff、edit applied/rejected/kept、holdout2/3、incumbent retained。断言 completed/succeeded/failed 分开，不能把 completed model chain 全算成功。再用 `formal=600,batch=60,holdout=200,G=3` 断言 per-generation executions=2056、run executions=6168、unique dataset origins=1392，证明所有预算均动态计算。

- [ ] **Step 2: 写 browser RED assertions**

浏览器 smoke 必须看到：

- `第 1/5 轮`、`64 时点初筛 -> Top 2`；
- finalist A/B 独立卡片、`batch 4/10`、revision short ID；
- `17/50` 当前 batch 与 `167/500` lane total；
- local edit `接受2项/保持/拒绝` 时间线；
- holdout `F1/F2/上一冠军 2/3 arms`；
- `实际在飞`、`等待许可`、route cooldown、最近 durable event；
- candidate-origins 与 scoring cells 两种预算；
- 页面源码不再出现硬编码“Top 2 各正式评估 500”“1,521评分单元作为每候选formal预算”。

- [ ] **Step 3: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_execution_projection tests.test_stage_projection
node plugins/ecology_evolution/test/smoke.mjs
```

Expected: FAIL。

- [ ] **Step 4: 构造单一 backend projection source**

- 从 RunState/事件索引读取，不从 CandidateStatus 猜 batch。
- 抽取一个 schedule projection helper，供完整 run 和 run-list summary 共用，删除 `projection.py` 3020/3151/3287 附近重复字段拼装。
- current stage 来自 `active_work_unit`；远程模型等待没有百分比，只显示 elapsed/heartbeat/retry。
- current-generation percent 与 whole-run percent 分开，只由各自 durable candidate-origin counts 计算；local editor/compile 等非 origin work 用 stage label、heartbeat 和 elapsed 表示，不虚构样本完成量。

- [ ] **Step 5: 渲染响应式 process UI**

- screening card 保持现有 Top-2 证据。
- 增加两列 trajectory cards；小屏改为纵向。
- batch timeline 只显示 revision/edit kept/applied/rejected category/digest short ID；`applied` 只表示 Host 已应用，不宣称指标改善。
- comparison 用 3 x 3 better/neutral/worse matrix 和 deterministic champion reason。
- frozen parameters card 读取 active run projection，禁止读取当前表单。
- `demo.js` 更新到新 schema，确保离线演示与真实响应一致。

- [ ] **Step 6: 运行 GREEN 和隐私检查**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_execution_projection tests.test_stage_projection
node plugins/ecology_evolution/test/smoke.mjs
git diff --check
```

Expected: PASS；projection JSON 不包含 fixture raw origin IDs/labels/predictions。

- [ ] **Step 7: 提交 projection/UI**

```bash
git add src/ecologyrsi_dsh/api/projection.py src/ecologyrsi_dsh/api/events.py \
  plugins/ecology_evolution/assets/js plugins/ecology_evolution/styles.css \
  plugins/ecology_evolution/test/smoke.mjs \
  tests/test_execution_projection.py tests/test_stage_projection.py
git commit -m "feat: show adaptive finalist trajectory progress"
```

---

### Task 13: 先交付可验证的旧运行静态归档工具

**Files:**
- Create: `scripts/archive_legacy_runs.py`
- Create: `src/ecologyrsi_dsh/application/legacy_archive.py`
- Create: `tests/test_cutover_archive.py`
- Modify: `src/ecologyrsi_dsh/application/cli.py`
- Modify: `tests/test_run_cleanup.py`
- Modify: `README.md`

**Interfaces:**
- Produces offline-only CLI `ecologyrsi-dsh archive-legacy --db OLD_DB --output NEW_DIRECTORY`。
- Produces immutable inventory, per-run event export, sanitized projection, old binary/source identity, SQLite metadata, and `SHA256SUMS`。
- Verification command fails if any archived file changes。

- [ ] **Step 1: 写归档 RED tests**

覆盖：目录必须不存在、路径安全编码、每个 run 恰好一个 event export 和 static projection、inventory 数量/状态/event count、SHA篡改检测、非终态 run 拒绝 final archive、static HTML/JSON 不含 advance/resume/control token。

- [ ] **Step 2: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_cutover_archive tests.test_run_cleanup
```

Expected: FAIL，新命令尚不存在。

- [ ] **Step 3: 实现只读归档编排**

- 在删除旧 runtime projector 前，把所需的 v7 inventory/event JSON/sanitized static projection 序列化逻辑冻结到 `application/legacy_archive.py`。它以 SQLite read-only URI 直接读取，只生成静态证据，不构造可执行 RunState，不调用新 `EventLedger`，也不提供 advance/resume/import。
- server、active projector、Director、ledger 均不得 import `legacy_archive`；CLI 只在显式 `archive-legacy` 子命令中 lazy import。该离线边界是归档能力，不是运行兼容层。
- inventory 写旧 app version、git SHA、ledger schema、run status、event count、archive timestamp。
- 输出目录存在时拒绝，不覆盖任何归档。
- 归档验证只读；失败时换新 timestamp 目录重跑。

- [ ] **Step 4: 运行 GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_cutover_archive tests.test_run_cleanup
```

Expected: PASS。

- [ ] **Step 5: 提交归档工具**

```bash
git add scripts/archive_legacy_runs.py src/ecologyrsi_dsh/application/legacy_archive.py \
  src/ecologyrsi_dsh/application/cli.py \
  tests/test_cutover_archive.py tests/test_run_cleanup.py README.md
git commit -m "feat: add verified legacy run archive"
```

---

### Task 14: 建立新数据库身份围栏并删除旧 one-shot/兼容路径

**Files:**
- Create: `tests/test_incompatible_cutover.py`
- Create: `tests/test_removed_legacy_contract.py`
- Modify: `src/ecologyrsi_dsh/core/ledger.py:1-100`
- Modify: `src/ecologyrsi_dsh/core/protocols.py`
- Modify: `src/ecologyrsi_dsh/core/state.py`
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py:72-170,143-168,1546-1930,2248-2480`
- Modify: `src/ecologyrsi_dsh/api/handler.py`
- Modify: `src/ecologyrsi_dsh/api/projection.py`
- Modify: `src/ecologyrsi_dsh/evaluators/registry.py`
- Modify: `src/ecologyrsi_dsh/evaluators/sample_execution.py`
- Modify: `src/ecologyrsi_dsh/evolution/batches.py`
- Modify: `src/ecologyrsi_dsh/evolution/analysis.py`
- Modify: `src/ecologyrsi_dsh/evolution/strategies.py`
- Modify: `scripts/dsh_native_e2e_acceptance.py`
- Modify: `scripts/install_dsh_ecology_runtime.mjs`
- Modify: `integrations/dsh_ecology_plugin/package.json`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `plugins/ecology_evolution/README.md`
- Modify: relevant protocol/replay/delivery tests
- Delete: `integrations/dsh_ecology_plugin/presets/ecology-coordinator-v3/`
- Delete: `integrations/dsh_ecology_plugin/presets/ecology-researcher-v6/`
- Delete: `integrations/dsh_ecology_plugin/presets/ecology-candidate-proposer-v3/`
- Delete: `integrations/dsh_ecology_plugin/presets/ecology-sample-planner-v3/`
- Delete: `integrations/dsh_ecology_plugin/presets/ecology-sample-critic-v3/`
- Delete: `integrations/dsh_ecology_plugin/presets/ecology-generation-judge-v6/`

**Interfaces:**
- New ledger `SCHEMA_VERSION=8` plus exact event identity `ecologyrsi-dsh.top2-adaptive-events/1`。
- Existing nonempty database with schema !=8 is rejected before WAL/DDL/auto-migration。
- New projector requires `optimization_protocol=top2_adaptive_epoch@1` and exact schedule in `RunCreated`。

- [ ] **Step 1: 写 database fence RED tests**

```python
class IncompatibleCutoverTests(unittest.TestCase):
    def test_new_binary_rejects_v7_before_wal_or_schema_write(self):
        with tempfile.TemporaryDirectory() as directory:
            old_db = build_v7_database(Path(directory) / "old.sqlite3")
            before = sha256_file(old_db)
            with self.assertRaisesRegex(
                IncompatibleLedgerError, "archive and use a new database"
            ):
                EventLedger(old_db)
            self.assertEqual(sha256_file(old_db), before)
            self.assertFalse(Path(str(old_db) + "-wal").exists())
            self.assertFalse(Path(str(old_db) + "-shm").exists())
```

同时测试：不存在路径初始化v8；v7 export 在首次 append 前拒绝且目标事件数0；health/doctor 报新 schema/event/optimization identity；缺 schedule 的 RunCreated replay 失败。

- [ ] **Step 2: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_incompatible_cutover
```

Expected: FAIL，当前 `EventLedger` 会先打开连接/启用 WAL 并升级旧库。

- [ ] **Step 3: 在 SQLite write 前实现 identity preflight**

- 若路径不存在或 size=0，允许创建并写 schema8/event identity。
- 若路径存在，使用只读 URI connection 只查询 `PRAGMA user_version` 和 identity；关闭后确认没有创建 sidecar files。
- 只接受 exact schema8/event identity；删除自动 v7->v8 migration。
- error message 包含 archive command、新数据库要求，不执行任何 DDL/WAL pragma。

- [ ] **Step 4: 写旧路径删除 RED scan**

测试 active API/schema/server/projector/package/runtime 中以下 token 为0：public `samples_per_update`、`dsh-strict-origin-bundle@3` 及多版本兼容 accept branch、one-shot formal phase、old formal progress copy、active v3/v6 presets。必须明确保留唯一当前 sample execution protocol `dsh-strict-origin-bundle@4` 和 v4/v7 presets；它们不是旧 run 兼容层。扫描只允许显式白名单 `application/legacy_archive.py`、`scripts/archive_legacy_runs.py`、其归档测试和静态 docs 出现旧字段；golden Top2 symbols必须仍存在。再断言 active runtime import graph 不依赖 archive-only module。

- [ ] **Step 5: 删除/替换旧实现**

删除：

- `_phase_task_manifest(...,"formal")` 与 `_evaluate_candidate()` one-shot formal-500 分支；
- strict native path 的 candidate-only artifact/evaluation/checkpoint lookup；
- public API/UI `samples_per_update` 和 cells->origins reverse conversion；
- 旧 formal control 重评、slot-0 control 和兼容 projection；
- `dsh-strict-origin-bundle@3`、把 @3/@4 当集合接受的兼容分支，以及只服务 @3/旧 run 的 tests；
- installer/package/source tree 中 v3 presets、researcher/judge-v6（实际删除目录和 package-data，不只从 active list 隐藏）；
- README 中旧运行/旧 API 兼容说明。

保留：

- `_screen_candidate()`、`_prepare_formal_finalists()`、现有 Top-2 selector 和三个 screening events；
- outer proposer/coordinator/sample v4、researcher/judge-v7 的当前语义；
- 唯一 sample-agent execution identity `dsh-strict-origin-bundle@4`；
- post-run `formal_holdout_exposures`、validation/final-test 和 exposure registry；
- `run_archives`/archive API；
- 只用于 derived display 的 scoring-cell helpers。

- [ ] **Step 6: 重写受影响测试，不保留兼容断言**

删除旧 replay fixtures 的执行期望；用新 schema8 complete-run fixture 覆盖 protocol contracts、runtime binding、delivery scripts、import/export refusal。不得为了让旧测试通过重新引入 compatibility branch。

- [ ] **Step 7: 运行 focused GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_incompatible_cutover tests.test_removed_legacy_contract \
  tests.test_protocol_contracts tests.test_dsh_native_runtime \
  tests.test_formal_stage_protocol tests.test_scientific_exposure_registry \
  tests.test_candidate_parallel_evaluation
node --test integrations/dsh_ecology_plugin/test/install_runtime.test.mjs
```

Expected: PASS；Top-2 golden 仍通过。

- [ ] **Step 8: 版本升级和提交**

不兼容事件/API 切换将版本升级到 `0.4.0`，更新 plugin/package/release metadata：

```bash
git add src scripts integrations plugins pyproject.toml README.md tests
git commit -m "refactor: remove one-shot formal evolution protocol"
```

---

### Task 15: 完整回归、真实恢复 smoke、发布与 8777/8848 切换

**Files:**
- Create: `tests/test_top2_adaptive_e2e.py`
- Modify: `scripts/verify_delivery.sh`
- Modify: `scripts/build_delivery.sh`
- Modify: `RELEASE-CHECKLIST.md`
- Modify: `README.md`
- Verify: all source, Node, browser, artifact, archive, runtime paths

**Interfaces:**
- Produces one clean `0.4.0` wheel/sdist/delivery archive and matching DSH plugin package。
- Produces minimal deterministic smoke and full default smoke evidence。
- Does not migrate old database or import new events into old service。

- [ ] **Step 1: 写端到端 RED tests**

使用 fake DSH/evaluator 的快速 fixture 运行完整：4 candidates -> same64 -> Top2 golden -> two trajectories -> local edit -> three holdout arms -> champion/retain -> next-generation parent effective revision。断言默认每轮 budget1763/15867、五轮 plan8815/79335、unique origins1665、并发<=64、每个 truncation/restart 与无中断 event digest 相同。

- [ ] **Step 2: 运行 targeted/full tests**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_top2_adaptive_e2e tests.test_candidate_parallel_evaluation
make test
node --test integrations/dsh_ecology_plugin/test/*.test.mjs
node plugins/ecology_evolution/test/smoke.mjs
make verify
git diff --check
```

Expected: 全部通过；分别记录 Python unittest、Node test、browser smoke 的 test counts/skip reason/耗时。`make verify` 仅证明 source delivery/package 清单，不把它计作测试套件通过。

- [ ] **Step 3: 构建并验证 release artifacts**

只在已提交且干净工作树执行：

```bash
make release
make verify-artifacts
```

Expected: wheel、sdist、delivery archive、nested DSH tgz 和 source checksum 全部同源。

- [ ] **Step 4: 归档旧 runs 和旧数据库**

切换不是 migration。先停止8848新操作，通过旧8777暂停 active runs并等实际 in-flight/queue 为0；制作 pre-cancel SQLite online backup。取消剩余非终态 runs、归档所有 runs，运行 Task13命令并验证。停止旧8777后执行 WAL checkpoint/integrity/final backup：

```bash
sqlite3 "$OLD_DB" 'PRAGMA wal_checkpoint(FULL); PRAGMA integrity_check; PRAGMA user_version;'
sqlite3 "$OLD_DB" ".backup '$ARCHIVE_DIR/db/ecologyrsi-dsh-v7-final.sqlite3'"
sqlite3 "$ARCHIVE_DIR/db/ecologyrsi-dsh-v7-final.sqlite3" \
  'PRAGMA integrity_check; PRAGMA user_version;'
```

归档必须包含 pre-cancel/final DB、逐run export/projection、inventory、旧wheel/plugin/presets和 SHA256SUMS；验证后只读保存。

- [ ] **Step 5: 只停止已核实的 8777/8848 PID**

先用 `lsof -nP -iTCP:8777 -sTCP:LISTEN` 和8848命令解析具体 PID/command/cwd；发送 TERM 并等待退出，不使用 `killall`，不影响其他项目。确认两个查询均无 listener。

- [ ] **Step 6: 用全新数据库启动0.4.0**

数据库路径必须不存在，例如 `.runtime/top2-adaptive-0.4.0.sqlite3`。按 README 安装新 DSH runtime package，设置 runtime/tool token 后启动：

```bash
ecologyrsi-dsh serve \
  --host 127.0.0.1 \
  --port 8777 \
  --db "$NEW_DB"
```

再启动 `dsh --profile web --port 8848`。检查 health 精确报告 app0.4.0、ledger8、event identity、optimization protocol、DSH role catalog含 local editor。

- [ ] **Step 7: 运行最小合法真实恢复 smoke**

配置保持4/64/Top2，使用 `formal=100, batch=10, edits=2, holdout=169, concurrency=64`。运行到至少一个 local edit 后暂停、重启8777、恢复；证明同 scope origin/revision/usage 不重复并完成冠军决策。

- [ ] **Step 8: 启动完整默认真实 run**

从网页创建 `5轮、4候选、64初筛、Top2、每 finalist 500、batch50、max edits2、holdout169、sample concurrency64`。至少观察：

- 四候选共享screening digest并冻结同一Top2；
- 两 lane batch1 各完成若干 origin，run active<=64、route<=128、queue bounded；
- batch1结束后 local edit 只影响batch2；
- 页面进度/ETA/heartbeat持续变化，无长时间伪7%；
- 第一个169 holdout三臂使用同digest并产生deterministic decision。

若真实成本不适合等待完整5轮，交付门禁至少要求完整完成1轮；其余轮次可继续后台运行，但不能在第一轮冠军事件前声明运行验收通过。

- [ ] **Step 9: Go/No-Go 与回滚**

以下任一情况立即 No-Go：旧库hash变化、archive inventory不一致、SQLite/checksum失败、同run active>64、route>128、cohort digest不同、重复origin/revision/usage、comparison前generation前移、下一轮parent仍指outer R0。

回滚只允许：停止新8777/8848并封存新库；从只读归档复制一份 pre-cancel 旧库；恢复校验过的旧 wheel/plugin/presets；旧 binary 只连接旧库副本。禁止将新事件导入旧库，也禁止让新 binary 打开旧库。

- [ ] **Step 10: 提交 release metadata**

```bash
git add tests/test_top2_adaptive_e2e.py scripts README.md RELEASE-CHECKLIST.md \
  pyproject.toml integrations plugins
git commit -m "release: deliver top2 adaptive epoch evolution"
```

---

## 自审检查表

- [ ] 设计中每个 locked decision 都映射到至少一个任务和测试。
- [ ] Top-2 golden 在 Task0、Task14、Task15 三次运行。
- [ ] `500/50/169` 在 API、state、evaluator、projection 中都以 origins 为 source of truth。
- [ ] outer proposer 的 maximum_operations 仍为1；local editor 才读取1–5配置。
- [ ] winner final revision 已通过 effective identity 传给下一轮所有下游。
- [ ] capacity estimator 先于 run 创建使用，前端不以 row_count 猜测。
- [ ] RunSampleAdmission/provider-stage-gate 没有重复实现。
- [ ] selection holdout 与 post-run formal validation/final-test 分离。
- [ ] 旧数据库围栏发生在 WAL/DDL 之前。
- [ ] 计划中没有占位标记或未定义接口。
- [ ] 每个任务有 RED 命令、GREEN 命令和独立 commit。
