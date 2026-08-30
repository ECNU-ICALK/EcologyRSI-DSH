# Champion–Challenger Adaptive Trajectory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让每条 schema-v2 finalist 轨迹始终保留经过同 cohort 验证的冠军，把局部编辑生成的 revision 仅作为挑战者；只有挑战者通过同批配对比较门禁后才能替换冠军，最终 holdout 永远绑定最后一个已验证冠军。

**Architecture:** 保留外层 `top2_adaptive_epoch@1`、四候选 screening、Top-2 finalist 和三臂 holdout，只把冻结 schedule 升级为 v2 并在 formal lane 内加入 warm-up、双臂评测和不可变 `FormalBatchComparison`。schema v1 继续走既有 prequential 路径；schema v2 的状态权威来自 append-only arm evaluations、comparison decisions 和由 comparison 派生的 champion，局部编辑器只能从 champion-after 生成下一 challenger。

**Tech Stack:** Python 3.10–3.12、stdlib dataclass/Enum/unittest、append-only SQLite event ledger、Vanilla HTML/CSS/JavaScript、Node.js smoke tests、DSH Cordis plugin structured roles。

**Spec:** `docs/superpowers/specs/2026-08-30-champion-challenger-adaptive-trajectory-design.md`

## Global Constraints

- 实现前使用 `superpowers:using-git-worktrees` 建立隔离工作区；当前主工作区已有用户未提交修改 `plugins/ecology_evolution/assets/js/data.js` 和 `plugins/ecology_evolution/test/smoke.mjs`，不得覆盖、暂存、提交或丢弃这些修改。
- `optimization_protocol` 保持 `top2_adaptive_epoch@1`；仅由 schedule schema 和 `local_evaluation_mode` 选择 lane 语义。
- schema v1 必须接受 `ecologyrsi-dsh.top2-adaptive-epoch-schedule/1 + prequential`，并保留现有 replay/resume、projection 和 UI 语义。
- schema v2 必须接受 `ecologyrsi-dsh.top2-adaptive-epoch-schedule/2 + paired_champion_challenger`；`OptimizationSchedule.default()` 必须返回 v2。
- 任何 schema-v2 challenger 都不能在缺少 `FormalBatchCompared` 的情况下成为冠军或进入 holdout。
- schema-v2 batch 0 只执行一次 champion evaluation；batch 1..B-1 在 champion/challenger revision 不同时执行两次，在 revision 相同时复用同一不可变 evaluation。
- 比较只允许同一 candidate、batch、cohort、dataset/split/objective/evaluator contract；禁止使用相邻不同 cohort 的分数决定冠军。
- challenger promotion 必须同时满足：两侧网格完整有限、challenger safety 通过、单 cell 回退不超过 `1e-12`、`challenger.score - champion.score > 0.005`。
- 负分不是单独拒绝条件；`-0.4` 可以在全部相对门禁通过时替换 `-0.6`。
- schema-v2 最后一批比较完成后不得再调用 local editor、创建 child revision、写 local-edit outcome 或 activation。
- 不改变 final holdout 的三臂比较、bootstrap/置信区间、coverage、constraint 和 no-cell-regression 门禁。
- 每个行为改动都采用 RED → GREEN → REFACTOR：先运行指定测试并观察目标失败，再修改生产代码。
- 每个任务只暂存该任务列出的文件；提交前运行 `git diff --check`，并检查 `git status --short` 没有把用户改动带入提交。

---

## Task 1: 版本化 schedule 并修正配对执行预算

**Files:**
- Modify: `src/ecologyrsi_dsh/evolution/schedule.py`
- Modify: `src/ecologyrsi_dsh/evolution/__init__.py`
- Modify: `tests/test_optimization_schedule.py`

**Interfaces:**
- Export: `LEGACY_SCHEDULE_SCHEMA_VERSION`, `SCHEDULE_SCHEMA_VERSION`, `PREQUENTIAL_LOCAL_EVALUATION_MODE`, `PAIRED_LOCAL_EVALUATION_MODE`。
- `OptimizationSchedule.from_dict()` 接受两个精确 schema/mode 组合并拒绝交叉组合。
- `max_local_edit_decisions_per_finalist` 对 v2 返回 `batch_count - 1`、对 v1 返回 `batch_count`；保留 `max_local_edits_per_finalist` 作为操作数兼容属性，v2 返回 `(batch_count - 1) * max_local_edits_per_batch`，v1 返回 `batch_count * max_local_edits_per_batch`。
- `generation_execution_budget()` 对 v1 保持 1763/15867，对默认 v2 返回 2663/23967。

- [ ] **Step 1: 写 schedule RED tests**

增加以下断言：

```python
def test_default_schedule_uses_paired_champion_challenger_budget(self):
    schedule = OptimizationSchedule.default()
    self.assertEqual(
        schedule.schema_version,
        "ecologyrsi-dsh.top2-adaptive-epoch-schedule/2",
    )
    self.assertEqual(schedule.local_evaluation_mode, "paired_champion_challenger")
    self.assertEqual(schedule.max_local_edit_decisions_per_finalist, 9)
    self.assertEqual(
        schedule.generation_execution_budget(cells_per_origin=9),
        {
            "screening_candidate_origins": 256,
            "formal_candidate_origins": 1900,
            "holdout_candidate_origins": 507,
            "total_candidate_origins": 2663,
            "total_scoring_cells": 23967,
        },
    )

def test_legacy_schedule_keeps_prequential_budget(self):
    value = OptimizationSchedule.default().to_dict()
    value.update(
        schema_version="ecologyrsi-dsh.top2-adaptive-epoch-schedule/1",
        local_evaluation_mode="prequential",
    )
    schedule = OptimizationSchedule.from_dict(value)
    self.assertEqual(schedule.generation_execution_budget(cells_per_origin=9)["total_candidate_origins"], 1763)
    self.assertEqual(schedule.required_unique_origins(5), 1665)
```

同时用 subtests 断言 v1+paired、v2+prequential、未知 schema 和未知 mode 均抛 `ValueError`。

- [ ] **Step 2: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v tests.test_optimization_schedule
```

Expected: 默认 schema/mode、v2 formal budget 和决策上限断言失败。

- [ ] **Step 3: 实现精确版本映射和 mode-aware 预算**

生产代码采用明确映射，不用 `startswith`：

```python
LEGACY_SCHEDULE_SCHEMA_VERSION = "ecologyrsi-dsh.top2-adaptive-epoch-schedule/1"
SCHEDULE_SCHEMA_VERSION = "ecologyrsi-dsh.top2-adaptive-epoch-schedule/2"
PREQUENTIAL_LOCAL_EVALUATION_MODE = "prequential"
PAIRED_LOCAL_EVALUATION_MODE = "paired_champion_challenger"

_SCHEDULE_MODES = {
    LEGACY_SCHEDULE_SCHEMA_VERSION: PREQUENTIAL_LOCAL_EVALUATION_MODE,
    SCHEDULE_SCHEMA_VERSION: PAIRED_LOCAL_EVALUATION_MODE,
}
```

v2 formal occurrence 上限按下式计算：

```python
if self.local_evaluation_mode == PAIRED_LOCAL_EVALUATION_MODE:
    per_finalist = self.local_batch_origin_count + (
        2 * (self.batch_count - 1) * self.local_batch_origin_count
    )
    formal = per_finalist * self.finalist_count
else:
    formal = self.formal_origin_count_per_finalist * self.finalist_count
```

- [ ] **Step 4: 运行 GREEN 和兼容性检查**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v tests.test_optimization_schedule
PYTHONPATH=src .venv/bin/python -m unittest -v tests.test_http tests.test_runtime_integration
```

Expected: 全部通过；旧 schedule 字典仍能解析。

- [ ] **Step 5: 提交 schedule 版本化**

```bash
git add src/ecologyrsi_dsh/evolution/schedule.py \
  src/ecologyrsi_dsh/evolution/__init__.py \
  tests/test_optimization_schedule.py
git commit -m "feat: version champion challenger schedule"
```

---

## Task 2: 建立 formal batch arm 和不可变 comparison 模型

**Files:**
- Modify: `src/ecologyrsi_dsh/core/trajectory.py`
- Modify: `src/ecologyrsi_dsh/core/__init__.py`
- Modify: `src/ecologyrsi_dsh/__init__.py`
- Modify: `tests/test_trajectory_models.py`

**Interfaces:**
- Add enum: `FormalBatchArm.CHAMPION`, `FormalBatchArm.CHALLENGER`。
- Add enum: `FormalBatchComparisonDecision.INITIAL_CHAMPION`, `CHALLENGER_PROMOTED`, `CHAMPION_RETAINED`。
- Extend `EvaluationScope.formal_batch_arm: FormalBatchArm | None = None` and serialize it.
- Add immutable `FormalBatchComparison` with exact identity, evidence, gate and decision fields from the approved spec。
- Add `BatchEvaluation.evaluation_digest`, computed from its full `to_dict()` payload。

- [ ] **Step 1: 写 arm 和 comparison RED tests**

覆盖以下规则：

```python
def test_formal_batch_scope_round_trips_arm(self):
    scope = make_formal_scope(formal_batch_arm=FormalBatchArm.CHALLENGER)
    self.assertEqual(EvaluationScope.from_dict(scope.to_dict()), scope)

def test_non_formal_scope_rejects_formal_batch_arm(self):
    with self.assertRaisesRegex(ValueError, "formal_batch_arm"):
        make_screening_scope(formal_batch_arm=FormalBatchArm.CHAMPION)

def test_comparison_rejects_inconsistent_champion_after(self):
    with self.assertRaisesRegex(ValueError, "champion_after"):
        make_comparison(
            decision=FormalBatchComparisonDecision.CHALLENGER_PROMOTED,
            champion_after_revision_id="revision:champion",
        )
```

另断言 warm-up `INITIAL_CHAMPION` 必须使用 batch 0、两侧 revision/evaluation 相同、delta 为 0；`CHAMPION_RETAINED` 与 `CHALLENGER_PROMOTED` 必须绑定正确 champion-after；所有 score 和 delta 有限；digest 字段为 sha256。

- [ ] **Step 2: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v tests.test_trajectory_models
```

Expected: 新 enum、scope 字段和 comparison 类型尚不存在导致失败。

- [ ] **Step 3: 实现类型、校验和序列化**

comparison 至少包含以下字段，命名与 event payload 保持一致：

```python
@dataclass(frozen=True, slots=True)
class FormalBatchComparison:
    comparison_id: str
    run_id: str
    generation: int
    candidate_id: str
    batch_index: int
    cohort_digest: str
    champion_before_revision_id: str
    challenger_revision_id: str
    champion_evaluation_id: str
    challenger_evaluation_id: str
    champion_evaluation_digest: str
    challenger_evaluation_digest: str
    champion_score: float
    challenger_score: float
    score_delta: float
    comparison_contract_digest: str
    safety_gate_passed: bool
    cell_regression_gate_passed: bool
    minimum_score_delta: float
    decision: FormalBatchComparisonDecision
    champion_after_revision_id: str
    reason: str
    created_at: str = field(default_factory=utc_now)
```

`EvaluationScope` 的兼容规则为：legacy formal evaluation 可以不带 arm；v2 是否必须带 arm由 director 根据 schedule 校验，避免 model 层破坏历史 v1 payload。

- [ ] **Step 4: 运行 GREEN 和导入回归**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_trajectory_models \
  tests.test_http
```

- [ ] **Step 5: 提交 durable model**

```bash
git add src/ecologyrsi_dsh/core/trajectory.py \
  src/ecologyrsi_dsh/core/__init__.py \
  src/ecologyrsi_dsh/__init__.py \
  tests/test_trajectory_models.py
git commit -m "feat: model paired formal comparisons"
```

---

## Task 3: 实现 Host-owned challenger selection gate

**Files:**
- Create: `src/ecologyrsi_dsh/evolution/champion_challenger.py`
- Modify: `src/ecologyrsi_dsh/evolution/__init__.py`
- Create: `tests/test_champion_challenger.py`

**Interfaces:**
- Export: `LOCAL_MINIMUM_SCORE_DELTA = 0.005`，引用 promotion 模块的 `V2_MINIMUM_SCORE_DELTA`，不能复制漂移值。
- Export: `LOCAL_CELL_REGRESSION_TOLERANCE = 1e-12`。
- Export pure function: `assess_local_challenger(champion, challenger, *, challenger_safety_gate_passed) -> LocalChallengerAssessment`。
- Assessment 提供 `score_delta`, `comparison_contract_digest`, `safety_gate_passed`, `cell_regression_gate_passed`, `decision`, `champion_after_revision_id`, `reason`。

- [ ] **Step 1: 写 gate truth-table RED tests**

使用显式 fixtures 覆盖：

1. `-0.6 → -0.4`、完整同 contract、无 cell regression、safety pass：`challenger_promoted/challenger_improved`。
2. `0.20 → 0.205`：严格 `> 0.005` 不成立，`champion_retained/below_practical_delta`。
3. 分数提高但任一 target/horizon skill 降低超过 `1e-12`：`challenger_cell_regression`。
4. cohort、origin count、evaluator digest 或 metric contract 不同：`incompatible_comparison_contract`。
5. champion/challenger 缺 cell、含 NaN/Inf：分别命中 `champion_evaluation_incomplete` 或 `challenger_evaluation_incomplete`。
6. challenger safety fail：`challenger_safety_gate_failed`。
7. challenger 更低或相等：`below_practical_delta`。

- [ ] **Step 2: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v tests.test_champion_challenger
```

Expected: 模块不存在。

- [ ] **Step 3: 实现确定性检查顺序**

函数必须按批准的 reason 顺序短路：

```python
_REASON_ORDER = (
    "incompatible_comparison_contract",
    "champion_evaluation_incomplete",
    "challenger_evaluation_incomplete",
    "challenger_safety_gate_failed",
    "challenger_cell_regression",
    "below_practical_delta",
    "challenger_improved",
)
```

从 `metrics["targets"]` 读取 `(target, horizon_hours, skill_score)`，先验证 key 集完全一致和全部有限，再逐 cell 比较：

```python
cell_regression_gate_passed = all(
    challenger_cells[key] + LOCAL_CELL_REGRESSION_TOLERANCE
    >= champion_cells[key]
    for key in champion_cells
)
```

comparison contract digest 只包含冻结 contract identity，不包含 score、predictions 或创建时间。

- [ ] **Step 4: 运行 GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_champion_challenger \
  tests.test_generation_comparison
```

- [ ] **Step 5: 提交选择策略**

```bash
git add src/ecologyrsi_dsh/evolution/champion_challenger.py \
  src/ecologyrsi_dsh/evolution/__init__.py \
  tests/test_champion_challenger.py
git commit -m "feat: add host challenger selection gate"
```

---

## Task 4: 持久化、回放和校验双臂 evidence/comparison

**Files:**
- Modify: `src/ecologyrsi_dsh/core/state.py`
- Modify: `src/ecologyrsi_dsh/core/director.py`
- Modify: `tests/test_trajectory_event_replay.py`
- Modify: `tests/test_director_invariants.py`

**Interfaces:**
- Add `RunState.formal_batch_comparisons`。
- Change evaluation index from `(candidate_id, batch_index)` to `(candidate_id, batch_index, arm | None)`。
- Extend `batch_evaluation_for(candidate_id, batch_index, arm=None)`：v1 returns unarmed evaluation；v2 no-arm alias returns warm-up champion or later challenger。
- Add `batch_comparison_for(candidate_id, batch_index)` and `trajectory_champion_revision_id(candidate_id)`。
- Add director `record_formal_batch_comparison(run_id, comparison)` writing `FormalBatchCompared` with deterministic event id。

- [ ] **Step 1: 写 replay/director RED tests**

测试事件序列：trajectory start → batch start → champion evaluated → challenger evaluated → compared。断言：

```python
self.assertEqual(
    replayed.batch_evaluation_for(candidate_id, 1, FormalBatchArm.CHAMPION),
    champion_evaluation,
)
self.assertEqual(
    replayed.batch_evaluation_for(candidate_id, 1, FormalBatchArm.CHALLENGER),
    challenger_evaluation,
)
self.assertEqual(
    replayed.trajectory_champion_revision_id(candidate_id),
    comparison.champion_after_revision_id,
)
```

另覆盖：相同 event id+相同 payload 幂等；相同 key+冲突 payload 拒绝；两臂 cohort/candidate/batch 不同拒绝；evaluation digest 不匹配拒绝；decision/champion-after 不一致拒绝；v1 unarmed ledger fixture replay 结果不变。

- [ ] **Step 2: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_trajectory_event_replay \
  tests.test_director_invariants
```

- [ ] **Step 3: 扩展 state 索引和 replay**

使用明确 key：

```python
def _evaluation_key(evaluation: BatchEvaluation) -> tuple[str, int, str | None]:
    arm = evaluation.scope.formal_batch_arm
    return (
        evaluation.scope.candidate_id,
        int(evaluation.scope.batch_index),
        arm.value if arm is not None else None,
    )
```

schema v2 director 规则：batch 0 只接受 champion arm；later batch 接受 champion/challenger；schema v1 只接受 `None`。comparison 回放先解析 `FormalBatchComparison.from_dict()`，再解析并比对两侧 evaluation、batch cohort 和 revisions，最后 `setdefault`。

- [ ] **Step 4: 实现 comparison director 入口**

director 必须重新验证 durable evidence，不信任 caller 传入的 score/digest：

```python
event_id = (
    f"{run_id}:generation:{comparison.generation}:trajectory:"
    f"{comparison.candidate_id}:batch:{comparison.batch_index}:compared"
)
```

相同 revision 两侧允许引用同一个 evaluation id/digest；不同 revision 必须引用不同 arm evaluations。

- [ ] **Step 5: 运行 GREEN 和全量 replay 回归**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_trajectory_event_replay \
  tests.test_director_invariants \
  tests.test_genome_replay \
  tests.test_screening_replay \
  tests.test_runtime_integration
```

- [ ] **Step 6: 提交事件模型**

```bash
git add src/ecologyrsi_dsh/core/state.py \
  src/ecologyrsi_dsh/core/director.py \
  tests/test_trajectory_event_replay.py \
  tests/test_director_invariants.py
git commit -m "feat: persist formal batch comparisons"
```

---

## Task 5: 实现 v2 warm-up、paired batch、恢复和最终批次状态机

**Files:**
- Modify: `src/ecologyrsi_dsh/api/formal_trajectory.py`
- Modify: `src/ecologyrsi_dsh/api/work_units.py`
- Modify: `src/ecologyrsi_dsh/api/auto_progress.py`
- Modify: `tests/test_formal_trajectory.py`
- Modify: `tests/test_work_units.py`
- Modify: `tests/test_auto_progress.py`

**Interfaces:**
- 保留公开入口 `execute_next_formal_batch()` 和 `execute_next_local_edit()`，入口内部按 schedule mode 分派 legacy/v2。
- v2 formal work unit 每次只补一个 durable boundary：missing arm evaluation、missing comparison、missing proposal、missing challenger decision/activation 或 trajectory completion。
- 新 challenger 的 `parent_revision_id` 始终为当前 comparison 的 `champion_after_revision_id`。
- v2 activation 表示“下一 challenger 已排程”，不能表示“已晋升冠军”；champion authority 仅来自 comparison。

- [ ] **Step 1: 写状态机 RED tests**

至少包含这些独立测试：

```python
def test_v2_warmup_evaluates_once_and_schedules_challenger(self): ...
def test_v2_paired_batch_evaluates_distinct_revisions_on_same_cohort(self): ...
def test_retained_champion_is_parent_of_next_challenger(self): ...
def test_promoted_challenger_is_parent_of_next_challenger(self): ...
def test_same_revision_pair_reuses_one_evaluation(self): ...
def test_last_batch_completes_without_local_edit_or_child(self): ...
def test_resume_after_each_boundary_executes_only_missing_work(self): ...
def test_v1_prequential_execution_is_unchanged(self): ...
```

对 final batch 断言 local proposal/outcome/activation 数量均为 `batch_count - 1`，`trajectory.final_revision_id == last_comparison.champion_after_revision_id`。

- [ ] **Step 2: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_formal_trajectory \
  tests.test_work_units \
  tests.test_auto_progress
```

- [ ] **Step 3: 分离 legacy 和 v2 分派**

保留现有 prequential 函数主体，改名为私有 legacy helper；新增 `_execute_next_paired_formal_batch` 和 `_execute_next_paired_local_edit`。顶层只做冻结 schedule 选择：

```python
schedule = OptimizationSchedule.from_dict(
    state.task_manifest.metadata["optimization_schedule"]
)
if schedule.local_evaluation_mode == PREQUENTIAL_LOCAL_EVALUATION_MODE:
    return _execute_next_prequential_formal_batch(endpoint, state, candidate_id)
return _execute_next_paired_formal_batch(endpoint, state, candidate_id)
```

- [ ] **Step 4: 实现 warm-up 与 paired evaluation**

batch 0：start batch、champion arm evaluate、record initial comparison。later batch：从 previous comparison 取 champion-before，从 previous activation 取 challenger；先补 champion arm，再补 challenger arm；identity 相同则 comparison 两侧引用 champion evaluation，不再次执行 evaluator。

评测 scope 必须显式设置 arm：

```python
scope = EvaluationScope(
    run_id=run_id,
    generation=batch.generation,
    candidate_id=candidate_id,
    candidate_revision_id=revision_id,
    phase=EvaluationPhase.FORMAL_BATCH,
    cohort_digest=batch.cohort_digest,
    origin_count=batch.origin_count,
    batch_index=batch.batch_index,
    formal_batch_arm=arm,
)
```

- [ ] **Step 5: 实现 comparison、下一 challenger 和 final completion**

later batch 调 `assess_local_challenger()`，构造并记录 `FormalBatchComparison`。只有非 final batch 才调用 local editor。local editor evidence 同时包含 comparison payload、两侧 bounded metrics、最近 accepted/rejected operations；应用 mutation 时把 selected champion genome/revision 作为唯一 parent。

final branch必须先于 local editor：

```python
if batch.batch_index == batch.batch_count - 1:
    endpoint.director.complete_formal_trajectory(
        run_id,
        candidate_id,
        comparison.champion_after_revision_id,
    )
    return True
```

- [ ] **Step 6: 修正 scheduler/progress completion 条件**

v1 继续按 activation 数推进；v2 按 comparison 数确认 completed batches，并把 `batch_count - 1` 作为 local decision/activation 上限。失败上下文使用 schedule budget，而不是硬编码旧 formal total。

- [ ] **Step 7: 运行 GREEN、恢复和并发回归**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_formal_trajectory \
  tests.test_work_units \
  tests.test_auto_progress \
  tests.test_candidate_parallel_evaluation \
  tests.test_sample_admission
```

- [ ] **Step 8: 提交状态机**

```bash
git add src/ecologyrsi_dsh/api/formal_trajectory.py \
  src/ecologyrsi_dsh/api/work_units.py \
  src/ecologyrsi_dsh/api/auto_progress.py \
  tests/test_formal_trajectory.py \
  tests/test_work_units.py \
  tests/test_auto_progress.py
git commit -m "feat: execute champion challenger trajectories"
```

---

## Task 6: 将 holdout 和 reflection 绑定到 durable champion

**Files:**
- Modify: `src/ecologyrsi_dsh/api/generation_execution.py`
- Modify: `src/ecologyrsi_dsh/evolution/batches.py`
- Modify: `tests/test_candidate_parallel_evaluation.py`
- Modify: `tests/test_adaptive_reflection_evidence.py`

**Interfaces:**
- Holdout finalist arm 只从 completed trajectory 的 `final_revision_id` 解析 revision。
- schema-v2 trajectory completion 前必须存在最后一批 comparison，且 final revision 等于其 champion-after。
- Generation reflection 追加最近 comparison decision/reason/delta 和 rejected challenger operations；不得放入 raw predictions、raw labels、完整 timestamps 或自由文本 rationale。
- 两条 lane 都未通过 final holdout 时，existing global incumbent 同时保留为 `incumbent_after_candidate_id` 和 `search_parent_candidate_id`。

- [ ] **Step 1: 写 holdout/reflection RED tests**

构造 final batch challenger 被拒绝的 lane，断言 holdout scope 的 `candidate_revision_id` 是 retained champion，不是最后 challenger。构造 accepted/rejected 历史，断言 reflection bounded evidence 同时包含 reason code 和 operation targets，而 mutation parent 仍为 retained champion。

- [ ] **Step 2: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_candidate_parallel_evaluation \
  tests.test_adaptive_reflection_evidence
```

- [ ] **Step 3: 增加 champion binding invariant 和 bounded evidence**

在启动 holdout 前验证：

```python
comparison = state.batch_comparison_for(
    trajectory.candidate_id,
    trajectory.batch_count - 1,
)
if (
    schedule.local_evaluation_mode == PAIRED_LOCAL_EVALUATION_MODE
    and (
        comparison is None
        or trajectory.final_revision_id != comparison.champion_after_revision_id
    )
):
    raise ValueError("paired trajectory final revision is not the durable champion")
```

reflection 只输出固定字段和有界最近记录；不得重新计算历史冠军。

- [ ] **Step 4: 运行 GREEN 和 promotion 回归**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_candidate_parallel_evaluation \
  tests.test_adaptive_reflection_evidence \
  tests.test_generation_comparison \
  tests.test_promotion
```

- [ ] **Step 5: 提交 champion 消费路径**

```bash
git add src/ecologyrsi_dsh/api/generation_execution.py \
  src/ecologyrsi_dsh/evolution/batches.py \
  tests/test_candidate_parallel_evaluation.py \
  tests/test_adaptive_reflection_evidence.py
git commit -m "feat: bind holdout to trajectory champion"
```

---

## Task 7: 更新 public events、projection、创建请求和 UI 语义

**Files:**
- Modify: `src/ecologyrsi_dsh/api/shared.py`
- Modify: `src/ecologyrsi_dsh/api/events.py`
- Modify: `src/ecologyrsi_dsh/api/projection.py`
- Modify: `tests/test_execution_projection.py`
- Modify: `tests/test_http.py`
- Modify: `plugins/ecology_evolution/assets/js/commands.js`
- Modify: `plugins/ecology_evolution/assets/js/render_process.js`
- Modify: `plugins/ecology_evolution/test/smoke.mjs`
- Modify: `README.md`

**Interfaces:**
- Public timeline supports `FormalBatchCompared`，标题和摘要不把 challenger creation 描述为 acceptance。
- v2 trajectory projection 每批提供 champion-before/challenger ids、两侧 score、delta、threshold、decision/reason、champion-after 和 next challenger operation。
- v1 projection 保持原字段并增加 `strategy_label="旧版连续更新策略"`。
- plugin create request 默认发送 v2 schema/mode。
- UI 对 v2 禁止使用泛化状态 `已应用`；只显示 spec 批准的五类中文标签。
- 容量摘要显示 2663 conservative origin occurrences、23967 scoring cells，并说明 unique source origins 不变。

- [ ] **Step 1: 写 API/projection RED tests**

断言 `FormalBatchCompared` 能投影；v2 batch row 包含：

```python
expected = {
    "champion_before_revision_id": champion_id,
    "challenger_revision_id": challenger_id,
    "champion_score": 0.21,
    "challenger_score": 0.19,
    "score_delta": -0.02,
    "minimum_score_delta": 0.005,
    "comparison_decision": "champion_retained",
    "comparison_reason": "below_practical_delta",
    "champion_after_revision_id": champion_id,
}
```

HTTP test 断言默认 create manifest 为 v2；显式传 legacy v1 schedule 仍成功创建并回显 v1。

- [ ] **Step 2: 写 browser RED assertions**

在现有 `smoke.mjs` 的用户改动基础上增量编辑，不能用 worktree 旧版本覆盖。增加 v2 fixture 并断言：

```javascript
assert.match(html, /挑战者未改善，继续使用原冠军/);
assert.doesNotMatch(html, />已应用</);
assert.equal(request.optimization_schedule.local_evaluation_mode, "paired_champion_challenger");
```

再增加 v1 fixture，断言 `旧版连续更新策略` 和原有 legacy 文案仍可见。

- [ ] **Step 3: 运行 RED**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_execution_projection \
  tests.test_http
node plugins/ecology_evolution/test/smoke.mjs
```

- [ ] **Step 4: 实现 event/projection 和前端映射**

v2 状态映射固定为：

```javascript
const pairedDecisionLabels = {
  initial_champion: "初始冠军已冻结",
  challenger_promoted: "挑战者已晋升为轨迹冠军",
  champion_retained: "挑战者未改善，继续使用原冠军",
};
```

当 comparison 后已有下一 challenger activation 时显示 `下一挑战版本已生成，等待同 cohort 对照验证`；Host 拒绝 proposal 时显示 `局部提案未通过宿主校验`。v1 才可继续解释 `LocalEditOutcome.APPLIED` 为旧版应用状态。

- [ ] **Step 5: 更新 README 数字和含义**

README 必须同时写出：默认每轮最多 2663 次 candidate-origin execution occurrences、其中 formal paired upper bound 1900、总 scoring cells 23967；500 formal unique origins 被两个 finalist/lane 和同 cohort 双臂确定性复用，不代表新增独立源数据。

- [ ] **Step 6: 运行 GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_execution_projection \
  tests.test_http \
  tests.test_adaptive_reflection_evidence
node plugins/ecology_evolution/test/smoke.mjs
```

- [ ] **Step 7: 提交 public/UI 契约**

```bash
git add src/ecologyrsi_dsh/api/shared.py \
  src/ecologyrsi_dsh/api/events.py \
  src/ecologyrsi_dsh/api/projection.py \
  tests/test_execution_projection.py \
  tests/test_http.py \
  plugins/ecology_evolution/assets/js/commands.js \
  plugins/ecology_evolution/assets/js/render_process.js \
  plugins/ecology_evolution/test/smoke.mjs \
  README.md
git commit -m "feat: expose champion challenger decisions"
```

---

## Task 8: 完整回归、失败注入和交付检查

**Files:**
- Modify if required by observed regression only: code/tests already listed in Tasks 1–7
- Modify: `CHANGELOG.md`

- [ ] **Step 1: 运行聚焦行为矩阵**

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_optimization_schedule \
  tests.test_trajectory_models \
  tests.test_champion_challenger \
  tests.test_trajectory_event_replay \
  tests.test_director_invariants \
  tests.test_formal_trajectory \
  tests.test_work_units \
  tests.test_auto_progress \
  tests.test_execution_projection \
  tests.test_adaptive_reflection_evidence \
  tests.test_candidate_parallel_evaluation \
  tests.test_http
```

Expected: 全部通过，且日志没有重复 arm execution 或 final-batch local edit。

- [ ] **Step 2: 运行完整 Python 和 browser suites**

```bash
make test
node --test integrations/dsh_ecology_plugin/test/*.test.mjs
node plugins/ecology_evolution/test/smoke.mjs
```

Expected: 全部 exit 0。

- [ ] **Step 3: 运行交付验证**

```bash
make verify
```

Expected: packaging、runtime、plugin 和文档检查全部 exit 0。若命令暴露与功能无关的既有失败，记录精确命令、失败断言和未修改证据，不得把它报告为本功能通过。

- [ ] **Step 4: 做 ledger/resume 失败注入审计**

在 `tests.test_formal_trajectory` 逐一运行 evaluation、comparison、proposal、decision、activation 和 completion 边界的恢复 tests；确认每种恢复都不重复 evaluator、LLM proposal 或 child revision identity。

```bash
PYTHONPATH=src .venv/bin/python -m unittest -v \
  tests.test_formal_trajectory.FormalTrajectoryTests.test_resume_after_each_boundary_executes_only_missing_work
```

- [ ] **Step 5: 检查 schema、文案和占位符**

```bash
rg -n 'top2-adaptive-epoch-schedule/1|top2-adaptive-epoch-schedule/2|prequential|paired_champion_challenger' \
  src tests plugins README.md
rg -n '已应用' plugins/ecology_evolution/assets/js plugins/ecology_evolution/test
rg -n 'NotImplementedError|^[[:space:]]*pass[[:space:]]*$|FIXME|XXX' \
  src tests plugins README.md CHANGELOG.md
git diff --check
git status --short
```

Expected: v1 只存在于兼容分支/fixtures；v2 为默认；`已应用` 只存在于明确 legacy 分支或 legacy test；没有新占位符；用户原有文件改动没有被覆盖。

- [ ] **Step 6: 更新变更记录并提交验证结果**

在 `CHANGELOG.md` 当前版本的 Changed/Fixed 小节记录：same-cohort paired validation、durable champion、legacy schedule compatibility、capacity upper bound 和 UI wording correction。

```bash
git add CHANGELOG.md
git commit -m "docs: record champion challenger evolution"
```

- [ ] **Step 7: 按 `superpowers:requesting-code-review` 做独立审查**

审查范围从本功能分支起点到 HEAD，重点检查八条 acceptance criteria、schema-v1 replay、final-batch 无 child、负分相对改善和 UI 无误导。修复任何 P1/P2 问题后重跑 Step 1–3。

- [ ] **Step 8: 按 `superpowers:verification-before-completion` 汇总证据**

最终汇报必须给出：功能分支 commit 列表、聚焦测试/全量测试/browser/verify 的最新 exit code、实际 test count、用户未提交修改保留状态，以及尚未执行的集成动作。未获得用户许可前不得合并、rebase、push 或删除 worktree。
