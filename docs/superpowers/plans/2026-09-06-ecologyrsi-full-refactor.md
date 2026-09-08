# Codex 分阶段实施计划

> 给执行者：先读 [CODEX_START.md](CODEX_START.md) 和 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。用户已授权整体重构，不需要重新交付一份泛泛建议；也不意味着获准调用付费服务或改动运行中的数据。

**Goal:** 将上游收口为一个可信科学实验内核，并以可验证产物支撑对象层与方法层进化。

**Architecture:** 模块化单体。保留原 DSH 插件、安装器和必要旧运行回放接口；新写路径通过同一合同、评测与晋级服务。

**Tech Stack:** 上游现有 Python 3.10+、标准库、SQLite、DSH/Node 插件。参考包没有外部运行时依赖。

**Spec:** [ARCHITECTURE.md](ARCHITECTURE.md)、[MIGRATION.md](MIGRATION.md)、[CURRENT_CODE.md](CURRENT_CODE.md)。

## 执行前必须知道

本包不是上游的补丁集，不要声称 T00—T19 已完成。已有代码解决了若干核心约束的可运行示例，目标任务还要迁移原生能力、处理历史协议并完成上游测试。

任务中的 `kernel/...`、`storage/...` 等路径默认相对于**上游** `src/ecologyrsi_dsh/`；`tests/refactor/...` 是需要新增的目标测试，不是本包已有测试路径。旧测试模块名必须以本地发现结果校正。

参考包可直接调用的是 `ProgramGenome.from_dict`、`compile_program`、`build_task`、`BatchBackend.run`、`evaluate`、`compare`、`Ledger.promote` 和 `run_search`。目标方案中更丰富的 DTO、独立认证和发布对象是**待实现合同**，不能 import 不存在的类后声称已经接入。

## 全局执行约束

- 不覆盖用户工作区、数据库、数据文件、原始安装说明。
- 先运行参考包，再记录上游 HEAD、差异和原测试基线。
- 先写回归测试，确认暴露缺陷或缺少功能，再实现；每任务记录真实命令与退出码。
- 原生算法、评测规则与旧摘要不能在“重构”中悄悄替换成演示算法或新摘要。
- 新建运行只启用一套写路径；影子模式只比较结果，不进行第二次晋级或付费执行。
- 真实 DSH、数据许可、密钥、数据库迁移及发布涉及外部边界时，记录具体阻塞项，不伪造成功。
- 有 Superpowers 时使用 executing-plans / subagent-driven-development 与 verification-before-completion；没有时按同样验收步骤执行，不作为运行时依赖。

## 19. Codex 分阶段实施任务

每个任务都遵循：先补测试并确认失败 → 实现 → 运行目标测试 → 运行受影响旧测试 → 更新进度 → 独立提交。不要一次移动所有文件后才开始测试。

**任务完成记录格式：** `task_id / HEAD / changed_files / tests_run / exit_codes / status / known_gaps / next_action`。

代码量大时可以把一个任务拆成多个小提交，但不得跳过任务出口条件。以下新测试文件均使用 `unittest.TestCase`，添加 `tests/refactor/__init__.py`，避免 `unittest discover` 无法递归发现。

### T00 — 建立可复现基线，不改业务行为

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** 无。

**文件：** 新增 `scripts/refactor/inspect_baseline.py`、`docs/refactor/BASELINE.md`、`docs/refactor/PROGRESS.md`、`tests/refactor/__init__.py`。

**动作：** 读取本地 HEAD、dirty 状态、Python/SQLite/Node/DSH 版本；记录原有导入路径和命令入口；运行现有离线测试；识别需要凭据/真实数据的验收命令并保持关闭。基线脚本读取信息而不自动创建分支、修改依赖或启动服务。

```bash
git status --short
git rev-parse HEAD
python --version
python -c "import sqlite3; print(sqlite3.sqlite_version)"
make test-fast
make compile
make test
```

`make test` 运行前先检查测试是否会读取本地凭据或发真实请求；外部验收只能在明确授权的环境执行。失败必须完整保留，不能为了得到绿色基线先改断言。

**验收：** BASELINE 包含真实结果和未运行原因；源代码与旧数据未变化；本地 HEAD 与本文基线有差异时，记录受影响模块而不回退。建议提交 `docs: record kernel refactor baseline`。

### T01 — 统一不可变对象与 Genome 身份

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T00。

**文件：** 新增 `kernel/identity.py`、`kernel/genome.py`、`kernel/errors.py`、`tests/refactor/test_identity.py`；迁移阶段不删除旧 `evolution/genome.py`。

**输出接口：** [身份实现](../src/ecologyrsi_kernel/kernel/identity.py) 的 `FrozenObject/content_id`；`ProgramGenome.from_dict()`、`ProgramGenome.behavior_id`、`ProgramGenome.genome_id`、`ResearchPolicyGenome.from_dict()`。

**实现：** 源级行为和谱系分开；所有输入复制后冻结；严格 schema；旧版本摘要走旧实现，新 schema 使用新 hash namespace。`from_dict()` 不把任意值 `str()` 化来掩盖类型错误。

**关键测试：** [自动化测试目录](../tests/) 的 `IdentityTests` 全部迁移；增加相同有效行为不同谱系、未知字段、错误 schema、父代循环引用由仓库层拒绝的测试。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_identity -v
PYTHONPATH=src python -m unittest tests.test_evolution_genome tests.test_genome_replay -v
```

**出口：** 输入/输出嵌套修改均不影响身份；单改说明不算新行为；旧身份测试不变。提交 `feat: introduce immutable kernel genome identities`。

### T02 — 冻结合同与后端无关编译器

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T01。

**文件：** 新增 `kernel/contracts.py`、`kernel/registry.py`、`kernel/compiler.py`、`tests/refactor/test_contracts.py`、`test_compiler.py`。

**输出接口：** `TaskContract.from_dict()`、`EvaluationContext.from_dict()`、`compile_program(genome, registry_snapshot, compiler_policy)`、`CompiledProgram.compiled_id`。

**实现：** schema 验证类型/单位/默认值；实际样本身份、种子、视图与指标进入 context；候选身份不进入比较 context；能力解析绑定内容摘要；编译规范化有效默认值；未知能力拒绝；源不同但编译相同返回 `NoEffectiveChange`。

**验收断言：**

```text
仅更换候选算法，其他评测条件不变 → context_id 相同、compiled_id 不同
样本数相同但任意 case 不同         → context_id 不同
参数缺省与显式默认值              → compiled_id 相同
未知能力/不兼容单位/带环 DAG        → 编译失败，未产生执行请求
```

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_contracts tests.refactor.test_compiler -v
PYTHONPATH=src python -m unittest tests.test_workflow_ir tests.test_algorithm_compilation -v
```

**出口：** 编译器不导入 DSH/SQLite，不发送模型请求；相同输入可重放。提交 `feat: compile backend-independent scientific programs`。

### T03 — 统一数据库、产物库与追加证据

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T01—T02。

**文件：** 新增 `storage/database.py`、`storage/artifacts.py`、`storage/repositories.py`、`storage/migrations/001_kernel.sql`、`tests/refactor/test_storage.py`。

**输入：** 已验证的不可变对象和编译计划。

**输出接口：** `Database.transaction()`、`ArtifactStore.put_bytes(data, media_type)`、`ArtifactStore.read_verified(artifact_id)`、`EvidenceRepository.append_evaluation()`、`append_comparison()`。

**实现：** [基础数据库 schema](../src/ecologyrsi_kernel/storage/schema.sql) 为冠军事务的基础表；同一迁移另建 `artifacts`、`contexts`、`compiled_programs` 与实体引用约束。保存产物前校验受控路径和大小，写后读取验证 hash；对象 ID 冲突时比较 payload，一致是幂等，不同是错误。所有评测与比较只追加；外键开启；内存连接生命周期正确。

**验收断言：** 同 ID 不同 payload 失败；已有报告不能覆盖；损坏产物读取失败；路径穿越和 symlink 逃逸失败；重启后的证据完整；`:memory:` 在持久连接内可重复读取。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_storage -v
```

**出口：** 无第二套独立晋级库；仓库安装态也能找到 SQL 资源。提交 `feat: add append-only evidence storage and verified artifacts`。

### T04 — 预测矩阵与可信评分

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T02—T03。

**文件：** 新增 `kernel/predictions.py`、`science/views.py`、`science/metrics.py`、`services/experiments.py`、`tests/refactor/test_prediction_contract.py`、`test_metrics.py`。

**输出接口：** [预测矩阵实现](../src/ecologyrsi_kernel/kernel/predictions.py) 的矩阵校验；`TrustedEvaluator.evaluate(execution_result, context, truth_view)` 返回宿主创建的 `EvaluationRecord`。

**实现：** 后端只提交预测/轨迹/计费/运行回执；宿主核验产物并计算分数。迁移现有已验证指标作为一个带版本的政策，不静默修改尺度/权重。按 expected case 精确对齐，固定分母，物理约束单独保存。

**验收断言：** NaN、Inf、bool、数字字符串、空/缺/重/多行全部拒绝；同样本数错身份拒绝；伪造 `scientific_score` 和 `stable` 字段拒绝；可信真值不出现在后端请求与公共日志。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_prediction_contract tests.refactor.test_metrics -v
PYTHONPATH=src python -m unittest tests.test_evaluation tests.test_greenhouse_prediction -v
```

**出口：** 评分结果来自预测+真值重算，不依赖候选声明。提交 `feat: score verified predictions in the trusted evaluator`。

### T05 — 配对统计与两个层级的门禁

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T04。

**文件：** 新增 `science/statistics.py`、`kernel/evidence.py`、`kernel/gates.py`、`tests/refactor/test_statistical_gate.py`。

**输出接口：** `build_paired_evidence(candidate_eval, baseline_eval, context, statistical_policy)`；`SearchGate.assess(evidence, gate_policy)`；`CertificationGate.assess(evidence, confirmation_plan)`。

**实现：** 逐单元配对，固定聚合目标；按合同做时间区块/episode 统计；保留独立 origin 与区块计数；数据不足返回无结论。搜索门禁不宣称正式显著性；确认门禁在没有有效独立计划时一律拒绝，T13 再接线。

**验收断言：** 人工构造正差、零差、负差样本方向正确；重复 origin 不增加独立块；错种子/错 context 不比较；没有足够数据不会因“平均更高”自动确认；同 seed 可重放；多目标容忍度变更产生新 policy ID。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_statistical_gate -v
PYTHONPATH=src python -m unittest tests.test_generation_comparison tests.test_promotion -v
```

**出口：** `stable` 不作为外部输入；统计记录可由原产物重算。提交 `feat: add paired evidence and versioned scientific gates`。

### T06 — 原子搜索晋级与受限回滚

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T03—T05。

**文件：** 新增 `storage/promotion.py`、`services/promotions.py`、`tests/refactor/test_atomic_promotion.py`。

**输出接口：** [原子账本实现](../src/ecologyrsi_kernel/storage/ledger.py) 的 `Ledger.promote()` 的原子比较更新语义；第 8.4 节规定的历史回滚接口。

**实现：** 先服务权限、后事务；幂等回执先于可变状态检查；绑定双边证据、run、context、policy、来源、seq 与有效期；更新冠军、回执、事件同事务。生产服务不允许直接插入一个任意 `decision=select` 的比较对象。

**关键测试：** 迁移[自动化测试目录](../tests/) `LedgerTests`；另加两个独立 SQLite 连接的并发竞争、ABA、跨 run 引用、回滚到非历史候选、两条命令竞争同一冠军和认证权限拒绝。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_atomic_promotion -v
```

**出口：** 最多一个当前搜索冠军；中断后原状态或完整新状态二选一；演示证据不可晋级；搜索晋级不能创建 ReleaseManifest。提交 `fix: make search promotion atomic and revision-bound`。

### T07 — Batch 后端纵向切片

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T02—T06。

**文件：** 新增 `execution/backends.py`、`adapters/batch_backend.py`、`adapters/synthetic_backend.py`、`science/prediction_adapter.py`、`tests/refactor/test_batch_backend.py`、`test_end_to_end.py`。

**输出接口：** 第 9.1 节后端合同；`fit/predict` 科学接口；最小串行实验服务。

**实现：** 包装现有真实数值预测器，不复写算法；冻结训练产物；批量生成矩阵；严格记录数据模式。先用串行运行闭合流程，T09 再加入并发租约恢复。保存 attempt、请求和结果，不能只在内存持有状态。

**验收断言：** 相同输入/种子确定性一致；改变生效参数在控制样例中改变产物；旧数值预测误差不回归；两个搜索轮次不发网络请求；合成数据只更新演示冠军，不生成真实搜索资格。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_batch_backend tests.refactor.test_end_to_end -v
```

**出口：** 可离线演示完整可信内核；未声明已完成真实 DSH 或正式确认。提交 `feat: run scientific experiments through the batch backend`。

### T08 — 完成真实 DSH 适配合同

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T07。

**文件：** 新增 `adapters/dsh_backend.py`、`docs/refactor/DSH_MAPPING.md`、`tests/refactor/test_dsh_backend_contract.py`；必要时修改现有 `integrations/dsh_ecology_plugin/lib/` 对应接口。

**实现：** 按 ARCHITECTURE.md 第 9.3 节及 DSH_INTEGRATION.md 读取本地接口，逐字段映射 submit/lookup/poll/cancel；验证真实支持能力；模型、角色、技能、工具和请求身份写入运行回执。将旧编译器的 DSH 特定绑定迁入适配器，不复制科学门禁。

**验收断言：** fake transport 测真实协议映射而非科学加分；错 request/attempt/context 的响应被拒绝；禁用工具无法调用；新技能内容 hash 进入真实运行；取消和超时不会产生成功评分。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_dsh_backend_contract -v
PYTHONPATH=src python -m unittest tests.test_dsh_native_runtime tests.test_dsh_reconciliation -v
make test-integration
```

**出口：** 离线合同验证完整；真实环境测试单列，缺凭据标记 `not_run`。提交 `feat: implement the DSH execution backend`。

### T09 — 持久化调度、预算与崩溃对账

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T07；DSH 对账部分依赖 T08。

**文件：** 新增 `execution/jobs.py`、`scheduler.py`、`budget.py`、`reconciliation.py`、`storage/migrations/002_jobs.sql`、`tests/refactor/test_jobs.py`、`test_budget.py`。

**输出接口：** `transition_job()`、`claim_job()`、`accept_result()`、`reserve_budget()`、`settle_budget()`、`reconcile_attempt()`。

**实现：** 第 10 节状态机；短事务认领；lease_epoch fencing；outbox；提交结果不明进入 UNKNOWN；按真实能力 lookup；不确定成本冻结预算。队列大小和并发严格有界，不在每个 origin 启动无限线程。

**验收断言：** 发送成功后崩溃能对账；旧 lease 回包无法写入；双 worker 不双重接受结果；重复结算不重复扣费；暂停/取消/恢复语义清楚；基础设施失败不扣候选科学分。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_jobs tests.refactor.test_budget -v
PYTHONPATH=src python -m unittest tests.test_dsh_cancel_race tests.test_sample_budget -v
```

**出口：** 注入断连、进程重启、晚返回与预算边界仍保持不变量。提交 `feat: make experiment jobs recoverable and budget-aware`。

### T10 — 搜索服务与互补档案

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T05—T09。

**文件：** 新增 `research/proposals.py`、`operators.py`、`search.py`、`repertoire.py`、`services/search.py`、`tests/refactor/test_search.py`。

**输出接口：** `propose_batch()`、`apply_operator()`、`admit_repertoire_entry()`、`plan_next_round()`；返回显式 proposal/candidate/experiment ID，不暴露 latest-record 快捷方法。

**实现：** 保留并迁移现有单轴/弱目标信息；加编译去重、分阶段同窗预算；互补档案分上下文保存；固定随机基线；算子选择 seed 固定；多保真原始分不混排。

**验收断言：** 同一候选重复提出不多跑；相同 context 中可保留短时/长时互补候选；不同 context 不排名；空变异不消费科学预算；同 seed 搜索安排一致；最低预算能提前终止并正确记录。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_search -v
PYTHONPATH=src python -m unittest tests.test_strategy_router tests.test_candidate_direction_contract -v
```

**出口：** 每轮能回答“为什么试这个、实际改了什么、为什么保留/拒绝”。提交 `feat: add identifiable search and complementary candidates`。

### T11 — 有证据的实验记忆

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T10。

**文件：** 新增 `research/memory.py`、`storage/migrations/003_research.sql`、`tests/refactor/test_memory.py`。

**输出接口：** `record_experience()`、`retrieve_experiences(task_context, failure_signature, scope)`、`snapshot_memory()`。

**实现：** 第 12 节字段；引用真实比较；先权限过滤后排序；成功、失败、无结论均保留；记忆快照参与 search_context 身份；禁止确认数据回流。

**验收断言：** 不存在 comparison 引用失败；伪造支持结论不能覆盖宿主评测；确认集记录无法检索；相同任务/快照检索确定性一致；反例不会被全部过滤。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_memory -v
```

**出口：** 记忆能解释来源，不自动提升为系统指令。提交 `feat: build evidence-grounded experiment memory`。

### T12 — 技能与工作流的真实变异效果

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T08、T10—T11。

**文件：** 修改 `kernel/compiler.py`、`research/operators.py`、`adapters/dsh_backend.py`；新增 `tests/refactor/test_runtime_variations.py`。

**实现：** 结构化技能内容和受限 DAG 纳入编译身份；生成新技能先验证 schema、准入工具和引用证据；在未参与提炼的开发任务上测试有/无技能；效率收益与科学收益分开评估。

**验收断言：** 改技能文本但没有被加载则行为测试失败；工具禁用后调用失败且被记录；修复策略在可控故障中确实改变动作；相同科学质量但更低成本可通过效率门禁，不需要伪造精度提升。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_runtime_variations -v
```

**出口：** Algorithm/Feature/Skill/Workflow 都有“配置→执行→产物”的追踪路径。提交 `feat: validate effective runtime skill and workflow mutations`。

### T13 — 独立验证与发布边界

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T05—T06、T09；不依赖元策略。

**文件：** 新增 `storage/exposure.py`、`services/certification.py`、`storage/migrations/004_certification.sql`、`tests/refactor/test_certification.py`、`test_data_isolation.py`；迁移旧暴露账本。

**输出接口：** `reserve_confirmation_plan()`、`open_confirmation_view()`、`seal_confirmation()`、`issue_release_manifest()`；四者权限分离。

**实现：** 候选、基线、指标、比较家族、统计方案和数据视图预冻结；读取真值前记录暴露；失败也不回收已暴露的“独立性”；同 authority 跨 run 检查；确认结果默认不回流研究者。正式发布还要求明确治理凭证。

**验收断言：** 改 run/cohort/目标权重无法绕过同源数据暴露；没有计划不能读真值；读后异常仍算暴露；合成和历史弱证据无法确认；研究策略/DSH 工具无发布权限；发布 manifest 精确绑定最终产物。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_certification tests.refactor.test_data_isolation -v
PYTHONPATH=src python -m unittest tests.test_scientific_boundary tests.test_scientific_exposure_registry -v
```

**出口：** 搜索、确认、发布三者可分别审计，历史 exposure 迁移不会被清空。提交 `feat: enforce independent confirmation and release authority`。

### T14 — 研究策略的跨任务进化

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T10—T13。

**文件：** 新增 `research/metaevolution.py`、`tests/refactor/test_metaevolution.py`；扩展 `003_research.sql` 或增加有明确顺序的新迁移，不能修改已应用迁移。

**输出接口：** `freeze_meta_trial()`、`run_policy_trial()`、`compare_research_policies()`；全部引用第 13.3 节的 `MetaEvaluationContract`。

**实现：** 两策略共享初始资源和任务；任务内部先搜索后冻结模型；任务级配对结果与成本；development/confirmation 明确区分；内层错误和预算失败纳入结果。

**验收断言：** 元策略不能读取确认任务真值；初始记忆不同则不可直接比较；随机种子计划可重放；内层失败不会被从平均值分母删除；元策略不能提高总预算或降低门槛。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_metaevolution -v
```

**出口：** 能分开报告对象层改善与方法层改善，未运行真实跨任务实验时不声称已经验证 RSI。提交 `feat: evaluate research-policy evolution across tasks`。

### T15 — 旧数据只读导入与影子对比

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T03、T06、T13。

**文件：** 新增 `adapters/legacy_import.py`、`scripts/refactor/migrate.py`、`tests/refactor/test_migration.py`。

**输出接口：** `inspect_legacy()`、`build_import_plan()`、`import_copy()`、`verify_import()`；命令 `inspect/dry-run/copy/verify`。

**实现：** SQLite backup；目标新库；源 ID/payload/schema 保留；明确 legacy/synthetic/unverifiable；映射幂等；不能把曾被覆盖的数据补造回来；新旧比较器对同一历史产物只读计算，不做晋级。

**验收断言：** 源库和产物 hash 不变；迁移两次无重复；半途退出重跑安全；未验证记录无比较资格；旧运行仍可回放；不接受未经验证的新字段替历史凑齐证据。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_migration -v
```

**出口：** 有迁移报告、隔离清单、源→新身份映射和回退读旧库方案。提交 `feat: migrate legacy evidence without fabricating eligibility`。

### T16 — CLI、API、投影与分发切换

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T07—T13、T15；方法层 UI 可在 T14 后启用。

**文件：** 修改 `application/cli.py`、现有 `api/`、`presentation/`、`pyproject.toml`、`MANIFEST.in`、相关插件投影；新增 `configs/refactor-smoke.json`、`tests/refactor/test_api_v2.py`、`test_installed_package.py`。

**实现：** 第 17 节入口；每个 run 绑定引擎版本；旧 run 禁止切换；默认关闭网络、生成代码、确认和发布；根 README 保留安装方法并新增 v2 使用说明；SQL/schema/注册表资源进入构建包。

**验收断言：** 缺真实数据不能自动退到 demo；公共投影不泄密；幂等状态码一致；事件分页有界；临时目录安装 wheel 后 CLI 和 SQL 资源可用；旧 UI/插件加载不回归。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_api_v2 tests.refactor.test_installed_package -v
make compile
make verify
```

**出口：** 用户可从现有入口运行新内核，明确看到证据层级。提交 `feat: expose the new engine through existing product entrypoints`。

### T17 — 架构收口、故障验收和等预算对照

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T00—T16。

**文件：** 新增 `tests/refactor/test_architecture.py`、`scripts/refactor/verify.py`、`benchmark.py`、`docs/refactor/BENCHMARK.md`；移除已无调用的旧写入口。

**实现：** 导入依赖约束；旧写路径调用扫描；注入崩溃/并发故障；科学 baseline 和消融 manifest；记录实际成本、失败与数据隔离。旧代码只保留只读兼容，不保留隐蔽第二套新建/晋级路径。

```bash
PYTHONPATH=src python -m unittest discover -s tests/refactor -p 'test_*.py' -v
make test
make compile
make verify
make test-integration
```

**验收：** 第 18 节软件约束全部具备自动化测试；真实外部验收按授权运行，缺环境明确 not_run；基线结果如实输出，没有科学收益时不捏造达标结论。提交 `test: verify the new kernel and close legacy write paths`。

### T18 — 扩展科学程序空间与行动条件任务

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T17；这是增强项，不阻断可信内核上线。

**文件：** 扩展 `science/domains.py`、`kernel/compiler.py`、已注册科学能力目录；新增 `tests/refactor/test_scientific_dag.py`、`test_action_conditional_task.py`。

**实现：** 第 14 节类型化 pipeline；受审核的新预测器/特征/残差/不确定性算子；行动计划视图与未来真实动作分离；物理约束与模拟器版本冻结。先小型模拟任务，不连接真实执行设备。

**验收断言：** 错维度/错单位/有环/未登记节点拒绝；对未来实际动作的信息泄漏被检测；训练数据改变会产生新产物；在固定模拟场景中能测量行动响应和长期约束，而非只改接口名。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_scientific_dag tests.refactor.test_action_conditional_task -v
```

**出口：** 程序空间扩大可被真实实验测量；不声称模拟收益即真实控制收益。提交 `feat: expand typed scientific programs and action-conditioned tasks`。

### T19 — 可选隔离代码能力构建

**每个任务的必做步骤：**
- [ ] 读取本任务输入接口及受影响旧代码，确认当前工作区。
- [ ] 新增本任务列出的回归测试，运行并记录预期失败。
- [ ] 实现目标接口；优先迁移已验证的原生语义，不接入合成报分。
- [ ] 运行本任务目标测试、受影响旧测试和跨模块验收。
- [ ] 逐项核对出口条件；更新 PROGRESS，并按用户版本管理规范提交独立变更。

**依赖：** T17—T18；需要单独配置隔离执行条件，默认不启用。

**文件：** 新增 `adapters/capability_builder.py`、受控沙箱配置、`tests/refactor/test_capability_builder.py`。

**实现：** 第 15.3 节隔离构建；固定依赖集合；无密钥/网络/确认标签；受限 CPU/内存/进程数/输出大小；测试报告与源码绑定；审核后注册新能力并在新 epoch 使用。

**验收断言：** 隔离不可用不降级到宿主；读取宿主秘密/越界目录/启动无限进程/写标签均失败；超时清理整个执行树；未经审核的产物不能进入能力注册表。

```bash
PYTHONPATH=src python -m unittest tests.refactor.test_capability_builder -v
```

**出口：** 能安全地扩大可实验程序集合，但不能改可信评分与发布规则。没有隔离环境则保持关闭并完整记录，不用假 sandbox 交付。提交 `feat: add an opt-in isolated capability build pipeline`。

---
