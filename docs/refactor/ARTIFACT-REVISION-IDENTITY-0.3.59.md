# 实际训练版本与提案来源的身份修复

本轮封存预测复算发现：局部修改产生 R1 后，训练产物参数与样本作用域已经来自 R1，但原 `ArtifactRecorded.identity_binding` 仍引用 `CandidateSpawned` 的 R0 编译身份。这会把原始提案来源误读为最终模型身份。修复不改变拟合公式、样本选择、评分、模型输出或历史数据库。

具有实际 `candidate_revision_id` 的新原生训练产物使用 `ecologyrsi-dsh.artifact-recorded/2`：

- `proposal_identity_binding` 完整保留初始提案的 genome、compiled behavior、phenotype、runtime 等来源摘要。
- `artifact_revision_binding` 单独封存实际 run/candidate/revision、revision/genome/behavior 摘要、预测器、参数、拟合参数、完整评测作用域以及产物摘要；绑定自身也有摘要。宿主在写入和重放时重算，不复用 R0 的编译摘要冒充 R1。
- 参数必须等于实际 revision 的 scientific program；存在的拟合子模型必须通过各自摘要校验。作用域必须来自已冻结的 holdout arm 或已记录的批次评分。冻结 holdout 先于产物入账，评分完成事件可以随后入账。

对应 `EvaluationRecorded/2` 必须引用同一产物封套、同一实际 revision 和完整作用域。后续评审不能改变 revision、作用域或原科学分数。新正式 validation/final_test 冻结实际产物版本；禁止将 v2 产物降级为旧 R0 正式封套。

历史 v1 事件保持可重放，不自动补写 v2 版本核验记录。历史正式留出若已冻结 R0 而实际产物为 R1，重新预留会明确返回 `legacy_formal_revision_conflict`，保留已有证据并拒绝重新开放。身份一致的旧 R0 正式记录可幂等返回，仍保留 v1；已冻结阶段不允许通过改分区或分析计划再次消耗留出数据。

API 的 artifacts 列表增加封套版本、核验状态、实际 revision 标签以及两类身份绑定。网页训练产物区分别显示“实际模型版本 R1”和“提案来源 R0”，并展示实际训练参数。历史数据明确显示缺少新版本封套，不宣称通过新核验。这些标记仅证明版本与证据对应关系，不代表模型科学质量通过。

验证命令：

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_artifact_revision_identity.py'
.venv/bin/python -m unittest discover -s tests -p 'test_sealed_prediction_shadow.py'
.venv/bin/python -m unittest discover -s tests -p 'test_judge_persistence.py'
node plugins/ecology_evolution/test/smoke.mjs
```

身份测试覆盖 R0 与 R1 的不同参数（R1 history=7、alpha=0.2）、有效版本重放、跨 revision/scope/artifact 篡改、评审越权改写、正式封套降级、历史 v1 重放、历史正式 token 幂等与冲突后的零写入。调度流程另由既有 trajectory/formal 测试覆盖；这些测试不调用付费模型。
