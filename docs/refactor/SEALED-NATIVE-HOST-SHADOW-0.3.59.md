# 0.3.59：已封存原生预测的同模型数值复算

本研究在已完成运行 `run:stable-20260907-b-kimi` 的已评分搜索数据上，直接复用原始 `ArtifactRecorded` 保存的完整拟合模型。**没有重新拟合、调用远程模型、写入进化账本或开启独立验证／最终测试。**

聚合报告为 [SEALED-NATIVE-HOST-SHADOW-0.3.59.json](SEALED-NATIVE-HOST-SHADOW-0.3.59.json)。文件不包含原始预测行、观测标签、逐原点特征值或模型隐式推理。

## 严格通过的结果

候选 `candidate:9edb7b36-cbeb-4c2a-849d-3564ecc8d27f`，原封存修订 `r0`，参数为 history=6、alpha=.3、残差倍率=.5。

| 检查 | 结果 |
| --- | --- |
| 完整原点向量 | 169 个，每个3目标×3时距 |
| 已评分预测单元 | 1,521 个 |
| 精确重现的真实原生工具回执 | 169 份 |
| 最大预测绝对差 | 0 |
| 原分数／复算分数 | −0.19336396542080195，两者相同 |
| 源模型与修订、scope、checkpoint绑定 | 通过 |
| 原始基线及归一化输入 | 通过 |
| 重拟合／新增远程调用／账本写入 | 0／0／0 |

本机该次运行的 Host 数值计算及输出组装约0.076秒，因果上下文构建约0.166秒。两项分别计时，数值计算时间不包含上下文构建；它们均不包含数据加载、工具回执摘要核验、分数核验或报告写入。这是一次本机诊断计时，不是可泛化的吞吐基准。

相同原点集合的历史原生链有169个会话，已报告2,990,439 tokens（非缓存输入780,781；输出181,690；缓存读取2,027,968），全部有完整用量记录。首个子会话启动到最后一个结构化结果接受的记录窗口为1,050.4秒；此窗口包含队列等待及其他运行时因素。**不能把该窗口除以 Host 纯计算时间宣称加速倍数，也不能把历史全部 tokens 当作已证明可节省的费用。**

## 如何保证不是换了模型后碰巧相等

脚本验证原始 artifact digest、九格模型各自的 fit digest、参数与冻结 candidate revision 的一致性，并核对 ArtifactRecorded 外层原生 identity binding。它进一步校验 EvaluationScope、样本 checkpoint 修订及 scope、已封存结果归档摘要、实际冻结 cohort 的数据集、episode、原点索引与时间、候选身份和九格完整集合。

后续 v2 事件采用独立的 `proposal_identity_binding` 与 `artifact_revision_binding`。前者必须与原 `CandidateSpawned` 的完整提案身份相同；后者交由生产代码 `validate_artifact_revision_binding` 按真实修订、科学参数、拟合模型、scope 和 artifact 摘要重新计算并逐字段核验。合法的 R1 可以保留 R0 提案来源，但不能借用 R0 的编译身份冒充最终模型身份。未知事件封套版本会明确拒绝。当前聚合 JSON 仍然只报告下面注明的历史 R0 实证，不把合成 v2 回归测试当作新原生运行证据。

每个特征仅按原点时刻及更早的可见历史重建。缺值前向填充按注册策略的实际小时范围处理，训练中位数和标准化统计直接来自原拟合产物。没有用新的系数或重新估计的预处理统计替代原产物。未来观测仅从已经封存的评分行读取，用于核对既有分数；不会从原始数据读取未来目标值。

完整九格工具输出再通过生产代码 `DshPredictionToolBinding.restore_recorded_result` 核验原始 request digest 和 output digest。未完全重现时脚本失败，不输出“等价”报告。

原生 Planner 的完整提示、共享上下文和 wave 输入未被归档。因此本研究**保留原始 wave digest，不能重新构造并校验完整 Planner 输入 wave**。它证明的是同源快照、同拟合产物的因果数值复算以及原生工具输出回执重现，不是完整 LLM 会话重放，也不是已经实施的生产绕过对照。

## 独立复核发现的历史不一致

另一最终候选 `candidate:e15839bf-e557-47f8-bfb2-4f029ccbd78e` 的最终 `batch:1` 修订参数、scope与拟合模型可复算，其169个工具回执和1,521个预测数值在初步检查中也重现。但它的 `ArtifactRecorded.identity_binding.genome_digest` 仍指向原候选R0；最终R1的genome digest不同。

因此，最终严格聚合报告只纳入上述原冠军。脚本默认检查全部artifact时会拒绝这个历史外层绑定不一致，不会因为数值相等就报告整条身份链通过。后续 v2 封套支持真实有效修订绑定；历史 v1 事件未被重写、补齐或降级放行。适配 v2 后重新执行了原冠军复算和旧 R1 拒绝检查，结论保持不变。

## 使用及验证

```bash
python scripts/shadow_replay_sealed_predictions.py \
  --database /path/to/evolution.sqlite3 \
  --run-id run:stable-20260907-b-kimi \
  --candidate-id candidate:9edb7b36-cbeb-4c2a-849d-3564ecc8d27f \
  --data-root /path/to/greenhouse \
  --output sealed-native-host-shadow.json
```

数据库使用 SQLite 只读 URI 和 `query_only`。若缺少完整拟合产物、修订身份、成功原生链或工具回执，脚本明确拒绝。本研究不生成新评分、晋级事件或独立科学验证结论。

新增回归覆盖：未来值访问陷阱、旧与基线对齐预测器的实际同模型数值复算、不同拟合状态、错误回执、改变后的封存预测、错绑revision、过期外层genome binding、v2 合法 R1 与原提案分离、伪造 v2 有效修订或提案绑定、未知封套版本、错误 episode／原点／候选、缺失用量以及累计快照不能重复相加。
