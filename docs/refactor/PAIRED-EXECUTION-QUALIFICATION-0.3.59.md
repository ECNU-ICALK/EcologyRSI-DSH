# 配对比较的执行与评分资格审查

本轮发现并修复一个 P1：incumbent 的操作失败惩罚可以被局部配对门禁和同代 holdout 的 centered maxT 误认为科学预测收益。修复只影响带明确新资格标记的决定；评分、bootstrap、冻结 cohort 和旧决定保持原有含义。

## 生产函数反例

用纯合成数据调用真实 `assess_local_challenger` 与 `build_generation_comparison`，没有调用模型或读取隐藏标签。双方全部 9 格成功预测的 skill 都为 0.1，每天 24 个 origin，incumbent 每日 1 个 origin 失败，challenger 全成功。原目标是 `q × skill − (1 − q)`，因此两个分数分别为 0.0541667 和 0.1，差值 0.0458333，实际科学 skill 差值为 0。

原局部 3 日比较会晋升 challenger；原 holdout 8 日、3 日移动块、10,000 次共享 centered maxT 的稳定性下界也是 0.0458333，两个 finalist 都有认证资格。其输出甚至可以同时出现 `strict_agent_chain_pass=false` 和 `certification_eligible=true`。统计程序正确处理了原目标的不确定性，但原目标含可用性惩罚，不能单独证明科学收益。

只检查严格执行链仍不足。生产 `_sample_agent_chain_attestation` 对包含完整角色及 DSH 工具回执的 `status=failed` 合成记录返回 `complete=true`。这符合其“链完成”的原定义；链后拒绝或失败评分仍需独立识别。

## 新决定的资格

显式 `paired_execution_qualification=paired_strict_agent_chain@1` 现在绑定“执行链完整且科学评分完整”的比较资格：

- 双方 `strict_agent_chain_pass` 必须为布尔 `true`。
- attempted/succeeded origin 以及完整 origin 链数量必须等于冻结 `scope.origin_count`。失败 origin、失败 cell 和 fallback 数必须为整数 0；缺失、布尔伪整数、非法或不一致计数拒绝资格。显式 cell 总数如存在也必须等于 `origin_count × 9`。
- 校验日块证据摘要及其评估参数绑定；网格必须是三个注册目标与 1、6、24 小时的完整 9 格。每一日每一格必须 `eligible > 0` 且 `succeeded == eligible`，同一天 9 格计数相同，每格跨日总数等于冻结 origin 数。
- 双方日块身份、唯一日索引及每日日计数一致。没有共同成功掩码、删点、重试评分或重新拟合。

链失败使用 `paired_strict_agent_chain_failed`，完整链但评分不完整使用 `paired_scoring_evidence_incomplete`。新 holdout gate 公开聚合布尔值 `paired_scoring_evidence_complete`。无合格 challenger 时仍保留 incumbent，但保留状态与本轮科学资格分开报告。

资格检查不要求 incumbent 科学分数为正，因此合法的基线种子仍可接受真正改善；其余既有实际收益、分项非劣和不确定性门禁继续独立生效。这里的 holdout 是同代探索性选择，仍然保留 `exploratory_adaptive_data`/`selection_only`，并非最终独立验证通过的声明。

## 历史重放与防降级

新 local 的资格标记写入比较对象，并绑定 `comparison_contract_digest`；新 holdout 标记位于 gate 对象，绑定 `comparison_digest`。local 未标记时序列化不补字段，历史摘要不变。state 根据已记录标记选择规则，而非按当前进程版本或研究 policy 推断。

Director 在两个新决定入口对 guarded run 强制要求标记；已经存在且完整内容相同的决定可幂等返回。未知标记、非 guard 使用标记、新标记借用 legacy gate shape 均拒绝。旧无标记 guarded 决定仍按原规则确定性重放。可信 Host 写入口能够阻止降级；本修复不声称能将人工直接伪造的旧形状数据库事件与真实历史事件区分开。

修复前只读核对当前 live ledger：已完成 stable-Kimi 有 4 次局部比较和 1 次同代 holdout 比较，显式 guard 的各 run 尚无封存比较。仍对“历史 guard 已有无标记决定”建立回归，而不依赖当前数据恰好不存在该情况。

## 验证

`tests/test_paired_execution_qualification.py` 覆盖中断链、完整链后失败评分、缺失/不一致计数、9 格完整性、日索引/跨日总数/双方配对、正常科学收益、压缩持久化局部指标、旧标记缺省回放、新标记绑定、写入防降级及幂等恢复。相关既有 guard、局部比较、generation comparison、promotion 正例继续回归。

上述五组共 54 项测试全部通过，耗时 50.464 秒。

同时只读检查旧真实 complete run 的 3 个全成功 holdout 聚合证据：各有 169 个 origin，均通过完整评分自配对资格检查。历史局部记录没有保存日块证据，不用于证明新资格；其旧无标记回放规则保持不变。

精确数值、分支前后结果与聚合检查见同目录 `PAIRED-EXECUTION-QUALIFICATION-0.3.59.json`。没有包含原始预测行或隐藏标签。
