# 0.3.58：基线诊断、时距控制与稳健搜索

本版在现有项目中增量实现最近进化数据审查的主要优化，保留历史账本、冻结协议和0.3.57交付物。**工程改进已实现，尚未证明模型在独立数据上稳定优于基线。**

## 可直接使用的改动

1. **统一诊断**：`ecologyrsi-dsh diagnose-training-fit` 先物理裁剪 `training_fit`，再做3折前推；每折只用前段拟合、按真实时间留出至少24小时，并在相同完整9格 origins 比较持续性、24小时季节性、fit-selected基线、历史参数及单轴变体。输出RMSE、MAE、偏差、skill和样本数，不输出原始标签。未来或非可见分区不能影响早期拟合和预测。
2. **基线对齐预测器**：新 `greenhouse-baseline-aligned-ridge@1` 配 `greenhouse_multihorizon_time_forward@3`。每格只在训练段选择基底，残差标签和推理使用相同基底。1／6／24小时各有独立修正倍率；倍率0精确退回该格已选基线。缺失季节历史时采用因果持续性回退并保留来源。正式参数为 `history_steps`、`ridge_alpha`、`residual_scale_1h/6h/24h`；新种子为 `6/.1/0/0/0`。
3. **稳健搜索门槛**：新方案默认冻结 `search_guard_policy=practical_delta_cell_noninferiority_paired_blocks@1`。局部更新须整体增益超过0.005、每格不退化、至少3个完整配对日块且配对日块bootstrap区间下界大于0。小幅正收益、证据不足或区间跨零保留冠军，记录具体待复核原因。全局选择继续使用更严格的8日块及认证门槛；exploration-only代不能更新全局冠军。日块存在时序相关性，局部搜索检查不等于独立验证。
4. **真实模型预检**：新方案在网页创建前调用本地 `/api/model-preflight`，检查researcher/search-plan及generation-judge/reflect的真实Skill、structured_output工具和结果证据。身份包含provider、model、role、preset、schema及内容/逻辑路由摘要，回执1小时有效。网页及API显式预检每次重新调用，避免复用同名配置热修改前的结果；直接脚本缓存不检测同名路由/preset热修改，修改后须使用`--force`重新预检。每角色默认1次、1024输出token、120秒、30000已报告token观察阈值；两角色总量相应有界。只有显式启用且最多2次的网络错误重试，确定性错误立即停止。预检不持全局操作锁，不阻塞暂停；同进程重复预检被拒绝，代理仅该路径延长至250秒。
5. **失败定位**：后续持久化失败可保留安全的transport、timeout、projection_lag等子码，不复制内部错误文本、凭据或模型隐式推理。

API的 `search_probation` 是从不可变配对决定生成的有限待复核记录，包含revision及原cohort摘要。它不会自动发起新实验；再次评测需要先冻结版本、登记未使用的配对cohort并检查剩余数据容量。不能用已经观察过的留出集重新声称独立确认。

## 实际数据结果

数据：AGC cucumber 2018，AiCU，训练段578行；3折共同 origins 为62、53、63。完成10组配置×3折=30次拟合，远程模型请求为0。以下均为9格等权skill，越大越好，0表示与本折冻结基线相当；不是RMSE百分比。

| 配置 | 折1 | 折2 | 折3 |
|---|---:|---:|---:|
| 历史种子参数（6/.1/.5） | -0.19610 | -0.04127 | -0.01164 |
| 最近冠军参数（6/.3/.5） | -0.19077 | -0.03691 | -0.01045 |
| 新基线对齐，三个时距均.25 | -0.07678 | +0.03915 | +0.00550 |
| 新基线对齐，仅6h改为0 | -0.01614 | +0.04313 | +0.02805 |
| 新基线对齐，三个时距均0 | 0 | 0 | 0 |

6h归零后，三个6h单元均精确返回基线，其他时距指标完全不变，证明该开关能够隔离当前有害的6h修正。第一折整体仍负，因此新默认种子从基线出发，不把探索性最佳配置当成已验证冠军。完整聚合结果见 [诊断JSON](TRAINING-FIT-DIAGNOSTIC-0.3.58.json)。

执行对照验证同一个拟合快照在宿主一次调用和逐origin调用时的预测、基线、掩码、评分及身份一致。**这不是native LLM与宿主执行的真实成本消融**，不能据此宣称已省去某个比例的线上token，也不能把本地诊断秒数与旧实验总时长计算加速倍数。原生sample.plan链仍保留；迁移它需要另行冻结执行协议并做同身份、同预算的真实对照。

## 使用

网页在“预测方案”选择“基线对齐岭回归”，会自动绑定v3评测器、稳健搜索门槛及工具/schema预检。旧方案仍使用其冻结身份；恢复旧运行不会补写新策略。预检只证明工具和schema传输能力，不证明sample.plan预测授权、研究语义质量或科学资格。

```bash
ecologyrsi-dsh diagnose-training-fit \
  --dataset agc_cucumber_2018 --episode agc_cucumber_2018:AiCU \
  --data-root /path/to/greenhouse \
  --output training-fit-diagnostic.json
```

可用 `--candidates` 传JSON列表，每项为 `{name, model_id, parameters}`，固定参数后做单因素比较。默认历史seed/champion是明确参数预设，不自动绑定或替换线上运行身份。

```bash
# request.json 使用创建运行的compact JSON，但此命令只做预检，不创建运行。
# 本地服务token通过ECOLOGYRSI_SERVICE_TOKEN环境变量提供，不写入JSON。
ecologyrsi-dsh model-preflight --request request.json --output model-preflight.json
```

下一阶段应预登记新的候选及未使用的时间段，在训练/搜索内单独检验1h和24h修正、alpha及history。只有锁定候选后才能开启独立验证/最终测试。本次不扩大远程进化预算、不恢复暂停自动化，也不改写旧holdout结果。
