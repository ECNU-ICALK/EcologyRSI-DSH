# Changelog

## ecogym 0.1.0 — 重构为领域无关的生态预测训练场

- 以新包 `ecogym` 取代 `ecologyrsi_dsh`、DSH 插件和旧网页（约 13 万行，保留在标签 `v0.10.33`）。分层为 kernel → data → task → protocol → solvers → evaluation → arena → agents → cli/service；上层只依赖下层，框架代码不出现任何领域名称，均由测试检查。
- 数据集规格和任务卡改为 TOML，放在领域包中；通用 `csv_tables` 加载器读取 AGC 2018 黄瓜和 2019 番茄，全部列与旧加载器逐位一致。
- 金标准对照（由 0.10.33 生成）：分区边界、逐格基线、计分起点（449／154／683）完全一致；专家模型 +0.1292／+0.1769、recipe 种子 +0.0878／+0.1179，逐格 RMSE 相差小于 1e-12。
- 竞技循环：提案 → 初筛 → 不重叠留出日上的配对比较 → 晋级，门槛沿用 0.10.33（逐格非劣改为对照在任方法）。宿主直接计算全部预测，取消逐样本 LLM 链；一次 3 代扫描约 0.2 秒，任何运行都可按记录的修改逐事件重放。
- 封存评测：先登记曝光再打开分区，每个方法在 validation 与 final_test 上各一次，final_test 须在 validation 之后。
- LLM 研究智能体：OpenAI 兼容 HTTP 与 DSH 无头两种传输（DSH 自身工具全部禁用），研究员／提案者／评审三个角色，研究工具只读 `calibration_fit` 和当代初筛日。
- 网页工作台（`ecogym serve`）与静态 HTML 报告；新增领域指南 `docs/new-domain.md`。

EcologyRSI-DSH 0.10.33 及更早版本的变更记录见标签 `v0.10.33` 中的 `CHANGELOG.md`。
