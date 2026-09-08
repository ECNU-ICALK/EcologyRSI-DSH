# EcologyRSI v1.0.0 逐模块迁移记录

> 历史记录：其中的独立实验内核和命令已在 0.4.0 移除，不代表当前架构或可用入口。当前实现见 [代码清理说明](CODE-CLEANUP-0.4.0.md)。

更新时间：2026-09-07。以下保留第 0、1 步的历史记录；后续实际接入见 [实施记录](IMPLEMENTATION-2026-09-06.md) 与 [整体复审](REVIEW-2026-09-07.md)。长期迁移尚未全部完成。

## 迁移边界

以现有 EcologyRSI-DSH 项目为主体，将用户提供的新项目思路和代码按模块接入。
用户明确要求“不是完全覆盖，请一步步进行替换”。不考虑旧接口兼容性，指的是
具体模块重构时可以调整合同；不能据此删除整个现有实现或替换为另一套产品。

保留现有 `ecologyrsi_dsh` 应用入口、HTTP API、DSH 集成、两个插件、数据目录、
历史数据库和原有未提交修改。每步先明确替换范围，建立数值或行为基线，迁入代码，
验证调用链后再接入下一层。压缩包内的设计与任务文档是参考材料，不作为执行授权。

输入：`EcologyRSI-Full-Repository-v1.0.0.zip`。
SHA-256：`8f8c46262fc0239fe778792b809d97ff23cc47e7d6b7d48c266f6dd1ae6ca440`。

## 第 0 步：恢复现有项目（已完成）

- 恢复原项目版本 0.3.55；475 个归档文件逐字节验证一致。
- 迁移前已有的 6 个已跟踪文件修改及全部未跟踪的重构文件均已恢复。
- 重新安装原项目的 editable 包，CLI 指向恢复后的本项目源码。
- 此前整体替换的代码已另行备份；其实验结果保留在 `runs/`，并从 Git 跟踪中排除。
- 原始压缩包解压内容保留于 `.runtime/migration-v1/upstream`；此前审查和修改过的
  完整新实现保留于 `.runtime/migration-v1/reviewed-successor`，均不作为默认程序入口。

备份目录位于本项目父目录的 `.ecologyrsi-backups/`：

- 原始工作区：`pre-v1-20260906T223714/workspace.tar.gz`。
- 整体替换版本：`superseded-wholesale-20260906T225812/workspace.tar.gz`。

完整恢复核验记录：`.runtime/migration-v1/incremental-restoration.json`。

## 第 1 步：冻结模型推理（已完成）

范围为原本已经存在的实验内核 `src/ecologyrsi_kernel/science/models.py`。
迁移来源是压缩包中 `src/ecologyrsi_dsh/science/models.py` 的 `predict_fitted`
和季节性预测公式，同时采用审查时形成的训练、推理共用调用路径。MIT 声明追加在
本项目 `NOTICE`，主项目许可证与版本保持原样。

- `BatchBackend.run` 负责训练并保存模型，再调用 `predict_fitted` 生成预测。
- `predict_fitted` 只需要冻结权重和预测输入；不接收训练数据或评测标签，拟合计数为 0。
- 保留原有 persistence、seasonal、ridge 三类模型。新增的 residual_ridge 待独立
  评估和模型注册完成后再迁入。
- 检查拟合参数维度、缺失的目标/时距拟合、非有限参数和非正标准化尺度，避免
  `zip` 截短参数后仍然产生看似正常的预测。
- 模型保留拟合上下文，预测批次使用推理上下文。模型来源核验和评测分区准入仍由
  调用方负责，不能凭调用此函数声称完成独立验证或允许发布。

原生温室评测器未被这一步替换：它的真实时间戳、缺失数据、多目标/多时距逻辑
比参考内核更完整，必须通过数据适配与影子评测后才能接入新模块。

### 验证结果

| 检查 | 结果 |
| --- | --- |
| 原项目 `make test-fast PYTHON=.venv/bin/python` | 81 项通过 |
| 冻结推理、科学计算、搜索与证据回归测试 | 34 项通过，含本步新增 6 项 |
| 迁移前后数值对照 | 三类模型、10,368 条预测逐值完全一致；模型文件内容和拟合计数一致 |
| 原前端 `node plugins/ecology_evolution/test/smoke.mjs` | 通过 |
| 实验内核 4 次评测及审计 | 22 个事件、27 个制品检查，`audit=consistent` |
| 原 DSH 前端、同源 health 与 runs | HTTP 200，能读取本次原项目演示运行 |

数值对照文件为 `.runtime/migration-v1/incremental-model-parity.json`；相关日志在
同目录的 `restored-test-fast.log`、`incremental-model-tests.log`、
`restored-ui-smoke.log`。实验内核演示证据在
`.runtime/incremental-preview/kernel-stage1/`。

### 已存在的未完成接入

全量 `tests/refactor` 在本步前为 89 项、4 项未通过；本步后为 95 项、仍是相同的
4 项未通过。不能将本步的局部测试通过表述为完整重构测试通过。

1. `test_research_context_excludes_confirmation_experiences`：`MethodCard` 可选空字段
   被文本校验拒绝，尚未走到测试要验证的分区边界。
2. `test_hypothesis_requires_evidence_and_applicability`：必填引用列表接受空元组。
3. `test_run_and_verify_evidence_use_explicit_output`：规划中的 `v2` 子命令尚未接入主 CLI。
4. `test_validate_config_returns_frozen_summary`：同上。

这些均可在迁移前日志 `restored-refactor-tests.log` 中复现。下一步需先统一合同，
明确配置中的预算如何实际执行，再接通实验入口；不能仅为让 CLI 测试通过就暴露
尚未完整实现预算控制的运行配置。

## 第1步完成时规划的后续替换顺序（历史快照）

下表的“待实施”记录当时状态。后续已有部分接入，当前范围以 [0.3.59 整体交付说明](FINAL-DELIVERY-0.3.59.md) 和 [实施记录](IMPLEMENTATION-2026-09-06.md) 为准；不能将这些历史待办直接当作当前缺失能力。

| 步骤 | 接入范围 | 完成条件 | 当前状态 |
| --- | --- | --- | --- |
| 2 | 研究合同、配置和实验入口 | 修复上述合同问题；预算和输出路径实际生效；通过真实命令运行和审计 | 待实施 |
| 3 | 数据分区、评分与比较证据 | 对接原数据注册与温室评测；保持精确时间戳和缺失值语义；影子评测对照 | 待实施 |
| 4 | 实验调度、预算和事件账本 | 接入原任务生命周期；并发、恢复、幂等与审计验证通过；不重建历史数据库 | 待实施 |
| 5 | 自适应候选策略与研究记忆 | 在相同预算下比较；只使用允许的搜索证据；再接入 DSH 提案链 | 待实施 |
| 6 | DSH、API 与界面投影 | 在现有工作台逐页接入新结果；端到端验证后清理已被替代的具体实现 | 待实施 |

每一步的替换应保持可单独审查、可单独撤销。清理旧实现应发生在对应新调用链验证后，
不再按整个目录删除或整体覆盖。

## 第1步完成时的本地预览（历史快照）

原 DSH 工作台：<http://127.0.0.1:8848/plugins/ecology/evolution/>。
Python sidecar：`127.0.0.1:8777`，包与插件均为 0.3.55。

本次预览使用独立数据库 `.runtime/incremental-preview/evolution.sqlite3`，
包含运行 `run:incremental-preview`。历史数据库保留在原处。页面展示的是原项目
演示流程；第 1 步实验内核尚未绑定到前端执行链，两者的实验记录不能混为一谈。
演示使用合成数据，验证的是工程流程。
