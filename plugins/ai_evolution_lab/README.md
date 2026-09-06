# AI for AI 自进化实验台

这是一个独立于 DSH 的外部控制插件，用来验证“改进 AI 的 AI”是否真的让下一轮研究更可靠。它维护 Skill、Tool Policy、Workflow 和算法候选版本，通过既有 DSH API 运行同 cohort 实验并读取投影；DSH 的协议、runtime、权限和审计不需要修改。

## 它优化什么

- **Skill**：研究、规划、诊断、反思的结构化提示和输出合同；
- **Tool Policy**：已登记科学工具的选择、顺序、超时和失败回退；
- **Workflow**：调研—编译—训练—评测—晋级的编排参数；
- **Algorithm**：开发者已实现并登记的预测器及其有界参数。

任意模型生成的 Python、Shell、依赖安装或网络代码都不会直接执行。新能力必须先由宿主登记，实验台只允许对声明式能力做受限变异。

## 一次候选实验

```text
Genome → 编译/静态检查 → 同 cohort DSH 实验 → 评测报告
       → 实用差异与逐单元门禁 → promote 或 rollback
```

`search_winner` 只表示搜索过程中暂时最优；只有同 cohort、科学门禁、稳定性和治理条件全部满足时，才会标记为 `certification_eligible` 并更新 incumbent。负分候选可以被记录为研究证据，但不能因为“相对最不差”而自动晋级。

## 本地验证

```bash
PYTHONPATH=src python -m ecologyrsi_dsh.evolution_lab --demo --db /tmp/evolution-lab.sqlite
```

生成 Skill 以结构化提案进入候选池；新 Tool 必须是声明式能力或已审核的 Host adapter，任意 Python/Shell 代码不会自动进入 DSH。
