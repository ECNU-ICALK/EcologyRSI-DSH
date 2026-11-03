# AI for AI 与递归自我改进（RSI）文献调研：哪些做法对 ecogym 有用

2026-11-01 · 检索范围是 2023 年至 2026 年 10 月的论文、技术报告和开源项目，重点是 2025–2026 年的 Harness 自进化、AI 研究智能体、经验记忆、评测方法和轨迹训练。本文把文献里的共识和 ecogym 的现状逐条对照，说明这次接入了什么、为什么不接入其余的。接入的代码和离线检验见第 4、6 节，设计文档的对应改动见 [evolution.md](evolution.md) 第 3、17 节。

## 1. 结论

- **项目已经具备文献里最牢靠的几条共识。** 2026 年的 RSI 综述（1,250 篇论文）把评分信号按可信程度排序，发现自我改进的实际效果和信号强弱一致；一篇对 ICLR 2026 RSI 研讨会（110 篇论文）的会后综述把“在评测上变好、在任务上没变好”概括为最主要的未解决问题；PACE 指出“分数高了就留下”的验收规则会在噪声里累积大量假改进。ecogym 的确定性宿主评分、隐藏期、留出任务、封存时期、曝光登记，以及配对比较加任务间 t 区间、Bonferroni 和单任务否决，正是这些工作建议的做法。这部分不需要改。（2026-11-02 补充：门槛的方法不改，但 4 个训练任务时它几乎确认不了已经不错的版本附近的小增益，进化因此停住；改为默认 8 个训练任务、每个 1 个回合，成本相同。分析和对照的文献见 [evolution.md](evolution.md) 第 19 节。）
- **缺口集中在评审会话一侧：它看到什么、从哪里出发。** 这次接入四个模块，都只改变评审员的工作区、证据和工具，门槛的统计方法、裁判和内循环都不变：
  1. 每个回合的**会话记录** `transcript.md`（Meta-Harness 的消融：原始轨迹是 Harness 搜索起作用的关键，只给分数或摘要差得多），以及评审员提早结束时请它对照会话记录的一段续跑（实测里评审员不会主动打开会话记录）；
  2. **同版本回合对照** `contrasts.md`（Training-Free GRPO、ReasoningBank：从同一提示下好坏不同的几次尝试中提炼经验）；
  3. **存档起点**：以往比较过的版本（本次运行的，以及训练任务都在本次之内的其他运行的；2026-11-02 前要求训练任务完全相同）的完整副本 `archive/` 和工具 `reset_candidate`，候选可以从没被采纳的版本出发；检查时就指出与比较过的版本重复的候选，两个候选改动大部分相同时给出提醒（Darwin Gödel Machine、Huxley-Gödel Machine、GEPA、AIRA₂、DarwinX、ShinkaEvolve）；实测里评审员第一代就从另一次运行的候选出发；
  4. **跨运行的评审经验** `prior_runs.md`：训练任务都在本次之内的以往运行里各候选的结果和评审笔记（MetaShare、Group-Evolving Agents）。
  另外按技能泛化和上下文工程的研究改写了评审员说明里写经验和技能的规则。
- **评估过但没有接入的**：回合层面的提前停止（离线重放只能省下 64 个候选回合里的 4 个）、让评审员修改自己的工作方式（Hyperagents）、改内循环的起点 Harness、多目标门槛、训练数据的 GRPO 分组导出等，理由见第 5 节。

## 2. 方法

按七个主题检索：Harness 与智能体的自进化、进化式程序搜索、AI 研究与机器学习工程智能体、经验与记忆、评测统计与验收、技能库、从轨迹训练。一条做法算作**共识**，需要至少两组独立的工作得到同方向的结论，或者有一项消融实验直接支持它。一条做法算作**相关**，需要它对应 [evolution.md](evolution.md) 第 11–16 节里实测到的问题，并且能在不改 DSH 内核、不改确定性评分的前提下接入。只满足其一的，列入第 5 节。

## 3. 按主题对照

### 3.1 评分信号与验收规则

| 文献的结论 | 来源 | ecogym |
|---|---|---|
| 自我改进的效果和评分信号的强弱一致：形式验证器最强，执行结果其次，LLM 评审和自我评估最弱；自我确认的循环和多样性坍缩来自违反这一点 | RSI 综述 [2607.07663] | 宿主用确定性代码在隐藏期上逐时刻评分，不交给 LLM。已具备 |
| 自我改进系统会学会针对评测本身优化，这是主要失败方式；缓解办法是盲测、留出、只评一次 | ICLR 2026 RSI 研讨会 [rsi-ws] 及其会后综述 [rsi-ws-review]；DGM 中智能体为得分去掉幻觉检测标记的例子 [2505.22954] | 评审员看不到留出任务和封存时期；封存分区每个版本只评一次并先登记曝光。已具备 |
| 验收规则比提议者更关键：在含噪声的开发集上反复做“分数高就留下”，等于不受控的多重检验；没有真实增益时贪心规则每次运行留下 13–21 个假改进 | PACE [2606.08106] | 配对比较、任务间 t 区间、Bonferroni、单任务否决，在任版本只用它在任期间的回合。已具备 |
| 把用于搜索的评价和用于最终选择的评价分开、固定、对智能体隐藏；先前报告的“过拟合”多半是评价噪声，固定评价后长时间搜索不再退化 | AIRA₂ [2603.26499]；AIRA-dojo [2507.02554] | 内循环用本地检验，比较用隐藏期，最终用留出和封存。已具备。第 16 节“本地小幅提升在隐藏期不成立”与 AIRA₂ 的噪声解释一致 |

### 3.2 噪声与统计

| 文献的结论 | 来源 | ecogym |
|---|---|---|
| 评测是实验：报告标准误，比较时用配对差，相关的题目要聚类，事先做检验力分析 | Miller [2411.00640] | 配对、按天块自助、同数据集任务共用天的抽样；第 4、16 节有检验力计算。已具备 |
| 同一配置重复运行的差别常常大于配置之间的差别；单次运行的结论不可靠，区分几个点的差别需要几十次以上的运行 | [2602.07150]；[2609.33812]；Fragility [2608.18066]（加上自我改进循环后 71% 的情况方差变大） | 每版本每任务 2 个回合，在任版本每代重跑。已具备。第 16 节的数据说明模板变强后回合间标准差从 0.032 降到 0.005，这时加任务比加回合更有用，和文献一致 |
| 比较两种方法在多个数据集上的表现，用符号秩检验而不是配对 t 检验：t 检验受个别数据集上的极端值影响，样本少时还要求差别近似正态 | Demšar 2006 [demsar06]；Benavoli 等 2017 [benavoli17] | 2026-11-02 补充：`greenhouse.wide.climate@2` 改用符号秩检验（evolution.md 第 4、19 节）。起因是验证运行里一个在 8 个任务上都更好的候选，因为一个任务上的大增益没过 t 区间。4 个任务时这个检验永远通不过，`greenhouse.climate@1` 仍用 t 区间 |

### 3.3 搜索结构：只沿在任版本爬山，还是保留踏脚石

| 文献的结论 | 来源 |
|---|---|
| 维护一个不断增长的存档，从存档里选父本，而不是总在当前最好的版本上改；去掉开放式探索的消融版本明显更差 | Darwin Gödel Machine [2505.22954] |
| 一个版本自己的分数不能很好预测它的后代能否改进；按整个分支的后代表现选择扩展哪个节点 | Huxley-Gödel Machine [2510.21614] |
| 保留在至少一个训练样本上最好的候选（帕累托前沿），按覆盖面抽父本，并把两个候选的有效部分合并 | GEPA [2507.19457]（ICLR 2026） |
| 父本按排名抽样，保持种群；没有共享状态的并行尝试（Best-of-K）很快饱和 | AIRA₂ [2603.26499]；ShinkaEvolve [2509.19349] |
| 单线搜索依赖路径，局部的胜利常常让别的任务变差；只接受“扩展覆盖而不退化”的变体，存档保留不同的谱系以便重组 | DarwinX [2608.07545]；Mixture of Self-Improving Branches [2609.37834] |

**ecogym 的现状**：只有一个在任版本，每个候选都从在任版本复制；没被采纳的候选只以改动的文件名、提案节选和结果留在 `archive.md` 里，评审员拿不到它们的完整内容。实测里这类候选并不少：`evo:35ad338b60c3` 的候选 b 在 4 个训练任务上都好于在任版本（+0.008 至 +0.093），因为区间太宽没有采纳；`evo:9eca5239360b` 第 2 代的 a 也是 4/4 个任务更好（+0.021，区间跨 0）。下一代的评审员只能从在任版本重新开始。**这次接入了存档起点**（第 4 节）。门槛不变：候选仍然只和在任版本比较，所以统计上的保证不受影响。

### 3.4 提议者应该看到什么

| 文献的结论 | 来源 |
|---|---|
| 让提议者通过文件系统读到以往所有候选的源代码、分数和执行轨迹。消融：只给分数时候选准确率的中位数 34.6，分数加 LLM 摘要 34.9，给原始轨迹 50.0；摘要补不回缺失的信息，还可能压掉有诊断价值的细节 | Meta-Harness [2603.28052] |
| 把每个祖先改了什么、哪些行为变好或变差、哪些改动无效显式地交给编辑者，并针对父本最严重的错误生成子代；与 HGM 达到同样的最终成绩，评估预算少 2–4 倍 | HGM-Dual [2610.30042] |
| 只看最终分数看不出进展在哪里得到、在哪里失去；智能体更像工程优化者，成绩随运行变化大 | Beyond Final Scores [2608.13417] |

**ecogym 的现状**：评审员看到每个回合的成绩表、本地检验与提交的时间线、笔记、工具使用计数、最终回复和被评分的代码，但看不到智能体的会话本身：它读了什么、试了什么、在哪里卡住、为什么在本地把分推高。第 13、16 节里评审员误读模板 0.000、智能体在好模板上过拟合本地检验，都需要看过程才能诊断。**这次接入了会话记录**。

### 3.5 从同一提示的多次尝试中学习

| 文献的结论 | 来源 |
|---|---|
| 对同一题目的一组尝试，让模型比较好坏、提炼“语义上的组内优势”，写进经验库，不更新参数也能起到类似 GRPO 的作用 | Training-Free GRPO [2510.08191] |
| 从成功和失败的经验里都提炼可泛化的策略；比只存成功轨迹或原始轨迹的记忆效果好；并行或顺序的多次尝试提供对比信号 | ReasoningBank [2509.25140]（ICLR 2026） |
| 让智能体对比自己的次优动作和更好的动作做反思，不需要奖励也能学到东西 | Agent Learning via Early Experience [2510.08558] |

**ecogym 的现状**：每个版本在每个任务上至少 2 个回合，提示完全相同。账本里 62 组这样的回合，同组最好和最差相差的中位数 0.016，29 组超过 0.02，最大 0.164；例如 `evo:9eca5239360b` 里起点版本在番茄 Automatoes 上的 6 个回合从 +0.048 到 +0.174。这个差别既是比较里的噪声，也说明同一份说明下智能体有时做对、有时做错。**这次接入了同版本回合对照**。

### 3.6 经验与记忆

| 文献的结论 | 来源 |
|---|---|
| 反复整篇改写上下文会把细节压成空泛的原则（“简短偏差”“上下文坍缩”）；改为逐条增量更新，并去重 | ACE [2510.04618]（ICLR 2026） |
| 记录成功和失败的改动及其带符号的增益；每代开始前从其他谱系检索表现好的策略和可信的失败，避免重复死路；中位数达到或超过 DGM-Hyperagents 的最终成绩，到达自己最好版本所需的迭代少 1.7–2.9 倍 | MetaShare [2610.21887] |
| 以一组智能体为进化单位、组内共享经验，比树状的单独进化效果好 | Group-Evolving Agents [2602.04837]（NeurIPS 2026） |
| 允许改进机制本身被修改时，系统自己长出了持久记忆和表现追踪，并能跨运行积累 | Hyperagents [2603.19461] |

**ecogym 的现状**：内循环的经验在 `memory/lessons.md` 里，评审员的笔记在本次运行内会出现在以后各代的 `archive.md` 里。但每次新的进化运行，评审员都从零开始：账本里 8 次 `greenhouse.climate@1` 运行的评审笔记（例如“模板里写死的启用门槛会让智能体整回合退化为朴素基线”“单独修 bug 的候选很难过 +0.005 的门槛，应与有可测增益的方向一起提交”）互相看不到。**这次接入了跨运行的评审经验**，只取训练任务完全相同的运行（2026-11-02 起放宽为训练任务都在本次之内的运行）；训练任务里有本次留出任务的运行（例如 `greenhouse.wide.climate@1` 训练用到了 `greenhouse.climate@1` 的 4 个留出隔间）不取，否则留出验证会被污染。经验的写法按 ACE 的结论改进了评审员说明：逐条修改，不整篇重写。

### 3.7 技能库

| 文献的结论 | 来源 |
|---|---|
| 自进化出的技能在训练任务上的提升不一定带到新任务：21 个在训练任务上变好的技能里，5 个完全保留、13 个部分保留、3 个完全没有。带不过去的技能常常写死了随任务变化的细节（列名、输出文件），或把一次失败的修法写成适用于所有任务的规则 | Do Self-Evolving Skills Generalize? [2609.39148] |
| 技能库变大时性能下降（从少量有用技能到 202 个技能下降最多 21%），主要原因是选错技能，不是上下文变长 | Skill Shadowing [2605.24050] |
| 用有界的增删改编辑技能，只有严格提高留出成绩才接受。在 DeepSeek V4 Flash 和 Qwen3.8-27B 上的复现：SkillOpt 在验证集上的提升到留出集上变成退步（Qwen）；GEPA 在两个模型上的明显收益都只出现在 MATH-500，HotpotQA 上两种方法都没有提升 | SkillOpt [2605.23904]；复现研究 [2610.26020] |

**ecogym 的现状**：起点 3 个技能；宿主拒绝含训练任务专有名称的候选；技能的效果由整版 Harness 的门槛检验，不单独接受。已经避开了最大的风险（不经检验就接受技能改动）。**这次只改了评审员说明**：不写死列名、目标名、文件名；把单次失败的修法写成带条件的做法；技能少而精、写清何时使用。

### 3.8 候选的多样性

| 文献的结论 | 来源 |
|---|---|
| 在评估之前拒绝与已有程序太相似的新程序，节省评估 | ShinkaEvolve [2509.19349] |
| 想法更多样的研究智能体表现更好，受控实验也成立 | Ideation Diversity [2511.15593] |
| 检索经验时按嵌入相似度降低冗余策略的排名 | MetaShare [2610.21887] |

**ecogym 的现状**：宿主按行为摘要去掉与在任版本或以往候选完全相同的候选，但近似重复的不管。9 个有两个候选都被比较过的代里，有 3 代两个候选新增或改动的行有一半以上相同：`evo:d4d52da29adc` 76%（+0.027、+0.029），`evo:1f51cb6f659c` 96%（−0.020、+0.012），`evo:35ad338b60c3` 52%（+0.023、+0.032）。这样的两个候选分走同一代的 Bonferroni 预算，却只检验了一个方向；1f51 的两个几乎相同的候选相差 0.032，正好是噪声的尺度。**这次接入了雷同提醒**（不阻止运行）和检查时的重复识别。

### 3.9 评估成本

| 文献的结论 | 来源 |
|---|---|
| 分级评估：先在便宜的小测试上筛掉坏程序，只有有希望的才做完整评估 | AlphaEvolve [2506.13131]；DGM [2505.22954] |
| 按顺序检验，证据足够时提前停止，评估成本下降约 18% 而假改进率受控 | PACE [2606.08106] |
| 评估前先预测改动在各任务上的结果，评估后用实际结果改进预测 | IDM [2610.13991] |

**ecogym 的现状**：模板改动时宿主先在每个训练任务的本地检验期上运行模板（4 个任务约 1 秒），这就是分级评估的第一级，第 13 节的两个问题都是它发现的。已具备。第二级（回合之间提前停止）做了离线重放，结论是现在不值得做，见第 5 节。

### 3.10 内循环：研究智能体本身

| 文献的结论 | 来源 |
|---|---|
| 把机器学习工程当作解空间里的树搜索（起草、调试、改进） | AIDE [2502.13138] |
| 先检索有效的模型，再用消融找出影响最大的代码块做针对性改进，最后集成多个解 | MLE-STAR [2506.15692]（NeurIPS 2025） |
| 用多步 ReAct 智能体代替固定的单步算子；在 AIRS-Bench 上超过已发表最好成绩的任务里，有几个靠的是多个模型的集成（例如一个出行需求预报任务集成了 8 个神经网络预报模型） | AIRA₂ [2603.26499] |
| 时间序列方向的智能体基准：TimeSeriesGym（33 个挑战）、AIRS-Bench（20 个研究任务，含时间序列预报）、1GC-7RC（7 个任务，含时间序列预报，每对智能体与任务跑 5 次） | [2505.13291]；[2602.06855]；[2605.17046] |
| 用更强模型的轨迹做 SFT 能明显提高预报智能体（34.8% → 42.8%）；跨日期携带“信念笔记”降低成本，但不一定提高预报质量 | Forecast-Dojo [2609.28876] |

**ecogym 的现状**：每个回合是一个 DSH 智能体加本地检验，外循环负责在 Harness 版本之间搜索。第 12 节的实测和这些工作一致：同一版本 4 个回合的预报平均后高 +0.043，比两个候选带来的增益还大；建模模板让单个回合变好，同时去掉了回合之间的多样性。这些是**评审员可以尝试的候选方向**（消融式的逐块改进、在一个回合内集成思路不同的模型），不应由宿主直接写进起点 Harness：起点 Harness 改了，以前的运行就不能和以后的运行比较。

### 3.11 训练数据

| 文献的结论 | 来源 |
|---|---|
| 按奖励过滤后做 SFT（拒绝采样微调）、统计上可信的偏好对，是从智能体轨迹训练的常规起点；之后的 RL 用同一提示的一组轨迹算组内相对优势，并去掉组内没有差别的组 | AReaL-SEA [2601.22607]；Forecast-Dojo [2609.28876] |
| 没有奖励的轨迹也能用（隐式环境建模、自我反思） | Early Experience [2510.08558] |

**ecogym 的现状**：`sft.jsonl` 只收被接受的提交且隐藏期好于朴素基线的回合；`preferences.jsonl` 要求配对按天自助区间在 0 以上；`--all` 带 `passes_filters` 标记。已具备。

## 4. 这次接入的模块

全部在评审会话一侧（`evolution/evidence.py`、`evolution/evolver.py`、`evolution/evolver.md`），外加 `episodes/export.py` 的一个公共函数、`loop.py` 记录候选的起点、面板候选卡片显示起点。门槛、裁判、内循环、导出格式不变。

| 模块 | 依据 | 对应的实测问题 | 实现 |
|---|---|---|---|
| 会话记录 `evidence/episodes/<回合>/transcript.md`，评审员提早结束时的一段续跑 | Meta-Harness 的消融；HGM-Dual；Beyond Final Scores | 评审员只看得到结果，看不到过程（第 13、16 节）；实测里有了会话记录也不打开（evolution.md 第 18 节） | 从 `llm_calls.jsonl.gz` 重建每个线程（主会话、续跑、子智能体），略去系统提示和 DSH 注入的上下文，推理、回复、工具参数和结果过长时保留首尾，路径换成 `/workspace`、令牌脱敏（与训练数据导出同一个函数 `sanitize`）。每个回合至多 20 万字。内循环智能体看不到隐藏期，会话里没有隐藏期的信息。评审员结束时剩余时间超过一半的，宿主开一段续跑，请它对照差距最大的一组的两份会话记录 |
| 同版本回合对照 `evidence/contrasts.md` | Training-Free GRPO；ReasoningBank；Early Experience | 同版本同任务的回合相差最多 0.164，是比较的主要噪声 | 每组（版本，任务）里隐藏期最好和最差的回合，相差至少 0.02 的列出：评分程序、本地分、检验和提交次数、用时、token，两个评分程序的 diff（差 → 好），两份会话记录的位置；按差距排序，至多 24 组 |
| 存档起点 `archive/` 与 `reset_candidate` | DGM；HGM；GEPA；AIRA₂；DarwinX | 好于在任但没通过门槛的候选不能被下一代或下一次运行继续（35ad b、9eca g2a） | 本次运行里比较过的每个候选、训练任务完全相同的其他运行里比较过的候选（先放和本次在任版本比较过的，按增益排序）和起点版本的完整副本，至多 24 个，`archive/README.md` 列出各自当时的结果；`reset_candidate(name, base)` 把候选目录重置为在任版本或某个存档版本，起点记在 `candidate_proposed.base` 里，面板显示。改动和比较仍相对在任版本 |
| 重复与雷同检查 | ShinkaEvolve；Ideation Diversity；MetaShare；PACE | 3/9 代两个候选的改动一半以上相同；重复的候选以前要到会话结束后才被发现，其他运行比较过的版本还能原样再跑 | `check_candidate` 直接指出与在任版本、本次运行或训练任务相同的其他运行里比较过的版本行为相同的候选（状态 `duplicate`，不运行：原样再跑只是重复检验）；两个候选新增或改动的行有 50% 以上相同时给出提醒，不阻止运行 |
| 跨运行的评审经验 `evidence/prior_runs.md` | MetaShare；GEA；Hyperagents 的持久记忆 | 每次运行的评审员从零开始，重复以前的失败方向 | 训练任务完全相同的以往运行：每个候选的结果和摘要，评审员的全部笔记；新的在前，至多 4 万字。训练任务不同的运行一律不取 |
| 经验与技能的写法（评审员说明） | ACE；技能泛化研究；Skill Shadowing；SkillOpt | 自进化的技能和经验在新任务上失效的风险 | `memory/lessons.md` 逐条修改、不整篇重写，删条目时记下原因；不写死列名、目标名、文件名；单次失败的修法写成带条件的做法；技能少而精 |

## 5. 评估过但没有接入的

| 做法 | 来源 | 为什么现在不接入 |
|---|---|---|
| 回合之间提前停止：第一个重复跑完后，明显不行的候选不再跑第二个 | PACE；racing；DGM 分级评估 | 离线重放了 3 次 `repeats = 2` 的运行（8 个候选、64 个候选回合）。按 80% 区间的上限低于 0 停止只能省 4 个回合，按 90% 一个也省不了；而且第一个重复的估计已经很接近最终结果（d4d52 a +0.022 对 +0.027，9eca 第 2 代 b +0.073 对 +0.075），第二个重复主要是让单任务否决可靠。省得少，还要在一代中间加一道等待，不划算。候选数增加时可以重新评估 |
| PACE 式的随时有效检验（e-process） | PACE | 它需要积累很多配对的观测；ecogym 的检验单位是任务（4 个），一代里每个候选只有 4 个任务增益，攒不出足够的证据 |
| 让评审员修改自己的工作方式和证据的读法 | Hyperagents；IDM | 评审员的 `AGENTS.md`、证据和门槛由宿主所有，是防止针对评测优化的一道墙。持久记忆这一部分好处已经由评审笔记和 `prior_runs.md` 提供 |
| 让评审员先预测各任务的增益，再与实际对照 | IDM [2610.13991] | 只有一篇新工作；技能泛化研究发现 LLM 预测单个改动的效果很差。提案里已经有“预期”一节，`archive.md` 同时列出提案和实际结果，评审员可以自己对照 |
| 多目标门槛（成绩、token、时间） | Meta-Harness；[2609.30967] | 门槛衡量的是同一模型用这版 Harness 能做到什么，预算已经由回合的时限和 token 上限固定 |
| 改起点 Harness（加入消融式改进、集成等内循环做法） | MLE-STAR；AIRA₂ | 起点 Harness 是所有运行的共同基线，改它会让以前的比较失效；这些做法留给评审员作为候选方向 |
| 交付“几个回合预报的平均”并做封存验证 | 预报组合的经验；AIRA₂ 的集成 | 对一个新任务的最终预报有价值（第 12 节 +0.043），但它属于产品交付而不是 Harness 的自我改进，而且要扩展封存曝光的规则（每个版本每个分区只评一次）。值得单独做 |
| 导出按提示分组、带组内相对优势的回合，供 GRPO 类训练 | AReaL-SEA 等 | 现在用的是不能微调的 API 模型；`episodes.jsonl` 和 `sft.jsonl` 的 `meta` 已经有分组需要的信息（Harness、任务、成绩），真正开始训练时再按训练框架的格式导出 |
| 嵌入相似度去重 | ShinkaEvolve；MetaShare | 新增行的重合比例已经能识别实测里的三次雷同，不需要额外的模型和依赖 |

## 6. 验证

- `make test` 的 97 个测试和 `make lint` 通过，`node --check client.js` 通过。按项目约定没有新增单元测试。
- 离线检验，只读账本的副本（`.runtime/rsi-survey-20261101/`，不写真实账本）：
  - 为 `evo:9eca5239360b`（56 个回合）和 `evo:35ad338b60c3`（32 个回合）生成下一代评审员的完整工作区：证据 1.4–2.9 秒写完；会话记录每个回合的中位数 10.7 万至 12.8 万字节、最大 16.2 万字节，一次运行的全部会话记录 3.4 至 6.9 MB；`contrasts.md`、`prior_runs.md`、`archive/README.md` 的内容逐项看过。
  - 按评审员的用法驱动工具：把候选重置为存档版本后不改动，检查直接给出 `duplicate`（“与第 1 代的候选 a（g1a）内容相同”）；改一行后通过，模板在 4 个训练任务上实际运行；第二个候选从同一版本出发、只多一行不同时，两个候选都得到“96% 相同”的提醒；只改 `harness.toml` 摘要的候选被识别为与在任版本相同。从 9eca 的 g2a 出发时，现有的模板检查提醒它的模板在 4 个任务上都低于在任版本（g2a 的模板仍是朴素基线），说明从存档出发时原有的检查照常起作用。
  - `prior_runs.md` 的筛选：只有 2 个训练任务的冒烟运行看不到任何 4 任务运行，4 任务运行互相可见。
- 真实运行（0.3.0rc1，`evo:01303fc4864c`，细节见 [evolution.md](evolution.md) 第 18 节）：从 11-01 采纳的版本出发，2 代 × 2 个候选，56 个回合、5,500 万 token、66 分钟。
  - **存档起点和跨运行经验被用上了。** 第 1 代的评审员读了 `contrasts.md`、`prior_runs.md` 和存档里几个版本相对在任版本的 diff，两个候选都用 `reset_candidate` 从另一次运行里四个任务都更好、但没过门槛的候选出发；它们在四个任务上都好于在任版本（+0.014 [−0.010, +0.037]、+0.016 [−0.007, +0.039]）。第 2 代接着本次的 `g1b` 做（+0.017 [−0.020, +0.053]）。另一次运行的评审笔记让评审员找到了在任模板在一个任务上本地 −0.957 的原因。
  - **没有采纳，原因是检验力。** 4 个训练任务、两个候选时 t 区间的半宽约 0.023，确认不了 +0.015 上下的增益；同样的增益和任务间差异在 8 个训练任务上约为 [+0.005, +0.027]。在当前最好的版本附近继续改进，要用 `greenhouse.wide.climate@1`。
  - **会话记录一次都没被打开，已修复。** 两次评审都只用了 15 分钟里的 4.5 分钟。加上评审员提早结束时的一段续跑后，在同一份证据上跑的评审会话（快照账本、临时 `ECOGYM_HOME`，不写真实账本）读了差距最大的一组的两份会话记录，发现差回合照着“任一滚动窗口转负就收缩”的说明误收缩了有效的格（隐藏期低 0.06 至 0.09），并据此改了候选的说明。
  - **雷同提醒的口径已修正。** 两个候选从同一个存档版本出发时，提醒现在写明共同的起点，并给出它们在起点之上各自的改动有多少相同。

## 7. 参考文献

评分与验收
- [2607.07663] Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops. https://arxiv.org/abs/2607.07663
- [rsi-ws] ICLR 2026 Workshop on AI with Recursive Self-Improvement. https://recursive-workshop.github.io/
- [rsi-ws-review] The External Anchor Principle: How to Build Self-Improving Agents That Can't Game Their Evals（研讨会的会后综述，博客）. https://eugenevyborov.substack.com/p/the-external-anchor-principle-how
- [2606.08106] PACE: Anytime-Valid Acceptance Tests for Self-Evolving Agents. https://arxiv.org/abs/2606.08106
- [2603.26499] AIRA₂: Overcoming Bottlenecks in AI Research Agents. https://arxiv.org/abs/2603.26499
- [2507.02554] AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench（AIRA-dojo）. https://arxiv.org/abs/2507.02554

噪声与统计
- [2411.00640] Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. https://arxiv.org/abs/2411.00640
- [demsar06] J. Demšar. Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR 7 (2006) 1–30. https://jmlr.org/papers/v7/demsar06a.html
- [benavoli17] A. Benavoli, G. Corani, J. Demšar, M. Zaffalon. Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis. JMLR 18 (2017). https://jmlr.org/papers/v18/16-305.html
- [2602.07150] On Randomness in Agentic Evals. https://arxiv.org/abs/2602.07150
- [2608.18066] On the Fragility of Self-Improving Agents: Variance, Task Order, and Underspecification. https://arxiv.org/abs/2608.18066
- [2609.33812] Benchmarking AI Coding Agents on Open-Weight Models. https://arxiv.org/abs/2609.33812

Harness 与智能体的自进化
- [2505.22954] Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents. https://arxiv.org/abs/2505.22954
- [2510.21614] Huxley-Gödel Machine. https://arxiv.org/abs/2510.21614
- [2610.30042] HGM-Dual: Lineage-Aware Guided Agent Harness Optimization for Long-Horizon Planning. https://arxiv.org/abs/2610.30042
- [2603.28052] Meta-Harness: End-to-End Optimization of Model Harnesses. https://arxiv.org/abs/2603.28052
- [2608.07545] DarwinX: Evolving Agent Harnesses Through Natural Selection. https://arxiv.org/abs/2608.07545
- [2609.37834] Mixture of Self-Improving Branches for Agent Harness Optimization. https://arxiv.org/abs/2609.37834
- [2603.19461] Hyperagents. https://arxiv.org/abs/2603.19461
- [2610.13991] Predict Before You Modify: Learning Improvement Dynamics for Self-Evolving Agents. https://arxiv.org/abs/2610.13991
- [2610.21887] MetaShare: Failure-Aware Collective Self-Improvement for Open-Ended Agents. https://arxiv.org/abs/2610.21887
- [2602.04837] Group-Evolving Agents: Open-Ended Self-Improvement via Experience Sharing. https://arxiv.org/abs/2602.04837
- [2507.19457] GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. https://arxiv.org/abs/2507.19457
- [2609.30967] Multi-Objective Optimization of LLM Harnesses over Accuracy, Safety, and Tokens. https://arxiv.org/abs/2609.30967

进化式程序搜索
- [2506.13131] AlphaEvolve: A coding agent for scientific and algorithmic discovery. https://arxiv.org/abs/2506.13131
- [2509.19349] ShinkaEvolve: Towards Open-Ended and Sample-Efficient Program Evolution. https://arxiv.org/abs/2509.19349

经验、记忆与技能
- [2510.04618] Agentic Context Engineering（ACE）. https://arxiv.org/abs/2510.04618
- [2509.25140] ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory. https://arxiv.org/abs/2509.25140
- [2510.08191] Training-Free Group Relative Policy Optimization. https://arxiv.org/abs/2510.08191
- [2609.39148] Do Self-Evolving Skills Generalize to Held-Out Tasks? https://arxiv.org/abs/2609.39148
- [2605.24050] More Skills, Worse Agents? Skill Shadowing Degrades Performance When Expanding Skill Libraries. https://arxiv.org/abs/2605.24050
- [2605.23904] SkillOpt: Executive Strategy for Self-Evolving Agent Skills. https://arxiv.org/abs/2605.23904
- [2610.26020] Understanding When Reflective Prompt and Skill Optimization Helps on Locally Served Reasoning Models. https://arxiv.org/abs/2610.26020

AI 研究与机器学习工程智能体
- [2502.13138] AIDE: AI-Driven Exploration in the Space of Code. https://arxiv.org/abs/2502.13138
- [2506.15692] MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement. https://arxiv.org/abs/2506.15692
- [2511.15593] What Does It Take to Be a Good AI Research Agent? Studying the Role of Ideation Diversity. https://arxiv.org/abs/2511.15593
- [2608.13417] Beyond Final Scores: A Systematic Evaluation of Agents for Long-Horizon AI Research and Development. https://arxiv.org/abs/2608.13417
- [2505.13291] TimeSeriesGym: A Scalable Benchmark for (Time Series) Machine Learning Engineering Agents. https://arxiv.org/abs/2505.13291
- [2602.06855] AIRS-Bench: a Suite of Tasks for Frontier AI Research Science Agents. https://arxiv.org/abs/2602.06855
- [2605.17046] 1GC-7RC: One Graphic Card – Seven Research Challenges. https://arxiv.org/abs/2605.17046
- [2608.23058] LLM-based Agents for Forecasting and Prediction: Methods, Training, Evaluation, and Applications. https://arxiv.org/abs/2608.23058

从轨迹训练
- [2609.28876] Forecast-Dojo: Replayable Environments for Benchmarking and Training LLM Forecasting Agents. https://arxiv.org/abs/2609.28876
- [2601.22607] From Self-Evolving Synthetic Data to Verifiable-Reward RL（AReaL-SEA）. https://arxiv.org/abs/2601.22607
- [2510.08558] Agent Learning via Early Experience. https://arxiv.org/abs/2510.08558

文中对各文献的介绍均为转述，数字以原文为准（Content was rephrased for compliance with licensing restrictions）。
