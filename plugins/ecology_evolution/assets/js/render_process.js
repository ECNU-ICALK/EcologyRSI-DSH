"use strict";

  function trajectoryIncumbentScore(point) {
    var item = point || {};
    var value = item.incumbent_score != null ? item.incumbent_score : item.best_score;
    return value == null || value === "" ? NaN : Number(value);
  }

  function renderTrajectory() {
    // Candidate scores are raw observations. Only the server-recorded
    // incumbent fields may form the green promotion sequence.
    var node = $("#trajectory-chart");
    var run = displayedRun();
    var legend = node && node.parentElement && node.parentElement.querySelector(".chart-legend span:last-child");
    var points = run && Array.isArray(run.trajectory) ? run.trajectory.filter(function (point) { var value = point.score != null ? point.score : point.candidate_score; return !isBlank(value) && Number.isFinite(Number(value)); }).slice().sort(function (left, right) {
      return Number(left.generation || 0) - Number(right.generation || 0) || Number(left.slot_index || 0) - Number(right.slot_index || 0) || String(left.candidate_id || "").localeCompare(String(right.candidate_id || ""));
    }) : [];
    if (!points.length) {
      if (legend) { legend.innerHTML = "<i class=\"legend-best\"></i>实际晋升序列（尚无评测）"; }
      node.innerHTML = "<div class=\"empty-state\">候选完成评测后将在这里显示得分曲线。</div>";
      return;
    }
    var measuredWidth = Math.round(node.getBoundingClientRect().width || 0);
    var compact = window.innerWidth <= 760 || measuredWidth && measuredWidth < 600;
    var width = compact ? Math.max(320, measuredWidth || 320) : Math.max(760, Math.min(1200, measuredWidth || 1000));
    var height = compact ? 240 : 280;
    var left = compact ? 42 : 55, right = compact ? 12 : 24, top = 20, bottom = 44;
    var scores = [];
    points.forEach(function (point) { scores.push(Number(point.score != null ? point.score : point.candidate_score)); scores.push(trajectoryIncumbentScore(point)); });
    scores = scores.filter(Number.isFinite);
    var min = Math.min.apply(null, scores), max = Math.max.apply(null, scores);
    var pad = Math.max(0.02, (max - min) * 0.2); min -= pad; max += pad;
    if (min >= 0 && max <= 1) { min = Math.max(0, min); max = Math.min(1, max); }
    if (max <= min) { max = min + 0.1; }
    var generations = Array.from(new Set(points.map(function (point) { return Number(point.generation || 0); }))).sort(function (a, b) { return a - b; });
    var generationIndex = {};
    generations.forEach(function (generation, index) { generationIndex[generation] = index; });
    var groupWidth = generations.length === 1 ? 0 : (width - left - right) / (generations.length - 1);
    var xCenter = function (generation) { return left + (generations.length === 1 ? (width - left - right) / 2 : generationIndex[generation] * groupWidth); };
    var x = function (point) {
      var generation = Number(point.generation || 0);
      var siblings = points.filter(function (item) { return Number(item.generation || 0) === generation; });
      var slot = siblings.indexOf(point);
      var spread = Math.min(24, groupWidth > 0 ? groupWidth * 0.22 : 24);
      var offset = siblings.length <= 1 ? 0 : (slot - (siblings.length - 1) / 2) * (spread / Math.max(1, siblings.length - 1));
      return xCenter(generation) + offset;
    };
    var y = function (value) { return top + (max - value) / (max - min) * (height - top - bottom); };
    var incumbentByGeneration = generations.map(function (generation) {
      var group = points.filter(function (point) { return Number(point.generation || 0) === generation; });
      var withIncumbent = group.filter(function (point) { return Number.isFinite(trajectoryIncumbentScore(point)); });
      var source = withIncumbent.length ? withIncumbent[withIncumbent.length - 1] : null;
      var value = source ? trajectoryIncumbentScore(source) : NaN;
      return { generation: generation, value: value };
    }).filter(function (item) { return Number.isFinite(item.value); });
    var cohortDigests = Array.from(new Set(points.map(function (point) { return String(point.evaluation_cohort_digest || ""); }).filter(Boolean)));
    var crossCohort = cohortDigests.length > 1 || String(run && run.best_observed_score_scope || "").indexOf("cross_cohort") >= 0;
    var incumbentPath = crossCohort ? "" : incumbentByGeneration.map(function (item, index) { return (index ? "L" : "M") + xCenter(item.generation).toFixed(1) + " " + y(item.value).toFixed(1); }).join(" ");
    if (legend) { legend.innerHTML = "<i class=\"legend-best\"></i>" + (crossCohort ? "跨 cohort 不连线" : incumbentPath ? "实际晋升序列" : "实际晋升序列（暂无可绘制记录）"); }
    var grid = [0, 1, 2, 3, 4].map(function (step) { var value = min + (max - min) * step / 4; var py = y(value); return "<line class=\"chart-grid\" x1=\"" + left + "\" y1=\"" + py + "\" x2=\"" + (width - right) + "\" y2=\"" + py + "\"/><text class=\"chart-axis\" x=\"" + (left - 8) + "\" y=\"" + (py + 4) + "\" text-anchor=\"end\">" + value.toFixed(2) + "</text>"; }).join("");
    var labelEvery = Math.max(1, Math.ceil(generations.length / (compact ? 6 : 12)));
    var axisLabels = generations.map(function (generation) {
      var groupIndex = generationIndex[generation];
      return groupIndex % labelEvery === 0 || groupIndex === generations.length - 1
        ? "<text class=\"chart-axis\" x=\"" + xCenter(generation) + "\" y=\"" + (height - 20) + "\" text-anchor=\"middle\">第 " + escapeHTML(String(generation)) + " 轮</text>"
        : "";
    }).join("");
    var circles = points.map(function (point, index) {
      var score = Number(point.score != null ? point.score : point.candidate_score);
      var label = point.candidate_id || "候选方案 " + (index + 1);
      var generation = Number(point.generation || 0);
      return "<circle class=\"chart-point\" cx=\"" + x(point) + "\" cy=\"" + y(score) + "\" r=\"4\"><title>" + escapeHTML(label) + "（第 " + generation + " 轮候选）：" + score.toFixed(4) + "</title></circle>";
    }).join("");
    node.innerHTML = "<svg viewBox=\"0 0 " + width + " " + height + "\" width=\"" + width + "\" height=\"" + height + "\" role=\"img\" aria-labelledby=\"trajectory-title trajectory-description\" preserveAspectRatio=\"xMidYMid meet\"><title id=\"trajectory-title\">按轮次分组的候选原始得分与实际晋升序列</title><desc id=\"trajectory-description\">蓝色点表示同轮候选的原始得分；只有同一冻结 cohort 的服务端 incumbent 得分才允许连线。</desc>" + grid + (incumbentPath ? "<path class=\"chart-best\" d=\"" + incumbentPath + "\"/>" : "") + circles + axisLabels + "<text class=\"chart-axis\" x=\"" + ((left + width - right) / 2) + "\" y=\"" + (height - 4) + "\" text-anchor=\"middle\">进化轮次</text></svg>" + (crossCohort ? "<p class=\"trajectory-comparability-note\">不同反馈窗口使用不同 cohort，原始得分不可直接纵向比较，因此不绘制趋势连线；晋升以各轮同 cohort holdout 为准。</p>" : "");
  }

  function evolutionCheckText(value) {
    return {
      joint_recheck_pending: "边界收益待一次双臂复核，暂不更新",
      joint_recheck_not_confirmed: "双臂复核未确认收益，保留原最佳版本",
      joint_recheck_retained: "未满足更新条件，保留原最佳版本",
      temporal_evidence_incomplete: "缺少可核验的跨日比较证据",
      additional_time_blocks_required: "时间覆盖不足，需要新的时间数据",
      temporal_coverage_failed: "部分时间段的有效预测覆盖不足",
      gain_not_consistent_across_time_blocks: "不同时间段的收益不一致",
      incompatible_comparison_contract: "新旧版本的比较条件不一致",
      champion_evaluation_incomplete: "原版本的评测结果不完整",
      challenger_evaluation_incomplete: "新版本的评测结果不完整",
      challenger_safety_gate_failed: "新版本未通过安全检查",
      challenger_cell_regression: "部分预测指标退化超过允许范围",
      cell_regression: "部分预测指标退化超过允许范围",
      no_positive_score_delta: "同批得分没有提高",
      probation_below_practical_delta: "待复核：收益未超过实际改进阈值，继续保留冠军",
      probation_incomplete_paired_blocks: "待复核：时间块内缺少完整配对评分证据，继续保留冠军",
      probation_insufficient_evidence: "待复核：配对时间块不足，继续保留冠军",
      probation_confidence_interval_crosses_zero: "待复核：配对置信区间包含零，继续保留冠军",
      probation_invalid_block_evidence: "待复核：缺少完整时间块证据，继续保留冠军",
      below_practical_delta: "提高值未超过本轮规定的阈值",
      below_practical_score_delta: "提高值未超过本轮实用增益阈值",
      challenger_improved: "同批得分提高，满足保留条件",
      scientific_gate_failed: "科学约束检查未通过",
      repeated_behavior_requires_revisit_reason: "回到既有策略须说明安全恢复、新批证据或同批复核依据",
      duplicate_recent_rejected_bundle: "重复提交了相同父版本上已拒绝的修订",
      constraint_violations: "存在约束违规",
      coverage_failed: "有效预测覆盖不足",
      objective_grid_incomplete: "预测目标或时距不完整",
      strict_agent_chain_failed: "预测执行过程未通过完整性检查",
      paired_strict_agent_chain_failed: "比较双方未同时通过执行完整性检查，运行失败不能视为科学改进",
      paired_scoring_evidence_incomplete: "比较双方缺少完整成功评分，失败惩罚不能作为科学晋级依据",
      screening_exploration_only: "初筛仅供探索，不满足认证条件"
    }[String(value || "")] || compactTechnicalText(value);
  }

  function evolutionEvidenceSigned(value) {
    var number = value == null || value === "" ? NaN : Number(value);
    if (!Number.isFinite(number)) { return "待计算"; }
    return (number > 0 ? "+" : "") + formatNumber(number, 4);
  }

  function modelContractPreflightDetails(run) {
    var evidence = run && run.model_contract_preflight || {};
    if (evidence.status !== "verified" || evidence.scope !== "tool_and_schema_transport_only") { return ""; }
    var roles = Array.isArray(evidence.roles) ? evidence.roles : [];
    return "<details class=\"model-preflight-evidence technical-details\"><summary>创建前模型预检审计（" + escapeHTML(formatNumber(roles.length)) + " 个职责）</summary><p>此处记录创建时通过的工具与结构化响应检查；有效期用于新运行准入，之后到期不会推翻本次历史记录。科学认证由评测证据另行判断。</p>"
      + roles.map(function (role) { return "<div class=\"model-preflight-role\"><strong>" + escapeHTML({researcher: "策略研究职责", "generation-judge": "轮末评审职责"}[role.role] || role.role || "未记录职责") + " · " + escapeHTML(modelReferenceLabel(role.model_id)) + "</strong><span>核验 " + escapeHTML(formatDate(role.checked_at)) + " · 当时回执有效至 " + escapeHTML(formatDate(role.expires_at)) + "</span><code>回执 " + escapeHTML(role.receipt_id || "未记录") + "</code><code>身份校验 " + escapeHTML(role.identity_digest || "未记录") + "</code><code>回执校验 " + escapeHTML(role.receipt_digest || "未记录") + "</code></div>"; }).join("")
      + "<code>整体审计校验 " + escapeHTML(evidence.audit_digest || "未记录") + "</code></details>";
  }

  function searchModeLabel(policy) {
    return policy === "cellwise_training_evidence@1" ? "逐目标、逐时距改进"
      : policy ? "未识别的冻结搜索策略：" + policy : "标准搜索（未启用逐目标、逐时距改进）";
  }

  function frozenRunFeatureRows(run) {
    // Use this run's frozen configuration, never the current creation form or
    // the presence of newer server code/diagnostic output.
    var configuration = run && (run.frozen_feature_policies || run.configuration) || {};
    var diagnostics = configuration.behavior_diagnostics_policy;
    var novelty = configuration.candidate_novelty_policy;
    var executability = configuration.candidate_executability_policy;
    var contribution = run && run.evolution_funnel && run.evolution_funnel.prediction_contribution;
    return [
      ["冻结搜索方式", searchModeLabel(configuration.targeted_search_policy)],
      ["预测贡献诊断", ["prediction_contribution@1", "prediction_contribution@2"].indexOf(diagnostics) >= 0
        ? "已启用" + (contribution && contribution.latest_training_summary ? "，已有训练证据" : "，等待训练证据")
        : diagnostics ? "未识别的冻结策略：" + diagnostics : "未启用（冻结合同未记录）"],
      ["运行内行为去重", novelty === "run_behavior_novelty@1" ? "已启用"
        : novelty ? "未识别的冻结策略：" + novelty : "未启用（冻结合同未记录）"],
      ["策略程序输入预检", executability === "strategy_tool_input_contract@1" ? "已启用（不代表预测效果通过）"
        : executability ? "未识别的冻结策略：" + executability : "未启用（冻结合同未记录）"]
    ];
  }

  function renderFrozenRunContract(run) {
    var node = $("#frozen-run-contract");
    if (!node) { return; }
    node.hidden = !run;
    node.innerHTML = run ? "<strong>本运行的冻结配置</strong><span>" + frozenRunFeatureRows(run).map(function (row) {
      return escapeHTML(row[0] + "：" + row[1]);
    }).join("；") + "。页面的新选项只应用于新创建的运行。</span>" : "";
  }

  function renderPairedToolEffects(diagnostics) {
    if (!diagnostics) { return ""; }
    var pairs = Array.isArray(diagnostics.paired_tool_effects) ? diagnostics.paired_tool_effects.filter(function (pair) { return pair && typeof pair === "object"; }) : [];
    if (!pairs.length && diagnostics.enabled !== true && !diagnostics.latest_training_summary) { return ""; }
    var boundary = "格指目标 × 时距组合。原始工具数值变化不代表性能提升；最终保留仍按本运行原有的配对与轮末门槛决定。";
    var header = "<div class=\"dataset-stat\"><span>原始预测工具的父子变化</span><strong>同输入比较证据</strong><small>" + boundary + "</small></div>";
    if (!pairs.length) {
      return header + "<div class=\"dataset-stat\"><span>父子工具比较</span><strong>尚无可核验的父子比较记录</strong><small>未记录变化计数，不能解释为没有变化。</small></div>";
    }
    var reasons = {
      same_cohort_required: "父子版本缺少同一训练样本组的证明。",
      same_diagnostic_version_required: "父子版本的贡献诊断版本不一致或缺失。",
      cell_coverage_mismatch: "父子版本的目标与时距覆盖不一致。",
      complete_matching_inputs_required: "父子版本缺少完整且一致的输入或原始工具输出证据。"
    };
    return header + pairs.map(function (pair) {
      var changed = pair.changed_cells;
      var measured = pair.status === "measured" && Number.isSafeInteger(pair.compared_cells) && pair.compared_cells > 0
        && Array.isArray(changed) && changed.length <= pair.compared_cells
        && changed.every(function (cell) { return typeof cell === "string" && cell.length > 0; });
      var title = "父子比较" + (pair.comparison_id ? " · " + shortId(pair.comparison_id) : "");
      var result = measured ? "原始工具有变化 " + changed.length + " / " + pair.compared_cells + " 格" : "同输入证据不足";
      var detail = measured ? changed.length ? "变化格：" + changed.map(function (cell) {
        var match = /^([^@]+)@(\d+(?:\.\d+)?)h$/.exec(cell);
        return match ? (targetLabels[match[1]] || match[1]) + " · " + match[2] + "h" : cell;
      }).join("；") : "所比较格未发现原始工具输出变化。"
        : reasons[pair.reason] || "父子比较记录不完整，未报告可核验的变化计数。";
      return "<div class=\"dataset-stat\"><span title=\"" + escapeHTML(pair.comparison_id || "") + "\">" + escapeHTML(title)
        + "</span><strong>" + escapeHTML(result) + "</strong><small>" + escapeHTML(detail) + "</small></div>";
    }).join("");
  }

  function renderEvidenceFunnel(run) {
    var funnel = run && run.evolution_funnel;
    if (!funnel) { return ""; }
    var labels = {structural_validity: "登记修改的结构解析覆盖", behavior_change: "受控行为差异", screening_coverage: "初筛覆盖", training_coverage: "训练覆盖", local_pair_completion: "局部配对完成", local_positive_gain: "局部点估计为正", local_acceptance: "局部门禁接受", search_adoption: "轮末更新工作版本", independent_certification: "独立认证"};
    var rows = Object.keys(labels).map(function (key) {
      var item = (funnel.stages || {})[key] || {};
      var measured = item.numerator != null && item.denominator != null;
      return "<div class=\"dataset-stat\"><span>" + escapeHTML(labels[key]) + "</span><strong>" + escapeHTML(measured ? String(item.numerator) + " / " + String(item.denominator) : "未记录可核验分母") + "</strong><small>" + escapeHTML(measured && item.denominator === 0 ? "尚未开始" : item.rate != null ? formatNumber(item.rate * 100, 1) + "%" : "不计算成功率") + "</small></div>";
    }).join("");
    var failures = funnel.failure_observations || {};
    var effects = funnel.execution_effect_observations;
    if (effects) { rows += "<div class=\"dataset-stat\"><span>执行观测回执 " + escapeHTML(String(effects.recorded)) + " 条</span><strong>通过 " + escapeHTML(String(effects.passed)) + "；失败 " + escapeHTML(String(effects.failed)) + "；不确定 " + escapeHTML(String(effects.inconclusive)) + "；未触发 " + escapeHTML(String(effects.not_exercised)) + "</strong><small>按训练回执去重；执行观测不等于受控行为差异或性能提升。登记修改的结构解析覆盖不包含生成前被拒绝的输出。</small></div>"; }
    var contribution = (funnel.prediction_contribution || {}).latest_training_summary;
    if (contribution) {
      rows += "<div class=\"dataset-stat\"><span>最近训练批次的预测行为</span><strong>工具输出与最终预测</strong><small>以下按评分单元计数；数值相等不能证明因果作用，也不等于性能提升。</small></div>";
      (contribution.cells || []).forEach(function (cell) {
        rows += "<div class=\"dataset-stat\"><span>" + escapeHTML(cell.target + " · " + cell.horizon_hours + "h") + "</span><strong>调用候选工具 " + escapeHTML(String(cell.candidate_tool_cells)) + " / " + escapeHTML(String(cell.cells)) + "</strong><small>工具等于基线 " + escapeHTML(String(cell.model_equals_baseline)) + "；最终等于工具 " + escapeHTML(String(cell.final_equals_model)) + "；最终等于基线 " + escapeHTML(String(cell.final_equals_baseline)) + "</small><small>完整轨迹 " + escapeHTML(String(cell.trace_available_cells)) + " / " + escapeHTML(String(cell.cells)) + "；缺失轨迹时不判断分支是否触发</small></div>";
      });
    }
    rows += renderPairedToolEffects(funnel.prediction_contribution);
    var reasons = [["operational", "服务/执行"], ["invalid_change", "修改无效"], ["insufficient_evidence", "证据不足"], ["performance", "性能未过门槛"], ["unclassified", "未分类"]].map(function (pair) { return pair[1] + " " + String(failures[pair[0]] || 0); }).join("；");
    return "<div class=\"dataset-stat\"><span>执行、采纳与科学结论</span><strong>分别核验，不能互相替代</strong><small>工程执行：结构与执行回执；搜索采纳：同组配对及轮末保留；科学结论：独立验证。effect passed 只表示执行观测通过；局部正收益不等于独立认证。</small></div>"
      + "<div class=\"dataset-stat\"><span>证据漏斗 · 仅本运行</span><strong>已登记候选 " + escapeHTML(String(funnel.registered_candidates)) + "；预算未分配 " + escapeHTML(String(funnel.budget_unallocated_candidates)) + "</strong><small>已应用修改 " + escapeHTML(String(funnel.applied_edits)) + " 次，应用不等于收益已验证</small><small>" + escapeHTML(reasons) + "（原因记录数，可重叠，不是候选失败率）</small></div>" + rows;
  }

  // All counts and progress below come from the current projection. Missing
  // evidence stays unknown; elapsed time never changes a run's status.
  function processNumber(value, digits) {
    return value == null || value === "" || !Number.isFinite(Number(value)) ? "待核验" : formatNumber(value, digits);
  }
  function processCards(rows) {
    return rows.map(function (row) {
      return "<div class=\"dataset-stat\"><span>" + escapeHTML(row[0]) + "</span><strong>" + escapeHTML(row[1]) + "</strong>"
        + (row[2] ? "<small>" + escapeHTML(row[2]) + "</small>" : "") + "</div>";
    }).join("");
  }
  function executionRound(run) {
    var rounds = run && Array.isArray(run.rounds) ? run.rounds : [];
    var generation = run && run.execution_progress && run.execution_progress.current_generation;
    return rounds.find(function (round) { return round.generation === generation; }) || rounds[rounds.length - 1] || null;
  }
  function executionCandidateFor(run) {
    if (!run) { return null; }
    var progress = run.execution_progress || {};
    var id = progress.current_candidate_id || progress.last_evidence_candidate_id;
    // A generation-scoped stage must not inherit the previous candidate.
    if (!progress.current_candidate_id && ["research", "proposal", "starting"].indexOf(progress.current_stage) >= 0) { return null; }
    return processCandidates(run).find(function (candidate) { return candidate.id === id; }) || null;
  }
  function renderExecutionSamples(candidate, run) {
    var node = $("#sample-inference-list");
    var page = candidate && run && candidateSampleSelectionMatches(run.id, candidate.id) ? state.candidateSamplePage : null;
    var rows = page && Array.isArray(page.rows) ? page.rows : [];
    $("#sample-inference-count").textContent = page ? processNumber(page.total) + " 格已封存" : "样本页尚未读取";
    node.innerHTML = candidate
      ? "<p>候选 <code>" + escapeHTML(shortId(candidate.id)) + "</code> · " + (rows.length ? "本页已读取 " + rows.length + " 格。" : "尚无该候选的样本页。")
        + " 在候选详情中查看逐格观测、预测、误差和执行状态。</p>"
      : "<p>本阶段没有绑定候选；不会显示其它候选的历史样本。</p>";
  }
  function renderProcessSummary(run) {
    renderFrozenRunContract(run);
    if (!run) {
      var creation = createDisplayStatus();
      $("#process-summary").innerHTML = creation
        ? "<p><strong>" + escapeHTML(createPhaseLabel(creation)) + "</strong> · " + escapeHTML(creation.message || "") + "</p>"
        : "<p>创建运行后显示冻结配置和预检回执。</p>";
      return;
    }
    var configuration = run.configuration || {};
    $("#process-summary").innerHTML = processCards([
      ["策略模型", modelReferenceLabel(run.strategy_model_id || configuration.strategy_model_id)],
      ["评审模型", modelReferenceLabel(run.review_model_id || configuration.review_model_id)],
      ["冻结轮数", processNumber(run.total_generations)],
      ["候选／样本并发上限", processNumber(run.candidate_concurrency) + " / " + processNumber(run.sample_concurrency)],
      ["执行预算", "每轮基础 200 次＋条件复核预留 100 次", "一次为一个候选版本在一个预测时点的执行；不同执行可使用同一环境起点。"],
      ["科学边界", "10 / 25 / 50 个时点", "初筛、父子配对、轮末比较分开；独立验证另行执行。"]
    ]) + modelContractPreflightDetails(run);
  }
  function processStageLabel(stage) {
    return evolutionStageLabels[stage] || {"sample.plan": "逐时点预测", "sample.critic": "预测评审", "candidate.local_edit": "局部修订", "generation.reflect": "轮末反思", "candidate.propose": "生成候选", "search-plan": "规划检索", "research-synthesis": "综合研究"}[stage] || stage || "等待阶段事件";
  }
  function renderExecutionMonitor(run) {
    var progress = run && run.execution_progress || {};
    var stage = progress.stage_progress || {};
    var activity = progress.dsh_activity || {};
    var retry = progress.retry_wait;
    var live = Boolean(run && ["running", "starting"].indexOf(run.status) >= 0);
    var status = $("#execution-monitor-status");
    status.textContent = run ? statusText(run.status) : "等待运行";
    status.className = "pill " + (run ? statusClass(run.status) : "pill-neutral");
    var percent = progress.overall_progress_percent;
    var known = percent != null && Number.isFinite(Number(percent));
    var value = known ? Math.max(0, Math.min(100, Number(percent))) : 0;
    $("#execution-progress-label").textContent = live ? processStageLabel(activity.dsh_stage || progress.current_stage) : run ? statusText(run.status) : "尚未开始";
    $("#execution-progress-percent").textContent = known ? formatNumber(value, 1) + "%" : "待核验";
    $("#execution-progress-fill").style.width = value + "%";
    var track = $("#execution-progress-track");
    if (known) { track.setAttribute("aria-valuenow", String(value)); } else { track.removeAttribute("aria-valuenow"); }
    $("#execution-progress-detail").textContent = live && retry
      ? "等待有界重试 · " + (retry.retry_at ? formatDate(retry.retry_at) : "以服务端调度为准") + " · 第 " + processNumber(retry.attempt) + " 次"
      : live ? "进度来自已封存事件；正在生成的响应可能尚未计入用量。" : "以下保留停点证据；已结束或暂停的运行不显示为持续执行。";
    $("#execution-generation-progress").textContent = "完成轮数 " + processNumber(run && run.generation) + " / " + processNumber(run && run.total_generations);
    $("#execution-epoch-progress").textContent = "本轮进度 " + (progress.epoch_progress_percent == null ? "待核验" : processNumber(progress.epoch_progress_percent, 1) + "%");
    $("#execution-candidate-progress").textContent = "已生成候选 " + processNumber(run && run.candidates_count);
    $("#execution-sample-progress").textContent = "本轮预算进度 " + processNumber(stage.completed_samples) + " / " + processNumber(stage.total_samples) + "（含条件复核预留）";
    $("#execution-last-activity").textContent = "最近账本活动 " + (progress.last_event_at ? formatDate(progress.last_event_at) : "未记录");
    var started = activity.started_at;
    var seconds = started ? Math.max(0, (Date.now() - Date.parse(started)) / 1000) : null;
    $("#execution-heartbeat").textContent = live && Number.isFinite(seconds)
      ? "本次阶段耗时 " + formatNumber(seconds, 0) + " 秒（不据此判断停滞）" : "阶段耗时未记录";
    renderRuntimeEvidence(run);
    var candidate = executionCandidateFor(run);
    $("#active-candidate-summary").innerHTML = candidate
      ? processCards([["当前候选", shortId(candidate.id), candidateStatusText(candidate.status)], ["得分", processNumber(candidate.score, 4), "只与相同冻结数据组比较"]])
      : "<p>当前阶段未绑定候选。所有候选、修改内容和九格样本在“候选方案”中查看。</p>";
    renderExecutionSamples(candidate, run);
  }
  function renderRuntimeEvidence(run) {
    var runtime = run && run.dsh_runtime || {};
    var usage = runtime.provider_usage || {};
    var coverage = usage.accounting_coverage || {};
    var pressure = runtime.context_pressure || {};
    var diagnostics = run && run.execution_diagnostics || {};
    var complete = coverage.complete === true
      && Number.isSafeInteger(coverage.launched_session_count) && coverage.launched_session_count > 0
      && coverage.complete_session_count === coverage.launched_session_count
      && coverage.observed_session_count === coverage.launched_session_count
      && coverage.settled_session_count === coverage.launched_session_count
      && coverage.unobserved_session_count === 0;
    $("#execution-token-progress").textContent = usage.available === true
      ? "供应商已报告 " + processNumber(usage.total_tokens) + " Token · " + (complete ? "本运行已启动会话回执完整" : "已观测下限；缺失用量未知")
      : "供应商用量尚未报告；不能据此认定免费";
    var rows = [
      ["实际预测时点执行", processNumber(diagnostics.adaptive_all_settled_origins), "不把复用的初筛记录再次计为执行"],
      ["评分单元", processNumber(diagnostics.adaptive_scored_cells), "完整九格样本另见候选详情"],
      ["完整用量回执／已启动会话", processNumber(coverage.complete_session_count) + " / " + processNumber(coverage.launched_session_count), "子任务成功、用量完整和科学通过分别判断"],
      ["活跃会话上下文压力", pressure.available === true && Number.isSafeInteger(pressure.maximum_total_tokens)
        ? processNumber(pressure.maximum_total_tokens) + " Token（活跃会话最大值）" : "未报告", "仅观测；未提供容量分母时不计算百分比，不自动缩减冻结模型、预算或样本"]
    ];
    var stages = usage.by_stage || {};
    Object.keys(stages).sort().forEach(function (key) {
      rows.push([processStageLabel(key), processNumber(stages[key].total_tokens) + " Token", "供应商已报告值"]);
    });
    $("#runtime-evidence").innerHTML = processCards(rows);
    var failure = run && run.latest_runtime_failure;
    $("#runtime-failure").hidden = !failure;
    $("#runtime-failure").textContent = failure
      ? "最近执行异常（可能已恢复）：" + [failure.stage, failure.error_code || failure.code, failure.public_error || failure.message].filter(Boolean).map(compactTechnicalText).join(" · ") + "。运行终态与恢复情况以阶段事件为准。"
      : "";
  }
  function renderEvolutionEvidence(run) {
    var evidence = run && run.evolution_evidence || {};
    var champion = evidence.search_version || evidence.global_champion;
    var certified = evidence.certified_version;
    var decisions = Array.isArray(evidence.generation_decisions) ? evidence.generation_decisions : [];
    $("#evolution-evidence-status").textContent = decisions.length ? decisions.length + " 轮比较记录" : "等待轮末比较";
    $("#global-champion-card").innerHTML = processCards([
      ["当前搜索版本", champion ? champion.is_initial_seed ? "初始种子基线" : shortId(champion.candidate_id) : "尚未冻结", champion ? "修订 " + shortId(champion.candidate_revision_id) : ""],
      ["稳健认证版本", certified ? shortId(certified.candidate_revision_id) : "暂无认证证据", "搜索保留不等于独立验证；发布前仍需独立最终测试。"]
    ]);
    var capacity = evidence.capacity || {};
    $("#evolution-capacity-note").textContent = capacity.available === true
      ? "计划使用 " + processNumber(capacity.required_unique_origins) + " 个不同环境起点；复用次数 " + processNumber(capacity.reused_origin_occurrences) + "。时间相关的起点不等于独立试验。"
      : "数据容量证据尚未记录。";
    $("#generation-decision-list").innerHTML = decisions.map(function (decision) {
      var selected = decision.selected_arm;
      var title = decision.committed === false ? "等待固化，不作为已保留结果" : selected === "incumbent" ? "保留轮初版本" : "保留挑战版本用于后续搜索";
      var arms = (decision.arms || []).map(function (arm) {
        var failures = Array.from(new Set([].concat(arm.failures || [], arm.search_failures || [], arm.certification_failures || [])));
        return "<tr><td>" + escapeHTML(arm.arm === "incumbent" ? "轮初版本" : "挑战版本") + (selected === arm.arm && decision.committed !== false ? "（已选择）" : "")
          + "</td><td>" + escapeHTML(processNumber(arm.absolute_score, 4)) + "</td><td>" + escapeHTML(evolutionEvidenceSigned(arm.holdout_delta))
          + "</td><td>" + escapeHTML(failures.map(evolutionCheckText).join("；") || (arm.arm === "incumbent" ? "同组对照" : "无已记录的未通过原因；以选择结果为准"))
          + (arm.joint_recheck ? "<small>条件复核：" + escapeHTML(arm.joint_recheck.status) + "；复测增益 " + escapeHTML(evolutionEvidenceSigned(arm.joint_recheck.repeated_delta)) + "</small>" : "") + "</td></tr>";
      }).join("");
      return "<article class=\"generation-evidence-card\"><h3>第 " + escapeHTML(processNumber(decision.generation)) + " 轮 · " + escapeHTML(title)
        + "</h3><p>" + (decision.same_cohort === true ? "同一冻结数据组比较" : "同组证据待核验")
        + "；搜索选择与认证选择分别记录。</p><div class=\"table-wrap\"><table><thead><tr><th>版本</th><th>基线技能分</th><th>同组增益</th><th>门槛／复核</th></tr></thead><tbody>" + arms + "</tbody></table></div></article>";
    }).join("") || "<p>暂无轮末比较，不能据此判断改进。</p>";
    $("#science-evidence").innerHTML = renderEvidenceFunnel(run);
  }
  function renderAdaptiveTrajectories(run) {
    var rows = [];
    (run && run.adaptive_trajectories || []).forEach(function (lane) {
      (lane.batches || []).forEach(function (batch) { rows.push({lane: lane, batch: batch}); });
    });
    $("#adaptive-trajectory-count").textContent = rows.length + " 个批次记录";
    $("#adaptive-trajectory-table").innerHTML = rows.map(function (entry) {
      var lane = entry.lane, batch = entry.batch;
      var operations = batch.next_challenger_operations || [];
      var decision = {initial_champion: "确定父版本", challenger_promoted: "接受修订", champion_retained: "保留父版本"}[batch.comparison_decision] || "比较尚未封存";
      var revision = batch.champion_after_revision_id || batch.candidate_revision_id;
      var changes = operations.map(function (operation) {
        return [mutationOperationLabels[operation.op] || operation.op, operation.name || operation.target || operation.program_id || operation.predictor_id, operation.value == null ? "" : JSON.stringify(operation.value)].filter(Boolean).join(" · ");
      }).join("；");
      return "<tr><td>第 " + escapeHTML(processNumber(Number(lane.generation) + 1)) + " 轮<small>" + escapeHTML(shortId(lane.candidate_id)) + "</small></td>"
        + "<td>批次 " + escapeHTML(processNumber(batch.batch_index)) + "<small>" + escapeHTML(processNumber(batch.origin_count)) + " 个时点</small><code>" + escapeHTML(shortId(batch.cohort_digest)) + "</code></td>"
        + "<td><code>" + escapeHTML(shortId(revision)) + "</code><small>新修订：" + escapeHTML(shortId(batch.next_challenger_revision_id || "未生成")) + "</small></td>"
        + "<td>父 " + escapeHTML(processNumber(batch.champion_score, 4)) + " · 子 " + escapeHTML(processNumber(batch.challenger_score, 4)) + "<small>增益 " + escapeHTML(evolutionEvidenceSigned(batch.score_delta)) + "；须超过 " + escapeHTML(processNumber(batch.minimum_score_delta, 4)) + "</small></td>"
        + "<td>" + escapeHTML(changes || "无新的修改摘要") + "</td><td>" + escapeHTML(decision) + "<small>" + escapeHTML(evolutionCheckText(batch.comparison_reason || batch.edit_reason || "")) + "</small></td></tr>";
    }).join("") || "<tr><td colspan=\"6\" class=\"empty-state\">等待局部配对证据。</td></tr>";
  }
  function renderProcessEvents(run, creating) {
    var events = creating ? [] : state.events;
    var total = creating ? 0 : Math.max(events.length, Number(state.eventTotal || 0));
    var toggle = $("#toggle-events-button");
    toggle.hidden = total <= 12;
    toggle.textContent = state.showAllEvents ? "收起事件" : "展开最近事件（共 " + total + " 条）";
    toggle.setAttribute("aria-expanded", String(state.showAllEvents));
    $("#event-list").innerHTML = (state.showAllEvents ? events : events.slice(0, 12)).map(function (event) {
      var payload = payloadOf(event);
      var title = eventLabels[event.type] || event.type;
      var detail = [payload.stage ? processStageLabel(payload.stage) : "", payload.status || "", payload.candidate_id ? shortId(payload.candidate_id) : "", payload.error_code || "", payload.public_error || ""];
      if (payload.completed_samples != null) { detail.push("已结算 " + processNumber(payload.completed_samples) + " / " + processNumber(payload.total_samples)); }
      if (payload.attempt != null) { detail.push("尝试 " + processNumber(payload.attempt)); }
      if (payload.retry_at) { detail.push("计划重试 " + formatDate(payload.retry_at)); }
      // A child failure can be recovered. Do not derive a run terminal state.
      return "<div class=\"event-row\"><time>" + escapeHTML(formatTime(event.occurred_at)) + "</time><div><strong>" + escapeHTML(title)
        + "</strong><small>" + escapeHTML(detail.filter(Boolean).map(compactTechnicalText).join(" · ")) + "</small></div></div>";
    }).join("") || "<p class=\"empty-state\">当前运行暂无事件。</p>";
    $("#projection-revision").textContent = "状态视图版本 " + (run && run.projection_revision || "—");
  }
  function renderProcess() {
    var creating = createDisplayStatus();
    var run = displayedRun();
    var status = $("#process-status-pill");
    status.textContent = creating ? createPhaseLabel(creating) : run ? statusText(run.status) : "暂无运行";
    status.className = "pill " + (creating && creating.state === "failed" ? "pill-red" : run ? statusClass(run.status) : "pill-neutral");
    $("#workspace-process").setAttribute("aria-busy", String(Boolean(creating && creating.state !== "failed" || run && run.status === "running")));
    renderProcessSummary(run);
    renderExecutionMonitor(run);
    renderEvolutionEvidence(run);
    renderAdaptiveTrajectories(run);
    renderTrajectory();
    renderProcessEvents(run, creating);
  }
