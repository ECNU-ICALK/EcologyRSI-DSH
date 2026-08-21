"use strict";

  // The training asset is an episode, not a single row.  This renderer keeps
  // the public view compact while preserving the causal order of the host
  // controlled workflow: input -> agent interaction -> feedback -> update ->
  // prediction.  It accepts both the new trajectory contract and older
  // episode/prediction fields so an upgraded plugin can read old runs.
  var trainingTracePhaseLabels = {
    input: "样本输入", input_context: "样本输入", sample_input: "样本输入", task_input: "样本输入", strategy_input: "样本输入",
    interaction: "智能体交互", agent_interaction: "智能体交互", agent_research: "智能体调研", agent_proposal: "智能体提案", proposal: "智能体交互", proposal_response: "智能体交互",
    implementation: "宿主能力编译", host_compile: "宿主能力编译", compile: "宿主能力编译", candidate: "宿主能力编译", training: "候选训练", training_prediction: "候选训练",
    feedback: "反馈评测", agent_feedback: "智能体反馈", training_feedback: "反馈评测", evaluation: "反馈评测", judge: "独立评审",
    optimization: "优化更新", agent_optimization: "优化更新", update: "优化更新", decision: "优化更新", promotion: "优化更新",
    prediction: "最终预测", final_prediction: "最终预测", final_result: "最终结果", result: "最终结果", output: "最终结果"
  };
  var trainingTraceSensitiveKeys = [
    "api_key", "apikey", "token", "secret", "password", "credential", "authorization",
    "prompt", "reasoning", "private_reasoning", "chain_of_thought", "messages", "raw", "raw_rows"
  ];

  function trainingTraceIsObject(value) {
    return value && typeof value === "object" && !Array.isArray(value);
  }
  function trainingTraceArray(value) {
    if (Array.isArray(value)) { return value; }
    if (!trainingTraceIsObject(value)) { return []; }
    for (var key of ["steps", "records", "events", "items", "trace", "trajectory"]) {
      if (Array.isArray(value[key])) { return value[key]; }
    }
    // The browser projection may use named, auditable stages instead of an
    // array.  Preserve their declared order when turning that shape into a
    // timeline.
    var namedStages = ["input_context", "agent_research", "agent_proposal", "host_compile", "training_prediction", "agent_feedback", "agent_optimization", "final_result"];
    var named = namedStages.filter(function (key) { return value[key] != null; }).map(function (key) {
      var item = value[key];
      return trainingTraceIsObject(item) ? Object.assign({ phase: key }, item) : { phase: key, summary: item };
    });
    if (named.length) { return named; }
    return [];
  }
  function trainingTraceFirst(value, keys) {
    if (!trainingTraceIsObject(value)) { return null; }
    for (var key of keys) {
      if (value[key] != null && value[key] !== "") { return value[key]; }
    }
    return null;
  }
  function trainingTraceSafeKey(key) {
    var normalized = String(key || "").toLowerCase().replace(/[\s-]+/g, "_");
    return trainingTraceSensitiveKeys.indexOf(normalized) < 0 && !/(?:^|_)(?:token|secret|password|credential|reasoning)$/.test(normalized);
  }
  function trainingTraceText(value, fallback) {
    if (value == null || value === "") { return fallback || "未记录"; }
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      return compactTechnicalText(String(value));
    }
    return fallback || "结构化记录";
  }
  function trainingTraceValueText(value) {
    if (value == null || value === "") { return "未提供"; }
    if (typeof value === "number") { return formatNumber(value); }
    if (typeof value === "boolean") { return value ? "是" : "否"; }
    if (typeof value === "string") { return humanizeTechnicalText(value); }
    if (Array.isArray(value)) { return value.slice(0, 4).map(trainingTraceValueText).join("、") || "无"; }
    if (trainingTraceIsObject(value)) {
      return Object.keys(value).filter(trainingTraceSafeKey).slice(0, 4).map(function (key) {
        return humanizeTechnicalText(key) + "=" + trainingTraceValueText(value[key]);
      }).join("；") || "结构化记录";
    }
    return String(value);
  }
  function trainingTracePhase(value, fallback) {
    var raw = String(value || fallback || "").toLowerCase().replace(/[\s-]+/g, "_");
    if (trainingTracePhaseLabels[raw]) { return raw; }
    if (/input|context|dataset/.test(raw)) { return "input"; }
    if (/interact|agent|proposal|policy|strategy/.test(raw)) { return "interaction"; }
    if (/compile|implement|capability|candidate/.test(raw)) { return "implementation"; }
    if (/train/.test(raw)) { return "training"; }
    if (/feedback|evaluat|metric|judge|review/.test(raw)) { return raw.indexOf("judge") >= 0 || raw.indexOf("review") >= 0 ? "judge" : "feedback"; }
    if (/optim|update|decision|promot|select/.test(raw)) { return "optimization"; }
    if (/predict|forecast/.test(raw)) { return "prediction"; }
    return fallback || "feedback";
  }
  function trainingTraceStatus(value) {
    var raw = String(value || "completed").toLowerCase();
    return { done: "completed", complete: "completed", success: "completed", running: "running", started: "running", pending: "pending", waiting: "pending", failed: "failed", error: "failed" }[raw] || raw;
  }
  function trainingTraceStatusText(value) {
    return { completed: "已完成", running: "执行中", pending: "等待", failed: "失败", skipped: "已跳过" }[trainingTraceStatus(value)] || "已记录";
  }
  function trainingTraceStatusClass(value) {
    var status = trainingTraceStatus(value);
    return status === "completed" ? "is-completed" : status === "running" ? "is-running" : status === "failed" ? "is-failed" : "is-pending";
  }
  function trainingTracePartitionText(value) {
    if (typeof partitionText === "function") { return partitionText(value); }
    return {
      training_fit: "训练拟合分区", training_feedback: "训练反馈分区", train: "训练集",
      development: "开发集", validation: "验证集", test: "测试集", hidden: "隐藏评测集"
    }[String(value || "")] || "受限数据分区";
  }
  function trainingTraceActor(raw, phase) {
    var actor = trainingTraceFirst(raw, ["actor", "agent", "owner", "role", "model_id", "model", "component"]);
    if (actor) { return trainingTraceValueText(actor); }
    return { input: "数据集与任务", interaction: "策略模型", implementation: "DSH 宿主", training: "预测模型", feedback: "评测器", judge: "独立评审", optimization: "进化控制器", prediction: "预测模型" }[phase] || "进化流程";
  }
  function trainingTraceChangesText(value) {
    if (Array.isArray(value)) {
      return value.slice(0, 5).map(function (item) {
        if (!trainingTraceIsObject(item)) { return trainingTraceValueText(item); }
        var name = item.parameter || item.name || item.key || "参数";
        if (item.before != null || item.after != null) {
          return humanizeTechnicalText(name) + " " + trainingTraceValueText(item.before) + "→" + trainingTraceValueText(item.after);
        }
        return humanizeTechnicalText(name) + "=" + trainingTraceValueText(item.value != null ? item.value : item.after);
      }).join("；");
    }
    if (!value || typeof value !== "object") { return ""; }
    var changes = Object.keys(value).filter(trainingTraceSafeKey).slice(0, 4).map(function (key) {
      var item = value[key];
      if (trainingTraceIsObject(item) && (item.before != null || item.after != null)) {
        return humanizeTechnicalText(key) + " " + trainingTraceValueText(item.before) + "→" + trainingTraceValueText(item.after);
      }
      return humanizeTechnicalText(key) + "=" + trainingTraceValueText(item);
    });
    return changes.join("；");
  }
  function trainingTraceSummary(raw, phase, asset) {
    if (!trainingTraceIsObject(raw)) { return trainingTraceText(raw); }
    var phaseGroup = {
      input_context: "input", agent_research: "interaction", agent_proposal: "interaction",
      host_compile: "implementation", training_prediction: "training", agent_feedback: "feedback",
      agent_optimization: "optimization", final_result: "result"
    }[phase] || phase;
    var input = trainingTraceFirst(raw, ["input_summary", "input", "context", "features"]);
    var changes = trainingTraceFirst(raw, ["parameter_changes", "changes", "updates", "optimization", "requested_parameters", "accepted_parameters"]) || raw.parameters;
    var feedback = trainingTraceFirst(raw, ["feedback", "metrics", "evaluation"]);
    var explicit = trainingTraceFirst(raw, ["summary_zh", "summary", "message_zh", "message", "description", "action", "result", "rationale", "reason"]);
    if (explicit != null && typeof explicit !== "object") {
      var explicitText = humanizeTechnicalText(trainingTraceText(explicit));
      var explicitChanges = changes && (phaseGroup === "interaction" || phaseGroup === "implementation" || phaseGroup === "optimization") ? trainingTraceChangesText(changes) : "";
      var explicitResearch = phaseGroup === "interaction" && (raw.team || raw.research || raw.prediction_model || raw.strategy) ? [raw.team ? "团队：" + trainingTraceValueText(raw.team) : "", raw.research ? "调研：" + trainingTraceValueText(raw.research) : "", raw.prediction_model ? "预测模型：" + trainingTraceValueText(raw.prediction_model) : "", raw.strategy ? "进化策略：" + trainingTraceValueText(raw.strategy) : ""].filter(Boolean).join("；") : "";
      var explicitDecision = phaseGroup === "result" && raw.decision ? "结论：" + trainingTraceValueText(raw.decision) : "";
      return [explicitText, explicitResearch, explicitChanges ? "参数变化：" + explicitChanges : "", explicitDecision].filter(Boolean).join("；");
    }
    if (phaseGroup === "interaction" && (raw.research || raw.team || raw.prediction_model || raw.strategy || raw.knowledge_used)) {
      var team = raw.team ? "团队：" + trainingTraceValueText(raw.team) : "";
      var research = raw.research ? "调研：" + trainingTraceValueText(raw.research) : "";
      var model = raw.prediction_model ? "预测模型：" + trainingTraceValueText(raw.prediction_model) : "";
      var strategy = raw.strategy ? "进化策略：" + trainingTraceValueText(raw.strategy) : "";
      return [team, research, model, strategy].filter(Boolean).join("；") || "策略模型已完成结构化调研";
    }
    if (phaseGroup === "input") {
      var parent = raw.parent_candidate_id ? "；父方案 " + shortId(raw.parent_candidate_id) : "";
      var assetInput = asset && asset.input && typeof asset.input === "object" ? asset.input : {};
      return (input ? "冻结 " + trainingTraceValueText(input) : "冻结数据集 " + trainingTraceValueText(assetInput.dataset_id || asset && asset.dataset_id)) + parent;
    }
    if (phaseGroup === "interaction" || phaseGroup === "implementation") {
      return changes ? "提出参数方案：" + trainingTraceChangesText(changes) : input ? "读取上下文：" + trainingTraceValueText(input) : "已形成候选方案";
    }
    if (phaseGroup === "feedback" || phaseGroup === "judge") {
      var current = trainingTraceIsObject(raw.current_candidate) ? raw.current_candidate : {};
      var score = trainingTraceFirst(raw, ["score", "feedback_score", "skill_score"]);
      if (score == null) { score = current.score; }
      var currentMetrics = current.metrics || raw.metrics || {};
      var metricText = trainingTraceIsObject(currentMetrics) ? [currentMetrics.rmse != null ? "RMSE " + formatNumber(currentMetrics.rmse) : "", currentMetrics.mae != null ? "MAE " + formatNumber(currentMetrics.mae) : "", current.passed === true || raw.passed === true ? "约束通过" : ""].filter(Boolean).join(" · ") : "";
      return score != null ? "综合得分 " + formatNumber(score) + (metricText ? " · " + metricText : "") : feedback ? "反馈：" + trainingTraceValueText(feedback) : metricText || "已完成误差与约束检查";
    }
    if (phaseGroup === "optimization") {
      var decision = trainingTraceFirst(raw, ["decision", "status", "selection_reason"]);
      var nextCandidates = raw.next_candidate_ids || raw.next_candidates;
      var nextText = nextCandidates && (Array.isArray(nextCandidates) ? "下一轮候选 " + nextCandidates.slice(0, 3).map(shortId).join("、") : "") || "";
      return decision ? trainingTraceValueText(decision) + (changes ? " · " + trainingTraceChangesText(changes) : "") + (nextText ? " · " + nextText : "") : "已更新候选参数与保留判断";
    }
    if (phaseGroup === "training" || phaseGroup === "prediction") {
      var count = trainingTraceFirst(raw, ["shown_count", "sample_count"]);
      var modelId = trainingTraceFirst(raw, ["model_id", "predictor_id"]);
      return "预测模型 " + (modelId ? shortId(modelId) : "已登记") + " 生成 " + (count != null ? formatNumber(count) : "逐样本") + " 条反馈预测";
    }
    if (phaseGroup === "prediction" || phaseGroup === "result") {
      if (phaseGroup === "result") {
        var finalDecision = trainingTraceFirst(raw, ["decision", "candidate_status"]);
        var finalScore = trainingTraceFirst(raw, ["score", "final_score"]);
        var finalReason = trainingTraceFirst(raw, ["decision_reason", "selection_reason", "failure_reason"]);
        var finalCandidate = raw.candidate_id ? "候选 " + shortId(raw.candidate_id) : "";
        return [finalCandidate, finalDecision ? "结论：" + trainingTraceValueText(finalDecision) : "", finalScore != null ? "得分 " + formatNumber(finalScore) : "", finalReason ? "依据：" + trainingTraceValueText(finalReason) : ""].filter(Boolean).join("；") || "已形成最终结果";
      }
      var prediction = trainingTraceFirst(raw, ["predicted", "prediction", "forecast", "forecast_value", "predicted_value"]);
      return prediction != null ? "预测值 " + trainingTraceValueText(prediction) : "已生成最终预测结果";
    }
    return "已记录流程步骤";
  }
  function trainingTraceStep(raw, index, asset) {
    var source = trainingTraceIsObject(raw) ? raw : { summary: raw };
    var phase = trainingTracePhase(trainingTraceFirst(source, ["phase", "stage", "kind", "type", "name"]), index === 0 ? "input" : "feedback");
    return {
      index: Number(source.step || source.sequence || source.seq || index + 1) || index + 1,
      phase: phase,
      label: trainingTracePhaseLabels[phase] || "流程步骤",
      actor: trainingTraceActor(source, phase),
      status: trainingTraceStatus(source.status || source.state),
      summary: trainingTraceSummary(source, phase, asset),
      timestamp: trainingTraceFirst(source, ["timestamp", "created_at", "occurred_at"])
    };
  }
  function trainingTracePredictions(source, asset) {
    var candidates = [
      source && source.predictions, source && source.sample_predictions, source && source.prediction_trace,
      source && source.prediction_records, source && source.results, source && source.inference_trace,
      source && source.training_prediction && source.training_prediction.prediction_records,
      asset && asset.predictions, asset && asset.sample_predictions, asset && asset.prediction_trace,
      asset && asset.prediction_records, asset && asset.inference_trace, asset && asset.metrics && asset.metrics.prediction_preview,
      asset && asset.evaluation && asset.evaluation.prediction_preview
    ];
    var rawRows = [];
    for (var candidate of candidates) {
      if (Array.isArray(candidate)) { rawRows = candidate; break; }
      if (candidate && Array.isArray(candidate.rows)) { rawRows = candidate.rows; break; }
    }
    return rawRows.filter(function (row) { return trainingTraceIsObject(row); }).slice(0, 48).map(function (raw, index) {
      var observed = trainingTraceFirst(raw, ["observed", "observed_value", "actual", "actual_value", "observation", "target_value"]);
      var predicted = trainingTraceFirst(raw, ["predicted", "predicted_value", "prediction", "forecast", "forecast_value"]);
      var baseline = trainingTraceFirst(raw, ["baseline", "baseline_value", "persistence"]);
      var error = trainingTraceFirst(raw, ["error", "absolute_error", "residual"]);
      if (error == null && observed != null && predicted != null && Number.isFinite(Number(observed)) && Number.isFinite(Number(predicted))) { error = Number(predicted) - Number(observed); }
      return {
        index: Number(raw.sample_index || raw.index || index + 1) || index + 1,
        target: trainingTraceValueText(trainingTraceFirst(raw, ["target", "target_name", "variable"]) || "目标变量"),
        input: trainingTraceValueText(trainingTraceFirst(raw, ["input_summary", "input_reference", "input", "features", "context", "input_context"])),
        observed: observed, predicted: predicted, error: error, baseline: baseline,
        unit: trainingTraceFirst(raw, ["unit", "target_unit"]),
        partition: trainingTraceFirst(raw, ["partition"]) || (trainingTraceIsObject(raw.input_reference) ? trainingTraceFirst(raw.input_reference, ["partition"]) : null),
        step: trainingTraceFirst(raw, ["step_summary", "method_step", "method"]),
        timestamp: trainingTraceFirst(raw, ["target_timestamp", "timestamp", "time"])
      };
    });
  }
  function trainingTraceSource(asset) {
    var episode = asset && asset.episode;
    var candidates = [asset && asset.trajectory, asset && asset.training_trajectory, asset && asset.interaction_trace, asset && asset.evolution_trace, episode && episode.trajectory, episode && episode.interaction_trace];
    for (var candidate of candidates) {
      if (Array.isArray(candidate)) { return { source: candidate, steps: candidate, predictions: trainingTracePredictions({}, asset), predictionCount: null, complete: true }; }
      if (trainingTraceIsObject(candidate)) {
        var steps = trainingTraceArray(candidate);
        var predictions = trainingTracePredictions(candidate, asset);
        if (steps.length || predictions.length || candidate.input || candidate.feedback || candidate.final_prediction) {
          var predictionStage = candidate.training_prediction && typeof candidate.training_prediction === "object" ? candidate.training_prediction : {};
          var summary = asset && asset.trajectory_summary && typeof asset.trajectory_summary === "object" ? asset.trajectory_summary : {};
          var predictionCount = predictionStage.sample_count != null ? predictionStage.sample_count : candidate.sample_count != null ? candidate.sample_count : summary.sample_count != null ? summary.sample_count : summary.prediction_count;
          return { source: candidate, steps: steps, predictions: predictions, predictionCount: predictionCount, complete: candidate.complete !== false };
        }
      }
    }
    var stages = episode && episode.stages;
    if (trainingTraceIsObject(stages)) {
      var fallback = [
        { phase: "input", status: stages.strategy_input && stages.strategy_input.status, input: stages.strategy_input, summary: "冻结数据集、训练序列和优化目标" },
        { phase: "interaction", status: stages.proposal_response && stages.proposal_response.status, action: stages.proposal_response, summary: "策略模型提交候选参数方案" },
        { phase: "training", status: stages.training && stages.training.status, output: stages.training, summary: "宿主调用已登记预测模型生成训练产物" },
        { phase: "feedback", status: stages.evaluation && stages.evaluation.status, feedback: stages.evaluation, summary: "评测器在训练反馈分区计算误差和约束" },
        { phase: "optimization", status: stages.decision && stages.decision.status, decision: stages.decision, summary: "依据反馈决定候选是否进入下一轮" }
      ];
      return { source: stages, steps: fallback, predictions: trainingTracePredictions({}, asset), predictionCount: null, complete: false };
    }
    return { source: null, steps: [], predictions: trainingTracePredictions({}, asset), predictionCount: null, complete: false };
  }
  function trainingTracePredictionTable(predictions, totalHint) {
    if (!predictions.length) { return "<div class=\"training-trace-empty\">服务端尚未提供逐样本预测记录；当前仅显示阶段反馈。</div>"; }
    var visible = predictions.slice(0, 12);
    var rows = visible.map(function (row) {
      var inputCell = row.input === "未提供" ? "—" : shortId(row.input);
      if (row.step) { inputCell += "<small title=\"" + escapeHTML(row.step) + "\">步骤：" + escapeHTML(shortId(row.step)) + "</small>"; }
      return "<tr><td>" + escapeHTML(formatNumber(row.index)) + (row.timestamp != null ? "<small>" + escapeHTML(formatObservationTime(row.timestamp)) + "</small>" : "") + "</td><td title=\"" + escapeHTML(row.target) + "\">" + escapeHTML(shortId(row.target)) + "</td><td title=\"" + escapeHTML(row.input) + "\">" + inputCell + "</td><td>" + escapeHTML(formatNumber(row.observed)) + "</td><td>" + escapeHTML(formatNumber(row.predicted)) + "</td><td>" + escapeHTML(formatNumber(row.error)) + "</td><td>" + escapeHTML(formatNumber(row.baseline)) + "</td></tr>";
    }).join("");
    var totalCandidate = Number(totalHint != null ? totalHint : predictions.length);
    var total = Number.isFinite(totalCandidate) && totalCandidate >= predictions.length ? totalCandidate : predictions.length;
    var note = total > visible.length ? " · 已显示前 " + visible.length + " 条" : "";
    var partitions = [...new Set(predictions.map(function (row) { return row.partition; }).filter(Boolean))];
    var partitionNote = partitions.length ? "；分区：" + partitions.map(trainingTracePartitionText).join("、") : "";
    return "<div class=\"training-trace-table-wrap\"><table class=\"training-trace-prediction-table\"><thead><tr><th>样本</th><th>预测目标</th><th>输入摘要</th><th>观测</th><th>预测</th><th>误差</th><th>基线</th></tr></thead><tbody>" + rows + "</tbody></table><small class=\"training-trace-count\">共 " + escapeHTML(formatNumber(total)) + " 条预测" + escapeHTML(note + partitionNote) + "。</small></div>";
  }
  function trainingTrajectorySummary(asset) {
    var trace = trainingTraceSource(asset);
    var metadata = asset && asset.trajectory_summary && typeof asset.trajectory_summary === "object" ? asset.trajectory_summary : {};
    var count = trace.predictions.length || Number(metadata.shown_count || metadata.sample_count || metadata.prediction_count || 0);
    var steps = trace.steps.length || Number(metadata.completed_stage_count || metadata.stage_count || metadata.total_stage_count || 0);
    return (steps ? "轨迹 " + steps + " 阶段" : "轨迹待生成") + (count ? " / " + count + " 条预测" : "");
  }
  function renderTrainingTrajectory(asset) {
    var trace = trainingTraceSource(asset);
    if (!trace.steps.length && !trace.predictions.length) { return ""; }
    var steps = trace.steps.slice(0, 16).map(function (raw, index) {
      var step = trainingTraceStep(raw, index, asset);
      var time = step.timestamp != null ? " · " + formatDate(step.timestamp) : "";
      return "<li class=\"training-trace-step " + trainingTraceStatusClass(step.status) + "\"><div class=\"training-trace-step-head\"><span>" + escapeHTML(String(step.index).padStart(2, "0")) + "</span><strong>" + escapeHTML(step.label) + "</strong><small>" + escapeHTML(trainingTraceStatusText(step.status)) + "</small></div><p>" + escapeHTML(step.summary) + "</p><div class=\"training-trace-step-meta\"><span>参与者：" + escapeHTML(step.actor) + "</span><span>" + escapeHTML(time.replace(/^ · /, "")) + "</span></div></li>";
    }).join("");
    var omitted = trace.steps.length > 16 ? "<small class=\"training-trace-count\">另有 " + escapeHTML(formatNumber(trace.steps.length - 16)) + " 个阶段已由服务端汇总。</small>" : "";
    return "<section class=\"training-trajectory-section\"><header><div><span>单条资产的完整执行记录</span><h4>训练轨迹</h4></div><strong>" + escapeHTML(trainingTrajectorySummary(asset)) + "</strong></header><p class=\"training-trajectory-note\">按时间顺序连接输入、智能体交互、反馈、优化和最终预测；不展示隐藏思维链。</p><ol class=\"training-trace-list\">" + steps + "</ol>" + omitted + (trace.predictions.length ? "<h5 class=\"training-trace-subheading\">最终预测与观测</h5>" + trainingTracePredictionTable(trace.predictions, trace.predictionCount) : "") + "</section>";
  }
