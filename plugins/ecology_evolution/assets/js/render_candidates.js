"use strict";

  function changeDisplay(change) {
    if (change && typeof change === "object" && !Array.isArray(change)) {
      var before = change.before != null ? change.before : change.from;
      var after = change.after != null ? change.after : change.to;
      if (before != null || after != null) { return formatNumber(before) + " → " + formatNumber(after); }
    }
    return formatNumber(change);
  }
  function artifactFieldLabel(key) {
    if (parameterLabels[key]) { return parameterLabels[key]; }
    if (metricLabels[key]) { return metricLabels[key]; }
    var learnedLabels = {
      feature_policy: "特征处理规则", models: "分目标与时距子模型",
      coverage_threshold: "特征覆盖率阈值", maximum_exogenous_features: "外生特征数量上限",
      short_forward_fill_hours: "短缺口前向填充上限（小时）", long_forward_fill_hours: "长缺口前向填充上限（小时）",
      target_lag_imputation: "目标历史值插补", label_imputation: "预测标签插补", baseline_imputation: "基线值插补",
      model_task_count: "子模型任务数", solver_fallback_count: "求解器退化任务数",
      selected_exogenous_feature_count: "已选外生特征数"
    };
    if (learnedLabels[key]) { return learnedLabels[key]; }
    var suffixes = { "_bias": "拟合偏差", "_rmse": "训练均方根误差", "_n": "训练样本数", "_missing_or_nonfinite_rows": "缺失或无效训练样本数" };
    var suffix = Object.keys(suffixes).find(function (item) { return key.slice(-item.length) === item; });
    if (suffix) {
      var target = key.slice(0, -suffix.length);
      var targetHorizon = target.match(/^(.+)_([0-9]+)h$/);
      return (targetHorizon ? (targetLabels[targetHorizon[1]] || targetHorizon[1]) + "（" + targetHorizon[2] + " 小时）" : targetLabels[target] || target) + suffixes[suffix];
    }
    return "字段：" + key;
  }
  function artifactValueText(value) {
    if (value === true) { return "是"; }
    if (value === false) { return "否"; }
    if (Array.isArray(value)) {
      if (!value.length) { return "无"; }
      if (value.every(function (item) { return item == null || ["string", "number", "boolean"].indexOf(typeof item) >= 0; })) {
        return value.slice(0, 6).map(function (item) { return artifactValueText(item); }).join("、") + (value.length > 6 ? "等 " + formatNumber(value.length) + " 项" : "");
      }
      return formatNumber(value.length) + " 项结构化记录";
    }
    if (value && typeof value === "object") {
      var keys = Object.keys(value).sort();
      if (!keys.length) { return "无"; }
      return keys.slice(0, 8).map(function (key) {
        var item = value[key];
        var text = item && typeof item === "object" ? (Array.isArray(item) ? formatNumber(item.length) + " 项" : formatNumber(Object.keys(item).length) + " 项") : artifactValueText(item);
        return artifactFieldLabel(key) + "=" + text;
      }).join("；") + (keys.length > 8 ? "；其余 " + formatNumber(keys.length - 8) + " 项" : "");
    }
    return formatNumber(value);
  }
  function artifactModelStatusText(model) {
    var status = String(model && model.status || "").toLowerCase();
    if (status === "fitted") { return "已拟合"; }
    if (status === "fallback_zero_residual") { return "已退化为持续性基线"; }
    return status ? "状态：" + status : "状态未提供";
  }
  function renderArtifactModels(models) {
    var values = Array.isArray(models) ? models : [];
    if (!values.length) { return "<div class=\"artifact-model-summary\"><span class=\"empty-state\">未提供分目标与时距子模型。</span></div>"; }
    return "<div class=\"artifact-model-summary\"><div class=\"change-row\"><span>分目标与时距子模型</span><strong>" + escapeHTML(formatNumber(values.length)) + " 个模型任务</strong></div>" + values.map(function (model) {
      var features = Array.isArray(model.selected_exogenous_features) ? model.selected_exogenous_features : [];
      var coefficients = model.coefficients && typeof model.coefficients === "object" ? Object.keys(model.coefficients) : [];
      var status = artifactModelStatusText(model);
      var fallbackReason = model.fallback_reason ? "；退化原因：" + compactTechnicalText(model.fallback_reason) : "";
      return "<section class=\"artifact-model-row\"><div><strong>" + escapeHTML(targetLabels[model.target] || model.target || "预测目标") + " · " + escapeHTML(formatNumber(model.horizon_hours)) + " 小时</strong><span title=\"" + escapeHTML(features.join("、")) + "\">" + escapeHTML(status + fallbackReason) + "</span></div><div class=\"artifact-model-grid\"><div class=\"artifact-model-value\"><span>训练样本</span><strong>" + escapeHTML(formatNumber(model.training_rows)) + " 行</strong></div><div class=\"artifact-model-value\"><span>反馈样本</span><strong>" + escapeHTML(formatNumber(model.feedback_rows)) + " 行</strong></div><div class=\"artifact-model-value\"><span>外生特征</span><strong>" + escapeHTML(formatNumber(features.length)) + " 个</strong></div><div class=\"artifact-model-value\"><span>拟合系数</span><strong>" + escapeHTML(formatNumber(coefficients.length)) + " 个</strong></div></div></section>";
    }).join("") + "</div>";
  }
  function renderArtifactMapping(mapping) {
    var values = mapping && typeof mapping === "object" ? mapping : {};
    var keys = Object.keys(values).sort();
    return keys.length ? keys.map(function (key) {
      if (key === "models" && Array.isArray(values[key])) { return renderArtifactModels(values[key]); }
      return "<div class=\"change-row\"><span title=\"" + escapeHTML(key) + "\">" + escapeHTML(artifactFieldLabel(key)) + "</span><strong>" + escapeHTML(artifactValueText(values[key])) + "</strong></div>";
    }).join("") : "<span class=\"empty-state\">未提供。</span>";
  }
  function artifactForCandidate(candidateId) {
    var artifacts = state.activeRun && state.activeRun.artifacts || [];
    return artifacts.find(function (item) { return item && item.candidate_id === candidateId; }) || null;
  }
  function candidateEvidenceObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }
  function candidateEvidenceText(value, fallback, limit) {
    var text = compactTechnicalText(isBlank(value) ? fallback || "未提供" : String(value));
    var maximum = limit == null ? 180 : limit;
    return text.length > maximum ? text.slice(0, Math.max(1, maximum - 1)) + "…" : text;
  }
  function candidateEvidencePercent(value) {
    if (isBlank(value)) { return "—"; }
    var number = Number(value);
    if (!Number.isFinite(number)) { return "—"; }
    return formatNumber(Math.abs(number) <= 1 ? number * 100 : number, 1) + "%";
  }
  function candidateProposalSourceText(value) {
    return {
      remote_model: "远程策略模型",
      dsh_native_agent: "DSH 原生候选生成智能体",
      host_strategy: "宿主策略",
      host_reserved_seed: "宿主保留种子",
      host_fallback: "宿主回退提案",
      legacy_unknown: "历史提案"
    }[String(value || "")] || candidateEvidenceText(value, "提案来源未标明", 80);
  }
  function candidateKnowledgeDecisionText(value) {
    return {
      adopted: "已采用",
      not_selected: "本候选未采用",
      research_only: "仅作研究依据"
    }[String(value || "")] || gateText(value);
  }
  function candidateAlgorithmStatusText(value) {
    return {
      debug_passed: "调试通过",
      compiled: "编译通过，等待调试",
      compile_failed: "编译失败",
      debug_failed: "调试失败",
      pending: "等待编译"
    }[String(value || "")] || candidateEvidenceText(value, "等待编译", 80);
  }
  function candidateAttemptPhaseText(value) {
    return {compile: "编译", debug: "调试"}[String(value || "")] || candidateEvidenceText(value, "执行", 40);
  }
  function candidateAgentRoleText(value) {
    return {
      forecast_agent: "预测智能体",
      constraint_critic: "约束审查智能体",
      repair_agent: "修复智能体",
      host_adjudicator: "宿主裁决器",
      host_evidence_verifier: "宿主证据核验器"
    }[String(value || "")] || candidateEvidenceText(value, "协作角色", 60);
  }
  function candidateEvidenceStatusDescriptor(value) {
    var status = String(value || "pending").toLowerCase();
    if (["passed", "completed", "debug_passed", "compiled"].indexOf(status) >= 0) { return {text: "已完成", className: "is-complete"}; }
    if (["failed", "compile_failed", "debug_failed"].indexOf(status) >= 0) { return {text: "失败", className: "is-failed"}; }
    if (status === "paused") { return {text: "已暂停", className: "is-warning"}; }
    if (["warning", "partial"].indexOf(status) >= 0) { return {text: "部分完成", className: "is-warning"}; }
    if (["running", "evaluating"].indexOf(status) >= 0) { return {text: "执行中", className: "is-running"}; }
    if (["skipped", "duplicate"].indexOf(status) >= 0) { return {text: "已跳过", className: "is-skipped"}; }
    return {text: "等待", className: "is-pending"};
  }
  function candidateEvidenceStage(index, title, status, value, note) {
    var descriptor = candidateEvidenceStatusDescriptor(status);
    return "<li class=\"candidate-evidence-stage " + descriptor.className + "\"><span class=\"candidate-evidence-step\">" + escapeHTML(index) + "</span><div><small>" + escapeHTML(descriptor.text) + "</small><strong>" + escapeHTML(title) + "</strong><p title=\"" + escapeHTML(candidateEvidenceText(value, "尚未产生证据", 240)) + "\">" + escapeHTML(candidateEvidenceText(value, "尚未产生证据", 92)) + "</p>" + (note ? "<em>" + escapeHTML(candidateEvidenceText(note, "", 88)) + "</em>" : "") + "</div></li>";
  }
  function candidateResearchSources(candidate, algorithmSpec) {
    var plan = candidateEvidenceObject(candidate && candidate.model_plan);
    var source = candidateEvidenceObject(plan.plan && typeof plan.plan === "object" ? plan.plan : plan);
    var research = Array.isArray(source.research) ? source.research : [];
    var mappings = Array.isArray(algorithmSpec.knowledge_mappings) ? algorithmSpec.knowledge_mappings : [];
    var values = [];
    mappings.slice(0, 24).forEach(function (item) {
      item = candidateEvidenceObject(item);
      values.push({
        id: item.knowledge_id || item.capability_id || item.source_url,
        title: item.knowledge_id || item.capability_id || "研究资料",
        source: item.source_url || "来源地址未提供",
        decision: candidateKnowledgeDecisionText(item.decision)
      });
    });
    research.slice(0, 16).forEach(function (item, index) {
      var entry = item && typeof item === "object" ? item : {title: item};
      values.push({
        id: entry.id || entry.knowledge_id || entry.url || entry.source_url || String(index),
        title: entry.title || entry.name || entry.knowledge_id || "研究线索 " + formatNumber(index + 1),
        source: entry.url || entry.source_url || entry.source || "来源地址未提供",
        decision: "已进入研究计划"
      });
    });
    var seen = Object.create(null);
    return values.filter(function (item) {
      var key = String(item.id || "") + "|" + String(item.source || "");
      if (seen[key]) { return false; }
      seen[key] = true;
      return true;
    });
  }
  function candidateLiveSampleProgress(candidate, run) {
    candidate = candidateEvidenceObject(candidate);
    var candidateExecution = candidateEvidenceObject(candidate.execution);
    var candidateProgress = candidateEvidenceObject(candidateExecution.stage_progress);
    var runExecution = candidateEvidenceObject(run && run.execution_progress);
    var activeCandidateId = runExecution.current_candidate_id || runExecution.active_candidate_id || runExecution.candidate_id;
    var candidateId = candidate.id || candidate.candidate_id;
    var runProgress = String(activeCandidateId || "") === String(candidateId || "")
      ? candidateEvidenceObject(runExecution.stage_progress)
      : {};
    var progress = Object.assign({}, candidateProgress, runProgress);
    return Object.keys(progress).length ? progress : null;
  }
  function candidateSampleEvidence(candidate, run) {
    var metrics = candidateEvidenceObject(candidate && candidate.metrics);
    var execution = candidateEvidenceObject(candidate && candidate.execution);
    var rawTrace = candidate && candidate.inference_trace;
    function hasTrace(value) {
      return Array.isArray(value) ? value.length > 0 : value && typeof value === "object" && Object.keys(value).length > 0;
    }
    if (!hasTrace(rawTrace)) { rawTrace = execution.inference_trace; }
    if (!hasTrace(rawTrace)) { rawTrace = metrics.inference_trace; }
    var legacyRows = Array.isArray(rawTrace) ? rawTrace : [];
    if (!legacyRows.length && Array.isArray(metrics.prediction_preview)) { legacyRows = metrics.prediction_preview; }
    var trace = Array.isArray(rawTrace) ? {rows: legacyRows} : Object.assign({}, candidateEvidenceObject(rawTrace));
    if (!Array.isArray(trace.rows) && legacyRows.length) { trace.rows = legacyRows; }
    if (!trace.status && legacyRows.length) {
      var candidateStatus = String(candidate && candidate.status || "").toLowerCase();
      trace.status = candidateStatus === "failed" ? "failed" : candidateStatus === "duplicate" ? "skipped" : candidateStatus === "evaluating" ? "evaluating" : "completed";
    }
    var traceSummary = candidateEvidenceObject(trace.sample_execution);
    var metricSummary = candidateEvidenceObject(metrics.sample_execution);
    var summary = Object.assign({}, metricSummary, traceSummary);
    return {trace: trace, metrics: metrics, summary: summary, progress: candidateLiveSampleProgress(candidate, run)};
  }
  function candidateEvidenceMetric(summary, metrics, key, metricKey) {
    var value = summary[key];
    if (value == null && metricKey) { value = metrics[metricKey]; }
    return isBlank(value) ? null : value;
  }
  function candidateEvidenceStat(label, value, note, className) {
    return "<div class=\"candidate-evidence-stat " + (className || "") + "\"><span>" + escapeHTML(label) + "</span><strong>" + escapeHTML(value) + "</strong><small>" + escapeHTML(note || "") + "</small></div>";
  }
  function candidateEvidenceFailureText(item) {
    var value = candidateEvidenceObject(item);
    var failure = candidateEvidenceObject(value.failure || value);
    var category = failure.class || failure.category || failure.failure_code || failure.error_type || value.failure_code || "样本执行失败";
    var action = value.failure_action || failure.action || "保守计分并反馈下一轮";
    return candidateEvidenceText(category, "样本执行失败", 80) + " · " + candidateEvidenceText(action, "记录失败", 100);
  }
  function candidateAttemptEvidenceText(evidence) {
    var source = candidateEvidenceObject(evidence);
    var labels = {
      registered_adapters_only: "仅限已登记适配器",
      tool_count: "工具数",
      knowledge_mapping_count: "已采用研究映射",
      knowledge_not_selected_count: "未采用研究映射",
      exception_type: "异常类型",
      check_count: "预检项",
      passed: "预检通过"
    };
    var values = Object.keys(labels).filter(function (key) { return source[key] != null; }).map(function (key) {
      return labels[key] + "=" + artifactValueText(source[key]);
    });
    if (Array.isArray(source.checks)) {
      values.push("检查=" + source.checks.slice(0, 8).map(function (item) { return candidateEvidenceText(item, "", 50); }).join("、"));
    }
    return values.length ? values.join("；") : "结构化预检完成";
  }
  function candidateEvidencePreviews(trace, summary) {
    var rows = Array.isArray(trace.rows) ? trace.rows.slice(0, 12) : [];
    var actionCatalog = Array.isArray(summary.action_catalog) ? summary.action_catalog.slice(0, 8) : [];
    var roleItems = [];
    var toolItems = [];
    var failureItems = [];
    var seenRoles = Object.create(null);
    var seenTools = Object.create(null);
    var seenFailures = Object.create(null);
    function sampleLabel(row) {
      return candidateEvidenceText(row.sample_id || (row.sample_index ? "样本 " + row.sample_index : "共享执行计划"), "样本", 48);
    }
    function collectActions(source, label) {
      var decisions = Array.isArray(source.agent_decisions) ? source.agent_decisions : [];
      var tools = Array.isArray(source.tool_calls) ? source.tool_calls : [];
      decisions.slice(0, 6).forEach(function (decision) {
        decision = candidateEvidenceObject(decision);
        var key = String(decision.role || "") + "|" + String(decision.decision || "");
        if (!seenRoles[key] && roleItems.length < 4) {
          seenRoles[key] = true;
          roleItems.push({label: label, name: candidateAgentRoleText(decision.role), detail: candidateEvidenceText(decision.decision, decision.status || "已记录", 110)});
        }
      });
      tools.slice(0, 8).forEach(function (tool) {
        tool = candidateEvidenceObject(tool);
        var toolId = tool.tool_id || tool.id || tool.name || "已登记工具";
        var key = String(toolId) + "|" + String(tool.version || "");
        if (!seenTools[key] && toolItems.length < 4) {
          seenTools[key] = true;
          toolItems.push({label: label, name: candidateEvidenceText(toolId, "已登记工具", 80), detail: (tool.version ? "版本 " + candidateEvidenceText(tool.version, "", 32) + " · " : "") + candidateEvidenceText(tool.status, "已调用", 44)});
        }
      });
    }
    function collectFailure(item) {
      item = candidateEvidenceObject(item);
      var failure = candidateEvidenceObject(item.failure || item);
      var key = String(item.sample_id || "") + "|" + String(failure.class || failure.category || failure.failure_code || failure.error_type || item.failure_code || "");
      if (seenFailures[key] || failureItems.length >= 3) { return; }
      seenFailures[key] = true;
      failureItems.push({label: sampleLabel(item), detail: candidateEvidenceFailureText(item)});
    }
    rows.forEach(function (row) {
      row = candidateEvidenceObject(row);
      collectActions(row, sampleLabel(row));
      if (row.failure) { collectFailure(row); }
    });
    actionCatalog.forEach(function (action) { collectActions(candidateEvidenceObject(action), "共享动作模板"); });
    var failures = Array.isArray(summary.failure_preview) ? summary.failure_preview : [];
    failures.slice(0, 6).forEach(collectFailure);
    function renderList(items, emptyText) {
      return items.length ? "<ul>" + items.map(function (item) { return "<li><span>" + escapeHTML(item.label) + "</span><strong>" + escapeHTML(item.name || item.detail) + "</strong>" + (item.name ? "<p>" + escapeHTML(item.detail) + "</p>" : "") + "</li>"; }).join("") + "</ul>" : "<span class=\"empty-state\">" + escapeHTML(emptyText) + "</span>";
    }
    return "<div class=\"candidate-evidence-preview-grid\"><section><h4>多智能体动作</h4>" + renderList(roleItems, "暂无公开的角色动作预览。") + "</section><section><h4>工具调用</h4>" + renderList(toolItems, "暂无公开的工具调用预览。") + "</section><section><h4>失败与修复反馈</h4>" + renderList(failureItems, "当前预览中没有失败样本。") + "</section></div>";
  }
  function renderCandidateExecutionEvidence(candidate, run) {
    candidate = candidateEvidenceObject(candidate);
    var execution = candidateEvidenceObject(candidate.algorithm_execution);
    var algorithmSpec = candidateEvidenceObject(execution.algorithm_spec);
    var attempts = Array.isArray(execution.attempts) ? execution.attempts : [];
    var sample = candidateSampleEvidence(candidate, run);
    var summary = sample.summary;
    var sources = candidateResearchSources(candidate, algorithmSpec);
    var compileAttempts = attempts.filter(function (item) { return item && item.phase === "compile"; });
    var debugAttempts = attempts.filter(function (item) { return item && item.phase === "debug"; });
    var latestCompile = compileAttempts[compileAttempts.length - 1] || {};
    var latestDebug = debugAttempts[debugAttempts.length - 1] || {};
    var coverage = candidateEvidenceMetric(summary, sample.metrics, "coverage", "sample_execution_coverage");
    var attempted = candidateEvidenceMetric(summary, sample.metrics, "attempted_examples", null);
    if (attempted == null && summary.eligible_examples != null) { attempted = summary.eligible_examples; }
    var projectedTraceStatus = String(sample.trace.status || "").toLowerCase();
    if (attempted == null && ["completed", "failed", "skipped"].indexOf(projectedTraceStatus) >= 0 && sample.trace.sample_count != null) { attempted = sample.trace.sample_count; }
    var succeeded = candidateEvidenceMetric(summary, sample.metrics, "succeeded_examples", null);
    var failed = candidateEvidenceMetric(summary, sample.metrics, "failed_examples", "sample_execution_failed_examples");
    var retries = candidateEvidenceMetric(summary, sample.metrics, "retry_count", null);
    var repairs = candidateEvidenceMetric(summary, sample.metrics, "repair_count", null);
    var sourcePlan = candidateEvidenceObject(candidate.model_plan);
    var hasResearch = sources.length > 0 || Object.keys(sourcePlan).length > 0 || candidate.proposal_source;
    var sampleStatus = sample.trace.status || (Object.keys(summary).length ? "completed" : "pending");
    var candidateStatus = String(candidate.status || "").toLowerCase();
    var terminalTrace = ["completed", "failed", "skipped"].indexOf(projectedTraceStatus) >= 0;
    var terminalCandidate = ["promoted", "rejected", "failed", "duplicate"].indexOf(candidateStatus) >= 0;
    var liveProgress = sample.progress && !terminalTrace && !terminalCandidate ? sample.progress : null;
    var progressKind = String(liveProgress && liveProgress.progress_kind || "").toLowerCase();
    var runStatus = String(run && run.status || "").toLowerCase();
    var progressPaused = Boolean(liveProgress) && (runStatus === "paused" || progressKind === "drained");
    var progressCompleted = candidateFiniteNumber(liveProgress && liveProgress.completed_samples);
    var progressTotal = candidateFiniteNumber(liveProgress && liveProgress.total_samples);
    var progressSucceeded = candidateFiniteNumber(liveProgress && liveProgress.succeeded_samples);
    var progressFailed = candidateFiniteNumber(liveProgress && liveProgress.failed_samples);
    if (liveProgress) {
      attempted = progressCompleted == null ? attempted : progressCompleted;
      succeeded = progressSucceeded == null ? succeeded : progressSucceeded;
      failed = progressFailed == null && progressCompleted != null && progressSucceeded != null
        ? Math.max(0, progressCompleted - progressSucceeded)
        : progressFailed == null ? failed : progressFailed;
      sampleStatus = progressPaused ? "paused" : "running";
    }
    if (sampleStatus === "completed" && summary.coverage_pass === false) { sampleStatus = "warning"; }
    var duplicateSkipped = String(candidate.status || "").toLowerCase() === "duplicate" || sampleStatus === "skipped";
    var specificationStatus = algorithmSpec.algorithm_id ? "completed" : duplicateSkipped ? "skipped" : "pending";
    var compileStatus = latestCompile.status || (execution.status === "compile_failed" ? "failed" : algorithmSpec.algorithm_id ? "passed" : duplicateSkipped ? "skipped" : "pending");
    var debugStatus = latestDebug.status || (execution.status === "debug_passed" ? "passed" : execution.status === "debug_failed" ? "failed" : duplicateSkipped ? "skipped" : "pending");
    var progressCountText = progressCompleted != null && progressTotal != null
      ? "已完成 " + formatNumber(progressCompleted) + " / " + formatNumber(progressTotal) + " 个样本"
      : progressCompleted != null ? "已完成 " + formatNumber(progressCompleted) + " 个样本" : "等待首个样本回执";
    var sampleStageValue = liveProgress
      ? (progressPaused ? (progressKind === "drained" ? "已暂停且请求已排空；" : "已暂停，正在排空在途请求；") : "") + progressCountText
      : coverage != null
        ? "覆盖率 " + candidateEvidencePercent(coverage)
      : sampleStatus === "skipped"
        ? "重复候选，未重复执行"
        : sampleStatus === "failed"
          ? "执行失败，未形成覆盖率"
          : sampleStatus === "completed"
            ? attempted == null ? "已完成，旧运行未记录覆盖率" : "已执行 " + formatNumber(attempted) + " 个样本；旧运行未记录覆盖率"
            : ["running", "evaluating"].indexOf(sampleStatus) >= 0
              ? "逐样本执行中"
              : "等待逐样本执行";
    var remainingSamples = candidateFiniteNumber(liveProgress && liveProgress.remaining_samples);
    if (remainingSamples == null && progressCompleted != null && progressTotal != null) {
      remainingSamples = Math.max(0, progressTotal - progressCompleted);
    }
    var remainingBatches = candidateFiniteNumber(liveProgress && liveProgress.remaining_batches);
    var remainingParts = [];
    if (remainingSamples != null) { remainingParts.push("剩余 " + formatNumber(remainingSamples) + " 个样本"); }
    if (remainingBatches != null) { remainingParts.push(formatNumber(remainingBatches) + " 个微批"); }
    var sampleStageNote = liveProgress
      ? (progressPaused ? "恢复后继续" : "训练反馈分区") + (remainingParts.length ? " · " + remainingParts.join("、") : "")
      : attempted == null ? "训练反馈分区" : formatNumber(attempted) + " 个可评测样本";
    var stages = [
      candidateEvidenceStage("01", "研究证据", hasResearch ? "completed" : "pending", sources.length ? formatNumber(sources.length) + " 条来源映射" : candidateProposalSourceText(candidate.proposal_source), "结构化资料与历史失败进入候选生成"),
      candidateEvidenceStage("02", "算法规范", specificationStatus, algorithmSpec.algorithm_id || (duplicateSkipped ? "重复参数，沿用已有候选证据" : "等待生成 AlgorithmSpec"), algorithmSpec.tool_ids ? formatNumber(algorithmSpec.tool_ids.length) + " 个已登记工具" : "只允许宿主登记能力"),
      candidateEvidenceStage("03", "编译", compileStatus, compileStatus === "failed" ? latestCompile.public_error || latestCompile.failure_code : compileStatus === "skipped" ? "重复候选，未重复编译" : algorithmSpec.adapter_id || "等待能力绑定", formatNumber(compileAttempts.length) + " 次编译尝试"),
      candidateEvidenceStage("04", "调试", debugStatus, debugStatus === "failed" ? latestDebug.public_error || latestDebug.failure_code : debugStatus === "skipped" ? "重复候选，未重复调试" : execution.training_authorized ? "预检通过，允许真实训练" : "等待预检通过", formatNumber(debugAttempts.length) + " 次调试尝试"),
      candidateEvidenceStage("05", "真实样本反馈", sampleStatus, sampleStageValue, sampleStageNote)
    ].join("");
    var algorithmName = algorithmSpec.algorithm_id ? algorithmSpec.algorithm_id + (algorithmSpec.algorithm_version ? " @ " + algorithmSpec.algorithm_version : "") : duplicateSkipped ? "重复候选，未重复编译" : candidateAlgorithmStatusText(execution.status);
    var algorithmDetails = [
      ["算法来源", candidateProposalSourceText(candidate.proposal_source)],
      ["已编译算法", algorithmName],
      ["预测适配器", algorithmSpec.adapter_id ? algorithmSpec.adapter_id + (algorithmSpec.adapter_version ? " @ " + algorithmSpec.adapter_version : "") : "等待绑定"],
      ["评测器", algorithmSpec.evaluator_id ? algorithmSpec.evaluator_id + (algorithmSpec.evaluator_version ? " @ " + algorithmSpec.evaluator_version : "") : sample.trace.evaluator_id || "等待绑定"],
      ["搜索策略", algorithmSpec.strategy_id || "等待绑定"],
      ["规范校验值", shortId(execution.algorithm_spec_digest || algorithmSpec.spec_digest || "未生成")]
    ];
    var coverageThreshold = summary.minimum_coverage == null ? "门槛未提供" : "最低 " + candidateEvidencePercent(summary.minimum_coverage);
    var stats = [
      candidateEvidenceStat("样本覆盖率", candidateEvidencePercent(coverage), coverageThreshold, summary.coverage_pass === false ? "is-warning" : coverage != null ? "is-positive" : "is-pending"),
      candidateEvidenceStat("成功", succeeded == null ? "—" : formatNumber(succeeded), liveProgress && progressTotal != null ? "已完成 " + formatNumber(attempted) + " / " + formatNumber(progressTotal) : attempted == null ? "逐样本预测" : "共尝试 " + formatNumber(attempted), succeeded == null ? "is-pending" : "is-positive"),
      candidateEvidenceStat("失败", failed == null ? "—" : formatNumber(failed), "失败样本使用保守计分", failed == null ? "is-pending" : Number(failed) > 0 ? "is-warning" : "is-positive"),
      candidateEvidenceStat("重试", retries == null ? "—" : formatNumber(retries), "瞬时错误按预算退避重试", retries == null ? "is-pending" : Number(retries) > 0 ? "is-running" : ""),
      candidateEvidenceStat("修复", repairs == null ? "—" : formatNumber(repairs), "越界预测采用有界修复", repairs == null ? "is-pending" : Number(repairs) > 0 ? "is-warning" : "")
    ].join("");
    var shownAttempts = attempts.slice(Math.max(0, attempts.length - 6));
    var attemptHtml = shownAttempts.length ? "<ol class=\"candidate-attempt-list\">" + shownAttempts.map(function (attempt) {
      attempt = candidateEvidenceObject(attempt);
      var status = candidateEvidenceStatusDescriptor(attempt.status);
      var detail = attempt.public_error || attempt.failure_code || candidateAttemptEvidenceText(attempt.evidence);
      return "<li class=\"" + status.className + "\"><span>" + escapeHTML(candidateAttemptPhaseText(attempt.phase) + " #" + formatNumber(attempt.attempt)) + "</span><strong>" + escapeHTML(status.text) + "</strong><p>" + escapeHTML(candidateEvidenceText(detail, "未提供公开详情", 150)) + "</p></li>";
    }).join("") + "</ol>" : "<span class=\"empty-state\">尚未记录编译或调试尝试。</span>";
    var sourceHtml = sources.length ? "<ul class=\"candidate-source-list\">" + sources.slice(0, 4).map(function (item) {
      return "<li><strong>" + escapeHTML(candidateEvidenceText(item.title, "研究资料", 90)) + "</strong><span>" + escapeHTML(candidateEvidenceText(item.decision, "已记录", 60)) + "</span><p title=\"" + escapeHTML(candidateEvidenceText(item.source, "来源地址未提供", 240)) + "\">" + escapeHTML(candidateEvidenceText(item.source, "来源地址未提供", 130)) + "</p></li>";
    }).join("") + "</ul>" + (sources.length > 4 ? "<small class=\"candidate-preview-note\">仅显示前 4 条来源映射，共 " + escapeHTML(formatNumber(sources.length)) + " 条。</small>" : "") : "<span class=\"empty-state\">旧运行未提供结构化研究来源。</span>";
    return "<section class=\"detail-section candidate-section candidate-execution-evidence\"><div class=\"candidate-section-heading\"><h3>研究证据 → 算法规范 → 编译 → 调试 → 真实样本反馈</h3><span>公开结构化动作，不展示私密推理</span></div><ol class=\"candidate-evidence-flow\" aria-label=\"候选进化执行证据链\">" + stages + "</ol><div class=\"candidate-algorithm-grid\">" + algorithmDetails.map(function (item) { return "<div><span>" + escapeHTML(item[0]) + "</span><strong title=\"" + escapeHTML(candidateEvidenceText(item[1], "未提供", 240)) + "\">" + escapeHTML(candidateEvidenceText(item[1], "未提供", 120)) + "</strong></div>"; }).join("") + "</div><div class=\"candidate-evidence-stats\">" + stats + "</div><details class=\"candidate-evidence candidate-evidence-sources\"><summary>查看研究来源 <span>最多显示 4 条</span></summary>" + sourceHtml + "</details><details class=\"candidate-evidence candidate-evidence-attempts\"><summary>查看编译与调试记录 <span>显示 " + escapeHTML(formatNumber(shownAttempts.length)) + " / " + escapeHTML(formatNumber(attempts.length)) + " 次</span></summary>" + attemptHtml + "</details><details class=\"candidate-evidence candidate-evidence-sample-preview\"><summary>查看多智能体与工具证据 <span>有界预览</span></summary>" + candidateEvidencePreviews(sample.trace, summary) + "<small class=\"candidate-preview-note\">仅展示宿主公开的角色动作、工具结果和失败分类；完整逐样本记录在下方按页读取。</small></details></section>";
  }
  function selectionReasonText(value) {
    return {
      generation_champion_strictly_improved_incumbent: "本轮冠军，已严格改善当前最优方案",
      generation_best_did_not_improve_incumbent: "本轮第一，但未严格改善当前最优方案",
      cohort_changed_search_parent_only: "本轮窗口第一，仅作为下一轮搜索父方案",
      cohort_changed_batch_champion: "历史语义：本轮窗口冠军，跨窗口未比较",
      cohort_digest_unavailable: "缺少可验证窗口摘要，未作正式晋升",
      lower_stable_rank_than_generation_best: "通过门禁，但轮内排名较低",
      scientific_gate_failed: "未通过固定科学门禁",
      judge_rejected: "科学门禁通过，独立评审未接受",
      judge_unavailable: "独立评审不可用，未作正式晋升",
      execution_failed: "训练或评测失败",
      duplicate: "参数重复，未重复评测"
    }[String(value || "")] || String(value || "等待轮末统一选择");
  }
  function candidateFiniteNumber(value) {
    var number = Number(value);
    return isBlank(value) || !Number.isFinite(number) ? null : number;
  }
  function candidateScoreValue(candidate) {
    if (!candidate) { return null; }
    var metrics = candidate.metrics && typeof candidate.metrics === "object" ? candidate.metrics : {};
    return candidateFiniteNumber(candidate.score != null ? candidate.score : metrics.score);
  }
  function candidateMetricValue(candidate, key) {
    var metrics = candidate && candidate.metrics && typeof candidate.metrics === "object" ? candidate.metrics : {};
    return candidateFiniteNumber(metrics[key]);
  }
  function candidateGateDescriptor(value) {
    var missing = isBlank(value) || ["pending", "waiting", "not_started", "unavailable", "restricted", "evaluating", "spawned"].indexOf(String(value).toLowerCase()) >= 0;
    var pass = value === true || ["true", "pass", "passed", "approved", "accepted", "promoted", "retained"].indexOf(String(value).toLowerCase()) >= 0;
    var fail = value === false || ["false", "fail", "failed", "denied", "rejected"].indexOf(String(value).toLowerCase()) >= 0;
    return {
      text: missing ? "待评测" : gateText(value),
      className: pass ? "pill-green" : fail ? "pill-red" : missing ? "pill-amber" : "pill-neutral",
      state: pass ? "pass" : fail ? "fail" : missing ? "pending" : "neutral"
    };
  }
  function candidateJudgeDescriptor(candidate, run) {
    var metrics = candidate && candidate.metrics && typeof candidate.metrics === "object" ? candidate.metrics : {};
    var status = String(metrics.judge_status || "").toLowerCase();
    var retryWait = run && run.execution_progress && run.execution_progress.retry_wait;
    var retrying = String(run && run.status || "").toLowerCase() === "running"
      && retryWait && retryWait.waiting === true
      && String(retryWait.error_code || "").toLowerCase() === "generation_judges_unavailable";
    if (["unavailable", "temporarily_unavailable"].indexOf(status) >= 0) {
      return { text: retrying ? "自动重试中" : "评审不可用", className: "pill-amber", state: "pending" };
    }
    if (["not_started", "pending", "waiting", "evaluating", "running"].indexOf(status) >= 0) {
      return candidateGateDescriptor(null);
    }
    if (status === "failed") {
      return { text: "评审失败", className: "pill-red", state: "fail" };
    }
    return candidateGateDescriptor(metrics.judge_accepted);
  }
  function candidateScientificPass(candidate) {
    if (!candidate || typeof candidate !== "object") { return null; }
    var metrics = candidate.metrics && typeof candidate.metrics === "object" ? candidate.metrics : {};
    if (metrics.scientific_pass === true || metrics.scientific_pass === false) { return metrics.scientific_pass; }
    // Historical projections stored only the combined result.  A combined
    // pass proves both gates passed; a combined failure is ambiguous unless
    // the judge result independently identifies the scientific outcome.
    if (candidate.passed === true) { return true; }
    var judgeAccepted = metrics.judge_accepted;
    var judgeCompleted = String(metrics.judge_status || "").toLowerCase() === "completed";
    if (judgeCompleted && String(metrics.judge_model_id || "").toLowerCase() === "rule_judge@1" && (judgeAccepted === true || judgeAccepted === false)) {
      return judgeAccepted;
    }
    if (candidate.passed === false && judgeCompleted && judgeAccepted === true) { return false; }
    var selectionReason = String(candidate.selection_reason || "").toLowerCase();
    if (selectionReason === "scientific_gate_failed") { return false; }
    if (selectionReason === "judge_rejected") { return true; }
    var constraints = candidateFiniteNumber(metrics.constraint_violations);
    var skillScore = candidateFiniteNumber(metrics.skill_score);
    var hasNoRegression = metrics.per_target_no_regression === true || metrics.per_target_no_regression === false;
    if (constraints != null && skillScore != null && hasNoRegression) {
      return constraints === 0 && metrics.per_target_no_regression === true && skillScore > 1e-9;
    }
    return null;
  }
  function candidateOutcome(candidate, run) {
    if (run && run.best_candidate_id && candidate.id === run.best_candidate_id) { return { text: "当前保留", className: "pill-green" }; }
    if (run && run.best_observed_candidate_id && candidate.id === run.best_observed_candidate_id) { return { text: "原始最高观测", className: "pill-blue" }; }
    if (String(candidate.status || "").toLowerCase() === "failed") { return { text: "执行失败", className: "pill-red" }; }
    if (String(candidate.status || "").toLowerCase() === "rejected") { return { text: "未保留", className: "pill-red" }; }
    if (String(candidate.status || "").toLowerCase() === "duplicate") { return { text: "重复跳过", className: "pill-neutral" }; }
    return { text: candidateStatusText(candidate.status), className: candidateStatusClass(candidate.status) };
  }
  function candidateParentScore(candidate, candidates) {
    if (!candidate || !Array.isArray(candidates)) { return null; }
    var parent = candidates.find(function (item) { return item.id === candidate.parent_id; });
    return candidateScoreValue(parent);
  }
  function candidateEvaluationCohortDigest(run, candidateId) {
    if (!run || !candidateId || !Array.isArray(run.trajectory)) { return ""; }
    var point = run.trajectory.find(function (item) { return String(item && item.candidate_id || "") === String(candidateId); });
    return String(point && point.evaluation_cohort_digest || "");
  }
  function candidateDelta(candidate, candidates, run) {
    var explicit = candidateMetricValue(candidate, "improvement");
    if (explicit != null) { return { value: explicit, label: "相对基线改进" }; }
    var score = candidateScoreValue(candidate);
    var parentScore = candidateParentScore(candidate, candidates);
    var candidateCohort = candidateEvaluationCohortDigest(run, candidate && candidate.id);
    var parentCohort = candidateEvaluationCohortDigest(run, candidate && candidate.parent_id);
    if (candidateCohort && parentCohort && candidateCohort !== parentCohort) {
      return { value: null, label: "跨窗口不可直接比较" };
    }
    return score != null && parentScore != null ? { value: score - parentScore, label: "相对父方案" } : { value: null, label: "相对基线改进" };
  }
  function candidateRounds(candidates) {
    return (Array.isArray(candidates) ? candidates : []).map(function (candidate) { return Number(candidate.generation || 0); }).filter(function (value, index, values) { return values.indexOf(value) === index; }).sort(function (left, right) { return left - right; });
  }
  function candidateListId(value) {
    var text = String(value || "");
    var prefix = "candidate:";
    return text.indexOf(prefix) === 0 ? "#" + text.slice(prefix.length, prefix.length + 8) : shortId(value);
  }
  function candidateSummaryCard(label, value, note, className) {
    return "<div class=\"candidate-summary-card " + (className || "") + "\"><span>" + escapeHTML(label) + "</span><strong>" + escapeHTML(value) + "</strong><small>" + escapeHTML(note || "") + "</small></div>";
  }
  function renderCandidateOverview(run, candidates) {
    var values = Array.isArray(candidates) ? candidates : [];
    var evaluated = values.filter(function (candidate) { return candidateScoreValue(candidate) != null; });
    var pending = values.filter(function (candidate) { return ["pending", "spawned", "evaluating"].indexOf(String(candidate.status || "").toLowerCase()) >= 0; });
    var retained = run && run.best_candidate_id ? values.find(function (candidate) { return candidate.id === run.best_candidate_id; }) : null;
    var observed = run && run.best_observed_candidate_id ? values.find(function (candidate) { return candidate.id === run.best_observed_candidate_id; }) : null;
    if (!observed) {
      observed = values.filter(function (candidate) { return candidateScoreValue(candidate) != null; }).sort(function (left, right) { return candidateScoreValue(right) - candidateScoreValue(left); })[0] || null;
    }
    var rounds = candidateRounds(values);
    var retainedScore = runRetainedScore(run);
    var rawObservedScore = runRawBestObservedScore(run);
    var retainedScoreText = retainedScore == null ? "—" : formatNumber(retainedScore, 3);
    var observedScoreText = rawObservedScore == null ? "—" : formatNumber(rawObservedScore, 3);
    var roundText = rounds.length ? formatNumber(rounds.length) + " 轮" : "尚未开始";
    var summary = [
      candidateSummaryCard("当前保留", retained ? shortId(retained.id) : "尚未产生", retained ? "训练反馈搜索保留" : "正式验证未开展", retained ? "is-winner" : "is-pending"),
      candidateSummaryCard("当前保留得分", retainedScoreText, retainedScore == null ? "尚未记录晋升得分" : "实际晋升序列", retainedScore != null ? "has-value" : "is-pending"),
      candidateSummaryCard("原始最高观测（跨窗口不可直接比较）", observedScoreText, observed ? shortId(observed.id) : "暂无已完成评测", rawObservedScore != null ? "has-value" : "is-pending"),
      candidateSummaryCard("评测进度", formatNumber(evaluated.length) + " / " + formatNumber(values.length), pending.length ? formatNumber(pending.length) + " 个等待反馈" : roundText, pending.length ? "is-pending" : "has-value")
    ];
    return "<div class=\"candidate-summary-strip\" id=\"candidate-summary-strip\" aria-label=\"候选评测概览\">" + summary.join("") + "</div>";
  }
  function updateCandidateOverview(run, candidates) {
    var area = document.querySelector(".candidate-table-area");
    if (!area) { return; }
    var node = document.getElementById("candidate-summary-strip");
    if (!node) {
      node = document.createElement("div");
      node.id = "candidate-summary-strip";
      node.className = "candidate-summary-strip";
      var tableWrap = area.querySelector(".table-wrap");
      area.insertBefore(node, tableWrap || null);
    }
    node.outerHTML = renderCandidateOverview(run, candidates);
  }
  function candidateMetricCard(label, value, note, className) {
    return "<div class=\"candidate-metric-card " + (className || "") + "\"><span>" + escapeHTML(label) + "</span><strong>" + escapeHTML(value) + "</strong>" + (note ? "<small>" + escapeHTML(note) + "</small>" : "") + "</div>";
  }
  function renderCandidateGates(candidate, run) {
    var metrics = candidate && candidate.metrics && typeof candidate.metrics === "object" ? candidate.metrics : {};
    var scientific = candidateGateDescriptor(candidateScientificPass(candidate));
    var judge = candidateJudgeDescriptor(candidate, run);
    var constraints = candidateFiniteNumber(metrics.constraint_violations);
    var constraintClass = constraints == null ? "pill-amber" : constraints === 0 ? "pill-green" : "pill-red";
    return "<div class=\"candidate-gate-strip\"><div class=\"candidate-gate-item\"><span>科学门禁</span><strong class=\"pill " + scientific.className + "\">" + escapeHTML(scientific.text) + "</strong></div><div class=\"candidate-gate-item\"><span>独立评审</span><strong class=\"pill " + judge.className + "\">" + escapeHTML(judge.text) + "</strong></div><div class=\"candidate-gate-item\"><span>约束违规</span><strong class=\"pill " + constraintClass + "\">" + escapeHTML(constraints == null ? "待评测" : formatNumber(constraints)) + "</strong></div></div>";
  }
  function renderCandidateChanges(changes) {
    var keys = Object.keys(changes || {});
    if (!keys.length) { return "<span class=\"empty-state\">未提供修改明细。</span>"; }
    return keys.map(function (key) {
      var change = changes[key];
      var before = change && typeof change === "object" && !Array.isArray(change) ? (change.before != null ? change.before : change.from) : null;
      var after = change && typeof change === "object" && !Array.isArray(change) ? (change.after != null ? change.after : change.to) : change;
      var hasDiff = before != null || (change && typeof change === "object" && !Array.isArray(change) && after != null);
      return "<div class=\"change-row candidate-change\"><span title=\"" + escapeHTML(key) + "\">" + escapeHTML(parameterLabels[key] || "参数：" + key) + "</span>" + (hasDiff ? "<strong class=\"candidate-change-before\">" + escapeHTML(formatNumber(before)) + "</strong><span class=\"candidate-change-arrow\" aria-hidden=\"true\">→</span><strong class=\"candidate-change-after\">" + escapeHTML(formatNumber(after)) + "</strong>" : "<strong class=\"candidate-change-after\">" + escapeHTML(changeDisplay(change)) + "</strong>") + "</div>";
    }).join("");
  }
  function renderCandidateTargets(targets) {
    if (!targets.length) { return "<span class=\"empty-state\">当前评测器未提供分目标结果。</span>"; }
    return targets.map(function (target) {
      var skill = candidateFiniteNumber(target.skill_score);
      var violation = candidateFiniteNumber(target.constraint_violations);
      var skillClass = skill == null ? "pill-amber" : skill >= 0 ? "pill-green" : "pill-red";
      return "<article class=\"target-result\"><div class=\"candidate-target-header\"><strong>" + escapeHTML(targetLabels[target.target] || target.target || "预测目标") + "</strong><span>" + escapeHTML(target.horizon_hours != null ? formatNumber(target.horizon_hours) + " 小时" : "时距未提供") + "</span><b class=\"pill " + skillClass + "\">技能 " + escapeHTML(skill == null ? "待评测" : signedNumber(skill, 3)) + "</b></div><div class=\"candidate-target-metrics\"><span>样本<strong>" + escapeHTML(formatNumber(target.n)) + "</strong></span><span>MAE<strong>" + escapeHTML(formatNumber(target.mae)) + "</strong></span><span>RMSE<strong>" + escapeHTML(formatNumber(target.rmse)) + "</strong></span><span>基线 RMSE<strong>" + escapeHTML(formatNumber(target.baseline_rmse)) + "</strong></span><span>违规<strong class=\"" + (violation != null && violation > 0 ? "is-warning" : "") + "\">" + escapeHTML(violation == null ? "—" : formatNumber(violation)) + "</strong></span><span>单位<strong>" + escapeHTML(unitText(target.unit)) + "</strong></span></div></article>";
    }).join("");
  }
  function candidateSampleStatusDescriptor(row) {
    var status = String(row && row.sample_status || "").toLowerCase();
    if (!status && row && row.failure_message) { status = "failed"; }
    if (!status && row && row.predicted != null && row.predicted !== "" && Number.isFinite(Number(row.predicted))) { status = "completed"; }
    if (["completed", "succeeded", "success", "accepted", "scored", "passed"].indexOf(status) >= 0) { return {text: "已完成", className: "pill-green", rowClass: "is-complete"}; }
    if (["failed", "error", "rejected", "timeout"].indexOf(status) >= 0) { return {text: "失败", className: "pill-red", rowClass: "is-failed"}; }
    if (["aborted", "cancelled"].indexOf(status) >= 0) { return {text: "已中止", className: "pill-red", rowClass: "is-failed"}; }
    if (["running", "evaluating", "in_progress", "predicting", "retrying"].indexOf(status) >= 0) { return {text: status === "retrying" ? "重试中" : "执行中", className: "pill-blue", rowClass: "is-running"}; }
    if (["skipped", "not_attempted"].indexOf(status) >= 0) { return {text: "已跳过", className: "pill-neutral", rowClass: "is-skipped"}; }
    return {text: "等待", className: "pill-amber", rowClass: "is-pending"};
  }
  function candidateSampleValue(value, digits) {
    if (value == null || value === "") { return "—"; }
    var number = Number(value);
    return Number.isFinite(number) ? formatNumber(number, digits == null ? 4 : digits) : displayText(value);
  }
  function renderCandidateSampleRows(rows, offset) {
    return (Array.isArray(rows) ? rows : []).map(function (row, index) {
      row = row && typeof row === "object" ? row : {};
      var status = candidateSampleStatusDescriptor(row);
      var targetTime = row.target_timestamp != null ? row.target_timestamp : row.timestamp;
      var originTime = row.origin_timestamp;
      var reward = Number(row.reward);
      var rewardClass = Number.isFinite(reward) ? reward > 0 ? "is-positive" : reward < 0 ? "is-negative" : "" : "is-missing";
      var sampleId = row.sample_id || "sample:" + (Number(offset || 0) + index + 1);
      var scoringFallback = row.scoring_fallback ? String(row.scoring_fallback) : "";
      var failureSource = row.failure_message || row.failure_class || (scoringFallback ? "未获得有效预测，已采用评分惩罚值" : "");
      var failure = failureSource ? candidateEvidenceText(failureSource, "样本执行失败", 100) : "";
      var attempts = Number(row.attempts);
      var retryCount = Number(row.retry_count);
      var retryDetail = Number.isFinite(retryCount) && retryCount > 0
        ? "已重试 " + formatNumber(retryCount) + " 次" + (Number.isFinite(attempts) && attempts > 0 ? "（共 " + formatNumber(attempts) + " 次尝试）" : "")
        : "";
      var executionDetail = [failure, retryDetail].filter(Boolean).join(" · ");
      var sampleMeta = [row.unit && row.unit !== "unknown" ? unitText(row.unit) : "", shortId(sampleId)].filter(Boolean).join(" · ");
      var prediction = scoringFallback
        ? "<span class=\"candidate-sample-value-stack\"><strong>" + escapeHTML(candidateSampleValue(row.predicted)) + "</strong><small>评分惩罚值</small></span>"
        : escapeHTML(candidateSampleValue(row.predicted));
      return "<tr class=\"candidate-sample-row " + status.rowClass + "\" data-sample-id=\"" + escapeHTML(sampleId) + "\"><td data-label=\"时间\"><span class=\"candidate-sample-time\"><strong>" + escapeHTML(targetTime == null ? "—" : formatObservationTime(targetTime)) + "</strong><small>" + escapeHTML(originTime == null ? "起点未提供" : "起点 " + formatObservationTime(originTime)) + "</small></span></td><td data-label=\"目标\"><span class=\"candidate-sample-target\"><strong>" + escapeHTML(targetLabels[row.target] || row.target || "预测目标") + "</strong><small title=\"" + escapeHTML(sampleId) + "\">" + escapeHTML(sampleMeta) + "</small></span></td><td data-label=\"时距\">" + escapeHTML(row.horizon_hours == null ? "—" : formatNumber(row.horizon_hours) + " 小时") + "</td><td data-label=\"真实值\" class=\"candidate-sample-number\">" + escapeHTML(candidateSampleValue(row.observed)) + "</td><td data-label=\"预测值\" class=\"candidate-sample-number candidate-sample-prediction\">" + prediction + "</td><td data-label=\"Reward（原始单位）\" class=\"candidate-sample-number candidate-sample-reward " + rewardClass + "\" title=\"相对持续性基线的绝对误差改善，正值更好；保留目标原始单位，只能在同一目标内比较\">" + escapeHTML(candidateSampleValue(row.reward)) + "</td><td data-label=\"状态\"><span class=\"candidate-sample-status\"><strong class=\"pill " + status.className + "\">" + escapeHTML(status.text) + "</strong>" + (executionDetail ? "<small title=\"" + escapeHTML(executionDetail) + "\">" + escapeHTML(executionDetail) + "</small>" : "") + "</span></td></tr>";
    }).join("");
  }
  function renderCandidateSamples() {
    var table = $("#candidate-samples-table");
    var candidateNode = $("#candidate-samples-candidate");
    var statusNode = $("#candidate-samples-status");
    var countNode = $("#candidate-samples-count");
    var pageNode = $("#candidate-samples-page");
    var previous = $("#candidate-samples-previous");
    var next = $("#candidate-samples-next");
    var retry = $("#candidate-samples-retry");
    var section = $("#candidate-samples");
    var followControl = $("#candidate-follow-active");
    if (!table || !candidateNode || !statusNode || !countNode || !pageNode || !previous || !next || !retry) { return; }
    var run = state.activeRun;
    var candidate = selectedCandidateForSamples();
    if (followControl) {
      followControl.checked = !state.candidateSelectionPinned;
      followControl.disabled = !run || !Array.isArray(run.candidates) || !run.candidates.length;
    }
    if (!run || !candidate) {
      candidateNode.textContent = "请选择候选方案";
      statusNode.className = "pill pill-neutral";
      statusNode.textContent = "等待数据";
      countNode.textContent = "0 条";
      pageNode.textContent = "0–0，共 0 条";
      previous.disabled = true; next.disabled = true; retry.hidden = true;
      if (section && typeof section.setAttribute === "function") { section.setAttribute("aria-busy", "false"); }
      table.innerHTML = "<tr><td colspan=\"7\" class=\"empty-state\">选择候选方案后显示样本结果。</td></tr>";
      return;
    }
    var candidateId = candidate.id || candidate.candidate_id;
    var page = candidateSampleSelectionMatches(run.id, candidateId) ? state.candidateSamplePage : null;
    var rows = page && Array.isArray(page.rows) ? page.rows : [];
    var offset = page ? Number(page.offset || 0) : 0;
    var total = page ? Math.max(Number(page.total || 0), offset + rows.length) : 0;
    var loading = state.candidateSampleLoading;
    var refreshing = state.candidateSampleRefreshing;
    var error = state.candidateSampleError;
    var unavailable = state.candidateSampleUnavailable;
    var permissionDenied = state.candidateSamplePermissionDenied;
    var pageStatus = String(page && page.status || "").toLowerCase();
    var expectedCount = Number(page && page.expected_count);
    var runStatus = String(run.status || "").toLowerCase();
    var candidateStatus = String(candidate.status || "").toLowerCase();
    var running = candidateSamplesAreLive(run, candidate);
    var paused = runStatus === "paused";
    var stoppedWithFailure = ["failed", "cancelled", "quarantined"].indexOf(runStatus) >= 0 || ["failed", "cancelled"].indexOf(candidateStatus) >= 0;
    if (section && typeof section.setAttribute === "function") { section.setAttribute("aria-busy", String(loading || refreshing)); }
    candidateNode.textContent = shortId(candidateId);
    candidateNode.title = candidateId;
    countNode.textContent = Number.isFinite(expectedCount) && expectedCount >= total
      ? formatNumber(total) + " / " + formatNumber(expectedCount) + " 条"
      : total > rows.length ? formatNumber(rows.length) + " / " + formatNumber(total) + " 条" : formatNumber(total) + " 条";
    pageNode.textContent = rows.length ? formatNumber(offset + 1) + "–" + formatNumber(offset + rows.length) + "，共 " + formatNumber(total) + " 条" : "0–0，共 " + formatNumber(total) + " 条";
    previous.disabled = loading || refreshing || offset <= 0;
    next.disabled = loading || refreshing || !page || page.has_more !== true;
    retry.hidden = !error;
    if (error) {
      statusNode.className = "pill pill-red";
      statusNode.textContent = "刷新失败";
    } else if (loading || refreshing) {
      statusNode.className = "pill pill-blue";
      statusNode.textContent = refreshing ? "刷新中" : "读取中";
    } else if (permissionDenied) {
      statusNode.className = "pill pill-red";
      statusNode.textContent = "权限受限";
    } else if (unavailable) {
      statusNode.className = "pill pill-amber";
      statusNode.textContent = "摘要预览";
    } else if (pageStatus === "aborted") {
      statusNode.className = "pill pill-red";
      statusNode.textContent = "执行中止";
    } else if (page && page.complete === true) {
      statusNode.className = "pill pill-green";
      statusNode.textContent = "已同步";
    } else if (running && (!page || page.complete !== true)) {
      statusNode.className = "pill pill-blue";
      statusNode.textContent = "实时更新";
    } else if (stoppedWithFailure) {
      statusNode.className = "pill pill-red";
      statusNode.textContent = "执行停止";
    } else if (paused) {
      statusNode.className = "pill pill-amber";
      statusNode.textContent = "已暂停";
    } else if (!page) {
      statusNode.className = "pill pill-neutral";
      statusNode.textContent = "无记录";
    } else {
      statusNode.className = "pill pill-amber";
      statusNode.textContent = "部分记录";
    }
    if (rows.length) {
      table.innerHTML = renderCandidateSampleRows(rows, offset);
      return;
    }
    if (loading) {
      table.innerHTML = "<tr><td colspan=\"7\" class=\"empty-state candidate-samples-loading\">正在读取样本结果…</td></tr>";
    } else if (error) {
      table.innerHTML = "<tr><td colspan=\"7\" class=\"empty-state candidate-samples-error\">样本结果读取失败：" + escapeHTML(error) + "</td></tr>";
    } else if (permissionDenied) {
      table.innerHTML = "<tr><td colspan=\"7\" class=\"empty-state candidate-samples-error\">当前 DSH 会话未授予逐样本结果读取能力。</td></tr>";
    } else if (unavailable) {
      table.innerHTML = "<tr><td colspan=\"7\" class=\"empty-state\">服务端尚未提供逐样本分页记录，当前候选也没有可用的脱敏预览。</td></tr>";
    } else if (pageStatus === "aborted") {
      table.innerHTML = "<tr><td colspan=\"7\" class=\"empty-state candidate-samples-error\">候选执行已中止，没有产生可展示的逐样本结果。</td></tr>";
    } else if (stoppedWithFailure) {
      table.innerHTML = "<tr><td colspan=\"7\" class=\"empty-state candidate-samples-error\">运行或候选已停止，没有产生可展示的逐样本结果。</td></tr>";
    } else if (paused) {
      table.innerHTML = "<tr><td colspan=\"7\" class=\"empty-state\">运行已暂停；恢复后将继续读取逐样本结果。</td></tr>";
    } else if (running) {
      table.innerHTML = "<tr><td colspan=\"7\" class=\"empty-state\">候选正在执行，第一条样本结果完成后将自动显示。</td></tr>";
    } else {
      table.innerHTML = "<tr><td colspan=\"7\" class=\"empty-state\">该候选未记录可展示的逐样本结果。</td></tr>";
    }
  }
  function renderCandidateDetail(candidate, run, candidates) {
    var node = $("#candidate-detail");
    if (!candidate) { node.innerHTML = "<div class=\"empty-state\">选择一个候选方案查看评测详情。</div>"; return; }
    run = run || state.activeRun || {};
    candidates = Array.isArray(candidates) ? candidates : run.candidates || [];
    var metricOrder = ["score", "passed", "skill_score", "normalized_rmse", "baseline_normalized_rmse", "improvement", "objective_profile", "objective_target_weights", "objective_horizon_weighting", "mae", "rmse", "mean_target_mae_unscaled", "mean_target_rmse_unscaled", "per_target_no_regression", "raw_units_comparable_across_targets", "water_balance_error", "n", "missing_or_nonfinite_rows", "constraint_violations", "non_negative_state", "scientific_pass", "judge_model_id", "judge_accepted", "judge_guidance", "evaluation_scope", "selection_scope", "formal_validation_status", "dataset_digest", "split_manifest_digest_sha256", "causal_interpretation"];
    var candidateMetrics = candidate.metrics || {};
    var scalarMetricKeys = Object.keys(candidateMetrics).filter(function (key) { return candidateMetrics[key] == null || ["string", "number", "boolean"].indexOf(typeof candidateMetrics[key]) >= 0; });
    var metrics = metricOrder.filter(function (key) { return candidateMetrics[key] != null; }).concat(scalarMetricKeys.filter(function (key) { return metricOrder.indexOf(key) < 0; }).sort());
    var targets = candidate.metrics && Array.isArray(candidate.metrics.targets) ? candidate.metrics.targets : [];
    var predictionRows = candidate.metrics && Array.isArray(candidate.metrics.prediction_preview) ? candidate.metrics.prediction_preview : [];
    var promotion = candidate.promotion || {};
    var artifact = artifactForCandidate(candidate.id);
    var score = candidateScoreValue(candidate);
    var delta = candidateDelta(candidate, candidates, run);
    var skill = candidateMetricValue(candidate, "skill_score");
    var rmse = candidateMetricValue(candidate, "rmse");
    var sampleCount = candidateMetricValue(candidate, "n");
    var constraints = candidateMetricValue(candidate, "constraint_violations");
    var outcome = candidateOutcome(candidate, run);
    var statusClass = candidateStatusClass(candidate.status);
    var scoreClass = score == null ? "is-pending" : score >= 0 ? "is-positive" : "is-negative";
    var keyMetricKeys = ["score", "skill_score", "improvement", "rmse", "normalized_rmse", "n", "constraint_violations", "water_balance_error"];
    var judgeCompleted = String(candidateMetrics.judge_status || "").toLowerCase() === "completed";
    var secondaryMetrics = metrics.filter(function (key) { return key !== "targets" && key !== "prediction_preview" && keyMetricKeys.indexOf(key) < 0 && (key !== "judge_accepted" || judgeCompleted); });
    var predictionLimit = 8;
    var shownPredictionRows = predictionRows.slice(0, predictionLimit);
    var predictionTable = predictionRows.length ? "<div class=\"table-wrap detail-table-wrap candidate-preview-table\" tabindex=\"0\" aria-label=\"预测效果预览，可横向滚动\"><table class=\"prediction-table\"><thead><tr><th>预测起点</th><th>目标时间</th><th>时距</th><th>目标</th><th>观测值</th><th>候选预测</th><th>基线预测</th><th>单位</th></tr></thead><tbody>" + shownPredictionRows.map(function (row) { return "<tr><td data-label=\"预测起点\">" + escapeHTML(row.origin_timestamp == null ? "—" : formatObservationTime(row.origin_timestamp)) + "</td><td data-label=\"目标时间\">" + escapeHTML(formatObservationTime(row.target_timestamp != null ? row.target_timestamp : row.timestamp)) + "</td><td data-label=\"时距\">" + escapeHTML(row.horizon_hours == null ? "—" : formatNumber(row.horizon_hours) + " 小时") + "</td><td data-label=\"目标\">" + escapeHTML(targetLabels[row.target] || row.target || "预测目标") + "</td><td data-label=\"观测值\">" + escapeHTML(formatNumber(row.observed)) + "</td><td data-label=\"候选预测\"><strong>" + escapeHTML(formatNumber(row.predicted)) + "</strong></td><td data-label=\"基线预测\">" + escapeHTML(formatNumber(row.baseline)) + "</td><td data-label=\"单位\">" + escapeHTML(unitText(row.unit)) + "</td></tr>"; }).join("") + "</tbody></table></div>" + (predictionRows.length > predictionLimit ? "<small class=\"candidate-preview-note\">已显示前 " + escapeHTML(formatNumber(predictionLimit)) + " 行，共 " + escapeHTML(formatNumber(predictionRows.length)) + " 行；完整记录可在下方逐样本结果中分页查看。</small>" : "") : "<span class=\"empty-state\">当前评测器未提供预测效果预览。</span>";
    var artifactSection = artifact ? "<div class=\"artifact-summary\"><div class=\"detail-grid\"><div class=\"detail-value\"><span>训练模型</span><strong title=\"" + escapeHTML(artifact.model_id || "") + "\">" + escapeHTML(predictionModelReferenceLabel(artifact.model_id)) + "</strong></div><div class=\"detail-value\"><span>训练分区</span><strong>" + escapeHTML(partitionText(artifact.training_partition)) + "</strong></div><div class=\"detail-value\"><span>训练样本数</span><strong>" + escapeHTML(formatNumber(artifact.training_rows)) + "</strong></div><div class=\"detail-value\"><span>产物校验值</span><strong title=\"" + escapeHTML(artifact.artifact_digest || "") + "\">" + escapeHTML(shortId(artifact.artifact_digest || "未提供")) + "</strong></div></div><h4>拟合参数</h4><div class=\"change-list\">" + renderArtifactMapping(artifact.learned_parameters) + "</div><h4>训练指标</h4><div class=\"change-list\">" + renderArtifactMapping(artifact.metrics) + "</div></div>" : "<span class=\"empty-state\">尚未记录该候选的训练产物。</span>";
    var failureReason = candidate.failure_reason || (candidate.status === "failed" ? "候选在训练或评测阶段失败，服务端未提供公开原因。" : "");
    var failureSection = failureReason ? "<section class=\"detail-section failure-detail candidate-section\"><h3>执行异常</h3><div class=\"failure-message\"><strong>" + escapeHTML(candidate.failed_stage ? "阶段：" + (evolutionStageLabels[candidate.failed_stage] || candidate.failed_stage) : "候选执行失败") + "</strong><p>" + escapeHTML(failureReason) + "</p><span>当前版本没有阶段级重试按钮；请保留同一任务证据并新建运行重试。</span></div></section>" : "";
    var header = "<header class=\"candidate-detail-header " + statusClass + "\"><div class=\"candidate-detail-heading\"><span class=\"candidate-detail-kicker\">候选方案 · 第 " + escapeHTML(candidate.generation || "—") + " 轮 / 槽位 " + escapeHTML(Number(candidate.slot_index || 0) + 1) + "</span><h2 title=\"" + escapeHTML(candidate.id) + "\">" + escapeHTML(shortId(candidate.id)) + "</h2><code title=\"" + escapeHTML(candidate.id) + "\">" + escapeHTML(candidate.id) + "</code><p>父方案：<span title=\"" + escapeHTML(candidate.parent_id || "") + "\">" + escapeHTML(shortId(candidate.parent_id || "当前基线")) + "</span> · 轮内排名 " + escapeHTML(candidate.generation_rank == null ? "—" : candidate.generation_rank) + "</p></div><div class=\"candidate-detail-outcome\"><span class=\"pill " + outcome.className + "\">" + escapeHTML(outcome.text) + "</span><strong class=\"candidate-detail-score " + scoreClass + "\">" + escapeHTML(score == null ? "—" : formatNumber(score, 3)) + "</strong><small>综合得分</small></div></header>";
    var keyMetrics = [
      candidateMetricCard("综合得分", score == null ? "—" : formatNumber(score, 3), outcome.text, "is-score " + scoreClass),
      candidateMetricCard(delta.label, delta.value == null ? "—" : signedNumber(delta.value, 3), "相对父方案或固定基线", delta.value == null ? "is-pending" : delta.value >= 0 ? "is-positive" : "is-negative"),
      candidateMetricCard("技能得分", skill == null ? "—" : signedNumber(skill, 3), "越高越好", skill == null ? "is-pending" : skill >= 0 ? "is-positive" : "is-negative"),
      candidateMetricCard("RMSE", rmse == null ? "—" : formatNumber(rmse, 4), sampleCount == null ? "反馈样本待提供" : formatNumber(sampleCount) + " 个样本", rmse == null ? "is-pending" : "")
    ].join("");
    var decisionReason = promotion.reason || candidate.selection_reason || "";
    var rationale = compactTechnicalText(candidate.rationale || "未提供");
    var details = [
      "<section class=\"detail-section candidate-section candidate-gate-section\"><h3>评测门禁</h3>" + renderCandidateGates(candidate, run) + "</section>",
      "<section class=\"detail-section candidate-section candidate-changes-section\"><div class=\"candidate-section-heading\"><h3>本轮修改</h3><span>从父方案到当前候选</span></div><div class=\"change-list candidate-change-list\">" + renderCandidateChanges(candidate.changes) + "</div></section>",
      renderCandidateExecutionEvidence(candidate, run),
      "<section class=\"detail-section candidate-section\"><div class=\"candidate-section-heading\"><h3>分目标表现</h3><span>每个目标单独计算，避免单位混淆</span></div><div class=\"target-results\">" + renderCandidateTargets(targets) + "</div></section>",
      "<details class=\"detail-section candidate-evidence\"><summary>查看预测效果预览 <span>" + escapeHTML(predictionRows.length ? formatNumber(predictionRows.length) + " 条记录" : "暂无记录") + "</span></summary>" + predictionTable + "</details>",
      secondaryMetrics.length ? "<details class=\"detail-section candidate-evidence candidate-metrics-evidence\"><summary>查看全部训练反馈指标 <span>" + escapeHTML(formatNumber(secondaryMetrics.length)) + " 项</span></summary><div class=\"detail-grid\">" + secondaryMetrics.map(function (key) { return "<div class=\"detail-value\"><span>" + escapeHTML(metricLabel(key)) + "</span><strong title=\"" + escapeHTML(String(candidateMetrics[key])) + "\">" + escapeHTML(metricValue(key, candidateMetrics[key])) + "</strong></div>"; }).join("") + "</div></details>" : "",
      "<section class=\"detail-section candidate-section candidate-decision-section\"><div class=\"candidate-section-heading\"><h3>轮末结论</h3><span>训练反馈搜索范围内</span></div><div class=\"candidate-decision-grid\"><div class=\"detail-value\"><span>选择结果</span><strong><span class=\"pill " + outcome.className + "\">" + escapeHTML(gateText(promotion.decision || candidate.status)) + "</span></strong></div><div class=\"detail-value\"><span>选择依据</span><strong title=\"" + escapeHTML(decisionReason) + "\">" + escapeHTML(compactTechnicalText(selectionReasonText(decisionReason || candidate.selection_reason))) + "</strong></div><div class=\"detail-value\"><span>正式验证</span><strong>未开展</strong></div></div></section>",
      "<details class=\"detail-section candidate-evidence candidate-rationale\"><summary>查看生成依据 <span>模型提案与知识依据</span></summary><p title=\"" + escapeHTML(candidate.rationale || "") + "\">" + escapeHTML(rationale) + "</p></details>",
      "<details class=\"detail-section candidate-evidence\"><summary>查看训练产物与可复现信息</summary>" + artifactSection + "</details>"
    ].filter(Boolean);
    node.innerHTML = "<div class=\"candidate-detail-shell\">" + header + "<section class=\"candidate-overview\"><div class=\"candidate-metric-grid\">" + keyMetrics + "</div></section>" + failureSection + details.join("") + "</div>";
  }

  function renderCandidates() {
    var run = state.activeRun;
    var candidates = run ? run.candidates : [];
    syncCandidateSelection(run);
    var body = $("#candidate-table");
    var acceptableText = run && run.best_candidate_id ? "训练反馈搜索保留候选：" + shortId(run.best_candidate_id) + "（正式验证未开展）" : "训练反馈搜索保留候选：尚未产生（正式验证未开展）";
    var observedText = rawBestObservedSummary(run);
    if (run && runOutcomeCode(run) === "budget_exhausted_without_acceptable_candidate") { acceptableText = "已完成预设进化规模，尚无候选通过全部评测门控（正式验证未开展）"; }
    $("#best-candidate-label").textContent = acceptableText + "；" + observedText;
    $("#best-candidate-label").title = run ? [run.best_candidate_id, run.best_observed_candidate_id].filter(Boolean).join("\n") : "";
    $("#export-button").disabled = state.busy || !run;
    updateCandidateOverview(run, candidates);
    if (!candidates.length) { body.innerHTML = "<tr><td colspan=\"11\" class=\"empty-state\">暂无候选方案。</td></tr>"; renderCandidateDetail(null); renderCandidateSamples(); return; }
    body.innerHTML = candidates.slice().sort(function (left, right) { return Number(left.generation || 0) - Number(right.generation || 0) || Number(left.generation_rank || 999) - Number(right.generation_rank || 999) || Number(left.slot_index || 0) - Number(right.slot_index || 0); }).map(function (candidate) {
      var metrics = candidate.metrics || {};
      var score = candidateScoreValue(candidate);
      var delta = candidateDelta(candidate, candidates, run);
      var scientific = candidateGateDescriptor(candidateScientificPass(candidate));
      var judge = candidateJudgeDescriptor(candidate, run);
      var constraints = candidateFiniteNumber(metrics.constraint_violations);
      var outcome = candidateOutcome(candidate, run);
      var selected = candidate.id === state.selectedCandidateId;
      var winner = run && run.best_candidate_id === candidate.id;
      var scoreText = score == null ? "待评测" : formatNumber(score, 3);
      var deltaText = delta.value == null ? "—" : signedNumber(delta.value, 3);
      return "<tr class=\"candidate-row " + (selected ? "is-selected " : "") + (winner ? "is-winner " : "") + "\" data-candidate-state=\"" + escapeHTML(String(candidate.status || "pending")) + "\"><td data-label=\"候选编号\"><button class=\"candidate-select-button candidate-select-button--rich\" type=\"button\" data-candidate-id=\"" + escapeHTML(candidate.id) + "\" aria-pressed=\"" + String(selected) + "\" aria-label=\"查看候选方案 " + escapeHTML(candidate.id) + "\" title=\"" + escapeHTML(candidate.id) + "\"><span class=\"candidate-identity\"><strong>" + escapeHTML(candidateListId(candidate.id)) + "</strong><small>" + escapeHTML(winner ? "当前保留" : outcome.text) + "</small></span></button></td><td data-label=\"进化轮次\"><span class=\"candidate-round-cell\"><strong>第 " + escapeHTML(candidate.generation || "—") + " 轮</strong><small>槽位 " + escapeHTML(Number(candidate.slot_index || 0) + 1) + "</small></span></td><td data-label=\"轮内槽位\"><span class=\"candidate-slot-badge\">" + escapeHTML(Number(candidate.slot_index || 0) + 1) + "</span></td><td data-label=\"父方案\" title=\"" + escapeHTML(candidate.parent_id || "") + "\"><span class=\"candidate-parent-cell\">" + escapeHTML(shortId(candidate.parent_id || "基线")) + "</span></td><td data-label=\"轮内排名\"><span class=\"candidate-rank-badge\">" + escapeHTML(candidate.generation_rank == null ? "—" : candidate.generation_rank) + "</span></td><td data-label=\"状态\"><span class=\"pill " + candidateStatusClass(candidate.status) + "\">" + escapeHTML(candidateStatusText(candidate.status)) + "</span></td><td data-label=\"科学门禁\"><span class=\"pill " + scientific.className + "\">" + escapeHTML(scientific.text) + "</span></td><td data-label=\"独立评审\"><span class=\"pill " + judge.className + "\">" + escapeHTML(judge.text) + "</span></td><td data-label=\"约束违规\"><span class=\"candidate-constraint-value " + (constraints != null && constraints > 0 ? "is-warning" : "") + "\">" + escapeHTML(constraints == null ? "待评测" : formatNumber(constraints)) + "</span></td><td data-label=\"综合得分\"><span class=\"candidate-score-cell " + (score == null ? "is-pending" : "") + "\"><strong>" + escapeHTML(scoreText) + "</strong><small>" + escapeHTML(delta.label) + " " + escapeHTML(deltaText) + "</small></span></td><td data-label=\"选择结论\" title=\"" + escapeHTML(selectionReasonText(candidate.selection_reason)) + "\"><span class=\"candidate-decision-cell\"><strong class=\"pill " + outcome.className + "\">" + escapeHTML(outcome.text) + "</strong><small>" + escapeHTML(compactTechnicalText(selectionReasonText(candidate.selection_reason))) + "</small></span></td></tr>";
    }).join("");
    renderCandidateDetail(candidates.find(function (candidate) { return candidate.id === state.selectedCandidateId; }), run, candidates);
    renderCandidateSamples();
  }
