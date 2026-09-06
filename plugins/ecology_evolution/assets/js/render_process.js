"use strict";

  function eventCategory(type) {
    var value = String(type || "").toLowerCase();
    var compactCategories = {
      runseedgenomematerialized: "运行", dshruntimebound: "DSH",
      generationresearchiterated: "研究", runadaptationcohortfrozen: "数据队列",
      generationcohortsfrozen: "数据队列", candidaterevisioncreated: "候选",
      algorithmattemptrecorded: "算法", dshchildlaunchreserved: "DSH",
      dshretrievalexecuted: "DSH", dshpredictiontoolexecuted: "DSH",
      dshstructuredresultaccepted: "DSH"
    };
    if (compactCategories[value]) { return compactCategories[value]; }
    return {
      run: "运行", generation: "轮次", knowledge: "知识", research: "研究",
      implementation: "编译", optimization: "优化", proposal: "提案", candidate: "候选",
      artifact: "产物", evaluation: "反馈", formal: "评测", promotion: "决策", intervention: "人工",
      expert_consultation: "专家", consultation: "专家", stage: "阶段", gateway: "网关",
      model: "模型", dsh: "DSH"
    }[value.split(".")[0]] || "系统";
  }
  function compactDuration(seconds) {
    var value = Number(seconds);
    if (!Number.isFinite(value) || value < 0) { return ""; }
    if (value < 3600) { return Math.max(1, Math.round(value / 60)) + " 分钟"; }
    if (value < 86400) { return formatNumber(value / 3600, 1) + " 小时"; }
    return formatNumber(value / 86400, 1) + " 天";
  }
  function modelUsageCallEfficiencyText(usage) {
    var source = usage && typeof usage === "object" ? usage : {};
    var physicalCalls = Number(source.physical_call_count);
    var logicalCalls = Number(source.logical_call_count);
    var replayedCalls = Number(source.replayed_call_count);
    var allCalls = Number(source.call_count);
    if (
      !Number.isInteger(physicalCalls) || physicalCalls < 1 ||
      !Number.isInteger(logicalCalls) || logicalCalls < 1 ||
      !Number.isInteger(replayedCalls) || replayedCalls < 0 ||
      logicalCalls + replayedCalls !== physicalCalls ||
      !Number.isInteger(allCalls) || allCalls !== physicalCalls
    ) { return ""; }
    var replayRate = replayedCalls / physicalCalls * 100;
    return "调用（全运行累计）：物理 " + formatNumber(physicalCalls) + " / 逻辑 " + formatNumber(logicalCalls) + " · 重复调用率 " + replayRate.toFixed(1) + "%";
  }
  function modelUsageStageEstimateText(usage, stageProgress, candidate) {
    var source = usage && typeof usage === "object" ? usage : {};
    var progress = stageProgress && typeof stageProgress === "object" ? stageProgress : {};
    var candidateId = candidate && (candidate.id || candidate.candidate_id);
    if (
      source.complete !== true ||
      !candidateId || source.scope_candidate_id !== candidateId ||
      !progress.revision || source.scope_revision !== progress.revision
    ) { return ""; }
    var totalTokens = Number(source.total_tokens);
    var completedSamples = Number(progress.completed_samples);
    var totalSamples = Number(progress.total_samples);
    if (
      !Number.isFinite(totalTokens) || totalTokens <= 0 ||
      !Number.isInteger(completedSamples) || completedSamples < 1 ||
      !Number.isInteger(totalSamples) || totalSamples < completedSamples
    ) { return ""; }
    var tokensPerCompletedSample = totalTokens / completedSamples;
    var projectedTokens = totalTokens * totalSamples / completedSamples;
    if (!Number.isFinite(tokensPerCompletedSample) || !Number.isFinite(projectedTokens)) { return ""; }
    return "阶段估算：Token / 已完成样本 " + formatNumber(tokensPerCompletedSample, 1) + " · 跑满 " + formatNumber(totalSamples) + " 样本约 " + formatNumber(Math.ceil(projectedTokens)) + " Token";
  }
  function modelUsageTokenProgressText(run, stageProgress, candidate) {
    var dshRuntime = run && run.dsh_runtime && typeof run.dsh_runtime === "object" ? run.dsh_runtime : {};
    if (dshRuntime.native === true) {
      var pressure = dshRuntime.context_pressure && typeof dshRuntime.context_pressure === "object" ? dshRuntime.context_pressure : {};
      var provider = dshRuntime.provider_usage && typeof dshRuntime.provider_usage === "object" ? dshRuntime.provider_usage : {};
      var pressureRatio = Number(pressure.ratio);
      var maximumCurrentTokens = Number(pressure.maximum_total_tokens);
      var pressureText = pressure.available === true && Number.isFinite(pressureRatio)
        ? "DSH 上下文压力（当前值）：" + formatNumber(pressureRatio * 100, 1) + "%"
        : pressure.available === true && Number.isFinite(maximumCurrentTokens)
          ? "DSH 上下文压力（会话最大当前值）：" + formatNumber(maximumCurrentTokens) + " Token"
          : "DSH 上下文压力：等待 Session 计量";
      var parts = [pressureText];
      if (Number.isFinite(pressureRatio)) {
        if (pressureRatio >= 1) { parts.push("硬上限已到达，停止新请求"); }
        else if (pressureRatio >= 0.85) { parts.push("高压力预警（≥85%）"); }
        else if (pressureRatio >= 0.70) { parts.push("压力提醒（≥70%）"); }
      }
      if (provider.available === true && Number.isFinite(Number(provider.total_tokens))) {
        parts.push("供应商报告累计用量：" + formatNumber(Number(provider.total_tokens)) + " Token");
      } else {
        parts.push("供应商累计用量：尚未报告");
      }
      return parts.join("（") + (parts.length > 1 ? "）" : "");
    }
    var usage = run && run.model_usage && typeof run.model_usage === "object" ? run.model_usage : {};
    var usedTokens = Number(run && run.tokens_used);
    var tokenLimit = Number(run && run.token_limit);
    var usageAvailable = run && run.token_usage_available === true || usage.available === true;
    var sampleAgentTokenBudget = runUsesSampleAgentTokenBudget(run);
    var tokenLabel = sampleAgentTokenBudget ? tokenBudgetSubjectText(run) + " Token" : "Token 账本（历史口径）";
    if (!usageAvailable) { return tokenLabel + "：等待真实账本"; }
    if (!Number.isFinite(usedTokens)) { return tokenLabel + "：账本记录异常"; }

    var missingUsageCalls = Number(usage.missing_call_count);
    var hasMissingUsage = Number.isInteger(missingUsageCalls) && missingUsageCalls > 0;
    var usageIncomplete = usage.complete === false || hasMissingUsage;
    var qualifiers = [];
    if (!Number.isFinite(tokenLimit) || tokenLimit <= 0) { qualifiers.push("仅计量，不限额"); }
    if (usageIncomplete) {
      qualifiers.push(hasMissingUsage ? formatNumber(missingUsageCalls) + " 次调用实际用量未知" : "部分调用实际用量未知");
      var rawTotalTokens = Number(usage.total_tokens);
      if (Number.isFinite(rawTotalTokens) && rawTotalTokens >= 0) {
        qualifiers.push("原始回执计数 " + formatNumber(rawTotalTokens) + " Token");
      }
    }
    var valueText = (usageIncomplete ? tokenLabel + " 预算计入：" : tokenLabel + "：") + formatNumber(usedTokens);
    if (Number.isFinite(tokenLimit) && tokenLimit > 0) { valueText += " / " + formatNumber(tokenLimit); }
    if (qualifiers.length) { valueText += "（" + qualifiers.join("；") + "）"; }
    return [
      valueText,
      modelUsageCallEfficiencyText(usage),
      modelUsageStageEstimateText(usage, stageProgress, candidate)
    ].filter(Boolean).join(" · ");
  }
  function supersededSampleRevisionText(value) {
    var summary = value && typeof value === "object" ? value : null;
    if (!summary) { return ""; }
    var completed = Number(summary.completed_samples);
    var total = Number(summary.total_samples);
    if (!Number.isFinite(completed) || completed < 0 || !Number.isFinite(total) || total < 1) { return ""; }
    var reason = {
      legacy_revision_without_checkpoint: "旧修订缺少可验证 checkpoint",
      checkpoint_digest_mismatch: "checkpoint 与当前执行上下文不一致",
      latest_revision_completed: "上一修订已完成"
    }[String(summary.resume_disposition || "")] || "无法安全复用上一修订";
    var succeeded = Number(summary.succeeded_samples);
    var failed = Number(summary.failed_samples);
    var outcome = Number.isFinite(succeeded) && succeeded >= 0 && Number.isFinite(failed) && failed >= 0
      ? "成功 " + formatNumber(succeeded) + "，失败 " + formatNumber(failed)
      : "仅保留汇总进度";
    return "上一修订已隔离：" + formatNumber(completed) + " / " + formatNumber(total) + " 条（" + outcome + "；" + reason + "）";
  }
  function stageStatusText(value) {
    return {
      started: "已开始", running: "进行中", in_progress: "进行中", completed: "已完成",
      failed: "失败", paused: "已暂停", aborted: "已中止", not_recorded: "未封存"
    }[String(value || "").toLowerCase()] || "状态已更新";
  }
  var pairedDecisionLabels = {
    initial_champion: "初始版本已确定",
    challenger_promoted: "新版本已保留用于优化",
    champion_retained: "未达到保留条件，沿用原版本"
  };
  function pairedDecisionText(value) {
    return pairedDecisionLabels[String(value || "").toLowerCase()] || "";
  }
  function promotionDecisionTitle(event) {
    var decision = String(payloadOf(event).decision || "").toLowerCase();
    if (["approved", "accepted", "promoted", "pass", "passed", "true"].indexOf(decision) >= 0) { return "候选方案已在训练反馈搜索中保留（正式验证未开展）"; }
    if (["rejected", "declined", "denied", "fail", "failed", "false"].indexOf(decision) >= 0) { return "候选方案未保留"; }
    return "搜索保留决策已记录";
  }
  function evaluationProgressEventTitle(run) {
    var runStatus = String(run && run.status || "").toLowerCase();
    var evidenceStatus = executionEvidenceStatus(run);
    if (evidenceStatus === "partial_live" && runStatus === "running") { return "真实样本评测正在推进"; }
    if (evidenceStatus === "retained_partial" || runStatus === "paused") { return "已保留部分样本评测证据"; }
    if (evidenceStatus === "aborted_partial" || evidenceStatus === "aborted" || runStatus === "cancelled" || runStatus === "failed") { return "已保留中止前样本评测证据"; }
    if (evidenceStatus === "mixed_partial") { return "部分样本评测证据已记录"; }
    if (["completed", "cancelled", "failed"].indexOf(runStatus) >= 0) { return "样本评测进度已记录"; }
    return "真实样本评测正在推进";
  }
  function eventTitle(event, run) {
    if (event.type === "candidate.spawned" && payloadOf(event).role === "incumbent_control") { return "初始种子对照已冻结"; }
    if (event.type === "evaluation.progress") { return evaluationProgressEventTitle(run); }
    if (event.type === "stage.recorded") {
      var stagePayload = payloadOf(event);
      return (evolutionStageLabels[stagePayload.stage] || "进化阶段") + stageStatusText(executionStatusForRun(run, stagePayload.status));
    }
    if (event.type === "formal.batch_compared") {
      return pairedDecisionText(payloadOf(event).decision) || "同 cohort 冠军—挑战者比较已记录";
    }
    if (event.type === "promotion.decided") { return promotionDecisionTitle(event); }
    if (event.type === "intervention.applied") {
      var payload = payloadOf(event);
      var intervention = payload.intervention && typeof payload.intervention === "object" ? payload.intervention : payload;
      var explicit = intervention.application_status || intervention.enforced === true && "enforced" || intervention.applied === true && "applied";
      return explicit ? "人工意见" + interventionApplicationText(explicit) : "人工意见执行状态已更新";
    }
    return eventLabels[event.type] || "系统事件";
  }
  function eventDetail(event, run) {
    var payload = payloadOf(event);
    if (event.type === "formal.batch_compared") {
      var championScore = Number(payload.champion_score);
      var challengerScore = Number(payload.challenger_score);
      var scoreDelta = Number(payload.score_delta);
      var minimumDelta = Number(payload.minimum_score_delta);
      return [
        Number.isFinite(championScore) ? "原版本 " + formatNumber(championScore, 4) : "",
        Number.isFinite(challengerScore) ? "新版本 " + formatNumber(challengerScore, 4) : "",
        Number.isFinite(scoreDelta) ? "提高值 " + signedNumber(scoreDelta, 4) : "",
        Number.isFinite(minimumDelta) ? "须超过 " + formatNumber(minimumDelta, 4) : "",
        payload.reason ? "原因 " + compactTechnicalText(payload.reason) : ""
      ].filter(Boolean).join(" · ");
    }
    if (event.type === "evaluation.progress") {
      var completed = Number(payload.completed_samples);
      var total = Number(payload.total_samples);
      return [
        payload.candidate_id ? "候选 " + shortId(payload.candidate_id) : "",
        Number.isFinite(completed) && completed >= 0 && Number.isFinite(total) && total > 0 ? "已完成 " + formatNumber(completed) + " / " + formatNumber(total) + " 个样本" : "",
        evaluationProgressEventTitle(run)
      ].filter(Boolean).join(" · ");
    }
    if (event.type === "stage.recorded") {
      return [
        payload.generation != null ? "第 " + formatNumber(Number(payload.generation) + 1) + " 轮" : "",
        payload.candidate_id ? "候选 " + shortId(payload.candidate_id) : "",
        Number(payload.attempt || 0) > 1 ? "第 " + formatNumber(payload.attempt) + " 次尝试" : "",
        payload.public_error ? "失败说明：" + compactTechnicalText(payload.public_error) : ""
      ].filter(Boolean).join(" · ");
    }
    if (event.type === "gateway.retry_scheduled") {
      return [payload.retry_at ? "下次重试 " + formatTime(payload.retry_at) : "等待下一次调用", payload.delay_seconds != null ? "延迟 " + formatNumber(payload.delay_seconds, 1) + " 秒" : "", payload.attempt ? "第 " + formatNumber(payload.attempt) + " 次" : "", payload.reason || "网关队列繁忙，运行保持存活"].filter(Boolean).join(" · ");
    }
    if (event.type === "dsh.child_execution_failed") {
      var childErrorCode = String(payload.error_code || "");
      var childErrorText = childErrorCode === "structured_child_model_error"
        ? "模型未返回可验收的结构化结果"
        : "DSH 子任务未完成";
      return [payload.stage ? "阶段 " + payload.stage : "", childErrorText, "不计为模型成功；最终失败预测时点按宿主结算口径记录"].filter(Boolean).join(" · ");
    }
    if (event.type === "promotion.decided") {
      var explanation = payload.reason;
      if (explanation && /[\u3400-\u9fff]/.test(String(explanation))) { return String(explanation); }
      return payload.candidate_id ? "候选 " + shortId(payload.candidate_id) + " 的搜索保留结论已写入追加式记录。" : "搜索保留结论已写入追加式记录。";
    }
    if (event.type === "evaluation.completed") {
      return [payload.candidate_id ? "候选 " + shortId(payload.candidate_id) : "候选方案", payload.score != null ? "得分 " + formatNumber(payload.score) : "", payload.passed === true ? "通过训练反馈检查" : payload.passed === false ? "未通过训练反馈检查" : ""].filter(Boolean).join(" · ");
    }
    if (event.type === "candidate.failed") {
      return [payload.candidate_id ? "候选 " + shortId(payload.candidate_id) : "候选方案", payload.reason ? "失败原因：" + compactTechnicalText(payload.reason) : "训练或评测失败"].filter(Boolean).join(" · ");
    }
    if (event.type === "generation.advanced" && payload.generation != null) { return "累计已完成 " + formatNumber(payload.generation) + " 轮。"; }
    if (event.type === "generation.batch_started") { return "第 " + formatNumber(Number(payload.generation || 0) + 1) + " 轮已冻结 " + formatNumber(payload.batch_size || 0) + " 个候选槽位。"; }
    if (event.type === "generation.analyzed") { return "第 " + formatNumber(Number(payload.generation || 0) + 1) + " 轮共分析 " + formatNumber(payload.candidate_count || 0) + " 个版本，其中 " + formatNumber(payload.eligible_count || 0) + " 个符合晋级条件。"; }
    if (event.type === "generation.champion_selected") { return payload.selection_reason || "本轮单一冠军选择已记录。"; }
    if (event.type === "knowledge.retrieved") { return "检索 " + formatNumber(payload.source_count || 0) + " 条来源，其中 " + formatNumber(payload.adopted_count || 0) + " 条映射到本轮可执行能力。"; }
    if (event.type === "knowledge.assessed") { return payload.conclusion || "本轮知识指导结果已记录。"; }
    if (event.type === "proposal.submitted") { return payload.title || (payload.proposal_id ? "提案 " + shortId(payload.proposal_id) : "提案内容已冻结。"); }
    if (event.type === "candidate.spawned" && payload.candidate_id) {
      return payload.role === "incumbent_control"
        ? "种子基线对照（不占候选预算） · 对照编号：" + shortId(payload.candidate_id)
        : "候选编号：" + shortId(payload.candidate_id);
    }
    if (event.type === "artifact.recorded") { return [payload.candidate_id ? "候选 " + shortId(payload.candidate_id) : "", payload.artifact_id ? "产物 " + shortId(payload.artifact_id) : ""].filter(Boolean).join(" · ") || "训练产物摘要已写入。"; }
    if (event.type === "intervention.recorded") { return "类型：" + (interventionLabels[payload.kind] || "人工意见") + " · 状态：已记录"; }
    if (event.type === "intervention.applied") {
      var intervention = payload.intervention && typeof payload.intervention === "object" ? payload.intervention : payload;
      return interventionExecutionText(intervention) || compactTechnicalText(payload.message || "人工意见的执行状态已记录。");
    }
    if (payload.message && /[\u3400-\u9fff]/.test(String(payload.message)) && String(payload.message).replace(/[。\s]/g, "") !== eventTitle(event, run).replace(/[。\s]/g, "")) { return String(payload.message); }
    var context = [payload.generation != null ? "第 " + formatNumber(payload.generation) + " 轮" : "", payload.candidate_id ? "候选 " + shortId(payload.candidate_id) : ""].filter(Boolean).join(" · ");
    return context;
  }
  function eventTone(event, run) {
    var type = typeof event === "string" ? event : event && event.type || "";
    if (type === "stage.recorded" && typeof event !== "string") {
      var stageStatus = String(executionStatusValue(executionStatusForRun(run, payloadOf(event).status)) || "").toLowerCase();
      if (stageStatus === "completed") { return "pill-green"; }
      if (stageStatus === "failed" || stageStatus === "aborted") { return "pill-red"; }
      if (["running", "started", "in_progress", "evaluating"].indexOf(stageStatus) >= 0) { return "pill-blue"; }
      return "pill-amber";
    }
    if (type === "promotion.decided") {
      var decision = String(payloadOf(event).decision || "").toLowerCase();
      if (["approved", "accepted", "promoted", "pass", "passed", "true"].indexOf(decision) >= 0) { return "pill-green"; }
      if (["rejected", "declined", "denied", "fail", "failed", "false"].indexOf(decision) >= 0) { return "pill-red"; }
    }
    if (type === "intervention.applied" && typeof event !== "string") {
      var payload = payloadOf(event);
      var intervention = payload.intervention && typeof payload.intervention === "object" ? payload.intervention : payload;
      var status = interventionApplicationStatus(intervention);
      return status === "recorded" ? "pill-amber" : "pill-green";
    }
    if (/completed|accepted|started|resumed/.test(type)) { return "pill-green"; }
    if (/failed|rejected|cancelled|blocked/.test(type)) { return "pill-red"; }
    return "pill-amber";
  }

  function trajectoryIncumbentScore(point) {
    var item = point || {};
    var value = item.incumbent_score != null ? item.incumbent_score : item.best_score;
    return value == null || value === "" ? NaN : Number(value);
  }

  function renderTrajectory() {
    // Candidate scores are raw observations. Only the server-recorded
    // incumbent fields may form the green promotion sequence.
    var node = $("#trajectory-chart");
    var run = state.activeRun;
    var legend = node && node.parentElement && node.parentElement.querySelector(".chart-legend span:last-child");
    var points = run && Array.isArray(run.trajectory) ? run.trajectory.filter(function (point) { return Number.isFinite(Number(point.score != null ? point.score : point.candidate_score)); }).slice().sort(function (left, right) {
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
    if (legend) { legend.innerHTML = "<i class=\"legend-best\"></i>" + (crossCohort ? "跨 cohort 不连线" : incumbentPath ? "实际晋升序列" : "实际晋升序列（历史运行未记录）"); }
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

  function roundStageStatus(round, key, run) {
    var stage = round && round.stages && round.stages[key];
    var progress = run && run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : {};
    var stageProgress = progress.stage_progress && typeof progress.stage_progress === "object" ? progress.stage_progress : {};
    var adaptivePhase = String(stageProgress.evaluation_phase || "").toLowerCase();
    var activeCandidateId = String(progress.current_candidate_id || "");
    var candidateId = String(round && (round.candidate_id || round.id) || "");
    var liveOrPaused = executionRunAllowsLiveStatus(run) || String(run && run.status || "").toLowerCase() === "paused";
    var trajectory = candidateId && Array.isArray(run && run.adaptive_trajectories) ? run.adaptive_trajectories.find(function (item) {
      return String(item && item.candidate_id || "") === candidateId;
    }) : null;
    var trajectoryStatus = String(trajectory && trajectory.status || "").toLowerCase();
    var trajectoryInProgress = trajectory && (
      ["running", "started", "in_progress"].indexOf(trajectoryStatus) >= 0
      || Number(trajectory.completed_batch_count || 0) < Number(trajectory.batch_count || 0)
    );
    if (liveOrPaused && trajectoryInProgress) {
      if (key === "training") { return "completed"; }
      if (key === "evaluation") { return String(run.status || "").toLowerCase() === "paused" ? "paused" : "running"; }
    }
    if (liveOrPaused && candidateId && candidateId === activeCandidateId && ["screening", "formal_batch", "holdout"].indexOf(adaptivePhase) >= 0) {
      if (key === "training") { return "completed"; }
      if (key === "evaluation") { return String(run.status || "").toLowerCase() === "paused" ? "paused" : "running"; }
    }
    if (key === "decision" && round && !isBlank(round.decision)) { return typeof round.decision === "object" ? round.decision.status || round.decision.decision || stage : round.decision; }
    if (stage && typeof stage === "object") { return stage.status || stage.state || stage.result || "pending"; }
    return stage || "pending";
  }
  function roundStageText(key, value, run) {
    var status = String(executionStatusValue(executionStatusForRun(run, value)) || "pending").toLowerCase();
    if (key === "decision") {
      if (["approved", "accepted", "retained", "promoted"].indexOf(status) >= 0) { return "训练反馈搜索保留"; }
      if (["rejected", "declined", "denied"].indexOf(status) >= 0) { return "未保留"; }
    }
    if (["completed", "done", "recorded", "passed", "approved", "accepted"].indexOf(status) >= 0) { return "已完成"; }
    if (status === "skipped" || status === "duplicate") { return "已跳过"; }
    if (["failed", "error", "rejected"].indexOf(status) >= 0) { return "未通过"; }
    if (status === "paused") { return "已暂停"; }
    if (status === "aborted") { return "已中止"; }
    if (status === "not_recorded") { return "未封存"; }
    if (["running", "in_progress", "evaluating"].indexOf(status) >= 0) { return "进行中"; }
    return "等待";
  }
  function roundStageClass(key, value, run) {
    var text = roundStageText(key, value, run);
    if (text === "已完成" || text === "训练反馈搜索保留") { return "is-complete"; }
    if (text === "未通过" || text === "未保留" || text === "已中止") { return "is-rejected"; }
    if (text === "进行中") { return "is-running"; }
    return "is-pending";
  }
  function roundUsesCrossCohortSearchParent(round) {
    return Boolean(round && Array.isArray(round.candidates) && round.candidates.some(function (candidate) {
      return String(candidate && candidate.selection_reason || "") === "cohort_changed_search_parent_only";
    }));
  }
  function generationOutcomeText(value, round) {
    if (String(value || "pending") === "no_improvement" && roundUsesCrossCohortSearchParent(round)) {
      return "跨窗口未比较，保留正式方案";
    }
    return { promoted: "产生新冠军", no_improvement: "未改善，保留原方案", no_eligible_candidate: "没有符合晋级条件的候选", pending: "等待轮末分析" }[String(value || "pending")] || String(value || "等待轮末分析");
  }
  function generationOutcomeClass(value) {
    var status = String(value || "pending");
    if (status === "promoted") { return "pill-green"; }
    if (status === "pending") { return "pill-amber"; }
    if (status === "no_improvement") { return "pill-blue"; }
    if (status === "no_eligible_candidate") { return "pill-amber"; }
    return "pill-red";
  }
  function generationWeaknessText(round) {
    var target = Array.isArray(round.target_weaknesses) && round.target_weaknesses[0];
    if (target) {
      var label = targetLabels[target.target] || target.target || "预测目标";
      return label + (target.horizon_hours != null ? " · " + formatNumber(target.horizon_hours) + " 小时" : "") + " · 中位技能得分 " + formatNumber(target.median_skill_score);
    }
    var failures = Array.isArray(round.common_failures) ? round.common_failures : [];
    return failures.length ? failures.map(selectionReasonText).join("、") : "尚未识别共同弱点";
  }
  function knowledgeStatusLabel(value) {
    return { adopted: "本轮采用", available_not_selected: "本地可用，未启用", research_only: "仅供研究", metadata_only: "在线元数据" }[String(value || "")] || "待判断";
  }
  function knowledgeStatusClass(value) {
    if (value === "adopted") { return "pill-green"; }
    if (value === "available_not_selected") { return "pill-blue"; }
    return "pill-neutral";
  }
  function renderRoundKnowledge(round) {
    var knowledge = round && round.knowledge;
    if (!knowledge || typeof knowledge !== "object") {
      var missingText = round && round.decision && round.decision !== "pending" ? "该历史轮次未记录知识快照" : "等待本轮知识快照";
      return "<section class=\"round-knowledge\"><div class=\"round-knowledge-heading\"><div><span>知识检索与方案筛选</span><strong>" + missingText + "</strong></div></div></section>";
    }
    var cards = Array.isArray(knowledge.cards) ? knowledge.cards : [];
    var adopted = cards.filter(function (item) { return item.execution_status === "adopted"; }).length;
    var queries = Array.isArray(knowledge.query_terms) ? knowledge.query_terms : [];
    var assessment = round.knowledge_assessment && typeof round.knowledge_assessment === "object" ? round.knowledge_assessment : null;
    var rows = cards.map(function (item) {
      var mapping = item.capability_id ? "执行映射：" + item.capability_id : "未接入本地执行器";
      var year = item.publication_year ? " · " + item.publication_year : "";
      return "<div class=\"knowledge-source-row\"><div><a href=\"" + escapeHTML(item.source_url || "#") + "\" target=\"_blank\" rel=\"noopener noreferrer\">" + escapeHTML(item.title || "未命名来源") + "</a><span>" + escapeHTML((item.source_authority || "公开来源") + year + " · " + mapping) + "</span></div><p>" + escapeHTML(item.summary || "暂无摘要") + "</p><span class=\"pill " + knowledgeStatusClass(item.execution_status) + "\" title=\"" + escapeHTML(item.selection_reason || "") + "\">" + escapeHTML(knowledgeStatusLabel(item.execution_status)) + "</span></div>";
    }).join("");
    var outcome = assessment ? "<div class=\"knowledge-assessment\"><span>轮末联合结果</span><strong>" + escapeHTML(assessment.conclusion || "已完成判断") + "</strong><small>下一轮：" + escapeHTML(assessment.next_action || "等待分析") + "</small></div>" : "<div class=\"knowledge-assessment is-pending\"><span>轮末联合结果</span><strong>等待候选完成评测</strong></div>";
    return "<section class=\"round-knowledge\"><div class=\"round-knowledge-heading\"><div><span>知识检索与方案筛选</span><strong>" + escapeHTML(cards.length) + " 条来源 · " + escapeHTML(adopted) + " 条本轮采用</strong></div><code title=\"" + escapeHTML(knowledge.snapshot_digest || "") + "\">快照 " + escapeHTML(shortId(knowledge.snapshot_digest || "未生成")) + "</code></div><div class=\"knowledge-query-list\"><span>检索词</span>" + (queries.length ? queries.map(function (item) { return "<code>" + escapeHTML(item) + "</code>"; }).join("") : "<code>仅使用内置目录</code>") + "</div><div class=\"knowledge-source-list\">" + (rows || "<div class=\"empty-state\">未找到匹配来源。</div>") + "</div>" + outcome + "</section>";
  }
  function roundResearchObject(value) { return value && typeof value === "object" && !Array.isArray(value) ? value : {}; }
  function roundResearchStatus(value) {
    var status = String(value || "pending").toLowerCase();
    if (["completed", "passed", "ready", "ready_for_host_compilation", "model_generated", "initial_frozen", "recovered_existing_proposal"].indexOf(status) >= 0) { return {text: "已完成", className: "is-complete"}; }
    if (["running", "started", "in_progress", "evaluating", "compiled"].indexOf(status) >= 0) { return {text: "进行中", className: "is-running"}; }
    if (status === "paused") { return {text: "已暂停", className: "is-warning"}; }
    if (status === "aborted") { return {text: "已中止", className: "is-failed"}; }
    if (status === "not_recorded") { return {text: "未封存", className: "is-pending"}; }
    if (["failed", "error", "compile_failed", "debug_failed"].indexOf(status) >= 0) { return {text: "失败", className: "is-failed"}; }
    if (["degraded", "research_only", "partial"].indexOf(status) >= 0) { return {text: "部分完成", className: "is-warning"}; }
    if (["skipped", "duplicate"].indexOf(status) >= 0) { return {text: "已跳过", className: "is-skipped"}; }
    return {text: "等待", className: "is-pending"};
  }
  function roundResearchCandidates(round, run) {
    var runCandidates = run && Array.isArray(run.candidates) ? run.candidates : [];
    var roundRows = round && Array.isArray(round.candidates) ? round.candidates : [];
    var ids = roundRows.map(function (item) { return String(item.candidate_id || item.id || ""); }).filter(Boolean);
    var matched = runCandidates.filter(function (candidate) { return ids.indexOf(String(candidate.id || candidate.candidate_id || "")) >= 0; });
    if (matched.length || ids.length) { return matched; }
    return runCandidates.filter(function (candidate) { return Number(candidate.generation || 0) === Number(round && round.generation || 0); });
  }
  function roundResearchStageCounts(round, candidates, key, run) {
    var rows = round && Array.isArray(round.candidates) ? round.candidates : [];
    var statuses = rows.map(function (row) { return String(executionStatusValue(executionStatusForRun(run, roundStageStatus(row, key))) || "pending").toLowerCase(); });
    if (!statuses.length) {
      statuses = candidates.map(function (candidate) {
        var stages = candidate && candidate.execution && candidate.execution.stages || candidate && candidate.stages || {};
        return String(executionStatusValue(executionStatusForRun(run, stages[key])) || "pending").toLowerCase();
      });
    }
    return {
      total: statuses.length,
      completed: statuses.filter(function (status) { return ["completed", "passed", "accepted", "approved", "recorded"].indexOf(status) >= 0; }).length,
      running: statuses.filter(function (status) { return ["running", "started", "in_progress", "evaluating"].indexOf(status) >= 0; }).length,
      paused: statuses.filter(function (status) { return status === "paused"; }).length,
      aborted: statuses.filter(function (status) { return status === "aborted"; }).length,
      notRecorded: statuses.filter(function (status) { return status === "not_recorded"; }).length,
      failed: statuses.filter(function (status) { return ["failed", "error", "rejected"].indexOf(status) >= 0; }).length
    };
  }
  function roundResearchAggregateStatus(counts) {
    if (counts.running > 0) { return "running"; }
    if (counts.completed > 0 && counts.completed === counts.total) { return "completed"; }
    if (counts.completed > 0) { return "partial"; }
    if (counts.paused > 0) { return "paused"; }
    if (counts.aborted > 0) { return "aborted"; }
    if (counts.failed > 0) { return "failed"; }
    if (counts.notRecorded > 0) { return "not_recorded"; }
    return "pending";
  }
  function roundResearchChainStep(index, title, statusValue, detail, meta, run) {
    var status = roundResearchStatus(executionStatusForRun(run, statusValue));
    return "<li class=\"round-research-step " + status.className + "\"><span class=\"round-research-index\">" + escapeHTML(String(index).padStart(2, "0")) + "</span><div><small>" + escapeHTML(status.text) + "</small><strong>" + escapeHTML(title) + "</strong><p title=\"" + escapeHTML(String(detail || "尚未产生记录")) + "\">" + escapeHTML(compactTechnicalText(String(detail || "尚未产生记录").slice(0, 180))) + "</p><em>" + escapeHTML(compactTechnicalText(String(meta || "").slice(0, 140))) + "</em></div></li>";
  }
  function renderRoundResearchEvidence(round, run) {
    var iteration = roundResearchObject(round && round.research_iteration);
    var analysis = roundResearchObject(iteration.analysis_summary);
    var finalPlan = roundResearchObject(iteration.final_plan);
    var candidates = roundResearchCandidates(round, run);
    var executions = candidates.map(function (candidate) { return roundResearchObject(candidate.algorithm_execution); });
    var compilePassed = executions.filter(function (execution) {
      return Boolean(execution.algorithm_spec) || (Array.isArray(execution.attempts) && execution.attempts.some(function (attempt) { return attempt && attempt.phase === "compile" && attempt.status === "passed"; }));
    }).length;
    var compileFailed = executions.filter(function (execution) { return String(execution.status || "").toLowerCase() === "compile_failed"; }).length;
    var compileStatus = compilePassed > 0 ? compilePassed === candidates.length ? "completed" : "partial" : compileFailed > 0 ? "failed" : "pending";
    var smokePassed = executions.filter(function (execution) {
      return execution.training_authorized === true || (Array.isArray(execution.attempts) && execution.attempts.some(function (attempt) { return attempt && attempt.phase === "debug" && attempt.status === "passed" && (!attempt.evidence || !attempt.evidence.source_partition || attempt.evidence.source_partition === "training_fit"); }));
    }).length;
    var smokeFailed = executions.filter(function (execution) { return String(execution.status || "").toLowerCase() === "debug_failed"; }).length;
    var smokeStatus = smokePassed > 0 ? smokePassed === candidates.length ? "completed" : "partial" : smokeFailed > 0 ? "failed" : "pending";
    var evaluationCounts = roundResearchStageCounts(round, candidates, "evaluation", run);
    var judgeCounts = roundResearchStageCounts(round, candidates, "judge", run);
    var findings = Array.isArray(analysis.key_findings) ? analysis.key_findings : [];
    var evidenceRefs = Array.isArray(analysis.evidence_refs) ? analysis.evidence_refs : [];
    var operators = Array.isArray(finalPlan.operator_ids) ? finalPlan.operator_ids : [];
    var parameters = Array.isArray(finalPlan.parameter_focus) && finalPlan.parameter_focus.length ? finalPlan.parameter_focus : Array.isArray(finalPlan.parameter_names) ? finalPlan.parameter_names : [];
    var analysisDetail = analysis.summary || (Object.keys(iteration).length ? "研究迭代已记录，等待分析摘要" : "等待策略模型形成分析摘要");
    var planIdentity = finalPlan.pipeline_id || finalPlan.predictor_id || "等待宿主可执行方案";
    var implementationMode = finalPlan.implementation_mode === "registered_host_components_only" ? "仅宿主登记组件" : "宿主执行边界待确认";
    var evaluationStatus = roundResearchAggregateStatus(evaluationCounts);
    var judgeStatus = roundResearchAggregateStatus(judgeCounts);
    var steps = [
      roundResearchChainStep(1, "分析总结", analysis.status || iteration.status, analysisDetail, formatNumber(evidenceRefs.length) + " 条证据引用 · " + formatNumber(findings.length) + " 条关键发现", run),
      roundResearchChainStep(2, "最终方案", finalPlan.status, planIdentity, formatNumber(operators.length) + " 个操作子 · 参数 " + (parameters.length ? parameters.slice(0, 5).map(humanizeTechnicalText).join("、") : "待生成"), run),
      roundResearchChainStep(3, "注册与编译", compileStatus, compilePassed ? formatNumber(compilePassed) + " / " + formatNumber(Math.max(candidates.length, compilePassed)) + " 个候选已编译" : "等待 AlgorithmSpec 与已登记适配器绑定", implementationMode, run),
      roundResearchChainStep(4, "training_fit smoke", smokeStatus, smokePassed ? formatNumber(smokePassed) + " / " + formatNumber(Math.max(candidates.length, smokePassed)) + " 个候选通过" : "等待训练拟合分区预检", "仅 training_fit，不读取后续分区", run),
      roundResearchChainStep(5, "training_feedback 正式评测", evaluationStatus, evaluationCounts.total ? formatNumber(evaluationCounts.completed) + " / " + formatNumber(evaluationCounts.total) + " 个候选完成" : "等待逐样本反馈评测", evaluationCounts.failed ? formatNumber(evaluationCounts.failed) + " 个失败" : "迭代训练反馈分区", run),
      roundResearchChainStep(6, "独立评审", judgeStatus, judgeCounts.total ? formatNumber(judgeCounts.completed) + " / " + formatNumber(judgeCounts.total) + " 个候选完成" : "等待独立模型评审", judgeCounts.failed ? formatNumber(judgeCounts.failed) + " 个未通过" : "与策略模型角色分离", run)
    ].join("");
    return "<section class=\"round-research-evidence\" aria-label=\"分析总结、最终方案、实现与测试证据链\"><div class=\"round-research-heading\"><div><span>研究迭代证据</span><strong>分析总结 → 最终方案 → 实现 → 测试</strong></div><code title=\"" + escapeHTML(iteration.iteration_digest || "") + "\">" + escapeHTML(shortId(iteration.iteration_digest || "未生成迭代校验值")) + "</code></div><ol class=\"round-research-chain\">" + steps + "</ol></section>";
  }
  function renderRoundStages() {
    var run = state.activeRun;
    var rounds = run && Array.isArray(run.rounds) ? run.rounds : [];
    var node = $("#round-stage-list");
    if (!rounds.length) { node.innerHTML = "<div class=\"empty-state\">运行提交首个提案后，将在这里展示逐轮阶段。</div>"; return; }
    // The backend emits one canonical six-stage schema.  Autonomous mode
    // changes the labels and explanatory copy, not the event keys; this keeps
    // the UI from showing a completed proposal as an unfinished fake stage.
    var autonomous = Boolean(run && run.configuration && run.configuration.autonomous_mode);
    var stageLabels = autonomous
      ? { proposal: "调研后提案", candidate: "宿主能力编译", training: "候选训练", evaluation: "科学反馈评测", judge: "独立评审", decision: "轮末优化决策" }
      : { proposal: "提案", candidate: "候选生成", training: "训练", evaluation: "评测", judge: "独立评审", decision: "保留决策" };
    node.innerHTML = rounds.slice().sort(function (left, right) { return Number(left.generation || 0) - Number(right.generation || 0); }).map(function (round) {
      var parentId = round.parent_id || round.parent_candidate_id || "当前基线";
      var candidates = Array.isArray(round.candidates) ? round.candidates : [];
      var candidateRows = candidates.length ? candidates.slice().sort(function (left, right) { return Number(left.slot_index || 0) - Number(right.slot_index || 0); }).map(function (candidate) {
        var stages = Object.keys(stageLabels).map(function (key) {
          var value = roundStageStatus(candidate, key, run);
          return "<div class=\"round-stage-step " + roundStageClass(key, value, run) + "\"><span>" + stageLabels[key] + "</span><strong>" + roundStageText(key, value, run) + "</strong></div>";
        }).join("");
        return "<div class=\"round-candidate-line\"><div class=\"round-candidate-meta\"><strong>槽位 " + escapeHTML(Number(candidate.slot_index || 0) + 1) + "</strong><code title=\"" + escapeHTML(candidate.candidate_id || "") + "\">" + escapeHTML(shortId(candidate.candidate_id || "候选尚未生成")) + "</code><span>排名 " + escapeHTML(candidate.rank || "—") + " · 得分 " + escapeHTML(formatNumber(candidate.score)) + "</span><span title=\"" + escapeHTML(selectionReasonText(candidate.selection_reason)) + "\">" + escapeHTML(compactTechnicalText(selectionReasonText(candidate.selection_reason))) + "</span></div><div class=\"round-stage-track\">" + stages + "</div></div>";
      }).join("") : "<div class=\"empty-state\">候选批次尚未生成。</div>";
      return "<section class=\"generation-round\"><div class=\"generation-round-heading\"><div><strong>第 " + escapeHTML(round.generation || "—") + " 轮</strong><span>候选 " + escapeHTML(round.candidate_count || 0) + " / " + escapeHTML(round.batch_size || 1) + " · 符合搜索条件 " + escapeHTML(round.eligible_count || 0) + "</span><span title=\"" + escapeHTML(parentId) + "\">共同父方案：" + escapeHTML(shortId(parentId)) + "</span></div><span class=\"pill " + generationOutcomeClass(round.decision) + "\">" + escapeHTML(generationOutcomeText(round.decision, round)) + "</span></div>" + renderRoundKnowledge(round) + renderRoundResearchEvidence(round, run) + "<div class=\"generation-analysis\"><div><span>下一轮搜索版本</span><strong title=\"" + escapeHTML(round.selected_candidate_id || round.search_parent_candidate_id || "") + "\">" + escapeHTML(shortId(round.selected_candidate_id || round.search_parent_candidate_id || "沿用原版本")) + "</strong></div><div><span>严格认证版本</span><strong title=\"" + escapeHTML(round.champion_candidate_id || "") + "\">" + escapeHTML(shortId(round.champion_candidate_id || "本轮未通过")) + "</strong></div><div><span>主要弱点</span><strong>" + escapeHTML(generationWeaknessText(round)) + "</strong></div><div><span>选择依据</span><strong title=\"" + escapeHTML(round.selection_reason || "") + "\">" + escapeHTML(selectionReasonText(round.selection_reason || "等待本轮完成")) + "</strong></div><div><span>下一轮重点</span><strong>" + escapeHTML(humanizeTechnicalText(round.next_generation_focus || "等待本轮分析")) + "</strong></div></div><div class=\"round-candidate-list\">" + candidateRows + "</div></section>";
    }).join("");
  }

  function autonomyCurrentRound(run) {
    var rounds = run && Array.isArray(run.rounds) ? run.rounds : [];
    return rounds.slice().sort(function (left, right) { return Number(right.generation || 0) - Number(left.generation || 0); })[0] || {};
  }
  function autonomyResearchStageEvent(run, round) {
    var displayGeneration = Number(round && round.generation);
    if (!Number.isFinite(displayGeneration) || displayGeneration < 1) {
      displayGeneration = Number(run && run.execution_progress && run.execution_progress.current_generation || Number(run && run.generation || 0) + 1);
    }
    var eventGeneration = Math.max(0, displayGeneration - 1);
    var matches = (Array.isArray(state.events) ? state.events : []).filter(function (event) {
      var payload = payloadOf(event);
      return String(event && event.type || "").toLowerCase() === "stage.recorded"
        && Number(payload.generation) === eventGeneration
        && String(payload.stage || "").toLowerCase() === "research";
    });
    matches.sort(function (left, right) {
      var leftSeq = Number(left && left.seq);
      var rightSeq = Number(right && right.seq);
      if (Number.isFinite(leftSeq) && Number.isFinite(rightSeq) && leftSeq !== rightSeq) { return rightSeq - leftSeq; }
      return (Date.parse(right && right.occurred_at || "") || 0) - (Date.parse(left && left.occurred_at || "") || 0);
    });
    return matches[0] || null;
  }
  function autonomyResearchRequestPolicy(run) {
    var configuration = run && run.configuration && typeof run.configuration === "object" ? run.configuration : {};
    var modelId = configuration.strategy_model_id || configuration.policy_model_id;
    var model = typeof catalogItem === "function" ? catalogItem("dsh_models", modelId) || catalogItem("models", modelId) : null;
    var connection = model && model.connection && typeof model.connection === "object" ? model.connection : {};
    var policy = connection.request_policy && typeof connection.request_policy === "object" ? connection.request_policy : model && model.request_policy && typeof model.request_policy === "object" ? model.request_policy : {};
    var lastRequest = connection.last_request && typeof connection.last_request === "object" ? connection.last_request : {};
    var timeoutSeconds = Number(policy.timeout_seconds != null ? policy.timeout_seconds : lastRequest.timeout_seconds);
    var maxAttempts = Number(policy.max_attempts);
    return {
      timeoutSeconds: Number.isFinite(timeoutSeconds) && timeoutSeconds > 0 ? timeoutSeconds : null,
      maxAttempts: Number.isInteger(maxAttempts) && maxAttempts > 0 ? maxAttempts : null
    };
  }
  function autonomyWaitDurationText(elapsedMs) {
    var totalSeconds = Math.max(0, Math.floor(Number(elapsedMs) / 1000));
    if (!Number.isFinite(totalSeconds)) { return ""; }
    if (totalSeconds < 60) { return totalSeconds + " 秒"; }
    var totalMinutes = Math.floor(totalSeconds / 60);
    var seconds = totalSeconds % 60;
    if (totalMinutes < 60) { return totalMinutes + " 分 " + seconds + " 秒"; }
    var hours = Math.floor(totalMinutes / 60);
    var minutes = totalMinutes % 60;
    return hours + " 小时 " + minutes + " 分";
  }
  function autonomyTimeoutDurationText(timeoutSeconds) {
    var seconds = Number(timeoutSeconds);
    if (!Number.isFinite(seconds) || seconds <= 0) { return ""; }
    if (seconds < 60) { return formatNumber(seconds, Number.isInteger(seconds) ? 0 : 1) + " 秒"; }
    var wholeSeconds = Math.round(seconds);
    var hours = Math.floor(wholeSeconds / 3600);
    var minutes = Math.floor(wholeSeconds % 3600 / 60);
    var remainder = wholeSeconds % 60;
    if (hours > 0) { return hours + " 小时" + (minutes ? " " + minutes + " 分" : "") + (remainder ? " " + remainder + " 秒" : ""); }
    return minutes + " 分钟" + (remainder ? " " + remainder + " 秒" : "");
  }
  function autonomyResearchStatus(run, current) {
    var iteration = current && current.research_iteration && typeof current.research_iteration === "object" ? current.research_iteration : null;
    var event = autonomyResearchStageEvent(run, current);
    var eventStatus = String(payloadOf(event).status || "").toLowerCase();
    var progress = run && run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : {};
    var phase = String(progress.phase || progress.current_stage || "").toLowerCase();
    if (iteration || ["completed", "done", "recorded", "passed"].indexOf(eventStatus) >= 0) { return "completed"; }
    var runStatus = String(run && run.status || "").toLowerCase();
    if (runStatus === "paused") { return "paused"; }
    if (executionRunIsTerminal(run)) { return runStatus === "failed" || runStatus === "cancelled" ? "aborted" : "not_recorded"; }
    if (!executionRunAllowsLiveStatus(run)) { return "not_recorded"; }
    if (["running", "started", "in_progress"].indexOf(eventStatus) >= 0 || phase === "research") { return "running"; }
    if (phase === "gateway_retry" && progress.retry_wait) { return "running"; }
    if (["failed", "error"].indexOf(eventStatus) >= 0) { return "failed"; }
    return (state.pendingAction === "advance" || state.pendingAction === "auto-advance") ? "running" : "pending";
  }
  function autonomyResearchPresentation(run, current) {
    var status = autonomyResearchStatus(run, current);
    var iteration = current && current.research_iteration && typeof current.research_iteration === "object" ? current.research_iteration : null;
    var event = autonomyResearchStageEvent(run, current);
    var progress = run && run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : {};
    var phase = String(progress.phase || progress.current_stage || "").toLowerCase();
    var knowledgeFrozen = Boolean(current && current.knowledge);
    var terminal = ["completed", "cancelled", "failed"].indexOf(String(run && run.status || "").toLowerCase()) >= 0;
    var policy = autonomyResearchRequestPolicy(run);
    var startedAt = Date.parse(event && event.occurred_at || current && current.timing && current.timing.started_at || "");
    var elapsedMs = Number.isFinite(startedAt) && startedAt <= Date.now() ? Date.now() - startedAt : null;
    var retrying = executionRunAllowsLiveStatus(run) && status === "running" && phase === "gateway_retry" && Boolean(progress.retry_wait);
    var waitingRemote = executionRunAllowsLiveStatus(run) && status === "running" && !retrying;
    var detail;
    if (status === "completed") {
      detail = iteration && iteration.status === "model_generated" ? "远端研究方案已返回并冻结"
        : iteration && iteration.status === "initial_frozen" ? "首轮研究方案已冻结"
          : iteration && iteration.status === "recovered_existing_proposal" ? "已从现有提案恢复研究方案"
            : iteration && ["host_fallback", "unavailable"].indexOf(String(iteration.status || "")) >= 0 ? "远端调研不可用，已记录受限降级方案"
              : iteration ? "研究方案已生成并冻结" : "远端调研完成信号已记录";
    } else if (status === "running") {
      detail = retrying
        ? (knowledgeFrozen ? "知识快照已冻结，远端研究请求等待网关重试" : "远端研究请求等待网关重试")
        : (knowledgeFrozen ? "知识快照已冻结，正在等待远端模型响应" : "正在等待远端模型响应");
      if (elapsedMs != null) { detail += " · 已等待 " + autonomyWaitDurationText(elapsedMs); }
      if (policy.timeoutSeconds != null) {
        detail += " · 目录策略单次调用超时上限 " + autonomyTimeoutDurationText(policy.timeoutSeconds);
        if (policy.maxAttempts != null && policy.maxAttempts > 1) { detail += "，最多 " + formatNumber(policy.maxAttempts) + " 次尝试"; }
      }
    } else if (status === "failed") {
      detail = "远端研究请求失败，等待既定恢复策略处理";
    } else if (status === "paused") {
      detail = knowledgeFrozen ? "知识快照已冻结；远端调研已暂停" : "远端调研已暂停";
    } else if (status === "aborted") {
      detail = knowledgeFrozen ? "知识快照已保留；远端调研已中止" : "远端调研已中止";
    } else if (status === "not_recorded") {
      detail = knowledgeFrozen ? "知识快照已冻结；该运行未封存远端调研完成记录" : "该运行未封存远端调研完成记录";
    } else if (knowledgeFrozen) {
      detail = terminal ? "知识快照已冻结；该运行未记录远端调研完成信号" : "知识快照已冻结，等待发起远端研究请求";
    } else {
      detail = "等待检索知识并发起远端研究请求";
    }
    return {
      status: status,
      detail: detail,
      statusText: retrying ? "等待重试" : waitingRemote ? "等待远端响应" : autonomyStepText(status),
      waitingRemote: waitingRemote,
      retrying: retrying
    };
  }
  function autonomyStepStatus(run, key) {
    if (!run) { return "pending"; }
    var current = autonomyCurrentRound(run);
    var stages = current.stages && typeof current.stages === "object" ? current.stages : {};
    var candidates = Array.isArray(current.candidates) ? current.candidates : [];
    function statusOf(value) {
      if (value && typeof value === "object") { value = value.status || value.state || value.result; }
      return value ? String(value).toLowerCase() : "pending";
    }
    function aggregate(keys) {
      var values = [];
      keys.forEach(function (stage) {
        if (stages[stage] != null) { values.push(statusOf(stages[stage])); }
        candidates.forEach(function (candidate) {
          if (candidate.stages && candidate.stages[stage] != null) { values.push(statusOf(candidate.stages[stage])); }
        });
      });
      if (values.indexOf("failed") >= 0 || values.indexOf("error") >= 0) { return "failed"; }
      if (values.indexOf("running") >= 0 || values.indexOf("started") >= 0 || values.indexOf("in_progress") >= 0) { return "running"; }
      if (values.length && values.every(function (value) { return ["completed", "done", "recorded", "passed", "accepted", "approved", "skipped", "duplicate"].indexOf(value) >= 0; })) { return "completed"; }
      return (state.pendingAction === "advance" || state.pendingAction === "auto-advance") && current.generation === Number(run.generation || 0) + 1 ? "running" : "pending";
    }
    if (key === "research") {
      return autonomyResearchStatus(run, current);
    }
    if (key === "implementation") { return aggregate(["proposal", "candidate"]); }
    if (key === "evaluation") { return aggregate(["training", "evaluation", "judge"]); }
    if (key === "optimization") { return aggregate(["decision"]); }
    return "pending";
  }
  function autonomyStepText(status) {
    var normalized = String(status || "pending").toLowerCase();
    if (["completed", "done", "recorded", "passed", "accepted", "approved"].indexOf(normalized) >= 0) { return "已完成"; }
    if (["failed", "error", "rejected"].indexOf(normalized) >= 0) { return "需要重试"; }
    if (normalized === "paused") { return "已暂停"; }
    if (normalized === "aborted") { return "已中止"; }
    if (normalized === "not_recorded") { return "未封存"; }
    if (["running", "in_progress", "evaluating", "started"].indexOf(normalized) >= 0) { return "进行中"; }
    return "等待";
  }
  function executionHeartbeatState(run) {
    var progress = run && run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : {};
    var stage = progress.stage_progress && typeof progress.stage_progress === "object" ? progress.stage_progress : {};
    var dshActivity = progress.dsh_activity && typeof progress.dsh_activity === "object" ? progress.dsh_activity : {};
    var timestamps = [progress.last_event_at, stage.updated_at, dshActivity.updated_at].map(function (value) { return Date.parse(value || ""); }).filter(Number.isFinite);
    var latest = timestamps.length ? Math.max.apply(Math, timestamps) : null;
    var ageSeconds = latest == null ? null : Math.max(0, (Date.now() - latest) / 1000);
    var stalled = String(run && run.status || "").toLowerCase() === "running" && ageSeconds != null && ageSeconds > 120;
    return {
      timestamp: latest == null ? null : new Date(latest).toISOString(),
      age_seconds: ageSeconds,
      age_text: ageSeconds == null ? "" : ageSeconds < 60 ? formatNumber(Math.floor(ageSeconds)) + " 秒前" : compactDuration(ageSeconds) + "前",
      stalled: stalled
    };
  }
  function renderAutonomyProgress(run) {
    var node = $("#autonomy-progress");
    var statusNode = $("#autonomy-progress-status");
    if (!node || !statusNode) { return; }
    if (!run) {
      statusNode.className = "pill pill-neutral";
      statusNode.textContent = "等待运行";
      node.innerHTML = "<div class=\"empty-state\">创建运行后，策略模型会检索知识并形成受限研究计划；宿主随后编译已登记能力、评测候选并执行轮末搜索决策。</div>";
      return;
    }
    var current = autonomyCurrentRound(run);
    var research = autonomyResearchPresentation(run, current);
    var steps = [
      ["research", "自主调研", research.detail, research.statusText],
      ["implementation", "受限能力编译", "只采用宿主已登记的预测器、参数和策略"],
      ["evaluation", "科学评测与独立评审", "在训练反馈分区比较误差、基线和约束"],
      ["optimization", "轮末搜索决策", "固定规则分析结果并选择下一轮父方案"]
    ];
    var statuses = steps.map(function (step) {
      return executionStatusForRun(run, step[0] === "research" ? research.status : autonomyStepStatus(run, step[0]));
    });
    var active = statuses.indexOf("running");
    var runStatus = String(run.status || "").toLowerCase();
    var paused = runStatus === "paused";
    var stageProgress = run.execution_progress && run.execution_progress.stage_progress;
    if ((executionRunAllowsLiveStatus(run) || paused) && stageProgress && ["screening", "formal_batch", "holdout"].indexOf(stageProgress.evaluation_phase) >= 0) {
      statuses[2] = paused ? "paused" : "running";
      active = 2;
    }
    var pausedDrained = paused && stageProgress && stageProgress.progress_kind === "drained";
    var overall = runStatus === "failed" ? "failed" : runStatus === "cancelled" ? "cancelled" : runStatus === "completed" ? "completed" : paused ? "paused" : active >= 0 ? "running" : run.generation > 0 ? "completed" : "pending";
    var explicitProgress = run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : {};
    var dshActivity = executionRunAllowsLiveStatus(run) && explicitProgress.dsh_activity && typeof explicitProgress.dsh_activity === "object"
      ? executionDshActivityPresentation(explicitProgress.dsh_activity)
      : null;
    var heartbeatState = executionHeartbeatState(run);
    var heartbeatStalled = heartbeatState.stalled && !research.retrying && !research.waitingRemote && !dshActivity;
    var automatic = runHasContinuousAutoProgress(run);
    statusNode.className = "pill " + (heartbeatStalled ? "pill-amber" : overall === "completed" ? "pill-green" : overall === "running" ? "pill-blue" : overall === "paused" || overall === "cancelled" ? "pill-amber" : overall === "failed" ? "pill-red" : "pill-neutral");
    statusNode.textContent = heartbeatStalled ? "疑似停滞，正在核验" : overall === "completed" ? "本轮已完成" : overall === "running" ? research.retrying ? "远端请求等待重试" : research.waitingRemote ? "等待远端模型响应" : dshActivity && dshActivity.statusText || "模型执行中" : overall === "paused" ? (pausedDrained ? "已暂停，请求已排空" : "已暂停") : overall === "cancelled" ? "已取消" : overall === "failed" ? "执行失败" : automatic ? "后台排队中" : "等待推进";
    node.innerHTML = steps.map(function (step, index) {
      var status = statuses[index];
      var tone = executionStatusClass(status);
      var statusText = step[3] || autonomyStepText(status);
      return "<div class=\"autonomy-progress-step " + tone + "\"><span>0" + (index + 1) + "</span><strong>" + escapeHTML(step[1]) + "</strong><small>" + escapeHTML(step[2]) + "</small><span class=\"pill " + (tone === "is-complete" ? "pill-green" : tone === "is-running" ? "pill-blue" : tone === "is-paused" ? "pill-amber" : tone === "is-failed" ? "pill-red" : "pill-neutral") + "\">" + escapeHTML(statusText) + "</span></div>";
    }).join("");
  }

  var executionStageKeys = ["proposal", "candidate", "training", "evaluation", "judge", "decision"];
  var executionStageLabels = {
    proposal: "方案提案", candidate: "能力编译", training: "候选训练",
    evaluation: "样本评测", judge: "独立评审", decision: "轮末决策"
  };
  function executionEvidenceStatus(run) {
    var diagnostics = run && run.execution_diagnostics && typeof run.execution_diagnostics === "object" ? run.execution_diagnostics : {};
    return String(diagnostics.execution_evidence_status || "").toLowerCase();
  }
  function executionHasRetainedEvidence(run) {
    return ["retained_partial", "aborted_partial", "aborted", "mixed_partial"].indexOf(executionEvidenceStatus(run)) >= 0;
  }
  function executionEvidenceQualifier(run) {
    var evidenceStatus = executionEvidenceStatus(run);
    if (evidenceStatus === "retained_partial") { return "部分证据已保留"; }
    if (evidenceStatus === "aborted_partial" || evidenceStatus === "aborted") { return "中止前证据已保留"; }
    if (evidenceStatus === "mixed_partial") { return "部分证据已记录"; }
    var runStatus = String(run && run.status || "").toLowerCase();
    if (runStatus === "paused") { return "暂停前记录"; }
    if (executionRunIsTerminal(run)) { return "评测记录已结束"; }
    return "";
  }
  function executionRunIsTerminal(run) {
    return ["completed", "failed", "cancelled"].indexOf(String(run && run.status || "").toLowerCase()) >= 0;
  }
  function executionRunAllowsLiveStatus(run) {
    var status = String(run && run.status || "").toLowerCase();
    var evidenceStatus = executionEvidenceStatus(run);
    return ["running", "starting"].indexOf(status) >= 0 && ["retained_partial", "aborted_partial", "aborted", "mixed_partial"].indexOf(evidenceStatus) < 0;
  }
  function executionStatusForRun(run, value) {
    if (!run) { return value; }
    var status = String(executionStatusValue(value)).toLowerCase();
    if (["running", "started", "in_progress", "evaluating"].indexOf(status) < 0 || executionRunAllowsLiveStatus(run)) { return value; }
    var runStatus = String(run.status || "").toLowerCase();
    if (runStatus === "paused") { return "paused"; }
    if (runStatus === "failed" || runStatus === "cancelled") { return "aborted"; }
    return "not_recorded";
  }
  function executionStatusValue(value) {
    if (value && typeof value === "object") { return value.status || value.state || value.result || value.decision || "pending"; }
    return value || "pending";
  }
  function executionStatusClass(value) {
    var status = String(executionStatusValue(value)).toLowerCase();
    if (["completed", "done", "recorded", "passed", "accepted", "approved", "retained", "promoted"].indexOf(status) >= 0) { return "is-complete"; }
    if (["running", "started", "in_progress", "evaluating"].indexOf(status) >= 0) { return "is-running"; }
    if (status === "paused") { return "is-paused"; }
    if (["failed", "error", "rejected", "declined", "aborted"].indexOf(status) >= 0) { return "is-failed"; }
    return "is-pending";
  }
  function executionStageText(value, key) {
    var status = String(executionStatusValue(value)).toLowerCase();
    if (key === "decision" && ["approved", "accepted", "retained", "promoted"].indexOf(status) >= 0) { return "训练反馈搜索保留"; }
    if (key === "decision" && ["rejected", "declined", "denied"].indexOf(status) >= 0) { return "未保留"; }
    if (["completed", "done", "recorded", "passed", "accepted", "approved", "retained", "promoted"].indexOf(status) >= 0) { return "已完成"; }
    if (["running", "started", "in_progress", "evaluating"].indexOf(status) >= 0) { return "进行中"; }
    if (status === "paused") { return "已暂停"; }
    if (status === "aborted") { return "已中止"; }
    if (status === "not_recorded") { return "未封存"; }
    if (["failed", "error", "rejected", "declined"].indexOf(status) >= 0) { return "失败"; }
    if (["skipped", "duplicate"].indexOf(status) >= 0) { return "已跳过"; }
    return "等待";
  }
  function executionCandidateStatusText(value) {
    var status = String(value || "pending").toLowerCase();
    if (status === "paused") { return "已暂停"; }
    if (status === "aborted") { return "已中止"; }
    if (status === "not_recorded") { return "未封存"; }
    return candidateStatusText(value);
  }
  function executionCandidateStatusClass(value) {
    var status = String(value || "pending").toLowerCase();
    if (status === "paused") { return "pill-amber"; }
    if (status === "aborted") { return "pill-red"; }
    if (status === "not_recorded") { return "pill-neutral"; }
    return candidateStatusClass(value);
  }
  function executionRound(run) {
    var rounds = run && Array.isArray(run.rounds) ? run.rounds.slice() : [];
    rounds.sort(function (left, right) { return Number(right.generation || 0) - Number(left.generation || 0); });
    var runStatus = String(run && run.status || "").toLowerCase();
    // run.generation is the number of completed zero-based generations,
    // while projected rounds use public one-based generation numbers.  Once
    // a run is terminal, the newest durable round is authoritative even when
    // that round stopped before GenerationAdvanced was recorded.
    if (executionRunIsTerminal(run)) { return rounds[0] || null; }
    if ((executionRunAllowsLiveStatus(run) || runStatus === "paused") && state.pendingAction !== "advance" && state.pendingAction !== "auto-advance") {
      var activeRound = rounds.find(function (round) {
        var values = [];
        if (round.stages && typeof round.stages === "object") { values = values.concat(Object.keys(round.stages).map(function (key) { return round.stages[key]; })); }
        (Array.isArray(round.candidates) ? round.candidates : []).forEach(function (candidate) {
          if (candidate.stages && typeof candidate.stages === "object") { values = values.concat(Object.keys(candidate.stages).map(function (key) { return candidate.stages[key]; })); }
        });
        return values.some(function (value) { return ["running", "started", "in_progress", "evaluating"].indexOf(String(executionStatusValue(value)).toLowerCase()) >= 0; });
      });
      if (activeRound) { return activeRound; }
    }
    var projectedGeneration = Number(run && run.execution_progress && run.execution_progress.current_generation);
    if (Number.isInteger(projectedGeneration) && projectedGeneration > 0) {
      var projectedRound = rounds.find(function (round) { return Number(round.generation || 0) === projectedGeneration; });
      if (projectedRound) { return projectedRound; }
    }
    var target = Number(run && run.generation || 0) + 1;
    return rounds.find(function (round) { return Number(round.generation || 0) === target; }) || rounds[0] || null;
  }
  function executionCandidateHasActiveStage(candidate, round) {
    var source = candidate && (candidate.execution && candidate.execution.stages || candidate.stages);
    if (!source && round && Array.isArray(round.candidates)) {
      var row = round.candidates.find(function (item) { return item.candidate_id === candidate.id || item.id === candidate.id; });
      source = row && row.stages;
    }
    return source && Object.keys(source).some(function (key) { return ["running", "started", "in_progress", "evaluating"].indexOf(String(executionStatusValue(source[key])).toLowerCase()) >= 0; });
  }
  function executionCandidateFor(run, round) {
    if (!run) { return null; }
    var candidates = Array.isArray(run.candidates) ? run.candidates.slice() : [];
    var progress = run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : run.execution && typeof run.execution === "object" ? run.execution : {};
    var requestedId = progress.active_candidate_id || progress.current_candidate_id || progress.candidate_id;
    if (requestedId) {
      var requested = candidates.find(function (candidate) { return candidate.id === requestedId || candidate.candidate_id === requestedId; });
      if (requested) { return requested; }
    }
    var batchStage = String(progress.current_stage || progress.phase || "").toLowerCase();
    if (!requestedId && ["research", "proposal", "gateway_retry", "starting"].indexOf(batchStage) >= 0) {
      // These stages are generation-scoped until the service records a
      // candidate identity. Showing the prior generation's last candidate
      // here makes its samples and parameters look like live work.
      return null;
    }
    if (round && Array.isArray(round.candidates)) {
      var rows = round.candidates.slice().sort(function (left, right) { return Number(left.slot_index || 0) - Number(right.slot_index || 0); });
      var activeRow = executionRunAllowsLiveStatus(run) || String(run.status || "").toLowerCase() === "paused"
        ? rows.find(function (row) { return executionCandidateHasActiveStage({ id: row.candidate_id }, round); })
        : null;
      var row = activeRow || rows[rows.length - 1];
      if (row) {
        var matched = candidates.find(function (candidate) { return candidate.id === row.candidate_id || candidate.candidate_id === row.candidate_id; });
        if (matched) { return matched; }
        return Object.assign({ id: row.candidate_id || "candidate-pending", candidate_id: row.candidate_id }, row);
      }
    }
    candidates.sort(function (left, right) {
      return Number(right.generation || 0) - Number(left.generation || 0) || Number(right.slot_index || 0) - Number(left.slot_index || 0) || String(right.created_at || "").localeCompare(String(left.created_at || ""));
    });
    return candidates[0] || null;
  }
  function executionStageSource(candidate, round) {
    var source = candidate && candidate.execution && candidate.execution.stages || candidate && candidate.stages || null;
    if (!source && round && Array.isArray(round.candidates) && candidate) {
      var row = round.candidates.find(function (item) { return item.candidate_id === candidate.id || item.id === candidate.id; });
      source = row && row.stages;
    }
    source = source && typeof source === "object" ? Object.assign({}, source) : {};
    if (round && round.stages && typeof round.stages === "object") {
      executionStageKeys.forEach(function (key) { if (source[key] == null && round.stages[key] != null) { source[key] = round.stages[key]; } });
    }
    var progress = candidate && candidate.execution && typeof candidate.execution === "object" ? candidate.execution : {};
    if (progress.current_stage && source[progress.current_stage] == null) { source[progress.current_stage] = "running"; }
    return source;
  }
  function executionStageValues(candidate, round, run) {
    var source = executionStageSource(candidate, round);
    var progress = run && run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : {};
    var stageProgress = progress.stage_progress && typeof progress.stage_progress === "object" ? progress.stage_progress : {};
    var adaptivePhase = String(stageProgress.evaluation_phase || "").toLowerCase();
    if ((executionRunAllowsLiveStatus(run) || String(run && run.status || "").toLowerCase() === "paused") && ["screening", "formal_batch", "holdout"].indexOf(adaptivePhase) >= 0) {
      if (["pending", "waiting", ""].indexOf(String(source.training || "pending").toLowerCase()) >= 0) { source.training = "completed"; }
      source.evaluation = String(run.status || "").toLowerCase() === "paused" ? "paused" : "running";
    }
    return executionStageKeys.map(function (key) { return { key: key, value: source[key] || "pending" }; });
  }
  function executionHasRunningStage(stages) {
    return stages.some(function (item) { return executionStatusClass(item.value) === "is-running"; });
  }
  function executionCompletedStageCount(stages) {
    return stages.filter(function (item) { return executionStatusClass(item.value) === "is-complete"; }).length;
  }
  function executionProgress(run, round, stages) {
    var total = Math.max(1, Number(run && (run.total_generations || run.rounds) || 1));
    var explicit = run && run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : {};
    var explicitPercent = Number(explicit.overall_progress_percent != null ? explicit.overall_progress_percent : explicit.progress_percent != null ? explicit.progress_percent : explicit.percent);
    var explicitCompletedSteps = Number(explicit.completed_steps);
    var explicitTotalSteps = Number(explicit.total_steps);
    if (Number.isFinite(explicitCompletedSteps) && Number.isFinite(explicitTotalSteps) && explicitTotalSteps > 0 && (!Number.isFinite(explicitPercent) || explicitPercent >= 100 && explicitCompletedSteps < explicitTotalSteps)) {
      explicitPercent = Math.max(0, Math.min(100, explicitCompletedSteps / explicitTotalSteps * 100));
    }
    var runStatus = String(run && run.status || "").toLowerCase();
    var inactiveControlState = runStatus === "paused" || executionRunIsTerminal(run) || !executionRunAllowsLiveStatus(run);
    var pending = !inactiveControlState && (state.pendingAction === "advance" || state.pendingAction === "auto-advance");
    var explicitPhase = String(explicit.phase || explicit.current_stage || "").toLowerCase();
    var explicitActive = ["research", "proposal", "candidate", "training", "evaluation", "judge", "decision", "gateway_retry", "starting", "running", "in_progress"].indexOf(explicitPhase) >= 0;
    var active = !inactiveControlState && (pending || explicitActive || executionHasRunningStage(stages) || runStatus === "starting");
    var percent;
    if (Number.isFinite(explicitPercent)) {
      percent = Math.max(0, Math.min(100, explicitPercent));
    } else {
      var completedRounds = Math.max(0, Math.min(total, Number(run && run.generation || 0)));
      if (active) {
        var activeGeneration = Number(round && round.generation || completedRounds + 1);
        var base = Math.max(0, Math.min(total - 1, activeGeneration - 1));
        var completedStages = executionCompletedStageCount(stages);
        var fraction = completedStages / executionStageKeys.length;
        if (executionHasRunningStage(stages)) { fraction = Math.min(0.98, (completedStages + 0.5) / executionStageKeys.length); }
        if (pending && !round) { fraction = 0.06; }
        percent = ((base + fraction) / total) * 100;
      } else {
        percent = (completedRounds / total) * 100;
      }
    }
    var completed = Math.min(total, Math.max(0, Math.floor(percent / 100 * total + 1e-6)));
    return { total: total, percent: percent, active: active, completed: completed, pending: pending };
  }
  function executionEpochProgressPercent(run) {
    var progress = run && run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : {};
    var stage = progress.stage_progress && typeof progress.stage_progress === "object" ? progress.stage_progress : {};
    var raw = progress.epoch_progress_percent != null ? progress.epoch_progress_percent : stage.epoch_progress_percent;
    var value = Number(raw);
    return raw != null && Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : null;
  }
  function executionActiveStageElapsedMs(run, stage) {
    var targetStage = String(stage || "").toLowerCase();
    if (!run || !targetStage) { return null; }
    var generation = Number(run.generation || 0);
    var latest = (Array.isArray(state.events) ? state.events : []).find(function (event) {
      var payload = payloadOf(event);
      return String(event && event.type || "").toLowerCase() === "stage.recorded"
        && Number(payload.generation) === generation
        && String(payload.stage || "").toLowerCase() === targetStage;
    });
    var payload = payloadOf(latest);
    if (["started", "running", "in_progress"].indexOf(String(payload.status || payload.state || "").toLowerCase()) < 0) { return null; }
    var startedAt = Date.parse(latest && latest.occurred_at || "");
    return Number.isFinite(startedAt) && startedAt <= Date.now() ? Date.now() - startedAt : null;
  }
  function executionElapsedText(elapsedMs, active) {
    if (elapsedMs == null || !Number.isFinite(Number(elapsedMs))) { return active ? " · 本轮计时中" : ""; }
    return " · 本轮耗时 " + formatNumber(Number(elapsedMs) / 1000, 1) + " 秒";
  }
  function executionSafeValue(value) {
    if (value == null) { return "—"; }
    if (typeof value === "boolean") { return value ? "是" : "否"; }
    if (typeof value === "number") { return formatNumber(value); }
    return compactTechnicalText(String(value));
  }
  function executionInputSummary(row) {
    var input = row && (row.input_summary || row.inputs || row.features || row.input);
    if (!input) { return "输入：按冻结数据分区取样"; }
    if (typeof input === "string") { return "输入：" + compactTechnicalText(input); }
    if (typeof input !== "object") { return "输入：" + executionSafeValue(input); }
    var keys = Object.keys(input).filter(function (key) { return !/(token|secret|password|reasoning|prompt|raw|observed|predicted)/i.test(key); }).slice(0, 3);
    return keys.length ? "输入：" + keys.map(function (key) { return humanizeTechnicalText(key) + "=" + executionSafeValue(input[key]); }).join(" · ") : "输入：按冻结数据分区取样";
  }
  function executionRowUsesScoringFallback(row) {
    var source = String(row && row.prediction_source || "").toLowerCase();
    var status = String(row && (row.sample_status || row.sample_execution_status || row.status) || "").toLowerCase();
    return Boolean(row && row.scoring_fallback) || source === "scoring_fallback" || source === "failed_no_model_prediction" || ["failed", "error", "rejected", "timeout", "aborted", "cancelled"].indexOf(status) >= 0 || Boolean(row && row.model_prediction_available === false);
  }
  function executionPredictionRows(candidate, run) {
    // The paged sample endpoint is the live source while a candidate is
    // evaluating.  Prefer it over the bounded projection preview so rows
    // arriving in microbatches are visible in this monitor as well as in the
    // candidate workspace.
    var liveCandidateId = candidate && (candidate.id || candidate.candidate_id);
    var livePage = state.candidateSamplePage;
    if (livePage && livePage.rows && typeof candidateSampleSelectionMatches === "function" && candidateSampleSelectionMatches(run && run.id, liveCandidateId)) {
      return Array.isArray(livePage.rows) ? livePage.rows : [];
    }
    var metrics = candidate && candidate.metrics && typeof candidate.metrics === "object" ? candidate.metrics : {};
    var execution = candidate && candidate.execution && typeof candidate.execution === "object" ? candidate.execution : {};
    var sources = [execution.inference_trace, execution.prediction_trace, candidate && candidate.inference_trace, candidate && candidate.prediction_trace, metrics.inference_trace, metrics.prediction_preview, metrics.predictions, run && run.inference_trace];
    for (var index = 0; index < sources.length; index += 1) {
      var source = sources[index];
      if (Array.isArray(source)) { return source; }
      if (source && typeof source === "object" && Array.isArray(source.rows)) { return source.rows; }
    }
    return [];
  }
  function executionPredictionTrace(candidate, run) {
    var liveCandidateId = candidate && (candidate.id || candidate.candidate_id);
    var livePage = state.candidateSamplePage;
    if (livePage && livePage.rows && typeof candidateSampleSelectionMatches === "function" && candidateSampleSelectionMatches(run && run.id, liveCandidateId)) {
      return {
        rows: livePage.rows,
        sample_count: livePage.total,
        truncated: livePage.truncated === true,
        status: livePage.status || (livePage.complete === true ? "completed" : "running")
      };
    }
    var metrics = candidate && candidate.metrics && typeof candidate.metrics === "object" ? candidate.metrics : {};
    var execution = candidate && candidate.execution && typeof candidate.execution === "object" ? candidate.execution : {};
    var sources = [execution.inference_trace, execution.prediction_trace, candidate && candidate.inference_trace, candidate && candidate.prediction_trace, metrics.inference_trace, metrics.prediction_preview, run && run.inference_trace];
    for (var index = 0; index < sources.length; index += 1) {
      var source = sources[index];
      if (source && typeof source === "object" && !Array.isArray(source) && Array.isArray(source.rows)) { return source; }
    }
    return null;
  }
  function executionPredictionMethod(candidate, index, row) {
    var trace = candidate && candidate.inference_trace && typeof candidate.inference_trace === "object" ? candidate.inference_trace : candidate && candidate.execution && candidate.execution.inference_trace && typeof candidate.execution.inference_trace === "object" ? candidate.execution.inference_trace : {};
    var methods = Array.isArray(trace.method_steps) ? trace.method_steps : [];
    return row && (row.step_summary || row.inference_summary || row.reasoning_summary || row.method) || methods[index] || "冻结预测器完成一次前向推理";
  }
  function executionChangeValue(value) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      var before = value.before != null ? value.before : value.from;
      var after = value.after != null ? value.after : value.to;
      if (before != null || after != null) { return executionSafeValue(before) + " → " + executionSafeValue(after); }
      return Object.keys(value).slice(0, 3).map(function (key) { return humanizeTechnicalText(key) + "=" + executionSafeValue(value[key]); }).join("；");
    }
    return executionSafeValue(value);
  }
  function executionPlanSource(candidate) {
    var plan = candidate && (candidate.model_plan || candidate.modelPlan || candidate.plan);
    if (!plan || typeof plan !== "object") { return null; }
    return plan.plan && typeof plan.plan === "object" ? plan.plan : plan;
  }
  function renderExecutionPlan(candidate, run) {
    var node = $("#implementation-summary");
    var statusNode = $("#implementation-status");
    if (!node || !statusNode) { return; }
    if (!candidate) {
      statusNode.className = "pill pill-neutral";
      statusNode.textContent = "未生成";
      node.innerHTML = "<div class=\"empty-state\">候选提案生成后显示模型计划和参数修改。</div>";
      return;
    }
    var plan = executionPlanSource(candidate) || {};
    var team = plan.team && typeof plan.team === "object" ? plan.team : {};
    var prediction = plan.prediction_model && typeof plan.prediction_model === "object" ? plan.prediction_model : {};
    var strategy = plan.strategy && typeof plan.strategy === "object" ? plan.strategy : {};
    var changes = candidate.changes && typeof candidate.changes === "object" ? candidate.changes : {};
    var changeKeys = Object.keys(changes);
    var currentStage = candidate.execution && candidate.execution.current_stage;
    var runStatus = String(run && run.status || "").toLowerCase();
    var paused = runStatus === "paused";
    var abortedRun = runStatus === "failed" || runStatus === "cancelled";
    var terminal = executionRunIsTerminal(run);
    var liveAllowed = executionRunAllowsLiveStatus(run);
    var inProgress = liveAllowed && (currentStage || executionCandidateHasActiveStage(candidate, executionRound(run)));
    var partialRetained = executionHasRetainedEvidence(run);
    statusNode.className = "pill " + (paused ? "pill-amber" : abortedRun ? "pill-red" : partialRetained ? "pill-amber" : terminal ? "pill-green" : inProgress ? "pill-blue" : executionCandidateStatusClass(candidate.status || "pending"));
    statusNode.textContent = paused ? "已暂停" : abortedRun ? "已中止" : partialRetained ? executionEvidenceQualifier(run) : terminal ? "执行已结束" : inProgress ? "编译／执行中" : changeKeys.length ? "方案已生成" : "等待方案";
    var title = candidate.title || plan.title || "模型提出的候选方案";
    var rationale = candidate.rationale || plan.rationale || plan.summary || "模型尚未提供公开方案摘要。";
    var meta = [
      team.name || team.id ? "团队：" + (team.name || team.id) : "",
      prediction.name || prediction.id ? "预测模型：" + (prediction.name || prediction.id) : "",
      strategy.name || strategy.id ? "策略：" + (strategy.name || strategy.id) : ""
    ].filter(Boolean);
    var changesHtml = changeKeys.length ? changeKeys.map(function (key) {
      return "<div class=\"execution-change-row\"><span>" + escapeHTML(parameterLabels[key] || humanizeTechnicalText(key)) + "</span><strong>" + escapeHTML(executionChangeValue(changes[key])) + "</strong></div>";
    }).join("") : "<div class=\"empty-state\">本候选没有记录参数修改。</div>";
    node.innerHTML = "<p class=\"execution-plan-title\">" + escapeHTML(compactTechnicalText(title)) + "</p><p class=\"execution-plan-rationale\">" + escapeHTML(compactTechnicalText(String(rationale).slice(0, 360))) + "</p>" + (meta.length ? "<div class=\"execution-plan-meta\">" + meta.map(function (item) { return "<span>" + escapeHTML(compactTechnicalText(item)) + "</span>"; }).join("") + "</div>" : "") + "<div class=\"execution-change-list\">" + changesHtml + "</div>";
  }
  function renderExecutionCandidate(candidate, run, round, stages) {
    var node = $("#active-candidate-summary");
    var statusNode = $("#active-candidate-status");
    if (!node || !statusNode) { return; }
    if (!candidate) {
      statusNode.className = "pill pill-neutral";
      statusNode.textContent = "暂无";
      node.innerHTML = "<div class=\"empty-state\">运行开始后显示正在处理的候选方案。</div>";
      return;
    }
    var explicitProgress = run && run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : {};
    var candidateId = candidate.id || candidate.candidate_id;
    var projectedCandidateId = explicitProgress.current_candidate_id;
    var projectedCurrentStage = !projectedCandidateId || projectedCandidateId === candidateId ? explicitProgress.current_stage : null;
    var rawCurrentStage = projectedCurrentStage || candidate.execution && candidate.execution.current_stage;
    var runStatus = String(run && run.status || "").toLowerCase();
    var paused = runStatus === "paused";
    var abortedRun = runStatus === "failed" || runStatus === "cancelled";
    var terminal = executionRunIsTerminal(run);
    var partialRetained = executionHasRetainedEvidence(run);
    var rawActive = executionHasRunningStage(stages) || rawCurrentStage || state.pendingAction === "advance" || state.pendingAction === "auto-advance";
    var active = executionRunAllowsLiveStatus(run) && rawActive;
    var status = paused ? "已暂停" : abortedRun ? "已中止" : partialRetained ? executionEvidenceQualifier(run) : terminal ? "执行已结束" : active ? "执行中" : executionCandidateStatusText(candidate.status || "pending");
    statusNode.className = "pill " + (paused ? "pill-amber" : abortedRun ? "pill-red" : partialRetained ? "pill-amber" : terminal ? "pill-green" : active ? "pill-blue" : executionCandidateStatusClass(candidate.status || "pending"));
    statusNode.textContent = status;
    var liveAllowed = executionRunAllowsLiveStatus(run);
    var currentStage = liveAllowed || paused ? rawCurrentStage : null;
    var retryWait = liveAllowed && run && run.execution_progress && typeof run.execution_progress.retry_wait === "object" ? run.execution_progress.retry_wait : null;
    var stageText = retryWait ? "网关等待重试" : currentStage ? (evolutionStageLabels[currentStage] || executionStageLabels[currentStage] || currentStage) : stages.map(function (item) {
      var presented = executionStatusForRun(run, item.value);
      return executionStageText(presented, item.key) === "进行中" ? executionStageLabels[item.key] : "";
    }).filter(Boolean)[0] || (partialRetained ? executionEvidenceQualifier(run) : "当前阶段已记录");
    var configuration = run && run.configuration || {};
    var predictionModelId = candidate.execution && candidate.execution.prediction_model_id || configuration.prediction_model_id;
    var previewRows = executionPredictionRows(candidate, run);
    var validPreviewCount = previewRows.filter(function (row) { return !executionRowUsesScoringFallback(row); }).length;
    var penaltyPreviewCount = previewRows.length - validPreviewCount;
    var adaptiveProgress = explicitProgress.stage_progress && typeof explicitProgress.stage_progress === "object" ? explicitProgress.stage_progress : {};
    var adaptiveBatchText = Number.isFinite(Number(adaptiveProgress.batch_index)) && Number.isFinite(Number(adaptiveProgress.batch_count))
      ? "批次 " + formatNumber(adaptiveProgress.batch_index) + " / " + formatNumber(adaptiveProgress.batch_count) + " · 本轮已结算 " + formatNumber(adaptiveProgress.completed_samples || 0) + " / " + formatNumber(adaptiveProgress.total_samples || 0) + " 个预测时点"
      : "";
    var previewText = previewRows.length
      ? (validPreviewCount ? "有效预测 " + validPreviewCount + " 条" : "尚无有效预测") + (penaltyPreviewCount ? " · 失败惩罚占位 " + penaltyPreviewCount + " 条" : "")
      : adaptiveBatchText || (liveAllowed ? "等待样本评测" : "未保留样本预览");
    node.innerHTML = "<div class=\"execution-candidate-identity\"><strong>" + escapeHTML(shortId(candidate.id || candidate.candidate_id || "候选尚未生成")) + "</strong><code title=\"" + escapeHTML(candidate.id || candidate.candidate_id || "") + "\">第 " + escapeHTML(candidate.generation || round && round.generation || "—") + " 轮 · 槽位 " + escapeHTML(Number(candidate.slot_index || 0) + 1) + "</code><span>" + escapeHTML(stageText + " · " + previewText) + "</span></div><dl class=\"execution-key-values\"><div><dt>父方案</dt><dd title=\"" + escapeHTML(candidate.parent_id || "") + "\">" + escapeHTML(shortId(candidate.parent_id || "当前基线")) + "</dd></div><div><dt>预测模型</dt><dd title=\"" + escapeHTML(predictionModelId || "") + "\">" + escapeHTML(predictionModelReferenceLabel(predictionModelId)) + "</dd></div><div><dt>综合得分</dt><dd>" + escapeHTML(formatNumber(candidate.score)) + "</dd></div></dl><p class=\"execution-rationale\">" + escapeHTML(compactTechnicalText(String(candidate.rationale || "等待模型方案摘要。").slice(0, 260))) + "</p>";
  }
  function renderExecutionSamples(candidate, run) {
    var list = $("#sample-inference-list");
    var countNode = $("#sample-inference-count");
    if (!list || !countNode) { return; }
    var rows = executionPredictionRows(candidate, run);
    var trace = executionPredictionTrace(candidate, run);
    var total = trace && Number.isFinite(Number(trace.sample_count)) ? Number(trace.sample_count) : rows.length;
    var penaltyCount = rows.filter(executionRowUsesScoringFallback).length;
    var validPredictionCount = rows.length - penaltyCount;
    var countPrefix = total > rows.length ? formatNumber(rows.length) + " / " + formatNumber(total) + " 条" : formatNumber(rows.length) + " 条";
    countNode.textContent = countPrefix + (penaltyCount ? "（有效预测 " + formatNumber(validPredictionCount) + " · 失败惩罚占位 " + formatNumber(penaltyCount) + "）" : "");
    countNode.title = trace && trace.truncated === true ? "当前展示脱敏预览；完整样本由候选评测记录保留。" : "";
    if (!rows.length) {
      list.innerHTML = "<div class=\"empty-state\">候选完成评测后显示样本结果。</div>";
      return;
    }
    list.innerHTML = rows.map(function (row, index) {
      row = row && typeof row === "object" ? row : {};
      var scoringFallback = executionRowUsesScoringFallback(row);
      var predicted = !scoringFallback && row.predicted != null && row.predicted !== "" ? Number(row.predicted) : NaN;
      var observed = row.observed != null && row.observed !== "" ? Number(row.observed) : NaN;
      var baseline = row.baseline != null && row.baseline !== "" ? Number(row.baseline) : NaN;
      var error = !scoringFallback && row.error != null && row.error !== "" ? Number(row.error) : NaN;
      if (!Number.isFinite(error) && Number.isFinite(predicted) && Number.isFinite(observed)) { error = predicted - observed; }
      var baselineError = Number(row.baseline_error);
      if (!Number.isFinite(baselineError) && Number.isFinite(baseline) && Number.isFinite(observed)) { baselineError = baseline - observed; }
      var improved = !scoringFallback && Number.isFinite(error) && Number.isFinite(baselineError) ? Math.abs(error) <= Math.abs(baselineError) : null;
      var tone = scoringFallback ? "is-failed" : improved === true ? "is-improved" : improved === false ? "is-regressed" : "";
      var target = targetLabels[row.target] || row.target || "预测目标";
      var targetTime = row.target_timestamp != null ? row.target_timestamp : row.timestamp;
      var timeText = targetTime == null ? "目标时间未提供" : "目标 " + formatObservationTime(targetTime);
      var horizon = row.horizon_hours != null ? " · " + formatNumber(row.horizon_hours) + " 小时" : "";
      var predictionText = scoringFallback ? "未产生有效预测" : Number.isFinite(predicted) ? formatNumber(predicted) : "未产生";
      var observedText = Number.isFinite(observed) ? formatNumber(observed) : "未提供";
      var errorText = scoringFallback ? "模型误差不可用" : Number.isFinite(error) ? "误差 " + signedNumber(error) : "误差未提供";
      var baselineText = Number.isFinite(baseline) ? "基线 " + formatNumber(baseline) : "基线未提供";
      var reward = !scoringFallback && row.reward != null && row.reward !== "" ? Number(row.reward) : NaN;
      var rewardText = scoringFallback ? "不作为有效预测改善" : Number.isFinite(reward) ? "辅助 MAE 改善 " + signedNumber(reward) : "辅助 MAE 改善未提供";
      var method = scoringFallback ? "子任务失败；固定最差惩罚仅用于内部门禁，未作为模型预测" : executionPredictionMethod(candidate, index, row);
      var resultText = scoringFallback ? "执行失败，未产生有效预测" : "预测 " + predictionText;
      var valueText = scoringFallback ? "失败" : predictionText;
      var valueUnit = scoringFallback ? "无模型预测" : unitText(row.unit);
      return "<article class=\"sample-inference-row " + tone + "\"><span class=\"sample-inference-index\">样本 " + escapeHTML(String(index + 1).padStart(2, "0")) + "</span><div class=\"sample-inference-main\"><strong>" + escapeHTML(String(target)) + " · " + escapeHTML(timeText + horizon) + "</strong><span>" + escapeHTML(executionInputSummary(row)) + "</span><small>" + escapeHTML(resultText) + " · 观测 " + escapeHTML(observedText) + " · " + escapeHTML(errorText) + " · " + escapeHTML(baselineText) + " · " + escapeHTML(rewardText) + "</small><small>步骤：" + escapeHTML(compactTechnicalText(String(method).slice(0, 180))) + "</small></div><div class=\"sample-inference-values\"><strong>" + escapeHTML(valueText) + "</strong><span>" + escapeHTML(valueUnit) + "</span></div></article>";
    }).join("");
  }

  function executionDiagnosticNumber(value) {
    if (value == null || value === "") { return null; }
    var number = Number(value);
    return Number.isFinite(number) && number >= 0 ? number : null;
  }

  function executionAdaptivePhaseProgressText(stageProgress, unitLabel) {
    var progress = stageProgress && typeof stageProgress === "object" ? stageProgress : {};
    var phase = String(progress.evaluation_phase || "").toLowerCase();
    var fields = {
      screening: ["当前初筛", "screening_completed_origins", "screening_total_origins"],
      formal_batch: ["正式阶段累计", "formal_completed_origins", "formal_total_origins"],
      holdout: ["留出阶段累计", "holdout_completed_origins", "holdout_total_origins"]
    }[phase];
    if (!fields) { return ""; }
    var completed = executionDiagnosticNumber(progress[fields[1]]);
    var total = executionDiagnosticNumber(progress[fields[2]]);
    if (completed == null || total == null || total < 1) { return ""; }
    return fields[0] + " " + formatNumber(Math.min(completed, total)) + " / " + formatNumber(total) + " 个" + (unitLabel || "预测时点");
  }

  function executionAdaptiveSkippedProgressText(stageProgress) {
    var progress = stageProgress && typeof stageProgress === "object" ? stageProgress : {};
    return [
      ["初筛", progress.screening_skipped_origins],
      ["正式评测", progress.formal_skipped_origins],
      ["留出评测", progress.holdout_skipped_origins]
    ].map(function (entry) {
      var count = executionDiagnosticNumber(entry[1]);
      return count != null && count > 0 ? entry[0] + "跳过/提前终止 " + formatNumber(count) : "";
    }).filter(Boolean).join(" · ");
  }

  function executionSampleProgressSnapshot(run, stageProgress) {
    var heartbeat = stageProgress && typeof stageProgress === "object" ? stageProgress : null;
    var diagnostics = run && run.execution_diagnostics && typeof run.execution_diagnostics === "object" ? run.execution_diagnostics : {};
    var heartbeatCompleted = executionDiagnosticNumber(heartbeat && heartbeat.completed_samples);
    var heartbeatTotal = executionDiagnosticNumber(heartbeat && heartbeat.total_samples);
    var durableCompleted = executionDiagnosticNumber(diagnostics.live_evaluation_completed_examples);
    var durableTotal = executionDiagnosticNumber(diagnostics.live_evaluation_total_examples);
    var originBundleProtocol = String(run && run.sample_agent_protocol || "").indexOf("dsh-strict-origin-bundle@") === 0;
    var adaptiveOriginProgress = originBundleProtocol && heartbeat && (
      String(heartbeat.schema_version || "").indexOf("ecologyrsi-dsh.adaptive-progress/") === 0
      || heartbeat.completed_origins != null && heartbeat.total_origins != null
    );
    if (adaptiveOriginProgress) {
      // The adaptive projection already reports authoritative, generation-wide
      // origin counts. Cumulative diagnostics may still be measured in scoring
      // cells and must never be merged back into this origin denominator.
      heartbeatCompleted = executionDiagnosticNumber(heartbeat.completed_origins != null ? heartbeat.completed_origins : heartbeat.completed_samples);
      heartbeatTotal = executionDiagnosticNumber(heartbeat.total_origins != null ? heartbeat.total_origins : heartbeat.total_samples);
      durableCompleted = null;
      durableTotal = null;
    }
    var cellsPerOrigin = Number(run && run.prediction_cells_per_origin);
    var configuredCellBudget = Number(run && run.samples_per_update);
    var durableUsesCellUnits = originBundleProtocol
      && Number.isInteger(cellsPerOrigin) && cellsPerOrigin > 1
      && durableTotal != null
      && (
        heartbeatTotal != null
          ? durableTotal >= heartbeatTotal * cellsPerOrigin
          : configuredCellBudget > 0 && durableTotal >= configuredCellBudget
      );
    if (durableUsesCellUnits) {
      // Durable result batches contain one row per target/horizon cell, while
      // strict DSH heartbeats contain one complete forecast origin. Normalize
      // before taking maxima so the UI never reports mixed units such as
      // 144 / 4,500 beside a 16 / 500 origin heartbeat.
      durableCompleted = durableCompleted == null ? null : Math.floor(durableCompleted / cellsPerOrigin);
      durableTotal = Math.ceil(durableTotal / cellsPerOrigin);
    }
    if (!heartbeat && durableCompleted == null && durableTotal == null) { return null; }

    var snapshot = Object.assign({}, heartbeat || {});
    var completedValues = [heartbeatCompleted, durableCompleted].filter(function (value) { return value != null; });
    var totalValues = [heartbeatTotal, durableTotal].filter(function (value) { return value != null; });
    var completed = completedValues.length ? Math.max.apply(Math, completedValues) : 0;
    var total = Math.max(completed, totalValues.length ? Math.max.apply(Math, totalValues) : completed);
    var durableSucceeded = executionDiagnosticNumber(diagnostics.live_evaluation_succeeded_examples);
    var durableFailed = executionDiagnosticNumber(diagnostics.live_evaluation_failed_examples);
    if (durableUsesCellUnits) {
      durableSucceeded = durableSucceeded == null ? null : Math.floor(durableSucceeded / cellsPerOrigin);
      durableFailed = durableFailed == null ? null : Math.floor(durableFailed / cellsPerOrigin);
    }
    var heartbeatSucceeded = executionDiagnosticNumber(heartbeat && heartbeat.succeeded_samples);
    var heartbeatFailed = executionDiagnosticNumber(heartbeat && heartbeat.failed_samples);
    var durableAhead = durableCompleted != null && (heartbeatCompleted == null || durableCompleted > heartbeatCompleted);
    var heartbeatOutcomesCurrent = heartbeatCompleted != null
      && heartbeatSucceeded != null && heartbeatFailed != null
      && !durableAhead;

    snapshot.completed_samples = completed;
    snapshot.total_samples = total;
    if (durableUsesCellUnits && heartbeatTotal != null && durableTotal > heartbeatTotal) {
      var aggregateInFlight = executionDiagnosticNumber(snapshot.in_flight_batches) || 0;
      var aggregateQueued = executionDiagnosticNumber(snapshot.queued_batches) || 0;
      snapshot.awaiting_submission_batches = Math.max(
        0,
        total - completed - aggregateInFlight - aggregateQueued,
      );
    }
    if (heartbeatOutcomesCurrent) {
      snapshot.succeeded_samples = heartbeatSucceeded;
      snapshot.failed_samples = heartbeatFailed;
    } else if (durableSucceeded != null || durableFailed != null) {
      snapshot.succeeded_samples = durableSucceeded;
      snapshot.failed_samples = durableFailed;
    } else if (durableAhead && !executionRunAllowsLiveStatus(run)) {
      // Outcome counts from an older heartbeat cover fewer samples than the
      // durable total.  Hiding them is more truthful than presenting 8/9 as a
      // complete success/failure breakdown for a retained 9/9 snapshot.
      snapshot.succeeded_samples = null;
      snapshot.failed_samples = null;
    } else {
      snapshot.succeeded_samples = heartbeatSucceeded;
      snapshot.failed_samples = heartbeatFailed;
    }
    snapshot.evidence_qualifier = executionEvidenceQualifier(run);
    snapshot.live = executionRunAllowsLiveStatus(run);
    var explicitVerification = heartbeat && (heartbeat.outcomes_verified === true || heartbeat.outcome_counts_verified === true || String(heartbeat.verification_status || "").toLowerCase() === "verified");
    var completeTerminalBreakdown = !snapshot.live && snapshot.succeeded_samples != null && snapshot.failed_samples != null && Number(snapshot.succeeded_samples) + Number(snapshot.failed_samples) === completed;
    snapshot.outcomes_verified = Boolean(explicitVerification || completeTerminalBreakdown);
    if (snapshot.settled_origins == null && originBundleProtocol) {
      // A DSH heartbeat can be ahead of Host scoring (or can end in a
      // structured child error after the prediction tool returned). Only
      // durable Host rows are settled; never relabel remote completion as
      // settlement merely because the protocol is origin-bundled.
      snapshot.settled_origins = durableCompleted == null ? 0 : durableCompleted;
    }
    return snapshot;
  }

  function executionDiagnosticName(value, labels) {
    if (Array.isArray(value)) {
      var names = value.map(function (item) { return executionDiagnosticName(item, labels); }).filter(Boolean);
      return names.filter(function (item, index) { return names.indexOf(item) === index; }).join("、");
    }
    var raw = String(value == null ? "" : value).trim();
    return labels[raw] || raw;
  }

  function executionFitMethodText(value) {
    return executionDiagnosticName(value, {
      closed_form_ridge: "岭回归闭式拟合",
      bias_fit: "偏差项闭式拟合",
      toy_score: "确定性轻量评分",
      pending: "等待拟合"
    }) || "尚未记录";
  }

  function executionModeText(value) {
    return executionDiagnosticName(value, {
      closed_form_ridge: "岭回归闭式执行",
      registered_lightweight: "已登记轻量执行器",
      pending: "等待执行"
    }) || "等待执行";
  }

  function proposalSourceText(value) {
    return executionDiagnosticName(value, {
      remote_strategy: "策略模型 API",
      remote_model: "策略模型 API",
      dsh_native_agent: "DSH 原生候选生成智能体",
      model_gateway: "策略模型 API",
      host_fallback: "宿主有界回退",
      host_reserved_seed: "宿主保留种子提案",
      host_strategy: "宿主策略提案",
      host_parameter_generator: "宿主参数生成器",
      intervention: "人工意见",
      legacy_unknown: "历史来源未标注",
      unknown: "来源未标注"
    }) || "尚未产生提案";
  }

  function proposalSourceCountsText(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) { return ""; }
    return Object.keys(value).sort().map(function (key) {
      var count = executionDiagnosticNumber(value[key]);
      return count == null ? "" : proposalSourceText(key) + " " + formatNumber(count);
    }).filter(Boolean).join(" · ");
  }

  function remoteStrategyStatusText(value, run) {
    var normalized = String(value || "").toLowerCase();
    if (normalized === "running" && run && !executionRunAllowsLiveStatus(run)) {
      var runStatus = String(run.status || "").toLowerCase();
      return runStatus === "paused" ? "远程策略调用已暂停" : runStatus === "failed" || runStatus === "cancelled" ? "远程策略调用已中止" : "远程策略调用状态未封存";
    }
    return executionDiagnosticName(value, {
      completed: "远程策略调用完成",
      running: "远程策略调用中",
      partial: "远程策略部分成功",
      partial_host_fallback: "远程策略部分成功，部分使用宿主回退",
      host_fallback: "已使用宿主回退",
      unavailable: "远程策略不可用",
      failed: "远程策略调用失败",
      incomplete: "远程策略调用未全部成功",
      not_called: "本轮提案未调用远程策略",
      unknown: "历史记录无法确认远程调用",
      not_started: "尚未调用远程策略"
    }) || "远程策略状态未提供";
  }

  function executionDiagnosticMarkup(label, value, detail, tone) {
    return "<div class=\"execution-diagnostic" + (tone ? " " + tone : "") + "\"><span>" + escapeHTML(label) + "</span><strong>" + escapeHTML(value) + "</strong><small>" + escapeHTML(detail) + "</small></div>";
  }

  function renderExecutionDiagnostics(run) {
    var summaryNode = $("#execution-diagnostics-summary");
    var gridNode = $("#execution-diagnostics-grid");
    if (!summaryNode || !gridNode) { return; }
    var diagnostics = run && run.execution_diagnostics && typeof run.execution_diagnostics === "object" ? run.execution_diagnostics : {};
    var trainingPartitionRows = executionDiagnosticNumber(diagnostics.training_partition_rows);
    var trainingEligible = executionDiagnosticNumber(diagnostics.training_eligible_examples);
    var trainingUsed = executionDiagnosticNumber(diagnostics.training_used_examples);
    var trainingSkipped = executionDiagnosticNumber(diagnostics.training_skipped_examples);
    var evaluationPartitionRows = executionDiagnosticNumber(diagnostics.evaluation_partition_rows);
    var evaluationEligible = executionDiagnosticNumber(diagnostics.evaluation_eligible_examples);
    var evaluationUsed = executionDiagnosticNumber(diagnostics.evaluation_used_examples);
    var evaluationSkipped = executionDiagnosticNumber(diagnostics.evaluation_skipped_examples);
    var artifacts = executionDiagnosticNumber(diagnostics.candidate_artifacts_count);
    var evaluations = executionDiagnosticNumber(diagnostics.candidate_evaluations_count);
    var workItems = executionDiagnosticNumber(diagnostics.candidate_work_items);
    var liveEvaluationCompleted = executionDiagnosticNumber(diagnostics.live_evaluation_completed_examples);
    var liveEvaluationTotal = executionDiagnosticNumber(diagnostics.live_evaluation_total_examples);
    var liveEvaluationSucceeded = executionDiagnosticNumber(diagnostics.live_evaluation_succeeded_examples);
    var liveEvaluationFailed = executionDiagnosticNumber(diagnostics.live_evaluation_failed_examples);
    var liveEvaluationCandidates = executionDiagnosticNumber(diagnostics.live_evaluation_candidate_count);
    var fitPasses = executionDiagnosticNumber(diagnostics.fit_passes_completed);
    var fitPassesPerCandidate = executionDiagnosticNumber(diagnostics.fit_passes_per_candidate);
    var iterativeEpochTraining = typeof diagnostics.iterative_epoch_training === "boolean" ? diagnostics.iterative_epoch_training : null;
    var legacyTrainingRows = executionDiagnosticNumber(diagnostics.training_rows);
    var legacyEvaluationRows = executionDiagnosticNumber(diagnostics.evaluation_rows);
    var legacyEpochs = executionDiagnosticNumber(diagnostics.epochs_completed);
    var sourceCounts = proposalSourceCountsText(diagnostics.proposal_sources) || proposalSourceCountsText(diagnostics.source_counts);
    var proposalSources = diagnostics.proposal_sources && typeof diagnostics.proposal_sources === "object" && !Array.isArray(diagnostics.proposal_sources) ? "" : proposalSourceText(diagnostics.proposal_sources);
    var remoteCalls = executionDiagnosticNumber(diagnostics.remote_strategy_calls);
    var remoteSuccesses = executionDiagnosticNumber(diagnostics.remote_strategy_successes);
    var fallbackCount = executionDiagnosticNumber(diagnostics.fallback_count);
    var evidenceStatus = String(diagnostics.execution_evidence_status || "").toLowerCase();
    var hasPartialEvidence = ["partial_live", "retained_partial", "aborted_partial", "aborted", "mixed_partial"].indexOf(evidenceStatus) >= 0 && liveEvaluationCompleted != null && liveEvaluationCompleted > 0;
    var hasLiveEvidence = evidenceStatus === "partial_live" && hasPartialEvidence;
    var partialEvidenceLabel = hasLiveEvidence ? "部分实时证据" : evidenceStatus === "retained_partial" ? "已保留部分证据" : evidenceStatus === "mixed_partial" ? "部分执行证据" : "已中止，保留部分证据";
    var partialCompletionLabel = hasLiveEvidence ? "实时完成" : evidenceStatus === "retained_partial" ? "已保留" : evidenceStatus === "mixed_partial" ? "累计完成" : "中止前完成";
    var partialWorkLabel = hasLiveEvidence ? "实时反馈目标" : evidenceStatus === "mixed_partial" ? "部分反馈目标" : "已保留反馈目标";
    var hasTrainingRecord = (artifacts != null && artifacts > 0) || [trainingPartitionRows, trainingEligible, trainingUsed, trainingSkipped].some(function (value) { return value != null && value > 0; });
    var hasEvaluationRecord = (evaluations != null && evaluations > 0) || [evaluationPartitionRows, evaluationEligible, evaluationUsed, evaluationSkipped].some(function (value) { return value != null && value > 0; });

    var trainingValue = hasTrainingRecord && trainingPartitionRows != null ? "累计扫描 " + formatNumber(trainingPartitionRows) + " 行次" : hasTrainingRecord && trainingUsed != null ? formatNumber(trainingUsed) + " 个训练目标样本" : legacyTrainingRows != null && legacyTrainingRows > 0 ? "累计扫描 " + formatNumber(legacyTrainingRows) + " 行次" : "尚未记录";
    var trainingDetail = hasTrainingRecord
      ? "可用目标 " + (trainingEligible == null ? "—" : formatNumber(trainingEligible)) + " · 用于拟合 " + (trainingUsed == null ? "—" : formatNumber(trainingUsed)) + " · 跳过 " + (trainingSkipped == null ? "—" : formatNumber(trainingSkipped))
      : legacyTrainingRows != null && legacyTrainingRows > 0 ? "旧投影累计值，未区分原始行与目标样本" : "等待候选训练产物";
    var evaluationValue = hasPartialEvidence ? partialCompletionLabel + " " + formatNumber(liveEvaluationCompleted) + " / " + formatNumber(liveEvaluationTotal || liveEvaluationCompleted) + " 个反馈目标样本" : hasEvaluationRecord && evaluationPartitionRows != null ? "累计扫描 " + formatNumber(evaluationPartitionRows) + " 行次" : hasEvaluationRecord && evaluationUsed != null ? formatNumber(evaluationUsed) + " 个反馈目标样本" : legacyEvaluationRows != null && legacyEvaluationRows > 0 ? "累计 " + formatNumber(legacyEvaluationRows) + " 行次" : "尚未记录";
    var evaluationDetail = hasPartialEvidence
      ? partialEvidenceLabel + " · 成功 " + (liveEvaluationSucceeded == null ? "—" : formatNumber(liveEvaluationSucceeded)) + " · 失败 " + (liveEvaluationFailed == null ? "—" : formatNumber(liveEvaluationFailed)) + " · 未封存候选 " + (liveEvaluationCandidates == null ? "—" : formatNumber(liveEvaluationCandidates))
      : hasEvaluationRecord
      ? "可用目标 " + (evaluationEligible == null ? "—" : formatNumber(evaluationEligible)) + " · 已评测 " + (evaluationUsed == null ? "—" : formatNumber(evaluationUsed)) + " · 跳过 " + (evaluationSkipped == null ? "—" : formatNumber(evaluationSkipped))
      : legacyEvaluationRows != null && legacyEvaluationRows > 0 ? "旧投影累计值，未区分 eligible 与 used" : "等待训练反馈评测";
    var workloadValue = workItems != null ? formatNumber(workItems) + " 个样本工作项" : artifacts != null || evaluations != null ? formatNumber((artifacts || 0) + (evaluations || 0)) + " 条候选记录" : "尚未产生";
    var workloadDetail = hasPartialEvidence
      ? "正式训练目标 " + (hasTrainingRecord && trainingUsed != null ? formatNumber(trainingUsed) : "—") + " · 正式反馈目标 " + (hasEvaluationRecord && evaluationUsed != null ? formatNumber(evaluationUsed) : "—") + " · " + partialWorkLabel + " " + formatNumber(liveEvaluationCompleted)
      : workItems != null
      ? "训练目标 " + (trainingUsed == null ? "—" : formatNumber(trainingUsed)) + " · 反馈目标 " + (evaluationUsed == null ? "—" : formatNumber(evaluationUsed)) + " · 产物/评测 " + (artifacts == null ? "—" : formatNumber(artifacts)) + "/" + (evaluations == null ? "—" : formatNumber(evaluations))
      : "训练产物 " + (artifacts == null ? "—" : formatNumber(artifacts)) + " · 候选评测 " + (evaluations == null ? "—" : formatNumber(evaluations));
    var effectivePasses = fitPasses != null ? fitPasses : legacyEpochs;
    var fitDetail;
    if (!hasTrainingRecord && !(effectivePasses != null && effectivePasses > 0)) {
      fitDetail = "等待拟合证据";
    } else if (iterativeEpochTraining === false) {
      fitDetail = "无神经网络 epoch · 累计 " + formatNumber(effectivePasses) + " 次 fit pass" + (fitPassesPerCandidate == null ? "" : " · 每候选 " + formatNumber(fitPassesPerCandidate) + " 次");
    } else if (iterativeEpochTraining === true) {
      fitDetail = "迭代式 epoch 训练 · 累计 " + formatNumber(effectivePasses) + " 次" + (fitPassesPerCandidate == null ? "" : " · 每候选最多 " + formatNumber(fitPassesPerCandidate) + " 次");
    } else {
      fitDetail = "完成 " + formatNumber(effectivePasses) + " 次拟合 pass" + (fitPasses == null ? "（兼容旧 epochs 字段，训练类型未标注）" : "（训练类型未标注）");
    }
    var sourceValue = sourceCounts || proposalSources;
    var remoteDetail = "调用 " + (remoteCalls == null ? "—" : formatNumber(remoteCalls)) + " · 成功 " + (remoteSuccesses == null ? "—" : formatNumber(remoteSuccesses));
    var fallbackDetail = fallbackCount != null && fallbackCount > 0 ? " · 回退 " + formatNumber(fallbackCount) : "";
    var remoteStatus = remoteStrategyStatusText(diagnostics.remote_strategy_status, run);
    var hasEvidence = hasPartialEvidence || [trainingPartitionRows, trainingUsed, evaluationPartitionRows, evaluationUsed, artifacts, evaluations, workItems].some(function (value) { return value != null && value > 0; });

    summaryNode.textContent = hasPartialEvidence
      ? partialEvidenceLabel + " · 反馈 " + formatNumber(liveEvaluationCompleted) + " / " + formatNumber(liveEvaluationTotal || liveEvaluationCompleted) + " · 候选工作量 " + (workItems == null ? formatNumber(liveEvaluationCompleted) : formatNumber(workItems))
      : hasEvidence
      ? "累计候选工作量 " + (workItems == null ? "—" : formatNumber(workItems)) + " · 拟合 pass " + (effectivePasses == null ? "—" : formatNumber(effectivePasses))
      : "尚无执行证据";
    gridNode.innerHTML = [
      executionDiagnosticMarkup("训练拟合分区", trainingValue, trainingDetail, trainingPartitionRows == null && legacyTrainingRows != null ? "is-muted" : ""),
      executionDiagnosticMarkup("训练反馈分区", evaluationValue, evaluationDetail, hasPartialEvidence ? "is-warning" : evaluationPartitionRows == null && legacyEvaluationRows != null ? "is-muted" : ""),
      executionDiagnosticMarkup("累计候选工作量", workloadValue, workloadDetail, ""),
      executionDiagnosticMarkup("拟合机制", executionFitMethodText(diagnostics.fit_method), fitDetail, ""),
      executionDiagnosticMarkup("候选提案来源", sourceValue || "尚未产生提案", sourceCounts ? "按已记录提案计数" : "等待来源计数", diagnostics.fallback_used ? "is-warning" : ""),
      executionDiagnosticMarkup("策略调用", executionModeText(diagnostics.execution_mode), remoteStatus + " · " + remoteDetail + fallbackDetail, diagnostics.fallback_used ? "is-warning" : "")
    ].join("");
  }

  function executionSchedulerQueueInfo(run) {
    if (run && !executionRunAllowsLiveStatus(run)) { return null; }
    var progress = run && run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : {};
    var scheduler = run && run.execution_scheduler && typeof run.execution_scheduler === "object" ? run.execution_scheduler : null;
    if (!scheduler || String(progress.phase || "").toLowerCase() !== "queued" || String(scheduler.run_state || "").toLowerCase() !== "queued") {
      return null;
    }
    var detail = ["等待其他运行当前轮结束"];
    var position = scheduler.queue_position == null ? NaN : Number(scheduler.queue_position);
    var ahead = scheduler.queued_ahead == null ? NaN : Number(scheduler.queued_ahead);
    var activeWorkers = scheduler.active_worker_count == null ? NaN : Number(scheduler.active_worker_count);
    var workerCount = scheduler.worker_count == null ? NaN : Number(scheduler.worker_count);
    if (Number.isInteger(position) && position > 0) { detail.push("队列第 " + formatNumber(position) + " 位"); }
    if (Number.isInteger(ahead) && ahead >= 0) { detail.push("前方 " + formatNumber(ahead) + " 个运行"); }
    if (Number.isInteger(activeWorkers) && activeWorkers >= 0 && Number.isInteger(workerCount) && workerCount > 0) {
      detail.push("工作器占用 " + formatNumber(activeWorkers) + " / " + formatNumber(workerCount));
    }
    return {detail: detail.join(" · ")};
  }

  function executionDshActivityPresentation(activity) {
    if (!activity || typeof activity !== "object") { return null; }
    var stateValue = String(activity.state || "").toLowerCase();
    var stage = compactTechnicalText(activity.dsh_stage || activity.evolution_stage || "DSH");
    var role = compactTechnicalText(activity.role || "");
    var roleText = role ? " · 角色 " + role : "";
    var attempt = Number(activity.launch_attempt);
    var attemptText = Number.isInteger(attempt) && attempt > 0 ? " · 第 " + formatNumber(attempt) + " 次" : "";
    var startedAt = Date.parse(activity.started_at || "");
    var elapsedText = Number.isFinite(startedAt)
      ? " · 已运行 " + autonomyWaitDurationText(Math.max(0, Date.now() - startedAt))
      : "";
    var updatedText = activity.updated_at ? " · DSH 更新 " + formatTime(activity.updated_at) : "";
    var labels = {
      model_running: "DSH 子模型执行中",
      model_retry_running: "DSH 子模型重试中",
      host_validating: "Host 正在校验 DSH 结果",
      waiting_for_model_slot: "等待 DSH provider 槽位",
    };
    if (!labels[stateValue]) { return null; }
    return {
      retrying: stateValue === "model_retry_running",
      statusText: stateValue === "model_retry_running" ? "模型重试中" : null,
      detail: "DSH 子阶段 " + stage + roleText + attemptText + " · " + labels[stateValue] + elapsedText,
      heartbeat: "DSH " + stage + roleText + " · " + labels[stateValue] + updatedText,
    };
  }

  function renderExecutionMonitor(run) {
    var statusNode = $("#execution-monitor-status");
    var labelNode = $("#execution-progress-label");
    var percentNode = $("#execution-progress-percent");
    var detailNode = $("#execution-progress-detail");
    var track = $("#execution-progress-track");
    var fill = $("#execution-progress-fill");
    var generationNode = $("#execution-generation-progress");
    var epochNode = $("#execution-epoch-progress");
    var candidateNode = $("#execution-candidate-progress");
    var sampleNode = $("#execution-sample-progress");
    var tokenNode = $("#execution-token-progress");
    var heartbeatNode = $("#execution-heartbeat");
    var activityNode = $("#execution-last-activity");
    var stageNode = $("#execution-stage-strip");
    if (!statusNode || !labelNode || !track || !fill || !stageNode) { return; }
    if (!run) {
      var submitting = state.pendingAction === "create" || state.createStatus && state.createStatus.state === "submitting";
      statusNode.className = submitting ? "pill pill-blue" : "pill pill-neutral";
      statusNode.textContent = submitting ? "提交已接收" : "等待运行";
      labelNode.textContent = submitting ? "正在创建运行" : "尚未开始";
      percentNode.textContent = "0%";
      detailNode.textContent = submitting ? "服务正在返回运行 ID；收到后将实时显示轮次、候选和逐样本预测。" : "等待模型提交第一轮执行";
      track.className = "execution-progress-track";
      track.setAttribute("aria-valuenow", "0");
      fill.style.width = "0%";
      generationNode.textContent = "进化轮次：0 / 0";
      if (epochNode) { epochNode.textContent = "本轮进度：—"; }
      candidateNode.textContent = "候选版本：0";
      sampleNode.textContent = submitting ? "预测样本：等待运行 ID" : "预测样本：0";
      if (tokenNode) { tokenNode.textContent = "逐样本智能体 Token：等待真实账本"; }
      if (heartbeatNode) { heartbeatNode.textContent = "评测心跳：—"; }
      activityNode.textContent = submitting ? "最近活动：提交已接收" : "最近活动：—";
      stageNode.innerHTML = "";
      renderExecutionDiagnostics(null);
      renderExecutionCandidate(null, null, null, []);
      renderExecutionSamples(null, null);
      renderExecutionPlan(null, null);
      return;
    }
    var round = executionRound(run);
    var candidate = executionCandidateFor(run, round);
    var stages = executionStageValues(candidate, round, run);
    var progress = executionProgress(run, round, stages);
    // Read the durable projection before deriving any status text.  This is
    // intentionally kept ahead of the retry/status branches below: a render
    // heartbeat must remain total even while the backend is waiting for a
    // provider retry and no candidate has been materialized yet.
    var explicitProgress = run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : {};
    var sampleRows = executionPredictionRows(candidate, run);
    var runStatus = String(run.status || "").toLowerCase();
    var paused = runStatus === "paused";
    var cancelled = runStatus === "cancelled";
    var liveAllowed = executionRunAllowsLiveStatus(run);
    var rawStageProgress = explicitProgress.stage_progress && typeof explicitProgress.stage_progress === "object" ? explicitProgress.stage_progress : candidate && candidate.execution && candidate.execution.stage_progress && typeof candidate.execution.stage_progress === "object" ? candidate.execution.stage_progress : null;
    var stageProgress = executionSampleProgressSnapshot(run, rawStageProgress);
    var pausedDrained = paused && stageProgress && stageProgress.progress_kind === "drained";
    var outcome = runOutcomeCode(run);
    var exhausted = outcome === "budget_exhausted_without_acceptable_candidate";
    var completionText = runOutcomeText(run);
    var retryWait = liveAllowed && explicitProgress.retry_wait && typeof explicitProgress.retry_wait === "object" ? explicitProgress.retry_wait : null;
    var failed = runStatus === "failed" || liveAllowed && !paused && !retryWait && stages.some(function (item) { return executionStatusClass(item.value) === "is-failed"; });
    var displayActive = progress.active && !paused;
    var finished = !paused && !displayActive && (runStatus === "completed" || progress.percent >= 99.99);
    var waiting = runNeedsAdvanceAction(run, state.events);
    var autoActive = liveAllowed && runHasContinuousAutoProgress(run);
    var autoManaged = liveAllowed && runHasContinuousAutoProgress(run);
    var autoBlocked = false;
    var hardTokenPause = runHasHardTokenPause(run);
    var retryCircuitPaused = runHasRetryCircuitPause(run);
    var schedulerQueue = executionSchedulerQueueInfo(run);
    var dshActivity = liveAllowed && explicitProgress.dsh_activity && typeof explicitProgress.dsh_activity === "object" ? executionDshActivityPresentation(explicitProgress.dsh_activity) : null;
    var heartbeatState = executionHeartbeatState(run);
    var heartbeatStalled = heartbeatState.stalled && !retryWait && !dshActivity && !retryCircuitPaused;
    var retainedEvidence = executionHasRetainedEvidence(run);
    var statusText = failed ? "执行失败" : hardTokenPause ? "逐样本智能体 Token 预算已暂停" : retryCircuitPaused ? retryCircuitStatusText(run) : pausedDrained ? "已暂停，请求已排空" : paused ? "已暂停，可人工干预" : cancelled ? "已取消" : retryWait ? retryWaitStatusText(retryWait) : schedulerQueue ? "后台排队中" : autoBlocked ? "自动推进已暂停" : heartbeatStalled ? "疑似停滞，正在核验" : displayActive ? (dshActivity && dshActivity.statusText || "模型执行中") : completionText || (finished ? "运行已结束" : retainedEvidence ? executionEvidenceQualifier(run) : autoActive ? "自动准备下一轮" : autoManaged ? "后台自动推进" : waiting ? "等待推进" : "等待下一轮");
    var statusClass = failed ? "pill-red" : hardTokenPause || paused || cancelled || retainedEvidence || heartbeatStalled ? "pill-amber" : schedulerQueue ? "pill-blue" : autoBlocked ? "pill-red" : displayActive ? "pill-blue" : exhausted ? "pill-amber" : finished ? "pill-green" : waiting ? "pill-amber" : autoActive || autoManaged ? "pill-blue" : "pill-neutral";
    statusNode.className = "pill " + statusClass;
    statusNode.textContent = statusText;
    var roundedPercent = Math.round(progress.percent);
    var supersededRevision = explicitProgress.superseded_sample_revision && typeof explicitProgress.superseded_sample_revision === "object"
      ? explicitProgress.superseded_sample_revision
      : candidate && candidate.execution && candidate.execution.superseded_sample_revision;
    var supersededRevisionText = supersededSampleRevisionText(supersededRevision);
    var currentGeneration = Number(explicitProgress.current_generation || round && round.generation || Number(run.generation || 0) + 1);
    labelNode.textContent = paused ? "第 " + String(currentGeneration) + " 轮已暂停" : failed ? "运行失败" : cancelled ? "运行已取消" : schedulerQueue ? "已进入后台执行队列" : displayActive ? "正在处理第 " + String(currentGeneration) + " 轮" : completionText || (finished ? "运行已结束" : retainedEvidence ? executionEvidenceQualifier(run) : "已完成 " + formatNumber(progress.completed) + " 轮");
    percentNode.textContent = roundedPercent + "%";
    // The run-level checkpoint is the authoritative live stage. Candidate
    // projections can legitimately lag while an adaptive batch is being
    // settled, so using them first labels active evaluation as old training.
    var recordedCurrentStage = explicitProgress.current_stage || candidate && candidate.execution && candidate.execution.current_stage;
    var currentStage = liveAllowed || paused ? recordedCurrentStage : null;
    var stageText = retryWait ? (evolutionStageLabels[retryWait.stage] || retryWait.stage || "等待重试") : currentStage ? (evolutionStageLabels[currentStage] || executionStageLabels[currentStage] || currentStage) : stages.map(function (item) { return executionStageText(item.value, item.key) === "进行中" ? executionStageLabels[item.key] : ""; }).filter(Boolean)[0];
    var observedDetail = run.best_observed_candidate_id || run.best_observed_score != null ? " · " + rawBestObservedSummary(run) : "";
    var elapsedMs = null;
    if (elapsedMs == null && displayActive) { elapsedMs = executionActiveStageElapsedMs(run, currentStage); }
    var measuredRoundMs = round && round.timing && Number.isFinite(Number(round.timing.duration_ms)) ? Number(round.timing.duration_ms) : null;
    var effectiveElapsedMs = measuredRoundMs != null && !displayActive ? measuredRoundMs : elapsedMs;
    var elapsedText = executionElapsedText(effectiveElapsedMs, displayActive);
    var holdoutArmLabels = { finalist_1: "Top 1 候选", finalist_2: "Top 2 候选", incumbent: "上轮最优" };
    var batchText = stageProgress && String(stageProgress.evaluation_phase || "").toLowerCase() === "holdout"
      ? " · 留出评测 " + (holdoutArmLabels[String(stageProgress.holdout_arm || "")] || "准备中")
      : stageProgress && Number.isFinite(Number(stageProgress.batch_index)) && Number.isFinite(Number(stageProgress.batch_count))
        ? " · 批次 " + formatNumber(stageProgress.batch_index) + " / " + formatNumber(stageProgress.batch_count)
        : "";
    var terminalEvidenceText = (failed || cancelled) && retainedEvidence ? " · " + executionEvidenceQualifier(run) : "";
    detailNode.textContent = hardTokenPause ? "逐样本智能体 Token 硬预算已耗尽；逐样本 checkpoint 已保留。" : retryCircuitPaused ? retryCircuitDetailText(run) : paused ? "暂停阶段：" + (stageText || "等待阶段状态") + (candidate ? " · " + shortId(candidate.id || candidate.candidate_id) : "") + batchText + (pausedDrained ? " · 请求已排空" : " · 已停止提交新请求") : retryWait ? retryWaitDetailText(retryWait) : schedulerQueue ? schedulerQueue.detail : heartbeatStalled ? "超过 120 秒没有新的执行心跳（最后更新 " + heartbeatState.age_text + "），正在核验模型与宿主状态。" : displayActive ? "当前阶段：" + (stageText || "等待事件回执") + (candidate ? " · " + shortId(candidate.id || candidate.candidate_id) : "") + batchText + (dshActivity ? " · " + dshActivity.detail : "") + elapsedText : statusText + terminalEvidenceText + (exhausted ? observedDetail + "；该分数不代表保留结果" : candidate ? " · 最近候选 " + shortId(candidate.id || candidate.candidate_id) : "") + (autoActive ? elapsedText : "");
    track.className = "execution-progress-track" + (failed ? " is-failed" : paused ? " is-paused" : heartbeatStalled ? " is-stalled" : displayActive ? " is-running" : "");
    track.setAttribute("aria-valuenow", String(roundedPercent));
    fill.style.width = Math.max(0, Math.min(100, progress.percent)).toFixed(1) + "%";
    var epochPercent = executionEpochProgressPercent(run);
    track.setAttribute("aria-valuetext", "全程 " + roundedPercent + "%" + (epochPercent == null ? "" : "；本轮 " + Math.round(epochPercent) + "%"));
    var completedGenerations = Number.isFinite(Number(explicitProgress.completed_generations)) ? Number(explicitProgress.completed_generations) : Number(run.generation || 0);
    generationNode.textContent = "进化轮次：" + formatNumber(Math.min(progress.total, completedGenerations)) + " / " + formatNumber(progress.total);
    if (epochNode) { epochNode.textContent = epochPercent == null ? "本轮进度：—" : "本轮进度：" + formatNumber(epochPercent, 1) + "%"; }
    candidateNode.textContent = "候选版本：" + formatNumber(Array.isArray(run.candidates) ? run.candidates.length : 0);
    var showLiveProgressDetail = Boolean(stageProgress && stageProgress.live);
    var showDrainedProgressDetail = Boolean(pausedDrained);
    var showPausedProgressDetail = Boolean(paused && stageProgress);
    var originBundleProtocol = String(run.sample_agent_protocol || "").indexOf("dsh-strict-origin-bundle@") === 0;
    var progressUnitLabel = originBundleProtocol ? "预测时点" : "样本";
    var sampleRate = showLiveProgressDetail && Number(stageProgress.samples_per_minute);
    var sampleRateText = Number.isFinite(sampleRate) && sampleRate > 0 ? " · " + formatNumber(sampleRate, 1) + " " + progressUnitLabel + "/分钟" : "";
    var inFlight = (showLiveProgressDetail || showDrainedProgressDetail || showPausedProgressDetail) && Number(stageProgress.in_flight_requests != null ? stageProgress.in_flight_requests : stageProgress.in_flight_batches);
    var progressKind = stageProgress && stageProgress.progress_kind;
    var inFlightLabel = progressKind === "drained" ? "DSH 子任务已排空" : runStatus === "paused" ? "暂停快照未终结子任务" : "DSH 子任务未终结";
    var inFlightText = Number.isInteger(inFlight) && inFlight >= 0 ? " · " + inFlightLabel + " " + formatNumber(inFlight) : "";
    var queueSemantics = stageProgress && stageProgress.queue_semantics;
    var exactProviderQueue = queueSemantics === "provider_gate_snapshot";
    var legacyAwaitingSubmission = queueSemantics === "awaiting_origin_submission";
    // queued_batches is normally reconstructed from durable Host child
    // events; it is not the private ProviderStageGate FIFO. Only label a value
    // as Provider waiting when the API explicitly supplies that exact snapshot.
    var queued = (showLiveProgressDetail || showDrainedProgressDetail || showPausedProgressDetail) && Number(
      exactProviderQueue
        ? stageProgress.provider_queued_requests
        : legacyAwaitingSubmission
          ? stageProgress.queued_batches
          : NaN
    );
    var queuedLabel = legacyAwaitingSubmission ? "当前阶段待提交" : progressKind === "drained" ? "暂停后 Provider 等待" : runStatus === "paused" ? "暂停快照 Provider 等待" : "Provider 等待";
    var queuedText = Number.isInteger(queued) && queued >= 0 ? " · " + queuedLabel + " " + formatNumber(queued) : "";
    var awaitingSubmission = (showLiveProgressDetail || showDrainedProgressDetail || showPausedProgressDetail) && Number(stageProgress.awaiting_submission_batches);
    var awaitingSubmissionText = Number.isInteger(awaitingSubmission) && awaitingSubmission >= 0 ? " · 当前阶段待提交 " + formatNumber(awaitingSubmission) : "";
    var showAdmissionDetail = showLiveProgressDetail || showDrainedProgressDetail || showPausedProgressDetail;
    var admissionActive = showAdmissionDetail ? executionDiagnosticNumber(stageProgress.admission_active) : null;
    var admissionBaseLimit = showAdmissionDetail ? executionDiagnosticNumber(stageProgress.admission_limit) : null;
    var adaptiveAdmissionLimit = showAdmissionDetail ? executionDiagnosticNumber(stageProgress.adaptive_admission_limit) : null;
    var admissionLimit = adaptiveAdmissionLimit == null ? admissionBaseLimit : adaptiveAdmissionLimit;
    var admissionWaiting = showAdmissionDetail ? executionDiagnosticNumber(stageProgress.admission_waiting) : null;
    var admissionText = admissionActive != null && admissionLimit != null
      ? " · 宿主并发 " + formatNumber(admissionActive) + " / " + formatNumber(admissionLimit)
      : admissionActive != null ? " · 宿主并发占用 " + formatNumber(admissionActive) : "";
    var admissionWaitingText = admissionWaiting != null ? " · 宿主等待 " + formatNumber(admissionWaiting) : "";
    var admissionLimitText = admissionBaseLimit != null && adaptiveAdmissionLimit != null && admissionBaseLimit !== adaptiveAdmissionLimit
      ? "（配置上限 " + formatNumber(admissionBaseLimit) + "）"
      : "";
    if (admissionText && admissionLimitText) { admissionText += admissionLimitText; }
    var awaitingSettlement = (showLiveProgressDetail || showDrainedProgressDetail || showPausedProgressDetail) && Number(stageProgress.awaiting_settlement_batches);
    var awaitingSettlementText = Number.isInteger(awaitingSettlement) && awaitingSettlement > 0 ? " · 待结算 " + formatNumber(awaitingSettlement) : "";
    var primaryRequests = showLiveProgressDetail && Number(stageProgress.primary_gateway_request_count);
    var repairRequests = showLiveProgressDetail && Number(stageProgress.repair_gateway_request_count);
    var requestBreakdownText = Number.isInteger(primaryRequests) && primaryRequests >= 0 && Number.isInteger(repairRequests) && repairRequests > 0
      ? " · 主预测请求 " + formatNumber(primaryRequests) + " · 修复请求 " + formatNumber(repairRequests)
      : "";
    var completedRepairWaves = showLiveProgressDetail && Number(stageProgress.completed_repair_waves);
    var activeRepairWaves = showLiveProgressDetail && Number(stageProgress.active_repair_waves);
    var hasCompletedRepairWaves = Number.isInteger(completedRepairWaves) && completedRepairWaves > 0;
    var hasActiveRepairWaves = Number.isInteger(activeRepairWaves) && activeRepairWaves > 0;
    var repairWaveText = hasCompletedRepairWaves
      ? " · 修复波完成 " + formatNumber(completedRepairWaves) + (hasActiveRepairWaves ? " / 在飞 " + formatNumber(activeRepairWaves) : "")
      : hasActiveRepairWaves ? " · 修复波在飞 " + formatNumber(activeRepairWaves) : "";
    var gatewayAttempts = showLiveProgressDetail && Number(stageProgress.gateway_request_count);
    var gatewayAttemptsText = Number.isInteger(gatewayAttempts) && gatewayAttempts >= 0 ? " · 网关尝试 " + formatNumber(gatewayAttempts) : "";
    var predictionPending = showLiveProgressDetail && Number(stageProgress.prediction_tool_pending_origins);
    var predictionPendingText = Number.isInteger(predictionPending) && predictionPending > 0 ? " · 预测已产出待验收 " + formatNumber(predictionPending) : "";
    var failedChildRequests = showLiveProgressDetail && Number(stageProgress.child_execution_failed_request_count);
    var structuredChildErrors = showLiveProgressDetail && Number(stageProgress.structured_child_model_error_count);
    var childFailureText = Number.isInteger(failedChildRequests) && failedChildRequests > 0
      ? " · 子任务失败请求 " + formatNumber(failedChildRequests) + (Number.isInteger(structuredChildErrors) && structuredChildErrors > 0 ? "（子模型终止错误 " + formatNumber(structuredChildErrors) + "）" : "")
      : "";
    var causalWave = showLiveProgressDetail && Number(stageProgress.causal_wave_sample_count);
    var causalWaveText = Number.isInteger(causalWave) && causalWave > 0 ? " · 本波次 " + formatNumber(causalWave) : "";
    var stageRemainingSeconds = showLiveProgressDetail && Number(stageProgress.estimated_remaining_seconds);
    var runRemainingSeconds = showLiveProgressDetail && Number(stageProgress.run_estimated_remaining_seconds);
    var remainingText = "";
    if (Number.isFinite(runRemainingSeconds) && runRemainingSeconds > 0) {
      remainingText = "预计全程剩余 " + compactDuration(runRemainingSeconds);
      if (Number.isFinite(stageRemainingSeconds) && stageRemainingSeconds > 0 && Math.abs(runRemainingSeconds - stageRemainingSeconds) > 60) {
        remainingText += "（当前阶段 " + compactDuration(stageRemainingSeconds) + "）";
      }
    } else if (Number.isFinite(stageRemainingSeconds) && stageRemainingSeconds > 0) {
      remainingText = "预计剩余 " + compactDuration(stageRemainingSeconds);
    }
    var outcomesVerified = Boolean(stageProgress && stageProgress.outcomes_verified === true);
    var succeeded = stageProgress && Number(stageProgress.succeeded_samples);
    var failedSamples = stageProgress && Number(stageProgress.failed_samples);
    var outcomeScope = String(stageProgress && stageProgress.outcome_scope || "");
    var outcomeScopeLabel = outcomeScope === "active_formal_batch"
      ? "本批"
      : outcomeScope === "active_holdout_arm" ? "本臂" : "";
    var verifiedOutcomeText = outcomesVerified && Number.isFinite(succeeded) && Number.isFinite(failedSamples) ? " · " + outcomeScopeLabel + "成功 " + formatNumber(succeeded) + " · " + outcomeScopeLabel + "失败 " + formatNumber(failedSamples) : "";
    var settled = stageProgress && stageProgress.settled_origins != null ? Number(stageProgress.settled_origins) : NaN;
    var displayedCompleted = stageProgress && Number(stageProgress.completed_samples);
    var settledText = Number.isFinite(settled) && settled >= 0 && (!Number.isFinite(displayedCompleted) || settled !== displayedCompleted) ? " · Host 已结算 " + formatNumber(settled) : "";
    var evidenceQualifierText = stageProgress && !stageProgress.live && stageProgress.evidence_qualifier ? " · " + stageProgress.evidence_qualifier : "";
    var phaseProgressText = executionAdaptivePhaseProgressText(stageProgress, progressUnitLabel);
    var skippedProgressText = executionAdaptiveSkippedProgressText(stageProgress);
    sampleNode.textContent = stageProgress ? "全轮" + progressUnitLabel + "进度：" + formatNumber(stageProgress.completed_samples) + " / " + formatNumber(stageProgress.total_samples) + (phaseProgressText ? " · " + phaseProgressText : "") + (skippedProgressText ? " · " + skippedProgressText : "") + settledText + verifiedOutcomeText + evidenceQualifierText + causalWaveText + admissionText + admissionWaitingText + inFlightText + queuedText + awaitingSubmissionText + awaitingSettlementText + predictionPendingText + childFailureText + requestBreakdownText + repairWaveText + gatewayAttemptsText + sampleRateText + (remainingText ? " · " + remainingText : "") + (supersededRevisionText ? " · " + supersededRevisionText : "") : "预测评分单元：" + formatNumber(sampleRows.length) + (supersededRevisionText ? " · " + supersededRevisionText : "");
    if (tokenNode) {
      tokenNode.title = tokenBudgetScopeText(run);
      tokenNode.textContent = modelUsageTokenProgressText(run, stageProgress, candidate);
    }
    if (heartbeatNode) {
      if (heartbeatStalled) {
        heartbeatNode.textContent = "执行心跳：" + (heartbeatState.timestamp ? formatTime(heartbeatState.timestamp) : "—") + " · " + heartbeatState.age_text + " · 疑似停滞，正在核验";
      } else if (stageProgress && stageProgress.updated_at && stageProgress.live) {
        heartbeatNode.textContent = "评测心跳：" + formatTime(heartbeatState.timestamp || stageProgress.updated_at) + (heartbeatState.age_text ? " · " + heartbeatState.age_text : "") + (stageProgress.progress_kind === "settling" ? " · 模型执行与宿主结算中" + inFlightText + queuedText + awaitingSubmissionText + awaitingSettlementText + gatewayAttemptsText : stageProgress.progress_kind === "waiting" ? " · 网关执行中" + inFlightText + queuedText + awaitingSubmissionText + awaitingSettlementText : causalWaveText ? " · 已提交" + causalWaveText + inFlightText + queuedText + awaitingSubmissionText + awaitingSettlementText : inFlightText + queuedText + awaitingSubmissionText + awaitingSettlementText) + (dshActivity ? " · " + dshActivity.heartbeat : "");
      } else if (stageProgress && stageProgress.updated_at) {
        heartbeatNode.textContent = (paused ? "最近评测记录：" : "评测证据：") + formatTime(stageProgress.updated_at) + (pausedDrained ? " · 暂停后请求已排空" + inFlightText + queuedText : paused ? " · 正在排空暂停前请求" + inFlightText + queuedText + awaitingSubmissionText + awaitingSettlementText : stageProgress.evidence_qualifier ? " · " + stageProgress.evidence_qualifier : "");
      } else if (liveAllowed) {
        heartbeatNode.textContent = dshActivity ? "DSH 活动：" + dshActivity.heartbeat : "评测心跳：等待首个波次完成";
      } else {
        heartbeatNode.textContent = "评测证据：" + (executionEvidenceQualifier(run) || "无逐批时间记录");
      }
    }
    var latestEvent = state.events && state.events[0];
    activityNode.textContent = "最近活动：" + (latestEvent ? formatTime(latestEvent.occurred_at) + " · " + compactTechnicalText(eventTitle(latestEvent, run)) : formatTime(run.updated_at));
    renderExecutionDiagnostics(run);
    stageNode.innerHTML = stages.map(function (item) {
      var presented = executionStatusForRun(run, item.value);
      var tone = executionStatusClass(presented);
      var stageStatusText = executionStageText(presented, item.key);
      return "<div class=\"execution-stage-chip " + tone + "\"><span>" + escapeHTML(executionStageLabels[item.key]) + "</span><strong>" + escapeHTML(stageStatusText) + "</strong></div>";
    }).join("");
    renderExecutionCandidate(candidate, run, round, stages);
    renderExecutionSamples(candidate, run);
    renderExecutionPlan(candidate, run);
  }

  function adaptiveTrajectoryOperationText(operation) {
    var item = operation && typeof operation === "object" ? operation : {};
    var name = item.name || item.op || item.program_id || "局部修改";
    var target = item.target == null ? "" : " · " + executionSafeValue(item.target);
    var value = item.value == null ? "" : " = " + executionSafeValue(item.value);
    return compactTechnicalText(name + target + value);
  }
  function adaptiveTrajectoryUsesPairedMode(run, lane) {
    var configuration = run && run.configuration && typeof run.configuration === "object" ? run.configuration : {};
    var schedule = run && run.optimization_schedule || configuration.optimization_schedule || {};
    return String(schedule.local_evaluation_mode || lane && lane.local_evaluation_mode || "").toLowerCase() === "paired_champion_challenger"
      || String(lane && lane.score_comparability || "").toLowerCase() === "same_batch_cohort_paired_comparison";
  }
  function adaptiveTrajectoryDecisionText(batch, paired) {
    var outcome = String(batch && batch.edit_outcome || "").toLowerCase();
    var decision = String(batch && batch.edit_decision || "").toLowerCase();
    var reason = String(batch && batch.edit_reason || "").toLowerCase();
    if (paired) {
      return pairedDecisionText(batch && batch.comparison_decision)
        || (outcome === "rejected" ? "修改提案未通过规则检查" : "")
        || (String(batch && batch.status || "").toLowerCase() === "running" ? "正在用同批数据比较" : "等待同批效果比较");
    }
    if (outcome === "rolled_back") { return "安全门回退"; }
    if (reason && outcome === "kept") { return "证据不足或安全门阻断，本批不改"; }
    return {
      applied: "新方案已生成，尚未比较效果", kept: "沿用当前版本", rejected: "修改提案未通过规则检查"
    }[outcome] || {
      mutate: "建议局部修改", keep: "建议保持"
    }[decision] || (String(batch && batch.status || "").toLowerCase() === "running" ? "评测中" : "等待局部决策");
  }
  function renderAdaptiveTrajectories(run) {
    var table = $("#adaptive-trajectory-table");
    var count = $("#adaptive-trajectory-count");
    if (!table || !count) { return; }
    var lanes = run && Array.isArray(run.adaptive_trajectories) ? run.adaptive_trajectories : [];
    var rows = [];
    lanes.forEach(function (lane) {
      (Array.isArray(lane.batches) ? lane.batches : []).forEach(function (batch) { rows.push({lane: lane, batch: batch || {}}); });
    });
    count.textContent = lanes.length ? formatNumber(lanes.length) + " 个候选 · " + formatNumber(rows.length) + " 个批次" : "等待入围方案";
    if (!rows.length) {
      table.innerHTML = lanes.length
        ? "<tr><td colspan=\"6\" class=\"empty-state\">入围方案已开始优化，正在等待首批比较结果。</td></tr>"
        : "<tr><td colspan=\"6\" class=\"empty-state\">选出 2 个入围方案后，这里会显示每批修改、效果比较和选择结果。</td></tr>";
      return;
    }
    table.innerHTML = rows.map(function (entry) {
      var lane = entry.lane;
      var batch = entry.batch;
      var paired = adaptiveTrajectoryUsesPairedMode(run, lane);
      var score = batch.score == null || batch.score === "" ? NaN : Number(batch.score);
      var rawCoverage = batch.prediction_cell_coverage != null ? batch.prediction_cell_coverage : batch.coverage;
      var coverage = rawCoverage == null ? NaN : Number(rawCoverage);
      var coverageText = Number.isFinite(coverage) ? formatNumber(coverage <= 1 ? coverage * 100 : coverage, 1) + "%" : "—";
      var originSuccessRate = batch.origin_success_rate == null ? NaN : Number(batch.origin_success_rate);
      var originSuccessText = Number.isFinite(originSuccessRate) ? formatNumber(originSuccessRate <= 1 ? originSuccessRate * 100 : originSuccessRate, 1) + "%" : "—";
      var originText = formatNumber(batch.origin_count || 0) + " 个预测时点";
      var operations = paired
        ? Array.isArray(batch.next_challenger_operations) ? batch.next_challenger_operations : []
        : Array.isArray(batch.operations) ? batch.operations : [];
      var batchStatus = String(batch.status || "").toLowerCase();
      var editOutcome = String(batch.edit_outcome || "").toLowerCase();
      var editDecision = String(batch.edit_decision || "").toLowerCase();
      var operationDetails = operations.map(function (operation) { return "<span class=\"adaptive-trajectory-operation\">" + escapeHTML(adaptiveTrajectoryOperationText(operation)) + "</span>"; }).join("");
      var operationHtml = paired
        ? editOutcome === "rejected"
          ? "<span>修改提案未通过规则检查</span>" + (batch.edit_reason ? "<small>规则检查原因 " + escapeHTML(compactTechnicalText(batch.edit_reason)) + "</small>" : "") + operationDetails
          : batch.next_challenger_revision_id
            ? "<span>下一次比较的新版本已生成，等待同批效果比较</span>" + operationDetails
            : batchStatus === "running" || !batch.comparison_decision
              ? "<span>等待本批效果比较</span>"
              : "<span>本批比较后没有新修改</span>"
        : operations.length
          ? operationDetails
          : editOutcome === "kept" || editDecision === "keep"
            ? "<span>本批不修改</span>"
            : batchStatus === "running" || !editOutcome && !editDecision
              ? "<span>等待本批评测与局部决策</span>"
              : "<span>已记录决策，未公开局部操作摘要</span>";
      var outcome = adaptiveTrajectoryDecisionText(batch, paired);
      var comparisonDecision = String(batch.comparison_decision || "").toLowerCase();
      var outcomeTone = paired
        ? comparisonDecision === "challenger_promoted" ? "pill-green" : comparisonDecision === "initial_champion" || batchStatus === "running" ? "pill-blue" : "pill-neutral"
        : editOutcome === "applied" ? "pill-blue" : editOutcome === "rejected" ? "pill-amber" : batchStatus === "running" ? "pill-blue" : "pill-neutral";
      var revisionHtml = paired
        ? "<small>原版本</small><code title=\"" + escapeHTML(batch.champion_before_revision_id || "") + "\">" + escapeHTML(shortId(batch.champion_before_revision_id || "—")) + "</code><span>对照</span><code title=\"" + escapeHTML(batch.challenger_revision_id || "") + "\">" + escapeHTML(shortId(batch.challenger_revision_id || "—")) + "</code><small>比较后保留版本</small><code title=\"" + escapeHTML(batch.champion_after_revision_id || "") + "\">" + escapeHTML(shortId(batch.champion_after_revision_id || "—")) + "</code>"
        : "<code title=\"" + escapeHTML(batch.candidate_revision_id || "") + "\">" + escapeHTML(shortId(batch.candidate_revision_id || "—")) + "</code><span>→</span><code title=\"" + escapeHTML(batch.active_revision_id || "") + "\">" + escapeHTML(shortId(batch.active_revision_id || "—")) + "</code>";
      var championScore = batch.champion_score == null || batch.champion_score === "" ? NaN : Number(batch.champion_score);
      var challengerScore = batch.challenger_score == null || batch.challenger_score === "" ? NaN : Number(batch.challenger_score);
      var scoreDelta = batch.score_delta == null || batch.score_delta === "" ? NaN : Number(batch.score_delta);
      var minimumScoreDelta = batch.minimum_score_delta == null || batch.minimum_score_delta === "" ? NaN : Number(batch.minimum_score_delta);
      revisionHtml = "<details class=\"technical-details\"><summary>查看版本编号</summary><div>" + revisionHtml + "</div></details>";
      var scoreHtml = paired
        ? "<strong>" + escapeHTML(Number.isFinite(championScore) ? "原版本 " + formatNumber(championScore, 4) : "原版本待评测") + " · " + escapeHTML(Number.isFinite(challengerScore) ? "新版本 " + formatNumber(challengerScore, 4) : "新版本待评测") + "</strong><small>" + escapeHTML(Number.isFinite(scoreDelta) ? "提高值 " + signedNumber(scoreDelta, 4) : "提高值待计算") + " · " + escapeHTML(Number.isFinite(minimumScoreDelta) ? "须超过 " + formatNumber(minimumScoreDelta, 4) : "阈值待确认") + "</small>"
        : "<strong>得分 " + escapeHTML(Number.isFinite(score) ? formatNumber(score, 4) : "等待评测") + "</strong><small>预测完成率 " + escapeHTML(originSuccessText) + " · 评分单元覆盖率 " + escapeHTML(coverageText) + "</small>";
      var executionCounts = [batch.succeeded_origins != null ? "成功时点 " + formatNumber(batch.succeeded_origins) : "", batch.failed_origins != null ? "失败时点 " + formatNumber(batch.failed_origins) : "", Number(batch.fallback_scoring_cells) > 0 ? "备用评分项 " + formatNumber(batch.fallback_scoring_cells) : ""].filter(Boolean).join(" · ");
      return "<tr>"
        + "<td><div class=\"adaptive-trajectory-cell\"><strong>第 " + escapeHTML(formatNumber(Number(lane.generation || 0) + 1)) + " 轮 · " + escapeHTML(shortId(lane.candidate_id)) + "</strong><small>轨迹 " + escapeHTML(formatNumber(lane.completed_batch_count || 0)) + " / " + escapeHTML(formatNumber(lane.batch_count || 0)) + "</small>" + (lane.strategy_label ? "<small>" + escapeHTML(lane.strategy_label) + "</small>" : "") + "</div></td>"
        + "<td><div class=\"adaptive-trajectory-cell\"><strong>批次 " + escapeHTML(formatNumber(batch.batch_index)) + " / " + escapeHTML(formatNumber(batch.batch_count || lane.batch_count || 0)) + "</strong><span>" + escapeHTML(originText) + "</span><details class=\"technical-details\"><summary>数据批次编号</summary><code>" + escapeHTML(batch.cohort_digest || "未记录") + "</code></details></div></td>"
        + "<td><div class=\"adaptive-trajectory-cell\">" + revisionHtml + "</div></td>"
        + "<td><div class=\"adaptive-trajectory-cell\">" + scoreHtml + "<small>" + escapeHTML(executionCounts || "批次执行统计待核验") + "</small><small class=\"adaptive-trajectory-warning\">" + (paired ? "仅本批新旧版本可直接比较" : "不同批次得分不能直接比较") + "</small></div></td>"
        + "<td><div class=\"adaptive-trajectory-cell\">" + operationHtml + "</div></td>"
        + "<td><div class=\"adaptive-trajectory-cell\"><span class=\"pill " + outcomeTone + "\">" + escapeHTML(outcome) + "</span>" + (batch.comparison_reason || batch.edit_reason ? "<small>原因：" + escapeHTML(evolutionCheckText(batch.comparison_reason || batch.edit_reason)) + "</small>" : "") + "<small>" + escapeHTML(formatDate(batch.created_at)) + "</small></div></td>"
        + "</tr>";
    }).join("");
  }

  function evolutionEvidenceArmLabel(arm) {
    return {finalist_1: "入围方案 1", finalist_2: "入围方案 2", incumbent: "上一冠军"}[String(arm || "")] || "对照方案";
  }
  function evolutionCheckText(value) {
    return {
      incompatible_comparison_contract: "新旧版本的比较条件不一致",
      champion_evaluation_incomplete: "原版本的评测结果不完整",
      challenger_evaluation_incomplete: "新版本的评测结果不完整",
      challenger_safety_gate_failed: "新版本未通过安全检查",
      challenger_cell_regression: "部分预测指标退化超过允许范围",
      cell_regression: "部分预测指标退化超过允许范围",
      no_positive_score_delta: "同批得分没有提高",
      below_practical_delta: "提高值未超过本轮规定的阈值",
      below_practical_score_delta: "提高值未超过认证阈值",
      challenger_improved: "同批得分提高，满足保留条件",
      scientific_gate_failed: "科学约束检查未通过",
      constraint_violations: "存在约束违规",
      coverage_failed: "有效预测覆盖不足",
      objective_grid_incomplete: "预测目标或时距不完整",
      strict_agent_chain_failed: "预测执行过程未通过完整性检查",
      screening_exploration_only: "初筛仅供探索，不满足认证条件"
    }[String(value || "")] || compactTechnicalText(value);
  }
  function evolutionEvidenceSigned(value) {
    var number = value == null || value === "" ? NaN : Number(value);
    if (!Number.isFinite(number)) { return "待计算"; }
    return (number > 0 ? "+" : "") + formatNumber(number, 4);
  }
  function generationSearchIsCertified(decision) {
    return Boolean(decision && ["finalist_1", "finalist_2"].indexOf(decision.selected_arm) >= 0
      && (decision.certification_selected_arm === decision.selected_arm
        || (decision.certification_selected_arm == null && ["certification_selected", "certified"].indexOf(decision.selected_search_certification_status) >= 0)));
  }
  function renderEvolutionEvidence(run) {
    var status = $("#evolution-evidence-status");
    var championNode = $("#global-champion-card");
    var decisionNode = $("#generation-decision-list");
    var capacityNode = $("#evolution-capacity-note");
    if (!status || !championNode || !decisionNode || !capacityNode) { return; }
    var evidence = run && run.evolution_evidence && typeof run.evolution_evidence === "object" ? run.evolution_evidence : {};
    var legacyAudit = evidence.protocol_scope === "legacy_audit";
    var positiveDeltaV3 = evidence.protocol_scope === "positive_delta_search_v3";
    var protocolLabel = legacyAudit
      ? "旧版更新记录，跨批次不可直接比较"
      : positiveDeltaV3
      ? "同批提高则继续优化，稳健认证另行判断"
      : "旧版严格对照规则";
    var protocolHtml = "<div class=\"evolution-protocol-label\"><span>选择协议</span><strong>" + escapeHTML(protocolLabel) + "</strong></div>";
    var champion = positiveDeltaV3 && evidence.search_version && typeof evidence.search_version === "object"
      ? evidence.search_version
      : evidence.global_champion && typeof evidence.global_champion === "object" ? evidence.global_champion : null;
    var certifiedVersion = evidence.certified_version && typeof evidence.certified_version === "object" ? evidence.certified_version : null;
    var decisions = Array.isArray(evidence.generation_decisions) ? evidence.generation_decisions : [];
    var committedDecisions = decisions.filter(function (decision) { return decision.committed !== false; });
    var latest = committedDecisions.length ? committedDecisions[committedDecisions.length - 1] : null;
    var latestSearchCertified = generationSearchIsCertified(latest);
    var latestHasDifferentCertification = Boolean(latest && latest.certification_selected_arm && latest.certification_selected_arm !== latest.selected_arm);
    if (!run) {
      status.className = "pill pill-neutral";
      status.textContent = "等待运行";
      championNode.innerHTML = "<div class=\"empty-state\">启动运行后，这里会分别显示下一轮搜索版本与严格认证状态。</div>";
      decisionNode.innerHTML = "<div class=\"empty-state\">完成首轮 holdout 后显示三臂同批决策证据。</div>";
      capacityNode.innerHTML = "尚无数据容量评估。";
      return;
    }
    status.className = "pill " + (!legacyAudit && latest && latest.selected_arm !== "incumbent" ? latestSearchCertified ? "pill-green" : "pill-blue" : !legacyAudit && latest ? "pill-blue" : "pill-neutral");
    status.textContent = legacyAudit ? "旧协议审计" : positiveDeltaV3 && latest
      ? latest.selected_arm === "incumbent" ? "未取得正增益，沿用优化版本" : latestSearchCertified ? "已保留用于优化，并成为稳健认证版本" : latestHasDifferentCertification ? "已保留用于优化，认证版本为另一方案" : "已保留用于优化，认证结果见下方"
      : latest
      ? latest.selected_arm === "incumbent" ? "上一冠军继续保留" : "挑战者已晋升"
      : champion ? "初始基线已冻结" : "等待冻结基线";
    championNode.innerHTML = protocolHtml + (positiveDeltaV3 && champion
      ? "<div><span>下一轮搜索版本</span><strong>" + escapeHTML(champion.is_initial_seed ? "初始种子基线" : "第 " + formatNumber(champion.generation || 0) + " 代正增益版本") + "</strong><small>" + escapeHTML(shortId(champion.candidate_id || "未知候选")) + " · " + escapeHTML(shortId(champion.candidate_revision_id || "未知修订")) + "</small></div>"
        + "<div><span>本轮比较得分</span><strong>" + escapeHTML(champion.absolute_score == null ? "等待首轮 holdout" : formatNumber(champion.absolute_score, 4)) + "</strong><small>得分可为负；是否保留取决于同批提高值</small></div>"
        + "<div><span>稳健认证版本</span><strong>" + escapeHTML(certifiedVersion ? "已有认证版本" : "暂无认证版本") + "</strong><small>" + escapeHTML(certifiedVersion ? shortId(certifiedVersion.candidate_revision_id || "认证版本") + "；不一定是当前优化版本" : "提高幅度、统计稳定性、各项退化和科学检查分别判断") + "</small><small>发布前仍需独立最终验证</small></div>"
      : champion
      ? "<div><span>" + escapeHTML(legacyAudit ? "旧协议保留结果" : "全局最优") + "</span><strong>" + escapeHTML(legacyAudit ? "仅供审计，不按 v2 全局冠军解释" : champion.is_initial_seed ? "初始种子基线" : "第 " + formatNumber(champion.generation || 0) + " 代晋升版本") + "</strong><small>" + escapeHTML(shortId(champion.candidate_id || "未知候选")) + " · " + escapeHTML(shortId(champion.candidate_revision_id || "未知修订")) + "</small></div>"
        + "<div><span>最近同批绝对分</span><strong>" + escapeHTML(champion.absolute_score == null ? "等待首轮 holdout" : formatNumber(champion.absolute_score, 4)) + "</strong><small>" + escapeHTML(legacyAudit ? "gen0 未验证真实 materialized seed；不回溯套用新协议" : "只与同代同 cohort 两个挑战者比较") + "</small></div>"
      : "<div class=\"empty-state\">正在固化初始种子基线；在首个全局决策前不会覆盖它。</div>");
    decisionNode.innerHTML = decisions.length ? decisions.map(function (decision) {
      var committed = decision.committed !== false;
      var selectedIncumbent = decision.selected_arm === "incumbent";
      var decisionSearchCertified = generationSearchIsCertified(decision);
      var decisionLabel = !committed ? "比较已计算，等待固化" : positiveDeltaV3 ? selectedIncumbent ? "无正增益，沿用搜索版本" : decisionSearchCertified ? "正增益并成为稳健认证版本" : "正增益，作为下一轮版本" : decision.exploration_only ? "仅探索，禁止晋升" : selectedIncumbent ? "上一冠军胜出" : "挑战者通过门禁";
      var arms = Array.isArray(decision.arms) ? decision.arms : [];
      var armHtml = arms.map(function (arm) {
        var failures = Array.isArray(arm.failures) && arm.failures.length ? "未通过：" + arm.failures.map(compactTechnicalText).join("、") : arm.eligible ? "符合晋升门禁" : "作为同批对照";
        if (positiveDeltaV3) {
          var certificationFailures = Array.isArray(arm.certification_failures) ? arm.certification_failures : [];
          var cellRisk = certificationFailures.indexOf("cell_regression") >= 0 || Number(arm.worst_cell_delta) < -0.01;
          failures = arm.arm === "incumbent" ? "本轮比较基准" : "保留条件：" + (arm.search_eligible === true ? "满足" : arm.search_eligible === false ? "未满足" : "待评估") + " · 认证条件：" + (arm.certification_eligible === true ? "满足（不等于已选为认证版本）" : arm.certification_eligible === false ? "未满足" : "待评估") + (cellRisk ? " · 部分指标有退化风险" : "");
          var reasons = (Array.isArray(arm.search_failures) ? arm.search_failures : []).concat(certificationFailures);
          if (arm.arm !== "incumbent" && reasons.length) {
            failures += "；原因：" + Array.from(new Set(reasons)).map(evolutionCheckText).join("、");
          }
        }
        return "<div class=\"generation-evidence-arm " + (arm.arm === decision.selected_arm ? "is-selected" : "") + "\"><span>" + escapeHTML(evolutionEvidenceArmLabel(arm.arm)) + "</span><strong>绝对分 " + escapeHTML(arm.absolute_score == null ? "待评测" : formatNumber(arm.absolute_score, 4)) + "</strong><small>" + escapeHTML(positiveDeltaV3 ? "同批提高值 " : "相对上一轮版本 ") + escapeHTML(evolutionEvidenceSigned(arm.holdout_delta)) + "</small><small>" + escapeHTML(failures) + "</small></div>";
      }).join("");
      return "<article class=\"generation-evidence-card\"><header><div><span>第 " + escapeHTML(formatNumber(decision.generation || 0)) + " 代</span><strong>" + escapeHTML(decisionLabel) + "</strong></div><span class=\"pill " + (!committed ? "pill-neutral" : selectedIncumbent ? "pill-blue" : "pill-green") + "\">" + escapeHTML(!committed ? positiveDeltaV3 ? "尚未写入搜索版本" : "尚未写入全局冠军" : decision.same_cohort ? "同一组比较数据" : "cohort 异常") + "</span></header><div class=\"generation-evidence-arms\">" + armHtml + "</div>"
        + (decision.replan_required ? "<p class=\"generation-evidence-replan\">连续 " + escapeHTML(formatNumber(decision.consecutive_exploration_generations || 0)) + " 代无候选通过初筛，下一代必须重新规划。</p>" : "") + "</article>";
    }).join("") : "<div class=\"empty-state\">暂无轮末版本选择记录，不能据此判断方案是否改进。</div>";
    var capacity = evidence.capacity && typeof evidence.capacity === "object" ? evidence.capacity : {};
    var finalReservation = capacity.final_validation_reservation;
    var finalValidationLocked = finalReservation && typeof finalReservation === "object"
      && finalReservation.schema_version === "ecologyrsi-dsh.final-validation-reservation/1"
      && finalReservation.status === "locked"
      && typeof finalReservation.reservation_id === "string"
      && finalReservation.reservation_id.trim();
    var finalValidationText = finalValidationLocked
      ? "已记录独立最终验证集预留，发布前仍需外部治理执行。"
      : "尚未锁定独立最终验证集。";
    if (!capacity.available) {
      capacityNode.innerHTML = "历史运行未记录数据容量与复用范围。" + finalValidationText;
    } else if (capacity.evidence_scope === "engineering_exploration_with_reused_origins") {
      capacityNode.innerHTML = "<strong>工程探索证据：</strong>" + escapeHTML(formatNumber(capacity.available_eligible_origins || capacity.available_source_origins || 0)) + " 个可用独立源，计划 " + escapeHTML(formatNumber(capacity.required_unique_origins || capacity.planned_origin_occurrences || 0)) + " 个 origin occurrences，其中 " + escapeHTML(formatNumber(capacity.reused_origin_occurrences || 0)) + " 次复用。结果可用于工程优化，不宣称为完全独立的科学验证。" + finalValidationText;
    } else {
      capacityNode.innerHTML = "<strong>独立源证据：</strong>本轮规划未复用 origin，仍只在当前冻结选择集边界内解释。" + finalValidationText;
    }
  }

  function renderProcessSummary(run) {
    var node = $("#process-summary");
    if (!run) {
      var submitting = state.pendingAction === "create" || state.createStatus && state.createStatus.state === "submitting";
      node.innerHTML = submitting
        ? "<div class=\"empty-state process-submit-progress\"><strong>提交已接收</strong><span>正在创建持久化运行；服务返回运行编号后，页面会自动刷新轮次、候选和样本预测。</span></div>"
        : "<div class=\"empty-state\">创建进化运行后可查看模型、时距、保留方案得分与人工协作摘要。</div>";
      return;
    }
    var configuration = run.configuration || {};
    var retainedScore = runRetainedScore(run);
    var rawObservedScore = runRawBestObservedScore(run);
    var outcomeText = runOutcomeText(run) || displayRunStatusText(run, state.events);
    var sampleAgentTokenBudget = runUsesSampleAgentTokenBudget(run);
    var declaredTokenBudget = sampleAgentTokenBudget;
    var nativeDshRuntime = run.dsh_runtime && run.dsh_runtime.native === true;
    var schedule = run.optimization_schedule || configuration.optimization_schedule || {};
    var formalOrigins = Number(schedule.formal_origin_count_per_finalist || 0);
    var batchOrigins = Number(schedule.local_batch_origin_count || 0);
    var batchCount = formalOrigins > 0 && batchOrigins > 0 ? formalOrigins / batchOrigins : 0;
    var holdoutOrigins = Number(schedule.selection_holdout_origin_count || 0);
    var pairedMode = String(schedule.local_evaluation_mode || "").toLowerCase() === "paired_champion_challenger";
    var formalOriginsPerFinalist = pairedMode && batchCount > 0
      ? batchOrigins + 2 * Math.max(0, batchCount - 1) * batchOrigins
      : formalOrigins;
    var formalCandidateOrigins = Number(schedule.finalist_count || 2) * formalOriginsPerFinalist;
    var fitnessProfile = configuration.fitness_profile && typeof configuration.fitness_profile === "object" ? configuration.fitness_profile : {};
    var vectorTargetCount = Array.isArray(fitnessProfile.expected_targets) ? fitnessProfile.expected_targets.length : 0;
    var vectorHorizonCount = Array.isArray(fitnessProfile.expected_horizons) ? fitnessProfile.expected_horizons.length : 0;
    var vectorCellCount = vectorTargetCount > 0 && vectorHorizonCount > 0 ? vectorTargetCount * vectorHorizonCount : null;
    var generationCandidateOrigins = formalOrigins > 0 && holdoutOrigins > 0
      ? 4 * 64 + formalCandidateOrigins + 3 * holdoutOrigins
      : 0;
    var generationScoringCells = generationCandidateOrigins > 0 && vectorCellCount != null ? generationCandidateOrigins * vectorCellCount : null;
    var values = [
      ["研究领域", catalogReferenceLabel("domain_packs", configuration.domain_pack_id, configuration.domain_pack_id || "未提供")],
      ["策略模型（API）", modelReferenceLabel(configuration.policy_model_id)],
      ["独立评审模型（API）", modelReferenceLabel(configuration.judge_model_id)],
      ["每轮候选", formatNumber(run.candidates_per_generation || 1) + " 个版本"],
      ["候选并发", Number(run.candidate_concurrency) > 0 ? formatNumber(run.candidate_concurrency) + " 个候选" : "历史运行按串行执行"],
      ["入围候选轨迹", batchCount > 0 ? pairedMode
        ? "Top 2 各 1 个 warm-up 批次 + " + formatNumber(Math.max(0, batchCount - 1)) + " 个冠军/挑战者配对微批；两条 lane 共用 " + formatNumber(formalOrigins) + " 个 formal origin occurrences"
        : "Top 2 各 " + formatNumber(batchCount) + " × " + formatNumber(batchOrigins) + " 个预测时点；每批最多 " + formatNumber(schedule.max_local_edits_per_batch) + " 处改动"
        : "等待冻结 schedule"],
      [pairedMode ? "正式配对上限" : "轨迹策略", pairedMode
        ? formatNumber(formalCandidateOrigins) + " candidate-origin occurrences；同 cohort 双臂复用上述 " + formatNumber(formalOrigins) + " 个 shared cohort occurrences，不是新的独立源数据"
        : schedule.strategy_label || "旧版连续更新策略"],
      ["轮末同 cohort 比较", holdoutOrigins > 0 ? "F1 / F2 / 上一冠军各 " + formatNumber(holdoutOrigins) + " 个预测时点" : "等待冻结 holdout"],
      ["单轮执行预算", generationCandidateOrigins > 0 ? pairedMode
        ? formatNumber(generationCandidateOrigins) + " candidate-origin execution occurrences" + (generationScoringCells != null ? " = " + formatNumber(generationScoringCells) + " scoring cells" : "")
        : formatNumber(generationCandidateOrigins) + " candidate-origins"
        : "等待冻结 schedule"],
      ["单时点向量链", vectorCellCount != null ? formatNumber(vectorTargetCount) + " 目标 × " + formatNumber(vectorHorizonCount) + " 时距 = " + formatNumber(vectorCellCount) + " 评分单元原子提交" : "等待绑定"],
      ["逐样本并发", Number(run.sample_concurrency) > 0 ? formatNumber(run.sample_concurrency) + " 条预测时点链准入" : "历史运行未配置"],
      [nativeDshRuntime ? "DSH 上下文管理" : declaredTokenBudget ? tokenBudgetSubjectText(run) + " Token 硬预算" : "Token 账本（历史口径）", nativeDshRuntime ? "Session 压缩与输出长度由 DSH 统一管理" : Number(run.token_limit) > 0 ? formatNumber(run.token_limit) : "仅计量"],
      [nativeDshRuntime ? "用量来源" : "Token 计量范围", declaredTokenBudget ? tokenBudgetScopeText(run) : nativeDshRuntime ? "当前压力来自 TokenMeter；累计用量仅采信 Session provider 回执" : tokenBudgetScopeText(run)],
      ["当前保留得分", retainedScore == null ? "尚未产生" : formatNumber(retainedScore) + "（实际晋升序列）"],
      ["原始最高观测（跨窗口不可直接比较）", rawObservedScore == null ? "尚未产生" : formatNumber(rawObservedScore)],
      ["本次运行结果", outcomeText]
    ];
    node.innerHTML = values.map(function (item) {
      return "<div class=\"dataset-stat\"><span>" + escapeHTML(item[0]) + "</span><strong title=\"" + escapeHTML(item[1]) + "\">" + escapeHTML(item[1]) + "</strong></div>";
    }).join("");
  }

  function renderProcess() {
    var run = state.activeRun;
    var pill = $("#process-status-pill");
    var liveAllowed = executionRunAllowsLiveStatus(run);
    var advancing = Boolean(run && liveAllowed && (state.pendingAction === "advance" || state.pendingAction === "auto-advance"));
    var autoActive = Boolean(run && liveAllowed && runHasContinuousAutoProgress(run));
    var autoBlocked = false;
    var submitting = !run && (state.pendingAction === "create" || state.createStatus && state.createStatus.state === "submitting");
    var schedulerQueue = executionSchedulerQueueInfo(run);
    pill.className = "pill " + (submitting || advancing || schedulerQueue ? "pill-blue" : autoBlocked ? "pill-red" : autoActive ? "pill-blue" : run ? displayRunStatusClass(run, state.events) : "pill-neutral");
    pill.textContent = submitting ? "正在创建运行" : advancing ? "正在执行第 " + (Number(run.generation || 0) + 1) + " 轮" : schedulerQueue ? "后台排队中" : autoBlocked ? "自动推进已暂停" : autoActive ? "自动连续推进" : run ? displayRunStatusText(run, state.events) : "暂无运行";
    $("#workspace-process").setAttribute("aria-busy", String(submitting || advancing || autoActive));
    $("#projection-revision").textContent = run ? "状态视图版本 " + (run.projection_revision || "—") : "状态视图版本 —";
    renderProcessSummary(run);
    renderAutonomyProgress(run);
    renderExecutionMonitor(run);
    renderEvolutionEvidence(run);
    renderAdaptiveTrajectories(run);
    renderTrajectory();
    renderRoundStages();
    var list = $("#event-list");
    var toggle = $("#toggle-events-button");
    var eventTotal = Math.max(state.events.length, Number(state.eventTotal || 0));
    toggle.hidden = eventTotal <= 12;
    toggle.textContent = state.showAllEvents
      ? "收起事件（当前最近 " + state.events.length + " / 总计 " + eventTotal + "）"
      : "查看最近 " + state.events.length + " 条（总计 " + eventTotal + "）";
    toggle.setAttribute("aria-expanded", String(state.showAllEvents));
    var visibleEvents = state.showAllEvents ? state.events : state.events.slice(0, 12);
    list.innerHTML = visibleEvents.length ? visibleEvents.map(function (event) {
      var detail = eventDetail(event, run);
      return "<div class=\"event-row\"><span class=\"event-time\">" + escapeHTML(formatTime(event.occurred_at)) + "</span><div class=\"event-main\"><span class=\"event-title\">" + escapeHTML(eventTitle(event, run)) + "</span>" + (detail ? "<span class=\"event-detail\" title=\"" + escapeHTML(detail) + "\">" + escapeHTML(compactTechnicalText(detail)) + "</span>" : "") + "</div><span class=\"pill " + eventTone(event, run) + "\">" + escapeHTML(eventCategory(event.type)) + "</span></div>";
    }).join("") : "<div class=\"empty-state\">当前进化运行暂无事件。</div>";
  }
