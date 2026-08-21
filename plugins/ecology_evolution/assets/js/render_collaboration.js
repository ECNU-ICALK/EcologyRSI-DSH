"use strict";

  function capabilityLabel(value) {
    return {
      "training.data.read": "查看训练拟合数据", "evaluation.samples.read": "查看候选逐样本结果", "evolution.run.create": "创建进化运行", "evolution.run.advance": "推进进化轮次",
      "evolution.projection.read": "查看脱敏状态", "run.control": "控制进化运行", "run.archive": "归档与恢复运行", "run.delete": "永久删除已归档运行", "intervention.write": "提交专家意见与答复",
      "hidden.read": "读取隐藏评测集", "final.read": "读取最终评测集", "release.write": "执行正式发布"
    }[value] || "扩展能力：" + value;
  }
  function interventionOverrideText(overrides) {
    if (!overrides || typeof overrides !== "object") { return ""; }
    return Object.keys(overrides).sort().map(function (key) {
      return (parameterLabels[key] || key) + "=" + formatNumber(overrides[key]);
    }).join("；");
  }
  function interventionApplicationStatus(item) {
    var explicit = String(item && item.application_status || "").toLowerCase();
    if (["recorded", "applied", "enforced"].indexOf(explicit) >= 0) { return explicit; }
    if (item && item.enforced === true) { return "enforced"; }
    if (item && item.applied === true) { return "applied"; }
    if (item && item.recorded === true) { return "recorded"; }
    var legacy = String(item && item.status || "");
    if (legacy === "已强制执行") { return "enforced"; }
    if (legacy === "已应用") { return "applied"; }
    return "recorded";
  }
  function interventionApplicationText(status, item) {
    if (status === "recorded") { return item && item.applied_proposal_id ? "仅记录（未执行）" : "等待下一轮"; }
    return { applied: "已应用", enforced: "已强制执行" }[status] || "仅记录（未执行）";
  }
  function interventionApplicationClass(status) {
    return status === "enforced" ? "pill-green" : status === "applied" ? "pill-blue" : "pill-amber";
  }
  function interventionExecutionText(item) {
    var details = [];
    if (item.parameter) { details.push("执行参数：" + (parameterLabels[item.parameter] || item.parameter)); }
    if (item.direction) { details.push("调整方向：" + ({ increase: "增加", decrease: "减少", up: "增加", down: "减少" }[String(item.direction).toLowerCase()] || item.direction)); }
    if (item.operator && item.bound != null) { details.push("约束：" + item.operator + formatNumber(item.bound)); }
    if (item.previous_value != null || item.result_value != null) { details.push("执行值：" + formatNumber(item.previous_value) + " → " + formatNumber(item.result_value)); }
    if (item.step != null) { details.push("调整步长：" + formatNumber(item.step)); }
    if (item.reason) { details.push("执行说明：" + compactTechnicalText(item.reason)); }
    return details.join(" · ");
  }
  function consultationValueText(value, fallback) {
    if (isBlank(value)) { return fallback != null ? fallback : "—"; }
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") { return String(value); }
    try { return JSON.stringify(value); } catch (error) { return fallback != null ? fallback : "—"; }
  }
  function consultationOptions(item) {
    return (Array.isArray(item && item.options) ? item.options : []).map(function (option, index) {
      if (option && typeof option === "object") {
        var id = consultationValueText(option.id != null ? option.id : option.value, "option-" + index);
        return { id: id, label: consultationValueText(option.label != null ? option.label : option.name, id) };
      }
      var value = consultationValueText(option, "option-" + index);
      return { id: value, label: value };
    });
  }
  function consultationConfidenceText(value) {
    if (isBlank(value) || !Number.isFinite(Number(value))) { return "置信度未记录"; }
    var number = Number(value);
    var percent = Math.abs(number) <= 1 ? number * 100 : number;
    return "模型置信度 " + Math.max(0, Math.min(100, percent)).toFixed(percent % 1 ? 1 : 0) + "%";
  }
  function consultationUncertaintyText(value) {
    return {
      scientific_assumption: "科学假设", data_interpretation: "数据解释", model_selection: "模型选择", tradeoff: "权衡判断", governance_boundary: "治理边界",
      scientific: "科学判断", method: "方法选择", methodology: "方法选择", data: "数据解释", parameter: "参数边界",
      safety: "安全边界", governance: "治理边界", domain: "领域知识", ambiguous: "语义歧义"
    }[String(value || "").toLowerCase()] || consultationValueText(value, "未分类不确定性");
  }
  function consultationMetaHTML(item) {
    var parts = [];
    if (item.source_generation != null) { parts.push("来源：第 " + consultationValueText(item.source_generation) + " 轮"); }
    if (item.candidate_id) { parts.push("候选：" + shortId(item.candidate_id)); }
    if (item.model_id) { parts.push("提问模型：" + shortId(item.model_id)); }
    parts.push("类型：" + consultationUncertaintyText(item.uncertainty_type));
    parts.push(consultationConfidenceText(item.confidence));
    if (Array.isArray(item.requested_expertise) && item.requested_expertise.length) { parts.push("希望获得：" + item.requested_expertise.map(function (value) { return consultationValueText(value); }).join("、")); }
    if (item.created_at) { parts.push("提出时间：" + formatDate(item.created_at)); }
    return parts.map(function (part) { return "<span>" + escapeHTML(part) + "</span>"; }).join("");
  }
  function consultationContextHTML(item) {
    var context = consultationValueText(item && item.context, "");
    var fallback = consultationValueText(item && item.fallback_assumption, "");
    return (context ? "<p class=\"consultation-context\">" + escapeHTML(context) + "</p>" : "") +
      (fallback ? "<p class=\"consultation-context consultation-fallback\"><strong>未答复时：</strong> " + escapeHTML(fallback) + "</p>" : "");
  }
  function consultationOptionsHTML(item) {
    var options = consultationOptions(item);
    if (!options.length) { return ""; }
    return "<div class=\"consultation-options\" aria-label=\"模型提出的参考选项\">" + options.map(function (option) {
      return "<span class=\"consultation-option\">" + escapeHTML(option.label) + "</span>";
    }).join("") + "</div>";
  }
  function captureExpertConsultationDrafts() {
    $$("#pending-consultations [data-consultation-answer-form]").forEach(updateExpertConsultationDraftFromForm);
  }
  function updateExpertConsultationDraftFromForm(form) {
    if (!form || !form.dataset) { return null; }
    var runId = form.dataset.runId || state.activeRun && state.activeRun.id;
    var consultationId = form.dataset.consultationId;
    if (!runId || !consultationId) { return null; }
    var draft = expertConsultationDraft(runId, consultationId);
    var elements = form.elements || {};
    draft.answer = elements.answer ? elements.answer.value : draft.answer;
    draft.selected_option = elements.selected_option ? elements.selected_option.value : draft.selected_option;
    draft.answered_by = elements.answered_by ? elements.answered_by.value : draft.answered_by;
    return draft;
  }
  function pendingConsultationAnswerHTML(item, run, ended) {
    var consultationId = expertConsultationId(item);
    var draft = expertConsultationDraft(run.id, consultationId);
    var options = consultationOptions(item);
    var disabled = state.busy || !hasCapability("intervention.write");
    var disabledAttribute = disabled ? " disabled" : "";
    var optionField = options.length ? "<label><span>参考选项（可选）</span><select name=\"selected_option\"" + disabledAttribute + "><option value=\"\">不指定选项</option>" + options.map(function (option) {
      return "<option value=\"" + escapeHTML(option.id) + "\"" + (String(draft.selected_option) === String(option.id) ? " selected" : "") + ">" + escapeHTML(option.label) + "</option>";
    }).join("") + "</select></label>" : "";
    var submitting = state.pendingAction === "expert-consultation:" + consultationId;
    var hint = !hasCapability("intervention.write") ? "当前 DSH 会话未授予提交专家答复的能力。" : ended ? "迟到答复仅进入审计记录，不会改写已完成的进化结果。" : "可稍后答复；提交后只影响后续轮次。";
    var auditNote = ended ? "<p class=\"consultation-audit-note\"><strong>本运行已结束：</strong>仍可补录专家答复；系统只归档，不设置生效轮次。</p>" : "";
    return "<form class=\"consultation-answer-form\" data-consultation-answer-form data-run-id=\"" + escapeHTML(run.id) + "\" data-consultation-id=\"" + escapeHTML(consultationId) + "\">" +
      auditNote +
      optionField +
      "<label><span>专家答复</span><textarea name=\"answer\" rows=\"3\" maxlength=\"4000\" placeholder=\"给出判断、依据或建议的后续检查\" required" + disabledAttribute + ">" + escapeHTML(draft.answer) + "</textarea></label>" +
      "<label><span>答复人</span><input name=\"answered_by\" type=\"text\" maxlength=\"120\" value=\"" + escapeHTML(draft.answered_by) + "\" placeholder=\"姓名或工作编号\" required" + disabledAttribute + "></label>" +
      "<div class=\"form-actions\"><button class=\"button button-primary button-small\" type=\"submit\"" + disabledAttribute + ">" + (submitting ? "正在提交答复" : ended ? "补录专家答复" : "提交专家答复") + "</button><span>" + escapeHTML(hint) + "</span></div></form>";
  }
  function renderPendingConsultation(item, run, ended) {
    var status = item.non_blocking === false
      ? "<span class=\"pill pill-red\">治理阻塞</span>"
      : "<span class=\"pill pill-green\">非阻塞 · 运行继续</span>";
    return "<article class=\"consultation-item consultation-item-pending\"><div class=\"consultation-main\"><div class=\"consultation-question-line\"><h3>" + escapeHTML(consultationValueText(item.question, "未提供问题内容")) + "</h3><div class=\"consultation-status\"><span class=\"pill pill-amber\">待答复</span>" + status + "</div></div>" +
      consultationContextHTML(item) + consultationOptionsHTML(item) +
      "<div class=\"consultation-meta\">" + consultationMetaHTML(item) + "</div></div>" + pendingConsultationAnswerHTML(item, run, ended) + "</article>";
  }
  function selectedConsultationOptionText(item) {
    if (isBlank(item.selected_option)) { return ""; }
    var selected = consultationOptions(item).find(function (option) { return String(option.id) === String(item.selected_option); });
    return selected ? selected.label : consultationValueText(item.selected_option);
  }
  function renderAnsweredConsultation(item, run, ended) {
    var applied = item.applied_generation != null;
    var lifecycle = applied ? "已在第 " + consultationValueText(item.applied_generation) + " 轮应用" : ended ? "本运行未应用" : "等待后续轮次应用";
    var lifecycleClass = applied ? "pill-green" : ended ? "pill-neutral" : "pill-blue";
    var effective = item.effective_generation != null ? "计划生效：第 " + consultationValueText(item.effective_generation) + " 轮" : "计划生效：未安排";
    var selected = selectedConsultationOptionText(item);
    var answerMeta = [item.answered_by ? "答复人：" + item.answered_by : "答复人未记录", item.answered_at ? "答复时间：" + formatDate(item.answered_at) : "答复时间未记录", effective, "实际应用：" + (applied ? "第 " + item.applied_generation + " 轮" : "尚未应用")];
    return "<article class=\"consultation-item consultation-item-answered\"><div class=\"consultation-main\"><div class=\"consultation-question-line\"><h3>" + escapeHTML(consultationValueText(item.question, "未提供问题内容")) + "</h3><div class=\"consultation-status\"><span class=\"pill " + lifecycleClass + "\">" + escapeHTML(lifecycle) + "</span>" + (item.non_blocking === false ? "<span class=\"pill pill-red\">治理问题</span>" : "<span class=\"pill pill-green\">非阻塞</span>") + "</div></div>" +
      consultationContextHTML(item) +
      "<p class=\"consultation-answer\"><strong>专家答复：</strong> " + escapeHTML(consultationValueText(item.answer, "未记录答复内容")) + (selected ? "<br><strong>所选参考项：</strong> " + escapeHTML(selected) : "") + "</p>" +
      "<div class=\"consultation-meta\">" + consultationMetaHTML(item) + answerMeta.map(function (part) { return "<span>" + escapeHTML(part) + "</span>"; }).join("") + "</div></div></article>";
  }
  function renderExpertConsultations(run, ended) {
    captureExpertConsultationDrafts();
    var consultations = run && Array.isArray(run.expert_consultations) ? run.expert_consultations : [];
    var pending = consultations.filter(function (item) { return item.status === "pending"; }).sort(function (left, right) {
      return Number(right.source_generation || 0) - Number(left.source_generation || 0);
    });
    var answered = consultations.filter(function (item) { return item.status === "answered"; }).sort(function (left, right) {
      return (Date.parse(right.answered_at || "") || 0) - (Date.parse(left.answered_at || "") || 0);
    });
    $("#pending-consultation-count").textContent = pending.length + " 项";
    $("#answered-consultation-count").textContent = answered.length + " 项";
    $("#pending-consultation-count").className = "pill " + (pending.length ? "pill-amber" : "pill-neutral");
    $("#answered-consultation-count").className = "pill " + (answered.length ? "pill-green" : "pill-neutral");
    $("#pending-consultations").innerHTML = !run ? "<div class=\"empty-state\">请选择一个进化运行。</div>" : pending.length ? pending.map(function (item) { return renderPendingConsultation(item, run, ended); }).join("") : "<div class=\"empty-state\">当前没有等待专家答复的问题。</div>";
    $("#answered-consultations").innerHTML = !run ? "<div class=\"empty-state\">请选择一个进化运行。</div>" : answered.length ? answered.map(function (item) { return renderAnsweredConsultation(item, run, ended); }).join("") : "<div class=\"empty-state\">尚无专家答复记录。</div>";
  }
  function renderCollaboration() {
    var run = state.activeRun;
    var paused = run && run.status === "paused";
    var ended = expertConsultationRunIsTerminal(run);
    var pill = $("#intervention-state-pill");
    pill.className = "pill " + (paused ? "pill-green" : ended || !run ? "pill-neutral" : "pill-blue");
    pill.textContent = paused ? "可提交意见与答复" : ended ? "运行已结束 · 可补录答复" : run ? "可异步答复" : "尚未创建运行";
    $("#submit-intervention").disabled = state.busy || !paused || !hasCapability("intervention.write");
    $("#submit-intervention").textContent = state.pendingAction === "intervention" ? "正在提交专家意见" : "提交专家意见";
    $("#intervention-hint").textContent = !run ? "请先创建进化运行。" : ended ? "运行已结束，不能再提交主动意见；未答咨询仍可补录专家答复并归档。" : !hasCapability("intervention.write") ? "当前 DSH 会话未授予提交专家意见与答复的能力。" : paused ? "提交后请恢复运行，意见将在下一轮处理。" : "主动意见需暂停后提交；模型咨询可在运行中异步答复。";
    renderExpertConsultations(run, ended);
    var candidates = run ? run.candidates.filter(function (candidate) {
      return ["retained", "rejected", "promoted", "accepted"].indexOf(String(candidate.status || "").toLowerCase()) >= 0 && candidate.promotion;
    }) : [];
    var targetSelect = $("#target-candidate-id");
    var selectedTarget = targetSelect.value;
    targetSelect.innerHTML = "<option value=\"\">请选择候选方案</option>" + candidates.map(function (candidate) {
      return "<option value=\"" + escapeHTML(candidate.id) + "\" title=\"" + escapeHTML(candidate.id) + "\">" + escapeHTML(shortId(candidate.id)) + " · 第 " + escapeHTML(candidate.generation || "—") + " 轮</option>";
    }).join("");
    if (selectedTarget && candidates.some(function (candidate) { return candidate.id === selectedTarget; })) { targetSelect.value = selectedTarget; }
    var gate = run && run.gate || {};
    $("#gate-summary").innerHTML = [
      ["训练反馈检查", gate.visible || "not_started", gateText], ["过程约束检查", gate.process || "not_started", gateText],
      ["候选选择范围", run && run.selection_scope || "iterative_training_feedback_only", selectionScopeText],
      ["正式验证", run && run.formal_validation_status || "not_run", formalValidationText],
      ["隐藏评测", gate.hidden || "restricted", gateText], ["正式发布", gate.release || "pending", gateText]
    ].map(function (item) { var value = item[2](item[1]); return "<div class=\"gate-check\"><span>" + item[0] + "</span><strong class=\"" + (value === "通过" || value === "已保留" || value === "外部正式验证通过" ? "ok" : "pending") + "\">" + escapeHTML(value) + "</strong></div>"; }).join("");
    var permissions = ["training.data.read", "evaluation.samples.read", "evolution.run.create", "evolution.run.advance", "evolution.projection.read", "run.control", "run.archive", "run.delete", "intervention.write"].map(function (name) { return { name: name, allowed: hasCapability(name) }; }).concat([{ name: "hidden.read", allowed: false }, { name: "final.read", allowed: false }, { name: "release.write", allowed: false }]);
    $("#capability-list").innerHTML = permissions.map(function (item) { return "<li><span>" + escapeHTML(capabilityLabel(item.name)) + "</span><strong class=\"permission-state " + (item.allowed ? "allowed" : "denied") + "\">" + (item.allowed ? "允许" : "外部受限") + "</strong></li>"; }).join("");
    var interventions = run ? run.interventions : [];
    $("#intervention-history").innerHTML = interventions.length ? interventions.map(function (item) {
      var applicationStatus = interventionApplicationStatus(item);
      var application = interventionApplicationText(applicationStatus, item);
      var generation = item.applied_proposal_id ? "第 " + item.effective_generation + (applicationStatus === "recorded" ? " 轮已处理" : " 轮生效") : "等待下一轮";
      var overrideText = interventionOverrideText(item.parameter_overrides);
      var executionText = interventionExecutionText(item);
      var details = [item.target_candidate_id ? "目标候选：" + shortId(item.target_candidate_id) : "", overrideText ? "参数覆盖：" + overrideText : "", item.applied_proposal_id ? "关联提案：" + shortId(item.applied_proposal_id) : "", executionText].filter(Boolean).join(" · ");
      return "<article class=\"intervention-item\"><strong>" + escapeHTML(interventionLabels[item.kind] || "专家意见") + "</strong><div><p>" + escapeHTML(item.message || "未提供说明") + "</p>" + (details ? "<p class=\"intervention-details\" title=\"" + escapeHTML([item.target_candidate_id, item.applied_proposal_id].filter(Boolean).join("\n")) + "\">" + escapeHTML(details) + "</p>" : "") + "</div><span><span class=\"pill " + interventionApplicationClass(applicationStatus) + "\">" + escapeHTML(application) + "</span> " + escapeHTML(item.created_by || "未署名") + " · " + escapeHTML(generation) + " · " + escapeHTML(formatDate(item.created_at)) + "</span></article>";
    }).join("") : "<div class=\"empty-state\">尚未提交专家主动意见。</div>";
  }

  function updateParameterOverrideHelp() {
    var predictorId = $("#prediction-model-id").value || state.activeRun && state.activeRun.configuration && state.activeRun.configuration.prediction_model_id || "";
    var input = $("#parameter-overrides");
    var help = $("#parameter-overrides-help");
    if (predictorId === "greenhouse-exogenous-ridge@1") {
      input.placeholder = "history_steps=6\nridge_alpha=0.05\nresidual_scale=0.75";
      help.textContent = "可覆盖：目标历史步数 1–12 小时、岭回归正则化强度 0.0001–1、预测残差缩放系数 0–1；每行一个“参数=值”。";
      return;
    }
    if (predictorId === "greenhouse-rolling-residual@1") {
      input.placeholder = "blend=0.4\nwindow=6\nbias_scale=0.8";
      help.textContent = "可覆盖：历史值混合权重 0–1、历史窗口 1–48 小时、偏差校正强度 0–2；每行一个“参数=值”。";
      return;
    }
    input.placeholder = "alpha=0.35\nwindow=5\nwater_threshold=0.4";
    help.textContent = "可覆盖：平滑系数 0.05–0.95、历史窗口 1–30、土壤水分阈值 0.05–0.85；每行一个“参数=值”。";
  }
  function updateInterventionFields() {
    var kind = $("#intervention-kind").value;
    var parameterOverride = kind === "parameter_override";
    $("#parameter-overrides-field").hidden = !parameterOverride;
    $("#parameter-overrides").required = parameterOverride;
    var parentSelection = kind === "parent_selection";
    $("#target-candidate-field").hidden = !parentSelection;
    $("#target-candidate-id").disabled = !parentSelection;
    $("#target-candidate-id").required = parentSelection;
    if (!parentSelection) { $("#target-candidate-id").value = ""; }
    updateParameterOverrideHelp();
  }
