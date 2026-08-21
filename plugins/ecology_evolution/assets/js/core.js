"use strict";

  var $ = function (selector) { return document.querySelector(selector); };
  var $$ = function (selector) { return Array.prototype.slice.call(document.querySelectorAll(selector)); };
  var dataRequestTimeout = 30000;
  var evolutionCommandTimeout = 120000;
  var query = new URLSearchParams(window.location.search);

  var statusLabels = {
    idle: "待启动", created: "已创建", preflight: "预检中", starting: "启动中",
    running: "运行中", paused: "已暂停", promotion_pending: "等待搜索保留决策",
    released: "已发布", completed: "已完成", quarantined: "已隔离",
    cancelled: "已取消", failed: "失败"
  };
  var candidateStatusLabels = {
    accepted: "训练反馈搜索保留", promoted: "训练反馈搜索保留", retained: "训练反馈搜索保留", released: "已发布", evaluating: "训练反馈检查中",
    evaluated: "训练反馈已检查", pending: "等待训练反馈", spawned: "等待训练反馈", rejected: "未保留", failed: "失败", duplicate: "重复版本",
    paused: "已暂停", aborted: "已中止", not_recorded: "未封存"
  };
  var metricLabels = {
    score: "综合得分", passed: "训练反馈检查结果", mae: "平均绝对误差", rmse: "均方根误差",
    water_balance_error: "水量平衡误差", constraint_violations: "约束违规数",
    n: "反馈样本数", non_negative_state: "状态非负约束", partition: "反馈数据分区",
    dataset_digest: "数据集校验值", evaluator_digest: "评测器校验值",
    normalized_rmse: "归一化均方根误差", baseline_normalized_rmse: "基线归一化误差",
    skill_score: "相对基线技能得分", improvement: "相对基线改进量",
    missing_or_nonfinite_rows: "缺失或无效样本数", scientific_pass: "科学约束检查",
    judge_accepted: "独立评审保留建议", judge_guidance: "独立评审意见",
    judge_model_id: "独立评审模型", causal_interpretation: "是否支持因果解释",
    baseline_mae: "基线平均绝对误差", baseline_rmse: "基线均方根误差",
    mean_target_mae_unscaled: "分目标未缩放平均绝对误差均值",
    mean_target_rmse_unscaled: "分目标未缩放均方根误差均值",
    objective_profile: "固定优化目标配置",
    objective_target_weights: "预测目标权重",
    objective_horizon_weighting: "预测时距加权方式",
    per_target_no_regression: "各预测目标均未退化",
    raw_units_comparable_across_targets: "不同目标原始单位可直接比较",
    training_mae: "训练平均绝对误差", training_rmse: "训练均方根误差",
    horizons: "预测时距汇总", prediction_model_id: "预测模型",
    evaluation_scope: "训练反馈范围", selection_scope: "候选选择范围",
    formal_validation_status: "正式验证状态", split_manifest_digest_sha256: "数据划分校验值"
  };
  var targetLabels = { air_temperature: "室内气温", relative_humidity: "室内相对湿度", co2_concentration: "室内 CO2 浓度", soil_moisture: "土壤含水量", soil_water: "土壤含水量" };
  var parameterLabels = {
    blend: "历史值混合权重", window: "历史窗口（小时）", bias_scale: "偏差校正强度",
    history_steps: "目标历史步数（小时）", ridge_alpha: "岭回归正则化强度", residual_scale: "预测残差缩放系数",
    alpha: "平滑系数", water_threshold: "土壤水分阈值"
  };
  var evolutionStageLabels = {
    research: "自主调研", knowledge: "知识检索", implementation: "能力编译",
    proposal: "变更提案", candidate: "候选生成", training: "候选训练",
    evaluation: "独立评测", judge: "独立评审", decision: "搜索保留决策", optimization: "迭代优化"
  };
  var knownModelLabels = {
    "host_parameter_generator@1": "内置有界参数生成器",
    "rule_judge@1": "内置规则独立评审",
    "toy-rolling-water@1": "合成土壤水分滚动预测模型",
    "greenhouse-rolling-residual@1": "温室环境滚动残差预测模型",
    "greenhouse-exogenous-ridge@1": "温室外生变量岭回归残差模型",
    "rolling-residual@1": "滚动残差预测模型"
  };
  var builtinEvolutionModelIds = ["host_parameter_generator@1", "rule_judge@1"];
  var unitLabels = {
    degC: "摄氏度（°C）", percent: "百分比（%）", ppm: "百万分之一（ppm）",
    kg_m2_cumulative: "累计千克/平方米", kg_m2_day: "千克/平方米/天",
    leaves_stem: "片/株", leaves_stem_week: "片/株/周", dS_m: "分西门子/米", pH: "pH",
    L_m2_day: "升/平方米/天", kWh_m2_day: "千瓦时/平方米/天",
    days: "天", minutes: "分钟", hours_day: "小时/天", g_m3: "克/立方米",
    W_m2: "瓦/平方米", m_s: "米/秒", EUR_cumulative: "累计欧元", cm_week: "厘米/周",
    normalized_water_depth_day: "归一化水深/天", fraction: "比例（0–1）", hours: "小时", hour: "小时"
  };
  var fieldLabels = {
    day: "日期序号", date: "日期", precipitation: "降水量", rainfall: "降水量",
    temperature: "气温", temp: "气温", soil_moisture: "土壤含水量",
    observed_soil_moisture: "观测土壤含水量", predicted_soil_moisture: "预测土壤含水量",
    evapotranspiration: "蒸散量", irrigation: "灌溉量", biomass: "生物量",
    water_stress: "水分胁迫指数", split: "数据分区"
  };
  var interventionLabels = {
    guidance: "方向建议", parameter_override: "参数覆盖", constraint: "新增约束", parent_selection: "指定父方案"
  };
  var eventLabels = {
    "run.created": "进化运行已创建", "run.started": "进化运行已启动", "run.paused": "进化运行已暂停",
    "run.resumed": "进化运行已恢复", "run.cancelled": "进化运行已取消", "run.completed": "进化运行已完成",
    "run.failed": "进化运行失败", "generation.started": "进化轮次已开始", "generation.advanced": "进化轮次已推进",
    "generation.completed": "进化轮次已完成", "generation.batch_started": "本轮候选批次已冻结", "generation.analyzed": "本轮结果分析已完成", "generation.champion_selected": "本轮冠军选择已完成", "proposal.submitted": "变更提案已提交", "candidate.spawned": "候选方案已生成",
    "knowledge.retrieved": "本轮知识检索已冻结", "knowledge.assessed": "知识指导结果已判断",
    "research.started": "模型自主调研已开始", "research.completed": "模型自主调研已完成",
    "implementation.started": "预测模型与进化策略能力编译已开始", "implementation.completed": "宿主能力编译已完成",
    "optimization.started": "迭代优化分析已开始", "optimization.completed": "迭代优化决策已完成",
    "candidate.submitted": "候选方案已提交", "candidate.accepted": "候选方案已在训练反馈搜索中保留", "candidate.failed": "候选方案生成失败", "candidate.duplicate": "重复候选已跳过",
    "artifact.recorded": "候选训练产物已记录",
    "evaluation.progress": "真实样本评测正在推进", "evaluation.completed": "训练反馈检查已完成", "promotion.decided": "搜索保留决策已记录", "promotion.pending": "等待搜索保留决策",
    "intervention.recorded": "专家主动意见已记录", "intervention.applied": "专家主动意见处理结果已记录", "intervention.submitted": "专家主动意见已提交",
    "expert_consultation.requested": "模型已提交专家咨询", "expert_consultation.answered": "专家咨询已答复", "expert_consultation.applied": "专家答复已用于后续轮次",
    "consultation.requested": "模型已提交专家咨询", "consultation.answered": "专家咨询已答复", "consultation.applied": "专家答复已用于后续轮次",
    "stage.recorded": "进化阶段状态已更新", "gateway.retry_scheduled": "网关繁忙，已安排延迟重试"
  };
  var state = {
    apiBase: EcologyDSHHost.getPublicContext().apiBase,
    hostContextReceived: false,
    hostContext: EcologyDSHHost.getPublicContext(),
    allowDemo: query.get("demo") === "1",
    usingDemo: false,
    connection: "checking",
    loadState: "loading",
    workspace: "settings",
    catalog: emptyCatalog(),
    runs: [],
    showCancelledEmptyRuns: false,
    showArchivedRuns: false,
    archivedRunCount: 0,
    activeRun: null,
    lastSelectedRunId: null,
    events: [],
    selectedCandidateId: null,
    candidateSelectionPinned: false,
    candidateSamplePage: null,
    candidateSampleLoading: false,
    candidateSampleRefreshing: false,
    candidateSampleError: null,
    candidateSampleUnavailable: false,
    candidateSamplePermissionDenied: false,
    candidateSampleRequest: 0,
    candidateSampleOffset: 0,
    candidateSampleRetryOffset: null,
    candidateSampleLimit: 25,
    candidateSampleLastRequestedAt: 0,
    datasetPage: null,
    datasetLoading: false,
    datasetError: null,
    datasetContext: null,
    datasetRequest: 0,
    datasetPartition: "training_fit",
    showAllFields: false,
    pageOffset: 0,
    pageLimit: 20,
    busy: false,
    refreshing: false,
    pendingAction: null,
    createStatus: null,
    lastError: null,
    commandError: null,
    showAllEvents: false,
    lastUpdated: null,
    viewEpoch: 0,
    contextEpoch: 0,
    runReadRequest: 0,
    commandKeys: {},
    // Answer forms are rebuilt from the latest projection on every poll. Keep
    // unsent text outside the DOM so a refresh cannot erase expert input.
    expertConsultationDrafts: {},
    // Continuous evolution is coordinated by one serial scheduler.  Timer
    // handles stay in page memory and are intentionally omitted from the
    // public projection exported to host integrations.
    autoAdvanceEnabled: true,
    autoAdvanceRunId: null,
    autoAdvanceContextEpoch: null,
    autoAdvanceOptOutRunIds: {},
    autoAdvanceTimer: null,
    autoAdvanceRetry: 0,
    autoAdvanceBlockedRunId: null,
    autoAdvanceError: null,
    autoAdvanceRoundStartedAt: null,
    // The durable worker is independent from the browser.  Keep a small
    // read-only monitor in the page so a newly submitted run exposes its
    // progress immediately instead of waiting for the coarse history refresh.
    runMonitorTimer: null,
    runMonitorRunId: null,
    runMonitorContextEpoch: null,
    runMonitorInFlight: false,
    runMonitorRetry: 0,
    runMonitorLastPollAt: 0,
    autoAdvanceLastDurationMs: null,
    autoAdvanceRoundsCompleted: 0,
    candidateBudgetManual: false
  };

  function emptyCatalog() {
    return { domain_packs: [], datasets: [], unavailable_datasets: [], prediction_models: [], strategies: [], evaluators: [], models: [], dsh_models: [], dsh_models_explicit: false, authenticated_models: [], policy_models: [], judge_models: [], dsh: {} };
  }

  function capabilities() {
    var serviceCapabilities = Array.isArray(state.catalog.dsh.capabilities) ? state.catalog.dsh.capabilities : [];
    var hostCapabilities = state.hostContextReceived && state.hostContext ? state.hostContext.capabilities : null;
    if (!Array.isArray(hostCapabilities)) { return serviceCapabilities; }
    return serviceCapabilities.filter(function (name) { return hostCapabilities.indexOf(name) >= 0; });
  }
  function hasCapability(name) { return capabilities().indexOf(name) >= 0; }

  function expertConsultationId(item) {
    return String(item && (item.consultation_id || item.id) || "");
  }
  function expertConsultationDraftKey(runId, consultationId) {
    return String(runId || "") + "::" + String(consultationId || "");
  }
  function defaultExpertIdentity() {
    var identity = state.hostContext && state.hostContext.identity || {};
    return String(identity.displayName || identity.subjectId || "");
  }
  function expertConsultationDraft(runId, consultationId) {
    var key = expertConsultationDraftKey(runId, consultationId);
    if (!state.expertConsultationDrafts[key]) {
      state.expertConsultationDrafts[key] = { answer: "", selected_option: "", answered_by: defaultExpertIdentity() };
    }
    return state.expertConsultationDrafts[key];
  }
  function clearExpertConsultationDraft(runId, consultationId) {
    delete state.expertConsultationDrafts[expertConsultationDraftKey(runId, consultationId)];
  }

  function escapeHTML(value) {
    return String(value == null ? "" : value).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  }

  function clone(value) { return JSON.parse(JSON.stringify(value)); }
  function nextEpoch() { state.viewEpoch += 1; return state.viewEpoch; }
  function isBlank(value) { return value == null || typeof value === "string" && value.trim() === ""; }
  function displayText(value, fallback) { return isBlank(value) ? fallback || "—" : String(value); }
  function shortId(value) {
    var text = displayText(value);
    return text.length > 32 ? text.slice(0, 19) + "…" + text.slice(-9) : text;
  }
  function formatNumber(value, digits) {
    var number = Number(value);
    if (isBlank(value)) { return "—"; }
    if (!Number.isFinite(number)) { return displayText(value); }
    return Number.isInteger(number) ? number.toLocaleString("zh-CN") : number.toFixed(digits == null ? 4 : digits);
  }
  function runCandidateScore(candidate) {
    if (!candidate || typeof candidate !== "object") { return null; }
    var metrics = candidate.metrics && typeof candidate.metrics === "object" ? candidate.metrics : {};
    var value = candidate.score != null ? candidate.score : metrics.score;
    if (isBlank(value) || !Number.isFinite(Number(value))) { return null; }
    return Number(value);
  }
  function runRawBestObservedScore(run) {
    if (!run) { return null; }
    if (!isBlank(run.best_observed_score) && Number.isFinite(Number(run.best_observed_score))) {
      return Number(run.best_observed_score);
    }
    var candidates = Array.isArray(run.candidates) ? run.candidates : [];
    var identified = candidates.find(function (candidate) { return candidate.id === run.best_observed_candidate_id; });
    var identifiedScore = runCandidateScore(identified);
    if (identifiedScore != null) { return identifiedScore; }
    var scores = candidates.map(runCandidateScore).filter(function (score) { return score != null; });
    return scores.length ? Math.max.apply(null, scores) : null;
  }
  function runRetainedScore(run) {
    if (!run) { return null; }
    var candidates = Array.isArray(run.candidates) ? run.candidates : [];
    var retained = candidates.find(function (candidate) { return candidate.id === run.best_candidate_id; });
    var retainedScore = runCandidateScore(retained);
    if (retainedScore != null) { return retainedScore; }
    var trajectory = Array.isArray(run.trajectory) ? run.trajectory : [];
    for (var index = trajectory.length - 1; index >= 0; index -= 1) {
      var point = trajectory[index] || {};
      var value = point.incumbent_score != null ? point.incumbent_score : point.best_score;
      if (!isBlank(value) && Number.isFinite(Number(value))) { return Number(value); }
    }
    return null;
  }
  function rawBestObservedSummary(run) {
    var label = "原始最高观测（跨窗口不可直接比较）";
    if (!run) { return label + "：尚未产生"; }
    var candidateId = run.best_observed_candidate_id;
    var score = runRawBestObservedScore(run);
    var detail = [candidateId ? shortId(candidateId) : "", score == null ? "" : formatNumber(score)].filter(Boolean).join("（") + (candidateId && score != null ? "）" : "");
    return label + "：" + (detail || "尚未产生");
  }
  function formatBytes(value) {
    var number = Number(value);
    if (!Number.isFinite(number) || number < 0) { return "未提供"; }
    if (number < 1024) { return formatNumber(number) + " 字节"; }
    var units = ["KB", "MB", "GB", "TB"];
    var scaled = number;
    var unitIndex = -1;
    do { scaled /= 1024; unitIndex += 1; } while (scaled >= 1024 && unitIndex < units.length - 1);
    return scaled.toLocaleString("zh-CN", { maximumFractionDigits: scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2 }) + " " + units[unitIndex];
  }
  function signedNumber(value, digits) {
    var number = Number(value);
    if (!Number.isFinite(number)) { return "—"; }
    return (number > 0 ? "+" : "") + number.toFixed(digits == null ? 4 : digits);
  }
  function unitText(value) {
    var raw = String(value == null ? "" : value).trim();
    return unitLabels[raw] || raw || "未提供";
  }
  function compactTechnicalText(value) {
    return String(value == null ? "" : value).replace(/(?:candidate|artifact|proposal|intervention):[0-9a-z:-]{24,}/gi, function (match) { return shortId(match); });
  }
  function humanizeTechnicalText(value) {
    var text = compactTechnicalText(value);
    Object.keys(targetLabels).concat(Object.keys(parameterLabels)).sort(function (left, right) { return right.length - left.length; }).forEach(function (key) {
      text = text.replace(new RegExp("\\b" + key + "\\b", "g"), targetLabels[key] || parameterLabels[key]);
    });
    return text;
  }
  function formatDate(value) {
    if (!value) { return "—"; }
    var date = new Date(value);
    if (Number.isNaN(date.getTime())) { return String(value); }
    return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(date);
  }
  function formatObservationTime(value) {
    var number = Number(value);
    if (Number.isFinite(number) && number > 500000) {
      var date = new Date(Date.UTC(1899, 11, 30) + number * 60 * 60 * 1000);
      if (!Number.isNaN(date.getTime())) { return date.toISOString().slice(0, 16).replace("T", " "); }
    }
    return formatNumber(value);
  }
  function fieldRoleText(value) {
    return {
      environment: "环境观测", outside_weather: "室外天气", root_zone: "根区观测",
      action: "管理动作", resource: "资源投入", crop: "作物观测", outcome: "结果指标",
      forcing: "外部驱动", state: "状态变量", target: "评测目标", input: "输入变量",
      time: "时间", integer: "整数"
    }[String(value || "").toLowerCase()] || String(value || "类型未提供");
  }
  function formatTime(value) {
    if (!value) { return "—"; }
    var date = new Date(value);
    if (Number.isNaN(date.getTime())) { return String(value); }
    return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(date);
  }
  function statusText(value) { return statusLabels[value] || "未知状态"; }
  function statusClass(value) {
    if (["running", "completed", "released"].indexOf(value) >= 0) { return "pill-green"; }
    if (["paused", "promotion_pending", "starting", "preflight", "created"].indexOf(value) >= 0) { return "pill-amber"; }
    if (["failed", "cancelled", "quarantined"].indexOf(value) >= 0) { return "pill-red"; }
    return "pill-neutral";
  }
  function runHasEvolutionProgress(run, events) {
    var progressArrays = ["trajectory", "rounds", "generation_batches", "generation_analyses", "knowledge_snapshots", "knowledge_assessments", "training_assets", "artifacts", "candidates"];
    if (run && (Number(run.generation || run.current_generation || 0) > 0
      || progressArrays.some(function (field) { return Array.isArray(run[field]) && run[field].length > 0; })
      || Number(run.execution_progress && run.execution_progress.completed_steps || 0) > 0)) {
      return true;
    }
    var source = Array.isArray(events) ? events : run && Array.isArray(run.events) ? run.events : null;
    if (!source || !source.length) { return false; }
    return source.some(function (event) {
      var type = String(event && (event.type || event.event_type || event.kind) || "").toLowerCase();
      return type.indexOf("generation") >= 0 || type.indexOf("knowledge") >= 0 || type.indexOf("proposal") >= 0 || type.indexOf("candidate") >= 0 || type.indexOf("artifact") >= 0 || type.indexOf("evaluation") >= 0 || type.indexOf("promotion") >= 0 || type.indexOf("stage") >= 0;
    });
  }
  function runWaitingForAdvance(run, events) {
    if (!run || String(run.status || "").toLowerCase() !== "running") { return false; }
    var totalGenerations = Number(run.total_generations || run.budget && run.budget.max_generations || 0);
    var maxCandidates = Number(run.max_candidates || run.budget && run.budget.max_candidates || 0);
    var generation = Number(run.generation || run.current_generation || 0);
    var candidates = Number(run.candidates_count != null ? run.candidates_count : run.candidate_count != null ? run.candidate_count : Array.isArray(run.candidates) ? run.candidates.length : 0);
    // A true legacy projection has no execution phase. Its /advance call is
    // synchronous, so any running run below both frozen budgets is a resumable
    // boundary, including a partially written generation after a page reload.
    return !(totalGenerations > 0 && generation >= totalGenerations) && !(maxCandidates > 0 && candidates >= maxCandidates);
  }
  function runNeedsAdvanceAction(run, events) {
    if (!run || String(run.status || "").toLowerCase() !== "running") { return false; }
    if (runHasContinuousAutoProgress(run)) { return false; }
    var progress = run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : {};
    var phase = String(progress.phase || "").toLowerCase();
    if (phase) { return phase === "waiting"; }
    return runWaitingForAdvance(run, events);
  }
  function runHasContinuousAutoProgress(run) {
    var progress = run && run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : {};
    var configuration = run && run.configuration && typeof run.configuration === "object" ? run.configuration : {};
    return Boolean(run && (
      run.auto_progress === true
      || progress.auto_progress === true
      || String(progress.auto_progress_policy || "").toLowerCase() === "continuous_generation_budget@1"
      || configuration.auto_progress === true
    ));
  }
  function runCandidateCount(run) {
    if (!run) { return 0; }
    if (run.candidates_count != null) { return Number(run.candidates_count); }
    if (run.candidate_count != null) { return Number(run.candidate_count); }
    return Array.isArray(run.candidates) ? run.candidates.length : 0;
  }
  function isCancelledEmptyRun(run) {
    return Boolean(run)
      && String(run.status || "").toLowerCase() === "cancelled"
      && runCandidateCount(run) === 0
      && !runHasEvolutionProgress(run, run.events);
  }
  function cancelledEmptyRunCount() {
    return state.runs.filter(isCancelledEmptyRun).length;
  }
  function visibleRuns() {
    return state.showCancelledEmptyRuns
      ? state.runs.slice()
      : state.runs.filter(function (run) { return !isCancelledEmptyRun(run); });
  }
  function runsListPath() {
    return state.showArchivedRuns ? "/runs?view=summary&include_archived=true" : "/runs?view=summary";
  }
  function runIsTerminal(run) {
    return Boolean(run) && ["completed", "cancelled", "failed"].indexOf(String(run.status || "").toLowerCase()) >= 0;
  }
  function expertConsultationRunIsTerminal(run) {
    return Boolean(run) && ["completed", "cancelled", "failed", "released", "quarantined"].indexOf(String(run.status || "").toLowerCase()) >= 0;
  }
  function reconcileVisibleRunSelection() {
    var runs = visibleRuns();
    if (state.activeRun && runs.some(function (run) { return run.id === state.activeRun.id; })) { return false; }
    if (state.autoAdvanceRunId && typeof stopAutoAdvance === "function") {
      stopAutoAdvance(state.autoAdvanceRunId);
    }
    nextEpoch();
    state.datasetRequest += 1;
    state.activeRun = runs[0] || null;
    state.events = [];
    state.showAllEvents = false;
    state.candidateSelectionPinned = false;
    syncCandidateSelection(state.activeRun);
    state.datasetPage = null;
    state.datasetContext = null;
    state.datasetError = null;
    state.datasetLoading = false;
    state.loadState = state.activeRun ? state.usingDemo ? "demo" : "ready" : "empty";
    return true;
  }
  function runOutcomeCode(run) {
    if (!run) { return ""; }
    var explicit = run.outcome || run.termination_reason;
    if (explicit) {
      var normalized = String(explicit).toLowerCase();
      if (["accepted", "completed_with_search_retained_candidate"].indexOf(normalized) >= 0) { return "completed_with_acceptable_candidate"; }
      return normalized;
    }
    if (String(run.status || "").toLowerCase() !== "completed") { return ""; }
    return run.best_candidate_id ? "completed_with_acceptable_candidate" : "budget_exhausted_without_acceptable_candidate";
  }
  function runOutcomeText(run) {
    var outcome = runOutcomeCode(run);
    if (outcome === "budget_exhausted_without_acceptable_candidate") {
      var progress = run && run.execution_progress || {};
      var generations = Number(progress.completed_generations || run && run.total_generations || 0);
      var candidates = Number(run && run.candidates_count || run && Array.isArray(run.candidates) && run.candidates.length || 0);
      var scale = [];
      if (Number.isInteger(generations) && generations > 0) { scale.push(formatNumber(generations) + " 代"); }
      if (Number.isInteger(candidates) && candidates > 0) { scale.push(formatNumber(candidates) + " 个候选"); }
      return "已完成预设进化规模" + (scale.length ? "（" + scale.join("、") + "）" : "") + "，尚无候选通过全部评测门控；正式验证未开展";
    }
    if (outcome === "completed_with_acceptable_candidate") { return "已完成，产生训练反馈搜索保留候选；正式验证未开展"; }
    if (outcome === "completed_without_acceptable_candidate") { return "运行已结束，未产生训练反馈搜索保留候选；正式验证未开展"; }
    return "";
  }
  function publicFailureText(value) {
    if (typeof value === "string") { return value.trim(); }
    if (!value || typeof value !== "object") { return ""; }
    var fields = ["public_error", "message", "reason", "detail", "error"];
    for (var index = 0; index < fields.length; index += 1) {
      var text = value[fields[index]];
      if (typeof text === "string" && text.trim()) { return text.trim(); }
    }
    return "";
  }
  function runFailureMessage(run, events) {
    if (String(run && run.failure_code || "") === "frozen_runtime_binding_drift") {
      return "该运行的冻结算法或模型绑定与当前服务版本不一致。为保证可复现性，系统已停止继续执行；请使用当前配置新建进化运行。";
    }
    var direct = [
      run && run.failure_reason,
      run && run.failure,
      run && run.public_error,
      run && run.error
    ];
    for (var directIndex = 0; directIndex < direct.length; directIndex += 1) {
      var directText = publicFailureText(direct[directIndex]);
      if (directText) { return directText; }
    }
    var termination = String(run && (run.terminal_reason || run.termination_reason) || "").trim();
    var ordinaryOutcomes = [
      "accepted", "completed_with_acceptable_candidate", "completed_with_search_retained_candidate",
      "completed_without_acceptable_candidate", "budget_exhausted_without_acceptable_candidate"
    ];
    if (termination && ordinaryOutcomes.indexOf(termination.toLowerCase()) < 0) { return termination; }
    var source = Array.isArray(events) ? events : [];
    var failedStage = source.find(function (event) {
      var payload = payloadOf(event);
      return String(event && event.type || "").toLowerCase() === "stage.recorded"
        && String(payload.status || payload.state || "").toLowerCase() === "failed"
        && publicFailureText(payload);
    });
    if (failedStage) { return publicFailureText(payloadOf(failedStage)); }
    var failedCandidate = source.find(function (event) {
      return String(event && event.type || "").toLowerCase() === "candidate.failed" && publicFailureText(payloadOf(event));
    });
    if (failedCandidate) { return publicFailureText(payloadOf(failedCandidate)); }
    var failedRun = source.find(function (event) {
      return String(event && event.type || "").toLowerCase() === "run.failed" && publicFailureText(payloadOf(event));
    });
    if (failedRun) { return publicFailureText(payloadOf(failedRun)); }
    return "后台进化运行失败，请查看进化过程中的失败阶段。";
  }
  function createStatusForRun(run, events) {
    if (!run) { return null; }
    var status = String(run.status || "").toLowerCase();
    var phase = String(run.execution_progress && run.execution_progress.phase || "").toLowerCase();
    if (status === "failed") {
      return { runId: run.id, state: "failed", message: "运行创建后失败：" + runFailureMessage(run, events) };
    }
    if (status === "completed") {
      return { runId: run.id, state: "completed", message: "进化运行已完成，可以继续创建新运行。" };
    }
    if (status === "cancelled") {
      return { runId: run.id, state: "completed", message: "进化运行已取消，可以继续创建新运行。" };
    }
    if (["queued", "waiting"].indexOf(phase) >= 0 || status === "created") {
      return { runId: run.id, state: "queued", message: "已排队，后台将自动执行全部轮次。" };
    }
    return { runId: run.id, state: "background", message: "已创建，后台正在自动执行全部轮次。" };
  }
  function runHasHardTokenPause(run) {
    return Boolean(run && run.status === "paused" && run.pause_code === "model_token_budget_exhausted");
  }
  function runPauseReason(run) {
    return run && typeof run.pause_reason === "string" && run.pause_reason.trim()
      ? run.pause_reason.trim()
      : "";
  }
  function runUsesSampleAgentTokenBudget(run) {
    var configuration = run && run.configuration && typeof run.configuration === "object" ? run.configuration : {};
    return Boolean(run && (run.token_budget_scope || configuration.token_budget_scope) === "sample_agent_gateway_calls_only@1");
  }
  function tokenBudgetScopeText(run) {
    if (runUsesSampleAgentTokenBudget(run)) {
      return "仅计 planner / repair / critic；不含 research / proposal / judge";
    }
    return "历史运行未声明完整的 Token 计量范围";
  }
  function displayRunStatusText(run, events) {
    if (runHasHardTokenPause(run)) { return "逐样本智能体 Token 预算已暂停"; }
    return runNeedsAdvanceAction(run, events) ? "等待推进" : runOutcomeText(run) || statusText(run && run.status);
  }
  function displayRunStatusClass(run, events) {
    if (runNeedsAdvanceAction(run, events)) { return "pill-amber"; }
    if (runOutcomeCode(run) === "budget_exhausted_without_acceptable_candidate") { return "pill-amber"; }
    return statusClass(run && run.status);
  }
  function candidateStatusText(value) { return candidateStatusLabels[value] || "未知状态"; }
  function candidateStatusClass(value) {
    if (["accepted", "promoted", "retained", "released"].indexOf(value) >= 0) { return "pill-green"; }
    if (["evaluating", "evaluated", "pending", "spawned", "paused"].indexOf(value) >= 0) { return "pill-amber"; }
    if (value === "duplicate" || value === "not_recorded") { return "pill-neutral"; }
    if (value === "failed" || value === "rejected" || value === "aborted") { return "pill-red"; }
    return "pill-neutral";
  }
  function partitionText(value) {
    return { training_fit: "训练拟合分区", training_feedback: "训练反馈分区", train: "训练集", development: "开发集", validation: "验证集", test: "测试集", hidden: "隐藏评测集", final: "最终评测集" }[value] || "受限数据分区";
  }
  function gateText(value) {
    var text = String(value == null ? "" : value);
    return {
      pass: "通过", passed: "通过", true: "通过", pending: "等待", waiting: "等待",
      not_started: "未开始", unavailable: "未开放", restricted: "外部受限",
      denied: "拒绝", fail: "未通过", failed: "未通过", false: "未通过",
      approved: "已保留", accepted: "已保留", promoted: "已保留", rejected: "未保留",
      evaluating: "训练反馈检查中", evaluated: "等待搜索保留决策", spawned: "等待训练反馈", duplicate: "重复版本已跳过"
    }[text.toLowerCase()] || text || "未提供";
  }
  function environmentText(value) {
    return { production: "正式环境", dsh_native: "DSH 原生智能体运行时", authenticated: "兼容模型网关已验证", configured: "兼容模型网关已配置", local: "本地模型环境", development: "开发测试环境" }[value] || "运行环境未标明";
  }
  function metricLabel(key) { return metricLabels[key] || "指标：" + key; }
  function metricValue(key, value) {
    if (isBlank(value)) { return "—"; }
    if (key === "passed") { return value ? "通过" : "未通过"; }
    if (key === "scientific_pass") { return value ? "通过" : "未通过"; }
    if (key === "judge_accepted") { return value ? "建议保留" : "不建议保留"; }
    if (key === "causal_interpretation") { return value ? "支持" : "不支持"; }
    if (["per_target_no_regression", "raw_units_comparable_across_targets"].indexOf(key) >= 0) { return value ? "是" : "否"; }
    if (key === "partition") { return partitionText(value); }
    if (key === "judge_model_id") { return modelReferenceLabel(value); }
    if (key === "prediction_model_id") { return predictionModelReferenceLabel(value); }
    if (key === "objective_target_weights" && value && typeof value === "object" && !Array.isArray(value)) {
      return Object.keys(value).map(function (target) {
        return (targetLabels[target] || target) + "=" + formatNumber(value[target]);
      }).join("；");
    }
    if (value && typeof value === "object") {
      return Object.keys(value).slice(0, 8).map(function (field) {
        return humanizeTechnicalText(field) + "=" + formatNumber(value[field]);
      }).join("；") || "—";
    }
    if (key === "evaluation_scope") {
      return { "visible/training_feedback/historical_replay": "可见训练反馈分区／历史回放", "visible/validation/demo": "可见验证分区／工程演示" }[value] || String(value);
    }
    if (key === "selection_scope") { return selectionScopeText(value); }
    if (key === "formal_validation_status") { return formalValidationText(value); }
    if (/digest/.test(key)) { return shortId(value); }
    if (key === "non_negative_state") { return Number(value) >= 1 ? "通过" : "未通过"; }
    return formatNumber(value);
  }
  function fieldLabel(key) { return fieldLabels[key] || "字段：" + key; }
  function selectionScopeText(value) {
    return { iterative_training_feedback_only: "仅用于训练反馈搜索保留" }[String(value || "").toLowerCase()] || String(value || "未标明");
  }
  function formalValidationText(value) {
    return {
      not_run: "未开展", pending: "等待外部正式验证", running: "外部正式验证中",
      passed: "外部正式验证通过", failed: "外部正式验证未通过", restricted: "由外部治理服务控制"
    }[String(value || "").toLowerCase()] || String(value || "未提供");
  }

  function itemId(item) {
    if (typeof item === "string") { return item; }
    return String(item && (item.id || item.value || item.model_id || item.strategy_model_id || item.review_model_id || item.reviewer_model_id || item.dataset_id || item.strategy_id || item.evaluator_id || item.domain_pack_id) || "");
  }
  function modelConnectionStateText(item) {
    if (!item || typeof item !== "object") { return ""; }
    if (item.local_model === true || String(item.authentication_state || "").toLowerCase() === "local") { return "本地内置"; }
    if (item.directory_available === false || item.configured === false) { return "当前后端不可执行"; }
    if (item.credential_configured !== true) { return "未配置凭据"; }
    var connectionState = String(item.connection && item.connection.state || "").toLowerCase();
    if (["error", "unavailable", "unreachable"].indexOf(connectionState) >= 0) { return "已配置，最近调用失败（可重试）"; }
    if (connectionState === "available") { return "已配置，最近调用成功"; }
    if (item.execution_available === false) { return "当前后端不可执行"; }
    if (item.credential_configured === true) { return "凭据已配置，运行时连接"; }
    return "";
  }
  function modelCredentialReady(item) {
    if (!item || typeof item !== "object") { return false; }
    if (item.local_model === true || String(item.authentication_state || "").toLowerCase() === "local") { return item.available !== false; }
    return item.configured === true && item.directory_available === true && item.execution_available === true && item.credential_configured === true;
  }
  function modelConnectionErrorText(item) {
    var value = String(item && item.connection && item.connection.last_error || "").trim();
    if (!value) { return ""; }
    var normalized = value.toLowerCase();
    var reasonCode = String(item && item.unavailable_reason && item.unavailable_reason.code || "").toLowerCase();
    var reasonLabels = {
      insecure_http_blocked: "模型网关使用非本机 HTTP，已被后端安全策略阻止。",
      host_route_not_available_to_sidecar: "该模型仅在 DSH 主目录中登记，当前后端没有对应的调用路由或凭据。",
      missing_gateway_url: "DSH 主目录未提供该模型的网关地址。",
      invalid_gateway_url: "DSH 主目录中的模型网关地址无效。",
      unsupported_provider_api: "该模型使用的提供方 API 不是当前后端支持的 OpenAI 兼容接口。"
    };
    if (reasonLabels[reasonCode]) { return reasonLabels[reasonCode]; }
    if (/^http \d{3}$/i.test(value)) { return "模型网关返回 " + value.toUpperCase() + "。"; }
    if (/timeout/.test(normalized)) { return "连接模型网关超时。"; }
    if (/json|unicode|decode/.test(normalized)) { return "模型响应不是有效的 JSON。"; }
    if (/response|choice|message|contract|unknown field|must /.test(normalized)) { return "模型响应不符合约定格式。"; }
    if (/url|connection|network|refused|unreachable/.test(normalized)) { return "无法连接模型网关。"; }
    return "最近一次模型调用失败，请检查服务端模型配置与网关日志后重试。";
  }
  function itemBaseLabel(item) {
    if (typeof item === "string") { return item; }
    return String(item && (item.display_name_zh || item.display_name || item.label || item.name || item.title || itemId(item)) || "未选择");
  }
  function itemLabel(item) {
    var label = itemBaseLabel(item);
    if (item && (Object.prototype.hasOwnProperty.call(item, "credential_configured") || Object.prototype.hasOwnProperty.call(item, "connection_available"))) { label += "（" + modelConnectionStateText(item) + "）"; }
    return label;
  }
  function catalogItem(collection, id) {
    var values = state.catalog[collection] || [];
    return values.find(function (item) { return itemId(item) === String(id || ""); }) || null;
  }
  function catalogReferenceLabel(collection, id, fallback) {
    var item = catalogItem(collection, id);
    return item ? itemBaseLabel(item) : fallback || String(id || "未提供");
  }
  function selectedModelCatalogItem(selector) {
    var id = $(selector).value;
    // Prefer the canonical DSH entry because it carries the authoritative
    // credential and connection state.  ``models`` remains the compatibility
    // fallback for older catalog responses and local demo entries.
    return catalogItem("dsh_models", id) || catalogItem("models", id);
  }
  function modelReferenceLabel(id) {
    var value = String(id || "");
    return catalogReferenceLabel("models", value, knownModelLabels[value] || value || "未提供");
  }
  function predictionModelReferenceLabel(id) {
    var value = String(id || "");
    return catalogReferenceLabel("prediction_models", value, knownModelLabels[value] || value || "未提供");
  }
  function itemDescription(item) {
    return typeof item === "object" && item ? String(item.description || item.summary || item.detail || "") : "";
  }
  function normalizeList(value) { return Array.isArray(value) ? value.filter(function (item) { return itemId(item); }) : []; }
  function mergeModelDirectory(lists) {
    var merged = [];
    var positions = Object.create(null);
    (Array.isArray(lists) ? lists : []).forEach(function (list) {
      normalizeList(list).forEach(function (item) {
        var id = itemId(item);
        if (!id) { return; }
        if (positions[id] == null) {
          positions[id] = merged.length;
          merged.push(Object.assign({}, item));
          return;
        }
        var index = positions[id];
        var current = merged[index];
        var roles = (Array.isArray(current.roles) ? current.roles : []).concat(Array.isArray(item.roles) ? item.roles : []).map(function (role) {
          var value = String(role || "").toLowerCase();
          return { strategy: "propose", policy: "propose", planner: "propose", proposer: "propose", review: "judge", reviewer: "judge", critic: "judge" }[value] || value;
        }).filter(function (role, roleIndex, values) {
          return role && values.indexOf(role) === roleIndex;
        });
        merged[index] = Object.assign({}, current, item, roles.length ? { roles: roles } : {});
      });
    });
    return merged;
  }
  function modelSupportsRole(item, role) {
    if (!item || !role) { return false; }
    var aliases = { strategy: "propose", policy: "propose", planner: "propose", proposer: "propose", review: "judge", reviewer: "judge", critic: "judge" };
    var roles = Array.isArray(item.roles) ? item.roles.map(function (value) {
      var normalized = String(value || "").toLowerCase();
      return aliases[normalized] || normalized;
    }) : [];
    // Older DSH catalogues omitted roles; leave those entries selectable and let
    // the server perform the authoritative role check.
    var required = String(role).toLowerCase();
    return !roles.length || roles.indexOf(aliases[required] || required) >= 0;
  }
  function finiteCatalogCount(value) {
    var number = Number(value);
    return Number.isFinite(number) && number >= 0 ? number : null;
  }
  function catalogCountFromFields(source, fields) {
    if (!source || typeof source !== "object") { return null; }
    for (var index = 0; index < fields.length; index += 1) {
      var field = fields[index];
      if (!Object.prototype.hasOwnProperty.call(source, field)) { continue; }
      var count = finiteCatalogCount(source[field]);
      if (count != null) { return count; }
    }
    return null;
  }
  function dshModelTotalCount(verified) {
    var catalog = state.catalog || {};
    var dsh = catalog.dsh || {};
    var fields = verified
      ? ["authenticated_dsh_model_count", "authenticated_model_count"]
      : ["dsh_model_count", "configured_model_count"];
    var direct = catalogCountFromFields(dsh, fields);
    if (!verified && state.hostContextReceived && catalog.dsh_models_explicit === true && typeof sharedDshModelItems === "function") {
      var backendCount = direct == null ? (catalog.dsh_models || []).length : direct;
      var hostOnlyCount = sharedDshModelItems().filter(function (item) { return item && item.model_source === "dsh_host_only"; }).length;
      return backendCount + hostOnlyCount;
    }
    if (direct != null) { return direct; }
    var list = verified ? catalog.authenticated_models : catalog.dsh_models;
    return Array.isArray(list) ? list.length : 0;
  }
  function dshModelRoleCount(role, verified) {
    var catalog = state.catalog || {};
    var dsh = catalog.dsh || {};
    var normalizedRole = String(role || "").toLowerCase();
    normalizedRole = normalizedRole === "review" || normalizedRole === "judge" ? "review" : "strategy";
    var fields = verified
      ? ["authenticated_" + normalizedRole + "_model_count", "authenticated_dsh_" + normalizedRole + "_model_count"]
      : ["dsh_" + normalizedRole + "_model_count", "configured_" + normalizedRole + "_model_count"];
    var direct = catalogCountFromFields(dsh, fields);
    if (!verified && state.hostContextReceived && catalog.dsh_models_explicit === true && typeof sharedDshModelItems === "function") {
      var backendRoleCount = direct;
      if (backendRoleCount == null) {
        backendRoleCount = (catalog.dsh_models || []).filter(function (item) { return modelSupportsRole(item, normalizedRole); }).length;
      }
      var hostOnlyRoleCount = sharedDshModelItems().filter(function (item) {
        return item && item.model_source === "dsh_host_only" && modelSupportsRole(item, normalizedRole);
      }).length;
      return backendRoleCount + hostOnlyRoleCount;
    }
    if (direct != null) { return direct; }
    var list = verified ? catalog.authenticated_models : catalog.dsh_models;
    if (Array.isArray(list) && list.length) {
      return list.filter(function (item) { return modelSupportsRole(item, normalizedRole); }).length;
    }
    return dshModelTotalCount(verified);
  }
  function normalizeCatalog(data) {
    var value = data && data.catalog || data || {};
    var hasExplicitDshDirectory = Array.isArray(value.dsh_models);
    var compatibilityModels = mergeModelDirectory([value.models, value.strategy_models, value.review_models, value.policy_models, value.judge_models, value.reviewer_models]);
    var explicitAuthenticated = mergeModelDirectory([value.authenticated_models]);
    var dshModels = hasExplicitDshDirectory ? mergeModelDirectory([explicitAuthenticated, value.dsh_models]) : compatibilityModels.filter(function (item) {
      return item.local_model !== true && String(item.authentication_state || "").toLowerCase() !== "local";
    });
    // Canonical DSH entries are merged last so compatibility lists cannot
    // overwrite current backend verification or availability state.
    var models = mergeModelDirectory([compatibilityModels, dshModels]);
    var authenticatedModels = dshModels.filter(function (item) {
      return item.authentication_verified === true;
    });
    var availableModels = dshModels.filter(function (item) { return item.available === true; });
    var sharedModels = hasExplicitDshDirectory ? dshModels : (dshModels.length ? dshModels : models);
    return {
      domain_packs: normalizeList(value.domain_packs), datasets: normalizeList(value.datasets),
      unavailable_datasets: normalizeList(value.unavailable_datasets),
      prediction_models: normalizeList(value.prediction_models),
      strategies: normalizeList(value.strategies), evaluators: normalizeList(value.evaluators),
      models: models,
      dsh_models: dshModels,
      dsh_models_explicit: hasExplicitDshDirectory,
      authenticated_models: authenticatedModels,
      available_models: availableModels,
      policy_models: sharedModels.filter(function (item) { return modelSupportsRole(item, "propose"); }),
      judge_models: sharedModels.filter(function (item) { return modelSupportsRole(item, "judge"); }),
      dsh: value.dsh && typeof value.dsh === "object" ? value.dsh : {}
    };
  }

  function normalizeCandidate(candidate, index) {
    var item = candidate || {};
    var status = item.status || item.state || "pending";
    if (String(status).toLowerCase() === "error") { status = "failed"; }
    return Object.assign({}, item, {
      id: item.id || item.candidate_id || "candidate-" + index,
      parent_id: item.parent_id || item.parent_candidate_id || null,
      generation: Number(item.generation || 0), slot_index: Number(item.slot_index || 0), generation_rank: item.generation_rank == null ? null : Number(item.generation_rank), status: status,
      failure_reason: item.failure_reason || item.error || item.failure || null,
      failed_stage: item.failed_stage || item.failure_stage || null,
      metrics: item.metrics && typeof item.metrics === "object" ? item.metrics : {},
      changes: item.changes && typeof item.changes === "object" ? item.changes : {}
    });
  }
  function normalizeExpertConsultation(value, index) {
    var item = value || {};
    var nestedAnswer = item.answer && typeof item.answer === "object" ? item.answer : {};
    var answer = typeof item.answer === "string" ? item.answer : nestedAnswer.answer || null;
    var rawStatus = String(item.status || (answer ? "answered" : "pending")).toLowerCase();
    var answered = rawStatus === "answered" || rawStatus === "answered_pending" || rawStatus === "applied" || rawStatus === "audit_only" || !isBlank(answer);
    var id = item.consultation_id || item.id || "consultation-" + index;
    return Object.assign({}, item, {
      id: item.id || id,
      consultation_id: id,
      status: answered ? "answered" : "pending",
      question: item.question || "",
      context: item.context || "",
      options: Array.isArray(item.options) ? item.options : [],
      non_blocking: item.non_blocking !== false,
      answer: answer,
      selected_option: Object.prototype.hasOwnProperty.call(item, "selected_option") ? item.selected_option : nestedAnswer.selected_option,
      answered_by: item.answered_by || nestedAnswer.answered_by || null,
      answered_at: item.answered_at || nestedAnswer.answered_at || nestedAnswer.created_at || null,
      effective_generation: Object.prototype.hasOwnProperty.call(item, "effective_generation") ? item.effective_generation : nestedAnswer.effective_generation,
      applied_generation: Object.prototype.hasOwnProperty.call(item, "applied_generation") ? item.applied_generation : nestedAnswer.applied_generation
    });
  }
  function normalizeRun(input) {
    var item = input && (input.projection || input.run_projection) || input || {};
    var configuration = item.configuration || {};
    var task = item.task || {};
    var candidates = Array.isArray(item.candidates) ? item.candidates.map(normalizeCandidate) : [];
    var observedCandidate = candidates.filter(function (candidate) {
      return !isBlank(candidate.score) && Number.isFinite(Number(candidate.score));
    }).sort(function (left, right) {
      return Number(right.score) - Number(left.score);
    })[0] || null;
    var bestObservedCandidateId = item.best_observed_candidate_id || observedCandidate && observedCandidate.id || null;
    var bestObservedCandidate = candidates.find(function (candidate) { return candidate.id === bestObservedCandidateId; }) || observedCandidate;
    var bestObservedScore = !isBlank(item.best_observed_score) && Number.isFinite(Number(item.best_observed_score))
      ? Number(item.best_observed_score)
      : bestObservedCandidate && !isBlank(bestObservedCandidate.score) && Number.isFinite(Number(bestObservedCandidate.score))
        ? Number(bestObservedCandidate.score)
        : null;
    var status = item.status || "idle";
    var bestCandidateId = item.best_candidate_id || null;
    var outcome = item.outcome || (String(status).toLowerCase() === "completed"
      ? bestCandidateId ? "completed_with_acceptable_candidate" : "budget_exhausted_without_acceptable_candidate"
      : null);
    var trajectory = Array.isArray(item.trajectory) ? item.trajectory : [];
    if (!trajectory.length) {
      trajectory = candidates.slice().sort(function (a, b) { return a.generation - b.generation; }).filter(function (candidate) {
        return Number.isFinite(Number(candidate.score));
      }).map(function (candidate, index) {
        var score = Number(candidate.score);
        // Legacy projections without a trajectory can still show candidate
        // scatter points. Do not synthesize a cross-window incumbent line.
        return { generation: candidate.generation || index + 1, candidate_id: candidate.id, score: score };
      });
    }
    return Object.assign({}, item, {
      id: item.id || item.run_id || "未知运行",
      status: status,
      archived: item.archived === true || Boolean(item.archived_at),
      archived_at: item.archived_at || null,
      outcome: outcome,
      termination_reason: item.termination_reason || outcome,
      best_candidate_id: bestCandidateId,
      best_observed_candidate_id: bestObservedCandidateId,
      best_observed_score: bestObservedScore,
      generation: Number(item.generation || item.current_generation || 0),
      total_generations: Number(item.total_generations || item.max_generations || item.budget && item.budget.max_generations || 0),
      candidates_count: Number(item.candidates_count != null ? item.candidates_count : item.candidate_count != null ? item.candidate_count : candidates.length),
      max_candidates: Number(item.max_candidates || item.budget && item.budget.max_candidates || 0),
      candidates_per_generation: Number(item.candidates_per_generation || configuration.candidates_per_generation || item.budget && item.budget.candidates_per_generation || 1),
      samples_per_update: Number(item.samples_per_update || configuration.samples_per_update || 0),
      sample_agent_batch_size: Number(item.sample_agent_batch_size || configuration.sample_agent_batch_size || 0),
      sample_concurrency: Number(item.sample_concurrency || configuration.sample_concurrency || 0),
      projection_revision: Number(item.projection_revision || item.revision || 0),
      configuration: Object.assign({
        domain_pack_id: task.domain_pack_id || item.domain_pack_id || item.domain,
        research_domain_id: task.research_domain_id || item.research_domain_id || item.domain_pack_id || item.domain,
        dataset_id: task.dataset_id || item.dataset_id,
        episode_id: item.dataset && item.dataset.episode_id || item.episode_id,
        strategy_id: item.strategy_id, prediction_model_id: item.prediction_model_id, evaluator_id: item.evaluator_id,
        policy_model_id: item.policy_model_id, judge_model_id: item.judge_model_id,
        strategy_model_id: item.strategy_model_id || item.policy_model_id,
        review_model_id: item.review_model_id || item.reviewer_model_id || item.judge_model_id,
        autonomous_mode: item.autonomous_mode === true || item.autonomous_mode === "true",
        slot: task.slot || item.slot
      }, configuration),
      dataset: item.dataset && typeof item.dataset === "object" ? item.dataset : {},
      trajectory: trajectory,
      artifacts: Array.isArray(item.artifacts) ? item.artifacts : [],
      interventions: Array.isArray(item.interventions) ? item.interventions : [],
      expert_consultations: Array.isArray(item.expert_consultations) ? item.expert_consultations.map(normalizeExpertConsultation) : [],
      training_assets: Array.isArray(item.training_assets) ? item.training_assets : [],
      rounds: Array.isArray(item.rounds) ? item.rounds : [],
      generation_analyses: Array.isArray(item.generation_analyses) ? item.generation_analyses : [],
      candidates: candidates,
      gate: item.gate && typeof item.gate === "object" ? item.gate : {},
      metrics: item.metrics && typeof item.metrics === "object" ? item.metrics : {}
    });
  }
  function listFrom(data, key) {
    if (Array.isArray(data)) { return data; }
    if (data && Array.isArray(data[key])) { return data[key]; }
    if (data && data.data && Array.isArray(data.data[key])) { return data.data[key]; }
    return [];
  }
  function payloadOf(event) { return event && (event.payload || event.data || event.detail) || {}; }
  function normalizeEvents(data) {
    return listFrom(data, "events").map(function (item, index) {
      return {
        id: item.id || item.event_id || "event-" + index,
        type: item.type || item.event_type || "system.updated",
        occurred_at: item.occurred_at || item.created_at || item.timestamp,
        payload: payloadOf(item)
      };
    }).sort(function (a, b) { return new Date(b.occurred_at) - new Date(a.occurred_at); });
  }

  function localizeError(message, errorCode) {
    var text = String(message || "");
    if (String(errorCode || "") === "frozen_runtime_binding_drift") {
      return "该运行的冻结算法或模型绑定与当前服务版本不一致。为保证可复现性，请使用当前配置新建进化运行。";
    }
    if (!text) { return "未知错误"; }
    if (/[\u3400-\u9fff]/.test(text)) { return text; }
    if (/failed to fetch|networkerror|load failed/i.test(text)) { return "无法连接服务网关。"; }
    if (/unauthorized|401|token/i.test(text)) { return "宿主授权已失效，请重新打开插件。"; }
    if (/forbidden|403|capability/i.test(text)) { return "当前 DSH 能力权限不允许执行该操作。"; }
    if (/validation partition/i.test(text)) { return "迭代搜索只能使用训练反馈分区，不能读取正式验证分区。"; }
    if (/training_fit/i.test(text)) { return "当前目录未开放训练拟合分区。"; }
    if (/run must be paused|expected paused/i.test(text)) { return "提交人工意见前必须先暂停进化运行。"; }
    if (/run must be running/i.test(text)) { return "只有运行中的任务可以执行下一轮。"; }
    if (/unknown run/i.test(text)) { return "未找到指定的进化运行。"; }
    if (/idempotency/i.test(text)) { return "请求标识与已有操作冲突。"; }
    return "服务请求失败，请检查配置和当前运行状态。";
  }
  function errorMessage(error) {
    if (error && error.name === "AbortError") { return "请求超时。"; }
    return localizeError(error && error.message || error, error && error.errorCode);
  }
  function showToast(message) {
    var node = $("#toast");
    node.textContent = message;
    node.hidden = false;
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(function () { node.hidden = true; }, 3800);
  }
  function setConnection(kind, text) {
    state.connection = kind;
    var node = $("#connection-status");
    node.className = "connection-status " + (kind === "online" ? "is-online" : kind === "demo" ? "is-demo" : kind === "offline" ? "is-offline" : "is-muted");
    $("#connection-label").textContent = text;
  }
  function commandKey(kind, signature) {
    var current = state.commandKeys[kind];
    if (!current || current.signature !== signature) {
      current = { signature: signature, key: "plugin-" + kind + "-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8) };
      state.commandKeys[kind] = current;
    }
    return current.key;
  }
  function clearCommandKey(kind) { delete state.commandKeys[kind]; }

  function request(path, options) {
    return EcologyDSHHost.request(path, options);
  }
