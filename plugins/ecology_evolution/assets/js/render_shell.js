"use strict";

  function renderSystemBanner() {
    var banner = $("#system-banner");
    var title = $("#system-title");
    var detail = $("#system-detail");
    var retry = $("#retry-button");
    banner.className = "system-banner";
    retry.hidden = true;
    if (state.autoAdvanceError) {
      banner.hidden = false; banner.classList.add("is-error"); title.textContent = "自动推进已暂停";
      detail.textContent = state.autoAdvanceError; retry.hidden = false; return;
    }
    if (state.commandError) {
      banner.hidden = false; banner.classList.add("is-error"); title.textContent = "操作未完成"; detail.textContent = state.commandError; return;
    }
    if (state.pendingAction === "create" || state.createStatus && state.createStatus.state === "submitting") {
      banner.hidden = false; banner.classList.add("is-loading"); title.textContent = "提交已接收";
      detail.textContent = "正在创建持久化运行并连接实时进度；页面无需等待提交按钮返回。"; return;
    }
    if (state.loadState === "loading") {
      banner.hidden = false; banner.classList.add("is-loading"); title.textContent = "正在连接服务网关"; detail.textContent = "正在读取可用配置目录和进化运行。"; return;
    }
    if (state.loadState === "error" || state.loadState === "stale") {
      banner.hidden = false; banner.classList.add("is-error"); title.textContent = state.loadState === "stale" ? "连接中断，正在显示上次状态" : "服务网关不可用"; detail.textContent = state.lastError || "请检查本地服务或 DSH 宿主上下文。"; retry.hidden = false; return;
    }
    if (state.usingDemo) {
      banner.hidden = false; banner.classList.add("is-demo"); title.textContent = "显式演示模式"; detail.textContent = "当前数据只存在于浏览器内，不会写入事件账本或治理服务。"; return;
    }
    if (state.catalog.dsh.harness_execution === "dsh_native_agent_runtime") {
      banner.hidden = false; banner.classList.add("is-demo"); title.textContent = "DSH 原生智能体运行时"; detail.textContent = "DSH 已登记 " + formatNumber(dshModelTotalCount(false)) + " 个模型。Agent Session、上下文压缩、多智能体 Workflow 和模型路由由 DSH 执行；Python sidecar 仅负责科学状态、幂等结果和治理边界。"; return;
    }
    if (state.catalog.dsh.environment && state.catalog.dsh.environment !== "production") {
      banner.hidden = false; banner.classList.add("is-demo"); title.textContent = environmentText(state.catalog.dsh.environment); detail.textContent = "DSH 已登记 " + formatNumber(dshModelTotalCount(false)) + " 个模型。当前使用历史兼容网关执行协议。"; return;
    }
    banner.hidden = true;
  }

  function renderWorkspace() {
    $$('[data-workspace]').forEach(function (button) {
      var active = button.dataset.workspace === state.workspace;
      button.classList.toggle("is-active", active); button.setAttribute("aria-selected", String(active)); button.tabIndex = active ? 0 : -1;
    });
    $$('[data-panel]').forEach(function (panel) { var active = panel.dataset.panel === state.workspace; panel.classList.toggle("is-visible", active); panel.hidden = !active; });
  }

  function renderContext() {
    var select = $("#run-select");
    var runs = visibleRuns();
    var cancelledCount = cancelledEmptyRunCount();
    select.innerHTML = runs.length ? runs.map(function (run) {
      var runEvents = state.activeRun && state.activeRun.id === run.id ? state.events : run.events;
      var archivePrefix = run.archived ? "已归档 · " : "";
      var isCurrent = state.activeRun && state.activeRun.id === run.id;
      var statusLabel = isCurrent && state.autoAdvanceRunId === run.id ? "自动推进中" : isCurrent && state.autoAdvanceBlockedRunId === run.id && state.autoAdvanceError ? "自动推进已暂停" : displayRunStatusText(run, runEvents);
      return "<option value=\"" + escapeHTML(run.id) + "\"" + (isCurrent ? " selected" : "") + ">" + escapeHTML(shortId(run.id)) + " · " + archivePrefix + escapeHTML(statusLabel) + "</option>";
    }).join("") : "<option value=\"\">暂无进化运行</option>";
    select.disabled = !runs.length || state.busy || state.refreshing;
    $("#show-cancelled-empty-runs").checked = state.showCancelledEmptyRuns;
    $("#show-cancelled-empty-runs").disabled = state.busy || state.refreshing;
    $("#cancelled-empty-count").textContent = String(cancelledCount);
    $("#cancelled-empty-filter").hidden = cancelledCount === 0;
    $("#show-archived-runs").checked = state.showArchivedRuns;
    $("#show-archived-runs").disabled = state.busy || state.refreshing;
    $("#archived-count").textContent = String(state.archivedRunCount);
    $("#archived-filter").hidden = state.archivedRunCount === 0 && !state.showArchivedRuns;
    var run = state.activeRun;
    var autoAdvanceActive = Boolean(run && state.autoAdvanceRunId === run.id);
    var autoAdvanceBlocked = Boolean(run && state.autoAdvanceBlockedRunId === run.id && state.autoAdvanceError);
    var contextStatus = autoAdvanceActive ? "自动推进中" : autoAdvanceBlocked ? "自动推进已暂停" : run ? displayRunStatusText(run, state.events) : "未创建";
    $("#run-status").textContent = run ? (run.archived ? "已归档 · " : "") + contextStatus : "未创建";
    $("#generation-label").textContent = run ? run.generation + " / " + (run.total_generations || "—") : "0 / 0";
    $("#candidate-count-label").textContent = run ? run.candidates_count + " / " + (run.max_candidates || "—") : "0 / 0";
    var online = state.usingDemo || state.connection === "online";
    var canControl = hasCapability("run.control");
    var hardTokenPause = runHasHardTokenPause(run);
    var retryCircuitPaused = runHasRetryCircuitPause(run);
    var canPauseOrResume = run && (run.status === "running" || run.status === "paused" && !hardTokenPause);
    var canCancel = run && ["created", "running", "paused"].indexOf(run.status) >= 0;
    var pausedAdvance = run && run.status === "paused";
    var canAdvanceStatus = run && (run.status === "running" || pausedAdvance && canControl && !hardTokenPause);
    var canAdvanceCapability = retryCircuitPaused ? canControl : hasCapability("evolution.run.advance");
    var waitingForAdvance = runNeedsAdvanceAction(run, state.events);
    var executionPhase = String(run && run.execution_progress && run.execution_progress.phase || "").toLowerCase();
    var recoveringCurrentRound = run && run.status === "running" && executionPhase && executionPhase !== "waiting" && executionPhase !== "completed";
    $("#advance-button").disabled = state.busy || autoAdvanceActive || !online || !canAdvanceCapability || !canAdvanceStatus;
    $("#advance-button").title = hardTokenPause ? "冻结的逐样本智能体 Token 硬预算已耗尽；请创建更高预算的新运行。" : retryCircuitPaused ? retryCircuitDetailText(run) : autoAdvanceActive ? "运行已进入自动连续推进；暂停后可人工干预。" : pausedAdvance ? "恢复运行后自动执行下一轮。" : waitingForAdvance ? "从已持久化的轮次进度继续执行。" : recoveringCurrentRound ? "继续当前未完成轮次。" : "";
    $("#pause-button").disabled = state.busy || !online || !canControl || !canPauseOrResume;
    $("#pause-button").textContent = state.pendingAction === "pause" ? "正在暂停" : state.pendingAction === "resume" ? "正在恢复" : hardTokenPause ? "预算已用尽" : retryCircuitPaused ? "恢复并重试检查点" : run && run.status === "paused" ? "恢复运行" : "暂停运行";
    $("#pause-button").title = hardTokenPause ? "冻结的逐样本智能体 Token 硬预算已耗尽，直接恢复不会产生新进展。" : retryCircuitPaused ? retryCircuitDetailText(run) : "";
    $("#advance-button").textContent = autoAdvanceActive ? "自动推进中" : state.pendingAction === "resume" ? "正在恢复" : state.pendingAction === "advance" || state.pendingAction === "auto-advance" ? "正在执行" : retryCircuitPaused ? "检查后重试当前检查点" : pausedAdvance ? "恢复并执行下一轮" : waitingForAdvance && Number(run && run.generation || 0) === 0 ? "执行第一轮" : recoveringCurrentRound ? "继续当前轮次" : "执行下一轮";
    $("#cancel-button").disabled = state.busy || !online || !canControl || !canCancel;
    $("#cancel-button").textContent = state.pendingAction === "cancel" ? "正在取消" : "取消运行";
    var terminal = runIsTerminal(run);
    $("#archive-button").disabled = state.busy || !online || !hasCapability("run.archive") || !terminal;
    $("#archive-button").textContent = state.pendingAction === "archive" ? "正在归档" : state.pendingAction === "restore" ? "正在恢复" : run && run.archived ? "恢复运行" : "归档运行";
    $("#archive-button").title = run && !terminal ? "请先完成或取消运行。" : run && run.archived ? "恢复到默认运行列表。" : "从默认列表隐藏，但保留完整历史证据。";
    $("#delete-button").hidden = !(run && run.archived);
    $("#delete-button").disabled = state.busy || !online || !hasCapability("run.delete") || !terminal || !(run && run.archived);
    $("#delete-button").textContent = state.pendingAction === "delete" ? "正在删除" : "永久删除";
    $("#delete-button").title = "永久删除该运行的事件和命令记录，不可恢复。";
  }

  function renderReadiness() {
    var checks = readiness();
    var allReady = checks.every(function (item) { return item.ready; });
    var unmetChecks = checks.filter(function (item) { return !item.ready; });
    $("#readiness-list").innerHTML = checks.map(function (item) { return "<li><span class=\"check-mark " + (item.ready ? "" : "pending") + "\">" + (item.ready ? "✓" : "·") + "</span><span>" + escapeHTML(item.label) + "</span></li>"; }).join("");
    var pill = $("#readiness-pill");
    pill.className = "pill " + (allReady ? "pill-green" : "pill-amber");
    pill.textContent = allReady ? "可以创建运行" : state.loadState === "loading" ? "正在读取目录" : "配置尚未就绪";
    // Keep the primary action discoverable. Submission still fails closed in
    // the form handler, which can now explain the exact unmet condition and
    // move the user to the relevant controls instead of exposing a dead button.
    $("#start-button").disabled = state.busy;
    var createStatus = state.createStatus;
    $("#start-button").textContent = state.pendingAction === "create"
      ? "正在创建并提交"
      : createRunButtonLabel(createStatus, Boolean(state.activeRun));
    $("#create-hint").textContent = createRunHint(createStatus, allReady, unmetChecks);
    var selectedDataset = selectedCatalogItem("datasets", "#dataset-id");
    var selectedEpisode = datasetEpisodes(selectedDataset).find(function (item) { return itemId(item) === $("#episode-id").value; });
    var effectiveBudget = normalizedEvolutionBudget(
      $("#max-generations").value,
      $("#candidates-per-generation").value,
      $("#max-candidates").value
    );
    var values = [
      ["训练数据集", itemLabel(selectedCatalogItem("datasets", "#dataset-id")) === "未选择" ? "未选择可运行数据集" : itemLabel(selectedCatalogItem("datasets", "#dataset-id"))],
      ["研究领域（自动推导）", itemLabel(selectedCatalogItem("domain_packs", "#domain-pack"))],
      ["策略模型（API）", itemLabel(selectedModelCatalogItem("#policy-model-id"))],
      ["独立评审模型（API）", itemLabel(selectedModelCatalogItem("#judge-model-id"))],
      ["进化预算", formatNumber(effectiveBudget.max_generations) + " 轮 · 每轮 " + formatNumber(effectiveBudget.candidates_per_generation) + " 个 · 总上限 " + formatNumber(effectiveBudget.requested_max_candidates) + " 个候选"],
      ["入围候选持续优化", formatNumber(Number($("#formal-origin-count").value)) + " origins · batch " + formatNumber(Number($("#local-batch-origin-count").value)) + " · 每批最多 " + formatNumber(Number($("#max-local-edits-per-batch").value)) + " 处改动 · 样本并发 " + formatNumber(normalizedSampleConcurrency($("#sample-concurrency").value))],
      ["自动绑定", "预测模型、进化策略、评测器由模型提出并由宿主登记能力校验确定"],
      ["知识检索", $("#knowledge-online-enabled").checked ? "每轮在线检索并冻结知识快照" : "仅使用内置知识目录"],
      ["运行环境", state.usingDemo ? "浏览器演示" : environmentText(state.catalog.dsh.environment)]
    ];
    $("#selected-summary").innerHTML = values.map(function (item) { return "<div><dt>" + escapeHTML(item[0]) + "</dt><dd title=\"" + escapeHTML(item[1]) + "\">" + escapeHTML(item[1]) + "</dd></div>"; }).join("");
    renderModelConnections();
  }

  function createRunButtonLabel(createStatus, hasActiveRun) {
    if (createStatus && createStatus.state === "failed") { return "重新创建进化运行"; }
    return hasActiveRun ? "创建新的进化运行" : "创建并启动进化运行";
  }

  function createRunHint(createStatus, allReady, unmetChecks) {
    if (!allReady) {
      return "未满足：" + unmetChecks.map(function (item) { return item.label; }).join("；");
    }
    if (createStatus) { return createStatus.message; }
    return "配置将在服务端冻结；可同时创建多个运行，后台会并行推进。";
  }

  function renderParameters() {
    var schedule;
    try { schedule = optimizationScheduleFromControls(); }
    catch (_error) { schedule = normalizedOptimizationSchedule({}); }
    var cellsPerOrigin = predictionCellsPerOrigin();
    var microbatch = normalizedSampleAgentBatchSize($("#sample-agent-batch-size").value);
    var candidateConcurrency = normalizedCandidateConcurrency($("#candidate-concurrency").value);
    var concurrency = normalizedSampleConcurrency($("#sample-concurrency").value);
    var budget = candidateBudgetStatus();
    var screeningCandidateOrigins = 4 * schedule.screening_origin_count;
    var formalCandidateOrigins = schedule.finalist_count * schedule.formal_origin_count_per_finalist;
    var holdoutCandidateOrigins = (schedule.finalist_count + 1) * schedule.selection_holdout_origin_count;
    var generationCandidateOrigins = screeningCandidateOrigins + formalCandidateOrigins + holdoutCandidateOrigins;
    var generationScoringCells = generationCandidateOrigins * cellsPerOrigin;
    var runCandidateOrigins = generationCandidateOrigins * budget.max_generations;
    var runScoringCells = generationScoringCells * budget.max_generations;
    var uniqueOrigins = schedule.formal_origin_count_per_finalist + budget.max_generations * (schedule.screening_origin_count + schedule.selection_holdout_origin_count);
    var batchCount = schedule.formal_origin_count_per_finalist / schedule.local_batch_origin_count;
    var maximumEdits = batchCount * schedule.max_local_edits_per_batch;
    $("#parameter-summary-pill").textContent = "每个入围候选 " + formatNumber(batchCount) + " × " + formatNumber(schedule.local_batch_origin_count);
    $("#agent-update-scope").textContent = "每个入围候选 " + formatNumber(batchCount) + " × " + formatNumber(schedule.local_batch_origin_count);
    var budgetState = $("#parameter-budget-state");
    budgetState.textContent = budget.budget_sufficient ? "预算完整" : "预算不足";
    budgetState.className = budget.budget_sufficient ? "" : "is-insufficient";
    var values = [
      ["迭代结构", formatNumber(budget.max_generations) + " 轮 · 每轮固定 4 个候选 · 同一 64 时点初筛后 Top 2"],
      ["局部持续优化", "每个入围候选 " + formatNumber(schedule.formal_origin_count_per_finalist) + " origins = " + formatNumber(batchCount) + " × " + formatNumber(schedule.local_batch_origin_count) + "；最多 " + formatNumber(maximumEdits) + " 处局部改动"],
      ["单轮执行预算", formatNumber(screeningCandidateOrigins) + " + " + formatNumber(formalCandidateOrigins) + " + " + formatNumber(holdoutCandidateOrigins) + " = " + formatNumber(generationCandidateOrigins) + " candidate-origins = " + formatNumber(generationScoringCells) + " cells"],
      ["全程执行预算", formatNumber(runCandidateOrigins) + " candidate-origins / " + formatNumber(runScoringCells) + " cells；需要 " + formatNumber(uniqueOrigins) + " 个不同数据时点"],
      ["请求组织", "按因果预测起点组成 origin wave · wave 上限 " + formatNumber(microbatch) + " · 实际完成数以运行进度为准"],
      ["并发上限", formatNumber(candidateConcurrency) + " 个编排候选；两条 lane 共享 " + formatNumber(concurrency) + " 个 run 级在飞请求"],
      ["候选总预算", formatNumber(budget.requested_max_candidates) + " 个（至少 " + formatNumber(budget.required_candidates) + " 个）"],
      ["上下文与输出", "由 DSH Session 压缩和模型路由统一管理，不设逐样本 Token 硬上限"],
      ["复现与检索", ($("#fixed-seed").checked ? "固定种子" : "记录生成种子") + " · " + ($("#knowledge-online-enabled").checked ? "在线检索" : "内置目录")]
    ];
    $("#parameter-summary").innerHTML = values.map(function (item) {
      return "<div><dt>" + escapeHTML(item[0]) + "</dt><dd>" + escapeHTML(item[1]) + "</dd></div>";
    }).join("");
  }

  function renderModelConnections() {
    var selections = [
      { role: "策略模型 API", item: selectedModelCatalogItem("#policy-model-id") },
      { role: "独立评审 API", item: selectedModelCatalogItem("#judge-model-id") }
    ];
    var readyCount = selections.filter(function (entry) { return modelCredentialReady(entry.item); }).length;
    var configured = dshModelTotalCount(false);
    var configuredStrategy = dshModelRoleCount("strategy", false);
    var configuredReview = dshModelRoleCount("review", false);
    $("#model-connection-summary").textContent = configured
      ? "已配置：策略职责 " + formatNumber(configuredStrategy) + "；评审职责 " + formatNumber(configuredReview)
      : readyCount + " / " + selections.length + " 项可用";
    $("#model-connection-list").innerHTML = selections.map(function (entry) {
      var item = entry.item;
      var id = itemId(item);
      var ready = modelCredentialReady(item);
      var connectionState = String(item && item.connection && item.connection.state || "").toLowerCase();
      var callFailed = ["error", "unavailable", "unreachable"].indexOf(connectionState) >= 0;
      var status = !item ? "未选择" : modelConnectionStateText(item) || (ready ? "已配置" : "不可执行");
      var tone = !ready ? "pill-red" : callFailed ? "pill-amber" : connectionState === "available" ? "pill-green" : "pill-blue";
      var digest = item && item.configuration_digest || "";
      var failureReason = callFailed ? modelConnectionErrorText(item) : "";
      var technicalTitle = [id ? "模型：" + id : "", digest ? "配置校验值：" + digest : ""].filter(Boolean).join("\n");
      return "<div class=\"model-connection-row\"><div><span>" + escapeHTML(entry.role) + "</span><strong>" + escapeHTML(item ? itemBaseLabel(item) : "未选择") + "</strong><code title=\"" + escapeHTML(technicalTitle) + "\">" + escapeHTML(shortId(id || "未提供")) + (digest ? " · " + escapeHTML(shortId(digest)) : "") + "</code>" + (failureReason ? "<small class=\"model-connection-error\">" + escapeHTML(failureReason) + "</small>" : "") + "</div><span class=\"pill " + tone + "\">" + escapeHTML(status) + "</span></div>";
    }).join("");
  }

  function normalizeSchema(schema, rows) {
    var normalized = [];
    if (Array.isArray(schema)) { normalized = schema.map(function (field) { return typeof field === "string" ? { name: field } : field; }); }
    else if (schema && typeof schema === "object") { normalized = Object.keys(schema).map(function (key) { return Object.assign({ name: key }, typeof schema[key] === "object" ? schema[key] : { display_name_zh: schema[key] }); }); }
    var first = rows && rows[0] && (rows[0].values || rows[0].features || rows[0]);
    if (!normalized.length && first) { normalized = Object.keys(first).map(function (key) { return { name: key }; }); }
    if (rows && rows[0] && rows[0].timestamp != null && !normalized.some(function (field) { return (field.name || field.id) === "timestamp"; })) { normalized.unshift({ name: "timestamp", display_name_zh: "观测时间（数据源本地时）", type: "time" }); }
    if (rows && rows[0] && rows[0].index != null && !normalized.some(function (field) { return (field.name || field.id) === "index"; })) { normalized.unshift({ name: "index", display_name_zh: "样本序号", type: "integer" }); }
    return normalized.filter(function (field, index, values) {
      var key = field.name || field.id;
      return key && values.findIndex(function (item) { return (item.name || item.id) === key; }) === index;
    });
  }
