"use strict";

  function renderSystemBanner() {
    var banner = $("#system-banner");
    var title = $("#system-title");
    var detail = $("#system-detail");
    var retry = $("#retry-button");
    banner.className = "system-banner";
    retry.hidden = true;
    if (state.commandError) {
      banner.hidden = false; banner.classList.add("is-error"); title.textContent = "操作未完成"; detail.textContent = state.commandError; return;
    }
    var notice = state.controlNotice;
    if (notice && state.activeRun && notice.runId === state.activeRun.id) {
      if (notice.mode !== "verifying" && controlStatusConfirmed(notice.action, state.activeRun)) {
        state.controlNotice = null;
        clearCommandKey("control");
      } else {
        banner.hidden = false; banner.classList.add(notice.mode === "unknown" ? "is-demo" : "is-loading");
        title.textContent = notice.mode === "unknown" ? "结果待确认" : "正在确认操作";
        detail.textContent = notice.message; return;
      }
    }
    var creation = pendingCreateStatus();
    if (creation) {
      banner.hidden = false; banner.classList.add("is-loading"); title.textContent = createPhaseLabel(creation);
      detail.textContent = creation.message + "请勿重复提交。"; return;
    }
    if (state.loadState === "loading") {
      banner.hidden = false; banner.classList.add("is-loading"); title.textContent = "正在连接"; detail.textContent = "正在读取数据集、模型和运行记录。"; return;
    }
    if (state.loadState === "error" || state.loadState === "stale") {
      banner.hidden = false; banner.classList.add("is-error"); title.textContent = state.loadState === "stale" ? "连接中断，显示上次状态" : "暂时无法连接"; detail.textContent = state.lastError || "请检查本地服务是否启动，再点击“重新连接”。"; retry.hidden = false; return;
    }
    if (state.usingDemo) {
      banner.hidden = false; banner.classList.add("is-demo"); title.textContent = "演示模式"; detail.textContent = "这里是示例数据；操作不会启动真实训练，也不会写入后台。"; return;
    }
    if (state.catalog.dsh.harness_execution === "dsh_native_agent_runtime") {
      banner.hidden = false; title.textContent = "服务已连接"; detail.textContent = "已登记 " + formatNumber(dshModelTotalCount(false)) + " 个模型，任务由后台执行。"; return;
    }
    if (state.catalog.dsh.environment && state.catalog.dsh.environment !== "production") {
      banner.hidden = false; banner.classList.add("is-demo"); title.textContent = environmentText(state.catalog.dsh.environment); detail.textContent = "当前不是生产环境，已登记 " + formatNumber(dshModelTotalCount(false)) + " 个模型。"; return;
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

  function contextGenerationText(run) {
    if (!run) { return "0 / 0"; }
    var total = Number(run.total_generations || 0);
    var progress = run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : {};
    var projected = Number(progress.current_generation);
    var status = String(run.status || "").toLowerCase();
    var current = Number.isInteger(projected) && projected > 0
      ? projected
      : ["running", "paused", "starting", "preflight"].indexOf(status) >= 0
        ? Number(run.generation || 0) + 1
        : Number(run.generation || 0);
    if (Number.isFinite(total) && total > 0) { current = Math.min(total, Math.max(0, current)); }
    return formatNumber(current) + " / " + (total > 0 ? formatNumber(total) : "—");
  }

  function renderContext() {
    var select = $("#run-select");
    var creation = pendingCreateStatus();
    var runs = visibleRuns();
    var cancelledCount = cancelledEmptyRunCount();
    select.innerHTML = runs.length ? runs.map(function (run) {
      var runEvents = state.activeRun && state.activeRun.id === run.id ? state.events : run.events;
      var archivePrefix = run.archived ? "已归档 · " : "";
      var isCurrent = !creation && state.activeRun && state.activeRun.id === run.id;
      var statusLabel = contextRunExplanation(run, runEvents).label;
      return "<option value=\"" + escapeHTML(run.id) + "\"" + (isCurrent ? " selected" : "") + ">" + escapeHTML(shortId(run.id)) + " · " + archivePrefix + escapeHTML(statusLabel) + "</option>";
    }).join("") : "<option value=\"\">暂无进化运行</option>";
    if (creation) { select.innerHTML = "<option value=\"\" selected>新运行：" + escapeHTML(createPhaseLabel(creation)) + "</option>" + select.innerHTML; }
    select.disabled = !runs.length || state.busy || state.refreshing || Boolean(creation);
    $("#load-older-runs").hidden = !state.runListCursor;
    $("#load-older-runs").disabled = Boolean(state.loadingOlderRuns || state.busy || state.refreshing);
    $("#show-cancelled-empty-runs").checked = state.showCancelledEmptyRuns;
    $("#show-cancelled-empty-runs").disabled = state.busy || state.refreshing;
    $("#cancelled-empty-count").textContent = String(cancelledCount);
    $("#cancelled-empty-filter").hidden = cancelledCount === 0;
    $("#show-archived-runs").checked = state.showArchivedRuns;
    $("#show-archived-runs").disabled = state.busy || state.refreshing;
    $("#archived-count").textContent = String(state.archivedRunCount);
    $("#archived-filter").hidden = state.archivedRunCount === 0 && !state.showArchivedRuns;
    var run = creation ? null : state.activeRun;
    var autoAdvanceActive = Boolean(run && run.status === "running" && runHasContinuousAutoProgress(run));
    var explanation = contextRunExplanation(run, state.events);
    var contextStatus = explanation.label;
    if (!run && state.runOverviewLoading) {
      contextStatus = "正在读取概况";
      select.value = state.runOverviewLoading;
    }
    $("#run-status").textContent = creation ? createPhaseLabel(creation) : run ? (run.archived ? "已归档 · " : "") + contextStatus : state.runOverviewLoading ? "正在读取概况" : state.runOverviewError ? "概况暂不可用" : "未创建";
    $("#run-status-detail").textContent = creation ? creation.message : run ? explanation.detail : state.runOverviewLoading ? "历史详情在后台加载，可先配置新的运行。" : state.runOverviewError || explanation.nextAction;
    $("#run-status-detail").title = run ? explanation.nextAction : "";
    $("#run-next-action").textContent = run ? explanation.nextAction : "";
    $("#generation-label").textContent = contextGenerationText(run);
    $("#candidate-count-label").textContent = run ? run.candidates_count + " / " + (run.max_candidates || "—") : "0 / 0";
    if (!run && (state.runOverviewLoading || state.runOverviewError)) {
      $("#generation-label").textContent = "—";
      $("#candidate-count-label").textContent = "—";
    }
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
    $("#advance-button").title = hardTokenPause ? "冻结的" + tokenBudgetSubjectText(run) + " Token 硬预算已耗尽；请创建更高预算的新运行。" : retryCircuitPaused ? retryCircuitDetailText(run) : autoAdvanceActive ? "运行已进入自动连续推进；暂停后可人工干预。" : pausedAdvance ? "恢复运行后自动执行下一轮。" : waitingForAdvance ? "从已持久化的轮次进度继续执行。" : recoveringCurrentRound ? "继续当前未完成轮次。" : "";
    $("#pause-button").disabled = state.busy || !online || !canControl || !canPauseOrResume;
    $("#pause-button").textContent = state.pendingAction === "pause" ? "正在暂停" : state.pendingAction === "resume" ? "正在恢复" : hardTokenPause ? "预算已用尽" : retryCircuitPaused ? "恢复并重试" : run && run.status === "paused" ? "恢复运行" : "暂停";
    $("#pause-button").title = hardTokenPause ? "冻结的" + tokenBudgetSubjectText(run) + " Token 硬预算已耗尽，直接恢复不会产生新进展。" : retryCircuitPaused ? retryCircuitDetailText(run) : "";
    $("#advance-button").textContent = autoAdvanceActive ? "自动执行中" : state.pendingAction === "resume" ? "正在恢复" : state.pendingAction === "advance" || state.pendingAction === "auto-advance" ? "正在执行" : retryCircuitPaused ? "检查后重试" : pausedAdvance && runHasContinuousAutoProgress(run) ? "恢复并自动执行" : pausedAdvance ? "恢复并执行" : waitingForAdvance && Number(run && run.generation || 0) === 0 ? "开始第一轮" : recoveringCurrentRound ? "继续当前轮次" : "继续下一步";
    $("#cancel-button").disabled = state.busy || !online || !canControl || !canCancel;
    $("#cancel-button").textContent = state.pendingAction === "cancel" ? "正在停止" : "停止运行";
    var terminal = runIsTerminal(run);
    $("#archive-button").disabled = state.busy || !online || !hasCapability("run.archive") || !terminal;
    $("#archive-button").textContent = state.pendingAction === "archive" ? "正在归档" : state.pendingAction === "restore" ? "正在移回" : run && run.archived ? "移回列表" : "归档记录";
    $("#archive-button").title = run && !terminal ? "请先完成或停止运行。" : run && run.archived ? "恢复到默认运行列表。" : "从默认列表隐藏，但保留完整历史证据。";
    $("#delete-button").hidden = !(run && run.archived);
    $("#delete-button").disabled = state.busy || !online || !hasCapability("run.delete") || !terminal || !(run && run.archived);
    $("#delete-button").textContent = state.pendingAction === "delete" ? "正在删除" : "永久删除";
    $("#delete-button").title = "永久删除该运行的事件和命令记录，不可恢复。";
  }

  function contextRunExplanation(run, events) {
    if (run && run.schema_version === "ecologyrsi-dsh.browser-run-summary/2") {
      return {label: run.status === "running" && run.auto_progress ? "自动执行中" : statusText(run.status), detail: "", nextAction: "选择运行以读取详细状态。"};
    }
    if (run && run.status === "running") {
      if (runHasContinuousAutoProgress(run)) {
        return {label: "自动执行中", detail: "系统会自动继续未完成的轮次。", nextAction: "无需重复操作；需要调整时可以先暂停。", tone: "running"};
      }
    }
    return runStatusExplanation(run, events);
  }

  function optimizationControlSnapshot() {
    var raw = {
      formal_origin_count: $("#formal-origin-count").value,
      local_batch_origin_count: $("#local-batch-origin-count").value,
      max_local_edits_per_batch: $("#max-local-edits-per-batch").value,
      selection_holdout_origin_count: $("#selection-holdout-origin-count").value,
      sample_agent_batch_size: $("#sample-agent-batch-size").value,
      candidate_concurrency: $("#candidate-concurrency").value,
      sample_concurrency: $("#sample-concurrency").value
    };
    try {
      return {
        valid: true,
        raw: raw,
        schedule: normalizedOptimizationSchedule(raw),
        sample_agent_batch_size: strictInteger(raw.sample_agent_batch_size, "单时点向量单元容量", 9, 9),
        candidate_concurrency: strictInteger(raw.candidate_concurrency, "候选并发数", 1, 8),
        sample_concurrency: strictInteger(raw.sample_concurrency, "逐样本并发请求数", 1, sampleConcurrencyMaximum)
      };
    } catch (error) {
      return {valid: false, raw: raw, message: error && error.message ? error.message : "请检查输入参数"};
    }
  }

  function optimizationControlInputText(snapshot) {
    var raw = snapshot && snapshot.raw || {};
    function shown(value) { return value == null || value === "" ? "空" : String(value); }
    return "更新时点 " + shown(raw.formal_origin_count)
      + " · batch " + shown(raw.local_batch_origin_count)
      + " · 每批改动 " + shown(raw.max_local_edits_per_batch)
      + " · 留出 " + shown(raw.selection_holdout_origin_count)
      + " · 向量单元 " + shown(raw.sample_agent_batch_size)
      + " · 候选并发 " + shown(raw.candidate_concurrency)
      + " · 样本并发 " + shown(raw.sample_concurrency);
  }

  function renderReadiness() {
    var checks = readiness();
    var allReady = checks.every(function (item) { return item.ready; });
    var unmetChecks = checks.filter(function (item) { return !item.ready; });
    $("#readiness-list").innerHTML = checks.map(function (item) { return "<li><span class=\"check-mark " + (item.ready ? "" : "pending") + "\">" + (item.ready ? "✓" : "·") + "</span><span>" + escapeHTML(item.label) + "</span></li>"; }).join("");
    var pill = $("#readiness-pill");
    pill.className = "pill " + (allReady ? "pill-green" : "pill-amber");
    pill.textContent = allReady ? "可以创建运行" : state.loadState === "loading" ? "正在读取目录" : "配置尚未就绪";
    // Wait for the exact current schedule's asynchronous capacity check.
    // Other invalid settings remain clickable so submission can explain them.
    var capacityPending = capacityVerificationPending();
    $("#start-button").disabled = state.busy || capacityPending;
    var createStatus = state.createStatus;
    $("#start-button").textContent = state.pendingAction === "create"
      ? createPhaseLabel(pendingCreateStatus())
      : capacityPending ? "正在核验数据容量"
      : createRunButtonLabel(createStatus, Boolean(state.activeRun));
    $("#create-hint").textContent = capacityPending && state.pendingAction !== "create" ? "正在核验当前参数对应的数据容量，完成后可创建运行。" : createRunHint(createStatus, allReady, unmetChecks);
    var selectedDataset = selectedCatalogItem("datasets", "#dataset-id");
    var selectedEpisode = datasetEpisodes(selectedDataset).find(function (item) { return itemId(item) === $("#episode-id").value; });
    var optimizationControls = optimizationControlSnapshot();
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
      ["入围候选持续优化", optimizationControls.valid
        ? formatNumber(optimizationControls.schedule.formal_origin_count_per_finalist) + " 个预测时点 · 每批 " + formatNumber(optimizationControls.schedule.local_batch_origin_count) + " · 每批最多 " + formatNumber(optimizationControls.schedule.max_local_edits_per_batch) + " 处改动 · 样本并发 " + formatNumber(optimizationControls.sample_concurrency)
        : "参数无效：" + optimizationControls.message + "（" + optimizationControlInputText(optimizationControls) + "）"],
      ["预测方案", "运行中由模型选择、切换和优化，可仅使用基线"],
      ["独立评测器", itemLabel(selectedCatalogItem("evaluators", "#evaluator-id"))],
      ["知识检索", $("#knowledge-online-enabled").checked ? "每轮在线检索并冻结知识快照" : "仅使用内置知识目录"],
      ["运行环境", state.usingDemo ? "浏览器演示" : environmentText(state.catalog.dsh.environment)]
    ];
    $("#selected-summary").innerHTML = values.map(function (item) { return "<div><dt>" + escapeHTML(item[0]) + "</dt><dd title=\"" + escapeHTML(item[1]) + "\">" + escapeHTML(item[1]) + "</dd></div>"; }).join("");
    renderModelConnections();
  }

  function createRunButtonLabel(createStatus, hasActiveRun) {
    if (createStatus && createStatus.state === "failed") { return "重新创建运行"; }
    if (createStatus && ["verifying", "pending"].indexOf(createStatus.state) >= 0) { return "核对创建状态"; }
    return hasActiveRun ? "创建新运行" : "创建并开始";
  }

  function createRunHint(createStatus, allReady, unmetChecks) {
    if (!allReady) {
      return "未满足：" + unmetChecks.map(function (item) { return item.label; }).join("；");
    }
    if (createStatus) { return createStatus.message; }
    return "开始后配置将固定。后台会持续执行，您可以在“运行过程”查看进度。";
  }

  function renderParameters() {
    var optimizationControls = optimizationControlSnapshot();
    if (!optimizationControls.valid) {
      $("#parameter-summary-pill").textContent = "参数无效";
      $("#agent-update-scope").textContent = "请修正后重新计算";
      var invalidBudgetState = $("#parameter-budget-state");
      invalidBudgetState.textContent = "参数无效";
      invalidBudgetState.className = "is-insufficient";
      var invalidValues = [
        ["参数状态", "无法计算：" + optimizationControls.message],
        ["当前输入", optimizationControlInputText(optimizationControls)]
      ];
      $("#parameter-summary").innerHTML = invalidValues.map(function (item) {
        return "<div><dt>" + escapeHTML(item[0]) + "</dt><dd>" + escapeHTML(item[1]) + "</dd></div>";
      }).join("");
      return;
    }
    var schedule = optimizationControls.schedule;
    var cellsPerOrigin = predictionCellsPerOrigin();
    var microbatch = optimizationControls.sample_agent_batch_size;
    var candidateConcurrency = optimizationControls.candidate_concurrency;
    var concurrency = optimizationControls.sample_concurrency;
    var budget = candidateBudgetStatus();
    var batchCount = schedule.formal_origin_count_per_finalist / schedule.local_batch_origin_count;
    var pairedMode = String(schedule.local_evaluation_mode || "").toLowerCase() === "paired_champion_challenger";
    var screeningCandidateOrigins = 4 * schedule.screening_origin_count;
    var formalOriginsPerFinalist = pairedMode
      ? schedule.local_batch_origin_count + 2 * Math.max(0, batchCount - 1) * schedule.local_batch_origin_count
      : schedule.formal_origin_count_per_finalist;
    var formalCandidateOrigins = schedule.finalist_count * formalOriginsPerFinalist;
    var holdoutReplicas = state.usingDemo ? 1 : 2;
    var holdoutCandidateOrigins = holdoutReplicas * (schedule.finalist_count + 1) * schedule.selection_holdout_origin_count;
    var generationCandidateOrigins = screeningCandidateOrigins + formalCandidateOrigins + holdoutCandidateOrigins;
    var generationScoringCells = generationCandidateOrigins * cellsPerOrigin;
    var runCandidateOrigins = generationCandidateOrigins * budget.max_generations;
    var runScoringCells = generationScoringCells * budget.max_generations;
    var uniqueOrigins = schedule.formal_origin_count_per_finalist + budget.max_generations * (schedule.screening_origin_count + schedule.selection_holdout_origin_count);
    var capacity = state.cohortCapacityReport;
    var capacityOriginText = capacity && capacity.sufficient === true && Number(capacity.reused_origin_occurrences || 0) > 0
      ? "计划 " + formatNumber(capacity.planned_origin_occurrences == null ? uniqueOrigins : capacity.planned_origin_occurrences) + " 个起点；不足部分按 occurrence 循环复用"
      : "需要 " + formatNumber(capacity && capacity.planned_origin_occurrences != null ? capacity.planned_origin_occurrences : uniqueOrigins) + " 个起点 occurrence";
    if (capacity && capacity.cohort_reuse_policy === "purged_no_reuse@1") {
      capacityOriginText += "；按目标成熟时间隔离，禁止循环复用；适应数据跨 " + formatNumber(Number((capacity.maturity_gaps || {}).adaptation_day_buckets) || 0) + " 个自然日";
    }
    var maximumEdits = Math.max(0, batchCount - (pairedMode ? 1 : 0)) * schedule.max_local_edits_per_batch;
    $("#parameter-summary-pill").textContent = "每个入围候选 " + formatNumber(batchCount) + " × " + formatNumber(schedule.local_batch_origin_count);
    $("#agent-update-scope").textContent = "每个入围候选 " + formatNumber(batchCount) + " × " + formatNumber(schedule.local_batch_origin_count);
    var budgetState = $("#parameter-budget-state");
    var capacitySufficient = state.usingDemo || Boolean(capacity && capacity.sufficient === true);
    var capacityPending = !state.usingDemo && capacityVerificationPending();
    budgetState.textContent = !budget.budget_sufficient ? "预算不足" : capacityPending ? "正在核验数据容量" : capacitySufficient ? "预算与数据容量完整" : state.cohortCapacityError ? "数据容量暂不可用" : "数据容量不足";
    budgetState.className = budget.budget_sufficient && (capacitySufficient || capacityPending) ? "" : "is-insufficient";
    var values = [
      ["迭代结构", formatNumber(budget.max_generations) + " 轮 · 每轮固定 4 个候选 · 同组 64 个时点评测后选出 2 个"],
      ["局部持续优化", pairedMode
        ? "两个入围候选共享 " + formatNumber(schedule.formal_origin_count_per_finalist) + " 个预测时点；每个方案包含 1 个初始批次和 " + formatNumber(Math.max(0, batchCount - 1)) + " 个新旧版本同批比较；" + "通过实际增益、分项不退化及配对证据检查后保留；证据不足待复核" + "；最多 " + formatNumber(maximumEdits) + " 处局部改动"
        : "每个入围候选 " + formatNumber(schedule.formal_origin_count_per_finalist) + " 个时点 = " + formatNumber(batchCount) + " × " + formatNumber(schedule.local_batch_origin_count) + "；最多 " + formatNumber(maximumEdits) + " 处局部改动"],
      ["单轮执行预算", formatNumber(screeningCandidateOrigins) + " + " + formatNumber(formalCandidateOrigins) + " + " + formatNumber(holdoutCandidateOrigins) + " = " + formatNumber(generationCandidateOrigins) + " 次时点预测 = " + formatNumber(generationScoringCells) + " 个评分项"],
      ["留出重复推理", "每个方案独立推理 " + formatNumber(holdoutReplicas) + " 次，已计入执行预算；独立观测数保持不变"],
      ["全程执行预算", formatNumber(runCandidateOrigins) + " 次时点预测 / " + formatNumber(runScoringCells) + " 个评分项；" + capacityOriginText],
      ["数据容量", state.cohortCapacityLoading ? "正在核验" : capacity ? cohortCapacityLabel(capacity) : state.cohortCapacityError || "等待核验"],
      ["请求组织", "每个预测时点使用一条完整向量链 · " + formatNumber(microbatch) + " 个评分单元原子提交"],
      ["并发上限", formatNumber(candidateConcurrency) + " 个方案；全运行共享 " + formatNumber(concurrency) + " 个同时预测的时点"],
      ["候选总预算", formatNumber(budget.requested_max_candidates) + " 个（至少 " + formatNumber(budget.required_candidates) + " 个）"],
      ["上下文与输出", "不设跨调用的逐样本 Token 总预算；Planner/Repair 最多 4,096 tokens，Critic 最多 2,048 tokens"],
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
