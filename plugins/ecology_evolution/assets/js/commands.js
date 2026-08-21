"use strict";

  // A generation is a durable transaction on the service.  The browser only
  // schedules the next transaction after the previous request has returned
  // and the projection is back at its waiting boundary.  This keeps retries
  // serial and prevents duplicate candidate batches when a user refreshes or
  // a DSH host sends a second context message.
  var autoAdvanceDelayMs = 450;
  var autoAdvancePollMs = 1200;
  var autoAdvanceRetryLimit = 3;

  function clearAutoAdvanceTimer() {
    if (state.autoAdvanceTimer != null) {
      window.clearTimeout(state.autoAdvanceTimer);
      state.autoAdvanceTimer = null;
    }
  }

  function stopAutoAdvance(runId, options) {
    var activeRunId = state.autoAdvanceRunId;
    if (runId && activeRunId && activeRunId !== runId) { return; }
    clearAutoAdvanceTimer();
    state.autoAdvanceRunId = null;
    state.autoAdvanceContextEpoch = null;
    state.autoAdvanceRoundStartedAt = null;
    state.autoAdvanceRetry = 0;
    var settings = options || {};
    if (settings.resetTiming === true) { state.autoAdvanceLastDurationMs = null; }
    if (settings.block === true) {
      state.autoAdvanceBlockedRunId = runId || activeRunId || null;
      state.autoAdvanceError = settings.message || state.commandError || "自动推进已暂停，请检查运行事件后重试。";
    } else if (settings.clearError !== false) {
      state.autoAdvanceBlockedRunId = null;
      state.autoAdvanceError = null;
    }
  }

  function queueAutoAdvance(runId, delay) {
    if (!runId || state.autoAdvanceRunId !== runId || state.autoAdvanceContextEpoch !== state.contextEpoch || !state.autoAdvanceEnabled) { return false; }
    clearAutoAdvanceTimer();
    state.autoAdvanceTimer = window.setTimeout(function () {
      state.autoAdvanceTimer = null;
      autoAdvanceTick(runId);
    }, Math.max(0, Number(delay) || 0));
    return true;
  }

  function autoAdvanceWaiting(run) {
    // New servers own continuous runs in a durable background worker.  Their
    // queued/waiting boundary is a read-only observation point; the browser
    // must never race that worker with a second advance command.
    if (serverAutoProgressEnabled(run)) { return false; }
    return Boolean(run) && runNeedsAdvanceAction(run, state.events);
  }

  function serverAutoProgressEnabled(run) {
    return runHasContinuousAutoProgress(run);
  }

  function autoAdvanceTerminalOrPaused(run) {
    if (!run) { return true; }
    var status = String(run.status || "").toLowerCase();
    return status !== "running" || runIsTerminal(run);
  }

  function autoAdvanceFailure(runId) {
    if (state.autoAdvanceRunId !== runId) { return; }
    state.autoAdvanceRetry = Number(state.autoAdvanceRetry || 0) + 1;
    var detail = state.commandError || "服务未完成本轮推进。";
    if (state.autoAdvanceRetry <= autoAdvanceRetryLimit) {
      var delay = Math.min(8000, autoAdvancePollMs * Math.pow(2, state.autoAdvanceRetry - 1));
      showToast("本轮推进未完成，" + Math.ceil(delay / 1000) + " 秒后自动重试（第 " + state.autoAdvanceRetry + " 次）。");
      queueAutoAdvance(runId, delay);
      return;
    }
    stopAutoAdvance(runId, { block: true, message: "自动推进已暂停：" + detail + " 可检查事件后手动重试。" });
    showToast(state.autoAdvanceError);
    renderAll();
  }

  function autoAdvanceTick(runId) {
    if (state.autoAdvanceRunId !== runId || state.autoAdvanceContextEpoch !== state.contextEpoch || !state.autoAdvanceEnabled) { return; }
    if (!state.activeRun || state.activeRun.id !== runId) {
      stopAutoAdvance(runId);
      renderAll();
      return;
    }
    var run = state.activeRun;
    if (autoAdvanceTerminalOrPaused(run)) {
      stopAutoAdvance(runId);
      renderAll();
      return;
    }
    if (!serverAutoProgressEnabled(run) && !hasCapability("evolution.run.advance")) {
      stopAutoAdvance(runId, { block: true, message: "自动推进已暂停：当前 DSH 会话未授予进化推进能力。" });
      renderAll();
      return;
    }
    if (state.busy || state.refreshing) {
      queueAutoAdvance(runId, autoAdvancePollMs);
      return;
    }
    // A stage may still be running even though the last request returned.  A
    // read-only poll lets the service finish its durable barrier before the
    // next advance command is issued.
    if (serverAutoProgressEnabled(run) || !autoAdvanceWaiting(run)) {
      refreshProgressForRun(runId).then(function () {
        if (state.autoAdvanceRunId !== runId) { return; }
        var current = state.activeRun;
        if (autoAdvanceTerminalOrPaused(current)) {
          stopAutoAdvance(runId);
          renderAll();
        } else if (serverAutoProgressEnabled(current)) {
          queueAutoAdvance(runId, autoAdvancePollMs);
        } else if (autoAdvanceWaiting(current)) {
          queueAutoAdvance(runId, autoAdvanceDelayMs);
        } else {
          queueAutoAdvance(runId, autoAdvancePollMs);
        }
      }).catch(function () { queueAutoAdvance(runId, autoAdvancePollMs); });
      return;
    }
    state.autoAdvanceRoundStartedAt = Date.now();
    state.pendingAction = "auto-advance";
    renderAll();
    advanceRun({ automatic: true }).then(function (ok) {
      if (state.autoAdvanceRunId !== runId) { return; }
      if (!ok) {
        autoAdvanceFailure(runId);
        return;
      }
      if (state.autoAdvanceRoundStartedAt != null) {
        state.autoAdvanceLastDurationMs = Math.max(0, Date.now() - state.autoAdvanceRoundStartedAt);
      }
      state.autoAdvanceRoundStartedAt = null;
      state.autoAdvanceRetry = 0;
      state.autoAdvanceRoundsCompleted = Number(state.autoAdvanceRoundsCompleted || 0) + 1;
      var current = state.activeRun;
      if (autoAdvanceTerminalOrPaused(current)) {
        stopAutoAdvance(runId);
        renderAll();
        return;
      }
      // Leave a small observable boundary between rounds.  This is not a
      // fake training delay; it gives the host projection and event ledger a
      // chance to settle and makes an unexpectedly empty round diagnosable.
      queueAutoAdvance(runId, autoAdvanceDelayMs);
      renderAll();
    }).catch(function () { autoAdvanceFailure(runId); });
  }

  function ensureAutoAdvanceForRun(runId) {
    if (!state.autoAdvanceEnabled) { return false; }
    var run = state.activeRun;
    var targetId = runId || run && run.id;
    if (!targetId || !run || run.id !== targetId) { return false; }
    if (state.autoAdvanceOptOutRunIds[targetId] === true) { return false; }
    if (state.autoAdvanceBlockedRunId === targetId) { return false; }
    if (autoAdvanceTerminalOrPaused(run)) {
      stopAutoAdvance(targetId);
      return false;
    }
    if (!serverAutoProgressEnabled(run) && !hasCapability("evolution.run.advance")) { return false; }
    if (serverAutoProgressEnabled(run)) {
      // The durable server worker owns every write for continuous runs. The
      // run monitor already polls their projection, so starting the legacy
      // browser scheduler as a second read loop only doubles ledger replay
      // load without advancing anything.
      if (state.autoAdvanceRunId && state.autoAdvanceRunId !== targetId) {
        stopAutoAdvance(state.autoAdvanceRunId);
      }
      clearAutoAdvanceTimer();
      state.autoAdvanceRunId = targetId;
      state.autoAdvanceContextEpoch = state.contextEpoch;
      state.autoAdvanceBlockedRunId = null;
      state.autoAdvanceError = null;
      if (typeof startRunMonitor === "function") { startRunMonitor(targetId); }
      return true;
    }
    if (state.autoAdvanceRunId && state.autoAdvanceRunId !== targetId) {
      stopAutoAdvance(state.autoAdvanceRunId);
    }
    if (state.autoAdvanceRunId === targetId) {
      if (state.autoAdvanceTimer == null && !state.busy) { queueAutoAdvance(targetId, autoAdvancePollMs); }
      return true;
    }
    state.autoAdvanceRunId = targetId;
    state.autoAdvanceContextEpoch = state.contextEpoch;
    state.autoAdvanceBlockedRunId = null;
    state.autoAdvanceError = null;
    state.autoAdvanceRetry = 0;
    queueAutoAdvance(targetId, 0);
    renderAll();
    return true;
  }

  function normalizedEvolutionBudget(generationsValue, candidatesValue, maximumValue) {
    var generations = Math.max(1, Math.floor(Number(generationsValue) || 1));
    var candidatesPerGeneration = Math.max(1, Math.floor(Number(candidatesValue) || 1));
    var requestedMaximum = Math.max(1, Math.floor(Number(maximumValue) || 1));
    var requiredCandidates = generations * candidatesPerGeneration;
    return {
      max_generations: generations,
      candidates_per_generation: candidatesPerGeneration,
      max_candidates: Math.max(requestedMaximum, requiredCandidates),
      requested_max_candidates: requestedMaximum,
      required_candidates: requiredCandidates,
      budget_sufficient: requestedMaximum >= requiredCandidates
    };
  }

  function normalizedSamplesPerUpdate(value) {
    var parsed = Math.floor(Number(value));
    if (!Number.isFinite(parsed) || parsed < 1) { return 500; }
    return Math.min(parsed, 100000);
  }

  function normalizedSampleAgentBatchSize(value) {
    var parsed = Math.floor(Number(value));
    if (!Number.isFinite(parsed) || parsed < 1) { return 64; }
    return Math.min(parsed, 128);
  }

  function normalizedSampleConcurrency(value) {
    var parsed = Math.floor(Number(value));
    if (!Number.isFinite(parsed) || parsed < 1) { return 2; }
    return Math.min(parsed, 8);
  }

  function candidateBudgetStatus() {
    var budget = normalizedEvolutionBudget(
      $("#max-generations").value,
      $("#candidates-per-generation").value,
      $("#max-candidates").value
    );
    return budget;
  }

  function syncCandidateBudget(options) {
    var settings = options || {};
    var field = $("#max-candidates");
    var help = $("#max-candidates-help");
    if (!field) { return null; }
    if (settings.markManual === true) { state.candidateBudgetManual = true; }
    var budget = candidateBudgetStatus();
    if (state.candidateBudgetManual !== true && budget.requested_max_candidates !== budget.required_candidates) {
      field.value = String(budget.required_candidates);
      budget = candidateBudgetStatus();
    }
    var valid = budget.budget_sufficient;
    field.setCustomValidity(valid ? "" : "最大候选方案数不能小于轮数乘以每轮候选数（当前至少 " + budget.required_candidates + "）。");
    if (help) {
      help.textContent = valid
        ? (state.candidateBudgetManual === true ? "已使用手工总预算；完整执行当前轮数至少需要 " : "默认随轮数和每轮候选数同步；当前完整预算需要 ") + formatNumber(budget.required_candidates) + " 个候选。"
        : "预算不足：当前轮数与每轮候选数至少需要 " + formatNumber(budget.required_candidates) + " 个候选。";
    }
    return budget;
  }

  function createRun(payload) {
    if (!hasCapability("evolution.run.create")) { showToast("当前 DSH 会话未授予创建进化运行的能力。"); return Promise.resolve(null); }
    var requestedSamplesPerUpdate = normalizedSamplesPerUpdate(payload.samples_per_update);
    var minimumSamplesPerUpdate = samplesPerUpdateMinimum();
    if (requestedSamplesPerUpdate < minimumSamplesPerUpdate) {
      showToast("每次更新样本数不足：当前评测至少需要 " + formatNumber(minimumSamplesPerUpdate) + " 个，确保每个目标与预测时距至少出现一次。");
      var samplesPerUpdateField = $("#samples-per-update");
      if (samplesPerUpdateField && typeof samplesPerUpdateField.focus === "function") { samplesPerUpdateField.focus(); }
      return Promise.resolve(null);
    }
    var effectiveBudget = normalizedEvolutionBudget(
      payload.rounds || payload.max_generations,
      payload.candidates_per_generation,
      payload.max_candidates
    );
    if (!effectiveBudget.budget_sufficient) {
      showToast("候选总预算不足：" + formatNumber(effectiveBudget.max_generations) + " 轮 × 每轮 " + formatNumber(effectiveBudget.candidates_per_generation) + " 个候选，至少需要 " + formatNumber(effectiveBudget.required_candidates) + " 个。");
      var maximumField = $("#max-candidates");
      if (maximumField && typeof maximumField.focus === "function") { maximumField.focus(); }
      return Promise.resolve(null);
    }
    var requestedAutoAdvance = payload.auto_advance;
    var continuousAutoAdvance = requestedAutoAdvance == null || requestedAutoAdvance === true || String(requestedAutoAdvance).toLowerCase() === "continuous";
    var body = {
      dataset_id: payload.dataset_id || payload.datasetId,
      episode_id: payload.episode_id || payload.episodeId,
      execution_protocol: "dsh_native_plugin_evolution@1",
      strategy_model_id: payload.strategy_model_id || payload.policy_model_id,
      review_model_id: payload.review_model_id || payload.judge_model_id,
      autonomous_mode: payload.autonomous_mode === true,
      rounds: effectiveBudget.max_generations,
      model_workflow: payload.model_workflow || "research_compile_evolve@1",
      knowledge_online_enabled: payload.knowledge_online_enabled,
      samples_per_update: requestedSamplesPerUpdate,
      sample_agent_batch_size: normalizedSampleAgentBatchSize(payload.sample_agent_batch_size),
      sample_concurrency: normalizedSampleConcurrency(payload.sample_concurrency),
      budget: {
        max_generations: effectiveBudget.max_generations,
        candidates_per_generation: effectiveBudget.candidates_per_generation,
        max_candidates: effectiveBudget.max_candidates
      },
      seed_policy: payload.fixed_seed ? "fixed" : "generated_and_recorded",
      requested_mode: "autonomous", auto_advance: payload.auto_advance === 0 ? 0 : continuousAutoAdvance ? true : 1
    };
    var reusableRun = state.runs.find(function (run) { return pendingRunMatchesRequest(run, body); });
    if (reusableRun) {
      if (body.auto_advance === 0) { state.autoAdvanceOptOutRunIds[reusableRun.id] = true; }
      else { delete state.autoAdvanceOptOutRunIds[reusableRun.id]; }
      state.workspace = "process";
      return selectRun(reusableRun.id, false).then(function (selected) {
        if (!selected || !state.activeRun) {
          showToast("检测到相同配置的等待运行，但重新读取失败。请刷新后重试。");
          return null;
        }
        if (!pendingRunMatchesRequest(state.activeRun, body)) {
          showToast("相同配置的运行状态已经变化。请再次启动以创建新运行。");
          return null;
        }
        if (body.auto_advance > 0) {
          if (serverAutoProgressEnabled(state.activeRun)) {
            state.createStatus = createStatusForRun(state.activeRun, state.events);
            ensureAutoAdvanceForRun(state.activeRun.id);
            showToast("已切换到相同配置的运行。" + state.createStatus.message);
            return state.activeRun;
          }
          showToast("已有相同配置的运行等待推进，正在继续执行首轮。");
          return Promise.resolve(advanceRun({ automatic: true })).then(function () {
            state.createStatus = createStatusForRun(state.activeRun, state.events);
            ensureAutoAdvanceForRun(state.activeRun && state.activeRun.id);
            return state.activeRun;
          });
        }
        showToast("已有相同配置的运行等待推进，已切换到该运行。");
        return state.activeRun;
      });
    }
    var signature = JSON.stringify(body);
    body.idempotency_key = commandKey("create", signature);
    state.busy = true;
    state.pendingAction = "create";
    state.createStatus = {state: "submitting", runId: null, message: "提交已接收，正在创建运行并连接实时进度。"};
    state.commandError = null;
    // Move to the process workspace before the POST resolves.  Creation is a
    // durable receipt operation; the user should see that receipt and the
    // monitor immediately even when the gateway is busy.
    state.workspace = "process";
    renderAll();
    showToast("提交已接收，正在连接实时进度。" );
    var operation;
    if (state.usingDemo) {
      operation = Promise.resolve({ projection: clone(demoRun) });
    } else {
      operation = request("/runs", { method: "POST", body: body, timeout: evolutionCommandTimeout });
    }
    return operation.then(function (data) {
      clearCommandKey("create");
      var run = normalizeRun(data);
      if (body.auto_advance === 0) { state.autoAdvanceOptOutRunIds[run.id] = true; }
      else { delete state.autoAdvanceOptOutRunIds[run.id]; }
      state.runs = [run].concat(state.runs.filter(function (item) { return item.id !== run.id; }));
      state.activeRun = run;
      state.lastSelectedRunId = run.id;
      state.showAllEvents = false;
      state.candidateSelectionPinned = false;
      syncCandidateSelection(run);
      state.events = state.usingDemo ? clone(demoEvents) : [];
      state.loadState = state.usingDemo ? "demo" : "ready";
      state.lastUpdated = new Date().toISOString();
      state.workspace = "process";
      if (typeof startRunMonitor === "function") { startRunMonitor(run.id); }
      loadSelectedDataset(0);
      loadCandidateSamples(0, {force: true});
      return refreshEventsForRun(run.id).then(function () {
        state.createStatus = createStatusForRun(state.activeRun || run, state.events);
        if (state.createStatus.state === "failed") {
          state.commandError = state.createStatus.message;
        }
        showToast(state.createStatus.message);
        if (body.auto_advance > 0 && !runIsTerminal(state.activeRun || run)) { ensureAutoAdvanceForRun(run.id); }
        return run;
      });
    }).catch(function (error) {
      state.commandError = "创建失败：" + errorMessage(error);
      state.createStatus = {state: "failed", runId: null, message: state.commandError};
      showToast(state.commandError);
      return null;
    }).finally(function () { state.busy = false; state.pendingAction = null; renderAll(); });
  }

  function sameOptionalText(left, right) {
    if (left == null || left === "") { return right == null || right === ""; }
    return String(left) === String(right);
  }

  function normalizedBoolean(value) {
    if (typeof value === "string") {
      var normalized = value.trim().toLowerCase();
      if (["false", "0", "no", "off", "否"].indexOf(normalized) >= 0) { return false; }
      if (["true", "1", "yes", "on", "是"].indexOf(normalized) >= 0) { return true; }
    }
    return Boolean(value);
  }

  function pendingRunMatchesRequest(run, requestBody) {
    // A server-managed continuous run is intentionally reusable while it is
    // active: its worker owns the generation boundary and a second create
    // request must not fork an identical search. Legacy runs are reusable only
    // at a boundary where the browser is allowed to issue the next advance.
    var autoManagedRunning = serverAutoProgressEnabled(run) && String(run.status || "").toLowerCase() === "running";
    if (!autoManagedRunning && !runNeedsAdvanceAction(run, run && run.events)) { return false; }
    var configuration = run.configuration || {};
    var budget = run.budget || {};
    var requestedBudget = requestBody.budget || {};
    var effectiveBudget = normalizedEvolutionBudget(
      requestBody.rounds || requestedBudget.max_generations,
      requestedBudget.candidates_per_generation,
      requestedBudget.max_candidates
    );
    return sameOptionalText(configuration.dataset_id || run.dataset_id, requestBody.dataset_id)
      && sameOptionalText(configuration.episode_id || run.episode_id, requestBody.episode_id)
      && sameOptionalText(configuration.execution_protocol || run.execution_protocol, requestBody.execution_protocol)
      && sameOptionalText(configuration.strategy_model_id || configuration.policy_model_id, requestBody.strategy_model_id)
      && sameOptionalText(configuration.review_model_id || configuration.judge_model_id, requestBody.review_model_id)
      && normalizedBoolean(configuration.autonomous_mode) === normalizedBoolean(requestBody.autonomous_mode)
      && sameOptionalText(configuration.model_workflow, requestBody.model_workflow)
      && normalizedBoolean(configuration.knowledge_online_enabled) === normalizedBoolean(requestBody.knowledge_online_enabled)
      && Number(run.total_generations || budget.max_generations || 0) === effectiveBudget.max_generations
      && Number(run.candidates_per_generation || budget.candidates_per_generation || 0) === effectiveBudget.candidates_per_generation
      && Number(run.max_candidates || budget.max_candidates || 0) === effectiveBudget.max_candidates
      && Number(run.samples_per_update || configuration.samples_per_update || 0) === normalizedSamplesPerUpdate(requestBody.samples_per_update)
      && Number(run.sample_agent_batch_size || configuration.sample_agent_batch_size || 0) === normalizedSampleAgentBatchSize(requestBody.sample_agent_batch_size)
      && Number(run.sample_concurrency || configuration.sample_concurrency || 0) === normalizedSampleConcurrency(requestBody.sample_concurrency)
      && sameOptionalText(run.seed_policy, requestBody.seed_policy);
  }

  function controlRun(action) {
    if (!state.activeRun || state.busy) { return Promise.resolve(false); }
    if (!hasCapability("run.control")) { showToast("当前 DSH 会话未授予运行控制能力。"); return Promise.resolve(false); }
    if (action === "resume" && runHasHardTokenPause(state.activeRun)) {
      showToast("该运行已达到冻结的逐样本智能体 Token 硬预算，不能直接恢复；请创建更高预算的新运行。");
      return Promise.resolve(false);
    }
    var runId = state.activeRun.id;
    if (action === "pause" || action === "cancel") {
      // A user control action is an explicit hand-off from the autonomous
      // scheduler.  Do not let a queued timer issue another advance after the
      // pause/cancel command has been accepted.
      stopAutoAdvance(runId);
    }
    if (action === "resume") {
      state.autoAdvanceBlockedRunId = null;
      state.autoAdvanceError = null;
    }
    var signature = JSON.stringify({ run_id: runId, action: action });
    var body = { action: action, idempotency_key: commandKey("control", signature) };
    state.busy = true;
    state.pendingAction = action;
    state.commandError = null;
    renderAll();
    var operation = state.usingDemo ? Promise.resolve(null) : request("/runs/" + encodeURIComponent(runId) + "/control", { method: "POST", body: body });
    return operation.then(function (data) {
      clearCommandKey("control");
      if (state.usingDemo) {
        state.activeRun.status = action === "pause" ? "paused" : action === "resume" ? "running" : "cancelled";
        state.activeRun.projection_revision += 1;
        state.events.unshift({ id: "演示事件-" + Date.now(), type: "run." + (action === "pause" ? "paused" : action === "resume" ? "resumed" : "cancelled"), occurred_at: new Date().toISOString(), payload: { message: "演示控制命令已应用。" } });
      } else {
        state.activeRun = normalizeRun(data);
        state.runs = state.runs.map(function (run) { return run.id === runId ? state.activeRun : run; });
      }
      state.lastUpdated = new Date().toISOString();
      showToast("运行状态已更新为“" + statusText(state.activeRun.status) + "”。");
      return refreshEventsForRun(runId).then(function () {
        var selectionChanged = reconcileVisibleRunSelection();
        if (selectionChanged && state.activeRun) { return selectRun(state.activeRun.id, false); }
        if (action === "resume" && state.activeRun && state.activeRun.status === "running") {
          ensureAutoAdvanceForRun(runId);
        }
        return true;
      });
    }).catch(function (error) {
      state.commandError = "控制失败：" + errorMessage(error);
      showToast(state.commandError);
      return false;
    }).finally(function () { state.busy = false; state.pendingAction = null; renderAll(); });
  }

  function selectAfterRunRemoval(runId) {
    state.runs = state.runs.filter(function (run) { return run.id !== runId; });
    if (state.activeRun && state.activeRun.id === runId) { state.activeRun = null; }
    if (state.lastSelectedRunId === runId) { state.lastSelectedRunId = null; }
    reconcileVisibleRunSelection();
    state.busy = false;
    state.pendingAction = null;
    renderAll();
    return state.activeRun ? selectRun(state.activeRun.id, false) : Promise.resolve(true);
  }

  function archiveRun() {
    if (!state.activeRun || state.busy) { return Promise.resolve(false); }
    if (!hasCapability("run.archive")) { showToast("当前 DSH 会话未授予归档运行的能力。"); return Promise.resolve(false); }
    if (!runIsTerminal(state.activeRun)) { showToast("请先完成或取消运行，再进行归档。"); return Promise.resolve(false); }
    var runId = state.activeRun.id;
    var restoring = state.activeRun.archived === true;
    state.busy = true;
    state.pendingAction = restoring ? "restore" : "archive";
    state.commandError = null;
    renderAll();
    return request("/runs/" + encodeURIComponent(runId) + (restoring ? "/restore" : "/archive"), { method: "POST", body: {} }).then(function (data) {
      var updated = normalizeRun(data);
      if (restoring) {
        state.archivedRunCount = Math.max(0, state.archivedRunCount - 1);
      } else {
        state.archivedRunCount += 1;
      }
      state.lastUpdated = new Date().toISOString();
      showToast(restoring ? "运行已恢复到运行列表。" : "运行已归档；历史证据仍保留。" );
      if (!restoring && !state.showArchivedRuns) {
        return selectAfterRunRemoval(runId);
      }
      state.activeRun = updated;
      state.runs = state.runs.map(function (run) { return run.id === runId ? updated : run; });
      state.busy = false;
      state.pendingAction = null;
      renderAll();
      return true;
    }).catch(function (error) {
      state.commandError = (restoring ? "恢复失败：" : "归档失败：") + errorMessage(error);
      state.busy = false;
      state.pendingAction = null;
      showToast(state.commandError);
      renderAll();
      return false;
    });
  }

  function deleteRun(confirmation) {
    if (!state.activeRun || state.busy) { return Promise.resolve(false); }
    if (!hasCapability("run.delete")) { showToast("当前 DSH 会话未授予永久删除运行的能力。"); return Promise.resolve(false); }
    if (!runIsTerminal(state.activeRun) || state.activeRun.archived !== true) { showToast("只有已归档的终态运行可以永久删除。"); return Promise.resolve(false); }
    var runId = state.activeRun.id;
    if (confirmation !== runId) { showToast("运行 ID 不匹配，未执行永久删除。"); return Promise.resolve(false); }
    state.busy = true;
    state.pendingAction = "delete";
    state.commandError = null;
    renderAll();
    return request("/runs/" + encodeURIComponent(runId), { method: "DELETE", body: { confirm_run_id: confirmation } }).then(function () {
      state.archivedRunCount = Math.max(0, state.archivedRunCount - 1);
      state.lastUpdated = new Date().toISOString();
      showToast("运行及其专属事件和命令记录已永久删除。" );
      return selectAfterRunRemoval(runId);
    }).catch(function (error) {
      state.commandError = "永久删除失败：" + errorMessage(error);
      state.busy = false;
      state.pendingAction = null;
      showToast(state.commandError);
      renderAll();
      return false;
    });
  }

  function advanceRun(options) {
    var automatic = Boolean(options && options.automatic === true);
    if (!state.activeRun || state.busy) { return Promise.resolve(false); }
    if (!hasCapability("evolution.run.advance")) { showToast("当前 DSH 会话未授予推进进化轮次的能力。"); return Promise.resolve(false); }
    if (state.activeRun.status === "paused") {
      if (!hasCapability("run.control")) { showToast("当前 DSH 会话未授予恢复运行的能力。"); return Promise.resolve(false); }
      var pausedRunId = state.activeRun.id;
      return controlRun("resume").then(function (resumed) {
        if (!resumed || !state.activeRun || state.activeRun.id !== pausedRunId || state.activeRun.status !== "running") { return false; }
        return advanceRun(options);
      });
    }
    if (state.activeRun.status !== "running") { showToast("只有运行中或已暂停的任务可以继续推进。"); return Promise.resolve(false); }
    var runId = state.activeRun.id;
    var commandContextEpoch = state.contextEpoch;
    var signature = JSON.stringify({ run_id: runId, steps: 1 });
    var body = { steps: 1, idempotency_key: commandKey("advance", signature) };
    state.busy = true;
    state.pendingAction = automatic ? "auto-advance" : "advance";
    state.commandError = null;
    renderAll();
    var eventPollTimer = null;
    var progressPollPending = false;
    if (!state.usingDemo) {
      eventPollTimer = window.setInterval(function () {
        if (progressPollPending || commandContextEpoch !== state.contextEpoch) { return; }
        progressPollPending = true;
        refreshProgressForRun(runId).finally(function () { progressPollPending = false; });
      }, 1500);
      refreshProgressForRun(runId);
    }
    var operation = state.usingDemo ? Promise.resolve({ projection: state.activeRun }) : request("/runs/" + encodeURIComponent(runId) + "/advance", { method: "POST", body: body, timeout: evolutionCommandTimeout });
    return operation.then(function (data) {
      if (commandContextEpoch !== state.contextEpoch || !state.activeRun || state.activeRun.id !== runId) { return false; }
      clearCommandKey("advance");
      if (state.usingDemo) {
        state.activeRun.generation = Math.min(state.activeRun.total_generations, state.activeRun.generation + 1);
        state.activeRun.projection_revision += 1;
        state.events.unshift({ id: "演示轮次-" + Date.now(), type: "generation.advanced", occurred_at: new Date().toISOString(), payload: { message: "演示运行已推进一轮。" } });
      } else {
        state.activeRun = normalizeRun(data);
        state.runs = state.runs.map(function (run) { return run.id === runId ? state.activeRun : run; });
      }
      state.lastUpdated = new Date().toISOString();
      if (!automatic) { showToast("下一轮进化已执行。" ); }
      return refreshEventsForRun(runId);
    }).catch(function (error) {
      if (commandContextEpoch !== state.contextEpoch || !state.activeRun || state.activeRun.id !== runId) { return false; }
      state.commandError = "推进失败：" + errorMessage(error);
      showToast(state.commandError);
      return false;
    }).finally(function () {
      if (eventPollTimer) { window.clearInterval(eventPollTimer); }
      if (commandContextEpoch !== state.contextEpoch || !state.activeRun || state.activeRun.id !== runId) { return; }
      state.busy = false;
      if (state.pendingAction === (automatic ? "auto-advance" : "advance")) { state.pendingAction = null; }
      renderAll();
    });
  }

  function parseOverrides(text) {
    var result = {};
    String(text || "").split(/\r?\n/).forEach(function (line) {
      var trimmed = line.trim();
      if (!trimmed) { return; }
      var index = trimmed.indexOf("=");
      if (index < 1) { throw new Error("参数覆盖值必须使用“参数=值”的格式。"); }
      var key = trimmed.slice(0, index).trim();
      var raw = trimmed.slice(index + 1).trim();
      if (!key || !raw) { throw new Error("参数名称和值都不能为空。"); }
      try { result[key] = JSON.parse(raw); } catch (error) { result[key] = raw; }
    });
    return result;
  }

  function submitIntervention(payload) {
    if (!state.activeRun || state.activeRun.status !== "paused") { showToast("请先暂停进化运行。" ); return Promise.resolve(false); }
    if (!hasCapability("intervention.write")) { showToast("当前 DSH 会话未授予提交专家意见与答复的能力。"); return Promise.resolve(false); }
    var runId = state.activeRun.id;
    var signature = JSON.stringify(payload);
    var body = Object.assign({}, payload, { idempotency_key: commandKey("intervention", signature) });
    state.busy = true;
    state.pendingAction = "intervention";
    state.commandError = null;
    renderAll();
    var operation = state.usingDemo ? Promise.resolve(null) : request("/runs/" + encodeURIComponent(runId) + "/interventions", { method: "POST", body: body });
    return operation.then(function (data) {
      clearCommandKey("intervention");
      if (state.usingDemo) {
        state.activeRun.interventions.unshift({ id: "意见-" + Date.now(), kind: payload.kind, message: payload.message, parameter_overrides: payload.parameter_overrides, target_candidate_id: payload.target_candidate_id, created_by: payload.created_by, created_at: new Date().toISOString(), effective_generation: state.activeRun.generation + 1, recorded: true, applied: false, enforced: false, application_status: "recorded" });
        state.activeRun.projection_revision += 1;
      } else if (data) {
        state.activeRun = normalizeRun(data);
        state.runs = state.runs.map(function (run) { return run.id === runId ? state.activeRun : run; });
      }
      state.lastUpdated = new Date().toISOString();
      $("#intervention-form").reset();
      updateInterventionFields();
      showToast("专家主动意见已记录；恢复运行后将在下一轮处理。" );
      return refreshEventsForRun(runId);
    }).catch(function (error) {
      state.commandError = "专家主动意见提交失败：" + errorMessage(error);
      showToast(state.commandError);
      return false;
    }).finally(function () { state.busy = false; state.pendingAction = null; renderAll(); });
  }

  function answerExpertConsultation(consultationId, payload) {
    var run = state.activeRun;
    var id = String(consultationId || "");
    if (!run || !id) { showToast("当前没有可答复的专家咨询。" ); return Promise.resolve(false); }
    if (state.busy) { return Promise.resolve(false); }
    if (!hasCapability("intervention.write")) { showToast("当前 DSH 会话未授予提交专家意见与答复的能力。" ); return Promise.resolve(false); }
    var consultation = (run.expert_consultations || []).find(function (item) { return expertConsultationId(item) === id; });
    if (!consultation || consultation.status !== "pending") { showToast("该咨询已答复或不再存在，请刷新后重试。" ); return Promise.resolve(false); }
    var answer = String(payload && payload.answer || "").trim();
    var answeredBy = String(payload && payload.answered_by || "").trim();
    var selectedOption = String(payload && payload.selected_option || "").trim();
    if (!answer) { showToast("请填写专家答复。" ); return Promise.resolve(false); }
    if (!answeredBy) { showToast("请填写答复人。" ); return Promise.resolve(false); }
    var runId = run.id;
    var auditOnly = expertConsultationRunIsTerminal(run);
    var signature = JSON.stringify({ run_id: runId, consultation_id: id, answer: answer, selected_option: selectedOption, answered_by: answeredBy });
    var body = { answer: answer, answered_by: answeredBy, idempotency_key: commandKey("expert-consultation-answer", signature) };
    if (selectedOption) { body.selected_option = selectedOption; }
    state.busy = true;
    state.pendingAction = "expert-consultation:" + id;
    state.commandError = null;
    renderAll();
    var operation = state.usingDemo ? Promise.resolve(null) : request("/runs/" + encodeURIComponent(runId) + "/expert-consultations/" + encodeURIComponent(id) + "/answer", { method: "POST", body: body });
    return operation.then(function (data) {
      clearCommandKey("expert-consultation-answer");
      if (state.usingDemo) {
        var answeredAt = new Date().toISOString();
        state.activeRun.expert_consultations = state.activeRun.expert_consultations.map(function (item) {
          if (expertConsultationId(item) !== id) { return item; }
          return Object.assign({}, item, { status: "answered", answer: answer, selected_option: selectedOption || null, answered_by: answeredBy, answered_at: answeredAt, effective_generation: auditOnly ? null : state.activeRun.generation + 1, applied_generation: null });
        });
        state.activeRun.projection_revision += 1;
        state.events.unshift({ id: "演示咨询事件-" + Date.now(), type: "expert_consultation.answered", occurred_at: answeredAt, payload: { consultation_id: id, audit_only: auditOnly, message: auditOnly ? "迟到专家答复已归档，不会改写运行结果。" : "专家答复已记录，将在后续轮次使用。" } });
      } else {
        var projection = data && (data.projection || data.run_projection) || data;
        if (projection && (projection.run_id || projection.id === runId || Array.isArray(projection.expert_consultations))) {
          state.activeRun = normalizeRun(projection);
        } else {
          state.activeRun.expert_consultations = state.activeRun.expert_consultations.map(function (item) {
            return expertConsultationId(item) === id ? Object.assign({}, item, { status: "answered", answer: answer, selected_option: selectedOption || null, answered_by: answeredBy, answered_at: new Date().toISOString(), effective_generation: auditOnly ? null : state.activeRun.generation + 1, applied_generation: null }) : item;
          });
        }
        state.runs = state.runs.map(function (item) { return item.id === runId ? state.activeRun : item; });
      }
      clearExpertConsultationDraft(runId, id);
      state.lastUpdated = new Date().toISOString();
      showToast(auditOnly ? "迟到专家答复已归档；不会改写已完成的进化结果。" : "专家答复已记录；不会改写历史候选，将从后续轮次开始使用。" );
      return refreshEventsForRun(runId);
    }).catch(function (error) {
      state.commandError = "专家答复提交失败：" + errorMessage(error);
      showToast(state.commandError + " 已保留当前草稿，可稍后重试。");
      return false;
    }).finally(function () { state.busy = false; state.pendingAction = null; renderAll(); });
  }

  function refreshAll(options) {
    var refreshDataset = Boolean(options && options.refreshDataset);
    if (state.usingDemo) {
      if (refreshDataset) { state.datasetContext = activeRunDatasetContext() || selectedDatasetContext(); state.datasetPage = demoDatasetPage(state.pageOffset, state.datasetPartition); state.datasetError = null; }
      state.lastUpdated = new Date().toISOString(); renderAll(); return Promise.resolve(true);
    }
    if (state.refreshing) { return Promise.resolve(false); }
    if (!state.activeRun) { return connectAndLoad(); }
    var runId = state.activeRun.id;
    var viewEpoch = state.viewEpoch;
    var requestId = state.runReadRequest + 1;
    state.runReadRequest = requestId;
    state.refreshing = true;
    renderAll();
    return Promise.all([request("/catalog", { timeout: dataRequestTimeout }), request(runsListPath()), request("/runs/" + encodeURIComponent(runId)), request("/runs/" + encodeURIComponent(runId) + "/events")]).then(function (results) {
      if (requestId !== state.runReadRequest || viewEpoch !== state.viewEpoch || !state.activeRun || state.activeRun.id !== runId) { return false; }
      state.catalog = normalizeCatalog(results[0]);
      var previousRun = state.activeRun;
      var incomingRun = normalizeRun(results[2]);
      if (!state.activeRun || incomingRun.projection_revision >= state.activeRun.projection_revision) {
        state.activeRun = incomingRun;
        state.events = normalizeEvents(results[3]);
      }
      state.runs = listFrom(results[1], "runs").map(normalizeRun).map(function (run) { return run.id === runId ? state.activeRun : run; }).sort(function (left, right) {
        var leftTime = Date.parse(left.updated_at || left.created_at || "") || 0;
        var rightTime = Date.parse(right.updated_at || right.created_at || "") || 0;
        return rightTime - leftTime;
      });
      state.archivedRunCount = Math.max(0, Number(results[1] && results[1].archived_count || 0));
      if (!state.runs.some(function (run) { return run.id === runId; })) { state.runs.unshift(state.activeRun); }
      var selectionChanged = reconcileVisibleRunSelection();
      var candidateSelectionChanged = state.activeRun ? syncCandidateSelection(state.activeRun) : false;
      state.lastUpdated = new Date().toISOString();
      state.loadState = state.activeRun ? "ready" : "empty";
      state.lastError = null;
      if (String(state.activeRun && state.activeRun.status || "").toLowerCase() !== "failed") {
        state.commandError = null;
      }
      observeRunStatus(previousRun, state.activeRun, state.events);
      setConnection("online", state.hostContextReceived ? "DSH 宿主已连接" : "本地服务已连接");
      populateCatalogControls();
      renderAll();
      if (state.activeRun && typeof startRunMonitor === "function") { startRunMonitor(state.activeRun.id); }
      if (candidateSelectionChanged) { loadCandidateSamples(0, {force: true, silent: true}); }
      else { refreshCandidateSamples({silent: true}); }
      if (state.activeRun && typeof ensureAutoAdvanceForRun === "function") {
        ensureAutoAdvanceForRun(state.activeRun.id);
      }
      if (selectionChanged) {
        return state.activeRun ? selectRun(state.activeRun.id, false) : true;
      }
      return refreshDataset ? loadSelectedDataset(state.pageOffset).then(function () { return true; }) : true;
    }).catch(function (error) {
      if (requestId !== state.runReadRequest || viewEpoch !== state.viewEpoch) { return false; }
      state.loadState = "stale";
      state.lastError = errorMessage(error);
      setConnection("offline", "连接已中断");
      renderAll();
      return false;
    }).finally(function () { state.refreshing = false; renderAll(); });
  }
