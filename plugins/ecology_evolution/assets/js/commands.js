"use strict";

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

  function strictInteger(value, label, minimum, maximum) {
    if (typeof value === "boolean" || value === "" || value == null || !Number.isInteger(Number(value))) {
      throw new Error(label + "必须是整数");
    }
    var parsed = Number(value);
    if (minimum != null && parsed < minimum) { throw new Error(label + "不得小于 " + minimum); }
    if (maximum != null && parsed > maximum) { throw new Error(label + "不得大于 " + maximum); }
    return parsed;
  }

  function normalizedOptimizationSchedule(values) {
    var input = values || {};
    var formal = strictInteger(input.formal_origin_count == null ? 500 : input.formal_origin_count, "每个入围候选更新时点数", 1);
    var batch = strictInteger(input.local_batch_origin_count == null ? 50 : input.local_batch_origin_count, "局部 batch 时点数", 1);
    var edits = strictInteger(input.max_local_edits_per_batch == null ? 2 : input.max_local_edits_per_batch, "每批最大局部改动数", 0, 5);
    var holdout = strictInteger(input.selection_holdout_origin_count == null ? 169 : input.selection_holdout_origin_count, "轮末比较时点数", 169);
    if (formal % batch !== 0) { throw new Error("局部 batch 必须整除每个入围候选的更新时点数"); }
    return {
      schema_version: "ecologyrsi-dsh.top2-adaptive-epoch-schedule/2",
      screening_origin_count: 64,
      finalist_count: 2,
      formal_origin_count_per_finalist: formal,
      local_batch_origin_count: batch,
      max_local_edits_per_batch: edits,
      selection_holdout_origin_count: holdout,
      local_evaluation_mode: "paired_champion_challenger"
    };
  }

  function optimizationScheduleFromControls() {
    return normalizedOptimizationSchedule({
      formal_origin_count: $("#formal-origin-count").value,
      local_batch_origin_count: $("#local-batch-origin-count").value,
      max_local_edits_per_batch: $("#max-local-edits-per-batch").value,
      selection_holdout_origin_count: $("#selection-holdout-origin-count").value
    });
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
    var optimizationSchedule;
    var candidateConcurrency;
    var sampleAgentBatchSize;
    var sampleConcurrency;
    try {
      optimizationSchedule = normalizedOptimizationSchedule(payload);
      candidateConcurrency = strictInteger(payload.candidate_concurrency == null ? 4 : payload.candidate_concurrency, "候选并发数", 1, 8);
      sampleAgentBatchSize = strictInteger(payload.sample_agent_batch_size == null ? 9 : payload.sample_agent_batch_size, "单时点向量单元容量", 9, 9);
      sampleConcurrency = strictInteger(payload.sample_concurrency == null ? 64 : payload.sample_concurrency, "逐样本并发请求数", 1, 128);
    } catch (error) {
      showToast(error.message);
      return Promise.resolve(null);
    }
    var effectiveBudget = normalizedEvolutionBudget(
      payload.rounds || payload.max_generations,
      4,
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
      optimization_protocol: "top2_adaptive_epoch@1",
      optimization_schedule: optimizationSchedule,
      strategy_model_id: payload.strategy_model_id || payload.policy_model_id,
      review_model_id: payload.review_model_id || payload.judge_model_id,
      autonomous_mode: payload.autonomous_mode === true,
      rounds: effectiveBudget.max_generations,
      model_workflow: payload.model_workflow || "research_compile_evolve@1",
      knowledge_online_enabled: payload.knowledge_online_enabled,
      candidate_concurrency: candidateConcurrency,
      sample_agent_batch_size: sampleAgentBatchSize,
      sample_concurrency: sampleConcurrency,
      budget: {
        max_generations: effectiveBudget.max_generations,
        candidates_per_generation: effectiveBudget.candidates_per_generation,
        max_candidates: effectiveBudget.max_candidates
      },
      seed_policy: payload.fixed_seed ? "fixed" : "generated_and_recorded",
      requested_mode: "autonomous", auto_advance: payload.auto_advance === 0 ? 0 : continuousAutoAdvance ? true : 1
    };
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
      state.runs = [run].concat(state.runs.filter(function (item) { return item.id !== run.id; }));
      state.activeRun = run;
      state.lastSelectedRunId = run.id;
      state.showAllEvents = false;
      state.candidateSelectionPinned = false;
      syncCandidateSelection(run);
      resetEventStream(run.id, state.usingDemo ? clone(demoEvents) : []);
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

  function pollCommand(commandId, runId, attempts) {
    if (!commandId || attempts <= 0) { return Promise.resolve(null); }
    return request("/commands/" + encodeURIComponent(commandId), { timeout: 5000 }).then(function (receipt) {
      if (receipt && receipt.status === "completed" && receipt.response) { return receipt.response; }
      return new Promise(function (resolve) { window.setTimeout(resolve, 500); }).then(function () {
        return pollCommand(commandId, runId, attempts - 1);
      });
    }).catch(function () { return null; });
  }

  function controlExpectedStatus(action) {
    return {
      start: "running",
      pause: "paused",
      resume: "running",
      cancel: "cancelled",
      complete: "completed"
    }[action] || null;
  }

  function controlSuccessMessage(action) {
    return {
      start: "启动成功。",
      pause: "暂停成功。",
      resume: "恢复成功。",
      cancel: "运行已停止。",
      complete: "运行已完成。"
    }[action] || "运行状态已更新。";
  }

  function mergeControlResponse(runId, data) {
    var envelope = data && data.response || data;
    var projection = envelope && (envelope.projection || envelope.run_projection) || envelope || {};
    var responseRunId = projection.run_id || projection.id;
    if (!responseRunId) { return null; }
    if (String(responseRunId) !== String(runId)) {
      throw new Error("运行控制响应返回了其他运行的数据。");
    }
    if (!state.activeRun || String(state.activeRun.id) !== String(runId)) { return null; }
    var refreshed = mergeRunProjection(state.activeRun, projection);
    if (refreshed.projection_revision < state.activeRun.projection_revision) {
      return state.activeRun;
    }
    state.activeRun = refreshed;
    state.runs = state.runs.map(function (run) { return run.id === runId ? refreshed : run; });
    return refreshed;
  }

  function controlStatusConfirmed(action, run) {
    var expected = controlExpectedStatus(action);
    return Boolean(expected && run && String(run.status).toLowerCase() === expected);
  }

  function controlRequestTimedOut(error) {
    return Boolean(error && (error.name === "AbortError" || /请求超时|timed out|timeout/i.test(String(error.message || ""))));
  }

  function setControlNotice(runId, action, mode) {
    var label = {start: "启动", pause: "暂停", resume: "恢复", cancel: "停止", complete: "结束"}[action] || "操作";
    var message = mode === "verifying"
      ? label + "请求等待较久，正在核对后台状态，请勿重复操作。"
      : mode === "pending"
        ? label + "请求已接收，等待后台确认；页面会自动更新。"
        : label + "结果尚未确认，不代表操作失败。请刷新状态；再次操作会使用同一请求标识。";
    state.controlNotice = {runId: runId, action: action, mode: mode, message: message};
    return message;
  }

  function reconcileTimedOutControl(runId, action, body) {
    var path = "/runs/" + encodeURIComponent(runId) + "/control";
    var retryAccepted = false;
    return request(path, { method: "POST", body: body, timeout: 5000 }).then(function (data) {
      retryAccepted = Boolean(data && (
        data.accepted === true
        || data.command_status === "pending"
        || data.status === "pending"
        || data.command_status === "completed"
        || data.status === "completed"
      ));
      var refreshed = mergeControlResponse(runId, data);
      return controlStatusConfirmed(action, refreshed)
        ? { confirmed: true, accepted: true }
        : null;
    }).catch(function () { return null; }).then(function (result) {
      if (result && result.confirmed) { return result; }
      return request("/runs/" + encodeURIComponent(runId) + "?view=monitor", { timeout: dataRequestTimeout }).then(function (monitor) {
        var refreshed = mergeControlResponse(runId, monitor);
        return {
          confirmed: controlStatusConfirmed(action, refreshed),
          accepted: retryAccepted
        };
      }).catch(function () {
        return { confirmed: false, accepted: retryAccepted };
      });
    });
  }

  function finishControlUi(action, runId, message) {
    state.lastUpdated = new Date().toISOString();
    showToast(message);
    return refreshEventsForRun(runId).then(function () {
      var selectionChanged = reconcileVisibleRunSelection();
      if (selectionChanged && state.activeRun) { return selectRun(state.activeRun.id, false); }
      if (action === "resume" && state.activeRun && state.activeRun.status === "running") {
      }
      return true;
    });
  }

  function controlRun(action) {
    if (!state.activeRun || state.busy) { return Promise.resolve(false); }
    if (!hasCapability("run.control")) { showToast("当前 DSH 会话未授予运行控制能力。"); return Promise.resolve(false); }
    if (action === "resume" && runHasHardTokenPause(state.activeRun)) {
      showToast("该运行已达到冻结的" + tokenBudgetSubjectText(state.activeRun) + " Token 硬预算，不能直接恢复；请创建更高预算的新运行。");
      return Promise.resolve(false);
    }
    var runId = state.activeRun.id;
    if (action === "pause" || action === "cancel") {
      // A user control action is an explicit hand-off from the autonomous
      // scheduler.  Do not let a queued timer issue another advance after the
      // pause/cancel command has been accepted.
    }
    if (action === "resume") {
    }
    var signature = JSON.stringify({ run_id: runId, action: action });
    var body = { action: action, idempotency_key: commandKey("control", signature) };
    state.busy = true;
    state.pendingAction = action;
    state.commandError = null;
    state.controlNotice = null;
    renderAll();
    var operation = state.usingDemo ? Promise.resolve(null) : request("/runs/" + encodeURIComponent(runId) + "/control", { method: "POST", body: body, timeout: 30000 });
    return operation.then(function (data) {
      if (state.usingDemo) {
        state.activeRun.status = action === "pause" ? "paused" : action === "resume" ? "running" : "cancelled";
        state.activeRun.projection_revision += 1;
        state.events.unshift({ id: "演示事件-" + Date.now(), type: "run." + (action === "pause" ? "paused" : action === "resume" ? "resumed" : "cancelled"), occurred_at: new Date().toISOString(), payload: { message: "演示控制命令已应用。" } });
      } else {
        if (!mergeControlResponse(runId, data)) {
          throw new Error("运行控制响应缺少可核验状态。");
        }
        var commandStatus = data && (data.command_status || data.status);
        if (commandStatus === "pending" && data.command_id) {
          // The Host boundary is already durable; continue observing the
          // remote DSH drain without keeping the button request open.
          pollCommand(data.command_id, runId, 60).then(function (completed) {
            if (!completed) { return; }
            var refreshed = mergeControlResponse(runId, completed);
            if (!refreshed) { return; }
            state.lastUpdated = new Date().toISOString();
            renderAll();
          });
        }
      }
      state.commandError = null;
      var confirmed = controlStatusConfirmed(action, state.activeRun);
      if (confirmed) { clearCommandKey("control"); }
      return finishControlUi(action, runId, confirmed
        ? controlSuccessMessage(action)
        : setControlNotice(runId, action, "pending"));
    }).catch(function (error) {
      if (!controlRequestTimedOut(error) || state.usingDemo) { throw error; }
      setControlNotice(runId, action, "verifying");
      renderAll();
      return reconcileTimedOutControl(runId, action, body).then(function (result) {
        state.commandError = null;
        if (!result.confirmed && !result.accepted) {
          showToast(setControlNotice(runId, action, "unknown"));
          return false;
        }
        if (result.confirmed) {
          clearCommandKey("control");
          state.controlNotice = null;
        }
        return finishControlUi(
          action,
          runId,
          result.confirmed
            ? controlSuccessMessage(action)
            : setControlNotice(runId, action, "pending")
        );
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
    return request("/runs/" + encodeURIComponent(runId) + (restoring ? "/restore" : "/archive"), { method: "POST", body: {}, timeout: 30000 }).then(function (data) {
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
    if (runHasRetryCircuitPause(state.activeRun)) {
      if (!hasCapability("run.control")) { showToast("当前 DSH 会话未授予恢复运行的能力。"); return Promise.resolve(false); }
      // The resume endpoint re-queues the preserved checkpoint.  Issuing a
      // second /advance here would race that worker and could duplicate work.
      return controlRun("resume");
    }
    if (!hasCapability("evolution.run.advance")) { showToast("当前 DSH 会话未授予推进进化轮次的能力。"); return Promise.resolve(false); }
    if (state.activeRun.status === "paused") {
      if (!hasCapability("run.control")) { showToast("当前 DSH 会话未授予恢复运行的能力。"); return Promise.resolve(false); }
      var pausedRunId = state.activeRun.id;
      var serverOwnedAutoProgress = runHasContinuousAutoProgress(state.activeRun);
      return controlRun("resume").then(function (resumed) {
        if (!resumed || !state.activeRun || state.activeRun.id !== pausedRunId || state.activeRun.status !== "running") { return false; }
        // Resuming a durable continuous run re-queues its preserved checkpoint
        // in the service. A second browser /advance races that worker and
        // produces a misleading conflict toast even though recovery succeeded.
        if (serverOwnedAutoProgress) { return true; }
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
    return Promise.all([request("/catalog", { timeout: dataRequestTimeout }), request(runsListPath()), request("/runs/" + encodeURIComponent(runId)), request(eventRequestPath(runId, false)).catch(function () { return null; })]).then(function (results) {
      if (requestId !== state.runReadRequest || viewEpoch !== state.viewEpoch || !state.activeRun || state.activeRun.id !== runId) { return false; }
      state.catalog = normalizeCatalog(results[0]);
      var previousRun = state.activeRun;
      var incomingRun = normalizeRun(results[2]);
      if (!state.activeRun || incomingRun.projection_revision >= state.activeRun.projection_revision) {
        state.activeRun = incomingRun;
        if (results[3]) { mergeEventStream(runId, results[3]); }
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
      state.structureHydrationStale = false;
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
