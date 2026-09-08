"use strict";

  function loadSelectedDataset(offset) {
    if (!state.usingDemo && state.workspace !== "training") { return Promise.resolve(false); }
    var context = trainingDatasetContext();
    if (!context) {
      state.datasetPage = null; state.datasetContext = null; state.datasetError = null; renderTraining(); return Promise.resolve(false);
    }
    var datasetId = context.dataset_id;
    var episodeId = context.episode_id;
    if (!hasCapability("training.data.read")) {
      state.datasetPage = null;
      state.datasetContext = context;
      state.datasetError = null;
      state.datasetLoading = false;
      renderTraining();
      showToast("当前 DSH 会话未授予训练数据读取能力。");
      return Promise.resolve(false);
    }
    var requestId = state.datasetRequest + 1;
    state.datasetRequest = requestId;
    state.pageOffset = Math.max(0, Number(offset || 0));
    state.datasetLoading = true;
    state.datasetError = null;
    state.datasetContext = context;
    renderTraining();
    var partition = state.datasetPartition === "training_feedback" ? "training_feedback" : "training_fit";
    var sampleQuery = new URLSearchParams({ partition: partition, episode_id: episodeId, offset: String(state.pageOffset), limit: String(state.pageLimit) });
    if (context.source === "active_run" && context.dataset_digest) { sampleQuery.set("expected_dataset_digest", context.dataset_digest); }
    if (context.source === "active_run" && context.split_manifest_digest) { sampleQuery.set("expected_split_manifest_digest", context.split_manifest_digest); }
    if (context.data_protocol_digest) { sampleQuery.set("expected_data_protocol_digest", context.data_protocol_digest); }
    var operation = state.usingDemo ? Promise.resolve(demoDatasetPage(state.pageOffset, partition)) : Promise.all([
      request("/datasets/" + encodeURIComponent(datasetId), { timeout: dataRequestTimeout }),
      request("/datasets/" + encodeURIComponent(datasetId) + "/samples?" + sampleQuery.toString(), { timeout: dataRequestTimeout })
    ]).then(function (results) {
      var description = results[0] || {};
      var samples = results[1] || {};
      var samplePage = samples.page || {
        rows: samples.rows || [], offset: samples.offset || 0,
        limit: samples.limit || state.pageLimit, total: samples.total || 0
      };
      return {
        dataset: Object.assign({}, description.dataset || {}, samples.dataset || {}), descriptor: description.descriptor || description.dataset || {},
        readiness: description.readiness || {}, source_integrity: description.source_integrity || description.readiness && description.readiness.source_integrity || null, profile: description.profile || {},
        features: description.features || samples.features || description.schema || samples.schema || [],
        partitions: samples.partitions || description.partitions || description.profile && description.profile.partitions || {},
        dataset_digest: samples.dataset_digest_sha256 || description.dataset_digest_sha256 || null,
        partition: samples.partition || partition,
        visible_partitions: description.visible_partitions || [],
        restricted_partitions: description.restricted_partitions || [],
        page: samplePage
      };
    });
    return operation.then(function (data) {
      if (requestId !== state.datasetRequest) { return false; }
      if (!sameDatasetContext(context, trainingDatasetContext())) { return false; }
      state.datasetPage = data || null;
      state.datasetContext = context;
      state.datasetError = null;
      return true;
    }).catch(function (error) {
      if (requestId !== state.datasetRequest) { return false; }
      state.datasetPage = null;
      state.datasetContext = context;
      state.datasetError = errorMessage(error);
      showToast("训练数据读取失败，可在页面中重试。");
      return false;
    }).finally(function () {
      if (requestId === state.datasetRequest) { state.datasetLoading = false; renderTraining(); }
    });
  }

  function candidateSampleRowsFrom(value) {
    if (Array.isArray(value)) { return value; }
    if (!value || typeof value !== "object") { return []; }
    if (Array.isArray(value.rows)) { return value.rows; }
    if (Array.isArray(value.samples)) { return value.samples; }
    return [];
  }

  function candidateEmbeddedSampleSource(candidate, run) {
    var metrics = candidate && candidate.metrics && typeof candidate.metrics === "object" ? candidate.metrics : {};
    var execution = candidate && candidate.execution && typeof candidate.execution === "object" ? candidate.execution : {};
    var sources = [
      execution.inference_trace, execution.prediction_trace,
      candidate && candidate.inference_trace, candidate && candidate.prediction_trace,
      metrics.inference_trace, metrics.prediction_preview, metrics.predictions,
      run && run.inference_trace
    ];
    for (var index = 0; index < sources.length; index += 1) {
      var rows = candidateSampleRowsFrom(sources[index]);
      if (rows.length) { return {source: sources[index], rows: rows}; }
    }
    return {source: null, rows: []};
  }

  function candidateSampleFirstValue(row, keys) {
    for (var index = 0; index < keys.length; index += 1) {
      if (row[keys[index]] != null) { return row[keys[index]]; }
    }
    return null;
  }

  function normalizeCandidateSampleRow(value, index) {
    var row = value && typeof value === "object" && !Array.isArray(value) ? Object.assign({}, value) : {};
    var failure = row.failure && typeof row.failure === "object" ? row.failure : {};
    var sampleStatus = candidateSampleFirstValue(row, ["sample_execution_status", "execution_status", "status"]);
    var observed = candidateSampleFirstValue(row, ["observed", "actual", "actual_value", "label"]);
    var predicted = candidateSampleFirstValue(row, ["predicted", "prediction", "predicted_value"]);
    var baseline = candidateSampleFirstValue(row, ["baseline", "baseline_value", "baseline_prediction"]);
    var reward = candidateSampleFirstValue(row, ["reward", "sample_reward", "reward_value", "score"]);
    var attempts = candidateSampleFirstValue(row, ["attempts", "sample_execution_attempts", "attempt_count"]);
    var retryCount = candidateSampleFirstValue(row, ["retry_count", "sample_execution_retry_count", "retries"]);
    var predictionSource = candidateSampleFirstValue(row, ["prediction_source", "source"]);
    var scoringFallback = row.scoring_fallback || (String(predictionSource || "").toLowerCase() === "scoring_fallback" ? "scoring_fallback" : null);
    var failureMessage = candidateSampleFirstValue(row, ["failure_message", "public_error", "failure_reason"])
      || failure.message || failure.public_error || failure.failure_code || failure.error_type || null;
    var failedStatus = ["failed", "error", "rejected", "timeout", "aborted", "cancelled"].indexOf(String(sampleStatus || "").toLowerCase()) >= 0 || Boolean(scoringFallback);
    if (!failureMessage && failedStatus && typeof row.error === "string" && row.error.trim() && !Number.isFinite(Number(row.error))) {
      failureMessage = row.error;
    }
    if (failedStatus) {
      predicted = null;
      reward = null;
      predictionSource = "failed_no_model_prediction";
    } else if (reward == null && [observed, predicted, baseline].every(function (item) { return item != null && item !== "" && Number.isFinite(Number(item)); })) {
      reward = Math.abs(Number(baseline) - Number(observed)) - Math.abs(Number(predicted) - Number(observed));
    }
    return Object.assign({}, row, {
      sample_id: candidateSampleFirstValue(row, ["sample_id", "id"]) || "sample:" + String(index + 1),
      origin_timestamp: candidateSampleFirstValue(row, ["origin_timestamp", "forecast_timestamp", "input_timestamp"]),
      target_timestamp: candidateSampleFirstValue(row, ["target_timestamp", "timestamp", "time"]),
      target: candidateSampleFirstValue(row, ["target", "target_name", "variable"]),
      horizon_hours: candidateSampleFirstValue(row, ["horizon_hours", "horizon", "lead_hours"]),
      observed: observed,
      predicted: predicted,
      baseline: baseline,
      reward: reward,
      attempts: attempts,
      retry_count: retryCount,
      prediction_source: predictionSource,
      scoring_fallback: scoringFallback,
      scoring_penalty_applied: row.scoring_penalty_applied === true || Boolean(scoringFallback),
      model_prediction_available: failedStatus ? false : row.model_prediction_available !== false && predicted != null && predicted !== "",
      sample_status: sampleStatus,
      failure_message: failureMessage
    });
  }

  function normalizeCandidateSamplePage(payload, runId, candidateId, requestedOffset, requestedLimit) {
    if (!payload || payload.schema_version !== "ecologyrsi-dsh.browser-sample-results/1" || !Array.isArray(payload.rows)) {
      throw new Error("逐样本接口响应格式无效。");
    }
    if (payload.run_id !== runId || payload.candidate_id !== candidateId) {
      throw new Error("逐样本结果身份与当前候选不一致。");
    }
    if (payload.offset !== requestedOffset || payload.limit !== requestedLimit
        || !Number.isSafeInteger(payload.total) || payload.total < 0 || payload.rows.length > requestedLimit
        || (payload.rows.length > 0 && payload.total < payload.offset + payload.rows.length)) {
      throw new Error("逐样本结果分页与请求不一致。");
    }
    return Object.assign({}, payload, {rows: payload.rows.map(normalizeCandidateSampleRow), source: "api", truncated: false});
  }

  function demoCandidateSamplePage(run, candidate, offset, limit) {
    var embedded = candidateEmbeddedSampleSource(candidate, run);
    var source = embedded.source && typeof embedded.source === "object" && !Array.isArray(embedded.source) ? embedded.source : {};
    var total = Number(source.sample_count != null ? source.sample_count : source.total);
    if (!Number.isFinite(total) || total < embedded.rows.length) { total = embedded.rows.length; }
    var start = Math.max(0, Number(offset) || 0);
    var pageLimit = Math.max(1, Number(limit) || 25);
    var rows = embedded.rows.slice(start, start + pageLimit).map(normalizeCandidateSampleRow);
    var candidateStatus = String(candidate && candidate.status || "").toLowerCase();
    return {
      schema_version: "ecologyrsi-dsh.demo-sample-page/1",
      run_id: run && run.id,
      candidate_id: candidate && (candidate.id || candidate.candidate_id),
      rows: rows,
      offset: start,
      limit: pageLimit,
      total: total,
      has_more: start + rows.length < embedded.rows.length,
      next_offset: start + rows.length < embedded.rows.length ? start + rows.length : null,
      complete: ["pending", "spawned", "evaluating", "running"].indexOf(candidateStatus) < 0,
      revision: run && run.projection_revision,
      updated_at: run && run.updated_at,
      source: "projection_preview",
      truncated: source.truncated === true || total > embedded.rows.length
    };
  }

  function resetCandidateSamples(runId, candidateId) {
    state.candidateSampleRequest += 1;
    state.candidateSamplePage = null;
    state.candidateSampleLoading = false;
    state.candidateSampleRefreshing = false;
    state.candidateSampleError = null;
    state.candidateSampleUnavailable = false;
    state.candidateSamplePermissionDenied = false;
    state.candidateSampleOffset = 0;
    state.candidateSampleRetryOffset = null;
    state.candidateSampleLastRequestedAt = 0;
    state.candidateSampleSelection = {run_id: runId || null, candidate_id: candidateId || null};
  }

  function sampleCandidates(run) {
    return state.workspace === "process" ? processCandidates(run) : (run && run.candidates || []);
  }

  function activeCandidateIdForSamples(run) {
    var activeId = run && run.execution_progress && run.execution_progress.current_candidate_id;
    if (!activeId) { return null; }
    var activeCandidate = sampleCandidates(run).find(function (candidate) {
      return String(candidate.id || candidate.candidate_id || "") === String(activeId);
    });
    return activeCandidate ? activeCandidate.id || activeCandidate.candidate_id : null;
  }

  function latestCandidateIdForSamples(run) {
    var candidates = sampleCandidates(run);
    var rounds = run && Array.isArray(run.rounds) ? run.rounds.slice() : [];
    rounds.sort(function (left, right) { return Number(right.generation || 0) - Number(left.generation || 0); });
    for (var roundIndex = 0; roundIndex < rounds.length; roundIndex += 1) {
      var rows = Array.isArray(rounds[roundIndex].candidates) ? rounds[roundIndex].candidates.slice() : [];
      rows.sort(function (left, right) { return Number(right.slot_index || 0) - Number(left.slot_index || 0); });
      for (var rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
        var rowId = rows[rowIndex].candidate_id || rows[rowIndex].id;
        var matched = candidates.find(function (candidate) {
          return String(candidate.id || candidate.candidate_id || "") === String(rowId || "");
        });
        if (matched) { return matched.id || matched.candidate_id; }
      }
    }
    var latest = candidates.slice().sort(function (left, right) {
      return Number(right.generation || 0) - Number(left.generation || 0)
        || Number(right.slot_index || 0) - Number(left.slot_index || 0);
    })[0];
    return latest ? latest.id || latest.candidate_id : null;
  }

  function syncCandidateSelection(run) {
    var candidates = sampleCandidates(run);
    var currentCandidate = candidates.find(function (candidate) {
      return String(candidate.id || candidate.candidate_id || "") === String(state.selectedCandidateId || "");
    });
    var nextCandidateId = !state.candidateSelectionPinned
      ? activeCandidateIdForSamples(run) || latestCandidateIdForSamples(run)
      : null;
    if (!nextCandidateId && currentCandidate) { nextCandidateId = currentCandidate.id || currentCandidate.candidate_id; }
    if (!nextCandidateId && candidates.length) {
      var fallbackId = run.best_candidate_id || run.best_observed_candidate_id;
      var fallbackCandidate = fallbackId && candidates.find(function (candidate) {
        return String(candidate.id || candidate.candidate_id || "") === String(fallbackId);
      });
      nextCandidateId = fallbackCandidate
        ? fallbackCandidate.id || fallbackCandidate.candidate_id
        : candidates[0].id || candidates[0].candidate_id || null;
    }
    var selectionChanged = String(state.selectedCandidateId || "") !== String(nextCandidateId || "");
    state.selectedCandidateId = nextCandidateId || null;
    if (!candidateSampleSelectionMatches(run && run.id, state.selectedCandidateId)) {
      resetCandidateSamples(run && run.id, state.selectedCandidateId);
    }
    return selectionChanged;
  }

  function selectedCandidateForSamples() {
    var run = state.activeRun;
    if (!run || !state.selectedCandidateId) { return null; }
    return sampleCandidates(run).find(function (candidate) {
      return String(candidate.id || candidate.candidate_id) === String(state.selectedCandidateId);
    }) || null;
  }

  function candidateSampleSelectionMatches(runId, candidateId) {
    var selection = state.candidateSampleSelection || {};
    return String(selection.run_id || "") === String(runId || "") && String(selection.candidate_id || "") === String(candidateId || "");
  }

  // Poll the durable projection while a run is active.  This is deliberately
  // a single-flight, read-only loop: the server-side generation worker remains
  // the only writer, while the browser can show stage and sample heartbeats
  // without waiting for the 10-second run-history refresh.
  var runMonitorBaseDelayMs = 15000;
  var runMonitorEvaluationDelayMs = 10000;
  var runMonitorMaxDelayMs = 30000;

  function clearRunMonitorTimer() {
    if (state.runMonitorTimer != null) {
      window.clearTimeout(state.runMonitorTimer);
      state.runMonitorTimer = null;
    }
  }

  function stopRunMonitor(runId) {
    if (runId && state.runMonitorRunId && String(runId) !== String(state.runMonitorRunId)) { return; }
    clearRunMonitorTimer();
    state.runMonitorRunId = null;
    state.runMonitorContextEpoch = null;
    state.runMonitorInFlight = false;
    state.runMonitorRetry = 0;
  }

  function runMonitorDelay(run) {
    var progress = run && run.execution_progress && typeof run.execution_progress === "object" ? run.execution_progress : {};
    var retryWait = progress.retry_wait && typeof progress.retry_wait === "object" ? progress.retry_wait : null;
    if (retryWait && retryWait.retry_at) {
      var retryAt = Date.parse(retryWait.retry_at);
      if (Number.isFinite(retryAt)) {
        // Keep the monitor alive during a provider cooldown without polling at
        // evaluation cadence; wake sooner as the durable retry deadline nears.
        var remaining = Math.max(0, retryAt - Date.now());
        return Math.min(runMonitorMaxDelayMs, Math.max(runMonitorBaseDelayMs, remaining));
      }
    }
    var phase = String(progress.phase || progress.current_stage || "").toLowerCase();
    return phase === "evaluation" || phase === "training" ? runMonitorEvaluationDelayMs : runMonitorBaseDelayMs;
  }

  function queueRunMonitor(runId, delay) {
    if (!runId || state.runMonitorRunId !== runId || state.runMonitorContextEpoch !== state.contextEpoch || state.usingDemo) { return false; }
    clearRunMonitorTimer();
    state.runMonitorTimer = window.setTimeout(function () {
      state.runMonitorTimer = null;
      runMonitorTick(runId);
    }, Math.max(0, Number(delay) || 0));
    return true;
  }

  function startRunMonitor(runId) {
    var run = state.activeRun;
    var runStatus = String(run && run.status || "").toLowerCase();
    if (state.usingDemo || !run || !runId || String(run.id) !== String(runId) || runStatus !== "running" || runIsTerminal(run)) {
      if (!runId || !run || runStatus !== "running" || runIsTerminal(run)) { stopRunMonitor(runId); }
      return false;
    }
    if (state.runMonitorRunId && String(state.runMonitorRunId) !== String(runId)) { stopRunMonitor(); }
    if (state.runMonitorRunId === runId) {
      if (state.runMonitorTimer == null && !state.runMonitorInFlight) { queueRunMonitor(runId, runMonitorDelay(run)); }
      return true;
    }
    state.runMonitorRunId = runId;
    state.runMonitorContextEpoch = state.contextEpoch;
    state.runMonitorRetry = 0;
    state.runMonitorLastPollAt = 0;
    queueRunMonitor(runId, 0);
    return true;
  }

  function runMonitorTick(runId) {
    if (state.runMonitorRunId !== runId || state.runMonitorContextEpoch !== state.contextEpoch || state.usingDemo) { return; }
    var run = state.activeRun;
    if (!run || run.id !== runId || String(run.status || "").toLowerCase() !== "running" || runIsTerminal(run)) {
      stopRunMonitor(runId);
      renderProgressViews();
      return;
    }
    if (document.hidden === true) {
      queueRunMonitor(runId, runMonitorMaxDelayMs);
      return;
    }
    if (state.runMonitorInFlight) {
      queueRunMonitor(runId, runMonitorBaseDelayMs);
      return;
    }
    // A full manual refresh owns the same projection read token.  Let it
    // finish, then resume the high-frequency monitor without racing it.
    if (state.refreshing || state.busy) {
      queueRunMonitor(runId, 500);
      return;
    }
    state.runMonitorInFlight = true;
    state.runMonitorLastPollAt = Date.now();
    refreshProgressForRun(runId).then(function (ok) {
      if (state.runMonitorRunId !== runId || state.runMonitorContextEpoch !== state.contextEpoch) { return; }
      if (!ok) {
        state.runMonitorRetry = Number(state.runMonitorRetry || 0) + 1;
        var retryDelay = Math.min(runMonitorMaxDelayMs, runMonitorBaseDelayMs * Math.pow(2, Math.min(3, state.runMonitorRetry - 1)));
        queueRunMonitor(runId, retryDelay);
        return;
      }
      var current = state.activeRun;
      state.runMonitorRetry = 0;
      if (!current || current.id !== runId || String(current.status || "").toLowerCase() !== "running" || runIsTerminal(current)) {
        stopRunMonitor(runId);
        renderProgressViews();
        return;
      }
      queueRunMonitor(runId, runMonitorDelay(current));
    }).catch(function () {
      if (state.runMonitorRunId !== runId || state.runMonitorContextEpoch !== state.contextEpoch) { return; }
      state.runMonitorRetry = Number(state.runMonitorRetry || 0) + 1;
      var delay = Math.min(runMonitorMaxDelayMs, runMonitorBaseDelayMs * Math.pow(2, Math.min(3, state.runMonitorRetry - 1)));
      queueRunMonitor(runId, delay);
    }).finally(function () {
      if (state.runMonitorRunId === runId) { state.runMonitorInFlight = false; }
    });
  }

  function candidateSamplesAreLive(run, candidate) {
    var runStatus = String(run && run.status || "").toLowerCase();
    var candidateStatus = String(candidate && candidate.status || "").toLowerCase();
    if (runStatus !== "running") { return false; }
    return ["accepted", "promoted", "retained", "released", "evaluated", "screened_out", "rejected", "failed", "duplicate", "cancelled"].indexOf(candidateStatus) < 0;
  }

  function renderCandidateSampleViews() {
    if (typeof renderCandidateSamples === "function") { renderCandidateSamples(); }
    if (state.workspace === "process" && typeof renderExecutionSamples === "function") {
      var round = typeof executionRound === "function" ? executionRound(state.activeRun) : null;
      var candidate = typeof executionCandidateFor === "function" ? executionCandidateFor(state.activeRun, round) : null;
      renderExecutionSamples(candidate, state.activeRun);
    }
  }

  function loadCandidateSamples(offset, options) {
    if (!state.usingDemo && ["process", "candidates"].indexOf(state.workspace) < 0) { return Promise.resolve(false); }
    var settings = options || {};
    var run = state.activeRun;
    var candidate = selectedCandidateForSamples();
    if (!run || !candidate) {
      resetCandidateSamples(run && run.id, null);
      renderCandidateSampleViews();
      return Promise.resolve(false);
    }
    var candidateId = candidate.id || candidate.candidate_id;
    if (!candidateSampleSelectionMatches(run.id, candidateId)) { resetCandidateSamples(run.id, candidateId); }
    if (!state.usingDemo && !hasCapability("evaluation.samples.read")) {
      state.candidateSamplePage = null;
      state.candidateSampleUnavailable = true;
      state.candidateSamplePermissionDenied = true;
      state.candidateSampleLoading = false;
      state.candidateSampleRefreshing = false;
      renderCandidateSampleViews();
      if (settings.silent !== true) { showToast("当前 DSH 会话未授予逐样本结果读取能力。"); }
      return Promise.resolve(false);
    }
    if (state.candidateSampleLoading || state.candidateSampleRefreshing) { return Promise.resolve(false); }
    if (state.candidateSampleUnavailable && settings.force !== true) { return Promise.resolve(false); }
    var pageOffset = Math.max(0, Math.floor(Number(offset == null ? state.candidateSampleOffset : offset) || 0));
    var pageLimit = Math.max(1, Math.min(100, Math.floor(Number(state.candidateSampleLimit) || 25)));
    var requestId = state.candidateSampleRequest + 1;
    state.candidateSampleRequest = requestId;
    state.candidateSampleRetryOffset = pageOffset;
    state.candidateSampleLastRequestedAt = Date.now();
    state.candidateSampleError = null;
    state.candidateSampleLoading = !state.candidateSamplePage;
    state.candidateSampleRefreshing = Boolean(state.candidateSamplePage);
    renderCandidateSampleViews();
    if (state.usingDemo) {
      state.candidateSamplePage = demoCandidateSamplePage(run, candidate, pageOffset, pageLimit);
      state.candidateSampleOffset = state.candidateSamplePage.offset;
      state.candidateSampleRetryOffset = null;
      state.candidateSampleUnavailable = true;
      state.candidateSampleLoading = false;
      state.candidateSampleRefreshing = false;
      renderCandidateSampleViews();
      return Promise.resolve(true);
    }
    var contextEpoch = state.contextEpoch;
    var query = new URLSearchParams({candidate_id: candidateId, offset: String(pageOffset), limit: String(pageLimit)});
    return request("/runs/" + encodeURIComponent(run.id) + "/samples?" + query.toString(), {timeout: dataRequestTimeout}).then(function (payload) {
      if (requestId !== state.candidateSampleRequest || contextEpoch !== state.contextEpoch || !candidateSampleSelectionMatches(run.id, candidateId)) { return false; }
      state.candidateSamplePage = normalizeCandidateSamplePage(payload, run.id, candidateId, pageOffset, pageLimit);
      state.candidateSampleOffset = state.candidateSamplePage.offset;
      state.candidateSampleRetryOffset = null;
      state.candidateSampleUnavailable = payload.supported === false;
      state.candidateSampleError = null;
      return true;
    }).catch(function (error) {
      if (requestId !== state.candidateSampleRequest || contextEpoch !== state.contextEpoch || !candidateSampleSelectionMatches(run.id, candidateId)) { return false; }
      if (Number(error && error.status) === 403) {
        state.candidateSamplePage = null;
        state.candidateSampleUnavailable = true;
        state.candidateSamplePermissionDenied = true;
        state.candidateSampleError = null;
        state.candidateSampleRetryOffset = null;
        return false;
      }
      state.candidateSampleError = errorMessage(error);
      if (settings.silent !== true) { showToast("逐样本结果读取失败，已保留当前数据。"); }
      return false;
    }).finally(function () {
      if (requestId === state.candidateSampleRequest) {
        state.candidateSampleLoading = false;
        state.candidateSampleRefreshing = false;
        renderCandidateSampleViews();
      }
    });
  }

  function refreshCandidateSamples(options) {
    if (!state.usingDemo && ["process", "candidates"].indexOf(state.workspace) < 0) { return Promise.resolve(false); }
    var settings = options || {};
    var run = state.activeRun;
    var candidate = selectedCandidateForSamples();
    if (!run || !candidate) { return Promise.resolve(false); }
    var candidateId = candidate.id || candidate.candidate_id;
    if (!candidateSampleSelectionMatches(run.id, candidateId)) { return loadCandidateSamples(0, settings); }
    if (state.candidateSampleLoading || state.candidateSampleRefreshing || state.candidateSampleUnavailable) { return Promise.resolve(false); }
    if (settings.force !== true && Date.now() - Number(state.candidateSampleLastRequestedAt || 0) < 3000) { return Promise.resolve(false); }
    var running = candidateSamplesAreLive(run, candidate);
    if (!running && state.candidateSamplePage && state.candidateSamplePage.complete === true && settings.force !== true) { return Promise.resolve(false); }
    var retryOffset = state.candidateSampleError && state.candidateSampleRetryOffset != null
      ? state.candidateSampleRetryOffset
      : state.candidateSampleOffset;
    return loadCandidateSamples(retryOffset, settings);
  }

  function refreshEventsForRun(runId) {
    if (state.usingDemo) { return Promise.resolve(true); }
    var contextEpoch = state.contextEpoch;
    var pendingKey = contextEpoch + "|" + runId;
    if (state.eventReadPending && state.eventReadPending.key === pendingKey) { return state.eventReadPending.promise; }
    var requestId = (state.eventReadRequest || 0) + 1;
    state.eventReadRequest = requestId;
    var operation = request(eventRequestPath(runId, false), {timeout: dataRequestTimeout}).then(function (data) {
      if (contextEpoch !== state.contextEpoch) { return false; }
      if (requestId !== state.eventReadRequest) { return true; }
      if (state.activeRun && state.activeRun.id === runId) {
        mergeEventStream(runId, data);
        if (state.workspace === "process") { renderProcess(); }
      }
      return true;
    }).catch(function () { return false; }).finally(function () {
      if (state.eventReadPending && state.eventReadPending.promise === operation) { state.eventReadPending = null; }
    });
    state.eventReadPending = {key: pendingKey, promise: operation};
    return operation;
  }

  function captureProgressUiState() {
    if (!document || typeof document.querySelector !== "function") { return null; }
    var rootSelectors = ["#candidate-detail", "#training-assets-table", "#round-stage-list", "#pending-consultations", "#answered-consultations"];
    var roots = rootSelectors.map(function (selector) {
      var root = document.querySelector(selector);
      if (!root || typeof root.querySelectorAll !== "function") { return null; }
      return {
        selector: selector,
        openDetails: Array.prototype.map.call(root.querySelectorAll("details"), function (detail) { return detail.open === true; }),
        scrollTop: Number(root.scrollTop || 0),
        scrollLeft: Number(root.scrollLeft || 0)
      };
    }).filter(Boolean);
    var active = document.activeElement;
    var focus = null;
    if (active && active !== document.body) {
      if (active.id) {
        focus = {id: active.id, selectionStart: active.selectionStart, selectionEnd: active.selectionEnd};
      } else {
        roots.some(function (entry) {
          var root = document.querySelector(entry.selector);
          if (!root || typeof root.contains !== "function" || !root.contains(active) || typeof root.querySelectorAll !== "function") { return false; }
          var focusable = Array.prototype.slice.call(root.querySelectorAll("button, input, select, textarea, a[href], summary, [tabindex]"));
          var index = focusable.indexOf(active);
          if (index >= 0) { focus = {root: entry.selector, index: index, selectionStart: active.selectionStart, selectionEnd: active.selectionEnd}; }
          return index >= 0;
        });
      }
    }
    return {roots: roots, focus: focus};
  }

  function restoreProgressUiState(snapshot) {
    if (!snapshot || !document || typeof document.querySelector !== "function") { return; }
    snapshot.roots.forEach(function (entry) {
      var root = document.querySelector(entry.selector);
      if (!root || typeof root.querySelectorAll !== "function") { return; }
      Array.prototype.forEach.call(root.querySelectorAll("details"), function (detail, index) {
        detail.open = entry.openDetails[index] === true;
      });
      root.scrollTop = entry.scrollTop;
      root.scrollLeft = entry.scrollLeft;
    });
    var focus = snapshot.focus;
    if (!focus) { return; }
    var target = focus.id && typeof document.getElementById === "function" ? document.getElementById(focus.id) : null;
    if (!target && focus.root) {
      var root = document.querySelector(focus.root);
      if (root && typeof root.querySelectorAll === "function") {
        target = Array.prototype.slice.call(root.querySelectorAll("button, input, select, textarea, a[href], summary, [tabindex]"))[focus.index] || null;
      }
    }
    if (!target || typeof target.focus !== "function") { return; }
    target.focus({preventScroll: true});
    if (typeof target.setSelectionRange === "function" && Number.isInteger(focus.selectionStart) && Number.isInteger(focus.selectionEnd)) {
      target.setSelectionRange(focus.selectionStart, focus.selectionEnd);
    }
  }

  function renderProgressViews() {
    var uiState = captureProgressUiState();
    if (typeof renderContext === "function") { renderContext(); }
    if (state.workspace === "process" && typeof renderProcess === "function") { renderProcess(); }
    if (state.workspace === "candidates" && typeof renderCandidates === "function") { renderCandidates(); }
    if (state.workspace === "training" && typeof renderTrainingAssets === "function") { renderTrainingAssets(); }
    if (state.workspace === "collaboration" && typeof renderCollaboration === "function") { renderCollaboration(); }
    var lastUpdated = $("#last-updated");
    if (lastUpdated) { lastUpdated.textContent = state.lastUpdated ? "最近同步：" + formatDate(state.lastUpdated) : "尚未同步"; }
    restoreProgressUiState(uiState);
  }

  function observeRunStatus(previousRun, incomingRun, events) {
    if (!incomingRun) { return false; }
    if (state.createStatus && state.createStatus.runId === incomingRun.id) {
      state.createStatus = createStatusForRun(incomingRun, events);
    }
    var incomingStatus = String(incomingRun.status || "").toLowerCase();
    if (incomingStatus !== "failed") { return false; }
    var previousStatus = String(previousRun && previousRun.status || "").toLowerCase();
    state.commandError = "后台进化失败：" + runFailureMessage(incomingRun, events);
    if (previousStatus !== "failed") { showToast(state.commandError); }
    return previousStatus !== "failed";
  }

  function refreshProgressForRun(runId) {
    if (state.usingDemo) { return Promise.resolve(true); }
    var contextEpoch = state.contextEpoch;
    var requestId = state.runReadRequest + 1;
    state.runReadRequest = requestId;
    // The projection is the authority for live run state. Event streaming is a
    // supplementary tail and must not prevent a fresh projection from being
    // accepted when that endpoint has a transient failure.
    return Promise.all([
      request("/runs/" + encodeURIComponent(runId) + "?view=monitor", {timeout: dataRequestTimeout}),
      Promise.resolve(null)
    ]).then(function (results) {
      if (requestId !== state.runReadRequest || contextEpoch !== state.contextEpoch) { return false; }
      if (!state.activeRun || state.activeRun.id !== runId) { return false; }
      var previousRun = state.activeRun;
      var compactProjection = results[0] && (results[0].projection || results[0].run_projection) || results[0] || {};
      var compactRunId = compactProjection.run_id || compactProjection.id;
      if (compactRunId != null && String(compactRunId) !== String(runId)) {
        throw new Error("运行监视响应返回了其他运行的数据。");
      }
      // Candidate collections remain intentionally absent from monitor
      // responses. Structural changes mark the retained detail as stale; full
      // candidates and training assets are read only when a user opens the run
      // detail or explicitly refreshes it.
      var structuralChange = (
        Number.isFinite(Number(compactProjection.candidates_count))
        && Number(compactProjection.candidates_count) !== Number(previousRun.candidates_count)
      ) || (
        Number.isFinite(Number(compactProjection.generation))
        && Number(compactProjection.generation) !== Number(previousRun.generation)
      ) || (runIsTerminal(compactProjection) && !runIsTerminal(previousRun));
      function commitProjection(authoritativeProjection) {
        if (requestId !== state.runReadRequest || contextEpoch !== state.contextEpoch) { return false; }
        if (!state.activeRun || state.activeRun.id !== runId) { return false; }
        var priorRun = state.activeRun;
        // Merge compact live fields into the last full projection so rounds,
        // candidates and assets survive between structural hydrations.
        var incomingRun = normalizeRun(Object.assign({}, state.activeRun, authoritativeProjection));
        if (incomingRun.projection_revision < state.activeRun.projection_revision) { return false; }
        state.activeRun = incomingRun;
        state.runs = state.runs.map(function (run) { return run.id === runId ? incomingRun : run; });
        if (results[1]) { mergeEventStream(runId, results[1]); }
        observeRunStatus(priorRun, state.activeRun, state.events);
        state.lastUpdated = new Date().toISOString();
        state.loadState = "ready";
        state.lastError = null;
        setConnection("online", state.hostContextReceived ? "DSH 宿主已连接" : "本地服务已连接");
        // The active candidate is discovered asynchronously after RunCreated.
        // Reconcile it on every projection heartbeat so the sample endpoint is
        // subscribed as soon as the first candidate enters evaluation.
        var candidateSelectionChanged = syncCandidateSelection(state.activeRun);
        var selectionChanged = reconcileVisibleRunSelection();
        if (selectionChanged) {
          renderProgressViews();
          return state.activeRun ? selectRun(state.activeRun.id, false) : true;
        }
        renderProgressViews();
        // Do not hold the run heartbeat on a potentially slow sample page. The
        // sample loader is single-flight and renders its own completion/error.
        if (candidateSelectionChanged) {
          loadCandidateSamples(0, {force: true, silent: true});
        } else {
          refreshCandidateSamples({silent: true});
        }
        // A read-only poll is also the recovery hook after a page reload or a
        // DSH reconnect. The scheduler itself remains the only writer.
        return true;
      }
      // Commit compact authority immediately. Never turn a heartbeat into an
      // implicit full-detail read merely because the run crossed a boundary.
      return Promise.resolve(commitProjection(compactProjection)).then(function (committed) {
        if (committed && structuralChange) { state.structureHydrationStale = true; }
        if (committed) {
          if (state.workspace === "process") { refreshEventsForRun(runId); }
          if (typeof ensureWorkspaceData === "function") { ensureWorkspaceData({force: structuralChange}); }
        }
        return committed;
      });
    }).catch(function (error) {
      if (requestId !== state.runReadRequest || contextEpoch !== state.contextEpoch) { return false; }
      state.loadState = "stale";
      state.lastError = errorMessage(error);
      setConnection("offline", "连接已中断 · 显示上次状态");
      renderProgressViews();
      return false;
    });
  }
