"use strict";

  function renderAll() {
    renderSystemBanner(); renderWorkspace(); renderContext(); renderReadiness(); renderParameters(); renderTraining(); renderProcess(); renderCandidates(); renderCollaboration();
    $("#last-updated").textContent = state.lastUpdated ? "最近同步：" + formatDate(state.lastUpdated) : "尚未同步";
    $("#refresh-button").disabled = state.busy || state.refreshing || state.loadState === "loading";
    $("#refresh-button").textContent = state.pendingAction === "refresh" ? "正在刷新" : "刷新数据";
  }

  function exportSummary() {
    if (!state.activeRun) { showToast("当前没有可导出的进化运行。" ); return; }
    var payload = { exported_at: new Date().toISOString(), projection: state.activeRun, events: state.events, redaction: "仅包含脱敏状态视图" };
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    var anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob); anchor.download = state.activeRun.id + "-运行摘要.json"; anchor.click(); URL.revokeObjectURL(anchor.href);
    showToast("运行摘要已导出。" );
  }

  function bindEvents() {
    var tabs = $$('[data-workspace]');
    var trajectoryRenderFrame = null;
    function scheduleTrajectoryRender() {
      if (state.workspace !== "process" || trajectoryRenderFrame != null) { return; }
      trajectoryRenderFrame = window.requestAnimationFrame(function () {
        trajectoryRenderFrame = null;
        renderTrajectory();
      });
    }
    function loadSelectionPreview() {
      if (state.activeRun) { renderTraining(); return; }
      loadSelectedDataset(0);
    }
    tabs.forEach(function (button) {
      button.addEventListener("click", function () {
        state.workspace = button.dataset.workspace;
        renderWorkspace();
        scheduleTrajectoryRender();
        if (state.workspace === "candidates") { refreshCandidateSamples({silent: true}); }
      });
      button.addEventListener("keydown", function (event) {
        var current = tabs.indexOf(button), next = null;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") { next = (current + 1) % tabs.length; }
        if (event.key === "ArrowLeft" || event.key === "ArrowUp") { next = (current - 1 + tabs.length) % tabs.length; }
        if (event.key === "Home") { next = 0; } if (event.key === "End") { next = tabs.length - 1; }
        if (next == null) { return; } event.preventDefault(); state.workspace = tabs[next].dataset.workspace; renderWorkspace(); scheduleTrajectoryRender(); tabs[next].focus();
      });
    });
    $("#domain-pack").addEventListener("change", function () { alignDomainDatasetBinding(); ensureAutonomousBindings(); updateSelectionHelp(); loadSelectionPreview(); });
    ["#policy-model-id", "#judge-model-id"].forEach(function (selector) { $(selector).addEventListener("change", updateSelectionHelp); });
    $("#prediction-model-id").addEventListener("change", function () { alignPredictionBinding(); updateSelectionHelp(); });
    $("#evaluator-id").addEventListener("change", function () { alignEvaluatorBinding(); updateSelectionHelp(); });
    $("#strategy-id").addEventListener("change", function () { alignStrategyModel(); updateSelectionHelp(); });
    $("#dataset-id").addEventListener("change", function () { alignDatasetBinding(); updateSelectionHelp(); loadSelectionPreview(); });
    $("#episode-id").addEventListener("change", function () { updateSelectionHelp(); loadSelectionPreview(); });
    ["#max-generations", "#candidates-per-generation"].forEach(function (selector) {
      $(selector).addEventListener("input", function () { syncCandidateBudget(); renderReadiness(); renderParameters(); });
    });
    $("#max-candidates").addEventListener("input", function () { syncCandidateBudget({ markManual: true }); renderReadiness(); renderParameters(); });
    ["#samples-per-update", "#sample-agent-batch-size", "#sample-concurrency", "#fixed-seed", "#knowledge-online-enabled"].forEach(function (selector) {
      $(selector).addEventListener("input", function () { renderReadiness(); renderParameters(); });
      $(selector).addEventListener("change", function () { renderReadiness(); renderParameters(); });
    });
    ["#max-generations", "#candidates-per-generation", "#samples-per-update", "#sample-agent-batch-size", "#sample-concurrency", "#max-candidates"].forEach(function (selector) {
      $(selector).addEventListener("invalid", function () { state.workspace = "parameters"; renderWorkspace(); });
    });
    $("#dataset-partition").addEventListener("change", function (event) { state.datasetPartition = event.target.value === "training_feedback" ? "training_feedback" : "training_fit"; loadSelectedDataset(0); });
    $("#show-all-fields").addEventListener("change", function (event) { state.showAllFields = event.target.checked === true; renderTraining(); });
    $("#show-cancelled-empty-runs").addEventListener("change", function (event) {
      state.showCancelledEmptyRuns = event.target.checked === true;
      var selectionChanged = reconcileVisibleRunSelection();
      if (selectionChanged && state.activeRun) { selectRun(state.activeRun.id, false); }
      else { renderAll(); }
    });
    $("#show-archived-runs").addEventListener("change", function (event) {
      setArchivedRunsVisible(event.target.checked === true);
    });
    $("#run-select").addEventListener("change", function (event) { selectRun(event.target.value, true); });
    $("#previous-page").addEventListener("click", function () { loadSelectedDataset(Math.max(0, state.pageOffset - state.pageLimit)); });
    $("#next-page").addEventListener("click", function () { loadSelectedDataset(state.pageOffset + state.pageLimit); });
    $("#retry-button").addEventListener("click", function () {
      state.autoAdvanceBlockedRunId = null;
      state.autoAdvanceError = null;
      state.commandError = null;
      connectAndLoad();
    });
    $("#retry-dataset-button").addEventListener("click", function () { loadSelectedDataset(state.pageOffset); });
    $("#toggle-events-button").addEventListener("click", function () { state.showAllEvents = !state.showAllEvents; renderProcess(); });
    $("#refresh-button").addEventListener("click", function () { state.busy = true; state.pendingAction = "refresh"; renderAll(); refreshAll({ refreshDataset: true }).then(function (ok) { showToast(ok ? "数据已刷新。" : "刷新失败，已保留上次状态。" ); }).finally(function () { state.busy = false; state.pendingAction = null; renderAll(); }); });
    $("#advance-button").addEventListener("click", advanceRun);
    $("#pause-button").addEventListener("click", function () { if (state.activeRun) { controlRun(state.activeRun.status === "paused" ? "resume" : "pause"); } });
    $("#cancel-button").addEventListener("click", function () { if (window.confirm("确定取消当前进化运行？取消后不能继续推进。")) { controlRun("cancel"); } });
    $("#archive-button").addEventListener("click", archiveRun);
    $("#delete-button").addEventListener("click", function () {
      if (!state.activeRun) { return; }
      var confirmation = window.prompt("永久删除不可恢复。请输入完整运行 ID 以确认：\n" + state.activeRun.id, "");
      if (confirmation !== null) { deleteRun(confirmation); }
    });
    $("#export-button").addEventListener("click", exportSummary);
    $("#candidate-table").addEventListener("click", function (event) {
      var row = event.target.closest("[data-candidate-id]");
      if (!row) { return; }
      state.candidateSelectionPinned = true;
      if (row.dataset.candidateId === state.selectedCandidateId) { renderCandidateSamples(); return; }
      state.selectedCandidateId = row.dataset.candidateId;
      resetCandidateSamples(state.activeRun && state.activeRun.id, state.selectedCandidateId);
      renderCandidates();
      loadCandidateSamples(0, {force: true});
    });
    $("#candidate-follow-active").addEventListener("change", function (event) {
      state.candidateSelectionPinned = event.target.checked !== true;
      if (state.candidateSelectionPinned) { renderCandidateSamples(); return; }
      syncCandidateSelection(state.activeRun);
      renderCandidates();
      loadCandidateSamples(0, {force: true});
    });
    $("#candidate-samples-previous").addEventListener("click", function () {
      loadCandidateSamples(Math.max(0, state.candidateSampleOffset - state.candidateSampleLimit), {force: true});
    });
    $("#candidate-samples-next").addEventListener("click", function () {
      var page = state.candidateSamplePage || {};
      if (page.has_more === true) { loadCandidateSamples(page.next_offset == null ? state.candidateSampleOffset + state.candidateSampleLimit : page.next_offset, {force: true}); }
    });
    $("#candidate-samples-retry").addEventListener("click", function () {
      loadCandidateSamples(state.candidateSampleRetryOffset == null ? state.candidateSampleOffset : state.candidateSampleRetryOffset, {force: true});
    });
    $("#intervention-kind").addEventListener("change", updateInterventionFields);
    ["input", "change"].forEach(function (eventName) {
      $("#pending-consultations").addEventListener(eventName, function (event) {
        var form = event.target && event.target.closest ? event.target.closest("[data-consultation-answer-form]") : null;
        updateExpertConsultationDraftFromForm(form);
      });
    });
    $("#pending-consultations").addEventListener("submit", function (event) {
      var form = event.target && event.target.closest ? event.target.closest("[data-consultation-answer-form]") : null;
      if (!form) { return; }
      event.preventDefault();
      var draft = updateExpertConsultationDraftFromForm(form);
      if (!draft) { showToast("无法读取专家答复，请刷新后重试。" ); return; }
      answerExpertConsultation(form.dataset.consultationId, {
        answer: draft.answer,
        selected_option: draft.selected_option,
        answered_by: draft.answered_by
      });
    });
    $("#start-form").addEventListener("submit", function (event) {
      event.preventDefault(); var form = new FormData(event.currentTarget);
      // The compact form exposes only two model roles; internal components
      // are selected by the autonomous runtime and are intentionally omitted.
      createRun({
        autonomous_mode: form.get("autonomous_mode") === "true" || form.get("autonomous_mode") === "on",
        model_workflow: form.get("model_workflow") || "research_compile_evolve@1",
        dataset_id: form.get("dataset_id"),
        episode_id: form.get("episode_id") || undefined,
        strategy_model_id: form.get("strategy_model_id") || form.get("policy_model_id"),
        review_model_id: form.get("review_model_id") || form.get("judge_model_id"),
        rounds: Number(form.get("rounds") || form.get("max_generations")),
        candidates_per_generation: Number(form.get("candidates_per_generation")),
        samples_per_update: Number(form.get("samples_per_update")),
        sample_agent_batch_size: Number(form.get("sample_agent_batch_size")),
        sample_concurrency: Number(form.get("sample_concurrency")),
        max_candidates: Number(form.get("max_candidates")),
        fixed_seed: form.get("fixed_seed") === "on", knowledge_online_enabled: form.get("knowledge_online_enabled") === "on"
      });
    });
    $("#intervention-form").addEventListener("submit", function (event) {
      event.preventDefault(); var form = new FormData(event.currentTarget); var kind = form.get("kind"); var overrides = {};
      try { overrides = kind === "parameter_override" ? parseOverrides(form.get("parameter_overrides")) : {}; } catch (error) { showToast(error.message); return; }
      if (kind === "parameter_override" && !Object.keys(overrides).length) { showToast("参数覆盖至少需要一行“参数=值”。"); $("#parameter-overrides").focus(); return; }
      submitIntervention({ kind: kind, message: String(form.get("message") || "").trim(), parameter_overrides: overrides, target_candidate_id: kind === "parent_selection" ? form.get("target_candidate_id") || null : null, created_by: String(form.get("created_by") || "").trim() });
    });
    window.addEventListener("message", function (event) {
      var result = EcologyDSHHost.acceptContextMessage(event);
      if (!result.accepted) { if (!result.ignored) { showToast(result.error); } return; }
      if (state.autoAdvanceRunId) { stopAutoAdvance(state.autoAdvanceRunId, { resetTiming: true }); }
      if (state.runMonitorRunId && typeof stopRunMonitor === "function") { stopRunMonitor(state.runMonitorRunId); }
      state.contextEpoch += 1;
      nextEpoch();
      state.datasetRequest += 1;
      resetCandidateSamples(null, null);
      state.busy = false;
      state.refreshing = false;
      state.pendingAction = null;
      state.commandKeys = {};
      state.expertConsultationDrafts = {};
      state.apiBase = result.context.apiBase;
      state.hostContext = result.context;
      state.hostContextReceived = true;
      var identity = result.context.identity;
      var createdBy = $("#created-by");
      if (identity && (identity.displayName || identity.subjectId)) {
        createdBy.value = identity.displayName || identity.subjectId;
        createdBy.readOnly = true;
      }
      connectAndLoad();
    });
    window.addEventListener("resize", scheduleTrajectoryRender);
  }

  function publicState() {
    return clone({ apiBase: state.apiBase, hostContextReceived: state.hostContextReceived, hostContext: state.hostContext, usingDemo: state.usingDemo, connection: state.connection, loadState: state.loadState, workspace: state.workspace, runs: state.runs, showCancelledEmptyRuns: state.showCancelledEmptyRuns, showArchivedRuns: state.showArchivedRuns, archivedRunCount: state.archivedRunCount, activeRun: state.activeRun, events: state.events, datasetPage: state.datasetPage, candidateSamples: state.candidateSamplePage, autoAdvance: { active: Boolean(state.autoAdvanceRunId), run_id: state.autoAdvanceRunId, blocked: Boolean(state.autoAdvanceBlockedRunId), error: state.autoAdvanceError, last_duration_ms: state.autoAdvanceLastDurationMs, rounds_completed: state.autoAdvanceRoundsCompleted } });
  }

    bindEvents();
  syncCandidateBudget();
  updateInterventionFields();
  EcologyDSHHost.postReady();
  connectAndLoad();
  // The active-run monitor already carries live execution progress.  This
  // slower sweep refreshes catalogs and run history without replaying every
  // large append-only projection six times per minute while the page is idle.
  window.setInterval(function () { if (!document.hidden && !state.busy && !state.refreshing && state.loadState !== "loading" && !state.usingDemo) { refreshAll(); } }, 60000);
  window.EcologyEvolutionPlugin = { getState: publicState, refresh: refreshAll };
