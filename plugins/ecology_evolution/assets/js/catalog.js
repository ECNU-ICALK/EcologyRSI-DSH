"use strict";

  var workspaceFieldNames = {
    process: ["rounds", "candidate_summaries"],
    candidates: ["candidates", "artifacts"],
    training: ["training_assets"],
    collaboration: ["interventions", "expert_consultations", "intervention_candidates"]
  };
  var workspaceLastRead = {};

  function workspaceRequestKey(runId, view) { return state.contextEpoch + "|" + runId + "|" + view; }

  function renderWorkspaceLoadState() {
    var node = $("#workspace-data-status");
    if (!node) { return; }
    var run = state.activeRun, view = state.workspace;
    var pending = run && state.workspaceRequests && state.workspaceRequests[workspaceRequestKey(run.id, view)];
    var error = state.workspaceErrors && state.workspaceErrors[view];
    var version = state.workspaceVersions && state.workspaceVersions[view];
    var stale = run && version != null && version < run.projection_revision;
    node.hidden = !pending && !error && !stale;
    $("#workspace-data-message").textContent = error ? "本区域暂时加载失败，已保留已有数据：" + error : pending ? "正在加载本区域的数据，其他页面仍可使用。" : stale ? "本区域详情为较早快照，运行状态已更新；可刷新详情。" : "";
    $("#workspace-data-retry").hidden = (!error && !stale) || Boolean(pending);
  }

  function ensureWorkspaceData(options) {
    var settings = options || {}, view = state.workspace, run = state.activeRun;
    if (view === "training" && (!state.datasetPage || !sameDatasetContext(state.datasetContext, trainingDatasetContext()) || settings.force)) {
      loadSelectedDataset(state.pageOffset);
    }
    if (state.usingDemo || !run || !workspaceFieldNames[view]) { return Promise.resolve(true); }
    state.workspaceVersions = state.workspaceVersions || {};
    state.workspaceRequests = state.workspaceRequests || {};
    state.workspaceErrors = state.workspaceErrors || {};
    var key = workspaceRequestKey(run.id, view), revision = run.projection_revision;
    if (state.workspaceRequests[key]) { return state.workspaceRequests[key]; }
    if (!settings.force && state.workspaceVersions[view] === revision) { return Promise.resolve(true); }
    if (!settings.force && !settings.navigation && state.workspaceVersions[view] != null && Date.now() - (workspaceLastRead[key] || 0) < 15000) { return Promise.resolve(false); }
    var epoch = state.contextEpoch;
    delete state.workspaceErrors[view];
    workspaceLastRead[key] = Date.now();
    if (view === "process") { refreshEventsForRun(run.id); }
    var operation = request("/runs/" + encodeURIComponent(run.id) + "?view=" + view, {timeout: dataRequestTimeout}).then(function (payload) {
      if (epoch !== state.contextEpoch || !state.activeRun || state.activeRun.id !== run.id) { return false; }
      var projection = payload.projection || {};
      if (payload.view !== view || String(projection.run_id || projection.id) !== String(run.id)) { throw new Error("页面数据身份不匹配"); }
      var incoming = Object.assign({}, state.activeRun);
      // Older section reads may populate their own panel, but must never roll
      // back a newer monitor/control status, progress or counters.
      if (Number(projection.projection_revision) >= incoming.projection_revision) { Object.assign(incoming, projection); }
      else { workspaceFieldNames[view].forEach(function (name) { if (Object.prototype.hasOwnProperty.call(projection, name)) { incoming[name] = projection[name]; } }); }
      state.activeRun = normalizeRun(incoming);
      state.runs = state.runs.map(function (item) { return item.id === run.id ? state.activeRun : item; });
      state.workspaceVersions[view] = Number(projection.projection_revision);
      syncCandidateSelection(state.activeRun);
      if (state.workspace === view && (view === "process" || view === "candidates")) { refreshCandidateSamples({silent: true}); }
      return true;
    }).catch(function (error) {
      if (epoch === state.contextEpoch && state.activeRun && state.activeRun.id === run.id) { state.workspaceErrors[view] = errorMessage(error); }
      return false;
    }).finally(function () {
      if (state.workspaceRequests[key] === operation) { delete state.workspaceRequests[key]; }
      if (epoch === state.contextEpoch && state.activeRun && state.activeRun.id === run.id && state.workspace === view) { renderAll(); }
    });
    state.workspaceRequests[key] = operation;
    renderWorkspaceLoadState();
    return operation;
  }

  function loadTrainingAsset(candidateId) {
    var run = state.activeRun;
    if (!run || state.usingDemo) { return Promise.resolve(false); }
    state.trainingAssetRequests = state.trainingAssetRequests || {};
    state.trainingAssetDetails = state.trainingAssetDetails || {};
    var key = workspaceRequestKey(run.id, "asset:" + candidateId), epoch = state.contextEpoch;
    if (state.trainingAssetRequests[key]) { return state.trainingAssetRequests[key]; }
    var operation = request("/runs/" + encodeURIComponent(run.id) + "?view=asset&candidate_id=" + encodeURIComponent(candidateId), {timeout: dataRequestTimeout}).then(function (payload) {
      if (epoch !== state.contextEpoch || !state.activeRun || state.activeRun.id !== run.id) { return false; }
      var projection = payload.projection || {}, asset = projection.training_asset;
      if (projection.run_id !== run.id || !asset || asset.candidate_id !== candidateId) { throw new Error("训练轨迹身份不匹配"); }
      state.trainingAssetDetails[candidateId] = {revision: projection.projection_revision, asset: Object.assign({}, asset, {details_loaded: true})};
      // Bound the retained full traces independently of the lightweight list.
      var keys = Object.keys(state.trainingAssetDetails);
      while (keys.length > 4) { delete state.trainingAssetDetails[keys.shift()]; }
      return true;
    }).catch(function (error) {
      if (epoch === state.contextEpoch && state.activeRun && state.activeRun.id === run.id) { showToast("训练轨迹加载失败，请重试：" + errorMessage(error)); }
      return false;
    }).finally(function () {
      if (state.trainingAssetRequests[key] === operation) { delete state.trainingAssetRequests[key]; }
      if (state.activeRun && state.activeRun.id === run.id && state.workspace === "training") { renderTrainingAssets(); }
    });
    state.trainingAssetRequests[key] = operation;
    renderTrainingAssets();
    return operation;
  }

  function connectAndLoad() {
    if (state.allowDemo) { loadDemo(); return Promise.resolve(true); }
    var preferredRunId = state.activeRun && state.activeRun.id || state.lastSelectedRunId;
    var hadRecoverableRunContext = Boolean(state.activeRun || state.runs.length);
    if (state.runMonitorRunId && typeof stopRunMonitor === "function") {
      stopRunMonitor(state.runMonitorRunId);
    }
    var epoch = nextEpoch();
    state.runReadRequest += 1;
    state.runOverviewLoading = null;
    state.runOverviewError = null;
    state.usingDemo = false;
    state.loadState = "loading";
    state.lastError = null;
    setConnection("checking", "正在检查服务");
    renderAll();
    // Health is a best-effort liveness hint, not the authority for the
    // browser read model.  Under a busy 64-request run it may exceed four
    // seconds even while catalog and projection reads remain healthy.
    request("/health", { timeout: 4000 }).catch(function () { return null; });
    // Both reads initialize the same view. A busy run's summary can exceed
    // the host's 8-second default even when the catalog is already healthy.
    return Promise.all([
      request("/catalog", { timeout: dataRequestTimeout }),
      request(runsListPath(), { timeout: dataRequestTimeout })
    ]).then(function (results) {
      if (epoch !== state.viewEpoch) { return false; }
      state.catalog = normalizeCatalog(results[0]);
      state.runs = listFrom(results[1], "runs").map(normalizeRun).sort(function (left, right) {
        var leftTime = Date.parse(left.updated_at || left.created_at || "") || 0;
        var rightTime = Date.parse(right.updated_at || right.created_at || "") || 0;
        return rightTime - leftTime;
      });
      state.runListCursor = results[1] && results[1].next_cursor || null;
      state.archivedRunCount = Math.max(0, Number(results[1] && results[1].archived_count || 0));
      var selectableRuns = visibleRuns();
      state.loadState = selectableRuns.length ? "ready" : "empty";
      state.lastError = null;
      var identity = state.hostContext && state.hostContext.identity;
      var hostLabel = identity && (identity.displayName || identity.subjectId);
      setConnection("online", state.hostContextReceived ? "DSH 宿主已连接" + (hostLabel ? " · " + hostLabel : "") : "本地服务已连接");
      populateCatalogControls();
      scheduleEvolutionCapacityRefresh();
      if (selectableRuns.length) {
        var preferredRun = selectableRuns.find(function (run) { return String(run.id) === String(preferredRunId || ""); });
        return selectRun((preferredRun || selectableRuns[0]).id, false, {background: true});
      }
      state.activeRun = null;
      resetEventStream(null);
      resetCandidateSamples(null, null);
      state.lastUpdated = new Date().toISOString();
      renderAll();
      if (state.workspace === "training") { loadSelectedDataset(0); }
      return true;
    }).catch(function (error) {
      if (epoch !== state.viewEpoch) { return false; }
      state.lastError = errorMessage(error);
      if (hadRecoverableRunContext && (state.activeRun || state.runs.length)) {
        state.loadState = "stale";
        setConnection("offline", "连接暂时中断 · 显示上次状态");
        renderAll();
        if (state.activeRun && String(state.activeRun.status || "").toLowerCase() === "running" && typeof startRunMonitor === "function") {
          startRunMonitor(state.activeRun.id);
        }
        return false;
      }
      state.loadState = "error";
      state.catalog = emptyCatalog();
      state.runs = [];
      state.archivedRunCount = 0;
      state.activeRun = null;
      resetEventStream(null);
      setConnection("offline", "服务网关不可用");
      populateCatalogControls();
      renderAll();
      return false;
    });
  }

  function syncSelectedRunAlerts(run, events) {
    var runId = run && run.id;
    state.commandError = String(run && run.status || "").toLowerCase() === "failed"
      ? "后台进化失败：" + runFailureMessage(run, events)
      : null;
  }

  function commitRunSelection(runId, previousRunId) {
    state.lastSelectedRunId = runId;
    if (previousRunId !== runId) { state.candidateSelectionPinned = false; }
    if (state.createStatus && state.createStatus.runId !== runId) { state.createStatus = null; }
    if (previousRunId !== runId) {
      state.candidateSampleRequest += 1;
      state.candidateSampleLoading = false;
      state.candidateSampleRefreshing = false;
    }
    if (state.runMonitorRunId && String(state.runMonitorRunId) !== String(runId) && typeof stopRunMonitor === "function") {
      stopRunMonitor(state.runMonitorRunId);
    }
  }

  function selectRun(runId, notify, options) {
    var background = Boolean(options && options.background);
    if (!runId) { return Promise.resolve(false); }
    var previousRunId = state.activeRun && state.activeRun.id;
    if (state.usingDemo) {
      commitRunSelection(runId, previousRunId);
      state.activeRun = state.runs.find(function (run) { return run.id === runId; }) || null;
      resetEventStream(runId, clone(demoEvents));
      syncSelectedRunAlerts(state.activeRun, state.events);
      state.showAllEvents = false;
      syncCandidateSelection(state.activeRun);
      state.datasetContext = activeRunDatasetContext();
      state.datasetError = null;
      state.datasetPage = demoDatasetPage(0, state.datasetPartition);
      renderAll();
      loadCandidateSamples(0, {force: true});
      if (state.activeRun && typeof startRunMonitor === "function") { startRunMonitor(state.activeRun.id); }
      if (notify) { showToast("已切换进化运行。" ); }
      return Promise.resolve(true);
    }
    // A projection monitor that starts after this selection request would
    // increment the shared read token and silently discard the selected run.
    // Stop it before reading the selected run overview; the
    // selected run starts its own monitor after the commit, while a failed
    // switch restores monitoring for the still-active run below.
    if (state.runMonitorRunId && typeof stopRunMonitor === "function") {
      stopRunMonitor(state.runMonitorRunId);
    }
    var epoch = nextEpoch();
    var requestId = state.runReadRequest + 1;
    state.runReadRequest = requestId;
    state.runOverviewLoading = runId;
    state.runOverviewError = null;
    if (!background) { state.busy = true; state.pendingAction = "select"; }
    renderAll();
    // Selection commits the small authoritative overview. Independent panels
    // must not delay switching the run or pull hidden traces into the page.
    return request("/runs/" + encodeURIComponent(runId) + "?view=overview", {timeout: dataRequestTimeout}).then(function (result) {
      if (requestId !== state.runReadRequest || epoch !== state.viewEpoch) { return false; }
      var selectedRun = normalizeRun(result);
      if (String(selectedRun.id) !== String(runId)) { throw new Error("运行概况身份不匹配"); }
      commitRunSelection(runId, previousRunId);
      state.activeRun = selectedRun;
      state.structureHydrationStale = false;
      resetEventStream(runId);
      state.workspaceVersions = {};
      state.workspaceErrors = {};
      state.trainingAssetDetails = {};
      if (previousRunId !== runId) { state.pageOffset = 0; }
      syncSelectedRunAlerts(state.activeRun, state.events);
      state.showAllEvents = false;
      state.runs = state.runs.map(function (run) { return run.id === runId ? state.activeRun : run; });
      syncCandidateSelection(state.activeRun);
      state.loadState = "ready";
      state.lastUpdated = new Date().toISOString();
      ensureWorkspaceData();
      if (state.activeRun && typeof startRunMonitor === "function") { startRunMonitor(state.activeRun.id); }
      if (notify) { showToast("已切换进化运行。" ); }
      return true;
    }).catch(function (error) {
      if (requestId !== state.runReadRequest || epoch !== state.viewEpoch) { return false; }
      state.runOverviewError = "无法读取进化运行：" + errorMessage(error);
      if (!background) { showToast(state.runOverviewError); }
      if (state.activeRun && String(state.activeRun.status || "").toLowerCase() === "running" && typeof startRunMonitor === "function") {
        startRunMonitor(state.activeRun.id);
      }
      return false;
    }).finally(function () {
      if (epoch === state.viewEpoch && requestId === state.runReadRequest) {
        state.runOverviewLoading = null;
        if (!background) { state.busy = false; state.pendingAction = null; }
        renderAll();
      }
    });
  }

  function setArchivedRunsVisible(visible) {
    var nextValue = visible === true;
    if (state.showArchivedRuns === nextValue) { return Promise.resolve(true); }
    if (state.busy || state.refreshing) { return Promise.resolve(false); }
    if (state.usingDemo) {
      state.showArchivedRuns = false;
      renderAll();
      return Promise.resolve(false);
    }
    var previousValue = state.showArchivedRuns;
    state.showArchivedRuns = nextValue;

    var currentRunId = state.activeRun && state.activeRun.id;
    state.busy = true;
    state.pendingAction = "run-history";
    renderAll();
    return request(runsListPath(), { timeout: dataRequestTimeout }).then(function (data) {
      state.runs = listFrom(data, "runs").map(normalizeRun).filter(function (run) { return nextValue || !run.archived; }).sort(function (left, right) {
        var leftTime = Date.parse(left.updated_at || left.created_at || "") || 0;
        var rightTime = Date.parse(right.updated_at || right.created_at || "") || 0;
        return rightTime - leftTime;
      });
      state.runListCursor = data && data.next_cursor || null;
      state.archivedRunCount = Math.max(0, Number(data && data.archived_count || 0));
      var current = state.runs.find(function (run) { return run.id === currentRunId; });
      if (current) {
        // Summary pages must not discard the selected run's hydrated details.
        if (state.activeRun && state.activeRun.id === current.id) {
          state.activeRun.archived = current.archived;
          state.runs = state.runs.map(function (run) { return run.id === current.id ? state.activeRun : run; });
        } else { state.activeRun = current; }
        state.busy = false;
        state.pendingAction = null;
        renderAll();
        return true;
      }
      state.activeRun = null;
      state.busy = false;
      state.pendingAction = null;
      reconcileVisibleRunSelection();
      renderAll();
      return state.activeRun ? selectRun(state.activeRun.id, false) : true;
    }).catch(function (error) {
      state.showArchivedRuns = previousValue;
      state.busy = false;
      state.pendingAction = null;
      showToast("归档历史读取失败：" + errorMessage(error));
      renderAll();
      return false;
    });
  }

  function populateSelect(selector, items, placeholder, preferred, requiredRole) {
    var select = $(selector);
    var previous = preferred || select.value;
    var isModelChoice = selector === "#policy-model-id" || selector === "#judge-model-id";
    var options = items.map(function (item) {
      // Both role selectors receive the same DSH directory. Role-incompatible
      // entries remain visible but disabled so the reason is explicit.
      var roleUnavailable = isModelChoice && requiredRole && !modelSupportsRole(item, requiredRole);
      // A configured model stays selectable after a transient call failure so
      // the next run can retry. Directory blocks, missing credentials, and
      // host-only routes are hard-disabled.
      var backendUnavailable = isModelChoice && item && (item.directory_available === false || item.configured === false || Object.prototype.hasOwnProperty.call(item, "credential_configured") && item.credential_configured !== true);
      var unavailable = roleUnavailable || backendUnavailable || item && item.readiness && item.readiness.ready === false || !isModelChoice && (item && item.available === false || item && Object.prototype.hasOwnProperty.call(item, "credential_configured") && item.credential_configured !== true);
      var reasonCode = String(item && item.unavailable_reason && item.unavailable_reason.code || "");
      var unavailableLabels = {
        insecure_http_blocked: "（非 HTTPS，后端已阻止）",
        host_route_not_available_to_sidecar: "（当前后端未配置）",
        missing_gateway_url: "（缺少网关地址）",
        invalid_gateway_url: "（网关地址无效）",
        unsupported_provider_api: "（接口不受支持）"
      };
      var suffix = roleUnavailable ? "（不支持此职责）" : unavailableLabels[reasonCode] || (unavailable ? "（未就绪）" : "");
      return "<option value=\"" + escapeHTML(itemId(item)) + "\"" + (unavailable ? " disabled" : "") + ">" + escapeHTML(itemLabel(item) + suffix) + "</option>";
    });
    select.innerHTML = options.length ? options.join("") : "<option value=\"\">" + escapeHTML(placeholder) + "</option>";
    if (items.some(function (item) { return itemId(item) === previous; })) { select.value = previous; }
    select.disabled = !items.length;
  }
  function preferredCatalogModelId(items, candidates, excluded) {
    var values = Array.isArray(candidates) ? candidates : [candidates];
    var blocked = String(excluded || "");
    for (var index = 0; index < values.length; index += 1) {
      var candidate = String(values[index] || "").trim();
      if (!candidate || candidate === blocked) { continue; }
      if (items.some(function (item) { return itemId(item) === candidate; })) { return candidate; }
    }
    return "";
  }
  function runnableDatasetItems() {
    var trainingDatasets = ["agc_cucumber_2018", "agc_tomato_2019"];
    return (state.catalog.datasets || []).filter(function (item) {
      if (!item || item.available === false || item.runnable === false) { return false; }
      if (!state.usingDemo && trainingDatasets.indexOf(itemId(item)) < 0) { return false; }
      return !item.readiness || item.readiness.ready !== false;
    }).sort(function (left, right) {
      return trainingDatasets.indexOf(itemId(left)) - trainingDatasets.indexOf(itemId(right));
    });
  }
  function datasetEpisodes(dataset) {
    var episodes = dataset && normalizeList(dataset.episodes);
    if (episodes && episodes.length) { return episodes; }
    return dataset && itemId(dataset) === "generated-toy-series@1" ? [{ id: "generated-toy-series@1:seed-0", label: "固定随机种子 0" }] : [];
  }
  function populateEpisodeControl(preferred) {
    var dataset = selectedCatalogItem("datasets", "#dataset-id");
    var episodes = datasetEpisodes(dataset);
    // Episode is a frozen data boundary, not a user strategy choice.  Select
    // the catalog default so the compact model-led form never asks the user
    // to choose a team or sequence manually.
    populateSelect("#episode-id", episodes, "没有可用的训练序列", preferred || episodes[0] && itemId(episodes[0]));
    var episode = datasetEpisodes(dataset).find(function (item) { return itemId(item) === $("#episode-id").value; });
    setHelp("#episode-help", episode, "由数据集自动冻结训练序列；不需要人工选择团队。");
  }
  function sharedDshModelItems() {
    // dsh_models is authoritative for execution configuration and health. The
    // host directory may contain additional entries that the sidecar cannot
    // call; retain them as explicitly disabled diagnostics instead of either
    // hiding backend models or silently dropping host-only models.
    var items = state.catalog.dsh_models_explicit === true ? (state.catalog.dsh_models || []).slice() : state.catalog.models || [];
    var hostModels = state.hostContextReceived && state.hostContext && Array.isArray(state.hostContext.models) ? state.hostContext.models : [];
    if (hostModels.length) {
      var modelIdentity = function (item) {
        if (!item || typeof item !== "object") {
          return { provider: "", exact: [itemId(item)].filter(Boolean), bare: [] };
        }
        var id = itemId(item);
        var modelId = String(item.model_id || "").trim();
        var model = String(item.model || "").trim();
        var provider = String(item.provider || "").trim().toLowerCase();
        var qualifiedValues = [id, modelId];
        if (!provider && model) {
          qualifiedValues.some(function (value) {
            var text = String(value || "").trim();
            var slashSuffix = "/" + model;
            var colonSuffix = ":" + model;
            if (text.endsWith(slashSuffix)) { provider = text.slice(0, -slashSuffix.length).toLowerCase(); return true; }
            if (text.endsWith(colonSuffix)) { provider = text.slice(0, -colonSuffix.length).toLowerCase(); return true; }
            return false;
          });
        }
        var aliases = Array.isArray(item.aliases) ? item.aliases.map(function (value) { return String(value || "").trim(); }).filter(Boolean) : [];
        var exact = qualifiedValues.concat(provider && model ? [provider + "/" + model, provider + ":" + model] : []).concat(aliases.filter(function (value) { return /[\/:]/.test(value); }));
        var bare = [model].concat(aliases.filter(function (value) { return !/[\/:]/.test(value); }));
        var unique = function (values) { return values.filter(function (value, index) { return value && values.indexOf(value) === index; }); };
        return { provider: provider, exact: unique(exact), bare: unique(bare) };
      };
      var sameModel = function (left, right) {
        var leftIdentity = modelIdentity(left);
        var rightIdentity = modelIdentity(right);
        if (leftIdentity.exact.some(function (key) { return rightIdentity.exact.indexOf(key) >= 0; })) { return true; }
        if (leftIdentity.provider && rightIdentity.provider && leftIdentity.provider !== rightIdentity.provider) { return false; }
        return leftIdentity.bare.some(function (key) { return rightIdentity.bare.indexOf(key) >= 0; });
      };
      hostModels.forEach(function (hostItem) {
        var matched = items.some(function (item) { return sameModel(item, hostItem); });
        if (matched) { return; }
        items.push(Object.assign({}, hostItem, {
          id: itemId(hostItem), model_id: itemId(hostItem), model_source: "dsh_host_only",
          credential_configured: false, configured: false,
          authentication_verified: false, authentication_state: "unavailable",
          directory_available: false, execution_available: false,
          connection_available: false, available: false,
          unavailable_reason: {
            code: "host_route_not_available_to_sidecar",
            message: "The model is registered by the DSH host but has no callable route in this backend."
          },
          connection: {state: "unavailable", last_checked_at: null, last_error: "host_route_not_available_to_sidecar"}
        }));
      });
    }
    return items;
  }
  function autonomousModelItems() {
    // Connection health is diagnostic only; actual operations report failures.
    var items = sharedDshModelItems();
    var remote = items.filter(function (item) {
      if (!item || item.local_model === true || String(item.authentication_state || "").toLowerCase() === "local") { return false; }
      return true;
    });
    // Explicit demo/local fallback keeps the browser demo usable while a
    // real DSH deployment still prefers authenticated API models exclusively.
    return remote.length || !state.usingDemo ? remote : items;
  }
  function populateCatalogControls() {
    populateSelect("#domain-pack", state.catalog.domain_packs, "没有可用的领域模型包");
    populateSelect("#dataset-id", runnableDatasetItems(), "没有可运行的训练数据集");
    populateSelect("#strategy-id", state.catalog.strategies, "没有可用的进化策略");
    populateSelect("#evaluator-id", state.catalog.evaluators, "没有可用的评测器");
    var policyItems = autonomousModelItems();
    var judgeItems = autonomousModelItems();
    var policyCompatibleItems = policyItems.filter(function (item) {
      return modelSupportsRole(item, "propose") && item.directory_available !== false && item.configured !== false && item.credential_configured !== false;
    });
    var currentPolicyId = $("#policy-model-id").value;
    var hostPolicyPreference = EcologyDSHHost.preferredModelId("propose", policyCompatibleItems.map(itemId), "");
    var policyPreference = preferredCatalogModelId(policyCompatibleItems, [currentPolicyId, hostPolicyPreference], "") || (policyCompatibleItems.find(modelCredentialReady) || policyCompatibleItems[0] || {}).id;
    populateSelect("#policy-model-id", policyItems, "没有可用的策略 API 模型", policyPreference, "propose");
    var policyModelId = $("#policy-model-id").value;
    var selectedJudgeId = $("#judge-model-id").value;
    var judgeCompatibleItems = judgeItems.filter(function (item) { return modelSupportsRole(item, "judge") && item.directory_available !== false && item.configured !== false && item.credential_configured !== false; });
    var hostJudgePreference = EcologyDSHHost.preferredModelId("judge", judgeCompatibleItems.map(itemId), policyModelId);
    var preferredJudgeId = preferredCatalogModelId(judgeCompatibleItems, [selectedJudgeId, hostJudgePreference], policyModelId);
    var preferredJudgeItem = judgeCompatibleItems.find(function (item) { return itemId(item) === preferredJudgeId; }) || judgeCompatibleItems.find(function (item) { return itemId(item) !== policyModelId; }) || judgeCompatibleItems[0] || judgeItems[0];
    populateSelect("#judge-model-id", judgeItems, "没有可用的独立评审 API 模型", preferredJudgeItem ? itemId(preferredJudgeItem) : "", "judge");
    alignDomainDatasetBinding();
    populateEpisodeControl($("#episode-id").value);
    alignDatasetBinding();
    ensureAutonomousBindings();
    updateSelectionHelp();
  }
  function selectedCatalogItem(collection, selector) {
    var value = $(selector).value;
    return state.catalog[collection].find(function (item) { return itemId(item) === value; });
  }
  function alignDomainDatasetBinding() {
    var current = selectedCatalogItem("datasets", "#dataset-id");
    var domainId = current && (current.domain_pack_id || current.domain_id || current.domain);
    if (!domainId) {
      $("#domain-pack").value = "";
      if ($("#research-domain-id")) { $("#research-domain-id").value = ""; }
      return;
    }
    if (Array.prototype.some.call($("#domain-pack").options, function (option) {
      return option.value === String(domainId) && !option.disabled;
    })) {
      $("#domain-pack").value = String(domainId);
    }
    if ($("#research-domain-id")) { $("#research-domain-id").value = String(domainId); }
  }
  function alignDatasetBinding() {
    var dataset = selectedCatalogItem("datasets", "#dataset-id");
    if (!dataset) { return; }
    alignDomainDatasetBinding();
    var domainPackId = dataset.domain_pack_id;
    if (domainPackId && Array.prototype.some.call($("#domain-pack").options, function (option) { return option.value === domainPackId && !option.disabled; })) {
      $("#domain-pack").value = domainPackId;
    }
    var datasetId = itemId(dataset);
    var evaluatorId = datasetId === "generated-toy-series@1" && state.usingDemo
      ? "toy_time_forward@1" : state.catalog.runtime_evaluator_id;
    var evaluator = state.catalog.evaluators.find(function (item) {
      return itemId(item) === evaluatorId && item.available !== false
        && (!Array.isArray(item.dataset_ids) || item.dataset_ids.indexOf(datasetId) >= 0);
    });
    $("#evaluator-id").value = evaluator ? itemId(evaluator) : "";
  }
  function alignStrategyModel() {
    var policyItems = autonomousModelItems().filter(function (item) { return modelSupportsRole(item, "propose"); });
    var current = selectedCatalogItem("policy_models", "#policy-model-id");
    var policy = current && policyItems.some(function (item) { return itemId(item) === itemId(current); }) ? current : policyItems.find(function (item) {
      return modelCredentialReady(item) || item.local_model === true;
    }) || policyItems[0];
    if (policy) { $("#policy-model-id").value = itemId(policy); }
  }
  function ensureAutonomousBindings() {
    // These bindings remain in the request for compatibility and auditability,
    // but are always derived from the selected data boundary and model output.
    var strategy = selectedCatalogItem("strategies", "#strategy-id") || state.catalog.strategies.find(function (item) { return item.available !== false; });
    if (strategy) { $("#strategy-id").value = itemId(strategy); }
    alignDatasetBinding();
    if (!$("#episode-id").value) { populateEpisodeControl(); }
    alignDomainDatasetBinding();
    if (!$("#research-domain-id").value) { $("#research-domain-id").value = $("#domain-pack").value || ""; }
  }
  function setHelp(selector, item, fallback) { $(selector).textContent = itemDescription(item) || fallback; }
  function setModelHelp(selector, item, fallback) {
    var status = modelConnectionStateText(item);
    $(selector).textContent = [itemDescription(item) || fallback, status ? "调用状态：" + status + "。" : ""].filter(Boolean).join(" ");
  }
  function samplesPerUpdateMinimum() {
    var evaluator = selectedCatalogItem("evaluators", "#evaluator-id");
    var value = Number(evaluator && evaluator.minimum_samples_per_update);
    return Number.isInteger(value) && value > 0 ? value : 1;
  }

  function samplesPerUpdateSelectionMinimum() {
    var evaluator = selectedCatalogItem("evaluators", "#evaluator-id");
    var value = Number(evaluator && evaluator.minimum_selection_samples_per_update);
    return Number.isInteger(value) && value > 0 ? value : samplesPerUpdateMinimum();
  }

  function predictionCellsPerOrigin() {
    var evaluator = selectedCatalogItem("evaluators", "#evaluator-id");
    var value = Number(evaluator && (evaluator.prediction_cells_per_origin || evaluator.prediction_task_count));
    return Number.isInteger(value) && value > 0 ? value : 1;
  }

  function predictionOriginsPerUpdateSelectionMinimum() {
    var evaluator = selectedCatalogItem("evaluators", "#evaluator-id");
    var value = Number(evaluator && evaluator.minimum_selection_origin_samples_per_update);
    return Number.isInteger(value) && value > 0
      ? value
      : Math.ceil(samplesPerUpdateSelectionMinimum() / predictionCellsPerOrigin());
  }

  function updateOptimizationScheduleBoundary() {
    var input = $("#selection-holdout-origin-count");
    input.min = String(Math.max(169, predictionOriginsPerUpdateSelectionMinimum()));
  }
  function updateSelectionHelp() {
    setHelp("#domain-pack-help", selectedCatalogItem("domain_packs", "#domain-pack"), "由所选训练数据集自动推导知识检索范围、科学约束和数据适配器。");
    setHelp("#dataset-help", selectedCatalogItem("datasets", "#dataset-id"), "当前训练支持 2018 黄瓜和 2019 番茄数据集；数据集自动匹配研究领域和授权评测边界。");
    populateEpisodeControl($("#episode-id").value);
    setHelp("#strategy-help", selectedCatalogItem("strategies", "#strategy-id"), "策略模型只能在宿主注册表提供的有界策略和参数空间内提出方案。");
    setHelp("#evaluator-help", selectedCatalogItem("evaluators", "#evaluator-id"), "由系统依据数据和候选产物自动绑定。");
    setModelHelp("#policy-model-help", selectedModelCatalogItem("#policy-model-id"), "负责查找资料、制定研究计划并提出允许范围内的修改。");
    setModelHelp("#judge-model-help", selectedModelCatalogItem("#judge-model-id"), "独立检查预测效果、科学约束和版本选择结果。");
    alignDomainDatasetBinding();
    updateOptimizationScheduleBoundary();
    renderReadiness();
  }

  function predictionBindingsReady() {
    var datasetId = $("#dataset-id").value;
    var evaluator = selectedCatalogItem("evaluators", "#evaluator-id");
    if (!evaluator || evaluator.available === false || (evaluator.readiness && evaluator.readiness.ready === false)) { return false; }
    var ids = evaluator.prediction_model_ids || [];
    return state.catalog.prediction_models.some(function (item) {
      return ids.indexOf(itemId(item)) >= 0 && item.available !== false
        && (!item.readiness || item.readiness.ready !== false)
        && (!Array.isArray(item.dataset_ids) || item.dataset_ids.indexOf(datasetId) >= 0);
    });
  }

  function readiness() {
    var availableDatasets = runnableDatasetItems();
    var configuredModels = autonomousModelItems().filter(function (item) { return item.directory_available !== false && item.configured !== false && item.credential_configured !== false; });
    var catalogReady = availableDatasets.length && state.catalog.domain_packs.length && configuredModels.length;
    // Data and model roles are chosen here; prediction methods are chosen during research.
    var selections = ["#dataset-id", "#policy-model-id", "#judge-model-id", "#max-generations", "#candidates-per-generation", "#max-candidates", "#formal-origin-count", "#local-batch-origin-count", "#max-local-edits-per-batch", "#selection-holdout-origin-count"].every(function (selector) { return Boolean($(selector).value); });
    var sampleAgentBatchSize = Number($("#sample-agent-batch-size").value);
    var candidateConcurrency = Number($("#candidate-concurrency").value);
    var sampleConcurrency = Number($("#sample-concurrency").value);
    var schedule = null;
    var scheduleReady = true;
    try { schedule = optimizationScheduleFromControls(); } catch (_error) { scheduleReady = false; }
    var executionParametersReady = Number($("#candidates-per-generation").value) === 4
      && Number.isInteger(candidateConcurrency) && candidateConcurrency >= 1 && candidateConcurrency <= 8
      && Number.isInteger(sampleAgentBatchSize) && sampleAgentBatchSize === 9
      && Number.isInteger(sampleConcurrency) && sampleConcurrency >= 1 && sampleConcurrency <= sampleConcurrencyMaximum;
    var separated = $("#policy-model-id").value && $("#judge-model-id").value && $("#policy-model-id").value !== $("#judge-model-id").value;
    var dshReady = (state.connection === "online" || state.usingDemo) && hasCapability("evolution.projection.read");
    var selectedDataset = selectedCatalogItem("datasets", "#dataset-id");
    var selectedDatasetId = selectedDataset && itemId(selectedDataset);
    var datasetReady = Boolean(selectedDatasetId) && availableDatasets.some(function (item) { return itemId(item) === selectedDatasetId; });
    var episodeReady = datasetReady && datasetEpisodes(selectedDataset).some(function (item) { return itemId(item) === $("#episode-id").value; });
    var selectedPolicy = selectedModelCatalogItem("#policy-model-id");
    var selectedJudge = selectedModelCatalogItem("#judge-model-id");
    var policyApiSelected = Boolean(selectedPolicy && selectedPolicy.local_model !== true && String(selectedPolicy.authentication_state || "").toLowerCase() !== "local");
    var judgeApiSelected = Boolean(selectedJudge && selectedJudge.local_model !== true && String(selectedJudge.authentication_state || "").toLowerCase() !== "local");
    var strategyRoleCatalogReady = state.usingDemo || dshModelRoleCount("strategy", false) > 0 || Boolean(selectedPolicy && selectedPolicy.local_model === true);
    var reviewRoleCatalogReady = state.usingDemo || dshModelRoleCount("review", false) > 0 || Boolean(selectedJudge && selectedJudge.local_model === true);
    var policyConnectionReady = modelSupportsRole(selectedPolicy, "propose") && modelCredentialReady(selectedPolicy) && (state.usingDemo || policyApiSelected);
    var judgeConnectionReady = modelSupportsRole(selectedJudge, "judge") && modelCredentialReady(selectedJudge) && (state.usingDemo || judgeApiSelected);
    var autoBindingsReady = predictionBindingsReady();
    var derivedDomain = selectedDataset && (selectedDataset.domain_pack_id || selectedDataset.domain_id || selectedDataset.domain);
    var domainDataMatch = Boolean(derivedDomain) && derivedDomain === $("#domain-pack").value && derivedDomain === $("#research-domain-id").value;
    var budget = candidateBudgetStatus();
    var capacity = state.cohortCapacityReport;
    var capacityReady = state.usingDemo || Boolean(
      capacity
      && state.cohortCapacitySignature === evolutionCapacityRequest().signature
      && capacity.sufficient === true
    );
    var capacityLabel = state.cohortCapacityLoading
      ? "服务端正在核验可用数据量"
      : state.cohortCapacityError
        ? "服务端 cohort 容量核验失败：" + state.cohortCapacityError
        : capacity
          ? cohortCapacityLabel(capacity)
          : "等待服务端核验可用数据量";
    return [
      { label: "配置目录已加载", ready: Boolean(catalogReady) },
      { label: "运行配置已完整选择", ready: selections },
      { label: "入围候选 " + formatNumber(schedule && schedule.formal_origin_count_per_finalist || Number($("#formal-origin-count").value)) + " 个优化时点、每批时点数及轮末比较参数有效", ready: scheduleReady },
      { label: "方案数量、预测内容和并发设置有效", ready: executionParametersReady },
      { label: "候选总预算可完整覆盖全部轮次（至少 " + formatNumber(budget.required_candidates) + " 个）", ready: budget.budget_sufficient },
      { label: capacityLabel, ready: capacityReady },
      { label: "所选训练数据集可运行", ready: datasetReady },
      { label: "训练序列已由数据集自动冻结", ready: episodeReady },
      { label: "研究领域已由数据集自动推导", ready: domainDataMatch },
      { label: "运行时预测工具目录与统一评测规则已就绪", ready: autoBindingsReady },
      { label: "策略模型职责已在 DSH 目录登记", ready: strategyRoleCatalogReady },
      { label: "独立评审职责已在 DSH 目录登记", ready: reviewRoleCatalogReady },
      { label: "策略模型 API 已安全配置", ready: policyConnectionReady },
      { label: "独立评审模型 API 已安全配置", ready: judgeConnectionReady },
      { label: "策略与独立评审职责已分离", ready: separated },
      { label: "具备创建进化运行的授权能力", ready: hasCapability("evolution.run.create") },
      { label: "运行服务与脱敏状态读取能力可用", ready: dshReady }
    ];
  }

  function cohortCapacityLabel(capacity) {
    if (!capacity) { return "等待服务端核验可用数据量"; }
    var required = formatNumber(capacity.planned_origin_occurrences == null ? capacity.required_unique_origins : capacity.planned_origin_occurrences);
    var available = formatNumber(capacity.available_source_origins == null ? capacity.available_eligible_origins : capacity.available_source_origins);
    var maximum = formatNumber(capacity.max_feasible_generations);
    if (capacity.sufficient !== true) {
      if (capacity.rejection_reason) { return String(capacity.rejection_reason); }
      return "可用数据量不足（需要 " + required + " / 可用 " + available + "；最多 " + maximum + " 轮）";
    }
    var reused = Number(capacity.reused_origin_occurrences || 0);
    var reuseNote = reused > 0
      ? "；数据耗尽后循环复用 " + formatNumber(reused) + " 个起点"
      : "；本次无需复用起点";
    return "数据量可满足运行（计划 " + required + " / 可用 " + available + reuseNote + "；最多 " + maximum + " 轮）";
  }

  function evolutionCapacityRequest() {
    var schedule = null;
    try { schedule = optimizationScheduleFromControls(); } catch (_error) {}
    var body = {
      dataset_id: $("#dataset-id").value || "",
      episode_id: $("#episode-id").value || null,
      optimization_schedule: schedule,
      planned_generations: Number($("#max-generations").value)
    };
    return { body: body, signature: JSON.stringify(body) };
  }

  function renderEvolutionCapacityState() {
    if (typeof renderReadiness === "function") { renderReadiness(); }
    if (typeof renderParameters === "function") { renderParameters(); }
  }

  function capacityVerificationPending() {
    if (state.usingDemo) { return false; }
    var planned = evolutionCapacityRequest();
    var body = planned.body;
    var valid = Boolean(body.dataset_id && body.episode_id && body.optimization_schedule
      && Number.isInteger(body.planned_generations) && body.planned_generations > 0);
    return valid && (state.cohortCapacityLoading || !state.cohortCapacityError
      && (!state.cohortCapacityReport || state.cohortCapacitySignature !== planned.signature));
  }

  function refreshEvolutionCapacity() {
    var planned = evolutionCapacityRequest();
    var body = planned.body;
    if (!body.dataset_id || !body.episode_id || !body.optimization_schedule || !Number.isInteger(body.planned_generations) || body.planned_generations < 1) {
      state.cohortCapacityReport = null;
      state.cohortCapacitySignature = null;
      state.cohortCapacityLoading = false;
      state.cohortCapacityError = null;
      renderEvolutionCapacityState();
      return Promise.resolve(null);
    }
    if (state.usingDemo) {
      state.cohortCapacityReport = {
        sufficient: true,
        required_unique_origins: body.optimization_schedule.formal_origin_count_per_finalist + body.planned_generations * (body.optimization_schedule.screening_origin_count + body.optimization_schedule.selection_holdout_origin_count),
        available_eligible_origins: 999999,
        max_feasible_generations: body.planned_generations,
        cohort_reuse_policy: "cycle_after_exhaustion@1",
        reused_origin_occurrences: 0
      };
      state.cohortCapacitySignature = planned.signature;
      state.cohortCapacityLoading = false;
      state.cohortCapacityError = null;
      renderEvolutionCapacityState();
      return Promise.resolve(state.cohortCapacityReport);
    }
    var requestId = state.cohortCapacityRequest + 1;
    state.cohortCapacityRequest = requestId;
    state.cohortCapacityLoading = true;
    state.cohortCapacityError = null;
    renderEvolutionCapacityState();
    return request("/evolution-capacity", { method: "POST", body: body, timeout: dataRequestTimeout }).then(function (report) {
      if (requestId !== state.cohortCapacityRequest || evolutionCapacityRequest().signature !== planned.signature) { return null; }
      state.cohortCapacityReport = report;
      state.cohortCapacitySignature = planned.signature;
      return report;
    }).catch(function (error) {
      if (requestId !== state.cohortCapacityRequest) { return null; }
      state.cohortCapacityReport = null;
      state.cohortCapacitySignature = null;
      state.cohortCapacityError = errorMessage(error);
      return null;
    }).finally(function () {
      if (requestId !== state.cohortCapacityRequest) { return; }
      state.cohortCapacityLoading = false;
      renderEvolutionCapacityState();
    });
  }

  function scheduleEvolutionCapacityRefresh() {
    if (state.cohortCapacityTimer != null) { window.clearTimeout(state.cohortCapacityTimer); }
    state.cohortCapacityError = null;
    renderEvolutionCapacityState();
    state.cohortCapacityTimer = window.setTimeout(function () {
      state.cohortCapacityTimer = null;
      refreshEvolutionCapacity();
    }, 250);
  }

  function activeRunDatasetContext() {
    var run = state.activeRun;
    if (!run) { return null; }
    var configuration = run.configuration || {};
    var dataset = run.dataset || {};
    var datasetId = configuration.dataset_id || dataset.dataset_id || dataset.id;
    var episodeId = configuration.episode_id || dataset.episode_id;
    if (!datasetId || !episodeId) { return null; }
    return {
      dataset_id: String(datasetId), episode_id: String(episodeId), source: "active_run", run_id: run.id,
      dataset_digest: dataset.digest || run.dataset_digest || null,
      data_protocol_digest: dataset.data_protocol_digest || null,
      split_manifest_digest: dataset.split_manifest_digest || null
    };
  }
  function selectedDatasetContext() {
    var datasetId = $("#dataset-id").value;
    var episodeId = $("#episode-id").value;
    if (!datasetId || !episodeId) { return null; }
    return { dataset_id: datasetId, episode_id: episodeId, source: "selection", run_id: null };
  }
  function trainingDatasetContext() {
    return activeRunDatasetContext() || selectedDatasetContext();
  }
  function sameDatasetContext(left, right) {
    return Boolean(left && right) && left.dataset_id === right.dataset_id && left.episode_id === right.episode_id && left.run_id === right.run_id && left.dataset_digest === right.dataset_digest && left.split_manifest_digest === right.split_manifest_digest && left.data_protocol_digest === right.data_protocol_digest;
  }

  function loadOlderRuns() {
    if (!state.runListCursor || state.loadingOlderRuns || state.refreshing || state.busy || state.usingDemo) { return Promise.resolve(false); }
    var epoch = state.viewEpoch;
    var cursor = state.runListCursor;
    var includeArchived = state.showArchivedRuns;
    state.loadingOlderRuns = true;
    renderContext();
    return request(runsListPath(cursor), {timeout: dataRequestTimeout}).then(function (data) {
      if (epoch !== state.viewEpoch || includeArchived !== state.showArchivedRuns) { return false; }
      var known = new Set(state.runs.map(function (run) { return run.id; }));
      listFrom(data, "runs").map(normalizeRun).forEach(function (run) {
        if (!known.has(run.id)) { state.runs.push(run); known.add(run.id); }
      });
      state.runListCursor = data.next_cursor || null;
      return true;
    }).catch(function (error) { showToast("历史运行读取失败：" + errorMessage(error)); return false; })
      .finally(function () { state.loadingOlderRuns = false; renderContext(); });
  }
