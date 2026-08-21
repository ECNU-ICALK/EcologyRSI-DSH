"use strict";

  function connectAndLoad() {
    if (state.allowDemo) { loadDemo(); return Promise.resolve(true); }
    var preferredRunId = state.activeRun && state.activeRun.id || state.lastSelectedRunId;
    if (state.autoAdvanceRunId && typeof stopAutoAdvance === "function") {
      stopAutoAdvance(state.autoAdvanceRunId, { resetTiming: true });
    }
    if (state.runMonitorRunId && typeof stopRunMonitor === "function") {
      stopRunMonitor(state.runMonitorRunId);
    }
    var epoch = nextEpoch();
    state.runReadRequest += 1;
    state.usingDemo = false;
    state.loadState = "loading";
    state.lastError = null;
    setConnection("checking", "正在检查服务");
    renderAll();
    return Promise.all([request("/health", { timeout: 4000 }), request("/catalog", { timeout: dataRequestTimeout }), request(runsListPath())]).then(function (results) {
      if (epoch !== state.viewEpoch) { return false; }
      state.catalog = normalizeCatalog(results[1]);
      state.runs = listFrom(results[2], "runs").map(normalizeRun).sort(function (left, right) {
        var leftTime = Date.parse(left.updated_at || left.created_at || "") || 0;
        var rightTime = Date.parse(right.updated_at || right.created_at || "") || 0;
        return rightTime - leftTime;
      });
      state.archivedRunCount = Math.max(0, Number(results[2] && results[2].archived_count || 0));
      var selectableRuns = visibleRuns();
      state.loadState = selectableRuns.length ? "ready" : "empty";
      state.lastError = null;
      var identity = state.hostContext && state.hostContext.identity;
      var hostLabel = identity && (identity.displayName || identity.subjectId);
      setConnection("online", state.hostContextReceived ? "DSH 宿主已连接" + (hostLabel ? " · " + hostLabel : "") : "本地服务已连接");
      populateCatalogControls();
      if (selectableRuns.length) {
        var preferredRun = selectableRuns.find(function (run) { return String(run.id) === String(preferredRunId || ""); });
        return selectRun((preferredRun || selectableRuns[0]).id, false);
      }
      state.activeRun = null;
      state.events = [];
      resetCandidateSamples(null, null);
      state.lastUpdated = new Date().toISOString();
      renderAll();
      loadSelectedDataset(0);
      return true;
    }).catch(function (error) {
      if (epoch !== state.viewEpoch) { return false; }
      state.loadState = "error";
      state.lastError = errorMessage(error);
      state.catalog = emptyCatalog();
      state.runs = [];
      state.archivedRunCount = 0;
      state.activeRun = null;
      state.events = [];
      setConnection("offline", "服务网关不可用");
      populateCatalogControls();
      renderAll();
      return false;
    });
  }

  function syncSelectedRunAlerts(run, events) {
    var runId = run && run.id;
    if (state.autoAdvanceBlockedRunId !== runId) {
      state.autoAdvanceBlockedRunId = null;
      state.autoAdvanceError = null;
    }
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
    if (state.autoAdvanceRunId && state.autoAdvanceRunId !== runId && typeof stopAutoAdvance === "function") {
      stopAutoAdvance(state.autoAdvanceRunId, { resetTiming: true });
    }
    if (state.runMonitorRunId && String(state.runMonitorRunId) !== String(runId) && typeof stopRunMonitor === "function") {
      stopRunMonitor(state.runMonitorRunId);
    }
    if (previousRunId && previousRunId !== runId) { state.autoAdvanceLastDurationMs = null; }
  }

  function selectRun(runId, notify) {
    if (!runId) { return Promise.resolve(false); }
    var previousRunId = state.activeRun && state.activeRun.id;
    if (state.usingDemo) {
      commitRunSelection(runId, previousRunId);
      state.activeRun = state.runs.find(function (run) { return run.id === runId; }) || null;
      state.events = clone(demoEvents);
      syncSelectedRunAlerts(state.activeRun, state.events);
      state.showAllEvents = false;
      syncCandidateSelection(state.activeRun);
      state.datasetContext = activeRunDatasetContext();
      state.datasetError = null;
      state.datasetPage = demoDatasetPage(0, state.datasetPartition);
      renderAll();
      loadCandidateSamples(0, {force: true});
      if (state.activeRun && typeof startRunMonitor === "function") { startRunMonitor(state.activeRun.id); }
      if (state.activeRun && typeof ensureAutoAdvanceForRun === "function") { ensureAutoAdvanceForRun(state.activeRun.id); }
      if (notify) { showToast("已切换进化运行。" ); }
      return Promise.resolve(true);
    }
    // A projection monitor that starts after this selection request would
    // increment the shared read token and silently discard the selected run.
    // Stop it before opening the transactional pair of run/event reads; the
    // selected run starts its own monitor after the commit, while a failed
    // switch restores monitoring for the still-active run below.
    if (state.runMonitorRunId && typeof stopRunMonitor === "function") {
      stopRunMonitor(state.runMonitorRunId);
    }
    var epoch = nextEpoch();
    var requestId = state.runReadRequest + 1;
    state.runReadRequest = requestId;
    state.busy = true;
    state.pendingAction = "select";
    renderAll();
    return Promise.all([request("/runs/" + encodeURIComponent(runId)), request("/runs/" + encodeURIComponent(runId) + "/events")]).then(function (results) {
      if (requestId !== state.runReadRequest || epoch !== state.viewEpoch) { return false; }
      var selectedRun = normalizeRun(results[0]);
      var selectedEvents = normalizeEvents(results[1]);
      commitRunSelection(runId, previousRunId);
      state.activeRun = selectedRun;
      state.events = selectedEvents;
      syncSelectedRunAlerts(state.activeRun, state.events);
      state.showAllEvents = false;
      state.runs = state.runs.map(function (run) { return run.id === runId ? state.activeRun : run; });
      syncCandidateSelection(state.activeRun);
      state.loadState = "ready";
      state.lastUpdated = new Date().toISOString();
      loadSelectedDataset(0);
      loadCandidateSamples(0, {force: true});
      if (state.activeRun && typeof startRunMonitor === "function") { startRunMonitor(state.activeRun.id); }
      if (notify) { showToast("已切换进化运行。" ); }
      if (state.activeRun && typeof ensureAutoAdvanceForRun === "function") { ensureAutoAdvanceForRun(state.activeRun.id); }
      return true;
    }).catch(function (error) {
      if (requestId !== state.runReadRequest || epoch !== state.viewEpoch) { return false; }
      state.commandError = "无法读取进化运行：" + errorMessage(error);
      showToast(state.commandError);
      if (state.activeRun && String(state.activeRun.status || "").toLowerCase() === "running" && typeof startRunMonitor === "function") {
        startRunMonitor(state.activeRun.id);
      }
      return false;
    }).finally(function () {
      if (epoch === state.viewEpoch) { state.busy = false; state.pendingAction = null; renderAll(); }
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
    state.showArchivedRuns = nextValue;
    if (!nextValue) {
      state.runs = state.runs.filter(function (run) { return !run.archived; });
      var selectionChanged = reconcileVisibleRunSelection();
      renderAll();
      return selectionChanged && state.activeRun
        ? selectRun(state.activeRun.id, false)
        : Promise.resolve(true);
    }

    var currentRunId = state.activeRun && state.activeRun.id;
    state.busy = true;
    state.pendingAction = "run-history";
    renderAll();
    return request(runsListPath(), { timeout: dataRequestTimeout }).then(function (data) {
      state.runs = listFrom(data, "runs").map(normalizeRun).sort(function (left, right) {
        var leftTime = Date.parse(left.updated_at || left.created_at || "") || 0;
        var rightTime = Date.parse(right.updated_at || right.created_at || "") || 0;
        return rightTime - leftTime;
      });
      state.archivedRunCount = Math.max(0, Number(data && data.archived_count || 0));
      var current = state.runs.find(function (run) { return run.id === currentRunId; });
      if (current) {
        state.activeRun = current;
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
      state.showArchivedRuns = false;
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
    return (state.catalog.datasets || []).filter(function (item) {
      if (!item || item.available === false || item.runnable === false) { return false; }
      return !item.readiness || item.readiness.ready !== false;
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
    populateSelect("#prediction-model-id", state.catalog.prediction_models, "没有可用的预测模型");
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
    var selectedPredictor = selectedCatalogItem("prediction_models", "#prediction-model-id");
    var predictor = selectedPredictor && (!Array.isArray(selectedPredictor.dataset_ids) || selectedPredictor.dataset_ids.indexOf(datasetId) >= 0) ? selectedPredictor : state.catalog.prediction_models.find(function (item) {
      return !Array.isArray(item.dataset_ids) || item.dataset_ids.indexOf(datasetId) >= 0;
    });
    if (predictor) { $("#prediction-model-id").value = itemId(predictor); }
    alignPredictionBinding();
  }
  function alignPredictionBinding() {
    var dataset = selectedCatalogItem("datasets", "#dataset-id");
    var predictor = selectedCatalogItem("prediction_models", "#prediction-model-id");
    if (!dataset || !predictor) { return; }
    var datasetId = itemId(dataset);
    var predictorId = itemId(predictor);
    var current = selectedCatalogItem("evaluators", "#evaluator-id");
    var compatible = function (item) {
      return (!Array.isArray(item.dataset_ids) || item.dataset_ids.indexOf(datasetId) >= 0) &&
        (!Array.isArray(item.prediction_model_ids) || item.prediction_model_ids.indexOf(predictorId) >= 0);
    };
    var defaultEvaluatorId = datasetId === "generated-toy-series@1"
      ? "toy_time_forward@1"
      : "greenhouse_multihorizon_time_forward@2";
    var evaluator = state.catalog.evaluators.find(function (item) {
      return itemId(item) === defaultEvaluatorId && compatible(item);
    }) || (current && compatible(current) ? current : state.catalog.evaluators.find(compatible));
    if (evaluator) { $("#evaluator-id").value = itemId(evaluator); }
  }
  function alignEvaluatorBinding() {
    var evaluator = selectedCatalogItem("evaluators", "#evaluator-id");
    var dataset = selectedCatalogItem("datasets", "#dataset-id");
    if (!evaluator || !dataset) { return; }
    var datasetId = itemId(dataset);
    var allowed = Array.isArray(evaluator.prediction_model_ids) ? evaluator.prediction_model_ids : [];
    var current = selectedCatalogItem("prediction_models", "#prediction-model-id");
    if (current && (!allowed.length || allowed.indexOf(itemId(current)) >= 0)) { return; }
    var predictor = state.catalog.prediction_models.find(function (item) {
      return (!allowed.length || allowed.indexOf(itemId(item)) >= 0) &&
        (!Array.isArray(item.dataset_ids) || item.dataset_ids.indexOf(datasetId) >= 0);
    });
    if (predictor) { $("#prediction-model-id").value = itemId(predictor); }
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
  function updateSamplesPerUpdateBoundary() {
    var evaluator = selectedCatalogItem("evaluators", "#evaluator-id");
    var minimum = samplesPerUpdateMinimum();
    var taskCount = Number(evaluator && evaluator.prediction_task_count);
    if (!Number.isInteger(taskCount) || taskCount < 1) { taskCount = minimum; }
    var input = $("#samples-per-update");
    input.min = String(minimum);
    $("#samples-per-update-help").textContent = "每轮冻结的 training_feedback 样本数；当前评测包含 " + formatNumber(taskCount) + " 个目标与预测时距单元，至少需要 " + formatNumber(minimum) + " 个样本，确保每个单元至少出现一次。";
  }
  function updateSelectionHelp() {
    setHelp("#domain-pack-help", selectedCatalogItem("domain_packs", "#domain-pack"), "由所选训练数据集自动推导知识检索范围、科学约束和数据适配器。");
    setHelp("#dataset-help", selectedCatalogItem("datasets", "#dataset-id"), "仅使用服务端确认可运行的数据集；数据集同时决定研究领域和授权评测边界。");
    populateEpisodeControl($("#episode-id").value);
    setHelp("#prediction-model-help", selectedCatalogItem("prediction_models", "#prediction-model-id"), "由策略模型提出，宿主从已登记预测模型中校验采用。");
    setHelp("#strategy-help", selectedCatalogItem("strategies", "#strategy-id"), "策略模型只能在宿主注册表提供的有界策略和参数空间内提出方案。");
    setHelp("#evaluator-help", selectedCatalogItem("evaluators", "#evaluator-id"), "由系统依据数据和候选产物自动绑定。");
    setModelHelp("#policy-model-help", selectedModelCatalogItem("#policy-model-id"), "负责检索公开元数据、形成结构化研究计划并生成有界候选参数；不会执行模型源码。");
    setModelHelp("#judge-model-help", selectedModelCatalogItem("#judge-model-id"), "独立检查预测效果、科学约束和搜索保留结论。");
    alignDomainDatasetBinding();
    updateSamplesPerUpdateBoundary();
    updateParameterOverrideHelp();
    renderReadiness();
  }

  function readiness() {
    var availableDatasets = runnableDatasetItems();
    var configuredModels = autonomousModelItems().filter(function (item) { return item.directory_available !== false && item.configured !== false && item.credential_configured !== false; });
    var catalogReady = availableDatasets.length && state.catalog.domain_packs.length && configuredModels.length;
    // The visible data boundary and two configured API roles are user
    // inputs.  The research domain and internal components are derived.
    var selections = ["#dataset-id", "#policy-model-id", "#judge-model-id", "#max-generations", "#candidates-per-generation", "#samples-per-update", "#sample-agent-batch-size", "#sample-concurrency", "#max-candidates"].every(function (selector) { return Boolean($(selector).value); });
    var samplesPerUpdate = Number($("#samples-per-update").value);
    var minimumSamplesPerUpdate = samplesPerUpdateMinimum();
    var sampleAgentBatchSize = Number($("#sample-agent-batch-size").value);
    var sampleConcurrency = Number($("#sample-concurrency").value);
    var executionParametersReady = Number.isInteger(samplesPerUpdate) && samplesPerUpdate >= minimumSamplesPerUpdate && samplesPerUpdate <= 100000
      && Number.isInteger(sampleAgentBatchSize) && sampleAgentBatchSize >= 1 && sampleAgentBatchSize <= 128
      && Number.isInteger(sampleConcurrency) && sampleConcurrency >= 1 && sampleConcurrency <= 8;
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
    var autoBindingsReady = $("#autonomous-mode").value === "true" || Boolean($("#episode-id").value && $("#strategy-id").value && $("#prediction-model-id").value && $("#evaluator-id").value);
    var derivedDomain = selectedDataset && (selectedDataset.domain_pack_id || selectedDataset.domain_id || selectedDataset.domain);
    var domainDataMatch = Boolean(derivedDomain) && derivedDomain === $("#domain-pack").value && derivedDomain === $("#research-domain-id").value;
    var budget = candidateBudgetStatus();
    return [
      { label: "配置目录已加载", ready: Boolean(catalogReady) },
      { label: "运行配置已完整选择", ready: selections },
      { label: "每轮样本覆盖全部评测目标与时距（至少 " + formatNumber(minimumSamplesPerUpdate) + " 个）", ready: executionParametersReady },
      { label: "候选总预算可完整覆盖全部轮次（至少 " + formatNumber(budget.required_candidates) + " 个）", ready: budget.budget_sufficient },
      { label: "所选训练数据集可运行", ready: datasetReady },
      { label: "训练序列已由数据集自动冻结", ready: episodeReady },
      { label: "研究领域已由数据集自动推导", ready: domainDataMatch },
      { label: "预测模型、进化策略与评测器将由模型自动确定", ready: autoBindingsReady },
      { label: "策略模型职责已在 DSH 目录登记", ready: strategyRoleCatalogReady },
      { label: "独立评审职责已在 DSH 目录登记", ready: reviewRoleCatalogReady },
      { label: "策略模型 API 已安全配置", ready: policyConnectionReady },
      { label: "独立评审模型 API 已安全配置", ready: judgeConnectionReady },
      { label: "策略与独立评审职责已分离", ready: separated },
      { label: "具备创建进化运行的授权能力", ready: hasCapability("evolution.run.create") },
      { label: "运行服务与脱敏状态读取能力可用", ready: dshReady }
    ];
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
    return Boolean(left && right) && left.dataset_id === right.dataset_id && left.episode_id === right.episode_id && left.run_id === right.run_id && left.dataset_digest === right.dataset_digest && left.split_manifest_digest === right.split_manifest_digest;
  }
