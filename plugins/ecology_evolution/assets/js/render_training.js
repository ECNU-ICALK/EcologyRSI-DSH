"use strict";

  var fieldGroupOrder = ["target", "control", "environment", "crop", "resource", "other"];
  var fieldGroupLabels = {
    target: "评测目标", control: "控制与管理", environment: "环境与根区",
    crop: "作物与产量", resource: "资源投入与消耗", other: "标识与其他"
  };
  function fieldGroup(field) {
    var name = String(field && (field.name || field.id) || "").toLowerCase();
    var role = String(field && (field.role || field.type) || "").toLowerCase();
    if (Object.prototype.hasOwnProperty.call(targetLabels, name)) { return "target"; }
    if (role === "action" || /setpoint|control|pruning|interval/.test(name)) { return "control"; }
    if (role === "resource" || /(?:^|_)(?:irrigation|water|co2|electricity|energy|labour)_?use$|electricity_|energy_use|resource/.test(name)) { return "resource"; }
    if (["crop", "outcome"].indexOf(role) >= 0 || /yield|production|fruit|biomass|truss|stem|leaf|harvest/.test(name)) { return "crop"; }
    if (["environment", "outside_weather", "root_zone", "forcing", "state"].indexOf(role) >= 0 || /temperature|humidity|radiation|wind|rain|soil|root|drain/.test(name)) { return "environment"; }
    return "other";
  }
  function groupedSchema(schema) {
    return fieldGroupOrder.map(function (id) {
      return { id: id, label: fieldGroupLabels[id], fields: schema.filter(function (field) { return fieldGroup(field) === id; }) };
    }).filter(function (group) { return group.fields.length; });
  }
  function coreSchema(schema) {
    if (schema.length <= 10) { return schema; }
    var selected = [];
    function add(field) { if (field && selected.indexOf(field) < 0 && selected.length < 10) { selected.push(field); } }
    schema.forEach(function (field) { var name = field.name || field.id; if (["index", "timestamp"].indexOf(name) >= 0 || fieldGroup(field) === "target") { add(field); } });
    ["control", "environment", "crop", "resource"].forEach(function (group) { add(schema.find(function (field) { return fieldGroup(field) === group && selected.indexOf(field) < 0; })); });
    schema.forEach(function (field) { if (field.required === true) { add(field); } });
    schema.forEach(add);
    return selected;
  }

  function datasetReadinessText(readiness) {
    if (readiness === true) { return "可以读取"; }
    if (readiness === false) { return "尚未就绪"; }
    var value = readiness && (readiness.status || readiness.state);
    if (readiness && readiness.ready === true || value === "ready" || value === "available") { return "可以读取"; }
    if (value === "building" || value === "pending") { return "准备中"; }
    if (value === "restricted" || value === "denied") { return "权限受限"; }
    if (value === "catalog_only") { return "仅登记"; }
    if (value === "missing") { return "源文件缺失"; }
    return "状态未提供";
  }

  function renderDatasetInventory() {
    var node = $("#unavailable-dataset-list");
    var datasets = state.catalog.unavailable_datasets || [];
    if (!datasets.length) { node.innerHTML = "<div class=\"empty-state\">当前没有其他未就绪的数据集。</div>"; return; }
    node.innerHTML = datasets.map(function (dataset) {
      var readiness = dataset.readiness || {};
      var missing = Array.isArray(readiness.missing_globs) && readiness.missing_globs.length ? "缺少源文件：" + readiness.missing_globs.join("、") : "尚未形成可运行数据链路。";
      var description = [itemDescription(dataset), missing].filter(Boolean).join(" ");
      return "<article class=\"dataset-inventory-item\"><div><strong>" + escapeHTML(itemLabel(dataset)) + "</strong><code>" + escapeHTML(itemId(dataset)) + "</code></div><p>" + escapeHTML(description) + "</p><span class=\"pill pill-amber\">" + escapeHTML(datasetReadinessText(readiness)) + "</span></article>";
    }).join("");
  }

  function sourceIntegrityStatusText(value) {
    return {
      verified: "校验通过", missing: "来源归档缺失", mismatch: "校验不一致",
      unreadable: "来源归档不可读", unverifiable: "无法完成校验",
      not_checked: "尚未校验", not_applicable: "无需校验"
    }[String(value || "").toLowerCase()] || "校验状态未提供";
  }
  function sourceIntegrityClass(value) {
    var status = String(value || "").toLowerCase();
    if (["verified", "missing", "mismatch", "unreadable", "unverifiable", "not_checked", "not_applicable"].indexOf(status) >= 0) {
      return "is-" + status.replace(/_/g, "-");
    }
    return "is-pending";
  }
  function sourceIntegrityTone(value) {
    var status = String(value || "").toLowerCase();
    if (status === "verified") { return "pill-green"; }
    if (["mismatch", "unreadable"].indexOf(status) >= 0) { return "pill-red"; }
    if (["missing", "unverifiable"].indexOf(status) >= 0) { return "pill-amber"; }
    return status === "not_applicable" ? "pill-neutral" : "pill-amber";
  }
  function integrityComparisonText(label, matches, actualValue, expectedValue, formatter) {
    var format = formatter || displayText;
    if (matches === true) { return label + "一致（" + format(actualValue != null ? actualValue : expectedValue) + "）"; }
    if (matches === false) { return label + "不一致（实际 " + format(actualValue) + "；目录 " + format(expectedValue) + "）"; }
    return label + "待校验（目录 " + format(expectedValue) + "）";
  }
  function renderSourceIntegrity(integrity, placeholder) {
    var node = $("#source-integrity");
    if (!integrity || typeof integrity !== "object") {
      node.innerHTML = "<div class=\"source-integrity-summary is-not-checked\"><div><strong>来源归档完整性</strong><span>" + escapeHTML(placeholder || "尚未读取来源校验记录。") + "</span></div><span class=\"pill pill-neutral\">等待数据</span></div>";
      return;
    }
    var status = String(integrity.status || "not_checked").toLowerCase();
    var sources = Array.isArray(integrity.sources) ? integrity.sources : [];
    var sourceCount = Number.isFinite(Number(integrity.source_count)) ? Number(integrity.source_count) : sources.length;
    var counts = [
      "共 " + formatNumber(sourceCount) + " 个来源",
      Number(integrity.verified_count || 0) ? "通过 " + formatNumber(integrity.verified_count) : "",
      Number(integrity.missing_count || 0) ? "缺失 " + formatNumber(integrity.missing_count) : "",
      Number(integrity.mismatch_count || 0) ? "不一致 " + formatNumber(integrity.mismatch_count) : "",
      Number(integrity.unverifiable_count || 0) ? "待核验 " + formatNumber(integrity.unverifiable_count) : ""
    ].filter(Boolean).join(" · ");
    var summary = "<div class=\"source-integrity-summary " + sourceIntegrityClass(status) + "\"><div><strong>来源归档完整性</strong><span>" + escapeHTML(integrity.message_zh || counts) + "</span><small>" + escapeHTML(counts) + "</small></div><span class=\"pill " + sourceIntegrityTone(status) + "\">" + escapeHTML(sourceIntegrityStatusText(status)) + "</span></div>";
    var list = sources.length ? "<div class=\"source-integrity-list\">" + sources.map(function (source) {
      var sourceStatus = String(source.status || "not_checked").toLowerCase();
      var sizeText = integrityComparisonText("文件大小", source.size_matches, source.actual_size_bytes, source.expected_size_bytes, formatBytes);
      var md5Text = integrityComparisonText("MD5", source.md5_matches, source.actual_md5, source.expected_md5, function (value) { return displayText(value, "未提供"); });
      var md5Title = [source.expected_md5 ? "目录 MD5：" + source.expected_md5 : "", source.actual_md5 ? "实际 MD5：" + source.actual_md5 : ""].filter(Boolean).join("\n");
      return "<article class=\"source-integrity-item " + sourceIntegrityClass(sourceStatus) + "\"><div><strong title=\"" + escapeHTML(source.name || "") + "\">" + escapeHTML(source.name || "未命名来源归档") + "</strong><span>" + escapeHTML(sizeText) + "</span><span title=\"" + escapeHTML(md5Title) + "\">" + escapeHTML(md5Text) + "</span><small>" + escapeHTML(source.message_zh || "未提供校验说明。") + "</small></div><span class=\"pill " + sourceIntegrityTone(sourceStatus) + "\">" + escapeHTML(sourceIntegrityStatusText(sourceStatus)) + "</span></article>";
    }).join("") + "</div>" : "";
    node.innerHTML = summary + list;
  }

  function trainingAdmissionValue(asset) {
    var admission = asset && asset.admission;
    if (admission && typeof admission === "object") { return admission.tier || admission.bucket || admission.status || admission.value || "pending"; }
    return admission || asset && (asset.admission_bucket || asset.bucket) || "pending";
  }
  function trainingAdmissionText(value) {
    return {
      iterative_positive: "迭代正例", iterative_negative: "迭代负例",
      quarantine: "隔离样本", pending: "待完成"
    }[String(value || "").toLowerCase()] || "待审核";
  }
  function trainingAdmissionClass(value) {
    return {
      iterative_positive: "pill-green", iterative_negative: "pill-amber",
      quarantine: "pill-red", pending: "pill-neutral"
    }[String(value || "").toLowerCase()] || "pill-neutral";
  }
  function assetInterventions(asset) {
    var input = asset && (asset.input || asset.input_summary) || {};
    var values = input.applied_interventions || asset && asset.applied_interventions || [];
    return Array.isArray(values) ? values : [];
  }
  function trainingStageStatusText(value) {
    return {
      completed: "已完成", approved: "已保留", rejected: "未保留", failed: "失败",
      pending: "等待", promoted: "已保留", evaluated: "已评测", not_recorded: "未记录"
    }[String(value || "").toLowerCase()] || displayText(value, "未记录");
  }
  function recordedStageStatus(stage) {
    if (stage && !isBlank(stage.status)) { return stage.status; }
    if (!stage || typeof stage !== "object") { return "not_recorded"; }
    return Object.keys(stage).some(function (key) { return key !== "status" && !isBlank(stage[key]); }) ? "completed" : "not_recorded";
  }
  function recordedBooleanText(value, trueText, falseText) {
    return value === true ? trueText : value === false ? falseText : "未记录";
  }
  function trainingParameterSummary(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) { return "未提供"; }
    var keys = Object.keys(value).sort();
    if (!keys.length) { return "无"; }
    return keys.map(function (key) {
      var item = value[key];
      return (parameterLabels[key] || key) + "=" + (item && typeof item === "object" ? "结构化记录" : formatNumber(item));
    }).join("；");
  }
  function trainingDetailValue(label, value, fullValue) {
    return "<div><span>" + escapeHTML(label) + "</span><strong title=\"" + escapeHTML(fullValue == null ? value : fullValue) + "\">" + escapeHTML(displayText(value)) + "</strong></div>";
  }
  function trainingStageBlock(index, label, status, values) {
    return "<section class=\"training-stage\"><header><span>" + escapeHTML(String(index).padStart(2, "0")) + "</span><h4>" + escapeHTML(label) + "</h4><strong>" + escapeHTML(trainingStageStatusText(status)) + "</strong></header><div>" + values.join("") + "</div></section>";
  }
  function trainingReceiptKindText(value) {
    return {
      ProposalSubmitted: "变更提案已提交", HumanInterventionRecorded: "人工意见已记录",
      HumanInterventionApplied: "人工意见处理结果已记录", CandidateSpawned: "候选方案已生成",
      ArtifactRecorded: "候选训练产物已记录", EvaluationRecorded: "训练反馈检查已完成",
      PromotionDecided: "搜索保留决策已记录", CandidateFailed: "候选方案生成失败"
    }[value] || displayText(value, "事件已记录");
  }
  function renderTrainingEpisodeDetails(asset) {
    var episode = asset && asset.episode;
    var trajectoryHtml = typeof renderTrainingTrajectory === "function" ? renderTrainingTrajectory(asset) : "";
    if ((!episode || typeof episode !== "object") && !trajectoryHtml) { return ""; }
    var hasEpisode = episode && typeof episode === "object" && Object.keys(episode).length > 0;
    episode = hasEpisode ? episode : {};
    var stages = episode.stages || {};
    var strategyInput = stages.strategy_input || {};
    var proposal = stages.proposal_response || {};
    var training = stages.training || {};
    var evaluation = stages.evaluation || {};
    var decision = stages.decision || {};
    var evaluationMetrics = evaluation.metrics || {};
    var admission = decision.admission || asset.admission || {};
    var reproducibility = episode.reproducibility || {};
    var receipts = Array.isArray(episode.event_receipts) ? episode.event_receipts : [];
    var stageHtml = [
      trainingStageBlock(1, "策略输入", recordedStageStatus(strategyInput), [
        trainingDetailValue("优化目标", compactTechnicalText(strategyInput.objective || "未提供"), strategyInput.objective),
        trainingDetailValue("数据与序列", shortId(strategyInput.dataset_id || "未提供") + " / " + shortId(strategyInput.episode_id || "未提供"), [strategyInput.dataset_id, strategyInput.episode_id].filter(Boolean).join(" / ")),
        trainingDetailValue("模型计划中的预测模型（宿主已登记）", predictionModelReferenceLabel(strategyInput.prediction_model_id), strategyInput.prediction_model_id),
        trainingDetailValue("策略模型 API", modelReferenceLabel(strategyInput.strategy_model_id || strategyInput.policy_model_id), strategyInput.strategy_model_id || strategyInput.policy_model_id),
        trainingDetailValue("独立评审模型 API", modelReferenceLabel(strategyInput.review_model_id || strategyInput.judge_model_id), strategyInput.review_model_id || strategyInput.judge_model_id)
      ]),
      trainingStageBlock(2, "提案响应", recordedStageStatus(proposal), [
        trainingDetailValue("提案标题", compactTechnicalText(proposal.title || "未提供"), proposal.title),
        trainingDetailValue("参数修改", trainingParameterSummary(proposal.parameters)),
        trainingDetailValue("父方案", shortId(proposal.parent_candidate_id || "当前基线"), proposal.parent_candidate_id),
        trainingDetailValue("生成依据", compactTechnicalText(proposal.rationale || "未提供"), proposal.rationale)
      ]),
      trainingStageBlock(3, "候选训练", training.status, [
        trainingDetailValue("训练模型", predictionModelReferenceLabel(training.model_id), training.model_id),
        trainingDetailValue("训练分区", partitionText(training.training_partition)),
        trainingDetailValue("训练样本数", formatNumber(training.training_rows)),
        trainingDetailValue("训练产物校验值", shortId(training.artifact_digest || "未提供"), training.artifact_digest)
      ]),
      trainingStageBlock(4, "独立评测", evaluation.status, [
        trainingDetailValue("评测器", catalogReferenceLabel("evaluators", evaluation.evaluator_id, evaluation.evaluator_id || "未提供"), evaluation.evaluator_id),
        trainingDetailValue("反馈分区", partitionText(evaluation.partition)),
        trainingDetailValue("综合得分", formatNumber(evaluation.score)),
        trainingDetailValue("科学约束／独立评审", recordedBooleanText(evaluationMetrics.scientific_pass, "通过", "未通过") + " / " + recordedBooleanText(evaluationMetrics.judge_accepted, "建议保留", "不建议保留"))
      ]),
      trainingStageBlock(5, "搜索决策", decision.status, [
        trainingDetailValue("候选状态", isBlank(decision.candidate_status) ? "未记录" : candidateStatusText(decision.candidate_status)),
        trainingDetailValue("样本准入", trainingAdmissionText(admission.tier)),
        trainingDetailValue("决策理由", compactTechnicalText(decision.reason || decision.failure_reason || "未提供"), decision.reason || decision.failure_reason),
        trainingDetailValue("正式训练条件", admission.formal_training_ready === true ? "已具备" : admission.formal_training_ready === false ? "未具备，需治理审核" : "未记录")
      ])
    ].join("");
    var receiptHtml = receipts.length ? receipts.map(function (receipt) {
      return "<li><span>序号 " + escapeHTML(formatNumber(receipt.seq)) + " · " + escapeHTML(trainingReceiptKindText(receipt.kind)) + "</span><code title=\"" + escapeHTML(receipt.payload_digest || "") + "\">" + escapeHTML(shortId(receipt.payload_digest || "未提供")) + "</code></li>";
    }).join("") : "<li><span>尚未形成事件收据</span><code>—</code></li>";
    var proofs = [
      ["运行任务清单", reproducibility.manifest_digest], ["数据集快照", reproducibility.dataset_digest],
      ["时间分区快照", reproducibility.split_manifest_digest], ["提案", reproducibility.proposal_digest],
      ["候选生成模型配置", reproducibility.policy_model_digest || reproducibility.policy_digest],
      ["预测模型配置", reproducibility.prediction_model_digest], ["独立评审模型配置", reproducibility.judge_model_digest], ["评测器", reproducibility.evaluator_digest],
      ["训练产物", reproducibility.artifact_digest], ["事件收据链", episode.event_chain_digest],
      ["完整训练记录", episode.episode_digest_sha256]
    ];
    var proofHtml = proofs.map(function (item) {
      return trainingDetailValue(item[0], shortId(item[1] || "未提供"), item[1]);
    }).join("");
    var legacyDetails = hasEpisode ? "<details class=\"training-legacy-details\"><summary>查看五阶段完整训练记录、事件收据链与复现校验值</summary><div class=\"training-stage-list\">" + stageHtml + "</div><section class=\"training-proof-section\"><div><h4>事件收据链</h4><ul>" + receiptHtml + "</ul></div><div><h4>复现校验值</h4><div class=\"training-proof-grid\">" + proofHtml + "</div></div></section></details>" : "";
    return "<tr class=\"training-episode-row\"><td colspan=\"7\"><details class=\"training-episode-details\"><summary>查看完整训练轨迹（输入 → 智能体交互 → 反馈 → 优化 → 预测）与五阶段完整训练记录</summary>" + trajectoryHtml + legacyDetails + "</details></td></tr>";
  }
  function renderTrainingAssets() {
    var run = state.activeRun;
    var assets = run && Array.isArray(run.training_assets) ? run.training_assets : [];
    var body = $("#training-assets-table");
    var modelColumnHeading = $(".training-assets-table thead th:nth-child(4)");
    if (modelColumnHeading) { modelColumnHeading.textContent = "策略模型／自主预测模型／评审"; }
    $("#training-assets-count").textContent = formatNumber(assets.length) + " 个样本";
    if (!assets.length) {
      body.innerHTML = "<tr><td colspan=\"7\" class=\"empty-state\">当前运行尚未生成进化训练资产。</td></tr>";
      return;
    }
    body.innerHTML = assets.slice().sort(function (left, right) { return Number(left.generation || 0) - Number(right.generation || 0); }).map(function (asset) {
      var input = asset.input || asset.input_summary || {};
      var output = asset.output || asset.output_summary || {};
      var evaluation = asset.evaluation || asset.evaluation_summary || {};
      var provenance = asset.provenance || {};
      var admission = trainingAdmissionValue(asset);
      var interventions = assetInterventions(asset);
      var strategyId = input.strategy_id || asset.strategy_id || "";
      var strategy = catalogReferenceLabel("strategies", strategyId, strategyId || "未提供");
      var artifact = output.artifact && typeof output.artifact === "object" ? output.artifact : {};
      var predictionModelId = input.prediction_model_id || output.model_id || artifact.model_id || asset.prediction_model_id || run && run.configuration && run.configuration.prediction_model_id || "";
      var predictionModel = predictionModelReferenceLabel(predictionModelId);
      var judgeRecord = evaluation.judge && typeof evaluation.judge === "object" ? evaluation.judge : {};
      var judgeId = judgeRecord.model_id || evaluation.judge_model_id || input.review_model_id || input.judge_model_id || asset.review_model_id || asset.judge_model_id || state.activeRun && state.activeRun.configuration && (state.activeRun.configuration.review_model_id || state.activeRun.configuration.judge_model_id) || "";
      var judge = modelReferenceLabel(judgeId);
      var score = evaluation.score != null ? evaluation.score : asset.score;
      var partition = evaluation.partition || asset.partition;
      var datasetDigest = provenance.dataset_digest || input.dataset_digest || "";
      var artifactDigest = provenance.artifact_digest || artifact.artifact_digest || output.artifact_digest || "";
      var traceTitle = [datasetDigest ? "数据 " + datasetDigest : "", artifactDigest ? "产物 " + artifactDigest : ""].filter(Boolean).join("\n");
      var traceText = [datasetDigest ? "数据 " + shortId(datasetDigest) : "", artifactDigest ? "产物 " + shortId(artifactDigest) : ""].filter(Boolean).join(" / ") || "溯源待完成";
      var candidateId = asset.candidate_id || "未提供";
      var sampleId = asset.sample_id || asset.id || "未提供";
      var admissionRecord = asset.admission && typeof asset.admission === "object" ? asset.admission : {};
      var governance = admissionRecord.formal_training_ready === true ? "已具备正式训练条件" : admissionRecord.requires_governance_review === false ? "仅供当前迭代" : "需治理审核";
      var executedCount = interventions.filter(function (item) { return interventionApplicationStatus(item) !== "recorded"; }).length;
      var recordedOnlyCount = interventions.length - executedCount;
      var interventionSummary = [executedCount ? executedCount + " 条已执行" : "", recordedOnlyCount ? recordedOnlyCount + " 条仅记录" : ""].filter(Boolean).join(" / ") || "无";
      var trajectorySummary = typeof trainingTrajectorySummary === "function" ? trainingTrajectorySummary(asset) : "轨迹待生成";
      return "<tr><td>第 " + escapeHTML(asset.generation || "—") + " 轮</td><td><code title=\"训练样本：" + escapeHTML(sampleId) + "\">" + escapeHTML(shortId(sampleId)) + "</code><small title=\"候选方案：" + escapeHTML(candidateId) + "\">候选：" + escapeHTML(shortId(candidateId)) + "</small><small class=\"training-asset-trace-summary\">" + escapeHTML(trajectorySummary) + "</small></td><td><span class=\"pill " + trainingAdmissionClass(admission) + "\">" + escapeHTML(trainingAdmissionText(admission)) + "</span><small>" + escapeHTML(governance) + "</small></td><td><span title=\"" + escapeHTML(strategyId) + "\">策略：" + escapeHTML(strategy) + "</span><small title=\"" + escapeHTML(predictionModelId) + "\">预测：" + escapeHTML(predictionModel) + "</small><small title=\"" + escapeHTML(judgeId) + "\">评审：" + escapeHTML(judge) + "</small></td><td><strong>" + escapeHTML(formatNumber(score)) + "</strong><small>" + escapeHTML(partitionText(partition)) + "</small></td><td>" + escapeHTML(interventionSummary) + "</td><td title=\"" + escapeHTML(traceTitle) + "\">" + escapeHTML(traceText) + "</td></tr>" + renderTrainingEpisodeDetails(asset);
    }).join("");
  }

  function renderDatasetContextState() {
    var context = state.datasetContext || trainingDatasetContext();
    var note = $("#dataset-context-note");
    var error = $("#dataset-error");
    var retry = $("#retry-dataset-button");
    if (!context) {
      note.textContent = "选择数据集和训练序列后可预览样本。";
      note.removeAttribute("title");
    } else {
      var dataset = catalogItem("datasets", context.dataset_id);
      var episode = datasetEpisodes(dataset).find(function (item) { return itemId(item) === context.episode_id; });
      var datasetName = dataset ? itemBaseLabel(dataset) : context.dataset_id;
      var episodeName = episode ? itemBaseLabel(episode) : context.episode_id;
      note.textContent = context.source === "active_run" ? "已绑定当前运行的冻结快照：" + datasetName + " / " + episodeName : "正在预览待创建任务的数据：" + datasetName + " / " + episodeName;
      note.title = context.source === "active_run" ? ["运行：" + context.run_id, "数据集：" + context.dataset_id, "训练序列：" + context.episode_id, context.dataset_digest ? "数据快照：" + context.dataset_digest : "", context.split_manifest_digest ? "分区快照：" + context.split_manifest_digest : ""].filter(Boolean).join("\n") : "数据集：" + context.dataset_id + "\n训练序列：" + context.episode_id;
    }
    error.hidden = !state.datasetError;
    $("#dataset-error-message").textContent = state.datasetError || "请重试。";
    retry.disabled = state.datasetLoading || !context;
  }

  function renderTraining() {
    var data = state.datasetPage;
    var summary = $("#dataset-summary");
    var schemaNode = $("#schema-list");
    var table = $("#training-table");
    var previous = $("#previous-page");
    var next = $("#next-page");
    $("#field-visibility-control").hidden = true;
    renderDatasetContextState();
    renderDatasetInventory();
    renderTrainingAssets();
    if (["ready", "empty", "stale"].indexOf(state.loadState) >= 0 && !hasCapability("training.data.read") && !state.usingDemo) {
      renderSourceIntegrity(null, "当前 DSH 会话未授予来源校验记录读取能力。");
      summary.innerHTML = "<div class=\"empty-state\">当前 DSH 会话未授予训练数据读取能力。</div>";
      schemaNode.innerHTML = ""; table.querySelector("thead").innerHTML = ""; table.querySelector("tbody").innerHTML = "<tr><td class=\"empty-state\">训练样本不可读取。</td></tr>";
      $("#dataset-digest").textContent = "权限受限"; $("#page-range").textContent = "0–0，共 0 行"; previous.disabled = true; next.disabled = true; return;
    }
    if (state.datasetLoading) {
      renderSourceIntegrity(null, "正在读取数据集来源校验记录…");
      summary.innerHTML = "<div class=\"empty-state\">正在读取训练数据目录…</div>";
      schemaNode.innerHTML = ""; table.querySelector("thead").innerHTML = ""; table.querySelector("tbody").innerHTML = ""; $("#dataset-digest").textContent = "正在读取"; previous.disabled = true; next.disabled = true; return;
    }
    if (!data || !data.page) {
      renderSourceIntegrity(null, state.datasetError ? "本次未能读取来源校验记录。" : "选择数据集和训练序列后可核验来源归档。");
      summary.innerHTML = "<div class=\"empty-state\">" + (state.datasetError ? "本次训练数据未能读取。" : "请选择目录中可用的数据集。") + "</div>";
      schemaNode.innerHTML = ""; table.querySelector("thead").innerHTML = ""; table.querySelector("tbody").innerHTML = "<tr><td class=\"empty-state\">" + (state.datasetError ? "读取失败，请使用上方“重新读取”。" : "暂无训练数据。") + "</td></tr>";
      $("#dataset-digest").textContent = "尚未加载"; $("#page-range").textContent = "0–0，共 0 行"; previous.disabled = true; next.disabled = true; return;
    }
    var sourceIntegrity = data.source_integrity || data.readiness && data.readiness.source_integrity;
    renderSourceIntegrity(sourceIntegrity, "服务端未提供来源归档校验记录。");
    var dataset = Object.assign({}, data.dataset || {}, data.descriptor || {});
    var profile = data.profile || {};
    var readinessText = datasetReadinessText(data.readiness);
    var partitions = data.partitions || {};
    var page = data.page;
    var rows = Array.isArray(page.rows) ? page.rows : [];
    var schema = normalizeSchema(data.features || data.schema, rows);
    var tableSchema = state.showAllFields ? schema : coreSchema(schema);
    var activePartition = data.partition || dataset.partition || state.datasetPartition;
    var partitionEntry = partitions[activePartition];
    var partitionCount = page.total || profile[activePartition + "_rows"] || profile.partition_counts && profile.partition_counts[activePartition] || partitionEntry && (partitionEntry.count || partitionEntry.rows) || partitionEntry || 0;
    var context = state.datasetContext || trainingDatasetContext() || {};
    var catalogDataset = catalogItem("datasets", context.dataset_id || dataset.dataset_id || dataset.id);
    var selectedEpisode = datasetEpisodes(catalogDataset).find(function (item) { return itemId(item) === (dataset.episode_id || context.episode_id); });
    summary.innerHTML = [
      ["数据集名称", dataset.display_name_zh || dataset.display_name || dataset.name || dataset.dataset_id || dataset.id || $("#dataset-id").value],
      ["训练序列／团队", itemLabel(selectedEpisode)],
      [partitionText(activePartition) + "样本", formatNumber(partitionCount) + " 行"],
      ["字段数量", schema.length + " 个"],
      ["目录就绪状态", readinessText + " · 当前查看" + partitionText(activePartition)]
    ].map(function (item) { return "<div class=\"dataset-stat\"><span>" + escapeHTML(item[0]) + "</span><strong title=\"" + escapeHTML(item[1]) + "\">" + escapeHTML(item[1]) + "</strong></div>"; }).join("");
    var digest = data.dataset_digest || dataset.digest || dataset.dataset_digest || "未提供校验值";
    $("#dataset-digest").textContent = shortId(digest); $("#dataset-digest").title = digest;
    schemaNode.innerHTML = schema.length ? groupedSchema(schema).map(function (group) {
      var fields = group.fields.map(function (field) {
        var name = field.name || field.id || "未命名字段";
        var label = field.display_name_zh || field.label || field.display_name || fieldLabel(name);
        var detail = [fieldRoleText(field.role || field.type), field.unit ? "单位：" + unitText(field.unit) : ""].filter(Boolean).join(" · ");
        return "<div class=\"schema-item\" title=\"" + escapeHTML(name) + "\"><strong>" + escapeHTML(label) + "</strong><span>" + escapeHTML(displayText(detail)) + "</span></div>";
      }).join("");
      return "<section class=\"schema-group\"><header><strong>" + escapeHTML(group.label) + "</strong><span>" + group.fields.length + " 个字段</span></header><div class=\"schema-group-fields\">" + fields + "</div></section>";
    }).join("") : "<div class=\"empty-state\">未提供字段结构。</div>";
    var fieldVisibility = $("#field-visibility-control");
    fieldVisibility.hidden = schema.length <= coreSchema(schema).length;
    $("#show-all-fields").checked = state.showAllFields;
    table.querySelector("thead").innerHTML = "<tr>" + tableSchema.map(function (field) { var name = field.name || field.id; return "<th title=\"" + escapeHTML(name) + "\">" + escapeHTML(field.display_name_zh || field.label || field.display_name || fieldLabel(name)) + "</th>"; }).join("") + "</tr>";
    table.querySelector("tbody").innerHTML = rows.length ? rows.map(function (row) {
      var values = row && (row.values || row.features) || row || {};
      return "<tr>" + tableSchema.map(function (field) { var key = field.name || field.id; var value = row[key] != null ? row[key] : values[key]; return "<td>" + escapeHTML(key === "timestamp" ? formatObservationTime(value) : formatNumber(value)) + "</td>"; }).join("") + "</tr>";
    }).join("") : "<tr><td colspan=\"" + Math.max(1, tableSchema.length) + "\" class=\"empty-state\">当前页没有样本。</td></tr>";
    var start = page.total ? Number(page.offset || 0) + 1 : 0;
    var end = Math.min(Number(page.total || 0), Number(page.offset || 0) + rows.length);
    $("#page-range").textContent = start + "–" + end + "，共 " + formatNumber(page.total || 0) + " 行";
    previous.disabled = state.datasetLoading || Number(page.offset || 0) <= 0;
    next.disabled = state.datasetLoading || Number(page.offset || 0) + Number(page.limit || state.pageLimit) >= Number(page.total || 0);
  }
