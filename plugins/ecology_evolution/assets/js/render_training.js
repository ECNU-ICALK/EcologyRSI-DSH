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
    if (["ready", "empty", "stale"].indexOf(state.loadState) >= 0 && !hasCapability("training.data.read")) {
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
