import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const read = (name) => fs.readFileSync(path.join(root, name), "utf8");

const manifest = JSON.parse(read("plugin.json"));
const html = read("index.html");
const css = read("styles.css");
const dshHostClient = fs.readFileSync(path.resolve(root, "../../integrations/dsh_ecology_plugin/lib/client.js"), "utf8");
const scripts = [
  "assets/js/host.js",
  "assets/js/core.js",
  "assets/js/catalog.js",
  "assets/js/data.js",
  "assets/js/commands.js",
  "assets/js/render_shell.js",
  "assets/js/render_training.js",
  "assets/js/render_process.js",
  "assets/js/render_candidates.js",
  "app.js",
];
const app = scripts.map(read).join("\n");
const catalogSource = read("assets/js/catalog.js");
const dataSource = read("assets/js/data.js");
const commandsSource = read("assets/js/commands.js");

// The parameter page is the public source of truth for the adaptive finalist
// schedule. These defaults are complete forecast 个预测时点, never scoring cells.
for (const [id, value] of [
  ["formal-origin-count", "35"],
  ["local-batch-origin-count", "25"],
  ["max-local-edits-per-batch", "2"],
  ["selection-holdout-origin-count", "50"],
]) {
  assert.match(html, new RegExp(`id="${id}"[^>]*value="${value}"`));
}
assert.match(html, /id="max-generations"[^>]*value="5"/);
assert.match(html, /默认最多运行 5 轮，并按数据容量调整/);
assert.doesNotMatch(html, /id="samples-per-update"/);
assert.match(commandsSource, /function normalizedOptimizationSchedule\(values\)/);
assert.match(commandsSource, /optimization_protocol: optimizationProtocolForSchedule\(optimizationSchedule\)/);
assert.match(commandsSource, /optimization_schedule: optimizationSchedule/);
assert.doesNotMatch(commandsSource, /samples_per_update:/);

const hostMessages = [];
const hostFetches = [];
const parentWindow = {
  postMessage(message, targetOrigin) { hostMessages.push({message, targetOrigin}); },
};
const hostWindow = {
  location: new URL("http://localhost:4173/plugins/ecology/evolution/?api=/api/ecology-evolution"),
  parent: parentWindow,
  setTimeout,
  clearTimeout,
};
const hostSandbox = {
  AbortController,
  URL,
  URLSearchParams,
  fetch: async (url, options) => {
    hostFetches.push({url, options});
    return {ok: true, status: 200, text: async () => '{"ok":true}'};
  },
  window: hostWindow,
};
vm.runInNewContext(read("assets/js/host.js"), hostSandbox);
const hostAdapter = hostWindow.EcologyDSHHost;

const defaultHostWindow = {
  location: new URL("http://localhost:4173/plugins/ecology/evolution/"),
  parent: parentWindow,
  setTimeout,
  clearTimeout,
};
const defaultHostSandbox = {
  AbortController,
  URL,
  URLSearchParams,
  fetch: async () => ({ok: true, status: 200, text: async () => '{"ok":true}'}),
  window: defaultHostWindow,
};
vm.runInNewContext(read("assets/js/host.js"), defaultHostSandbox);
assert.equal(
  defaultHostWindow.EcologyDSHHost.getPublicContext().apiBase,
  "/api/ecology-evolution",
);

assert.equal(hostAdapter.postReady(), true);
assert.match(manifest.version, /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/);
assert.equal(hostMessages[0].targetOrigin, "http://localhost:4173");
assert.equal(hostMessages[0].message.context_protocol, "ecology-evolution.host-context/1");
assert.equal(hostMessages[0].message.version, manifest.version);
assert.deepEqual(
  Array.from(hostMessages[0].message.supported_api_bases),
  ["/api/ecology-evolution"],
);

const rejectedContext = hostAdapter.acceptContextMessage({
  source: parentWindow,
  origin: "https://untrusted.example",
  data: {type: "dsh.context", api_base: "/api/ecology-evolution", capability_token: "secret"},
});
assert.equal(rejectedContext.accepted, false);
assert.match(rejectedContext.error, /来源未获授权/);

const acceptedContext = hostAdapter.acceptContextMessage({
  source: parentWindow,
  origin: "http://localhost:4173",
  data: {
    type: "dsh.context",
    api_base: "/api/ecology-evolution",
    capability_token: "secret-token",
    identity: {subject_id: "researcher-17", display_name: "研究员甲"},
    capabilities: ["evolution.projection.read"],
    models: [{model_id: "dsh-policy@1", roles: ["propose"]}],
  },
});
assert.equal(acceptedContext.accepted, true);
assert.equal(acceptedContext.context.apiBase, "/api/ecology-evolution");
assert.equal(acceptedContext.context.identity.displayName, "研究员甲");
assert.equal(JSON.stringify(acceptedContext.context).includes("secret-token"), false);
assert.equal(hostAdapter.preferredModelId("propose", ["dsh-policy@1"], ""), "dsh-policy@1");
assert.equal(hostAdapter.preferredModelId("judge", ["dsh-policy@1"], ""), "");
const aliasContext = hostAdapter.acceptContextMessage({
  source: parentWindow,
  origin: "http://localhost:4173",
  data: {
    type: "dsh.context",
    api_base: "/api/ecology-evolution",
    capability_token: "secret-token-2",
    models: [{model_id: "dsh-shared@1", roles: ["strategy", "review"]}],
  },
});
assert.equal(aliasContext.accepted, true);
assert.equal(hostAdapter.preferredModelId("propose", ["dsh-shared@1"], ""), "dsh-shared@1");
assert.equal(hostAdapter.preferredModelId("judge", ["dsh-shared@1"], ""), "dsh-shared@1");
const providerModelContext = hostAdapter.acceptContextMessage({
  source: parentWindow,
  origin: "http://localhost:4173",
  data: {
    type: "dsh.context",
    api_base: "/api/ecology-evolution",
    capability_token: "secret-token-provider",
    models: [{model_id: "newapi/glm-5.2", provider: "newapi", model: "glm-5.2", aliases: ["glm-5.2"]}],
  },
});
assert.equal(providerModelContext.accepted, true);
assert.equal(providerModelContext.context.models[0].provider, "newapi");
assert.deepEqual(Array.from(providerModelContext.context.models[0].aliases), ["glm-5.2"]);
const uppercaseRoleContext = hostAdapter.acceptContextMessage({
  source: parentWindow,
  origin: "http://localhost:4173",
  data: {
    type: "dsh.context",
    api_base: "/api/ecology-evolution",
    capability_token: "secret-token-3",
    models: [{model_id: "dsh-uppercase@1", roles: ["PROPOSE", "JUDGE"]}],
  },
});
assert.equal(uppercaseRoleContext.accepted, true);
assert.equal(hostAdapter.preferredModelId("propose", ["dsh-uppercase@1"], ""), "dsh-uppercase@1");
assert.equal(hostAdapter.preferredModelId("judge", ["dsh-uppercase@1"], ""), "dsh-uppercase@1");
const rolelessContext = hostAdapter.acceptContextMessage({
  source: parentWindow,
  origin: "http://localhost:4173",
  data: {
    type: "dsh.context",
    api_base: "/api/ecology-evolution",
    capability_token: "secret-token-4",
    models: [{model_id: "dsh-roleless@1"}],
  },
});
assert.equal(rolelessContext.accepted, true);
assert.equal(hostAdapter.preferredModelId("propose", ["dsh-roleless@1"], ""), "dsh-roleless@1");
assert.equal(hostAdapter.preferredModelId("judge", ["dsh-roleless@1"], ""), "dsh-roleless@1");

await hostAdapter.request("/health");
assert.equal(hostFetches[0].url, "/api/ecology-evolution/health");
assert.equal(hostFetches[0].options.headers.Authorization, "Bearer secret-token-4");

// Focused runtime contract for the shared DSH model directory.  This keeps
// the model-selection behavior testable without requiring a browser DOM.
const modelNode = () => ({value: "", textContent: "", className: "", disabled: false, innerHTML: "", hidden: false, addEventListener() {}});
const modelSandbox = {
  console, URL, URLSearchParams, AbortController, setTimeout, clearTimeout,
  window: {location: {search: ""}, setTimeout, clearTimeout},
  document: {querySelector: modelNode, querySelectorAll: () => []},
  EcologyDSHHost: {getPublicContext: () => ({apiBase: "/api", models: []}), preferredModelId: () => "", request: () => Promise.resolve({})},
};
vm.createContext(modelSandbox);
vm.runInContext(read("assets/js/core.js"), modelSandbox);
const actualReconcileRunSelection = modelSandbox.reconcileVisibleRunSelection;
vm.runInContext(read("assets/js/catalog.js"), modelSandbox);
// Workspace hydration has its own loading/race suite. Keep these legacy
// control fixtures focused on the command and monitor requests they model.
modelSandbox.ensureWorkspaceData = async () => true;
vm.runInContext(read("assets/js/data.js"), modelSandbox);
vm.runInContext(read("assets/js/commands.js"), modelSandbox);
vm.runInContext(read("assets/js/render_shell.js"), modelSandbox);
vm.runInContext(read("assets/js/render_candidates.js"), modelSandbox);
vm.runInContext(read("assets/js/render_process.js"), modelSandbox);


// Exercise the real form listener, not only the lower-level createRun helper.
// This catches omissions between the visible choice and the outgoing request.
const submitNodes = new Map();
const submitNode = (selector) => {
  if (!submitNodes.has(selector)) {
    submitNodes.set(selector, {...modelNode(), listeners: {}, setAttribute() {}, classList: {add() {}, toggle() {}},
      addEventListener(type, listener) { this.listeners[type] = listener; }});
  }
  return submitNodes.get(selector);
};
const formSubmissionSandbox = {
  console, URL, URLSearchParams, AbortController, setTimeout, clearTimeout,
  FormData: class { constructor(form) { this.fields = form.fields; } get(key) { return this.fields[key] ?? null; } },
  window: {location: {search: ""}, setTimeout, clearTimeout, addEventListener() {}},
  document: {querySelector: submitNode, querySelectorAll: () => []},
  EcologyDSHHost: {getPublicContext: () => ({apiBase: "/api", models: []})},
};
vm.createContext(formSubmissionSandbox);
for (const name of ["core", "catalog", "data", "commands", "render_shell", "render_process", "render_training", "render_candidates"]) {
  vm.runInContext(read(`assets/js/${name}.js`), formSubmissionSandbox);
}
vm.runInContext(read("app.js").split("\n    bindEvents();")[0], formSubmissionSandbox);
const capacityFailure = Object.assign(new Error(
  "insufficient causal cohort capacity: required=350, available=683, max_generations=4"
), {errorCode: "insufficient_cohort_capacity", status: 400});
assert.match(formSubmissionSandbox.errorMessage(capacityFailure), /最多支持 4 轮/);
assert.match(formSubmissionSandbox.errorMessage(capacityFailure), /不会自动缩减样本量/);
assert.doesNotMatch(formSubmissionSandbox.errorMessage(capacityFailure), /服务请求失败/);
// Existing servers retain their old generic code; the known numeric capacity
// message still has the same actionable interpretation.
assert.match(formSubmissionSandbox.errorMessage({...capacityFailure,
  message: capacityFailure.message, errorCode: "invalid_request"}), /最多支持 4 轮/);
assert.match(formSubmissionSandbox.errorMessage({message: "unexpected failure",
  errorCode: "invalid_request", status: 400}), /错误码 invalid_request；HTTP 400/);
assert.doesNotMatch(formSubmissionSandbox.errorMessage({message: "unexpected failure",
  errorCode: "secret=private-value", status: 400}), /private-value/);
const quickCapacity = {cohort_reuse_policy: "training_replay_fresh_epoch_holdouts@1",
  planned_generations: 5, max_feasible_generations: 4, sufficient: false,
  required_unique_origins: 350, available_eligible_origins: 683};
assert.match(formSubmissionSandbox.cohortCapacityLabel(quickCapacity), /计划 5 轮.*最多支持 4 轮/);
assert.equal(formSubmissionSandbox.cohortCapacityLabel({...quickCapacity,
  rejection_reason: "选择留出窗口不足"}), "选择留出窗口不足");
const jointCapacity = {...quickCapacity, cohort_reuse_policy: "fresh_training_race_and_confirmation@1",
  required_unique_origins: 425, max_feasible_generations: 3};
assert.match(formSubmissionSandbox.cohortCapacityLabel(jointCapacity), /计划 5 轮.*目标成熟.*最多支持 3 轮/);
assert.match(formSubmissionSandbox.cohortCapacityLabel({...jointCapacity, sufficient: true,
  planned_generations: 3, required_unique_origins: 255}), /255 个不同起点.*各阶段与各轮使用新时点/);
formSubmissionSandbox.readiness = () => [{ready: true}];
formSubmissionSandbox.hasCapability = () => true;
formSubmissionSandbox.showToast = () => {};
submitNode("#evaluator-id").value = "greenhouse_multihorizon_time_forward@4";
formSubmissionSandbox.state.catalog.evaluators = [{id: "greenhouse_multihorizon_time_forward@4", prediction_cells_per_origin: 9}];
formSubmissionSandbox.renderAll = () => {};
formSubmissionSandbox.adoptCreatedRun = async (data) => data.projection;
formSubmissionSandbox.bindEvents();
const submittedFields = {
  autonomous_mode: "true", model_workflow: "research_compile_evolve@1",
  dataset_id: "agc_cucumber_2018", episode_id: "episode-a",
  prediction_model_id: "greenhouse-baseline-aligned-ridge@1",
  evaluator_id: "greenhouse_multihorizon_time_forward@3",
  strategy_model_id: "provider/strategy", review_model_id: "provider/review",
  max_generations: "1", candidates_per_generation: "4", max_candidates: "4",
  formal_origin_count: "144", local_batch_origin_count: "72", max_local_edits_per_batch: "1",
  selection_holdout_origin_count: "169", candidate_concurrency: "4",
  sample_agent_batch_size: "9", sample_concurrency: "64", fixed_seed: "on",
};
const submittedRequests = [];
formSubmissionSandbox.state.activeRun = {id: "run:old-history", status: "completed"};
formSubmissionSandbox.request = async (route, options) => {
  submittedRequests.push({route, body: options.body, creationState: formSubmissionSandbox.state.createStatus.state});
  if (route === "/model-preflight") {
    assert.equal(formSubmissionSandbox.pendingCreateStatus().state, "preflight");
    formSubmissionSandbox.renderProcessSummary(null);
    assert.match(submitNode("#process-summary").innerHTML, /模型能力预检中/);
    assert.doesNotMatch(submitNode("#process-summary").innerHTML, /正在创建持久化运行/);
    return {passed: true, receipts: []};
  }
  return {projection: {id: "run:form-created", status: "running"}};
};
const submittedRun = await submitNode("#start-form").listeners.submit({preventDefault() {}, currentTarget: {fields: submittedFields}});
assert.equal(submittedRun.id, "run:form-created");
assert.deepEqual(submittedRequests.map((request) => request.route), ["/model-preflight", "/runs"]);
assert.deepEqual(submittedRequests.map((request) => request.creationState), ["preflight", "submitting"]);
for (const request of submittedRequests) {
  assert.equal(request.body.prediction_model_id, undefined);
  assert.equal(request.body.targeted_search_policy, undefined, "standard creation must not silently opt into cellwise search");
  assert.equal(request.body.prediction_selection_policy, "model_during_run@1");
  assert.equal(request.body.evaluator_id, undefined);
  assert.equal(request.body.search_guard_policy, "practical_delta_cell_noninferiority_paired_blocks@1");
  assert.equal(request.body.require_model_contract_preflight, true);
}
submittedRequests.length = 0;
const sharedRoleModel = {id: "provider/shared-model"};
const submittedSharedModelRun = await submitNode("#start-form").listeners.submit({preventDefault() {}, currentTarget: {
  fields: {...submittedFields, strategy_model_id: sharedRoleModel.id, review_model_id: sharedRoleModel.id},
}});
assert.equal(submittedSharedModelRun.id, "run:form-created");
assert.deepEqual(submittedRequests.map(request => request.route), ["/model-preflight", "/runs"]);
for (const request of submittedRequests) {
  assert.equal(request.body.strategy_model_id, sharedRoleModel.id);
  assert.equal(request.body.review_model_id, sharedRoleModel.id);
  assert.equal(request.body.require_model_contract_preflight, true);
}
submittedRequests.length = 0;
const submittedGuidedRun = await submitNode("#start-form").listeners.submit({preventDefault() {}, currentTarget: {
  fields: {...submittedFields, experiment_mode: "evidence_guided", formal_origin_count: "999", local_batch_origin_count: "999", targeted_search_policy: "cellwise_training_evidence@1"}
}});
assert.equal(submittedGuidedRun.id, "run:form-created");
assert.deepEqual(submittedRequests.map((request) => request.route), ["/model-preflight", "/runs"]);
for (const request of submittedRequests) {
  assert.equal(request.body.targeted_search_policy, "cellwise_training_evidence@1");
  assert.equal(request.body.optimization_protocol, "joint_plugin_epoch@2");
  assert.equal(request.body.optimization_schedule.schema_version, "ecologyrsi-dsh.joint-plugin-epoch-schedule/2");
  assert.equal(request.body.optimization_schedule.screening_origin_count, 10);
  assert.equal(request.body.optimization_schedule.formal_origin_count_per_finalist, 35);
  assert.equal(request.body.optimization_schedule.local_batch_origin_count, 25);
  assert.equal(request.body.optimization_schedule.selection_holdout_origin_count, 50);
  assert.equal(request.body.optimization_schedule.max_local_edits_per_batch, 2);
}
const savedScheduleFromControls = formSubmissionSandbox.optimizationScheduleFromControls;
formSubmissionSandbox.optimizationScheduleFromControls = () => submittedRequests[0].body.optimization_schedule;
const guidedCapacityRequest = formSubmissionSandbox.evolutionCapacityRequest().body;
assert.deepEqual(Object.keys(guidedCapacityRequest).sort(), ["dataset_id", "episode_id", "optimization_schedule", "planned_generations"]);
assert.equal(guidedCapacityRequest.optimization_schedule.local_batch_origin_count, 25);
formSubmissionSandbox.optimizationScheduleFromControls = savedScheduleFromControls;
submittedRequests.length = 0;
formSubmissionSandbox.request = async (route) => { submittedRequests.push(route); return {passed: false}; };
const failedPreflightRun = await submitNode("#start-form").listeners.submit({preventDefault() {}, currentTarget: {fields: submittedFields}});
assert.equal(failedPreflightRun, null);
assert.deepEqual(submittedRequests, ["/model-preflight"]);
assert.equal(formSubmissionSandbox.state.activeRun.id, "run:old-history");
assert.equal(formSubmissionSandbox.state.createStatus.state, "failed");
assert.equal(formSubmissionSandbox.pendingCreateStatus(), null);
assert.equal(formSubmissionSandbox.createDisplayStatus().state, "failed");
assert.equal(formSubmissionSandbox.createPhaseLabel(formSubmissionSandbox.createDisplayStatus()), "本次启动未成功");
assert.equal(formSubmissionSandbox.state.createStatus.runId, null);

submittedRequests.length = 0;
formSubmissionSandbox.request = async (route) => {
  submittedRequests.push(route);
  return {passed: false, receipts: [{passed: false,
    identity: {provider_id: "pjlab", model_id: "deepseek-v4-flash-0731", stage: "generation.reflect"},
    failure: {code: "structured_child_tool_protocol_error"}}]};
};
const protocolRejectedRun = await submitNode("#start-form").listeners.submit({preventDefault() {}, currentTarget: {fields: submittedFields}});
assert.equal(protocolRejectedRun, null);
assert.deepEqual(submittedRequests, ["/model-preflight"]);
assert.match(formSubmissionSandbox.state.commandError, /pjlab\/deepseek-v4-flash-0731/);
assert.match(formSubmissionSandbox.state.commandError, /工具调用返回为普通文本/);
assert.match(formSubmissionSandbox.state.commandError, /由服务方修复工具调用解析/);
assert.match(formSubmissionSandbox.state.commandError, /尚未创建运行/);

assert.equal(formSubmissionSandbox.pendingCreateStatus(), null);
// A lost preflight response must be reconciled before creating exactly once.
const recoveryRequests = [];
let preflightPolls = 0;
formSubmissionSandbox.request = async (route, options) => {
  recoveryRequests.push({route, body: options.body});
  if (route === "/model-preflight" && !options.body.check_only) {
    const error = new Error("请求超时。"); error.name = "AbortError"; throw error;
  }
  if (route === "/model-preflight") {
    preflightPolls += 1;
    return preflightPolls === 1 ? {passed: false, pending: true} : {passed: true};
  }
  return {projection: {id: "run:preflight-recovered", status: "running"}};
};
const recoveredPreflightRun = await submitNode("#start-form").listeners.submit({preventDefault() {}, currentTarget: {fields: submittedFields}});
assert.equal(recoveredPreflightRun.id, "run:preflight-recovered");
assert.equal(recoveryRequests.filter(r => r.route === "/runs").length, 1);
assert.equal(recoveryRequests.filter(r => r.body.check_only === true).length, 2);
assert.equal(new Set(recoveryRequests.map(r => r.body.idempotency_key)).size, 1);
// Missing/failed receipts never permit creation, including after a timeout.
recoveryRequests.length = 0;
formSubmissionSandbox.request = async (route, options) => {
  recoveryRequests.push(route);
  if (!options.body.check_only) { const e = new Error("请求超时。"); e.name = "AbortError"; throw e; }
  return {passed: false, pending: false};
};
const unconfirmedPreflightRun = await submitNode("#start-form").listeners.submit({preventDefault() {}, currentTarget: {fields: submittedFields}});
assert.equal(unconfirmedPreflightRun, null);
assert.equal(recoveryRequests.includes("/runs"), false);
assert.match(formSubmissionSandbox.state.commandError, /模型预检未完成.*尚未提交创建请求/);
// A delayed response for the previous form must neither enable submission
// nor replace the current schedule's capacity result.
for (const [selector, value] of Object.entries({"#dataset-id": "dataset:a", "#episode-id": "episode:a", "#max-generations": "1", "#formal-origin-count": "144", "#local-batch-origin-count": "72", "#max-local-edits-per-batch": "1", "#selection-holdout-origin-count": "169"})) {
  submitNode(selector).value = value;
}
formSubmissionSandbox.renderEvolutionCapacityState = () => {};
formSubmissionSandbox.state.cohortCapacityReport = null;
formSubmissionSandbox.state.cohortCapacityError = null;
assert.equal(formSubmissionSandbox.capacityVerificationPending(), true);
const capacityResolvers = [];
formSubmissionSandbox.request = () => new Promise((resolve) => capacityResolvers.push(resolve));
const oldCapacityRead = formSubmissionSandbox.refreshEvolutionCapacity();
submitNode("#max-generations").value = "2";
const currentCapacityRead = formSubmissionSandbox.refreshEvolutionCapacity();
capacityResolvers[0]({sufficient: false, marker: "old"});
await oldCapacityRead;
assert.equal(formSubmissionSandbox.state.cohortCapacityReport, null);
assert.equal(formSubmissionSandbox.capacityVerificationPending(), true);
capacityResolvers[1]({sufficient: true, marker: "current"});
await currentCapacityRead;
assert.equal(formSubmissionSandbox.state.cohortCapacityReport.marker, "current");
assert.equal(formSubmissionSandbox.capacityVerificationPending(), false);
submitNode("#max-generations").value = "3";
assert.equal(formSubmissionSandbox.capacityVerificationPending(), true);
formSubmissionSandbox.state.cohortCapacityError = "查询失败";
assert.equal(formSubmissionSandbox.capacityVerificationPending(), false, "failed verification remains explainable rather than permanently disabling the form");

// Untouched defaults must form an executable plan without overriding user
// input or reusing the capacity receipt for the originally rejected plan.
for (const [selector, value] of Object.entries({"#experiment-mode": "quick", "#max-generations": "5",
  "#candidates-per-generation": "4", "#max-candidates": "20", "#formal-origin-count": "100",
  "#local-batch-origin-count": "10", "#selection-holdout-origin-count": "50"})) {
  submitNode(selector).value = value;
}
submitNode("#max-candidates").setCustomValidity = () => {};
formSubmissionSandbox.state.busy = false;
formSubmissionSandbox.state.roundsManuallyEdited = false;
const defaultCapacityRequests = [];
formSubmissionSandbox.request = async (route, options) => {
  assert.equal(route, "/evolution-capacity");
  defaultCapacityRequests.push(options.body);
  return {...quickCapacity, planned_generations: options.body.planned_generations,
    sufficient: options.body.planned_generations <= 4};
};
await formSubmissionSandbox.refreshEvolutionCapacity();
assert.deepEqual(defaultCapacityRequests.map(r => r.planned_generations), [5, 4]);
assert.equal(submitNode("#max-generations").value, "4");
assert.equal(submitNode("#max-candidates").value, "16");
assert.equal(submitNode("#formal-origin-count").value, "100");
assert.equal(submitNode("#selection-holdout-origin-count").value, "50");
assert.equal(formSubmissionSandbox.state.cohortCapacityReport.sufficient, true);
assert.equal(formSubmissionSandbox.capacityVerificationPending(), false);
assert.match(formSubmissionSandbox.createRunHint(null, true, []), /默认轮数从 5 调整为 4/);
// New data can restore the normal default, but never invent more rounds.
assert.equal(formSubmissionSandbox.fitDefaultRoundsToCapacity({...quickCapacity, max_feasible_generations: 9}), true);
assert.equal(submitNode("#max-generations").value, "5");
assert.equal(formSubmissionSandbox.state.defaultRoundsAdjustment, null);
assert.equal(formSubmissionSandbox.capacityVerificationPending(), true);
assert.equal(formSubmissionSandbox.fitDefaultRoundsToCapacity({...quickCapacity, max_feasible_generations: 0}), false);
assert.equal(formSubmissionSandbox.fitDefaultRoundsToCapacity({...quickCapacity, rejection_reason: "证据不足"}), false);
// The unified joint plan uses a different cohort policy from legacy quick
// runs. Its untouched defaults must also be adjusted and freshly verified.
const jointCapacityRequests = [];
formSubmissionSandbox.request = async (route, options) => {
  assert.equal(route, "/evolution-capacity");
  jointCapacityRequests.push(options.body);
  return {...jointCapacity, planned_generations: options.body.planned_generations,
    sufficient: options.body.planned_generations <= 3};
};
await formSubmissionSandbox.refreshEvolutionCapacity();
assert.deepEqual(jointCapacityRequests.map(r => r.planned_generations), [5, 3]);
assert.ok(jointCapacityRequests.every(r => r.optimization_schedule.schema_version === "ecologyrsi-dsh.joint-plugin-epoch-schedule/2"));
assert.equal(submitNode("#max-generations").value, "3");
assert.equal(submitNode("#max-candidates").value, "12");
assert.equal(formSubmissionSandbox.state.cohortCapacityReport.sufficient, true);
assert.equal(formSubmissionSandbox.capacityVerificationPending(), false);
assert.match(formSubmissionSandbox.createRunHint(null, true, []), /默认轮数从 5 调整为 3/);
assert.equal(formSubmissionSandbox.fitDefaultRoundsToCapacity({...jointCapacity, max_feasible_generations: 9}), true);
assert.equal(submitNode("#max-generations").value, "5");
assert.equal(formSubmissionSandbox.fitDefaultRoundsToCapacity({...jointCapacity, rejection_reason: "证据不足"}), false);
// Even retyping the default while a request is in flight is an explicit choice.
formSubmissionSandbox.renderReadiness = () => {};
formSubmissionSandbox.renderParameters = () => {};
formSubmissionSandbox.scheduleEvolutionCapacityRefresh = () => {};
let resolveDefaultCapacity;
formSubmissionSandbox.request = () => new Promise(resolve => { resolveDefaultCapacity = resolve; });
const editingCapacity = formSubmissionSandbox.refreshEvolutionCapacity();
submitNode("#max-generations").listeners.input();
assert.equal(formSubmissionSandbox.state.roundsManuallyEdited, true);
resolveDefaultCapacity(jointCapacity);
await editingCapacity;
assert.equal(submitNode("#max-generations").value, "5");
assert.equal(formSubmissionSandbox.state.cohortCapacityReport.sufficient, false);
assert.equal(formSubmissionSandbox.state.defaultRoundsAdjustment, null);


// Render the actual funnel with the public paired-tool projection, retaining
// measured zero separately from absent/unverified evidence and old contracts.
const pairedToolFunnelFixture = {registered_candidates: 4, budget_unallocated_candidates: 0, applied_edits: 1,
  stages: {}, failure_observations: {}, prediction_contribution: {
    policy: "prediction_contribution@2", enabled: true, scope: "training_only",
    latest_training_summary: {cells: []}, paired_tool_effects: [
      {comparison_id: "comparison:measured", status: "measured", changed_cells: ["co2_concentration@6h"],
        compared_cells: 9, scientific_gain_established: false, cohort_digest: "fixture-cohort"},
      {comparison_id: "comparison:unchanged", status: "measured", changed_cells: [], compared_cells: 9,
        scientific_gain_established: false, cohort_digest: "fixture-cohort"},
      {comparison_id: "comparison:missing-inputs", status: "unverified", reason: "complete_matching_inputs_required"}
    ]
  }};
const pairedToolMarkup = modelSandbox.renderEvidenceFunnel({evolution_funnel: pairedToolFunnelFixture});
assert.match(pairedToolMarkup, /原始预测工具的父子变化/);
assert.match(pairedToolMarkup, /原始工具有变化 1 \/ 9 格/);
assert.match(pairedToolMarkup, /变化格：.* · 6h/);
assert.match(pairedToolMarkup, /原始工具有变化 0 \/ 9 格/);
assert.match(pairedToolMarkup, /所比较格未发现原始工具输出变化/);
assert.match(pairedToolMarkup, /同输入证据不足/);
assert.match(pairedToolMarkup, /缺少完整且一致的输入或原始工具输出证据/);
assert.match(pairedToolMarkup, /数值变化不代表性能提升/);
assert.match(pairedToolMarkup, /最终保留仍按本运行原有的配对与轮末门槛决定/);
for (const [reason, label] of [
  ["same_cohort_required", "缺少同一训练样本组"],
  ["same_diagnostic_version_required", "诊断版本不一致或缺失"],
  ["cell_coverage_mismatch", "目标与时距覆盖不一致"]
]) {
  const legacyMarkup = modelSandbox.renderEvidenceFunnel({evolution_funnel: {...pairedToolFunnelFixture,
    prediction_contribution: {policy: "prediction_contribution@1", enabled: true,
      paired_tool_effects: [{status: "unverified", reason}]}}});
  assert.ok(legacyMarkup.includes(label));
  assert.doesNotMatch(legacyMarkup, /原始工具有变化/);
}
for (const pair of [{status: "measured"}, {status: "measured", changed_cells: []},
  {status: "measured", compared_cells: 9}, {status: "measured", compared_cells: null, changed_cells: []}]) {
  const missingMarkup = modelSandbox.renderEvidenceFunnel({evolution_funnel: {...pairedToolFunnelFixture,
    prediction_contribution: {policy: "prediction_contribution@2", enabled: true, paired_tool_effects: [pair]}}});
  assert.match(missingMarkup, /同输入证据不足/);
  assert.doesNotMatch(missingMarkup, /原始工具有变化/);
}
const noPairMarkup = modelSandbox.renderEvidenceFunnel({evolution_funnel: {...pairedToolFunnelFixture,
  prediction_contribution: {policy: "prediction_contribution@1", enabled: true}}});
assert.match(noPairMarkup, /尚无可核验的父子比较记录/);
assert.doesNotMatch(noPairMarkup, /原始工具有变化/);
const noDiagnosticsMarkup = modelSandbox.renderEvidenceFunnel({evolution_funnel: {...pairedToolFunnelFixture,
  prediction_contribution: undefined}});
assert.doesNotMatch(noDiagnosticsMarkup, /原始预测工具的父子变化/);
const unsafePairMarkup = modelSandbox.renderEvidenceFunnel({evolution_funnel: {...pairedToolFunnelFixture,
  prediction_contribution: {paired_tool_effects: [{status: "measured", comparison_id: '<img src=x onerror="bad()">',
    compared_cells: 9, changed_cells: ["<script>bad()</script>"]}]}}});
assert.doesNotMatch(unsafePairMarkup, /<img|<script>/);
assert.match(unsafePairMarkup, /&lt;script&gt;/);
console.log("paired raw tool evidence renderer: ok");

assert.doesNotMatch(app, /state\.usingDemo|loadDemo|submitIntervention|answerExpertConsultation|renderTrainingAssets/);
assert.doesNotMatch(html, /workspace-collaboration|training-assets-table|intervention-form/);

// Integration regression: the real create command, selection command and
// incremental sample renderers share one display identity. Deferred history
// reads must not repaint a new attempt, including its error and finally paths.
function sampleCreationRaceFixture() {
  const nodes = new Map();
  const node = (selector) => {
    if (!nodes.has(selector)) {
      nodes.set(selector, {...modelNode(), setAttribute() {}, classList: {add() {}, toggle() {}}});
    }
    return nodes.get(selector);
  };
  const sandbox = {
    console, URL, URLSearchParams, AbortController, setTimeout, clearTimeout,
    window: {location: {search: ""}, setTimeout, clearTimeout},
    document: {querySelector: node, querySelectorAll: () => [], getElementById: (id) => node("#" + id)},
    EcologyDSHHost: {getPublicContext: () => ({apiBase: "/api", models: []})},
  };
  vm.createContext(sandbox);
  for (const name of ["core", "catalog", "data", "commands", "render_shell", "render_process", "render_candidates"]) {
    vm.runInContext(read(`assets/js/${name}.js`), sandbox);
  }
  const history = sandbox.normalizeRun({run_id: "run:history-race", status: "completed",
    execution_progress: {last_evidence_candidate_id: "candidate:history-race"},
    candidates: [{candidate_id: "candidate:history-race", status: "evaluated",
      inference_trace: {rows: [{sample_id: "projection-history", target: "projection-history", predicted: 11}]}}]});
  const payload = (marker) => ({schema_version: "ecologyrsi-dsh.browser-sample-results/1",
    run_id: history.id, candidate_id: history.candidates[0].id,
    offset: 0, limit: 25, total: 1, complete: true, rows: [
      {sample_id: marker, target: marker, predicted: 22, observed: 23, sample_status: "succeeded"}
    ]});
  sandbox.state.activeRun = history;
  sandbox.state.runs = [history];
  sandbox.state.workspace = "process";
  sandbox.state.catalog.evaluators = [{id: "fixture-evaluator", prediction_cells_per_origin: 9}];
  node("#evaluator-id").value = "fixture-evaluator";
  sandbox.state.selectedCandidateId = history.candidates[0].id;
  sandbox.resetCandidateSamples(history.id, history.candidates[0].id);
  const cachedPage = sandbox.normalizeCandidateSamplePage(payload("cached-history"), history.id, history.candidates[0].id, 0, 25);
  sandbox.state.candidateSamplePage = cachedPage;
  sandbox.hasCapability = () => true;
  const toasts = [];
  sandbox.showToast = (message) => { toasts.push(message); };
  sandbox.ensureWorkspaceData = async () => true;
  sandbox.startRunMonitor = () => false;
  let renders = 0;
  sandbox.renderAll = () => {
    renders += 1;
    sandbox.renderProcessSummary(sandbox.displayedRun());
    sandbox.renderCandidateSampleViews();
  };
  sandbox.renderAll();
  assert.match(node("#candidate-samples-table").innerHTML, /cached-history/);
  const createPayload = {dataset_id: "dataset:fixture", episode_id: "episode:fixture",
    strategy_model_id: "fixture/strategy", review_model_id: "fixture/review", rounds: 1,
    max_candidates: 4, candidates_per_generation: 4, candidate_concurrency: 1,
    sample_agent_batch_size: 9, sample_concurrency: 4};
  return {sandbox, node, history, payload, cachedPage, createPayload, toasts, renders: () => renders};
}
function deferredSampleRace() {
  let resolve, reject;
  const promise = new Promise((onResolve, onReject) => { resolve = onResolve; reject = onReject; });
  return {promise, resolve, reject};
}
for (const timing of ["preflight", "failed", "failed-403", "failed-503", "return-history"]) {
  const fixture = sampleCreationRaceFixture();
  const {sandbox, node, history, payload, cachedPage, createPayload} = fixture;
  const oldSamples = deferredSampleRace();
  const preflight = deferredSampleRace();
  const paths = [];
  sandbox.request = (route, options) => {
    paths.push(route);
    if (route.includes("/samples?")) { return oldSamples.promise; }
    if (route === "/model-preflight") {
      assert.equal(options.timeout, 500000, "client must outlast the 490 s native preflight proxy deadline");
      return preflight.promise;
    }
    if (route.endsWith("?view=overview")) { return Promise.resolve({projection: history}); }
    throw new Error("Unexpected request: " + route);
  };
  const pendingSamples = sandbox.loadCandidateSamples(0, {force: true});
  assert.equal(sandbox.state.candidateSampleRefreshing, true);
  const pendingCreate = sandbox.createRun(createPayload);
  assert.equal(sandbox.createDisplayStatus().state, "preflight");
  assert.equal(sandbox.state.candidateSampleRefreshing, false);
  assert.equal(sandbox.state.candidateSamplePage, cachedPage, "new creation preserves history cache");
  assert.equal(await sandbox.loadCandidateSamples(0, {force: true}), false);
  assert.equal(await sandbox.refreshCandidateSamples({force: true}), false);
  assert.equal(paths.filter((route) => route.includes("/samples?")).length, 1);
  assert.doesNotMatch(node("#candidate-samples-table").innerHTML, /cached-history|projection-history/);
  sandbox.renderCandidates();
  assert.doesNotMatch(node("#candidate-table").innerHTML, /candidate:history-race/);
  assert.equal(sandbox.state.candidateSamplePage, cachedPage, "hidden candidate workspace preserves selection and cache");
  if (timing !== "preflight") {
    preflight.resolve({passed: false});
    assert.equal(await pendingCreate, null);
    assert.equal(sandbox.createDisplayStatus().state, "failed");
    assert.match(node("#process-summary").innerHTML, /本次启动未成功/);
  }
  let freshSamples, pendingFreshSamples;
  if (timing === "return-history") {
    assert.equal(await sandbox.selectRun(history.id, false), true);
    assert.equal(sandbox.createDisplayStatus(), null);
    assert.match(node("#candidate-samples-table").innerHTML, /cached-history/);
    freshSamples = deferredSampleRace();
    sandbox.request = () => freshSamples.promise;
    pendingFreshSamples = sandbox.loadCandidateSamples(0, {force: true});
    assert.equal(sandbox.state.candidateSampleRefreshing, true);
  }
  const rendersBeforeCompletion = fixture.renders();
  const toastsBeforeCompletion = fixture.toasts.length;
  if (timing === "failed-403" || timing === "failed-503") {
    oldSamples.reject(Object.assign(new Error("obsolete sample response"), {status: timing === "failed-403" ? 403 : 503}));
  } else { oldSamples.resolve(payload("late-history")); }
  assert.equal(await pendingSamples, false);
  assert.equal(fixture.renders(), rendersBeforeCompletion);
  assert.equal(fixture.toasts.length, toastsBeforeCompletion);
  assert.equal(sandbox.state.candidateSamplePage, cachedPage);
  assert.equal(sandbox.state.candidateSamplePermissionDenied, false);
  assert.equal(sandbox.state.candidateSampleError, null);
  assert.doesNotMatch(node("#candidate-samples-table").innerHTML, /late-history/);
  if (timing === "preflight") {
    preflight.resolve({passed: false});
    assert.equal(await pendingCreate, null);
  }
  if (timing === "return-history") {
    assert.equal(sandbox.state.candidateSampleRefreshing, true, "stale finally must not finish a fresh request");
    freshSamples.resolve(payload("fresh-history"));
    assert.equal(await pendingFreshSamples, true);
    assert.match(node("#candidate-samples-table").innerHTML, /fresh-history/);
    assert.equal(sandbox.state.candidateSampleRefreshing, false);
  } else {
    sandbox.renderCandidateSampleViews();
    assert.doesNotMatch(node("#candidate-samples-table").innerHTML, /history/);
    assert.doesNotMatch(node("#candidate-samples-table").innerHTML, /history/);
    assert.equal(sandbox.state.activeRun, history);
    assert.equal(await sandbox.selectRun(history.id, false), true);
    assert.equal(sandbox.createDisplayStatus(), null);
    assert.match(node("#candidate-samples-table").innerHTML, /cached-history/);
    sandbox.request = async () => payload("fresh-history");
    assert.equal(await sandbox.loadCandidateSamples(0, {force: true}), true);
    assert.match(node("#candidate-samples-table").innerHTML, /fresh-history/);
  }
  assert.equal(paths.includes("/runs"), false, "failed preflight never creates a run");
}
// Independently exercise the host epoch and displayed identity guards, even
// when a caller has not yet reset the sample selection/request counter.
for (const changedIdentity of ["host-context", "run", "candidate"]) {
  const {sandbox, node, payload, cachedPage} = sampleCreationRaceFixture();
  const samples = deferredSampleRace();
  sandbox.request = () => samples.promise;
  const pendingSamples = sandbox.loadCandidateSamples(0, {force: true});
  let incrementalRenders = 0;
  sandbox.renderCandidateSampleViews = () => { incrementalRenders += 1; };
  if (changedIdentity === "host-context") { sandbox.state.contextEpoch += 1; }
  if (changedIdentity === "run") { sandbox.state.activeRun = {...sandbox.state.activeRun, id: "run:replacement"}; }
  if (changedIdentity === "candidate") { sandbox.state.selectedCandidateId = "candidate:replacement"; }
  samples.resolve(payload("obsolete-identity"));
  assert.equal(await pendingSamples, false);
  assert.equal(sandbox.state.candidateSamplePage, cachedPage);
  assert.equal(incrementalRenders, 0, "stale finally must not redraw another display context");
  assert.doesNotMatch(node("#candidate-samples-table").innerHTML, /obsolete-identity/);
}
console.log("new creation / late historical samples / explicit history return: ok");


await import("./independent-evaluation.mjs");
await import("./workspace-loading.mjs");
await import("./lean_ui.test.mjs");
console.log("ecology_evolution current workflow smoke: ok");
