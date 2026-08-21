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
  "assets/js/demo.js",
  "assets/js/catalog.js",
  "assets/js/data.js",
  "assets/js/commands.js",
  "assets/js/render_shell.js",
  "assets/js/render_training_trace.js",
  "assets/js/render_training.js",
  "assets/js/render_process.js",
  "assets/js/render_candidates.js",
  "assets/js/render_collaboration.js",
  "app.js",
];
const app = scripts.map(read).join("\n");
const catalogSource = read("assets/js/catalog.js");
const dataSource = read("assets/js/data.js");
const commandsSource = read("assets/js/commands.js");

const hostMessages = [];
const hostFetches = [];
const parentWindow = {
  postMessage(message, targetOrigin) { hostMessages.push({message, targetOrigin}); },
};
const hostWindow = {
  location: new URL("http://localhost:4173/plugins/ecology/evolution/?api=/api"),
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

assert.equal(hostAdapter.postReady(), true);
assert.match(manifest.version, /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/);
assert.equal(hostMessages[0].targetOrigin, "http://localhost:4173");
assert.equal(hostMessages[0].message.context_protocol, "ecology-evolution.host-context/1");
assert.equal(hostMessages[0].message.version, manifest.version);
assert.deepEqual(
  Array.from(hostMessages[0].message.supported_api_bases),
  ["/api", "/api/v1", "/api/ecology-evolution", "/api/ecology-evolution/v1"],
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
    capabilities: ["evolution.projection.read", "evolution.run.advance"],
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
vm.runInContext(read("assets/js/catalog.js"), modelSandbox);
vm.runInContext(read("assets/js/data.js"), modelSandbox);
vm.runInContext(read("assets/js/commands.js"), modelSandbox);
vm.runInContext(read("assets/js/render_shell.js"), modelSandbox);
vm.runInContext(read("assets/js/render_candidates.js"), modelSandbox);
vm.runInContext(read("assets/js/render_process.js"), modelSandbox);
vm.runInContext(read("assets/js/render_collaboration.js"), modelSandbox);

// Hidden automatic bindings must not inherit the first compatible evaluator
// from catalog sort order. New greenhouse runs use v2, while toy runs retain
// their own registered default.
const automaticBindingQuerySelector = modelSandbox.document.querySelector;
const automaticBindingCatalog = modelSandbox.state.catalog;
const automaticBindingNodes = {
  "#dataset-id": {value: "agc_cucumber_2018"},
  "#prediction-model-id": {value: "greenhouse-exogenous-ridge@1"},
  "#evaluator-id": {value: "greenhouse_multihorizon_time_forward@1"},
};
modelSandbox.document.querySelector = (selector) => automaticBindingNodes[selector] || modelNode();
modelSandbox.state.catalog = {
  datasets: [{id: "agc_cucumber_2018"}],
  prediction_models: [{id: "greenhouse-exogenous-ridge@1"}],
  evaluators: [
    {id: "greenhouse_multihorizon_time_forward@1", dataset_ids: ["agc_cucumber_2018"], prediction_model_ids: ["greenhouse-exogenous-ridge@1"]},
    {id: "greenhouse_multihorizon_time_forward@2", dataset_ids: ["agc_cucumber_2018"], prediction_model_ids: ["greenhouse-exogenous-ridge@1"]},
    {id: "greenhouse_time_forward@1", dataset_ids: ["agc_cucumber_2018"], prediction_model_ids: ["greenhouse-exogenous-ridge@1"]},
  ],
};
modelSandbox.alignPredictionBinding();
assert.equal(automaticBindingNodes["#evaluator-id"].value, "greenhouse_multihorizon_time_forward@2");

automaticBindingNodes["#dataset-id"].value = "generated-toy-series@1";
automaticBindingNodes["#prediction-model-id"].value = "toy-rolling-water@1";
automaticBindingNodes["#evaluator-id"].value = "greenhouse_multihorizon_time_forward@2";
modelSandbox.state.catalog = {
  datasets: [{id: "generated-toy-series@1"}],
  prediction_models: [{id: "toy-rolling-water@1"}],
  evaluators: [
    {id: "toy_time_forward@1", dataset_ids: ["generated-toy-series@1"], prediction_model_ids: ["toy-rolling-water@1"]},
  ],
};
modelSandbox.alignPredictionBinding();
assert.equal(automaticBindingNodes["#evaluator-id"].value, "toy_time_forward@1");
modelSandbox.document.querySelector = automaticBindingQuerySelector;
modelSandbox.state.catalog = automaticBindingCatalog;

const crossCohortProjection = modelSandbox.normalizeRun({
  run_id: "run:cross-cohort", status: "completed", best_candidate_id: "candidate:retained",
  best_observed_candidate_id: "candidate:raw-high", best_observed_score: 0.95,
  candidates: [
    {id: "candidate:retained", parent_id: "candidate:raw-high", generation: 2, status: "accepted", score: 0.61},
    {id: "candidate:raw-high", generation: 1, status: "evaluated", score: 0.95},
  ],
  trajectory: [
    {generation: 1, candidate_id: "candidate:raw-high", evaluation_cohort_digest: "sha256:window-a", score: 0.95, best_observed_score: 0.95, incumbent_score: 0.55, best_score: 0.50},
    {generation: 2, candidate_id: "candidate:retained", evaluation_cohort_digest: "sha256:window-b", score: 0.61, best_observed_score: 0.95, best_score: 0.61},
  ],
});
assert.match(modelSandbox.runFailureMessage({
  status: "failed",
  failure_code: "frozen_runtime_binding_drift",
  failure_reason: "FrozenRuntimeBindingDriftError [frozen_runtime_binding_drift]",
}, []), /当前配置新建进化运行/);
const consultationProjection = modelSandbox.normalizeRun({
  run_id: "run:consultation-normalize", status: "running", generation: 3,
  expert_consultations: [
    {consultation_id: "consultation:pending", status: "pending", question: "是否调整阈值？", options: ["保持", {id: "adjust", label: "调整"}], non_blocking: true},
    {id: "consultation:legacy-answer", status: "applied", question: "使用哪种窗口？", answer: {answer: "使用保守窗口", answered_by: "专家甲", selected_option: "keep", effective_generation: 3, applied_generation: 4, created_at: "2026-08-19T00:00:00Z"}},
  ],
});
assert.equal(consultationProjection.expert_consultations.length, 2);
assert.equal(consultationProjection.expert_consultations[0].status, "pending");
assert.equal(consultationProjection.expert_consultations[0].non_blocking, true);
assert.equal(consultationProjection.expert_consultations[1].status, "answered");
assert.equal(consultationProjection.expert_consultations[1].answer, "使用保守窗口");
assert.equal(consultationProjection.expert_consultations[1].answered_by, "专家甲");
assert.equal(consultationProjection.expert_consultations[1].selected_option, "keep");
assert.equal(consultationProjection.expert_consultations[1].applied_generation, 4);
const pendingConsultationHTML = modelSandbox.renderPendingConsultation(consultationProjection.expert_consultations[0], consultationProjection, false);
assert.ok(pendingConsultationHTML.includes("非阻塞 · 运行继续"));
assert.ok(pendingConsultationHTML.includes("提交专家答复"));
assert.ok(pendingConsultationHTML.includes("调整"));
modelSandbox.expertConsultationDraft(consultationProjection.id, "consultation:pending").answer = "轮询后仍应保留的草稿";
assert.ok(modelSandbox.renderPendingConsultation(consultationProjection.expert_consultations[0], consultationProjection, false).includes("轮询后仍应保留的草稿"));
const terminalConsultationHTML = modelSandbox.renderPendingConsultation(consultationProjection.expert_consultations[0], consultationProjection, true);
assert.ok(terminalConsultationHTML.includes("仍可补录专家答复"));
assert.ok(terminalConsultationHTML.includes("只归档"));
assert.equal(modelSandbox.expertConsultationRunIsTerminal({status: "released"}), true);
const answeredConsultationHTML = modelSandbox.renderAnsweredConsultation(consultationProjection.expert_consultations[1], consultationProjection, false);
assert.ok(answeredConsultationHTML.includes("专家答复"));
assert.ok(answeredConsultationHTML.includes("已在第 4 轮应用"));
assert.equal(modelSandbox.consultationUncertaintyText("scientific_assumption"), "科学假设");
assert.equal(modelSandbox.consultationUncertaintyText("data_interpretation"), "数据解释");
assert.equal(modelSandbox.consultationUncertaintyText("model_selection"), "模型选择");
assert.equal(modelSandbox.consultationUncertaintyText("tradeoff"), "权衡判断");
assert.equal(modelSandbox.consultationUncertaintyText("governance_boundary"), "治理边界");
assert.equal(modelSandbox.createRunButtonLabel({state: "background"}, true), "创建新的进化运行");
assert.equal(modelSandbox.createRunButtonLabel({state: "failed"}, true), "重新创建进化运行");
assert.equal(modelSandbox.runRetainedScore(crossCohortProjection), 0.61);
assert.equal(modelSandbox.runRawBestObservedScore(crossCohortProjection), 0.95);
assert.equal(modelSandbox.trajectoryIncumbentScore(crossCohortProjection.trajectory[0]), 0.55);
assert.equal(modelSandbox.trajectoryIncumbentScore(crossCohortProjection.trajectory[1]), 0.61);
assert.equal(Number.isNaN(modelSandbox.trajectoryIncumbentScore({best_observed_score: 0.99})), true);
assert.equal(
  modelSandbox.rawBestObservedSummary(crossCohortProjection),
  "原始最高观测（跨窗口不可直接比较）：candidate:raw-high（0.9500）",
);
assert.equal(modelSandbox.candidateOutcome(crossCohortProjection.candidates[1], crossCohortProjection).text, "原始最高观测");
const crossCohortOverview = modelSandbox.renderCandidateOverview(crossCohortProjection, crossCohortProjection.candidates);
assert.ok(crossCohortOverview.includes("当前保留得分"));
assert.ok(crossCohortOverview.includes("原始最高观测（跨窗口不可直接比较）"));
assert.equal(crossCohortOverview.includes("当前最高分"), false);
assert.equal(
  modelSandbox.generationOutcomeText("no_improvement", {
    candidates: [{selection_reason: "cohort_changed_search_parent_only"}],
  }),
  "跨窗口未比较，保留正式方案",
);
assert.equal(modelSandbox.generationOutcomeText("no_improvement", {candidates: []}), "未改善，保留原方案");
const crossCohortDelta = modelSandbox.candidateDelta(
  crossCohortProjection.candidates[0],
  crossCohortProjection.candidates,
  crossCohortProjection,
);
assert.equal(crossCohortDelta.value, null);
assert.equal(crossCohortDelta.label, "跨窗口不可直接比较");
const sameCohortRun = {
  trajectory: [
    {candidate_id: "candidate:parent", evaluation_cohort_digest: "sha256:same-window"},
    {candidate_id: "candidate:child", evaluation_cohort_digest: "sha256:same-window"},
  ],
};
const sameCohortDelta = modelSandbox.candidateDelta(
  {id: "candidate:child", parent_id: "candidate:parent", score: 0.72},
  [{id: "candidate:parent", score: 0.65}, {id: "candidate:child", parent_id: "candidate:parent", score: 0.72}],
  sameCohortRun,
);
assert.ok(Math.abs(sameCohortDelta.value - 0.07) < 1e-12);
assert.equal(sameCohortDelta.label, "相对父方案");

const trajectoryLegendNode = {innerHTML: ""};
const trajectoryChartNode = {
  innerHTML: "",
  getBoundingClientRect: () => ({width: 800}),
  parentElement: {querySelector: () => trajectoryLegendNode},
};
const trajectoryQuerySelector = modelSandbox.document.querySelector;
modelSandbox.document.querySelector = (selector) => selector === "#trajectory-chart" ? trajectoryChartNode : trajectoryQuerySelector(selector);
modelSandbox.window.innerWidth = 1200;
modelSandbox.state.activeRun = crossCohortProjection;
modelSandbox.renderTrajectory();
assert.ok(trajectoryChartNode.innerHTML.includes('class="chart-best"'));
assert.ok(trajectoryChartNode.innerHTML.includes("候选原始得分与实际晋升序列"));
assert.ok(trajectoryChartNode.innerHTML.includes("不同反馈窗口的原始得分不可直接比较"));
assert.ok(trajectoryLegendNode.innerHTML.includes("实际晋升序列"));

const legacyCrossCohortProjection = modelSandbox.normalizeRun({
  run_id: "run:legacy-cross-cohort", status: "completed", best_observed_score: 0.95,
  candidates: [
    {id: "candidate:legacy-a", generation: 1, score: 0.95},
    {id: "candidate:legacy-b", generation: 2, score: 0.60},
  ],
});
assert.equal(Object.hasOwn(legacyCrossCohortProjection.trajectory[0], "best_observed_score"), false);
modelSandbox.state.activeRun = legacyCrossCohortProjection;
modelSandbox.renderTrajectory();
assert.equal(trajectoryChartNode.innerHTML.includes('class="chart-best"'), false);
assert.ok(trajectoryLegendNode.innerHTML.includes("历史运行未记录"));
modelSandbox.document.querySelector = trajectoryQuerySelector;
modelSandbox.state.activeRun = null;
assert.equal(
  modelSandbox.supersededSampleRevisionText({
    revision: "revision:legacy",
    resume_disposition: "legacy_revision_without_checkpoint",
    completed_samples: 72,
    total_samples: 100,
    succeeded_samples: 70,
    failed_samples: 2,
  }),
  "上一修订已隔离：72 / 100 条（成功 70，失败 2；旧修订缺少可验证 checkpoint）",
);
assert.equal(modelSandbox.supersededSampleRevisionText(null), "");
assert.equal(modelSandbox.candidateScientificPass({metrics: {scientific_pass: true}}), true);
assert.equal(modelSandbox.candidateScientificPass({metrics: {scientific_pass: false}}), false);
assert.equal(modelSandbox.candidateScientificPass({passed: true, metrics: {}}), true);
assert.equal(modelSandbox.candidateScientificPass({
  passed: false,
  metrics: {judge_status: "completed", judge_model_id: "rule_judge@1", judge_accepted: false},
}), false);
assert.equal(modelSandbox.candidateScientificPass({
  passed: false,
  metrics: {judge_status: "completed", judge_model_id: "remote-judge", judge_accepted: true},
}), false);
assert.equal(modelSandbox.candidateScientificPass({
  passed: false,
  selection_reason: "judge_rejected",
  metrics: {judge_status: "completed", judge_accepted: false},
}), true);
assert.equal(modelSandbox.candidateScientificPass({
  passed: false,
  selection_reason: "scientific_gate_failed",
  metrics: {judge_status: "not_started"},
}), false);
assert.equal(modelSandbox.candidateScientificPass({
  metrics: {constraint_violations: 0, per_target_no_regression: true, skill_score: 0.1},
}), true);
assert.equal(modelSandbox.candidateScientificPass({
  passed: false,
  metrics: {judge_status: "completed", judge_accepted: false},
}), null);
assert.equal(modelSandbox.candidateScientificPass({passed: false, metrics: {}}), null);
const retryingJudge = modelSandbox.candidateJudgeDescriptor(
  {metrics: {judge_status: "unavailable", judge_accepted: false}},
  {status: "running", execution_progress: {retry_wait: {waiting: true, error_code: "generation_judges_unavailable"}}},
);
assert.equal(retryingJudge.text, "自动重试中");
assert.equal(retryingJudge.className, "pill-amber");
assert.equal(retryingJudge.state, "pending");
const unavailableJudge = modelSandbox.candidateJudgeDescriptor(
  {metrics: {judge_status: "unavailable", judge_accepted: false}},
  {status: "completed", execution_progress: {retry_wait: null}},
);
assert.equal(unavailableJudge.text, "评审不可用");
assert.equal(unavailableJudge.className, "pill-amber");
assert.equal(unavailableJudge.state, "pending");
assert.equal(
  modelSandbox.candidateJudgeDescriptor({metrics: {judge_status: "completed", judge_accepted: true}}, {}).state,
  "pass",
);
assert.equal(
  modelSandbox.candidateJudgeDescriptor({metrics: {judge_status: "completed", judge_accepted: false}}, {}).state,
  "fail",
);
assert.equal(
  modelSandbox.selectionReasonText("cohort_changed_search_parent_only"),
  "本轮窗口第一，仅作为下一轮搜索父方案",
);
assert.equal(
  modelSandbox.selectionReasonText("cohort_changed_batch_champion"),
  "历史语义：本轮窗口冠军，跨窗口未比较",
);
const executionEvidenceCandidate = {
  proposal_source: "remote_model",
  model_plan: {research: [
    {title: "回退研究来源", url: "https://legacy.example/source"},
  ]},
  algorithm_execution: {
    status: "debug_passed",
    training_authorized: true,
    algorithm_spec_digest: "sha256:algorithm-spec",
    algorithm_spec: {
      algorithm_id: "greenhouse-ridge+multihorizon-evaluator",
      algorithm_version: "registered-pipeline/1",
      adapter_id: "greenhouse-ridge",
      adapter_version: "2",
      evaluator_id: "multihorizon-evaluator",
      evaluator_version: "3",
      strategy_id: "adaptive-local",
      tool_ids: ["ridge-fit", "physical-range-check"],
      knowledge_mappings: Array.from({length: 6}, (_, index) => ({
        knowledge_id: `research-${index + 1}`,
        source_url: `https://source.example/${index + 1}`,
        decision: index === 0 ? "adopted" : "not_selected",
      })),
    },
    attempts: Array.from({length: 8}, (_, index) => ({
      phase: index < 4 ? "compile" : "debug",
      attempt: index < 4 ? index + 1 : index - 3,
      status: "passed",
      evidence: {check: `bounded-check-${index + 1}`, private_reasoning: "PRIVATE-ATTEMPT-MARKER"},
    })),
  },
  inference_trace: {
    status: "completed",
    sample_count: 40,
    evaluator_id: "multihorizon-evaluator",
    sample_execution: {
      eligible_examples: 40,
      attempted_examples: 40,
      succeeded_examples: 39,
      failed_examples: 1,
      retry_count: 3,
      repair_count: 2,
      coverage: 0.975,
      minimum_coverage: 0.8,
      coverage_pass: true,
      action_catalog: [{
        agent_decisions: [
          {role: "forecast_agent", decision: "use_registered_algorithm_prediction", private_reasoning: "PRIVATE-REASONING-MARKER"},
          {role: "constraint_critic", decision: "prediction_within_registered_range"},
          {role: "repair_agent", decision: "replace_with_bounded_persistence_fallback"},
          {role: "host_adjudicator", decision: "accept_validated_sample_prediction"},
          {role: "unused-role", decision: "MUST-NOT-RENDER-FIFTH-ROLE"},
        ],
        tool_calls: Array.from({length: 5}, (_, index) => ({tool_id: `tool-${index + 1}`, version: "1", status: "completed"})),
      }],
      failure_preview: Array.from({length: 5}, (_, index) => ({
        sample_id: `failed-sample-${index + 1}`,
        failure: {class: `failure-class-${index + 1}`},
        failure_action: "score_with_declared_fallback",
      })),
    },
    rows: [
      {sample_id: "sample-1", status: "succeeded", private_reasoning: "PRIVATE-ROW-MARKER"},
      {sample_id: "failed-sample-1", status: "failed", failure: {class: "failure-class-1"}, failure_action: "score_with_declared_fallback"},
    ],
  },
  metrics: {
    sample_execution_records: [{sample_id: "ALL-SAMPLE-RECORD-MARKER"}],
    sample_execution_trace_archive: "ALL-SAMPLE-ARCHIVE-MARKER",
  },
};
const executionEvidenceHtml = modelSandbox.renderCandidateExecutionEvidence(executionEvidenceCandidate);
for (const text of [
  "研究证据", "算法规范", "编译", "调试", "真实样本反馈",
  "远程策略模型", "97.5%", "成功", "39", "失败", "重试", "修复",
  "预测智能体", "tool-1", "failure-class-1", "完整逐样本记录在下方按页读取",
]) {
  assert.ok(executionEvidenceHtml.includes(text), `missing candidate execution evidence: ${text}`);
}
assert.equal((executionEvidenceHtml.match(/class="candidate-evidence-stage /g) || []).length, 5);
assert.equal((executionEvidenceHtml.match(/class="is-complete"/g) || []).length, 6);
assert.ok(executionEvidenceHtml.includes("research-4"));
assert.equal(executionEvidenceHtml.includes("research-5"), false);
assert.ok(executionEvidenceHtml.includes("tool-4"));
assert.equal(executionEvidenceHtml.includes("tool-5"), false);
assert.ok(executionEvidenceHtml.includes("failure-class-3"));
assert.equal(executionEvidenceHtml.includes("failure-class-4"), false);
for (const hiddenMarker of ["PRIVATE-REASONING-MARKER", "PRIVATE-ROW-MARKER", "PRIVATE-ATTEMPT-MARKER", "ALL-SAMPLE-RECORD-MARKER", "ALL-SAMPLE-ARCHIVE-MARKER", "MUST-NOT-RENDER-FIFTH-ROLE"]) {
  assert.equal(executionEvidenceHtml.includes(hiddenMarker), false, `private or unbounded evidence leaked: ${hiddenMarker}`);
}
const metricFallbackEvidence = modelSandbox.renderCandidateExecutionEvidence({
  metrics: {sample_execution: {attempted_examples: 10, succeeded_examples: 8, failed_examples: 2, coverage: 0.8}},
});
assert.ok(metricFallbackEvidence.includes("80%"));
assert.ok(metricFallbackEvidence.includes("共尝试 10"));
const mergedSummaryEvidence = modelSandbox.renderCandidateExecutionEvidence({
  inference_trace: {status: "completed", sample_execution: {coverage: 0.9}},
  metrics: {sample_execution: {attempted_examples: 10, succeeded_examples: 9, failed_examples: 1}},
});
assert.ok(mergedSummaryEvidence.includes("90%"));
assert.ok(mergedSummaryEvidence.includes("共尝试 10"));
assert.ok(mergedSummaryEvidence.includes(">9<"));
const emptyExecutionEvidence = modelSandbox.renderCandidateExecutionEvidence({});
assert.equal((emptyExecutionEvidence.match(/class="candidate-evidence-stage is-pending"/g) || []).length, 5);
assert.ok(emptyExecutionEvidence.includes("样本覆盖率"));
assert.equal(emptyExecutionEvidence.includes("0%"), false);
assert.equal(emptyExecutionEvidence.includes("undefined"), false);
assert.equal(emptyExecutionEvidence.includes("null"), false);
const explicitZeroEvidence = modelSandbox.renderCandidateExecutionEvidence({
  inference_trace: {status: "completed", sample_execution: {attempted_examples: 0, succeeded_examples: 0, failed_examples: 0, coverage: 0, coverage_pass: false}},
});
assert.ok(explicitZeroEvidence.includes("0%"));
assert.ok(explicitZeroEvidence.includes("共尝试 0"));
const legacyTraceEvidence = modelSandbox.renderCandidateExecutionEvidence({
  status: "evaluating",
  execution: {inference_trace: [{
    sample_id: "legacy-sample",
    agent_decisions: [{role: "forecast_agent", decision: "legacy_registered_prediction"}],
    tool_calls: [{tool_id: "legacy-tool", status: "completed"}],
  }]},
  metrics: {sample_execution: {attempted_examples: 1, succeeded_examples: 1, failed_examples: 0, coverage: 1}},
});
assert.ok(legacyTraceEvidence.includes("legacy_registered_prediction"));
assert.ok(legacyTraceEvidence.includes("legacy-tool"));
assert.ok(legacyTraceEvidence.includes("candidate-evidence-stage is-running"));
const skippedTraceEvidence = modelSandbox.renderCandidateExecutionEvidence({
  inference_trace: {status: "skipped", sample_count: 0, sample_execution: {}},
});
assert.ok(skippedTraceEvidence.includes("candidate-evidence-stage is-skipped"));
assert.ok(skippedTraceEvidence.includes("已跳过"));
assert.ok(skippedTraceEvidence.includes("重复候选，未重复执行"));
assert.equal((skippedTraceEvidence.match(/class="candidate-evidence-stage is-skipped"/g) || []).length, 4);
assert.ok(skippedTraceEvidence.includes("重复候选，未重复编译"));
assert.ok(skippedTraceEvidence.includes("重复候选，未重复调试"));
const pendingProjectionEvidence = modelSandbox.renderCandidateExecutionEvidence({
  inference_trace: {status: "pending", sample_count: 0, shown_count: 0, rows: []},
});
assert.ok(pendingProjectionEvidence.includes("等待逐样本执行"));
assert.equal(pendingProjectionEvidence.includes("共尝试 0"), false);
assert.equal(pendingProjectionEvidence.includes("0 个可评测样本"), false);
const liveProgressEvidence = modelSandbox.renderCandidateExecutionEvidence(
  {
    id: "candidate:live",
    status: "evaluating",
    inference_trace: {status: "pending", sample_count: 0, shown_count: 0, rows: []},
  },
  {
    status: "running",
    execution_progress: {
      current_candidate_id: "candidate:live",
      stage_progress: {
        progress_kind: "waiting",
        completed_samples: 591,
        total_samples: 7125,
        succeeded_samples: 580,
        failed_samples: 11,
      },
    },
  },
);
assert.ok(liveProgressEvidence.includes("已完成 591 / 7,125 个样本"));
assert.ok(liveProgressEvidence.includes("剩余 6,534 个样本"));
assert.ok(liveProgressEvidence.includes(">580<"));
assert.ok(liveProgressEvidence.includes(">11<"));
assert.ok(liveProgressEvidence.includes("candidate-evidence-stage is-running"));
assert.equal(liveProgressEvidence.includes("8.3%"), false, "execution progress must not become scientific coverage");
const pausedDrainedEvidence = modelSandbox.renderCandidateExecutionEvidence(
  {
    id: "candidate:paused",
    status: "evaluating",
    inference_trace: {status: "pending", sample_count: 0, shown_count: 0, rows: []},
  },
  {
    status: "paused",
    execution_progress: {
      current_candidate_id: "candidate:paused",
      stage_progress: {
        progress_kind: "drained",
        completed_samples: 594,
        total_samples: 7125,
        succeeded_samples: 582,
        failed_samples: 12,
        remaining_samples: 6531,
        remaining_batches: 736,
      },
    },
  },
);
assert.ok(pausedDrainedEvidence.includes("已暂停且请求已排空；已完成 594 / 7,125 个样本"));
assert.ok(pausedDrainedEvidence.includes("恢复后继续 · 剩余 6,531 个样本、736 个微批"));
assert.ok(pausedDrainedEvidence.includes("candidate-evidence-stage is-warning"));
const mismatchedProgressEvidence = modelSandbox.renderCandidateExecutionEvidence(
  {id: "candidate:pinned", status: "pending", inference_trace: {status: "pending", rows: []}},
  {
    status: "running",
    execution_progress: {
      current_candidate_id: "candidate:other",
      stage_progress: {progress_kind: "waiting", completed_samples: 591, total_samples: 7125},
    },
  },
);
assert.ok(mismatchedProgressEvidence.includes("等待逐样本执行"));
assert.equal(mismatchedProgressEvidence.includes("591"), false, "active progress must not leak into a pinned historical candidate");
const legacyCompletedEvidence = modelSandbox.renderCandidateExecutionEvidence({
  inference_trace: {status: "completed", sample_count: 7125, rows: [{sample_index: 1, predicted: 0.4}]},
});
assert.ok(legacyCompletedEvidence.includes("已执行 7,125 个样本；旧运行未记录覆盖率"));
assert.equal(legacyCompletedEvidence.includes("等待逐样本执行"), false);
const escapedEvidence = modelSandbox.renderCandidateExecutionEvidence({
  proposal_source: "remote_model",
  model_plan: {research: [{title: '<img src=x onerror="alert(1)">', url: 'javascript:<svg onload="alert(2)">' }]},
  algorithm_execution: {attempts: [{phase: "compile", attempt: 1, status: "failed", public_error: '<img src=x onerror="alert(3)">'}]},
  inference_trace: {rows: [{
    sample_id: "unsafe",
    agent_decisions: [{role: "forecast_agent", decision: '<svg onload="alert(4)">'}],
  }]},
});
assert.equal(/<(?:img|svg)\b/i.test(escapedEvidence), false);
assert.ok(escapedEvidence.includes("&lt;img"));
assert.ok(escapedEvidence.includes("&lt;svg"));
const legacyMetricOnlyEvidence = modelSandbox.renderCandidateExecutionEvidence({metrics: {n: 123}});
assert.equal(legacyMetricOnlyEvidence.includes("123"), false);

// Live sample results follow the backend's active candidate until the user
// pins a candidate, then resume immediately when following is re-enabled.
const candidateFollowingRun = {
  id: "run:candidate-following", status: "running", best_candidate_id: "candidate:a",
  execution_progress: {current_candidate_id: "candidate:a"},
  candidates: [{id: "candidate:a", status: "evaluating"}, {id: "candidate:b", status: "pending"}],
};
modelSandbox.state.activeRun = candidateFollowingRun;
modelSandbox.state.selectedCandidateId = null;
modelSandbox.state.candidateSelectionPinned = false;
modelSandbox.resetCandidateSamples(candidateFollowingRun.id, null);
assert.equal(modelSandbox.activeCandidateIdForSamples(candidateFollowingRun), "candidate:a");
assert.equal(modelSandbox.syncCandidateSelection(candidateFollowingRun), true);
assert.equal(modelSandbox.state.selectedCandidateId, "candidate:a");
modelSandbox.state.candidateSamplePage = {rows: [{sample_id: "candidate-a-sample"}]};
candidateFollowingRun.execution_progress.current_candidate_id = "candidate:b";
assert.equal(modelSandbox.syncCandidateSelection(candidateFollowingRun), true);
assert.equal(modelSandbox.state.selectedCandidateId, "candidate:b");
assert.equal(modelSandbox.state.candidateSamplePage, null);
assert.equal(modelSandbox.candidateSampleSelectionMatches(candidateFollowingRun.id, "candidate:b"), true);
modelSandbox.state.selectedCandidateId = "candidate:a";
modelSandbox.state.candidateSelectionPinned = true;
modelSandbox.resetCandidateSamples(candidateFollowingRun.id, "candidate:a");
modelSandbox.state.candidateSamplePage = {rows: [{sample_id: "pinned-sample"}]};
assert.equal(modelSandbox.syncCandidateSelection(candidateFollowingRun), false);
assert.equal(modelSandbox.state.selectedCandidateId, "candidate:a");
assert.equal(modelSandbox.state.candidateSamplePage.rows[0].sample_id, "pinned-sample");
modelSandbox.state.candidateSelectionPinned = false;
assert.equal(modelSandbox.syncCandidateSelection(candidateFollowingRun), true);
assert.equal(modelSandbox.state.selectedCandidateId, "candidate:b");

const terminalCandidateFollowingRun = {
  id: "run:terminal-candidate-following",
  status: "completed",
  generation: 1,
  best_observed_candidate_id: "candidate:prior-best",
  execution_progress: {phase: "completed", current_generation: 2, current_candidate_id: null},
  candidates: [
    {id: "candidate:prior-best", generation: 1, slot_index: 0, status: "rejected"},
    {id: "candidate:latest-aborted", generation: 2, slot_index: 0, status: "aborted"},
  ],
  rounds: [
    {generation: 1, candidates: [{candidate_id: "candidate:prior-best", slot_index: 0}]},
    {generation: 2, candidates: [{candidate_id: "candidate:latest-aborted", slot_index: 0}]},
  ],
};
modelSandbox.state.activeRun = terminalCandidateFollowingRun;
modelSandbox.state.selectedCandidateId = null;
modelSandbox.state.candidateSelectionPinned = false;
modelSandbox.resetCandidateSamples(terminalCandidateFollowingRun.id, null);
assert.equal(modelSandbox.latestCandidateIdForSamples(terminalCandidateFollowingRun), "candidate:latest-aborted");
assert.equal(modelSandbox.syncCandidateSelection(terminalCandidateFollowingRun), true);
assert.equal(modelSandbox.state.selectedCandidateId, "candidate:latest-aborted");
assert.equal(modelSandbox.candidateSampleSelectionMatches(terminalCandidateFollowingRun.id, "candidate:latest-aborted"), true);

const originalExecutionSamplesRenderer = modelSandbox.renderExecutionSamples;
const originalCandidateSamplesRenderer = modelSandbox.renderCandidateSamples;
const originalCandidateSampleWorkspace = modelSandbox.state.workspace;
let executionSampleRenders = 0;
let candidateSampleRenders = 0;
modelSandbox.renderExecutionSamples = () => { executionSampleRenders += 1; };
modelSandbox.renderCandidateSamples = () => { candidateSampleRenders += 1; };
modelSandbox.state.workspace = "process";
modelSandbox.renderCandidateSampleViews();
assert.equal(executionSampleRenders, 1);
assert.equal(candidateSampleRenders, 1);
modelSandbox.state.workspace = "candidates";
modelSandbox.renderCandidateSampleViews();
assert.equal(executionSampleRenders, 1);
assert.equal(candidateSampleRenders, 2);
modelSandbox.renderExecutionSamples = originalExecutionSamplesRenderer;
modelSandbox.renderCandidateSamples = originalCandidateSamplesRenderer;
modelSandbox.state.workspace = originalCandidateSampleWorkspace;

// Selected candidates load a bounded sample page and replace it on every
// running poll. Unsupported servers fall back to the projection preview.
const originalSampleRequest = modelSandbox.EcologyDSHHost.request;
const sampleRequestPaths = [];
const originalSampleCapabilities = modelSandbox.state.catalog.dsh.capabilities;
modelSandbox.state.usingDemo = false;
modelSandbox.state.contextEpoch = 7;
modelSandbox.state.activeRun = {
  id: "run:sample-progress", status: "running", projection_revision: 3,
  candidates: [{
    id: "candidate:sample-progress", status: "evaluating",
    inference_trace: {sample_count: 2, rows: [{sample_id: "fallback-1", target: "air_temperature", predicted: 20.1}]},
  }],
};
modelSandbox.state.selectedCandidateId = "candidate:sample-progress";
modelSandbox.resetCandidateSamples("run:sample-progress", "candidate:sample-progress");
let unauthorizedSampleRequests = 0;
modelSandbox.EcologyDSHHost.request = async () => { unauthorizedSampleRequests += 1; return {}; };
assert.equal(await modelSandbox.loadCandidateSamples(0, {force: true, silent: true}), false);
assert.equal(unauthorizedSampleRequests, 0);
assert.equal(modelSandbox.state.candidateSamplePermissionDenied, true);
modelSandbox.state.catalog.dsh.capabilities = ["evaluation.samples.read"];
modelSandbox.resetCandidateSamples("run:sample-progress", "candidate:sample-progress");
modelSandbox.EcologyDSHHost.request = async (path) => {
  sampleRequestPaths.push(path);
  return {
    schema_version: "ecologyrsi-dsh.run-sample-page/1",
    run_id: "run:sample-progress", candidate_id: "candidate:sample-progress",
    page: {offset: 0, limit: 25, total: 2, has_more: false, complete: false, revision: 4, rows: [
      {sample_id: "sample-1", target_timestamp: 45678, origin_timestamp: 45677, target: "air_temperature", horizon_hours: 1, observed: 22.4, predicted: 22.1, reward: 0.84, sample_execution_status: "succeeded"},
      {sample_id: "sample-2", target_timestamp: 45679, origin_timestamp: 45678, target: "relative_humidity", horizon_hours: 6, observed: 70, predicted: 95, reward: -20, sample_execution_status: "failed", sample_execution_attempts: 3, sample_execution_retry_count: 2, prediction_source: "scoring_fallback", failure_class: "tool_timeout"},
    ]},
  };
};
assert.equal(await modelSandbox.loadCandidateSamples(0, {force: true}), true);
assert.equal(sampleRequestPaths[0], "/runs/run%3Asample-progress/samples?candidate_id=candidate%3Asample-progress&offset=0&limit=25");
assert.equal(modelSandbox.state.candidateSamplePage.rows.length, 2);
assert.equal(modelSandbox.state.candidateSamplePage.rows[0].reward, 0.84);
assert.equal(modelSandbox.state.candidateSamplePage.rows[1].attempts, 3);
assert.equal(modelSandbox.state.candidateSamplePage.rows[1].retry_count, 2);
assert.equal(modelSandbox.state.candidateSamplePage.rows[1].scoring_fallback, "scoring_fallback");
const legacySampleRow = modelSandbox.normalizeCandidateSampleRow({
  sample_id: "legacy-sample", observed: 10, predicted: 8, baseline: 7, error: -2,
}, 0);
assert.equal(legacySampleRow.reward, 1);
assert.equal(legacySampleRow.failure_message, null);
assert.ok(modelSandbox.renderCandidateSampleRows([legacySampleRow], 0).includes("已完成"));
const explicitFailedSampleRow = modelSandbox.normalizeCandidateSampleRow({
  sample_id: "failed-sample", execution_status: "failed", error: "tool timeout",
}, 0);
assert.equal(explicitFailedSampleRow.failure_message, "tool timeout");
const sampleRowsHtml = modelSandbox.renderCandidateSampleRows(modelSandbox.state.candidateSamplePage.rows, 0);
for (const text of ["室内气温", "1 小时", "22.4000", "22.1000", "0.8400", "已完成", "失败"]) {
  assert.ok(sampleRowsHtml.includes(text), `missing candidate sample result: ${text}`);
}
for (const text of ["评分惩罚值", "tool_timeout", "已重试 2 次", "共 3 次尝试"]) {
  assert.ok(sampleRowsHtml.includes(text), `missing candidate sample execution detail: ${text}`);
}
assert.equal(modelSandbox.candidateSamplesAreLive(modelSandbox.state.activeRun, modelSandbox.state.activeRun.candidates[0]), true);
assert.equal(modelSandbox.candidateSamplesAreLive({...modelSandbox.state.activeRun, status: "paused"}, modelSandbox.state.activeRun.candidates[0]), false);
assert.equal(modelSandbox.candidateSamplesAreLive({...modelSandbox.state.activeRun, status: "cancelled"}, modelSandbox.state.activeRun.candidates[0]), false);
assert.equal(modelSandbox.candidateSamplesAreLive(modelSandbox.state.activeRun, {...modelSandbox.state.activeRun.candidates[0], status: "evaluated"}), false);
const failedSampleHtml = modelSandbox.renderCandidateSampleRows([{
  sample_id: "unsafe-sample", sample_status: "failed", failure_message: '<img src=x onerror="alert(1)">',
}], 0);
assert.ok(failedSampleHtml.includes("pill-red"));
assert.equal(failedSampleHtml.includes("<img"), false);

modelSandbox.EcologyDSHHost.request = async () => {
  const error = new Error("temporary gateway error");
  error.status = 503;
  throw error;
};
assert.equal(await modelSandbox.loadCandidateSamples(25, {force: true, silent: true}), false);
assert.equal(modelSandbox.state.candidateSamplePage.offset, 0);
assert.equal(modelSandbox.state.candidateSampleOffset, 0);
assert.equal(modelSandbox.state.candidateSampleRetryOffset, 25);
assert.ok(modelSandbox.state.candidateSampleError);

modelSandbox.resetCandidateSamples("run:sample-progress", "candidate:sample-progress");
modelSandbox.EcologyDSHHost.request = async () => {
  const error = new Error("not found");
  error.status = 404;
  throw error;
};
assert.equal(await modelSandbox.loadCandidateSamples(0, {force: true, silent: true}), true);
assert.equal(modelSandbox.state.candidateSampleUnavailable, true);
assert.equal(modelSandbox.state.candidateSamplePage.source, "projection_preview");
assert.equal(modelSandbox.state.candidateSamplePage.rows[0].sample_id, "fallback-1");
modelSandbox.resetCandidateSamples("run:sample-progress", "candidate:sample-progress");
modelSandbox.EcologyDSHHost.request = async () => {
  const error = new Error("forbidden");
  error.status = 403;
  throw error;
};
assert.equal(await modelSandbox.loadCandidateSamples(0, {force: true, silent: true}), false);
assert.equal(modelSandbox.state.candidateSamplePermissionDenied, true);
assert.equal(modelSandbox.state.candidateSampleUnavailable, true);
modelSandbox.EcologyDSHHost.request = originalSampleRequest;
modelSandbox.state.catalog.dsh.capabilities = originalSampleCapabilities;
modelSandbox.state.activeRun = null;
modelSandbox.state.selectedCandidateId = null;
modelSandbox.resetCandidateSamples(null, null);

const cancelledEmpty = modelSandbox.normalizeRun({run_id: "run:cancelled-empty", status: "cancelled", generation: 0, candidates_count: 0});
const cancelledWithEvidence = modelSandbox.normalizeRun({run_id: "run:cancelled-evidence", status: "cancelled", generation: 0, candidates_count: 1});
const cancelledWithStageEvidence = modelSandbox.normalizeRun({
  run_id: "run:cancelled-stage-evidence", status: "cancelled", generation: 0, candidates_count: 0,
  events: [{type: "stage.recorded"}],
});
const archivedRun = modelSandbox.normalizeRun({
  run_id: "run:archived", status: "cancelled", archived_at: "2026-08-17T00:00:00Z",
});
assert.equal(archivedRun.archived, true);
assert.equal(archivedRun.archived_at, "2026-08-17T00:00:00Z");
const waitingRun = modelSandbox.normalizeRun({
  run_id: "run:waiting", status: "running", generation: 0, candidates_count: 0,
  total_generations: 2, candidates_per_generation: 3, max_candidates: 6, seed_policy: "fixed",
  samples_per_update: 500, sample_agent_batch_size: 64, sample_concurrency: 2,
  token_limit: 100000000, token_budget_scope: "sample_agent_gateway_calls_only@1",
  run_wide_accounting_complete: false,
  budget: {max_generations: 2, candidates_per_generation: 3, max_candidates: 6, token_limit: 100000000},
  configuration: {
    dataset_id: "dataset-a", episode_id: "episode-a", strategy_model_id: "policy-a",
    review_model_id: "judge-a", autonomous_mode: true, model_workflow: "research_compile_evolve@1",
    knowledge_online_enabled: true,
  },
});
assert.equal(modelSandbox.runUsesSampleAgentTokenBudget(waitingRun), true);
assert.equal(
  modelSandbox.tokenBudgetScopeText(waitingRun),
  "仅计 planner / repair / critic；不含 research / proposal / judge",
);
assert.equal(modelSandbox.runUsesSampleAgentTokenBudget({token_limit: 20000000}), false);
assert.equal(
  modelSandbox.tokenBudgetScopeText({token_limit: 20000000}),
  "历史运行未声明完整的 Token 计量范围",
);
const processSummaryNode = modelNode();
const modelQuerySelector = modelSandbox.document.querySelector;
modelSandbox.document.querySelector = (selector) => selector === "#process-summary" ? processSummaryNode : modelQuerySelector(selector);
modelSandbox.renderProcessSummary(waitingRun);
assert.ok(processSummaryNode.innerHTML.includes("逐样本智能体 Token 硬预算"));
assert.ok(processSummaryNode.innerHTML.includes("仅计 planner / repair / critic；不含 research / proposal / judge"));
for (const configuredExecutionParameter of ["每轮智能体样本", "500 个固定反馈样本", "请求微批", "64 个样本", "逐样本并发", "2 个在飞请求"]) {
  assert.ok(processSummaryNode.innerHTML.includes(configuredExecutionParameter));
}
assert.ok(processSummaryNode.innerHTML.includes("origin wave"));
assert.ok(processSummaryNode.innerHTML.includes("实际请求数以运行进度为准"));
modelSandbox.renderProcessSummary(crossCohortProjection);
assert.ok(processSummaryNode.innerHTML.includes("当前保留得分"));
assert.ok(processSummaryNode.innerHTML.includes("原始最高观测（跨窗口不可直接比较）"));
assert.equal(processSummaryNode.innerHTML.includes("已观测最高得分"), false);
modelSandbox.document.querySelector = modelQuerySelector;
const usageEstimateCandidate = {id: "candidate:usage-estimate"};
const usageEstimateProgress = {
  revision: "revision:usage-estimate",
  completed_samples: 50,
  total_samples: 200,
};
const completeUsageRun = {
  ...waitingRun,
  token_usage_available: true,
  tokens_used: 600,
  token_limit: 20000,
  model_usage: {
    available: true,
    complete: true,
    call_count: 4,
    physical_call_count: 4,
    logical_call_count: 3,
    replayed_call_count: 1,
    total_tokens: 600,
    scope_candidate_id: usageEstimateCandidate.id,
    scope_revision: usageEstimateProgress.revision,
  },
};
const completeUsageText = modelSandbox.modelUsageTokenProgressText(
  completeUsageRun,
  usageEstimateProgress,
  usageEstimateCandidate,
);
assert.ok(completeUsageText.includes("Token / 已完成样本 12"));
assert.ok(completeUsageText.includes("物理 4 / 逻辑 3"));
assert.ok(completeUsageText.includes("重复调用率 25.0%"));
assert.ok(completeUsageText.includes("阶段估算"));
assert.ok(completeUsageText.includes("跑满 200 样本约 2,400 Token"));
assert.equal(
  modelSandbox.modelUsageStageEstimateText(
    completeUsageRun.model_usage,
    {...usageEstimateProgress, completed_samples: 0},
    usageEstimateCandidate,
  ),
  "",
);
assert.equal(
  modelSandbox.modelUsageStageEstimateText(
    completeUsageRun.model_usage,
    usageEstimateProgress,
    {id: "candidate:other"},
  ),
  "",
);
assert.equal(
  modelSandbox.modelUsageCallEfficiencyText({
    call_count: 4,
    physical_call_count: 3,
    logical_call_count: 2,
    replayed_call_count: 1,
  }),
  "",
);
const incompleteUsageText = modelSandbox.modelUsageTokenProgressText({
  ...completeUsageRun,
  tokens_used: 120,
  token_limit: 1000,
  model_usage: {
    ...completeUsageRun.model_usage,
    complete: false,
    missing_call_count: 1,
    total_tokens: 26,
  },
}, usageEstimateProgress, usageEstimateCandidate);
assert.ok(incompleteUsageText.includes("预算计入：120 / 1,000"));
assert.ok(incompleteUsageText.includes("1 次调用实际用量未知"));
assert.ok(incompleteUsageText.includes("原始回执计数 26 Token"));
assert.equal(incompleteUsageText.includes("阶段估算"), false);
assert.equal(incompleteUsageText.includes("≥"), false);

const nativeDshUsageText = modelSandbox.modelUsageTokenProgressText({
  dsh_runtime: {
    native: true,
    context_pressure: {
      available: true,
      session_count: 3,
      maximum_total_tokens: 4096,
      maximum_surface_tokens: 3072,
    },
    provider_usage: {
      available: true,
      session_count: 3,
      total_tokens: 8192,
    },
  },
});
assert.ok(nativeDshUsageText.includes("DSH 上下文压力（会话最大当前值）：4,096 Token"));
assert.ok(nativeDshUsageText.includes("供应商报告累计用量：8,192 Token"));

// A paused run can retain a durable active phase so it can resume from the
// exact checkpoint.  The control state must win in the UI without hiding the
// sample and usage evidence captured before the request queue drained.
const monitorNode = () => ({
  value: "", textContent: "", className: "", disabled: false, innerHTML: "",
  hidden: false, title: "", style: {}, attributes: {}, addEventListener() {},
  setAttribute(name, value) { this.attributes[name] = String(value); },
});
const monitorSelectors = [
  "#execution-monitor-status", "#execution-progress-label", "#execution-progress-percent",
  "#execution-progress-detail", "#execution-progress-track", "#execution-progress-fill",
  "#execution-generation-progress", "#execution-candidate-progress", "#execution-sample-progress",
  "#execution-token-progress", "#execution-heartbeat", "#execution-last-activity",
  "#execution-stage-strip", "#execution-diagnostics-summary", "#execution-diagnostics-grid",
  "#active-candidate-summary", "#active-candidate-status", "#sample-inference-list",
  "#sample-inference-count", "#implementation-summary", "#implementation-status",
  "#autonomy-progress", "#autonomy-progress-status", "#round-stage-list",
];
const monitorNodes = Object.fromEntries(monitorSelectors.map((selector) => [selector, monitorNode()]));
const monitorQuerySelector = modelSandbox.document.querySelector;
const previousPendingAction = modelSandbox.state.pendingAction;
const previousEvents = modelSandbox.state.events;
modelSandbox.document.querySelector = (selector) => monitorNodes[selector] || monitorQuerySelector(selector);
modelSandbox.state.pendingAction = null;
const staleEvaluationProgressEvent = {
  type: "evaluation.progress",
  occurred_at: "2026-08-20T10:00:00Z",
  payload: {
    candidate_id: "candidate:paused-drained",
    completed_samples: 594,
    total_samples: 7125,
    message: "真实样本评测正在推进",
  },
};
modelSandbox.state.events = [staleEvaluationProgressEvent];
assert.equal(modelSandbox.executionElapsedText(null, true), " · 本轮计时中");
assert.equal(modelSandbox.executionElapsedText(null, false), "");
assert.equal(modelSandbox.executionElapsedText(2500, true), " · 本轮耗时 2.5 秒");
const activeResearchStartedAt = new Date(Date.now() - 125000).toISOString();
modelSandbox.state.events = [{type: "stage.recorded", occurred_at: activeResearchStartedAt, payload: {generation: 0, stage: "research", status: "started"}}];
const activeResearchElapsed = modelSandbox.executionActiveStageElapsedMs({id: "run:active-research", generation: 0}, "research");
assert.ok(activeResearchElapsed >= 124000 && activeResearchElapsed < 130000);
assert.equal(modelSandbox.autonomyWaitDurationText(125000), "2 分 5 秒");
assert.equal(modelSandbox.autonomyTimeoutDurationText(2), "2 秒");
assert.equal(modelSandbox.autonomyTimeoutDurationText(900), "15 分钟");
const previousMonitorCatalog = modelSandbox.state.catalog;
modelSandbox.state.catalog = modelSandbox.normalizeCatalog({
  dsh_models: [{
    id: "newapi/glm-5.2",
    roles: ["propose"],
    connection: {request_policy: {timeout_seconds: 900, max_attempts: 4}},
  }],
});
const frozenKnowledgeRound = {
  generation: 1,
  knowledge: {snapshot_digest: "sha256:frozen-before-research"},
  research_iteration: null,
  stages: {},
};
const frozenKnowledgeRun = {
  id: "run:frozen-knowledge-only",
  status: "running",
  generation: 0,
  configuration: {strategy_model_id: "newapi/glm-5.2"},
  execution_progress: {phase: "queued", current_generation: 1},
  rounds: [frozenKnowledgeRound],
};
modelSandbox.state.events = [];
assert.equal(modelSandbox.autonomyStepStatus(frozenKnowledgeRun, "research"), "pending");
assert.equal(modelSandbox.autonomyResearchPresentation(frozenKnowledgeRun, frozenKnowledgeRound).detail, "知识快照已冻结，等待发起远端研究请求");

const activeResearchRound = {
  ...frozenKnowledgeRound,
  timing: {status: "running", started_at: activeResearchStartedAt},
  stages: {research: "running"},
};
const activeResearchRun = {
  ...frozenKnowledgeRun,
  id: "run:active-remote-research",
  execution_progress: {phase: "research", current_stage: "research", current_generation: 1},
  rounds: [activeResearchRound],
};
modelSandbox.state.events = [{
  seq: 8,
  type: "stage.recorded",
  occurred_at: activeResearchStartedAt,
  payload: {generation: 0, stage: "research", status: "started", attempt: 1},
}];
assert.equal(modelSandbox.autonomyStepStatus(activeResearchRun, "research"), "running");
const activeResearchPresentation = modelSandbox.autonomyResearchPresentation(activeResearchRun, activeResearchRound);
assert.equal(activeResearchPresentation.statusText, "等待远端响应");
for (const text of ["知识快照已冻结，正在等待远端模型响应", "已等待 2 分", "目录策略单次调用超时上限 15 分钟", "最多 4 次尝试"]) {
  assert.ok(activeResearchPresentation.detail.includes(text), `missing active research detail: ${text}`);
}
modelSandbox.renderAutonomyProgress(activeResearchRun);
assert.equal(monitorNodes["#autonomy-progress-status"].textContent, "等待远端模型响应");
assert.equal(monitorNodes["#autonomy-progress-status"].className, "pill pill-blue");
for (const text of ["等待远端响应", "正在等待远端模型响应", "目录策略单次调用超时上限 15 分钟"]) {
  assert.ok(monitorNodes["#autonomy-progress"].innerHTML.includes(text), `missing rendered research detail: ${text}`);
}

const completedResearchRound = {
  ...frozenKnowledgeRound,
  research_iteration: {status: "model_generated", iteration_digest: "sha256:research-complete"},
};
assert.equal(modelSandbox.autonomyStepStatus({...activeResearchRun, rounds: [completedResearchRound]}, "research"), "completed");
modelSandbox.state.events = [{
  seq: 9,
  type: "stage.recorded",
  occurred_at: new Date().toISOString(),
  payload: {generation: 0, stage: "research", status: "completed", attempt: 1},
}];
assert.equal(modelSandbox.autonomyStepStatus({...frozenKnowledgeRun, execution_progress: {phase: "proposal"}}, "research"), "completed");
assert.equal(modelSandbox.autonomyStepStatus({
  ...frozenKnowledgeRun,
  generation: 1,
  rounds: [{...frozenKnowledgeRound, generation: 2}],
  execution_progress: {phase: "queued", current_generation: 2},
}, "research"), "pending");

modelSandbox.state.catalog = modelSandbox.normalizeCatalog({
  dsh_models: [{id: "newapi/glm-5.2", roles: ["propose"], connection: {request_policy: {max_attempts: 4}}}],
});
modelSandbox.state.events = [{
  type: "stage.recorded",
  occurred_at: activeResearchStartedAt,
  payload: {generation: 0, stage: "research", status: "started", attempt: 1},
}];
assert.equal(modelSandbox.autonomyResearchPresentation(activeResearchRun, activeResearchRound).detail.includes("超时上限"), false);
modelSandbox.state.catalog = previousMonitorCatalog;
const priorGenerationCandidate = {id: "candidate:prior", generation: 0, slot_index: 1};
assert.equal(modelSandbox.executionCandidateFor({
  id: "run:research-without-candidate",
  generation: 1,
  candidates: [priorGenerationCandidate],
  execution_progress: {phase: "research", current_stage: "research", current_candidate_id: null},
}, {generation: 2, candidates: []}), null);
modelSandbox.state.events = [];
const pausedCandidate = {
  id: "candidate:paused-drained",
  candidate_id: "candidate:paused-drained",
  status: "evaluating",
  generation: 1,
  slot_index: 0,
  changes: {residual_scale_co2_24h: {before: 1, after: 1.2}},
  execution: {
    current_stage: "evaluation",
    stages: {
      proposal: "completed", candidate: "completed", training: "completed",
      evaluation: "running", judge: "pending", decision: "pending",
    },
  },
};
const pausedRound = {
  generation: 1,
  stages: pausedCandidate.execution.stages,
  candidates: [{candidate_id: pausedCandidate.id, stages: pausedCandidate.execution.stages}],
};
const pausedDrainedRun = {
  id: "run:paused-drained",
  status: "paused",
  generation: 0,
  total_generations: 3,
  token_limit: 20000000,
  token_budget_scope: "sample_agent_gateway_calls_only@1",
  token_usage_available: true,
  tokens_used: 1647711,
  configuration: {prediction_model_id: "greenhouse-ridge"},
  candidates: [pausedCandidate],
  rounds: [pausedRound],
  execution_progress: {
    phase: "evaluation",
    current_stage: "evaluation",
    current_generation: 1,
    completed_generations: 0,
    current_candidate_id: pausedCandidate.id,
    progress_percent: 8.3368421,
    stage_progress: {
      revision: "revision:paused-drained",
      progress_kind: "drained",
      completed_samples: 594,
      total_samples: 7125,
      succeeded_samples: 591,
      failed_samples: 3,
      in_flight_batches: 0,
      queued_batches: 0,
      updated_at: "2026-08-20T10:00:00Z",
    },
  },
  model_usage: {
    available: true,
    complete: false,
    call_count: 115,
    physical_call_count: 115,
    logical_call_count: 71,
    replayed_call_count: 44,
    missing_call_count: 1,
    total_tokens: 1397351,
    scope_candidate_id: pausedCandidate.id,
    scope_revision: "revision:paused-drained",
  },
  execution_diagnostics: {
    execution_evidence_status: "retained_partial",
    live_evaluation_completed_examples: 594,
    live_evaluation_total_examples: 7125,
    live_evaluation_succeeded_examples: 591,
    live_evaluation_failed_examples: 3,
  },
  updated_at: "2026-08-20T10:00:00Z",
};
assert.equal(modelSandbox.eventTitle(staleEvaluationProgressEvent, pausedDrainedRun), "已保留部分样本评测证据");
assert.equal(modelSandbox.eventDetail(staleEvaluationProgressEvent, pausedDrainedRun).includes("正在推进"), false);
modelSandbox.state.events = [staleEvaluationProgressEvent];
modelSandbox.renderExecutionMonitor(pausedDrainedRun);
assert.equal(monitorNodes["#execution-monitor-status"].textContent, "已暂停，请求已排空");
assert.equal(monitorNodes["#execution-monitor-status"].className, "pill pill-amber");
assert.equal(monitorNodes["#execution-progress-label"].textContent, "第 1 轮已暂停");
assert.ok(monitorNodes["#execution-progress-detail"].textContent.startsWith("暂停阶段："));
assert.ok(monitorNodes["#execution-progress-detail"].textContent.includes("独立评测"));
assert.ok(monitorNodes["#execution-progress-detail"].textContent.includes("请求已排空"));
assert.equal(monitorNodes["#execution-progress-detail"].textContent.includes("当前阶段"), false);
assert.equal(monitorNodes["#execution-progress-track"].className, "execution-progress-track is-paused");
assert.equal(monitorNodes["#execution-progress-track"].className.includes("is-running"), false);
assert.equal(monitorNodes["#execution-progress-fill"].style.width, "8.3%");
assert.ok(monitorNodes["#execution-sample-progress"].textContent.includes("样本进度：594 / 7,125 · 成功 591 · 失败 3"));
assert.ok(monitorNodes["#execution-token-progress"].textContent.includes("预算计入：1,647,711 / 20,000,000"));
assert.ok(monitorNodes["#execution-token-progress"].textContent.includes("物理 115 / 逻辑 71"));
assert.ok(monitorNodes["#execution-heartbeat"].textContent.includes("暂停后请求已排空"));
assert.equal(monitorNodes["#execution-heartbeat"].textContent.includes("评测心跳"), false);
assert.equal(monitorNodes["#execution-last-activity"].textContent.includes("正在推进"), false);
assert.equal(monitorNodes["#active-candidate-status"].textContent, "已暂停");
assert.equal(monitorNodes["#active-candidate-status"].className, "pill pill-amber");
assert.equal(monitorNodes["#implementation-status"].textContent, "已暂停");
assert.equal(monitorNodes["#implementation-status"].className, "pill pill-amber");
assert.ok(monitorNodes["#execution-stage-strip"].innerHTML.includes("execution-stage-chip is-paused"));
const completedHeartbeatSnapshot = modelSandbox.executionSampleProgressSnapshot({
  status: "completed",
  execution_diagnostics: {
    live_evaluation_completed_examples: 9,
    live_evaluation_total_examples: 9,
    live_evaluation_succeeded_examples: 0,
    live_evaluation_failed_examples: 0,
  },
}, {
  progress_kind: "completed_batch",
  completed_samples: 9,
  total_samples: 9,
  succeeded_samples: 9,
  failed_samples: 0,
});
assert.equal(completedHeartbeatSnapshot.succeeded_samples, 9);
assert.equal(completedHeartbeatSnapshot.failed_samples, 0);
modelSandbox.renderAutonomyProgress(pausedDrainedRun);
assert.equal(monitorNodes["#autonomy-progress-status"].textContent, "已暂停，请求已排空");
assert.equal(monitorNodes["#autonomy-progress-status"].className, "pill pill-amber");
assert.ok(monitorNodes["#autonomy-progress"].innerHTML.includes("autonomy-progress-step is-paused"));
const completedProgressRun = {
  ...pausedDrainedRun,
  status: "completed",
  generation: 1,
  total_generations: 1,
  candidates: [{
    ...pausedCandidate,
    execution: {
      ...pausedCandidate.execution,
      stage_progress: {
        progress_kind: "waiting",
        completed_samples: 8,
        total_samples: 9,
        succeeded_samples: 8,
        failed_samples: 0,
        in_flight_batches: 1,
        queued_batches: 1,
        samples_per_minute: 60,
        estimated_remaining_seconds: 1,
        updated_at: "2026-08-20T10:00:00Z",
      },
    },
  }],
  execution_progress: {
    ...pausedDrainedRun.execution_progress,
    phase: "completed",
    current_stage: null,
    completed_generations: 1,
    progress_percent: 75,
    stage_progress: null,
  },
  execution_diagnostics: {
    execution_evidence_status: "aborted_partial",
    live_evaluation_completed_examples: 9,
    live_evaluation_total_examples: 9,
  },
};
modelSandbox.state.events = [staleEvaluationProgressEvent];
modelSandbox.renderExecutionMonitor(completedProgressRun);
assert.ok(monitorNodes["#execution-sample-progress"].textContent.includes("样本进度：9 / 9"));
assert.equal(monitorNodes["#execution-sample-progress"].textContent.includes("8 / 9"), false);
assert.equal(monitorNodes["#execution-sample-progress"].textContent.includes("成功 8"), false);
assert.equal(monitorNodes["#execution-sample-progress"].textContent.includes("预计剩余"), false);
assert.equal(monitorNodes["#execution-monitor-status"].textContent.includes("模型执行中"), false);
assert.equal(monitorNodes["#active-candidate-status"].textContent.includes("执行中"), false);
assert.equal(monitorNodes["#implementation-status"].textContent.includes("编译／执行中"), false);
assert.equal(monitorNodes["#execution-stage-strip"].innerHTML.includes("is-running"), false);
assert.equal(monitorNodes["#execution-stage-strip"].innerHTML.includes("进行中"), false);
assert.ok(monitorNodes["#execution-stage-strip"].innerHTML.includes("未封存"));
assert.ok(monitorNodes["#execution-heartbeat"].textContent.startsWith("评测证据："));
for (const staleLiveText of ["网关执行中", "已提交", "正在推进"]) {
  assert.equal(monitorNodes["#execution-heartbeat"].textContent.includes(staleLiveText), false);
  assert.equal(monitorNodes["#execution-last-activity"].textContent.includes(staleLiveText), false);
}
modelSandbox.renderAutonomyProgress(completedProgressRun);
assert.equal(monitorNodes["#autonomy-progress-status"].textContent.includes("模型执行中"), false);
assert.equal(monitorNodes["#autonomy-progress"].innerHTML.includes("is-running"), false);
assert.equal(monitorNodes["#autonomy-progress"].innerHTML.includes("进行中"), false);
assert.equal(modelSandbox.eventTitle(staleEvaluationProgressEvent, completedProgressRun), "已保留中止前样本评测证据");
assert.equal(modelSandbox.eventDetail(staleEvaluationProgressEvent, completedProgressRun).includes("正在推进"), false);
const staleStageEvent = {type: "stage.recorded", payload: {stage: "evaluation", status: "running"}};
assert.equal(modelSandbox.eventTitle(staleStageEvent, completedProgressRun).includes("进行中"), false);
assert.equal(modelSandbox.eventTone(staleStageEvent, completedProgressRun), "pill-amber");

const terminalResearchHtml = modelSandbox.renderRoundResearchEvidence({
  generation: 1,
  research_iteration: {
    status: "running",
    analysis_summary: {status: "running"},
    final_plan: {status: "in_progress"},
  },
  candidates: [{candidate_id: pausedCandidate.id, stages: {evaluation: "running", judge: "running"}}],
}, completedProgressRun);
assert.equal(terminalResearchHtml.includes("is-running"), false);
assert.equal(terminalResearchHtml.includes("进行中"), false);
modelSandbox.state.activeRun = completedProgressRun;
modelSandbox.renderRoundStages();
assert.equal(monitorNodes["#round-stage-list"].innerHTML.includes("is-running"), false);
assert.equal(monitorNodes["#round-stage-list"].innerHTML.includes("进行中"), false);

// A terminal run can contain a newer, partially recorded round than
// run.generation: the latter only advances after the whole generation is
// committed.  The execution monitor must show that newest evidence instead
// of presenting the prior completed round as current.
const priorCompletedCandidate = {
  id: "candidate:prior-completed",
  candidate_id: "candidate:prior-completed",
  status: "evaluated",
  generation: 1,
  slot_index: 0,
  execution: {
    stages: {
      proposal: "completed", candidate: "completed", training: "completed",
      evaluation: "completed", judge: "completed", decision: "completed",
    },
  },
};
const latestAbortedCandidate = {
  id: "candidate:latest-aborted",
  candidate_id: "candidate:latest-aborted",
  status: "aborted",
  generation: 2,
  slot_index: 0,
  execution: {
    stages: {
      proposal: "completed", candidate: "completed", training: "completed",
      evaluation: "not_recorded", judge: "not_recorded", decision: "not_recorded",
    },
    stage_progress: {
      progress_kind: "waiting",
      completed_samples: 8,
      total_samples: 9,
      succeeded_samples: 8,
      failed_samples: 0,
      in_flight_batches: 1,
      queued_batches: 1,
    },
  },
};
const terminalTwoRoundRun = {
  ...completedProgressRun,
  id: "run:terminal-two-rounds",
  generation: 1,
  total_generations: 2,
  candidates: [priorCompletedCandidate, latestAbortedCandidate],
  rounds: [
    {generation: 1, stages: priorCompletedCandidate.execution.stages, candidates: [{candidate_id: priorCompletedCandidate.id, stages: priorCompletedCandidate.execution.stages}]},
    {generation: 2, stages: latestAbortedCandidate.execution.stages, candidates: [{candidate_id: latestAbortedCandidate.id, stages: latestAbortedCandidate.execution.stages}]},
  ],
  execution_progress: {
    ...completedProgressRun.execution_progress,
    current_generation: 2,
  },
};
const terminalLatestRound = modelSandbox.executionRound(terminalTwoRoundRun);
assert.equal(terminalLatestRound.generation, 2);
assert.equal(modelSandbox.executionCandidateFor(terminalTwoRoundRun, terminalLatestRound).id, latestAbortedCandidate.id);
modelSandbox.renderExecutionMonitor(terminalTwoRoundRun);
assert.ok(monitorNodes["#active-candidate-summary"].innerHTML.includes("latest-aborted"));
assert.ok(monitorNodes["#execution-stage-strip"].innerHTML.includes("未封存"));
assert.equal(monitorNodes["#execution-stage-strip"].innerHTML.includes("is-running"), false);
assert.ok(monitorNodes["#execution-sample-progress"].textContent.includes("样本进度：9 / 9"));
assert.equal(monitorNodes["#execution-sample-progress"].textContent.includes("成功 8"), false);

const queuedSecondRoundRun = {
  ...terminalTwoRoundRun,
  status: "running",
  execution_progress: {phase: "queued", current_generation: 2, completed_generations: 1},
};
assert.equal(modelSandbox.executionRound(queuedSecondRoundRun).generation, 2);
assert.equal(modelSandbox.executionRound({...queuedSecondRoundRun, execution_progress: {}}).generation, 2);
assert.equal(modelSandbox.executionRound({...queuedSecondRoundRun, status: "paused", execution_progress: {}}).generation, 2);
assert.equal(modelSandbox.executionRound({...terminalTwoRoundRun, status: "failed"}).generation, 2);
assert.equal(modelSandbox.executionRound({...terminalTwoRoundRun, status: "cancelled"}).generation, 2);
assert.equal(modelSandbox.executionRound({...queuedSecondRoundRun, rounds: []}), null);

const livePartialRun = {
  ...completedProgressRun,
  status: "running",
  execution_diagnostics: {execution_evidence_status: "partial_live"},
};
assert.equal(modelSandbox.eventTitle(staleEvaluationProgressEvent, livePartialRun), "真实样本评测正在推进");
assert.equal(modelSandbox.executionStatusForRun(livePartialRun, "running"), "running");
assert.equal(modelSandbox.executionStatusForRun(pausedDrainedRun, "running"), "paused");
assert.equal(modelSandbox.executionStatusForRun(completedProgressRun, "running"), "not_recorded");
assert.equal(modelSandbox.executionStatusForRun({...completedProgressRun, status: "failed"}, "running"), "aborted");
assert.equal(modelSandbox.executionStatusForRun({...completedProgressRun, status: "cancelled"}, "running"), "aborted");
for (const [candidateStatus, expectedText, expectedClass] of [
  ["paused", "已暂停", "pill-amber"],
  ["aborted", "已中止", "pill-red"],
  ["not_recorded", "未封存", "pill-neutral"],
]) {
  assert.equal(modelSandbox.candidateStatusText(candidateStatus), expectedText);
  assert.equal(modelSandbox.candidateStatusClass(candidateStatus), expectedClass);
}
for (const terminalStatus of ["failed", "cancelled"]) {
  const terminalRun = {
    ...completedProgressRun,
    status: terminalStatus,
    execution_progress: {
      ...completedProgressRun.execution_progress,
      phase: "gateway_retry",
      current_stage: "evaluation",
      retry_wait: {reason: "stale retry", retry_at: "2026-08-20T10:05:00Z"},
    },
    execution_scheduler: {
      run_state: "queued", queue_position: 2, queued_ahead: 1,
      active_worker_count: 1, worker_count: 1,
    },
  };
  modelSandbox.renderExecutionMonitor(terminalRun);
  const terminalUi = [
    monitorNodes["#execution-monitor-status"].textContent,
    monitorNodes["#execution-progress-label"].textContent,
    monitorNodes["#execution-progress-detail"].textContent,
    monitorNodes["#execution-heartbeat"].textContent,
    monitorNodes["#execution-stage-strip"].innerHTML,
    monitorNodes["#active-candidate-status"].textContent,
    monitorNodes["#implementation-status"].textContent,
  ].join(" ");
  for (const staleLiveText of ["模型执行中", "执行中", "进行中", "等待网关重试", "后台排队中", "网关执行中", "已提交", "is-running"]) {
    assert.equal(terminalUi.includes(staleLiveText), false, `${terminalStatus} leaked ${staleLiveText}`);
  }
  assert.equal(monitorNodes["#execution-monitor-status"].textContent, terminalStatus === "failed" ? "执行失败" : "已取消");
  assert.equal(monitorNodes["#execution-monitor-status"].className, terminalStatus === "failed" ? "pill pill-red" : "pill pill-amber");
  assert.equal(monitorNodes["#execution-progress-label"].textContent, terminalStatus === "failed" ? "运行失败" : "运行已取消");
  assert.equal(monitorNodes["#active-candidate-status"].textContent, "已中止");
  assert.equal(monitorNodes["#active-candidate-status"].className, "pill pill-red");
  assert.equal(monitorNodes["#implementation-status"].textContent, "已中止");
  assert.equal(monitorNodes["#implementation-status"].className, "pill pill-red");
  assert.ok(monitorNodes["#execution-stage-strip"].innerHTML.includes("已中止"));
  assert.equal(modelSandbox.executionSchedulerQueueInfo(terminalRun), null);
  assert.equal(modelSandbox.eventTone(staleStageEvent, terminalRun), "pill-red");
}
const cancelledWithoutPartialEvidence = {
  ...completedProgressRun,
  status: "cancelled",
  execution_diagnostics: {},
};
modelSandbox.renderExecutionMonitor(cancelledWithoutPartialEvidence);
assert.equal(monitorNodes["#execution-monitor-status"].textContent, "已取消");
assert.equal(monitorNodes["#execution-progress-label"].textContent, "运行已取消");
assert.equal(monitorNodes["#execution-progress-detail"].textContent.includes("等待下一轮"), false);
const schedulerQueuedRun = {
  ...waitingRun,
  id: "run:scheduler-queued",
  auto_progress: true,
  execution_progress: {phase: "queued", current_generation: 1, completed_generations: 0},
  execution_scheduler: {
    run_state: "queued",
    queue_position: 3,
    queued_ahead: 2,
    active_worker_count: 1,
    worker_count: 1,
  },
};
assert.equal(
  modelSandbox.executionSchedulerQueueInfo({...schedulerQueuedRun, execution_scheduler: undefined}),
  null,
);
modelSandbox.renderExecutionMonitor(schedulerQueuedRun);
assert.equal(monitorNodes["#execution-monitor-status"].textContent, "后台排队中");
assert.equal(monitorNodes["#execution-monitor-status"].className, "pill pill-blue");
assert.equal(monitorNodes["#execution-progress-label"].textContent, "已进入后台执行队列");
for (const detail of ["等待其他运行当前轮结束", "队列第 3 位", "前方 2 个运行", "工作器占用 1 / 1"]) {
  assert.ok(monitorNodes["#execution-progress-detail"].textContent.includes(detail));
}
modelSandbox.document.querySelector = monitorQuerySelector;
modelSandbox.state.pendingAction = previousPendingAction;
modelSandbox.state.events = previousEvents;

const waitingBetweenRounds = modelSandbox.normalizeRun({
  ...waitingRun,
  run_id: "run:waiting-between-rounds",
  generation: 1,
  candidates_count: 3,
  execution_progress: {phase: "waiting", completed_generations: 1, total_generations: 2},
});
const activelyExecutingRun = modelSandbox.normalizeRun({
  ...waitingBetweenRounds,
  run_id: "run:executing",
  execution_progress: {phase: "training", current_stage: "training"},
});
assert.equal(modelSandbox.runNeedsAdvanceAction(waitingRun, []), true);
assert.equal(modelSandbox.runNeedsAdvanceAction(waitingBetweenRounds, []), true);
assert.equal(modelSandbox.displayRunStatusText(waitingBetweenRounds, []), "等待推进");
assert.equal(modelSandbox.runNeedsAdvanceAction(activelyExecutingRun, []), false);
assert.equal(modelSandbox.runNeedsAdvanceAction({...waitingBetweenRounds, status: "paused"}, []), false);
const serverManagedRun = modelSandbox.normalizeRun({...waitingBetweenRounds, run_id: "run:server-managed", auto_progress: true});
assert.equal(modelSandbox.serverAutoProgressEnabled(serverManagedRun), true);
assert.equal(modelSandbox.autoAdvanceWaiting(serverManagedRun), false);
assert.equal(modelSandbox.runNeedsAdvanceAction(serverManagedRun, []), false);
assert.equal(modelSandbox.displayRunStatusText(serverManagedRun, []), "运行中");
modelSandbox.state.activeRun = serverManagedRun;
modelSandbox.state.autoAdvanceRunId = null;
modelSandbox.state.autoAdvanceTimer = null;
assert.equal(modelSandbox.ensureAutoAdvanceForRun(serverManagedRun.id), true);
assert.equal(modelSandbox.state.autoAdvanceRunId, serverManagedRun.id);
assert.equal(modelSandbox.state.autoAdvanceTimer, null);
const tokenPausedRun = modelSandbox.normalizeRun({
  ...serverManagedRun,
  run_id: "run:token-paused",
  status: "paused",
  pause_code: "model_token_budget_exhausted",
  pause_reason: "逐样本智能体 Token 硬预算已停止新调用。",
});
assert.equal(modelSandbox.runHasHardTokenPause(tokenPausedRun), true);
assert.equal(modelSandbox.displayRunStatusText(tokenPausedRun, []), "逐样本智能体 Token 预算已暂停");
const queuedServerRun = modelSandbox.normalizeRun({
  ...waitingRun,
  run_id: "run:queued-server",
  auto_progress: true,
  execution_progress: {phase: "queued", auto_progress: true},
});
assert.equal(modelSandbox.createStatusForRun(queuedServerRun, []).state, "queued");
assert.match(modelSandbox.createStatusForRun(queuedServerRun, []).message, /后台将自动执行全部轮次/);
assert.equal(modelSandbox.createStatusForRun(activelyExecutingRun, []).state, "background");
modelSandbox.state.activeRun = waitingRun;
modelSandbox.state.autoAdvanceOptOutRunIds[waitingRun.id] = true;
assert.equal(modelSandbox.ensureAutoAdvanceForRun(waitingRun.id), false);
delete modelSandbox.state.autoAdvanceOptOutRunIds[waitingRun.id];
modelSandbox.state.runs = [cancelledEmpty, cancelledWithEvidence, cancelledWithStageEvidence, waitingRun];
assert.equal(modelSandbox.isCancelledEmptyRun(cancelledEmpty), true);
assert.equal(modelSandbox.isCancelledEmptyRun(cancelledWithEvidence), false);
assert.equal(modelSandbox.isCancelledEmptyRun(cancelledWithStageEvidence), false);
assert.deepEqual(Array.from(modelSandbox.visibleRuns().map((run) => run.id)), ["run:cancelled-evidence", "run:cancelled-stage-evidence", "run:waiting"]);
modelSandbox.state.showCancelledEmptyRuns = true;
assert.deepEqual(Array.from(modelSandbox.visibleRuns().map((run) => run.id)), ["run:cancelled-empty", "run:cancelled-evidence", "run:cancelled-stage-evidence", "run:waiting"]);
modelSandbox.state.showCancelledEmptyRuns = false;
assert.equal(modelSandbox.pendingRunMatchesRequest(waitingRun, {
  dataset_id: "dataset-a", episode_id: "episode-a", strategy_model_id: "policy-a", review_model_id: "judge-a",
  autonomous_mode: true, model_workflow: "research_compile_evolve@1", knowledge_online_enabled: true,
  rounds: 2, budget: {max_generations: 2, candidates_per_generation: 3, max_candidates: 6}, seed_policy: "fixed",
}), true);
assert.equal(modelSandbox.pendingRunMatchesRequest(waitingRun, {
  dataset_id: "dataset-a", episode_id: "episode-a", strategy_model_id: "policy-a", review_model_id: "judge-a",
  autonomous_mode: true, model_workflow: "research_compile_evolve@1", knowledge_online_enabled: true,
  rounds: 2, budget: {max_generations: 2, candidates_per_generation: 3, max_candidates: 1}, seed_policy: "fixed",
}), true);
assert.equal(modelSandbox.pendingRunMatchesRequest(waitingRun, {
  dataset_id: "dataset-b", episode_id: "episode-a", strategy_model_id: "policy-a", review_model_id: "judge-a",
  autonomous_mode: true, model_workflow: "research_compile_evolve@1", knowledge_online_enabled: true,
  rounds: 2, budget: {max_generations: 2, candidates_per_generation: 3, max_candidates: 6}, seed_policy: "fixed",
}), false);
const matchingRequest = {
  dataset_id: "dataset-a", episode_id: "episode-a", strategy_model_id: "policy-a", review_model_id: "judge-a",
  autonomous_mode: true, model_workflow: "research_compile_evolve@1", knowledge_online_enabled: true,
  rounds: 2, budget: {max_generations: 2, candidates_per_generation: 3, max_candidates: 6}, seed_policy: "fixed",
};
const nativeMatchingRequest = {
  ...matchingRequest,
  execution_protocol: "dsh_native_plugin_evolution@1",
};
const nativeWaitingRun = modelSandbox.normalizeRun({
  ...waitingRun,
  run_id: "run:native-waiting",
  configuration: {
    ...waitingRun.configuration,
    execution_protocol: "dsh_native_plugin_evolution@1",
  },
});
assert.equal(modelSandbox.pendingRunMatchesRequest(waitingRun, nativeMatchingRequest), false);
assert.equal(modelSandbox.pendingRunMatchesRequest(nativeWaitingRun, nativeMatchingRequest), true);
for (const nonPending of [
  modelSandbox.normalizeRun({...waitingRun, run_id: "run:completed", status: "completed"}),
  modelSandbox.normalizeRun({...waitingRun, run_id: "run:cancelled", status: "cancelled"}),
  activelyExecutingRun,
]) {
  assert.equal(modelSandbox.pendingRunMatchesRequest(nonPending, matchingRequest), false);
}
for (const resumableLegacyRun of [
  modelSandbox.normalizeRun({...waitingRun, run_id: "run:with-candidate", candidates_count: 1}),
  modelSandbox.normalizeRun({...waitingRun, run_id: "run:advanced", generation: 1}),
  waitingBetweenRounds,
]) {
  assert.equal(modelSandbox.pendingRunMatchesRequest(resumableLegacyRun, matchingRequest), true);
}
const legacyBooleanRun = modelSandbox.normalizeRun({
  run_id: "run:legacy-boolean", status: "running", generation: 0, candidates_count: 0,
  total_generations: "2", candidates_per_generation: "3", max_candidates: "6", seed_policy: "fixed",
  samples_per_update: 500, sample_agent_batch_size: 64, sample_concurrency: 2,
  budget: {max_generations: 2, candidates_per_generation: 3, max_candidates: 6, token_limit: 100000000},
  configuration: {
    dataset_id: "dataset-a", episode_id: "episode-a", policy_model_id: "policy-a", judge_model_id: "judge-a",
    autonomous_mode: "false", model_workflow: "research_compile_evolve@1", knowledge_online_enabled: "false",
  },
});
assert.equal(modelSandbox.pendingRunMatchesRequest(legacyBooleanRun, {
  ...matchingRequest, autonomous_mode: false, knowledge_online_enabled: false,
}), true);
assert.equal(modelSandbox.pendingRunMatchesRequest(waitingRun, {
  ...matchingRequest, samples_per_update: 250,
}), false);
assert.equal(modelSandbox.pendingRunMatchesRequest(waitingRun, {
  ...matchingRequest, sample_agent_batch_size: 32,
}), false);
assert.equal(modelSandbox.pendingRunMatchesRequest(waitingRun, {
  ...matchingRequest, sample_concurrency: 4,
}), false);

modelSandbox.state.runs = [cancelledEmpty, waitingRun];
modelSandbox.state.activeRun = cancelledEmpty;
modelSandbox.state.events = [{type: "run.cancelled"}];
modelSandbox.state.datasetContext = {run_id: cancelledEmpty.id};
modelSandbox.state.datasetPage = {page: {rows: [1]}};
modelSandbox.state.datasetRequest = 7;
const epochBeforeReconcile = modelSandbox.state.viewEpoch;
assert.equal(modelSandbox.reconcileVisibleRunSelection(), true);
assert.equal(modelSandbox.state.activeRun.id, waitingRun.id);
assert.equal(modelSandbox.state.datasetContext, null);
assert.equal(modelSandbox.state.datasetPage, null);
assert.equal(modelSandbox.state.datasetRequest, 8);
assert.equal(modelSandbox.state.viewEpoch, epochBeforeReconcile + 1);
assert.equal(modelSandbox.state.loadState, "ready");

const reuseToasts = [];
const selectRunCommand = modelSandbox.selectRun;
modelSandbox.state.catalog.dsh.capabilities = ["evolution.run.create"];
modelSandbox.state.runs = [nativeWaitingRun];
modelSandbox.showToast = (message) => reuseToasts.push(message);
modelSandbox.request = () => { throw new Error("reused runs must not POST"); };
modelSandbox.selectRun = async () => { modelSandbox.state.activeRun = nativeWaitingRun; return true; };
const reusedRun = await modelSandbox.createRun({
  dataset_id: "dataset-a", episode_id: "episode-a", strategy_model_id: "policy-a", review_model_id: "judge-a",
  autonomous_mode: true, model_workflow: "research_compile_evolve@1", knowledge_online_enabled: true,
  rounds: 2, candidates_per_generation: 3, max_candidates: 6, fixed_seed: true, auto_advance: 0,
});
assert.equal(reusedRun.id, nativeWaitingRun.id);
assert.match(reuseToasts.at(-1), /已切换到该运行/);

const insufficientBudgetRun = await modelSandbox.createRun({
  dataset_id: "dataset-a", episode_id: "episode-a", strategy_model_id: "policy-a", review_model_id: "judge-a",
  autonomous_mode: true, model_workflow: "research_compile_evolve@1", knowledge_online_enabled: true,
  rounds: 2, candidates_per_generation: 3, max_candidates: 1, fixed_seed: true, auto_advance: 0,
});
assert.equal(insufficientBudgetRun, null);
assert.match(reuseToasts.at(-1), /候选总预算不足/);

const createRunQuerySelector = modelSandbox.document.querySelector;
const originalEvaluators = modelSandbox.state.catalog.evaluators;
const multiHorizonEvaluatorNode = {value: "greenhouse-multi-horizon@1"};
let samplesPerUpdateFocused = false;
modelSandbox.document.querySelector = (selector) => {
  if (selector === "#evaluator-id") { return multiHorizonEvaluatorNode; }
  if (selector === "#samples-per-update") {
    return {focus() { samplesPerUpdateFocused = true; }};
  }
  return createRunQuerySelector(selector);
};
modelSandbox.state.catalog.evaluators = [{
  id: "greenhouse-multi-horizon@1",
  prediction_task_count: 9,
  minimum_samples_per_update: 9,
}];
const requestBeforeSampleBoundaryTest = modelSandbox.request;
let rejectedSampleBoundaryRequestCount = 0;
modelSandbox.request = async () => { rejectedSampleBoundaryRequestCount += 1; return {}; };
const insufficientSamplesRun = await modelSandbox.createRun({
  dataset_id: "dataset-a", episode_id: "episode-a", strategy_model_id: "policy-a", review_model_id: "judge-a",
  autonomous_mode: true, model_workflow: "research_compile_evolve@1", knowledge_online_enabled: true,
  rounds: 2, candidates_per_generation: 3, max_candidates: 6,
  samples_per_update: 8, fixed_seed: true, auto_advance: 0,
});
assert.equal(insufficientSamplesRun, null);
assert.equal(rejectedSampleBoundaryRequestCount, 0);
assert.equal(samplesPerUpdateFocused, true);
assert.equal(reuseToasts.at(-1), "每次更新样本数不足：当前评测至少需要 9 个，确保每个目标与预测时距至少出现一次。");
modelSandbox.request = requestBeforeSampleBoundaryTest;
modelSandbox.state.catalog.evaluators = originalEvaluators;
modelSandbox.document.querySelector = createRunQuerySelector;

const advanceRunCommand = modelSandbox.advanceRun;
let reusedAdvanceCalls = 0;
modelSandbox.advanceRun = async () => { reusedAdvanceCalls += 1; return true; };
const advancedReusedRun = await modelSandbox.createRun({
  dataset_id: "dataset-a", episode_id: "episode-a", strategy_model_id: "policy-a", review_model_id: "judge-a",
  autonomous_mode: true, model_workflow: "research_compile_evolve@1", knowledge_online_enabled: true,
  rounds: 2, candidates_per_generation: 3, max_candidates: 6, fixed_seed: true,
});
assert.equal(advancedReusedRun.id, nativeWaitingRun.id);
assert.equal(reusedAdvanceCalls, 1);
assert.match(reuseToasts.at(-1), /正在继续执行首轮/);

modelSandbox.state.activeRun = null;
modelSandbox.selectRun = async () => false;
const failedReuse = await modelSandbox.createRun({
  dataset_id: "dataset-a", episode_id: "episode-a", strategy_model_id: "policy-a", review_model_id: "judge-a",
  autonomous_mode: true, model_workflow: "research_compile_evolve@1", knowledge_online_enabled: true,
  rounds: 2, candidates_per_generation: 3, max_candidates: 6, fixed_seed: true, auto_advance: 0,
});
assert.equal(failedReuse, null);
assert.match(reuseToasts.at(-1), /重新读取失败/);

// A continuous create returns immediately at generation 0.  The UI must say
// that work is queued instead of claiming the first generation already ran.
const createToasts = [];
const refreshEventsForRunCommand = modelSandbox.refreshEventsForRun;
modelSandbox.state.runs = [];
modelSandbox.state.activeRun = null;
modelSandbox.state.usingDemo = false;
modelSandbox.state.busy = false;
modelSandbox.state.commandKeys = {};
modelSandbox.showToast = (message) => createToasts.push(message);
modelSandbox.renderAll = () => {};
modelSandbox.loadSelectedDataset = async () => true;
modelSandbox.refreshEventsForRun = async () => true;
modelSandbox.ensureAutoAdvanceForRun = () => true;
modelSandbox.state.candidateSelectionPinned = true;
let queuedCreateBody = null;
modelSandbox.request = async (path, options) => {
  assert.equal(path, "/runs");
  assert.equal(options.method, "POST");
  queuedCreateBody = options.body;
  assert.equal(Object.hasOwn(options.body.budget, "token_limit"), false);
  assert.equal(options.body.samples_per_update, 500);
  assert.equal(options.body.sample_agent_batch_size, 64);
  assert.equal(options.body.sample_concurrency, 2);
  return {
    projection: {
      run_id: "run:new-queued", status: "running", generation: 0, candidates_count: 0,
      auto_progress: true, execution_progress: {phase: "queued", auto_progress: true},
      budget: {max_generations: 2, candidates_per_generation: 3, max_candidates: 6},
      configuration: {
        dataset_id: "dataset-a", episode_id: "episode-a", strategy_model_id: "policy-a", review_model_id: "judge-a",
        samples_per_update: 500, sample_agent_batch_size: 64, sample_concurrency: 2,
      },
    },
  };
};
const newlyQueuedRun = await modelSandbox.createRun({
  dataset_id: "dataset-a", episode_id: "episode-a", strategy_model_id: "policy-a", review_model_id: "judge-a",
  autonomous_mode: true, model_workflow: "research_compile_evolve@1", knowledge_online_enabled: true,
  rounds: 2, candidates_per_generation: 3, max_candidates: 6, fixed_seed: true,
});
assert.equal(queuedCreateBody.execution_protocol, "dsh_native_plugin_evolution@1");
assert.equal(newlyQueuedRun.id, "run:new-queued");
assert.equal(newlyQueuedRun.samples_per_update, 500);
assert.equal(newlyQueuedRun.sample_agent_batch_size, 64);
assert.equal(newlyQueuedRun.sample_concurrency, 2);
assert.equal(modelSandbox.state.createStatus.state, "queued");
assert.equal(modelSandbox.state.candidateSelectionPinned, false);
assert.match(createToasts.at(-1), /已排队/);
assert.doesNotMatch(createToasts.at(-1), /已.*执行首轮/);

// Non-default execution parameters must survive all three boundaries: the
// create request, the server projection, and the process-detail rendering.
let nonDefaultCreateBody = null;
modelSandbox.request = async (path, options) => {
  assert.equal(path, "/runs");
  assert.equal(options.method, "POST");
  nonDefaultCreateBody = options.body;
  return {
    projection: {
      run_id: "run:non-default-parameters", status: "running", generation: 0, candidates_count: 0,
      auto_progress: true, execution_progress: {phase: "queued", auto_progress: true},
      budget: {max_generations: 3, candidates_per_generation: 5, max_candidates: 15, token_limit: 60000000},
      configuration: {
        dataset_id: "dataset-a", episode_id: "episode-a", strategy_model_id: "policy-a", review_model_id: "judge-a",
        samples_per_update: 321, sample_agent_batch_size: 16, sample_concurrency: 3,
      },
    },
  };
};
const nonDefaultRun = await modelSandbox.createRun({
  dataset_id: "dataset-a", episode_id: "episode-a", strategy_model_id: "policy-a", review_model_id: "judge-a",
  autonomous_mode: true, model_workflow: "research_compile_evolve@1", knowledge_online_enabled: true,
  rounds: 3, candidates_per_generation: 5, max_candidates: 15,
  samples_per_update: 321, sample_agent_batch_size: 16, sample_concurrency: 3,
  fixed_seed: true,
});
assert.equal(nonDefaultCreateBody.budget.max_generations, 3);
assert.equal(nonDefaultCreateBody.budget.candidates_per_generation, 5);
assert.equal(nonDefaultCreateBody.samples_per_update, 321);
assert.equal(nonDefaultCreateBody.sample_agent_batch_size, 16);
assert.equal(nonDefaultCreateBody.sample_concurrency, 3);
assert.equal(nonDefaultRun.total_generations, 3);
assert.equal(nonDefaultRun.candidates_per_generation, 5);
assert.equal(nonDefaultRun.samples_per_update, 321);
assert.equal(nonDefaultRun.sample_agent_batch_size, 16);
assert.equal(nonDefaultRun.sample_concurrency, 3);

modelSandbox.document.querySelector = (selector) => selector === "#process-summary" ? processSummaryNode : modelQuerySelector(selector);
modelSandbox.renderProcessSummary(nonDefaultRun);
for (const nonDefaultExecutionParameter of [
  "每轮候选", "5 个版本", "每轮智能体样本", "321 个固定反馈样本",
  "请求微批", "16 个样本", "逐样本并发", "3 个在飞请求",
]) {
  assert.ok(processSummaryNode.innerHTML.includes(nonDefaultExecutionParameter));
}
modelSandbox.document.querySelector = modelQuerySelector;
modelSandbox.selectRun = selectRunCommand;
modelSandbox.refreshEventsForRun = refreshEventsForRunCommand;

// A failed background projection carries its public stage error into the
// persistent banner and emits one transition toast, not one per poll.
const pollToasts = [];
const pollingRun = modelSandbox.normalizeRun({
  ...waitingRun, run_id: "run:poll-failure", projection_revision: 1, auto_progress: true,
  execution_progress: {phase: "training", auto_progress: true},
});
const failedPollingRun = {
  ...pollingRun, status: "failed", projection_revision: 2,
  execution_progress: {phase: "failed", auto_progress: true},
};
modelSandbox.state.activeRun = pollingRun;
modelSandbox.state.runs = [pollingRun];
modelSandbox.state.events = [];
modelSandbox.state.createStatus = modelSandbox.createStatusForRun(pollingRun, []);
modelSandbox.state.autoAdvanceRunId = pollingRun.id;
modelSandbox.state.autoAdvanceContextEpoch = modelSandbox.state.contextEpoch;
modelSandbox.showToast = (message) => pollToasts.push(message);
modelSandbox.renderContext = () => {};
modelSandbox.renderProcess = () => {};
modelSandbox.renderCandidates = () => {};
modelSandbox.renderTrainingAssets = () => {};
modelSandbox.renderCollaboration = () => {};
modelSandbox.request = async (path) => path.endsWith("/events") ? {
  events: [{
    event_id: "event:failed-stage", type: "stage.recorded", occurred_at: "2026-08-18T06:00:00Z",
    payload: {status: "failed", public_error: "本轮证据门禁失败：未产生新的科学评测。"},
  }],
} : {projection: failedPollingRun};
assert.equal(await modelSandbox.refreshProgressForRun(pollingRun.id), true);
assert.equal(modelSandbox.state.activeRun.status, "failed");
assert.match(modelSandbox.state.commandError, /本轮证据门禁失败/);
assert.equal(modelSandbox.state.createStatus.state, "failed");
assert.equal(modelSandbox.state.autoAdvanceRunId, null);
assert.equal(pollToasts.length, 1);
assert.equal(await modelSandbox.refreshProgressForRun(pollingRun.id), true);
assert.equal(pollToasts.length, 1);

// A later run read owns the event view even if an older poll resolves last.
const raceRun = modelSandbox.normalizeRun({...waitingRun, run_id: "run:poll-race", projection_revision: 2});
modelSandbox.state.activeRun = raceRun;
modelSandbox.state.runs = [raceRun];
modelSandbox.state.events = [{id: "event:initial", type: "run.started", occurred_at: "2026-08-18T05:00:00Z", payload: {}}];
let resolveOldRun;
let resolveOldEvents;
const oldRunResponse = new Promise((resolve) => { resolveOldRun = resolve; });
const oldEventResponse = new Promise((resolve) => { resolveOldEvents = resolve; });
modelSandbox.request = (path) => path.endsWith("/events") ? oldEventResponse : oldRunResponse;
const oldPoll = modelSandbox.refreshProgressForRun(raceRun.id);
modelSandbox.request = async () => ({
  events: [{event_id: "event:newer", type: "stage.recorded", occurred_at: "2026-08-18T07:00:00Z", payload: {status: "running"}}],
});
assert.equal(await modelSandbox.refreshEventsForRun(raceRun.id), true);
resolveOldRun({projection: {...raceRun, projection_revision: 3}});
resolveOldEvents({events: [{event_id: "event:older", type: "stage.recorded", occurred_at: "2026-08-18T06:00:00Z", payload: {status: "running"}}]});
assert.equal(await oldPoll, false);
assert.equal(modelSandbox.state.activeRun.projection_revision, 2);
assert.equal(modelSandbox.state.events[0].id, "event:newer");

// A replica response behind the accepted projection must not replace its events.
modelSandbox.request = async (path) => path.endsWith("/events")
  ? {events: [{event_id: "event:stale", type: "stage.recorded", occurred_at: "2026-08-18T04:00:00Z", payload: {}}]}
  : {projection: {...raceRun, projection_revision: 1}};
assert.equal(await modelSandbox.refreshProgressForRun(raceRun.id), false);
assert.equal(modelSandbox.state.events[0].id, "event:newer");

// A failed run switch leaves the current page and its automatic progression intact.
const targetRun = modelSandbox.normalizeRun({...waitingRun, id: "run:switch-target", run_id: "run:switch-target", projection_revision: 1});
modelSandbox.state.activeRun = raceRun;
modelSandbox.state.runs = [raceRun, targetRun];
modelSandbox.state.createStatus = {runId: raceRun.id, state: "running"};
modelSandbox.state.autoAdvanceRunId = raceRun.id;
modelSandbox.state.autoAdvanceContextEpoch = modelSandbox.state.contextEpoch;
modelSandbox.state.autoAdvanceLastDurationMs = 4321;
const keptAutoAdvanceTimer = {id: "kept-auto-advance-timer"};
modelSandbox.state.autoAdvanceTimer = keptAutoAdvanceTimer;
modelSandbox.state.selectedCandidateId = "candidate:kept";
raceRun.candidates = [{id: "candidate:kept", status: "evaluated"}];
modelSandbox.resetCandidateSamples(raceRun.id, "candidate:kept");
modelSandbox.state.candidateSamplePage = {run_id: raceRun.id, candidate_id: "candidate:kept", offset: 0, rows: [{sample_id: "sample:kept"}]};
let stoppedSelectionMonitor = null;
let restoredSelectionMonitor = null;
modelSandbox.state.runMonitorRunId = raceRun.id;
modelSandbox.stopRunMonitor = (runId) => { stoppedSelectionMonitor = runId; modelSandbox.state.runMonitorRunId = null; };
modelSandbox.startRunMonitor = (runId) => { restoredSelectionMonitor = runId; modelSandbox.state.runMonitorRunId = runId; return true; };
modelSandbox.request = async () => { throw new Error("switch failed"); };
assert.equal(await modelSandbox.selectRun(targetRun.id, false), false);
assert.equal(modelSandbox.state.activeRun.id, raceRun.id);
assert.equal(modelSandbox.state.candidateSamplePage.rows[0].sample_id, "sample:kept");
assert.equal(modelSandbox.candidateSampleSelectionMatches(raceRun.id, "candidate:kept"), true);
assert.equal(modelSandbox.state.createStatus.runId, raceRun.id);
assert.equal(modelSandbox.state.autoAdvanceRunId, raceRun.id);
assert.equal(modelSandbox.state.autoAdvanceTimer, keptAutoAdvanceTimer);
assert.equal(modelSandbox.state.autoAdvanceLastDurationMs, 4321);
assert.equal(stoppedSelectionMonitor, raceRun.id);
assert.equal(restoredSelectionMonitor, raceRun.id);

// A successful switch owns the banner state; errors from the previous run
// must not leak into a healthy historical run.
modelSandbox.clearAutoAdvanceTimer();
modelSandbox.state.autoAdvanceRunId = null;
modelSandbox.state.commandError = "后台进化失败：旧运行失败";
modelSandbox.state.autoAdvanceBlockedRunId = "run:old-failure";
modelSandbox.state.autoAdvanceError = "旧运行自动推进失败";
modelSandbox.state.candidateSelectionPinned = true;
modelSandbox.loadSelectedDataset = async () => true;
modelSandbox.loadCandidateSamples = async () => true;
modelSandbox.ensureAutoAdvanceForRun = () => false;
modelSandbox.request = async (path) => path.endsWith("/events")
  ? {events: []}
  : {projection: targetRun};
assert.equal(await modelSandbox.selectRun(targetRun.id, false), true);
assert.equal(modelSandbox.state.activeRun.id, targetRun.id);
assert.equal(modelSandbox.state.commandError, null);
assert.equal(modelSandbox.state.autoAdvanceBlockedRunId, null);
assert.equal(modelSandbox.state.autoAdvanceError, null);
assert.equal(modelSandbox.state.candidateSelectionPinned, false);
assert.equal(modelSandbox.state.lastSelectedRunId, targetRun.id);

// Reconnects prefer the last successfully selected run when it still exists,
// even when another run is newer in the server-sorted history.
const reconnectSandbox = {
  console, URL, URLSearchParams, AbortController, setTimeout, clearTimeout,
  window: {location: {search: ""}, setTimeout, clearTimeout},
  document: {querySelector: modelNode, querySelectorAll: () => []},
  EcologyDSHHost: {getPublicContext: () => ({apiBase: "/api", models: []}), preferredModelId: () => "", request: () => Promise.resolve({})},
};
vm.createContext(reconnectSandbox);
vm.runInContext(read("assets/js/core.js"), reconnectSandbox);
vm.runInContext(read("assets/js/catalog.js"), reconnectSandbox);
vm.runInContext(read("assets/js/data.js"), reconnectSandbox);
reconnectSandbox.state.lastSelectedRunId = "run:preferred";
reconnectSandbox.renderAll = () => {};
reconnectSandbox.setConnection = () => {};
reconnectSandbox.populateCatalogControls = () => {};
let reconnectSelectedRunId = null;
reconnectSandbox.selectRun = async (runId) => { reconnectSelectedRunId = runId; return true; };
reconnectSandbox.request = async (path) => {
  if (path === "/health") { return {ok: true}; }
  if (path === "/catalog") { return {dsh: {capabilities: []}}; }
  return {runs: [
    {run_id: "run:newest", status: "running", created_at: "2026-08-19T09:00:00Z"},
    {run_id: "run:preferred", status: "running", created_at: "2026-08-18T09:00:00Z"},
  ]};
};
assert.equal(await reconnectSandbox.connectAndLoad(), true);
assert.equal(reconnectSelectedRunId, "run:preferred");

modelSandbox.advanceRun = advanceRunCommand;
modelSandbox.state.usingDemo = true;
modelSandbox.state.catalog.dsh.capabilities = ["run.control", "evolution.run.advance"];
modelSandbox.state.activeRun = modelSandbox.normalizeRun({
  ...waitingBetweenRounds,
  run_id: "run:paused-advance",
  status: "paused",
});
modelSandbox.state.runs = [modelSandbox.state.activeRun];
modelSandbox.state.events = [];
modelSandbox.state.busy = false;
modelSandbox.renderAll = () => {};
modelSandbox.refreshEventsForRun = async () => true;
modelSandbox.reconcileVisibleRunSelection = () => false;
assert.equal(await modelSandbox.advanceRun(), true);
assert.equal(modelSandbox.state.activeRun.status, "running");
assert.equal(modelSandbox.state.activeRun.generation, 2);
assert.deepEqual(
  Array.from(modelSandbox.state.events.map((event) => event.type)),
  ["generation.advanced", "run.resumed"],
);

const cleanupRequests = [];
modelSandbox.state.usingDemo = false;
modelSandbox.state.hostContextReceived = false;
modelSandbox.state.catalog.dsh.capabilities = ["run.archive", "run.delete"];
modelSandbox.state.showArchivedRuns = true;
modelSandbox.state.archivedRunCount = 0;
modelSandbox.state.busy = false;
modelSandbox.state.activeRun = modelSandbox.normalizeRun({run_id: "run:cleanup-ui", status: "cancelled"});
modelSandbox.state.runs = [modelSandbox.state.activeRun];
modelSandbox.request = async (path, options) => {
  cleanupRequests.push({path, options});
  if (options.method === "DELETE") return {ok: true};
  if (path.endsWith("/restore")) {
    return {projection: {run_id: "run:cleanup-ui", status: "cancelled", archived: false, archived_at: null}};
  }
  return {projection: {run_id: "run:cleanup-ui", status: "cancelled", archived: true, archived_at: "2026-08-17T00:00:00Z"}};
};
assert.equal(await modelSandbox.archiveRun(), true);
assert.equal(cleanupRequests[0].path, "/runs/run%3Acleanup-ui/archive");
assert.equal(cleanupRequests[0].options.method, "POST");
assert.equal(modelSandbox.state.activeRun.archived, true);
assert.equal(modelSandbox.state.archivedRunCount, 1);
assert.equal(await modelSandbox.archiveRun(), true);
assert.equal(cleanupRequests[1].path, "/runs/run%3Acleanup-ui/restore");
assert.equal(cleanupRequests[1].options.method, "POST");
assert.equal(modelSandbox.state.activeRun.archived, false);
assert.equal(modelSandbox.state.archivedRunCount, 0);
assert.equal(await modelSandbox.archiveRun(), true);
assert.equal(cleanupRequests[2].path, "/runs/run%3Acleanup-ui/archive");
assert.equal(modelSandbox.state.activeRun.archived, true);
assert.equal(modelSandbox.state.archivedRunCount, 1);
assert.equal(await modelSandbox.deleteRun("wrong"), false);
assert.equal(cleanupRequests.length, 3);
assert.equal(await modelSandbox.deleteRun("run:cleanup-ui"), true);
assert.equal(cleanupRequests[3].path, "/runs/run%3Acleanup-ui");
assert.equal(cleanupRequests[3].options.method, "DELETE");
assert.equal(cleanupRequests[3].options.body.confirm_run_id, "run:cleanup-ui");
assert.equal(modelSandbox.state.runs.length, 0);
assert.equal(modelSandbox.state.archivedRunCount, 0);

const consultationRequests = [];
modelSandbox.state.catalog.dsh.capabilities = ["intervention.write"];
modelSandbox.state.busy = false;
modelSandbox.state.commandKeys = {};
modelSandbox.state.activeRun = modelSandbox.normalizeRun({
  run_id: "run:consultation-ui", status: "running", generation: 4, projection_revision: 1,
  expert_consultations: [{consultation_id: "consultation:1", status: "pending", question: "是否保留保守阈值？", non_blocking: true}],
});
modelSandbox.state.runs = [modelSandbox.state.activeRun];
const savedDraft = modelSandbox.expertConsultationDraft("run:consultation-ui", "consultation:1");
savedDraft.answer = "保留阈值并补做敏感性分析";
savedDraft.answered_by = "领域专家";
assert.equal(modelSandbox.expertConsultationDraft("run:consultation-ui", "consultation:1").answer, "保留阈值并补做敏感性分析");
modelSandbox.request = async (path, options) => {
  consultationRequests.push({path, options});
  return {projection: {
    run_id: "run:consultation-ui", status: "running", generation: 4, projection_revision: 2,
    expert_consultations: [{consultation_id: "consultation:1", status: "answered", question: "是否保留保守阈值？", non_blocking: true, answer: options.body.answer, selected_option: options.body.selected_option, answered_by: options.body.answered_by, answered_at: "2026-08-20T00:00:00Z", effective_generation: 5, applied_generation: null}],
  }};
};
assert.equal(await modelSandbox.answerExpertConsultation("consultation:1", {answer: savedDraft.answer, selected_option: "keep", answered_by: savedDraft.answered_by}), true);
assert.equal(consultationRequests[0].path, "/runs/run%3Aconsultation-ui/expert-consultations/consultation%3A1/answer");
assert.equal(consultationRequests[0].options.method, "POST");
assert.equal(consultationRequests[0].options.body.answer, "保留阈值并补做敏感性分析");
assert.equal(consultationRequests[0].options.body.selected_option, "keep");
assert.equal(consultationRequests[0].options.body.answered_by, "领域专家");
assert.match(consultationRequests[0].options.body.idempotency_key, /^plugin-expert-consultation-answer-/);
assert.equal(modelSandbox.state.activeRun.expert_consultations[0].status, "answered");
assert.equal(modelSandbox.state.activeRun.expert_consultations[0].effective_generation, 5);
assert.equal(Object.hasOwn(modelSandbox.state.expertConsultationDrafts, "run:consultation-ui::consultation:1"), false);
modelSandbox.state.activeRun = modelSandbox.normalizeRun({run_id: "run:consultation-ended", status: "completed", expert_consultations: [{consultation_id: "consultation:ended", status: "pending", question: "迟到的问题"}]});
modelSandbox.state.runs = [modelSandbox.state.activeRun];
modelSandbox.request = async (path, options) => {
  consultationRequests.push({path, options});
  return {projection: {
    run_id: "run:consultation-ended", status: "completed", generation: 4,
    expert_consultations: [{consultation_id: "consultation:ended", status: "answered", question: "迟到的问题", non_blocking: true, answer: options.body.answer, answered_by: options.body.answered_by, answered_at: "2026-08-20T01:00:00Z", effective_generation: null, applied_generation: null}],
  }};
};
assert.equal(await modelSandbox.answerExpertConsultation("consultation:ended", {answer: "迟到答复", answered_by: "专家乙"}), true);
assert.equal(consultationRequests.length, 2);
assert.equal(consultationRequests[1].path, "/runs/run%3Aconsultation-ended/expert-consultations/consultation%3Aended/answer");
assert.equal(modelSandbox.state.activeRun.expert_consultations[0].effective_generation, null);

const liveFilterRun = modelSandbox.normalizeRun({run_id: "run:filter-live", status: "running"});
const archivedFilterRun = modelSandbox.normalizeRun({
  run_id: "run:filter-archived", status: "completed", archived: true, archived_at: "2026-08-17T00:00:00Z",
});
modelSandbox.state.activeRun = liveFilterRun;
modelSandbox.state.runs = [liveFilterRun];
modelSandbox.state.showArchivedRuns = false;
modelSandbox.state.busy = false;
modelSandbox.request = async (path) => {
  cleanupRequests.push({path});
  return {runs: [liveFilterRun, archivedFilterRun], archived_count: 1};
};
assert.equal(await modelSandbox.setArchivedRunsVisible(true), true);
assert.equal(cleanupRequests.at(-1).path, "/runs?view=summary&include_archived=true");
assert.deepEqual(Array.from(modelSandbox.state.runs.map((run) => run.id)), ["run:filter-live", "run:filter-archived"]);
assert.equal(modelSandbox.state.archivedRunCount, 1);
assert.equal(modelSandbox.state.activeRun.id, "run:filter-live");
assert.equal(await modelSandbox.setArchivedRunsVisible(false), true);
assert.deepEqual(Array.from(modelSandbox.state.runs.map((run) => run.id)), ["run:filter-live"]);
assert.equal(modelSandbox.state.activeRun.id, "run:filter-live");

const sharedCatalog = modelSandbox.normalizeCatalog({
  models: [{id: "local", local_model: true, roles: ["propose"]}],
  dsh_models: [
    {id: "shared", roles: ["propose", "judge"], configured: true, directory_available: true, execution_available: true, credential_configured: true, authentication_verified: true},
    {id: "pending", roles: ["judge"], configured: true, directory_available: true, execution_available: true, credential_configured: true, authentication_verified: false},
    {id: "missing", roles: ["propose"], configured: false, directory_available: true, execution_available: false, credential_configured: false, authentication_verified: false},
  ],
});
assert.equal(JSON.stringify(sharedCatalog.policy_models.map((item) => item.id)), JSON.stringify(["shared", "missing"]));
assert.equal(JSON.stringify(sharedCatalog.judge_models.map((item) => item.id)), JSON.stringify(["shared", "pending"]));
assert.equal(modelSandbox.modelCredentialReady(sharedCatalog.judge_models[1]), true);
modelSandbox.state.catalog = sharedCatalog;
modelSandbox.state.usingDemo = false;
modelSandbox.state.hostContextReceived = true;
modelSandbox.state.hostContext = {models: [{id: "shared"}, {id: "unregistered"}]};
const mergedHostItems = modelSandbox.autonomousModelItems();
assert.equal(JSON.stringify(mergedHostItems.map((item) => item.id)), JSON.stringify(["shared", "pending", "missing", "unregistered"]));
assert.equal(mergedHostItems.at(-1).model_source, "dsh_host_only");
assert.equal(mergedHostItems.at(-1).execution_available, false);
assert.equal(mergedHostItems.at(-1).configured, false);
assert.equal(mergedHostItems.at(-1).unavailable_reason.code, "host_route_not_available_to_sidecar");
modelSandbox.state.hostContextReceived = false;
modelSandbox.state.hostContext = {models: []};
assert.equal(JSON.stringify(modelSandbox.autonomousModelItems().map((item) => item.id)), JSON.stringify(["shared", "pending", "missing"]));
const aliasCatalog = modelSandbox.normalizeCatalog({
  dsh_models: [{id: "custom-strategy", model: "glm-5.2", roles: ["propose", "judge"], credential_configured: true, authentication_verified: true}],
});
modelSandbox.state.catalog = aliasCatalog;
modelSandbox.state.hostContextReceived = true;
modelSandbox.state.hostContext = {models: [{id: "newapi/glm-5.2", provider: "newapi", model: "glm-5.2", aliases: ["glm-5.2"]}]};
assert.equal(JSON.stringify(modelSandbox.autonomousModelItems().map((item) => item.id)), JSON.stringify(["custom-strategy"]));
const providerCatalog = modelSandbox.normalizeCatalog({
  dsh_models: [{id: "newapi/glm-5.2", provider: "newapi", model: "glm-5.2", roles: ["propose", "judge"], credential_configured: true, authentication_verified: true}],
});
modelSandbox.state.catalog = providerCatalog;
modelSandbox.state.hostContext = {models: [{id: "pjlab/glm-5.2", provider: "pjlab", model: "glm-5.2", aliases: ["glm-5.2"]}]};
const crossProviderItems = modelSandbox.autonomousModelItems();
assert.equal(JSON.stringify(crossProviderItems.map((item) => item.id)), JSON.stringify(["newapi/glm-5.2", "pjlab/glm-5.2"]));
assert.equal(crossProviderItems[1].model_source, "dsh_host_only");
const demoCatalog = modelSandbox.normalizeCatalog({
  models: [{id: "local", local_model: true, roles: ["propose"]}],
  policy_models: [{id: "local", local_model: true, roles: ["propose"]}],
  judge_models: [{id: "judge", local_model: true, roles: ["judge"]}],
});
modelSandbox.state.catalog = demoCatalog;
modelSandbox.state.usingDemo = true;
modelSandbox.state.hostContextReceived = false;
modelSandbox.state.hostContext = {models: []};
assert.equal(JSON.stringify(modelSandbox.autonomousModelItems().map((item) => item.id)), JSON.stringify(["local", "judge"]));

const exhaustedProjection = modelSandbox.normalizeRun({
  run_id: "run:exhausted",
  status: "completed",
  total_generations: 2,
  execution_progress: {progress_percent: 100, completed_steps: 5, total_steps: 12, phase: "completed"},
  candidates: [
    {candidate_id: "candidate:low", score: -0.4, status: "rejected"},
    {candidate_id: "candidate:near-miss", score: -0.1, status: "rejected"},
  ],
});
assert.equal(exhaustedProjection.outcome, "budget_exhausted_without_acceptable_candidate");
assert.equal(exhaustedProjection.best_candidate_id, null);
assert.equal(exhaustedProjection.best_observed_candidate_id, "candidate:near-miss");
assert.equal(exhaustedProjection.best_observed_score, -0.1);
assert.equal(
  modelSandbox.displayRunStatusText(exhaustedProjection, []),
  "已完成预设进化规模（2 代、2 个候选），尚无候选通过全部评测门控；正式验证未开展",
);
assert.equal(modelSandbox.displayRunStatusClass(exhaustedProjection, []), "pill-amber");
modelSandbox.document.querySelector = (selector) => monitorNodes[selector] || monitorQuerySelector(selector);
modelSandbox.state.events = [];
modelSandbox.renderExecutionMonitor(exhaustedProjection);
assert.ok(monitorNodes["#execution-progress-detail"].textContent.includes("原始最高观测（跨窗口不可直接比较）"));
assert.equal(monitorNodes["#execution-progress-detail"].textContent.includes("当前最高分"), false);
modelSandbox.document.querySelector = monitorQuerySelector;
assert.equal(modelSandbox.runOutcomeCode({outcome: "accepted"}), "completed_with_acceptable_candidate");
assert.equal(modelSandbox.runOutcomeCode({outcome: "completed_with_search_retained_candidate"}), "completed_with_acceptable_candidate");
assert.equal(
  modelSandbox.displayRunStatusText({status: "completed", outcome: "completed_with_search_retained_candidate"}, []),
  "已完成，产生训练反馈搜索保留候选；正式验证未开展",
);
vm.runInContext(read("assets/js/render_process.js"), modelSandbox);
modelSandbox.state.pendingAction = null;
// The process monitor must prefer the live paged sample heartbeat over the
// bounded candidate projection preview, so predictions become visible during
// evaluation rather than only after the candidate is terminal.
const liveSampleRun = {id: "run:live-samples", status: "running"};
const liveSampleCandidate = {
  id: "candidate:live-samples", status: "evaluating",
  inference_trace: {sample_count: 1, rows: [{sample_id: "projection-row", predicted: 1}]},
};
modelSandbox.state.activeRun = liveSampleRun;
modelSandbox.state.selectedCandidateId = liveSampleCandidate.id;
modelSandbox.state.candidateSampleSelection = {run_id: liveSampleRun.id, candidate_id: liveSampleCandidate.id};
modelSandbox.state.candidateSamplePage = {
  total: 2, truncated: true, complete: false,
  rows: [{sample_id: "live-row", predicted: 2.5, observed: 2.0, baseline: 1.0, reward: 0.5}],
};
assert.deepEqual(
  Array.from(modelSandbox.executionPredictionRows(liveSampleCandidate, liveSampleRun).map((row) => row.sample_id)),
  ["live-row"],
);
const liveTrace = modelSandbox.executionPredictionTrace(liveSampleCandidate, liveSampleRun);
assert.equal(liveTrace.sample_count, 2);
assert.equal(liveTrace.truncated, true);
modelSandbox.state.candidateSamplePage = null;
const exhaustedProgress = modelSandbox.executionProgress(exhaustedProjection, null, []);
assert.ok(exhaustedProgress.percent < 100);
assert.equal(Math.round(exhaustedProgress.percent), 42);
const researchEvidenceHtml = modelSandbox.renderRoundResearchEvidence({
  generation: 1,
  research_iteration: {
    status: "model_generated",
    iteration_digest: "sha256:research-iteration",
    analysis_summary: {
      status: "completed",
      summary: "比较公开证据后，采用已登记岭回归 <b>HTML-ANALYSIS-MARKER</b>。",
      evidence_refs: ["knowledge:paper"],
      key_findings: [{finding: "short history is stable"}],
      private_reasoning: "PRIVATE-ROUND-REASONING",
    },
    final_plan: {
      status: "ready_for_host_compilation",
      predictor_id: "greenhouse-exogenous-ridge@1",
      pipeline_id: "greenhouse-ridge-pipeline@1<script>HTML-PLAN-MARKER</script>",
      operator_ids: ["ridge.fit@1"],
      parameter_focus: ["ridge_alpha"],
      implementation_mode: "registered_host_components_only",
      private_reasoning: "PRIVATE-PLAN-REASONING",
    },
  },
  candidates: [{candidate_id: "candidate:research-chain", stages: {evaluation: "completed", judge: "completed"}}],
}, {
  candidates: [{
    id: "candidate:research-chain", generation: 1,
    algorithm_execution: {
      status: "debug_passed", training_authorized: true,
      algorithm_spec: {algorithm_id: "greenhouse-ridge-pipeline@1"},
      attempts: [
        {phase: "compile", status: "passed", evidence: {private_reasoning: "PRIVATE-COMPILE-REASONING"}},
        {phase: "debug", status: "passed", evidence: {source_partition: "training_fit"}},
      ],
    },
    execution: {stages: {evaluation: "completed", judge: "completed"}},
  }],
});
for (const text of ["分析总结 → 最终方案 → 实现 → 测试", "分析总结", "最终方案", "注册与编译", "training_fit smoke", "training_feedback 正式评测", "独立评审", "仅宿主登记组件"]) {
  assert.ok(researchEvidenceHtml.includes(text), `missing research iteration evidence: ${text}`);
}
assert.equal((researchEvidenceHtml.match(/class="round-research-step is-complete"/g) || []).length, 6);
assert.ok(researchEvidenceHtml.includes("&lt;b&gt;HTML-ANALYSIS-MARKER&lt;/b&gt;"));
assert.ok(researchEvidenceHtml.includes("&lt;script&gt;HTML-PLAN-MARKER&lt;/script&gt;"));
assert.equal(researchEvidenceHtml.includes("<b>HTML-ANALYSIS-MARKER</b>"), false);
assert.equal(researchEvidenceHtml.includes("<script>HTML-PLAN-MARKER</script>"), false);
for (const hiddenMarker of ["PRIVATE-ROUND-REASONING", "PRIVATE-PLAN-REASONING", "PRIVATE-COMPILE-REASONING"]) {
  assert.equal(researchEvidenceHtml.includes(hiddenMarker), false, `private reasoning leaked: ${hiddenMarker}`);
}
const generationMatchedCandidates = modelSandbox.roundResearchCandidates({generation: 1, candidates: []}, {
  candidates: [
    {id: "candidate:internal-zero-based-distractor", generation: 0},
    {id: "candidate:public-first-round", generation: 1},
    {id: "candidate:public-second-round", generation: 2},
  ],
});
assert.deepEqual(
  Array.from(generationMatchedCandidates, (candidate) => candidate.id),
  ["candidate:public-first-round"],
  "round and candidate projections must use the same public 1-based generation",
);

// Newer catalogues expose DSH inventory counts while older ones expose
// credential/configuration counts.  Both shapes must keep readiness and the
// model connection summary truthful.
modelSandbox.state.catalog = modelSandbox.normalizeCatalog({
  dsh: {
    dsh_model_count: 7,
    authenticated_dsh_model_count: 4,
    dsh_strategy_model_count: 7,
    dsh_review_model_count: 7,
  },
  dsh_models: [
    {id: "newapi/glm-5.2", roles: ["propose", "judge"], credential_configured: true, authentication_verified: true},
    {id: "pjlab/deepseek-v4-pro-0813", roles: ["propose", "judge"], credential_configured: true, authentication_verified: true},
  ],
  authenticated_models: [
    {id: "newapi/glm-5.2", roles: ["propose", "judge"], credential_configured: true, authentication_verified: true},
  ],
});
modelSandbox.state.hostContextReceived = true;
modelSandbox.state.hostContext = {models: [
  {id: "newapi/glm-5.2", provider: "newapi", model: "glm-5.2"},
  {id: "pjlab/deepseek-v4-pro-0813", provider: "pjlab", model: "deepseek-v4-pro-0813"},
  {id: "deepseek-official/deepseek-chat", provider: "deepseek-official", model: "deepseek-chat"},
  {id: "deepseek-official/deepseek-reasoner", provider: "deepseek-official", model: "deepseek-reasoner"},
]};
assert.equal(modelSandbox.dshModelTotalCount(false), 9);
assert.equal(modelSandbox.dshModelTotalCount(true), 4);
assert.equal(modelSandbox.dshModelRoleCount("strategy", false), 9);
assert.equal(modelSandbox.dshModelRoleCount("review", false), 9);
assert.equal(modelSandbox.dshModelRoleCount("strategy", true), 2);
const legacyCounts = modelSandbox.normalizeCatalog({
  dsh: {
    configured_model_count: 3,
    authenticated_model_count: 2,
    configured_strategy_model_count: 3,
    configured_review_model_count: 3,
    authenticated_strategy_model_count: 2,
    authenticated_review_model_count: 2,
  },
});
modelSandbox.state.catalog = legacyCounts;
assert.equal(modelSandbox.dshModelTotalCount(false), 3);
assert.equal(modelSandbox.dshModelTotalCount(true), 2);
assert.equal(modelSandbox.dshModelRoleCount("strategy", false), 3);
assert.equal(modelSandbox.dshModelRoleCount("review", true), 2);
assert.equal(modelSandbox.preferredCatalogModelId(
  [{id: "current"}, {id: "host"}],
  ["current", "host"],
  "",
), "current");
assert.equal(modelSandbox.preferredCatalogModelId(
  [{id: "current"}, {id: "host"}],
  ["missing", "host"],
  "",
), "host");
assert.match(app, /dshModelRoleCount\("strategy", false\)/);
assert.match(app, /dshModelRoleCount\("review", false\)/);
assert.match(app, /preferredCatalogModelId\(judgeCompatibleItems/);
assert.match(app, /\[currentPolicyId, hostPolicyPreference\]/);
assert.match(app, /\[selectedJudgeId, hostJudgePreference\]/);
assert.doesNotMatch(String(modelSandbox.populateCatalogControls), /activeRun|activeConfiguration/);
assert.doesNotMatch(catalogSource, /\bsyncFormToRun\(\);/);
assert.doesNotMatch(commandsSource, /\bsyncFormToRun\(\);/);
assert.doesNotMatch(catalogSource, /function syncFormToRun\s*\(/);
assert.match(html, /id="candidate-follow-active"[^>]*checked/);
assert.match(css, /\.candidate-follow-control/);

// The trajectory renderer has no DOM dependency; exercise the browser-safe
// projection shape directly so a renamed field cannot silently blank the
// training asset details panel.
const traceSandbox = {
  compactTechnicalText: (value) => String(value ?? ""),
  humanizeTechnicalText: (value) => String(value ?? ""),
  formatNumber: (value) => value == null ? "—" : String(value),
  formatObservationTime: (value) => String(value ?? ""),
  formatDate: (value) => String(value ?? ""),
  escapeHTML: (value) => String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"),
  shortId: (value) => String(value ?? ""),
};
vm.createContext(traceSandbox);
vm.runInContext(read("assets/js/render_training_trace.js"), traceSandbox);
const traceHtml = traceSandbox.renderTrainingTrajectory({
  input: {dataset_id: "dataset-demo"},
  trajectory_summary: {total_stage_count: 8, sample_count: 1},
  trajectory: {
    input_context: {status: "completed", dataset_id: "dataset-demo"},
    agent_research: {status: "completed", research: ["滚动残差"]},
    agent_proposal: {status: "completed", rationale: "调整历史窗口", parameter_changes: [{parameter: "window", before: 4, after: 5}]},
    host_compile: {status: "completed"},
    training_prediction: {status: "completed", prediction_records: [{sample_index: 1, target: "soil_moisture", input_reference: {partition: "training_feedback"}, observed_value: 0.3, predicted_value: 0.29, baseline_value: 0.28, error: -0.01}]},
    agent_feedback: {status: "completed", current_candidate: {score: 0.8, metrics: {rmse: 0.1}}},
    agent_optimization: {status: "completed", decision: "搜索保留"},
    final_result: {status: "completed", decision: "approved"},
  },
});
for (const text of ["样本输入", "智能体调研", "调整历史窗口", "候选训练", "0.29", "智能体反馈", "搜索保留", "最终结果"]) {
  assert.ok(traceHtml.includes(text), `trajectory renderer missing: ${text}`);
}

const makeControlNode = (value = "") => ({
  value, textContent: "", innerHTML: "", validationMessage: "",
  setCustomValidity(message) { this.validationMessage = message; },
  focus() {},
});
const budgetNodes = {
  "#max-generations": makeControlNode("5"),
  "#candidates-per-generation": makeControlNode("4"),
  "#samples-per-update": makeControlNode("500"),
  "#sample-agent-batch-size": makeControlNode("64"),
  "#sample-concurrency": makeControlNode("8"),
  "#max-candidates": makeControlNode("20"),
  "#max-candidates-help": makeControlNode(),
};
const budgetSandbox = {
  console, URL, URLSearchParams, AbortController, setTimeout, clearTimeout,
  window: {location: {search: ""}, setTimeout, clearTimeout},
  document: {querySelector: (selector) => budgetNodes[selector] || makeControlNode(), querySelectorAll: () => []},
  EcologyDSHHost: {getPublicContext: () => ({apiBase: "/api"}), request: () => Promise.resolve({})},
};
vm.createContext(budgetSandbox);
vm.runInContext(read("assets/js/core.js"), budgetSandbox);
vm.runInContext(read("assets/js/commands.js"), budgetSandbox);
budgetNodes["#max-generations"].value = "6";
budgetSandbox.syncCandidateBudget();
assert.equal(budgetNodes["#max-candidates"].value, "24");
budgetNodes["#max-candidates"].value = "30";
budgetSandbox.syncCandidateBudget({markManual: true});
budgetNodes["#candidates-per-generation"].value = "6";
budgetSandbox.syncCandidateBudget();
assert.equal(budgetNodes["#max-candidates"].value, "30");
assert.match(budgetNodes["#max-candidates"].validationMessage, /至少 36/);
assert.equal(budgetSandbox.candidateBudgetStatus().budget_sufficient, false);
assert.equal(budgetSandbox.normalizedSamplesPerUpdate("500"), 500);
assert.equal(budgetSandbox.normalizedSamplesPerUpdate("invalid"), 500);
assert.equal(budgetSandbox.normalizedSampleAgentBatchSize("64"), 64);
assert.equal(budgetSandbox.normalizedSampleAgentBatchSize("invalid"), 64);
assert.equal(budgetSandbox.normalizedSampleConcurrency("8"), 8);
assert.equal(budgetSandbox.normalizedSampleConcurrency("99"), 8);

const diagnosticNodes = {
  "#execution-diagnostics-summary": makeControlNode(),
  "#execution-diagnostics-grid": makeControlNode(),
};
const diagnosticSandbox = {
  console, URL, URLSearchParams, AbortController, setTimeout, clearTimeout,
  window: {location: {search: ""}, setTimeout, clearTimeout, innerWidth: 1280},
  document: {querySelector: (selector) => diagnosticNodes[selector] || null, querySelectorAll: () => []},
  EcologyDSHHost: {getPublicContext: () => ({apiBase: "/api"}), request: () => Promise.resolve({})},
};
vm.createContext(diagnosticSandbox);
vm.runInContext(read("assets/js/core.js"), diagnosticSandbox);
vm.runInContext(read("assets/js/render_process.js"), diagnosticSandbox);
diagnosticSandbox.renderExecutionDiagnostics({execution_diagnostics: {
  execution_mode: "registered_lightweight", fit_method: "bias_fit",
  training_partition_rows: 144, training_eligible_examples: 144, training_used_examples: 144, training_skipped_examples: 0,
  evaluation_partition_rows: 48, evaluation_eligible_examples: 48, evaluation_used_examples: 48, evaluation_skipped_examples: 0,
  candidate_artifacts_count: 4, candidate_evaluations_count: 4, candidate_work_items: 192,
  fit_passes_completed: 4, fit_passes_per_candidate: 1, iterative_epoch_training: false,
  proposal_sources: {remote_model: 4, host_fallback: 1}, remote_strategy_status: "partial_host_fallback",
}});
assert.equal(diagnosticNodes["#execution-diagnostics-summary"].textContent, "累计候选工作量 192 · 拟合 pass 4");
assert.ok(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("累计扫描 144 行次"));
assert.ok(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("策略模型 API 4"));
assert.ok(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("宿主有界回退 1"));
assert.ok(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("无神经网络 epoch"));
assert.equal(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("[object Object]"), false);

const partialDiagnostics = {
  execution_mode: "pending", fit_method: null,
  training_partition_rows: 0, training_eligible_examples: 0, training_used_examples: 0, training_skipped_examples: 0,
  evaluation_partition_rows: 0, evaluation_eligible_examples: 0, evaluation_used_examples: 0, evaluation_skipped_examples: 0,
  live_evaluation_completed_examples: 6, live_evaluation_total_examples: 9,
  live_evaluation_succeeded_examples: 6, live_evaluation_failed_examples: 0, live_evaluation_candidate_count: 1,
  execution_evidence_status: "partial_live",
  candidate_artifacts_count: 0, candidate_evaluations_count: 0, candidate_work_items: 6,
  fit_passes_completed: 0, fit_passes_per_candidate: 0, iterative_epoch_training: false,
};
diagnosticSandbox.renderExecutionDiagnostics({execution_diagnostics: partialDiagnostics});
assert.equal(diagnosticNodes["#execution-diagnostics-summary"].textContent, "部分实时证据 · 反馈 6 / 9 · 候选工作量 6");
assert.ok(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("实时完成 6 / 9 个反馈目标样本"));
assert.ok(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("部分实时证据 · 成功 6 · 失败 0"));
assert.ok(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("等待候选训练产物"));
assert.ok(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("等待拟合证据"));
assert.equal(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("已评测 0"), false);

diagnosticSandbox.renderExecutionDiagnostics({execution_diagnostics: Object.assign({}, partialDiagnostics, {execution_evidence_status: "retained_partial"})});
assert.equal(diagnosticNodes["#execution-diagnostics-summary"].textContent, "已保留部分证据 · 反馈 6 / 9 · 候选工作量 6");
assert.ok(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("已保留 6 / 9 个反馈目标样本"));
assert.equal(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("实时完成"), false);

diagnosticSandbox.renderExecutionDiagnostics({execution_diagnostics: Object.assign({}, partialDiagnostics, {execution_evidence_status: "aborted_partial"})});
assert.equal(diagnosticNodes["#execution-diagnostics-summary"].textContent, "已中止，保留部分证据 · 反馈 6 / 9 · 候选工作量 6");
assert.ok(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("中止前完成 6 / 9 个反馈目标样本"));
assert.equal(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("实时反馈目标"), false);

diagnosticSandbox.renderExecutionDiagnostics({execution_diagnostics: Object.assign({}, partialDiagnostics, {execution_evidence_status: "mixed_partial"})});
assert.equal(diagnosticNodes["#execution-diagnostics-summary"].textContent, "部分执行证据 · 反馈 6 / 9 · 候选工作量 6");
assert.ok(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("累计完成 6 / 9 个反馈目标样本"));
assert.equal(diagnosticNodes["#execution-diagnostics-grid"].innerHTML.includes("实时完成"), false);

for (const script of scripts) {
  assert.ok(html.includes(`src="${script}"`), `missing script include: ${script}`);
}

assert.equal(manifest.plugin_id, "ecologyrsi.evolution");
assert.match(html, /id="show-cancelled-empty-runs"/);
assert.match(html, /已取消空任务/);
assert.match(html, /id="show-archived-runs"/);
assert.match(html, /id="archive-button"/);
assert.match(html, /id="delete-button"/);
assert.equal(manifest.display_name, "生态模型进化工作台");
assert.equal(manifest.version, "0.3.15");
assert.equal(manifest.entrypoint.file, "index.html");
assert.equal(manifest.entrypoint.route, "/plugins/ecology/evolution/");
assert.equal(manifest.development_only, false);
assert.equal(manifest.release_stage, "delivery-candidate");
assert.equal(manifest.integrity.status, "delivery-candidate-unsigned");
assert.equal(manifest.integrity.signature, null);
assert.deepEqual(manifest.api.supported_bases, ["/api", "/api/v1", "/api/ecology-evolution", "/api/ecology-evolution/v1"]);
assert.equal(manifest.dsh_compatibility.recommended_proxy_base, "/api/ecology-evolution");
assert.equal(manifest.dsh_compatibility.backend_mode, "independent-service");
assert.equal(manifest.api.dataset_samples, "/datasets/{dataset_id}/samples");
assert.equal(manifest.api.samples, "/runs/{run_id}/samples?candidate_id={candidate_id}&offset={offset}&limit={limit}");
assert.equal(manifest.api.control, "/runs/{run_id}/control");
assert.equal(manifest.api.archive, "/runs/{run_id}/archive");
assert.equal(manifest.api.restore, "/runs/{run_id}/restore");
assert.equal(manifest.api.delete, "/runs/{run_id}");
assert.equal(manifest.api.expert_consultation_answer, "/runs/{run_id}/expert-consultations/{consultation_id}/answer");
assert.ok(manifest.events.includes("expert_consultation.*"));
assert.equal(Object.hasOwn(manifest.api, "model_verification"), false);
assert.equal(manifest.security.capability_token_storage, "memory-only");
assert.equal(manifest.security.token_scope, "service-process");
assert.equal(manifest.security.formal_task_run_session_scope, false);
assert.match(dshHostClient, /api\.llm\.models/);
assert.match(dshHostClient, /\/api\/llm\.models/);
assert.match(dshHostClient, /flattenHostModelDirectory/);
assert.match(dshHostClient, /authentication_state: "dsh_authenticated"/);
assert.doesNotMatch(dshHostClient, /models: \[\],/);
for (const capability of ["evolution.catalog.read", "training.data.read", "evaluation.samples.read", "run.archive", "run.delete", "intervention.write"]) {
  assert.ok(manifest.permissions.includes(capability), `missing capability: ${capability}`);
}
assert.match(dshHostClient, /evaluation\.samples\.read/);
assert.equal(manifest.permissions.includes("model.connection.verify"), false);
for (const unsupported of ["dsh.session.create", "dsh.session.resume", "evidence.query", "proposal.submit"]) {
  assert.equal(manifest.permissions.includes(unsupported), false, `unimplemented capability declared: ${unsupported}`);
}
for (const capability of ["hidden.read", "final.read", "release.write"]) {
  assert.ok(manifest.denied_capabilities.includes(capability), `missing denied capability: ${capability}`);
}

for (const workspace of ["settings", "parameters", "training", "process", "candidates", "collaboration"]) {
  assert.match(html, new RegExp(`data-workspace=\\"${workspace}\\"`));
  assert.match(html, new RegExp(`data-panel=\\"${workspace}\\"`));
}
for (const text of [
  "运行设置", "参数设计", "训练数据", "进化过程", "候选评测", "人工协作与治理",
  "训练数据集", "研究领域（自动推导）", "策略模型（API）", "独立评审模型（API）", "模型驱动的受限进化闭环", "训练拟合分区", "外部治理状态",
  "待专家答复", "已答复", "专家主动意见", "非阻塞咨询不会暂停进化"
]) {
  assert.ok(html.includes(text), `missing Chinese interface text: ${text}`);
}
assert.match(html, /<input id="slot" name="slot" type="hidden" value="bounded_predictor">/);
assert.doesNotMatch(html, /<select[^>]+name="slot"/);
assert.match(html, /name="strategy_model_id"/);
assert.match(html, /name="review_model_id"/);
assert.match(html, /name="autonomous_mode" type="hidden" value="true"/);
assert.doesNotMatch(html, /advanced-settings/);
assert.match(html, /id="workspace-parameters"[\s\S]*name="candidates_per_generation"/);
assert.match(html, /id="workspace-parameters"[\s\S]*name="samples_per_update"/);
assert.match(html, /id="workspace-parameters"[\s\S]*name="sample_agent_batch_size"/);
assert.match(html, /id="workspace-parameters"[\s\S]*name="sample_concurrency"/);
assert.match(html, /id="workspace-parameters"[\s\S]*name="max_candidates"/);
assert.doesNotMatch(html, /id="workspace-parameters"[\s\S]*name="token_limit"/);
assert.match(html, /id="workspace-parameters"[\s\S]*name="fixed_seed"/);
assert.match(html, /id="workspace-parameters"[\s\S]*name="knowledge_online_enabled"/);
assert.doesNotMatch(html, /id="workspace-settings"[\s\S]*id="workspace-parameters"[\s\S]*advanced-settings/);
assert.ok(html.includes("DSH 上下文：等待 Session 计量"));
assert.doesNotMatch(html, /name="token_limit"/);
assert.ok(html.includes("全量 training_fit"));
assert.ok(html.includes("每轮固定 500 个样本"));
assert.match(html, /<label><span>训练数据集<\/span><select id="dataset-id" name="dataset_id" required>/);
assert.match(html, /<label hidden><span>研究领域（自动推导）/);
assert.match(html, /<label hidden><span>预测模型/);
assert.match(html, /<label hidden><span>进化策略/);
assert.match(html, /<label hidden><span>独立评测器/);
assert.match(app, /prediction_models: normalizeList\(value\.prediction_models\)/);
assert.match(app, /dsh_models: dshModels/);
assert.match(app, /authenticated_models: authenticatedModels/);
assert.match(app, /policy_models: sharedModels\.filter\(function \(item\) \{ return modelSupportsRole\(item, "propose"\); \}\)/);
assert.match(app, /judge_models: sharedModels\.filter\(function \(item\) \{ return modelSupportsRole\(item, "judge"\); \}\)/);
assert.match(app, /var policyItems = autonomousModelItems\(\);/);
assert.match(app, /var judgeItems = autonomousModelItems\(\);/);
assert.match(app, /populateSelect\("#policy-model-id", policyItems,[\s\S]*"propose"\)/);
assert.match(app, /populateSelect\("#judge-model-id", judgeItems,[\s\S]*"judge"\)/);
assert.match(app, /modelSupportsRole\(item, requiredRole\)/);
assert.match(app, /var items = state\.catalog\.dsh_models_explicit === true/);
assert.match(app, /function sharedDshModelItems\(\)/);
assert.match(app, /state\.hostContextReceived && state\.hostContext && Array\.isArray\(state\.hostContext\.models\)/);
assert.match(app, /leftIdentity\.provider && rightIdentity\.provider && leftIdentity\.provider !== rightIdentity\.provider/);
assert.match(app, /return remote\.length \|\| !state\.usingDemo \? remote : items/);
assert.match(app, /autonomous_mode: payload\.autonomous_mode === true/);
assert.match(app, /dataset_id: payload\.dataset_id \|\| payload\.datasetId/);
assert.match(app, /总上限/);
assert.match(app, /\["#max-generations", "#candidates-per-generation"\]/);
assert.match(app, /\$\("#max-candidates"\)\.addEventListener\("input"/);
assert.match(app, /function runnableDatasetItems\(\)/);
assert.match(app, /populateSelect\("#dataset-id", runnableDatasetItems\(\)/);
for (const id of [
  "start-form", "training-table", "trajectory-chart", "candidate-table",
  "candidate-detail", "intervention-form", "intervention-history", "gate-summary",
    "pending-consultations", "answered-consultations", "pending-consultation-count", "answered-consultation-count",
    "training-assets-table", "round-stage-list", "dataset-partition", "show-all-fields",
    "field-visibility-control", "target-candidate-field", "parameter-overrides-field",
    "model-connection-list", "dataset-context-note", "source-integrity", "dataset-error", "retry-dataset-button",
    "process-summary", "autonomy-progress", "autonomy-progress-status", "prediction-model-id", "prediction-model-help", "toggle-events-button",
    "knowledge-online-enabled", "samples-per-update", "sample-agent-batch-size", "sample-concurrency", "parameter-summary", "agent-update-scope", "execution-monitor-status", "execution-progress-track", "execution-progress-fill", "execution-diagnostics-summary", "execution-diagnostics-grid", "show-archived-runs", "archived-count", "archive-button", "delete-button",
    "execution-stage-strip", "active-candidate-summary", "sample-inference-list", "implementation-summary"
]) {
  assert.match(html, new RegExp(`id=\\"${id}\\"`), `missing interface element: ${id}`);
}

assert.match(app, /dataRequestTimeout = 30000/);
assert.match(app, /evolutionCommandTimeout = 120000/);
assert.match(app, /request\("\/catalog", \{ timeout: dataRequestTimeout \}\)/);
assert.match(app, /state\.refreshing/);
assert.match(app, /finally\(function \(\) \{ state\.refreshing = false; renderAll\(\); \}\)/);
assert.match(app, /refreshAll\(\{ refreshDataset: true \}\)/);
assert.match(app, /rightTime - leftTime/);
assert.match(app, /\/datasets\/" \+ encodeURIComponent\(datasetId\)/);
assert.match(html, /id="episode-id"/);
assert.match(app, /new URLSearchParams\(\{ partition: partition, episode_id: episodeId/);
assert.match(app, /partition === "training_feedback"/);
assert.match(app, /episode_id: episodeId/);
for (const contractField of ["descriptor", "readiness", "profile", "features", "row.values"]) {
  assert.ok(app.includes(contractField), `missing dataset contract support: ${contractField}`);
}
assert.match(app, /request\("\/runs"/);
assert.match(html, /id="candidates-per-generation"[^>]*value="4"/);
assert.match(html, /id="samples-per-update"[^>]*value="500"/);
assert.match(html, /id="sample-agent-batch-size"[^>]*value="64"/);
assert.match(html, /id="sample-concurrency"[^>]*value="2"/);
assert.match(html, /id="max-candidates"[^>]*value="20"/);
assert.doesNotMatch(html, /id="token-limit"/);
assert.ok(html.includes("样本先按因果预测起点组成 origin wave"));
assert.ok(html.includes("实际请求数以运行进度为准"));
assert.doesNotMatch(app, /wavesPerCandidate|每候选约/);
for (const field of ["rounds", "candidates_per_generation", "samples_per_update", "sample_agent_batch_size", "sample_concurrency", "max_candidates", "fixed_seed", "knowledge_online_enabled"]) {
  assert.match(html, new RegExp(`name="${field}"[^>]*form="start-form"|form="start-form"[^>]*name="${field}"`));
}
assert.match(app, /function syncCandidateBudget\(options\)/);
assert.match(app, /var requestedSamplesPerUpdate = normalizedSamplesPerUpdate\(payload\.samples_per_update\)/);
assert.match(app, /requestedSamplesPerUpdate < minimumSamplesPerUpdate/);
assert.match(app, /samples_per_update: requestedSamplesPerUpdate/);
assert.match(app, /sample_agent_batch_size: normalizedSampleAgentBatchSize\(payload\.sample_agent_batch_size\)/);
assert.match(app, /sample_concurrency: normalizedSampleConcurrency\(payload\.sample_concurrency\)/);
assert.ok(app.includes("候选总预算不足"));
assert.match(app, /auto_advance: payload\.auto_advance === 0 \? 0 : continuousAutoAdvance \? true : 1/);
assert.ok(app.includes("已排队，后台将自动执行全部轮次"));
assert.ok(app.includes("创建新的进化运行"));
assert.ok(app.includes("后台排队中"));
assert.ok(app.includes("后台自动推进"));
assert.match(app, /function ensureAutoAdvanceForRun\(runId\)/);
assert.match(app, /function serverAutoProgressEnabled\(run\)/);
assert.match(app, /state\.autoAdvanceRunId/);
assert.match(app, /state\.autoAdvanceRetry/);
assert.match(app, /\/interventions"/);
assert.match(app, /\/expert-consultations\/" \+ encodeURIComponent\(id\) \+ "\/answer"/);
assert.match(app, /expertConsultationDrafts/);
assert.match(app, /迟到专家答复已归档/);
assert.match(app, /不会改写历史候选，将从后续轮次开始使用/);
assert.match(app, /function hasCapability\(name\)/);
assert.match(app, /eventPollTimer = window\.setInterval/);
assert.match(app, /window\.clearInterval\(eventPollTimer\)/);
assert.match(app, /refreshProgressForRun\(runId\)/);
assert.match(catalogSource, /stopRunMonitor\(state\.runMonitorRunId\)/);
assert.match(dataSource, /state\.refreshing \|\| state\.busy/);
assert.match(app, /\}, 60000\);/);
for (const diagnosticField of [
  "training_partition_rows", "training_used_examples", "training_skipped_examples",
  "evaluation_partition_rows", "evaluation_eligible_examples", "evaluation_used_examples",
  "evaluation_skipped_examples", "candidate_artifacts_count", "candidate_evaluations_count",
  "live_evaluation_completed_examples", "live_evaluation_total_examples", "execution_evidence_status",
  "candidate_work_items", "fit_passes_completed", "fit_passes_per_candidate", "iterative_epoch_training", "proposal_sources", "source_counts",
  "remote_strategy_calls", "remote_strategy_successes", "remote_strategy_status",
]) {
  assert.ok(app.includes(diagnosticField), `missing execution diagnostic field: ${diagnosticField}`);
}
assert.ok(app.includes("无神经网络 epoch"));
for (const productionDiagnosticValue of ["host_reserved_seed", "host_strategy", "legacy_unknown", "partial_host_fallback", "not_called", "incomplete"]) {
  assert.ok(app.includes(productionDiagnosticValue), `missing production diagnostic mapping: ${productionDiagnosticValue}`);
}
for (const capability of ["evolution.run.create", "evolution.run.advance", "evolution.projection.read", "training.data.read", "run.control", "run.archive", "run.delete", "intervention.write"]) {
  assert.ok(app.includes(`hasCapability("${capability}")`), `missing capability gate: ${capability}`);
}
assert.match(app, /unavailable_datasets: normalizeList\(value\.unavailable_datasets\)/);
assert.match(app, /artifactForCandidate\(candidate\.id\)/);
assert.match(app, /candidate\.metrics\.prediction_preview/);
assert.match(app, /var selectedTarget = targetSelect\.value/);
assert.match(app, /item\.parameter_overrides/);
assert.match(app, /item\.applied_proposal_id/);
for (const field of ["strategy_model_id", "review_model_id", "autonomous_mode", "rounds"]) {
  assert.ok(app.includes(field), `missing autonomous create field: ${field}`);
}
assert.match(app, /strategy_model_id: form\.get\("strategy_model_id"\)/);
assert.match(app, /review_model_id: form\.get\("review_model_id"\)/);
assert.match(app, /function alignPredictionBinding\(\)/);
assert.match(app, /function alignEvaluatorBinding\(\)/);
assert.match(app, /prediction_model_ids/);
assert.match(app, /horizons_hours/);
assert.match(app, /source_integrity/);
for (const field of ["expected_size_bytes", "expected_md5", "size_matches", "md5_matches"]) {
  assert.ok(app.includes(field), `missing source integrity field: ${field}`);
}
for (const selector of ["#prediction-model-id", "#evaluator-id"]) {
  assert.ok(app.includes(`$("${selector}").addEventListener("change"`), `missing compatibility binding: ${selector}`);
}
for (const field of ["credential_configured", "modelCredentialReady", "modelConnectionStateText"]) {
  assert.ok(app.includes(field), `missing model execution contract: ${field}`);
}
assert.doesNotMatch(html, /id="verify-models-button"/);
assert.doesNotMatch(app, /function verifySelectedModels\(\)/);
assert.doesNotMatch(app, /runModelVerificationIssues/);
assert.doesNotMatch(app, /hasCapability\("model\.connection\.verify"\)/);
assert.ok(app.includes("策略模型 API 已安全配置"));
assert.ok(app.includes("独立评审模型 API 已安全配置"));
assert.match(app, /function renderModelConnections\(\)/);
assert.match(app, /function modelConnectionErrorText\(item\)/);
assert.ok(app.includes("模型响应不符合约定格式"));
assert.match(css, /\.model-connection-list/);
assert.match(css, /\.model-connection-error/);
assert.match(css, /@media \(min-width: 761px\) and \(max-width: 1320px\)/);
for (const field of ["selection_scope", "formal_validation_status"]) {
  assert.ok(app.includes(field), `missing scientific-boundary field: ${field}`);
}
for (const field of ["mean_target_mae_unscaled", "mean_target_rmse_unscaled", "per_target_no_regression", "raw_units_comparable_across_targets"]) {
  assert.ok(app.includes(field), `missing evaluation metric localization: ${field}`);
}
for (const field of ["training_mae", "training_rmse"]) {
  assert.ok(app.includes(`${field}: "训练`), `missing training metric localization: ${field}`);
}
assert.match(app, /training_feedback: "训练反馈分区"/);
assert.match(app, /admission\.tier \|\| admission\.bucket/);
assert.match(app, /evaluation\.judge && typeof evaluation\.judge === "object"/);
assert.match(app, /artifact\.artifact_digest \|\| output\.artifact_digest/);
assert.match(app, /candidate: "候选生成"/);
assert.match(app, /round\.decision/);
assert.match(app, /function renderRoundKnowledge\(round\)/);
assert.match(app, /knowledge\.retrieved/);
assert.match(app, /knowledge_online_enabled: payload\.knowledge_online_enabled/);
assert.match(app, /function renderExecutionMonitor\(run\)/);
assert.match(app, /execution_progress/);
assert.match(app, /inference_trace/);
assert.match(app, /progress_percent/);
assert.match(app, /function executionPredictionTrace\(candidate, run\)/);
assert.match(app, /function renderCandidateExecutionEvidence\(candidate, run\)/);
assert.match(app, /candidate\.algorithm_execution/);
assert.match(app, /metrics\.sample_execution/);
assert.ok(app.includes("研究证据 → 算法规范 → 编译 → 调试 → 真实样本反馈"));
assert.ok(app.includes("公开结构化动作，不展示私密推理"));
assert.equal(app.includes("查看模型研究计划"), false);
assert.ok(html.includes("样本推理与预测") || app.includes("样本推理与预测"));
assert.ok(html.includes("具体实现方案") || app.includes("具体实现方案"));
assert.match(css, /\.execution-progress-track/);
assert.match(css, /\.sample-inference-row/);
assert.match(css, /\.execution-monitor-grid/);
assert.match(css, /\.candidate-evidence-flow/);
assert.match(css, /\.candidate-evidence-preview-grid/);
assert.match(css, /\.candidate-algorithm-grid/);
assert.match(html, /提案 → 能力编译 → 训练 → 科学评测 → 独立评审 → 轮末决策/);
assert.match(css, /\.round-stage-track[^}]*repeat\(6,/s);
assert.match(css, /\.knowledge-source-row/);
assert.match(app, /function promotionDecisionTitle\(event\)/);
assert.ok(app.includes("候选方案已在训练反馈搜索中保留（正式验证未开展）"));
assert.match(app, /return "候选方案未保留"/);
assert.match(app, /eventTitle\(event, run\)/);
assert.match(app, /候选编号：/);
assert.match(app, /得分 " \+ formatNumber\(payload\.score\)/);
assert.match(app, /eventTone\(event, run\)/);
assert.match(app, /best_observed_score/);
assert.ok(app.includes("原始最高观测（跨窗口不可直接比较）"));
assert.ok(app.includes("实际晋升序列"));
assert.match(app, /point\.incumbent_score != null \? point\.incumbent_score : point\.best_score/);
assert.doesNotMatch(app, /当前最高分/);
assert.doesNotMatch(app, /已观测最高得分/);
assert.ok(app.includes("已完成预设进化规模"));
assert.ok(app.includes("尚无候选通过全部评测门控；正式验证未开展"));
assert.ok(!app.includes("预算已耗尽，未产生训练反馈搜索保留候选"));
assert.ok(app.includes("训练反馈搜索保留候选："));
assert.ok(app.includes("正式验证未开展"));
assert.match(app, /run\.best_candidate_id \|\| run\.best_observed_candidate_id/);
assert.doesNotMatch(app, /delta === 0 \? "当前最佳"/);
for (const text of ["搜索保留", "训练反馈检查", "当前候选选择仅依据迭代训练反馈，不代表正式验证通过"]) {
  assert.ok(html.includes(text) || app.includes(text), `missing scientific-boundary copy: ${text}`);
}
assert.equal(app.includes('accepted: "已晋级"'), false);
assert.equal(app.includes('approved: "已批准"'), false);
assert.match(app, /retained: "训练反馈搜索保留"/);
assert.match(app, /return "本地内置"/);
assert.match(app, /formatObservationTime/);
assert.match(app, /environment: "环境观测"/);
assert.match(app, /forcing: "外部驱动", state: "状态变量"/);
assert.doesNotMatch(app, /"model\.connection\.verify":/);
assert.match(app, /function isBlank\(value\)/);
assert.match(app, /function displayText\(value, fallback\)/);
assert.match(app, /function humanizeTechnicalText\(value\)/);
assert.match(app, /humanizeTechnicalText\(round\.next_generation_focus/);
assert.match(app, /var fieldGroupOrder = \["target", "control", "environment", "crop", "resource", "other"\]/);
assert.match(app, /function coreSchema\(schema\)/);
for (const group of ["评测目标", "控制与管理", "环境与根区", "作物与产量", "资源投入与消耗"]) {
  assert.ok(app.includes(group), `missing training field group: ${group}`);
}
assert.match(app, /state\.showAllFields = event\.target\.checked === true/);
assert.match(app, /target_candidate_id: kind === "parent_selection"/);
assert.match(app, /\$\("#target-candidate-field"\)\.hidden = !parentSelection/);
assert.match(app, /\$\("#parameter-overrides"\)\.required = parameterOverride/);
assert.match(app, /!Object\.keys\(overrides\)\.length/);
assert.match(app, /function activeRunDatasetContext\(\)/);
assert.match(app, /return activeRunDatasetContext\(\) \|\| selectedDatasetContext\(\)/);
assert.match(app, /state\.datasetError = errorMessage\(error\)/);
assert.match(app, /\$\("#retry-dataset-button"\)\.addEventListener/);
assert.match(app, /state\.showAllEvents = !state\.showAllEvents/);
assert.match(app, /state\.showAllEvents \? state\.events : state\.events\.slice\(0, 12\)/);
assert.ok(app.includes("仅记录（未执行）"));
assert.ok(app.includes('recordedOnlyCount ? recordedOnlyCount + " 条仅记录"'));
assert.ok(app.includes('applied: "已应用"'));
assert.ok(app.includes('enforced: "已强制执行"'));
assert.ok(app.includes('? "仅记录（未执行）" : "等待下一轮"'));
assert.match(app, /item && item\.enforced === true/);
assert.match(app, /item && item\.applied === true/);
assert.match(app, /item && item\.recorded === true/);
assert.doesNotMatch(app, /return "事件已写入追加式记录。"/);
assert.match(app, /event\.data\.type !== "dsh\.context"/);
assert.match(app, /isTrustedParentOrigin\(event\.origin\)/);
assert.match(app, /supportedApiPaths\.indexOf\(path\) < 0/);
assert.match(app, /type: "plugin\.ready"/);
assert.match(app, /context_protocol: contextProtocol/);
assert.match(app, /"\/api\/ecology-evolution"/);
assert.doesNotMatch(app, /plugin_id: "ecologyrsi\.evolution", version: "0\.2\.0" \}, "\*"/);
assert.match(app, /headers\.Authorization = "Bearer " \+ capabilityToken/);
assert.match(app, /serviceCapabilities\.filter/);
assert.match(app, /preferredModelId\("propose"/);
assert.match(app, /createdBy\.readOnly = true/);
assert.doesNotMatch(app, /localStorage|sessionStorage/);
assert.match(app, /query\.get\("demo"\) === "1"/);
assert.doesNotMatch(app, /window\.location\.protocol === "file:"/);
assert.match(app, /viewBox=\\"0 0 " \+ width \+ " " \+ height/);
assert.match(app, /preserveAspectRatio=\\"xMidYMid meet\\"/);
assert.match(css, /\.trajectory-chart[^}]*aspect-ratio:\s*4\s*\/\s*1/s);
assert.match(css, /\.trajectory-chart svg[^}]*min-width:\s*0/s);
assert.doesNotMatch(css, /\.trajectory-chart svg[^}]*min-width:\s*720px/s);
assert.match(css, /\.source-integrity[^}]*grid-template-columns/s);
assert.match(css, /\.source-integrity:empty[^}]*display:\s*none/s);
assert.match(css, /\.source-integrity-summary:only-child[^}]*grid-column:\s*1\s*\/\s*-1/s);
for (const statusClass of ["is-verified", "is-missing", "is-mismatch", "is-not-checked", "is-not-applicable"]) {
  assert.ok(css.includes(`.source-integrity-item.${statusClass}`), `missing source integrity state style: ${statusClass}`);
}
assert.match(css, /\.process-summary[^}]*repeat\(6,/s);
assert.match(css, /\.stats-strip\s*>\s*\.empty-state[^}]*grid-column:\s*1\s*\/\s*-1/s);
assert.match(css, /\.artifact-model-summary/);
assert.match(css, /\.artifact-model-grid/);
assert.match(css, /\.artifact-model-row\s*>\s*div:first-child/);
for (const selector of [".app-shell", ".workspace-layout", ".workspace-panel", ".panel"]) {
  assert.match(css, new RegExp(`\\${selector}[^}]*max-width:\\s*(?:100%|1600px)`), `missing width containment: ${selector}`);
}
assert.match(app, /window\.addEventListener\("resize", scheduleTrajectoryRender\)/);
for (const text of ["内置有界参数生成器", "内置规则独立评审", "摄氏度（°C）", "累计千克\/平方米"]) {
  assert.ok(app.includes(text), `missing Chinese model or unit label: ${text}`);
}
assert.ok(html.includes("其他已登记数据集"));
assert.ok(html.includes("策略／预测模型／评审") || html.includes("策略模型／自主预测模型／评审"));
assert.ok(html.includes("训练轨迹／候选方案"));
assert.ok(html.includes("每条资产是一条完整训练轨迹"));
assert.ok(html.includes("可交付候选版（未签名）"));
assert.ok(html.includes("正式监督微调（SFT）"));
assert.ok(html.includes("追加式执行记录"));
assert.ok(app.includes("五阶段完整训练记录"));
assert.ok(app.includes("完整训练记录"));
assert.match(app, /function renderTrainingTrajectory\(asset\)/);
assert.match(app, /function trainingTraceSource\(asset\)/);
assert.match(app, /prediction_records/);
assert.ok(html.includes("相对持续性基线的绝对误差改善，正值更好"));
assert.ok(app.includes("评分惩罚值"));
for (const text of ["样本输入", "智能体交互", "反馈评测", "优化更新", "最终预测与观测"]) {
  assert.ok(app.includes(text), `missing training trajectory label: ${text}`);
}
for (const selector of [".training-trajectory-section", ".training-trace-list", ".training-trace-prediction-table", ".training-trace-step"]) {
  assert.match(css, new RegExp(`\\${selector}`), `missing training trajectory style: ${selector}`);
}
assert.ok(app.includes("运行已结束"));
assert.match(app, /function recordedBooleanText\(value, trueText, falseText\)/);
assert.doesNotMatch(app, /查看五阶段 episode/);
assert.doesNotMatch(app, /\["完整 episode"/);

for (const breakpoint of ["1100", "760", "430"]) {
  assert.match(css, new RegExp(`@media \\(max-width: ${breakpoint}px\\)`));
}
assert.match(css, /@media \(max-width: 760px\)[\s\S]*?#workspace-candidates \.candidate-evidence-flow \{ grid-template-columns: minmax\(0, 1fr\); \}/);
assert.match(css, /@media \(max-width: 430px\)[\s\S]*?#workspace-candidates \.candidate-algorithm-grid \{ grid-template-columns: minmax\(0, 1fr\); \}/);
assert.match(css, /#workspace-candidates \.candidate-attempt-list li p \{ grid-column: 1 \/ -1; \}/);
assert.match(css, /\.candidate-evidence-stage\.is-skipped/);
const spacingValues = [...css.matchAll(/letter-spacing:\s*([^;]+);/g)].map((match) => match[1].trim());
assert.ok(spacingValues.every((value) => value === "0" || value === "0px"));
const radii = [...css.matchAll(/border-radius:\s*([0-9]+)px/g)].map((match) => Number(match[1]));
assert.ok(radii.every((value) => value <= 8), "rectangular controls must use radius <= 8px");

for (const text of ["CONTROL ROOM", "NEW RUN", "EVENT LEDGER", "CANDIDATE ARCHIVE", "PROMOTION GATE", "暂无 Run", "启动 Run"]) {
  assert.equal(html.includes(text) || app.includes(text), false, `legacy English interface text remains: ${text}`);
}

console.log("ecology_evolution plugin smoke test: ok");
