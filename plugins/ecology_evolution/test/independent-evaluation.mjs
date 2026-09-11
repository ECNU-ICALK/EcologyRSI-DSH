import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const nodes = Object.fromEntries(["#independent-validation-start", "#independent-final-test-start", "#independent-evaluation-status"].map(id => [id, {}]));
let response;
const sandbox = {
  state: {activeRun: {id: "run:a"}, workspace: "evaluation", usingDemo: false},
  $: id => nodes[id], clearTimeout() {}, setTimeout() {},
  request: () => response,
  targetLabels: {air_temperature: "气温"},
  formatNumber: (value, digits) => Number(value).toFixed(digits || 0),
  escapeHTML: value => String(value).replaceAll("<", "&lt;").replaceAll(">", "&gt;")
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(new URL("../assets/js/commands.js", import.meta.url), "utf8"), sandbox);
response = Promise.resolve({stages: [
  {stage: "validation", label: "独立验证", status: "completed", outcome: "passed", available: false,
   assessment: {origin_count: 12, replicas: [{replica: 1, score: 0.2, targets: [{target: "air_temperature", unit: "degC", horizon_hours: 24,
     n: 12, mae: 0.4, rmse: 0.5, bias: -0.1, baseline_rmse: 0.7, skill_score: 0.2, sample_execution_coverage: 1}]}]}},
  {stage: "final_test", label: "最终测试", status: "not_started", available: true}
]});
await sandbox.loadIndependentEvaluation();
assert.equal(nodes["#independent-validation-start"].disabled, true);
assert.equal(nodes["#independent-final-test-start"].disabled, false);
assert.match(nodes["#independent-evaluation-status"].innerHTML, /气温 \(degC\)/);
assert.match(nodes["#independent-evaluation-status"].innerHTML, /100.0%/);
let late;
response = new Promise(resolve => { late = resolve; });
const loading = sandbox.loadIndependentEvaluation();
sandbox.state.activeRun = null;
await sandbox.loadIndependentEvaluation();
late({stages: []});
await loading;
assert.match(nodes["#independent-evaluation-status"].textContent, /先完成/);
assert.equal(nodes["#independent-final-test-start"].disabled, true);
console.log("independent evaluation UI: ok");
