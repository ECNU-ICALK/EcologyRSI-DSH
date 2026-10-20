import assert from "node:assert/strict";
import test from "node:test";
import {
  displaySourceDigest, projectSampleContext, expandSampleContext,
  projectPredictionToolResult, expandPredictionToolResult, modelContextView,
} from "../lib/runtime/display-projection.js";
import { registerRoleTools } from "../lib/tools/roles.js";
import { sampleContextFixture, predictionResultFixture } from "./fixtures/display_projection.mjs";

function freeze(value) {
  if (value && typeof value === "object") { Object.freeze(value); for (const child of Object.values(value)) freeze(child); }
  return value;
}
const size = value => Buffer.byteLength(JSON.stringify(value));

test("sample display shares common identity only and exactly restores the immutable source", () => {
  const original = freeze(sampleContextFixture()), before = JSON.stringify(original);
  const view = projectSampleContext(original);
  assert.ok(view._display_projection);
  assert.equal(view._display_projection.source_digest, displaySourceDigest(original));
  assert.ok(size(view) < size(original));
  assert.ok(!Object.hasOwn(view.samples[0], "candidate_id"));
  assert.deepEqual(view.samples.map(row => row.sample_id), original.samples.map(row => row.sample_id));
  assert.deepEqual(view.available_tools, original.available_tools);
  assert.deepEqual(expandSampleContext(view), original);
  assert.equal(JSON.stringify(original), before);
  assert.equal(modelContextView("sample.critic", original), original);
  assert.equal(modelContextView("sample.critic", original), original);
  assert.deepEqual(modelContextView("sample.plan", original), view);
});

test("sample display keeps heterogeneous and absent fields local and small contexts unchanged", () => {
  const original = sampleContextFixture();
  original.samples[1].attempt = 2; delete original.samples[2].partition;
  const view = projectSampleContext(original);
  assert.equal(view.samples[1].attempt, 2);
  assert.equal(view.samples[0].partition, original.samples[0].partition);
  assert.deepEqual(expandSampleContext(view), original);
  for (const source of [null, {}, { samples: [] }, { samples: [{ sample_id: "one" }] },
    { samples: [{ sample_id: "one", attempt: 1 }, { sample_id: "two", attempt: 1 }] }]) {
    assert.equal(projectSampleContext(source), source);
  }
});

test("prediction display shares equal parameter objects without changing source or receipts", () => {
  const original = freeze(predictionResultFixture()), before = JSON.stringify(original);
  const view = projectPredictionToolResult(original);
  assert.ok(view._display_projection);
  assert.ok(size(view) < size(original));
  assert.equal(view.output_digest, original.output_digest);
  assert.equal(view.remaining_calls, original.remaining_calls);
  assert.equal(view.wave_digest, original.wave_digest);
  assert.deepEqual(expandPredictionToolResult(view), original);
  assert.equal(JSON.stringify(original), before);
  view._display_projection.shared_parameters.p1.feature_recipe.window = 99;
  assert.equal(original.outputs[0].metadata.parameters.feature_recipe.window, 12);
});

test("prediction display does not merge different parameters or rewrite failed and small results", () => {
  const original = predictionResultFixture();
  original.outputs[8].metadata.parameters.ridge_alpha = 2;
  const view = projectPredictionToolResult(original);
  assert.ok(!Object.hasOwn(view._display_projection.parameter_refs, "8"));
  assert.equal(view.outputs[8].metadata.parameters.ridge_alpha, 2);
  assert.deepEqual(expandPredictionToolResult(view), original);
  const allDifferent = predictionResultFixture();
  allDifferent.outputs.forEach((row, index) => { row.metadata.parameters.ridge_alpha = index; });
  assert.equal(projectPredictionToolResult(allDifferent), allDifferent);
  const noGain = predictionResultFixture();
  noGain.outputs = noGain.outputs.slice(0, 2);
  noGain.outputs.forEach(row => { row.metadata.parameters = { detail: "x".repeat(130) }; });
  assert.equal(projectPredictionToolResult(noGain), noGain);
  for (const value of [null, {}, { accepted: false, error_code: "denied" },
    { ...original, status: "failed", error_detail: "capacity", outputs: original.outputs }]) {
    assert.equal(projectPredictionToolResult(value), value);
  }
});

test("display decoders reject unknown versions, missing references, overrides and digest drift", () => {
  const tool = projectPredictionToolResult(predictionResultFixture());
  const sample = projectSampleContext(sampleContextFixture());
  const changed = (source, mutate) => { const value = structuredClone(source); mutate(value); return value; };
  for (const mutate of [
    value => { value._display_projection.schema_version = "unknown"; },
    value => { value._display_projection.parameter_refs["0"] = "missing"; },
    value => { value._display_projection.parameter_refs["99"] = "p1"; },
    value => { value._display_projection.parameter_refs["01"] = "p1"; },
    value => { value.outputs[0].metadata.parameters = {}; },
    value => { value._display_projection.shared_parameters.p2 = {}; },
    value => { value._display_projection.shared_parameters.p1.ridge_alpha = 99; },
    value => { value.outputs[0].predicted += 1; },
    value => { value.output_digest = "0".repeat(64); },
    value => { value._display_projection.source_digest = "bad"; },
  ]) assert.throws(() => expandPredictionToolResult(changed(tool, mutate)), /invalid lossless display projection/);
  for (const mutate of [
    value => { value.samples[0].candidate_id = "wrong"; },
    value => { value.samples[0].sample_id = "wrong"; },
    value => { value._display_projection.shared_sample_fields.sample_id = "wrong"; },
    value => { value._display_projection.shared_sample_fields.dataset_digest = "wrong"; },
    value => { value._display_projection.kind = "prediction-tool-parameters"; },
    value => { value._display_projection.extra = true; },
  ]) assert.throws(() => expandSampleContext(changed(sample, mutate)), /invalid lossless display projection/);
  assert.throws(() => projectSampleContext(sample), /invalid lossless display projection/);
  assert.throws(() => projectPredictionToolResult(tool), /invalid lossless display projection/);
});

test("registered tool executes the unchanged full result and projects only the model rendering", async () => {
  const definitions = new Map(), original = freeze(predictionResultFixture());
  const ctx = { tools: { register(definition) { definitions.set(definition.name, definition); return () => {}; } } };
  registerRoleTools(ctx, { role: "sample-planner", bridge: {
    bindingFor: async () => ({ role: "sample-planner", run_id: "synthetic" }),
    sidecar: { request: async () => original },
  } });
  const prediction = definitions.get("ecology_execute_prediction_tool");
  const returned = await prediction.execute({ tool_id: "candidate-model", wave_digest: "f".repeat(64),
    call_id: "candidate-evidence", parameters: {} }, { agent: {} });
  assert.equal(returned, original);
  const text = prediction.output.render({}, returned)[0].text, view = JSON.parse(text);
  assert.ok(view._display_projection);
  assert.deepEqual(expandPredictionToolResult(view), original);
  const unrelated = definitions.get("web_search").output.render({}, original)[0].text;
  assert.equal(unrelated, JSON.stringify(original));
});
