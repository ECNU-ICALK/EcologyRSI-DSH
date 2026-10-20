import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { prepareStageOutputSchema } from "../lib/runtime/stage-runner.js";
import { researchMissingFieldFeedback, researchNestedRequiredFields,
  installResearchSchemaFeedback } from "../lib/runtime/research-schema-feedback.js";

const stage = "generation.research-synthesis";
const raw = JSON.parse(await readFile(new URL("../schemas/research-synthesis.schema.json", import.meta.url)));
const context = { required_candidate_direction_count: 4,
  knowledge_snapshot: { evidence_catalog: [{ knowledge_id: "paper:one" }] },
  synthesis_contract: {
    allowed_mutation_targets: { scientific_parameter: ["ridge_alpha"], registered_predictor: ["ridge@1"],
      instruction_profile: ["balanced@1"], instruction_parameter: ["confidence_threshold"] },
    mutation_directions_by_axis: { scientific_parameter: ["increase", "decrease"], registered_predictor: ["select"],
      instruction_profile: ["select"], instruction_parameter: ["increase", "decrease"] },
  } };
const schema = prepareStageOutputSchema(stage, raw, context);
function output() {
  return { schema_version: raw.properties.schema_version.const, summary: "offline", evidence: [],
    candidate_directions: schema.properties.candidate_directions.items.oneOf.map((branch, index) => ({
      direction_id: `d${index}`, title: "offline title", hypothesis: "offline hypothesis",
      target_weakness: "unverified weakness", capability_focus: "offline capability",
      mutation_axis: branch.properties.mutation_axis.const,
      mutation_target: branch.properties.mutation_target.enum[0],
      mutation_direction: branch.properties.mutation_direction.enum[0],
      evidence_refs: ["paper:one"], expected_tradeoff: "uncertain", success_criterion: "offline comparison",
    })) };
}
function missingTitles() {
  const value = output(); for (const direction of value.candidate_directions) delete direction.title;
  return value;
}

test("research prompt lists nested required keys from every submitted branch", () => {
  const before = structuredClone(schema);
  assert.deepEqual(researchNestedRequiredFields(stage, schema), {
    "evidence[*]": ["evidence_ref", "finding", "relevance"],
    "candidate_directions[*]": raw.properties.candidate_directions.items.required,
  });
  assert.deepEqual(researchNestedRequiredFields("sample.plan", schema), {});
  assert.deepEqual(schema, before);
});

test("O's second omission shape pinpoints each title and never changes arguments", () => {
  const value = missingTitles(), before = structuredClone(value);
  const feedback = JSON.parse(researchMissingFieldFeedback(schema, value));
  assert.deepEqual(feedback.missing_required_fields,
    value.candidate_directions.map((_, i) => `value.candidate_directions[${i}].title`));
  assert.equal(feedback.omitted_missing_field_count, 0);
  assert.deepEqual(value, before);
  for (const direction of value.candidate_directions) direction.title = "offline supplied title";
  assert.equal(researchMissingFieldFeedback(schema, value), null);
});

test("all mutation branches, unknown discriminator and malformed items keep diagnostic scope", () => {
  const value = output();
  assert.equal(researchMissingFieldFeedback(schema, value), null);
  for (const direction of value.candidate_directions) {
    delete direction.direction_id; delete direction.title; delete direction.target_weakness;
  }
  assert.equal(JSON.parse(researchMissingFieldFeedback(schema, value)).missing_required_fields.length, 12);
  value.candidate_directions[0].mutation_axis = "unknown-axis";
  const unknown = JSON.parse(researchMissingFieldFeedback(schema, value));
  assert.ok(unknown.missing_required_fields.includes("value.candidate_directions[0].title"));
  for (const malformed of [null, [], "bad", { candidate_directions: "bad" },
    { candidate_directions: [null, "bad", []] }]) assert.equal(researchMissingFieldFeedback(schema, malformed), null);
});

test("feedback is bounded even for oversized submissions and long schema field names", () => {
  const value = { candidate_directions: Array.from({ length: 1000 }, () => ({})) };
  const feedback = researchMissingFieldFeedback(schema, value), parsed = JSON.parse(feedback);
  assert.ok(Buffer.byteLength(feedback) <= 4096);
  assert.equal(parsed.missing_required_fields.length, 32);
  assert.equal(parsed.omitted_missing_field_count, 56);
  assert.equal(parsed.uninspected_direction_count, 992);
  const long = structuredClone(schema);
  long.properties.candidate_directions.items.oneOf.forEach(branch => branch.required.push("x".repeat(5000)));
  assert.ok(Buffer.byteLength(researchMissingFieldFeedback(long, value)) <= 4096);
});

function fixture() {
  let listener, unsubscriptions = 0;
  const ctx = { on(name, handler, options) {
    assert.equal(name, "tools/post-execute"); assert.equal(options.global, true);
    listener = handler; return () => { unsubscriptions++; };
  } };
  const session = { id: "child", header: { parentSession: "parent" },
    snapshotEvents: () => [{ type: "subagent/descriptor", data: { label: "reservation" } }] };
  const exec = { name: "structured_output", arguments: missingTitles(), agent: { session },
    signal: new AbortController().signal };
  const result = { isError: true, error: { message: "original error",
    info: { name: "ToolArgsError", code: "INVALID_ARGS" } },
    content: [{ type: "text", text: "original oneOf matched 0" }] };
  const handle = installResearchSchemaFeedback(ctx, { stage, schema, parentSessionId: "parent", label: "reservation" });
  return { exec, result, handle, call: (...args) => listener(...args), get unsubscriptions() { return unsubscriptions; } };
}

test("public post-execute enrichment preserves original error and downstream policies", async () => {
  const f = fixture(), before = structuredClone(f.result);
  const decision = await f.call(f.exec, f.result, async () => ({ kind: "accept", additionalContexts: [] }));
  assert.equal(decision.kind, "accept"); assert.equal(Object.hasOwn(decision, "value"), false);
  assert.deepEqual(decision.content[0], before.content[0]); assert.equal(decision.content.length, 2);
  assert.equal(JSON.parse(decision.content[1].text).missing_required_fields.length, 4);
  assert.deepEqual(f.result, before);
  const replacement = [{ type: "text", text: "downstream policy error content" }];
  const replaced = await f.call(f.exec, f.result, async () => ({ kind: "accept", content: replacement }));
  assert.deepEqual(replaced.content[0], replacement[0]);
  const block = { kind: "block", feedback: [{ type: "text", text: "blocked" }] };
  assert.equal(await f.call(f.exec, f.result, async () => block), block);
  const malformed = { kind: "accept", content: "not content blocks" };
  assert.equal(await f.call(f.exec, f.result, async () => malformed), malformed);
  const absent = { kind: "accept" };
  assert.equal(await f.call(f.exec, { ...f.result, content: undefined }, async () => absent), absent);
  f.handle.dispose(); f.handle.dispose(); assert.equal(f.unsubscriptions, 1);
  const accepted = { kind: "accept" };
  assert.equal(await f.call(f.exec, f.result, async () => accepted), accepted);
});

test("normalized hook error metadata takes precedence over explicit legacy flat projection", async () => {
  const flat = { name: "ToolArgsError", code: "INVALID_ARGS", message: "legacy error" };
  const f = fixture();
  const legacy = { ...f.result, error: flat }, before = structuredClone(legacy);
  const enriched = await f.call(f.exec, legacy, async () => ({ kind: "accept" }));
  assert.equal(JSON.parse(enriched.content.at(-1).text).missing_required_fields.length, 4);
  assert.deepEqual(legacy, before);
  for (const info of [undefined, null, {}, { name: "ProviderError", code: "TIMEOUT" },
    { name: "ToolArgsError", code: "OTHER" }]) {
    const decision = { kind: "accept" };
    const result = { ...f.result, error: { ...flat, info } };
    assert.equal(await f.call(f.exec, result, async () => decision), decision);
  }
  const normalized = { ...f.result, error: { ...f.result.error, name: "ProviderError", code: "TIMEOUT" } };
  assert.equal((await f.call(f.exec, normalized, async () => ({ kind: "accept" }))).content.length, 2);
  f.handle.dispose();
});

test("success, provider failures, aborted callers and unrelated children are untouched", async () => {
  for (const mutate of [
    f => { f.result.isError = false; },
    f => { f.result.error.info.code = "TIMEOUT"; },
    f => { f.result.error.info.name = "ProviderError"; },
    f => { f.exec.name = "other_tool"; },
    f => { f.exec.arguments = output(); },
    f => { f.exec.agent.session.header.parentSession = "other-parent"; },
    f => { f.exec.agent.session.snapshotEvents = () => [{ type: "subagent/descriptor", data: { label: "other-reservation" } }]; },
    f => { delete f.exec.agent; },
    f => { f.exec.signal = AbortSignal.abort(); },
  ]) {
    const f = fixture(); mutate(f); const decision = { kind: "accept" };
    assert.equal(await f.call(f.exec, f.result, async () => decision), decision);
    f.handle.dispose();
  }
  const f = fixture(), failure = new Error("policy failure");
  await assert.rejects(f.call(f.exec, f.result, async () => { throw failure; }), error => error === failure);
  f.handle.dispose();
});
