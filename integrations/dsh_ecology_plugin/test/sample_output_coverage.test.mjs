import assert from "node:assert/strict";
import test from "node:test";
import { installSampleOutputCoverage, sampleCoverageFeedback } from "../lib/runtime/sample-output-coverage.js";

const ids = Array.from({ length: 9 }, (_, i) => `sample-${i}`);
const valid = () => ({ decisions: ids.map(sample_id => ({ sample_id, predicted: 1, confidence: 0.8 })) });

test("exact frozen coverage permits reordered scientific values without mutation", () => {
  for (const value of [valid(), { decisions: valid().decisions.reverse() }]) {
    const before = structuredClone(value);
    assert.equal(sampleCoverageFeedback(ids, value), null);
    assert.deepEqual(value, before);
  }
});

test("R duplicate tenth decision, missing, duplicate replacement and unknown ids are rejected", () => {
  const extra = valid(); extra.decisions.push({ ...extra.decisions[8] });
  const missing = valid(); missing.decisions.pop();
  const duplicate = valid(); duplicate.decisions[8].sample_id = ids[7];
  const unknown = valid(); unknown.decisions[8].sample_id = "outside";
  for (const [value, count, duplicates, invalid, absent] of [
    [extra, 10, ["value.decisions[9].sample_id"], [], []],
    [missing, 8, [], [], [ids[8]]],
    [duplicate, 9, ["value.decisions[8].sample_id"], [], [ids[8]]],
    [unknown, 9, [], ["value.decisions[8].sample_id"], [ids[8]]],
  ]) {
    const before = structuredClone(value), feedback = JSON.parse(sampleCoverageFeedback(ids, value));
    assert.equal(feedback.expected_decision_count, 9); assert.equal(feedback.received_decision_count, count);
    assert.deepEqual(feedback.duplicate_sample_id_paths, duplicates);
    assert.deepEqual(feedback.unknown_or_missing_sample_id_paths, invalid);
    assert.deepEqual(feedback.missing_sample_ids, absent); assert.deepEqual(value, before);
  }
});

test("malformed array shape stays with native schema while diagnostics stay bounded", () => {
  for (const value of [null, [], {}, { decisions: null }]) assert.equal(sampleCoverageFeedback(ids, value), null);
  const longIds = Array.from({ length: 128 }, (_, i) => `sample-${i}-` + "x".repeat(225));
  const feedback = sampleCoverageFeedback(longIds, { decisions: Array.from({ length: 2000 }, () => ({ sample_id: "not-frozen" })) });
  assert.ok(Buffer.byteLength(feedback) <= 4096);
  assert.ok(JSON.parse(feedback).omitted_diagnostic_count > 0);
  assert.equal(JSON.parse(feedback).received_decision_count, 2000);
});

function fixture() {
  let hook, unsubscriptions = 0;
  const ctx = { on(name, callback, options) {
    assert.equal(name, "tools/pre-execute"); assert.equal(options.global, true);
    hook = callback; return () => { unsubscriptions++; };
  } };
  const samples = ids.map(sample_id => ({ sample_id }));
  const session = { id: "child", header: { parentSession: "parent" },
    snapshotEvents: () => [{ type: "subagent/descriptor", data: { label: "reserved" } }] };
  const exec = { name: "structured_output", arguments: { decisions: valid().decisions.slice(0, 8) },
    agent: { session }, signal: new AbortController().signal };
  const handle = installSampleOutputCoverage(ctx, { stage: "sample.plan", samples,
    parentSessionId: "parent", label: "reserved" });
  return { ctx, samples, exec, handle, call: (...args) => hook(...args), get unsubscriptions() { return unsubscriptions; } };
}

test("public pre-execute denies before capture with normalized INVALID_ARGS and frozen input", async () => {
  const f = fixture(); f.samples[8].sample_id = "mutated-after-registration";
  const decision = await f.call(f.exec, async () => ({ kind: "allow" }));
  assert.equal(decision.kind, "deny");
  assert.deepEqual(decision.info, { name: "ToolArgsError", code: "INVALID_ARGS" });
  assert.deepEqual(JSON.parse(decision.reason).missing_sample_ids, [ids[8]]);
  f.exec.arguments = valid(); assert.deepEqual(await f.call(f.exec, async () => ({ kind: "allow" })), { kind: "allow" });
});

test("scope, prior policies, cancellation and asynchronous disposal are never overridden", async () => {
  for (const mutate of [e => { e.name = "other"; }, e => { e.agent.session.header.parentSession = "other"; },
    e => { e.agent.session.snapshotEvents = () => [{ type: "subagent/descriptor", data: { label: "other" } }]; },
    e => { e.signal = AbortSignal.abort(); }]) {
    const f = fixture(); mutate(f.exec); const allow = { kind: "allow" };
    assert.equal(await f.call(f.exec, async () => allow), allow); f.handle.dispose();
  }
  for (const decision of [{ kind: "deny", reason: "original" }, { kind: "cancel" }, { kind: "ask" }, undefined]) {
    const f = fixture(); assert.equal(await f.call(f.exec, async () => decision), decision); f.handle.dispose();
  }
  const f = fixture(), allow = { kind: "allow" }; let release;
  const pending = f.call(f.exec, () => new Promise(resolve => { release = resolve; }));
  f.handle.dispose(); f.handle.dispose(); release(allow);
  assert.equal(await pending, allow); assert.equal(f.unsubscriptions, 1);
  assert.equal(await f.call(f.exec, async () => allow), allow);
});

test("invalid frozen context and unavailable native hook fail before child launch", () => {
  const fields = { stage: "sample.plan", parentSessionId: "parent", label: "reserved" };
  for (const samples of [[], [{ sample_id: "" }], [{ sample_id: "a" }, { sample_id: "a" }], Array(129).fill({ sample_id: "a" })]) {
    assert.throws(() => installSampleOutputCoverage({ on() {} }, { ...fields, samples }));
  }
  assert.throws(() => installSampleOutputCoverage({}, { ...fields, samples: [{ sample_id: "a" }] }), /scoped native tools hook/);
  assert.doesNotThrow(() => installSampleOutputCoverage({}, { stage: "sample.critic" }).dispose());
});
