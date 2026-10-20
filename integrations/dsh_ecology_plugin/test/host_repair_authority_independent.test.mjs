import assert from "node:assert/strict";
import test from "node:test";
import { installSampleOutputPrevalidation, trustedSamplePredictionRepair } from "../lib/runtime/sample-output-prevalidation.js";
import { jsonDigest, skillInvocationEvidence, synchronizedSkillInvocationEvidence } from "../lib/runtime/stage-runner.js";
import { SidecarError } from "../lib/sidecar/client.js";

// Independent, public synthetic events. The injected Host response is explicit;
// the production authenticator/HTTP path is covered by the separate real harness.
const schema = "ecologyrsi-dsh.prediction-prevalidation/1";
const baseOptions = { stage: "sample.plan", skillName: "origin-vector-forecasting", allowsPredictionTools: true };
const value = refs => ({ decisions: [{ sample_id: "sample-public", predicted: 12, confidence: .8,
  method: "blend", reason_code: "agent_blend", evidence_call_ids: refs }] });
const bind = args => ({ ...args, schema_version: "ecology-sample-predictions@2", wave_digest: "a".repeat(64) });

async function fixture({ firstCodes = ["blend_requires_two_results"], intervening = ["ecology_execute_prediction_tool"],
  failedToolFirst = false, finalValidated = true, hostError = null, callerSignal = null } = {}) {
  let callback, validations = 0, unsubscribed = 0;
  const events = [{ seq: 1, type: "subagent/descriptor", data: { label: "reservation-public" } },
    { seq: 2, type: "turn/start", data: { turn: 1 } }];
  const session = { id: "child-public", header: { parentSession: "parent-public" }, snapshotEvents: () => events };
  function call(name, step, args = {}) {
    events.push({ seq: events.length + 1, type: "step/start", data: { turn: 1, step } });
    const seq = events.length + 1;
    const item = { seq, type: "tool/call", data: { turn: 1, step, callId: "call-" + seq, name, arguments: structuredClone(args) } };
    events.push(item); return item;
  }
  function result(item, isError = false, error = null, text = "synthetic-public-result") {
    const out = { seq: events.length + 1, type: "tool/result", sourceEventSeqs: [item.seq],
      data: { turn: 1, step: item.data.step, message: { role: "tool", source: { kind: "tool", callId: item.data.callId }, toolCallId: item.data.callId, isError,
        content: [{ type: "text", text }] }, ...(error ? { error } : {}) } };
    events.push(out); return out;
  }
  result(call("skill", 1, { name: baseOptions.skillName }));
  result(call("ecology_execute_prediction_tool", 2));
  const handle = installSampleOutputPrevalidation({ on(_name, fn) { callback = fn; return () => { unsubscribed++; }; } }, {
    stage: "sample.plan", parentSessionId: "parent-public", label: "reservation-public", bindOutput: bind, signal: callerSignal,
    digest: jsonDigest, validate({ resultDigest }) {
      validations++;
      if (hostError) throw hostError;
      return { schema_version: schema, result_digest: resultDigest, valid: validations > 1,
        diagnostics: validations > 1 ? [] : firstCodes.map(code => ({ code, path: "value.decisions[0].evidence_call_ids" })) };
    },
  });
  async function invoke(item, customSession = session) {
    return callback({ name: item.data.name, callId: item.data.callId, arguments: item.data.arguments,
      signal: new AbortController().signal, agent: { session: customSession } }, async () => ({ kind: "allow" }));
  }
  const first = call("structured_output", 3, value(["prediction-1"]));
  const denial = await invoke(first);
  result(first, true, denial.info, denial.reason);
  if (hostError) return { events, handle, first, denial, invoke, session, validations: () => validations };
  let step = 4;
  for (const [index, name] of intervening.entries()) result(call(name, step++), failedToolFirst && index === 0);
  const last = call("structured_output", step, value(["prediction-1", "prediction-2"]));
  if (finalValidated) assert.equal((await invoke(last)).kind, "allow");
  result(last);
  events.push({ seq: events.length + 1, type: "turn/end", data: { turn: 1, reason: { kind: "completed" } } });
  const resultDigest = jsonDigest(bind(last.data.arguments));
  const proof = { authority: handle.repairAuthority, sessionId: session.id, resultDigest };
  return { events, handle, first, last, denial, invoke, session, proof, resultDigest, validations: () => validations, unsubscribed: () => unsubscribed };
}

function trusted(f, events = f.events, options = {}) {
  return trustedSamplePredictionRepair(f.handle.repairAuthority, events,
    { stage: "sample.plan", sessionId: f.session.id, resultDigest: f.resultDigest, ...options });
}

test("independent: only a Host-minted and final-validated repair authorizes settled supplemental prediction evidence", async () => {
  const f = await fixture();
  try {
    assert.equal(f.validations(), 2);
    assert.equal(trusted(f), true);
    const evidence = skillInvocationEvidence(f.events, { ...baseOptions, samplePredictionRepair: f.proof });
    assert.equal(evidence.order_verified, true);
    assert.equal(evidence.next_tool_call_seq, f.last.seq);
    assert.throws(() => skillInvocationEvidence(f.events, baseOptions));
  } finally { f.handle.dispose(); }
});

test("independent: forged/copied authority and cross-session, stage, digest or source sequence cannot grant repair", async () => {
  const f = await fixture();
  try {
    for (const authority of [null, {}, { ...f.handle.repairAuthority }, structuredClone(f.handle.repairAuthority)]) {
      assert.equal(trustedSamplePredictionRepair(authority, f.events,
        { stage: "sample.plan", sessionId: f.session.id, resultDigest: f.resultDigest }), false);
    }
    for (const options of [{ sessionId: "other-child" }, { stage: "sample.critic" }, { stage: "generation.reflect" }, { resultDigest: "b".repeat(64) }]) {
      assert.equal(trusted(f, f.events, options), false);
    }
    const mutations = [
      events => { events.find(e => e.seq === f.first.seq).data.callId = "forged-call"; },
      events => { events.find(e => e.type === "tool/result" && e.sourceEventSeqs[0] === f.first.seq).sourceEventSeqs = [999]; },
      events => { events.find(e => e.seq === f.last.seq).data.turn = 2; },
      events => { events.find(e => e.seq === f.last.seq).seq = 999; },
    ];
    // The authority binds calls; source-result/terminal integrity is additionally
    // checked by the terminal evidence verifier. Exercise the complete boundary.
    for (const mutate of mutations) {
      const events = structuredClone(f.events); mutate(events);
      assert.throws(() => skillInvocationEvidence(events, { ...baseOptions, samplePredictionRepair: f.proof }));
    }
  } finally { f.handle.dispose(); }
});

test("independent: ordinary schema diagnostics, mixed diagnostics and missing final Host validation stay strict", async () => {
  for (const options of [{ firstCodes: ["prediction_fields_invalid"] },
    { firstCodes: ["blend_requires_two_results", "method_not_allowed"] }, { finalValidated: false }]) {
    const f = await fixture(options);
    try {
      assert.equal(trusted(f), false);
      assert.throws(() => skillInvocationEvidence(f.events, { ...baseOptions, samplePredictionRepair: f.proof }));
    } finally { f.handle.dispose(); }
  }
});

test("independent: authorization/closed/timeout errors never become a semantic repair grant", async () => {
  for (const code of ["dsh_tool_authorization_failed", "dsh_tool_admission_closed", "dsh_prediction_binding_closed", "timeout", "invalid_response"]) {
    const cause = new SidecarError(code), f = await fixture({ hostError: cause });
    try {
      assert.equal(f.denial.info.code, "HOST_PREVALIDATION_FAILED");
      assert.equal(f.handle.error, cause);
      assert.equal(f.handle.signal.aborted, true);
      assert.equal(trustedSamplePredictionRepair(f.handle.repairAuthority, f.events,
        { stage: "sample.plan", sessionId: f.session.id, resultDigest: "a".repeat(64) }), false);
    } finally { f.handle.dispose(); }
  }
});

test("independent: the repair permits no search or new Skill and no work after an accepted output", async () => {
  for (const name of ["web_search", "skill", "arbitrary_tool"]) {
    const f = await fixture({ intervening: [name] });
    try { assert.throws(() => skillInvocationEvidence(f.events, { ...baseOptions, allowDynamicRetrieval: true, samplePredictionRepair: f.proof })); }
    finally { f.handle.dispose(); }
  }
  const f = await fixture();
  try {
    const events = structuredClone(f.events), terminal = events.pop(), seq = events.at(-1).seq + 1;
    events.push({ seq, type: "tool/call", data: { turn: 1, step: 9, callId: "late", name: "ecology_execute_prediction_tool", arguments: {} } },
      { seq: seq + 1, type: "tool/result", sourceEventSeqs: [seq], data: { turn: 1, step: 9, message: { role: "tool", source: { kind: "tool", callId: "late" }, toolCallId: "late", isError: false, content: [] } } },
      { ...terminal, seq: seq + 2 });
    assert.throws(() => skillInvocationEvidence(events, { ...baseOptions, samplePredictionRepair: f.proof }));
  } finally { f.handle.dispose(); }
});

test("independent: settled failed evidence remains visible; a later legal result may still support final output", async () => {
  const f = await fixture({ intervening: ["ecology_execute_prediction_tool", "ecology_execute_prediction_tool"], failedToolFirst: true });
  try {
    assert.equal(trusted(f), true);
    assert.equal(skillInvocationEvidence(f.events, { ...baseOptions, samplePredictionRepair: f.proof }).order_verified, true);
    assert.equal(f.events.filter(e => e.type === "tool/call" && e.data.name === "ecology_execute_prediction_tool").length, 3);
  } finally { f.handle.dispose(); }
});

test("independent: more than two outputs remain invalid", async () => {
  const f = await fixture();
  try {
    const events = structuredClone(f.events), terminal = events.pop(), seq = events.at(-1).seq + 1;
    events.push({ seq, type: "tool/call", data: { turn: 1, step: 9, callId: "third", name: "structured_output", arguments: value(["prediction-1", "prediction-2"]) } },
      { seq: seq + 1, type: "tool/result", sourceEventSeqs: [seq], data: { turn: 1, step: 9, message: { role: "tool", source: { kind: "tool", callId: "third" }, toolCallId: "third", isError: false, content: [] } } },
      { ...terminal, seq: seq + 2 });
    assert.equal(trusted(f, events), false);
    assert.throws(() => skillInvocationEvidence(events, { ...baseOptions, samplePredictionRepair: f.proof }));
  } finally { f.handle.dispose(); }
});


test("independent: caller cancellation, later authorization failure and disposal revoke minted repair authority", async () => {
  const controller = new AbortController(), cancelled = await fixture({ callerSignal: controller.signal });
  try {
    assert.equal(trusted(cancelled), true);
    controller.abort(new Error("original deadline expired"));
    assert.equal(trusted(cancelled), false);
    assert.throws(() => skillInvocationEvidence(cancelled.events, { ...baseOptions, samplePredictionRepair: cancelled.proof }));
  } finally { cancelled.handle.dispose(); }
  const revoked = await fixture();
  try {
    const decision = await revoked.invoke(revoked.last, { ...revoked.session, id: "different-child-same-label" });
    assert.equal(decision.info.code, "HOST_PREVALIDATION_FAILED");
    assert.equal(revoked.handle.error.code, "dsh_tool_authorization_failed");
    assert.equal(trusted(revoked), false);
  } finally { revoked.handle.dispose(); }
  const disposed = await fixture();
  assert.equal(trusted(disposed), true);
  disposed.handle.dispose(); disposed.handle.dispose();
  assert.equal(disposed.unsubscribed(), 1);
  assert.equal(trusted(disposed), false);
  assert.equal((await disposed.invoke(disposed.last)).kind, "allow");
  assert.equal(disposed.validations(), 2);
});

test("independent: unrelated parent or reservation never invokes Host validation or inherits the grant", async () => {
  const f = await fixture();
  try {
    for (const other of [
      { ...f.session, id: "other-parent-child", header: { parentSession: "other-parent" } },
      { ...f.session, id: "other-reservation-child", snapshotEvents: () => f.events.map(event => event.type === "subagent/descriptor"
        ? { ...event, data: { label: "other-reservation" } } : event) },
    ]) {
      assert.equal((await f.invoke(f.last, other)).kind, "allow");
      assert.equal(f.validations(), 2);
      assert.equal(trusted(f, f.events, { sessionId: other.id }), false);
      assert.equal(trusted(f), true);
    }
  } finally { f.handle.dispose(); }
});

test("independent: repair does not expand the existing receipt cap or grant prediction capability to another role", async () => {
  // Six is the pre-existing evidence receipt cap, NOT the frozen Host execution
  // budget. Real Host tests separately prove refused attempts do not execute.
  for (const [extraCalls, accepted] of [[5, true], [6, false]]) {
    const f = await fixture({ intervening: Array(extraCalls).fill("ecology_execute_prediction_tool") });
    try {
      const options = { ...baseOptions, samplePredictionRepair: f.proof };
      if (accepted) assert.equal(skillInvocationEvidence(f.events, options).order_verified, true);
      else assert.throws(() => skillInvocationEvidence(f.events, options), /capability\/call budget/);
      assert.throws(() => skillInvocationEvidence(f.events, { ...options, allowsPredictionTools: false }));
    } finally { f.handle.dispose(); }
  }
});

test("independent: supplemental receipts must be unique, settled, source-bound and in the same turn", async () => {
  const f = await fixture();
  try {
    const extraCall = f.events.find(e => e.type === "tool/call" && e.data.name === "ecology_execute_prediction_tool" && e.seq > f.first.seq);
    const receipt = f.events.find(e => e.type === "tool/result" && e.sourceEventSeqs[0] === extraCall.seq);
    for (const mutate of [
      events => { events.splice(events.findIndex(e => e.seq === receipt.seq), 1); },
      events => { events.push(structuredClone(receipt)); },
      events => { events.find(e => e.seq === receipt.seq).sourceEventSeqs = [f.first.seq]; },
      events => { events.find(e => e.seq === receipt.seq).data.step++; },
      events => { events.find(e => e.seq === receipt.seq).data.turn++; },
      events => { events.find(e => e.seq === receipt.seq).seq = f.last.seq + 1; },
      events => { events.find(e => e.seq === extraCall.seq).data.turn++; },
    ]) {
      const events = structuredClone(f.events); mutate(events);
      assert.throws(() => skillInvocationEvidence(events, { ...baseOptions, samplePredictionRepair: f.proof }));
    }
  } finally { f.handle.dispose(); }
});

test("independent: synchronized terminal verifier shares the grant and classifies successful invalid trajectories as evidence errors", async () => {
  const deadline = { throwIfExpired() {}, remainingTimeoutMs: () => 1000, signal: new AbortController().signal };
  for (const [options, succeeds] of [[{}, true], [{ firstCodes: ["prediction_fields_invalid"] }, false], [{ intervening: ["web_search"] }, false]]) {
    const f = await fixture(options);
    try {
      const ctx = { sessions: { get: id => id === f.session.id ? f.session : null } };
      const run = () => synchronizedSkillInvocationEvidence(ctx, f.session.id, { ...baseOptions, samplePredictionRepair: f.proof }, deadline);
      if (succeeds) assert.equal((await run()).order_verified, true);
      else await assert.rejects(run, error => error.code === "dsh_skill_evidence_invalid");
    } finally { f.handle.dispose(); }
  }
  const missing = await fixture();
  try {
    missing.events.splice(missing.events.findIndex(e => e.seq === missing.last.seq), 2);
    const ctx = { sessions: { get: () => missing.session } };
    await assert.rejects(() => synchronizedSkillInvocationEvidence(ctx, missing.session.id,
      { ...baseOptions, samplePredictionRepair: missing.proof }, deadline),
    error => error.code === "structured_result_missing");
  } finally { missing.handle.dispose(); }
});
