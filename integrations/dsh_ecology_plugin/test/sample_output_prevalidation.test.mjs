import assert from "node:assert/strict";
import test from "node:test";
import { installSampleOutputPrevalidation, predictionPrevalidationFeedback } from "../lib/runtime/sample-output-prevalidation.js";
import { jsonDigest } from "../lib/runtime/stage-runner.js";
import { sampleOutputTransport } from "../lib/runtime/sample-output-transport.js";
import { SidecarError } from "../lib/sidecar/client.js";

const schema = "ecologyrsi-dsh.prediction-prevalidation/1";
const args = () => ({ decisions: [{ sample_id: "synthetic", predicted: 1, confidence: .8, method: "blend", reason_code: "agent_blend", evidence_call_ids: ["cm1"] }] });
const bind = value => ({ ...value, schema_version: "ecology-sample-predictions@2", wave_digest: "a".repeat(64) });
const reply = (digest, valid = false) => ({ schema_version: schema, valid, result_digest: digest,
  diagnostics: valid ? [] : [{ code: "blend_requires_two_results", path: "value.decisions[0].evidence_call_ids" }] });
function fixture(validate = ({ resultDigest }) => reply(resultDigest), signal = null) {
  let hook, disposed = 0;
  const calls = [], ctx = { on(event, callback, options) { assert.equal(event, "tools/pre-execute"); assert.equal(options.global, true); hook = callback; return () => { disposed++; }; } };
  const exec = { name: "structured_output", arguments: args(), signal: new AbortController().signal,
    agent: { id: "child", session: { id: "child", header: { parentSession: "parent" }, snapshotEvents: () => [{ type: "subagent/descriptor", data: { label: "reserved" } }] } } };
  const handle = installSampleOutputPrevalidation(ctx, { stage: "sample.plan", parentSessionId: "parent", label: "reserved", bindOutput: bind,
    digest: jsonDigest, signal, validate: request => { calls.push(request); return validate(request); } });
  return { exec, handle, calls, invoke: (decision = { kind: "allow" }) => hook(exec, async () => decision), get disposed() { return disposed; } };
}

test("Host invalid feedback is bounded, code/path only, and successful bound scientific output is unchanged", async () => {
  const f = fixture(), before = structuredClone(f.exec.arguments), result = await f.invoke();
  assert.equal(result.kind, "deny"); assert.deepEqual(result.info, { name: "ToolArgsError", code: "INVALID_ARGS" });
  assert.deepEqual(f.exec.arguments, before); assert.equal(f.calls[0].structured.wave_digest, "a".repeat(64));
  assert.equal(f.calls[0].resultDigest, jsonDigest(bind(before))); assert.equal(f.calls[0].sessionId, "child");
  assert.ok(Buffer.byteLength(result.reason) <= 4096); assert.ok(!result.reason.includes("synthetic") && !result.reason.includes("cm1"));
  const pass = fixture(({resultDigest}) => reply(resultDigest, true)), allow = {kind: "allow"};
  assert.equal(await pass.invoke(allow), allow); assert.deepEqual(pass.exec.arguments, args());
});

test("exact response schema rejects malformed, digest mismatch, raw values, and unexpected diagnostic paths", () => {
  const digest = "a".repeat(64), good = reply(digest);
  for (const value of [null, {}, {...good, extra: "secret"}, {...good, valid: true}, {...good, diagnostics: []}, {...good, result_digest: "b".repeat(64)},
    {...good, diagnostics: [{code:"bad",path:"value.decisions[0].raw_value"}]}, {...good, diagnostics:[{code:"bad",path:"value",message:"secret"}]},
    {...good, diagnostics:Array(33).fill({code:"bad",path:"value"})}, {...good,diagnostics:[{code:"SECRET/value",path:"value"}]}]) {
    assert.throws(() => predictionPrevalidationFeedback(value, digest), error => error.code === "invalid_response");
  }
  for (const path of ["value", "value.decisions", "value.decisions[0]", "value.decisions[0].method", "value.decisions[0].evidence_call_ids[0]", "value.wave_digest"]) {
    assert.ok(predictionPrevalidationFeedback({...good, diagnostics:[{code:"invalid",path}]}, digest));
  }
});

test("unrelated parent, reservation, tool, prior denied policy and non-planner scope never call Host", async () => {
  for (const mutate of [f=>{f.exec.name="skill";},f=>{f.exec.agent.session.header.parentSession="other";},f=>{f.exec.agent.session.snapshotEvents=()=>[{type:"subagent/descriptor",data:{label:"other"}}];}]) {
    const f=fixture();mutate(f);const allow={kind:"allow"};assert.equal(await f.invoke(allow),allow);assert.equal(f.calls.length,0);f.handle.dispose();
  }
  const f=fixture(), denied={kind:"deny",reason:"prior"};assert.equal(await f.invoke(denied),denied);assert.equal(f.calls.length,0);
  assert.doesNotThrow(()=>installSampleOutputPrevalidation({}, {stage:"sample.critic"}).dispose());
});

test("authorization, unavailable, timeout and malformed response abort rather than enter INVALID_ARGS repair", async () => {
  for(const code of ["dsh_tool_authorization_failed","dsh_tool_admission_closed","dsh_prediction_binding_closed","timeout","unavailable","structured_result_persistence_unavailable"]){
    const cause=new SidecarError(code),f=fixture(()=>{throw cause;});const r=await f.invoke();
    assert.equal(r.info.code,"HOST_PREVALIDATION_FAILED");assert.equal(f.handle.error,cause);assert.equal(f.handle.signal.aborted,true);
    assert.equal((await f.invoke()).kind,"cancel");assert.equal(f.calls.length,1);
  }
  const f=fixture(()=>({valid:true}));await f.invoke();assert.equal(f.handle.error.code,"invalid_response");assert.equal(f.handle.signal.aborted,true);
});

test("same parent and label cannot authorize a second session", async () => {
  const f=fixture(({resultDigest})=>reply(resultDigest,true));await f.invoke();f.exec.agent.session.id="impostor";
  await f.invoke();assert.equal(f.handle.error.code,"dsh_tool_authorization_failed");assert.equal(f.calls.length,1);
});

test("late mutation, caller cancellation and disposal while Host is pending cannot capture", async () => {
  for(const action of ["mutate","cancel","dispose"]){
    const caller=new AbortController();let release;const f=fixture(({resultDigest})=>new Promise(resolve=>{release=()=>resolve(reply(resultDigest,true));}),caller.signal);
    const pending=f.invoke();await new Promise(resolve=>setImmediate(resolve));assert.equal(f.calls.length,1);
    if(action==="mutate")f.exec.arguments.decisions[0].predicted=999;
    if(action==="cancel")caller.abort();
    if(action==="dispose")f.handle.dispose();
    release();const result=await pending;assert.notEqual(result.kind,"allow");
    if(action==="mutate")assert.equal(f.handle.error.code,"invalid_response");else assert.equal(f.handle.error,null);
    f.handle.dispose();f.handle.dispose();assert.equal(f.disposed,1);
  }
});

test("pre-cancelled child never calls Host and cancellation does not wait for an uncooperative validator", async () => {
  const f=fixture(()=>new Promise(()=>{}));f.exec.signal=AbortSignal.abort();assert.equal((await f.invoke()).kind,"cancel");assert.equal(f.calls.length,0);
  const caller=new AbortController(),g=fixture(()=>new Promise(()=>{}),caller.signal),pending=g.invoke();await new Promise(resolve=>setImmediate(resolve));caller.abort();
  assert.equal((await pending).kind,"cancel");assert.equal(g.handle.error,null);g.handle.dispose();
});


test("native schema retains malformed primitive/array and frozen metadata correction before Host", async () => {
  const transport = sampleOutputTransport("sample.plan", { required: ["schema_version", "wave_digest", "decisions"],
    properties: { schema_version: { const: "ecology-sample-predictions@2" }, wave_digest: { const: "a".repeat(64) }, decisions: { type: "array" } } });
  let hook, calls = 0;
  const handle = installSampleOutputPrevalidation({ on(_event, callback) { hook = callback; return () => {}; } }, {
    stage: "sample.plan", parentSessionId: "p", label: "r", bindOutput: transport.bind,
    digest: jsonDigest, validate() { calls++; throw new Error("must not be called"); } });
  for (const value of [[], null, 42, { ...args(), wave_digest: "b".repeat(64) }, { ...args(), schema_version: "wrong" }]) {
    const exec = { name: "structured_output", arguments: value, agent: { session: { id: "c", header: { parentSession: "p" }, snapshotEvents: () => [{type:"subagent/descriptor",data:{label:"r"}}] } } };
    const allow = { kind: "allow" }; assert.equal(await hook(exec, async () => allow), allow);
    assert.equal(calls, 0); assert.equal(handle.error, null); assert.equal(handle.signal.aborted, false);
  }
  handle.dispose();
});

// Attestation proves an existing native binding, including direct predictions
// without a prior numerical tool result. Querying cannot create authority.
test("child activation attestation rejects forged, unclaimed, released and mismatched native state", async () => {
  const { ChildBindingRegistry } = await import("../lib/runtime/child-bindings.js");
  const registry = new ChildBindingRegistry(), frozen = { run_id: "r", role: "sample-planner", stage: "sample.plan",
    run_state_revision: 1, stage_attempt: 1, ledger_expected_revision: 2, idempotency_key: "i",
    genome_digest: "a".repeat(64), compiled_behavior_digest: "b".repeat(64), phenotype_instance_digest: "c".repeat(64), allowed_tools: [] };
  const reservation = registry.reserve("p", { reservation_id: "rid", launch_attempt: 1, run_id: "r", stage: "sample.plan", role: "sample-planner", item_digest: "d".repeat(64), idempotency_key: "i" }, frozen);
  const session = { id: "c", header: { parentSession: "p" }, snapshotEvents: () => [{type:"subagent/descriptor",data:{label:reservation.label}}] };
  const { allowed_tools, ...fields } = frozen;
  const identity = { ...fields, session_id: "c", child_reservation_id: "rid", activation_lease_id: "unissued" };
  assert.equal(registry.attest(identity, "p", session), false);assert.equal(registry.children.size,0);assert.equal(registry.activeByChild.size,0);
  registry.claimPublished("p",reservation.label,"c");identity.activation_lease_id=registry.openActivation("c",{revision:1,stage_attempt:1,idempotency_key:"i"}).lease_id;
  assert.equal(registry.attest(identity,"p",session),true);
  // turn/end alone does not release the identity needed by final persistence.
  const events=session.snapshotEvents();events.push({type:"turn/end",data:{reason:{kind:"completed"}}});session.snapshotEvents=()=>events;
  assert.equal(registry.attest(identity,"p",session),true);
  for(const key of Object.keys(identity)){
    const forged={...identity,[key]:typeof identity[key]==="number"?identity[key]+1:"other"};
    assert.equal(registry.attest(forged,"p",session),false,key);
  }
  assert.equal(registry.attest({...identity,unknown:"x"},"p",session),false);
  assert.equal(registry.attest(identity,"other",session),false);
  session.header.parentSession="other";assert.equal(registry.attest(identity,"p",session),false);session.header.parentSession="p";
  events[0].data.label="other";assert.equal(registry.attest(identity,"p",session),false);events[0].data.label=reservation.label;
  const oldLease=identity.activation_lease_id;registry.closeActivation(oldLease);identity.activation_lease_id=registry.openActivation("c",{revision:1,stage_attempt:1,idempotency_key:"i"}).lease_id;
  assert.equal(registry.attest({...identity,activation_lease_id:oldLease},"p",session),false);assert.equal(registry.attest(identity,"p",session),true);
  registry.releaseChild("c");assert.equal(registry.attest(identity,"p",session),false);assert.equal(registry.children.size,0);assert.equal(registry.activeByChild.size,0);
});
