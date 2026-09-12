import test from "node:test";
import assert from "node:assert/strict";
import { ModelContractCanary, CANARY_SCHEMA, validateCanaryRequest } from "../lib/runtime/model-canary.js";
import { structuredPhaseError, structuredFailureCode } from "../lib/runtime/structured-stage-errors.js";
import { STAGES } from "../lib/runtime/stage-runner.js";

function request(stage = "generation.search-plan") {
  return { schema_version: CANARY_SCHEMA,
    identity: { provider_id: "provider", model_id: "model", stage, role: STAGES[stage]?.role || "sample-planner",
      preset_id: ({"generation.reflect":"ecology-generation-judge-v8", "sample.plan":"ecology-sample-planner-v9", "sample.critic":"ecology-sample-critic-v5"})[stage] || "ecology-researcher-v12", output_schema_id: STAGES[stage]?.schema,
      preset_content_digest: "a".repeat(64), standing_tool_surface_digest: "b".repeat(64), route_config_digest: "c".repeat(64) },
    bounds: { max_attempts: 2, max_output_tokens: 1024, max_reported_tokens: 30000, total_timeout_ms: 1000, ttl_seconds: 3600 } };
}
function fixture(mode = "ok") {
  const stored = new Map(), launches = [], hosts = [];
  const ctx = { sessions: { get: sid => stored.get(sid) },
    sessionProjections: { snapshot: () => ({ values: { tokenUsage: { uncachedInputTokens: 200, outputTokens: 100, cacheReadTokens: 0, cacheWriteTokens: 0 } } }) },
    subagents: { async start(provider, value) {
      launches.push(value); assert.equal(provider, "spawn");
      const prompt = JSON.parse(value.prompt[0].text), structured = prompt.expected_fixture;
      const skillName = STAGES[prompt.stage].skillName || "origin-vector-forecasting";
      const sid = `child-${launches.length}`;
      const events = [{seq:1,type:"turn/start",data:{turn:1}}, {seq:2,type:"step/start",data:{turn:1,step:1}}];
      function call(seq,name,args,id) { events.push({seq,type:"tool/call",data:{turn:1,step:1,callId:id,name,arguments:JSON.stringify(args)}}); }
      function result(seq,id,isError=false) { events.push({seq,type:"tool/result",data:{turn:1,step:1,message:{content:[{type:"tool-result",toolCallId:id,isError,content:[]}]}}}); }
      const wireText = mode === "dsml" || mode === "dsml-once" && launches.length === 1;
      if (mode !== "text") {
        call(3,"skill",{name:skillName},"skill-1"); result(4,"skill-1");
        if (wireText) events.push({seq:5,type:"assistant/message",data:{turn:1,message:{content:[{type:"text",text:'<｜DSML｜tool_calls><｜DSML｜invoke name="structured_output"></｜DSML｜invoke></｜DSML｜tool_calls>'}]}}});
        else {call(5,"structured_output",structured,"result-1"); result(6,"result-1",mode === "rejected");}
      }
      events.push({seq:7,type:"assistant/message",data:{usage:{inputTokens:200,outputTokens:100}}}, {seq:8,type:"step/end",data:{turn:1,step:1}}, {seq:9,type:"turn/end",data:{turn:1,reason:{kind:"completed"}}});
      if (mode === "budget") events.at(-1).data.reason = {kind:"max-tokens"};
      if (mode === "server") events.at(-1).data.reason = {kind:"error",error:{code:"SERVER",status:503,message:"private upstream"}};
      if (mode === "transport" && launches.length === 1) events.at(-1).data.reason = {kind:"error",error:{code:"TRANSPORT",message:"secret credential must never be emitted"}};
      stored.set(sid,{events});
      const failed = wireText || mode === "budget" || mode === "server" || mode === "transport" && launches.length === 1;
      return {id:sid,result:Promise.resolve({stopReason:failed?"error":"completed",structured:failed?null:mode === "mismatch"?{success:true}:structured}),dispose:async()=>{if(mode === "server") stored.delete(sid);}};
    } },
  };
  const roleAgents = {async createRoleAgent(value) { hosts.push(value); return {agent:{id:"role-host"},binding:value}; }, async quiesceRun() {}};
  return {canary:new ModelContractCanary(ctx,{roleAgents}),ctx,launches,hosts};
}
test("real canary seam mounts exact roles and requires actual tool/schema captures", async () => {
  for (const stage of ["generation.search-plan","generation.reflect","sample.plan","sample.critic"]) {
    const f=fixture(); const receipt=await f.canary.run(request(stage));
    assert.equal(receipt.passed,true, JSON.stringify({stage,failure:receipt.failure})); assert.equal(receipt.scope,"tool_and_schema_transport_only");
    assert.equal(receipt.attempts.length,1); assert.equal(receipt.usage.reported_tokens,300);
    assert.equal(receipt.tool_evidence.source,"dsh_session_event_log");
    assert.equal(f.hosts[0].model,"provider/model"); assert.equal(f.hosts[0].preset_id,request(stage).identity.preset_id);
    assert.deepEqual(f.launches[0].agentOptions,{maxTokens:1024});
  }
});
test("text-only tool claims, failed tools and invented success cannot pass or retry", async()=>{
  for(const mode of ["text","rejected","mismatch"]){const f=fixture(mode),receipt=await f.canary.run(request());assert.equal(receipt.passed,false,mode);assert.equal(f.launches.length,1);assert.equal(receipt.recommendation,"isolate_configuration");}
});
test("output exhaustion is terminal; only classified transport gets a bounded second child",async()=>{
  const b=fixture("budget"),r=await b.canary.run(request()); assert.equal(r.passed,false); assert.equal(b.launches.length,1); assert.equal(r.failure.code,"structured_child_output_budget_exhausted");
  const t=fixture("transport"),rt=await t.canary.run(request()); assert.equal(rt.passed,true); assert.equal(t.launches.length,2); assert.doesNotMatch(JSON.stringify(rt),/secret credential/);
});
test("canary bounds and unsupported stages fail before any provider request", async()=>{
  for(const field of ["max_attempts","max_output_tokens","max_reported_tokens","total_timeout_ms","ttl_seconds"]){const q=request();q.bounds[field]=1e9;assert.throws(()=>validateCanaryRequest(q));}
  const q=request("unsupported.stage"); assert.throws(()=>validateCanaryRequest(q),e=>e.code==="model_canary_stage_unsupported");
});
test("missing usage fails closed and host creation is deadline bounded",async()=>{
  const f=fixture(); f.ctx.sessionProjections.snapshot=()=>null;const r=await f.canary.run(request());assert.equal(r.passed,false);assert.equal(r.failure.code,"model_canary_usage_unavailable");
  const stuck=new ModelContractCanary(f.ctx,{roleAgents:{createRoleAgent:()=>new Promise(()=>{}),quiesceRun:async()=>{}}});const before=Date.now();const result=await stuck.run(request());assert.equal(result.passed,false);assert.equal(result.failure.code,"structured_role_operational_timeout");assert.ok(Date.now()-before<1800);
});
test("persistence failure subcodes use an exact safe allowlist without changing retry phase",()=>{
  for(const [causeCode,expected] of [["timeout","structured_result_persist_timeout"],["dsh_skill_evidence_invalid","structured_result_persist_tool_evidence"],["unavailable","structured_result_persist_transport"],["secret/token/foo","structured_result_persist_failed"]]){
    const e=structuredPhaseError("persistence",{code:causeCode,message:"secret"});assert.equal(e.code,"structured_result_persist_failed");assert.equal(structuredFailureCode(e),expected);assert.doesNotMatch(structuredFailureCode(e),/secret/);
  }
});

test("reported-token threshold prevents success and further child retries",async()=>{
  const f=fixture("transport");
  f.ctx.sessionProjections.snapshot=()=>({values:{tokenUsage:{uncachedInputTokens:2000,outputTokens:10}}});
  const q=request();q.bounds.max_reported_tokens=1024;
  const receipt=await f.canary.run(q);
  assert.equal(receipt.passed,false);assert.equal(f.launches.length,1);
  assert.equal(receipt.usage.reported_tokens,2010);
  assert.equal(receipt.usage.budget_semantics,"reported_usage_abort_threshold_with_provider_streaming_lag");
});

test("receipt retains failed-attempt usage and never grants sample.plan qualification",async()=>{
  const f=fixture("transport");const r=await f.canary.run(request());
  assert.equal(r.usage.reported_tokens,600);assert.equal(r.usage.observed_session_count,2);
  assert.equal(r.usage.launched_session_count,2);assert.equal(r.scope,"tool_and_schema_transport_only");
  assert.equal(r.usage.complete,false);assert.equal(r.usage.complete_session_count,1);
  assert.equal(r.identity.stage,"generation.search-plan");
});


test("provider 503 classification survives disposal of the live child session", async () => {
  const f=fixture("server"); const q=request();q.bounds.max_attempts=1;
  const receipt=await f.canary.run(q);
  assert.equal(receipt.passed,false);
  assert.equal(receipt.failure.code,"model_canary_provider_unavailable");
  assert.equal(receipt.failure.provider_status,503);
  assert.doesNotMatch(JSON.stringify(receipt),/private upstream/);
});

test("serialized output after a real Skill gets one genuine protocol probe", async () => {
  const f=fixture("dsml-once"), r=await f.canary.run(request("sample.critic"));
  assert.equal(r.passed,true); assert.equal(f.launches.length,2);
  assert.equal(r.attempts[0].accepted,false);
  assert.equal(r.attempts[0].failure.code,"structured_child_tool_protocol_error");
  assert.equal(r.attempts[1].accepted,true);
  const bad=fixture("dsml"), rejected=await bad.canary.run(request("sample.critic"));
  assert.equal(rejected.passed,false); assert.equal(bad.launches.length,2);
  assert.equal(rejected.tool_evidence,null);
});
