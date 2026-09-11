import assert from "node:assert/strict";
import test from "node:test";

import { runtimeCapabilities, inferenceConfigDigest } from "../lib/runtime/capabilities.js";

test("wire and thinking config invalidate cached preflight without hashing credentials", async () => {
  const config = { providers: { p: { api: "openai-completions", apiKeyEnv: "KEY",
    baseURL: "https://example.test/v1?key=secret", headers: { authorization: "secret" },
    models: [{ id: "model", compat: { supportsStrictMode: false } }] } } };
  const initial = inferenceConfigDigest(config);
  config.providers.p.headers.authorization = "rotated";
  config.providers.p.baseURL = "https://example.test/v1?key=rotated";
  assert.equal(inferenceConfigDigest(config), initial);
  config.providers.p.models[0].compat.supportsStrictMode = true;
  assert.notEqual(inferenceConfigDigest(config), initial);
  const ctx = { settings: { get: () => config } };
  const first = await runtimeCapabilities(ctx, [{ preset_id: "p" }]);
  config.providers.p.reasoning = "off";
  const second = await runtimeCapabilities(ctx, [{ preset_id: "p" }]);
  assert.notEqual(first.presets[0].content_digest, second.presets[0].content_digest);
});


const ROOT_SERVICES = [
  "agents", "sessions", "tokenMeter", "subagents", "tools",
  "sessionPersistence", "sessionProjections", "agentPresets", "llm", "web",
];

test("missing root services are reported truthfully", async () => {
  const result = await runtimeCapabilities({ webServer: {} }, []);
  assert.equal(result.ready, false);
  assert.deepEqual(result.root_services.missing.sort(), ROOT_SERVICES.sort());
  assert.equal(result.live_agent_service_ready, false);
  assert.equal(result.first_call_verified, false);
});

test("preset mount, tool surface and route resolution do not create probe agents", async () => {
  let created = 0;
  const ctx = Object.fromEntries(ROOT_SERVICES.map((name) => [name, {}]));
  ctx.agents.create = () => { created += 1; throw new Error("must not create"); };
  ctx.agentPresets.standingKeyFor = async (id) => `standing:${id}`;
  ctx.tools.schemas = (key) => key.endsWith("ecology-researcher-v12")
    ? [{ name: "read_generation_context" }]
    : [];
  ctx.llm.resolveCallConfig = async ({ model }) => ({ model, provider: "fake" });
  const result = await runtimeCapabilities(ctx, [
    {
      preset_id: "ecology-researcher-v12",
      required_tools: ["read_generation_context"],
      model: "fake/model",
    },
  ]);
  assert.equal(created, 0);
  assert.equal(result.root_services.declared, true);
  assert.equal(result.presets[0].preset_mountable, true);
  assert.equal(result.presets[0].tool_surface_verified, true);
  assert.equal(result.presets[0].route_resolvable, true);
  assert.equal(result.presets[0].live_agent_service_ready, false);
  assert.equal(result.presets[0].first_call_verified, false);
});

test("tool surface verification rejects undeclared extra tools", async () => {
  const ctx = Object.fromEntries(ROOT_SERVICES.map((name) => [name, {}]));
  ctx.agentPresets.standingKeyFor = async (id) => `standing:${id}`;
  ctx.tools.schemas = () => [
    { name: "ecology_execute_prediction_tool" },
    { name: "unexpected_global_tool" },
  ];
  ctx.llm.resolveCallConfig = async () => ({ provider: "fake", model: "model" });
  const result = await runtimeCapabilities(ctx, [{
    preset_id: "ecology-sample-planner-v8",
    required_tools: ["ecology_execute_prediction_tool"],
  }]);
  assert.equal(result.presets[0].tool_surface_verified, false);
  assert.equal(result.ready, false);
});


test("runtime identity changes when a skill or execution limit changes", async () => {
  const {runtimeContentDigest}=await import("../lib/runtime/capabilities.js");
  const {mkdtemp,mkdir,writeFile,rm}=await import("node:fs/promises");
  const {tmpdir}=await import("node:os");
  const {pathToFileURL}=await import("node:url");
  const root=await mkdtemp(tmpdir()+"/ecology-fingerprint-");
  try {
    for(const dir of ["lib","presets","schemas"]) await mkdir(root+"/"+dir);
    await writeFile(root+"/presets/SKILL.md","two prediction calls");
    await writeFile(root+"/lib/budget.js","10");
    const url=pathToFileURL(root+"/");const initial=await runtimeContentDigest(url);
    await writeFile(root+"/presets/SKILL.md","one prediction call");
    const changedSkill=await runtimeContentDigest(url);assert.notEqual(initial,changedSkill);
    await writeFile(root+"/lib/budget.js","5");
    assert.notEqual(changedSkill,await runtimeContentDigest(url));
  } finally {await rm(root,{recursive:true,force:true});}
});
