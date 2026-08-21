import assert from "node:assert/strict";
import test from "node:test";

import { runtimeCapabilities } from "../lib/runtime/capabilities.js";


const ROOT_SERVICES = [
  "agents", "sessions", "tokenMeter", "subagents", "tools",
  "sessionPersistence", "sessionProjections", "agentPresets", "llm",
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
  ctx.tools.schemas = (key) => key.endsWith("researcher-v1")
    ? [{ name: "read_generation_context" }]
    : [];
  ctx.llm.resolveCallConfig = async ({ model }) => ({ model, provider: "fake" });
  const result = await runtimeCapabilities(ctx, [
    {
      preset_id: "ecology-researcher-v1",
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
