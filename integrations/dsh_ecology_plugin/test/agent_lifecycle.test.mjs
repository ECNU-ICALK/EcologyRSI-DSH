import assert from "node:assert/strict";
import test from "node:test";

import { RoleAgentManager, tokenSemantics } from "../lib/runtime/agents.js";

test("role-host creation is single-flight, resumable and has no token hard cap", async () => {
  const calls = [];
  const agent = { id: "unpredictable", session: { append: async (...event) => calls.push(["append", ...event]), flush: async () => calls.push(["flush"]) } };
  const handle = { agent, dispose: async () => calls.push(["dispose"]) };
  const ctx = {
    agents: { create: async (options) => { calls.push(["create", options]); return handle; } },
    agentPresets: {
      standingKeyFor: async (id) => `standing:${id}`,
      mount: async (_agentCtx, id) => ({ id }),
      serviceFor: (_agent, name) => ({ name }),
    },
  };
  const manager = new RoleAgentManager(ctx);
  const binding = { run_id: "r1", role: "coordinator", preset_id: "ecology-coordinator-v1", model: "p/m", cwd: "/tmp", require_workflow: true };
  const [a, b] = await Promise.all([manager.createRoleAgent(binding), manager.createRoleAgent(binding)]);
  assert.equal(a, b);
  assert.equal(calls.filter(([name]) => name === "create").length, 1);
  const options = calls.find(([name]) => name === "create")[1];
  assert.equal("maxTokens" in options, false);
  assert.deepEqual(options.agentOptions, { provider: "p", model: "m" });
  assert.equal(options.meta.agentPreset, binding.preset_id);
  assert.equal(options.meta.cwd, binding.cwd);
  await options.setup({});
  assert.match(options.sessionId, /^ecology-role-/);
  assert.deepEqual(calls.find(([name]) => name === "append").slice(1), [
    "agent-preset/selected",
    { agentPreset: binding.preset_id },
  ]);
  assert.deepEqual(a.services.compaction, { name: "compaction" });
  assert.deepEqual(a.services.workflowEngine, { name: "workflowEngine" });
  assert.equal(calls.some(([name]) => name === "flush"), true);
  await manager.dispose();
  assert.equal(calls.some(([name]) => name === "dispose"), true);
});

test("agent tool entry clears inherited tools without allowlisting local names", () => {
  const calls = [];
  const ctx = { tools: { restrict: (config) => calls.push(config), register: () => () => {} } };
  return import("../lib/tools/agent-plugin.js").then(({ apply }) => {
    apply(ctx, { role: "researcher" });
    assert.deepEqual(calls, [{ allow: [] }]);
  });
});

test("token pressure and cumulative provider usage stay distinct", () => {
  const ctx = {
    tokenMeter: { measure: () => ({ used: 80, limit: 100, ratio: 0.8 }) },
    sessionProjections: { snapshot: () => ({ values: { tokenUsage: { input: 300, output: 40 } } }) },
  };
  assert.deepEqual(tokenSemantics(ctx, {}), {
    context_pressure: { used: 80, limit: 100, ratio: 0.8 },
    provider_usage: { input: 300, output: 40 },
  });
});
