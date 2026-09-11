import assert from "node:assert/strict";
import test from "node:test";

import { RoleAgentManager, roleRequestOptions, tokenSemantics } from "../lib/runtime/agents.js";
import { withThinkingCapabilities } from "../../../scripts/configure_dsh_reasoning.mjs";
import { installRoleReasoning } from "../lib/tools/agent-plugin.js";

test("preset binds thinking at the real request waterfall for hosts and descendants", async () => {
  let listener, lookups = 0, disposed = false;
  const dispose = installRoleReasoning({
    on: (event, callback) => {
      assert.equal(event, "agent/request"); listener = callback;
      return () => { disposed = true; };
    },
    llm: { resolveModelInfo: async () => {
      lookups++;
      return { reasoning: { efforts: [{ id: "off" }, { id: "high" }] } };
    } },
  }, "sample-planner");
  const config = { provider: "p", model: "m", maxTokens: 8192, reasoningEffort: "high" };
  const requests = await Promise.all([listener({}, async () => config), listener({}, async () => config)]);
  assert.equal(lookups, 1);
  for (const request of requests) assert.deepEqual(request, { ...config, reasoningEffort: "off" });
  assert.equal(config.reasoningEffort, "high");
  dispose(); assert.equal(disposed, true);
});

test("sample reasoning uses declared off support and research keeps deep thinking", async () => {
  const ctx = { llm: { resolveModelInfo: async (provider, model, signal) => {
    assert.equal(provider, "p"); assert.equal(model, "m");
    assert.ok(signal instanceof AbortSignal);
    return { reasoning: { efforts: [{ id: "off" }, { id: "low" }, { id: "high" }] } };
  } } };
  for (const role of ["sample-planner", "sample-critic", "researcher", "candidate-proposer"]) {
    const options = await roleRequestOptions(ctx, { model: "p/m", role });
    assert.equal(options.reasoningEffort, role.startsWith("sample-") ? "off" : "high");
  }
  assert.equal((await roleRequestOptions(ctx, { model: "p/m", role: "generation-judge" })).reasoningEffort, "low");
  for (const reasoning of [undefined, { efforts: [{ id: "medium" }], defaultEffort: "medium" }]) {
    ctx.llm.resolveModelInfo = async () => ({ reasoning });
    const options = await roleRequestOptions(ctx, { model: "p/m", role: "sample-planner" });
    assert.equal(options.reasoningEffort, reasoning?.defaultEffort);
  }
});

test("GLM instance settings declare its actual thinking wire format without mutating other routes", () => {
  const settings = { "llm-pi-ai": { providers: {
    p: { api: "openai-completions", apiKeyEnv: "TEST_KEY", models: [{ id: "glm-5.2" }, { id: "another" }] },
  } } };
  const configured = withThinkingCapabilities(settings, "p/glm-5.2");
  assert.deepEqual(settings["llm-pi-ai"].providers.p.models[0], { id: "glm-5.2" });
  const provider = configured["llm-pi-ai"].providers.p;
  assert.equal(provider.apiKeyEnv, "TEST_KEY");
  assert.deepEqual(provider.models[1], { id: "another" });
  assert.deepEqual(provider.models[0].reasoningEfforts, { off: null, high: "high" });
  assert.equal(provider.models[0].compat.thinkingFormat, "zai");
  assert.equal(provider.models[0].compat.supportsReasoningEffort, false);
  assert.equal(provider.models[0].compat.supportsStrictMode, false);
  assert.equal(provider.models[0].compat.supportsDeveloperRole, false);
  assert.equal(provider.models[0].compat.maxTokensField, "max_tokens");
  assert.throws(() => withThinkingCapabilities(settings, "p/another"));
  assert.throws(() => withThinkingCapabilities(settings, "missing/glm-5.2"));
  settings["llm-pi-ai"].providers.p.models.push({ id: "deepseek-v4-flash-0731" });
  const dsk = withThinkingCapabilities(settings, "p/deepseek-v4-flash-0731")["llm-pi-ai"].providers.p.models[2];
  assert.deepEqual(dsk.reasoningEfforts, { off: null, low: "low", high: "high" });
  assert.equal(dsk.compat.thinkingFormat, "deepseek");
  assert.equal(dsk.compat.supportsReasoningEffort, true);
  assert.equal(dsk.compat.requiresReasoningContentOnAssistantMessages, true);
  assert.equal(dsk.compat.supportsStrictMode, false);
});

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function outcome(promise) {
  return Promise.resolve(promise).then(
    (value) => ({ status: "fulfilled", value }),
    (reason) => ({ status: "rejected", reason }),
  );
}

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
  const binding = { run_id: "r1", role: "coordinator", preset_id: "ecology-coordinator-v5", model: "p/m", cwd: "/tmp" };
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
  assert.equal("workflowEngine" in a.services, false);
  assert.equal(calls.some(([name]) => name === "flush"), true);
  await manager.dispose();
  assert.equal(calls.some(([name]) => name === "dispose"), true);
});

test("run quiescence waits pending creations and disposes every published host", async () => {
  const releaseResearcher = deferred();
  const releaseProposer = deferred();
  const creationEntered = [deferred(), deferred()];
  const disposals = [];
  const ctx = {
    agents: {
      create: async (options) => {
        const role = options.meta.ecologyRole;
        if (role === "researcher") {
          creationEntered[0].resolve();
          return await releaseResearcher.promise;
        }
        creationEntered[1].resolve();
        return await releaseProposer.promise;
      },
    },
    agentPresets: {
      standingKeyFor: async (presetId) => `standing:${presetId}`,
      mount: async (_agentCtx, presetId) => ({ id: presetId }),
      serviceFor: async () => ({ ready: true }),
    },
  };
  const manager = new RoleAgentManager(ctx);
  const common = {
    run_id: "run-pending-cleanup",
    model: "provider/model",
    cwd: "/tmp",
  };
  const researcher = manager.createRoleAgent({
    ...common,
    role: "researcher",
    preset_id: "ecology-researcher-v12",
  });
  const proposer = manager.createRoleAgent({
    ...common,
    role: "candidate-proposer",
    preset_id: "ecology-candidate-proposer-v4",
  });
  await Promise.all(creationEntered.map((item) => item.promise));

  let cleanupSettled = false;
  const cleanup = outcome(manager.quiesceRun("run-pending-cleanup", {
    dispose: true,
  })).finally(() => { cleanupSettled = true; });
  await new Promise((resolve) => setImmediate(resolve));
  const cleanupSettledBeforeCreation = cleanupSettled;
  releaseResearcher.resolve({
    agent: {
      session: { append: async () => {}, flush: async () => {} },
      waitForIdle: async () => {
        throw new Error("private researcher idle failure");
      },
    },
    dispose: async () => {
      disposals.push("researcher");
      throw new Error("private researcher disposal failure");
    },
  });
  releaseProposer.resolve({
    agent: {
      session: { append: async () => {}, flush: async () => {} },
      waitForIdle: async () => {},
    },
    dispose: async () => { disposals.push("candidate-proposer"); },
  });

  const [researcherResult, proposerResult, cleanupResult] = await Promise.all([
    outcome(researcher),
    outcome(proposer),
    cleanup,
  ]);

  assert.equal(cleanupSettledBeforeCreation, false);
  assert.equal(researcherResult.status, "fulfilled");
  assert.equal(proposerResult.status, "fulfilled");
  assert.equal(cleanupResult.status, "rejected");
  assert.match(cleanupResult.reason.message, /private researcher idle failure/);
  assert.deepEqual(disposals.sort(), ["candidate-proposer", "researcher"]);
  assert.equal(manager.get("run-pending-cleanup", "researcher"), null);
  assert.equal(manager.get("run-pending-cleanup", "candidate-proposer"), null);
});

test("role creation preserves its setup error when private disposal also fails", async () => {
  let disposals = 0;
  const manager = new RoleAgentManager({
    agents: {
      create: async () => ({
        agent: {
          session: { append: async () => {}, flush: async () => {} },
        },
        dispose: async () => {
          disposals += 1;
          throw new Error("private role disposal failure");
        },
      }),
    },
    agentPresets: {
      standingKeyFor: async (presetId) => `standing:${presetId}`,
      mount: async (_agentCtx, presetId) => ({ id: presetId }),
      serviceFor: async () => { throw new Error("primary role service failure"); },
    },
  });

  const result = await outcome(manager.createRoleAgent({
    run_id: "run-primary-role-error",
    role: "researcher",
    preset_id: "ecology-researcher-v12",
    model: "provider/model",
    cwd: "/tmp",
  }));

  assert.equal(result.status, "rejected");
  assert.match(result.reason.message, /primary role service failure/);
  assert.equal(disposals, 1);
});

test("agent tool entry keeps descendant-visible role tools and guards execution", () => {
  let guard;
  let restrictions = 0;
  const ctx = {
    tools: {
      restrict: () => { restrictions += 1; },
      register: () => () => {},
      guard: (candidate) => { guard = candidate; return () => {}; },
    },
  };
  return import("../lib/tools/agent-plugin.js").then(({ apply }) => {
    apply(ctx, { role: "researcher" });
    assert.equal(restrictions, 0);
    assert.equal(guard({ name: "structured_output" }), undefined);
    assert.match(guard({ name: "unexpected_tool" }), /outside the frozen researcher role surface/);
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
