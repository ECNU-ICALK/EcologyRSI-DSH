import assert from "node:assert/strict";
import test from "node:test";

import { RoleAgentManager, tokenSemantics } from "../lib/runtime/agents.js";

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
  const binding = { run_id: "r1", role: "coordinator", preset_id: "ecology-coordinator-v4", model: "p/m", cwd: "/tmp" };
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
    preset_id: "ecology-researcher-v7",
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
    preset_id: "ecology-researcher-v7",
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
