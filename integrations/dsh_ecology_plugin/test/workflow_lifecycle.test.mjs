import assert from "node:assert/strict";
import test from "node:test";

import { PendingChildStarts, startHomogeneousWorkflow } from "../lib/runtime/workflows.js";

test("workflow uses a host-authored script and structured bounded arguments", async () => {
  const starts = [];
  const run = { dispose: async () => starts.push(["dispose"]) };
  const roleHost = {
    agent: { id: "planner-role-host" },
    services: { workflowEngine: { start: (request) => { starts.push(request); return run; } } },
  };
  const items = [{
    label: "safe-1",
    prompt: "literal ${notInterpolated}",
    schema: { type: "object", properties: { x: { type: "integer" } } },
  }];
  const result = startHomogeneousWorkflow(roleHost, {
    template_id: "ecology-one-shot-v1", max_total_agents: 2, max_concurrent: 1, max_items: 2, sync_timeout_ms: 5000,
  }, items);
  assert.equal(result, run);
  assert.doesNotMatch(starts[0].script, /notInterpolated/);
  assert.match(starts[0].script, /agent\(item\.prompt/);
  assert.deepEqual(starts[0].args.items, items);
  assert.equal(starts[0].args.maxConcurrent, 1);
  assert.equal(starts[0].maxTotalAgents, 2);
  assert.equal(starts[0].parent, roleHost.agent);
  assert.equal(starts[0].meta.name, "ecology-one-shot-v1");
  assert.throws(() => startHomogeneousWorkflow(roleHost, { template_id: "ecology-one-shot-v1", max_items: 0 }, items), /max_items/);
});

test("cancellation covers a one-shot start before its promise settles", async () => {
  let resolveStart;
  let disposed = false;
  const starts = new PendingChildStarts({
    subagents: { start: (_request) => new Promise((resolve) => { resolveStart = resolve; }) },
  });
  const pending = starts.start("one-shot", { prompt: "work" }, {});
  assert.equal(starts.size, 1);
  const cancelling = starts.cancelAndQuiesce();
  resolveStart({ dispose: async () => { disposed = true; } });
  await cancelling;
  assert.equal(pending.signal.aborted, true);
  assert.equal(disposed, true);
  assert.equal(starts.size, 0);
});

test("accepted continuable starts are interrupted and drained", async () => {
  const calls = [];
  const starts = new PendingChildStarts({
    subagents: {
      startContinuable: async () => ({ childId: "child-1", messageId: "message-1" }),
      interrupt: async (id) => calls.push(["interrupt", id]),
      drainContinuableDescendants: async (parents) => calls.push(["drain", parents]),
    },
  });
  starts.start("continuable", { prompt: "work" }, { roleHostAgent: { id: "parent-1" } });
  await starts.cancelAndQuiesce();
  assert.deepEqual(calls, [["interrupt", "child-1"], ["drain", [{ id: "parent-1" }]]]);
});

test("reentrant drain observes pending work before the external starter returns", async () => {
  let draining;
  const releaseStart = {};
  let disposed = false;
  const startResult = new Promise((resolve) => {
    releaseStart.resolve = () => resolve({
      dispose: async () => { disposed = true; },
    });
  });
  const starts = new PendingChildStarts({
    subagents: {
      start: () => {
        draining = starts.cancelAndQuiesce({ runId: "run-reentrant" });
        return startResult;
      },
    },
  });

  const pending = starts.start(
    "one-shot",
    { provider: "spawn", prompt: "work" },
    { runId: "run-reentrant" },
  );
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(starts.size, 1);
  assert.equal(pending.signal.aborted, true);

  releaseStart.resolve();
  await draining;

  assert.equal(disposed, true);
  assert.equal(starts.size, 0);
});

test("Workflow launch is registered before engine start can reenter drain", async () => {
  let draining;
  let cancelled = false;
  const starts = new PendingChildStarts({ subagents: {} });

  const pending = starts.startWorkflow(() => {
    draining = starts.cancelAndQuiesce({ runId: "run-workflow-reentrant" });
    return {
      result: Promise.resolve({ stopReason: "aborted" }),
      cancel: () => { cancelled = true; },
      dispose: async () => {},
    };
  }, { runId: "run-workflow-reentrant" });

  assert.equal(pending.kind, "workflow");
  assert.equal(starts.size, 1);
  await draining;

  assert.equal(cancelled, true);
  assert.equal(starts.size, 0);
});

test("closed launch admission blocks one-shot, continuable and Workflow starters synchronously", () => {
  let externalStarts = 0;
  const launchFence = {
    assertRunOpen: () => {
      const error = new Error("provider stage admission is closed");
      error.code = "provider_stage_admission_closed";
      throw error;
    },
  };
  const starts = new PendingChildStarts({
    subagents: {
      start: () => { externalStarts += 1; },
      startContinuable: () => { externalStarts += 1; },
    },
  }, { launchFence });
  const rejected = (error) => error.code === "provider_stage_admission_closed";

  assert.throws(
    () => starts.start("one-shot", { provider: "spawn" }, { runId: "run-closed" }),
    rejected,
  );
  assert.throws(
    () => starts.start("continuable", {}, { runId: "run-closed" }),
    rejected,
  );
  assert.throws(
    () => starts.startWorkflow(() => { externalStarts += 1; }, { runId: "run-closed" }),
    rejected,
  );

  assert.equal(externalStarts, 0);
  assert.equal(starts.size, 0);
});
