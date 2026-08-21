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
