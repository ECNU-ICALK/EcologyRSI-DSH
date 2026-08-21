import assert from "node:assert/strict";
import test from "node:test";

import { RuntimeController } from "../lib/runtime/controller.js";

function binding(overrides = {}) {
  return {
    run_id: "run-1",
    run_state_revision: 7,
    stage_attempt: 2,
    ledger_expected_revision: 11,
    idempotency_key: "control-1",
    ...overrides,
  };
}

test("cancel closes runtime admission before child and role-host quiescence", async () => {
  const calls = [];
  const controller = new RuntimeController({}, {
    stageRunner: {
      quiesceRun: async (runId) => {
        calls.push(["children", runId, controller.registry.get(runId).status]);
      },
    },
  });
  controller.registry.start(binding());
  controller.roleAgents = {
    quiesceRun: async (runId, options) => {
      calls.push(["hosts", runId, options.dispose, controller.registry.get(runId).status]);
    },
  };

  await controller.cancel(binding());

  assert.deepEqual(calls, [
    ["children", "run-1", "cancelling"],
    ["hosts", "run-1", true, "cancelling"],
  ]);
  assert.equal(controller.registry.get("run-1").status, "cancelled");
});

test("pause quiesces but retains role-hosts for exact resume", async () => {
  const calls = [];
  const controller = new RuntimeController({}, {
    stageRunner: { quiesceRun: async () => calls.push("children") },
  });
  controller.registry.start(binding());
  controller.roleAgents = {
    quiesceRun: async (_runId, options) => calls.push(["hosts", options.dispose]),
  };
  await controller.pause(binding());
  assert.deepEqual(calls, ["children", ["hosts", false]]);
  assert.equal(controller.registry.get("run-1").status, "paused");
  await controller.resume(binding({ idempotency_key: "resume-1" }));
  assert.equal(controller.registry.get("run-1").status, "running");
});
