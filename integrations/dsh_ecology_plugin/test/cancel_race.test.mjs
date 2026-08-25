import assert from "node:assert/strict";
import test from "node:test";

import { RuntimeController } from "../lib/runtime/controller.js";
import { RuntimeRunRegistry } from "../lib/runtime/run-registry.js";

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

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

test("runtime creation freezes the Python-owned initial run status", () => {
  const registry = new RuntimeRunRegistry();
  registry.start(binding({ binding: { initial_run_status: "running" } }));
  assert.equal(registry.get("run-1").status, "running");
  assert.throws(
    () => new RuntimeRunRegistry().start(
      binding({ binding: { initial_run_status: "paused" } }),
    ),
    /invalid initial runtime run status/,
  );
});

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

test("resume waits for the pause drain before reopening launch admission", async () => {
  const drainEntered = deferred();
  const releaseDrain = deferred();
  const calls = [];
  const controller = new RuntimeController({}, {
    stageRunner: {
      closeLaunchFence: (runId) => calls.push(["close", runId]),
      quiesceRun: async (runId) => {
        calls.push(["drain", runId]);
        drainEntered.resolve();
        await releaseDrain.promise;
      },
      openLaunchFence: (runId) => calls.push(["open", runId]),
    },
  });
  controller.registry.start(binding());
  controller.roleAgents = { quiesceRun: async () => {} };

  const pausing = controller.pause(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "pause-drain-1",
  }));
  await drainEntered.promise;
  const resuming = controller.resume(binding({
    run_state_revision: 9,
    ledger_expected_revision: 13,
    idempotency_key: "resume-after-drain-1",
  }));
  let resumeSettled = false;
  void resuming.finally(() => { resumeSettled = true; });
  await Promise.resolve();

  assert.equal(resumeSettled, false);
  assert.deepEqual(calls, [["close", "run-1"], ["drain", "run-1"]]);

  releaseDrain.resolve();
  await pausing;
  await resuming;

  assert.deepEqual(calls, [
    ["close", "run-1"],
    ["drain", "run-1"],
    ["open", "run-1"],
  ]);
  assert.equal(controller.registry.get("run-1").status, "running");
});

test("cancelled runs cannot reopen their launch fence", async () => {
  const calls = [];
  const controller = new RuntimeController({}, {
    stageRunner: {
      closeLaunchFence: (runId) => calls.push(["close", runId]),
      quiesceRun: async () => {},
      openLaunchFence: (runId) => calls.push(["open", runId]),
    },
  });
  controller.registry.start(binding());
  controller.roleAgents = { quiesceRun: async () => {} };

  await controller.cancel(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "cancel-terminal-1",
  }));
  await assert.rejects(
    controller.resume(binding({
      run_state_revision: 9,
      ledger_expected_revision: 13,
      idempotency_key: "resume-cancelled-1",
    })),
    /cancelled runtime run cannot resume/,
  );

  assert.deepEqual(calls, [["close", "run-1"]]);
  assert.equal(controller.registry.get("run-1").status, "cancelled");
});

test("cancel queued during pause drain cannot be overwritten by the older control", async () => {
  const drainEntered = [deferred(), deferred()];
  const releaseDrain = [deferred(), deferred()];
  let drainsStarted = 0;
  const controller = new RuntimeController({}, {
    stageRunner: {
      closeLaunchFence: () => {},
      quiesceRun: async () => {
        const index = drainsStarted;
        drainsStarted += 1;
        drainEntered[index].resolve();
        await releaseDrain[index].promise;
      },
    },
  });
  controller.registry.start(binding());
  controller.roleAgents = { quiesceRun: async () => {} };

  const pausing = controller.pause(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "pause-before-cancel-1",
  }));
  await drainEntered[0].promise;
  const cancelling = controller.cancel(binding({
    run_state_revision: 9,
    ledger_expected_revision: 13,
    idempotency_key: "cancel-after-pause-1",
  }));
  await new Promise((resolve) => setImmediate(resolve));
  const drainsBeforePauseRelease = drainsStarted;

  releaseDrain[0].resolve();
  await pausing;
  await drainEntered[1].promise;
  releaseDrain[1].resolve();
  await cancelling;

  assert.equal(drainsBeforePauseRelease, 1);
  assert.equal(controller.registry.get("run-1").status, "cancelled");
});

test("late concurrent stage completion cannot reopen a paused run", async () => {
  let releaseStage;
  let stageStarted;
  const started = new Promise((resolve) => { stageStarted = resolve; });
  const stageResult = new Promise((resolve) => { releaseStage = resolve; });
  const controller = new RuntimeController({}, {
    stageRunner: {
      run: async () => {
        stageStarted();
        return stageResult;
      },
      quiesceRun: async () => {},
    },
  });
  controller.registry.start(binding());
  controller.roleAgents = { quiesceRun: async () => {} };

  const inFlight = controller.runStage(binding({idempotency_key: "stage-1"}));
  await started;
  await controller.pause(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "pause-1",
  }));
  assert.equal(controller.registry.get("run-1").status, "paused");

  releaseStage({
    structured: {accepted: true},
    result_digest: "a".repeat(64),
    session_id: "child-1",
    skill_invocation_evidence: {
      first_tool_call_verified: true,
      order_verified: true,
    },
  });
  const completed = await inFlight;

  assert.deepEqual(
    {
      status: controller.registry.get("run-1").status,
      run_state_revision: completed.run_state_revision,
      ledger_expected_revision: completed.ledger_expected_revision,
      idempotency_key: completed.idempotency_key,
    },
    {
      status: "paused",
      run_state_revision: 8,
      ledger_expected_revision: 12,
      idempotency_key: "pause-1",
    },
  );
});

test("late stage completion cannot replace a newer resumed receipt", () => {
  const registry = new RuntimeRunRegistry();
  registry.start(binding({ idempotency_key: "create-1" }));
  registry.transition("run-1", binding({
    run_state_revision: 9,
    ledger_expected_revision: 13,
    idempotency_key: "resume-1",
  }), "running");

  const accepted = registry.refresh(binding({
    run_state_revision: 7,
    ledger_expected_revision: 11,
    idempotency_key: "old-stage-1",
  }));

  assert.equal(accepted.status, "running");
  assert.equal(accepted.run_state_revision, 9);
  assert.equal(accepted.ledger_expected_revision, 13);
  assert.equal(accepted.idempotency_key, "resume-1");
});
