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

function outcome(promise) {
  return Promise.resolve(promise).then(
    (value) => ({ status: "fulfilled", value }),
    (reason) => ({ status: "rejected", reason }),
  );
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

test("terminal cancel rejects a queued pause and resume without reopening admission", async () => {
  const drainEntered = deferred();
  const releaseDrain = deferred();
  let drains = 0;
  let opens = 0;
  const controller = new RuntimeController({}, {
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => { opens += 1; },
      quiesceRun: async () => {
        drains += 1;
        drainEntered.resolve();
        await releaseDrain.promise;
      },
    },
  });
  controller.registry.start(binding());
  controller.roleAgents = { quiesceRun: async () => {} };

  const cancelling = controller.cancel(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "terminal-cancel-1",
  }));
  await drainEntered.promise;
  const pausing = controller.pause(binding({
    run_state_revision: 9,
    ledger_expected_revision: 13,
    idempotency_key: "pause-after-terminal-cancel-1",
  }));
  const resuming = controller.resume(binding({
    run_state_revision: 10,
    ledger_expected_revision: 14,
    idempotency_key: "resume-after-terminal-cancel-1",
  }));
  releaseDrain.resolve();

  const results = await Promise.all([
    outcome(cancelling),
    outcome(pausing),
    outcome(resuming),
  ]);

  assert.deepEqual(results.map((item) => item.status), [
    "fulfilled",
    "rejected",
    "rejected",
  ]);
  assert.equal(controller.registry.get("run-1").status, "cancelled");
  assert.equal(drains, 1);
  assert.equal(opens, 0);
  await assert.rejects(
    controller.pause(binding({
      run_state_revision: 11,
      ledger_expected_revision: 15,
      idempotency_key: "pause-after-cancelled-1",
    })),
    (error) => error.code === "runtime_control_transition_invalid",
  );
});

test("queued terminal cancel still drains after a prior pause drain rejects", async () => {
  const firstDrainEntered = deferred();
  const releaseFirstDrain = deferred();
  let drains = 0;
  const controller = new RuntimeController({}, {
    stageRunner: {
      closeLaunchFence: () => {},
      quiesceRun: async () => {
        drains += 1;
        if (drains === 1) {
          firstDrainEntered.resolve();
          await releaseFirstDrain.promise;
          throw new Error("private pause drain failure");
        }
      },
    },
  });
  controller.registry.start(binding());
  controller.roleAgents = { quiesceRun: async () => {} };

  const pausing = controller.pause(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "failing-pause-1",
  }));
  await firstDrainEntered.promise;
  const cancelling = controller.cancel(binding({
    run_state_revision: 9,
    ledger_expected_revision: 13,
    idempotency_key: "cancel-after-failing-pause-1",
  }));
  releaseFirstDrain.resolve();

  const results = await Promise.all([outcome(pausing), outcome(cancelling)]);

  assert.deepEqual(results.map((item) => item.status), ["rejected", "fulfilled"]);
  assert.equal(drains, 2);
  assert.equal(controller.registry.get("run-1").status, "cancelled");
});

test("a cancel retry reruns a failed cancelling drain and reaches terminal state", async () => {
  let drains = 0;
  const controller = new RuntimeController({}, {
    stageRunner: {
      closeLaunchFence: () => {},
      quiesceRun: async () => {
        drains += 1;
        if (drains === 1) throw new Error("private first cancel drain failure");
      },
    },
  });
  controller.registry.start(binding());
  controller.roleAgents = { quiesceRun: async () => {} };
  const cancelBinding = binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "retryable-cancel-1",
  });

  await assert.rejects(controller.cancel(cancelBinding), /private first cancel drain failure/);
  assert.equal(controller.registry.get("run-1").status, "cancelling");

  const retried = await controller.cancel(cancelBinding);

  assert.equal(retried.accepted, true);
  assert.equal(drains, 2);
  assert.equal(controller.registry.get("run-1").status, "cancelled");
});

test("terminal cancel supersedes an already queued resume without an open window", async () => {
  const pauseDrainEntered = deferred();
  const releasePauseDrain = deferred();
  let drains = 0;
  let opens = 0;
  const controller = new RuntimeController({}, {
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => { opens += 1; },
      quiesceRun: async () => {
        drains += 1;
        if (drains === 1) {
          pauseDrainEntered.resolve();
          await releasePauseDrain.promise;
        }
      },
    },
  });
  controller.registry.start(binding());
  controller.roleAgents = { quiesceRun: async () => {} };

  const pausing = controller.pause(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "pause-before-resume-cancel-1",
  }));
  await pauseDrainEntered.promise;
  const resuming = controller.resume(binding({
    run_state_revision: 9,
    ledger_expected_revision: 13,
    idempotency_key: "queued-resume-1",
  }));
  const cancelling = controller.cancel(binding({
    run_state_revision: 10,
    ledger_expected_revision: 14,
    idempotency_key: "cancel-after-queued-resume-1",
  }));
  releasePauseDrain.resolve();

  const results = await Promise.all([
    outcome(pausing),
    outcome(resuming),
    outcome(cancelling),
  ]);

  assert.equal(results[0].status, "fulfilled");
  assert.equal(results[1].status, "rejected");
  assert.equal(results[2].status, "fulfilled");
  assert.equal(controller.registry.get("run-1").status, "cancelled");
  assert.equal(opens, 0);
  assert.equal(drains, 2);
});

for (const control of ["pause", "cancel"]) {
  test(`stale startRun completion after ${control} cannot reopen and is quiesced`, async () => {
    const createEntered = deferred();
    const releaseCreate = deferred();
    const calls = [];
    let creationSettled = false;
    const controller = new RuntimeController({}, {
      presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
      stageRunner: {
        closeLaunchFence: () => calls.push(["close"]),
        openLaunchFence: () => calls.push(["open"]),
        quiesceRun: async () => {},
      },
    });
    controller.roleAgents = {
      createRoleAgent: async () => {
        createEntered.resolve();
        await releaseCreate.promise;
        creationSettled = true;
        return { dispose: async () => {} };
      },
      quiesceRun: async (_runId, { dispose }) => {
        calls.push(["hosts", dispose, creationSettled]);
      },
    };
    const startBinding = binding({
      binding: {
        initial_run_status: "running",
        strategy_model_id: "provider/model",
        review_model_id: "provider/model",
      },
      idempotency_key: "start-with-control-race-1",
    });

    const starting = controller.startRun(startBinding);
    await createEntered.promise;
    const controlling = controller[control](binding({
      run_state_revision: 8,
      ledger_expected_revision: 12,
      idempotency_key: `${control}-during-start-1`,
    }));
    await controlling;
    releaseCreate.resolve();
    await starting;

    assert.equal(
      controller.registry.get("run-1").status,
      control === "cancel" ? "cancelled" : "paused",
    );
    assert.deepEqual(calls, [
      ["close"],
      ["hosts", control === "cancel", false],
      ["hosts", control === "cancel", true],
    ]);
    assert.equal(controller.liveReady, false);
  });
}

test("stale failing startRun preserves terminal cancellation and disposes late handles", async () => {
  const createEntered = deferred();
  const releaseCreate = deferred();
  const calls = [];
  let creationSettled = false;
  const controller = new RuntimeController({}, {
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => calls.push(["close"]),
      openLaunchFence: () => calls.push(["open"]),
      quiesceRun: async () => {},
    },
  });
  controller.roleAgents = {
    createRoleAgent: async () => {
      createEntered.resolve();
      try {
        await releaseCreate.promise;
      } finally {
        creationSettled = true;
      }
    },
    quiesceRun: async (_runId, { dispose }) => {
      calls.push(["hosts", dispose, creationSettled]);
    },
  };
  const starting = outcome(controller.startRun(binding({
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/model",
      review_model_id: "provider/model",
    },
    idempotency_key: "failing-start-with-cancel-race-1",
  })));
  await createEntered.promise;
  await controller.cancel(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "cancel-during-failing-start-1",
  }));
  releaseCreate.reject(new Error("private role creation failure"));

  const startOutcome = await starting;

  assert.equal(startOutcome.status, "rejected");
  assert.match(startOutcome.reason.message, /private role creation failure/);
  assert.equal(controller.registry.get("run-1").status, "cancelled");
  assert.deepEqual(calls, [
    ["close"],
    ["hosts", true, false],
    ["hosts", true, true],
  ]);
});

test("resume waits for stale startRun cleanup before reopening admission", async () => {
  const createEntered = deferred();
  const releaseCreate = deferred();
  const calls = [];
  let creationSettled = false;
  const controller = new RuntimeController({}, {
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => calls.push(["close"]),
      openLaunchFence: () => calls.push(["open"]),
      quiesceRun: async () => {},
    },
  });
  controller.roleAgents = {
    createRoleAgent: async () => {
      createEntered.resolve();
      await releaseCreate.promise;
      creationSettled = true;
      return { dispose: async () => {} };
    },
    quiesceRun: async (_runId, { dispose }) => {
      calls.push(["hosts", dispose, creationSettled]);
    },
  };
  const starting = controller.startRun(binding({
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/model",
      review_model_id: "provider/model",
    },
    idempotency_key: "start-before-pause-resume-1",
  }));
  await createEntered.promise;
  await controller.pause(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "pause-during-start-before-resume-1",
  }));
  let resumeSettled = false;
  const resuming = controller.resume(binding({
    run_state_revision: 9,
    ledger_expected_revision: 13,
    idempotency_key: "resume-waits-for-start-cleanup-1",
  })).finally(() => { resumeSettled = true; });
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(resumeSettled, false);
  assert.equal(calls.some(([name]) => name === "open"), false);

  releaseCreate.resolve();
  await Promise.all([starting, resuming]);

  assert.equal(controller.registry.get("run-1").status, "running");
  assert.deepEqual(calls, [
    ["close"],
    ["hosts", false, false],
    ["hosts", false, true],
    ["open"],
  ]);
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
