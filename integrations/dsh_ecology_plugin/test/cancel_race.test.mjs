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

async function startReadyRun(controller, startBinding = binding()) {
  const stageRunner = controller.stageRunner;
  const presetCatalog = controller.presetCatalog;
  const roleAgents = controller.roleAgents;
  controller.stageRunner = null;
  controller.presetCatalog = [
    { preset_id: "ecology-researcher-v7", tool_profile: "test" },
  ];
  controller.roleAgents = {
    createRoleAgent: async () => ({ dispose: async () => {} }),
    quiesceRun: async () => {},
  };
  try {
    await controller.startRun(startBinding);
  } finally {
    controller.stageRunner = stageRunner;
    controller.presetCatalog = presetCatalog;
    controller.roleAgents = roleAgents;
  }
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
  const restored = new RuntimeRunRegistry();
  restored.start(binding({
    idempotency_key: "runtime-restore:run-1",
    binding: {
      initial_run_status: "paused",
      restore_provenance: {
        source: "python_durable_ledger",
        status: "paused",
      },
    },
  }));
  assert.equal(restored.get("run-1").status, "paused");
  assert.throws(
    () => new RuntimeRunRegistry().start(binding({
      binding: {
        initial_run_status: "paused",
        restore_provenance: {
          source: "python_durable_ledger",
          status: "running",
        },
      },
    })),
    /invalid initial runtime run status/,
  );
});

test("role hosts do not require an unused Workflow service", async () => {
  const created = [];
  const controller = new RuntimeController({}, {
    presetCatalog: [
      { preset_id: "ecology-coordinator-v4", tool_profile: "test" },
      { preset_id: "ecology-sample-planner-v5", tool_profile: "test" },
    ],
  });
  controller.roleAgents = {
    createRoleAgent: async (roleBinding) => {
      created.push(roleBinding);
      return { dispose: async () => {} };
    },
    quiesceRun: async () => {},
  };
  await controller.startRun(binding({
    idempotency_key: "direct-sample-planner-role-host",
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/strategy",
      review_model_id: "provider/review",
    },
  }));

  assert.equal(created.every((item) => !("require_workflow" in item)), true);
});

test("registry exact start replay returns the current record and preserves its generation", () => {
  const registry = new RuntimeRunRegistry();
  const startBinding = binding({
    idempotency_key: "generation-replay-1",
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/model-a",
      review_model_id: "provider/model-b",
    },
  });
  const started = registry.start(startBinding);
  const generation = registry.generationOf(started);
  const transitioned = registry.transition("run-1", binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "pause-generation-replay-1",
  }), "running");
  const refreshed = registry.refresh(binding({
    run_state_revision: 9,
    ledger_expected_revision: 13,
    idempotency_key: "stage-generation-replay-1",
  }));
  const replayed = registry.start(structuredClone(startBinding));

  assert.ok(generation);
  assert.strictEqual(registry.generationOf(transitioned), generation);
  assert.strictEqual(registry.generationOf(refreshed), generation);
  assert.strictEqual(replayed, refreshed);
  assert.strictEqual(registry.generationOf(replayed), generation);
  assert.equal(replayed.run_state_revision, 9);
  assert.equal(replayed.idempotency_key, "stage-generation-replay-1");
});

test("registry rejects a changed full start payload under the same idempotency key", () => {
  const registry = new RuntimeRunRegistry();
  const startBinding = binding({
    idempotency_key: "runtime-restore:run-1",
    binding: {
      initial_run_status: "paused",
      restore_provenance: {
        source: "python_durable_ledger",
        status: "paused",
      },
      strategy_model_id: "provider/model-a",
      review_model_id: "provider/model-b",
    },
  });
  const started = registry.start(startBinding);

  assert.throws(
    () => registry.start({
      ...structuredClone(startBinding),
      binding: {
        ...structuredClone(startBinding.binding),
        strategy_model_id: "provider/changed-model",
      },
    }),
    (error) => error?.code === "runtime_start_conflict",
  );
  assert.strictEqual(registry.get("run-1"), started);
  assert.equal(registry.get("run-1").status, "paused");
});

test("a genuine replacement receives a new registry-owned generation", () => {
  const registry = new RuntimeRunRegistry();
  const startBinding = binding({
    idempotency_key: "generation-replacement-1",
    caller_generation: { forged: true },
    binding: { initial_run_status: "running" },
  });
  const first = registry.start(startBinding);
  const firstGeneration = registry.generationOf(first);
  registry.delete("run-1");
  const replacementBinding = structuredClone(startBinding);
  replacementBinding.caller_generation = firstGeneration;
  const replacement = registry.start(replacementBinding);

  assert.ok(firstGeneration);
  assert.ok(registry.generationOf(replacement));
  assert.notStrictEqual(registry.generationOf(replacement), firstGeneration);
});

test("ordinary created runs cannot enter the restored-paused resume path", async () => {
  let opens = 0;
  const controller = new RuntimeController({}, {
    stageRunner: { openLaunchFence: () => { opens += 1; } },
  });
  controller.registry.start(binding({
    binding: { initial_run_status: "created" },
  }));

  await assert.rejects(
    controller.resume(binding({ idempotency_key: "ordinary-created-resume-1" })),
    /cannot resume from created/,
  );
  assert.equal(controller.registry.get("run-1").status, "created");
  assert.equal(opens, 0);
});

test("explicit activation starts a ready created run without weakening resume", async () => {
  let opens = 0;
  const controller = new RuntimeController({}, {
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => { opens += 1; },
    },
  });
  await startReadyRun(controller, binding({
    idempotency_key: "created-run-1",
    binding: { initial_run_status: "created" },
  }));

  const activated = await controller.activate(binding({
    idempotency_key: "activate-created-1",
  }));

  assert.equal(activated.accepted, true);
  assert.equal(controller.registry.get("run-1").status, "running");
  assert.equal(opens, 1);
});

test("a second controller cannot resume registry-only paused hosts", async () => {
  const registry = new RuntimeRunRegistry();
  const ownerCalls = [];
  const owner = new RuntimeController({}, {
    registry,
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => ownerCalls.push("close"),
      openLaunchFence: () => ownerCalls.push("open"),
    },
  });
  owner.roleAgents = {
    createRoleAgent: async () => ({ dispose: async () => {} }),
    quiesceRun: async () => {},
  };
  const restored = binding({
    idempotency_key: "runtime-restore:run-1",
    binding: {
      initial_run_status: "paused",
      restore_provenance: {
        source: "python_durable_ledger",
        status: "paused",
      },
    },
  });
  await owner.startRun(restored);
  assert.equal(owner.liveReady, true);
  assert.deepEqual(ownerCalls, ["close"]);

  const externalCalls = [];
  const external = new RuntimeController({}, {
    registry,
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => externalCalls.push("close"),
      openLaunchFence: () => externalCalls.push("open"),
    },
  });
  external.roleAgents = { quiesceRun: async () => {} };

  const resumed = await outcome(external.resume(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "external-resume-without-hosts-1",
  })));

  assert.equal(resumed.status, "rejected");
  assert.equal(resumed.reason.code, "runtime_role_hosts_incomplete");
  assert.equal(registry.get("run-1").status, "paused");
  assert.equal(external.liveReady, false);
  assert.equal(externalCalls.includes("open"), false);
  assert.ok(externalCalls.includes("close"));

  external.runLifecycles.get("run-1").hosts = "creating";
  const creatingWithoutStart = await outcome(external.resume(binding({
    run_state_revision: 9,
    ledger_expected_revision: 13,
    idempotency_key: "external-resume-creating-without-start-1",
  })));
  assert.equal(creatingWithoutStart.status, "rejected");
  assert.equal(creatingWithoutStart.reason.code, "runtime_role_hosts_incomplete");
  assert.equal(registry.get("run-1").status, "paused");
  assert.equal(externalCalls.includes("open"), false);
});

test("an exact shared-registry start replay does not duplicate or claim role hosts", async () => {
  const registry = new RuntimeRunRegistry();
  const restored = binding({
    idempotency_key: "runtime-restore:run-1",
    binding: {
      initial_run_status: "paused",
      restore_provenance: {
        source: "python_durable_ledger",
        status: "paused",
      },
      strategy_model_id: "provider/model-a",
      review_model_id: "provider/model-b",
    },
  });
  const owner = new RuntimeController({}, {
    registry,
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => assert.fail("restored paused owner must not open"),
    },
  });
  owner.roleAgents = {
    createRoleAgent: async () => ({ dispose: async () => {} }),
    quiesceRun: async () => {},
  };
  await owner.startRun(restored);
  const authoritative = registry.transition("run-1", binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "pause-after-replayed-start-1",
  }), "paused");
  const generation = registry.generationOf(authoritative);
  let replayCreates = 0;
  let replayOpens = 0;
  const replay = new RuntimeController({}, {
    registry,
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => { replayOpens += 1; },
    },
  });
  replay.roleAgents = {
    createRoleAgent: async () => {
      replayCreates += 1;
      return { dispose: async () => {} };
    },
    quiesceRun: async () => {},
  };

  const result = await replay.startRun(structuredClone(restored));

  assert.equal(result.accepted, true);
  assert.equal(result.idempotency_key, restored.idempotency_key);
  assert.strictEqual(registry.get("run-1"), authoritative);
  assert.equal(registry.get("run-1").idempotency_key, "pause-after-replayed-start-1");
  assert.strictEqual(registry.generationOf(registry.get("run-1")), generation);
  assert.equal(replayCreates, 0);
  assert.equal(replayOpens, 0);
  assert.equal(replay.liveReady, false);
  assert.equal(owner.liveReady, true);
});

test("a changed same-key start on another controller conflicts without replacing paused hosts", async () => {
  const registry = new RuntimeRunRegistry();
  const restored = binding({
    idempotency_key: "runtime-restore:run-1",
    binding: {
      initial_run_status: "paused",
      restore_provenance: {
        source: "python_durable_ledger",
        status: "paused",
      },
      strategy_model_id: "provider/model-a",
      review_model_id: "provider/model-b",
    },
  });
  const owner = new RuntimeController({}, {
    registry,
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: { closeLaunchFence: () => {}, openLaunchFence: () => {} },
  });
  owner.roleAgents = {
    createRoleAgent: async () => ({ dispose: async () => {} }),
    quiesceRun: async () => {},
  };
  await owner.startRun(restored);
  const authoritative = registry.get("run-1");
  let replacementCreates = 0;
  let replacementOpens = 0;
  const replacement = new RuntimeController({}, {
    registry,
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => { replacementOpens += 1; },
    },
  });
  replacement.roleAgents = {
    createRoleAgent: async () => {
      replacementCreates += 1;
      return { dispose: async () => {} };
    },
    quiesceRun: async () => {},
  };
  const changed = structuredClone(restored);
  changed.binding.strategy_model_id = "provider/changed-model";

  const result = await outcome(replacement.startRun(changed));

  assert.equal(result.status, "rejected");
  assert.equal(result.reason.code, "runtime_start_conflict");
  assert.strictEqual(registry.get("run-1"), authoritative);
  assert.equal(registry.get("run-1").status, "paused");
  assert.equal(replacementCreates, 0);
  assert.equal(replacementOpens, 0);
  assert.equal(owner.liveReady, true);
  assert.equal(replacement.liveReady, false);
});

test("stale ready hosts cannot resume while a new paused generation is creating", async () => {
  const registry = new RuntimeRunRegistry();
  const ownerCalls = [];
  const restored = binding({
    idempotency_key: "runtime-restore:run-1",
    binding: {
      initial_run_status: "paused",
      restore_provenance: {
        source: "python_durable_ledger",
        status: "paused",
      },
      strategy_model_id: "provider/model-a",
      review_model_id: "provider/model-b",
    },
  });
  const owner = new RuntimeController({}, {
    registry,
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => ownerCalls.push("close"),
      openLaunchFence: () => ownerCalls.push("open"),
    },
  });
  owner.roleAgents = {
    createRoleAgent: async () => ({ dispose: async () => {} }),
    quiesceRun: async () => {},
  };
  await owner.startRun(restored);
  const oldGeneration = registry.generationOf(registry.get("run-1"));
  ownerCalls.length = 0;
  registry.delete("run-1");

  const createEntered = deferred();
  const releaseCreate = deferred();
  const replacementCalls = [];
  const replacement = new RuntimeController({}, {
    registry,
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => replacementCalls.push("close"),
      openLaunchFence: () => replacementCalls.push("open"),
    },
  });
  replacement.roleAgents = {
    createRoleAgent: async () => {
      createEntered.resolve();
      await releaseCreate.promise;
      return { dispose: async () => {} };
    },
    quiesceRun: async () => {},
  };
  const replacementBinding = structuredClone(restored);
  replacementBinding.binding.strategy_model_id = "provider/replacement-model";
  const startingReplacement = outcome(replacement.startRun(replacementBinding));
  await createEntered.promise;
  assert.notStrictEqual(
    registry.generationOf(registry.get("run-1")),
    oldGeneration,
  );

  const resumed = await outcome(owner.resume(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "stale-owner-resume-1",
  })));

  assert.equal(resumed.status, "rejected");
  assert.equal(resumed.reason.code, "runtime_role_hosts_incomplete");
  assert.equal(registry.get("run-1").status, "paused");
  assert.equal(owner.liveReady, false);
  assert.equal(ownerCalls.includes("open"), false);
  assert.ok(ownerCalls.includes("close"));
  assert.equal(replacementCalls.includes("open"), false);

  releaseCreate.resolve();
  const replacementResult = await startingReplacement;
  assert.equal(replacementResult.status, "fulfilled");
  assert.equal(registry.get("run-1").status, "paused");
  assert.equal(replacement.liveReady, true);
  assert.equal(owner.liveReady, false);
  assert.equal(ownerCalls.includes("open"), false);
  assert.equal(replacementCalls.includes("open"), false);
});

test("stale ready hosts cannot resume across a replacement start failure", async () => {
  const registry = new RuntimeRunRegistry();
  let ownerOpens = 0;
  const restored = binding({
    idempotency_key: "runtime-restore:run-1",
    binding: {
      initial_run_status: "paused",
      restore_provenance: {
        source: "python_durable_ledger",
        status: "paused",
      },
      strategy_model_id: "provider/model-a",
      review_model_id: "provider/model-b",
    },
  });
  const owner = new RuntimeController({}, {
    registry,
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => { ownerOpens += 1; },
    },
  });
  owner.roleAgents = {
    createRoleAgent: async () => ({ dispose: async () => {} }),
    quiesceRun: async () => {},
  };
  await owner.startRun(restored);
  ownerOpens = 0;
  registry.delete("run-1");

  const createEntered = deferred();
  const releaseCreate = deferred();
  let replacementOpens = 0;
  const replacement = new RuntimeController({}, {
    registry,
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => { replacementOpens += 1; },
    },
  });
  replacement.roleAgents = {
    createRoleAgent: async () => {
      createEntered.resolve();
      await releaseCreate.promise;
      throw new Error("replacement host creation failed");
    },
    quiesceRun: async () => {},
  };
  const replacementBinding = structuredClone(restored);
  replacementBinding.binding.review_model_id = "provider/replacement-review";
  const startingReplacement = outcome(replacement.startRun(replacementBinding));
  await createEntered.promise;

  const resumed = await outcome(owner.resume(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "stale-owner-failed-replacement-resume-1",
  })));
  assert.equal(resumed.status, "rejected");
  assert.equal(resumed.reason.code, "runtime_role_hosts_incomplete");
  assert.equal(registry.get("run-1").status, "paused");

  releaseCreate.resolve();
  const replacementResult = await startingReplacement;
  assert.equal(replacementResult.status, "rejected");
  assert.match(replacementResult.reason.message, /replacement host creation failed/);
  assert.equal(registry.get("run-1"), null);
  assert.equal(ownerOpens, 0);
  assert.equal(replacementOpens, 0);
  assert.equal(owner.liveReady, false);
  assert.equal(replacement.liveReady, false);
});

test("a start superseded by a new generation cleans its hosts and never opens admission", async () => {
  const registry = new RuntimeRunRegistry();
  const createEntered = deferred();
  const releaseCreate = deferred();
  const calls = [];
  const controller = new RuntimeController({}, {
    registry,
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => calls.push(["close"]),
      openLaunchFence: () => calls.push(["open"]),
    },
  });
  controller.roleAgents = {
    createRoleAgent: async () => {
      createEntered.resolve();
      await releaseCreate.promise;
      return { dispose: async () => {} };
    },
    quiesceRun: async (_runId, { dispose }) => calls.push(["hosts", dispose]),
  };
  const oldBinding = binding({
    idempotency_key: "superseded-start-old-1",
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/old-model",
      review_model_id: "provider/old-review",
    },
  });
  const starting = outcome(controller.startRun(oldBinding));
  await createEntered.promise;
  const oldGeneration = registry.generationOf(registry.get("run-1"));
  registry.delete("run-1");
  const replacementBinding = binding({
    idempotency_key: "superseded-start-new-1",
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/new-model",
      review_model_id: "provider/new-review",
    },
  });
  const replacement = registry.start(replacementBinding);
  assert.notStrictEqual(registry.generationOf(replacement), oldGeneration);
  const localReplayBeforeCleanup = await outcome(
    controller.startRun(structuredClone(replacementBinding)),
  );

  releaseCreate.resolve();
  const result = await starting;

  assert.equal(result.status, "fulfilled");
  assert.equal(localReplayBeforeCleanup.status, "rejected");
  assert.equal(localReplayBeforeCleanup.reason.code, "runtime_start_conflict");
  assert.strictEqual(registry.get("run-1"), replacement);
  assert.deepEqual(calls.filter(([name]) => name === "hosts"), [["hosts", true]]);
  assert.equal(calls.some(([name]) => name === "open"), false);
  assert.equal(controller.liveReady, false);
});

test("stale ready host generation rejects stage admission before the runner", async () => {
  const registry = new RuntimeRunRegistry();
  const calls = [];
  let stageRuns = 0;
  const controller = new RuntimeController({}, {
    registry,
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => calls.push("close"),
      openLaunchFence: () => calls.push("open"),
      run: async () => {
        stageRuns += 1;
        return {
          structured: { accepted: true },
          result_digest: "a".repeat(64),
          session_id: "must-not-run",
          skill_invocation_evidence: {
            first_tool_call_verified: true,
            order_verified: true,
          },
        };
      },
    },
  });
  controller.roleAgents = {
    createRoleAgent: async () => ({ dispose: async () => {} }),
    quiesceRun: async () => {},
  };
  await controller.startRun(binding({
    idempotency_key: "admission-old-start-1",
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/old-model",
      review_model_id: "provider/old-review",
    },
  }));
  calls.length = 0;
  registry.delete("run-1");
  registry.start(binding({
    idempotency_key: "admission-new-start-1",
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/new-model",
      review_model_id: "provider/new-review",
    },
  }));

  const result = await outcome(controller.runStage(binding({
    idempotency_key: "stale-admission-stage-1",
  })));

  assert.equal(result.status, "rejected");
  assert.equal(result.reason.code, "runtime_role_hosts_incomplete");
  assert.equal(stageRuns, 0);
  assert.deepEqual(calls, ["close"]);
  assert.equal(controller.liveReady, false);
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
  await startReadyRun(controller);
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
  await startReadyRun(controller);
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

test("an exact pause retry reruns a rejected drain and reaches paused", async () => {
  let drains = 0;
  const controller = new RuntimeController({}, {
    stageRunner: {
      closeLaunchFence: () => {},
      quiesceRun: async () => {
        drains += 1;
        if (drains === 1) throw new Error("private first pause drain failure");
      },
    },
  });
  controller.registry.start(binding());
  controller.roleAgents = { quiesceRun: async () => {} };
  const pauseBinding = binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "retryable-pause-1",
  });

  await assert.rejects(controller.pause(pauseBinding), /private first pause drain failure/);
  assert.equal(controller.registry.get("run-1").status, "pausing");

  const retried = await controller.pause(pauseBinding);

  assert.equal(retried.accepted, true);
  assert.equal(drains, 2);
  assert.equal(controller.registry.get("run-1").status, "paused");
});

test("resume followed synchronously by pause serializes both controls and stays fenced", async () => {
  let drains = 0;
  let opens = 0;
  const controller = new RuntimeController({}, {
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => { opens += 1; },
      quiesceRun: async () => { drains += 1; },
    },
  });
  await startReadyRun(controller);
  controller.roleAgents = { quiesceRun: async () => {} };
  await controller.pause(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "initial-pause-1",
  }));
  const resumeBinding = binding({
    run_state_revision: 9,
    ledger_expected_revision: 13,
    idempotency_key: "resume-before-repause-1",
  });
  const repauseBinding = binding({
    run_state_revision: 10,
    ledger_expected_revision: 14,
    idempotency_key: "pause-after-resume-1",
  });

  const resuming = controller.resume(resumeBinding);
  assert.equal(controller.registry.get("run-1").status, "resuming");
  const repausing = controller.pause(repauseBinding);
  const results = await Promise.all([outcome(resuming), outcome(repausing)]);

  assert.deepEqual(results.map((item) => item.status), ["fulfilled", "fulfilled"]);
  assert.equal(drains, 2);
  assert.equal(opens, 0);
  assert.equal(controller.registry.get("run-1").status, "paused");
  assert.equal(controller.registry.get("run-1").idempotency_key, "pause-after-resume-1");
});

test("concurrent identical resumes join one queued transition", async () => {
  let opens = 0;
  const controller = new RuntimeController({}, {
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => { opens += 1; },
      quiesceRun: async () => {},
    },
  });
  await startReadyRun(controller);
  controller.roleAgents = { quiesceRun: async () => {} };
  await controller.pause(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "pause-before-identical-resumes-1",
  }));
  const resumeBinding = binding({
    run_state_revision: 9,
    ledger_expected_revision: 13,
    idempotency_key: "identical-resume-1",
  });

  const first = controller.resume(resumeBinding);
  assert.equal(controller.registry.get("run-1").status, "resuming");
  const second = controller.resume({ ...resumeBinding });
  assert.strictEqual(second, first);
  const results = await Promise.all([outcome(first), outcome(second)]);

  assert.deepEqual(results.map((item) => item.status), ["fulfilled", "fulfilled"]);
  assert.deepEqual(results.map((item) => item.value.idempotency_key), [
    "identical-resume-1",
    "identical-resume-1",
  ]);
  assert.equal(opens, 1);
  assert.equal(controller.registry.get("run-1").status, "running");
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

test("start failure preserves its primary error while cleanup closes and deletes the run", async () => {
  const calls = [];
  const ctx = {
    agents: {
      create: async (options) => {
        const role = options.meta.ecologyRole;
        calls.push(["create", role]);
        if (role === "researcher") throw new Error("primary researcher creation failure");
        return {
          agent: {
            session: {
              append: async () => {},
              flush: async () => {},
            },
            waitForIdle: async () => {},
          },
          dispose: async () => {
            calls.push(["dispose", role]);
            throw new Error("private cleanup failure");
          },
        };
      },
    },
    agentPresets: {
      standingKeyFor: async (presetId) => `standing:${presetId}`,
      mount: async (_agentCtx, presetId) => ({ id: presetId }),
      serviceFor: async () => ({ ready: true }),
    },
  };
  const controller = new RuntimeController(ctx, {
    presetCatalog: [
      { preset_id: "ecology-researcher-v7", tool_profile: "test" },
      { preset_id: "ecology-candidate-proposer-v4", tool_profile: "test" },
    ],
    stageRunner: {
      closeLaunchFence: () => calls.push(["close"]),
      openLaunchFence: () => calls.push(["open"]),
    },
  });

  const result = await outcome(controller.startRun(binding({
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/model",
      review_model_id: "provider/model",
    },
    idempotency_key: "atomic-start-failure-1",
  })));

  assert.equal(result.status, "rejected");
  assert.match(result.reason.message, /primary researcher creation failure/);
  assert.equal(controller.registry.get("run-1"), null);
  assert.equal(calls.filter(([name]) => name === "dispose").length, 1);
  assert.equal(calls.filter(([name]) => name === "open").length, 0);
  assert.ok(calls.filter(([name]) => name === "close").length >= 1);
});

test("failed restored-paused start is deleted when no control supersedes it", async () => {
  let opens = 0;
  const controller = new RuntimeController({}, {
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => { opens += 1; },
    },
  });
  controller.roleAgents = {
    createRoleAgent: async () => {
      throw new Error("restored role-host creation failure");
    },
    quiesceRun: async () => {},
  };

  const result = await outcome(controller.startRun(binding({
    idempotency_key: "runtime-restore:run-1",
    binding: {
      initial_run_status: "paused",
      restore_provenance: {
        source: "python_durable_ledger",
        status: "paused",
      },
    },
  })));

  assert.equal(result.status, "rejected");
  assert.match(result.reason.message, /restored role-host creation failure/);
  assert.equal(controller.registry.get("run-1"), null);
  assert.equal(opens, 0);
});

test("pause joins a failing start cleanup and failed hosts can never resume", async () => {
  const createEntered = deferred();
  const releaseCreate = deferred();
  let opens = 0;
  const ctx = {
    agents: {
      create: async () => {
        createEntered.resolve();
        return await releaseCreate.promise;
      },
    },
    agentPresets: {
      standingKeyFor: async (presetId) => `standing:${presetId}`,
      mount: async (_agentCtx, presetId) => ({ id: presetId }),
      serviceFor: async () => ({ ready: true }),
    },
  };
  const controller = new RuntimeController(ctx, {
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => { opens += 1; },
      quiesceRun: async () => {},
    },
  });
  const starting = outcome(controller.startRun(binding({
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/model",
      review_model_id: "provider/model",
    },
    idempotency_key: "pause-start-failure-1",
  })));
  await createEntered.promise;

  let pauseSettled = false;
  const pausing = outcome(controller.pause(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "pause-during-start-failure-1",
  }))).finally(() => { pauseSettled = true; });
  await new Promise((resolve) => setImmediate(resolve));
  const pauseSettledBeforeCreate = pauseSettled;
  releaseCreate.reject(new Error("primary late creation failure"));

  const [startResult, pauseResult] = await Promise.all([starting, pausing]);
  const resumeResult = await outcome(controller.resume(binding({
    run_state_revision: 9,
    ledger_expected_revision: 13,
    idempotency_key: "resume-after-start-failure-1",
  })));

  assert.equal(pauseSettledBeforeCreate, false);
  assert.equal(startResult.status, "rejected");
  assert.match(startResult.reason.message, /primary late creation failure/);
  assert.equal(pauseResult.status, "fulfilled");
  assert.equal(controller.registry.get("run-1").status, "paused");
  assert.equal(resumeResult.status, "rejected");
  assert.equal(resumeResult.reason.code, "runtime_role_hosts_incomplete");
  assert.equal(opens, 0);
});

test("queued resume rejected by a failing start leaves the completed pause durable", async () => {
  const createEntered = deferred();
  const releaseCreate = deferred();
  let opens = 0;
  const controller = new RuntimeController({}, {
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => { opens += 1; },
      quiesceRun: async () => {},
    },
  });
  controller.roleAgents = {
    createRoleAgent: async () => {
      createEntered.resolve();
      return await releaseCreate.promise;
    },
    quiesceRun: async () => {},
  };

  const starting = outcome(controller.startRun(binding({
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/model",
      review_model_id: "provider/model",
    },
    idempotency_key: "queued-resume-failing-start-1",
  })));
  await createEntered.promise;
  const pausing = outcome(controller.pause(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "pause-before-failing-start-resume-1",
  })));
  const resuming = outcome(controller.resume(binding({
    run_state_revision: 9,
    ledger_expected_revision: 13,
    idempotency_key: "resume-queued-before-start-failure-1",
  })));

  releaseCreate.reject(new Error("late role-host creation failure"));
  const [startResult, pauseResult, resumeResult] = await Promise.all([
    starting,
    pausing,
    resuming,
  ]);

  assert.equal(startResult.status, "rejected");
  assert.equal(pauseResult.status, "fulfilled");
  assert.equal(resumeResult.status, "rejected");
  assert.equal(resumeResult.reason.code, "runtime_role_hosts_incomplete");
  assert.equal(controller.registry.get("run-1").status, "paused");
  assert.equal(opens, 0);
});

test("identical starts are one flight and resume waits for every stale cleanup", async () => {
  const createEntered = deferred();
  const releaseCreate = deferred();
  const cleanupEntered = deferred();
  const releaseCleanup = deferred();
  let createCalls = 0;
  let cleanupCalls = 0;
  let opens = 0;
  const controller = new RuntimeController({}, {
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => { opens += 1; },
      quiesceRun: async () => {},
    },
  });
  controller.roleAgents = {
    createRoleAgent: async () => {
      createCalls += 1;
      createEntered.resolve();
      await releaseCreate.promise;
      return { dispose: async () => {} };
    },
    quiesceRun: async () => {
      cleanupCalls += 1;
      cleanupEntered.resolve();
      await releaseCleanup.promise;
    },
  };
  const startBinding = binding({
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/model",
      review_model_id: "provider/model",
    },
    idempotency_key: "concurrent-start-1",
  });

  const firstStart = controller.startRun(startBinding);
  const secondStart = controller.startRun({
    ...startBinding,
    binding: { ...startBinding.binding },
  });
  assert.strictEqual(secondStart, firstStart);
  await createEntered.promise;
  const pausing = controller.pause(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "pause-two-starts-1",
  }));
  const resuming = controller.resume(binding({
    run_state_revision: 9,
    ledger_expected_revision: 13,
    idempotency_key: "resume-two-starts-1",
  }));
  await new Promise((resolve) => setImmediate(resolve));
  const cleanupCallsBeforeCreateRelease = cleanupCalls;

  releaseCreate.resolve();
  await cleanupEntered.promise;
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(opens, 0);
  releaseCleanup.resolve();

  const results = await Promise.all([
    outcome(firstStart),
    outcome(secondStart),
    outcome(pausing),
    outcome(resuming),
  ]);

  assert.deepEqual(results.map((item) => item.status), [
    "fulfilled",
    "fulfilled",
    "fulfilled",
    "fulfilled",
  ]);
  assert.equal(createCalls, 1);
  assert.equal(cleanupCallsBeforeCreateRelease, 0);
  assert.equal(cleanupCalls, 2);
  assert.equal(opens, 1);
  assert.equal(controller.registry.get("run-1").status, "running");
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
    let controlSettled = false;
    void controlling.finally(() => { controlSettled = true; });
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(controlSettled, false);
    releaseCreate.resolve();
    await Promise.all([starting, controlling]);

    assert.equal(
      controller.registry.get("run-1").status,
      control === "cancel" ? "cancelled" : "paused",
    );
    assert.equal(calls.filter(([name]) => name === "close").length, 2);
    assert.equal(calls.some(([name]) => name === "open"), false);
    assert.deepEqual(calls.filter(([name]) => name === "hosts"), [
      ["hosts", control === "cancel", true],
      ["hosts", control === "cancel", true],
    ]);
    assert.equal(controller.liveReady, control === "pause");
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
  const cancelling = controller.cancel(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: "cancel-during-failing-start-1",
  }));
  releaseCreate.reject(new Error("private role creation failure"));

  const [startOutcome] = await Promise.all([starting, cancelling]);

  assert.equal(startOutcome.status, "rejected");
  assert.match(startOutcome.reason.message, /private role creation failure/);
  assert.equal(controller.registry.get("run-1").status, "cancelled");
  assert.ok(calls.filter(([name]) => name === "close").length >= 2);
  assert.equal(calls.some(([name]) => name === "open"), false);
  assert.deepEqual(calls.filter(([name]) => name === "hosts"), [
    ["hosts", true, true],
    ["hosts", true, true],
  ]);
});

for (const terminalStatus of ["cancelling", "cancelled"]) {
  test(`registry ${terminalStatus} tombstones reject an exact same-key start`, () => {
    const registry = new RuntimeRunRegistry();
    const exactKey = "shared-start-cancel-key";
    const startBinding = binding({
      idempotency_key: exactKey,
      binding: { initial_run_status: "running" },
    });
    const started = registry.start(startBinding);
    const generation = registry.generationOf(started);
    const tombstone = registry.transition(
      "run-1",
      binding({ idempotency_key: exactKey }),
      terminalStatus,
    );

    assert.throws(
      () => registry.start({ ...startBinding, binding: { ...startBinding.binding } }),
      new RegExp(`cannot start from ${terminalStatus}`),
    );
    assert.equal(registry.get("run-1").status, terminalStatus);
    assert.strictEqual(registry.generationOf(tombstone), generation);
  });
}

test("terminal cancel dominates an exact same-key retry after failed start cleanup", async () => {
  const createEntered = deferred();
  const releaseCreate = deferred();
  let createCalls = 0;
  let opens = 0;
  const controller = new RuntimeController({}, {
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => { opens += 1; },
      quiesceRun: async () => {},
    },
  });
  controller.roleAgents = {
    createRoleAgent: async () => {
      createCalls += 1;
      if (createCalls === 1) {
        createEntered.resolve();
        await releaseCreate.promise;
        throw new Error("deterministic first start failure");
      }
      return { dispose: async () => {} };
    },
    quiesceRun: async () => {},
  };
  const exactKey = "shared-failed-start-cancel-key";
  const startBinding = binding({
    idempotency_key: exactKey,
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/model",
      review_model_id: "provider/model",
    },
  });

  const starting = outcome(controller.startRun(startBinding));
  await createEntered.promise;
  const cancelling = controller.cancel(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: exactKey,
  }));
  const retryWhileCancelling = outcome(controller.startRun({
    ...startBinding,
    binding: { ...startBinding.binding },
  }));
  releaseCreate.resolve();
  const [failedStart, cancelled, cancellingRetry] = await Promise.all([
    starting,
    cancelling,
    retryWhileCancelling,
  ]);

  assert.equal(failedStart.status, "rejected");
  assert.match(failedStart.reason.message, /deterministic first start failure/);
  assert.equal(cancelled.accepted, true);
  assert.equal(cancellingRetry.status, "rejected");
  assert.match(cancellingRetry.reason.message, /cannot start from cancelling/);
  assert.equal(controller.registry.get("run-1").status, "cancelled");

  const retried = await outcome(controller.startRun({
    ...startBinding,
    binding: { ...startBinding.binding },
  }));

  assert.equal(retried.status, "rejected");
  assert.match(retried.reason.message, /cannot start from cancelled/);
  assert.equal(controller.registry.get("run-1").status, "cancelled");
  assert.equal(createCalls, 1);
  assert.equal(opens, 0);
});

test("failed start cleanup preserves another controller's same-key cancel tombstone", async () => {
  const registry = new RuntimeRunRegistry();
  const createEntered = deferred();
  const releaseCreate = deferred();
  const cancelDrainEntered = deferred();
  const releaseCancelDrain = deferred();
  let createCalls = 0;
  let opens = 0;
  const starter = new RuntimeController({}, {
    registry,
    presetCatalog: [{ preset_id: "ecology-researcher-v7", tool_profile: "test" }],
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => { opens += 1; },
    },
  });
  starter.roleAgents = {
    createRoleAgent: async () => {
      createCalls += 1;
      if (createCalls === 1) {
        createEntered.resolve();
        await releaseCreate.promise;
        throw new Error("deterministic external-cancel start failure");
      }
      return { dispose: async () => {} };
    },
    quiesceRun: async () => {},
  };
  const canceller = new RuntimeController({}, {
    registry,
    stageRunner: {
      closeLaunchFence: () => {},
      quiesceRun: async () => {
        cancelDrainEntered.resolve();
        await releaseCancelDrain.promise;
      },
    },
  });
  canceller.roleAgents = { quiesceRun: async () => {} };
  const exactKey = "shared-external-start-cancel-key";
  const startBinding = binding({
    idempotency_key: exactKey,
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/model",
      review_model_id: "provider/model",
    },
  });

  const starting = outcome(starter.startRun(startBinding));
  await createEntered.promise;
  const cancelling = outcome(canceller.cancel(binding({
    run_state_revision: 8,
    ledger_expected_revision: 12,
    idempotency_key: exactKey,
  })));
  await cancelDrainEntered.promise;
  assert.equal(registry.get("run-1").status, "cancelling");

  releaseCreate.resolve();
  const failedStart = await starting;
  const retryWhileCancelling = await outcome(starter.startRun({
    ...startBinding,
    binding: { ...startBinding.binding },
  }));
  releaseCancelDrain.resolve();
  const cancelled = await cancelling;
  const retryAfterCancel = await outcome(starter.startRun({
    ...startBinding,
    binding: { ...startBinding.binding },
  }));

  assert.equal(failedStart.status, "rejected");
  assert.match(failedStart.reason.message, /external-cancel start failure/);
  assert.equal(retryWhileCancelling.status, "rejected");
  assert.equal(retryWhileCancelling.reason.code, "runtime_start_transition_invalid");
  assert.equal(cancelled.status, "fulfilled");
  assert.equal(registry.get("run-1").status, "cancelled");
  assert.equal(retryAfterCancel.status, "rejected");
  assert.equal(retryAfterCancel.reason.code, "runtime_start_transition_invalid");
  assert.equal(createCalls, 1);
  assert.equal(opens, 0);
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
  const pausing = controller.pause(binding({
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
  await Promise.all([starting, pausing, resuming]);

  assert.equal(controller.registry.get("run-1").status, "running");
  assert.equal(calls.filter(([name]) => name === "close").length, 2);
  assert.deepEqual(calls.filter(([name]) => name === "hosts"), [
    ["hosts", false, true],
    ["hosts", false, true],
  ]);
  assert.equal(calls.filter(([name]) => name === "open").length, 1);
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
  await startReadyRun(controller);
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
