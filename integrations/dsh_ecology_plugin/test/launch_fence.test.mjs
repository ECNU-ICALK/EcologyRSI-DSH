import assert from "node:assert/strict";
import test from "node:test";

import { RuntimeController } from "../lib/runtime/controller.js";
import { ProviderStageGate } from "../lib/runtime/provider-stage-gate.js";
import { RuntimeRunRegistry } from "../lib/runtime/run-registry.js";
import { NativeStageRunner, jsonDigest } from "../lib/runtime/stage-runner.js";

const REALISTIC_PRESET_CATALOG = Object.freeze([
  "ecology-coordinator-v5",
  "ecology-researcher-v12",
  "ecology-candidate-proposer-v4",
  "ecology-sample-planner-v8",
  "ecology-sample-critic-v5",
  "ecology-generation-judge-v8",
].map((preset_id) => ({
  preset_id,
  tool_profile: "dynamic-retrieval-v1",
  required_tools: preset_id === "ecology-sample-planner-v8"
    ? ["ecology_execute_prediction_tool", "skill", "web_search"]
    : ["skill", "web_search"],
})));

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

async function startReadyControllerRun(controller, startBinding) {
  const stageRunner = controller.stageRunner;
  const presetCatalog = controller.presetCatalog;
  const roleAgents = controller.roleAgents;
  controller.stageRunner = null;
  controller.presetCatalog = [
    { preset_id: "ecology-researcher-v12", tool_profile: "test" },
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

function liveCapabilityContext({ beforeCreate = async () => {} } = {}) {
  const toolsByPreset = new Map(REALISTIC_PRESET_CATALOG.map((item) => [
    `standing:${item.preset_id}`,
    item.required_tools,
  ]));
  return {
    agents: {
      create: async (options) => {
        await beforeCreate(options);
        const agent = {
          session: { append: async () => {}, flush: async () => {} },
          waitForIdle: async () => {},
        };
        await options.setup?.(agent);
        return { agent, dispose: async () => {} };
      },
    },
    sessions: {},
    tokenMeter: {},
    subagents: {},
    tools: {
      schemas: async (standingKey) => (
        toolsByPreset.get(standingKey)?.map((name) => ({ name })) || []
      ),
    },
    sessionPersistence: {},
    sessionProjections: {},
    agentPresets: {
      standingKeyFor: async (presetId) => `standing:${presetId}`,
      mount: async (_agent, presetId) => ({ id: presetId }),
      serviceFor: async (_agent, serviceName) => ({ serviceName }),
    },
    llm: { resolveCallConfig: async () => ({ provider: "test", model: "model" }) },
    web: {},
  };
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map(
      (key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`,
    ).join(",")}}`;
  }
  return JSON.stringify(value);
}

function skillFirstEvents(skillName) {
  return [
    {
      seq: 1,
      type: "tool/call",
      data: { callId: "skill-call", name: "skill", arguments: { name: skillName } },
    },
    {
      seq: 2,
      type: "tool/result",
      data: {
        message: {
          content: [{
            type: "tool-result",
            toolCallId: "skill-call",
            isError: false,
            content: [],
          }],
        },
      },
    },
    {
      seq: 3,
      type: "tool/call",
      data: { callId: "structured-call", name: "structured_output", arguments: {} },
    },
  ];
}

function stageBinding({ samplePlan = false, suffix = "race", revision = 7 } = {}) {
  const context = samplePlan
    ? {
      schema_version: "ecologyrsi-dsh.sample-routing-wave/1",
      wave_digest: "f".repeat(64),
      samples: [{ sample_id: "origin-launch-race" }],
      context: {
        candidate_agent_profile: {
          schema_version: "ecologyrsi-dsh.candidate-agent-profile/1",
          role: "sample-planner",
          skill_name: "origin-vector-forecasting-balanced",
        },
      },
    }
    : { question: "bounded launch fence" };
  return {
    run_id: samplePlan ? "run-sample-plan-launch-race" : "run-child-launch-race",
    stage: samplePlan ? "sample.plan" : "generation.research",
    admission_id: `admission-${suffix}`,
    run_state_revision: revision,
    stage_attempt: 2,
    ledger_expected_revision: revision + 4,
    idempotency_key: `stage-${suffix}`,
    binding: { initial_run_status: "running" },
    request: {
      role: samplePlan ? "sample-planner" : "researcher",
      output_schema_id: samplePlan
        ? "ecology-sample-predictions@2"
        : "ecology-research-result@1",
      ...(samplePlan ? { max_tokens: 2048 } : {}),
      context,
      context_canonical_json: canonicalJson(context),
      context_digest: jsonDigest(context),
      identity_digests: {
        genome_digest: "a".repeat(64),
        compiled_behavior_digest: "b".repeat(64),
        phenotype_instance_digest: "c".repeat(64),
      },
    },
  };
}

function controlBinding(stage, { status, revision }) {
  return {
    run_id: stage.run_id,
    run_state_revision: revision,
    stage_attempt: stage.stage_attempt,
    ledger_expected_revision: revision + 4,
    idempotency_key: `${status}-${revision}`,
  };
}

function launchRaceHarness({ samplePlan = false } = {}) {
  const schemaEntered = deferred();
  const releaseSchema = deferred();
  const registry = new RuntimeRunRegistry();
  const providerStageGate = new ProviderStageGate({
    minimumIntervalMs: 0,
    failureCooldownMs: 0,
  });
  const structured = samplePlan
    ? {
      schema_version: "ecology-sample-predictions@2",
      wave_digest: "f".repeat(64),
      decisions: [],
    }
    : { schema_version: "ecology-research-result@1", findings: [] };
  const outputSchema = samplePlan
    ? {
      type: "object",
      properties: {
        wave_digest: { type: "string" },
        decisions: {
          type: "array",
          items: {
            type: "object",
            properties: { sample_id: { type: "string" } },
          },
        },
      },
    }
    : { type: "object", additionalProperties: true };
  let childStarts = 0;
  let reservations = 0;
  const roleHost = {
    sessionId: samplePlan ? "planner-parent" : "research-parent",
    agent: { id: samplePlan ? "planner-host" : "research-host" },
    binding: { model: "pjlab/deepseek-v4-pro-0813" },
  };
  const ctx = {
    on: () => () => {},
    sessions: {
      get: (sessionId) => ({
        id: sessionId,
        events: skillFirstEvents("autonomous-ecology-research"),
      }),
    },
    subagents: {
      start: async () => {
        childStarts += 1;
        const id = `direct-child-${childStarts}`;
        return {
          id,
          result: Promise.resolve({ stopReason: "completed", structured }),
          dispose: async () => {},
        };
      },
    },
  };
  const runner = new NativeStageRunner(ctx, {
    roleAgents: { get: () => roleHost },
    runRegistry: registry,
    sidecar: {
      request: async (path, options) => {
        if (path.endsWith("/child-reservations")) {
          reservations += 1;
          return {
            accepted: true,
            admission_id: options.body.admission_id,
            timeout_ms: options.body.timeout_ms,
            launch: {
              reservation_id: `launch-race-reservation-${reservations}`,
              launch_attempt: reservations,
            },
            ledger_expected_revision: options.body.ledger_expected_revision ?? 12,
          };
        }
        return { accepted: true, result_digest: options.body.result_digest };
      },
    },
    // The race must settle through the launch fence, not by waiting for the
    // structured-stage watchdog.
    structuredStageTimeoutMs: 60_000,
    researchStageTimeoutMs: 60_000,
    structuredStageMaxAttempts: 1,
    providerStageGate,
  });
  let firstSchema = true;
  runner.schema = async () => {
    if (firstSchema) {
      firstSchema = false;
      schemaEntered.resolve();
      await releaseSchema.promise;
    }
    return structuredClone(outputSchema);
  };
  const controller = new RuntimeController(ctx, { registry, stageRunner: runner });
  controller.roleAgents = { quiesceRun: async () => {} };
  return {
    controller,
    registry,
    runner,
    providerStageGate,
    schemaEntered,
    releaseSchema,
    counts: () => ({ childStarts }),
  };
}

test("pause closes the fence before a schema-blocked direct child can launch", { timeout: 2_000 }, async () => {
  const harness = launchRaceHarness();
  const first = stageBinding();
  await startReadyControllerRun(harness.controller, first);
  const stageOutcome = harness.controller.runStage(first).then(
    (value) => ({ value }),
    (error) => ({ error }),
  );
  await harness.schemaEntered.promise;

  const pausing = harness.controller.pause(controlBinding(first, {
    status: "pause",
    revision: 8,
  }));
  assert.equal(harness.registry.get(first.run_id).status, "pausing");
  harness.releaseSchema.resolve({ type: "object", additionalProperties: true });

  const [outcome] = await Promise.all([stageOutcome, pausing]);
  assert.ok(outcome.error);
  assert.equal(outcome.error.code, "provider_stage_admission_closed");
  assert.deepEqual(harness.counts(), { childStarts: 0 });
  assert.equal(harness.runner.pendingStarts.size, 0);
  assert.equal(harness.providerStageGate.records.size, 0);
  assert.equal(harness.registry.get(first.run_id).status, "paused");

  await harness.controller.resume(controlBinding(first, {
    status: "resume",
    revision: 9,
  }));
  const resumed = stageBinding({ suffix: "resumed", revision: 9 });
  const result = await harness.controller.runStage(resumed);

  assert.equal(result.accepted, true);
  assert.deepEqual(harness.counts(), { childStarts: 1 });
  assert.equal(harness.registry.get(first.run_id).status, "running");
});

test("cancel closes the fence before a schema-blocked sample planner child can launch", { timeout: 2_000 }, async () => {
  const harness = launchRaceHarness({ samplePlan: true });
  const first = stageBinding({ samplePlan: true });
  await startReadyControllerRun(harness.controller, first);
  const stageOutcome = harness.controller.runStage(first).then(
    (value) => ({ value }),
    (error) => ({ error }),
  );
  await harness.schemaEntered.promise;

  const cancelling = harness.controller.cancel(controlBinding(first, {
    status: "cancel",
    revision: 8,
  }));
  assert.equal(harness.registry.get(first.run_id).status, "cancelling");
  harness.releaseSchema.resolve({ type: "object", additionalProperties: true });

  const [outcome] = await Promise.all([stageOutcome, cancelling]);
  assert.ok(outcome.error);
  assert.equal(outcome.error.code, "provider_stage_admission_closed");
  assert.deepEqual(harness.counts(), { childStarts: 0 });
  assert.equal(harness.runner.pendingStarts.size, 0);
  assert.equal(harness.providerStageGate.records.size, 0);
  assert.equal(harness.registry.get(first.run_id).status, "cancelled");
});

test("real controller reconciles a durable paused restore with admission closed until resume", { timeout: 2_000 }, async () => {
  const harness = launchRaceHarness();
  const controller = new RuntimeController(liveCapabilityContext(), {
    registry: harness.registry,
    stageRunner: harness.runner,
    presetCatalog: REALISTIC_PRESET_CATALOG,
  });
  const restored = stageBinding({ suffix: "durable-restored-paused" });
  restored.idempotency_key = `runtime-restore:${restored.run_id}`;
  restored.binding = {
    initial_run_status: "paused",
    restore_provenance: {
      source: "python_durable_ledger",
      status: "paused",
    },
  };

  await controller.startRun(restored);
  assert.equal(harness.registry.get(restored.run_id).status, "paused");
  assert.equal(controller.liveReady, true);
  const capabilities = await controller.capabilities();
  assert.equal(capabilities.ready, true);
  assert.equal(capabilities.live_agent_service_ready, true);
  assert.equal(capabilities.presets.length, REALISTIC_PRESET_CATALOG.length);
  assert.equal(
    capabilities.presets.every((item) => item.live_agent_service_ready),
    true,
  );

  const fencedStage = stageBinding({ suffix: "before-restored-resume" });
  const rejected = await outcome(controller.runStage(fencedStage));

  assert.equal(rejected.status, "rejected");
  assert.equal(rejected.reason.code, "provider_stage_admission_closed");
  assert.deepEqual(harness.counts(), { childStarts: 0 });

  await controller.resume(controlBinding(restored, {
    status: "resume",
    revision: 8,
  }));
  const resumedStage = controller.runStage(stageBinding({
    suffix: "after-restored-resume",
    revision: 8,
  }));
  await harness.schemaEntered.promise;
  harness.releaseSchema.resolve({ type: "object", additionalProperties: true });
  const resumed = await resumedStage;

  assert.equal(resumed.accepted, true);
  assert.equal(harness.registry.get(restored.run_id).status, "running");
  assert.deepEqual(harness.counts(), { childStarts: 1 });
});

test("global live readiness survives one run cleanup and closes after the final live run", async () => {
  const secondCreateEntered = deferred();
  const releaseSecondCreate = deferred();
  const controller = new RuntimeController(liveCapabilityContext({
    beforeCreate: async (options) => {
      if (options.meta.ecologyRunId === "run-live-b") {
        secondCreateEntered.resolve();
        await releaseSecondCreate.promise;
      }
    },
  }), {
    presetCatalog: REALISTIC_PRESET_CATALOG,
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => {},
      quiesceRun: async () => {},
    },
  });
  const start = (runId) => ({
    run_id: runId,
    run_state_revision: 1,
    stage_attempt: 0,
    ledger_expected_revision: 1,
    idempotency_key: `start:${runId}`,
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/model",
      review_model_id: "provider/model",
    },
  });
  const cancel = (runId) => ({
    run_id: runId,
    run_state_revision: 2,
    stage_attempt: 0,
    ledger_expected_revision: 2,
    idempotency_key: `cancel:${runId}`,
  });

  await controller.startRun(start("run-live-a"));
  assert.equal(controller.liveReady, true);
  const secondStart = controller.startRun(start("run-live-b"));
  await secondCreateEntered.promise;
  assert.equal(controller.liveReady, false);
  assert.equal((await controller.capabilities()).live_agent_service_ready, false);
  releaseSecondCreate.resolve();
  await secondStart;
  assert.equal(controller.liveReady, true);

  await controller.cancel(cancel("run-live-a"));
  assert.equal(controller.liveReady, true);
  assert.equal((await controller.capabilities()).live_agent_service_ready, true);

  await controller.cancel(cancel("run-live-b"));
  assert.equal(controller.liveReady, false);
  assert.equal((await controller.capabilities()).live_agent_service_ready, false);
});

test("an empty preset catalog cannot manufacture live Agent readiness", async () => {
  const controller = new RuntimeController(liveCapabilityContext(), {
    presetCatalog: [],
    stageRunner: {
      closeLaunchFence: () => {},
      openLaunchFence: () => {},
    },
  });

  await controller.startRun({
    run_id: "run-empty-preset-catalog",
    run_state_revision: 1,
    stage_attempt: 0,
    ledger_expected_revision: 1,
    idempotency_key: "start:run-empty-preset-catalog",
    binding: {
      initial_run_status: "running",
      strategy_model_id: "provider/model",
      review_model_id: "provider/model",
    },
  });

  assert.equal(controller.liveReady, false);
  assert.equal((await controller.capabilities()).live_agent_service_ready, false);
});
