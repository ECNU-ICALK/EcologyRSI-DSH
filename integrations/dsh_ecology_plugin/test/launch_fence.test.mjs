import assert from "node:assert/strict";
import test from "node:test";

import { RuntimeController } from "../lib/runtime/controller.js";
import { ProviderStageGate } from "../lib/runtime/provider-stage-gate.js";
import { RuntimeRunRegistry } from "../lib/runtime/run-registry.js";
import { NativeStageRunner, jsonDigest } from "../lib/runtime/stage-runner.js";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
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

function stageBinding({ workflow = false, suffix = "race", revision = 7 } = {}) {
  const context = workflow
    ? {
      schema_version: "ecologyrsi-dsh.sample-routing-wave/1",
      wave_digest: "f".repeat(64),
      samples: [],
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
    run_id: workflow ? "run-workflow-launch-race" : "run-child-launch-race",
    stage: workflow ? "sample.plan" : "generation.research",
    admission_id: `admission-${suffix}`,
    run_state_revision: revision,
    stage_attempt: 2,
    ledger_expected_revision: revision + 4,
    idempotency_key: `stage-${suffix}`,
    binding: { initial_run_status: "running" },
    request: {
      role: workflow ? "sample-planner" : "researcher",
      output_schema_id: workflow
        ? "ecology-sample-decisions@1"
        : "ecology-research-result@1",
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

function launchRaceHarness({ workflow = false } = {}) {
  const schemaEntered = deferred();
  const releaseSchema = deferred();
  const registry = new RuntimeRunRegistry();
  const providerStageGate = new ProviderStageGate({
    minimumIntervalMs: 0,
    failureCooldownMs: 0,
  });
  const structured = workflow
    ? {
      schema_version: "ecology-sample-decisions@1",
      wave_digest: "f".repeat(64),
      decisions: [],
    }
    : { schema_version: "ecology-research-result@1", findings: [] };
  let childStarts = 0;
  let workflowStarts = 0;
  let reservations = 0;
  const roleHost = {
    sessionId: workflow ? "planner-parent" : "research-parent",
    agent: { id: workflow ? "planner-host" : "research-host" },
    binding: { model: "pjlab/deepseek-v4-pro-0813" },
    services: {
      workflowEngine: {
        start: () => {
          workflowStarts += 1;
          return {
            result: Promise.resolve({ value: [structured], stopReason: "completed" }),
            cancel: () => {},
            dispose: async () => {},
          };
        },
      },
    },
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
      return await releaseSchema.promise;
    }
    return { type: "object", additionalProperties: true };
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
    counts: () => ({ childStarts, workflowStarts }),
  };
}

test("pause closes the fence before a schema-blocked direct child can launch", { timeout: 2_000 }, async () => {
  const harness = launchRaceHarness();
  const first = stageBinding();
  harness.registry.start(first);
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
  assert.deepEqual(harness.counts(), { childStarts: 0, workflowStarts: 0 });
  assert.equal(harness.runner.pendingStarts.size, 0);
  assert.equal(harness.runner.activeWorkflows.size, 0);
  assert.equal(harness.providerStageGate.records.size, 0);
  assert.equal(harness.registry.get(first.run_id).status, "paused");

  await harness.controller.resume(controlBinding(first, {
    status: "resume",
    revision: 9,
  }));
  const resumed = stageBinding({ suffix: "resumed", revision: 9 });
  const result = await harness.controller.runStage(resumed);

  assert.equal(result.accepted, true);
  assert.deepEqual(harness.counts(), { childStarts: 1, workflowStarts: 0 });
  assert.equal(harness.registry.get(first.run_id).status, "running");
});

test("cancel closes the fence before a schema-blocked Workflow can launch", { timeout: 2_000 }, async () => {
  const harness = launchRaceHarness({ workflow: true });
  const first = stageBinding({ workflow: true });
  harness.registry.start(first);
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
  assert.deepEqual(harness.counts(), { childStarts: 0, workflowStarts: 0 });
  assert.equal(harness.runner.pendingStarts.size, 0);
  assert.equal(harness.runner.activeWorkflows.size, 0);
  assert.equal(harness.providerStageGate.records.size, 0);
  assert.equal(harness.registry.get(first.run_id).status, "cancelled");
});
