import assert from "node:assert/strict";
import test from "node:test";

import { NativeStageRunner, dshCompatibleSchema, jsonDigest } from "../lib/runtime/stage-runner.js";

test("full host schemas are projected to the DSH structured-output subset", () => {
  const projected = dshCompatibleSchema({
    $id: "schema@1",
    type: "object",
    properties: {
      version: { const: "v1" },
      values: { type: "array", minItems: 1, maxItems: 3, items: { type: "string", minLength: 1 } },
    },
    required: ["version", "values"],
    additionalProperties: false,
  });
  assert.equal("$id" in projected, false);
  assert.deepEqual(projected.properties.version, { const: "v1", type: "string" });
  assert.equal("minItems" in projected.properties.values, false);
  assert.equal("maxItems" in projected.properties.values, false);
  assert.equal("minLength" in projected.properties.values.items, false);
});

test("DSH projection normalizes enum and union type syntax", () => {
  const projected = dshCompatibleSchema({
    type: "object",
    properties: {
      role: { enum: ["sample-planner", "sample-repair"] },
      value: { type: ["string", "number", "integer", "boolean"] },
    },
    required: ["role", "value"],
    additionalProperties: false,
  });
  assert.deepEqual(projected.properties.role, {
    enum: ["sample-planner", "sample-repair"],
    type: "string",
  });
  assert.deepEqual(projected.properties.value, {
    oneOf: [
      { type: "string" },
      { type: "number" },
      { type: "integer" },
      { type: "boolean" },
    ],
  });
});

test("native stage runner reserves before first child tool and durably persists structured output", async () => {
  const roleHost = { sessionId: "parent-session", agent: { id: "role-host" } };
  const structured = {
    schema_version: "ecologyrsi-dsh.genome-mutation/1",
    operations: [],
  };
  const persisted = [];
  let disposed = false;
  let claimedIdentity;
  let runner;
  const ctx = {
    sessions: { get: (id) => (id === "child-session" ? { id } : undefined) },
    tokenMeter: {
      measure: () => ({
        logRevision: 9,
        baseline: { kind: "usage", tokens: 100 },
        surfaceDeltaTokens: 20,
        totalTokens: 120,
        surfaceTokens: 80,
        nodes: [],
      }),
    },
    sessionProjections: {
      snapshot: () => ({
        values: {
          tokenUsage: {
            uncachedInputTokens: 100,
            outputTokens: 20,
            cacheReadTokens: 30,
            cacheWriteTokens: 0,
          },
        },
      }),
    },
    subagents: {
      start: async (provider, request) => {
        assert.equal(provider, "spawn");
        assert.deepEqual(request.prompt[0], { type: "text", text: request.prompt[0].text });
        const prompt = JSON.parse(request.prompt[0].text);
        assert.match(prompt.instruction, /Do not narrate analysis/i);
        assert.match(prompt.instruction, /first response/i);
        assert.match(prompt.instruction, /structured_output exactly once/i);
        assert.match(prompt.instruction, /mutation delta/i);
        assert.match(prompt.instruction, /omit every unchanged/i);
        assert.match(prompt.instruction, /1-4 operations/i);
        assert.match(prompt.instruction, /do not reconstruct/i);
        const child = {
          id: "child-session",
          session: {
            header: { parentSession: "parent-session" },
            events: [{ type: "subagent/descriptor", data: { label: request.label } }],
          },
        };
        claimedIdentity = runner.bridge.bindingFor(
          { agent: child },
          { role: "candidate-proposer", toolName: "ecology_get_run_context" },
        );
        return {
          id: child.id,
          result: Promise.resolve({ structured, text: "ignored free text" }),
          dispose: async () => { disposed = true; },
        };
      },
    },
  };
  runner = new NativeStageRunner(ctx, {
    roleAgents: { get: () => roleHost },
    runRegistry: { get: () => ({ status: "created" }) },
    sidecar: {
      request: async (path, options) => {
        persisted.push({ path, body: options.body });
        if (path.endsWith("/child-reservations")) {
          return {
            accepted: true,
            launch: {
              reservation_id: "reservation-1",
              run_id: "run-1",
              stage: "candidate.propose",
              role: "candidate-proposer",
              item_digest: options.body.item_digest,
              idempotency_key: "proposal-1",
              launch_attempt: 1,
            },
            ledger_expected_revision: 12,
          };
        }
        return { accepted: true, result_digest: options.body.result_digest };
      },
    },
  });
  const context = { parent_genome: { genome_digest: "a".repeat(64) } };
  const result = await runner.run({
    run_id: "run-1",
    stage: "candidate.propose",
    run_state_revision: 7,
    stage_attempt: 2,
    ledger_expected_revision: 11,
    idempotency_key: "proposal-1",
    request: {
      role: "candidate-proposer",
      output_schema_id: "ecology-genome-mutation@1",
      context,
      context_canonical_json: JSON.stringify(context),
      context_digest: jsonDigest(context),
      identity_digests: {
        genome_digest: "a".repeat(64),
        compiled_behavior_digest: "b".repeat(64),
        phenotype_instance_digest: "c".repeat(64),
      },
    },
  });

  assert.equal(claimedIdentity.session_id, "child-session");
  assert.equal(claimedIdentity.role, "candidate-proposer");
  assert.match(claimedIdentity.activation_lease_id, /^ecology-lease-/);
  assert.equal(persisted[0].path, "/api/ecology-agent-sidecar/v1/child-reservations");
  assert.equal(persisted[1].path, "/api/ecology-agent-sidecar/v1/structured-results");
  assert.equal(persisted[1].body.identity.session_id, "child-session");
  assert.deepEqual(
    persisted[1].body.session_metrics.provider_usage.totals,
    {
      uncached_input_tokens: 100,
      output_tokens: 20,
      cache_read_tokens: 30,
      cache_write_tokens: 0,
      total_tokens: 150,
    },
  );
  assert.equal(persisted[1].body.session_metrics.context_pressure.total_tokens, 120);
  assert.deepEqual(result.structured, structured);
  assert.equal(result.result_digest, jsonDigest(structured));
  assert.equal(result.session_id, "child-session");
  assert.equal(disposed, true);
});

test("native stage runner rejects a changed context before starting DSH", async () => {
  let started = false;
  const runner = new NativeStageRunner(
    { subagents: { start: async () => { started = true; } } },
    {
      roleAgents: { get: () => ({ sessionId: "p", agent: {} }) },
      runRegistry: { get: () => ({ status: "created" }) },
      sidecar: { request: async () => ({ accepted: true }) },
    },
  );
  await assert.rejects(
    runner.run({
      run_id: "run-1",
      stage: "generation.research",
      run_state_revision: 1,
      stage_attempt: 1,
      ledger_expected_revision: 1,
      idempotency_key: "research-1",
      request: {
        role: "researcher",
        output_schema_id: "ecology-research-result@1",
        context: { frozen: true },
        context_canonical_json: JSON.stringify({ frozen: true }),
        context_digest: "0".repeat(64),
        identity_digests: {},
      },
    }),
    /context digest mismatch/,
  );
  assert.equal(started, false);
});

test("sample planner waves execute through the retained DSH Workflow Engine", async () => {
  const listeners = new Map();
  let directSubagentStarted = false;
  let workflowRequest;
  let workflowDisposed = false;
  const structured = {
    schema_version: "ecology-sample-decisions@1",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  const roleHost = {
    sessionId: "planner-parent-session",
    agent: { id: "planner-role-host" },
    services: {
      workflowEngine: {
        start: (request) => {
          workflowRequest = request;
          const item = request.args.items[0];
          listeners.get("workflow/agent-start")?.(
            { id: "workflow-run-1", meta: request.meta },
            { seq: 1, label: item.label, childId: "workflow-child-session" },
          );
          return {
            id: "workflow-run-1",
            result: Promise.resolve({
              value: [structured],
              stopReason: "completed",
              agentsStarted: 1,
            }),
            cancel: () => {},
            dispose: async () => { workflowDisposed = true; },
          };
        },
      },
    },
  };
  const persisted = [];
  const ctx = {
    on: (name, listener) => {
      listeners.set(name, listener);
      return () => listeners.delete(name);
    },
    subagents: {
      start: async () => {
        directSubagentStarted = true;
        throw new Error("sample planner must use Workflow Engine");
      },
    },
  };
  const runner = new NativeStageRunner(ctx, {
    roleAgents: { get: () => roleHost },
    runRegistry: { get: () => ({ status: "running" }) },
    sidecar: {
      request: async (path, options) => {
        persisted.push({ path, body: options.body });
        if (path.endsWith("/child-reservations")) {
          return {
            accepted: true,
            launch: {
              reservation_id: "workflow-reservation-1",
              launch_attempt: 1,
            },
            ledger_expected_revision: 12,
          };
        }
        return { accepted: true, result_digest: options.body.result_digest };
      },
    },
  });
  const context = { wave_digest: "f".repeat(64), samples: [] };
  const result = await runner.run({
    run_id: "run-workflow",
    stage: "sample.plan",
    run_state_revision: 7,
    stage_attempt: 2,
    ledger_expected_revision: 11,
    idempotency_key: "sample-plan-1",
    request: {
      role: "sample-planner",
      output_schema_id: "ecology-sample-decisions@1",
      context,
      context_canonical_json: JSON.stringify({ samples: [], wave_digest: "f".repeat(64) }),
      context_digest: jsonDigest(context),
      identity_digests: {
        genome_digest: "a".repeat(64),
        compiled_behavior_digest: "b".repeat(64),
        phenotype_instance_digest: "c".repeat(64),
      },
    },
  });

  assert.equal(directSubagentStarted, false);
  assert.equal(workflowRequest.parent, roleHost.agent);
  assert.match(workflowRequest.script, /parallel/);
  assert.equal(workflowRequest.args.items[0].schema.type, "object");
  assert.equal(persisted[1].body.identity.session_id, "workflow-child-session");
  assert.equal(result.session_id, "workflow-child-session");
  assert.deepEqual(result.structured, structured);
  assert.equal(workflowDisposed, true);
});
