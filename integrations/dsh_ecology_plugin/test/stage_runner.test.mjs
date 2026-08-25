import assert from "node:assert/strict";
import test from "node:test";

import { NativeStageRunner, STAGES, dshCompatibleSchema, jsonDigest } from "../lib/runtime/stage-runner.js";

function skillFirstEvents(skillName, { prediction = false } = {}) {
  const events = [
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
  ];
  if (prediction) {
    events.push(
      {
        seq: 3,
        type: "tool/call",
        data: {
          callId: "prediction-call",
          name: "ecology_execute_prediction_tool",
          arguments: { tool_id: "ridge@1", wave_digest: "f".repeat(64) },
        },
      },
      {
        seq: 4,
        type: "tool/result",
        data: {
          message: {
            content: [{
              type: "tool-result",
              toolCallId: "prediction-call",
              isError: false,
              content: [],
            }],
          },
        },
      },
    );
  }
  events.push({
    seq: prediction ? 5 : 3,
    type: "tool/call",
    data: { callId: "structured-call", name: "structured_output", arguments: {} },
  });
  return events;
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

test("post-score sample reflection is a registered structured DSH stage", () => {
  assert.deepEqual(
    {
      role: STAGES["sample.reflect"].role,
      schema: STAGES["sample.reflect"].schema,
      file: STAGES["sample.reflect"].file,
    },
    {
      role: "sample-critic",
      schema: "ecology-sample-reflection@1",
      file: "sample-reflection",
    },
  );
  assert.match(STAGES["sample.reflect"].instruction, /prediction vector is already immutable/i);
  assert.match(STAGES["sample.reflect"].instruction, /next-generation action/i);
});

test("candidate judging and batch reflection use distinct scientific Skills", () => {
  const judge = STAGES["generation.judge"];
  const reflection = STAGES["generation.reflect"];

  assert.equal(judge.skillName, "candidate-scientific-review");
  assert.equal(reflection.skillName, "batch-scientific-reflection");
  assert.notEqual(judge.skillName, reflection.skillName);
  assert.match(judge.instruction, /single candidate/i);
  assert.match(
    judge.instruction,
    /only schema_version, accepted, rationale, and flags/i,
  );
  assert.match(judge.instruction, /Do not propose next-generation directions/i);
  assert.match(judge.instruction, /not selection or promotion/i);
  assert.doesNotMatch(
    judge.instruction,
    /return exactly \d+ (?:distinct )?(?:next[- ]step )?directions/i,
  );
  assert.match(reflection.instruction, /mutation_direction/i);
  assert.match(reflection.instruction, /next research synthesis Host preflight/i);
});

test("research synthesis and candidate proposal bind structured mutation direction", () => {
  const synthesis = STAGES["generation.research-synthesis"];
  const proposal = STAGES["candidate.propose"];

  assert.match(synthesis.instruction, /mutation_direction/i);
  assert.match(synthesis.instruction, /increase or decrease/i);
  assert.match(synthesis.instruction, /exact parameter assignment/i);
  assert.match(proposal.instruction, /increase must be strictly above/i);
  assert.match(proposal.instruction, /decrease strictly below/i);
  assert.match(proposal.instruction, /must be select/i);
});

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
  const roleHost = {
    sessionId: "parent-session",
    agent: { id: "role-host" },
    binding: { model: "pjlab/deepseek-v4-pro-0813" },
  };
  const structured = {
    schema_version: "ecologyrsi-dsh.genome-mutation/1",
    operations: [],
  };
  const persisted = [];
  let disposed = false;
  let claimedIdentity;
  let runner;
  const gatedProviders = [];
  const candidateEvents = skillFirstEvents("bounded-plugin-experiment");
  const ctx = {
    sessions: {
      get: (id) => (
        id === "child-session" ? { id, events: candidateEvents } : undefined
      ),
    },
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
        assert.match(prompt.instruction, /call skill exactly once/i);
        assert.match(prompt.instruction, /structured_output exactly once/i);
        assert.match(prompt.instruction, /mutation delta/i);
        assert.match(prompt.instruction, /omit every unchanged/i);
        assert.match(prompt.instruction, /exactly one operation/i);
        assert.match(prompt.instruction, /assigned candidate direction/i);
        assert.match(prompt.instruction, /normalized trust-region step/i);
        assert.match(prompt.instruction, /do not reconstruct/i);
        const child = {
          id: "child-session",
          session: {
            header: { parentSession: "parent-session" },
          events: [
            { type: "subagent/descriptor", data: { label: request.label } },
            ...candidateEvents,
          ],
          },
        };
        claimedIdentity = runner.childBindings.claimPublished(
          "parent-session",
          request.label,
          child.id,
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
    runRegistry: { get: () => ({ status: "running" }) },
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
    providerStageGate: {
      run: async (provider, operation) => {
        gatedProviders.push(provider);
        return operation();
      },
      penalize: () => {},
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

  assert.equal(claimedIdentity.role, "candidate-proposer");
  assert.deepEqual(claimedIdentity.allowed_tools, ["skill"]);
  assert.equal(persisted[0].path, "/api/ecology-agent-sidecar/v1/child-reservations");
  assert.equal(persisted[1].path, "/api/ecology-agent-sidecar/v1/structured-results");
  assert.equal(persisted[1].body.identity.session_id, "child-session");
  assert.equal(
    persisted[1].body.skill_invocation_evidence.skill_name,
    "bounded-plugin-experiment",
  );
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
  assert.deepEqual(gatedProviders, ["pjlab"]);
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

test("native stage runner retries one transient child model failure with a fresh reservation", async () => {
  const roleHost = {
    sessionId: "parent-session",
    agent: { id: "role-host" },
    binding: { model: "freerouter/gpt-5.6-sol" },
  };
  const structured = {
    schema_version: "ecology-generation-review@1",
    accepted: false,
    rationale: "Insufficient evidence.",
    flags: ["insufficient_evidence"],
  };
  let starts = 0;
  let reservations = 0;
  let persisted = 0;
  let penalties = 0;
  const disposed = [];
  const judgeEvents = skillFirstEvents("candidate-scientific-review");
  const runner = new NativeStageRunner({
    sessions: {
      get: (id) => (id.startsWith("child-") ? { id, events: judgeEvents } : undefined),
    },
    subagents: {
      start: async () => {
        starts += 1;
        const id = `child-${starts}`;
        return {
          id,
          result: Promise.resolve(
            starts === 1
              ? { stopReason: "model_error" }
              : { stopReason: "completed", structured },
          ),
          dispose: async () => { disposed.push(id); },
        };
      },
    },
  }, {
    roleAgents: { get: () => roleHost },
    runRegistry: { get: () => ({ status: "running" }) },
    sidecar: {
      request: async (path, options) => {
        if (path.endsWith("/child-reservations")) {
          reservations += 1;
          return {
            accepted: true,
            launch: {
              reservation_id: `reservation-${reservations}`,
              launch_attempt: reservations,
            },
            ledger_expected_revision: 10 + reservations,
          };
        }
        persisted += 1;
        return { accepted: true, result_digest: options.body.result_digest };
      },
    },
    structuredStageMaxAttempts: 2,
    providerStageGate: {
      run: async (_provider, operation) => operation(),
      penalize: () => { penalties += 1; },
    },
  });
  const context = { candidate_id: "candidate-1" };

  const result = await runner.run({
    run_id: "run-retry",
    stage: "generation.judge",
    run_state_revision: 7,
    stage_attempt: 1,
    ledger_expected_revision: 9,
    idempotency_key: "judge-1",
    request: {
      role: "generation-judge",
      output_schema_id: "ecology-generation-review@1",
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

  assert.deepEqual(result.structured, structured);
  assert.equal(starts, 2);
  assert.equal(reservations, 2);
  assert.equal(persisted, 1);
  assert.equal(penalties, 1);
  assert.deepEqual(disposed, ["child-1", "child-2"]);
});

test("sample planner waves execute through the retained DSH Workflow Engine", async () => {
  const listeners = new Map();
  const sessions = new Map([[
    "workflow-child-session",
    {
      id: "workflow-child-session",
      events: skillFirstEvents("origin-vector-forecasting-balanced", { prediction: true }),
    },
  ]]);
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
          listeners.get("workflow/agent-end")?.(
            { id: "workflow-run-1", meta: request.meta },
            {
              seq: 1,
              label: item.label,
              childId: "workflow-child-session",
              outcome: "completed",
            },
          );
          sessions.delete("workflow-child-session");
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
    sessions: { get: (sessionId) => sessions.get(sessionId) },
    tokenMeter: {
      measure: () => ({
        logRevision: 4,
        baseline: { kind: "usage", tokens: 70 },
        totalTokens: 75,
        surfaceTokens: 50,
      }),
    },
    sessionProjections: {
      snapshot: () => ({
        values: {
          tokenUsage: {
            uncachedInputTokens: 50,
            outputTokens: 15,
            cacheReadTokens: 10,
            cacheWriteTokens: 0,
          },
        },
      }),
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
  const context = {
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
  };
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
      context_canonical_json: canonicalJson(context),
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
  const plannerPrompt = JSON.parse(workflowRequest.args.items[0].prompt);
  assert.match(
    plannerPrompt.instruction,
    /first response must call skill exactly once with name origin-vector-forecasting-balanced/i,
  );
  assert.match(
    plannerPrompt.instruction,
    /After the Skill result, call ecology_execute_prediction_tool exactly once/i,
  );
  assert.match(
    plannerPrompt.instruction,
    /After the prediction-tool result, call structured_output exactly once/i,
  );
  assert.equal(persisted[1].body.identity.session_id, "workflow-child-session");
  assert.deepEqual(
    persisted[1].body.session_metrics.provider_usage.totals,
    {
      uncached_input_tokens: 50,
      output_tokens: 15,
      cache_read_tokens: 10,
      cache_write_tokens: 0,
      total_tokens: 75,
    },
  );
  assert.equal(result.session_id, "workflow-child-session");
  assert.deepEqual(result.structured, structured);
  assert.equal(workflowDisposed, true);
});
