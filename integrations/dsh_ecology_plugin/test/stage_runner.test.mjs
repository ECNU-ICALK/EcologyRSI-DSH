import assert from "node:assert/strict";
import test from "node:test";

import {
  NativeStageRunner,
  STAGES,
  dshCompatibleSchema,
  jsonDigest,
  skillInvocationEvidence,
} from "../lib/runtime/stage-runner.js";

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

function blockFor(milliseconds) {
  const state = new Int32Array(new SharedArrayBuffer(4));
  Atomics.wait(state, 0, 0, milliseconds);
}

function samplePlanContext() {
  return {
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
}

function samplePlanBinding({ admissionId = "admission-workflow-deadline-1" } = {}) {
  const context = samplePlanContext();
  return {
    run_id: "run-workflow-deadline",
    stage: "sample.plan",
    admission_id: admissionId,
    run_state_revision: 7,
    stage_attempt: 2,
    ledger_expected_revision: 11,
    idempotency_key: "sample-plan-deadline-1",
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
  };
}

function workflowDeadlineHarness({
  startWorkflow,
  timeoutMs = 20,
  runRegistry = { get: () => ({ status: "running" }) },
  persist = async (options) => ({ accepted: true, result_digest: options.body.result_digest }),
  onReservation = () => {},
} = {}) {
  const listeners = new Map();
  let reservationCount = 0;
  const sessions = new Map([[
    "workflow-deadline-child",
    {
      id: "workflow-deadline-child",
      events: skillFirstEvents("origin-vector-forecasting-balanced", { prediction: true }),
    },
  ]]);
  const roleHost = {
    sessionId: "workflow-deadline-parent",
    agent: { id: "workflow-deadline-role-host" },
    binding: { model: "pjlab/deepseek-v4-pro-0813" },
    services: {
      workflowEngine: {
        start: (request) => startWorkflow({ request, listeners, sessions }),
      },
    },
  };
  const runner = new NativeStageRunner({
    on: (name, listener) => {
      listeners.set(name, listener);
      return () => listeners.delete(name);
    },
    sessions: { get: (sessionId) => sessions.get(sessionId) },
    subagents: { start: async () => { throw new Error("Workflow path required"); } },
  }, {
    roleAgents: { get: () => roleHost },
    runRegistry,
    sidecar: {
      request: async (path, options) => {
        if (path.endsWith("/child-reservations")) {
          reservationCount += 1;
          onReservation(options);
          return {
            accepted: true,
            admission_id: options.body.admission_id,
            timeout_ms: options.body.timeout_ms,
            launch: {
              reservation_id: `workflow-deadline-reservation-${reservationCount}`,
              launch_attempt: reservationCount,
            },
            ledger_expected_revision: 12,
          };
        }
        return persist(options);
      },
    },
    structuredStageTimeoutMs: timeoutMs,
    structuredStageMaxAttempts: 1,
    providerStageGate: {
      run: async (_provider, operation) => operation(),
      penalize: () => {},
    },
  });
  return { runner, listeners };
}

function publishWorkflowChild(listeners, request) {
  const agent = { label: request.args.items[0].label, childId: "workflow-deadline-child" };
  listeners.get("workflow/agent-start")?.({ meta: request.meta }, agent);
  listeners.get("workflow/agent-end")?.({ meta: request.meta }, agent);
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

test("pre-score sample critic has a bounded schema-recovery protocol", () => {
  const critic = STAGES["sample.critic"];

  assert.match(critic.instruction, /exact wave_digest/i);
  assert.match(critic.instruction, /every supplied sample_id/i);
  assert.match(critic.instruction, /never call structured_output with empty arguments/i);
  assert.match(critic.instruction, /rejected.*terminate.*fresh.*child attempt/i);
  assert.doesNotMatch(critic.instruction, /retry that same tool/i);
  assert.match(critic.instruction, /emit no prose/i);
});

test("Skill-first evidence permits optional retrieval before the terminal tool", () => {
  const evidence = skillInvocationEvidence([
    {
      seq: 1,
      type: "tool/call",
      data: { callId: "skill", name: "skill", arguments: { name: "autonomous-ecology-research" } },
    },
    {
      seq: 2,
      type: "tool/result",
      data: { message: { content: [{ type: "tool-result", toolCallId: "skill", isError: false }] } },
    },
    {
      seq: 3,
      type: "tool/call",
      data: { callId: "search", name: "web_search", arguments: { queries: ["greenhouse forecast"], retrieval_key: "evidence" } },
    },
    {
      seq: 4,
      type: "tool/result",
      data: { message: { content: [{ type: "tool-result", toolCallId: "search", isError: false }] } },
    },
    {
      seq: 5,
      type: "tool/call",
      data: { callId: "structured", name: "structured_output", arguments: {} },
    },
  ], {
    stage: "generation.research",
    skillName: "autonomous-ecology-research",
    allowDynamicRetrieval: true,
  });

  assert.equal(evidence.first_tool_call_verified, true);
  assert.equal(evidence.next_tool_name, "structured_output");
  assert.equal(evidence.next_tool_call_seq, 5);
});

test("Skill-first evidence rejects excessive or post-terminal retrieval", () => {
  const skillAndResult = [
    {
      seq: 1,
      type: "tool/call",
      data: { callId: "skill", name: "skill", arguments: { name: "autonomous-ecology-research" } },
    },
    {
      seq: 2,
      type: "tool/result",
      data: { message: { content: [{ type: "tool-result", toolCallId: "skill", isError: false }] } },
    },
  ];
  const fourSearches = [];
  for (let index = 0; index < 4; index += 1) {
    fourSearches.push(
      {
        seq: 3 + index * 2,
        type: "tool/call",
        data: { callId: `search-${index}`, name: "web_search", arguments: {} },
      },
      {
        seq: 4 + index * 2,
        type: "tool/result",
        data: { message: { content: [{ type: "tool-result", toolCallId: `search-${index}`, isError: false }] } },
      },
    );
  }
  const options = {
    stage: "generation.research",
    skillName: "autonomous-ecology-research",
    allowDynamicRetrieval: true,
  };
  assert.throws(
    () => skillInvocationEvidence([
      ...skillAndResult,
      { seq: 3, type: "tool/call", data: { callId: "search", name: "web_search", arguments: {} } },
      { seq: 4, type: "tool/result", data: { message: { content: [{ type: "tool-result", toolCallId: "search", isError: false }] } } },
      { seq: 5, type: "tool/call", data: { callId: "structured", name: "structured_output", arguments: {} } },
    ], {
      stage: "generation.research",
      skillName: "autonomous-ecology-research",
    }),
    /legacy tool profile/i,
  );
  assert.throws(
    () => skillInvocationEvidence([
      ...skillAndResult,
      ...fourSearches,
      { seq: 11, type: "tool/call", data: { callId: "structured", name: "structured_output", arguments: {} } },
    ], options),
    /zero to three|retrieval call budget/i,
  );
  assert.throws(
    () => skillInvocationEvidence([
      ...skillAndResult,
      { seq: 3, type: "tool/call", data: { callId: "structured", name: "structured_output", arguments: {} } },
      { seq: 4, type: "tool/call", data: { callId: "late-search", name: "web_search", arguments: {} } },
      { seq: 5, type: "tool/result", data: { message: { content: [{ type: "tool-result", toolCallId: "late-search", isError: false }] } } },
    ], options),
    /before the required terminal tool/i,
  );
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

test("native stage runner rejects over-ceiling structured timers at construction", () => {
  for (const option of [
    { structuredStageTimeoutMs: 1_800_001 },
    { researchStageTimeoutMs: 2_147_483_648 },
    { sampleCriticStageTimeoutMs: 1_800_001 },
  ]) {
    assert.throws(
      () => new NativeStageRunner({}, option),
      /must be at most 1800000/,
    );
  }
});

test("native stage runner reserves before first child tool and durably persists structured output", async () => {
  const roleHost = {
    sessionId: "parent-session",
    agent: { id: "role-host" },
    binding: {
      model: "pjlab/deepseek-v4-pro-0813",
      tool_profile: "dynamic-retrieval-v1",
    },
  };
  const structured = {
    schema_version: "ecologyrsi-dsh.genome-mutation/1",
    operations: [],
  };
  const persisted = [];
  let disposed = false;
  let claimedIdentity;
  let runner;
  let childStartUnixMs;
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
        childStartUnixMs = Date.now();
        blockFor(20);
        assert.equal(provider, "spawn");
        assert.deepEqual(request.prompt[0], { type: "text", text: request.prompt[0].text });
        const prompt = JSON.parse(request.prompt[0].text);
        assert.match(prompt.instruction, /Do not narrate analysis/i);
        assert.match(prompt.instruction, /first response/i);
        assert.match(prompt.instruction, /call skill exactly once/i);
        assert.match(prompt.instruction, /structured_output exactly once/i);
        assert.match(prompt.instruction, /zero to three web_search calls/i);
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
        persisted.push({
          path,
          body: options.body,
          signal: options.signal,
          timeoutMs: options.timeoutMs,
        });
        if (path.endsWith("/child-reservations")) {
          return {
            accepted: true,
            admission_id: options.body.admission_id,
            timeout_ms: options.body.timeout_ms,
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
    structuredStageTimeoutMs: 700_000,
  });
  const context = { parent_genome: { genome_digest: "a".repeat(64) } };
  const result = await runner.run({
    run_id: "run-1",
    stage: "candidate.propose",
    admission_id: "admission-main-1",
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
  assert.deepEqual(claimedIdentity.allowed_tools, ["skill", "web_search"]);
  assert.equal(persisted[0].path, "/api/ecology-agent-sidecar/v1/child-reservations");
  assert.equal(persisted[1].path, "/api/ecology-agent-sidecar/v1/structured-results");
  assert.deepEqual(
    {
      admission_id: persisted[0].body.admission_id,
      run_state_revision: persisted[0].body.run_state_revision,
      stage_attempt: persisted[0].body.stage_attempt,
      timeout_ms: persisted[0].body.timeout_ms,
    },
    {
      admission_id: "admission-main-1",
      run_state_revision: 7,
      stage_attempt: 2,
      timeout_ms: 700_000,
    },
  );
  assert.equal(persisted[1].body.identity.session_id, "child-session");
  assert.equal(persisted[1].body.admission_id, "admission-main-1");
  assert.equal("deadline_unix_ms" in persisted[1].body, false);
  assert.ok(childStartUnixMs > 0);
  assert.ok(persisted[1].signal instanceof AbortSignal);
  assert.ok(persisted[1].timeoutMs > 0 && persisted[1].timeoutMs <= 600_000);
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
    binding: {
      model: "freerouter/gpt-5.6-sol",
      tool_profile: "dynamic-retrieval-v1",
    },
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
            admission_id: options.body.admission_id,
            timeout_ms: options.body.timeout_ms,
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
    admission_id: "admission-retry-1",
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

test("sample critic retries a normally-ended missing result in a fresh child", async () => {
  const roleHost = {
    sessionId: "critic-retry-parent",
    agent: { id: "critic-role-host" },
    binding: { model: "pjlab/deepseek-v4-flash-0731" },
  };
  const waveDigest = "f".repeat(64);
  const structured = {
    schema_version: "ecology-sample-review@1",
    wave_digest: waveDigest,
    decisions: [],
  };
  const criticEvents = skillFirstEvents("origin-vector-review");
  let starts = 0;
  let reservations = 0;
  let persisted = 0;
  let penalties = 0;
  const runner = new NativeStageRunner({
    sessions: {
      get: (id) => (id.startsWith("critic-child-")
        ? { id, events: criticEvents }
        : undefined),
    },
    subagents: {
      start: async () => {
        starts += 1;
        const id = `critic-child-${starts}`;
        return {
          id,
          result: Promise.resolve(
            starts === 1
              ? { stopReason: "completed" }
              : { stopReason: "completed", structured },
          ),
          dispose: async () => {},
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
            admission_id: options.body.admission_id,
            timeout_ms: options.body.timeout_ms,
            launch: {
              reservation_id: `critic-retry-reservation-${reservations}`,
              launch_attempt: reservations,
            },
            ledger_expected_revision: 20 + reservations,
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
  const context = {
    schema_version: "ecologyrsi-dsh.sample-review-wave/1",
    wave_digest: waveDigest,
    samples: [],
  };

  const result = await runner.run({
    run_id: "run-critic-missing-result-retry",
    stage: "sample.critic",
    admission_id: "admission-critic-retry-1",
    run_state_revision: 7,
    stage_attempt: 1,
    ledger_expected_revision: 19,
    idempotency_key: "critic-missing-result-1",
    request: {
      role: "sample-critic",
      output_schema_id: "ecology-sample-review@1",
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

  assert.deepEqual(result.structured, structured);
  assert.equal(starts, 2);
  assert.equal(reservations, 2);
  assert.equal(persisted, 1);
  assert.equal(penalties, 1);
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
    binding: {
      model: "pjlab/deepseek-v4-pro-0813",
      tool_profile: "dynamic-retrieval-v1",
    },
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
            admission_id: options.body.admission_id,
            timeout_ms: options.body.timeout_ms,
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
    admission_id: "admission-workflow-1",
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
    /Then call ecology_execute_prediction_tool exactly once/i,
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

test("sample planner deadline bounds cancel-ignoring Workflow result and disposal", async () => {
  let cancelled = false;
  let disposeCalled = false;
  const never = new Promise(() => {});
  const { runner } = workflowDeadlineHarness({
    startWorkflow: () => ({
      result: never,
      cancel: () => { cancelled = true; },
      dispose: () => {
        disposeCalled = true;
        return never;
      },
    }),
  });
  const runPromise = runner.run(samplePlanBinding());
  const outcome = await Promise.race([
    runPromise.then(
      () => ({ kind: "success" }),
      (error) => ({ kind: "error", error }),
    ),
    new Promise((resolve) => setTimeout(() => resolve({ kind: "guard" }), 160)),
  ]);

  assert.equal(outcome.kind, "error", "Workflow result must not outlive its hard deadline");
  assert.equal(outcome.error?.code, "structured_role_operational_timeout");
  assert.equal(cancelled, true);
  assert.equal(disposeCalled, true);
  assert.equal(runner.activeWorkflows.size, 0);
});

test("sample planner normal cleanup stays active through disposal and drains rejection", async () => {
  let releaseDispose;
  const disposeRelease = new Promise((resolve) => { releaseDispose = resolve; });
  let disposeStarted;
  const startedDisposal = new Promise((resolve) => { disposeStarted = resolve; });
  const structured = {
    schema_version: "ecology-sample-decisions@1",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  const { runner } = workflowDeadlineHarness({
    timeoutMs: 1_000,
    startWorkflow: ({ request, listeners }) => {
      publishWorkflowChild(listeners, request);
      return {
        result: Promise.resolve({ value: [structured], stopReason: "completed" }),
        cancel: () => {},
        dispose: async () => {
          disposeStarted();
          await disposeRelease;
          throw new Error("private workflow disposal failure");
        },
      };
    },
  });
  const runPromise = runner.run(samplePlanBinding());
  await startedDisposal;
  const activeDuringDisposal = runner.activeWorkflows.size;
  releaseDispose();
  const outcome = await runPromise.then(
    (value) => ({ value }),
    (error) => ({ error }),
  );

  assert.equal(activeDuringDisposal, 1);
  assert.equal(outcome.error, undefined);
  assert.deepEqual(outcome.value.structured, structured);
  assert.equal(runner.activeWorkflows.size, 0);
});

test("sample planner classifies an admission boundary crossing as operational timeout", async () => {
  let persistCalls = 0;
  const structured = {
    schema_version: "ecology-sample-decisions@1",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  const { runner } = workflowDeadlineHarness({
    runRegistry: {
      get: () => {
        blockFor(30);
        return { status: "paused" };
      },
    },
    persist: async () => {
      persistCalls += 1;
      return { accepted: true };
    },
    startWorkflow: ({ request, listeners }) => {
      publishWorkflowChild(listeners, request);
      return {
        result: Promise.resolve({ value: [structured], stopReason: "completed" }),
        cancel: () => {},
        dispose: async () => {},
      };
    },
  });

  await assert.rejects(
    runner.run(samplePlanBinding()),
    (error) => error?.code === "structured_role_operational_timeout",
  );
  assert.equal(persistCalls, 0);
});

test("sample planner does not persist when its persistence clone crosses the deadline", async () => {
  let persistCalls = 0;
  let blockClone = true;
  const structured = {
    schema_version: "ecology-sample-decisions@1",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  const originalStructuredClone = globalThis.structuredClone;
  const { runner } = workflowDeadlineHarness({
    persist: async () => {
      persistCalls += 1;
      return { accepted: true };
    },
    startWorkflow: ({ request, listeners }) => {
      publishWorkflowChild(listeners, request);
      return {
        result: Promise.resolve({ value: [structured], stopReason: "completed" }),
        cancel: () => {},
        dispose: async () => {},
      };
    },
  });
  globalThis.structuredClone = (value, options) => {
    if (value === structured && blockClone) {
      blockClone = false;
      blockFor(30);
    }
    return originalStructuredClone(value, options);
  };
  try {
    await assert.rejects(
      runner.run(samplePlanBinding()),
      (error) => error?.code === "structured_role_operational_timeout",
    );
  } finally {
    globalThis.structuredClone = originalStructuredClone;
  }
  assert.equal(persistCalls, 0);
});

test("sample planner rechecks its hard deadline after the final return clone", async () => {
  let structuredCloneCalls = 0;
  const structured = {
    schema_version: "ecology-sample-decisions@1",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  const originalStructuredClone = globalThis.structuredClone;
  const { runner } = workflowDeadlineHarness({
    startWorkflow: ({ request, listeners }) => {
      publishWorkflowChild(listeners, request);
      return {
        result: Promise.resolve({ value: [structured], stopReason: "completed" }),
        cancel: () => {},
        dispose: async () => {},
      };
    },
  });
  globalThis.structuredClone = (value, options) => {
    if (value === structured) {
      structuredCloneCalls += 1;
      if (structuredCloneCalls === 2) blockFor(30);
    }
    return originalStructuredClone(value, options);
  };
  try {
    await assert.rejects(
      runner.run(samplePlanBinding()),
      (error) => error?.code === "structured_role_operational_timeout",
    );
  } finally {
    globalThis.structuredClone = originalStructuredClone;
  }
  assert.equal(structuredCloneCalls, 2);
});

test("sample planner deadline bounds persistence that ignores abort", async () => {
  const never = new Promise(() => {});
  const structured = {
    schema_version: "ecology-sample-decisions@1",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  const { runner } = workflowDeadlineHarness({
    persist: async () => never,
    startWorkflow: ({ request, listeners }) => {
      publishWorkflowChild(listeners, request);
      return {
        result: Promise.resolve({ value: [structured], stopReason: "completed" }),
        cancel: () => {},
        dispose: async () => {},
      };
    },
  });
  const outcome = await Promise.race([
    runner.run(samplePlanBinding()).then(
      () => ({ kind: "success" }),
      (error) => ({ kind: "error", error }),
    ),
    new Promise((resolve) => setTimeout(() => resolve({ kind: "guard" }), 160)),
  ]);

  assert.equal(outcome.kind, "error");
  assert.equal(outcome.error?.code, "structured_role_operational_timeout");
  assert.equal(runner.activeWorkflows.size, 0);
});

test("sample planner retries reuse one absolute local and frozen server deadline", async () => {
  let workflowStarts = 0;
  const reservationTimeouts = [];
  const armedTimeouts = [];
  const structured = {
    schema_version: "ecology-sample-decisions@1",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  const { runner } = workflowDeadlineHarness({
    timeoutMs: 100,
    onReservation: (options) => {
      reservationTimeouts.push(options.timeoutMs);
      armedTimeouts.push(options.body.timeout_ms);
    },
    startWorkflow: ({ request, listeners }) => {
      workflowStarts += 1;
      if (workflowStarts === 1) {
        return {
          result: Promise.resolve({ stopReason: "model_error", error: "private" }),
          cancel: () => {},
          dispose: async () => { blockFor(30); },
        };
      }
      publishWorkflowChild(listeners, request);
      return {
        result: Promise.resolve({ value: [structured], stopReason: "completed" }),
        cancel: () => {},
        dispose: async () => {},
      };
    },
  });
  runner.structuredStageMaxAttempts = 2;

  const result = await runner.run(samplePlanBinding());

  assert.deepEqual(result.structured, structured);
  assert.deepEqual(armedTimeouts, [100, 100]);
  assert.equal(reservationTimeouts.length, 2);
  assert.ok(
    reservationTimeouts[1] <= reservationTimeouts[0] - 20,
    "retry transport must use the first attempt's remaining absolute budget",
  );
});

test("sample critic uses its shorter independent operational timeout", async () => {
  let aborted = false;
  const roleHost = {
    sessionId: "critic-parent-session",
    agent: { id: "critic-role-host" },
    binding: { model: "pjlab/deepseek-v4-flash-0731" },
  };
  const ctx = {
    subagents: {
      start: async (_provider, request) => new Promise((_resolve, reject) => {
        request.signal.addEventListener("abort", () => {
          aborted = true;
          reject(new Error("critic child aborted"));
        }, { once: true });
      }),
    },
  };
  const runner = new NativeStageRunner(ctx, {
    roleAgents: { get: () => roleHost },
    runRegistry: { get: () => ({ status: "running" }) },
    sidecar: {
      request: async (_path, options) => ({
        accepted: true,
        admission_id: options.body.admission_id,
        timeout_ms: options.body.timeout_ms,
        launch: {
          reservation_id: "critic-timeout-reservation",
          launch_attempt: 1,
        },
        ledger_expected_revision: 12,
      }),
    },
    structuredStageTimeoutMs: 1_000,
    sampleCriticStageTimeoutMs: 20,
    structuredStageMaxAttempts: 1,
    providerStageGate: {
      run: async (_provider, operation) => operation(),
      penalize: () => {},
    },
  });
  const context = {
    schema_version: "ecologyrsi-dsh.sample-review-wave/1",
    wave_digest: "f".repeat(64),
    samples: [],
  };
  const startedAt = Date.now();

  await assert.rejects(
    runner.run({
      run_id: "run-critic-timeout",
      stage: "sample.critic",
      admission_id: "admission-critic-timeout-1",
      run_state_revision: 7,
      stage_attempt: 1,
      ledger_expected_revision: 11,
      idempotency_key: "critic-timeout-1",
      request: {
        role: "sample-critic",
        output_schema_id: "ecology-sample-review@1",
        context,
        context_canonical_json: canonicalJson(context),
        context_digest: jsonDigest(context),
        identity_digests: {
          genome_digest: "a".repeat(64),
          compiled_behavior_digest: "b".repeat(64),
          phenotype_instance_digest: "c".repeat(64),
        },
      },
    }),
    /operational timeout/,
  );
  assert.equal(aborted, true);
  assert.ok(Date.now() - startedAt < 500, "sample critic must not inherit the 1s general timeout");
});
