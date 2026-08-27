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

function rc6ConsumedEvents(skillName, {
  prediction = false,
  structuredResult = null,
  terminalKind = "completed",
} = {}) {
  let seq = 1;
  const turn = 1;
  const step = 0;
  const events = [
    { seq: seq++, type: "turn/start", data: { turn } },
    { seq: seq++, type: "step/start", data: { turn, step } },
  ];
  const appendCall = (callId, name, argumentsValue) => {
    const callSeq = seq++;
    events.push({
      seq: callSeq,
      type: "tool/call",
      data: { turn, step, callId, name, arguments: JSON.stringify(argumentsValue) },
    });
    return callSeq;
  };
  const appendResult = (callId, callSeq, { isError = false, error } = {}) => {
    events.push({
      seq: seq++,
      type: "tool/result",
      data: {
        turn,
        step,
        message: {
          content: [{
            type: "tool-result",
            toolCallId: callId,
            isError,
            content: [],
          }],
          role: "tool",
        },
        ...(error === undefined ? {} : { error }),
      },
      sourceEventSeqs: [callSeq],
    });
  };
  const skillSeq = appendCall("skill-call", "skill", { name: skillName });
  appendResult("skill-call", skillSeq);
  if (prediction) {
    const predictionSeq = appendCall(
      "prediction-call",
      "ecology_execute_prediction_tool",
      { tool_id: "ridge@1", wave_digest: "f".repeat(64) },
    );
    appendResult("prediction-call", predictionSeq);
  }
  if (structuredResult !== null) {
    const structuredSeq = appendCall("structured-call", "structured_output", {});
    appendResult("structured-call", structuredSeq, structuredResult);
  }
  events.push(
    { seq: seq++, type: "step/end", data: { turn, step } },
    {
      seq: seq++,
      type: "turn/end",
      data: {
        turn,
        reason: terminalKind === "error"
          ? { kind: "error", error: { message: "provider failed", code: "UPSTREAM" } }
          : { kind: terminalKind },
      },
    },
  );
  return events;
}

function rc6ReusedStructuredCallEvents(skillName) {
  const events = rc6ConsumedEvents(skillName, {
    structuredResult: {
      isError: true,
      error: { name: "ToolArgsError", code: "INVALID_ARGS" },
    },
  });
  const terminal = events.pop();
  const firstStepEnd = events.pop();
  const firstCall = events.find((event) => (
    event.type === "tool/call" && event.data.name === "structured_output"
  ));
  let seq = firstStepEnd.seq;
  events.push(
    firstStepEnd,
    { seq: ++seq, type: "step/start", data: { turn: 1, step: 1 } },
    {
      seq: ++seq,
      type: "tool/call",
      data: {
        turn: 1,
        step: 1,
        callId: firstCall.data.callId,
        name: "structured_output",
        arguments: "{}",
      },
    },
    {
      seq: ++seq,
      type: "tool/result",
      data: {
        turn: 1,
        step: 1,
        message: {
          role: "tool",
          content: [{
            type: "tool-result",
            toolCallId: firstCall.data.callId,
            isError: true,
            content: [],
          }],
        },
      },
      sourceEventSeqs: [firstCall.seq],
    },
    { seq: ++seq, type: "step/end", data: { turn: 1, step: 1 } },
    { ...terminal, seq: ++seq },
  );
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

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function samplePlanContext() {
  return {
    schema_version: "ecologyrsi-dsh.sample-routing-wave/1",
    wave_digest: "f".repeat(64),
    samples: [
      { sample_id: "origin-a" },
      { sample_id: "origin-b" },
    ],
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
  onPenalty = () => {},
  maxAttempts = 1,
} = {}) {
  const listeners = new Map();
  let reservationCount = 0;
  const failures = [];
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
          onReservation(options, reservationCount);
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
        if (path.endsWith("/child-failures")) {
          failures.push(structuredClone(options.body));
          return { accepted: true };
        }
        return persist(options);
      },
    },
    structuredStageTimeoutMs: timeoutMs,
    structuredStageMaxAttempts: maxAttempts,
    providerStageGate: {
      run: async (_provider, operation) => operation(),
      penalize: onPenalty,
    },
  });
  return { runner, listeners, sessions, failures };
}

function publishWorkflowChild(
  listeners,
  request,
  childId = "workflow-deadline-child",
  outcome = "completed",
) {
  const agent = { label: request.args.items[0].label, childId };
  listeners.get("workflow/agent-start")?.({ meta: request.meta }, agent);
  listeners.get("workflow/agent-end")?.(
    { meta: request.meta },
    { ...agent, outcome },
  );
}

function directSampleBinding(stage, context) {
  const reflection = stage === "sample.reflect";
  return {
    run_id: `run-${stage}`,
    stage,
    admission_id: `admission-${stage}-1`,
    run_state_revision: 7,
    stage_attempt: 1,
    ledger_expected_revision: 11,
    idempotency_key: `${stage}-1`,
    request: {
      role: "sample-critic",
      output_schema_id: reflection
        ? "ecology-sample-reflection@1"
        : "ecology-sample-review@1",
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

function directSampleHarness({
  stage,
  results,
  maxAttempts = 2,
  sessionEvents = () => skillFirstEvents("origin-vector-review"),
  failurePhase = null,
  failureCode = null,
}) {
  const starts = [];
  const reservations = [];
  const reservationRequests = [];
  const persisted = [];
  const failures = [];
  const sessions = new Map();
  const roleHost = {
    sessionId: `${stage}-parent`,
    agent: { id: `${stage}-role-host` },
    binding: { model: "pjlab/deepseek-v4-flash-0731" },
  };
  const runner = new NativeStageRunner({
    sessions: { get: (id) => sessions.get(id) },
    subagents: {
      start: async (_provider, request) => {
        const attempt = starts.length + 1;
        const id = `${stage}-child-${attempt}`;
        sessions.set(id, {
          id,
          events: sessionEvents(attempt),
        });
        starts.push({ id, request });
        if (failurePhase === "start") {
          const error = new Error(`private ${failurePhase} failure`);
          error.code = failureCode;
          throw error;
        }
        const result = failurePhase === "result"
          ? Promise.reject(Object.assign(
            new Error(`private ${failurePhase} failure`),
            { code: failureCode },
          ))
          : Promise.resolve(results[attempt - 1]);
        return {
          id,
          result,
          dispose: async () => {},
        };
      },
    },
  }, {
    roleAgents: { get: () => roleHost },
    runRegistry: {
      get: () => {
        if (failurePhase === "admission") {
          const error = new Error(`private ${failurePhase} failure`);
          error.code = failureCode;
          throw error;
        }
        return { status: "running" };
      },
    },
    sidecar: {
      request: async (path, options) => {
        if (path.endsWith("/child-reservations")) {
          reservationRequests.push(structuredClone(options.body));
          const attempt = reservations.length + 1;
          reservations.push({
            reservation_id: `${stage}-reservation-${attempt}`,
            launch_attempt: attempt,
          });
          return {
            accepted: true,
            admission_id: options.body.admission_id,
            timeout_ms: options.body.timeout_ms,
            launch: reservations.at(-1),
            ledger_expected_revision: 20 + attempt,
          };
        }
        if (path.endsWith("/child-failures")) {
          failures.push(structuredClone(options.body));
          return { accepted: true };
        }
        persisted.push(options.body);
        if (failurePhase === "persistence") {
          const error = new Error(`private ${failurePhase} failure`);
          error.code = failureCode;
          throw error;
        }
        return { accepted: true, result_digest: options.body.result_digest };
      },
    },
    structuredStageMaxAttempts: maxAttempts,
    providerStageGate: {
      run: async (_provider, operation) => operation(),
      penalize: () => {},
    },
  });
  return { runner, starts, reservations, reservationRequests, persisted, failures };
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
  let failures = 0;
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
        if (path.endsWith("/child-failures")) {
          failures += 1;
          return { accepted: true };
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
  assert.equal(failures, 1);
  assert.equal(penalties, 1);
  assert.deepEqual(disposed, ["child-1", "child-2"]);
});

test("sample critic retries a consumed completed turn with no capture in a fresh child", async () => {
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
  let starts = 0;
  let reservations = 0;
  let persisted = 0;
  let failures = 0;
  let penalties = 0;
  const childRequests = [];
  const runner = new NativeStageRunner({
    sessions: {
      get: (id) => (id.startsWith("critic-child-")
        ? {
          id,
          events: id.endsWith("-1")
            ? rc6ConsumedEvents("origin-vector-review")
            : skillFirstEvents("origin-vector-review"),
        }
        : undefined),
    },
    subagents: {
      start: async (_provider, childRequest) => {
        starts += 1;
        const id = `critic-child-${starts}`;
        childRequests.push(childRequest);
        return {
          id,
          result: Promise.resolve(
            starts === 1
              ? { stopReason: "error" }
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
        if (path.endsWith("/child-failures")) {
          failures += 1;
          return { accepted: true };
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
    samples: [
      { sample_id: "origin-a" },
      { sample_id: "origin-b" },
    ],
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
  assert.equal(failures, 1);
  assert.equal(penalties, 1);
  for (const childRequest of childRequests) {
    assert.deepEqual(childRequest.outputSchema.properties.wave_digest, {
      type: "string",
      const: waveDigest,
    });
    assert.deepEqual(
      childRequest.outputSchema.properties.decisions.items.properties.sample_id,
      { type: "string", enum: ["origin-a", "origin-b"] },
    );
    assert.match(
      JSON.parse(childRequest.prompt[0].text).instruction,
      /copy.*exact.*Host.*sample_id/i,
    );
  }
});

test("sample reflection binds the outer Host identity and retries one missing result", async () => {
  const waveDigest = "e".repeat(64);
  const structured = {
    schema_version: "ecology-sample-reflection@1",
    wave_digest: waveDigest,
    sample_id: "origin-reflection",
    outcome_class: "neutral",
    error_source: "unknown",
    next_action: "keep",
    confidence: 0.8,
    summary: "Keep the bounded configuration.",
  };
  const context = {
    schema_version: "ecologyrsi-dsh.sample-reflection-context/1",
    wave_digest: waveDigest,
    sample: {
      sample_id: "origin-reflection",
      prediction_cells: [{ sample_id: "prediction-cell-that-must-not-bind" }],
    },
    outcome: { cells: [] },
  };
  const harness = directSampleHarness({
    stage: "sample.reflect",
    results: [
      { stopReason: "error" },
      { stopReason: "completed", structured },
    ],
    sessionEvents: (attempt) => attempt === 1
      ? rc6ConsumedEvents("origin-vector-review")
      : skillFirstEvents("origin-vector-review"),
  });

  const result = await harness.runner.run(directSampleBinding("sample.reflect", context));

  assert.deepEqual(result.structured, structured);
  assert.deepEqual(harness.reservations, [
    { reservation_id: "sample.reflect-reservation-1", launch_attempt: 1 },
    { reservation_id: "sample.reflect-reservation-2", launch_attempt: 2 },
  ]);
  assert.deepEqual(
    harness.reservationRequests.map((request) => request.sample_member_digests),
    [
      [jsonDigest("prediction-cell-that-must-not-bind")],
      [jsonDigest("prediction-cell-that-must-not-bind")],
    ],
  );
  assert.deepEqual(harness.starts.map(({ id }) => id), [
    "sample.reflect-child-1",
    "sample.reflect-child-2",
  ]);
  assert.equal(harness.persisted.length, 1);
  for (const { request } of harness.starts) {
    assert.deepEqual(request.outputSchema.properties.wave_digest, {
      type: "string",
      const: waveDigest,
    });
    assert.deepEqual(request.outputSchema.properties.sample_id, {
      type: "string",
      const: "origin-reflection",
    });
    assert.notEqual(
      request.outputSchema.properties.sample_id.const,
      "prediction-cell-that-must-not-bind",
    );
    assert.match(
      JSON.parse(request.prompt[0].text).instruction,
      /prediction_cells\[\]\.sample_id.*must not|never.*prediction_cells\[\]\.sample_id/i,
    );
  }
});

test("sample reflection stops after two missing structured outputs without persistence", async () => {
  const context = {
    schema_version: "ecologyrsi-dsh.sample-reflection-context/1",
    wave_digest: "d".repeat(64),
    sample: { sample_id: "origin-two-missing" },
    outcome: { cells: [] },
  };
  const harness = directSampleHarness({
    stage: "sample.reflect",
    results: [
      { stopReason: "error" },
      { stopReason: "error" },
    ],
    sessionEvents: () => rc6ConsumedEvents("origin-vector-review"),
  });

  await assert.rejects(
    harness.runner.run(directSampleBinding("sample.reflect", context)),
    (error) => error?.code === "structured_result_missing",
  );
  assert.equal(harness.reservations.length, 2);
  assert.equal(harness.starts.length, 2);
  assert.equal(harness.persisted.length, 0);
});

test("direct sample exact INVALID_ARGS rejection is a bounded missing-capture retry", async () => {
  const context = {
    schema_version: "ecologyrsi-dsh.sample-reflection-context/1",
    wave_digest: "a".repeat(64),
    sample: { sample_id: "origin-direct-schema-rejection" },
    outcome: { cells: [] },
  };
  const harness = directSampleHarness({
    stage: "sample.reflect",
    results: [
      { stopReason: "error" },
      { stopReason: "error" },
    ],
    sessionEvents: () => rc6ConsumedEvents("origin-vector-review", {
      structuredResult: {
        isError: true,
        error: { name: "ToolArgsError", code: "INVALID_ARGS" },
      },
    }),
  });

  await assert.rejects(
    harness.runner.run(directSampleBinding("sample.reflect", context)),
    (error) => error?.code === "structured_result_missing",
  );
  assert.equal(harness.reservations.length, 2);
  assert.equal(harness.starts.length, 2);
  assert.equal(harness.persisted.length, 0);
});

test("Workflow completed null with failed child exact INVALID_ARGS is a bounded missing-capture retry", async () => {
  let workflowStarts = 0;
  let persistCalls = 0;
  const reservations = [];
  const { runner } = workflowDeadlineHarness({
    timeoutMs: 1_000,
    maxAttempts: 2,
    onReservation: (_options, attempt) => reservations.push(attempt),
    persist: async () => {
      persistCalls += 1;
      return { accepted: true };
    },
    startWorkflow: ({ request, listeners, sessions }) => {
      workflowStarts += 1;
      const childId = `workflow-schema-rejected-child-${workflowStarts}`;
      sessions.set(childId, {
        id: childId,
        events: rc6ConsumedEvents("origin-vector-forecasting-balanced", {
          prediction: true,
          structuredResult: {
            isError: true,
            error: { name: "ToolArgsError", code: "INVALID_ARGS" },
          },
        }),
      });
      publishWorkflowChild(listeners, request, childId, "failed");
      return {
        result: Promise.resolve({ stopReason: "completed", value: [null] }),
        cancel: () => {},
        dispose: async () => {},
      };
    },
  });

  await assert.rejects(
    runner.run(samplePlanBinding()),
    (error) => error?.code === "structured_result_missing",
  );
  assert.deepEqual(reservations, [1, 2]);
  assert.equal(workflowStarts, 2);
  assert.equal(persistCalls, 0);
});

test("direct and Workflow provider terminal errors are non-retryable model failures", async () => {
  const context = {
    schema_version: "ecologyrsi-dsh.sample-reflection-context/1",
    wave_digest: "9".repeat(64),
    sample: { sample_id: "origin-model-error" },
    outcome: { cells: [] },
  };
  const direct = directSampleHarness({
    stage: "sample.reflect",
    results: [{ stopReason: "error", output: [{ type: "text", text: "private transport failure" }] }],
    sessionEvents: () => rc6ConsumedEvents("origin-vector-review", {
      structuredResult: {
        isError: true,
        error: { name: "ToolArgsError", code: "INVALID_ARGS" },
      },
      terminalKind: "error",
    }),
  });
  await assert.rejects(
    direct.runner.run(directSampleBinding("sample.reflect", context)),
    (error) => error?.code === "structured_child_model_error",
  );
  assert.equal(direct.reservations.length, 1);
  assert.equal(direct.starts.length, 1);
  assert.equal(direct.persisted.length, 0);
  assert.equal(direct.failures.length, 1);

  let workflowStarts = 0;
  let workflowPenalties = 0;
  const workflowReservations = [];
  const workflow = workflowDeadlineHarness({
    timeoutMs: 1_000,
    maxAttempts: 2,
    onPenalty: () => { workflowPenalties += 1; },
    onReservation: (_options, attempt) => workflowReservations.push(attempt),
    startWorkflow: ({ request, listeners, sessions }) => {
      workflowStarts += 1;
      const childId = `workflow-provider-error-${workflowStarts}`;
      sessions.set(childId, {
        id: childId,
        events: rc6ConsumedEvents("origin-vector-forecasting-balanced", {
          prediction: true,
          terminalKind: "error",
        }),
      });
      publishWorkflowChild(listeners, request, childId, "failed");
      return {
        result: Promise.resolve({ stopReason: "completed", value: [null] }),
        cancel: () => {},
        dispose: async () => {},
      };
    },
  });
  await assert.rejects(
    workflow.runner.run(samplePlanBinding()),
    (error) => error?.code === "structured_child_model_error",
  );
  assert.deepEqual(workflowReservations, [1]);
  assert.equal(workflowStarts, 1);
  assert.equal(workflowPenalties, 1);
  assert.equal(workflow.failures.length, 1);
});

test("an unconsumed completed no-op turn cannot hide the prior consumed provider error", async () => {
  const context = {
    schema_version: "ecologyrsi-dsh.sample-reflection-context/1",
    wave_digest: "5".repeat(64),
    sample: { sample_id: "origin-consumed-terminal" },
    outcome: { cells: [] },
  };
  const harness = directSampleHarness({
    stage: "sample.reflect",
    results: [{ stopReason: "error" }],
    sessionEvents: () => {
      const events = rc6ConsumedEvents("origin-vector-review", { terminalKind: "error" });
      const seq = events.at(-1).seq;
      events.push(
        { seq: seq + 1, type: "turn/start", data: { turn: 2 } },
        {
          seq: seq + 2,
          type: "turn/end",
          data: { turn: 2, reason: { kind: "completed" } },
        },
      );
      return events;
    },
  });

  await assert.rejects(
    harness.runner.run(directSampleBinding("sample.reflect", context)),
    (error) => error?.code === "structured_child_model_error",
  );
  assert.equal(harness.reservations.length, 1);
  assert.equal(harness.starts.length, 1);
  assert.equal(harness.persisted.length, 0);
});

test("completed child events cannot turn a public model_error into missing capture", async () => {
  const context = {
    schema_version: "ecologyrsi-dsh.sample-reflection-context/1",
    wave_digest: "4".repeat(64),
    sample: { sample_id: "origin-public-model-error" },
    outcome: { cells: [] },
  };
  const harness = directSampleHarness({
    stage: "sample.reflect",
    results: [
      { stopReason: "model_error" },
      { stopReason: "model_error" },
    ],
    sessionEvents: () => rc6ConsumedEvents("origin-vector-review"),
  });

  await assert.rejects(
    harness.runner.run(directSampleBinding("sample.reflect", context)),
    (error) => error?.code === "structured_child_model_error",
  );
  assert.equal(harness.reservations.length, 2);
  assert.equal(harness.starts.length, 2);
  assert.equal(harness.persisted.length, 0);
});

test("direct structured-output authorization, abort, and unknown-tool failures never retry as missing", async () => {
  const context = {
    schema_version: "ecologyrsi-dsh.sample-reflection-context/1",
    wave_digest: "7".repeat(64),
    sample: { sample_id: "origin-control-rejection" },
    outcome: { cells: [] },
  };
  const rejectedErrors = [
    undefined,
    { name: "AbortError", code: "ABORTED" },
    { name: "ToolNotFoundError", code: "UNKNOWN_TOOL" },
  ];

  for (const error of rejectedErrors) {
    const harness = directSampleHarness({
      stage: "sample.reflect",
      results: [{ stopReason: "error" }],
      sessionEvents: () => rc6ConsumedEvents("origin-vector-review", {
        structuredResult: { isError: true, error },
      }),
    });
    await assert.rejects(
      harness.runner.run(directSampleBinding("sample.reflect", context)),
      (caught) => caught?.code === "structured_child_model_error",
      error?.code || "authorization",
    );
    assert.equal(harness.reservations.length, 1, error?.code || "authorization");
    assert.equal(harness.starts.length, 1, error?.code || "authorization");
    assert.equal(harness.persisted.length, 0, error?.code || "authorization");
  }
});

test("direct reused callId and mismatched source cannot authorize a missing retry", async () => {
  const context = {
    schema_version: "ecologyrsi-dsh.sample-reflection-context/1",
    wave_digest: "6".repeat(64),
    sample: { sample_id: "origin-reused-call" },
    outcome: { cells: [] },
  };
  const harness = directSampleHarness({
    stage: "sample.reflect",
    results: [{ stopReason: "error" }],
    sessionEvents: () => rc6ReusedStructuredCallEvents("origin-vector-review"),
  });

  await assert.rejects(
    harness.runner.run(directSampleBinding("sample.reflect", context)),
    (error) => error?.code === "structured_child_model_error",
  );
  assert.equal(harness.reservations.length, 1);
  assert.equal(harness.starts.length, 1);
  assert.equal(harness.persisted.length, 0);
});

test("Workflow completed null with failed child no-call completion retries as missing", async () => {
  let starts = 0;
  const reservations = [];
  const harness = workflowDeadlineHarness({
    timeoutMs: 1_000,
    maxAttempts: 2,
    onReservation: (_options, attempt) => reservations.push(attempt),
    startWorkflow: ({ request, listeners, sessions }) => {
      starts += 1;
      const childId = `workflow-no-capture-${starts}`;
      sessions.set(childId, {
        id: childId,
        events: rc6ConsumedEvents("origin-vector-forecasting-balanced", { prediction: true }),
      });
      publishWorkflowChild(listeners, request, childId, "failed");
      return {
        result: Promise.resolve({ stopReason: "completed", value: [null] }),
        cancel: () => {},
        dispose: async () => {},
      };
    },
  });

  await assert.rejects(
    harness.runner.run(samplePlanBinding()),
    (error) => error?.code === "structured_result_missing",
  );
  assert.deepEqual(reservations, [1, 2]);
  assert.equal(starts, 2);
});

test("Workflow completed null with authorization-rejected child is non-missing and non-retryable", async () => {
  let starts = 0;
  const reservations = [];
  const harness = workflowDeadlineHarness({
    timeoutMs: 1_000,
    maxAttempts: 2,
    onReservation: (_options, attempt) => reservations.push(attempt),
    startWorkflow: ({ request, listeners, sessions }) => {
      starts += 1;
      const childId = `workflow-authorization-${starts}`;
      sessions.set(childId, {
        id: childId,
        events: rc6ConsumedEvents("origin-vector-forecasting-balanced", {
          prediction: true,
          structuredResult: { isError: true },
        }),
      });
      publishWorkflowChild(listeners, request, childId, "failed");
      return {
        result: Promise.resolve({ stopReason: "completed", value: [null] }),
        cancel: () => {},
        dispose: async () => {},
      };
    },
  });

  await assert.rejects(
    harness.runner.run(samplePlanBinding()),
    (error) => error?.code === "structured_child_model_error",
  );
  assert.deepEqual(reservations, [1]);
  assert.equal(starts, 1);
});

test("Workflow completed null cannot infer missing from a non-failed child outcome", async () => {
  let starts = 0;
  const reservations = [];
  const harness = workflowDeadlineHarness({
    timeoutMs: 1_000,
    maxAttempts: 2,
    onReservation: (_options, attempt) => reservations.push(attempt),
    startWorkflow: ({ request, listeners, sessions }) => {
      starts += 1;
      const childId = `workflow-nonfailed-null-${starts}`;
      sessions.set(childId, {
        id: childId,
        events: rc6ConsumedEvents("origin-vector-forecasting-balanced", { prediction: true }),
      });
      publishWorkflowChild(listeners, request, childId, "completed");
      return {
        result: Promise.resolve({ stopReason: "completed", value: [null] }),
        cancel: () => {},
        dispose: async () => {},
      };
    },
  });

  await assert.rejects(
    harness.runner.run(samplePlanBinding()),
    (error) => error?.code === "structured_child_model_error",
  );
  assert.deepEqual(reservations, [1]);
  assert.equal(starts, 1);
});

test("Workflow failed child capture evidence cannot authorize a non-null-batch shape", async () => {
  let starts = 0;
  const reservations = [];
  const harness = workflowDeadlineHarness({
    timeoutMs: 1_000,
    maxAttempts: 2,
    onReservation: (_options, attempt) => reservations.push(attempt),
    startWorkflow: ({ request, listeners, sessions }) => {
      starts += 1;
      const childId = `workflow-wrong-batch-${starts}`;
      sessions.set(childId, {
        id: childId,
        events: rc6ConsumedEvents("origin-vector-forecasting-balanced", { prediction: true }),
      });
      publishWorkflowChild(listeners, request, childId, "failed");
      return {
        result: Promise.resolve({ stopReason: "completed", value: [] }),
        cancel: () => {},
        dispose: async () => {},
      };
    },
  });

  await assert.rejects(
    harness.runner.run(samplePlanBinding()),
    (error) => error?.code === "structured_child_model_error",
  );
  assert.deepEqual(reservations, [1]);
  assert.equal(starts, 1);
});

test("Workflow cancelled, error, unknown, and legacy aborted stops are exact nonretry boundaries", async () => {
  const cases = [
    { stopReason: "cancelled", expectedCode: "structured_child_aborted" },
    { stopReason: "error", expectedCode: "structured_child_model_error" },
    { stopReason: "future-stop-token", expectedCode: "structured_child_model_error" },
    { stopReason: "aborted", expectedCode: "structured_child_aborted" },
  ];

  for (const { stopReason, expectedCode } of cases) {
    let starts = 0;
    let penalties = 0;
    let persistCalls = 0;
    const reservations = [];
    const harness = workflowDeadlineHarness({
      timeoutMs: 1_000,
      maxAttempts: 2,
      onReservation: (_options, attempt) => reservations.push(attempt),
      onPenalty: () => { penalties += 1; },
      persist: async () => {
        persistCalls += 1;
        return { accepted: true };
      },
      startWorkflow: () => {
        starts += 1;
        return {
          result: Promise.resolve({ value: null, stopReason, agentsStarted: 0 }),
          cancel: () => {},
          dispose: async () => {},
        };
      },
    });

    const error = await harness.runner.run(samplePlanBinding()).then(
      () => null,
      (caught) => caught,
    );
    assert.deepEqual({
      code: error?.code,
      reservations,
      starts,
      penalties,
      persistCalls,
    }, {
      code: expectedCode,
      reservations: [1],
      starts: 1,
      penalties: expectedCode === "structured_child_model_error" ? 1 : 0,
      persistCalls: 0,
    }, stopReason);
  }
});

test("direct phase causes cannot spoof either retry allowlist code", async () => {
  const context = {
    schema_version: "ecologyrsi-dsh.sample-reflection-context/1",
    wave_digest: "8".repeat(64),
    sample: { sample_id: "origin-phase-spoof" },
    outcome: { cells: [] },
  };
  const structured = {
    schema_version: "ecology-sample-reflection@1",
    wave_digest: "8".repeat(64),
    sample_id: "origin-phase-spoof",
    outcome_class: "neutral",
    error_source: "unknown",
    next_action: "keep",
    confidence: 0.8,
    summary: "Keep the bounded configuration.",
  };
  const expectedPhaseCode = {
    start: "structured_child_start_failed",
    result: "structured_child_result_failed",
    admission: "structured_result_admission_failed",
    persistence: "structured_result_persist_failed",
  };

  for (const phase of Object.keys(expectedPhaseCode)) {
    for (const publicCode of [
      "structured_result_missing",
      "structured_child_model_error",
    ]) {
      const harness = directSampleHarness({
        stage: "sample.reflect",
        results: [{ stopReason: "completed", structured }],
        failurePhase: phase,
        failureCode: publicCode,
      });
      await assert.rejects(
        harness.runner.run(directSampleBinding("sample.reflect", context)),
        (error) => error?.code === expectedPhaseCode[phase]
          && error?.cause === undefined
          && !String(error).includes("private"),
        `${phase}:${publicCode}`,
      );
      assert.equal(harness.reservations.length, 1, `${phase}:${publicCode}`);
      assert.equal(harness.starts.length, 1, `${phase}:${publicCode}`);
      assert.equal(
        harness.persisted.length,
        phase === "persistence" ? 1 : 0,
        `${phase}:${publicCode}`,
      );
    }
  }
});

test("Workflow phase causes cannot spoof retry or duplicate a post-commit persist", async () => {
  const structured = {
    schema_version: "ecology-sample-decisions@1",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  const expectedPhaseCode = {
    start: "structured_child_start_failed",
    result: "structured_child_result_failed",
    admission: "structured_result_admission_failed",
    persistence: "structured_result_persist_failed",
  };

  for (const phase of Object.keys(expectedPhaseCode)) {
    for (const publicCode of [
      "structured_result_missing",
      "structured_child_model_error",
    ]) {
      let workflowStarts = 0;
      let persistCalls = 0;
      const reservations = [];
      const codedCause = () => Object.assign(
        new Error(`private ${phase} failure after possible external effect`),
        { code: publicCode },
      );
      const runRegistry = {
        get: () => {
          if (phase === "admission") throw codedCause();
          return { status: "running" };
        },
      };
      const harness = workflowDeadlineHarness({
        timeoutMs: 1_000,
        maxAttempts: 2,
        runRegistry,
        onReservation: (_options, attempt) => reservations.push(attempt),
        persist: async () => {
          persistCalls += 1;
          if (phase === "persistence") throw codedCause();
          return { accepted: true };
        },
        startWorkflow: ({ request, listeners, sessions }) => {
          workflowStarts += 1;
          if (phase === "start") throw codedCause();
          const childId = `workflow-${phase}-child-${workflowStarts}`;
          sessions.set(childId, {
            id: childId,
            events: skillFirstEvents(
              "origin-vector-forecasting-balanced",
              { prediction: true },
            ),
          });
          publishWorkflowChild(listeners, request, childId);
          return {
            result: phase === "result"
              ? Promise.reject(codedCause())
              : Promise.resolve({ value: [structured], stopReason: "completed" }),
            cancel: () => {},
            dispose: async () => {},
          };
        },
      });

      await assert.rejects(
        harness.runner.run(samplePlanBinding()),
        (error) => error?.code === expectedPhaseCode[phase]
          && error?.cause === undefined
          && !String(error).includes("private"),
        `${phase}:${publicCode}`,
      );
      assert.deepEqual(reservations, [1], `${phase}:${publicCode}`);
      assert.equal(workflowStarts, 1, `${phase}:${publicCode}`);
      assert.equal(
        persistCalls,
        phase === "persistence" ? 1 : 0,
        `${phase}:${publicCode}`,
      );
    }
  }
});

test("sample schema specialization is isolated to each schema clone", async () => {
  const firstWave = "1".repeat(64);
  const secondWave = "2".repeat(64);
  const harness = directSampleHarness({
    stage: "sample.critic",
    results: [
      {
        stopReason: "completed",
        structured: {
          schema_version: "ecology-sample-review@1",
          wave_digest: firstWave,
          decisions: [],
        },
      },
      {
        stopReason: "completed",
        structured: {
          schema_version: "ecology-sample-review@1",
          wave_digest: secondWave,
          decisions: [],
        },
      },
    ],
  });

  await harness.runner.run(directSampleBinding("sample.critic", {
    schema_version: "ecologyrsi-dsh.sample-review-wave/1",
    wave_digest: firstWave,
    samples: [{ sample_id: "origin-first" }],
  }));
  await harness.runner.run(directSampleBinding("sample.critic", {
    schema_version: "ecologyrsi-dsh.sample-review-wave/1",
    wave_digest: secondWave,
    samples: [{ sample_id: "origin-second" }],
  }));

  assert.deepEqual(
    harness.starts.map(({ request }) => ({
      wave: request.outputSchema.properties.wave_digest.const,
      sampleIds: request.outputSchema.properties.decisions.items.properties.sample_id.enum,
    })),
    [
      { wave: firstWave, sampleIds: ["origin-first"] },
      { wave: secondWave, sampleIds: ["origin-second"] },
    ],
  );
});

test("malformed sample Host identities fail locally before child launch or persistence", async () => {
  const validWave = "c".repeat(64);
  const invalidContexts = [
    {
      name: "uppercase wave digest",
      context: {
        schema_version: "ecologyrsi-dsh.sample-review-wave/1",
        wave_digest: "C".repeat(64),
        samples: [{ sample_id: "origin-a" }],
      },
    },
    {
      name: "empty sample set",
      context: {
        schema_version: "ecologyrsi-dsh.sample-review-wave/1",
        wave_digest: validWave,
        samples: [],
      },
    },
    {
      name: "empty sample id",
      context: {
        schema_version: "ecologyrsi-dsh.sample-review-wave/1",
        wave_digest: validWave,
        samples: [{ sample_id: "" }],
      },
    },
    {
      name: "duplicate sample ids",
      context: {
        schema_version: "ecologyrsi-dsh.sample-review-wave/1",
        wave_digest: validWave,
        samples: [{ sample_id: "origin-a" }, { sample_id: "origin-a" }],
      },
    },
    {
      name: "overlong sample id",
      context: {
        schema_version: "ecologyrsi-dsh.sample-review-wave/1",
        wave_digest: validWave,
        samples: [{ sample_id: "x".repeat(241) }],
      },
    },
    {
      name: "too many sample ids",
      context: {
        schema_version: "ecologyrsi-dsh.sample-review-wave/1",
        wave_digest: validWave,
        samples: Array.from({ length: 129 }, (_, index) => ({
          sample_id: `origin-${index}`,
        })),
      },
    },
  ];

  for (const { name, context } of invalidContexts) {
    const harness = directSampleHarness({
      stage: "sample.critic",
      results: [{ stopReason: "completed" }],
    });
    await assert.rejects(
      harness.runner.run(directSampleBinding("sample.critic", context)),
      (error) => error?.code === "sample_stage_context_invalid",
      name,
    );
    assert.equal(harness.starts.length, 0, name);
    assert.equal(harness.persisted.length, 0, name);
    assert.equal(harness.reservations.length, 1, name);
  }

  const malformedReflection = directSampleHarness({
    stage: "sample.reflect",
    results: [{ stopReason: "completed" }],
  });
  await assert.rejects(
    malformedReflection.runner.run(directSampleBinding("sample.reflect", {
      schema_version: "ecologyrsi-dsh.sample-reflection-context/1",
      wave_digest: validWave,
      sample: {
        prediction_cells: [{ sample_id: "prediction-cell-is-not-the-origin" }],
      },
      outcome: { cells: [] },
    })),
    (error) => error?.code === "sample_stage_context_invalid",
  );
  assert.equal(malformedReflection.starts.length, 0);
  assert.equal(malformedReflection.persisted.length, 0);
  assert.equal(malformedReflection.reservations.length, 1);
});

test("sample missing-output retry does not broaden to lifecycle or durable-boundary errors", async () => {
  const nonRetryableCodes = [
    "structured_child_aborted",
    "structured_child_start_failed",
    "structured_child_result_failed",
    "structured_result_admission_closed",
    "structured_role_operational_timeout",
    "dsh_tool_authorization_error",
    "provider_stage_admission_closed",
    "structured_result_not_accepted",
    "structured_result_persist_failed",
    "dsh_native_runtime_http_error",
  ];
  const context = {
    schema_version: "ecologyrsi-dsh.sample-reflection-context/1",
    wave_digest: "b".repeat(64),
    sample: { sample_id: "origin-no-retry" },
    outcome: { cells: [] },
  };

  for (const code of nonRetryableCodes) {
    let gateCalls = 0;
    const runner = new NativeStageRunner({}, {
      roleAgents: {
        get: () => ({
          sessionId: "sample-no-retry-parent",
          agent: { id: "sample-no-retry-role-host" },
          binding: { model: "pjlab/deepseek-v4-flash-0731" },
        }),
      },
      runRegistry: { get: () => ({ status: "running" }) },
      structuredStageMaxAttempts: 2,
      providerStageGate: {
        run: async () => {
          gateCalls += 1;
          const error = new Error(code);
          error.code = code;
          throw error;
        },
        penalize: () => {},
      },
    });
    await assert.rejects(
      runner.run(directSampleBinding("sample.reflect", context)),
      (error) => error?.code === code,
      code,
    );
    assert.equal(gateCalls, 1, code);
  }
});

test("sample planner retries a missing Workflow result with a fresh reservation and session", async () => {
  let workflowStarts = 0;
  let persistCalls = 0;
  const reservations = [];
  const workflowRequests = [];
  const childIds = [];
  const structured = {
    schema_version: "ecology-sample-decisions@1",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  const { runner } = workflowDeadlineHarness({
    timeoutMs: 1_000,
    maxAttempts: 2,
    onReservation: (_options, attempt) => reservations.push(attempt),
    persist: async (options) => {
      persistCalls += 1;
      return { accepted: true, result_digest: options.body.result_digest };
    },
    startWorkflow: ({ request, listeners, sessions }) => {
      workflowStarts += 1;
      workflowRequests.push(request);
      const childId = `workflow-retry-child-${workflowStarts}`;
      childIds.push(childId);
      sessions.set(childId, {
        id: childId,
        events: workflowStarts === 1
          ? rc6ConsumedEvents("origin-vector-forecasting-balanced", { prediction: true })
          : skillFirstEvents("origin-vector-forecasting-balanced", { prediction: true }),
      });
      publishWorkflowChild(
        listeners,
        request,
        childId,
        workflowStarts === 1 ? "failed" : "completed",
      );
      return {
        result: Promise.resolve(workflowStarts === 1
          ? { value: [null], stopReason: "completed" }
          : { value: [structured], stopReason: "completed" }),
        cancel: () => {},
        dispose: async () => {},
      };
    },
  });

  const result = await runner.run(samplePlanBinding());

  assert.deepEqual(result.structured, structured);
  assert.deepEqual(reservations, [1, 2]);
  assert.deepEqual(childIds, ["workflow-retry-child-1", "workflow-retry-child-2"]);
  assert.equal(workflowStarts, 2);
  assert.equal(persistCalls, 1);
  for (const request of workflowRequests) {
    const item = request.args.items[0];
    assert.deepEqual(item.schema.properties.wave_digest, {
      type: "string",
      const: "f".repeat(64),
    });
    assert.deepEqual(item.schema.properties.decisions.items.properties.sample_id, {
      type: "string",
      enum: ["origin-a", "origin-b"],
    });
    assert.match(JSON.parse(item.prompt).instruction, /copy.*exact.*Host.*sample_id/i);
  }
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
    samples: [
      { sample_id: "origin-a" },
      { sample_id: "origin-b" },
    ],
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
  assert.deepEqual(workflowRequest.args.items[0].schema.properties.wave_digest, {
    type: "string",
    const: "f".repeat(64),
  });
  assert.deepEqual(
    workflowRequest.args.items[0].schema.properties.decisions.items.properties.sample_id,
    { type: "string", enum: ["origin-a", "origin-b"] },
  );
  assert.deepEqual(
    persisted[0].body.sample_member_digests,
    [jsonDigest("origin-a"), jsonDigest("origin-b")].sort(),
  );
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

test("sample planner hard deadline releases a drain waiting on its cancel-ignoring Workflow", { timeout: 1_000 }, async () => {
  const workflowStarted = deferred();
  const never = new Promise(() => {});
  const { runner } = workflowDeadlineHarness({
    startWorkflow: () => {
      workflowStarted.resolve();
      return {
        result: never,
        cancel: () => {},
        dispose: () => never,
      };
    },
  });
  const role = runner.run(samplePlanBinding());
  await workflowStarted.promise;
  const draining = runner.quiesceRun("run-workflow-deadline");

  const roleOutcome = await role.then(
    () => ({ status: "fulfilled" }),
    (error) => ({ status: "rejected", error }),
  );
  const drainOutcome = await Promise.race([
    draining.then(() => "drained"),
    new Promise((resolve) => setTimeout(() => resolve("still-pending"), 100)),
  ]);

  assert.equal(roleOutcome.status, "rejected");
  assert.equal(roleOutcome.error.code, "structured_role_operational_timeout");
  assert.equal(drainOutcome, "drained");
  assert.equal(runner.activeWorkflows.size, 0);
  assert.equal(runner.pendingStarts.size, 0);
});

test("reentrant Workflow quiescence rescans the active publication gap before returning", { timeout: 1_000 }, async () => {
  const never = new Promise(() => {});
  const workflowStartEntered = deferred();
  let harness;
  let draining;
  let cancels = 0;
  let disposals = 0;
  harness = workflowDeadlineHarness({
    startWorkflow: () => {
      draining = harness.runner.quiesceRun("run-workflow-deadline");
      workflowStartEntered.resolve();
      return {
        result: never,
        cancel: () => { cancels += 1; },
        dispose: () => { disposals += 1; return never; },
      };
    },
  });

  const role = harness.runner.run(samplePlanBinding());
  await workflowStartEntered.promise;
  const drainSnapshot = draining.then(() => ({
    active: harness.runner.activeWorkflows.size,
    pending: harness.runner.pendingStarts.size,
  }));
  const [roleOutcome, snapshot] = await Promise.all([
    role.then(
      () => ({ status: "fulfilled" }),
      (error) => ({ status: "rejected", error }),
    ),
    drainSnapshot,
  ]);

  assert.equal(roleOutcome.status, "rejected");
  assert.equal(roleOutcome.error.code, "structured_role_operational_timeout");
  assert.deepEqual(snapshot, { active: 0, pending: 0 });
  assert.equal(cancels, 1);
  assert.equal(disposals, 1);
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
    startWorkflow: ({ request, listeners, sessions }) => {
      workflowStarts += 1;
      if (workflowStarts === 1) {
        const childId = "workflow-deadline-missing-child";
        sessions.set(childId, {
          id: childId,
          events: rc6ConsumedEvents("origin-vector-forecasting-balanced", {
            prediction: true,
          }),
        });
        publishWorkflowChild(listeners, request, childId, "failed");
        return {
          result: Promise.resolve({
            value: [null],
            stopReason: "completed",
            agentsStarted: 1,
          }),
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
    samples: [{ sample_id: "origin-timeout" }],
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
