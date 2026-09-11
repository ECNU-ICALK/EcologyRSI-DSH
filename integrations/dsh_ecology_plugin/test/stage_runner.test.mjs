import assert from "node:assert/strict";
import test from "node:test";

import { SidecarError } from "../lib/sidecar/client.js";
import { RESEARCH_EXECUTION_POLICY } from "../lib/runtime/research-execution-policy.js";
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
  terminalError = { message: "provider failed", code: "UPSTREAM" },
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
          ? { kind: "error", error: terminalError }
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

function samplePlanBinding({ admissionId = "admission-sample-plan-1" } = {}) {
  const context = samplePlanContext();
  return {
    run_id: "run-sample-plan",
    stage: "sample.plan",
    admission_id: admissionId,
    run_state_revision: 7,
    stage_attempt: 2,
    ledger_expected_revision: 11,
    idempotency_key: "sample-plan-1",
    request: {
      role: "sample-planner",
      output_schema_id: "ecology-sample-predictions@2",
      max_tokens: 2048,
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

function directSampleBinding(stage, context) {
  const reflection = stage === "sample.reflect";
  const planner = stage === "sample.plan";
  return {
    run_id: `run-${stage}`,
    stage,
    admission_id: `admission-${stage}-1`,
    run_state_revision: 7,
    stage_attempt: 1,
    ledger_expected_revision: 11,
    idempotency_key: `${stage}-1`,
    request: {
      role: planner ? "sample-planner" : "sample-critic",
      output_schema_id: planner
        ? "ecology-sample-predictions@2"
        : reflection
          ? "ecology-sample-reflection@1"
          : "ecology-sample-review@2",
      ...(planner ? { max_tokens: 2048 } : {}),
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
  persistenceError = null,
}) {
  const starts = [];
  const reservations = [];
  const reservationRequests = [];
  const persisted = [];
  const failures = [];
  const penalties = [];
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
        if (path.endsWith("/session-usage")) return { accepted: true };
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
        const error = typeof persistenceError === "function"
          ? persistenceError(persisted.length)
          : persistenceError;
        if (error) throw error;
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
      penalize: (provider, milliseconds, options) => {
        penalties.push({ provider, milliseconds, options });
      },
    },
  });
  return {
    runner,
    starts,
    reservations,
    reservationRequests,
    persisted,
    failures,
    penalties,
  };
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

test("candidate local edit must avoid same-revision rejected bundles", () => {
  const localEdit = STAGES["candidate.local_edit"];

  assert.match(localEdit.instruction, /current_candidate_state/i);
  assert.match(localEdit.instruction, /recent_edit_history/i);
  assert.match(localEdit.instruction, /same candidate revision/i);
  assert.match(localEdit.instruction, /no distinct registered local change.*return keep/i);
});

test("pre-score sample critic has a bounded schema-recovery protocol", () => {
  const critic = STAGES["sample.critic"];

  assert.match(critic.instruction, /exact wave_digest/i);
  assert.match(critic.instruction, /every supplied sample_id/i);
  assert.match(critic.instruction, /never call structured_output with empty arguments/i);
  assert.match(critic.instruction, /rejects its arguments.*correct.*once/i);
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

function correctedRetrievalEvents() {
  return [
    { seq: 1, type: "tool/call", data: { callId: "skill", name: "skill", arguments: { name: "autonomous-ecology-research" } } },
    { seq: 2, type: "tool/result", data: { message: { content: [{ type: "tool-result", toolCallId: "skill", isError: false }] } } },
    { seq: 3, type: "tool/call", data: { turn: 1, step: 2, callId: "bad-search", name: "web_search", arguments: JSON.stringify({ queries: "greenhouse forecast", retrieval_key: "evidence" }) } },
    { seq: 4, type: "tool/result", sourceEventSeqs: [3], data: { turn: 1, step: 2, message: { content: [{ type: "tool-result", toolCallId: "bad-search", isError: true, content: [{ type: "text", text: "Error: web_search requires one to four queries" }] }] } } },
    { seq: 5, type: "tool/call", data: { callId: "good-search", name: "web_search", arguments: { queries: ["greenhouse forecast"], retrieval_key: "evidence" } } },
    { seq: 6, type: "tool/result", data: { message: { content: [{ type: "tool-result", toolCallId: "good-search", isError: false }] } } },
    { seq: 7, type: "tool/call", data: { callId: "structured", name: "structured_output", arguments: {} } },
  ];
}

test("retrieval argument correction preserves Skill-first evidence without repeating model work", () => {
  const evidence = skillInvocationEvidence(correctedRetrievalEvents(), {
    stage: "generation.research-synthesis",
    skillName: "autonomous-ecology-research",
    allowDynamicRetrieval: true,
  });
  assert.equal(evidence.first_tool_call_verified, true);
  assert.equal(evidence.next_tool_call_seq, 7);
});

function emptyRetrievalEvents() {
  const events = correctedRetrievalEvents();
  events[2].data.arguments = {};
  events[3].data.message.content[0].content[0].text = "Error: web_search arguments have an invalid shape";
  const secondCall = structuredClone(events[2]);
  secondCall.seq = 5;
  secondCall.data.step = 3;
  secondCall.data.callId = "second-empty-search";
  const secondResult = structuredClone(events[3]);
  secondResult.seq = 6;
  secondResult.sourceEventSeqs = [5];
  secondResult.data.step = 3;
  secondResult.data.message.content[0].toolCallId = "second-empty-search";
  for (const event of events.slice(4)) event.seq += 2;
  events.splice(4, 0, secondCall, secondResult);
  return events;
}

test("two empty retrieval calls can recover within the original three-call budget", () => {
  const evidence = skillInvocationEvidence(emptyRetrievalEvents(), {
    stage: "generation.research-synthesis", skillName: "autonomous-ecology-research",
    allowDynamicRetrieval: true,
  });
  assert.equal(evidence.first_tool_call_verified, true);
  assert.equal(evidence.next_tool_call_seq, 9);
});

test("empty retrieval recovery does not admit uncorrected, unauthorized or unattributed failures", () => {
  const mutations = {
    "no successful correction": (events) => events.splice(6, 2),
    "wrong error": (events) => { events[3].data.message.content[0].content[0].text = "Error: role tool authorization failed"; },
    "wrong attribution": (events) => { events[5].sourceEventSeqs = [3]; },
    "different step": (events) => { events[5].data.step = 4; },
    "nonempty unbound arguments": (events) => { events[2].data.arguments = { queries: ["forecast"] }; },
    "array arguments": (events) => { events[2].data.arguments = []; events[3].data.message.content[0].content[0].text = "Error: web_search arguments must be an object"; },
    "correction failed": (events) => { events[7].data.message.content[0].isError = true; },
    "correction after terminal": (events) => { events[6].seq = 10; events[7].seq = 11; },
  };
  for (const [name, mutate] of Object.entries(mutations)) {
    const events = emptyRetrievalEvents();
    mutate(events);
    assert.throws(() => skillInvocationEvidence(events, {
      stage: "generation.research-synthesis", skillName: "autonomous-ecology-research",
      allowDynamicRetrieval: true,
    }), /dynamic retrieval must succeed/, name);
  }
});

test("empty retrieval recovery cannot increase the three-call budget", () => {
  const events = emptyRetrievalEvents();
  const extraCall = structuredClone(events[2]);
  extraCall.seq = 7;
  extraCall.data.callId = "third-empty-search";
  const extraResult = structuredClone(events[3]);
  extraResult.seq = 8;
  extraResult.sourceEventSeqs = [7];
  extraResult.data.message.content[0].toolCallId = "third-empty-search";
  for (const event of events.slice(6)) event.seq += 2;
  events.splice(6, 0, extraCall, extraResult);
  assert.throws(() => skillInvocationEvidence(events, {
    stage: "generation.research-synthesis", skillName: "autonomous-ecology-research",
    allowDynamicRetrieval: true,
  }), /retrieval/);
});

test("retrieval recovery requires exact error attribution and a successful correction", () => {
  const mutations = {
    "no correction": (events) => events.splice(4, 2),
    "different retrieval key": (events) => { events[4].data.arguments.retrieval_key = "unrelated"; },
    "wrong source event": (events) => { events[3].sourceEventSeqs = [99]; },
    "missing source event": (events) => { delete events[3].sourceEventSeqs; },
    "wrong turn": (events) => { events[3].data.turn = 2; },
    "wrong step": (events) => { events[3].data.step = 3; },
    "authorization error": (events) => { events[3].data.message.content[0].content[0].text = "Error: stage admission is closed"; },
    "valid arguments rejected": (events) => { events[2].data.arguments = { queries: ["forecast"], retrieval_key: "evidence" }; },
    "failed correction": (events) => { events[5].data.message.content[0].isError = true; },
    "reused call identity": (events) => { events[4].data.callId = "bad-search"; },
    "late correction": (events) => { events[4].seq = 8; events[5].seq = 9; },
  };
  for (const [name, mutate] of Object.entries(mutations)) {
    const events = correctedRetrievalEvents();
    mutate(events);
    assert.throws(() => skillInvocationEvidence(events, {
      stage: "generation.research-synthesis",
      skillName: "autonomous-ecology-research",
      allowDynamicRetrieval: true,
    }), /dynamic retrieval must succeed/, name);
  }
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
  assert.match(synthesis.instruction, /structured mutation coordinates are executable/i);
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
        assert.deepEqual(request.agentOptions, { maxTokens: 2048 });
        assert.equal("maxTokens" in request, false);
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
        if (path.endsWith("/session-usage")) return { accepted: true };
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
      max_tokens: 2048,
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

test("native stage runner rejects invalid or missing sample planner max_tokens locally", async () => {
  let roleHostLookups = 0;
  const runner = new NativeStageRunner({}, {
    roleAgents: {
      get: () => {
        roleHostLookups += 1;
        throw new Error("must not inspect a role host");
      },
    },
  });
  for (const maxTokens of [511, 8193, 2048.5, "2048"]) {
    const binding = samplePlanBinding();
    binding.request.max_tokens = maxTokens;
    await assert.rejects(
      runner.run(binding),
      /structured stage max_tokens is invalid/,
    );
  }
  const unbounded = samplePlanBinding();
  delete unbounded.request.max_tokens;
  await assert.rejects(
    runner.run(unbounded),
    /sample\.plan requires a bounded max_tokens/,
  );
  assert.equal(roleHostLookups, 0);
});

function boundedSynthesisBinding({ policy = true, maxTokens = 16384 } = {}) {
  const context = {
    ...(policy ? { research_execution_policy: { ...RESEARCH_EXECUTION_POLICY } } : {}),
    knowledge_snapshot: { evidence_catalog: [{ knowledge_id: "paper:one" }] },
    required_candidate_direction_count: 4,
    synthesis_contract: { allowed_mutation_targets: {
      scientific_parameter: ["ridge_alpha"], registered_predictor: [], instruction_profile: [],
    } },
  };
  return {
    run_id: "run:bounded-synthesis", stage: "generation.research-synthesis",
    run_state_revision: 1, stage_attempt: 1, ledger_expected_revision: 1,
    admission_id: "admission-bounded-synthesis", idempotency_key: "bounded-synthesis",
    request: {
      role: "researcher", output_schema_id: "ecology-research-synthesis@1",
      context, context_canonical_json: canonicalJson(context), context_digest: jsonDigest(context),
      max_tokens: maxTokens,
      identity_digests: { genome_digest: "a".repeat(64), compiled_behavior_digest: "b".repeat(64),
        phenotype_instance_digest: "c".repeat(64) },
    },
  };
}

test("new synthesis alone receives the frozen 16k budget and concise schema", async () => {
  const binding = boundedSynthesisBinding();
  const harness = directSampleHarness({ stage: binding.stage,
    results: [{ stopReason: "completed", structured: { summary: "bounded" } }],
    sessionEvents: () => skillFirstEvents("autonomous-ecology-research"),
  });
  await harness.runner.run(binding);
  assert.equal(harness.starts.length, 1);
  assert.deepEqual(harness.starts[0].request.agentOptions, { maxTokens: 16384 });
  assert.equal(harness.reservationRequests[0].item_digest, jsonDigest(binding.request.context));
  const schema = harness.starts[0].request.outputSchema;
  assert.match(schema.properties.summary.description, /1200 characters/);
  assert.equal(schema.properties.evidence.maxItems, undefined);
  assert.match(schema.properties.evidence.description, /at most 8/);
  assert.equal(schema.properties.candidate_directions.maxItems, undefined);
  assert.match(schema.properties.candidate_directions.description, /exactly 4/);
  const direction = schema.properties.candidate_directions.items;
  assert.match(direction.properties.hypothesis.description, /400 characters/);
  assert.equal(direction.properties.evidence_refs.maxItems, undefined);
  assert.match(direction.properties.evidence_refs.description, /at most 4/);
  assert.equal(harness.persisted.length, 1);
});

test("synthesis budget drift is rejected before reservation or model launch", async () => {
  const harness = directSampleHarness({ stage: "generation.research-synthesis", results: [] });
  for (const binding of [boundedSynthesisBinding({ policy: false }),
    boundedSynthesisBinding({ maxTokens: 8192 }), boundedSynthesisBinding({ maxTokens: 16385 }),
    boundedSynthesisBinding({ maxTokens: true })]) {
    await assert.rejects(harness.runner.run(binding), /max_tokens is invalid/);
  }
  const wrongStage = boundedSynthesisBinding();
  wrongStage.stage = "generation.search-plan";
  wrongStage.request.output_schema_id = "ecology-research-search-plan@1";
  await assert.rejects(harness.runner.run(wrongStage), /max_tokens is invalid/);
  assert.equal(harness.reservationRequests.length, 0);
  assert.equal(harness.starts.length, 0);
});

test("16k synthesis exhaustion is terminal without repeating the frozen request", async () => {
  const binding = boundedSynthesisBinding();
  const harness = directSampleHarness({ stage: binding.stage, maxAttempts: 4,
    results: [{ stopReason: "max-tokens" }],
    sessionEvents: () => skillFirstEvents("autonomous-ecology-research"),
  });
  await assert.rejects(harness.runner.run(binding),
    error => error.code === "structured_child_output_budget_exhausted");
  assert.equal(harness.starts.length, 1);
  assert.equal(harness.persisted.length, 0);
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
        if (path.endsWith("/session-usage")) return { accepted: true };
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
    schema_version: "ecology-sample-review@2",
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
        if (path.endsWith("/session-usage")) return { accepted: true };
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
      output_schema_id: "ecology-sample-review@2",
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
  assert.equal(penalties, 0);
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

test("repeated serialized tool text stops after one fresh probe without fabricated execution", async () => {
  for (const stopReason of ["completed", "error"]) {
    const harness = directSampleHarness({
      stage: "sample.reflect", results: [{ stopReason }, { stopReason }],
      sessionEvents: () => [
        { seq: 1, type: "turn/start", data: { turn: 1 } },
        { seq: 2, type: "step/start", data: { turn: 1, step: 0 } },
        { seq: 3, type: "assistant/message", data: { turn: 1, message: { content: [
          { type: "text", text: '<｜DSML｜tool_calls><｜DSML｜invoke name="skill"></｜DSML｜invoke></｜DSML｜tool_calls>' },
        ] } } },
        { seq: 4, type: "turn/end", data: { turn: 1, reason: { kind: "completed" } } },
      ],
    });
    await assert.rejects(harness.runner.run(directSampleBinding("sample.reflect", {
      schema_version: "ecologyrsi-dsh.sample-reflection-context/1",
      wave_digest: "d".repeat(64), sample: { sample_id: "origin-tool-text" }, outcome: { cells: [] },
    })), (error) => error?.code === "structured_child_tool_protocol_error");
    assert.equal(harness.starts.length, 2);
    assert.equal(harness.failures[0].error_code, "structured_child_tool_protocol_error");
    assert.equal(harness.persisted.length, 0);
  }
});

test("output budget exhaustion never repeats the identical capped model call", async () => {
  for (const stopReason of ["error", "max-tokens"]) {
    const harness = directSampleHarness({
      stage: "sample.reflect", results: [{ stopReason }],
      sessionEvents: () => rc6ConsumedEvents("origin-vector-review", { terminalKind: "max-tokens" }),
    });
    await assert.rejects(harness.runner.run(directSampleBinding("sample.reflect", {
      schema_version: "ecologyrsi-dsh.sample-reflection-context/1",
      wave_digest: "d".repeat(64), sample: { sample_id: "origin-output-budget" }, outcome: { cells: [] },
    })), (error) => error?.code === "structured_child_output_budget_exhausted");
    assert.equal(harness.starts.length, 1);
    assert.equal(harness.persisted.length, 0);
    assert.equal(harness.failures[0].error_code, "structured_child_output_budget_exhausted");
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

test("sample planner retries a zero-turn child after provider backpressure", async () => {
  const structured = {
    schema_version: "ecology-sample-predictions@2",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  const harness = directSampleHarness({
    stage: "sample.plan",
    results: [
      { stopReason: "error" },
      { stopReason: "completed", structured },
    ],
    sessionEvents: (attempt) => attempt === 1
      ? [{ type: "session", data: { id: "allocated-before-provider-rejection" } }]
      : skillFirstEvents("origin-vector-forecasting-balanced", { prediction: true }),
  });

  const result = await harness.runner.run(
    directSampleBinding("sample.plan", samplePlanContext()),
  );

  assert.deepEqual(result.structured, structured);
  assert.equal(harness.starts.length, 2);
  assert.equal(harness.reservations.length, 2);
  assert.equal(harness.failures.length, 1);
  assert.equal(harness.failures[0].error_code, "structured_child_model_error");
  assert.deepEqual(harness.penalties, [{
    provider: "pjlab",
    milliseconds: undefined,
    options: { reduceConcurrency: true },
  }]);
  assert.equal(harness.persisted.length, 1);
});

test("sample planner honors provider retry_after after a consumed RATE_LIMIT turn", async () => {
  const structured = {
    schema_version: "ecology-sample-predictions@2",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  const harness = directSampleHarness({
    stage: "sample.plan",
    results: [
      { stopReason: "error" },
      { stopReason: "completed", structured },
    ],
    sessionEvents: (attempt) => attempt === 1
      ? rc6ConsumedEvents("origin-vector-forecasting-balanced", {
        prediction: true,
        terminalKind: "error",
        terminalError: {
          code: "RATE_LIMIT",
          message: "429 user_rpm_rate_limit_exceeded {\"retry_after\":17}",
        },
      })
      : skillFirstEvents("origin-vector-forecasting-balanced", { prediction: true }),
  });

  const result = await harness.runner.run(
    directSampleBinding("sample.plan", samplePlanContext()),
  );

  assert.deepEqual(result.structured, structured);
  assert.equal(harness.starts.length, 2);
  assert.equal(harness.reservations.length, 2);
  assert.equal(harness.failures.length, 1);
  assert.equal(harness.failures[0].error_code, "structured_child_model_error");
  assert.deepEqual(harness.penalties, [{
    provider: "pjlab",
    milliseconds: 17_000,
    options: { reduceConcurrency: false },
  }]);
  assert.equal(harness.persisted.length, 1);
});

test("provider concurrency limit reduces physical capacity instead of RPM spacing", async () => {
  const structured = { schema_version: "ecology-sample-predictions@2", wave_digest: "f".repeat(64), decisions: [] };
  const harness = directSampleHarness({
    stage: "sample.plan",
    results: [{ stopReason: "error" }, { stopReason: "completed", structured }],
    sessionEvents: (attempt) => attempt === 1
      ? rc6ConsumedEvents("origin-vector-forecasting-balanced", {
        prediction: true, terminalKind: "error",
        terminalError: { code: "RATE_LIMIT",
          message: '429: {"code":"cluster_concurrency_rate_limit_exceeded","details":{"limit_type":"concurrency"}}' },
      }) : skillFirstEvents("origin-vector-forecasting-balanced", { prediction: true }),
  });
  const result = await harness.runner.run(directSampleBinding("sample.plan", samplePlanContext()));
  assert.deepEqual(result.structured, structured);
  assert.deepEqual(harness.penalties, [{provider: "pjlab", milliseconds: undefined,
    options: { reduceConcurrency: true }}]);
  assert.equal(harness.failures.length, 1);
  assert.equal(harness.persisted.length, 1);
});

test("sample planner prefers structured 429 Retry-After metadata", async () => {
  const structured = {
    schema_version: "ecology-sample-predictions@2",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  const harness = directSampleHarness({
    stage: "sample.plan",
    results: [
      { stopReason: "error" },
      { stopReason: "completed", structured },
    ],
    sessionEvents: (attempt) => attempt === 1
      ? rc6ConsumedEvents("origin-vector-forecasting-balanced", {
        prediction: true,
        terminalKind: "error",
        terminalError: {
          status: 429,
          providerRetryAfterMs: 23_000,
          message: "redacted provider failure",
        },
      })
      : skillFirstEvents("origin-vector-forecasting-balanced", { prediction: true }),
  });

  const result = await harness.runner.run(
    directSampleBinding("sample.plan", samplePlanContext()),
  );

  assert.deepEqual(result.structured, structured);
  assert.deepEqual(harness.penalties, [{
    provider: "pjlab",
    milliseconds: 23_000,
    options: { reduceConcurrency: false },
  }]);
  assert.equal(harness.failures.length, 1);
  assert.equal(harness.persisted.length, 1);
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
    assert.equal(harness.penalties.length, 0, error?.code || "authorization");
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

test("sample schema specialization is isolated to each schema clone", async () => {
  const firstWave = "1".repeat(64);
  const secondWave = "2".repeat(64);
  const harness = directSampleHarness({
    stage: "sample.critic",
    results: [
      {
        stopReason: "completed",
        structured: {
          schema_version: "ecology-sample-review@2",
          wave_digest: firstWave,
          decisions: [],
        },
      },
      {
        stopReason: "completed",
        structured: {
          schema_version: "ecology-sample-review@2",
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

  const malformedPlan = directSampleHarness({
    stage: "sample.plan",
    results: [{ stopReason: "completed" }],
  });
  const malformedPlanBinding = samplePlanBinding();
  const malformedPlanContext = samplePlanContext();
  malformedPlanContext.samples = [
    { sample_id: "origin-duplicate" },
    { sample_id: "origin-duplicate" },
  ];
  malformedPlanBinding.request.context = malformedPlanContext;
  malformedPlanBinding.request.context_canonical_json = canonicalJson(malformedPlanContext);
  malformedPlanBinding.request.context_digest = jsonDigest(malformedPlanContext);
  await assert.rejects(
    malformedPlan.runner.run(malformedPlanBinding),
    (error) => error?.code === "sample_stage_context_invalid",
  );
  assert.equal(malformedPlan.starts.length, 0);
  assert.equal(malformedPlan.persisted.length, 0);
  assert.equal(malformedPlan.reservations.length, 1);
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

test("sample planner uses a bounded native one-shot child and retries one missing capture", async () => {
  const skillName = "origin-vector-forecasting-balanced";
  const structured = {
    schema_version: "ecology-sample-predictions@2",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  const harness = directSampleHarness({
    stage: "sample.plan",
    results: [
      { stopReason: "error" },
      { stopReason: "completed", structured },
    ],
    sessionEvents: (attempt) => (
      attempt === 1
        ? rc6ConsumedEvents(skillName, {
          prediction: true,
          structuredResult: {
            isError: true,
            error: { name: "ToolArgsError", code: "INVALID_ARGS" },
          },
        })
        : skillFirstEvents(skillName, { prediction: true })
    ),
  });

  const result = await harness.runner.run(
    directSampleBinding("sample.plan", samplePlanContext()),
  );

  assert.deepEqual(result.structured, structured);
  assert.equal(harness.starts.length, 2);
  assert.equal(harness.reservations.length, 2);
  assert.equal(harness.failures.length, 1);
  assert.equal(harness.persisted.length, 1);
  for (const { request } of harness.starts) {
    assert.deepEqual(request.parent, { id: "sample.plan-role-host" });
    assert.deepEqual(request.agentOptions, { maxTokens: 2048 });
    assert.equal("maxTokens" in request, false);
    assert.deepEqual(request.outputSchema.properties.wave_digest, { type: "string", const: samplePlanContext().wave_digest });
    assert.deepEqual(
      request.outputSchema.properties.decisions.items.properties.sample_id,
      { type: "string", enum: samplePlanContext().samples.map(item => item.sample_id) },
    );
    const plannerPrompt = JSON.parse(request.prompt[0].text);
    assert.match(
      plannerPrompt.instruction,
      /first response must call skill exactly once with name origin-vector-forecasting-balanced/i,
    );
    assert.match(
      plannerPrompt.instruction,
      /call ecology_execute_prediction_tool zero to two times/i,
    );
    assert.match(
      plannerPrompt.instruction,
      /structured_output.*(?:once|finite)/i,
    );
    assert.match(
      plannerPrompt.instruction,
      /one output argument correction.*Host-owned retries/i,
    );
  }
  assert.equal(
    jsonDigest(harness.starts[0].request.outputSchema),
    jsonDigest(harness.starts[1].request.outputSchema),
  );
  assert.equal(result.skill_invocation_evidence.skill_name, skillName);
  assert.equal(
    result.skill_invocation_evidence.next_tool_name,
    "structured_output",
  );
  assert.equal(result.skill_invocation_evidence.order_verified, true);
});

test("sample planner waits for the child Session projection before persistence", async () => {
  const skillName = "origin-vector-forecasting-balanced";
  const structured = {
    schema_version: "ecology-sample-predictions@2",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  let projected = false;
  const events = skillFirstEvents(skillName, { prediction: true });
  const structuredCall = events.pop();
  const harness = directSampleHarness({
    stage: "sample.plan",
    results: [{ stopReason: "completed", structured }],
    sessionEvents: () => {
      setTimeout(() => {
        events.push(structuredCall);
        projected = true;
      }, 30);
      return events;
    },
  });

  const result = await harness.runner.run(
    directSampleBinding("sample.plan", samplePlanContext()),
  );

  assert.equal(projected, true);
  assert.deepEqual(result.structured, structured);
  assert.equal(harness.starts.length, 1);
  assert.equal(harness.persisted.length, 1);
  assert.equal(harness.failures.length, 0);
});

test("completed sample result after INVALID_ARGS retries as missing capture", async () => {
  const skillName = "origin-vector-forecasting-balanced";
  const structured = {
    schema_version: "ecology-sample-predictions@2",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  const harness = directSampleHarness({
    stage: "sample.plan",
    results: [
      { stopReason: "completed", structured },
      { stopReason: "completed", structured },
    ],
    sessionEvents: (attempt) => attempt === 1
      ? rc6ConsumedEvents(skillName, {
        prediction: true,
        structuredResult: {
          isError: true,
          error: { name: "ToolArgsError", code: "INVALID_ARGS" },
        },
      })
      : skillFirstEvents(skillName, { prediction: true }),
  });

  const result = await harness.runner.run(
    directSampleBinding("sample.plan", samplePlanContext()),
  );

  assert.deepEqual(result.structured, structured);
  assert.equal(harness.starts.length, 2);
  assert.equal(harness.failures.length, 1);
  assert.equal(harness.failures[0].error_code, "structured_result_missing");
  assert.equal(harness.persisted.length, 1);
});

test("error result waits for a late completed INVALID_ARGS projection before retry", async () => {
  const skillName = "origin-vector-forecasting-balanced";
  const structured = {
    schema_version: "ecology-sample-predictions@2",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  let terminalProjected = false;
  const harness = directSampleHarness({
    stage: "sample.plan",
    results: [
      { stopReason: "error" },
      { stopReason: "completed", structured },
    ],
    sessionEvents: (attempt) => {
      if (attempt !== 1) {
        return skillFirstEvents(skillName, { prediction: true });
      }
      const events = rc6ConsumedEvents(skillName, {
        prediction: true,
        structuredResult: {
          isError: true,
          error: { name: "ToolArgsError", code: "INVALID_ARGS" },
        },
      });
      const terminal = events.pop();
      const firstStepEnd = events.pop();
      setTimeout(() => {
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
              callId: "structured-call-late",
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
                  toolCallId: "structured-call-late",
                  isError: false,
                  content: [],
                }],
              },
            },
            sourceEventSeqs: [seq - 1],
          },
          { seq: ++seq, type: "step/end", data: { turn: 1, step: 1 } },
          { ...terminal, seq: ++seq },
        );
        terminalProjected = true;
      }, 30);
      return events;
    },
  });

  const result = await harness.runner.run(
    directSampleBinding("sample.plan", samplePlanContext()),
  );

  assert.equal(terminalProjected, true);
  assert.deepEqual(result.structured, structured);
  assert.equal(harness.starts.length, 2);
  assert.equal(harness.failures.length, 1);
  assert.equal(harness.failures[0].error_code, "structured_result_missing");
  assert.equal(harness.persisted.length, 1);
});

test("persistence preserves only the Sidecar safe public diagnostic", async () => {
  const structured = {
    schema_version: "ecology-sample-predictions@2",
    wave_digest: "f".repeat(64),
    decisions: [],
  };
  const harness = directSampleHarness({
    stage: "sample.plan",
    maxAttempts: 1,
    results: [{ stopReason: "completed", structured }],
    sessionEvents: () => skillFirstEvents(
      "origin-vector-forecasting-balanced",
      { prediction: true },
    ),
    persistenceError: new SidecarError("sidecar_rejected", "sidecar_rejected", {
      publicDetail: "sample.plan result does not match its prediction wave",
    }),
  });

  await assert.rejects(
    harness.runner.run(directSampleBinding("sample.plan", samplePlanContext())),
    (error) => (
      error?.code === "structured_result_persist_failed"
      && error?.publicDetail === "sample.plan result does not match its prediction wave"
    ),
  );
  assert.equal(harness.persisted.length, 1);
  assert.equal(harness.failures.length, 1);
});

test("sample persistence retry reuses one child and one frozen sidecar envelope", async () => {
  const waveDigest = "a".repeat(64);
  const structured = {
    schema_version: "ecology-sample-predictions@2",
    wave_digest: waveDigest,
    decisions: [],
  };
  const harness = directSampleHarness({
    stage: "sample.plan",
    maxAttempts: 2,
    results: [{ stopReason: "completed", structured }],
    sessionEvents: () => skillFirstEvents(
      "origin-vector-forecasting-balanced",
      { prediction: true },
    ),
    persistenceError: (attempt) => {
      if (attempt !== 1) return null;
      const error = new Error("temporarily unavailable");
      error.code = "structured_result_persistence_unavailable";
      return error;
    },
  });

  const result = await harness.runner.run(
    directSampleBinding("sample.plan", samplePlanContext()),
  );

  assert.deepEqual(result.structured, structured);
  assert.equal(harness.starts.length, 1);
  assert.equal(harness.reservations.length, 1);
  assert.equal(harness.persisted.length, 2);
  assert.deepEqual(harness.persisted[1], harness.persisted[0]);
  assert.equal(harness.failures.length, 0);
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
        output_schema_id: "ecology-sample-review@2",
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

function correctedOutputEvents() {
  const events = rc6ConsumedEvents('origin-vector-forecasting-balanced', {
    prediction: true,
    structuredResult: { isError: true, error: { name: 'ToolArgsError', code: 'INVALID_ARGS' } },
  });
  const terminal = events.pop();
  const previous = events.at(-1).seq;
  events.push(
    { seq: previous + 1, type: 'step/start', data: { turn: 1, step: 1 } },
    { seq: previous + 2, type: 'tool/call', data: { turn: 1, step: 1, callId: 'corrected', name: 'structured_output', arguments: '{}' } },
    { seq: previous + 3, type: 'tool/result', sourceEventSeqs: [previous + 2], data: { turn: 1, step: 1, message: { content: [{ type: 'tool-result', toolCallId: 'corrected', isError: false }] } } },
    { seq: previous + 4, type: 'step/end', data: { turn: 1, step: 1 } },
    { ...terminal, seq: previous + 5 },
  );
  return events;
}

test('one rejected output followed by a captured correction persists without repeating the Agent', async () => {
  const structured = { schema_version: 'ecology-sample-predictions@2', wave_digest: 'f'.repeat(64), decisions: [] };
  const harness = directSampleHarness({ stage: 'sample.plan', maxAttempts: 1,
    results: [{ stopReason: 'completed', structured }], sessionEvents: correctedOutputEvents });
  const result = await harness.runner.run(directSampleBinding('sample.plan', samplePlanContext()));
  assert.deepEqual(result.structured, structured);
  assert.equal(harness.starts.length, 1);
  assert.equal(harness.persisted.length, 1);
  assert.equal(harness.failures.length, 0);
  assert.equal(result.skill_invocation_evidence.next_tool_call_seq,
    correctedOutputEvents().find(e => e.data?.callId === 'corrected').seq);
});

test('output correction rejects ambiguous evidence, operational errors, extra calls and repeated successful output', () => {
  const changes = [
    events => { events.find(e => e.data?.error).data.error = { name: 'NetworkError', code: 'UNAVAILABLE' }; },
    events => { events.find(e => e.type === 'tool/result' && e.data?.step === 1).sourceEventSeqs = [0]; },
    events => { events.find(e => e.type === 'tool/call' && e.data?.callId === 'corrected').data.callId = 'structured-call'; },
    events => { const e = events.find(e => e.data?.error); delete e.data.error; e.data.message.content[0].isError = false; },
    events => { events.at(-1).data.reason = { kind: 'error' }; },
    events => { events.push({ seq: 100, type: 'tool/call', data: { name: 'ecology_execute_prediction_tool', callId: 'late-tool' } }); },
    events => { events.push({ seq: 100, type: 'tool/call', data: { name: 'structured_output', callId: 'third-output' } }); },
  ];
  for (const change of changes) {
    const events = correctedOutputEvents(); change(events);
    assert.throws(() => skillInvocationEvidence(events, { stage: 'sample.plan', skillName: 'origin-vector-forecasting-balanced', allowsPredictionTools: true }));
  }
});


test("SERVER 503 is recoverable and its bounded contract reaches failure accounting", async () => {
  const harness = directSampleHarness({stage: "sample.plan", maxAttempts: 1,
    results: [{stopReason: "error"}], sessionEvents: () => rc6ConsumedEvents("origin-vector-forecasting-balanced", {
      prediction: true, terminalKind: "error", terminalError: {code: "SERVER", status: 503, providerRetryAfterMs: 12000, message: "private provider message"},
    })});
  await assert.rejects(harness.runner.run(directSampleBinding("sample.plan", samplePlanContext())), e => e.code === "structured_child_model_error");
  assert.equal(harness.starts.length, 1);
  assert.equal(harness.failures[0].runtime_failure.provider_status, 503);
  assert.equal(harness.failures[0].runtime_failure.retryable, true);
  assert.equal(harness.failures[0].runtime_failure.retry_after_ms, 12000);
  assert.equal(harness.failures[0].runtime_failure.failure_domain, "provider");
  assert.doesNotMatch(JSON.stringify(harness.failures), /private provider/);
  assert.equal(harness.persisted.length, 0);
});

test("sample step exhaustion stops before structured acceptance without identical retry", async () => {
  const harness = directSampleHarness({stage: "sample.critic",
    results: [{stopReason: "completed", structured: {}}],
    sessionEvents: () => [...Array.from({length: 5}, (_, i) => ({seq: i + 1, type: "step/start", data: {turn: 1, step: i}})),
      ...skillFirstEvents("origin-vector-review").map(e => ({...e, seq: e.seq + 5}))]});
  await assert.rejects(harness.runner.run(directSampleBinding("sample.critic", {
    schema_version: "ecologyrsi-dsh.sample-review-wave/1", wave_digest: "a".repeat(64), samples: [{sample_id: "x"}],
  })), e => e.code === "structured_child_execution_budget_exhausted");
  assert.equal(harness.starts.length, 1);
  assert.equal(harness.persisted.length, 0);
});

test("a real protocol probe recovers after Skill without accepting serialized text", async () => {
  const context = {schema_version:"ecologyrsi-dsh.sample-reflection-context/1", wave_digest:"d".repeat(64), sample:{sample_id:"origin-probe"},outcome:{cells:[]}};
  const structured = {schema_version:"ecology-sample-reflection@1",wave_digest:context.wave_digest,sample_id:"origin-probe",outcome_class:"neutral",error_source:"unknown",next_action:"keep",confidence:0.8,summary:"Keep."};
  const harness = directSampleHarness({stage:"sample.reflect",results:[{stopReason:"completed"},{stopReason:"completed",structured}],
    sessionEvents: attempt => attempt === 1 ? [
      {seq:1,type:"turn/start",data:{turn:1}}, {seq:2,type:"step/start",data:{turn:1,step:0}},
      {seq:3,type:"tool/call",data:{turn:1,callId:"skill",name:"skill"}},
      {seq:4,type:"assistant/message",data:{turn:1,message:{content:[{type:"text",text:'<｜DSML｜tool_calls><｜DSML｜invoke name="structured_output"></｜DSML｜invoke></｜DSML｜tool_calls>'}]}}},
      {seq:5,type:"turn/end",data:{turn:1,reason:{kind:"completed"}}}
    ] : skillFirstEvents("origin-vector-review")});
  const result = await harness.runner.run(directSampleBinding("sample.reflect",context));
  assert.deepEqual(result.structured,structured);
  assert.equal(harness.starts.length,2); assert.equal(harness.persisted.length,1);
  assert.equal(harness.failures[0].error_code,"structured_child_tool_protocol_error");
  assert.deepEqual(harness.starts[0].request.prompt,harness.starts[1].request.prompt);
});
