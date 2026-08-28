import { createHash, randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";

import {
  DYNAMIC_RETRIEVAL_TOOL_PROFILE,
  roleToolNames,
} from "../tools/roles.js";
import { MAX_REQUEST_TIMEOUT_MS, SidecarClient } from "../sidecar/client.js";
import { dshSessionMetrics } from "./agents.js";
import { ChildBindingRegistry } from "./child-bindings.js";
import {
  MAX_STRUCTURED_STAGE_IN_FLIGHT,
  ProviderStageGate,
} from "./provider-stage-gate.js";
import {
  createStructuredDeadline,
  remainingStructuredDeadlineMs,
  structuredDeadlineExpired,
  validateStructuredTimeoutMs,
} from "./structured-deadline.js";
import { runStructuredRole } from "./structured-roles.js";
import {
  isStructuredProviderRateLimit,
  isTrustedStructuredPhase,
  structuredPhaseError,
  structuredRetryAfterMs,
} from "./structured-stage-errors.js";
import { PendingChildStarts } from "./pending-child-starts.js";

const STAGES = Object.freeze({
  "generation.research": Object.freeze({
    role: "researcher",
    schema: "ecology-research-result@1",
    file: "research-result",
    skillName: "autonomous-ecology-research",
  }),
  "generation.search-plan": Object.freeze({
    role: "researcher",
    schema: "ecology-research-search-plan@1",
    file: "research-search-plan",
    skillName: "autonomous-ecology-research",
    instruction: [
      "Plan the literature search from the previous aggregate result and reflection.",
      "Use every target and horizon in forecast_objective as the full scientific scope; a later generation may prioritize frozen weak cells but must not silently collapse the task to one target.",
      "Return focused scholarly queries only; do not claim that a source was found yet.",
      "Keep every search query within 180 characters and every focus area within 240 characters.",
      "If host_validation_feedback is present, correct its validation_detail in a fresh complete response.",
      "The Host will execute the queries through its bounded metadata retriever.",
    ].join(" "),
  }),
  "generation.research-synthesis": Object.freeze({
    role: "researcher",
    schema: "ecology-research-synthesis@1",
    file: "research-synthesis",
    skillName: "autonomous-ecology-research",
    instruction: [
      "Use only the frozen evidence_catalog supplied by the Host.",
      "Reason against the complete forecast_objective target-horizon matrix; a focused direction must state its local weakness without treating one target as the whole task.",
      "For evidence_ref use exactly a frozen knowledge_id, evidence_digest, or registered capability_id/capability_ids value; never invent a source.",
      "Return exactly required_candidate_direction_count distinct, implementable directions.",
      "Each direction must select exactly one mutation_axis and one matching target from synthesis_contract.allowed_mutation_targets.",
      "Set mutation_direction to increase or decrease for scientific_parameter, and to select for registered_predictor or instruction_profile. Do not put an exact parameter assignment in free-form direction prose.",
      "If host_validation_feedback is present, correct its validation_detail in a fresh complete response.",
    ].join(" "),
  }),
  "candidate.propose": Object.freeze({
    role: "candidate-proposer",
    schema: "ecology-genome-mutation@1",
    file: "genome-mutation",
    skillName: "bounded-plugin-experiment",
    instruction: [
      "The operations array is a mutation delta over parent_genome, not a replacement genome.",
      "Omit every unchanged parameter, policy, instruction, tool policy, and workflow setting.",
      "Choose exactly one operation so the candidate changes one identifiable axis.",
      "When an assigned candidate direction is present, implement its exact mutation_axis and mutation_target using mutation_contract.operation_by_axis.",
      "For a numeric parameter, implement mutation_direction exactly: increase must be strictly above the parent value and decrease strictly below it; stay within the Host-provided normalized trust-region step and avoid exact failed parameter sets.",
      "For registered_predictor and instruction_profile, mutation_direction must be select.",
      "Do not reconstruct or repeat the parent genome.",
    ].join(" "),
  }),
  "candidate.local_edit": Object.freeze({
    role: "candidate-proposer",
    schema: "ecology-local-edit@1",
    file: "local-edit",
    skillName: "bounded-plugin-experiment",
    instruction: [
      "Review only aggregate evidence and the Host mutation catalog for the completed batch.",
      "Read current_candidate_state and recent_edit_history before choosing an operation; never repeat an exact bundle rejected for the same candidate revision and never repeat an unchanged current value.",
      "Return exactly one local-edit object. Choose keep with an empty operations array or mutate with only registered operations.",
      "If no distinct registered local change is admissible, return keep instead of recycling a recent rejected bundle.",
      "Never include raw observations, predictions, timestamps, prompts, or executable code.",
      "The Host enforces the per-batch maximum operation count and applies valid operations atomically.",
    ].join(" "),
  }),
  "generation.judge": Object.freeze({
    role: "generation-judge",
    schema: "ecology-generation-review@1",
    file: "generation-review",
    skillName: "candidate-scientific-review",
    instruction: [
      "Review only the supplied evidence for the single candidate identified by candidate_id, proposal_id, and generation.",
      "Use only scientific_evaluation, including its Host-computed score, passed result, aggregate metrics, and evidence digests, together with fitness_profile_digest and evaluation_cohort_digest.",
      "Return exactly one ecology-generation-review@1 object containing only schema_version, accepted, rationale, and flags.",
      "Avoid causal claims that are not supported by the supplied candidate evidence.",
      "Do not propose next-generation directions, experiments, searches, mutations, or policy changes.",
      "Acceptance is an advisory candidate-evidence assessment, not selection or promotion; never claim that either occurred.",
    ].join(" "),
  }),
  "generation.reflect": Object.freeze({
    role: "generation-judge",
    schema: "ecology-generation-reflection@1",
    file: "generation-reflection",
    skillName: "batch-scientific-reflection",
    instruction: [
      "Reflect only on the supplied aggregate batch outcomes; raw sample rows are unavailable.",
      "Assess the complete forecast_objective target-horizon matrix and preserve non-targeted cells when recommending the next directions.",
      "Explain why candidate directions succeeded or failed without making causal claims.",
      "Return exactly direction_count distinct next-step directions and bounded search queries.",
      "Each direction must select one mutation_axis and one exact target from host_boundary.allowed_mutation_targets.",
      "Set mutation_direction to increase or decrease for scientific_parameter and select for the other axes; never encode an exact parameter assignment in direction prose.",
      "Cite only identifiers in the frozen knowledge_snapshot and correct host_validation_feedback when present.",
      "Every reflected direction is advisory and must pass the next research synthesis Host preflight before candidate use. Your stop recommendation is advisory; the Host owns selection and termination.",
    ].join(" "),
  }),
  "sample.plan": Object.freeze({
    role: "sample-planner",
    schema: "ecology-sample-decisions@1",
    file: "sample-decisions",
    requiresPredictionTool: true,
    instruction: [
      "After loading the candidate-selected Skill, call ecology_execute_prediction_tool exactly once using the sole available tool_id and the exact wave_digest from context.",
      "Wait for its complete target-horizon vector result.",
      "Copy the exact Host wave_digest and every exact Host samples[].sample_id into the structured result.",
      "Then call structured_output exactly once, selecting that same tool for every sample_id.",
      "Do not invent, replace, or calculate predictions yourself.",
      "If the prediction or structured_output call is rejected, terminate the child turn immediately: emit no prose and do not retry; the Host will start a fresh bounded child attempt.",
    ].join(" "),
  }),
  "sample.critic": Object.freeze({
    role: "sample-critic",
    schema: "ecology-sample-review@1",
    file: "sample-review",
    skillName: "origin-vector-review",
    instruction: [
      "Review exactly the supplied immutable prediction decisions and use the exact wave_digest from context.",
      "Copy the exact Host wave_digest and every exact Host samples[].sample_id into the structured result.",
      "Return one decision for every supplied sample_id and no additional decisions.",
      "Never call structured_output with empty arguments.",
      "If a skill or structured_output call is rejected, terminate the child turn immediately: emit no prose and do not retry; the Host will start a fresh bounded child attempt.",
    ].join(" "),
  }),
  "sample.reflect": Object.freeze({
    role: "sample-critic",
    schema: "ecology-sample-reflection@1",
    file: "sample-reflection",
    skillName: "origin-vector-review",
    instruction: [
      "Reflect on exactly one completed historical training-feedback forecast origin, including every supplied target-horizon cell.",
      "Copy the exact Host wave_digest and the exact outer Host context.sample.sample_id into the structured result.",
      "Never use context.sample.prediction_cells[].sample_id as the reflection sample_id.",
      "The prediction vector is already immutable; do not propose replacement values.",
      "Classify the outcome, identify the bounded error source, and suggest only a next-generation action.",
      "Do not make causal claims from observational prediction error."
    ].join(" "),
  }),
});

const DSH_SCHEMA_KEYS = new Set([
  "type", "oneOf", "properties", "required", "additionalProperties",
  "items", "enum", "const", "title", "description",
]);

function inferredJsonType(value) {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (Number.isInteger(value)) return "integer";
  return typeof value === "number" ? "number" : typeof value;
}

export function dshCompatibleSchema(schema) {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) return schema;
  if (Array.isArray(schema.type)) {
    return {
      oneOf: [...new Set(schema.type)].map((type) => ({ type })),
      ...("title" in schema ? { title: schema.title } : {}),
      ...(typeof schema.description === "string" ? { description: schema.description } : {}),
    };
  }
  const projected = {};
  for (const [key, value] of Object.entries(schema)) {
    if (!DSH_SCHEMA_KEYS.has(key)) continue;
    if (key === "properties") {
      projected.properties = Object.fromEntries(
        Object.entries(value || {}).map(([name, child]) => [name, dshCompatibleSchema(child)]),
      );
    } else if (key === "items") {
      projected.items = dshCompatibleSchema(value);
    } else if (key === "oneOf") {
      projected.oneOf = value.map(dshCompatibleSchema);
    } else {
      projected[key] = structuredClone(value);
    }
  }
  if ("const" in projected && !("type" in projected) && !("oneOf" in projected)) {
    projected.type = inferredJsonType(projected.const);
  }
  if ("enum" in projected && !("type" in projected) && !("oneOf" in projected)) {
    const types = [...new Set(projected.enum.map(inferredJsonType))];
    if (types.length === 1) projected.type = types[0];
    else {
      delete projected.enum;
      projected.oneOf = schema.enum.map((value) => ({
        const: structuredClone(value),
        type: inferredJsonType(value),
      }));
    }
  }
  return projected;
}

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

export function jsonDigest(value) {
  return createHash("sha256").update(canonical(value)).digest("hex");
}

function textDigest(value) {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function exactDigest(value, name) {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) {
    throw new Error(`${name} must be a SHA-256 digest`);
  }
  return value;
}

function positiveStageAttempts(value) {
  if (!Number.isSafeInteger(value) || value < 1 || value > 4) {
    throw new Error("structuredStageMaxAttempts must be an integer between 1 and 4");
  }
  return value;
}

function retryableStructuredStageError(error, stage) {
  return isTrustedStructuredPhase(error, "model")
    || (
      ["sample.plan", "sample.critic", "sample.reflect"].includes(stage)
      && isTrustedStructuredPhase(error, "capture")
    );
}

function sampleStageContextInvalid(detail) {
  const error = new Error(`invalid sample stage Host context: ${detail}`);
  error.code = "sample_stage_context_invalid";
  return error;
}

function validSampleId(value) {
  return typeof value === "string"
    && value.trim().length > 0
    && value.length <= 240;
}

function sampleMemberDigests(stage, context) {
  if (!["sample.plan", "sample.critic", "sample.reflect"].includes(stage)) {
    return null;
  }
  const samples = stage === "sample.reflect"
    ? context?.sample?.prediction_cells
    : context?.samples;
  if (!Array.isArray(samples) || samples.length < 1 || samples.length > 128) {
    return null;
  }
  const sampleIds = samples.map((sample) => sample?.sample_id);
  if (!sampleIds.every(validSampleId)) return null;
  return [...new Set(sampleIds.map((sampleId) => jsonDigest(sampleId)))].sort();
}

function specializeSampleOutputSchema(stage, schema, context) {
  if (!["sample.plan", "sample.critic", "sample.reflect"].includes(stage)) return schema;
  const waveDigest = context?.wave_digest;
  if (typeof waveDigest !== "string" || !/^[0-9a-f]{64}$/.test(waveDigest)) {
    throw sampleStageContextInvalid("wave_digest must be lowercase SHA-256");
  }
  const properties = schema?.properties;
  if (!properties?.wave_digest) {
    throw new Error("sample output schema has no wave_digest property");
  }
  if (stage === "sample.reflect") {
    const sampleId = context?.sample?.sample_id;
    if (!validSampleId(sampleId)) {
      throw sampleStageContextInvalid("outer sample.sample_id must be 1 to 240 characters");
    }
    if (!properties.sample_id) {
      throw new Error("sample reflection schema has no sample_id property");
    }
    properties.wave_digest = {
      ...properties.wave_digest,
      const: waveDigest,
    };
    properties.sample_id = {
      ...properties.sample_id,
      const: sampleId,
    };
    return schema;
  }
  const samples = context?.samples;
  if (!Array.isArray(samples) || samples.length < 1 || samples.length > 128) {
    throw sampleStageContextInvalid("samples must contain 1 to 128 items");
  }
  const sampleIds = samples.map((sample) => sample?.sample_id);
  if (!sampleIds.every(validSampleId)) {
    throw sampleStageContextInvalid("sample_id must be a nonempty string up to 240 characters");
  }
  if (new Set(sampleIds).size !== sampleIds.length) {
    throw sampleStageContextInvalid("sample_id values must be unique");
  }
  const sampleIdSchema = properties?.decisions?.items?.properties?.sample_id;
  if (!sampleIdSchema) {
    throw new Error("sample decision schema has no sample_id property");
  }
  // sample.plan is the hot path.  Keep its output schema identical across
  // origin waves so the provider can reuse the schema/prompt prefix cache.
  // The Host still validates the exact wave digest, sample-id set, sole
  // prediction-tool receipt, and one decision per prediction before accepting
  // a result.  Critic schemas retain exact per-wave specialization because
  // their much smaller volume benefits more from early schema rejection.
  if (stage === "sample.plan") return schema;
  properties.wave_digest = {
    ...properties.wave_digest,
    const: waveDigest,
  };
  properties.decisions.items.properties.sample_id = {
    ...sampleIdSchema,
    enum: sampleIds,
  };
  return schema;
}

function structuredOperationalTimeout() {
  const error = new Error("structured role operational timeout");
  error.code = "structured_role_operational_timeout";
  return error;
}

function requireStructuredDeadline(deadline) {
  if (structuredDeadlineExpired(deadline)) throw structuredOperationalTimeout();
}

async function withinStructuredDeadline(deadline, operation) {
  requireStructuredDeadline(deadline);
  const timeoutError = structuredOperationalTimeout();
  let timer;
  const timeoutPromise = new Promise((_resolve, reject) => {
    timer = setTimeout(
      () => reject(timeoutError),
      remainingStructuredDeadlineMs(deadline),
    );
  });
  timeoutPromise.catch(() => {});
  try {
    const value = await Promise.race([
      Promise.resolve().then(operation),
      timeoutPromise,
    ]);
    requireStructuredDeadline(deadline);
    return value;
  } catch (error) {
    if (error === timeoutError || structuredDeadlineExpired(deadline)) {
      throw structuredOperationalTimeout();
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

const SAMPLE_PLANNER_SKILLS = new Set([
  "origin-vector-forecasting-balanced",
  "origin-vector-forecasting-anomaly-aware",
  "origin-vector-forecasting-horizon-aware",
]);

function expectedSkillName(contract, request) {
  if (contract.skillName) return contract.skillName;
  const sampleContext = request?.context?.context;
  const profile = sampleContext?.candidate_agent_profile
    || sampleContext?.evolution_context?.candidate_agent_profile
    || sampleContext?.decision_context?.candidate_agent_profile;
  if (
    request?.context?.schema_version !== "ecologyrsi-dsh.sample-routing-wave/1"
    || profile?.schema_version !== "ecologyrsi-dsh.candidate-agent-profile/1"
    || profile?.role !== "sample-planner"
    || !SAMPLE_PLANNER_SKILLS.has(profile?.skill_name)
  ) {
    throw new Error("sample.plan has no registered candidate Skill profile");
  }
  return profile.skill_name;
}

function eventSequence(event, fallback) {
  return Number.isSafeInteger(event?.seq) && event.seq >= 0 ? event.seq : fallback;
}

function eventData(event) {
  return event?.data && typeof event.data === "object" ? event.data : {};
}

function callArguments(event) {
  const value = eventData(event).arguments;
  if (value && typeof value === "object" && !Array.isArray(value)) return value;
  if (typeof value !== "string") return null;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function toolResultIdentity(event) {
  const data = eventData(event);
  if (typeof data.callId === "string") {
    return { callId: data.callId, isError: data.isError === true };
  }
  const content = data.message?.content;
  if (!Array.isArray(content)) return null;
  const block = content.find((item) => item?.type === "tool-result");
  if (!block || typeof block.toolCallId !== "string") return null;
  return { callId: block.toolCallId, isError: block.isError === true };
}

function accountsForConsumedClaim(reason) {
  return reason?.kind !== "completed";
}

// Mirror the rc.6 foldConsumedWork(...).end selection used by readResult.
function consumedTerminalEnd(events) {
  const stepped = new Set();
  const claimed = new Set();
  let open;
  let end;
  for (const event of events) {
    const data = eventData(event);
    if (event?.type === "turn/start") {
      open = data.turn;
    } else if (event?.type === "step/start") {
      stepped.add(data.turn);
    } else if (event?.type === "agent/inbox/spliced") {
      if (data.removedCount !== undefined && data.outcome !== "canceled" && open !== undefined) {
        claimed.add(open);
      }
    } else if (event?.type === "turn/end") {
      open = undefined;
      if (
        stepped.delete(data.turn)
        || (claimed.delete(data.turn) && accountsForConsumedClaim(data.reason))
      ) {
        end = event;
      }
    }
  }
  return end;
}

function structuredCaptureDisposition(rawEvents) {
  if (!Array.isArray(rawEvents)) return "non-missing";
  // A provider can reject a burst after DSH has allocated the child Session
  // but before the first model turn is durable.  Such Sessions contain only
  // their identity record (no turn, step, or tool boundary) and consume zero
  // model tokens.  Treat that exact shape as a transient model failure so the
  // provider gate can back off and retry the same Host origin in a fresh child.
  // Once any turn/tool lifecycle is visible, keep the stricter classifiers
  // below: authorization and malformed-tool failures must never be hidden by
  // a retry.
  const modelTurnStarted = rawEvents.some((event) => [
    "turn/start",
    "step/start",
    "agent/inbox/spliced",
    "tool/call",
    "tool/result",
    "turn/end",
  ].includes(event?.type));
  if (!modelTurnStarted) return "retryable-model";
  const terminal = consumedTerminalEnd(rawEvents);
  const terminalData = eventData(terminal);
  const terminalError = terminalData.reason?.error;
  const retryFailures = rawEvents
    .filter((event) => event?.type === "llm/retry")
    .map((event) => eventData(event).failure)
    .filter((failure) => failure && typeof failure === "object");
  const isRateLimitFailure = (failure) => (
    failure?.code === "RATE_LIMIT" || failure?.status === 429
  );
  if (
    terminalData.reason?.kind === "error"
    && (
      isRateLimitFailure(terminalError)
      || retryFailures.some(isRateLimitFailure)
    )
  ) {
    // DSH rc.6 retries provider failures internally, but its sub-second retry
    // delay does not honor pjlab's user-RPM retry_after window.  Extract only
    // the bounded numeric cooldown from the trusted child event log; never
    // surface or persist the provider message/request identity.
    const structuredRetryAfterValues = [terminalError, ...retryFailures]
      .flatMap((failure) => {
        const value = failure?.providerRetryAfterMs;
        return Number.isSafeInteger(value) && value > 0 && value <= 3_600_000
          ? [value]
          : [];
      });
    const messageRetryAfterValues = rawEvents.flatMap((event) => {
      const data = eventData(event);
      const message = event?.type === "llm/retry"
        ? data.failure?.message
        : event === terminal
          ? terminalError?.message
          : null;
      if (typeof message !== "string") return [];
      const match = message.match(/["']?retry_after["']?\s*[:=]\s*(\d{1,4})/i);
      if (!match) return [];
      const seconds = Number(match[1]);
      return Number.isSafeInteger(seconds) && seconds > 0 && seconds <= 3_600
        ? [seconds * 1_000]
        : [];
    });
    const retryAfterValues = [
      ...structuredRetryAfterValues,
      ...messageRetryAfterValues,
    ];
    return {
      kind: "retryable-provider",
      retryAfterMs: retryAfterValues.length
        ? Math.max(...retryAfterValues)
        : null,
    };
  }
  if (terminalData.reason?.kind !== "completed") return "non-missing";
  // A schema-tool rejection always invalidates the child attempt. DSH may let
  // the model issue a second call and finish the turn, while SubagentRun has
  // already published the first failure. Never accept that projection race as
  // an exactly-once result; the Host retries one fresh bounded child instead.
  if (exactInvalidStructuredArgsSeen(rawEvents)) return "missing";
  const turn = terminalData.turn;
  const events = rawEvents.map((event, index) => ({
    event,
    seq: eventSequence(event, index),
  }));
  const calls = events.filter(({ event }) => {
    const data = eventData(event);
    return event?.type === "tool/call"
      && Number.isSafeInteger(event.seq)
      && event.seq >= 0
      && data.turn === turn
      && data.name === "structured_output"
      && typeof data.callId === "string"
      && data.callId.length > 0;
  });
  if (calls.length === 0) return "missing";
  const call = calls.at(-1);
  const callData = eventData(call.event);
  const terminalSeq = eventSequence(terminal, Number.POSITIVE_INFINITY);
  const result = events.find((item) => {
    if (
      item.event?.type !== "tool/result"
      || item.seq <= call.seq
      || item.seq >= terminalSeq
    ) return false;
    const data = eventData(item.event);
    const identity = toolResultIdentity(item.event);
    return data.turn === callData.turn
      && data.step === callData.step
      && identity?.callId === callData.callId
      && identity.isError === true
      && Array.isArray(item.event.sourceEventSeqs)
      && item.event.sourceEventSeqs.length === 1
      && item.event.sourceEventSeqs[0] === call.seq;
  });
  const error = eventData(result?.event).error;
  return error?.name === "ToolArgsError" && error?.code === "INVALID_ARGS"
    ? "missing"
    : "non-missing";
}

function successfulResultAfter(events, call) {
  const callId = eventData(call.event).callId;
  if (typeof callId !== "string" || !callId) return null;
  return events.find((item) => {
    if (item.event?.type !== "tool/result" || item.seq <= call.seq) return false;
    const result = toolResultIdentity(item.event);
    return result?.callId === callId && result.isError === false;
  }) || null;
}

function exactInvalidStructuredArgsSeen(rawEvents) {
  if (!Array.isArray(rawEvents)) return false;
  const events = rawEvents
    .map((event, index) => ({ event, seq: eventSequence(event, index) }))
    .sort((left, right) => left.seq - right.seq);
  const structuredCallIds = events
    .filter((item) => item.event?.type === "tool/call")
    .filter((item) => eventData(item.event).name === "structured_output")
    .map((item) => eventData(item.event).callId)
    .filter((callId) => typeof callId === "string" && callId.length > 0);
  // Reused call identities make source attribution ambiguous. Fail closed as
  // a model-terminal contract error instead of authorizing a missing retry.
  if (new Set(structuredCallIds).size !== structuredCallIds.length) return false;
  for (const call of events) {
    if (call.event?.type !== "tool/call") continue;
    const callData = eventData(call.event);
    if (callData.name !== "structured_output" || typeof callData.callId !== "string") {
      continue;
    }
    const result = events.find((item) => {
      if (item.event?.type !== "tool/result" || item.seq <= call.seq) return false;
      const identity = toolResultIdentity(item.event);
      return identity?.callId === callData.callId
        && identity.isError === true
        && Array.isArray(item.event.sourceEventSeqs)
        && item.event.sourceEventSeqs.length === 1
        && item.event.sourceEventSeqs[0] === call.seq;
    });
    const error = eventData(result?.event).error;
    if (error?.name === "ToolArgsError" && error?.code === "INVALID_ARGS") {
      return true;
    }
  }
  return false;
}

const SESSION_PROJECTION_SYNC_GRACE_MS = 2_000;
const SESSION_PROJECTION_SYNC_POLL_MS = 20;

function waitForSessionProjection(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason || new Error("session projection wait aborted"));
      return;
    }
    let timer;
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal.reason || new Error("session projection wait aborted"));
    };
    timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, milliseconds);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

async function synchronizedStructuredCaptureDisposition(
  ctx,
  sessionId,
  deadline,
) {
  let events = ctx?.sessions?.get?.(sessionId)?.events;
  const initialDisposition = structuredCaptureDisposition(events);
  // An allocated Session without a turn boundary is the exact zero-token
  // provider-rejection shape. Retrying it immediately cannot race a tool call.
  if (initialDisposition === "retryable-model") return initialDisposition;
  if (Array.isArray(events) && consumedTerminalEnd(events)) {
    return initialDisposition;
  }
  if (
    !deadline
    || typeof deadline.throwIfExpired !== "function"
    || typeof deadline.remainingTimeoutMs !== "function"
  ) {
    return initialDisposition;
  }
  const graceMs = Math.min(
    SESSION_PROJECTION_SYNC_GRACE_MS,
    Math.max(1, deadline.remainingTimeoutMs()),
  );
  const expiresAt = Date.now() + graceMs;
  while (true) {
    deadline.throwIfExpired();
    events = ctx?.sessions?.get?.(sessionId)?.events;
    if (Array.isArray(events) && consumedTerminalEnd(events)) {
      return structuredCaptureDisposition(events);
    }
    const remainingGraceMs = expiresAt - Date.now();
    if (remainingGraceMs <= 0) {
      return structuredCaptureDisposition(events);
    }
    await waitForSessionProjection(
      Math.min(SESSION_PROJECTION_SYNC_POLL_MS, remainingGraceMs),
      deadline.signal,
    );
  }
}

function verifiedSkillInvocationEvidence(events, options) {
  if (exactInvalidStructuredArgsSeen(events)) {
    throw structuredPhaseError("capture");
  }
  return skillInvocationEvidence(events, options);
}

async function synchronizedSkillInvocationEvidence(ctx, sessionId, options, deadline) {
  const graceMs = Math.min(
    SESSION_PROJECTION_SYNC_GRACE_MS,
    Math.max(1, deadline.remainingTimeoutMs()),
  );
  const expiresAt = Date.now() + graceMs;
  let lastError = null;
  while (true) {
    deadline.throwIfExpired();
    const events = ctx?.sessions?.get?.(sessionId)?.events;
    // Check the exact tool rejection before accepting ordering evidence: the
    // latter intentionally proves call order, not whether structured_output
    // accepted its arguments.
    try {
      return verifiedSkillInvocationEvidence(events, options);
    } catch (error) {
      lastError = error;
      // A schema-tool rejection is a bounded missing capture even when DSH
      // subsequently reports a structured result. Retrying it in a fresh
      // child preserves the exactly-once structured-output contract instead
      // of misreporting a local evidence rejection as persistence failure.
      // Once the consumed turn-end is visible, all prior tool events for that
      // turn have been projected. A remaining evidence error is a real
      // contract violation, not eventual-consistency lag.
      if (Array.isArray(events) && consumedTerminalEnd(events)) throw error;
    }
    const remainingGraceMs = expiresAt - Date.now();
    if (remainingGraceMs <= 0) {
      const error = new Error("DSH child Session projection did not synchronize");
      error.code = "dsh_session_projection_not_ready";
      error.cause = lastError;
      throw error;
    }
    await waitForSessionProjection(
      Math.min(SESSION_PROJECTION_SYNC_POLL_MS, remainingGraceMs),
      deadline.signal,
    );
  }
}

export function skillInvocationEvidence(
  rawEvents,
  {
    stage,
    skillName,
    requiresPredictionTool = false,
    allowDynamicRetrieval = false,
  } = {},
) {
  if (!Array.isArray(rawEvents)) throw new Error("DSH child Session event log is unavailable");
  const events = rawEvents
    .map((event, index) => ({ event, seq: eventSequence(event, index) }))
    .sort((left, right) => left.seq - right.seq);
  const calls = events.filter((item) => item.event?.type === "tool/call");
  const skillCalls = calls.filter((item) => eventData(item.event).name === "skill");
  if (skillCalls.length !== 1 || calls[0] !== skillCalls[0]) {
    throw new Error("the registered Skill must be the first and only skill call");
  }
  const skillCall = skillCalls[0];
  const args = callArguments(skillCall.event);
  if (!args || Object.keys(args).length !== 1 || args.name !== skillName) {
    throw new Error("the DSH child loaded a Skill outside its frozen stage profile");
  }
  const skillResult = successfulResultAfter(events, skillCall);
  if (!skillResult) throw new Error("the registered Skill call did not succeed");

  const nextToolName = requiresPredictionTool
    ? "ecology_execute_prediction_tool"
    : "structured_output";
  const nextCalls = calls.filter((item) => eventData(item.event).name === nextToolName);
  if (nextCalls.length !== 1 || nextCalls[0].seq <= skillResult.seq) {
    throw new Error(`the required ${nextToolName} call did not follow the Skill result`);
  }
  const retrievalCalls = calls.filter(
    (item) => eventData(item.event).name === "web_search",
  );
  if (!allowDynamicRetrieval && retrievalCalls.length > 0) {
    throw new Error("dynamic retrieval is outside the frozen legacy tool profile");
  }
  if (retrievalCalls.length > 3) {
    throw new Error("the zero to three dynamic retrieval call budget was exceeded");
  }
  for (const retrievalCall of retrievalCalls) {
    const retrievalResult = successfulResultAfter(events, retrievalCall);
    if (
      retrievalCall.seq <= skillResult.seq
      || retrievalCall.seq >= nextCalls[0].seq
      || !retrievalResult
      || retrievalResult.seq >= nextCalls[0].seq
    ) {
      throw new Error(
        "dynamic retrieval must succeed after Skill and before the required terminal tool",
      );
    }
  }
  if (requiresPredictionTool) {
    const predictionResult = successfulResultAfter(events, nextCalls[0]);
    const structuredCalls = calls.filter(
      (item) => eventData(item.event).name === "structured_output",
    );
    if (
      !predictionResult
      || structuredCalls.length !== 1
      || structuredCalls[0].seq <= predictionResult.seq
    ) {
      throw new Error(
        "sample.plan must complete Skill, prediction tool, then structured_output in order",
      );
    }
  }
  return Object.freeze({
    schema_version: "ecologyrsi-dsh.skill-invocation-evidence/1",
    stage,
    skill_name: skillName,
    call_count: 1,
    successful_call_count: 1,
    call_seq: skillCall.seq,
    result_seq: skillResult.seq,
    first_tool_call_verified: true,
    next_tool_name: nextToolName,
    next_tool_call_seq: nextCalls[0].seq,
    order_verified: true,
    source: "dsh_session_event_log",
  });
}

export class NativeStageRunner {
  constructor(ctx, {
    roleAgents,
    runRegistry,
    sidecar,
    structuredStageTimeoutMs = 600_000,
    researchStageTimeoutMs = 1_800_000,
    sampleCriticStageTimeoutMs = 600_000,
    structuredStageMinIntervalMs = 0,
    structuredStageMaxInFlight = MAX_STRUCTURED_STAGE_IN_FLIGHT,
    structuredStageFailureCooldownMs = 60_000,
    structuredStageMaxAttempts = 2,
    providerStageGate = null,
  } = {}) {
    this.ctx = ctx;
    this.roleAgents = roleAgents;
    this.runRegistry = runRegistry;
    this.sidecar = sidecar;
    this.structuredStageTimeoutMs = validateStructuredTimeoutMs(
      structuredStageTimeoutMs,
      "structuredStageTimeoutMs",
    );
    this.researchStageTimeoutMs = validateStructuredTimeoutMs(
      researchStageTimeoutMs,
      "researchStageTimeoutMs",
    );
    this.sampleCriticStageTimeoutMs = validateStructuredTimeoutMs(
      sampleCriticStageTimeoutMs,
      "sampleCriticStageTimeoutMs",
    );
    this.structuredStageMaxAttempts = positiveStageAttempts(structuredStageMaxAttempts);
    this.providerStageGate = providerStageGate || new ProviderStageGate({
      minimumIntervalMs: structuredStageMinIntervalMs,
      failureCooldownMs: structuredStageFailureCooldownMs,
      maxInFlight: structuredStageMaxInFlight,
    });
    this.pendingStarts = new PendingChildStarts(ctx, {
      launchFence: this.providerStageGate,
    });
    this.childBindings = new ChildBindingRegistry();
    this.schemaCache = new Map();
    this.bridge = Object.freeze({
      bindingFor: (exec, expected) => this.childBindings.bindingFor(exec?.agent || exec, expected),
      sidecar,
    });
  }

  static fromConfig(ctx, options) {
    const sidecar = new SidecarClient({
      origin: options.backendOrigin,
      token: options.sidecarToolToken,
      totalTimeoutMs: options.totalTimeoutMs,
      maxRequestBytes: options.maxBodyBytes,
      maxResponseBytes: options.maxResponseBytes,
    });
    return new NativeStageRunner(ctx, { ...options, sidecar });
  }

  async schema(file) {
    if (!this.schemaCache.has(file)) {
      const url = new URL(`../../schemas/${file}.schema.json`, import.meta.url);
      this.schemaCache.set(
        file,
        readFile(url, "utf8").then(JSON.parse).then(dshCompatibleSchema),
      );
    }
    return structuredClone(await this.schemaCache.get(file));
  }

  async #recordChildFailure(binding, error) {
    const supplied = String(error?.code || "structured_stage_failed");
    const errorCode = /^[a-z0-9_]{1,80}$/i.test(supplied)
      ? supplied
      : "structured_stage_failed";
    try {
      await this.sidecar.request(
        "/api/ecology-agent-sidecar/v1/child-failures",
        {
          method: "POST",
          timeoutMs: 5_000,
          body: {
            run_id: binding.run_id,
            stage: binding.stage,
            idempotency_key: binding.idempotency_key,
            error_code: errorCode,
          },
        },
      );
    } catch {
      // Failure accounting is best-effort and must never replace the original
      // stage error. The corresponding launch remains a conservative upper
      // bound in the projection if the local sidecar itself is unavailable.
    }
  }

  async run(binding) {
    const contract = STAGES[binding.stage];
    if (!contract) throw new Error("unsupported DSH structured stage");
    const request = binding.request;
    if (!request || request.role !== contract.role || request.output_schema_id !== contract.schema) {
      throw new Error("DSH structured stage role/schema mismatch");
    }
    let canonicalContext;
    try {
      canonicalContext = JSON.parse(request.context_canonical_json);
    } catch {
      canonicalContext = null;
    }
    if (
      !request.context
      || typeof request.context_canonical_json !== "string"
      || textDigest(request.context_canonical_json) !== request.context_digest
      || canonical(canonicalContext) !== canonical(request.context)
    ) {
      throw new Error("DSH structured stage context digest mismatch");
    }
    if (
      request.max_tokens !== undefined
      && (!Number.isSafeInteger(request.max_tokens)
        || request.max_tokens < 512
        || request.max_tokens > 8192)
    ) {
      throw new Error("DSH structured stage max_tokens is invalid");
    }
    if (binding.stage === "sample.plan" && request.max_tokens === undefined) {
      throw new Error("DSH sample.plan requires a bounded max_tokens value");
    }
    const identityDigests = request.identity_digests || {};
    const roleHost = this.roleAgents.get(binding.run_id, contract.role);
    if (!roleHost) throw new Error("DSH role-host is unavailable");
    if (typeof binding.admission_id !== "string" || !binding.admission_id) {
      throw new Error("DSH structured stage admission_id is required");
    }
    const stageTimeoutMs = contract.role === "researcher"
      ? this.researchStageTimeoutMs
      : binding.stage === "sample.critic"
        ? this.sampleCriticStageTimeoutMs
        : this.structuredStageTimeoutMs;
    // Start the stage deadline before provider admission so FIFO queueing,
    // reservation, model execution, and persistence share one budget.
    const lifecycle = {
      timeoutMs: stageTimeoutMs,
      deadline: createStructuredDeadline(stageTimeoutMs),
    };
    const provider = String(roleHost.binding?.model || "default").split("/", 1)[0] || "default";
    let lastError;
    for (let attempt = 1; attempt <= this.structuredStageMaxAttempts; attempt += 1) {
      try {
        const executeReservedStage = async (signal) => {
          try {
            return await this.#runReservedStage({
              binding,
              contract,
              request,
              identityDigests,
              roleHost,
              lifecycle,
              signal,
            });
          } catch (error) {
            await this.#recordChildFailure(binding, error);
            // Apply provider backpressure before the gate releases the failed
            // slot. Otherwise one burst can refill every slot before the
            // outer retry loop observes the rate-limit failure.
            if (isTrustedStructuredPhase(error, "model")) {
              const retryAfterMs = structuredRetryAfterMs(error);
              this.providerStageGate.penalize(
                provider,
                retryAfterMs ?? undefined,
                {
                  reduceConcurrency: !isStructuredProviderRateLimit(error),
                },
              );
            }
            throw error;
          }
        };
        const runAttempt = () => this.providerStageGate.run(provider, executeReservedStage, {
          runId: binding.run_id,
          deadline: lifecycle.deadline,
        });
        // The gate enforces the same absolute deadline for FIFO admission, and
        // the reserved-stage implementations enforce it again around every
        // remote/persistence operation. Avoid a second outer race that could
        // reject the caller before workflow cleanup has finished.
        return await runAttempt();
      } catch (error) {
        lastError = error;
        const retryable = retryableStructuredStageError(error, binding.stage);
        if (!retryable || attempt >= this.structuredStageMaxAttempts) {
          throw error;
        }
      }
    }
    throw lastError;
  }

  async #runReservedStage({
    binding,
    contract,
    request,
    identityDigests,
    roleHost,
  lifecycle,
    signal = null,
  }) {
    const dynamicRetrieval = (
      roleHost.binding?.tool_profile === DYNAMIC_RETRIEVAL_TOOL_PROFILE
    );
    if (lifecycle.deadline !== null) requireStructuredDeadline(lifecycle.deadline);
    const reservationTimeoutMs = lifecycle.deadline === null
      ? Math.min(lifecycle.timeoutMs, MAX_REQUEST_TIMEOUT_MS)
      : Math.min(
        Math.max(1, remainingStructuredDeadlineMs(lifecycle.deadline)),
        MAX_REQUEST_TIMEOUT_MS,
      );
    const memberDigests = sampleMemberDigests(binding.stage, request.context);
    const allocation = await this.sidecar.request(
      "/api/ecology-agent-sidecar/v1/child-reservations",
      {
        body: {
          request_id: randomUUID(),
          run_id: binding.run_id,
          parent_session_id: roleHost.sessionId,
          role: contract.role,
          stage: binding.stage,
          run_state_revision: binding.run_state_revision,
          stage_attempt: binding.stage_attempt,
          admission_id: binding.admission_id,
          timeout_ms: lifecycle.timeoutMs,
          item_digest: request.context_digest,
          idempotency_key: binding.idempotency_key,
          ...(memberDigests ? { sample_member_digests: memberDigests } : {}),
        },
        signal,
        timeoutMs: reservationTimeoutMs,
      },
    );
    if (
      !allocation?.accepted
      || !allocation.launch
      || allocation.admission_id !== binding.admission_id
      || allocation.timeout_ms !== lifecycle.timeoutMs
    ) {
      throw new Error("durable child reservation was not accepted");
    }
    requireStructuredDeadline(lifecycle.deadline);
    const launch = {
      ...allocation.launch,
      run_id: binding.run_id,
      stage: binding.stage,
      role: contract.role,
      item_digest: request.context_digest,
      idempotency_key: binding.idempotency_key,
    };
    const frozenIdentity = {
      run_id: binding.run_id,
      role: contract.role,
      stage: binding.stage,
      run_state_revision: binding.run_state_revision,
      stage_attempt: binding.stage_attempt,
      ledger_expected_revision: allocation.ledger_expected_revision,
      idempotency_key: binding.idempotency_key,
      genome_digest: exactDigest(identityDigests.genome_digest, "genome_digest"),
      compiled_behavior_digest: exactDigest(identityDigests.compiled_behavior_digest, "compiled_behavior_digest"),
      phenotype_instance_digest: exactDigest(identityDigests.phenotype_instance_digest, "phenotype_instance_digest"),
      allowed_tools: roleToolNames(
        contract.role,
        dynamicRetrieval ? DYNAMIC_RETRIEVAL_TOOL_PROFILE : null,
      ),
    };
    const reservation = this.childBindings.reserve(roleHost.sessionId, launch, frozenIdentity);
    const skillName = expectedSkillName(contract, request);
    let persistedSkillEvidence = null;
    try {
      requireStructuredDeadline(lifecycle.deadline);
      const outputSchema = await withinStructuredDeadline(
        lifecycle.deadline,
        () => this.schema(contract.file),
      );
      requireStructuredDeadline(lifecycle.deadline);
      specializeSampleOutputSchema(binding.stage, outputSchema, request.context);
      requireStructuredDeadline(lifecycle.deadline);
      const responseProtocol = [
        "Do not narrate analysis.",
        `Your first response must call skill exactly once with name ${skillName}.`,
        ...(dynamicRetrieval ? [
          "After the Skill result, make zero to three web_search calls only when current reasoning needs external evidence; submit queries and retrieval_key only, and never choose a provider.",
        ] : []),
        ...(contract.requiresPredictionTool ? [
          `${dynamicRetrieval ? "Then" : "After the Skill result,"} call ecology_execute_prediction_tool exactly once; do not call structured_output before its result arrives.`,
          "After the prediction-tool result, call structured_output exactly once and emit no prose.",
        ] : [
          `${dynamicRetrieval ? "Then" : "After the Skill result,"} call structured_output exactly once with one concise object matching the supplied output schema.`,
          "Do not emit prose before or after it.",
        ]),
      ];
      const prompt = canonical({
        instruction: [
          ...responseProtocol,
          contract.instruction || "",
        ].filter(Boolean).join(" "),
        stage: binding.stage,
        context: request.context,
      });
      requireStructuredDeadline(lifecycle.deadline);
      const admission = {
        isOpen: async () => {
          const current = this.runRegistry.get(binding.run_id);
          return current?.status === "running";
        },
      };
      let persistenceBody = null;
      const persist = async (
        structured,
        sessionId,
        capturedSessionMetrics = null,
        capturedSessionEvents = null,
        persistenceDeadline = null,
      ) => {
        if (
          typeof persistenceDeadline?.throwIfExpired !== "function"
          || typeof persistenceDeadline?.remainingTimeoutMs !== "function"
        ) {
          throw new Error("structured persistence requires a runtime deadline");
        }
        persistenceDeadline.throwIfExpired();
        if (persistenceBody === null) {
          if (!reservation.claimed_child_id) {
            this.childBindings.claimPublished(
              roleHost.sessionId,
              reservation.label,
              sessionId,
            );
          }
          persistenceDeadline.throwIfExpired();
          if (!this.childBindings.activeByChild.has(sessionId)) {
            this.childBindings.openActivation(sessionId, {
              revision: binding.run_state_revision,
              stage_attempt: binding.stage_attempt,
              idempotency_key: binding.idempotency_key,
            });
          }
          persistenceDeadline.throwIfExpired();
          const evidenceOptions = {
            stage: binding.stage,
            skillName,
            requiresPredictionTool: contract.requiresPredictionTool === true,
            allowDynamicRetrieval: dynamicRetrieval,
          };
          const evidence = capturedSessionEvents
            ? verifiedSkillInvocationEvidence(capturedSessionEvents, evidenceOptions)
            : await synchronizedSkillInvocationEvidence(
              this.ctx,
              sessionId,
              evidenceOptions,
              persistenceDeadline,
            );
          persistenceDeadline.throwIfExpired();
          persistedSkillEvidence = evidence;
          const resultDigest = jsonDigest(structured);
          persistenceDeadline.throwIfExpired();
          const { allowed_tools: _allowedTools, ...identity } = frozenIdentity;
          persistenceBody = Object.freeze({
            identity: {
              ...identity,
              session_id: sessionId,
              child_reservation_id: reservation.launch.reservation_id,
              activation_lease_id: (
                this.childBindings.activeByChild.get(sessionId)
                || `lease-${reservation.launch.reservation_id}`
              ),
            },
            output_schema_id: contract.schema,
            structured: structuredClone(structured),
            result_digest: resultDigest,
            session_metrics: capturedSessionMetrics || dshSessionMetrics(this.ctx, sessionId),
            skill_invocation_evidence: evidence,
            admission_id: binding.admission_id,
          });
        } else if (
          persistenceBody.identity.session_id !== sessionId
          || persistenceBody.result_digest !== jsonDigest(structured)
        ) {
          throw new Error("structured persistence retry changed the frozen result");
        }
        persistenceDeadline.throwIfExpired();
        const requestTimeoutMs = Math.min(
          persistenceDeadline.remainingTimeoutMs(),
          MAX_REQUEST_TIMEOUT_MS,
        );
        persistenceDeadline.throwIfExpired();
        const receipt = await this.sidecar.request("/api/ecology-agent-sidecar/v1/structured-results", {
          body: persistenceBody,
          signal: persistenceDeadline.signal,
          timeoutMs: requestTimeoutMs,
        });
        persistenceDeadline.throwIfExpired();
        return receipt;
      };
      const result = await runStructuredRole(
        roleHost,
        reservation,
        { prompt, outputSchema, maxTokens: request.max_tokens },
        {
          pendingStarts: this.pendingStarts,
          admission,
          persist: async ({ structured, session_id }, persistenceDeadline) => persist(
            structured,
            session_id,
            null,
            null,
            persistenceDeadline,
          ),
          deadline: lifecycle.deadline,
          signal,
          classifyMissingCapture: ({ run }, classificationDeadline) => synchronizedStructuredCaptureDisposition(
            this.ctx,
            String(run?.id || ""),
            classificationDeadline,
          ),
        },
      );
      requireStructuredDeadline(lifecycle.deadline);
      const returnedStructured = result.structured;
      requireStructuredDeadline(lifecycle.deadline);
      const returnedDigest = jsonDigest(returnedStructured);
      requireStructuredDeadline(lifecycle.deadline);
      const returnedSessionId = result.session_id;
      requireStructuredDeadline(lifecycle.deadline);
      return {
        structured: returnedStructured,
        result_digest: returnedDigest,
        session_id: returnedSessionId,
        skill_invocation_evidence: persistedSkillEvidence,
      };
    } finally {
      if (reservation.claimed_child_id) this.childBindings.releaseChild(reservation.claimed_child_id);
      else this.childBindings.revoke(roleHost.sessionId, reservation.label);
    }
  }

  closeLaunchFence(runId) {
    this.pendingStarts.closeRun(runId);
    if (typeof this.providerStageGate.closeRun === "function") {
      this.providerStageGate.closeRun(runId);
    } else {
      this.providerStageGate.cancelRun?.(runId);
    }
  }

  openLaunchFence(runId) {
    this.providerStageGate.openRun?.(runId);
    this.pendingStarts.openRun(runId);
  }

  async #drainLaunchLifecycles(runId) {
    while (true) {
      await this.pendingStarts.cancelAndQuiesce({ runId });
      if (!this.pendingStarts.hasRun(runId)) break;
    }
  }

  async quiesceRun(runId, { timeoutMs = 30_000, deadline = null } = {}) {
    this.closeLaunchFence(runId);
    await this.#drainLaunchLifecycles(runId);
    await this.providerStageGate.drainRun?.(runId, deadline || { timeoutMs });
    await this.#drainLaunchLifecycles(runId);
    this.childBindings.revokeRun(runId);
  }
}

export { STAGES };
