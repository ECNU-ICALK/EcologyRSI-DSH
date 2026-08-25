import { createHash, randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";

import {
  DYNAMIC_RETRIEVAL_TOOL_PROFILE,
  roleToolNames,
} from "../tools/roles.js";
import { SidecarClient } from "../sidecar/client.js";
import { dshSessionMetrics } from "./agents.js";
import { ChildBindingRegistry } from "./child-bindings.js";
import { ProviderStageGate } from "./provider-stage-gate.js";
import { runStructuredRole } from "./structured-roles.js";
import { PendingChildStarts, startHomogeneousWorkflow } from "./workflows.js";

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
      "Then call structured_output exactly once, selecting that same tool for every sample_id.",
      "Do not invent, replace, or calculate predictions yourself.",
    ].join(" "),
  }),
  "sample.critic": Object.freeze({
    role: "sample-critic",
    schema: "ecology-sample-review@1",
    file: "sample-review",
    skillName: "origin-vector-review",
  }),
  "sample.reflect": Object.freeze({
    role: "sample-critic",
    schema: "ecology-sample-reflection@1",
    file: "sample-reflection",
    skillName: "origin-vector-review",
    instruction: [
      "Reflect on exactly one completed historical training-feedback forecast origin, including every supplied target-horizon cell.",
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

function retryableStructuredStageError(error) {
  return error?.code === "structured_child_model_error";
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

function successfulResultAfter(events, call) {
  const callId = eventData(call.event).callId;
  if (typeof callId !== "string" || !callId) return null;
  return events.find((item) => {
    if (item.event?.type !== "tool/result" || item.seq <= call.seq) return false;
    const result = toolResultIdentity(item.event);
    return result?.callId === callId && result.isError === false;
  }) || null;
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
    structuredStageMinIntervalMs = 60_000,
    structuredStageFailureCooldownMs = 60_000,
    structuredStageMaxAttempts = 2,
    providerStageGate = null,
  } = {}) {
    this.ctx = ctx;
    this.roleAgents = roleAgents;
    this.runRegistry = runRegistry;
    this.sidecar = sidecar;
    this.structuredStageTimeoutMs = structuredStageTimeoutMs;
    this.researchStageTimeoutMs = researchStageTimeoutMs;
    this.structuredStageMaxAttempts = positiveStageAttempts(structuredStageMaxAttempts);
    this.providerStageGate = providerStageGate || new ProviderStageGate({
      minimumIntervalMs: structuredStageMinIntervalMs,
      failureCooldownMs: structuredStageFailureCooldownMs,
    });
    this.pendingStarts = new PendingChildStarts(ctx);
    this.childBindings = new ChildBindingRegistry();
    this.activeWorkflows = new Set();
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
    const identityDigests = request.identity_digests || {};
    const roleHost = this.roleAgents.get(binding.run_id, contract.role);
    if (!roleHost) throw new Error("DSH role-host is unavailable");
    const provider = String(roleHost.binding?.model || "default").split("/", 1)[0] || "default";
    let lastError;
    for (let attempt = 1; attempt <= this.structuredStageMaxAttempts; attempt += 1) {
      try {
        return await this.providerStageGate.run(provider, () => this.#runReservedStage({
            binding,
            contract,
            request,
            identityDigests,
            roleHost,
          }), { runId: binding.run_id });
      } catch (error) {
        lastError = error;
        if (retryableStructuredStageError(error)) {
          this.providerStageGate.penalize(provider);
        }
        if (!retryableStructuredStageError(error) || attempt >= this.structuredStageMaxAttempts) {
          throw error;
        }
      }
    }
    throw lastError;
  }

  async #runReservedStage({ binding, contract, request, identityDigests, roleHost }) {
    const dynamicRetrieval = (
      roleHost.binding?.tool_profile === DYNAMIC_RETRIEVAL_TOOL_PROFILE
    );
    const allocation = await this.sidecar.request(
      "/api/ecology-agent-sidecar/v1/child-reservations",
      {
        body: {
          request_id: randomUUID(),
          run_id: binding.run_id,
          parent_session_id: roleHost.sessionId,
          role: contract.role,
          stage: binding.stage,
          item_digest: request.context_digest,
          idempotency_key: binding.idempotency_key,
        },
      },
    );
    if (!allocation?.accepted || !allocation.launch) {
      throw new Error("durable child reservation was not accepted");
    }
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
      const outputSchema = await this.schema(contract.file);
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
      const stageTimeoutMs = contract.role === "researcher"
        ? this.researchStageTimeoutMs
        : this.structuredStageTimeoutMs;
      const prompt = canonical({
        instruction: [
          ...responseProtocol,
          contract.instruction || "",
        ].filter(Boolean).join(" "),
        stage: binding.stage,
        context: request.context,
      });
      const admission = {
        isOpen: async () => {
          const current = this.runRegistry.get(binding.run_id);
          return current?.status === "running";
        },
      };
      const persist = async (
        structured,
        sessionId,
        capturedSessionMetrics = null,
        capturedSessionEvents = null,
      ) => {
        if (!reservation.claimed_child_id) {
          this.childBindings.claimPublished(
            roleHost.sessionId,
            reservation.label,
            sessionId,
          );
        }
        if (!this.childBindings.activeByChild.has(sessionId)) {
          this.childBindings.openActivation(sessionId, {
            revision: binding.run_state_revision,
            stage_attempt: binding.stage_attempt,
            idempotency_key: binding.idempotency_key,
          });
        }
        const liveSession = this.ctx?.sessions?.get?.(sessionId);
        const sessionEvents = capturedSessionEvents || liveSession?.events;
        const evidence = skillInvocationEvidence(sessionEvents, {
          stage: binding.stage,
          skillName,
          requiresPredictionTool: contract.requiresPredictionTool === true,
          allowDynamicRetrieval: dynamicRetrieval,
        });
        persistedSkillEvidence = evidence;
        const resultDigest = jsonDigest(structured);
        const { allowed_tools: _allowedTools, ...identity } = frozenIdentity;
        return this.sidecar.request("/api/ecology-agent-sidecar/v1/structured-results", {
          body: {
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
            structured,
            result_digest: resultDigest,
            session_metrics: capturedSessionMetrics || dshSessionMetrics(this.ctx, sessionId),
            skill_invocation_evidence: evidence,
          },
        });
      };
      const result = binding.stage === "sample.plan"
        ? await this.#runWorkflowStage({
          binding,
          roleHost,
          reservation,
          prompt,
          outputSchema,
          admission,
          persist,
        })
        : await runStructuredRole(
          roleHost,
          reservation,
          { prompt, outputSchema },
          {
            pendingStarts: this.pendingStarts,
            admission,
            persist: async ({ structured, session_id }) => persist(
              structured,
              session_id,
            ),
            timeoutMs: stageTimeoutMs,
          },
        );
      return {
        structured: result.structured,
        result_digest: jsonDigest(result.structured),
        session_id: result.session_id,
        skill_invocation_evidence: persistedSkillEvidence,
      };
    } finally {
      if (reservation.claimed_child_id) this.childBindings.releaseChild(reservation.claimed_child_id);
      else this.childBindings.revoke(roleHost.sessionId, reservation.label);
    }
  }

  async #runWorkflowStage({
    binding,
    roleHost,
    reservation,
    prompt,
    outputSchema,
    admission,
    persist,
  }) {
    const workflowName = `ecology-wave-${jsonDigest({
      run_id: binding.run_id,
      reservation_id: reservation.launch.reservation_id,
    }).slice(0, 24)}`;
    let childSessionId = null;
    let capturedSessionMetrics = null;
    let capturedSessionEvents = null;
    const removeListener = this.ctx.on?.(
      "workflow/agent-start",
      (info, agent) => {
        if (info?.meta?.name !== workflowName || agent?.label !== reservation.label) return;
        childSessionId = String(agent.childId || "") || null;
        if (!childSessionId) return;
        this.childBindings.claimPublished(
          roleHost.sessionId,
          reservation.label,
          childSessionId,
        );
        if (!this.childBindings.activeByChild.has(childSessionId)) {
          this.childBindings.openActivation(childSessionId, {
            revision: binding.run_state_revision,
            stage_attempt: binding.stage_attempt,
            idempotency_key: binding.idempotency_key,
          });
        }
      },
    );
    const removeEndListener = this.ctx.on?.(
      "workflow/agent-end",
      (info, agent) => {
        if (info?.meta?.name !== workflowName || agent?.label !== reservation.label) return;
        const endedChildId = String(agent.childId || "") || null;
        if (!endedChildId || (childSessionId && endedChildId !== childSessionId)) return;
        capturedSessionMetrics = dshSessionMetrics(this.ctx, endedChildId);
        const endedSession = this.ctx?.sessions?.get?.(endedChildId);
        if (Array.isArray(endedSession?.events)) {
          capturedSessionEvents = structuredClone(endedSession.events);
        }
      },
    );
    let workflow;
    let timeout = null;
    let timedOut = false;
    try {
      workflow = startHomogeneousWorkflow(
        roleHost,
        {
          template_id: "ecology-one-shot-v1",
          workflow_name: workflowName,
          max_total_agents: 1,
          max_concurrent: 1,
          max_items: 1,
          sync_timeout_ms: this.structuredStageTimeoutMs,
        },
        [{ label: reservation.label, prompt, schema: outputSchema }],
      );
      const active = { runId: binding.run_id, workflow };
      this.activeWorkflows.add(active);
      timeout = setTimeout(() => {
        timedOut = true;
        workflow.cancel?.("structured workflow operational timeout");
      }, this.structuredStageTimeoutMs);
      const settled = await workflow.result;
      if (timedOut) throw new Error("structured workflow operational timeout");
      if (settled?.stopReason !== "completed") {
        const error = new Error(
          `structured workflow failed: ${settled?.error || settled?.stopReason || "unknown"}`,
        );
        error.code = settled?.stopReason === "aborted"
          ? "structured_child_aborted"
          : "structured_child_model_error";
        throw error;
      }
      if (!Array.isArray(settled.value) || settled.value.length !== 1) {
        throw new Error("structured workflow returned an invalid result batch");
      }
      const structured = settled.value[0];
      if (!structured || typeof structured !== "object" || Array.isArray(structured)) {
        throw new Error("structured workflow child returned no schema-bound result");
      }
      if (!childSessionId) throw new Error("structured workflow did not publish a real child session");
      if (!await admission.isOpen(reservation)) {
        throw new Error("structured result admission is closed");
      }
      const accepted = await persist(
        structuredClone(structured),
        childSessionId,
        capturedSessionMetrics,
        capturedSessionEvents,
      );
      if (!accepted || accepted.accepted !== true) {
        throw new Error("structured result was not durably accepted");
      }
      return Object.freeze({
        structured: structuredClone(structured),
        receipt: accepted,
        session_id: childSessionId,
      });
    } finally {
      if (timeout !== null) clearTimeout(timeout);
      if (typeof removeListener === "function") removeListener();
      if (typeof removeEndListener === "function") removeEndListener();
      if (workflow) {
        for (const active of this.activeWorkflows) {
          if (active.workflow === workflow) this.activeWorkflows.delete(active);
        }
        await workflow.dispose?.();
      }
    }
  }

  async quiesceRun(runId) {
    this.providerStageGate.cancelRun?.(runId);
    const workflows = [...this.activeWorkflows].filter((item) => item.runId === runId);
    for (const item of workflows) item.workflow.cancel?.("run quiescing");
    await Promise.allSettled(workflows.map((item) => item.workflow.result));
    await Promise.allSettled(workflows.map((item) => item.workflow.dispose?.()));
    for (const item of workflows) this.activeWorkflows.delete(item);
    await this.pendingStarts.cancelAndQuiesce({ runId });
    await this.providerStageGate.drainRun?.(runId);
    this.childBindings.revokeRun(runId);
  }
}

export { STAGES };
