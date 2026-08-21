import { createHash, randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";

import { ROLE_TOOL_NAMES } from "../tools/roles.js";
import { SidecarClient } from "../sidecar/client.js";
import { dshSessionMetrics } from "./agents.js";
import { ChildBindingRegistry } from "./child-bindings.js";
import { runStructuredRole } from "./structured-roles.js";
import { PendingChildStarts, startHomogeneousWorkflow } from "./workflows.js";

const STAGES = Object.freeze({
  "generation.research": Object.freeze({ role: "researcher", schema: "ecology-research-result@1", file: "research-result" }),
  "candidate.propose": Object.freeze({
    role: "candidate-proposer",
    schema: "ecology-genome-mutation@1",
    file: "genome-mutation",
    instruction: [
      "The operations array is a mutation delta over parent_genome, not a replacement genome.",
      "Omit every unchanged parameter, policy, instruction, tool policy, and workflow setting.",
      "Choose 1-4 operations that directly address the strongest research evidence and avoid exact failed parameter sets.",
      "Do not reconstruct or repeat the parent genome.",
    ].join(" "),
  }),
  "generation.judge": Object.freeze({ role: "generation-judge", schema: "ecology-generation-review@1", file: "generation-review" }),
  "sample.plan": Object.freeze({ role: "sample-planner", schema: "ecology-sample-decisions@1", file: "sample-decisions" }),
  "sample.critic": Object.freeze({ role: "sample-critic", schema: "ecology-sample-review@1", file: "sample-review" }),
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

export class NativeStageRunner {
  constructor(ctx, { roleAgents, runRegistry, sidecar, structuredStageTimeoutMs = 600_000 } = {}) {
    this.ctx = ctx;
    this.roleAgents = roleAgents;
    this.runRegistry = runRegistry;
    this.sidecar = sidecar;
    this.structuredStageTimeoutMs = structuredStageTimeoutMs;
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
      allowed_tools: ROLE_TOOL_NAMES[contract.role],
    };
    const reservation = this.childBindings.reserve(roleHost.sessionId, launch, frozenIdentity);
    try {
      const outputSchema = await this.schema(contract.file);
      const prompt = canonical({
        instruction: [
          "Do not narrate analysis.",
          "In your first response, call structured_output exactly once with one concise object matching the supplied output schema.",
          "Do not emit prose before or after it.",
          contract.instruction || "",
        ].filter(Boolean).join(" "),
        stage: binding.stage,
        context: request.context,
      });
      const admission = {
        isOpen: async () => {
          const current = this.runRegistry.get(binding.run_id);
          return current && current.status !== "cancelled";
        },
      };
      const persist = async (structured, sessionId) => {
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
            session_metrics: dshSessionMetrics(this.ctx, sessionId),
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
            timeoutMs: this.structuredStageTimeoutMs,
          },
        );
      return {
        structured: result.structured,
        result_digest: jsonDigest(result.structured),
        session_id: result.session_id,
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
        throw new Error(`structured workflow failed: ${settled?.error || settled?.stopReason || "unknown"}`);
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
      const accepted = await persist(structuredClone(structured), childSessionId);
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
      if (workflow) {
        for (const active of this.activeWorkflows) {
          if (active.workflow === workflow) this.activeWorkflows.delete(active);
        }
        await workflow.dispose?.();
      }
    }
  }

  async quiesceRun(runId) {
    const workflows = [...this.activeWorkflows].filter((item) => item.runId === runId);
    for (const item of workflows) item.workflow.cancel?.("run quiescing");
    await Promise.allSettled(workflows.map((item) => item.workflow.result));
    await Promise.allSettled(workflows.map((item) => item.workflow.dispose?.()));
    for (const item of workflows) this.activeWorkflows.delete(item);
    await this.pendingStarts.cancelAndQuiesce({ runId });
    this.childBindings.revokeRun(runId);
  }
}

export { STAGES };
