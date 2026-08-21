const EMPTY_INPUT = Object.freeze({ type: "object", additionalProperties: false, properties: {} });

function objectInput(properties, required = Object.keys(properties)) {
  return Object.freeze({ type: "object", additionalProperties: false, required, properties });
}

const text = (maxLength = 240) => ({ type: "string", minLength: 1, maxLength });

export const TOOL_DEFINITIONS = Object.freeze({
  ecology_get_run_context: {
    name: "ecology_get_run_context",
    description: "Read the frozen, redacted run and stage context.",
    parameters: EMPTY_INPUT,
  },
  ecology_get_research_evidence: {
    name: "ecology_get_research_evidence",
    description: "Read frozen research evidence for this generation.",
    parameters: EMPTY_INPUT,
  },
  ecology_get_generation_summary: {
    name: "ecology_get_generation_summary",
    description: "Read the redacted aggregate generation summary.",
    parameters: EMPTY_INPUT,
  },
  ecology_get_sample_wave: {
    name: "ecology_get_sample_wave",
    description: "Read one label-free frozen sample wave.",
    parameters: objectInput({ wave_digest: text(64) }),
  },
  ecology_execute_prediction_tool: {
    name: "ecology_execute_prediction_tool",
    description: "Execute one registered prediction tool with structured inputs.",
    parameters: objectInput({
      tool_id: text(160),
      inputs: { type: "object", additionalProperties: true },
    }),
  },
  ecology_submit_sample_decisions: {
    name: "ecology_submit_sample_decisions",
    description: "Persist a bounded set of sample predictions for the active wave.",
    parameters: objectInput({
      schema_version: { const: "ecology-sample-decisions@1" },
      wave_digest: text(64),
      decisions: { type: "array", maxItems: 128, items: objectInput({
        sample_id: text(240), next_tool: text(160), reason_code: text(160),
        confidence: { type: "number", minimum: 0, maximum: 1 },
      }) },
    }),
    concludesTurn: true,
  },
  ecology_get_prediction_summary: {
    name: "ecology_get_prediction_summary",
    description: "Read the frozen label-free prediction summary for review.",
    parameters: objectInput({ wave_digest: text(64) }),
  },
  ecology_submit_sample_review: {
    name: "ecology_submit_sample_review",
    description: "Persist the independent bounded sample review.",
    parameters: objectInput({
      schema_version: { const: "ecology-sample-review@1" },
      wave_digest: text(64),
      decisions: { type: "array", maxItems: 128, items: objectInput({
        sample_id: text(240), next_tool: text(160), reason_code: text(160),
        confidence: { type: "number", minimum: 0, maximum: 1 },
      }) },
    }),
    concludesTurn: true,
  },
});

export const BLOCKED_MODEL_IDENTITY_FIELDS = Object.freeze(new Set([
  "run_id", "role", "stage", "run_state_revision", "stage_attempt",
  "ledger_expected_revision", "session_id", "idempotency_key",
  "child_reservation_id", "activation_lease_id", "genome_digest",
  "compiled_behavior_digest", "phenotype_instance_digest",
]));

export function assertModelArgumentsSafe(value, { labelFree = false, path = "$" } = {}) {
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertModelArgumentsSafe(item, { labelFree, path: `${path}[${index}]` }));
    return;
  }
  if (!value || typeof value !== "object") return;
  for (const [key, item] of Object.entries(value)) {
    const normalized = key.toLowerCase().replaceAll("-", "_");
    if (BLOCKED_MODEL_IDENTITY_FIELDS.has(normalized)) throw new Error(`model argument cannot set Host identity: ${path}.${key}`);
    if (labelFree && /(^|_)(observed|observation|label|ground_truth|target_value)(_|$)/.test(normalized)) {
      throw new Error(`label-bearing planner argument is forbidden: ${path}.${key}`);
    }
    assertModelArgumentsSafe(item, { labelFree, path: `${path}.${key}` });
  }
}
