function objectInput(properties, required = Object.keys(properties)) {
  return Object.freeze({ type: "object", additionalProperties: false, required, properties });
}

const text = (maxLength = 240) => ({ type: "string", minLength: 1, maxLength });

export const TOOL_DEFINITIONS = Object.freeze({
  web_search: {
    name: "web_search",
    description: "Search for evidence needed at the current reasoning step. DSH chooses the primary provider and EcologyRSI automatically handles fallback; provide queries only, never a provider.",
    parameters: objectInput({
      queries: {
        type: "array",
        minItems: 1,
        maxItems: 4,
        items: text(180),
      },
      retrieval_key: text(120),
    }),
  },
  ecology_execute_prediction_tool: {
    name: "ecology_execute_prediction_tool",
    description: "Call an available prediction capability for this origin. Choose tool_id and allowed parameters; use a unique call_id. Results are evidence for your final prediction. Maximum six calls.",
    parameters: objectInput({
      tool_id: text(160),
      wave_digest: text(64),
      call_id: text(80),
      parameters: { type: "object", additionalProperties: { type: "number" } },
    }),
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
