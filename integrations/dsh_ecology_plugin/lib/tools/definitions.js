import { RETRIEVAL_KEY_CONTRACT } from "./retrieval.js";

function objectInput(properties, required = Object.keys(properties)) {
  return Object.freeze({ type: "object", additionalProperties: false, required, properties });
}

const text = (maxLength = 240) => ({ type: "string", minLength: 1, maxLength });

export const TOOL_DEFINITIONS = Object.freeze({
  web_search: {
    name: "web_search",
    description: `Search when evidence would help the current reasoning step, including while planning, comparing or revising a proposal. In formal evaluation, formal_retrieval restricts this tool to the Host's frozen evidence library: queries may evolve, the library cannot change, and returned URLs are citations only and must never be read remotely. Otherwise follow the Host retrieval_policy budget and online switch. Prefer focused queries of at most 128 Unicode characters when feasible for compatibility across online sources; site:, lang:, after: and before: filters are supported by online search. Routing and fallback are Host-owned; provide queries and retrieval_key, never a provider. ${RETRIEVAL_KEY_CONTRACT}.`,
    parameters: objectInput({
      queries: {
        type: "array",
        minItems: 1,
        maxItems: 12,
        items: text(600),
      },
      retrieval_key: { ...text(120), description: RETRIEVAL_KEY_CONTRACT },
    }),
  },
  ecology_execute_prediction_tool: {
    name: "ecology_execute_prediction_tool",
    description: "Call an available prediction capability for this origin. Choose tool_id and allowed parameters; use a unique call_id for new computation. After interruption, an exact recoverable_call_requests entry may be replayed using its original call_id and parameters to retrieve existing evidence even with remaining_calls=0; wait for its result before citing it. Results are evidence for your final prediction. Follow the current frozen prediction_contract.max_prediction_tool_calls budget and the Host-reported remaining_calls. Failed computations also consume budget.",
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
  "formal_retrieval", "comparison_retrieval", "comparison_retrieval_policy",
  "comparison_retrieval_scope", "snapshot_digest",
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
