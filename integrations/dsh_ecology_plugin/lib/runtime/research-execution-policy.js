// Keep these exact fields in sync with core/model_execution_policy.py.
export const RESEARCH_EXECUTION_POLICY = Object.freeze({
  schema_version: "ecologyrsi-dsh.research-execution-policy/1",
  synthesis_context_format: "compact@1",
  synthesis_report_format: "concise@2",
  synthesis_max_output_tokens: 16384,
  retry_identical_exhausted_request: false,
});

export function researchExecutionPolicy(context) {
  if (!context || !Object.hasOwn(context, "research_execution_policy")) return null;
  const value = context.research_execution_policy;
  const keys = Object.keys(RESEARCH_EXECUTION_POLICY);
  if (!value || typeof value !== "object" || Array.isArray(value)
    || Object.keys(value).length !== keys.length
    || keys.some(key => value[key] !== RESEARCH_EXECUTION_POLICY[key])) {
    throw new Error("invalid frozen research execution policy");
  }
  return RESEARCH_EXECUTION_POLICY;
}

export function validateStructuredStageBudget(stage, maxTokens, context, label) {
  const policy = stage === "generation.research-synthesis" ? researchExecutionPolicy(context) : null;
  const maximum = policy?.synthesis_max_output_tokens ?? 8192;
  if (maxTokens !== undefined && (!Number.isSafeInteger(maxTokens) || maxTokens < 512 || maxTokens > maximum)) {
    throw new Error(`${label} must be between 512 and ${maximum}`);
  }
  if (policy && maxTokens !== maximum) {
    throw new Error("DSH synthesis max_tokens must match the frozen research execution policy");
  }
}
