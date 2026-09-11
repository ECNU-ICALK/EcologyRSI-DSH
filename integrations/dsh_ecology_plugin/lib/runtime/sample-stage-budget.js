import { structuredPhaseError } from "./structured-stage-errors.js";
import { dshSessionMetrics } from "./agents.js";

export const SAMPLE_STAGE_LIMITS = Object.freeze({
  "sample.plan": { maxSteps: 10, maxReportedOutputTokens: 24576 },
  "sample.critic": { maxSteps: 4, maxReportedOutputTokens: 8192 },
});

export function checkSampleStageBudget(ctx, sessionId, stage) {
  const limits = SAMPLE_STAGE_LIMITS[stage];
  if (!limits) return;
  const events = ctx.sessions?.get?.(sessionId)?.events || [];
  const steps = events.filter(event => event.type === "step/start").length;
  const usage = dshSessionMetrics(ctx, sessionId).provider_usage;
  const output = usage?.available ? usage.totals.output_tokens : 0;
  if (steps > limits.maxSteps || output > limits.maxReportedOutputTokens) {
    throw structuredPhaseError("execution_budget");
  }
}
