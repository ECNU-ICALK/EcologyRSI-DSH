import { structuredPhaseError } from "./structured-stage-errors.js";
import { dshSessionMetrics } from "./agents.js";
import { sessionEventLog } from "./session-events.js";

// Keep these equal to sample_execution_limits in evolution/schedule.py. The
// reported thresholds abort a runaway child at twice one full response of the
// stage's frozen per-call output budget, not at a legitimate verbose one.
export const SAMPLE_STAGE_LIMITS = Object.freeze({
  "sample.plan": { maxSteps: 10, maxReportedOutputTokens: 32768 },
  "sample.critic": { maxSteps: 4, maxReportedOutputTokens: 16384 },
});

export function checkSampleStageBudget(ctx, sessionId, stage) {
  const limits = SAMPLE_STAGE_LIMITS[stage];
  if (!limits) return;
  const events = sessionEventLog(ctx, sessionId) || [];
  const steps = events.filter(event => event.type === "step/start").length;
  const usage = dshSessionMetrics(ctx, sessionId).provider_usage;
  const output = usage?.available ? usage.totals.output_tokens : 0;
  if (steps > limits.maxSteps || output > limits.maxReportedOutputTokens) {
    throw structuredPhaseError("execution_budget");
  }
}
