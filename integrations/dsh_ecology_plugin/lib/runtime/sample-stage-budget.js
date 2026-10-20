import { structuredPhaseError } from "./structured-stage-errors.js";
import { sessionEventLog } from "./session-events.js";

// Keep these equal to sample_execution_limits in evolution/schedule.py.
// Cumulative provider usage is observability, not an implicit Token quota.
// Bound execution with steps, the frozen tool-call allowance and the per-call
// response ceiling; a valid multi-step result may exceed two full responses.
export const SAMPLE_STAGE_LIMITS = Object.freeze({
  "sample.plan": { maxSteps: 10, maxReportedOutputTokens: null },
  "sample.critic": { maxSteps: 4, maxReportedOutputTokens: null },
});

export function checkSampleStageBudget(ctx, sessionId, stage) {
  const limits = SAMPLE_STAGE_LIMITS[stage];
  if (!limits) return;
  const events = sessionEventLog(ctx, sessionId) || [];
  const steps = events.filter(event => event.type === "step/start").length;
  if (steps > limits.maxSteps) {
    throw structuredPhaseError("execution_budget");
  }
}
