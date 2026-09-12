import { dshSessionMetrics } from "./agents.js";
import { sessionEventLog } from "./session-events.js";

export function sessionActivity(events) {
  const kinds = {
    "assistant/chunk": "streaming", "assistant/message": "streaming",
    "llm/retry": "retrying", "llm/retry-started": "retrying",
    "tool/call": "tool", "tool/result": "tool",
    "step/start": "waiting", "turn/start": "waiting", "turn/end": "settling",
  };
  if (!Array.isArray(events)) return null;
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    const kind = kinds[event?.type];
    if (!kind || !Number.isSafeInteger(event.seq) || event.seq < 0
      || !Number.isSafeInteger(event.time) || event.time <= 0 || event.time > 8640000000000000) continue;
    return {kind, log_revision: event.seq, updated_at: new Date(event.time).toISOString()};
  }
  return null;
}

export function sessionUsageComplete(events, { settlement, freshUsage } = {}) {
  const messages = Array.isArray(events) ? events.filter(e => e?.type === "assistant/message") : [];
  const endings = Array.isArray(events) ? events.filter(e => e?.type === "turn/end") : [];
  return !["active", "cancelled", "timed_out"].includes(settlement) && freshUsage === true && messages.length > 0
    && messages.every(e => e.data?.usage && typeof e.data.usage === "object")
    && endings.length > 0 && endings.every(e => e.data?.reason?.kind === "completed");
}

// Observe the public usage projection, never assistant text or reasoning.
// Accounting is independent of schema validity and result admission.
export function observeSessionUsage(ctx, sidecar, identity, { intervalMs = 5000 } = {}) {
  let queue = Promise.resolve();
  let lastAvailable = null;
  let lastAcknowledged = null;
  let stopped = false;
  const capture = (settlement) => {
    let metrics = dshSessionMetrics(ctx, identity.session_id);
    const freshUsage = metrics.provider_usage.available === true;
    if (freshUsage) lastAvailable = metrics;
    else if (lastAvailable) metrics = lastAvailable;
    const events = sessionEventLog(ctx, identity.session_id);
    const complete = sessionUsageComplete(events, { settlement, freshUsage });
    const body = {
      schema_version: "ecologyrsi-dsh.session-usage/1", identity: { ...identity },
      session_metrics: structuredClone(metrics), settlement, usage_complete: complete,
    };
    const activity = sessionActivity(events);
    if (activity) body.session_metrics.activity = activity;
    // Stream liveness is independent of billing receipts. Persist at most one
    // snapshot per 30-second event-time bucket, plus real state transitions.
    // A silent child never gets a fabricated heartbeat from this polling timer.
    const key = JSON.stringify([body.session_metrics.provider_usage, settlement, complete,
      activity && [activity.kind, Math.floor(Date.parse(activity.updated_at) / 30_000)]]);
    queue = queue.then(async () => {
      if (key === lastAcknowledged) return;
      for (let attempt = 0; attempt < 2; attempt += 1) {
        try {
          const receipt = await sidecar.request("/api/ecology-agent-sidecar/v1/session-usage", {
            method: "POST", timeoutMs: 2000, body,
          });
          if (receipt?.accepted !== true) throw new Error("usage was not accepted");
          lastAcknowledged = key;
          return;
        } catch {
          // An exact replay or later cumulative snapshot recovers transient
          // delivery errors. Missing records remain visible in Host coverage.
        }
      }
    }).catch(() => {});
    return queue;
  };
  capture("active");
  const timer = setInterval(() => { if (!stopped) capture("active"); }, intervalMs);
  timer.unref?.();
  return {
    async settle(settlement) {
      if (stopped) return;
      stopped = true;
      clearInterval(timer);
      await capture(settlement);
    },
  };
}
