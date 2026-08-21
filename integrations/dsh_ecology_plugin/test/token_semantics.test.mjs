import assert from "node:assert/strict";
import test from "node:test";

import { dshSessionMetrics, tokenSemantics } from "../lib/runtime/agents.js";

test("missing projection never fabricates cumulative usage from pressure", () => {
  const result = tokenSemantics({ tokenMeter: { measure: () => ({ used: 10, limit: 20 }) } }, {});
  assert.deepEqual(result.context_pressure, { used: 10, limit: 20 });
  assert.equal(result.provider_usage, null);
});

test("DSH token usage projection is read from its public flat four-bucket shape", () => {
  const session = {};
  const ctx = {
    sessions: { get: () => session },
    tokenMeter: {
      measure: () => ({
        logRevision: 12,
        baseline: { kind: "usage", tokens: 140 },
        totalTokens: 145,
        surfaceTokens: 90,
      }),
    },
    sessionProjections: {
      snapshot: () => ({
        values: {
          tokenUsage: {
            uncachedInputTokens: 100,
            outputTokens: 20,
            cacheReadTokens: 30,
            cacheWriteTokens: 0,
          },
        },
      }),
    },
  };
  assert.deepEqual(dshSessionMetrics(ctx, "child-1").provider_usage.totals, {
    uncached_input_tokens: 100,
    output_tokens: 20,
    cache_read_tokens: 30,
    cache_write_tokens: 0,
    total_tokens: 150,
  });
});

test("token observability failure never rejects an otherwise valid structured result", () => {
  const session = {};
  const result = dshSessionMetrics({
    sessions: { get: () => session },
    tokenMeter: { measure: () => { throw new Error("cold session"); } },
    sessionProjections: { snapshot: () => { throw new Error("projection unavailable"); } },
  }, "child-2");
  assert.deepEqual(result.context_pressure, {
    available: false,
    source: "dsh_token_meter",
  });
  assert.deepEqual(result.provider_usage, {
    available: false,
    source: "dsh_session_projection_token_usage",
  });
});
