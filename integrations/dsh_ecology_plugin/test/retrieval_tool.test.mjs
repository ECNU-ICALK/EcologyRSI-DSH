import assert from "node:assert/strict";
import test from "node:test";

import { registerRoleTools } from "../lib/tools/roles.js";
import { RETRIEVAL_POLICY_DEFAULTS, RETRIEVAL_KEY_ERROR, normalizeRetrievalArguments } from "../lib/tools/retrieval.js";

const policy = { online_enabled: true, ...RETRIEVAL_POLICY_DEFAULTS };

function roleHarness({ replay, search, complete, webAvailable = true, bindingExtra = {} }) {
  const handlers = new Map();
  const requests = [];
  const searches = [];
  const ctx = {
    tools: {
      register: (definition) => {
        handlers.set(definition.name, definition.execute);
        return () => {};
      },
    },
    web: {
      search: async (request, signal) => {
        searches.push({ request, signal });
        return search(request, signal);
      },
    },
  };
  if (!webAvailable) delete ctx.web;
  registerRoleTools(ctx, {
    role: "researcher",
    toolProfile: "dynamic-retrieval-v1",
    bridge: {
      bindingFor: async (_exec, expected) => ({
        role: expected.role,
        run_id: "run-1",
        stage: "generation.research",
        ...bindingExtra,
      }),
      sidecar: {
        request: async (path, options) => {
          requests.push({ path, options });
          if (path.endsWith("/replay")) return { policy, ...await replay(options.body) };
          if (path.endsWith("/complete")) return complete(options.body);
          throw new Error(`unexpected sidecar path: ${path}`);
        },
      },
    },
  });
  return { handler: handlers.get("web_search"), handlers, requests, searches };
}

const argumentsValue = {
  queries: ["greenhouse temperature forecasting"],
  retrieval_key: "check-temperature",
};

test("invalid retrieval key receives actionable repair feedback before any external work", async () => {
  const harness = roleHarness({
    replay: async () => { throw new Error("must not access the sidecar"); },
    search: async () => { throw new Error("must not search"); },
    complete: async () => { throw new Error("must not complete a rejected call"); },
  });
  await assert.rejects(harness.handler({ ...argumentsValue,
    retrieval_key: "origin-vector-review/co2-setpoint-dynamics" }, { agent: { id: "child-1" } }),
    error => error.message === RETRIEVAL_KEY_ERROR
      && error.message.includes("slashes")
      && /retry web_search before structured_output/.test(error.message));
  assert.equal(harness.requests.length, 0);
  assert.equal(harness.searches.length, 0);
  assert.equal(normalizeRetrievalArguments({ ...argumentsValue,
    retrieval_key: "origin-review-co2-1" }).retrieval_key, "origin-review-co2-1");
});

const frozenLibrary = {
  mode: "frozen_snapshot", snapshot_digest: "e".repeat(64), document_count: 2, online_enabled: false,
};

test("formal retrieval searches the frozen library while offline without any provider or completion call", async () => {
  for (const sources of [[], [{ url: "https://example.org/citation-only", title: "Frozen evidence" }]]) {
    const result = { sources, provider_route: "frozen_snapshot", snapshot_digest: frozenLibrary.snapshot_digest };
    const harness = roleHarness({
      bindingExtra: { formal_retrieval: frozenLibrary },
      replay: async body => {
        assert.equal(Object.hasOwn(body.identity, "formal_retrieval"), false);
        assert.deepEqual(body.arguments, argumentsValue);
        return { found: true, result, policy: { ...policy, online_enabled: false } };
      },
      search: async () => { throw new Error("network forbidden"); },
      complete: async () => { throw new Error("external completion forbidden"); },
    });
    assert.deepEqual(await harness.handler(argumentsValue, {}), result);
    assert.equal(harness.searches.length, 0);
    assert.equal(harness.requests.length, 1);
  }
});

test("formal retrieval fails closed on legacy misses, malformed snapshots and external routes", async () => {
  for (const response of [
    {},
    { found: false },
    { found: true, result: null },
    { found: true, result: { provider_route: "dsh_primary", snapshot_digest: frozenLibrary.snapshot_digest, sources: [] } },
    { found: true, result: { provider_route: "frozen_snapshot", sources: [] } },
    { found: true, result: { provider_route: "frozen_snapshot", snapshot_digest: "a".repeat(64), sources: [] } },
    { found: true, result: { provider_route: "frozen_snapshot", snapshot_digest: frozenLibrary.snapshot_digest } },
  ]) {
    const harness = roleHarness({
      bindingExtra: { formal_retrieval: frozenLibrary },
      replay: async () => response,
      search: async () => { throw new Error("network forbidden"); },
      complete: async () => { throw new Error("external completion forbidden"); },
    });
    await assert.rejects(harness.handler(argumentsValue, {}), /frozen snapshot result.*network fallback is forbidden/);
    assert.equal(harness.searches.length, 0);
    assert.equal(harness.requests.length, 1);
  }
});

test("invalid bound formal policy and model-authored overrides are rejected before retrieval", async () => {
  for (const formal of [null, {}, { ...frozenLibrary, online_enabled: true },
    { ...frozenLibrary, snapshot_digest: "" }, { ...frozenLibrary, document_count: -1 }]) {
    const harness = roleHarness({
      bindingExtra: { formal_retrieval: formal },
      replay: async () => { throw new Error("invalid policy cannot reach sidecar"); },
    });
    await assert.rejects(harness.handler(argumentsValue, {}), /formal retrieval requires/);
    assert.equal(harness.requests.length, 0);
    assert.equal(harness.searches.length, 0);
  }
  const harness = roleHarness({ bindingExtra: { formal_retrieval: frozenLibrary } });
  await assert.rejects(harness.handler({
    ...argumentsValue, formal_retrieval: { online_enabled: true },
  }, {}), /invalid shape/);
  assert.equal(harness.requests.length, 0);
});

test("web_search replays before touching the DSH web service", async () => {
  const persisted = {
    sources: [],
    truncated: false,
    provider_route: "dsh_primary",
    fallback_reason: null,
    result_digest: "a".repeat(64),
  };
  const harness = roleHarness({
    replay: async () => ({ found: true, result: persisted }),
    search: async () => { throw new Error("DSH search must not run on replay"); },
    complete: async () => { throw new Error("completion must not run on replay"); },
  });

  const result = await harness.handler(argumentsValue, { agent: { id: "child-1" } });

  assert.deepEqual(result, persisted);
  assert.equal(harness.searches.length, 0);
  assert.equal(harness.requests.length, 1);
  assert.match(harness.requests[0].path, /retrievals\/replay$/);
});

test("web_search collects DSH primary queries and delegates quality to the sidecar", async () => {
  let completedBody;
  const harness = roleHarness({
    replay: async () => ({ found: false, result: null }),
    search: async ({ query }) => ({
      content: `answer:${query}`,
      sources: [{ url: `https://example.org/${encodeURIComponent(query)}`, title: query }],
      truncated: false,
    }),
    complete: async (body) => {
      completedBody = body;
      return {
        ...body.primary_result,
        provider_route: "dsh_primary",
        fallback_reason: null,
        result_digest: "b".repeat(64),
      };
    },
  });
  const args = {
    queries: ["greenhouse temperature forecasting", "greenhouse humidity forecasting"],
    retrieval_key: "compare-targets",
  };

  const result = await harness.handler(args, { agent: { id: "child-2" } });

  assert.deepEqual(
    harness.searches.map((item) => item.request),
    args.queries.map((query) => ({ query, maxResults: 16 })),
  );
  assert.equal(completedBody.primary_result.sources.length, 2);
  assert.deepEqual(completedBody.arguments, args);
  assert.equal(result.provider_route, "dsh_primary");
  assert.equal(harness.requests.at(-1).options.timeoutMs, 420_000);
});

test("web_search maps provider failure and preserves partial primary evidence", async () => {
  let completedBody;
  const providerError = Object.assign(new Error("credential details must stay private"), {
    code: "WEB_PROVIDER_CREDENTIAL_MISSING",
  });
  const harness = roleHarness({
    replay: async () => ({ found: false, result: null }),
    search: async ({ query }) => {
      if (query.includes("humidity")) throw providerError;
      return {
        sources: [{ url: "https://example.org/temperature", title: query }],
        truncated: false,
      };
    },
    complete: async (body) => {
      completedBody = body;
      return {
        sources: body.primary_result.sources,
        truncated: false,
        provider_route: "dsh_primary_then_openalex_fallback",
        fallback_reason: body.primary_error_code,
        result_digest: "c".repeat(64),
      };
    },
  });

  await harness.handler(
    {
      queries: ["greenhouse temperature forecasting", "greenhouse humidity forecasting"],
      retrieval_key: "partial-query",
    },
    { agent: { id: "child-3" } },
  );

  assert.equal(completedBody.primary_error_code, "primary_provider_unavailable");
  assert.equal(completedBody.primary_result.sources.length, 1);
  assert.doesNotMatch(JSON.stringify(completedBody), /credential details/);
});

test("web_search propagates explicit cancellation without completion", async () => {
  const aborted = Object.assign(new Error("aborted"), { code: "WEB_ABORTED" });
  const harness = roleHarness({
    replay: async () => ({ found: false, result: null }),
    search: async () => { throw aborted; },
    complete: async () => { throw new Error("completion must not run after abort"); },
  });

  await assert.rejects(
    harness.handler(argumentsValue, { agent: { id: "child-4" } }),
    (error) => error === aborted,
  );
  assert.equal(harness.requests.length, 1);
});

test("Host refusal or disabled policy prevents any provider request", async () => {
  for (const replay of [
    async () => { throw new Error("retrieval stage call budget is exhausted"); },
    async () => ({ found: false, policy: { ...policy, online_enabled: false } }),
    async () => ({ found: false, policy: null }),
    async () => ({ found: false, policy: {} }),
  ]) {
    const harness = roleHarness({
      replay,
      search: async () => { throw new Error("provider must not run"); },
      complete: async () => { throw new Error("completion must not run"); },
    });
    await assert.rejects(harness.handler(argumentsValue, {}), /budget|policy/);
    assert.equal(harness.searches.length, 0);
    assert.equal(harness.requests.length, 1);
  }
});

test("missing DSH web service can replay or use the Host fallback", async () => {
  const saved = { sources: [], result_digest: "d".repeat(64) };
  const replayed = roleHarness({
    webAvailable: false,
    replay: async () => ({ found: true, result: saved }),
    complete: async () => { throw new Error("must replay"); },
  });
  assert.deepEqual(await replayed.handler(argumentsValue, {}), saved);
  const fallback = roleHarness({
    webAvailable: false,
    replay: async () => ({ found: false }),
    complete: async (body) => {
      assert.equal(body.primary_error_code, "primary_provider_unavailable");
      return saved;
    },
  });
  assert.deepEqual(await fallback.handler(argumentsValue, {}), saved);
  assert.equal(fallback.searches.length, 0);
});

test("retrieval respects Host query limits and expanded result limits", async () => {
  const restrictive = roleHarness({
    replay: async () => ({ found: false, policy: { ...policy, max_queries_per_call: 1 } }),
    search: async () => { throw new Error("provider must not run"); },
  });
  await assert.rejects(restrictive.handler({
    ...argumentsValue, queries: ["one", "two"],
  }, {}), /one to 1 queries/);
  assert.equal(restrictive.searches.length, 0);

  let completedBody;
  const expanded = roleHarness({
    replay: async () => ({ found: false, policy: { ...policy, max_results: 24 } }),
    search: async ({ query, maxResults }) => {
      assert.equal(maxResults, 24);
      return {
        sources: Array.from({ length: 24 }, (_, i) => ({
          url: `https://example.org/${query}/${i}`, title: `${query} ${i}`,
        })),
      };
    },
    complete: async (body) => { completedBody = body; return body.primary_result; },
  });
  const result = await expanded.handler({
    queries: ["mechanisms", "alternatives", "counterevidence"],
    retrieval_key: "coverage",
  }, {});
  assert.equal(result.sources.length, 24);
  assert.equal(result.truncated, true);
  assert.deepEqual(result.sources.slice(0, 3).map((source) => source.title),
    ["mechanisms 0", "alternatives 0", "counterevidence 0"]);
  assert.equal(completedBody.arguments.queries.length, 3);
});

test("dynamic queries use expanded Unicode character bounds without truncation", () => {
  const query = "🌱".repeat(600);
  assert.equal(normalizeRetrievalArguments({ queries: [query], retrieval_key: "unicode" }).queries[0], query);
  assert.throws(() => normalizeRetrievalArguments({
    queries: [query + "a"], retrieval_key: "unicode",
  }), /too long/);
  assert.throws(() => normalizeRetrievalArguments({
    queries: Array.from({ length: 13 }, (_, i) => `query ${i}`), retrieval_key: "many",
  }), /one to 12/);
});
