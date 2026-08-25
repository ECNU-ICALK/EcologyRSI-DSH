import assert from "node:assert/strict";
import test from "node:test";

import { registerRoleTools } from "../lib/tools/roles.js";

function roleHarness({ replay, search, complete }) {
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
  registerRoleTools(ctx, {
    role: "researcher",
    toolProfile: "dynamic-retrieval-v1",
    bridge: {
      bindingFor: async (_exec, expected) => ({
        role: expected.role,
        run_id: "run-1",
        stage: "generation.research",
      }),
      sidecar: {
        request: async (path, options) => {
          requests.push({ path, options });
          if (path.endsWith("/replay")) return replay(options.body);
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
    args.queries.map((query) => ({ query, maxResults: 8 })),
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
