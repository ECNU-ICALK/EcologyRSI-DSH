#!/usr/bin/env node
// Explicit offline integration: real installed adapter -> loopback fake HTTP.
// No settings files, user credentials or production services are read.
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { dshPackageJson } from "../../../scripts/install_dsh_ecology_runtime.mjs";
import { withThinkingCapabilities } from "../../../scripts/configure_dsh_reasoning.mjs";

const packageJson = await dshPackageJson(process.env.DSH_BIN || "dsh");
assert.ok(packageJson, "DSH_BIN must name an installed Harness executable");
assert.equal(JSON.parse(await readFile(packageJson, "utf8")).version, "0.2.0-rc.2");
const requireHost = createRequire(packageJson);
const { Config, apply } = await import(pathToFileURL(requireHost.resolve("@deepseek-ai/dsh-llm-pi-ai")).href);
const captured = [];
const server = createServer(async (req, res) => {
  assert.equal(req.url, "/v1/chat/completions");
  let raw = "";
  for await (const chunk of req) raw += chunk;
  const body = JSON.parse(raw); captured.push(body);
  res.writeHead(200, { "content-type": "text/event-stream" });
  const frame = (delta, finish_reason, usage) => "data: " + JSON.stringify({
    id: "local-reasoning-fixture", object: "chat.completion.chunk", created: 1, model: body.model,
    choices: [{ index: 0, delta, finish_reason }], ...(usage ? { usage } : {}),
  }) + "\n\n";
  res.end(frame({ role: "assistant", content: "local fixture" }, null)
    + frame({}, "stop", { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 }) + "data: [DONE]\n\n");
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const endpoint = `http://127.0.0.1:${server.address().port}/v1`;
const report = { schema_version: "ecologyrsi-dsh.reasoning-wire-verification/1", harness_version: "0.2.0-rc.2",
  transport: "installed_PiAiAdapter_to_loopback_fake_HTTP", remote_model_calls: 0,
  user_settings_read: false, user_credentials_read: false, cases: [] };

function adapterFor(model, defaultEffort) {
  const input = { "llm-pi-ai": { providers: { "local-reasoning-fixture": {
    api: "openai-completions", baseURL: endpoint, apiKeyEnv: "LOCAL_FIXTURE_KEY",
    ...(defaultEffort ? { reasoning: defaultEffort } : {}),
    models: [{ id: model, contextWindow: 8192, maxTokens: 2048 }],
  } } } };
  const generated = withThinkingCapabilities(input, `local-reasoning-fixture/${model}`);
  let adapter;
  const ctx = {
    fiber: { entry: { options: { id: "local-reasoning-fixture" } } },
    inject() {}, on() { return () => {}; },
    get(name) {
      assert.equal(name, "credentials", "unexpected service access in isolated adapter test");
      return { resolve: async ref => {
        assert.equal(ref, "LOCAL_FIXTURE_KEY");
        return { value: "local-test-placeholder" };
      } };
    },
    llm: {
      registerConfigurableProviders() { return { replace() {} }; },
      registerModelDiscovery() {},
      registerAdapter(_routes, value) { adapter = value; return { replace() {} }; },
    },
    logger: { warn() {}, error(error) { throw error; } },
  };
  apply(ctx, Config(generated["llm-pi-ai"]));
  assert.ok(adapter);
  return adapter;
}

try {
  for (const model of ["glm-5.2", "glm-5.3", "glm-5.3-flash"]) {
    const adapter = adapterFor(model);
    const info = await adapter.resolveModel("local-reasoning-fixture", model);
    assert.equal(info.reasoning.defaultEffort, "max");
    const cases = model === "glm-5.2"
      ? [[undefined, "max"], ["minimal", "minimal"], ["low", "high"], ["medium", "high"], ["high", "high"], ["xhigh", "max"], ["max", "max"], ["off", undefined]]
      : [[undefined, "max"], ["low", "low"], ["high", "high"], ["max", "max"]];
    for (const [effort, expected] of cases) {
      const before = captured.length;
      for await (const _chunk of adapter.stream({ provider: "local-reasoning-fixture", model,
        messages: [{ role: "user", content: [{ type: "text", text: "local fixture" }] }],
        maxTokens: 256, signal: AbortSignal.timeout(5000),
        ...(effort === undefined ? {} : { reasoningEffort: effort }),
      })) { /* consume the real adapter stream */ }
      assert.equal(captured.length, before + 1);
      const payload = captured.at(-1);
      assert.equal(payload.reasoning_effort, expected);
      assert.equal(payload.thinking?.type, effort === "off" ? "disabled" : "enabled");
      assert.equal(payload.max_tokens, 256);
      report.cases.push({ model, requested_effort: effort ?? "provider_default",
        wire_reasoning_effort: payload.reasoning_effort ?? null, wire_thinking_type: payload.thinking.type });
    }
    if (model !== "glm-5.2") {
      const before = captured.length;
      await assert.rejects(async () => {
        for await (const _chunk of adapter.stream({ provider: "local-reasoning-fixture", model,
          messages: [], reasoningEffort: "off", signal: AbortSignal.timeout(5000) })) {}
      }, error => error.code === "UNSUPPORTED_REASONING_EFFORT");
      assert.equal(captured.length, before, "unsupported effort must fail before HTTP");
    }
  }
  const adapter = adapterFor("glm-5.3", "low");
  assert.equal((await adapter.resolveModel("local-reasoning-fixture", "glm-5.3")).reasoning.defaultEffort, "low");
  for await (const _chunk of adapter.stream({ provider: "local-reasoning-fixture", model: "glm-5.3",
    messages: [{ role: "user", content: [{ type: "text", text: "local fixture" }] }], signal: AbortSignal.timeout(5000) })) {}
  assert.equal(captured.at(-1).reasoning_effort, "low");
  assert.equal(captured.at(-1).thinking.type, "enabled");
  report.cases.push({ model: "glm-5.3", requested_effort: "preserved_provider_default", wire_reasoning_effort: "low", wire_thinking_type: "enabled" });
  report.passed = true;
  if (process.argv[2]) await writeFile(path.resolve(process.argv[2]), JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify({ passed: true, payload_cases: report.cases.length,
    unsupported_efforts_rejected_before_http: 2, remote_model_calls: 0 }));
} finally {
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
