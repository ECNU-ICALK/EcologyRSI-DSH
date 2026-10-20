import assert from "node:assert/strict";
import test from "node:test";
import { withThinkingCapabilities, withThinkingCapabilitiesForRoutes } from "../../../scripts/configure_dsh_reasoning.mjs";

function settings(ids, fields = {}) {
  return { "llm-pi-ai": { providers: { local: {
    api: "openai-completions", baseURL: "http://127.0.0.1:1/v1",
    apiKeyEnv: "TEST_ONLY_REFERENCE", headers: { "x-test": "preserve" },
    models: ids.map(id => ({ id })), ...fields,
  } } }, unrelated: { preserved: true } };
}
const provider = value => value["llm-pi-ai"].providers.local;

test("GLM generations declare exact API effort mappings and safe provider defaults", () => {
  for (const id of ["glm-5.2", "glm-5.3", "glm-5.3-flash"]) {
    const original = settings([id]);
    const p = provider(withThinkingCapabilities(original, `local/${id}`));
    assert.equal(p.reasoning, "max");
    assert.equal(provider(original).reasoning, undefined);
    assert.equal(p.models[0].compat.thinkingFormat, "zai");
    assert.equal(p.models[0].compat.supportsReasoningEffort, true);
    if (id !== "glm-5.2") assert.deepEqual(p.models[0].reasoningEfforts, { low: "low", high: "high", max: "max" });
    else assert.deepEqual(p.models[0].reasoningEfforts, { off: null, minimal: "minimal", low: "high", medium: "high", high: "high", xhigh: "max", max: "max" });
  }
  for (const id of ["glm-5", "glm-5.1", "glm-4.7", "deepseek-v4-flash-0731", "deepseek-v4-pro"]) {
    const p = provider(withThinkingCapabilities(settings([id]), `local/${id}`));
    assert.equal(p.reasoning, "high");
    assert.equal(p.models[0].reasoningEfforts.off, null);
  }
});

test("batch defaults are order independent and never silently affect unselected models", () => {
  const input = settings(["glm-5.2", "glm-5.3-flash"]);
  const routes = ["local/glm-5.2", "local/glm-5.3-flash"];
  assert.deepEqual(withThinkingCapabilitiesForRoutes(input, routes), withThinkingCapabilitiesForRoutes(input, [...routes].reverse()));
  assert.throws(() => withThinkingCapabilities(input, routes[0]), /unselected models/);
  const mixed = settings(["glm-5.2", "deepseek-v4-pro"]);
  assert.throws(() => withThinkingCapabilitiesForRoutes(mixed, ["local/glm-5.2", "local/deepseek-v4-pro"]), /explicit common reasoning default/);
  provider(mixed).reasoning = "high";
  assert.equal(provider(withThinkingCapabilitiesForRoutes(mixed, ["local/glm-5.2", "local/deepseek-v4-pro"])).reasoning, "high");
});

test("explicit compatible settings and unrelated fields survive without input mutation", () => {
  const input = settings(["glm-5.3", "unselected"], { reasoning: "low", custom: "keep",
    compat: { supportsStore: true, supportsDeveloperRole: true } });
  provider(input).models[0].reasoningEfforts = { low: "low", max: "max" };
  provider(input).models[0].compat = { maxTokensField: "max_completion_tokens", supportsStrictMode: true };
  const before = structuredClone(input), result = withThinkingCapabilities(input, "local/glm-5.3");
  assert.deepEqual(input, before);
  const p = provider(result);
  assert.equal(p.reasoning, "low");
  assert.deepEqual(p.models[0].reasoningEfforts, { low: "low", max: "max" });
  assert.equal(p.models[0].compat.supportsStore, true);
  assert.equal(p.models[0].compat.supportsDeveloperRole, true);
  assert.equal(p.models[0].compat.supportsStrictMode, true);
  assert.equal(p.models[0].compat.maxTokensField, "max_completion_tokens");
  for (const key of ["baseURL", "apiKeyEnv", "headers", "custom"]) assert.deepEqual(p[key], provider(before)[key]);
  assert.deepEqual(p.models[1], provider(before).models[1]);
  assert.deepEqual(result.unrelated, input.unrelated);
});

test("unsupported models and explicit capability/default/compatibility conflicts fail clearly", () => {
  for (const [id, effort] of [["glm-5.3", "off"], ["glm-5.3-flash", "minimal"], ["glm-5.2", "none"]]) {
    assert.throws(() => withThinkingCapabilities(settings([id], { reasoning: effort }), `local/${id}`), /reasoning default conflicts/);
  }
  for (const efforts of [false, {}, { off: null, high: "high" }, { low: "medium" }]) {
    const input = settings(["glm-5.3"]); provider(input).models[0].reasoningEfforts = efforts;
    assert.throws(() => withThinkingCapabilities(input, "local/glm-5.3"), /capability conflict/);
  }
  for (const compat of [null, false, [], { supportsReasoningEffort: false }, { supportsReasoningEffort: null }, { thinkingFormat: "openai" }]) {
    assert.throws(() => withThinkingCapabilities(settings(["glm-5.2"], { compat }), "local/glm-5.2"), /compatibility conflict/);
  }
  const explicitModelNull = settings(["glm-5.2"], { compat: { supportsReasoningEffort: true } });
  provider(explicitModelNull).models[0].compat = { supportsReasoningEffort: null };
  assert.throws(() => withThinkingCapabilities(explicitModelNull, "local/glm-5.2"), /compatibility conflict/);
  assert.throws(() => withThinkingCapabilities(settings(["unknown"]), "local/unknown"), /supported GLM/);
  assert.throws(() => withThinkingCapabilities(settings(["glm-5.3"], { modelOverrides: { "glm-5.3": {} } }), "local/glm-5.3"), /modelOverrides conflict/);
  assert.throws(() => withThinkingCapabilitiesForRoutes(settings([]), []), /at least one route/);
  assert.throws(() => withThinkingCapabilities(settings(["glm-5.3"]), "local/glm-5.3/extra"), /provider\/model/);
});
