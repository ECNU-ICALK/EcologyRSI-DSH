import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { BLOCKED_MODEL_IDENTITY_FIELDS, TOOL_DEFINITIONS } from "../lib/tools/definitions.js";
import {
  ROLE_PLUGIN_TOOL_NAMES,
  ROLE_TOOL_NAMES,
  registerRoleToolGuard,
  registerRoleTools,
  rolePluginToolNames,
  roleToolNames,
} from "../lib/tools/roles.js";

test("role tool sets are exact and one-shot roles have no submit channel", () => {
  assert.deepEqual(
    ROLE_TOOL_NAMES["sample-planner"],
    ["skill", "web_search", "ecology_execute_prediction_tool"],
  );
  assert.deepEqual(
    ROLE_PLUGIN_TOOL_NAMES["sample-planner"],
    ["web_search", "ecology_execute_prediction_tool"],
  );
  for (const role of ["coordinator", "researcher", "candidate-proposer", "sample-critic", "generation-judge"]) {
    assert.deepEqual(ROLE_TOOL_NAMES[role], ["skill", "web_search"]);
    assert.deepEqual(ROLE_PLUGIN_TOOL_NAMES[role], ["web_search"]);
  }
  assert.deepEqual(roleToolNames("researcher"), ["skill"]);
  assert.deepEqual(rolePluginToolNames("researcher"), []);
  assert.deepEqual(
    roleToolNames("researcher", "dynamic-retrieval-v1"),
    ["skill", "web_search"],
  );
  assert.throws(
    () => roleToolNames("researcher", "model-selected-provider"),
    /unsupported.*profile/i,
  );
  for (const definition of Object.values(TOOL_DEFINITIONS)) {
    const input = JSON.stringify(definition.parameters);
    for (const field of BLOCKED_MODEL_IDENTITY_FIELDS) assert.doesNotMatch(input, new RegExp(`"${field}"`));
    assert.equal(definition.parameters.additionalProperties, false);
  }
});

test("every shared JSON schema is closed at its public object boundary", async () => {
  for (const name of [
    "stage-context", "research-result", "research-search-plan", "research-synthesis",
    "genome-mutation", "sample-wave", "sample-decisions", "sample-review",
    "generation-summary", "generation-review", "generation-reflection",
  ]) {
    const schema = JSON.parse(await readFile(new URL(`../schemas/${name}.schema.json`, import.meta.url), "utf8"));
    assert.equal(schema.additionalProperties, false);
    assert.match(schema.$id, /^ecology-/);
  }
});

test("planner vector tool is forwarded once through its Host-bound sidecar identity", async () => {
  const handlers = new Map();
  const calls = [];
  const ctx = { tools: { register: (definition) => { handlers.set(definition.name, definition.execute); return () => {}; } } };
  registerRoleTools(ctx, {
    role: "sample-planner",
    bridge: {
      bindingFor: async () => ({ role: "sample-planner", run_id: "r1" }),
      sidecar: { request: async (_path, options) => { calls.push(options.body); return { accepted: true }; } },
    },
  });
  await handlers.get("ecology_execute_prediction_tool")(
    { tool_id: "ridge@1", wave_digest: "a".repeat(64) },
    { agent: { id: "child" } },
  );
  assert.equal(calls[0].identity.role, "sample-planner");
  assert.deepEqual(calls[0].arguments, {
    tool_id: "ridge@1",
    wave_digest: "a".repeat(64),
  });
});

test("role guard allows only the frozen role tool and structured result channel", () => {
  let guard;
  registerRoleToolGuard({
    tools: {
      guard: (candidate) => { guard = candidate; return () => {}; },
    },
  }, "sample-planner", { toolProfile: "dynamic-retrieval-v1" });
  assert.equal(guard({ name: "ecology_execute_prediction_tool" }), undefined);
  assert.equal(guard({ name: "web_search" }), undefined);
  assert.equal(guard({ name: "skill" }), undefined);
  assert.equal(guard({ name: "structured_output" }), undefined);
  assert.match(guard({ name: "bash" }), /outside the frozen sample-planner role surface/);
});
