import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { BLOCKED_MODEL_IDENTITY_FIELDS, TOOL_DEFINITIONS } from "../lib/tools/definitions.js";
import { ROLE_TOOL_NAMES, registerRoleTools } from "../lib/tools/roles.js";

test("role tool sets are exact and one-shot roles have no submit channel", () => {
  assert.deepEqual(ROLE_TOOL_NAMES.researcher, ["ecology_get_run_context", "ecology_get_research_evidence"]);
  assert.deepEqual(ROLE_TOOL_NAMES["generation-judge"], ["ecology_get_generation_summary"]);
  for (const role of ["researcher", "candidate-proposer", "generation-judge"]) {
    assert.equal(ROLE_TOOL_NAMES[role].some((name) => name.includes("submit")), false);
  }
  for (const definition of Object.values(TOOL_DEFINITIONS)) {
    const input = JSON.stringify(definition.parameters);
    for (const field of BLOCKED_MODEL_IDENTITY_FIELDS) assert.doesNotMatch(input, new RegExp(`"${field}"`));
    assert.equal(definition.parameters.additionalProperties, false);
  }
});

test("every shared JSON schema is closed at its public object boundary", async () => {
  for (const name of [
    "stage-context", "research-result", "genome-mutation", "sample-wave",
    "sample-decisions", "sample-review", "generation-summary", "generation-review",
  ]) {
    const schema = JSON.parse(await readFile(new URL(`../schemas/${name}.schema.json`, import.meta.url), "utf8"));
    assert.equal(schema.additionalProperties, false);
    assert.match(schema.$id, /^ecology-/);
  }
});

test("planner submit concludes only after durable sidecar acceptance", async () => {
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
  let concluded = 0;
  await handlers.get("ecology_submit_sample_decisions")(
    { schema_version: "ecology-sample-decisions@1", wave_digest: "a".repeat(64), decisions: [] },
    { agent: { id: "child" }, concludeTurn: () => { concluded += 1; } },
  );
  assert.equal(concluded, 1);
  assert.equal(calls[0].identity.role, "sample-planner");
});
