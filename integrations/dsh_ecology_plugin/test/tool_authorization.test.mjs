import assert from "node:assert/strict";
import test from "node:test";

import { assertModelArgumentsSafe } from "../lib/tools/definitions.js";
import { registerRoleTools } from "../lib/tools/roles.js";

test("Host identity and labels cannot be forged recursively in model arguments", () => {
  assert.throws(() => assertModelArgumentsSafe({ nested: { run_id: "other" } }), /Host identity/);
  assert.throws(() => assertModelArgumentsSafe({ rows: [{ ground_truth: 5 }] }, { labelFree: true }), /label-bearing/);
  assert.doesNotThrow(() => assertModelArgumentsSafe({ rows: [{ temperature: 21.5 }] }, { labelFree: true }));
});

test("wrong Host-bound role is denied before sidecar execution", async () => {
  let handler;
  let called = false;
  registerRoleTools({ tools: { register: (definition) => {
    if (definition.name === "ecology_execute_prediction_tool") handler = definition.execute;
  } } }, {
    role: "sample-planner",
    bridge: {
      bindingFor: async () => ({ role: "researcher" }),
      sidecar: { request: async () => { called = true; } },
    },
  });
  await assert.rejects(
    handler(
      { tool_id: "ridge@1", wave_digest: "a".repeat(64) },
      { agent: { id: "child" } },
    ),
    /authorization failed/,
  );
  assert.equal(called, false);
});
