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
  registerRoleTools({ tools: { register: (definition) => { handler ||= definition.execute; } } }, {
    role: "researcher",
    bridge: {
      bindingFor: async () => ({ role: "sample-planner" }),
      sidecar: { request: async () => { called = true; } },
    },
  });
  await assert.rejects(handler({}, { agent: { id: "child" } }), /authorization failed/);
  assert.equal(called, false);
});
