import assert from "node:assert/strict";
import test from "node:test";

import { RoleAgentManager, roleSessionId } from "../lib/runtime/agents.js";

function binding(overrides = {}) {
  const value = {
    run_id: "run-1",
    role: "researcher",
    preset_id: "ecology-researcher-v12",
    model: "dsh/strategy",
    cwd: "/tmp",
    preset_content_digest: "a".repeat(64),
    standing_tool_surface_digest: "b".repeat(64),
    route_config_digest: "c".repeat(64),
    ...overrides,
  };
  // The persisted session id is minted from the binding, so a test that wants a
  // resumable host must derive it the same way #create does.
  return { ...value, session_id: overrides.session_id || roleSessionId(value) };
}

// SessionPersistence.stat returns detached header metadata; `id` and
// `agentPreset` are the only ecology-relevant fields DSH 0.1.5 still keeps.
function stored(value, overrides = {}) {
  return {
    header: {
      version: 3,
      id: value.session_id,
      createdAt: 1757548800000,
      cwd: value.cwd,
      isSeeded: false,
      agentPreset: value.preset_id,
      ...overrides,
    },
    revision: { toString: () => "rev-1" },
  };
}

test("empty flushed role-host resumes only with its exact frozen identity", async () => {
  const value = binding();
  const calls = [];
  const agent = { id: value.session_id, session: { flush: async () => {} } };
  const manager = new RoleAgentManager({
    sessionPersistence: { stat: async (id) => stored({ ...value, session_id: id }) },
    agents: { resume: async (options) => { calls.push(options); return { agent, dispose: async () => {} }; } },
    agentPresets: {
      standingKeyFor: async (id) => `standing:${id}`,
      mount: async (_agentCtx, id) => ({ id }),
      serviceFor: async () => ({ ready: true }),
    },
  });
  const resumed = await manager.resumeRoleAgent(value);
  assert.equal(resumed.sessionId, value.session_id);
  assert.equal(calls[0].resumeSessionId, value.session_id);
  assert.deepEqual(calls[0].agentOptions, { provider: "dsh", model: "strategy" });
  await calls[0].setup({});
});

test("resume rejects preset content, tool surface, or route drift before Agent creation", async () => {
  // Every one of these differs from the binding only in a fingerprinted field, so
  // the session it names is unreachable: the drift is caught by the id itself,
  // before any Agent is created.
  for (const drift of [
    { preset_content_digest: "f".repeat(64) },
    { standing_tool_surface_digest: "f".repeat(64) },
    { route_config_digest: "f".repeat(64) },
    { run_id: "run-2" },
    { role: "candidate-proposer" },
  ]) {
    let resumed = false;
    const persisted = binding();
    const value = { ...binding(drift), session_id: persisted.session_id };
    const manager = new RoleAgentManager({
      sessionPersistence: { stat: async () => stored(persisted) },
      agents: { resume: async () => { resumed = true; } },
    });
    await assert.rejects(
      manager.resumeRoleAgent(value),
      /role-host provenance drifted: session id carries [0-9a-f]{16}, binding fingerprints to [0-9a-f]{16}/,
    );
    assert.equal(resumed, false);
  }
});

test("resume rejects a header whose id or preset does not match", async () => {
  const value = binding();
  const cases = [
    [{ id: "session-elsewhere" }, /role-host header is invalid/],
    [{ agentPreset: "ecology-coordinator-v5" }, /role-host agentPreset drifted/],
  ];
  for (const [overrides, expected] of cases) {
    let resumed = false;
    const manager = new RoleAgentManager({
      sessionPersistence: { stat: async () => stored(value, overrides) },
      agents: { resume: async () => { resumed = true; } },
    });
    await assert.rejects(manager.resumeRoleAgent(value), expected);
    assert.equal(resumed, false);
  }
});

test("a pre-provenance session id is refused as a cut-over, not as digest drift", async () => {
  const value = binding({ session_id: "ecology-role-4f1c9a02-1b2d-4c3e-8f90-abcdef012345" });
  let resumed = false;
  const manager = new RoleAgentManager({
    sessionPersistence: { stat: async () => stored(value) },
    agents: { resume: async () => { resumed = true; } },
  });
  await assert.rejects(
    manager.resumeRoleAgent(value),
    /predates ecologyrsi-dsh\.role-session-provenance\/1 and cannot be resumed/,
  );
  assert.equal(resumed, false);
});
