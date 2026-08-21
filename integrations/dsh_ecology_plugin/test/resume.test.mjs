import assert from "node:assert/strict";
import test from "node:test";

import { RoleAgentManager } from "../lib/runtime/agents.js";

function binding(overrides = {}) {
  return {
    run_id: "run-1",
    role: "researcher",
    preset_id: "ecology-researcher-v1",
    session_id: "session-1",
    model: "dsh/strategy",
    cwd: "/tmp",
    require_workflow: false,
    preset_content_digest: "a".repeat(64),
    standing_tool_surface_digest: "b".repeat(64),
    route_config_digest: "c".repeat(64),
    ...overrides,
  };
}

test("empty flushed role-host resumes only with its exact frozen identity", async () => {
  const value = binding();
  const meta = {
    agentPreset: value.preset_id,
    ecologyRunId: value.run_id,
    ecologyRole: value.role,
    ecologyPresetContentDigest: value.preset_content_digest,
    ecologyToolSurfaceDigest: value.standing_tool_surface_digest,
    ecologyRouteConfigDigest: value.route_config_digest,
  };
  const calls = [];
  const agent = { id: value.session_id, session: { flush: async () => {} } };
  const manager = new RoleAgentManager({
    sessionPersistence: { inspect: async (id) => ({ header: { id, meta } }) },
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
  let resumed = false;
  const value = binding();
  const manager = new RoleAgentManager({
    sessionPersistence: {
      inspect: async () => ({
        header: {
          id: value.session_id,
          meta: {
            agentPreset: value.preset_id,
            ecologyRunId: value.run_id,
            ecologyRole: value.role,
            ecologyPresetContentDigest: "f".repeat(64),
            ecologyToolSurfaceDigest: value.standing_tool_surface_digest,
            ecologyRouteConfigDigest: value.route_config_digest,
          },
        },
      }),
    },
    agents: { resume: async () => { resumed = true; } },
  });
  await assert.rejects(manager.resumeRoleAgent(value), /PresetContentDigest drifted/);
  assert.equal(resumed, false);
});
