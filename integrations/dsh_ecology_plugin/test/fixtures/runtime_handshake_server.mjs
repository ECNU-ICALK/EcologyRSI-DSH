import { createServer } from "node:http";

import { RuntimeController } from "../../lib/runtime/controller.js";
import { registerRuntimeRoutes } from "../../lib/runtime/routes.js";

const presetCatalog = [
  "ecology-coordinator-v6",
  "ecology-researcher-v15",
  "ecology-candidate-proposer-v6",
  "ecology-sample-planner-v12",
  "ecology-sample-critic-v6",
  "ecology-generation-judge-v10",
].map((preset_id) => ({
  preset_id,
  tool_profile: "dynamic-retrieval-v1",
  required_tools: preset_id === "ecology-sample-planner-v12"
    ? ["ecology_execute_prediction_tool", "skill", "web_search"]
    : ["skill", "web_search"],
}));
const toolsByStandingKey = new Map(presetCatalog.map((item) => [
  item.preset_id,
  item.required_tools,
]));
let runtimeHandler;
const ctx = {
  webServer: {
    register: ({ handler }) => {
      runtimeHandler = handler;
      return () => {};
    },
  },
  agents: {
    create: async (options) => {
      const agent = {
        session: { append: async () => {}, flush: async () => {} },
        whenIdle: async () => {},
      };
      await options.setup?.(agent);
      return { agent, dispose: async () => {} };
    },
  },
  sessions: {},
  tokenMeter: {},
  subagents: {},
  tools: {
    schemas: async (standingKey) => (
      toolsByStandingKey.get(standingKey.presetId)?.map((name) => ({ name })) || []
    ),
  },
  sessionPersistence: {},
  sessionProjections: {},
  agentPresets: {
    resolve: async (presetId) => ({ id: presetId }),
      acquireScope: async (presetId) => ({ key: { presetId: presetId }, async [Symbol.asyncDispose]() {} }),
    mount: async (_agent, presetId) => ({ id: presetId }),
    serviceFor: async (_agent, serviceName) => ({ serviceName }),
  },
  llm: { resolveCallConfig: async () => ({ provider: "test", model: "model" }) },
  web: {},
};
const controller = new RuntimeController(ctx, {
  presetCatalog,
  stageRunner: {
    closeLaunchFence: () => {},
    openLaunchFence: () => {},
  },
});
registerRuntimeRoutes(ctx, controller, {
  runtimeToken: "runtime-secret",
  maxBodyBytes: 1024 * 1024,
});
const server = createServer((req, res) => { void runtimeHandler(req, res); });
server.listen(0, "127.0.0.1", () => {
  process.stdout.write(`${JSON.stringify(server.address())}\n`);
});
process.on("SIGTERM", () => server.close(() => process.exit(0)));
