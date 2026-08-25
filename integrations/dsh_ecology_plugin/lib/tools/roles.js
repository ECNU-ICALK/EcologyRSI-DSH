import { assertModelArgumentsSafe, TOOL_DEFINITIONS } from "./definitions.js";

export const ROLE_PLUGIN_TOOL_NAMES = Object.freeze({
  coordinator: [],
  researcher: [],
  "candidate-proposer": [],
  "sample-planner": ["ecology_execute_prediction_tool"],
  "sample-critic": [],
  "generation-judge": [],
});

export const ROLE_TOOL_NAMES = Object.freeze({
  coordinator: ["skill"],
  researcher: ["skill"],
  "candidate-proposer": ["skill"],
  "sample-planner": ["skill", "ecology_execute_prediction_tool"],
  "sample-critic": ["skill"],
  "generation-judge": ["skill"],
});

const STRUCTURED_OUTPUT_TOOL = "structured_output";

export function registerRoleToolGuard(ctx, role) {
  const names = ROLE_TOOL_NAMES[role];
  if (!names) throw new Error(`unknown ecology role: ${role}`);
  if (typeof ctx?.tools?.guard !== "function") {
    throw new Error("DSH tools.guard is required in the agent standing scope");
  }
  const allowed = new Set([...names, STRUCTURED_OUTPUT_TOOL]);
  return ctx.tools.guard((exec) => (
    allowed.has(exec?.name)
      ? undefined
      : `tool ${String(exec?.name || "<unknown>")} is outside the frozen ${role} role surface`
  ));
}

export function registerRoleTools(ctx, config = {}) {
  const role = String(config.role || "");
  const names = ROLE_PLUGIN_TOOL_NAMES[role];
  if (!names) throw new Error(`unknown ecology role: ${role}`);
  const register = ctx?.tools?.register || ctx?.tools?.define;
  if (typeof register !== "function") throw new Error("DSH tool registration service is required");
  const bridge = config.bridge || ctx.ecologyAgentTools;
  const disposers = [];
  for (const name of names) {
    const definition = TOOL_DEFINITIONS[name];
    const handler = async (args, exec) => {
      assertModelArgumentsSafe(args, { labelFree: role === "sample-planner" });
      if (!bridge?.bindingFor || !bridge?.sidecar?.request) throw new Error("ecology role tool bridge is unavailable");
      const binding = await bridge.bindingFor(exec, { role, toolName: name });
      if (!binding || binding.role !== role) throw new Error("role tool authorization failed");
      const result = await bridge.sidecar.request(`/api/ecology-agent-sidecar/v1/tools/${name}`, {
        body: { identity: binding, arguments: structuredClone(args) },
        signal: exec?.signal,
      });
      return result;
    };
    const disposer = register.call(ctx.tools, {
      ...definition,
      output: {
        schema: { type: "object", additionalProperties: true },
        render: (_args, value) => [{ type: "text", text: JSON.stringify(value) }],
      },
      execute: handler,
    });
    if (typeof disposer === "function") disposers.push(disposer);
  }
  return () => { for (const dispose of disposers.reverse()) dispose(); };
}
