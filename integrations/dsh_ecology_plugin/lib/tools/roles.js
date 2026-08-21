import { assertModelArgumentsSafe, TOOL_DEFINITIONS } from "./definitions.js";

export const ROLE_TOOL_NAMES = Object.freeze({
  coordinator: ["ecology_get_run_context"],
  researcher: ["ecology_get_run_context", "ecology_get_research_evidence"],
  "candidate-proposer": ["ecology_get_run_context", "ecology_get_research_evidence", "ecology_get_generation_summary"],
  "sample-planner": ["ecology_get_run_context", "ecology_get_sample_wave", "ecology_execute_prediction_tool", "ecology_submit_sample_decisions"],
  "sample-critic": ["ecology_get_sample_wave", "ecology_get_prediction_summary", "ecology_submit_sample_review"],
  "generation-judge": ["ecology_get_generation_summary"],
});

export function registerRoleTools(ctx, config = {}) {
  const role = String(config.role || "");
  const names = ROLE_TOOL_NAMES[role];
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
      if (definition.concludesTurn && result?.accepted === true) exec?.concludeTurn?.();
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
