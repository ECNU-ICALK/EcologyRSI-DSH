#!/usr/bin/env node
// Creates a private, instance-specific settings copy. Never edits global DSH
// settings or credentials; launch the desired DSH process with the printed patch.
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

export function withThinkingCapabilities(settings, route) {
  const separator = route.indexOf("/");
  const providerId = route.slice(0, separator);
  const modelId = route.slice(separator + 1);
  const glm = /^glm-(?:5(?:\.1|\.2)?|4\.7)$/.test(modelId);
  const deepseek = /^deepseek-v4-(?:flash|pro)(?:-\d{4})?$/.test(modelId);
  if (separator < 1 || (!glm && !deepseek)) {
    throw new Error("route must name a supported GLM or DeepSeek V4 thinking-mode model");
  }
  const result = structuredClone(settings);
  const provider = result?.["llm-pi-ai"]?.providers?.[providerId];
  const model = provider?.models?.find((entry) => entry.id === modelId);
  if (provider?.api !== "openai-completions" || !model) {
    throw new Error("route must be an explicitly configured OpenAI-completions model");
  }
  // GLM's official API uses thinking.type, not reasoning_effort. A custom
  // endpoint cannot be identified reliably from its hostname by pi-ai.
  model.reasoningEfforts = { off: null, ...(deepseek ? { low: "low" } : {}), high: "high" };
  model.compat = {
    ...model.compat,
    thinkingFormat: glm ? "zai" : "deepseek",
    supportsReasoningEffort: deepseek,
    supportsDeveloperRole: false,
    supportsStore: false,
    maxTokensField: "max_tokens",
    // PJLab's V4 route can emit DSML prose when an OpenAI strict field is
    // present, even when false. Omit unsupported provider-side constraints;
    // native tool capture and Host JSON-schema validation still apply.
    supportsStrictMode: false,
    ...(deepseek ? { requiresReasoningContentOnAssistantMessages: true } : {}),
  };
  return result;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const { values } = parseArgs({ options: {
    "dsh-package": { type: "string" }, source: { type: "string" },
    output: { type: "string" }, route: { type: "string", multiple: true },
  } });
  if (Object.values(values).length !== 4) throw new Error("required: --dsh-package --source --output --route");
  const yaml = createRequire(path.resolve(values["dsh-package"]))("yaml");
  const settings = yaml.parse(await readFile(values.source, "utf8"));
  const configured = values.route.reduce(withThinkingCapabilities, settings);
  const output = path.resolve(values.output);
  await mkdir(output, { recursive: true, mode: 0o700 });
  const settingsPath = path.join(output, "settings.yaml");
  const patchPath = path.join(output, "settings.patch.yml");
  await writeFile(settingsPath, yaml.stringify(configured), { mode: 0o600, flag: "wx" });
  await writeFile(patchPath, yaml.stringify([{ id: "settings", config: { path: settingsPath } }]), {
    mode: 0o600, flag: "wx",
  });
  console.log(`Instance settings prepared. Use dsh --profile web --patch ${patchPath} before app flags such as --port.`);
}
