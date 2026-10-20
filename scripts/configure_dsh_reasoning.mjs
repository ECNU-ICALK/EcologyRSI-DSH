#!/usr/bin/env node
// Creates a private, instance-specific settings copy. Never edits global DSH
// settings or credentials; launch the desired DSH process with the printed patch.
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

function thinkingContract(modelId) {
  // Z.AI API contract, checked 2026-10-20. Coding Plan aliases differ; these
  // declarations do not attest that a third-party gateway implements them.
  if (/^glm-5\.3(?:-flash)?$/.test(modelId)) return {
    efforts: { low: "low", high: "high", max: "max" },
    defaultEffort: "max", format: "zai", effortParameter: true,
  };
  if (modelId === "glm-5.2") return {
    efforts: { off: null, minimal: "minimal", low: "high", medium: "high",
      high: "high", xhigh: "max", max: "max" },
    defaultEffort: "max", format: "zai", effortParameter: true,
  };
  if (/^glm-(?:5(?:\.1)?|4\.7)$/.test(modelId)) return {
    efforts: { off: null, high: "high" },
    defaultEffort: "high", format: "zai", effortParameter: false,
  };
  if (/^deepseek-v4-(?:flash|pro)(?:-\d{4})?$/.test(modelId)) return {
    efforts: { off: null, low: "low", high: "high" },
    defaultEffort: "high", format: "deepseek", effortParameter: true,
  };
  throw new Error("route must name a supported GLM or DeepSeek V4 thinking-mode model");
}

function checkedEfforts(existing, contract, route) {
  if (existing === undefined) return { ...contract.efforts };
  if (!existing || typeof existing !== "object" || Array.isArray(existing)
    || !Object.keys(existing).some(key => key !== "off")
    || Object.entries(existing).some(([key, value]) =>
      !Object.hasOwn(contract.efforts, key) || value !== contract.efforts[key])) {
    throw new Error(`reasoning capability conflict for ${route}; reconcile the explicit mapping before generating settings`);
  }
  return existing;
}

function checkedCompat(value, route) {
  if (value === undefined) return {};
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`thinking compatibility conflict for ${route}; compat must be an object`);
  }
  return value;
}

export function withThinkingCapabilitiesForRoutes(settings, routes) {
  if (!Array.isArray(routes) || routes.length === 0) throw new Error("at least one route is required");
  const result = structuredClone(settings), selected = new Map();
  for (const route of new Set(routes)) {
    if (typeof route !== "string" || !/^[^/\s]+\/[^/\s]+$/.test(route)) throw new Error("route must be provider/model");
    const [providerId, modelId] = route.split("/");
    const contract = thinkingContract(modelId);
    const provider = result?.["llm-pi-ai"]?.providers?.[providerId];
    const matches = Array.isArray(provider?.models) ? provider.models.filter(entry => entry?.id === modelId) : [];
    if (provider?.api !== "openai-completions" || matches.length !== 1) {
      throw new Error("route must be one explicitly configured OpenAI-completions model");
    }
    if (Object.hasOwn(provider.modelOverrides || {}, modelId)) {
      throw new Error(`modelOverrides conflict for ${route}; reconcile the explicit override first`);
    }
    const model = matches[0];
    model.reasoningEfforts = checkedEfforts(model.reasoningEfforts, contract, route);
    const providerCompat = checkedCompat(provider.compat, route);
    const modelCompat = checkedCompat(model.compat, route);
    const required = { thinkingFormat: contract.format, supportsReasoningEffort: contract.effortParameter };
    for (const [key, value] of Object.entries(required)) {
      const existing = Object.hasOwn(modelCompat, key) ? modelCompat[key] : providerCompat[key];
      if (existing !== undefined && existing !== value) {
        throw new Error(`thinking compatibility conflict for ${route}: ${key}; explicit settings are not overwritten`);
      }
    }
    // Only absent compatibility values receive defaults. Existing endpoint,
    // authentication references, headers and unrelated model settings survive.
    model.compat = {
      supportsDeveloperRole: false, supportsStore: false,
      maxTokensField: "max_tokens", supportsStrictMode: false,
      ...(contract.format === "deepseek" ? { requiresReasoningContentOnAssistantMessages: true } : {}),
      ...providerCompat, ...modelCompat, ...required,
    };
    if (!selected.has(providerId)) selected.set(providerId, []);
    selected.get(providerId).push({ model, contract });
  }
  for (const [providerId, entries] of selected) {
    const provider = result["llm-pi-ai"].providers[providerId];
    if (provider.reasoning === undefined) {
      // Harness defaults are provider-wide, not model-local. Never silently
      // change unselected models while fixing the selected route's default.
      const ids = new Set(entries.map(entry => entry.model.id));
      if (provider.models.some(model => !ids.has(model.id))
        || Object.keys(provider.modelOverrides || {}).some(id => !ids.has(id))) {
        throw new Error(`provider ${providerId} has unselected models and no explicit reasoning default; use an isolated provider profile or select every model`);
      }
      const defaults = new Set(entries.map(entry => entry.contract.defaultEffort));
      if (defaults.size !== 1) throw new Error(`provider ${providerId} needs an explicit common reasoning default or separate provider profiles`);
      provider.reasoning = entries[0].contract.defaultEffort;
    }
    for (const { model } of entries) {
      if (!Object.hasOwn(model.reasoningEfforts, provider.reasoning)) {
        throw new Error(`reasoning default conflicts with ${providerId}/${model.id}; choose a declared effort explicitly`);
      }
    }
  }
  return result;
}

export function withThinkingCapabilities(settings, route) {
  return withThinkingCapabilitiesForRoutes(settings, [route]);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const { values } = parseArgs({ options: {
    "dsh-package": { type: "string" }, source: { type: "string" },
    output: { type: "string" }, route: { type: "string", multiple: true },
  } });
  if (Object.values(values).length !== 4) throw new Error("required: --dsh-package --source --output --route");
  const yaml = createRequire(path.resolve(values["dsh-package"]))("yaml");
  const settings = yaml.parse(await readFile(values.source, "utf8"));
  const configured = withThinkingCapabilitiesForRoutes(settings, values.route);
  const output = path.resolve(values.output);
  await mkdir(output, { recursive: true, mode: 0o700 });
  const settingsPath = path.join(output, "settings.yaml");
  const patchPath = path.join(output, "settings.patch.yml");
  await writeFile(settingsPath, yaml.stringify(configured), { mode: 0o600, flag: "wx" });
  // Harness 0.2 profiles consume plugin config overlays directly. The settings
  // service only migrates profile.home/settings.yaml; it has no config.path.
  // Keep the separate YAML snapshot for the Python model-directory reader.
  await writeFile(patchPath, yaml.stringify([{ id: "llm-pi-ai", config: configured["llm-pi-ai"] }]), {
    mode: 0o600, flag: "wx",
  });
  console.log(`Instance settings prepared. Use dsh --profile web --patch ${patchPath} before app flags such as --port.`);
}
