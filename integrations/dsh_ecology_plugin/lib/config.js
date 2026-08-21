import { validateLoopbackOrigin } from "./security.js";

export const DEFAULT_BACKEND_ORIGIN = "http://127.0.0.1:8777";

function optionalToken(value, name) {
  if (value == null || value === "") return null;
  if (typeof value !== "string" || !value.trim() || value.length > 8192) {
    throw new Error(`${name} must be bounded non-empty text`);
  }
  return value.trim();
}

function positiveInteger(value, fallback, name) {
  const result = value == null ? fallback : value;
  if (!Number.isSafeInteger(result) || result < 1) throw new Error(`${name} must be positive`);
  return result;
}

export function resolvePluginConfig(config = {}, { defaultStaticRoot, env = process.env } = {}) {
  return Object.freeze({
    staticRoot: config.staticRoot || defaultStaticRoot,
    backendOrigin: validateLoopbackOrigin(
      config.backendOrigin || DEFAULT_BACKEND_ORIGIN,
      DEFAULT_BACKEND_ORIGIN,
    ),
    serviceToken: optionalToken(
      config.serviceToken || config.service_token || env.ECOLOGYRSI_SERVICE_TOKEN,
      "serviceToken",
    ),
    runtimeToken: optionalToken(
      config.runtimeToken || env.ECOLOGYRSI_DSH_RUNTIME_TOKEN,
      "runtimeToken",
    ),
    sidecarToolToken: optionalToken(
      config.sidecarToolToken || env.ECOLOGYRSI_SIDECAR_TOOL_TOKEN,
      "sidecarToolToken",
    ),
    maxBodyBytes: positiveInteger(config.maxBodyBytes, 1024 * 1024, "maxBodyBytes"),
    maxResponseBytes: positiveInteger(
      config.maxResponseBytes, 16 * 1024 * 1024, "maxResponseBytes",
    ),
    totalTimeoutMs: positiveInteger(config.totalTimeoutMs, 30_000, "totalTimeoutMs"),
    structuredStageTimeoutMs: positiveInteger(
      config.structuredStageTimeoutMs,
      600_000,
      "structuredStageTimeoutMs",
    ),
  });
}
