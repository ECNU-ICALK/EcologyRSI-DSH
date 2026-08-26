import { validateLoopbackOrigin } from "./security.js";
import { validateStructuredTimeoutMs } from "./runtime/structured-deadline.js";

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

function nonNegativeInteger(value, fallback, name) {
  const result = value == null ? fallback : value;
  if (!Number.isSafeInteger(result) || result < 0) {
    throw new Error(`${name} must be a non-negative integer`);
  }
  return result;
}

function boundedConcurrency(value, fallback, name) {
  const result = value == null ? fallback : value;
  if (!Number.isSafeInteger(result) || result < 1 || result > 8) {
    throw new Error(`${name} must be between 1 and 8`);
  }
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
    structuredStageTimeoutMs: validateStructuredTimeoutMs(
      positiveInteger(config.structuredStageTimeoutMs, 600_000, "structuredStageTimeoutMs"),
      "structuredStageTimeoutMs",
    ),
    researchStageTimeoutMs: validateStructuredTimeoutMs(
      positiveInteger(config.researchStageTimeoutMs, 1_800_000, "researchStageTimeoutMs"),
      "researchStageTimeoutMs",
    ),
    sampleCriticStageTimeoutMs: validateStructuredTimeoutMs(
      positiveInteger(config.sampleCriticStageTimeoutMs, 600_000, "sampleCriticStageTimeoutMs"),
      "sampleCriticStageTimeoutMs",
    ),
    structuredStageMinIntervalMs: nonNegativeInteger(
      config.structuredStageMinIntervalMs,
      0,
      "structuredStageMinIntervalMs",
    ),
    structuredStageMaxInFlight: boundedConcurrency(
      config.structuredStageMaxInFlight,
      8,
      "structuredStageMaxInFlight",
    ),
    structuredStageFailureCooldownMs: positiveInteger(
      config.structuredStageFailureCooldownMs,
      60_000,
      "structuredStageFailureCooldownMs",
    ),
    structuredStageMaxAttempts: positiveInteger(
      config.structuredStageMaxAttempts,
      2,
      "structuredStageMaxAttempts",
    ),
  });
}
