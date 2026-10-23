const REPLAY_PATH = "/api/ecology-agent-sidecar/v1/retrievals/replay";
const COMPLETE_PATH = "/api/ecology-agent-sidecar/v1/retrievals/complete";
export const RETRIEVAL_KEY_CONTRACT = "Use a retrieval_key of 1-120 ASCII characters: start with a letter or digit; remaining characters may be letters, digits, period, underscore, colon or hyphen. Spaces, slashes and Unicode are not allowed. Example: origin-review-co2-1";
export const RETRIEVAL_KEY_ERROR = `web_search retrieval_key is invalid; ${RETRIEVAL_KEY_CONTRACT}. No search was executed. Correct the key and retry web_search before structured_output`;
export const RETRIEVAL_POLICY_DEFAULTS = Object.freeze({
  dynamic_max_calls_per_stage: 8,
  max_queries_per_call: 8,
  max_query_chars: 320,
  max_results: 16,
});
export const RETRIEVAL_POLICY_LIMITS = Object.freeze({
  dynamic_max_calls_per_stage: 24,
  max_queries_per_call: 12,
  max_query_chars: 600,
  max_results: 32,
});
const MAX_CONTENT_CHARS = 4_000;
const MAX_URL_CHARS = 2_048;
const MAX_TITLE_CHARS = 300;
const MAX_SNIPPET_CHARS = 1_200;
const MAX_PUBLISHED_AT_CHARS = 100;
const FALLBACK_COMPLETION_TIMEOUT_MS = 420_000;

const UNAVAILABLE_CODES = new Set([
  "WEB_PROVIDER_CONFIGURED_MISSING",
  "WEB_PROVIDER_CONFIGURED_UNAVAILABLE",
  "WEB_PROVIDER_UNAVAILABLE",
  "WEB_PROVIDER_AMBIGUOUS",
  "WEB_PROVIDER_CREDENTIAL_MISSING",
]);

function boundedText(value, limit) {
  if (typeof value !== "string") return undefined;
  const text = value.trim().replaceAll(/\s+/g, " ");
  return text ? text.slice(0, limit) : undefined;
}

export function retrievalPolicy(value, { required = false } = {}) {
  if (value === undefined && !required) return { online_enabled: true, ...RETRIEVAL_POLICY_DEFAULTS };
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Host retrieval policy is missing or invalid");
  }
  if (required && (typeof value.online_enabled !== "boolean"
    || Object.keys(RETRIEVAL_POLICY_LIMITS).some((name) => !Object.hasOwn(value, name)))) {
    throw new Error("Host retrieval policy is incomplete");
  }
  const policy = { online_enabled: value.online_enabled ?? true };
  if (typeof policy.online_enabled !== "boolean") throw new Error("Host retrieval online policy is invalid");
  for (const [name, maximum] of Object.entries(RETRIEVAL_POLICY_LIMITS)) {
    const configured = value[name] ?? RETRIEVAL_POLICY_DEFAULTS[name];
    if (!Number.isSafeInteger(configured) || configured < 1 || configured > maximum) {
      throw new Error(`Host retrieval policy ${name} is invalid`);
    }
    policy[name] = configured;
  }
  return policy;
}

export function formalRetrievalPolicy(value, { required = false } = {}) {
  if (value === undefined && !required) return null;
  if (!value || typeof value !== "object" || Array.isArray(value)
    || Object.keys(value).sort().join("\0") !== ["document_count", "mode", "online_enabled", "snapshot_digest"].join("\0")
    || value.mode !== "frozen_snapshot" || value.online_enabled !== false
    || typeof value.snapshot_digest !== "string" || !/^[a-f0-9]{64}$/.test(value.snapshot_digest)
    || !Number.isSafeInteger(value.document_count) || value.document_count < 0) {
    throw new Error("Host formal retrieval requires a valid frozen snapshot with network access disabled");
  }
  return Object.freeze({ ...value });
}

export function comparisonRetrievalPolicy(context, stage, { formalScope = false } = {}) {
  const value = context.comparison_retrieval;
  if (value === undefined) return null;
  if (formalScope || context.formal_retrieval !== undefined
    || !["sample.plan", "sample.critic", "sample.reflect", "generation.judge"].includes(stage)
    || context.comparison_retrieval_policy !== "paired_method_snapshot@1"
    || context.comparison_retrieval_scope !== "methods_only_no_dataset_results") {
    throw new Error("comparison retrieval policy cannot grant formal or research admission");
  }
  return formalRetrievalPolicy(value, { required: true });
}

export function normalizeRetrievalArguments(value, limits = RETRIEVAL_POLICY_LIMITS) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("web_search arguments must be an object");
  }
  if (Object.keys(value).sort().join("\0") !== ["queries", "retrieval_key"].sort().join("\0")) {
    throw new Error("web_search arguments have an invalid shape");
  }
  if (!Array.isArray(value.queries) || value.queries.length < 1 || value.queries.length > limits.max_queries_per_call) {
    throw new Error(`web_search requires one to ${limits.max_queries_per_call} queries`);
  }
  const seen = new Set();
  const queries = [];
  for (const raw of value.queries) {
    const query = typeof raw === "string" ? raw.trim().replaceAll(/\s+/g, " ") : "";
    if (!query || [...query].length > limits.max_query_chars) throw new Error("web_search query is empty or too long");
    const key = query.toLocaleLowerCase("und");
    if (!seen.has(key)) {
      seen.add(key);
      queries.push(query);
    }
  }
  if (
    typeof value.retrieval_key !== "string"
    || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,119}$/.test(value.retrieval_key)
  ) {
    throw new Error(RETRIEVAL_KEY_ERROR);
  }
  return { queries, retrieval_key: value.retrieval_key };
}

function projectSource(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const url = boundedText(value.url, MAX_URL_CHARS);
  if (!url) return null;
  const source = { url };
  for (const [name, limit] of [
    ["title", MAX_TITLE_CHARS],
    ["snippet", MAX_SNIPPET_CHARS],
    ["publishedAt", MAX_PUBLISHED_AT_CHARS],
  ]) {
    const text = boundedText(value[name], limit);
    if (text) source[name] = text;
  }
  return source;
}

function projectSearchResult(value, maxResults) {
  if (!value || typeof value !== "object" || !Array.isArray(value.sources)) {
    throw Object.assign(new Error("malformed DSH search result"), {
      safeRetrievalCode: "primary_malformed",
    });
  }
  const sources = value.sources.slice(0, maxResults).map(projectSource).filter(Boolean);
  const result = {
    sources,
    truncated: Boolean(value.truncated) || value.sources.length > maxResults,
  };
  const content = boundedText(value.content, MAX_CONTENT_CHARS);
  if (content) result.content = content;
  return result;
}

function mergePrimaryResults(results, maxResults) {
  const sources = [];
  const content = [];
  let truncated = false;
  const projectedResults = results.map((value) => projectSearchResult(value, maxResults));
  for (const projected of projectedResults) {
    truncated ||= projected.truncated;
    if (projected.content) content.push(projected.content);
  }
  // Interleave queries so a broad first query cannot consume every result slot.
  const seenUrls = new Set();
  for (let index = 0; index < maxResults; index += 1) {
    for (const projected of projectedResults) {
      const source = projected.sources[index];
      if (!source || seenUrls.has(source.url)) continue;
      seenUrls.add(source.url);
      if (sources.length >= maxResults) {
        truncated = true;
        continue;
      }
      sources.push(source);
    }
  }
  const merged = { sources, truncated };
  const summary = boundedText(content.join("\n\n"), MAX_CONTENT_CHARS);
  if (summary) merged.content = summary;
  return merged;
}

function explicitAbort(error, signal) {
  return signal?.aborted === true || error?.code === "WEB_ABORTED" || error?.name === "AbortError";
}

export function primaryFailureCode(error) {
  if (error?.safeRetrievalCode === "primary_malformed") return "primary_malformed";
  const code = String(error?.code || "").toUpperCase();
  if (UNAVAILABLE_CODES.has(code)) return "primary_provider_unavailable";
  const causeName = String(error?.cause?.name || "").toUpperCase();
  if (code.includes("TIMEOUT") || causeName.includes("TIMEOUT")) return "primary_timeout";
  return "primary_provider_error";
}

export async function executeDynamicRetrieval({ ctx, bridge, role, args, exec }) {
  if (!bridge?.bindingFor || !bridge?.sidecar?.request) {
    throw new Error("ecology retrieval bridge is unavailable");
  }
  const argumentsValue = normalizeRetrievalArguments(args);
  const binding = await bridge.bindingFor(exec, { role, toolName: "web_search" });
  if (!binding || binding.role !== role) throw new Error("role tool authorization failed");
  // This policy comes from the authenticated stage context and private child
  // reservation, never model arguments. Keep it out of the closed Host identity.
  const { formal_retrieval: rawFormalPolicy, ...identity } = binding;
  const formalPolicy = formalRetrievalPolicy(rawFormalPolicy);
  const request = { identity, arguments: argumentsValue };
  const replay = await bridge.sidecar.request(REPLAY_PATH, {
    body: structuredClone(request),
    signal: exec?.signal,
  });
  if (formalPolicy) {
    if (replay?.found !== true || replay.result?.provider_route !== "frozen_snapshot"
      || replay.result.snapshot_digest !== formalPolicy.snapshot_digest
      || !Array.isArray(replay.result.sources)) {
      throw new Error("formal retrieval requires a matching Host frozen snapshot result; network fallback is forbidden");
    }
    return replay.result;
  }
  if (replay?.found === true) {
    if (!replay.result || typeof replay.result !== "object") {
      throw new Error("dynamic retrieval replay result is invalid");
    }
    return replay.result;
  }

  const policy = retrievalPolicy(replay?.policy, { required: true });
  if (!policy.online_enabled) throw new Error("Host retrieval policy disables online search");
  normalizeRetrievalArguments(argumentsValue, policy);
  if (exec?.signal?.aborted) throw Object.assign(new Error("retrieval aborted"), { name: "AbortError" });
  const primaryAvailable = typeof ctx?.web?.search === "function";
  const settled = primaryAvailable ? await Promise.allSettled(
    argumentsValue.queries.map((query) => Promise.resolve().then(() => ctx.web.search(
      { query, maxResults: policy.max_results },
      exec?.signal,
    ))),
  ) : [];
  if (exec?.signal?.aborted) throw Object.assign(new Error("retrieval aborted"), { name: "AbortError" });
  const rejected = settled.filter((item) => item.status === "rejected");
  const aborted = rejected.find((item) => explicitAbort(item.reason, exec?.signal));
  if (aborted) throw aborted.reason;

  const successes = settled
    .filter((item) => item.status === "fulfilled")
    .map((item) => item.value);
  let primaryResult;
  let primaryErrorCode = primaryAvailable ? undefined : "primary_provider_unavailable";
  try {
    primaryResult = mergePrimaryResults(successes, policy.max_results);
  } catch (error) {
    primaryErrorCode = primaryFailureCode(error);
  }
  if (rejected.length > 0) primaryErrorCode ||= primaryFailureCode(rejected[0].reason);
  const completion = { ...request };
  if (primaryResult) completion.primary_result = primaryResult;
  if (primaryErrorCode) completion.primary_error_code = primaryErrorCode;
  if (!primaryResult && !primaryErrorCode) completion.primary_result = { sources: [], truncated: false };
  return bridge.sidecar.request(COMPLETE_PATH, {
    body: structuredClone(completion),
    signal: exec?.signal,
    timeoutMs: FALLBACK_COMPLETION_TIMEOUT_MS,
  });
}
