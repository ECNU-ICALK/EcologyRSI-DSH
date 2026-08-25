const REPLAY_PATH = "/api/ecology-agent-sidecar/v1/retrievals/replay";
const COMPLETE_PATH = "/api/ecology-agent-sidecar/v1/retrievals/complete";
const MAX_RESULTS = 8;
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

export function normalizeRetrievalArguments(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("web_search arguments must be an object");
  }
  if (Object.keys(value).sort().join("\0") !== ["queries", "retrieval_key"].sort().join("\0")) {
    throw new Error("web_search arguments have an invalid shape");
  }
  if (!Array.isArray(value.queries) || value.queries.length < 1 || value.queries.length > 4) {
    throw new Error("web_search requires one to four queries");
  }
  const seen = new Set();
  const queries = [];
  for (const raw of value.queries) {
    const query = boundedText(raw, 181);
    if (!query || query.length > 180) throw new Error("web_search query is empty or too long");
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
    throw new Error("web_search retrieval_key is invalid");
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

function projectSearchResult(value) {
  if (!value || typeof value !== "object" || !Array.isArray(value.sources)) {
    throw Object.assign(new Error("malformed DSH search result"), {
      safeRetrievalCode: "primary_malformed",
    });
  }
  const sources = value.sources.slice(0, MAX_RESULTS).map(projectSource).filter(Boolean);
  const result = {
    sources,
    truncated: Boolean(value.truncated) || value.sources.length > MAX_RESULTS,
  };
  const content = boundedText(value.content, MAX_CONTENT_CHARS);
  if (content) result.content = content;
  return result;
}

function mergePrimaryResults(results) {
  const sources = [];
  const content = [];
  let truncated = false;
  for (const value of results) {
    const projected = projectSearchResult(value);
    truncated ||= projected.truncated;
    if (projected.content) content.push(projected.content);
    for (const source of projected.sources) {
      if (sources.length >= MAX_RESULTS) {
        truncated = true;
        break;
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
  if (typeof ctx?.web?.search !== "function") {
    throw new Error("DSH web search service is unavailable");
  }
  const argumentsValue = normalizeRetrievalArguments(args);
  const binding = await bridge.bindingFor(exec, { role, toolName: "web_search" });
  if (!binding || binding.role !== role) throw new Error("role tool authorization failed");
  const request = { identity: binding, arguments: argumentsValue };
  const replay = await bridge.sidecar.request(REPLAY_PATH, {
    body: structuredClone(request),
    signal: exec?.signal,
  });
  if (replay?.found === true) {
    if (!replay.result || typeof replay.result !== "object") {
      throw new Error("dynamic retrieval replay result is invalid");
    }
    return replay.result;
  }

  const settled = await Promise.allSettled(
    argumentsValue.queries.map((query) => ctx.web.search(
      { query, maxResults: MAX_RESULTS },
      exec?.signal,
    )),
  );
  const rejected = settled.filter((item) => item.status === "rejected");
  const aborted = rejected.find((item) => explicitAbort(item.reason, exec?.signal));
  if (aborted) throw aborted.reason;

  const successes = settled
    .filter((item) => item.status === "fulfilled")
    .map((item) => item.value);
  let primaryResult;
  let primaryErrorCode;
  try {
    primaryResult = mergePrimaryResults(successes);
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
