import { validateLoopbackOrigin } from "../security.js";

const SIDECAR_BASE = "/api/ecology-agent-sidecar/v1";

export class SidecarError extends Error {
  constructor(code, message = code, { publicDetail = null } = {}) {
    super(message);
    this.name = "SidecarError";
    this.code = code;
    this.publicDetail = publicDetail;
  }
}

function redactedPublicDetail(value) {
  if (typeof value !== "string") return null;
  const normalized = value.replace(/[\u0000-\u001f\u007f]/g, " ").trim();
  return normalized ? normalized.slice(0, 256) : null;
}

export class SidecarClient {
  constructor({
    origin = "http://127.0.0.1:8777",
    token,
    totalTimeoutMs = 10_000,
    maxRequestBytes = 64 * 1024,
    maxResponseBytes = 512 * 1024,
  } = {}) {
    this.origin = validateLoopbackOrigin(origin, "http://127.0.0.1:8777");
    if (typeof token !== "string" || !token.trim()) throw new Error("sidecar token is required");
    this.token = token.trim();
    this.totalTimeoutMs = totalTimeoutMs;
    this.maxRequestBytes = maxRequestBytes;
    this.maxResponseBytes = maxResponseBytes;
  }

  async request(path, { method = "POST", body, signal } = {}) {
    if (typeof path !== "string" || !path.startsWith(`${SIDECAR_BASE}/`)
      || path.includes("?") || path.includes("#") || path.includes("..") || path.includes("://")) {
      throw new SidecarError("invalid_path", "sidecar path is not allowed");
    }
    if (!new Set(["GET", "POST"]).has(method)) {
      throw new SidecarError("invalid_method", "sidecar method is not allowed");
    }
    let encoded;
    if (body !== undefined) {
      encoded = Buffer.from(JSON.stringify(body));
      if (encoded.length > this.maxRequestBytes) {
        throw new SidecarError("request_too_large");
      }
    }
    const controller = new AbortController();
    let timedOut = false;
    const timeout = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, this.totalTimeoutMs);
    const onAbort = () => controller.abort();
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) controller.abort();
    try {
      const target = new URL(path, this.origin);
      const response = await fetch(target, {
        method,
        body: encoded,
        signal: controller.signal,
        headers: {
          authorization: `Bearer ${this.token}`,
          accept: "application/json",
          ...(encoded ? { "content-type": "application/json" } : {}),
        },
      });
      const declared = Number(response.headers.get("content-length"));
      if (Number.isFinite(declared) && declared > this.maxResponseBytes) {
        await response.body?.cancel();
        throw new SidecarError("response_too_large");
      }
      const chunks = [];
      let size = 0;
      if (response.body) {
        const reader = response.body.getReader();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          size += value.byteLength;
          if (size > this.maxResponseBytes) {
            await reader.cancel();
            throw new SidecarError("response_too_large");
          }
          chunks.push(Buffer.from(value));
        }
      }
      let result = {};
      try {
        const text = Buffer.concat(chunks).toString("utf8");
        result = text ? JSON.parse(text) : {};
      } catch {
        throw new SidecarError("invalid_response");
      }
      if (!response.ok) {
        throw new SidecarError("sidecar_rejected", "sidecar_rejected", {
          // The Python boundary has already passed this value through its
          // credential-redacting public-error policy.  Retaining the bounded
          // value here makes native-runtime failures diagnosable without
          // exposing the sidecar token or response headers.
          publicDetail: redactedPublicDetail(result?.error),
        });
      }
      return result;
    } catch (error) {
      if (error instanceof SidecarError) throw error;
      if (controller.signal.aborted) {
        throw new SidecarError(timedOut ? "timeout" : "aborted");
      }
      throw new SidecarError("unavailable");
    } finally {
      clearTimeout(timeout);
      signal?.removeEventListener("abort", onAbort);
    }
  }
}

export { SIDECAR_BASE };
