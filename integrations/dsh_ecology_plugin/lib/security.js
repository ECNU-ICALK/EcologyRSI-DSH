import { timingSafeEqual } from "node:crypto";

export function isLoopbackAddress(value) {
  const address = String(value || "").toLowerCase();
  return new Set([
    "127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost",
  ]).has(address);
}

export function bearerMatches(header, expected) {
  const value = typeof header === "string" ? header : "";
  const wanted = `Bearer ${expected}`;
  const left = Buffer.from(value);
  const right = Buffer.from(wanted);
  return left.length === right.length && timingSafeEqual(left, right);
}

export function safeJsonError(res, statusCode, code) {
  const body = Buffer.from(JSON.stringify({ error: code }));
  res.writeHead(statusCode, {
    "cache-control": "no-store",
    "content-length": String(body.length),
    "content-type": "application/json; charset=utf-8",
    "x-content-type-options": "nosniff",
  });
  res.end(body);
}

export async function readBoundedJson(req, maxBytes) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > maxBytes) throw new Error("body_too_large");
    chunks.push(Buffer.from(chunk));
  }
  if (size === 0) return {};
  let value;
  try {
    value = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new Error("invalid_json");
  }
  if (!value || Array.isArray(value) || typeof value !== "object") {
    throw new Error("invalid_json_object");
  }
  return value;
}

export function validateLoopbackOrigin(value, fallback) {
  const target = new URL(value || fallback);
  if (target.protocol !== "http:") {
    throw new Error("only supports a loopback HTTP backend");
  }
  if (!isLoopbackAddress(target.hostname)) {
    throw new Error("backend must use a loopback host");
  }
  if (target.username || target.password || target.pathname !== "/" || target.search || target.hash) {
    throw new Error("backend must be an origin without credentials or a path");
  }
  return target;
}
