#!/usr/bin/env node
// Explicit read-only regression against user-supplied local native Session
// paths. No assistant message/reasoning body is inspected or emitted.
// node .../display_projection_replay.mjs manifest.json evidence.json
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { zstdDecompressSync } from "node:zlib";
import {
  displaySourceDigest, projectSampleContext, expandSampleContext,
  projectPredictionToolResult, expandPredictionToolResult,
} from "../lib/runtime/display-projection.js";

const manifest = JSON.parse(await readFile(process.argv[2], "utf8"));
assert.ok(Array.isArray(manifest.sessions) && manifest.sessions.length > 0);
const bytes = value => Buffer.byteLength(JSON.stringify(value));
const totals = () => ({ count: 0, compressed: 0, source_bytes: 0, display_bytes: 0, saved_bytes: 0 });
const report = { schema_version: "ecologyrsi-dsh.display-projection-replay/1", read_only: true,
  remote_model_calls: 0, private_reasoning_inspected: false, contexts: totals(), tool_results: totals(),
  sessions: [], all_canonical_roundtrips_equal: true, all_authoritative_sources_unchanged: true };
function verify(value, project, expand, total) {
  const before = JSON.stringify(value), sourceDigest = displaySourceDigest(value);
  const view = project(value), full = expand(view);
  assert.deepEqual(full, value);
  assert.equal(displaySourceDigest(full), sourceDigest);
  assert.equal(JSON.stringify(value), before);
  const sourceBytes = bytes(value), displayBytes = bytes(view);
  assert.ok(displayBytes <= sourceBytes);
  total.count++; total.compressed += Number(Object.hasOwn(view, "_display_projection"));
  total.source_bytes += sourceBytes; total.display_bytes += displayBytes;
  total.saved_bytes += sourceBytes - displayBytes;
  return { source_digest: sourceDigest, source_bytes: sourceBytes, display_bytes: displayBytes,
    compressed: Object.hasOwn(view, "_display_projection"), roundtrip_equal: true };
}
for (const { session_id, file } of manifest.sessions) {
  const input = await readFile(file); let offset = 0, jsonl = "";
  while (offset < input.length) {
    const result = zstdDecompressSync(input.subarray(offset), { info: true });
    assert.ok(result.engine.bytesWritten > 0);
    offset += result.engine.bytesWritten; jsonl += result.buffer.toString("utf8");
  }
  const entry = { session_id, contexts: [], tool_results: [] };
  for (const line of jsonl.split("\n")) {
    // Filter before parsing: assistant message/stream contents never enter the
    // object inspection path. Only Host prompts and tool-result blocks do.
    if (!/"type"\s*:\s*"(?:user\/message|tool\/result)"/.test(line)) continue;
    const event = JSON.parse(line);
    if (!["user/message", "tool/result"].includes(event.type)) continue;
    const content = event.type === "user/message" ? event.data?.content : event.data?.message?.content;
    for (const block of content || []) {
      if (block.type !== "text") continue;
      let value; try { value = JSON.parse(block.text); } catch { continue; }
      if (event.type === "user/message" && value?.stage === "sample.plan" && value.context) {
        entry.contexts.push(verify(value.context, projectSampleContext, expandSampleContext, report.contexts));
      } else if (event.type === "tool/result" && value?.tool_id && value?.call_id && value?.wave_digest) {
        const result = verify(value, projectPredictionToolResult, expandPredictionToolResult, report.tool_results);
        result.status = value.status; result.output_digest = value.output_digest;
        assert.equal(projectPredictionToolResult(value).output_digest, value.output_digest);
        entry.tool_results.push(result);
      }
    }
  }
  assert.equal(entry.contexts.length, 1, "each selected origin has exactly one initial Host context");
  report.sessions.push(entry);
}
assert.equal(report.contexts.count, manifest.sessions.length);
assert.ok(report.contexts.compressed > 0 && report.tool_results.compressed > 0);
report.passed = true;
await writeFile(process.argv[3], JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify({ passed: true, contexts: report.contexts, tool_results: report.tool_results,
  remote_model_calls: 0 }));
