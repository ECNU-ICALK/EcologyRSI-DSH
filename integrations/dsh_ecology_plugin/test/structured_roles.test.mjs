import assert from "node:assert/strict";
import test from "node:test";

import { runStructuredRole } from "../lib/runtime/structured-roles.js";
import { PendingChildStarts } from "../lib/runtime/workflows.js";

test("one-shot structured role persists only structured output and disposes its run", async () => {
  let disposed = false;
  let request;
  const pendingStarts = new PendingChildStarts({
    subagents: {
      start: async (provider, value) => {
        assert.equal(provider, "spawn");
        request = value;
        return {
          id: "researcher-child-session",
          result: Promise.resolve({ structured: { schema_version: "ecology-research-result@1", summary: "ok", evidence: [] }, text: "ignored" }),
          dispose: async () => { disposed = true; },
        };
      },
    },
  });
  const persisted = [];
  const result = await runStructuredRole(
    { agent: { id: "researcher-host" } },
    { label: "safe-label", reservation_id: "r1" },
    { prompt: "research", outputSchema: { type: "object" } },
    {
      pendingStarts,
      admission: { isOpen: async () => true },
      persist: async (value) => { persisted.push(value); return { accepted: true, digest: "a".repeat(64) }; },
    },
  );
  assert.equal(request.label, "safe-label");
  assert.deepEqual(request.prompt, [{ type: "text", text: "research" }]);
  assert.deepEqual(result.structured, { schema_version: "ecology-research-result@1", summary: "ok", evidence: [] });
  assert.equal("text" in persisted[0], false);
  assert.equal(persisted[0].session_id, "researcher-child-session");
  assert.equal(result.session_id, "researcher-child-session");
  assert.equal(disposed, true);
  assert.equal(pendingStarts.size, 0);
});

test("closed admission between completion and persistence rejects late structured result", async () => {
  let persisted = false;
  let disposed = false;
  const pendingStarts = new PendingChildStarts({
    subagents: { start: async () => ({ structured: { value: 1 }, dispose: async () => { disposed = true; } }) },
  });
  await assert.rejects(
    runStructuredRole(
      { agent: { id: "judge-host" } },
      { label: "safe-label" },
      { prompt: "judge", outputSchema: { type: "object" } },
      {
        pendingStarts,
        admission: { isOpen: async () => false },
        persist: async () => { persisted = true; return { accepted: true }; },
      },
    ),
    /admission is closed/,
  );
  assert.equal(persisted, false);
  assert.equal(disposed, true);
});

test("structured role aborts a wedged DSH child at the operational timeout", async () => {
  let aborted = false;
  const pendingStarts = new PendingChildStarts({
    subagents: {
      start: async (_provider, request) => new Promise((_resolve, reject) => {
        request.signal.addEventListener("abort", () => {
          aborted = true;
          reject(new Error("child aborted"));
        }, { once: true });
      }),
    },
  });

  await assert.rejects(
    runStructuredRole(
      { agent: { id: "judge-host" } },
      { label: "safe-timeout-label" },
      { prompt: "judge", outputSchema: { type: "object" } },
      {
        pendingStarts,
        admission: { isOpen: async () => true },
        persist: async () => ({ accepted: true }),
        timeoutMs: 20,
      },
    ),
    /operational timeout/,
  );
  assert.equal(aborted, true);
  assert.equal(pendingStarts.size, 0);
});

test("structured role exposes bounded phase codes without reflecting provider errors", async () => {
  const pendingStarts = new PendingChildStarts({
    subagents: {
      start: async () => ({
        id: "child-without-structured-result",
        result: Promise.resolve({ text: "not schema-bound" }),
        dispose: async () => {},
      }),
    },
  });
  await assert.rejects(
    runStructuredRole(
      { agent: { id: "researcher-host" } },
      { label: "safe-label" },
      { prompt: "research", outputSchema: { type: "object" } },
      {
        pendingStarts,
        admission: { isOpen: async () => true },
        persist: async () => ({ accepted: true }),
      },
    ),
    (error) => error.code === "structured_result_missing"
      && !String(error).includes("not schema-bound"),
  );
});
