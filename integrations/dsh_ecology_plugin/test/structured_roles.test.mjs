import assert from "node:assert/strict";
import test from "node:test";

import { runStructuredRole } from "../lib/runtime/structured-roles.js";
import { PendingChildStarts } from "../lib/runtime/workflows.js";

function blockFor(milliseconds) {
  const state = new Int32Array(new SharedArrayBuffer(4));
  Atomics.wait(state, 0, 0, milliseconds);
}

function operationalTimeout(error) {
  return error?.code === "structured_role_operational_timeout";
}

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

test("structured role deadline includes synchronous child-start work", async () => {
  let persistCalls = 0;
  const pendingStarts = new PendingChildStarts({
    subagents: {
      start: () => {
        blockFor(30);
        return {
          id: "slow-synchronous-start",
          result: Promise.resolve({ structured: { value: 1 } }),
          dispose: async () => {},
        };
      },
    },
  });

  await assert.rejects(
    runStructuredRole(
      { agent: { id: "researcher-host" } },
      { label: "slow-synchronous-start-label" },
      { prompt: "research", outputSchema: { type: "object" } },
      {
        pendingStarts,
        admission: { isOpen: async () => true },
        persist: async () => { persistCalls += 1; return { accepted: true }; },
        timeoutMs: 5,
      },
    ),
    operationalTimeout,
  );
  assert.equal(persistCalls, 0);
  assert.equal(pendingStarts.size, 0);
});

test("structured role deadline bounds a child start that ignores abort forever", async () => {
  const pendingStarts = new PendingChildStarts({
    subagents: { start: () => new Promise(() => {}) },
  });
  let watchdog = null;

  const outcome = await Promise.race([
    runStructuredRole(
      { agent: { id: "researcher-host" } },
      { label: "never-start-label" },
      { prompt: "research", outputSchema: { type: "object" } },
      {
        pendingStarts,
        admission: { isOpen: async () => true },
        persist: async () => ({ accepted: true }),
        timeoutMs: 20,
      },
    ).then(
      () => ({ kind: "resolved" }),
      (error) => ({ kind: "rejected", error }),
    ),
    new Promise((resolve) => {
      watchdog = setTimeout(() => resolve({ kind: "watchdog" }), 250);
    }),
  ]);
  clearTimeout(watchdog);

  assert.equal(outcome.kind, "rejected");
  assert.equal(outcome.error.code, "structured_role_operational_timeout");
  assert.equal(pendingStarts.size, 0);
});

test("structured role classifies sync and async admission boundary crossings as timeout", async () => {
  for (const admissionMode of ["sync", "async"]) {
    for (const admissionOutcome of ["closed", "rejected"]) {
      let persistCalls = 0;
      const pendingStarts = new PendingChildStarts({
        subagents: {
          start: async () => ({
            id: `admission-${admissionMode}-${admissionOutcome}-child`,
            result: Promise.resolve({ structured: { value: 1 } }),
            dispose: async () => {},
          }),
        },
      });
      const crossBoundary = () => {
        blockFor(30);
        if (admissionOutcome === "rejected") {
          throw new Error("private admission failure");
        }
        return false;
      };
      const isOpen = admissionMode === "sync"
        ? crossBoundary
        : async () => {
          await Promise.resolve();
          return crossBoundary();
        };
      await assert.rejects(
        runStructuredRole(
          { agent: { id: "judge-host" } },
          { label: `admission-${admissionMode}-${admissionOutcome}-label` },
          { prompt: "judge", outputSchema: { type: "object" } },
          {
            pendingStarts,
            admission: { isOpen },
            persist: async () => { persistCalls += 1; return { accepted: true }; },
            timeoutMs: 5,
          },
        ),
        (error) => operationalTimeout(error)
          && !String(error).includes("admission")
          && !String(error).includes("private"),
      );
      assert.equal(persistCalls, 0);
      assert.equal(pendingStarts.size, 0);
    }
  }
});

test("structured role rechecks its deadline after the persistence clone", async () => {
  const nativeStructuredClone = globalThis.structuredClone;
  let persistCalls = 0;
  globalThis.structuredClone = (value, options) => {
    if (value?.deadline_test === "pre-persist-clone") blockFor(30);
    return nativeStructuredClone(value, options);
  };
  try {
    const pendingStarts = new PendingChildStarts({
      subagents: {
        start: async () => ({
          id: "pre-persist-clone-child",
          result: Promise.resolve({
            structured: { deadline_test: "pre-persist-clone" },
          }),
          dispose: async () => {},
        }),
      },
    });
    await assert.rejects(
      runStructuredRole(
        { agent: { id: "researcher-host" } },
        { label: "pre-persist-clone-label" },
        { prompt: "research", outputSchema: { type: "object" } },
        {
          pendingStarts,
          admission: { isOpen: async () => true },
          persist: async () => { persistCalls += 1; return { accepted: true }; },
          timeoutMs: 5,
        },
      ),
      operationalTimeout,
    );
    assert.equal(persistCalls, 0);
  } finally {
    globalThis.structuredClone = nativeStructuredClone;
  }
});

test("structured role rechecks its deadline after the final return clone", async () => {
  const nativeStructuredClone = globalThis.structuredClone;
  let targetClones = 0;
  globalThis.structuredClone = (value, options) => {
    if (value?.deadline_test === "return-clone") {
      targetClones += 1;
      if (targetClones === 2) blockFor(40);
    }
    return nativeStructuredClone(value, options);
  };
  try {
    const pendingStarts = new PendingChildStarts({
      subagents: {
        start: async () => ({
          id: "return-clone-child",
          result: Promise.resolve({ structured: { deadline_test: "return-clone" } }),
          dispose: async () => {},
        }),
      },
    });
    await assert.rejects(
      runStructuredRole(
        { agent: { id: "judge-host" } },
        { label: "return-clone-label" },
        { prompt: "judge", outputSchema: { type: "object" } },
        {
          pendingStarts,
          admission: { isOpen: async () => true },
          persist: async () => ({ accepted: true }),
          timeoutMs: 10,
        },
      ),
      operationalTimeout,
    );
    assert.equal(targetClones, 2);
  } finally {
    globalThis.structuredClone = nativeStructuredClone;
  }
});

test("structured role rechecks its deadline after reading the persistence receipt", async () => {
  const pendingStarts = new PendingChildStarts({
    subagents: {
      start: async () => ({
        id: "slow-receipt-child",
        result: Promise.resolve({ structured: { value: 1 } }),
        dispose: async () => {},
      }),
    },
  });

  await assert.rejects(
    runStructuredRole(
      { agent: { id: "judge-host" } },
      { label: "slow-receipt-label" },
      { prompt: "judge", outputSchema: { type: "object" } },
      {
        pendingStarts,
        admission: { isOpen: async () => true },
        persist: async () => ({
          get accepted() {
            blockFor(30);
            return true;
          },
        }),
        timeoutMs: 5,
      },
    ),
    operationalTimeout,
  );
});

test("structured role keeps normal bookkeeping pending until disposal completes", async () => {
  for (const timeoutMs of [undefined, 500]) {
    let releaseDispose;
    let markDisposeStarted;
    const disposeStarted = new Promise((resolve) => { markDisposeStarted = resolve; });
    const disposeGate = new Promise((resolve) => { releaseDispose = resolve; });
    const pendingStarts = new PendingChildStarts({
      subagents: {
        start: async () => ({
          id: `normal-cleanup-${timeoutMs ?? "unbounded"}`,
          result: Promise.resolve({ structured: { value: 1 } }),
          dispose: async () => {
            markDisposeStarted();
            await disposeGate;
          },
        }),
      },
    });
    let settled = false;
    const running = runStructuredRole(
      { agent: { id: "judge-host" } },
      { label: `normal-cleanup-${timeoutMs ?? "unbounded"}-label` },
      { prompt: "judge", outputSchema: { type: "object" } },
      {
        pendingStarts,
        admission: { isOpen: async () => true },
        persist: async () => ({ accepted: true }),
        ...(timeoutMs === undefined ? {} : { timeoutMs }),
      },
    ).finally(() => { settled = true; });

    await disposeStarted;
    await Promise.resolve();
    assert.equal(settled, false);
    assert.equal(pendingStarts.size, 1);
    releaseDispose();
    await running;
    assert.equal(pendingStarts.size, 0);
  }
});

test("structured role finishes bookkeeping when disposal crosses the deadline", async () => {
  const pendingStarts = new PendingChildStarts({
    subagents: {
      start: async () => ({
        id: "cleanup-timeout-child",
        result: Promise.resolve({ structured: { value: 1 } }),
        dispose: async () => new Promise(() => {}),
      }),
    },
  });

  await assert.rejects(
    runStructuredRole(
      { agent: { id: "judge-host" } },
      { label: "cleanup-timeout-label" },
      { prompt: "judge", outputSchema: { type: "object" } },
      {
        pendingStarts,
        admission: { isOpen: async () => true },
        persist: async () => ({ accepted: true }),
        timeoutMs: 20,
      },
    ),
    operationalTimeout,
  );
  assert.equal(pendingStarts.size, 0);
});

test("structured role rejects a result that succeeds after its operational deadline", async () => {
  let admissionCalls = 0;
  let persistCalls = 0;
  const pendingStarts = new PendingChildStarts({
    subagents: {
      start: () => {
        blockFor(15);
        return {
          id: "late-success-child",
          result: () => {
            blockFor(15);
            return { structured: { value: "too late" } };
          },
          dispose: async () => {},
        };
      },
    },
  });

  await assert.rejects(
    runStructuredRole(
      { agent: { id: "researcher-host" } },
      { label: "late-success-label" },
      { prompt: "research", outputSchema: { type: "object" } },
      {
        pendingStarts,
        admission: { isOpen: async () => { admissionCalls += 1; return true; } },
        persist: async () => { persistCalls += 1; return { accepted: true }; },
        timeoutMs: 20,
      },
    ),
    (error) => error.code === "structured_role_operational_timeout",
  );
  assert.equal(admissionCalls, 0);
  assert.equal(persistCalls, 0);
  assert.equal(pendingStarts.size, 0);
});

test("structured role classifies a result rejection after the deadline as an operational timeout", async () => {
  let persistCalls = 0;
  const pendingStarts = new PendingChildStarts({
    subagents: {
      start: async () => ({
        id: "late-rejection-child",
        result: () => {
          blockFor(30);
          throw new Error("private late child failure");
        },
        dispose: async () => {},
      }),
    },
  });

  await assert.rejects(
    runStructuredRole(
      { agent: { id: "researcher-host" } },
      { label: "late-rejection-label" },
      { prompt: "research", outputSchema: { type: "object" } },
      {
        pendingStarts,
        admission: { isOpen: async () => true },
        persist: async () => { persistCalls += 1; return { accepted: true }; },
        timeoutMs: 10,
      },
    ),
    (error) => error.code === "structured_role_operational_timeout"
      && error.code !== "structured_child_result_failed"
      && !String(error).includes("private late child failure"),
  );
  assert.equal(persistCalls, 0);
  assert.equal(pendingStarts.size, 0);
});

test("structured role deadline does not wait for abort-ignoring result or disposal", async () => {
  let aborted = false;
  let persistCalls = 0;
  const pendingStarts = new PendingChildStarts({
    subagents: {
      start: async (_provider, request) => {
        request.signal.addEventListener("abort", () => { aborted = true; }, { once: true });
        return {
          id: "abort-ignoring-child",
          result: new Promise(() => {}),
          dispose: async () => new Promise(() => {}),
        };
      },
    },
  });
  const startedAt = Date.now();
  let watchdog = null;

  const outcome = await Promise.race([
    runStructuredRole(
      { agent: { id: "judge-host" } },
      { label: "abort-ignoring-label" },
      { prompt: "judge", outputSchema: { type: "object" } },
      {
        pendingStarts,
        admission: { isOpen: async () => true },
        persist: async () => { persistCalls += 1; return { accepted: true }; },
        timeoutMs: 20,
      },
    ).then(
      () => ({ kind: "resolved" }),
      (error) => ({ kind: "rejected", error }),
    ),
    new Promise((resolve) => {
      watchdog = setTimeout(() => resolve({ kind: "watchdog" }), 250);
    }),
  ]);
  clearTimeout(watchdog);

  assert.equal(outcome.kind, "rejected");
  assert.equal(outcome.error.code, "structured_role_operational_timeout");
  assert.equal(aborted, true);
  assert.equal(persistCalls, 0);
  assert.equal(pendingStarts.size, 0);
  assert.ok(Date.now() - startedAt < 250, "structured role exceeded its deadline watchdog");
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

test("structured role distinguishes a failed DSH turn from a missing capture", async () => {
  const pendingStarts = new PendingChildStarts({
    subagents: {
      start: async () => ({
        id: "rate-limited-child",
        result: Promise.resolve({
          stopReason: "error",
          output: [{ type: "text", text: "private provider failure" }],
        }),
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
    (error) => error.code === "structured_child_model_error"
      && !String(error).includes("private provider failure"),
  );
});
