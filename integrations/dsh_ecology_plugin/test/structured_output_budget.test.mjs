import assert from "node:assert/strict";
import test from "node:test";
import { structuredOutputAttemptsExhausted, watchStructuredOutputBudget } from "../lib/runtime/structured-output-budget.js";
import { runStructuredRole } from "../lib/runtime/structured-roles.js";
import { PendingChildStarts } from "../lib/runtime/pending-child-starts.js";

function eventsFor(rejections) {
  const events = [];
  for (let index = 0; index < rejections; index++) {
    const seq = events.length, callId = `output-${index}`, step = index + 1;
    events.push({ seq, type: "tool/call", data: { turn: 1, step, callId, name: "structured_output" } });
    events.push({ seq: seq + 1, type: "tool/result", sourceEventSeqs: [seq], data: {
      turn: 1, step, error: { name: "ToolArgsError", code: "INVALID_ARGS" },
      message: { role: "tool", toolCallId: callId, source: { kind: "tool", callId }, isError: true, content: [] },
    } });
  }
  return events;
}

test("only source-bound native argument rejection exhausts the last allowed correction", () => {
  assert.equal(structuredOutputAttemptsExhausted(eventsFor(1), "sample.plan"), false);
  assert.equal(structuredOutputAttemptsExhausted(eventsFor(2), "sample.plan"), true);
  assert.equal(structuredOutputAttemptsExhausted(eventsFor(2), "generation.reflect"), false);
  assert.equal(structuredOutputAttemptsExhausted(eventsFor(3), "generation.reflect"), true);
  for (const mutate of [
    events => { events.at(-1).sourceEventSeqs = [999]; },
    events => { events.at(-1).data.step += 1; },
    events => { events.at(-1).data.turn += 1; },
    events => { events.at(-1).data.error.code = "TIMEOUT"; },
    events => { delete events.at(-1).data.error; },
    events => { events.at(-1).data.message.isError = false; },
    events => { events.push(structuredClone(events.at(-1))); },
    events => { events.at(-1).data.message.source.callId = "unbound"; },
  ]) {
    const events = eventsFor(2); mutate(events);
    assert.equal(structuredOutputAttemptsExhausted(events, "sample.plan"), false);
  }
});

function context() {
  const sessions = new Map(), handlers = new Set();
  const ctx = {
    sessions: { get: id => sessions.get(id) },
    on(name, handler, options) {
      assert.ok(["session/event", "tools/post-execute"].includes(name)); assert.equal(options.global, true);
      const entry = { name, handler };
      handlers.add(entry); return () => handlers.delete(entry);
    },
  };
  function session(id, parent = "host", label = "reserved") {
    const events = [], value = { id, header: { parentSession: parent }, snapshotEvents: () => events };
    sessions.set(id, value);
    value.append = (type, data, extra = {}) => {
      const event = { seq: events.length, type, data, ...extra }; events.push(event);
      for (const entry of handlers) if (entry.name === "session/event") entry.handler(value, event);
      return event;
    };
    value.append("subagent/descriptor", { label });
    return value;
  }
  function output(value, index, rejected = true) {
    const callId = `output-${index}`, step = index + 1;
    const call = value.append("tool/call", { turn: 1, step, callId, name: "structured_output" });
    value.append("tool/result", { turn: 1, step,
      ...(rejected ? { error: { name: "ToolArgsError", code: "INVALID_ARGS" } } : {}),
      message: { role: "tool", toolCallId: callId, source: { kind: "tool", callId }, isError: rejected, content: [] },
    }, { sourceEventSeqs: [call.seq] });
  }
  return { ctx, sessions, handlers, session, output };
}

test("synchronous publication stops at the role limit before another provider request, even before start returns", async () => {
  for (const stage of ["generation.research-synthesis", "sample.plan", "generation.reflect"]) {
    for (const beforeStartReturns of [false, true]) {
      const f = context(); let providerRequests = 0, disposed = 0, persisted = 0;
      const limit = stage === "generation.reflect" ? 3 : 2;
      f.ctx.subagents = { start: (_provider, request) => {
        const child = f.session("child");
        const execute = () => {
          for (let index = 0; index < 8; index++) {
            if (request.signal.aborted) break;
            providerRequests++;
            f.output(child, index);
          }
          return { stopReason: "aborted" };
        };
        const result = beforeStartReturns ? Promise.resolve(execute()) : new Promise(resolve => {
          setImmediate(() => resolve(execute()));
        });
        return { id: child.id, result, dispose: async () => { disposed++; } };
      } };
      const pendingStarts = new PendingChildStarts(f.ctx);
      await assert.rejects(runStructuredRole({ agent: { id: "host" } },
        { label: "reserved", binding: { stage } }, { prompt: "test", outputSchema: { type: "object" } },
        { eventContext: f.ctx, pendingStarts, timeoutMs: 500,
          persist: async () => { persisted++; return { accepted: true }; } }),
      error => error.code === "structured_output_attempt_limit_exceeded");
      await new Promise(resolve => setImmediate(resolve));
      assert.equal(providerRequests, limit, `${stage}:${beforeStartReturns}`);
      assert.equal(persisted, 0); assert.equal(disposed, 1);
      assert.equal(pendingStarts.size, 0); assert.equal(f.handlers.size, 0);
    }
  }
});

test("valid correction at the last allowed attempt is persisted without cancellation", async () => {
  for (const stage of ["generation.research-synthesis", "generation.reflect"]) {
    const f = context(), limit = stage === "generation.reflect" ? 3 : 2;
    let aborted = null, persisted = 0;
    f.ctx.subagents = { start: (_provider, request) => {
      const child = f.session("child");
      return { id: child.id, result: new Promise(resolve => setImmediate(() => {
        for (let index = 0; index < limit; index++) f.output(child, index, index < limit - 1);
        aborted = request.signal.aborted;
        resolve({ stopReason: "completed", structured: { answer: 1 } });
      })), dispose: async () => {} };
    } };
    const pendingStarts = new PendingChildStarts(f.ctx);
    await runStructuredRole({ agent: { id: "host" } }, { label: "reserved", binding: { stage } },
      { prompt: "test", outputSchema: { type: "object" } },
      { eventContext: f.ctx, pendingStarts, timeoutMs: 500, persist: async () => { persisted++; return { accepted: true }; } });
    assert.equal(aborted, false); assert.equal(persisted, 1); assert.equal(f.handlers.size, 0);
  }
});

test("before-handle guard rejects another parent or reservation and unsubscribes", () => {
  const f = context(), guard = watchStructuredOutputBudget(f.ctx, {
    stage: "sample.plan", parentSessionId: "host", label: "reserved",
  });
  for (const [id, parent, label] of [["wrong-parent", "other", "reserved"], ["wrong-label", "host", "other"]]) {
    const child = f.session(id, parent, label); f.output(child, 0); f.output(child, 1);
    assert.equal(guard.signal.aborted, false);
  }
  const own = f.session("own"); f.output(own, 0); f.output(own, 1);
  assert.equal(guard.signal.aborted, true);
  assert.throws(() => guard.throwIfExceeded(), error => error.code === "structured_output_attempt_limit_exceeded");
  guard.dispose(); guard.dispose(); assert.equal(f.handlers.size, 0);
});

test("existing projected rejection is checked at handle binding, while successful final output is allowed", () => {
  for (const rejected of [true, false]) {
    const f = context(), child = f.session("child"); f.output(child, 0); f.output(child, 1, rejected);
    const guard = watchStructuredOutputBudget(f.ctx, { stage: "sample.plan", parentSessionId: "host", label: "reserved" });
    guard.bindChild({ id: "child" }); assert.equal(guard.signal.aborted, rejected); guard.dispose();
  }
});

test("provider failure before attempt exhaustion keeps its original phase and cleanup", async () => {
  const f = context(); let persisted = false;
  f.ctx.subagents = { start: () => {
    const child = f.session("child"); f.output(child, 0);
    return { id: child.id, result: Promise.reject(new Error("private upstream error")), dispose: async () => {} };
  } };
  const pendingStarts = new PendingChildStarts(f.ctx);
  await assert.rejects(runStructuredRole({ agent: { id: "host" } },
    { label: "reserved", binding: { stage: "sample.plan" } },
    { prompt: "test", outputSchema: { type: "object" } },
    { eventContext: f.ctx, pendingStarts, timeoutMs: 500, persist: async () => { persisted = true; } }),
  error => error.code === "structured_child_result_failed" && !error.message.includes("private"));
  assert.equal(persisted, false); assert.equal(pendingStarts.size, 0); assert.equal(f.handlers.size, 0);
});

test("caller cancellation before or during synchronous start cannot persist a late result", async () => {
  for (const early of [true, false]) {
    const f = context(), controller = new AbortController();
    let childSignal, starts = 0, disposed = 0, persisted = false;
    if (early) controller.abort();
    f.ctx.subagents = { start: (_provider, request) => {
      starts++; childSignal = request.signal; controller.abort();
      return { id: "child", result: Promise.resolve({ structured: { answer: 1 } }), dispose: async () => { disposed++; } };
    } };
    const pendingStarts = new PendingChildStarts(f.ctx);
    await assert.rejects(runStructuredRole({ agent: { id: "host" } },
      { label: "reserved", binding: { stage: "sample.plan" } },
      { prompt: "test", outputSchema: { type: "object" } },
      { eventContext: f.ctx, pendingStarts, signal: controller.signal, timeoutMs: 500,
        persist: async () => { persisted = true; return { accepted: true }; } }),
    error => error.code === "structured_child_aborted");
    assert.equal(starts, early ? 0 : 1); assert.equal(persisted, false);
    if (!early) assert.equal(childSignal.aborted, true);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(disposed, early ? 0 : 1);
    assert.equal(pendingStarts.size, 0); assert.equal(f.handlers.size, 0);
  }
});
