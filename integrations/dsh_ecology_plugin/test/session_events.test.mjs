// Pins the shape of the live DSH event log this plugin reads its evidence from.
// @deepseek-ai/dsh-session 0.1.5-rc.2 publishes the ordered log through
// snapshotEvents() and has no `events` property; every reader here used to
// dereference `.events`, so against the real Host all of them saw undefined
// while array-shaped test doubles kept the suite green. These cases fail if a
// reader goes back to the property.
import assert from "node:assert/strict";
import test from "node:test";

import { sessionEventLog } from "../lib/runtime/session-events.js";
import { checkSampleStageBudget } from "../lib/runtime/sample-stage-budget.js";
import { sessionUsageComplete } from "../lib/runtime/session-usage.js";
import { ChildBindingRegistry } from "../lib/runtime/child-bindings.js";

const EVENTS = [
  { seq: 1, type: "turn/start", data: { turn: 1 } },
  { seq: 2, type: "tool/call", data: { turn: 1, step: 1, callId: "c1", name: "skill" } },
  { seq: 3, type: "turn/end", data: { turn: 1, reason: { kind: "stop" } } },
];

// The installed Session: an ordered log behind a method, plus unrelated members.
function snapshotSession(events = EVENTS) {
  return {
    seq: events.length,
    ownEvents: () => events,
    eventAt: (seq) => events.find((event) => event.seq === seq),
    snapshotEvents: (from, toExclusive) => events.slice(
      from === undefined ? 0 : from - 1,
      toExclusive === undefined ? events.length : toExclusive - 1,
    ),
  };
}

function ctxWith(session, sessionId = "child-1") {
  return { sessions: { get: (id) => (id === sessionId ? session : undefined) } };
}

test("the live snapshot-shaped Session yields its ordered event log", () => {
  assert.deepEqual(sessionEventLog(ctxWith(snapshotSession()), "child-1"), EVENTS);
});

test("an array-shaped Session stays readable", () => {
  assert.deepEqual(sessionEventLog(ctxWith({ events: EVENTS }), "child-1"), EVENTS);
});

test("an absent, opaque, or racing Session is absent evidence, never an empty log", () => {
  // undefined is what every caller treats as "not proven"; [] would read as
  // "proven to have no events" and could pass a check that must fail.
  assert.equal(sessionEventLog(ctxWith(snapshotSession()), "other-child"), undefined);
  assert.equal(sessionEventLog({}, "child-1"), undefined);
  assert.equal(sessionEventLog(ctxWith(null), "child-1"), undefined);
  assert.equal(sessionEventLog(ctxWith({}), "child-1"), undefined);
  assert.equal(sessionEventLog(ctxWith({ snapshotEvents: () => null }), "child-1"), undefined);
  assert.equal(sessionEventLog(ctxWith({
    snapshotEvents: () => { throw new Error("session disposed"); },
  }), "child-1"), undefined);
});

test("sample stage budgets are enforced against a snapshot-shaped Session", () => {
  const overrun = Array.from({ length: 11 }, (_value, index) => (
    { seq: index + 1, type: "step/start", data: { turn: 1, step: index + 1 } }
  ));
  assert.throws(
    () => checkSampleStageBudget(ctxWith(snapshotSession(overrun)), "child-1", "sample.plan"),
    (error) => error.code === "structured_child_execution_budget_exhausted",
  );
  // The same log below the step ceiling must not abort the child.
  checkSampleStageBudget(ctxWith(snapshotSession(overrun.slice(0, 10))), "child-1", "sample.plan");
});

test("usage completeness reads the same log the Host actually publishes", () => {
  const events = [
    ...EVENTS,
    { seq: 4, type: "session/usage", data: { turn: 1 } },
  ];
  const fromSnapshot = sessionEventLog(ctxWith(snapshotSession(events)), "child-1");
  assert.equal(
    sessionUsageComplete(fromSnapshot, { settlement: { kind: "completed" }, freshUsage: true }),
    sessionUsageComplete(events, { settlement: { kind: "completed" }, freshUsage: true }),
  );
});

test("a child claims its reservation from the snapshot-shaped Session it really has", () => {
  // Production evidence: every `ecology_execute_prediction_tool` call returned
  // "Error: missing reservation" because the claim resolved the child's own
  // `subagent/descriptor` through `.events`, which the installed Session does
  // not expose. Structured persistence claims by the reserved label instead, so
  // planners still submitted -- they just submitted with no tool evidence, and
  // the whole prediction-tool axis measured zero.
  const registry = new ChildBindingRegistry();
  const reserved = registry.reserve("parent-1", {
    reservation_id: "reservation-1",
    run_id: "run-1",
    stage: "sample.plan",
    role: "sample-planner",
    item_digest: "a".repeat(64),
    idempotency_key: "idem-1",
    launch_attempt: 1,
  }, { role: "sample-planner", allowed_tools: ["ecology_execute_prediction_tool"] });
  const events = [
    { seq: 1, type: "turn/start", data: { turn: 1 } },
    { seq: 2, type: "subagent/descriptor", data: { label: reserved.label } },
  ];
  const child = {
    id: "child-1",
    session: { header: { parentSession: "parent-1" }, ...snapshotSession(events) },
  };
  assert.equal(registry.claim(child).role, "sample-planner");
  const binding = registry.bindingFor(child, {
    role: "sample-planner",
    toolName: "ecology_execute_prediction_tool",
  });
  assert.equal(binding.session_id, "child-1");
  assert.equal(binding.child_reservation_id, "reservation-1");
});
