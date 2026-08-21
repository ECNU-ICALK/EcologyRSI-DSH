import assert from "node:assert/strict";
import test from "node:test";

import { ChildBindingRegistry, childReservationLabel } from "../lib/runtime/child-bindings.js";

function launch(overrides = {}) {
  return {
    reservation_id: "reservation-1",
    run_id: "run-1",
    stage: "research",
    role: "researcher",
    item_digest: "a".repeat(64),
    idempotency_key: "idem-1",
    launch_attempt: 1,
    ...overrides,
  };
}

test("ledger-issued launch attempts domain-separate labels and tombstones survive registry restart", () => {
  const old = launch();
  const next = launch({ reservation_id: "reservation-2", launch_attempt: 2 });
  assert.notEqual(childReservationLabel(old), childReservationLabel(next));
  assert.notEqual(childReservationLabel(old), childReservationLabel({ ...old, reservation_id: "r3", run_id: "run-2" }));
  const durable = { tombstones: new Set(), reservations: new Map() };
  const first = new ChildBindingRegistry({ durable });
  first.reserve("parent-1", old, { role: "researcher", revision: 3 });
  first.revoke("parent-1", childReservationLabel(old));
  const restarted = new ChildBindingRegistry({ durable });
  assert.throws(() => restarted.reserve("parent-1", old, {}), /tombstoned/);
});

test("pre-registered parent plus descriptor label authorizes the first tool call exactly once", () => {
  const registry = new ChildBindingRegistry();
  const durableLaunch = launch();
  const reserved = registry.reserve("parent-1", durableLaunch, { role: "researcher", revision: 3 });
  const child = {
    id: "child-1",
    session: {
      header: { parentSession: "parent-1" },
      events: [{ type: "subagent/descriptor", data: { label: reserved.label } }],
    },
  };
  assert.equal(registry.claim(child).role, "researcher");
  assert.throws(() => registry.claim({ ...child, id: "child-2" }), /already claimed/);
  assert.throws(() => registry.claim({ ...child, session: { ...child.session, header: { parentSession: "wrong" } } }), /missing reservation/);
});

test("activation leases rotate only after the previous turn is closed", () => {
  const registry = new ChildBindingRegistry();
  const { label } = registry.reserve("parent-1", launch(), { role: "researcher" });
  registry.claim({ id: "child-1", session: { header: { parentSession: "parent-1" }, events: [{ type: "subagent/descriptor", data: { label } }] } });
  const first = registry.openActivation("child-1", { revision: 1, stage_attempt: 1, wave_digest: "w1", idempotency_key: "i1" });
  assert.throws(() => registry.openActivation("child-1", { revision: 2, stage_attempt: 2, wave_digest: "w2", idempotency_key: "i2" }), /active lease/);
  registry.closeActivation(first.lease_id);
  const second = registry.openActivation("child-1", { revision: 2, stage_attempt: 2, wave_digest: "w2", idempotency_key: "i2" });
  assert.notEqual(first.lease_id, second.lease_id);
});
