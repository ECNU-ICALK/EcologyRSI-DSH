import assert from "node:assert/strict";
import test from "node:test";

import { reconcileStage } from "../lib/runtime/reconciliation.js";

const items = [
  { idempotency_key: "item-1", item_digest: "a".repeat(64) },
  { idempotency_key: "item-2", item_digest: "b".repeat(64) },
];

test("only Python-durable acceptance completes a Workflow item", () => {
  const result = reconcileStage(
    { old_workflow_id: "workflow-old", items },
    [{ idempotency_key: "item-1", item_digest: "a".repeat(64) }],
    [
      { idempotency_key: "item-1", status: "completed" },
      { idempotency_key: "item-2", status: "completed" },
    ],
  );
  assert.deepEqual(result.completed_idempotency_keys, ["item-1"]);
  assert.deepEqual(result.remaining_items, [items[1]]);
  assert.equal(result.ignored_session_completion_count, 1);
  assert.equal(result.resume_old_workflow, false);
});

test("zero and all durable acceptance reconcile deterministically", () => {
  assert.equal(reconcileStage({ items }, [], []).remaining_items.length, 2);
  assert.equal(reconcileStage({ items }, items, []).remaining_items.length, 0);
});
