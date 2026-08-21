import { createHash } from "node:crypto";

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function valueDigest(value) {
  return createHash("sha256").update(canonical(value)).digest("hex");
}

export function reconcileStage(stageBinding, durableAcceptedItems, sessionItems = []) {
  const items = Array.isArray(stageBinding?.items) ? stageBinding.items : [];
  const accepted = new Map(
    (durableAcceptedItems || []).map((item) => [item.idempotency_key, item]),
  );
  const completed = [];
  const remaining = [];
  for (const item of items) {
    const receipt = accepted.get(item.idempotency_key);
    if (receipt && receipt.item_digest === item.item_digest) {
      completed.push(item.idempotency_key);
    } else {
      remaining.push(structuredClone(item));
    }
  }
  return Object.freeze({
    schema_version: "ecology-workflow-reconciliation/1",
    old_workflow_id: stageBinding.old_workflow_id || null,
    completed_idempotency_keys: Object.freeze(completed.sort()),
    remaining_items: Object.freeze(remaining),
    remaining_item_digest: valueDigest(remaining),
    ignored_session_completion_count: (sessionItems || []).filter(
      (item) => !completed.includes(item.idempotency_key),
    ).length,
    resume_old_workflow: false,
  });
}
