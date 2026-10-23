import assert from "node:assert/strict";
import test from "node:test";
import { comparisonRetrievalPolicy } from "../lib/tools/retrieval.js";
import { assertModelArgumentsSafe } from "../lib/tools/definitions.js";

const context = {
  comparison_retrieval: { mode: "frozen_snapshot", snapshot_digest: "a".repeat(64),
    document_count: 8, online_enabled: false },
  comparison_retrieval_policy: "paired_method_snapshot@1",
  comparison_retrieval_scope: "methods_only_no_dataset_results",
};

test("comparison snapshot is accepted only for comparison stages and never certifies independent data", () => {
  for (const stage of ["sample.plan", "sample.critic", "sample.reflect", "generation.judge"]) {
    assert.deepEqual(comparisonRetrievalPolicy(context, stage), context.comparison_retrieval);
  }
  for (const stage of ["generation.research-synthesis", "generation.search-plan", "candidate.propose", "candidate.local_edit"]) {
    assert.throws(() => comparisonRetrievalPolicy(context, stage), /cannot grant/);
  }
  assert.throws(() => comparisonRetrievalPolicy(context, "sample.plan", { formalScope: true }), /cannot grant/);
  assert.throws(() => comparisonRetrievalPolicy({ ...context, formal_retrieval: context.comparison_retrieval }, "sample.plan"), /cannot grant/);
  assert.equal(comparisonRetrievalPolicy({}, "sample.plan"), null);
});

test("comparison snapshot identity is Host-owned and its offline boundary cannot be widened", () => {
  for (const key of ["comparison_retrieval", "comparison_retrieval_policy", "comparison_retrieval_scope"]) {
    assert.throws(() => assertModelArgumentsSafe({ [key]: context[key] }));
  }
  assert.throws(() => comparisonRetrievalPolicy({ ...context,
    comparison_retrieval: { ...context.comparison_retrieval, online_enabled: true } }, "sample.plan"), /network access disabled/);
  assert.throws(() => comparisonRetrievalPolicy({ ...context, comparison_retrieval_scope: "online_snippets" }, "sample.plan"), /cannot grant/);
});
