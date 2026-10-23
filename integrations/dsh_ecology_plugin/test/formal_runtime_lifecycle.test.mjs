import assert from "node:assert/strict";
import test from "node:test";
import { RuntimeController, CURRENT_PRESET_MANIFEST } from "../lib/runtime/controller.js";
import { roleSessionFingerprint } from "../lib/runtime/agents.js";

const identity = {
  run_id: "run:completed", run_state_revision: 12, stage_attempt: 0,
  ledger_expected_revision: 18, idempotency_key: "formal-prepare",
};
const frozen = {
  strategy_model_id: "test/policy", review_model_id: "test/review",
  preset_content_digest: "a".repeat(64), standing_tool_surface_digest: "b".repeat(64),
  resolved_policy_route_config_digest: "c".repeat(64), resolved_review_route_config_digest: "d".repeat(64),
};
function preparation(overrides = {}) {
  return { ...identity, binding: frozen, formal_execution: {
    execution_id: "1".repeat(64), stage: "validation", candidate_id: "candidate:1",
    candidate_revision_id: "revision:1", artifact_digest: "2".repeat(64),
    snapshot_digest: "3".repeat(64), ...overrides,
  } };
}
function harness() {
  const created = [], controls = [];
  const controller = new RuntimeController({}, { stageRunner: {
    closeLaunchFence: run => controls.push(["close", run]),
    openLaunchFence: run => controls.push(["open", run]),
    quiesceRun: async run => controls.push(["drain", run]),
    forgetRun: run => controls.push(["forget", run]),
  } });
  controller.roleAgents = {
    createRoleAgent: async binding => { created.push(binding); return {}; },
    quiesceRun: async (run, options) => controls.push(["roles", run, options.dispose]),
  };
  return { controller, created, controls };
}
const closeRequest = execution => ({ ...identity, idempotency_key: "formal-close",
  formal_execution_id: execution });

test("formal preparation replaces only a drained generation with fresh, frozen role identities", async () => {
  const { controller, created } = harness();
  await controller.startRun({ ...identity, binding: { ...frozen, initial_run_status: "running" } });
  await assert.rejects(controller.prepareFormalRun(preparation()), /quiesced|different start/);
  await controller.cancel({ ...identity, idempotency_key: "training-terminal" });
  const training = controller.registry.get(identity.run_id);
  const trainingGeneration = controller.registry.generationOf(training);
  const priorRoles = created.slice();
  const ready = await controller.prepareFormalRun(preparation());
  assert.equal(ready.status, "running");
  assert.equal(ready.formal_execution_id, "1".repeat(64));
  assert.equal(training.status, "cancelled");
  assert.notEqual(controller.registry.generationOf(controller.registry.get(identity.run_id)), trainingGeneration);
  assert.equal(created.length, CURRENT_PRESET_MANIFEST.presets.length * 2);
  for (const [i, binding] of created.slice(priorRoles.length).entries()) {
    assert.equal(binding.formal_execution_id, ready.formal_execution_id);
    assert.deepEqual({ ...binding, formal_execution_id: undefined }, { ...priorRoles[i], formal_execution_id: undefined });
    assert.notEqual(roleSessionFingerprint(binding), roleSessionFingerprint(priorRoles[i]));
  }
  assert.deepEqual(await controller.prepareFormalRun(preparation()), ready);
  assert.equal(created.length, priorRoles.length * 2);
  for (const method of ["startRun", "activate", "pause", "resume", "cancel"]) {
    await assert.rejects(controller[method]({ ...identity, binding: frozen }), /dedicated control/);
  }
  await assert.rejects(controller.closeFormalRun(closeRequest("9".repeat(64))), /identity mismatch/);
  assert.equal(controller.registry.get(identity.run_id).status, "running");
  const closed = await controller.closeFormalRun(closeRequest(ready.formal_execution_id));
  assert.equal(closed.status, "cancelled");
  assert.deepEqual(await controller.closeFormalRun(closeRequest(ready.formal_execution_id)), closed);
  await assert.rejects(controller.startRun({ ...identity, binding: frozen }), /dedicated control/);
  const next = await controller.prepareFormalRun(preparation({ execution_id: "4".repeat(64), stage: "final_test" }));
  assert.equal(next.formal_execution_id, "4".repeat(64));
  await assert.rejects(controller.closeFormalRun(closeRequest(ready.formal_execution_id)), /identity mismatch/);
  assert.equal(controller.registry.get(identity.run_id).status, "running");
});

test("formal preparation fails closed before readiness and restores the terminal generation", async () => {
  const { controller, controls } = harness();
  await controller.startRun({ ...identity, binding: { ...frozen, initial_run_status: "running" } });
  await controller.cancel(identity);
  const prior = controller.registry.get(identity.run_id);
  controller.roleAgents.createRoleAgent = async () => { throw new Error("role mount unavailable"); };
  controls.length = 0;
  await assert.rejects(controller.prepareFormalRun(preparation()), /mount unavailable/);
  assert.equal(controller.registry.get(identity.run_id), prior);
  assert.equal(controls.some(item => item[0] === "open"), false);
  await assert.rejects(controller.resume(identity), /cannot resume/);
});

test("cold formal preparation is bounded and rejects altered binding or execution replays", async () => {
  const { controller, created } = harness();
  for (const formal of [{ stage: "train" }, { snapshot_digest: "bad" }, { execution_id: "bad" },
    { candidate_id: "" }, { candidate_revision_id: "" }, { unknown: true }]) {
    await assert.rejects(controller.prepareFormalRun(preparation(formal)), /invalid formal/);
  }
  for (const extra of [{ initial_run_status: "running" }, { formal_execution: {} }, { restore_provenance: {} }]) {
    await assert.rejects(controller.prepareFormalRun({ ...preparation(), binding: { ...frozen, ...extra } }), /invalid formal/);
  }
  assert.equal(created.length, 0);
  await controller.prepareFormalRun(preparation());
  await assert.rejects(controller.prepareFormalRun({ ...preparation(), binding: { ...frozen, strategy_model_id: "other/model" } }), /different start/);
  await assert.rejects(controller.prepareFormalRun(preparation({ execution_id: "4".repeat(64) })), /different start/);
});

test("formal close joins concurrent cleanup with newer cursors and preparation waits for cancellation", async () => {
  const { controller } = harness();
  await controller.prepareFormalRun(preparation());
  let release, drains = 0;
  controller.stageRunner.quiesceRun = async () => {
    drains++;
    await new Promise(resolve => { release = resolve; });
  };
  const first = controller.closeFormalRun(closeRequest("1".repeat(64)));
  await new Promise(resolve => setImmediate(resolve));
  const second = controller.closeFormalRun({ ...closeRequest("1".repeat(64)),
    ledger_expected_revision: 22, run_state_revision: 20 });
  await assert.rejects(controller.prepareFormalRun(preparation({ execution_id: "4".repeat(64) })), /cannot start/);
  assert.equal(drains, 1);
  release();
  assert.equal((await first).status, "cancelled");
  const secondReceipt = await second;
  assert.equal(secondReceipt.status, "cancelled");
  assert.equal(secondReceipt.ledger_expected_revision, 22);
  assert.equal(secondReceipt.run_state_revision, 20);
});

test("a failed formal drain is retryable only for the matching execution", async () => {
  const { controller } = harness();
  await controller.prepareFormalRun(preparation());
  let drains = 0;
  controller.stageRunner.quiesceRun = async () => { if (++drains === 1) throw new Error("drain interrupted"); };
  await assert.rejects(controller.closeFormalRun(closeRequest("1".repeat(64))), /drain interrupted/);
  await assert.rejects(controller.closeFormalRun(closeRequest("4".repeat(64))), /identity mismatch/);
  assert.equal((await controller.closeFormalRun({ ...closeRequest("1".repeat(64)),
    ledger_expected_revision: 22 })).status, "cancelled");
  assert.equal(drains, 2);
});

test("closed execution IDs cannot be reused even after later formal generations", async () => {
  const { controller, created } = harness();
  const first = preparation(), second = preparation({ execution_id: "4".repeat(64) });
  await controller.prepareFormalRun(first);
  await controller.closeFormalRun(closeRequest(first.formal_execution.execution_id));
  const count = created.length;
  await assert.rejects(controller.prepareFormalRun({ ...first, ledger_expected_revision: 25 }), /closed formal execution/);
  assert.equal(created.length, count);
  await controller.prepareFormalRun(second);
  const generation = controller.registry.generationOf(controller.registry.get(identity.run_id));
  await assert.rejects(controller.closeFormalRun(closeRequest(first.formal_execution.execution_id)), /identity mismatch/);
  assert.equal(controller.registry.get(identity.run_id).status, "running");
  assert.equal(controller.registry.generationOf(controller.registry.get(identity.run_id)), generation);
  await controller.closeFormalRun(closeRequest(second.formal_execution.execution_id));
  await assert.rejects(controller.prepareFormalRun(first), /closed formal execution/);
});
