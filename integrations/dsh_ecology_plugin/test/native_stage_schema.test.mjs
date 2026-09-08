import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import os from "node:os";
import { pathToFileURL } from "node:url";
import test from "node:test";

import { STAGES, prepareStageOutputSchema } from "../lib/runtime/stage-runner.js";
import { runStructuredRole } from "../lib/runtime/structured-roles.js";
import { PendingChildStarts } from "../lib/runtime/pending-child-starts.js";
import { RESEARCH_EXECUTION_POLICY } from "../lib/runtime/research-execution-policy.js";

// Exercise the actual native guard, not another copy of its schema rules.
// Standalone source checkouts can omit the optional DSH peer installation.
async function installedNativeValidator() {
  const roots = [
    import.meta.url,
    pathToFileURL(path.resolve(path.dirname(process.execPath), "..", "lib", "node_modules", "@deepseek-ai", "dsh", "package.json")).href,
    pathToFileURL(path.join(os.homedir(), ".local", "share", "ecologyrsi-dsh-dsh-cli", "node_modules", "@deepseek-ai", "dsh", "package.json")).href,
  ];
  for (const root of roots) {
    let modulePath;
    try { modulePath = createRequire(root).resolve("@deepseek-ai/dsh-tools"); }
    catch (error) { if (error.code === "MODULE_NOT_FOUND") continue; throw error; }
    return (await import(pathToFileURL(modulePath).href)).assertObjectJsonSchema;
  }
  return null;
}
const assertNativeSchema = await installedNativeValidator();
const options = { skip: assertNativeSchema ? false : "optional installed DSH tools peer is unavailable" };
const baseContext = {
  wave_digest: "a".repeat(64), samples: [{ sample_id: "sample:one" }], sample: { sample_id: "origin:one" },
  knowledge_snapshot: { evidence_catalog: [{ knowledge_id: "paper:one" }] },
  required_candidate_direction_count: 4,
  synthesis_contract: { allowed_mutation_targets: {
    scientific_parameter: ["ridge_alpha"], registered_predictor: ["ridge@1"], instruction_profile: ["balanced@1"],
  } },
};

test("actual native schema guard accepts every stage and the new bounded synthesis", options, async () => {
  for (const [stage, contract] of Object.entries(STAGES)) {
    const raw = JSON.parse(await readFile(new URL(`../schemas/${contract.file}.schema.json`, import.meta.url), "utf8"));
    assert.doesNotThrow(() => assertNativeSchema(prepareStageOutputSchema(stage, raw, baseContext)), stage);
    if (stage === "generation.research-synthesis") {
      for (const count of [1, 4, 8]) {
        for (const singleAxis of [false, true]) {
          const context = structuredClone(baseContext);
          context.research_execution_policy = { ...RESEARCH_EXECUTION_POLICY };
          context.required_candidate_direction_count = count;
          if (singleAxis) {
            context.synthesis_contract.allowed_mutation_targets.registered_predictor = [];
            context.synthesis_contract.allowed_mutation_targets.instruction_profile = [];
          }
          const schema = prepareStageOutputSchema(stage, raw, context);
          assert.doesNotThrow(() => assertNativeSchema(schema), `${stage}:${count}:${singleAxis}`);
          assert.match(schema.properties.summary.description, /1200 characters/);
          assert.equal(Object.hasOwn(schema.properties.summary, "maxLength"), false);
        }
      }
    }
  }
});

test("actual native unsupported schema has a deterministic non-model failure code", options, async () => {
  let starts = 0, persisted = false;
  const pendingStarts = new PendingChildStarts({ subagents: { start: async (_provider, request) => {
    starts += 1;
    assertNativeSchema(request.outputSchema);
    throw new Error("invalid schema must not reach model work");
  } } });
  await assert.rejects(runStructuredRole(
    { agent: { id: "schema-host" } }, { label: "invalid-native-schema" },
    { prompt: "bounded", outputSchema: { type: "object", properties: {
      summary: { type: "string", maxLength: 1200 },
    } }, maxTokens: 1024 },
    { pendingStarts, admission: { isOpen: async () => true },
      persist: async () => { persisted = true; } },
  ), error => error.code === "structured_child_output_schema_invalid");
  assert.equal(starts, 1);
  assert.equal(persisted, false);
  assert.equal(pendingStarts.size, 0);
});
