import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const ids = [
  "ecology-coordinator-v1",
  "ecology-researcher-v1",
  "ecology-candidate-proposer-v1",
  "ecology-sample-planner-v1",
  "ecology-sample-critic-v1",
  "ecology-generation-judge-v1",
];

test("six legal role presets expose only the narrow agent plane", async () => {
  for (const id of ids) {
    assert.match(id, /^[a-z0-9][a-z0-9-]*$/);
    const root = new URL(`../presets/${id}/`, import.meta.url);
    const metadata = await readFile(new URL("preset.yml", root), "utf8");
    const composition = await readFile(new URL("agent.cordis.yml", root), "utf8");
    assert.match(metadata, /name:/);
    assert.match(composition, /@deepseek-ai\/dsh-persona/);
    assert.match(composition, /@ecologyrsi\/dsh-evolution-plugin\/agent-plugin/);
    assert.match(composition, /@deepseek-ai\/dsh-compaction-basic/);
    assert.match(composition, /isolate:/);
    assert.doesNotMatch(composition, /dsh-tool-workflow|dsh-tool-subagent|dsh-tool-bash|dsh-tool-fs|dsh-tool-web|dsh-tool-ask-user|mcp/i);
  }
});

test("only workflow-driving roles mount the non-model-facing worker service", async () => {
  const workerRoles = new Set(["ecology-coordinator-v1", "ecology-sample-planner-v1"]);
  for (const id of ids) {
    const text = await readFile(new URL(`../presets/${id}/agent.cordis.yml`, import.meta.url), "utf8");
    assert.equal(text.includes("@deepseek-ai/dsh-workflow-worker-thread"), workerRoles.has(id));
  }
});
