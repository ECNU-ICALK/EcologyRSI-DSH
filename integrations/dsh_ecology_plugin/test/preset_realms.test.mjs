import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const ids = [
  "ecology-coordinator-v4",
  "ecology-researcher-v7",
  "ecology-candidate-proposer-v4",
  "ecology-sample-planner-v4",
  "ecology-sample-critic-v4",
  "ecology-generation-judge-v7",
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
    assert.match(composition, /web_search/);
    assert.match(composition, /never\s+(?:choose|select|name)\s+(?:a\s+)?provider/i);
    assert.match(composition, /isolate:/);
    assert.doesNotMatch(composition, /dsh-tool-workflow|dsh-tool-subagent|dsh-tool-bash|dsh-tool-fs|dsh-tool-web|dsh-tool-ask-user|mcp/i);
  }
});

test("no retained role mounts the unused Workflow worker service", async () => {
  for (const id of ids) {
    const text = await readFile(new URL(`../presets/${id}/agent.cordis.yml`, import.meta.url), "utf8");
    assert.equal(text.includes("@deepseek-ai/dsh-workflow-worker-thread"), false);
  }
});

test("generation judge preset separates candidate review from batch reflection", async () => {
  const composition = await readFile(
    new URL(
      "../presets/ecology-generation-judge-v7/agent.cordis.yml",
      import.meta.url,
    ),
    "utf8",
  );
  const skillsRoot = new URL(
    "../presets/ecology-generation-judge-v7/skills/",
    import.meta.url,
  );
  const candidateReview = await readFile(
    new URL("candidate-scientific-review/SKILL.md", skillsRoot),
    "utf8",
  );
  const batchReflection = await readFile(
    new URL("batch-scientific-reflection/SKILL.md", skillsRoot),
    "utf8",
  );

  assert.match(composition, /Follow the stage-specific Skill/i);
  assert.match(composition, /For a candidate review/);
  assert.match(composition, /For batch reflection/);
  assert.match(candidateReview, /name: candidate-scientific-review/);
  assert.match(candidateReview, /candidate_id/);
  assert.match(candidateReview, /proposal_id/);
  assert.match(candidateReview, /scientific_evaluation/);
  assert.match(candidateReview, /fitness_profile_digest/);
  assert.match(candidateReview, /evaluation_cohort_digest/);
  assert.match(candidateReview, /schema_version[\s\S]*accepted[\s\S]*rationale[\s\S]*flags/);
  assert.match(candidateReview, /Never propose next-generation directions/i);
  assert.match(candidateReview, /not selection, ranking, promotion/i);
  assert.doesNotMatch(candidateReview, /Propose exactly the requested number/i);
  assert.match(batchReflection, /name: batch-scientific-reflection/);
  assert.match(batchReflection, /Propose exactly the requested number/i);
});
