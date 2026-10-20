#!/usr/bin/env node
// Boots the installed Harness against an isolated home and a deterministic local
// LLM adapter. No provider credentials, user sessions, or model API calls are used.
import assert from "node:assert/strict";
import { cp, mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dshPackageJson, installManagedPatch } from "./install_dsh_ecology_runtime.mjs";

const root = fileURLToPath(new URL("../", import.meta.url));
const packageJson = await dshPackageJson(process.env.DSH_BIN || "dsh");
assert.ok(packageJson, "a packaged DSH executable is required");
const installed = JSON.parse(await readFile(packageJson, "utf8"));
assert.equal(installed.version, "0.2.0-rc.2");
const requireHost = createRequire(packageJson);
const importHost = name => import(pathToFileURL(requireHost.resolve(name)).href);
const home = await realpath(await mkdtemp(path.join(os.tmpdir(), "ecology-harness-verify-")));
process.env.DSH_HOME = home;
process.env.DSH_TELEMETRY_DISABLED = "1";
process.env.ECOLOGYRSI_DSH_RUNTIME_TOKEN = "isolated-local-test";
process.env.ECOLOGYRSI_SIDECAR_TOOL_TOKEN = "isolated-local-sidecar";
process.chdir(home);
let host;
let manager;
try {
  const { prepareProfile, runProfile } = await importHost("@deepseek-ai/dsh/profile-boot");
  const { createLaunchEnvironmentSnapshot } = await importHost("@deepseek-ai/dsh-launch-environment");
  const { LlmAdapter } = await importHost("@deepseek-ai/dsh-llm");
  prepareProfile("web");
  const pkg = path.join(home, "profiles/web/node_modules/@ecologyrsi/dsh-evolution-plugin");
  await mkdir(pkg, { recursive: true });
  for (const entry of ["package.json", "lib", "presets", "schemas"]) {
    await cp(path.join(root, "integrations/dsh_ecology_plugin", entry), path.join(pkg, entry), { recursive: true });
  }
  await installManagedPatch({ dshHome: home, staticRoot: path.join(root, "plugins/ecology_evolution") });
  const overlay = path.join(home, "verify.patch.yml");
  await writeFile(overlay, `- id: webserver
  config:
    host: 127.0.0.1
    port: 0
- id: web-runtime
  config:
    openBrowser: false
    printUrl: false
- id: session-title-llm
  disabled: true
`);
  host = await runProfile({ profile: "web", patchFiles: [overlay], args: ["--no-open"],
    environment: createLaunchEnvironmentSnapshot([{ source: "process", values: process.env }]) });
  const { ctx } = host;
  let modelCalls = 0;
  let expectedFixture;
  class LocalAdapter extends LlmAdapter {
    async *stream(options) {
      modelCalls += 1;
      assert.ok(modelCalls <= 10, "unexpected extra generation");
      assert.ok(options.tools.some(tool => tool.name === "structured_output"));
      const block = { type: "tool-call", id: `local-call-${modelCalls}`,
        name: [7, 9].includes(modelCalls) ? "skill" : "structured_output",
        arguments: [7, 9].includes(modelCalls) ? JSON.stringify({ name: "autonomous-ecology-research" })
          : [8, 10].includes(modelCalls) ? JSON.stringify(expectedFixture) : '{"ok":true}' };
      yield { type: "block-start", index: 0, blockType: "tool-call" };
      yield { type: "tool-call-delta", index: 0, id: block.id, name: block.name, argumentsDelta: block.arguments };
      yield { type: "block-end", index: 0, block };
      yield { type: "usage", usage: { inputTokens: 200, outputTokens: 100 } };
      yield { type: "finish", reason: "tool-calls" };
    }
  }
  ctx.llm.registerAdapter(["ecology-local-test"], new LocalAdapter());
  const manifest = JSON.parse(await readFile(path.join(pkg, "presets/preset-manifest.json"), "utf8"));
  const { runtimeCapabilities } = await import(pathToFileURL(path.join(pkg, "lib/runtime/capabilities.js")));
  const capabilities = await runtimeCapabilities(ctx, manifest.presets);
  assert.equal(capabilities.ready, true, JSON.stringify(capabilities));
  const { RoleAgentManager } = await import(pathToFileURL(path.join(pkg, "lib/runtime/agents.js")));
  manager = new RoleAgentManager(ctx);
  const bindings = [];
  for (const preset of manifest.presets) {
    const role = preset.preset_id.replace(/^ecology-/, "").replace(/-v\d+$/, "");
    const binding = { run_id: "isolated-harness-verification", role, preset_id: preset.preset_id,
      model: "ecology-local-test/fake", cwd: home };
    const handle = await manager.createRoleAgent(binding);
    bindings.push({ ...binding, session_id: handle.sessionId });
    assert.ok(handle.services.compaction);
    const run = await ctx.subagents.start("spawn", { parent: handle.agent, label: `verify-${role}`,
      prompt: [{ type: "text", text: "Submit the local structured verification result." }],
      signal: AbortSignal.timeout(15000),
      outputSchema: { type: "object", properties: { ok: { type: "boolean" } }, required: ["ok"], additionalProperties: false },
    });
    try {
      const result = await run.result;
      assert.equal(result.stopReason, "completed", result.diagnostic);
      assert.deepEqual(result.structured, { ok: true });
    } finally { await run.dispose(); }
  }
  assert.equal(modelCalls, 6);
  await manager.quiesceRun("isolated-harness-verification", { dispose: true });
  for (const binding of bindings) {
    const resumed = await manager.resumeRoleAgent(binding);
    assert.equal(resumed.sessionId, binding.session_id);
    assert.ok(resumed.services.compaction);
  }
  await manager.quiesceRun("isolated-harness-verification", { dispose: true });
  // A real Skill -> structured_output chain catches Host event-shape changes
  // that array-shaped test doubles cannot exercise.
  const { ModelContractCanary } = await import(pathToFileURL(path.join(pkg, "lib/runtime/model-canary.js")));
  const { STAGES } = await import(pathToFileURL(path.join(pkg, "lib/runtime/stage-runner.js")));
  const stage = STAGES["generation.search-plan"];
  const schema = JSON.parse(await readFile(path.join(pkg, "schemas", `${stage.file}.schema.json`), "utf8"));
  function fixture(schema) {
    if ("const" in schema) return schema.const;
    if (schema.enum) return schema.enum[0];
    if (schema.type === "object") return Object.fromEntries((schema.required || []).map(key => [key, fixture(schema.properties[key])]));
    if (schema.type === "array") return Array.from({ length: schema.minItems || 1 }, () => fixture(schema.items));
    if (schema.type === "string") return schema.pattern === "^[0-9a-f]{64}$" ? "a".repeat(64) : "transport canary";
    if (schema.type === "boolean") return false;
    if (schema.type === "null") return null;
    return schema.minimum || 0;
  }
  expectedFixture = fixture(schema);
  const result = await new ModelContractCanary(ctx, { roleAgents: manager }).run({
    schema_version: "ecologyrsi-dsh.model-contract-canary/1",
    identity: { provider_id: "ecology-local-test", model_id: "fake", stage: "generation.search-plan",
      role: stage.role, preset_id: "ecology-researcher-v17", output_schema_id: stage.schema,
      preset_content_digest: "a".repeat(64), standing_tool_surface_digest: "b".repeat(64), route_config_digest: "c".repeat(64) },
    bounds: { max_attempts: 1, max_output_tokens: 1024, max_reported_tokens: 30000, total_timeout_ms: 10000, ttl_seconds: 3600 },
  });
  assert.equal(result.passed, true, JSON.stringify(result.failure));
  assert.equal(result.tool_evidence.order_verified, true);
  assert.equal(modelCalls, 8);
  // Exercise the actual native compiler and capture with absent protocol
  // metadata. The full research payload is still emitted by the local model.
  const { researchOutputTransport } = await import(pathToFileURL(path.join(pkg, "lib/runtime/research-contract.js")));
  const { dshCompatibleSchema, skillInvocationEvidence } = await import(pathToFileURL(path.join(pkg, "lib/runtime/stage-runner.js")));
  const { sessionEventLog } = await import(pathToFileURL(path.join(pkg, "lib/runtime/session-events.js")));
  const synthesisSchema = JSON.parse(await readFile(path.join(pkg, "schemas/research-synthesis.schema.json"), "utf8"));
  const expectedDomain = fixture(synthesisSchema);
  expectedFixture = { ...expectedDomain };
  delete expectedFixture.schema_version;
  const transport = researchOutputTransport("generation.research-synthesis", dshCompatibleSchema(synthesisSchema));
  const researcher = await manager.createRoleAgent({ run_id: "isolated-synthesis-transport", role: "researcher",
    preset_id: "ecology-researcher-v17", model: "ecology-local-test/fake", cwd: home });
  const synthesisRun = await ctx.subagents.start("spawn", { parent: researcher.agent, label: "verify-synthesis-version",
    prompt: [{ type: "text", text: "Submit the local transport fixture after Skill." }],
    signal: AbortSignal.timeout(15000), outputSchema: transport.schema });
  try {
    const captured = await synthesisRun.result;
    assert.equal(captured.stopReason, "completed", captured.diagnostic);
    assert.deepEqual(captured.structured, expectedFixture);
    assert.deepEqual(transport.bind(captured.structured), expectedDomain);
    const evidence = skillInvocationEvidence(sessionEventLog(ctx, synthesisRun.id), {
      stage: "generation.research-synthesis", skillName: "autonomous-ecology-research" });
    assert.equal(evidence.order_verified, true);
    assert.equal(modelCalls, 10);
  } finally { await synthesisRun.dispose(); }
  console.log(`Harness ${installed.version}: six presets, structured children, resume, cleanup, real tool-event canary and synthesis protocol binding passed (local adapter).`);
} finally {
  await manager?.dispose();
  await host?.shutdown.shutdown(0);
  process.chdir(root);
  await rm(home, { recursive: true, force: true });
}
