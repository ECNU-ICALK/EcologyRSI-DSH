#!/usr/bin/env node
// Explicit offline startup integration: generator CLI -> real profile/Cordis
// overlay -> registered LLM service -> installed adapter -> loopback fake HTTP.
// The temporary DSH_HOME and all credentials/configuration are synthetic.
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { createServer } from "node:http";
import { appendFile, cp, mkdir, mkdtemp, readFile, realpath, rm, stat, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";
import { dshPackageJson, installManagedPatch } from "../../../scripts/install_dsh_ecology_runtime.mjs";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const editableProfilePhase = process.env.ECOLOGYRSI_REASONING_IDENTITY_PHASE === "1";
const evidencePath = process.argv[2] ? path.resolve(process.argv[2]) : null;
const packageJson = await dshPackageJson(process.env.DSH_BIN || "dsh");
assert.ok(packageJson, "DSH_BIN must name an installed Harness executable");
const installed = JSON.parse(await readFile(packageJson, "utf8"));
assert.equal(installed.version, "0.2.0-rc.2");
const requireHost = createRequire(packageJson);
const importHost = name => import(pathToFileURL(requireHost.resolve(name)).href);
const yaml = requireHost("yaml");
const home = await realpath(await mkdtemp(path.join(os.tmpdir(), "ecology-reasoning-registration-")));
const generatorPath = path.join(root, "scripts/configure_dsh_reasoning.mjs");
const report = { schema_version: "ecologyrsi-dsh.reasoning-registration-verification/1",
  harness_version: installed.version, isolated_home: true, remote_model_calls: 0,
  user_settings_read: false, user_credentials_read: false,
  transport: "generator_CLI_to_real_profile_Cordis_LlmRuntime_PiAiAdapter_loopback_HTTP", cases: [],
  generator_source_sha256: createHash("sha256").update(await readFile(generatorPath)).digest("hex") };
const captured = [];
const server = createServer(async (req, res) => {
  assert.equal(req.url, "/v1/chat/completions");
  let raw = "";
  for await (const chunk of req) raw += chunk;
  const body = JSON.parse(raw); captured.push(body);
  res.writeHead(200, { "content-type": "text/event-stream" });
  const frame = (delta, finish_reason, usage) => "data: " + JSON.stringify({
    id: "local-registration-fixture", object: "chat.completion.chunk", created: 1, model: body.model,
    choices: [{ index: 0, delta, finish_reason }], ...(usage ? { usage } : {}),
  }) + "\n\n";
  res.end(frame({ role: "assistant", content: "local registration fixture" }, null)
    + frame({}, "stop", { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 }) + "data: [DONE]\n\n");
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let host;
try {
  const inputPath = path.join(home, "synthetic-input.yaml"), output = path.join(home, "generated");
  const providerId = "local-registration-fixture";
  const models = ["glm-5.2", "glm-5.3", "glm-5.3-flash"];
  const input = { "llm-pi-ai": { providers: { [providerId]: {
    api: "openai-completions", baseURL: `http://127.0.0.1:${server.address().port}/v1`,
    apiKeyEnv: "ECOLOGY_REGISTRATION_FIXTURE_KEY",
    models: models.map(id => ({ id, contextWindow: 8192, maxTokens: 2048 })),
  } } } };
  await writeFile(inputPath, yaml.stringify(input), { mode: 0o600 });
  const cli = await promisify(execFile)(process.execPath, [generatorPath,
    "--dsh-package", packageJson, "--source", inputPath, "--output", output,
    ...models.flatMap(model => ["--route", `${providerId}/${model}`])]);
  assert.ok(cli.stdout.includes("--profile web --patch"));
  assert.ok(!cli.stdout.includes("ECOLOGY_REGISTRATION_FIXTURE_KEY"));
  const patchPath = path.join(output, "settings.patch.yml");
  const patch = yaml.parse(await readFile(patchPath, "utf8"));
  const snapshotPath = path.join(output, "settings.yaml");
  const snapshot = yaml.parse(await readFile(snapshotPath, "utf8"));
  assert.deepEqual(patch, [{ id: "llm-pi-ai", config: snapshot["llm-pi-ai"] }]);
  for (const file of [patchPath, snapshotPath]) assert.equal((await stat(file)).mode & 0o777, 0o600);
  report.generated_patch_entry = patch[0].id;
  report.private_files_mode = "0600";
  await assert.rejects(promisify(execFile)(process.execPath, [generatorPath,
    "--dsh-package", packageJson, "--source", inputPath, "--output", output,
    ...models.flatMap(model => ["--route", `${providerId}/${model}`])]), /EEXIST/);
  assert.deepEqual(yaml.parse(await readFile(patchPath, "utf8")), patch, "repeat CLI must not overwrite the patch");

  process.env.DSH_HOME = home;
  process.env.DSH_TELEMETRY_DISABLED = "1";
  process.env.ECOLOGYRSI_DSH_RUNTIME_TOKEN = "isolated-registration-test";
  process.env.ECOLOGYRSI_SIDECAR_TOOL_TOKEN = "isolated-registration-sidecar";
  process.chdir(home);
  const { prepareProfile, runProfile } = await importHost("@deepseek-ai/dsh/profile-boot");
  const { createLaunchEnvironmentSnapshot } = await importHost("@deepseek-ai/dsh-launch-environment");
  const { createUserMessage } = await importHost("@deepseek-ai/dsh-llm");
  prepareProfile("web");
  const pkg = path.join(home, "profiles/web/node_modules/@ecologyrsi/dsh-evolution-plugin");
  await mkdir(pkg, { recursive: true });
  for (const entry of ["package.json", "lib", "presets", "schemas"]) {
    await cp(path.join(root, "integrations/dsh_ecology_plugin", entry), path.join(pkg, entry), { recursive: true });
  }
  await installManagedPatch({ dshHome: home, staticRoot: path.join(root, "plugins/ecology_evolution") });
  if (editableProfilePhase) {
    await appendFile(path.join(home, "profiles/web/cordis.patch.yml"), "\n" + yaml.stringify(patch));
  }
  const startupPatch = path.join(home, "startup.patch.yml");
  await writeFile(startupPatch, yaml.stringify([
    { id: "webserver", config: { host: "127.0.0.1", port: 0 } },
    { id: "web-runtime", config: { openBrowser: false, printUrl: false } },
    { id: "session-title-llm", disabled: true },
    { id: "ecologyrsi-evolution", config: { backendOrigin: "http://127.0.0.1:1" } },
  ]));
  const environment = createLaunchEnvironmentSnapshot([{ source: "process", values: {
      DSH_HOME: home, DSH_TELEMETRY_DISABLED: "1", ECOLOGY_REGISTRATION_FIXTURE_KEY: "local-test-placeholder",
    } }]);
  host = await runProfile({ profile: "web", patchFiles: editableProfilePhase ? [startupPatch] : [patchPath, startupPatch],
    args: ["--no-open"], environment });
  for (const [model, effort, expected] of [
    ["glm-5.2", undefined, "max"], ["glm-5.2", "off", undefined],
    ["glm-5.3", undefined, "max"], ["glm-5.3-flash", "low", "low"],
  ]) {
    const info = await host.ctx.llm.resolveModelInfo(providerId, model, AbortSignal.timeout(5000));
    assert.equal(info.reasoning.defaultEffort, "max");
    const prepared = await host.ctx.llm.prepareCall({ provider: providerId, model, maxTokens: 256,
      ...(effort === undefined ? {} : { reasoningEffort: effort }) }, AbortSignal.timeout(5000));
    assert.equal(prepared.config.reasoningEffort, effort ?? "max");
    const before = captured.length, chunks = [];
    for await (const chunk of prepared.stream({ ...prepared.config,
      messages: [createUserMessage({ content: [{ type: "text", text: "local fixture" }], source: { kind: "user" } })],
      signal: AbortSignal.timeout(5000) })) chunks.push(chunk);
    assert.equal(captured.length, before + 1);
    assert.deepEqual(chunks.filter(chunk => chunk.type === "finish").map(chunk => chunk.reason), [{ kind: "stop" }]);
    const body = captured.at(-1);
    assert.equal(body.reasoning_effort, expected);
    assert.equal(body.thinking.type, effort === "off" ? "disabled" : "enabled");
    assert.equal(body.max_tokens, 256);
    report.cases.push({ model, requested_effort: effort ?? "provider_default",
      resolved_effort: prepared.config.reasoningEffort, wire_reasoning_effort: body.reasoning_effort ?? null,
      wire_thinking_type: body.thinking.type });
  }
  const before = captured.length;
  await assert.rejects(host.ctx.llm.prepareCall({ provider: providerId, model: "glm-5.3-flash", reasoningEffort: "off" }),
    error => error.code === "UNSUPPORTED_REASONING_EFFORT");
  assert.equal(captured.length, before);
  report.unsupported_efforts_rejected_before_http = 1;

  // Read the actual installed service, then edit only this temporary profile.
  // This catches API drift that a fabricated settings.get() mock cannot catch.
  const { runtimeCapabilities } = await import(pathToFileURL(path.join(pkg, "lib/runtime/capabilities.js")).href);
  report.capabilities_source_sha256 = createHash("sha256")
    .update(await readFile(path.join(pkg, "lib/runtime/capabilities.js"))).digest("hex");
  const manifest = JSON.parse(await readFile(path.join(pkg, "presets/preset-manifest.json"), "utf8"));
  if (!editableProfilePhase) {
    // Command-line overlays are deliberately not editable through native
    // forms. A fresh child process tests the editable profile layer without
    // reusing Cordis's process-wide module/composition caches after shutdown.
    await assert.rejects(host.ctx.get("settings").update("llm-pi-ai", { providers: { [providerId]: { reasoning: "low" } } }),
      /overridden by a home patch or command-line overlay/);
    report.command_line_overlay_rejects_live_edit = true;
    await host.shutdown.shutdown(0);
    host = null;
    const identityEvidence = path.join(home, "identity-phase.json");
    await promisify(execFile)(process.execPath, [fileURLToPath(import.meta.url), identityEvidence], {
      env: { ...process.env, ECOLOGYRSI_REASONING_IDENTITY_PHASE: "1" }, timeout: 45000,
    });
    const childReport = JSON.parse(await readFile(identityEvidence, "utf8"));
    assert.equal(childReport.passed, true);
    report.live_configuration_identity = childReport.live_configuration_identity;
    report.identity_fixture_loopback_payloads = childReport.cases.length;
  } else {
  const settings = host.ctx.get("settings");
  assert.equal(typeof settings.get, "undefined");
  const snapshots = [];
  const captureCapabilities = async name => {
    const capabilities = await runtimeCapabilities(host.ctx, manifest.presets);
    assert.equal(capabilities.ready, true);
    assert.equal(capabilities.presets.length, manifest.presets.length);
    const serialized = JSON.stringify(capabilities);
    for (const privateValue of ["local-test-placeholder", "ECOLOGY_REGISTRATION_FIXTURE_KEY", "ROTATED_LOCAL_KEY", "authorization-fixture",
      "synthetic-user", "synthetic-password", "synthetic-query"]) {
      assert.ok(!serialized.includes(privateValue), "capabilities must not contain credential data");
    }
    snapshots.push({ name, capabilities });
    return capabilities.presets[0].content_digest;
  };
  const originalIdentity = await captureCapabilities("initial");
  await settings.update("llm-pi-ai", { providers: { [providerId]: { reasoning: "low" } } });
  await host.ctx.root.loader.await();
  assert.equal((await host.ctx.llm.resolveModelInfo(providerId, "glm-5.3-flash")).reasoning.defaultEffort, "low");
  const effortIdentity = await captureCapabilities("reasoning_low");
  assert.notEqual(effortIdentity, originalIdentity);
  await settings.update("llm-pi-ai", { providers: { [providerId]: { compat: { requiresToolResultName: true } } } });
  await host.ctx.root.loader.await();
  const compatIdentity = await captureCapabilities("compat_changed");
  assert.notEqual(compatIdentity, effortIdentity);
  await settings.update("llm-pi-ai", { providers: { [providerId]: {
    apiKeyEnv: "ROTATED_LOCAL_KEY", headers: { authorization: "authorization-fixture" },
    baseURL: `http://synthetic-user:synthetic-password@127.0.0.1:${server.address().port}/v1?key=synthetic-query`,
  } } });
  await host.ctx.root.loader.await();
  assert.equal(await captureCapabilities("authentication_rotated"), compatIdentity);

  // Exercise the real Python frozen-catalog and canary-identity helpers using
  // safe capability digests only. No model call or run is created here.
  const python = process.env.PYTHON_BIN || path.join(root, ".venv/bin/python");
  const identityCheck = await promisify(execFile)(python, ["-c", `
import json, sys
from ecologyrsi_dsh.api.handler import _dsh_native_stable_preset_catalog
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.integrations.model_canary import required_canary_identities
from ecologyrsi_dsh.integrations.dsh_native_runtime import DSH_NATIVE_EXECUTION_PROTOCOL
out=[]
for item in json.loads(sys.argv[1]):
    catalog=_dsh_native_stable_preset_catalog(item['capabilities'])
    cd=digest(catalog)
    metadata={'strategy_model_id':'local-registration-fixture/glm-5.2',
              'review_model_id':'local-registration-fixture/glm-5.3-flash',
              'preset_content_digest':cd,
              'standing_tool_surface_digest':digest([{'preset_id':p['preset_id'],'tool_surface_verified':p['tool_surface_verified']} for p in catalog])}
    for key, role, model in [('policy','strategy',metadata['strategy_model_id']),('review','review',metadata['review_model_id'])]:
        metadata[f'resolved_{key}_route_config_digest']=digest({'protocol':DSH_NATIVE_EXECUTION_PROTOCOL,'role':role,'model':model,'preset_catalog_digest':cd})
    identities=required_canary_identities(metadata)
    out.append({'name':item['name'],'preset_catalog_digest':cd,'route_digests':[metadata[f'resolved_{key}_route_config_digest'] for key in ('policy','review')],
                'canary_identity_digests':[digest(identity) for identity in identities]})
print(json.dumps(out))
`, JSON.stringify(snapshots)], { cwd: root, env: { ...process.env, PYTHONPATH: path.join(root, "src") } });
  const identities = JSON.parse(identityCheck.stdout);
  for (let index = 1; index <= 2; index++) {
    assert.notEqual(identities[index].preset_catalog_digest, identities[index - 1].preset_catalog_digest);
    assert.ok(identities[index].route_digests.every((value, i) => value !== identities[index - 1].route_digests[i]));
    assert.equal(identities[index].canary_identity_digests.length, 4);
    assert.ok(identities[index].canary_identity_digests.every((value, i) => value !== identities[index - 1].canary_identity_digests[i]));
  }
  assert.deepEqual(identities[3].canary_identity_digests, identities[2].canary_identity_digests);
  report.live_configuration_identity = { passed: true, uses_real_settings_describe: true,
    missing_legacy_get: true, cases: identities, authentication_rotation_preserves_identity: true };
  }
  report.passed = true;
  console.log(JSON.stringify({ passed: true, registered_models: models.length, loopback_payloads: report.cases.length,
    generated_patch_entry: patch[0].id, remote_model_calls: 0 }));
} finally {
  await host?.shutdown.shutdown(0);
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
  process.chdir(root);
  await rm(home, { recursive: true, force: true });
  report.isolated_home_removed = true;
  if (evidencePath) await writeFile(evidencePath, JSON.stringify(report, null, 2) + "\n");
}
