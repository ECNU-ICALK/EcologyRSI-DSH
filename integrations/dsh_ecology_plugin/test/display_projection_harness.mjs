#!/usr/bin/env node
// Explicit isolated integration: real rc.2 profile, PiAiAdapter and native
// three-step Agent against a loopback HTTP provider; no remote model calls.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createServer } from "node:http";
import { cp, mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dshPackageJson, installManagedPatch } from "../../../scripts/install_dsh_ecology_runtime.mjs";
import { sampleContextFixture, predictionResultFixture } from "./fixtures/display_projection.mjs";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const evidence = process.argv[2] ? path.resolve(process.argv[2]) : null;
const packageJson = await dshPackageJson(process.env.DSH_BIN || "dsh");
assert.ok(packageJson, "DSH_BIN must identify installed Harness");
const installed = JSON.parse(await readFile(packageJson, "utf8"));
assert.equal(installed.version, "0.2.0-rc.2");
const requireHost = createRequire(packageJson), yaml = requireHost("yaml");
const importHost = name => import(pathToFileURL(requireHost.resolve(name)).href);
const home = await realpath(await mkdtemp(path.join(os.tmpdir(), "ecology-display-projection-")));
const context = sampleContextFixture(), fullResult = predictionResultFixture(context);
const originalContext = structuredClone(context), originalResult = structuredClone(fullResult);
const wireOutput = { decisions: context.samples.map((sample, index) => ({
  sample_id: sample.sample_id, predicted: 10 + index, confidence: 0.8,
  method: "model", reason_code: "agent_model", evidence_call_ids: ["candidate-evidence"],
})) };
const canonical = value => Array.isArray(value) ? `[${value.map(canonical).join(",")}]`
  : value && typeof value === "object" ? `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`
  : JSON.stringify(value);
const bytes = value => Buffer.byteLength(JSON.stringify(value));
const report = { schema_version: "ecologyrsi-dsh.display-projection-harness/1", harness_version: installed.version,
  isolated_home: true, remote_model_calls: 0, transport: "real_Harness_native_Agent_PiAiAdapter_to_loopback_HTTP",
  sidecar: "in_memory_authoritative_values", requests: [], passed: false };
let host, manager, projection, serverError;
const reservations = [], persisted = [], failures = [], toolRequests = [];
const parseJson = text => { try { return JSON.parse(text); } catch { return null; } };
const textOf = message => typeof message.content === "string" ? message.content
  : (message.content || []).filter(b => b.type === "text").map(b => b.text).join("");
const server = createServer(async (request, response) => {
  try {
    assert.equal(request.url, "/v1/chat/completions");
    let raw = ""; for await (const chunk of request) raw += chunk;
    const body = JSON.parse(raw), index = report.requests.length;
    assert.ok(index < 3, "only Skill, prediction tool and structured output are allowed");
    const prompt = body.messages.filter(m => m.role === "user").map(textOf).map(parseJson)
      .find(p => p?.stage === "sample.plan" && p?.context);
    assert.ok(prompt, "native stage prompt must reach real HTTP payload");
    assert.equal(prompt.context._display_projection.kind, "sample-context");
    assert.deepEqual(projection.expandSampleContext(prompt.context), context);
    assert.ok(bytes(prompt.context) < bytes(context), "complete compressed prompt must be smaller");
    const tools = body.tools.map(t => t.function.name);
    for (const name of ["skill", "ecology_execute_prediction_tool", "structured_output"]) assert.ok(tools.includes(name));
    const requestSummary = { index: index + 1, prompt_before_bytes: bytes(context),
      prompt_after_bytes: bytes(prompt.context), prompt_roundtrip_equal: true, tool_schema_count: tools.length };
    if (index === 2) {
      const value = body.messages.filter(m => m.role === "tool").map(textOf).map(parseJson)
        .find(p => p?._display_projection?.kind === "prediction-tool-parameters");
      assert.ok(value, "compressed prediction render must reach real adapter request");
      assert.deepEqual(projection.expandPredictionToolResult(value), fullResult);
      assert.equal(value.output_digest, fullResult.output_digest);
      assert.ok(bytes(value) < bytes(fullResult));
      Object.assign(requestSummary, { tool_result_before_bytes: bytes(fullResult), tool_result_after_bytes: bytes(value),
        tool_roundtrip_equal: true, output_digest_preserved: true });
    }
    const name = ["skill", "ecology_execute_prediction_tool", "structured_output"][index];
    const args = index === 0 ? { name: "origin-vector-forecasting" } : index === 1
      ? { tool_id: "candidate-model", call_id: "candidate-evidence", wave_digest: context.wave_digest, parameters: {} }
      : wireOutput;
    requestSummary.next_tool = name; report.requests.push(requestSummary);
    response.writeHead(200, { "content-type": "text/event-stream" });
    const frame = (delta, finishReason, usage) => "data: " + JSON.stringify({ id: `local-display-${index}`,
      object: "chat.completion.chunk", created: 1, model: body.model,
      choices: [{ index: 0, delta, finish_reason: finishReason }], ...(usage ? { usage } : {}) }) + "\n\n";
    response.end(frame({ role: "assistant", tool_calls: [{ index: 0, id: `display-call-${index}`,
      type: "function", function: { name, arguments: JSON.stringify(args) } }] }, null)
      + frame({}, "tool_calls", { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }) + "data: [DONE]\n\n");
  } catch (error) {
    serverError ||= error;
    response.writeHead(400, { "content-type": "application/json" });
    response.end(JSON.stringify({ error: { message: "local fixture assertion failed", type: "invalid_request_error" } }));
  }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
try {
  process.env.DSH_HOME = home; process.env.DSH_TELEMETRY_DISABLED = "1";
  process.env.ECOLOGY_DISPLAY_FIXTURE_KEY = "isolated-local-placeholder";
  process.env.ECOLOGYRSI_DSH_RUNTIME_TOKEN = "isolated-display-runtime";
  process.env.ECOLOGYRSI_SIDECAR_TOOL_TOKEN = "isolated-display-sidecar";
  process.chdir(home);
  const { prepareProfile, runProfile } = await importHost("@deepseek-ai/dsh/profile-boot");
  const { createLaunchEnvironmentSnapshot } = await importHost("@deepseek-ai/dsh-launch-environment");
  prepareProfile("web");
  const pkg = path.join(home, "profiles/web/node_modules/@ecologyrsi/dsh-evolution-plugin");
  await mkdir(pkg, { recursive: true });
  for (const entry of ["package.json", "lib", "presets", "schemas"]) {
    await cp(path.join(root, "integrations/dsh_ecology_plugin", entry), path.join(pkg, entry), { recursive: true });
  }
  await installManagedPatch({ dshHome: home, staticRoot: path.join(root, "plugins/ecology_evolution") });
  const overlay = path.join(home, "verify.patch.yml");
  await writeFile(overlay, yaml.stringify([
    { id: "webserver", config: { host: "127.0.0.1", port: 0 } },
    { id: "web-runtime", config: { openBrowser: false, printUrl: false } },
    { id: "session-title-llm", disabled: true },
    { id: "ecologyrsi-evolution", disabled: true },
    { id: "llm-pi-ai", config: { providers: { "local-display": { api: "openai-completions",
      baseURL: `http://127.0.0.1:${server.address().port}/v1`, apiKeyEnv: "ECOLOGY_DISPLAY_FIXTURE_KEY",
      models: [{ id: "model", contextWindow: 131072, maxTokens: 32768 }] } } } },
  ]));
  host = await runProfile({ profile: "web", patchFiles: [overlay], args: ["--no-open"],
    environment: createLaunchEnvironmentSnapshot([{ source: "process", values: process.env }]) });
  const importPlugin = file => import(pathToFileURL(path.join(pkg, "lib/runtime", file)).href);
  projection = await importPlugin("display-projection.js");
  const { RoleAgentManager } = await importPlugin("agents.js");
  const { NativeStageRunner, jsonDigest } = await importPlugin("stage-runner.js");
  manager = new RoleAgentManager(host.ctx);
  const sidecar = { request: async (route, options) => {
    if (route.endsWith("/session-usage")) return { accepted: true };
    if (route.endsWith("/child-reservations")) {
      const launch = { reservation_id: "display-reservation", launch_attempt: 1 }; reservations.push(launch);
      return { accepted: true, admission_id: options.body.admission_id, timeout_ms: options.body.timeout_ms, launch,
        ledger_expected_revision: 2 };
    }
    if (route.endsWith("/tools/ecology_execute_prediction_tool")) {
      toolRequests.push(structuredClone(options.body)); return fullResult;
    }
    if (route.endsWith("/structured-results/validate")) return {
      schema_version: "ecologyrsi-dsh.prediction-prevalidation/1", valid: true,
      result_digest: options.body.result_digest, diagnostics: [],
    }; // Semantic Host validation is exercised by the authoritative prevalidation harness.
    if (route.endsWith("/child-failures")) { failures.push(options.body); return { accepted: true }; }
    assert.ok(route.endsWith("/structured-results"), "unexpected in-memory sidecar endpoint");
    persisted.push(structuredClone(options.body)); return { accepted: true, result_digest: options.body.result_digest };
  } };
  const runner = new NativeStageRunner(host.ctx, { roleAgents: manager, sidecar,
    runRegistry: { get: () => ({ status: "running" }) }, structuredStageMaxAttempts: 1,
    samplePlannerStageTimeoutMs: 15000, structuredStageMinIntervalMs: 0 });
  host.ctx.provide("ecologyAgentTools", runner.bridge);
  const runId = "local-display-projection";
  const manifest = JSON.parse(await readFile(path.join(pkg, "presets/preset-manifest.json"), "utf8"));
  await manager.createRoleAgent({ run_id: runId, role: "sample-planner",
    preset_id: manifest.presets.find(p => p.role === "sample-planner").preset_id,
    model: "local-display/model", cwd: home });
  const binding = { run_id: runId, stage: "sample.plan", run_state_revision: 1, stage_attempt: 1,
    ledger_expected_revision: 1, admission_id: "display-admission", idempotency_key: "display-idempotency",
    request: { role: "sample-planner", output_schema_id: "ecology-sample-predictions@2", max_tokens: 2048,
      context, context_canonical_json: canonical(context), context_digest: jsonDigest(context),
      identity_digests: { genome_digest: "a".repeat(64), compiled_behavior_digest: "b".repeat(64),
        phenotype_instance_digest: "c".repeat(64) } } };
  let result, caught;
  try { result = await runner.run(binding); } catch (error) { caught = error; }
  assert.equal(serverError, undefined, serverError?.message);
  assert.equal(caught, undefined, caught?.message);
  const expected = { ...wireOutput, schema_version: "ecology-sample-predictions@2", wave_digest: context.wave_digest };
  assert.deepEqual(result.structured, expected);
  assert.equal(report.requests.length, 3); assert.equal(reservations.length, 1);
  assert.equal(toolRequests.length, 1); assert.equal(persisted.length, 1); assert.equal(failures.length, 0);
  assert.equal(persisted[0].skill_invocation_evidence.order_verified, true);
  assert.equal(persisted[0].result_digest, jsonDigest(expected));
  assert.deepEqual(context, originalContext); assert.deepEqual(fullResult, originalResult);
  report.authoritative_context_unchanged = true; report.authoritative_tool_result_unchanged = true;
  report.structured_result_digest_preserved = true; report.persisted_results = persisted.length;
  report.reservations = reservations.length; report.predictions = expected.decisions.length;
  report.source_sha256 = {};
  for (const file of ["lib/runtime/display-projection.js", "lib/runtime/stage-runner.js", "lib/tools/roles.js"]) {
    report.source_sha256[file] = createHash("sha256").update(await readFile(path.join(pkg, file))).digest("hex");
  }
  report.passed = true;
  console.log(JSON.stringify(report));
} finally {
  await manager?.dispose(); await host?.shutdown.shutdown(0);
  await new Promise(resolve => server.close(resolve));
  process.chdir(root); await rm(home, { recursive: true, force: true });
  report.isolated_home_removed = true;
  if (evidence) { await mkdir(path.dirname(evidence), { recursive: true }); await writeFile(evidence, JSON.stringify(report, null, 2) + "\n"); }
}
