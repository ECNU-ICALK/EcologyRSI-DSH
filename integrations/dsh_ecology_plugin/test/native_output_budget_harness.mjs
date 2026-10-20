#!/usr/bin/env node
// Explicit integration test: installed Harness + native schema compiler/event
// loop, isolated home, in-process fake provider and in-memory sidecar receipts.
// Run: node integrations/dsh_ecology_plugin/test/native_output_budget_harness.mjs [evidence.json]
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { cp, mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dshPackageJson, installManagedPatch } from "../../../scripts/install_dsh_ecology_runtime.mjs";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const evidencePath = process.argv[2] ? path.resolve(process.argv[2]) : null;
const packageJson = await dshPackageJson(process.env.DSH_BIN || "dsh");
assert.ok(packageJson, "a packaged DSH executable is required for this explicit integration test");
const installed = JSON.parse(await readFile(packageJson, "utf8"));
assert.equal(installed.version, "0.2.0-rc.2");
const requireHost = createRequire(packageJson);
const importHost = name => import(pathToFileURL(requireHost.resolve(name)).href);
const home = await realpath(await mkdtemp(path.join(os.tmpdir(), "ecology-output-budget-")));
process.env.DSH_HOME = home;
process.env.DSH_TELEMETRY_DISABLED = "1";
process.env.ECOLOGYRSI_DSH_RUNTIME_TOKEN = "isolated-output-budget-test";
process.env.ECOLOGYRSI_SIDECAR_TOOL_TOKEN = "isolated-output-budget-sidecar";
process.chdir(home);
let host, manager, offEvents;
const report = {schema_version: "ecologyrsi-dsh.native-output-budget-verification/1",
  harness_version: installed.version, provider: "in_process_deterministic_adapter",
  sidecar: "in_memory_receipts", isolated_home: true, remote_model_calls: 0, cases: []};

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
  return JSON.stringify(value);
}
function schemaFixture(schema) {
  if ("const" in schema) return schema.const;
  if (schema.oneOf) return schemaFixture(schema.oneOf[0]);
  if (schema.enum) return schema.enum[0];
  if (schema.type === "object") return Object.fromEntries((schema.required || []).map(key => [key, schemaFixture(schema.properties[key])]));
  if (schema.type === "array") return Array.from({length: schema.minItems || 1}, () => schemaFixture(schema.items));
  if (schema.type === "string") return "local schema fixture";
  if (schema.type === "boolean") return false;
  if (schema.type === "null") return null;
  return schema.minimum || 0;
}

try {
  const {prepareProfile, runProfile} = await importHost("@deepseek-ai/dsh/profile-boot");
  const {createLaunchEnvironmentSnapshot} = await importHost("@deepseek-ai/dsh-launch-environment");
  const {LlmAdapter} = await importHost("@deepseek-ai/dsh-llm");
  prepareProfile("web");
  const pkg = path.join(home, "profiles/web/node_modules/@ecologyrsi/dsh-evolution-plugin");
  await mkdir(pkg, {recursive: true});
  for (const entry of ["package.json", "lib", "presets", "schemas"]) {
    await cp(path.join(root, "integrations/dsh_ecology_plugin", entry), path.join(pkg, entry), {recursive: true});
  }
  report.source_sha256 = {};
  for (const file of ["structured-output-budget.js", "structured-roles.js", "pending-child-starts.js", "stage-runner.js"]) {
    report.source_sha256[file] = createHash("sha256").update(await readFile(path.join(pkg, "lib/runtime", file))).digest("hex");
  }
  await installManagedPatch({dshHome: home, staticRoot: path.join(root, "plugins/ecology_evolution")});
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
- id: ecologyrsi-evolution
  config:
    backendOrigin: 'http://127.0.0.1:1'
`);
  host = await runProfile({profile: "web", patchFiles: [overlay], args: ["--no-open"],
    environment: createLaunchEnvironmentSnapshot([{source: "process", values: process.env}])});
  const {ctx} = host;
  let activeCase;
  class LocalAdapter extends LlmAdapter {
    async *stream(options) {
      const current = activeCase;
      assert.ok(current, "unexpected generation outside a test case");
      assert.ok(options.tools.some(tool => tool.name === "structured_output"));
      const index = current.providerCalls.length;
      assert.ok(index <= current.maximum + 1, "unbounded local repair loop");
      const tool = index === 0 ? "skill" : "structured_output";
      // If the guard is absent, the next call succeeds so the regression fails
      // promptly on a real extra provider request, without an infinite loop.
      const argumentsValue = tool === "skill" ? {name: current.skillName}
        : index <= current.invalidAttempts ? {} : current.validOutput;
      current.providerCalls.push({index: index + 1, tool, invalid_schema: tool === "structured_output" && index <= current.invalidAttempts});
      const block = {type: "tool-call", id: `${current.name}-call-${index + 1}`,
        name: tool, arguments: JSON.stringify(argumentsValue)};
      yield {type: "block-start", index: 0, blockType: "tool-call"};
      yield {type: "tool-call-delta", index: 0, id: block.id, name: block.name, argumentsDelta: block.arguments};
      yield {type: "block-end", index: 0, block};
      yield {type: "usage", usage: {inputTokens: 10, outputTokens: 5}};
      yield {type: "finish", reason: "tool-calls"};
    }
  }
  ctx.llm.registerAdapter(["ecology-output-budget-test"], new LocalAdapter());
  const importPlugin = file => import(pathToFileURL(path.join(pkg, "lib/runtime", file)).href);
  const {RoleAgentManager} = await importPlugin("agents.js");
  const {NativeStageRunner, STAGES, prepareStageOutputSchema, jsonDigest} = await importPlugin("stage-runner.js");
  const {researchOutputTransport} = await importPlugin("research-contract.js");
  const {RESEARCH_EXECUTION_POLICY} = await importPlugin("research-execution-policy.js");
  const {structuredFailureCode} = await importPlugin("structured-stage-errors.js");
  const {toolResultIdentity} = await importPlugin("session-events.js");
  const manifest = JSON.parse(await readFile(path.join(pkg, "presets/preset-manifest.json"), "utf8"));
  manager = new RoleAgentManager(ctx);
  offEvents = ctx.on("session/event", (session, event) => {
    if (activeCase && session.header?.parentSession === activeCase.parentId) {
      activeCase.events.push(event);
      activeCase.childIds.add(session.id);
    }
  }, {global: true});

  for (const specification of [
    {name: "synthesis-two-invalid-stop", stage: "generation.research-synthesis", maximum: 2, invalidAttempts: 2, expected: "rejected"},
    {name: "synthesis-one-invalid-repaired", stage: "generation.research-synthesis", maximum: 2, invalidAttempts: 1, expected: "accepted"},
    {name: "reflection-three-invalid-stop", stage: "generation.reflect", maximum: 3, invalidAttempts: 3, expected: "rejected"},
    {name: "reflection-two-invalid-repaired", stage: "generation.reflect", maximum: 3, invalidAttempts: 2, expected: "accepted"},
  ]) {
    const contract = STAGES[specification.stage];
    const preset = manifest.presets.find(item => item.role === contract.role);
    const runId = `isolated-output-budget-${specification.name}`;
    const role = await manager.createRoleAgent({run_id: runId, role: contract.role,
      preset_id: preset.preset_id, model: "ecology-output-budget-test/fake", cwd: home});
    const state = activeCase = {...specification, skillName: contract.skillName,
      parentId: role.sessionId, providerCalls: [], events: [], childIds: new Set(), reservations: [], persisted: [], failures: []};
    const sidecar = {request: async (route, options) => {
      if (route.endsWith("/session-usage")) return {accepted: true};
      if (route.endsWith("/child-reservations")) {
        const launch = {reservation_id: `${state.name}-reservation-${state.reservations.length + 1}`,
          launch_attempt: state.reservations.length + 1};
        state.reservations.push(launch);
        return {accepted: true, admission_id: options.body.admission_id, timeout_ms: options.body.timeout_ms,
          launch, ledger_expected_revision: 2};
      }
      if (route.endsWith("/child-failures")) { state.failures.push(options.body); return {accepted: true}; }
      assert.ok(route.endsWith("/structured-results"), `unexpected sidecar route ${route}`);
      state.persisted.push(options.body);
      return {accepted: true, result_digest: options.body.result_digest};
    }};
    const runner = new NativeStageRunner(ctx, {roleAgents: manager,
      runRegistry: {get: () => ({status: "running"})}, sidecar,
      structuredStageMaxAttempts: 2, structuredStageTimeoutMs: 15000, researchStageTimeoutMs: 15000,
      structuredStageMinIntervalMs: 0});
    const context = specification.stage === "generation.research-synthesis" ? {
      research_execution_policy: {...RESEARCH_EXECUTION_POLICY},
      knowledge_snapshot: {evidence_catalog: [{knowledge_id: "fixture:paper"}]},
      required_candidate_direction_count: 4,
      synthesis_contract: {mutation_directions_by_axis: {scientific_parameter: ["increase", "decrease"]},
        allowed_mutation_targets: {scientific_parameter: ["ridge_alpha"]}},
    } : {};
    const transport = researchOutputTransport(specification.stage,
      prepareStageOutputSchema(specification.stage, await runner.schema(contract.file), context));
    state.validOutput = schemaFixture(transport.schema);
    if (specification.stage === "generation.research-synthesis") {
      state.validOutput.candidate_directions = Array.from({length: 4}, (_, index) => ({
        ...state.validOutput.candidate_directions[0], direction_id: `fixture-direction-${index + 1}`}));
    }
    const binding = {run_id: runId, stage: specification.stage, run_state_revision: 1,
      stage_attempt: 1, ledger_expected_revision: 1, admission_id: `admission-${state.name}`,
      idempotency_key: `idempotency-${state.name}`, request: {
        role: contract.role, output_schema_id: contract.schema, context,
        context_canonical_json: canonical(context), context_digest: jsonDigest(context),
        max_tokens: specification.stage === "generation.research-synthesis" ? 16384 : 8192,
        identity_digests: {genome_digest: "a".repeat(64), compiled_behavior_digest: "b".repeat(64),
          phenotype_instance_digest: "c".repeat(64)},
      }};
    let result, caught;
    try { result = await runner.run(binding); } catch (error) { caught = error; }
    // Leave the retained parent alive briefly after the runner settles: a
    // detached late generation must not be hidden by test-owned cleanup.
    await new Promise(resolve => setTimeout(resolve, 25));
    const structuredCalls = state.providerCalls.filter(call => call.tool === "structured_output");
    const outputCalls = state.events.filter(event => event.type === "tool/call" && event.data?.name === "structured_output");
    const outputCallIds = new Set(outputCalls.map(event => event.data.callId));
    const invalidResults = state.events.filter(event => event.type === "tool/result"
      && outputCallIds.has(toolResultIdentity(event)?.callId)
      && event.data?.error?.code === "INVALID_ARGS" && toolResultIdentity(event)?.isError === true);
    const nativeOutputEvidence = outputCalls.map(call => {
      const result = state.events.find(event => event.type === "tool/result" && toolResultIdentity(event)?.callId === call.data.callId);
      assert.ok(result, "every output attempt must have a native tool result");
      assert.deepEqual(result.sourceEventSeqs, [call.seq]);
      return {call_seq: call.seq, result_seq: result.seq, source_event_seqs: result.sourceEventSeqs,
        error_code: result.data?.error?.code || null, is_error: toolResultIdentity(result).isError};
    });
    const summary = {name: state.name, stage: state.stage, expected: state.expected,
      output_attempt_limit: state.maximum, provider_requests: state.providerCalls,
      structured_provider_requests: structuredCalls.length, mandatory_skill_provider_requests: 1,
      native_invalid_args_results: invalidResults.length, child_sessions: state.childIds.size,
      native_output_evidence: nativeOutputEvidence,
      durable_launches: state.reservations.length, persisted_results: state.persisted.length,
      failure_code: caught ? structuredFailureCode(caught) || caught.code || caught.name : null};
    report.cases.push(summary);
    console.log(JSON.stringify(summary));
    assert.equal(state.reservations.length, 1, "schema budget exhaustion must not retry with another child");
    assert.equal(state.childIds.size, 1);
    assert.equal(invalidResults.length, state.invalidAttempts);
    assert.equal(structuredCalls.length, state.expected === "rejected" ? state.maximum : state.invalidAttempts + 1,
      "a rejected final allowed schema attempt must stop before the next provider request");
    if (state.expected === "rejected") {
      assert.equal(summary.failure_code, "structured_output_attempt_limit_exceeded");
      assert.equal(state.persisted.length, 0);
      assert.equal(state.failures.length, 1);
    } else {
      assert.equal(caught, undefined, caught?.message);
      assert.deepEqual(result.structured, transport.bind(state.validOutput));
      assert.equal(state.persisted.length, 1);
      assert.equal(state.persisted[0].skill_invocation_evidence.order_verified, true);
      assert.equal(state.failures.length, 0);
    }
    await manager.quiesceRun(runId, {dispose: true});
    activeCase = null;
  }
  report.passed = true;
  console.log(`Harness ${installed.version}: synchronous schema-attempt stop and bounded successful repairs passed (local adapter).`);
} finally {
  offEvents?.();
  await manager?.dispose();
  await host?.shutdown.shutdown(0);
  process.chdir(root);
  await rm(home, {recursive: true, force: true});
  report.isolated_home_removed = true;
  if (evidencePath) {
    await mkdir(path.dirname(evidencePath), {recursive: true});
    await writeFile(evidencePath, JSON.stringify(report, null, 2) + "\n");
  }
}
