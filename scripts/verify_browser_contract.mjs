#!/usr/bin/env node
// Read-only visual review plus intercepted form-contract checks. All creation
// and preflight responses are local fixtures; no model or evolution is started.
// Requires Playwright, optionally supplied through NODE_PATH.
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {createRequire} from "node:module";
import {fileURLToPath} from "node:url";
import {captureFrontendSource} from "./browser_frontend_source.mjs";
const {chromium} = createRequire(import.meta.url)("playwright");
const args = process.argv.slice(2);
function option(name, fallback) { const i = args.indexOf(name); return i < 0 ? fallback : args[i + 1]; }
const baseUrl = option("--url", "http://127.0.0.1:8848/plugins/ecology/evolution/");
const outputDirectory = path.resolve(option("--output-dir", path.join(os.tmpdir(), "ecologyrsi-browser-contract-" + Date.now())));
const executablePath = option("--browser", undefined);
const selectionReadDelayMs = Number(option("--selection-read-delay-ms", "0"));
assert.ok(Number.isFinite(selectionReadDelayMs) && selectionReadDelayMs >= 0 && selectionReadDelayMs <= 20000,
  "--selection-read-delay-ms must be between 0 and 20000; it delays only the selected run's read responses");
const initialListReadDelayMs = Number(option("--initial-list-read-delay-ms", "0"));
assert.ok(Number.isFinite(initialListReadDelayMs) && initialListReadDelayMs >= 0 && initialListReadDelayMs <= 20000,
  "--initial-list-read-delay-ms must be between 0 and 20000; it delays only the first run-list response");
await fs.mkdir(outputDirectory, {recursive: true});
const browser = await chromium.launch({headless: true, ...(executablePath ? {executablePath} : {channel: "chrome"})});
const page = await browser.newPage({viewport: {width: 1480, height: 1000}});
const frontendProof = captureFrontendSource(page, baseUrl,
  path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../plugins/ecology_evolution"));
const result = {base_url: baseUrl, mode: "read_only_and_intercepted_fixtures", errors: [], views: [], form: null, steps: [], requests: []};
const startedAt = Date.now();
function step(name) {
  result.current_step = name;
  result.steps.push({step: name, elapsed_ms: Date.now() - startedAt});
}
const requestRecords = new WeakMap();
page.on("request", request => {
  const url = new URL(request.url());
  if (!url.pathname.startsWith("/api/ecology-evolution/") || result.requests.length >= 250) { return; }
  const record = {method: request.method(), path: url.pathname + url.search, started_ms: Date.now() - startedAt};
  requestRecords.set(request, record);
  result.requests.push(record);
});
page.on("response", response => {
  const record = requestRecords.get(response.request());
  if (record) { record.status = response.status(); }
});
page.on("requestfinished", request => {
  const record = requestRecords.get(request);
  if (record) { record.duration_ms = Date.now() - startedAt - record.started_ms; }
});
page.on("requestfailed", request => {
  const record = requestRecords.get(request);
  if (record) { record.duration_ms = Date.now() - startedAt - record.started_ms; record.failure = request.failure()?.errorText; }
});
let mockProjection = null;
let mockPreflightPasses = true;
let delayedSelectionRunId = null;
let initialListDelayApplied = false;
const intercepted = [];
page.on("pageerror", error => result.errors.push(error.message));
// Defense in depth: capacity is read-only; every other mutating request is
// either a known local fixture or rejected before reaching the service.
await page.route("**/api/ecology-evolution/**", async route => {
  const request = route.request();
  const pathname = new URL(request.url()).pathname;
  if (request.method() === "POST" && pathname.endsWith("/model-preflight")) {
    intercepted.push({path: "model-preflight", body: request.postDataJSON()});
    await new Promise(resolve => setTimeout(resolve, 500));
    return route.fulfill({status: 200, contentType: "application/json", body: JSON.stringify({passed: mockPreflightPasses, receipts: []})});
  }
  if (request.method() === "POST" && pathname.endsWith("/runs")) {
    const body = request.postDataJSON();
    intercepted.push({path: "runs", body});
    mockProjection = {id: "run:browser-local-fixture", run_id: "run:browser-local-fixture", status: "paused", generation: 0,
      total_generations: 1, candidates_count: 0, candidates: [], rounds: [], trajectory: [], budget: body.budget,
      configuration: {...body, policy_model_id: body.strategy_model_id, judge_model_id: body.review_model_id},
      dsh_runtime: {native: true},
      research_execution_policy: {schema_version: "ecologyrsi-dsh.research-execution-policy/1",
        synthesis_context_format: "compact@1", synthesis_report_format: "concise@2",
        synthesis_max_output_tokens: 16384, retry_identical_exhausted_request: false},
      search_guard_policy: body.search_guard_policy, require_model_contract_preflight: body.require_model_contract_preflight};
    return route.fulfill({status: 201, contentType: "application/json", body: JSON.stringify({projection: mockProjection})});
  }
  if (pathname.includes("run%3Abrowser-local-fixture") || pathname.includes("run:browser-local-fixture")) {
    return route.fulfill({status: 200, contentType: "application/json", body: JSON.stringify(pathname.endsWith("/events") ? {events: [], total_public_events: 0} : {projection: mockProjection})});
  }
  if (!["GET", "HEAD", "OPTIONS"].includes(request.method()) && !pathname.endsWith("/evolution-capacity")) {
    result.errors.push("Blocked unexpected mutation: " + request.method() + " " + pathname);
    return route.abort();
  }
  if (initialListReadDelayMs && !initialListDelayApplied && request.method() === "GET"
      && pathname === "/api/ecology-evolution/runs") {
    initialListDelayApplied = true;
    result.initial_list_read_delay_ms = initialListReadDelayMs;
    const response = await route.fetch();
    await new Promise(resolve => setTimeout(resolve, initialListReadDelayMs));
    return route.fulfill({response});
  }
  if (selectionReadDelayMs && delayedSelectionRunId && request.method() === "GET"
      && ["/api/ecology-evolution/runs/" + encodeURIComponent(delayedSelectionRunId),
        "/api/ecology-evolution/runs/" + encodeURIComponent(delayedSelectionRunId) + "/events"].includes(pathname)) {
    const response = await route.fetch();
    await new Promise(resolve => setTimeout(resolve, selectionReadDelayMs));
    return route.fulfill({response});
  }
  return route.continue();
});
try {
  step("open_workbench");
  await page.goto(baseUrl, {waitUntil: "networkidle", timeout: 30000});
  step("wait_for_catalog");
  await page.waitForFunction(() => ["ready", "empty"].includes(window.EcologyEvolutionPlugin?.getState().loadState));
  assert.equal(await page.locator("#prediction-model-id").count(), 0, "Prediction tools belong to the running Agent");
  const completedId = await page.evaluate(() => window.EcologyEvolutionPlugin.getState().runs.find(run => run.status === "completed")?.id);
  result.completed_run_id = completedId || null;
  if (completedId) {
    delayedSelectionRunId = completedId;
    result.selection_read_delay_ms = selectionReadDelayMs;
    step("select_completed_run");
    await page.selectOption("#run-select", completedId);
    step("wait_for_completed_run_projection");
    await page.waitForFunction(id => window.EcologyEvolutionPlugin.getState().activeRun?.id === id, completedId);
  }
  for (const width of [1480, 390]) {
    await page.setViewportSize({width, height: 1000});
    for (const workspace of ["settings", "parameters", "process", "candidates"]) {
      step(`inspect_${width}_${workspace}`);
      await page.click("#tab-" + workspace);
      await page.waitForTimeout(200);
      const layout = await page.evaluate(() => ({viewport_width: innerWidth, page_width: document.documentElement.scrollWidth,
        monitor_status: document.querySelector("#execution-monitor-status").textContent,
        active_status: window.EcologyEvolutionPlugin.getState().activeRun?.status,
        clipped_buttons: [...document.querySelectorAll("button")].filter(button => {
          if (!button.getClientRects().length) { return false; }
          for (let parent = button.parentElement; parent; parent = parent.parentElement) {
            if (["auto", "scroll"].includes(getComputedStyle(parent).overflowX) && parent.scrollWidth > parent.clientWidth) { return false; }
          }
          return true;
        })
          .filter(button => {const r = button.getBoundingClientRect(); return r.left < -2 || r.right > innerWidth + 2;}).map(button => button.id)}));
      result.views.push({workspace, width, ...layout});
      assert.ok(layout.page_width <= width + 2, `${workspace} overflow at ${width}: ${layout.page_width}`);
      assert.deepEqual(layout.clipped_buttons, [], `${workspace} has clipped buttons at ${width}`);
      if (workspace === "process" && completedId) {
        assert.equal(layout.active_status, "completed");
        assert.doesNotMatch(layout.monitor_status, /模型执行中|模型重试中|自动准备下一轮|后台排队中/);
      }
      await page.screenshot({path: path.join(outputDirectory, `${width}-${workspace}.png`), fullPage: true});
    }
  }
  await page.setViewportSize({width: 1480, height: 1000});
  step("configure_intercepted_form");
  await page.click("#tab-settings");
  await page.click("#tab-parameters");
  await page.fill("#formal-origin-count", "144");
  await page.fill("#local-batch-origin-count", "72");
  await page.click("#tab-settings");
  const modelChoices = await page.locator("#policy-model-id option").evaluateAll(options => options.filter(item => !item.disabled && item.value).map(item => item.value));
  assert.ok(modelChoices.length >= 2, "Two configured model roles are needed for form validation; they are never called by this test");
  await page.selectOption("#policy-model-id", modelChoices[0]);
  const reviewChoice = await page.locator("#judge-model-id option").evaluateAll((options, excluded) => options.find(item => !item.disabled && item.value && item.value !== excluded)?.value, modelChoices[0]);
  assert.ok(reviewChoice, "No separate configured review role");
  await page.selectOption("#judge-model-id", reviewChoice);
  step("wait_for_form_readiness");
  await page.waitForFunction(() => document.querySelector("#readiness-pill").textContent === "可以创建运行", undefined, {timeout: 30000});
  const validity = await page.locator("#start-form").evaluate(form => ({valid: form.checkValidity(), invalid: [...form.elements]
    .filter(item => item.willValidate && !item.validity.valid).map(item => ({id: item.id, message: item.validationMessage}))}));
  assert.equal(validity.valid, true, JSON.stringify(validity.invalid));
  step("submit_intercepted_preflight");
  await page.click("#start-button");
  await page.waitForFunction(() => document.querySelector("#process-status-pill").textContent === "模型能力预检中");
  assert.equal(await page.locator("#run-status").textContent(), "模型能力预检中");
  await page.screenshot({path: path.join(outputDirectory, "form-preflight.png"), fullPage: true});
  step("wait_for_intercepted_creation");
  await page.waitForFunction(() => window.EcologyEvolutionPlugin.getState().activeRun?.id === "run:browser-local-fixture");
  assert.deepEqual(intercepted.map(item => item.path), ["model-preflight", "runs"]);
  for (const item of intercepted) {
    assert.equal(item.body.prediction_selection_policy, "model_during_run@1");
    assert.equal(Object.hasOwn(item.body, "prediction_model_id"), false);
    assert.equal(Object.hasOwn(item.body, "evaluator_id"), false);
    assert.equal(item.body.search_guard_policy, "practical_delta_cell_noninferiority_paired_blocks@1");
    assert.equal(item.body.require_model_contract_preflight, true);
  }
  result.form = {validity, request_order: intercepted.map(item => item.path), agent_prediction_selection_preserved: true};
  step("inspect_frozen_research_policy");
  await page.click("#tab-process");
  await page.locator("details.process-settings > summary").click();
  await page.locator("#toast").waitFor({state: "hidden", timeout: 10000});
  const frozenResearchSummary = await page.locator("#process-summary").textContent();
  assert.match(frozenResearchSummary, /综合单次输出上限 16,384 tokens/);
  assert.match(frozenResearchSummary, /相同预算耗尽请求不重试/);
  assert.match(frozenResearchSummary, /各阶段输出遵循冻结执行约束/);
  result.research_policy_fixture_views = [];
  for (const width of [1480, 390]) {
    await page.setViewportSize({width, height: 1000});
    const pageWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    assert.ok(pageWidth <= width + 2, `Frozen research policy overflow at ${width}: ${pageWidth}`);
    await page.screenshot({path: path.join(outputDirectory, `${width}-research-policy-fixture.png`), fullPage: true});
    await page.locator("#process-summary .dataset-stat").filter({hasText: "研究执行约束"}).screenshot({path: path.join(outputDirectory, `${width}-research-policy-row.png`)});
    result.research_policy_fixture_views.push({width, page_width: pageWidth});
  }
  await page.setViewportSize({width: 1480, height: 1000});
  mockPreflightPasses = false;
  step("verify_failed_preflight_blocks_creation");
  await page.click("#tab-settings");
  await page.click("#start-button");
  await page.waitForFunction(() => document.querySelector("#system-title").textContent === "操作未完成");
  assert.deepEqual(intercepted.map(item => item.path), ["model-preflight", "runs", "model-preflight"]);
  result.form.failed_preflight_did_not_create = true;
  assert.deepEqual(result.errors, []);
  result.passed = true;
  step("complete");
} catch (error) {
  result.passed = false;
  result.failure = error.message;
  result.failure_state = await page.evaluate(() => {
    const current = window.EcologyEvolutionPlugin?.getState();
    return {load_state: current?.loadState, active_run_id: current?.activeRun?.id, active_status: current?.activeRun?.status,
      selected_run_id: document.querySelector("#run-select")?.value,
      run_select_disabled: document.querySelector("#run-select")?.disabled,
      system_title: document.querySelector("#system-title")?.textContent,
      system_message: document.querySelector("#system-message")?.textContent,
      toast: document.querySelector("#toast")?.textContent};
  }).catch(() => null);
  result.overflow_elements = await page.evaluate(() => [...document.querySelectorAll("body *")]
    .filter(element => {const box = element.getBoundingClientRect(); return box.width > 0 && box.right > innerWidth + 2;})
    .map(element => ({tag: element.tagName, id: element.id, class: element.className, width: element.getBoundingClientRect().width, right: element.getBoundingClientRect().right}))
    .slice(0, 40)).catch(() => []);
  await page.screenshot({path: path.join(outputDirectory, "failure.png"), fullPage: true}).catch(() => {});
  process.exitCode = 1;
} finally {
  try { Object.assign(result, await frontendProof()); }
  catch (error) { result.passed = false; result.errors.push(error.message); process.exitCode = 1; }
  await fs.writeFile(path.join(outputDirectory, "result.json"), JSON.stringify(result, null, 2) + "\n");
  await browser.close();
}
console.log(JSON.stringify({passed: result.passed, report: path.join(outputDirectory, "result.json"), failure: result.failure}));
