"use strict";

const token = new URLSearchParams(location.hash.slice(1)).get("token");
const state = { catalog: null, runs: [], selected: null, timer: null };
const $ = (selector) => document.querySelector(selector);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
function num(value, digits = 4, signed = true) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  const text = Number(value).toFixed(digits);
  return signed && value > 0 ? `+${text}` : text;
}
function skillClass(value) { return value > 0 ? "good" : value < 0 ? "bad" : "muted"; }

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  if (options.method === "POST") headers["X-Ecogym"] = "1";
  const response = await fetch(`/api/${path}`, { ...options, headers });
  const type = response.headers.get("Content-Type") || "";
  const body = type.includes("json") ? await response.json() : await response.text();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

function changes(edit) {
  return Object.entries(edit.changes || {}).map(([path, value]) => {
    const shown = typeof value === "number" ? Number(value).toPrecision(4) : JSON.stringify(value).slice(0, 120);
    return `<span class="tag">${esc(path)} = ${esc(shown)}</span>`;
  }).join("") || "–";
}

function renderRuns() {
  $("#runs").innerHTML = state.runs.map((run) => {
    const result = run.result || {};
    const score = result.incumbent_score !== undefined ? `${num(result.seed_score)} → ${num(result.incumbent_score)}` : "";
    const status = run.active ? "running" : run.status;
    return `<li data-run="${esc(run.run_id)}" class="${run.run_id === state.selected ? "selected" : ""}">
      <div class="title">${esc(run.label || run.run_id)}</div>
      <div class="meta"><span class="badge ${esc(status)}">${esc(status)}</span> ${esc(run.proposer?.name)} · ${esc(score)}</div>
      <div class="meta">${esc(run.started_at?.slice(0, 16).replace("T", " "))} · ${esc(run.task_id)}</div></li>`;
  }).join("") || '<li class="muted">还没有运行</li>';
  document.querySelectorAll("#runs li[data-run]").forEach((item) => {
    item.addEventListener("click", () => select(item.dataset.run));
  });
}

function noteHtml(note) {
  const list = (title, items) => items && items.length
    ? `<b>${title}</b><ul>${items.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>` : "";
  return `<div class="note">${list("观察", note.observations)}${list("假设", note.hypotheses)}${list("方向", note.directions)}</div>`;
}

function generationHtml(generation) {
  const cohorts = generation.cohorts;
  const rows = (generation.screened?.candidates || []).map((item) => `<tr><td>${changes(item.edit)}</td>
      <td class="${skillClass(item.delta)}">${item.error ? '<span class="bad">失败</span>' : num(item.delta)}</td>
      <td>${esc(item.error || item.edit.rationale || "")}</td></tr>`);
  for (const item of generation.rejected || []) {
    rows.push(`<tr><td>${changes(item.edit)}</td><td class="bad">拒绝</td><td>${esc(item.problems.join("; "))}</td></tr>`);
  }
  for (const item of generation.dropped || []) {
    rows.push(`<tr><td>${changes(item)}</td><td class="muted">评审剔除</td><td>${esc(item.reason)}</td></tr>`);
  }
  let verdict = '<p class="muted">没有候选在初筛中超过在任方法，本代不做留出比较。</p>';
  if (generation.comparison) {
    const d = generation.comparison.decision;
    const failed = d.cells.filter((cell) => !cell.passed).map((cell) => `${cell.cell}（${cell.reason}）`);
    verdict = `<p>留出比较：Δ ${num(d.delta)}，95% 区间 [${num(d.interval[0])}, ${num(d.interval[1])}]，${d.blocks} 天 →
      <b class="${d.promoted ? "good" : "bad"}">${d.promoted ? "晋级" : esc(d.reason)}</b>
      ${failed.length ? `<br><span class="muted">未通过的格：${esc(failed.join("，"))}</span>` : ""}</p>`;
  }
  const errors = (generation.agent_errors || []).map((error) => `<p class="bad">智能体错误：${esc(error)}</p>`).join("");
  return `<div class="card"><h3>第 ${generation.generation} 代 ${generation.promoted ? '<span class="badge finished">已晋级</span>' : ""}</h3>
    <p class="muted">初筛 ${cohorts.screening.origins} 个起点 / ${cohorts.screening.blocks.length} 天；留出 ${cohorts.holdout.origins} 个起点 / ${cohorts.holdout.blocks.length} 天</p>
    ${(generation.notes || []).map(noteHtml).join("")}${errors}
    <table><thead><tr><th>修改</th><th>初筛 Δ</th><th>理由 / 问题</th></tr></thead><tbody>${rows.join("") || '<tr><td colspan="3" class="muted">没有候选</td></tr>'}</tbody></table>
    ${verdict}</div>`;
}

function cellsHtml(cells) {
  if (!cells) return "";
  const stage = "model_selection";
  const seed = cells.seed[stage].cells;
  const final = cells.final ? cells.final[stage].cells : null;
  const rows = seed.map((row, index) => {
    const after = final ? final[index] : null;
    return `<tr><td>${esc(row.cell)}</td><td>${num(row.baseline_rmse, 3, false)}</td>
      <td>${num(row.rmse, 3, false)} <span class="${skillClass(row.skill)}">(${num(row.skill, 3)})</span></td>
      <td>${after ? `${num(after.rmse, 3, false)} <span class="${skillClass(after.skill)}">(${num(after.skill, 3)})</span>` : "–"}</td></tr>`;
  });
  const line = (name) => cells[name] ? `${name === "seed" ? "种子" : "最终"}：model_selection ${num(cells[name].model_selection.score)}，calibration_uq ${num(cells[name].calibration_uq.score)}` : "";
  return `<div class="card"><h3>逐格结果（model_selection，括号内为技能）</h3>
    <table><thead><tr><th>格</th><th>朴素基线 RMSE</th><th>种子 RMSE</th><th>最终 RMSE</th></tr></thead><tbody>${rows.join("")}</tbody></table>
    <p class="muted">${esc(line("seed"))}<br>${esc(line("final"))}</p></div>`;
}

function sealedHtml(run) {
  const done = new Set(run.sealed.map((item) => item.stage));
  const results = run.sealed.map((item) => `<p><b>${esc(item.stage)}</b>：总技能 <b class="${skillClass(item.score)}">${num(item.score)}</b>，
      95% 区间 [${num(item.interval[0])}, ${num(item.interval[1])}]，${item.origins} 个起点；该分区累计评过 ${item.methods_exposed_on_stage} 个方法</p>`).join("");
  const finished = run.status === "finished";
  return `<div class="card"><h3>封存评测</h3>
    <p class="muted">每个方法在 validation 和 final_test 上各只能评一次，final_test 须在 validation 之后。评测结果会写入账本。</p>
    ${results}
    <p class="actions"><button data-seal="validation" ${!finished || done.has("validation") ? "disabled" : ""}>评测 validation</button>
    <button data-seal="final_test" ${!finished || !done.has("validation") || done.has("final_test") ? "disabled" : ""}>评测 final_test</button>
    <span id="seal-error" class="bad"></span></p></div>`;
}

async function renderRun(run) {
  const result = run.result || {};
  const options = run.proposer?.options || {};
  const config = run.config;
  const status = run.active ? "running" : run.status;
  let cells = null;
  if (!run.active) {
    try { cells = await api(`runs/${encodeURIComponent(run.run_id)}/cells`); } catch (error) { cells = null; }
  }
  $("#detail").innerHTML = `<div class="card"><h2>${esc(run.label || run.run_id)} <span class="badge ${esc(status)}">${esc(status)}</span></h2>
    <p class="muted">${esc(run.run_id)} · ${esc(run.task_id)} · 提案者 ${esc(run.proposer?.name)} ${options.model ? `(${esc(options.model)})` : ""} · ${esc(run.started_at)}</p>
    <div class="summary"><div>种子得分<b>${num(result.seed_score)}</b></div><div>最终得分<b>${num(result.incumbent_score)}</b></div>
      <div>晋级次数<b>${esc(result.promotions ?? "–")}</b></div><div>完成代数<b>${run.generations.length} / ${config.generations}</b></div>
      <div>LLM 调用<b>${run.agent_exchanges}</b></div><div>用时<b>${esc(result.seconds ?? "–")} s</b></div></div>
    ${run.error ? `<p class="bad">${esc(run.error)}</p>` : ""}
    <p class="actions">${run.active ? '<button id="stop">本代结束后停止</button>' : ""}<button id="report">HTML 报告</button></p></div>
    ${cellsHtml(cells)}${sealedHtml(run)}${run.generations.slice().reverse().map(generationHtml).join("")}`;
  $("#stop")?.addEventListener("click", async () => { await api(`runs/${encodeURIComponent(run.run_id)}/stop`, { method: "POST", body: "{}" }); });
  $("#report").addEventListener("click", async () => {
    const html = await api(`runs/${encodeURIComponent(run.run_id)}/report`);
    window.open(URL.createObjectURL(new Blob([html], { type: "text/html" })), "_blank");
  });
  document.querySelectorAll("[data-seal]").forEach((button) => button.addEventListener("click", async () => {
    const stage = button.dataset.seal;
    if (!confirm(`对最终方法做一次 ${stage} 封存评测？同一方法在该分区上只能评一次，不可撤销。`)) return;
    button.disabled = true;
    try { await api(`runs/${encodeURIComponent(run.run_id)}/seal`, { method: "POST", body: JSON.stringify({ stage }) }); await select(run.run_id); }
    catch (error) { $("#seal-error").textContent = error.message; button.disabled = false; }
  }));
}

async function select(runId) {
  state.selected = runId;
  clearTimeout(state.timer);
  renderRuns();
  const run = await api(`runs/${encodeURIComponent(runId)}`);
  await renderRun(run);
  if (run.active) state.timer = setTimeout(() => select(runId), 4000);
}

async function refreshRuns() {
  state.runs = await api("runs");
  renderRuns();
}

function fillForm() {
  const form = $("#new-run");
  form.task.innerHTML = state.catalog.tasks.map((task) => `<option value="${esc(task.id)}">${esc(task.title)}</option>`).join("");
  const presets = () => {
    const task = state.catalog.tasks.find((item) => item.id === form.task.value);
    form.preset.innerHTML = Object.entries(task?.presets || {}).map(([name, note]) => `<option value="${esc(name)}">${esc(name)} — ${esc(note)}</option>`).join("");
  };
  form.task.addEventListener("change", presets);
  presets();
  form.proposer.innerHTML = state.catalog.proposers.map((name) => `<option>${esc(name)}</option>`).join("");
  form.proposer.addEventListener("change", () => { $("#llm-options").hidden = form.proposer.value !== "llm"; });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    $("#new-run-error").textContent = "";
    const body = {
      task: form.task.value, preset: form.preset.value, proposer: form.proposer.value, label: form.label.value,
      generations: Number(form.generations.value), candidates: Number(form.candidates.value), seed: Number(form.seed.value),
    };
    if (body.proposer === "llm") {
      body.options = { transport: form.transport.value, model: form.model.value, key_env: form.key_env.value };
      if (form.base_url.value) body.options.base_url = form.base_url.value;
    }
    try {
      const { run_id: runId } = await api("runs", { method: "POST", body: JSON.stringify(body) });
      form.hidden = true;
      await refreshRuns();
      await select(runId);
    } catch (error) { $("#new-run-error").textContent = error.message; }
  });
}

async function main() {
  state.catalog = await api("catalog");
  fillForm();
  $("#new-run-toggle").addEventListener("click", () => { $("#new-run").hidden = !$("#new-run").hidden; });
  await refreshRuns();
  setInterval(refreshRuns, 6000);
  if (state.runs.length) await select(state.runs[0].run_id);
}

main().catch((error) => { $("#detail").innerHTML = `<p class="bad">${esc(error.message)}</p>`; });
