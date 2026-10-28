"use strict";

/* ecogym workbench: a run list, one page per run, and a form to start runs.
   Routes live in the URL hash (#run=<id>, #new). Every value that comes from the server is escaped. */

const SEALED = ["validation", "final_test"];
const TOKEN_KEY = "ecogym.token";
const SAVED_KEY = "ecogym.new-run";
const MINUS = "−";

const STATUS = {
  finished: ["已完成", "ok"],
  running: ["运行中", "live"],
  starting: ["启动中", "live"],
  failed: ["失败", "bad"],
  stopped: ["已停止", "warn"],
  interrupted: ["已中断", "warn"],
};
const PROPOSERS = {
  sweep: ["坐标扫描", "逐个坐标上下移动一个步长；晋级后从新的在任方法重新开始。"],
  random: ["随机搜索", "在信赖域内随机改动若干坐标，由运行的随机种子决定。"],
  llm: ["LLM 研究智能体", "研究员用只读工具分析开发数据，提案者把研究笔记变成修改，评审剔除理由不足的修改。"],
};
const ROLES = { researcher: "研究员", proposer: "提案者", reviewer: "评审" };
const CELL_REASONS = {
  better: ["更好", "ok"],
  within_uncertainty: ["在不确定范围内", "flat"],
  insufficient_evidence: ["证据不足", "warn"],
  regression_above_cap: ["退化超过上限", "bad"],
  credible_regression: ["可信退化", "bad"],
  no_paired_forecasts: ["没有成对预测", "warn"],
};
const STAGES = {
  model_selection: ["开发 · model_selection", "竞技循环在这里比较和选择方法"],
  calibration_uq: ["复核 · calibration_uq", "不参与任何选择的开发期复核"],
  validation: ["validation", "封存分区，最终方法只评一次"],
  final_test: ["final_test", "封存分区，须在 validation 之后，只评一次"],
};
const UNITS = { degC: "°C", percent: "%" };

const state = {
  catalog: null,
  tasks: new Map(),
  runs: [],
  runsJson: "",
  filter: "",
  route: { kind: "home" },
  run: null,
  signature: null,
  cells: null,
  cellsKey: null,
  cellsError: null,
  stage: "model_selection",
  seq: 0,
  timer: null,
  open: new Set(),
  closed: new Set(),
  stopping: new Set(),
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const main = () => $("#main");

/* ---- formatting --------------------------------------------------------- */

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
const isNum = (value) => typeof value === "number" && Number.isFinite(value);
function signed(value, digits = 4) {
  if (!isNum(value)) return "–";
  const text = Math.abs(value).toFixed(digits);
  if (Number(text) === 0) return (0).toFixed(digits);
  return (value > 0 ? "+" : MINUS) + text;
}
function plain(value, digits = 3) {
  if (!isNum(value)) return "–";
  return (value < 0 ? MINUS : "") + Math.abs(value).toFixed(digits);
}
function toneOf(value) {
  if (!isNum(value) || Math.abs(value) < 5e-5) return "flat";
  return value > 0 ? "good" : "bad";
}
function tint(value, scale) {
  if (!isNum(value) || value === 0) return "";
  const alpha = (0.05 + Math.min(Math.abs(value) / scale, 1) * 0.22).toFixed(3);
  return value > 0 ? `rgba(17,128,79,${alpha})` : `rgba(194,54,43,${alpha})`;
}
function short(value) {
  if (typeof value === "number") return String(Number(value.toPrecision(4)));
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}
const pad = (n) => String(n).padStart(2, "0");
function parseTime(iso) {
  const time = Date.parse(iso ?? "");
  return Number.isNaN(time) ? null : time;
}
function when(iso, { year = false, seconds = false } = {}) {
  const time = parseTime(iso);
  if (time === null) return "–";
  const d = new Date(time);
  const date = `${year ? `${d.getFullYear()}-` : ""}${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  return `${date} ${pad(d.getHours())}:${pad(d.getMinutes())}${seconds ? `:${pad(d.getSeconds())}` : ""}`;
}
function duration(seconds) {
  if (!isNum(seconds) || seconds < 0) return "–";
  if (seconds < 1) return "不到 1 秒";
  const s = Math.round(seconds);
  if (s < 60) return `${s} 秒`;
  const m = Math.floor(s / 60);
  if (m < 60) return s % 60 ? `${m} 分 ${s % 60} 秒` : `${m} 分`;
  return m % 60 ? `${Math.floor(m / 60)} 小时 ${m % 60} 分` : `${Math.floor(m / 60)} 小时`;
}
function runSeconds(run) {
  if (isNum(run.result?.seconds)) return run.result.seconds;
  const start = parseTime(run.started_at);
  const end = run.active ? Date.now() : parseTime(run.finished_at || run.updated_at);
  return start !== null && end !== null ? (end - start) / 1000 : null;
}

/* ---- tasks, cells, status ---------------------------------------------- */

const taskInfo = (id) => state.tasks.get(id) || null;
function splitCell(key) {
  const at = key.lastIndexOf("@");
  return [key.slice(0, at), Number(key.slice(at + 1))];
}
function horizonText(task, h) {
  const match = /^(\d+)(min|h|d)$/.exec(task?.step || "");
  if (!match) return `${h} 步`;
  const n = Number(match[1]) * h;
  return match[2] === "d" ? `${n} 天` : `${n} ${match[2]}`;
}
function targetOf(task, name) {
  return task?.targets.find((target) => target.name === name) || { name, label: "", unit: "" };
}
function cellLabel(task, key) {
  const [name, h] = splitCell(key);
  return `${targetOf(task, name).label || name} ${horizonText(task, h)}`;
}
function gridOf(task, keys) {
  if (task) return { targets: task.targets.map((target) => target.name), horizons: task.horizons };
  const targets = [];
  const horizons = [];
  for (const key of keys) {
    const [name, h] = splitCell(key);
    if (!targets.includes(name)) targets.push(name);
    if (!horizons.includes(h)) horizons.push(h);
  }
  return { targets, horizons: horizons.sort((a, b) => a - b) };
}
function statusOf(run) {
  if (run.active) return run.status === "starting" ? "starting" : "running";
  if (run.status === "running" || run.status === "starting") return "interrupted";
  if (run.status === "failed" && /^RunStopped\b/.test(run.error || "")) return "stopped";
  return STATUS[run.status] ? run.status : "failed";
}
function badge(text, tone, title = "") {
  return `<span class="badge tone-${tone}"${title ? ` title="${esc(title)}"` : ""}>${esc(text)}</span>`;
}
function statusBadge(key) {
  return badge(STATUS[key][0], STATUS[key][1]);
}
function proposerText(proposer) {
  const name = PROPOSERS[proposer?.name]?.[0] ?? proposer?.name ?? "–";
  const model = proposer?.model ?? proposer?.options?.model;
  return model ? `${name} · ${model}` : name;
}
function decisionText(reason, task) {
  if (!reason) return "";
  if (reason === "improved") return "通过全部门槛";
  if (reason === "below_practical_delta") {
    return `提升没有超过实用门槛${isNum(task?.gates?.min_delta) ? ` ${signed(task.gates.min_delta, 3)}` : ""}`;
  }
  if (reason === "insufficient_evidence") return "证据不足：留出天数太少";
  if (reason === "interval_crosses_zero") return "提升的置信区间包含 0";
  if (reason.startsWith("cell_regression:")) {
    return `有格可信退化：${reason.slice("cell_regression:".length).split(",").map((key) => cellLabel(task, key)).join("、")}`;
  }
  return reason;
}
function sealedReports(run) {
  const reports = {};
  for (const stage of SEALED) {
    const entry = run.sealing?.[stage];
    reports[stage] = entry?.report ?? (run.sealed || []).find((item) => item.stage === stage) ?? null;
  }
  return reports;
}

/* ---- API and small UI services ----------------------------------------- */

async function api(path, { method = "GET", body } = {}) {
  const headers = {};
  const token = sessionStorage.getItem(TOKEN_KEY);
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (method === "POST") headers["X-Ecogym"] = "1";
  const response = await fetch(`/api/${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const type = response.headers.get("Content-Type") || "";
  const data = type.includes("json") ? await response.json() : await response.text();
  if (!response.ok) {
    const message = (data && data.error) || `${response.status} ${response.statusText}`;
    const error = new Error(response.status === 401 ? `${message}。在地址末尾加上 #token=<令牌> 后回车。` : message);
    error.status = response.status;
    throw error;
  }
  return data;
}

/* A token in the hash (#token=...) is kept for this tab only and removed from the address. */
function takeToken() {
  const params = new URLSearchParams(location.hash.slice(1));
  const token = params.get("token");
  if (!token) return false;
  sessionStorage.setItem(TOKEN_KEY, token);
  params.delete("token");
  const rest = params.toString().replace(/=$/, "");
  history.replaceState(null, "", `${location.pathname}${location.search}${rest ? `#${rest}` : ""}`);
  return true;
}
const runPath = (id, suffix = "") => `runs/${encodeURIComponent(id)}${suffix}`;
const runHref = (id) => `#run=${encodeURIComponent(id)}`;

function toast(message, kind = "") {
  const item = document.createElement("div");
  item.className = `toast ${kind}`;
  item.textContent = message;
  $("#toasts").append(item);
  setTimeout(() => {
    item.classList.add("leaving");
    setTimeout(() => item.remove(), 350);
  }, kind === "bad" ? 8000 : 4000);
}

function details(key, fallback) {
  if (state.open.has(key)) return true;
  if (state.closed.has(key)) return false;
  return fallback;
}
document.addEventListener("toggle", (event) => {
  const element = event.target;
  if (!(element instanceof HTMLDetailsElement) || !element.dataset.key) return;
  const key = element.dataset.key;
  (element.open ? state.open : state.closed).add(key);
  (element.open ? state.closed : state.open).delete(key);
}, true);

/* ---- routing ------------------------------------------------------------ */

function readRoute() {
  const params = new URLSearchParams(location.hash.slice(1));
  if (params.has("new")) return { kind: "new" };
  if (params.get("run")) return { kind: "run", id: params.get("run") };
  return { kind: "home" };
}
function go(route, { replace = false } = {}) {
  const hash = route.kind === "new" ? "#new" : route.kind === "run" ? runHref(route.id) : "";
  if (replace) {
    history.replaceState(null, "", `${location.pathname}${location.search}${hash}`);
    onRoute();
  } else if (location.hash !== hash) {
    location.hash = hash;
  } else {
    onRoute();
  }
}
function onRoute() {
  takeToken();
  const route = readRoute();
  state.route = route;
  state.seq += 1;
  clearTimeout(state.timer);
  renderRunList();
  if (route.kind === "new") return showNewRun();
  if (route.kind === "run") return showRun(route.id);
  if (state.runs.length) return go({ kind: "run", id: state.runs[0].run_id }, { replace: true });
  return showEmpty();
}

/* ---- run list ----------------------------------------------------------- */

async function refreshRuns() {
  let runs;
  try {
    runs = await api("runs");
  } catch {
    return;
  }
  const json = JSON.stringify(runs);
  if (json === state.runsJson) return;
  state.runs = runs;
  state.runsJson = json;
  renderRunList();
}

function searchText(run) {
  return [run.label, run.run_id, run.task_id, taskInfo(run.task_id)?.title, run.proposer?.name,
    PROPOSERS[run.proposer?.name]?.[0], run.proposer?.model, STATUS[statusOf(run)][0]]
    .filter(Boolean).join(" ").toLowerCase();
}

function runItemHtml(run, current) {
  const key = statusOf(run);
  const result = run.result;
  let progress;
  if (result) {
    progress = `<span class="num">${signed(result.seed_score)} → ${signed(result.incumbent_score)}</span>`;
    progress += result.promotions ? ` · <span class="good">晋级 ${result.promotions} 次</span>` : " · 未晋级";
  } else if (run.active) {
    progress = `第 ${run.generations_done} / ${run.generations ?? "?"} 代`;
  } else {
    progress = `完成 ${run.generations_finished ?? 0} / ${run.generations ?? "?"} 代`;
  }
  if (run.sealed?.length) progress += ` · 封存 ${run.sealed.length}/${SEALED.length}`;
  const task = taskInfo(run.task_id);
  return `<li><a class="run-item" data-tone="${STATUS[key][1]}" href="${esc(runHref(run.run_id))}"${current ? ' aria-current="true"' : ""}>
    <span class="run-item-title">${esc(run.label || run.run_id)}</span>
    <span class="run-item-line">${statusBadge(key)}<span class="ellipsis">${esc(proposerText(run.proposer))}</span></span>
    <span class="run-item-line">${progress}</span>
    <span class="run-item-line faint"><span class="ellipsis" title="${esc(run.task_id)}">${when(run.started_at)} · ${duration(runSeconds(run))} · ${esc(task?.title || run.task_id || "")}</span></span>
  </a></li>`;
}

function renderRunList() {
  const filter = state.filter.trim().toLowerCase();
  const runs = state.runs.filter((run) => !filter || searchText(run).includes(filter));
  const current = state.route.kind === "run" ? state.route.id : null;
  $("#run-count").textContent = state.runs.length ? String(state.runs.length) : "";
  $("#runs").innerHTML = runs.map((run) => runItemHtml(run, run.run_id === current)).join("")
    || `<li class="empty-list">${state.runs.length ? "没有符合筛选条件的运行" : "还没有运行"}</li>`;
}

/* ---- run page: loading -------------------------------------------------- */

async function showRun(id) {
  const seq = state.seq;
  state.run = null;
  state.signature = null;
  if (state.cellsKey && !state.cellsKey.startsWith(`${id}|`)) {
    state.cells = null;
    state.cellsKey = null;
    state.cellsError = null;
    state.stage = "model_selection";
  }
  main().innerHTML = '<div class="placeholder">正在读取运行…</div>';
  main().scrollTop = 0;
  await refreshRun(id, seq, false);
}

function schedule(id, seq) {
  clearTimeout(state.timer);
  state.timer = setTimeout(() => {
    if (seq === state.seq) refreshRun(id, seq, true);
  }, 3000);
}

async function refreshRun(id, seq, quiet) {
  let run;
  try {
    run = await api(runPath(id));
  } catch (error) {
    if (seq !== state.seq) return;
    if (quiet) return schedule(id, seq);
    main().innerHTML = `<div class="alert bad"><b>无法读取运行 ${esc(id)}</b><div class="alert-body">${esc(error.message)}</div></div>`;
    return;
  }
  if (seq !== state.seq) return;
  const before = state.run;
  const signature = JSON.stringify([run.status, run.active, run.updated_at, run.agent_exchanges, run.sealed?.length, run.sealing]);
  const changed = signature !== state.signature;
  state.run = run;
  state.signature = signature;
  if (before?.active && !run.active) {
    const key = statusOf(run);
    toast(`运行${STATUS[key][0]}：${run.label || run.run_id}`, key === "finished" ? "ok" : "bad");
    refreshRuns();
  }
  if (!run.active && run.task_id) {
    const cellsKey = `${run.run_id}|${run.status}|${run.result?.incumbent_digest ?? ""}`;
    if (state.cellsKey !== cellsKey) {
      state.cellsKey = cellsKey;
      state.cells = null;
      state.cellsError = null;
      renderRun(quiet);
      try {
        const cells = await api(runPath(id, "/cells"));
        if (state.cellsKey === cellsKey) state.cells = cells;
      } catch (error) {
        if (state.cellsKey === cellsKey) state.cellsError = error.message;
      }
      if (seq === state.seq) renderRun(true);
      return;
    }
  }
  if (changed) renderRun(quiet);
  if (run.active) schedule(id, seq);
}

function renderRun(preserve) {
  const element = main();
  const top = element.scrollTop;
  const windowTop = window.scrollY;
  element.innerHTML = runHtml(state.run);
  if (preserve) {
    element.scrollTop = top;
    window.scrollTo(0, windowTop);
  } else {
    element.scrollTop = 0;
  }
}

/* ---- run page: sections -------------------------------------------------- */

function runHtml(run) {
  const task = taskInfo(run.task_id);
  const key = statusOf(run);
  return `<article class="run">${headHtml(run, task, key)}${statsHtml(run, task, key)}${forestHtml(run, task)}
    ${cellsHtml(run, task)}${sealHtml(run, task, key)}${generationsHtml(run, task)}${configHtml(run)}</article>`;
}

function headHtml(run, task, key) {
  const stopping = state.stopping.has(run.run_id);
  const meta = [
    ["任务", `<span title="${esc(run.task_id)}">${esc(task?.title || run.task_id || "–")}</span>`],
    ["提案者", esc(proposerText(run.proposer))],
    ["开始", `<span class="num">${when(run.started_at, { year: true })}</span>`],
    ["运行", `<span class="mono">${esc(run.run_id)}</span>`],
  ];
  if (run.ecogym) meta.push(["版本", `ecogym ${esc(run.ecogym)}`]);
  let notice = "";
  if (key === "failed") {
    notice = `<div class="alert bad"><b>运行失败</b><div class="alert-body">${esc(run.error || "没有记录原因")}</div></div>`;
  } else if (key === "stopped") {
    notice = `<div class="alert warn"><b>已按请求停止</b>：已完成的代保留在账本中；停止的运行不能做封存评测。</div>`;
  } else if (key === "interrupted") {
    notice = `<div class="alert warn"><b>运行被中断</b>：服务在运行结束前退出，账本中没有结束记录。已完成的代仍可查看，这条运行不能继续，也不能做封存评测。</div>`;
  }
  return `<header class="run-head">
    <div class="run-title"><h1>${esc(run.label || run.run_id)}</h1>${statusBadge(key)}</div>
    <dl class="meta">${meta.map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join("")}</dl>
    <div class="run-actions">
      ${run.active ? `<button class="warn" data-action="stop"${stopping ? " disabled" : ""}>${stopping ? "正在停止…" : "本代结束后停止"}</button>` : ""}
      ${run.task_id ? '<button data-action="report">HTML 报告</button>' : ""}
      <button data-action="copy">复制运行 ID</button>
    </div>
    ${notice}
  </header>`;
}

function stat(label, value, sub = "", extra = "") {
  return `<div class="stat"><div class="stat-label">${label}</div><div class="stat-value">${value}</div>
    ${sub ? `<div class="stat-sub">${sub}</div>` : ""}${extra}</div>`;
}

function statsHtml(run, task, key) {
  if (!run.config) return "";
  const result = run.result;
  const cells = state.cells;
  const seed = result?.seed_score ?? cells?.seed?.model_selection?.score;
  const tiles = [];
  if (run.active) {
    // scores of the whole stage are computed when the run ends
  } else if (result) {
    const change = result.incumbent_score - result.seed_score;
    tiles.push(stat("最终方法得分", `<span class="${toneOf(result.incumbent_score)}">${signed(result.incumbent_score)}</span>`,
      `种子 ${signed(seed)} · 变化 <span class="${toneOf(change)}">${signed(change)}</span>`));
  } else {
    tiles.push(stat("种子方法得分", signed(seed), "运行没有完成，没有最终方法"));
  }
  const review = cells?.final ?? cells?.seed;
  if (review) {
    tiles.push(stat("复核期得分", `<span class="${toneOf(review.calibration_uq.score)}">${signed(review.calibration_uq.score)}</span>`,
      `calibration_uq · ${cells.final ? "最终方法" : "种子方法"}，不参与选择`));
  }
  const total = run.config.generations;
  const finished = run.generations.filter((g) => "incumbent_after" in g).length;
  if (run.active) {
    const share = Math.min(finished / total, 1) * 100;
    tiles.push(stat("进度", `${Math.min(run.generations.length, total)} / ${total} 代`, `已完成 ${finished} 代`,
      `<div class="progress"><i style="width:${share.toFixed(1)}%"></i></div>`));
  } else {
    const promoted = result?.promotions ?? run.generations.filter((g) => g.promoted).length;
    tiles.push(stat("晋级", `${promoted} 次`, `完成 ${finished} / ${total} 代`));
  }
  if (key === "finished") {
    const reports = sealedReports(run);
    const done = SEALED.filter((stage) => run.sealing?.[stage]?.evaluated || reports[stage]);
    tiles.push(stat("封存评测", `${done.length} / ${SEALED.length}`,
      SEALED.filter((stage) => reports[stage]).map((stage) => `${stage} ${signed(reports[stage].score, 3)}`).join(" · ") || "尚未评测"));
  }
  if (run.proposer?.name === "llm") {
    tiles.push(stat("LLM 调用", String(run.agent_exchanges ?? 0), `每代上限 ${esc(run.proposer.options?.max_calls ?? 10)} 次`));
  }
  tiles.push(stat("用时", `<span${run.active ? ` data-elapsed="${esc(run.started_at)}"` : ""}>${duration(runSeconds(run))}</span>`,
    run.finished_at ? `结束于 ${when(run.finished_at)}` : run.active ? "进行中" : `最后记录 ${when(run.updated_at)}`));
  return `<section class="stats">${tiles.join("")}</section>`;
}

function niceTicks(lo, hi, count = 5) {
  const raw = (hi - lo) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw);
  const ticks = [];
  for (let i = Math.ceil(lo / step); i * step <= hi + step * 1e-9; i += 1) ticks.push(Number((i * step).toFixed(12)));
  let digits = 0;
  while (digits < 8 && Math.abs(Math.round(step * 10 ** digits) - step * 10 ** digits) > 1e-6) digits += 1;
  return { ticks, digits };
}

function forestHtml(run, task) {
  const generations = run.generations || [];
  if (!generations.length) return "";
  const gate = task?.gates?.min_delta;
  const rows = generations.map((g) => {
    const deltas = (g.screened?.candidates || []).map((item) => item.delta).filter(isNum);
    return { g, best: deltas.length ? Math.max(...deltas) : null, d: g.comparison?.decision || null };
  });
  const values = [0];
  if (isNum(gate)) values.push(gate);
  for (const row of rows) {
    if (isNum(row.best)) values.push(row.best);
    if (row.d) values.push(row.d.delta, ...row.d.interval);
  }
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  const span = Math.max(hi - lo, 0.002);
  lo -= span * 0.08;
  hi += span * 0.08;
  const x = (value) => (((value - lo) / (hi - lo)) * 100).toFixed(2);
  const { ticks, digits } = niceTicks(lo, hi);
  const gateX = isNum(gate) ? Number(x(gate)) : null;
  const tickHtml = ticks.filter((t) => gateX === null || Math.abs(Number(x(t)) - gateX) > 7)
    .map((t) => `<span style="left:${x(t)}%">${signed(t, digits)}</span>`).join("")
    + (gateX !== null ? `<span class="gate-label" style="left:${gateX}%">门槛 ${signed(gate, 3)}</span>` : "");
  const body = rows.map(({ g, best, d }) => {
    const finished = "incumbent_after" in g;
    const css = d ? (d.promoted ? "promoted" : d.interval[1] < 0 ? "worse" : "") : "";
    let value;
    if (d) value = `留出 <b class="${toneOf(d.delta)}">${signed(d.delta)}</b> [${signed(d.interval[0])}, ${signed(d.interval[1])}]${d.promoted ? ' · <span class="good">晋级</span>' : ""}`;
    else if (!finished && run.active) value = '<span class="muted">进行中</span>';
    else if (!finished) value = '<span class="muted">这一代没有完成</span>';
    else value = `<span class="muted">${isNum(best) ? "没有候选在初筛中超过在任方法" : "没有可评分的候选"}</span>`;
    return `<div class="forest-row"><div class="forest-label">第 ${g.generation} 代</div>
      <div class="forest-track ${css}"><i class="zero" style="left:${x(0)}%"></i>
        ${gateX !== null ? `<i class="gate" style="left:${gateX}%"></i>` : ""}
        ${d ? `<i class="ci" style="left:${x(d.interval[0])}%;width:${(Number(x(d.interval[1])) - Number(x(d.interval[0]))).toFixed(2)}%"></i>
          <i class="dot" style="left:${x(d.delta)}%" title="留出 Δ ${signed(d.delta)}"></i>` : ""}
        ${isNum(best) ? `<i class="screen" style="left:${x(best)}%" title="初筛最佳 Δ ${signed(best)}"></i>` : ""}
      </div><div class="forest-value">${value}</div></div>`;
  }).join("");
  return `<section class="panel">
    <div class="panel-head"><div><h2>代际进展</h2><p>每代最好的候选在留出日上相对在任方法的技能提升 Δ（候选 − 在任）。留出日与初筛日不重叠。</p></div></div>
    <div class="forest">${body}
      <div class="forest-axis"><div></div><div class="forest-ticks">${tickHtml}</div><div></div></div></div>
    <div class="legend"><span><i class="l-dot"></i>留出 Δ 与 95% 区间</span><span><i class="l-dot ok"></i>晋级</span>
      <span><i class="l-screen"></i>初筛最佳 Δ</span>${isNum(gate) ? `<span><i class="l-gate"></i>晋级门槛：提升须大于 ${signed(gate, 3)}，且区间下限大于 0</span>` : ""}</div>
  </section>`;
}

function tintStyle(value, scale) {
  const color = tint(value, scale);
  return color ? ` style="--tint:${color}"` : "";
}

function matrixHtml(task, rows, baseRows) {
  const byKey = new Map(rows.map((row) => [row.cell, row]));
  const base = baseRows ? new Map(baseRows.map((row) => [row.cell, row])) : null;
  const { targets, horizons } = gridOf(task, rows.map((row) => row.cell));
  const head = `<div class="mh"></div>${horizons.map((h) => `<div class="mh">${esc(horizonText(task, h))}</div>`).join("")}`;
  const body = targets.map((name) => {
    const target = targetOf(task, name);
    const unit = UNITS[target.unit] ?? target.unit;
    const cells = horizons.map((h) => {
      const row = byKey.get(`${name}@${h}`);
      if (!row) return '<div class="mc">–</div>';
      const before = base?.get(row.cell);
      const tip = `${row.n} 个预测 · MAE ${plain(row.mae, 3)} · 偏差 ${signed(row.bias, 3)}${unit ? ` ${unit}` : ""}`;
      const change = before ? row.skill - before.skill : null;
      let compared = "";
      if (before && before.rmse === row.rmse) compared = '<div class="sub faint">与种子相同</div>';
      else if (before) compared = `<div class="sub">种子 ${signed(before.skill, 3)} · 变化 <span class="${toneOf(change)}">${signed(change, 3)}</span></div>`;
      return `<div class="mc"${tintStyle(row.skill, 0.45)} title="${esc(tip)}">
        <div class="skill ${toneOf(row.skill)}">${signed(row.skill, 3)}</div>
        <div class="sub">RMSE ${plain(row.rmse, 3)} · 基线 ${plain(row.baseline_rmse, 3)}</div>
        ${compared}
        ${row.clipped ? `<div class="sub bad">${row.clipped} 个预测被裁剪到物理范围</div>` : ""}</div>`;
    }).join("");
    return `<div class="mr">${esc(target.label || name)}<div class="faint">${esc(name)}${unit ? ` · ${esc(unit)}` : ""}</div></div>${cells}`;
  }).join("");
  return `<div class="matrix-wrap"><div class="matrix" style="--cols:${horizons.length}">${head}${body}</div></div>`;
}

function cellsHtml(run, task) {
  if (!run.task_id) return "";
  const intro = "每格相对在 calibration_fit 上选定的朴素基线的技能：(RMSE基线 − RMSE) / max(RMSE, RMSE基线)，大于 0 表示优于基线。";
  const shell = (content, tabs = "") => `<section class="panel" id="cells">
    <div class="panel-head"><div><h2>逐格结果</h2><p>${intro}</p></div>${tabs}</div>${content}</section>`;
  if (run.active) return shell('<p class="note-line">运行结束后显示种子方法与最终方法的逐格结果。</p>');
  if (state.cellsError) return shell(`<div class="alert warn">无法计算逐格结果：${esc(state.cellsError)}</div>`);
  if (!state.cells) return shell('<p class="note-line">正在计算逐格结果…</p>');
  const cells = state.cells;
  const reports = sealedReports(run);
  const stages = ["model_selection", "calibration_uq", ...SEALED.filter((stage) => reports[stage])];
  const stage = stages.includes(state.stage) ? state.stage : "model_selection";
  const tabs = `<div class="tabs" role="tablist">${stages.map((name) => `<button role="tab" data-action="stage" data-stage="${name}"
    aria-selected="${name === stage}"${SEALED.includes(name) ? ' class="tab-sealed"' : ""}>${STAGES[name][0]}</button>`).join("")}</div>`;
  const same = !cells.final || run.result?.seed_digest === run.result?.incumbent_digest;
  let matrix;
  let foot;
  if (SEALED.includes(stage)) {
    const report = reports[stage];
    matrix = matrixHtml(task, report.cells, null);
    foot = `<span>最终方法总技能 <b class="${toneOf(report.score)}">${signed(report.score)}</b></span>
      <span>95% 区间 [${signed(report.interval[0])}, ${signed(report.interval[1])}]</span><span>${report.origins} 个起点</span>
      <span>该分区累计评过 ${report.methods_exposed_on_stage} 个方法</span>`;
  } else {
    const shown = (cells.final ?? cells.seed)[stage];
    matrix = matrixHtml(task, shown.cells, same ? null : cells.seed[stage].cells);
    const origins = Math.max(0, ...shown.cells.map((row) => row.eligible ?? row.n));
    foot = same
      ? `<span>${cells.final ? "最终方法与种子相同（没有晋级）" : "种子方法（运行没有给出最终方法）"}：总技能 <b class="${toneOf(shown.score)}">${signed(shown.score)}</b></span>`
      : `<span>种子 <b>${signed(cells.seed[stage].score)}</b> → 最终 <b class="${toneOf(shown.score)}">${signed(shown.score)}</b></span>`;
    foot += `<span>${origins} 个计分起点</span><span class="faint">${STAGES[stage][1]}</span>`;
  }
  return shell(`${matrix}<div class="matrix-foot">${foot}</div>`, tabs);
}

function sealHtml(run, task, key) {
  const intro = "validation 与 final_test 在运行期间不可见。每个方法（按方法摘要和数据判定，跨运行有效）在每个封存分区只能评一次，final_test 须在 validation 之后。";
  const shell = (content, method = "") => `<section class="panel">
    <div class="panel-head"><div><h2>封存评测</h2><p>${intro}</p></div>${method}</div>${content}</section>`;
  if (key !== "finished") {
    return shell(`<p class="note-line">${run.active ? "运行结束后可以对最终方法做封存评测。" : "只有正常完成的运行才能做封存评测。"}</p>`);
  }
  const sealing = run.sealing || {};
  if (sealing.error) return shell(`<div class="alert warn">无法确定封存状态：${esc(sealing.error)}</div>`);
  const reports = sealedReports(run);
  const cards = SEALED.map((stage) => {
    const entry = sealing[stage] || { evaluated: Boolean(reports[stage]) };
    const report = reports[stage];
    if (entry.evaluated) {
      const elsewhere = entry.run_id && entry.run_id !== run.run_id;
      const origin = elsewhere
        ? `在运行 <a href="${esc(runHref(entry.run_id))}">${esc(state.runs.find((item) => item.run_id === entry.run_id)?.label || entry.run_id)}</a> 中评测`
        : "在本运行中评测";
      if (!report) {
        return `<div class="seal-card done"><div class="seal-stage">${stage}</div><p>已登记曝光，但账本中没有结果记录（评测可能在写入前中断）。</p>
          <p class="sub">${origin} · ${when(entry.at)}</p></div>`;
      }
      return `<div class="seal-card done"><div class="seal-stage">${stage}</div>
        <div class="seal-score ${toneOf(report.score)}">${signed(report.score)}</div>
        <p class="sub num">95% 区间 [${signed(report.interval[0])}, ${signed(report.interval[1])}] · ${report.origins} 个起点</p>
        <p class="sub">${origin}${entry.at ? ` · ${when(entry.at)}` : ""} · 评测时该分区累计评过 ${report.methods_exposed_on_stage} 个方法</p>
        <button class="link" data-action="show-stage" data-stage="${stage}">查看逐格结果</button></div>`;
    }
    const blocked = stage === "final_test" && !(sealing.validation?.evaluated || reports.validation);
    return `<div class="seal-card"><div class="seal-stage">${stage}</div><p class="muted">尚未评测</p>
      <button class="primary" data-action="seal" data-stage="${stage}"${blocked ? " disabled" : ""}>评测 ${stage}</button>
      ${blocked ? '<p class="sub">须先评测 validation</p>' : ""}</div>`;
  }).join("");
  const method = sealing.method ? `<span class="faint">最终方法 <span class="mono">${esc(sealing.method.slice(0, 12))}</span></span>` : "";
  return shell(`<div class="seal-grid">${cards}</div>`, method);
}

function valueAt(params, path) {
  if (!params) return undefined;
  const dot = path.indexOf(".");
  if (dot < 0) return params[path];
  const head = params[path.slice(0, dot)];
  return head && typeof head === "object" && !Array.isArray(head) ? head[path.slice(dot + 1)] : undefined;
}
function termText(term) {
  if (!term || typeof term !== "object" || Array.isArray(term) || typeof term.op !== "string") return JSON.stringify(term);
  const keys = Object.keys(term).filter((k) => k !== "op").sort((a, b) => (a === "col" ? -1 : b === "col" ? 1 : a.localeCompare(b)));
  return keys.length ? `${term.op}(${keys.map((k) => short(term[k])).join(", ")})` : term.op;
}
function changeHtml(path, value, old) {
  let shown;
  if (typeof value === "number" && typeof old === "number") {
    return `<div class="change scalar"><span class="change-path">${esc(path)}</span>
      <span class="old">${esc(short(old))}</span><span class="arrow">→</span><b>${esc(short(value))}</b></div>`;
  } else if (Array.isArray(value) && (Array.isArray(old) || old === undefined)) {
    const before = (old || []).map((item) => JSON.stringify(item));
    const after = value.map((item) => JSON.stringify(item));
    const added = value.filter((item, i) => !before.includes(after[i]));
    const removed = (old || []).filter((item, i) => !after.includes(before[i]));
    shown = added.map((item) => `<span class="chip add">+ ${esc(termText(item))}</span>`).join(" ")
      + removed.map((item) => ` <span class="chip del">${MINUS} ${esc(termText(item))}</span>`).join("");
    if (!shown) shown = value.map((item) => `<span class="chip">${esc(termText(item))}</span>`).join(" ") || '<span class="chip">[]</span>';
  } else {
    const text = typeof value === "string" ? value : JSON.stringify(value);
    shown = `<span class="chip" title="${esc(text)}">${esc(text.length > 160 ? `${text.slice(0, 160)}…` : text)}</span>`;
  }
  return `<div class="change"><span class="change-path">${esc(path)}</span>${shown}</div>`;
}
function changesHtml(changes, incumbent) {
  const entries = Object.entries(changes || {});
  return entries.length ? entries.map(([path, value]) => changeHtml(path, value, valueAt(incumbent?.params, path))).join("") : "–";
}
function miniHtml(task, deltas) {
  if (!deltas) return "";
  const keys = Object.keys(deltas);
  const { targets, horizons } = gridOf(task, keys);
  const squares = targets.flatMap((name) => horizons.map((h) => {
    const key = `${name}@${h}`;
    const value = deltas[key];
    return `<i style="--tint:${tint(value, 0.05) || "var(--border)"}" title="${esc(`${cellLabel(task, key)}：${signed(value)}`)}"></i>`;
  })).join("");
  return `<span class="mini" style="--cols:${horizons.length}">${squares}</span>`;
}

function candidatesHtml(run, task, g, incumbent) {
  const finalist = g.comparison?.finalist;
  const rows = [];
  for (const item of g.screened?.candidates || []) {
    let outcome;
    if (item.error) outcome = badge("评估失败", "bad");
    else if (item.digest === finalist) outcome = g.promoted ? badge("晋级", "ok") : badge("决赛 · 未晋级", "flat");
    else if (item.delta > 0) outcome = badge("初筛淘汰", "flat");
    else outcome = badge("不如在任", "flat");
    rows.push({
      index: item.index,
      html: `<tr${item.error ? ' class="failed"' : ""}><td class="num">${item.index + 1}</td><td>${changesHtml(item.edit?.changes, incumbent)}</td>
        <td class="num ${toneOf(item.delta)}">${item.error ? "–" : signed(item.delta)}</td><td>${miniHtml(task, item.cell_deltas)}</td>
        <td>${outcome}</td><td class="reason"><div class="clamp">${esc(item.error || item.edit?.rationale || "")}</div></td></tr>`,
    });
  }
  for (const item of g.rejected || []) {
    rows.push({
      index: item.index,
      html: `<tr class="failed"><td class="num">${item.index + 1}</td><td>${changesHtml(item.edit?.changes, incumbent)}</td><td class="num">–</td><td></td>
        <td>${badge("被拒绝", "warn")}</td><td class="reason"><div class="clamp">${esc((item.problems || []).join("；"))}</div></td></tr>`,
    });
  }
  for (const item of g.dropped || []) {
    rows.push({
      index: Infinity,
      html: `<tr><td class="num">–</td><td>${changesHtml(item.changes, incumbent)}</td><td class="num">–</td><td></td>
        <td>${badge("评审剔除", "flat")}</td><td class="reason"><div class="clamp">${esc(item.reason || "")}</div></td></tr>`,
    });
  }
  if (!rows.length) return '<p class="note-line">这一代没有候选。</p>';
  rows.sort((a, b) => a.index - b.index);
  return `<div class="table-wrap"><table class="table cands"><thead><tr><th>#</th><th>修改（在任 → 候选）</th><th class="num">初筛 Δ</th>
    <th>逐格 Δ</th><th>结果</th><th>理由 / 问题</th></tr></thead><tbody>${rows.map((row) => row.html).join("")}</tbody></table></div>`;
}

function decisionHtml(run, task, g) {
  const d = g.comparison?.decision;
  if (!d) return "";
  const gates = task?.gates;
  const failed = d.cells.filter((cell) => !cell.passed);
  const key = `${run.run_id}:gen:${g.generation}:cells`;
  const table = d.cells.map((cell) => {
    const [text, tone] = CELL_REASONS[cell.reason] || [cell.reason, "flat"];
    const better = isNum(cell.d) ? -cell.d : null;
    return `<tr><td>${esc(cellLabel(task, cell.cell))}</td><td class="num ${toneOf(better)}">${signed(cell.d)}</td>
      <td class="num">${isNum(cell.d_lower) ? `[${signed(cell.d_lower)}, ${signed(cell.d_upper)}]` : "–"}</td>
      <td class="num">${plain(cell.cap, 4)}</td><td>${badge(text, tone)}</td></tr>`;
  }).join("");
  const level = gates ? `${Math.round((1 - gates.cell_alpha) * 100)}%` : "";
  return `<div class="decision${d.promoted ? " ok" : ""}">
    <div class="decision-line"><b>留出比较</b>：决赛候选相对在任方法 <b class="${toneOf(d.delta)}">${signed(d.delta)}</b>，
      95% 区间 [${signed(d.interval[0])}, ${signed(d.interval[1])}]，${d.blocks} 天 →
      <b class="${d.promoted ? "good" : ""}">${d.promoted ? "晋级" : "未晋级"}</b>：${esc(decisionText(d.reason, task))}</div>
    <div class="decision-figures">候选 ${signed(d.candidate)} · 在任 ${signed(d.incumbent)}${gates
      ? ` · 门槛：提升大于 ${signed(gates.min_delta, 3)}，区间下限大于 0，每格的 nRMSE 退化不超过在任的 ${Math.round(gates.cell_cap * 100)}% 且没有可信退化` : ""}</div>
    <details class="sub" data-key="${esc(key)}"${details(key, failed.length > 0) ? " open" : ""}>
      <summary>逐格非劣检验 <span class="faint">${failed.length ? `${failed.length} 格未通过` : `${d.cells.length} 格全部通过`}</span></summary>
      <div class="table-wrap"><table class="table"><thead><tr><th>格</th><th class="num">Δ nRMSE（候选 − 在任）</th>
        <th class="num">${level} 区间</th><th class="num">允许退化</th><th>结论</th></tr></thead><tbody>${table}</tbody></table></div>
    </details></div>`;
}

function noteHtml(run, g, note, index) {
  const key = `${run.run_id}:gen:${g.generation}:note:${index}`;
  const groups = [["观察", note.observations], ["假设", note.hypotheses], ["方向", note.directions]];
  const counts = groups.filter(([, items]) => items?.length).map(([title, items]) => `${title} ${items.length}`).join(" · ");
  const body = groups.filter(([, items]) => items?.length)
    .map(([title, items]) => `<div><h4>${title}</h4><ul>${items.map((item) => `<li>${esc(item)}</li>`).join("")}</ul></div>`).join("");
  return `<details class="note" data-key="${esc(key)}"${details(key, false) ? " open" : ""}>
    <summary>研究笔记 · ${esc(ROLES[note.role] || note.role || "智能体")} <span class="faint">${counts}</span></summary>
    <div class="note-body">${body}</div></details>`;
}

function generationHtml(run, task, g, incumbent) {
  const key = `${run.run_id}:gen:${g.generation}`;
  const finished = "incumbent_after" in g;
  const running = run.active && !finished;
  const d = g.comparison?.decision;
  let outcome;
  let summary;
  if (running) {
    outcome = badge("进行中", "live");
    summary = g.screened ? "初筛完成，等待留出比较" : g.notes.length || g.proposed.length ? "正在提出与评估候选" : "正在研究";
  } else if (g.promoted) {
    outcome = badge("已晋级", "ok");
    summary = `留出 ${signed(d?.delta)} [${signed(d?.interval[0])}, ${signed(d?.interval[1])}]`;
  } else if (d) {
    outcome = badge("未晋级", "flat");
    summary = `留出 ${signed(d.delta)} [${signed(d.interval[0])}, ${signed(d.interval[1])}] · ${decisionText(d.reason, task)}`;
  } else if (finished) {
    outcome = badge("无决赛候选", "flat");
    summary = "没有候选在初筛中超过在任方法";
  } else {
    outcome = badge("未完成", "warn");
    summary = "这一代在完成前结束";
  }
  const screened = g.screened?.candidates?.length || 0;
  const counts = [`${screened} 个候选`];
  if (g.rejected.length) counts.push(`${g.rejected.length} 个被拒绝`);
  if (g.dropped.length) counts.push(`${g.dropped.length} 个被评审剔除`);
  const { screening, holdout } = g.cohorts;
  const errors = (g.agent_errors || []).map((error) => `<div class="alert bad"><b>智能体错误</b><div class="alert-body">${esc(error)}</div></div>`).join("");
  return `<details class="gen" data-key="${esc(key)}"${details(key, running) ? " open" : ""}>
    <summary><span class="gen-no">第 ${g.generation} 代</span>${outcome}<span class="gen-sum">${esc(summary)}</span>
      <span class="gen-counts">${counts.join(" · ")}</span></summary>
    <div class="gen-body">
      <div class="cohorts">初筛 ${screening.origins} 个起点（${screening.blocks.length} 天）· 留出 ${holdout.origins} 个起点（${holdout.blocks.length} 天）·
        在任方法 <span class="mono">${esc(String(g.incumbent).slice(0, 12))}</span></div>
      ${(g.notes || []).map((note, index) => noteHtml(run, g, note, index)).join("")}${errors}
      ${candidatesHtml(run, task, g, incumbent)}${decisionHtml(run, task, g)}
    </div></details>`;
}

function generationsHtml(run, task) {
  const generations = run.generations || [];
  if (!generations.length) {
    return run.active ? '<section class="panel"><p class="note-line">等待第 0 代开始…</p></section>' : "";
  }
  let incumbent = run.seed;
  const cards = generations.map((g) => {
    const html = generationHtml(run, task, g, incumbent);
    if (g.promoted && g.comparison) incumbent = g.proposed.find((item) => item.digest === g.comparison.finalist)?.genome ?? incumbent;
    return html;
  }).join("");
  return `<section><div class="section-head"><h2>各代详情</h2><span class="faint">按时间顺序，点击展开研究笔记、候选和检验</span></div>
    <div class="gens">${cards}</div></section>`;
}

function kv(pairs) {
  return `<dl class="kv">${pairs.map(([key, value]) => `<dt>${esc(key)}</dt><dd>${value}</dd>`).join("")}</dl>`;
}
function json(value) {
  return `<pre class="json">${esc(JSON.stringify(value, null, 2))}</pre>`;
}
function configHtml(run) {
  if (!run.config) return "";
  const c = run.config;
  const key = `${run.run_id}:config`;
  const options = Object.entries(run.proposer?.options || {})
    .map(([name, value]) => [name, `<span class="mono">${esc(typeof value === "string" ? value : JSON.stringify(value))}</span>`]);
  const finalGenome = run.result && run.result.incumbent_digest !== run.result.seed_digest ? run.result.incumbent : null;
  return `<details class="panel config" data-key="${esc(key)}"${details(key, false) ? " open" : ""}>
    <summary><h2>运行配置与方法参数</h2></summary>
    <div class="config-grid">
      <div><h4>竞技设置</h4>${kv([
        ["代数 × 每代候选", `${c.generations} × ${c.candidates}`],
        ["随机种子", esc(c.seed)],
        ["信赖域", `每个坐标最多移动范围的 ${esc(c.max_step)}，每个修改最多改 ${esc(c.max_changes)} 个坐标`],
        ["初筛 / 留出", `按 ${esc(c.chunk_blocks)} 天一组抽签，约 ${Math.round(c.screen_fraction * 100)}% 的日子用于初筛`],
        ["选择分区", esc(c.stage)],
      ])}</div>
      <div><h4>提案者 · ${esc(proposerText(run.proposer))}</h4>${options.length ? kv(options) : '<p class="note-line">没有选项</p>'}</div>
      <div><h4>种子方法</h4>${json(run.seed)}</div>
      ${finalGenome ? `<div><h4>最终方法</h4>${json(finalGenome)}</div>` : ""}
    </div></details>`;
}

/* ---- run page: actions --------------------------------------------------- */

async function openReport(run) {
  const win = window.open("", "_blank");
  if (win) {
    win.opener = null;
    win.document.title = "ecogym 报告";
    win.document.body.textContent = "正在生成报告…";
  }
  try {
    const html = await api(runPath(run.run_id, "/report"));
    const url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
    if (win) {
      win.location.href = url;
    } else {
      const link = Object.assign(document.createElement("a"), { href: url, download: `${run.run_id.replace(":", "-")}.html` });
      link.click();
    }
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  } catch (error) {
    win?.close();
    toast(`报告生成失败：${error.message}`, "bad");
  }
}

async function stopRun(run) {
  if (!confirm("在当前这一代结束后停止这条运行？已完成的代保留在账本中，停止的运行不能做封存评测。")) return;
  state.stopping.add(run.run_id);
  renderRun(true);
  try {
    await api(runPath(run.run_id, "/stop"), { method: "POST", body: {} });
    toast("将在当前这一代结束后停止", "ok");
  } catch (error) {
    state.stopping.delete(run.run_id);
    renderRun(true);
    toast(`停止失败：${error.message}`, "bad");
  }
}

async function sealStage(run, stage, button) {
  const method = run.sealing?.method ? `（${run.sealing.method.slice(0, 12)}）` : "";
  if (!confirm(`对最终方法${method}做一次 ${stage} 封存评测？\n\n同一方法在每个封存分区只能评一次，跨运行有效；结果写入账本，不能撤销。`)) return;
  button.disabled = true;
  button.textContent = "正在评测…";
  try {
    const report = await api(runPath(run.run_id, "/seal"), { method: "POST", body: { stage } });
    toast(`${stage} 评测完成：总技能 ${signed(report.score)}`, "ok");
    state.stage = stage;
    state.signature = null;
    await refreshRun(run.run_id, state.seq, true);
    refreshRuns();
  } catch (error) {
    toast(`封存评测失败：${error.message}`, "bad");
    state.signature = null;
    await refreshRun(run.run_id, state.seq, true);
  }
}

main().addEventListener("click", (event) => {
  const clamp = event.target.closest(".clamp");
  if (clamp && !event.target.closest("a, button")) clamp.classList.toggle("expanded");
  const button = event.target.closest("button[data-action]");
  const run = state.run;
  if (!button || !run) return;
  const action = button.dataset.action;
  if (action === "stop") stopRun(run);
  else if (action === "report") openReport(run);
  else if (action === "seal") sealStage(run, button.dataset.stage, button);
  else if (action === "copy") {
    navigator.clipboard?.writeText(run.run_id).then(() => toast(`已复制 ${run.run_id}`), () => toast(run.run_id));
  } else if (action === "stage" || action === "show-stage") {
    state.stage = button.dataset.stage;
    renderRun(true);
    if (action === "show-stage") $("#cells")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
});

/* ---- empty state ---------------------------------------------------------- */

function showEmpty() {
  main().innerHTML = `<div class="empty-state">
    <svg class="logo" viewBox="0 0 32 32" style="width:56px;height:56px" aria-hidden="true"><rect width="32" height="32" rx="8"/><path d="M9 22c0-8 6-13 15-13-1 9-6 14-13 14"/><path d="M9 23l8-8"/></svg>
    <h1>还没有运行</h1>
    <p>选一个任务和种子方法，再选提案者（坐标扫描、随机搜索或 LLM 研究智能体）。训练场逐代提出修改，在留出日上检验，并把每一步写入账本。</p>
    <a class="button primary" href="#new">＋ 新建运行</a></div>`;
}

/* ---- new run form --------------------------------------------------------- */

function loadSaved() {
  try {
    return JSON.parse(localStorage.getItem(SAVED_KEY) || "{}") || {};
  } catch {
    return {};
  }
}
function choiceHtml(name, value, title, text, checked) {
  return `<label class="choice"><input type="radio" name="${name}" value="${esc(value)}"${checked ? " checked" : ""}>
    <span class="choice-body"><span class="choice-title">${title}</span>${text ? `<span class="choice-text">${esc(text)}</span>` : ""}</span></label>`;
}
function taskHint(task) {
  if (!task) return "";
  const targets = task.targets.map((target) => target.label || target.name).join("、");
  const horizons = task.horizons.map((h) => horizonText(task, h)).join("、");
  return `数据集 <code>${esc(task.dataset)}</code> · ${task.targets.length} 个目标（${esc(targets)}）× ${task.horizons.length} 个时距（${esc(horizons)}）
    = ${task.cells.length} 格 · 晋级门槛：留出日提升大于 ${signed(task.gates.min_delta, 3)}`;
}

const LLM_FIELDS = ["transport", "model", "proxy", "base_url", "key_env", "dsh_bin", "patches", "max_tool_calls", "review"];

function showNewRun() {
  const element = main();
  element.innerHTML = "";
  element.append($("#new-run-template").content.cloneNode(true));
  element.scrollTop = 0;
  const form = $("#new-run", element);
  const fields = form.elements;
  const saved = loadSaved();
  const tasks = state.catalog.tasks;
  fields.task.innerHTML = tasks.map((task) => `<option value="${esc(task.id)}">${esc(task.title)}</option>`).join("");
  if (tasks.some((task) => task.id === saved.task)) fields.task.value = saved.task;

  const renderPresets = () => {
    const task = taskInfo(fields.task.value);
    const names = Object.keys(task?.presets || {});
    const chosen = names.includes(saved.preset) ? saved.preset : names[0];
    $('[data-choices="preset"]', form).innerHTML = names
      .map((name) => choiceHtml("preset", name, `<code>${esc(name)}</code>`, task.presets[name], name === chosen)).join("")
      || '<p class="note-line">这个任务没有声明种子方法。</p>';
    $('[data-hint="task"]', form).innerHTML = taskHint(task);
  };
  fields.task.addEventListener("change", renderPresets);
  renderPresets();

  const known = Object.keys(PROPOSERS);
  const rank = (name) => (known.includes(name) ? known.indexOf(name) : known.length);
  const proposers = [...state.catalog.proposers].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  const chosen = proposers.includes(saved.proposer) ? saved.proposer : proposers[0];
  $('[data-choices="proposer"]', form).innerHTML = proposers
    .map((name) => choiceHtml("proposer", name, `${esc(PROPOSERS[name]?.[0] ?? name)}<code>${esc(name)}</code>`, PROPOSERS[name]?.[1] ?? "", name === chosen))
    .join("");
  for (const name of LLM_FIELDS) {
    const value = saved.llm?.[name];
    if (value === undefined || !fields[name]) continue;
    if (fields[name].type === "checkbox") fields[name].checked = Boolean(value);
    else fields[name].value = value;
  }
  const sync = () => {
    const proposer = fields.proposer.value;
    $$("[data-when]", form).forEach((item) => { item.hidden = item.dataset.when !== proposer; });
    $$("[data-transport]", form).forEach((item) => { item.hidden = item.dataset.transport !== fields.transport.value; });
  };
  form.addEventListener("change", sync);
  sync();
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitRun(form);
  });
  $("[data-cancel]", form).addEventListener("click", (event) => {
    event.preventDefault();
    if (history.length > 1 && state.runs.length) history.back();
    else go({ kind: "home" });
  });
}

function fieldError(input, message) {
  const error = new Error(message);
  error.input = input;
  return error;
}
function integer(input, label, lo, hi) {
  const value = Number(input.value);
  if (input.value.trim() === "" || !Number.isInteger(value) || value < lo || (hi !== undefined && value > hi)) {
    throw fieldError(input, `${label}应为 ${lo}${hi !== undefined ? ` 到 ${hi}` : " 或更大"} 的整数`);
  }
  return value;
}

async function submitRun(form) {
  const fields = form.elements;
  const error = $(".form-error", form);
  const submit = $('button[type="submit"]', form);
  error.textContent = "";
  try {
    const body = {
      task: fields.task.value,
      preset: fields.preset?.value || "",
      proposer: fields.proposer.value,
      generations: integer(fields.generations, "代数", 1, 20),
      candidates: integer(fields.candidates, "每代候选数", 1, 12),
      seed: integer(fields.seed, "随机种子", 0),
      max_changes: integer(fields.max_changes, "每个修改最多改动的坐标数", 1, 20),
      label: fields.label.value.trim(),
    };
    const step = Number(fields.max_step.value);
    if (!(step > 0 && step <= 1)) throw fieldError(fields.max_step, "信赖域应在 0 到 1 之间（不含 0）");
    body.max_step = step;
    if (!body.preset) throw new Error("请选择种子方法");
    const options = {};
    const llm = {};
    if (body.proposer === "llm") {
      options.transport = fields.transport.value;
      if (fields.model.value.trim()) options.model = fields.model.value.trim();
      if (options.transport === "http") {
        if (fields.base_url.value.trim()) options.base_url = fields.base_url.value.trim();
        options.key_env = fields.key_env.value.trim() || "ECOGYM_LLM_API_KEY";
        options.proxy = fields.proxy.value;
      } else {
        if (fields.dsh_bin.value.trim()) options.dsh_bin = fields.dsh_bin.value.trim();
        const patches = fields.patches.value.split("\n").map((line) => line.trim()).filter(Boolean);
        if (patches.length) options.patches = patches;
      }
      options.max_tool_calls = integer(fields.max_tool_calls, "研究员每代工具调用", 0, 20);
      options.review = fields.review.checked;
      for (const name of LLM_FIELDS) llm[name] = fields[name].type === "checkbox" ? fields[name].checked : fields[name].value;
    }
    const extra = fields.extra.value.trim();
    if (extra) {
      let parsed;
      try {
        parsed = JSON.parse(extra);
      } catch {
        throw fieldError(fields.extra, "其他提案者选项不是合法的 JSON");
      }
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw fieldError(fields.extra, '其他提案者选项应为 JSON 对象，例如 {"step": 0.1}');
      }
      Object.assign(options, parsed);
    }
    if (Object.keys(options).length) body.options = options;
    const saved = loadSaved();
    localStorage.setItem(SAVED_KEY, JSON.stringify({ task: body.task, preset: body.preset, proposer: body.proposer,
      llm: body.proposer === "llm" ? llm : saved.llm }));
    submit.disabled = true;
    submit.textContent = "正在启动…";
    const { run_id: runId } = await api("runs", { method: "POST", body });
    toast(`已开始运行 ${runId}`, "ok");
    await refreshRuns();
    go({ kind: "run", id: runId });
  } catch (failure) {
    error.textContent = failure.message;
    failure.input?.focus();
    submit.disabled = false;
    submit.textContent = "开始运行";
  }
}

/* ---- start --------------------------------------------------------------- */

async function start() {
  takeToken();
  try {
    state.catalog = await api("catalog");
  } catch (error) {
    const title = error.status === 401 ? "需要访问令牌" : "无法连接训练场服务";
    main().innerHTML = `<div class="alert bad"><b>${title}</b><div class="alert-body">${esc(error.message)}</div></div>`;
    const retry = () => {
      if (!takeToken()) return;
      window.removeEventListener("hashchange", retry);
      main().innerHTML = '<div class="placeholder">正在加载…</div>';
      start();
    };
    window.addEventListener("hashchange", retry);
    return;
  }
  state.tasks = new Map(state.catalog.tasks.map((task) => [task.id, task]));
  const ledger = state.catalog.ledger || "";
  const info = $("#ledger-info");
  info.textContent = `账本 ${ledger.split("/").slice(-2).join("/")} · ecogym ${state.catalog.version}`;
  info.title = ledger;
  $("#run-filter").addEventListener("input", (event) => {
    state.filter = event.target.value;
    renderRunList();
  });
  await refreshRuns();
  window.addEventListener("hashchange", onRoute);
  onRoute();
  setInterval(() => { if (!document.hidden) refreshRuns(); }, 5000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshRuns(); });
  setInterval(() => {
    for (const element of $$("[data-elapsed]")) {
      const started = parseTime(element.dataset.elapsed);
      if (started !== null) element.textContent = duration((Date.now() - started) / 1000);
    }
  }, 1000);
}

start();
