"""Static HTML report of one run, from its ledger events (plus a bench for cell tables)."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from typing import Any

from .evaluation.harness import Bench

_STYLE = """
body{font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:2rem auto;max-width:1100px;color:#1d2433}
h1{font-size:1.5rem;margin-bottom:.2rem}h2{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid #d8dde6}
table{border-collapse:collapse;margin:.5rem 0 1rem;width:100%}th,td{padding:.3rem .5rem;border-bottom:1px solid #eceff4;
text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left}th{background:#f5f7fa}
.muted{color:#6b7385}.good{color:#11734b;font-weight:600}.bad{color:#b42318}.mono{font-family:Menlo,monospace;font-size:12px}
pre{background:#f5f7fa;padding:.8rem;overflow:auto;font-size:12px}
.tag{display:inline-block;padding:0 .4rem;border-radius:4px;background:#eef1f6;margin-right:.3rem}
"""


def _e(value: Any) -> str:
    return html.escape(str(value))


def _num(value: Any, digits: int = 4, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "–"
    if isinstance(value, (int, float)):
        return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return _e(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{_e(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def cell_table(cells: Sequence[dict[str, Any]]) -> str:
    rows = []
    for cell in cells:
        css = "good" if (cell["skill"] or 0) > 0 else "bad"
        rows.append([
            _e(cell["cell"]), str(cell["n"]), _num(cell["baseline_rmse"], 3), _num(cell["rmse"], 3),
            f"<span class='{css}'>{_num(cell['skill'], 3, True)}</span>", _num(cell["bias"], 3, True),
        ])
    return _table(["cell", "origins", "baseline RMSE", "RMSE", "skill", "bias"], rows)


def _changes(edit: dict[str, Any]) -> str:
    changes = edit.get("changes", {})
    parts = []
    for path, value in changes.items():
        shown = _num(value, 4) if isinstance(value, float) else _e(json.dumps(value, ensure_ascii=False)[:120])
        parts.append(f"<span class='tag mono'>{_e(path)} = {shown}</span>")
    return "".join(parts) or "–"


def _generation(generation: dict[str, Any]) -> str:
    screen, hold = generation["cohorts"]["screening"], generation["cohorts"]["holdout"]
    parts = [
        f"<h3>Generation {generation['generation']}</h3>",
        f"<p class='muted'>screening {screen['origins']} origins / {len(screen['blocks'])} days; "
        f"holdout {hold['origins']} origins / {len(hold['blocks'])} days</p>",
    ]
    sections = (("Observations", "observations"), ("Hypotheses", "hypotheses"), ("Directions", "directions"))
    for note in generation.get("notes", []):
        for title, key in sections:
            items = note.get(key) or []
            if items:
                parts.append(f"<p class='muted'>{title}</p><ul>" + "".join(f"<li>{_e(item)}</li>" for item in items)
                             + "</ul>")
    for error in generation.get("agent_errors", []):
        parts.append(f"<p class='bad'>agent error: {_e(error)}</p>")
    screened = (generation.get("screened") or {}).get("candidates", [])
    rows = []
    for item in screened:
        delta = item.get("delta")
        css = "good" if (delta or 0) > 0 else "muted"
        rationale = item["edit"].get("rationale", "")
        rows.append([_changes(item["edit"]), f"<span class='{css}'>{_num(delta, 4, True)}</span>",
                     _e(item.get("error", rationale))])
    for item in generation.get("rejected", []):
        rows.append([_changes(item["edit"]), "<span class='bad'>rejected</span>", _e("; ".join(item["problems"]))])
    for item in generation.get("dropped", []):
        rows.append([_changes(item), "<span class='muted'>dropped by review</span>", _e(item.get("reason", ""))])
    parts.append(_table(["edit", "screening Δ", "rationale / problem"], rows))
    comparison = generation.get("comparison")
    if comparison:
        decision = comparison["decision"]
        css = "good" if decision["promoted"] else "bad"
        low, high = decision["interval"]
        parts.append(
            f"<p>Holdout: finalist <span class='mono'>{_e(comparison['finalist'][:12])}</span> vs incumbent: "
            f"Δ {_num(decision['delta'], 4, True)}, 95% CI [{_num(low, 4, True)}, {_num(high, 4, True)}], "
            f"{decision['blocks']} days → <span class='{css}'>{_e(decision['reason'])}</span></p>"
        )
        failed = [cell for cell in decision["cells"] if not cell["passed"]]
        if failed:
            parts.append(_table(
                ["cell", "Δ nRMSE", "lower", "reason"],
                [[_e(cell["cell"]), _num(cell["d"], 4, True), _num(cell["d_lower"], 4, True), _e(cell["reason"])]
                 for cell in failed],
            ))
    else:
        parts.append("<p class='muted'>No candidate beat the incumbent on screening; no holdout comparison.</p>")
    return "".join(parts)


def render_run(run: dict[str, Any], bench: Bench | None = None) -> str:
    result = run.get("result", {})
    title = f"ecogym run {run['run_id']}"
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{_e(title)}</title><style>{_STYLE}</style>"
             "</head><body>", f"<h1>{_e(title)}</h1>",
             f"<p class='muted'>{_e(run['task_id'])} · {_e(run.get('label') or 'no label')}"
             f" · status {_e(run['status'])}"
             f" · started {_e(run['started_at'])} · ecogym {_e(run.get('ecogym'))}</p>"]
    proposer = run["proposer"]
    config = run["config"]
    parts.append("<h2>Summary</h2>")
    parts.append(_table(["", "value"], [
        ["proposer", _e(f"{proposer['name']} {json.dumps(proposer['options'])}")],
        ["generations × candidates", _e(f"{config['generations']} × {config['candidates']} (seed {config['seed']})")],
        ["trust region", _e(f"step ≤ {config['max_step']} of range, ≤ {config['max_changes']} coordinates")],
        ["seed score (model_selection)", _num(result.get("seed_score"), 4, True)],
        ["final score (model_selection)", _num(result.get("incumbent_score"), 4, True)],
        ["promotions", _e(result.get("promotions", "–"))],
        ["seconds", _e(result.get("seconds", "–"))],
    ]))
    if bench is not None and result:
        incumbent = bench.genome(result["incumbent"])
        for stage in ("model_selection", "calibration_uq"):
            evaluation = bench.evaluate(incumbent, stage)
            parts.append(f"<h2>Final method on {stage} (overall {_num(evaluation.score(), 4, True)})</h2>")
            parts.append(cell_table(evaluation.cells()))
    for sealed in run.get("sealed", []):
        low, high = sealed["interval"]
        parts.append(f"<h2>Sealed {sealed['stage']}: {_num(sealed['score'], 4, True)} "
                     f"[{_num(low, 3, True)}, {_num(high, 3, True)}]</h2>")
        parts.append(f"<p class='muted'>{sealed['origins']} origins; methods ever exposed on this stage: "
                     f"{sealed['methods_exposed_on_stage']}</p>")
        parts.append(cell_table(sealed["cells"]))
    parts.append("<h2>Generations</h2>")
    parts.extend(_generation(generation) for generation in run["generations"])
    if result:
        parts.append("<h2>Final genome</h2>")
        parts.append(f"<pre>{_e(json.dumps(result['incumbent'], indent=2, ensure_ascii=False))}</pre>")
    parts.append("</body></html>")
    return "\n".join(parts)


__all__ = ["cell_table", "render_run"]
