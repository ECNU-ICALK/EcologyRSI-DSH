"""LLM research agents as an arena proposer.

Per generation, three roles share one briefing (task, rules of judgement,
domain knowledge, solver schema, current method, trust region, history):

* researcher: calls research tools (development data only) and writes a note
  of observations, hypotheses and directions;
* proposer: turns the note into up to k concrete edits;
* reviewer: drops edits only for a concrete reason (optional).

Every reply is one JSON object; an unusable reply gets one repair request.
All calls are logged to the ledger as ``agent_exchange`` events. The host
computes every forecast, so the agents work at the level of methods only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

from ..arena.genome import Edit, apply_edit, check_edit, edit_from_dict
from ..arena.proposers import ProposalContext
from ..kernel.canonical import jsonable
from ..solvers.base import SolverError, get_solver
from .clients import AgentError, ChatClient, make_client
from .replies import extract_json, require
from .tools import ResearchTools

LANGUAGE = ("Write observations, hypotheses, reasons and rationales in Simplified Chinese; keep parameter paths, "
            "column names and JSON keys in English.")


def _short(value: Any, limit: int = 4000) -> str:
    text = json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + " …(truncated)"


def history_summary(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for report in history:
        entry: dict[str, Any] = {
            "generation": report["generation"],
            "candidates": [
                {"changes": item["edit"]["changes"], "rationale": item["edit"].get("rationale", "")[:300],
                 "screening_delta": round(item["delta"], 4) if "delta" in item else None,
                 "error": item.get("error")}
                for item in report.get("screening", [])
            ],
            "outcome": report.get("outcome"),
        }
        decision = report.get("decision")
        if decision:
            entry["holdout"] = {
                "delta": round(decision["delta"], 4),
                "interval": [round(value, 4) for value in decision["interval"]],
                "promoted": decision["promoted"],
                "failed_cells": [cell["cell"] for cell in decision["cells"] if not cell["passed"]],
            }
        summary.append(entry)
    return summary


def briefing(context: ProposalContext, k: int) -> str:
    task, gates = context.task, context.task.gates
    spec = get_solver(context.incumbent.solver)
    rules = (
        f"Each generation you may submit up to {k} edits of the current method (the incumbent). The host fits "
        "every candidate on calibration_fit and forecasts model_selection. Candidates are ranked on this "
        "generation's screening days; the best one is compared with the incumbent on separate holdout days that "
        f"nobody sees beforehand. Promotion needs an overall skill gain above {gates.min_delta}, a "
        f"{gates.confidence:.0%} day-block bootstrap interval above zero, and no credible regression in any cell "
        f"(a cell may lose at most {gates.cell_cap:.0%} normalized RMSE). validation and final_test are sealed. "
        "Forecasts are computed by the host from the method's parameters; you never forecast numbers yourself."
    )
    sections = [
        "You are a senior ecological forecasting scientist working in ecogym, a reproducible training ground.",
        f"## Task\n{task.title} ({task.id}). Targets: {', '.join(task.target_names)}; horizons (steps): "
        f"{list(task.horizons)}; season: {task.season_steps} steps; scored origins have {task.history_steps} steps "
        f"of complete history. Inputs may come from roles {list(task.exogenous_roles)}. Skill per cell = "
        "(RMSE_base - RMSE) / max(RMSE, RMSE_base) against the naive baseline chosen on calibration_fit.",
        f"## How proposals are judged\n{rules}",
    ]
    if task.notes:
        sections.append(f"## Task notes\n{task.notes.strip()}")
    if task.knowledge:
        sections.append(f"## Domain knowledge\n{task.knowledge.strip()}")
    sections += [
        f"## Method schema ({spec.ref})\n{spec.doc}\n\nParameters: {_short(spec.describe()['params'], 6000)}",
        f"## Current method\n{_short(context.incumbent.params, 6000)}",
        f"## Trust region\nEach edited numeric coordinate may move at most {context.max_step} of its range "
        f"(log scale for wide positive ranges); an edit may change at most {context.max_changes} coordinates or "
        f"structured entries. Coordinates now: {_short(context.axes(), 4000)}",
        f"## Previous generations\n{_short(history_summary(context.history), 6000) if context.history else 'none'}",
        f"## Language\n{LANGUAGE}",
    ]
    return "\n\n".join(sections)


RESEARCHER = """## Your role: researcher
Study the data and the incumbent's errors, then write a research note. Prefer mechanisms (physics, physiology,
control logic) over curve fitting, and check a hypothesis with try_edit before recommending it.
Each reply is exactly one JSON object, either a tool call
  {"tool": "<name>", "args": {...}, "why": "<one sentence>"}
or, when you are done, the note
  {"note": {"observations": ["..."], "hypotheses": ["..."], "directions": ["..."]}}
You have %(budget)d tool calls. Tools: %(tools)s"""

PROPOSER = """## Your role: proposer
Turn the research note into %(k)d concrete edits of the current method (fewer only if you cannot find %(k)d
well-founded ones). Only the best edit on the screening days goes on to the holdout comparison, so make the edits
explore different directions (different cells, parameters or mechanisms), not small variations of one idea.
Reply with exactly one JSON object:
  {"edits": [{"changes": {"<path>": <value>}, "rationale": "<why, citing the note>"}]}
A path is a parameter name ("alpha"), a per-cell parameter "<name>.<cell>", or a key of a mapping parameter
("<name>.<key>"); a structured value replaces the whole entry and must follow the schema.

Research note:
%(note)s"""

REVIEWER = """## Your role: reviewer
Check the proposed edits before the host scores them. Drop an edit only for a concrete reason: it contradicts the
evidence in the note, would use information not available at the forecast origin, duplicates another edit in
effect, or the host check below reports a problem. Reply with exactly one JSON object:
  {"reviews": [{"index": <i>, "keep": true or false, "reason": "..."}]}

Research note:
%(note)s

Proposed edits with the host's trust-region check:
%(edits)s"""


NOTE_KEYS = ("observations", "hypotheses", "directions")


def _as_note(reply: dict[str, Any]) -> dict[str, Any]:
    """Accept a note sent without its {"note": ...} wrapper (common after a repair request)."""

    if "note" not in reply and "tool" not in reply and any(key in reply for key in NOTE_KEYS):
        return {"note": {key: reply.get(key, []) for key in NOTE_KEYS}}
    return reply


@dataclass
class _Conversation:
    role: str
    client: ChatClient
    context: ProposalContext
    calls: list[int]
    max_calls: int
    messages: list[dict[str, str]] = field(default_factory=list)
    logged: int = 0

    def ask(self, content: str) -> dict[str, Any]:
        self.messages.append({"role": "user", "content": content})
        for attempt in range(2):
            if self.calls[0] >= self.max_calls:
                raise AgentError(f"the generation's budget of {self.max_calls} LLM calls is spent")
            self.calls[0] += 1
            reply = self.client.complete(self.messages)
            self.messages.append({"role": "assistant", "content": reply.text})
            if self.context.log is not None:
                self.context.log({"role": self.role, "call": self.calls[0], "model": reply.model,
                                  "seconds": reply.seconds, "usage": reply.usage,
                                  "messages": self.messages[self.logged:]})
                self.logged = len(self.messages)
            try:
                return extract_json(reply.text)
            except AgentError as error:
                if attempt:
                    raise
                self.messages.append({"role": "user",
                                      "content": f"Your reply was not usable ({error}). Reply again with only the "
                                                 "JSON object."})
        raise AgentError("unreachable")


@dataclass
class LLMProposer:
    """Researcher -> proposer -> reviewer over one chat transport (``http`` or ``dsh``)."""

    transport: str = "http"
    model: str = ""
    base_url: str = ""
    key_env: str = "ECOGYM_LLM_API_KEY"
    temperature: float | None = 0.4
    max_tokens: int = 16000
    timeout: float = 300.0
    stream: bool = True
    retries: int = 4
    min_interval: float = 0.0
    max_backoff: float = 300.0
    proxy: str = "system"
    fallbacks: tuple[dict[str, Any], ...] = ()
    max_tool_calls: int = 5
    max_calls: int = 10
    review: bool = True
    dsh_bin: str | None = None
    dsh_home: str | None = None
    patches: tuple[str, ...] = ()
    path_prepend: str | None = None
    name: str = field(default="llm", init=False)
    client: ChatClient | None = field(default=None, repr=False)

    def options(self) -> dict[str, Any]:
        """Constructor options (no secrets: keys are named by ``key_env``, never stored)."""

        skip = {"name", "client"}
        return {item.name: getattr(self, item.name) for item in fields(self) if item.name not in skip}

    def _client(self) -> ChatClient:
        if self.client is None:
            self.client = make_client(self.options())
        return self.client

    def observe(self, report: Mapping[str, Any]) -> None:
        return None

    def propose(self, context: ProposalContext, k: int) -> list[Edit]:
        """One research round; a failed round is logged and yields no candidates."""

        if context.bench is None or context.screening is None:
            raise AgentError("the llm proposer needs the run's bench and screening cohort")
        try:
            return self._round(context, k)
        except AgentError as error:
            if context.log is not None:
                context.log({"role": "proposer", "error": f"{type(error).__name__}: {error}"[:2000]})
            return []

    def _round(self, context: ProposalContext, k: int) -> list[Edit]:
        client = self._client()
        calls = [0]
        tools = ResearchTools(context.bench, context.incumbent, context.screening, max_step=context.max_step,
                              max_changes=context.max_changes)
        brief = briefing(context, k)
        note = self._research(_Conversation("researcher", client, context, calls, self.max_calls), brief, tools)
        if context.log is not None:
            context.log({"role": "researcher", "note": note})
        note_text = json.dumps(note, ensure_ascii=False, indent=1)
        proposer = _Conversation("proposer", client, context, calls, self.max_calls,
                                 [{"role": "system", "content": brief}])
        reply = proposer.ask(PROPOSER % {"k": k, "note": note_text})
        edits = []
        for item in require(reply, "edits", list)[:k]:
            try:
                edit = edit_from_dict(item)
            except SolverError:
                continue
            edits.append(Edit(edit.changes, edit.rationale, {"author": "llm", "model": client.describe()["model"]}))
        if self.review and edits:
            edits = self._review(_Conversation("reviewer", client, context, calls, self.max_calls,
                                               [{"role": "system", "content": brief}]), context, note_text, edits)
        return edits

    def _research(self, conversation: _Conversation, brief: str, tools: ResearchTools) -> dict[str, Any]:
        conversation.messages.append({"role": "system", "content": brief})
        prompt = RESEARCHER % {"budget": self.max_tool_calls, "tools": _short(tools.catalog())}
        overview = _short(tools.call("overview", {}), 6000)
        reply = conversation.ask(f"{prompt}\n\nGeneration {conversation.context.generation}: begin. "
                                 f"Result of overview (already run for you):\n{overview}")
        for _ in range(self.max_tool_calls):
            reply = _as_note(reply)
            if "note" in reply:
                break
            if "tool" not in reply:
                reply = conversation.ask("Reply with a tool call or the note, as one JSON object.")
                continue
            result = tools.call(str(reply["tool"]), reply.get("args") or {})
            reply = conversation.ask(f"Tool result ({reply['tool']}):\n{_short(result, 6000)}")
        reply = _as_note(reply)
        if "note" not in reply:
            reply = _as_note(conversation.ask("The tool budget is spent. Reply now with the note as one JSON object."))
        note = require(reply, "note", dict)
        return {key: note.get(key, []) for key in ("observations", "hypotheses", "directions")}

    def _review(self, conversation: _Conversation, context: ProposalContext, note: str,
                edits: list[Edit]) -> list[Edit]:
        checked = []
        for index, edit in enumerate(edits):
            try:
                child = apply_edit(context.incumbent, edit, context.task)
                problems = check_edit(context.incumbent, child, max_step=context.max_step,
                                      max_changes=context.max_changes)
            except SolverError as error:
                problems = [str(error)]
            checked.append({"index": index, "changes": edit.changes, "rationale": edit.rationale,
                            "host_check": problems or "ok"})
        reply = conversation.ask(REVIEWER % {"note": note, "edits": _short(checked, 8000)})
        dropped = {}
        for review in require(reply, "reviews", list):
            if isinstance(review, Mapping) and review.get("keep") is False and isinstance(review.get("index"), int):
                dropped[review["index"]] = str(review.get("reason", ""))
        if context.log is not None:
            context.log({"role": "reviewer", "dropped": [
                {"changes": edits[index].changes, "reason": reason} for index, reason in sorted(dropped.items())
                if 0 <= index < len(edits)
            ]})
        return [
            Edit(edit.changes, edit.rationale, {**edit.meta, "review": "kept"})
            for index, edit in enumerate(edits)
            if index not in dropped
        ]


__all__ = ["LLMProposer", "briefing", "history_summary"]
