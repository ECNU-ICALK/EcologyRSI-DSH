"""Expert participation without waiting: questions, answers and notes, all in the ledger.

Agents leave ``expert_question`` events in their own runs (episodes and
evolution rounds) and keep working. An expert answers whenever convenient
(``ecogym expert answer``, or the workbench) and may add notes unprompted;
both are ``expert_answered`` / ``expert_note`` events of the ``expert`` run.
Every later episode on a matching task receives the relevant answers and
notes in ``data/expert.md``; the harness evolver sees those of its training
tasks.

A note's scope is ``*`` (every task), a dataset id, a task card id or a task
reference (``<card>:<episode>``). An answer applies to the dataset of the
task whose agent asked, to that task alone when the asking episode verified
a holdout task (nothing learned there may reach the training tasks), and to
every task when an evolver asked. ``until`` (a ledger sequence number) fixes
what one evolution round sees, so all of its episodes get the same guidance.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from ..kernel.ledger import Ledger
from ..task.spec import TaskSpec, base_id, task_spec

RUN = "expert"
TEXT_LIMIT = 8000


def questions(ledger: Ledger) -> list[dict[str, Any]]:
    """Every question with its asker (task and role of the asking episode), and its answers; newest first."""

    answers: dict[str, list[dict[str, Any]]] = {}
    for event in ledger.events(RUN, kind="expert_answered"):
        answers.setdefault(event.body["question"], []).append({"text": event.body["text"], "at": event.at,
                                                               "by": event.body.get("by", ""), "seq": event.seq})
    found = []
    askers: dict[str, tuple[str | None, str | None]] = {}
    for event in ledger.events(kind="expert_question"):
        if event.run_id not in askers:
            first = ledger.first(event.run_id)
            body = first.body if first is not None and first.kind == "episode_started" else {}
            askers[event.run_id] = (body.get("task_id"), (body.get("context") or {}).get("role"))
        task_id, role = askers[event.run_id]
        ident = f"{event.run_id}#{event.seq}"
        found.append({"id": ident, "run_id": event.run_id, "task_id": task_id, "role": role, "at": event.at,
                      "question": event.body.get("question", ""), "context": event.body.get("context", ""),
                      "answers": answers.get(ident, [])})
    return list(reversed(found))


def answer(ledger: Ledger, question: str, text: str, *, by: str = "") -> dict[str, Any]:
    known = {item["id"] for item in questions(ledger)}
    if question not in known:
        raise KeyError(f"no question {question}")
    if not text.strip():
        raise ValueError("an answer needs text")
    body = {"question": question, "text": text.strip()[:TEXT_LIMIT], "by": by[:200], "at_unix": time.time()}
    return {"seq": ledger.append(RUN, "expert_answered", body).seq, **body}


def add_note(ledger: Ledger, scope: str, text: str, *, by: str = "") -> dict[str, Any]:
    if not scope.strip() or not text.strip():
        raise ValueError("a note needs a scope and text")
    body = {"scope": scope.strip(), "text": text.strip()[:TEXT_LIMIT], "by": by[:200], "at_unix": time.time()}
    return {"seq": ledger.append(RUN, "expert_note", body).seq, **body}


def notes(ledger: Ledger) -> list[dict[str, Any]]:
    return [{"seq": event.seq, "at": event.at, **event.body} for event in ledger.events(RUN, kind="expert_note")]


def _scopes(task: TaskSpec) -> set[str]:
    return {"*", task.dataset, base_id(task.id), task.id}


def _dataset(task_id: str, cache: dict[str, str | None]) -> str | None:
    if task_id not in cache:
        try:
            cache[task_id] = task_spec(task_id).dataset
        except (KeyError, ValueError):
            cache[task_id] = None
    return cache[task_id]


def guidance(ledger: Ledger, tasks: TaskSpec | Sequence[TaskSpec] | None, *,
             until: int | None = None) -> list[dict[str, Any]]:
    """Notes and answers that apply to ``tasks`` (one task or several; None: all), recorded by ``until``."""

    targets = None if tasks is None else [tasks] if isinstance(tasks, TaskSpec) else list(tasks)
    scopes = None if targets is None else set().union(*(_scopes(task) for task in targets))
    entries = []
    for note in notes(ledger):
        if (until is None or note["seq"] <= until) and (scopes is None or note["scope"] in scopes):
            entries.append({"kind": "note", "scope": note["scope"], "text": note["text"], "at": note["at"]})
    datasets: dict[str, str | None] = {}
    for item in questions(ledger):
        replies = [reply for reply in item["answers"] if until is None or reply["seq"] <= until]
        if not replies:
            continue
        asked_on = item["task_id"]
        if targets is not None and asked_on is not None:  # None: an evolver asked, the answer is general
            if item["role"] == "holdout":
                if asked_on not in {task.id for task in targets}:
                    continue
            elif _dataset(asked_on, datasets) not in {task.dataset for task in targets}:
                continue
        for reply in replies:
            entries.append({"kind": "answer", "question": item["question"], "text": reply["text"], "at": reply["at"]})
    return entries


def render(entries: list[dict[str, Any]]) -> str:
    lines = ["# 专家意见", "", "领域专家留下的提示，以及对以往回合中问题的回答。它们是经验判断，仍要用数据检验。", ""]
    for entry in entries:
        if entry["kind"] == "note":
            lines += [f"## 提示（{entry['at'][:10]}，范围 {entry['scope']}）", "", entry["text"], ""]
        else:
            lines += [f"## 问：{entry['question'][:300]}", "", f"答（{entry['at'][:10]}）：{entry['text']}", ""]
    return "\n".join(lines)


__all__ = ["RUN", "add_note", "answer", "guidance", "notes", "questions", "render"]
