"""A recorded session as readable steps: what the agent was told, what it thought, called and got back.

The steps come from the session's model calls (``llm_calls.jsonl.gz``, rebuilt into conversation threads by
``episodes.export.threads``) and pass through ``episodes.export.sanitize`` first: scratch paths become
``/workspace`` and tokens ``<redacted>``. Long texts keep their head and tail. The panel shows them as a
session's process; the evolver reads them, with tighter limits, as ``transcript.md``.

A thread is ``{"label", "calls", "steps"}``; a step is one of

    {"kind": "system", "chars"}                    the system prompt (left out)
    {"kind": "context", "text", "chars"}           context DSH injects (AGENTS.md, the skill list, reminders), named
    {"kind": "prompt", "text"}                     a message from the host or a parent agent
    {"kind": "reasoning" | "reply", "step", "text"}
    {"kind": "call", "step", "name", "id", "text", "target"}
    {"kind": "result", "name", "id", "text", "data"?}

``step`` counts the assistant's turns in the thread; ``target`` is what a call works on (a file, a command, a tool's
arguments) in one line; ``data`` is a tool result that is a JSON object (the ecogym tools answer in JSON). With
``arguments=True`` a call also carries its parsed ``arguments``, uncut, for code that reads what the agent did
(``evolution.process``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .export import sanitize, threads

INJECTED = ("<system-reminder>", "Current runtime context")
FOLLOW_UP = re.compile(r"^这是同一个(?:回合|评审会话)的第")  # how the host opens a follow-up session
PANEL_LIMITS = {"prompt": 8000, "reasoning": 6000, "reply": 8000, "call": 8000, "result": 6000}
DATA_LIMIT = 50_000  # characters: a longer tool result is not parsed into ``data``
TARGET_LIMIT = 300


def cut(text: Any, limit: int) -> str:
    """Head and tail of a long text (what an agent read first and what it ended on)."""

    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return f"{text[:head]}\n…（中间略去 {len(text) - limit} 字）…\n{text[len(text) - (limit - head):]}"


def message_text(content: Any) -> str:
    """A message's content: a string, or a list of parts."""

    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def call_arguments(function: Mapping[str, Any]) -> tuple[str, dict[str, Any], str]:
    """A tool call's name, its arguments (parsed; {} when they are not a JSON object) and the arguments written
    out for reading, long strings (code, commands, file contents) on lines of their own."""

    name, raw = str(function.get("name") or "?"), function.get("arguments")
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return name, {}, str(raw)
    if not isinstance(arguments, dict):
        return name, {}, json.dumps(arguments, ensure_ascii=False)
    lines = []
    for key, value in arguments.items():
        if isinstance(value, str) and ("\n" in value or len(value) > 120):
            lines.append(f"[{key}]\n{value}")
        else:
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    return name, arguments, "\n".join(lines)


def call_target(name: str, arguments: Mapping[str, Any]) -> str:
    """What a call works on, in one line: the file it reads or writes, the command it runs, else its arguments."""

    for key in ("file_path", "path", "command", "pattern", "query", "queries"):
        value = arguments.get(key)
        if value:
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            return " ".join(text.split())[:TARGET_LIMIT]
    text = json.dumps(dict(arguments), ensure_ascii=False) if arguments else ""
    return text[:TARGET_LIMIT]


def injected_label(text: str) -> str:
    """What a piece of DSH-injected context is (AGENTS.md, the skill list, runtime facts, a reminder)."""

    if "Instructions from: AGENTS.md" in text:
        return "AGENTS.md（Harness 的指令和经验）"
    if "available_skills" in text:
        return "技能列表"
    if text.startswith("Current runtime context"):
        return "运行环境说明"
    plain = " ".join(text.replace("<system-reminder>", "").replace("</system-reminder>", "").split())
    return "提醒：" + (plain if len(plain) <= 160 else plain[:160] + "…")


def thread_label(index: int, messages: list[dict[str, Any]]) -> str:
    first = next((message_text(item.get("content")) for item in messages if item.get("role") == "user"
                  and not message_text(item.get("content")).lstrip().startswith(INJECTED)), "")
    if FOLLOW_UP.match(first):
        return "续跑会话"
    return "主会话" if index == 0 else "其他线程（子智能体，或上下文压缩后的续写）"


def session_threads(path: Path, roots: Iterable[str] = (), limits: Mapping[str, int] | None = None, *,
                    arguments: bool = False) -> list[dict[str, Any]]:
    """The threads of the session whose model calls are logged at ``path`` ([] when there are none or the log
    cannot be read); ``roots`` are the session's scratch directories, ``limits`` overrides ``PANEL_LIMITS``."""

    if not path.is_file():
        return []
    try:
        found = threads(path)
    except (OSError, EOFError, ValueError):  # a damaged log costs this view only
        return []
    limits = {**PANEL_LIMITS, **(limits or {})}
    roots = [root for root in roots if root]
    result = []
    for index, chain in enumerate(found):
        messages = sanitize(chain["messages"], roots)
        steps: list[dict[str, Any]] = []
        called: dict[str, str] = {}
        turn = 0
        for message in messages:
            role, text = message.get("role"), message_text(message.get("content"))
            if role == "system":
                steps.append({"kind": "system", "chars": len(text)})
            elif role == "user" and text.lstrip().startswith(INJECTED):
                steps.append({"kind": "context", "text": injected_label(text.strip()), "chars": len(text)})
            elif role == "user":
                steps.append({"kind": "prompt", "text": cut(text, limits["prompt"])})
            elif role == "assistant":
                turn += 1
                if message.get("reasoning_content"):
                    steps.append({"kind": "reasoning", "step": turn,
                                  "text": cut(message["reasoning_content"], limits["reasoning"])})
                if text.strip():
                    steps.append({"kind": "reply", "step": turn, "text": cut(text, limits["reply"])})
                for call in message.get("tool_calls") or []:
                    name, parsed, written = call_arguments(call.get("function") or {})
                    ident = str(call.get("id"))
                    called[ident] = name
                    step = {"kind": "call", "step": turn, "name": name, "id": ident,
                            "text": cut(written, limits["call"]), "target": call_target(name, parsed)}
                    if arguments:
                        step["arguments"] = parsed
                    steps.append(step)
            elif role == "tool":
                ident = str(message.get("tool_call_id"))
                item: dict[str, Any] = {"kind": "result", "name": called.get(ident, message.get("name") or ""),
                                        "id": ident, "text": cut(text, limits["result"])}
                if len(text) <= DATA_LIMIT and text.lstrip().startswith("{"):
                    try:
                        data = json.loads(text)
                    except ValueError:
                        data = None
                    if isinstance(data, dict):
                        item["data"] = data
                steps.append(item)
        result.append({"label": thread_label(index, messages), "calls": chain["calls"], "steps": steps})
    return result


__all__ = ["PANEL_LIMITS", "call_arguments", "call_target", "cut", "injected_label", "message_text",
           "session_threads", "thread_label"]
