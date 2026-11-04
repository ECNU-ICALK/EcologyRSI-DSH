"""What one generation's evolver did, read back from its recorded session: what it looked at in its workspace,
what it asked the ecogym tools and what they answered, which steps changed candidate files, step by step.

The evolver works in ``work/`` (``evolver.run_evolver``): ``EVOLVE.md`` (the brief), ``AGENTS.md`` (its guide),
``evidence/`` (``evidence.write_evidence``; one folder per episode), ``archive/`` (earlier versions it may start
from), ``incumbent/``, ``candidates/<name>/`` and ``proposals/<name>.md``. Its transcript is sanitized
(``episodes.transcripts``), so the workspace shows as ``/workspace/work``. Which files a shell command read or
wrote is read off the command line, so that part is a best guess; the steps themselves are as recorded.

The panel's evolver page shows this; the run page shows the reads of each generation (``steps=False``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..episodes.transcripts import PANEL_LIMITS, cut, session_threads

ECOGYM = "mcp__ecogym__"
LAYOUT = ("evidence", "archive", "incumbent", "candidates", "proposals")
PATH = re.compile(r"(?<![\w.-])(?:/workspace/work/)?((?:EVOLVE|AGENTS)\.md\b|(?:" + "|".join(LAYOUT)
                  + r")/[A-Za-z0-9_.*?@+{}/-]*)")
REDIRECT = re.compile(r"(?<![0-9&>])>>?\s*['\"]?(?:/workspace/work/)?([A-Za-z0-9_.*?@+{}/-]+)")
TEE = re.compile(r"\btee\s+(?:-a\s+)?['\"]?(?:/workspace/work/)?([A-Za-z0-9_.*?@+{}/-]+)")
# scripts run from the shell (python3 - <<EOF): a variable bound to a path, files opened or written
ASSIGNED = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*(?:Path\(\s*)?f?(['\"])(.+?)\2")
OPENED = re.compile(r"\bopen\(\s*(?:f?(['\"])(.+?)\1|([A-Za-z_]\w*))\s*,\s*['\"][wax]\+?b?['\"]")
WRITTEN = re.compile(r"(?:\bPath\(\s*(?:f?(['\"])(.+?)\1|([A-Za-z_]\w*))\s*\)|\b([A-Za-z_]\w*))"
                     r"\.write_(?:text|bytes)\(")
IN_PLACE = re.compile(r"\bsed\s+-i|\bperl\s+-p?i")
COPIES = re.compile(r"(?:^|[;&|]\s*)(?:cp|mv|rsync)\b")
FILE_TOOLS = ("edit", "write", "multi_edit")
LOOK_TOOLS = ("read", "grep", "glob", "ls")
LISTINGS = (*LAYOUT, "evidence/episodes")  # folders whose listing says nothing about what was read
RESULT_TEXT = 600


def workspace_paths(text: str) -> list[str]:
    """The paths into the evolver's workspace that a piece of text (a command, a file argument) names."""

    found = []
    for match in PATH.finditer(text):
        path = match.group(1).rstrip("./")
        if path and path not in found:
            found.append(path)
    return found


def classify(path: str) -> tuple[str, str]:
    """(kind, key) of a workspace path: ``("transcript", "ep:…")``, ``("archive", "35ad33-g1b")``, ``("evidence",
    "overview.md")``, ``("candidate", "a")``, ``("incumbent", "template/predictor.py")``…"""

    parts = path.split("/")
    head, rest = parts[0], parts[1:]
    if head in ("EVOLVE.md", "AGENTS.md"):
        return ("brief" if head == "EVOLVE.md" else "guide"), head
    if head == "evidence":
        if rest[:1] == ["episodes"] and len(rest) > 1:
            folder = rest[1]
            episode = folder.replace("-", ":", 1) if folder.startswith("ep-") else folder
            leaf = rest[2] if len(rest) > 2 else ""
            kinds = {"transcript.md": "transcript", "summary.md": "summary", "submission": "program"}
            return kinds.get(leaf, "episode"), episode
        return "evidence", "/".join(rest) or "evidence"
    if head == "archive":
        return "archive", rest[0] if rest else "archive"
    if head == "incumbent":
        return "incumbent", "/".join(rest) or "incumbent"
    if head == "candidates":
        return "candidate", rest[0] if rest else "candidates"
    if head == "proposals":
        return "proposal", (rest[0] if rest else "proposals").removesuffix(".md")
    return "other", path


def _output(path: str) -> bool:
    """A file of a candidate or a proposal (not the folders that hold them)."""

    depth = {"candidates": 3, "proposals": 2}.get(path.split("/")[0], 0)
    return bool(depth) and len(path.split("/")) >= depth


def _script_writes(command: str) -> tuple[bool, list[str]]:
    """Whether a script in the command writes files, and the paths it writes (a literal, or a variable bound to
    one: ``p=f'candidates/{name}/instructions.md' … open(p, 'w')``)."""

    bound: dict[str, list[str]] = {}
    for name, _, value in ASSIGNED.findall(command):
        bound.setdefault(name, []).append(value)
    found, targets = False, []
    for match in (*OPENED.finditer(command), *WRITTEN.finditer(command)):
        found = True
        literal, variable = match.group(2), match.group(3) or (match.group(4) if match.re is WRITTEN else None)
        targets += [literal] if literal else bound.get(variable or "", [])
    return found, [path for target in targets for path in workspace_paths(target)]


def _shell_writes(command: str) -> list[str]:
    """The candidate and proposal files a shell command may write: the targets of its redirections, ``tee`` and
    the scripts it runs; every one it names for ``sed -i`` (or a script whose targets cannot be read off); the
    last one for ``cp``/``mv``."""

    named = [path for path in workspace_paths(command) if _output(path)]
    written = [match.group(1).rstrip("./") for pattern in (REDIRECT, TEE) for match in pattern.finditer(command)]
    scripted, targets = _script_writes(command)
    written += targets
    if IN_PLACE.search(command) or (scripted and not any(_output(path) for path in targets)):
        written += named
    elif COPIES.search(command) and named:
        written.append(named[-1])  # cp/mv: the last path is the destination
    return [path for path in dict.fromkeys(written) if _output(path)]


def _condensed(tool: str, data: dict[str, Any] | None, text: str) -> dict[str, Any]:
    """What an ecogym tool answered, without what the panel shows elsewhere (full diffs, proposals)."""

    if data is None:  # a tool error is plain text
        return {"error": cut(text, RESULT_TEXT)} if text.strip() else {}
    if tool == "check_candidate":
        smoke = data.get("smoke") or {}
        return {"name": data.get("name"), "ok": data.get("ok"), "status": data.get("status"),
                "digest": data.get("digest"), "summary": data.get("summary"), "base": data.get("base"),
                "problems": data.get("problems") or [], "warnings": data.get("warnings") or [],
                "changes": [{key: item.get(key) for key in ("path", "status", "added", "removed")}
                            for item in data.get("changes") or [] if isinstance(item, dict)],
                "smoke": {key: smoke.get(key) for key in ("status", "score", "incumbent_score", "coverage", "tasks")
                          if key in smoke} if smoke else None}
    if tool == "reset_candidate":
        return {key: data.get(key) for key in ("name", "base", "digest", "summary")}
    if tool == "screen_templates":  # the per-task scores stay in the transcript
        return {"incumbent_mean": data.get("incumbent_mean"), "left": data.get("left"),
                "variants": [{key: item.get(key) for key in ("name", "status", "mean", "gain", "better", "worse",
                                                              "worst", "problem") if item.get(key) is not None}
                             for item in data.get("variants") or [] if isinstance(item, dict)]}
    if tool == "task_info":
        return {key: data.get(key) for key in ("minutes_left", "tokens_used", "tokens_left", "candidates")}
    return {key: (cut(value, RESULT_TEXT) if isinstance(value, str) else value) for key, value in data.items()}


def summarize(threads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """``reads``: every workspace item the session looked at, in the order it first did, with the steps that did
    (``[thread, step]``); ``timeline``: its ecogym tool calls and the steps that changed candidate files, in order.

    Call steps need their ``arguments`` (``session_threads(..., arguments=True)``)."""

    reads: dict[tuple[str, str], dict[str, Any]] = {}
    timeline: list[dict[str, Any]] = []

    def wrote(where: list[int], tool: str, paths: list[str], *, certain: bool, description: str = "") -> None:
        last = timeline[-1] if timeline else {}
        if (last.get("kind") == "write" and last["where"] == where and last["tool"] == tool
                and last["paths"] == paths):
            last["count"] += 1  # five edits of one file in one step read as one line
            return
        timeline.append({"where": where, "kind": "write", "tool": tool, "paths": paths, "certain": certain,
                         "count": 1, **({"description": description} if description else {})})

    for index, thread in enumerate(threads):
        results = {item["id"]: item for item in thread["steps"] if item["kind"] == "result"}
        for item in thread["steps"]:
            if item["kind"] != "call":
                continue
            where = [index, item["step"]]
            name, arguments = item["name"], item.get("arguments") or {}
            result = results.get(item["id"]) or {}
            looked: list[str] = []
            if name.startswith(ECOGYM):
                tool = name.removeprefix(ECOGYM)
                timeline.append({"where": where, "kind": "tool", "tool": tool,
                                 "arguments": {key: (cut(value, RESULT_TEXT) if isinstance(value, str) else value)
                                               for key, value in arguments.items()},
                                 "result": _condensed(tool, result.get("data"), result.get("text", ""))})
            elif name in FILE_TOOLS:
                paths = workspace_paths(str(arguments.get("file_path") or arguments.get("path") or ""))
                if paths:
                    wrote(where, name, paths, certain=True)
            elif name == "bash":
                command = str(arguments.get("command") or "")
                written = _shell_writes(command)
                if written:
                    wrote(where, name, written, certain=False,
                          description=str(arguments.get("description") or "")[:200])
                looked = [path for path in workspace_paths(command) if path not in written]
            elif name in LOOK_TOOLS:
                looked = [path for value in arguments.values() if isinstance(value, str)
                          for path in workspace_paths(value)]
            for path in looked:
                if path in LISTINGS:
                    continue
                kind, key = classify(path)
                entry = reads.setdefault((kind, key), {"kind": kind, "key": key, "paths": [], "steps": []})
                if path not in entry["paths"]:
                    entry["paths"].append(path)
                if where not in entry["steps"]:
                    entry["steps"].append(where)
    return {"reads": list(reads.values()), "timeline": timeline}


SESSION_FIELDS = ("status", "seconds", "tokens", "metered_tokens", "llm_calls", "answered_calls", "failed_calls",
                  "continuations", "error", "upstream_error", "stalled", "stalled_seconds", "activity", "rerun")


def evolver_session(view: dict[str, Any], generation: int, *, steps: bool = True) -> dict[str, Any] | None:
    """The evolver session of ``generation`` in the run ``view`` (``evolution.fold``): the round's facts, the
    ``summarize`` of its session and, with ``steps``, its threads (``episodes.transcripts``). None when the
    generation had no evolver session (generation 0, or a round that ended before one finished)."""

    rounds = [round_ for round_ in view["rounds"] if round_["generation"] == generation and round_.get("evolver")]
    if not rounds:
        return None
    round_ = rounds[-1]  # a round interrupted and run again: the last attempt
    evolver = round_["evolver"]
    directory = str((evolver.get("trajectory") or {}).get("dir") or "")
    found = session_threads(Path(directory) / "llm_calls.jsonl.gz", (), PANEL_LIMITS, arguments=True) \
        if directory else []
    summary = summarize(found)
    for thread in found:
        for item in thread["steps"]:
            item.pop("arguments", None)  # its text holds the same, written out
    result = {"run_id": view["run_id"], "generation": generation, "incumbent": round_["incumbent"],
              "started_at": round_["started_at"], "session": {key: evolver.get(key) for key in SESSION_FIELDS},
              "final_text": evolver.get("final_text") or "", "notes": round_["notes"],
              "candidates": [{key: candidate.get(key) for key in ("name", "status", "digest", "summary", "base")}
                             for candidate in round_["candidates"]],
              "turns": len({(index, item["step"]) for index, thread in enumerate(found) for item in thread["steps"]
                            if "step" in item}),
              "recorded": bool(found), **summary}
    if steps:
        result["threads"] = found
    return result


__all__ = ["classify", "evolver_session", "summarize", "workspace_paths"]
