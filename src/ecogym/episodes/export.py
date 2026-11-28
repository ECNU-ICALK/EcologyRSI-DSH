"""Export recorded agent sessions as training data.

Sources are the episodes and evolver sessions in the ledger whose trajectory
is stored. From a session's model calls (``llm_calls.jsonl.gz``) the
conversation threads are rebuilt: a call continues a thread when its request
starts with the thread's last request. Each thread becomes one multi-turn SFT
sample (system, user, assistant turns with reasoning and tool calls, tool
results), ending with the model's last reply in that thread; sub-agents and
post-compaction continuations are threads of their own.

Files written to the target directory:

    sft.jsonl          {"id", "messages", "tools", "meta"} for clean threads that pass the filters:
                       an episode thread contains the exact accepted submission scored by the judge, and the
                       episode ended normally and scored above ``min_score`` (0 = the naive baseline);
                       an evolver session proposed a valid candidate
    preferences.jsonl  {"id", "prompt", "chosen", "rejected", "meta"}: two episodes of one task with the same
                       prompt (same harness) whose hidden scores differ credibly (paired day-block bootstrap) and
                       by at least ``margin``; with ``cross_prompt`` also pairs whose prompts differ, which then
                       carry the rejected episode's own prompt as ``rejected_prompt``
    episodes.jsonl     one line per episode: task, harness, model, scores, status, trajectory files
    notes.jsonl        every note the agents and evolvers wrote: the experience library
    manifest.json      counts, filters and what was skipped and why

Episodes of an independent validation (``context.role`` ``holdout``: a run's
final check on tasks no evolving agent saw), and with ``holdout_of`` every
episode on a holdout task of those suites, are no training data: they never
enter ``sft.jsonl``, ``preferences.jsonl`` or ``notes.jsonl``. With
``include_failed`` (the audit export) their threads and notes are written,
marked ``independent_validation``, and still never pass the filters or pair.
``episodes.jsonl`` lists every episode with that mark.

A model call repeated with the same request (a retried stream) replaces the
earlier reply instead of starting a second thread, and identical threads are
written once. Scratch paths of a session are replaced by ``/workspace``. API
keys never reach the trajectories: the gateway records request bodies, never
headers.
"""

from __future__ import annotations

import gzip
import itertools
import json
import os
import re
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .. import __version__
from ..evaluation.harness import Bench
from ..evaluation.suite import TaskPair, compare_suite
from ..kernel.ledger import Ledger
from ..task.spec import task_spec
from ..task.suite import SuiteGates, suite_spec
from .gateway import assemble
from .history import episode_records
from .judge import load_predictions
from .sandbox import scratch_root

SESSION_DIR = r"/(?:ep|evo|seal)-[A-Za-z0-9_.-]+"
# episode roles of independent validations (``evolution.verify.HOLDOUT``; this layer does not import evolution)
VALIDATION_ROLES = frozenset({"holdout"})
# Per-session tokens reach the agent's own environment and DSH patch (it needs them to call the gateway and the
# tool server); they die with the session, but credentials never belong in training data.
REDACT = (
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]{12,}"), r"\1<redacted>"),
    (re.compile(r"((?:API_KEY|API_TOKEN|SECRET)[A-Z_]*\s*[=:]\s*\\?[\"']?)[A-Za-z0-9._~+/=-]{12,}"), r"\1<redacted>"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}"), "<redacted-key>"),
)
MESSAGE_KEYS = ("role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name")
SUBMIT_TOOL = "mcp__ecogym__submit"
COMPLETE_REPLIES = ("stop", "tool_calls", "function_call")
PAIR_GATES = SuiteGates(resamples=1000)
MAX_PAIRS_PER_TASK = 50


def _calls(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _reply(call: dict[str, Any]) -> dict[str, Any] | None:
    reply = call.get("reply")
    if reply is None and isinstance(call.get("response"), str) and call["response"].startswith("data:"):
        reply = assemble(call["response"], "text/event-stream")  # trajectories recorded before replies were assembled
    if not isinstance(reply, dict) or not isinstance(reply.get("message"), dict):
        return None
    message = reply["message"]
    if not message.get("content") and not message.get("tool_calls"):
        return None
    return reply


def _clean(message: dict[str, Any]) -> dict[str, Any]:
    return {key: message[key] for key in MESSAGE_KEYS if message.get(key) not in (None, "", [])} | (
        {"content": message.get("content") or ""} if message.get("role") == "assistant" else {})


def threads(path: Path) -> list[dict[str, Any]]:
    """Conversation threads of one session's model calls, oldest first."""

    chains: list[dict[str, Any]] = []
    for call in _calls(path):
        request = call.get("request")
        if not str(call.get("path", "")).endswith("/chat/completions") or call.get("status") != 200:
            continue
        if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
            continue
        reply = _reply(call)
        if reply is None:
            continue
        messages = [_clean(message) for message in request["messages"] if isinstance(message, dict)]
        keys = [json.dumps(message, sort_keys=True, ensure_ascii=False) for message in messages]
        # the same request again (a retried call) continues its thread too: its reply replaces the earlier one
        chain = next((item for item in reversed(chains)
                      if len(keys) >= len(item["keys"]) and keys[: len(item["keys"])] == item["keys"]), None)
        if chain is None:
            chain = {"calls": 0, "first_seq": call.get("seq")}
            chains.append(chain)
        chain.update(keys=keys, messages=[*messages, _clean(reply["message"])], tools=request.get("tools") or [],
                     model=call.get("model"), calls=chain["calls"] + 1, last_seq=call.get("seq"),
                     complete=not call.get("error") and reply.get("finish_reason") in COMPLETE_REPLIES)
    unique, seen = [], set()
    for chain in chains:
        text = json.dumps(chain["messages"], sort_keys=True, ensure_ascii=False)
        if text not in seen:
            seen.add(text)
            unique.append(chain)
    return unique


def scratch_pattern() -> re.Pattern[str]:
    """Session directories under the default and the configured scratch root, spelled with and without
    macOS's ``/private`` prefix (``/tmp`` and ``/var`` are links into it)."""

    roots = {"/tmp/ecogym", str(scratch_root()), os.environ.get("ECOGYM_SCRATCH") or ""}
    roots = {root.rstrip("/") for root in roots if root.strip("/")}
    roots |= {root.removeprefix("/private") for root in roots if root.startswith("/private/")}
    roots |= {"/private" + root for root in roots if root.startswith(("/tmp/", "/var/"))}
    escaped = "|".join(re.escape(json.dumps(root, ensure_ascii=False)[1:-1])
                       for root in sorted(roots, key=len, reverse=True))
    return re.compile(f"(?:{escaped}){SESSION_DIR}")


def _normalize(value: Any, roots: Iterable[str], scratch: re.Pattern[str]) -> Any:
    """Scratch paths -> ``/workspace``; tokens and keys -> ``<redacted>``."""

    text = json.dumps(value, ensure_ascii=False)
    for root in sorted({root for root in roots if root}, key=len, reverse=True):
        text = text.replace(json.dumps(root, ensure_ascii=False)[1:-1], "/workspace")
    text = scratch.sub("/workspace", text)
    for pattern, replacement in REDACT:
        text = pattern.sub(replacement, text)
    return json.loads(text)


def sanitize(value: Any, roots: Iterable[str] = ()) -> Any:
    """What may leave a session: its scratch paths (``roots`` and every session directory) become
    ``/workspace``, tokens and keys ``<redacted>``. Used for training data and for the evolver's transcripts."""

    return _normalize(value, roots, scratch_pattern())


def _sessions(ledger: Ledger) -> Iterator[dict[str, Any]]:
    """Episodes and evolver sessions that have a stored trajectory, with their metadata."""

    for record in episode_records(ledger):
        finished = record.get("finished") or {}
        trajectory = (finished.get("trajectory") or {}).get("dir")
        if trajectory:
            yield {"kind": "episode", "id": record["episode"], "record": record, "trajectory": Path(trajectory),
                   "root": record.get("root") or ""}
    for run_id in ledger.runs(first="evolve_started"):
        for event in ledger.events(run_id, kind="evolver_finished"):
            trajectory = (event.body.get("trajectory") or {}).get("dir")
            if trajectory:
                yield {"kind": "evolver", "id": f"{run_id}/g{event.body['generation']}", "run": run_id,
                       "body": event.body, "trajectory": Path(trajectory), "root": ""}


def _evolver_outcomes(ledger: Ledger) -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = {}
    for run_id in ledger.runs(first="evolve_started"):
        for event in ledger.events(run_id):
            if event.kind in ("candidate_compared", "harness_adopted"):
                entry = outcomes.setdefault(f"{run_id}/g{event.body['generation']}", {"adopted": None, "deltas": {}})
                if event.kind == "harness_adopted":
                    entry["adopted"] = event.body["name"]
                elif event.body["decision"].get("delta") is not None:
                    entry["deltas"][event.body["name"]] = event.body["decision"]["delta"]
    return outcomes


def _scored_submission(record: dict[str, Any]) -> dict[str, Any] | None:
    scored = record.get("scored") or {}
    number, digest = scored.get("n"), scored.get("digest")
    if (scored.get("start") or scored.get("auto") or type(number) is not int
            or not isinstance(digest, str) or not digest):
        return None
    return next((item for item in record["submissions"] if item.get("accepted") is True
                 and item.get("n") == number and item.get("digest") == digest), None)


def _submission_calls(messages: list[dict[str, Any]], submission: dict[str, Any]) -> set[str]:
    """IDs of actual submit calls whose tool response confirms the ledger's number and program digest."""
    submitted, matched = set(), set()
    digest = submission["digest"]
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if (call.get("function") or {}).get("name") == SUBMIT_TOOL and call.get("id"):
                    submitted.add(call["id"])
        elif message.get("role") == "tool" and message.get("tool_call_id") in submitted:
            content = message.get("content")
            if not isinstance(content, str):
                continue
            try:
                reply = json.loads(content)
            except ValueError:
                continue
            if (isinstance(reply, dict) and reply.get("accepted") is True and reply.get("n") == submission["n"]
                    and reply.get("digest") in (digest, digest[:12])):
                matched.add(message["tool_call_id"])
    return matched


def validation_use(record: dict[str, Any], holdout_tasks: frozenset[str] = frozenset()) -> str | None:
    """Why an episode is no training data, or None: it ran in an independent validation (its role), or on a holdout
    task of a suite the export is for. Training on either would weaken the validation that reuses those tasks."""

    role = (record.get("context") or {}).get("role")
    if role in VALIDATION_ROLES:
        return f"role:{role}"
    if record.get("task_id") in holdout_tasks:
        return "holdout_task"
    return None


def _episode_meta(record: dict[str, Any]) -> dict[str, Any]:
    scored, finished = record.get("scored") or {}, record.get("finished") or {}
    references = scored.get("references") or {}
    score = scored.get("score")
    return {
        "kind": "episode", "session": record["episode"], "task": record["task_id"],
        "harness": (record.get("harness") or {}).get("digest"), "model": record.get("model"),
        "status": record["status"], "score": score, "interval": scored.get("interval"),
        "references": references, "context": record.get("context") or {},
        "beats": {name: score > value for name, value in references.items()} if isinstance(score, float) else {},
        "tokens": finished.get("tokens"), "seconds": finished.get("seconds"),
        "submissions": len(record["submissions"]),
        "accepted_submissions": sum(1 for item in record["submissions"] if item.get("accepted")),
        # None: no submission was scored; the judge scored the starting program (start) or, in records from
        # before the starting program was judged, a snapshot of submission/ (auto)
        "scored_submission": scored.get("n"),
        "auto": bool(scored.get("auto")),
        "start": bool(scored.get("start")),
    }


def _holdout_evolver_runs(ledger: Ledger, holdout_tasks: frozenset[str]) -> frozenset[str]:
    """Evolution runs whose training tasks overlap ``holdout_tasks`` (evolver saw those hidden scores)."""

    if not holdout_tasks:
        return frozenset()
    tainted: set[str] = set()
    for run_id in ledger.runs(first="evolve_started"):
        first = ledger.first(run_id)
        train = set(((first.body.get("suite") or {}).get("train") or []) if first is not None else [])
        if train & holdout_tasks:
            tainted.add(run_id)
    return frozenset(tainted)


def export(ledger_path: Path, target: Path, *, min_score: float | None = 0.0, include_failed: bool = False,
           evolver: bool = True, margin: float = 0.02, cross_prompt: bool = False,
           holdout_of: Sequence[str] = ()) -> dict[str, Any]:
    """``min_score`` None drops the score filter (the accepted-submission requirement stays). ``holdout_of``:
    suites whose holdout tasks this training data must not touch, whatever role their episodes had."""

    target.mkdir(parents=True, exist_ok=True)
    counts = {"sessions": 0, "sft": 0, "preferences": 0, "episodes": 0, "notes": 0}
    skipped: dict[str, int] = {}
    scratch = scratch_pattern()
    holdout_tasks = frozenset(task.id for suite in holdout_of for task in suite_spec(suite).tasks("holdout"))

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    credited_threads: dict[str, dict[str, Any]] = {}
    scored_episodes: dict[str, list[dict[str, Any]]] = {}
    with Ledger(ledger_path) as ledger, \
            (target / "sft.jsonl").open("w", encoding="utf-8") as sft, \
            (target / "episodes.jsonl").open("w", encoding="utf-8") as episodes, \
            (target / "notes.jsonl").open("w", encoding="utf-8") as notes:
        outcomes = _evolver_outcomes(ledger)
        tainted_evolver_runs = _holdout_evolver_runs(ledger, holdout_tasks)
        for session in _sessions(ledger):
            counts["sessions"] += 1
            calls = session["trajectory"] / "llm_calls.jsonl.gz"
            submission = None
            if session["kind"] == "episode":
                record = session["record"]
                meta = _episode_meta(record)
                validation = validation_use(record, holdout_tasks)
                meta["independent_validation"] = validation
                episodes.write(json.dumps({**meta, "trajectory": str(session["trajectory"]),
                                           "files": (record["finished"].get("trajectory") or {}).get("files")},
                                          ensure_ascii=False) + "\n")
                counts["episodes"] += 1
                if validation and not include_failed:
                    skip(f"independent validation episode ({validation})")
                    continue
                for note in record["notes"]:
                    notes.write(json.dumps({"source": "episode", "session": record["episode"],
                                            "task": record["task_id"], "harness": meta["harness"],
                                            "kind": note.get("kind"), "text": note.get("text"),
                                            "sources": note.get("sources", []), "at": note.get("at"),
                                            "score": meta["score"],
                                            **({"independent_validation": validation} if validation else {})},
                                           ensure_ascii=False) + "\n")
                    counts["notes"] += 1
                good = meta["status"] in ("finished", "timeout") and validation is None
                submission = _scored_submission(record)
                local_score = ((submission or {}).get("local") or {}).get("score")
                start_score = ((record.get("start") or {}).get("local") or {}).get("score")
                local_gain = (local_score - start_score if isinstance(local_score, (float, int))
                              and isinstance(start_score, (float, int)) else None)
                passes = good and submission is not None and (
                    min_score is None or (isinstance(meta["score"], float) and meta["score"] > min_score))
            else:
                if not evolver:
                    continue
                body = session["body"]
                outcome = outcomes.get(session["id"], {"adopted": None, "deltas": {}})
                validation = "holdout_task" if session["run"] in tainted_evolver_runs else None
                meta = {"kind": "evolver", "session": session["id"], "run": session["run"],
                        "generation": body["generation"], "status": body.get("status"),
                        "candidates": body.get("candidates", []), "adopted": outcome["adopted"],
                        "deltas": outcome["deltas"], "tokens": body.get("tokens"), "seconds": body.get("seconds"),
                        "independent_validation": validation}
                if validation and not include_failed:
                    skip(f"independent validation evolver ({validation})")
                    continue
                proposed = any(item.get("status") == "proposed" for item in body.get("candidates", []))
                passes = validation is None and body.get("status") in ("finished", "timeout") and proposed
            if not calls.is_file():
                skip("no model-call log")
                continue
            if not (passes or include_failed):
                skip(f"{session['kind']} below the filters")
                continue
            try:
                found = threads(calls)
            except (OSError, EOFError, ValueError):
                skip("unreadable model-call log")
                continue
            links = [_submission_calls(chain["messages"], submission) if submission else set() for chain in found]
            call_ids = set().union(*links)
            tool_id = next(iter(call_ids)) if len(call_ids) == 1 else None
            if passes and submission and not any(chain["complete"] and tool_id in link
                                                 for chain, link in zip(found, links, strict=True)):
                skip("episode without an attributable scored submission thread")
            for index, chain in enumerate(found):
                thread_meta = {}
                thread_passes = passes and chain["complete"]
                if session["kind"] == "episode":
                    credited = chain["complete"] and tool_id in links[index]
                    thread_passes = thread_passes and credited
                    thread_meta = {"credited_tool_call_id": tool_id if credited else None,
                                   "episode_score": meta["score"],
                                   "episode_scored_submission": meta["scored_submission"],
                                   "local_gain_vs_start": local_gain if credited else None}
                    if not credited:
                        thread_meta.update(score=None, interval=None, beats={}, scored_submission=None)
                if not (thread_passes or include_failed):
                    continue
                sample = {"id": f"{session['id']}/{index}",
                          "messages": _normalize(chain["messages"], [session["root"]], scratch),
                          "tools": _normalize(chain["tools"], [session["root"]], scratch),
                          "meta": {**meta, **thread_meta, "thread": index, "threads": len(found),
                                   "model_calls": chain["calls"], "complete_reply": chain["complete"],
                                   "passes_filters": thread_passes,
                                   "assistant_turns": sum(1 for item in chain["messages"]
                                                          if item.get("role") == "assistant")}}
                sft.write(json.dumps(sample, ensure_ascii=False) + "\n")
                counts["sft"] += 1
                if session["kind"] == "episode" and thread_passes and session["id"] not in credited_threads:
                    scored = record.get("scored") or {}
                    if isinstance(meta["score"], float) and scored.get("predictions"):
                        credited_threads[session["id"]] = {"messages": sample["messages"]}
                        scored_episodes.setdefault(record["task_id"], []).append(
                            {**sample["meta"], "predictions": str(session["trajectory"] / scored["predictions"]),
                             "predictions_digest": scored.get("predictions_digest")})
        if evolver:
            for run_id in ledger.runs(first="evolve_started"):
                validation = "holdout_task" if run_id in tainted_evolver_runs else None
                if validation and not include_failed:
                    continue
                for event in ledger.events(run_id, kind="evolver_note"):
                    notes.write(json.dumps({"source": "evolver", "session": f"{run_id}/g{event.body['generation']}",
                                            "kind": event.body.get("kind"), "text": event.body.get("text"),
                                            "at": event.at,
                                            **({"independent_validation": validation} if validation else {})},
                                           ensure_ascii=False) + "\n")
                    counts["notes"] += 1
    counts["preferences"] = _preferences(target / "preferences.jsonl", scored_episodes, credited_threads, margin, skip,
                                         cross_prompt=cross_prompt)
    manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ecogym": __version__,
                "ledger": str(ledger_path), "filters": {"min_score": min_score, "include_failed": include_failed,
                                                        "evolver": evolver, "margin": margin,
                                                        "cross_prompt": cross_prompt,
                                                        "independent_validation": "marked" if include_failed
                                                        else "excluded",
                                                        "holdout_of": list(holdout_of),
                                                        "holdout_tasks": sorted(holdout_tasks)},
                "counts": counts, "skipped": skipped}
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


def _split(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first = next((index for index, item in enumerate(messages) if item.get("role") == "assistant"), len(messages))
    return messages[:first], messages[first:]


def _preferences(path: Path, scored: dict[str, list[dict[str, Any]]], main: dict[str, dict[str, Any]],
                 margin: float, skip: Any, *, cross_prompt: bool = False) -> int:
    written = 0
    with path.open("w", encoding="utf-8") as stream:
        for task_id, items in sorted(scored.items()):
            items = [item for item in items if item["session"] in main]
            if len(items) < 2:
                continue
            try:
                bench = Bench.open(task_spec(task_id))
                evaluations = {item["session"]: load_predictions(Path(item["predictions"]), bench,
                                                                 expect=item.get("predictions_digest"))
                               for item in items}
            except (OSError, KeyError, ValueError):
                skip("preference task without data or matching predictions")
                continue
            candidates = sorted(itertools.combinations(items, 2), key=lambda pair: -abs(pair[0]["score"]
                                                                                      - pair[1]["score"]))
            kept = 0
            for first, second in candidates:
                if kept >= MAX_PAIRS_PER_TASK or abs(first["score"] - second["score"]) < margin:
                    break
                chosen, rejected = (first, second) if first["score"] > second["score"] else (second, first)
                prompt, chosen_turns = _split(main[chosen["session"]]["messages"])
                other_prompt, rejected_turns = _split(main[rejected["session"]]["messages"])
                if prompt != other_prompt and not cross_prompt:
                    skip("preference pair with different prompts (different harnesses)")
                    continue
                decision = compare_suite([TaskPair(task_id, (evaluations[chosen["session"]],),
                                                   (evaluations[rejected["session"]],))],
                                         replace(PAIR_GATES, min_delta=margin), {"pair": [chosen["session"],
                                                                                          rejected["session"]]})
                if decision.interval[0] <= 0:
                    continue
                sample = {"id": f"{chosen['session']}>{rejected['session']}", "prompt": prompt,
                          "chosen": chosen_turns, "rejected": rejected_turns,
                          "meta": {"task": task_id, "chosen": chosen, "rejected": rejected, "delta": decision.delta,
                                   "interval": decision.interval, "same_prompt": prompt == other_prompt,
                                   "same_harness": chosen["harness"] == rejected["harness"]}}
                if prompt != other_prompt:
                    sample["rejected_prompt"] = other_prompt  # the context the rejected turns answered
                stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept += 1
                written += 1
    return written


__all__ = ["VALIDATION_ROLES", "export", "sanitize", "threads", "validation_use"]
