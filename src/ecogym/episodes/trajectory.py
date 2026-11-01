"""Collect an episode's trajectory: DSH sessions, model calls, tool calls, code and scores.

DSH writes each session (and each sub-agent's) as zstd-compressed JSON lines
under ``$DSH_HOME/sessions``. They are decoded with Node's zlib (Node runs DSH,
so it is always present) into one gzip file, one event per line, tagged with
its session id. The gateway's model calls and the MCP tool log are already
JSON lines; everything is stored under ``$ECOGYM_HOME/trajectories/<episode>``
with a digest per file.
"""

from __future__ import annotations

import collections
import contextlib
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

DECOMPRESS = r"""
const fs = require('fs'), zlib = require('zlib');
const buf = fs.readFileSync(process.argv[1]);
const magic = Buffer.from([0x28, 0xb5, 0x2f, 0xfd]);
let start = 0; const out = [];
while (start < buf.length) {
  let next = buf.indexOf(magic, start + 4);
  if (next < 0) next = buf.length;
  out.push(zlib.zstdDecompressSync(buf.subarray(start, next)));
  start = next;
}
process.stdout.write(Buffer.concat(out));
"""


def session_events(path: Path, node: Path) -> list[dict[str, Any]]:
    done = subprocess.run([str(node), "-e", DECOMPRESS, str(path)], capture_output=True, check=True, timeout=120)
    return [json.loads(line) for line in done.stdout.decode("utf-8", "replace").splitlines() if line.strip()]


def sessions(home: Path, node: Path) -> list[dict[str, Any]]:
    found = []
    for path in sorted(home.glob("sessions/*/*/session.v3.jsonl.zstd")):
        events = session_events(path, node)
        header = events[0] if events and events[0].get("type") == "session" else {}
        found.append({"id": header.get("id", path.parent.name), "parent": header.get("parentSession"),
                      "depth": header.get("delegationDepth", 0), "events": events})
    return found


def summarize(found: list[dict[str, Any]]) -> dict[str, Any]:
    tools: collections.Counter[str] = collections.Counter()
    searches: list[str] = []
    usage: collections.Counter[str] = collections.Counter()
    for session in found:
        for event in session["events"]:
            kind, data = event.get("type"), event.get("data") or {}
            if kind == "tool/call":
                tools[str(data.get("name"))] += 1
                if data.get("name") == "web_search":
                    with contextlib.suppress(json.JSONDecodeError, AttributeError):
                        searches.extend(json.loads(data.get("arguments") or "{}").get("queries") or [])
            elif kind == "assistant/message":
                found_usage = data.get("usage") or (data.get("message") or {}).get("usage") or {}
                for key, value in found_usage.items():
                    if isinstance(value, int):
                        usage[key] += value
    return {"sessions": len(found), "subagents": sum(1 for item in found if item["depth"]),
            "tool_calls": dict(tools.most_common()), "web_queries": searches[:200], "usage": dict(usage)}


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _no_links(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    return member if member.isfile() or member.isdir() else None


PROGRAM_SKIP = ("tmp/", "_ecogym", "__pycache__/")


def _members(archive: tarfile.TarFile, prefix: str, max_bytes: int) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for member in archive.getmembers():
        if not member.isfile() or not member.name.startswith(prefix) or member.size > max_bytes:
            continue
        relative = member.name[len(prefix):]
        if relative.startswith(PROGRAM_SKIP) or relative in ("stderr.log",) or relative.endswith(".pyc"):
            continue
        stream = archive.extractfile(member)
        if stream is not None:
            files[relative] = stream.read()
    return files


def _stored_submission(archive: tarfile.TarFile, digest: str | None, max_bytes: int) -> dict[str, bytes]:
    """The stored ``work/submission/`` as the host's snapshot takes it, if its digest is ``digest`` (else {}).

    An episode with no accepted submission is scored on a snapshot of ``work/submission/`` taken right after its
    session; the workspace is stored before that snapshot exists, so the program is read back from the stored
    directory, and the digest proves it is the one that was scored.
    """

    from .tools import ToolError, snapshot  # the tools import the judge, which imports this module's users

    with tempfile.TemporaryDirectory(prefix="ecogym-fallback-") as scratch:
        source, copy = Path(scratch) / "submission", Path(scratch) / "snapshot"
        for relative, data in _members(archive, "work/submission/", MAX_PROGRAM_BYTES).items():
            path = (source / relative).resolve()
            if path.is_relative_to(source.resolve()):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        try:
            if snapshot(source, copy) != digest:
                return {}
        except ToolError:
            return {}
        return {path.relative_to(copy).as_posix(): path.read_bytes() for path in sorted(copy.rglob("*"))
                if path.is_file() and path.stat().st_size <= max_bytes}


MAX_PROGRAM_BYTES = 20_000_000  # a submission's size limit (``ecogym.episodes.tools.MAX_BYTES``)


def submission_files(trajectory: Path, scored: dict[str, Any] | None, *, max_bytes: int = 400_000) -> dict[str, bytes]:
    """The files of the program an episode was scored with, read from its stored workspace."""

    archive_path = trajectory / "workspace.tar.gz"
    if not scored or not archive_path.is_file() or (scored.get("n") is None and not scored.get("auto")
                                                    and not scored.get("start")):
        return {}
    folder = "final-fallback" if scored.get("auto") else "start" if scored.get("start") else f"{int(scored['n']):03d}"
    with tarfile.open(archive_path) as archive:
        files = _members(archive, f"submissions/{folder}/program/", max_bytes)
        if not files and scored.get("auto"):
            files = _stored_submission(archive, scored.get("digest"), max_bytes)
    return files


def extract_submission(trajectory: Path, scored: dict[str, Any] | None, target: Path) -> Path:
    """Write the scored program into ``target`` (a fresh directory) for re-judging."""

    files = submission_files(trajectory, scored, max_bytes=MAX_PROGRAM_BYTES)
    if "predictor.py" not in files:
        raise FileNotFoundError(f"{trajectory}: the scored program is not in the stored workspace")
    target.mkdir(parents=True, exist_ok=False)
    for relative, data in files.items():
        path = (target / relative).resolve()
        if not path.is_relative_to(target.resolve()):
            continue  # never write outside the target, whatever the archive says
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return target


def store(root: Path, target: Path, node: Path | None) -> dict[str, Any]:
    """Write the trajectory files into ``target``; return per-file digests and a session summary."""

    target.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    if node is not None and (root / "home").is_dir():
        found = sessions(root / "home", node)
        summary = summarize(found)
        with gzip.open(target / "dsh_sessions.jsonl.gz", "wt", encoding="utf-8") as stream:
            for session in found:
                for event in session["events"]:
                    stream.write(json.dumps({"session": session["id"], "parent": session["parent"],
                                             "depth": session["depth"], **event}, ensure_ascii=False) + "\n")
    for name in ("llm_calls.jsonl", "mcp_calls.jsonl"):
        path = target / name
        if path.exists():
            with path.open("rb") as source, gzip.open(target / f"{name}.gz", "wb") as sink:
                shutil.copyfileobj(source, sink)
            path.unlink()
    with tarfile.open(target / "workspace.tar.gz", "w:gz") as archive:
        work = root / "work"
        for item in sorted(work.iterdir()) if work.is_dir() else []:
            if item.name != "data":
                archive.add(item, arcname=f"work/{item.name}", filter=_no_links)
        for folder in ("submissions", "evals"):
            if (root / folder).is_dir():
                archive.add(root / folder, arcname=folder, filter=_no_links)
    files = {path.name: _digest(path) for path in sorted(target.iterdir()) if path.is_file()}
    return {"files": files, "summary": summary}


__all__ = ["extract_submission", "session_events", "sessions", "store", "submission_files", "summarize"]
