"""Harness versions: what the agent brings to a task, versioned by content.

A harness is a directory:

    harness.toml            name, parent, author, summary; [agent] DSH options
    instructions.md         working rules (the workspace's AGENTS.md)
    skills/<name>/SKILL.md  DSH skills, optionally with more files
    template/               the starting submission: predictor.py and helper modules
    memory/lessons.md       experience carried between episodes

Its digest covers every file's relative path and bytes, so equal content is
the same version; ``behavior_digest`` leaves out the descriptive fields of
``harness.toml``, for telling whether two versions behave the same. The
[agent] table holds the DSH settings a harness may
choose (model, reasoning effort, web search, sub-agent depth, parallel tool
calls); safety settings (sandbox, network, credentials, disabled tools,
scoring) are not part of a harness and cannot be changed by evolving one.
``name``, ``parent`` and ``author`` are written by the host (``normalize``).
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import os
import re
import shutil
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from ..kernel.config import ConfigError, Fields, load_toml

SUFFIXES = {".md", ".py", ".toml", ".txt", ".json", ".yaml", ".yml", ".csv"}
MAX_FILES = 300
MAX_BYTES = 4_000_000
MODEL = re.compile(r"^deepseek-[a-z0-9.-]{1,40}$")
EFFORTS = ("off", "low", "high", "max")
SUMMARY_LIMIT = 2000


class HarnessError(ValueError):
    """A harness directory is malformed or breaks a limit."""


@dataclass(frozen=True, slots=True)
class AgentOptions:
    model: str = "deepseek-v4-pro"
    reasoning_effort: str = "high"
    web_search: bool = True
    subagent_depth: int = 1
    parallel_tool_calls: int = 10

    @classmethod
    def read(cls, agent: Fields) -> AgentOptions:
        default = cls()  # slots dataclass: class attributes are descriptors, not defaults
        options = cls(
            model=agent.str("model", default.model),
            reasoning_effort=agent.str("reasoning_effort", default.reasoning_effort, choices=EFFORTS),
            web_search=agent.bool("web_search", default.web_search),
            subagent_depth=agent.int("subagent_depth", default.subagent_depth, lo=0, hi=2),
            parallel_tool_calls=agent.int("parallel_tool_calls", default.parallel_tool_calls, lo=1, hi=16),
        )
        agent.done()
        if not MODEL.match(options.model):
            raise HarnessError(f"model {options.model!r} is not a DeepSeek model id")
        return options


@dataclass(frozen=True, slots=True)
class Harness:
    root: Path
    name: str
    parent: str
    author: str
    summary: str
    agent: AgentOptions
    digest: str

    @property
    def short(self) -> str:
        return self.digest[:12]

    @classmethod
    def load(cls, root: str | Path) -> Harness:
        root = Path(root)
        files = content_files(root)
        try:
            meta = Fields(load_toml(root / "harness.toml"), "harness.toml")
            options = AgentOptions.read(meta.table("agent"))
            summary = meta.raw("summary", "")
            if not isinstance(summary, str):
                raise HarnessError("harness.toml summary must be a string")
            harness = cls(
                root=root,
                name=meta.str("name"),
                parent=str(meta.raw("parent", "")),
                author=meta.str("author", "agent"),
                summary=summary,
                agent=options,
                digest=digest_files(root, files),
            )
            meta.done()
        except ConfigError as error:
            raise HarnessError(str(error)) from None
        for required in ("instructions.md", "template/predictor.py"):
            if not (root / required).is_file():
                raise HarnessError(f"{root}: missing {required}")
        return harness

    def install(self, episode_root: Path) -> None:
        """Lay the harness out for one episode: AGENTS.md, skills and the starting submission."""

        work = episode_root / "work"
        instructions = (self.root / "instructions.md").read_text(encoding="utf-8").strip()
        lessons = self.root / "memory" / "lessons.md"
        if lessons.is_file():
            instructions += "\n\n" + lessons.read_text(encoding="utf-8").strip()
        (work / "AGENTS.md").write_text(instructions + "\n", encoding="utf-8")
        skills = episode_root / "harness" / "skills"
        skills.parent.mkdir(parents=True, exist_ok=True)
        if (self.root / "skills").is_dir():
            shutil.copytree(self.root / "skills", skills, dirs_exist_ok=True)
        else:
            skills.mkdir(exist_ok=True)
        shutil.copytree(self.root / "template", work / "submission", dirs_exist_ok=True)

    def describe(self) -> dict[str, Any]:
        return {"digest": self.digest, "name": self.name, "parent": self.parent, "author": self.author,
                "summary": self.summary, "agent": dataclasses.asdict(self.agent)}


def content_files(root: Path) -> list[Path]:
    """Every file of a harness, relative to its root; rejects links and oversize content."""

    if not (root / "harness.toml").is_file():
        raise HarnessError(f"{root} has no harness.toml")
    files, total = [], 0
    for directory, names, filenames in os.walk(root, followlinks=False):
        for name in names:
            if os.path.islink(os.path.join(directory, name)):
                raise HarnessError(f"{Path(directory) / name}: links are not allowed in a harness")
        names[:] = sorted(name for name in names if not name.startswith(".") and name != "__pycache__")
        for name in sorted(filenames):
            path = Path(directory) / name
            if name.startswith("."):
                continue
            if path.is_symlink():
                raise HarnessError(f"{path}: links are not allowed in a harness")
            if path.suffix not in SUFFIXES:
                raise HarnessError(f"{path.relative_to(root)}: file type {path.suffix or '(none)'} is not allowed")
            total += path.stat().st_size
            files.append(path.relative_to(root))
    if len(files) > MAX_FILES or total > MAX_BYTES:
        raise HarnessError(f"{root}: {len(files)} files, {total} bytes exceed {MAX_FILES} files / {MAX_BYTES} bytes")
    return files


def digest_files(root: Path, files: list[Path]) -> str:
    sha = hashlib.sha256(b"ecogym.harness/1\n")
    for relative in files:
        data = (root / relative).read_bytes()
        sha.update(f"{relative.as_posix()}\n{len(data)}\n".encode())
        sha.update(data)
    return sha.hexdigest()


def behavior_digest(harness: Harness) -> str:
    """What reaches the agent: every content file except ``harness.toml``, plus the ``[agent]`` options.

    ``name``, ``parent`` and ``author`` are host-written and ``summary`` only describes, so two versions
    with equal behavior get equal digests even when they were proposed in different rounds.
    """

    sha = hashlib.sha256(b"ecogym.harness-behavior/1\n")
    sha.update(json.dumps(dataclasses.asdict(harness.agent), sort_keys=True).encode())
    for relative in content_files(harness.root):
        if relative.as_posix() == "harness.toml":
            continue
        data = (harness.root / relative).read_bytes()
        sha.update(f"\n{relative.as_posix()}\n{len(data)}\n".encode())
        sha.update(data)
    return sha.hexdigest()


def fork(harness: Harness, target: Path) -> Path:
    """Copy a harness's content files into ``target`` (created), for editing."""

    target.mkdir(parents=True, exist_ok=True)
    for relative in content_files(harness.root):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(harness.root / relative, destination)
    return target


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)  # a JSON string is a valid TOML basic string


def write_meta(root: Path, *, name: str, parent: str, author: str, summary: str, agent: AgentOptions) -> None:
    lines = [f"name = {_toml(name)}", f"parent = {_toml(parent)}", f"author = {_toml(author)}",
             f"summary = {_toml(summary)}", "", "[agent]"]
    lines += [f"{key} = {_toml(value)}" for key, value in dataclasses.asdict(agent).items()]
    (root / "harness.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize(root: Path, *, name: str, parent: str, author: str) -> Harness:
    """Rewrite an edited harness.toml with host-owned name, parent and author; keep summary and [agent]."""

    path = root / "harness.toml"
    if not path.is_file() or path.is_symlink():
        raise HarnessError("harness.toml is missing")
    try:
        meta = Fields(load_toml(path), "harness.toml")
        options = AgentOptions.read(meta.table("agent"))
        summary = meta.raw("summary", "")
        for key in ("name", "parent", "author"):
            meta.raw(key, "")
        meta.done()
    except ConfigError as error:
        raise HarnessError(str(error)) from None
    if not isinstance(summary, str):
        raise HarnessError("harness.toml summary must be a string")
    write_meta(root, name=name, parent=parent, author=author, summary=summary.strip()[:SUMMARY_LIMIT], agent=options)
    return Harness.load(root)


def diff(before: Path, after: Path, *, context: int = 2, limit: int = 60_000) -> list[dict[str, Any]]:
    """Per-file changes from ``before`` to ``after`` (both harness directories) with unified diffs."""

    old = {path.as_posix() for path in content_files(before)}
    new = {path.as_posix() for path in content_files(after)}
    changes, budget = [], limit
    for relative in sorted(old | new):
        a = (before / relative).read_text(encoding="utf-8", errors="replace").splitlines() if relative in old else []
        b = (after / relative).read_text(encoding="utf-8", errors="replace").splitlines() if relative in new else []
        if relative in old and relative in new and a == b:
            continue
        status = "added" if relative not in old else "removed" if relative not in new else "changed"
        lines = list(difflib.unified_diff(a, b, f"a/{relative}", f"b/{relative}", n=context, lineterm=""))
        patch = "\n".join(lines)
        if len(patch) > budget:
            patch = patch[: max(budget, 0)] + "\n…(truncated)"
        budget -= len(patch)
        changes.append({"path": relative, "status": status,
                        "added": sum(1 for line in lines if line.startswith("+") and not line.startswith("+++")),
                        "removed": sum(1 for line in lines if line.startswith("-") and not line.startswith("---")),
                        "patch": patch})
    return changes


def term_problems(root: Path, terms: Iterable[str]) -> list[str]:
    """Files that name any of ``terms`` (case-insensitive, whole words): task-specific content."""

    words = sorted({term for term in terms if term and len(term) >= 3}, key=str.lower)
    if not words:
        return []
    pattern = re.compile("|".join(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])" for word in words),
                         re.IGNORECASE)
    problems = []
    for relative in content_files(root):
        text = (root / relative).read_text(encoding="utf-8", errors="replace")
        found = sorted({match.group(0) for match in pattern.finditer(text)}, key=str.lower)
        if found:
            problems.append(f"{relative.as_posix()} names task-specific terms: {', '.join(found[:5])}")
    return problems


def store_root() -> Path:
    home = Path(os.environ.get("ECOGYM_HOME") or Path.home() / ".ecogym").expanduser()
    return home / "harnesses"


def store(harness: Harness) -> Harness:
    """Keep an immutable copy under ``$ECOGYM_HOME/harnesses/<digest>``."""

    target = store_root() / harness.digest
    if not target.exists():
        staging = target.with_name(f"{target.name}.{os.getpid()}-{threading.get_ident()}.tmp")
        shutil.rmtree(staging, ignore_errors=True)
        staging.parent.mkdir(parents=True, exist_ok=True)
        fork(harness, staging)
        try:
            staging.rename(target)
        except OSError:  # another process stored the same version first
            shutil.rmtree(staging, ignore_errors=True)
    stored = Harness.load(target)
    if stored.digest != harness.digest:
        raise HarnessError(f"stored harness {target} does not match its digest")
    return stored


def seed() -> Harness:
    with resources.as_file(resources.files("ecogym.harness").joinpath("seed")) as path:
        return store(Harness.load(path))


def resolve(ref: str) -> Harness:
    """``seed``, a directory, or a (prefix of a) stored digest."""

    if ref == "seed":
        return seed()
    path = Path(ref).expanduser()
    if path.is_dir():
        return store(Harness.load(path))
    matches = sorted(store_root().glob(f"{ref}*")) if re.fullmatch(r"[0-9a-f]{4,64}", ref) else []
    matches = [item for item in matches if item.is_dir() and not item.name.endswith(".tmp")]
    if len(matches) != 1:
        raise HarnessError(f"harness {ref!r} matches {len(matches)} stored versions")
    return Harness.load(matches[0])


__all__ = ["EFFORTS", "AgentOptions", "ConfigError", "Harness", "HarnessError", "behavior_digest", "content_files",
           "diff", "fork", "normalize", "resolve", "seed", "store", "term_problems", "write_meta"]
