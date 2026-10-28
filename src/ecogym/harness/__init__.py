"""Harness versions: what the agent brings to a task, versioned by content.

A harness is a directory:

    harness.toml            name, parent, author, summary; [agent] options
    instructions.md         working rules (the workspace's AGENTS.md)
    skills/<name>/SKILL.md  DSH skills, optionally with more files
    template/               the starting submission: predictor.py and helper modules
    memory/lessons.md       experience carried between episodes

Its digest covers every file's relative path and bytes, so equal content is
the same version. Safety settings (sandbox, network, credentials, disabled
tools) are not part of a harness and cannot be changed by evolving one.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ..kernel.config import ConfigError, Fields, load_toml

SUFFIXES = {".md", ".py", ".toml", ".txt", ".json", ".yaml", ".yml", ".csv"}
MAX_FILES = 300
MAX_BYTES = 4_000_000
MODEL = re.compile(r"^deepseek-[a-z0-9.-]{1,40}$")


class HarnessError(ValueError):
    """A harness directory is malformed or breaks a limit."""


@dataclass(frozen=True, slots=True)
class AgentOptions:
    model: str = "deepseek-v4-pro"
    web_search: bool = True
    subagent_depth: int = 1


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
        meta = Fields(load_toml(root / "harness.toml"), "harness.toml")
        agent = meta.table("agent")
        options = AgentOptions(
            model=agent.str("model", AgentOptions.model),
            web_search=agent.bool("web_search", AgentOptions.web_search),
            subagent_depth=agent.int("subagent_depth", AgentOptions.subagent_depth, lo=0, hi=2),
        )
        agent.done()
        if not MODEL.match(options.model):
            raise HarnessError(f"model {options.model!r} is not a DeepSeek model id")
        harness = cls(
            root=root,
            name=meta.str("name"),
            parent=meta.raw("parent", ""),
            author=meta.str("author", "agent"),
            summary=meta.raw("summary", ""),
            agent=options,
            digest=digest_files(root, files),
        )
        meta.done()
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


def content_files(root: Path) -> list[Path]:
    """Every file of a harness, relative to its root; rejects links and oversize content."""

    if not (root / "harness.toml").is_file():
        raise HarnessError(f"{root} has no harness.toml")
    files, total = [], 0
    for directory, names, filenames in os.walk(root, followlinks=False):
        names[:] = sorted(name for name in names if not name.startswith(".") and name != "__pycache__")
        for name in sorted(filenames):
            path = Path(directory) / name
            if name.startswith("."):
                continue
            if path.is_symlink():
                raise HarnessError(f"{path}: links are not allowed in a harness")
            if path.suffix not in SUFFIXES:
                raise HarnessError(f"{path}: file type {path.suffix or '(none)'} is not allowed")
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


def store_root() -> Path:
    home = Path(os.environ.get("ECOGYM_HOME") or Path.home() / ".ecogym").expanduser()
    return home / "harnesses"


def store(harness: Harness) -> Harness:
    """Keep an immutable copy under ``$ECOGYM_HOME/harnesses/<digest>``."""

    target = store_root() / harness.digest
    if not target.exists():
        staging = target.with_name(target.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        staging.parent.mkdir(parents=True, exist_ok=True)
        for relative in content_files(harness.root):
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(harness.root / relative, destination)
        staging.rename(target)
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


__all__ = ["AgentOptions", "ConfigError", "Harness", "HarnessError", "content_files", "resolve", "seed", "store"]
