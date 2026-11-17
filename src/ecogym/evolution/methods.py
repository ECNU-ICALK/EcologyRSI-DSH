"""Host-approved, source-only external predictors and their local reference judge.

An approval snapshots an offline program and records where it came from. A source URL is an identity, not a
download instruction. A model or its dependencies that need a different runtime are outside this registry's
contract. Agents may look up approved records and request a host-managed copy, but cannot approve records.
"""

from __future__ import annotations

import json
import re
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..episodes.judge import Limits, judge_program
from ..episodes.mcp import ToolError
from ..episodes.sandbox import DENIED_ROOTS, Runtime, protected_roots, scratch_root
from ..episodes.tools import snapshot
from ..evaluation.harness import Bench, Evaluation
from ..harness import MAX_BYTES, MAX_FILES, SUFFIXES, digest_files
from ..kernel.canonical import digest
from ..protocol.partitions import DEVELOPMENT_STAGES

SCHEMA = "ecogym.external-method/1"
METHOD_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}")
DIGEST = re.compile(r"[a-f0-9]{64}")
FIELDS = ("id", "source", "license", "revision", "summary", "program_digest")


class MethodError(ValueError):
    """An external method was unavailable, incompatible or did not match its approved bytes."""


def canonical_source(source: str) -> str:
    """Normalize URL spelling only; different hosts, paths, revisions and queries stay distinct."""

    if not isinstance(source, str) or len(source.strip()) > 500:
        raise MethodError("the method source must be an HTTP(S) URL of at most 500 characters")
    text = source.strip()
    try:
        url = urlsplit(text)
        port = url.port
    except ValueError:
        raise MethodError("the method source must be a public HTTP(S) URL") from None
    if (url.scheme not in ("http", "https") or not url.hostname or url.username is not None
            or any(character.isspace() or ord(character) < 32 for character in text)):
        raise MethodError("the method source must be a public HTTP(S) URL without credentials")
    host = url.hostname
    netloc = f"[{host}]" if ":" in host else host
    default = 443 if url.scheme == "https" else 80
    if port is not None and port != default:
        netloc += f":{port}"
    return urlunsplit((url.scheme, netloc, url.path.rstrip("/"), url.query, ""))


def _program_digest(root: Path) -> str:
    if root.is_symlink() or not root.is_dir() or not (root / "predictor.py").is_file():
        raise MethodError("the approved method must have a predictor.py")
    files, total = [], 0
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or path.suffix not in SUFFIXES:
            raise MethodError(f"unsupported method source file {path.relative_to(root)}")
        files.append(path.relative_to(root))
        total += info.st_size
        if len(files) > MAX_FILES or total > MAX_BYTES:
            raise MethodError(f"method source exceeds {MAX_FILES} files or {MAX_BYTES // 1_000_000} MB")
    return digest_files(root, files)


@dataclass(frozen=True, slots=True)
class RegisteredMethod:
    digest: str
    program: Path
    id: str
    source: str
    license: str
    revision: str
    summary: str
    program_digest: str

    def describe(self) -> dict[str, str]:
        return {"digest": self.digest, **{key: getattr(self, key) for key in FIELDS}}


class MethodRegistry:
    """One host-selected registry. No method is supplied by default."""

    def __init__(self, root: Path, *, approved: frozenset[str] | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self._approved = approved

    def pin(self) -> MethodRegistry:
        """Keep membership fixed while continuing to verify the approved bytes on use."""

        return MethodRegistry(self.root, approved=frozenset(method.digest for method in self.list()))

    def _load(self, path: Path) -> RegisteredMethod:
        descriptor = path / "method.json"
        if path.is_symlink() or descriptor.is_symlink() or not descriptor.is_file():
            raise MethodError(f"method {path.name} has no approved manifest")
        try:
            body = json.loads(descriptor.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise MethodError(f"method {path.name} has an unreadable manifest") from error
        if (not isinstance(body, dict) or set(body) != {"schema", *FIELDS} or body["schema"] != SCHEMA
                or any(not isinstance(body[key], str) for key in FIELDS) or digest(body) != path.name):
            raise MethodError(f"method {path.name} does not match its approved manifest")
        program = path / "program"
        if _program_digest(program) != body["program_digest"]:
            raise MethodError(f"method {path.name} source has changed after approval")
        return RegisteredMethod(path.name, program, **{key: body[key] for key in FIELDS})

    def approve(self, method_id: str, program_dir: Path, *, source: str, license: str,
                revision: str = "", summary: str = "") -> RegisteredMethod:
        """Save a local source tree with host-approved provenance; does not run or install it."""

        if self._approved is not None:
            raise MethodError("cannot approve a method in a pinned registry")
        if not METHOD_ID.fullmatch(method_id) or DIGEST.fullmatch(method_id):
            raise MethodError("method id must be lowercase letters, digits, dots, hyphens or underscores")
        if not license.strip():
            raise MethodError("method approval needs a license")
        source = canonical_source(source)
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="approve-", dir=self.root) as temporary:
            staging = Path(temporary) / "method"
            program = staging / "program"
            program.mkdir(parents=True)
            try:
                snapshot(Path(program_dir), program)
            except ToolError as error:
                raise MethodError(str(error)) from error
            body = {"schema": SCHEMA, "id": method_id, "source": source, "license": license.strip(),
                    "revision": revision.strip(), "summary": summary.strip(),
                    "program_digest": _program_digest(program)}
            key = digest(body)
            descriptor = json.dumps(body, ensure_ascii=False, indent=2)
            (staging / "method.json").write_text(descriptor + "\n", encoding="utf-8")
            target = self.root / key
            try:
                staging.rename(target)
            except OSError:
                if not target.is_dir():
                    raise
        return self._load(target)

    def list(self) -> tuple[RegisteredMethod, ...]:
        if self._approved is not None:
            return tuple(sorted((self._load(self.root / ref) for ref in self._approved),
                                key=lambda method: (method.id, method.digest)))
        if not self.root.exists():
            return ()
        if not self.root.is_dir():
            raise MethodError("the method registry must be a directory")
        methods = [self._load(path) for path in self.root.iterdir() if DIGEST.fullmatch(path.name)]
        return tuple(sorted(methods, key=lambda method: (method.id, method.digest)))

    def available(self) -> bool:
        return bool(self.list())

    def cards(self) -> tuple[dict[str, str], ...]:
        return tuple(method.describe() for method in self.list())

    def identity(self) -> str:
        return digest({"schema": SCHEMA, "methods": sorted(method.digest for method in self.list())})

    def find_source(self, source: str) -> tuple[RegisteredMethod, ...]:
        url = canonical_source(source)
        return tuple(method for method in self.list() if method.source == url)

    def resolve(self, ref: str) -> RegisteredMethod:
        if DIGEST.fullmatch(ref):
            if self._approved is not None and ref not in self._approved:
                raise MethodError("method is outside this run's approved registry")
            return self._load(self.root / ref)
        found = [method for method in self.list() if method.id == ref]
        if len(found) != 1:
            raise MethodError(f"method id {ref!r} matches {len(found)} approvals; use an exact digest")
        return found[0]

    def check_agent_isolation(self, session_root: Path, runtime: Runtime) -> None:
        """Keep the registry outside agent paths, even when a custom location was selected."""

        denied = [Path(root).resolve() for root in DENIED_ROOTS] + protected_roots()
        exposed = [session_root.resolve(), *(root.resolve() for root in runtime.read_roots)]
        if (not any(self.root.is_relative_to(root) for root in denied)
                or any(self.root.is_relative_to(root) or root.is_relative_to(self.root) for root in exposed)):
            raise MethodError("place the method registry in a host-only directory under ECOGYM_HOME or ECOGYM_SCRATCH")

    def materialize(self, ref: str, target: Path) -> RegisteredMethod:
        """Copy the approved program, without its host manifest, into a fresh template directory."""

        method = self.resolve(ref)
        target = Path(target)
        if target.exists() or target.is_symlink():
            raise MethodError("method materialization needs a fresh target directory")
        with tempfile.TemporaryDirectory(prefix="method-copy-", dir=target.parent) as temporary:
            staging = Path(temporary) / "program"
            staging.mkdir()
            try:
                snapshot(method.program, staging)
            except ToolError as error:
                raise MethodError(str(error)) from error
            if _program_digest(staging) != method.program_digest:
                raise MethodError("method source changed while being copied")
            staging.rename(target)
        return method

    def judge(self, ref: str, bench: Bench, stage: str, runtime: Runtime, *, limits: Limits | None = None,
              cancel: threading.Event | None = None) -> tuple[dict[str, Any], Evaluation]:
        """Replay a registered method using the ordinary causal judge and scoring protocol."""

        if stage not in DEVELOPMENT_STAGES:
            raise MethodError("external method replay is available only on development stages")
        with tempfile.TemporaryDirectory(prefix="reference-", dir=scratch_root()) as temporary:
            program = Path(temporary) / "program"
            method = self.materialize(ref, program)
            report, evaluation = judge_program(bench, program, stage, runtime, limits=limits, cancel=cancel)
        return {**report, "method": method.describe(), "bench": bench.digest, "task": bench.task.id}, evaluation


__all__ = ["MethodError", "MethodRegistry", "RegisteredMethod", "canonical_source"]
