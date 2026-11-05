"""Process confinement for agents and submitted programs (macOS ``sandbox-exec``).

Two Seatbelt profiles, both default-allow with explicit denials so ordinary
runtimes keep working:

* ``agent_profile``: the DSH agent and everything it spawns. Reads only its
  episode directory, the runtimes (Node, Python) and system locations; writes
  only its episode directory; connects only to the given loopback ports (the
  episode's LLM gateway and tool server). Everything else under /Users and
  /Volumes (raw data, the ledger, credentials, other episodes) is unreadable.
* ``program_profile``: a submitted predictor. Reads its own directory and the
  Python runtime, writes its own directory, no network at all. Hidden data
  reaches it only through the judge's pipe, one time step at a time.

Ancestors of allowed paths get metadata access only, so path resolution
works without listing directories. Linux (bubblewrap) is not implemented yet.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DENIED_ROOTS = ("/Users", "/Volumes", "/private/var/folders")


class SandboxUnavailable(RuntimeError):
    """This platform has no supported confinement."""


@dataclass(frozen=True, slots=True)
class Runtime:
    """Interpreters the agent and its programs may run."""

    python: Path  # the agent runtime's python (a venv with the scientific stack)
    python_roots: tuple[Path, ...]  # venv prefix and base interpreter prefix
    node_root: Path | None  # the Node installation holding dsh
    dsh: Path | None
    links: tuple[Path, ...] = ()  # symlinks met while resolving the interpreters (readable as literals)

    @property
    def read_roots(self) -> tuple[Path, ...]:
        return self.python_roots + ((self.node_root,) if self.node_root else ())


def symlinks(path: Path, limit: int = 40) -> list[Path]:
    """Every symlink met while resolving ``path`` component by component."""

    found: list[Path] = []
    pending = list(Path(os.path.abspath(path)).parts[1:])
    current = Path("/")
    while pending and len(found) < limit:
        candidate = current / pending.pop(0)
        if candidate.is_symlink():
            found.append(candidate)
            target = Path(os.readlink(candidate))
            if target.is_absolute():
                current, pending = Path("/"), list(target.parts[1:]) + pending
            else:
                pending = list(target.parts) + pending
        elif candidate.name == "..":
            current = current.parent
        else:
            current = candidate
    return found


def ecogym_home() -> Path:
    return Path(os.environ.get("ECOGYM_HOME") or Path.home() / ".ecogym").expanduser()


THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT")


def _count(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value.isascii() and value.isdigit() and int(value) > 0 else default


def thread_limits(count: int | None = None) -> dict[str, str]:
    """Numeric-library threads of a sandboxed process: ``count``, else ``ECOGYM_THREADS`` (1 by default) for the
    agents' own experiments. Unlimited, each of several sessions at once used every core (2026-11-04: six episodes
    at a time kept a 12-core laptop hot); with two, the small models fitted here ran slower than with one and used
    3.6–4.6 times the CPU (2026-11-05). ``LOKY_MAX_CPU_COUNT`` caps joblib's workers (scikit-learn's ``n_jobs=-1``)
    the same way. The judge passes a fixed count, so a score never depends on the setting."""

    value = str(count or _count("ECOGYM_THREADS", 1))
    return {name: value for name in THREAD_VARIABLES}


def check_workers(default: int = 4) -> int:
    """How many training tasks a template check judges at the same time: ``ECOGYM_CHECK_WORKERS`` when set, else
    ``default``. An evolution run passes its concurrency, the load chosen for its episodes: at four at a time, a
    low-load run's template checks still kept eight cores busy (2026-11-05)."""

    return _count("ECOGYM_CHECK_WORKERS", max(1, default))


def runtime_python() -> Path:
    configured = os.environ.get("ECOGYM_RUNTIME_PYTHON")
    return Path(configured).expanduser() if configured else ecogym_home() / "runtime" / "python" / "bin" / "python"


@lru_cache(maxsize=1)
def discover_runtime() -> Runtime:
    python = runtime_python()
    if not python.exists():
        raise SandboxUnavailable(f"agent runtime not found at {python}; run `ecogym setup`")
    probe = subprocess.run(
        [str(python), "-I", "-c", "import json, sys; print(json.dumps([sys.prefix, sys.base_prefix]))"],
        capture_output=True, text=True, check=True)
    roots = tuple(dict.fromkeys(Path(item).resolve() for item in json.loads(probe.stdout)))
    dsh = os.environ.get("ECOGYM_DSH_BIN") or shutil.which("dsh")
    node_root = None
    dsh_path = None
    links = symlinks(python)
    if dsh:
        dsh_path = Path(dsh).expanduser()
        node = shutil.which("node", path=str(dsh_path.parent)) or shutil.which("node")
        if node:
            node_root = Path(node).resolve().parent.parent
            links += symlinks(Path(node)) + symlinks(dsh_path)
    return Runtime(python=python, python_roots=roots, node_root=node_root, dsh=dsh_path,
                   links=tuple(dict.fromkeys(links)))


def available() -> bool:
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def _quote(path: Path | str) -> str:
    return json.dumps(str(path), ensure_ascii=False)


def _ancestors(paths: list[Path]) -> list[Path]:
    seen: dict[Path, None] = {}
    for path in paths:
        for parent in path.parents:
            seen.setdefault(parent, None)
    return list(seen)


def _profile(reads: list[Path], writes: list[Path], links: tuple[Path, ...], network: str) -> str:
    reads = [path.resolve() for path in reads]
    writes = [path.resolve() for path in writes]
    denied = " ".join(f"(subpath {_quote(root)})" for root in DENIED_ROOTS)
    literals = list(links) + _ancestors(reads + writes + list(links))
    lines = [
        "(version 1)",
        "(allow default)",
        f"(deny file-read* {denied} (subpath {_quote(scratch_root())}))",
        f"(deny file-write* {denied} (subpath \"/private/tmp\"))",
        "(allow file-read-metadata " + " ".join(f"(literal {_quote(path)})" for path in literals) + ")",
        "(allow file-read* " + " ".join(f"(literal {_quote(path)})" for path in links)
        + " " + " ".join(f"(subpath {_quote(path)})" for path in reads + writes) + ")",
        "(allow file-write* " + " ".join(f"(subpath {_quote(path)})" for path in writes) + ")",
        network,
    ]
    return "\n".join(lines) + "\n"


def scratch_root() -> Path:
    root = Path(os.environ.get("ECOGYM_SCRATCH") or "/private/tmp/ecogym")
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def agent_profile(episode_dir: Path, ports: list[int], runtime: Runtime) -> str:
    allowed = " ".join(f'(remote ip "localhost:{int(port)}")' for port in ports)
    network = f"(deny network-outbound)\n(allow network-outbound {allowed})"
    return _profile(list(runtime.read_roots), [episode_dir], runtime.links, network)


def program_profile(program_dir: Path, runtime: Runtime) -> str:
    return _profile(list(runtime.python_roots), [program_dir], runtime.links, "(deny network*)")


def confine(profile: str, profile_path: Path, argv: list[str]) -> list[str]:
    """Write the profile and return the confined command line."""

    if not available():
        raise SandboxUnavailable("process confinement needs macOS sandbox-exec; Linux support is not implemented yet")
    profile_path.write_text(profile, encoding="utf-8")
    return ["sandbox-exec", "-f", str(profile_path), *argv]


__all__ = [
    "Runtime",
    "SandboxUnavailable",
    "agent_profile",
    "available",
    "check_workers",
    "confine",
    "discover_runtime",
    "ecogym_home",
    "program_profile",
    "runtime_python",
    "scratch_root",
    "thread_limits",
]
