"""Process confinement for agents and submitted programs (macOS Seatbelt and Linux bubblewrap).

Two confinement modes, both preserving ordinary runtimes while isolating secrets and hidden data:

* ``agent_profile`` / ``confine_agent``: the DSH agent and everything it spawns. Reads only its
  episode directory, the runtimes (Node, Python) and system locations; writes
  only its episode directory; connects only to the given loopback ports on macOS (the
  episode's LLM gateway and tool server). Everything else under /Users and
  /Volumes (or /home, /root, /mnt, /media on Linux) is unreadable.
* ``program_profile`` / ``confine_program``: a submitted predictor. Reads its own directory and the
  Python runtime, writes its own directory, no network at all (``(deny network*)`` on macOS,
  ``--unshare-net`` on Linux). Hidden data reaches it only through the judge's pipe, one time step at a time.

Reads are denied under user/removable roots, the scratch root, and wherever ecogym keeps what no
session may see, configured anywhere: ``$ECOGYM_HOME`` (ledger, trajectories, sealed predictions),
the data root and the frame cache (``protected_roots``). A data root moved to /opt or /private/tmp
stays unreadable. On macOS, ancestors of allowed paths get metadata access only, so path resolution
works without listing directories. On Linux, ``BubblewrapBackend`` provides mount/PID/IPC/network
namespace isolation via ``bwrap``.
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
from typing import Protocol

from ..data.catalog import cache_dir, data_root

DENIED_ROOTS = ("/Users", "/Volumes", "/private/var/folders")
LINUX_DENIED_ROOTS = ("/home", "/root", "/mnt", "/media")


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


class SandboxBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def confine_program(self, program_dir: Path, runtime: Runtime, profile_path: Path,
                        argv: list[str]) -> list[str]: ...

    def confine_agent(self, episode_dir: Path, ports: list[int], runtime: Runtime, profile_path: Path,
                      argv: list[str]) -> list[str]: ...


class SeatbeltBackend:
    """macOS ``sandbox-exec`` confinement using Seatbelt SBPL profiles."""

    name = "seatbelt"

    def available(self) -> bool:
        return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None

    def confine_program(self, program_dir: Path, runtime: Runtime, profile_path: Path, argv: list[str]) -> list[str]:
        profile = program_profile(program_dir, runtime)
        return confine(profile, profile_path, argv)

    def confine_agent(self, episode_dir: Path, ports: list[int], runtime: Runtime, profile_path: Path,
                      argv: list[str]) -> list[str]:
        profile = agent_profile(episode_dir, ports, runtime)
        return confine(profile, profile_path, argv)


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


class BubblewrapBackend:
    """Linux ``bwrap`` namespace confinement (mount/PID/IPC isolation; unshared network for programs)."""

    name = "bubblewrap"

    def available(self) -> bool:
        return sys.platform.startswith("linux") and shutil.which("bwrap") is not None

    @staticmethod
    def _base_args(reads: list[Path], writes: list[Path], *, unshare_net: bool) -> list[str]:
        args = ["bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc"]
        if unshare_net:
            args.append("--unshare-net")
        args += ["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
        denied = sorted(
            (Path(root) for root in (*LINUX_DENIED_ROOTS, *protected_roots())),
            key=lambda p: len(p.parts),
        )
        masked: list[Path] = []
        for path in denied:
            if any(path == prior or prior in path.parents for prior in masked):
                continue
            if _path_exists(path):
                args += ["--tmpfs", str(path)]
                masked.append(path)
        for path in dict.fromkeys(p.resolve() for p in reads):
            if _path_exists(path):
                args += ["--ro-bind", str(path), str(path)]
        for path in dict.fromkeys(p.resolve() for p in writes):
            if _path_exists(path):
                args += ["--bind", str(path), str(path)]
        return args

    def confine_program(self, program_dir: Path, runtime: Runtime, profile_path: Path, argv: list[str]) -> list[str]:
        if not self.available():
            raise SandboxUnavailable("bubblewrap (bwrap) is not available on this Linux host")
        cmd = [*self._base_args(list(runtime.python_roots), [program_dir], unshare_net=True), "--", *argv]
        profile_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
        return cmd

    def confine_agent(self, episode_dir: Path, ports: list[int], runtime: Runtime, profile_path: Path,
                      argv: list[str]) -> list[str]:
        if not self.available():
            raise SandboxUnavailable("bubblewrap (bwrap) is not available on this Linux host")
        cmd = [*self._base_args(list(runtime.read_roots), [episode_dir], unshare_net=False), "--", *argv]
        profile_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
        return cmd


_SEATBELT = SeatbeltBackend()
_BUBBLEWRAP = BubblewrapBackend()


def active_backend() -> SandboxBackend | None:
    """Return the first available confinement backend for the current platform."""

    forced = os.environ.get("ECOGYM_SANDBOX_BACKEND", "").strip().lower()
    if forced == "seatbelt":
        return _SEATBELT if _SEATBELT.available() else None
    if forced == "bubblewrap":
        return _BUBBLEWRAP if _BUBBLEWRAP.available() else None
    for backend in (_SEATBELT, _BUBBLEWRAP):
        if backend.available():
            return backend
    return None


def backend_name() -> str | None:
    backend = active_backend()
    return backend.name if backend is not None else None


def available() -> bool:
    return active_backend() is not None


def _quote(path: Path | str) -> str:
    return json.dumps(str(path), ensure_ascii=False)


def _ancestors(paths: list[Path]) -> list[Path]:
    seen: dict[Path, None] = {}
    for path in paths:
        for parent in path.parents:
            seen.setdefault(parent, None)
    return list(seen)


def _spellings(path: Path) -> list[Path]:
    """A path as given (absolute) and resolved: a profile names both, whichever the system matches."""

    given = Path(os.path.abspath(path.expanduser()))
    return list(dict.fromkeys([given, given.resolve()]))


def protected_roots() -> list[Path]:
    """What no session may read wherever it is configured: ecogym's home, the data root, the frame cache and
    the scratch root (other sessions' directories). Allowed paths inside them (a session's own directory, a
    runtime installed under ``$ECOGYM_HOME``) are opened again explicitly after these denials."""

    found: list[Path] = []
    for root in (ecogym_home(), data_root(), cache_dir(), scratch_root()):
        found.extend(_spellings(root))
    return list(dict.fromkeys(found))


def _profile(reads: list[Path], writes: list[Path], links: tuple[Path, ...], network: str) -> str:
    reads = [path.resolve() for path in reads]
    writes = [path.resolve() for path in writes]
    denied = " ".join(f"(subpath {_quote(root)})" for root in [*DENIED_ROOTS, *protected_roots()])
    literals = list(links) + _ancestors(reads + writes + list(links))
    lines = [
        "(version 1)",
        "(allow default)",
        f"(deny file-read* {denied})",
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
    """Write a Seatbelt profile and return the confined command line (macOS)."""

    if not _SEATBELT.available():
        raise SandboxUnavailable("Seatbelt confinement needs macOS sandbox-exec")
    profile_path.write_text(profile, encoding="utf-8")
    return ["sandbox-exec", "-f", str(profile_path), *argv]


def confine_program(program_dir: Path, runtime: Runtime, profile_path: Path, argv: list[str]) -> list[str]:
    """Confine a submitted predictor using the active platform backend (Seatbelt or Bubblewrap)."""

    backend = active_backend()
    if backend is None:
        raise SandboxUnavailable("no supported process confinement backend (needs macOS sandbox-exec or Linux bwrap)")
    return backend.confine_program(program_dir, runtime, profile_path, argv)


def confine_agent(episode_dir: Path, ports: list[int], runtime: Runtime, profile_path: Path,
                  argv: list[str]) -> list[str]:
    """Confine an agent session using the active platform backend (Seatbelt or Bubblewrap)."""

    backend = active_backend()
    if backend is None:
        raise SandboxUnavailable("no supported process confinement backend (needs macOS sandbox-exec or Linux bwrap)")
    return backend.confine_agent(episode_dir, ports, runtime, profile_path, argv)


__all__ = [
    "BubblewrapBackend",
    "Runtime",
    "SandboxBackend",
    "SandboxUnavailable",
    "SeatbeltBackend",
    "active_backend",
    "agent_profile",
    "available",
    "backend_name",
    "check_workers",
    "confine",
    "confine_agent",
    "confine_program",
    "discover_runtime",
    "ecogym_home",
    "program_profile",
    "protected_roots",
    "runtime_python",
    "scratch_root",
    "thread_limits",
]
