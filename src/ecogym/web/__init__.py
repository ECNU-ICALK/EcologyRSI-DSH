"""The DSH plugin (``plugin/``) and its installer.

``ecogym dsh install`` makes ecogym part of a DSH profile (``web`` by default):

* ``dsh plugin --profile web add link:<plugin>`` puts ``@ecogym/dsh-plugin`` in the profile. The package
  declares a profile layer (``plugin/cordis.patch.yml``), so DSH loads it from then on; the link points at
  this directory, so updating ecogym needs no reinstall (restart DSH to load a new version).
* ``~/.ecogym/dsh-plugin.json`` (``$ECOGYM_DSH_SETTINGS``) records how the plugin starts the service: this
  Python, ECOGYM_HOME and the ecogym variables of the installing shell (data root, agent runtime, dsh).

DSH then starts ``python -m ecogym serve`` with the web app and stops it on exit (running evolutions end
as stopped and can be resumed), and its sidebar gets a 生态进化 panel. The earlier EcologyRSI plugin's
managed block in the profile's ``cordis.patch.yml`` is removed (after a backup), because its workbench
needs a service that no longer exists; a block whose end marker is missing is left for the user.
``ecogym dsh uninstall`` takes the plugin out again, and the settings file unless another profile's
install wrote it.

``ecogym dsh autostart on`` (macOS) makes launchd start ``dsh --profile web --no-open`` at login and again
after a crash or a kill, independent of any terminal or editor: a DSH started from an agent's shell ended
with the agent's turn (2026-11-04), and a restart of the machine ended everything. Its output, with the
web address and token, goes to ``~/.ecogym/dsh-web.log``; ``status`` prints the latest address and ``off``
stops it and removes the job. The job file and the log are private (0600): the job may carry proxy settings,
the log holds each start's token. Model keys are not copied into the job; a key that only this shell's
environment holds is reported, since the DSH launchd starts cannot see it.
"""

from __future__ import annotations

import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PACKAGE = "@ecogym/dsh-plugin"
LEGACY_BLOCK = ("# BEGIN ECOLOGYRSI DSH RUNTIME (managed)", "# END ECOLOGYRSI DSH RUNTIME (managed)")
CARRIED = ("ECOGYM_DATA_ROOT", "ECOGYM_RUNTIME_PYTHON", "ECOGYM_DSH_BIN", "ECOGYM_CACHE_DIR")
LAUNCH_PATH = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin")
LAUNCH_CARRIED = ("LANG", "DSH_HOME", "ECOGYM_DSH_SETTINGS", "ECOGYM_CREDENTIALS_FILE", "ECOGYM_UPSTREAM_PROXY",
                  "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "all_proxy",
                  "no_proxy")
LOG_TAIL = 65_536  # bytes of the log read for the latest address


def plugin_dir() -> Path:
    return Path(__file__).resolve().parent / "plugin"


def settings_path() -> Path:
    configured = os.environ.get("ECOGYM_DSH_SETTINGS")
    return Path(configured).expanduser() if configured else Path.home() / ".ecogym" / "dsh-plugin.json"


def profile_dir(profile: str) -> Path:
    home = Path(os.environ.get("DSH_HOME") or Path.home() / ".dsh").expanduser()
    return home / "profiles" / profile


def find_dsh() -> str:
    found = os.environ.get("ECOGYM_DSH_BIN") or shutil.which("dsh")
    if not found:
        raise FileNotFoundError("dsh not found: install DeepSeek Harness (npm install --global @deepseek-ai/dsh)")
    return found


def _dsh(dsh: str, *args: str) -> str:
    try:
        done = subprocess.run([dsh, *args], capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"dsh {' '.join(args)} did not finish in 15 minutes") from None
    if done.returncode != 0:
        raise RuntimeError(f"dsh {' '.join(args)} failed:\n{(done.stdout + done.stderr)[-3000:]}")
    return done.stdout + done.stderr


def _manifest(profile: Path) -> dict[str, Any]:
    try:
        return json.loads((profile / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _strip_block(text: str, begin: str, end: str) -> tuple[str, bool]:
    """Drop every block from a ``begin`` line to the next ``end`` line; a begin without an end raises."""

    lines, found = text.splitlines(keepends=True), False
    while True:
        marks = [line.strip() for line in lines]
        if begin not in marks:
            return "".join(lines), found
        start = marks.index(begin)
        if end not in marks[start + 1:]:
            raise ValueError(f"'{begin}' has no matching '{end}'")
        lines = lines[:start] + lines[marks.index(end, start + 1) + 1:]
        found = True


def remove_legacy(profile: Path) -> str | None:
    """Take the EcologyRSI block out of the profile's patch layer; returns the backup's name.

    A block whose end marker is missing or edited is left alone (``ValueError``): guessing where it ends
    could cut the rest of the layer, which a running DSH applies at once.
    """

    patch = profile / "cordis.patch.yml"
    if not patch.is_file():
        return None
    text = patch.read_text(encoding="utf-8")
    try:
        stripped, found = _strip_block(text, *LEGACY_BLOCK)
    except ValueError as error:
        raise ValueError(f"{patch}: {error}; remove the EcologyRSI block by hand") from None
    if not found:
        return None
    backup = patch.with_name(f"{patch.name}.ecogym-backup-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(patch, backup)
    if all(not line.strip() or line.lstrip().startswith("#") for line in stripped.splitlines()):
        stripped = stripped.rstrip("\n") + "\n[]\n"  # the layer stays a (now empty) patch list
    patch.write_text(stripped, encoding="utf-8")
    return backup.name


def install(*, profile: str = "web", home: Path | None = None, ledger: Path | None = None,
            env: dict[str, str] | None = None, keep_legacy: bool = False) -> dict[str, Any]:
    dsh = find_dsh()
    plugin = plugin_dir()
    if not (plugin / "package.json").is_file():
        raise FileNotFoundError(f"the plugin files are missing from {plugin}")
    _dsh(dsh, "plugin", "--profile", profile, "add", f"link:{plugin}")
    directory = profile_dir(profile)
    bundles = ((_manifest(directory).get("dsh") or {}).get("profile") or {}).get("bundles") or []
    if PACKAGE not in bundles:
        raise RuntimeError(f"dsh added {PACKAGE} to {directory} but not as a profile layer; this DSH may be "
                           "too old for plugin layers (dsh.bundle)")
    notes = []
    try:
        legacy = None if keep_legacy else remove_legacy(directory)
    except ValueError as error:
        legacy = None
        notes.append(str(error))
    path = settings_path()
    try:
        previous = json.loads(path.read_text(encoding="utf-8")).get("profile")
    except (OSError, ValueError, AttributeError):
        previous = None
    if previous and previous != profile:  # one settings file serves every profile with the plugin
        notes.append(f"{path} was written for profile {previous}; every profile with the plugin "
                     f"({previous} included) now uses these settings")
    carried = {key: os.environ[key] for key in CARRIED if os.environ.get(key)}
    carried.setdefault("ECOGYM_DSH_BIN", str(Path(dsh).absolute()))
    home = Path(home or os.environ.get("ECOGYM_HOME") or Path.home() / ".ecogym").expanduser().absolute()
    settings = {"python": sys.executable, "home": str(home), "env": {**carried, **(env or {})},
                "profile": profile, "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if ledger is not None:
        settings["ledger"] = str(Path(ledger).expanduser().absolute())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {**settings, "profile_dir": str(directory), "plugin": str(plugin), "settings": str(path),
            "legacy_backup": legacy, "notes": notes}


def uninstall(*, profile: str = "web") -> dict[str, Any]:
    """Take the plugin out of ``profile``; the settings file goes too unless it was written for another one."""

    dsh = find_dsh()
    directory = profile_dir(profile)
    removed = PACKAGE in (_manifest(directory).get("dependencies") or {})
    if removed:
        _dsh(dsh, "plugin", "--profile", profile, "remove", PACKAGE)
    path = settings_path()
    try:
        owner = json.loads(path.read_text(encoding="utf-8")).get("profile")
    except (OSError, ValueError, AttributeError):
        owner = None
    deleted = path.is_file() and owner in (None, profile)
    if deleted:
        path.unlink()
    return {"profile": str(directory), "removed": removed, "settings_removed": str(path) if deleted else None,
            "settings_kept_for": owner if path.is_file() else None}


def status(*, profile: str = "web") -> dict[str, Any]:
    directory = profile_dir(profile)
    manifest = _manifest(directory)
    patch = directory / "cordis.patch.yml"
    path = settings_path()
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        settings = None
    env = (settings or {}).get("env") or {}
    home = Path((settings or {}).get("home") or Path.home() / ".ecogym")
    runtime = Path(env.get("ECOGYM_RUNTIME_PYTHON") or home / "runtime" / "python" / "bin" / "python")
    return {
        "dsh": os.environ.get("ECOGYM_DSH_BIN") or shutil.which("dsh"),
        "profile": str(directory),
        "installed": PACKAGE in (manifest.get("dependencies") or {}),
        "layer": PACKAGE in (((manifest.get("dsh") or {}).get("profile") or {}).get("bundles") or []),
        "legacy_block": patch.is_file() and LEGACY_BLOCK[0] in patch.read_text(encoding="utf-8"),
        "settings": str(path) if settings is not None else None,
        "python": (settings or {}).get("python"),
        "python_found": bool(settings) and Path(settings.get("python", "")).exists(),
        "home": str(home),
        "agent_runtime": str(runtime),
        "agent_runtime_found": runtime.exists(),
        "data_root": env.get("ECOGYM_DATA_ROOT"),
    }


def launch_agent(profile: str = "web") -> tuple[str, Path]:
    """The launchd label and job file that start ``profile``."""

    label = f"local.ecogym.dsh.{profile}"
    return label, Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def autostart_log(profile: str = "web") -> Path:
    return Path.home() / ".ecogym" / f"dsh-{profile}.log"


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)


def _others(profile: str) -> list[int]:
    """Processes serving ``profile`` that this job did not start (they hold its port)."""

    names = f"(--profile {re.escape(profile)}|{re.escape(profile)})" if profile == "web" else \
        f"--profile {re.escape(profile)}"
    found = subprocess.run(["pgrep", "-f", f"/dsh {names}( |$)"], capture_output=True, text=True).stdout.split()
    return [int(pid) for pid in found if pid.isdigit()]


def _shell_only_keys() -> list[str]:
    """API keys of the model routes that only this shell's environment holds (no credential file has them)."""

    from ..episodes import providers
    from ..kernel.secrets import in_files

    try:
        names = {route.key_env for route in providers.routes().values() if route.key_env}
    except (providers.ModelError, OSError):
        return []
    return sorted(name for name in names if os.environ.get(name, "").strip() and not in_files(name))


def autostart(*, profile: str = "web", enable: bool) -> dict[str, Any]:
    """Start ``dsh --profile <profile> --no-open`` at login and after a crash (``enable``), or stop it and remove the
    job. Enabling again rewrites the job and restarts DSH with it."""

    if sys.platform != "darwin":
        raise RuntimeError("autostart uses launchd, so it works on macOS only")
    label, plist = launch_agent(profile)
    target = f"gui/{os.getuid()}/{label}"
    loaded = _launchctl("print", target).returncode == 0
    if not enable:
        if loaded:
            _launchctl("bootout", target)  # stops the DSH it started
        removed = plist.is_file()
        if removed:
            plist.unlink()
        return {"label": label, "plist": str(plist), "stopped": loaded, "removed": removed}
    dsh = str(Path(find_dsh()).absolute())
    if "--no-open" not in _dsh(dsh, "--profile", profile, "--help"):
        raise RuntimeError(f"profile {profile} does not serve the web UI (dsh --profile {profile} --help lists no "
                           "--no-open)")
    node = shutil.which("node")
    path = ":".join(dict.fromkeys([str(Path(dsh).parent), *([str(Path(node).parent)] if node else []),
                                   str(Path.home() / ".local" / "bin"), *LAUNCH_PATH]))
    env = {"PATH": path, "LANG": os.environ.get("LANG") or "en_US.UTF-8"}
    env.update({key: os.environ[key] for key in LAUNCH_CARRIED if os.environ.get(key) and key not in env})
    log = autostart_log(profile)
    log.parent.mkdir(parents=True, exist_ok=True)
    job = {
        "Label": label,
        "ProgramArguments": [dsh, "--profile", profile, "--no-open"],
        "EnvironmentVariables": env,
        "WorkingDirectory": str(Path.home()),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},  # again after a crash or a kill; not after DSH quits by itself
        "ThrottleInterval": 30,
        "AbandonProcessGroup": True,  # the ecogym service finishes its own shutdown: runs end as stopped
        "ExitTimeOut": 45,
        "StandardOutPath": str(log),
        "StandardErrorPath": str(log),
    }
    if loaded:
        _launchctl("bootout", target)
        for _ in range(30):  # bootout returns before the job is gone
            if _launchctl("print", target).returncode != 0:
                break
            time.sleep(1)
    notes = []
    others = _others(profile)
    if others:
        notes.append(f"another DSH serves profile {profile} (pid {', '.join(map(str, others))}); the started one "
                     "cannot take its port until it stops, and launchd tries again every 30 seconds")
    shell_only = _shell_only_keys()
    if shell_only:
        notes.append(f"{', '.join(shell_only)} is set only in this shell's environment, which the DSH launchd starts "
                     "does not see; store it in DSH's credential store (~/.dsh/.credentials.yaml) or in "
                     "$ECOGYM_CREDENTIALS_FILE, or runs that need it will fail")
    plist.parent.mkdir(parents=True, exist_ok=True)
    # private: the job may carry proxy settings, and the log gets each start's web address with its token
    with os.fdopen(os.open(plist, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as handle:
        plistlib.dump(job, handle)
    os.chmod(plist, 0o600)
    with os.fdopen(os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a", encoding="utf-8") as handle:
        handle.write(f"---- {time.strftime('%Y-%m-%d %H:%M:%S')} launchd job {label} (re)loaded\n")
    os.chmod(log, 0o600)
    done = _launchctl("bootstrap", f"gui/{os.getuid()}", str(plist))
    if done.returncode != 0:
        raise RuntimeError(f"launchctl bootstrap {plist} failed: {(done.stdout + done.stderr).strip()[:500]}")
    return {"label": label, "plist": str(plist), "log": str(log), "program": job["ProgramArguments"], "path": path,
            "notes": notes}


def autostart_status(*, profile: str = "web", since: int = 0) -> dict[str, Any]:
    """The launchd job of ``profile`` and the latest web address in its log (written after byte ``since``)."""

    label, plist = launch_agent(profile)
    log = autostart_log(profile)
    found: dict[str, Any] = {"label": label, "plist": str(plist) if plist.is_file() else None, "loaded": False,
                             "state": None, "pid": None, "last_exit": None, "log": str(log), "url": None,
                             "service": None}
    if sys.platform == "darwin":
        done = _launchctl("print", f"gui/{os.getuid()}/{label}")
        if done.returncode == 0:
            found["loaded"] = True
            for key, pattern in (("state", r"^\s*state = (.+)$"), ("pid", r"^\s*pid = (\d+)$"),
                                 ("last_exit", r"^\s*last exit code = (.+)$")):
                match = re.search(pattern, done.stdout, re.MULTILINE)
                if match:
                    found[key] = int(match.group(1)) if key == "pid" else match.group(1).strip()
    if log.is_file():
        size = log.stat().st_size
        with log.open("rb") as handle:
            handle.seek(max(since, size - LOG_TAIL, 0))
            lines = handle.read().decode("utf-8", "replace").splitlines()
        urls = [line.split("dsh web: ", 1)[1].strip() for line in lines if line.startswith("dsh web: http")]
        ready = [line.strip() for line in lines if "service ready" in line]
        found["url"] = urls[-1] if urls else None
        found["service"] = ready[-1] if ready else None
        found["log_bytes"] = size
    return found


__all__ = ["PACKAGE", "autostart", "autostart_log", "autostart_status", "install", "launch_agent", "plugin_dir",
           "profile_dir", "remove_legacy", "settings_path", "status", "uninstall"]
