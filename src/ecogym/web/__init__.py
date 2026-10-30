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
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PACKAGE = "@ecogym/dsh-plugin"
LEGACY_BLOCK = ("# BEGIN ECOLOGYRSI DSH RUNTIME (managed)", "# END ECOLOGYRSI DSH RUNTIME (managed)")
CARRIED = ("ECOGYM_DATA_ROOT", "ECOGYM_RUNTIME_PYTHON", "ECOGYM_DSH_BIN", "ECOGYM_CACHE_DIR")


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


__all__ = ["PACKAGE", "install", "plugin_dir", "profile_dir", "remove_legacy", "settings_path", "status",
           "uninstall"]
