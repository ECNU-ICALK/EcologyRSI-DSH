"""API keys, resolved in one place and never logged.

Order: the environment variable itself, then a ``NAME: value`` line in
``$ECOGYM_CREDENTIALS_FILE``, then DSH's local credential store
(``$DSH_HOME/.credentials.yaml``, default ``~/.dsh``), whose ``refs`` section
uses the same line shape.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path


class SecretMissing(LookupError):
    """No source holds the named key."""


def _files() -> list[Path]:
    files = []
    configured = os.environ.get("ECOGYM_CREDENTIALS_FILE", "").strip()
    if configured:
        files.append(Path(configured).expanduser())
    dsh_home = Path(os.environ.get("DSH_HOME") or Path.home() / ".dsh").expanduser()
    files.append(dsh_home / ".credentials.yaml")
    return files


def _lookup(name: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*:\s*(.+?)\s*$")
    for path in _files():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            match = pattern.match(line)
            if match:
                return match.group(1).strip("'\"")
    return None


def has_secret(name: str) -> bool:
    """Whether a source holds ``name`` (one look, no waiting; for listings)."""

    return bool(os.environ.get(name, "").strip()) or bool(_lookup(name))


def in_files(name: str) -> bool:
    """Whether a credential file holds ``name``: what a process started without this shell's environment sees."""

    return bool(_lookup(name))


def resolve_secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    for attempt in range(3):  # a file another program is rewriting can read empty for a moment

        found = _lookup(name)
        if found:
            return found
        time.sleep(0.3 * (attempt + 1))
    raise SecretMissing(
        f"no {name}: set the environment variable, point ECOGYM_CREDENTIALS_FILE at a file with a "
        f"'{name}: ...' line, or store it in DSH (~/.dsh/.credentials.yaml)"
    )


__all__ = ["SecretMissing", "has_secret", "in_files", "resolve_secret"]
