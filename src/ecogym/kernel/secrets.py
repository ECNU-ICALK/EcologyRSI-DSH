"""API keys, resolved in one place and never logged.

Order: the environment variable itself, then a ``NAME: value`` line in
``$ECOGYM_CREDENTIALS_FILE``, then DSH's local credential store
(``$DSH_HOME/.credentials.yaml``, default ``~/.dsh``), whose ``refs`` section
uses the same line shape.
"""

from __future__ import annotations

import os
import re
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


def resolve_secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*:\s*(.+?)\s*$")
    for path in _files():
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            match = pattern.match(line)
            if match:
                return match.group(1).strip("'\"")
    raise SecretMissing(
        f"no {name}: set the environment variable, point ECOGYM_CREDENTIALS_FILE at a file with a "
        f"'{name}: ...' line, or store it in DSH (~/.dsh/.credentials.yaml)"
    )


__all__ = ["SecretMissing", "resolve_secret"]
