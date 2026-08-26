#!/usr/bin/env python3
"""Fail-closed filesystem checks shared by release builders and verifiers."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def checked_lstat(
    root: Path,
    path: Path,
    label: str,
    *,
    allow_missing: bool = False,
) -> os.stat_result | None:
    """Return the final lstat without following any component symlink."""

    lexical_root = Path(os.path.abspath(root))
    lexical_path = Path(os.path.abspath(path))
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise RuntimeError(f"{label} is outside its allowed root: {path}") from exc

    current = lexical_root
    paths = [current]
    for part in relative.parts:
        current /= part
        paths.append(current)
    for current in paths:
        try:
            current_stat = current.lstat()
        except FileNotFoundError as exc:
            if allow_missing:
                return None
            raise RuntimeError(f"{label} is missing: {current}") from exc
        if stat.S_ISLNK(current_stat.st_mode):
            raise RuntimeError(f"{label} is a symlink: {current}")
    return current_stat


def require_regular_file(root: Path, path: Path, label: str) -> Path:
    """Require a regular file after rejecting symlinks in every component."""

    path_stat = checked_lstat(root, path, label)
    if path_stat is None or not stat.S_ISREG(path_stat.st_mode):
        raise RuntimeError(f"{label} is not a regular file: {path}")
    return path
