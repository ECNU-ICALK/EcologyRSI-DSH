#!/usr/bin/env python3
"""Fail-closed filesystem checks shared by release builders and verifiers."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def _lexical_absolute(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    if len(absolute.parts) < 2:
        return absolute
    anchor = Path(absolute.anchor)
    trusted_top_level = anchor / absolute.parts[1]
    canonical_top_level = Path(os.path.realpath(trusted_top_level))
    return canonical_top_level.joinpath(*absolute.parts[2:])


def checked_lstat(
    root: Path,
    path: Path,
    label: str,
    *,
    allow_missing: bool = False,
) -> os.stat_result | None:
    """Return the final lstat without following any component symlink."""

    for candidate in (root, path):
        if os.pardir in candidate.parts:
            raise RuntimeError(f"{label} contains parent traversal: {candidate}")
    lexical_root = _lexical_absolute(root)
    lexical_path = _lexical_absolute(path)
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


def require_directory(
    root: Path,
    path: Path,
    label: str,
    *,
    allow_missing: bool = False,
) -> Path:
    """Require a directory without resolving caller-controlled symlinks."""

    path_stat = checked_lstat(root, path, label, allow_missing=allow_missing)
    if path_stat is None and allow_missing:
        return path
    if path_stat is None or not stat.S_ISDIR(path_stat.st_mode):
        raise RuntimeError(f"{label} is not a directory: {path}")
    return path
