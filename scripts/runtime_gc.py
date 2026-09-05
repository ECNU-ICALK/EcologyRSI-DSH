#!/usr/bin/env python3
"""Preview-first cleanup for disposable files below the project's ``.runtime``.

The tool deliberately does not remove databases, lock/pid markers, or symlinks.
Use ``--apply`` only after reviewing the generated manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
from pathlib import Path
from typing import Any

PROTECTED_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
PROTECTED_NAMES = {".active", ".lock", ".pid"}


def _runtime_root(root: Path) -> Path:
    root = Path(root)
    if root.name != ".runtime":
        raise ValueError("runtime cleanup is restricted to a directory named .runtime")
    if root.exists() and root.is_symlink():
        raise RuntimeError(f"runtime root is a symlink: {root}")
    return root


def _files(root: Path) -> list[tuple[Path, os.stat_result]]:
    if not root.exists():
        return []
    if not root.is_dir():
        raise RuntimeError(f"runtime root is not a directory: {root}")
    result = []
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"symlinks are not allowed under runtime root: {path}")
        if stat.S_ISREG(info.st_mode):
            result.append((path, info))
    return result


def _protected(path: Path) -> bool:
    return path.name in PROTECTED_NAMES or path.suffix.lower() in PROTECTED_SUFFIXES


def plan_cleanup(root: Path, *, older_than_days: float = 7, keep: int = 10) -> list[dict[str, Any]]:
    """Return deletions without changing the filesystem."""
    root = _runtime_root(root)
    if older_than_days < 0 or keep < 0:
        raise ValueError("older_than_days and keep must be non-negative")
    now = time.time()
    cutoff = now - older_than_days * 86400
    candidates = [(path, info) for path, info in _files(root) if not _protected(path)]
    candidates.sort(key=lambda pair: (pair[1].st_mtime, str(pair[0])), reverse=True)
    retained = {path for path, _ in candidates[:keep]}
    planned = []
    for path, info in candidates[keep:]:
        if info.st_mtime < cutoff and path not in retained:
            planned.append(
                {"path": str(path.relative_to(root)), "size": info.st_size, "mtime": info.st_mtime}
            )
    return planned


def run_cleanup(
    root: Path,
    *,
    older_than_days: float = 7,
    keep: int = 10,
    apply: bool = False,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    root = _runtime_root(root)
    planned = plan_cleanup(root, older_than_days=older_than_days, keep=keep)
    manifest = {
        "root": str(root),
        "generated_at": time.time(),
        "older_than_days": older_than_days,
        "keep": keep,
        "dry_run": not apply,
        "planned": planned,
        "deleted": [],
    }
    if apply:
        for item in planned:
            target = root / item["path"]
            target.lstat()
            if target.is_symlink() or not target.is_file():
                raise RuntimeError(f"refusing to remove changed runtime entry: {target}")
            target.unlink()
            manifest["deleted"].append(item["path"])
    if manifest_path is None:
        manifest_path = root / "runtime-gc-manifest.json"
    manifest_path = Path(manifest_path)
    if manifest_path.is_symlink():
        raise RuntimeError(f"manifest path is a symlink: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".runtime"))
    parser.add_argument("--older-than-days", type=float, default=7)
    parser.add_argument("--keep", type=int, default=10)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--apply", action="store_true", help="delete the reviewed plan")
    args = parser.parse_args()
    result = run_cleanup(
        args.root,
        older_than_days=args.older_than_days,
        keep=args.keep,
        apply=args.apply,
        manifest_path=args.manifest,
    )
    mode = "applied" if args.apply else "preview"
    print(f"runtime gc {mode}: {len(result['planned'])} planned, {len(result['deleted'])} deleted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
