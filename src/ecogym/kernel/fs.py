"""Safe directory copying without symlinks, hardlinks, or unbounded file trees."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Collection
from pathlib import Path


class CopyError(RuntimeError):
    """A directory tree violates regular-file, link, or size limits."""


def copy_regular_tree(
    source: Path,
    target: Path,
    *,
    max_files: int,
    max_bytes: int,
    skip_dirs: Collection[str] = (),
    skip_file: Callable[[str], bool] | None = None,
    hash_header: bytes | None = None,
) -> str:
    """Copy regular files from ``source`` to ``target`` without following links; returns SHA-256 when ``hash_header``
    is provided, else an empty string."""

    if source.is_symlink() or not source.is_dir():
        raise CopyError(f"{source.name} must be a directory")
    sha = hashlib.sha256(hash_header) if hash_header is not None else None
    skip_set = set(skip_dirs)
    count = total = 0
    for directory, names, filenames in os.walk(source, followlinks=False):
        for name in list(names):
            if os.path.islink(os.path.join(directory, name)):
                raise CopyError(f"{name}: symlinks are not allowed")
        names[:] = sorted(name for name in names if name not in skip_set and not name.startswith("."))
        for name in sorted(filenames):
            if skip_file is not None and skip_file(name):
                continue
            path = Path(directory) / name
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise CopyError(f"{path.relative_to(source)}: only regular files are allowed")
            if info.st_nlink > 1:
                raise CopyError(f"{path.relative_to(source)}: hard links are not allowed")
            count += 1
            total += info.st_size
            if count > max_files or total > max_bytes:
                raise CopyError(f"more than {max_files} files or {max_bytes} bytes")
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                data = stream.read(max_bytes + 1)
            relative = path.relative_to(source)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            if sha is not None:
                sha.update(f"{relative.as_posix()}\n{len(data)}\n".encode())
                sha.update(data)
    return sha.hexdigest() if sha is not None else ""


__all__ = ["CopyError", "copy_regular_tree"]
