#!/usr/bin/env python3
"""Create a deterministic complete-delivery archive and checksums."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import tarfile


ROOT_FILES = (
    ".gitignore",
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "NOTICE",
    "RELEASE-CHECKLIST.md",
    "MANIFEST.in",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
)
SOURCE_DIRS = (
    "src",
    "tests",
    "examples",
    "datasets",
    "docs",
    "plugins",
    "integrations",
    "scripts",
)
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".DS_Store", "build", "dist"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".sqlite3"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_version(root: Path) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    if match is None:
        raise RuntimeError("project version is missing")
    return match.group(1)


def included_source_files(root: Path) -> list[Path]:
    files = [root / name for name in ROOT_FILES]
    for directory in SOURCE_DIRS:
        for path in (root / directory).rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if any(
                part in IGNORED_PARTS or part.endswith(".egg-info")
                for part in relative.parts
            ):
                continue
            if path.suffix in IGNORED_SUFFIXES:
                continue
            files.append(path)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError("missing delivery inputs: " + ", ".join(missing))
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes, *, mtime: int, executable: bool = False) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = mtime
    info.mode = 0o755 if executable else 0o644
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    archive.addfile(info, io.BytesIO(data))


def create_archive(root: Path, dist: Path) -> Path:
    version = project_version(root)
    wheel = next(iter(sorted(dist.glob(f"ecologyrsi_dsh-{version}-*.whl"))), None)
    sdist = next(iter(sorted(dist.glob(f"ecologyrsi_dsh-{version}.tar.gz"))), None)
    if wheel is None or sdist is None:
        raise RuntimeError("wheel and sdist must be built before the delivery archive")
    plugins = sorted(
        (root / "integrations/dsh_ecology_plugin/dist").glob(
            f"ecologyrsi-dsh-evolution-plugin-{version}.tgz"
        )
    )
    if len(plugins) != 1:
        raise RuntimeError(
            f"exactly one packed DSH plugin is required for version {version}"
        )
    plugin = plugins[0]
    distributed_plugin = dist / plugin.name
    shutil.copyfile(plugin, distributed_plugin)

    output = dist / f"ecologyrsi-dsh-{version}-delivery.tar.gz"
    prefix = f"ecologyrsi-dsh-{version}"
    timestamp = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    internal_checksums = (
        f"{sha256(wheel)}  artifacts/{wheel.name}\n"
        f"{sha256(sdist)}  artifacts/{sdist.name}\n"
        f"{sha256(distributed_plugin)}  artifacts/{distributed_plugin.name}\n"
    ).encode("ascii")

    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=timestamp) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in included_source_files(root):
                    relative = path.relative_to(root).as_posix()
                    executable = relative.startswith("scripts/") and path.suffix in {".sh", ".py"}
                    add_bytes(
                        archive,
                        f"{prefix}/{relative}",
                        path.read_bytes(),
                        mtime=timestamp,
                        executable=executable,
                    )
                for artifact in (wheel, sdist, distributed_plugin):
                    add_bytes(
                        archive,
                        f"{prefix}/artifacts/{artifact.name}",
                        artifact.read_bytes(),
                        mtime=timestamp,
                    )
                add_bytes(
                    archive,
                    f"{prefix}/SHA256SUMS",
                    internal_checksums,
                    mtime=timestamp,
                )

    checksums = dist / "SHA256SUMS"
    artifacts = (wheel, sdist, distributed_plugin, output)
    checksums.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="ascii",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dist", type=Path, required=True)
    args = parser.parse_args()
    output = create_archive(args.root.resolve(), args.dist.resolve())
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
