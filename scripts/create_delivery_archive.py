#!/usr/bin/env python3
"""Create a deterministic complete-delivery archive and checksums."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

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
    "docs/screenshots",
    "plugins",
    "integrations",
    "scripts",
)
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".DS_Store", "build", "dist"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".sqlite3"}
SENSITIVE_PARTS = {".dsh", ".ssh"}
SENSITIVE_NAMES = {".env", ".npmrc", ".pypirc"}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlink_components(root: Path, path: Path, label: str) -> None:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"symlink is not allowed for {label}: {current}")


def project_version(root: Path) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    if match is None:
        raise RuntimeError("project version is missing")
    return match.group(1)


def packed_plugin(root: Path, version: str) -> Path:
    plugin_dist = root / "integrations/dsh_ecology_plugin/dist"
    _reject_symlink_components(root, plugin_dist, "packed DSH plugin directory")
    candidates = sorted(plugin_dist.glob("ecologyrsi-dsh-evolution-plugin-*.tgz"))
    expected_name = f"ecologyrsi-dsh-evolution-plugin-{version}.tgz"
    if len(candidates) != 1 or candidates[0].name != expected_name:
        names = ", ".join(path.name for path in candidates) or "none"
        raise RuntimeError(
            f"exactly one packed DSH plugin is required for version {version}; "
            f"found: {names}"
        )
    plugin = candidates[0]
    _reject_symlink_components(root, plugin, "packed DSH plugin")
    return plugin


def _is_sensitive_source(relative: Path) -> bool:
    folded_parts = tuple(part.casefold() for part in relative.parts)
    name = relative.name.casefold()
    return (
        any(part in SENSITIVE_PARTS for part in folded_parts)
        or name in SENSITIVE_NAMES
        or name.startswith(".env.")
        or "credential" in name
        or relative.suffix.casefold() in SENSITIVE_SUFFIXES
    )


def _git_ignored_sources(root: Path, paths: list[Path]) -> set[str]:
    repository = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if repository.returncode != 0:
        return set()
    names = [path.relative_to(root).as_posix() for path in paths]
    if not names:
        return set()
    checked = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--no-index", "-z", "--stdin"],
        input="\0".join(names) + "\0",
        check=False,
        capture_output=True,
        text=True,
    )
    if checked.returncode not in {0, 1}:
        raise RuntimeError("cannot evaluate git-ignore rules for public source")
    return {name for name in checked.stdout.split("\0") if name}


def included_source_files(root: Path) -> list[Path]:
    explicit_files = [root / name for name in ROOT_FILES]
    explicit_files.append(packed_plugin(root, project_version(root)))
    for path in explicit_files:
        _reject_symlink_components(root, path, "delivery source")
    files = list(explicit_files)
    candidates: list[Path] = []
    for directory in SOURCE_DIRS:
        source_directory = root / directory
        _reject_symlink_components(root, source_directory, "delivery source directory")
        for path in source_directory.rglob("*"):
            relative = path.relative_to(root)
            if any(
                part in IGNORED_PARTS or part.endswith(".egg-info")
                for part in relative.parts
            ):
                continue
            if path.suffix in IGNORED_SUFFIXES or _is_sensitive_source(relative):
                continue
            candidates.append(path)
    ignored = _git_ignored_sources(root, candidates)
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        if relative in ignored:
            continue
        if path.is_symlink():
            raise RuntimeError(f"symlink is not allowed for delivery source: {path}")
        if path.is_file():
            files.append(path)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError("missing delivery inputs: " + ", ".join(missing))
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def _command_version(command: list[str]) -> str:
    try:
        value = subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot determine tool version: {command[0]}") from exc
    if not value:
        raise RuntimeError(f"empty tool version: {command[0]}")
    return value


def build_info(root: Path, version: str) -> dict[str, object]:
    commit = _command_version(["git", "-C", str(root), "rev-parse", "HEAD"])
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("build commit must be 40 lowercase hexadecimal characters")
    dirty_output = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
        text=True,
    )
    source_date_text = os.environ.get("SOURCE_DATE_EPOCH", "0")
    try:
        source_date_epoch = int(source_date_text)
    except ValueError as exc:
        raise RuntimeError("SOURCE_DATE_EPOCH must be an integer") from exc
    if not 0 <= source_date_epoch <= 2**63 - 1:
        raise RuntimeError("SOURCE_DATE_EPOCH is outside the supported range")
    return {
        "schema_version": "ecologyrsi-dsh.build-info/1",
        "version": version,
        "commit": commit,
        "dirty": bool(dirty_output),
        "source_date_epoch": source_date_epoch,
        "tools": {
            "python": _command_version([sys.executable, "--version"]),
            "node": _command_version(["node", "--version"]),
            "npm": _command_version(["npm", "--version"]),
            "uv": _command_version(["uv", "--version"]),
        },
    }


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
    plugin = packed_plugin(root, version)
    distributed_plugin = dist / plugin.name
    shutil.copyfile(plugin, distributed_plugin)

    output = dist / f"ecologyrsi-dsh-{version}-delivery.tar.gz"
    prefix = f"ecologyrsi-dsh-{version}"
    timestamp = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    build_info_path = dist / "BUILD-INFO.json"
    build_info_data = (
        json.dumps(build_info(root, version), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    build_info_path.write_bytes(build_info_data)
    internal_checksums = (
        f"{sha256(wheel)}  artifacts/{wheel.name}\n"
        f"{sha256(sdist)}  artifacts/{sdist.name}\n"
        f"{sha256(distributed_plugin)}  artifacts/{distributed_plugin.name}\n"
        f"{hashlib.sha256(build_info_data).hexdigest()}  BUILD-INFO.json\n"
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
                    f"{prefix}/BUILD-INFO.json",
                    build_info_data,
                    mtime=timestamp,
                )
                add_bytes(
                    archive,
                    f"{prefix}/SHA256SUMS",
                    internal_checksums,
                    mtime=timestamp,
                )

    checksums = dist / "SHA256SUMS"
    artifacts = (wheel, sdist, distributed_plugin, output, build_info_path)
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
