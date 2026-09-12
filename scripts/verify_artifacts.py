#!/usr/bin/env python3
"""Inspect and smoke-test the wheel, sdist, and complete delivery archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import urlopen

from create_delivery_archive import (
    included_source_files,
    is_sensitive_source,
    packed_plugin,
    project_version,
)

try:
    from .release_safety import checked_lstat, require_regular_file
except ImportError:
    from release_safety import checked_lstat, require_regular_file

INTERNAL_SOURCE_MARKERS = (
    "/docs/superpowers/",
    "/docs/项目整体Review与方案B优化报告.md",
)

CURRENT_DSH_PRESET_IDS = frozenset(
    {
        "ecology-coordinator-v5",
        "ecology-researcher-v12",
        "ecology-candidate-proposer-v4",
        "ecology-sample-planner-v9",
        "ecology-sample-critic-v5",
        "ecology-generation-judge-v8",
    }
)
_MANAGED_DSH_PRESET_ID = re.compile(
    r"ecology-(?:coordinator|researcher|candidate-proposer|sample-planner|sample-critic|generation-judge|local-editor)-v[0-9]+"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    require_regular_file(path.parent, path, "checksum input")
    return sha256_bytes(path.read_bytes())


def one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        names = ", ".join(path.name for path in paths) or "none"
        raise RuntimeError(f"expected one {label}, found: {names}")
    return paths[0]


def assert_suffixes(names: set[str], suffixes: tuple[str, ...], label: str) -> None:
    missing = [suffix for suffix in suffixes if not any(name.endswith(suffix) for name in names)]
    if missing:
        raise RuntimeError(f"{label} is missing: {', '.join(missing)}")


def assert_current_dsh_presets(names: set[str], label: str) -> None:
    found: set[str] = set()
    for name in names:
        parts = PurePosixPath(name).parts
        for index, part in enumerate(parts[:-1]):
            if part != "presets":
                continue
            candidate = parts[index + 1]
            if _MANAGED_DSH_PRESET_ID.fullmatch(candidate):
                found.add(candidate)
    if found != CURRENT_DSH_PRESET_IDS:
        missing = sorted(CURRENT_DSH_PRESET_IDS - found)
        obsolete = sorted(found - CURRENT_DSH_PRESET_IDS)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if obsolete:
            details.append("obsolete: " + ", ".join(obsolete))
        raise RuntimeError(f"{label} DSH preset inventory differs ({'; '.join(details)})")


def reject_internal_sources(names: set[str], label: str) -> None:
    leaked = sorted(
        name
        for name in names
        if any(marker in f"/{name}" for marker in INTERNAL_SOURCE_MARKERS)
    )
    if leaked:
        raise RuntimeError(f"{label} contains internal source: {', '.join(leaked)}")


def _reject_tar_links(archive: tarfile.TarFile, label: str) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    linked = [member.name for member in members if member.issym() or member.islnk()]
    if linked:
        raise RuntimeError(f"{label} contains linked member: {', '.join(linked)}")
    unsupported = [
        member.name
        for member in members
        if not member.isfile() and not member.isdir()
    ]
    if unsupported:
        raise RuntimeError(
            f"{label} contains unsupported member: {', '.join(unsupported)}"
        )
    return members


def _reject_sensitive_members(names: set[str], label: str) -> None:
    sensitive = sorted(name for name in names if is_sensitive_source(Path(name)))
    if sensitive:
        raise RuntimeError(f"{label} contains sensitive member: {', '.join(sensitive)}")


def _archive_parent_directories(names: set[str]) -> set[str]:
    directories: set[str] = set()
    for name in names:
        parent = PurePosixPath(name).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _matching_member(
    archive: Any,
    member_name: str,
    source_root: Path,
    source: Path,
    label: str,
) -> None:
    require_regular_file(source_root, source, f"{label} source")
    try:
        data = archive.read(member_name)
    except KeyError as exc:
        raise RuntimeError(f"{label} is missing current source file: {member_name}") from exc
    if data != source.read_bytes():
        raise RuntimeError(f"{label} is stale relative to current source: {member_name}")


def verify_wheel(wheel: Path, version: str, source_root: Path) -> None:
    require_regular_file(wheel.parent, wheel, "wheel archive")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert_current_dsh_presets(names, "wheel")
        reject_internal_sources(names, "wheel")
        if any(name.endswith("/plugins/ecology_evolution/test/smoke.mjs") for name in names):
            raise RuntimeError("wheel must not include the browser smoke test")
        assert_suffixes(
            names,
            (
                "ecologyrsi_dsh/__init__.py",
                "ecologyrsi_dsh/application/cli.py",
                "ecologyrsi_dsh/api/handler.py",
                "ecologyrsi_dsh/api/runtime.py",
                "ecologyrsi_dsh/application/candidate_scheduler.py",
                "ecologyrsi_dsh/application/generation_execution.py",
                "ecologyrsi_dsh/api/auto_progress.py",
                "ecologyrsi_dsh/api/events.py",
                "ecologyrsi_dsh/integrations/dsh_tools.py",
                "ecologyrsi_dsh/core/sample_results.py",
                "ecologyrsi_dsh/core/state.py",
                "ecologyrsi_dsh/evaluators/gateway_sample_adapter.py",
                "ecologyrsi_dsh/evaluators/sample_execution.py",
                "ecologyrsi_dsh/evaluators/dsh_sample_adapter.py",
                "ecologyrsi_dsh/evaluators/fitness.py",
                "ecologyrsi_dsh/evaluators/uncertainty.py",
                "ecologyrsi_dsh/evolution/analysis.py",
                "ecologyrsi_dsh/evolution/batches.py",
                "ecologyrsi_dsh/evolution/execution_plan.py",
                "ecologyrsi_dsh/evolution/genome.py",
                "ecologyrsi_dsh/evolution/workflow_ir.py",
                "ecologyrsi_dsh/integrations/dsh_native_runtime.py",
                "ecologyrsi_dsh/integrations/dsh_structured_roles.py",
                "ecologyrsi_dsh/knowledge/algorithm_ir.py",
                "ecologyrsi_dsh/knowledge/algorithm_smoke.py",
                "ecologyrsi_dsh/knowledge/algorithms.py",
                "ecologyrsi_dsh/knowledge/autonomous_cycle.py",
                "ecologyrsi_dsh/knowledge/research_iteration.py",
                "ecologyrsi_dsh/knowledge/retrieval.py",
                "ecologyrsi_dsh/knowledge/ecology_algorithms.json",
                "share/ecologyrsi-dsh/datasets/autonomous_greenhouse.json",
                "share/ecologyrsi-dsh/plugins/ecology_evolution/index.html",
                "share/ecologyrsi-dsh/plugins/ecology_evolution/app.js",
                "share/ecologyrsi-dsh/plugins/ecology_evolution/assets/js/host.js",
                "share/ecologyrsi-dsh/plugins/ecology_evolution/assets/js/core.js",
                "share/ecologyrsi-dsh/plugins/ecology_evolution/assets/js/commands.js",
                "share/ecologyrsi-dsh/plugins/ecology_evolution/assets/js/data.js",
                "share/ecologyrsi-dsh/plugins/ecology_evolution/assets/js/render_candidates.js",
                "share/ecologyrsi-dsh/plugins/ecology_evolution/assets/js/render_process.js",
                "share/ecologyrsi-dsh/plugins/ecology_evolution/assets/js/render_training_trace.js",
                "share/ecologyrsi-dsh/plugins/ecology_evolution/plugin.json",
                "share/ecologyrsi-dsh/plugins/ecology_evolution/styles.css",
                "share/ecologyrsi-dsh/integrations/dsh_ecology_plugin/package.json",
                "share/ecologyrsi-dsh/integrations/dsh_ecology_plugin/lib/index.js",
                "share/ecologyrsi-dsh/integrations/dsh_ecology_plugin/lib/client.js",
                "share/ecologyrsi-dsh/integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js",
                "share/ecologyrsi-dsh/integrations/dsh_ecology_plugin/lib/tools/agent-plugin.js",
                "share/ecologyrsi-dsh/integrations/dsh_ecology_plugin/schemas/genome-mutation.schema.json",
                "share/ecologyrsi-dsh/integrations/dsh_ecology_plugin/presets/ecology-coordinator-v5/preset.yml",
                f"share/ecologyrsi-dsh/integrations/dsh_ecology_plugin/dist/ecologyrsi-dsh-evolution-plugin-{version}.tgz",
                "share/ecologyrsi-dsh/scripts/install_dsh_ecology_runtime.mjs",
                ".dist-info/licenses/LICENSE",
                ".dist-info/licenses/NOTICE",
                ".dist-info/entry_points.txt",
                ".dist-info/METADATA",
            ),
            "wheel",
        )
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_name))
        if metadata["Name"] != "ecologyrsi-dsh" or metadata["Version"] != version:
            raise RuntimeError("wheel name/version metadata mismatch")
        if metadata["License-Expression"] != "LicenseRef-Proprietary":
            raise RuntimeError("wheel must carry the proprietary license expression")
        if metadata.get_all("Requires-Dist"):
            raise RuntimeError("wheel must not declare runtime dependencies")

        for source in sorted((source_root / "src/ecologyrsi_dsh").rglob("*")):
            if not source.is_file() or source.suffix not in {".py", ".json"}:
                continue
            member_name = source.relative_to(source_root / "src").as_posix()
            _matching_member(archive, member_name, source_root, source, "wheel")

        data_root = f"ecologyrsi_dsh-{version}.data/data/share/ecologyrsi-dsh"
        wheel_sources = [source_root / "datasets/autonomous_greenhouse.json"]
        wheel_sources += [
            path
            for path in sorted((source_root / "plugins/ecology_evolution").glob("*"))
            if path.is_file() and path.name in {
                "README.md", "app.js", "index.html", "plugin.json", "styles.css"
            }
        ]
        wheel_sources += sorted((source_root / "plugins/ecology_evolution/assets/js").glob("*.js"))
        integration_root = source_root / "integrations/dsh_ecology_plugin"
        wheel_sources += [integration_root / "README.md", integration_root / "package.json"]
        for directory in ("lib", "schemas", "presets", "dist"):
            wheel_sources += [
                path
                for path in sorted((integration_root / directory).rglob("*"))
                if path.is_file()
                and (
                    directory != "dist"
                    or path.name
                    == f"ecologyrsi-dsh-evolution-plugin-{version}.tgz"
                )
            ]
        wheel_sources += [source_root / "scripts/install_dsh_ecology_runtime.mjs"]
        for source in wheel_sources:
            relative = source.relative_to(source_root).as_posix()
            _matching_member(
                archive, f"{data_root}/{relative}", source_root, source, "wheel"
            )


def verify_sdist(sdist: Path, source_root: Path, version: str) -> None:
    require_regular_file(sdist.parent, sdist, "sdist archive")
    with tarfile.open(sdist, "r:gz") as archive:
        members = _reject_tar_links(archive, "sdist")
        names = {member.name for member in members}
        assert_current_dsh_presets(names, "sdist")
        reject_internal_sources(names, "sdist")
        _reject_sensitive_members(names, "sdist")
        prefix = f"ecologyrsi_dsh-{version}"
        sources = included_source_files(source_root)
        expected_files = {
            f"{prefix}/{source.relative_to(source_root).as_posix()}"
            for source in sources
            if source.relative_to(source_root).as_posix() != ".gitignore"
        }
        expected_files.update(
            {
                f"{prefix}/PKG-INFO",
                f"{prefix}/setup.cfg",
                f"{prefix}/src/ecologyrsi_dsh.egg-info/PKG-INFO",
                f"{prefix}/src/ecologyrsi_dsh.egg-info/SOURCES.txt",
                f"{prefix}/src/ecologyrsi_dsh.egg-info/dependency_links.txt",
                f"{prefix}/src/ecologyrsi_dsh.egg-info/entry_points.txt",
                f"{prefix}/src/ecologyrsi_dsh.egg-info/top_level.txt",
            }
        )
        expected_directories = _archive_parent_directories(expected_files)
        overlap = expected_files & expected_directories
        if overlap:
            raise RuntimeError(
                "sdist expected member types overlap: " + ", ".join(sorted(overlap))
            )
        wrong_type = sorted(
            member.name
            for member in members
            if (
                (member.name in expected_files and not member.isfile())
                or (
                    member.name in expected_directories
                    and not member.isdir()
                )
            )
        )
        if wrong_type:
            raise RuntimeError(
                "sdist contains unexpected member type: "
                + ", ".join(wrong_type)
            )
        unexpected = sorted(names - expected_files - expected_directories)
        if unexpected:
            raise RuntimeError(
                "sdist contains unexpected member: " + ", ".join(unexpected)
            )
        for source in sources:
            relative = source.relative_to(source_root).as_posix()
            if relative == ".gitignore":
                continue
            member_name = f"{prefix}/{relative}"
            try:
                member = archive.extractfile(member_name)
            except KeyError as exc:
                raise RuntimeError(
                    f"sdist is missing current source file: {relative}"
                ) from exc
            if member is None or member.read() != source.read_bytes():
                raise RuntimeError(f"sdist is stale relative to current source: {relative}")
    assert_suffixes(
        names,
        (
            "/README.md",
            "/docs/screenshots/01-run-settings.jpg",
            "/docs/screenshots/02-parameter-design.jpg",
            "/docs/screenshots/03-training-data.jpg",
            "/docs/screenshots/04-evolution-process.jpg",
            "/docs/screenshots/05-candidate-evaluation.jpg",
            "/docs/screenshots/06-human-governance.jpg",
            "/CHANGELOG.md",
            "/LICENSE",
            "/NOTICE",
            "/RELEASE-CHECKLIST.md",
            "/MANIFEST.in",
            "/examples/minimal_run.py",
            "/examples/local-config.json",
            "/datasets/autonomous_greenhouse.json",
            "/plugins/ecology_evolution/index.html",
            "/plugins/ecology_evolution/app.js",
            "/plugins/ecology_evolution/assets/js/host.js",
            "/plugins/ecology_evolution/assets/js/core.js",
            "/plugins/ecology_evolution/assets/js/commands.js",
            "/plugins/ecology_evolution/assets/js/render_process.js",
            "/plugins/ecology_evolution/test/smoke.mjs",
            "/integrations/dsh_ecology_plugin/package.json",
            "/integrations/dsh_ecology_plugin/lib/index.js",
            "/integrations/dsh_ecology_plugin/lib/client.js",
            "/integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js",
            "/integrations/dsh_ecology_plugin/lib/tools/agent-plugin.js",
            "/integrations/dsh_ecology_plugin/schemas/genome-mutation.schema.json",
            "/integrations/dsh_ecology_plugin/presets/ecology-coordinator-v5/preset.yml",
            f"/integrations/dsh_ecology_plugin/dist/ecologyrsi-dsh-evolution-plugin-{version}.tgz",
            "/integrations/dsh_ecology_plugin/test/proxy_security.mjs",
            "/scripts/install_dsh_ecology_runtime.mjs",
            "/scripts/build_delivery.sh",
            "/scripts/verify_delivery.sh",
            "/src/ecologyrsi_dsh/application/cli.py",
            "/src/ecologyrsi_dsh/api/handler.py",
            "/src/ecologyrsi_dsh/api/runtime.py",
            "/src/ecologyrsi_dsh/application/candidate_scheduler.py",
            "/src/ecologyrsi_dsh/application/generation_execution.py",
            "/src/ecologyrsi_dsh/core/state.py",
            "/src/ecologyrsi_dsh/evolution/analysis.py",
            "/src/ecologyrsi_dsh/evolution/batches.py",
            "/src/ecologyrsi_dsh/evolution/genome.py",
            "/src/ecologyrsi_dsh/evaluators/uncertainty.py",
            "/src/ecologyrsi_dsh/integrations/dsh_native_runtime.py",
            "/src/ecologyrsi_dsh/knowledge/autonomous_cycle.py",
            "/src/ecologyrsi_dsh/knowledge/retrieval.py",
            "/src/ecologyrsi_dsh/knowledge/ecology_algorithms.json",
            "/tests/test_core.py",
        ),
        "sdist",
    )


def parse_checksums(data: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in data.decode("ascii").splitlines():
        digest, name = line.split(None, 1)
        result[name.strip()] = digest
    return result


def verify_delivery_archive(
    delivery: Path,
    wheel: Path,
    sdist: Path,
    plugin: Path,
    build_info_path: Path,
    version: str,
    source_root: Path,
) -> None:
    prefix = f"ecologyrsi-dsh-{version}"
    require_regular_file(delivery.parent, delivery, "delivery archive")
    for artifact, label in (
        (wheel, "wheel artifact"),
        (sdist, "sdist artifact"),
        (plugin, "npm plugin artifact"),
        (build_info_path, "BUILD-INFO.json"),
    ):
        require_regular_file(artifact.parent, artifact, label)
    with tarfile.open(delivery, "r:gz") as archive:
        members = _reject_tar_links(archive, "delivery archive")
        names = {member.name for member in members}
        assert_current_dsh_presets(names, "delivery archive")
        reject_internal_sources(names, "delivery archive")
        _reject_sensitive_members(names, "delivery archive")
        sources = included_source_files(source_root)
        expected_names = {
            f"{prefix}/{source.relative_to(source_root).as_posix()}"
            for source in sources
        }
        expected_names.update(
            {
                f"{prefix}/artifacts/{artifact.name}"
                for artifact in (wheel, sdist, plugin)
            }
        )
        expected_names.update(
            {f"{prefix}/BUILD-INFO.json", f"{prefix}/SHA256SUMS"}
        )
        unexpected = sorted(names - expected_names)
        if unexpected:
            raise RuntimeError(
                "delivery archive contains unexpected member: "
                + ", ".join(unexpected)
            )
        assert_suffixes(
            names,
            (
                "/README.md",
                "/docs/screenshots/01-run-settings.jpg",
                "/docs/screenshots/02-parameter-design.jpg",
                "/docs/screenshots/03-training-data.jpg",
                "/docs/screenshots/04-evolution-process.jpg",
                "/docs/screenshots/05-candidate-evaluation.jpg",
                "/docs/screenshots/06-human-governance.jpg",
                "/LICENSE",
                "/NOTICE",
                "/examples/minimal_run.py",
                "/datasets/autonomous_greenhouse.json",
                "/plugins/ecology_evolution/index.html",
                "/plugins/ecology_evolution/assets/js/core.js",
                "/plugins/ecology_evolution/assets/js/render_process.js",
                "/integrations/dsh_ecology_plugin/package.json",
                "/integrations/dsh_ecology_plugin/lib/index.js",
                "/integrations/dsh_ecology_plugin/lib/client.js",
                "/integrations/dsh_ecology_plugin/lib/tools/retrieval.js",
                "/integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js",
                "/integrations/dsh_ecology_plugin/presets/ecology-coordinator-v5/preset.yml",
                "/scripts/install_dsh_ecology_runtime.mjs",
                "/integrations/dsh_ecology_plugin/test/proxy_security.mjs",
                "/src/ecologyrsi_dsh/application/cli.py",
                "/src/ecologyrsi_dsh/api/handler.py",
                "/src/ecologyrsi_dsh/api/runtime.py",
                "/src/ecologyrsi_dsh/application/candidate_scheduler.py",
                "/src/ecologyrsi_dsh/application/generation_execution.py",
                "/src/ecologyrsi_dsh/core/state.py",
                "/src/ecologyrsi_dsh/evolution/analysis.py",
                "/src/ecologyrsi_dsh/evolution/batches.py",
                "/src/ecologyrsi_dsh/knowledge/autonomous_cycle.py",
                "/src/ecologyrsi_dsh/knowledge/retrieval.py",
                "/src/ecologyrsi_dsh/knowledge/ecology_algorithms.json",
                f"/artifacts/{wheel.name}",
                f"/artifacts/{sdist.name}",
                f"/artifacts/{plugin.name}",
                "/BUILD-INFO.json",
                "/SHA256SUMS",
            ),
            "delivery archive",
        )
        for source in sources:
            relative = source.relative_to(source_root).as_posix()
            member = archive.extractfile(f"{prefix}/{relative}")
            if member is None:
                raise RuntimeError(
                    f"delivery archive is missing current source file: {relative}"
                )
            if member.read() != source.read_bytes():
                raise RuntimeError(
                    f"delivery archive is stale relative to current source: {relative}"
                )
        sums_member = archive.extractfile(f"{prefix}/SHA256SUMS")
        if sums_member is None:
            raise RuntimeError("delivery SHA256SUMS cannot be read")
        sums = parse_checksums(sums_member.read())
        for artifact in (wheel, sdist, plugin):
            member = archive.extractfile(f"{prefix}/artifacts/{artifact.name}")
            if member is None:
                raise RuntimeError(f"delivery artifact cannot be read: {artifact.name}")
            data = member.read()
            if sha256_bytes(data) != sums.get(f"artifacts/{artifact.name}"):
                raise RuntimeError(f"delivery checksum mismatch: {artifact.name}")
        build_info_member = archive.extractfile(f"{prefix}/BUILD-INFO.json")
        if build_info_member is None:
            raise RuntimeError("delivery BUILD-INFO.json cannot be read")
        build_info_data = build_info_member.read()
        if build_info_data != build_info_path.read_bytes():
            raise RuntimeError("delivery BUILD-INFO.json differs from external copy")
        if sha256_bytes(build_info_data) != sums.get("BUILD-INFO.json"):
            raise RuntimeError("delivery BUILD-INFO.json checksum mismatch")


def verify_external_checksums(dist: Path, artifacts: tuple[Path, ...]) -> None:
    sums_path = require_regular_file(
        dist, dist / "SHA256SUMS", "external SHA256SUMS"
    )
    sums = parse_checksums(sums_path.read_bytes())
    for artifact in artifacts:
        if sha256(artifact) != sums.get(artifact.name):
            raise RuntimeError(f"external checksum mismatch: {artifact.name}")


def verify_build_info(path: Path, version: str, source_root: Path) -> None:
    require_regular_file(path.parent, path, "BUILD-INFO.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "ecologyrsi-dsh.build-info/1":
        raise RuntimeError("BUILD-INFO.json schema version mismatch")
    if payload.get("version") != version:
        raise RuntimeError("BUILD-INFO.json release version mismatch")
    commit = payload.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("BUILD-INFO.json commit is invalid")
    current_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != current_commit:
        raise RuntimeError("BUILD-INFO.json commit differs from current source")
    if payload.get("dirty") is not False:
        raise RuntimeError("release BUILD-INFO.json must record dirty=false")
    source_date_epoch = payload.get("source_date_epoch")
    if (
        not isinstance(source_date_epoch, int)
        or isinstance(source_date_epoch, bool)
        or not 0 <= source_date_epoch <= 2**63 - 1
    ):
        raise RuntimeError("BUILD-INFO.json source_date_epoch is invalid")
    tools = payload.get("tools")
    if not isinstance(tools, dict) or set(tools) != {"python", "node", "npm", "uv"}:
        raise RuntimeError("BUILD-INFO.json tool inventory is invalid")
    if any(not isinstance(value, str) or not value.strip() for value in tools.values()):
        raise RuntimeError("BUILD-INFO.json tool versions must be non-empty strings")


def verify_npm_plugin(plugin: Path, version: str, source_root: Path) -> None:
    require_regular_file(plugin.parent, plugin, "npm plugin archive")
    integration_root = source_root / "integrations/dsh_ecology_plugin"
    package_json = integration_root / "package.json"
    require_regular_file(source_root, package_json, "npm package source")
    for legal_name in ("LICENSE", "NOTICE"):
        legal_source = source_root / legal_name
        require_regular_file(source_root, legal_source, "legal source")
    selected_symlinks = [
        path for path in integration_root.rglob("*") if path.is_symlink()
    ]
    if selected_symlinks:
        raise RuntimeError(f"npm plugin source is a symlink: {selected_symlinks[0]}")
    with tarfile.open(plugin, "r:gz") as archive:
        members = _reject_tar_links(archive, "npm plugin")
        names = {member.name for member in members}
        assert_current_dsh_presets(names, "npm plugin")
        _reject_sensitive_members(names, "npm plugin")
        file_names = {
            member.name for member in members if member.isfile()
        }
        required = {
            "package/package.json",
            "package/LICENSE",
            "package/NOTICE",
            "package/lib/index.js",
            "package/lib/tools/agent-plugin.js",
            "package/lib/tools/retrieval.js",
            "package/lib/runtime/stage-runner.js",
            "package/schemas/genome-mutation.schema.json",
            "package/presets/ecology-coordinator-v5/preset.yml",
            "package/presets/ecology-generation-judge-v8/agent.cordis.yml",
            "package/presets/ecology-generation-judge-v8/skills/batch-scientific-reflection/SKILL.md",
            "package/presets/ecology-generation-judge-v8/skills/candidate-scientific-review/SKILL.md",
        }
        missing = sorted(required - names)
        if missing:
            raise RuntimeError("npm plugin is missing: " + ", ".join(missing))
        package_member = archive.extractfile("package/package.json")
        if package_member is None:
            raise RuntimeError("npm plugin package.json cannot be read")
        package = json.loads(package_member.read())
        if package.get("version") != version:
            raise RuntimeError("npm plugin version mismatch")
        dependencies = package.get("dependencies")
        if dependencies != {}:
            raise RuntimeError("npm plugin must not install duplicate DSH runtime packages")
        peers = package.get("peerDependencies", {})
        peer_meta = package.get("peerDependenciesMeta", {})
        for name, required_version in peers.items():
            if name.startswith("@deepseek-ai/dsh-"):
                if required_version != "0.1.5-rc.2":
                    raise RuntimeError(
                        "npm plugin DSH peer dependency is not exact 0.1.5-rc.2"
                    )
                if peer_meta.get(name, {}).get("optional") is not True:
                    raise RuntimeError("npm plugin DSH peer must be host-provided and optional")
        exports = package.get("exports", {})
        if exports.get(".") != "./lib/index.js" or exports.get("./agent-plugin") != "./lib/tools/agent-plugin.js":
            raise RuntimeError("npm plugin Host/agent-plane exports are incomplete")
        forbidden = ("credential", "session.jsonl", ".sqlite", ".log", ".env", ".dsh/")
        if any(any(token in name.casefold() for token in forbidden) for name in names):
            raise RuntimeError("npm plugin contains private runtime material")

        expected_sources = {
            "package/package.json": integration_root / "package.json",
            "package/LICENSE": source_root / "LICENSE",
            "package/NOTICE": source_root / "NOTICE",
        }
        patterns = package.get("files")
        if not isinstance(patterns, list) or not all(
            isinstance(pattern, str) and pattern for pattern in patterns
        ):
            raise RuntimeError("npm plugin files contract is invalid")
        for pattern in patterns:
            if pattern in {"LICENSE", "NOTICE"}:
                continue
            for source in integration_root.glob(pattern):
                require_regular_file(source_root, source, "npm plugin source")
                if source.is_file():
                    relative = source.relative_to(integration_root).as_posix()
                    expected_sources[f"package/{relative}"] = source
        missing_selected = sorted(set(expected_sources) - file_names)
        if missing_selected:
            raise RuntimeError(
                "npm plugin is missing selected source: "
                + ", ".join(missing_selected)
            )
        unexpected = sorted(file_names - set(expected_sources))
        if unexpected:
            raise RuntimeError(
                "npm plugin contains unselected source: " + ", ".join(unexpected)
            )
        for member_name, source in expected_sources.items():
            require_regular_file(source_root, source, "npm plugin source")
            member = archive.extractfile(member_name)
            if member is None or member.read() != source.read_bytes():
                raise RuntimeError(f"npm plugin is stale: {member_name}")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def installed_smoke(wheel: Path, version: str) -> None:
    with tempfile.TemporaryDirectory(prefix="ecologyrsi-dsh-wheel-") as directory:
        root = Path(directory)
        environment = root / "venv"
        clean_environment = os.environ.copy()
        clean_environment.pop("PYTHONHOME", None)
        clean_environment.pop("PYTHONPATH", None)
        subprocess.run(
            [sys.executable, "-m", "venv", str(environment)],
            check=True,
            env=clean_environment,
        )
        bindir = environment / ("Scripts" if sys.platform == "win32" else "bin")
        python = bindir / ("python.exe" if sys.platform == "win32" else "python")
        command = bindir / ("ecologyrsi-dsh.exe" if sys.platform == "win32" else "ecologyrsi-dsh")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=clean_environment,
        )
        installed_version = subprocess.check_output(
            [str(python), "-c", "from importlib.metadata import version; print(version('ecologyrsi-dsh'))"],
            text=True,
            env=clean_environment,
        ).strip()
        if installed_version != version:
            raise RuntimeError("installed wheel version mismatch")
        knowledge_count = subprocess.check_output(
            [
                str(python),
                "-c",
                (
                    "from ecologyrsi_dsh.knowledge.retrieval import _catalog; "
                    "print(len(_catalog()))"
                ),
            ],
            text=True,
            env=clean_environment,
        ).strip()
        if int(knowledge_count) < 5:
            raise RuntimeError("installed wheel knowledge catalog is missing")

        if sys.platform != "win32":
            fake_dsh = root / "fake-dsh"
            fake_dsh.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake_dsh.chmod(0o755)
            dsh_home = root / "dsh-home"
            installed = subprocess.run(
                [
                    str(command),
                    "install-dsh-runtime",
                    "--profile",
                    "web",
                    "--dsh-home",
                    str(dsh_home),
                    "--dsh-bin",
                    str(fake_dsh),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=clean_environment,
            )
            if installed.returncode != 0:
                raise RuntimeError(
                    "installed wheel DSH installer failed:\n" + installed.stdout
                )
            if not (
                dsh_home
                / ".agent-presets"
                / "ecology-coordinator-v5"
                / "preset.yml"
            ).is_file():
                raise RuntimeError("installed wheel did not install DSH presets")
            patch = dsh_home / "profiles" / "web" / "cordis.patch.yml"
            if not patch.is_file() or "@ecologyrsi/dsh-evolution-plugin" not in patch.read_text(encoding="utf-8"):
                raise RuntimeError("installed wheel did not install the managed Host patch")

        output = subprocess.check_output(
            [
                str(command),
                "demo",
                "--db",
                str(root / "demo.sqlite3"),
                "--run-id",
                "run:wheel-verification",
                "--candidates",
                "2",
            ],
            text=True,
            env=clean_environment,
        )
        payload = json.loads(output)
        if payload["run"]["status"] != "completed" or payload["event_count"] < 8:
            raise RuntimeError("installed CLI demo did not complete")
        bundle = root / "wheel-export.json"
        subprocess.run(
            [
                str(command),
                "export",
                "run:wheel-verification",
                "--db",
                str(root / "demo.sqlite3"),
                "--output",
                str(bundle),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=clean_environment,
        )
        verified = subprocess.run(
            [str(command), "verify", str(bundle)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=clean_environment,
        )
        if verified.returncode != 0 or not json.loads(verified.stdout).get("valid"):
            raise RuntimeError("installed CLI export verification failed")
        imported = subprocess.run(
            [str(command), "import", str(bundle), "--db", str(root / "replayed.sqlite3")],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=clean_environment,
        )
        if imported.returncode != 0 or json.loads(imported.stdout).get("status") != "completed":
            raise RuntimeError("installed CLI export import failed")

        port = free_port()
        process = subprocess.Popen(
            [
                str(command),
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--db",
                str(root / "server.sqlite3"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=clean_environment,
        )
        try:
            deadline = time.monotonic() + 8.0
            health = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    raise RuntimeError(f"installed server exited early\n{stdout}\n{stderr}")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.5) as response:
                        health = json.loads(response.read())
                    break
                except OSError:
                    time.sleep(0.1)
            if not isinstance(health, dict) or health.get("ok") is not True:
                raise RuntimeError("installed server health check failed")
            if health.get("package_version") != version:
                raise RuntimeError("installed server version mismatch")
            with urlopen(
                f"http://127.0.0.1:{port}/plugins/ecology/evolution/", timeout=2
            ) as response:
                html = response.read().decode("utf-8")
            with urlopen(
                f"http://127.0.0.1:{port}/plugins/ecology/evolution/app.js", timeout=2
            ) as response:
                javascript = response.read().decode("utf-8")
            with urlopen(
                f"http://127.0.0.1:{port}/plugins/ecology/evolution/assets/js/core.js",
                timeout=2,
            ) as response:
                core_javascript = response.read().decode("utf-8")
            with urlopen(
                f"http://127.0.0.1:{port}/plugins/ecology/evolution/assets/js/host.js",
                timeout=2,
            ) as response:
                host_javascript = response.read().decode("utf-8")
            with urlopen(
                f"http://127.0.0.1:{port}/api/plugin/ecology_evolution", timeout=2
            ) as response:
                plugin_manifest = json.loads(response.read())
            if (
                "生态模型进化工作台" not in html
                or "EcologyEvolutionPlugin" not in javascript
                or "function request" not in core_javascript
                or "EcologyDSHHost" not in host_javascript
            ):
                raise RuntimeError("installed wheel did not serve plugin assets")
            if plugin_manifest.get("display_name") != "生态模型进化工作台":
                raise RuntimeError("installed wheel did not serve the Chinese plugin manifest")
        finally:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    args = parser.parse_args()
    dist = args.dist.absolute()
    dist_stat = checked_lstat(dist.parent, dist, "artifact directory")
    if dist_stat is None or not stat.S_ISDIR(dist_stat.st_mode):
        raise RuntimeError(f"artifact directory is not a directory: {dist}")
    source_root = Path(__file__).resolve().parents[1]
    expected_version = project_version(source_root)
    wheel = one(sorted(dist.glob("ecologyrsi_dsh-*.whl")), "wheel")
    sdist = one(sorted(dist.glob("ecologyrsi_dsh-*.tar.gz")), "sdist")
    delivery = one(sorted(dist.glob("ecologyrsi-dsh-*-delivery.tar.gz")), "delivery archive")
    plugin = one(sorted(dist.glob("ecologyrsi-dsh-evolution-plugin-*.tgz")), "npm plugin")
    build_info_path = one(sorted(dist.glob("BUILD-INFO.json")), "BUILD-INFO.json")
    require_regular_file(dist, wheel, "wheel archive")
    require_regular_file(dist, sdist, "sdist archive")
    require_regular_file(dist, delivery, "delivery archive")
    require_regular_file(dist, plugin, "npm plugin archive")
    require_regular_file(dist, build_info_path, "BUILD-INFO.json")
    version = wheel.name.split("-")[1]
    if version != expected_version:
        raise RuntimeError(
            f"release artifacts are version {version}, current source is {expected_version}"
        )
    expected_plugin_name = f"ecologyrsi-dsh-evolution-plugin-{version}.tgz"
    if plugin.name != expected_plugin_name:
        raise RuntimeError(
            f"npm plugin filename {plugin.name!r} does not match version {version}"
        )
    source_plugin = packed_plugin(source_root, version)
    verify_npm_plugin(plugin, version, source_root)
    if source_plugin.read_bytes() != plugin.read_bytes():
        raise RuntimeError("distributed npm plugin differs from nested source plugin")

    verify_wheel(wheel, version, source_root)
    verify_sdist(sdist, source_root, version)
    verify_build_info(build_info_path, version, source_root)
    verify_delivery_archive(
        delivery,
        wheel,
        sdist,
        plugin,
        build_info_path,
        version,
        source_root,
    )
    verify_external_checksums(
        dist, (wheel, sdist, plugin, delivery, build_info_path)
    )
    installed_smoke(wheel, version)
    print(f"release artifact verification: ok ({version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
