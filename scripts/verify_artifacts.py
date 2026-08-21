#!/usr/bin/env python3
"""Inspect and smoke-test the wheel, sdist, and complete delivery archive."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any
from urllib.request import urlopen
import zipfile

from create_delivery_archive import included_source_files, project_version


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
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


def _matching_member(archive: Any, member_name: str, source: Path, label: str) -> None:
    try:
        data = archive.read(member_name)
    except KeyError as exc:
        raise RuntimeError(f"{label} is missing current source file: {member_name}") from exc
    if data != source.read_bytes():
        raise RuntimeError(f"{label} is stale relative to current source: {member_name}")


def verify_wheel(wheel: Path, version: str, source_root: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert_suffixes(
            names,
            (
                "ecologyrsi_dsh/__init__.py",
                "ecologyrsi_dsh/server.py",
                "ecologyrsi_dsh/api/generation_execution.py",
                "ecologyrsi_dsh/api/auto_progress.py",
                "ecologyrsi_dsh/api/events.py",
                "ecologyrsi_dsh/api/dsh_tools.py",
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
                "share/ecologyrsi-dsh/integrations/dsh_ecology_plugin/presets/ecology-coordinator-v1/preset.yml",
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
            _matching_member(archive, member_name, source, "wheel")

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
        wheel_sources += [source_root / "plugins/ecology_evolution/test/smoke.mjs"]
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
            _matching_member(archive, f"{data_root}/{relative}", source, "wheel")


def verify_sdist(sdist: Path, source_root: Path, version: str) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        names = set(archive.getnames())
        prefix = f"ecologyrsi_dsh-{version}"
        for source in included_source_files(source_root):
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
            "/integrations/dsh_ecology_plugin/presets/ecology-coordinator-v1/preset.yml",
            f"/integrations/dsh_ecology_plugin/dist/ecologyrsi-dsh-evolution-plugin-{version}.tgz",
            "/integrations/dsh_ecology_plugin/test/proxy_security.mjs",
            "/scripts/install_dsh_ecology_runtime.mjs",
            "/scripts/build_delivery.sh",
            "/scripts/verify_delivery.sh",
            "/src/ecologyrsi_dsh/server.py",
            "/src/ecologyrsi_dsh/api/generation_execution.py",
            "/src/ecologyrsi_dsh/core/state.py",
            "/src/ecologyrsi_dsh/evolution/analysis.py",
            "/src/ecologyrsi_dsh/evolution/batches.py",
            "/src/ecologyrsi_dsh/evolution/genome.py",
            "/src/ecologyrsi_dsh/evaluators/uncertainty.py",
            "/src/ecologyrsi_dsh/integrations/dsh_native_runtime.py",
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
    version: str,
    source_root: Path,
) -> None:
    prefix = f"ecologyrsi-dsh-{version}"
    with tarfile.open(delivery, "r:gz") as archive:
        names = set(archive.getnames())
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
                "/integrations/dsh_ecology_plugin/lib/runtime/stage-runner.js",
                "/integrations/dsh_ecology_plugin/presets/ecology-coordinator-v1/preset.yml",
                "/scripts/install_dsh_ecology_runtime.mjs",
                "/integrations/dsh_ecology_plugin/test/proxy_security.mjs",
                "/src/ecologyrsi_dsh/server.py",
                "/src/ecologyrsi_dsh/api/generation_execution.py",
                "/src/ecologyrsi_dsh/core/state.py",
                "/src/ecologyrsi_dsh/evolution/analysis.py",
                "/src/ecologyrsi_dsh/evolution/batches.py",
                "/src/ecologyrsi_dsh/knowledge/retrieval.py",
                "/src/ecologyrsi_dsh/knowledge/ecology_algorithms.json",
                f"/artifacts/{wheel.name}",
                f"/artifacts/{sdist.name}",
                f"/artifacts/{plugin.name}",
                "/SHA256SUMS",
            ),
            "delivery archive",
        )
        for source in included_source_files(source_root):
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


def verify_external_checksums(dist: Path, artifacts: tuple[Path, ...]) -> None:
    sums = parse_checksums((dist / "SHA256SUMS").read_bytes())
    for artifact in artifacts:
        if sha256(artifact) != sums.get(artifact.name):
            raise RuntimeError(f"external checksum mismatch: {artifact.name}")


def verify_npm_plugin(plugin: Path, version: str, source_root: Path) -> None:
    with tarfile.open(plugin, "r:gz") as archive:
        names = set(archive.getnames())
        required = {
            "package/package.json",
            "package/lib/index.js",
            "package/lib/tools/agent-plugin.js",
            "package/lib/runtime/stage-runner.js",
            "package/lib/runtime/reconciliation.js",
            "package/schemas/genome-mutation.schema.json",
            "package/presets/ecology-coordinator-v1/preset.yml",
            "package/presets/ecology-generation-judge-v1/agent.cordis.yml",
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
                if required_version != "0.1.0-rc.6":
                    raise RuntimeError("npm plugin DSH peer dependency is not exact rc.6")
                if peer_meta.get(name, {}).get("optional") is not True:
                    raise RuntimeError("npm plugin DSH peer must be host-provided and optional")
        exports = package.get("exports", {})
        if exports.get(".") != "./lib/index.js" or exports.get("./agent-plugin") != "./lib/tools/agent-plugin.js":
            raise RuntimeError("npm plugin Host/agent-plane exports are incomplete")
        forbidden = ("credential", "session.jsonl", ".sqlite", ".log", ".env", ".dsh/")
        if any(any(token in name.casefold() for token in forbidden) for name in names):
            raise RuntimeError("npm plugin contains private runtime material")
        for member_name in required - {"package/package.json"}:
            source = source_root / "integrations/dsh_ecology_plugin" / member_name.removeprefix("package/")
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
                / "ecology-coordinator-v1"
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
    dist = args.dist.resolve()
    source_root = Path(__file__).resolve().parents[1]
    expected_version = project_version(source_root)
    wheel = one(sorted(dist.glob("ecologyrsi_dsh-*.whl")), "wheel")
    sdist = one(sorted(dist.glob("ecologyrsi_dsh-*.tar.gz")), "sdist")
    delivery = one(sorted(dist.glob("ecologyrsi-dsh-*-delivery.tar.gz")), "delivery archive")
    plugin = one(sorted(dist.glob("ecologyrsi-dsh-evolution-plugin-*.tgz")), "npm plugin")
    version = wheel.name.split("-")[1]
    if version != expected_version:
        raise RuntimeError(
            f"release artifacts are version {version}, current source is {expected_version}"
        )

    verify_wheel(wheel, version, source_root)
    verify_sdist(sdist, source_root, version)
    verify_npm_plugin(plugin, version, source_root)
    verify_delivery_archive(delivery, wheel, sdist, plugin, version, source_root)
    verify_external_checksums(dist, (wheel, sdist, plugin, delivery))
    installed_smoke(wheel, version)
    print(f"release artifact verification: ok ({version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
