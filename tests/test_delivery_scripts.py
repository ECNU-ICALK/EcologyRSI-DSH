from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.create_delivery_archive import (
    ROOT_FILES,
    create_archive,
    included_source_files,
    project_version,
)

ROOT = Path(__file__).resolve().parents[1]

PRIVATE_HOME_PATTERN = re.compile(
    rb"(?:/Users/|/home/)[A-Za-z0-9._-]+|[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+"
)
HIGH_CONFIDENCE_SECRET_PATTERNS = {
    "openai_style_key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(
        rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
    ),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
}
INTENTIONAL_PATH_FIXTURES = {
    "tests/test_public_redaction.py",
    "tests/test_real_api_acceptance_script.py",
}


class DeliveryScriptTests(unittest.TestCase):
    def test_public_source_selection_is_versioned_and_excludes_internal_docs(self) -> None:
        version = project_version(ROOT)
        included = {
            path.relative_to(ROOT).as_posix()
            for path in included_source_files(ROOT)
        }

        self.assertIn(
            "integrations/dsh_ecology_plugin/dist/"
            f"ecologyrsi-dsh-evolution-plugin-{version}.tgz",
            included,
        )
        self.assertIn("docs/screenshots/01-run-settings.jpg", included)
        self.assertFalse(
            any(name.startswith("docs/superpowers/") for name in included)
        )
        self.assertNotIn("docs/项目整体Review与方案B优化报告.md", included)

    def test_public_source_selection_rejects_wrong_plugin_archives(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-source-fixture-") as directory:
            fixture = Path(directory)
            self._write_minimal_source_fixture(fixture, version="0.3.33")
            wrong = (
                fixture
                / "integrations/dsh_ecology_plugin/dist/"
                / "ecologyrsi-dsh-evolution-plugin-0.3.32.tgz"
            )
            wrong.write_bytes(b"wrong-version")

            with self.assertRaisesRegex(RuntimeError, "exactly one.*0\\.3\\.33"):
                included_source_files(fixture)

    def test_create_archive_embeds_source_plugin_artifacts_and_build_info(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-archive-fixture-") as directory:
            fixture = Path(directory)
            version = "0.3.33"
            plugin = self._write_minimal_source_fixture(fixture, version=version)
            self._commit_fixture(fixture)
            dist = fixture / "dist"
            dist.mkdir()
            wheel = dist / f"ecologyrsi_dsh-{version}-py3-none-any.whl"
            sdist = dist / f"ecologyrsi_dsh-{version}.tar.gz"
            wheel.write_bytes(b"fake-wheel")
            sdist.write_bytes(b"fake-sdist")

            delivery = create_archive(fixture, dist)

            self.assertTrue((dist / "BUILD-INFO.json").is_file())
            external_build_info = json.loads(
                (dist / "BUILD-INFO.json").read_text(encoding="utf-8")
            )
            self.assertEqual(external_build_info["version"], version)
            self.assertRegex(external_build_info["commit"], r"^[0-9a-f]{40}$")
            self.assertIs(external_build_info["dirty"], False)
            self.assertEqual(external_build_info["source_date_epoch"], 0)
            self.assertEqual(
                set(external_build_info["tools"]), {"python", "node", "npm", "uv"}
            )
            with tarfile.open(delivery, "r:gz") as archive:
                prefix = f"ecologyrsi-dsh-{version}"
                source_plugin = archive.extractfile(
                    f"{prefix}/integrations/dsh_ecology_plugin/dist/{plugin.name}"
                )
                artifact_plugin = archive.extractfile(
                    f"{prefix}/artifacts/{plugin.name}"
                )
                embedded_build_info = archive.extractfile(
                    f"{prefix}/BUILD-INFO.json"
                )
                internal_checksums = archive.extractfile(f"{prefix}/SHA256SUMS")
                self.assertIsNotNone(source_plugin)
                self.assertIsNotNone(artifact_plugin)
                self.assertIsNotNone(embedded_build_info)
                self.assertIsNotNone(internal_checksums)
                self.assertEqual(source_plugin.read(), plugin.read_bytes())
                self.assertEqual(artifact_plugin.read(), plugin.read_bytes())
                self.assertEqual(
                    json.loads(embedded_build_info.read()), external_build_info
                )
                checksum_names = {
                    line.split(None, 1)[1]
                    for line in internal_checksums.read().decode("ascii").splitlines()
                }
                self.assertEqual(
                    checksum_names,
                    {
                        f"artifacts/{wheel.name}",
                        f"artifacts/{sdist.name}",
                        f"artifacts/{plugin.name}",
                        "BUILD-INFO.json",
                    },
                )

    def test_dsh_plugin_builder_stages_root_legal_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-plugin-fixture-") as directory:
            fixture = Path(directory)
            package_root = fixture / "integrations/dsh_ecology_plugin"
            (package_root / "lib").mkdir(parents=True)
            (package_root / "lib/index.js").write_text(
                "export const fixture = true;\n", encoding="utf-8"
            )
            (fixture / "LICENSE").write_bytes(b"fixture license\n")
            (fixture / "NOTICE").write_bytes(b"fixture notice\n")
            (package_root / "package.json").write_text(
                json.dumps(
                    {
                        "name": "@ecologyrsi/dsh-evolution-plugin",
                        "version": "0.3.33",
                        "files": ["lib/**/*.js", "LICENSE", "NOTICE"],
                    }
                ),
                encoding="utf-8",
            )
            output = fixture / "output"
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "python",
                    str(ROOT / "scripts/build_dsh_plugin.py"),
                    "--root",
                    str(fixture),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            archives = list(output.glob("*.tgz"))
            self.assertEqual(len(archives), 1)
            with tarfile.open(archives[0], "r:gz") as archive:
                license_member = archive.extractfile("package/LICENSE")
                notice_member = archive.extractfile("package/NOTICE")
                self.assertIsNotNone(license_member)
                self.assertIsNotNone(notice_member)
                self.assertEqual(license_member.read(), (fixture / "LICENSE").read_bytes())
                self.assertEqual(notice_member.read(), (fixture / "NOTICE").read_bytes())

    def test_npm_verifier_rejects_any_stale_selected_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-plugin-stale-") as directory:
            fixture = Path(directory)
            package_root = fixture / "integrations/dsh_ecology_plugin"
            shutil.copytree(ROOT / "integrations/dsh_ecology_plugin", package_root)
            shutil.copy2(ROOT / "LICENSE", fixture / "LICENSE")
            shutil.copy2(ROOT / "NOTICE", fixture / "NOTICE")
            output = fixture / "output"
            built = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "python",
                    str(ROOT / "scripts/build_dsh_plugin.py"),
                    "--root",
                    str(fixture),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            (package_root / "lib/config.js").write_text(
                "export const staleFixture = true;\n", encoding="utf-8"
            )
            plugin = next(output.glob("*.tgz"))
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT / "scripts")
            verified = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "python",
                    "-c",
                    (
                        "from pathlib import Path; "
                        "from verify_artifacts import verify_npm_plugin; "
                        "verify_npm_plugin(Path(__import__('sys').argv[1]), "
                        "__import__('sys').argv[2], Path(__import__('sys').argv[3]))"
                    ),
                    str(plugin),
                    project_version(ROOT),
                    str(fixture),
                ],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(verified.returncode, 0)
            self.assertIn("package/lib/config.js", verified.stdout + verified.stderr)

    def test_artifacts_only_does_not_run_source_suites(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-artifacts-only-") as directory:
            temporary = Path(directory)
            log = temporary / "python.log"
            fake_python = temporary / "python"
            fake_python.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$*" >> "$ECOLOGYRSI_TEST_COMMAND_LOG"\nexit 0\n',
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            environment = os.environ.copy()
            environment["PYTHON"] = str(fake_python)
            environment["ECOLOGYRSI_TEST_COMMAND_LOG"] = str(log)

            result = subprocess.run(
                ["bash", "scripts/verify_delivery.sh", "--artifacts-only"],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            commands = log.read_text(encoding="utf-8")
            self.assertIn("scripts/verify_artifacts.py dist", commands)
            self.assertNotIn("unittest", commands)
            self.assertNotIn("examples/minimal_run.py", commands)

    def test_source_verification_runs_the_full_node_suite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-source-suite-") as directory:
            temporary = Path(directory)
            log = temporary / "commands.log"
            for name in ("python", "node"):
                command = temporary / name
                command.write_text(
                    '#!/bin/sh\nprintf "%s %s\\n" "$(basename "$0")" "$*" '
                    '>> "$ECOLOGYRSI_TEST_COMMAND_LOG"\nexit 0\n',
                    encoding="utf-8",
                )
                command.chmod(0o755)
            environment = os.environ.copy()
            environment["PYTHON"] = str(temporary / "python")
            environment["PATH"] = f"{temporary}{os.pathsep}{environment['PATH']}"
            environment["ECOLOGYRSI_TEST_COMMAND_LOG"] = str(log)

            result = subprocess.run(
                ["bash", "scripts/verify_delivery.sh", "--source-only"],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            commands = log.read_text(encoding="utf-8")
            self.assertIn("node --test", commands)
            self.assertIn("agent_lifecycle.test.mjs", commands)
            self.assertIn("proxy_security.mjs", commands)

    def test_internal_planning_documents_stay_local_only(self) -> None:
        internal_documents = (
            "EcologyRSI-DSH-完整框架与详细实施方案.md",
            "EcologyRSI-DSH-批次进化闭环实施方案.md",
            "EcologyRSI-DSH-旧API迁移映射与验收附录.md",
            "EcologyRSI-DSH-最终交付审查与运行说明.md",
            "EcologyRSI-DSH-迁移清单模板.yaml",
        )
        included = {
            path.relative_to(ROOT).as_posix()
            for path in included_source_files(ROOT)
        }
        published = [name for name in internal_documents if name in included]
        not_ignored = [
            name
            for name in internal_documents
            if subprocess.run(
                ["git", "check-ignore", "-q", name],
                cwd=ROOT,
                check=False,
            ).returncode
            != 0
        ]
        self.assertEqual(published, [])
        self.assertEqual(not_ignored, [])

    def _release_payloads(self) -> list[tuple[str, bytes]]:
        """Read the exact source set used by the delivery archive.

        This intentionally includes newly created, not-yet-committed files and
        excludes tracked files deleted from the working tree.
        """

        payloads: list[tuple[str, bytes]] = []
        for path in included_source_files(ROOT):
            name = path.relative_to(ROOT).as_posix()
            payloads.append((name, path.read_bytes()))
            if path.suffix == ".tgz":
                with tarfile.open(path, "r:gz") as archive:
                    for member in archive.getmembers():
                        if not member.isfile():
                            continue
                        handle = archive.extractfile(member)
                        if handle is not None:
                            payloads.append((f"{name}:{member.name}", handle.read()))
        return payloads

    def test_release_content_contains_no_private_material(self) -> None:
        configured_email = subprocess.run(
            ["git", "config", "user.email"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).stdout.strip()
        private_home = str(Path.home()).encode("utf-8")
        violations: set[tuple[str, str]] = set()

        for name, payload in self._release_payloads():
            source_name = name.split(":", 1)[0]
            if source_name not in INTENTIONAL_PATH_FIXTURES:
                if private_home in payload or PRIVATE_HOME_PATTERN.search(payload):
                    violations.add((name, "private_home_path"))
            if configured_email and configured_email in payload:
                violations.add((name, "configured_personal_email"))
            for category, pattern in HIGH_CONFIDENCE_SECRET_PATTERNS.items():
                if pattern.search(payload):
                    violations.add((name, category))

        self.assertEqual(sorted(violations), [])

    def test_readme_references_six_repository_screenshots(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        references = re.findall(r"!\[[^]]*\]\((docs/screenshots/[^)]+)\)", readme)
        expected = [
            "docs/screenshots/01-run-settings.jpg",
            "docs/screenshots/02-parameter-design.jpg",
            "docs/screenshots/03-training-data.jpg",
            "docs/screenshots/04-evolution-process.jpg",
            "docs/screenshots/05-candidate-evaluation.jpg",
            "docs/screenshots/06-human-governance.jpg",
        ]
        self.assertEqual(references, expected)
        self.assertTrue(all((ROOT / reference).is_file() for reference in references))

    def test_git_ignores_common_local_secret_files(self) -> None:
        local_secret_paths = (
            ".env",
            ".env.local",
            ".dsh/settings.yaml",
            "runtime.credentials.yaml",
            "deployment.pem",
            "client.key",
            "identity.p12",
        )
        missed = [
            path
            for path in local_secret_paths
            if subprocess.run(
                ["git", "check-ignore", "-q", path],
                cwd=ROOT,
                check=False,
            ).returncode
            != 0
        ]
        self.assertEqual(missed, [])

    def test_default_python_selector_returns_supported_interpreter(self) -> None:
        environment = os.environ.copy()
        environment.pop("PYTHON", None)
        selected = subprocess.run(
            ["bash", "scripts/select_python.sh"],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        version = subprocess.run(
            [selected, "-c", "import sys; print(sys.version_info[:2])"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertIn(version, {"(3, 10)", "(3, 11)", "(3, 12)"})

    @staticmethod
    def _write_minimal_source_fixture(root: Path, *, version: str) -> Path:
        for name in ROOT_FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "pyproject.toml":
                path.write_text(
                    f'[project]\nname = "ecologyrsi-dsh"\nversion = "{version}"\n',
                    encoding="utf-8",
                )
            elif name == ".gitignore":
                path.write_text("dist/\n", encoding="utf-8")
            else:
                path.write_text(f"fixture {name}\n", encoding="utf-8")
        screenshot = root / "docs/screenshots/01-run-settings.jpg"
        screenshot.parent.mkdir(parents=True)
        screenshot.write_bytes(b"fixture screenshot")
        internal = root / "docs/superpowers/plans/private.md"
        internal.parent.mkdir(parents=True)
        internal.write_text("private\n", encoding="utf-8")
        review = root / "docs/项目整体Review与方案B优化报告.md"
        review.write_text("private review\n", encoding="utf-8")
        plugin = (
            root
            / "integrations/dsh_ecology_plugin/dist"
            / f"ecologyrsi-dsh-evolution-plugin-{version}.tgz"
        )
        plugin.parent.mkdir(parents=True)
        plugin.write_bytes(b"fixture plugin")
        return plugin

    @staticmethod
    def _commit_fixture(root: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(
            ["git", "add", "-f", "integrations/dsh_ecology_plugin/dist"],
            cwd=root,
            check=True,
        )
        environment = os.environ.copy()
        environment.update(
            {
                "GIT_AUTHOR_NAME": "Fixture",
                "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
                "GIT_COMMITTER_NAME": "Fixture",
                "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
            }
        )
        subprocess.run(
            ["git", "commit", "-qm", "fixture"], cwd=root, env=environment, check=True
        )


if __name__ == "__main__":
    unittest.main()
