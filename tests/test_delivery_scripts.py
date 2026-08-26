from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
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
    def test_non_git_public_source_selection_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-non-git-source-") as directory:
            fixture = Path(directory)
            version = "0.3.33"
            self._write_minimal_source_fixture(fixture, version=version)
            runtime_auth = (
                fixture / "integrations/dsh_ecology_plugin/.runtime/auth.json"
            )
            runtime_auth.parent.mkdir(parents=True)
            runtime_auth.write_text('{"token": "cornflower"}\n', encoding="utf-8")
            dist = fixture / "dist"
            dist.mkdir()
            (dist / f"ecologyrsi_dsh-{version}-py3-none-any.whl").write_bytes(
                b"fake-wheel"
            )
            (dist / f"ecologyrsi_dsh-{version}.tar.gz").write_bytes(b"fake-sdist")

            with self.assertRaisesRegex(RuntimeError, "Git repository"):
                included_source_files(fixture)
            with self.assertRaisesRegex(RuntimeError, "Git repository"):
                create_archive(fixture, dist)
            self.assertFalse((dist / "BUILD-INFO.json").exists())

    def test_sensitive_runtime_directories_cannot_enter_delivery(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-sensitive-delivery-") as directory:
            fixture = Path(directory)
            version = "0.3.33"
            self._write_minimal_source_fixture(fixture, version=version)
            sensitive = {
                "integrations/dsh_ecology_plugin/.runtime/auth.json": (
                    '{"token": "cornflower"}\n'
                ),
                "integrations/dsh_ecology_plugin/.dsh/settings.yml": (
                    "password: meadow\n"
                ),
                "integrations/dsh_ecology_plugin/.ssh/config.json": (
                    '{"host": "example.invalid"}\n'
                ),
            }
            for relative, content in sensitive.items():
                path = fixture / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            self._commit_fixture(fixture)
            dist = fixture / "dist"
            dist.mkdir()
            (dist / f"ecologyrsi_dsh-{version}-py3-none-any.whl").write_bytes(
                b"fake-wheel"
            )
            (dist / f"ecologyrsi_dsh-{version}.tar.gz").write_bytes(b"fake-sdist")

            included = {
                path.relative_to(fixture).as_posix()
                for path in included_source_files(fixture)
            }
            delivery = create_archive(fixture, dist)

            self.assertTrue(set(sensitive).isdisjoint(included))
            build_info = json.loads(
                (dist / "BUILD-INFO.json").read_text(encoding="utf-8")
            )
            self.assertIs(build_info["dirty"], False)
            with tarfile.open(delivery, "r:gz") as archive:
                names = set(archive.getnames())
            self.assertFalse(
                any(any(f"/{part}/" in f"/{name}/" for part in (".runtime", ".dsh", ".ssh")) for name in names)
            )

    def test_sensitive_runtime_directories_cannot_enter_sdist(self) -> None:
        sensitive = {
            "integrations/dsh_ecology_plugin/.runtime/auth.json": (
                '{"token": "cornflower"}\n'
            ),
            "integrations/dsh_ecology_plugin/.dsh/settings.yml": "password: meadow\n",
            "integrations/dsh_ecology_plugin/.ssh/config.json": (
                '{"host": "example.invalid"}\n'
            ),
        }
        try:
            for relative, content in sensitive.items():
                path = ROOT / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            with tempfile.TemporaryDirectory(
                prefix="ecologyrsi-sensitive-sdist-"
            ) as directory:
                result = subprocess.run(
                    ["uv", "build", "--sdist", "--out-dir", directory, str(ROOT)],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                sdist = next(Path(directory).glob("*.tar.gz"))
                with tarfile.open(sdist, "r:gz") as archive:
                    names = set(archive.getnames())
                self.assertFalse(any(any(name.endswith(f"/{relative}") for name in names) for relative in sensitive))
                verified = self._verify_sdist(sdist)
                self.assertEqual(
                    verified.returncode, 0, verified.stdout + verified.stderr
                )
        finally:
            for relative in sensitive:
                path = ROOT / relative
                path.unlink(missing_ok=True)
                path.parent.rmdir()

    def test_sdist_verifier_rejects_unexpected_sensitive_member(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-sdist-leak-") as directory:
            temporary = Path(directory)
            built = subprocess.run(
                ["uv", "build", "--sdist", "--out-dir", str(temporary), str(ROOT)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            sdist = next(temporary.glob("*.tar.gz"))
            leaked = temporary / "leaked.tar.gz"
            self._copy_tar_with_extra_file(
                sdist,
                leaked,
                f"ecologyrsi_dsh-{project_version(ROOT)}/"
                "integrations/dsh_ecology_plugin/.runtime/auth.json",
                b'{"token": "cornflower"}\n',
            )

            verified = self._verify_sdist(leaked)

            self.assertNotEqual(verified.returncode, 0)
            self.assertIn("sensitive member", verified.stdout + verified.stderr)

    def test_sdist_verifier_rejects_unexpected_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-sdist-directory-") as directory:
            temporary = Path(directory)
            built = subprocess.run(
                ["uv", "build", "--sdist", "--out-dir", str(temporary), str(ROOT)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            sdist = next(temporary.glob("*.tar.gz"))
            modified = temporary / "unexpected-directory.tar.gz"
            self._copy_tar_with_extra_directory(
                sdist,
                modified,
                f"ecologyrsi_dsh-{project_version(ROOT)}/"
                "unexpected-empty-directory",
            )

            verified = self._verify_sdist(modified)

            self.assertNotEqual(verified.returncode, 0)
            self.assertIn("unexpected member", verified.stdout + verified.stderr)

    def test_sdist_verifier_rejects_file_at_expected_directory_name(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-sdist-type-") as directory:
            temporary = Path(directory)
            built = subprocess.run(
                ["uv", "build", "--sdist", "--out-dir", str(temporary), str(ROOT)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            sdist = next(temporary.glob("*.tar.gz"))
            modified = temporary / "directory-as-file.tar.gz"
            self._copy_tar_with_extra_file(
                sdist,
                modified,
                f"ecologyrsi_dsh-{project_version(ROOT)}/src",
                b"not a directory\n",
            )

            verified = self._verify_sdist(modified)

            self.assertNotEqual(verified.returncode, 0)
            self.assertIn("unexpected member type", verified.stdout + verified.stderr)

    def test_plugin_builder_cli_rejects_symlinked_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-plugin-root-link-") as directory:
            temporary = Path(directory)
            root = temporary / "source"
            self._write_plugin_build_fixture(root)
            linked_root = temporary / "linked-source"
            linked_root.symlink_to(root, target_is_directory=True)
            output = temporary / "output"

            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "python",
                    str(ROOT / "scripts/build_dsh_plugin.py"),
                    "--root",
                    str(linked_root),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("build root is a symlink", result.stdout + result.stderr)
            self.assertFalse(output.exists())

    def test_plugin_builder_cli_rejects_symlinked_output_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-plugin-output-link-") as directory:
            temporary = Path(directory)
            root = temporary / "source"
            self._write_plugin_build_fixture(root)
            real_output = temporary / "real-output"
            real_output.mkdir()
            linked_output = temporary / "linked-output"
            linked_output.symlink_to(real_output, target_is_directory=True)

            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "python",
                    str(ROOT / "scripts/build_dsh_plugin.py"),
                    "--root",
                    str(root),
                    "--output-dir",
                    str(linked_output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "plugin output directory is a symlink",
                result.stdout + result.stderr,
            )
            self.assertEqual(list(real_output.iterdir()), [])

    def test_plugin_builder_cli_rejects_parent_traversal_in_root(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ecologyrsi-plugin-root-parent-"
        ) as directory:
            temporary = Path(directory)
            pivot = temporary / "pivot"
            (pivot / "child").mkdir(parents=True)
            actual_root = pivot / "source"
            self._write_plugin_build_fixture(actual_root)
            (temporary / "source").mkdir()
            link = temporary / "link"
            link.symlink_to(pivot / "child", target_is_directory=True)
            traversal_root = link / ".." / "source"
            output = temporary / "output"

            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "python",
                    str(ROOT / "scripts/build_dsh_plugin.py"),
                    "--root",
                    str(traversal_root),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "build root contains parent traversal",
                result.stdout + result.stderr,
            )
            self.assertFalse(output.exists())

    def test_plugin_builder_cli_rejects_parent_traversal_in_output(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ecologyrsi-plugin-output-parent-"
        ) as directory:
            temporary = Path(directory)
            root = temporary / "source"
            self._write_plugin_build_fixture(root)
            pivot = temporary / "pivot"
            (pivot / "child").mkdir(parents=True)
            link = temporary / "link"
            link.symlink_to(pivot / "child", target_is_directory=True)
            traversal_output = link / ".." / "output"

            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "python",
                    str(ROOT / "scripts/build_dsh_plugin.py"),
                    "--root",
                    str(root),
                    "--output-dir",
                    str(traversal_output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "plugin output directory contains parent traversal",
                result.stdout + result.stderr,
            )
            self.assertFalse((temporary / "output").exists())
            self.assertFalse((pivot / "output").exists())

    def test_delivery_archive_cli_rejects_symlinked_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-delivery-root-link-") as directory:
            temporary = Path(directory)
            root = temporary / "source"
            version = "0.3.33"
            self._write_minimal_source_fixture(root, version=version)
            self._commit_fixture(root)
            linked_root = temporary / "linked-source"
            linked_root.symlink_to(root, target_is_directory=True)
            dist = temporary / "dist"
            self._write_fake_python_artifacts(dist, version)

            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "python",
                    str(ROOT / "scripts/create_delivery_archive.py"),
                    "--root",
                    str(linked_root),
                    "--dist",
                    str(dist),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("delivery root is a symlink", result.stdout + result.stderr)
            self.assertFalse((dist / "BUILD-INFO.json").exists())

    def test_delivery_archive_cli_rejects_symlinked_dist(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-delivery-dist-link-") as directory:
            temporary = Path(directory)
            root = temporary / "source"
            version = "0.3.33"
            self._write_minimal_source_fixture(root, version=version)
            self._commit_fixture(root)
            real_dist = temporary / "real-dist"
            self._write_fake_python_artifacts(real_dist, version)
            linked_dist = temporary / "linked-dist"
            linked_dist.symlink_to(real_dist, target_is_directory=True)

            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "python",
                    str(ROOT / "scripts/create_delivery_archive.py"),
                    "--root",
                    str(root),
                    "--dist",
                    str(linked_dist),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "delivery output directory is a symlink",
                result.stdout + result.stderr,
            )
            self.assertFalse((real_dist / "BUILD-INFO.json").exists())

    def test_delivery_archive_cli_rejects_parent_traversal_in_root(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ecologyrsi-delivery-root-parent-"
        ) as directory:
            temporary = Path(directory)
            version = "0.3.33"
            pivot = temporary / "pivot"
            (pivot / "child").mkdir(parents=True)
            actual_root = pivot / "source"
            self._write_minimal_source_fixture(actual_root, version=version)
            self._commit_fixture(actual_root)
            (temporary / "source").mkdir()
            link = temporary / "link"
            link.symlink_to(pivot / "child", target_is_directory=True)
            traversal_root = link / ".." / "source"
            dist = temporary / "dist"
            self._write_fake_python_artifacts(dist, version)

            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "python",
                    str(ROOT / "scripts/create_delivery_archive.py"),
                    "--root",
                    str(traversal_root),
                    "--dist",
                    str(dist),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "delivery root contains parent traversal",
                result.stdout + result.stderr,
            )
            self.assertFalse((dist / "BUILD-INFO.json").exists())

    def test_delivery_archive_cli_rejects_parent_traversal_in_dist(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ecologyrsi-delivery-dist-parent-"
        ) as directory:
            temporary = Path(directory)
            version = "0.3.33"
            root = temporary / "source"
            self._write_minimal_source_fixture(root, version=version)
            self._commit_fixture(root)
            pivot = temporary / "pivot"
            (pivot / "child").mkdir(parents=True)
            actual_dist = pivot / "dist"
            actual_dist.mkdir()
            (temporary / "dist").mkdir()
            link = temporary / "link"
            link.symlink_to(pivot / "child", target_is_directory=True)
            traversal_dist = link / ".." / "dist"

            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "python",
                    str(ROOT / "scripts/create_delivery_archive.py"),
                    "--root",
                    str(root),
                    "--dist",
                    str(traversal_dist),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "delivery output directory contains parent traversal",
                result.stdout + result.stderr,
            )
            self.assertEqual(list(actual_dist.iterdir()), [])
            self.assertEqual(list((temporary / "dist").iterdir()), [])

    def test_delivery_archive_cli_accepts_lexical_temp_paths(self) -> None:
        parent = "/var/tmp" if sys.platform == "darwin" else None
        with tempfile.TemporaryDirectory(
            prefix="ecologyrsi-delivery-cli-", dir=parent
        ) as directory:
            temporary = Path(directory)
            if sys.platform == "darwin":
                self.assertEqual(temporary.parts[1], "var")
                self.assertTrue(Path("/var").samefile("/private/var"))
            root = temporary / "source"
            version = "0.3.33"
            self._write_minimal_source_fixture(root, version=version)
            self._commit_fixture(root)
            dist = temporary / "dist"
            self._write_fake_python_artifacts(dist, version)

            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "python",
                    str(ROOT / "scripts/create_delivery_archive.py"),
                    "--root",
                    str(root),
                    "--dist",
                    str(dist),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(
                (dist / f"ecologyrsi-dsh-{version}-delivery.tar.gz").is_file()
            )

    def test_project_version_rejects_symlink_before_reading(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-version-symlink-") as directory:
            fixture = Path(directory)
            external = fixture / "external.toml"
            external.write_text(
                '[project]\nname = "ecologyrsi-dsh"\nversion = "0.3.33"\n',
                encoding="utf-8",
            )
            (fixture / "pyproject.toml").symlink_to(external)

            with self.assertRaisesRegex(RuntimeError, "symlink.*pyproject.toml"):
                project_version(fixture)

    def test_artifact_verifier_rejects_distributed_plugin_symlink_first(self) -> None:
        version = project_version(ROOT)
        source_plugin = next(
            (ROOT / "integrations/dsh_ecology_plugin/dist").glob("*.tgz")
        )
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-plugin-artifact-link-") as directory:
            dist = Path(directory)
            (dist / f"ecologyrsi_dsh-{version}-py3-none-any.whl").write_bytes(
                b"not-a-wheel"
            )
            (dist / f"ecologyrsi_dsh-{version}.tar.gz").write_bytes(b"not-an-sdist")
            (dist / f"ecologyrsi-dsh-{version}-delivery.tar.gz").write_bytes(
                b"not-a-delivery"
            )
            (dist / "BUILD-INFO.json").write_text("{}\n", encoding="utf-8")
            (dist / source_plugin.name).symlink_to(source_plugin)

            verified = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "python",
                    str(ROOT / "scripts/verify_artifacts.py"),
                    str(dist),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(verified.returncode, 0)
            self.assertIn(
                "npm plugin archive is a symlink",
                verified.stdout + verified.stderr,
            )

    def test_sdist_verifier_rejects_link_member_before_extracting(self) -> None:
        version = project_version(ROOT)
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-sdist-link-") as directory:
            sdist = Path(directory) / f"ecologyrsi_dsh-{version}.tar.gz"
            with tarfile.open(sdist, "w:gz") as archive:
                member = tarfile.TarInfo(f"ecologyrsi_dsh-{version}/README.md")
                member.type = tarfile.SYMTYPE
                member.linkname = "../../outside"
                archive.addfile(member)

            verified = self._verify_sdist(sdist)

            self.assertNotEqual(verified.returncode, 0)
            self.assertIn("sdist contains linked member", verified.stdout + verified.stderr)

    def test_delivery_verifier_rejects_link_member_before_extracting(self) -> None:
        version = project_version(ROOT)
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-delivery-link-") as directory:
            temporary = Path(directory)
            delivery = temporary / f"ecologyrsi-dsh-{version}-delivery.tar.gz"
            with tarfile.open(delivery, "w:gz") as archive:
                member = tarfile.TarInfo(f"ecologyrsi-dsh-{version}/README.md")
                member.type = tarfile.LNKTYPE
                member.linkname = f"ecologyrsi-dsh-{version}/outside"
                archive.addfile(member)
            placeholders = [
                temporary / f"ecologyrsi_dsh-{version}-py3-none-any.whl",
                temporary / f"ecologyrsi_dsh-{version}.tar.gz",
                temporary / f"ecologyrsi-dsh-evolution-plugin-{version}.tgz",
                temporary / "BUILD-INFO.json",
            ]
            for path in placeholders:
                path.write_bytes(b"placeholder")
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
                        "from pathlib import Path; from verify_artifacts import "
                        "verify_delivery_archive; import sys; "
                        "verify_delivery_archive(*map(Path, sys.argv[1:6]), "
                        "sys.argv[6], Path(sys.argv[7]))"
                    ),
                    str(delivery),
                    *(str(path) for path in placeholders),
                    version,
                    str(ROOT),
                ],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(verified.returncode, 0)
            self.assertIn(
                "delivery archive contains linked member",
                verified.stdout + verified.stderr,
            )

    def test_ignored_credentials_cannot_enter_clean_delivery(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-credential-fixture-") as directory:
            fixture = Path(directory)
            version = "0.3.33"
            self._write_minimal_source_fixture(fixture, version=version)
            credential = (
                fixture
                / "integrations/dsh_ecology_plugin/release.credentials.yml"
            )
            credential.write_text(
                "token: cornflower\npassword: field-not-secret\n",
                encoding="utf-8",
            )
            self._commit_fixture(fixture)
            self.assertEqual(
                subprocess.run(
                    ["git", "check-ignore", "-q", str(credential.relative_to(fixture))],
                    cwd=fixture,
                    check=False,
                ).returncode,
                0,
            )
            dist = fixture / "dist"
            dist.mkdir()
            (dist / f"ecologyrsi_dsh-{version}-py3-none-any.whl").write_bytes(
                b"fake-wheel"
            )
            (dist / f"ecologyrsi_dsh-{version}.tar.gz").write_bytes(b"fake-sdist")

            included = {
                path.relative_to(fixture).as_posix()
                for path in included_source_files(fixture)
            }
            delivery = create_archive(fixture, dist)

            self.assertNotIn(
                "integrations/dsh_ecology_plugin/release.credentials.yml", included
            )
            build_info = json.loads(
                (dist / "BUILD-INFO.json").read_text(encoding="utf-8")
            )
            self.assertIs(build_info["dirty"], False)
            with tarfile.open(delivery, "r:gz") as archive:
                self.assertFalse(
                    any(
                        name.endswith("/release.credentials.yml")
                        for name in archive.getnames()
                    )
                )

    def test_ignored_credentials_cannot_enter_sdist(self) -> None:
        credential = (
            ROOT
            / "integrations/dsh_ecology_plugin/review.credentials.yml"
        )
        credential.write_text(
            "token: cornflower\npassword: field-not-secret\n", encoding="utf-8"
        )
        try:
            self.assertEqual(
                subprocess.run(
                    ["git", "check-ignore", "-q", str(credential.relative_to(ROOT))],
                    cwd=ROOT,
                    check=False,
                ).returncode,
                0,
            )
            with tempfile.TemporaryDirectory(
                prefix="ecologyrsi-credential-sdist-"
            ) as directory:
                result = subprocess.run(
                    [
                        "uv",
                        "build",
                        "--sdist",
                        "--out-dir",
                        directory,
                        str(ROOT),
                    ],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                sdist = next(Path(directory).glob("*.tar.gz"))
                with tarfile.open(sdist, "r:gz") as archive:
                    self.assertFalse(
                        any(
                            name.endswith("/review.credentials.yml")
                            for name in archive.getnames()
                        )
                    )
        finally:
            credential.unlink(missing_ok=True)

    def test_source_selection_rejects_explicit_root_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-root-symlink-") as directory:
            fixture = Path(directory)
            self._write_minimal_source_fixture(fixture, version="0.3.33")
            external = fixture / "external-license"
            external.write_text("must not be read\n", encoding="utf-8")
            license_path = fixture / "LICENSE"
            license_path.unlink()
            license_path.symlink_to(external)

            with self.assertRaisesRegex(RuntimeError, "symlink.*LICENSE"):
                included_source_files(fixture)

    def test_source_selection_rejects_nested_plugin_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-plugin-symlink-") as directory:
            fixture = Path(directory)
            plugin = self._write_minimal_source_fixture(fixture, version="0.3.33")
            external = fixture / "external-plugin.tgz"
            external.write_bytes(b"must not be read")
            plugin.unlink()
            plugin.symlink_to(external)

            with self.assertRaisesRegex(RuntimeError, "symlink.*plugin"):
                included_source_files(fixture)

    def test_source_selection_rejects_selected_directory_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-directory-symlink-") as directory:
            fixture = Path(directory)
            self._write_minimal_source_fixture(fixture, version="0.3.33")
            screenshots = fixture / "docs/screenshots"
            shutil.rmtree(screenshots)
            external = fixture / "external-screenshots"
            external.mkdir()
            (external / "secret.jpg").write_bytes(b"must not be read")
            screenshots.symlink_to(external, target_is_directory=True)

            with self.assertRaisesRegex(RuntimeError, "symlink.*docs/screenshots"):
                included_source_files(fixture)

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
        parent = "/var/tmp" if sys.platform == "darwin" else None
        with tempfile.TemporaryDirectory(
            prefix="ecologyrsi-plugin-fixture-", dir=parent
        ) as directory:
            fixture = Path(directory)
            if sys.platform == "darwin":
                self.assertEqual(fixture.parts[1], "var")
                self.assertTrue(Path("/var").samefile("/private/var"))
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

    def test_dsh_plugin_builder_rejects_selected_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-npm-symlink-") as directory:
            fixture = Path(directory)
            package_root = fixture / "integrations/dsh_ecology_plugin"
            (package_root / "lib").mkdir(parents=True)
            external = fixture / "external.js"
            external.write_text("export const external = true;\n", encoding="utf-8")
            (package_root / "lib/external.js").symlink_to(external)
            (fixture / "LICENSE").write_text("license\n", encoding="utf-8")
            (fixture / "NOTICE").write_text("notice\n", encoding="utf-8")
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
                    str(fixture / "output"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("symlink", (result.stdout + result.stderr).casefold())

    def test_source_gate_rejects_stale_same_version_plugin(self) -> None:
        plugin = next(
            (ROOT / "integrations/dsh_ecology_plugin/dist").glob(
                "ecologyrsi-dsh-evolution-plugin-*.tgz"
            )
        )
        original_plugin = plugin.read_bytes()
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-source-stale-") as directory:
            temporary = Path(directory)
            fixture = temporary / "fixture"
            package_root = fixture / "integrations/dsh_ecology_plugin"
            shutil.copytree(ROOT / "integrations/dsh_ecology_plugin", package_root)
            shutil.copy2(ROOT / "LICENSE", fixture / "LICENSE")
            shutil.copy2(ROOT / "NOTICE", fixture / "NOTICE")
            (package_root / "lib/config.js").write_text(
                "export const staleFixture = true;\n", encoding="utf-8"
            )
            output = temporary / "output"
            subprocess.run(
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
                check=True,
                capture_output=True,
                text=True,
            )
            stale_plugin = next(output.glob("*.tgz"))
            fake_python = temporary / "python"
            fake_python.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *verify_npm_plugin*) exec \"$ECOLOGYRSI_REAL_PYTHON\" \"$@\" ;;\n"
                "esac\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            fake_node = temporary / "node"
            fake_node.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake_node.chmod(0o755)
            environment = os.environ.copy()
            environment["PYTHON"] = str(fake_python)
            environment["ECOLOGYRSI_REAL_PYTHON"] = subprocess.check_output(
                ["bash", "scripts/select_python.sh"], cwd=ROOT, text=True
            ).strip()
            environment["PATH"] = f"{temporary}{os.pathsep}{environment['PATH']}"
            try:
                plugin.write_bytes(stale_plugin.read_bytes())
                result = subprocess.run(
                    ["bash", "scripts/verify_delivery.sh", "--source-only"],
                    cwd=ROOT,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            finally:
                plugin.write_bytes(original_plugin)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("npm plugin is stale: package/lib/config.js", result.stderr)

    def test_dirty_build_override_is_not_advertised_or_honored(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-dirty-release-") as directory:
            temporary = Path(directory)
            for name in ("python", "uv", "node", "find", "npm"):
                command = temporary / name
                command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                command.chmod(0o755)
            fake_git = temporary / "git"
            fake_git.write_text(
                "#!/bin/sh\nprintf ' M fixture\\n'\n", encoding="utf-8"
            )
            fake_git.chmod(0o755)
            environment = os.environ.copy()
            environment["PYTHON"] = str(temporary / "python")
            environment["PATH"] = f"{temporary}{os.pathsep}{environment['PATH']}"
            environment["ECOLOGYRSI_ALLOW_DIRTY_BUILD"] = "1"

            result = subprocess.run(
                ["bash", "scripts/build_delivery.sh"],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("clean worktree", result.stderr)
            self.assertNotIn("ECOLOGYRSI_ALLOW_DIRTY_BUILD", result.stderr)

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

    def test_npm_verifier_rejects_legal_symlink_before_reading(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-legal-symlink-") as directory:
            fixture = Path(directory)
            shutil.copytree(
                ROOT / "integrations/dsh_ecology_plugin",
                fixture / "integrations/dsh_ecology_plugin",
            )
            (fixture / "LICENSE").symlink_to(fixture / "missing-external-license")
            shutil.copy2(ROOT / "NOTICE", fixture / "NOTICE")
            plugin = next(
                (ROOT / "integrations/dsh_ecology_plugin/dist").glob("*.tgz")
            )
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
            self.assertIn("legal source is a symlink", verified.stdout + verified.stderr)

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

    def test_source_verification_rejects_extra_cli_arguments(self) -> None:
        result = subprocess.run(
            [
                "bash",
                "scripts/verify_delivery.sh",
                "--artifacts-only",
                "unexpected",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("usage:", result.stderr)

    def test_source_verification_propagates_node_syntax_failures(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ecologyrsi-node-check-") as directory:
            temporary = Path(directory)
            fake_python = temporary / "python"
            fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake_python.chmod(0o755)
            fake_node = temporary / "node"
            fake_node.write_text(
                "#!/bin/sh\n"
                "if [ \"${1:-}\" = \"--check\" ]; then exit 19; fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_node.chmod(0o755)
            environment = os.environ.copy()
            environment["PYTHON"] = str(fake_python)
            environment["PATH"] = f"{temporary}{os.pathsep}{environment['PATH']}"

            result = subprocess.run(
                ["bash", "scripts/verify_delivery.sh", "--source-only"],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 19, result.stdout + result.stderr)

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
    def _copy_tar_with_extra_file(
        source: Path, destination: Path, name: str, data: bytes
    ) -> None:
        with tarfile.open(source, "r:gz") as incoming:
            with tarfile.open(destination, "w:gz") as outgoing:
                for member in incoming.getmembers():
                    handle = incoming.extractfile(member) if member.isfile() else None
                    outgoing.addfile(member, handle)
                extra = tarfile.TarInfo(name)
                extra.size = len(data)
                outgoing.addfile(extra, io.BytesIO(data))

    @staticmethod
    def _copy_tar_with_extra_directory(
        source: Path, destination: Path, name: str
    ) -> None:
        with tarfile.open(source, "r:gz") as incoming:
            with tarfile.open(destination, "w:gz") as outgoing:
                for member in incoming.getmembers():
                    handle = incoming.extractfile(member) if member.isfile() else None
                    outgoing.addfile(member, handle)
                extra = tarfile.TarInfo(name)
                extra.type = tarfile.DIRTYPE
                outgoing.addfile(extra)

    @staticmethod
    def _verify_sdist(sdist: Path) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "scripts")
        return subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "python",
                "-c",
                (
                    "from pathlib import Path; from verify_artifacts import "
                    "verify_sdist; import sys; verify_sdist(Path(sys.argv[1]), "
                    "Path(sys.argv[2]), sys.argv[3])"
                ),
                str(sdist),
                str(ROOT),
                project_version(ROOT),
            ],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

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
                path.write_text("dist/\n*credentials*.yml\n", encoding="utf-8")
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
    def _write_plugin_build_fixture(root: Path) -> None:
        package_root = root / "integrations/dsh_ecology_plugin"
        (package_root / "lib").mkdir(parents=True)
        (package_root / "lib/index.js").write_text(
            "export const fixture = true;\n", encoding="utf-8"
        )
        (root / "LICENSE").write_text("fixture license\n", encoding="utf-8")
        (root / "NOTICE").write_text("fixture notice\n", encoding="utf-8")
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

    @staticmethod
    def _write_fake_python_artifacts(dist: Path, version: str) -> None:
        dist.mkdir()
        (dist / f"ecologyrsi_dsh-{version}-py3-none-any.whl").write_bytes(
            b"fake-wheel"
        )
        (dist / f"ecologyrsi_dsh-{version}.tar.gz").write_bytes(b"fake-sdist")

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
