"""Offline archive-boundary regressions using only public source fixtures."""

from __future__ import annotations

import hashlib
import io
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
with mock.patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    import verify_artifacts as verifier
    from create_delivery_archive import ROOT_FILES, included_source_files, project_version


class DeliveryArtifactGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="ecology-public-artifacts-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.fixture = Path(cls.temporary.name) / "source"
        cls.fixture.mkdir()
        cls.version = project_version(ROOT)
        # Copy only declared public source types, never runtime directories,
        # private settings, original data, or an installed Harness profile.
        patterns = (
            "src/ecologyrsi_dsh/**/*.py", "src/ecologyrsi_dsh/knowledge/*.json",
            "datasets/*.json", "plugins/ecology_evolution/*.js",
            "plugins/ecology_evolution/*.html", "plugins/ecology_evolution/*.css",
            "plugins/ecology_evolution/*.json", "plugins/ecology_evolution/README.md",
            "plugins/ecology_evolution/assets/js/*.js",
            "integrations/dsh_ecology_plugin/lib/**/*.js",
            "integrations/dsh_ecology_plugin/schemas/*.json",
            "integrations/dsh_ecology_plugin/presets/**/*.yml",
            "integrations/dsh_ecology_plugin/presets/**/*.json",
            "integrations/dsh_ecology_plugin/presets/**/*.md",
            "integrations/dsh_ecology_plugin/package.json",
            "integrations/dsh_ecology_plugin/README.md",
            "integrations/dsh_ecology_plugin/test/proxy_security.mjs",
            "scripts/install_dsh_ecology_runtime.mjs",
        )
        sources = {ROOT / relative for relative in ROOT_FILES}
        for pattern in patterns:
            sources.update(ROOT.glob(pattern))
        for source in sources:
            destination = cls.fixture / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        cls.nested_plugin = (
            cls.fixture / "integrations/dsh_ecology_plugin/dist"
            / f"ecologyrsi-dsh-evolution-plugin-{cls.version}.tgz"
        )
        cls.nested_plugin.parent.mkdir()
        cls.nested_plugin.write_bytes(b"public synthetic plugin fixture\n")
        subprocess.run(
            ["git", "-c", "init.defaultBranch=fixture", "init", "-q", str(cls.fixture)],
            check=True, capture_output=True,
        )
        cls.dist_info = f"ecologyrsi_dsh-{cls.version}.dist-info"
        cls.wheel_payloads: dict[str, bytes] = {}
        for source in (cls.fixture / "src/ecologyrsi_dsh").rglob("*"):
            if source.is_file():
                cls.wheel_payloads[source.relative_to(cls.fixture / "src").as_posix()] = source.read_bytes()
        data_root = f"ecologyrsi_dsh-{cls.version}.data/data/share/ecologyrsi-dsh"
        for directory in ("datasets", "plugins", "integrations", "scripts"):
            for source in (cls.fixture / directory).rglob("*"):
                if source.is_file() and "/test/" not in source.as_posix():
                    cls.wheel_payloads[f"{data_root}/{source.relative_to(cls.fixture).as_posix()}"] = source.read_bytes()
        cls.wheel_payloads.update({
            f"{cls.dist_info}/METADATA": (
                f"Metadata-Version: 2.4\nName: ecologyrsi-dsh\nVersion: {cls.version}\n"
                "License-Expression: LicenseRef-Proprietary\n\n"
            ).encode(),
            f"{cls.dist_info}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            f"{cls.dist_info}/RECORD": b"",
            f"{cls.dist_info}/entry_points.txt": b"[console_scripts]\necologyrsi-dsh = ecologyrsi_dsh.application.cli:main\n",
            f"{cls.dist_info}/top_level.txt": b"ecologyrsi_dsh\n",
            f"{cls.dist_info}/licenses/LICENSE": (cls.fixture / "LICENSE").read_bytes(),
            f"{cls.dist_info}/licenses/NOTICE": (cls.fixture / "NOTICE").read_bytes(),
        })

    def _write_wheel(self, path: Path, *, extra=None, omit=()) -> Path:
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in self.wheel_payloads.items():
                if name not in omit:
                    archive.writestr(name, data)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for name, data in extra or []:
                    archive.writestr(name, data)
        return path

    def test_valid_wheel_inventory_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = self._write_wheel(Path(directory) / "valid.whl")
            verifier.verify_wheel(wheel, self.version, self.fixture)

    def test_sensitive_extra_wheel_members_are_rejected(self) -> None:
        for member in (
            "ecologyrsi_dsh/credentials.yaml", "ecologyrsi_dsh/.env",
            "ecologyrsi_dsh/.runtime/session.json",
        ):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                wheel = self._write_wheel(
                    Path(directory) / "sensitive.whl", extra=[(member, b"public fixture")],
                )
                with self.assertRaisesRegex(RuntimeError, "wheel contains sensitive member"):
                    verifier.verify_wheel(wheel, self.version, self.fixture)

    def test_duplicate_wheel_members_are_rejected_before_reading(self) -> None:
        for member in ("ecologyrsi_dsh/__init__.py", f"{self.dist_info}/METADATA"):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                wheel = self._write_wheel(
                    Path(directory) / "duplicate.whl",
                    extra=[(member, self.wheel_payloads[member])],
                )
                with self.assertRaisesRegex(RuntimeError, "wheel contains duplicate member"):
                    verifier.verify_wheel(wheel, self.version, self.fixture)

    def test_unselected_wheel_code_and_metadata_are_rejected(self) -> None:
        for member in (
            "ecologyrsi_dsh/extra.py", f"{self.dist_info}/extra.json",
            "other_distribution-1.0.dist-info/METADATA", "unexpected.pth",
        ):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                wheel = self._write_wheel(
                    Path(directory) / "extra.whl", extra=[(member, b"public fixture")],
                )
                with self.assertRaisesRegex(RuntimeError, "wheel contains unexpected member"):
                    verifier.verify_wheel(wheel, self.version, self.fixture)

    def test_wheel_paths_and_link_types_are_rejected(self) -> None:
        for member in ("../outside.py", "/absolute.py", "ecologyrsi_dsh\\extra.py"):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                wheel = self._write_wheel(
                    Path(directory) / "unsafe.whl", extra=[(member, b"public fixture")],
                )
                with self.assertRaisesRegex(RuntimeError, "wheel contains unsafe member path"):
                    verifier.verify_wheel(wheel, self.version, self.fixture)
        with tempfile.TemporaryDirectory() as directory:
            member = zipfile.ZipInfo("ecologyrsi_dsh/link")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            wheel = self._write_wheel(
                Path(directory) / "linked.whl", extra=[(member, b"public-target")],
            )
            with self.assertRaisesRegex(RuntimeError, "wheel contains unsupported member type"):
                verifier.verify_wheel(wheel, self.version, self.fixture)

    def test_wheel_metadata_inventory_and_legal_source_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = self._write_wheel(
                Path(directory) / "missing.whl", omit=[f"{self.dist_info}/RECORD"],
            )
            with self.assertRaisesRegex(RuntimeError, "wheel is missing expected member"):
                verifier.verify_wheel(wheel, self.version, self.fixture)
            license_member = f"{self.dist_info}/licenses/LICENSE"
            wheel = self._write_wheel(
                Path(directory) / "legal.whl", omit=[license_member],
                extra=[(license_member, b"different legal text")],
            )
            with self.assertRaisesRegex(RuntimeError, "wheel is stale relative to current source"):
                verifier.verify_wheel(wheel, self.version, self.fixture)

    def test_wheel_console_scripts_cannot_add_or_redirect_commands(self) -> None:
        member = f"{self.dist_info}/entry_points.txt"
        normal = self.wheel_payloads[member]
        cases = (
            normal + b"unexpected_cli = ecologyrsi_dsh.application.cli:main\n",
            normal.replace(b"cli:main", b"cli:other"),
            normal + b"\n[other_plugins]\nextra = ecologyrsi_dsh.application.cli:main\n",
            b"[DEFAULT]\nextra = ecologyrsi_dsh.application.cli:main\n" + normal,
        )
        for index, payload in enumerate(cases):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as directory:
                wheel = self._write_wheel(
                    Path(directory) / "entry-points.whl", omit=[member],
                    extra=[(member, payload)],
                )
                with self.assertRaisesRegex(RuntimeError, "wheel (console scripts|entry point groups) differ"):
                    verifier.verify_wheel(wheel, self.version, self.fixture)

    def test_console_script_contract_comes_from_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            metadata = source / "pyproject.toml"
            metadata.write_text(
                "[project.scripts]\ncustom-command = 'example.module:main' # declared\n\n[other]\nvalue = 1\n"
            )
            verifier._verify_wheel_entry_points(
                b"[console_scripts]\ncustom-command = example.module:main\n", source,
            )
            with self.assertRaisesRegex(RuntimeError, "console scripts differ"):
                verifier._verify_wheel_entry_points(
                    b"[console_scripts]\ncustom-command = example.module:other\n", source,
                )
            for table in (
                "[project.scripts]\ncommand = ['not-a-string']\n",
                "[project.scripts]\ncommand = 'a:main'\ncommand = 'b:main'\n",
                "[project.scripts]\ncommand = 'a:main'\n[project.scripts]\nother = 'b:main'\n",
            ):
                metadata.write_text(table)
                with self.assertRaises(RuntimeError):
                    verifier._project_console_scripts(source)

    def test_delivery_source_selects_lint_config_and_linked_reasoning_doc(self) -> None:
        selected = {p.relative_to(self.fixture).as_posix() for p in included_source_files(self.fixture)}
        resources = {"ruff.toml", "docs/reasoning-configuration-2026-10-20.md"}
        self.assertTrue(resources <= selected)
        explicit_manifest = {
            name for line in (ROOT / "MANIFEST.in").read_text().splitlines()
            if line.startswith("include ") for name in line.split()[1:]
        }
        self.assertTrue(resources <= explicit_manifest)

    def _delivery_fixture(self, directory: Path, *, replace_index=None):
        wheel = self._write_wheel(directory / f"ecologyrsi_dsh-{self.version}-py3-none-any.whl")
        sdist = directory / f"ecologyrsi_dsh-{self.version}.tar.gz"
        sdist.write_bytes(b"public synthetic sdist fixture\n")
        plugin = directory / self.nested_plugin.name
        shutil.copy2(self.nested_plugin, plugin)
        build_info = directory / "BUILD-INFO.json"
        build_info.write_bytes(b"{}\n")
        delivery = directory / f"ecologyrsi-dsh-{self.version}-delivery.tar.gz"
        prefix = f"ecologyrsi-dsh-{self.version}"
        members = {
            p.relative_to(self.fixture).as_posix(): p.read_bytes()
            for p in included_source_files(self.fixture)
        }
        for index, artifact in enumerate((wheel, sdist, plugin)):
            members[f"artifacts/{artifact.name}"] = (
                b"different embedded artifact with recomputed internal checksum\n"
                if index == replace_index else artifact.read_bytes()
            )
        members["BUILD-INFO.json"] = build_info.read_bytes()
        members["SHA256SUMS"] = "".join(
            f"{hashlib.sha256(data).hexdigest()}  {name}\n"
            for name, data in members.items()
            if name.startswith("artifacts/") or name == "BUILD-INFO.json"
        ).encode()
        with tarfile.open(delivery, "w:gz") as archive:
            for name, data in members.items():
                member = tarfile.TarInfo(f"{prefix}/{name}")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        return delivery, wheel, sdist, plugin, build_info, self.version, self.fixture

    def test_delivery_embedded_artifacts_match_external_copies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verifier.verify_delivery_archive(*self._delivery_fixture(Path(directory)))

    def test_recomputed_internal_checksum_cannot_hide_replaced_artifacts(self) -> None:
        for index, artifact_type in enumerate(("wheel", "sdist", "plugin")):
            with self.subTest(artifact=artifact_type), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, "delivery artifact differs from external copy"):
                    verifier.verify_delivery_archive(
                        *self._delivery_fixture(Path(directory), replace_index=index)
                    )


if __name__ == "__main__":
    unittest.main()
