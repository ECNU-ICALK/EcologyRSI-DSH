from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tarfile
import unittest

from scripts.create_delivery_archive import included_source_files


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

    def _tracked_payloads(self) -> list[tuple[str, bytes]]:
        names = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        payloads: list[tuple[str, bytes]] = []
        for raw_name in names:
            if not raw_name:
                continue
            name = raw_name.decode("utf-8")
            path = ROOT / name
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

    def test_tracked_release_content_contains_no_private_material(self) -> None:
        configured_email = subprocess.run(
            ["git", "config", "user.email"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).stdout.strip()
        private_home = str(Path.home()).encode("utf-8")
        violations: set[tuple[str, str]] = set()

        for name, payload in self._tracked_payloads():
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


if __name__ == "__main__":
    unittest.main()
