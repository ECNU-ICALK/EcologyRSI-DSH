from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import real_api_agent_tool_acceptance as acceptance


def _reference_binding() -> dict[str, object]:
    return {
        "run_id": "run:reference",
        "event_seq": 7,
        "created_at": "2026-08-19T00:00:00+00:00",
        "source": "RunCreated.task_manifest.metadata",
        "planner": {
            "model_id": "pjlab/glm-5.2",
            "configuration_digest": "a" * 64,
        },
        "critic": {
            "model_id": "pjlab/deepseek-v4-flash-0731",
            "configuration_digest": "b" * 64,
        },
    }


class RealApiAcceptanceScriptTests(unittest.TestCase):
    def test_gateway_args_never_auto_allow_an_insecure_provider(self) -> None:
        args = SimpleNamespace(
            planner="newapi/glm-5.2",
            timeout_seconds=900.0,
            gateway_attempts=4,
        )
        sentinel = object()
        with (
            patch.dict(acceptance.os.environ, {}, clear=True),
            patch.object(acceptance.ModelGateway, "from_env", return_value=sentinel) as factory,
        ):
            self.assertIs(acceptance._gateway_from_args(args), sentinel)  # noqa: SLF001

        configured_env = factory.call_args.args[0]
        self.assertNotIn(
            "ECOLOGYRSI_DSH_ALLOW_INSECURE_HTTP_PROVIDERS", configured_env
        )
        self.assertEqual(configured_env["ECOLOGYRSI_DSH_MODEL_TIMEOUT"], "900.0")
        self.assertEqual(configured_env["ECOLOGYRSI_DSH_MODEL_MAX_ATTEMPTS"], "4")

    def test_main_atomically_persists_failed_acceptance_report(self) -> None:
        report = {
            "schema_version": "ecologyrsi-dsh.real-api-agent-tool-acceptance/1",
            "passed": False,
            "failed_checks": [{"code": "planner_evidence_missing"}],
        }
        with TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "acceptance.json"
            binding = {"package_version": "0.2.1"}
            with (
                patch.object(
                    acceptance,
                    "_reference_binding_from_args",
                    return_value=_reference_binding(),
                ),
                patch.object(acceptance, "_release_binding", return_value=binding),
                patch.object(acceptance, "run_acceptance", return_value=report),
            ):
                with redirect_stdout(io.StringIO()):
                    status = acceptance.main(
                        ["--db", "unused.sqlite3", "--output", str(output)]
                    )

            self.assertEqual(status, 1)
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(persisted["passed"])
            self.assertEqual(persisted["release_binding"], binding)
            self.assertEqual(persisted["report_path"], str(output.resolve()))
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])

    def test_main_redacts_unexpected_exception_message(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "acceptance.json"
            with (
                patch.object(
                    acceptance,
                    "_reference_binding_from_args",
                    return_value=_reference_binding(),
                ),
                patch.object(
                    acceptance,
                    "_release_binding",
                    return_value={"package_version": "0.2.1"},
                ),
                patch.object(
                    acceptance,
                    "run_acceptance",
                    side_effect=RuntimeError(
                        "Bearer should-not-be-persisted at https://private.example/v1"
                    ),
                ),
            ):
                with redirect_stdout(io.StringIO()):
                    status = acceptance.main(
                        ["--db", "unused.sqlite3", "--output", str(output)]
                    )

            self.assertEqual(status, 1)
            encoded = output.read_text(encoding="utf-8")
            persisted = json.loads(encoded)
            self.assertEqual(persisted["failure"]["error_type"], "RuntimeError")
            self.assertEqual(
                persisted["failure"]["error_code"], "acceptance_exception"
            )
            self.assertNotIn("should-not-be-persisted", encoded)
            self.assertNotIn("private.example", encoded)

    def test_main_redacts_cli_values_when_reference_setup_fails(self) -> None:
        class UntrustedSetupError(RuntimeError):
            error_code = "looks_safe_but_is_not_host_owned"

        with TemporaryDirectory() as directory:
            output = Path(directory) / "acceptance.json"
            with patch.object(
                acceptance,
                "_reference_binding_from_args",
                side_effect=UntrustedSetupError("Bearer raw-exception-secret"),
            ):
                with redirect_stdout(io.StringIO()):
                    status = acceptance.main(
                        [
                            "--db",
                            "unused.sqlite3",
                            "--reference-run-id",
                            r"C:\private\reference",
                            "--planner",
                            "Bearer planner-secret",
                            "--critic",
                            "/Users/private/critic-secret",
                            "--dataset",
                            "api_key=dataset-secret",
                            "--episode",
                            "https://private.example/episode",
                            "--output",
                            str(output),
                        ]
                    )

            self.assertEqual(status, 1)
            encoded = output.read_text(encoding="utf-8")
            persisted = json.loads(encoded)
            self.assertEqual(
                persisted["failure"]["error_code"], "acceptance_exception"
            )
            self.assertEqual(persisted["binding_reference_run_id"], "[REDACTED]")
            self.assertEqual(persisted["request"]["planner_model_id"], "[REDACTED]")
            self.assertEqual(persisted["request"]["critic_model_id"], "[REDACTED]")
            self.assertEqual(persisted["request"]["dataset_id"], "[REDACTED]")
            self.assertEqual(persisted["request"]["episode_id"], "[REDACTED]")
            for secret in (
                "planner-secret",
                "critic-secret",
                "dataset-secret",
                "private.example",
                "raw-exception-secret",
                "looks_safe_but_is_not_host_owned",
            ):
                self.assertNotIn(secret, encoded)

    def test_main_rejects_release_binding_drift_after_network_execution(self) -> None:
        report = {
            "schema_version": "ecologyrsi-dsh.real-api-agent-tool-acceptance/1",
            "passed": True,
        }
        with TemporaryDirectory() as directory:
            output = Path(directory) / "acceptance.json"
            with (
                patch.object(
                    acceptance,
                    "_reference_binding_from_args",
                    return_value=_reference_binding(),
                ),
                patch.object(
                    acceptance,
                    "_release_binding",
                    side_effect=(
                        {"package_version": "0.2.1", "source": "before"},
                        {"package_version": "0.2.1", "source": "after"},
                    ),
                ),
                patch.object(acceptance, "run_acceptance", return_value=report),
            ):
                with redirect_stdout(io.StringIO()):
                    status = acceptance.main(
                        ["--db", "unused.sqlite3", "--output", str(output)]
                    )

            self.assertEqual(status, 1)
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(persisted["passed"])
            self.assertEqual(
                persisted["failure"]["error_code"], "release_binding_changed"
            )
            self.assertNotIn("release_binding", persisted)

    def test_reference_binding_selects_latest_matching_run_read_only(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "evolution.sqlite3"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    """
                    CREATE TABLE evolution_events (
                        seq INTEGER PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                for seq, run_id, planner, critic, planner_digest in (
                    (1, "run:unrelated", "local", "local", "1" * 64),
                    (
                        2,
                        "run:older",
                        "pjlab/glm-5.2",
                        "pjlab/deepseek-v4-flash-0731",
                        "2" * 64,
                    ),
                    (
                        3,
                        "run:latest",
                        "pjlab/glm-5.2",
                        "pjlab/deepseek-v4-flash-0731",
                        "3" * 64,
                    ),
                ):
                    metadata = {
                        "strategy_model_id": planner,
                        "policy_model_id": planner,
                        "review_model_id": critic,
                        "judge_model_id": critic,
                        "strategy_model_digest": planner_digest,
                        "policy_model_digest": planner_digest,
                        "review_model_digest": "4" * 64,
                        "judge_model_digest": "4" * 64,
                    }
                    connection.execute(
                        "INSERT INTO evolution_events VALUES (?, ?, ?, ?, ?)",
                        (
                            seq,
                            run_id,
                            "RunCreated",
                            json.dumps(
                                {"task_manifest": {"metadata": metadata}}
                            ),
                            f"2026-08-19T00:00:0{seq}+00:00",
                        ),
                    )
            before = database.stat().st_mtime_ns

            binding = acceptance._read_reference_run_binding(database)  # noqa: SLF001

            self.assertEqual(binding["run_id"], "run:latest")
            self.assertEqual(binding["event_seq"], 3)
            self.assertEqual(
                binding["planner"]["configuration_digest"], "3" * 64
            )
            self.assertEqual(database.stat().st_mtime_ns, before)
            selected = acceptance._read_reference_run_binding(  # noqa: SLF001
                database,
                reference_run_id="run:older",
                planner_override="pjlab/glm-5.2",
                critic_override="pjlab/deepseek-v4-flash-0731",
                planner_digest_override="2" * 64,
                critic_digest_override="4" * 64,
            )
            self.assertEqual(selected["run_id"], "run:older")

    def test_reference_binding_rejects_conflicts_and_override_drift(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "evolution.sqlite3"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    """
                    CREATE TABLE evolution_events (
                        seq INTEGER PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                metadata = {
                    "strategy_model_id": "pjlab/glm-5.2",
                    "policy_model_id": "other/glm-5.2",
                    "review_model_id": "pjlab/deepseek-v4-flash-0731",
                    "judge_model_id": "pjlab/deepseek-v4-flash-0731",
                    "strategy_model_digest": "a" * 64,
                    "review_model_digest": "b" * 64,
                }
                connection.execute(
                    "INSERT INTO evolution_events VALUES (?, ?, ?, ?, ?)",
                    (
                        1,
                        "run:conflict",
                        "RunCreated",
                        json.dumps({"task_manifest": {"metadata": metadata}}),
                        "2026-08-19T00:00:00+00:00",
                    ),
                )

            with self.assertRaisesRegex(ValueError, "conflicting"):
                acceptance._read_reference_run_binding(database)  # noqa: SLF001

            metadata["policy_model_id"] = metadata["strategy_model_id"]
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "UPDATE evolution_events SET payload_json = ?",
                    (json.dumps({"task_manifest": {"metadata": metadata}}),),
                )
            with self.assertRaisesRegex(ValueError, "--planner"):
                acceptance._read_reference_run_binding(  # noqa: SLF001
                    database,
                    planner_override="wrong/glm-5.2",
                )

    def test_model_matchers_reject_adjacent_versions_and_impostors(self) -> None:
        for model_id in (
            "pjlab/glm-5.2",
            "secure/glm52",
        ):
            with self.subTest(model_id=model_id):
                self.assertTrue(acceptance._is_glm_52(model_id))  # noqa: SLF001
        for model_id in (
            "pjlab/glm-5.20",
            "attacker/notglm52",
            "attacker/glm52proxy",
        ):
            with self.subTest(model_id=model_id):
                self.assertFalse(acceptance._is_glm_52(model_id))  # noqa: SLF001

        self.assertTrue(  # noqa: SLF001
            acceptance._is_deepseek_flash("pjlab/deepseek-v4-flash-0731")
        )
        for model_id in (
            "attacker/deepseek-not-the-model-flash-proxy",
            "attacker/notdeepseek-flash",
            "attacker/deepseek-flash-proxy",
        ):
            with self.subTest(model_id=model_id):
                self.assertFalse(
                    acceptance._is_deepseek_flash(model_id)  # noqa: SLF001
                )

    def test_source_manifest_is_order_stable_and_content_sensitive(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "scripts" / "real_api_agent_tool_acceptance.py"
            module = root / "src" / "ecologyrsi_dsh" / "module.py"
            script.parent.mkdir(parents=True)
            module.parent.mkdir(parents=True)
            script.write_bytes(b"script-v1\n")
            module.write_bytes(b"module-v1\n")

            first = acceptance._source_manifest(  # noqa: SLF001
                root, (module, script)
            )
            reordered = acceptance._source_manifest(  # noqa: SLF001
                root, (script, module)
            )
            self.assertEqual(first, reordered)
            script_digest = hashlib.sha256(b"script-v1\n").hexdigest()
            module_digest = hashlib.sha256(b"module-v1\n").hexdigest()
            expected_lines = (
                f"{script_digest}  "
                "scripts/real_api_agent_tool_acceptance.py\n"
                f"{module_digest}  "
                "src/ecologyrsi_dsh/module.py\n"
            ).encode("utf-8")
            self.assertEqual(
                first["manifest_sha256"], hashlib.sha256(expected_lines).hexdigest()
            )

            module.write_bytes(b"module-v2\n")
            changed = acceptance._source_manifest(root, (script, module))  # noqa: SLF001
            self.assertNotEqual(first["manifest_sha256"], changed["manifest_sha256"])

    def test_release_binding_records_source_script_and_three_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            script = root / "scripts" / "real_api_agent_tool_acceptance.py"
            script.write_bytes(b"acceptance-script\n")
            (root / "pyproject.toml").write_text(
                f'[project]\nversion = "{acceptance.__version__}"\n',
                encoding="utf-8",
            )
            dist = root / "dist"
            dist.mkdir()
            artifacts = {
                "wheel": dist
                / f"ecologyrsi_dsh-{acceptance.__version__}-py3-none-any.whl",
                "sdist": dist / f"ecologyrsi_dsh-{acceptance.__version__}.tar.gz",
                "delivery_archive": dist
                / f"ecologyrsi-dsh-{acceptance.__version__}-delivery.tar.gz",
            }
            for kind, path in artifacts.items():
                path.write_bytes(f"{kind}\n".encode("ascii"))

            with patch.object(acceptance, "_verify_release_artifacts") as verifier:
                binding = acceptance._release_binding(  # noqa: SLF001
                    dist,
                    source_root=root,
                    source_files=(script, root / "pyproject.toml"),
                )

            verifier.assert_called_once()
            self.assertEqual(binding["package_version"], acceptance.__version__)
            self.assertEqual(binding["source_manifest"]["file_count"], 2)
            self.assertEqual(
                binding["acceptance_script_sha256"],
                hashlib.sha256(script.read_bytes()).hexdigest(),
            )
            self.assertEqual(set(binding["artifacts"]), set(artifacts))
            for kind, path in artifacts.items():
                self.assertEqual(binding["artifacts"][kind]["filename"], path.name)
                self.assertEqual(
                    binding["artifacts"][kind]["sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )

    def test_release_binding_rejects_a_missing_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            script = root / "scripts" / "real_api_agent_tool_acceptance.py"
            script.write_bytes(b"acceptance-script\n")
            (root / "pyproject.toml").write_text(
                f'[project]\nversion = "{acceptance.__version__}"\n',
                encoding="utf-8",
            )
            dist = root / "dist"
            dist.mkdir()
            (dist / f"ecologyrsi_dsh-{acceptance.__version__}.tar.gz").write_bytes(
                b"sdist"
            )
            with self.assertRaisesRegex(RuntimeError, "wheel release artifact"):
                acceptance._release_binding(  # noqa: SLF001
                    dist,
                    source_root=root,
                    source_files=(script, root / "pyproject.toml"),
                )

    def test_release_binding_rejects_named_plaintext_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text(
                f'[project]\nversion = "{acceptance.__version__}"\n',
                encoding="utf-8",
            )
            dist = root / "dist"
            dist.mkdir()
            for path in (
                dist / f"ecologyrsi_dsh-{acceptance.__version__}-py3-none-any.whl",
                dist / f"ecologyrsi_dsh-{acceptance.__version__}.tar.gz",
                dist / f"ecologyrsi-dsh-{acceptance.__version__}-delivery.tar.gz",
            ):
                path.write_bytes(b"not-an-archive\n")
            (dist / "SHA256SUMS").write_text("not-a-checksum\n", encoding="ascii")

            with self.assertRaises(acceptance.ReleaseArtifactVerificationError):
                acceptance._release_binding(dist, source_root=root)  # noqa: SLF001

    def test_report_output_rejects_sources_artifacts_checksums_and_database(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dist = root / "dist"
            scripts = root / "scripts"
            dist.mkdir()
            scripts.mkdir()
            source = scripts / "existing-report.json"
            source.write_text("{}\n", encoding="utf-8")
            database = Path(directory) / "ledger.sqlite3"
            database.write_bytes(b"ledger")
            protected = (
                source,
                scripts / "new-report.json",
                dist / "SHA256SUMS",
                dist / f"ecologyrsi_dsh-{acceptance.__version__}-py3-none-any.whl",
                dist / f"ecologyrsi_dsh-{acceptance.__version__}.tar.gz",
                dist / f"ecologyrsi-dsh-{acceptance.__version__}-delivery.tar.gz",
                database,
            )
            for output in protected:
                with self.subTest(output=output):
                    with self.assertRaises(acceptance.ReportOutputConflictError):
                        acceptance._validate_report_output(  # noqa: SLF001
                            output,
                            dist,
                            source_root=root,
                            extra_protected_paths=(database,),
                        )

            allowed = dist / "real-api-acceptance.json"
            self.assertEqual(
                acceptance._validate_report_output(  # noqa: SLF001
                    allowed,
                    dist,
                    source_root=root,
                    extra_protected_paths=(database,),
                ),
                allowed.resolve(),
            )

            skipped = Path(directory) / "must-not-be-written.json"
            stdout = io.StringIO()
            with (
                patch.object(
                    acceptance,
                    "_validate_report_output",
                    side_effect=acceptance.ReportOutputConflictError("private path"),
                ),
                patch.object(acceptance, "_reference_binding_from_args") as reference,
                redirect_stdout(stdout),
            ):
                status = acceptance.main(
                    ["--db", str(database), "--output", str(skipped)]
                )
            self.assertEqual(status, 1)
            self.assertFalse(skipped.exists())
            reference.assert_not_called()
            self.assertEqual(
                json.loads(stdout.getvalue())["failure"]["error_code"],
                "report_output_conflict",
            )

    def test_gateway_diagnostics_omit_error_text_and_retry_reason(self) -> None:
        class Gateway:
            @staticmethod
            def connection_status(model_id: str) -> dict[str, object]:
                return {
                    "state": "error",
                    "last_checked_at": "2026-08-19T00:00:00+00:00",
                    "last_error": "https://private.example/v1 Bearer secret",
                    "last_request": {
                        "operation": "sample.planner",
                        "outcome": "error",
                        "attempts": 2,
                        "retry_count": 1,
                        "classification": "transient",
                        "last_error": "https://private.example/v1 Bearer secret",
                        "retries": [
                            {
                                "attempt": 1,
                                "reason": "https://private.example/v1 Bearer secret",
                                "delay_seconds": 1.25,
                            }
                        ],
                    },
                }

        diagnostics = acceptance._gateway_diagnostics(  # noqa: SLF001
            Gateway(),  # type: ignore[arg-type]
            ("newapi/glm-5.2",),
        )
        encoded = json.dumps(diagnostics)

        self.assertNotIn("private.example", encoded)
        self.assertNotIn("Bearer", encoded)
        request = diagnostics["newapi/glm-5.2"]["last_request"]
        assert isinstance(request, dict)
        self.assertEqual(request["retries"], [{"attempt": 1, "delay_seconds": 1.25}])


if __name__ == "__main__":
    unittest.main()
