from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest

from ecologyrsi_dsh.core.director import EvolutionDirector
from ecologyrsi_dsh.core.ledger import EventLedger, SCHEMA_VERSION
from ecologyrsi_dsh.core.models import Run
from ecologyrsi_dsh.core.model_preflight import AUDIT_METADATA_KEY
from ecologyrsi_dsh.core.state import project_run_state
from ecologyrsi_dsh.api.handler import _runtime_build_compatible
from tests.native_fixtures import native_manifest, NativeRoleStub, start_native_run


class CurrentRunContractTests(unittest.TestCase):
    def test_old_protocols_are_rejected_before_append(self):
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            base = native_manifest()
            for overrides in ({"execution_protocol": "legacy"},
                              {"optimization_protocol": "joint_plugin_epoch@1"},
                              {"sample_agent_protocol": "dsh-strict-origin-bundle@3"},
                              {"search_guard_policy": None},
                              {"paired_execution_qualification": None},
                              {"samples_per_update": 10},
                              {"local_comparison_policy": "exploratory_paired_point_comparison"}):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    director.create_run(replace(base, metadata={**base.metadata, **overrides}))
                self.assertEqual(ledger.count(), 0)

    def test_current_run_cannot_bypass_four_role_preflight_requirement(self):
        with EventLedger() as ledger:
            director = EvolutionDirector(ledger, NativeRoleStub())
            base = native_manifest()
            for overrides in ({"require_model_contract_preflight": False},
                              {AUDIT_METADATA_KEY: None},
                              {AUDIT_METADATA_KEY: "ecologyrsi-dsh.model-contract-preflight-recorded/1"}):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    task = replace(base, metadata={**base.metadata, **overrides})
                    director.create_run(task)
                self.assertEqual(ledger.count(), 0)

    def test_old_contract_cannot_be_imported_by_raw_event_replay(self):
        with EventLedger() as ledger:
            task = native_manifest(metadata={"optimization_protocol": "joint_plugin_epoch@1"})
            run = Run(run_id="old", task_id=task.task_id, task_manifest_digest=task.digest)
            ledger.append("old", "RunCreated", {"task_manifest": task.to_dict(), "run": run.to_dict()})
            with self.assertRaises(ValueError):
                project_run_state(ledger.events("old"))

    def test_runtime_identity_requires_exact_current_package(self):
        build = dict(native_manifest().metadata["host_runtime_build"])
        self.assertTrue(_runtime_build_compatible(build))
        for mutation in ({"package_version": "0.10.11"},
                         {"evolution_runtime_schema": "ecologyrsi-dsh.evolution-runtime/2"}):
            self.assertFalse(_runtime_build_compatible({**build, **mutation}))

    def test_current_pause_reopen_resume_keeps_frozen_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            with EventLedger(path) as ledger:
                director = EvolutionDirector(ledger, NativeRoleStub())
                state = start_native_run(director)
                before_digest = state.task_manifest.digest
                seed = state.materialized_seed_genome().genome_digest
                director.pause_run(state.run.run_id)
                count = ledger.count()
            with EventLedger(path) as reopened:
                self.assertEqual(reopened.schema_version, SCHEMA_VERSION)
                self.assertEqual(reopened.count(), count)
                director = EvolutionDirector(reopened, NativeRoleStub())
                director.resume_run(state.run.run_id)
                restored = director.state(state.run.run_id)
                self.assertEqual(restored.task_manifest.digest, before_digest)
                self.assertEqual(restored.materialized_seed_genome().genome_digest, seed)
                self.assertEqual(restored.run.status.value, "running")

    def test_legacy_database_rejection_does_not_migrate_or_erase_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.sqlite3"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE old_evidence(value TEXT)")
                db.execute("INSERT INTO old_evidence VALUES ('preserved')")
                db.execute("PRAGMA user_version=8")
            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                EventLedger(path)
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 8)
                self.assertEqual(db.execute("SELECT * FROM old_evidence").fetchall(), [("preserved",)])
                self.assertEqual(db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [("old_evidence",)])
