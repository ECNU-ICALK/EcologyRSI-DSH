from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ecogym.episodes.judge import JUDGE_PROTOCOL, judge_program
from ecogym.episodes.workspace import LOCAL_STAGE
from ecogym.evolution.methods import MethodError, MethodRegistry, canonical_source

from .helpers import synthetic_bench
from .test_episodes import RUNTIME, requires_sandbox

SOURCE = "https://example.org/forecasts/reference"
PREDICTOR = """
class Predictor:
    def __init__(self, spec):
        self.targets = spec['targets']
        self.horizons = spec['horizons']

    def fit(self, history):
        pass

    def forecast(self, history):
        return {name: [history[name].iloc[-1]] * len(self.horizons) for name in self.targets}
"""


class MethodRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.registry = MethodRegistry(self.root / "registry")
        self.source = self.root / "offline-source"
        self.source.mkdir()
        self.predictor = self.source / "predictor.py"
        self.predictor.write_text(PREDICTOR, encoding="utf-8")

    def approve(self, revision: str = ""):
        return self.registry.approve("test-method", self.source, source=SOURCE, license="MIT", revision=revision)

    def test_approval_is_content_addressed_and_does_not_run_code(self) -> None:
        self.assertFalse(self.registry.available())
        self.assertFalse(self.registry.root.exists())
        initial = self.registry.identity()
        self.predictor.write_text("raise RuntimeError('must never execute on approval')\n", encoding="utf-8")
        original = self.predictor.read_bytes()
        approved = self.approve()
        self.assertNotEqual(initial, self.registry.identity())
        self.assertEqual(self.approve(), approved)
        self.predictor.write_text("# unpublished edit\n", encoding="utf-8")
        target = self.root / "new-template"
        self.assertEqual(self.registry.materialize(approved.digest, target), approved)
        self.assertEqual((target / "predictor.py").read_bytes(), original)
        self.assertFalse((target / "method.json").exists())
        self.assertNotIn(str(self.root), repr(self.registry.cards()))
        with self.assertRaisesRegex(MethodError, "fresh target"):
            self.registry.materialize(approved.digest, target)

    def test_exact_source_identity_and_ambiguous_id(self) -> None:
        first, second = self.approve("one"), self.approve("two")
        self.assertEqual(canonical_source("HTTPS://EXAMPLE.ORG:443/forecasts/reference/#section"), SOURCE)
        self.assertEqual(set(self.registry.find_source(SOURCE)), {first, second})
        self.assertEqual(self.registry.find_source(SOURCE + "?revision=two"), ())
        self.assertEqual(self.registry.find_source(SOURCE + "/different"), ())
        with self.assertRaisesRegex(MethodError, "matches 2 approvals"):
            self.registry.resolve("test-method")
        with self.assertRaisesRegex(MethodError, "without credentials"):
            canonical_source("https://user@example.org/forecasts/reference")
        with self.assertRaisesRegex(MethodError, "500 characters"):
            canonical_source(SOURCE + "/" + "a" * 500)

    def test_changed_registry_source_and_binary_payload_are_refused(self) -> None:
        (self.source / "checkpoint.safetensors").write_bytes(b"test")
        with self.assertRaisesRegex(MethodError, "unsupported method source file"):
            self.approve()
        (self.source / "checkpoint.safetensors").unlink()
        approved = self.approve()
        (approved.program / "predictor.py").write_text("# changed\n", encoding="utf-8")
        with self.assertRaisesRegex(MethodError, "source has changed"):
            self.registry.resolve(approved.digest)
        with self.assertRaisesRegex(MethodError, "source has changed"):
            self.registry.available()

    def test_pinned_registry_does_not_expose_new_approvals(self) -> None:
        first = self.approve("one")
        pinned = self.registry.pin()
        second = self.approve("two")
        self.assertEqual(pinned.find_source(SOURCE), (first,))
        self.assertNotEqual(pinned.identity(), self.registry.identity())
        with self.assertRaisesRegex(MethodError, "outside this run"):
            pinned.resolve(second.digest)

    @requires_sandbox
    def test_registered_reference_uses_the_same_local_judge(self) -> None:
        approved = self.approve()
        bench = synthetic_bench()
        expected, direct = judge_program(bench, self.source, LOCAL_STAGE, RUNTIME)
        report, registered = self.registry.judge(approved.digest, bench, LOCAL_STAGE, RUNTIME)
        self.assertEqual(report["judge_protocol"], JUDGE_PROTOCOL)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["method"], approved.describe())
        self.assertEqual(report["score"], expected["score"])
        np.testing.assert_array_equal(registered.predicted, direct.predicted)
        with self.assertRaisesRegex(MethodError, "development stages"):
            self.registry.judge(approved.digest, bench, "validation", RUNTIME)

    @requires_sandbox
    def test_registry_must_be_outside_agent_and_runtime_paths(self) -> None:
        self.registry.check_agent_isolation(self.root / "other-session", RUNTIME)
        with self.assertRaisesRegex(MethodError, "host-only directory"):
            self.registry.check_agent_isolation(self.root, RUNTIME)
