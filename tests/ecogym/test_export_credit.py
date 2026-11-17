"""Training credit follows the submitted program recorded by the judge."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ecogym.episodes.export import export
from ecogym.kernel.ledger import Ledger


def call(messages, reply, *, reason="stop", **extra):
    return {"path": "/v1/chat/completions", "status": 200, "request": {"messages": messages},
            "reply": {"message": {"role": "assistant", **reply}, "finish_reason": reason}, **extra}


def submit_turns(call_id, digest, *, number=3):
    return (
        {"role": "assistant", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "mcp__ecogym__submit", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id,
         "content": json.dumps({"n": number, "digest": digest[:12], "accepted": True})},
    )


class ExportCreditTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.ledger_path = self.root / "ledger.sqlite3"

    def episode(self, name, score, *, start=False, wrong_digest=False, interrupted=False, ambiguous=False):
        trajectory = self.root / name
        trajectory.mkdir()
        digest = "a" * 64
        main = {"role": "user", "content": "Original request"}
        continuation = {"role": "user", "content": "Compacted request"}
        submit, result = submit_turns(f"submit-{name}", "b" * 64 if wrong_digest else digest)
        calls = [
            call([main], {"content": "I am looking at the files."}),
            call([continuation], submit, reason="tool_calls"),
            call([continuation, submit, result], {"content": f"Finished {name}."},
                 **({"error": "connection lost"} if interrupted else {})),
            call([{"role": "user", "content": "Unrelated continuation"}], {"content": "A partial answer"},
                 reason=None),
        ]
        if ambiguous:
            other, other_result = submit_turns(f"other-submit-{name}", digest)
            calls.append(call([{"role": "user", "content": "Separate request"}, other, other_result],
                              {"content": "Also claims credit."}))
        with gzip.open(trajectory / "llm_calls.jsonl.gz", "wt", encoding="utf-8") as stream:
            for item in calls:
                stream.write(json.dumps(item) + "\n")
        with Ledger(self.ledger_path) as ledger:
            ledger.append(name, "episode_started", {"task_id": "task", "harness": {"digest": "h"},
                                                   "start": {"local": {"score": 0.1}}})
            ledger.append(name, "episode_submitted", {"n": 3, "digest": digest, "accepted": True,
                                                     "local": {"score": 0.13}})
            ledger.append(name, "episode_scored", {"n": 3, "digest": digest, "start": start, "score": score,
                                                  "predictions": "score.npz", "references": {"reference": 0.2}})
            ledger.append(name, "episode_finished", {"status": "finished", "trajectory": {"dir": str(trajectory)}})

    @staticmethod
    def rows(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_default_and_preferences_use_only_the_credited_continuations(self):
        self.episode("ep:a", 0.4)
        self.episode("ep:b", 0.3)
        self.episode("ep:below", -0.1)
        target = self.root / "clean"
        decision = SimpleNamespace(delta=0.1, interval=(0.05, 0.15))
        with patch("ecogym.episodes.export.Bench.open"), patch("ecogym.episodes.export.task_spec"), \
                patch("ecogym.episodes.export.load_predictions"), \
                patch("ecogym.episodes.export.compare_suite", return_value=decision):
            export(self.ledger_path, target, evolver=False)
            export(self.ledger_path, self.root / "all", evolver=False, include_failed=True)
        samples = self.rows(target / "sft.jsonl")
        self.assertEqual([sample["id"] for sample in samples], ["ep:a/1", "ep:b/1"])
        self.assertEqual(samples[0]["meta"]["credited_tool_call_id"], "submit-ep:a")
        self.assertAlmostEqual(samples[0]["meta"]["local_gain_vs_start"], 0.03)
        pair, = self.rows(target / "preferences.jsonl")
        self.assertEqual(pair["prompt"][0]["content"], "Compacted request")
        self.assertEqual(pair["chosen"][-1]["content"], "Finished ep:a.")
        self.assertEqual(pair["rejected"][-1]["content"], "Finished ep:b.")
        self.assertEqual(self.rows(self.root / "all" / "preferences.jsonl"), [pair])

    def test_diagnostics_keep_uncredited_threads_without_training_labels(self):
        self.episode("ep:a", 0.4)
        self.episode("ep:fallback", 0.5, start=True)
        target = self.root / "diagnostic"
        export(self.ledger_path, target, evolver=False, include_failed=True)
        samples = {sample["id"]: sample["meta"] for sample in self.rows(target / "sft.jsonl")}
        self.assertTrue(samples["ep:a/1"]["passes_filters"])
        self.assertEqual(samples["ep:a/1"]["score"], 0.4)
        self.assertFalse(samples["ep:a/2"]["complete_reply"])
        for name in ("ep:a/0", "ep:a/2", "ep:fallback/1"):
            with self.subTest(name=name):
                self.assertFalse(samples[name]["passes_filters"])
                self.assertIsNone(samples[name]["score"])
                self.assertIsNone(samples[name]["scored_submission"])
                self.assertIsNone(samples[name]["credited_tool_call_id"])
                self.assertEqual(samples[name]["beats"], {})
        self.assertEqual(samples["ep:fallback/1"]["episode_score"], 0.5)

    def test_wrong_or_ambiguous_lineage_and_partial_reply_skip_training_only(self):
        self.episode("ep:wrong", 0.4, wrong_digest=True)
        self.episode("ep:partial", 0.4, interrupted=True)
        self.episode("ep:ambiguous", 0.4, ambiguous=True)
        target = self.root / "clean"
        result = export(self.ledger_path, target, evolver=False)
        self.assertEqual(result["skipped"]["episode without an attributable scored submission thread"], 3)
        self.assertEqual(self.rows(target / "sft.jsonl"), [])
        self.assertEqual(self.rows(target / "preferences.jsonl"), [])
        self.assertEqual(len(self.rows(target / "episodes.jsonl")), 3)
