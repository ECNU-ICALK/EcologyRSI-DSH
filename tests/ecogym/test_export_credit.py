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

    def episode(self, name, score, *, start=False, wrong_digest=False, interrupted=False, ambiguous=False,
                task="task", role=None):
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
            ledger.append(name, "episode_started", {"task_id": task, "harness": {"digest": "h"},
                                                   "start": {"local": {"score": 0.1}},
                                                   **({"context": {"role": role}} if role else {})})
            ledger.append(name, "episode_note", {"kind": "finding", "text": f"note of {name}"})
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

    def test_holdout_verification_episodes_are_never_training_data(self):
        """A run's holdout verification (2026-11-18 review: 20 of 23 holdout episodes passed the default filters)
        must not reach SFT, preference pairs or notes; the audit export keeps them, marked and unpaired."""

        self.episode("ep:a", 0.4)
        self.episode("ep:b", 0.3)
        self.episode("ep:holdout", 0.5, role="holdout")
        decision = SimpleNamespace(delta=0.1, interval=(0.05, 0.15))
        with patch("ecogym.episodes.export.Bench.open"), patch("ecogym.episodes.export.task_spec"), \
                patch("ecogym.episodes.export.load_predictions"), \
                patch("ecogym.episodes.export.compare_suite", return_value=decision):
            clean = export(self.ledger_path, self.root / "clean", evolver=False)
            audit = export(self.ledger_path, self.root / "audit", evolver=False, include_failed=True)
        self.assertEqual([sample["id"] for sample in self.rows(self.root / "clean" / "sft.jsonl")],
                         ["ep:a/1", "ep:b/1"])
        self.assertEqual(clean["skipped"]["independent validation episode (role:holdout)"], 1)
        self.assertEqual(clean["filters"]["independent_validation"], "excluded")
        self.assertEqual([note["session"] for note in self.rows(self.root / "clean" / "notes.jsonl")],
                         ["ep:a", "ep:b"])
        for target in ("clean", "audit"):
            with self.subTest(target=target):
                pairs = self.rows(self.root / target / "preferences.jsonl")
                self.assertEqual([(pair["meta"]["chosen"]["session"], pair["meta"]["rejected"]["session"])
                                  for pair in pairs], [("ep:a", "ep:b")])
                indexed = {row["session"]: row["independent_validation"]
                           for row in self.rows(self.root / target / "episodes.jsonl")}
                self.assertEqual(indexed, {"ep:a": None, "ep:b": None, "ep:holdout": "role:holdout"})
        marked = [sample["meta"] for sample in self.rows(self.root / "audit" / "sft.jsonl")
                  if sample["meta"]["session"] == "ep:holdout"]
        self.assertTrue(marked)
        self.assertTrue(all(meta["independent_validation"] == "role:holdout" and not meta["passes_filters"]
                            for meta in marked))
        self.assertEqual(audit["filters"]["independent_validation"], "marked")
        held_note, = [note for note in self.rows(self.root / "audit" / "notes.jsonl")
                      if note["session"] == "ep:holdout"]
        self.assertEqual(held_note["independent_validation"], "role:holdout")

    def test_holdout_of_a_suite_leaves_out_its_holdout_tasks_whatever_the_role(self):
        self.episode("ep:train", 0.4)
        self.episode("ep:elsewhere", 0.4, task="held-task", role="incumbent")
        suite = SimpleNamespace(tasks=lambda side: [SimpleNamespace(id="held-task")] if side == "holdout" else [])
        with patch("ecogym.episodes.export.suite_spec", return_value=suite):
            manifest = export(self.ledger_path, self.root / "clean", evolver=False, holdout_of=["suite@1"])
        self.assertEqual([sample["id"] for sample in self.rows(self.root / "clean" / "sft.jsonl")], ["ep:train/1"])
        self.assertEqual(manifest["skipped"]["independent validation episode (holdout_task)"], 1)
        self.assertEqual(manifest["filters"]["holdout_tasks"], ["held-task"])

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
