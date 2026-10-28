from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from unittest import mock

import numpy as np

from ecogym.agents import LLMProposer, OpenAICompatibleClient, ResearchTools
from ecogym.agents.clients import AgentError, Reply, make_client, render_transcript, resolve_secret
from ecogym.agents.replies import extract_json
from ecogym.arena import Arena, RunConfig, fold, replay
from ecogym.arena.proposers import make_proposer
from ecogym.kernel.ledger import Ledger
from ecogym.protocol.cohorts import plan_generation

from .helpers import synthetic_bench
from .test_arena import WEAK


class ScriptedClient:
    """Replays canned replies and records what the agents were asked."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def describe(self):
        return {"transport": "scripted", "model": "scripted"}

    def complete(self, messages):
        self.seen.append([dict(message) for message in messages])
        return Reply(self.replies.pop(0), "scripted", 0.0)


def script(edits, *, review=None):
    replies = [
        json.dumps({"tool": "residual_profile", "args": {"cell": "y@1", "stage": "screening"}, "why": "look"}),
        "Here you go:\n```json\n" + json.dumps({"tool": "try_edit", "args": {"changes": edits[0]}}) + "\n```",
        # a note without its {"note": ...} wrapper is accepted too
        json.dumps({"observations": ["误差在白天更大"], "hypotheses": ["h"], "directions": ["d"]}),
        json.dumps({"edits": [{"changes": changes, "rationale": f"edit {i}"} for i, changes in enumerate(edits)]}),
    ]
    if review is not None:
        replies.append(json.dumps({"reviews": review}))
    return replies


class ReplyTests(unittest.TestCase):
    def test_extract_json_skips_prose_and_fences(self):
        self.assertEqual(extract_json('ok {"a": 1} and {"b": 2}'), {"a": 1})
        self.assertEqual(extract_json('```json\n{"x": [1, {"y": 2}]}\n```'), {"x": [1, {"y": 2}]})
        with self.assertRaises(AgentError):
            extract_json("no json here {")

    def test_transcript_keeps_roles(self):
        text = render_transcript([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
        self.assertIn("[system]\nS", text)
        self.assertTrue(text.endswith("with the reply only."))

    def test_secrets_come_from_env_or_a_credentials_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials.yaml"
            path.write_text("refs:\n  MY_TEST_KEY: 'abc123'\n", encoding="utf-8")
            previous = os.environ.get("ECOGYM_CREDENTIALS_FILE")
            os.environ["ECOGYM_CREDENTIALS_FILE"] = str(path)
            try:
                self.assertEqual(resolve_secret("MY_TEST_KEY"), "abc123")
                with self.assertRaises(AgentError):
                    resolve_secret("ABSENT_TEST_KEY")
            finally:
                if previous is None:
                    os.environ.pop("ECOGYM_CREDENTIALS_FILE")
                else:
                    os.environ["ECOGYM_CREDENTIALS_FILE"] = previous


class _FakeProvider(BaseHTTPRequestHandler):
    calls: ClassVar[list[dict]] = []
    fail_first = False
    cut_stream = False

    def log_message(self, *args):
        return None

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append({"body": body, "auth": self.headers.get("Authorization")})
        if type(self).fail_first and len(type(self).calls) == 1:
            self.send_response(503)
            self.end_headers()
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in ('{"ok":', " true}"):
                chunk = {"model": "fake", "choices": [{"delta": {"content": piece}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                if type(self).cut_stream:
                    return  # the connection closes after the first piece
            self.wfile.write(b'data: {"choices": [], "usage": {"total_tokens": 7}}\n\ndata: [DONE]\n\n')
        else:
            payload = json.dumps({"model": "fake", "choices": [{"message": {"content": '{"ok": false}'}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


class HttpClientTests(unittest.TestCase):
    def setUp(self):
        _FakeProvider.calls = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeProvider)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        os.environ["ECOGYM_TEST_KEY"] = "secret-123"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        os.environ.pop("ECOGYM_TEST_KEY", None)

    def test_streamed_and_plain_replies(self):
        client = OpenAICompatibleClient(self.base, "fake", key_env="ECOGYM_TEST_KEY", retries=0)
        reply = client.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(reply.text, '{"ok": true}')
        self.assertEqual(reply.usage, {"total_tokens": 7})
        self.assertEqual(_FakeProvider.calls[0]["auth"], "Bearer secret-123")
        plain = OpenAICompatibleClient(self.base, "fake", key_env="ECOGYM_TEST_KEY", retries=0, stream=False)
        self.assertEqual(plain.complete([{"role": "user", "content": "hi"}]).text, '{"ok": false}')
        self.assertNotIn("secret-123", json.dumps(client.describe()))

    def test_fallback_endpoints_answer_when_the_primary_fails(self):
        options = {"base_url": "http://127.0.0.1:9/v1", "model": "down", "key_env": "ECOGYM_TEST_KEY", "retries": 0,
                   "fallbacks": [{"base_url": self.base, "model": "fake"}]}
        client = make_client(options)
        self.assertEqual(client.describe()["fallbacks"][0]["model"], "fake")
        self.assertEqual(client.complete([{"role": "user", "content": "hi"}]).text, '{"ok": true}')

    def test_a_cut_stream_is_not_taken_for_a_whole_reply(self):
        _FakeProvider.cut_stream = True
        try:
            client = OpenAICompatibleClient(self.base, "fake", key_env="ECOGYM_TEST_KEY", retries=1)
            with mock.patch("ecogym.agents.clients.time.sleep"), self.assertRaises(AgentError) as caught:
                client.complete([{"role": "user", "content": "hi"}])
            self.assertIn("IncompleteRead", str(caught.exception))
            self.assertEqual(len(_FakeProvider.calls), 2)
        finally:
            _FakeProvider.cut_stream = False

    def test_transient_errors_are_retried(self):
        _FakeProvider.fail_first = True
        try:
            client = OpenAICompatibleClient(self.base, "fake", key_env="ECOGYM_TEST_KEY", retries=1)
            with mock.patch("ecogym.agents.clients.time.sleep"):
                self.assertEqual(client.complete([{"role": "user", "content": "hi"}]).text, '{"ok": true}')
            self.assertEqual(len(_FakeProvider.calls), 2)
        finally:
            _FakeProvider.fail_first = False


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.bench = synthetic_bench()
        stage = self.bench.stage("model_selection")
        self.screening, self.holdout = plan_generation(stage.time, block_steps=24, chunk_blocks=3,
                                                       screen_fraction=1 / 3, seed=0, generation=0)
        self.tools = ResearchTools(self.bench, self.bench.genome(WEAK), self.screening.index, max_step=0.15,
                                   max_changes=3)

    def test_tools_answer_and_report_errors(self):
        overview = self.tools.call("overview", {})
        self.assertEqual(overview["screening_origins"], self.screening.size)
        profile = self.tools.call("column_profile", {"column": "x", "stage": "calibration_fit"})
        self.assertEqual(len(profile["mean_by_phase"]), 24)
        residual = self.tools.call("residual_profile", {"cell": "y@6", "stage": "screening"})
        self.assertEqual(residual["origins"], self.screening.size)
        self.assertIn("error", self.tools.call("column_profile", {"column": "x", "stage": "validation"}))
        self.assertIn("error", self.tools.call("nope", {}))
        self.assertFalse(self.tools.call("try_edit", {"changes": {"scale.y@1": 1.0}})["valid"])
        self.assertTrue(self.tools.call("try_edit", {"changes": {"scale.y@1": 0.4}})["valid"])

    def test_tools_never_read_holdout_days(self):
        seen: list[np.ndarray] = []
        original = self.bench.stage

        def spy(name, **kwargs):
            data = original(name, **kwargs)
            seen.append(name)
            return data

        self.bench.stage = spy
        try:
            for stage in ("calibration_fit", "screening"):
                self.tools.call("column_profile", {"column": "x", "stage": stage})
                self.tools.call("residual_profile", {"cell": "y@1", "stage": stage})
        finally:
            del self.bench.stage
        self.assertTrue(set(seen) <= {"calibration_fit", "model_selection"})
        self.assertFalse(set(self.screening.blocks) & set(self.holdout.blocks))


class LLMProposerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.ledger = Ledger(Path(self.directory.name) / "ledger.sqlite3")
        self.bench = synthetic_bench()

    def tearDown(self):
        self.ledger.close()
        self.directory.cleanup()

    def test_research_propose_review_round_is_logged_and_replayable(self):
        edits = [{"scale.y@1": 0.4}, {"scale.y@6": 0.4}, {"scale.y@1": 0.95}]
        client = ScriptedClient(script(edits, review=[{"index": 1, "keep": False, "reason": "duplicate in effect"}]))
        proposer = LLMProposer(model="scripted", client=client)
        result = Arena(self.bench, self.ledger, RunConfig(generations=1), proposer).run(self.bench.genome(WEAK))
        view = fold(self.ledger.events(result["run_id"]))
        generation = view["generations"][0]
        self.assertEqual(view["agent_exchanges"], 5)
        self.assertEqual(len(generation["proposed"]), 1)  # edit 0 kept; edit 1 dropped by review
        self.assertEqual(len(generation["rejected"]), 1)  # edit 2 leaves the trust region
        self.assertEqual(generation["notes"][0]["observations"], ["误差在白天更大"])
        self.assertEqual(generation["dropped"], [{"changes": {"scale.y@6": 0.4}, "reason": "duplicate in effect"}])
        system = client.seen[0][0]["content"]
        self.assertIn("How proposals are judged", system)
        self.assertNotIn("validation data", system.lower())
        tool_result = client.seen[1][-1]["content"]
        self.assertTrue(tool_result.startswith("Tool result (residual_profile)"))
        self.assertTrue(replay(self.ledger, result["run_id"], self.bench)["identical"])
        self.assertNotIn("api_key", json.dumps(view["proposer"]))

    def test_a_failed_round_is_logged_and_yields_no_candidates(self):
        client = ScriptedClient(["not json", "still not json"])
        proposer = LLMProposer(model="scripted", client=client)
        result = Arena(self.bench, self.ledger, RunConfig(generations=1), proposer).run(self.bench.genome(WEAK))
        view = fold(self.ledger.events(result["run_id"]))
        self.assertEqual(view["status"], "finished")
        self.assertEqual(view["generations"][0]["proposed"], [])
        exchanges = self.ledger.events(result["run_id"], "agent_exchange")
        errors = [event.body for event in exchanges if "error" in event.body]
        self.assertEqual(len(errors), 1)
        self.assertIn("no JSON object", errors[0]["error"])

    def test_registered_as_a_proposer(self):
        proposer = make_proposer("llm", {"model": "m", "base_url": "http://127.0.0.1:9/v1"})
        self.assertFalse(proposer.deterministic)
        self.assertEqual(proposer.options()["key_env"], "ECOGYM_LLM_API_KEY")


if __name__ == "__main__":
    unittest.main()
