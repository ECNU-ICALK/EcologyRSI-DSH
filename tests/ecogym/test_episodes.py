"""Episode machinery: the LLM gateway, the MCP server, the judge's streaming, the sandbox.

The security claims phase 1 must prove, as tests:

* the agent never holds the real API key (the gateway injects it, meters tokens,
  enforces the deadline, and never logs the key);
* a submitted predictor is fitted and streamed one row at a time, so it can see
  no value at or after the origin it forecasts, and the fit sees nothing from the
  scored stage;
* the exported workspace holds no row at or after the scored stage;
* under the sandbox, a process cannot read the credentials file or the dataset
  and cannot open a socket (darwin only; skipped where sandbox-exec is absent).
"""

from __future__ import annotations

import http.client
import json
import tempfile
import time
import unittest
from importlib import resources
from pathlib import Path

from ecogym.episodes import sandbox as sbx
from ecogym.episodes.gateway import Gateway, tokens_of, usage_of
from ecogym.episodes.judge import Limits, iso, judge, predictor_spec, run_predictor, stage_stream
from ecogym.episodes.mcp import McpServer, Tool
from ecogym.episodes.workspace import LOCAL_STAGE, SCORED_STAGE, boundaries, export_visible

from .helpers import synthetic_bench, synthetic_frame

try:
    RUNTIME = sbx.discover_runtime()
except sbx.SandboxUnavailable:
    RUNTIME = None
HAVE_SANDBOX = sbx.available() and RUNTIME is not None
requires_sandbox = unittest.skipUnless(HAVE_SANDBOX, "agent runtime or sandbox-exec not available")

LEAKAGE_PREDICTOR = """
import json, os
LOG = os.path.join(os.getcwd(), "seen.jsonl")

class Predictor:
    def __init__(self, spec):
        self.targets = spec["targets"]
        self.horizons = spec["horizons"]

    def _log(self, op, history):
        last = None if len(history) == 0 else history.index[-1].strftime("%Y-%m-%dT%H:%M")
        with open(LOG, "a") as handle:
            handle.write(json.dumps({"op": op, "n": int(len(history)), "last": last}) + "\\n")

    def fit(self, history):
        self._log("fit", history)

    def forecast(self, history):
        self._log("forecast", history)
        now = {t: float(history[t].to_numpy()[-1]) for t in self.targets}
        return {t: [now[t]] * len(self.horizons) for t in self.targets}
"""

PROBE = """
import os, socket, sys
secret, data = sys.argv[1], sys.argv[2]

def attempt(label, fn):
    try:
        fn()
        print(label, "ALLOWED")
    except OSError as error:
        print(label, "DENIED", type(error).__name__)

def read(path):
    with open(path, "rb") as handle:
        handle.read(1)

def connect():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        sock.connect(("127.0.0.1", 9))
    finally:
        sock.close()

attempt("own", lambda: read(os.path.join(os.getcwd(), "probe.py")))
attempt("secret", lambda: read(secret))
attempt("data", lambda: read(data))
attempt("net", connect)
"""


def _post(port, path, body, token):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    connection.request("POST", path, body=json.dumps(body), headers=headers)
    response = connection.getresponse()
    data = response.read()
    out = (response.status, data, response.getheader("Content-Type"))
    connection.close()
    return out


class UsageTests(unittest.TestCase):
    def test_tokens_prefers_total_then_pairs(self):
        self.assertEqual(tokens_of({"total_tokens": 42}), 42)
        self.assertEqual(tokens_of({"input_tokens": 40, "output_tokens": 7}), 47)
        self.assertEqual(tokens_of({"prompt_tokens": 3, "completion_tokens": 5}), 8)
        self.assertEqual(tokens_of(None), 0)

    def test_usage_from_event_stream(self):
        stream = ('data: {"type":"message_start","message":{"usage":{"input_tokens":40}}}\n\n'
                  'data: {"type":"message_delta","usage":{"output_tokens":7}}\n\n'
                  "data: [DONE]\n\n")
        usage = usage_of(stream, "text/event-stream")
        self.assertEqual(tokens_of(usage), 47)


class _FakeUpstream:
    """A minimal upstream that records the key it is handed and returns usage."""

    def __init__(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.seen_auth: list[str | None] = []
        self.hits = 0
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                return None

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                upstream.hits += 1
                upstream.seen_auth.append(self.headers.get("Authorization"))
                if body.get("stream"):
                    payload = (b'data: {"type":"message_start","message":{"usage":{"input_tokens":40}}}\n\n'
                               b'data: {"type":"message_delta","usage":{"output_tokens":7}}\n\n'
                               b"data: [DONE]\n\n")
                    ctype = "text/event-stream"
                else:
                    payload = json.dumps({"id": "resp", "usage": {"total_tokens": 100}}).encode()
                    ctype = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        import threading

        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class GatewayTests(unittest.TestCase):
    REAL = "sk-real-upstream-key-SECRET"
    TOKEN = "episode-token-not-the-key"

    def setUp(self):
        self.upstream = _FakeUpstream()
        self.addCleanup(self.upstream.stop)
        self.log = Path(tempfile.mkdtemp(prefix="gw-")) / "llm_calls.jsonl"
        self.logged: list[dict] = []

    def _gateway(self, **kwargs) -> Gateway:
        gateway = Gateway(key=self.REAL, token=self.TOKEN, log_path=self.log, upstream=self.upstream.url,
                          token_budget=kwargs.pop("token_budget", 10_000_000), **kwargs).start()
        gateway.on_call = self.logged.append  # fires after a forwarded call is fully recorded
        self.addCleanup(gateway.stop)
        return gateway

    def _settle(self, expected: int, timeout: float = 5.0) -> None:
        """Wait for the gateway's async handler thread to finish recording ``expected`` calls."""

        deadline = time.time() + timeout
        while len(self.logged) < expected and time.time() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(len(self.logged), expected)

    def test_injects_real_key_meters_tokens_and_hides_key(self):
        gateway = self._gateway()
        status, data, _ = _post(gateway.port, "/deepseek/v1/chat/completions",
                                {"model": "deepseek-v4-pro", "messages": []}, self.TOKEN)
        self._settle(1)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["usage"]["total_tokens"], 100)
        # the gateway handed the upstream the real key, not the agent's token
        self.assertEqual(self.upstream.seen_auth, [f"Bearer {self.REAL}"])
        self.assertEqual(gateway.tokens, 100)
        self.assertEqual(gateway.calls, 1)
        log_text = self.log.read_text(encoding="utf-8")
        self.assertNotIn(self.REAL, log_text)
        self.assertNotIn(self.TOKEN, log_text)
        self.assertIn("deepseek-v4-pro", log_text)  # the request body is recorded

    def test_rejects_a_bad_token(self):
        gateway = self._gateway()
        status, _, _ = _post(gateway.port, "/deepseek/v1/chat/completions", {"messages": []}, "wrong-token")
        self.assertEqual(status, 401)
        self.assertEqual(self.upstream.hits, 0)  # never forwarded

    def test_refuses_once_the_budget_is_spent(self):
        gateway = self._gateway(token_budget=100)
        first, _, _ = _post(gateway.port, "/deepseek/v1/x", {"messages": []}, self.TOKEN)
        self.assertEqual(first, 200)
        self._settle(1)  # the first call's tokens are counted before the second is sent
        second, body, _ = _post(gateway.port, "/deepseek/v1/x", {"messages": []}, self.TOKEN)
        self.assertEqual(second, 429)
        self.assertIn("budget", json.loads(body)["error"]["message"])
        self.assertEqual(gateway.refused, 1)
        self.assertEqual(self.upstream.hits, 1)  # the refused call was not forwarded

    def test_refuses_after_the_deadline(self):
        gateway = self._gateway(deadline=time.time() - 1)
        status, body, _ = _post(gateway.port, "/deepseek/v1/x", {"messages": []}, self.TOKEN)
        self.assertEqual(status, 429)
        self.assertIn("deadline", json.loads(body)["error"]["message"])
        self.assertEqual(self.upstream.hits, 0)

    def test_streams_a_server_sent_response(self):
        gateway = self._gateway()
        status, data, ctype = _post(gateway.port, "/deepseek/v1/messages",
                                    {"stream": True, "messages": []}, self.TOKEN)
        self._settle(1)
        self.assertEqual(status, 200)
        self.assertIn("event-stream", ctype)
        self.assertIn("[DONE]", data.decode())
        self.assertEqual(gateway.tokens, 47)


class McpTests(unittest.TestCase):
    def setUp(self):
        tool = Tool("echo", "echo text back",
                    {"type": "object", "properties": {"text": {"type": "string"}}},
                    lambda text="": {"echo": text})
        self.server = McpServer([tool], token="mcp-token")
        self.server.start()
        self.addCleanup(self.server.stop)

    def _call(self, message, token="mcp-token"):
        status, data, _ = _post(self.server.port, "/mcp", message, token)
        return status, (json.loads(data) if data else None)

    def test_initialize_list_and_call(self):
        status, reply = self._call({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(status, 200)
        self.assertIn("protocolVersion", reply["result"])
        _, listed = self._call({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertEqual([tool["name"] for tool in listed["result"]["tools"]], ["echo"])
        _, called = self._call({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                "params": {"name": "echo", "arguments": {"text": "hi there"}}})
        self.assertFalse(called["result"]["isError"])
        self.assertIn("hi there", called["result"]["content"][0]["text"])

    def test_unknown_tool_is_an_error_not_a_crash(self):
        _, reply = self._call({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                               "params": {"name": "nope", "arguments": {}}})
        self.assertTrue(reply["result"]["isError"])
        self.assertIn("unknown tool", reply["result"]["content"][0]["text"])

    def test_requires_authorization(self):
        status, _ = self._call({"jsonrpc": "2.0", "id": 5, "method": "tools/list"}, token=None)
        self.assertEqual(status, 401)


class ExportTests(unittest.TestCase):
    def test_visible_export_stops_before_the_scored_stage(self):
        import csv

        bench = synthetic_bench()
        bounds = boundaries(bench)
        self.assertEqual((LOCAL_STAGE, SCORED_STAGE), ("calibration_uq", "model_selection"))
        scored_start = bounds["scored"][0]  # ISO minute strings sort lexically
        with tempfile.TemporaryDirectory() as tmp:
            result = export_visible(bench, Path(tmp) / "data")
            with (Path(tmp) / "data" / "history.csv").open(encoding="utf-8") as stream:
                times = [row["time"] for row in csv.DictReader(stream)]
        self.assertGreater(len(times), 0)
        self.assertEqual(result["rows"], len(times))
        self.assertTrue(all(stamp < scored_start for stamp in times))
        self.assertEqual(bounds["visible"][1], scored_start)  # visible ends exactly where scoring starts
        self.assertGreater(bounds["scored"][1], scored_start)  # the scored window is real and lies after


@requires_sandbox
class JudgeStreamingTests(unittest.TestCase):
    def _bench(self):
        return synthetic_bench(frame=synthetic_frame(days=20))

    def test_baseline_submission_scores_about_zero(self):
        bench = self._bench()
        with tempfile.TemporaryDirectory(dir=sbx.scratch_root()) as tmp:
            program = Path(tmp) / "program"
            program.mkdir()
            seed = resources.files("ecogym.harness").joinpath("seed/template/predictor.py")
            (program / "predictor.py").write_bytes(seed.read_bytes())
            report = judge(bench, program, SCORED_STAGE, RUNTIME, limits=Limits(), references=False)
        self.assertEqual(report["status"], "ok")
        self.assertAlmostEqual(report["coverage"], 1.0, places=6)
        self.assertLess(abs(report["score"]), 1e-6)

    def test_predictor_never_sees_the_origin_or_later(self):
        bench = self._bench()
        stream, _ = stage_stream(bench, SCORED_STAGE)
        with tempfile.TemporaryDirectory(dir=sbx.scratch_root()) as tmp:
            program = Path(tmp) / "program"
            program.mkdir()
            (program / "predictor.py").write_text(LEAKAGE_PREDICTOR, encoding="utf-8")
            run = run_predictor(program, stream, predictor_spec(bench), RUNTIME, limits=Limits())
            self.assertEqual(run.status, "ok", run.stderr)
            seen = [json.loads(line) for line in (program / "seen.jsonl").read_text().splitlines()]
        fits = [entry for entry in seen if entry["op"] == "fit"]
        forecasts = [entry for entry in seen if entry["op"] == "forecast"]
        fit_rows = int(stream.fit_values.shape[0])
        # fit sees exactly the rows before the scored stage, nothing from it
        self.assertEqual(len(fits), 1)
        self.assertEqual(fits[0]["n"], fit_rows)
        self.assertEqual(fits[0]["last"], iso(stream.fit_time[-1:], bench.frame.step)[0])
        # each forecast sees every row up to and including its origin, and none after
        origins = iso(stream.time[stream.ask], bench.frame.step)
        self.assertEqual([entry["last"] for entry in forecasts], origins)
        self.assertEqual([entry["n"] for entry in forecasts],
                         [fit_rows + int(position) + 1 for position in stream.ask])


@requires_sandbox
class SandboxTests(unittest.TestCase):
    def _run(self, profile, root, cwd):
        secret = str(Path.home() / ".dsh" / ".credentials.yaml")
        data = str(Path.home() / ".ecologyrsi-dsh" / "data")
        argv = sbx.confine(profile, root / "profile.sb",
                           [str(RUNTIME.python), "-I", "-B", "probe.py", secret, data])
        import subprocess

        done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=120)
        lines = dict(line.split(" ", 1) for line in done.stdout.splitlines() if " " in line)
        return {key: value.strip() for key, value in lines.items()}

    def test_program_cannot_read_secrets_data_or_network(self):
        with tempfile.TemporaryDirectory(dir=sbx.scratch_root()) as tmp:
            program = Path(tmp) / "program"
            program.mkdir()
            (program / "probe.py").write_text(PROBE, encoding="utf-8")
            out = self._run(sbx.program_profile(program, RUNTIME), program, program)
        self.assertEqual(out["own"], "ALLOWED")  # positive control: python works under the sandbox
        self.assertTrue(out["secret"].startswith("DENIED"), out)
        self.assertTrue(out["data"].startswith("DENIED"), out)
        self.assertTrue(out["net"].startswith("DENIED"), out)

    def test_agent_cannot_read_secrets_or_data(self):
        with tempfile.TemporaryDirectory(dir=sbx.scratch_root()) as tmp:
            episode = Path(tmp)
            work = episode / "work"
            work.mkdir()
            (work / "probe.py").write_text(PROBE, encoding="utf-8")
            out = self._run(sbx.agent_profile(episode, [65000], RUNTIME), episode, work)
        self.assertEqual(out["own"], "ALLOWED")
        self.assertTrue(out["secret"].startswith("DENIED"), out)
        self.assertTrue(out["data"].startswith("DENIED"), out)
        self.assertTrue(out["net"].startswith("DENIED"), out)  # port 9 is not the gateway port


if __name__ == "__main__":
    unittest.main()
