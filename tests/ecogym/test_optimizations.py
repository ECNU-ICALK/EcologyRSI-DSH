"""Unit tests covering P0/P1/P2 architectural and detail optimizations."""

from __future__ import annotations

import gzip
import http.client
import http.server
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ecogym.episodes import sandbox as sbx
from ecogym.episodes.agent import AgentSpec, _matches_session_root, dsh_patch
from ecogym.episodes.export import export
from ecogym.episodes.gateway import Gateway, _ConnPool, _RepetitionBreaker, _read_request_body
from ecogym.episodes.governor import ComputeGovernor, SlotCancelled
from ecogym.episodes.judge import Limits, Run, report_of
from ecogym.episodes.workspace import export_visible
from ecogym.evaluation.harness import Evaluation
from ecogym.evolution.evolver import brief
from ecogym.evolution.loop import Evolution, EvolveConfig
from ecogym.evolution.pareto import crossover_pairs, extract_points, niche_leaders, pareto_front, render_pareto_section
from ecogym.harness import AgentOptions, Harness, HarnessError
from ecogym.kernel.config import Fields
from ecogym.kernel.ledger import Ledger
from ecogym.kernel.secrets import _strip_yaml_scalar
from ecogym.service import Service
from ecogym.solvers import Genome
from ecogym.task.suite import SuiteGates

from .helpers import synthetic_bench, synthetic_frame


class GatewayPoolAndBodyTests(unittest.TestCase):
    def test_read_request_body_handles_chunked_and_rejects_oversize_or_bad_length(self):
        import io

        chunked = io.BytesIO(b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n")
        self.assertEqual(_read_request_body(chunked, {"Transfer-Encoding": "chunked"}, 100), b"hello world")

        with self.assertRaises(ValueError):
            _read_request_body(io.BytesIO(b""), {"Content-Length": "not-a-number"}, 100)
        with self.assertRaises(ValueError):
            _read_request_body(io.BytesIO(b""), {"Content-Length": "-5"}, 100)
        with self.assertRaises(ValueError):
            _read_request_body(io.BytesIO(b"x" * 20), {"Content-Length": "20"}, 10)

    def test_conn_pool_reuses_and_expires_idle_connections(self):
        created = []

        class DummyConn:
            def __init__(self):
                self.sock = object()
                self.closed = False
                created.append(self)

            def close(self):
                self.closed = True
                self.sock = None

        pool = _ConnPool(max_idle=2, ttl=60.0)
        key = ("https", "example.org", 443, "")
        c1, reused1 = pool.acquire(key, DummyConn)
        self.assertFalse(reused1)
        pool.release(key, c1)
        c2, reused2 = pool.acquire(key, DummyConn)
        self.assertTrue(reused2)
        self.assertIs(c1, c2)
        pool.release(key, c2)
        pool.close_all()
        self.assertTrue(c1.closed)
        self.assertEqual(len(created), 1)

    def test_repetition_breaker_catches_degenerate_loop_without_false_positives(self):
        breaker = _RepetitionBreaker()
        # Normal prose and markdown dashed separators must not trigger
        normal = (
            'data: {"choices":[{"delta":{"reasoning_content":"Let us inspect the residual model '
            + "-" * 160 + " and fit Ridge with alpha=1.0." + " " * 160 + '"}}]}\n\n'
        ).encode("utf-8")
        self.assertIsNone(breaker.feed(normal))
        # Degenerate token loop (like pjlab's carecarecare...) triggers promptly
        hit = None
        for _ in range(10):
            chunk = ('data: {"choices":[{"delta":{"reasoning_content":"' + ("care" * 8) + '"}}]}\n\n').encode("utf-8")
            hit = breaker.feed(chunk)
            if hit is not None:
                break
        self.assertIsNotNone(hit)
        self.assertIn("repetition loop", hit)

    def test_aborted_by_close_does_not_inflate_stalled_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            gw = Gateway(key="k", token="t", log_path=Path(tmp) / "calls.jsonl")
            try:
                gw.record({"status": 200, "reply": {"message": {"role": "assistant", "content": "ok"}}}, started=100.0)
                gw.record({"status": 200, "error": "BrokenPipeError: closed", "aborted_by_close": True}, started=120.0)
                self.assertEqual(gw.stalled_seconds(until=600.0), 0.0)
                self.assertFalse(gw.failing)
            finally:
                gw.close()

    def test_route_session_profile_caps_default_max_tokens(self):
        from ecogym.episodes.providers import DEFAULT_ROUTE_MAX_TOKENS, Route

        route = Route("pjlab", "pjlab", "openai-completions", "https://api.example/v1", "K", ("m1", "m2"),
                      profile={"models": [{"id": "m1"}, {"id": "m2", "maxTokens": 16000}]})
        p1 = route.session_profile("m1", "high", "http://127.0.0.1:1/route/pjlab", "RK")
        p2 = route.session_profile("m2", "high", "http://127.0.0.1:1/route/pjlab", "RK")
        self.assertEqual(p1["models"][0]["maxTokens"], DEFAULT_ROUTE_MAX_TOKENS)
        self.assertEqual(p2["models"][0]["maxTokens"], 16000)

    def test_gateway_reuses_keep_alive_connection_over_loopback(self):
        accepted = 0

        class KeepAliveServer(http.server.ThreadingHTTPServer):
            def get_request(self):
                nonlocal accepted
                pair = super().get_request()
                accepted += 1
                return pair

        class KeepAliveHandler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                if length:
                    self.rfile.read(length)
                payload = json.dumps({
                    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        upstream = KeepAliveServer(("127.0.0.1", 0), KeepAliveHandler)
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                log_path = Path(tmp) / "calls.jsonl"
                gw = Gateway(
                    key="real-key", token="tok", log_path=log_path,
                    upstream=f"http://127.0.0.1:{upstream.server_address[1]}",
                ).start()
                try:
                    body = json.dumps({"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]})
                    for _ in range(2):
                        conn = http.client.HTTPConnection("127.0.0.1", gw.port, timeout=5)
                        try:
                            conn.request(
                                "POST", "/deepseek/v1/chat/completions", body=body,
                                headers={"Authorization": "Bearer tok", "Content-Type": "application/json"},
                            )
                            resp = conn.getresponse()
                            self.assertEqual(resp.status, 200)
                            resp.read()
                        finally:
                            conn.close()
                finally:
                    gw.close()
                records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(records), 2)
            self.assertFalse(records[0]["reused_connection"])
            self.assertTrue(records[1]["reused_connection"])
            self.assertEqual(accepted, 1)
        finally:
            upstream.shutdown()
            upstream.server_close()


class ExportEvolverHoldoutTests(unittest.TestCase):
    def test_holdout_of_excludes_evolver_sessions_and_notes_from_overlapping_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "ledger.sqlite3"
            traj_tainted = root / "evo-tainted"
            traj_clean = root / "evo-clean"
            for traj in (traj_tainted, traj_clean):
                traj.mkdir()
                call = {
                    "path": "/v1/chat/completions", "status": 200,
                    "request": {"messages": [{"role": "user", "content": "Review"}]},
                    "reply": {"message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"},
                }
                with gzip.open(traj / "llm_calls.jsonl.gz", "wt", encoding="utf-8") as stream:
                    stream.write(json.dumps(call) + "\n")
            with Ledger(ledger_path) as ledger:
                ledger.append("evo:tainted", "evolve_started", {
                    "suite": {"id": "wide", "train": ["train-a", "held-task"]},
                    "seed": {"digest": "s"}, "config": {},
                })
                ledger.append("evo:tainted", "evolver_note", {
                    "generation": 1, "kind": "lesson", "text": "tainted note",
                })
                ledger.append("evo:tainted", "evolver_finished", {
                    "generation": 1, "status": "finished",
                    "candidates": [{"name": "a", "status": "proposed"}],
                    "trajectory": {"dir": str(traj_tainted)},
                })
                ledger.append("evo:clean", "evolve_started", {
                    "suite": {"id": "small", "train": ["train-a"]},
                    "seed": {"digest": "s"}, "config": {},
                })
                ledger.append("evo:clean", "evolver_note", {"generation": 1, "kind": "lesson", "text": "clean note"})
                ledger.append("evo:clean", "evolver_finished", {
                    "generation": 1, "status": "finished",
                    "candidates": [{"name": "a", "status": "proposed"}],
                    "trajectory": {"dir": str(traj_clean)},
                })

            suite = SimpleNamespace(tasks=lambda side: [SimpleNamespace(id="held-task")] if side == "holdout" else [])
            with patch("ecogym.episodes.export.suite_spec", return_value=suite):
                clean = export(ledger_path, root / "out-clean", evolver=True, holdout_of=["suite@1"])
                audit = export(ledger_path, root / "out-audit", evolver=True, include_failed=True,
                               holdout_of=["suite@1"])

            clean_sft = [json.loads(line) for line in (root / "out-clean" / "sft.jsonl").read_text().splitlines()]
            clean_notes = [json.loads(line) for line in (root / "out-clean" / "notes.jsonl").read_text().splitlines()]
            self.assertEqual([item["meta"]["session"] for item in clean_sft], ["evo:clean/g1"])
            self.assertEqual([item["text"] for item in clean_notes], ["clean note"])
            self.assertEqual(clean["skipped"]["independent validation evolver (holdout_task)"], 1)

            audit_sft = [json.loads(line) for line in (root / "out-audit" / "sft.jsonl").read_text().splitlines()]
            by_session = {item["meta"]["session"]: item["meta"] for item in audit_sft}
            self.assertTrue(by_session["evo:clean/g1"]["passes_filters"])
            self.assertFalse(by_session["evo:tainted/g1"]["passes_filters"])
            self.assertEqual(by_session["evo:tainted/g1"]["independent_validation"], "holdout_task")
            audit_notes = [json.loads(line) for line in (root / "out-audit" / "notes.jsonl").read_text().splitlines()]
            tainted_note = next(n for n in audit_notes if n["text"] == "tainted note")
            self.assertEqual(tainted_note["independent_validation"], "holdout_task")
            self.assertEqual(audit["counts"]["sft"], 2)


class HarnessSpecSecretsAgentTests(unittest.TestCase):
    def test_harness_agent_options_accepts_provider_slash_model(self):
        for model in ("deepseek-v4-flash", "pjlab/glm-5.3", "token-plan/Sage-S2.1-Flash"):
            opts = AgentOptions.read(Fields({"model": model}, "agent"))
            self.assertEqual(opts.model, model)
        for bad in ("bad model", "/leading", "no-slash-non-deepseek", "a/b/c space"):
            with self.assertRaises(HarnessError):
                AgentOptions.read(Fields({"model": bad}, "agent"))

    def test_workspace_spec_includes_structured_local_fit_end_and_weights(self):
        bench = synthetic_bench()
        with tempfile.TemporaryDirectory() as tmp:
            out = export_visible(bench, Path(tmp) / "data")
            spec = json.loads((Path(tmp) / "data" / "spec.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["local_fit_end"], out["boundaries"]["local_fit_end"])
        self.assertEqual(set(spec["target_weights"]), set(bench.task.target_names))

    def test_secrets_strip_inline_comments_without_corrupting_quoted_values(self):
        self.assertEqual(_strip_yaml_scalar("sk-abc123 # prod key"), "sk-abc123")
        self.assertEqual(_strip_yaml_scalar('"sk-abc#123" # comment'), "sk-abc#123")
        self.assertEqual(_strip_yaml_scalar("'sk-single#456'"), "sk-single#456")
        self.assertEqual(_strip_yaml_scalar("sk-no-space#kept"), "sk-no-space#kept")

    def test_matches_session_root_avoids_sibling_prefix_overmatch(self):
        root = "/private/tmp/ecogym/evo-abc-g1-20261128"
        self.assertTrue(_matches_session_root(f"python {root}/work/script.py", root))
        self.assertTrue(_matches_session_root(f"dsh --patch {root}/session.patch.yml", root))
        self.assertFalse(_matches_session_root(f"dsh --patch {root}-discovery/session.patch.yml", root))
        self.assertFalse(_matches_session_root("python /other/path.py", root))

    def test_dsh_patch_routes_web_search_through_gateway_and_matches_dsh_0_2(self):
        gw = SimpleNamespace(
            base_url="http://127.0.0.1:19999/deepseek", has_deepseek=True,
            route_url=lambda p: f"http://127.0.0.1:19999/route/{p}",
        )
        mcp = SimpleNamespace(port=19998, token="tok")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p1 = dsh_patch(AgentSpec(model="deepseek-v4-flash", subagent_depth=1),
                           home=root / "home", skills=root / "skills", gateway=gw, mcp=mcp)
            p0 = dsh_patch(AgentSpec(model="deepseek-v4-flash", subagent_depth=0),
                           home=root / "home", skills=root / "skills", gateway=gw, mcp=mcp)
            for text in (p1, p0):
                self.assertIn("searchProvider: deepseek-official", text)
                self.assertIn("- id: web-deepseek-api\n  disabled: true", text)
                self.assertIn("- id: llm-deepseek-account\n  disabled: true", text)
                self.assertIn("- id: tool-agent-team\n  disabled: true", text)
                self.assertNotIn("backgroundMode", text)
                self.assertNotIn("tool-ralph", text)
                self.assertNotIn("tool-subagent-list-agents", text)
            self.assertIn("maxDepth: 1", p1)
            self.assertIn("- id: tool-subagent\n  disabled: true", p0)

    def test_providers_reads_dsh_0_2_profile_cordis_patch(self):
        from ecogym.episodes import providers

        with tempfile.TemporaryDirectory() as tmp:
            dsh_home = Path(tmp) / "dsh"
            patch_path = dsh_home / "profiles" / "web" / "cordis.patch.yml"
            patch_path.parent.mkdir(parents=True)
            patch_path.write_text(
                "- id: demo-js\n"
                "  config:\n"
                "    expr: !!js '1 + 1'\n"
                "- id: llm-pi-ai\n"
                "  name: '@deepseek-ai/dsh-llm-pi-ai'\n"
                "  config:\n"
                "    providers:\n"
                "      demo-route:\n"
                "        api: openai-completions\n"
                "        baseURL: https://example.invalid/v1\n"
                "        apiKeyEnv: DEMO_API_KEY\n"
                "        models:\n"
                "          - id: demo-model\n"
                "- id: agent-default-model\n"
                "  config:\n"
                "    provider: demo-route\n"
                "    model: demo-model\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"DSH_HOME": str(dsh_home), "ECOGYM_DSH_PROFILE": "web"}):
                providers._CACHE.clear()
                self.assertEqual(providers.settings_path(), patch_path)
                try:
                    loaded = providers.dsh_settings()
                except providers.ModelError:
                    self.skipTest("dsh runtime unavailable for YAML parse")
                self.assertIn("demo-route", loaded["providers"])
                self.assertEqual(loaded["default"], {"provider": "demo-route", "model": "demo-model"})
                providers._CACHE.clear()


class LedgerServiceCacheAndSseTests(unittest.TestCase):
    def test_ledger_batch_reads_service_incremental_cache_pagination_and_sse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.sqlite3"
            service = Service(path)
            with Ledger(path) as ledger:
                for i in range(5):
                    ep = f"ep:{i:04d}"
                    ledger.append(ep, "episode_started", {
                        "task_id": "t1", "harness": {"digest": "h1"}, "model": "m",
                        "context": {"evolution": "evo:1", "generation": i % 2},
                    })
                    ledger.append(ep, "episode_finished", {"status": "finished" if i % 2 == 0 else "timeout"})
                grouped = ledger.events_by_runs(["ep:0000", "ep:0002", "ep:missing"])
                self.assertEqual(len(grouped["ep:0000"]), 2)
                self.assertEqual(len(grouped["ep:0002"]), 2)
                self.assertEqual(grouped["ep:missing"], [])
                since = ledger.events_since(6)
                self.assertEqual([ev.seq for ev in since], [7, 8, 9, 10])

            # First read populates read-model cache; pagination slices newest-first cleanly
            all_eps = service.episodes()
            self.assertEqual(len(all_eps), 5)
            page1 = service.episodes(limit="2", offset="0")
            page2 = service.episodes(limit="2", offset="2")
            self.assertEqual([item["episode"] for item in page1], ["ep:0004", "ep:0003"])
            self.assertEqual([item["episode"] for item in page2], ["ep:0002", "ep:0001"])
            filtered = service.episodes(status="timeout")
            self.assertEqual([item["episode"] for item in filtered], ["ep:0003", "ep:0001"])

            # External append invalidates only touched runs via last_seq
            prev_seq = service._cache_seq
            with Ledger(path) as ledger:
                ledger.append("ep:0005", "episode_started", {
                    "task_id": "t1", "harness": {"digest": "h1"}, "context": {},
                })
                ledger.append("ep:0005", "episode_finished", {"status": "finished"})
            delta_eps = service.episodes(since_seq=str(prev_seq))
            self.assertEqual([item["episode"] for item in delta_eps], ["ep:0005"])

            # SSE stream emits sync then update when ledger advances
            stream = service.events_stream(since=str(prev_seq))
            name1, payload1 = next(stream)
            self.assertEqual(name1, "sync")
            self.assertGreater(payload1["seq"], prev_seq)
            name2, payload2 = next(stream)
            self.assertEqual(name2, "update")
            self.assertIn("ep:0005", payload2["runs"])
            service._closing.set()


class GovernorAndSandboxTests(unittest.TestCase):
    def test_compute_governor_enforces_limit_and_cancellation(self):
        gov = ComputeGovernor(limit=1)
        active = 0
        max_seen = 0
        lock = threading.Lock()

        def worker():
            nonlocal active, max_seen
            with gov.slot():
                with lock:
                    active += 1
                    max_seen = max(max_seen, active)
                time.sleep(0.03)
                with lock:
                    active -= 1

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(max_seen, 1)
        self.assertEqual(gov.peak, 1)

        cancel = threading.Event()
        with gov.slot():
            cancel.set()
            with self.assertRaises(SlotCancelled):
                with gov.slot(cancel=cancel):
                    pass

    def test_bubblewrap_backend_builds_unshared_network_and_binds(self):
        backend = sbx.BubblewrapBackend()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "prog"
            work.mkdir()
            args = backend._base_args([Path(tmp)], [work], unshare_net=True)
        self.assertIn("--unshare-net", args)
        self.assertIn("--unshare-pid", args)
        self.assertIn("--bind", args)
        self.assertIn(str(work.resolve()), args)


class ParetoFastTrackAndPruneTests(unittest.TestCase):
    def test_pareto_frontier_niche_leaders_and_crossover_pairs(self):
        by_label = {
            "g1a": {
                "status": "not_adopted", "parent": "inc",
                "decision": {
                    "delta": 0.012,
                    "tasks": [{"delta": 0.01}, {"delta": 0.014}],
                    "horizons": {"1": 0.025, "24": -0.002},
                    "groups": {"ds1": {"delta": 0.02}, "ds2": {"delta": 0.004}},
                },
            },
            "g1b": {
                "status": "not_adopted", "parent": "inc",
                "decision": {
                    "delta": 0.010,
                    "tasks": [{"delta": 0.01}, {"delta": 0.01}],
                    "horizons": {"1": 0.001, "24": 0.022},
                    "groups": {"ds1": {"delta": 0.002}, "ds2": {"delta": 0.018}},
                },
            },
            "g1c": {
                "status": "not_adopted", "parent": "inc",
                "decision": {
                    "delta": -0.005,
                    "tasks": [{"delta": -0.005}, {"delta": -0.005}],
                    "horizons": {"1": -0.005, "24": -0.005},
                    "groups": {"ds1": {"delta": -0.005}, "ds2": {"delta": -0.005}},
                },
            },
        }
        digests = {"g1a": "aaa", "g1b": "bbb", "g1c": "ccc"}
        points = extract_points(by_label, digests)
        front = pareto_front(points)
        self.assertEqual({pt.label for pt in front}, {"g1a", "g1b"})
        leaders = niche_leaders(points)
        self.assertEqual(leaders["h=1"].label, "g1a")
        self.assertEqual(leaders["h=24"].label, "g1b")
        pairs = crossover_pairs(points)
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0]["a"], pairs[0]["b"]}, {"g1a", "g1b"})
        rendered = render_pareto_section(by_label, digests)
        self.assertIn("非支配帕累托前沿", rendered)
        self.assertIn("推荐交叉重组父本对", rendered)

    def test_evolve_brief_lists_max_horizon_loss_gate(self):
        bench = synthetic_bench()
        suite = SimpleNamespace(
            id="s", title="s", train=[bench.task.id], holdout=[],
            gates=SuiteGates(min_delta=0.005, confidence=0.95, max_horizon_loss=((24, 0.01),)),
        )
        inc = SimpleNamespace(digest="a" * 64, agent=AgentOptions())
        text = brief(
            generation=1, suite=suite, tasks=[bench.task], benches={bench.task.id: bench},
            names=["a"], repeats=2, episode_minutes=10, episode_tokens=1000, model=None,
            incumbent=inc, limits=Limits(), forbidden=[], minutes=10, tokens=1000, deadline=time.time() + 600,
        )
        self.assertIn("关键时距平均损失不超过上限", text)
        self.assertIn("h=24 ≤ 0.01", text)
        self.assertIn("<提供方>/<模型>", text)

    def test_early_prune_skips_replicate_two_for_dominated_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.sqlite3"
            tasks = [SimpleNamespace(id=f"t{i}", dataset="ds") for i in range(4)]
            suite = SimpleNamespace(
                id="s", title="s", train=[t.id for t in tasks], holdout=[], digest="sd",
                gates=SuiteGates(min_delta=0.005, confidence=0.95, test="t"),
                tasks=lambda _: tasks,
            )
            config = EvolveConfig(suite="s", repeats=2, concurrency=2, early_prune=True)
            with patch("ecogym.evolution.loop.suite_spec", return_value=suite):
                evo = Evolution(config, path)

            calls = []

            def fake_episode(run_id, generation, harness, role, name, task, replicate, snapshot, plan_id,
                             first_attempt):
                calls.append((role, name, task.id, replicate))
                score = 0.20 if role == "incumbent" else (0.10 if name == "bad" else 0.25)
                return {
                    "episode": f"ep:{role}:{name}:{task.id}:{replicate}",
                    "task": task.id, "harness": harness.digest, "role": role, "candidate": name,
                    "replicate": replicate, "status": "finished", "score": score, "predictions": "p.npz",
                }

            inc = SimpleNamespace(digest="inc")
            good = SimpleNamespace(digest="good")
            bad = SimpleNamespace(digest="bad")
            jobs = [(inc, "incumbent", None), (good, "candidate", "good"), (bad, "candidate", "bad")]
            snap = SimpleNamespace(guidance={t.id: None for t in tasks})
            with patch.object(evo, "_episode", side_effect=fake_episode):
                entries = evo._episodes("evo:r", 1, jobs, snap, "plan1")
            # Wave 1 ran all 12 slots; Wave 2 ran incumbent (4) + good (4) = 8, skipping bad (4)!
            self.assertEqual(len(calls), 20)
            self.assertFalse(any(name == "bad" and rep == 1 for _, name, _, rep in calls))
            pruned_entries = [e for e in entries if e.get("status") == "pruned"]
            self.assertEqual(len(pruned_entries), 4)
            self.assertTrue(all(e["candidate"] == "bad" for e in pruned_entries))

    def test_template_fast_track_evaluates_template_only_round_without_llm_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "ledger.sqlite3"
            bench = synthetic_bench(frame=synthetic_frame(days=40))
            stage = bench.stage("model_selection")
            suite = SimpleNamespace(
                id="s", title="s", train=[bench.task.id], holdout=[], digest="sd",
                gates=SuiteGates(min_delta=0.005, confidence=0.95, test="t"),
                tasks=lambda _: [bench.task],
            )
            # Build two minimal harness directories on disk
            for folder, val in (("inc", 0.0), ("cand", 1.0)):
                hdir = root / folder
                (hdir / "template").mkdir(parents=True)
                (hdir / "harness.toml").write_text(f'name = "{folder}"\n[agent]\nmodel = "deepseek-v4-flash"\n')
                (hdir / "instructions.md").write_text("hi\n")
                (hdir / "template" / "predictor.py").write_text(f"# v={val}\nclass Predictor: pass\n")
            inc_h = Harness.load(root / "inc")
            cand_h = Harness.load(root / "cand")
            config = EvolveConfig(suite="s", repeats=2, concurrency=1, template_fast_track=True)
            with patch("ecogym.evolution.loop.suite_spec", return_value=suite):
                evo = Evolution(config, ledger_path)
            evo.benches[bench.task.id] = bench
            with Ledger(ledger_path) as ledger:
                ledger.append("evo:ft", "evolve_started", {"seed": {"digest": inc_h.digest}, "suite": {"digest": "sd"},
                                                           "config": config.to_dict()})
                ledger.append("evo:ft", "round_started", {"generation": 0, "incumbent": inc_h.digest})
                ledger.append("evo:ft", "round_finished", {"incumbent": inc_h.digest, "outcome": "baseline"})

            inc_eval = Evaluation(Genome("inc"), stage, stage.baseline.copy(), np.zeros(len(stage.cells), dtype=int))
            cand_eval = Evaluation(Genome("cand"), stage, stage.observed.copy(), np.zeros(len(stage.cells), dtype=int))

            def fake_judge(b, program_dir, stage_name, runtime, **kwargs):
                text = (Path(program_dir) / "predictor.py").read_text()
                ev = cand_eval if "v=1.0" in text else inc_eval
                return report_of(b, ev, Run(np.empty(0), "ok", [], 0.05, 0.1, "")), ev

            harnesses = {inc_h.digest: inc_h, cand_h.digest: cand_h}
            proposed = [{"name": "a", "digest": cand_h.digest, "harness": cand_h, "template_only": True}]
            with (
                patch("ecogym.evolution.loop.resolve", side_effect=harnesses.get),
                patch("ecogym.evolution.loop.discover_runtime", return_value=object()),
                patch("ecogym.evolution.loop.judge_program", side_effect=fake_judge),
                patch.dict(os.environ, {"ECOGYM_HOME": str(root / "home")}),
                patch.object(evo, "_measurement", return_value="m1"),
                patch.object(evo, "_propose", return_value=proposed),
                patch.object(evo, "_episodes", side_effect=AssertionError("fast track must not run LLM episodes")),
                patch.object(evo, "_record_template_attribution"),
            ):
                view = evo.view("evo:ft")
                evo._round("evo:ft", view, evo.state(view))

            final_view = evo.view("evo:ft")
            self.assertEqual(final_view["incumbent"], cand_h.digest)
            self.assertEqual(final_view["rounds"][-1]["outcome"], "adopted")


class RunnerAndBacktestTests(unittest.TestCase):
    @staticmethod
    def _pandas_python() -> str | None:
        if importlib.util.find_spec("pandas") is not None:
            return sys.executable
        try:
            candidate = sbx.runtime_python()
            if candidate.exists():
                return str(candidate)
        except OSError:
            return None
        return None

    def test_predictor_runner_step_fast_path_and_backtest_bounds_weights(self):
        python = self._pandas_python()
        if python is None:
            self.skipTest("pandas runtime not available")
        root = Path(__file__).resolve().parents[2]
        runner = root / "src" / "ecogym" / "episodes" / "predictor_runner.py"
        backtest_dir = root / "src" / "ecogym" / "harness" / "seed" / "template"
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "predictor.py").write_text(
                "class Predictor:\n"
                "    def __init__(self, spec):\n"
                "        self.spec = spec\n"
                "        self.step_calls = 0\n"
                "    def fit(self, history):\n"
                "        self.cols = list(history.columns)\n"
                "    def forecast_step(self, ts, row):\n"
                "        self.step_calls += 1\n"
                "        return {t: [row[t] + 100.0 for _ in self.spec['horizons']] for t in self.spec['targets']}\n"
                "    def forecast(self, history):\n"
                "        raise AssertionError('forecast_step fast path should be used')\n",
                encoding="utf-8",
            )
            messages = [
                {
                    "op": "start",
                    "spec": {"targets": ["y1", "y2"], "horizons": [1, 2]},
                    "columns": ["y1", "y2"],
                    "time": ["2020-01-01T00:00:00", "2020-01-01T01:00:00"],
                    "rows": [[1.0, 2.0], [1.5, 2.5]],
                    "stream": 2,
                },
                {"op": "step", "time": "2020-01-01T02:00:00", "values": [2.0, 3.0]},
                {"op": "row", "time": "2020-01-01T03:00:00", "values": [3.0, 4.0]},
                {"op": "forecast"},
                {"op": "end"},
            ]
            proc = subprocess.run(
                [python, str(runner)],
                input="\n".join(json.dumps(m) for m in messages) + "\n",
                cwd=work, capture_output=True, text=True, check=True, timeout=30,
            )
            replies = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
            self.assertEqual([r["op"] for r in replies], ["ready", "forecast", "forecast"])
            self.assertEqual(replies[1]["values"], {"y1": [102.0, 102.0], "y2": [103.0, 103.0]})
            self.assertEqual(replies[2]["values"], {"y1": [103.0, 103.0], "y2": [104.0, 104.0]})

            probe = (
                f"import json, sys, numpy as np, pandas as pd\n"
                f"sys.path.insert(0, {str(backtest_dir)!r})\n"
                f"from ecotk import backtest\n"
                f"idx = pd.date_range('2020-01-01', periods=20, freq='1h')\n"
                f"df = pd.DataFrame({{'y1': np.linspace(1.0, 2.0, 20), 'y2': np.linspace(5.0, 6.0, 20)}}, index=idx)\n"
                f"spec = {{\n"
                f"    'targets': ['y1', 'y2'], 'horizons': [1], 'season_steps': 24, 'history_steps': 1,\n"
                f"    'baselines': {{'y1@1': 'persistence', 'y2@1': 'persistence'}},\n"
                f"    'bounds': {{'y1': [0.0, 10.0], 'y2': [0.0, 10.0]}},\n"
                f"    'target_weights': {{'y1': 3.0, 'y2': 1.0}},\n"
                f"    'local_fit_end': '2020-01-01T16:00:00',\n"
                f"}}\n"
                f"class ClippedPred:\n"
                f"    def __init__(self, spec): self.spec = spec; self.used_step = 0\n"
                f"    def fit(self, h): pass\n"
                f"    def forecast_step(self, ts, row):\n"
                f"        self.used_step += 1\n"
                f"        return {{'y1': [999.0], 'y2': [row['y2']]}}\n"
                f"    def forecast(self, h): raise AssertionError('expected forecast_step')\n"
                f"res = backtest.run(ClippedPred, spec, df, '2020-01-01T10:00:00', '2020-01-01T16:00:00')\n"
                f"fit_end = backtest._detect_fit_end(spec)\n"
                f"print(json.dumps({{'score': res['score'], 'rmse_y1': res['cells']['y1@1']['rmse'], "
                f"'fit_end': str(fit_end)}}))\n"
            )
            out = subprocess.run([python, "-c", probe], capture_output=True, text=True, check=True, timeout=30)
            parsed = json.loads(out.stdout.strip())
            self.assertEqual(parsed["fit_end"], "2020-01-01 16:00:00")
            # y1 prediction 999.0 was clipped to bound 10.0 (so RMSE < 10.0 instead of ~997)
            self.assertLess(parsed["rmse_y1"], 10.0)


class BatchThreeRefactorTests(unittest.TestCase):
    def test_start_cache_shares_healthy_local_and_pre_evals_across_replicates(self):
        from ecogym.episodes.tools import EpisodeTools, StartCache, starting_program

        bench = synthetic_bench(frame=synthetic_frame(days=40))
        stage = bench.stage("calibration_uq")
        healthy_eval = Evaluation(Genome("ok"), stage, stage.observed.copy(), np.zeros(len(stage.cells), dtype=int))
        healthy_rep = report_of(bench, healthy_eval, Run(np.empty(0), "ok", [], 0.05, 0.1, ""))
        bad_rep = {**healthy_rep, "status": "error"}

        cache = StartCache()
        limits = Limits()
        calls: list[str] = []

        def make_ws(base: Path, code: str) -> Path:
            sub = base / "work" / "submission"
            sub.mkdir(parents=True)
            (sub / "predictor.py").write_text(code, encoding="utf-8")
            return base

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws1 = make_ws(root / "ep1", "class Predictor: pass\n")
            ws2 = make_ws(root / "ep2", "class Predictor: pass\n")
            ws_bad = make_ws(root / "ep_bad", "class Predictor: bad = 1\n")

            def fake_judge(b, program_dir, stage_name, runtime, **kwargs):
                calls.append(stage_name)
                if "bad = 1" in (Path(program_dir) / "predictor.py").read_text(encoding="utf-8"):
                    return bad_rep, healthy_eval
                return healthy_rep, healthy_eval

            with patch("ecogym.episodes.tools.judge_program", side_effect=fake_judge):
                s1 = starting_program(bench, ws1, object(), limits, cache=cache)
                s2 = starting_program(bench, ws2, object(), limits, cache=cache)
                self.assertEqual(calls, ["calibration_uq"])
                self.assertEqual((cache.misses, cache.hits), (1, 1))
                self.assertEqual(s1["local"]["score"], s2["local"]["score"])

                # Unhealthy start is NOT cached, so a retry invokes judge_program again
                sb1 = starting_program(bench, ws_bad, object(), limits, cache=cache)
                sb2 = starting_program(bench, make_ws(root / "ep_bad2", "class Predictor: bad = 1\n"),
                                       object(), limits, cache=cache)
                self.assertFalse(sb1["healthy"])
                self.assertFalse(sb2["healthy"])
                self.assertEqual(calls.count("calibration_uq"), 3)

                # PRE_LOCAL_STAGE evaluation cached across two EpisodeTools sharing StartCache
                t1 = EpisodeTools(bench=bench, root=ws1, runtime=object(), limits=limits, deadline=time.time() + 60,
                                  usage=lambda: {}, event=lambda *_: None, start=s1, start_cache=cache)
                t2 = EpisodeTools(bench=bench, root=ws2, runtime=object(), limits=limits, deadline=time.time() + 60,
                                  usage=lambda: {}, event=lambda *_: None, start=s2, start_cache=cache)
                pre1 = t1._start_pre_evaluation()
                pre2 = t2._start_pre_evaluation()
                self.assertIs(pre1, pre2)
                self.assertEqual(calls.count("calibration_pre"), 1)

    def test_follow_up_stops_early_on_diminishing_returns(self):
        from ecogym.episodes.runner import _follow_up
        from ecogym.kernel import clock

        class DummyTools:
            selection_policy = "best_local/1"

            def __init__(self) -> None:
                self.evaluations = 1
                self.submissions = [{"n": 1}]
                self.score: float | None = 0.20
                self.has_final = True

            def best_local(self):
                return {"local": {"score": self.score}} if self.score is not None else None

            def final(self):
                return {"n": 1} if self.has_final else None

            def scored_text(self) -> str:
                return "第 1 次提交"

        tools = DummyTools()
        ask = _follow_up(tools, deadline=clock.now() + 3600.0, minutes=30.0)
        # Continuation 1 proceeds and records baseline best_local=0.20
        self.assertIsNotNone(ask(1, "done part 1"))
        # Continuation 2 with new work AND improved score (0.25 > 0.20) continues
        tools.evaluations = 2
        tools.score = 0.25
        self.assertIsNotNone(ask(2, "done part 2"))
        # Continuation 3 with new work BUT no improvement (still 0.25) triggers stop-loss
        tools.evaluations = 3
        self.assertIsNone(ask(3, "done part 3"))

        # Without a passing final submission yet, continuation 2 is NOT stopped early
        tools2 = DummyTools()
        tools2.has_final = False
        ask2 = _follow_up(tools2, deadline=clock.now() + 3600.0, minutes=30.0)
        self.assertIsNotNone(ask2(1, "part 1"))
        tools2.evaluations = 2
        self.assertIsNotNone(ask2(2, "part 2"))

    def test_gateway_consecutive_degenerate_loops_graceful_sse_stop_and_evidence(self):
        from ecogym.evolution.evidence import episode_summary

        class LoopSSEHandler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):
                return

            def do_POST(self):
                self.close_connection = True
                length = int(self.headers.get("Content-Length", "0"))
                if length:
                    self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                chunk = ('data: {"choices":[{"delta":{"reasoning_content":"' + ("care" * 8) + '"}}]}\n\n').encode()
                try:
                    for _ in range(15):
                        self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except OSError:
                    pass

        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), LoopSSEHandler)
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                log_path = Path(tmp) / "calls.jsonl"
                gw = Gateway(
                    key="k", token="tok", log_path=log_path,
                    upstream=f"http://127.0.0.1:{upstream.server_address[1]}",
                ).start()
                try:
                    body = json.dumps({
                        "model": "deepseek-v4-flash", "stream": True,
                        "messages": [{"role": "user", "content": "hi"}],
                    })
                    # 1st degenerate loop: stream cut for retry
                    conn1 = http.client.HTTPConnection("127.0.0.1", gw.port, timeout=5)
                    try:
                        conn1.request("POST", "/deepseek/v1/chat/completions", body=body,
                                      headers={"Authorization": "Bearer tok", "Content-Type": "application/json"})
                        resp1 = conn1.getresponse()
                        try:
                            resp1.read()
                        except http.client.IncompleteRead:
                            pass
                    finally:
                        conn1.close()
                    self.assertEqual(gw.degenerate_aborts, 1)

                    # 2nd consecutive degenerate loop: graceful SSE finish_reason="length" + [DONE]
                    conn2 = http.client.HTTPConnection("127.0.0.1", gw.port, timeout=5)
                    try:
                        conn2.request("POST", "/deepseek/v1/chat/completions", body=body,
                                      headers={"Authorization": "Bearer tok", "Content-Type": "application/json"})
                        resp2 = conn2.getresponse()
                        raw2 = resp2.read().decode("utf-8", "replace")
                    finally:
                        conn2.close()
                    self.assertEqual(resp2.status, 200)
                    self.assertIn('"finish_reason":"length"', raw2)
                    self.assertIn("data: [DONE]", raw2)
                    self.assertEqual(gw.degenerate_aborts, 2)
                    self.assertEqual(gw.answered, 1)
                finally:
                    gw.close()
                entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
                self.assertEqual(len(entries), 2)
                self.assertTrue(entries[0]["degenerate_loop"])
                self.assertTrue(entries[1]["degenerate_loop"])
                self.assertTrue(entries[1]["degenerate_graceful_stop"])
        finally:
            upstream.shutdown()
            upstream.server_close()

        summary_text = episode_summary({
            "episode": "ep:deg", "task_id": "t1", "status": "finished",
            "harness": {"name": "seed", "digest": "a" * 20},
            "model": "m", "minutes": 10, "tokens": 1000, "evaluations": [], "submissions": [],
            "notes": [], "questions": [],
            "finished": {"status": "finished", "seconds": 12.0, "tokens": 500, "llm_calls": 2,
                         "degenerate_aborts": 2, "activity": {}},
        }, role="候选 a")
        self.assertIn("复读熔断 ×2", summary_text)

    def test_service_evolution_cache_hit_and_broadcaster_fanout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.sqlite3"
            with Ledger(path) as ledger:
                ledger.append("evo:c1", "evolve_started", {
                    "suite": {"id": "s", "train": ["t1"]}, "seed": {"digest": "seed"}, "config": {"generations": 1},
                })
                ledger.append("evo:c1", "round_started", {"generation": 0, "incumbent": "seed"})
                ledger.append("ep:e1", "episode_started", {
                    "task_id": "t1", "harness": {"digest": "seed"}, "model": "m",
                    "context": {"evolution": "evo:c1", "generation": 0, "role": "incumbent"},
                })
                ledger.append("ep:e1", "episode_scored", {
                    "score": 0.12, "references": {"expert": 0.15, "naive": 0.0},
                })
                ledger.append("ep:e1", "episode_finished", {"status": "finished", "tokens": 100})
                ledger.append("evo:c1", "episodes_finished", {
                    "generation": 0,
                    "episodes": [{"episode": "ep:e1", "task": "t1", "harness": "seed", "role": "incumbent",
                                  "status": "finished", "score": 0.12}],
                })
                ledger.append("evo:c1", "round_finished", {"generation": 0, "incumbent": "seed", "outcome": "baseline"})

            service = Service(path)
            try:
                first = service.evolution("evo:c1")
                self.assertEqual(first["task_references"]["t1"]["expert"], 0.15)
                # Repeated read hits incremental read-model cache without calling Ledger.events
                with patch.object(Ledger, "events", side_effect=AssertionError("should use cache")):
                    second = service.evolution("evo:c1")
                self.assertEqual(second["task_references"]["t1"]["expert"], 0.15)
                self.assertEqual(second["ancestry"], [])

                # Two concurrent SSE subscribers both receive an update from the shared broadcaster
                s1 = service.events_stream()
                s2 = service.events_stream()
                self.assertEqual(next(s1)[0], "sync")
                self.assertEqual(next(s2)[0], "sync")
                received: list[tuple[str, dict]] = []

                def pull(gen):
                    received.append(next(gen))

                t1 = threading.Thread(target=pull, args=(s1,))
                t2 = threading.Thread(target=pull, args=(s2,))
                t1.start()
                t2.start()
                time.sleep(0.05)
                with Ledger(path) as ledger:
                    ledger.append("ep:e2", "episode_started", {
                        "task_id": "t1", "harness": {"digest": "seed"}, "context": {},
                    })
                t1.join(timeout=5)
                t2.join(timeout=5)
                self.assertEqual(len(received), 2)
                self.assertTrue(all(ev == "update" and "ep:e2" in payload["runs"] for ev, payload in received))
                s1.close()
                s2.close()
            finally:
                service.close()

    def test_copy_regular_tree_and_horizon_scores(self):
        from ecogym.evaluation.suite import report_horizon_scores
        from ecogym.kernel.fs import CopyError, copy_regular_tree

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            (src / "sub").mkdir(parents=True)
            (src / "a.py").write_text("x = 1\n", encoding="utf-8")
            (src / "sub" / "b.py").write_text("y = 2\n", encoding="utf-8")
            dst = root / "dst"
            digest = copy_regular_tree(src, dst, max_files=10, max_bytes=1024, hash_header=b"test/1\n")
            self.assertEqual(len(digest), 64)
            self.assertEqual((dst / "sub" / "b.py").read_text(encoding="utf-8"), "y = 2\n")

            with self.assertRaises(CopyError):
                copy_regular_tree(src, root / "dst_small", max_files=1, max_bytes=1024)
            with self.assertRaises(CopyError):
                copy_regular_tree(src, root / "dst_bytes", max_files=10, max_bytes=4)
            (src / "link.py").symlink_to(src / "a.py")
            with self.assertRaises(CopyError):
                copy_regular_tree(src, root / "dst_sym", max_files=10, max_bytes=1024)

        bench = synthetic_bench(frame=synthetic_frame(days=40))
        stage = bench.stage("model_selection")
        ev = Evaluation(Genome("g"), stage, stage.observed.copy(), np.zeros(len(stage.cells), dtype=int))
        h_ev = ev.horizon_scores()
        self.assertTrue(h_ev)
        self.assertTrue(all(abs(v - 1.0) < 1e-6 for v in h_ev.values()))
        rep = report_of(bench, ev, Run(np.empty(0), "ok", [], 0.01, 0.02, ""))
        h_rep = report_horizon_scores(rep)
        self.assertEqual(set(h_ev), set(h_rep))
        for horizon in h_ev:
            self.assertAlmostEqual(h_ev[horizon], h_rep[horizon], places=6)


class HarnessSelfEvolutionTests(unittest.TestCase):
    def test_workspace_data_profile_and_baseline_weak_cells(self):
        bench = synthetic_bench(frame=synthetic_frame(days=40))
        start_report = {
            "status": "ok",
            "score": 0.085,
            "cells": [
                {"cell": f"{bench.task.target_names[0]}@1", "skill": 0.14, "rmse": 0.86, "baseline_rmse": 1.0},
                {"cell": f"{bench.task.target_names[0]}@6", "skill": -0.02, "rmse": 1.02, "baseline_rmse": 1.0},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            export_visible(bench, data_dir, start_report=start_report)
            profile = (data_dir / "profile.md").read_text(encoding="utf-8")
            readme = (data_dir / "README.md").read_text(encoding="utf-8")
        self.assertIn("数据与基线画像", profile)
        self.assertIn("Lag-1 自相关", profile)
        self.assertIn("起始模板本地检验短板", profile)
        self.assertIn("最弱计分格", profile)
        self.assertIn(f"`{bench.task.target_names[0]}@6`", profile)
        self.assertIn("profile.md", readme)

    def test_budget_rescue_audit_hooks_and_bloat_checks(self):
        from ecogym.episodes.sandbox import available as sandbox_available, discover_runtime
        from ecogym.episodes.tools import EpisodeTools, builtin_submission_audit, run_audit_hook
        from ecogym.evolution.evolver import bloat_warnings, hook_problems
        from ecogym.harness import store
        from ecogym.kernel import clock

        # 1. Built-in submission audit catches identical code, zero skill, degraded horizon, and pre-fold overfit
        msgs = builtin_submission_audit(
            {"status": "ok", "cells": {"y@1": {"skill": 0.0}}},
            {"delta": 0.05, "pre_delta": -0.02, "uq_horizons": {"1": 0.02, "24": -0.05}},
            start_digest="abc",
            digest="abc",
        )
        self.assertEqual(len(msgs), 4)

        # 2. Budget-rescue pacing notice reads custom hooks/rescue.md and flips rescue_triggered
        bench = synthetic_bench(frame=synthetic_frame(days=30))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hooks_dir = root / "harness" / "hooks"
            hooks_dir.mkdir(parents=True)
            (hooks_dir / "rescue.md").write_text("自定义救援提示：立即提交最优候选。", encoding="utf-8")
            now = clock.now()
            # Plenty of time left -> no rescue
            tools_ok = EpisodeTools(
                bench=bench, root=root, runtime=None, limits=Limits(), deadline=now + 600,
                usage=lambda: {}, event=lambda *_: None, budget_minutes=20.0,
            )
            self.assertIsNone(tools_ok._pacing_notice())
            self.assertFalse(tools_ok.rescue_triggered)
            # Tight deadline (< min(300s, 25% of 20m = 300s)) -> triggers rescue
            tools_tight = EpisodeTools(
                bench=bench, root=root, runtime=None, limits=Limits(), deadline=now + 120,
                usage=lambda: {}, event=lambda *_: None, budget_minutes=20.0,
            )
            notice = tools_tight._pacing_notice()
            self.assertIsNotNone(notice)
            self.assertIn("自定义救援提示", notice)
            self.assertTrue(tools_tight.rescue_triggered)
            # Once a replacing submission exists -> rescue no longer fires
            tools_tight.submissions.append({
                "n": 1, "accepted": True, "replaces_start": True,
                "local": {"score": 0.2},
                "local_selection": {"passed": True, "delta": 0.05, "candidate_score": 0.2},
            })
            self.assertIsNone(tools_tight._pacing_notice())

            # 3. Anti-bloat warnings + Harness.install copies hooks/ + hook_problems static/sandbox checks
            staging = root / "cand"
            (staging / "template").mkdir(parents=True)
            (staging / "memory").mkdir(parents=True)
            (staging / "hooks").mkdir(parents=True)
            (staging / "harness.toml").write_text(
                'name = "test"\nsummary = "hook test"\n[agent]\nmodel = "deepseek-v4-flash"\n',
                encoding="utf-8",
            )
            (staging / "template" / "predictor.py").write_text("class Predictor:\n    pass\n", encoding="utf-8")
            (staging / "instructions.md").write_text("x" * 15_000, encoding="utf-8")
            (staging / "memory" / "lessons.md").write_text("- keep concise\n", encoding="utf-8")
            for idx in range(5):
                sdir = staging / "skills" / f"skill_{idx}"
                sdir.mkdir(parents=True)
                (sdir / "SKILL.md").write_text(f"---\nname: skill_{idx}\ndescription: test\n---\n", encoding="utf-8")
            (staging / "hooks" / "audit_submission.py").write_text(
                "def audit(summary, selection, files):\n"
                "    return ['custom audit triggered'] if summary.get('score', 0) <= 0 else []\n",
                encoding="utf-8",
            )
            (staging / "hooks" / "rescue.md").write_text("rescue text\n", encoding="utf-8")
            b_warns = bloat_warnings(staging)
            self.assertEqual(len(b_warns), 2)

            harness = Harness.load(staging)
            ep_root = root / "ep"
            (ep_root / "work").mkdir(parents=True)
            harness.install(ep_root)
            self.assertTrue((ep_root / "harness" / "hooks" / "audit_submission.py").is_file())
            self.assertTrue((ep_root / "harness" / "hooks" / "rescue.md").is_file())

            # Bad syntax hook is rejected statically
            bad_staging = root / "bad_cand"
            (bad_staging / "hooks").mkdir(parents=True)
            (bad_staging / "hooks" / "audit_submission.py").write_text("def audit(:\n", encoding="utf-8")
            self.assertTrue(any("语法错误" in p for p in hook_problems(bad_staging, None)))

            # Missing audit() entrypoint is rejected statically
            no_fn = root / "nofn_cand"
            (no_fn / "hooks").mkdir(parents=True)
            (no_fn / "hooks" / "audit_submission.py").write_text("x = 1\n", encoding="utf-8")
            self.assertTrue(any("audit(" in p for p in hook_problems(no_fn, None)))

            if sandbox_available():
                rt = discover_runtime()
                warns, err = run_audit_hook(
                    staging / "hooks" / "audit_submission.py", rt,
                    summary={"status": "ok", "score": 0.0, "cells": {}},
                    selection={"delta": 0.0},
                )
                self.assertIsNone(err)
                self.assertEqual(warns, ["custom audit triggered"])
                self.assertEqual(hook_problems(staging, rt), [])
                # Runtime error inside sandbox is caught by hook_problems
                boom = root / "boom_cand"
                (boom / "hooks").mkdir(parents=True)
                (boom / "hooks" / "audit_submission.py").write_text(
                    "def audit(summary, selection, files):\n    raise ValueError('boom')\n",
                    encoding="utf-8",
                )
                self.assertTrue(any("沙箱冒烟测试失败" in p for p in hook_problems(boom, rt)))

                # ACT hook_fired stays False when hook returns [], and flips True only when warning emitted
                ep_tools = EpisodeTools(
                    bench=bench, root=ep_root, runtime=rt, limits=Limits(), deadline=now + 600,
                    usage=lambda: {}, event=lambda *_: None, budget_minutes=20.0,
                )
                w_quiet, err_q = ep_tools._audit_submission(
                    staging / "template", "d1", {"status": "ok", "score": 0.2, "cells": {}}, {"delta": 0.05}
                )
                self.assertEqual((w_quiet, err_q), ([], None))
                self.assertFalse(ep_tools.hook_fired)
                w_fired, err_f = ep_tools._audit_submission(
                    staging / "template", "d1", {"status": "ok", "score": 0.0, "cells": {}}, {"delta": 0.0}
                )
                self.assertIn("custom audit triggered", w_fired)
                self.assertIsNone(err_f)
                self.assertTrue(ep_tools.hook_fired)
            _ = store  # keep import clean

    def test_zero_llm_triage_act_and_hda_metrics(self):
        from ecogym.evolution.evidence import triage

        with tempfile.TemporaryDirectory() as tmp:
            h_root = Path(tmp) / "h_inc"
            (h_root / "skills" / "active_skill").mkdir(parents=True)
            (h_root / "skills" / "unused_skill").mkdir(parents=True)
            inc_h = SimpleNamespace(digest="inc_digest_1234", root=h_root)

            view = {
                "incumbent": "inc_digest_1234",
                "rounds": [{
                    "generation": 1,
                    "finished": True,
                    "candidates": [{
                        "name": "a", "digest": "cand_digest_5678",
                        "decision": {"delta": 0.005},
                    }],
                    "episodes": [
                        {"episode": "ep:1", "task": "t1", "harness": "inc_digest_1234", "status": "timeout"},
                        {"episode": "ep:2", "task": "t1", "harness": "inc_digest_1234", "status": "finished"},
                        {"episode": "ep:3", "task": "t1", "harness": "cand_digest_5678", "status": "finished"},
                    ],
                }],
            }
            records = {
                "ep:1": {
                    "episode": "ep:1", "task_id": "t1", "status": "timeout",
                    "start": {"local": {"score": 0.05}},
                    "evaluations": [{"local": {"score": 0.14}}],
                    "submissions": [],
                    "scored": {
                        "score": 0.05, "start": True,
                        "cells": [{"cell": "y@1", "skill": 0.10}, {"cell": "y@24", "skill": -0.04}],
                    },
                    "finished": {
                        "tokens": 100_000, "seconds": 300,
                        "rescue_triggered": True, "hook_fired": False, "final_text": "",
                    },
                    "notes": [],
                },
                "ep:2": {
                    "episode": "ep:2", "task_id": "t1", "status": "finished",
                    "start": {"local": {"score": 0.05}},
                    "evaluations": [],
                    "submissions": [{"n": 1, "local": {"score": 0.22}}],
                    "scored": {
                        "score": 0.12, "n": 1, "start": False,
                        "cells": [{"cell": "y@1", "skill": 0.18}, {"cell": "y@24", "skill": 0.01}],
                    },
                    "finished": {
                        "tokens": 100_000, "seconds": 250,
                        "rescue_triggered": False, "hook_fired": True, "final_text": "used active_skill",
                    },
                    "notes": [],
                },
                "ep:3": {
                    "episode": "ep:3", "task_id": "t1", "status": "finished",
                    "start": {"local": {"score": 0.05}},
                    "evaluations": [],
                    "submissions": [{"n": 1, "local": {"score": 0.11}}],
                    "scored": {
                        "score": 0.09, "n": 1, "start": False,
                        "cells": [{"cell": "y@1", "skill": 0.12}, {"cell": "y@24", "skill": 0.02}],
                    },
                    "finished": {
                        "tokens": 200_000, "seconds": 400,
                        "rescue_triggered": False, "hook_fired": False, "final_text": "",
                    },
                    "notes": [],
                },
            }
            md = triage(
                view, records,
                {"inc_digest_1234": "inc", "cand_digest_5678": "g1a"},
                harnesses={"inc_digest_1234": inc_h},
            )
        self.assertIn("高分未交", md)
        self.assertIn("本地→隐藏期过拟合反转", md)
        self.assertIn("`h=24`", md)
        self.assertIn("hook_fired", md)
        self.assertIn("active_skill", md)
        self.assertIn("零激活技能", md)
        self.assertIn("unused_skill", md)
        self.assertIn("高成本低增益", md)

    def test_triage_distinguishes_strong_template_keep_gate_block_and_trajectory_v4(self):
        from ecogym.episodes.trajectory import sessions, summarize
        from ecogym.evolution.evidence import triage

        # 1. Verify DSH 0.2 session.v4.jsonl.zstd parsing & skill/web_search extraction
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sdir = root / "sessions" / "work-a" / "session-1"
            sdir.mkdir(parents=True)
            v3_lines = [
                json.dumps({"type": "session", "id": "s1", "delegationDepth": 0}),
                json.dumps({"type": "tool/call", "data": {"name": "read", "arguments": "{}"}}),
            ]
            v4_lines = [
                json.dumps({"type": "session", "version": 4, "id": "s1", "delegationDepth": 0}),
                json.dumps({"type": "tool/call", "data": {"name": "skill", "arguments": '{"name":"my-skill"}'}}),
                json.dumps({"type": "tool/call", "data": {"name": "web_search", "arguments": '{"query":"q1"}'}}),
            ]
            # Write compressed zstd via node (available in runtime)
            from ecogym.episodes.sandbox import discover_runtime

            rt = discover_runtime()
            if rt.node_root and (rt.node_root / "bin" / "node").is_file():
                node = rt.node_root / "bin" / "node"
                compress = (
                    "const fs=require('fs'),z=require('zlib');"
                    "fs.writeFileSync(process.argv[1],z.zstdCompressSync(Buffer.from(process.argv[2])));"
                )
                subprocess.run([str(node), "-e", compress, str(sdir / "session.v3.jsonl.zstd"), "\n".join(v3_lines)],
                               check=True)
                subprocess.run([str(node), "-e", compress, str(sdir / "session.v4.jsonl.zstd"), "\n".join(v4_lines)],
                               check=True)
                found = sessions(root, node)
                self.assertEqual(len(found), 1)
                summary = summarize(found)
                self.assertEqual(summary["tool_calls"], {"skill": 1, "web_search": 1})
                self.assertEqual(summary["skills"], {"my-skill": 1})
                self.assertEqual(summary["web_queries"], ["q1"])

            # 2. Verify triage distinguishes strong-template keep vs gate rejection vs weak zero-sub
            h_root = root / "h_strong"
            (h_root / "skills" / "my-skill").mkdir(parents=True)
            (h_root / "skills" / "dead-skill").mkdir(parents=True)
            strong_h = SimpleNamespace(digest="strong_1234", root=h_root)
            calls_dir = root / "traj_ep_strong"
            calls_dir.mkdir()
            with gzip.open(calls_dir / "llm_calls.jsonl.gz", "wt", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "reply": {"message": {"tool_calls": [
                        {"function": {"name": "skill", "arguments": '{"name": "my-skill"}'}}
                    ]}}
                }) + "\n")
            view = {
                "incumbent": "strong_1234",
                "rounds": [{
                    "generation": 1, "finished": True, "candidates": [],
                    "episodes": [
                        {"episode": "ep:keep", "task": "t1", "harness": "strong_1234", "status": "finished"},
                        {"episode": "ep:gate", "task": "t1", "harness": "strong_1234", "status": "finished"},
                        {"episode": "ep:weak", "task": "t1", "harness": "strong_1234", "status": "finished"},
                    ],
                }],
            }
            records = {
                "ep:keep": {
                    "episode": "ep:keep", "task_id": "t1", "status": "finished",
                    "start": {"local": {"score": 0.22}},
                    "evaluations": [{"local": {"score": 0.223}, "local_selection": {"passed": False}}],
                    "submissions": [],
                    "scored": {"score": 0.21, "start": True, "cells": []},
                    "finished": {"tokens": 50_000, "seconds": 100, "trajectory": {"dir": str(calls_dir)}},
                    "notes": [],
                },
                "ep:gate": {
                    "episode": "ep:gate", "task_id": "t1", "status": "finished",
                    "start": {"local": {"score": 0.05}},
                    "evaluations": [{
                        "local": {"score": 0.09},
                        "local_selection": {"passed": False, "reason": "pre_fold_regression"},
                    }],
                    "submissions": [],
                    "scored": {"score": 0.05, "start": True, "cells": []},
                    "finished": {"tokens": 50_000, "seconds": 100, "refused_calls": 1},
                    "notes": [],
                },
                "ep:weak": {
                    "episode": "ep:weak", "task_id": "t1", "status": "finished",
                    "start": {"local": {"score": 0.0}},
                    "evaluations": [], "submissions": [],
                    "scored": {"score": 0.0, "start": True, "cells": []},
                    "finished": {"tokens": 50_000, "seconds": 100},
                    "notes": [],
                },
            }
            md = triage(view, records, {"strong_1234": "g1a"}, harnesses={"strong_1234": strong_h})
        self.assertIn("强模板合规沿用 1", md)
        self.assertIn("触顶超时 1", md)
        self.assertNotIn("ep:keep", md)
        self.assertIn("门控拦截:pre_fold_regression", md)
        self.assertIn("弱模板零提交", md)
        self.assertIn("`my-skill` | 1/3", md)
        self.assertIn("`dead-skill`（0/3 回合）", md)

    def test_dgm_parent_weights_and_operator_schedule(self):
        from ecogym.evolution.pareto import dgm_parent_weights, operator_schedule_section

        by_label = {
            "g1a": {"status": "rejected", "parent": "seed", "decision": {
                "delta": 0.02, "tasks": [{"delta": 0.02}, {"delta": 0.02}], "horizons": {"1": 0.03, "24": 0.01},
            }},
            "g1b": {"status": "rejected", "parent": "seed", "decision": {
                "delta": 0.02, "tasks": [{"delta": 0.02}, {"delta": 0.02}], "horizons": {"1": 0.01, "24": 0.04},
            }},
        }
        digests = {"g1a": "d_a", "g1b": "d_b"}
        points = extract_points(by_label, digests)
        # g1a already spawned 3 children -> g1b (0 children) gets 4x weight despite identical delta
        weights = dgm_parent_weights(points, {"d_a": 3, "d_b": 0})
        self.assertEqual(weights[0]["label"], "g1b")
        self.assertAlmostEqual(weights[0]["weight"] / weights[1]["weight"], 4.0, places=4)

        pareto_md = render_pareto_section(by_label, digests, children_counts={"d_a": 3, "d_b": 0})
        self.assertIn("DGM 踏脚石父本推荐", pareto_md)
        self.assertIn("衍生 0 次", pareto_md)

        pairs = crossover_pairs(points)
        op_md = operator_schedule_section(["a", "b", "c"], points=points, children_counts={"d_a": 3, "d_b": 0},
                                          pairs=pairs)
        self.assertIn("定向开发", op_md)
        self.assertIn("结构探索", op_md)
        self.assertIn("精简消融或护栏加固", op_md)
        self.assertIn("g1a", op_md)
        self.assertIn("g1b", op_md)


if __name__ == "__main__":
    unittest.main()
