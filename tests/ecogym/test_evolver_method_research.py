"""Method research treatment and source-only probe controls."""

from __future__ import annotations

import gzip
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from ecogym.episodes.agent import Session
from ecogym.episodes.judge import Limits
from ecogym.episodes.mcp import ToolError
from ecogym.evolution.evidence import archive, overview
from ecogym.evolution.evolver import (
    EvolverSettings,
    EvolverTools,
    _discover_methods,
    _discovery_handoff,
    _discovery_sources,
    _include_discovery,
    _method_guidance,
    _publish_adapter,
    _should_discover_methods,
    archive_readme,
    run_evolver,
)
from ecogym.evolution.loop import _builder_stalled
from ecogym.kernel import clock

SOURCE = "https://doi.org/10.1234/example"
METHOD = "MethodOne"
DIGEST = "a" * 64


class ResearchTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        (self.root / "work" / "variants").mkdir(parents=True)
        self.method = SimpleNamespace(id=METHOD, digest=DIGEST,
                                      describe=lambda: {"id": METHOD, "source": SOURCE, "digest": DIGEST})
        self.registry = Mock()
        self.registry.cards.return_value = [self.method.describe()]
        self.registry.find_source.side_effect = lambda url: (self.method,) if url == SOURCE else ()

    def tools(self, mode, registry=None):
        return EvolverTools(root=self.root, incumbent=None, names=["a"], tag="test", smoke=None, runtime=None,
                            limits=Limits(), forbidden=(), deadline=clock.now() + 1_000, usage=dict,
                            event=lambda *_: None, generation=1, method_research=mode, method_registry=registry)

    def test_treatments_only_supplied_mode_enumerates_methods(self):
        control = self.tools("internal", self.registry)
        self.assertEqual({tool.name for tool in control.catalog()},
                         {"task_info", "check_candidate", "screen_templates", "reset_candidate", "note", "ask_expert"})
        auto = _method_guidance("autonomous", generation=1, registry=self.registry, search=True, view={"rounds": []})
        self.assertNotIn(METHOD, auto)
        self.assertIn("web_search", auto)
        self.registry.cards.assert_not_called()
        supplied = _method_guidance("supplied", generation=1, registry=self.registry, search=False,
                                    view={"rounds": []})
        self.assertIn(METHOD, supplied)
        self.assertIn(SOURCE, supplied)
        self.registry.cards.assert_called_once()

    def test_probe_requires_exact_host_source_and_stages_verified_program(self):
        tools = self.tools("autonomous", self.registry)

        def materialize(digest, template):
            self.assertEqual(digest, DIGEST)
            template.mkdir()
            (template / "predictor.py").write_text("# verified source\n", encoding="utf-8")
            return self.method

        self.registry.materialize.side_effect = materialize
        self.assertEqual(tools.probe_method("https://example.test/unknown")["status"], "unregistered")
        self.registry.materialize.assert_not_called()
        report = {"task": {"status": "ok", "score": 0.3, "coverage": 1.0}}
        with (patch("ecogym.evolution.evolver.scratch_root", return_value=self.root),
              patch.object(tools, "incumbent_smoke", return_value={"task": {"score": 0.2}}),
              patch.object(tools, "_judge_template", return_value=report) as judge):
            observed = tools.probe_method(SOURCE, variant="example")
        judge.assert_called_once()
        self.assertEqual(observed["method"]["digest"], DIGEST)
        self.assertAlmostEqual(observed["result"]["gain"], 0.1)
        self.assertEqual((self.root / "work/variants/example/predictor.py").read_text(), "# verified source\n")

    def test_probe_publishing_does_not_follow_workspace_symlink(self):
        variants = self.root / "work/variants"
        variants.rmdir()
        outside = self.root / "outside"
        outside.mkdir()
        variants.symlink_to(outside, target_is_directory=True)
        adapter = self.root / "adapter"
        adapter.mkdir()
        with self.assertRaises(OSError):
            _publish_adapter(self.root, adapter, "test")
        self.assertEqual(list(outside.iterdir()), [])

    def test_records_keep_source_and_candidate_without_approving_unknown_imports(self):
        tools = self.tools("autonomous")
        tools.record_method(SOURCE + "#section", METHOD, "adapt", "Local checks used the same information.", "a")
        self.assertEqual(tools.method_records[0]["source"], SOURCE)
        self.assertEqual(tools.method_records[0]["candidate"], "a")
        supplied = self.tools("supplied", self.registry)
        with self.assertRaises(ToolError):
            supplied.record_method("https://example.test/unknown", "Unknown", "adapt", "A claim")
        with self.assertRaises(ToolError):
            tools.record_method("https://user@publisher.test/private", "Invalid", "adapt", "A claim")

    def test_later_guidance_can_reuse_own_development_reference(self):
        view = {"rounds": [{"generation": 1, "finished": True,
                            "evolver": {"method_research": {"records": [{"source": SOURCE}]}},
                            "public_references": [{"model": METHOD, "delta": 0.04}]}]}
        first = _method_guidance("autonomous", generation=1, registry=None, search=True, view=view)
        later = _method_guidance("autonomous", generation=2, registry=None, search=True, view=view)
        self.assertNotIn(SOURCE, first)
        self.assertIn(SOURCE, later)
        self.assertIn(METHOD, later)
        self.assertIn("开发期", later)

    def test_discovery_handoff_uses_host_search_results_only(self):
        trajectory = self.root / "research-trajectory"
        trajectory.mkdir()
        seen = "https://example.test/official?id=7"
        claimed = "https://example.test/made-up"
        body = {"content": [
            {"type": "web_search_tool_result", "content": [
                {"type": "web_search_result", "url": seen + "#section", "title": "Official benchmark"},
            ]},
            {"type": "text", "text": f"Model-generated claim {claimed}"},
        ]}
        with gzip.open(trajectory / "llm_calls.jsonl.gz", "wt", encoding="utf-8") as log:
            log.write(json.dumps({"path": "/anthropic/v1/messages", "status": 200,
                                  "request": {"messages": [{"content": [{
                                      "text": "Perform a web search for the query: recent benchmark"}]}]},
                                  "response": json.dumps(body)}) + "\n")
        gateway = Mock()
        researcher = self.root / "research"
        task = SimpleNamespace(id="private-id", title="Hourly process", targets=("y",), horizons=(1, 6),
                               season_steps=24, history_steps=48, history_scope="rolling", exogenous_roles=())
        bench = SimpleNamespace(frame=SimpleNamespace(step="1h"))

        def research_agent(**kwargs):
            self.assertEqual(kwargs["root"], researcher)
            self.assertEqual({item.name for item in kwargs["tools"]}, {"record_discovery"})
            self.assertEqual({item.name for item in (researcher / "work").iterdir()}, {"TASK_PROFILE.md"})
            self.assertNotIn(task.id, (researcher / "work/TASK_PROFILE.md").read_text(encoding="utf-8"))
            tool = kwargs["tools"][0].handler
            year = time.gmtime().tm_year
            tool("competitive", seen, "A", "Family A", year, "Protocol and result", "Borrow", "Check scope")
            tool("practical", claimed, "B", "Family B", year, "Unsupported", "Borrow", "Check scope")
            return Session("finished", 0, False, 3, 100, 70, 2, 2, 0, 0, "done", {},
                           {"web_queries": ["recent benchmark", "lightweight alternative"]})

        start = clock.now()
        with (patch("ecogym.evolution.evolver.make_gateway", return_value=gateway) as make,
              patch("ecogym.evolution.evolver.run_agent", side_effect=research_agent)):
            handoff, session = _discover_methods(
                root=researcher, trajectory=trajectory, runtime=None, settings=EvolverSettings(minutes=4, tokens=400),
                tasks=[task], benches={task.id: bench}, limits=Limits(), started=start, cancel=None,
                observe_search=Mock())
        self.assertEqual(make.call_args.kwargs["deadline"], start + 72)
        self.assertEqual(make.call_args.kwargs["tokens"], 120)
        self.assertEqual(handoff["status"], "partial")
        self.assertEqual(handoff["unmatched_records"], 1)
        self.assertEqual(handoff["records"][0]["source_title"], "Official benchmark")
        self.assertEqual(handoff["search_leads"], [{"query": "recent benchmark", "sources": [
            {"source": seen, "title": "Official benchmark"},
        ]}])
        self.assertFalse(researcher.exists())
        builder = Session("finished", 0, False, 5, 230, 150, 4, 4, 0, 0, "candidate", {}, {})
        _include_discovery(builder, session)
        self.assertEqual((builder.seconds, builder.metered_tokens, builder.llm_calls), (8, 220, 6))
        self.assertEqual(builder.final_text, "candidate")
        self.assertEqual(builder.activity["web_queries"], session.activity["web_queries"])

    def test_every_search_query_counts_though_only_two_become_leads(self):
        """evo:ee1aa9c321df's researcher ran 14 distinct queries; the handoff reported 2 (the leads)."""

        trajectory = self.root / "searches"
        trajectory.mkdir()
        with gzip.open(trajectory / "llm_calls.jsonl.gz", "wt", encoding="utf-8") as log:
            for index, query in enumerate(("first", "second", "Second", "third")):
                body = {"content": [{"type": "web_search_tool_result", "content": [
                    {"type": "web_search_result", "url": f"https://example.test/{index}", "title": f"T{index}"}]}]}
                log.write(json.dumps({"path": "/anthropic/v1/messages", "status": 200, "response": json.dumps(body),
                                      "request": {"messages": [{"content": [{
                                          "text": f"Perform a web search for the query: {query}"}]}]}}) + "\n")
        sources, leads, queries = _discovery_sources(trajectory)
        self.assertEqual(len(sources), 4)
        self.assertEqual([lead["query"] for lead in leads], ["first", "second"])
        self.assertEqual(queries, ["first", "second", "third"])
        self.assertEqual(_discovery_handoff([], sources, year=2026, queries=queries)["distinct_queries"], 3)

    def test_discovery_completion_requires_recent_and_distinct_evidence(self):
        other = "https://example.test/method"
        records = [{"lane": "competitive", "source": SOURCE, "year": 2025, "family": "A"},
                   {"lane": "practical", "source": other, "year": 2000, "family": "B"}]
        sources = {SOURCE: "A", other: "B"}
        self.assertEqual(_discovery_handoff(records, sources, year=2026, queries=["Q", "q"])["status"], "partial")
        self.assertEqual(_discovery_handoff(records, sources, year=2026, queries=["q", "different"])["status"],
                         "complete")
        self.assertEqual(_discovery_handoff(records, sources, year=2030, queries=["q", "different"])["status"],
                         "partial")


class BuilderBudgetTests(unittest.TestCase):
    """Generation 1's research session runs on its own time and tokens and the builder then gets its whole budget.
    Until 0.4.2 the research came out of the builder's: in evo:ee1aa9c321df generation 1 it took 272 s of a
    15-minute session, the builder began with 8.4 minutes and screened one of its five template variants."""

    MINUTES, TOKENS, RESEARCH, OPENED = 15.0, 4_000_000, 272.0, 10_000.0

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def evolve(self, generation: int, method_research: str = "autonomous"):
        now = [self.OPENED]
        seen = {}
        gateway = SimpleNamespace(has_deepseek=True, on_call=None, usage=dict)

        def make_gateway(trajectory, *, tokens, deadline, model):
            gateway.token_budget, gateway.deadline = tokens, deadline
            return gateway

        def research(**kwargs):
            seen["research_started"] = kwargs["started"]
            now[0] += self.RESEARCH
            session = Session("timeout", 0, True, self.RESEARCH, 280_000, 270_000, 30, 25, 5, 0, "", {}, {},
                              stalled_seconds=65.5)
            return {"status": "partial", "records": [], "search_leads": [], "session": session.to_dict()}, session

        def builder(**kwargs):
            seen["builder"] = {"deadline": kwargs["deadline"], "gateway_deadline": gateway.deadline,
                               "tokens": gateway.token_budget}
            now[0] += 600.0
            return Session("finished", 0, False, 600.0, 900_000, 860_000, 21, 15, 6, 0, "", {}, {},
                           stalled_seconds=180.0)

        tools = Mock(method_probes=[], method_records=[])
        tools.collect.return_value = []
        make_tools, write_brief, follow_up = Mock(return_value=tools), Mock(return_value=""), Mock(return_value=None)
        task = SimpleNamespace(id="greenhouse.example")
        incumbent = SimpleNamespace(digest="d" * 64, name="seed")
        with (patch.object(clock, "now", side_effect=lambda: now[0]),
              patch.multiple("ecogym.evolution.evolver", discover_runtime=Mock(return_value=None),
                             scratch_root=Mock(return_value=self.root / "scratch"),
                             ecogym_home=Mock(return_value=self.root / "home"), make_gateway=make_gateway,
                             _discover_methods=research, run_agent=builder, fork=Mock(), Ledger=MagicMock(),
                             related_runs=Mock(return_value=[]), archive_versions=Mock(return_value={}),
                             forbidden_terms=Mock(return_value=[]), write_evidence=Mock(return_value={}),
                             ensemble_scores=Mock(return_value={}), known_versions=Mock(return_value={}),
                             brief=write_brief, evolver_follow_up=follow_up, Events=Mock(), EvolverTools=make_tools)):
            outcome = run_evolver(
                run_id="evo:test", generation=generation, incumbent=incumbent, suite=None, tasks=[task],
                benches={task.id: None}, view={"rounds": [], "seed": {"digest": incumbent.digest}},
                ledger_path=self.root / "ledger.sqlite3",
                settings=EvolverSettings(minutes=self.MINUTES, tokens=self.TOKENS), repeats=2, episode_minutes=8,
                episode_tokens=3_000_000, model=None, limits=Limits(), method_research=method_research)
        budgets = {"brief": write_brief.call_args.kwargs, "tools": make_tools.call_args.kwargs,
                   "follow_up": follow_up.call_args.kwargs}
        return seen, outcome, budgets

    def test_builder_starts_its_whole_budget_after_the_research(self):
        seen, outcome, budgets = self.evolve(generation=1)
        self.assertEqual(seen["research_started"], self.OPENED)
        after_research = self.OPENED + self.RESEARCH + self.MINUTES * 60
        self.assertEqual(seen["builder"], {"deadline": after_research, "gateway_deadline": after_research,
                                           "tokens": self.TOKENS})
        self.assertEqual((budgets["brief"]["minutes"], budgets["brief"]["tokens"], budgets["brief"]["deadline"]),
                         (self.MINUTES, self.TOKENS, after_research))
        self.assertEqual(budgets["tools"]["deadline"], after_research)
        self.assertEqual((budgets["follow_up"]["minutes"], budgets["follow_up"]["deadline"]),
                         (self.MINUTES, after_research))
        # the round's record holds both sessions; the stall rule reads the builder's waiting alone
        self.assertEqual((outcome["session"]["seconds"], outcome["session"]["stalled_seconds"]), (872.0, 245.5))
        self.assertEqual(outcome["session"]["metered_tokens"], 1_130_000)
        self.assertEqual(_builder_stalled(outcome), 180.0)

    def test_generations_without_research_keep_the_budget_from_the_start(self):
        for generation, mode in ((2, "autonomous"), (1, "internal")):
            seen, outcome, budgets = self.evolve(generation=generation, method_research=mode)
            self.assertNotIn("research_started", seen)
            start = self.OPENED + self.MINUTES * 60
            self.assertEqual(seen["builder"], {"deadline": start, "gateway_deadline": start, "tokens": self.TOKENS})
            self.assertEqual(budgets["brief"]["minutes"], self.MINUTES)
            self.assertEqual(_builder_stalled(outcome), 180.0)

    def test_stall_rule_ignores_only_the_research_sessions_waiting(self):
        merged = {"session": {"stalled_seconds": 500.0},
                  "method_research": {"discovery": {"session": {"stalled_seconds": 200.0}}}}
        self.assertEqual(_builder_stalled(merged), 300.0)  # under half of 900 s: the builder was not crippled
        self.assertEqual(_builder_stalled({"session": {"stalled_seconds": 500.0},
                                           "method_research": {"discovery": {"status": "search_unavailable"}}}),
                         500.0)
        self.assertEqual(_builder_stalled({"session": {"stalled_seconds": None}}), 0.0)

    def test_stagnation_triggers_autonomous_discovery_with_cooldown(self):
        r1_disc = {"generation": 1, "finished": True, "outcome": "kept",
                   "evolver": {"method_research": {"discovery": {"status": "complete"}}}}
        r2_kept = {"generation": 2, "finished": True, "outcome": "kept", "evolver": {"method_research": {}}}
        r3_kept = {"generation": 3, "finished": True, "outcome": "kept", "evolver": {"method_research": {}}}
        r3_adopted = {"generation": 3, "finished": True, "outcome": "adopted", "evolver": {"method_research": {}}}
        self.assertTrue(_should_discover_methods("autonomous", 1, {"rounds": []}))
        self.assertFalse(_should_discover_methods("internal", 1, {"rounds": []}))
        self.assertFalse(_should_discover_methods("autonomous", 2, {"rounds": [r1_disc]}))
        # Gen 3 has r1_disc in its last 2 rounds -> cooldown prevents back-to-back discovery
        self.assertFalse(_should_discover_methods("autonomous", 3, {"rounds": [r1_disc, r2_kept]}))
        # Gen 4 after two consecutive non-adopting non-discovery rounds (r2, r3) -> triggers!
        self.assertTrue(_should_discover_methods("autonomous", 4, {"rounds": [r1_disc, r2_kept, r3_kept]}))
        # Gen 4 when r3 adopted -> does not trigger
        self.assertFalse(_should_discover_methods("autonomous", 4, {"rounds": [r1_disc, r2_kept, r3_adopted]}))

    def test_screen_result_and_evidence_expose_per_horizon_gains_and_stepping_stones(self):
        ref = {"t1": {"score": 0.10, "horizons": {1: 0.20, 6: 0.05, 24: 0.05}}}
        rep = {"t1": {"status": "ok", "score": 0.12, "coverage": 1.0,
                      "cells": [{"cell": "y@1", "objective": 0.21},
                                {"cell": "y@6", "objective": 0.06},
                                {"cell": "y@24", "objective": 0.09}]}}
        screened = EvolverTools._screen_result("v1", rep, ref)
        self.assertEqual(screened["horizons"], {"1": 0.01, "6": 0.01, "24": 0.04})
        self.assertEqual(screened["worst_horizon"], 0.01)

        decision = {
            "delta": -0.002,
            "interval": [-0.01, 0.008],
            "reason": "below_practical_delta",
            "tasks": [{"task": "t1", "delta": 0.01}, {"task": "t2", "delta": -0.014}],
            "horizons": {1: -0.015, 6: -0.003, 24: 0.012},
            "cells": {},
        }
        cand = {"name": "a", "digest": "b" * 64, "status": "rejected", "parent": "c" * 64, "decision": decision}
        view = {"run_id": "evo:1", "incumbent": "c" * 64, "suite": {"train": ["t1", "t2"]},
                "rounds": [{"generation": 1, "finished": True, "incumbent": "c" * 64, "candidates": [cand],
                            "episodes": [], "notes": []}]}
        ov = overview(view, {}, ["t1", "t2"], {})
        self.assertIn("各时距：h=1 -0.015，h=6 -0.003，h=24 +0.012", ov)
        ar = archive(view, {"g1a": "b" * 64})
        self.assertIn("分时距平均变化：h=1 -0.015，h=6 -0.003，h=24 +0.012", ar)
        readme = archive_readme(view, {"g1a": SimpleNamespace(digest="b" * 64, summary="long horizon model")},
                                incumbent="c" * 64)
        self.assertIn("◇ Δ -0.0020", readme)
        self.assertIn("h=24 +0.0120", readme)


if __name__ == "__main__":
    unittest.main()
