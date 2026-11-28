from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ecogym.episodes.agent import AgentSpec, Cancelled
from ecogym.evaluation.suite import SuiteDecision
from ecogym.evolution.evidence import overview
from ecogym.evolution.fold import evolver_sessions
from ecogym.evolution.loop import Evolution, EvolveConfig
from ecogym.evolution.relay import (
    effective_relay_batch,
    effective_relay_candidates,
    pick_stage_leader,
    rank_stage_candidates,
    render_relay_brief,
    select_finalists,
    select_stage_tasks,
)
from ecogym.kernel.ledger import Ledger
from ecogym.service import Service

from .helpers import synthetic_bench

_BASE_TASK = synthetic_bench().task


def _fake_task(task_id: str, dataset: str):
    return replace(_BASE_TASK, id=task_id, dataset=dataset)


def _fake_harness(name: str, digest: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        digest=digest,
        summary=f"summary {name}",
        agent=AgentSpec(model="deepseek-chat"),
        root=Path("/tmp"),
    )


class RelayHelpersTests(unittest.TestCase):
    def test_select_stage_tasks_stratified_and_complementary(self):
        tasks = [
            _fake_task("a1", "dsA"),
            _fake_task("a2", "dsA"),
            _fake_task("a3", "dsA"),
            _fake_task("a4", "dsA"),
            _fake_task("b1", "dsB"),
            _fake_task("b2", "dsB"),
            _fake_task("b3", "dsB"),
            _fake_task("b4", "dsB"),
        ]
        s1 = select_stage_tasks(tasks, generation=1, stage=1, total_stages=2)
        s2 = select_stage_tasks(tasks, generation=1, stage=2, total_stages=2)
        self.assertEqual(len(s1), 4)
        self.assertEqual(len(s2), 4)
        # Stratified: 2 from dsA and 2 from dsB in each stage
        self.assertEqual(sum(t.dataset == "dsA" for t in s1), 2)
        self.assertEqual(sum(t.dataset == "dsB" for t in s1), 2)
        self.assertEqual(sum(t.dataset == "dsA" for t in s2), 2)
        self.assertEqual(sum(t.dataset == "dsB" for t in s2), 2)
        # Strictly complementary (disjoint union == full suite)
        self.assertEqual({t.id for t in s1} & {t.id for t in s2}, set())
        self.assertEqual({t.id for t in s1} | {t.id for t in s2}, {t.id for t in tasks})

        # 3-stage covers all 8 tasks with batch size 3
        tri = [select_stage_tasks(tasks, generation=1, stage=s, total_stages=3) for s in (1, 2, 3)]
        self.assertEqual([len(chunk) for chunk in tri], [3, 3, 3])
        self.assertEqual({t.id for chunk in tri for t in chunk}, {t.id for t in tasks})

        # Generation 2 rotates offset deterministically
        g2_s1 = select_stage_tasks(tasks, generation=2, stage=1, total_stages=2)
        self.assertNotEqual([t.id for t in g2_s1], [t.id for t in s1])

    def test_select_stage_tasks_prioritizes_high_regret_difficulty(self):
        tasks = [
            _fake_task("a1", "dsA"),
            _fake_task("a2", "dsA"),
            _fake_task("a3", "dsA"),
            _fake_task("a4", "dsA"),
            _fake_task("b1", "dsB"),
            _fake_task("b2", "dsB"),
            _fake_task("b3", "dsB"),
            _fake_task("b4", "dsB"),
        ]
        # Make a4, a3 hardest in dsA and b3, b4 hardest in dsB
        difficulty = {"a1": 0.01, "a2": 0.02, "a3": 0.15, "a4": 0.25, "b1": 0.01, "b2": 0.03, "b3": 0.30, "b4": 0.18}
        s1 = select_stage_tasks(tasks, generation=2, stage=1, total_stages=2, task_difficulty=difficulty)
        s2 = select_stage_tasks(tasks, generation=2, stage=2, total_stages=2, task_difficulty=difficulty)
        self.assertEqual({t.id for t in s1}, {"a4", "a3", "b3", "b4"})
        self.assertEqual({t.id for t in s2}, {"a1", "a2", "b1", "b2"})
        self.assertEqual({t.id for t in s1} & {t.id for t in s2}, set())

    def test_rank_stage_candidates_and_outlier_guard(self):
        tasks = [_fake_task(f"t{i}", "dsA") for i in range(4)]
        cands = [
            {"name": "r1a", "label": "g1r1a", "digest": "da", "behavior": "ba", "summary": "consistent"},
            {"name": "r1b", "label": "g1r1b", "digest": "db", "behavior": "bb", "summary": "fluke outlier"},
            {"name": "r1c", "label": "g1r1c", "digest": "dc", "behavior": "bc", "summary": "worse"},
        ]
        baseline = {"t0": 0.10, "t1": 0.10, "t2": 0.10, "t3": 0.10}
        entries = [
            # r1a: wins 3/4 (+0.03, +0.02, +0.02, -0.01) => mean +0.015, beats=True
            {"role": "candidate", "candidate": "r1a", "task": "t0", "status": "finished", "score": 0.13},
            {"role": "candidate", "candidate": "r1a", "task": "t1", "status": "finished", "score": 0.12},
            {"role": "candidate", "candidate": "r1a", "task": "t2", "status": "finished", "score": 0.12},
            {"role": "candidate", "candidate": "r1a", "task": "t3", "status": "finished", "score": 0.09},
            # r1b: 1 huge fluke (+0.20) and 3 losses (-0.01 each) => mean +0.0425, better=1/4 < half => beats=False!
            {"role": "candidate", "candidate": "r1b", "task": "t0", "status": "finished", "score": 0.30},
            {"role": "candidate", "candidate": "r1b", "task": "t1", "status": "finished", "score": 0.09},
            {"role": "candidate", "candidate": "r1b", "task": "t2", "status": "finished", "score": 0.09},
            {"role": "candidate", "candidate": "r1b", "task": "t3", "status": "finished", "score": 0.09},
            # r1c: loses all 4
            {"role": "candidate", "candidate": "r1c", "task": "t0", "status": "finished", "score": 0.05},
            {"role": "candidate", "candidate": "r1c", "task": "t1", "status": "finished", "score": 0.06},
            {"role": "candidate", "candidate": "r1c", "task": "t2", "status": "finished", "score": 0.07},
            {"role": "candidate", "candidate": "r1c", "task": "t3", "status": "finished", "score": 0.08},
        ]
        ranked = rank_stage_candidates(cands, entries, tasks, incumbent_baseline=baseline)
        self.assertEqual(ranked[0]["name"], "r1a")
        self.assertTrue(ranked[0]["beats_incumbent"])
        self.assertFalse(next(r for r in ranked if r["name"] == "r1b")["beats_incumbent"])

        leader = pick_stage_leader(ranked, fallback_digest="seed", fallback_label="incumbent")
        self.assertTrue(leader["advanced"])
        self.assertEqual(leader["digest"], "da")
        self.assertEqual(leader["label"], "g1r1a")

        # If only r1b and r1c ran, neither beats incumbent consistently -> stay on fallback!
        only_bad = rank_stage_candidates(cands[1:], entries, tasks, incumbent_baseline=baseline)
        kept = pick_stage_leader(only_bad, fallback_digest="seed", fallback_label="incumbent")
        self.assertFalse(kept["advanced"])
        self.assertEqual(kept["digest"], "seed")

    def test_select_finalists_falls_back_to_stage1_when_stage2_regresses(self):
        stages = [
            {
                "stage": 1,
                "candidates": [{"name": "r1a", "label": "g1r1a", "digest": "d1", "behavior": "b1"}],
                "ranking": [{"name": "r1a", "label": "g1r1a", "digest": "d1", "beats_incumbent": True,
                             "paired": 4, "better": 4, "total_tasks": 4, "mean_delta": 0.045, "win_rate": 1.0,
                             "mean_score": 0.15}],
            },
            {
                "stage": 2,
                "candidates": [{"name": "r2a", "label": "g1r2a", "digest": "d2", "behavior": "b2"}],
                "ranking": [{"name": "r2a", "label": "g1r2a", "digest": "d2", "beats_incumbent": False,
                             "paired": 4, "better": 1, "total_tasks": 4, "mean_delta": -0.012, "win_rate": 0.25,
                             "mean_score": 0.09}],
            },
        ]
        finalists = select_finalists(stages, limit=1)
        self.assertEqual(len(finalists), 1)
        self.assertEqual(finalists[0]["digest"], "d1")
        self.assertEqual(finalists[0]["relay_origin"]["stage"], 1)
        self.assertEqual(finalists[0]["relay_origin"]["label"], "g1r1a")

    def test_select_finalists_merges_cross_stage_deltas_and_prunes_all_worse(self):
        stages = [
            {
                "stage": 1,
                "candidates": [{"name": "r1a", "label": "g1r1a", "digest": "d1", "behavior": "b1"}],
                "ranking": [{
                    "name": "r1a", "label": "g1r1a", "digest": "d1", "beats_incumbent": True,
                    "paired": 2, "better": 2, "total_tasks": 2, "mean_delta": 0.05, "win_rate": 1.0,
                    "mean_score": 0.15,
                    "task_deltas": [{"task": "t0", "delta": 0.05}, {"task": "t1", "delta": 0.05}],
                }],
            },
            {
                "stage": 2,
                "candidates": [
                    {"name": "r2a", "label": "g1r2a", "digest": "d2", "behavior": "b2"},
                    {"name": "r2_lead", "label": "g1r1a", "digest": "d1", "behavior": "b1", "carried_leader": True},
                ],
                "ranking": [
                    {
                        "name": "r2_lead", "label": "g1r1a", "digest": "d1", "beats_incumbent": True,
                        "paired": 2, "better": 2, "total_tasks": 2, "mean_delta": 0.03, "win_rate": 1.0,
                        "mean_score": 0.13,
                        "task_deltas": [{"task": "t2", "delta": 0.03}, {"task": "t3", "delta": 0.03}],
                    },
                    {
                        "name": "r2a", "label": "g1r2a", "digest": "d2", "beats_incumbent": False,
                        "paired": 2, "better": 1, "total_tasks": 2, "mean_delta": 0.0175, "win_rate": 0.5,
                        "mean_score": 0.12,
                        "task_deltas": [{"task": "t2", "delta": 0.045}, {"task": "t3", "delta": -0.01}],
                    },
                ],
            },
        ]
        finalists = select_finalists(stages, limit=1)
        self.assertEqual(len(finalists), 1)
        self.assertEqual(finalists[0]["digest"], "d1")
        origin = finalists[0]["relay_origin"]
        self.assertEqual(origin["paired"], 4)
        self.assertEqual(origin["better"], 4)
        self.assertAlmostEqual(origin["mean_delta"], 0.04, places=5)

        # Early exit when all candidates regress below -min_delta
        all_worse = [{
            "stage": 1,
            "candidates": [{"name": "r1a", "label": "g1r1a", "digest": "d_bad", "behavior": "b_bad"}],
            "ranking": [{
                "name": "r1a", "label": "g1r1a", "digest": "d_bad", "beats_incumbent": False,
                "paired": 2, "better": 0, "total_tasks": 2, "mean_delta": -0.04, "win_rate": 0.0,
                "mean_score": 0.06,
                "task_deltas": [{"task": "t0", "delta": -0.04}, {"task": "t1", "delta": -0.04}],
            }],
        }]
        self.assertEqual(select_finalists(all_worse, limit=1, min_delta=0.005, skip_if_all_worse=True), [])
        self.assertEqual(len(select_finalists(all_worse, limit=1, min_delta=0.005, skip_if_all_worse=False)), 1)
        # Neutral candidate (-0.001 > -0.005) is NOT pruned
        neutral = [{
            "stage": 1,
            "candidates": [{"name": "r1a", "label": "g1r1a", "digest": "d_neu", "behavior": "b_neu"}],
            "ranking": [{
                "name": "r1a", "label": "g1r1a", "digest": "d_neu", "beats_incumbent": False,
                "paired": 2, "better": 1, "total_tasks": 2, "mean_delta": -0.001, "win_rate": 0.5,
                "mean_score": 0.099,
                "task_deltas": [{"task": "t0", "delta": 0.002}, {"task": "t1", "delta": -0.004}],
            }],
        }]
        self.assertEqual(len(select_finalists(neutral, limit=1, min_delta=0.005, skip_if_all_worse=True)), 1)

    def test_render_relay_brief_and_auto_defaults(self):
        self.assertEqual(effective_relay_candidates(1, 0), 2)
        self.assertEqual(effective_relay_candidates(3, 0), 3)
        self.assertEqual(effective_relay_candidates(1, 4), 4)
        self.assertEqual(effective_relay_batch(8, total_stages=2, batch_size=0), 4)
        self.assertEqual(effective_relay_batch(8, total_stages=3, batch_size=0), 3)
        text = render_relay_brief(
            generation=1,
            stage=2,
            total_stages=2,
            stage_tasks=[_fake_task("t2", "dsA"), _fake_task("t3", "dsB")],
            prior_stages=[{
                "stage": 1, "tasks": ["t0", "t1"],
                "leader": {"advanced": True, "label": "g1r1a", "mean_delta": 0.03, "better": 2, "paired": 2},
                "ranking": [{"label": "g1r1a", "digest": "d1" * 10, "beats_incumbent": True,
                             "mean_delta": 0.03, "better": 2, "paired": 2, "summary": "better model",
                             "task_deltas": [{"task": "t0", "delta": 0.04}, {"task": "t1", "delta": 0.02}]}],
            }],
            starting_label="g1r1a",
            starting_digest="d1" * 10,
            incumbent_digest="d0" * 10,
            final_limit=1,
        )
        self.assertIn("阶段 2/2", text)
        self.assertIn("g1r1a", text)
        self.assertIn("前序接力阶段结果", text)


class RelayLoopTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.ledger_path = self.root / "ledger.sqlite3"
        self.tasks = [
            _fake_task("t0", "dsA"),
            _fake_task("t1", "dsA"),
            _fake_task("t2", "dsB"),
            _fake_task("t3", "dsB"),
        ]
        self.suite = SimpleNamespace(
            id="synthetic.relay",
            title="Synthetic Relay",
            train=[t.id for t in self.tasks],
            holdout=[],
            digest="suite-digest",
            gates=SimpleNamespace(test="t", confidence=0.95, min_delta=0.005, task_cap=0.03, max_horizon_loss=()),
            tasks=lambda _: list(self.tasks),
        )
        self.harnesses = {
            "seed": _fake_harness("seed", "seed"),
            "h_r1a": _fake_harness("r1a", "h_r1a"),
            "h_r1b": _fake_harness("r1b", "h_r1b"),
            "h_r2a": _fake_harness("r2a", "h_r2a"),
            "h_r2b": _fake_harness("r2b", "h_r2b"),
        }
        self.scores = {
            "seed": 0.10,
            "h_r1a": 0.11,
            "h_r1b": 0.14,  # wins Stage 1
            "h_r2a": 0.19,  # wins Stage 2 & Final Gate
            "h_r2b": 0.12,
        }

    def _seed_gen0(self, config: EvolveConfig) -> None:
        with Ledger(self.ledger_path) as ledger:
            ledger.append("evo:relay", "evolve_started", {
                "config": config.to_dict(),
                "suite": {"id": self.suite.id, "train": list(self.suite.train), "digest": self.suite.digest},
                "seed": {"digest": "seed", "behavior": "beh:seed", "name": "seed"},
            })
            ledger.append("evo:relay", "round_started", {"generation": 0, "incumbent": "seed",
                                                          "guidance": {t.id: None for t in self.tasks}})
            gen0_eps = [
                {"episode": f"ep:g0-{t.id}", "task": t.id, "harness": "seed", "role": "incumbent",
                 "candidate": None, "replicate": 0, "guidance": None, "status": "finished",
                 "score": self.scores["seed"], "predictions": str(self.root / f"g0-{t.id}.npz")}
                for t in self.tasks
            ]
            ledger.append("evo:relay", "episodes_finished", {"generation": 0, "episodes": gen0_eps})
            ledger.append("evo:relay", "round_finished", {"generation": 0, "incumbent": "seed", "outcome": "baseline"})

    def test_two_stage_relay_end_to_end_and_resume(self):
        config = EvolveConfig(
            suite="synthetic.relay", generations=1, candidates=1, repeats=1,
            concurrency=2, relay_stages=2, relay_candidates=2, relay_batch=2,
        )
        self._seed_gen0(config)
        with patch("ecogym.evolution.loop.suite_spec", return_value=self.suite):
            evo = Evolution(config, self.ledger_path)

        evolver_calls: list[dict] = []
        episode_calls: list[tuple[str, str | None, str, int | None]] = []

        def fake_run_evolver(**kwargs):
            evolver_calls.append({
                "starting": kwargs.get("starting").digest if kwargs.get("starting") else None,
                "starting_label": kwargs.get("starting_label"),
                "extra_archived": sorted((kwargs.get("extra_archived") or {}).keys()),
                "allow_discovery": kwargs.get("allow_discovery"),
                "stage_tag": kwargs.get("stage_tag"),
                "relay_note": kwargs.get("relay_note"),
                "candidates_requested": kwargs["settings"].candidates,
            })
            stage_tag = kwargs.get("stage_tag") or ""
            if stage_tag == "-r1":
                cands = [
                    {"name": "a", "status": "proposed", "ok": True, "digest": "h_r1a",
                     "harness": self.harnesses["h_r1a"], "summary": "r1a", "proposal": "p" * 100,
                     "agent": asdict(self.harnesses["h_r1a"].agent), "changes": [], "problems": [], "warnings": []},
                    {"name": "b", "status": "proposed", "ok": True, "digest": "h_r1b",
                     "harness": self.harnesses["h_r1b"], "summary": "r1b", "proposal": "p" * 100,
                     "agent": asdict(self.harnesses["h_r1b"].agent), "changes": [], "problems": [], "warnings": []},
                ]
            else:
                cands = [
                    {"name": "a", "status": "proposed", "ok": True, "digest": "h_r2a",
                     "harness": self.harnesses["h_r2a"], "summary": "r2a refined", "proposal": "p" * 100,
                     "base": {"label": kwargs.get("starting_label"), "digest": kwargs["starting"].digest},
                     "agent": asdict(self.harnesses["h_r2a"].agent), "changes": [], "problems": [], "warnings": []},
                    {"name": "b", "status": "proposed", "ok": True, "digest": "h_r2b",
                     "harness": self.harnesses["h_r2b"], "summary": "r2b", "proposal": "p" * 100,
                     "base": {"label": kwargs.get("starting_label"), "digest": kwargs["starting"].digest},
                     "agent": asdict(self.harnesses["h_r2b"].agent), "changes": [], "problems": [], "warnings": []},
                ]
            return {
                "session": {"status": "finished", "seconds": 12.0, "tokens": 5000, "metered_tokens": 2500,
                            "llm_calls": 3, "answered_calls": 3, "failed_calls": 0, "final_text": "done"},
                "candidates": cands,
            }

        ep_counter = {"n": 0}

        def fake_episode(run_id, generation, harness, role, name, task, replicate, snapshot, plan_id,
                         first_attempt, *, relay_stage=None):
            ep_counter["n"] += 1
            episode_calls.append((role, name, task.id, relay_stage))
            ep_id = f"ep:sim-{ep_counter['n']}"
            return {
                "episode": ep_id, "task": task.id, "harness": harness.digest, "role": role,
                "candidate": name, "replicate": replicate, "guidance": snapshot.guidance[task.id],
                "status": "finished", "score": self.scores[harness.digest],
                "predictions": str(self.root / f"{ep_id}.npz"), "predictions_digest": "sha",
                "tokens": 1000, "metered_tokens": 500, "seconds": 5.0,
                **({"relay_stage": relay_stage} if relay_stage is not None else {}),
            }

        def fake_compare(pairs, gates, meta, confidence=0.95):
            return SuiteDecision(
                adopted=True, reason="improved", delta=0.09, interval=(0.06, 0.12),
                confidence=confidence, task_confidence=0.95, basis="tasks_t",
                tasks=tuple({"task": p.task, "delta": 0.09, "candidate": 0.19, "incumbent": 0.10,
                             "interval": [0.06, 0.12], "replicates": [1, 1], "reason": "ok"} for p in pairs),
                cells={},
            )

        with patch("ecogym.evolution.loop.resolve", side_effect=self.harnesses.get), \
                patch("ecogym.evolution.loop.behavior_digest", side_effect=lambda h: f"beh:{h.digest}"), \
                patch("ecogym.evolution.loop.run_evolver", side_effect=fake_run_evolver), \
                patch("ecogym.evolution.loop.compare_suite", side_effect=fake_compare), \
                patch.object(evo, "_measurement", return_value="m1"), \
                patch.object(evo, "_episode", side_effect=fake_episode), \
                patch.object(evo, "_evaluation", return_value=object()), \
                patch.object(evo, "_record_template_attribution"):
            view = evo.view("evo:relay")
            evo._round("evo:relay", view, evo.state(view))

        # 2 evolver sessions: Stage 1 (from seed, discovery allowed) -> Stage 2 (from g1r1b, discovery off)
        self.assertEqual(len(evolver_calls), 2)
        self.assertEqual(evolver_calls[0]["starting"], "seed")
        self.assertTrue(evolver_calls[0]["allow_discovery"])
        self.assertEqual(evolver_calls[0]["candidates_requested"], 2)

        self.assertEqual(evolver_calls[1]["starting"], "h_r1b")
        self.assertEqual(evolver_calls[1]["starting_label"], "g1r1b")
        self.assertEqual(evolver_calls[1]["extra_archived"], ["g1r1a", "g1r1b"])
        self.assertFalse(evolver_calls[1]["allow_discovery"])
        self.assertIn("g1r1b", evolver_calls[1]["relay_note"])

        # Screening reused Gen 0 incumbent baseline:
        # Stage 1 = 4 cand eps (r1a, r1b x 2 tasks)
        # Stage 2 = 6 cand eps (r2a, r2b, + carried leader g1r1b as r2_lead on complementary batch x 2 tasks)
        # Final = 8 paired eps (incumbent + finalist 'a' on all 4 tasks)
        stage1_eps = [c for c in episode_calls if c[3] == 1]
        stage2_eps = [c for c in episode_calls if c[3] == 2]
        final_eps = [c for c in episode_calls if c[3] is None]
        self.assertEqual(len(stage1_eps), 4)
        self.assertEqual(len(stage2_eps), 6)
        self.assertEqual(sum(name == "r2_lead" for _, name, *_ in stage2_eps), 2)
        self.assertTrue(all(role == "candidate" for role, *_ in stage1_eps + stage2_eps))
        # Stage 1 and Stage 2 screened complementary task subsets
        self.assertEqual({t for _, _, t, _ in stage1_eps} & {t for _, _, t, _ in stage2_eps}, set())
        self.assertEqual(
            {t for _, _, t, _ in stage1_eps} | {t for _, _, t, _ in stage2_eps},
            {t.id for t in self.tasks},
        )
        # Final gate ran fresh contemporary paired episodes (incumbent + finalist 'a' on all 4 tasks)
        self.assertEqual(len(final_eps), 8)
        self.assertEqual(sum(role == "incumbent" for role, *_ in final_eps), 4)
        self.assertEqual(sum(role == "candidate" and name == "a" for role, name, *_ in final_eps), 4)

        # Ledger fold, evolver_sessions accounting, per-stage evolver API, and adoption
        final_view = evo.view("evo:relay")
        self.assertEqual(final_view["incumbent"], "h_r2a")
        round1 = final_view["rounds"][-1]
        self.assertTrue(round1["finished"])
        self.assertEqual(round1["adopted"], "a")
        self.assertEqual(len(round1["relay_stages"]), 2)
        self.assertEqual(round1["relay_stages"][0]["leader"]["label"], "g1r1b")
        self.assertEqual(round1["relay_stages"][1]["leader"]["label"], "g1r2a")
        self.assertTrue(any(c.get("carried_leader") and c["digest"] == "h_r1b"
                            for c in round1["relay_stages"][1]["candidates"]))
        self.assertEqual(round1["candidates"][0]["relay_origin"]["label"], "g1r2a")
        self.assertEqual(len(evolver_sessions(final_view)), 2)

        svc = Service(self.ledger_path)
        s1_view = svc.evolver_session("evo:relay", "1", steps="0", stage="1")
        s2_view = svc.evolver_session("evo:relay", "1", steps="0", stage="2")
        self.assertEqual(s1_view["stage"], 1)
        self.assertEqual(s2_view["stage"], 2)
        self.assertTrue(any(c.get("status") == "carried" for c in s2_view["candidates"]))

        # Evidence overview includes relay episodes
        fake_records = {
            ep["episode"]: {"scored": {"score": ep["score"], "cells": []}}
            for st in round1["relay_stages"] for ep in st["episodes"]
        }
        ov = overview(final_view, fake_records, [t.id for t in self.tasks],
                      {"seed": "seed", "h_r1b": "g1r1b", "h_r2a": "g1r2a"})
        self.assertIn("g1r1b", ov)

    def test_relay_resume_after_stage1_reuses_stage1_from_ledger(self):
        config = EvolveConfig(
            suite="synthetic.relay", generations=1, candidates=1, repeats=1,
            concurrency=2, relay_stages=2, relay_candidates=2, relay_batch=2,
        )
        self._seed_gen0(config)
        with patch("ecogym.evolution.loop.suite_spec", return_value=self.suite):
            evo = Evolution(config, self.ledger_path)

        seen_stages: list[str] = []

        def interrupt_stage2(**kwargs):
            tag = kwargs.get("stage_tag") or ""
            seen_stages.append(tag)
            if tag == "-r2" and len(seen_stages) == 2:
                raise Cancelled("interrupted before stage 2")
            if tag == "-r1":
                return {
                    "session": {"status": "finished", "seconds": 5.0, "tokens": 1000, "metered_tokens": 500,
                                "llm_calls": 1, "answered_calls": 1, "failed_calls": 0, "final_text": "s1"},
                    "candidates": [
                        {"name": "a", "status": "proposed", "ok": True, "digest": "h_r1b",
                         "harness": self.harnesses["h_r1b"], "summary": "r1b", "proposal": "p" * 100,
                         "agent": asdict(self.harnesses["h_r1b"].agent), "changes": [], "problems": [], "warnings": []},
                    ],
                }
            return {
                "session": {"status": "finished", "seconds": 5.0, "tokens": 1000, "metered_tokens": 500,
                            "llm_calls": 1, "answered_calls": 1, "failed_calls": 0, "final_text": "s2"},
                "candidates": [
                    {"name": "a", "status": "proposed", "ok": True, "digest": "h_r2a",
                     "harness": self.harnesses["h_r2a"], "summary": "r2a", "proposal": "p" * 100,
                     "agent": asdict(self.harnesses["h_r2a"].agent), "changes": [], "problems": [], "warnings": []},
                ],
            }

        def fake_episode(run_id, generation, harness, role, name, task, replicate, snapshot, plan_id,
                         first_attempt, *, relay_stage=None):
            return {
                "episode": f"ep:{role}-{name}-{task.id}-{relay_stage}", "task": task.id, "harness": harness.digest,
                "role": role, "candidate": name, "replicate": replicate, "guidance": snapshot.guidance[task.id],
                "status": "finished", "score": self.scores[harness.digest],
                "predictions": str(self.root / "p.npz"), "predictions_digest": "sha",
                "tokens": 100, "metered_tokens": 50, "seconds": 1.0,
            }

        def fake_compare(pairs, gates, meta, confidence=0.95):
            return SuiteDecision(
                adopted=True, reason="improved", delta=0.09, interval=(0.06, 0.12),
                confidence=confidence, task_confidence=0.95, basis="tasks_t",
                tasks=tuple({"task": p.task, "delta": 0.09, "candidate": 0.19, "incumbent": 0.10,
                             "interval": [0.06, 0.12], "replicates": [1, 1], "reason": "ok"} for p in pairs),
                cells={},
            )

        with patch("ecogym.evolution.loop.resolve", side_effect=self.harnesses.get), \
                patch("ecogym.evolution.loop.behavior_digest", side_effect=lambda h: f"beh:{h.digest}"), \
                patch("ecogym.evolution.loop.run_evolver", side_effect=interrupt_stage2), \
                patch("ecogym.evolution.loop.compare_suite", side_effect=fake_compare), \
                patch.object(evo, "_measurement", return_value="m1"), \
                patch.object(evo, "_episode", side_effect=fake_episode), \
                patch.object(evo, "_evaluation", return_value=object()), \
                patch.object(evo, "_record_template_attribution"):
            view = evo.view("evo:relay")
            with self.assertRaises(Cancelled):
                evo._round("evo:relay", view, evo.state(view))
            # Resume: Stage 1 must NOT be called again!
            resumed_view = evo.view("evo:relay")
            self.assertEqual(len(resumed_view["rounds"]), 2)
            self.assertEqual(len(resumed_view["rounds"][-1]["relay_stages"]), 1)
            evo._round("evo:relay", resumed_view, evo.state(resumed_view))

        # Called: -r1 once, -r2 (interrupted), -r2 (resumed) -> never re-ran -r1!
        self.assertEqual(seen_stages, ["-r1", "-r2", "-r2"])
        final_view = evo.view("evo:relay")
        self.assertEqual(len(final_view["rounds"]), 2)
        self.assertEqual(final_view["incumbent"], "h_r2a")

    def test_relay_all_worse_early_exit_skips_final_gate(self):
        config = EvolveConfig(
            suite="synthetic.relay", generations=1, candidates=1, repeats=1,
            concurrency=2, relay_stages=2, relay_candidates=2, relay_batch=2, early_prune=True,
        )
        self._seed_gen0(config)
        with patch("ecogym.evolution.loop.suite_spec", return_value=self.suite):
            evo = Evolution(config, self.ledger_path)

        worse_scores = {"seed": 0.10, "h_r1a": 0.02, "h_r1b": 0.03, "h_r2a": 0.01, "h_r2b": 0.04}
        episode_calls: list[int | None] = []

        def fake_run_evolver(**kwargs):
            tag = kwargs.get("stage_tag") or ""
            keys = ("h_r1a", "h_r1b") if tag == "-r1" else ("h_r2a", "h_r2b")
            return {
                "session": {"status": "finished", "seconds": 5.0, "tokens": 1000, "metered_tokens": 500,
                            "llm_calls": 1, "answered_calls": 1, "failed_calls": 0, "final_text": "bad"},
                "candidates": [
                    {"name": name, "status": "proposed", "ok": True, "digest": key,
                     "harness": self.harnesses[key], "summary": key, "proposal": "p" * 100,
                     "agent": asdict(self.harnesses[key].agent), "changes": [], "problems": [], "warnings": []}
                    for name, key in zip(("a", "b"), keys, strict=True)
                ],
            }

        def fake_episode(run_id, generation, harness, role, name, task, replicate, snapshot, plan_id,
                         first_attempt, *, relay_stage=None):
            episode_calls.append(relay_stage)
            return {
                "episode": f"ep:{role}-{name}-{task.id}-{relay_stage}", "task": task.id, "harness": harness.digest,
                "role": role, "candidate": name, "replicate": replicate, "guidance": snapshot.guidance[task.id],
                "status": "finished", "score": worse_scores[harness.digest],
                "predictions": str(self.root / "p.npz"), "predictions_digest": "sha",
                "tokens": 100, "metered_tokens": 50, "seconds": 1.0,
            }

        with patch("ecogym.evolution.loop.resolve", side_effect=self.harnesses.get), \
                patch("ecogym.evolution.loop.behavior_digest", side_effect=lambda h: f"beh:{h.digest}"), \
                patch("ecogym.evolution.loop.run_evolver", side_effect=fake_run_evolver), \
                patch.object(evo, "_measurement", return_value="m1"), \
                patch.object(evo, "_episode", side_effect=fake_episode), \
                patch.object(evo, "_evaluation", return_value=object()), \
                patch.object(evo, "_record_template_attribution"):
            view = evo.view("evo:relay")
            evo._round("evo:relay", view, evo.state(view))

        # Final gate (relay_stage=None) must be skipped completely!
        self.assertNotIn(None, episode_calls)
        final_view = evo.view("evo:relay")
        round1 = final_view["rounds"][-1]
        self.assertTrue(round1["finished"])
        self.assertEqual(round1["outcome"], "relay_no_viable_candidate")
        self.assertEqual(final_view["incumbent"], "seed")

    def test_config_validation_and_service_catalog_defaults(self):
        with self.assertRaises(ValueError):
            EvolveConfig(suite="s", relay_stages=0)
        with self.assertRaises(ValueError):
            EvolveConfig(suite="s", relay_stages=4)
        with self.assertRaises(ValueError):
            EvolveConfig(suite="s", relay_batch=-1)
        with self.assertRaises(ValueError):
            EvolveConfig(suite="s", relay_candidates=7)
        restored = EvolveConfig.from_dict({"suite": "s"})
        self.assertEqual((restored.relay_stages, restored.relay_batch, restored.relay_candidates), (1, 0, 0))
        service = Service(self.ledger_path)
        defaults = service.catalog()["evolve_defaults"]
        self.assertEqual(defaults["relay_stages"], 1)
        self.assertEqual(defaults["relay_batch"], 0)
        self.assertEqual(defaults["relay_candidates"], 0)


if __name__ == "__main__":
    unittest.main()
