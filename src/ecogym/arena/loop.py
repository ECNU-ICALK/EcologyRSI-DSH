"""The arena: generations of propose -> screen -> holdout -> promote.

Each generation:

1. the proposer suggests up to ``candidates`` edits of the incumbent;
2. edits are validated (solver schema, trust region); rejects are recorded;
3. every valid candidate is scored on the generation's screening cohort and
   the best one becomes the finalist if it beats the incumbent there;
4. finalist and incumbent are compared on the disjoint holdout cohort with
   the task's gates (``ecogym.evaluation.gates``);
5. a passing finalist becomes the incumbent.

All randomness derives from the run seed, every step is a ledger event, and
the host computes every forecast, so a run with a deterministic proposer can
be replayed event for event (``replay``).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .. import __version__
from ..evaluation.gates import compare
from ..evaluation.harness import Bench, Evaluation
from ..kernel.canonical import digest
from ..kernel.ledger import Ledger
from ..protocol.cohorts import plan_generation
from ..solvers.base import Genome, SolverError
from .genome import apply_edit, check_edit
from .proposers import ProposalContext, Proposer, make_proposer

EVENT_KINDS = frozenset({
    "run_started",
    "generation_started",
    "candidate_proposed",
    "candidate_rejected",
    "candidates_screened",
    "holdout_compared",
    "incumbent_promoted",
    "generation_finished",
    "run_finished",
    "run_failed",
    "sealed_evaluated",
    "agent_exchange",
})


@dataclass(frozen=True, slots=True)
class RunConfig:
    generations: int = 3
    candidates: int = 4
    seed: int = 0
    screen_fraction: float = 1 / 3
    chunk_blocks: int = 3
    max_step: float = 0.15
    max_changes: int = 3
    stage: str = "model_selection"

    def __post_init__(self) -> None:
        if self.generations < 1 or self.candidates < 1 or self.max_changes < 1:
            raise ValueError("generations, candidates and max_changes must be positive")
        if not 0 < self.max_step <= 1 or not 0 < self.screen_fraction < 1:
            raise ValueError("max_step must lie in (0, 1] and screen_fraction in (0, 1)")
        if self.stage not in ("model_selection", "calibration_uq"):
            raise ValueError("the arena selects on a development stage")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "ecogym.run-config/1", **asdict(self)}

    @classmethod
    def from_dict(cls, body: Mapping[str, Any]) -> RunConfig:
        return cls(**{key: value for key, value in body.items() if key != "schema"})


def _cells(evaluation: Evaluation, subset: np.ndarray) -> dict[str, float]:
    return {row["cell"]: row["objective"] for row in evaluation.cells(subset)}


class Arena:
    def __init__(self, bench: Bench, ledger: Ledger, config: RunConfig, proposer: Proposer) -> None:
        self.bench = bench
        self.ledger = ledger
        self.config = config
        self.proposer = proposer

    def _log(self, run_id: str, kind: str, body: Mapping[str, Any]) -> None:
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {kind}")
        self.ledger.append(run_id, kind, body)

    def run(self, seed: Genome, *, run_id: str | None = None, label: str = "") -> dict[str, Any]:
        run_id = run_id or f"run:{uuid.uuid4().hex[:12]}"
        started = time.monotonic()
        config, bench = self.config, self.bench
        self._log(run_id, "run_started", {
            "label": label,
            "config": config.to_dict(),
            "bench": bench.identity,
            "task_id": bench.task.id,
            "seed": seed.to_dict(),
            "proposer": {"name": self.proposer.name, "options": self.proposer.options()},
            "ecogym": __version__,
        })
        try:
            result = self._generations(run_id, seed)
        except Exception as error:
            self._log(run_id, "run_failed", {"error": f"{type(error).__name__}: {error}"})
            raise
        result["seconds"] = round(time.monotonic() - started, 3)
        self._log(run_id, "run_finished", result)
        return {"run_id": run_id, **result}

    def _generations(self, run_id: str, seed: Genome) -> dict[str, Any]:
        config, bench = self.config, self.bench
        stage = bench.stage(config.stage)
        incumbent = seed
        history: list[dict[str, Any]] = []
        for generation in range(config.generations):
            screening, holdout = plan_generation(
                stage.time, block_steps=bench.task.block_steps, chunk_blocks=config.chunk_blocks,
                screen_fraction=config.screen_fraction, seed=config.seed, generation=generation,
            )
            self._log(run_id, "generation_started", {
                "generation": generation,
                "incumbent": incumbent.digest,
                "screening": {"origins": screening.size, "blocks": list(screening.blocks), "digest": screening.digest},
                "holdout": {"origins": holdout.size, "blocks": list(holdout.blocks), "digest": holdout.digest},
            })
            context = ProposalContext(
                bench.task, incumbent, generation, config.seed, config.max_step, config.max_changes, tuple(history)
            )
            candidates: list[tuple[int, Genome, dict[str, Any]]] = []
            seen = {incumbent.digest}
            for index, edit in enumerate(self.proposer.propose(context, config.candidates)[: config.candidates]):
                try:
                    genome = apply_edit(incumbent, edit, bench.task)
                    problems = check_edit(incumbent, genome, max_step=config.max_step, max_changes=config.max_changes)
                    if genome.digest in seen:
                        problems.append("duplicates the incumbent or an earlier candidate")
                except SolverError as error:
                    genome, problems = None, [str(error)]
                if problems:
                    self._log(run_id, "candidate_rejected", {"generation": generation, "index": index,
                                                             "edit": edit.to_dict(), "problems": problems})
                    continue
                seen.add(genome.digest)
                candidates.append((index, genome, edit.to_dict()))
                self._log(run_id, "candidate_proposed", {"generation": generation, "index": index,
                                                         "edit": edit.to_dict(), "genome": genome.to_dict(),
                                                         "digest": genome.digest})
            report = self._judge(run_id, generation, incumbent, candidates, screening.index, holdout.index)
            if report["promoted"]:
                incumbent = next(genome for _, genome, _ in candidates if genome.digest == report["finalist"])
            history.append(report)
            self.proposer.observe(report)
            self._log(run_id, "generation_finished", {"generation": generation, "incumbent": incumbent.digest,
                                                      "promoted": report["promoted"]})
        seed_eval = bench.evaluate(seed, config.stage)
        final_eval = bench.evaluate(incumbent, config.stage)
        return {
            "incumbent": incumbent.to_dict(),
            "incumbent_digest": incumbent.digest,
            "seed_digest": seed.digest,
            "seed_score": seed_eval.score(),
            "incumbent_score": final_eval.score(),
            "promotions": sum(1 for item in history if item["promoted"]),
            "generations": len(history),
        }

    def _judge(
        self,
        run_id: str,
        generation: int,
        incumbent: Genome,
        candidates: list[tuple[int, Genome, dict[str, Any]]],
        screening: np.ndarray,
        holdout: np.ndarray,
    ) -> dict[str, Any]:
        bench, config = self.bench, self.config
        base = bench.evaluate(incumbent, config.stage)
        base_score = base.score(screening)
        base_cells = _cells(base, screening)
        scored = []
        for index, genome, edit in candidates:
            try:
                evaluation = bench.evaluate(genome, config.stage)
            except SolverError as error:
                scored.append({"index": index, "digest": genome.digest, "edit": edit, "error": str(error)})
                continue
            cells = _cells(evaluation, screening)
            scored.append({
                "index": index,
                "digest": genome.digest,
                "edit": edit,
                "score": evaluation.score(screening),
                "delta": evaluation.score(screening) - base_score,
                "cell_deltas": {key: cells[key] - base_cells[key] for key in cells},
            })
        self._log(run_id, "candidates_screened", {"generation": generation, "incumbent_score": base_score,
                                                  "candidates": scored})
        report: dict[str, Any] = {"generation": generation, "incumbent": incumbent.digest, "screening": scored,
                                  "incumbent_screening_score": base_score, "promoted": False, "finalist": None}
        valid = [item for item in scored if "score" in item]
        if not valid or max(item["delta"] for item in valid) <= 0:
            report["outcome"] = "no_finalist"
            return report
        finalist = max(valid, key=lambda item: (item["delta"], -item["index"]))
        genome = next(genome for _, genome, _ in candidates if genome.digest == finalist["digest"])
        decision = compare(
            bench.evaluate(genome, config.stage), base, holdout, bench.task.gates,
            {"run_seed": config.seed, "generation": generation},
        )
        body = {"generation": generation, "finalist": finalist["digest"], "incumbent": incumbent.digest,
                "decision": decision.to_dict()}
        self._log(run_id, "holdout_compared", body)
        report.update({"finalist": finalist["digest"], "decision": decision.to_dict(), "promoted": decision.promoted,
                       "outcome": decision.reason})
        if decision.promoted:
            self._log(run_id, "incumbent_promoted", {"generation": generation, "genome": genome.to_dict(),
                                                     "digest": genome.digest, "delta": decision.delta,
                                                     "interval": list(decision.interval)})
        return report


def replay(ledger: Ledger, run_id: str, bench: Bench) -> dict[str, Any]:
    """Re-execute a run with a deterministic proposer and diff its events."""

    events = ledger.events(run_id)
    if not events or events[0].kind != "run_started":
        raise ValueError(f"{run_id} has no run_started event")
    start = events[0].body
    if start["bench"] != bench.identity:
        raise ValueError("the bench differs from the one the run used (data, task or partitions changed)")
    proposer = make_proposer(start["proposer"]["name"], start["proposer"]["options"])
    seed = bench.genome(start["seed"])
    with Ledger(":memory:") as scratch:
        Arena(bench, scratch, RunConfig.from_dict(start["config"]), proposer).run(seed, run_id=run_id,
                                                                                   label=start.get("label", ""))
        fresh = scratch.events(run_id)

    def comparable(event) -> tuple[str, str]:
        body = {key: value for key, value in event.body.items() if key != "seconds"}
        return event.kind, digest(body)

    recorded = [comparable(event) for event in events if event.kind != "sealed_evaluated"]
    replayed = [comparable(event) for event in fresh]
    mismatches = [index for index, (a, b) in enumerate(zip(recorded, replayed, strict=False)) if a != b]
    return {"run_id": run_id, "events": len(recorded), "replayed": len(replayed),
            "identical": not mismatches and len(recorded) == len(replayed), "first_mismatch": mismatches[:1]}


__all__ = ["EVENT_KINDS", "Arena", "RunConfig", "replay"]
