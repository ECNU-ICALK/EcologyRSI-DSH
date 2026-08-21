#!/usr/bin/env python3
"""Run and replay the bounded EcologyRSI-DSH example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecologyrsi_dsh import EventLedger, EvolutionDirector, FakeDSHAdapter, TaskManifest, ToyCropSoilWater


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="SQLite ledger path")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("task-manifest.json"))
    parser.add_argument("--run-id", default="run:example")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = TaskManifest.from_dict(json.loads(args.manifest.read_text(encoding="utf-8")))
    dataset_seed = int(manifest.metadata.get("dataset_seed", 0))
    evaluator = ToyCropSoilWater(seed=dataset_seed)

    with EventLedger(args.db) as ledger:
        director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=manifest.max_candidates))
        director.start_evolution(manifest, run_id=args.run_id)
        for _ in range(manifest.max_candidates):
            candidate = director.propose_and_spawn(args.run_id)
            proposal = director.state(args.run_id).proposal(candidate.proposal_id)
            evaluation = evaluator.evaluate_candidate(args.run_id, candidate, proposal)
            director.evaluate_and_decide(evaluation)
        director.complete_run(args.run_id)
        first = director.state(args.run_id)

    with EventLedger(args.db) as ledger:
        replayed = EvolutionDirector(ledger).replay(args.run_id)

    if first.run.to_dict() != replayed.run.to_dict() or len(first.events) != len(replayed.events):
        raise RuntimeError("replayed projection differs from the original projection")

    print(
        json.dumps(
            {
                "run_id": replayed.run.run_id,
                "status": replayed.run.status.value,
                "manifest_digest": replayed.task_manifest.digest,
                "candidate_count": len(replayed.candidates),
                "evaluation_count": len(replayed.evaluations),
                "event_count": len(replayed.events),
                "best_candidate_id": replayed.run.best_candidate_id,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
