#!/usr/bin/env python3
"""Write an aggregate training-fit-only diagnostic without remote calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecologyrsi_dsh.data.registry import DatasetRegistry
from ecologyrsi_dsh.science.diagnostics import candidates_from_payload, run_training_fit_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="agc_cucumber_2018")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--data-root", type=Path, help="local AGC data root; otherwise use the standard registry environment")
    parser.add_argument("--candidates", type=Path, help="JSON list of {name, model_id, parameters}; defaults are explicit diagnostic presets, not imported run champions")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--initial-fit-fraction", type=float, default=.4)
    parser.add_argument("--purge-hours", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = candidates_from_payload(json.loads(args.candidates.read_text())) if args.candidates else None
    view = DatasetRegistry(data_root=args.data_root).selection_view(args.dataset, args.episode)
    report = run_training_fit_diagnostics(view, candidates=candidates, folds=args.folds,
                                          initial_fit_fraction=args.initial_fit_fraction,
                                          purge_hours=args.purge_hours)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "scope": report["scope"],
                      "fit_rows": report["source"]["fit_rows"],
                      "common_origins": [fold["common_origin_count"] for fold in report["folds"]],
                      "remote_requests": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
