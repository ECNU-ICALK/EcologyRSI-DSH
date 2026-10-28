"""ecogym command line.

    ecogym list                                 datasets, tasks (with presets), solvers
    ecogym describe TASK                        partitions, origins, baselines
    ecogym evaluate TASK --preset expert        score a method on a development stage
    ecogym run TASK --preset expert             arena run (sweep or random proposer)
    ecogym runs | show RUN | replay RUN         inspect and re-execute runs
    ecogym seal RUN --stage validation          sealed evaluation of a run's final method
    ecogym report RUN --out run.html            static HTML report
    ecogym verify                               check the ledger hash chain

Runs and sealed exposures go to one ledger, ``$ECOGYM_HOME/ledger.sqlite3``
(``~/.ecogym`` by default) unless ``--ledger`` names another file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .arena import Arena, RunConfig, fold, make_proposer, replay, seal
from .arena.proposers import PROPOSERS
from .data.catalog import data_root, dataset_specs
from .domains import load_domains
from .evaluation.gates import objective_interval
from .evaluation.harness import Bench
from .kernel.canonical import jsonable
from .kernel.ledger import Ledger
from .report import render_run
from .solvers.base import Genome, solver_specs
from .task.spec import task_spec, task_specs


def default_ledger() -> Path:
    home = Path(os.environ.get("ECOGYM_HOME") or Path.home() / ".ecogym").expanduser()
    return home / "ledger.sqlite3"


def _print_json(value: Any) -> None:
    print(json.dumps(jsonable(value), indent=2, ensure_ascii=False))


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    text = [[str(item) for item in row] for row in rows]
    widths = [max(len(str(header)), *(len(row[index]) for row in text)) for index, header in enumerate(headers)]
    line = "  ".join(str(header).ljust(width) for header, width in zip(headers, widths, strict=True))
    body = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)) for row in text]
    return "\n".join([line, "  ".join("-" * width for width in widths), *body])


def _genome(bench: Bench, args: argparse.Namespace) -> Genome:
    if getattr(args, "genome", None):
        path = Path(args.genome)
        body = tomllib.loads(path.read_text()) if path.suffix == ".toml" else json.loads(path.read_text())
        return bench.genome(body)
    return bench.preset(args.preset)


def _cells(cells: Sequence[dict[str, Any]]) -> str:
    return _table(
        ["cell", "n", "baseline RMSE", "RMSE", "skill", "bias", "clipped"],
        [[c["cell"], c["n"], f"{c['baseline_rmse']:.4f}", f"{c['rmse']:.4f}", f"{c['skill']:+.4f}",
          f"{c['bias']:+.4f}", c["clipped"]] for c in cells],
    )


def cmd_list(args: argparse.Namespace) -> int:
    datasets = [{"id": d.id, "title": d.title, "step": d.step, "path": d.path} for d in dataset_specs()]
    tasks = [
        {"id": t.id, "dataset": t.dataset, "cells": len(t.cells), "presets": sorted(t.presets)} for t in task_specs()
    ]
    solvers = [s.describe() for s in solver_specs()]
    if args.json:
        _print_json({"data_root": str(data_root()), "datasets": datasets, "tasks": tasks, "solvers": solvers})
        return 0
    print(f"data root: {data_root()}\n")
    print(_table(["dataset", "step", "title"], [[d["id"], d["step"], d["title"]] for d in datasets]), "\n")
    print(_table(["task", "dataset", "cells", "presets"],
                 [[t["id"], t["dataset"], t["cells"], ", ".join(t["presets"])] for t in tasks]), "\n")
    print(_table(["solver", "title", "parameters"],
                 [[s["ref"], s["title"], ", ".join(s["params"])] for s in solvers]))
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    bench = Bench.open(args.task)
    stages = {}
    for stage, by_episode in bench.partitions.summary().items():
        sealed = stage in ("validation", "final_test")
        stages[stage] = {"episodes": by_episode, "eligible_origins": None if sealed else bench.stage(stage).size}
    body = {"task": bench.task.id, "bench": bench.identity, "stages": stages,
            "baselines": bench.baselines.kinds, "scales": bench.scales}
    if args.json:
        _print_json(body)
        return 0
    print(f"{bench.task.title}\nbench {bench.digest[:16]}\n")
    rows = []
    for stage, item in stages.items():
        for episode, part in item["episodes"].items():
            rows.append([stage, episode, part["start"], part["end"], part["rows"],
                         "sealed" if item["eligible_origins"] is None else item["eligible_origins"]])
    print(_table(["stage", "episode", "start", "end", "rows", "eligible origins"], rows), "\n")
    print(_table(["cell", "baseline"], list(bench.baselines.kinds.items())))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    bench = Bench.open(args.task)
    genome = _genome(bench, args)
    evaluation = bench.evaluate(genome, args.stage)
    low, high = objective_interval(evaluation, None, bench.task.gates, {"evaluate": genome.digest})
    body = {"genome": genome.to_dict(), "digest": genome.digest, "stage": args.stage, "origins": evaluation.stage.size,
            "score": evaluation.score(), "interval": [low, high], "cells": evaluation.cells()}
    if args.json:
        _print_json(body)
        return 0
    print(f"{genome.solver}  {genome.digest[:12]}  on {args.stage} ({evaluation.stage.size} origins)\n")
    print(_cells(body["cells"]))
    print(f"\noverall skill {body['score']:+.4f}  95% day-block interval [{low:+.4f}, {high:+.4f}]")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    bench = Bench.open(args.task)
    seed = _genome(bench, args)
    config = RunConfig(generations=args.generations, candidates=args.candidates, seed=args.seed,
                       max_step=args.max_step, max_changes=args.max_changes)
    with Ledger(args.ledger) as ledger:
        arena = Arena(bench, ledger, config, make_proposer(args.proposer))
        result = arena.run(seed, label=args.label)
        view = fold(ledger.events(result["run_id"]))
    if args.json:
        _print_json(result)
        return 0
    print(f"{result['run_id']}  ({result['seconds']} s, ledger {args.ledger})")
    for generation in view["generations"]:
        comparison = generation["comparison"]
        outcome = comparison["decision"]["reason"] if comparison else "no finalist"
        delta = f"{comparison['decision']['delta']:+.4f}" if comparison else ""
        print(f"  generation {generation['generation']}: {len(generation['proposed'])} candidates, "
              f"{len(generation['rejected'])} rejected -> {outcome} {delta}")
    print(f"seed {result['seed_score']:+.4f} -> final {result['incumbent_score']:+.4f} "
          f"({result['promotions']} promotions); final method {result['incumbent_digest'][:12]}")
    return 0


def _run_view(ledger: Ledger, run_id: str) -> dict[str, Any]:
    matches = [run for run in ledger.runs() if run == run_id or run.startswith(run_id) or run == f"run:{run_id}"]
    if len(matches) != 1:
        raise SystemExit(f"run {run_id!r} matches {len(matches)} runs in {ledger.path}")
    return fold(ledger.events(matches[0]))


def cmd_runs(args: argparse.Namespace) -> int:
    with Ledger(args.ledger) as ledger:
        runs = [fold(ledger.events(run_id)) for run_id in ledger.runs()]
    rows = []
    for run in runs:
        result = run.get("result", {})
        score = f"{result['seed_score']:+.4f} -> {result['incumbent_score']:+.4f}" if result else ""
        rows.append([run["run_id"], run["started_at"][:19], run["task_id"], run["proposer"]["name"], run["status"],
                     score, len(run["sealed"])])
    print(_table(["run", "started", "task", "proposer", "status", "score", "sealed"], rows) if rows else "no runs")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    with Ledger(args.ledger) as ledger:
        _print_json(_run_view(ledger, args.run))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    with Ledger(args.ledger) as ledger:
        run = _run_view(ledger, args.run)
        result = replay(ledger, run["run_id"], Bench.open(run["task_id"]))
    _print_json(result)
    return 0 if result["identical"] else 1


def cmd_seal(args: argparse.Namespace) -> int:
    with Ledger(args.ledger) as ledger:
        run = _run_view(ledger, args.run)
        if run["status"] != "finished":
            raise SystemExit(f"{run['run_id']} has not finished")
        bench = Bench.open(run["task_id"])
        genome = bench.genome(run["result"]["incumbent"])
        report = seal(bench, ledger, genome, args.stage, run_id=run["run_id"], label=args.label)
    if args.json:
        _print_json(report)
        return 0
    low, high = report["interval"]
    print(f"{args.stage}: {genome.digest[:12]} on {report['origins']} origins\n")
    print(_cells(report["cells"]))
    print(f"\noverall skill {report['score']:+.4f}  95% day-block interval [{low:+.4f}, {high:+.4f}]")
    print(f"methods evaluated on this {args.stage} so far: {report['methods_exposed_on_stage']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    with Ledger(args.ledger) as ledger:
        run = _run_view(ledger, args.run)
    page = render_run(run, Bench.open(run["task_id"]))
    out = Path(args.out or f"{run['run_id'].replace(':', '-')}.html")
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    with Ledger(args.ledger) as ledger:
        print(f"{ledger.verify()} events verified in {args.ledger}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecogym", description="Ecological forecasting training ground")
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, handler, help_text: str, *, task: bool = False, ledger: bool = False, method: bool = False):
        sub = commands.add_parser(name, help=help_text)
        sub.set_defaults(handler=handler)
        sub.add_argument("--json", action="store_true", help="machine-readable output")
        if task:
            sub.add_argument("task", help="task id (see `ecogym list`)")
        if ledger:
            sub.add_argument("--ledger", type=Path, default=default_ledger(), help="ledger file")
        if method:
            group = sub.add_mutually_exclusive_group(required=True)
            group.add_argument("--preset", help="seed method declared in the task card")
            group.add_argument("--genome", help="JSON or TOML file holding {solver, params}")
        return sub

    command("list", cmd_list, "datasets, tasks and solvers")
    command("describe", cmd_describe, "partitions, origins and baselines of a task", task=True)
    evaluate = command("evaluate", cmd_evaluate, "score a method on a development stage", task=True, method=True)
    evaluate.add_argument("--stage", default="model_selection", choices=("model_selection", "calibration_uq"))
    run = command("run", cmd_run, "run the arena from a seed method", task=True, ledger=True, method=True)
    run.add_argument("--proposer", default="sweep", choices=sorted(PROPOSERS))
    run.add_argument("--generations", type=int, default=3)
    run.add_argument("--candidates", type=int, default=4)
    run.add_argument("--seed", type=int, default=0, help="run seed (cohorts, random proposals)")
    run.add_argument("--max-step", type=float, default=0.15, help="trust region per coordinate")
    run.add_argument("--max-changes", type=int, default=3, help="coordinates an edit may change")
    run.add_argument("--label", default="")
    command("runs", cmd_runs, "list runs in the ledger", ledger=True)
    for name, handler, text in (("show", cmd_show, "a run's folded record"),
                                ("replay", cmd_replay, "re-execute a run and compare its events")):
        command(name, handler, text, ledger=True).add_argument("run")
    seal_parser = command("seal", cmd_seal, "sealed evaluation of a run's final method", ledger=True)
    seal_parser.add_argument("run")
    seal_parser.add_argument("--stage", required=True, choices=("validation", "final_test"))
    seal_parser.add_argument("--label", default="")
    report = command("report", cmd_report, "static HTML report of a run", ledger=True)
    report.add_argument("run")
    report.add_argument("--out")
    command("verify", cmd_verify, "check the ledger hash chain", ledger=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_domains()
    try:
        if getattr(args, "task", None):
            task_spec(args.task)  # fail early with the list of known tasks
        return int(args.handler(args) or 0)
    except (KeyError, ValueError, PermissionError, FileNotFoundError) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else error
        print(f"ecogym: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
