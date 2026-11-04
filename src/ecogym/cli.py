"""ecogym command line.

DeepSeek Harness plugin (the user interface):

    ecogym dsh install [--profile web]          DSH starts ecogym and shows a 生态进化 panel
    ecogym dsh status | uninstall               what is installed where; remove it again

Agent harnesses (inner and outer loop):

    ecogym setup                                agent runtime (Python with the scientific stack)
    ecogym suites                               task suites: training and holdout tasks
    ecogym models                               model routes: DeepSeek and the providers in DSH's settings
    ecogym episode run TASK [--harness H]       one sandboxed agent episode, judged on the hidden stage
    ecogym episode list | show EP               recorded episodes
    ecogym evolve run --suite S                 evolve a harness on the suite's training tasks
    ecogym evolve run --resume EVO              continue an interrupted evolution run
    ecogym evolve list | show EVO               evolution runs: rounds, candidates, decisions
    ecogym evolve verify EVO --holdout          one-time check on the holdout tasks
    ecogym evolve verify EVO --sealed validation   one-time check on a sealed period
    ecogym harness show H | diff A B            inspect harness versions
    ecogym export --out DIR                     SFT threads, preference pairs, notes, episode index
    ecogym expert list | answer Q TEXT | note SCOPE TEXT   expert participation

Method arena (parameter-level baselines):

    ecogym list                                 datasets, tasks (with presets), solvers
    ecogym describe TASK                        partitions, origins, baselines
    ecogym evaluate TASK --preset expert        score a method on a development stage
    ecogym run TASK --preset expert             arena run (sweep, random or LLM proposer)
    ecogym runs | show RUN | replay RUN         inspect and re-execute runs
    ecogym seal RUN --stage validation          sealed evaluation of a run's final method
    ecogym report RUN --out run.html            static HTML report
    ecogym verify                               check the ledger hash chain

Everything goes to one ledger, ``$ECOGYM_HOME/ledger.sqlite3`` (``~/.ecogym``
by default) unless ``--ledger`` names another file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import agents as _agents  # noqa: F401  (registers the llm proposer)
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


def _options(pairs: Sequence[str]) -> dict[str, Any]:
    """``key=value`` pairs; values are parsed as JSON when they can be (numbers, lists, true)."""

    options: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--option needs key=value, got {pair!r}")
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value
    return options


def cmd_run(args: argparse.Namespace) -> int:
    bench = Bench.open(args.task)
    seed = _genome(bench, args)
    config = RunConfig(generations=args.generations, candidates=args.candidates, seed=args.seed,
                       max_step=args.max_step, max_changes=args.max_changes)
    with Ledger(args.ledger) as ledger:
        arena = Arena(bench, ledger, config, make_proposer(args.proposer, _options(args.option)))
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
    matches = [run for run in ledger.runs(first="run_started")
               if run == run_id or run.startswith(run_id) or run == f"run:{run_id}"]
    if len(matches) != 1:
        raise SystemExit(f"run {run_id!r} matches {len(matches)} runs in {ledger.path}")
    return fold(ledger.events(matches[0]))


def cmd_runs(args: argparse.Namespace) -> int:
    with Ledger(args.ledger) as ledger:
        runs = [fold(ledger.events(run_id)) for run_id in ledger.runs(first="run_started")]
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


def cmd_serve(args: argparse.Namespace) -> int:
    """The JSON API; the DSH plugin starts it with ``--port 0 --exit-with-stdin`` and reads the port back."""

    import signal
    import threading

    from .service import Service, make_server

    service = Service(args.ledger, token=args.token or os.environ.get("ECOGYM_SERVICE_TOKEN") or None)
    server = make_server(service, args.host, args.port)
    print(f"ecogym serving http://{args.host}:{server.server_address[1]}/  (ledger {args.ledger}; Ctrl-C to stop)",
          flush=True)
    stopping = threading.Event()  # the first signal or end of stdin starts the shutdown; later ones wait for it

    def interrupt(*_: Any) -> None:
        if not stopping.is_set():
            stopping.set()
            raise KeyboardInterrupt

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):  # Ctrl-C twice or a closed terminal included
        signal.signal(signum, interrupt)
    if args.exit_with_stdin:  # the parent holds the pipe open; EOF means it is gone
        def watch() -> None:
            try:  # the raw descriptor: a thread blocked in a buffered read would hold its lock at exit
                while os.read(sys.stdin.fileno(), 4096):
                    pass
            except (OSError, ValueError):
                pass
            if not stopping.is_set():
                stopping.set()
                server.shutdown()

        threading.Thread(target=watch, name="stdin watch", daemon=True).start()
    threading.Thread(target=service.warm, name="warm up", daemon=True).start()
    if not args.no_auto_resume:
        threading.Thread(target=service.auto_resume, name="auto resume", daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stopping.set()
        server.server_close()
        if not service.close():  # a step outlived the deadline; close() recorded its run as stopped
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)  # do not wait for it: the interpreter would join the episode workers first
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    with Ledger(args.ledger) as ledger:
        print(f"{ledger.verify()} events verified in {args.ledger}")
    return 0


RUNTIME_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels")


def cmd_setup(args: argparse.Namespace) -> int:
    """Create the agent runtime: a Python venv with the scientific stack, outside any project."""

    import shutil
    import subprocess

    from .episodes.sandbox import runtime_python

    python = runtime_python()
    venv = python.parent.parent
    if python.exists() and not args.force:
        print(f"agent runtime already at {venv} (use --force to rebuild)")
    else:
        uv = shutil.which("uv")
        if uv:
            subprocess.run([uv, "venv", "-q", "--python", "3.12", str(venv), *(["--clear"] if args.force else [])],
                           check=True)
            subprocess.run([uv, "pip", "install", "-q", "--python", str(python), *RUNTIME_PACKAGES], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", *(["--clear"] if args.force else []), str(venv)], check=True)
            subprocess.run([str(python), "-m", "pip", "install", "-q", *RUNTIME_PACKAGES], check=True)
        print(f"agent runtime ready at {venv}")
    dsh = shutil.which("dsh")
    print(f"dsh: {dsh or 'not found (install DeepSeek Harness, or set ECOGYM_DSH_BIN)'}")
    return 0


def cmd_dsh_install(args: argparse.Namespace) -> int:
    from .web import install, status

    env = {}
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep or not key.isidentifier():
            raise ValueError(f"--env takes KEY=VALUE, not {item!r}")
        env[key] = value
    try:
        result = install(profile=args.profile, home=args.home, ledger=args.ledger, env=env,
                         keep_legacy=args.keep_legacy)
    except (RuntimeError, OSError) as error:
        print(f"ecogym: {error}", file=sys.stderr)
        return 2
    check = status(profile=args.profile)
    if args.json:
        _print_json({**result, "status": check})
        return 0
    print(f"@ecogym/dsh-plugin is a layer of DSH profile {args.profile} ({result['profile_dir']})\n"
          f"  DSH starts: {result['python']} -m ecogym serve   (ECOGYM_HOME {result['home']})\n"
          f"  settings:   {result['settings']}")
    for key, value in result["env"].items():
        print(f"              {key}={value}")
    if result["legacy_backup"]:
        print(f"  removed the EcologyRSI plugin block from cordis.patch.yml (backup {result['legacy_backup']})")
    for note in result.get("notes") or []:
        print(f"  note: {note}")
    if not check["agent_runtime_found"]:
        print(f"  warning: no agent runtime at {check['agent_runtime']}; run `ecogym setup` "
              "(or pass --env ECOGYM_RUNTIME_PYTHON=...)")
    print(f"\nstart DSH with `dsh --profile {args.profile}`{' (or `dsh web`)' if args.profile == 'web' else ''} "
          "and open 生态进化 in the sidebar")
    return 0


def cmd_dsh_uninstall(args: argparse.Namespace) -> int:
    from .web import uninstall

    try:
        result = uninstall(profile=args.profile)
    except (RuntimeError, OSError) as error:
        print(f"ecogym: {error}", file=sys.stderr)
        return 2
    if args.json:
        _print_json(result)
        return 0
    print(f"{'removed @ecogym/dsh-plugin from' if result['removed'] else 'no @ecogym/dsh-plugin in'} "
          f"{result['profile']}" + (f"; deleted {result['settings_removed']}" if result["settings_removed"] else "")
          + (f"; kept the settings, they configure profile {result['settings_kept_for']}"
             if result.get("settings_kept_for") else ""))
    return 0


def cmd_dsh_status(args: argparse.Namespace) -> int:
    from .web import status

    result = status(profile=args.profile)
    if args.json:
        _print_json(result)
        return 0
    width = max(len(key) for key in result)
    for key, value in result.items():
        print(f"{key:<{width}}  {value}")
    return 0


def cmd_dsh_autostart(args: argparse.Namespace) -> int:
    from .web import autostart, autostart_log, autostart_status

    if args.action == "status":
        result = autostart_status(profile=args.profile)
    else:
        log = autostart_log(args.profile)
        before = log.stat().st_size if log.is_file() else 0
        try:
            result = autostart(profile=args.profile, enable=args.action == "on")
        except (RuntimeError, OSError) as error:
            print(f"ecogym: {error}", file=sys.stderr)
            return 2
        if args.action == "on":
            deadline = time.monotonic() + args.wait
            while True:  # the address DSH prints once it listens, with this start's token
                state = autostart_status(profile=args.profile, since=before)
                if state["url"] or time.monotonic() > deadline:
                    break
                time.sleep(2)
            result["status"] = state
    if args.json:
        _print_json(result)
        return 0
    if args.action == "off":
        print(f"{result['label']}: {'stopped' if result['stopped'] else 'was not loaded'}; "
              f"{'removed ' + result['plist'] if result['removed'] else 'no job file'}")
        return 0
    state = result.get("status") or result
    if args.action == "on":
        print(f"launchd job {result['label']} ({result['plist']}) starts {' '.join(result['program'])} at login "
              f"and after a crash")
        for note in result.get("notes") or []:
            print(f"  note: {note}")
    elif not state["plist"]:
        print(f"no launchd job for profile {args.profile}; `ecogym dsh autostart on` adds it")
        return 0
    else:
        print(f"launchd job {state['label']} ({state['plist']}){'' if state['loaded'] else ', not loaded'}")
    print(f"  {'running, pid ' + str(state['pid']) if state['pid'] else 'not running'}"
          f"{'; last exit ' + str(state['last_exit']) if state.get('last_exit') else ''} (log {state['log']})")
    if state["url"] and state["pid"]:
        print(f"  open {state['url']}")
    elif state["url"]:
        print(f"  last address (DSH is not running now): {state['url']}")
    elif args.action == "on":
        print(f"  no web address yet; see {state['log']}")
    if state.get("service"):
        print(f"  {state['service']}")
    return 0


def cmd_episode_run(args: argparse.Namespace) -> int:
    from .episodes.runner import EpisodeConfig, run_episode

    config = EpisodeConfig(task=args.task, harness=args.harness, minutes=args.minutes, tokens=args.tokens,
                           model=args.model, label=args.label, keep=args.keep)
    print(f"episode on {args.task} with harness {args.harness}, {args.minutes:g} min; this runs the agent end to end")
    result = run_episode(config, args.ledger)
    if args.json:
        _print_json(result)
        return 0
    _print_episode(result)
    return 0


def _print_episode(result: dict[str, Any]) -> None:
    scored = result.get("scored")
    print(f"\n{result['episode']}  {result['status']}  {result['seconds']:.0f} s  {result['tokens']:,} tokens "
          f"({result['metered_tokens']:,} metered)  {result['llm_calls']} model calls  "
          f"{result['submissions']} submissions")
    if scored:
        low, high = scored["interval"]
        print(f"\nhidden {scored['stage']}: {scored['score']:+.4f}  95% day-block interval [{low:+.4f}, {high:+.4f}]  "
              f"coverage {scored['coverage']:.0%}  ({scored['status']}{', auto-submitted' if scored['auto'] else ''}"
              f"{', the starting program' if scored.get('start') else ''})")
        print("references on the same origins: "
              + ", ".join(f"{name} {value:+.4f}" for name, value in scored["references"].items()))
        print(_cells(scored["cells"]))
    print(f"\ntrajectory: {result['trajectory']['dir']}")


def cmd_episode_list(args: argparse.Namespace) -> int:
    from .episodes.history import episode_records, summary

    with Ledger(args.ledger) as ledger:
        episodes = [summary(record) for record in episode_records(ledger)]
    if args.json:
        _print_json(episodes)
        return 0
    rows = []
    for item in episodes:
        role = item["context"].get("role", "")
        rows.append([item["episode"], item["started_at"][:16].replace("T", " "), item["task_id"],
                     (item["harness"].get("digest") or "")[:12], item["model"], item["status"],
                     f"{item['score']:+.4f}" if isinstance(item["score"], float) else "", role, item["label"]])
    print(_table(["episode", "started", "task", "harness", "model", "status", "hidden score", "role", "label"], rows)
          if rows else "no episodes in this ledger")
    return 0


def cmd_episode_show(args: argparse.Namespace) -> int:
    with Ledger(args.ledger) as ledger:
        matches = [run for run in ledger.runs(first="episode_started") if run.startswith(args.episode)
                   or run == f"ep:{args.episode}"]
        if len(matches) != 1:
            raise SystemExit(f"episode {args.episode!r} matches {len(matches)} episodes in {ledger.path}")
        events = [{"seq": event.seq, "at": event.at, "kind": event.kind, "body": event.body}
                  for event in ledger.events(matches[0])]
    if args.json:
        _print_json(events)
        return 0
    for event in events:
        body = event["body"]
        detail = ""
        if event["kind"] in ("episode_evaluated", "episode_submitted"):
            detail = f"local {body['local']['score']:+.4f} ({body['local']['status']})"
        elif event["kind"] == "episode_note":
            detail = f"[{body['kind']}] {body['text'][:120]}"
        elif event["kind"] == "episode_scored":
            detail = f"hidden {body['score']:+.4f}"
        elif event["kind"] == "episode_finished":
            detail = f"{body['status']}, {body['seconds']:.0f} s, {body['tokens']:,} tokens"
        print(f"{event['at'][:19].replace('T', ' ')}  {event['kind']:<18} {detail}")
    return 0


def cmd_suites(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from .task.suite import suite_specs

    suites = [{"id": suite.id, "title": suite.title, "train": list(suite.train), "holdout": list(suite.holdout),
               "gates": asdict(suite.gates), "notes": suite.notes.strip()} for suite in suite_specs()]
    if args.json:
        _print_json(suites)
        return 0
    for suite in suites:
        print(f"{suite['id']}  {suite['title']}")
        print(f"  train   ({len(suite['train'])}): " + ", ".join(suite["train"]))
        print(f"  holdout ({len(suite['holdout'])}): " + ", ".join(suite["holdout"]))
        gates = suite["gates"]
        basis = ("tasks x replicates x days interval" if len(suite["train"]) < 3
                 else "signed-rank (Hodges-Lehmann) interval of the task gains" if gates["test"] == "signed_rank"
                 else "t interval of the task gains")
        print(f"  gates: mean gain > {gates['min_delta']}; {gates['confidence']:.0%} {basis} above 0 (Bonferroni "
              f"across a round's candidates); no task credibly losing more than {gates['task_cap']} (judged once "
              "both versions have 2+ episodes on it)\n")
    return 0 if suites else 1


def cmd_models(args: argparse.Namespace) -> int:
    from .episodes import providers

    body = providers.describe()
    if args.json:
        _print_json(body)
        return 0
    print(f"model routes ({body['settings']}; write --model <provider>/<model>):")
    for route in body["routes"]:
        state = "runnable" if route["runnable"] else f"not runnable: {route['problem']}"
        print(f"  {route['provider']}  {route['name']}  {route['api']}  {route['host']}  key {route['key_env']}  "
              f"[{state}]")
        for model in route["models"]:
            print(f"      {route['provider']}/{model}")
    if body["dsh_default"]:
        print(f"DSH's own default: {body['dsh_default'].get('provider')}/{body['dsh_default'].get('model')}")
    if body["error"]:
        print(f"note: {body['error']}")
    return 0


def _match(ledger: Ledger, ref: str, first: str, prefix: str) -> str:
    matches = [run for run in ledger.runs(first=first) if run == ref or run.startswith(ref) or run == f"{prefix}{ref}"]
    if len(matches) != 1:
        raise SystemExit(f"{ref!r} matches {len(matches)} runs in {ledger.path}")
    return matches[0]


def _progress(kind: str, body: dict[str, Any]) -> None:
    stamp = time.strftime("%H:%M:%S")
    if kind == "round_started":
        print(f"{stamp} round {body['generation']}: incumbent {body['incumbent'][:12]}", flush=True)
    elif kind == "episode":
        score = body.get("score")
        who = "incumbent" if body["role"] == "incumbent" else f"candidate {body['candidate']}"
        print(f"{stamp}   {who:<12} {body['task']:<52} {body.get('status', ''):<12} "
              f"{f'{score:+.4f}' if isinstance(score, float) else '–'}  {body.get('episode') or body.get('error', '')}",
              flush=True)
    elif kind == "evolver_finished":
        found = ", ".join(f"{item['name']}={item['status']}" for item in body.get("candidates", []))
        print(f"{stamp}   evolver {body.get('status')} in {body.get('seconds', 0):.0f} s: {found or body.get('error')}",
              flush=True)
    elif kind == "candidate_proposed":
        print(f"{stamp}   candidate {body['name']} {body['digest'][:12]}: {body.get('summary', '')[:100]}", flush=True)
    elif kind == "candidate_rejected":
        print(f"{stamp}   candidate {body['name']} {body['status']}: {'; '.join(body.get('problems', []))[:160]}",
              flush=True)
    elif kind == "candidate_compared":
        decision = body["decision"]
        if decision.get("delta") is None:
            print(f"{stamp}   {body['name']}: {decision['reason']}", flush=True)
        else:
            low, high = decision["interval"]
            print(f"{stamp}   {body['name']}: gain {decision['delta']:+.4f} [{low:+.4f}, {high:+.4f}] -> "
                  f"{decision['reason']}", flush=True)
    elif kind == "harness_adopted":
        print(f"{stamp}   adopted {body['name']} {body['digest'][:12]} (gain {body['delta']:+.4f})", flush=True)
    elif kind == "round_finished":
        print(f"{stamp} round {body['generation']} done: {body.get('outcome')}", flush=True)
    elif kind == "adoption_checked":
        print(f"{stamp}   round {body.get('generation')} adoption {_adoption_check(body)}", flush=True)
    elif kind == "outage_started":
        print(f"{stamp}   {body['model']} gives no reply ({body.get('probe')}); waiting up to "
              f"{body.get('limit_minutes')} min, no episode starts meanwhile", flush=True)
    elif kind == "outage_ended":
        print(f"{stamp}   {body['model']} {'answers again' if body.get('recovered') else 'still gives no reply'} "
              f"after {body.get('seconds', 0) / 60:.0f} min", flush=True)


def cmd_evolve_run(args: argparse.Namespace) -> int:
    from .evolution import Evolution, EvolveConfig, fold

    run_id = None
    if args.resume:
        from .evolution.loop import RunBusy, claim_resume
        from .kernel.ledger import LedgerConflict

        with Ledger(args.ledger) as ledger:
            run_id = _match(ledger, args.resume, "evolve_started", "evo:")
            config = EvolveConfig.from_dict(fold(ledger.events(run_id))["config"])
        try:  # recorded before it starts, so no other process continues it too
            claim_resume(args.ledger, run_id, by="cli", force=args.force)
        except (ValueError, RunBusy, LedgerConflict) as error:
            raise SystemExit(f"{error}" + ("; pass --force if it is not" if isinstance(error, RunBusy)
                                           else "")) from None
    else:
        if not args.suite:
            raise SystemExit("evolve run needs --suite (or --resume RUN)")
        from .task.suite import suite_spec

        repeats = args.repeats if args.repeats is not None else suite_spec(args.suite).repeats
        config = EvolveConfig(suite=args.suite, harness=args.harness, generations=args.generations,
                              candidates=args.candidates, repeats=repeats, minutes=args.minutes,
                              tokens=args.tokens, model=args.model, evolver_minutes=args.evolver_minutes,
                              evolver_tokens=args.evolver_tokens, evolver_model=args.evolver_model,
                              concurrency=args.concurrency, retries=args.retries,
                              outage_minutes=args.outage_minutes, auto_resume=args.auto_resume,
                              check_adoptions=not args.no_check_adoptions)
    evolution = Evolution(config, args.ledger, on_event=None if args.json else _progress)
    run_id = run_id or evolution.start(label=args.label)
    first, later = config.episodes_per_round(len(evolution.tasks))
    if not args.json:
        print(f"{run_id} on {config.suite}: {len(evolution.tasks)} training tasks, {config.generations} rounds x "
              f"{config.candidates} candidates, {config.repeats} episode(s) per harness and task\n"
              f"  {first + config.generations * later} episodes (plus retries and rerun rounds) of "
              f"{config.minutes:g} min and "
              f"{config.generations} evolver sessions of {config.evolver_minutes:g} min, "
              f"{config.concurrency} at a time (ledger {args.ledger})", flush=True)
    result = evolution.run(run_id)
    if args.json:
        _print_json({"run_id": run_id, **result})
        return 0
    print(f"\n{run_id} finished: seed {result['seed'][:12]} -> final {result['final'][:12]}, "
          f"{result['adoptions']} adoption(s), {result['episodes']} episodes, {result['tokens']:,} tokens")
    print(f"next: ecogym evolve verify {run_id} --holdout")
    return 0


def cmd_evolve_list(args: argparse.Namespace) -> int:
    from .evolution import fold, summary

    with Ledger(args.ledger) as ledger:
        runs = [summary(fold(ledger.events(run_id))) for run_id in ledger.runs(first="evolve_started")]
    if args.json:
        _print_json(runs)
        return 0
    rows = [[item["run_id"], item["started_at"][:16].replace("T", " "), item["suite"]["id"], item["status"],
             f"{item['rounds_finished']}/{item['generations']}", item["adoptions"], item["episodes"],
             f"{item['seed']['digest'][:12]} -> {item['incumbent'][:12]}",
             ", ".join((["holdout"] if item["verified"]["holdout"] else []) + item["verified"]["sealed"]),
             item["label"]] for item in runs]
    print(_table(["run", "started", "suite", "status", "rounds", "adopted", "episodes", "seed -> final", "verified",
                  "label"], rows) if rows else "no evolution runs in this ledger")
    return 0


def cmd_evolve_show(args: argparse.Namespace) -> int:
    from .evolution import fold

    with Ledger(args.ledger) as ledger:
        view = fold(ledger.events(_match(ledger, args.run, "evolve_started", "evo:")))
    if args.json:
        _print_json(view)
        return 0
    print(f"{view['run_id']}  {view['status']}  suite {view['suite']['id']}  seed {view['seed']['digest'][:12]} -> "
          f"incumbent {view['incumbent'][:12]}")
    for round_ in view["rounds"]:
        scores: dict[str, list[float]] = {}
        for entry in round_["episodes"]:
            if isinstance(entry.get("score"), float):
                key = "incumbent" if entry["role"] == "incumbent" else f"candidate {entry['candidate']}"
                scores.setdefault(key, []).append(entry["score"])
        means = ", ".join(f"{key} {sum(values) / len(values):+.4f} ({len(values)})" for key, values in scores.items())
        print(f"\nround {round_['generation']}  {round_.get('outcome') or 'unfinished'}  episodes: {means or '–'}")
        for candidate in round_["candidates"]:
            decision = candidate.get("decision") or {}
            gain = (f"gain {decision['delta']:+.4f} [{decision['interval'][0]:+.4f}, {decision['interval'][1]:+.4f}] "
                    f"{decision['reason']}" if decision.get("delta") is not None else "; ".join(
                        candidate.get("problems", []))[:120])
            print(f"  {candidate['name']} {str(candidate.get('digest') or '')[:12]:<12} {candidate['status']:<12} "
                  f"{gain}")
            if candidate.get("summary"):
                print(f"      {candidate['summary'][:160]}")
        if round_.get("sealed_check"):
            print(f"  {_adoption_check(round_['sealed_check'])}")
    if view.get("ended_early"):
        print(f"\nended early after round {view['result'].get('last_generation')} of "
              f"{view['result'].get('planned_generations')}")
    verification = view["verification"]
    for name, body in [("holdout", verification["holdout"]), *verification["sealed"].items()]:
        if body:
            decision = body.get("decision") or {}
            means = ", ".join(f"{key} {body['means'][key]:+.4f}" for key in ("seed", "final")
                              if body["means"].get(key) is not None)
            gain = f"; final - seed {decision['delta']:+.4f} -> {decision['reason']}" if decision else ""
            print(f"\nverified on {name}: {means}{gain}")
    return 0


def cmd_evolve_verify(args: argparse.Namespace) -> int:
    from .evolution import verify_holdout, verify_sealed

    with Ledger(args.ledger) as ledger:
        run_id = _match(ledger, args.run, "evolve_started", "evo:")
    body = verify_holdout(args.ledger, run_id) if args.holdout else verify_sealed(args.ledger, run_id, args.sealed)
    if args.json:
        _print_json(body)
        return 0
    stage = "holdout" if args.holdout else args.sealed
    print(f"{run_id} on {stage}:")
    for label in ("seed", "final"):
        version = body["versions"].get(label)
        if version is None:
            continue
        items = version.get("episodes") or version.get("tasks") or {}
        cells = ", ".join(f"{task} {item['score']:+.4f}" for task, item in items.items()
                          if isinstance(item.get("score"), float))
        mean = body["means"].get(label)
        print(f"  {label} {version['digest'][:12]}: mean {f'{mean:+.4f}' if mean is not None else '–'}; {cells}")
    decision = body.get("decision")
    if decision:
        low, high = decision["interval"]
        print(f"  final - seed: {decision['delta']:+.4f} [{low:+.4f}, {high:+.4f}] -> {decision['reason']}")
    return 0


def _adoption_check(body: dict[str, Any]) -> str:
    """One line on an adoption's check on the training tasks' sealed period."""

    stage = body.get("stage", "validation")
    if body.get("error"):
        return f"check on {stage} failed: {body['error'][:200]}"
    decision = body.get("decision")
    if not decision:
        return f"check on {stage}: no task to compare"
    tasks = decision.get("tasks") or []
    better = sum(1 for item in tasks if (item.get("delta") or 0) > 0)
    low, high = decision["interval"]
    return (f"check on {stage}: adopted - replaced {decision['delta']:+.4f} [{low:+.4f}, {high:+.4f}], "
            f"better on {better}/{len(tasks)} tasks")


def _not_running(view: dict[str, Any], force: bool) -> None:
    from .evolution.loop import stale

    if view["status"] == "running" and not stale(view) and not force:
        raise SystemExit(f"{view['run_id']} may still be running (owner {view.get('owner') or 'unknown'}, last "
                         f"event {view['updated_at']}); stop it first, or pass --force if it is not")


def cmd_evolve_finish(args: argparse.Namespace) -> int:
    from .evolution.loop import RunBusy, finish_run
    from .kernel.ledger import LedgerConflict

    with Ledger(args.ledger) as ledger:
        run_id = _match(ledger, args.run, "evolve_started", "evo:")
    try:
        body = finish_run(args.ledger, run_id, force=args.force)
    except (ValueError, RunBusy, LedgerConflict) as error:
        raise SystemExit(f"{error}" + ("; pass --force if it is not" if isinstance(error, RunBusy) else "")) from None
    if args.json:
        _print_json({"run_id": run_id, **body})
        return 0
    print(f"{run_id} ended after round {body['last_generation']} of {body['planned_generations']}: final "
          f"{body['final'][:12]} ({body['adoptions']} adoption(s)); it cannot be resumed now\n"
          f"next: ecogym evolve verify {run_id} --holdout")
    return 0


def cmd_evolve_check_adoptions(args: argparse.Namespace) -> int:
    from .evolution import fold
    from .evolution.verify import check_adoptions, unchecked

    with Ledger(args.ledger) as ledger:
        run_id = _match(ledger, args.run, "evolve_started", "evo:")
        view = fold(ledger.events(run_id))
    _not_running(view, args.force)
    if not args.json:
        pending = unchecked(view)
        print(f"{run_id}: {len(pending)} adoption(s) to check" + (f" (generations {pending})" if pending else ""),
              flush=True)
    bodies = check_adoptions(args.ledger, run_id)
    if args.json:
        _print_json(bodies)
        return 0
    for body in bodies:
        print(f"  round {body['generation']} {body['parent'][:12]} -> {body['digest'][:12]}: {_adoption_check(body)}")
    return 0


def cmd_harness_show(args: argparse.Namespace) -> int:
    from .harness import content_files, resolve

    harness = resolve(args.harness)
    body = {**harness.describe(), "root": str(harness.root),
            "files": [path.as_posix() for path in content_files(harness.root)]}
    if args.json:
        _print_json(body)
        return 0
    print(f"{harness.name}  {harness.digest}\n  parent {harness.parent or '–'}  author {harness.author}\n"
          f"  {harness.summary}\n  agent {body['agent']}\n  {harness.root}")
    for path in body["files"]:
        print(f"    {path}")
    return 0


def cmd_harness_diff(args: argparse.Namespace) -> int:
    from .harness import diff, resolve

    changes = diff(resolve(args.before).root, resolve(args.after).root)
    if args.json:
        _print_json(changes)
        return 0
    for change in changes:
        print(change["patch"] or f"{change['status']} {change['path']}")
    print(f"\n{len(changes)} file(s) changed")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .episodes.export import export

    manifest = export(args.ledger, Path(args.out), min_score=None if args.all else args.min_score,
                      include_failed=args.all, evolver=not args.no_evolver, margin=args.margin,
                      cross_prompt=args.cross_prompt)
    if args.json:
        _print_json(manifest)
        return 0
    counts = manifest["counts"]
    print(f"wrote {args.out}: {counts['sft']} SFT threads from {counts['sessions']} sessions, "
          f"{counts['preferences']} preference pairs, {counts['episodes']} episodes, {counts['notes']} notes")
    for reason, count in manifest["skipped"].items():
        print(f"  skipped {count}: {reason}")
    return 0


def cmd_expert_list(args: argparse.Namespace) -> int:
    from .episodes import expert

    with Ledger(args.ledger) as ledger:
        questions, notes = expert.questions(ledger), expert.notes(ledger)
    if args.json:
        _print_json({"questions": questions, "notes": notes})
        return 0
    shown = [item for item in questions if args.all or not item["answers"]]
    for item in shown:
        print(f"{item['id']}  {item['at'][:16].replace('T', ' ')}  {item['task_id'] or 'evolver'}"
              f"{'  (answered)' if item['answers'] else ''}\n  Q: {item['question']}")
        if item["context"]:
            print(f"     {item['context'][:300]}")
        for answer in item["answers"]:
            print(f"  A: {answer['text']}")
    for note in notes:
        print(f"note {note['seq']}  [{note['scope']}]  {note['text']}")
    if not shown and not notes:
        print("no open questions" if questions else "no questions or notes yet")
    return 0


def cmd_expert_answer(args: argparse.Namespace) -> int:
    from .episodes import expert

    with Ledger(args.ledger) as ledger:
        body = expert.answer(ledger, args.question, args.text, by=args.by)
    print(f"answered {args.question} (seq {body['seq']}); later episodes on matching tasks will see it")
    return 0


def cmd_expert_note(args: argparse.Namespace) -> int:
    from .episodes import expert

    with Ledger(args.ledger) as ledger:
        body = expert.add_note(ledger, args.scope, args.text, by=args.by)
    print(f"noted for {args.scope} (seq {body['seq']})")
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
    run.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                     help="proposer option, e.g. model=glm-5.3 (repeatable)")
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
    serve = command("serve", cmd_serve, "JSON API for the DSH plugin (DSH starts it)", ledger=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8850, help="0 lets the OS choose; the URL line reports it")
    serve.add_argument("--token", default=None,
                       help="bearer token for every API call (default $ECOGYM_SERVICE_TOKEN); required beyond loopback")
    serve.add_argument("--exit-with-stdin", action="store_true", help="shut down when stdin reaches end of file")
    serve.add_argument("--no-auto-resume", action="store_true",
                       help="never resume runs by itself (by default it resumes, every 5 minutes, the runs started "
                            "with auto_resume that a shutdown, an outage of the model service or a crash stopped)")
    setup = command("setup", cmd_setup, "create the agent runtime (Python with the scientific stack)")
    setup.add_argument("--force", action="store_true", help="rebuild it")

    dsh = commands.add_parser("dsh", help="the DSH plugin: DSH starts ecogym and shows the 生态进化 panel")
    dshs = dsh.add_subparsers(dest="dsh_command", required=True)
    for name, handler, text in (("install", cmd_dsh_install, "add the plugin to a DSH profile"),
                                ("uninstall", cmd_dsh_uninstall, "remove the plugin from a DSH profile"),
                                ("status", cmd_dsh_status, "where the plugin is installed and what it starts")):
        sub = dshs.add_parser(name, help=text)
        sub.set_defaults(handler=handler)
        sub.add_argument("--json", action="store_true", help="machine-readable output")
        sub.add_argument("--profile", default="web", help="DSH profile under $DSH_HOME/profiles (default web)")
        if name == "install":
            sub.add_argument("--home", type=Path, default=None,
                             help="ECOGYM_HOME of the service (default $ECOGYM_HOME or ~/.ecogym)")
            sub.add_argument("--ledger", type=Path, default=None, help="ledger file (default <home>/ledger.sqlite3)")
            sub.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                             help="extra environment for the service (repeatable); ECOGYM_DATA_ROOT, "
                                  "ECOGYM_RUNTIME_PYTHON, ECOGYM_DSH_BIN and ECOGYM_CACHE_DIR are taken from "
                                  "this shell")
            sub.add_argument("--keep-legacy", action="store_true",
                             help="leave the EcologyRSI plugin block in the profile's cordis.patch.yml")
    auto = dshs.add_parser("autostart", help="macOS: launchd starts `dsh --profile web --no-open` at login and after "
                                             "a crash (on), stops it and removes the job (off), or reports it")
    auto.set_defaults(handler=cmd_dsh_autostart)
    auto.add_argument("action", choices=("on", "off", "status"))
    auto.add_argument("--json", action="store_true", help="machine-readable output")
    auto.add_argument("--profile", default="web", help="a DSH profile that serves the web UI (default web)")
    auto.add_argument("--wait", type=float, default=90.0, help="seconds `on` waits for the web address (default 90)")

    episode = commands.add_parser("episode", help="agent episodes: a DSH agent solves a task, the judge scores it")
    episodes = episode.add_subparsers(dest="episode_command", required=True)

    def episode_command(name: str, handler, help_text: str):
        sub = episodes.add_parser(name, help=help_text)
        sub.set_defaults(handler=handler)
        sub.add_argument("--json", action="store_true", help="machine-readable output")
        sub.add_argument("--ledger", type=Path, default=default_ledger(), help="ledger file")
        return sub

    run_episode = episode_command("run", cmd_episode_run, "run one sandboxed agent episode on a task")
    run_episode.add_argument("task", help="task id, or <task id>:<episode> for another episode of its dataset")
    run_episode.add_argument("--harness", default="seed", help="seed, a harness directory, or a stored digest prefix")
    run_episode.add_argument("--minutes", type=float, default=30.0, help="time budget (awake time)")
    run_episode.add_argument("--tokens", type=int, default=4_000_000,
                             help="token budget (cost-weighted: cached prompt tokens count a tenth)")
    run_episode.add_argument("--model", default=None,
                             help="override the harness model: <provider>/<model> (see `ecogym models`) or a "
                                  "DeepSeek model id")
    run_episode.add_argument("--label", default="")
    run_episode.add_argument("--keep", action="store_true", help="keep the scratch workspace after the episode")
    episode_command("list", cmd_episode_list, "episodes in the ledger")
    episode_command("show", cmd_episode_show, "an episode's events").add_argument("episode")

    command("suites", cmd_suites, "task suites: training and holdout tasks")
    command("models", cmd_models, "model routes the agents can use: DeepSeek and DSH's providers")

    evolve = commands.add_parser("evolve", help="evolve an agent harness on a suite's training tasks")
    evolves = evolve.add_subparsers(dest="evolve_command", required=True)

    def evolve_command(name: str, handler, help_text: str):
        sub = evolves.add_parser(name, help=help_text)
        sub.set_defaults(handler=handler)
        sub.add_argument("--json", action="store_true", help="machine-readable output")
        sub.add_argument("--ledger", type=Path, default=default_ledger(), help="ledger file")
        return sub

    start = evolve_command("run", cmd_evolve_run, "start (or --resume) an evolution run")
    start.add_argument("--suite", help="suite id (see `ecogym suites`) or a suite TOML file")
    start.add_argument("--resume", metavar="RUN", help="continue this run after its last finished round")
    start.add_argument("--harness", default="seed", help="starting harness: seed, a directory or a stored digest")
    start.add_argument("--generations", type=int, default=3, help="proposal rounds after the baseline round")
    start.add_argument("--candidates", type=int, default=2, help="candidate harnesses per round (1-6)")
    start.add_argument("--repeats", type=int, default=None,
                       help="episodes per harness and training task per round (default: what the suite is designed "
                            "for, 2 unless it says otherwise)")
    start.add_argument("--minutes", type=float, default=20.0, help="time budget of one episode (awake time)")
    start.add_argument("--tokens", type=int, default=3_000_000, help="cost-weighted token budget of one episode")
    start.add_argument("--model", default=None,
                       help="pin the inner agents' model, <provider>/<model> (default: each harness's own)")
    start.add_argument("--evolver-minutes", type=float, default=30.0)
    start.add_argument("--evolver-tokens", type=int, default=4_000_000)
    start.add_argument("--evolver-model", default="deepseek-v4-pro", help="the evolver's model, <provider>/<model>")
    start.add_argument("--concurrency", type=int, default=3, help="episodes running at the same time")
    start.add_argument("--retries", type=int, default=1, help="reruns of an episode that never got a model reply")
    start.add_argument("--outage-minutes", type=float, default=120.0,
                       help="how long a round waits for a model service that stopped answering (0: stop at once)")
    start.add_argument("--auto-resume", action="store_true",
                       help="let the service (ecogym serve, the DSH panel) resume the run if it stops for an outage "
                            "or this process dies (off by default here; on by default for runs the panel starts)")
    start.add_argument("--no-check-adoptions", action="store_true",
                       help="do not check each adoption on the training tasks' sealed validation period")
    start.add_argument("--force", action="store_true", help="--resume even though the ledger says it is running")
    start.add_argument("--label", default="")
    evolve_command("list", cmd_evolve_list, "evolution runs in the ledger")
    evolve_command("show", cmd_evolve_show, "rounds, candidates and decisions of a run").add_argument("run")
    check = evolve_command("verify", cmd_evolve_verify, "one-time verification on holdout tasks or a sealed period")
    check.add_argument("run")
    target = check.add_mutually_exclusive_group(required=True)
    target.add_argument("--holdout", action="store_true", help="run seed and final harness on the holdout tasks")
    target.add_argument("--sealed", choices=("validation", "final_test"),
                        help="re-judge their programs on a sealed period")
    end = evolve_command("finish", cmd_evolve_finish,
                         "end a stopped run where it is (its incumbent becomes the final version), to verify it")
    end.add_argument("run")
    end.add_argument("--force", action="store_true", help="even though the ledger says it is running")
    recheck = evolve_command("check-adoptions", cmd_evolve_check_adoptions,
                             "check each adoption not checked yet on the training tasks' sealed validation period")
    recheck.add_argument("run")
    recheck.add_argument("--force", action="store_true", help="even though the ledger says it is running")

    harness = commands.add_parser("harness", help="inspect harness versions")
    harnesses = harness.add_subparsers(dest="harness_command", required=True)
    show = harnesses.add_parser("show", help="a harness's metadata and files")
    show.set_defaults(handler=cmd_harness_show)
    show.add_argument("harness", help="seed, a directory or a stored digest prefix")
    show.add_argument("--json", action="store_true")
    compare = harnesses.add_parser("diff", help="unified diff between two harness versions")
    compare.set_defaults(handler=cmd_harness_diff)
    compare.add_argument("before")
    compare.add_argument("after")
    compare.add_argument("--json", action="store_true")

    export = command("export", cmd_export, "training data: SFT threads, preference pairs, notes", ledger=True)
    export.add_argument("--out", required=True, help="output directory")
    export.add_argument("--min-score", type=float, default=0.0,
                        help="SFT only from episodes with an accepted submission scoring above this on the hidden "
                             "stage (0: the naive baseline)")
    export.add_argument("--all", action="store_true", help="export every session; filter later with meta")
    export.add_argument("--margin", type=float, default=0.02, help="smallest score gap of a preference pair")
    export.add_argument("--cross-prompt", action="store_true",
                        help="also pair episodes whose prompts differ (they carry rejected_prompt)")
    export.add_argument("--no-evolver", action="store_true", help="leave out evolver sessions")

    expert = commands.add_parser("expert", help="expert questions, answers and notes")
    experts = expert.add_subparsers(dest="expert_command", required=True)

    def expert_command(name: str, handler, help_text: str):
        sub = experts.add_parser(name, help=help_text)
        sub.set_defaults(handler=handler)
        sub.add_argument("--json", action="store_true", help="machine-readable output")
        sub.add_argument("--ledger", type=Path, default=default_ledger(), help="ledger file")
        return sub

    expert_command("list", cmd_expert_list, "open questions (--all: answered too) and notes").add_argument(
        "--all", action="store_true")
    reply = expert_command("answer", cmd_expert_answer, "answer a question (later episodes see the answer)")
    reply.add_argument("question", help="question id, as shown by `ecogym expert list`")
    reply.add_argument("text")
    reply.add_argument("--by", default="")
    note = expert_command("note", cmd_expert_note, "leave a note for every episode in a scope")
    note.add_argument("scope", help="* (all tasks), a dataset id, a task id or <task id>:<episode>")
    note.add_argument("text")
    note.add_argument("--by", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    load_domains()
    try:
        if getattr(args, "task", None):
            task_spec(args.task)  # fail early with the list of known tasks
        return int(args.handler(args) or 0)
    except (KeyError, ValueError, PermissionError, FileNotFoundError, LookupError) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else error
        print(f"ecogym: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
