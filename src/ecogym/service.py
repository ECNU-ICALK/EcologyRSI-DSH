"""Thin HTTP service: a JSON API over the ledger.

The user interface is the DSH plugin in ``ecogym/web/plugin``: DSH starts this
service with ``--port 0 --exit-with-stdin`` and proxies ``/api/ecogym/*`` here
(the plugin answers ``_status`` and ``_restart`` itself). ``ROUTES`` lists every
endpoint with the ``Service`` method that answers it; ``GET /api/routes`` serves
the same table, and the panel's introduction page shows it. The panel uses every
group but the method arena, which is there for programs using the service
directly and for the tests (the command line runs the arena itself). Every view
is folded from the ledger on request, so the service holds no state a restart
would lose beyond the runs it is executing.

The service binds to 127.0.0.1 by default. POSTs must carry ``X-Ecogym: 1``
(a custom header no cross-site form can send); with ``token`` set, every API
call needs ``Authorization: Bearer <token>``. ``close`` stops what the service
runs (agents and judges included), so evolution runs end as stopped and can be
resumed; a run still busy at its deadline is recorded as stopped by ``close``
itself. A run that says running belongs to the process named in its latest
round (``owner``); it is stale once that process is gone.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import socket
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from . import __version__
from . import agents as _agents  # noqa: F401  (registers the llm proposer)
from .arena import Arena, RunConfig, RunStopped, fold, make_proposer, seal
from .arena.proposers import PROPOSERS
from .arena.sealed import exposure_key
from .data.catalog import dataset_spec
from .evaluation.harness import Bench
from .kernel.canonical import jsonable
from .kernel.ledger import Ledger
from .protocol.partitions import SEALED_STAGES, STAGES
from .report import render_run
from .solvers.base import solver_specs
from .task.spec import TaskSpec, task_specs

MAX_BODY = 1_000_000
MODEL_CHECK_SECONDS = 600
HTML = "text/html; charset=utf-8"
SEALED_ORDER = tuple(stage for stage in STAGES if stage in SEALED_STAGES)
EXPORT_ID = re.compile(r"^\d{8}-\d{6}(?:-\d+)?$")
PREVIEW_MESSAGES = 16
PREVIEW_CHARS = 600
EVOLVE_FIELDS = {"harness": str, "generations": int, "candidates": int, "repeats": int, "minutes": float,
                 "tokens": int, "model": str, "evolver_minutes": float, "evolver_tokens": int, "evolver_model": str,
                 "concurrency": int, "retries": int}
HARNESS_TEXT = ("instructions.md", "memory/lessons.md", "harness.toml")
ARENA = "方法竞技"


@dataclass(frozen=True, slots=True)
class Route:
    """One endpoint under ``/api/``, answered by the ``Service`` method named ``handler``.

    ``{name}`` in ``path`` matches one segment. The handler gets the matched segments in order, then the
    POST's JSON object when ``body`` is set, then the ``query`` parameters as keywords (None when absent).
    ``html``: the handler returns a page instead of JSON. ``group`` and ``text`` describe it for people.
    """

    method: str
    path: str
    handler: str
    group: str
    text: str
    status: int = 200
    body: bool = False
    query: tuple[str, ...] = ()
    html: bool = False

    def match(self, method: str, parts: list[str]) -> list[str] | None:
        """The path arguments when ``method`` and ``parts`` (the path's segments) are this route's, else None."""

        names = self.path.split("/")
        if method != self.method or len(names) != len(parts):
            return None
        found = []
        for name, part in zip(names, parts, strict=True):
            if name.startswith("{"):
                found.append(part)
            elif name != part:
                return None
        return found


ROUTES = (
    Route("GET", "catalog", "catalog", "目录", "任务、任务组、模型路由和进化的默认设置（面板打开时读一次）"),
    Route("GET", "routes", "routes", "目录", "这张接口表"),
    Route("GET", "tasks/{task}/stages", "task_stages", "目录",
          "一个任务的时间分段：拟合、本地检验、隐藏期、封存时期"),
    Route("GET", "evolutions", "evolutions", "进化", "进化运行列表"),
    Route("POST", "evolutions", "start_evolution", "进化",
          "在后台开始一次进化（任务组、起点、代数、候选、回合、模型…）", status=201, body=True),
    Route("GET", "evolutions/{run}", "evolution", "进化",
          "一次进化运行：各代、候选与比较、验证、各回合预报的平均、训练任务的参考成绩、起点是哪些运行进化出来的"),
    Route("GET", "evolutions/{run}/evolver/{generation}", "evolver_session", "进化",
          "一代的评审会话：看了哪些证据和版本、调了哪些工具、改了哪些文件，以及每一步（?steps=0 不带每一步）",
          query=("steps",)),
    Route("POST", "evolutions/{run}/stop", "stop", "进化", "这一代结束后停止"),
    Route("POST", "evolutions/{run}/abort", "abort", "进化", "立即中止：正在运行的智能体停下，继续时这一代重跑"),
    Route("POST", "evolutions/{run}/resume", "resume_evolution", "进化",
          "继续停下或中断的运行（{force: true}：仍像在另一个进程里运行时）", status=202, body=True),
    Route("POST", "evolutions/{run}/verify", "verify_evolution", "进化",
          "在后台验证：{kind: holdout} 或 {kind: sealed, stage}", status=202, body=True),
    Route("GET", "episodes", "episodes", "回合",
          "回合列表（?evolution=运行 只看一个运行，&status=running 只看运行中的）", query=("evolution", "status")),
    Route("GET", "episodes/{episode}", "episode", "回合",
          "一个回合：隐藏期成绩、本地检验与提交、笔记、最终回复、工具使用"),
    Route("GET", "episodes/{episode}/transcript", "episode_transcript", "回合",
          "一个回合的会话记录：智能体每一步的思考、工具调用和结果（回合结束后才有）"),
    Route("GET", "harnesses/{ref}", "harness", "Harness", "一个 Harness 版本：文件、指令、经验、DSH 设置"),
    Route("GET", "harnesses/{before}/diff/{after}", "harness_diff", "Harness", "两个版本逐文件的差异"),
    Route("POST", "models/check", "model_check", "模型", "{model}：现在能否调用（发一个很小的请求，结果缓存十分钟）",
          body=True),
    Route("GET", "expert", "expert", "专家", "智能体提给专家的问题、回答和主动提示"),
    Route("POST", "expert/answer", "expert_answer", "专家", "回答一个问题 {question, text, by}", status=201, body=True),
    Route("POST", "expert/note", "expert_note", "专家", "写一条主动提示 {scope, text, by}", status=201, body=True),
    Route("GET", "exports", "exports", "训练数据", "账本旁的训练数据导出"),
    Route("POST", "exports", "start_export", "训练数据",
          "在后台导出 SFT 线程和偏好对 {min_score, include_failed, evolver, margin, cross_prompt}",
          status=202, body=True),
    Route("GET", "exports/{export}", "export_detail", "训练数据", "一次导出：清单、文件和第一条样本"),
    Route("GET", "runs", "runs", ARENA, "方法竞技的运行列表"),
    Route("POST", "runs", "start", ARENA, "在后台开始一次方法竞技", status=201, body=True),
    Route("GET", "runs/{run}", "run", ARENA, "一次方法竞技运行（结束后带各封存时期是否评过）"),
    Route("GET", "runs/{run}/cells", "cells", ARENA, "起点和最终方法的逐格成绩"),
    Route("GET", "runs/{run}/report", "report", ARENA, "静态 HTML 报告", html=True),
    Route("POST", "runs/{run}/stop", "stop", ARENA, "这一代结束后停止"),
    Route("POST", "runs/{run}/seal", "seal", ARENA, "封存评测 {stage: validation | final_test}", body=True),
)


def _suite_entry(suite: Any) -> dict[str, Any]:
    from .evaluation.suite import signed_rank_critical, t_quantile

    # by candidates per round (Bonferroni): the t interval's half-width per unit spread of the task gains, or the
    # signed-rank gate's critical count (adopt when at most this many Walsh averages of the task gains are <= 0)
    tasks, tails = len(suite.train), {str(k): (1 - suite.gates.confidence) / (2 * k) for k in range(1, 7)}
    ranked = suite.gates.test == "signed_rank" and tasks >= 3
    width = {key: t_quantile(1 - tail, tasks - 1) / tasks ** 0.5 for key, tail in tails.items()} \
        if tasks >= 3 and not ranked else None
    critical = {key: signed_rank_critical(tasks, tail) for key, tail in tails.items()} if ranked else None
    return {"id": suite.id, "title": suite.title, "train": list(suite.train), "holdout_tasks": len(suite.holdout),
            "gates": asdict(suite.gates), "notes": suite.notes.strip(), "repeats": suite.repeats,
            "recommended": suite.recommended, "interval_per_sd": width, "signed_rank_critical": critical}


def _task_entry(task: TaskSpec) -> dict[str, Any]:
    """What the workbench needs to label a task's cells and explain its gates."""

    dataset = dataset_spec(task.dataset)
    columns = {column.name: column for column in dataset.columns}
    return {
        "id": task.id,
        "title": task.title,
        "dataset": task.dataset,
        "step": dataset.step,
        "cells": [cell.key for cell in task.cells],
        "targets": [{"name": target.name, "weight": target.weight, "label": columns[target.name].label,
                     "unit": columns[target.name].unit} for target in task.targets],
        "horizons": list(task.horizons),
        "gates": {"min_delta": task.gates.min_delta, "confidence": task.gates.confidence,
                  "cell_cap": task.gates.cell_cap, "cell_alpha": task.gates.cell_alpha},
        "presets": {name: body.get("note", "") for name, body in task.presets.items()},
        "episodes": list(task.episodes),
    }


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists and belongs to someone else
        return True
    return True


class Service:
    def __init__(self, ledger_path: Path, *, token: str | None = None) -> None:
        self.ledger_path = Path(ledger_path)
        self.token = token
        self._benches: dict[str, Bench] = {}
        self._lock = threading.Lock()
        self._jobs = threading.Lock()  # held from "is it running?" until the new thread is registered
        self._stops: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._errors: dict[str, str] = {}  # failures raised before a run could log anything
        self._ensembles: dict[str, tuple[str, dict[str, Any]]] = {}  # run id -> (its last event time, scores)
        self._verifying: dict[str, threading.Thread] = {}
        self._evolutions: dict[str, Any] = {}  # run id -> Evolution, for abort and shutdown
        self._cancels: dict[str, threading.Event] = {}  # holdout verifications
        self._exporting: dict[str, threading.Thread] = {}
        self._model_checks: dict[str, tuple[float, dict[str, Any]]] = {}
        self.started = datetime.now(UTC)
        Ledger(self.ledger_path).close()  # create the file and schema up front

    def warm(self) -> None:
        """Import the evolution and episode code ahead of the first request (it takes seconds when cold)."""

        try:
            self.catalog()
            self.evolutions()
            self.episodes()
        except Exception:  # only a head start; the request will report any real problem
            pass

    def bench(self, task_id: str) -> Bench:
        with self._lock:
            if task_id not in self._benches:
                self._benches[task_id] = Bench.open(task_id)
            return self._benches[task_id]

    def catalog(self) -> dict[str, Any]:
        from .episodes import providers
        from .evolution import EvolveConfig
        from .harness import HarnessError, resolve
        from .task.suite import suite_specs

        defaults = asdict(EvolveConfig(suite=""))
        try:
            harness_model = resolve("seed").agent.model
        except (HarnessError, OSError, ValueError):
            harness_model = None
        return {
            "version": __version__,
            "ledger": str(self.ledger_path),
            "harness_model": harness_model,
            "models": providers.describe(),  # model routes for the pickers: DeepSeek and DSH's providers
            "tasks": [_task_entry(task) for task in task_specs()],
            "suites": [_suite_entry(suite) for suite in suite_specs()],
            "evolve_defaults": {key: defaults[key] for key in EVOLVE_FIELDS},
            "datasets": sorted({task.dataset for task in task_specs()}),
            "solvers": [s.describe() for s in solver_specs()],
            "proposers": sorted(PROPOSERS),
        }

    def routes(self) -> list[dict[str, Any]]:
        return [{"method": route.method, "path": f"/api/{route.path}", "group": route.group, "text": route.text}
                for route in ROUTES]

    def task_stages(self, task_id: str) -> dict[str, Any]:
        """One task's time line: every stage's interval in days from the task's first row (no values)."""

        from .episodes.judge import episode_of, iso

        bench = self.bench(task_id)
        episode, step = episode_of(bench), bench.frame.step
        per_day = 24 if step == "1h" else 1
        times = bench.frame.time[bench.frame.episode_rows(episode)]
        first, last = int(times[0]), int(times[-1]) + 1
        stages = []
        for stage in STAGES:
            start, end = bench.partitions.intervals[stage][episode]
            begin, until = iso(np.array([start, end]), step)
            stages.append({"stage": stage, "start": begin, "end": until,
                           "from_days": round((start - first) / per_day, 2), "days": round((end - start) / per_day, 2)})
        begin, until = iso(np.array([first, last]), step)
        return {"task": task_id, "step": step, "start": begin, "end": until,
                "days": round((last - first) / per_day, 2), "stages": stages}

    # ---- harness evolution, episodes, experts ---------------------------------------------------------------

    def evolutions(self) -> list[dict[str, Any]]:
        from .evolution import fold, summary

        with Ledger(self.ledger_path) as ledger:
            views = [fold(ledger.events(run_id)) for run_id in ledger.runs(first="evolve_started")]
        runs = []
        for view in views:
            item = summary(view)
            item["active"] = self._alive(item["run_id"])
            item["verifying"] = self._alive(item["run_id"], verifying=True)
            item["stale"] = not item["active"] and self._stale(view)
            runs.append(item)
        known = {item["run_id"] for item in runs}
        for run_id, error in self._errors.items():
            if run_id.startswith("evo:") and run_id not in known:
                runs.append({"run_id": run_id, "status": "failed", "error": error, "active": False})
        return list(reversed(runs))

    def _alive(self, run_id: str, *, verifying: bool = False) -> bool:
        thread = (self._verifying if verifying else self._threads).get(run_id)
        return thread is not None and thread.is_alive()

    def _stale(self, view: dict[str, Any]) -> bool:
        """Says running, but no thread of this service runs it and its owner process is gone.

        Callers check this service's own threads first. Without an owner on this host (older runs, another
        machine) the run is stale once it has logged nothing for longer than any of its steps can take.
        """

        if view["status"] != "running":
            return False
        owner = view.get("owner") or {}
        if owner.get("host") == socket.gethostname() and isinstance(owner.get("pid"), int):
            return owner["pid"] == os.getpid() or not _pid_alive(owner["pid"])
        config = view["config"]
        quiet = (datetime.now(UTC) - datetime.fromisoformat(view["updated_at"])).total_seconds()
        longest = max(float(config.get("minutes") or 0), float(config.get("evolver_minutes") or 0)) * 60
        return quiet > longest * 2 + 1800

    def evolution(self, run_id: str) -> dict[str, Any]:
        from .episodes.history import fold_episode
        from .evolution import fold
        from .evolution.ancestry import ancestry

        with Ledger(self.ledger_path) as ledger:
            events = ledger.events(run_id)
            if not events:
                raise ApiError(HTTPStatus.NOT_FOUND, f"no evolution run {run_id}")
            view = fold(events)
            references: dict[str, Any] = {}
            for round_ in view["rounds"]:
                for entry in round_["episodes"]:
                    if entry.get("episode") and entry["task"] not in references:
                        scored = fold_episode(ledger.events(entry["episode"])).get("scored") or {}
                        if scored.get("references"):
                            references[entry["task"]] = scored["references"]
            view["ancestry"] = ancestry(ledger, view["seed"]["digest"], exclude=run_id)
        view["active"] = self._alive(run_id)
        view["verifying"] = self._alive(run_id, verifying=True)
        view["stale"] = not view["active"] and self._stale(view)
        view["error"] = view.get("error") or self._errors.get(run_id)
        view["task_references"] = references
        view["ensembles"] = self._ensemble_scores(run_id, view)
        return view

    def evolver_session(self, run_id: str, generation: str, *, steps: str | None = None) -> dict[str, Any]:
        from .evolution import fold
        from .evolution.process import evolver_session

        if not generation.isdigit():
            raise ApiError(HTTPStatus.BAD_REQUEST, "generation must be a whole number")
        with Ledger(self.ledger_path) as ledger:
            events = ledger.events(run_id)
        if not events or events[0].kind != "evolve_started":
            raise ApiError(HTTPStatus.NOT_FOUND, f"no evolution run {run_id}")
        found = evolver_session(fold(events), int(generation), steps=steps not in ("0", "false", "no"))
        if found is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"{run_id} has no finished evolver session in generation {generation}")
        return found

    def _ensemble_scores(self, run_id: str, view: dict[str, Any]) -> dict[str, Any]:
        """Averaged forecasts per version and task (``evolution.ensemble``), recomputed when the run logs."""

        from .evolution.ensemble import ensemble_scores

        cached = self._ensembles.get(run_id)
        if cached is not None and cached[0] == view["updated_at"]:
            return cached[1]
        failures: list[str] = []
        scores = ensemble_scores(view, self.bench, failures=failures)
        if not failures:  # a file that could not be read is tried again on the next request
            self._ensembles[run_id] = (view["updated_at"], scores)
        return scores

    def start_evolution(self, body: dict[str, Any]) -> dict[str, Any]:
        from .evolution import Evolution, EvolveConfig
        from .task.suite import suite_spec

        try:
            values = {key: kind(body[key]) for key, kind in EVOLVE_FIELDS.items()
                      if body.get(key) not in (None, "")}
            values.setdefault("repeats", suite_spec(str(body["suite"])).repeats)  # the suite's design
            evolution = Evolution(EvolveConfig(suite=str(body["suite"]), **values), self.ledger_path)
            run_id = evolution.start(label=str(body.get("label", ""))[:200])
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{type(error).__name__}: {error}") from None
        with self._jobs:
            self._launch(run_id, evolution)
        return {"run_id": run_id}

    def _launch(self, run_id: str, evolution: Any) -> None:
        """Start a run's thread; the caller holds ``_jobs``. The thread forgets the run when it ends."""

        stop = threading.Event()

        def work() -> None:
            try:
                evolution.run(run_id, stop=stop.is_set)
            except Exception as error:  # logged as evolve_failed by the loop
                self._errors[run_id] = f"{type(error).__name__}: {error}"
            finally:
                with self._jobs:  # a finished run keeps no benches or predictions in memory
                    if self._evolutions.get(run_id) is evolution:
                        del self._evolutions[run_id], self._stops[run_id], self._threads[run_id]

        thread = threading.Thread(target=work, name=run_id, daemon=True)
        self._stops[run_id] = stop
        self._threads[run_id] = thread
        self._evolutions[run_id] = evolution
        thread.start()

    def resume_evolution(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Continue after the last finished round; the interrupted round starts again."""

        from .evolution import Evolution, EvolveConfig, fold

        with self._jobs:
            if self._alive(run_id) or self._alive(run_id, verifying=True):
                raise ApiError(HTTPStatus.CONFLICT, f"{run_id} is already running in this service")
            with Ledger(self.ledger_path) as ledger:
                events = ledger.events(run_id)
            if not events or events[0].kind != "evolve_started":
                raise ApiError(HTTPStatus.NOT_FOUND, f"no evolution run {run_id}")
            view = fold(events)
            if view["status"] == "finished":
                raise ApiError(HTTPStatus.CONFLICT, f"{run_id} has finished")
            if view["status"] == "running" and not body.get("force") and not self._stale(view):
                raise ApiError(HTTPStatus.CONFLICT, f"{run_id} may still be running in another process "
                                                    f"(owner {view.get('owner') or 'unknown'}, last event "
                                                    f"{view['updated_at']}); resume with force")
            try:
                evolution = Evolution(EvolveConfig.from_dict(view["config"]), self.ledger_path)
            except (KeyError, TypeError, ValueError, OSError) as error:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"{type(error).__name__}: {error}") from None
            self._errors.pop(run_id, None)
            self._launch(run_id, evolution)
        return {"run_id": run_id, "resumed": True}

    def abort(self, run_id: str) -> dict[str, Any]:
        with self._jobs:
            evolution, stop = self._evolutions.get(run_id), self._stops.get(run_id)
        if evolution is None or stop is None or not self._alive(run_id):
            raise ApiError(HTTPStatus.NOT_FOUND, f"{run_id} is not running in this service")
        stop.set()
        evolution.cancel.set()
        return {"run_id": run_id, "aborting": True}

    def verify_evolution(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        from .evolution import fold, verify_holdout, verify_sealed

        kind, stage = body.get("kind"), body.get("stage")
        if kind not in ("holdout", "sealed") or (kind == "sealed" and stage not in SEALED_ORDER):
            raise ApiError(HTTPStatus.BAD_REQUEST, "verify needs kind holdout, or kind sealed with a sealed stage")
        with self._jobs:  # one at a time: verifications of runs that share a version claim the same exposures
            busy = next((other for other, thread in self._verifying.items() if thread.is_alive()), None)
            if busy is not None:
                raise ApiError(HTTPStatus.CONFLICT, f"{run_id} is already being verified" if busy == run_id
                               else f"{busy} is being verified; verifications run one at a time")
            with Ledger(self.ledger_path) as ledger:
                events = ledger.events(run_id)
            if not events or fold(events)["status"] != "finished":
                raise ApiError(HTTPStatus.CONFLICT, f"{run_id} has not finished")
            cancel = self._cancels[run_id] = threading.Event()

            def work() -> None:
                try:
                    if kind == "holdout":
                        verify_holdout(self.ledger_path, run_id, cancel=cancel)
                    else:
                        verify_sealed(self.ledger_path, run_id, str(stage), cancel=cancel)
                except Exception as error:
                    self._errors[run_id] = f"verification failed: {type(error).__name__}: {error}"

            self._errors.pop(run_id, None)
            thread = threading.Thread(target=work, name=f"verify {run_id}", daemon=True)
            self._verifying[run_id] = thread
            thread.start()
        return {"run_id": run_id, "verifying": kind if kind == "holdout" else stage}

    def _orphaned(self, item: dict[str, Any]) -> bool:
        """Says running, belongs to a run this service does not run, and started before the service did."""

        run_id = (item.get("context") or {}).get("evolution")
        if item.get("status") != "running" or not run_id or self._alive(run_id) or self._alive(run_id, verifying=True):
            return False
        return datetime.fromisoformat(item["started_at"]) < self.started

    def episodes(self, *, evolution: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        """Episode summaries, newest first; ``evolution`` keeps one run's episodes (only those are folded)."""

        from .episodes.history import episode_records, summary

        with Ledger(self.ledger_path) as ledger:
            ids = None if not evolution else [
                event.run_id for event in ledger.events(kind="episode_started")
                if (event.body.get("context") or {}).get("evolution") == evolution]
            items = list(reversed([summary(record) for record in episode_records(ledger, ids)]))
        if status:
            items = [item for item in items if item["status"] == status]
        for item in items:
            item["orphaned"] = self._orphaned(item)
        return items

    def episode(self, episode_id: str) -> dict[str, Any]:
        from .episodes.history import fold_episode

        with Ledger(self.ledger_path) as ledger:
            events = ledger.events(episode_id)
        if not events or events[0].kind != "episode_started":
            raise ApiError(HTTPStatus.NOT_FOUND, f"no episode {episode_id}")
        record = fold_episode(events)
        record["orphaned"] = self._orphaned(record)
        return record

    def episode_transcript(self, episode_id: str) -> dict[str, Any]:
        """The episode's sessions as steps (``episodes.transcripts``); none while it runs (its model calls are
        stored when it ends)."""

        from .episodes.transcripts import session_threads

        record = self.episode(episode_id)
        directory = str(((record.get("finished") or {}).get("trajectory") or {}).get("dir") or "")
        found = session_threads(Path(directory) / "llm_calls.jsonl.gz", [record.get("root") or ""]) if directory else []
        return {"episode": episode_id, "status": record["status"], "recorded": bool(found), "threads": found}

    def harness(self, ref: str) -> dict[str, Any]:
        from .harness import HarnessError, content_files, resolve

        try:
            harness = resolve(ref)
        except HarnessError as error:
            raise ApiError(HTTPStatus.NOT_FOUND, str(error)) from None
        files = [path.as_posix() for path in content_files(harness.root)]
        texts = {name: (harness.root / name).read_text(encoding="utf-8") for name in HARNESS_TEXT
                 if (harness.root / name).is_file()}
        return {**harness.describe(), "files": files, "texts": texts}

    def harness_diff(self, before: str, after: str) -> list[dict[str, Any]]:
        from .harness import HarnessError, diff, resolve

        try:
            return diff(resolve(before).root, resolve(after).root)
        except HarnessError as error:
            raise ApiError(HTTPStatus.NOT_FOUND, str(error)) from None

    def model_check(self, body: dict[str, Any]) -> dict[str, Any]:
        """Can this model run now? Route, model and key, then one tiny request (cached for ten minutes)."""

        from .episodes import providers

        ref = str(body.get("model") or "").strip()
        cached = self._model_checks.get(ref)
        if cached is not None and time.monotonic() - cached[0] < MODEL_CHECK_SECONDS:
            return cached[1]
        try:
            canonical = providers.check(ref)
            result = {"model": canonical, "ok": True, "note": providers.probe(canonical)}
        except providers.ModelError as error:
            result = {"model": ref, "ok": False, "error": str(error)}
        self._model_checks[ref] = (time.monotonic(), result)
        return result

    def expert(self) -> dict[str, Any]:
        from .episodes import expert

        with Ledger(self.ledger_path) as ledger:
            return {"questions": expert.questions(ledger), "notes": expert.notes(ledger)}

    def expert_answer(self, body: dict[str, Any]) -> dict[str, Any]:
        from .episodes import expert

        with Ledger(self.ledger_path) as ledger:
            try:
                return expert.answer(ledger, str(body.get("question", "")), str(body.get("text", "")),
                                     by=str(body.get("by", "")))
            except KeyError as error:
                raise ApiError(HTTPStatus.NOT_FOUND, str(error.args[0])) from None

    def expert_note(self, body: dict[str, Any]) -> dict[str, Any]:
        from .episodes import expert

        with Ledger(self.ledger_path) as ledger:
            return expert.add_note(ledger, str(body.get("scope", "")), str(body.get("text", "")),
                                   by=str(body.get("by", "")))

    # ---- training data -----------------------------------------------------------------------------------------

    def _exports_root(self) -> Path:
        return self.ledger_path.parent / "exports"

    def exports(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for export_id, thread in sorted(self._exporting.items(), reverse=True):
            if thread.is_alive():
                items.append({"id": export_id, "path": str(self._exports_root() / export_id), "running": True})
            elif export_id in self._errors:
                items.append({"id": export_id, "running": False, "error": self._errors[export_id]})
        for manifest in sorted(self._exports_root().glob("*/manifest.json"), reverse=True):
            try:
                body = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            files = {path.name: path.stat().st_size for path in sorted(manifest.parent.glob("*.jsonl"))}
            items.append({"id": manifest.parent.name, "path": str(manifest.parent), "running": False, **body,
                          "files": files})
        return items

    def start_export(self, body: dict[str, Any]) -> dict[str, Any]:
        from .episodes.export import export

        try:
            min_score = None if body.get("min_score") is None else float(body["min_score"])
            options = {"min_score": min_score, "include_failed": bool(body.get("include_failed", False)),
                       "evolver": bool(body.get("evolver", True)), "margin": float(body.get("margin", 0.02)),
                       "cross_prompt": bool(body.get("cross_prompt", False))}
        except (TypeError, ValueError) as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{type(error).__name__}: {error}") from None
        with self._jobs:
            if any(thread.is_alive() for thread in self._exporting.values()):
                raise ApiError(HTTPStatus.CONFLICT, "an export is already running")
            export_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            target = self._exports_root() / export_id
            suffix = 1
            while target.exists() or target.name in self._exporting:
                suffix += 1
                target = self._exports_root() / f"{export_id}-{suffix}"
            export_id = target.name

            def work() -> None:
                try:
                    export(self.ledger_path, target, **options)
                except Exception as error:
                    self._errors[export_id] = f"{type(error).__name__}: {error}"

            thread = threading.Thread(target=work, name=f"export {export_id}", daemon=True)
            self._exporting[export_id] = thread
            thread.start()
        return {"id": export_id, "path": str(target)}

    def export_detail(self, export_id: str) -> dict[str, Any]:
        directory = self._exports_root() / export_id
        if not EXPORT_ID.match(export_id) or not (directory / "manifest.json").is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, f"no export {export_id}")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

        def first(name: str) -> dict[str, Any] | None:
            path = directory / name
            if not path.is_file():
                return None
            with path.open(encoding="utf-8") as stream:
                line = stream.readline()
            return json.loads(line) if line.strip() else None

        def brief(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out = []
            for message in messages[:PREVIEW_MESSAGES]:
                content = message.get("content")
                text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                out.append({"role": message.get("role"), "chars": len(text or ""), "text": (text or "")[:PREVIEW_CHARS],
                            "reasoning_chars": len(message.get("reasoning_content") or ""),
                            "tool_calls": [call.get("function", {}).get("name") for call in
                                           message.get("tool_calls") or []]})
            return out

        sft, pair = first("sft.jsonl"), first("preferences.jsonl")
        return {
            "id": export_id, "path": str(directory), **manifest,
            "files": {path.name: path.stat().st_size for path in sorted(directory.glob("*.jsonl"))},
            "sft_preview": {"id": sft["id"], "meta": sft["meta"], "messages": len(sft["messages"]),
                            "tools": len(sft.get("tools") or []), "head": brief(sft["messages"])} if sft else None,
            "preference_preview": {"id": pair["id"], "meta": pair["meta"], "prompt": len(pair["prompt"]),
                                   "chosen": len(pair["chosen"]), "rejected": len(pair["rejected"])}
            if pair else None,
        }

    def close(self, timeout: float = 30.0) -> bool:
        """Stop everything this service runs; True when every thread ended in time.

        Evolution runs end as stopped (resumable), holdout episodes as interrupted, a sealed verification
        keeps the tasks it finished. A run whose thread is still busy at the deadline is recorded as stopped
        here, so the ledger never says running for a process that is about to exit.
        """

        with self._jobs:
            stops, cancels = list(self._stops.values()), list(self._cancels.values())
            evolutions = dict(self._evolutions)
            threads = {**self._threads, **{f"verify {key}": value for key, value in self._verifying.items()}}
        for event in stops:
            event.set()
        for evolution in evolutions.values():
            evolution.cancel.set()
        for event in cancels:
            event.set()
        deadline = time.monotonic() + timeout
        for thread in threads.values():
            thread.join(max(0.0, deadline - time.monotonic()))
        busy = [run_id for run_id, thread in threads.items() if thread.is_alive()]
        for run_id in busy:
            if run_id in evolutions:
                with Ledger(self.ledger_path) as ledger:
                    ledger.append(run_id, "evolve_failed", {"error": "the service stopped while a step was still "
                                                                     "running", "stopped": True})
        return not busy

    def _run_view(self, ledger: Ledger, run_id: str) -> dict[str, Any]:
        events = ledger.events(run_id)
        thread = self._threads.get(run_id)
        alive = thread is not None and thread.is_alive()
        if not events:
            if thread is None:
                raise ApiError(HTTPStatus.NOT_FOUND, f"no run {run_id}")
            status = "starting" if alive else "failed"
            return {"run_id": run_id, "status": status, "active": alive, "error": self._errors.get(run_id),
                    "generations": [], "sealed": []}
        view = fold(events)
        view["active"] = alive
        view["updated_at"] = events[-1].at
        return view

    def runs(self) -> list[dict[str, Any]]:
        with Ledger(self.ledger_path) as ledger:
            views = [self._run_view(ledger, run_id) for run_id in ledger.runs(first="run_started")]
        summaries = []
        for view in reversed(views):
            result = view.get("result")
            proposer = view.get("proposer") or {}
            summaries.append({
                "run_id": view["run_id"],
                "task_id": view.get("task_id"),
                "label": view.get("label", ""),
                "status": view["status"],
                "active": view["active"],
                "error": (view.get("error") or "")[:300] or None,
                "started_at": view.get("started_at"),
                "finished_at": view.get("finished_at"),
                "updated_at": view.get("updated_at"),
                "proposer": {"name": proposer.get("name"), "model": (proposer.get("options") or {}).get("model")},
                "result": {key: result.get(key) for key in ("seed_score", "incumbent_score", "promotions",
                                                            "seconds")} if result else None,
                "generations": (view.get("config") or {}).get("generations"),
                "generations_done": len(view["generations"]),
                "generations_finished": sum(1 for item in view["generations"] if "incumbent_after" in item),
                "sealed": [item["stage"] for item in view["sealed"]],
            })
        return summaries

    def run(self, run_id: str) -> dict[str, Any]:
        with Ledger(self.ledger_path) as ledger:
            view = self._run_view(ledger, run_id)
            if view["status"] == "finished":
                try:
                    view["sealing"] = self._sealing(ledger, view)
                except (OSError, KeyError, ValueError) as error:  # e.g. the data are not available here
                    view["sealing"] = {"error": f"{type(error).__name__}: {error}"}
            return view

    def _sealing(self, ledger: Ledger, view: dict[str, Any]) -> dict[str, Any]:
        """Per sealed stage: has the final method been evaluated in any run of this ledger, and with what result.

        Exposures belong to the method and the data, not to a run, so a second run that ends with the same
        method shows the first run's result and cannot evaluate again.
        """

        bench = self.bench(view["task_id"])
        genome = bench.genome(view["result"]["incumbent"])
        claimed = {item["key"]: item for item in ledger.exposures(method=genome.digest)}
        sealing: dict[str, Any] = {"method": genome.digest}
        for stage in SEALED_ORDER:
            exposure = claimed.get(exposure_key(bench, genome, stage))
            entry: dict[str, Any] = {"evaluated": exposure is not None}
            if exposure is not None:
                reports = view["sealed"] if exposure["run_id"] == view["run_id"] else [
                    event.body for event in ledger.events(exposure["run_id"], kind="sealed_evaluated")]
                entry |= {
                    "run_id": exposure["run_id"],
                    "at": exposure["at"],
                    "report": next((item for item in reports
                                    if item["stage"] == stage and item["digest"] == genome.digest), None),
                }
            sealing[stage] = entry
        return sealing

    def start(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            bench = self.bench(str(body["task"]))
            genome = body.get("genome")
            seed = bench.genome(genome) if genome else bench.preset(str(body.get("preset", "")))
            config = RunConfig(
                generations=int(body.get("generations", 3)), candidates=int(body.get("candidates", 4)),
                seed=int(body.get("seed", 0)), max_step=float(body.get("max_step", 0.15)),
                max_changes=int(body.get("max_changes", 3)),
            )
            proposer = make_proposer(str(body.get("proposer", "sweep")), body.get("options") or {})
        except (KeyError, TypeError, ValueError) as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{type(error).__name__}: {error}") from None
        run_id = f"run:{uuid.uuid4().hex[:12]}"
        stop = threading.Event()

        def work() -> None:
            try:
                with Ledger(self.ledger_path) as ledger:
                    Arena(bench, ledger, config, proposer).run(seed, run_id=run_id, label=str(body.get("label", "")),
                                                               stop=stop.is_set)
            except RunStopped:
                pass
            except Exception as error:  # logged as run_failed by the arena once the run has started
                self._errors[run_id] = f"{type(error).__name__}: {error}"

        thread = threading.Thread(target=work, name=run_id, daemon=True)
        self._stops[run_id] = stop
        self._threads[run_id] = thread
        thread.start()
        return {"run_id": run_id}

    def stop(self, run_id: str) -> dict[str, Any]:
        if run_id not in self._stops:
            raise ApiError(HTTPStatus.NOT_FOUND, f"{run_id} is not running in this service")
        self._stops[run_id].set()
        return {"run_id": run_id, "stopping": True}

    def _finished(self, run_id: str) -> tuple[dict[str, Any], Bench]:
        view = self.run(run_id)
        if view["status"] != "finished":
            raise ApiError(HTTPStatus.CONFLICT, f"{run_id} has not finished")
        return view, self.bench(view["task_id"])

    def seal(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        stage = str(body.get("stage"))
        view, bench = self._finished(run_id)
        genome = bench.genome(view["result"]["incumbent"])
        with Ledger(self.ledger_path) as ledger:
            try:
                return seal(bench, ledger, genome, stage, run_id=run_id)
            except (PermissionError, ValueError) as error:
                raise ApiError(HTTPStatus.CONFLICT, str(error)) from None

    def cells(self, run_id: str) -> dict[str, Any]:
        view = self.run(run_id)
        bench = self.bench(view["task_id"])
        methods = {"seed": bench.genome(view["seed"])}
        if view.get("result"):
            methods["final"] = bench.genome(view["result"]["incumbent"])
        return {
            name: {stage: {"score": (evaluation := bench.evaluate(genome, stage)).score(),
                           "cells": evaluation.cells()}
                   for stage in ("model_selection", "calibration_uq")}
            for name, genome in methods.items()
        }

    def report(self, run_id: str) -> str:
        view = self.run(run_id)
        return render_run(view, self.bench(view["task_id"]) if view.get("result") else None)


def _handler(service: Service) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ecogym"

        def log_message(self, format: str, *args: Any) -> None:  # quiet
            return None

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: Any) -> None:
            self._send(status, json.dumps(jsonable(value), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _authorized(self) -> bool:
            given = (self.headers.get("Authorization") or "").encode("utf-8", "replace")
            return service.token is None or hmac.compare_digest(given, f"Bearer {service.token}".encode())

        def _body(self) -> dict[str, Any]:
            """A POST's JSON object; checked before any route runs."""

            if self.headers.get("X-Ecogym") != "1":
                raise ApiError(HTTPStatus.FORBIDDEN, "POST requests need the X-Ecogym: 1 header")
            length = int(self.headers.get("Content-Length") or 0)
            if self.headers.get("Transfer-Encoding") or length > MAX_BODY:
                self.close_connection = True  # the body stays unread; it must not be parsed as a request
                if length > MAX_BODY:
                    raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body too large")
                raise ApiError(HTTPStatus.LENGTH_REQUIRED, "POST bodies need a Content-Length")
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ApiError(HTTPStatus.BAD_REQUEST, "the body must be a JSON object")
            return body

        def _route(self, method: str) -> None:
            url = urlparse(self.path)
            try:
                if method == "GET" and url.path == "/":
                    return self._send(200, resources.files("ecogym.web").joinpath("index.html").read_bytes(), HTML)
                if not url.path.startswith("/api/"):
                    raise ApiError(HTTPStatus.NOT_FOUND, "not found")
                if not self._authorized():
                    raise ApiError(HTTPStatus.UNAUTHORIZED, "missing or wrong token")
                body = self._body() if method == "POST" else {}
                parts = [unquote(part) for part in url.path.removeprefix("/api/").split("/") if part]
                for route in ROUTES:
                    args = route.match(method, parts)
                    if args is None:
                        continue
                    query = parse_qs(url.query)
                    answer = getattr(service, route.handler)(
                        *args, *([body] if route.body else []),
                        **{name: (query.get(name) or [None])[0] for name in route.query})
                    if route.html:
                        return self._send(route.status, answer.encode("utf-8"), HTML)
                    return self._json(route.status, answer)
                raise ApiError(HTTPStatus.NOT_FOUND, f"no route {method} {url.path}")
            except ApiError as error:
                self._json(error.status, {"error": str(error)})
            except (json.JSONDecodeError, KeyError, ValueError) as error:
                self._json(400, {"error": f"{type(error).__name__}: {error}"})
            except (BrokenPipeError, ConnectionResetError):
                raise  # the client went away; the server drops it quietly
            except Exception as error:  # anything else (e.g. missing data files) still gets an answer
                traceback.print_exc()
                self._json(500, {"error": f"{type(error).__name__}: {error}"})

        def do_GET(self) -> None:
            self._route("GET")

        def do_POST(self) -> None:
            self._route("POST")

    return Handler


class _Server(ThreadingHTTPServer):
    """A browser that leaves before its answer is written is normal (the proxy then drops the request)."""

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            super().handle_error(request, client_address)


def make_server(service: Service, host: str = "127.0.0.1", port: int = 8850) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1") and not service.token:
        raise ValueError("binding beyond loopback needs --token")
    return _Server((host, port), _handler(service))


__all__ = ["ROUTES", "ApiError", "Route", "Service", "make_server"]
