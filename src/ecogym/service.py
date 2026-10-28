"""Thin HTTP service: a JSON API over the ledger plus the static web workbench.

    GET  /api/catalog                tasks (targets with labels and units, horizons, gates, presets),
                                     solvers, proposers, ledger path
    GET  /api/runs                   run summaries
    GET  /api/runs/<id>              one folded run (``active`` while it runs); finished runs carry
                                     ``sealing``: per sealed stage, whether the final method has been
                                     evaluated in any run of this ledger, and the result
    GET  /api/runs/<id>/cells        seed and final method per cell (model_selection, calibration_uq)
    GET  /api/runs/<id>/report       static HTML report
    POST /api/runs                   start a run in the background
    POST /api/runs/<id>/stop         stop after the current generation
    POST /api/runs/<id>/seal         sealed evaluation {"stage": "validation" | "final_test"}

The service binds to 127.0.0.1 by default. POSTs must carry ``X-Ecogym: 1``
(a custom header no cross-site form can send); with ``token`` set, every API
call needs ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import json
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

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
STATIC = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8",
          "style.css": "text/css; charset=utf-8"}
SEALED_ORDER = tuple(stage for stage in STAGES if stage in SEALED_STAGES)


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
    }


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class Service:
    def __init__(self, ledger_path: Path, *, token: str | None = None) -> None:
        self.ledger_path = Path(ledger_path)
        self.token = token
        self._benches: dict[str, Bench] = {}
        self._lock = threading.Lock()
        self._stops: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._errors: dict[str, str] = {}  # failures raised before a run could log anything
        Ledger(self.ledger_path).close()  # create the file and schema up front

    def bench(self, task_id: str) -> Bench:
        with self._lock:
            if task_id not in self._benches:
                self._benches[task_id] = Bench.open(task_id)
            return self._benches[task_id]

    def catalog(self) -> dict[str, Any]:
        return {
            "version": __version__,
            "ledger": str(self.ledger_path),
            "tasks": [_task_entry(task) for task in task_specs()],
            "solvers": [s.describe() for s in solver_specs()],
            "proposers": sorted(PROPOSERS),
        }

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
            views = [self._run_view(ledger, run_id) for run_id in ledger.runs()]
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

    def seal(self, run_id: str, stage: str) -> dict[str, Any]:
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
            return service.token is None or self.headers.get("Authorization") == f"Bearer {service.token}"

        def _route(self, method: str) -> None:
            path = urlparse(self.path).path
            try:
                if method == "GET" and (path == "/" or path.startswith("/static/")):
                    name = "index.html" if path == "/" else path.removeprefix("/static/")
                    if name not in STATIC:
                        raise ApiError(HTTPStatus.NOT_FOUND, "not found")
                    data = resources.files("ecogym.web").joinpath(name).read_bytes()
                    return self._send(200, data, STATIC[name])
                if not path.startswith("/api/"):
                    raise ApiError(HTTPStatus.NOT_FOUND, "not found")
                if not self._authorized():
                    raise ApiError(HTTPStatus.UNAUTHORIZED, "missing or wrong token")
                parts = [unquote(part) for part in path.removeprefix("/api/").split("/") if part]
                if method == "POST":
                    if self.headers.get("X-Ecogym") != "1":
                        raise ApiError(HTTPStatus.FORBIDDEN, "POST requests need the X-Ecogym: 1 header")
                    length = int(self.headers.get("Content-Length") or 0)
                    if length > MAX_BODY:
                        raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body too large")
                    body = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(body, dict):
                        raise ApiError(HTTPStatus.BAD_REQUEST, "the body must be a JSON object")
                    if parts == ["runs"]:
                        return self._json(201, service.start(body))
                    if len(parts) == 3 and parts[0] == "runs" and parts[2] == "stop":
                        return self._json(200, service.stop(parts[1]))
                    if len(parts) == 3 and parts[0] == "runs" and parts[2] == "seal":
                        return self._json(200, service.seal(parts[1], str(body.get("stage"))))
                elif method == "GET":
                    if parts == ["catalog"]:
                        return self._json(200, service.catalog())
                    if parts == ["runs"]:
                        return self._json(200, service.runs())
                    if len(parts) == 2 and parts[0] == "runs":
                        return self._json(200, service.run(parts[1]))
                    if len(parts) == 3 and parts[0] == "runs" and parts[2] == "cells":
                        return self._json(200, service.cells(parts[1]))
                    if len(parts) == 3 and parts[0] == "runs" and parts[2] == "report":
                        return self._send(200, service.report(parts[1]).encode("utf-8"), STATIC["index.html"])
                raise ApiError(HTTPStatus.NOT_FOUND, f"no route {method} {path}")
            except ApiError as error:
                self._json(error.status, {"error": str(error)})
            except (json.JSONDecodeError, KeyError, ValueError) as error:
                self._json(400, {"error": f"{type(error).__name__}: {error}"})

        def do_GET(self) -> None:
            self._route("GET")

        def do_POST(self) -> None:
            self._route("POST")

    return Handler


def make_server(service: Service, host: str = "127.0.0.1", port: int = 8850) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1") and not service.token:
        raise ValueError("binding beyond loopback needs --token")
    return ThreadingHTTPServer((host, port), _handler(service))


__all__ = ["ApiError", "Service", "make_server"]
