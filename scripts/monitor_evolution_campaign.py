#!/usr/bin/env python3
"""Monitor explicitly listed local runs and pause at bounded campaign limits.

Read-only by default. --watch enables pause requests, never create/resume.
The token threshold uses delayed provider reports and is NOT a hard cap.
State, decisions and bounded public metrics are stored beside the manifest.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from ecologyrsi_dsh.application.campaign import (
    CampaignLimits, observe_progress, pause_reason, utc_timestamp,
)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--origin", default="http://127.0.0.1:8777")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    origin = urlparse(args.origin)
    if (origin.scheme != "http" or origin.hostname not in {"127.0.0.1", "localhost", "::1"}
            or origin.username or origin.password or origin.path not in {"", "/"}
            or origin.query or origin.fragment):
        parser.error("origin must be a local HTTP service origin")
    token = os.environ.get("ECOLOGYRSI_SERVICE_TOKEN", "")
    if args.token_file:
        for line in args.token_file.read_text().splitlines():
            key, _, value = line.strip().removeprefix("export ").partition("=")
            if key == "ECOLOGYRSI_SERVICE_TOKEN":
                parts = shlex.split(value)
                if len(parts) != 1:
                    parser.error("invalid service token assignment")
                token = parts[0]
    if not token:
        parser.error("ECOLOGYRSI_SERVICE_TOKEN or --token-file is required")
    config = json.loads(args.manifest.read_text())
    if config.get("schema_version") != "ecologyrsi-dsh.evolution-campaign/1":
        parser.error("unsupported campaign schema")
    runs = config["runs"]
    if not runs or len({r["run_id"] for r in runs}) != len(runs):
        parser.error("campaign requires unique, explicitly listed run IDs")
    limits = CampaignLimits(utc_timestamp(config["deadline"]),
                            config["reported_token_pause_threshold_per_run"],
                            config["stall_seconds"])
    interval = config["poll_seconds"]
    if isinstance(interval, bool) or not isinstance(interval, int) or not 5 <= interval <= 60:
        parser.error("poll_seconds must be an integer from 5 to 60")
    directory = args.manifest.resolve().parent
    # One writer across manual launches and heartbeat invocations.
    lock = (directory / "monitor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"status": "monitor_already_running"}))
        return 0
    state_path = directory / "monitor-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    def request(path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = Request(args.origin.rstrip("/") + "/api/" + path, data=data,
                      headers={"Authorization": "Bearer " + token,
                               "Content-Type": "application/json"})
        with urlopen(req, timeout=20) as response:
            return json.load(response)

    while True:
        summaries = []
        inactive = 0
        for run in runs:
            rid = run["run_id"]
            saved = state.setdefault(rid, {})
            try:
                envelope = request("runs/" + quote(rid, safe="") + "?view=monitor")
                p = envelope["projection"]
                if p.get("run_id") != rid:
                    raise ValueError("monitor response run identity mismatch")
                now = time.time()
                saved = observe_progress(p, saved, now=now)
                state[rid] = saved
                run_limits = CampaignLimits(limits.deadline,
                    run.get('reported_token_pause_threshold', limits.reported_token_pause_threshold),
                    limits.stall_seconds)
                reason = pause_reason(p, run_limits, now=now,
                                      last_progress_at=saved["last_progress_at"])
                progress = p.get("execution_progress") or {}
                stage = progress.get("stage_progress") or {}
                summary = {"label": run["label"], "run_id": rid,
                           "status": p["status"], "phase": stage.get("evaluation_phase") or progress.get("current_stage"),
                           "completed_origins": stage.get("run_completed_origins", stage.get("completed_origins", 0)),
                           "failed_origins": stage.get("phase_failed_origins") if "phase_outcomes_verified" in stage else stage.get("failed_samples"),
                           "failed_origins_scope": stage.get("phase_outcome_scope", stage.get("outcome_scope", stage.get("evaluation_phase"))),
                           "outcomes_verified": stage.get("phase_outcomes_verified", stage.get("outcomes_verified", False)),
                           "candidates": p.get("candidates_count", 0),
                           "reported_tokens": p.get("tokens_used"),
                           "usage_available": p.get("token_usage_available"),
                           "pause_reason": reason or saved.get("pause_reason"),
                           "failure_code": p.get("failure_code") or p.get("pause_code"),
                           "projection_revision": p.get("projection_revision")}
                write_json(directory / (run["label"] + "-monitor.json"), envelope)
                if reason and args.watch:
                    # Reuse a pending key after timeouts/restarts. A new start
                    # incarnation/revision gets a new command after settlement.
                    if not saved.get("pause_pending"):
                        saved.update(pause_pending=True, pause_reason=reason,
                                     pause_key=f"campaign:{config['campaign_id']}:{rid}:{p['projection_revision']}:{reason}")
                        write_json(state_path, state)
                    receipt = request("runs/" + quote(rid, safe="") + "/control",
                                      {"action": "pause", "idempotency_key": saved["pause_key"]})
                    saved["pause_response_received"] = True
                    with (directory / "decisions.jsonl").open("a") as log:
                        log.write(json.dumps({"time": now, **summary, "action": "pause_requested"}) + "\n")
                if p["status"] in {"paused", "completed", "failed", "cancelled"}:
                    saved["pause_pending"] = False
                    inactive += 1
                summaries.append(summary)
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError) as error:
                # Never print remote bodies, headers or credentials. Unreadable
                # runs remain unresolved; they are not counted as stopped.
                summaries.append({"run_id": rid, "status": "monitor_error",
                                  "error_type": type(error).__name__,
                                  "http_status": getattr(error, "code", None)})
        write_json(state_path, state)
        result = {"observed_at": time.time(), "runs": summaries,
                  "all_inactive": inactive == len(runs), "watch": args.watch}
        write_json(directory / "summary.json", result)
        with (directory / "history.jsonl").open("a") as log:
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not args.watch or inactive == len(runs):
            return 0
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
