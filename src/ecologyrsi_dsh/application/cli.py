"""Small command-line entry point for the local evolution runtime."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import sysconfig
from typing import Any

from .queries import RunQueries
from .config import bind_toy_dataset, load_json_object, load_local_config, load_task_manifest
from ..data.registry import DatasetRegistry
from ..evolution.strategies import FakeDSHAdapter
from ..core.director import EvolutionDirector
from ..core.ledger import EventLedger, SCHEMA_VERSION
from ..core.models import TaskManifest, digest
from ..presentation.reporting import (
    export_errors,
    run_export,
    run_summary,
    state_snapshot,
    write_json_atomic,
)
from ..data.toy import ToyCropSoilWater
from ..version import __version__

def _open(path: str | Path) -> EventLedger:
    value = str(path)
    return EventLedger(":memory:" if value == ":memory:" else Path(value).expanduser())


def _demo_manifest(args: argparse.Namespace) -> TaskManifest:
    manifest_path = getattr(args, "manifest", None)
    if manifest_path:
        return load_task_manifest(manifest_path)
    candidates = getattr(args, "candidates", None)
    seed = getattr(args, "seed", None)
    task = TaskManifest(
        task_id="toy-forecast",
        objective="predict soil water",
        domain_pack="crop-soil-water@toy",
        visible_datasets=("generated-toy-series@1",),
        budget={"max_candidates": 3 if candidates is None else candidates},
        seed=7 if seed is None else seed,
        seed_policy="fixed",
        metadata={
            "scientific_scope": "prediction_demo_non_causal",
            "evaluation_partition": "validation",
        },
    )
    return bind_toy_dataset(task, required=True)


def demo(args: argparse.Namespace) -> int:
    ledger = _open(args.db)
    try:
        task = _demo_manifest(args)
        director = EvolutionDirector(ledger, FakeDSHAdapter(max_proposals=task.max_candidates))
        state = director.start_evolution(task, run_id=args.run_id)
        toy = ToyCropSoilWater(seed=task.seed)
        for _ in range(task.max_candidates):
            candidate = director.propose_and_spawn(state.run.run_id)
            current = director.state(state.run.run_id)
            proposal = current.proposal(candidate.proposal_id)
            evaluation = toy.evaluate_candidate(
                state.run.run_id, candidate, proposal, split="validation"
            )
            director.evaluate_and_decide(evaluation)
        director.complete_run(state.run.run_id)
        final_state = director.state(state.run.run_id)
        output = state_snapshot(final_state)
        output["summary"] = run_summary(final_state)
        print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        ledger.close()
    return 0


def status(args: argparse.Namespace) -> int:
    ledger = _open(args.db)
    try:
        state = RunQueries(EvolutionDirector(ledger)).state(args.run_id)
        print(json.dumps(state_snapshot(state), ensure_ascii=False, indent=2))
    finally:
        ledger.close()
    return 0


def summary(args: argparse.Namespace) -> int:
    ledger = _open(args.db)
    try:
        state = RunQueries(EvolutionDirector(ledger)).state(args.run_id)
        print(json.dumps(run_summary(state), ensure_ascii=False, indent=2))
    finally:
        ledger.close()
    return 0


def export_run(args: argparse.Namespace) -> int:
    ledger = _open(args.db)
    try:
        state = RunQueries(EvolutionDirector(ledger)).state(args.run_id)
        payload = run_export(state)
        target = write_json_atomic(args.output, payload, force=bool(args.force))
        print(
            json.dumps(
                {
                    "output": str(target),
                    "run_id": args.run_id,
                    "export_digest": payload["export_digest"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        ledger.close()
    return 0


def export_agent_policy(args: argparse.Namespace) -> int:
    from ..integrations.agent_policy_bundle import export_policy_bundle
    ledger = _open(args.db)
    try:
        state = EvolutionDirector(ledger).state(args.run_id)
        artifact = state.artifact_for(args.candidate_id)
        if artifact is None:
            raise ValueError("candidate has no sealed inference artifact")
        payload = export_policy_bundle(artifact)
        write_json_atomic(args.output, payload, force=args.force)
        print(json.dumps({"bundle_digest": payload["bundle_digest"], "output": str(args.output)}, ensure_ascii=False))
    finally:
        ledger.close()
    return 0


def verify_export(args: argparse.Namespace) -> int:
    payload = load_json_object(args.path)
    stored = payload.get("export_digest")
    unsigned = dict(payload)
    unsigned.pop("export_digest", None)
    try:
        computed = digest(unsigned)
    except (TypeError, ValueError):
        computed = None
    errors = export_errors(payload)
    valid = not errors
    result = {
        "valid": valid,
        "path": str(Path(args.path).expanduser().resolve()),
        "stored_digest": stored,
        "computed_digest": computed,
        "format": payload.get("format"),
        "format_version": payload.get("format_version"),
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if valid else 1


def import_export(args: argparse.Namespace) -> int:
    """Import an export bundle into an append-only SQLite ledger."""

    payload = load_json_object(args.path)
    errors = export_errors(payload)
    if errors:
        raise ValueError("invalid export bundle: " + "; ".join(errors))
    summary = payload["summary"]
    run_id = str(summary["run_id"])
    events = payload["events"]
    ledger = _open(args.db)
    try:
        for raw in events:
            ledger.append(
                str(raw["run_id"]),
                str(raw["kind"]),
                dict(raw["payload"]),
                event_id=str(raw["event_id"]),
                created_at=str(raw["created_at"]),
            )
        state = EvolutionDirector(ledger).replay(run_id)
        actual = run_summary(state)
        if actual["manifest_digest"] != summary.get("manifest_digest"):
            raise ValueError("imported manifest digest does not match export summary")
        if len(state.events) != len(events):
            raise ValueError("imported event count does not match export")
        print(json.dumps(actual, ensure_ascii=False, indent=2))
    finally:
        ledger.close()
    return 0


def doctor(args: argparse.Namespace) -> int:
    """Run local preflight checks without appending to the event stream."""

    issues: list[str] = []
    report: dict[str, Any] = {
        "ok": False,
        "package": "ecologyrsi-dsh",
        "package_version": __version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "schema_expected": SCHEMA_VERSION,
    }
    ledger: EventLedger | None = None
    try:
        ledger = _open(args.db)
        integrity = ledger.integrity_check()
        report["database"] = {
            "path": str(Path(args.db).expanduser().resolve()),
            "schema_version": ledger.schema_version,
            "integrity_check": integrity,
            "event_count": ledger.count(),
        }
        if integrity.lower() != "ok":
            issues.append(f"sqlite integrity check: {integrity}")
        if ledger.schema_version != SCHEMA_VERSION:
            issues.append("sqlite schema version is unsupported")

        replay_errors: list[dict[str, str]] = []
        runs: list[dict[str, Any]] = []
        director = EvolutionDirector(ledger)
        for run_id in ledger.run_ids():
            try:
                state = director.replay(run_id)
                runs.append(run_summary(state))
            except Exception as exc:  # corrupt streams are reported, not hidden
                replay_errors.append({"run_id": run_id, "error": str(exc)})
        report["runs"] = {"count": len(runs), "summaries": runs, "replay_errors": replay_errors}
        issues.extend(f"run {item['run_id']}: {item['error']}" for item in replay_errors)

        pending = list(ledger.pending_command_keys())
        report["command_receipts"] = {
            "total": ledger.command_count(),
            "completed": ledger.command_count(status="completed"),
            "pending": pending,
        }
        if pending:
            issues.append("pending idempotent command receipts require review")
    except Exception as exc:
        report["database_error"] = str(exc)
        issues.append(f"database unavailable: {exc}")
    finally:
        if ledger is not None:
            ledger.close()

    manifest_path = getattr(args, "manifest", None)
    if manifest_path:
        try:
            manifest = load_task_manifest(manifest_path)
            report["manifest"] = {
                "path": str(Path(manifest_path).expanduser().resolve()),
                "task_id": manifest.task_id,
                "digest": manifest.digest,
                "dataset_id": manifest.dataset,
                "dataset_digest": manifest.metadata.get("dataset_digest"),
            }
        except Exception as exc:
            report["manifest"] = {"path": str(manifest_path), "error": str(exc)}
            issues.append(f"manifest: {exc}")
    else:
        report["manifest"] = {"checked": False}

    try:
        from ..api.shared import _PLUGIN_FILES, _plugin_root

        plugin_root = _plugin_root()
        missing = [name for name in _PLUGIN_FILES if not (plugin_root / name).is_file()]
        report["plugin"] = {
            "root": str(plugin_root),
            "files": sorted(_PLUGIN_FILES),
            "missing": missing,
        }
        if missing:
            issues.append("plugin static resources are missing")
    except Exception as exc:
        report["plugin"] = {"error": str(exc)}
        issues.append(f"plugin preflight: {exc}")

    report["issues"] = issues
    report["ok"] = not issues
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def serve_command(args: argparse.Namespace) -> int:
    from ..api.runtime import serve

    config = load_local_config(args.config) if args.config else None
    if config is not None and config.manifest:
        # Validate the configured manifest at startup; the HTTP API still
        # accepts an explicit manifest in each create request.
        load_task_manifest(config.manifest)
    db = args.db or (config.db if config else "ecologyrsi-dsh.sqlite3")
    host = args.host or (config.host if config else "127.0.0.1")
    port = args.port or (config.port if config else 8765)
    serve(host=host, port=port, db=db)
    return 0


def data_audit(args: argparse.Namespace) -> int:
    registry = DatasetRegistry(data_root=args.data_root)
    dataset_ids = tuple(args.dataset_ids) if args.dataset_ids else None
    print(json.dumps(registry.audit_data(dataset_ids), ensure_ascii=False, indent=2))
    return 0


def data_fetch(args: argparse.Namespace) -> int:
    registry = DatasetRegistry(data_root=args.data_root)
    result = registry.fetch_data(
        tuple(args.dataset_ids),
        extract=not bool(args.archive_only),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _bundled_dsh_plugin_archive(plugin_root: Path) -> Path:
    """Resolve only the archive that belongs to this Python package version."""

    archive = (
        plugin_root
        / "dist"
        / f"ecologyrsi-dsh-evolution-plugin-{__version__}.tgz"
    )
    if not archive.is_file():
        raise RuntimeError(
            "bundled DSH plugin archive is missing for package version "
            f"{__version__}"
        )
    return archive


def install_dsh_runtime(args: argparse.Namespace) -> int:
    """Install the bundled Cordis plugin and role presets into one DSH profile."""

    installed_root = (
        Path(sysconfig.get_path("data")) / "share" / "ecologyrsi-dsh"
    )
    source_root = Path(__file__).resolve().parents[3]
    asset_root = next(
        (
            root
            for root in (installed_root, source_root)
            if (root / "scripts" / "install_dsh_ecology_runtime.mjs").is_file()
            and (root / "integrations" / "dsh_ecology_plugin" / "package.json").is_file()
        ),
        None,
    )
    if asset_root is None:
        raise RuntimeError("bundled DSH runtime assets are missing")
    plugin_root = asset_root / "integrations" / "dsh_ecology_plugin"
    archive = _bundled_dsh_plugin_archive(plugin_root)
    installer = asset_root / "scripts" / "install_dsh_ecology_runtime.mjs"
    static_root = asset_root / "plugins" / "ecology_evolution"
    environment = dict(os.environ)
    if args.dsh_home:
        environment["DSH_HOME"] = str(Path(args.dsh_home).expanduser().resolve())
    if args.dsh_bin:
        environment["DSH_BIN"] = args.dsh_bin
    command = [
        "node",
        str(installer),
        "--plugin-root",
        str(plugin_root),
        "--static-root",
        str(static_root),
        "--tgz",
        str(archive),
        "--profile",
        args.profile,
    ]
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(
            f"DSH runtime installer exited with status {completed.returncode}"
        )
    return 0


def advance_command(args: argparse.Namespace) -> int:
    from .generation_execution import execute_generation, complete_if_budget_exhausted
    from .runtime import ApplicationRuntime
    from ..execution.ownership import RuntimeOwnerLease
    from ..core.models import RunStatus
    if type(args.steps) is not int or not 1 <= args.steps <= 100:
        raise ValueError("steps must be between 1 and 100")
    owner = RuntimeOwnerLease(args.db)
    try:
        ledger = _open(args.db)
        try:
            services = ApplicationRuntime(ledger)
            state = complete_if_budget_exhausted(services, args.run_id)
            for _ in range(args.steps):
                if state.run.status is not RunStatus.RUNNING:
                    break
                state = execute_generation(services, args.run_id)
            print(json.dumps(state_snapshot(state), ensure_ascii=False, indent=2, allow_nan=False))
        finally:
            ledger.close()
    finally:
        owner.release()
    return 0






def diagnose_training_fit_command(args: argparse.Namespace) -> int:
    from ..science.diagnostics import candidates_from_payload, run_training_fit_diagnostics
    candidates = candidates_from_payload(json.loads(Path(args.candidates).read_text())) if args.candidates else None
    view = DatasetRegistry(data_root=args.data_root).selection_view(args.dataset, args.episode)
    report = run_training_fit_diagnostics(view, candidates=candidates, folds=args.folds,
        initial_fit_fraction=args.initial_fit_fraction, purge_hours=args.purge_hours)
    write_json_atomic(Path(args.output), report)
    print(json.dumps({"output": str(Path(args.output).resolve()), "scope": report["scope"],
        "common_origins": [fold["common_origin_count"] for fold in report["folds"]], "remote_requests": 0}, ensure_ascii=False))
    return 0


def model_preflight_command(args: argparse.Namespace) -> int:
    from urllib.request import Request, urlopen
    from urllib.parse import urlparse
    endpoint = args.api_base.rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("model-preflight requires an HTTP loopback API endpoint")
    payload = load_json_object(args.request)
    token = os.environ.get(args.token_env, "")
    if not token:
        raise ValueError(f"set {args.token_env} to the local service token")
    request = Request(endpoint + "/model-preflight", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token}, method="POST")
    with urlopen(request, timeout=250) as response:
        result = json.load(response)
    write_json_atomic(Path(args.output), result)
    print(json.dumps({"output": str(Path(args.output).resolve()), "passed": result.get("passed"), "scope": result.get("scope")}, ensure_ascii=False))
    return 0 if result.get("passed") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecologyrsi-dsh", description="Minimal replayable evolution mode"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    demo_parser = sub.add_parser("demo", help="run the deterministic toy evolution loop")
    demo_parser.add_argument("--db", default="ecologyrsi-dsh.sqlite3")
    demo_parser.add_argument("--run-id", default="run:demo")
    demo_parser.add_argument("--manifest", help="JSON TaskManifest (uses its fixed seed/budget)")
    demo_parser.add_argument("--candidates", type=int, default=None)
    demo_parser.add_argument("--seed", type=int, default=None)
    demo_parser.set_defaults(handler=demo)

    status_parser = sub.add_parser("status", help="replay and print a stored run")
    status_parser.add_argument("run_id")
    status_parser.add_argument("--db", default="ecologyrsi-dsh.sqlite3")
    status_parser.set_defaults(handler=status)

    summary_parser = sub.add_parser("summary", help="print a compact run summary")
    summary_parser.add_argument("run_id")
    summary_parser.add_argument("--db", default="ecologyrsi-dsh.sqlite3")
    summary_parser.set_defaults(handler=summary)

    export_parser = sub.add_parser("export", help="export a replayable run bundle")
    export_parser.add_argument("run_id")
    export_parser.add_argument("--db", default="ecologyrsi-dsh.sqlite3")
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--force", action="store_true", help="allow replacing output")
    export_parser.set_defaults(handler=export_run)

    policy_parser = sub.add_parser("export-policy", help="export a deployable Agent prediction policy")
    policy_parser.add_argument("run_id")
    policy_parser.add_argument("candidate_id")
    policy_parser.add_argument("--db", required=True)
    policy_parser.add_argument("--output", required=True)
    policy_parser.add_argument("--force", action="store_true")
    policy_parser.set_defaults(handler=export_agent_policy)

    verify_parser = sub.add_parser("verify", help="verify an exported run bundle digest")
    verify_parser.add_argument("path")
    verify_parser.set_defaults(handler=verify_export)

    import_parser = sub.add_parser("import", help="import and replay an exported run bundle")
    import_parser.add_argument("path")
    import_parser.add_argument("--db", required=True)
    import_parser.set_defaults(handler=import_export)

    doctor_parser = sub.add_parser("doctor", help="check local runtime and replay health")
    doctor_parser.add_argument("--db", default="ecologyrsi-dsh.sqlite3")
    doctor_parser.add_argument("--manifest", help="also validate a TaskManifest JSON")
    doctor_parser.set_defaults(handler=doctor)

    serve_parser = sub.add_parser("serve", help="start the localhost JSON API")
    serve_parser.add_argument("--config", help="JSON local runtime configuration")
    serve_parser.add_argument("--db")
    serve_parser.add_argument("--host")
    serve_parser.add_argument("--port", type=int)
    serve_parser.set_defaults(handler=serve_command)

    data_parser = sub.add_parser("data", help="审计或准备真实 AGC 数据")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)

    audit_parser = data_commands.add_parser("audit", help="只读检查归档和解压文件")
    audit_parser.add_argument(
        "dataset_ids",
        nargs="*",
        help="数据集标识；省略时检查两份可运行 AGC 数据",
    )
    audit_parser.add_argument(
        "--data-root",
        help="数据根目录；默认读取 ECOLOGYRSI_DATA_ROOT",
    )
    audit_parser.set_defaults(handler=data_audit)

    fetch_parser = data_commands.add_parser("fetch", help="下载、校验并安全解压数据")
    fetch_parser.add_argument(
        "dataset_ids",
        nargs="+",
        help="agc_cucumber_2018 和/或 agc_tomato_2019",
    )
    fetch_parser.add_argument(
        "--data-root",
        help="数据根目录；默认读取 ECOLOGYRSI_DATA_ROOT",
    )
    fetch_parser.add_argument(
        "--archive-only",
        action="store_true",
        help="只下载并校验归档，不解压",
    )
    fetch_parser.set_defaults(handler=data_fetch)

    install_parser = sub.add_parser(
        "install-dsh-runtime",
        help="install the bundled DSH-native plugin and role presets",
    )
    install_parser.add_argument("--profile", default="web")
    install_parser.add_argument("--dsh-home")
    install_parser.add_argument("--dsh-bin")
    install_parser.set_defaults(handler=install_dsh_runtime)
    advance = sub.add_parser("advance", help="advance an existing run through the shared application service")
    advance.add_argument("--db", required=True)
    advance.add_argument("--run-id", required=True)
    advance.add_argument("--steps", type=int, default=1)
    advance.set_defaults(handler=advance_command)
    diagnostic = sub.add_parser("diagnose-training-fit", help="仅使用training_fit进行基线/候选前推对照；不调用远程模型")
    diagnostic.add_argument("--dataset", default="agc_cucumber_2018")
    diagnostic.add_argument("--episode", required=True)
    diagnostic.add_argument("--data-root")
    diagnostic.add_argument("--candidates")
    diagnostic.add_argument("--folds", type=int, default=3)
    diagnostic.add_argument("--initial-fit-fraction", type=float, default=.4)
    diagnostic.add_argument("--purge-hours", type=int, default=24)
    diagnostic.add_argument("--output", required=True)
    diagnostic.set_defaults(handler=diagnose_training_fit_command)
    preflight = sub.add_parser("model-preflight", help="通过本地API执行真实工具与结构化输出预检")
    preflight.add_argument("--request", required=True, help="JSON compact create-run request; no run will be created")
    preflight.add_argument("--api-base", default="http://127.0.0.1:8777/api")
    preflight.add_argument("--token-env", default="ECOLOGYRSI_SERVICE_TOKEN")
    preflight.add_argument("--output", required=True)
    preflight.set_defaults(handler=model_preflight_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (KeyError, RuntimeError, ValueError, TypeError, StopIteration, OSError) as exc:
        parser.error(str(exc))
    return 2  # pragma: no cover
