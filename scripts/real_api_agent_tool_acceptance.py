#!/usr/bin/env python3
"""Run a small, non-scientific real-API acceptance over real greenhouse data.

The full training partition is used to fit the registered ridge model. Only a
deterministic, label-independent sample of training_feedback is sent through
the planner/tool/critic loop. The resulting report is engineering evidence and
must never be used for promotion, scientific scoring, or training assets.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from create_delivery_archive import included_source_files, project_version  # noqa: E402
from ecologyrsi_dsh.core.models import digest  # noqa: E402
from ecologyrsi_dsh.core.redaction import (  # noqa: E402
    REDACTED,
    is_sensitive_key,
    public_error_summary,
    safe_error_code,
)
from ecologyrsi_dsh.data.registry import DatasetRegistry  # noqa: E402
from ecologyrsi_dsh.evaluators.gateway_sample_adapter import (  # noqa: E402
    GatewaySampleCollaborationAdapter,
    GatewaySampleTool,
)
from ecologyrsi_dsh.evaluators.greenhouse_prediction import (  # noqa: E402
    EXOGENOUS_RIDGE_MODEL_ID,
    MAX_EXOGENOUS_RIDGE_HISTORY_STEPS,
    ExogenousRidgeConfig,
    fit_predict_exogenous_ridge,
    predict_fitted_exogenous_ridge,
)
from ecologyrsi_dsh.evaluators.sample_execution import (  # noqa: E402
    CollaborativeSampleExecutor,
    SampleExecutionPolicy,
    SamplePredictionRequest,
)
from ecologyrsi_dsh.evaluators.shared_sample_context import (  # noqa: E402
    ORIGIN_SHARED_CONTEXT_PROFILE,
)
from ecologyrsi_dsh.integrations.model_gateway import (  # noqa: E402
    GatewayResponseError,
    ModelGateway,
)
from ecologyrsi_dsh.version import __version__  # noqa: E402
import verify_artifacts as artifact_verifier  # noqa: E402


PROFILE_ID = "real_api_agent_tool_acceptance@1"
RELEASE_BINDING_SCHEMA_VERSION = "ecologyrsi-dsh.release-binding/1"
SOURCE_MANIFEST_SCHEMA_VERSION = "ecologyrsi-dsh.delivery-source-manifest/1"
DEFAULT_DATASET_ID = "agc_cucumber_2018"
DEFAULT_EPISODE_ID = "agc_cucumber_2018:AiCU"
DEFAULT_OUTPUT = Path(os.environ.get("TMPDIR") or "/tmp") / (
    "ecologyrsi-dsh-real-api-agent-tool-acceptance-latest.json"
)
DEFAULT_DIST_DIR = PROJECT_ROOT / "dist"
DEFAULT_HORIZONS = (1, 6, 24)
TARGETS = (
    ("air_temperature", "degC", -10.0, 60.0),
    ("relative_humidity", "percent", 0.0, 100.0),
    ("co2_concentration", "ppm", 0.0, 5000.0),
)
FORBIDDEN_REMOTE_LABEL_KEYS = frozenset(
    {
        "actual",
        "actual_value",
        "ground_truth",
        "label",
        "labels",
        "observed",
        "observation",
        "target_value",
    }
)
GLM_52_MODEL_ALIASES = frozenset(
    {
        "glm-5.2",
        "glm-5-2",
        "glm5.2",
        "glm52",
    }
)
DEEPSEEK_FLASH_MODEL_ALIASES = frozenset(
    {
        "deepseek-flash",
        "deepseek-v4-flash",
        "deepseek-v4-flash-0731",
    }
)


class ReleaseArtifactVerificationError(RuntimeError):
    error_code = "release_artifact_verification_failed"


class ReleaseBindingChangedError(RuntimeError):
    error_code = "release_binding_changed"


class ReportOutputConflictError(ValueError):
    error_code = "report_output_conflict"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise a real planner and critic over agent-selected host tools "
            "without creating a promotable evolution run."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    parser.add_argument("--episode", default=DEFAULT_EPISODE_ID)
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help=(
            "read-only evolution ledger used to select the latest RunCreated "
            "binding for GLM-5.2 and DeepSeek Flash"
        ),
    )
    parser.add_argument(
        "--reference-run-id",
        help="select one RunCreated binding by exact run ID instead of the latest match",
    )
    parser.add_argument(
        "--planner",
        help="optional model-ID assertion; must equal the frozen policy/strategy binding",
    )
    parser.add_argument(
        "--critic",
        help="optional model-ID assertion; must equal the frozen review/judge binding",
    )
    parser.add_argument(
        "--planner-digest",
        help="optional digest assertion; must equal the frozen policy/strategy digest",
    )
    parser.add_argument(
        "--critic-digest",
        help="optional digest assertion; must equal the frozen review/judge digest",
    )
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--gateway-attempts", type=int, default=4)
    parser.add_argument("--sample-attempts", type=int, default=3)
    parser.add_argument("--minimum-coverage", type=float, default=0.8)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=DEFAULT_DIST_DIR,
        help=(
            "directory containing the wheel, sdist, and complete delivery archive "
            "whose SHA-256 values will be bound into the report"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "atomic JSON report path; defaults to a latest report in the system "
            "temporary directory and is written for both pass and failure"
        ),
    )
    return parser


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _stable_file_record(path: Path) -> dict[str, Any]:
    before = path.stat()
    sha256 = _sha256_file(path)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise ReleaseBindingChangedError(
            "release input changed while its identity was calculated"
        )
    return {"size_bytes": before.st_size, "sha256": sha256}


def _source_manifest(
    source_root: Path,
    files: Sequence[Path] | None = None,
) -> dict[str, Any]:
    root = source_root.resolve()
    selected = included_source_files(root) if files is None else list(files)
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in selected:
        if raw_path.is_symlink():
            raise RuntimeError("release source manifest cannot contain symbolic links")
        path = raw_path.resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError("release source manifest path escapes the project root") from exc
        if relative in seen:
            raise RuntimeError(f"duplicate release source manifest path: {relative}")
        if not path.is_file():
            raise RuntimeError(f"invalid release source manifest file: {relative}")
        seen.add(relative)
        entries.append({"path": relative, **_stable_file_record(path)})
    entries.sort(key=lambda item: item["path"].encode("utf-8"))
    canonical = "".join(
        f'{item["sha256"]}  {item["path"]}\n' for item in entries
    ).encode("utf-8")
    return {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "selection": "complete_delivery_source_files@1",
        "canonicalization": "utf8_sha256sum_lines_posix_path_bytewise_sorted@1",
        "file_count": len(entries),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": entries,
    }


def _one_release_artifact(paths: Sequence[Path], kind: str) -> Path:
    files = [
        path.resolve() for path in paths if path.is_file() and not path.is_symlink()
    ]
    if len(files) != 1:
        raise RuntimeError(
            f"expected exactly one {kind} release artifact, found {len(files)}"
        )
    return files[0]


def _verify_release_artifacts(
    artifacts: Mapping[str, Path],
    dist: Path,
    source_root: Path,
) -> None:
    try:
        artifact_verifier.verify_wheel(
            artifacts["wheel"], __version__, source_root
        )
        artifact_verifier.verify_sdist(
            artifacts["sdist"], source_root, __version__
        )
        artifact_verifier.verify_delivery_archive(
            artifacts["delivery_archive"],
            artifacts["wheel"],
            artifacts["sdist"],
            __version__,
            source_root,
        )
        artifact_verifier.verify_external_checksums(
            dist,
            (
                artifacts["wheel"],
                artifacts["sdist"],
                artifacts["delivery_archive"],
            ),
        )
    except Exception as exc:  # noqa: BLE001 - replace parser details with a host code
        raise ReleaseArtifactVerificationError(
            "release artifacts failed structural and source identity verification"
        ) from exc


def _release_binding(
    dist_dir: Path,
    *,
    source_root: Path = PROJECT_ROOT,
    source_files: Sequence[Path] | None = None,
) -> dict[str, Any]:
    root = source_root.resolve()
    declared_version = project_version(root)
    if declared_version != __version__:
        raise RuntimeError(
            "source package version does not match the runtime package version"
        )
    dist = dist_dir.expanduser().resolve()
    artifacts = {
        "wheel": _one_release_artifact(
            sorted(dist.glob(f"ecologyrsi_dsh-{__version__}-*.whl")), "wheel"
        ),
        "sdist": _one_release_artifact(
            sorted(dist.glob(f"ecologyrsi_dsh-{__version__}.tar.gz")), "sdist"
        ),
        "delivery_archive": _one_release_artifact(
            sorted(dist.glob(f"ecologyrsi-dsh-{__version__}-delivery.tar.gz")),
            "complete delivery archive",
        ),
    }
    _verify_release_artifacts(artifacts, dist, root)
    manifest = _source_manifest(root, source_files)
    script_entry = next(
        (
            item
            for item in manifest["files"]
            if item["path"] == "scripts/real_api_agent_tool_acceptance.py"
        ),
        None,
    )
    if script_entry is None:
        raise RuntimeError("release source manifest does not include the acceptance script")
    return {
        "schema_version": RELEASE_BINDING_SCHEMA_VERSION,
        "package_name": "ecologyrsi-dsh",
        "package_version": __version__,
        "python_version": ".".join(str(item) for item in sys.version_info[:3]),
        "source_manifest": manifest,
        "acceptance_script_sha256": script_entry["sha256"],
        "artifacts": {
            kind: {
                "filename": path.name,
                **_stable_file_record(path),
            }
            for kind, path in artifacts.items()
        },
    }


def _validate_report_output(
    output_path: Path,
    dist_dir: Path,
    *,
    source_root: Path = PROJECT_ROOT,
    extra_protected_paths: Sequence[Path] = (),
) -> Path:
    output = output_path.expanduser().resolve()
    root = source_root.resolve()
    dist = dist_dir.expanduser().resolve()
    protected = {
        (dist / "SHA256SUMS").resolve(),
        *(path.expanduser().resolve() for path in extra_protected_paths),
    }
    protected.update(
        path.resolve()
        for pattern in (
            f"ecologyrsi_dsh-{__version__}-*.whl",
            f"ecologyrsi_dsh-{__version__}.tar.gz",
            f"ecologyrsi-dsh-{__version__}-delivery.tar.gz",
        )
        for path in dist.glob(pattern)
    )
    artifact_name = (
        output.parent == dist
        and (
            (
                output.name.startswith(f"ecologyrsi_dsh-{__version__}-")
                and output.suffix == ".whl"
            )
            or output.name == f"ecologyrsi_dsh-{__version__}.tar.gz"
            or output.name
            == f"ecologyrsi-dsh-{__version__}-delivery.tar.gz"
        )
    )
    if output in protected or artifact_name:
        raise ReportOutputConflictError(
            "acceptance report output conflicts with a protected release input"
        )
    if output.is_relative_to(root) and not output.is_relative_to(root / "dist"):
        raise ReportOutputConflictError(
            "acceptance report must remain outside the delivery source manifest"
        )
    return output


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.samples_per_task <= 8:
        raise ValueError("samples-per-task must be between 1 and 8")
    if not 1 <= args.microbatch_size <= 128:
        raise ValueError("microbatch-size must be between 1 and 128")
    if not 1 <= args.gateway_attempts <= 8:
        raise ValueError("gateway-attempts must be between 1 and 8")
    if not 1 <= args.sample_attempts <= 8:
        raise ValueError("sample-attempts must be between 1 and 8")
    if not math.isfinite(args.minimum_coverage) or not 0 < args.minimum_coverage <= 1:
        raise ValueError("minimum-coverage must be in (0, 1]")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")


def _frozen_alias(
    metadata: Mapping[str, Any],
    names: Sequence[str],
    label: str,
) -> str | None:
    values: list[str] = []
    for name in names:
        raw = metadata.get(name)
        if raw is None:
            continue
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"RunCreated {label} must be a non-empty string")
        values.append(raw.strip())
    if not values:
        return None
    if len(set(values)) != 1:
        raise ValueError(f"RunCreated contains conflicting {label} aliases")
    return values[0]


def _model_alias(model_id: str) -> str:
    return model_id.strip().casefold().rsplit("/", 1)[-1]


def _is_glm_52(model_id: str) -> bool:
    return _model_alias(model_id) in GLM_52_MODEL_ALIASES


def _is_deepseek_flash(model_id: str) -> bool:
    return _model_alias(model_id) in DEEPSEEK_FLASH_MODEL_ALIASES


def _frozen_configuration_digest(
    metadata: Mapping[str, Any],
    names: Sequence[str],
    label: str,
) -> str:
    value = _frozen_alias(metadata, names, label)
    if value is None:
        raise ValueError(f"RunCreated is missing the frozen {label}")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"RunCreated {label} is not a lowercase SHA-256 digest")
    return value


def _binding_from_run_created_row(row: sqlite3.Row) -> dict[str, Any] | None:
    try:
        payload = json.loads(row["payload_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("RunCreated payload JSON is malformed") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("RunCreated payload must be an object")
    task_manifest = payload.get("task_manifest")
    if not isinstance(task_manifest, Mapping):
        return None
    metadata = task_manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        return None

    planner_model_id = _frozen_alias(
        metadata,
        ("strategy_model_id", "policy_model_id"),
        "policy/strategy model ID",
    )
    critic_model_id = _frozen_alias(
        metadata,
        ("review_model_id", "judge_model_id"),
        "review/judge model ID",
    )
    if (
        planner_model_id is None
        or critic_model_id is None
        or not _is_glm_52(planner_model_id)
        or not _is_deepseek_flash(critic_model_id)
    ):
        return None

    return {
        "run_id": str(row["run_id"]),
        "event_seq": int(row["seq"]),
        "created_at": str(row["created_at"]),
        "source": "RunCreated.task_manifest.metadata",
        "planner": {
            "model_id": planner_model_id,
            "configuration_digest": _frozen_configuration_digest(
                metadata,
                ("strategy_model_digest", "policy_model_digest"),
                "policy/strategy model configuration digest",
            ),
        },
        "critic": {
            "model_id": critic_model_id,
            "configuration_digest": _frozen_configuration_digest(
                metadata,
                ("review_model_digest", "judge_model_digest"),
                "review/judge model configuration digest",
            ),
        },
    }


def _assert_frozen_override(
    supplied: str | None,
    expected: str,
    option: str,
) -> None:
    if supplied is None:
        return
    if not isinstance(supplied, str) or not supplied.strip():
        raise ValueError(f"{option} must be a non-empty string")
    if supplied.strip() != expected:
        raise ValueError(f"{option} does not match the selected RunCreated binding")


def _read_reference_run_binding(
    db_path: str | Path,
    *,
    reference_run_id: str | None = None,
    planner_override: str | None = None,
    critic_override: str | None = None,
    planner_digest_override: str | None = None,
    critic_digest_override: str | None = None,
) -> dict[str, Any]:
    resolved = Path(db_path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError("evolution ledger does not exist")
    if reference_run_id is not None:
        if not isinstance(reference_run_id, str) or not reference_run_id.strip():
            raise ValueError("reference-run-id must be a non-empty string")
        reference_run_id = reference_run_id.strip()

    uri = f"{resolved.as_uri()}?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            if reference_run_id is None:
                rows = connection.execute(
                    """
                    SELECT seq, run_id, payload_json, created_at
                    FROM evolution_events
                    WHERE kind = 'RunCreated'
                    ORDER BY seq DESC
                    """
                )
            else:
                rows = connection.execute(
                    """
                    SELECT seq, run_id, payload_json, created_at
                    FROM evolution_events
                    WHERE kind = 'RunCreated' AND run_id = ?
                    ORDER BY seq DESC
                    LIMIT 1
                    """,
                    (reference_run_id,),
                )
            binding = next(
                (
                    candidate
                    for row in rows
                    if (candidate := _binding_from_run_created_row(row)) is not None
                ),
                None,
            )
    except sqlite3.Error as exc:
        raise ValueError("evolution ledger could not be read in read-only mode") from exc

    if binding is None:
        if reference_run_id is not None:
            raise ValueError(
                "reference-run-id does not identify a RunCreated binding for "
                "GLM-5.2 and DeepSeek Flash"
            )
        raise ValueError(
            "evolution ledger has no RunCreated binding for GLM-5.2 and DeepSeek Flash"
        )

    planner = binding["planner"]
    critic = binding["critic"]
    _assert_frozen_override(planner_override, planner["model_id"], "--planner")
    _assert_frozen_override(critic_override, critic["model_id"], "--critic")
    _assert_frozen_override(
        planner_digest_override,
        planner["configuration_digest"],
        "--planner-digest",
    )
    _assert_frozen_override(
        critic_digest_override,
        critic["configuration_digest"],
        "--critic-digest",
    )
    return binding


def _reference_binding_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return _read_reference_run_binding(
        args.db,
        reference_run_id=args.reference_run_id,
        planner_override=args.planner,
        critic_override=args.critic,
        planner_digest_override=args.planner_digest,
        critic_digest_override=args.critic_digest,
    )


def _quantile_indices(length: int, count: int) -> list[int]:
    if length < count:
        raise ValueError(
            f"prediction task has only {length} rows, fewer than requested {count}"
        )
    if count == 1:
        return [length // 2]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def stratified_feedback_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples_per_task: int,
) -> list[dict[str, Any]]:
    """Select fixed time quantiles without consulting labels or predictions."""

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        if raw.get("partition") != "training_feedback":
            continue
        if "predicted" in raw:
            raise ValueError("acceptance feedback rows must defer prediction")
        target = raw.get("target")
        horizon = raw.get("horizon_hours")
        if not isinstance(target, str) or not isinstance(horizon, int):
            raise ValueError("feedback row lacks a target-horizon identity")
        grouped[(target, horizon)].append(dict(raw))

    expected = {
        (target, horizon)
        for target, _unit, _minimum, _maximum in TARGETS
        for horizon in DEFAULT_HORIZONS
    }
    if set(grouped) != expected:
        missing = sorted(expected.difference(grouped))
        extra = sorted(set(grouped).difference(expected))
        raise ValueError(f"unexpected feedback tasks; missing={missing}, extra={extra}")

    selected: list[dict[str, Any]] = []
    for task in sorted(grouped):
        task_rows = sorted(
            grouped[task],
            key=lambda row: (
                row["origin_timestamp"],
                row.get("target_timestamp", row["timestamp"]),
            ),
        )
        selected.extend(
            task_rows[index]
            for index in _quantile_indices(len(task_rows), samples_per_task)
        )
    return selected


def _assert_label_free(value: Any, *, planner: bool) -> None:
    if isinstance(value, Mapping):
        normalized_keys = {
            str(key).casefold().replace("-", "_") for key in value
        }
        forbidden = set(FORBIDDEN_REMOTE_LABEL_KEYS)
        if planner:
            forbidden.update({"predicted", "prediction", "proposed_prediction"})
        leaked = sorted(normalized_keys.intersection(forbidden))
        if leaked:
            raise AssertionError("remote payload contains forbidden keys: " + ", ".join(leaked))
        for item in value.values():
            _assert_label_free(item, planner=planner)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_label_free(item, planner=planner)


class _AuditedRealGateway:
    """Record bounded call evidence while delegating every request to the real gateway."""

    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway
        self.calls: list[dict[str, Any]] = []

    def sample_decide(
        self,
        model_id: str,
        *,
        role: str,
        samples: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        available_tools: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        result, _receipt = self._invoke_sample_decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
            allow_format_retry=None,
        )
        return result

    def sample_decide_with_diagnostics(
        self,
        model_id: str,
        *,
        role: str,
        samples: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        available_tools: Sequence[Mapping[str, Any]],
        allow_format_retry: bool,
    ) -> tuple[Mapping[str, Any], Mapping[str, int]]:
        return self._invoke_sample_decide(
            model_id,
            role=role,
            samples=samples,
            context=context,
            available_tools=available_tools,
            allow_format_retry=allow_format_retry,
        )

    def _invoke_sample_decide(
        self,
        model_id: str,
        *,
        role: str,
        samples: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        available_tools: Sequence[Mapping[str, Any]],
        allow_format_retry: bool | None,
    ) -> tuple[Mapping[str, Any], Mapping[str, int]]:
        _assert_label_free(samples, planner=role == "planner")
        _assert_label_free(context, planner=role == "planner")
        call_started = time.monotonic()
        call = {
            "model_id": model_id,
            "role": role,
            "sample_count": len(samples),
            "sample_ids_digest": digest(
                [str(item.get("sample_id") or "") for item in samples]
            ),
            "available_tool_ids": [
                str(item.get("tool_id") or "") for item in available_tools
            ],
            "started_at": _utc_now(),
        }
        self.calls.append(call)
        try:
            if allow_format_retry is None:
                result = self.gateway.sample_decide(
                    model_id,
                    role=role,
                    samples=samples,
                    context=context,
                    available_tools=available_tools,
                )
                receipt: Mapping[str, int] = {"http_attempts": 1}
            else:
                result, receipt = self.gateway.sample_decide_with_diagnostics(
                    model_id,
                    role=role,
                    samples=samples,
                    context=context,
                    available_tools=available_tools,
                    allow_format_retry=allow_format_retry,
                )
        except Exception as exc:
            call["failed_at"] = _utc_now()
            call["duration_seconds"] = round(time.monotonic() - call_started, 3)
            call["error_type"] = type(exc).__name__
            if isinstance(exc, GatewayResponseError):
                call["error_code"] = exc.error_code
                call["retryable"] = exc.retryable
                call["split_eligible"] = exc.split_eligible
                call["attempts"] = exc.attempts
                call["status_code"] = exc.status_code
                call["finish_reason"] = exc.finish_reason
                call["usage"] = exc.usage
            raise
        call["completed_at"] = _utc_now()
        call["duration_seconds"] = round(time.monotonic() - call_started, 3)
        call["response_digest"] = digest(result)
        http_attempts = receipt.get("http_attempts")
        if isinstance(http_attempts, int) and not isinstance(http_attempts, bool):
            call["attempts"] = http_attempts
        return result, receipt


def _gateway_diagnostics(
    gateway: ModelGateway,
    model_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Return request diagnostics without prompts, responses, URLs, or credentials."""

    result: dict[str, dict[str, Any]] = {}
    for model_id in dict.fromkeys(model_ids):
        status = gateway.connection_status(model_id)
        item: dict[str, Any] = {
            "state": status.get("state"),
            "last_checked_at": status.get("last_checked_at"),
        }
        request = status.get("last_request")
        if isinstance(request, Mapping):
            public_request = {
                name: request[name]
                for name in (
                    "operation",
                    "started_at",
                    "updated_at",
                    "outcome",
                    "attempts",
                    "retry_count",
                    "next_retry_seconds",
                    "classification",
                    "timeout_seconds",
                    "response_metadata",
                )
                if name in request
            }
            retries = request.get("retries")
            if isinstance(retries, (list, tuple)):
                public_request["retries"] = [
                    {
                        name: retry[name]
                        for name in ("attempt", "delay_seconds")
                        if name in retry
                    }
                    for retry in retries
                    if isinstance(retry, Mapping)
                ]
            item["last_request"] = public_request
        else:
            item["last_request"] = None
        result[model_id] = item
    return result


def _gateway_from_args(args: argparse.Namespace) -> ModelGateway:
    env = dict(os.environ)
    env.setdefault("ECOLOGYRSI_DSH_DISCOVERY", "1")
    env["ECOLOGYRSI_DSH_MODEL_TIMEOUT"] = str(args.timeout_seconds)
    env["ECOLOGYRSI_DSH_MODEL_MAX_ATTEMPTS"] = str(args.gateway_attempts)
    return ModelGateway.from_env(env)


def _bound_model(
    catalog: Mapping[str, Mapping[str, Any]],
    model_id: str,
    expected_digest: str,
    role: str,
) -> dict[str, Any]:
    try:
        model = catalog[model_id]
    except KeyError as exc:
        raise ValueError(f"{role} model is not configured: {model_id}") from exc
    if model.get("configuration_digest") != expected_digest:
        raise ValueError(f"{role} model configuration digest does not match the frozen run")
    if model.get("credential_configured") is not True:
        raise ValueError(f"{role} model has no configured credential")
    return {
        "model_id": model_id,
        "configuration_digest": expected_digest,
        "binding_source": model.get("binding_source"),
        "credential_configured": True,
    }


def _model_agent_counts(records: Sequence[Mapping[str, Any]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        trace = record.get("agent_trace")
        if not isinstance(trace, (list, tuple)):
            continue
        for step in trace:
            if not isinstance(step, Mapping):
                continue
            model_id = step.get("model_id")
            role = step.get("role")
            if isinstance(model_id, str) and isinstance(role, str):
                counts[(model_id, role)] += 1
    return counts


def _selected_tool_counts(records: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        trace = record.get("attempt_trace")
        if not isinstance(trace, (list, tuple)):
            continue
        prior_tools: set[str] = set()
        for attempt in trace:
            if not isinstance(attempt, Mapping):
                continue
            selected = attempt.get("selected_tool")
            if not isinstance(selected, Mapping):
                continue
            tool_id = str(selected.get("tool_id") or "")
            if not tool_id:
                raise AssertionError("attempt trace is missing selected tool id")
            if tool_id in prior_tools:
                raise AssertionError("repair repeated a previously attempted tool")
            prior_tools.add(tool_id)
            counts[tool_id] += 1
            if selected.get("status") == "completed":
                output_digest = selected.get("output_digest")
                if not isinstance(output_digest, str) or len(output_digest) != 64:
                    raise AssertionError("completed selected tool lacks output digest")
    return counts


def run_acceptance(
    args: argparse.Namespace,
    *,
    reference_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_args(args)
    if reference_binding is None:
        reference_binding = _reference_binding_from_args(args)
    planner_reference = reference_binding["planner"]
    critic_reference = reference_binding["critic"]
    planner_model_id = str(planner_reference["model_id"])
    critic_model_id = str(critic_reference["model_id"])
    started_at = _utc_now()
    started = time.monotonic()
    gateway = _gateway_from_args(args)
    catalog = {item["model_id"]: item for item in gateway.catalog()}
    planner_binding = _bound_model(
        catalog,
        planner_model_id,
        str(planner_reference["configuration_digest"]),
        "planner",
    )
    critic_binding = _bound_model(
        catalog,
        critic_model_id,
        str(critic_reference["configuration_digest"]),
        "critic",
    )

    datasets = DatasetRegistry()
    series = datasets.series(args.dataset, args.episode)
    config = ExogenousRidgeConfig(
        history_steps=6,
        ridge_alpha=0.1,
        residual_scale=0.7,
    )
    fitted = fit_predict_exogenous_ridge(
        series,
        targets=tuple(item[0] for item in TARGETS),
        horizons=DEFAULT_HORIZONS,
        config=config,
        evaluation_history_steps=MAX_EXOGENOUS_RIDGE_HISTORY_STEPS,
        defer_prediction_partitions=("training_feedback",),
    )
    full_feedback = [
        row
        for row in fitted["prediction_rows"]
        if row["partition"] == "training_feedback"
    ]
    selected_rows = stratified_feedback_rows(
        full_feedback,
        samples_per_task=args.samples_per_task,
    )
    models = fitted["models"]
    handler_calls: Counter[str] = Counter()
    handler_outputs: dict[str, dict[str, float]] = defaultdict(dict)
    algorithm_name, separator, algorithm_version = EXOGENOUS_RIDGE_MODEL_ID.rpartition(
        "@"
    )
    if not separator:
        algorithm_name = EXOGENOUS_RIDGE_MODEL_ID
        algorithm_version = "unversioned"

    def ridge_value(request: SamplePredictionRequest) -> float:
        return predict_fitted_exogenous_ridge(
            target=request.target,
            horizon_hours=request.horizon_hours,
            baseline=request.baseline,
            label_free_context=request.label_free_context,
            models=models,
            config=config,
        )

    def ridge_tool(request: SamplePredictionRequest) -> Mapping[str, Any]:
        handler_calls[algorithm_name] += 1
        predicted = ridge_value(request)
        handler_outputs[algorithm_name][request.sample_id] = predicted
        return {
            "predicted": predicted,
            "metadata": {
                "model_id": EXOGENOUS_RIDGE_MODEL_ID,
                "execution_mode": "agent_selected_host_tool_on_demand",
            },
        }

    def conservative_ridge_tool(
        request: SamplePredictionRequest,
        execution_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del execution_context
        tool_id = "greenhouse-ridge-conservative@1"
        handler_calls[tool_id] += 1
        raw = ridge_value(request)
        predicted = request.baseline + 0.5 * (raw - request.baseline)
        handler_outputs[tool_id][request.sample_id] = predicted
        return {
            "predicted": predicted,
            "metadata": {
                "source_model_id": EXOGENOUS_RIDGE_MODEL_ID,
                "residual_multiplier": 0.5,
                "execution_mode": "agent_selected_host_tool_on_demand",
            },
        }

    audited_gateway = _AuditedRealGateway(gateway)
    adapter = GatewaySampleCollaborationAdapter(
        audited_gateway,
        strategy_model_id=planner_model_id,
        review_model_id=critic_model_id,
        remote_review_enabled=True,
        forecast_tool=ridge_tool,
        tools=(
            GatewaySampleTool(
                tool_id="greenhouse-ridge-conservative@1",
                version="1",
                handler=conservative_ridge_tool,
                purpose="conservative_half_residual_ridge_prediction_on_demand",
            ),
        ),
        microbatch_size=args.microbatch_size,
        sample_planner_prompt_profile={
            "version": ORIGIN_SHARED_CONTEXT_PROFILE,
        },
    )
    batch = CollaborativeSampleExecutor(adapter).execute(
        selected_rows,
        context={
            "run_id": "acceptance:real-api-agent-tools",
            "candidate_id": "acceptance:ridge-tool-candidate",
            "dataset_digest": series.digest,
            "split_manifest_digest_sha256": series.split_manifest_digest_sha256,
            "partition": "training_feedback",
            "algorithm_id": algorithm_name,
            "algorithm_version": algorithm_version,
            "evaluator_id": "real-api-agent-tool-acceptance",
            "horizons_hours": list(DEFAULT_HORIZONS),
            "candidate_parameters": config.to_dict(),
            "strategy_model_id": planner_model_id,
            "review_model_id": critic_model_id,
            "sample_agent_mode": "gateway_microbatch",
            "sample_agent_batch_size": args.microbatch_size,
            "algorithm_artifact_digest": digest(
                {
                    "model_id": EXOGENOUS_RIDGE_MODEL_ID,
                    "parameters": config.to_dict(),
                    "models": models,
                    "dataset_digest": series.digest,
                }
            ),
        },
        target_bounds={
            name: {"unit": unit, "minimum": minimum, "maximum": maximum}
            for name, unit, minimum, maximum in TARGETS
        },
        algorithm_id=algorithm_name,
        algorithm_version=algorithm_version,
        policy=SampleExecutionPolicy(
            max_attempts=args.sample_attempts,
            plan_max_attempts=args.sample_attempts,
            minimum_coverage=args.minimum_coverage,
            minimum_task_coverage=args.minimum_coverage,
            retry_backoff_seconds=1.0,
        ),
    )

    model_agent_counts = _model_agent_counts(batch.records)
    selected_tool_counts = _selected_tool_counts(batch.records)
    planner_steps = sum(
        count
        for (model_id, role), count in model_agent_counts.items()
        if model_id == planner_model_id and role == "remote_planner_agent"
    )
    critic_steps = sum(
        count
        for (model_id, role), count in model_agent_counts.items()
        if model_id == critic_model_id and role == "remote_critic_agent"
    )
    failed_checks: list[dict[str, str]] = []
    if planner_steps < 1:
        failed_checks.append(
            {
                "code": "planner_evidence_missing",
                "message": "no completed planner evidence from the selected real API",
            }
        )
    if critic_steps < 1:
        failed_checks.append(
            {
                "code": "critic_evidence_missing",
                "message": "no completed critic evidence from the selected real API",
            }
        )
    if not batch.summary["tool_performance"]:
        failed_checks.append(
            {
                "code": "tool_performance_missing",
                "message": "tool performance was not aggregated",
            }
        )
    if batch.summary["coverage_pass"] is not True:
        failed_checks.append(
            {
                "code": "coverage_below_threshold",
                "message": "real planner/tool/critic coverage is below the threshold",
            }
        )
    incomplete_tasks = [
        item
        for item in batch.summary["tasks"]
        if float(item["coverage"]) < args.minimum_coverage
    ]
    if incomplete_tasks:
        failed_checks.append(
            {
                "code": "task_coverage_below_threshold",
                "message": (
                    "one or more target-horizon tasks are below the coverage threshold"
                ),
            }
        )

    scoring_by_id = {
        str(row["sample_id"]): row for row in batch.scoring_rows if "sample_id" in row
    }
    for tool_id, outputs in handler_outputs.items():
        for sample_id, predicted in outputs.items():
            row = scoring_by_id.get(sample_id)
            if row is None or row.get("sample_execution_status") != "succeeded":
                continue
            attempts = next(
                record.get("attempt_trace", [])
                for record in batch.records
                if record.get("sample_id") == sample_id
            )
            final_tool = attempts[-1].get("selected_tool", {}).get("tool_id")
            if final_tool == tool_id and float(row["predicted"]) != predicted:
                failed_checks.append(
                    {
                        "code": "selected_tool_output_mismatch",
                        "message": "final prediction differs from selected tool output",
                    }
                )
                break

    task_counts = Counter(
        (str(row["target"]), int(row["horizon_hours"])) for row in selected_rows
    )
    return {
        "schema_version": "ecologyrsi-dsh.real-api-agent-tool-acceptance/1",
        "profile_id": PROFILE_ID,
        "binding_reference_run_id": reference_binding["run_id"],
        "binding_reference": {
            "source": reference_binding["source"],
            "event_seq": reference_binding["event_seq"],
            "created_at": reference_binding["created_at"],
        },
        "passed": not failed_checks,
        "scientific_score_authoritative": False,
        "promotion_eligible": False,
        "training_asset_eligible": False,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "dataset": {
            "dataset_id": series.dataset_id,
            "episode_id": series.episode_id,
            "dataset_digest": series.digest,
            "split_manifest_digest_sha256": series.split_manifest_digest_sha256,
        },
        "cohort": {
            "full_training_feedback_examples": len(full_feedback),
            "acceptance_examples": len(selected_rows),
            "samples_per_prediction_task": args.samples_per_task,
            "selection": "fixed_target_horizon_time_quantiles_without_labels@1",
            "selection_digest": digest(
                [
                    {
                        "target": row["target"],
                        "horizon_hours": row["horizon_hours"],
                        "origin_timestamp": row["origin_timestamp"],
                        "target_timestamp": row.get(
                            "target_timestamp", row["timestamp"]
                        ),
                    }
                    for row in selected_rows
                ]
            ),
            "tasks": [
                {
                    "target": target,
                    "horizon_hours": horizon,
                    "examples": count,
                }
                for (target, horizon), count in sorted(task_counts.items())
            ],
        },
        "fit": {
            "model_id": fitted["model_id"],
            "parameters": config.to_dict(),
            "fitted_task_count": len(models),
            "training_partition": fitted["training_partition"],
            "evaluation_partition": fitted["evaluation_partition"],
        },
        "models": {
            "planner": {
                **planner_binding,
                "completed_agent_steps": planner_steps,
            },
            "critic": {
                **critic_binding,
                "completed_agent_steps": critic_steps,
            },
        },
        "gateway_calls": audited_gateway.calls,
        "gateway_diagnostics": _gateway_diagnostics(
            gateway,
            (planner_model_id, critic_model_id),
        ),
        "failed_checks": failed_checks,
        "execution": {
            "sample_planner_prompt_profile": {
                "version": ORIGIN_SHARED_CONTEXT_PROFILE,
            },
            "attempted_examples": batch.summary["attempted_examples"],
            "succeeded_examples": batch.summary["succeeded_examples"],
            "failed_examples": batch.summary["failed_examples"],
            "coverage": batch.summary["coverage"],
            "minimum_coverage": args.minimum_coverage,
            "coverage_pass": batch.summary["coverage_pass"],
            "task_coverage_pass": not incomplete_tasks,
            "retry_count": batch.summary["retry_count"],
            "recovered_examples": batch.summary["recovered_examples"],
            "failure_counts": batch.summary["failure_counts"],
            "selected_tool_counts": dict(sorted(selected_tool_counts.items())),
            "host_tool_handler_calls": dict(sorted(handler_calls.items())),
            "tool_performance": batch.summary["tool_performance"],
            "trace_digest": batch.summary["trace_digest"],
            "attempt_trace_record_count": sum(
                bool(record.get("attempt_trace")) for record in batch.records
            ),
        },
    }


def _exception_report(
    args: argparse.Namespace,
    exc: BaseException,
    *,
    reference_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    def public_value(value: Any, *, depth: int = 0) -> Any:
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, str):
            stripped = value.strip()
            if (
                "://" in stripped
                or stripped.startswith(("/", "\\\\"))
                or (
                    len(stripped) >= 3
                    and stripped[0].isalpha()
                    and stripped[1] == ":"
                    and stripped[2] in {"/", "\\"}
                )
            ):
                return REDACTED
            return public_error_summary(value, limit=300)
        if depth >= 5:
            return None
        if isinstance(value, Mapping):
            return {
                key: public_value(item, depth=depth + 1)
                for key, item in list(value.items())[:32]
                if isinstance(key, str) and not is_sensitive_key(key)
            }
        if isinstance(value, (list, tuple)):
            return [public_value(item, depth=depth + 1) for item in value[:32]]
        return None

    error_code = "acceptance_exception"
    if isinstance(
        exc,
        (
            GatewayResponseError,
            ReleaseArtifactVerificationError,
            ReleaseBindingChangedError,
            ReportOutputConflictError,
        ),
    ):
        error_code = (
            safe_error_code(getattr(exc, "error_code", None), error_code)
            or error_code
        )
    failure: dict[str, Any] = {
        "stage": "acceptance_setup_or_execution",
        "error_type": safe_error_code(type(exc).__name__, "Exception") or "Exception",
        "error_code": error_code,
    }
    for name in ("retryable", "split_eligible", "attempts", "status_code"):
        value = getattr(exc, name, None)
        if value is not None:
            failure[name] = public_value(value)
    reference_run_id = (
        reference_binding["run_id"]
        if reference_binding is not None
        else args.reference_run_id
    )
    planner_model_id = (
        reference_binding["planner"]["model_id"]
        if reference_binding is not None
        else args.planner
    )
    critic_model_id = (
        reference_binding["critic"]["model_id"]
        if reference_binding is not None
        else args.critic
    )
    return {
        "schema_version": "ecologyrsi-dsh.real-api-agent-tool-acceptance/1",
        "profile_id": PROFILE_ID,
        "binding_reference_run_id": public_value(reference_run_id),
        "passed": False,
        "scientific_score_authoritative": False,
        "promotion_eligible": False,
        "training_asset_eligible": False,
        "completed_at": _utc_now(),
        "request": {
            "dataset_id": public_value(args.dataset),
            "episode_id": public_value(args.episode),
            "planner_model_id": public_value(planner_model_id),
            "critic_model_id": public_value(critic_model_id),
            "samples_per_prediction_task": args.samples_per_task,
            "minimum_coverage": args.minimum_coverage,
        },
        "failure": failure,
    }


def _write_report(path: Path, encoded: str) -> Path:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(encoded + "\n", encoding="utf-8")
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    release_binding: dict[str, Any] | None = None
    reference_binding: dict[str, Any] | None = None
    output: Path | None = None
    try:
        output = _validate_report_output(
            args.output,
            args.dist_dir,
            extra_protected_paths=(args.db,),
        )
        reference_binding = _reference_binding_from_args(args)
        initial_release_binding = _release_binding(args.dist_dir)
        try:
            report = run_acceptance(args, reference_binding=reference_binding)
        finally:
            final_release_binding = _release_binding(args.dist_dir)
            if final_release_binding != initial_release_binding:
                raise ReleaseBindingChangedError(
                    "release inputs changed during real API acceptance"
                )
            release_binding = final_release_binding
    except Exception as exc:  # noqa: BLE001 - persist a bounded failure artifact
        report = _exception_report(
            args,
            exc,
            reference_binding=reference_binding,
        )
    if release_binding is not None:
        report["release_binding"] = release_binding
    report["report_path"] = str(output) if output is not None else None
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if output is not None:
        _write_report(output, encoded)
    print(encoded)
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
