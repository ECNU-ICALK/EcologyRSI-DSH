"""Concrete HTTP server and request handler composition."""

from __future__ import annotations

from collections import OrderedDict
import fcntl
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from ..application.config import bind_toy_dataset
from ..core.director import EvolutionDirector
from ..core.errors import (
    DshNativeRuntimeUnavailableError,
    FrozenRuntimeBindingDriftError,
)
from ..core.ledger import (
    CommandInProgressError,
    CommandReceipt,
    ConcurrentRunMutationError,
    EventLedger,
)
from ..core.models import (
    ExpertConsultationAnswer,
    HumanIntervention,
    InterventionKind,
    Run,
    TaskManifest,
    digest,
    utc_now,
)
from ..core.sample_results import MAX_SAMPLE_RESULTS_UNCOMPRESSED_BYTES
from ..data.registry import DatasetRegistry
from ..evaluators.registry import TOY_DATASET_ID, EvaluatorRegistry
from ..evolution.strategies import StrategyRouterDSHAdapter
from ..integrations.model_bindings import (
    HOST_PARAMETER_GENERATOR_ID,
    RULE_JUDGE_ID,
    builtin_model_configuration_digest,
)
from ..integrations.model_gateway import ModelGateway
from ..integrations.dsh_native_runtime import (
    DSH_NATIVE_EXECUTION_PROTOCOL,
    DshNativeAgentRuntimeClient,
)
from ..integrations.dsh_structured_roles import DshStructuredRoleRuntime
from ..version import __version__
from .auto_progress import AutoProgressManager
from .dsh_tools import DshToolService
from .projection import _state_payload
from .shared import (
    AUTO_ADVANCE_CONTINUOUS,
    _assert_http_scope,
    _assert_manifest_http_scope,
    _auto_advance_steps,
    _derived_seed,
    _is_loopback_host,
    _public_http_error,
    _request_integer,
)

_DEFAULT_REAL_RUN_TOKEN_LIMIT = 100_000_000
# Historical payloads call this a "wave" reservation. In the strict v2
# scheduler it is the frozen cap for each logical gateway call, including all
# request-local HTTP attempts, not one top-level recursive schedule.
_REAL_RUN_TOKEN_RESERVATION_PER_CALL = 262_144
_SAMPLE_TOKEN_BUDGET_POLICY = "hard_gateway_call_reservation@1"
_SAMPLE_TOKEN_BUDGET_SCOPE = "sample_agent_gateway_calls_only@1"
_DEFAULT_REAL_SAMPLE_CONCURRENCY = 2
_MAX_REAL_SAMPLE_CONCURRENCY = 8
_DEFAULT_SAMPLES_PER_UPDATE = 500
_MAX_SAMPLES_PER_UPDATE = 100_000
_DEFAULT_SAMPLE_AGENT_BATCH_SIZE = 64
_MAX_SAMPLE_AGENT_BATCH_SIZE = 128
_DEFAULT_SAMPLE_OPERATION_MAX_TOKENS = {
    # Production evidence showed that reasoning-capable planners frequently
    # exhausted 3072 tokens before emitting the bounded decision object.  A
    # larger first attempt is cheaper than replaying the same prompt at 8192.
    "sample.planner": 6144,
    "sample.repair": 3072,
    "sample.critic": 2048,
}
_DEFAULT_SAMPLE_TRUNCATION_RETRY_POLICY = {
    "version": "escalate_once@1",
    "max_tokens": 8192,
}
_DEFAULT_SAMPLE_PLANNER_PROMPT_PROFILE = {
    "version": "origin_shared_context@1",
}
_DEFAULT_SAMPLE_REMOTE_CRITIC_POLICY = {
    "version": "always@1",
}
_DSH_NATIVE_PRESET_IDS = (
    "ecology-coordinator-v1",
    "ecology-researcher-v1",
    "ecology-candidate-proposer-v1",
    "ecology-sample-planner-v1",
    "ecology-sample-critic-v1",
    "ecology-generation-judge-v1",
)
_DSH_NATIVE_STABLE_PRESET_FIELDS = (
    "preset_id",
    "declared",
    "preset_mountable",
    "tool_surface_verified",
    "route_resolvable",
)


def _dsh_native_stable_preset_catalog(
    capabilities: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the immutable capability subset used in frozen run identities."""

    by_id = {
        item.get("preset_id"): item
        for item in capabilities.get("presets", [])
        if isinstance(item, dict)
    }
    return [
        {field: by_id[preset_id].get(field) for field in _DSH_NATIVE_STABLE_PRESET_FIELDS}
        for preset_id in _DSH_NATIVE_PRESET_IDS
        if preset_id in by_id
    ]

PLUGIN_MANIFEST = {
    "id": "ecology_evolution",
    "plugin_id": "ecologyrsi.evolution",
    "display_name": "生态模型进化工作台",
    "description": "用于创建、控制和查看生态模型进化运行的轻量 DSH 工作台。",
    "version": __version__,
    "mode": "evolution",
    "workflow": "research_compile_evolve@1",
    "request_schema": {
        "required": ["dataset_id", "strategy_model_id", "review_model_id"],
        "optional": [
            "domain_pack_id",
            "domain",
            "dataset",
            "episode_id",
            "rounds",
            "candidates_per_generation",
            "max_candidates",
            "model_workflow",
            "research_domain",
            "knowledge_online_enabled",
            "auto_progress",
            "allow_host_fallback",
            "samples_per_update",
            "sample_concurrency",
            "sample_agent_batch_size",
        ],
        "internal_components": "由策略模型在运行中调研并由宿主注册表编译；旧字段仅作兼容输入",
    },
    "api_prefix": "/api",
    "recommended_dsh_proxy_base": "/api/ecology-evolution",
    "supported_bases": [
        "/api",
        "/api/v1",
        "/api/ecology-evolution",
        "/api/ecology-evolution/v1",
    ],
    "capabilities": [
        "start_run",
        "read_projection",
        "control_run",
        "advance_run",
        "read_events",
        "read_sample_results",
        "read_training_data",
        "write_intervention",
        "select_configured_model",
        "run.archive",
        "run.delete",
    ],
    "advance": "/runs/{run_id}/advance",
    "samples": "/runs/{run_id}/samples?candidate_id={candidate_id}&offset={offset}&limit={limit}",
    "expert_consultation_answer": (
        "/runs/{run_id}/expert-consultations/{consultation_id}/answer"
    ),
}


def _request_boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    # FormData and a few older DSH bridges serialize checkbox values as
    # strings.  Normalize only the unambiguous spellings and keep all other
    # values fail-closed.
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on", "是"}:
            return True
        if normalized in {"false", "0", "no", "off", "否"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _mark_auto_progress_task(
    task: TaskManifest, *, allow_host_fallback: bool = False
) -> TaskManifest:
    """Persist the continuous execution policy in the immutable manifest."""

    data = task.to_dict()
    metadata = dict(data.get("metadata", {}))
    metadata["auto_progress"] = True
    metadata["auto_progress_policy"] = "continuous_generation_budget@1"
    metadata["allow_host_fallback"] = bool(allow_host_fallback)
    metadata["remote_fallback_policy"] = (
        "record_and_continue" if allow_host_fallback else "fail_run"
    )
    data["metadata"] = metadata
    return TaskManifest.from_dict(data)


class _SidecarOwnerLease:
    """Process-scoped advisory lock for one persistent event ledger."""

    def __init__(self, db_path: str | Path) -> None:
        self._fd: int | None = None
        self.lock_path: Path | None = None
        if str(db_path) == ":memory:":
            return
        resolved_db = Path(db_path).expanduser().resolve()
        resolved_db.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = resolved_db.with_name(resolved_db.name + ".sidecar.lock")
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise RuntimeError(
                "another EcologyRSI-DSH sidecar already owns this database: "
                + str(resolved_db)
            ) from exc
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        owner = f"pid={os.getpid()}\ndatabase={resolved_db}\n".encode("utf-8")
        os.ftruncate(fd, 0)
        os.write(fd, owner)
        os.fsync(fd)

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class _GenerationLease:
    """Reference-counted per-run lock that can be retired without replacement races."""

    def __init__(self, server: "EvolutionHTTPServer", run_id: str) -> None:
        self._server = server
        self.run_id = run_id
        self._lock = threading.RLock()
        self._reservations = 0
        self._retiring = False
        self._retired = False

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        server = self._server
        with server._generation_locks_guard:
            if (
                self._retired
                or self._retiring
                or server._generation_locks.get(self.run_id) is not self
            ):
                return False
            # Count both owners and waiters. A purge may retire this entry only
            # when its own reservation is the sole remaining reference.
            self._reservations += 1
        try:
            acquired = (
                self._lock.acquire(blocking)
                if timeout == -1
                else self._lock.acquire(blocking, timeout)
            )
        except BaseException:
            with server._generation_locks_guard:
                self._reservations -= 1
            raise
        if acquired:
            return True
        with server._generation_locks_guard:
            self._reservations -= 1
        return False

    def release(self) -> None:
        self._lock.release()
        with self._server._generation_locks_guard:
            self._reservations -= 1

    def __enter__(self) -> "_GenerationLease":
        if not self.acquire():
            raise CommandInProgressError("run generation lease is being retired")
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.release()


class EvolutionHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def handle_error(self, request: Any, client_address: tuple[Any, ...]) -> None:
        error = sys.exc_info()[1]
        if isinstance(
            error,
            (ConnectionResetError, BrokenPipeError, ConnectionAbortedError),
        ):
            return
        super().handle_error(request, client_address)

    def __init__(self, address: tuple[str, int], db_path: str | Path) -> None:
        host = address[0]
        configured_token = os.environ.get("ECOLOGYRSI_SERVICE_TOKEN", "").strip()
        if not _is_loopback_host(host) and not configured_token:
            raise ValueError(
                "非本机监听必须设置 ECOLOGYRSI_SERVICE_TOKEN 作为 DSH 能力令牌"
            )
        self.capability_token = configured_token or None
        configured_tool_token = os.environ.get(
            "ECOLOGYRSI_SIDECAR_TOOL_TOKEN", ""
        ).strip()
        self.dsh_tool_token = configured_tool_token or None
        self._owner_lease = _SidecarOwnerLease(db_path)
        try:
            self.ledger = EventLedger(db_path)
            self.dsh_tools = DshToolService(self.ledger)
            self.datasets = DatasetRegistry()
            self.model_gateway = ModelGateway.from_env(verification_store=self.ledger)
            runtime_origin = os.environ.get("ECOLOGYRSI_DSH_RUNTIME_URL", "").strip()
            runtime_token = os.environ.get("ECOLOGYRSI_DSH_RUNTIME_TOKEN", "").strip()
            self.dsh_native_runtime = (
                DshNativeAgentRuntimeClient(runtime_origin, token=runtime_token)
                if runtime_origin and runtime_token
                else None
            )
            self.strategy_router = StrategyRouterDSHAdapter(
                self.model_gateway,
                native_runtime_provider=lambda: DshStructuredRoleRuntime(
                    self.dsh_native_runtime,
                    admission=self.dsh_tools,
                ),
            )
            self.evaluators = EvaluatorRegistry(
                self.datasets,
                self.model_gateway,
                dsh_runtime_provider=lambda: DshStructuredRoleRuntime(
                    self.dsh_native_runtime,
                    admission=self.dsh_tools,
                ),
                dsh_revision_provider=lambda run_id: {
                    "run_state_revision": self.director.state(run_id).events[-1].seq,
                    "ledger_expected_revision": self.ledger.latest_seq(),
                },
                dsh_identity_provider=lambda run_id, candidate_id: (
                    self.director.state(run_id).candidate_identity_binding(candidate_id)
                ),
            )
            self.director = EvolutionDirector(self.ledger, self.strategy_router)
            # A mutation spans several append-only events.  Serial execution keeps
            # that unit coherent without introducing a queue or transaction layer.
            self.mutation_lock = threading.RLock()
            self.sample_result_cache_lock = threading.RLock()
            self.sample_result_cache: OrderedDict[
                tuple[str, str], tuple[int, tuple[dict[str, Any], ...]]
            ] = OrderedDict()
            self.sample_result_cache_bytes = 0
            self.sample_result_cache_max_bytes = min(
                MAX_SAMPLE_RESULTS_UNCOMPRESSED_BYTES,
                32 * 1024 * 1024,
            )
            self._generation_locks_guard = threading.Lock()
            self._generation_locks: dict[str, _GenerationLease] = {}
            super().__init__(address, EvolutionRequestHandler)
            # A process can stop after RunCreated was durably bound to its
            # create receipt but before the HTTP response was sealed.  At this
            # point the sidecar owner lease guarantees there is no prior HTTP
            # process still completing that command, so close only receipts
            # whose exact run can be replayed into a public projection.
            self._recover_bound_create_receipts()
            # Continuous autonomous runs are progressed by a bounded worker pool.
            # Recovery is projection-driven and only picks up manifests that
            # explicitly opted into this mode.
            self.auto_progress = AutoProgressManager(self)
            self.auto_progress.recover_running()
        except BaseException:
            if hasattr(self, "socket"):
                self.server_close()
            if hasattr(self, "ledger"):
                self.ledger.close()
            self._owner_lease.release()
            raise

    def _recover_bound_create_receipts(self) -> int:
        """Seal bound create receipts only after their requested work finished."""

        recovered = 0
        for command_key in self.ledger.pending_command_keys():
            receipt = self.ledger.command_receipt(command_key)
            if (
                receipt is None
                or receipt.status != "pending"
                or receipt.command_kind != "create_run"
                or receipt.resource_run_id is None
            ):
                continue
            try:
                state = self.director.state(receipt.resource_run_id)
                if not self._create_receipt_is_complete(receipt, state):
                    continue
                payload = _state_payload(state)
            except (KeyError, TypeError, ValueError, RuntimeError):
                # Unbound, missing, or malformed recovery evidence remains
                # pending for an explicit same-key retry or operator review.
                continue
            self.ledger.complete_command(command_key, payload)
            recovered += 1
        return recovered

    def _create_receipt_is_complete(self, receipt: CommandReceipt, state: Any) -> bool:
        """Check the durable projection against the original create request."""

        start = receipt.request.get("start", True)
        if not isinstance(start, bool):
            return False
        if not start:
            return True

        status = state.run.status.value
        if status in {"completed", "failed", "cancelled", "paused"}:
            return True
        if status == "created":
            return False
        if status != "running":
            return False
        if state.task_manifest.metadata.get("auto_progress") is True:
            return True

        try:
            target_steps = _auto_advance_steps(receipt.request.get("auto_advance", 1))
        except (TypeError, ValueError):
            return False
        if target_steps == AUTO_ADVANCE_CONTINUOUS or target_steps <= 0:
            return True
        completed_steps = sum(
            event.kind == "GenerationAdvanced"
            for event in self.ledger.events(
                state.run.run_id,
                after_seq=receipt.start_seq,
            )
        )
        return completed_steps >= target_steps

    def validate_frozen_runtime_bindings(
        self,
        task: TaskManifest,
        *,
        run_id: str | None = None,
    ) -> None:
        """Apply the same immutable-runtime checks to HTTP and worker execution."""

        EvolutionRequestHandler._validate_frozen_runtime_bindings_for_server(
            self, task, run_id=run_id
        )

    def generation_lock(self, run_id: str) -> _GenerationLease:
        """Return the process-local execution lock for one durable run."""

        key = str(run_id).strip()
        if not key:
            raise ValueError("run_id must be non-empty")
        with self._generation_locks_guard:
            lease = self._generation_locks.get(key)
            if lease is None or lease._retired:
                lease = _GenerationLease(self, key)
                self._generation_locks[key] = lease
            return lease

    def acquire_generation_lease(
        self,
        run_id: str,
        *,
        blocking: bool = True,
    ) -> _GenerationLease | None:
        """Atomically resolve, reserve, and acquire one run's generation lease."""

        key = str(run_id).strip()
        if not key:
            raise ValueError("run_id must be non-empty")
        with self._generation_locks_guard:
            lease = self._generation_locks.get(key)
            if lease is None or lease._retired:
                lease = _GenerationLease(self, key)
                self._generation_locks[key] = lease
            if lease._retiring:
                return None
            lease._reservations += 1
        try:
            acquired = lease._lock.acquire(blocking)
        except BaseException:
            with self._generation_locks_guard:
                lease._reservations -= 1
            raise
        if acquired:
            return lease
        with self._generation_locks_guard:
            lease._reservations -= 1
        return None

    def retire_generation_lease_if_idle(self, lease: _GenerationLease) -> bool:
        """Drop an idle lock created only to reject stale incarnation work."""

        with self._generation_locks_guard:
            if (
                self._generation_locks.get(lease.run_id) is not lease
                or lease._retired
                or lease._retiring
                or lease._reservations != 0
                or not lease._lock.acquire(blocking=False)
            ):
                return False
            self._generation_locks.pop(lease.run_id)
            lease._retired = True
            lease._lock.release()
            return True

    def try_acquire_generation_purge_lease(
        self, run_id: str
    ) -> _GenerationLease | None:
        """Exclusively fence one run before deleting its durable incarnation."""

        lease = self.generation_lock(run_id)
        with self._generation_locks_guard:
            if (
                lease._retired
                or lease._retiring
                or self._generation_locks.get(lease.run_id) is not lease
            ):
                return None
            lease._retiring = True
            lease._reservations += 1
        if not lease._lock.acquire(blocking=False):
            with self._generation_locks_guard:
                lease._reservations -= 1
                lease._retiring = False
            return None
        with self._generation_locks_guard:
            if lease._reservations == 1:
                return lease
            # A normal owner or waiter reserved this exact lock before the purge
            # fence was raised. Keep the mapping stable and make the caller retry.
            lease._lock.release()
            lease._reservations -= 1
            lease._retiring = False
        return None

    def release_generation_purge_lease(
        self, lease: _GenerationLease, *, purged: bool
    ) -> None:
        """Release a purge fence, atomically retiring its lock after success."""

        with self._generation_locks_guard:
            if (
                self._generation_locks.get(lease.run_id) is not lease
                or not lease._retiring
                or lease._reservations != 1
            ):
                raise RuntimeError("generation purge lease invariant was violated")
            if purged:
                # Remove and unlock while holding the registry guard. A future
                # incarnation cannot obtain a replacement until the old lock is
                # retired and no owner or waiter can still reference it.
                self._generation_locks.pop(lease.run_id)
                lease._retired = True
            lease._lock.release()
            lease._reservations = 0
            lease._retiring = False

    def close(self) -> None:
        worker_stopped = self.auto_progress.close()
        # A urllib request cannot be interrupted safely. If shutdown reaches
        # its bounded join timeout, leave the ledger owned by that daemon
        # worker instead of closing SQLite underneath an in-flight generation.
        # A later close call will reclaim it after the worker exits.
        if worker_stopped:
            with self.mutation_lock:
                self.ledger.close()
            self._owner_lease.release()
        self.server_close()


from .catalog import CatalogEndpointsMixin  # noqa: E402
from .events import EventEndpointsMixin  # noqa: E402
from .execution import ExecutionEndpointsMixin  # noqa: E402
from .transport import TransportMixin  # noqa: E402


class EvolutionRequestHandler(
    CatalogEndpointsMixin,
    EventEndpointsMixin,
    ExecutionEndpointsMixin,
    TransportMixin,
    BaseHTTPRequestHandler,
):
    server: EvolutionHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Keep the demo quiet unless a caller overrides the handler logger.
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed_path = urlparse(self.path)
        raw_path = parsed_path.path
        if raw_path == "/plugins/ecology/evolution":
            location = raw_path + "/"
            if parsed_path.query:
                location += "?" + parsed_path.query
            self.send_response(HTTPStatus.MOVED_PERMANENTLY)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if raw_path.startswith("/plugins/ecology/evolution"):
            self._serve_plugin(raw_path)
            return
        if not self._authorize_api():
            return
        path = self._route()
        if path == ["health"]:
            self._send(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "package": "ecologyrsi-dsh",
                    "package_version": __version__,
                    "api_version": "v0.2",
                    "plugin_version": PLUGIN_MANIFEST["version"],
                    "schema_version": self.server.ledger.schema_version,
                    # Kept for the v0.1 health contract; the explicit list is
                    # authoritative for the expanded runtime.
                    "evaluation_partition": "visible/validation/demo",
                    "scientific_scope": "prediction_demo_non_causal",
                    "supported_evaluation_partitions": [
                        "validation",
                        "training_feedback",
                    ],
                    "scientific_scopes": [
                        "prediction_demo_non_causal",
                        "historical_replay_prediction_non_causal",
                    ],
                    "dsh_authenticated_models": sum(
                        1
                        for item in self.server.model_gateway.catalog()
                        if item.get("authenticated")
                    ),
                },
            )
            return
        if path == ["plugin", "ecology_evolution"]:
            self._send(HTTPStatus.OK, PLUGIN_MANIFEST)
            return
        if path == ["catalog"]:
            self._call(self._catalog_payload)
            return
        if len(path) == 2 and path[0] == "datasets":
            self._call(lambda: self._dataset_payload(path[1]))
            return
        if len(path) == 3 and path[0] == "datasets" and path[2] == "samples":
            self._call(lambda: self._dataset_sample_payload(path[1]))
            return
        if path == ["runs"]:
            self._call(self._list_runs)
            return
        if len(path) == 2 and path[0] == "runs":
            self._call(lambda: self._run_payload(path[1]))
            return
        if len(path) == 3 and path[0] == "runs" and path[2] == "events":
            self._call(lambda: self._events_payload(path[1]))
            return
        if len(path) == 3 and path[0] == "runs" and path[2] == "samples":
            self._call(lambda: self._sample_results_payload(path[1]))
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        raw_path = urlparse(self.path).path
        tool_prefix = "/api/ecology-agent-sidecar/v1/tools/"
        if raw_path == "/api/ecology-agent-sidecar/v1/child-reservations":
            if not self._authorize_dsh_tool():
                return
            try:
                result = self.server.dsh_tools.allocate_child_reservation(self._body())
                self._send(HTTPStatus.OK, result)
            except PermissionError as exc:
                self._send(HTTPStatus.FORBIDDEN, {"error": _public_http_error(exc)})
            except (RuntimeError, TypeError, ValueError) as exc:
                self._send(HTTPStatus.CONFLICT, {"error": _public_http_error(exc)})
            return
        if raw_path == "/api/ecology-agent-sidecar/v1/structured-results":
            if not self._authorize_dsh_tool():
                return
            try:
                result = self.server.dsh_tools.accept_structured(self._body())
                self._send(HTTPStatus.OK, result)
            except PermissionError as exc:
                self._send(HTTPStatus.FORBIDDEN, {"error": _public_http_error(exc)})
            except (RuntimeError, TypeError, ValueError) as exc:
                self._send(HTTPStatus.CONFLICT, {"error": _public_http_error(exc)})
            return
        if raw_path.startswith(tool_prefix):
            if not self._authorize_dsh_tool():
                return
            tool_name = raw_path[len(tool_prefix):]
            if not tool_name or "/" in tool_name:
                self._send(HTTPStatus.NOT_FOUND, {"error": "unknown DSH role tool"})
                return
            try:
                result = self.server.dsh_tools.execute(tool_name, self._body())
                self._send(HTTPStatus.OK, result)
            except PermissionError as exc:
                self._send(HTTPStatus.FORBIDDEN, {"error": _public_http_error(exc)})
            except (RuntimeError, TypeError, ValueError) as exc:
                self._send(HTTPStatus.CONFLICT, {"error": _public_http_error(exc)})
            return
        if not self._authorize_api():
            return
        path = self._route()
        self._active_command: dict[str, Any] | None = None
        try:
            body = self._body()
            long_running_advance = (
                len(path) == 3
                and path[0] == "runs"
                and path[2] in ("advance", "step")
            )
            if long_running_advance:
                # Generation execution owns a per-run lease and already uses
                # thread-safe ledger writes.  Keeping the global mutation lock
                # across remote model waits would block pause/cancel and every
                # unrelated run for the full generation.
                self._dispatch_post(path, body)
            else:
                with self.server.mutation_lock:
                    self._dispatch_post(path, body)
        except CommandInProgressError as exc:
            self._send_post_error(HTTPStatus.CONFLICT, exc)
        except DshNativeRuntimeUnavailableError as exc:
            self._send_post_error(HTTPStatus.SERVICE_UNAVAILABLE, exc)
        except KeyError as exc:
            self._send_post_error(HTTPStatus.NOT_FOUND, exc)
        except (RuntimeError, StopIteration, TypeError, ValueError) as exc:
            self._send_post_error(HTTPStatus.BAD_REQUEST, exc)
        except Exception as exc:  # pragma: no cover - last-resort receipt cleanup
            self._send_post_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                RuntimeError(f"命令执行出现未预期错误：{type(exc).__name__}"),
            )

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._authorize_api():
            return
        path = self._route()
        self._active_command: dict[str, Any] | None = None
        try:
            body = self._body()
            with self.server.mutation_lock:
                if len(path) == 2 and path[0] == "runs":
                    self._delete_run(path[1], body)
                    return
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except CommandInProgressError as exc:
            self._send_post_error(HTTPStatus.CONFLICT, exc)
        except KeyError as exc:
            self._send_post_error(HTTPStatus.NOT_FOUND, exc)
        except (RuntimeError, StopIteration, TypeError, ValueError) as exc:
            self._send_post_error(HTTPStatus.BAD_REQUEST, exc)
        except Exception as exc:  # pragma: no cover - last-resort safety net
            self._send_post_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                RuntimeError(f"删除运行出现未预期错误：{type(exc).__name__}"),
            )

    def _dispatch_post(self, path: list[str], body: dict[str, Any]) -> None:
        if path == ["runs"]:
            self._create_run(body)
            return
        if len(path) == 3 and path[0] == "runs" and path[2] == "archive":
            self._archive_run(path[1], body)
            return
        if len(path) == 3 and path[0] == "runs" and path[2] == "restore":
            self._restore_run(path[1], body)
            return
        if len(path) == 3 and path[0] == "runs" and path[2] in ("advance", "step"):
            run_id = path[1]
            cache_key = self._command_key(run_id, body)
            if self._serve_existing_command(cache_key, run_id, "advance", body):
                return
            self._validate_advance_request(run_id, body)
            cached = self._claim_command(cache_key, run_id, "advance", body)
            if cached is not None:
                self._send(HTTPStatus.OK, cached)
                return
            state = self._advance_run(run_id, body)
            payload = _state_payload(state)
            self._complete_command(cache_key, payload)
            # Keep a continuous run moving even when a user or recovery tool
            # manually consumes one generation while the worker is idle.
            self._schedule_auto_progress(state)
            self._send(HTTPStatus.OK, payload)
            return
        if len(path) == 3 and path[0] == "runs" and path[2] in ("action", "control"):
            self._action(path[1], body)
            return
        if len(path) == 3 and path[0] == "runs" and path[2] == "interventions":
            self._record_intervention(path[1], body)
            return
        if (
            len(path) == 5
            and path[0] == "runs"
            and path[2] == "expert-consultations"
            and path[4] == "answer"
        ):
            self._answer_expert_consultation(path[1], path[3], body)
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _archive_run(self, run_id: str, body: dict[str, Any]) -> None:
        if body:
            raise ValueError("归档运行请求不接受额外参数")
        state = self.server.director.state(run_id)
        _assert_http_scope(state)
        if state.run.status.value not in {"completed", "cancelled", "failed"}:
            raise RuntimeError("只能归档已完成、已取消或失败的终态运行")
        archived_at = self.server.ledger.archive_run(run_id)
        payload = self._run_payload(run_id)
        payload["archive"] = {
            "run_id": run_id,
            "archived": True,
            "archived_at": archived_at,
        }
        self._send(HTTPStatus.OK, payload)

    def _restore_run(self, run_id: str, body: dict[str, Any]) -> None:
        if body:
            raise ValueError("恢复归档请求不接受额外参数")
        state = self.server.director.state(run_id)
        _assert_http_scope(state)
        restored = self.server.ledger.restore_run(run_id)
        payload = self._run_payload(run_id)
        payload["archive"] = {
            "run_id": run_id,
            "archived": False,
            "restored": restored,
        }
        self._send(HTTPStatus.OK, payload)

    def _delete_run(self, run_id: str, body: dict[str, Any]) -> None:
        if set(body) != {"confirm_run_id"}:
            raise ValueError("永久删除必须且只能提交 confirm_run_id")
        confirmation = body["confirm_run_id"]
        if not isinstance(confirmation, str) or confirmation != run_id:
            raise ValueError("confirm_run_id 必须与待删除运行 ID 完全一致")
        # Reject unknown, non-terminal, and unarchived targets before allocating
        # a lease entry. The projection is checked again under the purge fence.
        initial_state = self.server.director.state(run_id)
        _assert_http_scope(initial_state)
        if initial_state.run.status.value not in {"completed", "cancelled", "failed"}:
            raise RuntimeError("只能永久删除已完成、已取消或失败的终态运行")
        if self.server.ledger.archived_at(run_id) is None:
            raise ValueError("运行必须先归档，才能永久删除")
        purge_lease = self.server.try_acquire_generation_purge_lease(run_id)
        if purge_lease is None:
            raise CommandInProgressError(
                "run still has an active or waiting generation; retry deletion "
                "after that generation releases its lease"
            )
        purged = False
        state = None
        try:
            state = self.server.director.state(run_id)
            _assert_http_scope(state)
            terminal_status = state.run.status.value
            if terminal_status not in {"completed", "cancelled", "failed"}:
                raise RuntimeError("只能永久删除已完成、已取消或失败的终态运行")
            deleted = self.server.ledger.purge_run(
                run_id,
                confirmation=confirmation,
                terminal_status=terminal_status,
            )
            purged = True
            # Queue entries cannot be physically removed. Withdraw their exact
            # incarnation while the retiring lease still prevents dequeue from
            # creating a replacement lock for the deleted run.
            self.server.auto_progress.forget(state)
        finally:
            self.server.release_generation_purge_lease(
                purge_lease,
                purged=purged,
            )
        assert state is not None
        self._send(
            HTTPStatus.OK,
            {"ok": True, "run_id": run_id, "permanently_deleted": True, "deleted": deleted},
        )

    def _create_run(self, body: dict[str, Any]) -> None:
        # Resolve execution policy before task binding. Autonomous research is
        # always deferred to generation execution so creating a durable run is
        # independent of current upstream queue health.
        start_value = body.get("start", True)
        if not isinstance(start_value, bool):
            raise TypeError("start must be a bool")
        start = start_value
        requested_auto_advance = _auto_advance_steps(body.get("auto_advance", 1))
        requested_auto_progress = body.get("auto_progress", False)
        if not isinstance(requested_auto_progress, bool):
            requested_auto_progress = _request_boolean(
                requested_auto_progress, "auto_progress"
            )
        requested_allow_host_fallback = body.get("allow_host_fallback", False)
        if requested_allow_host_fallback is None:
            allow_host_fallback = False
        elif isinstance(requested_allow_host_fallback, bool):
            allow_host_fallback = requested_allow_host_fallback
        else:
            allow_host_fallback = _request_boolean(
                requested_allow_host_fallback, "allow_host_fallback"
            )
        auto_advance = requested_auto_advance if start else 0
        # Preserve a continuous opt-in even when the caller creates a paused
        # run with ``start=false``; the later start/resume control transition
        # will schedule it from the frozen manifest.
        continuous_auto_progress = (
            requested_auto_advance == AUTO_ADVANCE_CONTINUOUS
            or requested_auto_progress
        )
        native_protocol = body.get("execution_protocol") == DSH_NATIVE_EXECUTION_PROTOCOL
        native_capabilities = None
        live_capabilities = None
        if native_protocol:
            if body.get("allow_host_fallback") not in (None, False):
                raise ValueError("DSH-native runs do not permit host/model gateway fallback")
            runtime = self.server.dsh_native_runtime
            if runtime is None:
                raise DshNativeRuntimeUnavailableError()
            native_capabilities = runtime.capabilities()
            runtime.require_capabilities(
                native_capabilities,
                _DSH_NATIVE_PRESET_IDS,
                require_live=False,
            )
            self._dsh_native_capabilities = native_capabilities
        task = self._task_from_request(
            body,
            # Creating a run is a local durability operation. Research and
            # strategy calls always belong to generation execution, regardless
            # of fallback policy, so an upstream queue can never block POST
            # /runs or be mistaken for an invalid API configuration.
            strict_remote_plan=False,
            defer_remote_plan=True,
        )
        autonomous_mode = task.metadata.get("autonomous_mode") is True
        if autonomous_mode and requested_auto_advance != 0:
            # Preserve explicit paused creation (auto_advance=0), while making
            # every started autonomous run asynchronous even for legacy
            # callers that still send the old one-step integer value.
            continuous_auto_progress = True
            auto_advance = AUTO_ADVANCE_CONTINUOUS if start else 0
        # Validate the immutable manifest before claiming an idempotency
        # receipt or appending RunCreated.  Invalid scope requests must not
        # leave a durable pending command or a partially-created Run.
        _assert_manifest_http_scope(task)
        if continuous_auto_progress:
            # The mode is part of the frozen task digest so a restart can
            # discover and resume a still-running autonomous run without
            # relying on an in-memory queue.
            task = _mark_auto_progress_task(
                task, allow_host_fallback=allow_host_fallback
            )
            self._validate_continuous_remote_bindings(task)
        idempotency_key = task.metadata.get("idempotency_key")
        cache_key = self._command_key("create", body)
        run_id = body.get("run_id")
        if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
            raise ValueError("run_id must be a non-empty string")
        if self._serve_existing_command(cache_key, "create", "create_run", body):
            return
        existing_idempotent_run = self._find_existing_idempotent_run(idempotency_key, task)
        existing_idempotent_state = None
        if existing_idempotent_run is not None:
            existing_idempotent_state = self.server.director.state(existing_idempotent_run)
            _assert_http_scope(existing_idempotent_state)
        if run_id is not None:
            existing_events = self.server.ledger.events(run_id)
            if existing_events:
                if existing_idempotent_run == run_id:
                    cached = self._claim_command(cache_key, "create", "create_run", body)
                    if cached is not None:
                        self._send(HTTPStatus.OK, cached)
                        return
                    if cache_key is not None:
                        self.server.ledger.bind_command_resource_run(cache_key, run_id)
                    state = self._resume_created_run(
                        existing_idempotent_state, start=start, auto_advance=auto_advance
                    )
                    payload = _state_payload(state)
                    self._complete_command(cache_key, payload)
                    self._schedule_auto_progress(state)
                    self._send(HTTPStatus.OK, payload)
                    return
                if existing_idempotent_run is not None:
                    raise ValueError("idempotency key already belongs to a different run")
                raise ValueError(f"run already exists: {run_id}")
        cached = self._claim_command(cache_key, "create", "create_run", body)
        if cached is not None:
            self._send(HTTPStatus.OK, cached)
            return
        if existing_idempotent_run is not None:
            if cache_key is not None:
                self.server.ledger.bind_command_resource_run(
                    cache_key, existing_idempotent_run
                )
            state = self._resume_created_run(
                existing_idempotent_state, start=start, auto_advance=auto_advance
            )
            payload = _state_payload(state)
            self._complete_command(cache_key, payload)
            self._schedule_auto_progress(state)
            self._send(HTTPStatus.OK, payload)
            return
        if native_protocol:
            run_id = run_id or f"run:{uuid4()}"
            native_request = {
                "run_id": run_id,
                "run_state_revision": 0,
                "stage_attempt": 0,
                "ledger_expected_revision": self.server.ledger.latest_seq(),
                "idempotency_key": str(idempotency_key or f"create:{run_id}"),
                "binding": {
                    "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                    "task_manifest_digest": task.digest,
                    "preset_catalog_digest": task.metadata.get("dsh_preset_catalog_digest"),
                    "data_protocol_digest": task.metadata.get("data_protocol_digest"),
                    "preset_content_digest": task.metadata.get("preset_content_digest"),
                    "standing_tool_surface_digest": task.metadata.get(
                        "standing_tool_surface_digest"
                    ),
                    "resolved_policy_route_config_digest": task.metadata.get(
                        "resolved_policy_route_config_digest"
                    ),
                    "resolved_review_route_config_digest": task.metadata.get(
                        "resolved_review_route_config_digest"
                    ),
                    "strategy_model_id": task.metadata.get("strategy_model_id"),
                    "review_model_id": task.metadata.get("review_model_id"),
                },
            }
            try:
                self.server.dsh_native_runtime.create_run(native_request)
                live_capabilities = self.server.dsh_native_runtime.capabilities()
                self.server.dsh_native_runtime.require_capabilities(
                    live_capabilities,
                    _DSH_NATIVE_PRESET_IDS,
                    require_live=True,
                )
            except BaseException:
                try:
                    self.server.dsh_native_runtime.cancel(
                        {
                            "run_id": run_id,
                            "run_state_revision": 0,
                            "stage_attempt": 0,
                            "ledger_expected_revision": self.server.ledger.latest_seq(),
                            "idempotency_key": str(idempotency_key or f"create:{run_id}"),
                        }
                    )
                except DshNativeRuntimeUnavailableError:
                    pass
                raise
        try:
            run = self.server.director.create_run(task, run_id=run_id)
        except BaseException:
            if native_protocol and run_id is not None:
                try:
                    self.server.dsh_native_runtime.cancel(
                        {
                            "run_id": run_id,
                            "run_state_revision": 0,
                            "stage_attempt": 0,
                            "ledger_expected_revision": self.server.ledger.latest_seq(),
                            "idempotency_key": str(idempotency_key or f"create:{run_id}"),
                        }
                    )
                except DshNativeRuntimeUnavailableError:
                    pass
            raise
        if native_protocol:
            self.server.ledger.append(
                run.run_id,
                "DshRuntimeBound",
                {
                    "schema_version": "ecologyrsi-dsh.runtime-bound/1",
                    "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                    "capabilities_digest": digest(live_capabilities),
                    "preset_ids": list(_DSH_NATIVE_PRESET_IDS),
                    "first_call_verified": False,
                },
                event_id=f"{run.run_id}:dsh-runtime-bound",
                expected_run_seq=self.server.director.state(run.run_id).events[-1].seq,
            )
        if cache_key is not None:
            # Bind immediately after RunCreated.  A failure in start or the
            # first automatic generation must not leave an ownerless receipt.
            self.server.ledger.bind_command_resource_run(cache_key, run.run_id)
        if start:
            self.server.director.start_run(run.run_id)
        state = self.server.director.state(run.run_id)
        # Continuous runs return their durable generation-zero state and let
        # the worker execute every generation.  Numeric values retain the old
        # bounded synchronous behavior.
        if start and not continuous_auto_progress and auto_advance > 0:
            state = self._advance_run(
                run.run_id,
                {"steps": auto_advance},
                target_steps=auto_advance,
            )
        payload = _state_payload(state)
        self._complete_command(cache_key, payload)
        self._schedule_auto_progress(state)
        self._send(HTTPStatus.CREATED, payload)

    def _schedule_auto_progress(self, state: Any) -> None:
        """Queue a running continuous run after its response state is built."""

        if state is None or state.run.status.value != "running":
            return
        self.server.auto_progress.schedule_if_enabled(state.run.run_id)

    def _validate_continuous_remote_bindings(self, task: TaskManifest) -> None:
        """Validate callable role bindings without treating old health as truth.

        Connection health is observational and can be stale after a queued or
        timed-out request.  Creation therefore validates only the durable
        prerequisites; the background model call owns timeout and retry policy.
        """

        metadata = task.metadata
        if metadata.get("execution_protocol") == DSH_NATIVE_EXECUTION_PROTOCOL:
            return
        remote_models = {
            str(item.get("model_id") or ""): item
            for item in self.server.model_gateway.catalog()
        }
        for field, label, role in (
            ("strategy_model_id", "策略模型", "propose"),
            ("judge_model_id", "独立评审模型", "judge"),
        ):
            model_id = str(metadata.get(field) or "").strip()
            if not model_id or builtin_model_configuration_digest(model_id) is not None:
                continue
            remote_model = remote_models.get(model_id)
            if remote_model is None:
                raise ValueError(
                    f"连续进化的{label}未配置：{model_id}"
                )
            if role not in remote_model.get("roles", []):
                raise ValueError(
                    f"连续进化的{label}缺少 {role} 权限：{model_id}"
                )
            if not remote_model.get("credential_configured"):
                raise ValueError(
                    f"连续进化的{label}尚未配置认证凭据：{model_id}"
                )
            if not self._remote_model_execution_configured(remote_model):
                raise ValueError(
                    f"连续进化的{label}后端调用配置不可执行：{model_id}"
                )

    def _resume_created_run(
        self,
        state: Any,
        *,
        start: bool,
        auto_advance: int,
    ) -> Any:
        """Finish a create command from its durable run projection."""

        run_id = state.run.run_id
        if start and state.run.status.value == "created":
            self.server.director.start_run(run_id)
            state = self.server.director.state(run_id)
        continuous = auto_advance == AUTO_ADVANCE_CONTINUOUS or bool(
            state.task_manifest.metadata.get("auto_progress") is True
        )
        if (
            start
            and not continuous
            and auto_advance > 0
            and state.run.status.value == "running"
        ):
            state = self._advance_run(
                run_id,
                {"steps": auto_advance},
                target_steps=auto_advance,
            )
        return state

    def _find_existing_idempotent_run(
        self,
        idempotency_key: Any,
        task: TaskManifest,
    ) -> str | None:
        """Find a prior create request without claiming a new receipt."""

        if not idempotency_key:
            return None
        for existing_run_id in self.server.ledger.run_ids():
            existing_events = self.server.ledger.events(existing_run_id)
            if not existing_events or existing_events[0].kind != "RunCreated":
                continue
            existing_task = existing_events[0].payload.get("task_manifest", {})
            if not isinstance(existing_task, dict):
                continue
            metadata = existing_task.get("metadata", {})
            if not isinstance(metadata, dict) or metadata.get("idempotency_key") != idempotency_key:
                continue
            existing_manifest = TaskManifest.from_dict(existing_task)
            if existing_manifest.digest != task.digest:
                raise ValueError("idempotency key already belongs to a different task")
            return existing_run_id
        return None

    def _task_from_request(
        self,
        body: dict[str, Any],
        *,
        strict_remote_plan: bool = False,
        defer_remote_plan: bool = False,
    ) -> TaskManifest:
        """Accept a full manifest or the compact Chinese workbench payload."""

        if isinstance(body.get("task_manifest"), dict):
            raw = dict(body["task_manifest"])
            if "domain_pack" not in raw:
                supplied_domain = body.get(
                    "domain_pack",
                    body.get(
                        "domain_pack_id",
                        body.get(
                            "domain",
                            body.get(
                                "domain_id",
                                raw.get(
                                    "domain_pack_id",
                                    raw.get("domain", raw.get("domain_id")),
                                ),
                            ),
                        ),
                    ),
                )
                if supplied_domain is not None:
                    raw["domain_pack"] = str(supplied_domain)
            if body.get("idempotency_key") or body.get("episode_id") or any(
                key in body
                for key in (
                    "strategy_model_id",
                    "review_model_id",
                    "reviewer_model_id",
                    "autonomous_mode",
                    "rounds",
                    "model_workflow",
                    "workflow",
                    "research_domain",
                    "research_domain_id",
                    "research_scope",
                    "research_area",
                    "candidates_per_generation",
                    "candidates_per_round",
                    "variants_per_round",
                    "max_candidates",
                    "samples_per_update",
                    "sample_concurrency",
                    "sample_agent_batch_size",
                    "execution_protocol",
                )
            ):
                metadata = dict(raw.get("metadata", {}))
                if body.get("idempotency_key"):
                    metadata["idempotency_key"] = str(body["idempotency_key"])
                if body.get("episode_id"):
                    metadata["episode_id"] = str(body["episode_id"])
                strategy_model_id = body.get("strategy_model_id")
                review_model_id = body.get(
                    "review_model_id",
                    body.get("reviewer_model_id"),
                )
                if strategy_model_id is not None:
                    metadata["strategy_model_id"] = str(strategy_model_id)
                    metadata.setdefault("policy_model_id", str(strategy_model_id))
                if review_model_id is not None:
                    metadata["review_model_id"] = str(review_model_id)
                    metadata.setdefault("judge_model_id", str(review_model_id))
                if "autonomous_mode" in body:
                    metadata["autonomous_mode"] = _request_boolean(
                        body["autonomous_mode"], "autonomous_mode"
                    )
                if "execution_protocol" in body:
                    metadata["execution_protocol"] = str(body["execution_protocol"])
                if "model_workflow" in body or "workflow" in body:
                    metadata["model_workflow"] = str(
                        body.get("model_workflow", body.get("workflow"))
                    ).strip()
                if any(key in body for key in ("research_domain", "research_domain_id", "research_scope", "research_area")):
                    metadata["research_domain"] = str(
                        body.get(
                            "research_domain",
                            body.get(
                                "research_domain_id",
                                body.get("research_scope", body.get("research_area")),
                            ),
                        )
                    ).strip()
                if "rounds" in body:
                    metadata["requested_rounds"] = _request_integer(
                        body["rounds"], "rounds", minimum=1
                    )
                if "sample_concurrency" in body:
                    sample_concurrency = _request_integer(
                        body["sample_concurrency"], "sample_concurrency", minimum=1
                    )
                    if sample_concurrency > _MAX_REAL_SAMPLE_CONCURRENCY:
                        raise ValueError(
                            "sample_concurrency must be between 1 and "
                            f"{_MAX_REAL_SAMPLE_CONCURRENCY}"
                        )
                    metadata["sample_concurrency"] = sample_concurrency
                if "samples_per_update" in body:
                    samples_per_update = _request_integer(
                        body["samples_per_update"], "samples_per_update", minimum=1
                    )
                    if samples_per_update > _MAX_SAMPLES_PER_UPDATE:
                        raise ValueError(
                            "samples_per_update must be between 1 and "
                            f"{_MAX_SAMPLES_PER_UPDATE}"
                        )
                    metadata["samples_per_update"] = samples_per_update
                if "sample_agent_batch_size" in body:
                    sample_agent_batch_size = _request_integer(
                        body["sample_agent_batch_size"],
                        "sample_agent_batch_size",
                        minimum=1,
                    )
                    if sample_agent_batch_size > _MAX_SAMPLE_AGENT_BATCH_SIZE:
                        raise ValueError(
                            "sample_agent_batch_size must be between 1 and "
                            f"{_MAX_SAMPLE_AGENT_BATCH_SIZE}"
                        )
                    metadata["sample_agent_batch_size"] = sample_agent_batch_size
                raw["metadata"] = metadata
            raw_dataset = raw.get("visible_datasets")
            supplied_dataset = body.get(
                "dataset",
                body.get(
                    "dataset_id",
                    raw.get("dataset", raw.get("dataset_id")),
                ),
            )
            if raw_dataset and supplied_dataset is not None:
                raw_dataset_id = self._single_dataset_id(raw_dataset)
                supplied_dataset_id = self._single_dataset_id(supplied_dataset)
                if raw_dataset_id != supplied_dataset_id:
                    raise ValueError(
                        "task_manifest.visible_datasets 与 dataset_id 不一致"
                    )
            if not raw_dataset:
                if supplied_dataset is not None:
                    raw["visible_datasets"] = [
                        self._single_dataset_id(supplied_dataset)
                    ]
                else:
                    domain = raw.get("domain_pack", raw.get("domain"))
                    if domain is None:
                        raise ValueError(
                            "dataset_id（或 domain_pack_id/domain）至少需要提供一项"
                        )
                    raw["visible_datasets"] = [
                        self._default_dataset_for_domain(str(domain))
                    ]
            else:
                raw["visible_datasets"] = [self._single_dataset_id(raw_dataset)]
            # Dataset aliases are request conveniences, not TaskManifest
            # fields; keep the immutable manifest schema canonical.
            raw.pop("dataset", None)
            raw.pop("dataset_id", None)
            raw.pop("domain_pack_id", None)
            raw.pop("domain_id", None)
            inferred_domain = self._domain_pack_for_dataset(
                raw["visible_datasets"]
            )
            supplied_domain = raw.get("domain_pack", raw.get("domain"))
            if supplied_domain is None:
                raw["domain_pack"] = inferred_domain
            else:
                self._validate_dataset_domain(
                    str(supplied_domain), inferred_domain, raw["visible_datasets"]
                )
                raw["domain_pack"] = self._canonical_domain_pack(
                    str(supplied_domain)
                )
            raw.pop("domain", None)
            if "rounds" in body:
                rounds = _request_integer(body["rounds"], "rounds", minimum=1)
                budget = raw.get("budget", 1)
                if isinstance(budget, dict):
                    budget = dict(budget)
                    budget["max_generations"] = rounds
                else:
                    budget = {"max_generations": rounds}
                raw["budget"] = budget
            if isinstance(raw.get("budget"), dict):
                budget = dict(raw["budget"])
                compact_candidates = body.get(
                    "candidates_per_generation",
                    body.get("candidates_per_round", body.get("variants_per_round")),
                )
                if compact_candidates is not None:
                    budget["candidates_per_generation"] = _request_integer(
                        compact_candidates, "candidates_per_generation", minimum=1
                    )
                if body.get("max_candidates") is not None:
                    budget["max_candidates"] = _request_integer(
                        body["max_candidates"], "max_candidates", minimum=1
                    )
                raw["budget"] = budget
            seed_policy = str(raw.get("seed_policy", "fixed"))
            if "seed" not in raw and seed_policy == "generated_and_recorded":
                raw["seed"] = _derived_seed(
                    body.get("idempotency_key") or {"task_manifest": raw}
                )
            return self._bind_runtime_task(
                TaskManifest.from_dict(raw),
                strict_remote_plan=strict_remote_plan,
                defer_remote_plan=defer_remote_plan,
            )

        domain_pack = body.get(
            "domain_pack",
            body.get(
                "domain_pack_id",
                body.get(
                    "domain",
                    body.get("domain_id", body.get("research_domain_id")),
                ),
            ),
        )
        dataset = body.get("dataset", body.get("dataset_id"))
        if domain_pack is None and dataset is None:
            raise ValueError(
                "dataset_id（或 domain_pack_id/domain）至少需要提供一项"
            )
        if dataset is None:
            dataset = self._default_dataset_for_domain(str(domain_pack))
        inferred_domain = self._domain_pack_for_dataset(dataset)
        if domain_pack is None:
            domain_pack = inferred_domain
        else:
            self._validate_dataset_domain(
                str(domain_pack), inferred_domain, dataset
            )
            domain_pack = self._canonical_domain_pack(str(domain_pack))
        datasets = tuple(dataset) if isinstance(dataset, (list, tuple)) else (str(dataset),)
        rounds_value = body.get(
            "rounds",
            body.get(
                "generations",
                body.get("max_rounds", body.get("max_generations", body.get("num_rounds"))),
            ),
        )
        if rounds_value is not None:
            rounds = _request_integer(rounds_value, "rounds", minimum=1)
        else:
            rounds = None
        budget_value = body.get("budget")
        if budget_value is None:
            budget_value = {"max_generations": rounds} if rounds is not None else 3
        if isinstance(budget_value, dict):
            budget = dict(budget_value)
        else:
            budget = {"max_candidates": _request_integer(budget_value, "budget", minimum=1)}
        if rounds is not None:
            budget["max_generations"] = rounds
        candidates_value = body.get(
            "candidates_per_generation",
            body.get("candidates_per_round", body.get("variants_per_round")),
        )
        if candidates_value is not None:
            budget["candidates_per_generation"] = _request_integer(
                candidates_value,
                "candidates_per_generation",
                minimum=1,
            )
        if body.get("max_candidates") is not None:
            budget["max_candidates"] = _request_integer(
                body["max_candidates"], "max_candidates", minimum=1
            )
        # ``slot`` is a UI slot name in the plugin.  Numeric slots are also a
        # convenient candidate budget for a local demo.
        slot = body.get("slot")
        if isinstance(slot, int) and not isinstance(slot, bool):
            budget["max_candidates"] = slot
        strategy_model_id = body.get("strategy_model_id")
        # Keep canonical role fields separate from legacy policy/judge aliases
        # while deciding which workflow was requested.  A legacy request that
        # only names ``judge_model_id`` must remain on the compatibility path.
        review_model_id = body.get(
            "review_model_id", body.get("reviewer_model_id")
        )
        explicit_autonomous_mode = _request_boolean(
            body.get("autonomous_mode", False), "autonomous_mode"
        )
        role_model_requested = (
            strategy_model_id is not None or review_model_id is not None
        )
        autonomous_requested = explicit_autonomous_mode or role_model_requested or bool(
            str(body.get("requested_mode", "")).casefold()
            in {"autonomous", "self_evolution", "model_driven", "自主进化"}
        )
        explicit_budget_controls = any(
            body.get(name) is not None
            for name in (
                "budget",
                "rounds",
                "generations",
                "max_rounds",
                "max_generations",
                "num_rounds",
                "candidates_per_generation",
                "candidates_per_round",
                "variants_per_round",
                "max_candidates",
            )
        ) or (isinstance(slot, int) and not isinstance(slot, bool))
        autonomous_default_budget = autonomous_requested and not explicit_budget_controls
        if autonomous_default_budget:
            budget = {
                "max_generations": 5,
                "candidates_per_generation": 4,
                "max_candidates": 20,
            }
        if autonomous_requested and strategy_model_id is None:
            # ``policy_model_id`` is retained as a migration alias, but a new
            # request is expected to name it as the strategy model.
            strategy_model_id = body.get("policy_model_id")
        if autonomous_requested and review_model_id is None:
            review_model_id = body.get("judge_model_id")
        if autonomous_requested and strategy_model_id is None:
            raise ValueError("自主进化必须同时提供 strategy_model_id（策略模型 API）")
        if autonomous_requested and review_model_id is None:
            raise ValueError("自主进化必须同时提供 review_model_id（独立评审模型 API）")
        # In the compact contract the role-specific names are canonical.  Keep
        # the older policy/judge aliases only as fallbacks so a request that
        # carries both generations cannot bind the UI selection to one model
        # while executing another.
        legacy_policy_model_id = body.get("policy_model_id")
        legacy_judge_model_id = body.get("judge_model_id")
        canonical_policy_model_id = (
            strategy_model_id if autonomous_requested else legacy_policy_model_id
        )
        canonical_judge_model_id = (
            review_model_id if autonomous_requested else legacy_judge_model_id
        )
        metadata = {
            "slot": slot,
            "requested_mode": body.get("requested_mode", "evolution"),
            "idempotency_key": body.get("idempotency_key"),
            "strategy_id": body.get(
                "strategy_id",
                "autonomous_model@1" if autonomous_requested else "parameter_sweep@1",
            ),
            "prediction_model_id": body.get("prediction_model_id"),
            "evaluator_id": body.get("evaluator_id"),
            "policy_model_id": (
                canonical_policy_model_id
                if canonical_policy_model_id is not None
                else HOST_PARAMETER_GENERATOR_ID
            ),
            "judge_model_id": (
                canonical_judge_model_id
                if canonical_judge_model_id is not None
                else RULE_JUDGE_ID
            ),
            "strategy_model_id": strategy_model_id,
            "review_model_id": review_model_id,
            "autonomous_mode": autonomous_requested,
            "model_selection_policy": (
                "model_research_and_runtime_compile@1"
                if autonomous_requested
                else None
            ),
            "model_workflow": body.get(
                "model_workflow",
                body.get(
                    "workflow",
                    "research_compile_evolve@1"
                    if autonomous_requested
                    else "legacy_component_search@1",
                ),
            ),
            "research_domain": body.get(
                "research_domain",
                body.get(
                    "research_domain_id",
                    body.get("research_scope", body.get("research_area", domain_pack)),
                ),
            ),
            "requested_rounds": rounds,
            "budget_profile": (
                "autonomous_default_5x4"
                if autonomous_default_budget
                else "request_defined_or_legacy_default"
            ),
            "knowledge_online_enabled": _request_boolean(
                body.get("knowledge_online_enabled", autonomous_requested),
                "knowledge_online_enabled",
            ),
            "episode_id": (
                str(body.get("episode_id")).strip()
                if body.get("episode_id")
                else None
            ),
            "sample_concurrency": body.get("sample_concurrency"),
            "samples_per_update": body.get("samples_per_update"),
            "sample_agent_batch_size": body.get("sample_agent_batch_size"),
            "execution_protocol": body.get("execution_protocol"),
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}
        seed_policy = str(body.get("seed_policy", "fixed"))
        if "seed" in body:
            seed = _request_integer(body["seed"], "seed")
        elif seed_policy == "generated_and_recorded":
            seed = _derived_seed(body.get("idempotency_key") or body)
        else:
            seed = 0
        metadata["recorded_seed"] = seed
        return self._bind_runtime_task(
            TaskManifest(
                task_id=str(
                    body.get("task_id")
                    or f"plugin:{body.get('idempotency_key') or uuid4()}"
                ),
                objective=str(
                    body.get("objective")
                    or f"evolution:{body.get('requested_mode', 'default')}"
                ),
                domain_pack=str(domain_pack),
                visible_datasets=datasets,
                budget=budget,
                seed=seed,
                seed_policy=seed_policy,
                policy_version=str(body.get("policy_version", "policy@1")),
                metadata=metadata,
            ),
            strict_remote_plan=strict_remote_plan,
            defer_remote_plan=defer_remote_plan,
        )

    @staticmethod
    def _single_dataset_id(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError("本地运行必须且只能选择一个 dataset_id")
            value = value[0]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("dataset_id 必须是非空字符串")
        return value.strip()

    @staticmethod
    def _canonical_domain_pack(value: str) -> str:
        normalized = value.strip().casefold().replace("_", "-")
        if normalized in {
            "crop-soil-water",
            "crop-soil-water@toy",
            "toy",
        }:
            return "crop_soil_water"
        if normalized == "greenhouse" or normalized.startswith("greenhouse-"):
            return "greenhouse_environment@1"
        return value.strip()

    def _domain_pack_for_dataset(self, dataset: Any) -> str:
        """Derive the executable domain pack from one registered dataset."""

        dataset_id = self._single_dataset_id(dataset)
        if dataset_id in {TOY_DATASET_ID, "toy-dataset@1"}:
            return "crop_soil_water"
        try:
            descriptor = self.server.datasets.describe(dataset_id)["descriptor"]
        except KeyError:
            raise ValueError(
                f"未知数据集：{dataset_id}；当前本地运行支持 {TOY_DATASET_ID}"
            ) from None
        domain_id = str(descriptor.get("domain_id", "")).strip()
        adapter_id = str(descriptor.get("adapter_id", "")).strip()
        canonical = self._canonical_domain_pack(domain_id)
        if adapter_id == "toy_crop_soil_water" or canonical == "crop_soil_water":
            return "crop_soil_water"
        if adapter_id.startswith("greenhouse") or canonical == "greenhouse_environment@1":
            return "greenhouse_environment@1"
        raise ValueError(
            f"数据集尚未映射到可执行领域模型包：{dataset_id}（{domain_id or '未知领域'}）"
        )

    def _validate_dataset_domain(
        self,
        supplied_domain: str,
        inferred_domain: str,
        dataset: Any,
    ) -> None:
        canonical = self._canonical_domain_pack(supplied_domain)
        if canonical != inferred_domain:
            dataset_id = self._single_dataset_id(dataset)
            raise ValueError(
                f"领域与数据集不一致：{dataset_id} 属于 {inferred_domain}，"
                f"请求提供了 {supplied_domain}"
            )

    def _default_dataset_for_domain(self, domain: str) -> str:
        """Choose the first ready dataset when autonomous mode omits one."""

        normalized = domain.casefold().replace("_", "-")
        if "greenhouse" not in normalized and "温室" not in normalized:
            return TOY_DATASET_ID
        catalog = self.server.datasets.catalog().get("datasets", [])
        candidates = [
            item
            for item in catalog
            if isinstance(item, dict)
            and item.get("runnable")
            and isinstance(item.get("readiness"), dict)
            and item["readiness"].get("ready")
            and (
                "greenhouse" in str(item.get("domain_id", "")).casefold()
                or "greenhouse" in str(item.get("dataset_id", "")).casefold()
            )
        ]
        if candidates:
            return str(sorted(candidates, key=lambda item: str(item.get("dataset_id")))[0]["dataset_id"])
        raise ValueError("当前领域没有可运行的温室数据集，请先准备数据集")

    def _bind_runtime_task(
        self,
        manifest: TaskManifest,
        *,
        strict_remote_plan: bool = False,
        defer_remote_plan: bool = False,
    ) -> TaskManifest:
        if len(manifest.visible_datasets) != 1:
            raise ValueError("本地运行必须且只能冻结一个数据集")
        dataset_id = manifest.visible_datasets[0]
        native_protocol = (
            manifest.metadata.get("execution_protocol")
            == DSH_NATIVE_EXECUTION_PROTOCOL
        )
        normalized_domain_pack = manifest.domain_pack.casefold().replace("_", "-")
        toy_domain = normalized_domain_pack in {
            "crop-soil-water@toy",
            "crop-soil-water",
        }
        expected_partition = "validation" if toy_domain else "training_feedback"
        requested_partition = manifest.metadata.get("evaluation_partition", expected_partition)
        if requested_partition != expected_partition:
            raise ValueError(
                f"运行只允许 {expected_partition} 分区；任务清单请求了：{requested_partition}"
            )
        if toy_domain:
            bound = bind_toy_dataset(manifest, required=True)
            dataset_id = TOY_DATASET_ID
            metadata = dict(bound.metadata)
            series = self.server.datasets.series(dataset_id, metadata.get("episode_id"))
            supplied_split_digest = metadata.get("split_manifest_digest")
            if (
                supplied_split_digest is not None
                and supplied_split_digest != series.split_manifest_digest_sha256
            ):
                raise ValueError("任务清单中的时间分区校验值与当前快照不一致")
            metadata["domain"] = "toy"
            metadata["dataset_digest"] = series.digest
            metadata["split_manifest_digest"] = series.split_manifest_digest_sha256
            metadata["episode_id"] = series.episode_id
            metadata["dataset_display_name"] = "作物—土壤—水分合成演示序列"
            metadata["scientific_scope"] = "prediction_demo_non_causal"
            metadata["evaluation_partition"] = "validation"
            manifest = bound
        else:
            description = self.server.datasets.describe(dataset_id)
            descriptor = description["descriptor"]
            readiness = description["readiness"]
            if not descriptor["runnable"] or not readiness["ready"]:
                raise ValueError(f"数据集当前不可运行：{dataset_id}")
            if "greenhouse" not in manifest.domain_pack.casefold():
                raise ValueError("真实温室数据集必须绑定 greenhouse_environment@1 领域模型包")
            metadata = dict(manifest.metadata)
            requested_episode = metadata.get("episode_id")
            series = self.server.datasets.series(
                dataset_id,
                str(requested_episode) if requested_episode is not None else None,
            )
            existing_digest = metadata.get("dataset_digest")
            if existing_digest is not None and existing_digest != series.digest:
                raise ValueError("任务清单中的数据集校验值与当前快照不一致")
            existing_split_digest = metadata.get("split_manifest_digest")
            if (
                existing_split_digest is not None
                and existing_split_digest != series.split_manifest_digest_sha256
            ):
                raise ValueError("任务清单中的时间分区校验值与当前快照不一致")
            metadata["domain"] = "greenhouse"
            metadata["dataset_digest"] = series.digest
            metadata["split_manifest_digest"] = series.split_manifest_digest_sha256
            metadata["episode_id"] = series.episode_id
            metadata["dataset_display_name"] = descriptor["display_name_zh"]
            metadata["scientific_scope"] = "historical_replay_prediction_non_causal"
            metadata["evaluation_partition"] = "training_feedback"
            if native_protocol:
                selection_view = self.server.datasets.selection_view(
                    dataset_id,
                    series.episode_id,
                    expected_dataset_digest=series.digest,
                    expected_split_manifest_digest=series.split_manifest_digest_sha256,
                )
                metadata["data_protocol_digest"] = selection_view.data_protocol_digest
                metadata["selection_view_digest"] = selection_view.selection_view_digest

        autonomous_mode = _request_boolean(
            metadata.get("autonomous_mode", False), "autonomous_mode"
        ) or bool(
            metadata.get("strategy_model_id")
            or metadata.get("strategy_id") == "autonomous_model@1"
        )
        metadata.setdefault(
            "model_workflow",
            "research_compile_evolve@1"
            if autonomous_mode
            else "legacy_component_search@1",
        )
        metadata.setdefault("research_domain", manifest.domain_pack)
        strategy_id = str(
            metadata.get("strategy_id")
            or ("autonomous_model@1" if autonomous_mode else "parameter_sweep@1")
        )
        evaluator_id = str(
            metadata.get("evaluator_id")
            or self.server.evaluators.default_evaluator(dataset_id)
        )
        prediction_model_id = str(
            metadata.get("prediction_model_id")
            or self.server.evaluators.default_predictor(dataset_id)
        )
        strategy_model_id = str(
            metadata.get("strategy_model_id")
            or metadata.get("policy_model_id")
            or HOST_PARAMETER_GENERATOR_ID
        )
        policy_model_id = str(
            # Autonomous runs use the canonical strategy role as the actual
            # proposal binding.  ``policy_model_id`` remains a read/restore
            # alias for legacy manifests, but must not override a new role
            # selection when both fields are present.
            strategy_model_id
            if autonomous_mode
            else metadata.get("policy_model_id")
            or HOST_PARAMETER_GENERATOR_ID
        )
        judge_model_id = str(
            metadata.get("review_model_id")
            or metadata.get("judge_model_id")
            or RULE_JUDGE_ID
        )
        if autonomous_mode:
            strategy_id = "autonomous_model@1"
            if strategy_model_id == HOST_PARAMETER_GENERATOR_ID:
                raise ValueError("自主进化必须选择已接入的策略模型")
            if judge_model_id == RULE_JUDGE_ID:
                raise ValueError("自主进化必须选择已接入的独立评审模型")
        if strategy_id not in StrategyRouterDSHAdapter.SUPPORTED_STRATEGIES:
            raise ValueError(f"未知进化策略：{strategy_id}")
        self.server.evaluators.validate_binding(
            dataset_id, evaluator_id, prediction_model_id
        )
        strategy_digest = self.server.strategy_router.configuration_digest(strategy_id)
        evaluator_digest = self.server.evaluators.evaluator_configuration_digest(
            evaluator_id
        )
        objective_profile = self.server.evaluators.objective_profile(evaluator_id)
        prediction_model_digest = (
            self.server.evaluators.predictor_configuration_digest(prediction_model_id)
        )
        for field, computed, label in (
            ("strategy_digest", strategy_digest, "进化策略"),
            ("evaluator_digest", evaluator_digest, "评测器"),
            ("prediction_model_digest", prediction_model_digest, "预测模型"),
        ):
            supplied = metadata.get(field)
            if supplied is not None and supplied != computed:
                raise ValueError(f"任务清单中的{label}校验值与服务端实现不一致")
        remote_models = (
            {}
            if native_protocol
            else {
                str(item["model_id"]): item
                for item in self.server.model_gateway.catalog()
            }
        )
        if not native_protocol and policy_model_id != HOST_PARAMETER_GENERATOR_ID:
            model = remote_models.get(policy_model_id)
            if model is None or "propose" not in model.get("roles", []):
                raise ValueError(f"候选生成模型不可用或缺少 propose 权限：{policy_model_id}")
        if native_protocol:
            if not policy_model_id or not judge_model_id:
                raise ValueError("DSH-native roles require frozen strategy and review routes")
        elif strategy_id in {"dsh_authenticated@1", "autonomous_model@1"}:
            if policy_model_id == HOST_PARAMETER_GENERATOR_ID:
                raise ValueError("DSH 模型策略必须选择已配置的候选生成模型")
            policy_model = remote_models[policy_model_id]
            if not policy_model.get("credential_configured"):
                raise ValueError("所选 DSH 候选生成模型尚未配置认证凭据")
            if not self._remote_model_execution_configured(policy_model):
                raise ValueError("所选 DSH 候选生成模型的后端调用配置不可执行")
        elif policy_model_id != HOST_PARAMETER_GENERATOR_ID:
            raise ValueError("本地参数策略必须使用内置有界参数生成器")
        if not native_protocol and judge_model_id != RULE_JUDGE_ID:
            judge = remote_models.get(judge_model_id)
            if judge is None or "judge" not in judge.get("roles", []):
                raise ValueError(f"独立评审模型不可用或缺少 judge 权限：{judge_model_id}")
            if not judge.get("credential_configured"):
                raise ValueError("所选 DSH 独立评审模型尚未配置认证凭据")
            if not self._remote_model_execution_configured(judge):
                raise ValueError("所选 DSH 独立评审模型的后端调用配置不可执行")
        if policy_model_id == judge_model_id:
            raise ValueError("候选生成模型与独立评审模型必须相互分离")
        if native_protocol:
            capabilities = getattr(self, "_dsh_native_capabilities", None)
            if not isinstance(capabilities, dict):
                raise DshNativeRuntimeUnavailableError()
            preset_catalog = _dsh_native_stable_preset_catalog(capabilities)
            preset_catalog_digest = digest(preset_catalog)
            metadata["dsh_preset_ids"] = list(_DSH_NATIVE_PRESET_IDS)
            metadata["dsh_preset_catalog_digest"] = preset_catalog_digest
            metadata["dsh_runtime_capabilities_digest"] = digest(capabilities)
            metadata["dsh_first_call_verified"] = False
            metadata["dsh_live_agent_service_ready"] = True
            policy_model_digest = digest(
                {
                    "protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                    "role": "strategy",
                    "model": policy_model_id,
                    "preset_catalog_digest": preset_catalog_digest,
                }
            )
            judge_model_digest = digest(
                {
                    "protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                    "role": "review",
                    "model": judge_model_id,
                    "preset_catalog_digest": preset_catalog_digest,
                }
            )
            policy_binding_source = "dsh_native_frozen_route"
            judge_binding_source = "dsh_native_frozen_route"
            metadata.setdefault(
                "seed_genome_template_id",
                "toy-default@1" if toy_domain else "greenhouse-default@1",
            )
            metadata.setdefault(
                "dataset_snapshot_set_digest",
                digest(
                    {
                        "dataset_id": dataset_id,
                        "episode_id": metadata.get("episode_id"),
                        "dataset_digest": metadata.get("dataset_digest"),
                    }
                ),
            )
            metadata.setdefault(
                "data_protocol_digest",
                digest(
                    {
                        "protocol": "toy-validation-compat@1",
                        "dataset_digest": metadata.get("dataset_digest"),
                        "split_manifest_digest": metadata.get("split_manifest_digest"),
                    }
                ),
            )
            metadata.setdefault(
                "stage_policy_digest",
                digest({"policy": "dsh-native-four-stage-evolution@1"}),
            )
            metadata.setdefault(
                "fitness_profile_digest",
                digest(
                    {
                        "policy": "lexicographic-validity-skill-uq-efficiency@1",
                        "objective_profile": objective_profile,
                    }
                ),
            )
            metadata.setdefault(
                "security_kernel_digest",
                digest(
                    {
                        "policy": "dsh-standing-role-tools-fail-closed@1",
                        "preset_catalog_digest": preset_catalog_digest,
                    }
                ),
            )
            metadata.setdefault(
                "selection_reviewer_program_digest",
                digest({"program": "selection-reviewer@1"}),
            )
            metadata["required_capability_digest"] = digest(capabilities)
            metadata["resolved_policy_route_digest"] = policy_model_digest
            metadata["resolved_review_route_digest"] = judge_model_digest
            metadata["resolved_policy_route_config_digest"] = policy_model_digest
            metadata["resolved_review_route_config_digest"] = judge_model_digest
            metadata["preset_content_digest"] = preset_catalog_digest
            metadata["standing_tool_surface_digest"] = digest(
                [
                    {
                        "preset_id": item.get("preset_id"),
                        "tool_surface_verified": item.get("tool_surface_verified"),
                    }
                    for item in preset_catalog
                ]
            )
            metadata.setdefault(
                "evaluation_cohort_digest",
                digest(
                    {
                        "dataset": metadata.get("dataset_digest"),
                        "protocol": metadata.get("data_protocol_digest"),
                        "stage": "model_selection",
                    }
                ),
            )
        else:
            policy_model_digest, policy_binding_source = self._current_model_binding(
                policy_model_id
            )
            judge_model_digest, judge_binding_source = self._current_model_binding(
                judge_model_id
            )
        for field, computed, label in (
            ("policy_model_digest", policy_model_digest, "候选生成模型"),
            ("judge_model_digest", judge_model_digest, "独立评审模型"),
        ):
            supplied = metadata.get(field)
            if supplied is not None and supplied != computed:
                raise ValueError(f"任务清单中的{label}校验值与服务端配置不一致")
        autonomous_plan: dict[str, Any] = {}
        evaluator_catalog = [
            dict(item)
            for item in self.server.evaluators.catalog()
            if dataset_id in item.get("dataset_ids", [])
        ]
        selected_evaluator = next(
            item for item in evaluator_catalog if item.get("id") == evaluator_id
        )
        compatible_predictor_ids = set(
            selected_evaluator.get("prediction_model_ids", [])
        )
        schema_resolver = getattr(
            self.server.strategy_router, "parameter_schemas_for_task", None
        )
        semantics_resolver = getattr(
            self.server.strategy_router, "parameter_semantics_for_task", None
        )
        compatible_predictors = []
        for raw_predictor in self.server.evaluators.predictor_catalog():
            predictor_id = str(raw_predictor.get("id") or "")
            if (
                dataset_id not in raw_predictor.get("dataset_ids", [])
                or predictor_id not in compatible_predictor_ids
            ):
                continue
            predictor = dict(raw_predictor)
            predictor_task_data = manifest.to_dict()
            predictor_task_data["visible_datasets"] = [dataset_id]
            predictor_task_data["metadata"] = {
                **metadata,
                "prediction_model_id": predictor_id,
            }
            predictor_task = TaskManifest.from_dict(predictor_task_data)
            if callable(schema_resolver):
                predictor["parameter_schemas"] = schema_resolver(predictor_task)
            if callable(semantics_resolver):
                predictor["parameter_semantics"] = semantics_resolver(predictor_task)
            compatible_predictors.append(predictor)
        runtime_component_catalog = {
            "prediction_models": compatible_predictors,
            "evaluators": evaluator_catalog,
            "selected_prediction_model_id": prediction_model_id,
            "selected_evaluator_id": evaluator_id,
            "objective_profile": objective_profile,
        }
        if autonomous_mode and not defer_remote_plan:
            planning_metadata = {
                **metadata,
                "strategy_id": strategy_id,
                "strategy_digest": strategy_digest,
                "evaluator_id": evaluator_id,
                "evaluator_digest": evaluator_digest,
                "objective_profile": objective_profile,
                "prediction_model_id": prediction_model_id,
                "prediction_model_digest": prediction_model_digest,
                "runtime_component_catalog": runtime_component_catalog,
            }
            planning_data = manifest.to_dict()
            planning_data["visible_datasets"] = [dataset_id]
            planning_data["metadata"] = planning_metadata
            planning_manifest = TaskManifest.from_dict(planning_data)
            autonomous_plan = self._autonomous_plan_for_task(
                planning_manifest,
                strategy_model_id,
                strict_remote_plan=strict_remote_plan,
            )
            selected_predictor = autonomous_plan.get("prediction_model")
            selected_predictor_id = (
                selected_predictor.get("id")
                if isinstance(selected_predictor, dict)
                else None
            )
            if isinstance(selected_predictor_id, str) and selected_predictor_id:
                try:
                    self.server.evaluators.validate_binding(
                        dataset_id, evaluator_id, selected_predictor_id
                    )
                except (TypeError, ValueError):
                    autonomous_plan["prediction_model_adoption"] = {
                        "status": "research_only",
                        "requested_id": selected_predictor_id,
                        "adopted_id": prediction_model_id,
                        "reason": "模型提出的预测模型未在当前数据集评测注册表中登记",
                    }
                else:
                    prediction_model_id = selected_predictor_id
                    prediction_model_digest = (
                        self.server.evaluators.predictor_configuration_digest(
                            prediction_model_id
                        )
                    )
                    autonomous_plan["prediction_model_adoption"] = {
                        "status": "adopted",
                        "requested_id": selected_predictor_id,
                        "adopted_id": selected_predictor_id,
                    }
            else:
                autonomous_plan.setdefault(
                    "prediction_model_adoption",
                    {
                        "status": "host_default",
                        "adopted_id": prediction_model_id,
                    },
                )
        if autonomous_mode and defer_remote_plan:
            metadata["autonomous_plan_execution"] = "deferred_to_first_generation"
        sample_concurrency = metadata.get("sample_concurrency")
        samples_per_update = metadata.get("samples_per_update")
        sample_agent_batch_size = metadata.get("sample_agent_batch_size")
        if autonomous_mode and not toy_domain:
            if sample_concurrency is None:
                sample_concurrency = _DEFAULT_REAL_SAMPLE_CONCURRENCY
            if (
                isinstance(sample_concurrency, bool)
                or not isinstance(sample_concurrency, int)
                or not 1 <= sample_concurrency <= _MAX_REAL_SAMPLE_CONCURRENCY
            ):
                raise ValueError(
                    "sample_concurrency must be an integer between 1 and "
                    f"{_MAX_REAL_SAMPLE_CONCURRENCY}"
                )
            if samples_per_update is None:
                samples_per_update = _DEFAULT_SAMPLES_PER_UPDATE
            minimum_samples_per_update = (
                self.server.evaluators.minimum_samples_per_update(evaluator_id)
            )
            if (
                isinstance(samples_per_update, bool)
                or not isinstance(samples_per_update, int)
                or not (
                    minimum_samples_per_update
                    <= samples_per_update
                    <= _MAX_SAMPLES_PER_UPDATE
                )
            ):
                raise ValueError(
                    "samples_per_update must be an integer between "
                    f"{minimum_samples_per_update} and {_MAX_SAMPLES_PER_UPDATE} "
                    "so every evaluator target and horizon can be represented"
                )
            if sample_agent_batch_size is None:
                sample_agent_batch_size = _DEFAULT_SAMPLE_AGENT_BATCH_SIZE
            if (
                isinstance(sample_agent_batch_size, bool)
                or not isinstance(sample_agent_batch_size, int)
                or not 1 <= sample_agent_batch_size <= _MAX_SAMPLE_AGENT_BATCH_SIZE
            ):
                raise ValueError(
                    "sample_agent_batch_size must be an integer between 1 and "
                    f"{_MAX_SAMPLE_AGENT_BATCH_SIZE}"
                )
        else:
            sample_concurrency = None
            samples_per_update = None
            # Preserve the pre-existing host/toy manifest value. It is not
            # used unless sample_agent_mode selects the gateway adapter.
            sample_agent_batch_size = 128
        runtime_component_catalog["selected_prediction_model_id"] = (
            prediction_model_id
        )
        metadata.update(
            {
                "strategy_id": strategy_id,
                "strategy_digest": strategy_digest,
                "evaluator_id": evaluator_id,
                "evaluator_digest": evaluator_digest,
                # Freeze the objective definition with the run so the score
                # cannot be misread as a free-form natural-language goal.
                "objective_profile": objective_profile,
                "prediction_model_id": prediction_model_id,
                "prediction_model_digest": prediction_model_digest,
                "policy_model_id": policy_model_id,
                "judge_model_id": judge_model_id,
                "strategy_model_id": strategy_model_id,
                "review_model_id": judge_model_id,
                "autonomous_mode": autonomous_mode,
                # Existing runs without this marker keep their historical host
                # execution semantics. New real autonomous runs freeze the
                # gateway-backed microbatch protocol into the task digest.
                "sample_agent_mode": (
                    "dsh_native_workflow"
                    if native_protocol
                    else "gateway_microbatch"
                    if autonomous_mode and not toy_domain
                    else "host_feedback_state_machine"
                ),
                "sample_agent_batch_size": sample_agent_batch_size,
                # Causal origin waves are independent schedules. New real
                # runs use eight bounded workers by default; an explicit
                # lower value remains part of the immutable manifest. Runs
                # created before this field continue to restore their legacy
                # registry default of four workers.
                "sample_concurrency": sample_concurrency,
                # The model fit still consumes all training_fit rows. This
                # bounded, rotating window applies only to per-sample agent
                # execution over training_feedback.
                "samples_per_update": samples_per_update,
                # These limits are part of the task manifest rather than a
                # process-wide gateway default, so later configuration changes
                # cannot silently change an already-created real run.
                "sample_operation_max_tokens": (
                    dict(_DEFAULT_SAMPLE_OPERATION_MAX_TOKENS)
                    if autonomous_mode and not toy_domain and not native_protocol
                    else None
                ),
                # Every new real sample receives independent remote review in
                # addition to the host constraint critic. Persisting the policy
                # prevents restart-time drift; older runs retain their freeze.
                "sample_remote_critic_policy": (
                    dict(_DEFAULT_SAMPLE_REMOTE_CRITIC_POLICY)
                    if autonomous_mode and not toy_domain
                    else None
                ),
                # Repeated origin-level features are losslessly factored into a
                # content-addressed context only for newly created real runs.
                "sample_planner_prompt_profile": (
                    dict(_DEFAULT_SAMPLE_PLANNER_PROMPT_PROFILE)
                    if autonomous_mode and not toy_domain
                    else None
                ),
                "sample_truncation_retry_policy": (
                    dict(_DEFAULT_SAMPLE_TRUNCATION_RETRY_POLICY)
                    if autonomous_mode and not toy_domain and not native_protocol
                    else None
                ),
                "model_selection_policy": (
                    "model_research_and_runtime_compile@1"
                    if autonomous_mode
                    else metadata.get("model_selection_policy")
                ),
                "model_workflow": metadata.get(
                    "model_workflow",
                    "research_compile_evolve@1"
                    if autonomous_mode
                    else "legacy_component_search@1",
                ),
                "research_domain": metadata.get(
                    "research_domain", manifest.domain_pack
                ),
                "autonomous_plan": autonomous_plan if autonomous_mode else None,
                "autonomous_plan_digest": digest(autonomous_plan)
                if autonomous_mode
                else None,
                "runtime_component_catalog": runtime_component_catalog,
                "policy_model_digest": policy_model_digest,
                "judge_model_digest": judge_model_digest,
                "policy_model_binding_source": policy_binding_source,
                "judge_model_binding_source": judge_binding_source,
            }
        )
        data = manifest.to_dict()
        data["visible_datasets"] = [dataset_id]
        data["metadata"] = metadata
        if native_protocol:
            native_budget = dict(data["budget"])
            native_budget.pop("token_limit", None)
            native_budget.pop("token_reservation_per_wave", None)
            data["budget"] = native_budget
            for legacy_token_field in (
                "sample_operation_max_tokens",
                "sample_truncation_retry_policy",
                "sample_token_budget_policy",
                "token_budget_scope",
                "run_wide_accounting_complete",
                "sample_token_budget_defaulted",
            ):
                metadata.pop(legacy_token_field, None)
            data["metadata"] = metadata
        elif autonomous_mode and not toy_domain:
            budget = dict(data["budget"])
            token_limit = int(budget.get("token_limit", 0))
            if token_limit <= 0:
                token_limit = _DEFAULT_REAL_RUN_TOKEN_LIMIT
            # The server owns this per-logical-call admission cap. A client
            # cannot lower it independently of the frozen output limits and
            # gateway retry policy. Recursive split and critic calls each pass
            # through a fresh admission check.
            budget["token_limit"] = token_limit
            budget["token_reservation_per_wave"] = min(
                token_limit, _REAL_RUN_TOKEN_RESERVATION_PER_CALL
            )
            data["budget"] = budget
            metadata["sample_token_budget_policy"] = _SAMPLE_TOKEN_BUDGET_POLICY
            # This hard limit currently governs only sample-agent gateway
            # calls. Research planning, candidate proposal, and judge calls do
            # not yet emit receipts into the same admission ledger.
            metadata["token_budget_scope"] = _SAMPLE_TOKEN_BUDGET_SCOPE
            metadata["run_wide_accounting_complete"] = False
            metadata["sample_token_budget_defaulted"] = (
                int(manifest.budget.get("token_limit", 0)) <= 0
            )
            data["metadata"] = metadata
        return TaskManifest.from_dict(data)

    def _autonomous_plan_for_task(
        self,
        manifest: TaskManifest,
        strategy_model_id: str,
        *,
        strict_remote_plan: bool = False,
    ) -> dict[str, Any]:
        """Collect one replayable model research plan at run creation time."""

        run = Run(
            run_id=f"plan:{manifest.task_id}",
            task_id=manifest.task_id,
            task_manifest_digest=manifest.digest,
        )
        planner = getattr(self.server.strategy_router, "research_plan", None)
        if not callable(planner):
            if strict_remote_plan:
                raise ValueError(
                    "连续自动进化无法获取策略模型研究计划：研究计划接口不可用；"
                    "请修复策略模型 API，或显式设置 allow_host_fallback=true 后重试"
                )
            return {
                "status": "host_fallback",
                "fallback_diagnostics": {
                    "stage": "research_plan",
                    "fallback_applied": True,
                    "reason": "planner_not_available",
                },
                "team": {
                    "id": "host-validated-team@1",
                    "name": "宿主验证团队",
                    "roles": ["预测建模", "科学评测", "进化搜索"],
                },
                "strategy": {"id": "autonomous_model@1", "name": "模型自主调研与进化"},
                "research": [],
            }
        schema_resolver = getattr(
            self.server.strategy_router, "parameter_schemas_for_task", None
        )
        parameter_schemas = (
            schema_resolver(manifest) if callable(schema_resolver) else {}
        )
        try:
            plan = planner(
                strategy_model_id,
                run=run,
                task=manifest,
                parameter_schemas=parameter_schemas,
            )
        except Exception as exc:  # model availability must not corrupt a run
            if strict_remote_plan:
                raise ValueError(
                    "连续自动进化无法获取策略模型研究计划："
                    f"{strategy_model_id} 调用失败（{type(exc).__name__}）；"
                    "请修复策略模型 API，或显式设置 allow_host_fallback=true 后重试"
                ) from exc
            return {
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "fallback_diagnostics": {
                    "stage": "research_plan",
                    "fallback_applied": True,
                    "reason": "planner_error",
                    "error_type": type(exc).__name__,
                },
                "team": {"name": "待模型确认的自主团队", "roles": []},
                "strategy": {"id": "autonomous_model@1", "name": "模型自主调研与进化"},
                "research": [],
            }
        if not isinstance(plan, dict):
            raise TypeError("模型研究计划必须是 JSON 对象")
        # A creation-time planner uses a temporary ``plan:`` run and therefore
        # cannot create a durable, answerable consultation. The generation
        # research path materializes the same optional field atomically with
        # its real ResearchIteration; never freeze this one-shot side channel
        # into TaskManifest.autonomous_plan.
        plan.pop("expert_consultation", None)
        # Ensure a stable status is present for the UI and audit consumers.
        plan.setdefault("status", "model_generated")
        plan_status = str(plan.get("status", "")).casefold()
        if strict_remote_plan and plan_status in {
            "host_fallback",
            "unavailable",
        }:
            raise ValueError(
                "连续自动进化未获得远程研究计划："
                f"{strategy_model_id} 返回状态 {plan['status']}；"
                "请修复策略模型 API，或显式设置 allow_host_fallback=true 后重试"
            )
        if plan_status in {"host_fallback", "unavailable"}:
            plan.setdefault(
                "fallback_diagnostics",
                {
                    "stage": "research_plan",
                    "fallback_applied": True,
                    "reason": f"planner_status_{plan_status}",
                },
            )
        return plan

    def _current_model_binding(self, model_id: str) -> tuple[str, str]:
        builtin_digest = builtin_model_configuration_digest(model_id)
        if builtin_digest is not None:
            return builtin_digest, "builtin_implementation"
        try:
            return (
                self.server.model_gateway.configuration_digest(model_id),
                "server_gateway_configuration",
            )
        except KeyError:
            raise ValueError(f"服务端已找不到运行绑定的模型：{model_id}") from None

    @staticmethod
    def _validate_frozen_model_aliases(metadata: dict[str, Any]) -> None:
        """Reject conflicting canonical and legacy model role aliases.

        New manifests use ``strategy_model_id``/``review_model_id`` while
        older projections retain ``policy_model_id``/``judge_model_id``.
        Both names are persisted for compatibility, so a tampered or
        partially migrated manifest must not let the proposal path read one
        model while the restore checks authenticate another.
        """

        for canonical, legacy, label in (
            ("strategy_model_id", "policy_model_id", "策略模型"),
            ("review_model_id", "judge_model_id", "评审模型"),
        ):
            canonical_value = metadata.get(canonical)
            legacy_value = metadata.get(legacy)
            if (
                canonical_value is not None
                and legacy_value is not None
                and str(canonical_value).strip() != str(legacy_value).strip()
            ):
                raise ValueError(
                    f"冻结任务中的{label}字段冲突：{canonical} 与 {legacy} 必须指向同一模型"
                )

    @staticmethod
    def _remote_model_execution_configured(
        remote_model: dict[str, Any] | None,
    ) -> bool:
        """Return whether a redacted directory item is safe to call at runtime."""

        return bool(
            remote_model
            and remote_model.get("configured") is True
            and remote_model.get("directory_available") is True
            and remote_model.get("execution_available") is True
            and remote_model.get("credential_configured") is True
        )

    @staticmethod
    def _validate_frozen_remote_model(
        model_id: str,
        remote_model: dict[str, Any] | None,
        *,
        role: str,
        label: str,
        check_credentials: bool = True,
    ) -> None:
        """Validate a remote role binding during post-restart recovery."""

        if remote_model is None:
            raise ValueError(f"{label}已从当前服务模型目录移除，旧运行已拒绝继续：{model_id}")
        roles = remote_model.get("roles", [])
        if role not in roles:
            raise ValueError(f"冻结的{label}缺少 {role} 权限，旧运行已拒绝继续：{model_id}")
        if (
            remote_model.get("configured") is not True
            or remote_model.get("directory_available") is not True
            or remote_model.get("execution_available") is not True
        ):
            raise ValueError(f"冻结的{label}后端调用配置不可执行，旧运行已拒绝继续：{model_id}")
        if not check_credentials:
            return
        if not remote_model.get("credential_configured"):
            raise ValueError(f"冻结的{label}尚未配置认证凭据，旧运行已拒绝继续：{model_id}")
        if not EvolutionRequestHandler._remote_model_execution_configured(remote_model):
            raise ValueError(f"冻结的{label}后端调用配置不可执行，旧运行已拒绝继续：{model_id}")

    @classmethod
    def _validate_frozen_runtime_bindings_for_server(
        cls,
        server: EvolutionHTTPServer,
        task: TaskManifest,
        *,
        run_id: str | None = None,
    ) -> None:
        """Validate a frozen task without requiring an HTTP handler instance."""

        dataset_id = task.dataset
        if dataset_id is None:
            raise ValueError("运行缺少冻结数据集")
        metadata = task.metadata
        dataset_digest = metadata.get("dataset_digest")
        split_digest = metadata.get("split_manifest_digest")
        if not isinstance(dataset_digest, str) or not dataset_digest.strip():
            raise ValueError("旧运行缺少冻结的数据集快照校验值，已拒绝继续")
        if not isinstance(split_digest, str) or not split_digest.strip():
            raise ValueError("旧运行缺少冻结的时间分区快照校验值，已拒绝继续")
        episode_id = metadata.get("episode_id")
        series = server.datasets.series(
            dataset_id,
            str(episode_id) if episode_id is not None else None,
            expected_dataset_digest=dataset_digest,
            expected_split_manifest_digest=split_digest,
        )
        if series.digest != dataset_digest:
            raise FrozenRuntimeBindingDriftError("数据集快照")
        if series.split_manifest_digest_sha256 != split_digest:
            raise FrozenRuntimeBindingDriftError("时间分区快照")

        for id_field, digest_field, label, resolver in (
            (
                "strategy_id",
                "strategy_digest",
                "进化策略",
                server.strategy_router.configuration_digest,
            ),
            (
                "evaluator_id",
                "evaluator_digest",
                "评测器",
                server.evaluators.evaluator_configuration_digest,
            ),
            (
                "prediction_model_id",
                "prediction_model_digest",
                "预测模型",
                server.evaluators.predictor_configuration_digest,
            ),
        ):
            runtime_id = metadata.get(id_field)
            expected_digest = metadata.get(digest_field)
            if not isinstance(runtime_id, str) or not runtime_id.strip():
                raise ValueError(f"旧运行缺少冻结的{label}标识，已拒绝继续")
            if not isinstance(expected_digest, str) or not expected_digest.strip():
                raise ValueError(f"旧运行缺少冻结的{label}校验值，已拒绝继续")
            try:
                current_digest = resolver(runtime_id)
            except (KeyError, ValueError) as exc:
                raise FrozenRuntimeBindingDriftError(f"{label}实现") from exc
            if expected_digest != current_digest:
                raise FrozenRuntimeBindingDriftError(f"{label}实现")

        if metadata.get("execution_protocol") == DSH_NATIVE_EXECUTION_PROTOCOL:
            runtime = server.dsh_native_runtime
            if runtime is None:
                raise DshNativeRuntimeUnavailableError()
            capabilities = runtime.capabilities()
            runtime.require_capabilities(
                capabilities,
                _DSH_NATIVE_PRESET_IDS,
                require_live=False,
            )
            preset_catalog_digest = digest(
                _dsh_native_stable_preset_catalog(capabilities)
            )
            if metadata.get("dsh_preset_catalog_digest") != preset_catalog_digest:
                raise FrozenRuntimeBindingDriftError("DSH preset capability")
            cls._validate_frozen_model_aliases(metadata)
            for id_field, digest_field, role, label in (
                ("policy_model_id", "policy_model_digest", "strategy", "候选生成模型"),
                ("judge_model_id", "judge_model_digest", "review", "独立评审模型"),
            ):
                model_id = str(metadata.get(id_field) or "").strip()
                if not model_id:
                    raise ValueError(f"旧运行缺少冻结的{label}标识，已拒绝继续")
                current_digest = digest(
                    {
                        "protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                        "role": role,
                        "model": model_id,
                        "preset_catalog_digest": preset_catalog_digest,
                    }
                )
                if metadata.get(digest_field) != current_digest:
                    raise FrozenRuntimeBindingDriftError(f"{label} DSH 路由")
            if run_id is not None:
                target_run_id = str(run_id).strip()
                if not target_run_id:
                    raise ValueError("DSH runtime restore run_id must be non-empty")
                try:
                    runtime.status(target_run_id)
                except DshNativeRuntimeUnavailableError as exc:
                    if exc.status_code != HTTPStatus.NOT_FOUND:
                        raise
                    state = server.director.state(target_run_id)
                    latest_run_revision = (
                        int(state.events[-1].seq) if state.events else 0
                    )
                    runtime.create_run(
                        {
                            "run_id": target_run_id,
                            "run_state_revision": latest_run_revision,
                            "stage_attempt": 0,
                            "ledger_expected_revision": server.ledger.latest_seq(),
                            "idempotency_key": f"runtime-restore:{target_run_id}",
                            "binding": {
                                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                                "task_manifest_digest": task.digest,
                                "preset_catalog_digest": metadata.get(
                                    "dsh_preset_catalog_digest"
                                ),
                                "data_protocol_digest": metadata.get(
                                    "data_protocol_digest"
                                ),
                                "preset_content_digest": metadata.get(
                                    "preset_content_digest"
                                ),
                                "standing_tool_surface_digest": metadata.get(
                                    "standing_tool_surface_digest"
                                ),
                                "resolved_policy_route_config_digest": metadata.get(
                                    "resolved_policy_route_config_digest"
                                ),
                                "resolved_review_route_config_digest": metadata.get(
                                    "resolved_review_route_config_digest"
                                ),
                                "strategy_model_id": metadata.get("strategy_model_id"),
                                "review_model_id": metadata.get("review_model_id"),
                            },
                        }
                    )
                live_capabilities = runtime.capabilities()
                runtime.require_capabilities(
                    live_capabilities,
                    _DSH_NATIVE_PRESET_IDS,
                    require_live=True,
                )
            else:
                runtime.require_capabilities(
                    capabilities,
                    _DSH_NATIVE_PRESET_IDS,
                    require_live=True,
                )
            return

        remote_models = {
            str(item["model_id"]): item
            for item in server.model_gateway.catalog()
        }
        cls._validate_frozen_model_aliases(metadata)
        for id_field, digest_field, label, role in (
            ("policy_model_id", "policy_model_digest", "候选生成模型", "propose"),
            ("judge_model_id", "judge_model_digest", "独立评审模型", "judge"),
        ):
            model_id = str(metadata.get(id_field) or "")
            if not model_id:
                raise ValueError(f"旧运行缺少冻结的{label}标识，已拒绝继续")
            remote_model = remote_models.get(model_id)
            if builtin_model_configuration_digest(model_id) is None:
                cls._validate_frozen_remote_model(
                    model_id,
                    remote_model,
                    role=role,
                    label=label,
                    check_credentials=False,
                )
            builtin_digest = builtin_model_configuration_digest(model_id)
            if builtin_digest is not None:
                current_digest = builtin_digest
            else:
                try:
                    current_digest = server.model_gateway.configuration_digest(model_id)
                except KeyError as exc:
                    raise FrozenRuntimeBindingDriftError(f"{label}配置") from exc
            expected_digest = metadata.get(digest_field)
            if not isinstance(expected_digest, str) or not expected_digest.strip():
                if builtin_model_configuration_digest(model_id) is not None:
                    expected_digest = current_digest
                else:
                    raise ValueError(
                        f"旧运行缺少冻结的{label}配置校验值，远程模型已拒绝继续"
                    )
            if expected_digest != current_digest:
                raise FrozenRuntimeBindingDriftError(f"{label}配置")
            if builtin_model_configuration_digest(model_id) is None:
                cls._validate_frozen_remote_model(
                    model_id,
                    remote_model,
                    role=role,
                    label=label,
                )

    def _validate_frozen_runtime_bindings(
        self,
        task: TaskManifest,
        *,
        run_id: str | None = None,
    ) -> None:
        self.server.validate_frozen_runtime_bindings(task, run_id=run_id)

    def _action(self, run_id: str, body: dict[str, Any]) -> None:
        action = str(body.get("action", body.get("command", ""))).strip().lower()
        director = self.server.director
        valid_actions = {"start", "pause", "resume", "cancel", "complete", "advance", "step"}
        if action not in valid_actions:
            raise ValueError("action must be start, pause, resume, cancel, complete, or advance")
        cache_key = self._command_key(run_id, body)
        if self._serve_existing_command(cache_key, run_id, f"control:{action}", body):
            return
        state = director.state(run_id)
        active = getattr(self, "_active_command", None)
        target_status = {
            "start": "running",
            "pause": "paused",
            "resume": "running",
            "cancel": "cancelled",
            "complete": "completed",
        }.get(action)
        if (
            active is not None
            and target_status is not None
            and state.run.status.value == target_status
        ):
            payload = _state_payload(state)
            self._complete_command(cache_key, payload)
            self._send(HTTPStatus.OK, payload)
            return
        if action in ("advance", "step"):
            self._validate_advance_request(run_id, body)
        else:
            self._validate_control_request(run_id, action)
        cached = self._claim_command(cache_key, run_id, f"control:{action}", body)
        if cached is not None:
            self._send(HTTPStatus.OK, cached)
            return
        native_protocol = (
            state.task_manifest.metadata.get("execution_protocol")
            == DSH_NATIVE_EXECUTION_PROTOCOL
        )
        native_request = {
            "run_id": run_id,
            "run_state_revision": state.events[-1].seq,
            "stage_attempt": 0,
            "ledger_expected_revision": self.server.ledger.latest_seq(),
            "idempotency_key": str(
                body.get("idempotency_key") or f"{action}:{run_id}:{state.events[-1].seq}"
            ),
        }
        if native_protocol and action in {"pause", "cancel"}:
            self.server.dsh_tools.close_run_admissions(run_id)
            if action == "pause":
                self.server.dsh_native_runtime.pause(native_request)
            else:
                self.server.dsh_native_runtime.cancel(native_request)
        elif native_protocol and action == "resume":
            self.server.dsh_native_runtime.resume(native_request)
        if action == "start":
            director.start_run(run_id)
        elif action == "pause":
            # Preserve an operator-supplied pause cause in the append-only
            # event stream so the projection can distinguish manual pauses
            # from budget and gateway back-pressure pauses.
            pause_reason = body.get("reason")
            pause_code = body.get("code")
            if pause_reason is not None and not isinstance(pause_reason, str):
                raise TypeError("pause reason must be a string")
            if pause_code is not None and not isinstance(pause_code, str):
                raise TypeError("pause code must be a string")
            director.pause_run(
                run_id,
                reason=pause_reason,
                code=pause_code,
            )
        elif action == "resume":
            director.resume_run(run_id)
            if native_protocol:
                self.server.dsh_tools.open_run_admissions(run_id)
        elif action == "cancel":
            director.cancel_run(run_id, str(body.get("reason", "cancelled by user")))
        elif action == "complete":
            director.complete_run(run_id)
        elif action in ("advance", "step"):
            state = self._advance_run(run_id, body)
            payload = _state_payload(state)
            self._complete_command(cache_key, payload)
            self._schedule_auto_progress(state)
            self._send(HTTPStatus.OK, payload)
            return
        resulting_state = director.state(run_id)
        payload = _state_payload(resulting_state)
        self._complete_command(cache_key, payload)
        if action in {"start", "resume"}:
            # A paused autonomous run resumes at the next generation boundary;
            # the manager deduplicates this with any worker that is already
            # finishing the previous boundary.
            self._schedule_auto_progress(resulting_state)
        self._send(HTTPStatus.OK, payload)

    def _record_intervention(self, run_id: str, body: dict[str, Any]) -> None:
        cache_key = self._command_key(run_id, body)
        if self._serve_existing_command(
            cache_key, run_id, "record_intervention", body
        ):
            return
        state = self.server.director.state(run_id)
        _assert_http_scope(state)
        if state.run.status.value != "paused":
            raise RuntimeError("提交人工意见前必须先暂停运行")
        allowed_fields = {
            "kind",
            "message",
            "created_by",
            "parameter_overrides",
            "target_candidate_id",
            "idempotency_key",
        }
        unknown = set(body) - allowed_fields
        if unknown:
            raise ValueError("人工意见包含未知字段：" + ", ".join(sorted(unknown)))
        try:
            kind = InterventionKind(str(body.get("kind", "guidance")))
        except ValueError:
            raise ValueError("未知人工干预类型") from None
        message = str(body.get("message", "")).strip()
        created_by = str(body.get("created_by") or "本地研究员").strip()
        if not message or len(message) > 2000:
            raise ValueError("意见说明必须为 1 至 2000 个字符")
        if not created_by or len(created_by) > 120:
            raise ValueError("提交人必须为 1 至 120 个字符")
        overrides = body.get("parameter_overrides", {})
        if not isinstance(overrides, dict):
            raise TypeError("parameter_overrides 必须是对象")
        if kind is not InterventionKind.PARAMETER_OVERRIDE and overrides:
            raise ValueError("只有参数覆盖类型可以提交参数覆盖值")
        if kind is InterventionKind.PARAMETER_OVERRIDE:
            if not overrides:
                raise ValueError("参数覆盖类型至少需要一个参数")
            self.server.evaluators.validate_parameter_overrides(
                state.task_manifest, overrides
            )
        target_candidate_id = body.get("target_candidate_id")
        if target_candidate_id is not None:
            target_candidate_id = str(target_candidate_id).strip() or None
        if kind is InterventionKind.PARENT_SELECTION and target_candidate_id is None:
            raise ValueError("指定父方案时必须选择一个候选方案")
        if target_candidate_id is not None:
            state.candidate(target_candidate_id)
        active = getattr(self, "_active_command", None)
        if active is not None:
            receipt: CommandReceipt = active["receipt"]
            for event in self.server.ledger.events(
                run_id, after_seq=receipt.start_seq
            ):
                if event.kind != "HumanInterventionRecorded":
                    continue
                recorded = event.payload.get("intervention", {})
                if (
                    recorded.get("kind") == kind.value
                    and recorded.get("message") == message
                    and recorded.get("created_by") == created_by
                    and recorded.get("parameter_overrides", {}) == overrides
                    and recorded.get("target_candidate_id") == target_candidate_id
                ):
                    payload = _state_payload(self.server.director.state(run_id))
                    self._complete_command(cache_key, payload)
                    self._send(HTTPStatus.OK, payload)
                    return
        cached = self._claim_command(
            cache_key, run_id, "record_intervention", body
        )
        if cached is not None:
            self._send(HTTPStatus.OK, cached)
            return
        intervention = HumanIntervention(
            intervention_id=(
                f"intervention:{digest({'command_key': cache_key})[:24]}"
                if cache_key is not None
                else f"intervention:{uuid4()}"
            ),
            run_id=run_id,
            kind=kind,
            message=message,
            created_by=created_by,
            parameter_overrides=overrides,
            target_candidate_id=target_candidate_id,
        )
        self.server.director.record_intervention(intervention)
        payload = _state_payload(self.server.director.state(run_id))
        self._complete_command(cache_key, payload)
        self._send(HTTPStatus.CREATED, payload)

    def _answer_expert_consultation(
        self,
        run_id: str,
        consultation_id: str,
        body: dict[str, Any],
    ) -> None:
        """Record a non-blocking expert answer without changing run control."""

        consultation_id = str(consultation_id).strip()
        if not consultation_id:
            raise ValueError("consultation_id must be non-empty")
        allowed_fields = {
            "answer",
            "selected_option",
            "answered_by",
            "idempotency_key",
        }
        unknown = set(body) - allowed_fields
        if unknown:
            raise ValueError(
                "专家答复包含未知字段：" + ", ".join(sorted(unknown))
            )
        raw_answer = body.get("answer")
        raw_answered_by = body.get("answered_by")
        if not isinstance(raw_answer, str):
            raise TypeError("专家答复必须是字符串")
        if not isinstance(raw_answered_by, str):
            raise TypeError("答复人必须是字符串")
        answer_text = raw_answer.strip()
        answered_by = raw_answered_by.strip()
        if not answer_text or len(answer_text) > 4000:
            raise ValueError("专家答复必须为 1 至 4000 个字符")
        if not answered_by or len(answered_by) > 120:
            raise ValueError("答复人必须为 1 至 120 个字符")
        selected_option = body.get("selected_option")
        if selected_option is not None:
            if not isinstance(selected_option, str):
                raise TypeError("selected_option 必须是字符串或 null")
            selected_option = selected_option.strip()
            if not selected_option or len(selected_option) > 500:
                raise ValueError("所选选项必须为 1 至 500 个字符")

        command_body = {**body, "consultation_id": consultation_id}
        cache_key = self._command_key(run_id, body)
        if self._serve_existing_command(
            cache_key,
            run_id,
            "answer_expert_consultation",
            command_body,
        ):
            return

        # A pending command may already have appended its answer before the
        # HTTP response was interrupted. Reconcile that event before treating
        # the consultation's single-answer invariant as a conflict.
        active = getattr(self, "_active_command", None)
        if active is not None:
            receipt: CommandReceipt = active["receipt"]
            for event in self.server.ledger.events(
                run_id, after_seq=receipt.start_seq
            ):
                if event.kind != "ExpertConsultationAnswered":
                    continue
                recorded = event.payload.get("answer", {})
                if (
                    recorded.get("consultation_id") == consultation_id
                    and recorded.get("answer") == answer_text
                    and recorded.get("answered_by") == answered_by
                    and recorded.get("selected_option") == selected_option
                ):
                    payload = _state_payload(self.server.director.state(run_id))
                    self._complete_command(cache_key, payload)
                    self._send(HTTPStatus.OK, payload)
                    return

        state = self.server.director.state(run_id)
        _assert_http_scope(state)
        state.consultation(consultation_id)
        if state.run.status.value == "created":
            raise RuntimeError("尚未启动的运行不能接收专家答复")
        existing_answer = state.answer_for_consultation(consultation_id)
        if existing_answer is not None:
            raise RuntimeError("该专家咨询已有答复")

        cached = self._claim_command(
            cache_key,
            run_id,
            "answer_expert_consultation",
            command_body,
        )
        if cached is not None:
            self._send(HTTPStatus.OK, cached)
            return

        answer_id = (
            f"expert-answer:{digest({'command_key': cache_key})[:24]}"
            if cache_key is not None
            else f"expert-answer:{uuid4()}"
        )
        created_at = utc_now()
        terminal_statuses = {"completed", "cancelled", "failed"}
        last_conflict: ConcurrentRunMutationError | None = None
        for _attempt in range(8):
            current = self.server.director.state(run_id)
            _assert_http_scope(current)
            consultation = current.consultation(consultation_id)
            existing_answer = current.answer_for_consultation(consultation_id)
            if existing_answer is not None:
                if (
                    existing_answer.answer_id == answer_id
                    and existing_answer.answer == answer_text
                    and existing_answer.answered_by == answered_by
                    and existing_answer.selected_option == selected_option
                ):
                    break
                raise RuntimeError("该专家咨询已有答复")
            if current.run.status.value == "created":
                raise RuntimeError("尚未启动的运行不能接收专家答复")
            if selected_option is not None and selected_option not in consultation.options:
                raise ValueError("所选选项不在该咨询的可选项中")
            effective_generation = (
                None
                if current.run.status.value in terminal_statuses
                else max(consultation.generation + 1, current.run.generation + 1)
            )
            answer = ExpertConsultationAnswer(
                answer_id=answer_id,
                run_id=run_id,
                consultation_id=consultation_id,
                answer=answer_text,
                answered_by=answered_by,
                selected_option=selected_option,
                effective_generation=effective_generation,
                created_at=created_at,
            )
            try:
                self.server.director.answer_expert_consultation(answer)
            except ConcurrentRunMutationError as exc:
                last_conflict = exc
                continue
            break
        else:
            assert last_conflict is not None
            raise last_conflict

        payload = _state_payload(self.server.director.state(run_id))
        self._complete_command(cache_key, payload)
        self._send(HTTPStatus.CREATED, payload)
