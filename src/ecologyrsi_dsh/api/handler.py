"""Concrete HTTP server and request handler composition."""

from __future__ import annotations

from ..integrations.role_contracts import PRESET_IDS as _DSH_NATIVE_PRESET_IDS

from ..evolution.diversity import DIVERSITY_POLICY, enabled as diversity_enabled
from ..evolution.novelty import CANDIDATE_NOVELTY_POLICY, CANDIDATE_NOVELTY_POLICY_KEY

from ..evolution.schedule import ADAPTIVE_PROTOCOLS

from ..core.model_execution_policy import (
    NATIVE_SAMPLE_OPERATION_MAX_TOKENS,
    LOCAL_EDIT_CONTEXT_POLICY,
)

from ..data.adapters import dataset_adapter
from ..core.prediction_policy import (
    RUNTIME_PREDICTION_POLICY,
    RUNTIME_EVALUATOR_ID, CELLWISE_EVALUATOR_ID, TARGETED_SEARCH_POLICY, CONTRIBUTION_POLICY,
    BASELINE_REFERENCE_PREDICTOR_ID,
)

from ..application.runtime import initialize_runtime
from ..execution.ownership import RuntimeOwnerLease as _SidecarOwnerLease
from ..application.runtime_bindings import (
    dsh_revision_snapshot as _dsh_revision_snapshot,
    ValidatedCandidateIdentityCache as _ValidatedCandidateIdentityCache,
)

import fcntl
import json
import os
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4
from weakref import WeakValueDictionary

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
    Run,
    TaskManifest,
    digest,
    utc_now,
)
from ..core.protocols import is_strict_origin_protocol
from ..core.redaction import safe_error_code
from ..core.sample_results import MAX_SAMPLE_RESULTS_UNCOMPRESSED_BYTES
from ..core.state import (
    EVOLUTION_RUNTIME_SCHEMA_V3,
    validate_identity_binding,
)
from ..data.registry import DatasetRegistry
from ..evaluators.epoch_cohorts import CohortCapacityError, estimate_epoch_capacity
from ..evaluators.greenhouse_prediction import (
    COHORT_HISTORY_HOURS,
    MAX_EXOGENOUS_RIDGE_HISTORY_STEPS,
)
from ..evaluators.registry import (
    EvaluatorRegistry,
)
from ..evolution.schedule import (
    OPTIMIZATION_PROTOCOL,
    TRAINING_SCHEDULE_SCHEMA_VERSION,
    OptimizationSchedule,
    PREQUENTIAL_PERFORMANCE_POLICY,
)
from ..evolution.genome import mutation_policy_contract
from ..evolution import joint_policy
from ..evolution.parameters import (
    PARAMETER_RULES,
    run_parameter_contract,
    validate_run_parameter,
)
from ..evolution.strategies import StrategyRouterDSHAdapter
from ..core.search_policy import SEARCH_GUARD_POLICY, PAIRED_EXECUTION_QUALIFICATION
from ..integrations.model_canary import run_preflight, require_model_preflight
from ..evaluators.agent_stability import REPLICA_COUNT, capacity_with_inference_replicas
from ..core.model_preflight import (
    AUDIT_EVENT,
    AUDIT_SCHEMA,
    AUDIT_METADATA_KEY,
    preflight_audit_required,
)
from ..integrations.dsh_native_runtime import (
    DSH_NATIVE_EXECUTION_PROTOCOL,
    DshNativeAgentRuntimeClient,
    configured_stage_timeout,
)
from ..integrations.dsh_structured_roles import DshStructuredRoleRuntime
from ..knowledge.autonomous_cycle import AUTONOMOUS_RESEARCH_PROTOCOL
from ..knowledge.program_registry import current_program_registry
from ..version import __version__
from .auto_progress import AutoProgressManager
from ..integrations.dsh_tools import DshStructuredResultPersistenceError, DshToolService
from .errors import public_error_payload
from .projection import _control_payload, _state_payload
from ..execution.sample_admission import (
    DEFAULT_SAMPLE_CONCURRENCY,
    RunSampleAdmission,
    validate_sample_concurrency,
)
from .shared import (
    _assert_http_scope,
    _assert_manifest_http_scope,
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
_HOST_RUNTIME_COMPATIBILITY = {
    "evolution_runtime_schema": EVOLUTION_RUNTIME_SCHEMA_V3,
    "generation_comparison_schema": "ecologyrsi-dsh.generation-comparison/1",
    "projection_schema": "ecologyrsi-dsh.execution-projection/2",
}
_DEFAULT_REAL_CANDIDATE_CONCURRENCY = PARAMETER_RULES["candidate_concurrency"]["default"]
_MAX_REAL_CANDIDATE_CONCURRENCY = PARAMETER_RULES["candidate_concurrency"]["maximum"]
_DSH_SIDECAR_PUBLIC_ERROR_CODES = frozenset(
    {
        "dsh_tool_admission_closed",
        "dsh_tool_authorization_failed",
        "dsh_prediction_binding_closed",
        "structured_role_operational_timeout",
        "structured_result_persistence_unavailable",
    }
)


def _dsh_sidecar_error(exc: BaseException) -> dict[str, str]:
    payload = public_error_payload(exc)
    if isinstance(exc, DshStructuredResultPersistenceError):
        # Do not make storage-driver or filesystem text observable to the DSH
        # process. The stable machine code is sufficient for its bounded retry.
        payload["error"] = "structured result persistence is temporarily unavailable"
    # Sidecar exposes only allow-listed provider codes; unknown implementation
    # exceptions remain the stable internal error contract.
    error_code = payload.get("error_code")
    if error_code not in _DSH_SIDECAR_PUBLIC_ERROR_CODES:
        # Preserve the established sidecar contract for unapproved provider
        # codes: detailed implementation failures are intentionally generic
        # and do not advertise a machine code to the DSH process.
        payload.pop("error_code", None)
        payload["error"] = "internal server error"
    return payload


def _record_sidecar_rejection(
    directory: Path, envelope: Any, exc: BaseException, *, stage: str | None = None
) -> None:
    """Keep the redacted rejection reason where only the operator can read it.

    The DSH process deliberately receives ``internal server error`` for every
    unapproved code, which also left the operator with no way to tell a
    contract violation from a Host defect.  ``public_error_payload`` has
    already applied the credential-redacting policy, so persisting that same
    text next to the ledger adds no disclosure while making a rejected
    structured result diagnosable.  Never raises: diagnostics must not turn a
    bounded rejection into a request failure.
    """

    try:
        identity = envelope.get("identity") if isinstance(envelope, Mapping) else None
        identity = identity if isinstance(identity, Mapping) else {}
        detail = str(public_error_payload(exc).get("error") or "")[:300]
        line = json.dumps({
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "stage": stage or str(identity.get("stage") or ""),
            "run_id": str(identity.get("run_id") or ""),
            "candidate_id": str(identity.get("candidate_id") or ""),
            "exception": type(exc).__name__,
            "detail": detail,
        }, ensure_ascii=False)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "sidecar-rejections.log").open("a", encoding="utf-8") as sink:
            sink.write(f"{line}\n")
    except (OSError, TypeError, ValueError):
        return


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
    "version": "uncertain_or_failure@1",
    "min_planner_confidence": 0.9,
}
_STRICT_SAMPLE_AGENT_PROTOCOL = "dsh-strict-origin-bundle@4"
_STRICT_SAMPLE_REMOTE_CRITIC_POLICY = {
    "version": "uncertain_or_failure@1",
    "min_planner_confidence": 0.5,
}
_STRICT_SAMPLE_REFLECTION_POLICY = "candidate_aggregate_post_score@1"

_DSH_NATIVE_STABLE_PRESET_FIELDS = (
    "preset_id",
    "declared",
    "preset_mountable",
    "tool_surface_verified",
    "route_resolvable",
    "content_digest",
)
_DSH_NATIVE_SEED_TEMPLATE_BY_PREDICTOR = {
    "greenhouse-exogenous-ridge@1": "greenhouse-exogenous-default@2",
    "greenhouse-baseline-aligned-ridge@1": "greenhouse-baseline-aligned-default@2",
    "greenhouse-baseline-aligned-ridge@2": "greenhouse-baseline-aligned-cellwise-default@2",
    "greenhouse-targetwise-ridge@1": "greenhouse-targetwise-default@2",
    "greenhouse-horizon-targetwise-ridge@1": "greenhouse-default@2",
}


def _dsh_native_stable_preset_catalog(
    capabilities: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the immutable capability subset used in frozen run identities."""

    by_id = {
        item.get("preset_id"): item
        for item in capabilities.get("presets", [])
        if isinstance(item, dict)
    }
    missing = [pid for pid in _DSH_NATIVE_PRESET_IDS if pid not in by_id]
    if missing:
        raise ValueError(
            f"required DSH preset(s) missing from capabilities: {missing!r}; "
            "check DSH version compatibility before creating a run"
        )
    return [
        {field: by_id[preset_id].get(field) for field in _DSH_NATIVE_STABLE_PRESET_FIELDS}
        for preset_id in _DSH_NATIVE_PRESET_IDS
    ]

_CREATE_REQUIRED_FIELDS = ('dataset_id', 'execution_protocol', 'strategy_model_id', 'review_model_id',)
_CREATE_OPTIONAL_FIELDS = (
    'episode_id', 'optimization_protocol', 'optimization_schedule', 'autonomous_mode',
    'rounds', 'budget', 'candidates_per_generation', 'max_candidates',
    'model_workflow', 'knowledge_online_enabled', 'candidate_concurrency', 'sample_concurrency',
    'sample_agent_batch_size', 'seed_policy', 'seed', 'requested_mode',
    'auto_advance', 'auto_progress', 'start', 'idempotency_key',
    'run_id', 'task_id', 'objective', 'prediction_selection_policy',
    'targeted_search_policy', 'prediction_model_id', 'evaluator_id', 'seed_genome_template_id',
    'search_guard_policy', 'require_model_contract_preflight', 'allow_host_fallback', 'research_domain',
    CANDIDATE_NOVELTY_POLICY_KEY,
)

PLUGIN_MANIFEST = {
    "id": "ecology_evolution",
    "plugin_id": "ecologyrsi.evolution",
    "display_name": "生态模型进化工作台",
    "description": "用于创建、控制和查看生态模型进化运行的轻量 DSH 工作台。",
    "version": __version__,
    "mode": "evolution",
    "workflow": "research_compile_evolve@1",
    "request_schema": {
        "required": list(_CREATE_REQUIRED_FIELDS),
        "optional": list(_CREATE_OPTIONAL_FIELDS),
        "internal_components": "仅接受当前原生联合进化合同；组件由模型提出并由宿主注册表编译",
    },
    "api_prefix": "/api",
    "recommended_dsh_proxy_base": "/api/ecology-evolution",
    "supported_bases": ["/api/ecology-evolution"],
    "capabilities": [
        "start_run",
        "read_projection",
        "control_run",
        "read_events",
        "read_sample_results",
        "read_training_data",
        "select_configured_model",
        "run.archive",
        "run.delete",
    ],
    "samples": "/runs/{run_id}/samples?candidate_id={candidate_id}&offset={offset}&limit={limit}",
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


def _host_runtime_build() -> dict[str, str]:
    """Return the auditable build plus the explicitly compatible schemas."""

    return {
        "package_version": __version__,
        **_HOST_RUNTIME_COMPATIBILITY,
    }


def _runtime_build_compatible(value: Any) -> bool:
    """Only the exact package/runtime identity may resume this release's ledger."""
    return isinstance(value, Mapping) and dict(value) == _host_runtime_build()


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


class _RunWorkReservation:
    """Transferable lifetime reservation for work executed on another thread.

    Unlike an RLock ownership, this reference may be released by the worker.
    It fences purge even between sample requests, without serializing unrelated
    runs or requiring the HTTP thread to wait for scientific work.
    """

    def __init__(self, lease: "_GenerationLease") -> None:
        self._lease = lease
        self._released = False

    def release(self) -> None:
        with self._lease._server._generation_locks_guard:
            if not self._released:
                self._lease._reservations -= 1
                self._released = True


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
    # The runtime accepts up to 128 concurrent sample callbacks in addition to
    # operator/UI requests.  TCPServer's tiny default listen backlog can drop
    # otherwise healthy bursts before a worker thread is created.
    request_queue_size = 256

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
        self.model_preflight_lock = threading.Lock()
        self._owner_lease = _SidecarOwnerLease(db_path)
        try:
            self.ledger = EventLedger(db_path)
            # Health must remain available while workers contend on the event
            # ledger. Freeze its static contract once during server startup.
            self.health_payload = {
                "ok": True,
                "package": "ecologyrsi-dsh",
                "package_version": __version__,
                "api_version": "v0.2",
                "plugin_version": PLUGIN_MANIFEST["version"],
                "schema_version": self.ledger.schema_version,
                "evaluation_partition": "training_feedback",
                "scientific_scope": "historical_replay_prediction_non_causal",
                "supported_evaluation_partitions": [
                    "training_feedback",
                    "validation",
                    "test",
                ],
                "scientific_scopes": [
                    "historical_replay_prediction_non_causal",
                ],
            }
            initialize_runtime(self, self.ledger)
            from ..application.read_cache import ReadSectionCache
            self.workspace_cache = ReadSectionCache()
            self._control_locks_guard = threading.Lock()
            self._control_locks: WeakValueDictionary[str, threading.RLock] = (
                WeakValueDictionary()
            )
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
            # Track the runtime instance that owns each asynchronous drain.  A
            # runtime process may be replaced while a previous drain is still
            # unwinding; the replacement must be allowed to restore/resume the
            # durable Host boundary without being blocked by that stale worker.
            self._native_control_inflight_lock = threading.RLock()
            self._native_control_inflight: dict[
                str, int | tuple[int, object]
            ] = {}
            super().__init__(address, EvolutionRequestHandler)
            # A process can stop after RunCreated was durably bound to its
            # create receipt but before the HTTP response was sealed.  At this
            # point the sidecar owner lease guarantees there is no prior HTTP
            # process still completing that command, so close only receipts
            # whose exact run can be replayed into a public projection.
            self._recover_bound_create_receipts()
            self._recover_terminal_command_receipts()
            # Continuous autonomous runs are progressed by a bounded worker pool.
            # Recovery is projection-driven and only picks up manifests that
            # explicitly opted into this mode.
            self.host_activity.start()
            self.auto_progress = AutoProgressManager(self)
            self.auto_progress.recover_native_quiescence()
            self.auto_progress.recover_running()
            from ..application.independent_evaluation import IndependentEvaluationService
            self.independent_evaluation = IndependentEvaluationService(self)
            self.independent_evaluation.recover_interrupted()
        except BaseException:
            if hasattr(self, "host_activity"):
                self.host_activity.close()
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

    def _recover_terminal_command_receipts(self) -> int:
        """Seal stale run mutations when a later terminal event proves success.

        A process can commit the requested cancel/advance transition and stop
        before sealing its HTTP receipt.  We only reconcile non-create
        receipts when this exact run has a terminal event newer than the
        receipt claim; receipts claimed after termination remain ambiguous.
        """

        proof_kinds = {
            "control:cancel": {"RunCancelled"},
            "control:complete": {"RunCompleted"},
        }
        recovered = 0
        for command_key in self.ledger.pending_command_keys():
            receipt = self.ledger.command_receipt(command_key)
            if (
                receipt is None
                or receipt.status != "pending"
                or receipt.command_kind == "create_run"
                or receipt.resource_run_id is None
                or receipt.command_kind not in proof_kinds
            ):
                continue
            try:
                state = self.director.state(receipt.resource_run_id)
                if state.run.status.value not in {"completed", "cancelled", "failed"}:
                    continue
                if not any(
                    event.kind in proof_kinds[receipt.command_kind]
                    and int(event.seq) > receipt.start_seq
                    for event in state.events
                ):
                    continue
                payload = _state_payload(state)
            except (KeyError, TypeError, ValueError, RuntimeError):
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

        return state.task_manifest.metadata.get("auto_progress") is True

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

    def control_lock(self, run_id: str) -> threading.RLock:
        """Return the process-local control lock for one durable run."""

        key = str(run_id).strip()
        if not key:
            raise ValueError("run_id must be non-empty")
        with self._control_locks_guard:
            lock = self._control_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._control_locks[key] = lock
            return lock

    def claim_native_control_inflight(
        self,
        run_id: str,
        marker: int | tuple[int, object],
        *,
        replace: bool = False,
    ) -> bool:
        """Atomically claim ownership of one asynchronous native drain."""

        key = str(run_id).strip()
        if not key:
            raise ValueError("run_id must be non-empty")
        with self._native_control_inflight_lock:
            if not replace and key in self._native_control_inflight:
                return False
            self._native_control_inflight[key] = marker
            return True

    def native_control_inflight_owned(
        self,
        run_id: str,
        marker: int | tuple[int, object],
    ) -> bool:
        """Return whether the exact marker still owns this run's drain."""

        key = str(run_id).strip()
        if not key:
            return False
        with self._native_control_inflight_lock:
            current = self._native_control_inflight.get(key)
            return current is marker or current == marker

    def native_control_inflight_for(self, run_id: str, runtime: Any) -> bool:
        """Atomically test whether the installed runtime owns a live drain."""

        key = str(run_id).strip()
        if not key:
            return False
        with self._native_control_inflight_lock:
            marker = self._native_control_inflight.get(key)
            runtime_id = (
                marker[0]
                if isinstance(marker, tuple) and len(marker) == 2
                else marker
            )
            return runtime_id == id(runtime)

    def clear_native_control_inflight(
        self,
        run_id: str,
        marker: int | tuple[int, object],
    ) -> bool:
        """Atomically clear a drain only while its exact owner remains current."""

        key = str(run_id).strip()
        if not key:
            return False
        with self._native_control_inflight_lock:
            current = self._native_control_inflight.get(key)
            if current is not marker and current != marker:
                return False
            self._native_control_inflight.pop(key, None)
            return True

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

    def reserve_run_work(self, run_id: str) -> _RunWorkReservation:
        """Reserve one run until asynchronous work and its cleanup finish."""

        key = str(run_id).strip()
        if not key:
            raise ValueError("run_id must be non-empty")
        with self._generation_locks_guard:
            lease = self._generation_locks.get(key)
            if lease is None or lease._retired:
                lease = _GenerationLease(self, key)
                self._generation_locks[key] = lease
            if lease._retiring or lease._retired:
                raise CommandInProgressError("run deletion is in progress")
            lease._reservations += 1
            return _RunWorkReservation(lease)

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

    def health_liveness_payload(self) -> dict[str, Any]:
        """Return a process-local liveness signal without touching dependencies."""

        return {
            "ok": True,
            "status": "live",
            "service": "ecologyrsi-dsh",
            "package_version": __version__,
        }

    def health_readiness_payload(self) -> tuple[bool, dict[str, Any]]:
        """Check the local dependencies required before accepting new work."""

        checks: dict[str, Any] = {
            "sqlite_integrity": "unknown",
            "plugin_manifest": False,
            "director": False,
            "dsh_runtime": "not_configured",
        }
        try:
            checks["sqlite_integrity"] = self.ledger.integrity_check()
        except Exception:
            checks["sqlite_integrity"] = "error"
        checks["plugin_manifest"] = (
            isinstance(PLUGIN_MANIFEST, Mapping)
            and PLUGIN_MANIFEST.get("version") == __version__
        )
        checks["director"] = getattr(self, "director", None) is not None
        if self.dsh_native_runtime is not None:
            checks["dsh_runtime"] = "configured"
        ready = (
            checks["sqlite_integrity"] == "ok"
            and checks["plugin_manifest"] is True
            and checks["director"] is True
        )
        return ready, {
            "ok": ready,
            "status": "ready" if ready else "not_ready",
            "service": "ecologyrsi-dsh",
            "package_version": __version__,
            "checks": checks,
        }

    def close(self) -> None:
        self.host_activity.close()
        independent_stopped = self.independent_evaluation.close()
        worker_stopped = self.auto_progress.close()
        # A urllib request cannot be interrupted safely. If shutdown reaches
        # its bounded join timeout, leave the ledger owned by that daemon
        # worker instead of closing SQLite underneath an in-flight generation.
        # A later close call will reclaim it after the worker exits.
        if worker_stopped and independent_stopped:
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

    def log_message(self, _format: str, *_args: Any) -> None:
        # Keep HTTP access logging quiet unless a caller overrides the logger.
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
        if path == ["health", "live"]:
            self._send(HTTPStatus.OK, self.server.health_liveness_payload())
            return
        if path == ["health", "ready"]:
            ready, payload = self.server.health_readiness_payload()
            self._send(HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE, payload)
            return
        if path == ["health"]:
            self._send(HTTPStatus.OK, self.server.health_payload)
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
        if len(path) == 2 and path[0] == "commands":
            self._call(lambda: self._command_status(path[1]))
            return
        if len(path) == 2 and path[0] == "runs":
            self._call(lambda: self._run_payload(path[1]))
            return
        if len(path) == 3 and path[0] == "runs" and path[2] == "independent-evaluation":
            self._call(lambda: self.server.independent_evaluation.status(path[1]))
            return
        if len(path) == 3 and path[0] == "runs" and path[2] == "events":
            self._call(lambda: self._events_payload(path[1]))
            return
        if len(path) == 3 and path[0] == "runs" and path[2] == "samples":
            self._call(lambda: self._sample_results_payload(path[1]))
            return
        self._send(HTTPStatus.NOT_FOUND, public_error_payload(
            FileNotFoundError("not found"), status=HTTPStatus.NOT_FOUND
        ))

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
                self._send(HTTPStatus.FORBIDDEN, _dsh_sidecar_error(exc))
            except (RuntimeError, TypeError, ValueError) as exc:
                self._send(HTTPStatus.CONFLICT, _dsh_sidecar_error(exc))
            return
        if raw_path == "/api/ecology-agent-sidecar/v1/child-failures":
            if not self._authorize_dsh_tool():
                return
            try:
                result = self.server.dsh_tools.record_child_failure(self._body())
                self._send(HTTPStatus.OK, result)
            except PermissionError as exc:
                self._send(HTTPStatus.FORBIDDEN, _dsh_sidecar_error(exc))
            except (RuntimeError, TypeError, ValueError) as exc:
                self._send(HTTPStatus.CONFLICT, _dsh_sidecar_error(exc))
            return
        if raw_path == "/api/ecology-agent-sidecar/v1/session-usage":
            if not self._authorize_dsh_tool():
                return
            try:
                self._send(HTTPStatus.OK, self.server.dsh_tools.record_session_usage(self._body()))
            except (RuntimeError, TypeError, ValueError) as exc:
                self._send(HTTPStatus.CONFLICT, _dsh_sidecar_error(exc))
            return
        if raw_path == "/api/ecology-agent-sidecar/v1/structured-results/validate":
            if not self._authorize_dsh_tool():
                return
            try:
                self._send(HTTPStatus.OK, self.server.dsh_tools.validate_structured(self._body()))
            except DshStructuredResultPersistenceError as exc:
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, _dsh_sidecar_error(exc))
            except PermissionError as exc:
                self._send(HTTPStatus.FORBIDDEN, _dsh_sidecar_error(exc))
            except (RuntimeError, TypeError, ValueError) as exc:
                # An operational/identity fault is never model INVALID_ARGS.
                # The only correctable response is the service's bounded 200
                # valid=false result; no rejected scientific event is written.
                self._send(HTTPStatus.CONFLICT, _dsh_sidecar_error(exc))
            return
        if raw_path == "/api/ecology-agent-sidecar/v1/structured-results":
            if not self._authorize_dsh_tool():
                return
            envelope: Any = None
            try:
                envelope = self._body()
                result = self.server.dsh_tools.accept_structured(envelope)
                self._send(HTTPStatus.OK, result)
            except DshStructuredResultPersistenceError as exc:
                # The DSH child already has one immutable structured result.
                # Tell its stage runner to retry that persistence boundary;
                # never present this as a bounded prediction/contract error.
                self._send(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    _dsh_sidecar_error(exc),
                )
            except PermissionError as exc:
                self._record_structured_rejection(envelope, exc)
                self._send(HTTPStatus.FORBIDDEN, _dsh_sidecar_error(exc))
            except (RuntimeError, TypeError, ValueError) as exc:
                self._record_structured_rejection(envelope, exc)
                self._send(HTTPStatus.CONFLICT, _dsh_sidecar_error(exc))
            return
        if raw_path == "/api/ecology-agent-sidecar/v1/retrievals/replay":
            if not self._authorize_dsh_tool():
                return
            try:
                result = self.server.dsh_tools.replay_retrieval(self._body())
                self._send(
                    HTTPStatus.OK,
                    {"found": result is not None, "result": result},
                )
            except PermissionError as exc:
                self._send(HTTPStatus.FORBIDDEN, public_error_payload(exc, status=HTTPStatus.FORBIDDEN))
            except (RuntimeError, TypeError, ValueError) as exc:
                self._send(HTTPStatus.CONFLICT, public_error_payload(exc, status=HTTPStatus.CONFLICT))
            return
        if raw_path == "/api/ecology-agent-sidecar/v1/retrievals/complete":
            if not self._authorize_dsh_tool():
                return
            try:
                result = self.server.dsh_tools.complete_retrieval(self._body())
                self._send(HTTPStatus.OK, result)
            except PermissionError as exc:
                self._send(HTTPStatus.FORBIDDEN, public_error_payload(exc, status=HTTPStatus.FORBIDDEN))
            except (RuntimeError, TypeError, ValueError) as exc:
                self._send(HTTPStatus.CONFLICT, public_error_payload(exc, status=HTTPStatus.CONFLICT))
            return
        if raw_path.startswith(tool_prefix):
            if not self._authorize_dsh_tool():
                return
            tool_name = raw_path[len(tool_prefix):]
            if not tool_name or "/" in tool_name:
                self._send(HTTPStatus.NOT_FOUND, public_error_payload(
                    FileNotFoundError("unknown DSH role tool"), status=HTTPStatus.NOT_FOUND
                ))
                return
            tool_envelope: Any = None
            try:
                tool_envelope = self._body()
                result = self.server.dsh_tools.execute(tool_name, tool_envelope)
                self._send(HTTPStatus.OK, result)
            except PermissionError as exc:
                self._record_structured_rejection(tool_envelope, exc, stage=f"tool:{tool_name}")
                self._send(HTTPStatus.FORBIDDEN, _dsh_sidecar_error(exc))
            except (RuntimeError, TypeError, ValueError) as exc:
                self._record_structured_rejection(tool_envelope, exc, stage=f"tool:{tool_name}")
                self._send(HTTPStatus.CONFLICT, _dsh_sidecar_error(exc))
            return
        if not self._authorize_api():
            return
        path = self._route()
        self._active_command: dict[str, Any] | None = None
        try:
            body = self._body()
            control_request = (
                len(path) == 3
                and path[0] == "runs"
                and path[2] in ("action", "control")
            )
            draining_control = control_request and str(
                body.get("action", body.get("command", ""))
            ).strip().lower() in {"pause", "cancel"}
            if path in (["model-preflight"], ["evolution-capacity"]):
                # Generation execution owns a per-run lease and already uses
                # thread-safe ledger writes.  Keeping the global mutation lock
                # across remote model waits would block pause/cancel and every
                # unrelated run for the full generation.
                # Capacity checks only read data; they must neither wait for
                # run mutations nor hold up pause/cancel while loading data.
                self._dispatch_post(path, body)
            elif control_request:
                # Serializing controls per run keeps a resume or second control
                # from overtaking a pause/cancel drain.  Pause and cancel keep
                # their small durable-write locks in _action, but must never
                # hold the server-wide lock while DSH and generation work drain.
                with self.server.control_lock(path[1]):
                    if draining_control:
                        self._dispatch_post(path, body)
                    else:
                        with self.server.mutation_lock:
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
                self._send(HTTPStatus.NOT_FOUND, public_error_payload(
                    FileNotFoundError("not found"), status=HTTPStatus.NOT_FOUND
                ))
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
        if path == ["model-preflight"]:
            self._model_preflight(body)
            return
        if path == ["evolution-capacity"]:
            self._evolution_capacity(body)
            return
        if len(path) == 3 and path[0] == "runs" and path[2] == "independent-evaluation":
            if set(body) != {"stage"}:
                raise ValueError("独立评测请求只接受 stage")
            self._send(HTTPStatus.ACCEPTED, self.server.independent_evaluation.start(path[1], body["stage"]))
            return
        if path == ["runs"]:
            self._create_run(body)
            return
        if len(path) == 3 and path[0] == "runs" and path[2] == "control":
            self._action(path[1], body)
            return
        if len(path) == 3 and path[0] == "runs" and path[2] == "archive":
            self._archive_run(path[1], body)
            return
        if len(path) == 3 and path[0] == "runs" and path[2] == "restore":
            self._restore_run(path[1], body)
            return
        self._send(HTTPStatus.NOT_FOUND, public_error_payload(
            FileNotFoundError("not found"), status=HTTPStatus.NOT_FOUND
        ))

    def _evolution_capacity(self, body: dict[str, Any]) -> None:
        fields = {
            "dataset_id",
            "episode_id",
            "optimization_schedule",
            "planned_generations",
        }
        if set(body) != fields:
            raise ValueError(
                "evolution-capacity requires exactly dataset_id, episode_id, "
                "optimization_schedule, and planned_generations"
            )
        dataset_id = body["dataset_id"]
        episode_id = body["episode_id"]
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("dataset_id must be non-empty")
        if episode_id is not None and (
            not isinstance(episode_id, str) or not episode_id.strip()
        ):
            raise ValueError("episode_id must be null or non-empty text")
        schedule = OptimizationSchedule.from_dict(body["optimization_schedule"])
        planned_generations = _request_integer(
            body["planned_generations"], "planned_generations", minimum=1
        )
        validate_run_parameter("rounds", planned_generations)
        dataset = self.server.datasets.selection_view(dataset_id, episode_id)
        # The browser previews native runs. Use the same history envelope as
        # _task_from_request, including the day-block evidence check below.
        history_steps = COHORT_HISTORY_HOURS
        report = estimate_epoch_capacity(
            dataset,
            schedule=schedule,
            planned_generations=planned_generations,
            seed=0,
            history_steps=history_steps,
            scoring_cells_per_origin=dataset_adapter(dataset_id).contract()["prediction_cells_per_origin"],
        )
        payload = report.to_dict()
        from ..evolution.evidence_capacity import require_guarded_cohort_evidence_capacity
        profile = self.server.evaluators.fitness_profile(dataset_adapter(dataset_id).evaluator_id, dataset_id)
        payload = capacity_with_inference_replicas(report, schedule)
        if report.sufficient:
            try:
                payload["guarded_evidence"] = require_guarded_cohort_evidence_capacity(
                    dataset=dataset, schedule=schedule,
                    planned_generations=planned_generations, seed=0,
                    profile=profile,
                    history_steps=history_steps,
                )
            except ValueError as exc:
                payload = capacity_with_inference_replicas(replace(report, sufficient=False), schedule)
                payload["rejection_reason"] = str(exc)
        payload["run_parameters"] = run_parameter_contract(profile)
        payload["data_partition_summary"] = self.server.datasets.partition_summary(dataset_id, episode_id)
        payload["capacity_enforced_for_run_creation"] = (
            True
        )
        payload["execution_plan"] = schedule.execution_plan(planned_generations,
            cells_per_origin=dataset_adapter(dataset_id).contract()["prediction_cells_per_origin"],
            native=True)
        self._send(HTTPStatus.OK, payload)

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
                "run still has active generation or independent evaluation work; "
                "retry deletion after that work finishes"
            )
        purged = False
        state = None
        try:
            state = self.server.director.state(run_id)
            _assert_http_scope(state)
            terminal_status = state.run.status.value
            if terminal_status not in {"completed", "cancelled", "failed"}:
                raise RuntimeError("只能永久删除已完成、已取消或失败的终态运行")
            admission = self.server.sample_admission.snapshot(run_id)
            if int(admission.get("active", 0)) or int(admission.get("waiting", 0)):
                raise CommandInProgressError(
                    "run still has active or waiting sample admissions"
                )
            deleted = self.server.ledger.purge_run(
                run_id,
                confirmation=confirmation,
                terminal_status=terminal_status,
            )
            self.server.dsh_identity_cache.forget(run_id)
            self.server.sample_admission.forget(run_id)
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

    def _model_preflight_directory(self) -> Path:
        return Path(self.server.ledger.path).expanduser().resolve().parent / "model-preflight"

    def _record_structured_rejection(
        self, envelope: Any, exc: BaseException, *, stage: str | None = None
    ) -> None:
        """Write the operator-only diagnostic for a rejected structured result."""
        try:
            directory = Path(self.server.ledger.path).expanduser().resolve().parent
        except (AttributeError, OSError, TypeError, ValueError):
            return
        _record_sidecar_rejection(directory, envelope, exc, stage=stage)

    @staticmethod
    def _request_binding(body: Mapping[str, Any], name: str) -> Any:
        """Resolve one binding identically for compact and manifest requests."""
        manifest = body.get("task_manifest")
        metadata = manifest.get("metadata", {}) if isinstance(manifest, Mapping) else {}
        if not isinstance(metadata, Mapping):
            raise TypeError("task_manifest.metadata must be an object")
        if name in body and name in metadata and body[name] != metadata[name]:
            raise ValueError(f"conflicting {name} in request and task_manifest.metadata")
        return body[name] if name in body else metadata.get(name)

    def _model_preflight(self, body: dict[str, Any]) -> None:
        check_only = body.get("check_only", False)
        if not isinstance(check_only, bool):
            raise TypeError("check_only must be a bool")
        if self._request_binding(body, "execution_protocol") != DSH_NATIVE_EXECUTION_PROTOCOL:
            raise ValueError("model preflight requires the native execution protocol")
        runtime = self.server.dsh_native_runtime
        if runtime is None:
            raise DshNativeRuntimeUnavailableError()
        self._dsh_native_capabilities = runtime.capabilities()
        task = self._task_from_request(
            {key: value for key, value in body.items() if key != "check_only"},
            defer_remote_plan=True,
        )
        if not self.server.model_preflight_lock.acquire(blocking=False):
            self._send(HTTPStatus.ACCEPTED, {"passed": False, "pending": True})
            return
        try:
            if check_only:
                # A browser timeout does not cancel the server's canaries.
                # Reconcile their exact bound identities without more model calls.
                try:
                    receipts = require_model_preflight(
                        task.metadata, self._model_preflight_directory(),
                    )
                    result = {"passed": True, "pending": False, "receipts": list(receipts)}
                except (OSError, ValueError, TypeError):
                    result = {"passed": False, "pending": False}
            else:
                result = run_preflight(
                    runtime, metadata=task.metadata,
                    receipt_directory=self._model_preflight_directory(),
                    force=False,
                )
        finally:
            self.server.model_preflight_lock.release()
        self._send(HTTPStatus.OK, result)

    def _create_run(self, body: dict[str, Any]) -> None:
        requested_workflow = body.get("model_workflow", body.get("workflow"))
        if requested_workflow is not None and str(requested_workflow) != "research_compile_evolve@1":
            raise ValueError("仅支持 research_compile_evolve@1；旧版组件搜索流程已移除")
        # Resolve execution policy before task binding. Autonomous research is
        # always deferred to generation execution so creating a durable run is
        # independent of current upstream queue health.
        start_value = body.get("start", True)
        if not isinstance(start_value, bool):
            raise TypeError("start must be a bool")
        start = start_value
        if body.get("auto_advance", "continuous") not in (0, "continuous"):
            raise ValueError("only continuous automatic execution is supported")
        if body.get("auto_progress", True) is not True:
            raise ValueError("auto_progress must be true; use start=false for an inactive run")
        if body.get("allow_host_fallback", False) is not False:
            raise ValueError("native execution does not allow host fallback")
        allow_host_fallback = False
        continuous_auto_progress = True
        # A durable command owns its frozen run even if live capability flags
        # or cached canary TTL changed after the first request. Validate the
        # identical HTTP body before recovering that resource, without
        # rebuilding a different TaskManifest from today's runtime state.
        cache_key = self._command_key("create", body)
        if self._serve_existing_command(cache_key, "create", "create_run", body):
            return
        existing_command = self.server.ledger.command_receipt(cache_key) if cache_key is not None else None
        if existing_command is not None and existing_command.resource_run_id:
            state = self.server.director.state(existing_command.resource_run_id)
            _assert_http_scope(state)
            state = self._resume_created_run(state, start=start)
            payload = _state_payload(state)
            self._complete_command(cache_key, payload)
            self._schedule_auto_progress(state)
            self._send(HTTPStatus.OK, payload)
            return
        native_protocol = self._request_binding(body, "execution_protocol") == DSH_NATIVE_EXECUTION_PROTOCOL
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
        task_workflow = task.metadata.get("model_workflow")
        if task_workflow is not None and task_workflow != "research_compile_evolve@1":
            raise ValueError("仅支持 research_compile_evolve@1；旧版组件搜索流程已移除")
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
                        existing_idempotent_state, start=start
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
                existing_idempotent_state, start=start
            )
            payload = _state_payload(state)
            self._complete_command(cache_key, payload)
            self._schedule_auto_progress(state)
            self._send(HTTPStatus.OK, payload)
            return
        preflight_receipts = None
        if task.metadata.get("require_model_contract_preflight") is True:
            if not native_protocol:
                raise ValueError("model preflight requires the native execution protocol")
            # Only new runs opt in. Dynamic receipts live in a separate event,
            # so refreshing their TTL cannot change the immutable task digest.
            task = replace(task, metadata={**dict(task.metadata), AUDIT_METADATA_KEY: AUDIT_SCHEMA})
            try:
                preflight_receipts = require_model_preflight(task.metadata, self._model_preflight_directory())
            except FileNotFoundError as exc:
                raise ValueError("缺少当前模型的有效预检回执，请先运行模型预检") from exc
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
                    "initial_run_status": "running" if start else "created",
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
            durable_creation = False
            if run_id is not None:
                try:
                    events = self.server.ledger.events(run_id)
                    durable_creation = bool(
                        events and events[0].kind == "RunCreated"
                        and TaskManifest.from_dict(events[0].payload["task_manifest"]).digest == task.digest
                    )
                    if durable_creation and cache_key is not None:
                        self.server.ledger.bind_command_resource_run(cache_key, run_id)
                except Exception as recovery_error:
                    # Preserve the original creation error; a failed secondary
                    # resource binding must not turn it into apparent success.
                    self.log_error("create recovery binding failed: %s", type(recovery_error).__name__)
            if native_protocol and run_id is not None and not durable_creation:
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
        if cache_key is not None:
            # Claim this durable resource before any later boundary can fail.
            self.server.ledger.bind_command_resource_run(cache_key, run.run_id)
        if preflight_receipts is not None:
            self.server.director.record_model_contract_preflight(run.run_id, preflight_receipts)
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
        if start:
            self.server.director.start_run(run.run_id)
        state = self.server.director.state(run.run_id)
        # Continuous runs return their durable generation-zero state and let
        # the worker execute every generation.  Numeric values retain the old
        # bounded synchronous behavior.
        payload = _state_payload(state)
        self._complete_command(cache_key, payload)
        self._schedule_auto_progress(state)
        self._send(HTTPStatus.CREATED, payload)

    def _schedule_auto_progress(self, state: Any) -> None:
        """Queue a running continuous run after its response state is built."""

        if state is None or state.run.status.value != "running":
            return
        self.server.auto_progress.schedule_if_enabled(state.run.run_id)


    def _resume_created_run(
        self,
        state: Any,
        *,
        start: bool,
    ) -> Any:
        """Finish a create command from its durable run projection."""

        run_id = state.run.run_id
        if state.run.status.value == "created":
            state = self._ensure_recorded_model_preflight(state)
        if start and state.run.status.value == "created":
            self._validate_frozen_runtime_bindings(state.task_manifest, run_id=run_id)
            if state.task_manifest.metadata.get("execution_protocol") == DSH_NATIVE_EXECUTION_PROTOCOL:
                self.server.dsh_native_runtime.activate({
                    "run_id": run_id, "run_state_revision": state.events[-1].seq,
                    "stage_attempt": 0, "ledger_expected_revision": self.server.ledger.latest_seq(),
                    "idempotency_key": f"create-recovery:{run_id}",
                })
            self.server.director.start_run(run_id)
            state = self.server.director.state(run_id)
        return state

    def _ensure_recorded_model_preflight(self, state: Any) -> Any:
        """Repair an interrupted new create; historical gates remain untouched."""
        if (state.run.status.value == "created"
                and state.task_manifest.metadata.get("execution_protocol") == DSH_NATIVE_EXECUTION_PROTOCOL):
            self.server.director.recover_run_initialization(state.run.run_id)
            state = self.server.director.state(state.run.run_id)
        if (not preflight_audit_required(state.task_manifest.metadata)
                or any(event.kind == AUDIT_EVENT for event in state.events)):
            return state
        try:
            receipts = require_model_preflight(state.task_manifest.metadata, self._model_preflight_directory())
        except FileNotFoundError as exc:
            raise ValueError("缺少当前模型的有效预检回执，请先运行模型预检") from exc
        self.server.director.record_model_contract_preflight(state.run.run_id, receipts)
        return self.server.director.state(state.run.run_id)

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
            expected_task = task
            if (preflight_audit_required(existing_manifest.metadata)
                    and task.metadata.get("require_model_contract_preflight") is True
                    and AUDIT_METADATA_KEY not in task.metadata):
                expected_task = replace(task, metadata={**dict(task.metadata), AUDIT_METADATA_KEY: AUDIT_SCHEMA})
            if existing_manifest.digest != expected_task.digest:
                raise ValueError("idempotency key already belongs to a different task")
            return existing_run_id
        return None

    def _task_from_request(self, body: dict[str, Any], *, strict_remote_plan: bool = False,
                           defer_remote_plan: bool = True) -> TaskManifest:
        """Bind the single public native joint contract; reject historical aliases."""
        allowed = {*_CREATE_REQUIRED_FIELDS, *_CREATE_OPTIONAL_FIELDS}
        unknown = set(body) - allowed
        if unknown:
            raise ValueError("unsupported create fields: " + ", ".join(sorted(unknown)))
        if body.get("execution_protocol") != DSH_NATIVE_EXECUTION_PROTOCOL:
            raise ValueError("new runs require dsh_native_plugin_evolution@1")
        if body.get("optimization_protocol", joint_policy.UNIFIED_PROTOCOL) != joint_policy.UNIFIED_PROTOCOL:
            raise ValueError("new runs require joint_plugin_epoch@2")
        if body.get("autonomous_mode", True) is not True:
            raise ValueError("native joint runs require autonomous_mode=true")
        if body.get("allow_host_fallback", False) is not False:
            raise ValueError("native joint runs do not permit host fallback")
        if body.get("require_model_contract_preflight", True) is not True:
            raise ValueError("native joint runs require four-role model preflight")
        if body.get("search_guard_policy", SEARCH_GUARD_POLICY) != SEARCH_GUARD_POLICY:
            raise ValueError("native joint runs require the frozen search guard")
        if body.get("model_workflow", "research_compile_evolve@1") != "research_compile_evolve@1":
            raise ValueError("unsupported model workflow")
        dataset_id = body.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id.strip() or dataset_id == "toy-crop-water@1":
            raise ValueError("dataset_id must identify a runnable greenhouse dataset")
        role_models = {}
        for key in ("strategy_model_id", "review_model_id"):
            value = body.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be non-empty")
            role_models[key] = value.strip()
        raw_budget = body.get("budget", {})
        if not isinstance(raw_budget, dict) or set(raw_budget) - {"max_generations", "candidates_per_generation", "max_candidates"}:
            raise ValueError("budget only accepts generation and candidate limits")
        rounds = body.get("rounds", raw_budget.get("max_generations", PARAMETER_RULES["rounds"]["default"]))
        rounds = validate_run_parameter("rounds", rounds)
        if "rounds" in body and "max_generations" in raw_budget and body["rounds"] != raw_budget["max_generations"]:
            raise ValueError("rounds differs from budget.max_generations")
        candidates = body.get("candidates_per_generation", raw_budget.get("candidates_per_generation", 4))
        if type(candidates) is not int or candidates != 4:
            raise ValueError("joint_plugin_epoch@2 requires four candidates per generation")
        maximum = body.get("max_candidates", raw_budget.get("max_candidates", rounds * 4))
        maximum = validate_run_parameter("max_candidates", maximum)
        if maximum < rounds * 4:
            raise ValueError("candidate budget cannot cover the frozen generations")
        for key in ("candidates_per_generation", "max_candidates"):
            if key in body and key in raw_budget and body[key] != raw_budget[key]:
                raise ValueError(f"{key} differs from budget.{key}")
        seed_policy = body.get("seed_policy", "fixed")
        if seed_policy not in {"fixed", "generated_and_recorded"}:
            raise ValueError("unsupported seed_policy")
        seed = (_request_integer(body["seed"], "seed") if "seed" in body else
                _derived_seed(body.get("idempotency_key") or body) if seed_policy == "generated_and_recorded" else 0)
        metadata = {key: body[key] for key in (
            "episode_id", "idempotency_key", "optimization_schedule", "prediction_selection_policy",
            "targeted_search_policy", "prediction_model_id", "evaluator_id", "seed_genome_template_id",
            "candidate_concurrency", "sample_concurrency", "sample_agent_batch_size", CANDIDATE_NOVELTY_POLICY_KEY,
        ) if key in body and body[key] is not None}
        metadata.update(role_models)
        metadata.update({
            "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
            "optimization_protocol": joint_policy.UNIFIED_PROTOCOL,
            "search_guard_policy": SEARCH_GUARD_POLICY,
            "paired_execution_qualification": PAIRED_EXECUTION_QUALIFICATION,
            "require_model_contract_preflight": True,
            "strategy_id": "autonomous_model@1", "autonomous_mode": True,
            "model_workflow": "research_compile_evolve@1", "requested_rounds": rounds,
            "recorded_seed": seed, "research_domain": body.get("research_domain", "greenhouse_environment@1"),
            "knowledge_online_enabled": _request_boolean(body.get("knowledge_online_enabled", True), "knowledge_online_enabled"),
        })
        task = TaskManifest(
            task_id=str(body.get("task_id") or f"plugin:{body.get('idempotency_key') or uuid4()}"),
            objective=str(body.get("objective") or "greenhouse multihorizon prediction"),
            domain_pack="greenhouse_environment@1", visible_datasets=(dataset_id.strip(),),
            budget={"max_generations": rounds, "candidates_per_generation": 4, "max_candidates": maximum},
            seed=seed, seed_policy=seed_policy, metadata=metadata,
        )
        return self._bind_runtime_task(task, defer_remote_plan=True)






    def _bind_runtime_task(
        self,
        manifest: TaskManifest,
        *,
        strict_remote_plan: bool = False,
        defer_remote_plan: bool = False,
    ) -> TaskManifest:
        if manifest.metadata.get("execution_protocol") != DSH_NATIVE_EXECUTION_PROTOCOL or manifest.metadata.get("optimization_protocol") != "joint_plugin_epoch@2":
            raise ValueError("only native joint_plugin_epoch@2 manifests are supported")
        if manifest.domain_pack != "greenhouse_environment@1":
            raise ValueError("only the greenhouse_environment@1 domain is supported")
        if len(manifest.visible_datasets) != 1:
            raise ValueError("本地运行必须且只能冻结一个数据集")
        dataset_id = manifest.visible_datasets[0]
        native_protocol = (
            manifest.metadata.get("execution_protocol")
            == DSH_NATIVE_EXECUTION_PROTOCOL
        )
        selection_view = None
        expected_partition = "training_feedback"
        requested_partition = manifest.metadata.get("evaluation_partition", expected_partition)
        if requested_partition != expected_partition:
            raise ValueError(
                f"运行只允许 {expected_partition} 分区；任务清单请求了：{requested_partition}"
            )
        description = self.server.datasets.describe(dataset_id)
        descriptor = description["descriptor"]
        readiness = description["readiness"]
        if not descriptor["runnable"] or not readiness["ready"]:
            raise ValueError(f"数据集当前不可运行：{dataset_id}")
        if "greenhouse" not in manifest.domain_pack.casefold():
            raise ValueError("真实温室数据集必须绑定 greenhouse_environment@1 领域模型包")
        metadata = dict(manifest.metadata)
        adapter = dataset_adapter(dataset_id)
        contract = adapter.contract()
        supplied_contract = metadata.get("dataset_task")
        if supplied_contract is not None and supplied_contract != contract:
            raise ValueError("任务清单中的数据集任务与服务端适配器不一致")
        if metadata.get("dataset_task_digest", contract["contract_digest"]) != contract["contract_digest"]:
            raise ValueError("dataset_task_digest must match the dataset adapter")
        metadata["dataset_task"] = contract
        metadata["dataset_task_digest"] = contract["contract_digest"]
        for key, value in (("objective_targets", list(adapter.target_names)),
                           ("objective_horizons", list(adapter.horizons_hours))):
            if key in metadata and metadata[key] != value:
                raise ValueError(f"{key} must match the dataset adapter")
            metadata[key] = value
        requested_episode = metadata.get("episode_id")
        series = self.server.datasets.selection_view(
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
            "research_compile_evolve@1",
        )
        metadata.setdefault("research_domain", manifest.domain_pack)
        strategy_id = str(
            metadata.get("strategy_id")
            or ("autonomous_model@1")
        )
        novelty_policy = metadata.get(CANDIDATE_NOVELTY_POLICY_KEY)
        from ..evolution.program_preflight import POLICY_KEY as executable_key, POLICY as executable_policy, enabled as executable_enabled
        executable_enabled(metadata)
        metadata[executable_key] = executable_policy
        if novelty_policy not in (None, CANDIDATE_NOVELTY_POLICY):
            raise ValueError("unsupported candidate_novelty_policy")
        metadata[CANDIDATE_NOVELTY_POLICY_KEY] = CANDIDATE_NOVELTY_POLICY
        targeted_policy = metadata.get("targeted_search_policy")
        if targeted_policy not in (None, TARGETED_SEARCH_POLICY):
            raise ValueError("unsupported targeted_search_policy")
        if targeted_policy:
            if not native_protocol or not autonomous_mode or dataset_id == "toy-crop-water@1":
                raise ValueError("targeted search requires a native autonomous greenhouse run")
            metadata["behavior_diagnostics_policy"] = CONTRIBUTION_POLICY
        selection_policy = metadata.get("prediction_selection_policy")
        if selection_policy not in (None, RUNTIME_PREDICTION_POLICY):
            raise ValueError("unsupported prediction_selection_policy")
        runtime_choice = selection_policy == RUNTIME_PREDICTION_POLICY or (
            native_protocol and autonomous_mode and True and not metadata.get("prediction_model_id")
            and not metadata.get("evaluator_id")
        )
        if runtime_choice:
            if not native_protocol or not autonomous_mode or dataset_id == "toy-crop-water@1":
                raise ValueError("runtime predictor selection requires a native autonomous greenhouse run")
            if any(metadata.get(key) for key in ("prediction_model_id", "evaluator_id", "seed_genome_template_id")):
                raise ValueError("runtime predictor selection does not accept a preselected predictor, evaluator or seed template")
            metadata["prediction_selection_policy"] = RUNTIME_PREDICTION_POLICY
            # This is a no-residual comparison anchor, not a user's chosen model.
            metadata["prediction_model_id"] = ("greenhouse-baseline-aligned-ridge@2" if targeted_policy else BASELINE_REFERENCE_PREDICTOR_ID)
            metadata["evaluator_id"] = (CELLWISE_EVALUATOR_ID if targeted_policy else dataset_adapter(dataset_id).evaluator_id)
        prediction_model_id = str(
            metadata.get("prediction_model_id")
            or self.server.evaluators.default_predictor(dataset_id)
        )
        evaluator_id = str(
            metadata.get("evaluator_id")
            or (CELLWISE_EVALUATOR_ID if prediction_model_id == "greenhouse-baseline-aligned-ridge@2"
                else self.server.evaluators.default_evaluator(dataset_id))
        )
        if (
            native_protocol
            and True
            and evaluator_id not in {RUNTIME_EVALUATOR_ID, CELLWISE_EVALUATOR_ID}
        ):
            raise ValueError(
                "DSH-native greenhouse runs require "
                f"a registered multi-horizon evaluator ({RUNTIME_EVALUATOR_ID} for runtime model choice)"
            )
        if targeted_policy and evaluator_id != CELLWISE_EVALUATOR_ID:
            raise ValueError("targeted search requires the versioned cellwise evaluator")
        strategy_model_id = str(
            metadata.get("strategy_model_id")
            or ""
        )
        policy_model_id = strategy_model_id
        judge_model_id = str(
            metadata.get("review_model_id")
            or ""
        )
        strategy_id = "autonomous_model@1"
        if not strategy_model_id:
            raise ValueError("自主进化必须选择已接入的策略模型")
        if not judge_model_id:
            raise ValueError("自主进化必须选择已接入的独立评审模型")
        if strategy_id not in StrategyRouterDSHAdapter.SUPPORTED_STRATEGIES:
            raise ValueError(f"未知进化策略：{strategy_id}")
        self.server.evaluators.validate_binding(
            dataset_id, evaluator_id, prediction_model_id
        )
        strategy_digest = self.server.strategy_router.configuration_digest(strategy_id)
        evaluator_digest = self.server.evaluators.evaluator_configuration_digest(
            evaluator_id, dataset_id
        )
        objective_profile = self.server.evaluators.objective_profile(evaluator_id, dataset_id)
        fitness_profile = self.server.evaluators.fitness_profile(evaluator_id, dataset_id)
        adapter = dataset_adapter(dataset_id)
        if fitness_profile.expected_targets != adapter.target_names or fitness_profile.expected_horizons != adapter.horizons_hours:
            raise ValueError("native evaluator grid must match the dataset adapter")
        fitness_profile_data = fitness_profile.to_dict()
        supplied_fitness_profile = metadata.get("fitness_profile")
        if (
            supplied_fitness_profile is not None
            and supplied_fitness_profile != fitness_profile_data
        ):
            raise ValueError("任务清单中的 fitness_profile 与服务端评测器不一致")
        supplied_fitness_digest = metadata.get("fitness_profile_digest")
        if (
            supplied_fitness_digest is not None
            and supplied_fitness_digest != fitness_profile.profile_digest
        ):
            raise ValueError("任务清单中的 fitness_profile_digest 与服务端评测器不一致")
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
        if not policy_model_id or not judge_model_id:
            raise ValueError("DSH-native roles require frozen strategy and review routes")
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
            digest({"policy": "dsh-native-search-reflect-evolution@2"}),
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
        for field, computed, label in (
            ("policy_model_digest", policy_model_digest, "候选生成模型"),
            ("judge_model_digest", judge_model_digest, "独立评审模型"),
        ):
            supplied = metadata.get(field)
            if supplied is not None and supplied != computed:
                raise ValueError(f"任务清单中的{label}校验值与服务端配置不一致")
        if prediction_model_id in ("greenhouse-baseline-aligned-ridge@1", "greenhouse-baseline-aligned-ridge@2"):
            metadata.setdefault("search_guard_policy", SEARCH_GUARD_POLICY)
        from ..core.model_execution_policy import (
            RESEARCH_EXECUTION_POLICY, RESEARCH_EXECUTION_POLICY_KEY,
            research_execution_policy,
        )
        metadata.setdefault(RESEARCH_EXECUTION_POLICY_KEY, dict(RESEARCH_EXECUTION_POLICY))
        research_execution_policy(metadata)
        autonomous_plan: dict[str, Any] = {}
        evaluator_catalog = [
            dict(item)
            for item in self.server.evaluators.catalog(dataset_id)
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
        if runtime_choice:
            metadata["prediction_selection"] = {
                "policy": RUNTIME_PREDICTION_POLICY,
                "owner": "research_model",
                "reference_mode": "baseline_only",
                "reference_predictor_id": BASELINE_REFERENCE_PREDICTOR_ID,
                "allowed_predictor_ids": [item["id"] for item in compatible_predictors],
                "can_disable_residual_model": True,
                "can_switch_predictor": True,
                "can_optimize_parameters": True,
                "evaluation_rules_frozen": True,
            }
        metadata["autonomous_plan_execution"] = "deferred_to_first_generation"
        seed_template_id = metadata.get("seed_genome_template_id")
        if seed_template_id is None:
            seed_template_id = _DSH_NATIVE_SEED_TEMPLATE_BY_PREDICTOR.get(
                prediction_model_id
            )
        if not isinstance(seed_template_id, str) or not seed_template_id.strip():
            raise ValueError(
                "DSH-native prediction model has no registered seed genome template"
            )
        seed_template_id = seed_template_id.strip()
        seed_template = current_program_registry().seed_template(
            seed_template_id
        )
        seed_predictor_id = str(
            seed_template.to_dict()["scientific_program"]["predictor_ref"]["id"]
        )
        if seed_predictor_id != prediction_model_id:
            raise ValueError(
                "DSH-native seed genome predictor does not match the frozen "
                "prediction_model_id"
            )
        metadata["seed_genome_template_id"] = seed_template_id
        schedule: OptimizationSchedule | None = None
        requested_optimization_protocol = metadata.get("optimization_protocol")
        requested_schedule = metadata.get("optimization_schedule")
        if requested_optimization_protocol not in (None, joint_policy.UNIFIED_PROTOCOL):
            raise ValueError("New experiments require joint_plugin_epoch@2; legacy protocols are replay-only")
        schedule = (OptimizationSchedule.for_new_run() if requested_schedule is None
                    else OptimizationSchedule.from_dict(requested_schedule))
        pass
        if requested_optimization_protocol is not None and requested_optimization_protocol != schedule.protocol:
            raise ValueError("optimization_protocol differs from the frozen schedule")
        if manifest.candidates_per_generation != 4:
            raise ValueError(
                "joint_plugin_epoch@2 requires candidates_per_generation == 4"
            )

        sample_concurrency = metadata.get("sample_concurrency")
        candidate_concurrency = metadata.get("candidate_concurrency")
        sample_agent_batch_size = metadata.get("sample_agent_batch_size")
        selection_fitness_profile = self.server.evaluators.fitness_profile(
            evaluator_id, dataset_id
        )
        validate_run_parameter("rounds", manifest.max_generations)
        validate_run_parameter("max_candidates", manifest.max_candidates)
        validate_run_parameter("selection_holdout_origin_count", schedule.selection_holdout_origin_count,
                               profile=selection_fitness_profile)
        minimum_selection_origin_samples_per_update = (
            selection_fitness_profile.minimum_origins_for_schedule(schedule)
        )
        prediction_cells_per_origin = selection_fitness_profile.prediction_cell_count
        minimum_selection_samples_per_update = (
            minimum_selection_origin_samples_per_update * prediction_cells_per_origin
        )
        # The one depth this run guarantees: the planner bakes it into every
        # ``origin_id`` and the evaluator intersects the cohort down to it, so
        # both have to read this single value. DSH-native greenhouse runs freeze
        # the full cohort ceiling, which is what makes declarative feature
        # recipes reachable at all -- the recipe predictor needs all 48 and the
        # seed recipe tool 25, and below that depth both are refused rather
        # than offered and then failed per origin. Every other mode keeps the
        # historical 12 so its origin set, and therefore its score, is unmoved.
        origin_history_alignment = (
            COHORT_HISTORY_HOURS
        )
        cohort_capacity_report = None
        pass
        cohort_capacity_report = estimate_epoch_capacity(
            selection_view,
            schedule=schedule,
            planned_generations=manifest.max_generations,
            seed=manifest.seed,
            scoring_cells_per_origin=prediction_cells_per_origin,
            history_steps=origin_history_alignment,
        )
        if not cohort_capacity_report.sufficient:
            raise CohortCapacityError(
                required=cohort_capacity_report.required_unique_origins,
                available=cohort_capacity_report.available_eligible_origins,
                max_generations=cohort_capacity_report.max_feasible_generations,
            )
        from ..evolution.evidence_capacity import (
            require_guarded_cohort_evidence_capacity,
        )
        metadata["guarded_cohort_evidence_capacity"] = require_guarded_cohort_evidence_capacity(
            dataset=selection_view,
            schedule=schedule,
            planned_generations=manifest.max_generations,
            seed=manifest.seed,
            profile=selection_fitness_profile,
            history_steps=origin_history_alignment,
        )
        sample_budget_class = None
        # DSH-native sample execution needs the same frozen scheduling values
        # for every dataset.  The synthetic fixture is small, but it still
        # runs planner/critic child sessions; clearing these fields leaves the
        # adapter with an explicit ``None`` concurrency and makes evaluation
        # fail before the first sample checkpoint.
        if candidate_concurrency is None:
            candidate_concurrency = _DEFAULT_REAL_CANDIDATE_CONCURRENCY
        if (
            isinstance(candidate_concurrency, bool)
            or not isinstance(candidate_concurrency, int)
            or not (
                1
                <= candidate_concurrency
                <= _MAX_REAL_CANDIDATE_CONCURRENCY
            )
        ):
            raise ValueError(
                "candidate_concurrency must be an integer between 1 and "
                f"{_MAX_REAL_CANDIDATE_CONCURRENCY}"
            )
        if sample_concurrency is None:
            sample_concurrency = DEFAULT_SAMPLE_CONCURRENCY
        sample_concurrency = validate_sample_concurrency(sample_concurrency)
        if sample_agent_batch_size is not None and (
            isinstance(sample_agent_batch_size, bool)
            or not isinstance(sample_agent_batch_size, int)
            or sample_agent_batch_size != prediction_cells_per_origin
        ):
            raise ValueError(
                "sample_agent_batch_size is derived from the frozen fitness "
                "profile and must equal prediction_cells_per_origin="
                f"{prediction_cells_per_origin}"
            )
        sample_agent_batch_size = prediction_cells_per_origin
        sample_budget_class = (
            "selection_eligible"
            if schedule.selection_holdout_origin_count
            >= minimum_selection_origin_samples_per_update
            else "insufficient_holdout"
        )
        runtime_component_catalog["selected_prediction_model_id"] = (
            prediction_model_id
        )
        metadata["optimization_protocol"] = schedule.protocol
        supplied_runtime_build = metadata.get("host_runtime_build")
        diversity_enabled(metadata)
        if (
            supplied_runtime_build is not None
            and not _runtime_build_compatible(supplied_runtime_build)
        ):
            raise ValueError(
                "任务清单中的 Host 进化运行协议与当前服务不兼容"
            )
        metadata.update(
            {
                "host_runtime_build": _host_runtime_build(),
                "evolution_diversity_policy": metadata.get("evolution_diversity_policy", joint_policy.DIVERSITY_POLICY),
                "strategy_id": strategy_id,
                "strategy_digest": strategy_digest,
                "mutation_policy": mutation_policy_contract(metadata),
                "local_edit_context_policy": LOCAL_EDIT_CONTEXT_POLICY,
                "mutation_policy_digest": digest(mutation_policy_contract(metadata)),
                "prequential_performance_policy": (
                    PREQUENTIAL_PERFORMANCE_POLICY
                ),
                "evaluator_id": evaluator_id,
                "evaluator_digest": evaluator_digest,
                # Freeze the objective definition with the run so the score
                # cannot be misread as a free-form natural-language goal.
                "objective_profile": objective_profile,
                "fitness_profile": fitness_profile_data,
                "fitness_profile_digest": fitness_profile.profile_digest,
                "prediction_model_id": prediction_model_id,
                "prediction_model_digest": prediction_model_digest,
                "policy_model_id": policy_model_id,
                "judge_model_id": judge_model_id,
                "strategy_model_id": strategy_model_id,
                "review_model_id": judge_model_id,
                "autonomous_mode": autonomous_mode,
                "autonomous_research_protocol": (
                    AUTONOMOUS_RESEARCH_PROTOCOL
                ),
                # DSH-native runs freeze the agent-owned origin-vector protocol
                # into the task digest. Other execution modes retain their own
                # explicit sample adapters.
                "sample_agent_mode": (
                    "dsh_native_agent"
                ),
                # Every DSH-native origin keeps a real model-backed Planner.
                # Critic is reserved for uncertain/failing routes and the
                # existing candidate/generation review owns post-score reflection.
                "sample_agent_protocol": (
                    _STRICT_SAMPLE_AGENT_PROTOCOL
                ),
                "sample_prompt_batch_size": 1,
                "sample_reflection_policy": (
                    _STRICT_SAMPLE_REFLECTION_POLICY
                ),
                "allow_host_route_bypass": False,
                "allow_host_prediction_fallback": (
                    False
                ),
                "sample_agent_batch_size": sample_agent_batch_size,
                # Causal origin waves are independent schedules. New real
                # runs default to 64 sample requests, bounded to 1–128; the
                # DSH host separately enforces a provider-wide physical cap of
                # 128. An explicit value remains immutable in the manifest.
                # Candidate-level parallelism is frozen separately.
                "sample_concurrency": sample_concurrency,
                # Sibling candidates share one frozen cohort but execute in
                # independent DSH child sessions. Historical manifests omit
                # this field and therefore remain serial on replay.
                "candidate_concurrency": candidate_concurrency,
                "optimization_protocol": (
                    schedule.protocol
                ),
                "optimization_schedule": (
                    schedule.to_dict()
                ),
                "local_comparison_policy": (
                    None
                ),
                "cohort_capacity_report": (
                    capacity_with_inference_replicas(cohort_capacity_report, schedule)
                ),
                "cohort_capacity_enforced": (
                    True
                ),
                "derived_execution_budget": (
                    schedule.generation_execution_budget(
                        cells_per_origin=prediction_cells_per_origin,
                        holdout_inference_replicas=REPLICA_COUNT,
                    )
                ),
                "execution_plan": schedule.execution_plan(manifest.max_generations,
                    cells_per_origin=prediction_cells_per_origin, native=native_protocol),
                "derived_run_execution_budget": (
                    schedule.run_execution_budget(
                        manifest.max_generations,
                        cells_per_origin=prediction_cells_per_origin,
                        holdout_inference_replicas=REPLICA_COUNT,
                    )
                ),
                "minimum_selection_samples_per_update": (
                    minimum_selection_samples_per_update
                ),
                "minimum_selection_origin_samples_per_update": (
                    minimum_selection_origin_samples_per_update
                ),
                "prediction_cells_per_origin": prediction_cells_per_origin,
                # Frozen with the run, not read from a package constant, so a
                # later default change cannot re-plan an existing cohort.
                "origin_history_alignment_hours": origin_history_alignment,
                "sample_budget_class": sample_budget_class,
                # These limits are part of the task manifest rather than a
                # process-wide gateway default, so later configuration changes
                # cannot silently change an already-created real run.
                "sample_operation_max_tokens": (
                    dict(
                        NATIVE_SAMPLE_OPERATION_MAX_TOKENS
                    )
                ),
                # Sparse remote review preserves Host scoring while avoiding
                # one advisory model call for every successful origin. Persisting
                # the policy prevents restart-time drift; older runs retain their
                # frozen always-review policy.
                "sample_remote_critic_policy": (
                    dict(
                        _STRICT_SAMPLE_REMOTE_CRITIC_POLICY
                    )
                ),
                # Repeated origin-level features are losslessly factored into a
                # content-addressed context only for newly created real runs.
                "sample_planner_prompt_profile": (
                    dict(_DEFAULT_SAMPLE_PLANNER_PROMPT_PROFILE)
                ),
                "sample_truncation_retry_policy": (
                    None
                ),
                "model_selection_policy": (
                    "model_research_and_runtime_compile@1"
                ),
                "model_workflow": metadata.get(
                    "model_workflow",
                    "research_compile_evolve@1",
                ),
                "research_domain": metadata.get(
                    "research_domain", manifest.domain_pack
                ),
                "autonomous_plan": autonomous_plan,
                "autonomous_plan_digest": digest(autonomous_plan),
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
        native_budget = dict(data["budget"])
        native_token_limit = int(native_budget.get("token_limit", 0))
        if native_token_limit > 0:
            raise ValueError(
                "DSH-native token_limit is unsupported because provider "
                "session reports are telemetry, not an atomic reservation ledger"
            )
        # DSH-native runs do not receive a hidden static cap: the observed
        # default 5-epoch plan needs well above the legacy sample-gateway
        # budget. Session context bounds remain owned by DSH, while the
        # sidecar exposes provider-reported cumulative usage as telemetry.
        native_budget.pop("token_limit", None)
        native_budget.pop("token_reservation_per_wave", None)
        data["budget"] = native_budget
        # Native provider usage is telemetry rather than an atomic Host
        # reservation ledger, but each child still needs a frozen output
        # cap. Keep sample_operation_max_tokens in the task manifest while
        # removing only the legacy Host-side accounting policy.
        for legacy_token_field in (
            "sample_truncation_retry_policy",
            "sample_token_budget_policy",
            "token_budget_scope",
            "run_wide_accounting_complete",
            "sample_token_budget_defaulted",
            "dsh_provider_token_budget_policy",
        ):
            metadata.pop(legacy_token_field, None)
        data["metadata"] = metadata
        return TaskManifest.from_dict(data)



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
        frozen_runtime_build = metadata.get("host_runtime_build")
        # Historical runs predate this field and keep their already-frozen
        # schedule semantics. New runs fail closed only when an explicitly
        # persisted compatibility schema no longer matches this Host.
        if frozen_runtime_build is not None and not _runtime_build_compatible(
            frozen_runtime_build
        ):
            raise FrozenRuntimeBindingDriftError("Host evolution runtime", expected_digest=digest(frozen_runtime_build), current_digest=digest(_host_runtime_build()))
        if True:
            contract = dataset_adapter(dataset_id).contract()
            if metadata.get("dataset_task") != contract or metadata.get("dataset_task_digest") != contract["contract_digest"]:
                raise FrozenRuntimeBindingDriftError("dataset task adapter", expected_digest=digest(metadata.get("dataset_task")), current_digest=digest(contract))
        dataset_digest = metadata.get("dataset_digest")
        split_digest = metadata.get("split_manifest_digest")
        if not isinstance(dataset_digest, str) or not dataset_digest.strip():
            raise ValueError("旧运行缺少冻结的数据集快照校验值，已拒绝继续")
        if not isinstance(split_digest, str) or not split_digest.strip():
            raise ValueError("旧运行缺少冻结的时间分区快照校验值，已拒绝继续")
        episode_id = metadata.get("episode_id")
        series = server.datasets.selection_view(
            dataset_id,
            str(episode_id) if episode_id is not None else None,
            expected_dataset_digest=dataset_digest,
            expected_split_manifest_digest=split_digest,
        )
        if series.digest != dataset_digest:
            raise FrozenRuntimeBindingDriftError("数据集快照", expected_digest=dataset_digest, current_digest=series.digest)
        if series.split_manifest_digest_sha256 != split_digest:
            raise FrozenRuntimeBindingDriftError("时间分区快照", expected_digest=split_digest, current_digest=series.split_manifest_digest_sha256)

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
                lambda evaluator_id: server.evaluators.evaluator_configuration_digest(evaluator_id, dataset_id),
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
                raise FrozenRuntimeBindingDriftError(f"{label}实现", expected_digest=expected_digest, current_digest=current_digest)

        if metadata.get("execution_protocol") == DSH_NATIVE_EXECUTION_PROTOCOL:
            evaluator_id = str(metadata.get("evaluator_id") or "").strip()
            expected_profile = server.evaluators.fitness_profile(evaluator_id, dataset_id)
            frozen_profile = metadata.get("fitness_profile")
            if not isinstance(frozen_profile, dict):
                raise ValueError(
                    "DSH-native 运行缺少冻结的完整 fitness_profile，已拒绝继续"
                )
            if frozen_profile != expected_profile.to_dict():
                raise FrozenRuntimeBindingDriftError("fitness profile", expected_digest=digest(frozen_profile), current_digest=digest(expected_profile.to_dict()))
            if metadata.get("fitness_profile_digest") != expected_profile.profile_digest:
                raise FrozenRuntimeBindingDriftError("fitness profile digest", expected_digest=metadata.get("fitness_profile_digest"), current_digest=expected_profile.profile_digest)
            raw_schedule = metadata.get("optimization_schedule")
            expected_origins = expected_profile.minimum_origins_for_schedule(
                OptimizationSchedule.from_dict(raw_schedule) if raw_schedule is not None else None
            )
            if metadata.get("minimum_selection_samples_per_update") != (
                expected_origins * expected_profile.prediction_cell_count
            ):
                raise FrozenRuntimeBindingDriftError("selection sample threshold")
            if is_strict_origin_protocol(
                metadata.get("sample_agent_protocol")
            ) and (
                metadata.get("minimum_selection_origin_samples_per_update")
                != expected_origins
                or metadata.get("prediction_cells_per_origin")
                != expected_profile.prediction_cell_count
            ):
                raise FrozenRuntimeBindingDriftError(
                    "origin-bundle selection threshold"
                )
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
                raise FrozenRuntimeBindingDriftError("DSH preset capability", expected_digest=metadata.get("dsh_preset_catalog_digest"), current_digest=preset_catalog_digest)
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
                    raise FrozenRuntimeBindingDriftError(f"{label} DSH 路由", expected_digest=metadata.get(digest_field), current_digest=current_digest)
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
                    restored_status = state.run.status.value
                    if restored_status not in {"created", "running", "paused"}:
                        restored_status = "created"
                    runtime.create_run(
                        {
                            "run_id": target_run_id,
                            "run_state_revision": latest_run_revision,
                            "stage_attempt": 0,
                            "ledger_expected_revision": server.ledger.latest_seq(),
                            "idempotency_key": f"runtime-restore:{target_run_id}",
                            "binding": {
                                "execution_protocol": DSH_NATIVE_EXECUTION_PROTOCOL,
                                "initial_run_status": restored_status,
                                "restore_provenance": {
                                    "source": "python_durable_ledger",
                                    "status": restored_status,
                                },
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
        valid_actions = {"start", "pause", "resume", "cancel", "complete"}
        if action not in valid_actions:
            raise ValueError("action must be start, pause, resume, cancel, or complete")
        # Controls are always durable commands. Generate an idempotency key
        # for callers that omitted one so a fast 202 response remains pollable.
        if action in {"pause", "cancel"} and not body.get("idempotency_key"):
            body = dict(body)
            body["idempotency_key"] = f"auto-{action}-{uuid4()}"
        cache_key = self._command_key(run_id, body)
        existing_receipt = (
            self.server.ledger.command_receipt(cache_key)
            if cache_key is not None
            else None
        )
        if existing_receipt is not None and existing_receipt.status == "pending":
            self._send(HTTPStatus.ACCEPTED, self._command_status(cache_key))
            return
        if action in {"start", "resume"}:
            if self.server.native_control_inflight_for(
                run_id,
                self.server.dsh_native_runtime,
            ):
                raise CommandInProgressError(
                    "run control is draining; retry start/resume after native quiescence completes"
                )
        if self._serve_existing_command(cache_key, run_id, f"control:{action}", body):
            return
        state = director.state(run_id)
        native_protocol = (
            state.task_manifest.metadata.get("execution_protocol")
            == DSH_NATIVE_EXECUTION_PROTOCOL
        )
        active = getattr(self, "_active_command", None)
        target_status = {
            "start": "running",
            "pause": "paused",
            "resume": "running",
            "cancel": "cancelled",
            "complete": "completed",
        }.get(action)
        resumes_durable_native_quiescence = (
            active is not None
            and action in {"pause", "cancel"}
            and native_protocol
            and state.run.status.value == target_status
        )
        if (
            active is not None
            and target_status is not None
            and state.run.status.value == target_status
            and not resumes_durable_native_quiescence
        ):
            payload = _control_payload(
                state,
                self.server.sample_admission.snapshot(run_id),
            )
            self._complete_command(cache_key, payload)
            self._send(HTTPStatus.OK, payload)
            return
        if not resumes_durable_native_quiescence:
            self._validate_control_request(run_id, action)
        cached = self._claim_command(cache_key, run_id, f"control:{action}", body)
        if cached is not None:
            self._send(HTTPStatus.OK, cached)
            return
        if action == "start" and state.run.status.value == "created":
            state = self._ensure_recorded_model_preflight(state)
        native_request = {
            "run_id": run_id,
            "run_state_revision": state.events[-1].seq,
            "stage_attempt": 0,
            "ledger_expected_revision": self.server.ledger.latest_seq(),
            "idempotency_key": str(
                body.get("idempotency_key") or f"{action}:{run_id}:{state.events[-1].seq}"
            ),
        }
        pause_reason: str | None = None
        pause_code: str | None = None
        if action == "pause":
            # Validate the operator metadata before closing any admission fence.
            # Native runs persist the paused boundary below before waiting for
            # DSH quiescence so active sample executors stop treating aborted
            # child calls as ordinary sample failures.
            raw_pause_reason = body.get("reason")
            raw_pause_code = body.get("code")
            if raw_pause_reason is not None and not isinstance(
                raw_pause_reason, str
            ):
                raise TypeError("pause reason must be a string")
            if raw_pause_code is not None and not isinstance(raw_pause_code, str):
                raise TypeError("pause code must be a string")
            pause_reason = raw_pause_reason if raw_pause_reason is not None else "运行控制接口收到暂停请求，当前进度已保存。"
            pause_code = raw_pause_code if raw_pause_code is not None else "operator_pause"
        if native_protocol and action in {"pause", "cancel"}:
            runtime = self.server.dsh_native_runtime
            native_marker = (id(runtime), object())
            if not self.server.claim_native_control_inflight(
                run_id,
                native_marker,
            ):
                raise CommandInProgressError(
                    "run control is already draining; retry after native quiescence completes"
                )
            self.server.dsh_tools.close_run_admissions(run_id)
            # Persist the Host safety boundary synchronously, then let remote
            # DSH quiesce/drain in a daemon worker. This prevents browser or
            # proxy timeouts from being coupled to provider latency.
            if not resumes_durable_native_quiescence:
                try:
                    with self.server.mutation_lock:
                        if action == "pause":
                            director.pause_run(
                                run_id,
                                reason=pause_reason,
                                code=pause_code,
                            )
                        else:
                            director.cancel_run(
                                run_id,
                                str(body.get("reason", "cancelled by user")),
                            )
                except Exception:
                    # Another lifecycle writer may win after the state snapshot
                    # above but before this compare-and-swap transition.  A
                    # transition helper can also raise after its durable event
                    # committed (for example while closing a Session).  Re-read
                    # the ledger before deciding who owns the native drain.
                    try:
                        reconciled_state = director.state(run_id)
                    except Exception:  # noqa: TRY203 - the bare re-raise is the decision
                        # The durable outcome is unknown.  Keep the admission
                        # fence and ownership marker fail-closed; startup recovery
                        # can prove the lifecycle before allowing a resume.
                        # Deliberately not folded into the caller: the reason this
                        # path does *not* reconcile has to stay next to the code
                        # that skips reconciliation.
                        raise
                    reconciled_status = reconciled_state.run.status.value
                    if reconciled_status == target_status:
                        # Our event committed, or an equivalent control won.
                        # Continue with the same remote quiescence boundary.
                        state = reconciled_state
                    elif reconciled_status in {
                        "paused",
                        "cancelled",
                        "completed",
                        "failed",
                    }:
                        # A different Host lifecycle boundary won after this
                        # request had already claimed the only native drain.
                        # That writer could not claim its own marker, so this
                        # request must still finish the original pause/cancel;
                        # merely clearing the marker would leave DSH running
                        # behind a paused or terminal Host projection.
                        state = reconciled_state
                    else:
                        # The run is still live and this transition did not
                        # commit. Restore its admission before returning the
                        # retryable command error.
                        self.server.clear_native_control_inflight(
                            run_id,
                            native_marker,
                        )
                        self.server.dsh_tools.open_run_admissions(run_id)
                        raise
            accepted_state = director.state(run_id)
            # A lifecycle race can advance the Host cursor after the request
            # was assembled. Bind the remote quiescence to the state that is
            # actually being acknowledged, while keeping the caller's stable
            # idempotency key.
            native_request["run_state_revision"] = accepted_state.events[-1].seq
            native_request["ledger_expected_revision"] = (
                self.server.ledger.latest_seq()
            )
            accepted_payload = _control_payload(
                accepted_state,
                self.server.sample_admission.snapshot(run_id),
            )
            accepted_payload.update(
                {
                    "command_id": cache_key,
                    "command_status": "pending",
                    "accepted": True,
                    "message": "主机状态边界已落盘，DSH 资源正在后台排空。",
                }
            )
            completed_inline = self._launch_native_control_worker(
                cache_key,
                run_id,
                action,
                native_request,
                runtime,
                native_marker,
            )
            self._active_command = None
            if completed_inline:
                receipt = self.server.ledger.command_receipt(cache_key)
                if (
                    receipt is not None
                    and receipt.status == "completed"
                    and receipt.response is not None
                ):
                    self._send(HTTPStatus.OK, dict(receipt.response))
                    return
            self._send(HTTPStatus.ACCEPTED, accepted_payload)
            return
        if action in {"start", "resume"} and state.run.status.value == "paused":
            # The generation/native drain barriers above must precede clearing
            # the failure that rejected queued sibling origins.
            self.server.sample_admission.reset_execution_failure(run_id)
        if native_protocol and action == "resume":
            self.server.dsh_native_runtime.resume(native_request)
        if action == "start":
            if native_protocol:
                if state.run.status.value == "created":
                    self.server.dsh_native_runtime.activate(native_request)
                else:
                    # The public control contract has historically accepted
                    # `start` as an alias for resuming a paused run.  Native
                    # activation is deliberately narrower (created/running),
                    # so keep the DSH and Host state machines aligned by
                    # resuming rather than reactivating this paused run.
                    self.server.dsh_native_runtime.resume(native_request)
            director.start_run(run_id)
            if native_protocol:
                self.server.dsh_tools.open_run_admissions(run_id)
        elif action == "pause":
            if not native_protocol:
                # Preserve an operator-supplied pause cause in the append-only
                # event stream so the projection can distinguish manual pauses
                # from budget and gateway back-pressure pauses.
                with self.server.mutation_lock:
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
            if not native_protocol:
                with self.server.mutation_lock:
                    director.cancel_run(
                        run_id,
                        str(body.get("reason", "cancelled by user")),
                    )
        elif action == "complete":
            director.complete_run(run_id)
        if native_protocol and action in {"pause", "cancel"}:
            # DSH quiescence covers child Sessions and Workflows. The Host may
            # still be unwinding a completed local training/evaluation call, so
            # join the per-run generation lease before acknowledging control.
            generation_barrier = self.server.acquire_generation_lease(run_id)
            if generation_barrier is None:
                raise RuntimeError("run generation quiescence barrier is unavailable")
            generation_barrier.release()
        resulting_state = director.state(run_id)
        payload = _control_payload(
            resulting_state,
            self.server.sample_admission.snapshot(run_id),
        )
        self._complete_command(cache_key, payload)
        if action in {"start", "resume"}:
            # A paused autonomous run resumes at the next generation boundary;
            # the manager deduplicates this with any worker that is already
            # finishing the previous boundary.
            self._schedule_auto_progress(resulting_state)
        self._send(HTTPStatus.OK, payload)

    @staticmethod
    def _start_native_control_thread(worker: threading.Thread) -> None:
        """Small seam for deterministic launch-failure recovery tests."""

        worker.start()

    def _launch_native_control_worker(
        self,
        cache_key: str,
        run_id: str,
        action: str,
        native_request: dict[str, Any],
        runtime: Any,
        native_marker: tuple[int, object],
    ) -> bool:
        """Launch a native drain, falling back to an in-request takeover.

        Thread creation can fail transiently under process resource pressure.
        Retrying with a fresh Thread avoids reusing a partially-started object;
        if process-local thread creation remains unavailable, the request thread
        executes the same idempotent drain synchronously so ownership can never
        become a permanent orphan.  ``True`` means that fallback completed
        inline and the caller may return the completed receipt directly.
        """

        args = (
            cache_key,
            run_id,
            action,
            native_request,
            runtime,
            native_marker,
        )
        for _attempt in range(2):
            worker = threading.Thread(
                target=self._finish_native_control,
                args=args,
                name=f"ecology-control-{action}-{run_id[:8]}",
                daemon=True,
            )
            try:
                self._start_native_control_thread(worker)
            except Exception:
                continue
            return False
        self._finish_native_control(*args)
        return True

    def _finish_native_control(
        self,
        cache_key: str,
        run_id: str,
        action: str,
        native_request: dict[str, Any],
        runtime: Any,
        native_marker: tuple[int, object],
    ) -> None:
        """Complete a native pause/cancel receipt after remote quiescence."""

        error_code = None
        error_message = None
        try:
            quiescence_error = self.server.auto_progress.finish_native_quiescence(
                action,
                native_request,
                runtime,
                native_marker,
            )
            if quiescence_error is not None:
                raise quiescence_error
            generation_barrier = self.server.acquire_generation_lease(run_id)
            if generation_barrier is None:
                raise RuntimeError(
                    "run generation quiescence barrier is unavailable"
                )
            generation_barrier.release()
            self.server.clear_native_control_inflight(
                run_id,
                native_marker,
            )
        except Exception as exc:  # remote drain errors remain observable
            error_code = safe_error_code(getattr(exc, "error_code", None)) or type(exc).__name__
            error_message = _public_http_error(exc)
        try:
            payload = _control_payload(
                self.server.director.state(run_id),
                self.server.sample_admission.snapshot(run_id),
            )
            payload.update(
                {
                    "command_id": cache_key,
                    "command_status": "completed",
                    "accepted": True,
                    "remote_quiesced": error_code is None,
                }
            )
            if error_code is not None:
                payload.update(
                    {
                        "remote_quiesce_error_code": error_code,
                        "remote_quiesce_error": error_message,
                    }
                )
            self.server.ledger.complete_command(cache_key, payload)
        except Exception:
            # The Host state boundary is already safe. Leave the receipt
            # pending for startup recovery or an explicit same-key retry.
            return
