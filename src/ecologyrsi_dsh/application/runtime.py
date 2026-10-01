"""One composition root for HTTP, CLI and background generation use cases."""
from __future__ import annotations
import os
import threading
from datetime import datetime
from ..core.director import EvolutionDirector
from ..core.ledger import EventLedger
from ..core.models import digest
from ..data.registry import DatasetRegistry
from ..evaluators.registry import EvaluatorRegistry
from ..evolution.strategies import StrategyRouterDSHAdapter
from ..integrations.model_gateway import ModelGateway
from ..integrations.dsh_native_runtime import DshNativeAgentRuntimeClient, configured_stage_timeout
from ..integrations.dsh_structured_roles import DshStructuredRoleRuntime
from ..integrations.dsh_tools import DshToolService
from ..execution.sample_admission import RunSampleAdmission
from .runtime_bindings import ValidatedCandidateIdentityCache, dsh_revision_snapshot
from .ports import GenerationRuntime


def initialize_runtime(self: GenerationRuntime, ledger: EventLedger) -> None:
    self.ledger = ledger
    self.dsh_tools = DshToolService(self.ledger)
    def abort_failed_evaluation(run_id: str) -> None:
        self.dsh_tools.close_run_admissions(run_id)
        manager = getattr(self, "auto_progress", None)
        if manager is not None:
            manager.abort_native_evaluation(run_id)

    from ..execution.host_activity import HostActivityMonitor, HOST_INTERRUPTION_EVENT
    def record_host_gap(payload):
        # Run incarnation and terminal checks use the same lock as lifecycle
        # mutations. A removed/recreated run can never receive a late heartbeat.
        with self.mutation_lock:
            for run_id in self.ledger.run_ids(include_archived=False):
                if self.ledger.latest_run_lifecycle_kind(run_id) not in {"RunStarted", "RunResumed"}:
                    continue
                created_events = self.ledger.events_by_kind(run_id, "RunCreated")
                if not created_events:
                    continue
                created = created_events[0]
                # A delayed diagnostic must not be attached to an experiment
                # created after the interruption it describes.
                if datetime.fromisoformat(created.created_at.replace("Z", "+00:00")) > datetime.fromisoformat(payload["resumed_at"]):
                    continue
                self.ledger.append(run_id, HOST_INTERRUPTION_EVENT, payload,
                    event_id="host-gap:" + digest([run_id, created.seq, payload]),
                    expected_run_created_seq=created.seq)
    self.host_activity = HostActivityMonitor(on_gap=record_host_gap)
    self.sample_admission = RunSampleAdmission(on_execution_failure=abort_failed_evaluation,
                                               host_activity=self.host_activity)
    self.datasets = DatasetRegistry()
    self.model_gateway = ModelGateway.from_env(verification_store=self.ledger)
    runtime_origin = os.environ.get("ECOLOGYRSI_DSH_RUNTIME_URL", "").strip()
    runtime_token = os.environ.get("ECOLOGYRSI_DSH_RUNTIME_TOKEN", "").strip()
    self.dsh_native_runtime = (
        DshNativeAgentRuntimeClient(
            runtime_origin,
            token=runtime_token,
            stage_timeout=configured_stage_timeout(),
        )
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
    self.director = EvolutionDirector(self.ledger, self.strategy_router)
    self.dsh_identity_cache = ValidatedCandidateIdentityCache(
        self.ledger,
        self.director.state,
    )
    self.evaluators = EvaluatorRegistry(
        self.datasets,
        self.model_gateway,
        dsh_runtime_provider=lambda: DshStructuredRoleRuntime(
            self.dsh_native_runtime,
            admission=self.dsh_tools,
        ),
        dsh_revision_provider=lambda run_id: dsh_revision_snapshot(
            self.ledger, run_id
        ),
        dsh_identity_provider=self.dsh_identity_cache.get,
        dsh_prediction_tool_binder=self.dsh_tools.bind_prediction_tool,
        origin_admission_provider=self.sample_admission.admit,
        origin_admission_snapshot_provider=self.sample_admission.snapshot,
    )
    self.mutation_lock = threading.RLock()


class ApplicationRuntime:
    def __init__(self, ledger: EventLedger):
        initialize_runtime(self, ledger)
