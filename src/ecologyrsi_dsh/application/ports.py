"""Dependencies required by generation use cases; no HTTP endpoint contract."""
from __future__ import annotations
from contextlib import AbstractContextManager
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.director import EvolutionDirector
    from ..core.ledger import EventLedger
    from ..data.registry import DatasetRegistry
    from ..evaluators.registry import EvaluatorRegistry
    from ..integrations.dsh_native_runtime import DshNativeAgentRuntimeClient
    from ..integrations.dsh_tools import DshToolService
    from ..integrations.model_gateway import ModelGateway
    from ..execution.sample_admission import RunSampleAdmission
    from ..evolution.strategies import StrategyRouterDSHAdapter
    from .runtime_bindings import ValidatedCandidateIdentityCache


class GenerationRuntime(Protocol):
    director: EvolutionDirector
    ledger: EventLedger
    datasets: DatasetRegistry
    evaluators: EvaluatorRegistry
    mutation_lock: AbstractContextManager
    dsh_native_runtime: DshNativeAgentRuntimeClient | None
    dsh_tools: DshToolService
    sample_admission: RunSampleAdmission
    model_gateway: ModelGateway
    strategy_router: StrategyRouterDSHAdapter
    dsh_identity_cache: ValidatedCandidateIdentityCache
