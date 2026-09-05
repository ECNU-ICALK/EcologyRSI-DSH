"""Minimal, replayable core for the EcologyRSI-DSH evolution mode.

The package intentionally has no runtime dependencies outside the Python
standard library.  A :class:`EvolutionDirector` owns the small state machine;
all mutations are persisted as events before a projection is returned.
"""

from .core.director import EvolutionDirector, RunState
from .core.ledger import Event, EventLedger
from .core.trajectory import (
    FormalBatchArm,
    FormalBatchComparison,
    FormalBatchComparisonDecision,
)
from .core.models import (
    Candidate,
    CandidateRole,
    CandidateStatus,
    Evaluation,
    ExpertConsultation,
    ExpertConsultationAnswer,
    ExpertUncertaintyType,
    HumanIntervention,
    InterventionKind,
    ModelArtifact,
    Promotion,
    PromotionDecision,
    Proposal,
    Run,
    RunStatus,
    TaskManifest,
)
from .data.toy import Observation, ToyCropSoilWater
from .evolution.strategies import (
    DSHAdapter,
    FakeDSHAdapter,
    StrategyRouterDSHAdapter,
)
from .integrations.model_gateway import (
    GatewayConfigurationError,
    GatewayResponseError,
    ModelConnection,
    ModelGateway,
)
from .version import __version__

__all__ = [
    "Candidate",
    "CandidateRole",
    "CandidateStatus",
    "DSHAdapter",
    "Evaluation",
    "ExpertConsultation",
    "ExpertConsultationAnswer",
    "ExpertUncertaintyType",
    "HumanIntervention",
    "InterventionKind",
    "Event",
    "EventLedger",
    "EvolutionDirector",
    "FakeDSHAdapter",
    "FormalBatchArm",
    "FormalBatchComparison",
    "FormalBatchComparisonDecision",
    "GatewayConfigurationError",
    "GatewayResponseError",
    "ModelConnection",
    "ModelGateway",
    "ModelArtifact",
    "Observation",
    "Promotion",
    "PromotionDecision",
    "Proposal",
    "Run",
    "RunState",
    "RunStatus",
    "StrategyRouterDSHAdapter",
    "TaskManifest",
    "ToyCropSoilWater",
    "__version__",
]
