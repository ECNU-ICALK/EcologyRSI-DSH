"""Minimal, replayable core for the EcologyRSI-DSH evolution mode.

The package intentionally has no runtime dependencies outside the Python
standard library.  A :class:`EvolutionDirector` owns the small state machine;
all mutations are persisted as events before a projection is returned.
"""

from .dsh import DSHAdapter, FakeDSHAdapter, MockDSHAdapter, StrategyRouterDSHAdapter
from .director import EvolutionDirector, RunState
from .ledger import Event, EventLedger
from .model_gateway import (
    GatewayConfigurationError,
    GatewayResponseError,
    ModelConnection,
    ModelGateway,
)
from .models import (
    Candidate,
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
from .toy import Observation, ToyCropSoilWater
from .version import __version__

__all__ = [
    "Candidate",
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
    "GatewayConfigurationError",
    "GatewayResponseError",
    "MockDSHAdapter",
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
