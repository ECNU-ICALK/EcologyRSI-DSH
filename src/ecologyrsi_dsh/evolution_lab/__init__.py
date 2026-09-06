"""External AI-for-AI plugin evolution control plane.

This package is deliberately independent from the DSH runtime.  It owns
candidate capabilities and promotion state, while adapters call the existing
DSH API as an execution backend.
"""

from .adapters import DshExperimentAdapter, ExperimentBackend
from .capabilities import CapabilityProposal, CapabilityRegistry, CapabilitySpec
from .controller import EvolutionController
from .evaluator import EvaluationReport, PromotionDecision, evaluate_candidate
from .genome import Mutation, MutationPlanner, PluginGenome
from .store import EvolutionStore

__all__ = [
    "CapabilityProposal", "CapabilityRegistry", "CapabilitySpec",
    "DshExperimentAdapter", "ExperimentBackend", "Mutation", "MutationPlanner", "PluginGenome",
    "EvolutionController", "EvaluationReport", "PromotionDecision", "EvolutionStore", "evaluate_candidate",
]
