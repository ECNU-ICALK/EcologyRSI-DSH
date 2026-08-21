"""Auditable knowledge retrieval for bounded evolution runs."""

from .algorithm_ir import AlgorithmIR, build_registered_algorithm_ir
from .algorithm_smoke import AlgorithmSmokeError, smoke_test_algorithm_spec
from .algorithms import (
    AlgorithmAttempt,
    AlgorithmCompileError,
    AlgorithmSpec,
    compile_algorithm_spec,
    debug_algorithm_spec,
)
from .models import KnowledgeAssessment, KnowledgeCard, KnowledgeSnapshot
from .program_registry import (
    LEGACY_PROGRAM_CATALOG_0_2_2,
    ProgramRegistrySnapshot,
    current_program_registry,
)
from .research_iteration import ResearchIteration
from .retrieval import assess_generation_knowledge, retrieve_generation_knowledge

__all__ = [
    "AlgorithmAttempt",
    "AlgorithmCompileError",
    "AlgorithmIR",
    "AlgorithmSmokeError",
    "AlgorithmSpec",
    "KnowledgeAssessment",
    "KnowledgeCard",
    "KnowledgeSnapshot",
    "LEGACY_PROGRAM_CATALOG_0_2_2",
    "ProgramRegistrySnapshot",
    "ResearchIteration",
    "assess_generation_knowledge",
    "build_registered_algorithm_ir",
    "compile_algorithm_spec",
    "current_program_registry",
    "debug_algorithm_spec",
    "retrieve_generation_knowledge",
    "smoke_test_algorithm_spec",
]
