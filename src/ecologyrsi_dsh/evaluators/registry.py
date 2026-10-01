"""Public evaluator service composed from catalog and execution responsibilities."""
from .catalog import EvaluatorCatalog
from .pipeline import EvaluationPipeline
from .metrics import artifact_set_digest
from .catalog import (
    BASELINE_ALIGNED_RIDGE_MODEL_ID,
    GREENHOUSE_MULTIHORIZON_EVALUATOR_V3_ID,
    DEFAULT_SAMPLE_EXECUTION_MIN_COVERAGE,
    EXOGENOUS_RIDGE_MODEL_ID,
    HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
    TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
    GREENHOUSE_EVALUATOR_ID,
    GREENHOUSE_MULTIHORIZON_EVALUATOR_ID,
    GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID,
    GREENHOUSE_ROLLING_PREDICTOR_ID,
    GREENHOUSE_RECIPE_EVALUATOR_ID,
    RECIPE_RIDGE_MODEL_ID,
    TOY_EVALUATOR_ID,
    TOY_PREDICTOR_MODEL_ID,
)
from .pipeline import (
    RULE_JUDGE_ID,
    CollaborativeSampleExecutor,
    EvaluationBundle,
    SampleExecutionPolicy,
)


class EvaluatorRegistry(EvaluatorCatalog, EvaluationPipeline):
    """Registered evaluators with one shared scientific execution pipeline."""


__all__ = [
    "BASELINE_ALIGNED_RIDGE_MODEL_ID",
    "GREENHOUSE_MULTIHORIZON_EVALUATOR_V3_ID",
    "DEFAULT_SAMPLE_EXECUTION_MIN_COVERAGE",
    "EXOGENOUS_RIDGE_MODEL_ID",
    "HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID",
    "TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID",
    "GREENHOUSE_EVALUATOR_ID",
    "GREENHOUSE_MULTIHORIZON_EVALUATOR_ID",
    "GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID",
    "GREENHOUSE_ROLLING_PREDICTOR_ID",
    "GREENHOUSE_RECIPE_EVALUATOR_ID",
    "RECIPE_RIDGE_MODEL_ID",
    "RULE_JUDGE_ID",
    "TOY_EVALUATOR_ID",
    "TOY_PREDICTOR_MODEL_ID",
    "CollaborativeSampleExecutor",
    "EvaluationBundle",
    "EvaluatorRegistry",
    "SampleExecutionPolicy",
    "artifact_set_digest",
]
