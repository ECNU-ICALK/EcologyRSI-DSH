"""Model registration, parameter contracts and frozen evaluator identities."""
from __future__ import annotations

from ..core.prediction_policy import RUNTIME_EVALUATOR_ID
import math
from collections.abc import Mapping
from typing import Any
from ..core.models import TaskManifest, digest
from ..core.sample_results import SAMPLE_REWARD_DEFINITION
from ..data.adapters import DatasetAdapter, dataset_adapter
from ..evolution.promotion import (
    PROMOTION_BLOCK_EVIDENCE_VERSION,
    PROMOTION_BLOCK_HOURS,
    PROMOTION_BOOTSTRAP_RESAMPLES,
    PROMOTION_CONFIDENCE_LEVEL,
    PROMOTION_CONFIDENCE_METHOD,
    PROMOTION_MINIMUM_PAIRED_BLOCKS,
    PROMOTION_POLICY_VERSION,
    V2_MINIMUM_SCORE_DELTA,
)
from .fitness import FitnessProfile
from .noninferiority import (
    CELL_NONINFERIORITY_GATE_ID,
    CELL_NONINFERIORITY_STATISTIC,
    EVIDENCE_SUFFICIENCY_GATE_ID,
    RELAXATION_DEBT_GATE_ID,
    NoninferiorityPolicy,
)
from .baselines import BASELINE_PROFILE_VERSION, BASELINE_SELECTION_TOLERANCE
from .greenhouse_prediction import (
    BASELINE_ALIGNED_RIDGE_MODEL_ID,
    EXOGENOUS_RIDGE_MODEL_ID,
    HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
    RECIPE_FEATURE_POLICY_ID,
    RECIPE_RIDGE_MODEL_ID,
    TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
)
from .metrics import NORMALIZATION_SCALE_METHOD
from .objectives import (
    DEFAULT_TARGET_WEIGHTS,
    OBJECTIVE_AGGREGATION_VERSION,
    OBJECTIVE_COMPONENT_BOUND,
    OBJECTIVE_MISSING_PENALTY,
)
from .sample_execution import DEFAULT_SAMPLE_EXECUTION_MIN_COVERAGE

TOY_DATASET_ID = "generated-toy-series@1"


GREENHOUSE_EVALUATOR_ID = "greenhouse_time_forward@1"


GREENHOUSE_MULTIHORIZON_EVALUATOR_ID = "greenhouse_multihorizon_time_forward@1"


GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID = (
    "greenhouse_multihorizon_time_forward@2"
)


GREENHOUSE_MULTIHORIZON_EVALUATOR_V3_ID = (
    "greenhouse_multihorizon_time_forward@3"
)


GREENHOUSE_RECIPE_EVALUATOR_ID = "greenhouse_recipe_multihorizon_forward@1"


TOY_EVALUATOR_ID = "toy_time_forward@1"


TOY_PREDICTOR_MODEL_ID = "toy-rolling-water@1"


GREENHOUSE_ROLLING_PREDICTOR_ID = "greenhouse-rolling-residual@1"


GREENHOUSE_OBJECTIVE_PROFILE_ID = "greenhouse_equal_weight_skill@1"


GREENHOUSE_OBJECTIVE_AGGREGATION_VERSION = OBJECTIVE_AGGREGATION_VERSION


GREENHOUSE_OBJECTIVE_MISSING_PENALTY = OBJECTIVE_MISSING_PENALTY


GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS = dict(DEFAULT_TARGET_WEIGHTS)


GREENHOUSE_MIN_SKILL_EXCLUSIVE = 1e-9


GREENHOUSE_NO_REGRESSION_TOLERANCE = 1e-12


GREENHOUSE_MAX_CONSTRAINT_VIOLATIONS = 0


def _noninferiority_policy(adapter: DatasetAdapter | None) -> NoninferiorityPolicy:
    """Resolve the per-cell gate's constants from the dataset contract.

    The defaults live in ``NoninferiorityPolicy`` so a dataset that predates the
    two new adapter fields still describes a complete gate; the adapter is the
    override, not the source.
    """

    default = NoninferiorityPolicy()
    if adapter is None:
        return default
    return NoninferiorityPolicy(
        hard_cap_ratio=float(
            getattr(adapter, "cell_regression_hard_cap", default.hard_cap_ratio)
        ),
        alpha=float(
            getattr(adapter, "cell_noninferiority_alpha", default.alpha)
        ),
    )


def _greenhouse_hard_gates(adapter: DatasetAdapter | None = None) -> list[dict[str, Any]]:
    """Describe the exact evaluator checks in a machine-readable form."""

    policy = _noninferiority_policy(adapter)
    return [
        {
            "id": "positive_overall_skill",
            "scope": "overall",
            "metric": "objective_score",
            "operator": ">",
            "threshold": adapter.minimum_skill if adapter else GREENHOUSE_MIN_SKILL_EXCLUSIVE,
        },
        {
            # Ordered before the per-cell gate on purpose. A per-cell interval
            # widens as paired blocks are lost, so a candidate could otherwise
            # buy tolerance by shrinking its own cohort. Sufficiency is decided
            # first, on the cohort's shape alone.
            "id": EVIDENCE_SUFFICIENCY_GATE_ID,
            "scope": "cohort",
            "metric": "paired_block_count",
            "operator": ">=",
            "threshold": policy.minimum_paired_blocks,
            "companion_metric": "valid_three_day_start_count",
            "companion_operator": ">=",
            "companion_threshold": policy.minimum_valid_three_day_starts,
            "moving_block_days": policy.moving_block_days,
        },
        {
            "id": CELL_NONINFERIORITY_GATE_ID,
            "scope": "per_prediction_cell",
            "reduction": "all",
            "statistic": CELL_NONINFERIORITY_STATISTIC,
            "metric": "d_lcb",
            "operator": "<=",
            "threshold": 0.0,
            "absolute_cap_metric": "d",
            "absolute_cap_operator": "<=",
            "absolute_cap_ratio": policy.hard_cap_ratio,
            "absolute_cap_reference_metric": "nrmse_base",
            "confidence_level": policy.confidence_level,
            "bootstrap_resamples": policy.bootstrap_resamples,
            "minimum_paired_blocks": policy.minimum_paired_blocks,
            "prerequisite_gate": EVIDENCE_SUFFICIENCY_GATE_ID,
            # The rule this replaces, named so the audit trail is explicit about
            # what changed rather than leaving it to be inferred from a diff.
            "replaces": "all_targets_no_regression",
            "legacy_tolerance": adapter.no_regression_tolerance
            if adapter
            else GREENHOUSE_NO_REGRESSION_TOLERANCE,
        },
        {
            # Conditional, and deliberately so. It is dormant for a candidate
            # whose every cell is strictly better, and escalates the overall
            # requirement only for one that drew on the per-cell relaxation.
            # Declared after the gate it is conditioned on so the ordering in
            # this list reads as the implication it is.
            "id": RELAXATION_DEBT_GATE_ID,
            "scope": "overall",
            "metric": "objective_score",
            "operator": ">",
            "threshold": adapter.selection_minimum_score_delta
            if adapter
            else V2_MINIMUM_SCORE_DELTA,
            "applies_when": "gate_relaxation_audit.requires_stronger_overall_evidence",
            "conditioned_on_gate": CELL_NONINFERIORITY_GATE_ID,
            "escalates_gate": "positive_overall_skill",
        },
        {
            "id": "no_constraint_violations",
            "scope": "aggregate",
            "metric": "constraint_violations",
            "operator": "<=",
            "threshold": adapter.maximum_constraint_violations if adapter else GREENHOUSE_MAX_CONSTRAINT_VIOLATIONS,
        },
        {
            "id": "minimum_sample_execution_coverage",
            "scope": "overall_and_per_prediction_task",
            "metric": "sample_execution_coverage",
            "operator": ">=",
            "threshold": adapter.minimum_coverage if adapter else DEFAULT_SAMPLE_EXECUTION_MIN_COVERAGE,
        },
    ]


def _greenhouse_scoring_contract(adapter: DatasetAdapter | None = None) -> dict[str, Any]:
    """Return every constant that can change scoring or promotion semantics."""

    return {
        "objective_aggregation_version": GREENHOUSE_OBJECTIVE_AGGREGATION_VERSION,
        "baseline_profile_version": BASELINE_PROFILE_VERSION,
        "reward_definition": SAMPLE_REWARD_DEFINITION,
        "target_weights": adapter.target_weights if adapter else dict(GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS),
        "horizon_weighting": "equal",
        "missing_task_penalty": GREENHOUSE_OBJECTIVE_MISSING_PENALTY,
        "objective_component_bound": OBJECTIVE_COMPONENT_BOUND,
        "normalization_scale_method": NORMALIZATION_SCALE_METHOD,
        "baseline_selection_tolerance": BASELINE_SELECTION_TOLERANCE,
        "minimum_practical_score_delta": adapter.selection_minimum_score_delta if adapter else V2_MINIMUM_SCORE_DELTA,
        "promotion_policy_version": PROMOTION_POLICY_VERSION,
        "promotion_evidence_schema_version": PROMOTION_BLOCK_EVIDENCE_VERSION,
        "confidence_method": PROMOTION_CONFIDENCE_METHOD,
        "confidence_level": PROMOTION_CONFIDENCE_LEVEL,
        "minimum_paired_blocks": PROMOTION_MINIMUM_PAIRED_BLOCKS,
        "block_hours": PROMOTION_BLOCK_HOURS,
        "bootstrap_resamples": PROMOTION_BOOTSTRAP_RESAMPLES,
        # The per-cell gate's constants ride into `evaluator_digest` here, so
        # `_common_contract_matches` refuses to compare a candidate judged under
        # these semantics against one judged under the retired 1e-12 rule. That
        # machinery already exists; it only had to be told about the new numbers.
        "cell_noninferiority": _noninferiority_policy(adapter).to_dict(),
        "hard_gates": _greenhouse_hard_gates(adapter),
    }



class EvaluatorCatalog:
    """Pure catalog and binding validation, independent of execution lifecycle."""

    def catalog(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        items = [
            {
                "id": TOY_EVALUATOR_ID,
                "label": "合成水分时间前向评测",
                "description": "仅用于工程演示的固定验证分区评测。",
                "dataset_ids": [TOY_DATASET_ID],
                "evaluation_partition": "validation",
                "scientific_scope": "prediction_demo_non_causal",
                "prediction_model_ids": [TOY_PREDICTOR_MODEL_ID],
                "horizons_hours": [1],
                "objective_profile": "toy_validation_skill@1",
                "implementation": "toy-forward-split/6",
            },
            {
                "id": GREENHOUSE_EVALUATOR_ID,
                "label": "温室环境时间前向评测",
                "description": "在固定训练反馈 cohort 中逐样本协作预测，允许受控失败并与仅由训练拟合分区选择的强基线比较。",
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "evaluation_partition": "training_feedback",
                "scientific_scope": "historical_replay_prediction_non_causal",
                "prediction_model_ids": [
                    GREENHOUSE_ROLLING_PREDICTOR_ID,
                    EXOGENOUS_RIDGE_MODEL_ID,
                    TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
                ],
                "horizons_hours": [1],
                "objective_profile": GREENHOUSE_OBJECTIVE_PROFILE_ID,
                "implementation": "greenhouse-one-hour-forward/8",
            },
            {
                "id": GREENHOUSE_MULTIHORIZON_EVALUATOR_ID,
                "label": "温室环境多时距时间前向评测",
                "description": "仅在训练拟合分区学习，并在固定训练反馈 cohort 中逐样本协作评测 1、6、24 小时预测。",
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "evaluation_partition": "training_feedback",
                "scientific_scope": "historical_replay_prediction_non_causal",
                "prediction_model_ids": [
                    EXOGENOUS_RIDGE_MODEL_ID,
                    TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
                ],
                "horizons_hours": [1, 6, 24],
                "objective_profile": GREENHOUSE_OBJECTIVE_PROFILE_ID,
                "implementation": "greenhouse-multihorizon-forward/7",
            },
            {
                "id": GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID,
                "label": "温室环境多时距时间前向评测 v2",
                "description": (
                    "仅在训练拟合分区学习，并在固定训练反馈 cohort 中逐样本协作"
                    "评测 1、6、24 小时预测；支持目标与时距独立残差修正。"
                ),
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "evaluation_partition": "training_feedback",
                "scientific_scope": "historical_replay_prediction_non_causal",
                "prediction_model_ids": [
                    EXOGENOUS_RIDGE_MODEL_ID,
                    TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
                    HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
                ],
                "horizons_hours": [1, 6, 24],
                "objective_profile": GREENHOUSE_OBJECTIVE_PROFILE_ID,
                "implementation": "greenhouse-multihorizon-forward/8",
            },
            {
                "id": RUNTIME_EVALUATOR_ID,
                "label": "温室多方案统一时间前向评测",
                "description": "固定数据、基线、目标和时距；模型在运行中选择是否启用残差模型、切换已登记预测器并优化参数。",
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "evaluation_partition": "training_feedback",
                "scientific_scope": "historical_replay_prediction_non_causal",
                "prediction_model_ids": [BASELINE_ALIGNED_RIDGE_MODEL_ID, EXOGENOUS_RIDGE_MODEL_ID,
                                         TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID, HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID],
                "horizons_hours": [1, 6, 24],
                "objective_profile": GREENHOUSE_OBJECTIVE_PROFILE_ID,
                "implementation": "greenhouse-runtime-model-selection-forward/1",
            },
            {
                "id": GREENHOUSE_MULTIHORIZON_EVALUATOR_V3_ID,
                "label": "温室基线对齐多时距评测 v3",
                "description": (
                    "仅用训练拟合分区选择持续性或24小时季节基底并拟合岭回归残差，"
                    "在固定训练反馈 cohort 中评测1、6、24小时预测。"
                ),
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "evaluation_partition": "training_feedback",
                "scientific_scope": "historical_replay_prediction_non_causal",
                "prediction_model_ids": [BASELINE_ALIGNED_RIDGE_MODEL_ID],
                "horizons_hours": [1, 6, 24],
                "objective_profile": GREENHOUSE_OBJECTIVE_PROFILE_ID,
                "implementation": "greenhouse-baseline-aligned-multihorizon-forward/1",
            },
            {
                "id": GREENHOUSE_RECIPE_EVALUATOR_ID,
                "label": "温室声明式配方多时距评测",
                "description": (
                    "与基线对齐评测共用基底选择、cohort 与分块集合；候选可在基线"
                    "对齐岭回归与声明式特征配方之间切换，因此配方层带来的改进可"
                    "以在同一评测下直接归因。"
                ),
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "evaluation_partition": "training_feedback",
                "scientific_scope": "historical_replay_prediction_non_causal",
                "prediction_model_ids": [
                    BASELINE_ALIGNED_RIDGE_MODEL_ID,
                    RECIPE_RIDGE_MODEL_ID,
                ],
                "horizons_hours": [1, 6, 24],
                "objective_profile": GREENHOUSE_OBJECTIVE_PROFILE_ID,
                "implementation": "greenhouse-recipe-multihorizon-forward/1",
            },
        ]
        adapter = dataset_adapter(dataset_id) if dataset_id and dataset_id != TOY_DATASET_ID else None
        if dataset_id is not None:
            items = [item for item in items if dataset_id in item["dataset_ids"]]
        for item in items:
            if adapter is not None:
                item["targets"] = [target.name for target in adapter.targets]
                item["task_adapter"] = adapter.contract()
                if item["id"] == adapter.evaluator_id:
                    item["horizons_hours"] = list(adapter.horizons_hours)
                    item["label"] = adapter.label + "评测"
            profile = self._fitness_profile_for_catalog_item(item, adapter)
            target_count = len(profile.expected_targets)
            item["prediction_task_count"] = max(
                1,
                target_count * len(item.get("horizons_hours", ())),
            )
            item["minimum_samples_per_update"] = item["prediction_task_count"]
            from ..evolution.parameters import run_parameter_contract
            from ..evolution.schedule import OptimizationSchedule
            minimum_origins = profile.minimum_origins_for_schedule(OptimizationSchedule.for_new_run())
            item["minimum_selection_samples_per_update"] = (
                minimum_origins * profile.prediction_cell_count
            )
            item["minimum_selection_origin_samples_per_update"] = (
                minimum_origins
            )
            item["run_parameters"] = run_parameter_contract(profile)
            item["prediction_cells_per_origin"] = profile.prediction_cell_count
            item["fitness_profile"] = profile.to_dict()
            item["fitness_profile_digest"] = profile.profile_digest
            digest_payload = {
                "evaluator_id": item["id"],
                "implementation": item.pop("implementation"),
                "evaluation_partition": item["evaluation_partition"],
                "prediction_model_ids": item["prediction_model_ids"],
                "horizons_hours": item["horizons_hours"],
                "fitness_profile": item["fitness_profile"],
            }
            if item.get("objective_profile") == GREENHOUSE_OBJECTIVE_PROFILE_ID:
                digest_payload["scoring_contract"] = _greenhouse_scoring_contract(adapter)
            if adapter is not None:
                digest_payload["dataset_task_digest"] = adapter.contract()["contract_digest"]
            item["configuration_digest"] = digest(digest_payload)
        return items


    @staticmethod
    def _fitness_profile_for_catalog_item(
        item: Mapping[str, Any],
        adapter: DatasetAdapter | None = None,
    ) -> FitnessProfile:
        targets = (
            tuple(item.get("targets") or GREENHOUSE_OBJECTIVE_TARGET_WEIGHTS)
            if item.get("objective_profile") == GREENHOUSE_OBJECTIVE_PROFILE_ID
            else ("soil_water",)
        )
        return FitnessProfile(
            expected_targets=targets,
            expected_horizons=tuple(int(value) for value in item["horizons_hours"]),
            **({"selection_minimum_coverage": adapter.selection_minimum_coverage,
                "selection_minimum_score_delta": adapter.selection_minimum_score_delta} if adapter else {}),
        )


    def predictor_catalog(self) -> list[dict[str, Any]]:
        items = [
            {
                "id": TOY_PREDICTOR_MODEL_ID,
                "label": "合成水分滚动预测模型",
                "description": "仅用于合成作物—土壤—水分工程演示。",
                "dataset_ids": [TOY_DATASET_ID],
                "parameter_names": ["alpha", "window", "water_threshold"],
                "scientific_scope": "prediction_demo_non_causal",
                "implementation": "toy-rolling-water/1",
            },
            {
                "id": GREENHOUSE_ROLLING_PREDICTOR_ID,
                "label": "温室持续性偏差预测模型",
                "description": "使用目标变量历史滚动值和训练拟合分区偏差进行 1 小时预测。",
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "parameter_names": ["blend", "window", "bias_scale"],
                "scientific_scope": "historical_replay_prediction_non_causal",
                "implementation": "greenhouse-rolling-residual/2",
            },
            {
                "id": EXOGENOUS_RIDGE_MODEL_ID,
                "label": "温室外生变量岭回归残差模型",
                "description": "融合温室外气象、设定值、动作和根区观测，预测相对持续性基线的残差。",
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "parameter_names": ["history_steps", "ridge_alpha", "residual_scale"],
                "scientific_scope": "historical_replay_prediction_non_causal",
                "implementation": "greenhouse-exogenous-ridge/1",
            },
            {
                "id": TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
                "label": "温室分目标岭回归残差模型",
                "description": (
                    "复用登记的外生变量岭回归，但分别缩放温度、湿度和 CO2 "
                    "残差；任一缩放为 0 时使用持续性预测。"
                ),
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "parameter_names": [
                    "history_steps",
                    "ridge_alpha",
                    "air_temperature_residual_scale",
                    "relative_humidity_residual_scale",
                    "co2_concentration_residual_scale",
                ],
                "scientific_scope": "historical_replay_prediction_non_causal",
                "implementation": "greenhouse-targetwise-ridge/1",
            },
            {
                "id": HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID,
                "label": "温室分目标分时距岭回归残差模型",
                "description": (
                    "在 1、6、24 小时时距分别缩放温度、湿度和 CO2 残差；"
                    "任一目标时距的缩放为 0 时仅该单元使用持续性预测。"
                ),
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "parameter_names": [
                    "history_steps",
                    "ridge_alpha",
                    "air_temperature_1h_residual_scale",
                    "air_temperature_6h_residual_scale",
                    "air_temperature_24h_residual_scale",
                    "relative_humidity_1h_residual_scale",
                    "relative_humidity_6h_residual_scale",
                    "relative_humidity_24h_residual_scale",
                    "co2_concentration_1h_residual_scale",
                    "co2_concentration_6h_residual_scale",
                    "co2_concentration_24h_residual_scale",
                ],
                "scientific_scope": "historical_replay_prediction_non_causal",
                "implementation": "greenhouse-horizon-targetwise-ridge/1",
            },
            {
                "id": BASELINE_ALIGNED_RIDGE_MODEL_ID,
                "label": "温室基线对齐岭回归残差模型",
                "description": (
                    "仅用训练拟合分区为每个目标和时距选定持续性或24小时季节基底，"
                    "训练和预测使用相同基底；残差缩放为0时还原选定因果基线。"
                ),
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                "parameter_names": ["history_steps", "ridge_alpha",
                                    "residual_scale_1h", "residual_scale_6h", "residual_scale_24h"],
                "scientific_scope": "historical_replay_prediction_non_causal",
                "implementation": "greenhouse-baseline-aligned-ridge/1",
            },
            {
                "id": RECIPE_RIDGE_MODEL_ID,
                "label": "声明式特征配方岭回归残差模型",
                "description": (
                    "候选提交声明式特征配方（宿主白名单原语），宿主校验、编译并"
                    "仅在训练拟合分区拟合；逐目标时距的残差缩放由配方自身承载，"
                    "因此不再逐字段枚举本任务的目标名。"
                ),
                "dataset_ids": ["agc_cucumber_2018", "agc_tomato_2019"],
                # Every other tunable lives inside the recipe, which the host
                # grammar bounds. Only the scalar the recipe does not own is
                # declared here, so this list stays independent of the 3x3 grid.
                "parameter_names": ["ridge_alpha"],
                "feature_policy_id": RECIPE_FEATURE_POLICY_ID,
                "scientific_scope": "historical_replay_prediction_non_causal",
                "implementation": "greenhouse-recipe-ridge/1",
            },
        ]
        for item in items:
            body = {
                "prediction_model_id": item["id"],
                "implementation": item.pop("implementation"),
                "parameter_names": item["parameter_names"],
                "causal_interpretation": False,
            }
            # A predictor whose tunables live in a declarative recipe is only
            # identified once the governing grammar is named, so the policy id
            # joins its fingerprint. Predictors without one keep the historical
            # body byte-for-byte, so their configuration_digest does not move.
            if "feature_policy_id" in item:
                body["feature_policy_id"] = item["feature_policy_id"]
            item["configuration_digest"] = digest(body)
        return items


    def default_evaluator(self, dataset_id: str) -> str:
        return (
            TOY_EVALUATOR_ID
            if dataset_id == TOY_DATASET_ID
            else dataset_adapter(dataset_id).evaluator_id
        )


    def default_predictor(self, dataset_id: str) -> str:
        return (
            TOY_PREDICTOR_MODEL_ID
            if dataset_id == TOY_DATASET_ID
            else EXOGENOUS_RIDGE_MODEL_ID
        )


    def minimum_samples_per_update(self, evaluator_id: str, dataset_id: str | None = None) -> int:
        """Return the smallest diagnostic cohort covering every scoring task."""

        for item in self.catalog(dataset_id):
            if item["id"] == evaluator_id:
                return int(item["minimum_samples_per_update"])
        raise ValueError(f"unknown evaluator_id: {evaluator_id}")


    def minimum_selection_samples_per_update(self, evaluator_id: str, dataset_id: str | None = None) -> int:
        """Return the smallest balanced cohort that can pass selection gates."""

        for item in self.catalog(dataset_id):
            if item["id"] == evaluator_id:
                return int(item["minimum_selection_samples_per_update"])
        raise ValueError(f"unknown evaluator_id: {evaluator_id}")


    def fitness_profile(self, evaluator_id: str, dataset_id: str | None = None) -> FitnessProfile:
        """Return the immutable Host-owned fitness profile for an evaluator."""

        for item in self.catalog(dataset_id):
            if item["id"] == evaluator_id:
                raw = item["fitness_profile"]
                if not isinstance(raw, Mapping):
                    raise RuntimeError("evaluator fitness profile is invalid")
                return FitnessProfile(**dict(raw))
        raise ValueError(f"unknown evaluator_id: {evaluator_id}")


    def evaluator_configuration_digest(self, evaluator_id: str, dataset_id: str | None = None) -> str:
        for item in self.catalog(dataset_id):
            if item["id"] == evaluator_id:
                return str(item["configuration_digest"])
        raise ValueError(f"unknown evaluator_id: {evaluator_id}")


    def objective_profile(self, evaluator_id: str, dataset_id: str | None = None) -> dict[str, Any]:
        """Return the frozen optimization objective for UI and audit output."""

        for item in self.catalog(dataset_id):
            if item["id"] != evaluator_id:
                continue
            if item.get("objective_profile") == GREENHOUSE_OBJECTIVE_PROFILE_ID:
                return {
                    "id": GREENHOUSE_OBJECTIVE_PROFILE_ID,
                    **_greenhouse_scoring_contract(dataset_adapter(dataset_id) if dataset_id else None),
                }
            return {
                "id": str(item.get("objective_profile") or "unspecified"),
                "target_weights": None,
                "horizon_weighting": "evaluator_defined",
                "hard_gates": [
                    {
                        "id": "evaluator_passed",
                        "scope": "evaluation",
                        "metric": "passed",
                        "operator": "==",
                        "threshold": True,
                    }
                ],
            }
        raise ValueError(f"unknown evaluator_id: {evaluator_id}")


    def predictor_configuration_digest(self, predictor_model_id: str) -> str:
        for item in self.predictor_catalog():
            if item["id"] == predictor_model_id:
                return str(item["configuration_digest"])
        raise ValueError(f"unknown prediction_model_id: {predictor_model_id}")


    def validate_binding(
        self,
        dataset_id: str,
        evaluator_id: str,
        predictor_model_id: str | None = None,
    ) -> None:
        predictor_model_id = predictor_model_id or self.default_predictor(dataset_id)
        for item in self.catalog():
            if item["id"] == evaluator_id:
                if dataset_id not in item["dataset_ids"]:
                    raise ValueError("evaluator_id is incompatible with dataset_id")
                if predictor_model_id not in item["prediction_model_ids"]:
                    raise ValueError(
                        "prediction_model_id is incompatible with evaluator_id"
                    )
                predictor = next(
                    (
                        model
                        for model in self.predictor_catalog()
                        if model["id"] == predictor_model_id
                    ),
                    None,
                )
                if predictor is None:
                    raise ValueError(
                        f"unknown prediction_model_id: {predictor_model_id}"
                    )
                if dataset_id not in predictor["dataset_ids"]:
                    raise ValueError(
                        "prediction_model_id is incompatible with dataset_id"
                    )
                return
        raise ValueError(f"unknown evaluator_id: {evaluator_id}")


    @staticmethod
    def validate_parameter_overrides(
        task: TaskManifest, overrides: Mapping[str, Any]
    ) -> None:
        if not isinstance(overrides, Mapping):
            raise TypeError("parameter_overrides must be an object")
        greenhouse = "greenhouse" in task.domain_pack.lower()
        predictor_model_id = str(
            task.metadata.get("prediction_model_id")
            or (
                EXOGENOUS_RIDGE_MODEL_ID
                if greenhouse
                else TOY_PREDICTOR_MODEL_ID
            )
        )
        if predictor_model_id == RECIPE_RIDGE_MODEL_ID:
            # Every tunable for this predictor lives inside the declarative
            # recipe, which the host grammar validates. A scalar override has
            # nowhere to land, so it is refused outright rather than silently
            # checked against another predictor's schema by the chain below.
            if overrides:
                raise ValueError(
                    "parameter_overrides is not supported by "
                    f"{RECIPE_RIDGE_MODEL_ID}; change the feature_recipe instead"
                )
            return
        schemas = (
            {
                "history_steps": (int, 1, 12),
                "ridge_alpha": (float, 0.0001, 1.0),
                "residual_scale_1h": (float, 0.0, 1.0),
                "residual_scale_6h": (float, 0.0, 1.0),
                "residual_scale_24h": (float, 0.0, 1.0),
            }
            if predictor_model_id == BASELINE_ALIGNED_RIDGE_MODEL_ID
            else
            {
                "history_steps": (int, 1, 12),
                "ridge_alpha": (float, 0.0001, 1.0),
                "air_temperature_1h_residual_scale": (float, 0.0, 1.0),
                "air_temperature_6h_residual_scale": (float, 0.0, 1.0),
                "air_temperature_24h_residual_scale": (float, 0.0, 1.0),
                "relative_humidity_1h_residual_scale": (float, 0.0, 1.0),
                "relative_humidity_6h_residual_scale": (float, 0.0, 1.0),
                "relative_humidity_24h_residual_scale": (float, 0.0, 1.0),
                "co2_concentration_1h_residual_scale": (float, 0.0, 1.0),
                "co2_concentration_6h_residual_scale": (float, 0.0, 1.0),
                "co2_concentration_24h_residual_scale": (float, 0.0, 1.0),
            }
            if predictor_model_id == HORIZON_TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
            else
            {
                "history_steps": (int, 1, 12),
                "ridge_alpha": (float, 0.0001, 1.0),
                "air_temperature_residual_scale": (float, 0.0, 1.0),
                "relative_humidity_residual_scale": (float, 0.0, 1.0),
                "co2_concentration_residual_scale": (float, 0.0, 1.0),
            }
            if predictor_model_id == TARGETWISE_EXOGENOUS_RIDGE_MODEL_ID
            else
            {
                "history_steps": (int, 1, 12),
                "ridge_alpha": (float, 0.0001, 1.0),
                "residual_scale": (float, 0.0, 1.0),
            }
            if predictor_model_id == EXOGENOUS_RIDGE_MODEL_ID
            else {
                "blend": (float, 0.0, 1.0),
                "window": (int, 1, 48),
                "bias_scale": (float, 0.0, 2.0),
            }
            if greenhouse
            else {
                "alpha": (float, 0.05, 0.95),
                "window": (int, 1, 30),
                "water_threshold": (float, 0.05, 0.85),
            }
        )
        unknown = set(overrides) - set(schemas)
        if unknown:
            raise ValueError(
                "parameter_overrides contains unsupported fields: "
                + ", ".join(sorted(unknown))
            )
        for name, value in overrides.items():
            expected, minimum, maximum = schemas[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"parameter_overrides.{name} must be numeric")
            if expected is int and not isinstance(value, int):
                raise ValueError(f"parameter_overrides.{name} must be an integer")
            if (
                not math.isfinite(float(value))
                or not minimum <= float(value) <= maximum
            ):
                raise ValueError(
                    f"parameter_overrides.{name} is outside the allowed range"
                )
