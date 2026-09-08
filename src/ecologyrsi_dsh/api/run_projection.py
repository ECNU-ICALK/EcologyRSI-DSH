"""Shared run configuration projection builders."""

from __future__ import annotations

from typing import Any

from ..evaluators.registry import (
    EXOGENOUS_RIDGE_MODEL_ID,
    GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID,
    TOY_DATASET_ID,
    TOY_EVALUATOR_ID,
    TOY_PREDICTOR_MODEL_ID,
)
from ..integrations.model_bindings import HOST_PARAMETER_GENERATOR_ID, RULE_JUDGE_ID


def build_configuration(task: Any, state: Any, profile: str = "full") -> dict[str, object]:
    """Build the one public configuration shape used by every run view.

    ``profile`` is retained for callers while deliberately not changing the
    key set: summary/detail consumers must remain schema-compatible.
    """
    metadata = dict(getattr(task, "metadata", {}) or {})
    dataset_id = task.visible_datasets[0] if task.visible_datasets else None
    autonomous_plan = metadata.get("autonomous_plan")
    token_budget_scope = metadata.get("token_budget_scope")
    if token_budget_scope is None and metadata.get("sample_token_budget_policy") == "hard_gateway_call_reservation@1":
        budget = getattr(task, "budget", {})
        try:
            has_limit = float(budget.get("token_limit", 0) or 0) > 0
        except (TypeError, ValueError):
            has_limit = False
        if has_limit:
            token_budget_scope = "sample_agent_gateway_calls_only@1"
    return {
        "execution_protocol": metadata.get("execution_protocol", "dsh_native_plugin_evolution@1"),
        "optimization_protocol": metadata.get("optimization_protocol"),
        "optimization_schedule": metadata.get("optimization_schedule"),
        "derived_execution_budget": metadata.get("derived_execution_budget"),
        "derived_run_execution_budget": metadata.get("derived_run_execution_budget"),
        "cohort_capacity_report": metadata.get("cohort_capacity_report"),
        "cohort_capacity_enforced": metadata.get("cohort_capacity_enforced"),
        "host_runtime_build": metadata.get("host_runtime_build"),
        "domain_pack_id": task.domain_pack,
        "dataset_id": dataset_id,
        "dataset_task": metadata.get("dataset_task"),
        "dataset_task_digest": metadata.get("dataset_task_digest"),
        "episode_id": metadata.get("episode_id"),
        "strategy_id": metadata.get("strategy_id", "parameter_sweep@1"),
        "strategy_digest": metadata.get("strategy_digest"),
        "prediction_model_id": metadata.get("prediction_model_id", TOY_PREDICTOR_MODEL_ID if dataset_id == TOY_DATASET_ID else EXOGENOUS_RIDGE_MODEL_ID),
        "prediction_model_digest": metadata.get("prediction_model_digest"),
        "prediction_selection": metadata.get("prediction_selection"),
        "evaluator_id": metadata.get("evaluator_id", TOY_EVALUATOR_ID if dataset_id == TOY_DATASET_ID else GREENHOUSE_MULTIHORIZON_EVALUATOR_V2_ID),
        "evaluator_digest": metadata.get("evaluator_digest"),
        "objective_profile": metadata.get("objective_profile"),
        "fitness_profile": metadata.get("fitness_profile"),
        "fitness_profile_digest": metadata.get("fitness_profile_digest"),
        "policy_model_id": metadata.get("policy_model_id", HOST_PARAMETER_GENERATOR_ID),
        "judge_model_id": metadata.get("judge_model_id", RULE_JUDGE_ID),
        "strategy_model_id": metadata.get("strategy_model_id", metadata.get("policy_model_id", HOST_PARAMETER_GENERATOR_ID)),
        "review_model_id": metadata.get("review_model_id", metadata.get("judge_model_id", RULE_JUDGE_ID)),
        "sample_agent_mode": metadata.get("sample_agent_mode", "host_feedback_state_machine"),
        "sample_agent_batch_size": metadata.get("sample_agent_batch_size"),
        "samples_per_update": metadata.get("samples_per_update"),
        "minimum_selection_samples_per_update": metadata.get("minimum_selection_samples_per_update"),
        "minimum_selection_origin_samples_per_update": metadata.get("minimum_selection_origin_samples_per_update"),
        "prediction_cells_per_origin": metadata.get("prediction_cells_per_origin"),
        "sample_agent_protocol": metadata.get("sample_agent_protocol"),
        "sample_budget_class": metadata.get("sample_budget_class"),
        "sample_concurrency": metadata.get("sample_concurrency", 4),
        "candidate_concurrency": metadata.get("candidate_concurrency"),
        "two_stage_evaluation_enabled": metadata.get("two_stage_evaluation_enabled", True),
        "sample_operation_max_tokens": metadata.get("sample_operation_max_tokens"),
        "sample_remote_critic_policy": metadata.get("sample_remote_critic_policy"),
        "sample_planner_prompt_profile": metadata.get("sample_planner_prompt_profile"),
        "sample_truncation_retry_policy": metadata.get("sample_truncation_retry_policy"),
        "sample_token_budget_policy": metadata.get("sample_token_budget_policy"),
        "token_budget_scope": token_budget_scope,
        "run_wide_accounting_complete": False,
        "autonomous_mode": bool(metadata.get("autonomous_mode", False)),
        "auto_progress": metadata.get("auto_progress") is True,
        "auto_progress_policy": metadata.get("auto_progress_policy"),
        "allow_host_fallback": metadata.get("allow_host_fallback") is True,
        "remote_fallback_policy": metadata.get("remote_fallback_policy"),
        "model_selection_policy": metadata.get("model_selection_policy"),
        "model_workflow": metadata.get("model_workflow"),
        "autonomous_plan_execution": metadata.get("autonomous_plan_execution"),
        "research_domain": metadata.get("research_domain", task.domain_pack),
        "research_domain_id": metadata.get("research_domain", task.domain_pack),
        "autonomous_plan": autonomous_plan,
        "autonomous_plan_digest": metadata.get("autonomous_plan_digest"),
        "model_team": autonomous_plan.get("team") if isinstance(autonomous_plan, dict) else None,
        "model_selected_prediction": autonomous_plan.get("prediction_model") if isinstance(autonomous_plan, dict) else None,
        "model_selected_strategy": autonomous_plan.get("strategy") if isinstance(autonomous_plan, dict) else None,
        "policy_model_digest": metadata.get("policy_model_digest"),
        "judge_model_digest": metadata.get("judge_model_digest"),
        "policy_model_binding_source": metadata.get("policy_model_binding_source"),
        "judge_model_binding_source": metadata.get("judge_model_binding_source"),
        "slot": metadata.get("slot"),
        "candidates_per_generation": task.candidates_per_generation,
        "candidates_per_round": task.candidates_per_generation,
        "variants_per_round": task.candidates_per_generation,
        "knowledge_online_enabled": bool(metadata.get("knowledge_online_enabled", False)),
    }
