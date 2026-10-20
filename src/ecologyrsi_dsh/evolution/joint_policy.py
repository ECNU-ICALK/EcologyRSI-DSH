"""Opt-in identities; historical manifests never acquire new search semantics."""

PROTOCOL = "joint_plugin_epoch@1"
UNIFIED_PROTOCOL = "joint_plugin_epoch@2"
UNIFIED_SCHEDULE_SCHEMA = "ecologyrsi-dsh.joint-plugin-epoch-schedule/2"
PROTOCOLS = (UNIFIED_PROTOCOL,)
SCHEDULE_SCHEMA = "ecologyrsi-dsh.joint-plugin-epoch-schedule/1"
MUTATION_OPERATOR = "joint-plugin-mutation@1"
DIVERSITY_POLICY = "joint_behavior_search@1"
PROGRAM_SCHEMA = "ecologyrsi-dsh.strategy-program/1"
PROGRAM_POLICY = "bounded_strategy_program@1"
INFERENCE_POLICY = "autonomous-plugin-tools@1"


def enabled(metadata):
    return metadata.get("optimization_protocol") in PROTOCOLS


def contract():
    return {"schema_version": "ecologyrsi-dsh.mutation-policy/2",
            "mutation_operator_id": MUTATION_OPERATOR,
            "local_edit_operator_id": "bounded-batch-local-edit@1",
            "maximum_normalized_step": 0.15,
            "coordinate_policy": "log_if_positive_range_ratio_gte_100_else_linear@1",
            "candidate_maximum_operations": 4, "local_maximum_operations": 2,
            "composition": "primary_hypothesis_first_then_related_cross_component_edits",
            "protected": ["harness", "evaluator", "data_partitions", "budgets", "review_gates"]}
