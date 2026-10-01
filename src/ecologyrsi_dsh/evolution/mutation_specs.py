"""One mutation catalog for shape validation, research axes and local edit targets."""
from __future__ import annotations

from dataclasses import dataclass

SKILL_POLICY_ID = "causal_planner_skills@1"
AUTHORED_DIRECTIVE_POLICY_ID = "authored_directive@1"
RECIPE_FEATURE_POLICY_ID = "authored_causal_features@1"


@dataclass(frozen=True)
class MutationSpec:
    axis: str
    fields: str
    target: str
    path: str
    directions: tuple[str, ...] = ("select",)
    roles: tuple[str, ...] = ()

    def coordinates(self, operation):
        values = {field: str(operation.get(field) or "") for field in self.fields.split()}
        return self.axis, self.target.format_map(values), self.path.format_map(values)


_NUMERIC = ("increase", "decrease")
_AUTHORING = ("author", "revise")
_EXECUTION_ROLES = ("sample-planner", "sample-repair")
MUTATION_SPECS = {
    "set_bounded_parameter": MutationSpec(
        "scientific_parameter", "name value", "{name}", "parameter:{name}", _NUMERIC),
    "select_registered_pipeline": MutationSpec(
        "registered_predictor", "predictor_id", "{predictor_id}", "predictor"),
    "select_instruction_template": MutationSpec(
        "instruction_profile", "role instruction_template_id", "{instruction_template_id}",
        "instruction:{role}", roles=_EXECUTION_ROLES),
    "set_instruction_parameter": MutationSpec(
        "instruction_parameter", "role name value", "{name}", "instruction-parameter:{role}:{name}",
        _NUMERIC, _EXECUTION_ROLES),
    "author_role_directive": MutationSpec(
        "instruction_directive", "role authored_directive", AUTHORED_DIRECTIVE_POLICY_ID,
        "instruction-directive:{role}", _AUTHORING, ("sample-planner",)),
    "author_skill_program": MutationSpec(
        "skill_program", "role skill_program", SKILL_POLICY_ID,
        "skill-program:{role}", _AUTHORING, ("sample-planner",)),
    "narrow_role_tool_policy": MutationSpec(
        "instruction_tool_policy", "role enabled_tool_ids", "{role}",
        "tool-policy:{role}", ("narrow",), _EXECUTION_ROLES),
    "select_registered_workflow_template": MutationSpec(
        "workflow_template", "workflow_template_id", "{workflow_template_id}", "workflow"),
    "set_bounded_workflow_parameter": MutationSpec(
        "workflow_parameter", "name value", "{name}", "workflow:{name}", _NUMERIC),
    "select_registered_feature_policy": MutationSpec(
        "feature_policy", "program_id", "{program_id}", "feature_policy"),
    "author_feature_recipe": MutationSpec(
        "feature_recipe", "feature_recipe", RECIPE_FEATURE_POLICY_ID, "feature-recipe", _AUTHORING),
    "select_registered_fit_policy": MutationSpec(
        "fit_policy", "program_id", "{program_id}", "fit_policy"),
    "select_registered_uncertainty_policy": MutationSpec(
        "uncertainty_policy", "program_id", "{program_id}", "uncertainty_policy"),
}


def mutation_spec(op):
    try:
        return MUTATION_SPECS[op]
    except (KeyError, TypeError):
        raise ValueError(f"mutation operation {op or '<missing>'} is not registered") from None


def mutation_coordinates(operation):
    return mutation_spec(operation.get("op")).coordinates(operation)
