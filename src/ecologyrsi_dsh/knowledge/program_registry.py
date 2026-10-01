"""Immutable source-program catalogs for plugin-genome compilation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from ..evolution.genome import (
    FrozenJsonObject,
    SeedGenomeTemplate,
    _domain_digest,
    deep_freeze_json,
    deep_thaw_json,
)
from ..evaluators.authored_directive import (
    AUTHORED_DIRECTIVE_POLICY_ID,
    MAX_DIRECTIVE_RATIONALE_LENGTH,
    MAX_DIRECTIVE_TOOL_PLAN_STEPS,
    directive_grammar,
)
from ..evaluators.feature_recipe import (
    MAX_RECIPE_LAG_HOURS,
    MAX_RECIPE_TERMS,
    MAX_ROLLING_WINDOW_HOURS,
    recipe_grammar,
)
from ..evaluators.greenhouse_prediction import (
    GREENHOUSE_SEED_EXOGENOUS_COLUMNS,
    seed_feature_recipe,
)
from ..evaluators.skill_program import SKILL_POLICY_ID, skill_grammar


REGISTRY_SCHEMA_VERSION = "ecologyrsi-dsh.program-registry/1"

# One Skill file for every sample-side instruction template. It carries only the
# invariant protocol (call budget, evidence citation, the single structured
# output, the label prohibition); the per-template strategy lives in the
# registry's `directive` and is delivered through the candidate agent profile.
SAMPLE_FORECASTING_SKILL = "origin-vector-forecasting"
# Roles whose directive actually reaches the model. Registering a directive for a
# role that has no delivery path would be an interface that cannot change
# anything, so the validator requires one exactly here and nowhere else.
_DIRECTIVE_ROLES = frozenset({"sample-planner", "sample-repair"})
# A registered template's directive and an authored one land in the same prompt
# slot, so they share one ceiling rather than drifting apart. The constant is
# defined next to the authoring grammar because that is where the bound is also
# published to the model.
_MAX_DIRECTIVE_LENGTH = MAX_DIRECTIVE_RATIONALE_LENGTH


def _parameter(
    *, minimum: int | float, maximum: int | float, default: int | float, integer: bool = False
) -> dict[str, Any]:
    return {
        "minimum": minimum,
        "maximum": maximum,
        "default": default,
        "integer": integer,
    }


# `confidence_threshold` is the sample planner's escalation threshold: a decision
# whose reported confidence falls below it is sent to the remote critic for an
# independent look (`DshSampleCollaborationAdapter`, via
# `remote_critic_policy.min_planner_confidence`). So this scalar buys accuracy
# with API calls, and every sample-planner template shares one contract on
# purpose:
#
#   * the bounds are the *cost* window, not the mathematically legal one. The
#     contract could say [0, 1], but the planner's own reported confidences sit
#     in 0.70-0.85 in production, so a threshold of 0.9 escalates nearly every
#     cell. 0.75 keeps escalation sparse; below 0.5 escalation stops happening at
#     all, and one inert value is enough.
#   * the default equals the minimum, and equals the run-level strict policy the
#     greenhouse runs with today (0.5). `select_instruction_template` clears the
#     candidate's `instruction_parameters`, so a template's default becomes the
#     live threshold the moment that template is selected. Giving the templates
#     different defaults would make the `instruction_profile` axis silently move
#     cost as well as strategy, and the score change could not be attributed to
#     either. Identical defaults keep template selection a pure directive change
#     and leave `set_instruction_parameter` as the one axis that moves cost.
def _confidence_threshold_parameter() -> dict[str, Any]:
    return _parameter(minimum=0.5, maximum=0.75, default=0.5)


_CURRENT_PROGRAMS: dict[str, dict[str, dict[str, Any]]] = {
    "predictors": {
        "toy-rolling-water@1": {
            "version": "toy-water-operator-graph/1",
            "parameters": {
                "alpha": _parameter(minimum=0.0, maximum=1.0, default=0.5),
                "window": _parameter(minimum=1, maximum=168, default=5, integer=True),
                "water_threshold": _parameter(
                    minimum=0.0, maximum=1.0, default=0.4
                ),
            },
        },
        "greenhouse-rolling-residual@1": {
            "version": "greenhouse-rolling-operator-graph/1",
            "parameters": {
                "blend": _parameter(minimum=0.0, maximum=1.0, default=0.5),
                "window": _parameter(minimum=1, maximum=168, default=6, integer=True),
                "bias_scale": _parameter(minimum=0.0, maximum=2.0, default=0.5),
            },
        },
        "greenhouse-exogenous-ridge@1": {
            "version": "greenhouse-ridge-operator-graph/1",
            "parameters": {
                "history_steps": _parameter(
                    minimum=1, maximum=168, default=6, integer=True
                ),
                "ridge_alpha": _parameter(
                    minimum=0.000001, maximum=1000.0, default=0.1
                ),
                "residual_scale": _parameter(
                    minimum=0.0, maximum=2.0, default=0.5
                ),
            },
        },
        "greenhouse-baseline-aligned-ridge@1": {
            "version": "greenhouse-baseline-aligned-ridge-operator-graph/1",
            "parameters": {
                "history_steps": _parameter(
                    minimum=1, maximum=12, default=6, integer=True
                ),
                "ridge_alpha": _parameter(
                    minimum=0.0001, maximum=1.0, default=0.1
                ),
                **{
                    f"residual_scale_{horizon}h": _parameter(
                        minimum=0.0, maximum=1.0, default=0.0
                    ) for horizon in (1, 6, 24)
                },
            },
        },
        "greenhouse-targetwise-ridge@1": {
            "version": "greenhouse-targetwise-ridge-operator-graph/1",
            "parameters": {
                "history_steps": _parameter(
                    minimum=1, maximum=168, default=6, integer=True
                ),
                "ridge_alpha": _parameter(
                    minimum=0.000001, maximum=1000.0, default=0.1
                ),
                "air_temperature_residual_scale": _parameter(
                    minimum=0.0, maximum=2.0, default=0.8
                ),
                "relative_humidity_residual_scale": _parameter(
                    minimum=0.0, maximum=2.0, default=0.7
                ),
                "co2_concentration_residual_scale": _parameter(
                    minimum=0.0, maximum=2.0, default=0.0
                ),
            },
        },
        "greenhouse-horizon-targetwise-ridge@1": {
            "version": "greenhouse-horizon-targetwise-ridge-operator-graph/1",
            "parameters": {
                "history_steps": _parameter(
                    minimum=1, maximum=168, default=6, integer=True
                ),
                "ridge_alpha": _parameter(
                    minimum=0.000001, maximum=1000.0, default=0.1
                ),
                **{
                    f"{target}_{horizon}_residual_scale": _parameter(
                        minimum=0.0,
                        maximum=2.0,
                        default=(0.0 if target == "co2_concentration" and horizon == "1h" else 0.8),
                    )
                    for target in (
                        "air_temperature",
                        "relative_humidity",
                        "co2_concentration",
                    )
                    for horizon in ("1h", "6h", "24h")
                },
            },
        },
        "greenhouse-recipe-ridge@1": {
            "version": "greenhouse-recipe-operator-graph/1",
            # Intentionally empty. This predictor has no scalar tunables: the
            # ridge alpha, the baseline anchor and every per-cell residual
            # scale live inside scientific_program.feature_recipe, which the
            # authored_causal_features@1 grammar below bounds. _parameter()
            # entries can only express numeric scalars, so a recipe cannot be
            # described here even in principle.
            "parameters": {},
            "feature_policy_id": "authored_causal_features@1",
        },
    },
    "feature_policies": {
        "registered_greenhouse_features@1": {
            "version": "registered-greenhouse-causal-features/1",
            "parameters": {},
        },
        "registered_toy_features@1": {
            "version": "registered-toy-causal-features/1",
            "parameters": {},
        },
        "authored_causal_features@1": {
            "version": "authored-causal-feature-recipe/1",
            # The scalar ceilings a mutation may legally target. The full
            # primitive whitelist is carried as static grammar below, because
            # per-op bounds are not numeric scalars of the recipe itself.
            "parameters": {
                "max_terms": _parameter(
                    minimum=1, maximum=MAX_RECIPE_TERMS,
                    default=MAX_RECIPE_TERMS, integer=True,
                ),
                "max_lag_hours": _parameter(
                    minimum=1, maximum=MAX_RECIPE_LAG_HOURS,
                    default=MAX_RECIPE_LAG_HOURS, integer=True,
                ),
                "max_rolling_window": _parameter(
                    minimum=2, maximum=MAX_ROLLING_WINDOW_HOURS,
                    default=MAX_ROLLING_WINDOW_HOURS, integer=True,
                ),
            },
            # Content-addressed via _program_digest, so the grammar an Agent
            # was shown is recoverable from the genome's catalog_digest alone.
            "grammar": recipe_grammar(),
        },
    },
    "fit_policies": {
        "time_forward_fit@1": {
            "version": "time-forward-training-fit/1",
            "parameters": {},
        }
    },
    "uncertainty_policies": {
        "none@1": {"version": "no-predictive-uncertainty/1", "parameters": {}},
        "cellwise_time_block_calibrated_residual@1": {
            "version": "cellwise-time-block-calibrated-residual/1",
            "parameters": {
                "alpha": _parameter(minimum=0.01, maximum=0.2, default=0.1)
            },
        },
    },
    "workflow_templates": {
        "candidate-sample-execution@1": {
            "version": "candidate-sample-workflow/1",
            "parameters": {
                "max_concurrent": _parameter(
                    minimum=1, maximum=8, default=4, integer=True
                ),
                "wave_size": _parameter(minimum=1, maximum=32, default=8, integer=True),
                # Floored at the host's own retry depth rather than at 1. The
                # derived execution plan contributes a reliability floor of at
                # least ``min(8, 3 + extra)`` attempts for transient recovery,
                # and the synthesis in ``_sample_execution_policy`` takes the
                # larger of the two, so a candidate asking for 1 or 2 would be
                # overruled every generation. Publishing 3 as the minimum keeps
                # every legal value observable in the evaluation record.
                "max_attempts": _parameter(
                    minimum=3, maximum=8, default=3, integer=True
                ),
            },
            "graph": {
                "nodes": [
                    {
                        "id": "sample-plan",
                        "role": "sample-planner",
                        "script_id": "candidate-sample-plan-wave@1",
                    }
                ],
                "edges": [],
                "allowed_roles": ["sample-planner"],
                "session_policy": "continuable-per-candidate",
            },
        },
        "research-and-propose@1": {
            "version": "research-and-propose-workflow/1",
            "parameters": {},
            "graph": {
                "nodes": [
                    {
                        "id": "research",
                        "role": "researcher",
                        "script_id": "generation-research@1",
                    },
                    {
                        "id": "propose",
                        "role": "candidate-proposer",
                        "script_id": "generation-propose@1",
                    },
                ],
                "edges": [{"from": "research", "to": "propose"}],
                "allowed_roles": ["researcher", "candidate-proposer"],
                "session_policy": "fresh-one-shot-per-role",
            },
        },
    },
    # The Skill file holds only the invariant protocol; the evolvable strategy is
    # this `directive` text, delivered at run time through the candidate agent
    # profile. Growing this category is therefore a registry edit, not a new
    # shipped preset, which is what makes per-generation instruction search
    # possible at all. The model selects a template id and never authors the
    # text: the directive is part of the entry digest and of run provenance.
    "instruction_templates": {
        "sample-planner-balanced@1": {
            "version": "sample-planner-balanced-instruction/3",
            "role": "sample-planner",
            "skill_name": SAMPLE_FORECASTING_SKILL,
            "directive": (
                "Balance available observations and model evidence across all "
                "target-horizon cells."
            ),
            "parameters": {
                "confidence_threshold": _confidence_threshold_parameter()
            },
        },
        "sample-planner-anomaly-aware@1": {
            "version": "sample-planner-anomaly-aware-instruction/3",
            "role": "sample-planner",
            "skill_name": SAMPLE_FORECASTING_SKILL,
            "directive": (
                "Check missing values and unusual current observations; consider "
                "an alternative model or a conservative adjustment when the "
                "evidence supports it."
            ),
            "parameters": {
                "confidence_threshold": _confidence_threshold_parameter()
            },
        },
        "sample-planner-horizon-aware@1": {
            "version": "sample-planner-horizon-aware-instruction/3",
            "role": "sample-planner",
            "skill_name": SAMPLE_FORECASTING_SKILL,
            "directive": (
                "Distinguish short and long horizons. You may choose different "
                "prediction methods or combine model evidence by target and "
                "horizon."
            ),
            "parameters": {
                "confidence_threshold": _confidence_threshold_parameter()
            },
        },
        # The four templates below all route through a prediction tool. Without
        # them the live `instruction_profile` axis was three variations on
        # "decide well", every one of which the planner could satisfy by
        # answering from context alone -- and a genome whose numbers are never
        # executed makes the `scientific_parameter` and `registered_predictor`
        # axes unmeasurable. These make the call part of the strategy, so a
        # parameter edit has a scored consequence.
        "sample-planner-model-anchored@1": {
            "version": "sample-planner-model-anchored-instruction/1",
            "role": "sample-planner",
            "skill_name": SAMPLE_FORECASTING_SKILL,
            "directive": (
                "Anchor on your own evolved model: call candidate-model once for the "
                "whole origin vector and cite it. Submit its value for every cell "
                "unless a causal observation contradicts it, and when you depart from "
                "it say by how much and why in the reason code."
            ),
            "parameters": {
                "confidence_threshold": _confidence_threshold_parameter()
            },
        },
        "sample-planner-residual-blend@1": {
            "version": "sample-planner-residual-blend-instruction/1",
            "role": "sample-planner",
            "skill_name": SAMPLE_FORECASTING_SKILL,
            "directive": (
                "Call candidate-model once and cite it, then blend it with the latest "
                "causal observation: weight the observation more at the short horizon "
                "and the model more as the horizon grows. Report the blend you used "
                "per horizon in the reason code."
            ),
            "parameters": {
                "confidence_threshold": _confidence_threshold_parameter()
            },
        },
        "sample-planner-horizon-split-model@1": {
            "version": "sample-planner-horizon-split-model-instruction/1",
            "role": "sample-planner",
            "skill_name": SAMPLE_FORECASTING_SKILL,
            "directive": (
                "Split the vector by horizon. Cite one candidate-model call and take "
                "its value directly for the longer horizons, where history alone "
                "carries little signal. For the shortest horizon let the observed "
                "trend lead and use the model only to bound the step."
            ),
            "parameters": {
                "confidence_threshold": _confidence_threshold_parameter()
            },
        },
        "sample-planner-tool-comparison@1": {
            "version": "sample-planner-tool-comparison-instruction/1",
            "role": "sample-planner",
            "skill_name": SAMPLE_FORECASTING_SKILL,
            "directive": (
                "Spend the call budget on a comparison: call candidate-model, then "
                "call one registered prediction tool whose parameters you expect to "
                "suit this origin better. Cite both, predict with the one the causal "
                "history supports, and name the disagreement in the reason code so a "
                "later generation can inherit the finding."
            ),
            "parameters": {
                "confidence_threshold": _confidence_threshold_parameter()
            },
        },
        "sample-repair@1": {
            "version": "sample-repair-instruction/2",
            "role": "sample-repair",
            "skill_name": SAMPLE_FORECASTING_SKILL,
            "directive": (
                "Re-derive only the cells the Host reported as missing or "
                "invalid; keep every already-accepted cell unchanged."
            ),
            "parameters": {},
        },
        "researcher@1": {
            "version": "researcher-instruction/1",
            "role": "researcher",
            "skill_name": "autonomous-ecology-research",
            "parameters": {},
        },
        "candidate-proposer@1": {
            "version": "candidate-proposer-instruction/1",
            "role": "candidate-proposer",
            "skill_name": "bounded-plugin-experiment",
            "parameters": {},
        },
    },
    # The authoring counterpart to `instruction_templates` above. There a
    # candidate picks one of eight sentences somebody else wrote; here it writes
    # its own, out of clauses the host can check. The two coexist on purpose: a
    # genome that names no directive policy keeps selecting templates and
    # projects byte-identically to every archived genome.
    "directive_policies": {
        AUTHORED_DIRECTIVE_POLICY_ID: {
            "version": "authored-planner-directive/1",
            # The scalar ceilings a mutation may legally target. The clause
            # whitelist itself is carried as static grammar below, because an
            # enum of anchors is not a numeric scalar of the directive.
            "parameters": {
                "max_tool_plan_steps": _parameter(
                    minimum=1,
                    maximum=MAX_DIRECTIVE_TOOL_PLAN_STEPS,
                    default=MAX_DIRECTIVE_TOOL_PLAN_STEPS,
                    integer=True,
                ),
                "max_rationale_length": _parameter(
                    minimum=1,
                    maximum=MAX_DIRECTIVE_RATIONALE_LENGTH,
                    default=MAX_DIRECTIVE_RATIONALE_LENGTH,
                    integer=True,
                ),
            },
            # Content-addressed via _program_digest, so the grammar an Agent
            # was shown is recoverable from the genome's catalog_digest alone.
            "grammar": directive_grammar(),
        },
    },
    "skill_policies": {
        SKILL_POLICY_ID: {"version": "causal-planner-skills/1", "grammar": skill_grammar()},
    },
    "tool_policies": {
        "sample-planner-tools@1": {
            "version": "sample-planner-tools/1",
            "tool_ids": ["ecology_execute_prediction_tool"],
        },
        "sample-repair-tools@1": {
            "version": "sample-repair-tools/1",
            "tool_ids": [
                "ecology_execute_prediction_tool",
                "ecology_execute_registered_repair_tool",
            ],
        },
    },
}


def _program_digest(category: str, program_id: str, value: Mapping[str, Any]) -> str:
    return _domain_digest(
        "ecologyrsi-dsh/program-registry-entry/1",
        {"category": category, "program_id": program_id, "content": dict(value)},
    )


def _program_ref(
    programs: Mapping[str, Mapping[str, Any]], category: str, program_id: str
) -> dict[str, str]:
    return {
        "id": program_id,
        "catalog_digest": _program_digest(category, program_id, programs[category][program_id]),
    }


def _validate_instruction_templates(
    programs: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    """Keep the evolvable directive well-formed and the Skill surface single.

    Instruction search only works if adding a strategy is a registry edit. That
    holds exactly while every sample-side template shares one Skill file and
    differs only in `directive`, so both halves are enforced here rather than
    left to review.
    """

    raw_templates = programs.get("instruction_templates")
    if not isinstance(raw_templates, Mapping) or not raw_templates:
        raise ValueError("program registry requires instruction templates")
    for template_id, raw_template in raw_templates.items():
        if not isinstance(raw_template, Mapping):
            raise TypeError(f"instruction template {template_id} must be an object")
        role = raw_template.get("role")
        directive = raw_template.get("directive")
        if role not in _DIRECTIVE_ROLES:
            if directive is not None:
                raise ValueError(
                    f"instruction template {template_id} registers a directive for a "
                    "role that has no delivery path"
                )
            continue
        if not isinstance(directive, str) or not directive.strip():
            raise ValueError(
                f"instruction template {template_id} requires a strategy directive"
            )
        if len(directive) > _MAX_DIRECTIVE_LENGTH:
            raise ValueError(
                f"instruction template {template_id} directive exceeds its bound"
            )
        if raw_template.get("skill_name") != SAMPLE_FORECASTING_SKILL:
            raise ValueError(
                f"instruction template {template_id} must use the shared sample Skill"
            )


def _validate_workflow_templates(
    programs: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    raw_templates = programs.get("workflow_templates")
    if not isinstance(raw_templates, Mapping) or not raw_templates:
        raise ValueError("program registry requires workflow templates")
    reviewer_roles = {"sample-critic", "generation-judge"}
    for template_id, raw_template in raw_templates.items():
        if not isinstance(raw_template, Mapping):
            raise TypeError(f"workflow template {template_id} must be an object")
        graph = raw_template.get("graph")
        if not isinstance(graph, Mapping) or set(graph) != {
            "nodes",
            "edges",
            "allowed_roles",
            "session_policy",
        }:
            raise ValueError(f"workflow template {template_id} graph is incomplete")
        nodes = graph["nodes"]
        edges = graph["edges"]
        roles = graph["allowed_roles"]
        if not isinstance(nodes, list) or not nodes:
            raise ValueError(f"workflow template {template_id} requires nodes")
        if not isinstance(edges, list) or not isinstance(roles, list):
            raise TypeError(f"workflow template {template_id} graph arrays are invalid")
        node_ids: list[str] = []
        node_roles: list[str] = []
        for node in nodes:
            if not isinstance(node, Mapping) or set(node) != {"id", "role", "script_id"}:
                raise ValueError(f"workflow template {template_id} node is invalid")
            node_id = str(node["id"]).strip()
            role = str(node["role"]).strip()
            script_id = str(node["script_id"]).strip()
            if not node_id or not role or not script_id:
                raise ValueError(f"workflow template {template_id} node is incomplete")
            node_ids.append(node_id)
            node_roles.append(role)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError(f"workflow template {template_id} node ids must be unique")
        if len(roles) != len(set(str(item) for item in roles)):
            raise ValueError(f"workflow template {template_id} allowed roles must be unique")
        if set(str(item) for item in roles) != set(node_roles):
            raise ValueError(f"workflow template {template_id} allowed roles mismatch")
        if reviewer_roles & set(node_roles) and set(node_roles) - reviewer_roles:
            raise ValueError(
                f"workflow template {template_id} has mixed reviewer privilege"
            )
        if template_id == "candidate-sample-execution@1" and not set(
            node_roles
        ).issubset({"sample-planner", "sample-repair"}):
            raise ValueError("candidate workflow cannot include reviewer privilege")
        if reviewer_roles & set(node_roles) and graph["session_policy"] != "fresh-per-item":
            raise ValueError("reviewer workflow must use fresh-per-item sessions")

        adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
        indegree = {node_id: 0 for node_id in node_ids}
        for edge in edges:
            if not isinstance(edge, Mapping) or set(edge) != {"from", "to"}:
                raise ValueError(f"workflow template {template_id} edge is invalid")
            source = str(edge["from"])
            target = str(edge["to"])
            if source not in adjacency or target not in adjacency:
                raise ValueError(f"workflow template {template_id} edge is unresolved")
            if target not in adjacency[source]:
                adjacency[source].add(target)
                indegree[target] += 1
        ready = [node_id for node_id in node_ids if indegree[node_id] == 0]
        visited = 0
        while ready:
            current = ready.pop()
            visited += 1
            for target in adjacency[current]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited != len(node_ids):
            raise ValueError(f"workflow template {template_id} contains a cycle")


def _agent_program(programs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_execution_program": {
            "workflow_template_ref": _program_ref(
                programs, "workflow_templates", "candidate-sample-execution@1"
            ),
            "workflow_overrides": {
                "max_concurrent": 4,
                "wave_size": 8,
                "max_attempts": 3,
            },
            "role_profiles": [
                {
                    "role": "sample-planner",
                    "preset_id": "ecology-sample-planner-v11",
                    "instruction_template_ref": _program_ref(
                        programs,
                        "instruction_templates",
                        "sample-planner-balanced@1",
                    ),
                    # Deliberately equal to the run-level strict policy's
                    # `min_planner_confidence` (0.5), which is what the host used
                    # before this parameter had a consumer. Generation 0 must
                    # therefore cost exactly what it costs today: connecting the
                    # axis is not allowed to be a silent price increase, and any
                    # escalation-rate change from here on is an evolved choice
                    # that a candidate can be scored for.
                    "instruction_parameters": {"confidence_threshold": 0.5},
                    "response_schema_id": "ecology-sample-predictions@2",
                    "base_tool_policy_id": "sample-planner-tools@1",
                    "enabled_tool_ids": ["ecology_execute_prediction_tool"],
                }
            ],
        },
        "reproduction_program": {
            "workflow_template_ref": _program_ref(
                programs, "workflow_templates", "research-and-propose@1"
            ),
            "workflow_overrides": {},
            "role_template_refs": ["researcher@1", "candidate-proposer@1"],
        },
    }


def _seed_template(
    programs: Mapping[str, Mapping[str, Any]],
    *,
    template_id: str,
    predictor_id: str,
    feature_policy_id: str,
    feature_recipe: Mapping[str, Any] | None = None,
) -> SeedGenomeTemplate:
    predictor = programs["predictors"][predictor_id]
    parameters = {
        name: contract["default"]
        for name, contract in predictor["parameters"].items()
    }
    scientific_program = {
        "predictor_ref": _program_ref(programs, "predictors", predictor_id),
        "parameter_overrides": parameters,
        "feature_policy_ref": {
            **_program_ref(programs, "feature_policies", feature_policy_id),
            "overrides": {},
        },
        "fit_policy_ref": {
            **_program_ref(programs, "fit_policies", "time_forward_fit@1"),
            "overrides": {},
        },
        "uncertainty_policy_ref": {
            **_program_ref(programs, "uncertainty_policies", "none@1"),
            "overrides": {},
        },
    }
    # Absent rather than null when unset: the historical five-key shape has to
    # project byte-identically or every archived genome digest moves.
    if feature_recipe is not None:
        scientific_program["feature_recipe"] = dict(feature_recipe)
    return SeedGenomeTemplate.from_dict(
        {
            "schema_version": "ecologyrsi-dsh.seed-genome-template/1",
            "template_id": template_id,
            "scientific_program": scientific_program,
            "agent_program": _agent_program(programs),
            "evidence_refs": [],
        }
    )


@dataclass(frozen=True, slots=True)
class ProgramRegistrySnapshot:
    """A recursively immutable registry with content-addressed entries."""

    _programs: FrozenJsonObject
    _seed_templates: tuple[SeedGenomeTemplate, ...]
    _catalog_digest: str

    @classmethod
    def from_programs(
        cls,
        programs: Mapping[str, Mapping[str, Mapping[str, Any]]],
        *,
        seed_templates: tuple[SeedGenomeTemplate, ...] | None = None,
    ) -> "ProgramRegistrySnapshot":
        _validate_workflow_templates(programs)
        _validate_instruction_templates(programs)
        frozen = deep_freeze_json(programs)
        if not isinstance(frozen, FrozenJsonObject):
            raise TypeError("program registry must be an object")
        thawed = deep_thaw_json(frozen)
        templates = seed_templates or (
            _seed_template(
                thawed,
                template_id="greenhouse-default@1",
                predictor_id="greenhouse-horizon-targetwise-ridge@1",
                feature_policy_id="registered_greenhouse_features@1",
            ),
            _seed_template(
                thawed,
                template_id="greenhouse-targetwise-default@1",
                predictor_id="greenhouse-targetwise-ridge@1",
                feature_policy_id="registered_greenhouse_features@1",
            ),
            _seed_template(
                thawed,
                template_id="greenhouse-exogenous-default@1",
                predictor_id="greenhouse-exogenous-ridge@1",
                feature_policy_id="registered_greenhouse_features@1",
            ),
            _seed_template(
                thawed,
                template_id="greenhouse-baseline-aligned-default@1",
                predictor_id="greenhouse-baseline-aligned-ridge@1",
                feature_policy_id="registered_greenhouse_features@1",
            ),
            _seed_template(
                thawed,
                template_id="greenhouse-rolling-default@1",
                predictor_id="greenhouse-rolling-residual@1",
                feature_policy_id="registered_greenhouse_features@1",
            ),
            _seed_template(
                thawed,
                template_id="greenhouse-recipe-default@1",
                predictor_id="greenhouse-recipe-ridge@1",
                feature_policy_id="authored_causal_features@1",
                # The information-symmetry seed: seasonal_reference reads the
                # same cell the selected seasonal baseline reads, so a recipe
                # candidate starts able to see everything its baseline sees
                # instead of having to search its way there under a
                # log-space trust region that cannot step 12 -> 24 lags.
                feature_recipe=seed_feature_recipe(
                    horizons=(1, 6, 24),
                    exogenous_columns=GREENHOUSE_SEED_EXOGENOUS_COLUMNS,
                ),
            ),
            _seed_template(
                thawed,
                template_id="toy-default@1",
                predictor_id="toy-rolling-water@1",
                feature_policy_id="registered_toy_features@1",
            ),
        )
        if seed_templates is None:
            # Harness 0.2 changes the executable preset identity. Preserve the
            # archived @1 templates byte for byte; new runs select @2 explicitly.
            harness_templates = []
            for template in templates:
                value = template.to_dict()
                value.pop("template_digest", None)
                value["template_id"] = value["template_id"].removesuffix("@1") + "@2"
                profiles = value["agent_program"]["candidate_execution_program"]["role_profiles"]
                for profile in profiles:
                    profile["preset_id"] = "ecology-sample-planner-v12"
                harness_templates.append(SeedGenomeTemplate.from_dict(value))
            templates = (*templates, *harness_templates)
        identity = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "programs": thawed,
            "seed_templates": [item.to_dict() for item in templates],
        }
        catalog_digest = _domain_digest("ecologyrsi-dsh/program-registry/1", identity)
        return cls(frozen, tuple(templates), catalog_digest)

    @property
    def catalog_digest(self) -> str:
        return self._catalog_digest

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "programs": deep_thaw_json(self._programs),
            "seed_templates": [item.to_dict() for item in self._seed_templates],
            "catalog_digest": self.catalog_digest,
        }

    def program(self, category: str, program_id: str) -> dict[str, Any]:
        programs = deep_thaw_json(self._programs)
        try:
            value = programs[category][program_id]
        except KeyError:
            raise ValueError(f"unregistered {category} program: {program_id}") from None
        return value

    def program_ref(self, category: str, program_id: str) -> dict[str, str]:
        value = self.program(category, program_id)
        return {
            "id": program_id,
            "catalog_digest": _program_digest(category, program_id, value),
        }

    def program_ids(self, category: str) -> tuple[str, ...]:
        programs = deep_thaw_json(self._programs)
        try:
            values = programs[category]
        except KeyError:
            raise ValueError(f"unregistered program category: {category}") from None
        if not isinstance(values, Mapping):
            raise ValueError(f"invalid program category: {category}")
        return tuple(sorted(str(program_id) for program_id in values))

    def seed_template(self, template_id: str) -> SeedGenomeTemplate:
        for template in self._seed_templates:
            if template.template_id == template_id:
                return SeedGenomeTemplate.from_dict(template.to_dict())
        raise ValueError(f"unregistered seed template: {template_id}")

    def with_program_override(
        self, category: str, program_id: str, override: Mapping[str, Any]
    ) -> "ProgramRegistrySnapshot":
        programs = deep_thaw_json(self._programs)
        if category not in programs or program_id not in programs[category]:
            raise ValueError("cannot override an unregistered program")
        programs[category][program_id] = {
            **programs[category][program_id],
            **dict(override),
        }
        return ProgramRegistrySnapshot.from_programs(programs)

    def predictor_defaults(self, predictor_id: str) -> dict[str, int | float]:
        predictor = self.program("predictors", predictor_id)
        return {
            name: contract["default"]
            for name, contract in predictor["parameters"].items()
        }

    def validate_parameter(
        self, predictor_id: str, name: str, value: int | float
    ) -> None:
        predictor = self.program("predictors", predictor_id)
        try:
            contract = predictor["parameters"][name]
        except KeyError:
            raise ValueError(f"unregistered predictor parameter: {name}") from None
        _validate_scalar_contract(value, contract, f"predictor parameter {name}")

    def workflow_defaults(self, workflow_id: str) -> dict[str, int | float]:
        workflow = self.program("workflow_templates", workflow_id)
        return {
            name: contract["default"]
            for name, contract in workflow["parameters"].items()
        }

    def validate_workflow_parameter(
        self, workflow_id: str, name: str, value: int | float
    ) -> None:
        workflow = self.program("workflow_templates", workflow_id)
        try:
            contract = workflow["parameters"][name]
        except KeyError:
            raise ValueError(f"unregistered workflow parameter: {name}") from None
        _validate_scalar_contract(value, contract, f"workflow parameter {name}")

    def validate_instruction_parameter(
        self, instruction_id: str, name: str, value: Any
    ) -> None:
        instruction = self.program("instruction_templates", instruction_id)
        try:
            contract = instruction["parameters"][name]
        except KeyError:
            raise ValueError(f"unregistered instruction parameter: {name}") from None
        _validate_scalar_contract(value, contract, f"instruction parameter {name}")

    def tool_policy(self, policy_id: str) -> tuple[str, ...]:
        policy = self.program("tool_policies", policy_id)
        return tuple(str(item) for item in policy["tool_ids"])

def _validate_scalar_contract(value: Any, contract: Mapping[str, Any], name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if contract.get("integer") and not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not float(contract["minimum"]) <= float(value) <= float(contract["maximum"]):
        raise ValueError(f"{name} is outside its registered bounds")


_CURRENT_PROGRAM_REGISTRY = ProgramRegistrySnapshot.from_programs(_CURRENT_PROGRAMS)


def current_program_registry() -> ProgramRegistrySnapshot:
    """Return the immutable current V1 registry snapshot."""

    return _CURRENT_PROGRAM_REGISTRY


__all__ = [
    "ProgramRegistrySnapshot",
    "REGISTRY_SCHEMA_VERSION",
    "current_program_registry",
]
