"""Immutable source-program catalogs for plugin-genome compilation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from ..evolution.genome import (
    FrozenJsonObject,
    SeedGenomeTemplate,
    _domain_digest,
    deep_freeze_json,
    deep_thaw_json,
)


REGISTRY_SCHEMA_VERSION = "ecologyrsi-dsh.program-registry/1"


def _parameter(
    *, minimum: int | float, maximum: int | float, default: int | float, integer: bool = False
) -> dict[str, Any]:
    return {
        "minimum": minimum,
        "maximum": maximum,
        "default": default,
        "integer": integer,
    }


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
                "max_attempts": _parameter(minimum=1, maximum=8, default=3, integer=True),
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
    "instruction_templates": {
        "sample-planner@1": {
            "version": "sample-planner-instruction/1",
            "parameters": {
                "confidence_threshold": _parameter(
                    minimum=0.0, maximum=1.0, default=0.7
                )
            },
        },
        "sample-repair@1": {
            "version": "sample-repair-instruction/1",
            "parameters": {},
        },
        "researcher@1": {"version": "researcher-instruction/1", "parameters": {}},
        "candidate-proposer@1": {
            "version": "candidate-proposer-instruction/1",
            "parameters": {},
        },
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
                    "preset_id": "ecology-sample-planner-v1",
                    "instruction_template_ref": _program_ref(
                        programs, "instruction_templates", "sample-planner@1"
                    ),
                    "instruction_parameters": {"confidence_threshold": 0.7},
                    "response_schema_id": "sample-decisions@1",
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
) -> SeedGenomeTemplate:
    predictor = programs["predictors"][predictor_id]
    parameters = {
        name: contract["default"]
        for name, contract in predictor["parameters"].items()
    }
    return SeedGenomeTemplate.from_dict(
        {
            "schema_version": "ecologyrsi-dsh.seed-genome-template/1",
            "template_id": template_id,
            "scientific_program": {
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
            },
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
                template_id="toy-default@1",
                predictor_id="toy-rolling-water@1",
                feature_policy_id="registered_toy_features@1",
            ),
        )
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

    def migration_template(self, template_id: str) -> FrozenJsonObject:
        if template_id != "legacy-dsh-native@1":
            raise ValueError(f"unregistered migration template: {template_id}")
        programs = deep_thaw_json(self._programs)
        template: dict[str, Any] = {
            "schema_version": "ecologyrsi-dsh.legacy-migration-template/1",
            "template_id": template_id,
            "predictor_refs": {
                predictor_id: self.program_ref("predictors", predictor_id)
                for predictor_id in programs["predictors"]
            },
            "feature_policy_ref": {
                **self.program_ref(
                    "feature_policies", "registered_greenhouse_features@1"
                ),
                "overrides": {},
            },
            "fit_policy_ref": {
                **self.program_ref("fit_policies", "time_forward_fit@1"),
                "overrides": {},
            },
            "uncertainty_policy_ref": {
                **self.program_ref("uncertainty_policies", "none@1"),
                "overrides": {},
            },
            "agent_program": _agent_program(programs),
            "evidence_refs": [],
        }
        template["template_digest"] = _domain_digest(
            "ecologyrsi-dsh/legacy-migration-template/1", template
        )
        frozen = deep_freeze_json(template)
        assert isinstance(frozen, FrozenJsonObject)
        return frozen


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


@dataclass(frozen=True, slots=True)
class LegacyProgramCatalog:
    _value: FrozenJsonObject

    def to_dict(self) -> dict[str, Any]:
        return deep_thaw_json(self._value)


def _load_legacy_catalog() -> LegacyProgramCatalog:
    path = Path(__file__).with_name("legacy_program_catalog_0_2_2.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = dict(value)
    supplied = identity.pop("catalog_digest", None)
    expected = _domain_digest(
        "ecologyrsi-dsh/legacy-program-catalog/0.2.2", identity
    )
    if supplied is not None and supplied != expected:
        raise RuntimeError("legacy 0.2.2 program catalog digest mismatch")
    value["catalog_digest"] = expected
    frozen = deep_freeze_json(value)
    assert isinstance(frozen, FrozenJsonObject)
    return LegacyProgramCatalog(frozen)


LEGACY_PROGRAM_CATALOG_0_2_2 = _load_legacy_catalog()


__all__ = [
    "LEGACY_PROGRAM_CATALOG_0_2_2",
    "LegacyProgramCatalog",
    "ProgramRegistrySnapshot",
    "REGISTRY_SCHEMA_VERSION",
    "current_program_registry",
]
