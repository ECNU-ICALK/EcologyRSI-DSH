"""Declarative, host-compiled planner directives authored by the Agent.

The registry's ``instruction_templates`` let a candidate *select* one of a fixed
set of strategy sentences; this module lets it *write* one. The Agent authors
JSON that names host clauses -- never prose the host cannot check, and never
code -- and the host renders that structure into the same
``instruction_directive`` slot the selected template already fills. Delivery is
therefore unchanged: the plugin serializes whatever text the compiled profile
carries, so nothing on the JavaScript side has to know this file exists.

Three of the four clauses are checkable, and each is checked at the layer that
owns it:

* ``tool_plan`` names tools; the compiler intersects it with the role's
  ``enabled_tool_ids``, so a directive can never instruct the planner toward a
  tool the tool policy withheld.
* ``blend_rule`` narrows the prediction methods the host will accept, enforced
  in ``core.agent_prediction.validate_predictions`` rather than only advertised.
* ``anchor`` and ``rationale`` are delivered to the planner as strategy. They
  are labelled that way in the grammar instead of being dressed up as
  constraints the host enforces.

Escalation is deliberately *not* a clause. The planner's critic-escalation
threshold already has one owner -- ``instruction_parameters.confidence_threshold``
-- and a second field writing the same runtime number would make the effective
value depend on which layer was consulted last. The renderer reads that single
value and states it in the directive text, so the prose the planner sees and the
threshold the host enforces cannot drift.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from ..core.agent_prediction import PREDICTION_TOOL_CALL_BUDGET
from ..core.models import canonical_json, digest

AUTHORED_DIRECTIVE_SCHEMA_VERSION = "ecologyrsi-dsh.authored-directive/1"
# The registry entry that publishes this grammar. Named here rather than spelled
# out at each use site so the genome's `directive_policy_ref` and the registered
# entry cannot come to disagree about which policy bounded a directive.
from ..evolution.mutation_specs import AUTHORED_DIRECTIVE_POLICY_ID

# Matches the registered-directive ceiling in `knowledge.program_registry`,
# which imports this name so the two cannot drift. A registry template and an
# authored directive land in the same prompt slot, so one length rule governs
# both.
MAX_DIRECTIVE_RATIONALE_LENGTH = 600
# A plan step is a tool call the planner is told to make, so the ceiling is the
# call budget itself rather than an independent number.
MAX_DIRECTIVE_TOOL_PLAN_STEPS = PREDICTION_TOOL_CALL_BUDGET

DIRECTIVE_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "anchor", "blend_rule", "tool_plan", "rationale"}
)
_TOOL_PLAN_STEP_KEYS = frozenset({"tool_id", "purpose"})

ALLOWED_DIRECTIVE_ANCHORS = frozenset(
    {"fit_selected_baseline", "persistence", "candidate_model"}
)
# The prediction methods each rule admits. `validate_predictions` receives this
# set, so a directive that forbids blending is enforced and not merely stated.
# Every rule keeps `direct` and `model`: withholding those would leave a planner
# whose tools all failed no legal way to answer at all.
BLEND_RULE_METHODS: dict[str, tuple[str, ...]] = {
    "none": ("direct", "model"),
    "mean": ("direct", "model", "blend"),
    "confidence_weighted": ("direct", "model", "blend", "adjusted"),
}
ALLOWED_BLEND_RULES = frozenset(BLEND_RULE_METHODS)
ALLOWED_TOOL_PURPOSES = frozenset(
    {
        "candidate_model_baseline",
        "discrepancy_check",
        "sensitivity_probe",
        "alternate_parameterization",
    }
)

_TOOL_ID_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")

_ANCHOR_PHRASES = {
    "fit_selected_baseline": (
        "Anchor every cell on the baseline the fit selected, and treat model "
        "output as a correction to it."
    ),
    "persistence": (
        "Anchor every cell on persistence, and justify each departure from the "
        "last observed value."
    ),
    "candidate_model": (
        "Anchor every cell on this candidate's own model output, and justify "
        "each departure from it against the causal observations."
    ),
}
_BLEND_PHRASES = {
    "none": (
        "Do not blend: commit to one source per cell. The host rejects a "
        "blended method."
    ),
    "mean": (
        "Where two tool results disagree without either being refuted, blend "
        "them evenly."
    ),
    "confidence_weighted": (
        "Where tool results disagree, blend them by how well each is supported, "
        "and adjust a cell when the observations refute both."
    ),
}
_PURPOSE_PHRASES = {
    "candidate_model_baseline": "establish this candidate's model baseline",
    "discrepancy_check": "resolve a discrepancy the previous call left open",
    "sensitivity_probe": "probe sensitivity to one parameter",
    "alternate_parameterization": "test an alternate parameterization",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must use string keys")
    return value


def _exact_keys(value: Mapping[str, Any], name: str, allowed: frozenset[str]) -> None:
    unknown = set(value) - allowed
    missing = allowed - set(value)
    if unknown:
        raise ValueError(f"{name} has unsupported fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(sorted(missing))}")


def _choice(value: Any, name: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(allowed))}")
    return value


def _bounded_text(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    # Whitespace-collapsed before the length check so a padded rationale cannot
    # buy room the bound was meant to deny, and so two directives that differ
    # only in line wrapping share one digest.
    result = " ".join(value.split())
    if len(result) > maximum:
        raise ValueError(f"{name} must be at most {maximum} characters")
    return result


@dataclass(frozen=True, slots=True)
class FrozenDirective:
    """A validated directive: plain data, already normalized and content addressed."""

    anchor: str
    blend_rule: str
    tool_plan: tuple[Mapping[str, str], ...]
    rationale: str

    @property
    def step_count(self) -> int:
        return len(self.tool_plan)

    @property
    def tool_ids(self) -> tuple[str, ...]:
        return tuple(sorted({str(step["tool_id"]) for step in self.tool_plan}))

    @property
    def allowed_methods(self) -> tuple[str, ...]:
        return BLEND_RULE_METHODS[self.blend_rule]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AUTHORED_DIRECTIVE_SCHEMA_VERSION,
            "anchor": self.anchor,
            "blend_rule": self.blend_rule,
            "tool_plan": [dict(step) for step in self.tool_plan],
            "rationale": self.rationale,
        }

    @property
    def digest(self) -> str:
        return digest(self.to_dict())


def validate_authored_directive(
    value: Any, *, allowed_tool_ids: Sequence[str] | None = None
) -> FrozenDirective:
    """Validate an Agent-authored directive against the host clause whitelist.

    Structural validation only by default: ``tool_id`` is checked for shape
    here and against the role's real tool policy in the compiler, because genome
    validation runs without a resolved registry.
    """

    raw = _mapping(value, "authored_directive")
    unknown = set(raw) - DIRECTIVE_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"authored_directive has unsupported fields: {', '.join(sorted(unknown))}"
        )
    declared = raw.get("schema_version", AUTHORED_DIRECTIVE_SCHEMA_VERSION)
    if declared != AUTHORED_DIRECTIVE_SCHEMA_VERSION:
        raise ValueError("unsupported authored_directive schema version")

    anchor = _choice(raw.get("anchor"), "authored_directive.anchor", ALLOWED_DIRECTIVE_ANCHORS)
    blend_rule = _choice(
        raw.get("blend_rule"), "authored_directive.blend_rule", ALLOWED_BLEND_RULES
    )

    steps = raw.get("tool_plan")
    if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
        raise TypeError("authored_directive.tool_plan must be a list")
    if not 1 <= len(steps) <= MAX_DIRECTIVE_TOOL_PLAN_STEPS:
        raise ValueError(
            "authored_directive.tool_plan must hold "
            f"1..{MAX_DIRECTIVE_TOOL_PLAN_STEPS} steps"
        )
    normalized_steps: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, step in enumerate(steps):
        name = f"authored_directive.tool_plan[{index}]"
        entry = _mapping(step, name)
        _exact_keys(entry, name, _TOOL_PLAN_STEP_KEYS)
        tool_id = entry["tool_id"]
        if not isinstance(tool_id, str) or _TOOL_ID_RE.fullmatch(tool_id) is None:
            raise ValueError(f"{name}.tool_id must be a registered tool identifier")
        purpose = _choice(entry["purpose"], f"{name}.purpose", ALLOWED_TOOL_PURPOSES)
        normalized = {"tool_id": tool_id, "purpose": purpose}
        key = canonical_json(normalized)
        if key in seen:
            raise ValueError(f"{name} duplicates an earlier step")
        seen.add(key)
        normalized_steps.append(normalized)

    rationale = _bounded_text(
        raw.get("rationale"),
        "authored_directive.rationale",
        MAX_DIRECTIVE_RATIONALE_LENGTH,
    )

    if allowed_tool_ids is not None:
        permitted = set(allowed_tool_ids)
        unavailable = sorted(
            {step["tool_id"] for step in normalized_steps} - permitted
        )
        if unavailable:
            raise ValueError(
                "authored_directive plans unavailable tools: "
                + ", ".join(unavailable)
            )

    frozen = FrozenDirective(
        anchor=anchor,
        blend_rule=blend_rule,
        tool_plan=tuple(normalized_steps),
        rationale=rationale,
    )
    _reject_executable_fields(frozen.to_dict())
    return frozen


def _reject_executable_fields(body: Mapping[str, Any]) -> None:
    """Defence in depth: the key whitelist above already makes this unreachable.

    Imported lazily because ``ecologyrsi_dsh.knowledge`` pulls in the evaluators
    package, and importing it at module scope would close an import cycle.
    """

    from ..knowledge.autonomous_cycle import reject_executable_fields

    reject_executable_fields(body, path="$.authored_directive")


def render_authored_directive(
    directive: FrozenDirective | Mapping[str, Any],
    *,
    escalation_threshold: float | None = None,
) -> str:
    """Render the validated structure into the directive text the planner reads.

    Deterministic and clause-ordered: the same structure always renders the same
    sentence sequence, so the rendered text is a pure function of the genome and
    the rendered directive travels inside ``compiled_behavior_digest`` without
    adding a second, independently drifting source of identity.

    ``escalation_threshold`` is the effective ``confidence_threshold`` from the
    role's instruction parameters -- the number the host will actually enforce.
    Stating it here is what keeps the planner's instructions and the host's
    escalation rule from describing two different policies.
    """

    frozen = (
        directive
        if isinstance(directive, FrozenDirective)
        else validate_authored_directive(directive)
    )
    steps = "; ".join(
        f"{index + 1}) call {step['tool_id']} to {_PURPOSE_PHRASES[step['purpose']]}"
        for index, step in enumerate(frozen.tool_plan)
    )
    sentences = [
        _ANCHOR_PHRASES[frozen.anchor],
        _BLEND_PHRASES[frozen.blend_rule],
        f"Planned evidence: {steps}.",
    ]
    if escalation_threshold is not None:
        sentences.append(
            "Any cell you submit below confidence "
            f"{float(escalation_threshold):.2f} is sent to the remote critic for "
            "an independent look, so state a confidence you can defend."
        )
    sentences.append(f"Why this strategy: {frozen.rationale}")
    return " ".join(sentences)


def directive_grammar() -> dict[str, Any]:
    """The clause whitelist as data, for the program registry and the editor.

    Zero-argument and deterministic because the program registry hashes it into
    ``catalog_digest``: the grammar an Agent was shown has to be recoverable
    from the genome's digest alone. Keyed flat -- ``"escalation.min_confidence"``
    style rather than nested objects -- for the same reason
    ``feature_recipe.recipe_grammar`` is: this travels to the model inside
    ``plan.tools[*].parameters``, where ``sample_contracts._safe_value`` refuses
    anything nested past ten levels.

    ``enforced_by`` is part of the published grammar on purpose. Two clauses are
    checked by the host and two are strategy delivered to the planner, and an
    editor that cannot tell them apart will spend its one operation on the
    clause it thought was a constraint.
    """

    return {
        "schema_version": AUTHORED_DIRECTIVE_SCHEMA_VERSION,
        "clauses": ["anchor", "blend_rule", "tool_plan", "rationale"],
        "anchor": {
            "kind": "choice",
            "choices": sorted(ALLOWED_DIRECTIVE_ANCHORS),
            "enforced_by": "planner",
        },
        "blend_rule": {
            "kind": "choice",
            "choices": sorted(ALLOWED_BLEND_RULES),
            "enforced_by": "host",
        },
        "blend_rule.allowed_methods": {
            rule: list(methods) for rule, methods in sorted(BLEND_RULE_METHODS.items())
        },
        "tool_plan": {
            "kind": "list",
            "minimum_steps": 1,
            "maximum_steps": MAX_DIRECTIVE_TOOL_PLAN_STEPS,
            "enforced_by": "host",
        },
        "tool_plan.tool_id": {
            "kind": "string",
            "resolved_against": "enabled_tool_ids",
        },
        "tool_plan.purpose": {
            "kind": "choice",
            "choices": sorted(ALLOWED_TOOL_PURPOSES),
        },
        "rationale": {
            "kind": "text",
            "maximum_length": MAX_DIRECTIVE_RATIONALE_LENGTH,
            "enforced_by": "planner",
        },
        "duplicate_tool_plan_steps": "rejected",
        "escalation_threshold": "instruction_parameters.confidence_threshold",
        "bounds_enforced_by": "host",
    }


def directive_digest(directive: FrozenDirective | Mapping[str, Any]) -> str:
    if isinstance(directive, FrozenDirective):
        return directive.digest
    return digest(validate_authored_directive(directive).to_dict())


__all__ = [
    "ALLOWED_BLEND_RULES",
    "ALLOWED_DIRECTIVE_ANCHORS",
    "ALLOWED_TOOL_PURPOSES",
    "AUTHORED_DIRECTIVE_POLICY_ID",
    "AUTHORED_DIRECTIVE_SCHEMA_VERSION",
    "BLEND_RULE_METHODS",
    "DIRECTIVE_TOP_LEVEL_KEYS",
    "FrozenDirective",
    "MAX_DIRECTIVE_RATIONALE_LENGTH",
    "MAX_DIRECTIVE_TOOL_PLAN_STEPS",
    "directive_digest",
    "directive_grammar",
    "render_authored_directive",
    "validate_authored_directive",
]
