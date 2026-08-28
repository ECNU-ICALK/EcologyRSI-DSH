"""Bounded, atomic local edits between adaptive finalist batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..core.models import canonical_json, digest
from ..core.trajectory import LocalEditOutcome, LocalEditProposalDecision
from .genome import (
    LOCAL_EDIT_MUTATION_OPERATOR_ID,
    EcologyEvolutionPluginGenome,
    GenomeMutationContextV1,
    apply_genome_mutation,
)


LOCAL_EDIT_SCHEMA_VERSION = "ecology-local-edit@1"
LOCAL_EDIT_MAXIMUM_OPERATIONS = 5


def _digest_text(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _texts(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique non-empty text")
    return result


@dataclass(frozen=True, slots=True)
class LocalEditContext:
    run_id: str
    generation: int
    candidate_id: str
    candidate_revision_id: str
    batch_index: int
    evidence_scope_digest: str
    parent_genome_digest: str
    maximum_operations: int
    allowed_mutation_targets: Mapping[str, Sequence[str]]
    allowed_evidence_refs: Sequence[str]
    allowed_effect_cells: Sequence[str]
    parameter_schemas: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        for name in ("run_id", "candidate_id", "candidate_revision_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        for name in ("generation", "batch_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        _digest_text(self.evidence_scope_digest, "evidence_scope_digest")
        _digest_text(self.parent_genome_digest, "parent_genome_digest")
        if (
            isinstance(self.maximum_operations, bool)
            or not isinstance(self.maximum_operations, int)
            or not 1 <= self.maximum_operations <= LOCAL_EDIT_MAXIMUM_OPERATIONS
        ):
            raise ValueError("maximum_operations must be between 1 and 5")
        if not isinstance(self.allowed_mutation_targets, Mapping):
            raise TypeError("allowed_mutation_targets must be an object")
        object.__setattr__(
            self,
            "allowed_mutation_targets",
            {
                str(axis): _texts(values, f"allowed_mutation_targets.{axis}")
                for axis, values in self.allowed_mutation_targets.items()
            },
        )
        object.__setattr__(
            self,
            "allowed_evidence_refs",
            _texts(self.allowed_evidence_refs, "allowed_evidence_refs"),
        )
        object.__setattr__(
            self,
            "allowed_effect_cells",
            _texts(self.allowed_effect_cells, "allowed_effect_cells"),
        )
        if not isinstance(self.parameter_schemas, Mapping):
            raise TypeError("parameter_schemas must be an object")
        canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "generation": self.generation,
            "candidate_id": self.candidate_id,
            "candidate_revision_id": self.candidate_revision_id,
            "batch_index": self.batch_index,
            "evidence_scope_digest": self.evidence_scope_digest,
            "parent_genome_digest": self.parent_genome_digest,
            "maximum_operations": self.maximum_operations,
            "allowed_mutation_targets": {
                axis: list(values)
                for axis, values in self.allowed_mutation_targets.items()
            },
            "allowed_evidence_refs": list(self.allowed_evidence_refs),
            "allowed_effect_cells": list(self.allowed_effect_cells),
            "parameter_schemas": {
                str(name): dict(schema)
                for name, schema in self.parameter_schemas.items()
            },
        }


@dataclass(frozen=True, slots=True)
class LocalEditProposal:
    decision: LocalEditProposalDecision | str
    operations: Sequence[Mapping[str, Any]]
    evidence_refs: Sequence[str]
    expected_effect_cells: Sequence[str]
    risk_cells: Sequence[str]
    schema_version: str = LOCAL_EDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOCAL_EDIT_SCHEMA_VERSION:
            raise ValueError("unsupported local edit schema_version")
        object.__setattr__(self, "decision", LocalEditProposalDecision(self.decision))
        if isinstance(self.operations, (str, bytes)) or not isinstance(
            self.operations, Sequence
        ):
            raise TypeError("local edit operations must be an array")
        operations = tuple(dict(item) for item in self.operations)
        canonical_json(operations)
        object.__setattr__(self, "operations", operations)
        for name in ("evidence_refs", "expected_effect_cells", "risk_cells"):
            object.__setattr__(self, name, _texts(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision": self.decision.value,
            "operations": [dict(item) for item in self.operations],
            "evidence_refs": list(self.evidence_refs),
            "expected_effect_cells": list(self.expected_effect_cells),
            "risk_cells": list(self.risk_cells),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LocalEditProposal":
        expected = {
            "schema_version",
            "decision",
            "operations",
            "evidence_refs",
            "expected_effect_cells",
            "risk_cells",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("local edit proposal fields do not match schema")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class LocalEditResult:
    outcome: LocalEditOutcome
    operations: tuple[Mapping[str, Any], ...]
    child: EcologyEvolutionPluginGenome | None
    proposal_digest: str


def _operation_target(operation: Mapping[str, Any]) -> tuple[str, str, str]:
    op = str(operation.get("op") or "")
    if op == "set_bounded_parameter":
        name = str(operation.get("name") or "")
        return "scientific_parameter", name, f"parameter:{name}"
    if op == "select_registered_pipeline":
        return (
            "registered_predictor",
            str(operation.get("predictor_id") or ""),
            "predictor",
        )
    if op in {
        "select_registered_feature_policy",
        "select_registered_fit_policy",
        "select_registered_uncertainty_policy",
    }:
        category = {
            "select_registered_feature_policy": "feature_policy",
            "select_registered_fit_policy": "fit_policy",
            "select_registered_uncertainty_policy": "uncertainty_policy",
        }[op]
        return category, str(operation.get("program_id") or ""), category
    if op == "select_instruction_template":
        role = str(operation.get("role") or "")
        target = str(operation.get("instruction_template_id") or "")
        return "instruction_profile", target, f"instruction:{role}"
    if op == "set_bounded_workflow_parameter":
        name = str(operation.get("name") or "")
        return "workflow_parameter", name, f"workflow:{name}"
    if op == "select_registered_workflow_template":
        return (
            "workflow_template",
            str(operation.get("workflow_template_id") or ""),
            "workflow",
        )
    if op == "set_instruction_parameter":
        role = str(operation.get("role") or "")
        name = str(operation.get("name") or "")
        return "instruction_parameter", name, f"instruction-parameter:{role}:{name}"
    if op == "narrow_role_tool_policy":
        role = str(operation.get("role") or "")
        return "instruction_tool_policy", role, f"tool-policy:{role}"
    raise ValueError(f"local edit operation {op or '<missing>'} is not registered")


def validate_local_edit_proposal(
    proposal: LocalEditProposal | Mapping[str, Any],
    context: LocalEditContext,
) -> LocalEditProposal:
    result = (
        proposal
        if isinstance(proposal, LocalEditProposal)
        else LocalEditProposal.from_dict(proposal)
    )
    if not isinstance(context, LocalEditContext):
        raise TypeError("context must be LocalEditContext")
    count = len(result.operations)
    if result.decision is LocalEditProposalDecision.KEEP:
        if count:
            raise ValueError("keep local edit must have zero operations")
    elif not 1 <= count <= context.maximum_operations:
        raise ValueError(
            "local edit operations exceed maximum_operations or are empty"
        )
    seen_paths: set[str] = set()
    for operation in result.operations:
        axis, target, path = _operation_target(operation)
        if target not in context.allowed_mutation_targets.get(axis, ()):
            raise ValueError(f"local edit target {axis}:{target} is not registered")
        if path in seen_paths:
            raise ValueError(f"duplicate local edit target: {path}")
        seen_paths.add(path)
    if not set(result.evidence_refs).issubset(context.allowed_evidence_refs):
        raise ValueError("local edit evidence_refs contain an unknown metric")
    if not set(result.expected_effect_cells).issubset(
        context.allowed_effect_cells
    ) or not set(result.risk_cells).issubset(context.allowed_effect_cells):
        raise ValueError("local edit effect/risk cells are outside the frozen objective")
    return result


def apply_local_edit_bundle(
    parent: EcologyEvolutionPluginGenome,
    proposal: LocalEditProposal | Mapping[str, Any],
    context: LocalEditContext,
    registry: Any,
) -> LocalEditResult:
    validated = validate_local_edit_proposal(proposal, context)
    proposal_digest = digest(validated.to_dict())
    if context.parent_genome_digest != parent.genome_digest:
        raise ValueError("local edit parent genome digest mismatch")
    if validated.decision is LocalEditProposalDecision.KEEP:
        return LocalEditResult(
            outcome=LocalEditOutcome.KEPT,
            operations=(),
            child=None,
            proposal_digest=proposal_digest,
        )
    mutation_context = GenomeMutationContextV1(
        run_id=context.run_id,
        generation=context.generation,
        slot_index=context.batch_index,
        slot_seed=int(proposal_digest[:16], 16),
        parent_candidate_id=None if context.generation == 0 else context.candidate_id,
        parent_genome_digest=parent.genome_digest,
        generation_batch_digest=context.evidence_scope_digest,
        research_iteration_digest=digest(
            {"local_edit_evidence": context.evidence_scope_digest}
        ),
        knowledge_snapshot_digest=digest(
            {"candidate_revision_id": context.candidate_revision_id}
        ),
        mutation_budget_digest=digest(
            {
                "maximum_operations": context.maximum_operations,
                "allowed_mutation_targets": context.to_dict()[
                    "allowed_mutation_targets"
                ],
            }
        ),
        mutation_operator_id=LOCAL_EDIT_MUTATION_OPERATOR_ID,
    )
    child = apply_genome_mutation(
        parent,
        {
            "schema_version": "ecologyrsi-dsh.genome-mutation/1",
            "operations": [dict(item) for item in validated.operations],
        },
        mutation_context,
        registry,
        parameter_schemas=context.parameter_schemas,
    )
    return LocalEditResult(
        outcome=LocalEditOutcome.APPLIED,
        operations=tuple(validated.operations),
        child=child,
        proposal_digest=proposal_digest,
    )


def apply_or_reject_local_edit_bundle(
    parent: EcologyEvolutionPluginGenome,
    proposal: LocalEditProposal | Mapping[str, Any],
    context: LocalEditContext,
    registry: Any,
) -> LocalEditResult:
    """Apply one model-authored edit, or reject invalid bounded content.

    A local editor is advisory: an unknown target or out-of-range value must
    keep the active revision and advance the trajectory as ``rejected``.  It
    must not fail the whole evolution run.  A parent identity mismatch remains
    a Host invariant violation and is deliberately not downgraded.
    """

    if context.parent_genome_digest != parent.genome_digest:
        raise ValueError("local edit parent genome digest mismatch")
    try:
        return apply_local_edit_bundle(parent, proposal, context, registry)
    except (TypeError, ValueError):
        proposal_data = (
            proposal.to_dict()
            if isinstance(proposal, LocalEditProposal)
            else dict(proposal)
        )
        raw_operations = proposal_data.get("operations")
        operations = (
            tuple(dict(item) for item in raw_operations)
            if isinstance(raw_operations, Sequence)
            and not isinstance(raw_operations, (str, bytes))
            and all(isinstance(item, Mapping) for item in raw_operations)
            else ()
        )
        return LocalEditResult(
            outcome=LocalEditOutcome.REJECTED,
            operations=operations,
            child=None,
            proposal_digest=digest(proposal_data),
        )


__all__ = [
    "LOCAL_EDIT_MAXIMUM_OPERATIONS",
    "LOCAL_EDIT_SCHEMA_VERSION",
    "LocalEditContext",
    "LocalEditProposal",
    "LocalEditResult",
    "apply_local_edit_bundle",
    "apply_or_reject_local_edit_bundle",
    "validate_local_edit_proposal",
]
