from __future__ import annotations

import unittest

from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.trajectory import LocalEditOutcome
from ecologyrsi_dsh.evolution.genome import (
    FrozenRunInitialization,
    materialize_seed_genome,
)
from ecologyrsi_dsh.evolution.local_edits import (
    LocalEditContext,
    LocalEditProposal,
    apply_local_edit_bundle,
    apply_or_reject_local_edit_bundle,
    validate_local_edit_proposal,
)
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry


def _parent():
    registry = current_program_registry()
    initialization = FrozenRunInitialization(
        run_id="run:local-edit",
        task_manifest_digest="1" * 64,
        dataset_snapshot_set_digest="2" * 64,
        split_manifest_digest="3" * 64,
        data_protocol_digest="4" * 64,
        stage_policy_digest="5" * 64,
        evaluator_digest="6" * 64,
        fitness_profile_digest="7" * 64,
        security_kernel_digest="8" * 64,
        selection_reviewer_program_digest="9" * 64,
        protocol="dsh_native_plugin_evolution@1",
        required_capability_digest="a" * 64,
        resolved_policy_route_digest="b" * 64,
        resolved_review_route_digest="c" * 64,
        registry_catalog_digest=registry.catalog_digest,
        compiler_digest="d" * 64,
    )
    return materialize_seed_genome(
        registry.seed_template("greenhouse-default@1"), initialization
    )


def _context(parent, maximum: int = 2) -> LocalEditContext:
    return LocalEditContext(
        run_id="run:local-edit",
        generation=0,
        candidate_id="candidate:local-edit",
        candidate_revision_id="revision:local-edit:0",
        batch_index=0,
        evidence_scope_digest=digest({"scope": 0}),
        parent_genome_digest=parent.genome_digest,
        maximum_operations=maximum,
        allowed_mutation_targets={
            "scientific_parameter": ("history_steps", "ridge_alpha"),
            "registered_predictor": ("greenhouse-targetwise-ridge@1",),
            "instruction_profile": ("sample-planner-anomaly-aware@1",),
        },
        allowed_evidence_refs=("metric:overall", "metric:co2:24"),
        allowed_effect_cells=(
            "air_temperature@1h",
            "co2_concentration@24h",
        ),
        parameter_schemas={
            "history_steps": {"minimum": 1, "maximum": 12},
            "ridge_alpha": {"minimum": 0.0001, "maximum": 1.0},
        },
    )


class LocalEditTests(unittest.TestCase):
    def test_keep_is_valid_and_does_not_create_revision(self) -> None:
        parent = _parent()
        result = apply_local_edit_bundle(
            parent,
            LocalEditProposal(
                decision="keep",
                operations=(),
                evidence_refs=("metric:overall",),
                expected_effect_cells=(),
                risk_cells=(),
            ),
            _context(parent),
            current_program_registry(),
        )

        self.assertIs(result.outcome, LocalEditOutcome.KEPT)
        self.assertIsNone(result.child)

    def test_zero_maximum_allows_keep_and_rejects_mutation(self) -> None:
        parent = _parent()
        context = _context(parent, maximum=0)
        keep = LocalEditProposal(
            decision="keep",
            operations=(),
            evidence_refs=(),
            expected_effect_cells=(),
            risk_cells=(),
        )

        self.assertIs(validate_local_edit_proposal(keep, context), keep)
        mutate = LocalEditProposal(
            decision="mutate",
            operations=(
                {
                    "op": "set_bounded_parameter",
                    "name": "ridge_alpha",
                    "value": 0.2,
                },
            ),
            evidence_refs=(),
            expected_effect_cells=(),
            risk_cells=(),
        )
        with self.assertRaisesRegex(ValueError, "maximum_operations"):
            validate_local_edit_proposal(mutate, context)

    def test_two_registered_edits_are_applied_atomically(self) -> None:
        parent = _parent()
        proposal = LocalEditProposal(
            decision="mutate",
            operations=(
                {
                    "op": "set_bounded_parameter",
                    "name": "ridge_alpha",
                    "value": 0.2,
                },
                {
                    "op": "set_bounded_parameter",
                    "name": "history_steps",
                    "value": 7,
                },
            ),
            evidence_refs=("metric:overall",),
            expected_effect_cells=("air_temperature@1h",),
            risk_cells=("co2_concentration@24h",),
        )

        result = apply_local_edit_bundle(
            parent, proposal, _context(parent), current_program_registry()
        )

        self.assertIs(result.outcome, LocalEditOutcome.APPLIED)
        self.assertIsNotNone(result.child)
        assert result.child is not None
        self.assertEqual(
            result.child.scientific_program["parameter_overrides"]["ridge_alpha"],
            0.2,
        )
        self.assertEqual(
            result.child.scientific_program["parameter_overrides"]["history_steps"],
            7,
        )
        self.assertEqual(
            result.child.lineage["parent_genome_digest"], parent.genome_digest
        )

    def test_dynamic_maximum_duplicate_and_unregistered_targets_are_rejected(self) -> None:
        parent = _parent()
        base = {
            "op": "set_bounded_parameter",
            "name": "ridge_alpha",
            "value": 0.2,
        }
        too_many = LocalEditProposal(
            decision="mutate",
            operations=(base, {**base, "name": "history_steps", "value": 7}),
            evidence_refs=(),
            expected_effect_cells=(),
            risk_cells=(),
        )
        with self.assertRaisesRegex(ValueError, "maximum_operations"):
            validate_local_edit_proposal(too_many, _context(parent, maximum=1))

        duplicate = LocalEditProposal(
            decision="mutate",
            operations=(base, {**base, "value": 0.15}),
            evidence_refs=(),
            expected_effect_cells=(),
            risk_cells=(),
        )
        with self.assertRaisesRegex(ValueError, "duplicate local edit target"):
            validate_local_edit_proposal(duplicate, _context(parent))

        forbidden = LocalEditProposal(
            decision="mutate",
            operations=({**base, "name": "unregistered_parameter"},),
            evidence_refs=(),
            expected_effect_cells=(),
            risk_cells=(),
        )
        with self.assertRaisesRegex(ValueError, "not registered"):
            validate_local_edit_proposal(forbidden, _context(parent))

    def test_bundle_failure_leaves_parent_unchanged(self) -> None:
        parent = _parent()
        before = parent.to_dict()
        proposal = LocalEditProposal(
            decision="mutate",
            operations=(
                {
                    "op": "set_bounded_parameter",
                    "name": "ridge_alpha",
                    "value": 0.2,
                },
                {
                    "op": "set_bounded_parameter",
                    "name": "history_steps",
                    "value": 99,
                },
            ),
            evidence_refs=(),
            expected_effect_cells=(),
            risk_cells=(),
        )

        with self.assertRaises(ValueError):
            apply_local_edit_bundle(
                parent, proposal, _context(parent), current_program_registry()
            )
        self.assertEqual(parent.to_dict(), before)

    def test_invalid_model_authored_bundle_is_rejected_without_changing_parent(self) -> None:
        parent = _parent()
        before = parent.to_dict()
        proposal = LocalEditProposal(
            decision="mutate",
            operations=(
                {
                    "op": "set_bounded_parameter",
                    "name": "air_temperature_6h_residual_scale",
                    "value": 0,
                },
            ),
            evidence_refs=("metric:overall",),
            expected_effect_cells=("air_temperature@1h",),
            risk_cells=(),
        )

        result = apply_or_reject_local_edit_bundle(
            parent, proposal, _context(parent), current_program_registry()
        )

        self.assertIs(result.outcome, LocalEditOutcome.REJECTED)
        self.assertIsNone(result.child)
        self.assertEqual(result.operations, proposal.operations)
        self.assertEqual(parent.to_dict(), before)

    def test_parent_identity_mismatch_is_never_downgraded_to_rejection(self) -> None:
        parent = _parent()
        context = _context(parent)
        mismatched = LocalEditContext(
            **{
                **context.to_dict(),
                "parent_genome_digest": "f" * 64,
            }
        )

        with self.assertRaisesRegex(ValueError, "parent genome digest mismatch"):
            apply_or_reject_local_edit_bundle(
                parent,
                LocalEditProposal(
                    decision="keep",
                    operations=(),
                    evidence_refs=(),
                    expected_effect_cells=(),
                    risk_cells=(),
                ),
                mismatched,
                current_program_registry(),
            )

    def test_outer_candidate_contract_remains_one_operation(self) -> None:
        source = (
            __import__("pathlib").Path(__file__).parents[1]
            / "src/ecologyrsi_dsh/evolution/strategies.py"
        ).read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count('"maximum_operations": 1'), 2)


if __name__ == "__main__":
    unittest.main()
