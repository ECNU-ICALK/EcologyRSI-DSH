"""Bounded prior-generation edit evidence for the local editor.

The creation-time policy enables this context only for new runs. Historical
manifests without it retain their original local-edit context.
"""
from collections.abc import Mapping

from ..core.immutable import thaw_json
from ..core.models import canonical_json, digest
from ..core.trajectory import EvaluationPhase, FormalBatchComparisonDecision


MAX_RECORDS = 6
MAX_BYTES = 8000
POLICY_KEY = "local_edit_memory_policy"
POLICY = "prior_generation_pairs@1"
_BINDINGS = (
    "dataset_digest", "split_manifest_digest_sha256", "evaluation_index_digest",
    "baseline_profile_digest", "baseline_metrics_digest", "feedback_update_cohort_digest",
)
_OP_FIELDS = (
    "op", "role", "name", "value", "predictor_id", "instruction_template_id",
    "workflow_template_id", "tool_policy_id", "skill_policy_id", "component_id",
)


def enabled(metadata):
    value = metadata.get(POLICY_KEY)
    if value not in (None, POLICY):
        raise ValueError("unsupported local_edit_memory_policy")
    return value == POLICY


def local_edit_memory_context(state, *, generation):
    metadata = getattr(getattr(state, "task_manifest", None), "metadata", {}) or {}
    if not enabled(metadata):
        return {}
    return {"prior_generation_edit_memory": build_local_edit_memory(state, generation=generation)}


def _operation_summary(operation):
    # Never copy authored prose, raw examples, recipe expressions or programs
    # into the next model's memory. Exact content remains bound by its digest.
    result = {key: operation[key] for key in _OP_FIELDS if key in operation
              and (type(operation[key]) in (int, float, bool)
                   or (key != "value" and isinstance(operation[key], str)
                       and len(operation[key]) <= 160))}
    result["operation_digest"] = digest(thaw_json(operation))
    return result


def _complete_pair(left, right):
    if left.scope.origin_count != right.scope.origin_count:
        return False
    for key in _BINDINGS:
        if not left.metrics.get(key) or left.metrics.get(key) != right.metrics.get(key):
            return False
    for evaluation in (left, right):
        sample = evaluation.metrics.get("sample_execution", {})
        if (any(type(sample.get(key)) is not int for key in (
                    "attempted_origin_samples", "succeeded_origin_samples",
                    "failed_origin_samples", "scoring_fallback_examples"))
                or sample.get("attempted_origin_samples") != evaluation.scope.origin_count
                or sample.get("succeeded_origin_samples") != evaluation.scope.origin_count
                or sample.get("failed_origin_samples") != 0
                or sample.get("scoring_fallback_examples") != 0
                or sample.get("strict_agent_chain_pass") is not True
                or type(evaluation.metrics.get("sample_execution_coverage")) not in (int, float)
                or evaluation.metrics.get("sample_execution_coverage") != 1.0):
            return False
    return True


def build_local_edit_memory(state, *, generation):
    """Use verified same-run training pairs from strictly earlier generations.

    This is advisory evidence, not a transferred score or an operation ban.
    Different parent genomes can respond differently to the same operation.
    Incomplete executions retain their rejection information but contribute
    no performance delta. Independent evaluation and raw labels are not read.
    """
    if type(generation) is not int or generation < 0:
        raise ValueError("generation must be a non-negative integer")
    run_id = state.run.run_id
    evaluations = {item.evaluation_id: item for item in state.formal_batch_evaluations}
    comparisons = sorted(
        (item for item in state.formal_batch_comparisons
         if item.run_id == run_id and item.generation < generation
         and item.decision is not FormalBatchComparisonDecision.INITIAL_CHAMPION),
        key=lambda item: (item.generation, item.batch_index, item.comparison_id), reverse=True,
    )
    pool, seen = [], set()
    for comparison in comparisons:
        left = evaluations.get(comparison.champion_evaluation_id)
        right = evaluations.get(comparison.challenger_evaluation_id)
        if left is None or right is None:
            continue
        expected = (
            (left, comparison.champion_before_revision_id, comparison.champion_evaluation_digest),
            (right, comparison.challenger_revision_id, comparison.challenger_evaluation_digest),
        )
        if any(item.scope.phase is not EvaluationPhase.FORMAL_BATCH
               or item.scope.run_id != run_id
               or item.scope.generation != comparison.generation
               or item.scope.candidate_id != comparison.candidate_id
               or item.scope.batch_index != comparison.batch_index
               or item.scope.cohort_digest != comparison.cohort_digest
               or item.scope.candidate_revision_id != revision_id
               or item.evaluation_digest != expected_digest
               for item, revision_id, expected_digest in expected):
            continue
        child = state.revision(comparison.challenger_revision_id)
        parent = state.revision(comparison.champion_before_revision_id)
        if (any(item.run_id != run_id or item.generation != comparison.generation
                or item.candidate_id != comparison.candidate_id for item in (child, parent))
                or child.parent_revision_id != parent.revision_id
                or type(child.source_batch_index) is not int
                or child.source_batch_index >= comparison.batch_index):
            continue
        recorded = state.local_edit_proposal_for(comparison.candidate_id, child.source_batch_index)
        if not isinstance(recorded, Mapping):
            continue
        proposal = recorded.get("proposal", recorded)
        operations = proposal.get("operations", ())
        if (not operations or len(operations) > 5
                or any(not isinstance(operation, Mapping) for operation in operations)):
            continue
        signature = digest({"operations": sorted(canonical_json(thaw_json(op)) for op in operations)})
        identity = (parent.genome_digest, signature)
        if identity in seen:
            continue
        seen.add(identity)
        complete = _complete_pair(left, right)
        pool.append({
            "comparison_id": comparison.comparison_id,
            "generation": comparison.generation,
            "candidate_id": comparison.candidate_id,
            "parent_revision_id": parent.revision_id,
            "parent_genome_digest": parent.genome_digest,
            "child_revision_id": child.revision_id,
            "child_genome_digest": child.genome_digest,
            "cohort_digest": comparison.cohort_digest,
            "operation_bundle_digest": signature,
            "operations": [_operation_summary(op) for op in operations],
            "decision": comparison.decision.value,
            "reason": comparison.reason[:200],
            "complete_paired_scoring": complete,
            "score_delta": comparison.score_delta if complete else None,
            "safety_gate_passed": comparison.safety_gate_passed,
            "cell_regression_gate_passed": comparison.cell_regression_gate_passed,
            "qualification": "paired_training_only" if complete else "execution_incomplete_no_gain_claim",
        })
    # First cover distinct operation bundles, then add differing parent contexts.
    distinct, repeats, signatures = [], [], set()
    for row in pool:
        (repeats if row["operation_bundle_digest"] in signatures else distinct).append(row)
        signatures.add(row["operation_bundle_digest"])
    body = {
        "schema_version": "ecologyrsi-dsh.local-edit-memory/1",
        "run_id": run_id, "before_generation": generation,
        "source": "prior_generation_training_pairs_only",
        "advisory_only": True, "scientific_improvement_established": False,
        "interpretation": "Compare each delta only within its recorded cohort and parent. "
            "Do not rank absolute scores across cohorts or treat a failed edit as universally invalid. "
            "When revisiting the same operation, explain the new evidence or changed parent context; "
            "all new children still require fresh paired evaluation.",
        "records": [],
    }
    for row in distinct + repeats:
        if len(body["records"]) >= MAX_RECORDS:
            break
        trial = {**body, "records": [*body["records"], row]}
        # Reserve the exact serialized size of the final digest field, too.
        if len(canonical_json({**trial, "memory_digest": "0" * 64}).encode()) <= MAX_BYTES:
            body = trial
    return {**body, "memory_digest": digest(body)}
