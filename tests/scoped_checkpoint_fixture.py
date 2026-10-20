"""Current scope-bound checkpoint fixture without lifecycle side effects."""
from ecologyrsi_dsh.core.models import digest
from ecologyrsi_dsh.core.trajectory import EvaluationScope
def sha(label):
    return digest({"label":label})

def checkpoint(scope: EvaluationScope) -> dict[str, object]:
        return {
            "schema_version": "ecologyrsi-dsh.sample-checkpoint/2",
            "candidate_revision_id": scope.candidate_revision_id,
            "evaluation_phase": scope.phase.value,
            "formal_batch_index": scope.batch_index,
            "holdout_arm": None,
            "cohort_digest": scope.cohort_digest,
            "execution_scope_digest": scope.scope_key,
            "sample_cohort_digest": sha(f"samples:{scope.scope_key}"),
            "execution_context_digest": sha(f"context:{scope.scope_key}"),
            "sample_count": scope.origin_count * 9,
        }
