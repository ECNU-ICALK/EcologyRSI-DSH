"""Shared HTTP helpers with no dependency on the server compatibility facade."""

from __future__ import annotations

import ipaddress
import os
import sysconfig
from pathlib import Path
from typing import Any

from ..core.models import TaskManifest, digest
from ..core.redaction import REDACTED, public_error_summary
from ..evaluators.registry import TOY_DATASET_ID


_EVENT_TYPE_ALIASES = {
    "RunCreated": "run.created",
    "RunStarted": "run.started",
    "RunPaused": "run.paused",
    "RunResumed": "run.resumed",
    "RunCancelled": "run.cancelled",
    "RunFailed": "run.failed",
    "RunCompleted": "run.completed",
    "GenerationAdvanced": "generation.advanced",
    "GenerationBatchStarted": "generation.batch_started",
    "GenerationKnowledgeRetrieved": "knowledge.retrieved",
    "GenerationKnowledgeAssessed": "knowledge.assessed",
    "GenerationAnalyzed": "generation.analyzed",
    "GenerationChampionSelected": "generation.champion_selected",
    "ProposalSubmitted": "proposal.submitted",
    "CandidateSpawned": "candidate.spawned",
    "CandidateFailed": "candidate.failed",
    "CandidateMarkedDuplicate": "candidate.duplicate",
    "ArtifactRecorded": "artifact.recorded",
    "EvaluationRecorded": "evaluation.completed",
    "EvaluationProgressRecorded": "evaluation.progress",
    "EvaluationSampleResultsStarted": "evaluation.sample_results_started",
    "EvaluationSampleResultBatchRecorded": "evaluation.sample_result_batch",
    "EvaluationSampleResultsRecorded": "evaluation.sample_results_completed",
    "EvaluationJudged": "evaluation.judged",
    "PromotionDecided": "promotion.decided",
    "HumanInterventionRecorded": "intervention.recorded",
    "HumanInterventionApplied": "intervention.applied",
    "ExpertConsultationRequested": "expert_consultation.requested",
    "ExpertConsultationAnswered": "expert_consultation.answered",
    "ExpertConsultationApplied": "expert_consultation.applied",
    "EvolutionStageRecorded": "stage.recorded",
    "GatewayRetryScheduled": "gateway.retry_scheduled",
    "ModelUsageRecorded": "model.usage_recorded",
}

_PLUGIN_FILES = {
    "index.html": "text/html; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "assets/js/host.js": "text/javascript; charset=utf-8",
    "assets/js/core.js": "text/javascript; charset=utf-8",
    "assets/js/demo.js": "text/javascript; charset=utf-8",
    "assets/js/catalog.js": "text/javascript; charset=utf-8",
    "assets/js/data.js": "text/javascript; charset=utf-8",
    "assets/js/commands.js": "text/javascript; charset=utf-8",
    "assets/js/render_shell.js": "text/javascript; charset=utf-8",
    "assets/js/render_training.js": "text/javascript; charset=utf-8",
    "assets/js/render_process.js": "text/javascript; charset=utf-8",
    "assets/js/render_candidates.js": "text/javascript; charset=utf-8",
    "assets/js/render_collaboration.js": "text/javascript; charset=utf-8",
    "plugin.json": "application/json; charset=utf-8",
}


def _plugin_root() -> Path:
    configured = os.environ.get("ECOLOGYRSI_PLUGIN_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    # Source checkout layout: <repo>/src/ecologyrsi_dsh/api/shared.py.
    source_root = (
        Path(__file__).resolve().parents[3] / "plugins" / "ecology_evolution"
    )
    if source_root.is_dir():
        return source_root.resolve()
    # Wheel layout configured through pyproject data-files.
    return (
        Path(sysconfig.get_path("data"))
        / "share"
        / "ecologyrsi-dsh"
        / "plugins"
        / "ecology_evolution"
    ).resolve()


def _event_type(kind: str) -> str:
    return _EVENT_TYPE_ALIASES.get(kind, kind.lower().replace("_", "."))


def _public_http_error(exc: BaseException) -> str:
    """Return a useful HTTP error without reflecting credential-like text."""

    summary = public_error_summary(str(exc))
    return type(exc).__name__ if summary in {None, REDACTED} else summary


def _budget_value(task: TaskManifest, key: str, default: int = 0) -> int:
    value = task.budget.get(key, default)
    if isinstance(value, bool):
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _max_generations(task: TaskManifest) -> int:
    """Return the bounded generation budget for a local run."""

    return max(1, task.max_generations)


def _derived_seed(material: Any) -> int:
    """Derive a stable, non-secret local demo seed."""

    return int(digest({"local_demo_seed": material})[:8], 16) & 0x7FFFFFFF


def _expected_partition(task: TaskManifest) -> str:
    dataset_id = task.dataset
    return "validation" if dataset_id == TOY_DATASET_ID else "training_feedback"


def _evaluation_partition(task: TaskManifest, body: dict[str, Any]) -> str:
    expected = _expected_partition(task)
    split = str(body.get("split", expected)).strip().lower().replace("-", "_")
    if split != expected:
        if expected == "validation":
            raise ValueError("HTTP prediction demo only allows the validation partition")
        raise ValueError("温室运行只允许使用训练反馈分区进行自适应评测")
    return split


def _assert_manifest_http_scope(task: TaskManifest) -> None:
    """Reject manifests that try to move the evaluator onto a holdout split."""

    expected = _expected_partition(task)
    task_partition = task.metadata.get("evaluation_partition", expected)
    if task_partition != expected:
        if expected == "validation":
            raise ValueError(
                "HTTP projection only allows the validation partition; task manifest found: "
                + str(task_partition)
            )
        raise ValueError(
            "温室运行只允许训练反馈分区；任务清单请求了：" + str(task_partition)
        )


def _assert_http_scope(state: Any) -> None:
    """Fail closed if a run contains an evaluator partition outside HTTP scope."""

    _assert_manifest_http_scope(state.task_manifest)
    expected = _expected_partition(state.task_manifest)
    partitions = sorted(
        {item.partition for item in state.evaluations if item.partition != expected}
    )
    if partitions:
        if expected == "validation":
            raise ValueError(
                "HTTP projection only allows the validation partition; found: "
                + ", ".join(partitions)
            )
        raise ValueError("温室运行发现越界评测分区：" + ", ".join(partitions))


def _is_loopback_host(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# A private negative sentinel distinguishes continuous advancement from legacy
# non-negative bounded step counts.
AUTO_ADVANCE_CONTINUOUS = -1


def _auto_advance_steps(value: Any) -> int:
    """Validate creation-time advance policy."""

    if isinstance(value, bool):
        return AUTO_ADVANCE_CONTINUOUS if value else 0
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"continuous", "all", "auto", "automatic", "自动"}:
            return AUTO_ADVANCE_CONTINUOUS
        raise ValueError(
            "auto_advance must be false, continuous, or an integer between 0 and 32"
        )
    if not isinstance(value, int) or not 0 <= value <= 32:
        raise ValueError(
            "auto_advance must be false, continuous, or an integer between 0 and 32"
        )
    return value


def _parse_steps(body: dict[str, Any]) -> int:
    """Parse the bounded step count shared by advance command paths."""

    raw_steps = body.get("steps", 1)
    if isinstance(raw_steps, bool) or not isinstance(raw_steps, int):
        raise TypeError("steps must be an integer")
    if raw_steps < 1 or raw_steps > 32:
        raise ValueError("steps must be between 1 and 32")
    return raw_steps


def _request_integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    """Validate a compact HTTP integer before manifest construction."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value
