"""External campaign guard decisions; never changes a frozen run contract.

Provider usage is delayed and incomplete, so the token threshold requests a
pause after observed usage crosses it. It is not a billing or admission cap.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import math


def utc_timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("campaign timestamps must include a timezone")
    return parsed.timestamp()


@dataclass(frozen=True)
class CampaignLimits:
    deadline: float
    reported_token_pause_threshold: int
    stall_seconds: int

    def __post_init__(self) -> None:
        if isinstance(self.deadline, bool) or not math.isfinite(self.deadline):
            raise ValueError("deadline must be a finite timestamp")
        for value in (self.reported_token_pause_threshold, self.stall_seconds):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("campaign limits must be positive integers")


def progress_signature(projection: Mapping) -> tuple:
    """Only completed work advances the stall clock, never usage/heartbeats."""
    progress = projection.get("execution_progress") or {}
    stage = progress.get("stage_progress") or {}
    skills = (projection.get("dsh_runtime") or {}).get("skill_invocation") or {}
    return (
        projection.get("generation", 0), projection.get("candidates_count", 0),
        progress.get("completed_steps", 0), progress.get("completed_generations", 0),
        stage.get("completed_origins", 0), skills.get("verified_call_count", 0),
    )


def pause_reason(projection: Mapping, limits: CampaignLimits, *, now: float,
                 last_progress_at: float) -> str | None:
    if projection.get("status") != "running":
        return None
    if now >= limits.deadline:
        return "campaign_deadline"
    tokens = projection.get("tokens_used")
    if (projection.get("token_usage_available") is True
            and isinstance(tokens, int) and not isinstance(tokens, bool)
            and tokens >= limits.reported_token_pause_threshold):
        return "campaign_reported_token_threshold"
    if now - last_progress_at >= limits.stall_seconds:
        return "campaign_no_completed_work"
    return None


def observe_progress(projection: Mapping, saved: Mapping, *, now: float) -> dict:
    """Start a fresh stall interval on an observed resume, not every poll.

    Persisted running observations survive watcher restarts. Paused maintenance
    time is not failed execution time; deadline and token checks remain absolute.
    """
    result = dict(saved)
    signature = list(progress_signature(projection))
    status = projection.get("status")
    resumed = status == "running" and saved.get("observed_status") != "running"
    if resumed or signature != saved.get("signature") or "last_progress_at" not in saved:
        result.update(signature=signature, last_progress_at=now)
    if resumed and saved.get("observed_status") == "paused":
        result.pop("pause_reason", None)
    result["observed_status"] = status
    return result
