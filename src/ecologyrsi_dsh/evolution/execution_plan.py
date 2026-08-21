"""Deterministic execution plans derived from aggregate generation feedback.

The plan is host-owned.  It is derived only from ``GenerationAnalysis``
aggregates, never from raw samples, predictions, or observed labels.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..core.models import JsonObject, canonical_json, digest
from .analysis import GenerationAnalysis

DERIVED_EXECUTION_PLAN_VERSION = "ecologyrsi-dsh.derived-execution-plan/1"

_REPAIR_TOOLS = (
    "bounded-projection-repair",
    "bounded-persistence-fallback",
)
_TRANSIENT_FAILURES = frozenset(
    {"timeout", "rate_limited", "connection", "remote_transient"}
)
_CRITIC_FAILURES = frozenset({"constraint_rejected", "invalid_output"})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _normalize_failure_class(value: Any) -> str:
    name = str(value or "").strip().casefold().replace("-", "_")[:100]
    if not name:
        return "tool_error"
    if "timeout" in name or "timed_out" in name:
        return "timeout"
    if "429" in name or "rate" in name or "queue" in name:
        return "rate_limited"
    if "connect" in name:
        return "connection"
    if "transient" in name or "unavailable" in name:
        return "remote_transient"
    if "constraint" in name or "physical" in name or "range" in name:
        return "constraint_rejected"
    if "invalid_output" in name or "contract" in name or "parse" in name:
        return "invalid_output"
    if "invalid_sample" in name or "invalid_input" in name:
        return "invalid_sample_input"
    if "numeric" in name or "arithmetic" in name or "floating" in name:
        return "numerical"
    return name


def _route_for(failure_class: str) -> str:
    if failure_class in _TRANSIENT_FAILURES:
        return "retry_with_exponential_backoff"
    if failure_class in _CRITIC_FAILURES:
        return "critic_feedback_then_registered_repair"
    if failure_class == "invalid_sample_input":
        return "record_input_failure_and_continue_fixed_cohort"
    return "record_failure_then_continue_fixed_cohort"


@dataclass(frozen=True, slots=True)
class DerivedExecutionPlan:
    """Frozen, replayable controls for the next generation's sample runtime."""

    source_generation: int | None
    source_analysis_digest: str | None
    failure_profile: tuple[Mapping[str, Any], ...] = ()
    sample_max_attempts: int = 3
    plan_max_attempts: int = 3
    retry_backoff_seconds: float = 0.0
    repair_sequence: tuple[str, ...] = _REPAIR_TOOLS
    schema_version: str = DERIVED_EXECUTION_PLAN_VERSION
    plan_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != DERIVED_EXECUTION_PLAN_VERSION:
            raise ValueError("unsupported derived execution plan version")
        if self.source_generation is None:
            if self.source_analysis_digest is not None:
                raise ValueError("initial execution plan cannot reference an analysis digest")
        else:
            object.__setattr__(
                self,
                "source_generation",
                _integer(
                    self.source_generation,
                    "source_generation",
                    minimum=0,
                    maximum=1_000_000,
                ),
            )
            object.__setattr__(
                self,
                "source_analysis_digest",
                _text(self.source_analysis_digest, "source_analysis_digest"),
            )
        object.__setattr__(
            self,
            "sample_max_attempts",
            _integer(
                self.sample_max_attempts,
                "sample_max_attempts",
                minimum=1,
                maximum=8,
            ),
        )
        object.__setattr__(
            self,
            "plan_max_attempts",
            _integer(
                self.plan_max_attempts,
                "plan_max_attempts",
                minimum=1,
                maximum=8,
            ),
        )
        delay = self.retry_backoff_seconds
        if isinstance(delay, bool) or not isinstance(delay, (int, float)):
            raise TypeError("retry_backoff_seconds must be a number")
        if not math.isfinite(float(delay)) or not 0 <= float(delay) <= 60:
            raise ValueError("retry_backoff_seconds must be between 0 and 60")
        object.__setattr__(self, "retry_backoff_seconds", float(delay))

        profile: list[dict[str, Any]] = []
        seen_classes: set[str] = set()
        for raw in self.failure_profile:
            if not isinstance(raw, Mapping):
                raise TypeError("failure_profile items must be objects")
            unknown = set(raw) - {"failure_class", "count", "route"}
            if unknown:
                raise ValueError("failure_profile contains unsupported fields")
            failure_class = _text(raw.get("failure_class"), "failure_class")
            if failure_class != _normalize_failure_class(failure_class):
                raise ValueError("failure_class must be normalized")
            if failure_class in seen_classes:
                raise ValueError("failure_profile classes must be unique")
            seen_classes.add(failure_class)
            count = _integer(raw.get("count"), "failure count", minimum=1, maximum=10**9)
            route = _text(raw.get("route"), "failure route")
            if route != _route_for(failure_class):
                raise ValueError("failure route does not match the host registry")
            profile.append(
                {"failure_class": failure_class, "count": count, "route": route}
            )
        if profile != sorted(
            profile, key=lambda item: (-int(item["count"]), str(item["failure_class"]))
        ):
            raise ValueError("failure_profile must use deterministic ordering")
        canonical_json(profile)
        object.__setattr__(self, "failure_profile", tuple(profile))

        sequence = tuple(_text(item, "repair tool") for item in self.repair_sequence)
        if len(sequence) != len(_REPAIR_TOOLS) or set(sequence) != set(_REPAIR_TOOLS):
            raise ValueError("repair_sequence must contain each registered repair tool once")
        object.__setattr__(self, "repair_sequence", sequence)

        expected = digest(self.identity_dict())
        if self.plan_digest and self.plan_digest != expected:
            raise ValueError("derived execution plan digest mismatch")
        object.__setattr__(self, "plan_digest", expected)

    @property
    def derived_from_feedback(self) -> bool:
        return self.source_generation is not None

    @property
    def execution_digest(self) -> str:
        """Digest only controls that can change candidate execution semantics."""

        return digest(
            {
                "schema_version": self.schema_version,
                "failure_profile": [dict(item) for item in self.failure_profile],
                "sample_max_attempts": self.sample_max_attempts,
                "plan_max_attempts": self.plan_max_attempts,
                "retry_backoff_seconds": self.retry_backoff_seconds,
                "repair_sequence": list(self.repair_sequence),
            }
        )

    def identity_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "source_generation": self.source_generation,
            "source_analysis_digest": self.source_analysis_digest,
            "derived_from_feedback": self.derived_from_feedback,
            "execution_digest": self.execution_digest,
            "failure_profile": [dict(item) for item in self.failure_profile],
            "sample_max_attempts": self.sample_max_attempts,
            "plan_max_attempts": self.plan_max_attempts,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "repair_sequence": list(self.repair_sequence),
        }

    def to_dict(self) -> JsonObject:
        return {**self.identity_dict(), "plan_digest": self.plan_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DerivedExecutionPlan:
        if not isinstance(value, Mapping):
            raise TypeError("derived execution plan must be an object")
        data = dict(value)
        derived = data.pop("derived_from_feedback", None)
        stored_execution_digest = data.pop("execution_digest", None)
        item = cls(**data)
        if derived is not None and derived is not item.derived_from_feedback:
            raise ValueError("derived_from_feedback does not match source generation")
        if (
            stored_execution_digest is not None
            and stored_execution_digest != item.execution_digest
        ):
            raise ValueError("derived execution semantics digest mismatch")
        return item


def derive_execution_plan(
    previous_analysis: GenerationAnalysis | None,
) -> DerivedExecutionPlan:
    """Purely derive next-generation controls from aggregate failure evidence."""

    if previous_analysis is None:
        return DerivedExecutionPlan(
            source_generation=None,
            source_analysis_digest=None,
        )

    failure_counts: Counter[str] = Counter()
    recovered_counts: Counter[str] = Counter()
    attempted = 0
    failed = 0
    coverage_missed = False
    repair_attempts = 0
    repair_successes = 0
    for row in previous_analysis.sample_failures:
        raw_attempted = row.get("attempted")
        raw_failed = row.get("failed")
        if isinstance(raw_attempted, int) and not isinstance(raw_attempted, bool):
            attempted += max(0, raw_attempted)
        if isinstance(raw_failed, int) and not isinstance(raw_failed, bool):
            failed += max(0, raw_failed)
        if row.get("coverage_pass") is False:
            coverage_missed = True
        raw_counts = row.get("failure_counts")
        if isinstance(raw_counts, Mapping):
            for raw_name, raw_count in raw_counts.items():
                if (
                    isinstance(raw_count, int)
                    and not isinstance(raw_count, bool)
                    and raw_count > 0
                ):
                    failure_counts[_normalize_failure_class(raw_name)] += raw_count
        raw_recovered = row.get("recovered_by_failure_class")
        if isinstance(raw_recovered, Mapping):
            for raw_name, raw_count in raw_recovered.items():
                if (
                    isinstance(raw_count, int)
                    and not isinstance(raw_count, bool)
                    and raw_count > 0
                ):
                    recovered_counts[_normalize_failure_class(raw_name)] += raw_count
        raw_repair_outcomes = row.get("repair_tool_outcomes")
        if isinstance(raw_repair_outcomes, Mapping):
            for tool_id in _REPAIR_TOOLS:
                outcomes = raw_repair_outcomes.get(tool_id)
                if not isinstance(outcomes, Mapping):
                    continue
                for status, count in outcomes.items():
                    if (
                        status in {"completed", "failed", "rejected"}
                        and isinstance(count, int)
                        and not isinstance(count, bool)
                        and count > 0
                    ):
                        repair_attempts += count
        raw_repair_count = row.get("repair_count")
        if (
            isinstance(raw_repair_count, int)
            and not isinstance(raw_repair_count, bool)
            and raw_repair_count > 0
        ):
            repair_successes += raw_repair_count

    profile = tuple(
        {
            "failure_class": failure_class,
            "count": count,
            "route": _route_for(failure_class),
        }
        for failure_class, count in sorted(
            failure_counts.items(), key=lambda item: (-item[1], item[0])
        )[:32]
    )
    unresolved_counts = Counter(
        {
            name: max(0, count - recovered_counts[name])
            for name, count in failure_counts.items()
        }
    )
    transient_count = sum(unresolved_counts[name] for name in _TRANSIENT_FAILURES)
    critic_count = sum(failure_counts[name] for name in _CRITIC_FAILURES)
    failure_rate = failed / attempted if attempted else 0.0
    repair_success_rate = (
        min(1.0, repair_successes / repair_attempts)
        if repair_attempts
        else None
    )

    extra_attempts = int(bool(failed)) + int(coverage_missed or failure_rate >= 0.2)
    if transient_count:
        extra_attempts += 1
    sample_attempts = min(8, 3 + extra_attempts)
    plan_attempts = min(8, 3 + int(bool(transient_count)) + int(coverage_missed))
    retry_backoff = 2.0 if transient_count else 0.0
    repair_sequence = (
        tuple(reversed(_REPAIR_TOOLS))
        if critic_count and (repair_success_rate is None or repair_success_rate < 0.75)
        else _REPAIR_TOOLS
    )
    return DerivedExecutionPlan(
        source_generation=previous_analysis.generation,
        source_analysis_digest=previous_analysis.analysis_digest,
        failure_profile=profile,
        sample_max_attempts=sample_attempts,
        plan_max_attempts=plan_attempts,
        retry_backoff_seconds=retry_backoff,
        repair_sequence=repair_sequence,
    )


__all__ = [
    "DERIVED_EXECUTION_PLAN_VERSION",
    "DerivedExecutionPlan",
    "derive_execution_plan",
]
