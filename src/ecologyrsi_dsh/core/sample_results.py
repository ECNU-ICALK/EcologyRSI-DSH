"""Private, bounded archives for completed per-sample evaluation results."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
import json
import math
from typing import Any
import zlib

from .models import Evaluation, canonical_json, digest
from .redaction import safe_remote_reason_code


SAMPLE_RESULTS_SCHEMA_VERSION = "ecologyrsi-dsh.evaluation-sample-results/1"
SAMPLE_RESULT_BATCH_SCHEMA_VERSION = (
    "ecologyrsi-dsh.evaluation-sample-result-batch/1"
)
SAMPLE_RESULTS_ARCHIVE_VERSION = "ecologyrsi-dsh.evaluation-sample-results-archive/1"
SAMPLE_RESULTS_ENCODING = "zlib+base64+canonical-json"
SAMPLE_REWARD_DEFINITION_V1 = "absolute_error_improvement_vs_persistence@1"
SAMPLE_REWARD_DEFINITION_V2 = (
    "absolute_error_improvement_vs_fit_selected_baseline@2"
)
SAMPLE_REWARD_DEFINITION = SAMPLE_REWARD_DEFINITION_V2
SUPPORTED_SAMPLE_REWARD_DEFINITIONS = frozenset(
    {SAMPLE_REWARD_DEFINITION_V1, SAMPLE_REWARD_DEFINITION_V2}
)
MAX_SAMPLE_RESULTS_RECORDS = 100_000
MAX_SAMPLE_RESULTS_COMPRESSED_BYTES = 32 * 1024 * 1024
MAX_SAMPLE_RESULTS_UNCOMPRESSED_BYTES = 128 * 1024 * 1024


def build_sample_results(
    candidate_id: str,
    scoring_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Project evaluator rows onto the stable browser result contract."""

    candidate_id = _required_text(candidate_id, "candidate_id")
    if len(scoring_rows) > MAX_SAMPLE_RESULTS_RECORDS:
        raise ValueError("sample result archive exceeds the record limit")
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(scoring_rows):
        if not isinstance(source, Mapping):
            raise TypeError(f"scoring_rows[{index}] must be a mapping")
        observed = _finite_number(source.get("observed"), f"scoring_rows[{index}].observed")
        predicted = _finite_number(
            source.get("predicted"), f"scoring_rows[{index}].predicted"
        )
        baseline = _finite_number(
            source.get("baseline"), f"scoring_rows[{index}].baseline"
        )
        absolute_error = abs(predicted - observed)
        baseline_absolute_error = abs(baseline - observed)
        reward = baseline_absolute_error - absolute_error
        status = str(
            source.get("sample_execution_status", source.get("status", "succeeded"))
        ).strip().casefold()
        if status not in {"succeeded", "failed"}:
            raise ValueError(f"scoring_rows[{index}].status is unsupported")
        attempts = _bounded_integer(
            source.get(
                "sample_execution_attempts", source.get("attempts", 0)
            ),
            f"scoring_rows[{index}].attempts",
            maximum=8,
        )
        retry_count = _bounded_integer(
            source.get(
                "sample_execution_retry_count",
                source.get("retry_count", max(0, attempts - 1)),
            ),
            f"scoring_rows[{index}].retry_count",
            maximum=8,
        )
        raw_fallback = source.get("scoring_fallback")
        scoring_fallback = (
            str(raw_fallback).strip()[:160]
            if raw_fallback is not None and str(raw_fallback).strip()
            else None
        )
        raw_fallback_source = source.get("scoring_fallback_source")
        scoring_fallback_source = (
            str(raw_fallback_source).strip()[:160]
            if raw_fallback_source is not None
            and str(raw_fallback_source).strip()
            else None
        )
        raw_prediction_source = source.get("prediction_source")
        prediction_source = (
            "scoring_fallback"
            if scoring_fallback is not None
            else str(raw_prediction_source).strip()[:160]
            if raw_prediction_source is not None
            and str(raw_prediction_source).strip()
            else "agent_tool_prediction"
        )
        raw_failure = source.get("sample_execution_failure")
        failure_class = (
            str(raw_failure.get("class") or "").strip()[:120] or None
            if isinstance(raw_failure, Mapping)
            else str(source.get("failure_class") or "").strip()[:120] or None
        )
        failure_summary = _bounded_failure_summary(
            source.get(
                "sample_execution_failure_summary",
                source.get("failure_summary"),
            ),
            f"scoring_rows[{index}].failure_summary",
        )
        target_timestamp = source.get("target_timestamp", source.get("timestamp"))
        origin_timestamp = source.get("origin_timestamp", source.get("timestamp"))
        projected = {
            "sample_index": _bounded_integer(
                source.get("sample_index", index + 1),
                f"scoring_rows[{index}].sample_index",
                maximum=MAX_SAMPLE_RESULTS_RECORDS,
                minimum=1,
            ),
            "sample_id": _required_text(
                source.get("sample_id"), f"scoring_rows[{index}].sample_id"
            ),
            "candidate_id": candidate_id,
            "target": _required_text(
                source.get("target"), f"scoring_rows[{index}].target"
            ),
            "unit": _required_text(
                source.get("unit") or "unknown",
                f"scoring_rows[{index}].unit",
            ),
            "horizon_hours": _horizon(
                source.get("horizon_hours"),
                f"scoring_rows[{index}].horizon_hours",
            ),
            "origin_timestamp": _timestamp(
                origin_timestamp, f"scoring_rows[{index}].origin_timestamp"
            ),
            "target_timestamp": _timestamp(
                target_timestamp, f"scoring_rows[{index}].target_timestamp"
            ),
            "observed": observed,
            "predicted": predicted,
            "prediction_source": prediction_source,
            "baseline": baseline,
            "absolute_error": absolute_error,
            # Positive reward means lower absolute error than the explicitly
            # versioned scoring comparator.
            "reward": reward,
            "status": status,
            "attempts": attempts,
            "retry_count": retry_count,
            "scoring_fallback": scoring_fallback,
            "scoring_fallback_source": scoring_fallback_source,
            "failure_class": failure_class,
        }
        has_baseline_profile = any(
            name in source
            for name in (
                "baseline_id",
                "baseline_profile_digest",
                "model_reference_baseline",
            )
        )
        if has_baseline_profile:
            projected["baseline_id"] = _required_text(
                source.get("baseline_id"), f"scoring_rows[{index}].baseline_id"
            )
            if projected["baseline_id"] not in {"persistence", "seasonal_24h"}:
                raise ValueError(f"scoring_rows[{index}].baseline_id is unsupported")
            projected["baseline_profile_digest"] = _required_text(
                source.get("baseline_profile_digest"),
                f"scoring_rows[{index}].baseline_profile_digest",
            )
            projected["model_reference_baseline"] = _finite_number(
                source.get("model_reference_baseline"),
                f"scoring_rows[{index}].model_reference_baseline",
            )
            raw_baseline_fallback = source.get("baseline_fallback")
            if raw_baseline_fallback is not None:
                projected["baseline_fallback"] = _required_text(
                    raw_baseline_fallback,
                    f"scoring_rows[{index}].baseline_fallback",
                )[:160]
        if "normalization_scale" in source:
            normalization_scale = _finite_number(
                source.get("normalization_scale"),
                f"scoring_rows[{index}].normalization_scale",
            )
            if normalization_scale <= 0:
                raise ValueError(
                    f"scoring_rows[{index}].normalization_scale must be positive"
                )
            raw_normalized_reward = reward / normalization_scale
            projected.update(
                {
                    "normalization_scale": normalization_scale,
                    "raw_normalized_reward": raw_normalized_reward,
                    "normalized_reward": max(-1.0, min(1.0, raw_normalized_reward)),
                }
            )
        if "failed_reward_policy" in source:
            failed_reward_policy = _required_text(
                source.get("failed_reward_policy"),
                f"scoring_rows[{index}].failed_reward_policy",
            )
            if failed_reward_policy != "nonpositive" or status != "failed":
                raise ValueError(
                    f"scoring_rows[{index}].failed_reward_policy is invalid"
                )
            if projected["reward"] > 1e-12:
                raise ValueError(
                    f"scoring_rows[{index}] failed reward must be non-positive"
                )
            projected["failed_reward_policy"] = failed_reward_policy
        if failure_summary is not None:
            if status != "failed":
                raise ValueError(
                    f"scoring_rows[{index}].failure_summary requires failed status"
                )
            projected["failure_summary"] = failure_summary
        rows.append(projected)
    canonical_json(rows)
    return tuple(rows)


def sample_results_event_payload(
    evaluation: Evaluation,
    rows: Sequence[Mapping[str, Any]],
    *,
    revision: str,
) -> dict[str, Any]:
    """Build the private event payload stored beside a scientific evaluation."""

    projected = [dict(row) for row in rows]
    if not projected:
        raise ValueError("sample result completion must not be empty")
    for index, row in enumerate(projected):
        if row.get("candidate_id") != evaluation.candidate_id:
            raise ValueError(
                f"sample_results[{index}] belongs to a different candidate"
            )
    archive = encode_sample_results(projected)
    cohort_digest = sample_results_cohort_digest(projected)
    return {
        "schema_version": SAMPLE_RESULTS_SCHEMA_VERSION,
        "run_id": evaluation.run_id,
        "revision": _required_text(revision, "revision"),
        "evaluation_id": evaluation.evaluation_id,
        "candidate_id": evaluation.candidate_id,
        "reward_definition": SAMPLE_REWARD_DEFINITION,
        "positive_is_better": True,
        "record_count": archive["record_count"],
        "result_digest": archive["result_digest"],
        "cohort_digest": cohort_digest,
        "archive": archive,
    }


def sample_result_batch_event_payload(
    run_id: str,
    candidate_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    revision: str,
    batch_index: int,
) -> dict[str, Any]:
    """Build one append-only, host-finalized result batch event."""

    run_id = _required_text(run_id, "run_id")
    candidate_id = _required_text(candidate_id, "candidate_id")
    revision = _required_text(revision, "revision")
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 1:
        raise ValueError("batch_index must be a positive integer")
    projected = [dict(row) for row in rows]
    if not projected:
        raise ValueError("sample result batch must not be empty")
    for index, row in enumerate(projected):
        if row.get("candidate_id") != candidate_id:
            raise ValueError(
                f"sample_results[{index}] belongs to a different candidate"
            )
    archive = encode_sample_results(projected)
    return {
        "schema_version": SAMPLE_RESULT_BATCH_SCHEMA_VERSION,
        "run_id": run_id,
        "revision": revision,
        "candidate_id": candidate_id,
        "batch_index": batch_index,
        "reward_definition": SAMPLE_REWARD_DEFINITION,
        "positive_is_better": True,
        "record_count": archive["record_count"],
        "sample_ids": [str(row["sample_id"]) for row in projected],
        "sample_indices": [int(row["sample_index"]) for row in projected],
        "result_digest": archive["result_digest"],
        "cohort_digest": sample_results_cohort_digest(projected),
        "archive": archive,
    }


def encode_sample_results(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Encode complete result rows without placing labels in evaluation metrics."""

    if len(rows) > MAX_SAMPLE_RESULTS_RECORDS:
        raise ValueError("sample result archive exceeds the record limit")
    projected = [dict(row) for row in rows]
    raw = canonical_json(projected).encode("utf-8")
    if len(raw) > MAX_SAMPLE_RESULTS_UNCOMPRESSED_BYTES:
        raise ValueError("sample result archive exceeds the uncompressed byte limit")
    compressed = zlib.compress(raw, level=9)
    if len(compressed) > MAX_SAMPLE_RESULTS_COMPRESSED_BYTES:
        raise ValueError("sample result archive exceeds the compressed byte limit")
    return {
        "schema_version": SAMPLE_RESULTS_ARCHIVE_VERSION,
        "encoding": SAMPLE_RESULTS_ENCODING,
        "record_count": len(projected),
        "uncompressed_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "result_digest": digest(projected),
        "payload": base64.b64encode(compressed).decode("ascii"),
    }


def decode_sample_results(
    event_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Validate and safely decode one private sample-results event payload."""

    if event_payload.get("schema_version") != SAMPLE_RESULTS_SCHEMA_VERSION:
        raise ValueError("unsupported sample results schema version")
    _required_text(event_payload.get("run_id"), "run_id")
    _required_text(event_payload.get("revision"), "revision")
    _validate_reward_contract(event_payload)
    candidate_id = _required_text(event_payload.get("candidate_id"), "candidate_id")
    _required_text(event_payload.get("evaluation_id"), "evaluation_id")
    record_count = _bounded_integer(
        event_payload.get("record_count"),
        "record_count",
        maximum=MAX_SAMPLE_RESULTS_RECORDS,
    )
    expected_digest = _required_text(
        event_payload.get("result_digest"), "result_digest"
    )
    expected_cohort_digest = _required_text(
        event_payload.get("cohort_digest"), "cohort_digest"
    )
    rows = _decode_sample_results_archive(
        event_payload.get("archive"),
        candidate_id=candidate_id,
        record_count=record_count,
        expected_digest=expected_digest,
    )
    if sample_results_cohort_digest(rows) != expected_cohort_digest:
        raise ValueError("sample results cohort digest does not match")
    return rows


def decode_sample_result_batch(
    event_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Decode one host-finalized batch after validating its revision metadata."""

    if event_payload.get("schema_version") != SAMPLE_RESULT_BATCH_SCHEMA_VERSION:
        raise ValueError("unsupported sample result batch schema version")
    _required_text(event_payload.get("run_id"), "run_id")
    candidate_id = _required_text(event_payload.get("candidate_id"), "candidate_id")
    _required_text(event_payload.get("revision"), "revision")
    _bounded_integer(
        event_payload.get("batch_index"),
        "batch_index",
        maximum=10_000_000,
        minimum=1,
    )
    _validate_reward_contract(event_payload)
    record_count = _bounded_integer(
        event_payload.get("record_count"),
        "record_count",
        maximum=MAX_SAMPLE_RESULTS_RECORDS,
    )
    expected_digest = _required_text(
        event_payload.get("result_digest"), "result_digest"
    )
    expected_cohort_digest = _required_text(
        event_payload.get("cohort_digest"), "cohort_digest"
    )
    rows = _decode_sample_results_archive(
        event_payload.get("archive"),
        candidate_id=candidate_id,
        record_count=record_count,
        expected_digest=expected_digest,
    )
    if sample_results_cohort_digest(rows) != expected_cohort_digest:
        raise ValueError("sample result batch cohort digest does not match")
    sample_ids = event_payload.get("sample_ids")
    sample_indices = event_payload.get("sample_indices")
    if sample_ids != [str(row["sample_id"]) for row in rows]:
        raise ValueError("sample result batch sample_ids do not match its archive")
    if sample_indices != [int(row["sample_index"]) for row in rows]:
        raise ValueError("sample result batch sample_indices do not match its archive")
    return rows


def _decode_sample_results_archive(
    archive: Any,
    *,
    candidate_id: str,
    record_count: int,
    expected_digest: str,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(archive, Mapping):
        raise ValueError("sample results archive is missing")
    if archive.get("schema_version") != SAMPLE_RESULTS_ARCHIVE_VERSION:
        raise ValueError("unsupported sample results archive version")
    if archive.get("encoding") != SAMPLE_RESULTS_ENCODING:
        raise ValueError("unsupported sample results archive encoding")
    archive_count = _bounded_integer(
        archive.get("record_count"),
        "archive.record_count",
        maximum=MAX_SAMPLE_RESULTS_RECORDS,
    )
    uncompressed_bytes = _bounded_integer(
        archive.get("uncompressed_bytes"),
        "archive.uncompressed_bytes",
        maximum=MAX_SAMPLE_RESULTS_UNCOMPRESSED_BYTES,
    )
    compressed_bytes = _bounded_integer(
        archive.get("compressed_bytes"),
        "archive.compressed_bytes",
        maximum=MAX_SAMPLE_RESULTS_COMPRESSED_BYTES,
    )
    archive_digest = _required_text(
        archive.get("result_digest"), "archive.result_digest"
    )
    if archive_count != record_count or archive_digest != expected_digest:
        raise ValueError("sample results archive metadata does not match its event")
    encoded = archive.get("payload")
    if not isinstance(encoded, str):
        raise ValueError("sample results archive payload must be text")
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("sample results archive payload is not valid base64") from exc
    if len(compressed) != compressed_bytes:
        raise ValueError("sample results compressed byte count does not match")
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(
            compressed, MAX_SAMPLE_RESULTS_UNCOMPRESSED_BYTES + 1
        )
    except zlib.error as exc:
        raise ValueError("sample results archive cannot be decompressed") from exc
    if inflater.unconsumed_tail or len(raw) > MAX_SAMPLE_RESULTS_UNCOMPRESSED_BYTES:
        raise ValueError("sample results archive exceeds the decompression limit")
    try:
        raw += inflater.flush(
            max(1, MAX_SAMPLE_RESULTS_UNCOMPRESSED_BYTES - len(raw) + 1)
        )
    except zlib.error as exc:
        raise ValueError("sample results archive cannot be decompressed") from exc
    if (
        len(raw) > MAX_SAMPLE_RESULTS_UNCOMPRESSED_BYTES
        or len(raw) != uncompressed_bytes
        or inflater.unconsumed_tail
        or inflater.unused_data
        or not inflater.eof
    ):
        raise ValueError("sample results uncompressed byte count does not match")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("sample results archive is not valid JSON") from exc
    if not isinstance(decoded, list) or len(decoded) != record_count:
        raise ValueError("sample results archive record count does not match")
    rows = build_sample_results(candidate_id, decoded)
    if digest(rows) != expected_digest:
        raise ValueError("sample results archive digest does not match")
    return rows


def _validate_reward_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("reward_definition") not in SUPPORTED_SAMPLE_REWARD_DEFINITIONS:
        raise ValueError("unsupported sample reward definition")
    if payload.get("positive_is_better") is not True:
        raise ValueError("sample reward direction is invalid")


def _bounded_failure_summary(value: Any, name: str) -> dict[str, Any] | None:
    """Project only bounded public decisions and tool identifiers."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"decisions", "tools"}:
        raise ValueError(f"{name} must contain decisions and tools")
    raw_decisions = value.get("decisions")
    raw_tools = value.get("tools")
    if (
        not isinstance(raw_decisions, (list, tuple))
        or not 1 <= len(raw_decisions) <= 8
    ):
        raise ValueError(f"{name}.decisions must contain 1-8 items")
    if (
        not isinstance(raw_tools, (list, tuple))
        or not 1 <= len(raw_tools) <= 8
    ):
        raise ValueError(f"{name}.tools must contain 1-8 items")

    decisions: list[dict[str, str]] = []
    decision_fields = {"role", "decision", "status", "model_id", "reason_code"}
    for index, raw in enumerate(raw_decisions):
        if not isinstance(raw, Mapping) or set(raw) - decision_fields:
            raise ValueError(f"{name}.decisions[{index}] has unsupported fields")
        item: dict[str, str] = {}
        for field in ("role", "decision", "status"):
            item[field] = _bounded_summary_text(
                raw.get(field), f"{name}.decisions[{index}].{field}"
            )
        for field in ("model_id", "reason_code"):
            if field not in raw:
                continue
            projected = _bounded_summary_text(
                raw.get(field), f"{name}.decisions[{index}].{field}", maximum=200
            )
            if field == "reason_code" and item["role"].casefold().startswith("remote_"):
                projected = safe_remote_reason_code(projected)
            item[field] = projected
        decisions.append(item)

    tools: list[dict[str, str]] = []
    tool_fields = {"tool_id", "version", "status"}
    for index, raw in enumerate(raw_tools):
        if not isinstance(raw, Mapping) or set(raw) - tool_fields:
            raise ValueError(f"{name}.tools[{index}] has unsupported fields")
        tools.append(
            {
                field: _bounded_summary_text(
                    raw.get(field), f"{name}.tools[{index}].{field}"
                )
                for field in ("tool_id", "version", "status")
            }
        )
    return {"decisions": decisions, "tools": tools}


def _bounded_summary_text(value: Any, name: str, *, maximum: int = 160) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()[:maximum]


def sample_results_cohort_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Digest a cohort independently of microbatch completion order."""

    projected = [dict(row) for row in rows]
    sample_ids = [
        _required_text(row.get("sample_id"), "sample result sample_id")
        for row in projected
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample result cohort contains duplicate sample_id values")
    sample_indices = [
        _bounded_integer(
            row.get("sample_index"),
            "sample result sample_index",
            maximum=MAX_SAMPLE_RESULTS_RECORDS,
            minimum=1,
        )
        for row in projected
    ]
    if len(sample_indices) != len(set(sample_indices)):
        raise ValueError("sample result cohort contains duplicate sample_index values")
    return digest(
        sorted(projected, key=lambda row: str(row.get("sample_id", "")))
    )


def sample_results_event_id(evaluation: Evaluation) -> str:
    """Return a run-scoped idempotency key for the private result event."""

    identity = digest(
        {
            "run_id": evaluation.run_id,
            "evaluation_id": evaluation.evaluation_id,
            "candidate_id": evaluation.candidate_id,
        }
    )
    return f"evaluation-sample-results:{identity}"


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _horizon(value: Any, name: str) -> int | float:
    number = _finite_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return int(number) if number.is_integer() else number


def _timestamp(value: Any, name: str) -> str | int | float:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return int(number) if number.is_integer() else number
    raise ValueError(f"{name} must be a finite number or non-empty text")


def _bounded_integer(
    value: Any,
    name: str,
    *,
    maximum: int,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} is outside the supported range")
    return value


__all__ = [
    "MAX_SAMPLE_RESULTS_RECORDS",
    "SAMPLE_RESULT_BATCH_SCHEMA_VERSION",
    "SAMPLE_REWARD_DEFINITION",
    "SAMPLE_REWARD_DEFINITION_V1",
    "SAMPLE_REWARD_DEFINITION_V2",
    "SUPPORTED_SAMPLE_REWARD_DEFINITIONS",
    "SAMPLE_RESULTS_ARCHIVE_VERSION",
    "SAMPLE_RESULTS_ENCODING",
    "SAMPLE_RESULTS_SCHEMA_VERSION",
    "build_sample_results",
    "decode_sample_result_batch",
    "decode_sample_results",
    "encode_sample_results",
    "sample_results_event_id",
    "sample_result_batch_event_payload",
    "sample_results_cohort_digest",
    "sample_results_event_payload",
]
