"""Host-owned cellwise calibrated residual prediction intervals.

Point predictors are frozen before this module sees calibration-UQ rows.  The
module never fits or modifies a point model; it only binds finite-sample
residual quantiles to the supplied point-artifact and data-protocol digests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from ..core.models import digest


UQ_POLICY_ID = "cellwise_time_block_calibrated_residual@1"
UQ_ARTIFACT_SCHEMA = "ecologyrsi-dsh.baseline-uq-artifact/1"
INTERVAL_EVIDENCE_SCHEMA = "ecologyrsi-dsh.interval-evidence/1"


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _alpha(value: Any) -> float:
    result = _finite(value, "alpha")
    if not 0.0 < result < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return result


def _row_identity(row: Mapping[str, Any], index: int) -> tuple[str, int, int]:
    target = row.get("target")
    horizon = row.get("horizon_hours")
    timestamp = row.get("target_timestamp", row.get("timestamp"))
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"rows[{index}].target must be non-empty text")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"rows[{index}].horizon_hours must be positive")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise ValueError(f"rows[{index}].target_timestamp must be an integer")
    return target.strip(), horizon, timestamp


def _day_blocks(timestamps: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Return calendar-day blocks, starting a new segment across time gaps."""

    days = tuple(sorted({timestamp // 24 for timestamp in timestamps}))
    if not days:
        return ()
    blocks: list[list[int]] = [[days[0]]]
    for day in days[1:]:
        if day != blocks[-1][-1] + 1:
            blocks.append([])
        blocks[-1].append(day)
    return tuple(tuple(block) for block in blocks)


def calibrate_residual_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    point_artifact_digest: str,
    data_protocol_digest: str,
    alpha: float = 0.1,
    calibration_partition: str = "calibration_uq",
) -> dict[str, Any]:
    """Freeze normalized residual quantiles for each target × horizon cell."""

    point_artifact_digest = _sha256(point_artifact_digest, "point_artifact_digest")
    data_protocol_digest = _sha256(data_protocol_digest, "data_protocol_digest")
    alpha = _alpha(alpha)
    if calibration_partition != "calibration_uq":
        raise ValueError("uncertainty calibration partition must be calibration_uq")
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("calibration-UQ rows must not be empty")

    grouped: dict[tuple[str, int], list[tuple[int, float, float]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"rows[{index}] must be an object")
        if row.get("partition") != calibration_partition:
            raise ValueError(f"rows[{index}] is not from calibration_uq")
        target, horizon, timestamp = _row_identity(row, index)
        observed = _finite(row.get("observed"), f"rows[{index}].observed")
        predicted = _finite(row.get("predicted"), f"rows[{index}].predicted")
        scale = _finite(
            row.get("normalization_scale"),
            f"rows[{index}].normalization_scale",
        )
        if scale <= 0.0:
            raise ValueError("normalization scale must be positive")
        grouped.setdefault((target, horizon), []).append(
            (timestamp, abs(observed - predicted) / scale, scale)
        )

    cells: list[dict[str, Any]] = []
    for (target, horizon), values in sorted(grouped.items()):
        scales = {item[2] for item in values}
        if len(scales) != 1:
            raise ValueError(
                f"normalization scale drift in {target}@{horizon}h calibration"
            )
        scale = next(iter(scales))
        scores = sorted(item[1] for item in values)
        quantile_index = min(len(scores), math.ceil((len(scores) + 1) * (1 - alpha)))
        quantile = scores[quantile_index - 1]
        timestamps = sorted(item[0] for item in values)
        blocks = _day_blocks(timestamps)
        cell_body = {
            "target": target,
            "horizon_hours": horizon,
            "n": len(scores),
            "alpha": alpha,
            "quantile_index": quantile_index,
            "normalization_scale": scale,
            "normalized_residual_quantile": quantile,
            "half_width": quantile * scale,
            "timestamp_digest": digest(timestamps),
            "normalized_residual_digest": digest(scores),
            "calendar_day_block_count": sum(len(block) for block in blocks),
            "continuous_segment_count": len(blocks),
            "continuous_day_segments_digest": digest(blocks),
        }
        cells.append(cell_body)

    body = {
        "schema_version": UQ_ARTIFACT_SCHEMA,
        "policy_id": UQ_POLICY_ID,
        "point_artifact_digest": point_artifact_digest,
        "data_protocol_digest": data_protocol_digest,
        "calibration_partition": calibration_partition,
        "alpha": alpha,
        "cells": cells,
    }
    return {**body, "artifact_digest": digest(body)}


def _validated_artifact(artifact: Mapping[str, Any]) -> dict[tuple[str, int], dict]:
    if not isinstance(artifact, Mapping):
        raise TypeError("uncertainty artifact must be an object")
    artifact_digest = artifact.get("artifact_digest")
    body = {key: value for key, value in artifact.items() if key != "artifact_digest"}
    if artifact_digest != digest(body):
        raise ValueError("uncertainty artifact digest mismatch")
    if artifact.get("schema_version") != UQ_ARTIFACT_SCHEMA:
        raise ValueError("uncertainty artifact schema is unsupported")
    if artifact.get("policy_id") != UQ_POLICY_ID:
        raise ValueError("uncertainty policy is unsupported")
    _alpha(artifact.get("alpha"))
    raw_cells = artifact.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("uncertainty artifact cells are missing")
    cells: dict[tuple[str, int], dict] = {}
    for index, cell in enumerate(raw_cells):
        if not isinstance(cell, Mapping):
            raise TypeError(f"artifact cells[{index}] must be an object")
        target, horizon, _timestamp = _row_identity(
            {**cell, "target_timestamp": 0}, index
        )
        half_width = _finite(cell.get("half_width"), "half_width")
        scale = _finite(cell.get("normalization_scale"), "normalization_scale")
        if half_width < 0.0 or scale <= 0.0:
            raise ValueError("uncertainty artifact width/scale is invalid")
        key = (target, horizon)
        if key in cells:
            raise ValueError("uncertainty artifact contains duplicate cells")
        cells[key] = dict(cell)
    return cells


def apply_calibrated_intervals(
    rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Attach immutable symmetric intervals to locked point predictions."""

    cells = _validated_artifact(artifact)
    result: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise TypeError(f"rows[{index}] must be an object")
        target, horizon, _timestamp = _row_identity(source, index)
        try:
            cell = cells[(target, horizon)]
        except KeyError:
            raise ValueError(
                f"uncertainty artifact does not cover {target}@{horizon}h"
            ) from None
        predicted = _finite(source.get("predicted"), f"rows[{index}].predicted")
        scale = _finite(
            source.get("normalization_scale"),
            f"rows[{index}].normalization_scale",
        )
        if scale != float(cell["normalization_scale"]):
            raise ValueError("formal row normalization scale differs from calibration")
        half_width = float(cell["half_width"])
        result.append(
            {
                **dict(source),
                "prediction_lower": predicted - half_width,
                "prediction_upper": predicted + half_width,
                "interval_policy_id": UQ_POLICY_ID,
                "interval_artifact_digest": artifact["artifact_digest"],
            }
        )
    return result


def summarize_interval_evidence(
    rows: Sequence[Mapping[str, Any]], *, alpha: float = 0.1
) -> dict[str, Any]:
    """Aggregate PICP, normalized width, and normalized Winkler score."""

    alpha = _alpha(alpha)
    grouped: dict[tuple[str, int], list[tuple[int, bool, float, float]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"rows[{index}] must be an object")
        target, horizon, timestamp = _row_identity(row, index)
        observed = _finite(row.get("observed"), f"rows[{index}].observed")
        lower = _finite(row.get("prediction_lower"), f"rows[{index}].prediction_lower")
        upper = _finite(row.get("prediction_upper"), f"rows[{index}].prediction_upper")
        scale = _finite(
            row.get("normalization_scale"),
            f"rows[{index}].normalization_scale",
        )
        if upper < lower or scale <= 0.0:
            raise ValueError("formal interval or normalization scale is invalid")
        covered = lower <= observed <= upper
        width = (upper - lower) / scale
        penalty = (
            (2.0 / alpha) * (lower - observed) / scale
            if observed < lower
            else (2.0 / alpha) * (observed - upper) / scale
            if observed > upper
            else 0.0
        )
        grouped.setdefault((target, horizon), []).append(
            (timestamp, covered, width, width + penalty)
        )

    cells = []
    for (target, horizon), values in sorted(grouped.items()):
        n = len(values)
        timestamps = [item[0] for item in values]
        blocks = _day_blocks(timestamps)
        cells.append(
            {
                "target": target,
                "horizon_hours": horizon,
                "n": n,
                "coverage": sum(item[1] for item in values) / n,
                "normalized_interval_width": sum(item[2] for item in values) / n,
                "normalized_interval_score": sum(item[3] for item in values) / n,
                "calendar_day_block_count": sum(len(block) for block in blocks),
                "continuous_segment_count": len(blocks),
            }
        )
    body = {
        "schema_version": INTERVAL_EVIDENCE_SCHEMA,
        "policy_id": UQ_POLICY_ID,
        "alpha": alpha,
        "cells": cells,
    }
    return {**body, "evidence_digest": digest(body)}


__all__ = [
    "INTERVAL_EVIDENCE_SCHEMA",
    "UQ_ARTIFACT_SCHEMA",
    "UQ_POLICY_ID",
    "apply_calibrated_intervals",
    "calibrate_residual_policy",
    "summarize_interval_evidence",
]
