"""Deterministic parsing and enforcement of bounded human interventions."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from ..core.models import TaskManifest
from .strategies import (
    _bounded_parameters,
    _task_parameter_boundary,
)


_PARAMETER_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "toy": {
        "alpha": ("alpha", "平滑权重", "平滑系数"),
        "window": ("window", "时间窗口", "历史窗口"),
        "water_threshold": (
            "water_threshold",
            "water threshold",
            "土壤水分阈值",
            "水分阈值",
        ),
    },
    "greenhouse": {
        "blend": ("blend", "混合权重", "融合权重"),
        "window": ("window", "时间窗口", "历史窗口"),
        "bias_scale": (
            "bias_scale",
            "bias scale",
            "偏差缩放系数",
            "偏差缩放",
        ),
    },
    "greenhouse_ridge": {
        "history_steps": (
            "history_steps",
            "history steps",
            "历史步数",
            "滞后步数",
        ),
        "ridge_alpha": (
            "ridge_alpha",
            "ridge alpha",
            "岭回归强度",
            "正则化强度",
        ),
        "residual_scale": (
            "residual_scale",
            "residual scale",
            "残差缩放系数",
            "残差缩放",
        ),
    },
    "greenhouse_targetwise_ridge": {
        "history_steps": (
            "history_steps",
            "history steps",
            "历史步数",
            "滞后步数",
        ),
        "ridge_alpha": (
            "ridge_alpha",
            "ridge alpha",
            "岭回归强度",
            "正则化强度",
        ),
        "air_temperature_residual_scale": (
            "air_temperature_residual_scale",
            "temperature residual scale",
            "温度残差缩放",
        ),
        "relative_humidity_residual_scale": (
            "relative_humidity_residual_scale",
            "humidity residual scale",
            "湿度残差缩放",
        ),
        "co2_concentration_residual_scale": (
            "co2_concentration_residual_scale",
            "co2 residual scale",
            "二氧化碳残差缩放",
        ),
    },
}
_GUIDANCE_STEPS: dict[str, int | float] = {
    "alpha": 0.1,
    "blend": 0.1,
    "bias_scale": 0.1,
    "water_threshold": 0.05,
    "window": 1,
    "history_steps": 1,
    "ridge_alpha": 0.05,
    "residual_scale": 0.1,
    "air_temperature_residual_scale": 0.1,
    "relative_humidity_residual_scale": 0.1,
    "co2_concentration_residual_scale": 0.1,
}
_GUIDANCE_DIRECTIONS: dict[str, tuple[str, ...]] = {
    "decrease": ("缩短", "降低", "减小", "下调", "减少", "decrease", "shorten", "lower"),
    "increase": ("延长", "提高", "增大", "上调", "增加", "increase", "extend", "raise"),
}
_NEGATED_GUIDANCE = re.compile(
    r"(?:不要|不得|禁止|不应|无需)(?:[^，。；,;]{0,12})$", re.IGNORECASE
)
_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


def _matching_parameters(message: str, domain: str) -> list[str]:
    normalized = message.casefold()
    result: list[str] = []
    for parameter, aliases in _PARAMETER_ALIASES[domain].items():
        for alias in aliases:
            token = alias.casefold()
            if token.isascii():
                pattern = rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])"
                if re.search(pattern, normalized):
                    result.append(parameter)
                    break
            elif token in normalized:
                result.append(parameter)
                break
    return result


def _operation_is_negated(message: str, start: int) -> bool:
    prefix = message.casefold()[max(0, start - 24) : start]
    return _NEGATED_GUIDANCE.search(prefix) is not None


def _parse_guidance(message: str, domain: str) -> tuple[str, str] | str:
    parameters = _matching_parameters(message, domain)
    if len(parameters) != 1:
        return "未唯一识别一个允许调整的参数"
    normalized = message.casefold()
    directions: list[str] = []
    for direction, words in _GUIDANCE_DIRECTIONS.items():
        for word in words:
            match = re.search(re.escape(word.casefold()), normalized)
            if match is None:
                continue
            if _operation_is_negated(normalized, match.start()):
                return "调整方向含否定表达，未自动执行"
            directions.append(direction)
            break
    if len(directions) != 1:
        return "未唯一识别提高或降低方向"
    return parameters[0], directions[0]


def _parse_constraint(
    message: str,
    domain: str,
) -> tuple[str, str, float] | str:
    parameters = _matching_parameters(message, domain)
    if len(parameters) != 1:
        return "未唯一识别一个允许约束的参数"
    parameter = parameters[0]
    aliases = sorted(
        _PARAMETER_ALIASES[domain][parameter], key=len, reverse=True
    )
    alias_pattern = "(?:" + "|".join(re.escape(item) for item in aliases) + ")"
    patterns = (
        (rf"{alias_pattern}\s*(?:<=|≤)\s*({_NUMBER_PATTERN})", "<="),
        (rf"{alias_pattern}\s*(?:>=|≥)\s*({_NUMBER_PATTERN})", ">="),
        (
            rf"{alias_pattern}\s*(?:保持|控制)?\s*(?:在)?\s*({_NUMBER_PATTERN})\s*(?:以下|以内)",
            "<=",
        ),
        (
            rf"{alias_pattern}\s*(?:保持|控制)?\s*(?:在)?\s*({_NUMBER_PATTERN})\s*(?:以上)",
            ">=",
        ),
        (rf"{alias_pattern}\s*(?:不超过|至多|最多)\s*({_NUMBER_PATTERN})", "<="),
        (rf"{alias_pattern}\s*(?:不低于|不少于|至少)\s*({_NUMBER_PATTERN})", ">="),
    )
    matches: list[tuple[str, float]] = []
    for pattern, operator in patterns:
        for match in re.finditer(pattern, message, flags=re.IGNORECASE):
            if _operation_is_negated(message, match.start()):
                return "数值约束含否定表达，未自动执行"
            number = float(match.group(1))
            if math.isfinite(number):
                matches.append((operator, number))
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        return "未唯一识别 <= 或 >= 数值边界"
    operator, bound = unique[0]
    return parameter, operator, bound


def _base_receipt(control: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "intervention_id": str(control.get("intervention_id", "unknown")),
        "kind": str(control.get("kind", "unknown")),
        "recorded": True,
        "applied": False,
        "enforced": False,
        "application_status": "recorded",
        "reason": "意见已记录，但未执行",
    }


def _set_receipt_status(
    receipt: dict[str, Any],
    status: str,
    *,
    reason: str,
    **details: Any,
) -> None:
    receipt.update(
        {
            "applied": status in {"applied", "enforced"},
            "enforced": status == "enforced",
            "application_status": status,
            "reason": reason,
            **details,
        }
    )


def apply_bounded_interventions(
    task: TaskManifest,
    parameters: Mapping[str, Any],
    interventions: Sequence[Mapping[str, Any]],
    *,
    selected_parent_candidate_id: str | None = None,
) -> tuple[dict[str, int | float], list[dict[str, Any]]]:
    """Apply deterministic human controls inside the task parameter boundary.

    Guidance is advisory and uses one fixed step. Parameter overrides and
    parseable constraints are host-enforced, with constraints applied last.
    Every input receives a receipt; ambiguous free text is consumed but remains
    explicitly recorded-only.
    """

    if not isinstance(task, TaskManifest):
        raise TypeError("task must be a TaskManifest")
    if len(interventions) > 64:
        raise ValueError("too many interventions")
    domain, schemas = _task_parameter_boundary(task)
    result = _bounded_parameters(
        parameters,
        schemas,
        partial=False,
        source="proposal",
    )
    receipts = [_base_receipt(item) for item in interventions]

    for index, control in enumerate(interventions):
        kind = str(control.get("kind", ""))
        if kind != "guidance":
            continue
        message = str(control.get("message", ""))
        parsed = _parse_guidance(message, domain)
        if isinstance(parsed, str):
            receipts[index]["reason"] = parsed
            continue
        parameter, direction = parsed
        schema = schemas[parameter]
        previous = result[parameter]
        signed_step = float(_GUIDANCE_STEPS[parameter]) * (
            -1 if direction == "decrease" else 1
        )
        candidate = min(
            float(schema["maximum"]),
            max(float(schema["minimum"]), float(previous) + signed_step),
        )
        value: int | float = (
            int(round(candidate))
            if schema["type"] == "integer"
            else round(candidate, 6)
        )
        result[parameter] = value
        _set_receipt_status(
            receipts[index],
            "applied",
            reason="已按固定步长应用人工调整指引",
            parameter=parameter,
            direction=direction,
            step=abs(_GUIDANCE_STEPS[parameter]),
            previous_value=previous,
            result_value=value,
        )

    override_requests: list[tuple[int, dict[str, int | float]]] = []
    for index, control in enumerate(interventions):
        if str(control.get("kind", "")) != "parameter_override":
            continue
        try:
            override = _bounded_parameters(
                control.get("parameter_overrides", {}),
                schemas,
                partial=True,
                source="parameter_override",
            )
        except (TypeError, ValueError) as exc:
            receipts[index]["reason"] = f"参数覆盖未执行：{exc}"
            continue
        if not override:
            receipts[index]["reason"] = "参数覆盖为空，未执行"
            continue
        previous = {name: result[name] for name in override}
        result.update(override)
        override_requests.append((index, override))
        _set_receipt_status(
            receipts[index],
            "enforced",
            reason="参数覆盖已通过宿主范围校验并强制执行",
            parameters=sorted(override),
            previous_values=previous,
            result_values=dict(override),
        )

    parsed_constraints: dict[str, list[tuple[int, str, float, float]]] = {}
    for index, control in enumerate(interventions):
        if str(control.get("kind", "")) != "constraint":
            continue
        message = str(control.get("message", ""))
        parsed = _parse_constraint(message, domain)
        if isinstance(parsed, str):
            receipts[index]["reason"] = parsed
            continue
        parameter, operator, bound = parsed
        schema = schemas[parameter]
        effective_bound = bound
        if schema["type"] == "integer":
            effective_bound = (
                float(math.floor(bound))
                if operator == "<="
                else float(math.ceil(bound))
            )
        impossible = (
            operator == "<=" and effective_bound < float(schema["minimum"])
        ) or (
            operator == ">=" and effective_bound > float(schema["maximum"])
        )
        if impossible:
            receipts[index].update(
                {
                    "parameter": parameter,
                    "operator": operator,
                    "bound": bound,
                    "reason": "约束与宿主允许范围冲突，未执行",
                }
            )
            continue
        parsed_constraints.setdefault(parameter, []).append(
            (index, operator, bound, effective_bound)
        )

    for parameter, constraints in parsed_constraints.items():
        schema = schemas[parameter]
        lower = float(schema["minimum"])
        upper = float(schema["maximum"])
        for _, operator, _, effective_bound in constraints:
            if operator == ">=":
                lower = max(lower, effective_bound)
            else:
                upper = min(upper, effective_bound)
        if lower > upper:
            for index, operator, bound, _ in constraints:
                receipts[index].update(
                    {
                        "parameter": parameter,
                        "operator": operator,
                        "bound": bound,
                        "reason": "同一参数的人工约束相互冲突，均未执行",
                    }
                )
            continue
        previous = result[parameter]
        candidate = min(upper, max(lower, float(previous)))
        value = (
            int(round(candidate))
            if schema["type"] == "integer"
            else round(candidate, 6)
        )
        result[parameter] = value
        for index, operator, bound, _ in constraints:
            _set_receipt_status(
                receipts[index],
                "enforced",
                reason="参数约束已由宿主在最终提案边界强制执行",
                parameter=parameter,
                operator=operator,
                bound=bound,
                previous_value=previous,
                result_value=value,
            )

    for index, requested in override_requests:
        final_values = {name: result[name] for name in requested}
        receipts[index]["result_values"] = final_values
        if any(final_values[name] != value for name, value in requested.items()):
            _set_receipt_status(
                receipts[index],
                "applied",
                reason="参数覆盖已应用，但最终值受人工硬约束收敛",
                parameters=sorted(requested),
                result_values=final_values,
            )

    for index, control in enumerate(interventions):
        if str(control.get("kind", "")) != "parent_selection":
            continue
        target = control.get("target_candidate_id")
        if target is not None and str(target) == selected_parent_candidate_id:
            _set_receipt_status(
                receipts[index],
                "enforced",
                reason="人工选择的父候选已用于本轮提案",
                target_candidate_id=str(target),
            )
        else:
            receipts[index].update(
                {
                    "target_candidate_id": target,
                    "reason": "人工父候选与本轮实际父候选不一致，未执行",
                }
            )

    result = _bounded_parameters(
        result,
        schemas,
        partial=False,
        source="proposal after interventions",
    )
    return result, receipts
