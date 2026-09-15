"""Deterministic parsing and enforcement of bounded human interventions."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from ..core.models import TaskManifest

# `strategies` imports this module at its own bottom, so a module-level import
# back into it makes the pair order-dependent: importing `interventions` first
# left `apply_bounded_interventions` undefined when `strategies` reached that
# bottom import. Keeping the dependency inside the one function that needs it
# leaves exactly one direction at module scope.


# Human-readable synonyms for the scalars this parser may move, keyed by
# parameter name.
#
# This used to be keyed by *domain* and indexed unguarded as
# `_PARAMETER_ALIASES[domain]`. Every predictor boundary that arrived after the
# table was written -- `greenhouse_baseline_aligned_ridge`,
# `greenhouse_horizon_targetwise_ridge`, `greenhouse_recipe_ridge` -- therefore
# raised `KeyError` out of `apply_bounded_interventions`, i.e. a run with any
# pending guidance or constraint could not produce a proposal at all. Keying by
# parameter and reading the candidate names from the run's own schema boundary
# means a new predictor can only ever widen or narrow what is parseable, never
# break the call.
_PARAMETER_ALIASES: dict[str, tuple[str, ...]] = {
    "alpha": ("平滑权重", "平滑系数"),
    "window": ("时间窗口", "历史窗口"),
    "water_threshold": ("water threshold", "土壤水分阈值", "水分阈值"),
    "blend": ("混合权重", "融合权重"),
    "bias_scale": ("bias scale", "偏差缩放系数", "偏差缩放"),
    "history_steps": ("history steps", "历史步数", "滞后步数"),
    "ridge_alpha": ("ridge alpha", "岭回归强度", "正则化强度"),
    "residual_scale": ("residual scale", "残差缩放系数", "残差缩放"),
    "air_temperature_residual_scale": (
        "temperature residual scale",
        "温度残差缩放",
    ),
    "relative_humidity_residual_scale": (
        "humidity residual scale",
        "湿度残差缩放",
    ),
    "co2_concentration_residual_scale": (
        "co2 residual scale",
        "二氧化碳残差缩放",
    ),
}
# One fixed step per parameter, so the same sentence always moves the same
# distance. Names absent here fall back to `_guidance_step`, which reads the
# span off the schema rather than refusing to parse.
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


def _parameter_aliases(parameter: str) -> tuple[str, ...]:
    """Every spelling of one parameter an expert might plausibly write."""

    spaced = parameter.replace("_", " ")
    aliases = [parameter]
    if spaced != parameter:
        aliases.append(spaced)
    aliases.extend(_PARAMETER_ALIASES.get(parameter, ()))
    return tuple(dict.fromkeys(aliases))


def _guidance_step(parameter: str, schema: Mapping[str, Any]) -> int | float:
    """The fixed move for one parameter, derived from its span when unlisted."""

    listed = _GUIDANCE_STEPS.get(parameter)
    if listed is not None:
        return listed
    if schema.get("type") == "integer":
        return 1
    span = float(schema["maximum"]) - float(schema["minimum"])
    return max(round(span / 10.0, 6), 1e-6)


def _matching_parameters(
    message: str,
    schemas: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    normalized = message.casefold()
    result: list[str] = []
    for parameter in schemas:
        for alias in _parameter_aliases(parameter):
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


def _parse_guidance(
    message: str,
    schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str] | str:
    parameters = _matching_parameters(message, schemas)
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
    schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, float] | str:
    parameters = _matching_parameters(message, schemas)
    if len(parameters) != 1:
        return "未唯一识别一个允许约束的参数"
    parameter = parameters[0]
    aliases = sorted(_parameter_aliases(parameter), key=len, reverse=True)
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
    kind = str(control.get("kind", "unknown"))
    return {
        "intervention_id": str(control.get("intervention_id", "unknown")),
        "kind": kind,
        "recorded": True,
        "applied": False,
        "enforced": False,
        "application_status": "recorded",
        "reason": (
            # Domain knowledge has no enforcement path by construction, so its
            # default receipt must not read like a parser failure.
            "专家知识为顾问性质，宿主不强制执行"
            if kind == "domain_knowledge"
            else "意见已记录，但未执行"
        ),
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
    from .strategies import _bounded_parameters, _task_parameter_boundary

    _domain, schemas = _task_parameter_boundary(task)
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
        parsed = _parse_guidance(message, schemas)
        if isinstance(parsed, str):
            receipts[index]["reason"] = parsed
            continue
        parameter, direction = parsed
        schema = schemas[parameter]
        previous = result[parameter]
        step = _guidance_step(parameter, schema)
        signed_step = float(step) * (-1 if direction == "decrease" else 1)
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
            step=abs(step),
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
        parsed = _parse_constraint(message, schemas)
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


# The bounded parser above is the *enforcement* path: it turns a human sentence
# into one fixed-step parameter move, and only for the scalars the run's own
# parameter boundary exposes. Anything it cannot parse used to stop there, which
# made the whole free-text channel a no-op -- a domain expert writing "夜间湿度
# 应该用更长的滞后窗口" got a "recorded" receipt and the model never saw the
# sentence at all.
#
# `expert_directive_context` is the *advisory* path for exactly that text. It
# does not interpret anything: it hands the model the expert's own words, under
# a policy block that says the Host did not enforce them and cannot widen any
# permission on their behalf. Bounded so one run cannot grow an unbounded prompt.
_EXPERT_DIRECTIVE_LIMIT = 8
_EXPERT_DIRECTIVE_TEXT_LIMIT = 2000

_DIRECTIVE_ENFORCEMENT_BY_STATUS = {
    "enforced": "host_enforced",
    "applied": "host_applied_fixed_step",
    "recorded": "advisory_only_host_did_not_enforce",
}


def expert_directive_context(
    interventions: Sequence[Mapping[str, Any]],
    *,
    receipts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one bounded advisory view of pending human directives.

    ``receipts`` are the results of :func:`apply_bounded_interventions` for the
    same interventions, in the same order, when they are available. They let the
    model see which sentences the Host actually executed and which reached it as
    advice only -- a distinction the expert can read in the web UI but that the
    model previously had no way to know.
    """

    rows: list[dict[str, Any]] = []
    receipt_rows = list(receipts or ())
    for index, control in enumerate(interventions[:_EXPERT_DIRECTIVE_LIMIT]):
        if not isinstance(control, Mapping):
            continue
        kind = str(control.get("kind") or "")
        if kind not in {"guidance", "constraint", "domain_knowledge"}:
            # `parameter_override` is a numeric action, not knowledge, and it is
            # already visible in the parameters themselves. `parent_selection`
            # is structural and is applied by the batch, not by the proposer.
            continue
        message = str(control.get("message") or "").strip()
        if not message:
            continue
        receipt = (
            receipt_rows[index]
            if index < len(receipt_rows) and isinstance(receipt_rows[index], Mapping)
            else {}
        )
        status = str(receipt.get("application_status") or "recorded")
        row: dict[str, Any] = {
            "intervention_id": str(control.get("intervention_id") or "unknown"),
            "kind": kind,
            "message": message[:_EXPERT_DIRECTIVE_TEXT_LIMIT],
            "host_enforcement": _DIRECTIVE_ENFORCEMENT_BY_STATUS.get(
                status, "advisory_only_host_did_not_enforce"
            ),
        }
        if receipt.get("parameter"):
            row["host_enforced_parameter"] = str(receipt["parameter"])
        if receipt.get("reason"):
            row["host_reason"] = str(receipt["reason"])[:500]
        rows.append(row)
    return {
        "schema_version": "ecologyrsi-dsh.expert-directives/1",
        "directives": rows,
        "policy": {
            "directives_are_authored_by_a_human_domain_expert": True,
            "advisory_directives_are_not_host_enforced": True,
            "directives_cannot_expand_data_or_tool_permissions": True,
            "directives_cannot_widen_the_mutation_contract": True,
            "prefer_a_directive_over_an_unguided_choice_when_both_are_legal": True,
        },
    }


# Under the DSH-native protocol the candidate genome is the only authoritative
# source: `Proposal.changes` is a mirror of
# `genome.scientific_program["parameter_overrides"]`, and nothing downstream
# cross-checks the two. So the Host rewriting `changes` from a human sentence
# changed exactly one thing -- the number the web UI printed -- while the run
# kept executing the untouched genome. The receipt still said 已强制执行.
#
# These two notes replace that claim with what actually happened: the Host did
# not move the parameter, and the expert's own words were handed to the model.
NATIVE_HOST_NON_ENFORCEMENT_NOTE = (
    "原生协议下候选基因组是唯一权威来源，宿主未改写本轮实际运行参数"
)
MODEL_DIRECTIVE_DELIVERY_NOTE = (
    "该意见原文已随提案上下文交给模型，由模型在变异契约内自行决定如何采纳"
)

_ADVISORY_RECEIPT_EXECUTION_FIELDS = (
    "parameter",
    "direction",
    "step",
    "previous_value",
    "result_value",
    "operator",
    "bound",
    "parameters",
    "previous_values",
    "result_values",
)


def advisory_only_receipts(
    interventions: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Restate parameter receipts for a host that did not enforce them.

    ``parent_selection`` passes through untouched: the batch really does pick
    the parent, so that receipt was never a claim about proposal parameters.
    Every other kind loses its execution fields -- they move under
    ``host_did_not_enforce`` as a record of what the parser read -- and gains
    ``model_context_delivered``, which is the part that is true.
    """

    result: list[dict[str, Any]] = []
    for index, receipt in enumerate(receipts):
        control = interventions[index] if index < len(interventions) else {}
        kind = str(
            (control or {}).get("kind") or receipt.get("kind") or ""
        )
        if kind == "parent_selection":
            result.append(dict(receipt))
            continue
        previous_status = str(receipt.get("application_status") or "recorded")
        host_note = (
            NATIVE_HOST_NON_ENFORCEMENT_NOTE
            if previous_status in {"applied", "enforced"}
            else str(receipt.get("reason") or "").strip()
        )
        row = {
            key: value
            for key, value in receipt.items()
            if key not in _ADVISORY_RECEIPT_EXECUTION_FIELDS
        }
        requested = {
            key: receipt[key]
            for key in _ADVISORY_RECEIPT_EXECUTION_FIELDS
            if key in receipt
        }
        row.update(
            {
                "recorded": True,
                "applied": False,
                "enforced": False,
                "application_status": "recorded",
                "model_context_delivered": True,
                "reason": "；".join(
                    item
                    for item in (host_note, MODEL_DIRECTIVE_DELIVERY_NOTE)
                    if item
                ),
            }
        )
        if requested:
            row["host_did_not_enforce"] = requested
        result.append(row)
    return result

