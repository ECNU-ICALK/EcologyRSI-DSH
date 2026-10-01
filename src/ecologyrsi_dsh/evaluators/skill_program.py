"""Versioned, declarative skills executed on causal inputs before planning.

The host computes observations and activates conditional guidance. A receipt
proves that computation and delivery, never that an LLM obeyed the guidance.
Programs contain data only and cannot change tools, labels, budgets or gates.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math

from ..core.models import digest

SKILL_PROGRAM_SCHEMA = "ecologyrsi-dsh.skill-program/1"
from ..evolution.mutation_specs import SKILL_POLICY_ID
MAX_SKILL_STEPS = 3
MAX_GUIDANCE_LENGTH = 360
SKILL_MODULES = {
    "observation-audit@1": {
        "input": "causal history_window",
        "output": "history length, missing/nonfinite count, latest value and recent change",
        "guidance": "Check missing observations and the recent change before departing from model evidence.",
    },
    "horizon-routing@1": {
        "input": "forecast horizon and causal history",
        "output": "short/long horizon branch and available history depth",
        "guidance": "Use recent observations for the short horizon; assess model evidence separately for longer horizons.",
    },
    "trend-check@1": {
        "input": "causal history_window",
        "output": "recent mean, signed change and direction reversals",
        "guidance": "Check whether the recent trend is sustained or reversing before adjusting the model prediction.",
    },
}
TRIGGERS = ("always", "missing_history", "long_horizon", "trend_reversal")


def skill_grammar() -> dict:
    return {
        "schema_version": SKILL_PROGRAM_SCHEMA,
        "maximum_steps": MAX_SKILL_STEPS,
        "maximum_guidance_length": MAX_GUIDANCE_LENGTH,
        "step_fields": ["module_id", "when", "guidance"],
        "triggers": list(TRIGGERS),
        "modules": deepcopy(SKILL_MODULES),
        "execution": "ordered host diagnostics, followed by conditional planner guidance",
        "bounds": "causal inputs only; no additional model/tool calls; immutable evaluation contract",
    }


def validate_skill_program(value: object) -> dict:
    if not isinstance(value, Mapping) or set(value) - {"schema_version", "steps"}:
        raise ValueError("skill_program requires only schema_version and steps")
    if value.get("schema_version", SKILL_PROGRAM_SCHEMA) != SKILL_PROGRAM_SCHEMA:
        raise ValueError("unsupported skill_program schema version")
    steps = value.get("steps")
    if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence) or not 1 <= len(steps) <= MAX_SKILL_STEPS:
        raise ValueError(f"skill_program requires 1..{MAX_SKILL_STEPS} steps")
    normalized = []
    seen = set()
    for step in steps:
        if not isinstance(step, Mapping) or set(step) != {"module_id", "when", "guidance"}:
            raise ValueError("skill step requires module_id, when and guidance")
        module_id, trigger = step["module_id"], step["when"]
        if not isinstance(module_id, str) or module_id not in SKILL_MODULES:
            raise ValueError("skill module must be registered")
        if not isinstance(trigger, str) or trigger not in TRIGGERS:
            raise ValueError("skill trigger must be registered")
        if module_id in seen:
            raise ValueError("skill modules must be unique within a program")
        seen.add(module_id)
        text = step["guidance"]
        if not isinstance(text, str) or not 1 <= len(" ".join(text.split())) <= MAX_GUIDANCE_LENGTH:
            raise ValueError("skill guidance must be bounded non-empty text")
        normalized.append({"module_id": module_id, "when": trigger, "guidance": " ".join(text.split())})
    return {"schema_version": SKILL_PROGRAM_SCHEMA, "steps": normalized}


def seed_skill_program(module_id: str = "observation-audit@1") -> dict:
    return validate_skill_program({"steps": [{"module_id": module_id, "when": "always",
                                             "guidance": SKILL_MODULES[module_id]["guidance"]}]})


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def execute_skill_program(program: Mapping, causal_context: Mapping, horizon_hours: int) -> dict:
    """Read an explicit causal allowlist; ignore any scoring/label fields."""
    program = validate_skill_program(program)
    raw = causal_context.get("history_window", ())
    history = list(raw) if isinstance(raw, (list, tuple)) else []
    # Sort only when timestamps are complete. Missing time provenance must not
    # become a claim that an arbitrary array's final element is most recent.
    provenance = causal_context.get("causal_provenance", {})
    times = provenance.get("history_timestamps", ()) if isinstance(provenance, Mapping) else ()
    comparable_times = (all(isinstance(t, str) for t in times) or all(_finite(t) for t in times)) if isinstance(times, (list, tuple)) else False
    ordered = bool(history and isinstance(times, (list, tuple)) and len(times) == len(history)
                   and comparable_times and len(set(times)) == len(times))
    if ordered:
        history = [history[i] for i in sorted(range(len(times)), key=lambda i: times[i])]
    missing = sum(not _finite(value) for value in history)
    # Do not bridge gaps when claiming a trend.
    tail = []
    if ordered:
        for value in reversed(history[-6:]):
            if not _finite(value):
                break
            tail.append(float(value))
        tail.reverse()
    changes = [b - a for a, b in zip(tail, tail[1:])]
    reversals = sum(a * b < 0 for a, b in zip(changes, changes[1:]))
    long_horizon = type(horizon_hours) is int and horizon_hours >= 6
    conditions = {"always": True, "missing_history": not history or missing > 0 or not ordered,
                  "long_horizon": long_horizon, "trend_reversal": reversals > 0}
    outputs = {
        "observation-audit@1": {"history_count": len(history), "missing_count": missing,
            "timestamp_order_verified": ordered, "latest_value": tail[-1] if tail else None,
            "recent_change": changes[-1] if changes else None},
        "horizon-routing@1": {"horizon_hours": horizon_hours, "branch": "long" if long_horizon else "short",
                              "history_count": len(history)},
        "trend-check@1": {"recent_count": len(tail), "recent_mean": sum(tail) / len(tail) if tail else None,
                          "signed_change": tail[-1] - tail[0] if len(tail) > 1 else None,
                          "direction_reversals": reversals},
    }
    receipts = []
    for step in program["steps"]:
        active = conditions[step["when"]]
        receipts.append({"module_id": step["module_id"], "when": step["when"], "triggered": active,
                         "status": "executed" if active else "skipped",
                         "output": outputs[step["module_id"]] if active else {},
                         "guidance": step["guidance"] if active else ""})
    return {"schema_version": "ecologyrsi-dsh.skill-receipt/1", "program_digest": digest(program),
            "execution_owner": "host_causal_skill", "planner_compliance": "not_measured",
            "steps": receipts, "receipt_digest": digest(receipts)}


def skill_preflight_programs() -> tuple[dict, ...]:
    return tuple(seed_skill_program(module) for module in SKILL_MODULES) + (
        validate_skill_program({"steps": [seed_skill_program(module)["steps"][0] for module in SKILL_MODULES]}),
    )
