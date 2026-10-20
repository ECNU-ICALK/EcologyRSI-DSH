"""Candidate-authored bounded numerical programs. No eval, exec or imports.

Each program invokes registered capabilities, then interprets one finite scalar
expression per cell. Names bind only to actual tool outputs and the horizon.
The enclosing tool binding reserves all calls plus the calculation beforehand.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
from copy import deepcopy
import math
import operator
import re

from ..core.models import digest
from ..core.immutable import thaw_json
from ..evolution.joint_policy import PROGRAM_SCHEMA, PROGRAM_POLICY

TOOL_IDS = ("candidate-model", "persistence", "causal-trend@1", "causal-mean@1",
            "causal-seasonal@1", "greenhouse-exogenous-ridge@1", "greenhouse-baseline-aligned-ridge@1",
            "greenhouse-targetwise-ridge@1", "greenhouse-horizon-targetwise-ridge@1", "greenhouse-recipe-ridge@1")
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
_FUNCTIONS = {"min": min, "max": max, "abs": abs}


def _expression(text, names):
    if not isinstance(text, str) or not 1 <= len(text) <= 512:
        raise ValueError("strategy expression must contain 1..512 characters")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("invalid strategy expression") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > 96:
        raise ValueError("strategy expression exceeds operation budget")
    for node in nodes:
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float) or abs(node.value) > 1e6 or not math.isfinite(node.value):
                raise ValueError("strategy constants must be bounded finite numbers")
        elif isinstance(node, ast.Name):
            if node.id not in names | {"horizon"} | _FUNCTIONS.keys():
                raise ValueError("strategy expression names must bind tool outputs or horizon")
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY:
                raise ValueError("unsupported strategy arithmetic")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise ValueError("unsupported strategy unary operation")
        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1 or type(node.ops[0]) not in _COMPARE:
                raise ValueError("unsupported strategy comparison")
        elif isinstance(node, ast.Call):
            if (not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS or node.keywords
                    or not 1 <= len(node.args) <= 4 or (node.func.id == "abs" and len(node.args) != 1)
                    or (node.func.id in {"min", "max"} and len(node.args) < 2)):
                raise ValueError("unsupported strategy function call")
        elif not isinstance(node, (ast.Expression, ast.Load, ast.IfExp, ast.Add, ast.Sub, ast.Mult, ast.Div,
                                   ast.UAdd, ast.USub, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            raise ValueError("strategy expression contains unsupported syntax")
    return tree.body


def _calculate(node, values, used=None):
    if used is None:
        used = set()
    if isinstance(node, ast.Constant):
        value = node.value
    elif isinstance(node, ast.Name):
        if node.id not in values:
            raise ValueError("strategy name is not a numeric input")
        used.add(node.id)
        value = values[node.id]
    elif isinstance(node, ast.BinOp):
        value = _BINARY[type(node.op)](_calculate(node.left, values, used), _calculate(node.right, values, used))
    elif isinstance(node, ast.UnaryOp):
        value = _calculate(node.operand, values, used) * (-1 if isinstance(node.op, ast.USub) else 1)
    elif isinstance(node, ast.Compare):
        return _COMPARE[type(node.ops[0])](_calculate(node.left, values, used), _calculate(node.comparators[0], values, used))
    elif isinstance(node, ast.IfExp):
        value = _calculate(node.body if _calculate(node.test, values, used) else node.orelse, values, used)
    else:
        value = _FUNCTIONS[node.func.id](*[_calculate(arg, values, used) for arg in node.args])
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e12:
        raise ValueError("strategy calculation must produce bounded finite numbers")
    return float(value)


def validate_strategy_program(value):
    fields = {"schema_version", "planner_prompt", "research_prompt", "proposal_prompt", "memory", "calls", "expression"}
    if not isinstance(value, Mapping) or set(value) != fields or value["schema_version"] != PROGRAM_SCHEMA:
        raise ValueError("invalid strategy program fields/schema")
    # Durable revisions contain nested read-only mappings and tuples. Validate
    # and compile detached JSON data just as for the original model response.
    value = thaw_json(value)
    for field in ("planner_prompt", "research_prompt", "proposal_prompt", "memory"):
        if not isinstance(value[field], str) or len(value[field]) > 1200:
            raise ValueError("strategy guidance must be text of at most 1200 characters")
    calls = value["calls"]
    if not isinstance(calls, (list, tuple)) or not 1 <= len(calls) <= 3:
        raise ValueError("strategy requires 1..3 registered calls plus one calculation")
    names = set()
    for call in calls:
        if not isinstance(call, Mapping) or set(call) != {"name", "tool_id", "parameters"}:
            raise ValueError("strategy call requires name, tool_id, parameters")
        name = call["name"]
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,23}", name)
                or name in names | {"horizon"} | _FUNCTIONS.keys()):
            raise ValueError("invalid or duplicate strategy variable")
        names.add(name)
        if call["tool_id"] not in TOOL_IDS or not isinstance(call["parameters"], Mapping):
            raise ValueError("strategy call is outside the numerical tool registry")
        if call["tool_id"] in {"candidate-model", "persistence"} and call["parameters"]:
            raise ValueError("candidate-model and persistence take no parameters")
        if call["tool_id"].startswith("causal-"):
            _causal_parameters(call["tool_id"], call["parameters"])
        else:
            from .agent_model_tools import _CONFIGS, _defaults
            if call["tool_id"] in _CONFIGS:
                config = _CONFIGS[call["tool_id"]]
                config.from_mapping({**_defaults(config), **dict(call["parameters"])})
        # Parameters are data; the concrete registered capability validates its
        # own schema before execution. Bound payload size before compilation.
        from ..core.models import canonical_json
        if len(canonical_json(call["parameters"])) > 6000:
            raise ValueError("strategy parameters exceed size budget")
    _expression(value["expression"], names)
    return value


def strategy_program_grammar():
    return {"schema_version": PROGRAM_SCHEMA, "policy": PROGRAM_POLICY,
            "maximum_calls": 3, "calculation_call_cost": 1, "maximum_guidance_characters": 1200,
            "tool_ids": list(TOOL_IDS), "fields": list(seed_strategy_program()),
            "expression": "finite arithmetic + - * /, min/max/abs, horizon comparisons and conditional expressions",
            "example": seed_strategy_program(),
            "bounds": "No labels, imports, filesystem, network, loops or mutable state; nested calls consume the frozen budget."}


def seed_strategy_program():
    return {"schema_version": PROGRAM_SCHEMA,
            "planner_prompt": "Assess the computed strategy against causal observations; cite actual evidence.",
            "research_prompt": "Investigate errors by target and horizon using training evidence.",
            "proposal_prompt": "Propose a bounded related patch and state which executed mechanism it changes.",
            "memory": "No measured gain yet.",
            "calls": [{"name": "recent", "tool_id": "persistence", "parameters": {}},
                      {"name": "trend", "tool_id": "causal-trend@1", "parameters": {"window": 6, "damping": 0.1}}],
            "expression": "trend if horizon <= 6 else recent"}


def execute_strategy_program(program, requests, executor):
    program = validate_strategy_program(program)
    tree = _expression(program["expression"], {c["name"] for c in program["calls"]})
    results, receipts = {}, []
    for call in program["calls"]:
        output = executor(call["tool_id"], dict(call["parameters"]))
        if set(output) != {r.sample_id for r in requests}:
            raise ValueError("strategy tool returned incomplete evidence")
        for row in output.values():
            predicted = row.get("predicted") if isinstance(row, Mapping) else None
            if type(predicted) not in (int, float) or not math.isfinite(predicted):
                raise ValueError("strategy tool returned nonfinite evidence")
        results[call["name"]] = output
        receipts.append({"name": call["name"], "tool_id": call["tool_id"],
                         "parameters": call["parameters"], "output_digest": digest(output)})
    outputs = {}
    for request in requests:
        values = {name: rows[request.sample_id]["predicted"] for name, rows in results.items()}
        used = set()
        value = _calculate(tree, {**values, "horizon": request.horizon_hours}, used)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("strategy must return a finite numeric prediction")
        cell_receipts = [{**receipt, "output": deepcopy(results[receipt["name"]][request.sample_id]),
                          "used_in_expression": receipt["name"] in used} for receipt in receipts]
        outputs[request.sample_id] = {"predicted": value, "metadata": {"program_digest": digest(program),
            "parameters": {"program_digest": digest(program)}, "internal_calls": cell_receipts,
            "calculation_digest": digest({"calls": program["calls"], "expression": program["expression"]}),
            "expression": program["expression"], "execution_owner": "bounded_strategy_interpreter"}}
    return outputs


def causal_tool_catalog():
    return [{"tool_id": tool, "version": "1", "purpose": purpose, "parameters": parameters}
        for tool, purpose, parameters in [
            ("causal-trend@1", "Damped trend from timestamped history only",
             {"window": {"default": 6, "minimum": 2, "maximum": 12, "type": "integer"},
              "damping": {"default": 0.1, "minimum": 0, "maximum": 1, "type": "number"}}),
            ("causal-mean@1", "Rolling mean from timestamped history only",
             {"window": {"default": 6, "minimum": 2, "maximum": 12, "type": "integer"}}),
            ("causal-seasonal@1", "Repeat an observed causal seasonal phase",
             {"period": {"default": 24, "minimum": 1, "maximum": 24, "type": "integer"}})]]


def _causal_parameters(tool_id, parameters):
    specs = next((r["parameters"] for r in causal_tool_catalog() if r["tool_id"] == tool_id), None)
    if specs is None or set(parameters) - set(specs):
        raise ValueError("unknown causal tool or parameters")
    params = {key: parameters.get(key, spec["default"]) for key, spec in specs.items()}
    for key, val in params.items():
        spec = specs[key]
        if (type(val) not in (int, float) or not math.isfinite(val) or not spec["minimum"] <= val <= spec["maximum"]
                or (spec["type"] == "integer" and type(val) is not int)):
            raise ValueError("causal tool parameter outside bounds")
    return params


def execute_causal_tool(requests, tool_id, parameters):
    params = _causal_parameters(tool_id, parameters)
    output = {}
    for request in requests:
        context = request.label_free_context
        history = context.get("history_window", ())
        times = context.get("causal_provenance", {}).get("history_timestamps", ())
        if (not history or len(times) != len(history) or len(set(times)) != len(times)
                or any(type(t) not in (int, float) or not math.isfinite(t) or t > request.origin_timestamp for t in times)):
            raise ValueError("causal tool requires unique numeric timestamps no later than origin")
        points = sorted(zip(times, history))
        if points[-1][0] != request.origin_timestamp:
            raise ValueError("causal tool requires an observation at origin")
        if tool_id == "causal-seasonal@1":
            period = params["period"]
            offset = request.horizon_hours % period
            stamp = request.origin_timestamp + offset - (period if offset else 0)
            selected = [v for t, v in points if t == stamp]
            if not selected:
                raise ValueError("seasonal phase is outside available causal history")
            value = selected[0]
        else:
            if len(points) < params["window"]:
                raise ValueError("causal tool history shorter than window")
            tail = points[-params["window"]:]
            if any(type(v) not in (int, float) or not math.isfinite(v) for _, v in tail):
                raise ValueError("causal tool history contains missing values")
            if tool_id == "causal-mean@1":
                value = math.fsum(v for _, v in tail) / len(tail)
            else:
                tmean = math.fsum(t for t, _ in tail) / len(tail)
                vmean = math.fsum(v for _, v in tail) / len(tail)
                slope = math.fsum((t-tmean)*(v-vmean) for t, v in tail) / math.fsum((t-tmean)**2 for t, _ in tail)
                value = tail[-1][1] + slope * request.horizon_hours * params["damping"]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("causal tool output must be finite")
        output[request.sample_id] = {"predicted": float(value), "metadata": {"parameters": params, "source": tool_id}}
    return output
