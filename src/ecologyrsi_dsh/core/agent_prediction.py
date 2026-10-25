"""Agent-owned numerical predictions and auditable, bounded evidence references."""

from collections.abc import Mapping, Sequence
import math

from .models import digest
from .redaction import REMOTE_REASON_CODES

AGENT_PREDICTION_SCHEMA = "ecology-sample-predictions@2"
MAX_PREDICTION_CALLS = 6  # Maximum receipt size; admission sets the execution budget.
# The execution budget a NEW run freezes into its Agent policy. It is not read
# directly at execution time: every run carries its own value in
# agent_policy["inference"]["max_tool_calls_per_attempt"], and
# rebind_agent_policy preserves it, so raising this default cannot invalidate an
# archived artifact. Two left a planner that wants to `blend` -- which needs two
# successful results (see below) -- no slack to probe any predictor, because a
# failed call still consumes budget; observed runs spent all of it and then cited
# evidence they never received.
PREDICTION_TOOL_CALL_BUDGET = 4
# Every method the schema knows. A candidate that authored a directive narrows
# this set for itself through `blend_rule`; the default is the full set, so a
# genome that authored nothing is validated exactly as before.
PREDICTION_METHODS = frozenset({"direct", "model", "blend", "adjusted"})
EVIDENCE_FORMULA_POLICY = "evidence-mean-or-model@1"


def successful_agent_provenance_passes(sample):
    """Qualify published successes independently of missing scientific pairs."""
    if not isinstance(sample, Mapping):
        return False
    value = sample.get("successful_agent_provenance_pass")
    if value is None:
        value = sample.get("strict_agent_chain_pass")
    return value is True


class PredictionContractViolation(ValueError):
    """Known, correctable prediction errors with value-free bounded diagnostics.

    The legacy exception text remains available to internal callers. Only the
    fixed code/path projection is suitable for model feedback; never forward
    arbitrary exception strings across that boundary.
    """

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = tuple(dict(item) for item in diagnostics[:32])


def frozen_prediction_methods(allowed_methods=None, *, formula_policy=None):
    permitted = PREDICTION_METHODS if allowed_methods is None else frozenset(allowed_methods)
    if not permitted or not permitted <= PREDICTION_METHODS:
        raise ValueError("allowed prediction methods are invalid")
    if formula_policy == EVIDENCE_FORMULA_POLICY:
        permitted &= {"model", "blend"}
        if not permitted:
            raise ValueError("candidate prediction methods conflict with frozen formula policy")
    return permitted


def prediction_formula_contract(metadata):
    """Prospective, run-frozen contract shared with proposer and local editor."""
    if not isinstance(metadata, Mapping) or metadata.get("prediction_formula_policy") != EVIDENCE_FORMULA_POLICY:
        return {}
    return {
        "formula_policy": EVIDENCE_FORMULA_POLICY,
        "allowed_methods": ["model", "blend"],
        "allowed_blend_rules": ["none", "mean"],
        "model": "Copy one successful tool call's exact output for each sample; cite exactly that call.",
        "blend": "Equal weights 1/n over at least two distinct successful computations; cite exactly those calls.",
        "instruction": (
            "The formula policy is immutable for this run. Do not author confidence_weighted, direct or "
            "adjusted final predictions. Different executable calculations belong in a frozen candidate "
            "strategy program whose actual result is then copied with method=model. Tool selection and "
            "permitted mid-reasoning retrieval remain autonomous. No unsupported numeric adjustment or "
            "silent fallback is accepted. Changing a parameter and changing fusion are separate experiments."
        ),
    }


def validate_formula_profile(profile, formula_policy):
    """Reject incompatible candidate instructions before scientific execution."""
    if formula_policy != EVIDENCE_FORMULA_POLICY:
        return
    profile = profile if isinstance(profile, Mapping) else {}
    frozen_prediction_methods(profile.get("allowed_prediction_methods"), formula_policy=formula_policy)
    directive = profile.get("authored_directive")
    if isinstance(directive, Mapping) and directive.get("blend_rule") not in {"none", "mean"}:
        raise ValueError("frozen prediction formula conflicts with authored blend_rule; use none or mean")


def _canonical_computation_key(payload):
    """Optional Host proof on new numerical results; never infer it for old events."""
    outputs = payload["result"].get("outputs", ())
    proofs = [(row["metadata"].get("canonical_computation")
               if isinstance(row.get("metadata"), Mapping) else None) for row in outputs]
    if not any(proof is not None for proof in proofs):
        return None
    for proof in proofs:
        if (not isinstance(proof, Mapping) or set(proof) != {
                "schema_version", "model_id", "parameters_digest", "fit_resource_key"}
                or proof["schema_version"] != "ecologyrsi-dsh.canonical-computation/1"
                or not isinstance(proof["model_id"], str) or not 1 <= len(proof["model_id"]) <= 160
                or any(not isinstance(proof[key], str) or len(proof[key]) != 64
                       or any(c not in "0123456789abcdef" for c in proof[key])
                       for key in ("parameters_digest", "fit_resource_key"))):
            raise ValueError("invalid Host canonical computation proof")
    if any(proof != proofs[0] for proof in proofs):
        raise ValueError("inconsistent Host canonical computation proof across samples")
    return digest({"wave_digest": payload["wave_digest"],
                   "sample_ids": payload["sample_ids"], "computation": proofs[0]})


def unique_prediction_evidence(calls, *, provenance_calls=None):
    """Keep one vote per executed computation, never deduplicate equal numbers.

    Calls map IDs to (persisted payload, event ID). Reuse links cover equivalent
    default/explicit parameters; exact argument identity also protects replay.
    Distinct operators with coincident predictions remain distinct evidence.
    """
    by_event = {event_id: payload for payload, event_id in
                (calls if provenance_calls is None else provenance_calls).values()}
    seen_arguments, seen_sources, unique = set(), set(), []
    for call_id, (payload, event_id) in calls.items():
        if payload["result"]["status"] != "completed":
            continue
        arguments = payload["arguments"]
        canonical = _canonical_computation_key(payload)
        # New Host proofs bind the actual model, fit, parameters and wave.
        # Old arguments cannot distinguish aliases or changed fit resources;
        # retain that historical rule only for untagged archived events.
        computation = (("canonical", canonical) if canonical is not None else
                       ("legacy", digest({"tool_id": payload["tool_id"],
                                          "parameters": arguments["parameters"]})))
        source, visited = event_id, set()
        while source in by_event and source not in visited:
            visited.add(source)
            previous = by_event[source]["result"].get("reused_from_event_id")
            if previous is None:
                break
            source = previous
        if computation not in seen_arguments and source not in seen_sources:
            unique.append(call_id)
        seen_arguments.add(computation)
        seen_sources.add(source)
    return unique


def validate_evidence_formulas(rows, calls):
    """Validate the new frozen formula using actual, sample-bound tool outputs.

    The Agent submits every number. Host checks but never replaces it, chooses
    a predictor, infers weights, clips values, or accesses evaluation labels.
    """
    issues = []
    for index, row in enumerate(rows):
        path = f"value.decisions[{index}]"
        refs, method = row["evidence_call_ids"], row["method"]
        if method not in {"model", "blend"}:
            issues.append({"code": "prediction_method_forbidden", "path": path + ".method"})
            continue
        if method == "model" and len(refs) != 1:
            issues.append({"code": "model_requires_one_result", "path": path + ".evidence_call_ids"})
            continue
        if method == "blend" and len(refs) < 2:
            issues.append({"code": "blend_requires_two_results", "path": path + ".evidence_call_ids"})
            continue
        cited = {ref: calls[ref] for ref in refs if ref in calls}
        values = []
        for ref in refs:
            payload = cited.get(ref, ({}, None))[0]
            result = payload.get("result", {})
            output = next((item for item in result.get("outputs", ())
                           if item.get("sample_id") == row["sample_id"]), None)
            if result.get("status") != "completed" or output is None:
                break
            values.append(output["predicted"])
        if len(values) != len(refs):
            issues.append({"code": "evidence_unavailable", "path": path + ".evidence_call_ids"})
            continue
        if len(unique_prediction_evidence(cited, provenance_calls=calls)) != len(refs):
            issues.append({"code": "duplicate_computation_evidence", "path": path + ".evidence_call_ids"})
            continue
        expected = values[0] if method == "model" else math.fsum(value / len(values) for value in values)
        if not math.isclose(row["predicted"], expected, rel_tol=1e-9, abs_tol=1e-9):
            issues.append({"code": "model_copy_mismatch" if method == "model" else "mean_blend_mismatch",
                           "path": path + ".predicted"})
    if issues:
        raise PredictionContractViolation("prediction does not match its frozen evidence formula", issues)


def validate_predictions(value, sample_ids: Sequence[str], *, wave_digest: str, allowed_methods=None):
    permitted = frozen_prediction_methods(allowed_methods)
    issues = []
    messages = []

    def invalid(message, code, path):
        messages.append(message)
        if len(issues) < 32:
            issues.append({"code": code, "path": path})

    def fail(message, code, path):
        raise PredictionContractViolation(message, [{"code": code, "path": path}])

    if not isinstance(value, Mapping) or set(value) != {"schema_version", "wave_digest", "decisions"}:
        fail("agent prediction result fields are invalid", "prediction_fields_invalid", "value")
    if value["schema_version"] != AGENT_PREDICTION_SCHEMA:
        fail("agent prediction schema/wave mismatch", "prediction_binding_mismatch", "value.schema_version")
    if value["wave_digest"] != wave_digest:
        fail("agent prediction schema/wave mismatch", "prediction_binding_mismatch", "value.wave_digest")
    rows = value["decisions"]
    if not isinstance(rows, list):
        fail("agent must predict every sample exactly once", "sample_coverage_invalid", "value.decisions")
    if len(rows) != len(sample_ids):
        invalid("agent must predict every sample exactly once", "sample_coverage_invalid", "value.decisions")
    seen = set()
    for index, row in enumerate(rows):
        path = f"value.decisions[{index}]"
        if not isinstance(row, Mapping) or set(row) != {
            "sample_id", "predicted", "confidence", "reason_code", "method", "evidence_call_ids"
        }:
            invalid("agent prediction fields are invalid", "prediction_fields_invalid", path)
            continue
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or sample_id not in sample_ids or sample_id in seen:
            invalid("unknown or duplicate prediction sample", "sample_identity_invalid", path + ".sample_id")
        else:
            seen.add(sample_id)
        finite = {}
        for name in ("predicted", "confidence"):
            number = row[name]
            try:
                finite[name] = not isinstance(number, bool) and isinstance(number, (int, float)) and math.isfinite(number)
            except OverflowError:
                finite[name] = False
            if not finite[name]:
                invalid(f"agent {name} must be finite", "prediction_number_nonfinite", path + "." + name)
        if finite["confidence"] and not 0 <= row["confidence"] <= 1:
            invalid("agent confidence must be in [0, 1]", "confidence_out_of_range", path + ".confidence")
        method = row["method"]
        if not isinstance(method, str) or method not in PREDICTION_METHODS:
            invalid("unknown agent prediction method", "prediction_method_unknown", path + ".method")
        elif method not in permitted:
            invalid("agent prediction method is outside the candidate's directive", "prediction_method_forbidden", path + ".method")
        if not isinstance(row["reason_code"], str) or row["reason_code"] not in REMOTE_REASON_CODES:
            invalid("agent reason code is invalid", "prediction_reason_invalid", path + ".reason_code")
        refs = row["evidence_call_ids"]
        refs_valid = isinstance(refs, list) and len(refs) <= MAX_PREDICTION_CALLS and all(
            isinstance(ref, str) and 1 <= len(ref) <= 80 for ref in refs
        ) and len(set(refs)) == len(refs)
        if not refs_valid:
            invalid("agent evidence references are invalid", "evidence_references_invalid", path + ".evidence_call_ids")
        elif method != "direct" and not refs:
            invalid("model-based predictions require tool evidence", "evidence_required", path + ".evidence_call_ids")
        elif method == "blend" and len(refs) < 2:
            invalid("blended predictions require at least two tool results", "blend_requires_two_results", path + ".evidence_call_ids")
    if issues:
        raise PredictionContractViolation(messages[0], issues)
    return rows


def validate_prediction_receipt(structured, receipt, *, event_lookup, identity, before_seq):
    """Validate replay from persisted evidence; never re-run a model during replay."""
    formula_bound = isinstance(receipt, Mapping) and receipt.get("schema_version") == "ecologyrsi-dsh.agent-prediction-receipt/3"
    fields = {
        "schema_version", "wave_digest", "sample_ids", "result_digest", "calls"
    } | ({"formula_policy"} if formula_bound else set())
    if not isinstance(receipt, Mapping) or set(receipt) != fields or (
        receipt["schema_version"] != "ecologyrsi-dsh.agent-prediction-receipt/2" and not formula_bound
    ) or (formula_bound and receipt["formula_policy"] != EVIDENCE_FORMULA_POLICY):
        raise ValueError("agent prediction receipt is invalid")
    sample_ids = receipt["sample_ids"]
    if not isinstance(sample_ids, list) or not sample_ids or not all(isinstance(s, str) for s in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("agent receipt sample ids are invalid")
    rows = validate_predictions(structured, sample_ids, wave_digest=receipt["wave_digest"])
    if receipt["result_digest"] != digest(structured):
        raise ValueError("agent prediction receipt digest mismatch")
    calls = receipt["calls"]
    if not isinstance(calls, list) or len(calls) > MAX_PREDICTION_CALLS:
        raise ValueError("agent prediction call budget exceeded")
    successful = set()
    seen = set()
    formula_calls = {}
    for call in calls:
        if not isinstance(call, Mapping) or set(call) != {"call_id", "event_id", "output_digest"} or call["call_id"] in seen:
            raise ValueError("agent prediction call receipt is invalid")
        seen.add(call["call_id"])
        event = event_lookup(call["event_id"])
        if event is None or event.kind != "DshPredictionToolExecuted" or event.seq >= before_seq:
            raise ValueError("agent evidence is not bound to a prior tool event")
        p = event.payload
        validate_tool_event(p)
        if any(p[k] != identity[k] for k in ("stage_attempt", "idempotency_key")) or p["wave_digest"] != receipt["wave_digest"] or p["sample_ids"] != sample_ids or p["call_id"] != call["call_id"] or p["output_digest"] != call["output_digest"]:
            raise ValueError("agent evidence belongs to another wave")
        if p["result"]["status"] == "completed":
            successful.add(call["call_id"])
        # State replay deliberately exposes only the validated event payload,
        # kind and sequence. The lookup key is already the receipt's event ID;
        # requiring an event_id attribute would invalidate archived v2 runs.
        formula_calls[call["call_id"]] = (p, call["event_id"])
    if any(set(row["evidence_call_ids"]) - successful for row in rows):
        raise ValueError("agent cites missing or failed prediction evidence")
    if formula_bound:
        validate_evidence_formulas(rows, formula_calls)


def validate_tool_event(p):
    # `origin_timestamp` is optional: it names the forecast origin a wave served,
    # which the payload never carried, so reading the ledger could not tell which
    # origin a tool call belonged to without joining through sample_ids. Archived
    # events predate it and stay valid; it is outside `arguments`, so neither
    # `request_digest` nor the reuse comparison in the binding is affected.
    if not isinstance(p, Mapping) or set(p) - {"origin_timestamp"} != {
        "schema_version", "stage", "stage_attempt", "idempotency_key", "tool_id", "wave_digest",
        "sample_ids", "prediction_count", "request_digest", "output_digest", "execution_owner",
        "call_id", "arguments", "result"
    } or p["schema_version"] != "ecologyrsi-dsh.dsh-prediction-tool-executed/2":
        raise ValueError("prediction tool event schema is invalid")
    if "origin_timestamp" in p and (
        isinstance(p["origin_timestamp"], bool)
        or not isinstance(p["origin_timestamp"], (str, int, float))
        or (isinstance(p["origin_timestamp"], str) and not 0 < len(p["origin_timestamp"]) <= 64)
    ):
        raise ValueError("prediction tool event origin is invalid")
    if p["stage"] != "sample.plan" or p["execution_owner"] != "dsh_agent_tool_call":
        raise ValueError("prediction tool event owner is invalid")
    result = p["result"]
    if not isinstance(result, Mapping) or p["output_digest"] != digest(result) or p["request_digest"] != digest(p["arguments"]):
        raise ValueError("prediction tool event digest mismatch")
    if result.get("status") not in {"completed", "failed"} or result.get("tool_id") != p["tool_id"] or result.get("call_id") != p["call_id"] or result.get("wave_digest") != p["wave_digest"]:
        raise ValueError("prediction tool event result mismatch")
    cost = result.get("charged_calls", 1)
    if type(cost) is not int or not 1 <= cost <= 4:
        raise ValueError("prediction tool event call cost is invalid")
    arguments = p["arguments"]
    if (not isinstance(arguments, Mapping)
            or set(arguments) != {"tool_id", "call_id", "wave_digest", "parameters"}
            or any(arguments[key] != p[key] for key in ("tool_id", "call_id", "wave_digest"))
            or not isinstance(arguments["parameters"], Mapping)):
        raise ValueError("prediction tool event arguments mismatch")
    sample_ids = p["sample_ids"]
    if (not isinstance(sample_ids, list) or not sample_ids
            or not all(isinstance(s, str) and s for s in sample_ids)
            or len(set(sample_ids)) != len(sample_ids)
            or type(p["prediction_count"]) is not int
            or p["prediction_count"] != len(sample_ids)):
        raise ValueError("prediction tool event sample set is invalid")
    if result["status"] == "completed":
        outputs = result.get("outputs")
        if (not isinstance(outputs, list) or len(outputs) != len(sample_ids)
                or any(not isinstance(row, Mapping) for row in outputs)
                or [row.get("sample_id") for row in outputs] != sample_ids):
            raise ValueError("prediction tool event output sample set mismatch")
        for row in outputs:
            value = row.get("predicted")
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("prediction tool event output must be finite")
        _canonical_computation_key(p)


AGENT_REVIEW_SCHEMA = "ecology-sample-review@2"


def validate_agent_review(value, sample_ids):
    """Pre-score criticism can request Agent revision, never select a Host forecast."""
    if not isinstance(value, Mapping) or set(value) != {"decisions"}:
        raise ValueError("agent review result fields are invalid")
    rows = value["decisions"]
    if not isinstance(rows, list) or len(rows) != len(sample_ids):
        raise ValueError("critic must review every selected sample exactly once")
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"sample_id", "action", "reason_code", "confidence"}:
            raise ValueError("agent review fields are invalid")
        sid = row["sample_id"]
        if not isinstance(sid, str) or sid not in sample_ids or sid in seen:
            raise ValueError("unknown or duplicate critic sample")
        seen.add(sid)
        if row["action"] not in {"accept", "revise", "uncertain"} or row["reason_code"] not in REMOTE_REASON_CODES:
            raise ValueError("critic action or reason is invalid")
        confidence = row["confidence"]
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("critic confidence must be in [0, 1]")
    return rows
