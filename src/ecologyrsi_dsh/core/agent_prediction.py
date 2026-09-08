"""Agent-owned numerical predictions and auditable, bounded evidence references."""

from collections.abc import Mapping, Sequence
import math

from .models import digest
from .redaction import REMOTE_REASON_CODES

AGENT_PREDICTION_SCHEMA = "ecology-sample-predictions@2"
MAX_PREDICTION_CALLS = 6


def validate_predictions(value, sample_ids: Sequence[str], *, wave_digest: str):
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "wave_digest", "decisions"}:
        raise ValueError("agent prediction result fields are invalid")
    if value["schema_version"] != AGENT_PREDICTION_SCHEMA or value["wave_digest"] != wave_digest:
        raise ValueError("agent prediction schema/wave mismatch")
    rows = value["decisions"]
    if not isinstance(rows, list) or len(rows) != len(sample_ids):
        raise ValueError("agent must predict every sample exactly once")
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "sample_id", "predicted", "confidence", "reason_code", "method", "evidence_call_ids"
        }:
            raise ValueError("agent prediction fields are invalid")
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or sample_id not in sample_ids or sample_id in seen:
            raise ValueError("unknown or duplicate prediction sample")
        seen.add(sample_id)
        for name in ("predicted", "confidence"):
            number = row[name]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ValueError(f"agent {name} must be finite")
        if not 0 <= row["confidence"] <= 1:
            raise ValueError("agent confidence must be in [0, 1]")
        if row["method"] not in {"direct", "model", "blend", "adjusted"}:
            raise ValueError("unknown agent prediction method")
        if row["reason_code"] not in REMOTE_REASON_CODES:
            raise ValueError("agent reason code is invalid")
        refs = row["evidence_call_ids"]
        if not isinstance(refs, list) or len(refs) > MAX_PREDICTION_CALLS or not all(
            isinstance(ref, str) and 1 <= len(ref) <= 80 for ref in refs
        ) or len(set(refs)) != len(refs):
            raise ValueError("agent evidence references are invalid")
        if row["method"] != "direct" and not refs:
            raise ValueError("model-based predictions require tool evidence")
        if row["method"] == "blend" and len(refs) < 2:
            raise ValueError("blended predictions require at least two tool results")
    return rows


def validate_prediction_receipt(structured, receipt, *, event_lookup, identity, before_seq):
    """Validate replay from persisted evidence; never re-run a model during replay."""
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema_version", "wave_digest", "sample_ids", "result_digest", "calls"
    } or receipt["schema_version"] != "ecologyrsi-dsh.agent-prediction-receipt/2":
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
    if any(set(row["evidence_call_ids"]) - successful for row in rows):
        raise ValueError("agent cites missing or failed prediction evidence")


def validate_tool_event(p):
    if not isinstance(p, Mapping) or set(p) != {
        "schema_version", "stage", "stage_attempt", "idempotency_key", "tool_id", "wave_digest",
        "sample_ids", "prediction_count", "request_digest", "output_digest", "execution_owner",
        "call_id", "arguments", "result"
    } or p["schema_version"] != "ecologyrsi-dsh.dsh-prediction-tool-executed/2":
        raise ValueError("prediction tool event schema is invalid")
    if p["stage"] != "sample.plan" or p["execution_owner"] != "dsh_agent_tool_call":
        raise ValueError("prediction tool event owner is invalid")
    result = p["result"]
    if not isinstance(result, Mapping) or p["output_digest"] != digest(result) or p["request_digest"] != digest(p["arguments"]):
        raise ValueError("prediction tool event digest mismatch")
    if result.get("status") not in {"completed", "failed"} or result.get("tool_id") != p["tool_id"] or result.get("call_id") != p["call_id"] or result.get("wave_digest") != p["wave_digest"]:
        raise ValueError("prediction tool event result mismatch")
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
