"""Allowlisted runtime failure metadata shared by transport and ledger replay."""
from collections.abc import Mapping


def validate_runtime_failure(value, *, error_code):
    fields = {"schema_version", "error_code", "failure_domain", "retryable",
              "provider_status", "retry_after_ms", "affected_scope"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("runtime failure fields are invalid")
    if (value["schema_version"] != "ecology-runtime-failure/1"
            or value["error_code"] != error_code
            or type(value["retryable"]) is not bool
            or value["failure_domain"] not in {"provider", "execution_contract"}
            or value["affected_scope"] not in {"model_route", "stage"}):
        raise ValueError("runtime failure contract is invalid")
    retryable = value["retryable"]
    if ((value["failure_domain"] == "provider") != retryable
            or (value["affected_scope"] == "model_route") != retryable):
        raise ValueError("runtime failure recovery semantics are inconsistent")
    for name, lower, upper in (("provider_status", 400, 599), ("retry_after_ms", 1, 3_600_000)):
        item = value[name]
        if item is not None and (type(item) is not int or not lower <= item <= upper):
            raise ValueError("runtime failure numeric metadata is invalid")
    return dict(value)
