"""Lossless sharing of repeated causal context within one Agent vector."""
from collections import Counter
import json


def compact_origin_contexts(contexts):
    counts = Counter()

    def encoded(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def count(value):
        if not isinstance(value, (dict, list)):
            return
        key = encoded(value)
        if len(key) >= 128:
            counts[key] += 1
        for child in value.values() if isinstance(value, dict) else value:
            count(child)

    for value in contexts.values():
        count(value)
    references, shared = {}, {}

    def replace(value):
        if not isinstance(value, (dict, list)):
            return value
        key = encoded(value)
        if counts[key] > 1:
            if key not in references:
                ref = f"v{len(shared) + 1}"
                references[key] = ref
                shared[ref] = value
            return {"shared_origin_ref": references[key]}
        if isinstance(value, dict):
            return {name: replace(child) for name, child in value.items()}
        return [replace(child) for child in value]

    compact = {name: replace(value) for name, value in contexts.items()}
    if len(encoded({"contexts": compact, "shared": shared})) >= len(encoded(contexts)):
        return contexts, {}
    return compact, shared
