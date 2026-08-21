"""Local JSON API and DSH plugin host for the ecology evolution workbench.

The runtime stays deliberately small: one process, one append-only SQLite
ledger, bounded structured proposals, and fixed evaluator implementations.
Credentials remain server-side and browser responses expose only redacted
catalogue and projection data.
"""

from __future__ import annotations

from .api.shared import (
    AUTO_ADVANCE_CONTINUOUS,
    _PLUGIN_FILES,
    _assert_http_scope,
    _assert_manifest_http_scope,
    _auto_advance_steps,
    _budget_value,
    _derived_seed,
    _evaluation_partition,
    _event_type,
    _expected_partition,
    _is_loopback_host,
    _max_generations,
    _parse_steps,
    _plugin_root,
    _request_integer,
)
from .api.projection import (
    _candidate_projection,
    _intervention_projection,
    _projection_json,
    _public_intervention_receipt,
    _state_payload,
)
from .api.handler import (
    PLUGIN_MANIFEST,
    EvolutionHTTPServer,
    EvolutionRequestHandler,
)
from .api.runtime import serve
