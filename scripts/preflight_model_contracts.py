#!/usr/bin/env python3
"""Run transport-only preflight from a Host-bound task metadata JSON (no secrets)."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from ecologyrsi_dsh.integrations.dsh_native_runtime import DshNativeAgentRuntimeClient
from ecologyrsi_dsh.integrations.model_canary import CanaryBounds, run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="Host-bound task metadata JSON, or a task_manifest with metadata")
    parser.add_argument("--receipts", type=Path, default=Path(".runtime/model-canaries"))
    parser.add_argument("--dsh-origin", default="http://127.0.0.1:8848")
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--max-reported-tokens", type=int, default=30000)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--force", action="store_true", help="rerun fresh receipts; makes new paid provider requests")
    args = parser.parse_args()
    bounds = CanaryBounds(max_attempts=args.max_attempts, max_output_tokens=args.max_output_tokens,
                          max_reported_tokens=args.max_reported_tokens, total_timeout_ms=args.timeout_seconds * 1000)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    metadata = metadata.get("task_manifest", metadata)
    metadata = metadata.get("metadata", metadata)
    client = DshNativeAgentRuntimeClient(args.dsh_origin, token=os.environ.get("ECOLOGYRSI_DSH_RUNTIME_TOKEN", ""))
    result = run_preflight(client, metadata=metadata, receipt_directory=args.receipts, bounds=bounds, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
