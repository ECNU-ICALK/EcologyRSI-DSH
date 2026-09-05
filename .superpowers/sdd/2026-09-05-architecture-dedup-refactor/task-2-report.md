# Phase 2 projection refactor report

## Scope

Added shared run configuration, progress, and evidence projection builders. The
existing full and summary projections now obtain their configuration from the
same builder and retain their public schema and scoring behavior.

## Verification

- `PYTHONPATH=src .venv/bin/python -m unittest tests.test_projection_contract tests.test_execution_projection tests.test_http` — 113 tests passed.
- `node --test plugins/ecology_evolution/test/smoke.mjs` — passed.

## Risk

Progress lifecycle counts are conservative and derived only from durable
candidate states; provider/admission-specific counters remain zero until their
events are available to the projection layer.
