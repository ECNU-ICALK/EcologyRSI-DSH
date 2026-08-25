# Dynamic DSH Retrieval Routing

## Objective

Make retrieval an on-demand capability throughout the EcologyRSI problem-solving loop instead of a one-time generation-start phase. Every EcologyRSI DSH role receives one model-facing tool named `web_search`. The model decides whether the current reasoning step needs evidence and supplies only bounded queries; it never selects a search provider.

The tool uses DSH's configured `ctx.web` service as the primary route. If that route fails technically or returns quantitatively insufficient evidence, the EcologyRSI sidecar automatically invokes the existing OpenAlex-backed retriever, merges the usable evidence, and persists the final result. DSH core remains unchanged.

## Goals

- Permit retrieval after the mandatory Skill lookup and before the terminal structured or prediction tool in every research-evolution stage.
- Prefer DSH's internal search without asking the model to name or understand providers.
- Fall back automatically and deterministically when the primary route fails or lacks usable evidence.
- Persist enough information to reproduce a completed retrieval without a second network call.
- Keep search advisory: it may inform reasoning but cannot bypass frozen data, registered-tool, evaluation, or promotion gates.
- Preserve the existing generation bootstrap search-plan and frozen evidence catalog for compatibility.

## Non-goals

- No direct model-facing `openalex_search`, provider selector, or fallback switch.
- No `web_fetch`, arbitrary URL fetching, shell execution, dependency installation, or dynamic code execution.
- No automatic conversion of dynamic search hits into trusted candidate `evidence_ref` values in this version.
- No removal or reinterpretation of historical run events.
- No change to hidden-label, holdout, scoring, or promotion boundaries.

## Architecture

```text
EcologyRSI stage Agent
  |  web_search({queries, idempotency_key})
  v
EcologyRSI DSH wrapper
  |-- replay lookup --------------------------> Python sidecar / event ledger
  |       | hit: return persisted result
  |       ` miss
  |
  |-- primary search: inherited ctx.web.search
  |       | success: bounded normalized result
  |       ` failure: safe machine-readable code
  |
  `-- complete request -----------------------> Python sidecar
          |-- recompute deterministic quality
          |-- if needed: OpenAlex fallback
          |-- merge, deduplicate, bound
          |-- append DshRetrievalExecuted
          `-- return final model-facing result
```

The wrapper is registered by `@ecologyrsi/dsh-evolution-plugin/agent-plugin`. It depends on DSH's inherited `web` service but does not mount or patch DSH's model-facing web tool. This leaves a single auditable tool definition and avoids exposing provider selection to the model.

## Tool contract

The model-facing input is:

```json
{
  "queries": ["one to four focused questions"],
  "idempotency_key": "stable stage-local key"
}
```

Constraints:

- 1-4 non-empty queries per call.
- 180 Unicode code points maximum per query.
- At most three distinct retrieval calls for one stage attempt.
- The idempotency key is stage-local, bounded, and supplied by the Agent from a purpose-oriented slug such as `check-vpd-literature`.
- The final result contains at most eight sources and bounded text fields.

The model-facing result includes a short `content` summary, normalized `sources`, and safe routing metadata (`provider_route`, `fallback_reason`, `result_digest`). Credentials, raw provider errors, stack traces, sidecar addresses, and internal tokens are never returned.

## Role and stage policy

`web_search` is authorized for coordinator, researcher, candidate proposer, sample planner, sample critic, and generation judge. The canonical call order is:

```text
required skill lookup
    -> zero to three web_search calls, only when evidence is needed
    -> required prediction tool when applicable
    -> required structured_output
```

Skill remains the first tool call. Search is optional, may occur at any reasoning point after Skill, and does not replace the terminal tool. Sample Planner must still call the registered vector predictor exactly once. Sample Critic cannot rewrite predictions. Retrieval queries receive only the already authorized stage context and never hidden labels.

## Primary route and cancellation

For each query, the wrapper invokes DSH's configured `ctx.web.search` service. Multi-query calls may execute concurrently, but result ordering is normalized to query order before completion.

The following failures are eligible for automatic fallback:

- configured provider unavailable or missing credentials;
- provider error or timeout;
- malformed provider response;
- empty usable result set.

Explicit cancellation is not eligible. If the stage or request signal is aborted, the wrapper propagates the cancellation and does not start a fallback request.

Provider failures are mapped to a small allowlist of safe codes before being sent to the sidecar. Raw error text is discarded.

## Deterministic evidence quality

The Python sidecar, rather than the model or JavaScript wrapper, is the authority for deciding whether primary evidence is sufficient. A normalized primary result is insufficient when any of these conditions holds:

1. It contains fewer than two distinct valid HTTPS source URLs.
2. It contains fewer than two evidence-bearing sources, where a source has a non-empty title, snippet, or content excerpt.
3. No normalized query term overlaps any source title, snippet, or content excerpt.

Normalization is deterministic: case folding, Unicode normalization, URL canonicalization, ASCII word tokens of at least three characters, and contiguous CJK bigrams. Common URL tracking parameters are removed before deduplication. Quality metrics and the selected reason code are written to the event.

Fallback reason codes are limited to:

- `primary_provider_unavailable`
- `primary_provider_error`
- `primary_timeout`
- `primary_malformed`
- `primary_empty`
- `insufficient_distinct_sources`
- `insufficient_evidence_sources`
- `insufficient_query_overlap`

## Fallback and merge behavior

On an eligible failure, the sidecar invokes the existing allowlisted OpenAlex metadata retrieval path using the same normalized queries. The fallback is metadata-only and cannot execute remote content.

Primary sources are retained first when usable. Fallback sources are appended in query/result order, deduplicated by canonical URL and stable source identity, then capped at eight. The final content is generated from bounded normalized fields; remote HTML is never injected. If fallback also yields no usable evidence, the tool returns a valid empty/partial result with the recorded failure route rather than inventing evidence.

Routes are recorded as:

- `dsh_primary`
- `dsh_primary_then_openalex_fallback`

## Persistence and replay

Before making any primary network request, the wrapper asks the sidecar for a replay result. The lookup key is derived from:

- run ID;
- stage name and role;
- stage attempt;
- idempotency key;
- digest of normalized queries.

On a miss, completion appends exactly one `DshRetrievalExecuted` event containing:

- schema version and execution owner;
- stage identity, idempotency key, and query digest;
- bounded normalized queries;
- provider route and optional fallback reason;
- primary quality metrics;
- bounded final result and result digest.

A repeated request with the same lookup key returns the persisted result before touching DSH or OpenAlex. A conflicting reuse of an idempotency key with different queries is rejected. Concurrent duplicate completions are serialized and converge on the first durable event.

The public projection exposes aggregate retrieval count, fallback count, routes, stages, and result digests. It does not expose credentials or unbounded source text.

## Scientific and security boundaries

- Dynamic retrieval is advisory context. It cannot add a trusted `evidence_ref` to the frozen generation evidence catalog in this version.
- Only the existing bootstrap retrieval path freezes catalog entries that may satisfy structured citation requirements.
- Retrieval never changes the immutable prediction vector, score, cohort, evaluator, or promotion decision.
- The model cannot name a provider, URL-fetch arbitrary content, or send headers/credentials.
- Sidecar validation rejects unauthorized roles, stale fences, over-limit queries, non-HTTPS sources, oversized payloads, and malformed results.
- Stored result text is bounded and treated as untrusted evidence, never as executable instructions.

## Compatibility and migration

The existing `generation.search-plan -> Host retrieval -> generation.research-synthesis` path remains the generation bootstrap and continues to freeze its evidence catalog. The new tool adds retrieval opportunities inside later stages; it does not replace that protocol or mutate existing event meanings.

New presets require `web_search` for all EcologyRSI roles. Historical runs whose frozen preset does not include it continue under their original tool surface. Replay accepts the new event kind while preserving prior ledgers unchanged.

## Acceptance criteria

- All six EcologyRSI roles expose exactly one `web_search` and no provider-specific search tool.
- Skill remains the first tool call, and optional searches are valid before each stage's required terminal tool.
- A sufficient DSH primary result never invokes OpenAlex.
- A technical primary failure or deterministic insufficiency invokes OpenAlex automatically.
- Explicit cancellation never invokes fallback.
- Replay returns the same result without calling either search backend.
- Conflicting idempotency reuse, excessive searches, stale fences, unauthorized roles, and malformed payloads fail closed.
- `DshRetrievalExecuted` survives ledger replay and appears in the bounded runtime projection.
- Existing generation bootstrap retrieval and all scientific gates remain intact.
- Python tests, DSH plugin Node tests, static checks, build verification, and a native smoke test pass; unavailable external credentials are reported as a diagnostic rather than misrepresented as success.
