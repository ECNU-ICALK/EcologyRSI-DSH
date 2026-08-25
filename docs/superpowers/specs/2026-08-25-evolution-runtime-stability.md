# Evolution Runtime Stability Design

## Context

EcologyRSI-DSH 0.3.27 passed its existing Python and Node suites, but isolated
DSH-native runs exposed four runtime defects that the synthetic tests did not
cover:

1. Diagnostic statements such as "no eligibility, gate passage, or promotion
   is claimed" were read as affirmative promotion claims because negation was
   lost across comma-separated items.
2. A research response already receives two semantic repair calls. Exhausting
   both calls was still classified as a provider outage, so the outer scheduler
   could start a fresh pair forever. An older 0.3.25 run repeated this path more
   than 78 times and consumed roughly 7.7 million tokens.
3. `sample.critic` could call `structured_output` with `{}`, then continue
   producing prose until the shared ten-minute stage timeout. The affected
   cucumber run took 19 minutes 58 seconds; one critic child alone emitted
   32,768 output tokens before a later retry succeeded.
4. The browser projection recorded only the broad evolution stage. Provider
   pacing, active DSH execution, model retry, and host validation were therefore
   indistinguishable from a frozen run. A recovered stage failure also remained
   in `failed_stage` after the run completed.

The first clean 0.3.27 reproduction was
`run:0c31b32b-807f-4020-aa52-0cd393d093cb`. The fixed runtime was verified with
the cucumber run `run:3eb2f485-4811-4b0c-963a-24f72dbebecf` and tomato run
`run:265e8fa2-9885-4867-b1e3-541845868791`.

## Goals

- Accept genuinely negated diagnostic-only claims without weakening rejection
  of affirmative eligibility, gate-passage, selection, or promotion claims.
- Stop exhausted research semantic repair at a resumable, explicit circuit
  breaker instead of repeatedly spending model calls.
- Keep the critic's failure budget independent from unrelated long-running
  stages and make its structured-output protocol unambiguous.
- Project durable DSH substages and attempts without inventing percentages or
  live token counts.
- Ensure a completed run does not expose a recovered stage failure as current.

## Non-goals

- Diagnostic cohorts remain ineligible for selection or promotion.
- JSON-schema validation, executable-field rejection, mutation realizability,
  and frozen-evidence binding remain fail-closed.
- Production provider pacing remains 60 seconds. The isolated verification
  runtime uses a 10-second interval only to separate application behavior from
  provider throttling latency.
- Scores from different frozen cohorts or datasets are not compared.

## Design

### 1. Negated enumeration semantics

The claim scanner recognizes shared negation across bounded English list forms,
including `no ...`, `no claim of ...`, `does not constitute ...`, and
diagnostic selection/run-level variants observed in live output. Affirmative and
mixed-clause controls remain rejected, including a noun-starting independent
clause such as `promotion should proceed` after a completed negated assertion.

### 2. Research semantic circuit breaker

The two-call semantic repair loop remains the response-correction budget. Once
it is exhausted, `ResearchResponseContractError` is not a deferred provider
failure. Auto-progress durably pauses the run with:

- code `research_contract_retry_exhausted`;
- a bounded, redacted host-validation detail;
- no `GatewayRetryScheduled` event.

The run can be resumed from the research checkpoint after code or configuration
changes. Transport and rate-limit errors keep their existing delayed retry path.

### 3. Sample critic contract and timeout

`sample.critic` has a dedicated 180-second production timeout. Its instruction
requires the exact `wave_digest`, every sample identifier, and non-empty
structured arguments. A rejected tool call ends that child immediately without
prose; the Host performs any bounded retry in a fresh child reservation so the
one-call evidence contract remains exact. Other stages keep their existing
timeout.

### 4. Durable DSH activity projection

The run projection derives a compact `execution_progress.dsh_activity` object
from durable stage, reservation, and structured-result events. It distinguishes:

- `waiting_for_model_slot`;
- `model_running`;
- `model_retry_running`;
- `host_validating`.

The browser shows the evolution stage, DSH stage, role, attempt, elapsed time,
and last DSH heartbeat. Search-plan children are mapped to the broad `search`
stage rather than `research`. The projection does not expose reservation
identifiers or fabricate percent complete. Terminal runs hide this active
projection, and completed runs clear recovered `failed_stage` data.

## Real-run evidence

The fixed cucumber diagnostic completed in about 7 minutes 24 seconds, compared
with 19 minutes 58 seconds for the earlier run. Its critic succeeded on the
first attempt in about 20 seconds rather than reaching the ten-minute boundary.

The tomato diagnostic initially paused safely after two newly observed negated
phrases were rejected. After adding those exact regression forms, it resumed
from the research checkpoint and completed. During reflection, the host
correctly rejected a model-copied evidence digest missing one character, then a
later retry used the exact frozen digest and completed. Both completed runs pass
the DSH-native acceptance checks with one diagnostic candidate, nine prediction
cells, no Python model requests, and no promotion claim.

## Verification

1. Exact live-output unit tests plus affirmative controls.
2. Auto-progress integration tests for semantic pause and transport retry.
3. Projection tests for DSH states and recovered terminal state.
4. Node tests for critic timeout and prompt contract.
5. Browser rendering smoke tests.
6. Full Python and Node suites.
7. Fresh isolated cucumber and tomato DSH-native diagnostics.
