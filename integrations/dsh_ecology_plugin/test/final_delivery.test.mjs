import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { specializeResearchOutputSchema } from '../lib/runtime/research-contract.js';
import { observeSessionUsage } from '../lib/runtime/session-usage.js';

const base = JSON.parse(readFileSync(new URL('../schemas/research-synthesis.schema.json', import.meta.url)));
function context(ref) {
  return { knowledge_snapshot: { evidence_catalog: [{ knowledge_id: ref, capability_ids: ['cap-a'] }] },
    synthesis_contract: { allowed_mutation_targets: {
      scientific_parameter: ['ridge_alpha'], registered_predictor: ['ridge@1'], instruction_profile: ['profile@1'],
    } }, required_candidate_direction_count: 4 };
}
test('research schema binds evidence and axis/target choices without changing cached schemas', () => {
  const schema = structuredClone(base);
  specializeResearchOutputSchema('generation.research-synthesis', schema, context('ref-a'));
  assert.deepEqual(schema.properties.evidence.items.properties.evidence_ref.enum, ['cap-a', 'ref-a']);
  const variants = schema.properties.candidate_directions.items.oneOf;
  assert.equal(variants.length, 3);
  assert.deepEqual(variants[0].properties.mutation_target.enum, ['ridge_alpha']);
  assert.deepEqual(variants[0].properties.mutation_direction.enum, ['increase', 'decrease']);
  assert.deepEqual(variants[1].properties.mutation_direction.enum, ['select']);
  assert.equal(base.properties.evidence.items.properties.evidence_ref.enum, undefined);
  const next = structuredClone(base);
  specializeResearchOutputSchema('generation.research-synthesis', next, context('ref-b'));
  assert.equal(JSON.stringify(next).includes('ref-a'), false);
  const invalid = context(''); invalid.knowledge_snapshot.evidence_catalog = [];
  assert.throws(() => specializeResearchOutputSchema('generation.research-synthesis', structuredClone(base), invalid), /empty/);
});

test('usage observer persists failed/cancelled calls and retries transport independently of result admission', async () => {
  for (const outcome of ['failed', 'cancelled', 'timed_out', 'succeeded']) {
    const calls = [];
    let input = 10;
    const session = { events: [{type: 'assistant/message', data: {usage: {inputTokens: 10}}}, {type: 'turn/end', data: {reason: {kind: 'completed'}}}] };
    const ctx = { sessions: { get: () => session }, sessionProjections: {snapshot: () => ({values: {tokenUsage: {uncachedInputTokens: input}}})} };
    let first = true;
    const observer = observeSessionUsage(ctx, {request: async (path, options) => {
      assert.ok(path.endsWith('/session-usage'));
      assert.equal(options.timeoutMs, 2000);
      if (first) { first = false; throw new Error('transient'); }
      calls.push(options.body); return {accepted: true};
    }}, {run_id: 'r', stage: 's', idempotency_key: 'k', child_reservation_id: 'c', session_id: 'sid'});
    input = 20;
    await observer.settle(outcome);
    assert.deepEqual(calls.map(c => c.settlement), ['active', outcome]);
    assert.equal(calls[0].usage_complete, false);
    assert.equal(calls.at(-1).usage_complete, !['cancelled', 'timed_out'].includes(outcome));
    assert.equal(calls.at(-1).session_metrics.provider_usage.totals.total_tokens, 20);
    await observer.settle(outcome);
    assert.equal(calls.length, 2);
  }
});

test('missing provider usage is reported as incomplete and transport errors do not replace execution result', async () => {
  const calls = [];
  const observer = observeSessionUsage({}, {request: async (_, options) => {
    calls.push(options.body); throw new Error('offline');
  }}, {session_id: 'missing'});
  await observer.settle('failed');
  assert.equal(calls.length, 4);
  assert.equal(calls.at(-1).usage_complete, false);
  assert.equal(calls.at(-1).session_metrics.provider_usage.available, false);
});


test('a failed partial provider stream is never marked complete from earlier messages', async () => {
  const session = {events: [{type: 'assistant/message', data: {usage: {inputTokens: 10}}},
    {type: 'turn/end', data: {reason: {kind: 'error'}}}]};
  const ctx = {sessions: {get: () => session}, sessionProjections: {snapshot: () => ({values: {tokenUsage: {uncachedInputTokens: 10}}})}};
  const calls = [];
  const observer = observeSessionUsage(ctx, {request: async (_, options) => {
    calls.push(options.body); return {accepted: true};
  }}, {session_id: 'partial'});
  await observer.settle('failed');
  assert.equal(calls.at(-1).usage_complete, false);
  assert.equal(calls.at(-1).session_metrics.provider_usage.totals.total_tokens, 10);
});

test('streaming and provider retry activity is visible without new billing or idle heartbeats', async () => {
  const session = {events: [{seq: 1, time: 300001, type: 'assistant/chunk', data: {privateText: 'never export'}}]};
  const ctx = {sessions: {get: () => session}, sessionProjections: {snapshot: () => ({values: {tokenUsage: {uncachedInputTokens: 10}}})}};
  const calls = [];
  const observer = observeSessionUsage(ctx, {request: async (_, {body}) => {calls.push(body); return {accepted: true};}},
    {session_id: 'stream'}, {intervalMs: 5});
  const waitFor = async (count) => {for (let i=0; i<100 && calls.length<count; i++) await new Promise(r=>setTimeout(r,5)); assert.equal(calls.length,count);};
  try {
    await waitFor(1);
    session.events.push({seq: 2, time: 300002, type: 'assistant/chunk'});
    await new Promise(r=>setTimeout(r,25));
    assert.equal(calls.length,1, 'chunks in one interval must be coalesced');
    session.events.push({seq: 3, time: 330001, type: 'assistant/chunk'});
    await waitFor(2);
    session.events.push({seq: 4, time: 330002, type: 'llm/retry', data: {failure: {message: 'private'}}});
    await waitFor(3);
    assert.equal(calls.at(-1).session_metrics.activity.kind,'retrying');
    assert.equal(calls.at(-1).session_metrics.provider_usage.totals.total_tokens,10);
    assert.equal(JSON.stringify(calls).includes('private'),false);
    await new Promise(r=>setTimeout(r,25));
    assert.equal(calls.length,3, 'a timer must not pretend a silent model progressed');
  } finally {await observer.settle('succeeded');}
});
