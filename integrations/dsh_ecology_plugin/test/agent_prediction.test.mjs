import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { skillInvocationEvidence } from '../lib/runtime/stage-runner.js';
import { TOOL_DEFINITIONS } from '../lib/tools/definitions.js';

function chain(count, { errorAt = -1, lateTool = false } = {}) {
  const events = [];
  function pair(name, args = {}, error = false) {
    const callId = 'call-' + events.length;
    events.push({seq: events.length + 1, type: 'tool/call', data: {name, callId, arguments: args}});
    events.push({seq: events.length + 1, type: 'tool/result', data: {message: {content: [{type: 'tool-result', toolCallId: callId, isError: error}]}}});
  }
  pair('skill', {name: 'origin-vector-forecasting'});
  for (let i = 0; i < count; i++) pair('ecology_execute_prediction_tool', {}, i === errorAt);
  pair('structured_output');
  if (lateTool) pair('ecology_execute_prediction_tool');
  return events;
}
const options = { stage: 'sample.plan', skillName: 'origin-vector-forecasting', allowsPredictionTools: true };

test('sample Agent can finalize directly or after multiple tool results', () => {
  for (const count of [0, 2, 6]) {
    const evidence = skillInvocationEvidence(chain(count), options);
    assert.equal(evidence.next_tool_name, 'structured_output');
    assert.equal(evidence.order_verified, true);
  }
  assert.doesNotThrow(() => skillInvocationEvidence(chain(2, {errorAt: 0}), options));
});

test('sample Agent tool budget and finalization ordering are enforced', () => {
  assert.throws(() => skillInvocationEvidence(chain(7), options), /budget/);
  assert.throws(() => skillInvocationEvidence(chain(0, {lateTool: true}), options), /settle/);
  assert.throws(() => skillInvocationEvidence(chain(1), {...options, allowsPredictionTools: false}), /capability/);
});

test('model tool accepts bounded parameter requests and Agent result owns numeric output', async () => {
  const definition = TOOL_DEFINITIONS.ecology_execute_prediction_tool;
  assert.deepEqual(definition.parameters.required, ['tool_id','wave_digest','call_id','parameters']);
  const schema = JSON.parse(await readFile(new URL('../schemas/sample-decisions.schema.json', import.meta.url)));
  assert.equal(schema.$id, 'ecology-sample-predictions@2');
  assert.equal(schema.properties.decisions.items.properties.predicted.type, 'number');
  assert.equal('next_tool' in schema.properties.decisions.items.properties, false);
});
