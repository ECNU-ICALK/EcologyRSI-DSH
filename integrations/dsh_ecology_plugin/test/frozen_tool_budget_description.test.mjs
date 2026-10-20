import assert from 'node:assert/strict';
import test from 'node:test';
import { TOOL_DEFINITIONS } from '../lib/tools/definitions.js';
import { registerRoleTools } from '../lib/tools/roles.js';

test('registered planner tool follows its frozen budget rather than a fixed count', () => {
  const registered=[];
  registerRoleTools({tools:{register(tool){registered.push(tool);return ()=>{};}}}, {
    role:'sample-planner',bridge:{bindingFor:async()=>({}),sidecar:{request:async()=>({})}},
  });
  const tool=registered.find(item=>item.name==='ecology_execute_prediction_tool');
  assert.ok(tool);
  assert.match(tool.description,/prediction_contract\.max_prediction_tool_calls/);
  assert.match(tool.description,/remaining_calls/);
  assert.match(tool.description,/Failed calls also consume budget/);
  assert.doesNotMatch(tool.description,/at most (?:two|six|2|6)|third call|maximum (?:2|6)/i);
  assert.deepEqual(tool.parameters,TOOL_DEFINITIONS.ecology_execute_prediction_tool.parameters);
  assert.deepEqual(Object.keys(tool.parameters.properties),['tool_id','wave_digest','call_id','parameters']);
});
