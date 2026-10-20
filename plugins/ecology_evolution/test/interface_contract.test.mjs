import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';

const root = new URL('../', import.meta.url);
const read = name => readFileSync(new URL(name, root), 'utf8');
const html = read('index.html');
const manifest = JSON.parse(read('plugin.json'));
const scripts = Array.from(html.matchAll(/<script src="([^"]+)" defer><\/script>/g), match => match[1]);
const code = scripts.map(src => read(src.split('?')[0])).join('\n');

test('every static application selector has exactly one visible document node', () => {
  const ids = Array.from(html.matchAll(/\bid="([^"]+)"/g), match => match[1]);
  assert.equal(new Set(ids).size, ids.length, 'duplicate DOM identities');
  for (const match of code.matchAll(/\$\("#([\w-]+)"\)/g)) {
    assert.ok(ids.includes(match[1]), `missing DOM node #${match[1]}`);
  }
  assert.ok(ids.includes('last-updated'));
  for (const id of ['dataset-id', 'sample-inference-list', 'candidate-table', 'pause-button',
    'archive-button', 'delete-button', 'independent-validation-start']) {
    assert.ok(ids.includes(id), `required workflow absent: ${id}`);
  }
  assert.match(code, /controlRun\(state\.activeRun\.status === \"paused\" \? \"resume\" : \"pause\"\)/);
});

test('the loaded surface contains only current workspaces and versioned local scripts', () => {
  assert.equal(scripts.length, 10);
  for (const src of scripts) {
    assert.ok(src.endsWith(`?v=${manifest.version}`), `stale cache identity: ${src}`);
    assert.match(src, /^(?:assets\/js\/[a-z_]+|app)\.js\?v=[\d.]+$/);
  }
  assert.doesNotMatch(html + code, /workspace-collaboration|training-assets-table|intervention-form|loadDemo|submitIntervention|answerExpertConsultation|state\.usingDemo/);
  assert.doesNotMatch(code, /inference_trace|prediction_preview|run_projection/);
});

test('static parameter defaults match the bootstrap contract before catalog loading', () => {
  const contract = JSON.parse(/var bootstrapRunParameters = (.*);/.exec(read('assets/js/commands.js'))[1]);
  const fields = {
    rounds: 'max-generations', candidates_per_generation: 'candidates-per-generation',
    max_candidates: 'max-candidates', formal_origin_count: 'formal-origin-count',
    local_batch_origin_count: 'local-batch-origin-count', max_local_edits_per_batch: 'max-local-edits-per-batch',
    selection_holdout_origin_count: 'selection-holdout-origin-count',
    candidate_concurrency: 'candidate-concurrency', sample_concurrency: 'sample-concurrency',
  };
  for (const [name, id] of Object.entries(fields)) {
    const input = html.match(new RegExp(`<input[^>]*id="${id}"[^>]*>`))?.[0];
    assert.ok(input, `missing parameter ${name}`);
    assert.equal(Number(/\bvalue="([^"]+)"/.exec(input)[1]), contract.parameters[name].default, name);
  }
  for (const name of ['formal_origin_count','local_batch_origin_count','max_local_edits_per_batch','selection_holdout_origin_count']) {
    const rule = contract.parameters[name];
    assert.equal(rule.minimum, rule.default, `${name} is a frozen protocol value`);
    assert.equal(rule.maximum, rule.default, `${name} is a frozen protocol value`);
  }
});
