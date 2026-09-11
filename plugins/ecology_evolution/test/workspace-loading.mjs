import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const requests = [];
let renders = 0;
const state = {
  contextEpoch: 1, workspace: 'candidates', usingDemo: false,
  activeRun: {id: 'run', projection_revision: 100, workspace_revisions: {candidates: 3, candidate: 3}},
  workspaceVersions: {candidates: 3}, runs: [], selectedCandidateId: 'a',
};
const context = vm.createContext({state, Promise, Date, Number, Object, String, encodeURIComponent,
  dataRequestTimeout: 1000, $: () => null, renderAll() {}, renderCandidates() { renders++; },
  syncCandidateSelection() {}, refreshCandidateSamples() {}, normalizeRun: x => x,
  normalizeCandidate: x => x, errorMessage: e => e.message,
  request(path) { return new Promise(resolve => requests.push({path, resolve})); },
});
vm.runInContext(fs.readFileSync(new URL('../assets/js/catalog.js', import.meta.url), 'utf8'), context);

// Global telemetry revisions alone must not issue another workspace read.
assert.equal(await context.ensureWorkspaceData({navigation: true}), true);
assert.equal(requests.length, 0);
state.activeRun.projection_revision = 120;
assert.equal(await context.ensureWorkspaceData({navigation: true}), true);
assert.equal(requests.length, 0);
state.activeRun.workspace_revisions.candidates = 4;
const refresh = context.ensureWorkspaceData({navigation: true});
assert.equal(requests.length, 1);
requests[0].resolve({view: 'candidates', projection: {run_id: 'run', id: 'run',
  projection_revision: 120, workspace_revisions: {candidates: 4, candidate: 4}, candidates: [], artifacts: []}});
assert.equal(await refresh, true);
assert.equal(state.workspaceVersions.candidates, 4);

// Repeated renders coalesce one selected detail; late A may not render over B.
const a = context.loadCandidateDetail('a');
assert.equal(context.loadCandidateDetail('a'), a);
assert.equal(requests.length, 2);
assert.match(requests[1].path, /view=candidate&candidate_id=a$/);
state.selectedCandidateId = 'b';
requests[1].resolve({view: 'candidate', projection: {run_id: 'run', workspace_revisions: {candidate: 4},
  candidate: {candidate_id: 'a'}, artifacts: []}});
assert.equal(await a, true);
assert.equal(renders, 0);
const b = context.loadCandidateDetail('b');
requests[2].resolve({view: 'candidate', projection: {run_id: 'run', workspace_revisions: {candidate: 4},
  candidate: {candidate_id: 'b'}, artifacts: [{candidate_id: 'b'}]}});
assert.equal(await b, true);
assert.equal(renders, 1);
const foreign = context.loadCandidateDetail('b');
requests[3].resolve({view: 'candidate', projection: {run_id: 'different-run', candidate: {candidate_id: 'b'}}});
assert.equal(await foreign, false);
assert.match(state.candidateDetails['1|run|candidate:b'].error, /身份不匹配/);
console.log('workspace lazy loading, telemetry cache and selection races: ok');
