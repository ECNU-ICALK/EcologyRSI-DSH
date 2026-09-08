import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

function sandbox() {
  const nodes = new Map();
  const node = selector => {
    if (!nodes.has(selector)) nodes.set(selector, {textContent:'', hidden:false, value:'', innerHTML:'', disabled:false});
    return nodes.get(selector);
  };
  const c = {console, URL, URLSearchParams, AbortController, setTimeout, clearTimeout,
    window: {location:{search:''},setTimeout,clearTimeout}, document:{querySelector:node,querySelectorAll:()=>[]},
    EcologyDSHHost:{getPublicContext:()=>({apiBase:'/api'}),request:async()=>({})}};
  vm.createContext(c);
  for (const file of ['core','catalog','data']) vm.runInContext(readFileSync(new URL('../assets/js/'+file+'.js',import.meta.url),'utf8'),c);
  c.renderAll=()=>{};c.renderProcess=()=>{};c.renderTrainingAssets=()=>{};c.showToast=()=>{};
  c.refreshEventsForRun=async()=>true;c.refreshCandidateSamples=async()=>true;c.loadSelectedDataset=async()=>true;
  c.state.activeRun=c.normalizeRun({run_id:'run:a',status:'running',projection_revision:5,candidates_count:1,candidates:[{id:'candidate:a',status:'evaluating'}]});
  c.state.runs=[c.state.activeRun];
  return c;
}
const section = (view, revision=5, fields={}) => ({view,projection:{run_id:'run:a',status:'running',projection_revision:revision,...fields}});

test('dataset adapters control the training selector and per-dataset evaluation grid',()=>{
  const c=sandbox();
  const dataset=(id,cells)=>({id,training_selectable:true,task_adapter:{evaluator_id:'engine'},
    evaluation:{id:'engine',prediction_cells_per_origin:cells,minimum_samples_per_update:cells}});
  c.state.usingDemo=false;
  c.state.catalog.datasets=[dataset('dataset:a',9),dataset('dataset:b',4),
    {id:'synthetic',available:true}, {...dataset('unready',2),readiness:{ready:false}}];
  c.state.catalog.evaluators=[{id:'engine',prediction_cells_per_origin:99}];
  assert.deepEqual(Array.from(c.runnableDatasetItems(),x=>x.id),['dataset:a','dataset:b']);
  for(const [id,cells] of [['dataset:a',9],['dataset:b',4]]){
    c.document.querySelector('#dataset-id').value=id;
    c.alignDatasetBinding();
    assert.equal(c.document.querySelector('#evaluator-id').value,'engine');
    assert.equal(c.predictionCellsPerOrigin(),cells);
    assert.equal(c.samplesPerUpdateMinimum(),cells);
  }
  c.state.catalog.datasets[1].task_adapter.evaluator_id='missing';
  c.alignDatasetBinding();
  assert.equal(c.document.querySelector('#evaluator-id').value,'');
});

test('sample pages require the current envelope and requested run, candidate and page',()=>{
  const c=sandbox();
  const page={schema_version:'ecologyrsi-dsh.browser-sample-results/1',run_id:'run:a',candidate_id:'candidate:a',rows:[],offset:0,limit:25,total:0};
  const parse=payload=>c.normalizeCandidateSamplePage(payload,'run:a','candidate:a',0,25);
  assert.equal(parse(page).source,'api');
  assert.throws(()=>parse({...page,schema_version:'ecologyrsi-dsh.run-sample-page/1'}),/响应格式/);
  assert.throws(()=>parse({...page,run_id:'run:b'}),/身份/);
  assert.throws(()=>parse({...page,candidate_id:'candidate:b'}),/身份/);
  assert.throws(()=>parse({...page,offset:25}),/分页/);
  assert.throws(()=>parse({...page,rows:Array(26).fill({}),total:26}),/分页/);
  assert.equal(c.normalizeCandidateSamplePage({...page,offset:25},'run:a','candidate:a',25,25).rows.length,0);
});

test('hidden sections do not load and concurrent visits share one section request',async()=>{
  const c=sandbox();let calls=0,finish;
  c.request=()=>{calls++;return new Promise(resolve=>{finish=resolve;});};
  c.state.workspace='settings';await c.ensureWorkspaceData();assert.equal(calls,0);
  c.state.workspace='process';const a=c.ensureWorkspaceData(),b=c.ensureWorkspaceData();
  assert.equal(a,b);assert.equal(calls,1);
  finish(section('process',5,{rounds:[{generation:1}],candidate_summaries:[{id:'candidate:a',score:.1}]}));
  await a;assert.equal(c.state.activeRun.rounds.length,1);assert.equal(c.state.activeRun.candidate_summaries[0].score,.1);
  assert.equal(c.state.activeRun.candidates[0].score,undefined);
  await c.ensureWorkspaceData({navigation:true});assert.equal(calls,1);
});

test('late workspace response cannot roll back newer control status or progress',async()=>{
  const c=sandbox();let finish;c.state.workspace='candidates';
  c.request=()=>new Promise(resolve=>{finish=resolve;});const loading=c.ensureWorkspaceData();
  c.state.activeRun={...c.state.activeRun,status:'paused',projection_revision:8,tokens_used:100};
  finish(section('candidates',5,{tokens_used:1,candidates:[{id:'candidate:a',score:.2}],artifacts:[]}));
  await loading;assert.equal(c.state.activeRun.status,'paused');assert.equal(c.state.activeRun.projection_revision,8);
  assert.equal(c.state.activeRun.tokens_used,100);assert.equal(c.state.workspaceVersions.candidates,5);
  c.renderWorkspaceLoadState();assert.match(c.document.querySelector('#workspace-data-message').textContent,/较早快照/);
});

test('process summaries and candidate detail cannot overwrite each other across revisions',async()=>{
  const c=sandbox();let finish;c.state.workspace='candidates';
  c.request=()=>new Promise(resolve=>{finish=resolve;});
  const oldDetail=c.ensureWorkspaceData();
  c.state.workspace='process';
  c.request=async()=>section('process',8,{candidate_summaries:[{id:'candidate:a',status:'evaluated',score:.8}],rounds:[]});
  await c.ensureWorkspaceData();
  finish(section('candidates',5,{candidates:[{id:'candidate:a',status:'evaluating',score:.2,metrics:{version:'old'}}],artifacts:[]}));
  await oldDetail;
  assert.equal(c.state.activeRun.projection_revision,8);
  assert.equal(c.processCandidates(c.state.activeRun)[0].score,.8);
  assert.equal(c.processCandidates(c.state.activeRun)[0].metrics.version,undefined);
  assert.equal(c.state.activeRun.candidates[0].metrics.version,'old');
  assert.equal(c.state.workspaceVersions.candidates,5);
  assert.equal(c.selectedCandidateForSamples().score,.8);
});

test('collaboration options stay independent of scientific candidate snapshots',async()=>{
  const c=sandbox();c.state.workspace='collaboration';
  c.state.activeRun.candidates=[{id:'candidate:a',score:.4,metrics:{bound:'detail'}}];
  c.state.activeRun.candidate_summaries=[{id:'candidate:a',score:.6}];
  c.request=async()=>section('collaboration',9,{intervention_candidates:[{id:'candidate:b',generation:2}],interventions:[],expert_consultations:[]});
  await c.ensureWorkspaceData();
  assert.equal(c.state.activeRun.intervention_candidates[0].id,'candidate:b');
  assert.equal(c.state.activeRun.candidates[0].metrics.bound,'detail');
  assert.equal(c.state.activeRun.candidate_summaries[0].score,.6);
});

test('switching runs rejects old section data and failed reads remain retryable',async()=>{
  const c=sandbox();let finish;c.state.workspace='process';c.request=()=>new Promise(resolve=>{finish=resolve;});
  const pending=c.ensureWorkspaceData();c.state.activeRun=c.normalizeRun({run_id:'run:b',projection_revision:9,status:'completed'});
  finish(section('process',5,{rounds:[{generation:1}]}));assert.equal(await pending,false);assert.equal(c.state.activeRun.rounds.length,0);
  c.state.activeRun=c.normalizeRun({run_id:'run:a',projection_revision:5,status:'running'});
  c.request=async()=>{throw new Error('temporary outage');};assert.equal(await c.ensureWorkspaceData({force:true}),false);
  assert.equal(c.state.workspaceErrors.process,c.errorMessage(new Error('temporary outage')));
  c.request=async()=>section('process',5,{rounds:[]});assert.equal(await c.ensureWorkspaceData({force:true}),true);
  assert.equal(c.state.workspaceErrors.process,undefined);
});

test('training trace loads only by explicit candidate request and verifies identity',async()=>{
  const c=sandbox();c.state.workspace='training';let calls=[];
  c.request=async path=>{calls.push(path);return path.includes('view=asset')
    ? {projection:{run_id:'run:a',projection_revision:5,training_asset:{candidate_id:'candidate:a',episode:{stages:{}}}}}
    : section('training',5,{training_assets:[{candidate_id:'candidate:a',details_loaded:false}]});};
  await c.ensureWorkspaceData();assert.equal(calls.length,1);assert.ok(!calls[0].includes('view=asset'));
  assert.equal(await c.loadTrainingAsset('candidate:a'),true);assert.ok(calls[1].includes('candidate_id=candidate%3Aa'));
  assert.equal(c.state.trainingAssetDetails['candidate:a'].asset.details_loaded,true);
  c.request=async()=>({projection:{run_id:'run:b',training_asset:{candidate_id:'candidate:a'}}});
  assert.equal(await c.loadTrainingAsset('candidate:a'),false);
});

test('GET requests coalesce, metadata cache is bounded and scoped to host context',async()=>{
  const c=sandbox();let calls=0,finish;
  c.EcologyDSHHost.request=()=>{calls++;return new Promise(resolve=>{finish=resolve;});};
  const a=c.request('/catalog'),b=c.request('/catalog');assert.equal(calls,1);finish({datasets:[1]});
  const first=await a,second=await b;first.datasets.push(2);assert.equal(second.datasets.length,1);
  await c.request('/catalog');assert.equal(calls,1);
  c.state.contextEpoch++;const other=c.request('/catalog');assert.equal(calls,2);finish({datasets:[3]});await other;
  c.EcologyDSHHost.request=async()=>({rows:[]});
  for(let i=0;i<30;i++)await c.request('/datasets/d/samples?offset='+i);
  assert.ok(c.cachedReads.size<=16);
});

test('renders only the visible heavy workspace',()=>{
  const c=sandbox();const app=readFileSync(new URL('../app.js',import.meta.url),'utf8').split('\n    bindEvents();')[0];
  vm.runInContext(app,c);const called=[];
  for(const name of ['renderTraining','renderProcess','renderCandidates','renderCollaboration'])c[name]=()=>called.push(name);
  c.state.workspace='settings';c.renderActiveWorkspace();assert.deepEqual(called,[]);
  c.state.workspace='candidates';c.renderActiveWorkspace();assert.deepEqual(called,['renderCandidates']);
});

test('startup overview loads in the background without blocking configuration',async()=>{
  const c=sandbox();let finish;
  c.state.activeRun=null;c.state.runs=[];c.state.workspace='settings';
  c.populateCatalogControls=()=>{};c.scheduleEvolutionCapacityRefresh=()=>{};
  c.ensureWorkspaceData=async()=>true;c.startRunMonitor=()=>false;
  c.request=path=>path.includes('view=overview') ? new Promise(resolve=>{finish=resolve;}) : Promise.resolve(path.startsWith('/runs') ? {runs:[{run_id:'run:a',status:'completed',schema_version:'ecologyrsi-dsh.browser-run-summary/2'}]} : {});
  const startup=c.connectAndLoad();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(c.state.connection,'online');assert.equal(c.state.loadState,'ready');
  assert.equal(c.state.busy,false);assert.equal(c.state.runOverviewLoading,'run:a');
  assert.equal(c.state.activeRun,null);
  finish({projection:{run_id:'run:a',status:'completed',projection_revision:8}});
  assert.equal(await startup,true);assert.equal(c.state.activeRun.id,'run:a');assert.equal(c.state.runOverviewLoading,null);
});

test('background overview failure stays local and a newer selection wins',async()=>{
  const c=sandbox();c.startRunMonitor=()=>false;c.state.connection='online';c.state.loadState='ready';
  c.request=async()=>{throw new Error('slow history unavailable');};
  assert.equal(await c.selectRun('run:a',false,{background:true}),false);
  assert.equal(c.state.connection,'online');assert.equal(c.state.commandError,null);
  assert.match(c.state.runOverviewError,/无法读取进化运行/);assert.equal(c.state.busy,false);
  let finish;c.request=()=>new Promise(resolve=>{finish=resolve;});
  const old=c.selectRun('run:a',false,{background:true});
  c.state.runReadRequest++;c.state.runOverviewLoading=null;c.state.activeRun=c.normalizeRun({run_id:'run:new',status:'running'});
  finish({projection:{run_id:'run:a',status:'completed'}});
  assert.equal(await old,false);assert.equal(c.state.activeRun.id,'run:new');
});

test('picker rows do not invent scientific outcomes or hide cancelled research',()=>{
  const c=sandbox();const row=c.normalizeRun({schema_version:'ecologyrsi-dsh.browser-run-summary/2',run_id:'run:a',status:'completed'});
  assert.equal(row.outcome,null);
  assert.equal(c.isCancelledEmptyRun({...row,status:'cancelled',has_evolution_progress:true}),false);
});
