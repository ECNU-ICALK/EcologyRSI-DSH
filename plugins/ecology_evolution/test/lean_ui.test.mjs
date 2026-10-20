import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

function fixture() {
  const nodes = new Map();
  const node = selector => {
    if (!nodes.has(selector)) nodes.set(selector, {
      value: '', textContent: '', innerHTML: '', hidden: false, style: {}, attributes: {},
      setAttribute(key,value) {this.attributes[key]=value;}, removeAttribute(key) {delete this.attributes[key];},
      getBoundingClientRect: () => ({width:900}), parentElement:{querySelector:()=>null}
    });
    return nodes.get(selector);
  };
  const c = {console,URL,URLSearchParams,AbortController,setTimeout,clearTimeout,
    document:{querySelector:node,querySelectorAll:()=>[]},window:{location:{search:''},innerWidth:1200},
    EcologyDSHHost:{getPublicContext:()=>({apiBase:'/api',models:[]})}};
  vm.createContext(c);
  for (const name of ['core','catalog','data','commands','render_shell','render_process','render_candidates']) {
    vm.runInContext(readFileSync(new URL('../assets/js/'+name+'.js',import.meta.url),'utf8'),c);
  }
  const run = c.normalizeRun({run_id:'run:a',status:'running',generation:0,total_generations:2,candidates_count:1,
    execution_progress:{current_stage:'evaluation',current_candidate_id:'candidate:a',overall_progress_percent:12.5,
      epoch_progress_percent:25,stage_progress:{completed_samples:5,total_samples:20}},
    candidates:[{candidate_id:'candidate:a',status:'evaluating',score:null}]});
  c.state.activeRun=run;c.state.runs=[run];c.state.workspace='process';
  c.state.selectedCandidateId='candidate:a';
  return {c,node,run};
}

test('current process renders service progress and unavailable evidence without synthetic zero',()=>{
  const {c,node,run}=fixture();c.renderProcess();
  assert.equal(node('#execution-progress-percent').textContent,'12.5%');
  assert.match(node('#execution-sample-progress').textContent,/5 \/ 20/);
  assert.match(node('#runtime-evidence').innerHTML,/待核验/);
  assert.match(node('#execution-token-progress').textContent,/不能据此认定免费/);
  run.execution_progress={current_stage:'research',overall_progress_percent:null};
  c.renderProcess();
  assert.equal(node('#execution-progress-percent').textContent,'待核验');
  assert.equal(node('#execution-progress-track').attributes['aria-valuenow'],undefined);
  assert.equal(c.executionCandidateFor(run),null);
  assert.doesNotMatch(node('#active-candidate-summary').innerHTML,/candidate:a/);
});

test('paused and terminal views retain counters without claiming an active old stage',()=>{
  const {c,node,run}=fixture();
  run.execution_progress.dsh_activity={dsh_stage:'sample.plan',started_at:'2020-01-01T00:00:00Z'};
  for(const status of ['paused','failed','cancelled','completed']){
    run.status=status;c.renderProcess();
    assert.equal(node('#execution-progress-label').textContent,c.statusText(status));
    assert.doesNotMatch(node('#execution-heartbeat').textContent,/本次阶段耗时/);
    assert.match(node('#execution-progress-detail').textContent,/保留停点证据/);
  }
});

test('canonical runtime projection shows the actual child stage and measured context tokens',()=>{
  const {c,node,run}=fixture();
  run.execution_progress.dsh_activity={dsh_stage:'sample.plan',role:'sample-planner',
    started_at:'2026-10-20T03:55:52.639Z',remote_activity:{kind:'waiting',updated_at:'2026-10-20T03:56:32.389Z'}};
  run.dsh_runtime={provider_usage:{available:true,total_tokens:12345,accounting_coverage:{
    complete:true,launched_session_count:1,observed_session_count:1,settled_session_count:1,
    complete_session_count:1,unobserved_session_count:0}},context_pressure:{
      available:true,scope:'active_sessions',session_count:1,maximum_total_tokens:12406,maximum_surface_tokens:10117}};
  c.renderProcess();
  assert.equal(node('#execution-progress-label').textContent,'逐时点预测');
  assert.match(node('#execution-token-progress').textContent,/本运行已启动会话回执完整/);
  assert.doesNotMatch(node('#execution-token-progress').textContent,/缺失用量/);
  assert.match(node('#runtime-evidence').innerHTML,/12,406 Token（活跃会话最大值）/);
  assert.doesNotMatch(node('#runtime-evidence').innerHTML,/活跃会话上下文压力<\/span><strong>未报告/);
  run.dsh_runtime.context_pressure={available:false,scope:'active_sessions',session_count:0};
  c.renderProcess();
  assert.match(node('#runtime-evidence').innerHTML,/活跃会话上下文压力<\/span><strong>未报告/);
});

test('one complete child receipt never marks all launched run sessions complete',()=>{
  const {c,node,run}=fixture();
  const all={complete:true,launched_session_count:2,observed_session_count:2,
    settled_session_count:2,complete_session_count:2,unobserved_session_count:0};
  run.dsh_runtime={provider_usage:{available:true,total_tokens:20000,complete:true,accounting_coverage:all}};
  for(const partial of [
    {...all,complete:false}, {...all,complete_session_count:1}, {...all,settled_session_count:1},
    {...all,observed_session_count:1,unobserved_session_count:1},
    {complete:true,launched_session_count:0,observed_session_count:0,settled_session_count:0,
      complete_session_count:0,unobserved_session_count:0}, {}
  ]){
    run.dsh_runtime.provider_usage.accounting_coverage=partial;c.renderProcess();
    assert.match(node('#execution-token-progress').textContent,/已观测下限；缺失用量未知/);
    assert.doesNotMatch(node('#execution-token-progress').textContent,/会话回执完整/);
  }
});

test('child failures and retry events are observational, not inferred run failure',()=>{
  const {c,node,run}=fixture();
  run.latest_runtime_failure={stage:'sample.plan',error_code:'structured_result_missing',message:'<script>unsafe</script>'};
  c.state.events=[{type:'dsh.child_execution_failed',occurred_at:'2026-01-01',payload:{stage:'sample.plan',error_code:'structured_result_missing'}},
    {type:'stage.recorded',payload:{stage:'evaluation',status:'completed'}}];
  c.renderProcess();
  assert.equal(node('#process-status-pill').textContent,'执行中');
  assert.match(node('#runtime-failure').textContent,/可能已恢复/);
  assert.match(node('#event-list').innerHTML,/structured_result_missing/);
  assert.doesNotMatch(node('#event-list').innerHTML,/运行已终止|<script>/);
});

test('sample summary is bound to both displayed run and candidate, and failed creation is empty',()=>{
  const {c,node,run}=fixture();
  c.state.candidateSampleSelection={run_id:'run:a',candidate_id:'candidate:a'};
  c.state.candidateSamplePage={rows:[{sample_id:'sample:a'}],total:9};
  c.renderProcess();assert.match(node('#sample-inference-count').textContent,/9 格已封存/);
  c.state.candidateSampleSelection.run_id='run:old';c.renderProcess();
  assert.match(node('#sample-inference-count').textContent,/尚未读取/);
  c.state.createStatus={state:'failed',runId:null,message:'preflight failed',displayFocus:true};
  // The shared display selector is authoritative for a new attempt.
  c.displayedRun=()=>null;c.createDisplayStatus=()=>c.state.createStatus;c.renderProcess();
  assert.match(node('#process-status-pill').textContent,/本次启动未成功/);
  assert.doesNotMatch(node('#sample-inference-list').innerHTML,/candidate:a|sample:a/);
  assert.match(node('#event-list').innerHTML,/暂无事件/);
});

test('paired changes, raw tool effects, and selected version remain separate scientific evidence',()=>{
  const {c,node,run}=fixture();
  run.evolution_evidence={search_version:{candidate_id:'seed',is_initial_seed:true},
    generation_decisions:[{generation:1,committed:true,same_cohort:true,selected_arm:'incumbent',arms:[
      {arm:'incumbent',absolute_score:-.2},{arm:'finalist_1',absolute_score:-.19,holdout_delta:.01,
        search_failures:['cell_regression'],joint_recheck:{status:'not_confirmed',repeated_delta:-.01}}]}]};
  run.adaptive_trajectories=[{generation:0,candidate_id:'candidate:a',batches:[{batch_index:1,origin_count:25,
    champion_score:.1,challenger_score:.2,score_delta:.1,minimum_score_delta:.005,
    comparison_decision:'champion_retained',comparison_reason:'probation_incomplete_paired_blocks',
    next_challenger_revision_id:'revision:next',next_challenger_operations:[{op:'set_bounded_parameter',name:'residual_scale',value:0}]}]}];
  c.renderProcess();
  assert.match(node('#generation-decision-list').innerHTML,/保留轮初版本/);
  assert.match(node('#generation-decision-list').innerHTML,/部分预测指标退化/);
  assert.match(node('#global-champion-card').innerHTML,/暂无认证证据/);
  assert.match(node('#adaptive-trajectory-table').innerHTML,/保留父版本/);
  assert.match(node('#adaptive-trajectory-table').innerHTML,/residual_scale · 0/);
  assert.doesNotMatch(node('#adaptive-trajectory-table').innerHTML,/已证明性能提升/);
});

test('candidate chart never joins different cohorts or uses maximum observed as incumbent',()=>{
  const {c,node,run}=fixture();
  run.trajectory=[{generation:1,candidate_id:'a',score:.7,incumbent_score:.5,evaluation_cohort_digest:'A'},
    {generation:2,candidate_id:'b',score:.6,incumbent_score:.6,evaluation_cohort_digest:'B'}];
  c.renderTrajectory();assert.doesNotMatch(node('#trajectory-chart').innerHTML,/<path/);
  assert.match(node('#trajectory-chart').innerHTML,/不能|不可直接纵向比较/);
  run.trajectory[1].evaluation_cohort_digest='A';c.renderTrajectory();
  assert.match(node('#trajectory-chart').innerHTML,/<path/);
  assert.equal(Number.isNaN(c.trajectoryIncumbentScore({best_observed_score:99})),true);
});

test('nine-cell sample table preserves real zero, unavailable outputs and scoring failure distinction',()=>{
  const {c}=fixture();
  const row=(extra={})=>c.normalizeCandidateSampleRow({sample_id:'s',target:'co2_concentration',horizon_hours:6,
    origin_timestamp:'2026-01-01T00:00:00Z',target_timestamp:'2026-01-01T06:00:00Z',observed:0,predicted:0,...extra});
  assert.equal(row().predicted,0);
  assert.equal(row({predicted:null}).model_prediction_available,false);
  const failed=row({predicted:-999,prediction_source:'scoring_fallback',status:'failed'});
  assert.equal(failed.model_prediction_available,false);
  assert.match(c.candidateSampleStatusDescriptor(failed).text,/失败/);
  assert.doesNotMatch(c.renderCandidateSampleRows([failed],0),/>-999</);
});

test('registered 1017 route remains selectable independently in both model roles', () => {
  const {c,node}=fixture();
  const id='pjlab/deepseek-v4.1-flash-1017';
  const model={id,roles:['propose','judge'],configured:true,credential_configured:true,
    directory_available:true,execution_available:true};
  for(const [selector,role] of [['#policy-model-id','propose'],['#judge-model-id','judge']]) {
    c.populateSelect(selector,[model],'missing',id,role);
    assert.equal(node(selector).value,id);
    assert.match(node(selector).innerHTML,/deepseek-v4\.1-flash-1017/);
    assert.doesNotMatch(node(selector).innerHTML,/ disabled/);
  }
  c.populateSelect('#judge-model-id',[{...model,roles:['propose']}],'missing',id,'judge');
  assert.match(node('#judge-model-id').innerHTML,/ disabled/);
  assert.match(node('#judge-model-id').innerHTML,/不支持此职责/);
});
