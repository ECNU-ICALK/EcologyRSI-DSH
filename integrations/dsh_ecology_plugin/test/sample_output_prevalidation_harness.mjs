#!/usr/bin/env node
// Real rc.2 + PiAiAdapter + SidecarClient over loopback. The local Host fixture
// tests native lifecycle/wire boundaries; Python authoritative semantics have
// separate service regressions. No production config or external calls.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {createHash} from 'node:crypto';
import {createServer} from 'node:http';
import {cp,mkdir,mkdtemp,readFile,realpath,rm,writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {sampleContextFixture,predictionResultFixture} from './fixtures/display_projection.mjs';
import {dshPackageJson,installManagedPatch} from '../../../scripts/install_dsh_ecology_runtime.mjs';
const root=fileURLToPath(new URL('../../../',import.meta.url)), evidence=process.argv[2]?path.resolve(process.argv[2]):null;
const hostPackage=await dshPackageJson(process.env.DSH_BIN||'dsh');assert.equal(JSON.parse(await readFile(hostPackage,'utf8')).version,'0.2.0-rc.2');
const requireHost=createRequire(hostPackage),yaml=requireHost('yaml'),importHost=name=>import(pathToFileURL(requireHost.resolve(name)).href);
const home=await realpath(await mkdtemp(path.join(os.tmpdir(),'ecology-prevalidation-')));
const nativeAttestations=[];
const report={schema_version:'ecologyrsi-dsh.prevalidation-harness/1',harness_version:'0.2.0-rc.2',remote_model_calls:0,cases:[],passed:false,host:'loopback controlled protocol fixture; authoritative Python service tested separately'};
let host,manager,runner,offEvents,state,serverError,jsonDigest,pythonHost,pythonLines,hostMetadata,attestationRoute;
const authoritative=process.env.ECOLOGY_PREVALIDATION_AUTHORITATIVE_HOST==='1';
if(authoritative){
 assert.ok(process.env.PYTHON_BIN,'PYTHON_BIN required for isolated authoritative Host');
 pythonHost=spawn(process.env.PYTHON_BIN,[path.join(root,'integrations/dsh_ecology_plugin/test/fixtures/prevalidation_host.py')],{cwd:root,env:{...process.env,PYTHONPATH:path.join(root,'src')},stdio:['pipe','pipe','pipe']});
 let pythonErrors='';pythonHost.stderr.on('data',chunk=>{pythonErrors+=chunk.toString();});
 pythonLines=createInterface({input:pythonHost.stdout})[Symbol.asyncIterator]();
 const line=await pythonLines.next();assert.equal(line.done,false,pythonErrors);hostMetadata=JSON.parse(line.value);
 report.host='production Python EvolutionHTTPServer + DshToolService + authenticated SidecarClient + actual rc.2';
}

const schemaVersion='ecologyrsi-dsh.prediction-prevalidation/1';
const textOf=m=>typeof m.content==='string'?m.content:(m.content||[]).filter(b=>b.type==='text').map(b=>b.text).join('');
const canonical=v=>Array.isArray(v)?`[${v.map(canonical).join(',')}]`:v&&typeof v==='object'?`{${Object.keys(v).sort().map(k=>`${JSON.stringify(k)}:${canonical(v[k])}`).join(',')}}`:JSON.stringify(v);
const respond=(res,status,body)=>{res.writeHead(status,{'content-type':'application/json'});res.end(JSON.stringify(body));};
const server=createServer(async(req,res)=>{try{
 if(req.url.startsWith('/api/ecology-agent-runtime/v1/')){assert.ok(attestationRoute);return await attestationRoute.handler(req,res);}
 let raw='';for await(const c of req)raw+=c;const body=raw?JSON.parse(raw):null,s=state;assert.ok(s);
 if(req.url.startsWith('/api/ecology-agent-sidecar/v1/')){
  assert.equal(req.headers.authorization,'Bearer local-only-prevalidation');
  if(req.url.endsWith('/session-usage'))return respond(res,200,{accepted:true});
  if(req.url.endsWith('/child-reservations')){s.reservations++;return respond(res,200,{accepted:true,admission_id:body.admission_id,timeout_ms:body.timeout_ms,launch:{reservation_id:`${s.name}-reservation`,launch_attempt:1},ledger_expected_revision:2});}
  if(req.url.endsWith('/child-failures')){s.failures.push({code:body.error_code,runtime_failure:body.runtime_failure});return respond(res,200,{accepted:true});}
  if(req.url.endsWith('/tools/ecology_execute_prediction_tool')){s.toolCalls++;assert.ok(['cm1','cm2'].includes(body.arguments.call_id));return respond(res,200,{...s.toolResult,call_id:body.arguments.call_id});}
  if(req.url.endsWith('/structured-results/validate')){
   assert.deepEqual(Object.keys(body).sort(),['admission_id','identity','output_schema_id','result_digest','structured']);
   assert.equal(body.identity.run_id,s.runId);assert.equal(body.identity.role,'sample-planner');assert.equal(body.identity.stage,'sample.plan');
   assert.equal(body.identity.child_reservation_id,`${s.name}-reservation`);assert.equal(body.identity.idempotency_key,s.binding.idempotency_key);assert.ok(body.identity.activation_lease_id);
   assert.equal(body.output_schema_id,'ecology-sample-predictions@2');assert.equal(body.structured.wave_digest,s.context.wave_digest);assert.equal(body.result_digest,jsonDigest(body.structured));
   s.validations.push({digest:body.result_digest,session_id:body.identity.session_id,identity:body.identity,structured:structuredClone(body.structured)});
   if(s.kind==='unavailable')return respond(res,503,{error_code:'structured_result_persistence_unavailable',error:'Synthetic unavailable'});
   if(s.kind==='authorization')return respond(res,403,{error_code:'dsh_tool_authorization_failed',error:'Synthetic denied'});
   if(s.kind==='closed')return respond(res,409,{error_code:'dsh_tool_admission_closed',error:'Synthetic closed'});
   if(s.kind==='malformed')return respond(res,200,{valid:true});
   if(s.kind==='wrong-digest')return respond(res,200,{schema_version:schemaVersion,valid:true,result_digest:'f'.repeat(64),diagnostics:[]});
   if(s.kind==='deadline'){await new Promise(resolve=>setTimeout(resolve,500));return respond(res,200,{schema_version:schemaVersion,valid:true,result_digest:body.result_digest,diagnostics:[]});}
   const invalid=body.structured.decisions.some(d=>d.method==='blend'&&d.evidence_call_ids.length<2);
   const diagnostics=invalid?body.structured.decisions.map((_,i)=>({code:'blend_requires_two_results',path:`value.decisions[${i}].evidence_call_ids`})) : [];
   return respond(res,200,{schema_version:schemaVersion,valid:!invalid,result_digest:body.result_digest,diagnostics});
  }
  assert.ok(req.url.endsWith('/structured-results'));
  if(s.kind==='persist-closed'){s.persistAttempts++;return respond(res,409,{error_code:'dsh_tool_admission_closed',error:'Closed after prevalidation'});}
  assert.equal(body.result_digest,jsonDigest(body.structured));assert.deepEqual(body.structured,s.validations.at(-1).structured);
  assert.ok(body.structured.decisions.every(d=>d.method===(s.predictionRepair&&!s.nativeInvalid?'blend':'model')));s.persisted.push(body);return respond(res,200,{accepted:true,result_digest:body.result_digest});
 }
 assert.equal(req.url,'/v1/chat/completions');const outsider=body.model==='outsider-model',calls=outsider?s.outsider:s.requests,index=calls.length;
 assert.ok(index<(outsider?1:s.predictionRepair?5:4),'No extra provider response or unbounded retry');
 if(!outsider&&index===0&&s.isolation)await runOutsider(s);
 const feedback=body.messages.filter(m=>m.role==='tool').map(textOf).filter(t=>t.includes(schemaVersion));
 if(outsider)assert.equal(feedback.length,0);
 if(!outsider&&index===3&&s.invalidAttempts&&!s.nativeInvalid){assert.equal(feedback.length,1);assert.ok(feedback[0].includes('value.decisions[8].evidence_call_ids'));s.feedbackReachedNextHttp=true;}
 const tool=outsider?'structured_output':index===0?'skill':index===1||(s.predictionRepair&&index===3)?'ecology_execute_prediction_tool':'structured_output';
 const outputNumber=calls.filter(call=>call.tool==='structured_output').length+1;
 const invalid=outsider||(tool==='structured_output'&&outputNumber<=s.invalidAttempts);
 const args=tool==='skill'?{name:'origin-vector-forecasting'}:tool==='ecology_execute_prediction_tool'?{tool_id:'ridge@1',call_id:index===3?'cm2':'cm1',wave_digest:s.context.wave_digest,parameters:index===3?{synthetic_variant:1}:{}}:invalid?s.invalidOutput:s.validOutput;
 calls.push({tool,invalid,feedback:feedback.length});
 res.writeHead(200,{'content-type':'text/event-stream'});const frame=(delta,finish,usage)=>'data: '+JSON.stringify({id:`local-${s.name}-${index}`,object:'chat.completion.chunk',created:1,model:body.model,choices:[{index:0,delta,finish_reason:finish}],...(usage?{usage}:{})})+'\n\n';
 res.end(frame({role:'assistant',tool_calls:[{index:0,id:`${outsider?'out':'main'}-${index}`,type:'function',function:{name:tool,arguments:JSON.stringify(args)}}]},null)+frame({},'tool_calls',{prompt_tokens:10,completion_tokens:5,total_tokens:15})+'data: [DONE]\n\n');
 }catch(error){serverError??=error;respond(res,400,{error:{message:'Local fixture assertion failed',type:'invalid_request_error'}});}});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
if(authoritative){pythonHost.stdin.write(JSON.stringify({native_url:`http://127.0.0.1:${server.address().port}`})+'\n');}
async function runOutsider(s){const run=await host.ctx.subagents.start('spawn',{parent:s.role.agent,label:'unrelated-reservation',prompt:[{type:'text',text:'Synthetic unrelated output.'}],outputSchema:s.schema,agentOptions:{provider:'local-fixture',model:'outsider-model',maxTokens:2048},signal:AbortSignal.timeout(10000)});try{const result=await(typeof run.result==='function'?run.result():run.result);assert.deepEqual(result.structured,s.invalidOutput);}finally{await run.dispose?.();}assert.equal(s.validations.length,0);}
try{
 process.env.DSH_HOME=home;process.env.DSH_TELEMETRY_DISABLED='1';process.env.ECOLOGY_PREVALIDATION_FIXTURE_KEY='isolated-placeholder';process.env.ECOLOGYRSI_DSH_RUNTIME_TOKEN='isolated-runtime';process.env.ECOLOGYRSI_SIDECAR_TOOL_TOKEN='isolated-sidecar';process.chdir(home);
 const {prepareProfile,runProfile}=await importHost('@deepseek-ai/dsh/profile-boot');const {createLaunchEnvironmentSnapshot}=await importHost('@deepseek-ai/dsh-launch-environment');prepareProfile('web');
 const pkg=path.join(home,'profiles/web/node_modules/@ecologyrsi/dsh-evolution-plugin');await mkdir(pkg,{recursive:true});for(const entry of ['package.json','lib','presets','schemas'])await cp(path.join(root,'integrations/dsh_ecology_plugin',entry),path.join(pkg,entry),{recursive:true});
 await installManagedPatch({dshHome:home,staticRoot:path.join(root,'plugins/ecology_evolution')});
 const overlay=path.join(home,'verify.patch.yml');await writeFile(overlay,yaml.stringify([{id:'webserver',config:{host:'127.0.0.1',port:0}},{id:'web-runtime',config:{openBrowser:false,printUrl:false}},{id:'session-title-llm',disabled:true},{id:'ecologyrsi-evolution',disabled:true},{id:'llm-pi-ai',config:{providers:{'local-fixture':{api:'openai-completions',baseURL:`http://127.0.0.1:${server.address().port}/v1`,apiKeyEnv:'ECOLOGY_PREVALIDATION_FIXTURE_KEY',models:[{id:'model',contextWindow:131072,maxTokens:32768},{id:'outsider-model',contextWindow:131072,maxTokens:32768}]}}}}]));
 host=await runProfile({profile:'web',patchFiles:[overlay],args:['--no-open'],environment:createLaunchEnvironmentSnapshot([{source:'process',values:process.env}])});
 const plugin=file=>import(pathToFileURL(path.join(pkg,'lib',file)).href);
 const {RoleAgentManager}=await plugin('runtime/agents.js');const native=await plugin('runtime/stage-runner.js');jsonDigest=native.jsonDigest;
 const {sampleOutputTransport}=await plugin('runtime/sample-output-transport.js'),{SidecarClient}=await plugin('sidecar/client.js'),{structuredFailureCode}=await plugin('runtime/structured-stage-errors.js');
 manager=new RoleAgentManager(host.ctx);const manifest=JSON.parse(await readFile(path.join(pkg,'presets/preset-manifest.json'),'utf8'));
 const realSidecar=new SidecarClient({origin:`http://127.0.0.1:${authoritative?hostMetadata.port:server.address().port}`,token:'local-only-prevalidation'});
 const sidecar=authoritative?{request:async(route,options)=>{
  const s=state,b=options.body;
  if(route.endsWith('/child-reservations'))s.reservations++;
  if(route.endsWith('/tools/ecology_execute_prediction_tool'))s.toolCalls++;
  if(route.endsWith('/structured-results/validate')&&s.validations.length===0){
   s.forgedIdentitiesRejected=[];
   for(const field of ['session_id','activation_lease_id','genome_digest','compiled_behavior_digest','phenotype_instance_digest']){
    const forged=structuredClone(b);forged.identity[field]=field.endsWith('_digest')?'e'.repeat(64):'forged-public-fixture';
    await assert.rejects(realSidecar.request(route,{...options,body:forged}),e=>e.code==='dsh_tool_authorization_failed');s.forgedIdentitiesRejected.push(field);
   }
  }
  if(route.endsWith('/structured-results/validate'))s.validations.push({digest:b.result_digest,session_id:b.identity.session_id,identity:structuredClone(b.identity),structured:structuredClone(b.structured)});
  if(route.endsWith('/structured-results')&&s.kind==='persist-revoked'){s.persistAttempts++;runner.childBindings.releaseChild(b.identity.session_id);}
  const result=await realSidecar.request(route,options);
  if(route.endsWith('/structured-results'))s.persisted.push(structuredClone(b));
  if(route.endsWith('/child-failures'))s.failures.push({code:b.error_code,runtime_failure:b.runtime_failure});
  return result;
 }}:realSidecar;
 runner=new native.NativeStageRunner(host.ctx,{roleAgents:manager,sidecar,runRegistry:{get:()=>({status:'running'})},structuredStageMaxAttempts:2,samplePlannerStageTimeoutMs:15000,structuredStageMinIntervalMs:0});host.ctx.provide('ecologyAgentTools',runner.bridge);
 const {registerRuntimeRoutes}=await plugin('runtime/routes.js');
 registerRuntimeRoutes({webServer:{register(route){attestationRoute=route;return()=>{};}}},{attestChildActivation:body=>{
  const result=runner.attestChildActivation(body),session=host.ctx.sessions.get(body.identity?.session_id);
  nativeAttestations.push({run_id:body.identity?.run_id,valid:result.valid,turn_ended:session?.snapshotEvents?.().some(e=>e.type==='turn/end')||false});return result;
 }},{runtimeToken:'local-only-native-attestation'});
 offEvents=host.ctx.on('session/event',(session,event)=>{if(state?.role&&session.header?.parentSession===state.role.sessionId)state.events.push({session_id:session.id,event});},{global:true});
 const repairSpecs=[{name:'T-blend-repair-extra-tool',invalidAttempts:1,predictionRepair:true},{name:'blend-only-repair-extra-tool',invalidAttempts:1,predictionRepair:true,blendOnly:true},{name:'two-blend-invalid-with-tool',invalidAttempts:2,predictionRepair:true},{name:'ordinary-schema-extra-tool-denied',invalidAttempts:1,predictionRepair:true,nativeInvalid:'wave',expectedFailure:'structured_result_persist_tool_evidence'}];
 const specs=[{name:'S-nine-blend-single-ref-corrected',invalidAttempts:1,isolation:true},{name:'two-invalid-no-third',invalidAttempts:2},{name:'wrong-wave-native-schema',invalidAttempts:1,nativeInvalid:'wave'},{name:'array-native-schema',invalidAttempts:1,nativeInvalid:'array'},...['unavailable','authorization','closed','malformed','wrong-digest','deadline','persist-closed'].map(kind=>({name:kind,kind,invalidAttempts:0}))];
 for(const spec of authoritative?[...specs.slice(0,2),{name:'persist-revoked',kind:'persist-revoked',invalidAttempts:0},...repairSpecs]:[...specs,...repairSpecs]){
  state={...spec,runId:`local-${spec.name}`,requests:[],outsider:[],events:[],reservations:0,validations:[],toolCalls:0,persisted:[],persistAttempts:0,failures:[],feedbackReachedNextHttp:false};const s=state;
  s.context=sampleContextFixture();if(s.blendOnly)s.context.context.prediction_contract.allowed_methods=['blend'];s.toolResult=predictionResultFixture(s.context);
  s.role=await manager.createRoleAgent({run_id:s.runId,role:'sample-planner',preset_id:manifest.presets.find(p=>p.role==='sample-planner').preset_id,model:'local-fixture/model',cwd:home});
  const transport=sampleOutputTransport('sample.plan',native.prepareStageOutputSchema('sample.plan',await runner.schema(native.STAGES['sample.plan'].file),s.context));s.schema=transport.schema;
  s.validOutput={decisions:s.context.samples.map((sample,i)=>({sample_id:sample.sample_id,predicted:10+i,confidence:.8,method:'model',reason_code:'agent_model',evidence_call_ids:['cm1']}))};s.invalidOutput=structuredClone(s.validOutput);if(s.predictionRepair&&!s.nativeInvalid)s.validOutput.decisions.forEach(d=>{d.method='blend';d.reason_code='agent_blend';d.evidence_call_ids=['cm1','cm2'];});s.invalidOutput.decisions.forEach(d=>{d.method='blend';d.reason_code='agent_blend';});if(s.nativeInvalid==='wave')s.invalidOutput={...s.validOutput,wave_digest:'e'.repeat(64)};if(s.nativeInvalid==='array')s.invalidOutput=[];
  s.binding={run_id:s.runId,stage:'sample.plan',run_state_revision:1,stage_attempt:1,ledger_expected_revision:1,admission_id:`admission-${s.name}`,idempotency_key:`idempotency-${s.name}`,request:{role:'sample-planner',output_schema_id:'ecology-sample-predictions@2',max_tokens:2048,context:s.context,context_canonical_json:canonical(s.context),context_digest:jsonDigest(s.context),identity_digests:{genome_digest:'a'.repeat(64),compiled_behavior_digest:'b'.repeat(64),phenotype_instance_digest:'c'.repeat(64)}}};
  if(authoritative){s.binding.admission_id=hostMetadata.cases[s.name].admission_id;}
  runner.samplePlannerStageTimeoutMs=s.kind==='deadline'?250:15000;
  let result,caught;try{result=await runner.run(s.binding);}catch(error){caught=error;}await new Promise(resolve=>setTimeout(resolve,25));
  assert.equal(serverError,undefined,serverError?.message);assert.equal(s.reservations,1);assert.equal(s.toolCalls,s.predictionRepair?2:1);
  const ids=new Set(s.events.filter(({event:e})=>e.type==='request/header'&&e.data?.header?.config?.model==='model').map(e=>e.session_id));assert.equal(ids.size,1);
  const events=s.events.filter(e=>ids.has(e.session_id)).map(e=>e.event),invalid=events.filter(e=>e.type==='tool/result'&&(e.data?.error?.code||e.data?.error?.info?.code)==='INVALID_ARGS');
  const failure=caught?structuredFailureCode(caught)||caught.code:null;
  if(s.expectedFailure){assert.equal(failure,s.expectedFailure);assert.equal(s.persisted.length,0);assert.equal(s.requests.length,5);assert.equal(s.validations.length,1);}
  else if(s.invalidAttempts===1){assert.equal(caught,undefined,caught?.message);assert.deepEqual(result.structured,transport.bind(s.validOutput));assert.equal(s.persisted.length,1);
   if(authoritative){
    const accepted=s.persisted[0];const parent=s.role.sessionId;
    const response=await fetch(`http://127.0.0.1:${server.address().port}/api/ecology-agent-runtime/v1/child-activations/attest`,{method:'POST',headers:{'authorization':'Bearer local-only-native-attestation','content-type':'application/json'},body:JSON.stringify({identity:accepted.identity,parent_session_id:parent})});
    assert.equal((await response.json()).valid,false,'Stage finally must release identity');
    assert.ok(nativeAttestations.some(x=>x.run_id===s.runId&&x.valid&&x.turn_ended),'Final persist attestation must work after native turn/end');
    const replay=await realSidecar.request('/api/ecology-agent-sidecar/v1/structured-results',{body:accepted});assert.equal(replay.accepted,true);s.exactReplayAfterRelease=true;
   }
   assert.equal(s.validations.length,s.nativeInvalid?1:2);assert.equal(s.requests.length,s.predictionRepair?5:4);if(!s.nativeInvalid)assert.ok(s.feedbackReachedNextHttp);}
  else if(s.invalidAttempts===2){assert.equal(failure,'structured_output_attempt_limit_exceeded');assert.equal(s.validations.length,2);assert.equal(s.requests.length,s.predictionRepair?5:4);assert.equal(s.persisted.length,0);assert.ok(s.feedbackReachedNextHttp);}
  else{assert.ok(caught);assert.equal(s.validations.length,1);assert.equal(s.requests.length,3);assert.equal(s.persisted.length,0);assert.equal(invalid.length,0);if(s.kind==='deadline')assert.ok(['structured_role_operational_timeout','provider_queue_timeout'].includes(failure));if(['persist-closed','persist-revoked'].includes(s.kind))assert.equal(s.persistAttempts,1);if(s.kind==='persist-revoked')assert.equal(failure,'structured_result_persist_authorization');}
  assert.equal(invalid.length,s.invalidAttempts);report.cases.push({name:s.name,provider_requests:s.requests.length,structured_requests:s.requests.filter(x=>x.tool==='structured_output').length,host_validation_requests:s.validations.length,prediction_tool_requests:s.toolCalls,invalid_args:invalid.length,child_count:ids.size,reservations:s.reservations,persisted:s.persisted.length,persist_attempts:s.persistAttempts,failure_code:failure,feedback_next_http:s.feedbackReachedNextHttp,other_reservation_requests:s.outsider.length,forged_identities_rejected:s.forgedIdentitiesRejected||[],exact_idempotent_replay_after_release:s.exactReplayAfterRelease||false});
  await manager.quiesceRun(s.runId,{dispose:true});state=null;
 }
 if(authoritative){pythonHost.stdin.write('finish\n');const line=await pythonLines.next();assert.equal(line.done,false);report.authoritative_host=JSON.parse(line.value);report.native_attestations=nativeAttestations;
  assert.equal(report.authoritative_host.validations.length,12);assert.ok(report.authoritative_host.validations.every(v=>v.zero_state_mutation));
  const success=report.authoritative_host.cases['S-nine-blend-single-ref-corrected'],stop=report.authoritative_host.cases['two-invalid-no-third'];
  assert.equal(success.events.DshStructuredResultAccepted,1);assert.equal(stop.events.DshStructuredResultAccepted||0,0);
  for(const name of ['T-blend-repair-extra-tool','blend-only-repair-extra-tool']){assert.equal(report.authoritative_host.cases[name].events.DshStructuredResultAccepted,1);assert.equal(report.authoritative_host.cases[name].tool_executions,2);assert.equal(report.authoritative_host.cases[name].tool_budget_spent,2);}
  assert.equal(report.authoritative_host.cases['two-blend-invalid-with-tool'].events.DshStructuredResultAccepted||0,0);assert.equal(report.authoritative_host.cases['ordinary-schema-extra-tool-denied'].events.DshStructuredResultAccepted||0,0);
  assert.equal(success.tool_executions,1);assert.equal(stop.tool_executions,1);assert.equal(success.tool_budget_spent,1);assert.equal(stop.tool_budget_spent,1);
  assert.equal(report.authoritative_host.cases['persist-revoked'].events.DshStructuredResultAccepted||0,0);
  assert.ok(nativeAttestations.some(x=>x.run_id==='local-persist-revoked'&&!x.valid&&x.turn_ended));
 }
 report.source_sha256={};for(const file of ['sample-output-prevalidation.js','stage-runner.js','child-bindings.js','controller.js','routes.js'])report.source_sha256[file]=createHash('sha256').update(await readFile(path.join(pkg,'lib/runtime',file))).digest('hex');report.passed=true;console.log(JSON.stringify(report));
}finally{pythonHost?.stdin.end();offEvents?.();await manager?.dispose();await host?.shutdown.shutdown(0);server.closeAllConnections();await new Promise(resolve=>server.close(resolve));process.chdir(root);await rm(home,{recursive:true,force:true});report.isolated_home_removed=true;if(evidence)await writeFile(evidence,JSON.stringify(report,null,2)+'\n');}
