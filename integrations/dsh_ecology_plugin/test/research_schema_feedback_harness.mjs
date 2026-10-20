#!/usr/bin/env node
// Explicit real rc.2 + PiAiAdapter loopback regression. Public synthetic inputs;
// no production settings, scientific samples or remote model calls.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createServer} from 'node:http';
import {cp,mkdir,mkdtemp,readFile,realpath,rm,writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {dshPackageJson,installManagedPatch} from '../../../scripts/install_dsh_ecology_runtime.mjs';
const root=fileURLToPath(new URL('../../../',import.meta.url));
const evidence=process.argv[2]?path.resolve(process.argv[2]):null;
const hostPackage=await dshPackageJson(process.env.DSH_BIN||'dsh');assert.ok(hostPackage);
assert.equal(JSON.parse(await readFile(hostPackage,'utf8')).version,'0.2.0-rc.2');
const requireHost=createRequire(hostPackage),yaml=requireHost('yaml');
const importHost=name=>import(pathToFileURL(requireHost.resolve(name)).href);
const home=await realpath(await mkdtemp(path.join(os.tmpdir(),'ecology-research-feedback-')));
const report={schema_version:'ecologyrsi-dsh.research-feedback-harness/1',harness_version:'0.2.0-rc.2',remote_model_calls:0,transport:'real_Harness_NativeStageRunner_PiAiAdapter_loopback_HTTP',cases:[],passed:false};
let host,manager,offEvents,activeCase,serverError;
const textOf=m=>typeof m.content==='string'?m.content:(m.content||[]).filter(b=>b.type==='text').map(b=>b.text).join('');
const parse=text=>{try{return JSON.parse(text);}catch{return null;}};
const marker='ecologyrsi-dsh.research-schema-feedback/1';
function canonical(v){return Array.isArray(v)?`[${v.map(canonical).join(',')}]`:v&&typeof v==='object'?`{${Object.keys(v).sort().map(k=>`${JSON.stringify(k)}:${canonical(v[k])}`).join(',')}}`:JSON.stringify(v);}
function fixture(schema){if('const'in schema)return schema.const;if(schema.oneOf)return fixture(schema.oneOf[0]);if(schema.enum)return schema.enum[0];if(schema.type==='object')return Object.fromEntries((schema.required||[]).map(k=>[k,fixture(schema.properties[k])]));if(schema.type==='array')return Array.from({length:schema.minItems||1},()=>fixture(schema.items));if(schema.type==='string')return'public synthetic fixture';if(schema.type==='boolean')return false;if(schema.type==='null')return null;return schema.minimum||0;}
async function resultOf(run){return typeof run.result==='function'?run.result():run.result;}
const server=createServer(async(req,res)=>{try{
 assert.equal(req.url,'/v1/chat/completions');let raw='';for await(const c of req)raw+=c;const body=JSON.parse(raw);const state=activeCase;assert.ok(state);
 const outsider=body.model==='outsider-model',calls=outsider?state.outsider.requests:state.requests,index=calls.length;
 assert.ok(index<(outsider?2:4),'Unbounded fixture request loop');
 if(!outsider){
  const prompt=body.messages.filter(m=>m.role==='user').map(textOf).map(parse).find(p=>p?.stage==='generation.research-synthesis');assert.ok(prompt);
  assert.ok(prompt.output_contract.nested_required_fields['candidate_directions[*]'].includes('title'));
  if(index===0&&state.checkIsolation)await runOutsider(state);
 }
 const errors=body.messages.filter(m=>m.role==='tool').map(textOf).filter(t=>t.includes('value.candidate_directions[0]'));
 if(outsider&&index===1){assert.equal(errors.length,1);assert.match(errors[0],/oneOf/,'Fixture must preserve the original ambiguous oneOf failure');assert.ok(!errors[0].includes(marker),'Unrelated child must not receive scoped research feedback');state.outsider.originalError=errors[0];}
 if(!outsider&&index===2){
  assert.ok(errors.length>=1);const error=errors.at(-1);assert.ok(error.includes(marker),'Next real HTTP request must include appended schema diagnostics');
  for(let i=0;i<4;i++)assert.ok(error.includes(`value.candidate_directions[${i}].title`));
  if(state.outsider.originalError)assert.ok(error.includes(state.outsider.originalError),'Original validator error must be retained verbatim');
  state.feedbackReachedNextHttp=true;
 }
 const tool=outsider?'structured_output':index===0?'skill':'structured_output';
 const invalid=tool==='structured_output'&&(outsider?index===0:index<=state.invalidAttempts);
 const args=tool==='skill'?{name:'autonomous-ecology-research'}:invalid?state.invalidOutput:state.validOutput;
 calls.push({request:index+1,tool,invalid_schema:invalid,feedback_in_tool_message:errors.some(t=>t.includes(marker))});
 res.writeHead(200,{'content-type':'text/event-stream'});
 const frame=(delta,finish,usage)=>'data: '+JSON.stringify({id:`local-feedback-${state.name}-${outsider?'outside':'main'}-${index}`,object:'chat.completion.chunk',created:1,model:body.model,choices:[{index:0,delta,finish_reason:finish}],...(usage?{usage}:{})})+'\n\n';
 res.end(frame({role:'assistant',tool_calls:[{index:0,id:`${outsider?'outside':'main'}-${index}`,type:'function',function:{name:tool,arguments:JSON.stringify(args)}}]},null)+frame({},'tool_calls',{prompt_tokens:10,completion_tokens:5,total_tokens:15})+'data: [DONE]\n\n');
 }catch(e){serverError??=e;res.writeHead(400,{'content-type':'application/json'});res.end(JSON.stringify({error:{message:'local fixture assertion failed',type:'invalid_request_error'}}));}});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
async function runOutsider(state){
 const run=await host.ctx.subagents.start('spawn',{parent:state.role.agent,label:'unrelated-reservation',prompt:[{type:'text',text:'Public unrelated fixture: submit the schema result.'}],outputSchema:state.schema,agentOptions:{provider:'local-feedback',model:'outsider-model',maxTokens:2048},signal:AbortSignal.timeout(10000)});
 try{await resultOf(run);}finally{await run.dispose?.();}
 assert.equal(state.outsider.requests.length,2);assert.ok(state.outsider.originalError);
}
try{
 process.env.DSH_HOME=home;process.env.DSH_TELEMETRY_DISABLED='1';process.env.ECOLOGY_FEEDBACK_FIXTURE_KEY='isolated-placeholder';process.env.ECOLOGYRSI_DSH_RUNTIME_TOKEN='local-feedback-runtime';process.env.ECOLOGYRSI_SIDECAR_TOOL_TOKEN='local-feedback-sidecar';process.chdir(home);
 const {prepareProfile,runProfile}=await importHost('@deepseek-ai/dsh/profile-boot');const {createLaunchEnvironmentSnapshot}=await importHost('@deepseek-ai/dsh-launch-environment');prepareProfile('web');
 const pkg=path.join(home,'profiles/web/node_modules/@ecologyrsi/dsh-evolution-plugin');await mkdir(pkg,{recursive:true});for(const entry of ['package.json','lib','presets','schemas'])await cp(path.join(root,'integrations/dsh_ecology_plugin',entry),path.join(pkg,entry),{recursive:true});
 await installManagedPatch({dshHome:home,staticRoot:path.join(root,'plugins/ecology_evolution')});
 const overlay=path.join(home,'verify.patch.yml');await writeFile(overlay,yaml.stringify([{id:'webserver',config:{host:'127.0.0.1',port:0}},{id:'web-runtime',config:{openBrowser:false,printUrl:false}},{id:'session-title-llm',disabled:true},{id:'ecologyrsi-evolution',disabled:true},{id:'llm-pi-ai',config:{providers:{'local-feedback':{api:'openai-completions',baseURL:`http://127.0.0.1:${server.address().port}/v1`,apiKeyEnv:'ECOLOGY_FEEDBACK_FIXTURE_KEY',models:[{id:'model',contextWindow:131072,maxTokens:32768},{id:'outsider-model',contextWindow:131072,maxTokens:32768}]}}}}]));
 host=await runProfile({profile:'web',patchFiles:[overlay],args:['--no-open'],environment:createLaunchEnvironmentSnapshot([{source:'process',values:process.env}])});
 const plugin=file=>import(pathToFileURL(path.join(pkg,'lib/runtime',file)).href);
 const {RoleAgentManager}=await plugin('agents.js');const {NativeStageRunner,STAGES,prepareStageOutputSchema,jsonDigest}=await plugin('stage-runner.js');const {researchOutputTransport}=await plugin('research-contract.js');const {RESEARCH_EXECUTION_POLICY}=await plugin('research-execution-policy.js');const {structuredFailureCode}=await plugin('structured-stage-errors.js');
 manager=new RoleAgentManager(host.ctx);const manifest=JSON.parse(await readFile(path.join(pkg,'presets/preset-manifest.json'),'utf8'));
 host.ctx.on('tools/post-execute',async(exec,result,next)=>{const decision=await next();if(exec.name==='structured_output'){report.hook_diagnostics??=[];report.hook_diagnostics.push({name:exec.name,argument_type:typeof exec.arguments,argument_keys:Object.keys(exec.arguments||{}),is_error:result.isError,error:result.error,decision_kind:decision.kind,decision_has_value:Object.hasOwn(decision,'value'),decision_has_feedback:JSON.stringify(decision.content||[]).includes(marker),result_has_feedback:JSON.stringify(result.content||[]).includes(marker),signal_aborted:exec.signal?.aborted,session_id:exec.agent?.session?.id,parent_session:exec.agent?.session?.header?.parentSession,expected_parent:activeCase?.role?.sessionId,descriptor:exec.agent?.session?.snapshotEvents?.().findLast(e=>e.type==='subagent/descriptor')?.data});}return decision;},{global:true});
 offEvents=host.ctx.on('session/event',(session,event)=>{if(activeCase?.role&&session.header?.parentSession===activeCase.role.sessionId)activeCase.events.push({session_id:session.id,event});},{global:true});
 for(const spec of [{name:'missing-title-repaired',invalidAttempts:1,checkIsolation:true},{name:'two-invalid-title-stop',invalidAttempts:2,checkIsolation:false}]){
  const stage='generation.research-synthesis',contract=STAGES[stage],runId=`local-feedback-${spec.name}`;
  const state=activeCase={...spec,role:null,requests:[],outsider:{requests:[]},events:[],reservations:[],persisted:[],failures:[],feedbackReachedNextHttp:false};
  const sidecar={request:async(route,options)=>{
   if(route.endsWith('/session-usage'))return{accepted:true};
   if(route.endsWith('/child-reservations')){const launch={reservation_id:`${state.name}-reservation`,launch_attempt:1};state.reservations.push(launch);return{accepted:true,admission_id:options.body.admission_id,timeout_ms:options.body.timeout_ms,launch,ledger_expected_revision:2};}
   if(route.endsWith('/child-failures')){state.failures.push(options.body);return{accepted:true};}
   assert.ok(route.endsWith('/structured-results'));state.persisted.push(options.body);return{accepted:true,result_digest:options.body.result_digest};
  }};
  const runner=new NativeStageRunner(host.ctx,{roleAgents:manager,sidecar,runRegistry:{get:()=>({status:'running'})},structuredStageMaxAttempts:2,researchStageTimeoutMs:15000,structuredStageTimeoutMs:15000,structuredStageMinIntervalMs:0});
  if(!host.ctx.ecologyAgentTools)host.ctx.provide('ecologyAgentTools',runner.bridge);
  state.role=await manager.createRoleAgent({run_id:runId,role:contract.role,preset_id:manifest.presets.find(p=>p.role===contract.role).preset_id,model:'local-feedback/model',cwd:home});
  const context={research_execution_policy:{...RESEARCH_EXECUTION_POLICY},knowledge_snapshot:{evidence_catalog:[{knowledge_id:'fixture:public-paper'}]},required_candidate_direction_count:4,synthesis_contract:{mutation_directions_by_axis:{scientific_parameter:['increase','decrease'],registered_predictor:['select'],instruction_profile:['select'],instruction_parameter:['increase','decrease']},allowed_mutation_targets:{scientific_parameter:['ridge_alpha'],registered_predictor:['ridge@1'],instruction_profile:['balanced@1'],instruction_parameter:['confidence_threshold']}}};
  const transport=researchOutputTransport(stage,prepareStageOutputSchema(stage,await runner.schema(contract.file),context));state.schema=transport.schema;state.validOutput=fixture(transport.schema);state.validOutput.candidate_directions=Array.from({length:4},(_,i)=>({...state.validOutput.candidate_directions[0],direction_id:`fixture-direction-${i+1}`}));assert.ok(state.validOutput.candidate_directions.every(v=>typeof v.title==='string'));state.invalidOutput=structuredClone(state.validOutput);for(const d of state.invalidOutput.candidate_directions)delete d.title;
  const binding={run_id:runId,stage,run_state_revision:1,stage_attempt:1,ledger_expected_revision:1,admission_id:`admission-${spec.name}`,idempotency_key:`idempotency-${spec.name}`,request:{role:contract.role,output_schema_id:contract.schema,max_tokens:16384,context,context_canonical_json:canonical(context),context_digest:jsonDigest(context),identity_digests:{genome_digest:'a'.repeat(64),compiled_behavior_digest:'b'.repeat(64),phenotype_instance_digest:'c'.repeat(64)}}};
  let result,caught;try{result=await runner.run(binding);}catch(e){caught=e;}await new Promise(resolve=>setTimeout(resolve,25));
  assert.equal(serverError,undefined,serverError?.message);assert.equal(state.requests.length,3,'Only Skill and two structured provider requests are allowed');assert.ok(state.feedbackReachedNextHttp);assert.equal(state.reservations.length,1,'No extra child repair');
  const primaryIds=new Set(state.events.filter(({event:e})=>e.type==='request/header'&&e.data?.header?.config?.model==='model').map(e=>e.session_id));assert.equal(primaryIds.size,1);
  const ev=state.events.filter(e=>primaryIds.has(e.session_id)).map(e=>e.event);const errors=ev.filter(e=>e.type==='tool/result'&&e.data?.error?.code==='INVALID_ARGS');assert.equal(errors.length,spec.invalidAttempts);
  for(const e of errors){assert.equal(e.data.error.name,'ToolArgsError');assert.equal(e.data.message.isError,true);assert.ok(e.data.message.content.some(b=>b.type==='text'&&b.text.includes(marker)));}
  const failure=caught?structuredFailureCode(caught)||caught.code:null;
  if(spec.invalidAttempts===1){assert.equal(caught,undefined,caught?.message);assert.deepEqual(result.structured,transport.bind(state.validOutput));assert.equal(state.persisted.length,1);assert.equal(state.persisted[0].skill_invocation_evidence.order_verified,true);assert.equal(state.failures.length,0);}else{assert.equal(failure,'structured_output_attempt_limit_exceeded');assert.equal(state.persisted.length,0);assert.equal(state.failures.length,1);}
  report.cases.push({name:spec.name,primary_provider_requests:state.requests,structured_provider_requests:2,invalid_args:errors.length,feedback_reached_next_http:true,original_error_code_preserved:true,same_parent_other_label_requests:state.outsider.requests.length,same_parent_other_label_feedback_absent:spec.checkIsolation?state.outsider.requests.every(r=>!r.feedback_in_tool_message):null,original_error_text_preserved:spec.checkIsolation?true:null,primary_child_count:primaryIds.size,reservations:state.reservations.length,persisted:state.persisted.length,failure_code:failure});
  await manager.quiesceRun(runId,{dispose:true});activeCase=null;
 }
 report.source_sha256={};for(const file of ['research-schema-feedback.js','structured-roles.js','stage-runner.js'])report.source_sha256[file]=createHash('sha256').update(await readFile(path.join(pkg,'lib/runtime',file))).digest('hex');report.passed=true;
 console.log(JSON.stringify({passed:true,cases:report.cases,remote_model_calls:0}));
}finally{
 offEvents?.();await manager?.dispose();await host?.shutdown.shutdown(0);server.closeAllConnections();await new Promise(resolve=>server.close(resolve));process.chdir(root);await rm(home,{recursive:true,force:true});report.isolated_home_removed=true;if(evidence)await writeFile(evidence,JSON.stringify(report,null,2)+'\n');
}
