#!/usr/bin/env python3
"""Exercise real native Agent planning, optional tools, criticism and bounded retry.

Run against an isolated DSH runtime whose backendOrigin points to backend-port.
Uses one complete causal origin from verified AGC data and two independent
execution scopes. Writes an isolated diagnostic ledger; cannot promote a model.
The runtime must have its provider credentials already configured.
"""
from pathlib import Path
import os,json,threading,time,math,sys
import argparse
from uuid import uuid4
parser=argparse.ArgumentParser(description="Bounded real-data Agent prediction protocol canary; never a scientific promotion test")
parser.add_argument('--runtime-url',required=True)
parser.add_argument('--backend-port',type=int,default=8878)
parser.add_argument('--data-root',type=Path,required=True)
parser.add_argument('--episode',default='agc_cucumber_2018:AiCU')
parser.add_argument('--model',required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
work=args.output_dir.resolve();work.mkdir(parents=True,exist_ok=True)
os.environ['ECOLOGYRSI_DSH_RUNTIME_URL']=args.runtime_url
if not os.environ.get('ECOLOGYRSI_DSH_RUNTIME_TOKEN') or not os.environ.get('ECOLOGYRSI_SIDECAR_TOOL_TOKEN'):
 raise SystemExit('Supply the existing runtime and sidecar credentials through environment variables')
from ecologyrsi_dsh.api.handler import EvolutionHTTPServer
from ecologyrsi_dsh.integrations.dsh_native_runtime import DshNativeAgentRuntimeClient
from ecologyrsi_dsh.integrations.dsh_structured_roles import DshStructuredRoleRuntime
from ecologyrsi_dsh.evaluators.dsh_sample_adapter import DshSampleCollaborationAdapter
from ecologyrsi_dsh.evaluators.agent_model_tools import AgentModelTools
from ecologyrsi_dsh.evaluators.sample_execution import SamplePredictionRequest, _validated_result, _host_critic_result
from ecologyrsi_dsh.evaluators.greenhouse_prediction import fit_predict_exogenous_ridge, ExogenousRidgeConfig
from ecologyrsi_dsh.data.registry import DatasetRegistry
server=EvolutionHTTPServer(('127.0.0.1',args.backend_port),work/'probe.sqlite3')
threading.Thread(target=server.serve_forever,daemon=True).start()
run_id='run:agent-policy-probe:'+uuid4().hex
server.ledger.append(run_id,'RunCreated',{'test':True})
client=DshNativeAgentRuntimeClient(args.runtime_url,token=os.environ['ECOLOGYRSI_DSH_RUNTIME_TOKEN'],stage_timeout=240)
identity={'run_id':run_id,'run_state_revision':1,'ledger_expected_revision':server.ledger.latest_seq(),'stage_attempt':1,'idempotency_key':run_id+':create'}
print('Starting isolated DSH role hosts',flush=True)
client.create_run({**identity,'binding':{'initial_run_status':'running','strategy_model_id':args.model,'review_model_id':args.model}})
series=DatasetRegistry(data_root=args.data_root).series('agc_cucumber_2018', episode_id=args.episode);targets=('air_temperature','relative_humidity','co2_concentration');horizons=(1,6,24)
bank=AgentModelTools(series,targets=targets,horizons=horizons)
prepared=fit_predict_exogenous_ridge(series,targets=targets,horizons=horizons,config=ExogenousRidgeConfig(6,.1,.5),evaluation_history_steps=12,defer_prediction_partitions=('training_feedback',))
feedback=[row for row in prepared['prediction_rows'] if row['partition']=='training_feedback']
from collections import Counter
counts=Counter(row['origin_timestamp'] for row in feedback)
origin=min(time for time,n in counts.items() if n==9)
rows=[row for row in feedback if row['origin_timestamp']==origin]
bounds={'air_temperature':(-20,80,'degC'),'relative_humidity':(0,100,'%'),'co2_concentration':(0,5000,'ppm')}
requests=[]
for i,row in enumerate(rows):
 lo,hi,unit=bounds[row['target']]
 requests.append(SamplePredictionRequest(sample_id=f'origin-{origin}:{i}',candidate_id='candidate:agent-probe',dataset_digest=series.digest,partition='training_feedback',target=row['target'],unit=unit,horizon_hours=row['horizon_hours'],origin_timestamp=origin,target_timestamp=row['timestamp'],baseline=row['baseline'],proposed_prediction=None,minimum=lo,maximum=hi,algorithm_id='greenhouse-exogenous-ridge',algorithm_version='1',label_free_context=row['label_free_context']))
adapter=DshSampleCollaborationAdapter(run_id=run_id,
 runtime_provider=lambda:DshStructuredRoleRuntime(client,admission=server.dsh_tools),
 revision_provider=lambda _:{'run_state_revision':server.ledger.latest_seq(),'ledger_expected_revision':server.ledger.latest_seq()},
 identity_digests={'genome_digest':'a'*64,'compiled_behavior_digest':'b'*64,'phenotype_instance_digest':'c'*64},
 strategy_model_id=args.model,review_model_id=args.model,
 forecast_bundle_tool=lambda r:bank.execute(r,'greenhouse-exogenous-ridge@1',{}),
 prediction_tool_binder=server.dsh_tools.bind_prediction_tool,
 prediction_tool_catalog=bank.catalog(),prediction_tool_executor=bank.execute,
 sample_reflection_policy='candidate_aggregate_post_score@1',
 remote_critic_policy={'version':'always@1'},
 operation_max_tokens={'sample.planner':8192,'sample.repair':8192,'sample.critic':8192})
plan=adapter.plan_batch({'run_id':run_id,'candidate_id':'candidate:agent-probe','algorithm_id':'greenhouse-exogenous-ridge','algorithm_version':'1',
 'candidate_agent_profile':{'schema_version':'ecologyrsi-dsh.candidate-agent-profile/1','role':'sample-planner','skill_name':'origin-vector-forecasting','instruction_parameters':{'confidence_threshold':.7}}})
print('Running one real Agent origin with nine scoring cells',flush=True)
started=time.monotonic()
try:
 reports=[]
 for replica in range(2):
  replica_plan=adapter.plan_batch({**plan['decision_context'], 'evaluation_scope': {'inference_replica':replica, 'cohort_digest':series.digest}})
  replica_start=time.monotonic()
  from ecologyrsi_dsh.evaluators.sample_execution import _failure_feedback_from_exception, classify_sample_failure
  outcomes=[None]*len(requests)
  plans=[dict(replica_plan) for _ in requests]
  pending=list(range(len(requests)))
  for attempt in range(1,4):
   if not pending: break
   current=adapter.predict_samples([requests[i] for i in pending],[plans[i] for i in pending],attempts=[attempt]*len(pending))
   next_pending=[]
   for i, outcome in zip(pending,current):
    outcomes[i]=outcome
    if outcome.error is not None:
     feedback,_,_=_failure_feedback_from_exception(outcome.error,attempt=attempt,failure=classify_sample_failure(outcome.error))
     plans[i]={**replica_plan,'sample_retry_feedback':[*plans[i].get('sample_retry_feedback',[]),feedback]}
     next_pending.append(i)
   pending=next_pending
   print(json.dumps({'replica':replica,'attempt':attempt,'pending_cells':len(pending)}),flush=True)
  results=[]
  for request,row,outcome in zip(requests,rows,outcomes):
   if outcome.error is not None:raise outcome.error
   final=_host_critic_result(_validated_result(outcome.result),request)
   results.append({'sample_id':request.sample_id,'target':request.target,'horizon_hours':request.horizon_hours,'predicted':final['predicted'],'observed':row['observed'],'baseline':row['baseline'],'agent_trace':final['agent_decisions'],'tool_trace':final['tool_calls']})
  reports.append({'replica':replica,'seconds':time.monotonic()-replica_start,'results':results})
  print(json.dumps({'replica_completed':replica,'cells':len(results),'seconds':reports[-1]['seconds']}),flush=True)
 events=server.ledger.events(run_id)
 report={'passed':True,'evidence':'real AGC AiCU causal origin, real DSH children and provider calls; engineering protocol acceptance, not scientific certification','seconds':time.monotonic()-started,'origins':1,'replicas':reports,'cells':sum(len(r['results']) for r in reports),'accepted_stages':[e.payload['identity']['stage'] for e in events if e.kind=='DshStructuredResultAccepted'],'tool_call_count':len([e for e in events if e.kind=='DshPredictionToolExecuted'])}
 (work/'real-agent-acceptance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
 print(json.dumps({k:v for k,v in report.items() if k!='replicas'},ensure_ascii=False),flush=True)

except Exception as exc:
 import traceback
 (work/'real-agent-error.log').write_text(traceback.format_exc())
 print('Agent acceptance failed:',type(exc).__name__,getattr(exc,'error_code',None),flush=True)
 raise
finally:
 try:client.cancel({**identity,'idempotency_key':run_id+':cancel','run_state_revision':server.ledger.latest_seq(),'ledger_expected_revision':server.ledger.latest_seq()})
 except Exception:pass
 server.shutdown();server.server_close();server.ledger.close()
