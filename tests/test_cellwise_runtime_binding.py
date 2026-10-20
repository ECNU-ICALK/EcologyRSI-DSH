from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
from ecologyrsi_dsh.api import handler
from ecologyrsi_dsh.core.prediction_policy import CELLWISE_EVALUATOR_ID, TARGETED_SEARCH_POLICY, CONTRIBUTION_POLICY
from ecologyrsi_dsh.data.splits import IndexRange
from ecologyrsi_dsh.knowledge.algorithms import compile_algorithm_spec
from tests import test_runtime_integration as fixtures

class CellwiseRuntimeBindingTests(unittest.TestCase):
    setUp=fixtures.RuntimeIntegrationTests.setUp
    tearDown=fixtures.RuntimeIntegrationTests.tearDown
    request=fixtures.RuntimeIntegrationTests.request
    def test_opt_in_frozen_new_contract_and_compilable_seed(self):
        description={'descriptor':{'runnable':True,'display_name_zh':'test','domain_id':'greenhouse_test','adapter_id':'greenhouse_test'},'readiness':{'ready':True}}
        series=SimpleNamespace(digest='d'*64,split_manifest_digest_sha256='b'*64,episode_id='agc_cucumber_2018:test')
        selection=SimpleNamespace(dataset_id='agc_cucumber_2018',episode_id=series.episode_id,timestamps=tuple(range(2000)),partitions={'model_selection':IndexRange(0,2000)},data_protocol_digest='c'*64,selection_view_digest='e'*64)
        native=Mock();native.capabilities.return_value={'schema_version':'ecology-agent-runtime-capabilities/1','ready':True,'root_services':{'required':['agents'],'missing':[],'declared':True},'presets':[{'preset_id':p,'declared':True,'preset_mountable':True,'tool_surface_verified':True,'route_resolvable':True,'live_agent_service_ready':True,'first_call_verified':False} for p in handler._DSH_NATIVE_PRESET_IDS],'live_agent_service_ready':True,'first_call_verified':False}
        self.server.dsh_native_runtime=native
        body={'execution_protocol':'dsh_native_plugin_evolution@1','domain_pack_id':'greenhouse_environment@1','dataset_id':'agc_cucumber_2018',
              'strategy_model_id':'stub/strategy','review_model_id':'stub/review','autonomous_mode':True,'prediction_selection_policy':'model_during_run@1',
              'targeted_search_policy':TARGETED_SEARCH_POLICY,'budget':{'max_generations':1,'max_candidates':4,'candidates_per_generation':4},'auto_advance':0,'idempotency_key':'cellwise-test'}
        with patch.object(self.server.datasets,'describe',return_value=description),patch.object(self.server.datasets,'series',return_value=series),patch.object(self.server.datasets,'selection_view',return_value=selection):
            status,result=self.request('/runs','POST',body)
            self.assertEqual(status,201,result)
            rid=result['projection']['run_id'];state=self.server.director.state(rid);meta=state.task_manifest.metadata
            self.assertEqual(meta['evaluator_id'],CELLWISE_EVALUATOR_ID);self.assertEqual(meta['targeted_search_policy'],TARGETED_SEARCH_POLICY)
            self.assertEqual(meta['behavior_diagnostics_policy'],CONTRIBUTION_POLICY)
            self.assertEqual(meta['candidate_novelty_policy'],'run_behavior_novelty@1')
            self.assertEqual(meta['seed_genome_template_id'],'greenhouse-baseline-aligned-cellwise-default@2')
            self.assertIn('greenhouse-baseline-aligned-ridge@2',[p['id'] for p in meta['runtime_component_catalog']['prediction_models']])
            if state.run.status.value=='created':self.server.director.start_run(rid)
            seed=self.server.director.ensure_seed_incumbent_control(rid);state=self.server.director.state(rid)
            spec=compile_algorithm_spec(state.task_manifest,state.proposal(seed.proposal_id),None)
            self.assertEqual(spec.adapter_id,'greenhouse-baseline-aligned-ridge@2');self.assertEqual(len(spec.parameters),11)
            status,result=self.request('/runs','POST',{**body,'targeted_search_policy':'unknown','idempotency_key':'bad-policy'})
            self.assertEqual(status,400,result)
            status,result=self.request('/runs','POST',{**body,'candidate_novelty_policy':'unknown','idempotency_key':'bad-novelty-policy'})
            self.assertEqual(status,400,result)
            # Standard new runs get novelty checks too, without silently
            # enabling the separate cellwise scientific contract.
            standard={k:v for k,v in body.items() if k!='targeted_search_policy'}
            status,result=self.request('/runs','POST',{**standard,'idempotency_key':'standard-novelty'})
            self.assertEqual(status,201,result)
            standard_state=self.server.director.state(result['projection']['run_id'])
            self.assertEqual(standard_state.task_manifest.metadata['candidate_novelty_policy'],'run_behavior_novelty@1')
            self.assertIsNone(standard_state.task_manifest.metadata.get('targeted_search_policy'))
            self.assertEqual(standard_state.task_manifest.metadata['evaluator_id'],'greenhouse_multihorizon_time_forward@4')
        native.run_stage.assert_not_called()

if __name__=='__main__':unittest.main()
