from copy import deepcopy
from ecologyrsi_dsh.core.sample_results import build_sample_results
from tests.test_evaluation import _DshOriginRuntimeStub, _dsh_prediction_tool_binder
from dataclasses import replace
import unittest
from ecologyrsi_dsh.core.models import Candidate, Proposal, TaskManifest
from ecologyrsi_dsh.core.prediction_policy import CELLWISE_EVALUATOR_ID
from ecologyrsi_dsh.evaluators.greenhouse_prediction import BaselineAlignedRidgeConfig, BaselineAlignedCellwiseRidgeConfig, BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID, fit_predict_exogenous_ridge
from ecologyrsi_dsh.evaluators.registry import EvaluatorRegistry
from ecologyrsi_dsh.evaluators.agent_model_tools import AgentModelTools
from ecologyrsi_dsh.knowledge.algorithms import compile_algorithm_spec
from ecologyrsi_dsh.knowledge.algorithm_smoke import smoke_test_algorithm_spec
from ecologyrsi_dsh.knowledge.program_registry import current_program_registry
from tests.test_baseline_aligned_ridge import periodic_series, run, TARGETS
from tests.test_evaluation import _DatasetStub, _native_registry, _scope_cohort, _profile_metadata
from tests.native_fixtures import native_manifest
from tests.native_candidate_fixture import seed_proposal

class CellwisePredictionTests(unittest.TestCase):
    def test_explicit_conversion_and_one_cell_change(self):
        series = periodic_series()
        old = BaselineAlignedRidgeConfig(6,.1,.3,residual_scale_6h=.2,residual_scale_24h=.1)
        new = BaselineAlignedCellwiseRidgeConfig.from_v1(old.to_dict())
        self.assertEqual(BaselineAlignedCellwiseRidgeConfig.from_mapping(new.to_dict()),new)
        before, after = run(series,config=old), run(series,config=new)
        self.assertEqual([r['predicted'] for r in before['prediction_rows']], [r['predicted'] for r in after['prediction_rows']])
        isolated=run(series,config=replace(new,relative_humidity_6h_residual_scale=0.))
        affected=[]
        for a,b in zip(after['prediction_rows'],isolated['prediction_rows']):
            if (b['target'],b['horizon_hours']) == ('relative_humidity',6):
                self.assertEqual(b['predicted'],b['baseline']); affected.append(a['predicted']!=b['predicted'])
            else:self.assertEqual(a['predicted'],b['predicted'])
        self.assertTrue(any(affected))
        for params in (old.to_dict(),{**new.to_dict(),'relative_humidity_6h_residual_scale':1.1}):
            with self.assertRaises(ValueError): BaselineAlignedCellwiseRidgeConfig.from_mapping(params)

    def test_new_registry_compile_smoke_and_score(self):
        series=periodic_series(); params=BaselineAlignedCellwiseRidgeConfig.from_v1(BaselineAlignedRidgeConfig(6,.1,0.).to_dict()).to_dict()
        task=TaskManifest(task_id='cellwise',objective='test',domain_pack='greenhouse_environment@1',visible_datasets=(series.dataset_id,),budget={'max_candidates':1},metadata={"search_guard_policy":"practical_delta_cell_noninferiority_paired_blocks@1",
            'episode_id':series.episode_id,'prediction_model_id':BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID,'evaluator_id':CELLWISE_EVALUATOR_ID,
            'dataset_digest':series.digest,'split_manifest_digest':series.split_manifest_digest_sha256})
        task=native_manifest(metadata={**task.metadata,"seed_genome_template_id":"greenhouse-baseline-aligned-cellwise-default@2","data_protocol_digest":"f"*64})
        proposal=Proposal(proposal_id='proposal:cellwise',run_id='run:cellwise',generation=0,title='cellwise',changes=params,metadata=_profile_metadata())
        candidate=Candidate(candidate_id='candidate:cellwise',run_id=proposal.run_id,proposal_id=proposal.proposal_id,generation=0)
        registry=_native_registry(series)
        registry.validate_binding(series.dataset_id,CELLWISE_EVALUATOR_ID,BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID)
        with self.assertRaises((ValueError,KeyError)):registry.validate_binding(series.dataset_id,'greenhouse_multihorizon_time_forward@3',BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID)
        spec=compile_algorithm_spec(task,seed_proposal(task),None)
        smoke=smoke_test_algorithm_spec(spec,task,datasets=_DatasetStub(series));self.assertEqual(smoke['status'],'passed')
        bundle=registry.evaluate_scientific(task,candidate,proposal,**_scope_cohort(series,candidate))
        self.assertEqual(bundle.evaluation.score,0.);self.assertEqual(len(bundle.evaluation.metrics['targets']),9)
        self.assertEqual(bundle.artifact.parameters,params)
        template=current_program_registry().seed_template('greenhouse-baseline-aligned-cellwise-default@2')
        self.assertEqual(template.to_dict()['scientific_program']['predictor_ref']['id'],BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID)
        old_bank=AgentModelTools(series,targets=TARGETS,horizons=(1,6,24),default_config=BaselineAlignedRidgeConfig(6,.1,0.))
        self.assertNotIn(BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID,[v['tool_id'] for v in old_bank.catalog()])
        bank=AgentModelTools(series,targets=TARGETS,horizons=(1,6,24),default_config=BaselineAlignedCellwiseRidgeConfig.from_mapping(params))
        self.assertIn(BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID,[v['tool_id'] for v in bank.catalog()])

    def test_future_labels_do_not_change_fit_or_origin_prediction(self):
        series=periodic_series();config=BaselineAlignedCellwiseRidgeConfig.from_v1(BaselineAlignedRidgeConfig(6,.1,.2).to_dict())
        result=run(series,config=config)
        changed=run(replace(series,values={k:tuple(v if i<=200 else v+100 for i,v in enumerate(vs)) for k,vs in series.values.items()}),config=config)
        self.assertEqual(result['models'],changed['models']);self.assertEqual(result['baseline_profile'],changed['baseline_profile'])
        self.assertEqual([r['predicted'] for r in result['prediction_rows'] if r['origin_timestamp']==200],[r['predicted'] for r in changed['prediction_rows'] if r['origin_timestamp']==200])

    def test_native_agent_path_executes_aligned_fitted_tool_with_complete_origin_vector(self):
        series = periodic_series()
        runtime = _DshOriginRuntimeStub()
        task = TaskManifest(
            task_id="aligned-native", objective="test native aligned forecasting",
            domain_pack="greenhouse_environment@1", visible_datasets=(series.dataset_id,),
            budget={"max_candidates": 1}, metadata={"search_guard_policy":"practical_delta_cell_noninferiority_paired_blocks@1",
                "behavior_diagnostics_policy": "prediction_contribution@1",
                "episode_id": series.episode_id,
                "prediction_model_id": BASELINE_ALIGNED_CELLWISE_RIDGE_MODEL_ID,
                "evaluator_id": CELLWISE_EVALUATOR_ID,
                "dataset_digest": series.digest,
                "split_manifest_digest": series.split_manifest_digest_sha256,
                "sample_agent_mode": "dsh_native_agent",
                "sample_agent_protocol": "dsh-strict-origin-bundle@4",
                "sample_agent_batch_size": 9, "sample_concurrency": 1,
                "prediction_cells_per_origin": 9, "samples_per_update": 9,
                "strategy_model_id": "dsh/strategy", "review_model_id": "dsh/review",
            })
        proposal = Proposal(proposal_id="proposal:aligned-native", run_id="run:aligned-native",
                            generation=0, title="aligned native",
                            changes=BaselineAlignedCellwiseRidgeConfig.from_v1(BaselineAlignedRidgeConfig(6, .1, 0.0).to_dict()).to_dict(),metadata=_profile_metadata())
        candidate = Candidate(candidate_id="candidate:aligned-native", run_id=proposal.run_id,
                              proposal_id=proposal.proposal_id, generation=0)
        task=native_manifest(metadata={**task.metadata,"seed_genome_template_id":"greenhouse-baseline-aligned-cellwise-default@2","data_protocol_digest":"f"*64})
        registry = EvaluatorRegistry(
            _DatasetStub(series), dsh_runtime_provider=lambda: runtime,
            dsh_revision_provider=lambda _: {"run_state_revision": 1, "ledger_expected_revision": 1},
            dsh_identity_provider=lambda *_: {"genome_digest": "a" * 64,
                "compiled_behavior_digest": "b" * 64, "phenotype_instance_digest": "c" * 64},
            dsh_prediction_tool_binder=_dsh_prediction_tool_binder,
        )
        checkpoints = []
        durable_rows = {}

        def capture_checkpoint(checkpoint):
            checkpoints.append(deepcopy(checkpoint))
            return {}

        def capture_rows(rows):
            durable_rows.update({row["sample_id"]: deepcopy(row)
                                 for row in build_sample_results(candidate.candidate_id, rows)})

        bundle = registry.evaluate_scientific(
            task, candidate, proposal, **_scope_cohort(series,candidate),
            on_sample_checkpoint=capture_checkpoint, on_sample_results=capture_rows,
        )
        summary = bundle.evaluation.metrics["sample_execution"]
        self.assertEqual(len(summary["prediction_contribution"]["cells"]), 9)
        self.assertEqual(summary["prediction_cell_count"], 9)
        self.assertEqual(summary["dsh_agent_prediction_tool_invocations"], 1)
        self.assertEqual(summary["failed_examples"], 0)
        self.assertEqual(bundle.evaluation.score, 0.0)
        self.assertEqual([r["stage"] for r in runtime.requests],
                         ["sample.plan", "sample.critic"])
        self.assertEqual(len(durable_rows), 9)
        runtime.requests.clear()

        def resume_checkpoint(checkpoint):
            self.assertEqual(checkpoint, checkpoints[0])
            return {"rows": list(durable_rows.values())}

        replay = registry.evaluate_scientific(
            task, candidate, proposal, **_scope_cohort(series,candidate), on_sample_checkpoint=resume_checkpoint,
        )
        self.assertEqual(runtime.requests, [])
        self.assertEqual(replay.evaluation.score, bundle.evaluation.score)
        self.assertEqual(replay.artifact.learned_parameters, bundle.artifact.learned_parameters)

if __name__=='__main__':unittest.main()
