"""Separate, single-use evaluation of a locked Agent on later time partitions."""
from dataclasses import replace
import logging
import threading
import time
from typing import Any

from ..core.exposure_registry import ScientificExposureRegistry
from ..core.models import digest
from ..core.immutable import thaw_json
from ..core.trajectory import EvaluationPhase, EvaluationScope
from ..data.adapters import dataset_adapter
from ..evaluators.epoch_cohorts import PlannedCohort, _eligible_origins


class IndependentEvaluationService:
    def __init__(self, server):
        self.server = server
        self.lock = threading.RLock()
        self.jobs: dict[tuple[str, str], dict[str, Any]] = {}
        self.workers: list[threading.Thread] = []
        self.stopping = threading.Event()

    def recover_interrupted(self):
        registry = ScientificExposureRegistry(self.server.ledger)
        for token in registry.formal_stage_tokens():
            if token.idempotency_key != f"{token.run_id}:independent:{token.stage}":
                continue
            events = self.server.ledger.events(token.run_id)
            if any(e.kind == "FormalStageSealed" and e.payload.get("token_digest") == token.token_digest
                   for e in events):
                continue
            completed = next((e for e in events if e.kind == "FormalStageCompleted"
                              and e.payload.get("token_digest") == token.token_digest), None)
            outcome = completed.payload["outcome"] if completed else "inconclusive"
            exposure = registry.formal_exposure(token.holdout_exposure_key)
            if exposure["state"] != "sealed":
                registry.seal_formal_stage(token, outcome=outcome)
            if any(e.kind == "FormalStageFrozen" and e.payload.get("token_digest") == token.token_digest
                   for e in events):
                self.server.ledger.append(token.run_id, "FormalStageSealed",
                    {"stage": token.stage, "candidate_id": token.candidate_id,
                     "token_digest": token.token_digest, "outcome": outcome},
                    event_id=f"{token.run_id}:formal:{token.stage}:sealed")

    def close(self):
        self.stopping.set()
        deadline = time.monotonic() + 5
        for worker in self.workers:
            worker.join(max(0, deadline - time.monotonic()))
        return not any(worker.is_alive() for worker in self.workers)

    def status(self, run_id):
        state = self.server.director.state(run_id)
        stages = []
        for stage, label in (("validation", "独立验证"), ("final_test", "最终测试")):
            completed = next((e.payload for e in state.events if e.kind == "FormalStageCompleted"
                              and e.payload.get("stage") == stage), None)
            frozen = next((e.payload for e in state.events if e.kind == "FormalStageFrozen"
                           and e.payload.get("stage") == stage), None)
            sealed = next((e.payload for e in state.events if e.kind == "FormalStageSealed"
                           and e.payload.get("stage") == stage), None)
            with self.lock:
                live = dict(self.jobs.get((run_id, stage), {}))
            candidate_id = (state.run.selection_incumbent_id if stage == "validation"
                            else state.run.validated_candidate_id)
            reason = ""
            if state.run.status.value != "completed":
                reason = "先完成进化训练，再冻结版本进行独立评测。"
            elif not candidate_id:
                reason = "尚无可评测的冻结候选。" if stage == "validation" else "需先通过独立验证。"
            elif not state.artifact_for(candidate_id):
                reason = "候选缺少已冻结的模型产物。"
            if frozen:
                reason = "该评测分区已锁定，不能重复试验或更换候选。"
            stages.append({"stage": stage, "label": label, "candidate_id": candidate_id,
                           "available": not reason, "reason": reason,
                           "status": "completed" if completed else "sealed" if sealed else live.get("status", "interrupted" if frozen else "not_started"),
                           "outcome": completed.get("outcome") if completed else sealed.get("outcome") if sealed else None,
                           "assessment": thaw_json(completed.get("assessment")) if completed else None,
                           "progress": live.get("progress", {}),
                           "error": live.get("error")})
        return {"schema_version": "ecologyrsi-dsh.independent-evaluation/1", "run_id": run_id,
                "stages": stages, "feedback_to_evolution": False}

    def start(self, run_id: str, stage: str):
        if stage not in {"validation", "final_test"}:
            raise ValueError("独立评测阶段必须为 validation 或 final_test")
        with self.lock:
            if self.stopping.is_set():
                raise ValueError("服务正在关闭，请稍后重试")
            report = self.status(run_id)
            item = next(s for s in report["stages"] if s["stage"] == stage)
            if item["status"] in {"running", "completed", "sealed"}:
                return report
            if not item["available"]:
                raise ValueError(item["reason"])
            if any(worker.is_alive() for worker in self.workers):
                raise ValueError("已有独立评测正在执行，请等待完成后再启动")
            state = self.server.director.state(run_id)
            task = state.task_manifest
            if task.metadata.get("sample_agent_mode") != "dsh_native_agent":
                raise ValueError("独立评测要求已冻结的逐样本 Agent 运行")
            self.server.validate_frozen_runtime_bindings(task, run_id=run_id)
            protocol = self.server.datasets.data_protocol(task.dataset, task.metadata["episode_id"])
            if protocol.protocol_digest != task.metadata.get("data_protocol_digest"):
                raise ValueError("时间分区已变化，请完成当前协议下的新训练运行")
            candidate_id = item["candidate_id"]
            artifact = state.artifact_for(candidate_id)
            if artifact.candidate_revision_id is None:
                raise ValueError("独立评测要求明确的候选版本和模型产物绑定")
            adapter = dataset_adapter(task.dataset)
            plan = {"schema_version": "ecologyrsi-dsh.independent-analysis-plan/1", "stage": stage,
                    "dataset_task": adapter.contract(), "partition_digest": protocol.partition_digests[stage],
                    "sampling": "all_eligible_origins_in_time_order", "inference_replicas": 2,
                    "fit_policy": "original_calibration_fit_only_verify_frozen_coefficients",
                    "candidate_updates": False, "raw_results_exposed": False,
                    "post_score_agent_feedback": False}
            token = self.server.director.reserve_formal_stage(run_id, stage=stage,
                candidate_id=candidate_id, objective_family_digest=adapter.contract()["contract_digest"],
                analysis_plan_digest=digest(plan), partition_digest=protocol.partition_digests[stage],
                idempotency_key=f"{run_id}:independent:{stage}")
            self.jobs[(run_id, stage)] = {"status": "running", "progress": {}}
            worker = threading.Thread(target=self._execute, args=(run_id, token, plan), daemon=True,
                                      name=f"independent-{stage}")
            self.workers = [worker]
            worker.start()
            return self.status(run_id)

    def _execute(self, run_id, token, plan):
        def evaluate():
            self.server.sample_admission.forget(run_id)
            self.server.dsh_tools.open_run_admissions(run_id)
            return self._evaluate(run_id, token, plan)
        try:
            self.server.director.execute_formal_stage(run_id, token, evaluate)
            with self.lock:
                self.jobs[(run_id, token.stage)]["status"] = "completed"
        except Exception:
            logging.getLogger(__name__).exception("Independent evaluation failed for %s/%s", run_id, token.stage)
            with self.lock:
                self.jobs[(run_id, token.stage)].update(status="failed", error="评测中断，当前分区已封存。详情见服务日志。")
        finally:
            self.server.dsh_tools.close_run_admissions(run_id)

    def _evaluate(self, run_id, token, plan):
        from .formal_trajectory import _revision_evaluation_inputs
        state = self.server.director.state(run_id)
        task = state.task_manifest
        candidate = state.candidate(token.candidate_id)
        artifact = state.artifact_for(candidate.candidate_id)
        if artifact.digest != token.artifact_digest:
            raise ValueError("independent evaluation artifact changed after freezing")
        series = self.server.datasets.formal_view(task.dataset, token, ScientificExposureRegistry(self.server.ledger))
        origins, _gaps = _eligible_origins(series)
        if not origins:
            raise ValueError("独立评测分区没有满足历史窗口和预测时距的样本")
        adapter = dataset_adapter(task.dataset)
        cohort = PlannedCohort(token.stage, origins, max(adapter.horizons_hours))
        _, proposal, spec = _revision_evaluation_inputs(state, candidate, artifact.candidate_revision_id, task)
        if proposal.metadata["genome_digest"] != token.genome_digest:
            raise ValueError("independent evaluation genome changed after freezing")
        frozen_policy = thaw_json(artifact.learned_parameters.get("agent_policy"))
        if not frozen_policy:
            raise ValueError("frozen artifact has no Agent policy")
        proposal = replace(proposal, metadata={**dict(proposal.metadata), "agent_policy": frozen_policy,
                                               "tool_experience": frozen_policy["experience"]["rows"]})
        replicas = []
        for replica in range(plan["inference_replicas"]):
            if self.stopping.is_set():
                raise RuntimeError("independent evaluation interrupted by shutdown")
            scope = EvaluationScope(run_id, candidate.generation, candidate.candidate_id,
                artifact.candidate_revision_id, EvaluationPhase(token.stage), cohort.cohort_digest,
                cohort.origin_count, inference_replica=replica)
            formal_task = replace(task, metadata={**dict(task.metadata),
                "_evaluation_scope": scope.to_dict(), "_planned_evaluation_cohort": cohort.to_dict(),
                "independent_evaluation_stage": token.stage})
            def progress(payload):
                # Do not publish raw labels, predictions, or Agent messages.
                counts = {k: v for k, v in payload.items() if k in {
                    "completed_origins", "total_origins", "completed_examples", "total_examples",
                    "succeeded_examples", "failed_examples", "completed_samples", "total_samples",
                    "completed_origin_samples", "total_origin_samples", "succeeded_origin_samples"}
                    and isinstance(v, (int, float))}
                with self.lock:
                    self.jobs[(run_id, token.stage)]["progress"] = {
                        "replica": replica + 1, "replicas": plan["inference_replicas"],
                        "origin_count": cohort.origin_count, **counts}
            bundle = self.server.evaluators._evaluate_greenhouse_ridge(
                formal_task, candidate, proposal, series, horizons=adapter.horizons_hours,
                predictor_model_id=artifact.model_id, frozen_artifact=artifact,
                execution_plan=self.server.evaluators._resolve_execution_plan(candidate, proposal, spec),
                on_sample_control=lambda: "cancelled" if self.stopping.is_set() else "running",
                on_evaluation_progress=progress)
            metrics = bundle.evaluation.metrics
            if metrics.get("prediction_owner") != "sample_agent":
                raise ValueError("independent evaluation did not execute the frozen sample Agent")
            replicas.append({"replica": replica + 1, "score": bundle.evaluation.score,
                "passed": bundle.evaluation.passed,
                "coverage": metrics["sample_execution_coverage"],
                "targets": [{key: row.get(key) for key in ("target", "unit", "horizon_hours", "n",
                    "mae", "rmse", "bias", "baseline_rmse", "skill_score", "sample_execution_coverage",
                    "constraint_violations")} for row in metrics["targets"]],
                "prediction_trace_digest": metrics.get("sample_execution_trace_digest")})
        coverage_ok = all(r["coverage"] >= adapter.selection_minimum_coverage for r in replicas)
        outcome = "inconclusive" if not coverage_ok else "passed" if all(r["passed"] for r in replicas) else "failed"
        return {"schema_version": "ecologyrsi-dsh.independent-assessment/1", "outcome": outcome,
                "stage": token.stage, "artifact_digest": token.artifact_digest,
                "genome_digest": token.genome_digest, "analysis_plan": plan,
                "origin_count": cohort.origin_count, "replicas": replicas,
                "feedback_to_evolution": False, "candidate_updated": False}
