"""Host-owned diagnosis from visible, completed native research evidence.

This is an aggregate projection, not a new evaluator or a causal diagnosis.
Raw samples and model-authored root-cause claims never enter this report.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from ..core.research import DiagnosticReport, HypothesisProposal, content_id


def diagnose_generation(state, knowledge) -> DiagnosticReport:
    generation = state.run.generation
    previous = state.analysis_for(generation - 1) if generation else None
    comparison = state.comparison_for(generation - 1) if generation else None
    return diagnose_evidence(state.task_manifest, previous, comparison, knowledge)


def diagnose_evidence(task, previous, comparison, knowledge) -> DiagnosticReport:
    """Pure projection shared by command validation and historical replay."""
    references = ["task:" + task.digest, "knowledge:" + knowledge.snapshot_digest]
    weaknesses: list[str] = []
    failures: list[str] = []
    questions: list[str] = []
    if previous is not None:
        references.append("analysis:" + previous.analysis_digest)
        for row in (*previous.target_weaknesses, *previous.horizon_weaknesses):
            values = (row.get("median_skill_score"), row.get("median_normalized_mean_reward"))
            if not any(type(value) in (int, float) and math.isfinite(value) and value < 0 for value in values):
                continue
            target = row.get("target")
            horizon = row.get("horizon_hours")
            label = str(target) if isinstance(target, str) and target else "all_targets"
            if type(horizon) is int and horizon > 0:
                label += f"@{horizon}h"
            if label not in weaknesses:
                weaknesses.append(label[:240])
        failures = list(dict.fromkeys(previous.common_failures))[:8]
        if previous.insufficient_evidence:
            failures.append("insufficient_evidence")
            questions.append("当前比较样本不足，先增加同条件证据；不能据此禁止该科学程序。")
    else:
        questions.append("尚无已完成的基线实验，先建立可比较的参考结果。")
    operational = any(name in {"execution_failed", "judge_unavailable"} for name in failures)
    causes = []
    if operational:
        causes.append("可能存在执行或评审服务问题；这不是科学模型失效的证据。")
        questions.append("运行失败能否在固定程序和数据条件下复现？")
    if weaknesses:
        causes.append("误差可能与历史特征、参数或模型结构有关，尚未确定原因。")
        questions.append("先核对时间可见性、单位和缺失处理，再进行有界改动的配对实验。")
    return DiagnosticReport(
        task_id=task.task_id,
        program_id=(previous.search_parent_candidate_id or previous.incumbent_after_candidate_id or "seed:" + task.digest) if previous else "seed:" + task.digest,
        comparison_id=comparison.comparison_id if comparison else None,
        weak_cells=tuple(weaknesses[:12]),
        failure_patterns=tuple(dict.fromkeys(failures)),
        evidence_refs=tuple(references),
        possible_causes=tuple(causes),
        unresolved_questions=tuple(questions),
    )


def hypothesis_for_proposal(proposal, diagnostic: DiagnosticReport) -> dict | None:
    """Bind an existing structured direction to the actual host-bounded change."""
    direction = proposal.metadata.get("candidate_direction")
    if not isinstance(direction, Mapping):
        return None
    change = {"parameters": dict(proposal.changes),
              "mutation_operations": proposal.metadata.get("mutation_operations", []),
              "genome_digest": proposal.metadata.get("genome_digest")}
    change_ref = "change@1:" + content_id("ecologyrsi/proposal-change@1", change)
    refs = tuple(dict.fromkeys((*diagnostic.evidence_refs, *direction.get("evidence_refs", ()))))
    hypothesis = HypothesisProposal(
        proposal_id=proposal.proposal_id,
        parent_program_id=proposal.parent_candidate_id or diagnostic.program_id,
        problem=direction["target_weakness"],
        hypothesis=direction["hypothesis"],
        evidence_refs=refs,
        change_kind=direction["mutation_axis"],
        change_spec_ref=change_ref,
        expected_observation=direction["success_criterion"],
        falsification_condition="在冻结的比较合同下，未满足上述预期或违反预先设定的逐单元、资源及物理约束。",
        applicability_conditions=("task:" + diagnostic.task_id, "search_evidence_only"),
    )
    return {"hypothesis": hypothesis.to_dict(), "change_spec": change,
            "diagnostic_report_id": diagnostic.report_id, "evidence_class": "exploratory"}
