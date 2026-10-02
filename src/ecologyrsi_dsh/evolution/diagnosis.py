"""Host-owned diagnosis from visible, completed native research evidence.

This is an aggregate projection, not a new evaluator or a causal diagnosis.
Raw samples and model-authored root-cause claims never enter this report.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from ..core.research import DiagnosticReport, HypothesisProposal, content_id


def failure_category(code: str) -> str:
    """Classify recorded reasons without treating absent evidence as poor fit."""
    code = str(code)
    if any(part in code for part in ("insufficient", "incomplete", "probation", "requires_independent", "not_established")):
        return "insufficient_evidence"
    if any(part in code for part in ("structured_", "timeout", "execution_failed", "judge_unavailable", "transport", "aborted")):
        return "operational"
    if any(part in code for part in ("invalid", "contract_mismatch", "incompatible_comparison", "safety_gate", "constraint_violations")):
        return "invalid_change"
    if any(part in code for part in ("regression", "no_positive", "below_practical", "negative_skill")):
        return "performance"
    return "unclassified"


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
    paired_diagnostics = getattr(task, "metadata", {}).get("optimization_protocol") == "evidence_guided_epoch@1"
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
    if paired_diagnostics and comparison is not None:
        references.append("comparison:" + comparison.comparison_id)
        gates = comparison.gate_results.get("arms", {})
        for arm, gate in gates.items():
            if arm == "incumbent" or not isinstance(gate, Mapping):
                continue
            for cell, value in gate.get("cell_deltas", {}).items():
                if type(value) in (int, float) and math.isfinite(value) and value < 0 and cell not in weaknesses:
                    weaknesses.append(cell)
            failures.extend(str(code) for code in gate.get("search_failures", ()))
        questions.append("逐单元负差值参考同组 incumbent；批次基线技能分不能当作局部修改增益。")
    operational = any(failure_category(name) == "operational" for name in failures) if paired_diagnostics else any(name in {"execution_failed", "judge_unavailable"} for name in failures)
    causes = []
    if operational:
        causes.append("可能存在执行或评审服务问题；这不是科学模型失效的证据。")
        questions.append("运行失败能否在固定程序和数据条件下复现？")
    if weaknesses:
        causes.append("误差可能与历史特征、参数或模型结构有关，尚未确定原因。")
        questions.append("先核对时间可见性、单位和缺失处理，再进行有界改动的配对实验。")
    if paired_diagnostics and "independent_review_not_accepted" in failures:
        questions.append("分开核验探索选择与独立认证的评审要求；评审拒绝本身不能证明预测性能退化。")
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
