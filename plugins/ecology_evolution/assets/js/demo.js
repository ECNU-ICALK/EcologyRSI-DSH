"use strict";

  var demoCatalog = {
    domain_packs: [{ id: "crop_soil_water", name: "作物—土壤—水分模型包", description: "模拟作物生长、土壤含水量和水量平衡。" }],
    datasets: [{ id: "generated-toy-series@1", domain_pack_id: "crop_soil_water", name: "合成农田水分时间序列（第 1 版）", description: "包含气象、灌溉和土壤水分观测。", rows: 60, episodes: [{ id: "generated-toy-series@1:seed-0", label: "固定随机种子 0" }] }],
    prediction_models: [{ id: "toy-rolling-water@1", name: "合成土壤水分滚动预测模型", description: "用于本地工程演示。", dataset_ids: ["generated-toy-series@1"] }],
    strategies: [{ id: "parameter_sweep@1", name: "有界参数扫描", description: "在允许参数范围内逐轮生成候选方案。", requires_authenticated_model: false }],
    evaluators: [{ id: "toy_time_forward@1", name: "合成水分时间前向评测", description: "计算预测误差、水量平衡和约束违规。", dataset_ids: ["generated-toy-series@1"], prediction_model_ids: ["toy-rolling-water@1"], horizons_hours: [1] }],
    models: [
      { id: "host_parameter_generator@1", name: "内置有界参数生成器", roles: ["propose"], authentication_state: "local", local_model: true, available: true },
      { id: "rule_judge@1", name: "内置规则独立评审", roles: ["judge"], authentication_state: "local", local_model: true, available: true }
    ],
    policy_models: [{ id: "host_parameter_generator@1", name: "内置有界参数生成器", roles: ["propose"], authentication_state: "local", local_model: true, available: true }],
    judge_models: [{ id: "rule_judge@1", name: "内置规则独立评审", roles: ["judge"], authentication_state: "local", local_model: true, available: true }],
    unavailable_datasets: [{ id: "lettuce-catalog-only@1", name: "生菜环境观测数据（仅登记）", description: "当前缺少可运行的数据适配链路。", readiness: { ready: false, status: "catalog_only" } }],
    dsh: { connected: true, environment: "development", capabilities: ["training.data.read", "evaluation.samples.read", "evolution.run.create", "evolution.run.advance", "evolution.projection.read", "run.control", "intervention.write"] }
  };
  var demoRows = Array.from({ length: 60 }, function (_, index) {
    return {
      index: index,
      timestamp: index + 1,
      values: {
        rainfall: index % 7 === 0 ? 8.4 : 0,
        temperature: 18 + (index % 9) * 0.8,
        irrigation: index % 10 === 3 ? 5 : 0,
        soil_moisture: 0.31 - (index % 6) * 0.012
      }
    };
  });
  var demoRun = normalizeRun({
    id: "运行-演示-001", status: "running", generation: 3, total_generations: 6, candidates_count: 5, max_candidates: 12,
    projection_revision: 18, created_at: "2026-08-16T08:20:00Z", updated_at: "2026-08-16T08:36:00Z",
    manifest_digest: "sha256:demo-manifest-71c0", best_candidate_id: "候选-004",
    execution_progress: { progress_percent: 58, completed_rounds: 3, total_rounds: 6, current_stage: "evaluation", active_candidate_id: "候选-005", status: "running" },
    execution_diagnostics: { execution_mode: "registered_lightweight", fit_method: "bias_fit", training_partition_rows: 144, training_eligible_examples: 144, training_used_examples: 144, training_skipped_examples: 0, evaluation_partition_rows: 48, evaluation_eligible_examples: 48, evaluation_used_examples: 48, evaluation_skipped_examples: 0, candidate_artifacts_count: 4, candidate_evaluations_count: 4, candidate_work_items: 192, fit_passes_completed: 4, fit_passes_per_candidate: 1, iterative_epoch_training: false, proposal_sources: { remote_model: 4, host_fallback: 1 }, remote_strategy_calls: 5, remote_strategy_successes: 4, remote_strategy_status: "partial_host_fallback", fallback_used: true, fallback_count: 1 },
    configuration: { domain_pack_id: "crop_soil_water", research_domain_id: "crop_soil_water", autonomous_mode: "model_led_research", model_workflow: "research_compile_evolve@1", dataset_id: "generated-toy-series@1", episode_id: "generated-toy-series@1:seed-0", prediction_model_id: "toy-rolling-water@1", strategy_id: "parameter_sweep@1", evaluator_id: "toy_time_forward@1", policy_model_id: "host_parameter_generator@1", judge_model_id: "rule_judge@1", slot: "bounded_predictor" },
    dataset: { id: "generated-toy-series@1", episode_id: "generated-toy-series@1:seed-0", partition: "training_fit", digest: "sha256:demo-dataset-8b42" },
    selection_scope: "iterative_training_feedback_only", formal_validation_status: "not_run",
    trajectory: [
      { generation: 1, candidate_id: "候选-001", score: 0.714, best_score: 0.714 },
      { generation: 1, candidate_id: "候选-002", score: 0.746, best_score: 0.746 },
      { generation: 2, candidate_id: "候选-003", score: 0.731, best_score: 0.746 },
      { generation: 3, candidate_id: "候选-004", score: 0.802, best_score: 0.802 }
    ],
    gate: { visible: "passed", process: "passed", hidden: "restricted", release: "pending" },
    candidates: [
      { id: "候选-004", parent_id: "候选-002", generation: 3, status: "accepted", score: 0.802, created_at: "2026-08-16T08:35:00Z", rationale: "降低干旱期残差放大系数，改善连续无降水日的预测稳定性。", changes: { stress_gain: { before: 1.12, after: 0.94 }, memory_days: { before: 4, after: 5 } }, metrics: { rmse: 0.082, mae: 0.061, water_balance_error: 0.047, constraint_violations: 0, prediction_preview: [{ timestamp: 49, target: "soil_moisture", observed: 0.28, predicted: 0.274, baseline: 0.262, unit: "体积比" }, { timestamp: 50, target: "soil_moisture", observed: 0.267, predicted: 0.263, baseline: 0.251, unit: "体积比" }] }, promotion: { decision: "approved", reason: "训练反馈误差下降且过程约束满足，因此在搜索阶段保留。" } },
      { id: "候选-003", parent_id: "候选-002", generation: 2, status: "rejected", score: 0.731, created_at: "2026-08-16T08:31:00Z", changes: { stress_gain: { before: 1.12, after: 1.28 } }, metrics: { rmse: 0.109, mae: 0.083, water_balance_error: 0.091, constraint_violations: 1 }, promotion: { decision: "rejected", reason: "出现水量平衡约束违规。" } },
      { id: "候选-002", parent_id: "候选-001", generation: 1, status: "accepted", score: 0.746, created_at: "2026-08-16T08:27:00Z", changes: { stress_gain: { before: 1, after: 1.12 } }, metrics: { rmse: 0.096, mae: 0.074, water_balance_error: 0.066, constraint_violations: 0 } },
      { id: "候选-001", parent_id: null, generation: 1, status: "evaluated", score: 0.714, created_at: "2026-08-16T08:24:00Z", changes: { memory_days: { before: 3, after: 4 } }, metrics: { rmse: 0.105, mae: 0.081, water_balance_error: 0.072, constraint_violations: 0 } },
      { id: "候选-005", parent_id: "候选-004", generation: 4, slot_index: 0, status: "evaluating", created_at: "2026-08-16T08:37:10Z", title: "干旱窗口的滚动残差校正", rationale: "根据上一轮共同弱点，先缩小无降雨连续时段的残差放大，再用反馈样本检查是否改善短期预测。", changes: { stress_gain: { before: 0.94, after: 0.88 }, memory_days: { before: 5, after: 6 } }, model_plan: { team: { name: "作物水分预测研究小组" }, prediction_model: { id: "toy-rolling-water@1", name: "合成土壤水分滚动预测模型" }, strategy: { id: "adaptive_local@1", name: "局部自适应搜索" }, research: ["连续干旱日残差稳定性"], summary: "优先验证稳定性，再比较误差和水量平衡约束。" }, execution: { current_stage: "evaluation", progress_percent: 58, active_candidate_id: "候选-005", stages: { proposal: "completed", candidate: "completed", training: "completed", evaluation: "running", judge: "pending", decision: "pending" }, inference_trace: [{ sample_index: 1, origin_timestamp: 49, target_timestamp: 50, horizon_hours: 1, target: "soil_moisture", input_summary: { soil_moisture: 0.267, rainfall: 0, temperature: 20.4 }, observed: 0.263, predicted: 0.265, baseline: 0.251, unit: "体积比", step_summary: "读取最近 6 个时间步，应用滚动窗口和干旱残差校正。" }, { sample_index: 2, origin_timestamp: 50, target_timestamp: 51, horizon_hours: 1, target: "soil_moisture", input_summary: { soil_moisture: 0.255, rainfall: 0, temperature: 21.2 }, observed: 0.252, predicted: 0.254, baseline: 0.244, unit: "体积比", step_summary: "沿用上一时刻预测作为状态，检查含水量边界。" }, { sample_index: 3, origin_timestamp: 51, target_timestamp: 52, horizon_hours: 1, target: "soil_moisture", input_summary: { soil_moisture: 0.244, rainfall: 0, temperature: 21.8 }, observed: 0.241, predicted: 0.239, baseline: 0.232, unit: "体积比", step_summary: "完成一次前向预测，等待本轮误差汇总。" }] }, metrics: { prediction_preview: [{ timestamp: 50, target_timestamp: 50, origin_timestamp: 49, horizon_hours: 1, target: "soil_moisture", observed: 0.263, predicted: 0.265, baseline: 0.251, unit: "体积比" }, { timestamp: 51, target_timestamp: 51, origin_timestamp: 50, horizon_hours: 1, target: "soil_moisture", observed: 0.252, predicted: 0.254, baseline: 0.244, unit: "体积比" }, { timestamp: 52, target_timestamp: 52, origin_timestamp: 51, horizon_hours: 1, target: "soil_moisture", observed: 0.241, predicted: 0.239, baseline: 0.232, unit: "体积比" }] } }
    ],
    interventions: [{ id: "意见-001", kind: "guidance", message: "优先降低连续干旱日的预测偏差，不放宽水量平衡约束。", created_by: "示例研究员", created_at: "2026-08-16T08:29:00Z", effective_generation: 3 }],
    expert_consultations: [
      { id: "咨询-002", consultation_id: "咨询-002", status: "pending", question: "连续无降雨窗口是否应按作物生育阶段设置不同的残差阈值？", context: "候选-005 在短窗口内改善了误差，但模型无法从当前训练反馈确认该阈值能否跨生育阶段复用。未答复时将沿用保守的统一阈值并继续可逆搜索。", options: [{ id: "stage_specific", label: "按生育阶段设置" }, { id: "shared_conservative", label: "保留统一保守阈值" }], uncertainty_type: "scientific", confidence: 0.58, source_generation: 4, candidate_id: "候选-005", model_id: "host_parameter_generator@1", non_blocking: true, created_at: "2026-08-16T08:37:25Z" },
      { id: "咨询-001", consultation_id: "咨询-001", status: "answered", question: "水量平衡误差接近门限时，优先收缩残差增益还是延长历史窗口？", context: "两种调整均可降低短期误差，但对物理约束的影响不同。", options: ["先收缩残差增益", "先延长历史窗口"], uncertainty_type: "method", confidence: 0.63, source_generation: 2, candidate_id: "候选-003", model_id: "host_parameter_generator@1", non_blocking: true, answer: "先收缩残差增益，并保持水量平衡门限不变；若下一轮仍不稳定，再延长历史窗口。", selected_option: "先收缩残差增益", answered_by: "示例领域专家", answered_at: "2026-08-16T08:33:00Z", effective_generation: 3, applied_generation: 3, created_at: "2026-08-16T08:31:20Z" }
    ],
    artifacts: [{ artifact_id: "产物-004", candidate_id: "候选-004", model_id: "土壤水分滚动预测模型@1", training_partition: "training_fit", training_rows: 36, artifact_digest: "sha256:demo-artifact-004", learned_parameters: { alpha: 0.42 }, metrics: { rmse: 0.071, n: 36 } }],
    training_assets: [
      { sample_id: "样本-004", generation: 3, candidate_id: "候选-004", admission: { tier: "iterative_positive", formal_training_ready: false, requires_governance_review: true }, input: { strategy_id: "bounded_component_search", policy_model_id: "policy-model-local", applied_interventions: [{ kind: "guidance" }] }, output: { artifact: { artifact_digest: "sha256:demo-artifact-004" } }, evaluation: { score: 0.802, partition: "training_feedback", judge: { model_id: "judge-model-local", accepted: true } }, provenance: { dataset_digest: "sha256:demo-dataset-8b42", artifact_digest: "sha256:demo-artifact-004" }, trajectory_summary: { stage_count: 8, prediction_count: 3, status: "completed" }, trajectory: { input_context: { status: "completed", summary: "冻结合成农田水分序列第 3 轮反馈样本，目标为降低短期预测误差。", dataset_id: "generated-toy-series@1", episode_id: "generated-toy-series@1:seed-0" }, agent_research: { status: "completed", summary: "策略模型检索滚动残差校正方法，选择已登记的有界组件。", actor: "策略模型 API" }, agent_proposal: { status: "completed", summary: "提出降低干旱期残差放大并延长历史窗口。", parameter_changes: { stress_gain: { before: 1.12, after: 0.94 }, memory_days: { before: 4, after: 5 } } }, host_compile: { status: "completed", summary: "DSH 宿主将参数提案编译为可执行候选-004。", actor: "DSH 宿主" }, training_prediction: { status: "completed", summary: "在训练拟合分区训练滚动预测模型，并在训练反馈分区逐样本预测。", prediction_records: [{ sample_index: 1, target: "soil_moisture", input_summary: { soil_moisture: 0.28, rainfall: 0 }, observed_value: 0.28, predicted_value: 0.274, baseline_value: 0.262, error: -0.006, unit: "体积比", step_summary: "读取最近 5 个时间步并执行滚动前向预测。" }, { sample_index: 2, target: "soil_moisture", input_summary: { soil_moisture: 0.267, rainfall: 0 }, observed_value: 0.267, predicted_value: 0.263, baseline_value: 0.251, error: -0.004, unit: "体积比", step_summary: "沿用上一时刻状态并检查含水量边界。" }, { sample_index: 3, target: "soil_moisture", input_summary: { soil_moisture: 0.255, rainfall: 0 }, observed_value: 0.255, predicted_value: 0.254, baseline_value: 0.244, error: -0.001, unit: "体积比", step_summary: "完成反馈样本预测并计算相对持续性基线误差。" }], sample_count: 3, shown_count: 3 }, agent_feedback: { status: "completed", summary: "训练反馈误差下降，水量平衡约束满足。", score: 0.802, metrics: { rmse: 0.082, mae: 0.061 }, actor: "训练反馈评测器" }, agent_optimization: { status: "completed", summary: "保留候选-004作为当前搜索版本，下一轮继续验证短期稳定性。", decision: "搜索保留", parameter_changes: { stress_gain: "1.12→0.94", memory_days: "4→5" } }, final_result: { status: "completed", summary: "候选-004 在迭代搜索中保留；正式训练与发布仍需外部治理。", prediction_records: [{ sample_index: 1, target: "soil_moisture", observed_value: 0.28, predicted_value: 0.274, baseline_value: 0.262, error: -0.006, unit: "体积比" }] } } },
      { sample_id: "样本-003", generation: 2, candidate_id: "候选-003", admission: { tier: "iterative_negative", formal_training_ready: false, requires_governance_review: true }, input: { strategy_id: "bounded_component_search", policy_model_id: "policy-model-local", applied_interventions: [] }, evaluation: { score: 0.731, partition: "training_feedback", judge: { model_id: "judge-model-local", accepted: false } }, provenance: { dataset_digest: "sha256:demo-dataset-8b42" } }
    ],
    rounds: [
      { generation: 1, candidate_id: "候选-002", parent_candidate_id: "候选-001", candidate_count: 1, batch_size: 1, eligible_count: 1, score: 0.746, decision: "promoted", champion_candidate_id: "候选-002", selection_reason: "generation_champion_strictly_improved_incumbent", next_generation_focus: "继续检查干旱时段的残差稳定性", applied_intervention_count: 0, stages: { proposal: "completed", candidate: "completed", training: "completed", evaluation: "completed", judge: "completed", decision: "completed" }, candidates: [{ candidate_id: "候选-002", slot_index: 0, score: 0.746, rank: 1, selection_reason: "generation_champion_strictly_improved_incumbent", stages: { proposal: "completed", candidate: "completed", training: "completed", evaluation: "completed", judge: "completed", decision: "completed" } }] },
      { generation: 2, candidate_id: "候选-003", parent_candidate_id: "候选-002", candidate_count: 1, batch_size: 1, eligible_count: 0, score: 0.731, decision: "no_eligible_candidate", champion_candidate_id: null, selection_reason: "scientific_gate_failed", next_generation_focus: "保留候选-002，避免放宽水量平衡约束", applied_intervention_count: 0, stages: { proposal: "completed", candidate: "completed", training: "completed", evaluation: "completed", judge: "completed", decision: "completed" }, candidates: [{ candidate_id: "候选-003", slot_index: 0, score: 0.731, rank: 1, selection_reason: "scientific_gate_failed", stages: { proposal: "completed", candidate: "completed", training: "completed", evaluation: "completed", judge: "completed", decision: "completed" } }] },
      { generation: 3, candidate_id: "候选-004", parent_candidate_id: "候选-002", candidate_count: 1, batch_size: 1, eligible_count: 1, score: 0.802, decision: "promoted", champion_candidate_id: "候选-004", selection_reason: "generation_champion_strictly_improved_incumbent", next_generation_focus: "继续验证短期预测误差是否稳定下降", applied_intervention_count: 1, stages: { proposal: "completed", candidate: "completed", training: "completed", evaluation: "completed", judge: "completed", decision: "completed" }, candidates: [{ candidate_id: "候选-004", slot_index: 0, score: 0.802, rank: 1, selection_reason: "generation_champion_strictly_improved_incumbent", stages: { proposal: "completed", candidate: "completed", training: "completed", evaluation: "completed", judge: "completed", decision: "completed" } }] },
      { generation: 4, candidate_id: "候选-005", parent_candidate_id: "候选-004", candidate_count: 1, batch_size: 2, eligible_count: 0, decision: "pending", knowledge: { snapshot_digest: "sha256:demo-knowledge-004", query_terms: ["soil moisture rolling forecast drought residual"], cards: [{ title: "时间序列预测：滚动验证与残差诊断", source_url: "https://otexts.com/fpp3/", source_authority: "公开教材", publication_year: 2021, capability_id: "toy-rolling-water@1", execution_status: "adopted", selection_reason: "可映射到已登记滚动预测器" }, { title: "Attention Is All You Need", source_url: "https://arxiv.org/abs/1706.03762", source_authority: "公开论文", publication_year: 2017, execution_status: "research_only", selection_reason: "用于研究比较，未直接接入执行器" }] }, knowledge_assessment: { conclusion: "优先验证干旱窗口残差稳定性，再比较反馈误差与水量平衡约束。", next_action: "若候选-005 严格改善且无约束违规，则进入轮末保留判断。" }, stages: { proposal: "completed", candidate: "completed", training: "completed", evaluation: "running", judge: "pending", decision: "pending" }, candidates: [{ candidate_id: "候选-005", slot_index: 0, score: null, rank: null, selection_reason: "等待样本误差和水量平衡检查", stages: { proposal: "completed", candidate: "completed", training: "completed", evaluation: "running", judge: "pending", decision: "pending" } }], selection_reason: "等待样本误差和水量平衡检查", next_generation_focus: "根据本轮反馈决定是否保留残差校正版本", common_failures: [] }
    ]
  });
  var demoEvents = [
    { id: "事件-22", type: "stage.recorded", occurred_at: "2026-08-16T08:37:42Z", payload: { stage: "evaluation", status: "running", generation: 3, candidate_id: "候选-005", message: "候选-005 正在逐样本执行训练反馈评测。" } },
    { id: "事件-21", type: "candidate.spawned", occurred_at: "2026-08-16T08:37:10Z", payload: { candidate_id: "候选-005", generation: 3, message: "模型方案已编译为候选-005，等待样本评测。" } },
    { id: "事件-18", type: "evaluation.completed", occurred_at: "2026-08-16T08:36:00Z", payload: { message: "候选-004 完成独立评测，综合得分为 0.802。" } },
    { id: "事件-17", type: "candidate.accepted", occurred_at: "2026-08-16T08:35:20Z", payload: { message: "候选-004 满足本轮训练反馈与过程约束检查，已在搜索阶段保留。" } },
    { id: "事件-14", type: "generation.advanced", occurred_at: "2026-08-16T08:32:00Z", payload: { message: "第 3 轮进化已完成。" } },
    { id: "事件-10", type: "intervention.recorded", occurred_at: "2026-08-16T08:29:00Z", payload: { message: "人工方向建议已写入下一轮输入。" } },
    { id: "事件-01", type: "run.started", occurred_at: "2026-08-16T08:20:00Z", payload: { message: "任务清单已冻结，进化运行开始。" } }
  ];

  function demoDatasetPage(offset, partition) {
    var selectedPartition = partition === "training_feedback" ? "training_feedback" : "training_fit";
    var partitionRows = selectedPartition === "training_feedback" ? demoRows.slice(36, 48) : demoRows.slice(0, 36);
    return {
      dataset: { id: "generated-toy-series@1", episode_id: "generated-toy-series@1:seed-0", partition: selectedPartition },
      descriptor: { id: "generated-toy-series@1", name: "合成农田水分时间序列（第 1 版）", digest: "sha256:demo-dataset-8b42", description: "用于本地交付测试的 60 天合成序列。" },
      readiness: { ready: true, status: "ready", source_integrity: { schema_version: "ecologyrsi-dsh.source-integrity/1", status: "not_applicable", verified: null, source_count: 0, verified_count: 0, missing_count: 0, mismatch_count: 0, unverifiable_count: 0, message_zh: "合成演示数据没有需要校验的本地来源归档。", sources: [] } },
      profile: { rows: 60, training_fit_rows: 36, training_feedback_rows: 12, hidden_rows: 12 },
      features: {
        rainfall: { name: "rainfall", display_name_zh: "降雨", role: "驱动变量", unit: "毫米" },
        temperature: { name: "temperature", display_name_zh: "气温", role: "驱动变量", unit: "摄氏度" },
        irrigation: { name: "irrigation", display_name_zh: "灌溉量", role: "控制变量", unit: "毫米" },
        soil_moisture: { name: "soil_moisture", display_name_zh: "土壤含水量", role: "目标变量", unit: "体积比" }
      },
      partition: selectedPartition,
      partitions: { training_fit: 36, training_feedback: 12, hidden: 12 },
      page: { rows: partitionRows.slice(offset, offset + state.pageLimit), offset: offset, limit: state.pageLimit, total: partitionRows.length }
    };
  }

  function loadDemo() {
    state.usingDemo = true;
    state.catalog = normalizeCatalog(demoCatalog);
    state.runs = [clone(demoRun)].map(normalizeRun);
    state.showArchivedRuns = false;
    state.archivedRunCount = 0;
    state.activeRun = state.runs[0];
    state.lastSelectedRunId = state.activeRun.id;
    state.events = clone(demoEvents);
    state.candidateSelectionPinned = false;
    syncCandidateSelection(state.activeRun);
    state.datasetContext = activeRunDatasetContext();
    state.datasetError = null;
    state.datasetPage = demoDatasetPage(0, state.datasetPartition);
    state.loadState = "demo";
    state.lastError = null;
    state.lastUpdated = new Date().toISOString();
    setConnection("demo", "显式演示模式");
    populateCatalogControls();
    renderAll();
  }
