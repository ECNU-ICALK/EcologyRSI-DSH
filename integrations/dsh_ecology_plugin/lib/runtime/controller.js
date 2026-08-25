import { RuntimeRunRegistry } from "./run-registry.js";
import { runtimeCapabilities } from "./capabilities.js";
import { RoleAgentManager } from "./agents.js";
import { NativeStageRunner } from "./stage-runner.js";

const DEFAULT_PRESETS = Object.freeze([
  "ecology-coordinator-v3",
  "ecology-researcher-v6",
  "ecology-candidate-proposer-v3",
  "ecology-sample-planner-v3",
  "ecology-sample-critic-v3",
  "ecology-generation-judge-v6",
].map((preset_id) => ({
  preset_id,
  required_tools: preset_id === "ecology-sample-planner-v3"
    ? ["ecology_execute_prediction_tool", "skill"]
    : ["skill"],
})));

export class RuntimeController {
  constructor(ctx, { registry = new RuntimeRunRegistry(), presetCatalog = DEFAULT_PRESETS, stageRunner = null } = {}) {
    this.ctx = ctx;
    this.registry = registry;
    this.presetCatalog = presetCatalog;
    this.roleAgents = new RoleAgentManager(ctx);
    this.liveReady = false;
    this.stageRunner = stageRunner;
  }

  configureStageRunner(config) {
    if (this.stageRunner) return this.stageRunner;
    this.stageRunner = NativeStageRunner.fromConfig(this.ctx, {
      roleAgents: this.roleAgents,
      runRegistry: this.registry,
      ...config,
    });
    return this.stageRunner;
  }

  async capabilities() {
    const value = await runtimeCapabilities(this.ctx, this.presetCatalog);
    if (!this.liveReady) return value;
    const livePresets = value.presets.map((item) => ({
      ...item,
      live_agent_service_ready: Boolean(
        item.declared && item.preset_mountable && item.tool_surface_verified && item.route_resolvable,
      ),
    }));
    return {
      ...value,
      ready: Boolean(value.ready && livePresets.every((item) => item.live_agent_service_ready)),
      live_agent_service_ready: livePresets.every((item) => item.live_agent_service_ready),
      presets: livePresets,
    };
  }

  async startRun(binding) {
    const accepted = this.registry.start(binding);
    const frozen = binding.binding || {};
    const strategyModel = frozen.strategy_model_id;
    const reviewModel = frozen.review_model_id;
    try {
      await Promise.all(this.presetCatalog.map(({ preset_id }) => {
      const role = preset_id.replace(/^ecology-/, "").replace(/-v\d+$/, "");
      const reviewRole = role === "sample-critic" || role === "generation-judge";
      return this.roleAgents.createRoleAgent({
        run_id: binding.run_id,
        role,
        preset_id,
        model: reviewRole ? reviewModel : strategyModel,
        cwd: process.cwd(),
        require_workflow: role === "coordinator" || role === "sample-planner",
        preset_content_digest: frozen.preset_content_digest,
        standing_tool_surface_digest: frozen.standing_tool_surface_digest,
        route_config_digest: reviewRole
          ? frozen.resolved_review_route_config_digest
          : frozen.resolved_policy_route_config_digest,
      });
      }));
    } catch (error) {
      this.registry.delete(binding.run_id);
      throw error;
    }
    this.liveReady = true;
    return {
      accepted: true,
      run_id: accepted.run_id,
      run_state_revision: accepted.run_state_revision,
      stage_attempt: accepted.stage_attempt,
      ledger_expected_revision: accepted.ledger_expected_revision,
      idempotency_key: accepted.idempotency_key,
    };
  }

  async runStage(binding) {
    if (!this.stageRunner?.run) throw new Error("structured DSH stage runner is unavailable");
    const stageResult = await this.stageRunner.run(binding);
    if (
      stageResult?.skill_invocation_evidence?.first_tool_call_verified !== true
      || stageResult?.skill_invocation_evidence?.order_verified !== true
    ) {
      throw new Error("DSH stage has no verified Skill-first execution evidence");
    }
    // Stage completion updates its frozen revision receipt but never owns run
    // control. Pause/cancel may have won while this child was in flight.
    const accepted = this.registry.refresh(binding);
    return {
      accepted: true,
      run_id: accepted.run_id,
      run_state_revision: accepted.run_state_revision,
      stage_attempt: accepted.stage_attempt,
      ledger_expected_revision: accepted.ledger_expected_revision,
      idempotency_key: accepted.idempotency_key,
      structured: stageResult.structured,
      result_digest: stageResult.result_digest,
      session_id: stageResult.session_id,
      first_call_verified: true,
    };
  }
  async pause(binding) {
    this.#mutation(binding, "pausing");
    await this.stageRunner?.quiesceRun?.(binding.run_id);
    await this.roleAgents.quiesceRun(binding.run_id, { dispose: false });
    return this.#mutation(binding, "paused");
  }

  async cancel(binding) {
    this.#mutation(binding, "cancelling");
    await this.stageRunner?.quiesceRun?.(binding.run_id);
    await this.roleAgents.quiesceRun(binding.run_id, { dispose: true });
    return this.#mutation(binding, "cancelled");
  }
  async resume(binding) { return this.#mutation(binding, "running"); }

  async status(runId) {
    const current = this.registry.get(runId);
    if (!current) throw new Error("unknown runtime run");
    return {
      run_id: current.run_id,
      status: current.status,
      run_state_revision: current.run_state_revision,
      stage_attempt: current.stage_attempt,
      ledger_expected_revision: current.ledger_expected_revision,
      idempotency_key: current.idempotency_key,
    };
  }

  #mutation(binding, status) {
    const accepted = this.registry.transition(binding.run_id, binding, status);
    return {
      accepted: true,
      run_id: accepted.run_id,
      run_state_revision: accepted.run_state_revision,
      stage_attempt: accepted.stage_attempt,
      ledger_expected_revision: accepted.ledger_expected_revision,
      idempotency_key: accepted.idempotency_key,
    };
  }
}
