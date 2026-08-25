import { RuntimeRunRegistry } from "./run-registry.js";
import { runtimeCapabilities } from "./capabilities.js";
import { RoleAgentManager } from "./agents.js";
import { NativeStageRunner } from "./stage-runner.js";

const DEFAULT_PRESETS = Object.freeze([
  "ecology-coordinator-v4",
  "ecology-researcher-v7",
  "ecology-candidate-proposer-v4",
  "ecology-sample-planner-v4",
  "ecology-sample-critic-v4",
  "ecology-generation-judge-v7",
].map((preset_id) => ({
  preset_id,
  tool_profile: "dynamic-retrieval-v1",
  required_tools: preset_id === "ecology-sample-planner-v4"
    ? ["ecology_execute_prediction_tool", "skill", "web_search"]
    : ["skill", "web_search"],
})));

const START_OPEN_STATUSES = Object.freeze(["created", "running"]);
const CONTROL_TRANSITIONS = Object.freeze({
  pause: Object.freeze(["created", "running"]),
  cancel: Object.freeze(["created", "running", "pausing", "paused", "cancelling"]),
  resume: Object.freeze(["pausing", "paused"]),
});

export class RuntimeController {
  constructor(ctx, { registry = new RuntimeRunRegistry(), presetCatalog = DEFAULT_PRESETS, stageRunner = null } = {}) {
    this.ctx = ctx;
    this.registry = registry;
    this.presetCatalog = presetCatalog;
    this.roleAgents = new RoleAgentManager(ctx);
    this.liveReady = false;
    this.stageRunner = stageRunner;
    this.controlDrains = new Map();
    this.runLifecycles = new Map();
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
    const lifecycle = this.#lifecycle(binding.run_id);
    const startEpoch = lifecycle.epoch;
    let releaseStart;
    const startSettlement = new Promise((resolve) => { releaseStart = resolve; });
    const startToken = { promise: startSettlement };
    lifecycle.start = startToken;
    let startReleased = false;
    const settleStart = () => {
      if (startReleased) return;
      startReleased = true;
      releaseStart();
      if (lifecycle.start === startToken) lifecycle.start = null;
    };
    const frozen = binding.binding || {};
    const strategyModel = frozen.strategy_model_id;
    const reviewModel = frozen.review_model_id;
    try {
      const creations = await Promise.allSettled(this.presetCatalog.map(({ preset_id, tool_profile }) => {
        const role = preset_id.replace(/^ecology-/, "").replace(/-v\d+$/, "");
        const reviewRole = role === "sample-critic" || role === "generation-judge";
        return this.roleAgents.createRoleAgent({
          run_id: binding.run_id,
          role,
          preset_id,
          model: reviewRole ? reviewModel : strategyModel,
          cwd: process.cwd(),
          tool_profile,
          require_workflow: role === "coordinator" || role === "sample-planner",
          preset_content_digest: frozen.preset_content_digest,
          standing_tool_surface_digest: frozen.standing_tool_surface_digest,
          route_config_digest: reviewRole
            ? frozen.resolved_review_route_config_digest
            : frozen.resolved_policy_route_config_digest,
        });
      }));
      const failed = creations.find((item) => item.status === "rejected");
      if (failed) throw failed.reason;
    } catch (error) {
      try {
        await this.roleAgents.quiesceRun(binding.run_id, { dispose: true });
        if (
          lifecycle.epoch === startEpoch
          && START_OPEN_STATUSES.includes(this.registry.get(binding.run_id)?.status)
        ) {
          this.registry.delete(binding.run_id);
        }
      } catch {
        // Best-effort private cleanup must not replace the role creation error.
      } finally {
        settleStart();
      }
      throw error;
    }
    const current = this.registry.get(binding.run_id);
    if (
      lifecycle.epoch !== startEpoch
      || !START_OPEN_STATUSES.includes(current?.status)
    ) {
      try {
        await this.roleAgents.quiesceRun(binding.run_id, {
          dispose: current?.status === "cancelling" || current?.status === "cancelled",
        });
      } finally {
        settleStart();
      }
      return this.#response(accepted);
    }
    this.stageRunner?.openLaunchFence?.(binding.run_id);
    this.liveReady = true;
    settleStart();
    return this.#response(accepted);
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
  pause(binding) {
    const current = this.#current(binding.run_id);
    const lifecycle = this.#lifecycle(binding.run_id);
    if (current.status === "paused") return Promise.resolve(this.#response(current));
    if (current.status === "pausing" && lifecycle.intent === "pause") {
      return this.controlDrains.get(binding.run_id) || Promise.resolve(this.#response(current));
    }
    if (!CONTROL_TRANSITIONS.pause.includes(current.status)) {
      return Promise.reject(this.#transitionError("pause", current.status));
    }
    const epoch = this.#advanceLifecycle(lifecycle, "pause");
    this.#mutation(binding, "pausing");
    this.stageRunner?.closeLaunchFence?.(binding.run_id);
    return this.#enqueueControl(binding.run_id, async () => {
      await this.stageRunner?.quiesceRun?.(binding.run_id);
      await this.roleAgents.quiesceRun(binding.run_id, { dispose: false });
      const latest = this.#current(binding.run_id);
      if (lifecycle.epoch === epoch && latest.status === "pausing") {
        return this.#mutation(binding, "paused");
      }
      return this.#response(latest);
    });
  }

  cancel(binding) {
    const current = this.#current(binding.run_id);
    const lifecycle = this.#lifecycle(binding.run_id);
    if (current.status === "cancelled") return Promise.resolve(this.#response(current));
    if (current.status === "cancelling" && lifecycle.intent === "cancel") {
      const active = this.controlDrains.get(binding.run_id);
      if (active) return active;
    }
    if (!CONTROL_TRANSITIONS.cancel.includes(current.status)) {
      return Promise.reject(this.#transitionError("cancel", current.status));
    }
    const epoch = this.#advanceLifecycle(lifecycle, "cancel");
    this.#mutation(binding, "cancelling");
    this.stageRunner?.closeLaunchFence?.(binding.run_id);
    return this.#enqueueControl(binding.run_id, async () => {
      await this.stageRunner?.quiesceRun?.(binding.run_id);
      await this.roleAgents.quiesceRun(binding.run_id, { dispose: true });
      const latest = this.#current(binding.run_id);
      if (lifecycle.epoch === epoch && latest.status === "cancelling") {
        return this.#mutation(binding, "cancelled");
      }
      return this.#response(latest);
    }, { continueAfterRejection: true });
  }
  async resume(binding) {
    const current = this.#current(binding.run_id);
    const lifecycle = this.#lifecycle(binding.run_id);
    const resumable = current.status === "paused"
      || (
        current.status === "pausing"
        && lifecycle.intent === "pause"
        && this.controlDrains.has(binding.run_id)
      );
    if (!resumable) {
      throw this.#transitionError("resume", current.status);
    }
    const epoch = this.#advanceLifecycle(lifecycle, "resume");
    return await this.#enqueueControl(binding.run_id, async () => {
      if (lifecycle.start?.promise) await lifecycle.start.promise;
      const latest = this.#current(binding.run_id);
      if (
        lifecycle.epoch !== epoch
        || latest.status === "cancelled"
        || latest.status === "cancelling"
      ) {
        throw this.#transitionError("resume", latest.status);
      }
      if (!CONTROL_TRANSITIONS.resume.includes(latest.status)) {
        throw this.#transitionError("resume", latest.status);
      }
      const resumed = this.#mutation(binding, "running");
      this.stageRunner?.openLaunchFence?.(binding.run_id);
      return resumed;
    });
  }

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
    return this.#response(accepted);
  }

  #response(accepted) {
    return {
      accepted: true,
      run_id: accepted.run_id,
      run_state_revision: accepted.run_state_revision,
      stage_attempt: accepted.stage_attempt,
      ledger_expected_revision: accepted.ledger_expected_revision,
      idempotency_key: accepted.idempotency_key,
    };
  }

  #current(runId) {
    const current = this.registry.get(runId);
    if (!current) throw new Error("unknown runtime run");
    return current;
  }

  #lifecycle(runId) {
    if (!this.runLifecycles.has(runId)) {
      this.runLifecycles.set(runId, { epoch: 0, intent: null, start: null });
    }
    return this.runLifecycles.get(runId);
  }

  #advanceLifecycle(lifecycle, intent) {
    lifecycle.epoch += 1;
    lifecycle.intent = intent;
    return lifecycle.epoch;
  }

  #transitionError(action, status) {
    const terminal = status === "cancelled" || status === "cancelling";
    const error = new Error(
      terminal && action === "resume"
        ? "cancelled runtime run cannot resume"
        : `runtime run cannot ${action} from ${status}`,
    );
    error.code = "runtime_control_transition_invalid";
    return error;
  }

  #enqueueControl(runId, execute, { continueAfterRejection = false } = {}) {
    const prior = this.controlDrains.get(runId);
    const operation = prior
      ? continueAfterRejection
        ? prior.then(execute, execute)
        : prior.then(execute)
      : Promise.resolve().then(execute);
    let tracked;
    tracked = operation.finally(() => {
      if (this.controlDrains.get(runId) === tracked) {
        this.controlDrains.delete(runId);
      }
    });
    tracked.catch(() => {});
    this.controlDrains.set(runId, tracked);
    return tracked;
  }
}
