import { isDeepStrictEqual } from "node:util";
import { readFileSync } from "node:fs";

import { RuntimeRunRegistry } from "./run-registry.js";
import { runtimeCapabilities } from "./capabilities.js";
import { RoleAgentManager } from "./agents.js";
import { NativeStageRunner } from "./stage-runner.js";
import { ModelContractCanary } from "./model-canary.js";

const PRESET_MANIFEST = Object.freeze(JSON.parse(readFileSync(
  new URL("../../presets/preset-manifest.json", import.meta.url),
  "utf8",
)));
if (
  PRESET_MANIFEST.schema_version !== "ecologyrsi-dsh.preset-manifest/1"
  || !Array.isArray(PRESET_MANIFEST.presets)
  || PRESET_MANIFEST.presets.length === 0
) {
  throw new Error("preset-manifest.json is invalid");
}
const DEFAULT_PRESETS = Object.freeze(PRESET_MANIFEST.presets.map((item) => (
  Object.freeze({
    preset_id: String(item.preset_id),
    tool_profile: String(item.tool_profile),
    required_tools: Object.freeze([...item.required_tools].map(String)),
  })
)));

export const CURRENT_PRESET_MANIFEST = PRESET_MANIFEST;

const START_OPEN_STATUSES = Object.freeze(["created", "running"]);
const TERMINAL_START_STATUSES = Object.freeze(["cancelling", "cancelled"]);
const CONTROL_TRANSITIONS = Object.freeze({
  pause: Object.freeze(["created", "running", "pausing", "resuming"]),
  cancel: Object.freeze([
    "created", "running", "pausing", "paused", "resuming", "cancelling",
  ]),
  resume: Object.freeze(["pausing", "paused"]),
});

export class RuntimeController {
  constructor(ctx, { registry = new RuntimeRunRegistry(), presetCatalog = DEFAULT_PRESETS, stageRunner = null } = {}) {
    this.ctx = ctx;
    this.modelCanary = new ModelContractCanary(ctx);
    this.registry = registry;
    this.presetCatalog = presetCatalog;
    this.roleAgents = new RoleAgentManager(ctx);
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

  runCanary(request) { return this.modelCanary.run(request); }

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

  get liveReady() {
    if (this.presetCatalog.length === 0) return false;
    const liveRuns = this.registry.values().filter(
      (run) => !TERMINAL_START_STATUSES.includes(run.status),
    );
    return liveRuns.length > 0 && liveRuns.every(
      (run) => this.#hostsReadyFor(
        this.runLifecycles.get(run.run_id),
        run,
      ),
    );
  }

  startRun(binding) {
    const current = this.registry.get(binding.run_id);
    if (current && TERMINAL_START_STATUSES.includes(current.status)) {
      return Promise.reject(this.#startTransitionError(current.status));
    }
    const lifecycle = this.#lifecycle(binding.run_id);
    const currentGeneration = this.registry.generationOf(current);
    if (
      lifecycle.start !== null
      && lifecycle.start.generation === currentGeneration
    ) {
      if (isDeepStrictEqual(lifecycle.start.binding, binding)) {
        return lifecycle.start.promise;
      }
      return Promise.reject(this.#startConflictError());
    }
    if (lifecycle.start !== null && !lifecycle.start.finalized) {
      return Promise.reject(this.#startConflictError());
    }
    let accepted;
    try {
      accepted = this.registry.start(binding);
    } catch (error) {
      return Promise.reject(error);
    }
    if (current !== null && accepted === current) {
      return Promise.resolve({
        ...this.#response(accepted),
        idempotency_key: binding.idempotency_key,
      });
    }
    const generation = this.registry.generationOf(accepted);
    if (!generation) {
      return Promise.reject(new Error("runtime run generation metadata is missing"));
    }
    let releaseFinalization;
    const finalization = new Promise((resolve) => { releaseFinalization = resolve; });
    const startToken = {
      binding: structuredClone(binding),
      accepted,
      generation,
      startEpoch: lifecycle.epoch,
      promise: null,
      finalization,
      releaseFinalization,
      finalized: false,
      status: "pending",
    };
    lifecycle.start = startToken;
    lifecycle.hosts = "creating";
    lifecycle.hostGeneration = generation;
    this.stageRunner?.closeLaunchFence?.(binding.run_id);
    startToken.promise = Promise.resolve().then(
      () => this.#performStart(startToken.binding, lifecycle, startToken),
    );
    startToken.promise.catch(() => {});
    return startToken.promise;
  }

  async #performStart(binding, lifecycle, startToken) {
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
          preset_content_digest: frozen.preset_content_digest,
          standing_tool_surface_digest: frozen.standing_tool_surface_digest,
          route_config_digest: reviewRole
            ? frozen.resolved_review_route_config_digest
            : frozen.resolved_policy_route_config_digest,
        });
      }));
      const failed = creations.find((item) => item.status === "rejected");
      if (failed) throw failed.reason;
      const current = this.registry.get(binding.run_id);
      if (
        !this.#ownsStartGeneration(lifecycle, startToken)
        || this.registry.generationOf(current) !== startToken.generation
      ) {
        await this.roleAgents.quiesceRun(binding.run_id, { dispose: true });
        if (this.#ownsStartGeneration(lifecycle, startToken)) {
          lifecycle.hosts = "superseded";
        }
        startToken.status = "fulfilled";
        return this.#response(startToken.accepted);
      }
      lifecycle.hosts = "ready";
      if (
        lifecycle.epoch !== startToken.startEpoch
        || !START_OPEN_STATUSES.includes(current?.status)
      ) {
        await this.roleAgents.quiesceRun(binding.run_id, {
          dispose: current?.status === "cancelling" || current?.status === "cancelled",
        });
        startToken.status = "fulfilled";
        return this.#response(startToken.accepted);
      }
      if (!this.#openLaunchFenceFor(binding.run_id, lifecycle, startToken.generation)) {
        await this.roleAgents.quiesceRun(binding.run_id, { dispose: true });
        lifecycle.hosts = "superseded";
      }
      startToken.status = "fulfilled";
      return this.#response(startToken.accepted);
    } catch (primaryError) {
      if (this.#ownsStartGeneration(lifecycle, startToken)) {
        lifecycle.hosts = "failed";
      }
      this.stageRunner?.closeLaunchFence?.(binding.run_id);
      try {
        await this.roleAgents.quiesceRun(binding.run_id, { dispose: true });
      } catch {
        // Best-effort private cleanup must not replace the start failure.
      } finally {
        const current = this.registry.get(binding.run_id);
        if (
          lifecycle.epoch === startToken.startEpoch
          && current === startToken.accepted
        ) {
          this.registry.delete(binding.run_id);
        }
      }
      startToken.status = "rejected";
      if (lifecycle.start === startToken) lifecycle.start = null;
      throw primaryError;
    } finally {
      if (!startToken.finalized) {
        startToken.finalized = true;
        startToken.releaseFinalization();
      }
    }
  }

  async runStage(binding) {
    if (!this.stageRunner?.run) throw new Error("structured DSH stage runner is unavailable");
    const lifecycle = this.#lifecycle(binding.run_id);
    const admitted = this.#current(binding.run_id);
    const admittedGeneration = this.registry.generationOf(admitted);
    if (!this.#hostsReadyFor(lifecycle, admitted, admittedGeneration)) {
      this.stageRunner?.closeLaunchFence?.(binding.run_id);
      throw this.#hostsIncompleteError();
    }
    const stageResult = await this.stageRunner.run(binding);
    if (
      stageResult?.skill_invocation_evidence?.first_tool_call_verified !== true
      || stageResult?.skill_invocation_evidence?.order_verified !== true
    ) {
      throw new Error("DSH stage has no verified Skill-first execution evidence");
    }
    const latest = this.#current(binding.run_id);
    if (!this.#hostsReadyFor(lifecycle, latest, admittedGeneration)) {
      this.stageRunner?.closeLaunchFence?.(binding.run_id);
      throw this.#hostsIncompleteError();
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
  activate(binding) {
    const current = this.#current(binding.run_id);
    const lifecycle = this.#lifecycle(binding.run_id);
    const generation = this.registry.generationOf(current);
    if (current.status === "running") {
      return Promise.resolve(this.#response({ ...current, ...binding }));
    }
    if (current.status !== "created") {
      return Promise.reject(this.#transitionError("start", current.status));
    }
    if (!this.#hostsReadyFor(lifecycle, current, generation)) {
      this.stageRunner?.closeLaunchFence?.(binding.run_id);
      return Promise.reject(this.#hostsIncompleteError());
    }
    const activated = this.registry.transition(binding.run_id, binding, "running");
    if (!this.#openLaunchFenceFor(binding.run_id, lifecycle, generation)) {
      return Promise.reject(this.#hostsIncompleteError());
    }
    return Promise.resolve(this.#response(activated));
  }
  pause(binding) {
    const current = this.#current(binding.run_id);
    const lifecycle = this.#lifecycle(binding.run_id);
    const existing = this.#matchingControl(lifecycle, "pause", binding);
    if (existing) return existing.promise;
    if (current.status === "paused") return Promise.resolve(this.#response(current));
    const exactRetry = (
      current.status === "pausing"
      && lifecycle.failed.pause !== null
      && isDeepStrictEqual(lifecycle.failed.pause, binding)
    );
    if (current.status === "pausing" && !exactRetry) {
      return Promise.reject(this.#transitionError("pause", current.status));
    }
    if (!CONTROL_TRANSITIONS.pause.includes(current.status)) {
      return Promise.reject(this.#transitionError("pause", current.status));
    }
    this.#advanceLifecycle(lifecycle, "pause");
    const token = this.#controlToken(lifecycle, "pause", binding);
    if (current.status !== "resuming") this.#mutation(binding, "pausing");
    const operation = this.#queueControl(lifecycle, token, async () => {
      const latestBeforeDrain = this.#current(binding.run_id);
      if (
        lifecycle.terminalEpoch !== token.terminalEpoch
        || latestBeforeDrain.status === "cancelling"
        || latestBeforeDrain.status === "cancelled"
      ) {
        return this.#response(latestBeforeDrain);
      }
      if (latestBeforeDrain.status !== "pausing") {
        this.#mutation(binding, "pausing");
      }
      await this.stageRunner?.quiesceRun?.(binding.run_id);
      await this.#waitForStartFinalization(lifecycle);
      await this.roleAgents.quiesceRun(binding.run_id, { dispose: false });
      const latest = this.#current(binding.run_id);
      if (
        lifecycle.terminalEpoch !== token.terminalEpoch
        || latest.status === "cancelling"
        || latest.status === "cancelled"
      ) {
        return this.#response(latest);
      }
      const later = this.#laterControl(lifecycle, token);
      if (later) return this.#response(latest);
      return this.#mutation(binding, "paused");
    }, {
      continueAfterRejection: true,
      rememberFailure: true,
    });
    this.stageRunner?.closeLaunchFence?.(binding.run_id);
    return operation;
  }

  cancel(binding) {
    const current = this.#current(binding.run_id);
    const lifecycle = this.#lifecycle(binding.run_id);
    const existing = this.#matchingControl(lifecycle, "cancel", binding);
    if (existing) return existing.promise;
    if (current.status === "cancelled") return Promise.resolve(this.#response(current));
    const exactRetry = (
      current.status === "cancelling"
      && lifecycle.failed.cancel !== null
      && isDeepStrictEqual(lifecycle.failed.cancel, binding)
    );
    if (current.status === "cancelling" && !exactRetry) {
      return Promise.reject(this.#transitionError("cancel", current.status));
    }
    if (!CONTROL_TRANSITIONS.cancel.includes(current.status)) {
      return Promise.reject(this.#transitionError("cancel", current.status));
    }
    this.#advanceLifecycle(lifecycle, "cancel");
    lifecycle.terminalEpoch += 1;
    const token = this.#controlToken(lifecycle, "cancel", binding);
    this.#mutation(binding, "cancelling");
    const operation = this.#queueControl(lifecycle, token, async () => {
      await this.stageRunner?.quiesceRun?.(binding.run_id);
      await this.#waitForStartFinalization(lifecycle);
      await this.roleAgents.quiesceRun(binding.run_id, { dispose: true });
      const latest = this.#current(binding.run_id);
      if (lifecycle.terminalEpoch === token.terminalEpoch && latest.status === "cancelling") {
        return this.#mutation(binding, "cancelled");
      }
      return this.#response(latest);
    }, {
      continueAfterRejection: true,
      rememberFailure: true,
    });
    this.stageRunner?.closeLaunchFence?.(binding.run_id);
    return operation;
  }
  resume(binding) {
    const current = this.#current(binding.run_id);
    const lifecycle = this.#lifecycle(binding.run_id);
    const existing = this.#matchingControl(lifecycle, "resume", binding);
    if (existing) return existing.promise;
    const resumable = current.status === "paused"
      || (
        current.status === "pausing"
        && this.#hasControl(lifecycle, "pause")
      );
    if (!resumable) {
      return Promise.reject(this.#transitionError("resume", current.status));
    }
    const currentGeneration = this.registry.generationOf(current);
    const liveCreatingStart = (
      lifecycle.hosts === "creating"
      && lifecycle.start !== null
      && !lifecycle.start.finalized
      && lifecycle.hostGeneration === currentGeneration
      && lifecycle.start.generation === currentGeneration
    );
    if (!this.#hostsReadyFor(lifecycle, current, currentGeneration) && !liveCreatingStart) {
      this.stageRunner?.closeLaunchFence?.(binding.run_id);
      return Promise.reject(this.#hostsIncompleteError());
    }
    this.#advanceLifecycle(lifecycle, "resume");
    const source = current;
    const token = this.#controlToken(lifecycle, "resume", binding);
    token.hostGeneration = currentGeneration;
    this.#mutation(binding, "resuming");
    return this.#queueControl(lifecycle, token, async () => {
      await this.#waitForStartFinalization(lifecycle);
      const latest = this.registry.get(binding.run_id);
      if (!this.#hostsReadyFor(lifecycle, latest, token.hostGeneration)) {
        this.stageRunner?.closeLaunchFence?.(binding.run_id);
        throw this.#hostsIncompleteError();
      }
      if (
        lifecycle.terminalEpoch !== token.terminalEpoch
        || latest.status === "cancelled"
        || latest.status === "cancelling"
      ) {
        throw this.#transitionError("resume", latest.status);
      }
      if (![...CONTROL_TRANSITIONS.resume, "resuming"].includes(latest.status)) {
        throw this.#transitionError("resume", latest.status);
      }
      const resumed = this.registry.transition(binding.run_id, binding, "running");
      if (!this.#laterControl(lifecycle, token, new Set(["pause", "cancel"]))) {
        if (!this.#openLaunchFenceFor(binding.run_id, lifecycle, token.hostGeneration)) {
          throw this.#hostsIncompleteError();
        }
      }
      return this.#response(resumed);
    }, {
      onRejected: () => {
        const latest = this.registry.get(binding.run_id);
        if (
          lifecycle.terminalEpoch === token.terminalEpoch
          && this.registry.generationOf(latest) === token.hostGeneration
          && (
            latest?.status === "resuming"
            || (source.status === "pausing" && latest?.status === "pausing")
          )
          && !this.#laterControl(lifecycle, token, new Set(["pause", "cancel"]))
        ) {
          const rollbackStatus = (
            source.status === "pausing"
            && lifecycle.failed.pause === null
          )
            ? "paused"
            : source.status;
          this.registry.transition(binding.run_id, source, rollbackStatus);
        }
      },
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
      this.runLifecycles.set(runId, {
        epoch: 0,
        terminalEpoch: 0,
        intent: null,
        start: null,
        hosts: "unknown",
        hostGeneration: null,
        controls: new Set(),
        nextControlSequence: 0,
        failed: { pause: null, cancel: null },
      });
    }
    return this.runLifecycles.get(runId);
  }

  #hostsReadyFor(lifecycle, record, expectedGeneration = undefined) {
    if (!lifecycle || !record || lifecycle.hosts !== "ready") return false;
    const generation = this.registry.generationOf(record);
    return Boolean(
      generation
      && lifecycle.hostGeneration === generation
      && (expectedGeneration === undefined || generation === expectedGeneration)
    );
  }

  #ownsStartGeneration(lifecycle, startToken) {
    return lifecycle.start === startToken
      && lifecycle.hostGeneration === startToken.generation;
  }

  #openLaunchFenceFor(runId, lifecycle, expectedGeneration) {
    const current = this.registry.get(runId);
    if (
      !START_OPEN_STATUSES.includes(current?.status)
      || !this.#hostsReadyFor(lifecycle, current, expectedGeneration)
    ) {
      this.stageRunner?.closeLaunchFence?.(runId);
      return false;
    }
    this.stageRunner?.openLaunchFence?.(runId);
    return true;
  }

  #advanceLifecycle(lifecycle, intent) {
    lifecycle.epoch += 1;
    lifecycle.intent = intent;
    return lifecycle.epoch;
  }

  #controlToken(lifecycle, action, binding) {
    lifecycle.nextControlSequence += 1;
    return {
      action,
      binding: structuredClone(binding),
      sequence: lifecycle.nextControlSequence,
      terminalEpoch: lifecycle.terminalEpoch,
      promise: null,
    };
  }

  #matchingControl(lifecycle, action, binding) {
    return [...lifecycle.controls].find(
      (token) => token.action === action && isDeepStrictEqual(token.binding, binding),
    ) || null;
  }

  #hasControl(lifecycle, action) {
    return [...lifecycle.controls].some((token) => token.action === action);
  }

  #laterControl(lifecycle, token, actions = null) {
    return [...lifecycle.controls].find((candidate) => (
      candidate.sequence > token.sequence
      && (actions === null || actions.has(candidate.action))
    )) || null;
  }

  async #waitForStartFinalization(lifecycle) {
    if (lifecycle.start?.finalization) await lifecycle.start.finalization;
  }

  #queueControl(
    lifecycle,
    token,
    execute,
    {
      continueAfterRejection = false,
      rememberFailure = false,
      onRejected = null,
    } = {},
  ) {
    lifecycle.controls.add(token);
    const promise = this.#enqueueControl(token.binding.run_id, execute, {
      continueAfterRejection,
    });
    token.promise = promise;
    promise.then(
      () => {
        if (rememberFailure) lifecycle.failed[token.action] = null;
        lifecycle.controls.delete(token);
        this.#pruneRun(token.binding.run_id);
      },
      () => {
        if (rememberFailure) lifecycle.failed[token.action] = token.binding;
        try { onRejected?.(); } catch {}
        lifecycle.controls.delete(token);
        this.#pruneRun(token.binding.run_id);
      },
    );
    return promise;
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

  #startTransitionError(status) {
    const error = new Error(`runtime run cannot start from ${status}`);
    error.code = "runtime_start_transition_invalid";
    return error;
  }

  #startConflictError() {
    const error = new Error("run already has a different start command");
    error.code = "runtime_start_conflict";
    return error;
  }

  #hostsIncompleteError() {
    const error = new Error("runtime role hosts are incomplete");
    error.code = "runtime_role_hosts_incomplete";
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

  // The run registry keeps every record so `status` stays answerable, but the
  // per-run control lifecycle and the launch-fence markers only mean anything
  // while a transition can still occur. Drop them once the run is fully
  // cancelled and its control queue has drained; `#lifecycle` recreates a
  // neutral entry if a late idempotent replay still arrives.
  //
  // `cancelling` is deliberately excluded even though it is a terminal *start*
  // status: a failed cancel is still exact-retryable from there, and that retry
  // is recognised through `lifecycle.failed.cancel`.
  #pruneRun(runId) {
    const lifecycle = this.runLifecycles.get(runId);
    if (!lifecycle) return;
    if (this.registry.get(runId)?.status !== "cancelled") return;
    if (lifecycle.controls.size > 0) return;
    if (lifecycle.start !== null && !lifecycle.start.finalized) return;
    this.runLifecycles.delete(runId);
    this.stageRunner?.forgetRun?.(runId);
  }

  async dispose() {
    const runIds = new Set(this.registry.values().map((run) => run.run_id));
    await Promise.allSettled([...this.controlDrains.values()]);
    await Promise.allSettled(
      [...runIds].map((runId) => this.roleAgents.quiesceRun(runId, { dispose: true })),
    );
    await Promise.allSettled([
      Promise.resolve().then(() => this.stageRunner?.dispose?.()),
      Promise.resolve().then(() => this.roleAgents.dispose()),
    ]);
    this.controlDrains.clear();
    this.runLifecycles.clear();
  }
}
