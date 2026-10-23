import { isDeepStrictEqual } from "node:util";

function deepFreeze(value, seen = new Set()) {
  if (value === null || typeof value !== "object" || seen.has(value)) return value;
  seen.add(value);
  for (const key of Reflect.ownKeys(value)) deepFreeze(value[key], seen);
  return Object.freeze(value);
}

function immutableClone(value) {
  return deepFreeze(structuredClone(value));
}

export class RuntimeRunRegistry {
  #runs = new Map();
  #metadata = new WeakMap();
  #closedFormalExecutions = new Set();

  startFormal(binding) {
    const executionKey = `${binding.run_id}\0${binding.binding?.formal_execution?.execution_id}`;
    if (this.#closedFormalExecutions.has(executionKey)) {
      throw new Error("closed formal execution cannot be prepared again");
    }
    const prior = this.#runs.get(binding.run_id);
    if (prior && prior.status !== "cancelled") {
      if (prior.binding?.formal_execution
        && isDeepStrictEqual(this.#metadata.get(prior)?.startBinding, binding)) return prior;
      throw new Error("formal runtime requires a quiesced prior generation");
    }
    const startBinding = immutableClone(binding);
    return this.#publish(binding.run_id, { ...startBinding, status: "running" },
      Object.freeze({ generation: Object.freeze({}), startBinding }));
  }

  restore(record) {
    if (!this.#metadata.has(record)) throw new Error("unknown runtime generation");
    this.#runs.set(record.run_id, record);
  }

  start(binding) {
    const prior = this.#runs.get(binding.run_id);
    if (prior && ["cancelling", "cancelled"].includes(prior.status)) {
      const error = new Error(`runtime run cannot start from ${prior.status}`);
      error.code = "runtime_start_transition_invalid";
      throw error;
    }
    if (prior) {
      const metadata = this.#metadata.get(prior);
      if (!metadata || !isDeepStrictEqual(metadata.startBinding, binding)) {
        const error = new Error("run already has a different start command");
        error.code = "runtime_start_conflict";
        throw error;
      }
      return prior;
    }

    const startBinding = immutableClone(binding);
    const requestedStatus = startBinding?.binding?.initial_run_status || "created";
    const provenance = startBinding?.binding?.restore_provenance;
    const exactRestore = (
      provenance !== null
      && typeof provenance === "object"
      && !Array.isArray(provenance)
      && Object.keys(provenance).length === 2
      && provenance.source === "python_durable_ledger"
      && provenance.status === requestedStatus
      && startBinding.idempotency_key === `runtime-restore:${startBinding.run_id}`
    );
    if (
      (provenance !== undefined && !exactRestore)
      || (
        !["created", "running"].includes(requestedStatus)
        && !(requestedStatus === "paused" && exactRestore)
      )
    ) {
      throw new Error("invalid initial runtime run status");
    }
    const metadata = Object.freeze({
      generation: Object.freeze({}),
      startBinding,
    });
    return this.#publish(startBinding.run_id, {
      ...startBinding,
      status: requestedStatus,
    }, metadata);
  }

  transition(runId, binding, status) {
    const prior = this.#runs.get(runId);
    if (!prior) throw new Error("unknown runtime run");
    if (prior.run_id !== binding.run_id) throw new Error("runtime run identity mismatch");
    if (status === "cancelled" && prior.binding?.formal_execution) {
      this.#closedFormalExecutions.add(`${runId}\0${prior.binding.formal_execution.execution_id}`);
    }
    return this.#publish(runId, { ...prior, ...binding, status }, this.#metadata.get(prior));
  }

  refresh(binding) {
    const prior = this.#runs.get(binding.run_id);
    if (!prior) throw new Error("unknown runtime run");
    if (prior.run_id !== binding.run_id) throw new Error("runtime run identity mismatch");
    const staleStageReceipt = (
      prior.status !== "running"
      || binding.run_state_revision < prior.run_state_revision
      || binding.ledger_expected_revision < prior.ledger_expected_revision
    );
    if (staleStageReceipt) return prior;
    return this.#publish(
      binding.run_id,
      { ...prior, ...binding, status: prior.status },
      this.#metadata.get(prior),
    );
  }

  generationOf(record) {
    return this.#metadata.get(record)?.generation || null;
  }

  #publish(runId, value, metadata) {
    if (!metadata) throw new Error("runtime run generation metadata is missing");
    const frozen = immutableClone(value);
    this.#metadata.set(frozen, metadata);
    this.#runs.set(runId, frozen);
    return frozen;
  }

  get(runId) { return this.#runs.get(runId) || null; }
  delete(runId) { return this.#runs.delete(runId); }
  values() { return Object.freeze([...this.#runs.values()]); }
}
