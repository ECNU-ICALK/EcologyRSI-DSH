export class RuntimeRunRegistry {
  #runs = new Map();

  start(binding) {
    const prior = this.#runs.get(binding.run_id);
    if (prior && ["cancelling", "cancelled"].includes(prior.status)) {
      const error = new Error(`runtime run cannot start from ${prior.status}`);
      error.code = "runtime_start_transition_invalid";
      throw error;
    }
    if (prior && prior.idempotency_key !== binding.idempotency_key) {
      throw new Error("run already has a different active command");
    }
    const requestedStatus = binding?.binding?.initial_run_status || "created";
    const provenance = binding?.binding?.restore_provenance;
    const exactRestore = (
      provenance !== null
      && typeof provenance === "object"
      && !Array.isArray(provenance)
      && Object.keys(provenance).length === 2
      && provenance.source === "python_durable_ledger"
      && provenance.status === requestedStatus
      && binding.idempotency_key === `runtime-restore:${binding.run_id}`
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
    const frozen = Object.freeze({ ...binding, status: requestedStatus });
    this.#runs.set(binding.run_id, frozen);
    return frozen;
  }

  transition(runId, binding, status) {
    const prior = this.#runs.get(runId);
    if (!prior) throw new Error("unknown runtime run");
    if (prior.run_id !== binding.run_id) throw new Error("runtime run identity mismatch");
    const frozen = Object.freeze({ ...prior, ...binding, status });
    this.#runs.set(runId, frozen);
    return frozen;
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
    const frozen = Object.freeze({ ...prior, ...binding, status: prior.status });
    this.#runs.set(binding.run_id, frozen);
    return frozen;
  }

  get(runId) { return this.#runs.get(runId) || null; }
  delete(runId) { return this.#runs.delete(runId); }
  values() { return Object.freeze([...this.#runs.values()]); }
}
