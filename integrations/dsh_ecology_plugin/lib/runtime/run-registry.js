export class RuntimeRunRegistry {
  #runs = new Map();

  start(binding) {
    const prior = this.#runs.get(binding.run_id);
    if (prior && prior.idempotency_key !== binding.idempotency_key) {
      throw new Error("run already has a different active command");
    }
    const requestedStatus = binding?.binding?.initial_run_status || "created";
    if (!["created", "running"].includes(requestedStatus)) {
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
