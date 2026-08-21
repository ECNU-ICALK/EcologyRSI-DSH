export class RuntimeRunRegistry {
  #runs = new Map();

  start(binding) {
    const prior = this.#runs.get(binding.run_id);
    if (prior && prior.idempotency_key !== binding.idempotency_key) {
      throw new Error("run already has a different active command");
    }
    const frozen = Object.freeze({ ...binding, status: "created" });
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

  get(runId) { return this.#runs.get(runId) || null; }
  delete(runId) { return this.#runs.delete(runId); }
  values() { return Object.freeze([...this.#runs.values()]); }
}
