import { createHash, randomUUID } from "node:crypto";

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

export function childReservationLabel(launch) {
  const identity = {
    schema: "ecology-child-reservation/1",
    reservation_id: launch.reservation_id,
    run_id: launch.run_id,
    stage: launch.stage,
    role: launch.role,
    item_digest: launch.item_digest,
    idempotency_key: launch.idempotency_key,
    launch_attempt: launch.launch_attempt,
  };
  return `ecology-${createHash("sha256").update(canonical(identity)).digest("hex")}`;
}

function descriptorLabel(events = []) {
  const descriptor = [...events].reverse().find((event) => event?.type === "subagent/descriptor");
  return descriptor?.data?.label ?? descriptor?.payload?.label ?? null;
}

export class ChildBindingRegistry {
  constructor({ durable = null, foldDescriptor = descriptorLabel } = {}) {
    this.durable = durable || { tombstones: new Set(), reservations: new Map(), activations: new Map() };
    this.durable.tombstones ||= new Set();
    this.durable.reservations ||= new Map();
    this.durable.activations ||= new Map();
    this.foldDescriptor = foldDescriptor;
    this.children = new Map();
    this.activeByChild = new Map();
  }

  reserve(parentSessionId, launch, frozenBinding) {
    if (!launch?.reservation_id || !Number.isSafeInteger(launch?.launch_attempt) || launch.launch_attempt < 1) {
      throw new Error("durable ledger launch allocation is required");
    }
    const label = childReservationLabel(launch);
    const key = `${parentSessionId}\u0000${label}`;
    if (this.durable.tombstones.has(key)) throw new Error("reservation is permanently tombstoned");
    if (this.durable.reservations.has(key)) throw new Error("reservation already exists");
    const reservation = {
      parent_session_id: parentSessionId,
      label,
      launch: Object.freeze({ ...launch }),
      binding: Object.freeze({ ...frozenBinding }),
      claimed_child_id: null,
      status: "reserved",
    };
    this.durable.reservations.set(key, reservation);
    return reservation;
  }

  claim(execAgent) {
    const parent = execAgent?.session?.header?.parentSession;
    const label = this.foldDescriptor(execAgent?.session?.events || []);
    return this.claimPublished(parent, label, execAgent.id);
  }

  claimPublished(parent, label, childId) {
    const key = `${parent}\u0000${label}`;
    const reservation = this.durable.reservations.get(key);
    if (!reservation) throw new Error("missing reservation");
    if (reservation.claimed_child_id || reservation.status === "claimed") {
      if (
        reservation.claimed_child_id === childId
        && reservation.status === "claimed"
      ) return reservation.binding;
      throw new Error("reservation already claimed");
    }
    if (reservation.status !== "reserved") throw new Error("missing reservation");
    reservation.claimed_child_id = childId;
    reservation.status = "claimed";
    this.children.set(childId, reservation);
    return reservation.binding;
  }

  bindingFor(execAgent, { role, toolName } = {}) {
    let reservation = this.children.get(execAgent?.id);
    if (!reservation) {
      this.claim(execAgent);
      reservation = this.children.get(execAgent?.id);
    }
    if (!reservation || reservation.binding.role !== role) {
      throw new Error("child role binding mismatch");
    }
    const allowed = reservation.binding.allowed_tools;
    if (Array.isArray(allowed) && !allowed.includes(toolName)) {
      throw new Error("child tool is outside its frozen role surface");
    }
    let leaseId = this.activeByChild.get(execAgent.id);
    if (!leaseId) {
      leaseId = this.openActivation(execAgent.id, {
        revision: reservation.binding.run_state_revision,
        stage_attempt: reservation.binding.stage_attempt,
        idempotency_key: reservation.binding.idempotency_key,
      }).lease_id;
    }
    const { allowed_tools: _allowedTools, ...identity } = reservation.binding;
    return Object.freeze({
      ...identity,
      session_id: execAgent.id,
      child_reservation_id: reservation.launch.reservation_id,
      activation_lease_id: leaseId,
    });
  }

  releaseChild(childId) {
    const reservation = this.children.get(childId);
    if (!reservation) return;
    const leaseId = this.activeByChild.get(childId);
    if (leaseId) this.closeActivation(leaseId);
    this.children.delete(childId);
    this.terminal(reservation.parent_session_id, reservation.label);
  }

  openActivation(childId, activation) {
    if (!this.children.has(childId)) throw new Error("child identity is not claimed");
    if (this.activeByChild.has(childId)) throw new Error("child already has an active lease");
    const lease = Object.freeze({
      ...activation,
      child_id: childId,
      lease_id: `ecology-lease-${randomUUID()}`,
      status: "active",
    });
    this.durable.activations.set(lease.lease_id, lease);
    this.activeByChild.set(childId, lease.lease_id);
    return lease;
  }

  closeActivation(leaseId) {
    const lease = this.durable.activations.get(leaseId);
    if (!lease || lease.status !== "active") throw new Error("activation lease is not active");
    this.durable.activations.set(leaseId, Object.freeze({ ...lease, status: "closed" }));
    this.activeByChild.delete(lease.child_id);
  }

  revoke(parentSessionId, label) {
    const key = `${parentSessionId}\u0000${label}`;
    const reservation = this.durable.reservations.get(key);
    if (reservation) reservation.status = "revoked";
    this.durable.reservations.delete(key);
    this.durable.tombstones.add(key);
  }

  terminal(parentSessionId, label) {
    this.revoke(parentSessionId, label);
  }

  revokeRun(runId) {
    for (const reservation of [...this.durable.reservations.values()]) {
      if (reservation.launch.run_id !== runId) continue;
      if (reservation.claimed_child_id) {
        const leaseId = this.activeByChild.get(reservation.claimed_child_id);
        if (leaseId) this.closeActivation(leaseId);
        this.children.delete(reservation.claimed_child_id);
      }
      this.revoke(reservation.parent_session_id, reservation.label);
    }
  }
}

export { descriptorLabel };
