// The single adapter between this plugin's event-log evidence readers and the
// live DSH Session. @deepseek-ai/dsh-session (0.1.5-rc.2) exposes the ordered
// log through snapshotEvents(), not as an `events` array; a plain array is still
// accepted so an array-shaped Session stays readable. A Session that is not live
// in this store yields undefined, which every caller already treats as absent
// evidence rather than as a passing check.
export function sessionEventLog(ctx, sessionId) {
  const session = ctx?.sessions?.get?.(sessionId);
  if (session === undefined || session === null) return undefined;
  if (Array.isArray(session.events)) return session.events;
  if (typeof session.snapshotEvents !== "function") return undefined;
  try {
    const events = session.snapshotEvents();
    return Array.isArray(events) ? events : undefined;
  } catch {
    // A snapshot raced against session teardown is absent evidence, not proof.
    return undefined;
  }
}
