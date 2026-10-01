// The single adapter between this plugin's event-log evidence readers and the
// live DSH Session. @deepseek-ai/dsh-session (0.1.5-rc.2) exposes the ordered
// log through snapshotEvents(), not as an `events` array; a plain array is still
// accepted so an array-shaped Session stays readable. A Session that is not live
// in this store yields undefined, which every caller already treats as absent
// evidence rather than as a passing check.
export function sessionEvents(session) {
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

export function sessionEventLog(ctx, sessionId) {
  return sessionEvents(ctx?.sessions?.get?.(sessionId));
}

// Harness 0.2 stores tool identity on the tool message, whose content is the
// result body. Keep every evidence reader on this one normalization boundary.
export function toolResultIdentity(event) {
  if (event?.type !== "tool/result") return null;
  const data = event.data || {};
  const message = data.message;
  if (message?.role === "tool" && ("toolCallId" in message || message.source?.kind === "tool")) {
    const callId = message.toolCallId;
    if (typeof callId !== "string" || !callId || message.source?.kind !== "tool"
      || message.source.callId !== callId) return null;
    return { callId, isError: message.isError === true, content: message.content };
  }
  if (typeof data.callId === "string") {
    return { callId: data.callId, isError: data.isError === true, content: data.content };
  }
  const block = Array.isArray(message?.content)
    ? message.content.find(item => item?.type === "tool-result") : null;
  if (typeof block?.toolCallId !== "string") return null;
  return { callId: block.toolCallId, isError: block.isError === true, content: block.content };
}
