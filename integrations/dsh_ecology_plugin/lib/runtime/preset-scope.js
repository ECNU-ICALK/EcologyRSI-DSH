// Catalog reads own a DSH 0.2 scope lease; live agents own their own mount.
export async function withPresetScope(registry, presetId, read) {
  const lease = await registry.acquireScope(presetId);
  const dispose = lease?.[Symbol.asyncDispose];
  if (typeof dispose !== "function") throw new Error("DSH preset scope has no disposer");
  try {
    if (!lease.key || typeof lease.key !== "object") throw new Error("DSH preset scope has no key");
    return await read(lease.key);
  } finally {
    await dispose.call(lease);
  }
}

export async function assertPresetAvailable(registry, presetId) {
  const preset = await registry.resolve(presetId);
  if (!preset || preset.id !== presetId || preset.broken) {
    throw new Error(`DSH preset is not mountable: ${presetId}`);
  }
}
