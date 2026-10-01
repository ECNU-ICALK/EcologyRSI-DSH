import { readFileSync } from "node:fs";

function freeze(value) {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}

// This shipped manifest is shared by the Host, installer, tools and canaries.
export const PRESET_MANIFEST = freeze(JSON.parse(readFileSync(
  new URL("../../presets/preset-manifest.json", import.meta.url), "utf8",
)));
if (PRESET_MANIFEST.schema_version !== "ecologyrsi-dsh.preset-manifest/1"
  || !PRESET_MANIFEST.presets?.length || !PRESET_MANIFEST.stages) {
  throw new Error("preset-manifest.json is invalid");
}
export const ROLE_CONTRACTS = freeze(Object.fromEntries(
  PRESET_MANIFEST.presets.map(item => [item.role, item]),
));
export const STAGE_CONTRACTS = PRESET_MANIFEST.stages;
export const CANARY_PRESETS = freeze(Object.fromEntries(
  PRESET_MANIFEST.canary_stages.map(stage =>
    [stage, ROLE_CONTRACTS[STAGE_CONTRACTS[stage].role].preset_id]),
));
