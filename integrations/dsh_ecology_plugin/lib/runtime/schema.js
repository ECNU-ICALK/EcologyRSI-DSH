import { readFileSync } from "node:fs";

const definitions = JSON.parse(readFileSync(new URL("../../schemas/shared.json", import.meta.url), "utf8"));

// Only two bundled definitions are shared. No remote references, file paths,
// recursive definitions or sibling overrides can enter a model schema.
export function expandSharedSchema(value) {
  if (Array.isArray(value)) return value.map(expandSharedSchema);
  if (!value || typeof value !== "object") return value;
  if (Object.hasOwn(value, "$ref")) {
    const name = value.$ref === "shared.json#/mutation_operation" ? "mutation_operation"
      : value.$ref === "shared.json#/research_direction" ? "research_direction" : null;
    if (!name || Object.keys(value).length !== 1) throw new Error("unsupported shared schema reference");
    return structuredClone(definitions[name]);
  }
  return Object.fromEntries(Object.entries(value).map(([key, child]) => [key, expandSharedSchema(child)]));
}
