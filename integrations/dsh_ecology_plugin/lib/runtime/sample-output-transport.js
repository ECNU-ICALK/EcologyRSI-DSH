import { structuredPhaseError } from "./structured-stage-errors.js";

// The frozen request already determines these protocol fields. Scientific
// decisions, sample IDs and evidence still come from the native Agent.
export function sampleOutputTransport(stage, schema) {
  if (stage !== "sample.plan") {
    return { schema, hostFields: {}, bind: value => value };
  }
  const hostFields = Object.freeze(Object.fromEntries(
    ["schema_version", "wave_digest"].map(name => {
      const value = schema.properties?.[name]?.const;
      if (typeof value !== "string" || !schema.required?.includes(name)
        || (name === "wave_digest" && !/^[0-9a-f]{64}$/.test(value))) {
        throw new Error("sample planner requires frozen protocol metadata");
      }
      return [name, value];
    }),
  ));
  const wire = structuredClone(schema);
  wire.required = wire.required.filter(name => !Object.hasOwn(hostFields, name));
  for (const name of Object.keys(hostFields)) {
    wire.properties[name].description = "Optional protocol metadata; the Host binds this exact value from the frozen request. Do not change it.";
  }
  return {
    schema: wire,
    hostFields,
    bind(value) {
      if (!value || typeof value !== "object" || Array.isArray(value)
        || Object.entries(hostFields).some(([name, expected]) =>
          Object.hasOwn(value, name) && value[name] !== expected)) {
        throw structuredPhaseError("output_schema");
      }
      return { ...value, ...hostFields };
    },
  };
}
