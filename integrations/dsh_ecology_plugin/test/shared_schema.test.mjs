import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { expandSharedSchema } from "../lib/runtime/schema.js";
const canonical = value => Array.isArray(value) ? `[${value.map(canonical).join(",")}]`
  : value && typeof value === "object" ? `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}` : JSON.stringify(value);
const baseline = {
  "genome-mutation": "dd067fdc92ae67db4525f1d64b35d1b413dc70db57d428c7dbffb7c5592113a9",
  "local-edit": "df7b146fa307a0404aeb25253052f5f6d40919505af4e2ad618da27be6fb3b22",
  "research-synthesis": "70bf9ef2f205ea29329b7ec6313757214ba3e3a5781f9bc96e1b974e1373454b",
  "generation-reflection": "42f94f87b8f630168ea5885d61c670af9bc080ffc4a3780f763f6b9389834a17"
};
test("expanded stage schemas equal the complete frozen pre-refactor contracts", async () => {
  for (const [name, expected] of Object.entries(baseline)) {
    const raw = JSON.parse(await readFile(new URL(`../schemas/${name}.schema.json`, import.meta.url)));
    const expanded = expandSharedSchema(raw);
    assert.equal(createHash("sha256").update(canonical(expanded)).digest("hex"), expected, name);
    expanded.properties.schema_version.const = "changed-local-copy";
    assert.notEqual(expandSharedSchema(raw).properties.schema_version.const, "changed-local-copy");
  }
});
test("schema sharing rejects paths, remote refs, unknown definitions and sibling overrides", () => {
  for (const $ref of ["https://example.invalid/schema", "../shared.json#/mutation_operation", "shared.json#/unknown"]) {
    assert.throws(() => expandSharedSchema({$ref}), /unsupported/);
  }
  assert.throws(() => expandSharedSchema({$ref: "shared.json#/mutation_operation", additionalProperties: true}), /unsupported/);
});
