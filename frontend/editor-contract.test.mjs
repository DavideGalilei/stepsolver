import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { ComputeEngine } from "@cortex-js/compute-engine";

const cases = JSON.parse(
  readFileSync(new URL("./fixtures/editor-cases.json", import.meta.url), "utf8")
);
const computeEngine = new ComputeEngine();

for (const editorCase of cases) {
  test(`MathJSON contract: ${editorCase.name}`, () => {
    const parsed = computeEngine.parse(editorCase.latex, { form: "raw" });
    assert.deepEqual(parsed.json, editorCase.math_json);
  });
}
