import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { ComputeEngine } from "@cortex-js/compute-engine";

const cases = JSON.parse(
  readFileSync(new URL("./fixtures/editor-cases.json", import.meta.url), "utf8")
);
const computeEngine = new ComputeEngine();

test("MathJSON contract: indexed-root series", () => {
  const parsed = computeEngine.parse(
    String.raw`\sum_{n=1}^{+\infty}{\sqrt[n]{2^n}+2^n}`,
    { form: "raw" },
  );
  assert.deepEqual(parsed.json, [
    "Sum",
    ["Add", ["Root", ["Power", 2, "n"], "n"], ["Power", 2, "n"]],
    ["Tuple", "n", 1, "PositiveInfinity"],
  ]);
});

for (const editorCase of cases) {
  test(`MathJSON contract: ${editorCase.name}`, () => {
    const parsed = computeEngine.parse(editorCase.latex, { form: "raw" });
    assert.deepEqual(parsed.json, editorCase.math_json);
  });
}
